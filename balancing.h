#include <cstdint>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream> // debug
#include <iomanip> // setw
#include <sys/time.h> // linux time

#include "storm.h"

class SpinLock {
public:
    inline
    void lock() {
        while (lck.test_and_set(std::memory_order_acquire))
        {}
    }

    inline 
    void unlock() {lck.clear(std::memory_order_release);}

private:
    std::atomic_flag lck = ATOMIC_FLAG_INIT;
};

/**<
 * Simple timer class for tracking time differences between two timepoints.
 * Internally use the `chrono::high_resolution_clock` struct such that we can
 * track very short time frames.
 */
class Timer {
public:
	explicit Timer() {}

	/**<
	 * Start the timer by setting the current timestamp as the reference
	 * time.
	 */
	void Start(void) { this->_start = std::chrono::high_resolution_clock::now(); }

	/**<
	 * Returns the number `chrono::duration<double>` object for time elapsed.
	 * If you are interested in the number of seconds elapsed then chain this
	 * function with the child function `count`: `timer.Elapsed().count()`.
	 * @return
	 */
	inline std::chrono::duration<double> Elapsed() const{
		return(std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - this->_start));
	}

	friend std::ostream& operator<<(std::ostream& out, const Timer& timer) {
		return out << timer.Elapsed().count();
	}

	std::string ElapsedString(void) { return this->SecondsToTimestring(this->Elapsed().count()); }

private:
	std::string SecondsToTimestring(const double seconds) {
		const int32_t hours     = ((int32_t)seconds / 60 / 60);
		const int32_t minutes   = ((int32_t)seconds / 60) % 60;
		const int32_t sec       = (int32_t)seconds % 60;
		const int32_t remainder = (seconds - (int32_t)seconds)*1000;

		if (hours > 0) {
			sprintf(&this->buffer[0], "%02uh%02um%02u,%03us",
					hours,
					minutes,
					sec,
					remainder);

			return(std::string(&this->buffer[0], 13));
		} else if (minutes > 0) {
			sprintf(&this->buffer[0], "%02um%02u,%03us",
					minutes,
					sec,
					remainder);

			return(std::string(&this->buffer[0], 10));
		} else {
			sprintf(&this->buffer[0], "%02u,%03us",
					sec,
					remainder);

			return(std::string(&this->buffer[0], 7));
		}
	}

private:
	char buffer[64];
	std::chrono::high_resolution_clock::time_point _start;
};

static inline
std::string datetime() {
	time_t t = time(0);
	struct timeval  tv;
	struct timezone tz;
	struct tm      	*now = localtime(&t);
	gettimeofday(&tv, &tz);

	char buffer[48];
	sprintf(buffer, "%04u-%02u-%02u %02u:%02u:%02u,%03u",
			now->tm_year + 1900,
			now->tm_mon + 1,
			now->tm_mday,
			now->tm_hour,
			now->tm_min,
			now->tm_sec,
			(uint32_t)tv.tv_usec / 1000);

	return std::string(&buffer[0], 23);
}

static inline
std::string timestamp(const std::string type) {
	std::stringstream ret;

	ret << "[" << datetime() << "]";
	ret << "[" << type << "] ";

	return(ret.str());
}

static inline
std::string timestamp(const std::string type, const std::string type2) {
	std::stringstream ret;

	ret << "[" << datetime() << "]";
	ret << "[" << type << "]";
	ret << "[" << type2 << "] ";

	return(ret.str());
}

static inline
std::string SecondsToTimestring(const double& value) {
	uint32_t internalVal = value;
	std::string retVal;
	const uint32_t hours = internalVal / 3600;
	if (hours > 0) retVal += std::to_string(hours) + "h";
	internalVal %= 3600;
	const uint32_t min = internalVal / 60;
	if (min > 0) retVal += std::to_string(min) + "m";
	internalVal %= 60;
	const uint32_t sec = internalVal;
	retVal += std::to_string(sec) + "s";

	return(retVal);
}

static inline
std::string NumberThousandsSeparator(std::string number) {
	int insertPosition = number.length() - 3;
	char EndPos = 0;
	if (number[0] == '-')
		EndPos = 1;

	// Todo: fix NNNN.MMMM
	while (insertPosition > EndPos) {
	    number.insert(insertPosition, ",");
	    insertPosition -= 3;
	}

	if (number[0] == '-') {
		std::string numberTemp = "&ndash;";
		numberTemp += number.substr(1,10000);
		return numberTemp;
	}

	return number;
}

template <class T>
std::string ToPrettyString(const T& data) {
	return NumberThousandsSeparator(std::to_string(data));
}

template <class T>
std::string ToPrettyString(const std::vector<T>& data) {
	std::string ret;
	for (uint32_t i = 0; i < data.size() - 1; ++i) {
		ret += NumberThousandsSeparator(std::to_string(data[i]));
		ret += ", ";
	}
	ret += std::to_string(data[data.size()-1]);
	return ret;
}


/**<
 * Progress ticker for calculating linkage-disequilbirium. Spawns and detaches
 * a thread to tick occasionally in the background. Slaves computing linkage-
 * disequilibrium send their progress to this ticker that collates and summarize
 * that data.
 */
struct twk_ld_progress {
	twk_ld_progress() :
		is_ticking(false), n_s(0), n_cmps(0), n_var(0),
		n_pair(0), n_out(0), b_out(0), thread(nullptr)
	{}
	~twk_ld_progress() = default;

	/**<
	 * Starts the progress ticker. Spawns a detached thread ticking every 30 seconds
	 * in the background until the flag `is_ticking` is set to FALSE or the program
	 * finishes.
	 * @return Returns a pointer to the detached thread.
	 */
	std::thread* Start() {
		delete thread;
		is_ticking = true;
		thread = new std::thread(&twk_ld_progress::StartTicking, this);
		thread->detach();
		return(thread);
	}

	/**<
	 * Internal function displaying the progress message every 30 seconds. This
	 * function is called exclusively by the detached thread.
	 */
	void StartTicking() {
		timer.Start();

		uint64_t variant_overflow = 99E9, genotype_overflow = 999E12;
		uint8_t  variant_width = 15, genotype_width = 20;

        n_var = 0; n_pair = 0;
        n_out = 0; b_out = 0;

		//char support_buffer[256];
		std::cerr << timestamp("PROGRESS")
				<< std::setw(12) << "Time elapsed"
				<< std::setw(variant_width) << "Variants"
				<< std::setw(genotype_width) << "2-locus cmps."
				<< std::setw(15) << "Output"
				<< std::setw(10) << "Progress"
				<< "\tEst. Time left" << std::endl;

		std::this_thread::sleep_for (std::chrono::seconds(30)); // first sleep
		while (is_ticking) {
			if (n_var.load() > variant_overflow)      { variant_width  += 3; variant_overflow  *= 1e3; }
			if (n_var.load()*n_s > genotype_overflow) { genotype_width += 3; genotype_overflow *= 1e3; }

			if (n_cmps) {
				std::cerr << timestamp("PROGRESS")
						<< std::setw(12) << timer.ElapsedString()
						<< std::setw(variant_width)  << ToPrettyString(n_var.load())
						<< std::setw(genotype_width) << ToPrettyString(n_var.load()*n_s)
						<< std::setw(15) << ToPrettyString(n_out.load())
						<< std::setw(10) << (double)n_var.load()/n_cmps*100 << "%\t"
						<< SecondsToTimestring((n_cmps - n_var.load()) / ((double)n_var.load()/timer.Elapsed().count())) << std::endl;
			} else {
				std::cerr << timestamp("PROGRESS")
						<< std::setw(12) << timer.ElapsedString()
						<< std::setw(variant_width)  << ToPrettyString(n_var.load())
						<< std::setw(genotype_width) << ToPrettyString(n_var.load()*n_s)
						<< std::setw(15) << ToPrettyString(n_out.load())
						<< std::setw(10) << 0 << '\t'
						<< 0 << std::endl;
			}
			std::this_thread::sleep_for (std::chrono::seconds(30));
		}
	}

	/**<
	 * Print out the final tally of time elapsed, number of variants computed,
	 * and average throughput. This method cannot be made const as the function
	 * ElapsedString in the Timer class internally updates a buffer for performance
	 * reasons. This has no consequence as this function is ever only called once.
	 */
	void PrintFinal() {
		std::cerr << timestamp("PROGRESS") << "Finished in " << this->timer.ElapsedString()
				<< ". Variants: " << ToPrettyString(n_var.load()) << ", genotypes: "
				<< ToPrettyString(n_var.load()*n_s) << ", output: "
				<< ToPrettyString(n_out.load()) << std::endl;
		std::cerr << timestamp("PROGRESS") << ToPrettyString((uint64_t)((double)n_var.load()/timer.Elapsed().count())) << " variants/s and "
				<< ToPrettyString((uint64_t)(((double)n_var.load()*n_s)/timer.Elapsed().count())) << " genotypes/s" << std::endl;
	}

public:
	bool is_ticking;
	uint32_t n_s; // number of samples
	uint64_t n_cmps; // number of comparisons we estimate to perform
	std::atomic<uint64_t> n_var, n_pair, n_out, b_out; // counters used by ld threads
	std::thread* thread; // detached thread
	Timer timer; // timer instance
};

/**<
 * Work balancer for twk_ld_engine threads. Uses a non-blocking spinlock to produce a
 * tuple (from,to) of integers representing the start ref block and dst block.
 * This approach allows perfect load-balancing at a small overall CPU cost. This is
 * directly equivalent to dynamic load-balancing (out-of-order execution) when n > 1.
 */
struct twk_ld_dynamic_balancer {
	twk_ld_dynamic_balancer() :
		fL(0), tL(0), fR(0), tR(0)
	{}
	~twk_ld_dynamic_balancer() {}

	/**<
	 * Retrieves (x,y)-coordinates from the selected load-balancing subproblem.
	 * Uses a spin-lock to make this function thread-safe.
	 * @param from Row position
	 * @param to   Column position
	 * @param type Diagonal (1) or square (0)
	 * @return     Returns TRUE if it is possible to retrieve a new (x,y)-pair or FALSE otherwise.
	 */
	bool GetBlockPair(uint32_t& from, uint32_t& to, uint8_t& type) {
		spinlock.lock();

        if (fL == 0) {
            if (tR == 0) {
                from = 0; to = 0; type = 2;
                spinlock.unlock();
			    return false;
            }
            from = fL++;
            to   = fR;
            type = 1;

            spinlock.unlock();
			return true;
        } else if (fL >= tR) { // if current position is at the last column
            if (++fR >= tR) {
                from = 0; to = 0; type = 2;
                spinlock.unlock(); 
                return false;
            }

            fL   = fR;
            from = fL;
            to   = fL;
            type = 1;
            ++fL;

			spinlock.unlock();
			return true;
		} else {
            from = fL++;
            to   = fR;
            type = 0;
        }
		
		spinlock.unlock();
		return true;
	}

public:
	uint32_t fL, tL, fR, tR;
	SpinLock spinlock;
};

struct twk_ld_slave {
	twk_ld_slave() : ticker(nullptr), thread(nullptr), progress(nullptr), optimal_b(0), count(0), comps(0) {}
	~twk_ld_slave() {
        delete thread;
    }

	/**<
	 * Primary subroutine for starting the slave thread to compute its designated
	 * region(s) of linkage-disequilibrium.
	 * @return Returns a pointer to the spawned slave thread.
	 */
	std::thread* Start() {
        delete thread; thread = nullptr;

        thread = new std::thread(&twk_ld_slave::Calculate, this);

        return(thread);
    }

    bool Calculate() {
        uint32_t from = 0, to = 0;
        uint8_t type = 0;

        count = 0;
        comps = 0;
        while (true) {
		    if (!ticker->GetBlockPair(from, to, type)) break;
            ++comps;
            // std::cerr << "tick: " << from << "," << to << " type=" << (int)type << std::endl;

            if (from == to) {
                count += STORM_contig_pairw_intersect_cardinality_blocked(&twk_cont_vec[to], optimal_b);
                // std::cerr << "adding: " << (twk_cont_vec[to].n_data * (twk_cont_vec[to].n_data - 1) / 2) << " for " << twk_cont_vec[to].n_data << std::endl;
                progress->n_var += twk_cont_vec[to].n_data * (twk_cont_vec[to].n_data - 1) / 2;
            }  else {
                count += STORM_contig_pairw_sq_intersect_cardinality_blocked(&twk_cont_vec[to], &twk_cont_vec[from], optimal_b >> 1);
                // std::cerr << "adding: " << (twk_cont_vec[to].n_data * twk_cont_vec[from].n_data) << " for " << twk_cont_vec[from].n_data << "*" << twk_cont_vec[from].n_data << std::endl;
                
                progress->n_var += twk_cont_vec[to].n_data * twk_cont_vec[from].n_data;
            }
        }

        std::cerr << "total=" << count << " comps=" << comps << std::endl;
    }

public:
    twk_ld_dynamic_balancer* ticker;
    std::thread* thread;
    twk_ld_progress* progress;
    STORM_contiguous_t* twk_cont_vec;
    uint32_t optimal_b;
    uint64_t count, comps;
};
