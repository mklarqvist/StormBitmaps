#include <cstdint>
#include <atomic>
#include <thread>
#include <iostream> // debug

#include "storm.h"

class SpinLock {
public:
    inline
    void lock(){
        while(lck.test_and_set(std::memory_order_acquire))
        {}
    }

    inline 
    void unlock(){lck.clear(std::memory_order_release);}

private:
    std::atomic_flag lck = ATOMIC_FLAG_INIT;
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
	~twk_ld_dynamic_balancer(){}

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
        } else if (fL >= tR){ // if current position is at the last column
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
	twk_ld_slave() : ticker(nullptr), thread(nullptr), optimal_b(0), count(0) {}
	~twk_ld_slave() {
        delete thread;
    }

	/**<
	 * Primary subroutine for starting the slave thread to compute its designated
	 * region(s) of linkage-disequilibrium.
	 * @return Returns a pointer to the spawned slave thread.
	 */
	std::thread* Start(){
        delete thread; thread = nullptr;

        thread = new std::thread(&twk_ld_slave::Calculate, this);

        return(thread);
    }

    bool Calculate() {
        uint32_t from = 0, to = 0;
        uint8_t type = 0;

        count = 0;
        while(true){
		    if(!ticker->GetBlockPair(from, to, type)) break;
            std::cerr << "tick: " << from << "," << to << " type=" << (int)type << std::endl;

            if (from == to) {
                count += STORM_contig_pairw_intersect_cardinality_blocked(&twk_cont_vec[to], optimal_b);
            }  else {
                count += STORM_contig_pairw_sq_intersect_cardinality_blocked(&twk_cont_vec[to], &twk_cont_vec[from], optimal_b >> 1);
            }
        }

        std::cerr << "total=" << count << std::endl;
    }

public:
    twk_ld_dynamic_balancer* ticker;
    std::thread* thread;
    STORM_contiguous_t* twk_cont_vec;
    uint32_t optimal_b;
    uint64_t count;
};
