#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <memory>
#include <bitset>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib> // EXIT_SUCCESS, EXIT_FAILURE

#include "balancing.h"

#define USE_ROARING
#define ALLOW_LINUX
#define USE_HTSLIB

#ifdef USE_ROARING
#include <roaring/roaring.h>
#endif

#ifdef USE_HTSLIB
#include "vcf_reader.h"
#endif

#include "storm.h"
// #include "classes.h"
// #include "experimental.h"

#if defined(_MSC_VER)
inline
uint64_t get_cpu_cycles() {
    // _mm_lfence();  // optionally wait for earlier insns to retire before reading the clock
    uint64_t tsc = __rdtsc();
    // _mm_lfence();  // optionally block later instructions until rdtsc retires
    return tsc;
}
#else
uint64_t get_cpu_cycles() {
    uint64_t result;
    __asm__ volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax":"=a"
                     (result)::"%rdx");
    return result;
};
#endif

// Convenience wrapper
struct bench_t {
    bench_t() :
        total(0), 
        instructions_cycle(0), 
        cycles_word(0), 
        instructions_word(0),
        cycles(0), 
        instructions(0), 
        MinBranchMiss(0),
        MinCacheRef(0), 
        MinCacheMiss(0),
        throughput(0),
        time_ms(0)
    {
    }

    bench_t(std::vector<unsigned long long>& results, const uint64_t n_total_integer_cmps) :
        total(0), 
        instructions_cycle(double(results[1]) / results[0]), 
        cycles_word(double(results[0]) / (n_total_integer_cmps)), 
        instructions_word(double(results[1]) / (n_total_integer_cmps)),
        cycles(results[0]), 
        instructions(results[1]), 
        MinBranchMiss(results[2]),
        MinCacheRef(results[3]), 
        MinCacheMiss(results[4]),
        throughput(0),
        time_ms(0)
    {
    }

    void PrintPretty() const {
        printf("%llu\t%.2f\t%.3f\t%.3f\t%llu\t%llu\t%llu\t%llu\t%llu\t%.2f\t%u\n",
            (long long unsigned int)total,
            instructions_cycle,
            cycles_word, 
            instructions_word, 
            cycles,
            instructions,
            MinBranchMiss,
            MinCacheRef,
            MinCacheMiss,
            throughput,
            time_ms);
    }

    long long unsigned int total;
    double instructions_cycle;
    double cycles_word;
    double instructions_word;
    long long unsigned int cycles;
    long long unsigned int instructions;
    long long unsigned int MinBranchMiss;
    long long unsigned int MinCacheRef;
    long long unsigned int MinCacheMiss;
    double throughput;
    uint32_t time_ms;
};

#if defined __linux__ && defined ALLOW_LINUX
#include <asm/unistd.h>       // for __NR_perf_event_open
#include <linux/perf_event.h> // for perf event constants
#include <sys/ioctl.h>        // for ioctl
#include <unistd.h>           // for syscall
#include <iostream>
#include <cerrno>  // for errno
#include <cstring> // for memset
#include <stdexcept>

#define PERF_PRE std::vector<int> evts;        \
evts.push_back(PERF_COUNT_HW_CPU_CYCLES);      \
evts.push_back(PERF_COUNT_HW_INSTRUCTIONS);    \
evts.push_back(PERF_COUNT_HW_BRANCH_MISSES);   \
evts.push_back(PERF_COUNT_HW_CACHE_REFERENCES);\
evts.push_back(PERF_COUNT_HW_CACHE_MISSES);    \
evts.push_back(PERF_COUNT_HW_REF_CPU_CYCLES);  \
LinuxEvents<PERF_TYPE_HARDWARE> unified(evts); \
std::vector<unsigned long long> results;       \
results.resize(evts.size());                   \
std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); \
unified.start();

#define PERF_POST unified.end(results); \
std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now(); \
auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); \
uint64_t n_comps = (n_variants*n_variants - n_variants) / 2; \
bench_t b(results, n_comps * 2*n_ints_sample); \
b.total = total; b.time_ms = time_span.count(); \
b.throughput = (( ( n_comps * 2*n_ints_sample ) * sizeof(uint64_t)) / (1024*1024.0)) / (b.time_ms / 1000.0);
#else // not linux
#define PERF_PRE uint64_t cycles_before = get_cpu_cycles(); \
std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#define PERF_POST uint64_t cycles_after = get_cpu_cycles(); \
std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now(); \
auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); \
uint64_t n_comps = (n_variants*n_variants - n_variants) / 2; \
bench_t b; b.cycles = cycles_after - cycles_before; b.cycles_word = b.cycles / (2*n_comps); \
b.total = total; b.time_ms = time_span.count(); \
b.throughput = (( ( n_comps * 2*n_ints_sample ) * sizeof(uint64_t)) / (1024*1024.0)) / (b.time_ms / 1000.0);
#endif


#if defined __linux__ && defined ALLOW_LINUX

template <int TYPE = PERF_TYPE_HARDWARE> 
class LinuxEvents {
private:
    int fd;
    bool working;
    perf_event_attr attribs;
    int num_events;
    std::vector<uint64_t> temp_result_vec;
    std::vector<uint64_t> ids;

public:
    explicit LinuxEvents(std::vector<int> config_vec) : fd(0), working(true) {
        memset(&attribs, 0, sizeof(attribs));
        attribs.type = TYPE;
        attribs.size = sizeof(attribs);
        attribs.disabled = 1;
        attribs.exclude_kernel = 1;
        attribs.exclude_hv = 1;

        attribs.sample_period = 0;
        attribs.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
        const int pid = 0;  // the current process
        const int cpu = -1; // all CPUs
        const unsigned long flags = 0;

        int group = -1; // no group
        num_events = config_vec.size();
        uint32_t i = 0;
        for (auto config : config_vec) {
            attribs.config = config;
            fd = syscall(__NR_perf_event_open, &attribs, pid, cpu, group, flags);
            if (fd == -1) {
                report_error("perf_event_open");
            }
                ioctl(fd, PERF_EVENT_IOC_ID, &ids[i++]);
                if (group == -1) {
                group = fd;
            }
        }

        temp_result_vec.resize(num_events * 2 + 1);
    }

    ~LinuxEvents() { close(fd); }

    inline void start() {
        if (ioctl(fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == -1) {
            report_error("ioctl(PERF_EVENT_IOC_RESET)");
        }

        if (ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1) {
            report_error("ioctl(PERF_EVENT_IOC_ENABLE)");
        }
    }

    inline void end(std::vector<unsigned long long> &results) {
        if (ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1) {
            report_error("ioctl(PERF_EVENT_IOC_DISABLE)");
        }

        if (read(fd, &temp_result_vec[0], temp_result_vec.size() * 8) == -1) {
            report_error("read");
        }
        // our actual results are in slots 1,3,5, ... of this structure
        // we really should be checking our ids obtained earlier to be safe
        for (uint32_t i = 1; i < temp_result_vec.size(); i += 2) {
            results[i / 2] = temp_result_vec[i];
        }
    }

private:
    void report_error(const std::string &context) {
    if (working)
        std::cerr << (context + ": " + std::string(strerror(errno))) << std::endl;
        working = false;
    }
};
#endif // end is linux

/**<
 * Upper-triangular component of variant square matrix. This templated function
 * requires the target function pointer as a template. Example usage:
 *
 * fwrapper<&intersect_bitmaps_sse4>(n_variants, vals, n_ints_sample);
 *
 * @param n_variants Number of input variants.
 * @param vals       Raw data.
 * @param n_ints     Number of ints/sample.
 * @return           Returns a populated bench_t struct.
 */
template <uint64_t (f)(const uint64_t*  b1, const uint64_t*  b2, const size_t n_ints_sample)>
bench_t fwrapper(const uint32_t n_variants, const uint64_t* vals, const size_t n_ints_sample) {    
    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;
    
    PERF_PRE
    for (int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints_sample;
        for (int j = i + 1; j < n_variants; ++j, inner_offset += n_ints_sample) {
            total += (*f)(&vals[offset], &vals[inner_offset], n_ints_sample);
        }
        offset += n_ints_sample;
    }
    PERF_POST
    
    return(b);
}

template <uint64_t (f)(const uint64_t*  b1, const uint64_t*  b2, const size_t n_ints_sample)>
bench_t fwrapper_blocked(const uint32_t n_variants, const uint64_t* vals, const size_t n_ints_sample, uint32_t bsize = 200) {    
    uint64_t total = 0;

    bsize = (bsize == 0 ? 10 : bsize);
    const uint32_t n_blocks1 = n_variants / bsize;
    const uint32_t n_blocks2 = n_variants / bsize;
    // uint64_t d = 0;
    uint32_t i  = 0;
    uint32_t tt = 0;

    PERF_PRE
    for (/**/; i + bsize <= n_variants; i += bsize) {
        // diagonal component
        uint32_t left = i*n_ints_sample;
        uint32_t right = 0;
        for (uint32_t j = 0; j < bsize; ++j, left += n_ints_sample) {
            right = left + n_ints_sample;
            for (uint32_t jj = j + 1; jj < bsize; ++jj, right += n_ints_sample) {
                total += (*f)(&vals[left], &vals[right], n_ints_sample);
                // ++d;
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + bsize;
        for (/**/; j + bsize <= n_variants; j += bsize) {
            left = curi*n_ints_sample;
            for (uint32_t ii = 0; ii < bsize; ++ii, left += n_ints_sample) {
                right = j*n_ints_sample;
                for (uint32_t jj = 0; jj < bsize; ++jj, right += n_ints_sample) {
                    total += (*f)(&vals[left], &vals[right], n_ints_sample);
                    // ++d;
                }
            }
        }

        // residual
        right = j*n_ints_sample;
        for (/**/; j < n_variants; ++j, right += n_ints_sample) {
            left = curi*n_ints_sample;
            for (uint32_t jj = 0; jj < bsize; ++jj, left += n_ints_sample) {
                total += (*f)(&vals[left], &vals[right], n_ints_sample);
                // ++d;
            }
        }
    }
    // residual tail
    uint32_t left = i*n_ints_sample;
    for (/**/; i < n_variants; ++i, left += n_ints_sample) {
        uint32_t right = left + n_ints_sample;
        for (uint32_t j = i + 1; j < n_variants; ++j, right += n_ints_sample) {
            total += (*f)(&vals[left], &vals[right], n_ints_sample);
            // ++d;
        }
    }
    PERF_POST

    return(b);
}

template <uint64_t (f)(const uint64_t*  b1, const uint64_t*  b2, const uint32_t* l1, const uint32_t* l2, const size_t len1, const size_t len2)>
bench_t flwrapper(const uint32_t n_variants, const uint64_t* vals, const size_t n_ints_sample, const std::vector< std::vector<uint32_t> >& pos) {
    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;

    PERF_PRE
    for (int i = 0; i < n_variants; ++i) {
    inner_offset = offset + n_ints_sample;
       for (int j = i + 1; j < n_variants; ++j, inner_offset += n_ints_sample) {
           total += (*f)(&vals[offset], &vals[inner_offset], &pos[i][0], &pos[j][0], (uint32_t)pos[i].size(), (uint32_t)pos[j].size());
       }
       offset += n_ints_sample;
    }
    PERF_POST

    return(b);
}

template <uint64_t (f)(const uint64_t*  b1, const uint64_t*  b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2)>
bench_t flwrapper_blocked(const uint32_t n_variants, const uint64_t* vals, const size_t n_ints_sample, const std::vector< std::vector<uint32_t> >& pos, uint32_t bsize = 200) { 
    uint64_t total = 0;

    bsize = (bsize == 0 ? 10 : bsize);
    const uint32_t n_blocks1 = n_variants / bsize;
    const uint32_t n_blocks2 = n_variants / bsize;

    uint32_t i  = 0;
    uint32_t tt = 0;

    PERF_PRE
    for (/**/; i + bsize <= n_variants; i += bsize) {
        // diagonal component
        uint32_t left = i*n_ints_sample;
        uint32_t right = 0;
        for (uint32_t j = 0; j < bsize; ++j, left += n_ints_sample) {
            right = left + n_ints_sample;
            for (uint32_t jj = j + 1; jj < bsize; ++jj, right += n_ints_sample) {
                total += (*f)(&vals[left], &vals[right], pos[i+j], pos[i+jj]);
                //total += (*f)(&vals[left], &vals[right], n_ints_sample);
                // ++d;
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + bsize;
        for (/**/; j + bsize <= n_variants; j += bsize) {
            left = curi*n_ints_sample;
            for (uint32_t ii = 0; ii < bsize; ++ii, left += n_ints_sample) {
                right = j*n_ints_sample;
                for (uint32_t jj = 0; jj < bsize; ++jj, right += n_ints_sample) {
                    // total += (*f)(&vals[left], &vals[right], n_ints_sample);
                    total += (*f)(&vals[left], &vals[right], pos[curi + ii], pos[j + jj]);
                    // ++d;
                }
            }
        }

        // residual
        right = j*n_ints_sample;
        for (/**/; j < n_variants; ++j, right += n_ints_sample) {
            left = curi*n_ints_sample;
            for (uint32_t jj = 0; jj < bsize; ++jj, left += n_ints_sample) {
                // total += (*f)(&vals[left], &vals[right], n_ints_sample);
                total += (*f)(&vals[left], &vals[right], pos[curi + jj], pos[j]);
                // ++d;
            }
        }
    }
    // residual tail
    uint32_t left = i*n_ints_sample;
    for (/**/; i < n_variants; ++i, left += n_ints_sample) {
        uint32_t right = left + n_ints_sample;
        for (uint32_t j = i + 1; j < n_variants; ++j, right += n_ints_sample) {
            // total += (*f)(&vals[left], &vals[right], n_ints_sample);
            total += (*f)(&vals[left], &vals[right], pos[i], pos[j]);
            // ++d;
        }
    }
    PERF_POST

    return(b);
}

template <class int_t, uint64_t (f)(const std::vector<int_t>& rle1, const std::vector<int_t>& rle2)>
bench_t frlewrapper(const std::vector< std::vector<int_t> >& rle, const size_t n_ints_sample) {
    uint64_t total = 0;
    uint32_t n_variants =  n_ints_sample;
    
    PERF_PRE
    for (int i = 0; i < n_variants; ++i) {
        for (int j = i + 1; j < n_variants; ++j) {
            total += (*f)(rle[i], rle[j]);
        }
    }
    PERF_POST

    return(b);
}

// template <uint64_t (f)(const uint16_t* v1, const uint16_t* v2, const uint32_t len1, const uint32_t len2)>
// bench_t frawwrapper(const uint32_t n_variants, const uint32_t n_ints_sample, const std::vector< std::vector<uint16_t> >& pos) {
//     uint64_t total = 0;
    
//     PERF_PRE
//     for (int k = 0; k < n_variants; ++k) {
//         for (int p = k + 1; p < n_variants; ++p) {
//             total += (*f)(&pos[k][0], &pos[p][0], pos[k].size(), pos[p].size());
//         }
//     }
//     PERF_POST

//     return(b);
// }

#ifdef USE_ROARING
bench_t froarwrapper(const uint32_t n_variants, const uint32_t n_ints_sample, roaring_bitmap_t** bitmaps) {
    uint64_t total = 0;

    PERF_PRE
    for (int k = 0; k < n_variants; ++k) {
        for (int p = k + 1; p < n_variants; ++p) {
            total += roaring_bitmap_and_cardinality(bitmaps[k], bitmaps[p]);
        }
    }
    PERF_POST

    return(b);
}

bench_t froarwrapper_blocked(const uint32_t n_variants, const uint32_t n_ints_sample, roaring_bitmap_t** bitmaps, const uint32_t bsize) {
    // std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    PERF_PRE
    uint64_t total = 0;
    uint64_t blocked_con_tot = 0;
    uint32_t i  = 0;
    uint32_t tt = 0;

    for (/**/; i + bsize <= n_variants; i += bsize) {
        // diagonal component
        for (uint32_t j = 0; j < bsize; ++j) {
            for (uint32_t jj = j + 1; jj < bsize; ++jj) {
                blocked_con_tot += roaring_bitmap_and_cardinality(bitmaps[i+j], bitmaps[i+jj]);
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + bsize;
        for (/**/; j + bsize <= n_variants; j += bsize) {
            for (uint32_t ii = 0; ii < bsize; ++ii) {
                for (uint32_t jj = 0; jj < bsize; ++jj) {
                    blocked_con_tot += roaring_bitmap_and_cardinality(bitmaps[curi+ii], bitmaps[j+jj]);
                }
            }
        }

        // residual
        for (/**/; j < n_variants; ++j) {
            for (uint32_t jj = 0; jj < bsize; ++jj) {
                blocked_con_tot += roaring_bitmap_and_cardinality(bitmaps[curi+jj], bitmaps[j]);
            }
        }
    }
    // residual tail
    for (/**/; i < n_variants; ++i) {
        for (uint32_t j = i + 1; j < n_variants; ++j) {
            blocked_con_tot += roaring_bitmap_and_cardinality(bitmaps[i], bitmaps[j]);
        }
    }
    total = blocked_con_tot;
    PERF_POST

    return(b);
}
#endif

#define PRINT(name,bench) std::cout << n_samples << "\t" << n_alts[a] << "\t" << name << "\t" << bench.time_ms << "\t" << bench.cycles << "\t" << bench.total << "\t" << \
        bench.throughput << "\t" << \
        (bench.time_ms == 0 ? 0 : (int_comparisons*1000.0 / bench.time_ms / 1000000.0)) << "\t" << \
        (n_intersects*1000.0 / (bench.time_ms) / 1000000.0) << "\t" << \
        (bench.time_ms == 0 ? 0 : n_total_integer_cmps*sizeof(uint64_t) / (bench.time_ms/1000.0) / (1024.0*1024.0)) << "\t" << \
        (bench.cycles == 0 ? 0 : bench.cycles / (double)n_total_integer_cmps) << "\t" << \
        (bench.cycles == 0 ? 0 : bench.cycles / (double)n_intersects) << std::endl

void benchmark_large(uint32_t n_samples, uint32_t n_variants, std::vector<uint32_t>* loads) {
    std::cout << "Samples\tAlts\tMethod\tTime(ms)\tCPUCycles\tCount\tThroughput(MB/s)\tInts/s(1e6)\tIntersect/s(1e6)\tActualThroughput(MB/s)\tCycles/int\tCycles/intersect" << std::endl;
    // std::cerr << "Generating: " << n_samples << " samples for " << n_variants << " variants" << std::endl;

    std::vector<uint32_t> n_alts;
    if (loads == nullptr) {
        n_alts = {n_samples/2, n_samples/4, n_samples/10, n_samples/25, n_samples/50, n_samples/100, n_samples/250, n_samples/1000, n_samples/5000, 5, 1};
    } else {
        n_alts = *loads;
    }

    // no use
    uint64_t n_intersects = 0;
    uint64_t n_total_integer_cmps = 0;
    uint64_t int_comparisons = 0;

    STORM_t* twk2 = STORM_new();

    for (int a = 0; a < n_alts.size(); ++a) {
        // break if no more data
        if (n_alts[a] == 0) {
            // Make sure we always compute n_alts = 1
            if (a != 0) {
                if (n_alts[a-1] != 1) 
                    n_alts[a] = 1;
            } else {
                // std::cerr << "there are no alts..." << std::endl;
                break;
            }
        }

        // Break if data has converged
        if (a != 0) {
            if (n_alts[a] == n_alts[a-1]) 
                break;
        }


        STORM_clear(twk2);

#ifdef USE_ROARING
        roaring_bitmap_t** roaring = new roaring_bitmap_t*[n_variants];
        for (int i = 0; i < n_variants; ++i) roaring[i] = roaring_bitmap_create();
#endif

        // PRNG
        std::uniform_int_distribution<uint32_t> distr(0, n_samples-1); // right inclusive

        // Positional information
        std::vector<uint32_t> pos;
        
        std::random_device rd;  // obtain a random number from hardware
        std::mt19937 eng(rd()); // seed the generator

        // Draw
        // std::cerr << "Constructing..."; std::cerr.flush();
        
        // For each variant site.
        for (uint32_t j = 0; j < n_variants; ++j) {
            // Generate n_alts[a] random positions.
            for (uint32_t i = 0; i < n_alts[a]; ++i) {
                uint64_t val = distr(eng);
                pos.push_back(val);
            }

            // Sort to simplify
            std::sort(pos.begin(), pos.end());
            pos.erase( unique( pos.begin(), pos.end() ), pos.end() );

#ifdef USE_ROARING
            for (int p = 0; p < pos.size(); ++p) {
                roaring_bitmap_add(roaring[j], pos[p]);
            }
#endif
            STORM_add(twk2, &pos[0], pos.size());
            pos.clear();
        }
        // std::cerr << "Done!" << std::endl;

        uint32_t n_ints_sample = std::ceil(n_samples / 64.0);
        const uint64_t n_intersects = ((n_variants * n_variants) - n_variants) / 2;
        const uint64_t n_total_integer_cmps = n_intersects * n_ints_sample;
        // std::cerr << "Total integer comparisons=" << n_total_integer_cmps << std::endl;

        uint64_t storm_size = STORM_serialized_size(twk2);
        // std::cerr << "[MEMORY][STORM][" << n_alts[a] << "] Memory for Storm=" << storm_size << "b" << std::endl;

        // Debug
        std::chrono::high_resolution_clock::time_point t1_blocked = std::chrono::high_resolution_clock::now();
        // uint64_t d = 0, diag = 0;
        // {
        //     PERF_PRE
        //     uint64_t total = STORM_pairw_intersect_cardinality(twk2);
        //     PERF_POST
        //     std::cout << "storm\t" << n_alts[a] << "\t" << storm_size << "\t" ;
        //     b.PrintPretty();
        //     // LINUX_PRINT("storm")
        //     // PRINT("storm",b);
        // }

        {
            PERF_PRE
            uint64_t total = STORM_pairw_intersect_cardinality_blocked(twk2,0);
            PERF_POST
            // LINUX_PRINT("storm-blocked")
            std::cout << "storm-blocked\t" << n_alts[a] << "\t" << storm_size << "\t" ;
            b.PrintPretty();
            // PRINT("storm-blocked",b);
        }


#ifdef USE_ROARING
            uint64_t roaring_bytes_used = 0;
            for (int k = 0; k < n_variants; ++k) {
                roaring_bytes_used += roaring_bitmap_portable_size_in_bytes(roaring[k]);
            }
            // std::cerr << "[MEMORY][ROARING][" << n_alts[a] << "] Memory for Roaring=" << roaring_bytes_used << "b" << std::endl;

            uint32_t roaring_optimal_b = STORM_CACHE_BLOCK_SIZE / (roaring_bytes_used / n_variants);
            roaring_optimal_b = roaring_optimal_b < 5 ? 5 : roaring_optimal_b;

            bench_t m8_2_block = froarwrapper_blocked(n_variants, n_ints_sample, roaring, roaring_optimal_b);
            // PRINT("roaring-blocked-" + std::to_string(roaring_optimal_b),m8_2_block);
            std::string m8_2_block_name = "roaring-blocked-" + std::to_string(roaring_optimal_b);
            std::cout << m8_2_block_name << "\t" << n_alts[a] << "\t" << roaring_bytes_used << "\t" ;
            m8_2_block.PrintPretty();
#endif

#ifdef USE_ROARING
        for (int i = 0; i < n_variants; ++i) roaring_bitmap_free(roaring[i]);
        delete[] roaring;
#endif
    }
    // for (int i = 0; i < n_variants; ++i) STORM_bitmap_cont_free(twk[i]);
    // delete[] twk;
    STORM_free(twk2);

}

void intersect_test(uint32_t n_samples, uint32_t n_variants, std::vector<uint32_t>* loads) {
    // uint64_t* a = nullptr;
    // intersect(a,0,0);

#if defined(STORM_HAVE_CPUID)
    #if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = STORM_get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = STORM_get_cpuid();

    #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
    #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
    #endif
    }
    #endif
#endif

    // Setup
    // std::vector<uint32_t> samples = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216};
    // std::vector<uint32_t> samples = {131072, 196608, 589824};
    
    std::cout << "Samples\tAlts\tMethod\tTime(ms)\tCPUCycles\tCount\tThroughput(MB/s)\tInts/s(1e6)\tIntersect/s(1e6)\tActualThroughput(MB/s)\tCycles/int\tCycles/intersect" << std::endl;

    // for (int s = 0; s < samples.size(); ++s) {
        uint32_t n_ints_sample = std::ceil(n_samples / 64.0);

        // Limit memory usage to 10e6 but no more than 50k records.
        uint32_t desired_mem = 40 * 1024 * 1024;
        // b_total / (b/obj) = n_ints
        // uint32_t n_variants = std::max(std::min((uint32_t)150000, (uint32_t)std::ceil(desired_mem/(n_ints_sample*sizeof(uint64_t)))), (uint32_t)64);
        // uint32_t n_variants = 10000;

        // std::cerr << "Generating: " << n_samples << " samples for " << n_variants << " variants" << std::endl;
        const uint64_t memory_used = n_ints_sample*n_variants*sizeof(uint64_t);
        // std::cerr << "Allocating: " << memory_used/(1024 * 1024.0) << "Mb" << std::endl;

        uint64_t* vals = (uint64_t*)STORM_aligned_malloc(STORM_get_alignment(), n_ints_sample*n_variants*sizeof(uint64_t));
        
        // 1:500, 1:167, 1:22
        // std::vector<uint32_t> n_alts = {2,32,65,222,512,1024}; // 1kgp3 dist 
        // std::vector<uint32_t> n_alts = {21,269,9506}; // HRC dist

        // std::vector<uint32_t> n_alts = {n_samples/1000, n_samples/500, n_samples/100, n_samples/20, n_samples/10, n_samples/4, n_samples/2};
        std::vector<uint32_t> n_alts;
        if (loads == nullptr) {
            n_alts = {n_samples/2, n_samples/4, n_samples/10, n_samples/25, n_samples/50, n_samples/100, n_samples/250, n_samples/1000, n_samples/5000, 5, 1};
        } else {
            n_alts = *loads;
        }

        // std::vector<uint32_t> n_alts = {n_samples/100, n_samples/20, n_samples/10, n_samples/4, n_samples/2};
        // std::vector<uint32_t> n_alts = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096};
        //std::vector<uint32_t> n_alts = {512,1024,2048,4096};

        // bitmap_container_t bcont(n_variants,n_samples);
        // bitmap_container_t bcont2(n_variants,n_samples,true,true);
        // STORM_bitmap_container twk(n_samples, n_variants);
        // STORM_bitmap_cont_t** twk = new STORM_bitmap_cont_t*[n_variants];
        // for (int i = 0; i < n_variants; ++i)
            // twk[i] = STORM_bitmap_cont_new();
        STORM_t* twk2 = STORM_new();
        STORM_contiguous_t* twk_cont = STORM_contig_new(n_samples);

        for (int a = 0; a < n_alts.size(); ++a) {
            // break if no more data
            if (n_alts[a] == 0) {
                // Make sure we always compute n_alts = 1
                if (a != 0) {
                    if (n_alts[a-1] != 1) 
                        n_alts[a] = 1;
                } else {
                    // std::cerr << "there are no alts..." << std::endl;
                    break;
                }
            }

            // Break if data has converged
            if (a != 0) {
                if (n_alts[a] == n_alts[a-1]) 
                    break;
            }

            // bcont.clear();
            // bcont2.clear();
            // twk.clear();
            // for (int i = 0; i < n_variants; ++i) {
                // STORM_bitmap_cont_clear(twk[i]);
            // }
            STORM_clear(twk2);
            STORM_contig_clear(twk_cont);

#ifdef USE_ROARING
            roaring_bitmap_t** roaring = new roaring_bitmap_t*[n_variants];
            for (int i = 0; i < n_variants; ++i) roaring[i] = roaring_bitmap_create();
#endif
            
            // Allocation
            memset(vals, 0, n_ints_sample*n_variants*sizeof(uint64_t));

            // PRNG
            std::uniform_int_distribution<uint32_t> distr(0, n_samples-1); // right inclusive

            // Positional information
            std::vector< std::vector<uint32_t> > pos(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint16_t> > pos16(n_variants, std::vector<uint16_t>());

            std::random_device rd;  // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator

            // Draw
            // std::cerr << "Constructing..."; std::cerr.flush();
            uint64_t* vals2 = vals;
            for (uint32_t j = 0; j < n_variants; ++j) {
                for (uint32_t i = 0; i < n_alts[a]; ++i) {
                    uint64_t val = distr(eng);
                    if ((vals2[val / 64] & (1L << (val % 64))) == 0) {
                        pos[j].push_back(val);
                    }
                    vals2[val / 64] |= (1L << (val % 64));
                }

                // Sort to simplify
                std::sort(pos[j].begin(), pos[j].end());

                for (int p = 0; p < pos[j].size(); ++p)
                    pos16[j].push_back(pos[j][p]);

                // Assertion of sortedness.
                for (int p = 1; p < pos[j].size(); ++p) {
                    assert(pos[j][p-1] < pos[j][p]);
                }

                //for (int p = 0; p < pos.back().size(); ++p) std::cerr << " " << pos.back()[p];
                //std::cerr << std::endl;

#ifdef USE_ROARING
                for (int p = 0; p < pos[j].size(); ++p) {
                    roaring_bitmap_add(roaring[j], pos[j][p]);
                    // bcont.Add(j,pos[j][p]);
                }
                // bcont2.Add(j,pos[j]);
                // twk.Add(j, &pos[j][0], pos[j].size());
                // STORM_bitmap_cont_add(twk[j], &pos[j][0], pos[j].size());
#endif
                STORM_add(twk2, &pos[j][0], pos[j].size());
                STORM_contig_add(twk_cont, &pos[j][0], pos[j].size());
                vals2 += n_ints_sample;
            }
            // std::cerr << "Done!" << std::endl;

            uint64_t storm_size = STORM_serialized_size(twk2);
            // std::cerr << "[MEMORY][STORM][" << n_alts[a] << "] Memory for Storm=" << storm_size << "b" << std::endl;


            // uint32_t total_screech = 0;
            // for (uint32_t i = 0; i < n_variants; ++i) {
            //     total_screech += STORM_bitmap_cont_serialized_size(twk[i]);
            // }
            // std::cerr << "Memory used by screech=" << total_screech << "/" << memory_used << " (" << (double)memory_used/total_screech << "-fold)" << std::endl;

            uint64_t int_comparisons = 0;
            for (int k = 0; k < n_variants; ++k) {
                for (int p = k + 1; p < n_variants; ++p) {
                    int_comparisons += pos[k].size() + pos[p].size();
                }
            }
            // std::cerr << "Size of intersections=" << int_comparisons << std::endl;

            const uint64_t n_intersects = ((n_variants * n_variants) - n_variants) / 2;
            const uint64_t n_total_integer_cmps = n_intersects * n_ints_sample;
            // std::cerr << "Total integer comparisons=" << n_total_integer_cmps << std::endl;
            //

            // Optimal cache size is computed as MEMORY_LIMIT / (#bitmaps * 8)
            uint32_t optimal_b = STORM_CACHE_BLOCK_SIZE/(n_ints_sample * 8);
            optimal_b = optimal_b < 5 ? 5 : optimal_b;

            std::vector<uint32_t> block_range;
            uint32_t block_step_size = 4*optimal_b > 25 ? 4*optimal_b/25 : 4*optimal_b;

            for (int i = 1; i < 4*optimal_b; i += block_step_size) {
                block_range.push_back(i);
            }

            // const std::vector<uint32_t> block_range = {3,5,10,25,50,100,200,400,600,800, optimal_b }; // last one is auto


            // Debug
            std::chrono::high_resolution_clock::time_point t1_blocked = std::chrono::high_resolution_clock::now();
            // uint64_t d = 0, diag = 0;
            if (n_samples >= 65536) {
                {
                    PERF_PRE
                    uint64_t total = STORM_pairw_intersect_cardinality(twk2);
                    PERF_POST
                    std::cout << "storm\t" << n_alts[a] << "\t" ;
                    b.PrintPretty();
                    // LINUX_PRINT("storm")
                    // PRINT("storm",b);
                }

                // for (int i = 0; i < block_range.size(); ++i) {
                {
                    PERF_PRE
                    uint64_t total = STORM_pairw_intersect_cardinality_blocked(twk2,0);
                    PERF_POST
                    // LINUX_PRINT("storm-blocked")
                    std::cout << "storm-blocked" << "\t" << n_alts[a] << "\t" ;
                    b.PrintPretty();
                    // PRINT("storm-blocked",b);
                }
                // }
            }

            // {
            //     PERF_PRE
            //     uint64_t total = bcont.intersect();
            //     PERF_POST
            //     // LINUX_PRINT("test-opt")
            //     std::cout << "test-opt\t" << n_alts[a] << "\t" ;
            //     b.PrintPretty();
            //     // PRINT("test-opt",b);
            // }

            // {
            //     PERF_PRE
            //     uint64_t total = bcont.intersect_blocked(optimal_b);
            //     PERF_POST
            //     std::string name = "test-opt-blocked-" + std::to_string(optimal_b);
            //     // LINUX_PRINT(name.c_str())
            //     std::cout << name << "\t" << n_alts[a] << "\t" ;
            //     b.PrintPretty();
            //     // PRINT("test-opt-blocked-" + std::to_string(optimal_b),b);
            // }

            // {
            //     PERF_PRE
            //     uint64_t total = bcont2.intersect_cont();
            //     PERF_POST
            //     // LINUX_PRINT("test-opt-cont-only")
            //     std::cout << "test-opt-cont-only\t" << n_alts[a] << "\t" ;
            //     b.PrintPretty();
            //     // PRINT("test-opt-cont-only",b);
            // }

            // {
            //     PERF_PRE
            //     uint64_t total = bcont2.intersect_blocked_cont(optimal_b);
            //     PERF_POST
            //     std::string name = "test-opt-cont-blocked-" + std::to_string(optimal_b);
            //     // LINUX_PRINT(name.c_str())
            //     std::cout << name << "\t" << n_alts[a] << "\t" ;
            //     b.PrintPretty();
            //     // PRINT("test-opt-cont-blocked-" + std::to_string(optimal_b),b);
            // }

            {
                PERF_PRE
                uint64_t total = STORM_contig_pairw_intersect_cardinality(twk_cont);
                PERF_POST
                // LINUX_PRINT("STORM-contig")
                std::cout << "STORM-contig\t" << n_alts[a] << "\t" ;
                b.PrintPretty();
                // PRINT("STORM-contig",b);
            }

            {
                // for (int i = 0; i < block_range.size(); ++i) {
                    PERF_PRE
                    // Call argument subroutine pointer.
                    uint64_t total = STORM_contig_pairw_intersect_cardinality_blocked(twk_cont, optimal_b);
                    PERF_POST
                    std::string name = "STORM-contig-" + std::to_string(optimal_b);
                    // LINUX_PRINT(name.c_str())
                    std::cout << name << "\t" << n_alts[a] << "\t" ;
                    b.PrintPretty();
                    // PRINT("STORM-contig-" + std::to_string(optimal_b),b);
                // }
            }

            // {
            //     PERF_PRE
            //     uint64_t total = bcont2.intersect_cont_auto();
            //     PERF_POST
            //     // PRINT("automatic",b);
            //     // LINUX_PRINT("automatic")
            //     std::cout << "automatic\t" << n_alts[a] << "\t" ;
            //     b.PrintPretty();
            // }

            // // std::vector<uint32_t> o = {10, 50, 100, 250, 500};

            // // for (int z = 0; z < 5; ++z) {
            // {
            //     uint32_t cutoff = ceil(n_ints_sample*64 / 200.0);
            //     PERF_PRE
            //     uint64_t total = bcont2.intersect_cont_blocked_auto(optimal_b);
            //     PERF_POST
            //     // std::cerr << "[cnt] count=" << cont_count << std::endl;
            //     std::string name = "automatic-list-" + std::to_string(cutoff);
            //     // LINUX_PRINT(name.c_str())
            //     std::cout << name << "\t" << n_alts[a] << "\t" ;
            //     b.PrintPretty();
            //     // PRINT("automatic-list-" + std::to_string(cutoff),b);
            // }
            // }



#if defined(STORM_HAVE_AVX512)
            if ((cpuid & STORM_CPUID_runtime_bit_AVX512BW))
            {
                // SIMD AVX512
                // for (int k = 0; k < block_range.size(); ++k) {
                //     bench_t m8_2_block = fwrapper_blocked<&intersect_bitmaps_avx512_csa>(n_variants, vals, n_ints_sample,block_range[k]);
                //     PRINT("bitmap-avx512-csa-blocked-" + std::to_string(block_range[k]),m8_2_block);
                // }

                // bench_t m8_2 = fwrapper<&intersect_bitmaps_avx512_csa>(n_variants, vals, n_ints_sample);
                // PRINT("bitmap-avx512-csa",m8_2);

                bench_t m8_avx512_block = fwrapper_blocked<&STORM_intersect_count_avx512>(n_variants, vals, n_ints_sample, optimal_b);
                // PRINT("bitmap-avx512-csa-blocked-" + std::to_string(optimal_b), m8_avx512_block );
                std::string m8_avx512_block_name = "bitmap-avx512-csa-blocked-" + std::to_string(optimal_b);
                std::cout << m8_avx512_block_name << "\t" << n_alts[a] << "\t" ;
                m8_avx512_block.PrintPretty();
            }
#endif

#if defined(USE_ROARING)
            // for (int k = 0; k < block_range.size(); ++k) {
            //     bench_t m8_2_block = froarwrapper_blocked(n_variants, n_ints_sample, roaring, block_range[k]);
            //     PRINT("roaring-blocked-" + std::to_string(block_range[k]),m8_2_block);
            // }

            // bench_t broaring = froarwrapper(n_variants, n_ints_sample, roaring);
            // PRINT("roaring",broaring);

            if (n_samples >= 65536) {
            uint64_t roaring_bytes_used = 0;
            for (int k = 0; k < n_variants; ++k) {
                roaring_bytes_used += roaring_bitmap_portable_size_in_bytes(roaring[k]);
            }
            // std::cerr << "[MEMORY][ROARING][" << n_alts[a] << "] Memory for Roaring=" << roaring_bytes_used << "b" << std::endl;

            uint32_t roaring_optimal_b = STORM_CACHE_BLOCK_SIZE / (roaring_bytes_used / n_variants);
            roaring_optimal_b = roaring_optimal_b < 5 ? 5 : roaring_optimal_b;

            bench_t m8_2_block = froarwrapper_blocked(n_variants, n_ints_sample, roaring, roaring_optimal_b);
            // PRINT("roaring-blocked-" + std::to_string(roaring_optimal_b),m8_2_block);
            std::string m8_2_block_name = "roaring-blocked-" + std::to_string(roaring_optimal_b);
            std::cout << m8_2_block_name << "\t" << n_alts[a] << "\t" ;
            m8_2_block.PrintPretty();
            }
#endif

#if defined(STORM_HAVE_AVX2)
            if ((cpuid & STORM_CPUID_runtime_bit_AVX2))
            {
                // uint64_t xx = c_fwrapper(n_variants, vals, n_ints_sample, &intersect_bitmaps_avx2);
                // std::cerr << "test output=" << xx << std::endl;

                // uint64_t xxx = c_fwrapper_blocked(n_variants, vals, n_ints_sample, &intersect_bitmaps_avx2, 100);
                // std::cerr << "test output blocked=" << xxx << std::endl;

                // // SIMD AVX256
                // for (int k = 0; k < block_range.size(); ++k) {
                //     bench_t m3_block3 = fwrapper_blocked<&intersect_bitmaps_avx2>(n_variants, vals, n_ints_sample,block_range[k]);
                //     PRINT("bitmap-avx256-blocked-" + std::to_string(block_range[k]),m3_block3);
                // }
                // bench_t m3_0 = fwrapper<&intersect_bitmaps_avx2>(n_variants, vals, n_ints_sample);
                // PRINT("bitmap-avx256",m3_0);

                bench_t m3_block3 = fwrapper_blocked<&STORM_intersect_count_avx2>(n_variants, vals, n_ints_sample, optimal_b);
                // PRINT("bitmap-avx256-blocked-" + std::to_string(optimal_b), m3_block3);
                std::string m3_block3_name = "bitmap-avx256-blocked-" + std::to_string(optimal_b);
                std::cout << m3_block3_name << "\t" << n_alts[a] << "\t" ;
                m3_block3.PrintPretty();
            }
#endif
            // SIMD SSE4
#if defined(STORM_HAVE_SSE42)
            if ((cpuid & STORM_CPUID_runtime_bit_SSE42))
            {
                // for (int k = 0; k < block_range.size(); ++k) {
                //     bench_t m2_block3 = fwrapper_blocked<&intersect_bitmaps_sse4>(n_variants, vals, n_ints_sample,block_range[k]);
                //     PRINT("bitmap-sse4-csa-blocked-" + std::to_string(block_range[k]),m2_block3);
                // }
                // bench_t m2 = fwrapper<&intersect_bitmaps_sse4>(n_variants, vals, n_ints_sample);
                // PRINT("bitmap-sse4-csa",m2);

                bench_t m2_block3 = fwrapper_blocked<&STORM_intersect_count_sse4>(n_variants, vals, n_ints_sample, optimal_b);
                // PRINT("bitmap-sse4-csa-blocked-" + std::to_string(optimal_b), m2_block3);
                std::string m2_block3_name = "bitmap-sse4-csa-blocked-" + std::to_string(optimal_b);
                std::cout << m2_block3_name << "\t" << n_alts[a] << "\t" ;
                m2_block3.PrintPretty();
            }
#endif

            if (n_alts[a] <= 300) {
                bench_t m4 = flwrapper<&STORM_intersect_count_scalar_list>(n_variants, vals, n_ints_sample, pos);
                // PRINT("bitmap-scalar-skip-list",m4);
                std::string name = "bitmap-scalar-skip-list";
                std::cout << name << "\t" << n_alts[a] << "\t" ;
                m4.PrintPretty();
            }
             
        
#ifdef USE_ROARING
            for (int i = 0; i < n_variants; ++i) roaring_bitmap_free(roaring[i]);
            delete[] roaring;
#endif
        }
        // for (int i = 0; i < n_variants; ++i) STORM_bitmap_cont_free(twk[i]);
        // delete[] twk;
        STORM_free(twk2);
        STORM_contig_free(twk_cont);
        STORM_aligned_free(vals);
    // }
}

int benchmark_hts_contiguous(std::unique_ptr<djinn::VcfReader>& reader, const uint64_t n_samples, uint32_t cutoff = 20000, uint32_t n_threads = std::thread::hardware_concurrency()) {
    if (reader.get() == nullptr) {
      std::cerr << "Illegal VcfReader in benchmark_hts_contiguous!" << std::endl;
        return -1;
    }

    std::cerr << "constructing dense" << std::endl;

    uint32_t* vals = new uint32_t[n_samples];
    uint32_t n_vals;
    uint32_t n_variants_read   = 0;
    uint32_t n_variants_loaded = 0;

    // debug
    uint32_t m_vec = 100;
    STORM_contiguous_t* twk_cont_vec = (STORM_contiguous_t*)malloc(m_vec * sizeof(STORM_contiguous_t));
    for (int i = 0; i < m_vec; ++i) STORM_contig_init(&twk_cont_vec[i], n_samples);
    STORM_contiguous_t* twk_cont_vec_tgt = &twk_cont_vec[0];
    uint32_t n_cont_vec = 1;

    // 32 MB total memory in 2*n_threads
    uint32_t variants_block = 8e6 / (ceil((n_samples) / 64.0) * 8);
    std::cerr << "variants block=" << variants_block << std::endl;
    uint64_t mem_used = 0;

    while (reader->Next()) {
        // Error handling: if either bcf1_t or bcf_hdr_t pointers are NULL then
        // a problem has occured.
        if (reader->bcf1_   == NULL) return -2;
        if (reader->header_ == NULL) return -3;

        // Retrieve pointer to FORMAT field that holds GT data.
        const bcf_fmt_t* fmt = bcf_get_fmt(reader->header_, reader->bcf1_, "GT");
        if (fmt == NULL) continue; // if not found
        if (reader->bcf1_->n_allele != 2) continue; // bi-allelic only
        if (fmt->n != 2) continue; // diplod only

        n_vals = 0;
        for (int i = 0; i < fmt->p_len; ++i) {
            if (((fmt->p[i] >> 1) - 1) != 0) {
                vals[n_vals++] = i;
            }
        }

        if (n_vals == 0) continue;
        STORM_contig_add(twk_cont_vec_tgt, vals, n_vals); // vec add

        ++n_variants_read;
        ++n_variants_loaded;

        if (n_variants_read == variants_block) {
            mem_used += twk_cont_vec_tgt->m_data*twk_cont_vec_tgt->n_bitmaps_vector*sizeof(uint64_t);
            mem_used += twk_cont_vec_tgt->m_scalar*sizeof(uint32_t);
            
            if (n_cont_vec == m_vec) {
                m_vec += 100;
                twk_cont_vec = (STORM_contiguous_t*)realloc(twk_cont_vec, m_vec * sizeof(STORM_contiguous_t));
                for (int i = n_cont_vec; i < m_vec; ++i) {
                    STORM_contig_init(&twk_cont_vec[i], n_samples);
                }
            }
            n_variants_read = 0;
            
            twk_cont_vec_tgt = &twk_cont_vec[n_cont_vec];
            ++n_cont_vec;
        }
        
        if (n_variants_loaded >= cutoff) break;
    }
    uint32_t n_variants = n_variants_loaded;
    std::cerr << "Number of blocks=" << n_cont_vec << " for variants=" << n_variants << std::endl;

    uint32_t n_ints_sample = ceil((n_samples)/64.0);
    uint32_t optimal_b = STORM_CACHE_BLOCK_SIZE/(n_ints_sample * 8);
    optimal_b = optimal_b < 5 ? 5 : optimal_b;

    {
        uint64_t total = 0;
        twk_ld_dynamic_balancer test;
        test.tR = n_cont_vec;
        twk_ld_progress progress;
        progress.n_s = n_samples;
        progress.n_cmps = (uint64_t)n_variants * (n_variants - 1) / 2;
        // std::cerr << "settings cmps=" << progress.n_cmps << std::endl;

        n_threads = n_cont_vec < n_threads ? n_cont_vec : n_threads;
        std::cerr << "spawning threads=" << n_threads << std::endl;
        std::vector<twk_ld_slave> slaves(n_threads);
        std::vector<std::thread*> threads(slaves.size(), nullptr);
        for (int i = 0; i < slaves.size(); ++i) {
            slaves[i].optimal_b = optimal_b;
            slaves[i].ticker = &test;
            slaves[i].twk_cont_vec = twk_cont_vec;
            slaves[i].progress = &progress;
        }
        
        uint32_t from = 0, to = 0, total_comps = 0; uint8_t type = 0;
        progress.Start();

        for (int i = 0; i < slaves.size(); ++i)
            threads[i] = slaves[i].Start();

        PERF_PRE
        for (int i = 0; i < slaves.size(); ++i)
            threads[i]->join();

        uint64_t comps = 0;
        for (int i = 0; i < slaves.size(); ++i) {
            total += slaves[i].count;
            comps += slaves[i].comps;
        }
        PERF_POST

        std::string name = "STORM-dist-" + std::to_string(optimal_b);
        std::cout << name << "\t";
        b.PrintPretty();
        progress.PrintFinal();

        std::cerr << "comps=" << comps << std::endl;
    }
    
    delete[] vals;
    for (int i = 0; i < m_vec; ++i) STORM_contig_free(&twk_cont_vec[i]);
    delete[] twk_cont_vec;

    return EXIT_SUCCESS;
}

int benchmark_hts_sparse(std::unique_ptr<djinn::VcfReader>& reader, const uint64_t n_samples, uint32_t cutoff = 20000, uint32_t n_threads = std::thread::hardware_concurrency()) {
    if (reader.get() == nullptr) {
      std::cerr << "Illegal VcfReader in benchmark_hts_sparse!" << std::endl;
        return -1;
    }

    std::cerr << "constructing sparse" << std::endl;

    uint32_t* vals = new uint32_t[n_samples+65536];
    uint32_t n_vals;
    uint32_t n_variants_read = 0;
    uint32_t n_variants_loaded = 0;

    // debug
    uint32_t m_vec = 100;
    STORM_t* twk_cont_vec = (STORM_t*)malloc(m_vec * sizeof(STORM_t));
    for (int i = 0; i < m_vec; ++i) STORM_init(&twk_cont_vec[i]);
    STORM_t* twk_cont_vec_tgt = &twk_cont_vec[0];
    uint32_t n_cont_vec = 1;
    uint64_t mem_used = 0;

    while (reader->Next()) {
        // Error handling: if either bcf1_t or bcf_hdr_t pointers are NULL then
        // a problem has occured.
        if (reader->bcf1_   == NULL) return -2;
        if (reader->header_ == NULL) return -3;

        // Retrieve pointer to FORMAT field that holds GT data.
        const bcf_fmt_t* fmt = bcf_get_fmt(reader->header_, reader->bcf1_, "GT");
        if (fmt == NULL) continue; // if not found
        if (reader->bcf1_->n_allele != 2) continue; // bi-allelic only
        if (fmt->n != 2) continue; // diplod only

        n_vals = 0;
        for (int i = 0; i < fmt->p_len; ++i) {
            if (((fmt->p[i] >> 1) - 1) != 0) {
                vals[n_vals++] = i;
            }
        }

        if (n_vals == 0) continue;
        STORM_add(twk_cont_vec_tgt, vals, n_vals); // vec add

        ++n_variants_read;
        ++n_variants_loaded;
        

        mem_used = STORM_serialized_size(twk_cont_vec_tgt);
        // std::cerr << "mem used=" << mem_used << std::endl;

        if (mem_used > 3000000) {
            std::cerr << "mem_used: " << mem_used << " -> " << " variants: " << n_variants_read << "/" << n_variants_loaded << std::endl;
            // mem_used += twk_cont_vec_tgt->m_data*twk_cont_vec_tgt->n_bitmaps_vector*sizeof(uint64_t);
            // mem_used += twk_cont_vec_tgt->m_scalar*sizeof(uint32_t);
            
            if (n_cont_vec == m_vec) {
                m_vec += 100;
                twk_cont_vec = (STORM_t*)realloc(twk_cont_vec, m_vec * sizeof(STORM_t));
                for (int i = n_cont_vec; i < m_vec; ++i) {
                    STORM_init(&twk_cont_vec[i]);
                }
            }
            n_variants_read = 0;
            
            twk_cont_vec_tgt = &twk_cont_vec[n_cont_vec];
            ++n_cont_vec;
        }
        
        if (n_variants_loaded >= cutoff) break;
    }
    uint32_t n_variants = n_variants_loaded;
    std::cerr << "Number of blocks=" << n_cont_vec << " for variants=" << n_variants << std::endl;

    uint32_t n_ints_sample = ceil((n_samples)/64.0);
    // uint32_t optimal_b = STORM_CACHE_BLOCK_SIZE/(n_ints_sample * 8);
    // optimal_b = optimal_b < 5 ? 5 : optimal_b;

    {
        uint64_t total = 0;
        twk_ld_dynamic_balancer test;
        test.tR = n_cont_vec;
        twk_ld_progress progress;
        progress.n_s = n_samples;
        progress.n_cmps = (uint64_t)n_variants * (n_variants - 1) / 2;
        // std::cerr << "settings cmps=" << progress.n_cmps << std::endl;

        n_threads = n_cont_vec < n_threads ? n_cont_vec : n_threads;
        std::cerr << "spawning threads=" << n_threads << std::endl;
        std::vector<twk_ld_slave> slaves(n_threads);
        std::vector<std::thread*> threads(slaves.size(), nullptr);
        for (int i = 0; i < slaves.size(); ++i) {
            slaves[i].optimal_b = 5;
            slaves[i].ticker    = &test;
            slaves[i].twk_vec   = twk_cont_vec;
            slaves[i].progress  = &progress;
        }
        
        uint32_t from = 0, to = 0, total_comps = 0; uint8_t type = 0;
        progress.Start();

        for (int i = 0; i < slaves.size(); ++i)
            threads[i] = slaves[i].StartSparse();

        PERF_PRE
        for (int i = 0; i < slaves.size(); ++i)
            threads[i]->join();

        uint64_t comps = 0;
        for (int i = 0; i < slaves.size(); ++i) {
            total += slaves[i].count;
            comps += slaves[i].comps;
        }
        PERF_POST

        std::string name = "STORM-dist-" + std::to_string(5);
        std::cout << name << "\t";
        b.PrintPretty();
        progress.PrintFinal();

        std::cerr << "comps=" << comps << std::endl;
    }
    
    delete[] vals;
    for (int i = 0; i < m_vec; ++i) STORM_free(&twk_cont_vec[i]);
    delete[] twk_cont_vec;

    return EXIT_SUCCESS;
}

int benchmark_hts(std::string input_file, uint32_t cutoff = 20000, uint32_t n_threads = std::thread::hardware_concurrency()) {
    // function to get the instance.
    std::unique_ptr<djinn::VcfReader> reader = djinn::VcfReader::FromFile(input_file, std::thread::hardware_concurrency());
    
    // If the file or stream could not be opened we exit here.
    if (reader.get() == nullptr) {
        std::cerr << "Could not open input handle \"" << input_file << "\"!" << std::endl;
        return -1;
    }

    const uint64_t n_samples = 2*reader->n_samples_;
    std::cerr << "n_samples = " << n_samples << std::endl;
    // return benchmark_hts_contiguous(reader, n_samples, cutoff, n_threads);
    // return benchmark_hts_sparse(reader, n_samples, cutoff, n_threads);
    if (n_samples < 250000) return benchmark_hts_contiguous(reader, n_samples, cutoff, n_threads);
    else return benchmark_hts_sparse(reader, n_samples, cutoff, n_threads);

    uint32_t* vals = new uint32_t[n_samples];
    uint32_t n_vals;
    // STORM_contiguous_t* twk_cont = STORM_contig_new(n_samples);
    uint32_t n_variants_read = 0;
    uint32_t n_variants_loaded = 0;

    // debug
    uint32_t m_vec = 100;
    STORM_contiguous_t* twk_cont_vec = (STORM_contiguous_t*)malloc(m_vec * sizeof(STORM_contiguous_t));
    // STORM_contiguous_t* twk_cont_vec = new STORM_contiguous_t[m_vec];
    for (int i = 0; i < m_vec; ++i) STORM_contig_init(&twk_cont_vec[i], n_samples);
    STORM_contiguous_t* twk_cont_vec_tgt = &twk_cont_vec[0];
    uint32_t n_cont_vec = 1;

    // STORM_t* twk2 = STORM_new();

    // uint64_t total = STORM_pairw_intersect_cardinality(twk2);

    // STORM_contiguous_t* twk_cont_block1 = STORM_contig_new(n_samples);
    // STORM_contiguous_t* twk_cont_block2 = STORM_contig_new(n_samples);
    // STORM_contiguous_t* twk_cont_tgt = twk_cont_block1;

// #ifdef USE_ROARING
//     roaring_bitmap_t** roaring = new roaring_bitmap_t*[cutoff];
//     for (int i = 0; i < cutoff; ++i) roaring[i] = roaring_bitmap_create();
// #endif

    // 32 MB total memory in 2*n_threads
    uint32_t variants_block = 8e6 / (ceil((n_samples) / 64.0) * 8);
    std::cerr << "variants block=" << variants_block << std::endl;
    uint64_t mem_used = 0;

    while (reader->Next()) {
        // Error handling: if either bcf1_t or bcf_hdr_t pointers are NULL then
        // a problem has occured.
        if (reader->bcf1_   == NULL) return -2;
        if (reader->header_ == NULL) return -3;

        // Retrieve pointer to FORMAT field that holds GT data.
        const bcf_fmt_t* fmt = bcf_get_fmt(reader->header_, reader->bcf1_, "GT");
        if (fmt == NULL) continue; // if not found
        if (reader->bcf1_->n_allele != 2) continue; // bi-allelic only
        if (fmt->n != 2) continue; // diplod only

        n_vals = 0;
        for (int i = 0; i < fmt->p_len; ++i) {
            if (((fmt->p[i] >> 1) - 1) != 0) {
                vals[n_vals++] = i;
            }
        }

        if (n_vals == 0) continue;

        
        // STORM_contig_add(twk_cont, vals, n_vals);
        // STORM_contig_add(twk_cont_tgt, vals, n_vals);
        STORM_contig_add(twk_cont_vec_tgt, vals, n_vals); // vec add
// #ifdef USE_ROARING
//         roaring_bitmap_add_many(roaring[n_variants_loaded++], n_vals, vals);
// #endif

        ++n_variants_read;
        ++n_variants_loaded;
        

        if (n_variants_read == variants_block) {
            // std::cerr << "block=" << n_cont_vec << std::endl;
            // std::cerr << "variants=" << n_variants_read << "," << n_variants_loaded << "@" << n_cont_vec << "/" << m_vec << " n_vals=" << n_vals << std::endl;
            // std::cerr << "scalars=" << twk_cont_vec_tgt->m_scalar << " for size=" << twk_cont_vec_tgt->m_data*sizeof(uint32_t) << std::endl;
            // std::cerr << "bitmaps=" << twk_cont_vec_tgt->m_data << "," << twk_cont_vec_tgt->n_bitmaps_vector << " for size=" << twk_cont_vec_tgt->m_data*twk_cont_vec_tgt->n_bitmaps_vector*sizeof(uint64_t) << std::endl;
            mem_used += twk_cont_vec_tgt->m_data*twk_cont_vec_tgt->n_bitmaps_vector*sizeof(uint64_t);
            mem_used += twk_cont_vec_tgt->m_scalar*sizeof(uint32_t);
            // std::cerr << "[mem] " << mem_used << std::endl;

            if (n_cont_vec == m_vec) {
                m_vec += 100;
                // std::cerr << "reallocating: " << n_cont_vec << "->" << m_vec << std::endl;
                twk_cont_vec = (STORM_contiguous_t*)realloc(twk_cont_vec, m_vec * sizeof(STORM_contiguous_t));
                for (int i = n_cont_vec; i < m_vec; ++i) {
                    STORM_contig_init(&twk_cont_vec[i], n_samples);
                }
            }
            n_variants_read = 0;
            
            twk_cont_vec_tgt = &twk_cont_vec[n_cont_vec];
            ++n_cont_vec;
        }
        
        // if (twk_cont->n_data >= 50000) twk_cont_tgt = twk_cont_block2;


        if (n_variants_loaded >= cutoff) break;

        // Encode from htslib Bcf encoding by passing the arguments:
        // p: pointer to genotype data array
        // p_len: length of data
        // n: stride size (number of bytes per individual = base ploidy)
        // n_allele: number of alleles
    }
    uint32_t n_variants = n_variants_loaded;
    std::cerr << "Number of blocks=" << n_cont_vec << " for variants=" << n_variants << std::endl;

    uint32_t n_ints_sample = ceil((n_samples)/64.0);
    uint32_t optimal_b = STORM_CACHE_BLOCK_SIZE/(n_ints_sample * 8);
    optimal_b = optimal_b < 5 ? 5 : optimal_b;

    // test
    {
        uint64_t total = 0;
        twk_ld_dynamic_balancer test;
        test.tR = n_cont_vec;
        twk_ld_progress progress;
        progress.n_s = n_samples;
        progress.n_cmps = (uint64_t)n_variants * (n_variants - 1) / 2;
        // std::cerr << "settings cmps=" << progress.n_cmps << std::endl;

        n_threads = n_cont_vec < n_threads ? n_cont_vec : n_threads;
        std::cerr << "spawning threads=" << n_threads << std::endl;
        std::vector<twk_ld_slave> slaves(n_threads);
        std::vector<std::thread*> threads(slaves.size(), nullptr);
        for (int i = 0; i < slaves.size(); ++i) {
            slaves[i].optimal_b = optimal_b;
            slaves[i].ticker = &test;
            slaves[i].twk_cont_vec = twk_cont_vec;
            slaves[i].progress = &progress;
        }
        
        uint32_t from = 0, to = 0, total_comps = 0; uint8_t type = 0;
        progress.Start();

        for (int i = 0; i < slaves.size(); ++i)
            threads[i] = slaves[i].Start();

        PERF_PRE
        // while (test.GetBlockPair(from, to, type)) {
        //     std::cerr << from << "," << to << " type=" << (int)type << " total=" << total_comps << std::endl;
            
        //     if (from == to) {
        //         total += STORM_contig_pairw_intersect_cardinality_blocked(&twk_cont_vec[to], optimal_b);
        //     }  else {
        //         total += STORM_contig_pairw_sq_intersect_cardinality_blocked(&twk_cont_vec[to], &twk_cont_vec[from], optimal_b >> 1);
        //     }
        //     ++total_comps;
        // }
        for (int i = 0; i < slaves.size(); ++i)
            threads[i]->join();

        uint64_t comps = 0;
        for (int i = 0; i < slaves.size(); ++i) {
            total += slaves[i].count;
            comps += slaves[i].comps;
        }

        PERF_POST

        std::string name = "STORM-dist-" + std::to_string(optimal_b);
        // LINUX_PRINT(name.c_str())
        std::cout << name << "\t";
        b.PrintPretty();
        progress.PrintFinal();

        std::cerr << "comps=" << comps << std::endl;
    }

    // {
    //     // for (int i = 0; i < block_range.size(); ++i) {
    //         PERF_PRE
    //         // Call argument subroutine pointer.
    //         uint64_t total = STORM_contig_pairw_intersect_cardinality_blocked(twk_cont, optimal_b);
    //         PERF_POST
    //         std::string name = "STORM-contig-" + std::to_string(optimal_b);
    //         // LINUX_PRINT(name.c_str())
    //         std::cout << name << "\t";
    //         b.PrintPretty();
    //         // PRINT("STORM-contig-" + std::to_string(optimal_b),b);
    //     // }
    // }

    // {
    //     uint64_t roaring_bytes_used = 0;
    //     for (int k = 0; k < n_variants; ++k) {
    //         roaring_bytes_used += roaring_bitmap_portable_size_in_bytes(roaring[k]);
    //     }
    //     // std::cerr << "[MEMORY][ROARING][" << n_alts[a] << "] Memory for Roaring=" << roaring_bytes_used << "b" << std::endl;

    //     uint32_t roaring_optimal_b = STORM_CACHE_BLOCK_SIZE / (roaring_bytes_used / n_variants);
    //     roaring_optimal_b = roaring_optimal_b < 5 ? 5 : roaring_optimal_b;

    //     bench_t m8_2_block = froarwrapper_blocked(n_variants, n_ints_sample, roaring, roaring_optimal_b);
    //     // PRINT("roaring-blocked-" + std::to_string(roaring_optimal_b),m8_2_block);
    //     std::string m8_2_block_name = "roaring-blocked-" + std::to_string(roaring_optimal_b);
    //     std::cout << m8_2_block_name << "\t" ;
    //     m8_2_block.PrintPretty();
    // }

    // for (int i = 10; i < 800; i += 10) {
    //     PERF_PRE
    //      uint64_t total = STORM_contig_pairw_sq_intersect_cardinality_blocked(twk_cont_block1, twk_cont_block2, optimal_b >> 1);
    //     unified.end(results); 
    //     std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now(); 
    //     auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1); 
    //     uint64_t n_comps = twk_cont_block1->n_data * twk_cont_block2->n_data; 
    //     bench_t b(results, n_comps * 2*n_ints_sample); 
    //     b.total = total; 
    //     b.time_ms = time_span.count(); 
    //     b.throughput = (( ( n_comps * 2*n_ints_sample ) * sizeof(uint64_t)) / (1024*1024.0)) / (b.time_ms / 1000.0);
        
    //     std::string name = "STORM-contig-parts-" + std::to_string(i);
    //     // LINUX_PRINT(name.c_str())
    //     std::cout << name << "\t";
    //     b.PrintPretty();
    // }

    // {
    //     std::cerr << "test: " << twk_cont_block1->n_data << " and " << twk_cont_block2->n_data << std::endl;
    //     PERF_PRE
    //     uint64_t total_b1 = STORM_contig_pairw_intersect_cardinality_blocked(twk_cont_block1, optimal_b);
    //     uint64_t total_b2 = STORM_contig_pairw_intersect_cardinality_blocked(twk_cont_block2, optimal_b);
    //     uint64_t total_b1b2 = STORM_contig_pairw_sq_intersect_cardinality_blocked(twk_cont_block1, twk_cont_block2, optimal_b >> 1);
    //     uint64_t total = total_b1 + total_b2 + total_b1b2;
    //     PERF_POST
    //     std::string name = "STORM-contig-parts-" + std::to_string(optimal_b);
    //     // LINUX_PRINT(name.c_str())
    //     std::cout << name << "\t";
    //     b.PrintPretty();

    //     std::cerr << total_b1  << " + " << total_b2 << " + " << total_b1b2 << " = " << (total_b1 + total_b2 + total_b1b2) << std::endl;
    // }

    // {
    //     PERF_PRE
    //     uint64_t total = STORM_contig_pairw_intersect_cardinality_many2(twk_cont_vec, n_cont_vec + 1);
    //     PERF_POST
    //     std::string name = "STORM-contig-blocks-" + std::to_string(optimal_b);
    //     // LINUX_PRINT(name.c_str())
    //     std::cout << name << "\t";
    //     b.PrintPretty();
    // }

// #ifdef USE_ROARING
//     for (int i = 0; i < n_variants; ++i) roaring_bitmap_free(roaring[i]);
//     delete[] roaring;
// #endif
    
    delete[] vals;
    // STORM_contig_free(twk_cont);st

    // debug
    // STORM_contig_free(twk_cont_block1);
    // STORM_contig_free(twk_cont_block2);
    for (int i = 0; i < m_vec; ++i) STORM_contig_free(&twk_cont_vec[i]);
    delete[] twk_cont_vec;

    return EXIT_SUCCESS;
}

const char* usage(void) {
    return
        "\n"
        "About:   Computes sum(popcnt(A & B)) for the all-vs-all comparison of N integer\n"
        "         lists bounded by [0, M). This benchmark will compute this statistics using\n"
        "         different algorithms.\n"
        "Usage:   benchmark <M> <N> [v1[,v2]] \n"
        "\n"
        "Example:\n"
        "   benchmark 4092 10000\n"
        "   benchmark 4092 1,10,100,1000\n"
        "\n";
}

std::vector<std::string> split(const std::string& s, char delimiter) {
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
   }
   return tokens;
}

int main(int argc, char **argv) { 

    return benchmark_hts(std::string(argv[1]), atoi(argv[2]), atoi(argv[3]));

    if (argc == 1) {
        printf("%s",usage());
        return EXIT_FAILURE;
    }

    std::vector<uint32_t>* loads = nullptr;

    if (argc > 3) {
        std::string s(argv[3]);
        std::vector<std::string> ret = split(s, ',');
        loads = new std::vector<uint32_t>();
        for (int i = 0; i < ret.size(); ++i) {
            loads->push_back(std::atoi(ret[i].c_str()));
        }
    }
    
    if (argc == 2) {
        intersect_test(std::atoi(argv[1]), 10000, loads);
    } else {
        int64_t n_samples = std::atoi(argv[1]);
        if (n_samples <= 0) {
            std::cerr << "Cannot have non-positive number of samples..." << std::endl;
            return EXIT_FAILURE;
        }

        int64_t n_vals = std::atoi(argv[2]);
        if (n_vals <= 0) {
            std::cerr << "Cannot have non-positive number of vectors..." << std::endl;
            return EXIT_FAILURE;
        }

        // if (n_samples < 256000) {
            intersect_test(n_samples, n_vals, loads);
        // } else {
        //     benchmark_large(n_samples, n_vals, loads);
        // }
    }
    delete loads;
    return EXIT_SUCCESS;
}
