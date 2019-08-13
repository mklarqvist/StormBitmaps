#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <memory>
#include <bitset>

#define USE_ROARING

#ifdef USE_ROARING
#include <roaring/roaring.h>
#endif

#include "fast_intersect_count.h"
#include "classes.h"

#if defined(__AVX512F__) && __AVX512F__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    6
#define SIMD_ALIGNMENT  64
#elif defined(__AVX2__) && __AVX2__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    5
#define SIMD_ALIGNMENT  32
#elif defined(__AVX__) && __AVX__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    4
#define SIMD_ALIGNMENT  16
#elif defined(__SSE4_1__) && __SSE4_1__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    3
#define SIMD_ALIGNMENT  16
#elif defined(__SSE2__) && __SSE2__ == 1
#define SIMD_AVAILABLE  0 // unsupported version
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#elif defined(__SSE__) && __SSE__ == 1
#define SIMD_AVAILABLE  0 // unsupported version
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#else
#define SIMD_AVAILABLE  0
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#endif

uint64_t get_cpu_cycles() {
    uint64_t result;
    __asm__ volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax":"=a"
                     (result)::"%rdx");
    return result;
};

// Convenience wrapper
struct bench_t {
    uint64_t count;
    uint32_t milliseconds;
    double throughput;
    uint64_t cpu_cycles;
};

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
template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints)>
bench_t fwrapper(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints) {    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;
    
    const uint64_t cycles_start = get_cpu_cycles();
    for (int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
        for (int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
            total += (*f)(&vals[offset], &vals[inner_offset], n_ints);
        }
        offset += n_ints;
    }
    const uint64_t cycles_end = get_cpu_cycles();
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2; 
    b.throughput = ((n_comps*n_ints*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
    b.cpu_cycles = cycles_end - cycles_start;

    return(b);
}

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints)>
bench_t fwrapper_blocked(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, uint32_t bsize = 200) {    
    uint64_t total = 0;

    bsize = (bsize == 0 ? 10 : bsize);
    const uint32_t n_blocks1 = n_variants / bsize;
    const uint32_t n_blocks2 = n_variants / bsize;
    // uint64_t d = 0;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    const uint64_t cycles_start = get_cpu_cycles();
    uint32_t i  = 0;
    uint32_t tt = 0;
    for (/**/; i + bsize <= n_variants; i += bsize) {
        // diagonal component
        uint32_t left = i*n_ints;
        uint32_t right = 0;
        for (uint32_t j = 0; j < bsize; ++j, left += n_ints) {
            right = left + n_ints;
            for (uint32_t jj = j + 1; jj < bsize; ++jj, right += n_ints) {
                total += (*f)(&vals[left], &vals[right], n_ints);
                // ++d;
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + bsize;
        for (/**/; j + bsize <= n_variants; j += bsize) {
            left = curi*n_ints;
            for (uint32_t ii = 0; ii < bsize; ++ii, left += n_ints) {
                right = j*n_ints;
                for (uint32_t jj = 0; jj < bsize; ++jj, right += n_ints) {
                    total += (*f)(&vals[left], &vals[right], n_ints);
                    // ++d;
                }
            }
        }

        // residual
        right = j*n_ints;
        for (/**/; j < n_variants; ++j, right += n_ints) {
            left = curi*n_ints;
            for (uint32_t jj = 0; jj < bsize; ++jj, left += n_ints) {
                total += (*f)(&vals[left], &vals[right], n_ints);
                // ++d;
            }
        }
    }
    // residual tail
    uint32_t left = i*n_ints;
    for (/**/; i < n_variants; ++i, left += n_ints) {
        uint32_t right = left + n_ints;
        for (uint32_t j = i + 1; j < n_variants; ++j, right += n_ints) {
            total += (*f)(&vals[left], &vals[right], n_ints);
            // ++d;
        }
    }

    const uint64_t cycles_end = get_cpu_cycles();
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2; 
    b.throughput = ((n_comps*n_ints*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
    b.cpu_cycles = cycles_end - cycles_start;

    // std::cerr << "d=" << d << std::endl;

    return(b);
}

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t* l1, const uint32_t* l2, const uint32_t len1, const uint32_t len2)>
bench_t flwrapper(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, const std::vector< std::vector<uint32_t> >& pos) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;

    const uint64_t cycles_start = get_cpu_cycles();
    for (int i = 0; i < n_variants; ++i) {
    inner_offset = offset + n_ints;
       for (int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
           total += (*f)(&vals[offset], &vals[inner_offset], &pos[i][0], &pos[j][0], (uint32_t)pos[i].size(), (uint32_t)pos[j].size());
       }
       offset += n_ints;
    }
    const uint64_t cycles_end = get_cpu_cycles();

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_ints*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
    b.cpu_cycles = cycles_end - cycles_start;

    return(b);
}

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2)>
bench_t flwrapper_blocked(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, const std::vector< std::vector<uint32_t> >& pos, uint32_t bsize = 200) { 
    uint64_t total = 0;

    bsize = (bsize == 0 ? 10 : bsize);
    const uint32_t n_blocks1 = n_variants / bsize;
    const uint32_t n_blocks2 = n_variants / bsize;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // uint64_t blocked_con_tot = 0;
    const uint64_t cycles_start = get_cpu_cycles();
    uint32_t i  = 0;
    uint32_t tt = 0;
    for (/**/; i + bsize <= n_variants; i += bsize) {
        // diagonal component
        uint32_t left = i*n_ints;
        uint32_t right = 0;
        for (uint32_t j = 0; j < bsize; ++j, left += n_ints) {
            right = left + n_ints;
            for (uint32_t jj = j + 1; jj < bsize; ++jj, right += n_ints) {
                total += (*f)(&vals[left], &vals[right], pos[i+j], pos[i+jj]);
                //total += (*f)(&vals[left], &vals[right], n_ints);
                // ++d;
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + bsize;
        for (/**/; j + bsize <= n_variants; j += bsize) {
            left = curi*n_ints;
            for (uint32_t ii = 0; ii < bsize; ++ii, left += n_ints) {
                right = j*n_ints;
                for (uint32_t jj = 0; jj < bsize; ++jj, right += n_ints) {
                    // total += (*f)(&vals[left], &vals[right], n_ints);
                    total += (*f)(&vals[left], &vals[right], pos[curi + ii], pos[j + jj]);
                    // ++d;
                }
            }
        }

        // residual
        right = j*n_ints;
        for (/**/; j < n_variants; ++j, right += n_ints) {
            left = curi*n_ints;
            for (uint32_t jj = 0; jj < bsize; ++jj, left += n_ints) {
                // total += (*f)(&vals[left], &vals[right], n_ints);
                total += (*f)(&vals[left], &vals[right], pos[curi + jj], pos[j]);
                // ++d;
            }
        }
    }
    // residual tail
    uint32_t left = i*n_ints;
    for (/**/; i < n_variants; ++i, left += n_ints) {
        uint32_t right = left + n_ints;
        for (uint32_t j = i + 1; j < n_variants; ++j, right += n_ints) {
            // total += (*f)(&vals[left], &vals[right], n_ints);
            total += (*f)(&vals[left], &vals[right], pos[i], pos[j]);
            // ++d;
        }
    }
    const uint64_t cycles_end = get_cpu_cycles();
    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2; 
    b.throughput = ((n_comps*n_ints*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
    b.cpu_cycles = cycles_end - cycles_start;

    return(b);
}

template <class int_t, uint64_t (f)(const std::vector<int_t>& rle1, const std::vector<int_t>& rle2)>
bench_t frlewrapper(const std::vector< std::vector<int_t> >& rle, const uint32_t n_ints) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;
    const uint32_t n_variants = rle.size();

    const uint64_t cycles_start = get_cpu_cycles();
    for (int i = 0; i < n_variants; ++i) {
        for (int j = i + 1; j < n_variants; ++j) {
            total += (*f)(rle[i], rle[j]);
        }
    }
    const uint64_t cycles_end = get_cpu_cycles();

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_ints*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
    b.cpu_cycles = cycles_end - cycles_start;

    return(b);
}

template <uint64_t (f)(const uint16_t* v1, const uint16_t* v2, const uint32_t len1, const uint32_t len2)>
bench_t frawwrapper(const uint32_t n_variants, const uint32_t n_vals_actual, const std::vector< std::vector<uint16_t> >& pos) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;

    const uint64_t cycles_start = get_cpu_cycles();
    for (int k = 0; k < n_variants; ++k) {
        for (int p = k + 1; p < n_variants; ++p) {
            total += (*f)(&pos[k][0], &pos[p][0], pos[k].size(), pos[p].size());
        }
    }
    const uint64_t cycles_end = get_cpu_cycles();

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_vals_actual*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
    b.cpu_cycles = cycles_end - cycles_start;

    return(b);
}

#ifdef USE_ROARING
bench_t froarwrapper(const uint32_t n_variants, const uint32_t n_vals_actual, roaring_bitmap_t** bitmaps) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;

    const uint64_t cycles_start = get_cpu_cycles();
    for (int k = 0; k < n_variants; ++k) {
        for (int p = k + 1; p < n_variants; ++p) {
            total += roaring_bitmap_and_cardinality(bitmaps[k], bitmaps[p]);
        }
    }
    const uint64_t cycles_end = get_cpu_cycles();

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_vals_actual*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
    b.cpu_cycles = cycles_end - cycles_start;

    return(b);
}

bench_t froarwrapper_blocked(const uint32_t n_variants, const uint32_t n_vals_actual, roaring_bitmap_t** bitmaps, const uint32_t bsize) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;
    uint64_t blocked_con_tot = 0;
    uint32_t i  = 0;
    uint32_t tt = 0;

    const uint64_t cycles_start = get_cpu_cycles();
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
    const uint64_t cycles_end = get_cpu_cycles();

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = blocked_con_tot; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_vals_actual*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
    b.cpu_cycles = cycles_end - cycles_start;

    return(b);
}
#endif

void intersect_test(uint32_t n_samples, uint32_t n_variants) {
    // uint64_t* a = nullptr;
    // intersect(a,0,0);

#define PRINT(name,bench) std::cout << n_samples << "\t" << n_alts[a] << "\t" << name << "\t" << bench.milliseconds << "\t" << bench.cpu_cycles << "\t" << bench.count << "\t" << \
        bench.throughput << "\t" << \
        (bench.milliseconds == 0 ? 0 : (int_comparisons*1000.0 / bench.milliseconds / 1000000.0)) << "\t" << \
        (n_intersects*1000.0 / (bench.milliseconds) / 1000000.0) << "\t" << \
        (bench.milliseconds == 0 ? 0 : n_total_integer_cmps*sizeof(uint64_t) / (bench.milliseconds/1000.0) / (1024.0*1024.0)) << "\t" << \
        (bench.cpu_cycles == 0 ? 0 : bench.cpu_cycles / (double)n_total_integer_cmps) << "\t" << \
        (bench.cpu_cycles == 0 ? 0 : bench.cpu_cycles / (double)n_intersects) << std::endl

    
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

        std::cerr << "Generating: " << n_samples << " samples for " << n_variants << " variants" << std::endl;
        const uint64_t memory_used = n_ints_sample*n_variants*sizeof(uint64_t);
        std::cerr << "Allocating: " << memory_used/(1024 * 1024.0) << "Mb" << std::endl;

        uint64_t* vals = (uint64_t*)TWK_aligned_malloc(SIMD_ALIGNMENT, n_ints_sample*n_variants*sizeof(uint64_t));
        
        // 1:500, 1:167, 1:22
        // std::vector<uint32_t> n_alts = {2,32,65,222,512,1024}; // 1kgp3 dist 
        // std::vector<uint32_t> n_alts = {21,269,9506}; // HRC dist

        // std::vector<uint32_t> n_alts = {n_samples/1000, n_samples/500, n_samples/100, n_samples/20, n_samples/10, n_samples/4, n_samples/2};
        std::vector<uint32_t> n_alts = {n_samples/2, n_samples/4, n_samples/10, n_samples/25, n_samples/50, n_samples/100, n_samples/250, n_samples/1000, n_samples/5000, 5, 1};
        // std::vector<uint32_t> n_alts = {n_samples/100, n_samples/20, n_samples/10, n_samples/4, n_samples/2};
        // std::vector<uint32_t> n_alts = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096};
        //std::vector<uint32_t> n_alts = {512,1024,2048,4096};

        bitmap_container_t bcont(n_variants,n_samples);
        bitmap_container_t bcont2(n_variants,n_samples,true,true);
        // TWK_bitmap_container twk(n_samples, n_variants);
        // TWK_bitmap_cont_t** twk = new TWK_bitmap_cont_t*[n_variants];
        // for (int i = 0; i < n_variants; ++i)
            // twk[i] = TWK_bitmap_cont_new();
        TWK_cont_t* twk2 = TWK_cont_new();
        

        for (int a = 0; a < n_alts.size(); ++a) {
            // break if no more data
            if (n_alts[a] == 0) {
                // Make sure we always compute n_alts = 1
                if (a != 0) {
                    if (n_alts[a-1] != 1) 
                        n_alts[a] = 1;
                } else {
                    std::cerr << "there are no alts..." << std::endl;
                    break;
                }
            }

            // Break if data has converged
            if (a != 0) {
                if (n_alts[a] == n_alts[a-1]) 
                    break;
            }

            bcont.clear();
            bcont2.clear();
            // twk.clear();
            // for (int i = 0; i < n_variants; ++i) {
                // TWK_bitmap_cont_clear(twk[i]);
            // }
            TWK_cont_clear(twk2);

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
            // std::vector< std::vector<uint16_t> > pos16(n_variants, std::vector<uint16_t>());

            std::random_device rd;  // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator

            // Draw
            std::cerr << "Constructing..."; std::cerr.flush();
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

                // for (int p = 0; p < pos[j].size(); ++p)
                //     pos16[j].push_back(pos[j][p]);

                // Assertion of sortedness.
                for (int p = 1; p < pos[j].size(); ++p) {
                    assert(pos[j][p-1] < pos[j][p]);
                }

                //for (int p = 0; p < pos.back().size(); ++p) std::cerr << " " << pos.back()[p];
                //std::cerr << std::endl;

#ifdef USE_ROARING
                for (int p = 0; p < pos[j].size(); ++p) {
                    roaring_bitmap_add(roaring[j], pos[j][p]);
                    bcont.Add(j,pos[j][p]);
                }
                bcont2.Add(j,pos[j]);
                // twk.Add(j, &pos[j][0], pos[j].size());
                // TWK_bitmap_cont_add(twk[j], &pos[j][0], pos[j].size());
                TWK_cont_add(twk2, &pos[j][0], pos[j].size());
#endif
                vals2 += n_ints_sample;
            }
            std::cerr << "Done!" << std::endl;

            // uint32_t total_screech = 0;
            // for (uint32_t i = 0; i < n_variants; ++i) {
            //     total_screech += TWK_bitmap_cont_serialized_size(twk[i]);
            // }
            // std::cerr << "Memory used by screech=" << total_screech << "/" << memory_used << " (" << (double)memory_used/total_screech << "-fold)" << std::endl;

            uint64_t int_comparisons = 0;
            for (int k = 0; k < n_variants; ++k) {
                for (int p = k + 1; p < n_variants; ++p) {
                    int_comparisons += pos[k].size() + pos[p].size();
                }
            }
            const uint64_t n_intersects = ((n_variants * n_variants) - n_variants) / 2;
            std::cerr << "Size of intersections=" << int_comparisons << std::endl;

            const uint64_t n_total_integer_cmps = n_intersects * n_ints_sample;
            std::cerr << "Total integer comparisons=" << n_total_integer_cmps << std::endl;
            //

            // Debug
            std::chrono::high_resolution_clock::time_point t1_blocked = std::chrono::high_resolution_clock::now();
            // uint64_t d = 0, diag = 0;
            {
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                const uint64_t cycles_start = get_cpu_cycles();
                uint64_t cont_count = TWK_cont_intersect_cardinality(twk2);
                const uint64_t cycles_end = get_cpu_cycles();

                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

                bench_t b; b.count = cont_count; b.milliseconds = time_span.count();
                uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
                b.throughput = ((n_comps*n_ints_sample*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
                b.cpu_cycles = cycles_end - cycles_start;
                // std::cerr << "[cnt] count=" << cont_count << std::endl;
                PRINT("storm",b);
            }

            {
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                const uint64_t cycles_start = get_cpu_cycles();
                uint64_t cont_count = bcont.intersect();
                const uint64_t cycles_end = get_cpu_cycles();

                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

                bench_t b; b.count = cont_count; b.milliseconds = time_span.count();
                uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
                b.throughput = ((n_comps*n_ints_sample*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
                b.cpu_cycles = cycles_end - cycles_start;
                // std::cerr << "[cnt] count=" << cont_count << std::endl;
                PRINT("test-opt",b);
            }

            {
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                const uint64_t cycles_start = get_cpu_cycles();
                uint64_t cont_count = bcont.intersect_blocked(TWK_CACHE_BLOCK_SIZE/(n_ints_sample*8));
                const uint64_t cycles_end = get_cpu_cycles();

                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

                bench_t b; b.count = cont_count; b.milliseconds = time_span.count();
                uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
                b.throughput = ((n_comps*n_ints_sample*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
                b.cpu_cycles = cycles_end - cycles_start;
                // std::cerr << "[cnt] count=" << cont_count << std::endl;
                PRINT("test-opt-blocked-" + std::to_string(TWK_CACHE_BLOCK_SIZE/(n_ints_sample*8)),b);
            }

            {
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                const uint64_t cycles_start = get_cpu_cycles();
                uint64_t cont_count = bcont2.intersect_cont();
                const uint64_t cycles_end = get_cpu_cycles();

                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

                bench_t b; b.count = cont_count; b.milliseconds = time_span.count();
                uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
                b.throughput = ((n_comps*n_ints_sample*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
                b.cpu_cycles = cycles_end - cycles_start;
                // std::cerr << "[cnt] count=" << cont_count << std::endl;
                PRINT("test-opt-cont-only",b);
            }

            {
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                const uint64_t cycles_start = get_cpu_cycles();
                uint64_t cont_count = bcont2.intersect_blocked_cont(TWK_CACHE_BLOCK_SIZE/(n_ints_sample*8));
                const uint64_t cycles_end = get_cpu_cycles();

                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

                bench_t b; b.count = cont_count; b.milliseconds = time_span.count();
                uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
                b.throughput = ((n_comps*n_ints_sample*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
                b.cpu_cycles = cycles_end - cycles_start;
                // std::cerr << "[cnt] count=" << cont_count << std::endl;
                PRINT("test-opt-cont-blocked-" + std::to_string(TWK_CACHE_BLOCK_SIZE/(n_ints_sample*8)),b);
            }

            // {
            //     std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            //     const uint64_t cycles_start = get_cpu_cycles();
            //     uint64_t cont_count = bcont2.intersect_cont_auto();
            //     const uint64_t cycles_end = get_cpu_cycles();

            //     std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            //     auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

            //     bench_t b; b.count = cont_count; b.milliseconds = time_span.count();
            //     uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
            //     b.throughput = ((n_comps*n_ints_sample*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
            //     b.cpu_cycles = cycles_end - cycles_start;
            //     // std::cerr << "[cnt] count=" << cont_count << std::endl;
            //     PRINT("test-avx2-cont-auto",b);
            // }

            // {
            //     std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            //     const uint64_t cycles_start = get_cpu_cycles();
            //     uint64_t cont_count = bcont2.intersect_cont_blocked_auto(TWK_CACHE_BLOCK_SIZE/(n_ints_sample*8));
            //     const uint64_t cycles_end = get_cpu_cycles();

            //     std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            //     auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

            //     bench_t b; b.count = cont_count; b.milliseconds = time_span.count();
            //     uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
            //     b.throughput = ((n_comps*n_ints_sample*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
            //     b.cpu_cycles = cycles_end - cycles_start;
            //     // std::cerr << "[cnt] count=" << cont_count << std::endl;
            //     PRINT("test-avx2-cont-auto-blocked-" + std::to_string(TWK_CACHE_BLOCK_SIZE/(n_ints_sample*8)),b);
            // }

            {
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                const uint64_t cycles_start = get_cpu_cycles();
                uint64_t cont_count = bcont2.intersect_cont_auto();
                const uint64_t cycles_end = get_cpu_cycles();

                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

                bench_t b; b.count = cont_count; b.milliseconds = time_span.count();
                uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
                b.throughput = ((n_comps*n_ints_sample*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
                b.cpu_cycles = cycles_end - cycles_start;
                // std::cerr << "[cnt] count=" << cont_count << std::endl;
                PRINT("automatic",b);
            }

            // std::vector<uint32_t> o = {10, 50, 100, 250, 500};

            // for (int z = 0; z < 5; ++z) {
            {
                uint32_t cutoff = ceil(n_ints_sample*64 / 200.0);
                std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
                const uint64_t cycles_start = get_cpu_cycles();
                uint64_t cont_count = bcont2.intersect_cont_blocked_auto(TWK_CACHE_BLOCK_SIZE/(n_ints_sample*8));
                const uint64_t cycles_end = get_cpu_cycles();

                std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
                auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

                bench_t b; b.count = cont_count; b.milliseconds = time_span.count();
                uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
                b.throughput = ((n_comps*n_ints_sample*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);
                b.cpu_cycles = cycles_end - cycles_start;
                // std::cerr << "[cnt] count=" << cont_count << std::endl;
                PRINT("automatic-list-" + std::to_string(cutoff),b);
            }
            // }

            const std::vector<uint32_t> block_range = {3,5,10,25,50,100,200,400,600,800, 32e3/(n_ints_sample*8) }; // last one is auto

            uint32_t optimal_b = TWK_CACHE_BLOCK_SIZE/(n_ints_sample*8);

#if SIMD_VERSION >= 6
            // SIMD AVX512
            // for (int k = 0; k < block_range.size(); ++k) {
            //     bench_t m8_2_block = fwrapper_blocked<&intersect_bitmaps_avx512_csa>(n_variants, vals, n_ints_sample,block_range[k]);
            //     PRINT("bitmap-avx512-csa-blocked-" + std::to_string(block_range[k]),m8_2_block);
            // }

            // bench_t m8_2 = fwrapper<&intersect_bitmaps_avx512_csa>(n_variants, vals, n_ints_sample);
            // PRINT("bitmap-avx512-csa",m8_2);

            bench_t m8_avx512_block = fwrapper_blocked<&TWK_intersect_avx512>(n_variants, vals, n_ints_sample, optimal_b);
            PRINT("bitmap-avx512-csa-blocked-" + std::to_string(optimal_b), m8_avx512_block );
#endif

#ifdef USE_ROARING
            // for (int k = 0; k < block_range.size(); ++k) {
            //     bench_t m8_2_block = froarwrapper_blocked(n_variants, n_ints_sample, roaring, block_range[k]);
            //     PRINT("roaring-blocked-" + std::to_string(block_range[k]),m8_2_block);
            // }

            // bench_t broaring = froarwrapper(n_variants, n_ints_sample, roaring);
            // PRINT("roaring",broaring);

            uint64_t roaring_bytes_used = 0;
            for (int k = 0; k < n_variants; ++k) {
                roaring_bytes_used += roaring_bitmap_portable_size_in_bytes(roaring[k]);
            }
            std::cerr << "Memory used by roaring=" << roaring_bytes_used << "(" << (float)memory_used/roaring_bytes_used << ")" << std::endl;

            uint32_t roaring_optimal_b = TWK_CACHE_BLOCK_SIZE / (roaring_bytes_used / n_variants);

            bench_t m8_2_block = froarwrapper_blocked(n_variants, n_ints_sample, roaring, roaring_optimal_b);
            PRINT("roaring-blocked-" + std::to_string(roaring_optimal_b),m8_2_block);
#endif

#if SIMD_VERSION >= 5
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

            bench_t m3_block3 = fwrapper_blocked<&TWK_intersect_avx2>(n_variants, vals, n_ints_sample, optimal_b);
            PRINT("bitmap-avx256-blocked-" + std::to_string(optimal_b), m3_block3);
#endif
            // SIMD SSE4
#if SIMD_VERSION >= 3
            // for (int k = 0; k < block_range.size(); ++k) {
            //     bench_t m2_block3 = fwrapper_blocked<&intersect_bitmaps_sse4>(n_variants, vals, n_ints_sample,block_range[k]);
            //     PRINT("bitmap-sse4-csa-blocked-" + std::to_string(block_range[k]),m2_block3);
            // }
            // bench_t m2 = fwrapper<&intersect_bitmaps_sse4>(n_variants, vals, n_ints_sample);
            // PRINT("bitmap-sse4-csa",m2);

            bench_t m2_block3 = fwrapper_blocked<&TWK_intersect_sse4>(n_variants, vals, n_ints_sample, optimal_b);
            PRINT("bitmap-sse4-csa-blocked-" + std::to_string(optimal_b), m2_block3);
#endif

            if (n_alts[a] <= 300) {
                bench_t m4 = flwrapper<&TWK_intersect_scalar_list>(n_variants, vals, n_ints_sample, pos);
                PRINT("bitmap-scalar-skip-list",m4);

                /*
                bench_t m1 = fwrapper<&intersect_bitmaps_scalar>(n_variants, vals, n_ints_sample);
                PRINT("bitmap-scalar-popcnt",m1);

                bench_t raw1 = frawwrapper<&intersect_raw_naive>(n_variants, n_ints_sample, pos16);
                PRINT("raw-naive",raw1);

                bench_t raw2 = frawwrapper<&intersect_raw_sse4_broadcast>(n_variants, n_ints_sample, pos16);
                PRINT("raw-broadcast-sse4",raw2);

                bench_t raw3 = frawwrapper<&intersect_raw_avx2_broadcast>(n_variants, n_ints_sample, pos16);
                PRINT("raw-broadcast-avx2",raw3);

                bench_t raw2_skip = frawwrapper<&intersect_raw_sse4_broadcast_skip>(n_variants, n_ints_sample, pos16);
                PRINT("raw-broadcast-sse4-skip",raw2_skip);

                bench_t raw_gallop = frawwrapper<&intersect_raw_gallop>(n_variants, n_ints_sample, pos16);
                PRINT("raw-gallop",raw_gallop);

                bench_t raw_gallop_sse = frawwrapper<&intersect_raw_gallop_sse4>(n_variants, n_ints_sample, pos16);
                PRINT("raw-gallop-sse4",raw_gallop_sse);

                bench_t raw_binary = frawwrapper<&intersect_raw_binary>(n_variants, n_ints_sample, pos16);
                PRINT("raw-binary",raw_binary);

                bench_t raw_roaring = frawwrapper<&intersect_roaring_cardinality>(n_variants, n_ints_sample, pos16);
                PRINT("raw-roaring",raw_roaring);

                bench_t raw_roaring2 = frawwrapper<&intersect_vector16_cardinality_roar>(n_variants, n_ints_sample, pos16);
                PRINT("raw-roaring2",raw_roaring2);

                

                bench_t raw1_roaring_sse4 = frawwrapper<&intersect_raw_rotl_gallop_sse4>(n_variants, n_ints_sample, pos16);
                PRINT("raw-rotl-gallop-sse4",raw1_roaring_sse4);

                bench_t raw1_roaring_avx2= frawwrapper<&intersect_raw_rotl_gallop_avx2>(n_variants, n_ints_sample, pos16);
                PRINT("raw-rotl-gallop-avx2",raw1_roaring_avx2);
                */

                /*
                std::vector< std::vector<uint32_t> > rle_32(n_variants, std::vector<uint32_t>());
                std::vector< std::vector<uint64_t> > rle_64(n_variants, std::vector<uint64_t>());

                uint32_t offset = 0;
                for (int i = 0; i < n_variants; ++i) {
                    rle_32[i] = construct_rle<uint32_t>(&vals[offset], n_ints_sample);
                    rle_64[i] = construct_rle<uint64_t>(&vals[offset], n_ints_sample);
                    offset += n_ints_sample;
                }

                bench_t mrle32 = frlewrapper< uint32_t, &intersect_rle<uint32_t> >(rle_32, n_ints_sample);
                //std::cout << n_samples << "\t" << n_alts[a] << "\trle-32\t" << mrle32.milliseconds << "\t" << mrle32.count << "\t" << mrle32.throughput << std::endl;
                PRINT("rle-32",mrle32);

                bench_t mrle32_b = frlewrapper< uint32_t, &intersect_rle_branchless<uint32_t> >(rle_32, n_ints_sample);
                //std::cout << n_samples << "\t" << n_alts[a] << "\trle-32-branchless\t" << mrle32_b.milliseconds << "\t" << mrle32_b.count << "\t" << mrle32_b.throughput << std::endl;
                PRINT("rle-32-branchless",mrle32_b);

                bench_t mrle64 = frlewrapper< uint64_t, &intersect_rle<uint64_t> >(rle_64, n_ints_sample);
                //std::cout << n_samples << "\t" << n_alts[a] << "\trle-64\t" << mrle64.milliseconds << "\t" << mrle64.count << "\t" << mrle64.throughput << std::endl;
                PRINT("rle-64",mrle64);

                bench_t mrle64_b = frlewrapper< uint64_t, &intersect_rle_branchless<uint64_t> >(rle_64, n_ints_sample);
                //std::cout << n_samples << "\t" << n_alts[a] << "\trle-64-branchless\t" << mrle64_b.milliseconds << "\t" << mrle64_b.count << "\t" << mrle64_b.throughput << std::endl;
                PRINT("rle-64-branchless",mrle64_b);

                rle_32.clear(); rle_64.clear();
                */

            } 
        
#ifdef USE_ROARING
            for (int i = 0; i < n_variants; ++i) roaring_bitmap_free(roaring[i]);
            delete[] roaring;
#endif
        }
        // for (int i = 0; i < n_variants; ++i) TWK_bitmap_cont_free(twk[i]);
        // delete[] twk;
        TWK_cont_free(twk2);
        TWK_aligned_free(vals);
    // }
}

const char* usage(void) {
    return
        "\n"
        "About:   Computes sum(popcnt(A & B)) for the all-vs-all comparison of N integer\n"
        "         lists bounded by [0, M). This benchmark will compute this statistics using\n"
        "         different algorithms.\n"
        "Usage:   benchmark <M> <N> \n"
        "\n"
        "Example:\n"
        "   benchmark 4092 10000 \n"
        "\n";
}

int main(int argc, char **argv) { 
    if (argc == 1) {
        printf("%s",usage());
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        intersect_test(std::atoi(argv[1]), 10000);
    } else {
        intersect_test(std::atoi(argv[1]), std::atoi(argv[2]));
    }
    return EXIT_SUCCESS;
}
