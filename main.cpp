#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <memory>
#include <bitset>

#include "classes.h"

#define USE_ROARING

#ifdef USE_ROARING
#include <roaring/roaring.h>
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

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, uint64_t* buffer)>
bench_t fwrapper_buffered(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;
    //uint64_t* buffer = new uint64_t[16];
    uint64_t* buffer;
    assert(!posix_memalign((void**)&buffer, SIMD_ALIGNMENT, 16));

    const uint64_t cycles_start = get_cpu_cycles();
    for (int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
        for (int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
            total += (*f)(&vals[offset], &vals[inner_offset], n_ints, buffer);
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

    delete[] buffer;
    return(b);
}

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const std::pair<uint32_t,uint32_t>& p1, const std::pair<uint32_t,uint32_t>& p2)>
bench_t fpswrapper(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, const std::vector< std::pair<uint32_t,uint32_t> >& pairs) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;

    const uint64_t cycles_start = get_cpu_cycles();
    for (int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
        for (int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
            total += (*f)(&vals[offset], &vals[inner_offset], n_ints, pairs[i], pairs[j]);
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
bench_t flwrapper(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, const std::vector< std::vector<uint32_t> >& pos) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;

    const uint64_t cycles_start = get_cpu_cycles();
    for (int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
       for (int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
           total += (*f)(&vals[offset], &vals[inner_offset], pos[i], pos[j]);
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

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2)>
bench_t fsqwrapper(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, const std::vector< std::vector<uint64_t> >& squash) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;
    const uint32_t n_squash = squash[0].size();

    const uint64_t cycles_start = get_cpu_cycles();
    for (int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
        for (int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
            total += (*f)(&vals[offset], &vals[inner_offset], n_ints, n_squash, squash[i], squash[j]);
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

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2)>
bench_t flsqwrapper(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, const std::vector< std::vector<uint32_t> >& pos, const std::vector< std::vector<uint64_t> >& squash) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;
    const uint32_t n_squash = squash[0].size();

    const uint64_t cycles_start = get_cpu_cycles();
    for (int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
        for (int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
            total += (*f)(&vals[offset], &vals[inner_offset], pos[i], pos[j], n_squash, squash[i], squash[j]);
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

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint16_t>& l1, const std::vector<uint16_t>& l2)>
bench_t fredwrapper(const uint32_t n_variants, const uint32_t n_vals_actual, const uint64_t* vals, const std::vector< std::vector<uint16_t> >& pos16) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;
    uint32_t l_offset = 0;
    uint32_t l_offset_inner = 0;

    const uint64_t cycles_start = get_cpu_cycles();
    for (int k = 0; k < n_variants; ++k) {
        l_offset_inner = l_offset + pos16[k].size();
        for (int p = k + 1; p < n_variants; ++p) {
            total += (*f)(&vals[l_offset], &vals[l_offset_inner], pos16[k], pos16[p]);
            l_offset_inner += pos16[p].size();
        }
        l_offset += pos16[k].size();
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

template <uint64_t (f)(const range_bin& b1, const range_bin& b2, const uint8_t n_ints_bin)>
bench_t frbinswrapper(const uint32_t n_variants, const uint32_t n_vals_actual, const std::vector< range_bin >& bins, const uint8_t n_ints_bin) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;

    const uint64_t cycles_start = get_cpu_cycles();
    for (int k = 0; k < n_variants; ++k) {
        for (int p = k + 1; p < n_variants; ++p) {
            total += (*f)(bins[k], bins[p], n_ints_bin);
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

template <uint64_t (f)(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2)>
bench_t frawwrapper(const uint32_t n_variants, const uint32_t n_vals_actual, const std::vector< std::vector<uint16_t> >& pos) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;

    const uint64_t cycles_start = get_cpu_cycles();
    for (int k = 0; k < n_variants; ++k) {
        for (int p = k + 1; p < n_variants; ++p) {
            total += (*f)(pos[k], pos[p]);
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
#endif

void intersect_test(uint32_t n, uint32_t cycles = 1) {
    // Setup
    std::vector<uint32_t> samples = {4864, 65536, 131072, 196608, 589824};
    // std::vector<uint32_t> samples = {131072, 196608, 589824};
    for (int s = 0; s < samples.size(); ++s) {
        uint32_t n_ints_sample = samples[s] / 64;

        // Limit memory usage to 10e6 but no more than 50k records.
        uint32_t desired_mem = 20 * 1024 * 1024;
        // b_total / (b/obj) = n_ints
        uint32_t n_variants = std::max(std::min((uint32_t)50000, (uint32_t)std::ceil(desired_mem/(n_ints_sample*sizeof(uint64_t)))), (uint32_t)64);
        // uint32_t n_variants = 10000;

        std::cerr << "Generating: " << samples[s] << " samples for " << n_variants << " variants" << std::endl;
        std::cerr << "Allocating: " << n_ints_sample*n_variants*sizeof(uint64_t)/(1024 * 1024.0) << "Mb" << std::endl;
        uint64_t* vals;
        assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, n_ints_sample*n_variants*sizeof(uint64_t)));
        uint64_t* vals_reduced;
        assert(!posix_memalign((void**)&vals_reduced, SIMD_ALIGNMENT, n_ints_sample*n_variants*sizeof(uint64_t)));

        // 1:500, 1:167, 1:22
        std::vector<uint32_t> n_alts = {2, 15, 30, 65, 222}; // 1kgp3 dist 
        // std::vector<uint32_t> n_alts = {21,269,9506}; // HRC dist

        // std::vector<uint32_t> n_alts = {samples[s]/1000, samples[s]/500, samples[s]/100, samples[s]/20}; //, samples[s]/10, samples[s]/4, samples[s]/2};
        // std::vector<uint32_t> n_alts = {samples[s]/100, samples[s]/20, samples[s]/10, samples[s]/4, samples[s]/2};
        // std::vector<uint32_t> n_alts = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096};
        //std::vector<uint32_t> n_alts = {512,1024,2048,4096};

        for (int a = 0; a < n_alts.size(); ++a) {
#ifdef USE_ROARING
            roaring_bitmap_t** roaring = new roaring_bitmap_t*[n_variants];
            for (int i = 0; i < n_variants; ++i) roaring[i] = roaring_bitmap_create();
#endif

            if (n_alts[a] == 0) continue;
            // Allocation
            memset(vals, 0, n_ints_sample*n_variants*sizeof(uint64_t));

            // PRNG
            std::uniform_int_distribution<uint32_t> distr(0, samples[s]-1); // right inclusive

            // Positional information
            uint32_t n_squash4096 = std::min((uint32_t)std::ceil((double)samples[s]/4096), (uint32_t)4);
            uint32_t divisor = samples[s]/n_squash4096;
            if (n_squash4096 == 1) divisor = samples[s];
            std::cerr << "nsq=" << n_squash4096 << " div=" << divisor << std::endl;
            std::vector< std::vector<uint32_t> > pos(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint16_t> > pos16(n_variants, std::vector<uint16_t>());
            std::vector< std::vector<uint32_t> > pos_integer(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint16_t> > pos_integer16(n_variants, std::vector<uint16_t>());
            std::vector< std::vector<uint32_t> > pos_reg128(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_reg256(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_reg512(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint64_t> > squash_4096(n_variants, std::vector<uint64_t>(n_squash4096, 0));

            std::vector< std::pair<uint32_t, uint32_t> > prefix_suffix_pos(n_variants, std::pair<uint32_t, uint32_t>(0,0));

            // const uint8_t n_ints_bin = std::min(n_ints_sample, (uint32_t)(4*SIMD_WIDTH/64));
            const uint8_t n_ints_bin = std::min(n_ints_sample, (uint32_t)8);
            const uint32_t bin_size = std::ceil(n_ints_sample / (float)n_ints_bin);
            std::cerr << "bin-size=" << bin_size << std::endl;
            std::vector< range_bin > bins(n_variants, bin_size);
            std::vector< range_bin > bins_bit(n_variants, bin_size);

            std::random_device rd;  // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator

            uint32_t n_vals_reduced = 0;

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

                for (int p = 0; p < pos[j].size(); ++p)
                    pos16[j].push_back(pos[j][p]);

                // Assertion of sortedness.
                for (int p = 1; p < pos[j].size(); ++p) {
                    assert(pos[j][p-1] < pos[j][p]);
                }

                //for (int p = 0; p < pos.back().size(); ++p) std::cerr << " " << pos.back()[p];
                //std::cerr << std::endl;

#ifdef USE_ROARING
                for (int p = 0; p < pos[j].size(); ++p)
                    roaring_bitmap_add(roaring[j], pos[j][p]);
#endif

                // Todo
                // Collapse positions into integers
               // pos_integer.push_back(std::vector<uint32_t>());
                pos_integer[j].push_back(pos[j][0] / 64);

                for (int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 64;
                    if (pos_integer[j].back() != idx) pos_integer[j].push_back(idx);
                }

                //std::cerr << "0->" << pos_integer[j].front() << " " << pos_integer[j].back()+1 << "<-" << n_ints_sample << std::endl;
                prefix_suffix_pos[j].first = pos_integer[j].front();
                prefix_suffix_pos[j].second = pos_integer[j].back()+1;

                //std::cerr << "bin=" << bin_size << std::endl;
                bins[j].n_ints     = pos[j].size();
                bins_bit[j].n_ints = pos[j].size();
                std::vector< std::vector<uint16_t> > vv(bin_size); // integers
                std::vector< std::vector<uint16_t> > vv2(bin_size); // bits
                for (int p = 0; p < pos[j].size(); ++p) {
                    const uint32_t target_bin = pos[j][p] / 64 / n_ints_bin;
                    const uint32_t FOR = (target_bin*64*n_ints_bin); // frame of reference value
                    const uint32_t local_val = (pos[j][p] - FOR);
                    const uint32_t local_int = local_val / 64;

                    //std::cerr << " " << pos[j][p] << ":" << target_bin << " FOR=" << FOR << "->" << local_val << "F=" << local_int << "|" << local_val % 64;

                    // Allocate memory in target bins
                    if (bins[j].bins[target_bin].n_vals == 0) bins[j].bins[target_bin].Allocate(n_ints_bin);
                    if (bins_bit[j].bins[target_bin].n_vals == 0) bins_bit[j].bins[target_bin].Allocate(n_ints_bin);

                    // Add integers
                    if (vv[target_bin].size() == 0) vv[target_bin].push_back(local_int);
                    else if (vv[target_bin].back() != (local_int)) vv[target_bin].push_back(local_int);

                    // Add bits
                    if (vv2[target_bin].size() == 0) vv2[target_bin].push_back(local_val);
                    else if (vv2[target_bin].back() != (local_val)) vv2[target_bin].push_back(local_val);

                    // Add data to bins and bitmap
                    bins[j].bins[target_bin].vals[local_int] |= (1L << (local_val % 64));
                    bins[j].bins[target_bin].bitmap |= (1L << local_int);
                    bins[j].bin_bitmap |= (1L << target_bin);

                    bins_bit[j].bins[target_bin].vals[local_int] |= (1L << (local_val % 64));
                    bins_bit[j].bins[target_bin].bitmap |= (1L << local_int);
                    bins_bit[j].bin_bitmap |= (1L << target_bin);
                }
                //std::cerr << std::endl;

                // for integer
                for (int p = 0; p < vv.size(); ++p) {
                    if (vv[p].size() < 3 && vv[p].size() != 0) {
                        //std::cerr << "setting internal pos" << std::endl;
                        bins[j].bins[p].list = true;
                        bins[j].bins[p].pos = std::make_shared< std::vector<uint16_t> >(vv[p]);
                        bins[j].bins[p].n_list = vv[p].size();
                    }
                }

                std::vector< uint16_t > v;
                for (int p = 0; p < bins[j].bins.size(); ++p) {
                    if (bins[j].bins[p].n_vals) v.push_back(p);
                }

                if (v.size() / (float)bin_size < 0.5) {
                    //std::cerr << v.size() << "/" << bin_size << "->" << (v.size() / (float)bin_size) << std::endl;
                    bins[j].list = true;
                    bins[j].pos = std::make_shared< std::vector<uint16_t> >(v);
                    bins[j].n_list = v.size();
                }

                // for bits
                for (int p = 0; p < vv2.size(); ++p) {
                    if (vv2[p].size() < 10 && vv2[p].size() != 0) {
                        //std::cerr << "setting internal pos" << std::endl;
                        bins_bit[j].bins[p].list = true;
                        bins_bit[j].bins[p].pos = std::make_shared< std::vector<uint16_t> >(vv2[p]);
                        bins_bit[j].bins[p].n_list = vv2[p].size();
                    }
                }

                v.clear();
                for (int p = 0; p < bins_bit[j].bins.size(); ++p) {
                    if (bins_bit[j].bins[p].n_vals) v.push_back(p);
                }

                if (v.size() / (float)bin_size < 0.5) {
                    //std::cerr << v.size() << "/" << bin_size << "->" << (v.size() / (float)bin_size) << std::endl;
                    bins_bit[j].list = true;
                    bins_bit[j].pos = std::make_shared< std::vector<uint16_t> >(v);
                    bins_bit[j].n_list = v.size();
                }

                //

                pos_integer16[j].push_back(pos[j][0] / 64);

                for (int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 64;
                    if (pos_integer16[j].back() != idx) pos_integer16[j].push_back(idx);
                }

                //
                for (int p = 0; p < pos_integer16[j].size(); ++p) {
                    //std::cerr << vals2[pos_integer[j][p]] << std::endl;
                    vals_reduced[n_vals_reduced++] = vals2[pos_integer16[j][p]];
                }

                //for (int p = 0; p < pos_integer.back().size(); ++p) std::cerr << " " << pos_integer.back()[p];
                //std::cerr << std::endl;

                // Todo
                // Collapse positions into registers
                //pos_reg128.push_back(std::vector<uint32_t>());
                pos_reg128[j].push_back(pos[j][0] / 128);

                for (int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 128;
                    if (pos_reg128[j].back() != idx) pos_reg128[j].push_back(idx);
                }

                // Todo
                // Collapse positions into 256-registers
                //pos_reg256.push_back(std::vector<uint32_t>());
                pos_reg256[j].push_back(pos[j][0] / 256);

                for (int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 256;
                    if (pos_reg256[j].back() != idx) pos_reg256[j].push_back(idx);
                }

                // Todo
                // Collapse positions into 512-registers
                //pos_reg512.push_back(std::vector<uint32_t>());
                pos_reg512[j].push_back(pos[j][0] / 512);

                for (int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 512;
                    if (pos_reg512[j].back() != idx) pos_reg512[j].push_back(idx);
                }

                // Todo
                // Squash into 4096 bins
                for (int p = 0; p < pos[j].size(); ++p) {
                    //std::cerr << (pos[j][p] / 4096) << "/" << n_squash4096 << "/" << squash_4096[j].size() << std::endl;
                    squash_4096[j][pos[j][p] / divisor] |= 1L << (pos[j][p] % divisor);
                }
                //for (int p = 0; p < squash_4096[j].size(); ++p) std::cerr << " " << std::bitset<64>(squash_4096[j][p]);
                //std::cerr << std::endl;

                // Todo print averages
                //std::cerr << pos.back().size() << "->" << pos_integer.back().size() << "->" << pos_reg128.back().size() << std::endl;
                vals2 += n_ints_sample;
            }
            std::cerr << "Done!" << std::endl;

            //
            assert(n_variants == pos.size());
            std::vector<IntersectContainer> itcontainers(pos.size());
            for (int i = 0; i < pos.size(); ++i) {
                itcontainers[i].construct(&pos[i][0], pos[i].size(), samples[s]);
            }
            std::cerr << "Done containers!" << std::endl;

            std::chrono::high_resolution_clock::time_point t1_debug = std::chrono::high_resolution_clock::now();
            const uint64_t cycles_start_debug = get_cpu_cycles();

            uint64_t con_tot = 0, con_tot_cycles = 0;
            for (int k = 0; k < n_variants; ++k) {
                for (int p = k + 1; p < n_variants; ++p, ++con_tot_cycles) {
                    con_tot += itcontainers[k].IntersectCount(itcontainers[p]);
                }
            }


            const uint64_t cycles_end_debug = get_cpu_cycles();

            std::chrono::high_resolution_clock::time_point t2_debug = std::chrono::high_resolution_clock::now();
            auto time_span_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t2_debug - t1_debug);

            std::cerr << "container time=" << time_span_debug.count() << std::endl;

            std::cerr << "con_tot=" << con_tot << std::endl;
            std::cerr << "cycles=" << con_tot_cycles << std::endl;

            //

            uint64_t tot_reg256 = 0;
            for (int i = 0; i < pos_reg256.size(); ++i) {
                tot_reg256 += pos_reg256[i].size();
            }
            std::cerr << "[VECTORS AVX] average tot_256=" << (double)tot_reg256 / pos_reg256.size() << "/" << samples[s] / 256 << std::endl;

            std::cerr << "n_reduced=" << n_vals_reduced << std::endl;

            uint32_t mem_bins = 0;
            for (int i = 0; i < bins.size(); ++i) {
                mem_bins += sizeof(uint64_t);
                for (int j = 0; j < bins[i].bins.size(); ++j) {
                    mem_bins += bins[i].bins[j].n_vals * sizeof(uint64_t);
                }
                mem_bins += sizeof(uint8_t);
            }
            std::cerr << "mem_bins=" << mem_bins << " (" << (n_ints_sample*n_variants*sizeof(uint64_t)) / (double)mem_bins << ")" << std::endl;

            //uint32_t offset = 0;
            /*for (int i = 0; i < n_variants; ++i) {
                construct_ewah64(&vals[offset], n_ints_sample);
                offset += n_ints_sample;
            }
            */

            uint64_t int_comparisons = 0;
            for (int k = 0; k < n_variants; ++k) {
                for (int p = k + 1; p < n_variants; ++p) {
                    int_comparisons += pos[k].size() + pos[p].size();
                }
            }
            const uint64_t n_intersects = ((n_variants * n_variants) - n_variants) / 2;
            std::cerr << "Size of intersections=" << int_comparisons << std::endl;

        //

        // Cache blocking
		// uint32_t bsize = (256e3/2) / (samples[s]/8);
        uint32_t bsize = 50;
		bsize = (bsize == 0 ? 10 : bsize);
		const uint32_t n_blocks1 = n_variants / bsize;
        const uint32_t n_blocks2 = n_variants / bsize;
        std::cerr << "blocking size=" << bsize << std::endl;


        // Debug
        std::chrono::high_resolution_clock::time_point t1_blocked = std::chrono::high_resolution_clock::now();
        // uint64_t d = 0, diag = 0;

        uint64_t blocked_con_tot = 0;
        uint32_t i  = 0;
        uint32_t tt = 0;
        for (/**/; i + bsize <= n_variants; i += bsize) {
            // diagonal component
            for (uint32_t j = 0; j < bsize; ++j) {
                for (uint32_t jj = j + 1; jj < bsize; ++jj) {
                    // blocked_con_tot += itcontainers[i + j].IntersectCount(itcontainers[i + jj]);
                    blocked_con_tot += roaring_bitmap_and_cardinality(roaring[i+j], roaring[i+jj]);
                    // total += (*f)(&vals[left], &vals[right], pos[i+j], pos[i+jj]);
                    //total += (*f)(&vals[left], &vals[right], n_ints);
                    // ++d;
                }
            }

            // square component
            uint32_t curi = i;
            uint32_t j = curi + bsize;
            for (/**/; j + bsize <= n_variants; j += bsize) {
                for (uint32_t ii = 0; ii < bsize; ++ii) {
                    for (uint32_t jj = 0; jj < bsize; ++jj) {
                        // total += (*f)(&vals[left], &vals[right], n_ints);
                        // total += (*f)(&vals[left], &vals[right], pos[curi + ii], pos[j + jj]);
                        // blocked_con_tot += itcontainers[curi + ii].IntersectCount(itcontainers[j + jj]);
                        blocked_con_tot += roaring_bitmap_and_cardinality(roaring[curi+ii], roaring[j+jj]);
                        // ++d;
                    }
                }
            }

            // residual
            for (/**/; j < n_variants; ++j) {
                for (uint32_t jj = 0; jj < bsize; ++jj) {
                    // total += (*f)(&vals[left], &vals[right], n_ints);
                    // total += (*f)(&vals[left], &vals[right], pos[curi + jj], pos[j]);
                    // blocked_con_tot += itcontainers[curi + jj].IntersectCount(itcontainers[j]);
                    blocked_con_tot += roaring_bitmap_and_cardinality(roaring[curi+jj], roaring[j]);
                    // ++d;
                }
            }
        }
        // residual tail
        for (/**/; i < n_variants; ++i) {
            for (uint32_t j = i + 1; j < n_variants; ++j) {
                // total += (*f)(&vals[left], &vals[right], n_ints);
                // total += (*f)(&vals[left], &vals[right], pos[i], pos[j]);
                // blocked_con_tot += itcontainers[i].IntersectCount(itcontainers[j]);
                blocked_con_tot += roaring_bitmap_and_cardinality(roaring[i], roaring[j]);
                // ++d;
            }
        }
        std::chrono::high_resolution_clock::time_point t2_blocked = std::chrono::high_resolution_clock::now();
        auto time_span_blocked = std::chrono::duration_cast<std::chrono::milliseconds>(t2_blocked - t1_blocked);

        // std::cerr << "BLOCKIING=" << d << "/" << n_intersects << " with diag " << diag << std::endl;
        // assert(d == n_intersects);
        std::cerr << "blocked tot=" << blocked_con_tot << " time=" << time_span_blocked.count() << std::endl;
        //

            std::cerr << "Samples\tAlts\tMethod\tTime(ms)\tCount\tThroughput(MB/s)\tInts/s(1e6)\tIntersect/s(1e6)\tActualThroughput(MB/s)\tCycles/int\tCycles/intersect" << std::endl;
#define PRINT(name,bench) std::cout << samples[s] << "\t" << n_alts[a] << "\t" << name << "\t" << bench.milliseconds << "\t" << bench.count << "\t" << \
        bench.throughput << "\t" << \
        (bench.milliseconds == 0 ? 0 : (int_comparisons*1000.0 / bench.milliseconds / 1000000.0)) << "\t" << \
        (n_intersects*1000.0 / (bench.milliseconds) / 1000000.0) << "\t" << \
        (bench.milliseconds == 0 ? 0 : int_comparisons*sizeof(uint32_t) / (bench.milliseconds/1000.0) / (1024.0*1024.0)) << "\t" << \
        (bench.cpu_cycles == 0 ? 0 : bench.cpu_cycles / (double)int_comparisons) << "\t" << \
        (bench.cpu_cycles == 0 ? 0 : bench.cpu_cycles / (double)n_intersects) << std::endl


#ifdef USE_ROARING
            // temp
            uint64_t roaring_bytes_used = 0;
            for (int k = 0; k < n_variants; ++k) {
                roaring_bytes_used += roaring_bitmap_portable_size_in_bytes(roaring[k]);
            }
            std::cerr << "Memory used by roaring=" << roaring_bytes_used << std::endl;

            bench_t broaring = froarwrapper(n_variants, n_ints_sample, roaring);
            PRINT("roaring",broaring);
#endif

#if SIMD_VERSION >= 3
            // SIMD SSE4

            std::vector<uint32_t> block_range = {3,5,10,25,50,100,200,400,600,800, 32e3/(n_ints_sample*8) }; // last one is auto

            for (int k = 0; k < block_range.size(); ++k) {
               bench_t m2_block3 = fwrapper_blocked<&intersect_bitmaps_sse4>(n_variants, vals, n_ints_sample,block_range[k]);
                PRINT("bitmap-sse4-blocked-" + std::to_string(block_range[k]),m2_block3);
            }
            bench_t m2 = fwrapper<&intersect_bitmaps_sse4>(n_variants, vals, n_ints_sample);
            PRINT("bitmap-sse4",m2);
#endif
#if SIMD_VERSION >= 4
            // SIMD AVX256
            for (int k = 0; k < block_range.size(); ++k) {
                bench_t m3_block3 = fwrapper_blocked<&intersect_bitmaps_avx2>(n_variants, vals, n_ints_sample,block_range[k]);
                PRINT("bitmap-avx256-blocked-" + std::to_string(block_range[k]),m3_block3);
            }
            bench_t m3_0 = fwrapper<&intersect_bitmaps_avx2>(n_variants, vals, n_ints_sample);
            PRINT("bitmap-avx256",m3_0);

            for (int k = 0; k < block_range.size(); ++k) {
                bench_t m7 = flwrapper_blocked<&intersect_bitmaps_avx2_list>(n_variants, vals, n_ints_sample, pos_reg256, block_range[k]);
                PRINT("bitmap-avx2-skip-list-blocked-" + std::to_string(block_range[k]),m7);
            }
            bench_t m7 = flwrapper<&intersect_bitmaps_avx2_list>(n_variants, vals, n_ints_sample, pos_reg256);
            PRINT("bitmap-avx2-skip-list",m7);



            bench_t raw_roaring = frawwrapper<&intersect_roaring_cardinality>(n_variants, n_ints_sample, pos16);
            PRINT("raw-roaring",raw_roaring);

            bench_t raw_roaring2 = frawwrapper<&intersect_vector16_cardinality_roar>(n_variants, n_ints_sample, pos16);
            PRINT("raw-roaring2",raw_roaring2);

            
            for (int k = 0; k < block_range.size(); ++k) {
                bench_t m4_b10 = flwrapper_blocked<&intersect_bitmaps_scalar_list>(n_variants, vals, n_ints_sample, pos, block_range[k]);
                PRINT("bitmap-scalar-skip-list-4way-blocked-" + std::to_string(block_range[k]),m4_b10);
            }
            bench_t m4 = flwrapper<&intersect_bitmaps_scalar_list>(n_variants, vals, n_ints_sample, pos);
            PRINT("bitmap-scalar-skip-list",m4);


            for (int k = 0; k < block_range.size(); ++k) {
                bench_t m41x4_b10 = flwrapper_blocked<&intersect_bitmaps_scalar_list_1x4way>(n_variants, vals, n_ints_sample, pos, block_range[k]);
                PRINT("bitmap-scalar-skip-list-1x4way-blocked-" + std::to_string(block_range[k]),m41x4_b10);
            }
            bench_t m4_1x4way = flwrapper<&intersect_bitmaps_scalar_list_1x4way>(n_variants, vals, n_ints_sample, pos);
            PRINT("bitmap-scalar-skip-list-1x4way",m4_1x4way);
#endif

// #ifdef USE_ROARING
//             for (int i = 0; i < n_variants; ++i) roaring_bitmap_free(roaring[i]);
//                 delete[] roaring;
// #endif
//             continue;


            bench_t bins1 = frbinswrapper<&intersect_range_bins>(n_variants, n_ints_sample, bins, n_ints_bin);
            PRINT("bins-popcnt",bins1);

            bench_t bins_bitwise = frbinswrapper<&intersect_range_bins_bit>(n_variants, n_ints_sample, bins_bit, n_ints_bin);
            PRINT("bins-bit",bins_bitwise);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tbins-popcnt\t" << bins1.milliseconds << "\t" << bins1.count << "\t" << bins1.throughput << "\t" << (int_comparisons*1000 / (bins1.milliseconds)) << std::endl;

            // bench_t raw1_roaring = frawwrapper<&intersect_raw_naive_roaring>(n_variants, n_ints_sample, pos16);
            // PRINT("raw-naive-roaring",raw1_roaring);

            bench_t raw1_roaring_sse4 = frawwrapper<&intersect_raw_rotl_gallop_sse4>(n_variants, n_ints_sample, pos16);
            PRINT("raw-rotl-gallop-sse4",raw1_roaring_sse4);

            bench_t raw1_roaring_avx2= frawwrapper<&intersect_raw_rotl_gallop_avx2>(n_variants, n_ints_sample, pos16);
            PRINT("raw-rotl-gallop-avx2",raw1_roaring_avx2);

            


            if (n_alts[a] <= 200) {
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
                //std::cout << samples[s] << "\t" << n_alts[a] << "\trle-32\t" << mrle32.milliseconds << "\t" << mrle32.count << "\t" << mrle32.throughput << std::endl;
                PRINT("rle-32",mrle32);

                bench_t mrle32_b = frlewrapper< uint32_t, &intersect_rle_branchless<uint32_t> >(rle_32, n_ints_sample);
                //std::cout << samples[s] << "\t" << n_alts[a] << "\trle-32-branchless\t" << mrle32_b.milliseconds << "\t" << mrle32_b.count << "\t" << mrle32_b.throughput << std::endl;
                PRINT("rle-32-branchless",mrle32_b);

                bench_t mrle64 = frlewrapper< uint64_t, &intersect_rle<uint64_t> >(rle_64, n_ints_sample);
                //std::cout << samples[s] << "\t" << n_alts[a] << "\trle-64\t" << mrle64.milliseconds << "\t" << mrle64.count << "\t" << mrle64.throughput << std::endl;
                PRINT("rle-64",mrle64);

                bench_t mrle64_b = frlewrapper< uint64_t, &intersect_rle_branchless<uint64_t> >(rle_64, n_ints_sample);
                //std::cout << samples[s] << "\t" << n_alts[a] << "\trle-64-branchless\t" << mrle64_b.milliseconds << "\t" << mrle64_b.count << "\t" << mrle64_b.throughput << std::endl;
                PRINT("rle-64-branchless",mrle64_b);

                rle_32.clear(); rle_64.clear();
                */

                // Reduced
                bench_t red1 = fredwrapper<&insersect_reduced_sse4>(n_variants, n_ints_sample, vals_reduced, pos_integer16);
                //std::cout << samples[s] << "\t" << n_alts[a] << "\treduced-sse4-popcnt\t" << red1.milliseconds << "\t" << red1.count << "\t" << red1.throughput << std::endl;
                PRINT("reduced-sse4-popcnt",red1);

                bench_t red2 = fredwrapper<&insersect_reduced_scalar>(n_variants, n_ints_sample, vals_reduced, pos_integer16);
                //std::cout << samples[s] << "\t" << n_alts[a] << "\treduced-scalar-popcnt\t" << red2.milliseconds << "\t" << red2.count << "\t" << red2.throughput << std::endl;
                PRINT("reduced-scalar-popcnt",red2);

            } 
            // else {
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\traw-naive\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\traw-broadcast-sse4\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\traw-broadcast-avx2\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\traw-galloping-avx2\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;

            //     std::cout << samples[s] << "\t" << n_alts[a] << "\trle-32\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\trle-32-branchless\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\trle-64\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\trle-64-branchless\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;

            //     std::cout << samples[s] << "\t" << n_alts[a] << "\treduced-sse4-popcnt\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\treduced-scalar-popcnt\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            // }

            // Scalar 1
            bench_t m1 = fwrapper<&intersect_bitmaps_scalar>(n_variants, vals, n_ints_sample);
            PRINT("bitmap-scalar-popcnt",m1);
            
            // Scalar 4-way
            bench_t m4_way = fwrapper<&intersect_bitmaps_scalar_4way>(n_variants, vals, n_ints_sample);
            PRINT("bitmap-scalar-popcnt-4way",m4_way);
            
            // Scalar 8-way
            bench_t m8_way = fwrapper<&intersect_bitmaps_scalar_8way>(n_variants, vals, n_ints_sample);
            PRINT("bitmap-scalar-popcnt-8way",m8_way);

            // Scalar 1x4-way
            bench_t m1x4_way = fwrapper<&intersect_bitmaps_scalar_1x4way>(n_variants, vals, n_ints_sample);
            PRINT("bitmap-scalar-popcnt-1x4way",m1x4_way);

            // Scalar 1x8-way
            bench_t m1x8_way = fwrapper<&intersect_bitmaps_scalar_1x8way>(n_variants, vals, n_ints_sample);
            PRINT("bitmap-scalar-popcnt-1x8way",m1x8_way);

            // Scalar prefix-suffix 1x4-way
            bench_t ps_m1x4_way = fpswrapper<&intersect_bitmaps_scalar_prefix_suffix>(n_variants, vals, n_ints_sample, prefix_suffix_pos);
            PRINT("bitmap-scalar-prefix-suffix-popcnt-1x4way",ps_m1x4_way);

            // Scalar-list
            // if (n_alts[a] < 200 || (double)n_alts[a]/samples[a] < 0.05) {
                
            // } else {
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\tbitmap-scalar-skip-list\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\tbitmap-scalar-skip-list-4way\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            //     std::cout << samples[s] << "\t" << n_alts[a] << "\tbitmap-scalar-skip-list-1x4way\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            // }

            // Scalar-int-list
            bench_t m5 = flwrapper<&intersect_bitmaps_scalar_intlist>(n_variants, vals, n_ints_sample, pos_integer);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-int-list\t" << m5.milliseconds << "\t" << m5.count << "\t" << m5.throughput << std::endl;
            PRINT("bitmap-scalar-int-skip-list",m5);

            // Scalar-int-list
            bench_t m5_1x4 = flwrapper<&intersect_bitmaps_scalar_intlist_1x4way>(n_variants, vals, n_ints_sample, pos_integer);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-int-list-1x4\t" << m5_1x4.milliseconds << "\t" << m5_1x4.count << "\t" << m5_1x4.throughput << std::endl;
            PRINT("bitmap-scalar-int-skip-list-1x4",m5_1x4);

#if SIMD_VERSION >= 3
            // SIMD SSE 2-way
            bench_t m2_2way = fwrapper<&intersect_bitmaps_sse4_2way>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-2way\t" << m2_2way.milliseconds << "\t" << m2_2way.count << "\t" << m2_2way.throughput << std::endl;
            PRINT("bitmap-sse4-2way",m2_2way);

            // SIMD SSE 2-way
            bench_t m2_1x2way = fwrapper<&intersect_bitmaps_sse4_1x2way>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-1x2way\t" << m2_1x2way.milliseconds << "\t" << m2_1x2way.count << "\t" << m2_1x2way.throughput << std::endl;
            PRINT("bitmap-sse4-1x2way",m2_1x2way);


            // SIMD SSE4-list
            bench_t m6 = flwrapper<&intersect_bitmaps_sse4_list>(n_variants, vals, n_ints_sample, pos_reg128);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-list\t" << m6.milliseconds << "\t" << m6.count << "\t" << m6.throughput << std::endl;
            PRINT("bitmap-sse4-list",m6);

            // SIMD SSE4-squash
            bench_t m13 = fsqwrapper<&intersect_bitmaps_sse4_squash>(n_variants, vals, n_ints_sample, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-squash\t" << m13.milliseconds << "\t" << m13.count << "\t" << m13.throughput << std::endl;
            PRINT("bitmap-sse4-squash",m13);

            // SIMD SSE4-list-squash
            bench_t m14 = flsqwrapper<&intersect_bitmaps_sse4_list_squash>(n_variants, vals, n_ints_sample, pos_reg128, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-list-squash\t" << m14.milliseconds << "\t" << m14.count << "\t" << m14.throughput << std::endl;
            PRINT("bitmap-sse4-list-squash",m14);
#endif
#if SIMD_VERSION >= 5
            // SIMD AVX2
        
            // bench_t m3 = fwrapper<&intersect_bitmaps_avx2>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2\t" << m3.milliseconds << "\t" << m3.count << "\t" << m3.throughput << std::endl;
            // PRINT("bitmap-avx2",m3);

            // SIMD AVX2
            //bench_t m3twist = fwrapper_buffered<&intersect_bitmaps_avx2_twister>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2\t" << m3.milliseconds << "\t" << m3.count << "\t" << m3.throughput << std::endl;
            //PRINT("bitmap-avx2-twist",m3twist);

            // SIMD AVX2-list
            // if (n_alts[a] > 8) {
            // bench_t m7 = flwrapper<&intersect_bitmaps_avx2_list>(n_variants, vals, n_ints_sample, pos_reg256);
            // //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2-list\t" << m7.milliseconds << "\t" << m7.count << "\t" << m7.throughput << std::endl;
            // PRINT("bitmap-avx2-skip-list",m7);

            // SIMD AVX2-squash
            bench_t m10 = fsqwrapper<&intersect_bitmaps_avx2_squash>(n_variants, vals, n_ints_sample, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2-squash\t" << m10.milliseconds << "\t" << m10.count << "\t" << m10.throughput << std::endl;
            PRINT("bitmap-avx2-squash",m10);

            // SIMD AVX2-list-squash
            bench_t m12 = flsqwrapper<&intersect_bitmaps_avx2_list_squash>(n_variants, vals, n_ints_sample, pos_reg256, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2-list-squash\t" << m12.milliseconds << "\t" << m12.count << "\t" << m12.throughput << std::endl;
            PRINT("bitmap-avx2-skip-list-squash",m12);
            // }
#endif
#if SIMD_VERSION >= 6
            // SIMD AVX512
            bench_t m8 = fwrapper<&intersect_bitmaps_avx512>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512\t" << m8.milliseconds << "\t" << m8.count << "\t" << m8.throughput << std::endl;
            PRINT("bitmap-avx512",m8);

            // SIMD AVX512-list
            bench_t m9 = flwrapper<&intersect_bitmaps_avx512_list>(n_variants, vals, n_ints_sample, pos_reg512);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512-list\t" << m9.milliseconds << "\t" << m9.count << "\t" << m9.throughput << std::endl;
            PRINT("bitmap-avx512-skip-list",m9);

            // SIMD AVX512-squash
            bench_t m11 = fsqwrapper<&intersect_bitmaps_avx512_squash>(n_variants, vals, n_ints_sample, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512-squash\t" << m11.milliseconds << "\t" << m11.count << "\t" << m11.throughput << std::endl;
            PRINT("bitmap-avx512-squash",m11);

            // SIMD AVX512-list-squash
            bench_t m15 = flsqwrapper<&intersect_bitmaps_avx512_list_squash>(n_variants, vals, n_ints_sample, pos_reg512, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512-list-squash\t" << m15.milliseconds << "\t" << m15.count << "\t" << m15.throughput << std::endl;
            PRINT("bitmap-avx512-skip-list-squash",m15);
#endif
        
#ifdef USE_ROARING
        for (int i = 0; i < n_variants; ++i) roaring_bitmap_free(roaring[i]);
        delete[] roaring;
#endif
        }

        delete[] vals;
        delete[] vals_reduced;
    }
}

int debug(uint32_t n = 1000000) {
    uint64_t* vals1;
    assert(!posix_memalign((void**)&vals1, SIMD_ALIGNMENT, n*sizeof(uint64_t)));
    uint64_t* vals2;
    assert(!posix_memalign((void**)&vals2, SIMD_ALIGNMENT, n*sizeof(uint64_t)));
    
    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<uint64_t> distr(0, std::numeric_limits<uint64_t>::max()); // right inclusive

    for (int i = 0; i < n; ++i) {
        vals1[i] = distr(eng);
        vals2[i] = distr(eng);
    }
    
    const uint64_t cycles_start = get_cpu_cycles();
    intersect_bitmaps_avx2(vals1, vals2, n);
    const uint64_t cycles_end = get_cpu_cycles();

    std::cerr << "Cycles=" << cycles_end - cycles_start << " for " << (cycles_end - cycles_start)/((double)n) << " @ " << n << std::endl;

    delete[] vals1;
    delete[] vals2;

    return 0;
}

void debug_classes() {
    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<uint64_t> distr(0, 5008); // right inclusive

    std::vector<uint32_t> data;
    for (int i = 0; i < 256; ++i) {
        data.push_back(distr(eng));
    }

    std::sort(data.begin(), data.end());

    IntersectContainer it;
    it.construct(&data[0], data.size(), 5008);

    std::uniform_int_distribution<uint64_t> distr2(3000, 5000); // right inclusive
    data.clear();
    for (int i = 0; i < 34; ++i) {
        data.push_back(distr2(eng));
    }

    std::sort(data.begin(), data.end());
    IntersectContainer it2;
    it2.construct(&data[0], data.size(), 5008);

    // Intersect
    it.IntersectCount(it2);
}


int main(int argc, char **argv) {
    // debug_classes();
    // return EXIT_FAILURE;

    // debug(1000);
    // debug(10000);
    // debug(100000);
    // debug(1000000);
    // debug(10000000);
    
    intersect_test(100000000, 10);
    return(0);
}
