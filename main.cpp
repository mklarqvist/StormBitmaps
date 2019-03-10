#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>
#include <algorithm>

/****************************
*  SIMD definitions
****************************/
#if defined(_MSC_VER)
     /* Microsoft C/C++-compatible compiler */
     #include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
     #include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
     /* GCC-compatible compiler, targeting ARM with NEON */
     #include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
     /* GCC-compatible compiler, targeting ARM with WMMX */
     #include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
     /* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
     #include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
     /* GCC-compatible compiler, targeting PowerPC with SPE */
     #include <spe.h>
#endif

//temp
//#define __AVX512F__ 1
//#define __AVX2__ 1

#if defined(__AVX512F__) && __AVX512F__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    6
#define SIMD_WIDTH      512
#define SIMD_ALIGNMENT  64
#elif defined(__AVX2__) && __AVX2__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    5
#define SIMD_WIDTH      256
#define SIMD_ALIGNMENT  32
#elif defined(__AVX__) && __AVX__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    4
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      128
#elif defined(__SSE4_1__) && __SSE4_1__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    3
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      128
#elif defined(__SSE2__) && __SSE2__ == 1
#define SIMD_AVAILABLE  0 // unsupported version
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      128
#elif defined(__SSE__) && __SSE__ == 1
#define SIMD_AVAILABLE  0 // unsupported version
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      0
#else
#define SIMD_AVAILABLE  0
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      0
#endif

#ifdef _mm_popcnt_u64
#define PIL_POPCOUNT _mm_popcnt_u64
#else
#define PIL_POPCOUNT __builtin_popcountll
#endif

#if SIMD_VERSION >= 3
__attribute__((always_inline))
static inline void PIL_POPCOUNT_SSE(uint64_t& a, const __m128i n) {
    a += PIL_POPCOUNT(_mm_cvtsi128_si64(n)) + PIL_POPCOUNT(_mm_cvtsi128_si64(_mm_unpackhi_epi64(n, n)));
}
#endif

uint64_t intersect_bitmaps_scalar(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    for(int i = 0; i < n_ints; ++i) {
        count += PIL_POPCOUNT(b1[i] & b2[i]);
    }

    return(count);
}

uint64_t intersect_bitmaps_scalar_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            count += ((b2[l1[i] / 64] & (1L << (l1[i] % 64))) != 0);
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            count += ((b1[l2[i] / 64] & (1L << (l2[i] % 64))) != 0);
        }
    }
    return(count);
}

uint64_t intersect_bitmaps_scalar_intlist(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            count += PIL_POPCOUNT(b1[l1[i]] & b2[l1[i]]);
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            count += PIL_POPCOUNT(b1[l2[i]] & b2[l2[i]]);
        }
    }
    return(count);
}

#if SIMD_VERSION >= 3
uint64_t intersect_bitmaps_sse4(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    for(int i = 0; i < n_cycles; ++i) {
        PIL_POPCOUNT_SSE(count, _mm_and_si128(r1[i], r2[i]));
    }

    return(count);
}

uint64_t intersect_bitmaps_sse4_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            PIL_POPCOUNT_SSE(count, _mm_and_si128(r1[l1[i]], r2[l1[i]]));
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            PIL_POPCOUNT_SSE(count, _mm_and_si128(r1[l2[i]], r2[l2[i]]));
        }
    }
    return(count);
}
#else
uint64_t intersect_bitmaps_sse4(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_sse4_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
#endif // sse4 available

#if SIMD_VERSION >= 5

#ifndef PIL_POPCOUNT_AVX2
#define PIL_POPCOUNT_AVX2(A, B) {                  \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 0)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 1)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 2)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 3)); \
}
#endif

uint64_t intersect_bitmaps_avx2(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;
    const uint32_t n_cycles = n_ints / 4;

    for(int i = 0; i < n_cycles; ++i) {
        PIL_POPCOUNT_AVX2(count, _mm256_and_si256(r1[i], r2[i]));
    }

    return(count);
}

uint64_t intersect_bitmaps_avx2_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            PIL_POPCOUNT_AVX2(count, _mm256_and_si256(r1[l1[i]], r2[l1[i]]));
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            PIL_POPCOUNT_AVX2(count, _mm256_and_si256(r1[l2[i]], r2[l2[i]]));
        }
    }
    return(count);
}

#else
uint64_t intersect_bitmaps_avx2(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_avx2_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
#endif // endif avx2

#if SIMD_VERSION >= 6

__attribute__((always_inline))
static inline __m512i avx512_popcount(const __m512i v) {
    const __m512i m1 = _mm512_set1_epi8(0x55);
    const __m512i m2 = _mm512_set1_epi8(0x33);
    const __m512i m4 = _mm512_set1_epi8(0x0F);

    const __m512i t1 = _mm512_sub_epi8(v,       (_mm512_srli_epi16(v,  1) & m1));
    const __m512i t2 = _mm512_add_epi8(t1 & m2, (_mm512_srli_epi16(t1, 2) & m2));
    const __m512i t3 = _mm512_add_epi8(t2, _mm512_srli_epi16(t2, 4)) & m4;
    return _mm512_sad_epu8(t3, _mm512_setzero_si512());
}

uint64_t intersect_bitmaps_avx512(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    const __m512i* r1 = (__m512i*)b1;
    const __m512i* r2 = (__m512i*)b2;
    const uint32_t n_cycles = n_ints / 8;
    __m512i sum = _mm512_set1_epi32(0);

    for(int i = 0; i < n_cycles; ++i) {
        sum = _mm512_add_epi32(sum, avx512_popcount(_mm512_and_si512(r1[i], r2[i])));
    }

    uint32_t* v = reinterpret_cast<uint32_t*>(&sum);
    for(int i = 0; i < 16; ++i)
        count += v[i];

    return(count);
}

uint64_t intersect_bitmaps_avx512_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;
    const __m512i* r1 = (__m512i*)b1;
    const __m512i* r2 = (__m512i*)b2;
    const uint32_t n_cycles = n_ints / 8;
    __m512i sum = _mm512_set1_epi32(0);

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            sum = _mm512_add_epi32(sum, avx512_popcount(_mm512_and_si512(r1[l1[i]], r2[l1[i]])));
            //PIL_POPCOUNT_AVX2(count, _mm256_and_si256(r1[l1[i]], r2[l1[i]]));
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            sum = _mm512_add_epi32(sum, avx512_popcount(_mm512_and_si512(r1[l2[i]], r2[l2[i]])));
            //PIL_POPCOUNT_AVX2(count, _mm256_and_si256(r1[l2[i]], r2[l2[i]]));
        }
    }

    uint32_t* v = reinterpret_cast<uint32_t*>(&sum);
    for(int i = 0; i < 16; ++i)
        count += v[i];

    return(count);
}


#else
uint64_t intersect_bitmaps_avx512(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_avx512_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
#endif // endif avx2

void construct_ewah64(const uint64_t* input, const uint32_t n_vals) {
    struct control_word {
        uint64_t type: 1, symbol: 1, length: 62;
    };

    std::vector<control_word> words;
    control_word current;
    uint32_t i = 0;
    // first word
    if(input[0] == 0) {
        current.type = 1;
        current.symbol = 0;
        current.length = 1;
        for(i = 1; i < n_vals; ++i) {
            if(input[i] != 0) break;
            ++current.length;
        }
        words.push_back(current);
    }
    else if(input[0] == std::numeric_limits<uint64_t>::max()) {
        current.type = 1;
        current.symbol = 1;
        current.length = 1;
        for(i = 1; i < n_vals; ++i) {
            if(input[i] != std::numeric_limits<uint64_t>::max()) break;
            ++current.length;
        }
        words.push_back(current);
    } else {
        current.type = 0;
        current.symbol = 0;
        current.length = 1;
        for(i = 1; i < n_vals; ++i) {
            if(input[i] == 0 || input[i] == std::numeric_limits<uint64_t>::max()) break;
            ++current.length;
        }
        words.push_back(current);
    }
    current = control_word();
    if(i < n_vals) {
        // remainder words
        while(true) {
            if(input[i] == 0) {
                current.type = 1;
                current.symbol = 0;
                current.length = 1;
                ++i;
                for(; i < n_vals; ++i) {
                    if(input[i] != 0) break;
                    ++current.length;
                }
                words.push_back(current);
            }
            else if(input[i] == std::numeric_limits<uint64_t>::max()) {
                current.type = 1;
                current.symbol = 1;
                current.length = 1;
                ++i;
                for(; i < n_vals; ++i) {
                    if(input[i] != std::numeric_limits<uint64_t>::max()) break;
                    ++current.length;
                }
                words.push_back(current);
            } else {
                current.type = 0;
                current.symbol = 0;
                current.length = 1;
                ++i;
                for(; i < n_vals; ++i) {
                    if(input[i] == 0 || input[i] == std::numeric_limits<uint64_t>::max()) break;
                    ++current.length;
                }
                words.push_back(current);
            }
            if(i >= n_vals) break;
            current = control_word();
        }
    }

    uint32_t sum = 0;
    std::cerr << words.size() << ":";
    for(int i = 0; i < words.size(); ++i) {
        std::cerr << " {" << words[i].length << "," << words[i].symbol << "," << (words[i].type == 0 ? "dirty" : "clean") << "}";
        sum += words[i].length;
    }
    std::cerr << " : total=" << sum << std::endl;

}

// Convenience wrapper
struct bench_t {
    uint64_t count;
    uint32_t milliseconds;
    double throughput;
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
    for(int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
        for(int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
            total += (*f)(&vals[offset], &vals[inner_offset], n_ints);
        }
        offset += n_ints;
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_ints*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);

    return(b);
}

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2)>
bench_t flwrapper(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, const std::vector< std::vector<uint32_t> >& pos) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;
    for(int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
       for(int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
           total += (*f)(&vals[offset], &vals[inner_offset], pos[i], pos[j]);
       }
       offset += n_ints;
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_ints*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);

    return(b);
}

void intersect_test(uint32_t n, uint32_t cycles = 1) {
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    //uint64_t times[9] = {0};
    //uint64_t times_local[9];

    // Setup
    std::vector<uint32_t> samples = {256, 2048, 8192, 32768};
    for(int s = 0; s < samples.size(); ++s) {
        uint32_t n_ints_sample = samples[s] / 64;
        uint32_t n_variants = 5000;

        std::cerr << "Generating: " << samples[s] << " samples for " << n_variants << " variants" << std::endl;
        std::cerr << "Allocating: " << n_ints_sample*n_variants*sizeof(uint32_t) << std::endl;
        uint64_t* vals;
        assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, n_ints_sample*n_variants*sizeof(uint64_t)));

        std::vector<uint32_t> n_alts = {3, samples[s]/1000, samples[s]/500, samples[s]/100, samples[s]/20, samples[s]/10, samples[s]/4, samples[s]/2};

        for(int a = 0; a < n_alts.size(); ++a) {
            if(n_alts[a] == 0) continue;
            // Allocation
            memset(vals, 0, n_ints_sample*n_variants*sizeof(uint64_t));

            // PRNG
            std::uniform_int_distribution<uint32_t> distr(0, samples[s]-1); // right inclusive

            // Positional information
            std::vector< std::vector<uint32_t> > pos(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_integer(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_reg128(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_reg256(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_reg512(n_variants, std::vector<uint32_t>());

            // Draw
            //uint32_t offset = 0;
            uint64_t* vals2 = vals;
            for(uint32_t j = 0; j < n_variants; ++j) {
                //pos.push_back(std::vector<uint32_t>());
                for(uint32_t i = 0; i < n_alts[a]; ++i) {
                    //vals[(offset + i) / 32] |= 1 << distr(eng);
                    uint32_t val = distr(eng);
                    if((vals2[val / 64] & (1L << (val % 64))) == 0) {
                        pos[j].push_back(val);
                    }
                    //std::cerr << "setting=" << offset+i << std::endl;
                    vals2[val / 64] |= (1L << (val % 64));
                }
                vals2 += n_ints_sample;

                // Sort to simplify
                std::sort(pos[j].begin(), pos[j].end());
                //for(int p = 0; p < pos.back().size(); ++p) std::cerr << " " << pos.back()[p];
                //std::cerr << std::endl;

                // Todo
                // Collapse positions into integers
               // pos_integer.push_back(std::vector<uint32_t>());
                pos_integer[j].push_back(pos[j][0] / 64);

                for(int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 64;
                    if(pos_integer[j].back() != idx) pos_integer[j].push_back(idx);
                }

                //for(int p = 0; p < pos_integer.back().size(); ++p) std::cerr << " " << pos_integer.back()[p];
                //std::cerr << std::endl;

                // Todo
                // Collapse positions into registers
                //pos_reg128.push_back(std::vector<uint32_t>());
                pos_reg128[j].push_back(pos[j][0] / 128);

                for(int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 128;
                    if(pos_reg128[j].back() != idx) pos_reg128[j].push_back(idx);
                }

                // Todo
                // Collapse positions into 256-registers
                //pos_reg256.push_back(std::vector<uint32_t>());
                pos_reg256[j].push_back(pos[j][0] / 256);

                for(int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 256;
                    if(pos_reg256[j].back() != idx) pos_reg256[j].push_back(idx);
                }

                // Todo
                // Collapse positions into 512-registers
                //pos_reg512.push_back(std::vector<uint32_t>());
                pos_reg512[j].push_back(pos[j][0] / 512);

                for(int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 512;
                    if(pos_reg512[j].back() != idx) pos_reg512[j].push_back(idx);
                }

                // Todo print averages
                //std::cerr << pos.back().size() << "->" << pos_integer.back().size() << "->" << pos_reg128.back().size() << std::endl;
            }

            //uint32_t offset = 0;
            /*for(int i = 0; i < n_variants; ++i) {
                construct_ewah64(&vals[offset], n_ints_sample);
                offset += n_ints_sample;
            }
            */

            // Scalar 1
            bench_t m1 = fwrapper<&intersect_bitmaps_scalar>(n_variants, vals, n_ints_sample);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-bit\t" << m1.milliseconds << "\t" << m1.count << "\t" << m1.throughput << std::endl;

            // Scalar-list
            if(n_alts[a] < 200) {
                bench_t m4 = flwrapper<&intersect_bitmaps_scalar_list>(n_variants, vals, n_ints_sample, pos);
                std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-list\t" << m4.milliseconds << "\t" << m4.count << "\t" << m4.throughput << std::endl;
            } else {
                std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-list\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            }

            // Scalar-int-list
            bench_t m5 = flwrapper<&intersect_bitmaps_scalar_intlist>(n_variants, vals, n_ints_sample, pos_integer);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-int-list\t" << m5.milliseconds << "\t" << m5.count << "\t" << m5.throughput << std::endl;

            // SIMD SSE4
            bench_t m2 = fwrapper<&intersect_bitmaps_sse4>(n_variants, vals, n_ints_sample);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4\t" << m2.milliseconds << "\t" << m2.count << "\t" << m2.throughput << std::endl;

            // SIMD SSE4-list
            bench_t m6 = flwrapper<&intersect_bitmaps_sse4_list>(n_variants, vals, n_ints_sample, pos_reg128);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-list\t" << m6.milliseconds << "\t" << m6.count << "\t" << m6.throughput << std::endl;

            // SIMD AVX2
            bench_t m3 = fwrapper<&intersect_bitmaps_avx2>(n_variants, vals, n_ints_sample);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2\t" << m3.milliseconds << "\t" << m3.count << "\t" << m3.throughput << std::endl;

            // SIMD AVX2-list
            bench_t m7 = flwrapper<&intersect_bitmaps_avx2_list>(n_variants, vals, n_ints_sample, pos_reg256);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2-list\t" << m7.milliseconds << "\t" << m7.count << "\t" << m7.throughput << std::endl;

            // SIMD AVX512
            bench_t m8 = fwrapper<&intersect_bitmaps_avx512>(n_variants, vals, n_ints_sample);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512\t" << m8.milliseconds << "\t" << m8.count << "\t" << m8.throughput << std::endl;

            // SIMD AVX512-list
            bench_t m9 = flwrapper<&intersect_bitmaps_avx512_list>(n_variants, vals, n_ints_sample, pos_reg512);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512-list\t" << m9.milliseconds << "\t" << m9.count << "\t" << m9.throughput << std::endl;
        }

        delete[] vals;
    }
}

int main(int argc, char **argv) {
    intersect_test(100000000, 10);
    return(0);
}
