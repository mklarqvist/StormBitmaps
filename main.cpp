#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>
#include <algorithm>

#include <bitset>

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
#define TWK_POPCOUNT _mm_popcnt_u64
#else
#define TWK_POPCOUNT __builtin_popcountll
#endif

#if SIMD_VERSION >= 3
__attribute__((always_inline))
static inline void TWK_POPCOUNT_SSE(uint64_t& a, const __m128i n) {
    a += TWK_POPCOUNT(_mm_cvtsi128_si64(n)) + TWK_POPCOUNT(_mm_cvtsi128_si64(_mm_unpackhi_epi64(n, n)));
}
#endif

uint64_t intersect_bitmaps_scalar(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    for(int i = 0; i < n_ints; ++i) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
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
            count += TWK_POPCOUNT(b1[l1[i]] & b2[l1[i]]);
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            count += TWK_POPCOUNT(b1[l2[i]] & b2[l2[i]]);
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
        TWK_POPCOUNT_SSE(count, _mm_and_si128(r1[i], r2[i]));
    }

    return(count);
}

uint64_t intersect_bitmaps_sse4_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            TWK_POPCOUNT_SSE(count, _mm_and_si128(r1[l1[i]], r2[l1[i]]));
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            TWK_POPCOUNT_SSE(count, _mm_and_si128(r1[l2[i]], r2[l2[i]]));
        }
    }
    return(count);
}

uint64_t intersect_bitmaps_sse4_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    count = 0;
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    for(int i = 0; i < n_cycles; ++i) {
        TWK_POPCOUNT_SSE(count, _mm_and_si128(r1[i], r2[i]));
    }

    return(count);
}
#else
uint64_t intersect_bitmaps_sse4(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_sse4_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
uint64_t intersect_bitmaps_sse4_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
#endif // sse4 available

#if SIMD_VERSION >= 5

#ifndef TWK_POPCOUNT_AVX2
#define TWK_POPCOUNT_AVX2(A, B) {                  \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 0)); \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 1)); \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 2)); \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 3)); \
}
#endif

uint64_t intersect_bitmaps_avx2(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;
    const uint32_t n_cycles = n_ints / 4;

    for(int i = 0; i < n_cycles; ++i) {
        TWK_POPCOUNT_AVX2(count, _mm256_and_si256(r1[i], r2[i]));
    }

    return(count);
}

uint64_t intersect_bitmaps_avx2_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            TWK_POPCOUNT_AVX2(count, _mm256_and_si256(r1[l1[i]], r2[l1[i]]));
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            TWK_POPCOUNT_AVX2(count, _mm256_and_si256(r1[l2[i]], r2[l2[i]]));
        }
    }
    return(count);
}

uint64_t intersect_bitmaps_avx2_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    count = 0;
    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;
    const uint32_t n_cycles = n_ints / 4;

    for(int i = 0; i < n_cycles; ++i) {
        TWK_POPCOUNT_AVX2(count, _mm256_and_si256(r1[i], r2[i]));
    }

    return(count);
}

uint64_t intersect_bitmaps_avx2_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    count = 0;

    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            TWK_POPCOUNT_AVX2(count, _mm256_and_si256(r1[l1[i]], r2[l1[i]]));
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            TWK_POPCOUNT_AVX2(count, _mm256_and_si256(r1[l2[i]], r2[l2[i]]));
        }
    }
    return(count);
}

#else
uint64_t intersect_bitmaps_avx2(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_avx2_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
uint64_t intersect_bitmaps_avx2_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
uint64_t intersect_bitmaps_avx2_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
#endif // endif avx2

#if SIMD_VERSION >= 6

__attribute__((always_inline))
static inline __m512i TWK_AVX512_POPCNT(const __m512i v) {
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
        sum = _mm512_add_epi32(sum, TWK_AVX512_POPCNT(_mm512_and_si512(r1[i], r2[i])));
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
    __m512i sum = _mm512_set1_epi32(0);

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            sum = _mm512_add_epi32(sum, TWK_AVX512_POPCNT(_mm512_and_si512(r1[l1[i]], r2[l1[i]])));
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            sum = _mm512_add_epi32(sum, TWK_AVX512_POPCNT(_mm512_and_si512(r1[l2[i]], r2[l2[i]])));
        }
    }

    uint32_t* v = reinterpret_cast<uint32_t*>(&sum);
    for(int i = 0; i < 16; ++i)
        count += v[i];

    return(count);
}

uint64_t intersect_bitmaps_avx512_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
       count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    count = 0;

    const __m512i* r1 = (__m512i*)b1;
    const __m512i* r2 = (__m512i*)b2;
    const uint32_t n_cycles = n_ints / 8;
    __m512i sum = _mm512_set1_epi32(0);

    for(int i = 0; i < n_cycles; ++i) {
        sum = _mm512_add_epi32(sum, TWK_AVX512_POPCNT(_mm512_and_si512(r1[i], r2[i])));
    }

    uint32_t* v = reinterpret_cast<uint32_t*>(&sum);
    for(int i = 0; i < 16; ++i)
        count += v[i];

    return(count);
}


#else
uint64_t intersect_bitmaps_avx512(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_avx512_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
uint64_t intersect_bitmaps_avx512_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
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

template <class int_t>
std::vector<int_t> construct_rle(const uint64_t* input, const uint32_t n_vals) {
    uint32_t n_runs = 0;
    uint32_t l_run  = 1;
    const uint32_t n_limit = sizeof(int_t)*8 - 1;
    uint32_t ref = (input[0] & 1);
    std::vector<int_t> vals;

    for(int i = 1; i < n_vals*sizeof(uint64_t)*8; ++i) {
        if(((input[i / 64] & (1L << (i % 64))) >> (i % 64)) != ref || l_run == n_limit) {
            vals.push_back(((int_t)l_run << 1) | ref);
            ++n_runs;
            l_run = 0;
            ref = (input[i / 64] & (1L << (i % 64))) >> (i % 64);
        }
        ++l_run;
    }
    ++n_runs;
    vals.push_back(((int_t)l_run << 1) | ref);
    assert(vals.size() == n_runs);
    vals.push_back(0); // 1 value of nonsense for algorithm
    return(vals);
}

template <class int_t>
uint64_t intersect_rle(const std::vector<int_t>& rle1, const std::vector<int_t>& rle2) {

    int_t lenA = (rle1[0] >> 1);
    int_t lenB = (rle2[0] >> 1);
    uint32_t offsetA = 0;
    uint32_t offsetB = 0;
    const size_t limA = rle1.size() - 1;
    const size_t limB = rle2.size() - 1;
    uint64_t ltot = 0;

    while(true) {
        if(lenA > lenB) {
            lenA -= lenB;
            ltot += lenB * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));
            lenB = rle2[++offsetB] >> 1;
        } else if(lenA < lenB) {
            lenB -= lenA;
            ltot += lenA * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));
            lenA = rle1[++offsetA] >> 1;
        } else { // lenA == lenB
            ltot += lenB * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));
            lenA = rle1[++offsetA] >> 1;
            lenB = rle2[++offsetB] >> 1;
        }

        if(offsetA == limA && offsetB == limB) break;
    }

    return(ltot);
}

template <class int_t>
uint64_t intersect_rle_branchless(const std::vector<int_t>& rle1, const std::vector<int_t>& rle2) {

    int_t lenA = (rle1[0] >> 1);
    int_t lenB = (rle2[0] >> 1);
    uint32_t offsetA = 0;
    uint32_t offsetB = 0;
    const size_t limA = rle1.size() - 1;
    const size_t limB = rle2.size() - 1;

    int_t lA = 0, lB = 0;
    int64_t ltot = 0;
    bool predicate1 = false, predicate2 = false;
    while(true) {
        lA = lenA, lB = lenB;
        predicate1 = (lA >= lB);
        predicate2 = (lB >= lA);
        ltot += (predicate1 * lB + !predicate1 * lA) * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));

        offsetB += predicate1;
        offsetA += predicate2;

        lenA -= predicate1 * lB + !predicate1 * lA;
        lenB -= predicate2 * lA + !predicate2 * lB;
        lenA += predicate2 * (rle1[offsetA] >> 1);
        lenB += predicate1 * (rle2[offsetB] >> 1);

        if(offsetA == limA && offsetB == limB) break;
    }

    return(ltot);
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

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2)>
bench_t fsqwrapper(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, const std::vector< std::vector<uint64_t> >& squash) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;
    const uint32_t n_squash = squash[0].size();
    for(int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
        for(int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
            total += (*f)(&vals[offset], &vals[inner_offset], n_ints, n_squash, squash[i], squash[j]);
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

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2)>
bench_t flsqwrapper(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, const std::vector< std::vector<uint32_t> >& pos, const std::vector< std::vector<uint64_t> >& squash) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;
    const uint32_t n_squash = squash[0].size();
    for(int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
        for(int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
            total += (*f)(&vals[offset], &vals[inner_offset], pos[i], pos[j], n_squash, squash[i], squash[j]);
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

template <class int_t, uint64_t (f)(const std::vector<int_t>& rle1, const std::vector<int_t>& rle2)>
bench_t frlewrapper(const std::vector< std::vector<int_t> >& rle, const uint32_t n_ints) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;
    const uint32_t n_variants = rle.size();

    for(int i = 0; i < n_variants; ++i) {
        for(int j = i + 1; j < n_variants; ++j) {
            total += (*f)(rle[i], rle[j]);
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_ints*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);

    return(b);
}

void intersect_test(uint32_t n, uint32_t cycles = 1) {
    // Setup
    std::vector<uint32_t> samples = {512, 2048, 8192, 32768, 131072, 196608};
    for(int s = 0; s < samples.size(); ++s) {
        uint32_t n_ints_sample = samples[s] / 64;

        // Limit memory usage to 10e6 but no more than 10k records.
        uint32_t desired_mem = 10 * 1024 * 1024;
        // b_total / (b/obj) = n_ints
        uint32_t n_variants = std::min((uint32_t)10000, (uint32_t)std::ceil(desired_mem/(n_ints_sample*sizeof(uint64_t))));

        std::cerr << "Generating: " << samples[s] << " samples for " << n_variants << " variants" << std::endl;
        std::cerr << "Allocating: " << n_ints_sample*n_variants*sizeof(uint64_t)/(1024 * 1024.0) << "Mb" << std::endl;
        uint64_t* vals;
        assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, n_ints_sample*n_variants*sizeof(uint64_t)));

        std::vector<uint32_t> n_alts = {3, samples[s]/1000, samples[s]/500, samples[s]/100, samples[s]/20, samples[s]/10, samples[s]/4, samples[s]/2};
        //std::vector<uint32_t> n_alts = {3};

        for(int a = 0; a < n_alts.size(); ++a) {
            if(n_alts[a] == 0) continue;
            // Allocation
            memset(vals, 0, n_ints_sample*n_variants*sizeof(uint64_t));

            // PRNG
            std::uniform_int_distribution<uint32_t> distr(0, samples[s]-1); // right inclusive

            // Positional information
            uint32_t n_squash4096 = std::min((uint32_t)std::ceil((double)samples[s]/4096), (uint32_t)4);
            uint32_t divisor = samples[s]/n_squash4096;
            if(n_squash4096 == 1) divisor = samples[s];
            std::cerr << "nsq=" << n_squash4096 << " div=" << divisor << std::endl;
            std::vector< std::vector<uint32_t> > pos(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_integer(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_reg128(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_reg256(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_reg512(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint64_t> > squash_4096(n_variants, std::vector<uint64_t>(n_squash4096, 0));

            std::random_device rd;  // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator

            // Draw
            std::cerr << "Constructing..."; std::cerr.flush();
            uint64_t* vals2 = vals;
            for(uint32_t j = 0; j < n_variants; ++j) {
                for(uint32_t i = 0; i < n_alts[a]; ++i) {
                    uint64_t val = distr(eng);
                    if((vals2[val / 64] & (1L << (val % 64))) == 0) {
                        pos[j].push_back(val);
                    }
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

                // Todo
                // Squash into 4096 bins
                for(int p = 0; p < pos[j].size(); ++p) {
                    //std::cerr << (pos[j][p] / 4096) << "/" << n_squash4096 << "/" << squash_4096[j].size() << std::endl;
                    squash_4096[j][pos[j][p] / divisor] |= 1L << (pos[j][p] % divisor);
                }
                //for(int p = 0; p < squash_4096[j].size(); ++p) std::cerr << " " << std::bitset<64>(squash_4096[j][p]);
                //std::cerr << std::endl;

                // Todo print averages
                //std::cerr << pos.back().size() << "->" << pos_integer.back().size() << "->" << pos_reg128.back().size() << std::endl;
            }
            std::cerr << "Done!" << std::endl;

            //uint32_t offset = 0;
            /*for(int i = 0; i < n_variants; ++i) {
                construct_ewah64(&vals[offset], n_ints_sample);
                offset += n_ints_sample;
            }
            */

            if(n_alts[a] <= 20) {
                std::vector< std::vector<uint32_t> > rle_32(n_variants, std::vector<uint32_t>());
                std::vector< std::vector<uint64_t> > rle_64(n_variants, std::vector<uint64_t>());

                uint32_t offset = 0;
                for(int i = 0; i < n_variants; ++i) {
                    rle_32[i]  = construct_rle<uint32_t> (&vals[offset], n_ints_sample);
                    rle_64[i] = construct_rle<uint64_t>(&vals[offset], n_ints_sample);
                    offset += n_ints_sample;
                }

                bench_t mrle32 = frlewrapper< uint32_t, &intersect_rle<uint32_t> >(rle_32, n_ints_sample);
                std::cout << samples[s] << "\t" << n_alts[a] << "\trle-32\t" << mrle32.milliseconds << "\t" << mrle32.count << "\t" << mrle32.throughput << std::endl;

                //bench_t mrle32_b = frlewrapper< uint32_t, &intersect_rle_branchless<uint32_t> >(rle_32, n_ints_sample);
                //std::cout << samples[s] << "\t" << n_alts[a] << "\trle-32-branchless\t" << mrle32_b.milliseconds << "\t" << mrle32_b.count << "\t" << mrle32_b.throughput << std::endl;

                bench_t mrle64 = frlewrapper< uint64_t, &intersect_rle<uint64_t> >(rle_64, n_ints_sample);
                std::cout << samples[s] << "\t" << n_alts[a] << "\trle-64\t" << mrle64.milliseconds << "\t" << mrle64.count << "\t" << mrle64.throughput << std::endl;

                //bench_t mrle64_b = frlewrapper< uint64_t, &intersect_rle_branchless<uint64_t> >(rle_64, n_ints_sample);
                //std::cout << samples[s] << "\t" << n_alts[a] << "\trle-64-branchless\t" << mrle64_b.milliseconds << "\t" << mrle64_b.count << "\t" << mrle64_b.throughput << std::endl;

                rle_32.clear(); rle_64.clear();

            } else {
                std::cout << samples[s] << "\t" << n_alts[a] << "\trle-32\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
                //std::cout << samples[s] << "\t" << n_alts[a] << "\trle-32-branchless\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
                std::cout << samples[s] << "\t" << n_alts[a] << "\trle-64\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
                //std::cout << samples[s] << "\t" << n_alts[a] << "\trle-64-branchless\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            }

            // Scalar 1
            bench_t m1 = fwrapper<&intersect_bitmaps_scalar>(n_variants, vals, n_ints_sample);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-bit\t" << m1.milliseconds << "\t" << m1.count << "\t" << m1.throughput << std::endl;

            // Scalar-list
            if(n_alts[a] < 200 || (double)n_alts[a]/samples[a] < 0.05) {
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

            // SIMD SSE4-squash
            bench_t m13 = fsqwrapper<&intersect_bitmaps_sse4_squash>(n_variants, vals, n_ints_sample, squash_4096);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-squash\t" << m13.milliseconds << "\t" << m13.count << "\t" << m13.throughput << std::endl;

            // SIMD AVX2
            bench_t m3 = fwrapper<&intersect_bitmaps_avx2>(n_variants, vals, n_ints_sample);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2\t" << m3.milliseconds << "\t" << m3.count << "\t" << m3.throughput << std::endl;

            // SIMD AVX2-list
            bench_t m7 = flwrapper<&intersect_bitmaps_avx2_list>(n_variants, vals, n_ints_sample, pos_reg256);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2-list\t" << m7.milliseconds << "\t" << m7.count << "\t" << m7.throughput << std::endl;

            // SIMD AVX2-squash
            bench_t m10 = fsqwrapper<&intersect_bitmaps_avx2_squash>(n_variants, vals, n_ints_sample, squash_4096);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2-squash\t" << m10.milliseconds << "\t" << m10.count << "\t" << m10.throughput << std::endl;

            // SIMD AVX2-list-squash
           bench_t m12 = flsqwrapper<&intersect_bitmaps_avx2_list_squash>(n_variants, vals, n_ints_sample, pos_reg256, squash_4096);
           std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2-list-squash\t" << m12.milliseconds << "\t" << m12.count << "\t" << m12.throughput << std::endl;

            // SIMD AVX512
            bench_t m8 = fwrapper<&intersect_bitmaps_avx512>(n_variants, vals, n_ints_sample);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512\t" << m8.milliseconds << "\t" << m8.count << "\t" << m8.throughput << std::endl;

            // SIMD AVX512-list
            bench_t m9 = flwrapper<&intersect_bitmaps_avx512_list>(n_variants, vals, n_ints_sample, pos_reg512);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512-list\t" << m9.milliseconds << "\t" << m9.count << "\t" << m9.throughput << std::endl;

            // SIMD AVX512-squash
            bench_t m11 = fsqwrapper<&intersect_bitmaps_avx512_squash>(n_variants, vals, n_ints_sample, squash_4096);
            std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512-squash\t" << m11.milliseconds << "\t" << m11.count << "\t" << m11.throughput << std::endl;


        }

        delete[] vals;
    }
}

int main(int argc, char **argv) {
    intersect_test(100000000, 10);
    return(0);
}
