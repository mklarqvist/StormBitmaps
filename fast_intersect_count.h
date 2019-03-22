/*
* Copyright (c) 2019 Marcus D. R. Klarqvist
* Author(s): Marcus D. R. Klarqvist
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
#ifndef FAST_INTERSECT_COUNT_H_
#define FAST_INTERSECT_COUNT_H_

#include <cstdint>
#include <cassert>
#include <memory>
#include <cstring>
#include <vector>
#include <limits>
// temp
#include <iostream>


/*
 * Reference data transfer rates:
 *
 * DDR4 2133：17 GB/s
 * DDR4 2400：19.2 GB/s
 * DDR4 2666：21.3 GB/s
 * DDR4 3200：25.6 GB/s
 */

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
#define __AVX2__ 1

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
#ifndef TWK_POPCOUNT_SSE4
#define TWK_POPCOUNT_SSE4(A, B) {               \
    A += TWK_POPCOUNT(_mm_extract_epi64(B, 0)); \
    A += TWK_POPCOUNT(_mm_extract_epi64(B, 1)); \
}
#endif

__attribute__((always_inline))
static inline void TWK_POPCOUNT_SSE(uint64_t& a, const __m128i n) {
    a += TWK_POPCOUNT(_mm_cvtsi128_si64(n)) + TWK_POPCOUNT(_mm_cvtsi128_si64(_mm_unpackhi_epi64(n, n)));
}
#endif

/****************************
*  Class definitions
****************************/
struct bin {
    bin() : list(false), n_vals(0), n_list(std::numeric_limits<uint32_t>::max()), bitmap(0), vals(nullptr){}
    ~bin(){ delete[] vals; }

    void Allocate(const uint16_t n) {
        delete[] vals;
        assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, n*sizeof(uint64_t)));
        memset(vals, 0, n*sizeof(uint64_t));
        n_vals = n;
    }

    inline const uint64_t& operator[](const uint16_t p) const { return(vals[p]); }

    bool list;
    uint16_t n_vals; // limited to 64 uint64_t
    uint32_t n_list;
    uint64_t bitmap; // bitmap of bitmaps (equivalent to squash)
    uint64_t* vals; // pointer to data
    std::shared_ptr< std::vector<uint16_t> > pos;
};

struct parent_bin {
    inline const uint16_t& size() const { return(n_vals); }

    uint8_t bitmap: 1, list: 1, unused: 6;
    uint16_t n_vals;
    uint8_t* raw;
    std::shared_ptr< std::vector<uint16_t> > skip_list; // used only for bitmap
};

// interpret data as uint64_t
struct bitmap_bin : public parent_bin {
    inline const uint64_t* data() const { return(reinterpret_cast<const uint64_t*>(raw)); }
};

// intepret data as uint16_t
struct array_bin : public parent_bin {
    inline const uint16_t* data() const { return(reinterpret_cast<const uint16_t*>(raw)); }
};

struct range_bin {
    range_bin() : list(false), n_list(std::numeric_limits<uint32_t>::max()), n_ints(0), bin_bitmap(0){}
    range_bin(uint32_t n_bins) : list(false), n_list(std::numeric_limits<uint32_t>::max()), n_ints(0), bin_bitmap(0){ bins.resize(n_bins); }

    bool list;
    uint32_t n_list, n_ints;
    uint64_t bin_bitmap; // what bins are set
    std::vector< bin > bins;
    std::shared_ptr< std::vector<uint16_t> > pos;
    //uint64_t* vals_con;
};

uint64_t intersect_range_bins(const range_bin& b1, const range_bin& b2, const uint8_t n_ints_bin);
uint64_t intersect_range_bins_bit(const range_bin& b1, const range_bin& b2, const uint8_t n_ints_bin);

/****************************
*  Function definitions
****************************/
uint64_t intersect_bitmaps_scalar(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints);
uint64_t intersect_bitmaps_scalar_4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints);
uint64_t intersect_bitmaps_scalar_1x4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints);
uint64_t intersect_bitmaps_scalar_8way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints);
uint64_t intersect_bitmaps_scalar_1x8way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints);
uint64_t intersect_bitmaps_scalar_prefix_suffix(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const std::pair<uint32_t,uint32_t>& p1, const std::pair<uint32_t,uint32_t>& p2);
uint64_t intersect_bitmaps_scalar_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2);
uint64_t intersect_bitmaps_scalar_list_4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2);
uint64_t intersect_bitmaps_scalar_list_1x4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2);
uint64_t intersect_bitmaps_scalar_intlist(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2);
uint64_t intersect_bitmaps_scalar_intlist_1x4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2);

// SSE4
uint64_t intersect_bitmaps_sse4(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints);
uint64_t intersect_bitmaps_sse4_2way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints);
uint64_t intersect_bitmaps_sse4_1x2way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints);
uint64_t intersect_bitmaps_sse4_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2);
uint64_t intersect_bitmaps_sse4_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2);
uint64_t intersect_bitmaps_sse4_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2);uint64_t insersect_reduced_sse4(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint16_t>& l1, const std::vector<uint16_t>& l2);

// AVX2
uint64_t intersect_bitmaps_avx2(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints);
uint64_t intersect_bitmaps_avx2_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2);
uint64_t intersect_bitmaps_avx2_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2);
uint64_t intersect_bitmaps_avx2_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2);

// AVX512
uint64_t intersect_bitmaps_avx512(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints);
uint64_t intersect_bitmaps_avx512_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2);
uint64_t intersect_bitmaps_avx512_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2);
uint64_t intersect_bitmaps_avx512_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2);

// Reduced
uint64_t insersect_reduced_scalar(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint16_t>& l1, const std::vector<uint16_t>& l2);

/****************************
*  Run-length encoding
****************************/
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

/****************************
*  Intersect vectors of values directly
****************************/
/**<
 * Compare pairs of uncompressed 16-bit integers from two sets pairwise.
 * Naive: This function compares values from the two lists pairwise in
 *    O(n*m)-time.
 * Broadcast: Vectorized approach where a value from the smaller vector is broadcast
 *    to a reference vector and compared against N values from the other vector.
 * @param v1
 * @param v2
 * @return
 */
uint64_t intersect_raw_naive(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_raw_naive_roaring(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_raw_naive_roaring_sse4(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_raw_sse4_broadcast(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_raw_rotl_gallop_sse4(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_raw_rotl_gallop_avx2(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_raw_sse4_broadcast_skip(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_raw_avx2_broadcast(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_raw_gallop(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_raw_gallop_sse4(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_raw_binary(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_roaring_cardinality(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);
uint64_t intersect_vector16_cardinality_roar(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2);

// construct ewah
void construct_ewah64(const uint64_t* input, const uint32_t n_vals);

#endif /* FAST_INTERSECT_COUNT_H_ */
