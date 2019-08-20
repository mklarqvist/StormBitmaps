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
#ifndef FAST_INTERSECT_EXPERIMENTAL_H_
#define FAST_INTERSECT_EXPERIMENTAL_H_

// #include <vector>
// #include <memory> //unique_ptr
// #include <iostream>//debug
// #include <bitset> //

#include "libalgebra/libalgebra.h"

#if defined(__AVX512F__)
#define SIMD_VERSION    6
#elif defined(__AVX2__)
#define SIMD_VERSION    5
#elif defined(__SSE4_2__) || defined(__AVX__)
#define SIMD_VERSION    3
#else
#define SIMD_VERSION    0
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if SIMD_VERSION >= 5
#ifndef STORM_POPCOUNT_AVX2
#define STORM_POPCOUNT_AVX2(A, B) {                  \
    A += STORM_POPCOUNT(_mm256_extract_epi64(B, 0)); \
    A += STORM_POPCOUNT(_mm256_extract_epi64(B, 1)); \
    A += STORM_POPCOUNT(_mm256_extract_epi64(B, 2)); \
    A += STORM_POPCOUNT(_mm256_extract_epi64(B, 3)); \
}
#endif
#endif

/****************************
*  Intersect vectors of scalars directly
****************************/
/**<
 * Compare pairs of uncompressed 16-bit integers from two sets pairwise.
 * Naive: This function compares scalars from the two lists pairwise in
 *    O(n*m)-time.
 * Broadcast: Vectorized approach where a value from the smaller vector is broadcast
 *    to a reference vector and compared against N scalars from the other vector.
 * @param v1
 * @param v2
 * @return
 */
uint64_t EXP_intersect_raw_naive(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_raw_naive_roaring(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_raw_naive_roaring_sse4(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_raw_sse4_broadcast(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_raw_rotl_gallop_sse4(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_raw_rotl_gallop_avx2(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_raw_sse4_broadcast_skip(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_raw_avx2_broadcast(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_raw_gallop(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_raw_gallop_sse4(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_raw_binary(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_roaring_cardinality(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t EXP_intersect_vector16_cardinality_roar(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);

#ifdef __cplusplus
} /* extern "C" */
#endif

/****************************
*  Run-length encoding
****************************/
/*
template <class int_t>
std::vector<int_t> construct_rle(const uint64_t* input, const uint32_t n_vals) {
    uint32_t n_runs = 0;
    uint32_t l_run  = 1;
    const uint32_t n_limit = sizeof(int_t)*8 - 1;
    uint32_t ref = (input[0] & 1);
    std::vector<int_t> vals;

    for (int i = 1; i < n_vals*sizeof(uint64_t)*8; ++i) {
        if (((input[i / 64] & (1L << (i % 64))) >> (i % 64)) != ref || l_run == n_limit) {
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

    while (true) {
        if (lenA > lenB) {
            lenA -= lenB;
            ltot += lenB * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));
            lenB = rle2[++offsetB] >> 1;
        } else if (lenA < lenB) {
            lenB -= lenA;
            ltot += lenA * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));
            lenA = rle1[++offsetA] >> 1;
        } else { // lenA == lenB
            ltot += lenB * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));
            lenA = rle1[++offsetA] >> 1;
            lenB = rle2[++offsetB] >> 1;
        }

        if (offsetA == limA && offsetB == limB) break;
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
    while (true) {
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

        if (offsetA == limA && offsetB == limB) break;
    }

    return(ltot);
}
*/


#endif /* FAST_INTERSECT_COUNT_H_ */