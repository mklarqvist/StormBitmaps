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
#include "experimental.h"

#ifndef UNSAFE_MAX
#define UNSAFE_MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef UNSAFE_MIN
#define UNSAFE_MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

//
uint64_t EXP_intersect_raw_naive(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    uint64_t count = 0;
    for(int i = 0; i < len1; ++i) {
        for(int j = 0; j < len2; ++j) {
            count += (v1[i] == v2[j]);
        }
    }
    return(count);
}

uint64_t EXP_intersect_raw_naive_roaring(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    uint64_t answer = 0;
    if (len1 == 0 || len2 == 0) return 0;
    const uint16_t *A = v1;
    const uint16_t *B = v2;
    const uint16_t *endA = A + len1;
    const uint16_t *endB = B + len2;

    while (1) {
        while (*A < *B) {
            SKIP_FIRST_COMPARE:
            if (++A == endA) return answer;
        }
        while (*A > *B) {
            if (++B == endB) return answer;
        }
        if (*A == *B) {
            ++answer;
            if (++A == endA || ++B == endB) return answer;
        } else {
            goto SKIP_FIRST_COMPARE;
        }
    }
    return answer; // NOTREACHED
}

uint64_t EXP_intersect_raw_naive_roaring_sse4(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    uint64_t answer = 0;
    if (len1 == 0 || len2 == 0) return 0;
    //const uint16_t *A = v1;
    //const uint16_t *B = v2;
    uint32_t A = 0, B = 0;
    const uint32_t endA = len1;
    const uint32_t endB = len2;
    //const __m128i one_mask = _mm_set1_epi16(1);

    while (1) {
        while (v1[A] < v2[B]) {
            SKIP_FIRST_COMPARE:
            if (++A == endA) return answer;
        }
        while (v1[A] > v2[B]) {
            if (++B == endB) return answer;
        }
        if (v1[A] == v2[B]) {
            ++answer;
            if (++A == endA || ++B == endB) return answer;
        } else {
            goto SKIP_FIRST_COMPARE;
        }
    }
    return answer; // NOTREACHED
}

#if SIMD_VERSION >= 3
uint64_t EXP_intersect_raw_sse4_broadcast(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    uint64_t count = 0;
    const __m128i one_mask = _mm_set1_epi8(255);
    if(len1 < len2) { // broadcast-compare V1-values to vectors of V2 values
        if(v1[len1-1] < v2[0]) return 0;

        const uint32_t n_cycles = len2 / 8;
       // const __m128i* y = (const __m128i*)(&v2[0]);
        int i = 0;
        while (v1[i] < v2[0]) {
            if(v1[++i] == len1) return 0;
        }

        for(; i < len1; ++i) {
            if(v1[i] < v2[0]) continue;
            if(v1[i] > v2[len2-1]) {
                // will never overlap
                return count;
            }

            const __m128i r = _mm_set1_epi16(v1[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m128i y = _mm_loadu_si128((const __m128i*)&v2[j*8]);
                count += _mm_testnzc_si128(_mm_cmpeq_epi16(r, y), one_mask);
                //STORM_POPCOUNT_SSE4(count, _mm_and_si128(_mm_cmpeq_epi16(r, y),one_mask));
            }
            j *= 8;
            for(; j < len2; ++j) count += (v1[i] == v2[j]);
        }
    } else {
        if(v2[len2-1] < v1[0]) return 0;
        const uint32_t n_cycles = len1 / 8;

        int i = 0;
        while (v2[i] < v1[0]) {
            if(v2[++i] == len2) return 0;
        }

        for(; i < len2; ++i) {
            if(v2[i] < v1[0]) continue;
            if(v2[i] > v1[len1-1]) {
                // will never overlap
                return count;
            }
            const __m128i r = _mm_set1_epi16(v2[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m128i y = _mm_loadu_si128((const __m128i*)&v1[j*8]);
                //STORM_POPCOUNT_SSE4(count, _mm_and_si128(_mm_cmpeq_epi16(r, y),one_mask));
                count += _mm_testnzc_si128(_mm_cmpeq_epi16(r, y), one_mask);
            }
            j *= 8;
            for(; j < len1; ++j) count += (v1[j] == v2[i]);
        }
    }
    return(count);
}

uint64_t EXP_intersect_raw_rotl_gallop_sse4(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    if(len1 == 0 || len2 == 0) return 0;
    if(v1[0] > v2[len2-1] || v2[0] > v1[len1-1]) return 0;

    size_t count = 0;
    size_t i_a = 0, i_b = 0;
    const int vectorlength = sizeof(__m128i) / sizeof(uint16_t);
    const size_t st_a = (len1 / vectorlength) * vectorlength;
    const size_t st_b = (len2 / vectorlength) * vectorlength;
    __m128i v_a, v_b;
    const __m128i one_mask  = _mm_set1_epi16(1);
    const __m128i rotl_mask = _mm_set_epi8(13, 12, 11, 10, 9, 8,  7,  6,
                                            5,  4,  3,  2, 1, 0, 15, 14);
    __m128i rcount = _mm_setzero_si128();
    uint16_t buffer[8];

    if ((i_a < st_a) && (i_b < st_b)) {
        v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
        v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);

        while (1) {

#define UPDATE {                                        \
    v_b = _mm_shuffle_epi8(v_b, rotl_mask);             \
    res = _mm_or_si128(res, _mm_cmpeq_epi16(v_a, v_b)); \
}
            __m128i res = _mm_cmpeq_epi16(v_a, v_b);
            // 7 rotation comparisons
            UPDATE UPDATE UPDATE UPDATE
            UPDATE UPDATE UPDATE

#undef UPDATE

            //count += ((_mm_popcnt_u32(_mm_movemask_epi8(res))) >> 1); // option 1: popcnt of bit-mask
            rcount = _mm_add_epi16(rcount, _mm_and_si128(res, one_mask)); // option 2: horizontal accumulator

            const uint16_t a_max = v1[i_a + vectorlength - 1];
            const uint16_t b_max = v2[i_b + vectorlength - 1];

            if (a_max <= b_max) {
                i_a += vectorlength;
                if (i_a == st_a) break;
                v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
            }

            if (b_max <= a_max) {
                i_b += vectorlength;
                if (i_b == st_b) break;
                v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);
            }
        } // end while

        // uint16_t* c = (uint16_t*)&rcount;
        // for(int i = 0; i < vectorlength; ++i) count += c[i];
        _mm_storeu_si128((__m128i*)buffer, rcount);
        // uint16_t* c = (uint16_t*)&rcount;
        for(int i = 0; i < vectorlength; ++i) count += buffer[i];
    }

    // intersect the tail using scalar intersection
    while (i_a < len1 && i_b < len2) {
        uint16_t a = v1[i_a];
        uint16_t b = v2[i_b];
        if (a < b) {
            i_a++;
        } else if (b < a) {
            i_b++;
        } else {
            count++;
            i_a++;
            i_b++;
        }
    }
    return (uint64_t)count;
}
#endif

#if SIMD_VERSION >= 5
uint64_t EXP_intersect_raw_rotl_gallop_avx2(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    if(len1 == 0 || len2 == 0) return 0;
    if(v1[0] > v2[len2-1] || v2[0] > v1[len1-1]) return 0;

    size_t count = 0;
    size_t i_a = 0, i_b = 0;
    const int vectorlength = sizeof(__m256i) / sizeof(uint16_t);
    const size_t st_a = (len1 / vectorlength) * vectorlength;
    const size_t st_b = (len2 / vectorlength) * vectorlength;
    __m256i v_a, v_b;
    const __m256i one_mask  = _mm256_set1_epi16(1);
    const __m256i rotl_mask = _mm256_set_epi8(29, 28, 27, 26, 25, 24, 23, 22,
                                              21, 20, 19, 18, 17, 16, 15, 14,
                                              13, 12, 11, 10,  9,  8,  7,  6,
                                               5,  4,  3,  2,  1,  0, 31, 30);
    __m256i rcount = _mm256_setzero_si256();
    uint16_t buffer[16];

    if ((i_a < st_a) && (i_b < st_b)) {
        v_a = _mm256_lddqu_si256((__m256i *)&v1[i_a]);
        v_b = _mm256_lddqu_si256((__m256i *)&v2[i_b]);

        while (1) {
#define UPDATE {                                              \
    v_b = _mm256_shuffle_epi8(v_b, rotl_mask);                \
    res = _mm256_or_si256(res, _mm256_cmpeq_epi16(v_a, v_b)); \
}
            __m256i res = _mm256_cmpeq_epi16(v_a, v_b);
            // 15 rotation comparisons
            UPDATE UPDATE UPDATE UPDATE UPDATE
            UPDATE UPDATE UPDATE UPDATE UPDATE
            UPDATE UPDATE UPDATE UPDATE UPDATE

#undef UPDATE

            //count += ((_mm_popcnt_u32(_mm_movemask_epi8(res))) >> 1); // option 1: popcnt of bit-mask
            rcount = _mm256_add_epi16(rcount, _mm256_and_si256(res, one_mask)); // option 2: horizontal accumulator

            const uint16_t a_max = v1[i_a + vectorlength - 1];
            const uint16_t b_max = v2[i_b + vectorlength - 1];

            if (a_max <= b_max) {
                i_a += vectorlength;
                if (i_a == st_a) break;
                v_a = _mm256_lddqu_si256((__m256i *)&v1[i_a]);
            }

            if (b_max <= a_max) {
                i_b += vectorlength;
                if (i_b == st_b) break;
                v_b = _mm256_lddqu_si256((__m256i *)&v2[i_b]);
            }
        } // end while

        // uint16_t* c = (uint16_t*)&rcount;
        // for(int i = 0; i < vectorlength; ++i) count += c[i];
        _mm256_storeu_si256((__m256i*)buffer, rcount);
        // uint16_t* c = (uint16_t*)&rcount;
        for(int i = 0; i < vectorlength; ++i) count += buffer[i];
    }

    // intersect the tail using scalar intersection
    while (i_a < len1 && i_b < len2) {
        uint16_t a = v1[i_a];
        uint16_t b = v2[i_b];
        if (a < b) {
            i_a++;
        } else if (b < a) {
            i_b++;
        } else {
            count++;
            i_a++;
            i_b++;
        }
    }
    return (uint64_t)count;
}
#endif

#if SIMD_VERSION >= 3
uint64_t EXP_intersect_raw_sse4_broadcast_skip(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    uint64_t count = 0;
    const __m128i one_mask = _mm_set1_epi8(255);
    if(len1 < len2) { // broadcast-compare V1-values to vectors of V2 values
        if(v1[len1-1] < v2[0]) return 0;
        int from = 0;

        __m128i r;
        for(int i = 0; i < len1; ++i) {
            if(v1[i] < v2[0]) continue;
            if(v1[i] > v2[len2-1]) {
                // will never overlap
                return count;
            }
            r = _mm_set1_epi16(v1[i]);

            uint32_t loc = 0;
            int j = from;
            for(; j + 8 < len2; j += 8) {
                const __m128i y = _mm_loadu_si128((const __m128i*)&v2[j]);
                loc = _mm_testnzc_si128(_mm_cmpeq_epi16(r, y), one_mask);
                //loc = STORM_POPCOUNT((_mm_extract_epi64(res, 0) << 1) | _mm_extract_epi64(res, 1));
                from = (loc ? j + 1 : from);
                count += loc;
            }
            for(; j < len2; ++j) {
                count += (v1[i] == v2[j]);
                from = (v1[i] == v2[j] ? j + 1 : from);
            }
        }
    } else {
        if(v2[len2-1] < v1[0]) return 0;

        int from = 0;
        __m128i r;
        for(int i = 0; i < len2; ++i) {
            if(v2[i] < v1[0]) continue;
            if(v2[i] > v1[len1-1]) {
                // will never overlap
                return count;
            }
            r = _mm_set1_epi16(v2[i]);

            uint32_t loc = 0;
            int j = from;
            for(; j + 8 < len1; j += 8) {
                const __m128i y = _mm_loadu_si128((const __m128i*)&v1[j]);
                loc = _mm_testnzc_si128(_mm_cmpeq_epi16(r, y), one_mask);
                //loc = STORM_POPCOUNT((_mm_extract_epi64(res, 0) << 1) | _mm_extract_epi64(res, 1));
                from = (loc ? j + 1 : from);
                count += loc;
            }
            for(; j < len1; ++j) {
                count += (v2[i] == v1[j]);
                from = (v2[i] == v1[j] ? j + 1 : from);
            }
        }
    }
    return(count);
}
#endif

#if SIMD_VERSION >= 5
uint64_t EXP_intersect_raw_avx2_broadcast(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    uint64_t count = 0;
    const __m256i one_mask = _mm256_set1_epi16(1);
    if(len1 < len2) { // broadcast-compare V1-values to vectors of V2 values
        const uint32_t n_cycles = len2 / 16;

        for(int i = 0; i < len1; ++i) {
            if(v1[i] > v2[len2-1]) {
                // will never overlap
                return count;
            }
            __m256i r = _mm256_set1_epi16(v1[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m256i y = _mm256_loadu_si256((const __m256i*)&v2[j*16]);
                STORM_POPCOUNT_AVX2(count, _mm256_and_si256(_mm256_cmpeq_epi16(r, y), one_mask));
            }
            j *= 16;
            for(; j < len2; ++j) count += (v1[i] == v2[j]);
        }
    } else {
        const uint32_t n_cycles = len1 / 16;

        for(int i = 0; i < len2; ++i) {
            if(v2[i] > v1[len1-1]) {
                // will never overlap
                return count;
            }

            __m256i r = _mm256_set1_epi16(v2[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m256i y = _mm256_loadu_si256((const __m256i*)&v1[j*16]);
                STORM_POPCOUNT_AVX2(count, _mm256_and_si256(_mm256_cmpeq_epi16(r, y), one_mask));
            }
            j *= 16;
            for(; j < len1; ++j)  count += (v1[j] == v2[i]);
        }
    }
    return(count);
}
#endif


static int BinarySearch(const uint16_t* array, int n_a, uint16_t key) {
    if(n_a == 0) return -1;
    int low = 0, high = n_a-1, mid;
    while(low <= high) {
        mid = (low + high)/2;

        // low path
#if defined (__builtin_prefetch)
        __builtin_prefetch(&array[(mid + 1 + high)/2], 0, 1);
        // high path
        __builtin_prefetch(&array[(low + mid - 1)/2], 0, 1);
#endif
        if(array[mid] < key) low = mid + 1;
        else if(array[mid] == key) return mid;
        else if(array[mid] > key) high = mid - 1;
    }
    return -1;
 }

static void BinarySearch2(const uint16_t* array, int32_t n, uint16_t target1, uint16_t target2, int32_t* index1, int32_t* index2) {
    const uint16_t *base1 = array;
    const uint16_t *base2 = array;
    if (n == 0) return;
    while (n > 1) {
        int32_t half = n >> 1;
        base1 = (base1[half] < target1) ? &base1[half] : base1;
        base2 = (base2[half] < target2) ? &base2[half] : base2;
        n -= half;
    }
    *index1 = (int32_t)((*base1 < target1) + base1 - array);
    *index2 = (int32_t)((*base2 < target2) + base2 - array);
}

uint64_t EXP_intersect_raw_binary(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    uint64_t count = 0;

    if(len1 < len2) {
        for(int i = 0; i < len1; ++i) {
            if(v1[i] < v2[0]) {
                // will never overlap
                continue;
            }

            if(v1[i] > v2[len2-1]) {
                // will never overlap
                return count;
            }

            count += (BinarySearch(v2, len2, v1[i]) != -1);
        }
    } else {
        for(int i = 0; i < len2; ++i) {
            if(v2[i] < v1[0]) {
                // will never overlap
                continue;
            }

            if(v2[i] > v1[len1-1]) {
                // will never overlap
                return count;
            }

            count += (BinarySearch(v1, len1, v2[i]) != -1);
        }
    }
    return(count);
}

/*
low := 1
for i := 1 to m:
    diff := 1
    while low + diff <= n and A[low + diff] < B[i]:
        diff *= 2

    high := UNSAFE_MIN(n, low + diff)
    k = binary_search(A, low, high)
    if A[k] == B[i]:
        output B[i]

    low = k
 */
uint64_t EXP_intersect_raw_gallop(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    uint64_t count = 0;

    int low = 0;
    if(len1 < len2) {
        for(int i = 0; i < len1; ++i) {
            if(v1[i] < v2[0]) {
                // will never overlap
                continue;
            }

            if(v1[i] > v2[len2-1]) {
                // will never overlap
                return count;
            }

            int diff = 1;
            while(low + diff <= len2 && v2[low + diff] < v1[i])
                diff <<= 1;

            int high = UNSAFE_MIN((int)len2, low + diff);
            int k    = BinarySearch(&v2[low], high - low + 1, v1[i]);

            count += (k != -1);
            low = (k == -1 ? low : k);
        }
    } else {
        for(int i = 0; i < len2; ++i) {
            if(v2[i] < v1[0]) {
                // will never overlap
                continue;
            }

            if(v2[i] > v1[len1-1]) {
                // will never overlap
                return count;
            }

            int diff = 1;
            while(low + diff <= len1 && v1[low + diff] < v2[i])
                diff <<= 1;

            int high = UNSAFE_MIN((int)len1, low + diff);
            int k    = BinarySearch(&v1[low], high - low + 1, v2[i]);

            count += (k != -1);
            low = (k == -1 ? low : k);
        }
    }
    return(count);
}

#if SIMD_VERSION >= 3
uint64_t EXP_intersect_roaring_cardinality(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    size_t count = 0;
    size_t i_a = 0, i_b = 0;
    const int vectorlength = sizeof(__m128i) / sizeof(uint16_t);
    const size_t st_a = (len1 / vectorlength) * vectorlength;
    const size_t st_b = (len2 / vectorlength) * vectorlength;
    __m128i v_a, v_b;
    if ((i_a < st_a) && (i_b < st_b)) {
        v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
        v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);

        while ((v1[i_a] == 0) || (v2[i_b] == 0)) {
            const __m128i res_v = _mm_cmpestrm(
                v_b, vectorlength, v_a, vectorlength,
                _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
            const int r = _mm_extract_epi32(res_v, 0);
            count += _mm_popcnt_u32(r);
            const uint16_t a_max = v1[i_a + vectorlength - 1];
            const uint16_t b_max = v2[i_b + vectorlength - 1];

            if (a_max <= b_max) {
                i_a += vectorlength;
                if (i_a == st_a) break;
                v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
            }

            if (b_max <= a_max) {
                i_b += vectorlength;
                if (i_b == st_b) break;
                v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);
            }
        }

        if ((i_a < st_a) && (i_b < st_b))
            while (1) {
                const __m128i res_v = _mm_cmpistrm(
                    v_b, v_a,
                    _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
                const int r = _mm_extract_epi32(res_v, 0);
                count += _mm_popcnt_u32(r);
                const uint16_t a_max = v1[i_a + vectorlength - 1];
                const uint16_t b_max = v2[i_b + vectorlength - 1];

                if (a_max <= b_max) {
                    i_a += vectorlength;
                    if (i_a == st_a) break;
                    v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
                }

                if (b_max <= a_max) {
                    i_b += vectorlength;
                    if (i_b == st_b) break;
                    v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);
                }
            }
    }
    // intersect the tail using scalar intersection
    while (i_a < len1 && i_b < len2) {
        uint16_t a = v1[i_a];
        uint16_t b = v2[i_b];
        if (a < b) {
            i_a++;
        } else if (b < a) {
            i_b++;
        } else {
            count++;
            i_a++;
            i_b++;
        }
    }
    return (uint64_t)count;
}

uint64_t EXP_intersect_vector16_cardinality_roar(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2) {
    size_t count = 0;
    size_t i_a = 0, i_b = 0;
    const int vectorlength = sizeof(__m128i) / sizeof(uint16_t);
    const size_t st_a = (len1 / vectorlength) * vectorlength;
    const size_t st_b = (len2 / vectorlength) * vectorlength;
    __m128i v_a, v_b;
    if ((i_a < st_a) && (i_b < st_b)) {
        v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
        v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);
        while ((v1[i_a] == 0) || (v2[i_b] == 0)) {
            const __m128i res_v = _mm_cmpestrm(
                v_b, vectorlength, v_a, vectorlength,
                _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
            const int r = _mm_extract_epi32(res_v, 0);
            count += _mm_popcnt_u32(r);
            const uint16_t a_max = v1[i_a + vectorlength - 1];
            const uint16_t b_max = v2[i_b + vectorlength - 1];
            if (a_max <= b_max) {
                i_a += vectorlength;
                if (i_a == st_a) break;
                v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
            }
            if (b_max <= a_max) {
                i_b += vectorlength;
                if (i_b == st_b) break;
                v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);
            }
        }
        if ((i_a < st_a) && (i_b < st_b))
            while (1) {
                const __m128i res_v = _mm_cmpistrm(
                    v_b, v_a,
                    _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
                const int r = _mm_extract_epi32(res_v, 0);
                count += _mm_popcnt_u32(r);
                const uint16_t a_max = v1[i_a + vectorlength - 1];
                const uint16_t b_max = v2[i_b + vectorlength - 1];
                if (a_max <= b_max) {
                    i_a += vectorlength;
                    if (i_a == st_a) break;
                    v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
                }
                if (b_max <= a_max) {
                    i_b += vectorlength;
                    if (i_b == st_b) break;
                    v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);
                }
            }
    }
    // intersect the tail using scalar intersection
    while (i_a < len1 && i_b < len2) {
        uint16_t a = v1[i_a];
        uint16_t b = v2[i_b];
        if (a < b) {
            i_a++;
        } else if (b < a) {
            i_b++;
        } else {
            count++;
            i_a++;
            i_b++;
        }
    }
    return (uint64_t)count;
}
#endif