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
#ifndef FAST_INTERSECT_CLASSES_COUNT_H_
#define FAST_INTERSECT_CLASSES_COUNT_H_

#include "fast_intersect_count.h"

#include <vector>
#include <memory> //unique_ptr
#include <iostream>//debug
#include <bitset> //debug

static uint64_t intersect_vector16_cardinality_roar(const uint16_t* __restrict__ v1, const ssize_t len1, const uint16_t* __restrict__ v2, const ssize_t len2) {
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
        if ((i_a < st_a) && (i_b < st_b)) {
            while (true) {
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

/////// START

struct base {
    base() : len(0), vals(nullptr){}
    ~base() { 
        delete[] vals; 
    }

    uint8_t control; // control sequence.
    size_t len; // length in either 16 bits or 64 bits depending on the downstream container archetype
    uint8_t* vals;
};

// Array of literal 16-bit integers.
struct array : public base {
    array(const uint32_t* input_vals, const ssize_t input_len) {
        len = input_len;
        vals = (uint8_t*)new uint16_t[len];
        for (int i = 0; i < len; ++i) {
            assert(input_vals[i] < 65536);
            vals[i] = input_vals[i];
        }
    }

    void build(const uint32_t* input_vals, const ssize_t input_len) {
        // Remove previous.
        delete[] vals;

        len = input_len;
        // vals = (uint8_t*)new uint16_t[len];
        assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, len*sizeof(uint16_t)));
        for (int i = 0; i < len; ++i) {
            // assert(input_vals[i] < 65536);
            reinterpret_cast<uint16_t*>(vals)[i] = input_vals[i];
            assert(reinterpret_cast<uint16_t*>(vals)[i] == input_vals[i]);
            // std::cerr << "," << reinterpret_cast<uint16_t*>(vals)[i] << "/" << input_vals[i];
        }
        // std::cerr << std::endl;
    }
};

// Bitmap of packed integers;
struct bitmap : public base {
    void build(const uint32_t* input_vals, const ssize_t input_len, const ssize_t bitmap_length) {
        // Remove previous.
        delete[] vals;

        len = bitmap_length;
        // vals = (uint8_t*)new uint64_t[bitmap_length];
        assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, bitmap_length*sizeof(uint64_t)));
        memset(vals, 0, bitmap_length*sizeof(uint64_t));
        uint64_t* bitmaps = (uint64_t*)vals;

        for (int i = 0; i < input_len; ++i) {
            // assert(input_vals[i] < 8196);
            bitmaps[input_vals[i] / 64] |= (1ULL << (input_vals[i] % 64));
            //assert(reinterpret_cast<uint16_t*>(vals)[i] == input_vals[i]);
            // std::cerr << "," << std::bitset<64>(bitmaps[input_vals[i]/64]) << "/" << input_vals[i] << std::endl;
        }
        // std::cerr << std::endl;
    }
};

class IntersectContainer {
public:
    IntersectContainer() : n_bins(0), bin_size(0){}
    ~IntersectContainer() {
        delete[] buckets;
        delete[] bitmap_bucket;
        delete[] bitmap_type;
    }

    void construct(const uint32_t* vals, const ssize_t len, ssize_t max_value) {
        // cleanup if repeatedly calling construct
        is_local = false;
        bool val_exceed_16bit = false;
        n_entries = len;

        // Fixme: this is incorrect
        // for (int i = 0; i < len; ++i) val_exceed_16bit += (len >= 65536);

        // val did not exceed 16-bit limit.
        if (val_exceed_16bit == false) {
            // if (len < 32 || max_value < 128) { // all values < 2^16 and len < 32 OR max_value < 512
            //     is_local = true;
            //     std::cerr << "constructing local array: " << len << "/" << max_value << std::endl;
            //     array_global = std::make_shared<array>(vals, len);
            // } else 
            { // all values < 2^16 but len >= 64

                // 1) bucket range
                bin_size = 8192;
                n_bins   = (max_value / bin_size) + 1;
                buckets  = new base[n_bins];
                bitmap_bucket = new uint64_t[(n_bins / 64)+1];
                bitmap_type   = new uint64_t[(n_bins / 64)+1];
                memset(bitmap_bucket, 0, sizeof(uint64_t)*((n_bins / 64)+1));
                memset(bitmap_type,   0, sizeof(uint64_t)*((n_bins / 64)+1));
                
                // fixme
                // pre-test buckets
                // uint32_t* n_bu = new uint32_t[n_bins];
                // memset(n_bu, 0, sizeof(uint32_t)*n_bins);

                // std::cerr << "Number of bins=" << n_bins << " with bin-size=" << bin_size << std::endl;

                // for (int i = 0; i < len; ++i) {
                //     ++n_bu[vals[i] / bin_size];
                    // bu_vals[vals[i]/ bin_size].push_back(vals[i]);
                //}

                // Local construction.
                uint32_t prev_bin = vals[0] / bin_size;
                uint32_t prev_idx = 0;

                std::vector<uint32_t> l_val;
                l_val.push_back(vals[0]);

                for (int i = 1; i < len; ++i) {
                    if (vals[i] / bin_size != prev_bin) {
                        // std::cerr << "bin-" << prev_bin << ": " << prev_idx << "->" << i << "(" << i-prev_idx << "==" << l_val.size() << ")" << std::endl;
                        
                        // Adjust values such that positions in this bin start at 0.
                        uint32_t start_pos_bin = prev_bin * bin_size;
                        for (int j = 0; j < l_val.size(); ++j) {
                            // std::cerr << l_val[j] << "/" << start_pos_bin << "->" << l_val[j] - start_pos_bin << std::endl;
                            assert(l_val[j] >= start_pos_bin);
                            l_val[j] -= start_pos_bin;
                        }

                        // Construct array
                        if (l_val.size() < 64)
                        {
                            // std::cerr << "construct array: " << l_val.size() << std::endl;
                            
                            // Construct array archetype and add values to it.
                            reinterpret_cast<array*>(&buckets[prev_bin])->build(l_val.data(), l_val.size());
                            // Set type to 0
                            //bitmap_type.get()[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                            // Set the bin position in the presence/absence bitmap.
                            bitmap_bucket[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                        } 
                        // Construct bitmap
                        else {
                            // std::cerr << "construct bitmap: " << l_val.size() << std::endl;
                            // Construct array archetype and add values to it.
                            reinterpret_cast<bitmap*>(&buckets[prev_bin])->build(l_val.data(), l_val.size(), (bin_size / 64)+1);
                            // Set type to 1
                            bitmap_type[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                            // Set the bin position in the presence/absence bitmap.
                            bitmap_bucket[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                        }

                        // Update and reset values.
                        prev_bin = vals[i] / bin_size;
                        prev_idx = i;
                        l_val.clear();
                    }
                    l_val.push_back(vals[i]);
                }

                if (prev_idx != len) {
                    // std::cerr << "bin-" << prev_bin << ": " << prev_idx << "->" << len << "(" << len-prev_idx << "==" << l_val.size() << ")" << std::endl;
                    
                    // Adjust values such that positions in this bin start at 0.
                    uint32_t start_pos_bin = prev_bin * bin_size;
                    for (int j = 0; j < l_val.size(); ++j) {
                        // std::cerr << l_val[j] << "/" << start_pos_bin << "->" << l_val[j] - start_pos_bin << std::endl;
                        assert(l_val[j] >= start_pos_bin);
                        l_val[j] = l_val[j] - start_pos_bin;
                    }

                    // Construct array
                    if (l_val.size() < 64)
                    {
                        // std::cerr << "construct array: " << l_val.size() << std::endl;
                        
                        // Construct array archetype and add values to it.
                        reinterpret_cast<array*>(&buckets[prev_bin])->build(l_val.data(), l_val.size());
                        // Set type to 0
                        //bitmap_type.get()[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                        // Set the bin position in the presence/absence bitmap.
                        bitmap_bucket[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                    } 
                    // Construct bitmap
                    else {
                        // std::cerr << "construct bitmap: " << l_val.size() << std::endl;
                        // Construct array archetype and add values to it.
                        reinterpret_cast<bitmap*>(&buckets[prev_bin])->build(l_val.data(), l_val.size(), (bin_size / 64)+1);
                        // Set type to 1
                        bitmap_type[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                        // Set the bin position in the presence/absence bitmap.
                        bitmap_bucket[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                    }
                }

                // for (int i = 0; i < n_bins; ++i) {
                //     std::cerr << "bucket-" << i<< ": " << (n_bu[i] < 64 ? "array" : "bitmap") << " with " << n_bu[i] << std::endl;
                // }

                // delete[] n_bu;

                // std::cerr << "map=" << std::bitset<64>(bitmap_bucket.get()[0]) << std::endl;
                // std::cerr << "types=" << std::bitset<64>(bitmap_type.get()[0]) << std::endl;
            }
        } else {
            std::cerr << "here in else" << std::endl;
            exit(1);
        }
    }

    /**
     * @brief Compute the cardinality of the resulting intersection between this set and the given set.
     * 
     * @param other 
     * @return 
     */
    uint64_t IntersectCount(const IntersectContainer& other) const {
        uint64_t overlap = 0;
        
        // for (int i = 0; i < n_bins; ++i) {
            
        //     overlap += IntersectCount(reinterpret_cast<array*>(&(*buckets)[i]), reinterpret_cast<array*>(&(*other.buckets)[i]));
        // }  
        // return(overlap);
        
        const uint32_t bb = (n_bins / 64)+1;
        for (int i = 0; i < bb; ++i) {
            uint64_t diff = this->bitmap_bucket[i] & other.bitmap_bucket[i];
            if (!diff) continue;
            // std::cerr <<"[][]diff=" << std::bitset<64>(diff) << std::endl;

            uint32_t target_bin = 0;
            uint32_t offset = 0;
            while (diff) {
                // uint32_t offset = __builtin_clzll(diff);
#ifdef _lzcnt_u64
                offset = _lzcnt_u64(diff);
#else
                offset = __builtin_clzl(diff);
#endif
                target_bin = 64 - offset;
                // assert(target_bin != 0);
                // std::cerr << " " << (int)target_bin << " " << std::bitset<64>(diff) << std::endl; 

                // do intersection here
                // std::cerr << "@" << i << " type=" << (bitmap_type.get()[i] & (1ULL << (target_bin-1)) != 0) << " is=" << std::bitset<64>((1ULL << (target_bin-1))) << " and " << std::bitset<64>(bitmap_type.get()[i]) << std::endl;
                // std::cerr << "target_bin=" << target_bin << ">>" << std::bitset<64>(1ULL << (target_bin-1)) << std::endl;
                // std::cerr << "type1=" << std::bitset<64>(bitmap_type.get()[i]) << ">>" << std::bitset<64>(bitmap_type.get()[i] & (1ULL << (target_bin-1))) << std::endl;
                // std::cerr << "type2=" << std::bitset<64>(other.bitmap_type.get()[i]) << ">>" << std::bitset<64>(other.bitmap_type.get()[i] & (1ULL << (target_bin-1))) << std::endl;
                
                // bool type1 = (bitmap_type.get()[i] & (1ULL << (target_bin-1))) != 0;
                // bool type2 = (other.bitmap_type.get()[i] & (1ULL << (target_bin-1))) != 0;
                const uint32_t target = target_bin-1;
                const uint8_t ref = (((bitmap_type[i] & (1ULL << target)) != 0) << 1) | ((other.bitmap_type[i] & (1ULL << target)) != 0);

                // std::cerr << "type1=" << type1 << ",type2=" << type2 << std::endl;
                switch(ref){
                    case 0: overlap += IntersectCount(reinterpret_cast<const array*>(&buckets[target]),  reinterpret_cast<const array*>(&other.buckets[target])); break;
                    case 1: overlap += IntersectCount(reinterpret_cast<const bitmap*>(&buckets[target]), reinterpret_cast<const array*>(&other.buckets[target])); break;
                    case 2: overlap += IntersectCount(reinterpret_cast<const array*>(&buckets[target]),  reinterpret_cast<const bitmap*>(&other.buckets[target])); break;
                    case 3: overlap += IntersectCount(reinterpret_cast<const bitmap*>(&buckets[target]), reinterpret_cast<const bitmap*>(&other.buckets[target])); break;
                }
                
                // else {
                //     std::cerr << "illegal" << std::endl;
                //     // exit(1);
                // }

                // std::cerr << "overlap=" << overlap << std::endl;

                // unset bit
                // std::cerr << "mask out=" << std::bitset<64>(~(1ULL << (target_bin-1))) << std::endl;
                diff &= ~(1ULL << target);
            }
            // std::cerr << "done while" << std::endl;
        }

        return overlap;
    }

private:
    // Internal.
    static inline uint64_t IntersectCount(const array* __restrict__ s1,  const array* __restrict__ s2) {
        // Debug prints.
        // for (int i = 0; i < s1->len; ++i) {
        //     std::cerr << "," << reinterpret_cast<uint16_t*>(s1->vals)[i];
        // }
        // std::cerr << std::endl;

        // for (int i = 0; i < s2->len; ++i) {
        //     std::cerr << "," << reinterpret_cast<uint16_t*>(s2->vals)[i];
        // }
        // std::cerr << std::endl;

        // temp removed
        if (s1->len == 0 || s2->len == 0) return 0;
        // if (s1->vals[0] > s2->vals[s2->len - 1]) return 0;
        // if (s2->vals[0] > s1->vals[s1->len - 1]) return 0;
        
        // 1,2,3,4,5
        // 6,7,8,9,10

        return(intersect_vector16_cardinality_roar(reinterpret_cast<uint16_t*>(s1->vals), s1->len, reinterpret_cast<uint16_t*>(s2->vals), s2->len));
    }

    static inline uint64_t IntersectCount(const bitmap* __restrict__ s1, const bitmap* __restrict__ s2) {
        // assert(s1->len == s2->len);
#if SIMD_VERSION >= 5
        return(intersect_bitmaps_avx2((uint64_t*)s1->vals, (uint64_t*)s2->vals, s1->len));
#elif SIMD_VERSION >= 3
        return(intersect_bitmaps_sse4((uint64_t*)s1->vals, (uint64_t*)s2->vals, s1->len));
#endif
        return(0);
    }

    static inline uint64_t IntersectCount(const array* s1,  const bitmap* s2) {
        // std::cerr << "in array->bitmap intersect. use fingering" << std::endl;
        return 0;
    }
    // Alias for the mirrored prototype (array, bitmap).
    static inline uint64_t IntersectCount(const bitmap* s1, const array* s2) { return IntersectCount(s2,s1); }

    // Second set is a bitmap.
    uint64_t IntersectCountGlobal(const bitmap* s2) const;
    // Second set is normal array.
    uint64_t IntersectCountGlobal(const array* s2) const;
    // Second set is also global.
    uint64_t IntersectCountGlobalGlobal(const array* s2) const;

public:
    bool is_local;
    ssize_t n_entries;
    ssize_t n_bins;
    ssize_t bin_size; // variable sized (unlike Roaring)
    uint64_t* bitmap_bucket; // Bitmap used for querying if a particular bucket contains values.
    uint64_t* bitmap_type; // 0 -> array, 1-> bitmap.
    base* buckets; // Must be cast to either the array or bitmap archetype.
    // Special case when there are very few values limited to the range [0,65536)
    // then we store those as array literals in a special vector.
    std::unique_ptr<base> base_global; // Unbucketed.
};

///// new
struct bitmap_t {
    bitmap_t() : n_set(0), n_bitmap(0), own(false), data(nullptr) {}
    bitmap_t(uint64_t* in, uint32_t n, uint32_t m) : n_set(n), n_bitmap(m), own(false), data(in) {}
    ~bitmap_t() {
        if (own) aligned_free_port(data);
    }

    int Allocate(uint32_t n) {
        if (data == nullptr) {
            n_bitmap = n;
            data  = (uint64_t*)aligned_malloc_port(SIMD_ALIGNMENT, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        } else {
            if (own) aligned_free_port(data);
            n_bitmap = n;
            data  = (uint64_t*)aligned_malloc_port(SIMD_ALIGNMENT, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        }
        memset(data,0,n_bitmap*sizeof(uint64_t));
        return n;
    }

    int AllocateSamples(uint32_t n) {
        const uint64_t n_vals = ceil(n / 64.0);
        if (data == nullptr) {
            n_bitmap = n_vals;
            data  = (uint64_t*)aligned_malloc_port(SIMD_ALIGNMENT, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        } else {
            if (own) aligned_free_port(data);
            n_bitmap = n_vals;
            data  = (uint64_t*)aligned_malloc_port(SIMD_ALIGNMENT, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        }
        memset(data,0,n_bitmap*sizeof(uint64_t));
        return n;
    }

    inline void Add(const uint64_t pos) { data[pos / 64] |= 1ULL << (pos % 64); }

    void clear() {
        memset(data, 0, n_bitmap*sizeof(uint64_t));
    }

    uint64_t intersect(const bitmap_t& other) const;

    // uint32_t intersect_count(const bitmap_t& other) const {
    //     return intersect_bitmaps_avx512_csa(data, other.data, n_bitmap);
    // }

    uint32_t n_set, n_bitmap: 31, own: 1; // number of values set, number of bitmaps, ownership
    uint64_t* data;
};

struct bitmap_container_t {
    bitmap_container_t(uint32_t n, uint32_t m) : 
        n_alt_cutoff(0),
        n_bitmaps(n), 
        n_samples(m), 
        own(true), 
        type(0),
        n_bitmaps_sample(ceil(n_samples / 64.0)),
        bmaps(nullptr), 
        bitmaps(new bitmap_t[n]),
        n_alts_tot(0), m_alts(0),
        alt_positions(nullptr),
        alt_offsets(nullptr),
        n_alts(nullptr)
    {
        for (int i = 0; i < n_bitmaps; ++i) {
            bitmaps[i].AllocateSamples(n_samples);
        }
    }

    bitmap_container_t(uint32_t n, uint32_t m, bool yes) : 
        n_alt_cutoff(0),
        n_bitmaps(n), 
        n_samples(m), 
        own(true), 
        type(1),
        n_bitmaps_sample(ceil(n_samples / 64.0)),
        bmaps((uint64_t*)aligned_malloc_port(SIMD_ALIGNMENT, n_bitmaps*n_bitmaps_sample*sizeof(uint64_t))), 
        bitmaps(nullptr),
        n_alts_tot(0), m_alts(0),
        alt_positions(nullptr),
        alt_offsets(nullptr),
        n_alts((uint32_t*)aligned_malloc_port(SIMD_ALIGNMENT, n_bitmaps*sizeof(uint32_t)))
    {
        memset(bmaps,0,n_bitmaps*n_bitmaps_sample*sizeof(uint64_t));
        memset(n_alts,0,n_bitmaps*sizeof(uint32_t));
        // for (int i = 0; i < n_bitmaps; ++i) {
        //     bitmaps[i].data = &bmaps[i*n_bitmaps_sample];
        //     bitmaps[i].own = false;
        //     bitmaps[i].n_set = 0;
        //     bitmaps[i].n_bitmap = n_bitmaps_sample;
        // }
    }

    bitmap_container_t(uint32_t n, uint32_t m, bool yes, bool yes2) : 
        n_alt_cutoff(300),
        n_bitmaps(n), 
        n_samples(m), 
        own(true), 
        type(1),
        n_bitmaps_sample(ceil(n_samples / 64.0)),
        bmaps((uint64_t*)aligned_malloc_port(SIMD_ALIGNMENT, n_bitmaps*n_bitmaps_sample*sizeof(uint64_t))), 
        bitmaps(nullptr),
        alt_positions(nullptr),
        alt_offsets((uint32_t*)aligned_malloc_port(SIMD_ALIGNMENT, n_bitmaps*sizeof(uint32_t))),
        n_alts((uint32_t*)aligned_malloc_port(SIMD_ALIGNMENT, n_bitmaps*sizeof(uint32_t)))
    {
        memset(bmaps,0,n_bitmaps*n_bitmaps_sample*sizeof(uint64_t));
        memset(n_alts,0,n_bitmaps*sizeof(uint32_t));
        // for (int i = 0; i < n_bitmaps; ++i) {
        //     bitmaps[i].data = &bmaps[i*n_bitmaps_sample];
        //     bitmaps[i].own = false;
        //     bitmaps[i].n_set = 0;
        //     bitmaps[i].n_bitmap = n_bitmaps_sample;
        // }
    }

    ~bitmap_container_t() {
        if (own) {
            delete[] bitmaps;
            aligned_free_port(bmaps);
        }
        aligned_free_port(alt_offsets);
        aligned_free_port(n_alts);
        aligned_free_port(alt_positions);
    }

    void Add(const uint32_t target, uint32_t value) { 
        if (type == 0) {
            assert(bitmaps!=nullptr);
            bitmaps[target].Add(value); 
        }
        else {
            assert(bmaps!=nullptr);
            uint64_t* x = &bmaps[target*n_bitmaps_sample];
            x[value / 64] |= 1ULL << (value % 64);
            ++n_alts[target]; // increment the number of alts for this site
        }
    }

    void Add(const uint32_t target, const std::vector<uint32_t>& values) { 
        if (type == 0) {
            assert(bitmaps!=nullptr);
            for (int i = 0; i < values.size(); ++i)
                bitmaps[target].Add(values[i]); 
        }
        else { // if type is 1
            assert(bmaps != nullptr);

            uint64_t* x = &bmaps[target*n_bitmaps_sample];
            for (int i = 0; i < values.size(); ++i) {
                x[values[i] / 64] |= 1ULL << (values[i] % 64);
            }
            n_alts[target] = values.size(); // set the number of alts for this site

            alt_offsets[target] = n_alts_tot; // always set offset

            // todo: fix
            if (values.size() < n_alt_cutoff) {
                // resize if required
                if (n_alts_tot + values.size() >= m_alts || alt_positions == nullptr) {
                    uint32_t* old = alt_positions;
                    const uint32_t add = values.size() < 65535 ? 65535 : values.size() * 5;
                    uint32_t new_pos = (n_alts_tot == 0 ? add : n_alts_tot + add);
                    // std::cerr << "resizing: " << n_alts_tot << "->" << new_pos << std::endl;
                    alt_positions = (uint32_t*)aligned_malloc_port(SIMD_ALIGNMENT, new_pos*sizeof(uint32_t));
                    memcpy(alt_positions, old, n_alts_tot*sizeof(uint32_t));
                    m_alts = new_pos;
                    aligned_free_port(old);
                }

                // std::cerr << "adding: " << values.size() << " at " << alt_offsets[target] << std::endl;
    
                uint32_t* tgt = &alt_positions[alt_offsets[target]];
                for (int i = 0; i < values.size(); ++i) {
                    // std::cerr << values[i] << std::endl;
                    tgt[i] = values[i];
                }
                n_alts_tot += values.size();
            }
        }
    }
    
    void clear() {
        if (type == 0) {
        for (int i = 0; i < n_bitmaps; ++i)
            bitmaps[i].clear();
        } else {
            memset(bmaps,0,n_bitmaps*n_bitmaps_sample*sizeof(uint64_t));
            if (n_alts != nullptr) memset(n_alts,0,n_bitmaps*sizeof(uint32_t));
            if (alt_positions != nullptr) memset(alt_positions,0,m_alts*sizeof(uint32_t));
            if (alt_offsets != nullptr) memset(alt_offsets,0,n_bitmaps*sizeof(uint32_t));
            n_alts_tot = 0;
        }
    }

    uint64_t intersect() const;
    uint64_t intersect_cont() const;
    uint64_t intersect_blocked(uint32_t bsize) const;
    uint64_t intersect_blocked_cont(uint32_t bsize) const;
    uint64_t intersect_cont_auto() const;
    uint64_t intersect_cont_blocked_auto(uint32_t bsize) const;

    uint32_t n_alt_cutoff;
    uint32_t n_bitmaps, n_samples: 30, own: 1, type: 1;
    uint32_t n_bitmaps_sample;
    uint64_t* bmaps;
    bitmap_t* bitmaps;
    //
    uint32_t n_alts_tot, m_alts;
    uint32_t* alt_positions; // positions of alts at a site
    uint32_t* alt_offsets; // virtual offsets to start of alt_positions for a site
    uint32_t* n_alts; // number of alts at a position
};

struct roaring2_t {
    roaring2_t(uint32_t n_s, uint32_t n_ss) : n_samples(n_s), n_sites(n_ss), block_size(65535), n_total_blocks(0),
        n_total_bitmaps(0), m_blocks(0), m_bitmaps(0), n_blocks(nullptr), blocks(nullptr), data_bitmaps(nullptr)
    {}

    ~roaring2_t() {
        aligned_free_port(n_blocks);
        aligned_free_port(blocks);
        aligned_free_port(data_bitmaps);
    }

    int Add(const uint32_t target, const std::vector<uint32_t>& pos) {
        if (pos.size() == 0) return 0;

        if (n_blocks == nullptr) {
            n_blocks = (uint32_t*)aligned_malloc_port(SIMD_ALIGNMENT, n_sites*sizeof(uint32_t));
        }

        if (blocks == nullptr) {
            m_blocks = n_blocks_site * 25;
            blocks = (uint16_t*)aligned_malloc_port(SIMD_ALIGNMENT, m_blocks*sizeof(uint16_t));
            n_total_blocks = 0;
        }

        if (data_bitmaps == nullptr) {
            m_bitmaps = 65535;
            data_bitmaps = (uint64_t*)aligned_malloc_port(SIMD_ALIGNMENT, m_bitmaps*sizeof(uint64_t));
            memset(data_bitmaps, 0, m_bitmaps*sizeof(uint64_t));
            n_total_bitmaps = 0;
        }


        // First target block
        uint32_t current_block = pos[0] / 8192; // block_size
        // Allocate
        uint64_t* tgt_bitmap = &data_bitmaps[n_total_bitmaps];
        n_total_bitmaps += 128; // 8192 bits = 16 * 512
        
        for (int i = 0; i < pos.size(); ++i) {
            uint32_t adj_pos = pos[i] - (current_block*8192);
            tgt_bitmap[adj_pos / 64] |= 1ULL << (adj_pos % 64);
        }

        return 1;
    }

    // N containers (blocks) per site
    // each container share memory region
    uint32_t n_samples; // number of samples per site
    uint32_t n_sites; // number of sites
    uint32_t block_size; // 8192 values per block by default
    uint32_t n_blocks_site; // Maximum number of blocks for a site
    uint32_t n_total_blocks; // total sum of blocks
    uint32_t n_total_bitmaps; // total sum for data_bitmaps
    uint32_t m_blocks, m_bitmaps; // allocation for blocks,bitmaps
    uint32_t* n_blocks; // number of set blocks per site (fixed size of n_sites)
    uint16_t* blocks; // block id (count order)
    uint64_t* data_bitmaps; // shared array for bitmap data
};


#endif /* FAST_INTERSECT_COUNT_H_ */