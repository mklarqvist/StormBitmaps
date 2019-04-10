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
        if ((i_a < st_a) && (i_b < st_b))
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

struct base {
    base() : len(0), vals(nullptr){}
    ~base() { delete[] vals; }

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
                buckets  = std::unique_ptr<std::vector<base>>(new std::vector<base>(n_bins));
                bitmap_bucket = std::unique_ptr<uint64_t[]>(new uint64_t[(n_bins / 64)+1]{0});
                bitmap_type   = std::unique_ptr<uint64_t[]>(new uint64_t[(n_bins / 64)+1]{0});
                
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
                            reinterpret_cast<array*>(&buckets->at(prev_bin))->build(l_val.data(), l_val.size());
                            // Set type to 0
                            //bitmap_type.get()[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                            // Set the bin position in the presence/absence bitmap.
                            bitmap_bucket.get()[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                        } 
                        // Construct bitmap
                        else {
                            // std::cerr << "construct bitmap: " << l_val.size() << std::endl;
                            // Construct array archetype and add values to it.
                            reinterpret_cast<bitmap*>(&buckets->at(prev_bin))->build(l_val.data(), l_val.size(), (bin_size / 64)+1);
                            // Set type to 1
                            bitmap_type.get()[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                            // Set the bin position in the presence/absence bitmap.
                            bitmap_bucket.get()[prev_bin / 64] |= (1ULL << (prev_bin % 64));
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
                        reinterpret_cast<array*>(&buckets->at(prev_bin))->build(l_val.data(), l_val.size());
                        // Set type to 0
                        //bitmap_type.get()[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                        // Set the bin position in the presence/absence bitmap.
                        bitmap_bucket.get()[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                    } 
                    // Construct bitmap
                    else {
                        // std::cerr << "construct bitmap: " << l_val.size() << std::endl;
                        // Construct array archetype and add values to it.
                        reinterpret_cast<bitmap*>(&buckets->at(prev_bin))->build(l_val.data(), l_val.size(), (bin_size / 64)+1);
                        // Set type to 1
                        bitmap_type.get()[prev_bin / 64] |= (1ULL << (prev_bin % 64));
                        // Set the bin position in the presence/absence bitmap.
                        bitmap_bucket.get()[prev_bin / 64] |= (1ULL << (prev_bin % 64));
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
        
        
        for (int i = 0; i < (n_bins / 64)+1; ++i) {
            uint64_t diff = this->bitmap_bucket.get()[i] & other.bitmap_bucket.get()[i];
            // std::cerr <<"[][]diff=" << std::bitset<64>(diff) << std::endl;

            // temp
            while (diff) {
                // uint32_t offset = __builtin_clzll(diff);
#ifdef _lzcnt_u64
                uint32_t offset = _lzcnt_u64(diff);
#else
                uint32_t offset = __builtin_clzl(diff);
#endif
                uint32_t target_bin = 64 - offset;
                // assert(target_bin != 0);
                // std::cerr << " " << (int)target_bin << " " << std::bitset<64>(diff) << std::endl; 

                // do intersection here
                // std::cerr << "@" << i << " type=" << (bitmap_type.get()[i] & (1ULL << (target_bin-1)) != 0) << " is=" << std::bitset<64>((1ULL << (target_bin-1))) << " and " << std::bitset<64>(bitmap_type.get()[i]) << std::endl;
                // std::cerr << "target_bin=" << target_bin << ">>" << std::bitset<64>(1ULL << (target_bin-1)) << std::endl;
                // std::cerr << "type1=" << std::bitset<64>(bitmap_type.get()[i]) << ">>" << std::bitset<64>(bitmap_type.get()[i] & (1ULL << (target_bin-1))) << std::endl;
                // std::cerr << "type2=" << std::bitset<64>(other.bitmap_type.get()[i]) << ">>" << std::bitset<64>(other.bitmap_type.get()[i] & (1ULL << (target_bin-1))) << std::endl;
                bool type1 = (bitmap_type.get()[i] & (1ULL << (target_bin-1))) != 0;
                bool type2 = (other.bitmap_type.get()[i] & (1ULL << (target_bin-1))) != 0;

                // std::cerr << "type1=" << type1 << ",type2=" << type2 << std::endl;

                if (type1 && type2) {
                    overlap += IntersectCount(reinterpret_cast<bitmap*>(&buckets->at(target_bin-1)), reinterpret_cast<bitmap*>(&other.buckets->at(target_bin-1)));
                } else if (!type1 && !type2) {
                    overlap += IntersectCount(reinterpret_cast<array*>(&buckets->at(target_bin-1)), reinterpret_cast<array*>(&other.buckets->at(target_bin-1)));
                } else if (!type1 && type2) {
                    overlap += IntersectCount(reinterpret_cast<array*>(&buckets->at(target_bin-1)), reinterpret_cast<bitmap*>(&other.buckets->at(target_bin-1)));
                } else if (type1 && !type2) {
                    overlap += IntersectCount(reinterpret_cast<bitmap*>(&buckets->at(target_bin-1)), reinterpret_cast<array*>(&other.buckets->at(target_bin-1)));
                }
                else {
                    std::cerr << "illegal" << std::endl;
                    // exit(1);
                }

                // std::cerr << "overlap=" << overlap << std::endl;

                // unset bit
                // std::cerr << "mask out=" << std::bitset<64>(~(1ULL << (target_bin-1))) << std::endl;
                diff &= ~(1ULL << (target_bin-1));
            }
            // std::cerr << "done while" << std::endl;
        }

        return overlap;
    }

private:
    // Internal.
    uint64_t IntersectCount(const array* __restrict__ s1,  const array* __restrict__ s2)  const {
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

    uint64_t IntersectCount(const bitmap* __restrict__ s1, const bitmap* __restrict__ s2) const {
        // assert(s1->len == s2->len);
#if SIMD_VERSION >= 5
        uint64_t ret = intersect_bitmaps_avx2((uint64_t*)s1->vals, (uint64_t*)s2->vals, s1->len);
#elif SIMD_VERSION >= 3
        uint64_t ret = intersect_bitmaps_sse4((uint64_t*)s1->vals, (uint64_t*)s2->vals, s1->len);
#endif
        return(ret);
    }

    uint64_t IntersectCount(const array* s1,  const bitmap* s2) const {
        // std::cerr << "in array->bitmap intersect. use fingering" << std::endl;
        return 0;
    }
    // Alias for the mirrored prototype (array, bitmap).
    uint64_t IntersectCount(const bitmap* s1, const array* s2) const { return IntersectCount(s2,s1); }

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
    std::unique_ptr<uint64_t[]> bitmap_bucket; // Bitmap used for querying if a particular bucket contains values.
    std::unique_ptr<uint64_t[]> bitmap_type; // 0 -> array, 1-> bitmap.
    std::unique_ptr< std::vector<base> > buckets; // Must be cast to either the array or bitmap archetype.
    // Special case when there are very few values limited to the range [0,65536)
    // then we store those as array literals in a special vector.
    std::unique_ptr<array> array_global; // Unbucketed.
};

#endif /* FAST_INTERSECT_COUNT_H_ */