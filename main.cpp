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

    void Allocate(const uint8_t n) {
        delete[] vals;
        assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, n*sizeof(uint64_t)));
        memset(vals, 0, n*sizeof(uint64_t));
        n_vals = n;
    }

    inline const uint64_t& operator[](const uint8_t p) const { return(vals[p]); }

    bool list;
    uint8_t n_vals; // limited to 64 uint64_t
    uint32_t n_list;
    uint64_t bitmap; // bitmap of bitmaps (equivalent to squash)
    uint64_t* vals; // pointer to data
    std::shared_ptr< std::vector<uint16_t> > pos;
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

uint64_t intersect_range_bins(const range_bin& b1, const range_bin& b2, const uint8_t n_ints_bin) {
    // Squash
    if((b1.bin_bitmap & b2.bin_bitmap) == 0) return(0);

    uint64_t count = 0;

    if(b1.list || b2.list) {
        //std::cerr << "double list" << std::endl;
        if(b1.n_list < b2.n_list) {
            for(int i = 0; i < b1.pos->size(); ++i) {
                if((b1.bins[b1.pos->at(i)].n_vals == 0) || (b2.bins[b1.pos->at(i)].n_vals == 0))
                    continue;

                if(b1.bins[b1.pos->at(i)].list || b2.bins[b1.pos->at(i)].list) {

                    const bin& bin1 = b1.bins[b1.pos->at(i)];
                    const bin& bin2 = b2.bins[b1.pos->at(i)];

                    if((bin1.bitmap & bin2.bitmap) == 0) {
                        continue;
                    }

                    //std::cerr << "in list-b1" << std::endl;
                    if(bin1.list || bin2.list) {
                        if(bin1.n_list < bin2.n_list) {
                            for(int j = 0; j < bin1.n_list; ++j) {
                                count += TWK_POPCOUNT(bin1.vals[bin1.pos->at(j)] & bin2.vals[bin1.pos->at(j)]);
                            }
                        } else {
                            for(int j = 0; j < bin2.n_list; ++j) {
                                count += TWK_POPCOUNT(bin1.vals[bin2.pos->at(j)] & bin2.vals[bin2.pos->at(j)]);
                            }
                        }
                    }
                } else {
                    for(int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(b1.bins[b1.pos->at(i)].vals[j] & b2.bins[b1.pos->at(i)].vals[j]);
                    }
                }
            }
        } else {
            for(int i = 0; i < b2.n_list; ++i) {
                if((b1.bins[b2.pos->at(i)].n_vals == 0) || (b2.bins[b2.pos->at(i)].n_vals == 0))
                    continue;

                const bin& bin1 = b1.bins[b2.pos->at(i)];
                const bin& bin2 = b2.bins[b2.pos->at(i)];

                if((bin1.bitmap & bin2.bitmap) == 0) {
                    continue;
                }

                if(bin1.list || bin2.list) {
                    if(bin1.n_list < bin2.n_list) {
                        for(int j = 0; j < bin1.n_list; ++j) {
                            count += TWK_POPCOUNT(bin1.vals[bin1.pos->at(j)] & bin2.vals[bin1.pos->at(j)]);
                        }
                    } else {
                        for(int j = 0; j < bin2.n_list; ++j) {
                            count += TWK_POPCOUNT(bin1.vals[bin2.pos->at(j)] & bin2.vals[bin2.pos->at(j)]);
                        }
                    }
                } else {
                    // all
                    for(int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(bin1.vals[j] & bin2.vals[j]);
                    }
                }
            }
        }
    } else { // no lists for either at upper level
        const uint32_t n = b1.bins.size();
        for(int i = 0; i < n; ++i) {
            if(b1.bins[i].n_vals && b2.bins[i].n_vals) {
                if(b1.bins[i].list || b2.bins[i].list) {
                    //std::cerr << "in list-full" << std::endl;
                    const bin& bin1 = b1.bins[i];
                    const bin& bin2 = b2.bins[i];

                    if((bin1.bitmap & bin2.bitmap) == 0)
                       continue;

                    if(bin1.list || bin2.list) {
                        if(bin1.n_list < bin2.n_list) {
                            for(int j = 0; j < bin1.n_list; ++j) {
                                count += TWK_POPCOUNT(bin1.vals[bin1.pos->at(j)] & bin2.vals[bin1.pos->at(j)]);
                            }
                        } else {
                            for(int j = 0; j < bin2.n_list; ++j) {
                                count += TWK_POPCOUNT(bin1.vals[bin2.pos->at(j)] & bin2.vals[bin2.pos->at(j)]);
                            }
                        }
                    } else {
                        // all
                        for(int j = 0; j < n_ints_bin; ++j) {
                            count += TWK_POPCOUNT(bin1.vals[j] & bin2.vals[j]);
                        }
                    }
                } else {
                    // compare values in b
                    for( int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(b1.bins[i].vals[j] & b2.bins[i].vals[j]);
                    }
                }
            }
        }
    }

    return(count);
}

uint64_t intersect_range_bins_bit(const range_bin& b1, const range_bin& b2, const uint8_t n_ints_bin) {
    // Squash
    if((b1.bin_bitmap & b2.bin_bitmap) == 0) return(0);

    uint64_t count = 0;

    if(b1.list || b2.list) {
        if(b1.n_list < b2.n_list) {
            for(int i = 0; i < b1.pos->size(); ++i) {
                if((b1.bins[b1.pos->at(i)].n_vals == 0) || (b2.bins[b1.pos->at(i)].n_vals == 0))
                    continue;

                if(b1.bins[b1.pos->at(i)].list || b2.bins[b1.pos->at(i)].list) {

                    const bin& bin1 = b1.bins[b1.pos->at(i)];
                    const bin& bin2 = b2.bins[b1.pos->at(i)];

                    if((bin1.bitmap & bin2.bitmap) == 0)
                       continue;

                    if(bin1.list || bin2.list) {
                        if(bin1.n_list < bin2.n_list) {
                            for(int j = 0; j < bin1.n_list; ++j) {
                                count += ((bin2.vals[bin1.pos->at(j) / 64] & (1L << (bin1.pos->at(j) % 64))) != 0);
                            }
                        } else {
                            for(int j = 0; j < bin2.n_list; ++j) {
                                count += ((bin1.vals[bin2.pos->at(j) / 64] & (1L << (bin2.pos->at(j) % 64))) != 0);
                            }
                        }
                    }
                } else { // no lists available
                    for(int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(b1.bins[b1.pos->at(i)].vals[j] & b2.bins[b1.pos->at(i)].vals[j]);
                    }
                }
            }
        } else {
            for(int i = 0; i < b2.n_list; ++i) {
                if((b1.bins[b2.pos->at(i)].n_vals == 0) || (b2.bins[b2.pos->at(i)].n_vals == 0))
                    continue;

                const bin& bin1 = b1.bins[b2.pos->at(i)];
                const bin& bin2 = b2.bins[b2.pos->at(i)];

                if((bin1.bitmap & bin2.bitmap) == 0)
                   continue;

                if(bin1.list || bin2.list) {
                    if(bin1.n_list < bin2.n_list) {
                        for(int j = 0; j < bin1.n_list; ++j) {
                            count += ((bin2.vals[bin1.pos->at(j) / 64] & (1L << (bin1.pos->at(j) % 64))) != 0);
                            //count += TWK_POPCOUNT(bin1.vals[bin1.pos->at(j)] & bin2.vals[bin1.pos->at(j)]);
                        }
                    } else {
                        for(int j = 0; j < bin2.n_list; ++j) {
                            count += ((bin1.vals[bin2.pos->at(j) / 64] & (1L << (bin2.pos->at(j) % 64))) != 0);
                            //count += TWK_POPCOUNT(bin1.vals[bin2.pos->at(j)] & bin2.vals[bin2.pos->at(j)]);
                        }
                    }
                } else {
                    // all
                    for(int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(bin1.vals[j] & bin2.vals[j]);
                    }
                }
            }
        }
    } else { // no lists for either at upper level
        const uint32_t n = b1.bins.size();
        for(int i = 0; i < n; ++i) {
            if(b1.bins[i].n_vals && b2.bins[i].n_vals) {
                if(b1.bins[i].list || b2.bins[i].list) {
                    const bin& bin1 = b1.bins[i];
                    const bin& bin2 = b2.bins[i];

                    if((bin1.bitmap & bin2.bitmap) == 0)
                       continue;

                    if(bin1.list || bin2.list) {
                        if(bin1.n_list < bin2.n_list) {
                            for(int j = 0; j < bin1.n_list; ++j) {
                                count += ((bin2.vals[bin1.pos->at(j) / 64] & (1L << (bin1.pos->at(j) % 64))) != 0);
                            }
                        } else {
                            for(int j = 0; j < bin2.n_list; ++j) {
                                count += ((bin1.vals[bin2.pos->at(j) / 64] & (1L << (bin2.pos->at(j) % 64))) != 0);
                            }
                        }
                    } else {
                        // all
                        for(int j = 0; j < n_ints_bin; ++j) {
                            count += TWK_POPCOUNT(bin1.vals[j] & bin2.vals[j]);
                        }
                    }
                } else {
                    // compare values in b
                    for( int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(b1.bins[i].vals[j] & b2.bins[i].vals[j]);
                    }
                }
            }
        }
    }

    return(count);
}

/****************************
*  Function definitions
****************************/
uint64_t intersect_bitmaps_scalar(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    for(int i = 0; i < n_ints; ++i) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
    }

    return(count);
}

uint64_t intersect_bitmaps_scalar_4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count[4] = {0};
    for(int i = 0; i < n_ints; i += 4) {
        count[0] += TWK_POPCOUNT(b1[i+0] & b2[i+0]);
        count[1] += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
        count[2] += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
        count[3] += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
    }

    uint64_t tot_count = count[0] + count[1] + count[2] + count[3];

    return(tot_count);
}

uint64_t intersect_bitmaps_scalar_1x4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    for(int i = 0; i < n_ints; i += 4) {
        count += TWK_POPCOUNT(b1[i+0] & b2[i+0]);
        count += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
        count += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
        count += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
    }

    return(count);
}

uint64_t intersect_bitmaps_scalar_8way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count[8] = {0};
    for(int i = 0; i < n_ints; i += 8) {
        count[0] += TWK_POPCOUNT(b1[i+0] & b2[i+0]);
        count[1] += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
        count[2] += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
        count[3] += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
        count[4] += TWK_POPCOUNT(b1[i+4] & b2[i+4]);
        count[5] += TWK_POPCOUNT(b1[i+5] & b2[i+5]);
        count[6] += TWK_POPCOUNT(b1[i+6] & b2[i+6]);
        count[7] += TWK_POPCOUNT(b1[i+7] & b2[i+7]);
    }

    uint64_t tot_count = count[0] + count[1] + count[2] + count[3] + count[4] + count[5] + count[6] + count[7];

    return(tot_count);
}

uint64_t intersect_bitmaps_scalar_1x8way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    for(int i = 0; i < n_ints; i += 8) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
        count += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
        count += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
        count += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
        count += TWK_POPCOUNT(b1[i+4] & b2[i+4]);
        count += TWK_POPCOUNT(b1[i+5] & b2[i+5]);
        count += TWK_POPCOUNT(b1[i+6] & b2[i+6]);
        count += TWK_POPCOUNT(b1[i+7] & b2[i+7]);
    }

    return(count);
}

uint64_t intersect_bitmaps_scalar_prefix_suffix(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const std::pair<uint32_t,uint32_t>& p1, const std::pair<uint32_t,uint32_t>& p2) {
    const uint32_t from = std::max(p1.first, p2.first);
    const uint32_t to   = std::min(p1.second,p2.second);

    uint64_t count = 0;
    int i = from;
    for(; i + 4 < to; i += 4) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
        count += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
        count += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
        count += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
    }

    for(; i + 2 < to; i += 2) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
        count += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
    }

    for(; i < to; ++i) {
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

uint64_t intersect_bitmaps_scalar_list_4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count[4] = {0};

    if(l1.size() < l2.size()) {
        int i = 0;

        for(; i + 4 < l1.size(); i += 4) {
            count[0] += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
            count[1] += ((b2[l1[i+1] / 64] & (1L << (l1[i+1] % 64))) != 0);
            count[2] += ((b2[l1[i+2] / 64] & (1L << (l1[i+2] % 64))) != 0);
            count[3] += ((b2[l1[i+3] / 64] & (1L << (l1[i+3] % 64))) != 0);
        }


        for(; i + 2 < l1.size(); i += 2) {
            count[0] += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
            count[1] += ((b2[l1[i+1] / 64] & (1L << (l1[i+1] % 64))) != 0);
        }


        for(; i < l1.size(); ++i) {
            count[0] += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
        }
    } else {
        int i = 0;

        for(; i + 4 < l2.size(); i += 4) {
            count[0] += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
            count[1] += ((b1[l2[i+1] / 64] & (1L << (l2[i+1] % 64))) != 0);
            count[2] += ((b1[l2[i+2] / 64] & (1L << (l2[i+2] % 64))) != 0);
            count[3] += ((b1[l2[i+3] / 64] & (1L << (l2[i+3] % 64))) != 0);
        }

        for(; i + 2 < l2.size(); i += 2) {
            count[0] += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
            count[1] += ((b1[l2[i+1] / 64] & (1L << (l2[i+1] % 64))) != 0);
        }


        for(; i < l2.size(); ++i) {
            count[0] += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
        }
    }
    return(count[0] + count[1] + count[2] + count[3]);
}

uint64_t intersect_bitmaps_scalar_list_1x4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    if(l1.size() < l2.size()) {
        int i = 0;

        for(; i + 4 < l1.size(); i += 4) {
            count += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
            count += ((b2[l1[i+1] / 64] & (1L << (l1[i+1] % 64))) != 0);
            count += ((b2[l1[i+2] / 64] & (1L << (l1[i+2] % 64))) != 0);
            count += ((b2[l1[i+3] / 64] & (1L << (l1[i+3] % 64))) != 0);
        }

        for(; i + 2 < l1.size(); i += 2) {
            count += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
            count += ((b2[l1[i+1] / 64] & (1L << (l1[i+1] % 64))) != 0);
        }


        for(; i < l1.size(); ++i) {
            count += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
        }
    } else {
        int i = 0;

        for(; i + 4 < l2.size(); i += 4) {
            count += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
            count += ((b1[l2[i+1] / 64] & (1L << (l2[i+1] % 64))) != 0);
            count += ((b1[l2[i+2] / 64] & (1L << (l2[i+2] % 64))) != 0);
            count += ((b1[l2[i+3] / 64] & (1L << (l2[i+3] % 64))) != 0);
        }

        for(; i + 2 < l2.size(); i += 2) {
            count += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
            count += ((b1[l2[i+1] / 64] & (1L << (l2[i+1] % 64))) != 0);
        }


        for(; i < l2.size(); ++i) {
            count += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
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

uint64_t intersect_bitmaps_scalar_intlist_1x4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    if(l1.size() < l2.size()) {
        int i = 0;
        for(; i + 4 < l1.size(); i += 4) {
            count += TWK_POPCOUNT(b1[l1[i+0]] & b2[l1[i+0]]);
            count += TWK_POPCOUNT(b1[l1[i+1]] & b2[l1[i+1]]);
            count += TWK_POPCOUNT(b1[l1[i+2]] & b2[l1[i+2]]);
            count += TWK_POPCOUNT(b1[l1[i+3]] & b2[l1[i+3]]);
        }

        for(; i + 2 < l1.size(); i += 2) {
            count += TWK_POPCOUNT(b1[l1[i+0]] & b2[l1[i+0]]);
            count += TWK_POPCOUNT(b1[l1[i+1]] & b2[l1[i+1]]);
        }

        for(; i < l1.size(); ++i) {
            count += TWK_POPCOUNT(b1[l1[i]] & b2[l1[i]]);
        }
    } else {
        int i = 0;
        for(; i + 4 < l2.size(); i += 4) {
            count += TWK_POPCOUNT(b1[l2[i+0]] & b2[l2[i+0]]);
            count += TWK_POPCOUNT(b1[l2[i+1]] & b2[l2[i+1]]);
            count += TWK_POPCOUNT(b1[l2[i+2]] & b2[l2[i+2]]);
            count += TWK_POPCOUNT(b1[l2[i+3]] & b2[l2[i+3]]);
        }

        for(; i + 2 < l2.size(); i += 2) {
            count += TWK_POPCOUNT(b1[l2[i+0]] & b2[l2[i+0]]);
            count += TWK_POPCOUNT(b1[l2[i+1]] & b2[l2[i+1]]);
        }

        for(; i < l2.size(); ++i) {
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
        __m128i v1 = _mm_and_si128(r1[i], r2[i]);
        TWK_POPCOUNT_SSE4(count, v1);
    }

    return(count);
}

uint64_t intersect_bitmaps_sse4_2way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count[2] = {0};
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    int i = 0;
    for(; i + 2 < n_cycles; i += 2) {
        __m128i v1 = _mm_and_si128(r1[i+0], r2[i+0]);
        TWK_POPCOUNT_SSE4(count[0], v1);
        v1 = _mm_and_si128(r1[i+1], r2[i+1]);
        TWK_POPCOUNT_SSE4(count[1], v1);
    }

    for(; i < n_cycles; ++i) {
        __m128i v1 = _mm_and_si128(r1[i+0], r2[i+0]);
        TWK_POPCOUNT_SSE4(count[0], v1);
    }

    return(count[0] + count[1]);
}

uint64_t intersect_bitmaps_sse4_1x2way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    int i = 0;
    for(; i + 2 < n_cycles; i += 2) {
        __m128i v1 = _mm_and_si128(r1[i+0], r2[i+0]);
        TWK_POPCOUNT_SSE4(count, v1);
        v1 = _mm_and_si128(r1[i+1], r2[i+1]);
        TWK_POPCOUNT_SSE4(count, v1);
    }

    for(; i < n_cycles; ++i) {
        __m128i v1 = _mm_and_si128(r1[i+0], r2[i+0]);
        TWK_POPCOUNT_SSE4(count, v1);
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

    return(intersect_bitmaps_sse4(b1,b2,n_ints));
}

uint64_t intersect_bitmaps_sse4_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    return(intersect_bitmaps_sse4_list(b1,b2,l1,l2));
}

uint64_t insersect_reduced_sse4(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint16_t>& l1, const std::vector<uint16_t>& l2) {
    const __m128i full_vec = _mm_set1_epi16(0xFFFF);
    const __m128i one_mask = _mm_set1_epi16(1);
    const __m128i range    = _mm_set_epi16(8,7,6,5,4,3,2,1);
    uint64_t count = 0; // helper

    if(l1.size() < l2.size()) {
        const __m128i* y = (const __m128i*)&l2[0];
        const uint32_t n_y = l2.size() / 8; // 128 / 16 vectors

        for(int i = 0; i < l1.size(); ++i) {
            const __m128i x = _mm_set1_epi16(l1[i]); // Broadcast single reference value
            int j = 0;
            for(; j < n_y; ++j) {
                if(l2[j*8] > l1[i]) goto done; // if the current value is larger than the reference value break
                __m128i cmp = _mm_cmpeq_epi16(x, y[j]);
                if(_mm_testz_si128(cmp, full_vec) == false) {
                    const __m128i v = _mm_mullo_epi16(_mm_and_si128(cmp, one_mask), range);
                    const uint16_t* vv = (const uint16_t*)&v;
                    const uint32_t pp = (vv[0] + vv[1] + vv[2] + vv[3] + vv[4] + vv[5] + vv[6] + vv[7]) - 1;
                    count += TWK_POPCOUNT(b1[i] & b2[j*8 + pp]);
                    goto done;
                }
            }

            // Scalar residual
            j *= 8;
            for(; j < l2.size(); ++j) {
                if(l2[j] > l1[i]) goto done;

                if(l1[i] == l2[j]) {
                    //std::cerr << "overlap in scalar=" << l1[i] << "," << l2[j] << std::endl;
                    count += TWK_POPCOUNT(b1[i] & b2[j]);
                    goto done;
                }
            }
            done:
            continue;
        }
    } else {
        const __m128i* y = (const __m128i*)&l1[0];
        const uint32_t n_y = l1.size() / 8; // 128 / 16 vectors

        for(int i = 0; i < l2.size(); ++i) {
            const __m128i x = _mm_set1_epi16(l2[i]); // Broadcast single reference value
            int j = 0;
            for(; j < n_y; ++j) {
                if(l1[j*8] > l2[i]) goto doneLower; // if the current value is larger than the reference value break
                __m128i cmp = _mm_cmpeq_epi16(x, y[j]);
                if(_mm_testz_si128(cmp, full_vec) == false) {
                    const __m128i v = _mm_mullo_epi16(_mm_and_si128(cmp, one_mask), range);
                    const uint16_t* vv = (const uint16_t*)&v;
                    const uint32_t pp = (vv[0] + vv[1] + vv[2] + vv[3] + vv[4] + vv[5] + vv[6] + vv[7]) - 1;
                    count += TWK_POPCOUNT(b2[i] & b1[j*8 + pp]);
                    goto doneLower;
                }
            }

            // Scalar residual
            j *= 8;
            for(; j < l1.size(); ++j) {
                if(l1[j] > l2[i]) goto doneLower;

                if(l2[i] == l1[j]) {
                    //std::cerr << "overlap in scalar=" << l1[i] << "," << l2[j] << std::endl;
                    count += TWK_POPCOUNT(b2[i] & b1[j]);
                    goto doneLower;
                }
            }
            doneLower:
            continue;
        }
    }

    //std::cerr << "final=" << count << std::endl;
    return(count);
}

uint64_t insersect_reduced_scalar(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint16_t>& l1, const std::vector<uint16_t>& l2) {
    uint64_t count = 0; // helper

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            for(int j = 0; j < l2.size(); ++j) {
                if(l2[j] > l1[i]) break;

                if(l1[i] == l2[j]) {
                    count += TWK_POPCOUNT(b1[i] & b2[j]);
                    break;
                }
            }
            continue;
        }

    } else {
        for(int i = 0; i < l2.size(); ++i) {
            for(int j = 0; j < l1.size(); ++j) {
                if(l1[j] > l2[i]) break;

                if(l2[i] == l1[j]) {
                    count += TWK_POPCOUNT(b2[i] & b1[j]);
                    break;
                }
            }
            continue;
        }
    }

    return(count);
}

#else
uint64_t intersect_bitmaps_sse4(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_sse4_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
uint64_t intersect_bitmaps_sse4_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
uint64_t intersect_bitmaps_sse4_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
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

    return(intersect_bitmaps_avx2(b1,b2,n_ints));
}

uint64_t intersect_bitmaps_avx2_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    return(intersect_bitmaps_avx2_list(b1,b2,l1,l2));
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

    return(intersect_bitmaps_avx512(b1, b2, n_ints));
}

uint64_t intersect_bitmaps_avx512_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    return(intersect_bitmaps_avx512_list(b1,b2,l1,l2));
}


#else
uint64_t intersect_bitmaps_avx512(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_avx512_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
uint64_t intersect_bitmaps_avx512_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
uint64_t intersect_bitmaps_avx512_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
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

/**<
 * Compare the uncompressed integers from two sets pairwise.
 * @param v1
 * @param v2
 * @return
 */
uint64_t intersect_raw_naive(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2) {
    uint64_t count = 0;
    for(int i = 0; i < v1.size(); ++i) {
        for(int j = 0; j < v2.size(); ++j) {
            count += (v1[i] == v2[j]);
        }
    }
    return(count);
}

#if SIMD_VERSION >= 3
uint64_t intersect_raw_sse4_broadcast(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2) {
    uint64_t count = 0;
    const __m128i one_mask = _mm_set1_epi16(1);
    if(v1.size() < v2.size()) { // broadcast-compare V1-values to vectors of V2 values
        const uint32_t n_cycles = v2.size() / 8;
       // const __m128i* y = (const __m128i*)(&v2[0]);

        for(int i = 0; i < v1.size(); ++i) {
            const __m128i r = _mm_set1_epi16(v1[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m128i y = _mm_loadu_si128((const __m128i*)&v2[j*8]);
                TWK_POPCOUNT_SSE4(count, _mm_and_si128(_mm_cmpeq_epi16(r, y),one_mask));
            }
            j *= 8;
            for(; j < v2.size(); ++j) count += (v1[i] == v2[j]);
        }
    } else {
        const uint32_t n_cycles = v1.size() / 8;
        const __m128i* y = (const __m128i*)(&v1[0]);

        for(int i = 0; i < v2.size(); ++i) {
            const __m128i r = _mm_set1_epi16(v2[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m128i y = _mm_loadu_si128((const __m128i*)&v1[j*8]);
                TWK_POPCOUNT_SSE4(count, _mm_and_si128(_mm_cmpeq_epi16(r, y),one_mask));
            }
            j *= 8;
            for(; j < v1.size(); ++j) count += (v1[j] == v2[i]);
        }
    }
    return(count);
}
#else
uint64_t intersect_raw_sse4_broadcast(const std::vector<uint32_t>& v1, const std::vector<uint32_t>& v2) { return(0); }
#endif

#if SIMD_VERSION >= 5
uint64_t intersect_raw_avx2_broadcast(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2) {
    uint64_t count = 0;
    const __m256i one_mask = _mm256_set1_epi16(1);
    if(v1.size() < v2.size()) { // broadcast-compare V1-values to vectors of V2 values
        const uint32_t n_cycles = v2.size() / 16;

        for(int i = 0; i < v1.size(); ++i) {
            __m256i r = _mm256_set1_epi16(v1[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m256i y = _mm256_loadu_si256((const __m256i*)&v2[j*16]);
                TWK_POPCOUNT_AVX2(count, _mm256_and_si256(_mm256_cmpeq_epi16(r, y), one_mask));
            }
            j *= 16;
            for(; j < v2.size(); ++j) count += (v1[i] == v2[j]);
        }
    } else {
        const uint32_t n_cycles = v1.size() / 16;

        for(int i = 0; i < v2.size(); ++i) {
            __m256i r = _mm256_set1_epi16(v2[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m256i y = _mm256_loadu_si256((const __m256i*)&v1[j*16]);
                TWK_POPCOUNT_AVX2(count, _mm256_and_si256(_mm256_cmpeq_epi16(r, y), one_mask));
            }
            j *= 16;
            for(; j < v1.size(); ++j)  count += (v1[j] == v2[i]);
        }
    }
    return(count);
}
#else
uint64_t intersect_raw_avx2_broadcast(const std::vector<uint32_t>& v1, const std::vector<uint32_t>& v2) { return(0); }
#endif

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

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const std::pair<uint32_t,uint32_t>& p1, const std::pair<uint32_t,uint32_t>& p2)>
bench_t fpswrapper(const uint32_t n_variants, const uint64_t* vals, const uint32_t n_ints, const std::vector< std::pair<uint32_t,uint32_t> >& pairs) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;
    for(int i = 0; i < n_variants; ++i) {
        inner_offset = offset + n_ints;
        for(int j = i + 1; j < n_variants; ++j, inner_offset += n_ints) {
            total += (*f)(&vals[offset], &vals[inner_offset], n_ints, pairs[i], pairs[j]);
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

template <uint64_t (f)(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint16_t>& l1, const std::vector<uint16_t>& l2)>
bench_t fredwrapper(const uint32_t n_variants, const uint32_t n_vals_actual, const uint64_t* vals, const std::vector< std::vector<uint16_t> >& pos16) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;
    uint32_t l_offset = 0;
    uint32_t l_offset_inner = 0;
    for(int k = 0; k < n_variants; ++k) {
        l_offset_inner = l_offset + pos16[k].size();
        for(int p = k + 1; p < n_variants; ++p) {
            total += (*f)(&vals[l_offset], &vals[l_offset_inner], pos16[k], pos16[p]);
            l_offset_inner += pos16[p].size();
        }
        l_offset += pos16[k].size();
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_vals_actual*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);

    return(b);
}

template <uint64_t (f)(const range_bin& b1, const range_bin& b2, const uint8_t n_ints_bin)>
bench_t frbinswrapper(const uint32_t n_variants, const uint32_t n_vals_actual, const std::vector< range_bin >& bins, const uint8_t n_ints_bin) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;

    for(int k = 0; k < n_variants; ++k) {
        for(int p = k + 1; p < n_variants; ++p) {
            total += (*f)(bins[k], bins[p], n_ints_bin);
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_vals_actual*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);

    return(b);
}

template <uint64_t (f)(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2)>
bench_t frawwrapper(const uint32_t n_variants, const uint32_t n_vals_actual, const std::vector< std::vector<uint16_t> >& pos) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;

    for(int k = 0; k < n_variants; ++k) {
        for(int p = k + 1; p < n_variants; ++p) {
            total += (*f)(pos[k], pos[p]);
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_vals_actual*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);

    return(b);
}

#ifdef USE_ROARING
bench_t froarwrapper(const uint32_t n_variants, const uint32_t n_vals_actual, roaring_bitmap_t** bitmaps) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint64_t total = 0;

    for(int k = 0; k < n_variants; ++k) {
        for(int p = k + 1; p < n_variants; ++p) {
            total += roaring_bitmap_and_cardinality(bitmaps[k], bitmaps[p]);
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    bench_t b; b.count = total; b.milliseconds = time_span.count();
    uint64_t n_comps = (n_variants*n_variants - n_variants) / 2;
    b.throughput = ((n_comps*n_vals_actual*sizeof(uint64_t)) / (1024*1024.0)) / (b.milliseconds / 1000.0);

    return(b);
}
#endif

void intersect_test(uint32_t n, uint32_t cycles = 1) {
    // Setup
    std::vector<uint32_t> samples = {512, 2048, 8192, 32768, 131072, 196608, 589824};
    //std::vector<uint32_t> samples = {32768, 131072, 196608, 589824};
    for(int s = 0; s < samples.size(); ++s) {
        uint32_t n_ints_sample = samples[s] / 64;

        // Limit memory usage to 10e6 but no more than 10k records.
        uint32_t desired_mem = 10 * 1024 * 1024;
        // b_total / (b/obj) = n_ints
        //uint32_t n_variants = std::min((uint32_t)10000, (uint32_t)std::ceil(desired_mem/(n_ints_sample*sizeof(uint64_t))));
        uint32_t n_variants = 10000;

        std::cerr << "Generating: " << samples[s] << " samples for " << n_variants << " variants" << std::endl;
        std::cerr << "Allocating: " << n_ints_sample*n_variants*sizeof(uint64_t)/(1024 * 1024.0) << "Mb" << std::endl;
        uint64_t* vals;
        assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, n_ints_sample*n_variants*sizeof(uint64_t)));
        uint64_t* vals_reduced;
        assert(!posix_memalign((void**)&vals_reduced, SIMD_ALIGNMENT, n_ints_sample*n_variants*sizeof(uint64_t)));

#ifdef USE_ROARING
        // roaring_bitmap_t *r1 = roaring_bitmap_create();
        // uint64_t roaring_bitmap_and_cardinality(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
        // roaring_bitmap_free(r1);
        roaring_bitmap_t** roaring = new roaring_bitmap_t*[n_variants];
        for(int i = 0; i < n_variants; ++i) roaring[i] = roaring_bitmap_create();
        std::cerr << "after roaring init" << std::endl;
#endif

        std::vector<uint32_t> n_alts = {3, samples[s]/1000, samples[s]/500, samples[s]/100, samples[s]/20}; //, samples[s]/10, samples[s]/4, samples[s]/2};
        //std::vector<uint32_t> n_alts = {1,5,10,15,20,25,50,100};

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
            std::vector< std::vector<uint16_t> > pos16(n_variants, std::vector<uint16_t>());
            std::vector< std::vector<uint32_t> > pos_integer(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint16_t> > pos_integer16(n_variants, std::vector<uint16_t>());
            std::vector< std::vector<uint32_t> > pos_reg128(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_reg256(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint32_t> > pos_reg512(n_variants, std::vector<uint32_t>());
            std::vector< std::vector<uint64_t> > squash_4096(n_variants, std::vector<uint64_t>(n_squash4096, 0));

            std::vector< std::pair<uint32_t, uint32_t> > prefix_suffix_pos(n_variants, std::pair<uint32_t, uint32_t>(0,0));

            const uint8_t n_ints_bin = std::min(n_ints_sample, (uint32_t)(4*SIMD_WIDTH/64));
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
            for(uint32_t j = 0; j < n_variants; ++j) {
                for(uint32_t i = 0; i < n_alts[a]; ++i) {
                    uint64_t val = distr(eng);
                    if((vals2[val / 64] & (1L << (val % 64))) == 0) {
                        pos[j].push_back(val);
                    }
                    vals2[val / 64] |= (1L << (val % 64));
                }

                // Sort to simplify
                std::sort(pos[j].begin(), pos[j].end());

                for(int p = 0; p < pos[j].size(); ++p)
                    pos16[j].push_back(pos[j][p]);

                //for(int p = 0; p < pos.back().size(); ++p) std::cerr << " " << pos.back()[p];
                //std::cerr << std::endl;

#ifdef USE_ROARING
                for(int p = 0; p < pos[j].size(); ++p)
                    roaring_bitmap_add(roaring[j], pos[j][p]);
#endif

                // Todo
                // Collapse positions into integers
               // pos_integer.push_back(std::vector<uint32_t>());
                pos_integer[j].push_back(pos[j][0] / 64);

                for(int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 64;
                    if(pos_integer[j].back() != idx) pos_integer[j].push_back(idx);
                }

                //std::cerr << "0->" << pos_integer[j].front() << " " << pos_integer[j].back()+1 << "<-" << n_ints_sample << std::endl;
                prefix_suffix_pos[j].first = pos_integer[j].front();
                prefix_suffix_pos[j].second = pos_integer[j].back()+1;

                //std::cerr << "bin=" << bin_size << std::endl;
                bins[j].n_ints     = pos[j].size();
                bins_bit[j].n_ints = pos[j].size();
                std::vector< std::vector<uint16_t> > vv(bin_size); // integers
                std::vector< std::vector<uint16_t> > vv2(bin_size); // bits
                for(int p = 0; p < pos[j].size(); ++p) {
                    const uint32_t target_bin = pos[j][p] / 64 / n_ints_bin;
                    const uint32_t FOR = (target_bin*64*n_ints_bin); // frame of reference value
                    const uint32_t local_val = (pos[j][p] - FOR);
                    const uint32_t local_int = local_val / 64;

                    //std::cerr << " " << pos[j][p] << ":" << target_bin << " FOR=" << FOR << "->" << local_val << "F=" << local_int << "|" << local_val % 64;

                    // Allocate memory in target bins
                    if(bins[j].bins[target_bin].n_vals == 0) bins[j].bins[target_bin].Allocate(n_ints_bin);
                    if(bins_bit[j].bins[target_bin].n_vals == 0) bins_bit[j].bins[target_bin].Allocate(n_ints_bin);

                    // Add integers
                    if(vv[target_bin].size() == 0) vv[target_bin].push_back(local_int);
                    else if(vv[target_bin].back() != (local_int)) vv[target_bin].push_back(local_int);

                    // Add bits
                    if(vv2[target_bin].size() == 0) vv2[target_bin].push_back(local_val);
                    else if(vv2[target_bin].back() != (local_val)) vv2[target_bin].push_back(local_val);

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
                for(int p = 0; p < vv.size(); ++p) {
                    if(vv[p].size() < 3 && vv[p].size() != 0) {
                        //std::cerr << "setting internal pos" << std::endl;
                        bins[j].bins[p].list = true;
                        bins[j].bins[p].pos = std::make_shared< std::vector<uint16_t> >(vv[p]);
                        bins[j].bins[p].n_list = vv[p].size();
                    }
                }

                std::vector< uint16_t > v;
                for(int p = 0; p < bins[j].bins.size(); ++p) {
                    if(bins[j].bins[p].n_vals) v.push_back(p);
                }

                if(v.size() / (float)bin_size < 0.5) {
                    //std::cerr << v.size() << "/" << bin_size << "->" << (v.size() / (float)bin_size) << std::endl;
                    bins[j].list = true;
                    bins[j].pos = std::make_shared< std::vector<uint16_t> >(v);
                    bins[j].n_list = v.size();
                }

                // for bits
                for(int p = 0; p < vv2.size(); ++p) {
                    if(vv2[p].size() < 10 && vv2[p].size() != 0) {
                        //std::cerr << "setting internal pos" << std::endl;
                        bins_bit[j].bins[p].list = true;
                        bins_bit[j].bins[p].pos = std::make_shared< std::vector<uint16_t> >(vv2[p]);
                        bins_bit[j].bins[p].n_list = vv2[p].size();
                    }
                }

                v.clear();
                for(int p = 0; p < bins_bit[j].bins.size(); ++p) {
                    if(bins_bit[j].bins[p].n_vals) v.push_back(p);
                }

                if(v.size() / (float)bin_size < 0.5) {
                    //std::cerr << v.size() << "/" << bin_size << "->" << (v.size() / (float)bin_size) << std::endl;
                    bins_bit[j].list = true;
                    bins_bit[j].pos = std::make_shared< std::vector<uint16_t> >(v);
                    bins_bit[j].n_list = v.size();
                }

                //

                pos_integer16[j].push_back(pos[j][0] / 64);

                for(int p = 1; p < pos[j].size(); ++p) {
                    uint32_t idx = pos[j][p] / 64;
                    if(pos_integer16[j].back() != idx) pos_integer16[j].push_back(idx);
                }

                //
                for(int p = 0; p < pos_integer16[j].size(); ++p) {
                    //std::cerr << vals2[pos_integer[j][p]] << std::endl;
                    vals_reduced[n_vals_reduced++] = vals2[pos_integer16[j][p]];
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
                vals2 += n_ints_sample;
            }
            std::cerr << "Done!" << std::endl;

            std::cerr << "n_reduced=" << n_vals_reduced << std::endl;

            uint32_t mem_bins = 0;
            for(int i = 0; i < bins.size(); ++i) {
                mem_bins += sizeof(uint64_t);
                for(int j = 0; j < bins[i].bins.size(); ++j) {
                    mem_bins += bins[i].bins[j].n_vals * sizeof(uint64_t);
                }
                mem_bins += sizeof(uint8_t);
            }
            std::cerr << "mem_bins=" << mem_bins << " (" << (n_ints_sample*n_variants*sizeof(uint64_t)) / (double)mem_bins << ")" << std::endl;

            //uint32_t offset = 0;
            /*for(int i = 0; i < n_variants; ++i) {
                construct_ewah64(&vals[offset], n_ints_sample);
                offset += n_ints_sample;
            }
            */

            /*
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
            */

            uint64_t int_comparisons = 0;
            for(int k = 0; k < n_variants; ++k) {
                for(int p = k + 1; p < n_variants; ++p) {
                    int_comparisons += pos[k].size() + pos[p].size();
                }
            }
            const uint64_t n_intersects = ((n_variants * n_variants) - n_variants) / 2;

#define PRINT(name,bench) std::cout << samples[s] << "\t" << n_alts[a] << "\t" << name << "\t" << bench.milliseconds << "\t" << bench.count << "\t" << bench.throughput << "\t" << (bench.milliseconds == 0 ? 0 : (int_comparisons*1000.0 / bench.milliseconds / 1000000.0)) << "\t" << (n_intersects*1000.0 / (bench.milliseconds) / 1000000.0) << std::endl

#ifdef USE_ROARING
            // temp
            bench_t broaring = froarwrapper(n_variants, n_ints_sample, roaring);
            PRINT("roaring",broaring);
#endif

            //
            // const uint32_t n_variants, const uint32_t n_vals_actual, const std::vector< range_bin >& bins, const uint8_t n_ints_bin

            bench_t bins1 = frbinswrapper<&intersect_range_bins>(n_variants, n_ints_sample, bins, n_ints_bin);
            PRINT("bins-popcnt",bins1);

            bench_t bins_bitwise = frbinswrapper<&intersect_range_bins_bit>(n_variants, n_ints_sample, bins_bit, n_ints_bin);
            PRINT("bins-bit",bins_bitwise);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tbins-popcnt\t" << bins1.milliseconds << "\t" << bins1.count << "\t" << bins1.throughput << "\t" << (int_comparisons*1000 / (bins1.milliseconds)) << std::endl;


            bench_t raw1 = frawwrapper<&intersect_raw_naive>(n_variants, n_ints_sample, pos16);
            PRINT("raw-naive",raw1);

            bench_t raw2 = frawwrapper<&intersect_raw_sse4_broadcast>(n_variants, n_ints_sample, pos16);
            PRINT("raw-naive-sse4",raw2);

            bench_t raw3 = frawwrapper<&intersect_raw_avx2_broadcast>(n_variants, n_ints_sample, pos16);
            PRINT("raw-naive-avx2",raw3);

            // Reduced
            //bench_t red1 = fredwrapper<&insersect_reduced_sse4>(n_variants, n_ints_sample, vals_reduced, pos_integer16);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\treduced-sse4-popcnt\t" << red1.milliseconds << "\t" << red1.count << "\t" << red1.throughput << std::endl;

            //bench_t red2 = fredwrapper<&insersect_reduced_scalar>(n_variants, n_ints_sample, vals_reduced, pos_integer16);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\treduced-scalar-popcnt\t" << red2.milliseconds << "\t" << red2.count << "\t" << red2.throughput << std::endl;

            // Scalar 1
            bench_t m1 = fwrapper<&intersect_bitmaps_scalar>(n_variants, vals, n_ints_sample);
            PRINT("scalar-popcnt",m1);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-popcnt\t" << m1.milliseconds << "\t" << m1.count << "\t" << m1.throughput << std::endl;

            // Scalar 4-way
            bench_t m4_way = fwrapper<&intersect_bitmaps_scalar_4way>(n_variants, vals, n_ints_sample);
            PRINT("scalar-popcnt-4way",m4_way);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-popcnt-4way\t" << m4_way.milliseconds << "\t" << m4_way.count << "\t" << m4_way.throughput << std::endl;

            // Scalar 8-way
            bench_t m8_way = fwrapper<&intersect_bitmaps_scalar_8way>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-popcnt-8way\t" << m8_way.milliseconds << "\t" << m8_way.count << "\t" << m8_way.throughput << std::endl;
            PRINT("scalar-popcnt-8way",m8_way);

            // Scalar 1x4-way
            bench_t m1x4_way = fwrapper<&intersect_bitmaps_scalar_1x4way>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-popcnt-1x4way\t" << m1x4_way.milliseconds << "\t" << m1x4_way.count << "\t" << m1x4_way.throughput << std::endl;
            PRINT("scalar-popcnt-1x4way",m1x4_way);

            // Scalar 1x8-way
            bench_t m1x8_way = fwrapper<&intersect_bitmaps_scalar_1x8way>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-popcnt-1x8way\t" << m1x8_way.milliseconds << "\t" << m1x8_way.count << "\t" << m1x8_way.throughput << std::endl;
            PRINT("scalar-popcnt-1x8way",m1x8_way);

            // Scalar prefix-suffix 1x4-way
            bench_t ps_m1x4_way = fpswrapper<&intersect_bitmaps_scalar_prefix_suffix>(n_variants, vals, n_ints_sample, prefix_suffix_pos);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-prefix-suffix-popcnt-1x4way\t" << ps_m1x4_way.milliseconds << "\t" << ps_m1x4_way.count << "\t" << ps_m1x4_way.throughput << std::endl;
            PRINT("scalar-prefix-suffix-popcnt-1x4way",ps_m1x4_way);

            // Scalar-list
            if(n_alts[a] < 200 || (double)n_alts[a]/samples[a] < 0.05) {
                bench_t m4 = flwrapper<&intersect_bitmaps_scalar_list>(n_variants, vals, n_ints_sample, pos);
                //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-list\t" << m4.milliseconds << "\t" << m4.count << "\t" << m4.throughput << std::endl;
                PRINT("scalar-skip-list",m4);

                bench_t m4_4way = flwrapper<&intersect_bitmaps_scalar_list_4way>(n_variants, vals, n_ints_sample, pos);
                //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-list-4way\t" << m4_4way.milliseconds << "\t" << m4_4way.count << "\t" << m4_4way.throughput << std::endl;
                PRINT("scalar-skip-list-4way",m4_4way);

                bench_t m4_1x4way = flwrapper<&intersect_bitmaps_scalar_list_1x4way>(n_variants, vals, n_ints_sample, pos);
                //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-list-1x4way\t" << m4_1x4way.milliseconds << "\t" << m4_1x4way.count << "\t" << m4_1x4way.throughput << std::endl;
                PRINT("scalar-skip-list-1x4way",m4_1x4way);
            } else {
                std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-skip-list\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
                std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-skip-list-4way\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
                std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-skip-list-1x4way\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << std::endl;
            }

            // Scalar-int-list
            bench_t m5 = flwrapper<&intersect_bitmaps_scalar_intlist>(n_variants, vals, n_ints_sample, pos_integer);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-int-list\t" << m5.milliseconds << "\t" << m5.count << "\t" << m5.throughput << std::endl;
            PRINT("scalar-int-skip-list",m5);

            // Scalar-int-list
            bench_t m5_1x4 = flwrapper<&intersect_bitmaps_scalar_intlist_1x4way>(n_variants, vals, n_ints_sample, pos_integer);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tscalar-int-list-1x4\t" << m5_1x4.milliseconds << "\t" << m5_1x4.count << "\t" << m5_1x4.throughput << std::endl;
            PRINT("scalar-int-skip-list-1x4",m5_1x4);


            // SIMD SSE4
            bench_t m2 = fwrapper<&intersect_bitmaps_sse4>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4\t" << m2.milliseconds << "\t" << m2.count << "\t" << m2.throughput << std::endl;
            PRINT("sse4",m2);

            // SIMD SSE 2-way
            bench_t m2_2way = fwrapper<&intersect_bitmaps_sse4_2way>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-2way\t" << m2_2way.milliseconds << "\t" << m2_2way.count << "\t" << m2_2way.throughput << std::endl;
            PRINT("sse4-2way",m2_2way);

            // SIMD SSE 2-way
            bench_t m2_1x2way = fwrapper<&intersect_bitmaps_sse4_1x2way>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-1x2way\t" << m2_1x2way.milliseconds << "\t" << m2_1x2way.count << "\t" << m2_1x2way.throughput << std::endl;
            PRINT("sse4-1x2way",m2_1x2way);


            // SIMD SSE4-list
            bench_t m6 = flwrapper<&intersect_bitmaps_sse4_list>(n_variants, vals, n_ints_sample, pos_reg128);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-list\t" << m6.milliseconds << "\t" << m6.count << "\t" << m6.throughput << std::endl;
            PRINT("sse4-list",m6);

            // SIMD SSE4-squash
            bench_t m13 = fsqwrapper<&intersect_bitmaps_sse4_squash>(n_variants, vals, n_ints_sample, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-squash\t" << m13.milliseconds << "\t" << m13.count << "\t" << m13.throughput << std::endl;
            PRINT("sse4-squash",m13);

            // SIMD SSE4-list-squash
            bench_t m14 = flsqwrapper<&intersect_bitmaps_sse4_list_squash>(n_variants, vals, n_ints_sample, pos_reg128, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tsse4-list-squash\t" << m14.milliseconds << "\t" << m14.count << "\t" << m14.throughput << std::endl;
            PRINT("sse4-list-squash",m14);

            // SIMD AVX2
            bench_t m3 = fwrapper<&intersect_bitmaps_avx2>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2\t" << m3.milliseconds << "\t" << m3.count << "\t" << m3.throughput << std::endl;
            PRINT("avx2",m3);

            // SIMD AVX2-list
            bench_t m7 = flwrapper<&intersect_bitmaps_avx2_list>(n_variants, vals, n_ints_sample, pos_reg256);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2-list\t" << m7.milliseconds << "\t" << m7.count << "\t" << m7.throughput << std::endl;
            PRINT("avx2-skip-list",m7);

            // SIMD AVX2-squash
            bench_t m10 = fsqwrapper<&intersect_bitmaps_avx2_squash>(n_variants, vals, n_ints_sample, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2-squash\t" << m10.milliseconds << "\t" << m10.count << "\t" << m10.throughput << std::endl;
            PRINT("avx2-squash",m10);

            // SIMD AVX2-list-squash
            bench_t m12 = flsqwrapper<&intersect_bitmaps_avx2_list_squash>(n_variants, vals, n_ints_sample, pos_reg256, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx2-list-squash\t" << m12.milliseconds << "\t" << m12.count << "\t" << m12.throughput << std::endl;
            PRINT("avx2-skip-list-squash",m12);

            // SIMD AVX512
            bench_t m8 = fwrapper<&intersect_bitmaps_avx512>(n_variants, vals, n_ints_sample);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512\t" << m8.milliseconds << "\t" << m8.count << "\t" << m8.throughput << std::endl;
            PRINT("avx512",m8);

            // SIMD AVX512-list
            bench_t m9 = flwrapper<&intersect_bitmaps_avx512_list>(n_variants, vals, n_ints_sample, pos_reg512);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512-list\t" << m9.milliseconds << "\t" << m9.count << "\t" << m9.throughput << std::endl;
            PRINT("avx512-skip-list",m9);

            // SIMD AVX512-squash
            bench_t m11 = fsqwrapper<&intersect_bitmaps_avx512_squash>(n_variants, vals, n_ints_sample, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512-squash\t" << m11.milliseconds << "\t" << m11.count << "\t" << m11.throughput << std::endl;
            PRINT("avx512-squash",m11);

            // SIMD AVX512-list-squash
            bench_t m15 = flsqwrapper<&intersect_bitmaps_avx512_list_squash>(n_variants, vals, n_ints_sample, pos_reg512, squash_4096);
            //std::cout << samples[s] << "\t" << n_alts[a] << "\tavx512-list-squash\t" << m15.milliseconds << "\t" << m15.count << "\t" << m15.throughput << std::endl;
            PRINT("avx512-skip-list-squash",m15);
        }

        delete[] vals;
        delete[] vals_reduced;
#ifdef USE_ROARING
        // Cleanup
        for(int i = 0; i < n_variants; ++i) roaring_bitmap_free(roaring[i]);
        delete[] roaring;
#endif
    }
}

int main(int argc, char **argv) {
    // Debug
    /*
    std::vector<uint64_t> d1 = {0, 0}; d1[1] |= 1L << 32; d1[1] |= 1L << 63;
    std::vector<uint16_t> v1 = {21, 32};
    std::vector<uint64_t> d2 = {0, 0, 0,0,0,0,0,0,0,0,0,0}; d2[1] |= 1L << 32; d2[1] |= 1L << 33; d2[1] |= 1L << 34; d2[1] |= 1L << 63;
    std::vector<uint16_t> v2 = {15, 32,52,53,66,71,91,127,451,5091, 12401, 14000};
    insersect_reduced_sse4(&d1[0], &d2[0], v1,v2);

    return(1);
    */

    intersect_test(100000000, 10);
    return(0);
}
