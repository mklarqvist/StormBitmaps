#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <cstring>

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

#if SIMD_AVAILABLE
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

void intersect_test(uint32_t n, uint32_t cycles = 1) {
    std::cerr << "Generating flags: " << n << std::endl;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    //uint64_t times[9] = {0};
    //uint64_t times_local[9];

    // Setup
    uint32_t n_samples = 8192;
    uint32_t n_ints_sample = 8192 / 64;
    uint32_t n_variants = 50000;

    std::cerr << "Allocating: " << n_ints_sample*n_variants*sizeof(uint32_t) << std::endl;
    uint64_t* vals;
    assert(!posix_memalign((void**)&vals, 16, n_ints_sample*n_variants*sizeof(uint64_t)));

    std::vector<uint32_t> n_alts = {10, 100, 1000, 4000};

    for(int a = 0; a < n_alts.size(); ++a) {
        // Allocation
        memset(vals, 0, n_ints_sample*n_variants*sizeof(uint32_t));

        // PRNG
        std::uniform_int_distribution<uint32_t> distr(0, n_samples-1); // right inclusive

        // Positional information
        std::vector< std::vector<uint32_t> > pos;
        std::vector< std::vector<uint32_t> > pos_integer;
        std::vector< std::vector<uint32_t> > pos_reg128;

        // Draw
        //uint32_t offset = 0;
        uint64_t* vals2 = vals;
        for(uint32_t j = 0; j < n_variants; ++j) {
            pos.push_back(std::vector<uint32_t>());
            for(uint32_t i = 0; i < n_alts[a]; ++i) {
                //vals[(offset + i) / 32] |= 1 << distr(eng);
                uint32_t val = distr(eng);
                if((vals2[val / 64] & (1L << (val % 64))) == 0) {
                    pos.back().push_back(val);
                }
                //std::cerr << "setting=" << offset+i << std::endl;
                vals2[val / 64] |= (1L << (val % 64));
            }
            vals2 += n_ints_sample;

            // Sort to simplify
            std::sort(pos.back().begin(), pos.back().end());
            //for(int p = 0; p < pos.back().size(); ++p) std::cerr << " " << pos.back()[p];
            //std::cerr << std::endl;

            // Todo
            // Collapse positions into integers
            pos_integer.push_back(std::vector<uint32_t>());
            pos_integer.back().push_back(pos.back()[0] / 64);

            for(int p = 1; p < pos.back().size(); ++p) {
                uint32_t idx = pos.back()[p] / 64;
                if(pos_integer.back().back() != idx) pos_integer.back().push_back(idx);
            }

            //for(int p = 0; p < pos_integer.back().size(); ++p) std::cerr << " " << pos_integer.back()[p];
            //std::cerr << std::endl;

            // Todo
            // Collapse positions into registers
            pos_reg128.push_back(std::vector<uint32_t>());
            pos_reg128.back().push_back(pos.back()[0] / 128);

            for(int p = 1; p < pos.back().size(); ++p) {
                uint32_t idx = pos.back()[p] / 128;
                if(pos_reg128.back().back() != idx) pos_reg128.back().push_back(idx);
            }

            // Todo print averages
            //std::cerr << pos.back().size() << "->" << pos_integer.back().size() << "->" << pos_reg128.back().size() << std::endl;
        }

        uint32_t offset = 0;
        for(int i = 0; i < n_variants; ++i) {
            construct_ewah64(&vals[offset], n_ints_sample);
            offset += n_ints_sample;
        }

        // Scalar 1
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        offset = 0;
        uint32_t inner_offset = 0;
        uint64_t total = 0;
        for(int i = 0; i < n_samples; ++i) {
            inner_offset = offset + n_ints_sample;
            for(int j = i + 1; j < n_samples; ++j, inner_offset += n_ints_sample) {
                total += intersect_bitmaps_scalar(&vals[offset], &vals[inner_offset], n_ints_sample);
            }
            offset += n_ints_sample;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

        std::cerr << n_alts[a] << "\tscalar-bit time=" << time_span.count() << " total=" << total << std::endl;

        // SIMD SSE4
        t1 = std::chrono::high_resolution_clock::now();

        offset = 0;
        inner_offset = 0;
        total = 0;
        for(int i = 0; i < n_samples; ++i) {
            inner_offset = offset + n_ints_sample;
            for(int j = i + 1; j < n_samples; ++j, inner_offset += n_ints_sample) {
                total += intersect_bitmaps_sse4(&vals[offset], &vals[inner_offset], n_ints_sample);
            }
            offset += n_ints_sample;
        }

        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

        std::cerr << n_alts[a] << "\tsse4 time=" << time_span.count() << " total=" << total << std::endl;

        // Scalar list
        t1 = std::chrono::high_resolution_clock::now();

        offset = 0;
        inner_offset = 0;
        total = 0;
        for(int i = 0; i < n_samples; ++i) {
           inner_offset = offset + n_ints_sample;
           for(int j = i + 1; j < n_samples; ++j, inner_offset += n_ints_sample) {
               total += intersect_bitmaps_scalar_list(&vals[offset], &vals[inner_offset], pos[i], pos[j]);
           }
           offset += n_ints_sample;
        }

        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

        std::cerr << n_alts[a] << "\tscalar-list time=" << time_span.count() << " total=" << total << std::endl;

        // Scalar integer list
        t1 = std::chrono::high_resolution_clock::now();

        offset = 0;
        inner_offset = 0;
        total = 0;
        for(int i = 0; i < n_samples; ++i) {
           inner_offset = offset + n_ints_sample;
           for(int j = i + 1; j < n_samples; ++j, inner_offset += n_ints_sample) {
               total += intersect_bitmaps_scalar_intlist(&vals[offset], &vals[inner_offset], pos_integer[i], pos_integer[j]);
           }
           offset += n_ints_sample;
        }

        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

        std::cerr << n_alts[a] << "\tscalar-intlist time=" << time_span.count() << " total=" << total << std::endl;


        // SSE4 list
        t1 = std::chrono::high_resolution_clock::now();

        offset = 0;
        inner_offset = 0;
        total = 0;
        for(int i = 0; i < n_samples; ++i) {
           inner_offset = offset + n_ints_sample;
           for(int j = i + 1; j < n_samples; ++j, inner_offset += n_ints_sample) {
               total += intersect_bitmaps_sse4_list(&vals[offset], &vals[inner_offset], pos_reg128[i], pos_reg128[j]);
           }
           offset += n_ints_sample;
        }

        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

        std::cerr << n_alts[a] << "\tsse4-list time=" << time_span.count() << " total=" << total << std::endl;
    }

    delete[] vals;
}

int main(int argc, char **argv) {
    intersect_test(100000000, 10);
    return(0);
}
