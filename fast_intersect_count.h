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

/* *************************************
*  Includes
***************************************/
#include <stdint.h>
#include <assert.h>
#include <memory.h>
#include <string.h>
#include <math.h>

// Default size of a memory block. This is by default set to 256kb which is what
// most commodity processors have as L2/L3 cache.
#ifndef TWK_CACHE_BLOCK_SIZE
#define TWK_CACHE_BLOCK_SIZE 256e3
#endif

// Safety
#if !(defined(__APPLE__)) && !(defined(__FreeBSD__))
#include <malloc.h>  // this should never be needed but there are some reports that it is needed.
#endif

/* *************************************
 *  Support.
 * 
 *  These subroutines and definitions are taken from the CRoaring repo
 *  by Daniel Lemire et al. available under the Apache 2.0 License
 *  (same as libintersect.h):
 *  https://github.com/RoaringBitmap/CRoaring/ 
 ***************************************/
#if defined(__SIZEOF_LONG_LONG__) && __SIZEOF_LONG_LONG__ != 8
#error This code assumes 64-bit long longs (by use of the GCC intrinsics). Your system is not currently supported.
#endif

/* ===   Compiler specifics   === */

#if defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L   /* >= C99 */
#  define TWK_RESTRICT   restrict
#else
/* note : it might be useful to define __restrict or __restrict__ for some C++ compilers */
#  define TWK_RESTRICT   /* disable */
#endif

#include <x86intrin.h>

/****************************
*  Memory management
* 
*  The subroutines aligned_malloc and aligned_free had to be renamed to
*  TWK_aligned_malloc and aligned_free_port to stop clashing with the
*  same subroutines in Roaring. These subroutines are included here
*  since there is no hard dependency on using Roaring bitmaps.
****************************/
// portable version of  posix_memalign
#ifndef TWK_aligned_malloc
static 
void* TWK_aligned_malloc(size_t alignment, size_t size) {
    void *p;
#ifdef _MSC_VER
    p = _aligned_malloc(size, alignment);
#elif defined(__MINGW32__) || defined(__MINGW64__)
    p = __mingw_aligned_malloc(size, alignment);
#else
    // somehow, if this is used before including "x86intrin.h", it creates an
    // implicit defined warning.
    if (posix_memalign(&p, alignment, size) != 0) 
        return NULL;
#endif
    return p;
}
#endif

#ifndef TWK_aligned_free
static 
void TWK_aligned_free(void* memblock) {
#ifdef _MSC_VER
    _aligned_free(memblock);
#elif defined(__MINGW32__) || defined(__MINGW64__)
    __mingw_aligned_free(memblock);
#else
    free(memblock);
#endif
}
#endif

// portable alignment
#if defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)   /* C11+ */
#  include <stdalign.h>
#  define TWK_ALIGN(n)      alignas(n)
#elif defined(__GNUC__)
#  define TWK_ALIGN(n)      __attribute__ ((aligned(n)))
#elif defined(_MSC_VER)
#  define TWK_ALIGN(n)      __declspec(align(n))
#else
#  define TWK_ALIGN(n)   /* disabled */
#endif

// Taken from XXHASH
#ifdef _MSC_VER    /* Visual Studio */
#  pragma warning(disable : 4127)      /* disable: C4127: conditional expression is constant */
#  define TWK_FORCE_INLINE static __forceinline
#  define TWK_NO_INLINE static __declspec(noinline)
#else
#  if defined (__cplusplus) || defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L   /* C99 */
#    ifdef __GNUC__
#      define TWK_FORCE_INLINE static inline __attribute__((always_inline))
#      define TWK_NO_INLINE static __attribute__((noinline))
#    else
#      define TWK_FORCE_INLINE static inline
#      define TWK_NO_INLINE static
#    endif
#  else
#    define TWK_FORCE_INLINE static
#    define TWK_NO_INLINE static
#  endif /* __STDC_VERSION__ */
#endif

// disable noise
#ifdef __GNUC__
#define WARN_UNUSED __attribute__((warn_unused_result))
#else
#define WARN_UNUSED
#endif

/*------ SIMD definitions --------*/

#define TWK_SSE_ALIGNMENT    16
#define TWK_AVX2_ALIGNMENT   32
#define TWK_AVX512_ALIGNMENT 64

/****************************
*  General checks
****************************/

#ifndef __has_builtin
  #define __has_builtin(x) 0
#endif

#ifndef __has_attribute
  #define __has_attribute(x) 0
#endif

#ifdef __GNUC__
  #define GNUC_PREREQ(x, y) \
      (__GNUC__ > x || (__GNUC__ == x && __GNUC_MINOR__ >= y))
#else
  #define GNUC_PREREQ(x, y) 0
#endif

#ifdef __clang__
  #define CLANG_PREREQ(x, y) \
      (__clang_major__ > x || (__clang_major__ == x && __clang_minor__ >= y))
#else
  #define CLANG_PREREQ(x, y) 0
#endif

#if (defined(__i386__) || \
     defined(__x86_64__) || \
     defined(_M_IX86) || \
     defined(_M_X64))
  #define X86_OR_X64
#endif

#if defined(X86_OR_X64) && \
   (defined(__cplusplus) || \
    defined(_MSC_VER) || \
   (GNUC_PREREQ(4, 2) || \
    __has_builtin(__sync_val_compare_and_swap)))
  #define HAVE_CPUID
#endif

#if GNUC_PREREQ(4, 2) || \
    __has_builtin(__builtin_popcount)
  #define HAVE_BUILTIN_POPCOUNT
#endif

#if GNUC_PREREQ(4, 2) || \
    CLANG_PREREQ(3, 0)
  #define HAVE_ASM_POPCNT
#endif

#if defined(HAVE_CPUID) && \
   (defined(HAVE_ASM_POPCNT) || \
    defined(_MSC_VER))
  #define HAVE_POPCNT
#endif

#if defined(HAVE_CPUID) && \
    GNUC_PREREQ(4, 9)
  #define HAVE_SSE42
  #define HAVE_AVX2
#endif

#if defined(HAVE_CPUID) && \
    GNUC_PREREQ(5, 0)
  #define HAVE_AVX512
#endif

#if defined(HAVE_CPUID) && \
    defined(_MSC_VER) && \
    defined(__AVX2__)
  #define HAVE_SSE42
  #define HAVE_AVX2
#endif

#if defined(HAVE_CPUID) && \
    defined(_MSC_VER) && \
    defined(__AVX512__)
  #define HAVE_AVX512
#endif

#if defined(HAVE_CPUID) && \
    CLANG_PREREQ(3, 8) && \
    __has_attribute(target) && \
   (!defined(_MSC_VER) || defined(__AVX2__)) && \
   (!defined(__apple_build_version__) || __apple_build_version__ >= 8000000)
  #define HAVE_SSE42
  #define HAVE_AVX2
  #define HAVE_AVX512
#endif

#ifdef __cplusplus
extern "C" {
#endif

/****************************
*  CPUID
****************************/
#if defined(HAVE_CPUID)

#if defined(_MSC_VER)
  #include <intrin.h>
  #include <immintrin.h>
#endif

// CPUID flags. See https://en.wikipedia.org/wiki/CPUID for more info.
/* %ecx bit flags */
#define TWK_bit_POPCNT   (1 << 23) // POPCNT instruction 
#define TWK_bit_SSE41    (1 << 19) // CPUID.01H:ECX.SSE41[Bit 19]
#define TWK_bit_SSE42    (1 << 20) // CPUID.01H:ECX.SSE41[Bit 20]

/* %ebx bit flags */
#define TWK_bit_AVX2     (1 << 5)  // CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]
#define TWK_bit_AVX512BW (1 << 30) // AVX-512 Byte and Word Instructions

/* xgetbv bit flags */
#define TWK_XSTATE_SSE (1 << 1)
#define TWK_XSTATE_YMM (1 << 2)
#define TWK_XSTATE_ZMM (7 << 5)

static  
void TWK_run_cpuid(int eax, int ecx, int* abcd) {
#if defined(_MSC_VER)
  __cpuidex(abcd, eax, ecx);
#else
  int ebx = 0;
  int edx = 0;

  #if defined(__i386__) && \
      defined(__PIC__)
    /* in case of PIC under 32-bit EBX cannot be clobbered */
    __asm__ ("movl %%ebx, %%edi;"
             "cpuid;"
             "xchgl %%ebx, %%edi;"
             : "=D" (ebx),
               "+a" (eax),
               "+c" (ecx),
               "=d" (edx));
  #else
    __asm__ ("cpuid;"
             : "+b" (ebx),
               "+a" (eax),
               "+c" (ecx),
               "=d" (edx));
  #endif

  abcd[0] = eax;
  abcd[1] = ebx;
  abcd[2] = ecx;
  abcd[3] = edx;
#endif
}

#if defined(HAVE_AVX2) || \
    defined(HAVE_AVX512)

static 
int get_xcr0()
{
  int xcr0;

#if defined(_MSC_VER)
  xcr0 = (int) _xgetbv(0);
#else
  __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
#endif

  return xcr0;
}

#endif

static  
int get_cpuid() {
    int flags = 0;
    int abcd[4];

    TWK_run_cpuid(1, 0, abcd);

    // Check for POPCNT instruction
    if ((abcd[2] & TWK_bit_POPCNT) == TWK_bit_POPCNT)
        flags |= TWK_bit_POPCNT;

    // Check for SSE4.1 instruction set
    if ((abcd[2] & TWK_bit_SSE41) == TWK_bit_SSE41)
        flags |= TWK_bit_SSE41;

    // Check for SSE4.2 instruction set
    if ((abcd[2] & TWK_bit_SSE42) == TWK_bit_SSE42)
        flags |= TWK_bit_SSE42;

#if defined(HAVE_AVX2) || \
    defined(HAVE_AVX512)

    int osxsave_mask = (1 << 27);

    /* ensure OS supports extended processor state management */
    if ((abcd[2] & osxsave_mask) != osxsave_mask)
        return 0;

    int ymm_mask = TWK_XSTATE_SSE | TWK_XSTATE_YMM;
    int zmm_mask = TWK_XSTATE_SSE | TWK_XSTATE_YMM | TWK_XSTATE_ZMM;

    int xcr0 = get_xcr0();

    if ((xcr0 & ymm_mask) == ymm_mask) {
        TWK_run_cpuid(7, 0, abcd);

        if ((abcd[1] & TWK_bit_AVX2) == TWK_bit_AVX2)
            flags |= TWK_bit_AVX2;

        if ((xcr0 & zmm_mask) == zmm_mask) {
            if ((abcd[1] & TWK_bit_AVX512BW) == TWK_bit_AVX512BW)
            flags |= TWK_bit_AVX512BW;
        }
    }

#endif

  return flags;
}
#endif // defined(HAVE_CPUID)

/// Taken from libpopcnt.h
#if defined(HAVE_ASM_POPCNT) && \
    defined(__x86_64__)

TWK_FORCE_INLINE
uint64_t TWK_POPCOUNT(uint64_t x)
{
  __asm__ ("popcnt %1, %0" : "=r" (x) : "0" (x));
  return x;
}

#elif defined(HAVE_ASM_POPCNT) && \
      defined(__i386__)

TWK_FORCE_INLINE
uint32_t popcnt32(uint32_t x)
{
  __asm__ ("popcnt %1, %0" : "=r" (x) : "0" (x));
  return x;
}

TWK_FORCE_INLINE
uint64_t TWK_POPCOUNT(uint64_t x)
{
  return popcnt32((uint32_t) x) +
         popcnt32((uint32_t)(x >> 32));
}

#elif defined(_MSC_VER) && \
      defined(_M_X64)

#include <nmmintrin.h>

TWK_FORCE_INLINE
uint64_t TWK_POPCOUNT(uint64_t x) {
  return _mm_popcnt_u64(x);
}

#elif defined(_MSC_VER) && \
      defined(_M_IX86)

#include <nmmintrin.h>

TWK_FORCE_INLINE
uint64_t TWK_POPCOUNT(uint64_t x)
{
  return _mm_popcnt_u32((uint32_t) x) + 
         _mm_popcnt_u32((uint32_t)(x >> 32));
}

/* non x86 CPUs */
#elif defined(HAVE_BUILTIN_POPCOUNT)

TWK_FORCE_INLINE
uint64_t TWK_POPCOUNT(uint64_t x) {
  return __builtin_popcountll(x);
}

/* no hardware POPCNT,
 * use pure integer algorithm */
#else

TWK_FORCE_INLINE
uint64_t TWK_POPCOUNT(uint64_t x) {
  return popcount64(x);
}

#endif


static 
uint64_t TWK_intersect_unrolled(const uint64_t* TWK_RESTRICT data1, 
                                const uint64_t* TWK_RESTRICT data2, 
                                uint64_t size)
{
    const uint64_t limit = size - size % 4;
    uint64_t cnt = 0;
    uint64_t i   = 0;

    for (/**/; i < limit; i += 4) {
        cnt += TWK_POPCOUNT(data1[i+0] & data2[i+0]);
        cnt += TWK_POPCOUNT(data1[i+1] & data2[i+1]);
        cnt += TWK_POPCOUNT(data1[i+2] & data2[i+2]);
        cnt += TWK_POPCOUNT(data1[i+3] & data2[i+3]);
    }

    for (/**/; i < size; ++i)
        cnt += TWK_POPCOUNT(data1[i] & data2[i]);

    return cnt;
}

static
const uint8_t TWK_lookup8bit[256] = {
	/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
	/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
	/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
	/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
	/* 10 */ 1, /* 11 */ 2, /* 12 */ 2, /* 13 */ 3,
	/* 14 */ 2, /* 15 */ 3, /* 16 */ 3, /* 17 */ 4,
	/* 18 */ 2, /* 19 */ 3, /* 1a */ 3, /* 1b */ 4,
	/* 1c */ 3, /* 1d */ 4, /* 1e */ 4, /* 1f */ 5,
	/* 20 */ 1, /* 21 */ 2, /* 22 */ 2, /* 23 */ 3,
	/* 24 */ 2, /* 25 */ 3, /* 26 */ 3, /* 27 */ 4,
	/* 28 */ 2, /* 29 */ 3, /* 2a */ 3, /* 2b */ 4,
	/* 2c */ 3, /* 2d */ 4, /* 2e */ 4, /* 2f */ 5,
	/* 30 */ 2, /* 31 */ 3, /* 32 */ 3, /* 33 */ 4,
	/* 34 */ 3, /* 35 */ 4, /* 36 */ 4, /* 37 */ 5,
	/* 38 */ 3, /* 39 */ 4, /* 3a */ 4, /* 3b */ 5,
	/* 3c */ 4, /* 3d */ 5, /* 3e */ 5, /* 3f */ 6,
	/* 40 */ 1, /* 41 */ 2, /* 42 */ 2, /* 43 */ 3,
	/* 44 */ 2, /* 45 */ 3, /* 46 */ 3, /* 47 */ 4,
	/* 48 */ 2, /* 49 */ 3, /* 4a */ 3, /* 4b */ 4,
	/* 4c */ 3, /* 4d */ 4, /* 4e */ 4, /* 4f */ 5,
	/* 50 */ 2, /* 51 */ 3, /* 52 */ 3, /* 53 */ 4,
	/* 54 */ 3, /* 55 */ 4, /* 56 */ 4, /* 57 */ 5,
	/* 58 */ 3, /* 59 */ 4, /* 5a */ 4, /* 5b */ 5,
	/* 5c */ 4, /* 5d */ 5, /* 5e */ 5, /* 5f */ 6,
	/* 60 */ 2, /* 61 */ 3, /* 62 */ 3, /* 63 */ 4,
	/* 64 */ 3, /* 65 */ 4, /* 66 */ 4, /* 67 */ 5,
	/* 68 */ 3, /* 69 */ 4, /* 6a */ 4, /* 6b */ 5,
	/* 6c */ 4, /* 6d */ 5, /* 6e */ 5, /* 6f */ 6,
	/* 70 */ 3, /* 71 */ 4, /* 72 */ 4, /* 73 */ 5,
	/* 74 */ 4, /* 75 */ 5, /* 76 */ 5, /* 77 */ 6,
	/* 78 */ 4, /* 79 */ 5, /* 7a */ 5, /* 7b */ 6,
	/* 7c */ 5, /* 7d */ 6, /* 7e */ 6, /* 7f */ 7,
	/* 80 */ 1, /* 81 */ 2, /* 82 */ 2, /* 83 */ 3,
	/* 84 */ 2, /* 85 */ 3, /* 86 */ 3, /* 87 */ 4,
	/* 88 */ 2, /* 89 */ 3, /* 8a */ 3, /* 8b */ 4,
	/* 8c */ 3, /* 8d */ 4, /* 8e */ 4, /* 8f */ 5,
	/* 90 */ 2, /* 91 */ 3, /* 92 */ 3, /* 93 */ 4,
	/* 94 */ 3, /* 95 */ 4, /* 96 */ 4, /* 97 */ 5,
	/* 98 */ 3, /* 99 */ 4, /* 9a */ 4, /* 9b */ 5,
	/* 9c */ 4, /* 9d */ 5, /* 9e */ 5, /* 9f */ 6,
	/* a0 */ 2, /* a1 */ 3, /* a2 */ 3, /* a3 */ 4,
	/* a4 */ 3, /* a5 */ 4, /* a6 */ 4, /* a7 */ 5,
	/* a8 */ 3, /* a9 */ 4, /* aa */ 4, /* ab */ 5,
	/* ac */ 4, /* ad */ 5, /* ae */ 5, /* af */ 6,
	/* b0 */ 3, /* b1 */ 4, /* b2 */ 4, /* b3 */ 5,
	/* b4 */ 4, /* b5 */ 5, /* b6 */ 5, /* b7 */ 6,
	/* b8 */ 4, /* b9 */ 5, /* ba */ 5, /* bb */ 6,
	/* bc */ 5, /* bd */ 6, /* be */ 6, /* bf */ 7,
	/* c0 */ 2, /* c1 */ 3, /* c2 */ 3, /* c3 */ 4,
	/* c4 */ 3, /* c5 */ 4, /* c6 */ 4, /* c7 */ 5,
	/* c8 */ 3, /* c9 */ 4, /* ca */ 4, /* cb */ 5,
	/* cc */ 4, /* cd */ 5, /* ce */ 5, /* cf */ 6,
	/* d0 */ 3, /* d1 */ 4, /* d2 */ 4, /* d3 */ 5,
	/* d4 */ 4, /* d5 */ 5, /* d6 */ 5, /* d7 */ 6,
	/* d8 */ 4, /* d9 */ 5, /* da */ 5, /* db */ 6,
	/* dc */ 5, /* dd */ 6, /* de */ 6, /* df */ 7,
	/* e0 */ 3, /* e1 */ 4, /* e2 */ 4, /* e3 */ 5,
	/* e4 */ 4, /* e5 */ 5, /* e6 */ 5, /* e7 */ 6,
	/* e8 */ 4, /* e9 */ 5, /* ea */ 5, /* eb */ 6,
	/* ec */ 5, /* ed */ 6, /* ee */ 6, /* ef */ 7,
	/* f0 */ 4, /* f1 */ 5, /* f2 */ 5, /* f3 */ 6,
	/* f4 */ 5, /* f5 */ 6, /* f6 */ 6, /* f7 */ 7,
	/* f8 */ 5, /* f9 */ 6, /* fa */ 6, /* fb */ 7,
	/* fc */ 6, /* fd */ 7, /* fe */ 7, /* ff */ 8
};

/****************************
*  SSE4.1 functions
****************************/

#if defined(HAVE_SSE42)

#include <immintrin.h>

#ifndef TWK_POPCOUNT_SSE4
#define TWK_POPCOUNT_SSE4(A, B) {               \
    A += TWK_POPCOUNT(_mm_extract_epi64(B, 0)); \
    A += TWK_POPCOUNT(_mm_extract_epi64(B, 1)); \
}
#endif

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
TWK_FORCE_INLINE  
uint64_t TWK_POPCOUNT_SSE(const __m128i n) {
    return(TWK_POPCOUNT(_mm_cvtsi128_si64(n)) + 
           TWK_POPCOUNT(_mm_cvtsi128_si64(_mm_unpackhi_epi64(n, n))));
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
TWK_FORCE_INLINE 
void TWK_CSA128(__m128i* h, __m128i* l, __m128i a, __m128i b, __m128i c) {
    __m128i u = _mm_xor_si128(a, b);
    *h = _mm_or_si128(_mm_and_si128(a, b), _mm_and_si128(u, c));
    *l = _mm_xor_si128(u, c);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static 
uint64_t TWK_intersect_csa_sse4(const __m128i* TWK_RESTRICT data1, 
                                const __m128i* TWK_RESTRICT data2, 
                                uint64_t size)
{
    __m128i ones     = _mm_setzero_si128();
    __m128i twos     = _mm_setzero_si128();
    __m128i fours    = _mm_setzero_si128();
    __m128i eights   = _mm_setzero_si128();
    __m128i sixteens = _mm_setzero_si128();
    __m128i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t cnt64 = 0;

    for (/**/; i < limit; i += 16) {
        TWK_CSA128(&twosA,   &ones,   ones,  (_mm_loadu_si128(&data1[i+0])  & _mm_loadu_si128(&data2[i+0])), (_mm_loadu_si128(&data1[i+1]) & _mm_loadu_si128(&data2[i+1])));
        TWK_CSA128(&twosB,   &ones,   ones,  (_mm_loadu_si128(&data1[i+2])  & _mm_loadu_si128(&data2[i+2])), (_mm_loadu_si128(&data1[i+3]) & _mm_loadu_si128(&data2[i+3])));
        TWK_CSA128(&foursA,  &twos,   twos,  twosA,  twosB);
        TWK_CSA128(&twosA,   &ones,   ones,  (_mm_loadu_si128(&data1[i+4])  & _mm_loadu_si128(&data2[i+4])), (_mm_loadu_si128(&data1[i+5]) & _mm_loadu_si128(&data2[i+5])));
        TWK_CSA128(&twosB,   &ones,   ones,  (_mm_loadu_si128(&data1[i+6])  & _mm_loadu_si128(&data2[i+6])), (_mm_loadu_si128(&data1[i+7]) & _mm_loadu_si128(&data2[i+7])));
        TWK_CSA128(&foursB,  &twos,   twos,  twosA,  twosB);
        TWK_CSA128(&eightsA, &fours,  fours, foursA, foursB);
        TWK_CSA128(&twosA,   &ones,   ones,  (_mm_loadu_si128(&data1[i+8] ) & _mm_loadu_si128(&data2[i+8])),  (_mm_loadu_si128(&data1[i+9] ) & _mm_loadu_si128(&data2[i+9])));
        TWK_CSA128(&twosB,   &ones,   ones,  (_mm_loadu_si128(&data1[i+10]) & _mm_loadu_si128(&data2[i+10])), (_mm_loadu_si128(&data1[i+11]) & _mm_loadu_si128(&data2[i+11])));
        TWK_CSA128(&foursA,  &twos,   twos,  twosA,  twosB);
        TWK_CSA128(&twosA,   &ones,   ones,  (_mm_loadu_si128(&data1[i+12]) & _mm_loadu_si128(&data2[i+12])), (_mm_loadu_si128(&data1[i+13]) & _mm_loadu_si128(&data2[i+13])));
        TWK_CSA128(&twosB,   &ones,   ones,  (_mm_loadu_si128(&data1[i+14]) & _mm_loadu_si128(&data2[i+14])), (_mm_loadu_si128(&data1[i+15]) & _mm_loadu_si128(&data2[i+15])));
        TWK_CSA128(&foursB,  &twos,   twos,  twosA,  twosB);
        TWK_CSA128(&eightsB, &fours,  fours, foursA, foursB);
        TWK_CSA128(&sixteens,&eights, eights,eightsA,eightsB);

        cnt64 += TWK_POPCOUNT_SSE(sixteens);
    }

    cnt64 <<= 4;
    cnt64 += TWK_POPCOUNT_SSE(eights) << 3;
    cnt64 += TWK_POPCOUNT_SSE(fours)  << 2;
    cnt64 += TWK_POPCOUNT_SSE(twos)   << 1;
    cnt64 += TWK_POPCOUNT_SSE(ones)   << 0;

    for (/**/; i < size; ++i)
        cnt64 = TWK_POPCOUNT_SSE(_mm_loadu_si128(&data1[i]) & _mm_loadu_si128(&data2[i]));

    return cnt64;
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("sse4.2")))
#endif
static 
uint64_t TWK_intersect_sse4(const uint64_t* TWK_RESTRICT b1, 
                            const uint64_t* TWK_RESTRICT b2, 
                            const uint32_t n_ints) 
{
    uint64_t count = 0;
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    count += TWK_intersect_csa_sse4(r1, r2, n_cycles);

    for (int i = n_cycles*2; i < n_ints; ++i) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
    }

    return(count);
}
#endif

/****************************
*  AVX256 functions
****************************/

#if defined(HAVE_AVX2)

#include <immintrin.h>

#ifndef TWK_POPCOUNT_AVX2
#define TWK_POPCOUNT_AVX2(A, B) {                  \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 0)); \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 1)); \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 2)); \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 3)); \
}
#endif

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
TWK_FORCE_INLINE 
void TWK_CSA256(__m256i* h, __m256i* l, __m256i a, __m256i b, __m256i c) {
    __m256i u = _mm256_xor_si256(a, b);
    *h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    *l = _mm256_xor_si256(u, c);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
__m256i TWK_popcnt256(__m256i v) {
    __m256i lookup1 = _mm256_setr_epi8(
        4, 5, 5, 6, 5, 6, 6, 7,
        5, 6, 6, 7, 6, 7, 7, 8,
        4, 5, 5, 6, 5, 6, 6, 7,
        5, 6, 6, 7, 6, 7, 7, 8
    );

    __m256i lookup2 = _mm256_setr_epi8(
        4, 3, 3, 2, 3, 2, 2, 1,
        3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1,
        3, 2, 2, 1, 2, 1, 1, 0
    );

    __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    __m256i popcnt1 = _mm256_shuffle_epi8(lookup1, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup2, hi);

    return _mm256_sad_epu8(popcnt1, popcnt2);
}

// modified from https://github.com/WojciechMula/sse-popcount
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static
uint64_t TWK_intersect_lookup_avx2_func(const uint8_t* TWK_RESTRICT data1, 
                                        const uint8_t* TWK_RESTRICT data2, 
                                        const size_t n)
{

    size_t i = 0;

    const __m256i lookup = _mm256_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    __m256i acc = _mm256_setzero_si256();

#define ITER { \
        const __m256i vec = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(data1 + i)), \
            _mm256_loadu_si256((const __m256i*)(data2 + i))); \
        const __m256i lo  = _mm256_and_si256(vec, low_mask); \
        const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask); \
        const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo); \
        const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi); \
        local = _mm256_add_epi8(local, popcnt1); \
        local = _mm256_add_epi8(local, popcnt2); \
        i += 32; \
    }

    while (i + 8*32 <= n) {
        __m256i local = _mm256_setzero_si256();
        ITER ITER ITER ITER
        ITER ITER ITER ITER
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));
    }

    __m256i local = _mm256_setzero_si256();

    while (i + 32 <= n) {
        ITER;
    }

    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));

#undef ITER

    uint64_t result = 0;

    result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 0));
    result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 1));
    result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 2));
    result += static_cast<uint64_t>(_mm256_extract_epi64(acc, 3));

    for (/**/; i < n; i++) {
        result += TWK_lookup8bit[data1[i] & data2[i]];
    }

    return result;
}

/*
 * AVX2 Harley-Seal popcount (4th iteration).
 * The algorithm is based on the paper "Faster Population Counts
 * using AVX2 Instructions" by Daniel Lemire, Nathan Kurz and
 * Wojciech Mula (23 Nov 2016).
 * @see https://arxiv.org/abs/1611.07612
 */
// In this version we perform the operation A&B as input into the CSA operator.
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t TWK_intersect_csa_avx2(const __m256i* TWK_RESTRICT data1, 
                                const __m256i* TWK_RESTRICT data2, 
                                uint64_t size)
{
    __m256i cnt      = _mm256_setzero_si256();
    __m256i ones     = _mm256_setzero_si256();
    __m256i twos     = _mm256_setzero_si256();
    __m256i fours    = _mm256_setzero_si256();
    __m256i eights   = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t* cnt64;

    for (/**/; i < limit; i += 16) {
        TWK_CSA256(&twosA,   &ones,   ones,  (_mm256_loadu_si256(&data1[i+0])  & _mm256_loadu_si256(&data2[i+0])), (_mm256_loadu_si256(&data1[i+1]) & _mm256_loadu_si256(&data2[i+1])));  
        TWK_CSA256(&twosB,   &ones,   ones,  (_mm256_loadu_si256(&data1[i+2])  & _mm256_loadu_si256(&data2[i+2])), (_mm256_loadu_si256(&data1[i+3]) & _mm256_loadu_si256(&data2[i+3])));
        TWK_CSA256(&foursA,  &twos,   twos,  twosA, twosB);
        TWK_CSA256(&twosA,   &ones,   ones,  (_mm256_loadu_si256(&data1[i+4])  & _mm256_loadu_si256(&data2[i+4])), (_mm256_loadu_si256(&data1[i+5]) & _mm256_loadu_si256(&data2[i+5])));
        TWK_CSA256(&twosB,   &ones,   ones,  (_mm256_loadu_si256(&data1[i+6])  & _mm256_loadu_si256(&data2[i+6])), (_mm256_loadu_si256(&data1[i+7]) & _mm256_loadu_si256(&data2[i+7])));
        TWK_CSA256(&foursB,  &twos,   twos,  twosA, twosB);
        TWK_CSA256(&eightsA, &fours,  fours, foursA, foursB);
        TWK_CSA256(&twosA,   &ones,   ones,  (_mm256_loadu_si256(&data1[i+8] ) & _mm256_loadu_si256(&data2[i+8])), (_mm256_loadu_si256(&data1[i+9]  ) & _mm256_loadu_si256(&data2[i+9])));
        TWK_CSA256(&twosB,   &ones,   ones,  (_mm256_loadu_si256(&data1[i+10]) & _mm256_loadu_si256(&data2[i+10])), (_mm256_loadu_si256(&data1[i+11]) & _mm256_loadu_si256(&data2[i+11])));
        TWK_CSA256(&foursA,  &twos,   twos,  twosA, twosB);
        TWK_CSA256(&twosA,   &ones,   ones,  (_mm256_loadu_si256(&data1[i+12]) & _mm256_loadu_si256(&data2[i+12])), (_mm256_loadu_si256(&data1[i+13]) & _mm256_loadu_si256(&data2[i+13])));
        TWK_CSA256(&twosB,   &ones,   ones,  (_mm256_loadu_si256(&data1[i+14]) & _mm256_loadu_si256(&data2[i+14])), (_mm256_loadu_si256(&data1[i+15]) & _mm256_loadu_si256(&data2[i+15])));
        TWK_CSA256(&foursB,  &twos,   twos,  twosA, twosB);
        TWK_CSA256(&eightsB, &fours,  fours, foursA, foursB);
        TWK_CSA256(&sixteens,&eights, eights,eightsA, eightsB);

        cnt = _mm256_add_epi64(cnt, TWK_popcnt256(sixteens));
    }

    cnt = _mm256_slli_epi64(cnt, 4);
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(TWK_popcnt256(eights), 3));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(TWK_popcnt256(fours),  2));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(TWK_popcnt256(twos),   1));
    cnt = _mm256_add_epi64(cnt, TWK_popcnt256(ones));

    for (/**/; i < size; ++i)
        cnt = _mm256_add_epi64(cnt, TWK_popcnt256(_mm256_loadu_si256(&data1[i]) & _mm256_loadu_si256(&data2[i])));

    cnt64 = (uint64_t*) &cnt;

    return cnt64[0] +
            cnt64[1] +
            cnt64[2] +
            cnt64[3];
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
static 
uint64_t TWK_intersect_avx2(const uint64_t* TWK_RESTRICT b1, 
                   const uint64_t* TWK_RESTRICT b2, 
                   const uint32_t n_ints)
{
    uint64_t count = 0;
    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;
    const uint32_t n_cycles = n_ints / 4;

    count += TWK_intersect_csa_avx2(r1, r2, n_cycles);

    for (int i = n_cycles*4; i < n_ints; ++i) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
    }

    return(count);
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx2")))
#endif
TWK_FORCE_INLINE 
uint64_t TWK_intersect_lookup_avx2(const uint64_t* TWK_RESTRICT b1, 
                                   const uint64_t* TWK_RESTRICT b2, 
                                   const uint32_t n_ints)
{
    return TWK_intersect_lookup_avx2_func((uint8_t*)b1, (uint8_t*)b2, n_ints*8);
}
#endif

/****************************
*  AVX512BW functions
****************************/

#if defined(HAVE_AVX512)

#include <immintrin.h>

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
TWK_FORCE_INLINE  
__m512i TWK_popcnt512(__m512i v) {
    __m512i m1 = _mm512_set1_epi8(0x55);
    __m512i m2 = _mm512_set1_epi8(0x33);
    __m512i m4 = _mm512_set1_epi8(0x0F);
    __m512i t1 = _mm512_sub_epi8(v,       (_mm512_srli_epi16(v, 1)   & m1));
    __m512i t2 = _mm512_add_epi8(t1 & m2, (_mm512_srli_epi16(t1, 2)  & m2));
    __m512i t3 = _mm512_add_epi8(t2,       _mm512_srli_epi16(t2, 4)) & m4;

    return _mm512_sad_epu8(t3, _mm512_setzero_si512());
}

#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
TWK_FORCE_INLINE  
void TWK_CSA512(__m512i* h, __m512i* l, __m512i a, __m512i b, __m512i c) {
    *l = _mm512_ternarylogic_epi32(c, b, a, 0x96);
    *h = _mm512_ternarylogic_epi32(c, b, a, 0xe8);
}

/*
 * AVX512 Harley-Seal popcount (4th iteration).
 * The algorithm is based on the paper "Faster Population Counts
 * using AVX2 Instructions" by Daniel Lemire, Nathan Kurz and
 * Wojciech Mula (23 Nov 2016).
 * @see https://arxiv.org/abs/1611.07612
 */
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static 
uint64_t TWK_intersect_csa_avx512(const __m512i* TWK_RESTRICT data1, 
                                  const __m512i* TWK_RESTRICT data2, 
                                  uint64_t size)
{
    __m512i cnt      = _mm512_setzero_si512();
    __m512i ones     = _mm512_setzero_si512();
    __m512i twos     = _mm512_setzero_si512();
    __m512i fours    = _mm512_setzero_si512();
    __m512i eights   = _mm512_setzero_si512();
    __m512i sixteens = _mm512_setzero_si512();
    __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

    uint64_t i = 0;
    uint64_t limit = size - size % 16;
    uint64_t* cnt64;

    for (/**/; i < limit; i += 16) {
        TWK_CSA512(&twosA,   &ones,   ones,  (_mm512_loadu_si512(&data1[i+0])  & _mm512_loadu_si512(&data2[i+0])), (_mm512_loadu_si512(&data1[i+1]) & _mm512_loadu_si512(&data2[i+1])));
        TWK_CSA512(&twosB,   &ones,   ones,  (_mm512_loadu_si512(&data1[i+2])  & _mm512_loadu_si512(&data2[i+2])), (_mm512_loadu_si512(&data1[i+3]) & _mm512_loadu_si512(&data2[i+3])));
        TWK_CSA512(&foursA,  &twos,   twos,  twosA, twosB);
        TWK_CSA512(&twosA,   &ones,   ones,  (_mm512_loadu_si512(&data1[i+4])  & _mm512_loadu_si512(&data2[i+4])), (_mm512_loadu_si512(&data1[i+5]) & _mm512_loadu_si512(&data2[i+5])));
        TWK_CSA512(&twosB,   &ones,   ones,  (_mm512_loadu_si512(&data1[i+6])  & _mm512_loadu_si512(&data2[i+6])), (_mm512_loadu_si512(&data1[i+7]) & _mm512_loadu_si512(&data2[i+7])));
        TWK_CSA512(&foursB,  &twos,   twos,  twosA, twosB);
        TWK_CSA512(&eightsA, &fours,  fours, foursA, foursB);
        TWK_CSA512(&twosA,   &ones,   ones,  (_mm512_loadu_si512(&data1[i+8] ) & _mm512_loadu_si512(&data2[i+8])), (_mm512_loadu_si512(&data1[i+9]  ) & _mm512_loadu_si512(&data2[i+9])));
        TWK_CSA512(&twosB,   &ones,   ones,  (_mm512_loadu_si512(&data1[i+10]) & _mm512_loadu_si512(&data2[i+10])), (_mm512_loadu_si512(&data1[i+11]) & _mm512_loadu_si512(&data2[i+11])));
        TWK_CSA512(&foursA,  &twos,   twos,  twosA, twosB);
        TWK_CSA512(&twosA,   &ones,   ones,  (_mm512_loadu_si512(&data1[i+12]) & _mm512_loadu_si512(&data2[i+12])), (_mm512_loadu_si512(&data1[i+13]) & _mm512_loadu_si512(&data2[i+13])));
        TWK_CSA512(&twosB,   &ones,   ones,  (_mm512_loadu_si512(&data1[i+14]) & _mm512_loadu_si512(&data2[i+14])), (_mm512_loadu_si512(&data1[i+15]) & _mm512_loadu_si512(&data2[i+15])));
        TWK_CSA512(&foursB,  &twos,   twos,  twosA, twosB);
        TWK_CSA512(&eightsB, &fours,  fours, foursA, foursB);
        TWK_CSA512(&sixteens,&eights, eights,eightsA, eightsB);

        cnt = _mm512_add_epi64(cnt, TWK_popcnt512(sixteens));
    }

    cnt = _mm512_slli_epi64(cnt, 4);
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(TWK_popcnt512(eights), 3));
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(TWK_popcnt512(fours), 2));
    cnt = _mm512_add_epi64(cnt, _mm512_slli_epi64(TWK_popcnt512(twos), 1));
    cnt = _mm512_add_epi64(cnt,  TWK_popcnt512(ones));

    for (/**/; i < size; ++i)
        cnt = _mm512_add_epi64(cnt, TWK_popcnt512(_mm512_loadu_si512(&data1[i]) & _mm512_loadu_si512(&data2[i])));

    cnt64 = (uint64_t*)&cnt;

    return cnt64[0] +
            cnt64[1] +
            cnt64[2] +
            cnt64[3] +
            cnt64[4] +
            cnt64[5] +
            cnt64[6] +
            cnt64[7];
}

// Functions
// AVX512
#if !defined(_MSC_VER)
  __attribute__ ((target ("avx512bw")))
#endif
static 
uint64_t TWK_intersect_avx512(const uint64_t* TWK_RESTRICT b1, 
                              const uint64_t* TWK_RESTRICT b2, 
                              const uint32_t n_ints) 
{
    uint64_t count = 0;
    const __m512i* r1 = (const __m512i*)(b1);
    const __m512i* r2 = (const __m512i*)(b2);
    const uint32_t n_cycles = n_ints / 8;

    count += TWK_intersect_csa_avx512(r1, r2, n_cycles);

    for (int i = n_cycles*8; i < n_ints; ++i) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
    }

    return(count);
}
#endif

/****************************
*  Scalar functions
****************************/

TWK_FORCE_INLINE 
uint64_t TWK_intersect_scalar(const uint64_t* TWK_RESTRICT b1, 
                              const uint64_t* TWK_RESTRICT b2, 
                              const uint32_t n_ints)
{
    return TWK_intersect_unrolled(b1, b2, n_ints);
}

static
uint64_t TWK_intersect_scalar_list(const uint64_t* TWK_RESTRICT b1, 
                                   const uint64_t* TWK_RESTRICT b2, 
                                   const uint32_t* TWK_RESTRICT l1, 
                                   const uint32_t* TWK_RESTRICT l2,
                                   const uint32_t n1, 
                                   const uint32_t n2) 
{
    uint64_t count = 0;

#define MOD(x) (( (x) * 64 ) >> 6)
    if(n1 < n2) {
        for(int i = 0; i < n1; ++i) {
            count += ((b2[l1[i] >> 6] & (1L << MOD(l1[i]))) != 0); 
            __builtin_prefetch(&b2[l1[i] >> 6], 0, _MM_HINT_T0);
        }
    } else {
        for(int i = 0; i < n2; ++i) {
            count += ((b1[l2[i] >> 6] & (1L << MOD(l2[i]))) != 0);
            __builtin_prefetch(&b1[l2[i] >> 6], 0, _MM_HINT_T0);
        }
    }
#undef MOD
    return(count);
}


/* *************************************
*  Alignment and retrieve intersection function
***************************************/
// Function pointer definitions.
typedef uint64_t (*TWK_intersect_func)(const uint64_t*, const uint64_t*, const uint32_t);
typedef uint64_t (*TWK_intersect_lfunc)(const uint64_t*, const uint64_t*, 
    const uint32_t*, const uint32_t*,
    const uint32_t, const uint32_t);


// Return the best alignment given the available instruction set at
// run-time.
static 
uint32_t TWK_get_alignment() {

#if defined(HAVE_CPUID)
    #if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

        #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
        #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
        #endif
    }
    #endif
#endif

    uint32_t alignment = 0;
#if defined(HAVE_AVX512)
    if ((cpuid & TWK_bit_AVX512BW)) { // 16*512
        alignment = TWK_AVX512_ALIGNMENT;
    }
#endif

#if defined(HAVE_AVX2)
    if ((cpuid & TWK_bit_AVX2) && alignment == 0) { // 16*256
        alignment = TWK_AVX2_ALIGNMENT;
    }
#endif

#if defined(HAVE_SSE42)
    if ((cpuid & TWK_bit_SSE41) && alignment == 0) { // 16*128
        alignment = TWK_SSE_ALIGNMENT;
    }
#endif

    if (alignment == 0) alignment = 8;
    return alignment;
}

// Return the optimal intersection function given the range [0, n_bitmaps_vector)
// and the available instruction set at run-time.
static
TWK_intersect_func TWK_get_intersect_func(const uint32_t n_bitmaps_vector) {
    #if defined(HAVE_CPUID)
    #if defined(__cplusplus)
    /* C++11 thread-safe singleton */
    static const int cpuid = get_cpuid();
    #else
    static int cpuid_ = -1;
    int cpuid = cpuid_;
    if (cpuid == -1) {
        cpuid = get_cpuid();

        #if defined(_MSC_VER)
        _InterlockedCompareExchange(&cpuid_, cpuid, -1);
        #else
        __sync_val_compare_and_swap(&cpuid_, -1, cpuid);
        #endif
    }
    #endif
#endif


#if defined(HAVE_AVX512)
    if ((cpuid & TWK_bit_AVX512BW) && n_bitmaps_vector >= 128) { // 16*512
        return &TWK_intersect_avx512;
    }
#endif

#if defined(HAVE_AVX2)
    if ((cpuid & TWK_bit_AVX2) && n_bitmaps_vector >= 64) { // 16*256
        return &TWK_intersect_avx2;
    }
    
    if ((cpuid & TWK_bit_AVX2) && n_bitmaps_vector >= 4) {
        return &TWK_intersect_lookup_avx2;
    }
#endif

#if defined(HAVE_SSE42)
    if ((cpuid & TWK_bit_SSE41) && n_bitmaps_vector >= 32) { // 16*128
        return &TWK_intersect_sse4;
    }
#endif

    return &TWK_intersect_scalar;
}

/* *************************************
*  Example wrappers
*
*  These wrappers compute sum(popcnt(A & B)) for all N input bitmaps
*  pairwise. All input bitmaps must be of the same length M. The
*  functions starting with TWK_wrapper_diag* assumes that all data
*  comes from the same contiguous memory buffer. Use TWK_wrapper_square*
*  if you have data from two distinct, but contiguous, memory buffers
*  B1 and B2.
*
*  The TWK_wrapper_*_list* functions make use of auxilliary information
*  to accelerate computation when the vectors are very sparse.
*
***************************************/

/**
 * This within-block wrapper computes the n_vectors choose 2
 * pairwise comparisons in a non-blocking fashion. This
 * function is used mostly as reference.
 * 
 * @param n_vectors Number of input vectors
 * @param vals      Pointers to 64-bit bitmaps
 * @param n_ints    Number of bitmaps per vector
 * @param f         Function pointer
 * @return uint64_t Returns the sum total POPCNT(A & B).
 */
static
uint64_t TWK_wrapper_diag(const uint32_t n_vectors, 
                    const uint64_t* vals, 
                    const uint32_t n_ints, 
                    const TWK_intersect_func f)
{
    uint32_t offset = 0;
    uint32_t inner_offset = 0;
    uint64_t total = 0;
    
    for (int i = 0; i < n_vectors; ++i) {
        inner_offset = offset + n_ints;
        for (int j = i + 1; j < n_vectors; ++j, inner_offset += n_ints) {
            total += (*f)(&vals[offset], &vals[inner_offset], n_ints);
        }
        offset += n_ints;
    }

    return total;
}

static
uint64_t TWK_wrapper_square(const uint32_t n_vectors1,
                           const uint64_t* TWK_RESTRICT vals1, 
                           const uint32_t n_vectors2,
                           const uint64_t* TWK_RESTRICT vals2, 
                           const uint32_t n_ints, 
                           const TWK_intersect_func f)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint64_t total   = 0;
    
    for (int i = 0; i < n_vectors1; ++i, offset1 += n_ints) {
        for (int j = 0; j < n_vectors2; ++j, offset2 += n_ints) {
            total += (*f)(&vals1[offset1], &vals2[offset2], n_ints);
        }
    }

    return total;
}

/**
 * This within-block function is identical to c_fwrapper but additionally
 * make use of an auxilliary positional vector to accelerate compute in
 * the sparse case. This function is mostly used as reference.
 * 
 * @param n_vectors 
 * @param vals 
 * @param n_ints 
 * @param n_alts 
 * @param alt_positions 
 * @param alt_offsets 
 * @param f 
 * @param fl 
 * @param cutoff 
 * @return uint64_t 
 */
static
uint64_t TWK_wrapper_diag_list(const uint32_t n_vectors, 
                     const uint64_t* TWK_RESTRICT vals,
                     const uint32_t n_ints,
                     const uint32_t* TWK_RESTRICT n_alts,
                     const uint32_t* TWK_RESTRICT alt_positions,
                     const uint32_t* TWK_RESTRICT alt_offsets, 
                     const TWK_intersect_func f, 
                     const TWK_intersect_lfunc fl, 
                     const uint32_t cutoff)
{
    uint64_t offset1 = 0;
    uint64_t offset2 = n_ints;
    uint64_t count = 0;

    for (int i = 0; i < n_vectors; ++i, offset1 += n_ints) {
        offset2 = offset1 + n_ints;
        for (int j = i+1; j < n_vectors; ++j, offset2 += n_ints) {
            if (n_alts[i] <= cutoff || n_alts[j] <= cutoff) {
                count += (*fl)(&vals[offset1], 
                               &vals[offset2], 
                               &alt_positions[alt_offsets[i]], 
                               &alt_positions[alt_offsets[j]], 
                               n_alts[i], n_alts[j]);
            } else {
                count += (*f)(&vals[offset1], &vals[offset2], n_ints);
            }
        }
    }
    return count;
}

static
uint64_t TWK_wrapper_diag_blocked(const uint32_t n_vectors, 
                            const uint64_t* vals, 
                            const uint32_t n_ints, 
                            const TWK_intersect_func f,
                            uint32_t block_size)
{
    uint64_t total = 0;

    block_size = (block_size == 0 ? 3 : block_size);
    const uint32_t n_blocks1 = n_vectors / block_size;
    const uint32_t n_blocks2 = n_vectors / block_size;

    uint32_t i  = 0;
    uint32_t tt = 0;
    for (/**/; i + block_size <= n_vectors; i += block_size) {
        // diagonal component
        uint32_t left = i*n_ints;
        uint32_t right = 0;
        for (uint32_t j = 0; j < block_size; ++j, left += n_ints) {
            right = left + n_ints;
            for (uint32_t jj = j + 1; jj < block_size; ++jj, right += n_ints) {
                total += (*f)(&vals[left], &vals[right], n_ints);
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + block_size;
        for (/**/; j + block_size <= n_vectors; j += block_size) {
            left = curi*n_ints;
            for (uint32_t ii = 0; ii < block_size; ++ii, left += n_ints) {
                right = j*n_ints;
                for (uint32_t jj = 0; jj < block_size; ++jj, right += n_ints) {
                    total += (*f)(&vals[left], &vals[right], n_ints);
                }
            }
        }

        // residual
        right = j*n_ints;
        for (/**/; j < n_vectors; ++j, right += n_ints) {
            left = curi*n_ints;
            for (uint32_t jj = 0; jj < block_size; ++jj, left += n_ints) {
                total += (*f)(&vals[left], &vals[right], n_ints);
            }
        }
    }
    // residual tail
    uint32_t left = i*n_ints;
    for (/**/; i < n_vectors; ++i, left += n_ints) {
        uint32_t right = left + n_ints;
        for (uint32_t j = i + 1; j < n_vectors; ++j, right += n_ints) {
            total += (*f)(&vals[left], &vals[right], n_ints);
        }
    }

    return total;
}

static
uint64_t TWK_wrapper_diag_list_blocked(const uint32_t n_vectors, 
                             const uint64_t* TWK_RESTRICT vals,
                             const uint32_t n_ints,
                             const uint32_t* TWK_RESTRICT n_alts,
                             const uint32_t* TWK_RESTRICT alt_positions,
                             const uint32_t* TWK_RESTRICT alt_offsets, 
                             const TWK_intersect_func f, 
                             const TWK_intersect_lfunc fl, 
                             const uint32_t cutoff,
                             uint32_t block_size)
{
    uint64_t total = 0;

    block_size = (block_size == 0 ? 3 : block_size);
    const uint32_t n_blocks1 = n_vectors / block_size;
    const uint32_t n_blocks2 = n_vectors / block_size;

    uint32_t i  = 0;
    uint32_t tt = 0;

    for (/**/; i + block_size <= n_vectors; i += block_size) {
        // diagonal component
        uint32_t left = i*n_ints;
        uint32_t right = 0;
        for (uint32_t j = 0; j < block_size; ++j, left += n_ints) {
            right = left + n_ints;
            for (uint32_t jj = j + 1; jj < block_size; ++jj, right += n_ints) {
                if (n_alts[i+j] < cutoff || n_alts[i+jj] < cutoff) {
                    total += (*fl)(&vals[left], &vals[right], 
                        &alt_positions[alt_offsets[i+j]], &alt_positions[alt_offsets[i+jj]], 
                        n_alts[i+j], n_alts[i+jj]);
                } else {
                    total += (*f)(&vals[left], &vals[right], n_ints);
                }
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + block_size;
        for (/**/; j + block_size <= n_vectors; j += block_size) {
            left = curi*n_ints;
            for (uint32_t ii = 0; ii < block_size; ++ii, left += n_ints) {
                right = j*n_ints;
                for (uint32_t jj = 0; jj < block_size; ++jj, right += n_ints) {
                    if (n_alts[curi+ii] < cutoff || n_alts[j+jj] < cutoff) {
                        total += (*fl)(&vals[left], &vals[right], 
                            &alt_positions[alt_offsets[curi+ii]], &alt_positions[alt_offsets[j+jj]], 
                            n_alts[curi+ii], n_alts[j+jj]);
                    } else {
                        total += (*f)(&vals[left], &vals[right], n_ints);
                    }
                }
            }
        }

        // residual
        right = j*n_ints;
        for (/**/; j < n_vectors; ++j, right += n_ints) {
            left = curi*n_ints;
            for (uint32_t jj = 0; jj < block_size; ++jj, left += n_ints) {
                if (n_alts[curi+jj] < cutoff || n_alts[j] < cutoff) {
                    total += (*fl)(&vals[left], &vals[right], 
                        &alt_positions[alt_offsets[curi+jj]], &alt_positions[alt_offsets[j]], 
                        n_alts[curi+jj], n_alts[j]);
                } else {
                    total += (*f)(&vals[left], &vals[right], n_ints);
                }
            }
        }
    }
    // residual tail
    uint32_t left = i*n_ints;
    for (/**/; i < n_vectors; ++i, left += n_ints) {
        uint32_t right = left + n_ints;
        for (uint32_t j = i + 1; j < n_vectors; ++j, right += n_ints) {
            if (n_alts[i] < cutoff || n_alts[j] < cutoff) {
                total += (*fl)(&vals[left], &vals[right], 
                    &alt_positions[alt_offsets[i]], &alt_positions[alt_offsets[j]], 
                    n_alts[i], n_alts[j]);
            } else {
                total += (*f)(&vals[left], &vals[right], n_ints);
            }
        }
    }

    return total;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FAST_INTERSECT_COUNT_H_ */
