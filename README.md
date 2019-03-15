# FastIntersectCount

These functions compute the set intersection count of pairs of integer sets with equal upper bounds [0,m). Several of the functions presented here exploit set sparsity by using auxiliary information such as positional indices, bitmaps, or reduction preprocessors. There are no union and difference routines in this repository.

Compile test suite with: `make` and run `./fast_intersect_count`

### History

These functions were originally developed for [Tomahawk](https://github.com/mklarqvist/Tomahawk) for computing genome-wide linkage-disequilibrium but can be applied to any intersect-count problem.

---

## Problem statement

To implement a set of integers, a particularly appealing strategy is the bitmap (also called bitset or bit vector). Using n bits, we can represent any set made of the integers from the range [0,n): it suffices to set the ith bit is set to one if integer i is present in the set. Commodity processors use words of W=32 or W=64 bits. By combining many such words, we can support large values of n. Intersections, unions and differences can then be implemented as bitwise AND, OR and ANDNOT operations.

In [Tomahawk](https://github.com/mklarqvist/Tomahawk), we must compute N!2 set intersections of size M, where N and M are typically in the many millions and thousands, respectively.

## Goals

* Achieve high-performance on large arrays of values.
* Support both small and large sets (sparse and dense).

## Technical approach

In all the methods below, `b1` and `b2` are pointers to the start of each `uint64_t` vector and `n_ints` is the length of the vectors.

### Approach 0a: Scalar bitmap accumulator

We compare our proposed algorithms to a naive implementation using standard incrementors:

```python
count = 0
for i in 1..n # n -> n_records
    count += TWK_POPCOUNT(set1[j] & set2[j])
```

This simple code will optimize extremely well on most machines. Knowledge of the host-architecture by the compiler makes this codes difficult to outperform on average.

### Aproach 0b: Loop unrolled + software pipelined scalar accumulator

Example 1x4 approach
```c++
uint64_t count = 0;
int i = 0;
for(; i < (n_ints & ~3); i += 4) {
    count += TWK_POPCOUNT(b1[i+0] & b2[i+0]);
    count += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
    count += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
    count += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
}
for(; i < n_ints; ++i)
    count += TWK_POPCOUNT(b1[i+0] & b2[i+0]);

return(count);
```

### Approach 1: SIMD-acceleration of bitmap accumulator

Set intersections of bitmaps can be trivially vectorized with all available SIMD-instruction sets. The bit-wise population count (`popcnt`) operation consumes most of the elapsed time.

Example C++ implementation using SSE4.1 instructions:
```c++
uint64_t count = 0;
const __m128i* r1 = (__m128i*)b1;
const __m128i* r2 = (__m128i*)b2;
const uint32_t n_cycles = n_ints / 2;

for(int i = 0; i < n_cycles; ++i) {
    TWK_POPCOUNT_SSE(count, _mm_and_si128(r1[i], r2[i]));
}

return(count);
```

Example C++ implementation using AVX2 instructions:
```c++
uint64_t count = 0;
const __m256i* r1 = (__m256i*)b1;
const __m256i* r2 = (__m256i*)b2;
const uint32_t n_cycles = n_ints / 4;

for(int i = 0; i < n_cycles; ++i) {
    TWK_POPCOUNT_AVX2(count, _mm256_and_si256(r1[i], r2[i]));
}

return(count);
```

Example C++ implementation using AVX-512 instructions and a partial sums accumulator:
```c++
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
```

### Approach 2: Bitmap accumulator with a positional index

For sparse set comparisons we can apply a form of search space reduction by storing an additional positional index for each set storing the offsets for set bits, enabling O(1)-time random-access lookups. Let `l1` and `l2` be vectors of positional indices. Logically, we can further limit our search space by using the positions in the smallest index to query intersections. When operating on the bit-level, the positional index is the input integer set. For the other non-bit-wise approaches, the relationship is input integer / bits per primitive.

Example scalar implementation in C++ using a positional index for individual bits:
```c++
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
```

Example scalar implementation in C++ using a positional index for individual primitives:
```c++
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
```

Example scalar implementation in C++ using a positional index for 128-bit registers:
```c++
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
```

Example scalar implementation in C++ using a positional index for 256-bit registers:
```c++
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
```

Example scalar implementation in C++ using a positional index for 512-bit registers and a partial sums accumulator:
```c++
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
```

### Approach 3: Reduce-intersect preprocessor

Reduce bitmap of size N to size M such that M << N.

```c++
uint64_t count = 0;

for(int i = 0; i < n_squash; ++i)
    count += ((sq1[i] & sq2[i]) != 0);

if(count == 0) return 0;

return((*f)(b1,b2,l1,l2)); // intersect using target function pointer
```

### Approach 4: Compare run-length encoded sets

We can compare two run-length encoded sets, A and B, in O(|A| + |B| + 1)-time.

```c++
int_t lenA = (rle1[0] >> 1);
int_t lenB = (rle2[0] >> 1);
uint32_t offsetA = 0;
uint32_t offsetB = 0;
const size_t limA = rle1.size() - 1;
const size_t limB = rle2.size() - 1;
uint64_t ltot = 0;

while(true) {
    if(lenA > lenB) {
        // Subtract current length of B from current length of A
        lenA -= lenB;
        // Branchless predicate multiplication of partial sum
        ltot += lenB * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));
        lenB  = rle2[++offsetB] >> 1; // Retrieve new length for B
    } else if(lenA < lenB) {
        // Subtract current length of A from current length of B
        lenB -= lenA;
        ltot += lenA * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));
        lenA  = rle1[++offsetA] >> 1; // Retrieve new length for A
    } else { // length equality
        ltot += lenB * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));
        lenA  = rle1[++offsetA] >> 1; // Retrieve new length for A
        lenB  = rle2[++offsetB] >> 1; // Retrieve new length for B
    }

    if(offsetA == limA && offsetB == limB) break; // Both sets are empty
}

return(ltot);
```

Branchless inner code:
```c++
int_t lenA = (rle1[0] >> 1);
int_t lenB = (rle2[0] >> 1);
uint32_t offsetA = 0;
uint32_t offsetB = 0;
const size_t limA = rle1.size() - 1;
const size_t limB = rle2.size() - 1;
int64_t ltot = 0;

// Internal parameters.
int_t lA = 0, lB = 0;
bool predicate1 = false, predicate2 = false;

while(true) {
    lA = lenA, lB = lenB;
    predicate1 = (lA >= lB);
    predicate2 = (lB >= lA);
    
    // Use predicate multiplication to compute number of overlaps.
    ltot += (predicate1 * lB + !predicate1 * lA) * ((rle1[offsetA] & 1) & (rle2[offsetB] & 1));

    // Update the correct offset.
    offsetB += predicate1;
    offsetA += predicate2;

    // Use predicate multiplications to update the correct lenghts.
    lenA -= predicate1 * lB + !predicate1 * lA;
    lenB -= predicate2 * lA + !predicate2 * lB;
    lenA += predicate2 * (rle1[offsetA] >> 1);
    lenB += predicate1 * (rle2[offsetB] >> 1);

    if(offsetA == limA && offsetB == limB) break;
}

return(ltot);
```

### Approach 5: Reduced integer set intersection

Compare 64-bit encoded integer sets directly and perform the set intersection on those bitmaps that overlap.

```c++
uint64_t count = 0; 

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
} // else not shown (symmetry of above)
```

SSE4 version:
```c++
const __m128i full_vec = _mm_set1_epi16(0xFFFF);
const __m128i one_mask = _mm_set1_epi16(1);
const __m128i range    = _mm_set_epi16(8,7,6,5,4,3,2,1);
uint64_t count = 0;

if(l1.size() < l2.size()) {
    const __m128i* y = (const __m128i*)&l2[0];
    const uint32_t n_y = l2.size() / 8; // 128 / 16 vectors

    for(int i = 0; i < l1.size(); ++i) {
        const __m128i x = _mm_set1_epi16(l1[i]); // Broadcast single reference value
        int j = 0;
        for(; j < n_y; ++j) {
            if(l2[j*8] > l1[i]) goto done; // if the current value is larger than the reference value break
            __m128i cmp = _mm_cmpeq_epi16(x, y[j]);
            // Predicate check for all empty bits
            if(_mm_testz_si128(cmp, full_vec) == false) {
                // Compute overlap position by predicate-multiplication
                const __m128i v = _mm_mullo_epi16(_mm_and_si128(cmp, one_mask), range);
                const uint16_t* vv = (const uint16_t*)&v;
                // Horizontal sum of predicate-multiplication
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
                count += TWK_POPCOUNT(b1[i] & b2[j]);
                goto done;
            }
        }
        done:
        continue;
    }
} // else not shown (symmetry of above)
```

### Approach 6: Auxiliary prefix- and suffix-run lengths

Space reduction by limiting the intersection operation to the range [max(prefixA,prefixB), min(suffixA,suffixB)].

```c++
const uint32_t from = std::max(p1.first, p2.first);
const uint32_t to   = std::min(p1.second,p2.second);

uint64_t count = 0;
for(int i = from; i < to; ++i) {
    count += TWK_POPCOUNT(b1[i] & b2[i]);
}

return(count);
```

### Results



### Reference systems information

Intel Xeon Skylake
```bash
$ lscpu
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                60
On-line CPU(s) list:   0-59
Thread(s) per core:    1
Core(s) per socket:    1
Socket(s):             60
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 85
Model name:            Intel Xeon Processor (Skylake, IBRS)
Stepping:              4
CPU MHz:               2599.998
BogoMIPS:              5199.99
Hypervisor vendor:     KVM
Virtualization type:   full
L1d cache:             32K
L1i cache:             32K
L2 cache:              4096K
L3 cache:              16384K
NUMA node0 CPU(s):     0-59
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology eagerfpu pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single spec_ctrl ibpb_support fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 arat
```
```bash
$ hostnamectl
    Virtualization: kvm
  Operating System: Red Hat Enterprise Linux
       CPE OS Name: cpe:/o:redhat:enterprise_linux:7.4:GA:server
            Kernel: Linux 3.10.0-693.21.1.el7.x86_64
      Architecture: x86-64
```

MacBook Air
```bash
$ sysctl -n machdep.cpu.brand_string
Intel(R) Core(TM) i5-3427U CPU @ 1.80GHz

$ sysctl -a | grep cpu.feat
machdep.cpu.feature_bits: 9203919476061109247
machdep.cpu.features: FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 CLFSH DS ACPI MMX FXSR SSE SSE2 SS HTT TM PBE SSE3 PCLMULQDQ DTES64 MON DSCPL VMX SMX EST TM2 SSSE3 CX16 TPR PDCM SSE4.1 SSE4.2 x2APIC POPCNT AES PCID XSAVE OSXSAVE TSCTMR AVX1.0 RDRAND F16C
```

```bash
$ system_profiler SPSoftwareDataType
Software:

    System Software Overview:

      System Version: macOS 10.14.3 (18D109)
      Kernel Version: Darwin 18.2.0
```