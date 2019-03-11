# FastIntersectCount

These functions compute the set intersection count of pairs of **equal-sized** integer sets.  
Compile test suite with: `make` and run `./fast_intersect_count`

## Computing set intersections

## Problem statement

## Goals

* Achieve high-performance on large arrays of values.

## Technical approach

In all the methods below, `b1` and `b2` are pointers to the start of each `uint64_t` vector and `n_ints` is the length of the vectors.

### Approach 0: Naive bitmap iterator (scalar)

We compare our proposed algorithms to a naive implementation using standard incrementors:

```python
count = 0
for i in 1..n # n -> n_records
    count += TWK_POPCOUNT(set1[j] & set2[j])
```

This simple code will optimize extremely well on most machines. Knowledge of the host-architecture by the compiler makes this codes difficult to outperform on average.

### Approach 1: SIMD-acceleration of bitmaps

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

### Approach 2: Bitmap and positional index

Let `l1` and `l2` be vectors of positional indices.

Scalar + bit-index
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

Scalar + primitive-index
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

SSE4 + SSE4-index
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

AVX2 + AVX2-index
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

AVX512 + AVX512-index
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

### Approach 3: Bitmap, positional index, and reduction test

Reduce bitmap of size N to size M such that M << N.

### Approach 4: Compare run-length encoded sets

### Results

