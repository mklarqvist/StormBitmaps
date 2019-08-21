[![Build Status](https://travis-ci.com/mklarqvist/StormBitmaps.svg)](https://travis-ci.com/mklarqvist/StormBitmaps)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/mklarqvist/StormBitmaps?branch=master&svg=true)](https://ci.appveyor.com/project/mklarqvist/StormBitmaps)
[![Github Releases](https://img.shields.io/github/release/mklarqvist/StormBitmaps.svg)](https://github.com/mklarqvist/StormBitmaps/releases)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

# Storm bitmaps

These algorithms and bitmaps are used to compute XX<sup>T</sup> for a _binary_ input matrix with dimensions (N,M) using specialized CPU instructions i.e.
[POPCNT](https://en.wikipedia.org/wiki/SSE4#POPCNT_and_LZCNT),
[SSE4.2](https://en.wikipedia.org/wiki/SSE4#SSE4.2),
[AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions),
[AVX512BW](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions),
[NEON](https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_.28NEON.29). This is equivalent to computing the all-vs-all set intersection cardinality (|X<sub>i</sub> ∩ X<sup>T</sup><sub>j</sub>|) for pairs of _symmetric_ integer sets. These algorithms are fast in the worst case and _extremely_ fast when the input matrix is sparse. 

![screenshot](binary_matrix_multiplication.jpg)

Using large registers (AVX-512BW), contiguous and aligned memory, and
cache-aware blocking, we can achieve ~114 GB/s (~0.2 CPU cycles / 64-bit word)
of sustained throughput (~14 billion 64-bit bitmaps / second or up to ~912
billion implicit integers / second) using `STORM_contiguous_t` when the input
data is small (N < 256,000). When input data is large, we can achieve around
0.4-0.6 CPU cycles / 64-bit word using `STORM_t` while using considerably less
memory. Both of these models make use of scalar-bitmap or scalar-scalar
comparisons when the data density is small. Storm selects the optimal memory
alignment and subroutines given the available SIMD instruction at run-time by
using [libalgebra](https://github.com/mklarqvist/libalgebra).

The core algorithms are described in the papers:

* [Faster Population Counts using AVX2 Instructions](https://arxiv.org/abs/1611.07612) by Daniel Lemire, Nathan Kurz
  and Wojciech Muła (23 Nov 2016).
* Efficient Computation of Positional Population Counts Using SIMD Instructions,
  by Marcus D. R. Klarqvist, Wojciech Muła, and Daniel Lemire (upcoming)
* [Consistently faster and smaller compressed bitmaps with Roaring](https://arxiv.org/abs/1603.06549) by D. Lemire, G. Ssi-Yan-Kai,
  and O. Kaser (21 Mar 2016).

## Performance

All performance tests were run on a host machine with a 10 nm Cannon Lake Core
i3-8121U with gcc (GCC) 8.2.1 20180905 (Red Hat 8.2.1-3). Detailed benchmarking requires the
Linux `perf` subsystem. In all the examples below we do not output the result matrix but instead report its total sum. This was done to restrict our measurements to algorithmic performance while being unaffacted by disk I/O.

### Small input matrix

Sample performance metrics (practical upper limit) for **dense** matrices using a host machine with AVX512BW available. We
simulate many data arrays in aligned memory and compute the upper triangular or XX<sup>T</sup>
using the command `benchmark 65536 10000` 

| Set bits | CPU cycles / 64-bit word | MB/s      |
|----------|--------------------------|-----------|
| 32768    | 0.209                    | 109915 |
| 16384    | 0.21                     | 113591 |
| 6553     | 0.21                     | 114524 |
| 2621     | 0.21                     | 114256 |
| 1310     | 0.21                     | 114625 |
| 655      | 0.21                     | 114709 |
| 262      | 0.21                     | 114659 |
| 65       | 0.21                     | 114390 |
| 13       | 0.21                     | 114726 |
| 5        | 0.21                     | 114574 |
| 1        | 0.21                     | 114457 |

### Large input matrix

Next we simulated 10,000 vectors with [0,262144] random bits set for each to form a (10000,524288)-dimension matrix with different data densities. The following (misleading) figures
represent CPU cycles / 64-bit word equivalent if the analysis was run in uncompressed **bitmap space**. We compare our simple approach to [Roaring bitmaps](https://github.com/RoaringBitmap/CRoaring) demonstrating we can achieve performance parity on large bitmaps:

| Set bits | Storm-CPU-cycles | Roaring-CPU-cycles |
|----------|------------------|--------------------|
| 262144   | 0.483            | 0.567              |
| 131072   | 0.477            | 0.567              |
| 52428    | 0.474            | 0.567              |
| 20971    | 3.431            | 3.342              |
| 10485    | 1.765            | 1.756              |
| 5242     | 0.935            | 0.945              |
| 2097     | 0.43             | 0.451              |
| 524      | 0.19             | 0.196              |
| 104      | 0.117            | 0.133              |
| 5        | 0.017            | 0.029              |
| 1        | 0.003            | 0.004              |

### Compilation

Compile test suite with: `cmake .; make` and run `./benchmark`.

### Note

This is a collaborative effort between Marcus D. R. Klarqvist
([@klarqvist](https://github.com/mklarqvist/)) and Daniel Lemire
([@lemire](https://github.com/lemire/)).

### History

These functions were originally developed for
[Tomahawk](https://github.com/mklarqvist/Tomahawk) for computing genome-wide
linkage-disequilibrium but can be applied to any intersect-count problem.