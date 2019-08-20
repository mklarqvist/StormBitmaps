[![Build Status](https://travis-ci.com/mklarqvist/FastIntersectCount.svg)](https://travis-ci.com/mklarqvist/FastIntersectCount)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/mklarqvist/FastIntersectCount?branch=master&svg=true)](https://ci.appveyor.com/project/mklarqvist/FastIntersectCount)
[![Github Releases](https://img.shields.io/github/release/mklarqvist/FastIntersectCount.svg)](https://github.com/mklarqvist/FastIntersectCount/releases)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

# Storm bitmaps

These functions compute the set intersection count (|A \in B|) of pairs of
_symmetric_ integer sets with equal upper bounds [0,N). Several of the
functions presented here exploit set sparsity by using auxiliary information
such as positional indices, bitmaps, or reduction preprocessors. Using large
registers (AVX-512BW), contiguous and aligned memory, and cache-aware blocking,
we can achieve ~114 GB/s (~0.2 CPU cycles / 64-bit word) of sustained throughput
(~14 billion 64-bit bitmaps / second or up to ~912 billion implicit integers /
second) using `STORM_contiguous_t` when the input data is small (N < 256000).
When input data is large, we can achieve around 0.4-0.6 CPU cycles / 64-bit word
using `STORM_t` while using considerably less memory. Both of these models make
use of scalar-bitmap or scalar-scalar comparisons when the data density is
small.

Storm bitmaps have several interesting properties:
* Superior performance when the universe is small (M < 256000).
* Parity in perormance to Roaring bitmaps when the unvierse is large.
* Selects the optimal memory alignment and subroutines given the available SIMD
  instruction at run-time by using
  [libalgebra](https://github.com/mklarqvist/libalgebra).

The core algorithms are described in the papers:

* [Faster Population Counts using AVX2 Instructions](https://arxiv.org/abs/1611.07612) by Daniel Lemire, Nathan Kurz
  and Wojciech Muła (23 Nov 2016).
* Efficient Computation of Positional Population Counts Using SIMD Instructions,
  by Marcus D. R. Klarqvist, Wojciech Muła, and Daniel Lemire (upcoming)
* [Consistently faster and smaller compressed bitmaps with Roaring](https://arxiv.org/abs/1603.06549) by D. Lemire, G. Ssi-Yan-Kai,
  and O. Kaser (21 Mar 2016).

### Compilation

Compile test suite with: `cmake .; make` and run `./benchmark`. For better
performance, pass optimization flags to CMAKE: 
`cmake -DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_C_FLAGS="-march=native" .`

### Note

This is a collaborative effort between Marcus D. R. Klarqvist
([@klarqvist](https://github.com/mklarqvist/)) and Daniel Lemire
([@lemire](https://github.com/lemire/)).

### History

These functions were originally developed for
[Tomahawk](https://github.com/mklarqvist/Tomahawk) for computing genome-wide
linkage-disequilibrium but can be applied to any intersect-count problem.