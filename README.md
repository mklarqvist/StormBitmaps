# FastIntersectCount

These functions compute the set intersection count (|A \in B|) of pairs of integer sets with equal upper bounds [0,N). Several of the functions presented here exploit set sparsity by using auxiliary information such as positional indices, bitmaps, or reduction preprocessors. There are no union and difference routines in this repository. Using large registers (AVX-512BW) and cache-blocking, we can achieve ~75 GB/s (~0.3 CPU cycles / 64-bit word) throughput (80 billion 64-bit bitmaps / second).

The core algorithms are described in the papers:

* [Faster Population Counts using AVX2 Instructions](https://arxiv.org/abs/1611.07612) by Daniel Lemire, Nathan Kurz
  and Wojciech Muła (23 Nov 2016).
* Efficient Computation of Positional Population Counts Using SIMD Instructions,
  by Marcus D. R. Klarqvist, Wojciech Muła, and Daniel Lemire (upcoming)
* [Consistently faster and smaller compressed bitmaps with Roaring](https://arxiv.org/pdf/1603.06549.pdf) by D. Lemire, G. Ssi-Yan-Kai,
  and O. Kaser.

Compile test suite with: `make` and run `./fast_intersect_count`


### Note

This is a collaborative effort between Marcus D. R. Klarqvist ([@klarqvist](https://github.com/mklarqvist/)) and Daniel Lemire ([@lemire](https://github.com/lemire/)). M.D.R.K. acknowledge Chris Wallace ([@chr1swallace](https://github.com/chr1swallace)) for informal discussions that eventually resulted in this work.

### History

These functions were originally developed for [Tomahawk](https://github.com/mklarqvist/Tomahawk) for computing genome-wide linkage-disequilibrium but can be applied to any intersect-count problem.