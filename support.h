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
#ifndef FAST_INTERSECT_SUPPORT_H_
#define FAST_INTERSECT_SUPPORT_H_

#include "libalgebra/libalgebra.h"

/* *************************************
*  Alignment and retrieve intersection function
***************************************/
// Function pointer definitions.
typedef uint64_t (*STORM_compute_lfunc)(const uint64_t*, const uint64_t*, 
    const uint32_t*, const uint32_t*, const size_t, const size_t);

/* *************************************
*  Example wrappers
*
*  These wrappers compute sum(popcnt(A & B)) for all N input bitmaps
*  pairwise. All input bitmaps must be of the same length M. The
*  functions starting with STORM_wrapper_diag* assumes that all data
*  comes from the same contiguous memory buffer. Use STORM_wrapper_square*
*  if you have data from two distinct, but contiguous, memory buffers
*  B1 and B2.
*
*  The STORM_wrapper_*_list* functions make use of auxilliary information
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
uint64_t STORM_wrapper_diag(const uint32_t n_vectors, 
                            const uint64_t* vals, 
                            const uint32_t n_ints, 
                            const STORM_compute_func f)
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
uint64_t STORM_wrapper_square(const uint32_t n_vectors1,
                              const uint64_t* STORM_RESTRICT vals1, 
                              const uint32_t n_vectors2,
                              const uint64_t* STORM_RESTRICT vals2, 
                              const uint32_t n_ints, 
                              const STORM_compute_func f)
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
uint64_t STORM_wrapper_diag_list(const uint32_t n_vectors, 
                                 const uint64_t* STORM_RESTRICT vals,
                                 const uint32_t n_ints,
                                 const uint32_t* STORM_RESTRICT n_alts,
                                 const uint32_t* STORM_RESTRICT alt_positions,
                                 const uint32_t* STORM_RESTRICT alt_offsets, 
                                 const STORM_compute_func f, 
                                 const STORM_compute_lfunc fl, 
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
uint64_t STORM_wrapper_diag_blocked(const uint32_t n_vectors, 
                                    const uint64_t* vals, 
                                    const uint32_t n_ints, 
                                    const STORM_compute_func f,
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
uint64_t STORM_wrapper_diag_list_blocked(const uint32_t n_vectors, 
                             const uint64_t* STORM_RESTRICT vals,
                             const uint32_t n_ints,
                             const uint32_t* STORM_RESTRICT n_alts,
                             const uint32_t* STORM_RESTRICT alt_positions,
                             const uint32_t* STORM_RESTRICT alt_offsets, 
                             const STORM_compute_func f, 
                             const STORM_compute_lfunc fl, 
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

#endif