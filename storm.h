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
#ifndef STORM_H_7263478623842093
#define STORM_H_7263478623842093

/* *************************************
*  Includes
***************************************/
#include <stdint.h>
#include <assert.h>
#include <memory.h>
#include <string.h>
#include <math.h>

/* *************************************
*  Dependencies
***************************************/
#include "libalgebra/libalgebra.h"

// Default size of a memory block. This is by default set to 256kb which is what
// most commodity processors have as L2/L3 cache.
#ifndef STORM_CACHE_BLOCK_SIZE
#define STORM_CACHE_BLOCK_SIZE 256e3
#endif

#ifndef STORM_DEFAULT_BLOCK_SIZE
#define STORM_DEFAULT_BLOCK_SIZE 65536
#endif

#ifndef STORM_DEFAULT_SCALAR_THRESHOLD
#define STORM_DEFAULT_SCALAR_THRESHOLD 4096
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*======   Generic intersection functions   ======*/
// Todo: this require SSE4.2
// Credit: Lemire et. al (Roaring bitmaps)
uint64_t STORM_intersect_vector16_cardinality(const uint16_t* STORM_RESTRICT v1, const uint16_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t STORM_intersect_vector32_unsafe(const uint32_t* STORM_RESTRICT v1, const uint32_t* STORM_RESTRICT v2, const uint32_t len1, const uint32_t len2, uint32_t* STORM_RESTRICT out);
uint64_t STORM_intersect_bitmaps_scalar_list(const uint64_t* STORM_RESTRICT b1, 
    const uint64_t* STORM_RESTRICT b2, 
    const uint32_t* l1, const uint32_t* l2,
    const uint32_t  n1, const uint32_t  n2);


/*======   Wrappers   ======*/
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
uint64_t STORM_wrapper_diag(const uint32_t n_vectors, 
                            const uint64_t* vals, 
                            const uint32_t n_ints, 
                            const STORM_compute_func f);

uint64_t STORM_wrapper_square(const uint32_t n_vectors1,
                              const uint64_t* STORM_RESTRICT vals1, 
                              const uint32_t n_vectors2,
                              const uint64_t* STORM_RESTRICT vals2, 
                              const uint32_t n_ints, 
                              const STORM_compute_func f);

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
uint64_t STORM_wrapper_diag_list(const uint32_t n_vectors, 
                                 const uint64_t* STORM_RESTRICT vals,
                                 const uint32_t n_ints,
                                 const uint32_t* STORM_RESTRICT n_alts,
                                 const uint32_t* STORM_RESTRICT alt_positions,
                                 const uint32_t* STORM_RESTRICT alt_offsets, 
                                 const STORM_compute_func f, 
                                 const STORM_compute_lfunc fl, 
                                 const uint32_t cutoff);

uint64_t STORM_wrapper_diag_blocked(const uint32_t n_vectors, 
                                    const uint64_t* vals, 
                                    const uint32_t n_ints, 
                                    const STORM_compute_func f,
                                    uint32_t block_size);

uint64_t STORM_wrapper_diag_list_blocked(const uint32_t n_vectors, 
                             const uint64_t* STORM_RESTRICT vals,
                             const uint32_t n_ints,
                             const uint32_t* STORM_RESTRICT n_alts,
                             const uint32_t* STORM_RESTRICT alt_positions,
                             const uint32_t* STORM_RESTRICT alt_offsets, 
                             const STORM_compute_func f, 
                             const STORM_compute_lfunc fl, 
                             const uint32_t cutoff,
                             uint32_t block_size);

/*======   Canonical representation   ======*/
typedef struct STORM_bitmap_s STORM_bitmap_t;
typedef struct STORM_bitmap_cont_s STORM_bitmap_cont_t;
typedef struct STORM_s STORM_t;
typedef struct STORM_contiguous_bitmap_s STORM_contiguous_bitmap_t;
typedef struct STORM_contiguous_s STORM_contiguous_t;

// Storm bitmaps
struct STORM_bitmap_s {
    STORM_ALIGN(64) uint64_t* data; // data array
    STORM_ALIGN(64) uint16_t* scalar; // scalar array
    uint32_t n_bitmap: 30, own_data: 1, own_scalar: 1;
    uint32_t n_bits_set;
    uint32_t n_scalar: 31, n_scalar_set: 1, n_missing;
    uint32_t m_scalar;
    uint32_t id; // block id
};

struct STORM_bitmap_cont_s {
    STORM_bitmap_t* bitmaps; // bitmaps array
    uint32_t* block_ids; // block ids (redundant but better data locality)
    uint32_t n_bitmaps, m_bitmaps;
    uint32_t prev_inserted_value;
};

struct STORM_s {
    STORM_bitmap_cont_t* conts;
    uint32_t n_conts, m_conts;
};

// Contiguous memory bitmaps
struct STORM_contiguous_bitmap_s {
    uint64_t* data; // not owner of this data
    uint32_t* scalar; // not owner of this data
    // width of data is described outside
    uint32_t n_scalar; // copy from outside
};

struct STORM_contiguous_s {
    uint64_t* data; // bitmaps (aligned)
    uint32_t* scalar; // scalar values (aligned)
    uint32_t* n_scalar; // scalar values per bitmap (aligned)
    STORM_contiguous_bitmap_t* bitmaps; // interpret of data
    uint64_t n_data, m_data; // m_data is reported per _VECTOR_ not per machine word
    uint64_t tot_scalar, m_scalar;
    uint64_t vector_length;
    uint32_t n_bitmaps_vector; // _MUST_ be divisible by largest alignment!
    STORM_compute_func intsec_func; // determined during ctor
    uint32_t alignment; // determined during ctor
    uint32_t scalar_cutoff; // cutoff for storing scalars
};

// implementation ----->
STORM_bitmap_t* STORM_bitmap_new();
void STORM_bitmap_init(STORM_bitmap_t* all);
void STORM_bitmap_free(STORM_bitmap_t* bitmap);
int STORM_bitmap_add(STORM_bitmap_t* bitmap, const uint32_t* values, const uint32_t n_values);
int STORM_bitmap_add_with_scalar(STORM_bitmap_t* bitmap, const uint32_t* values, const uint32_t n_values);
int STORM_bitmap_add_scalar_only(STORM_bitmap_t* bitmap, const uint32_t* values, const uint32_t n_values);
uint64_t STORM_bitmap_intersect_cardinality(STORM_bitmap_t* STORM_RESTRICT bitmap1, STORM_bitmap_t* STORM_RESTRICT bitmap2);
uint64_t STORM_bitmap_intersect_cardinality_func(STORM_bitmap_t* STORM_RESTRICT bitmap1, STORM_bitmap_t* STORM_RESTRICT bitmap2, const STORM_compute_func func);
int STORM_bitmap_clear(STORM_bitmap_t* bitmap);
uint32_t STORM_bitmap_serialized_size(STORM_bitmap_t* bitmap);

// bit container
STORM_bitmap_cont_t* STORM_bitmap_cont_new();
void STORM_bitmap_cont_init(STORM_bitmap_cont_t* bitmap);
void STORM_bitmap_cont_free(STORM_bitmap_cont_t* bitmap);
int STORM_bitmap_cont_add(STORM_bitmap_cont_t* bitmap, const uint32_t* values, const uint32_t n_values);
int STORM_bitmap_cont_clear(STORM_bitmap_cont_t* bitmap);
uint64_t STORM_bitmap_cont_intersect_cardinality(const STORM_bitmap_cont_t* STORM_RESTRICT bitmap1, const STORM_bitmap_cont_t* STORM_RESTRICT bitmap2);
uint64_t STORM_bitmap_cont_intersect_cardinality_premade(const STORM_bitmap_cont_t* STORM_RESTRICT bitmap1, const STORM_bitmap_cont_t* STORM_RESTRICT bitmap2, const STORM_compute_func func, uint32_t* out);
uint32_t STORM_bitmap_cont_serialized_size(STORM_bitmap_cont_t* bitmap);

// container
STORM_t* STORM_new();
void STORM_free(STORM_t* bitmap);
int STORM_add(STORM_t* bitmap, const uint32_t* values, const uint32_t n_values);
int STORM_clear(STORM_t* bitmap);
uint64_t STORM_pairw_intersect_cardinality(STORM_t* bitmap);
uint64_t STORM_pairw_intersect_cardinality_blocked(STORM_t* bitmap, uint32_t bsize);
uint64_t STORM_intersect_cardinality_square(const STORM_t* STORM_RESTRICT bitmap1, const STORM_t* STORM_RESTRICT bitmap2);
uint64_t STORM_serialized_size(const STORM_t* bitmap);

// contig
STORM_contiguous_t* STORM_contig_new(size_t vector_length);
void STORM_contig_free(STORM_contiguous_t* bitmap);
int STORM_contig_add(STORM_contiguous_t* bitmap, const uint32_t* values, const uint32_t n_values);
int STORM_contig_clear(STORM_contiguous_t* bitmap);
uint64_t STORM_contig_pairw_intersect_cardinality(STORM_contiguous_t* bitmap);
uint64_t STORM_contig_pairw_intersect_cardinality_blocked(STORM_contiguous_t* bitmap, uint32_t bsize);
uint64_t STORM_contig_pairw_intersect_cardinality_list(STORM_contiguous_t* bitmap);
uint64_t STORM_contig_pairw_intersect_cardinality_blocked_list(STORM_contiguous_t* bitmap, uint32_t bsize);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* STORM_H_7263478623842093 */