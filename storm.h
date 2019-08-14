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
#include "fast_intersect_count.h"

#ifndef TWK_DEFAULT_BLOCK_SIZE
#define TWK_DEFAULT_BLOCK_SIZE 65536
#endif

#ifndef TWK_DEFAULT_SCALAR_THRESHOLD
#define TWK_DEFAULT_SCALAR_THRESHOLD 4096
#endif

#ifdef __cplusplus
extern "C" {
#endif

//
uint64_t TWK_intersect_vector16_cardinality(const uint16_t* TWK_RESTRICT v1, const uint16_t* TWK_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t TWK_intersect_vector32_unsafe(const uint32_t* TWK_RESTRICT v1, const uint32_t* TWK_RESTRICT v2, const uint32_t len1, const uint32_t len2, uint32_t* TWK_RESTRICT out);
//

typedef struct TWK_bitmap_s TWK_bitmap_t;
typedef struct TWK_bitmap_cont_s TWK_bitmap_cont_t;
typedef struct TWK_cont_s TWK_cont_t;
typedef struct TWK_contiguous_bitmap_s TWK_contiguous_bitmap_t;
typedef struct TWK_contiguous_s TWK_contiguous_t;

// Storm bitmaps
struct TWK_bitmap_s {
    TWK_ALIGN(64) uint64_t* data; // data array
    TWK_ALIGN(64) uint16_t* scalar; // scalar array
    uint32_t n_bitmap: 30, own_data: 1, own_scalar: 1;
    uint32_t n_bits_set;
    uint32_t n_scalar: 31, n_scalar_set: 1, n_missing;
    uint32_t m_scalar;
    uint32_t id; // block id
};

struct TWK_bitmap_cont_s {
    TWK_bitmap_t* bitmaps; // bitmaps array
    uint32_t* block_ids; // block ids (redundant but better data locality)
    uint32_t n_bitmaps, m_bitmaps;
    uint32_t prev_inserted_value;
};

struct TWK_cont_s {
    TWK_bitmap_cont_t* conts;
    uint32_t n_conts, m_conts;
};

// Contiguous memory bitmaps
struct TWK_contiguous_bitmap_s {
    uint64_t* data; // not owner of this data
    // width of data is described outside
};

struct TWK_contiguous_s {
    TWK_ALIGN(64) uint64_t* data;
    TWK_contiguous_bitmap_t* bitmaps; // interpret of data
    uint64_t n_data, m_data;
    uint64_t n_samples;
    uint32_t n_bitmaps_vector; // _MUST_ be divisible by largest alignment!
};

// implementation ----->
TWK_bitmap_t* TWK_bitmap_new();
void TWK_bitmap_init(TWK_bitmap_t* all);
void TWK_bitmap_free(TWK_bitmap_t* bitmap);
int TWK_bitmap_add(TWK_bitmap_t* bitmap, const uint32_t* values, const uint32_t n_values);
int TWK_bitmap_add_with_scalar(TWK_bitmap_t* bitmap, const uint32_t* values, const uint32_t n_values);
int TWK_bitmap_add_scalar_only(TWK_bitmap_t* bitmap, const uint32_t* values, const uint32_t n_values);
uint64_t TWK_bitmap_intersect_cardinality(TWK_bitmap_t* TWK_RESTRICT bitmap1, TWK_bitmap_t* TWK_RESTRICT bitmap2);
uint64_t TWK_bitmap_intersect_cardinality_func(TWK_bitmap_t* TWK_RESTRICT bitmap1, TWK_bitmap_t* TWK_RESTRICT bitmap2, const TWK_intersect_func func);
int TWK_bitmap_clear(TWK_bitmap_t* bitmap);
uint32_t TWK_bitmap_serialized_size(TWK_bitmap_t* bitmap);

// bit container
TWK_bitmap_cont_t* TWK_bitmap_cont_new();
void TWK_bitmap_cont_init(TWK_bitmap_cont_t* bitmap);
void TWK_bitmap_cont_free(TWK_bitmap_cont_t* bitmap);
int TWK_bitmap_cont_add(TWK_bitmap_cont_t* bitmap, const uint32_t* values, const uint32_t n_values);
int TWK_bitmap_cont_clear(TWK_bitmap_cont_t* bitmap);
uint64_t TWK_bitmap_cont_intersect_cardinality(const TWK_bitmap_cont_t* TWK_RESTRICT bitmap1, const TWK_bitmap_cont_t* TWK_RESTRICT bitmap2);
uint64_t TWK_bitmap_cont_intersect_cardinality_premade(const TWK_bitmap_cont_t* TWK_RESTRICT bitmap1, const TWK_bitmap_cont_t* TWK_RESTRICT bitmap2, const TWK_intersect_func func, uint32_t* out);
uint32_t TWK_bitmap_cont_serialized_size(TWK_bitmap_cont_t* bitmap);

// container
TWK_cont_t* TWK_cont_new();
void TWK_cont_free(TWK_cont_t* bitmap);
int TWK_cont_add(TWK_cont_t* bitmap, const uint32_t* values, const uint32_t n_values);
int TWK_cont_clear(TWK_cont_t* bitmap);
uint64_t TWK_cont_pairw_intersect_cardinality(TWK_cont_t* bitmap);
uint64_t TWK_cont_pairw_intersect_cardinality_blocked(TWK_cont_t* bitmap, uint32_t bsize);
uint64_t TWK_cont_intersect_cardinality_square(const TWK_cont_t* TWK_RESTRICT bitmap1, const TWK_cont_t* TWK_RESTRICT bitmap2);
uint64_t TWK_cont_serialized_size(const TWK_cont_t* bitmap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* STORM_H_7263478623842093 */