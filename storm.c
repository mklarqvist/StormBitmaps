#include "storm.h"
#include <stdlib.h> // EXIT_SUCCESS, EXIT_FAILURE

uint64_t STORM_intersect_vector16_cardinality(const uint16_t* STORM_RESTRICT v1, 
                                              const uint16_t* STORM_RESTRICT v2, 
                                              const uint32_t len1, 
                                              const uint32_t len2) 
{
    size_t count = 0;
    size_t i_a = 0, i_b = 0;
    const int vectorlength = sizeof(__m128i) / sizeof(uint16_t);
    const size_t st_a = (len1 / vectorlength) * vectorlength;
    const size_t st_b = (len2 / vectorlength) * vectorlength;
    __m128i v_a, v_b;
    if ((i_a < st_a) && (i_b < st_b)) {
        v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
        v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);
        while ((v1[i_a] == 0) || (v2[i_b] == 0)) {
            const __m128i res_v = _mm_cmpestrm(
                v_b, vectorlength, v_a, vectorlength,
                _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
            const int r = _mm_extract_epi32(res_v, 0);
            count += _mm_popcnt_u32(r);
            const uint16_t a_max = v1[i_a + vectorlength - 1];
            const uint16_t b_max = v2[i_b + vectorlength - 1];
            if (a_max <= b_max) {
                i_a += vectorlength;
                if (i_a == st_a) break;
                v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
            }
            if (b_max <= a_max) {
                i_b += vectorlength;
                if (i_b == st_b) break;
                v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);
            }
        }
        if ((i_a < st_a) && (i_b < st_b))
            while (1) {
                const __m128i res_v = _mm_cmpistrm(
                    v_b, v_a,
                    _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
                const int r = _mm_extract_epi32(res_v, 0);
                count += _mm_popcnt_u32(r);
                const uint16_t a_max = v1[i_a + vectorlength - 1];
                const uint16_t b_max = v2[i_b + vectorlength - 1];
                if (a_max <= b_max) {
                    i_a += vectorlength;
                    if (i_a == st_a) break;
                    v_a = _mm_lddqu_si128((__m128i *)&v1[i_a]);
                }
                if (b_max <= a_max) {
                    i_b += vectorlength;
                    if (i_b == st_b) break;
                    v_b = _mm_lddqu_si128((__m128i *)&v2[i_b]);
                }
            }
    }
    // intersect the tail using scalar intersection
    while (i_a < len1 && i_b < len2) {
        uint16_t a = v1[i_a];
        uint16_t b = v2[i_b];
        if (a < b) {
            i_a++;
        } else if (b < a) {
            i_b++;
        } else {
            count++;
            i_a++;
            i_b++;
        }
    }
    return (uint64_t)count;
}

uint64_t STORM_intersect_vector32_unsafe(const uint32_t* STORM_RESTRICT v1, 
                                   const uint32_t* STORM_RESTRICT v2, 
                                   const uint32_t len1, 
                                   const uint32_t len2, 
                                   uint32_t* STORM_RESTRICT out)
{
    if (out == NULL) return 0;
    if (v1  == NULL) return 0;
    if (v2  == NULL) return 0;
    if (len1 == 0 || len2 == 0) return 0;
    uint64_t answer = 0;
    uint32_t A = 0, B = 0;

    uint32_t offset = 0;
    while (1) {
        while (v1[A] < v2[B]) {
            SKIP_FIRST_COMPARE:
            if (++A == len1) return answer;
        }
        while (v1[A] > v2[B]) {
            if (++B == len2) return answer;
        }
        if (v1[A] == v2[B]) {
            out[answer++] = A;
            out[answer++] = B;
            if (++A == len1 || ++B == len2) return answer;
        } else {
            goto SKIP_FIRST_COMPARE;
        }
    }
    return answer; // NOTREACHED
}

uint64_t STORM_intersect_bitmaps_scalar_list(const uint64_t* STORM_RESTRICT b1, 
    const uint64_t* STORM_RESTRICT b2, 
    const uint32_t* l1, const uint32_t* l2,
    const uint32_t  n1, const uint32_t  n2)
{
    uint64_t count = 0;

#define MOD(x) (( (x) * 64 ) >> 6)
    if(n1 < n2) {
        for(int i = 0; i < n1; ++i) {
            count += ((b2[l1[i] >> 6] & (1L << MOD(l1[i]))) != 0);
            // __builtin_prefetch(&b2[l1[i] >> 6], 0, _MM_HINT_T0);
        }
    } else {
        for(int i = 0; i < n2; ++i) {
            count += ((b1[l2[i] >> 6] & (1L << MOD(l2[i]))) != 0);
            // __builtin_prefetch(&b1[l2[i] >> 6], 0, _MM_HINT_T0);
        }
    }
#undef MOD
    return(count);
}
//

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
//
 
uint32_t STORM_bitmap_serialized_size(STORM_bitmap_t* bitmap) {
    uint32_t total = 0;
    total += sizeof(uint64_t) * bitmap->n_bitmap;
    if (bitmap->n_scalar_set) {
        total += sizeof(uint16_t) * bitmap->n_scalar;
    }
    total += 4*sizeof(uint32_t);
    return total;
}

 
uint32_t STORM_bitmap_cont_serialized_size(STORM_bitmap_cont_t* bitmap) {
    uint32_t total = 0;
    if (bitmap->bitmaps != NULL) {
        for (int i = 0; i < bitmap->n_bitmaps; ++i) {
            STORM_bitmap_t* x = (STORM_bitmap_t*)&bitmap->bitmaps[i];
            total += STORM_bitmap_serialized_size(x);
        }
    }
    total += sizeof(uint32_t) * bitmap->n_bitmaps;
    total += 3*sizeof(uint32_t);
    return total;
}

//

STORM_bitmap_t* STORM_bitmap_new() {
    STORM_bitmap_t* all = (STORM_bitmap_t*)malloc(sizeof(STORM_bitmap_t));
    if (all == NULL) return NULL;
    uint32_t alignment = STORM_get_alignment();
    all->data = NULL;
    all->scalar = NULL;
    all->n_bitmap = 0;
    all->own_data = 1;
    all->own_scalar = 1;
    all->n_scalar = 0;
    all->n_scalar_set = 0;
    all->n_missing = 0;
    all->m_scalar = 0;
    all->id = 0;
    all->n_bits_set = 0;
    return all;
}


void STORM_bitmap_init(STORM_bitmap_t* all) {
    uint32_t alignment = STORM_get_alignment();
    uint32_t n_bitmap = ceil(STORM_DEFAULT_BLOCK_SIZE / 64.0);
    all->data = NULL;
    all->scalar = NULL;
    all->n_bitmap = 0;
    all->own_data = 1;
    all->own_scalar = 1;
    all->n_scalar = 0;
    all->n_scalar_set = 0;
    all->n_missing = 0;
    all->m_scalar = 0;
    all->id = 0;
    all->n_bits_set = 0;    
}


void STORM_bitmap_free(STORM_bitmap_t* bitmap) {
    if (bitmap == NULL) return;
    if (bitmap->own_data) STORM_aligned_free(bitmap->data);
    if (bitmap->own_scalar) STORM_aligned_free(bitmap->scalar);
    free(bitmap);
}

 
int STORM_bitmap_add(STORM_bitmap_t* bitmap, const uint32_t* values, const uint32_t n_values) {
    if (bitmap == NULL) return -1;
    if (values == NULL) return -2;
    if (n_values == 0)  return -3;

    uint32_t adjust = bitmap->id * STORM_DEFAULT_BLOCK_SIZE;

    if (bitmap->data == NULL) {
        uint32_t n_bitmap = ceil(STORM_DEFAULT_BLOCK_SIZE / 64.0);
        uint32_t alignment = STORM_get_alignment();
        bitmap->data = (uint64_t*)STORM_aligned_malloc(alignment, n_bitmap*sizeof(uint64_t));
        memset(bitmap->data, 0, n_bitmap*sizeof(uint64_t));
    }
    bitmap->n_bitmap = ceil(STORM_DEFAULT_BLOCK_SIZE / 64.0);

    for (int i = 0; i < n_values; ++i) {
        assert(adjust <= values[i]);
        uint32_t v = values[i] - adjust;
        assert(v < STORM_DEFAULT_BLOCK_SIZE);
        bitmap->n_bits_set += (bitmap->data[v / 64] & 1ULL << (v % 64)) == 0;
        bitmap->data[v / 64] |= 1ULL << (v % 64);
    }
    return n_values;
}

 
int STORM_bitmap_add_with_scalar(STORM_bitmap_t* bitmap, const uint32_t* values, const uint32_t n_values) {
    if (bitmap == NULL) return -1;
    if (values == NULL) return -3;
    if (n_values == 0) return -4;

    if (bitmap->data == NULL) {
        uint32_t n_bitmap = ceil(STORM_DEFAULT_BLOCK_SIZE / 64.0);
        uint32_t alignment = STORM_get_alignment();
        bitmap->data = (uint64_t*)STORM_aligned_malloc(alignment, n_bitmap*sizeof(uint64_t));
        memset(bitmap->data, 0, n_bitmap*sizeof(uint64_t));
    }
    
    if (bitmap->scalar == NULL) {
        uint32_t new_m = 256 < n_values ? n_values + 256 : 256;
        uint32_t alignment = STORM_get_alignment();
        bitmap->scalar = (uint16_t*)STORM_aligned_malloc(alignment, new_m*sizeof(uint16_t));
        bitmap->m_scalar = new_m;
        bitmap->own_scalar = 1;
    }

    if (bitmap->n_scalar >= bitmap->m_scalar) {
        uint32_t new_m = bitmap->n_scalar + n_values + 1024;
        bitmap->m_scalar = new_m;
        uint16_t* old = bitmap->scalar;
        uint32_t alignment = STORM_get_alignment();
        bitmap->scalar = (uint16_t*)STORM_aligned_malloc(alignment, new_m*sizeof(uint16_t));
        memcpy(bitmap->scalar, old, bitmap->n_scalar*sizeof(uint16_t));
        STORM_aligned_free(old);
        bitmap->own_scalar = 1;
    }
    bitmap->n_bitmap = ceil(STORM_DEFAULT_BLOCK_SIZE / 64.0);

    uint32_t adjust = bitmap->id * STORM_DEFAULT_BLOCK_SIZE;
    bitmap->n_scalar_set = 1;

    for (int i = 0; i < n_values; ++i) {
        assert(adjust <= values[i]);
        uint32_t v = values[i] - adjust;
        assert(v < STORM_DEFAULT_BLOCK_SIZE);
        
        int is_unique = (bitmap->data[v / 64] & 1ULL << (v % 64)) == 0;
        bitmap->data[v / 64] |= 1ULL << (v % 64);
    
        if (is_unique) {
            bitmap->scalar[bitmap->n_scalar] = v;
            ++bitmap->n_scalar;
            ++bitmap->n_bits_set;
        }
    }
    return n_values;
}


int STORM_bitmap_add_scalar_only(STORM_bitmap_t* bitmap, const uint32_t* values, const uint32_t n_values) {
    if (bitmap == NULL) return -1;
    if (values == NULL) return -3;
    if (n_values == 0) return -4;

    if (bitmap->scalar == NULL) {
        uint32_t new_m = 256 < n_values ? n_values + 256 : 256;
        uint32_t alignment = STORM_get_alignment();
        bitmap->scalar = (uint16_t*)STORM_aligned_malloc(alignment, new_m*sizeof(uint16_t));
        bitmap->m_scalar = new_m;
        bitmap->own_scalar = 1;
    }

    if (bitmap->n_scalar >= bitmap->m_scalar) {
        uint32_t new_m = bitmap->n_scalar + n_values + 1024;
        bitmap->m_scalar = new_m;
        uint16_t* old = bitmap->scalar;
        uint32_t alignment = STORM_get_alignment();
        bitmap->scalar = (uint16_t*)STORM_aligned_malloc(alignment, new_m*sizeof(uint16_t));
        memcpy(bitmap->scalar, old, bitmap->n_scalar*sizeof(uint16_t));
        STORM_aligned_free(old);
        bitmap->own_scalar = 1;
    }

    uint32_t adjust = bitmap->id * STORM_DEFAULT_BLOCK_SIZE;
    bitmap->n_scalar_set = 1;

    for (int i = 0; i < n_values; ++i) {
        assert(adjust <= values[i]);
        uint32_t v = values[i] - adjust;
        assert(v < STORM_DEFAULT_BLOCK_SIZE);

        bitmap->scalar[bitmap->n_scalar] = v;
        ++bitmap->n_scalar;
        ++bitmap->n_bits_set;
    }
    return n_values;
}

 
int STORM_bitmap_clear(STORM_bitmap_t* bitmap) {
    if (bitmap == NULL) return -1;
    if (bitmap->data != NULL)
        memset(bitmap->data, 0, sizeof(uint64_t)*bitmap->n_bitmap);
    bitmap->n_scalar = 0;
    bitmap->n_bits_set = 0;
    bitmap->n_bitmap = 0;
    return 1;
}


uint64_t STORM_bitmap_intersect_cardinality(STORM_bitmap_t* STORM_RESTRICT bitmap1, 
                                          STORM_bitmap_t* STORM_RESTRICT bitmap2)
{
    if (bitmap1 == NULL) return 0;
    if (bitmap2 == NULL) return 0;

    // printf("blocks %u and %u\n",bitmap1->id,bitmap2->id);

    if (bitmap1->id != bitmap2->id) 
        return 0;

    if (bitmap1->n_bitmap == 0 && bitmap2->n_bitmap == 0) {
        // scalar-scalar comparison
       return STORM_intersect_vector16_cardinality(bitmap1->scalar, 
                                                 bitmap2->scalar, 
                                                 bitmap1->n_scalar, 
                                                 bitmap2->n_scalar);
        
    } else if (bitmap1->n_bitmap && bitmap2->n_bitmap == 0) {
        // bitmap-scalar comparison
        uint64_t count = 0;
        for (uint32_t i = 0; i < bitmap2->n_scalar; ++i) {
            count += bitmap1->data[bitmap2->scalar[i] / 64] & (1ULL << (bitmap2->scalar[i] % 64)) != 0;
        }
        return count;

    } else if (bitmap1->n_bitmap == 0 && bitmap2->n_bitmap) {
        // scalar-bitmap comparison
        uint64_t count = 0;
        for (uint32_t i = 0; i < bitmap1->n_scalar; ++i) {
            count += bitmap2->data[bitmap1->scalar[i] / 64] & (1ULL << (bitmap1->scalar[i] % 64)) != 0;
        }
        return count;

    } else if (bitmap1->n_bitmap && bitmap2->n_bitmap) {
        // bitmap-bitmap comparison
        const uint32_t n_bitmaps = ceil(STORM_DEFAULT_BLOCK_SIZE / 64.0);
        const STORM_compute_func f = STORM_get_intersect_count_func(n_bitmaps);
        return (*f)(bitmap1->data, bitmap2->data, n_bitmaps);
    } else {
        exit(EXIT_FAILURE);
    }

    return 0;
}

uint64_t STORM_bitmap_intersect_cardinality_func(STORM_bitmap_t* STORM_RESTRICT bitmap1, 
                                               STORM_bitmap_t* STORM_RESTRICT bitmap2, 
                                               const STORM_compute_func func)
{
    if (bitmap1 == NULL) return 0;
    if (bitmap2 == NULL) return 0;

    if (bitmap1->id != bitmap2->id) 
        return 0;

    if (bitmap1->n_bitmap == 0 && bitmap2->n_bitmap == 0) {
        // scalar-scalar comparison
        return STORM_intersect_vector16_cardinality(bitmap1->scalar, bitmap2->scalar, bitmap1->n_scalar, bitmap2->n_scalar);
        
    } else if (bitmap1->n_bitmap && bitmap2->n_bitmap == 0) {
        // bitmap-scalar comparison
        uint64_t count = 0;
        for (uint32_t i = 0; i < bitmap2->n_scalar; ++i) {
            count += bitmap1->data[bitmap2->scalar[i] / 64] & (1ULL << (bitmap2->scalar[i] % 64)) != 0;
        }
        return count;

    } else if (bitmap1->n_bitmap == 0 && bitmap2->n_bitmap) {
        // scalar-bitmap comparison
        uint64_t count = 0;
        for (uint32_t i = 0; i < bitmap1->n_scalar; ++i) {
            count += bitmap2->data[bitmap1->scalar[i] / 64] & (1ULL << (bitmap1->scalar[i] % 64)) != 0;
        }
        return count;

    } else if (bitmap1->n_bitmap && bitmap2->n_bitmap) {
        // bitmap-bitmap comparison
        return (*func)(bitmap1->data, bitmap2->data, bitmap1->n_bitmap);
    } else {
        exit(EXIT_FAILURE);
    }

    return 0;
}

// container
STORM_bitmap_cont_t* STORM_bitmap_cont_new() {
    STORM_bitmap_cont_t* all = (STORM_bitmap_cont_t*)malloc(sizeof(STORM_bitmap_cont_t));
    if (all == NULL) return NULL;
    all->bitmaps   = NULL;
    all->block_ids = NULL;
    all->n_bitmaps = 0;
    all->m_bitmaps = 0;
    all->prev_inserted_value = 0;
    return all;
}

void STORM_bitmap_cont_init(STORM_bitmap_cont_t* bitmap) {
    if (bitmap == NULL) return;
    bitmap->bitmaps   = NULL;
    bitmap->block_ids = NULL;
    bitmap->n_bitmaps = 0;
    bitmap->m_bitmaps = 0;
    bitmap->prev_inserted_value = 0;
}

void STORM_bitmap_cont_free(STORM_bitmap_cont_t* bitmap) {
    if (bitmap == NULL) return;
    if (bitmap->bitmaps != NULL) {
        // for (uint32_t i = 0; i < bitmap->n_bitmaps; ++i) {
        //     STORM_bitmap_free(&bitmap->bitmaps[i]);
        // }
        free(bitmap->bitmaps);
    }
    free(bitmap->block_ids);
    free(bitmap);
}


int STORM_bitmap_cont_add(STORM_bitmap_cont_t* bitmap, const uint32_t* values, const uint32_t n_values) {
    if (bitmap == NULL) return -1;
    if (values == NULL) return -2;
    if (n_values == 0)  return 0; 

    if (bitmap->bitmaps == NULL) {
        bitmap->m_bitmaps = 2;
        bitmap->bitmaps = (STORM_bitmap_t*)malloc(sizeof(STORM_bitmap_t) * bitmap->m_bitmaps);
        for (int i = 0; i < bitmap->m_bitmaps; ++i) {
            STORM_bitmap_init(&bitmap->bitmaps[i]);
        }
    }

    if (bitmap->block_ids == NULL) {
        bitmap->block_ids = (uint32_t*)malloc(sizeof(uint32_t) * bitmap->m_bitmaps);
    }

    // Input data must be guaranteed to be in sorted order.
    uint32_t start = 0, stop = 0;
    uint32_t target_block = values[0] / STORM_DEFAULT_BLOCK_SIZE;

    while (stop < n_values) {
        // printf("1: %u,%u/%u with %u,%u\n",start,stop,n_values,values[start],values[stop]);
        for (/**/; stop < n_values; ++stop) {
             if ((values[stop] / STORM_DEFAULT_BLOCK_SIZE) != target_block) {
                //  printf("start,stop: %u->%u/%u. block %u->%u\n",start,stop,n_values,target_block,values[stop] / STORM_DEFAULT_BLOCK_SIZE);
                 break;
             }
        }
        // printf("2: %u,%u/%u with %u,%u\n",start,stop,n_values,values[start],values[stop]);
        // const uint32_t debug = target_block;
        const uint32_t new_block = values[stop] / STORM_DEFAULT_BLOCK_SIZE;

        // Resize if required.
        // printf("bitmaps: %u/%u\n",bitmap->n_bitmaps,bitmap->m_bitmaps);
        if (bitmap->n_bitmaps == bitmap->m_bitmaps) {
            // printf("Resizing container %u->%u\n", bitmap->m_bitmaps, bitmap->m_bitmaps+8);
            uint32_t old_m = bitmap->m_bitmaps;
            bitmap->m_bitmaps += 8;
            bitmap->bitmaps = (STORM_bitmap_t*)realloc(bitmap->bitmaps, sizeof(STORM_bitmap_t) * bitmap->m_bitmaps);
            for (int i = old_m; i < bitmap->m_bitmaps; ++i) {
                STORM_bitmap_init(&bitmap->bitmaps[i]);
            }
            bitmap->block_ids = (uint32_t*)realloc(bitmap->block_ids, sizeof(uint32_t) * bitmap->m_bitmaps);
        }
        // printf("Adding new bitmap for value %u to %u\n", values[i], target_block);

        assert(stop != start);
        assert(stop - start > 0);
        STORM_bitmap_t* x = (STORM_bitmap_t*)&bitmap->bitmaps[bitmap->n_bitmaps];
        x->id = target_block;
        bitmap->block_ids[bitmap->n_bitmaps] = target_block;

        if (stop - start < STORM_DEFAULT_SCALAR_THRESHOLD) {
            STORM_bitmap_add_scalar_only(x, &values[start], stop - start);
        } else {
            STORM_bitmap_add(x, &values[start], stop - start);
        }
        
        ++bitmap->n_bitmaps;
        bitmap->prev_inserted_value = values[stop-1];
        target_block = new_block;
        start = stop;
    }    

    return 1;
}

uint64_t STORM_bitmap_cont_intersect_cardinality(const STORM_bitmap_cont_t* STORM_RESTRICT bitmap1, 
                                               const STORM_bitmap_cont_t* STORM_RESTRICT bitmap2)
{
    if (bitmap1 == NULL) return 0;
    if (bitmap2 == NULL) return 0;
    if (bitmap1->n_bitmaps == 0) return 0;
    if (bitmap2->n_bitmaps == 0) return 0;

    // Move this out to recycle memory.
    uint32_t* out = (uint32_t*)malloc(8192*sizeof(uint32_t));

    uint32_t ret = STORM_intersect_vector32_unsafe(bitmap1->block_ids, 
                                                 bitmap2->block_ids, 
                                                 bitmap1->n_bitmaps, 
                                                 bitmap2->n_bitmaps, 
                                                 out);
    // Retrieve optimal intersect function.
    const STORM_compute_func f = STORM_get_intersect_count_func(bitmap1->n_bitmaps);

    uint64_t count = 0;
    for (uint32_t i = 0; i < ret; i += 2) {
        assert(bitmap1->bitmaps[out[i+0]].id == bitmap2->bitmaps[out[i+1]].id);
        count += STORM_bitmap_intersect_cardinality_func(&bitmap1->bitmaps[out[i+0]], &bitmap2->bitmaps[out[i+1]], f);
    }

    free(out);

    return count;
}

uint64_t STORM_bitmap_cont_intersect_cardinality_premade(const STORM_bitmap_cont_t* STORM_RESTRICT bitmap1, 
                                                       const STORM_bitmap_cont_t* STORM_RESTRICT bitmap2, 
                                                       const STORM_compute_func func, 
                                                       uint32_t* out)
{
    if (bitmap1 == NULL) return 0;
    if (bitmap2 == NULL) return 0;
    if (bitmap1->n_bitmaps == 0) return 0;
    if (bitmap2->n_bitmaps == 0) return 0;
    if (out == NULL) return 0;

    uint32_t ret = STORM_intersect_vector32_unsafe(bitmap1->block_ids, 
                                                 bitmap2->block_ids, 
                                                 bitmap1->n_bitmaps, 
                                                 bitmap2->n_bitmaps, 
                                                 out);

    uint64_t count = 0;
    for (uint32_t i = 0; i < ret; i += 2) {
        assert(bitmap1->bitmaps[out[i+0]].id == bitmap2->bitmaps[out[i+1]].id);
        count += STORM_bitmap_intersect_cardinality_func(&bitmap1->bitmaps[out[i+0]], &bitmap2->bitmaps[out[i+1]], func);
    }
    
    return count;
}
 
int STORM_bitmap_cont_clear(STORM_bitmap_cont_t* bitmap) {
    if (bitmap == NULL) return -1;
    for (uint32_t i = 0; i < bitmap->n_bitmaps; ++i) {
        STORM_bitmap_clear(&bitmap->bitmaps[i]);
    }
    bitmap->n_bitmaps = 0;
    bitmap->prev_inserted_value = 0;
    return 1;
}

// cont
STORM_t* STORM_new() {
    STORM_t* all = (STORM_t*)malloc(sizeof(STORM_t));
    if (all == NULL) return NULL;
    all->conts = NULL;
    all->n_conts = 0;
    all->m_conts = 0;
    return all;
}

void STORM_free(STORM_t* bitmap) {
    if (bitmap == NULL) return;
    // for (uint32_t i = 0; i < bitmap->m_conts; ++i) {
    //     STORM_bitmap_cont_free(&bitmap->conts[i]);
    // }
    free(bitmap->conts);
}

int STORM_add(STORM_t* bitmap, const uint32_t* values, const uint32_t n_values) {
    if (bitmap == NULL) return -1;
    
    if (bitmap->m_conts == 0) {
        bitmap->m_conts = 1024;
        bitmap->conts = (STORM_bitmap_cont_t*)malloc(bitmap->m_conts*sizeof(STORM_bitmap_cont_t));
        bitmap->n_conts = 0;
        for (uint32_t i = 0; i < bitmap->m_conts; ++i) {
            STORM_bitmap_cont_init(&bitmap->conts[i]);
        }
    }

    if (bitmap->n_conts == bitmap->m_conts) {
        bitmap->m_conts += 1024;
        bitmap->conts = (STORM_bitmap_cont_t*)realloc(bitmap->conts, bitmap->m_conts*sizeof(STORM_bitmap_cont_t));
        for (uint32_t i = bitmap->n_conts; i < bitmap->m_conts; ++i) {
            STORM_bitmap_cont_init(&bitmap->conts[i]);
        }
    }

    STORM_bitmap_cont_add(&bitmap->conts[bitmap->n_conts++], values, n_values);
    return 1;
}

int STORM_clear(STORM_t* bitmap) {
    if (bitmap == NULL) return -1;
    
    for (int i = 0; i < bitmap->n_conts; ++i)
        STORM_bitmap_cont_clear(&bitmap->conts[i]);
    bitmap->n_conts = 0;
    return 1;
}

uint64_t STORM_pairw_intersect_cardinality(STORM_t* bitmap) {
    if (bitmap == NULL) return -1;

    uint32_t* out = (uint32_t*)malloc(sizeof(uint32_t)*2*STORM_DEFAULT_SCALAR_THRESHOLD);
    const STORM_compute_func f = STORM_get_intersect_count_func(ceil(STORM_DEFAULT_BLOCK_SIZE/64.0));

    // printf("running for: %u vectors\n", bitmap->n_conts);

    uint64_t total = 0;
    for (uint32_t i = 0; i < bitmap->n_conts; ++i) {
        for (uint32_t j = i + 1; j < bitmap->n_conts; ++j) {
            total += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[i], &bitmap->conts[j], f, out);
        }
    }

    free(out);

    return total;
}

uint64_t STORM_pairw_intersect_cardinality_blocked(STORM_t* bitmap, uint32_t bsize) {
    if (bitmap == NULL) return -1;

    uint32_t* out = (uint32_t*)malloc(sizeof(uint32_t)*2*STORM_DEFAULT_SCALAR_THRESHOLD);
    const STORM_compute_func f = STORM_get_intersect_count_func(ceil(STORM_DEFAULT_BLOCK_SIZE/64.0));
    
    if (bsize == 0) {
        uint64_t tot = 0;
        for (uint32_t i = 0; i < bitmap->n_conts; ++i) {
            tot += STORM_bitmap_cont_serialized_size(&bitmap->conts[i]);
        }
        uint32_t average_size = tot / bitmap->n_conts;
        bsize = ceil((double)STORM_CACHE_BLOCK_SIZE / average_size);
        // printf("guestimating block-size to %u\n", bsize);
    }
    
    // Make sure block size is not <5.
    bsize = bsize < 5 ? 5 : bsize;

    // printf("running for: %u vectors\n", bitmap->n_conts);

    uint64_t count = 0;
    uint32_t i = 0;

    for (/**/; i + bsize <= bitmap->n_conts; i += bsize) {
        // diagonal component
        for (uint32_t j = 0; j < bsize; ++j) {
            for (uint32_t jj = j + 1; jj < bsize; ++jj) {
                // count += (*func)(bitmaps[i+j].data, bitmaps[i+jj].data, n_bitmaps_sample);
                count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[i+j], &bitmap->conts[i+jj], f, out);
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + bsize;
        for (/**/; j + bsize <= bitmap->n_conts; j += bsize) {
            for (uint32_t ii = 0; ii < bsize; ++ii) {
                for (uint32_t jj = 0; jj < bsize; ++jj) {
                    // count += (*func)(bitmaps[curi+ii].data, bitmaps[j+jj].data, n_bitmaps_sample);
                    count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[curi+ii], &bitmap->conts[j+jj], f, out);
                }
            }
        }

        // residual
        for (/**/; j < bitmap->n_conts; ++j) {
            for (uint32_t jj = 0; jj < bsize; ++jj) {
                // count += (*func)(bitmaps[curi+jj].data, bitmaps[j].data, n_bitmaps_sample);
                count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[curi+jj], &bitmap->conts[j], f, out);
            }
        }
    }
    // residual tail
    for (/**/; i < bitmap->n_conts; ++i) {
        for (uint32_t j = i + 1; j < bitmap->n_conts; ++j) {
            // count += (*func)(bitmaps[i].data, bitmaps[j].data, n_bitmaps_sample);
            count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[i], &bitmap->conts[j], f, out);
        }
    }

    free(out);

    return count;
}

uint64_t STORM_serialized_size(const STORM_t* bitmap) {
    if (bitmap == NULL) return 0;

    uint64_t tot = 0;
    for (uint32_t i = 0; i < bitmap->n_conts; ++i) {
        tot += STORM_bitmap_cont_serialized_size(&bitmap->conts[i]);
    }
    tot += 2*sizeof(uint32_t);

    return tot;
}

uint64_t STORM_intersect_cardinality_square(const STORM_t* STORM_RESTRICT bitmap1, const STORM_t* STORM_RESTRICT bitmap2);

// contig

// Contiguous memory bitmaps
// struct STORM_contiguous_bitmap_s {
//     uint64_t* data; // not owner of this data
//     uint32_t* scalar; // not owner of this data
//     // width of data is described outside
//     uint32_t n_scalar; // copy from outside
// };

// struct STORM_contiguous_s {
//     uint64_t* data; // bitmaps
//     uint32_t* scalar; // scalar values
//     uint32_t* n_scalar; // scalar values per bitmap
//     STORM_contiguous_bitmap_t* bitmaps; // interpret of data
//     uint64_t n_data, m_data; // m_data is reported per _VECTOR_ not per machine word
//     uint64_t vector_length;
//     uint32_t n_bitmaps_vector; // _MUST_ be divisible by largest alignment!
//     STORM_compute_func intsec_func; // determined during ctor
//     uint32_t alignment; // determined during ctor
//     uint32_t scalar_cutoff; // cutoff for storing scalars
// };


STORM_contiguous_t* STORM_contig_new(size_t vector_length) {
    STORM_contiguous_t* all = (STORM_contiguous_t*)malloc(sizeof(STORM_contiguous_t));
    if (all == NULL) return NULL;
    all->data    = NULL;
    all->scalar  = NULL;
    all->n_scalar= NULL;
    all->bitmaps = NULL;
    all->n_data  = 0;
    all->m_data  = 0;
    all->tot_scalar = 0;
    all->m_scalar   = 0;
    all->vector_length = vector_length;
    all->n_bitmaps_vector = ceil(vector_length / 64.0);
    all->alignment     = STORM_get_alignment();
    all->intsec_func   = STORM_get_intersect_count_func(all->n_bitmaps_vector);
    all->scalar_cutoff = vector_length / 200 > 200 ? 200 : vector_length / 200;
    return all;
}

void STORM_contig_free(STORM_contiguous_t* bitmap) {
    if (bitmap == NULL) return;
    // for (uint32_t i = 0; i < bitmap->m_conts; ++i) {
    //     STORM_bitmap_cont_free(&bitmap->conts[i]);
    // }
    STORM_aligned_free(bitmap->data);
    free(bitmap->bitmaps);
    STORM_aligned_free(bitmap->scalar);
    STORM_aligned_free(bitmap->n_scalar);
}

int STORM_contig_add(STORM_contiguous_t* bitmap, const uint32_t* values, const uint32_t n_values) {
    if (bitmap == NULL) return -1;
    if (values == NULL) return -2;
    if (n_values == 0)  return 0;

    // If scalar is not set then allocate some memory
    if (bitmap->scalar == NULL) {
        bitmap->m_scalar   = 512*32;
        bitmap->tot_scalar = 0;
        bitmap->scalar     = (uint32_t*)STORM_aligned_malloc(bitmap->alignment, bitmap->m_scalar*sizeof(uint32_t));
    }

    // If data is not set then allocate some memory
    // Coupled with n_scalar through m_data
    if (bitmap->data == NULL) {
        bitmap->m_data   = 512;
        bitmap->data     = (uint64_t*)STORM_aligned_malloc(bitmap->alignment, bitmap->n_bitmaps_vector*bitmap->m_data*sizeof(uint64_t));
        bitmap->n_scalar = (uint32_t*)STORM_aligned_malloc(bitmap->alignment, bitmap->m_data*sizeof(uint32_t));
        memset(bitmap->data, 0, bitmap->n_bitmaps_vector*bitmap->m_data*sizeof(uint64_t));
        bitmap->bitmaps = (STORM_contiguous_bitmap_t*)malloc(bitmap->m_data*sizeof(STORM_contiguous_bitmap_t));
        for (int i = 0; i < bitmap->m_data; ++i) {
            bitmap->bitmaps[i].data     = &bitmap->data[bitmap->n_bitmaps_vector*i];
            bitmap->bitmaps[i].scalar   = NULL;
            bitmap->bitmaps[i].n_scalar = 0;
        }
    }

    // If number of added values plus current values exceeds the allocated
    // number then allocate more memory.
    if (bitmap->tot_scalar + n_values >= bitmap->m_scalar) {
        uint32_t add = 5*n_values < 65535 ? 65535 : 5*n_values;
        // printf("resizing scalar: %u/%u->%u\n",bitmap->tot_scalar,bitmap->m_scalar,bitmap->m_scalar+add);
        bitmap->m_scalar += add;
        uint32_t* old = bitmap->scalar;
        bitmap->scalar = (uint32_t*)STORM_aligned_malloc(bitmap->alignment, bitmap->m_scalar*sizeof(uint32_t));
        memcpy(bitmap->scalar, old, bitmap->tot_scalar);
        STORM_aligned_free(old);
        // Update pointers.
        for (int i = 0, j = 0; i < bitmap->n_data; ++i) {
            bitmap->bitmaps[i].scalar = &bitmap->scalar[j];
            j += bitmap->n_scalar[j] < bitmap->scalar_cutoff ? bitmap->n_scalar[j] : 0;
        }
    }

    // If data needs resizing we will:
    //   resize data and n_scalar
    //   update pointer references in bitmapsto bitmaps->data, and bitmaps->n_scalar, and bitmaps->scalar
    if (bitmap->n_data >= bitmap->m_data) {
        // printf("realloc %u->%u\n",bitmap->m_data,bitmap->m_data+512);
        uint64_t* old = bitmap->data;
        uint32_t* old_n_scalar = bitmap->n_scalar;
        bitmap->m_data += 512;
        bitmap->data     = (uint64_t*)STORM_aligned_malloc(bitmap->alignment, bitmap->n_bitmaps_vector*bitmap->m_data*sizeof(uint64_t));
        bitmap->n_scalar = (uint32_t*)STORM_aligned_malloc(bitmap->alignment, bitmap->m_data*sizeof(uint32_t));
        // memset(bitmap->data, 0, bitmap->n_bitmaps_vector*bitmap->m_data*sizeof(uint64_t));
        memcpy(bitmap->n_scalar, old_n_scalar, bitmap->n_data*sizeof(uint32_t));
        memcpy(bitmap->data, old, bitmap->n_bitmaps_vector*bitmap->n_data*sizeof(uint64_t));
        memset(&bitmap->data[bitmap->n_bitmaps_vector*bitmap->n_data], 0, 
                (bitmap->n_bitmaps_vector*bitmap->m_data*sizeof(uint64_t)) - (bitmap->n_bitmaps_vector*bitmap->n_data*sizeof(uint64_t)));
        STORM_aligned_free(old);
        STORM_aligned_free(old_n_scalar);
        bitmap->bitmaps = (STORM_contiguous_bitmap_t*)realloc(bitmap->bitmaps, bitmap->m_data*sizeof(STORM_contiguous_bitmap_t));
        for (int i = 0, j = 0; i < bitmap->m_data; ++i) {
            bitmap->bitmaps[i].data     = &bitmap->data[bitmap->n_bitmaps_vector*i];
            bitmap->bitmaps[i].n_scalar = bitmap->n_scalar[i];
            bitmap->bitmaps[i].scalar   = &bitmap->scalar[j];
            // printf("loop %u/%u and %u/%u. scalar=%u\n", i, bitmap->m_data, j, bitmap->m_scalar, bitmap->n_scalar[i]);
            j += bitmap->n_scalar[j] < bitmap->scalar_cutoff ? bitmap->n_scalar[j] : 0;
        }
    }

    // printf("adding start with %u/%u bitmaps/vector=%u\n",bitmap->n_data,bitmap->m_data,bitmap->n_bitmaps_vector);
    uint32_t n_values_used = n_values;
    for (int i = 0; i < n_values; ++i) {
        if (i != 0) {
            if (values[i] == values[i-1]) {
                --n_values_used;
                continue;
            } else {
                // printf("%u !> %u\n", values[i], values[i-1]);
                assert(values[i] > values[i-1]);
            }
        }
        bitmap->bitmaps[bitmap->n_data].data[values[i] / 64] |= 1ULL << (values[i] % 64);
    }

    // Add scalar values if the total number of values does not exceed
    // the threshold scalar_cutoff.
    if (n_values_used < bitmap->scalar_cutoff) {
        // printf("adding scalar: %u @ %u\n",n_values,bitmap->tot_scalar);
        bitmap->bitmaps[bitmap->n_data].scalar = &bitmap->scalar[bitmap->tot_scalar];
        bitmap->bitmaps[bitmap->n_data].scalar[0] = values[0];
        
        for (int i = 1; i < n_values; ++i) {
            if (values[i] == values[i-1]) continue;
            bitmap->bitmaps[bitmap->n_data].scalar[i] = values[i];
        }
        bitmap->tot_scalar += n_values_used;
    }

    // Store number of set bits (n_values)
    bitmap->n_scalar[bitmap->n_data] = n_values_used;
    bitmap->bitmaps[bitmap->n_data].n_scalar = n_values_used;
    ++bitmap->n_data; // Advance data pointer

    return n_values;
}

int STORM_contig_clear(STORM_contiguous_t* bitmap) {
    if (bitmap == NULL) return -1;
    if (bitmap->data == NULL) return 0;
    memset(bitmap->data, 0, bitmap->n_bitmaps_vector*bitmap->m_data*sizeof(uint64_t));
    bitmap->n_data = 0;
    bitmap->tot_scalar = 0;
    
    return 1;
}

uint64_t STORM_contig_pairw_intersect_cardinality(STORM_contiguous_t* bitmap) {
    if (bitmap == NULL) return -1;
    if (bitmap->scalar != NULL) {
        // check list
        uint32_t valid = 0;
        for (int i = 0; i < bitmap->n_data; ++i) {
            valid += bitmap->bitmaps[i].n_scalar < bitmap->scalar_cutoff;
        }
        // printf("checking for scalar: %u<%u\n", valid, bitmap->scalar_cutoff);
        if (valid) {
            // printf("using scalar\n");
            return STORM_contig_pairw_intersect_cardinality_list(bitmap);
        }
    }

    uint64_t total = 0;
    for (uint32_t i = 0; i < bitmap->n_data; ++i) {
        for (uint32_t j = i + 1; j < bitmap->n_data; ++j) {
            total += (*bitmap->intsec_func)(bitmap->bitmaps[i].data, bitmap->bitmaps[j].data, bitmap->n_bitmaps_vector);
            // total += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->bitmaps[i].data, &bitmap->bitmaps[j].data, bitmap->intsec_func, out);
        }
    }

    return total;
}

uint64_t STORM_contig_pairw_intersect_cardinality_blocked(STORM_contiguous_t* bitmap, uint32_t bsize) {
    if (bitmap == NULL) return -1;

    if (bitmap->scalar != NULL) {
        // check list
        uint32_t valid = 0;
        for (int i = 0; i < bitmap->n_data; ++i) {
            valid += bitmap->bitmaps[i].n_scalar < bitmap->scalar_cutoff;
        }
        // printf("checking for blocked-scalar: %u<%u\n", valid, bitmap->scalar_cutoff);
        if (valid) {
            // printf("using scalar\n");
            return STORM_contig_pairw_intersect_cardinality_blocked_list(bitmap, bsize);
        }
    }

    if (bsize <= 2)
        return STORM_contig_pairw_intersect_cardinality(bitmap);

    // printf("running for: %u vectors\n", bitmap->n_conts);

    uint64_t count = 0;
    uint32_t i = 0;

    for (/**/; i + bsize <= bitmap->n_data; i += bsize) {
        // diagonal component
        for (uint32_t j = 0; j < bsize; ++j) {
            for (uint32_t jj = j + 1; jj < bsize; ++jj) {
                // count += (*func)(bitmaps[i+j].data, bitmaps[i+jj].data, n_bitmaps_sample);
                // count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[i+j], &bitmap->conts[i+jj], f, out);
                count += (*bitmap->intsec_func)(bitmap->bitmaps[i+j].data, bitmap->bitmaps[i+jj].data, bitmap->n_bitmaps_vector);
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + bsize;
        for (/**/; j + bsize <= bitmap->n_data; j += bsize) {
            for (uint32_t ii = 0; ii < bsize; ++ii) {
                for (uint32_t jj = 0; jj < bsize; ++jj) {
                    // count += (*func)(bitmaps[curi+ii].data, bitmaps[j+jj].data, n_bitmaps_sample);
                    // count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[curi+ii], &bitmap->conts[j+jj], f, out);
                    count += (*bitmap->intsec_func)(bitmap->bitmaps[curi+ii].data, bitmap->bitmaps[j+jj].data, bitmap->n_bitmaps_vector);
                }
            }
        }

        // residual
        for (/**/; j < bitmap->n_data; ++j) {
            for (uint32_t jj = 0; jj < bsize; ++jj) {
                // count += (*func)(bitmaps[curi+jj].data, bitmaps[j].data, n_bitmaps_sample);
                // count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[curi+jj], &bitmap->conts[j], f, out);
                count += (*bitmap->intsec_func)(bitmap->bitmaps[curi+jj].data, bitmap->bitmaps[j].data, bitmap->n_bitmaps_vector);
            }
        }
    }
    // residual tail
    for (/**/; i < bitmap->n_data; ++i) {
        for (uint32_t j = i + 1; j < bitmap->n_data; ++j) {
            // count += (*func)(bitmaps[i].data, bitmaps[j].data, n_bitmaps_sample);
            // count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[i], &bitmap->conts[j], f, out);
            count += (*bitmap->intsec_func)(bitmap->bitmaps[i].data, bitmap->bitmaps[j].data, bitmap->n_bitmaps_vector);
        }
    }

    return count;
}

uint64_t STORM_contig_pairw_intersect_cardinality_list(STORM_contiguous_t* bitmap) {
    if (bitmap == NULL) return -1;
    if (bitmap->scalar == NULL) return -2;
    if (bitmap->n_scalar == NULL) return -3;

    // printf("using list\n");

    uint64_t total = 0;
    for (uint32_t i = 0; i < bitmap->n_data; ++i) {
        for (uint32_t j = i + 1; j < bitmap->n_data; ++j) {
            if (bitmap->bitmaps[i].n_scalar < bitmap->scalar_cutoff || bitmap->bitmaps[j].n_scalar < bitmap->scalar_cutoff) {
                total += STORM_intersect_bitmaps_scalar_list(bitmap->bitmaps[i].data, bitmap->bitmaps[j].data, bitmap->bitmaps[i].scalar, bitmap->bitmaps[j].scalar, bitmap->bitmaps[i].n_scalar, bitmap->bitmaps[j].n_scalar);
            } else {
                total += (*bitmap->intsec_func)(bitmap->bitmaps[i].data, bitmap->bitmaps[j].data, bitmap->n_bitmaps_vector);
                // total += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->bitmaps[i].data, &bitmap->bitmaps[j].data, bitmap->intsec_func, out);
            }
        }
    }

    return total;
}

uint64_t STORM_contig_pairw_intersect_cardinality_blocked_list(STORM_contiguous_t* bitmap, uint32_t bsize) {
    if (bitmap == NULL) return -1;
    if (bitmap->scalar == NULL) return -2;
    if (bitmap->n_scalar == NULL) return -3;

    if (bsize <= 2)
        return STORM_contig_pairw_intersect_cardinality_list(bitmap);

    // printf("using blocked-list\n");

    // printf("running for: %u vectors\n", bitmap->n_conts);

    uint64_t count = 0;
    uint32_t i = 0;

    for (/**/; i + bsize <= bitmap->n_data; i += bsize) {
        // diagonal component
        for (uint32_t j = 0; j < bsize; ++j) {
            for (uint32_t jj = j + 1; jj < bsize; ++jj) {
                if (bitmap->bitmaps[i+j].n_scalar < bitmap->scalar_cutoff || bitmap->bitmaps[i+jj].n_scalar < bitmap->scalar_cutoff) {
                    count += STORM_intersect_bitmaps_scalar_list(bitmap->bitmaps[i+j].data, bitmap->bitmaps[i+jj].data, 
                        bitmap->bitmaps[i+j].scalar, bitmap->bitmaps[i+jj].scalar, 
                        bitmap->bitmaps[i+j].n_scalar, bitmap->bitmaps[i+jj].n_scalar);
                } else {
                    count += (*bitmap->intsec_func)(bitmap->bitmaps[i+j].data, bitmap->bitmaps[i+jj].data, bitmap->n_bitmaps_vector);
                }
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + bsize;
        for (/**/; j + bsize <= bitmap->n_data; j += bsize) {
            for (uint32_t ii = 0; ii < bsize; ++ii) {
                for (uint32_t jj = 0; jj < bsize; ++jj) {
                    if (bitmap->bitmaps[curi+ii].n_scalar < bitmap->scalar_cutoff || bitmap->bitmaps[j+jj].n_scalar < bitmap->scalar_cutoff) {
                        count += STORM_intersect_bitmaps_scalar_list(bitmap->bitmaps[curi+ii].data, bitmap->bitmaps[j+jj].data, 
                            bitmap->bitmaps[curi+ii].scalar, bitmap->bitmaps[j+jj].scalar, 
                            bitmap->bitmaps[curi+ii].n_scalar, bitmap->bitmaps[j+jj].n_scalar);
                    } else {
                        count += (*bitmap->intsec_func)(bitmap->bitmaps[curi+ii].data, bitmap->bitmaps[j+jj].data, bitmap->n_bitmaps_vector);
                    }
                    // count += (*func)(bitmaps[curi+ii].data, bitmaps[j+jj].data, n_bitmaps_sample);
                    // count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[curi+ii], &bitmap->conts[j+jj], f, out);
                    // count += (*bitmap->intsec_func)(bitmap->bitmaps[curi+ii].data, bitmap->bitmaps[j+jj].data, bitmap->n_bitmaps_vector);
                }
            }
        }

        // residual
        for (/**/; j < bitmap->n_data; ++j) {
            for (uint32_t jj = 0; jj < bsize; ++jj) {
                if (bitmap->bitmaps[curi+jj].n_scalar < bitmap->scalar_cutoff || bitmap->bitmaps[j].n_scalar < bitmap->scalar_cutoff) {
                    count += STORM_intersect_bitmaps_scalar_list(bitmap->bitmaps[curi+jj].data, bitmap->bitmaps[j].data, 
                        bitmap->bitmaps[curi+jj].scalar, bitmap->bitmaps[j].scalar, 
                        bitmap->bitmaps[curi+jj].n_scalar, bitmap->bitmaps[j].n_scalar);
                } else {
                    count += (*bitmap->intsec_func)(bitmap->bitmaps[curi+jj].data, bitmap->bitmaps[j].data, bitmap->n_bitmaps_vector);
                }
                // count += (*func)(bitmaps[curi+jj].data, bitmaps[j].data, n_bitmaps_sample);
                // count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[curi+jj], &bitmap->conts[j], f, out);
                // count += (*bitmap->intsec_func)(bitmap->bitmaps[curi+jj].data, bitmap->bitmaps[j].data, bitmap->n_bitmaps_vector);
            }
        }
    }
    // residual tail
    for (/**/; i < bitmap->n_data; ++i) {
        for (uint32_t j = i + 1; j < bitmap->n_data; ++j) {
            if (bitmap->bitmaps[i].n_scalar < bitmap->scalar_cutoff || bitmap->bitmaps[j].n_scalar < bitmap->scalar_cutoff) {
                count += STORM_intersect_bitmaps_scalar_list(bitmap->bitmaps[i].data, bitmap->bitmaps[j].data, 
                    bitmap->bitmaps[i].scalar, bitmap->bitmaps[j].scalar, 
                    bitmap->bitmaps[i].n_scalar, bitmap->bitmaps[j].n_scalar);
            } else {
                count += (*bitmap->intsec_func)(bitmap->bitmaps[i].data, bitmap->bitmaps[j].data, bitmap->n_bitmaps_vector);
            }
            // count += (*func)(bitmaps[i].data, bitmaps[j].data, n_bitmaps_sample);
            // count += STORM_bitmap_cont_intersect_cardinality_premade(&bitmap->conts[i], &bitmap->conts[j], f, out);
            // count += (*bitmap->intsec_func)(bitmap->bitmaps[i].data, bitmap->bitmaps[j].data, bitmap->n_bitmaps_vector);
        }
    }

    return count;
}