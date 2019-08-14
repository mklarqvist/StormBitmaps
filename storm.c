#include "storm.h"

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
        memset(bitmap->data, 0, sizeof(uint32_t)*bitmap->n_bitmap);
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