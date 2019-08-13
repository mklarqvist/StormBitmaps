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
#ifndef FAST_INTERSECT_COUNT_CLASSES_H_
#define FAST_INTERSECT_COUNT_CLASSES_H_

#include "fast_intersect_count.h"
// #include "experimental.h"

#include <vector>
#include <memory> //unique_ptr

#include <iostream>//debug
#include <bitset> //debug

// temp C
#ifndef TWK_DEFAULT_BLOCK_SIZE
#define TWK_DEFAULT_BLOCK_SIZE 65536
#endif

#ifndef TWK_DEFAULT_SCALAR_THRESHOLD
#define TWK_DEFAULT_SCALAR_THRESHOLD 4096
#endif

//
uint64_t intersect_vector16_cardinality(const uint16_t* TWK_RESTRICT v1, const uint16_t* TWK_RESTRICT v2, const uint32_t len1, const uint32_t len2);
uint64_t intersect_vector32_unsafe(const uint32_t* TWK_RESTRICT v1, const uint32_t* TWK_RESTRICT v2, const uint32_t len1, const uint32_t len2, uint32_t* TWK_RESTRICT out);
//

// Fixed width bitmaps
struct TWK_bitmap_t {
    TWK_ALIGN(64) uint64_t* data; // data array
    TWK_ALIGN(64) uint16_t* scalar; // scalar array
    uint32_t n_bitmap: 30, own_data: 1, own_scalar: 1;
    uint32_t n_bits_set;
    uint32_t n_scalar: 31, n_scalar_set: 1, n_missing;
    uint32_t m_scalar;
    uint32_t id; // block id
};

struct TWK_bitmap_cont_t {
    TWK_bitmap_t* bitmaps; // bitmaps array
    uint32_t* block_ids; // block ids (redundant but better data locality)
    uint32_t n_bitmaps, m_bitmaps;
    uint32_t prev_inserted_value;
};

struct TWK_cont_t {
    TWK_bitmap_cont_t* conts;
    uint32_t n_conts, m_conts;
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
uint32_t TWK_bitmap_cont_serialized_size(TWK_bitmap_cont_t* bitmap);

// bit container
TWK_bitmap_cont_t* TWK_bitmap_cont_new();
void TWK_bitmap_cont_free(TWK_bitmap_cont_t* bitmap);
int TWK_bitmap_cont_add(TWK_bitmap_cont_t* bitmap, const uint32_t* values, const uint32_t n_values);
int TWK_bitmap_cont_clear(TWK_bitmap_cont_t* bitmap);
uint64_t TWK_bitmap_cont_intersect_cardinality(const TWK_bitmap_cont_t* TWK_RESTRICT bitmap1, const TWK_bitmap_cont_t* TWK_RESTRICT bitmap2);
uint64_t TWK_bitmap_cont_intersect_cardinality_buffer(const TWK_bitmap_cont_t* TWK_RESTRICT bitmap1, const TWK_bitmap_cont_t* TWK_RESTRICT bitmap2, uint32_t* out);
// use TWK_bitmap_add if value > threshold
// otherwise use TWK_bitmap_add_with_alts

// container
TWK_cont_t* TWK_cont_new();
void TWK_cont_free(TWK_cont_t* bitmap);
int TWK_cont_add(TWK_cont_t* bitmap, const uint32_t* values, const uint32_t n_values);
int TWK_cont_clear(TWK_cont_t* bitmap);
uint64_t TWK_cont_intersect_cardinality(TWK_cont_t* bitmap);
uint64_t TWK_cont_intersect_cardinality_square(const TWK_cont_t* TWK_RESTRICT bitmap1, const TWK_cont_t* TWK_RESTRICT bitmap2);

//  c++ stuff
class TWK_bitmap_container {
public:
    TWK_bitmap_container(const uint32_t n, const uint32_t m) :
        n_samples(n), m_vectors(m),
        n_bitmaps_vector(ceil(n_samples / 64.0)),
        bitmaps(new TWK_bitmap_t[m_vectors]),
        store_alts(true),
        alt_limit(0)
    {
        if (n > 1024) {
            alt_limit = n / 200 < 5 ? 5 : n / 200;
            store_alts = true;
        } else store_alts = false;
        
        // for (int i = 0; i < m_vectors; ++i)
        //     bitmaps[i] = TWK_bitmap_new();
    }

    ~TWK_bitmap_container() {
        // for (int i = 0; i < m_vectors; ++i)
            // TWK_bitmap_free(bitmaps[i]);
        delete[] bitmaps;
    }

    void Add(const uint32_t pos, const uint32_t* vals, const uint32_t n_vals) {
        if (store_alts && n_vals < alt_limit) TWK_bitmap_add_with_scalar(&bitmaps[pos], vals, n_vals);
        else TWK_bitmap_add(&bitmaps[pos], vals, n_vals);
    }

    void clear() {
        for (int i = 0; i < m_vectors; ++i)
            TWK_bitmap_clear(&bitmaps[i]);
    }

public:
    uint32_t n_samples, m_vectors;
    uint32_t n_bitmaps_vector;
    bool store_alts;
    uint32_t alt_limit;
    uint32_t n_bitmaps, m_bitmaps;
    TWK_bitmap_t* bitmaps;
    uint32_t* bitmap_offsets; // virtual offset into bitmaps for variant m
};

//////////

struct bitmap_t {
    bitmap_t() : alignment(TWK_get_alignment()), n_set(0), n_bitmap(0), own(false), data(nullptr) {}
    bitmap_t(uint64_t* in, uint32_t n, uint32_t m) : alignment(TWK_get_alignment()), n_set(n), n_bitmap(m), own(false), data(in) {}
    ~bitmap_t() {
        if (own) TWK_aligned_free(data);
    }

    int Allocate(uint32_t n) {
        if (data == nullptr) {
            n_bitmap = n;
            data  = (uint64_t*)TWK_aligned_malloc(alignment, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        } else {
            if (own) TWK_aligned_free(data);
            n_bitmap = n;
            data  = (uint64_t*)TWK_aligned_malloc(alignment, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        }
        memset(data,0,n_bitmap*sizeof(uint64_t));
        return n;
    }

    int AllocateSamples(uint32_t n) {
        const uint64_t n_vals = ceil(n / 64.0);
        if (data == nullptr) {
            n_bitmap = n_vals;
            data  = (uint64_t*)TWK_aligned_malloc(alignment, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        } else {
            if (own) TWK_aligned_free(data);
            n_bitmap = n_vals;
            data  = (uint64_t*)TWK_aligned_malloc(alignment, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        }
        memset(data,0,n_bitmap*sizeof(uint64_t));
        return n;
    }

    inline void Add(const uint64_t pos) {
        n_set += (data[pos / 64] & 1ULL << (pos % 64)) == 0; // predicate add
        data[pos / 64] |= 1ULL << (pos % 64); 
    }

    void clear() {
        n_set = 0;
        memset(data, 0, n_bitmap*sizeof(uint64_t));
    }

    uint32_t alignment;
    uint32_t n_set, n_bitmap: 31, own: 1; // number of values set, number of bitmaps, ownership
    uint64_t* data;
    // uint32_t n_alts: 31, n_alts_set: 1, n_missing
    // uint32_t* alts
};

struct bitmap_container_t {
    bitmap_container_t(uint32_t n, uint32_t m) : 
        alignment(TWK_get_alignment()),
        n_alt_cutoff(0),
        n_bitmaps(n), 
        n_samples(m), 
        own(true), 
        type(0),
        n_bitmaps_sample(ceil(n_samples / 64.0)),
        bmaps(nullptr), 
        bitmaps(new bitmap_t[n]),
        n_alts_tot(0), m_alts(0),
        alt_positions(nullptr),
        alt_offsets(nullptr),
        n_alts(nullptr)
    {
        for (int i = 0; i < n_bitmaps; ++i) {
            bitmaps[i].AllocateSamples(n_samples);
        }
    }

    bitmap_container_t(uint32_t n, uint32_t m, bool yes) : 
        alignment(TWK_get_alignment()),
        n_alt_cutoff(0),
        n_bitmaps(n), 
        n_samples(m), 
        own(true), 
        type(1),
        n_bitmaps_sample(ceil(n_samples / 64.0)),
        bmaps((uint64_t*)TWK_aligned_malloc(alignment, n_bitmaps*n_bitmaps_sample*sizeof(uint64_t))), 
        bitmaps(nullptr),
        n_alts_tot(0), m_alts(0),
        alt_positions(nullptr),
        alt_offsets(nullptr),
        n_alts((uint32_t*)TWK_aligned_malloc(alignment, n_bitmaps*sizeof(uint32_t)))
    {
        memset(bmaps,0,n_bitmaps*n_bitmaps_sample*sizeof(uint64_t));
        memset(n_alts,0,n_bitmaps*sizeof(uint32_t));
    }

    bitmap_container_t(uint32_t n, uint32_t m, bool yes, bool yes2) : 
        alignment(TWK_get_alignment()),
        n_alt_cutoff(0),
        n_bitmaps(n), 
        n_samples(m), 
        own(true), 
        type(1),
        n_bitmaps_sample(ceil(n_samples / 64.0)),
        bmaps((uint64_t*)TWK_aligned_malloc(alignment, n_bitmaps*n_bitmaps_sample*sizeof(uint64_t))), 
        bitmaps(nullptr),
        alt_positions(nullptr),
        alt_offsets((uint32_t*)TWK_aligned_malloc(alignment, n_bitmaps*sizeof(uint32_t))),
        n_alts((uint32_t*)TWK_aligned_malloc(alignment, n_bitmaps*sizeof(uint32_t)))
    {
        n_alt_cutoff = ceil(n_samples / 200.0);
        // Cap cutoff to values over 1024.
        n_alt_cutoff = n_samples < 1024 ? 0 : n_alt_cutoff;
        memset(bmaps,0,n_bitmaps*n_bitmaps_sample*sizeof(uint64_t));
        memset(n_alts,0,n_bitmaps*sizeof(uint32_t));
    }

    ~bitmap_container_t() {
        if (own) {
            delete[] bitmaps;
            TWK_aligned_free(bmaps);
        }
        TWK_aligned_free(alt_offsets);
        TWK_aligned_free(n_alts);
        TWK_aligned_free(alt_positions);
    }

    void Add(const uint32_t target, uint32_t value) { 
        if (type == 0) {
            assert(bitmaps!=nullptr);
            bitmaps[target].Add(value); 
        }
        else {
            assert(bmaps!=nullptr);
            uint64_t* x = &bmaps[target*n_bitmaps_sample];
            x[value / 64] |= 1ULL << (value % 64);
            ++n_alts[target]; // increment the number of alts for this site
        }
    }

    void Add(const uint32_t target, const std::vector<uint32_t>& values) { 
        if (type == 0) {
            assert(bitmaps!=nullptr);
            for (int i = 0; i < values.size(); ++i)
                bitmaps[target].Add(values[i]); 
        }
        else { // if type is 1
            assert(bmaps != nullptr);

            uint64_t* x = &bmaps[target*n_bitmaps_sample];
            for (int i = 0; i < values.size(); ++i) {
                x[values[i] / 64] |= 1ULL << (values[i] % 64);
            }
            n_alts[target] = values.size(); // set the number of alts for this site

            alt_offsets[target] = n_alts_tot; // always set offset

            // todo: fix
            if (values.size() < n_alt_cutoff) {
                // resize if required
                if (n_alts_tot + values.size() >= m_alts || alt_positions == nullptr) {
                    uint32_t* old = alt_positions;
                    const uint32_t add = values.size() < 65535 ? 65535 : values.size() * 5;
                    uint32_t new_pos = (n_alts_tot == 0 ? add : n_alts_tot + add);
                    // std::cerr << "resizing: " << n_alts_tot << "->" << new_pos << std::endl;
                    alt_positions = (uint32_t*)TWK_aligned_malloc(alignment, new_pos*sizeof(uint32_t));
                    memcpy(alt_positions, old, n_alts_tot*sizeof(uint32_t));
                    m_alts = new_pos;
                    TWK_aligned_free(old);
                }

                // std::cerr << "adding: " << values.size() << " at " << alt_offsets[target] << std::endl;
    
                uint32_t* tgt = &alt_positions[alt_offsets[target]];
                for (int i = 0; i < values.size(); ++i) {
                    // std::cerr << values[i] << std::endl;
                    tgt[i] = values[i];
                }
                n_alts_tot += values.size();
            }
        }
    }
    
    void clear() {
        if (type == 0) {
        for (int i = 0; i < n_bitmaps; ++i)
            bitmaps[i].clear();
        } else {
            memset(bmaps,0,n_bitmaps*n_bitmaps_sample*sizeof(uint64_t));
            if (n_alts != nullptr) memset(n_alts,0,n_bitmaps*sizeof(uint32_t));
            if (alt_positions != nullptr) memset(alt_positions,0,m_alts*sizeof(uint32_t));
            if (alt_offsets != nullptr) memset(alt_offsets,0,n_bitmaps*sizeof(uint32_t));
            n_alts_tot = 0;
        }
    }

    uint64_t intersect() const;
    uint64_t intersect_cont() const;
    uint64_t intersect_blocked(uint32_t bsize) const;
    uint64_t intersect_blocked_cont(uint32_t bsize) const;
    uint64_t intersect_cont_auto() const;
    uint64_t intersect_cont_blocked_auto(uint32_t bsize) const;

    uint32_t alignment;
    uint32_t n_alt_cutoff;
    uint32_t n_bitmaps, n_samples: 30, own: 1, type: 1;
    uint32_t n_bitmaps_sample;
    uint64_t* bmaps;
    bitmap_t* bitmaps;
    //
    uint32_t n_alts_tot, m_alts;
    uint32_t* alt_positions; // positions of alts at a site
    uint32_t* alt_offsets; // virtual offsets to start of alt_positions for a site
    uint32_t* n_alts; // number of alts at a position
};

struct roaring2_t {
    roaring2_t(uint32_t n_s, uint32_t n_ss) : alignment(TWK_get_alignment()), n_samples(n_s), 
        n_sites(n_ss), block_size(128), n_total_blocks(0),
        n_total_bitmaps(0), m_blocks(0), m_bitmaps(0), n_blocks(nullptr), 
        blocks(nullptr), data_bitmaps(nullptr)
    {}

    ~roaring2_t() {
        TWK_aligned_free(n_blocks);
        TWK_aligned_free(blocks);
        TWK_aligned_free(data_bitmaps);
    }

    int Add(const uint32_t target, const std::vector<uint32_t>& pos) {
        if (pos.size() == 0) return 0;

        if (n_blocks == nullptr) {
            n_blocks = (uint32_t*)TWK_aligned_malloc(alignment, 2*n_sites*sizeof(uint32_t));
        }

        if (blocks == nullptr) {
            assert(n_blocks_site != 0);
            m_blocks = n_blocks_site * 25;
            blocks = (uint16_t*)TWK_aligned_malloc(alignment, m_blocks*sizeof(uint16_t));
            n_total_blocks = 0;
        }

        if (data_bitmaps == nullptr) {
            m_bitmaps = 65535;
            data_bitmaps = (uint64_t*)TWK_aligned_malloc(alignment, m_bitmaps*sizeof(uint64_t));
            memset(data_bitmaps, 0, m_bitmaps*sizeof(uint64_t));
            n_total_bitmaps = 0;
        }

        // First target block
        uint32_t current_block = pos[0] / 128; // block_size
        // Allocate
        uint64_t* tgt_bitmap = &data_bitmaps[n_total_bitmaps];
        n_total_bitmaps += 128; // 8192 bits = 16 * 512
        
        for (int i = 0; i < pos.size(); ++i) {
            uint32_t adj_pos = pos[i] - (current_block*8192);
            tgt_bitmap[adj_pos / 64] |= 1ULL << (adj_pos % 64);
        }

        return 1;
    }

    struct pair_t {
        uint32_t n_blocks;
        uint32_t offset;
    };

    // N containers (blocks) per site
    // each container share memory region
    uint32_t alignment;
    uint32_t n_samples; // number of samples per site
    uint32_t n_sites; // number of sites
    uint32_t block_size; // 8192 values per block by default
    uint32_t n_blocks_site; // Maximum number of blocks for a site
    uint32_t n_total_blocks; // cumsum of blocks
    uint32_t n_total_bitmaps; // cumsum for data_bitmapsblock_sizeTWK_aligned_malloc
    uint32_t m_blocks, m_bitmaps; // allocation for blocblock_sizeTWK_aligned_malloc
    uint32_t* n_blocks; // number of set blocks per site (fixed size of n_sites) and its offset
    uint16_t* blocks; // block id (count order) in [0,n_blocks_site)
    uint64_t* data_bitmaps; // shared array for bitmap data each block has block_size of bitmaps
    pair_t* offsets; // (uint32_t*)offsets;
};


#endif /* FAST_INTERSECT_COUNT_H_ */