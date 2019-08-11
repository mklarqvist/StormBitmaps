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
#ifndef FAST_INTERSECT_CLASSES_COUNT_H_
#define FAST_INTERSECT_CLASSES_COUNT_H_

#include "fast_intersect_count.h"

#include <vector>
#include <memory> //unique_ptr
#include <iostream>//debug
#include <bitset> //debug

struct bitmap_t {
    bitmap_t() : alignment(get_alignment()), n_set(0), n_bitmap(0), own(false), data(nullptr) {}
    bitmap_t(uint64_t* in, uint32_t n, uint32_t m) : alignment(get_alignment()), n_set(n), n_bitmap(m), own(false), data(in) {}
    ~bitmap_t() {
        if (own) aligned_free_port(data);
    }

    int Allocate(uint32_t n) {
        if (data == nullptr) {
            n_bitmap = n;
            data  = (uint64_t*)aligned_malloc_port(alignment, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        } else {
            if (own) aligned_free_port(data);
            n_bitmap = n;
            data  = (uint64_t*)aligned_malloc_port(alignment, n_bitmap*sizeof(uint64_t));
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
            data  = (uint64_t*)aligned_malloc_port(alignment, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        } else {
            if (own) aligned_free_port(data);
            n_bitmap = n_vals;
            data  = (uint64_t*)aligned_malloc_port(alignment, n_bitmap*sizeof(uint64_t));
            own = true;
            n_set = 0;
        }
        memset(data,0,n_bitmap*sizeof(uint64_t));
        return n;
    }

    inline void Add(const uint64_t pos) { data[pos / 64] |= 1ULL << (pos % 64); }

    void clear() {
        memset(data, 0, n_bitmap*sizeof(uint64_t));
    }

    uint64_t intersect(const bitmap_t& other) const;

    // uint32_t intersect_count(const bitmap_t& other) const {
    //     return intersect_bitmaps_avx512_csa(data, other.data, n_bitmap);
    // }

    uint32_t alignment;
    uint32_t n_set, n_bitmap: 31, own: 1; // number of values set, number of bitmaps, ownership
    uint64_t* data;
};

struct bitmap_container_t {
    bitmap_container_t(uint32_t n, uint32_t m) : 
        alignment(get_alignment()),
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
        alignment(get_alignment()),
        n_alt_cutoff(0),
        n_bitmaps(n), 
        n_samples(m), 
        own(true), 
        type(1),
        n_bitmaps_sample(ceil(n_samples / 64.0)),
        bmaps((uint64_t*)aligned_malloc_port(alignment, n_bitmaps*n_bitmaps_sample*sizeof(uint64_t))), 
        bitmaps(nullptr),
        n_alts_tot(0), m_alts(0),
        alt_positions(nullptr),
        alt_offsets(nullptr),
        n_alts((uint32_t*)aligned_malloc_port(alignment, n_bitmaps*sizeof(uint32_t)))
    {
        memset(bmaps,0,n_bitmaps*n_bitmaps_sample*sizeof(uint64_t));
        memset(n_alts,0,n_bitmaps*sizeof(uint32_t));
        // for (int i = 0; i < n_bitmaps; ++i) {
        //     bitmaps[i].data = &bmaps[i*n_bitmaps_sample];
        //     bitmaps[i].own = false;
        //     bitmaps[i].n_set = 0;
        //     bitmaps[i].n_bitmap = n_bitmaps_sample;
        // }
    }

    bitmap_container_t(uint32_t n, uint32_t m, bool yes, bool yes2) : 
        alignment(get_alignment()),
        n_alt_cutoff(300),
        n_bitmaps(n), 
        n_samples(m), 
        own(true), 
        type(1),
        n_bitmaps_sample(ceil(n_samples / 64.0)),
        bmaps((uint64_t*)aligned_malloc_port(alignment, n_bitmaps*n_bitmaps_sample*sizeof(uint64_t))), 
        bitmaps(nullptr),
        alt_positions(nullptr),
        alt_offsets((uint32_t*)aligned_malloc_port(alignment, n_bitmaps*sizeof(uint32_t))),
        n_alts((uint32_t*)aligned_malloc_port(alignment, n_bitmaps*sizeof(uint32_t)))
    {
        std::cerr << "alignment is = " << alignment << std::endl;
        memset(bmaps,0,n_bitmaps*n_bitmaps_sample*sizeof(uint64_t));
        memset(n_alts,0,n_bitmaps*sizeof(uint32_t));
        // for (int i = 0; i < n_bitmaps; ++i) {
        //     bitmaps[i].data = &bmaps[i*n_bitmaps_sample];
        //     bitmaps[i].own = false;
        //     bitmaps[i].n_set = 0;
        //     bitmaps[i].n_bitmap = n_bitmaps_sample;
        // }
    }

    ~bitmap_container_t() {
        if (own) {
            delete[] bitmaps;
            aligned_free_port(bmaps);
        }
        aligned_free_port(alt_offsets);
        aligned_free_port(n_alts);
        aligned_free_port(alt_positions);
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
                    alt_positions = (uint32_t*)aligned_malloc_port(alignment, new_pos*sizeof(uint32_t));
                    memcpy(alt_positions, old, n_alts_tot*sizeof(uint32_t));
                    m_alts = new_pos;
                    aligned_free_port(old);
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
    roaring2_t(uint32_t n_s, uint32_t n_ss) : alignment(get_alignment()), n_samples(n_s), 
        n_sites(n_ss), block_size(128), n_total_blocks(0),
        n_total_bitmaps(0), m_blocks(0), m_bitmaps(0), n_blocks(nullptr), 
        blocks(nullptr), data_bitmaps(nullptr)
    {}

    ~roaring2_t() {
        aligned_free_port(n_blocks);
        aligned_free_port(blocks);
        aligned_free_port(data_bitmaps);
    }

    int Add(const uint32_t target, const std::vector<uint32_t>& pos) {
        if (pos.size() == 0) return 0;

        if (n_blocks == nullptr) {
            n_blocks = (uint32_t*)aligned_malloc_port(alignment, n_sites*sizeof(uint32_t));
        }

        if (blocks == nullptr) {
            m_blocks = n_blocks_site * 25;
            blocks = (uint16_t*)aligned_malloc_port(alignment, m_blocks*sizeof(uint16_t));
            n_total_blocks = 0;
        }

        if (data_bitmaps == nullptr) {
            m_bitmaps = 65535;
            data_bitmaps = (uint64_t*)aligned_malloc_port(alignment, m_bitmaps*sizeof(uint64_t));
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

    // N containers (blocks) per site
    // each container share memory region
    uint32_t alignment;
    uint32_t n_samples; // number of samples per site
    uint32_t n_sites; // number of sites
    uint32_t block_size; // 8192 values per block by default
    uint32_t n_blocks_site; // Maximum number of blocks for a site
    uint32_t n_total_blocks; // total sum of blocks
    uint32_t n_total_bitmaps; // total sum for data_bitmaps
    uint32_t m_blocks, m_bitmaps; // allocation for blocks,bitmaps
    uint32_t* n_blocks; // number of set blocks per site (fixed size of n_sites)
    uint16_t* blocks; // block id (count order)
    uint64_t* data_bitmaps; // shared array for bitmap data
};


#endif /* FAST_INTERSECT_COUNT_H_ */