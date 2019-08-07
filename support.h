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

/**
 * Idea: use memory-aligned data for skip-list pointers.
 * 
 */
struct skip_list_soa {
    skip_list_soa() : nv(0), no(0), vals(nullptr), offsets(nullptr) {}
    ~skip_list_soa() {
        delete[] vals;
        delete[] offsets;
    }
    
    /**
     * @brief Construct from a positional vector-vector of integers.
     * 
     * @param in 
     * @return int 
     */
    int Build(const std::vector< std::vector<uint32_t> >& in) {
        if (in.size() == 0) return 0;

        // Clean up previous.
        delete[] vals;
        delete[] offsets;
        
        no = in.size();
        // Compute the total number of values.
        nv = 0;
        for (int i = 0; i < in.size(); ++i) nv += in[i].size();
        // Allocate new.
        // offsets = new int32_t[no];
        assert(!posix_memalign((void**)&offsets, SIMD_ALIGNMENT, no*sizeof(int32_t)));
        // vals = new int32_t[nv];
        assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, nv*sizeof(int32_t)));

        uint32_t cum_offset = 0;
        for (int i = 0; i < in.size(); ++i) {
            offsets[i] = in[i].size();
            for (int j = 0; j < in[i].size(); ++j) {
                vals[cum_offset + j] = in[i][j];
            }
            cum_offset += in.size();
        }
        assert(cum_offset == nv);

        return 1;
    }

    int32_t nv;
    int32_t no;
    int32_t* vals;
    int32_t* offsets;
};

#endif