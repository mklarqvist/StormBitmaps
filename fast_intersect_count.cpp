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
#include "fast_intersect_count.h"

// ranges
uint64_t intersect_range_bins(const range_bin& b1, const range_bin& b2, const uint8_t n_ints_bin) {
    // Squash
    if((b1.bin_bitmap & b2.bin_bitmap) == 0) return(0);

    uint64_t count = 0;

    if(b1.list || b2.list) {
        //std::cerr << "double list" << std::endl;
        if(b1.n_list < b2.n_list) {
            for(int i = 0; i < b1.pos->size(); ++i) {
                if((b1.bins[b1.pos->at(i)].n_vals == 0) || (b2.bins[b1.pos->at(i)].n_vals == 0))
                    continue;

                if(b1.bins[b1.pos->at(i)].list || b2.bins[b1.pos->at(i)].list) {

                    const bin& bin1 = b1.bins[b1.pos->at(i)];
                    const bin& bin2 = b2.bins[b1.pos->at(i)];

                    if((bin1.bitmap & bin2.bitmap) == 0) {
                        continue;
                    }

                    //std::cerr << "in list-b1" << std::endl;
                    if(bin1.list || bin2.list) {
                        if(bin1.n_list < bin2.n_list) {
                            for(int j = 0; j < bin1.n_list; ++j) {
                                count += TWK_POPCOUNT(bin1.vals[bin1.pos->at(j)] & bin2.vals[bin1.pos->at(j)]);
                            }
                        } else {
                            for(int j = 0; j < bin2.n_list; ++j) {
                                count += TWK_POPCOUNT(bin1.vals[bin2.pos->at(j)] & bin2.vals[bin2.pos->at(j)]);
                            }
                        }
                    }
                } else {
                    for(int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(b1.bins[b1.pos->at(i)].vals[j] & b2.bins[b1.pos->at(i)].vals[j]);
                    }
                }
            }
        } else {
            for(int i = 0; i < b2.n_list; ++i) {
                if((b1.bins[b2.pos->at(i)].n_vals == 0) || (b2.bins[b2.pos->at(i)].n_vals == 0))
                    continue;

                const bin& bin1 = b1.bins[b2.pos->at(i)];
                const bin& bin2 = b2.bins[b2.pos->at(i)];

                if((bin1.bitmap & bin2.bitmap) == 0) {
                    continue;
                }

                if(bin1.list || bin2.list) {
                    if(bin1.n_list < bin2.n_list) {
                        for(int j = 0; j < bin1.n_list; ++j) {
                            count += TWK_POPCOUNT(bin1.vals[bin1.pos->at(j)] & bin2.vals[bin1.pos->at(j)]);
                        }
                    } else {
                        for(int j = 0; j < bin2.n_list; ++j) {
                            count += TWK_POPCOUNT(bin1.vals[bin2.pos->at(j)] & bin2.vals[bin2.pos->at(j)]);
                        }
                    }
                } else {
                    // all
                    for(int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(bin1.vals[j] & bin2.vals[j]);
                    }
                }
            }
        }
    } else { // no lists for either at upper level
        const uint32_t n = b1.bins.size();
        for(int i = 0; i < n; ++i) {
            if(b1.bins[i].n_vals && b2.bins[i].n_vals) {
                if(b1.bins[i].list || b2.bins[i].list) {
                    //std::cerr << "in list-full" << std::endl;
                    const bin& bin1 = b1.bins[i];
                    const bin& bin2 = b2.bins[i];

                    if((bin1.bitmap & bin2.bitmap) == 0)
                       continue;

                    if(bin1.list || bin2.list) {
                        if(bin1.n_list < bin2.n_list) {
                            for(int j = 0; j < bin1.n_list; ++j) {
                                count += TWK_POPCOUNT(bin1.vals[bin1.pos->at(j)] & bin2.vals[bin1.pos->at(j)]);
                            }
                        } else {
                            for(int j = 0; j < bin2.n_list; ++j) {
                                count += TWK_POPCOUNT(bin1.vals[bin2.pos->at(j)] & bin2.vals[bin2.pos->at(j)]);
                            }
                        }
                    } else {
                        // all
                        for(int j = 0; j < n_ints_bin; ++j) {
                            count += TWK_POPCOUNT(bin1.vals[j] & bin2.vals[j]);
                        }
                    }
                } else {
                    // compare values in b
                    for( int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(b1.bins[i].vals[j] & b2.bins[i].vals[j]);
                    }
                }
            }
        }
    }

    return(count);
}

uint64_t intersect_range_bins_bit(const range_bin& b1, const range_bin& b2, const uint8_t n_ints_bin) {
    // Squash
    if((b1.bin_bitmap & b2.bin_bitmap) == 0) return(0);

    uint64_t count = 0;

    if(b1.list || b2.list) {
        if(b1.n_list < b2.n_list) {
            for(int i = 0; i < b1.pos->size(); ++i) {
                if((b1.bins[b1.pos->at(i)].n_vals == 0) || (b2.bins[b1.pos->at(i)].n_vals == 0))
                    continue;

                if(b1.bins[b1.pos->at(i)].list || b2.bins[b1.pos->at(i)].list) {

                    const bin& bin1 = b1.bins[b1.pos->at(i)];
                    const bin& bin2 = b2.bins[b1.pos->at(i)];

                    if((bin1.bitmap & bin2.bitmap) == 0)
                       continue;

                    if(bin1.list || bin2.list) {
                        if(bin1.n_list < bin2.n_list) {
                            for(int j = 0; j < bin1.n_list; ++j) {
                                count += ((bin2.vals[bin1.pos->at(j) / 64] & (1L << (bin1.pos->at(j) % 64))) != 0);
                            }
                        } else {
                            for(int j = 0; j < bin2.n_list; ++j) {
                                count += ((bin1.vals[bin2.pos->at(j) / 64] & (1L << (bin2.pos->at(j) % 64))) != 0);
                            }
                        }
                    }
                } else { // no lists available
                    for(int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(b1.bins[b1.pos->at(i)].vals[j] & b2.bins[b1.pos->at(i)].vals[j]);
                    }
                }
            }
        } else {
            for(int i = 0; i < b2.n_list; ++i) {
                if((b1.bins[b2.pos->at(i)].n_vals == 0) || (b2.bins[b2.pos->at(i)].n_vals == 0))
                    continue;

                const bin& bin1 = b1.bins[b2.pos->at(i)];
                const bin& bin2 = b2.bins[b2.pos->at(i)];

                if((bin1.bitmap & bin2.bitmap) == 0)
                   continue;

                if(bin1.list || bin2.list) {
                    if(bin1.n_list < bin2.n_list) {
                        for(int j = 0; j < bin1.n_list; ++j) {
                            count += ((bin2.vals[bin1.pos->at(j) / 64] & (1L << (bin1.pos->at(j) % 64))) != 0);
                            //count += TWK_POPCOUNT(bin1.vals[bin1.pos->at(j)] & bin2.vals[bin1.pos->at(j)]);
                        }
                    } else {
                        for(int j = 0; j < bin2.n_list; ++j) {
                            count += ((bin1.vals[bin2.pos->at(j) / 64] & (1L << (bin2.pos->at(j) % 64))) != 0);
                            //count += TWK_POPCOUNT(bin1.vals[bin2.pos->at(j)] & bin2.vals[bin2.pos->at(j)]);
                        }
                    }
                } else {
                    // all
                    for(int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(bin1.vals[j] & bin2.vals[j]);
                    }
                }
            }
        }
    } else { // no lists for either at upper level
        const uint32_t n = b1.bins.size();
        for(int i = 0; i < n; ++i) {
            if(b1.bins[i].n_vals && b2.bins[i].n_vals) {
                if(b1.bins[i].list || b2.bins[i].list) {
                    const bin& bin1 = b1.bins[i];
                    const bin& bin2 = b2.bins[i];

                    if((bin1.bitmap & bin2.bitmap) == 0)
                       continue;

                    if(bin1.list || bin2.list) {
                        if(bin1.n_list < bin2.n_list) {
                            for(int j = 0; j < bin1.n_list; ++j) {
                                count += ((bin2.vals[bin1.pos->at(j) / 64] & (1L << (bin1.pos->at(j) % 64))) != 0);
                            }
                        } else {
                            for(int j = 0; j < bin2.n_list; ++j) {
                                count += ((bin1.vals[bin2.pos->at(j) / 64] & (1L << (bin2.pos->at(j) % 64))) != 0);
                            }
                        }
                    } else {
                        // all
                        for(int j = 0; j < n_ints_bin; ++j) {
                            count += TWK_POPCOUNT(bin1.vals[j] & bin2.vals[j]);
                        }
                    }
                } else {
                    // compare values in b
                    for( int j = 0; j < n_ints_bin; ++j) {
                        count += TWK_POPCOUNT(b1.bins[i].vals[j] & b2.bins[i].vals[j]);
                    }
                }
            }
        }
    }

    return(count);
}

//
uint64_t intersect_bitmaps_scalar(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    for(int i = 0; i < n_ints; ++i) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
    }

    return(count);
}

uint64_t intersect_bitmaps_scalar_4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count[4] = {0};
    for(int i = 0; i < n_ints; i += 4) {
        count[0] += TWK_POPCOUNT(b1[i+0] & b2[i+0]);
        count[1] += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
        count[2] += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
        count[3] += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
    }

    uint64_t tot_count = count[0] + count[1] + count[2] + count[3];

    return(tot_count);
}

uint64_t intersect_bitmaps_scalar_1x4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    for(int i = 0; i < n_ints; i += 4) {
        count += TWK_POPCOUNT(b1[i+0] & b2[i+0]);
        count += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
        count += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
        count += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
    }

    return(count);
}

uint64_t intersect_bitmaps_scalar_8way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count[8] = {0};
    for(int i = 0; i < n_ints; i += 8) {
        count[0] += TWK_POPCOUNT(b1[i+0] & b2[i+0]);
        count[1] += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
        count[2] += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
        count[3] += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
        count[4] += TWK_POPCOUNT(b1[i+4] & b2[i+4]);
        count[5] += TWK_POPCOUNT(b1[i+5] & b2[i+5]);
        count[6] += TWK_POPCOUNT(b1[i+6] & b2[i+6]);
        count[7] += TWK_POPCOUNT(b1[i+7] & b2[i+7]);
    }

    uint64_t tot_count = count[0] + count[1] + count[2] + count[3] + count[4] + count[5] + count[6] + count[7];

    return(tot_count);
}

uint64_t intersect_bitmaps_scalar_1x8way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    for(int i = 0; i < n_ints; i += 8) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
        count += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
        count += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
        count += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
        count += TWK_POPCOUNT(b1[i+4] & b2[i+4]);
        count += TWK_POPCOUNT(b1[i+5] & b2[i+5]);
        count += TWK_POPCOUNT(b1[i+6] & b2[i+6]);
        count += TWK_POPCOUNT(b1[i+7] & b2[i+7]);
    }

    return(count);
}

uint64_t intersect_bitmaps_scalar_prefix_suffix(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const std::pair<uint32_t,uint32_t>& p1, const std::pair<uint32_t,uint32_t>& p2) {
    const uint32_t from = std::max(p1.first, p2.first);
    const uint32_t to   = std::min(p1.second,p2.second);

    uint64_t count = 0;
    int i = from;
    for(; i + 4 < to; i += 4) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
        count += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
        count += TWK_POPCOUNT(b1[i+2] & b2[i+2]);
        count += TWK_POPCOUNT(b1[i+3] & b2[i+3]);
    }

    for(; i + 2 < to; i += 2) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
        count += TWK_POPCOUNT(b1[i+1] & b2[i+1]);
    }

    for(; i < to; ++i) {
        count += TWK_POPCOUNT(b1[i] & b2[i]);
    }

    return(count);
}

uint64_t intersect_bitmaps_scalar_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            count += ((b2[l1[i] / 64] & (1L << (l1[i] % 64))) != 0);
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            count += ((b1[l2[i] / 64] & (1L << (l2[i] % 64))) != 0);
        }
    }
    return(count);
}

uint64_t intersect_bitmaps_scalar_list_4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count[4] = {0};

    if(l1.size() < l2.size()) {
        int i = 0;

        for(; i + 4 < l1.size(); i += 4) {
            count[0] += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
            count[1] += ((b2[l1[i+1] / 64] & (1L << (l1[i+1] % 64))) != 0);
            count[2] += ((b2[l1[i+2] / 64] & (1L << (l1[i+2] % 64))) != 0);
            count[3] += ((b2[l1[i+3] / 64] & (1L << (l1[i+3] % 64))) != 0);
        }


        for(; i + 2 < l1.size(); i += 2) {
            count[0] += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
            count[1] += ((b2[l1[i+1] / 64] & (1L << (l1[i+1] % 64))) != 0);
        }


        for(; i < l1.size(); ++i) {
            count[0] += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
        }
    } else {
        int i = 0;

        for(; i + 4 < l2.size(); i += 4) {
            count[0] += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
            count[1] += ((b1[l2[i+1] / 64] & (1L << (l2[i+1] % 64))) != 0);
            count[2] += ((b1[l2[i+2] / 64] & (1L << (l2[i+2] % 64))) != 0);
            count[3] += ((b1[l2[i+3] / 64] & (1L << (l2[i+3] % 64))) != 0);
        }

        for(; i + 2 < l2.size(); i += 2) {
            count[0] += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
            count[1] += ((b1[l2[i+1] / 64] & (1L << (l2[i+1] % 64))) != 0);
        }


        for(; i < l2.size(); ++i) {
            count[0] += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
        }
    }
    return(count[0] + count[1] + count[2] + count[3]);
}

uint64_t intersect_bitmaps_scalar_list_1x4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    if(l1.size() < l2.size()) {
        int i = 0;

        for(; i + 4 < l1.size(); i += 4) {
            count += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
            count += ((b2[l1[i+1] / 64] & (1L << (l1[i+1] % 64))) != 0);
            count += ((b2[l1[i+2] / 64] & (1L << (l1[i+2] % 64))) != 0);
            count += ((b2[l1[i+3] / 64] & (1L << (l1[i+3] % 64))) != 0);
        }

        for(; i + 2 < l1.size(); i += 2) {
            count += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
            count += ((b2[l1[i+1] / 64] & (1L << (l1[i+1] % 64))) != 0);
        }


        for(; i < l1.size(); ++i) {
            count += ((b2[l1[i+0] / 64] & (1L << (l1[i+0] % 64))) != 0);
        }
    } else {
        int i = 0;

        for(; i + 4 < l2.size(); i += 4) {
            count += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
            count += ((b1[l2[i+1] / 64] & (1L << (l2[i+1] % 64))) != 0);
            count += ((b1[l2[i+2] / 64] & (1L << (l2[i+2] % 64))) != 0);
            count += ((b1[l2[i+3] / 64] & (1L << (l2[i+3] % 64))) != 0);
        }

        for(; i + 2 < l2.size(); i += 2) {
            count += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
            count += ((b1[l2[i+1] / 64] & (1L << (l2[i+1] % 64))) != 0);
        }


        for(; i < l2.size(); ++i) {
            count += ((b1[l2[i+0] / 64] & (1L << (l2[i+0] % 64))) != 0);
        }
    }
    return(count);
}

uint64_t intersect_bitmaps_scalar_intlist(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            count += TWK_POPCOUNT(b1[l1[i]] & b2[l1[i]]);
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            count += TWK_POPCOUNT(b1[l2[i]] & b2[l2[i]]);
        }
    }
    return(count);
}

uint64_t intersect_bitmaps_scalar_intlist_1x4way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    if(l1.size() < l2.size()) {
        int i = 0;
        for(; i + 4 < l1.size(); i += 4) {
            count += TWK_POPCOUNT(b1[l1[i+0]] & b2[l1[i+0]]);
            count += TWK_POPCOUNT(b1[l1[i+1]] & b2[l1[i+1]]);
            count += TWK_POPCOUNT(b1[l1[i+2]] & b2[l1[i+2]]);
            count += TWK_POPCOUNT(b1[l1[i+3]] & b2[l1[i+3]]);
        }

        for(; i + 2 < l1.size(); i += 2) {
            count += TWK_POPCOUNT(b1[l1[i+0]] & b2[l1[i+0]]);
            count += TWK_POPCOUNT(b1[l1[i+1]] & b2[l1[i+1]]);
        }

        for(; i < l1.size(); ++i) {
            count += TWK_POPCOUNT(b1[l1[i]] & b2[l1[i]]);
        }
    } else {
        int i = 0;
        for(; i + 4 < l2.size(); i += 4) {
            count += TWK_POPCOUNT(b1[l2[i+0]] & b2[l2[i+0]]);
            count += TWK_POPCOUNT(b1[l2[i+1]] & b2[l2[i+1]]);
            count += TWK_POPCOUNT(b1[l2[i+2]] & b2[l2[i+2]]);
            count += TWK_POPCOUNT(b1[l2[i+3]] & b2[l2[i+3]]);
        }

        for(; i + 2 < l2.size(); i += 2) {
            count += TWK_POPCOUNT(b1[l2[i+0]] & b2[l2[i+0]]);
            count += TWK_POPCOUNT(b1[l2[i+1]] & b2[l2[i+1]]);
        }

        for(; i < l2.size(); ++i) {
            count += TWK_POPCOUNT(b1[l2[i]] & b2[l2[i]]);
        }
    }
    return(count);
}

#if SIMD_VERSION >= 3
uint64_t intersect_bitmaps_sse4(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    for(int i = 0; i < n_cycles; ++i) {
        __m128i v1 = _mm_and_si128(r1[i], r2[i]);
        TWK_POPCOUNT_SSE4(count, v1);
    }

    return(count);
}

uint64_t intersect_bitmaps_sse4_2way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count[2] = {0};
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    int i = 0;
    for(; i + 2 < n_cycles; i += 2) {
        __m128i v1 = _mm_and_si128(r1[i+0], r2[i+0]);
        TWK_POPCOUNT_SSE4(count[0], v1);
        v1 = _mm_and_si128(r1[i+1], r2[i+1]);
        TWK_POPCOUNT_SSE4(count[1], v1);
    }

    for(; i < n_cycles; ++i) {
        __m128i v1 = _mm_and_si128(r1[i+0], r2[i+0]);
        TWK_POPCOUNT_SSE4(count[0], v1);
    }

    return(count[0] + count[1]);
}

uint64_t intersect_bitmaps_sse4_1x2way(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;
    const uint32_t n_cycles = n_ints / 2;

    int i = 0;
    for(; i + 2 < n_cycles; i += 2) {
        __m128i v1 = _mm_and_si128(r1[i+0], r2[i+0]);
        TWK_POPCOUNT_SSE4(count, v1);
        v1 = _mm_and_si128(r1[i+1], r2[i+1]);
        TWK_POPCOUNT_SSE4(count, v1);
    }

    for(; i < n_cycles; ++i) {
        __m128i v1 = _mm_and_si128(r1[i+0], r2[i+0]);
        TWK_POPCOUNT_SSE4(count, v1);
    }

    return(count);
}

uint64_t intersect_bitmaps_sse4_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    const __m128i* r1 = (__m128i*)b1;
    const __m128i* r2 = (__m128i*)b2;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            TWK_POPCOUNT_SSE(count, _mm_and_si128(r1[l1[i]], r2[l1[i]]));
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            TWK_POPCOUNT_SSE(count, _mm_and_si128(r1[l2[i]], r2[l2[i]]));
        }
    }
    return(count);
}

uint64_t intersect_bitmaps_sse4_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    return(intersect_bitmaps_sse4(b1,b2,n_ints));
}

uint64_t intersect_bitmaps_sse4_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    return(intersect_bitmaps_sse4_list(b1,b2,l1,l2));
}

uint64_t insersect_reduced_sse4(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint16_t>& l1, const std::vector<uint16_t>& l2) {
    const __m128i full_vec = _mm_set1_epi16(0xFFFF);
    const __m128i one_mask = _mm_set1_epi16(1);
    const __m128i range    = _mm_set_epi16(8,7,6,5,4,3,2,1);
    uint64_t count = 0; // helper

    if(l1.size() < l2.size()) {
        const __m128i* y = (const __m128i*)&l2[0];
        const uint32_t n_y = l2.size() / 8; // 128 / 16 vectors

        for(int i = 0; i < l1.size(); ++i) {
            const __m128i x = _mm_set1_epi16(l1[i]); // Broadcast single reference value
            int j = 0;
            for(; j < n_y; ++j) {
                if(l2[j*8] > l1[i]) goto done; // if the current value is larger than the reference value break
                __m128i cmp = _mm_cmpeq_epi16(x, y[j]);
                if(_mm_testz_si128(cmp, full_vec) == false) {
                    const __m128i v = _mm_mullo_epi16(_mm_and_si128(cmp, one_mask), range);
                    const uint16_t* vv = (const uint16_t*)&v;
                    const uint32_t pp = (vv[0] + vv[1] + vv[2] + vv[3] + vv[4] + vv[5] + vv[6] + vv[7]) - 1;
                    count += TWK_POPCOUNT(b1[i] & b2[j*8 + pp]);
                    goto done;
                }
            }

            // Scalar residual
            j *= 8;
            for(; j < l2.size(); ++j) {
                if(l2[j] > l1[i]) goto done;

                if(l1[i] == l2[j]) {
                    //std::cerr << "overlap in scalar=" << l1[i] << "," << l2[j] << std::endl;
                    count += TWK_POPCOUNT(b1[i] & b2[j]);
                    goto done;
                }
            }
            done:
            continue;
        }
    } else {
        const __m128i* y = (const __m128i*)&l1[0];
        const uint32_t n_y = l1.size() / 8; // 128 / 16 vectors

        for(int i = 0; i < l2.size(); ++i) {
            const __m128i x = _mm_set1_epi16(l2[i]); // Broadcast single reference value
            int j = 0;
            for(; j < n_y; ++j) {
                if(l1[j*8] > l2[i]) goto doneLower; // if the current value is larger than the reference value break
                __m128i cmp = _mm_cmpeq_epi16(x, y[j]);
                if(_mm_testz_si128(cmp, full_vec) == false) {
                    const __m128i v = _mm_mullo_epi16(_mm_and_si128(cmp, one_mask), range);
                    const uint16_t* vv = (const uint16_t*)&v;
                    const uint32_t pp = (vv[0] + vv[1] + vv[2] + vv[3] + vv[4] + vv[5] + vv[6] + vv[7]) - 1;
                    count += TWK_POPCOUNT(b2[i] & b1[j*8 + pp]);
                    goto doneLower;
                }
            }

            // Scalar residual
            j *= 8;
            for(; j < l1.size(); ++j) {
                if(l1[j] > l2[i]) goto doneLower;

                if(l2[i] == l1[j]) {
                    //std::cerr << "overlap in scalar=" << l1[i] << "," << l2[j] << std::endl;
                    count += TWK_POPCOUNT(b2[i] & b1[j]);
                    goto doneLower;
                }
            }
            doneLower:
            continue;
        }
    }

    //std::cerr << "final=" << count << std::endl;
    return(count);
}

uint64_t insersect_reduced_scalar(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint16_t>& l1, const std::vector<uint16_t>& l2) {
    uint64_t count = 0; // helper

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            for(int j = 0; j < l2.size(); ++j) {
                if(l2[j] > l1[i]) break;

                if(l1[i] == l2[j]) {
                    count += TWK_POPCOUNT(b1[i] & b2[j]);
                    break;
                }
            }
            continue;
        }

    } else {
        for(int i = 0; i < l2.size(); ++i) {
            for(int j = 0; j < l1.size(); ++j) {
                if(l1[j] > l2[i]) break;

                if(l2[i] == l1[j]) {
                    count += TWK_POPCOUNT(b2[i] & b1[j]);
                    break;
                }
            }
            continue;
        }
    }

    return(count);
}
#else
uint64_t intersect_bitmaps_sse4(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_sse4_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
uint64_t intersect_bitmaps_sse4_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
uint64_t intersect_bitmaps_sse4_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
#endif // sse4 available


#if SIMD_VERSION >= 5

#ifndef TWK_POPCOUNT_AVX2
#define TWK_POPCOUNT_AVX2(A, B) {                  \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 0)); \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 1)); \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 2)); \
    A += TWK_POPCOUNT(_mm256_extract_epi64(B, 3)); \
}
#endif

uint64_t intersect_bitmaps_avx2(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;
    const uint32_t n_cycles = n_ints / 4;

    for(int i = 0; i < n_cycles; ++i) {
        TWK_POPCOUNT_AVX2(count, _mm256_and_si256(r1[i], r2[i]));
    }

    return(count);
}

uint64_t intersect_bitmaps_avx2_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;

    const __m256i* r1 = (__m256i*)b1;
    const __m256i* r2 = (__m256i*)b2;

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            TWK_POPCOUNT_AVX2(count, _mm256_and_si256(r1[l1[i]], r2[l1[i]]));
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            TWK_POPCOUNT_AVX2(count, _mm256_and_si256(r1[l2[i]], r2[l2[i]]));
        }
    }
    return(count);
}

uint64_t intersect_bitmaps_avx2_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    return(intersect_bitmaps_avx2(b1,b2,n_ints));
}

uint64_t intersect_bitmaps_avx2_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    return(intersect_bitmaps_avx2_list(b1,b2,l1,l2));
}

#else
uint64_t intersect_bitmaps_avx2(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_avx2_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
uint64_t intersect_bitmaps_avx2_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
uint64_t intersect_bitmaps_avx2_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
#endif // endif avx2


#if SIMD_VERSION >= 6

__attribute__((always_inline))
static inline __m512i TWK_AVX512_POPCNT(const __m512i v) {
    const __m512i m1 = _mm512_set1_epi8(0x55);
    const __m512i m2 = _mm512_set1_epi8(0x33);
    const __m512i m4 = _mm512_set1_epi8(0x0F);

    const __m512i t1 = _mm512_sub_epi8(v,       (_mm512_srli_epi16(v,  1) & m1));
    const __m512i t2 = _mm512_add_epi8(t1 & m2, (_mm512_srli_epi16(t1, 2) & m2));
    const __m512i t3 = _mm512_add_epi8(t2, _mm512_srli_epi16(t2, 4)) & m4;
    return _mm512_sad_epu8(t3, _mm512_setzero_si512());
}

uint64_t intersect_bitmaps_avx512(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) {
    uint64_t count = 0;
    const __m512i* r1 = (__m512i*)b1;
    const __m512i* r2 = (__m512i*)b2;
    const uint32_t n_cycles = n_ints / 8;
    __m512i sum = _mm512_set1_epi32(0);

    for(int i = 0; i < n_cycles; ++i) {
        sum = _mm512_add_epi32(sum, TWK_AVX512_POPCNT(_mm512_and_si512(r1[i], r2[i])));
    }

    uint32_t* v = reinterpret_cast<uint32_t*>(&sum);
    for(int i = 0; i < 16; ++i)
        count += v[i];

    return(count);
}

uint64_t intersect_bitmaps_avx512_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) {
    uint64_t count = 0;
    const __m512i* r1 = (__m512i*)b1;
    const __m512i* r2 = (__m512i*)b2;
    __m512i sum = _mm512_set1_epi32(0);

    if(l1.size() < l2.size()) {
        for(int i = 0; i < l1.size(); ++i) {
            sum = _mm512_add_epi32(sum, TWK_AVX512_POPCNT(_mm512_and_si512(r1[l1[i]], r2[l1[i]])));
        }
    } else {
        for(int i = 0; i < l2.size(); ++i) {
            sum = _mm512_add_epi32(sum, TWK_AVX512_POPCNT(_mm512_and_si512(r1[l2[i]], r2[l2[i]])));
        }
    }

    uint32_t* v = reinterpret_cast<uint32_t*>(&sum);
    for(int i = 0; i < 16; ++i)
        count += v[i];

    return(count);
}

uint64_t intersect_bitmaps_avx512_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
       count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    return(intersect_bitmaps_avx512(b1, b2, n_ints));
}

uint64_t intersect_bitmaps_avx512_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) {
    uint64_t count = 0;

    for(int i = 0; i < n_squash; ++i) {
        count += ((sq1[i] & sq2[i]) != 0);
    }
    if(count == 0) return 0;

    return(intersect_bitmaps_avx512_list(b1,b2,l1,l2));
}


#else
uint64_t intersect_bitmaps_avx512(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints) { return(0); }
uint64_t intersect_bitmaps_avx512_list(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2) { return(0); }
uint64_t intersect_bitmaps_avx512_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const uint32_t n_ints, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
uint64_t intersect_bitmaps_avx512_list_squash(const uint64_t* __restrict__ b1, const uint64_t* __restrict__ b2, const std::vector<uint32_t>& l1, const std::vector<uint32_t>& l2, const uint32_t n_squash, const std::vector<uint64_t>& sq1, const std::vector<uint64_t>& sq2) { return(0); }
#endif // endif avx512

uint64_t intersect_raw_naive(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2) {
    uint64_t count = 0;
    for(int i = 0; i < v1.size(); ++i) {
        for(int j = 0; j < v2.size(); ++j) {
            count += (v1[i] == v2[j]);
        }
    }
    return(count);
}

#if SIMD_VERSION >= 3
uint64_t intersect_raw_sse4_broadcast(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2) {
    uint64_t count = 0;
    const __m128i one_mask = _mm_set1_epi16(1);
    if(v1.size() < v2.size()) { // broadcast-compare V1-values to vectors of V2 values
        const uint32_t n_cycles = v2.size() / 8;
       // const __m128i* y = (const __m128i*)(&v2[0]);

        for(int i = 0; i < v1.size(); ++i) {
            const __m128i r = _mm_set1_epi16(v1[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m128i y = _mm_loadu_si128((const __m128i*)&v2[j*8]);
                TWK_POPCOUNT_SSE4(count, _mm_and_si128(_mm_cmpeq_epi16(r, y),one_mask));
            }
            j *= 8;
            for(; j < v2.size(); ++j) count += (v1[i] == v2[j]);
        }
    } else {
        const uint32_t n_cycles = v1.size() / 8;

        for(int i = 0; i < v2.size(); ++i) {
            const __m128i r = _mm_set1_epi16(v2[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m128i y = _mm_loadu_si128((const __m128i*)&v1[j*8]);
                TWK_POPCOUNT_SSE4(count, _mm_and_si128(_mm_cmpeq_epi16(r, y),one_mask));
            }
            j *= 8;
            for(; j < v1.size(); ++j) count += (v1[j] == v2[i]);
        }
    }
    return(count);
}
#else
uint64_t intersect_raw_sse4_broadcast(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2) { return(0); }
#endif

#if SIMD_VERSION >= 5
uint64_t intersect_raw_avx2_broadcast(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2) {
    uint64_t count = 0;
    const __m256i one_mask = _mm256_set1_epi16(1);
    if(v1.size() < v2.size()) { // broadcast-compare V1-values to vectors of V2 values
        const uint32_t n_cycles = v2.size() / 16;

        for(int i = 0; i < v1.size(); ++i) {
            __m256i r = _mm256_set1_epi16(v1[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m256i y = _mm256_loadu_si256((const __m256i*)&v2[j*16]);
                TWK_POPCOUNT_AVX2(count, _mm256_and_si256(_mm256_cmpeq_epi16(r, y), one_mask));
            }
            j *= 16;
            for(; j < v2.size(); ++j) count += (v1[i] == v2[j]);
        }
    } else {
        const uint32_t n_cycles = v1.size() / 16;

        for(int i = 0; i < v2.size(); ++i) {
            __m256i r = _mm256_set1_epi16(v2[i]);

            int j = 0;
            for(; j < n_cycles; ++j) {
                const __m256i y = _mm256_loadu_si256((const __m256i*)&v1[j*16]);
                TWK_POPCOUNT_AVX2(count, _mm256_and_si256(_mm256_cmpeq_epi16(r, y), one_mask));
            }
            j *= 16;
            for(; j < v1.size(); ++j)  count += (v1[j] == v2[i]);
        }
    }
    return(count);
}
#else
uint64_t intersect_raw_avx2_broadcast(const std::vector<uint16_t>& v1, const std::vector<uint16_t>& v2) { return(0); }
#endif

// ewah
void construct_ewah64(const uint64_t* input, const uint32_t n_vals) {
    struct control_word {
        uint64_t type: 1, symbol: 1, length: 62;
    };

    std::vector<control_word> words;
    control_word current;
    uint32_t i = 0;
    // first word
    if(input[0] == 0) {
        current.type = 1;
        current.symbol = 0;
        current.length = 1;
        for(i = 1; i < n_vals; ++i) {
            if(input[i] != 0) break;
            ++current.length;
        }
        words.push_back(current);
    }
    else if(input[0] == std::numeric_limits<uint64_t>::max()) {
        current.type = 1;
        current.symbol = 1;
        current.length = 1;
        for(i = 1; i < n_vals; ++i) {
            if(input[i] != std::numeric_limits<uint64_t>::max()) break;
            ++current.length;
        }
        words.push_back(current);
    } else {
        current.type = 0;
        current.symbol = 0;
        current.length = 1;
        for(i = 1; i < n_vals; ++i) {
            if(input[i] == 0 || input[i] == std::numeric_limits<uint64_t>::max()) break;
            ++current.length;
        }
        words.push_back(current);
    }
    current = control_word();
    if(i < n_vals) {
        // remainder words
        while(true) {
            if(input[i] == 0) {
                current.type = 1;
                current.symbol = 0;
                current.length = 1;
                ++i;
                for(; i < n_vals; ++i) {
                    if(input[i] != 0) break;
                    ++current.length;
                }
                words.push_back(current);
            }
            else if(input[i] == std::numeric_limits<uint64_t>::max()) {
                current.type = 1;
                current.symbol = 1;
                current.length = 1;
                ++i;
                for(; i < n_vals; ++i) {
                    if(input[i] != std::numeric_limits<uint64_t>::max()) break;
                    ++current.length;
                }
                words.push_back(current);
            } else {
                current.type = 0;
                current.symbol = 0;
                current.length = 1;
                ++i;
                for(; i < n_vals; ++i) {
                    if(input[i] == 0 || input[i] == std::numeric_limits<uint64_t>::max()) break;
                    ++current.length;
                }
                words.push_back(current);
            }
            if(i >= n_vals) break;
            current = control_word();
        }
    }

    uint32_t sum = 0;
    std::cerr << words.size() << ":";
    for(int i = 0; i < words.size(); ++i) {
        std::cerr << " {" << words[i].length << "," << words[i].symbol << "," << (words[i].type == 0 ? "dirty" : "clean") << "}";
        sum += words[i].length;
    }
    std::cerr << " : total=" << sum << std::endl;
}
