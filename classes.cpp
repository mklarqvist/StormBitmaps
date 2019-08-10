#include "classes.h"

uint64_t bitmap_t::intersect(const bitmap_t& other) const {
    return intersect_bitmaps_avx2(data, other.data, n_bitmap);
}

uint64_t bitmap_container_t::intersect() const {
    uint64_t count = 0;
    for (int i = 0; i < n_bitmaps; ++i) {
        for (int j = i+1; j < n_bitmaps; ++j) {
            count += bitmaps[i].intersect(bitmaps[j]);
        }
    }
    return count;
}

uint64_t bitmap_container_t::intersect_cont() const {
    return c_fwrapper(n_bitmaps, bmaps, n_bitmaps_sample, &intersect_bitmaps_avx2);


    // uint64_t count = 0;
    // uint64_t offset1 = 0;
    // uint64_t offset2 = n_bitmaps_sample;
    // for (int i = 0; i < n_bitmaps; ++i, offset1 += n_bitmaps_sample) {
    //     offset2 = offset1 + n_bitmaps_sample;
    //     for (int j = i+1; j < n_bitmaps; ++j, offset2 += n_bitmaps_sample) {
    //         count += intersect_bitmaps_avx2(&bmaps[offset1], &bmaps[offset2], n_bitmaps_sample);
    //     }
    // }
    // return count;
}

uint64_t bitmap_container_t::intersect_blocked_cont(uint32_t bsize) const {
    return c_fwrapper_blocked(n_bitmaps, bmaps, n_bitmaps_sample, &intersect_bitmaps_avx2, bsize);

    // uint64_t total = 0;

    // bsize = (bsize == 0 ? 10 : bsize);
    // const uint32_t n_blocks1 = n_bitmaps / bsize;
    // const uint32_t n_blocks2 = n_bitmaps / bsize;
    // // uint64_t d = 0;

    // uint32_t i  = 0;
    // uint32_t tt = 0;

    // for (/**/; i + bsize <= n_bitmaps; i += bsize) {
    //     // diagonal component
    //     uint32_t left = i*n_bitmaps_sample;
    //     uint32_t right = 0;
    //     for (uint32_t j = 0; j < bsize; ++j, left += n_bitmaps_sample) {
    //         right = left + n_bitmaps_sample;
    //         for (uint32_t jj = j + 1; jj < bsize; ++jj, right += n_bitmaps_sample) {
    //             total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
    //             // ++d;
    //         }
    //     }

    //     // square component
    //     uint32_t curi = i;
    //     uint32_t j = curi + bsize;
    //     for (/**/; j + bsize <= n_bitmaps; j += bsize) {
    //         left = curi*n_bitmaps_sample;
    //         for (uint32_t ii = 0; ii < bsize; ++ii, left += n_bitmaps_sample) {
    //             right = j*n_bitmaps_sample;
    //             for (uint32_t jj = 0; jj < bsize; ++jj, right += n_bitmaps_sample) {
    //                 total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
    //                 // ++d;
    //             }
    //         }
    //     }

    //     // residual
    //     right = j*n_bitmaps_sample;
    //     for (/**/; j < n_bitmaps; ++j, right += n_bitmaps_sample) {
    //         left = curi*n_bitmaps_sample;
    //         for (uint32_t jj = 0; jj < bsize; ++jj, left += n_bitmaps_sample) {
    //             total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
    //             // ++d;
    //         }
    //     }
    // }
    // // residual tail
    // uint32_t left = i*n_bitmaps_sample;
    // for (/**/; i < n_bitmaps; ++i, left += n_bitmaps_sample) {
    //     uint32_t right = left + n_bitmaps_sample;
    //     for (uint32_t j = i + 1; j < n_bitmaps; ++j, right += n_bitmaps_sample) {
    //         total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
    //         // ++d;
    //     }
    // }

    // return total;
}

uint64_t bitmap_container_t::intersect_cont_auto() const {
    uint64_t count = 0;
    uint64_t offset1 = 0;
    uint64_t offset2 = n_bitmaps_sample;
    for (int i = 0; i < n_bitmaps; ++i, offset1 += n_bitmaps_sample) {
        offset2 = offset1 + n_bitmaps_sample;
        for (int j = i+1; j < n_bitmaps; ++j, offset2 += n_bitmaps_sample) {
            if (n_alts[i] < n_alt_cutoff || n_alts[j] < n_alt_cutoff) {
                count += intersect_bitmaps_scalar_list(&bmaps[offset1], &bmaps[offset2], &alt_positions[alt_offsets[i]], &alt_positions[alt_offsets[j]], n_alts[i], n_alts[j]);
            } else {
                count += intersect_bitmaps_avx2(&bmaps[offset1], &bmaps[offset2], n_bitmaps_sample);
            }
        }
    }
    return count;
}


uint64_t bitmap_container_t::intersect_cont_blocked_auto(uint32_t bsize) const {
    uint64_t total = 0;

    bsize = (bsize == 0 ? 10 : bsize);
    const uint32_t n_blocks1 = n_bitmaps / bsize;
    const uint32_t n_blocks2 = n_bitmaps / bsize;
    // uint64_t d = 0;

    uint32_t i  = 0;
    uint32_t tt = 0;

    for (/**/; i + bsize <= n_bitmaps; i += bsize) {
        // diagonal component
        uint32_t left = i*n_bitmaps_sample;
        uint32_t right = 0;
        for (uint32_t j = 0; j < bsize; ++j, left += n_bitmaps_sample) {
            right = left + n_bitmaps_sample;
            for (uint32_t jj = j + 1; jj < bsize; ++jj, right += n_bitmaps_sample) {
                if (n_alts[i+j] < n_alt_cutoff || n_alts[i+jj] < n_alt_cutoff) {
                    total += intersect_bitmaps_scalar_list(&bmaps[left], &bmaps[right], 
                        &alt_positions[alt_offsets[i+j]], &alt_positions[alt_offsets[i+jj]], 
                        n_alts[i+j], n_alts[i+jj]);
                } else {
                    total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
                }
                // total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
                // ++d;
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + bsize;
        for (/**/; j + bsize <= n_bitmaps; j += bsize) {
            left = curi*n_bitmaps_sample;
            for (uint32_t ii = 0; ii < bsize; ++ii, left += n_bitmaps_sample) {
                right = j*n_bitmaps_sample;
                for (uint32_t jj = 0; jj < bsize; ++jj, right += n_bitmaps_sample) {
                    if (n_alts[curi+ii] < n_alt_cutoff || n_alts[j+jj] < n_alt_cutoff) {
                        total += intersect_bitmaps_scalar_list(&bmaps[left], &bmaps[right], 
                            &alt_positions[alt_offsets[curi+ii]], &alt_positions[alt_offsets[j+jj]], 
                            n_alts[curi+ii], n_alts[j+jj]);
                    } else {
                        total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
                    }
                    // total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
                    // ++d;
                }
            }
        }

        // residual
        right = j*n_bitmaps_sample;
        for (/**/; j < n_bitmaps; ++j, right += n_bitmaps_sample) {
            left = curi*n_bitmaps_sample;
            for (uint32_t jj = 0; jj < bsize; ++jj, left += n_bitmaps_sample) {
                if (n_alts[curi+jj] < n_alt_cutoff || n_alts[j] < n_alt_cutoff) {
                    total += intersect_bitmaps_scalar_list(&bmaps[left], &bmaps[right], 
                        &alt_positions[alt_offsets[curi+jj]], &alt_positions[alt_offsets[j]], 
                        n_alts[curi+jj], n_alts[j]);
                } else {
                    total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
                }
                // total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
                // ++d;
            }
        }
    }
    // residual tail
    uint32_t left = i*n_bitmaps_sample;
    for (/**/; i < n_bitmaps; ++i, left += n_bitmaps_sample) {
        uint32_t right = left + n_bitmaps_sample;
        for (uint32_t j = i + 1; j < n_bitmaps; ++j, right += n_bitmaps_sample) {
            if (n_alts[i] < n_alt_cutoff || n_alts[j] < n_alt_cutoff) {
                total += intersect_bitmaps_scalar_list(&bmaps[left], &bmaps[right], 
                    &alt_positions[alt_offsets[i]], &alt_positions[alt_offsets[j]], 
                    n_alts[i], n_alts[j]);
            } else {
                total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
            }
            // total += intersect_bitmaps_avx2(&bmaps[left], &bmaps[right], n_bitmaps_sample);
            // ++d;
        }
    }

    return total;
}