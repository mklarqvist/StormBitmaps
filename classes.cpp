#include "classes.h"

// cpp

uint64_t bitmap_container_t::intersect_blocked(uint32_t bsize) const {
    uint64_t count = 0;
    uint32_t i = 0;
    const STORM_compute_func func = STORM_get_intersect_func(n_bitmaps_sample);

    for (/**/; i + bsize <= n_bitmaps; i += bsize) {
        // diagonal component
        for (uint32_t j = 0; j < bsize; ++j) {
            for (uint32_t jj = j + 1; jj < bsize; ++jj) {
                count += (*func)(bitmaps[i+j].data, bitmaps[i+jj].data, n_bitmaps_sample);
            }
        }

        // square component
        uint32_t curi = i;
        uint32_t j = curi + bsize;
        for (/**/; j + bsize <= n_bitmaps; j += bsize) {
            for (uint32_t ii = 0; ii < bsize; ++ii) {
                for (uint32_t jj = 0; jj < bsize; ++jj) {
                    count += (*func)(bitmaps[curi+ii].data, bitmaps[j+jj].data, n_bitmaps_sample);
                }
            }
        }

        // residual
        for (/**/; j < n_bitmaps; ++j) {
            for (uint32_t jj = 0; jj < bsize; ++jj) {
                count += (*func)(bitmaps[curi+jj].data, bitmaps[j].data, n_bitmaps_sample);
            }
        }
    }
    // residual tail
    for (/**/; i < n_bitmaps; ++i) {
        for (uint32_t j = i + 1; j < n_bitmaps; ++j) {
            count += (*func)(bitmaps[i].data, bitmaps[j].data, n_bitmaps_sample);
        }
    }

    return count;
}

uint64_t bitmap_container_t::intersect() const {
    uint64_t count = 0;
    const STORM_compute_func func = STORM_get_intersect_func(n_bitmaps_sample);

    for (int i = 0; i < n_bitmaps; ++i) {
        for (int j = i+1; j < n_bitmaps; ++j) {
            count += (*func)(bitmaps[i].data, bitmaps[j].data, n_bitmaps_sample);
        }
    }
    return count;
}

uint64_t bitmap_container_t::intersect_cont() const {
    return STORM_wrapper_diag(n_bitmaps, bmaps, n_bitmaps_sample, STORM_get_intersect_func(n_bitmaps_sample));
}

uint64_t bitmap_container_t::intersect_blocked_cont(uint32_t bsize) const {
    return STORM_wrapper_diag_blocked(n_bitmaps, bmaps, n_bitmaps_sample, STORM_get_intersect_func(n_bitmaps_sample), bsize);
}

uint64_t bitmap_container_t::intersect_cont_auto() const {
    return STORM_wrapper_diag_list(n_bitmaps, bmaps, n_bitmaps_sample, n_alts, alt_positions, alt_offsets, STORM_get_intersect_func(n_bitmaps_sample), &STORM_intersect_scalar_list, n_alt_cutoff);
}

uint64_t bitmap_container_t::intersect_cont_blocked_auto(uint32_t bsize) const {
    return STORM_wrapper_diag_list_blocked(n_bitmaps, bmaps, n_bitmaps_sample, n_alts, alt_positions, alt_offsets, STORM_get_intersect_func(n_bitmaps_sample), &STORM_intersect_scalar_list, n_alt_cutoff, bsize);
}