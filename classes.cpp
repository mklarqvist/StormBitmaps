#include "classes.h"

uint64_t bitmap_t::intersect(const bitmap_t& other) const {
    return TWK_intersect_avx2(data, other.data, n_bitmap);
}

uint64_t bitmap_container_t::intersect_blocked(uint32_t bsize) const {
    return TWK_wrapper_diag_blocked(n_bitmaps, bmaps, n_bitmaps_sample, &TWK_intersect_avx2, bsize);
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
    return TWK_wrapper_diag(n_bitmaps, bmaps, n_bitmaps_sample, &TWK_intersect_avx2);
}

uint64_t bitmap_container_t::intersect_blocked_cont(uint32_t bsize) const {
    return TWK_wrapper_diag_blocked(n_bitmaps, bmaps, n_bitmaps_sample, &TWK_intersect_avx2, bsize);
}

uint64_t bitmap_container_t::intersect_cont_auto() const {
    return TWK_intersect_list(bmaps, n_bitmaps, n_bitmaps_sample, n_alts, alt_positions, alt_offsets, n_alt_cutoff);
}


uint64_t bitmap_container_t::intersect_cont_blocked_auto(uint32_t bsize) const {
    return TWK_intersect_list(bmaps, n_bitmaps, n_bitmaps_sample, n_alts, alt_positions, alt_offsets, n_alt_cutoff, bsize);
}