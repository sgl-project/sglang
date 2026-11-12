#pragma once

#include <cute/util/type_traits.hpp>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::epilogue {

// Warp-cooperative `-inf` filler for out-of-coverage logits columns.
// The aligned interior of `[start_col, end_col)` is filled with 16-byte vectorized
// stores, while the misaligned head/tail (and too-short ranges) fall back to scalar stores.
// `kNumLanes` cooperating lanes may span multiple warps; `lane_idx` must be in `[0, kNumLanes)`.
template <typename logits_dtype_t, uint32_t kNumLanes = 32>
struct LogitsCleaner {
    static constexpr uint32_t kNumElemsPerInt4 = static_cast<uint32_t>(sizeof(int4) / sizeof(logits_dtype_t));

    logits_dtype_t neg_inf;
    int4 neg_inf_int4;
    uint32_t lane_idx;

    CUTLASS_DEVICE explicit LogitsCleaner(const uint32_t& lane_idx): lane_idx(lane_idx) {
        neg_inf = -cute::numeric_limits<logits_dtype_t>::infinity();
        const auto neg_inf_elems = reinterpret_cast<logits_dtype_t*>(&neg_inf_int4);
        #pragma unroll
        for (uint32_t i = 0; i < kNumElemsPerInt4; ++ i)
            neg_inf_elems[i] = neg_inf;
    }

    CUTLASS_DEVICE void fill_row(logits_dtype_t* row, const uint32_t& start_col, const uint32_t& end_col) const {
        const auto aligned_start = math::align(start_col, kNumElemsPerInt4);
        const auto aligned_end = math::align<uint32_t, false>(end_col, kNumElemsPerInt4);
        if (aligned_start >= aligned_end) {
            for (uint32_t j = start_col + lane_idx; j < end_col; j += kNumLanes)
                row[j] = neg_inf;
            __syncwarp();
            return;
        }
        for (uint32_t j = start_col + lane_idx; j < aligned_start; j += kNumLanes)
            row[j] = neg_inf;
        for (uint32_t j = aligned_end + lane_idx; j < end_col; j += kNumLanes)
            row[j] = neg_inf;
        __syncwarp();
        for (uint32_t j = aligned_start + lane_idx * kNumElemsPerInt4; j < aligned_end; j += kNumLanes * kNumElemsPerInt4)
            *reinterpret_cast<int4*>(row + j) = neg_inf_int4;
        __syncwarp();
    }
};

} // namespace deep_gemm::epilogue
