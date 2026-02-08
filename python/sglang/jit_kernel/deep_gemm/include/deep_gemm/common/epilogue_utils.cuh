#pragma once

#include <deep_gemm/common/types.hpp>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

struct EpilogueIdentity {
    template <uint32_t STORE_BLOCK_N>
    __device__ __forceinline__ static uint32_t apply_index_n(const uint32_t &n_idx) {
        return n_idx;
    }
};

template <uint32_t kLeft, uint32_t kMid, uint32_t kRight>
struct EpilogueHeadSplits: EpilogueIdentity {
    template <uint32_t STORE_BLOCK_N>
    __device__ __forceinline__ static uint32_t apply_index_n(const uint32_t &n_idx) {
        DG_STATIC_ASSERT(kLeft % STORE_BLOCK_N == 0 and kMid % STORE_BLOCK_N == 0 
                         and kRight % STORE_BLOCK_N == 0, "Invalid head splits config");
        return n_idx + (n_idx + kRight) / (kLeft + kRight) * kMid;
    }
};

#pragma clang diagnostic pop

} // namespace deep_gemm
