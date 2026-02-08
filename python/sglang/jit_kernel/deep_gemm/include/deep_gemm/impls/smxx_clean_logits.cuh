#pragma once

#include <cutlass/arch/barrier.h>
#include <cute/arch/cluster_sm90.hpp>

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

template <uint32_t kNextN, uint32_t BLOCK_KV, uint32_t kNumWarps>
__global__ __launch_bounds__(kNumWarps * 32, 1)
void smxx_clean_logits(const uint32_t seq_len, const uint32_t seq_len_kv, const uint64_t stride_logits,
                       const uint32_t* cu_seq_len_k_start, const uint32_t* cu_seq_len_k_end, float* logits) {
    const uint32_t& num_sms = gridDim.x;
    const uint32_t& sm_idx = blockIdx.x;
    const uint32_t& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    constexpr float neg_inf = -cute::numeric_limits<float>::infinity();

    // Allocate filled `-inf` shared memory
    extern __shared__ __align__(1024) float smem_buffer[];
    #pragma unroll
    for (uint32_t i = threadIdx.x; i < BLOCK_KV; i += kNumWarps * 32)
        smem_buffer[i] = neg_inf;
    cute::tma_store_fence();
    __syncthreads();

    // Assign sequence to each warp
    const auto& assign_task = [&](const uint32_t& num, const uint32_t& idx,
                                  const uint32_t& start, const uint32_t& total) -> cute::tuple<uint32_t, uint32_t> {
        const auto& per = total / num, rem = total % num;
        return {start + idx * per + min(idx, rem), per + (idx < rem)};
    };
    CUTE_TIE_DECL(assign_task(num_sms, sm_idx, 0, seq_len), sm_seq_start, sm_seq_len);
    CUTE_TIE_DECL(assign_task(kNumWarps, warp_idx, sm_seq_start, sm_seq_len), warp_seq_start, warp_seq_len);

    if (cute::elect_one_sync()) {
        for (uint32_t i = warp_seq_start; i < warp_seq_start + warp_seq_len; ++ i) {
            const auto& ks = cu_seq_len_k_start == nullptr ? 0 : __ldg(cu_seq_len_k_start + i / kNextN);
            const auto& ke = __ldg(cu_seq_len_k_end + i / kNextN) - kNextN + i % kNextN + 1;
            const auto& aligned_ks = ks / 4 * 4, aligned_ke = (ke + 3) / 4 * 4;

            for (uint32_t left = 0; left < seq_len_kv; left += BLOCK_KV) {
                const auto& right = min(left + BLOCK_KV, static_cast<uint32_t>(stride_logits));
                if (right <= ks or ke <= left) {
                    cute::SM90_BULK_COPY_S2G::copy(smem_buffer, logits + i * stride_logits + left, (right - left) * sizeof(float));
                } else {
                    if (left < aligned_ks)
                        cute::SM90_BULK_COPY_S2G::copy(smem_buffer, logits + i * stride_logits + left, (aligned_ks - left) * sizeof(float));
                    if (aligned_ke < right)
                        cute::SM90_BULK_COPY_S2G::copy(smem_buffer, logits + i * stride_logits + aligned_ke, (right - aligned_ke) * sizeof(float));
                }
            }
        }
    }

    for (uint32_t i = warp_seq_start; i < warp_seq_start + warp_seq_len; ++ i) {
        const auto& ks = cu_seq_len_k_start == nullptr ? 0 : __ldg(cu_seq_len_k_start + i / kNextN);
        const auto& ke = __ldg(cu_seq_len_k_end + i / kNextN) - kNextN + i % kNextN + 1;
        const auto& aligned_ks = ks / 4 * 4, aligned_ke = (ke + 3) / 4 * 4;
        for (uint32_t j = aligned_ks; j < ks; ++ j)
            logits[i * stride_logits + j] = neg_inf;
        for (uint32_t j = ke; j < aligned_ke; ++ j)
            logits[i * stride_logits + j] = neg_inf;
    }
}

}
