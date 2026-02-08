#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSplitFactor,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_bmn_bnk_mn_gemm_impl(const uint32_t shape_s,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                          float *d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Types
    using WGMMA = typename BF16MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    // Shared memory
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);

    // Configs
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_idx();
    DG_STATIC_ASSERT(BLOCK_M == 128, "Invalid block M");
    DG_STATIC_ASSERT(kNumTMAThreads == 128, "Invalid number of TMA threads");
    DG_STATIC_ASSERT(kNumMathThreads == 256, "Invalid number of math threads");

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    // Fill shared memory pointers
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_bfloat16*>(smem_buffer + (i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_bfloat16*>(smem_buffer + (kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers     = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers    = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumMathThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

   // Block indices
    const uint32_t num_n_blocks = ceil_div(SHAPE_N, BLOCK_N);
    const uint32_t num_mn_blocks = num_n_blocks * ceil_div(SHAPE_M, BLOCK_M);
    const uint32_t mn_block_idx = blockIdx.x % num_mn_blocks;
    const uint32_t sk_block_idx = blockIdx.x / num_mn_blocks;
    const uint32_t n_block_idx = mn_block_idx % num_n_blocks;
    const uint32_t m_block_idx = mn_block_idx / num_n_blocks;
    const uint32_t num_total_stages = cute::min(kSplitFactor, shape_s * (SHAPE_K / BLOCK_K) - sk_block_idx * kSplitFactor);

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
            // Persistently schedule over blocks
            #pragma unroll
            for (uint32_t s = 0; s < num_total_stages; ++ s) {
                // Wait consumer release
                const auto& stage_idx = s % kNumStages;
                empty_barriers[stage_idx]->wait((s / kNumStages + 1) & 1);

                auto& full_barrier = *full_barriers[stage_idx];
                const uint32_t& sk_idx = (sk_block_idx * kSplitFactor + s) * BLOCK_K;
                const uint32_t& k_idx = sk_idx % SHAPE_K;
                const uint32_t& s_idx = sk_idx / SHAPE_K;

                constexpr uint32_t kSwizzle = BLOCK_K * sizeof(nv_bfloat16);
                tma_copy<BLOCK_K, BLOCK_M, kSwizzle>(
                    &tensor_map_a, &full_barrier, smem_a[stage_idx], k_idx, m_block_idx * BLOCK_M + s_idx * SHAPE_M, 1);
                tma_copy<BLOCK_K, BLOCK_N, kSwizzle>(
                    &tensor_map_b, &full_barrier, smem_b[stage_idx], k_idx, n_block_idx * BLOCK_N + s_idx * SHAPE_N, 1);
                full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        float accum[WGMMA::kNumAccum] = {0};

        // Launch MMAs
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            // Wait TMA arrivals
            const auto& stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);

            // Commit WGMMA instructions
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                warpgroup_fence_operand(accum[i]);
            warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                auto desc_a = make_smem_desc(smem_a[stage_idx] + (math_wg_idx * WGMMA::M) * BLOCK_K + k * WGMMA::K, 1);
                auto desc_b = make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
                WGMMA::wgmma(desc_a, desc_b, accum, 1);
            }
            warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                warpgroup_fence_operand(accum[i]);
            warpgroup_wait<0>();

            // Notify barrier arrival at the last warpgroup wave
            empty_barriers[stage_idx]->arrive();
        }

        const auto& row = m_block_idx * BLOCK_M + warp_idx * 16 + lane_idx / 4;
        const auto& col = n_block_idx * BLOCK_N + (lane_idx % 4) * 2;
        #pragma unroll
        for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
            if (col + i * 8 >= SHAPE_N)
                break;
            if (row < SHAPE_M) {
                atomicAdd(reinterpret_cast<float2*>(d + (row + 0) * SHAPE_N + col + i * 8),
                          make_float2(accum[i * 4 + 0], accum[i * 4 + 1]));
            }
            if (row + 8 < SHAPE_M) {
                atomicAdd(reinterpret_cast<float2*>(d + (row + 8) * SHAPE_N + col + i * 8),
                          make_float2(accum[i * 4 + 2], accum[i * 4 + 3]));
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm
