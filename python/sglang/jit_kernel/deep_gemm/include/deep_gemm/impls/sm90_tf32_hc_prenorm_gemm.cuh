#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/reduction.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;

template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
__device__ __forceinline__
uint32_t get_swizzled_bank_group_idx(const uint32_t& offset, const uint32_t& lane_idx) {
    constexpr uint32_t kGroupsInSwizzleRange = kSwizzleMode / kSwizzleBase;

    const auto& bank_group_idx = offset + lane_idx * kGroupsInSwizzleRange;

    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
    constexpr bool kHasShortcut = kGroupsInSwizzleRange == kNumBankGroups;
    auto row = kHasShortcut ? (offset / kNumBankGroups + lane_idx) : (bank_group_idx / kNumBankGroups);
    auto col = kHasShortcut ? (offset) : (bank_group_idx % kNumBankGroups);
    col ^= row % kGroupsInSwizzleRange;

    return (row * kNumBankGroups + col) % kGroupsInSwizzleRange;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumSplits,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumMathThreads, uint32_t kNumTMAThreads>
__global__ void __launch_bounds__(kNumMathThreads + kNumTMAThreads, 1)
sm90_tf32_hc_prenorm_gemm_impl(const uint32_t shape_m,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                               float* sqr_sum) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // kSwizzleAMode and kSwizzleBMode must be 128 for now
    constexpr uint32_t kSwizzleAMode = cute::min(BLOCK_K * sizeof(nv_bfloat16), 128);
    constexpr uint32_t kSwizzleBMode = cute::min(BLOCK_K * sizeof(float), 128);
    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K");
    DG_STATIC_ASSERT(kSwizzleAMode == 128, "Invalid swizzle A mode");
    DG_STATIC_ASSERT(kSwizzleBMode == 128, "Invalid swizzle B mode");

    DG_STATIC_ASSERT(kSwizzleCDMode / sizeof(float) == BLOCK_N, "Invalid block N");
    DG_STATIC_ASSERT(kNumMathThreads == 128, "Invalid MMA threads");

    // Utils
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(float);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }

    // Data on shared memory (layout as ordered below)
    // Fill D/A/B pointers
    auto smem_cd = reinterpret_cast<float*>(smem_buffer);
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<nv_bfloat16*>(smem_buffer + (SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers           = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers          = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(128);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumKBlocks = constexpr_ceil_div(SHAPE_K, BLOCK_K);
    constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;
    constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;
    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit + cute::min(k_split_idx, kRemainKBlocks)) * BLOCK_K;
    const uint32_t m_offset = shape_m * k_split_idx;
    const uint32_t num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);
    constexpr uint32_t kNumTMARegisters = 40;
    constexpr uint32_t kNumMathRegisters = 256;

    // TMA load warp
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            // Wait consumer release
            const auto& stage_idx = s % kNumStages;
            empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);

            // Compute offsets
            uint32_t m_idx = m_block_idx * BLOCK_M;
            uint32_t k_idx = k_offset + s * BLOCK_K;

            // Issue TMAs
            tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(&tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx);
            tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(&tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, 0);

            // Arrive at full barriers
            constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
            full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes);
        }

        for (uint32_t s = num_total_stages; s < num_total_stages + kNumStages; ++ s) {
            const auto& stage_idx = s % kNumStages;
            empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);
        }
    } else if (warp_idx < kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        DG_STATIC_ASSERT(BLOCK_M == 64, "Invalid block M");
        DG_STATIC_ASSERT(BLOCK_K * sizeof(nv_bfloat16) == kSwizzleAMode, "Invalid block K");
        constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;
        constexpr uint32_t WGMMA_M = 64;
        constexpr uint32_t WGMMA_N = BLOCK_N;
        constexpr uint32_t WGMMA_K = 8;

        using WGMMA = typename TF32MMASelector<WGMMA_N, true>::type;
        float accum[WGMMA::kNumAccum] = {0};

        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(nv_bfloat16);
        constexpr uint32_t kNumLoads = BLOCK_K / kNumElemsPerBankGroup;
        float sqr_sum_acc_0 = 0;
        float sqr_sum_acc_1 = 0;

        #pragma unroll kNumStages < 8 ? kNumStages : kNumStages / 2
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            // Wait TMA arrival
            const auto& stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);

            constexpr uint32_t kNumRegPerWgmma = WGMMA::M * WGMMA::K / 128;
            constexpr uint32_t kNumWgmmaPerBlockK = BLOCK_K / WGMMA::K;

            float a[kNumRegPerWgmma * kNumWgmmaPerBlockK];
            // Assume swizzle A mode is 128
            DG_STATIC_ASSERT(kSwizzleAMode == 128, "Invalid swizzle A mode");

            // Load BF16 A fragment from shared memory into registers, and transpose to FP32
            uint32_t row = warp_idx * 16 + lane_idx / 4;
            #pragma unroll
            for (uint32_t i = 0; i < kNumLoads; ++ i) {
                // Refer to the A layout in https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n8-a
                uint32_t bank_group_idx = (row ^ i) % 8;
                nv_bfloat16* a_bf16_smem_ptr_upper = smem_a[stage_idx] + row * BLOCK_K + bank_group_idx * kNumElemsPerBankGroup;
                nv_bfloat16* a_bf16_smem_ptr_lower = smem_a[stage_idx] + (row + 8) * BLOCK_K + bank_group_idx * kNumElemsPerBankGroup;

                uint32_t elem_offset = lane_idx % 4;
                nv_bfloat16 a_bf16[kNumRegPerWgmma];
                a_bf16[0] = a_bf16_smem_ptr_upper[elem_offset];
                a_bf16[2] = a_bf16_smem_ptr_upper[elem_offset + 4];
                a_bf16[1] = a_bf16_smem_ptr_lower[elem_offset];
                a_bf16[3] = a_bf16_smem_ptr_lower[elem_offset + 4];

                auto a_bf16x2_ptr = reinterpret_cast<nv_bfloat162*>(a_bf16);
                auto a_float2_ptr = reinterpret_cast<float2*>(a);
                float2 a_float2_0 = __bfloat1622float2(a_bf16x2_ptr[0]);
                float2 a_float2_1 = __bfloat1622float2(a_bf16x2_ptr[1]);
                a_float2_ptr[i * 2 + 0] = a_float2_0;
                a_float2_ptr[i * 2 + 1] = a_float2_1;
                sqr_sum_acc_0 += a_float2_0.x * a_float2_0.x + a_float2_1.x * a_float2_1.x;
                sqr_sum_acc_1 += a_float2_0.y * a_float2_0.y + a_float2_1.y * a_float2_1.y;
            }

            warpgroup_wait<0>();
            if (s > 0)
                empty_barriers[(s - 1) % kNumStages]->arrive();

            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                warpgroup_fence_operand(accum[i]);
            warpgroup_arrive();

            constexpr int kNumElemsInSwizzleRange = 128 / sizeof(float);
            constexpr uint32_t kNumWgmmaInSwizzleRange = kNumElemsInSwizzleRange / WGMMA::K;
            DG_STATIC_ASSERT(BLOCK_K % kNumElemsInSwizzleRange == 0, "Invalid block K");

            #pragma unroll
            for (int i = 0; i < BLOCK_K / kNumElemsInSwizzleRange; i++) {
                #pragma unroll
                for (int k = 0; k < kNumElemsInSwizzleRange / WGMMA::K; k++) {
                    auto b_desc = make_smem_desc(smem_b[stage_idx] + i * BLOCK_N * kNumElemsInSwizzleRange + k * WGMMA::K, 1);
                    WGMMA::wgmma(a + (i * kNumWgmmaInSwizzleRange + k) * kNumRegPerWgmma, b_desc, accum, 1);
                }
            }
            warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                warpgroup_fence_operand(accum[i]);
        }

        const auto& reduced_sum_0 = warp_reduce_sum<4>(sqr_sum_acc_0);
        const auto& reduced_sum_1 = warp_reduce_sum<4>(sqr_sum_acc_1);

        const auto& m_idx = m_block_idx * BLOCK_M + (warp_idx * BLOCK_M_PER_WARP + lane_idx / 4);
        if (lane_idx % 4 == 0) {
            if (m_idx < shape_m)
                sqr_sum[m_offset + m_idx] = reduced_sum_0;
            if (m_idx + 8 < shape_m)
                sqr_sum[m_offset + m_idx + 8] = reduced_sum_1;
        }
        warpgroup_wait<0>();
        empty_barriers[(num_total_stages-1) % kNumStages]->arrive();

        // Write accum to shared memory
        // Every 2 threads (one pair) will write to the same bank group (16 bytes).
        // Refer to the D layout in https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n8-d
        uint32_t is_odd_pair = lane_idx / 2 % 2;

        // Four threads per group; write the data to the same row.
        uint32_t row_idx = lane_idx / 4;

        // Even/odd index pairs write to the same column, we need to reorder idx:
        // group even pair indices consecutively, and likewise for odd ones.
        uint32_t reordered_pair_idx = is_odd_pair * 8 + row_idx;

        auto shifted_smem_ptr = reinterpret_cast<uint8_t*>(smem_cd) +
                                (warp_idx * BLOCK_M_PER_WARP + row_idx) * kSwizzleCDMode +  // Row offset, each warp has 16 rows
                                lane_idx % 2 * 8;                                           // One thread of a pair writes 8 bytes

        #pragma unroll
        for (uint32_t i = 0; i < (kSwizzleCDMode / sizeof(float)) / 4; i += 2) {
            // Get the swizzled bank group index (16 bytes per group)
            uint32_t bank_group_idx = get_swizzled_bank_group_idx<kSwizzleCDMode>(i + is_odd_pair, reordered_pair_idx);
            auto smem_ptr = shifted_smem_ptr + bank_group_idx * kNumBankGroupBytes; // Col offset, 16 bytes per group

            // 0/1 write to the same row, 2/3 write to another row
            auto values = reinterpret_cast<uint32_t*>(accum + i * 2);
            st_shared(smem_ptr, values[0], values[1]);
            st_shared(smem_ptr + 8 * kSwizzleCDMode, values[2], values[3]);
        }
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(128, 1);

        // Issue TMA stores
        if (warp_idx == 0 and cute::elect_one_sync()) {
            if constexpr (kNumSplits == 1) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M);
            } else {
                cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_cd, 0, m_block_idx * BLOCK_M, k_split_idx);
            }
            cute::tma_store_arrive();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
