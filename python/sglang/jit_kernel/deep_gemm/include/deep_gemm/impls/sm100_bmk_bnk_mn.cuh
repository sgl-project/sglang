#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cute/util/type_traits.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSplitFactor,
          uint32_t kSwizzleABMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages, uint32_t kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
sm100_bmn_bnk_mn_gemm_impl(uint32_t shape_s,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Utils
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();
    DG_STATIC_ASSERT(BLOCK_M == LAYOUT_AD_M and BLOCK_N == 128 and BLOCK_K == 64, "Invalid block size");
    DG_STATIC_ASSERT(kSwizzleABMode == 128 and kSwizzleCDMode == 128, "Invalid swizzle mode");

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Shared memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }

    // Real tensor memory size and offsets
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<BLOCK_N>();

    // Fill D/A/B
    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + (i * SMEM_CD_SIZE_PER_STAGE));
    });
    auto smem_a  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + (SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE));
    });
    auto smem_b  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + (SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE +
            kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers     = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers    = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barrier = barrier_start_ptr + (kNumStages * 2);

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 2 + 1);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
        }
        tmem_full_barrier->init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Block indices
    const uint32_t num_n_blocks = ceil_div(SHAPE_N, BLOCK_N);
    const uint32_t num_mn_blocks = num_n_blocks * ceil_div(SHAPE_M, BLOCK_M);
    const uint32_t mn_block_idx = blockIdx.x % num_mn_blocks;
    const uint32_t sk_block_idx = blockIdx.x / num_mn_blocks;
    const uint32_t n_block_idx = mn_block_idx % num_n_blocks;
    const uint32_t m_block_idx = mn_block_idx / num_n_blocks;
    const uint32_t num_total_stages = cute::min(kSplitFactor, shape_s * (SHAPE_K / BLOCK_K) - sk_block_idx * kSplitFactor);

    if (warp_idx == 0) {
        // TMA load warp
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            const auto& stage_idx = s % kNumStages;
            empty_barriers[stage_idx]->wait(((s / kNumStages) & 1) ^ 1);

            uint32_t m_idx = BLOCK_M * m_block_idx;
            uint32_t n_idx = BLOCK_N * n_block_idx;
            uint32_t sk_idx = (sk_block_idx * kSplitFactor + s) * BLOCK_K;
            uint32_t k_idx = sk_idx % SHAPE_K;
            uint32_t s_idx = sk_idx / SHAPE_K;

            // Issue TMAs
            if (cute::elect_one_sync()) {
                tma_copy<BLOCK_K, BLOCK_M, kSwizzleABMode>(&tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx + s_idx * SHAPE_M);
                tma_copy<BLOCK_K, BLOCK_N, kSwizzleABMode>(&tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx + s_idx * SHAPE_N);
            }

            // Arrive at full barriers
            constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
            if (cute::elect_one_sync())
                full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes);
        }
    } else if (warp_idx == 1) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        constexpr uint32_t UMMA_M = LAYOUT_AD_M;
        constexpr uint32_t UMMA_N = BLOCK_N;
        constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::bfloat16_t);
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float, UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = make_umma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, kSwizzleABMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, kSwizzleABMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Wait tensor memory empty barrier arrival
        tcgen05_after_thread_sync();

        // Launch MMAs
        for (uint32_t s = 0; s < num_total_stages; ++ s) {
            // Wait TMA arrival
            const auto& stage_idx = s % kNumStages;
            full_barriers[stage_idx]->wait((s / kNumStages) & 1);
            tcgen05_after_thread_sync();

            // Issue UMMA in the leader CTA
            const auto& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
            const auto& a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, stage_idx);
            const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, stage_idx);
            if (cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                    a_desc.lo = advance_umma_desc_lo<cute::UMMA::Major::K, BLOCK_M, kSwizzleABMode, cutlass::bfloat16_t>(a_desc_base_lo, 0, k * UMMA_K);
                    b_desc.lo = advance_umma_desc_lo<cute::UMMA::Major::K, BLOCK_N, kSwizzleABMode, cutlass::bfloat16_t>(b_desc_base_lo, 0, k * UMMA_K);
                    SM100_MMA_F16BF16_SS::fma(a_desc, b_desc, 0, s > 0 or k > 0, runtime_instr_desc);
                }
            }

            // Commit to the mbarrier object
            // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
        }
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barrier));
    }

    // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
    // i.e., no need for `tmem_ptr |= (warp_idx * 32) << 16`.
    // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
    if (warp_idx == 2)
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

    // TMA checks
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(float);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(float);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

    // Wait UMMA arrival
    tmem_full_barrier->wait(0);
    tcgen05_after_thread_sync();

    // Load from tensor memory into registers, and write shared memory with STSM
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

    // Issue every swizzled atom and pipeline STSM and TMA store
    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
    #pragma unroll
    for (uint32_t s = 0; s < kNumStores; ++ s) {
        // Wait shared memory to be released
        if (s >= kNumTMAStoreStages) {
            if (warp_idx == 0 and cute::elect_one_sync())
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier(kNumThreads).sync();
        }

        // The pipeline stage
        const auto tma_stage_idx = s % kNumTMAStoreStages;
        const auto m_idx = m_block_idx * BLOCK_M;
        const auto n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

        // Store into shared memory
        #pragma unroll
        for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
            // Calculate the index of the bank group to be written in the atom
            auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

            // Reshape the atom in another view and swizzle
            //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
            //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
            // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
            auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
            auto col = kHasShortcut ? (i) : (bank_group_index % 8);
            col ^= row % (kSwizzleCDMode / 16);

            // Source and destination memory address
            uint32_t tmem_addr = s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;         // In-block offset
            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +        // Base pointer
                            warp_idx * 32 * kSwizzleCDMode +                            // Warp offset
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // In-atom offset

            // Load from tensor memory, store into shared memory
            uint32_t values[kNumElemsPerBankGroup];
            DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                values[0], values[1], values[2], values[3]);
            cutlass::arch::fence_view_async_tmem_load();
            st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
        }

        // Synchronize all threads and issue TMA
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier(kNumThreads).sync();
        if (warp_idx == 0 and cute::elect_one_sync()) {
            cute::SM90_TMA_REDUCE_ADD_2D::copy(&tensor_map_d, smem_cd[tma_stage_idx], n_idx, m_idx);
            cute::tma_store_arrive();
        }
    }

    // Deallocate tensor memory by warp 1
    // NOTES: warp 0 is doing TMA stores
    if (warp_idx == 1)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

}
