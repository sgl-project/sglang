// Derived only at the output-store hook from installed DeepGEMM sm100_mqa_logits.cuh.
// Original SHA256 3cbd2c04a9190b742a749b6646b18b28be473ab403a848c764b18471cb9a70c3.
// Copyright (c) 2025 DeepSeek, MIT. See LICENSE.DeepGEMM.
#pragma once

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/mqa_logits.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

// Shared SM100 MQA logits core plus contiguous-KV and paged entries
// Both entries use the same q / sf_q / kv / sf_kv / weights TMA signature

namespace deep_gemm {

// Ring-buffer counter avoiding `% kNumStages`, which ptxas can lower poorly for TMEM paths
template <uint32_t kNumStages>
struct RingPipeline {
    uint32_t stage_idx = 0, phase = 0;

    CUTLASS_DEVICE cute::tuple<uint32_t, uint32_t> advance(const uint32_t& step = 1) {
        const uint32_t current_stage_idx = stage_idx, current_phase = phase;
        stage_idx += step;
        if (stage_idx >= kNumStages) {
            stage_idx -= kNumStages;
            phase ^= 1;
        }
        return {current_stage_idx, current_phase};
    }
};

// Convert runtime valid-token count to `cute::Int` so token loops stay compile-time constant
template <uint32_t kBlockQ, uint32_t kCandidate = kBlockQ, typename Fn>
CUTLASS_DEVICE void dispatch_num_block_tokens(const uint32_t& num_block_tokens, Fn&& fn) {
    if constexpr (kCandidate <= 1) {
        fn(cute::Int<1>{});
    } else if (num_block_tokens >= kCandidate) {
        fn(cute::Int<kCandidate>{});
    } else {
        dispatch_num_block_tokens<kBlockQ, kCandidate - 1>(num_block_tokens, static_cast<Fn&&>(fn));
    }
}

// Shared device core parameterized by dtype and scheduler geometry/addressing
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF, bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t, typename MakeScheduler, typename Emit,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_DEVICE void sm100_mqa_logits_filter_core_impl(const uint32_t logits_stride,
                                               logits_dtype_t* logits,
                                               const cute::TmaDescriptor& tensor_map_q,
                                               const cute::TmaDescriptor& tensor_map_sf_q,
                                               const cute::TmaDescriptor& tensor_map_kv,
                                               const cute::TmaDescriptor& tensor_map_sf_kv,
                                               const cute::TmaDescriptor& tensor_map_weights,
                                               const MakeScheduler& make_scheduler, Emit& emit) {
    constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;

    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    static constexpr uint32_t kNumTmemStages = 3;
    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;
    static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    static constexpr uint32_t kNumSFQ  = kIsMXSF ? math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kNumSFKV = kIsMXSF ? math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kRealNumSFQ = BLOCK_Q * kNumHeads;
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = kIsMXSF ? (kRealNumSFQ * sizeof(int)) : 0;
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kIsMXSF ? (kNumSFKV * sizeof(int)) : (SPLIT_KV * sizeof(float));
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(reduce_dtype_t);

    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");

    using SharedStorage = layout::MQALogitsSharedStorage<kNumHeads, kHeadDim, kIsMXSF, BLOCK_Q, SPLIT_KV,
                                                         kNumQStages, kNumKVStages, kNumTmemStages, qk_dtype_t, reduce_dtype_t>;
    extern __shared__ __align__(SharedStorage::kSwizzleAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    constexpr uint32_t kNumAccumTmemCols = BLOCK_Q * kNumHeads * kNumTmemStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 + kNumSFKV / 32>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            smem.full_q_barriers[i].init(1);
            smem.empty_q_barriers[i].init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            smem.full_kv_barriers[i].init(1);
            smem.empty_kv_barriers[i].init(kIsMXSF ? 1 : kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++i) {
            smem.full_tmem_barriers[i].init(1);
            smem.empty_tmem_barriers[i].init(128);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();

    if (warp_idx == kSpecWarpStart + 2)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
    __syncthreads();

    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];

    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;

    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumMathRegisters = 224;

    cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (cute::elect_one_sync()) {
            auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
            // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

                const uint32_t q_token_base = scheduler.get_q_tma_token_base(q_block_idx);
                tma::copy<kNumQKBytesPerToken, BLOCK_Q * kNumHeads, 0>(
                    &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                    smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
                if constexpr (kIsMXSF)
                    tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx], smem.smem_sf_q[q_stage_idx], 0, q_token_base);
                tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, &smem.full_q_barriers[q_stage_idx], smem.smem_weights[q_stage_idx], 0, q_token_base);
                smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        uint32_t cached_kv_page_base = 0;
        uint32_t cached_kv_page_coord = 0;
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            cached_kv_page_base = cute::numeric_limits<uint32_t>::max();
            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                if constexpr (decltype(scheduler)::kIsPaged) {
                    constexpr uint32_t kPageKV = decltype(scheduler)::kPageKV;
                    constexpr uint32_t kNumPagesPerSplit = decltype(scheduler)::kNumPagesPerSplit;
                    DG_STATIC_ASSERT(kNumPagesPerSplit <= 32, "Split spans more pages than a warp can cache");

                    const uint32_t kv_page_base = (kv_base + kv_split_idx) * kNumPagesPerSplit;
                    if (kv_page_base < cached_kv_page_base or kv_page_base + kNumPagesPerSplit > cached_kv_page_base + 32) {
                        cached_kv_page_base = (kv_page_base / 32) * 32;
                        cached_kv_page_coord = scheduler.get_kv_page_coord_by_page_offset(cached_kv_page_base + lane_idx);
                    }

                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    if (cute::elect_one_sync())
                        smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
                    __syncwarp();

                    int page_coords[kNumPagesPerSplit];
                    #pragma unroll
                    for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx) {
                        const auto src_lane = static_cast<int>(kv_page_base - cached_kv_page_base + page_idx);
                        page_coords[page_idx] = __shfl_sync(0xffffffff, cached_kv_page_coord, src_lane);
                    }

                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx) {
                            tma::copy<kNumQKBytesPerToken, kPageKV, 0, qk_dtype_t, true>(
                                &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                                smem.smem_kv[kv_stage_idx] + page_idx * kPageKV * kNumQKBytesPerToken,
                                0, 0, 1, page_coords[page_idx]);
                            tma::copy<kPageKV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                                     smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                                                     0, page_coords[page_idx]);
                        }
                        smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                    }
                    __syncwarp();
                } else if (cute::elect_one_sync()) {
                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);

                    const uint32_t kv_tma_offset = scheduler.get_kv_tma_offset(kv_base, kv_split_idx);
                    tma::copy<kNumQKBytesPerToken, SPLIT_KV, 0>(
                        &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                        smem.smem_kv[kv_stage_idx], 0, kv_tma_offset);
                    tma::copy<SPLIT_KV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                              smem.smem_sf_kv[kv_stage_idx], kv_tma_offset, 0);
                    smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kSpecWarpStart + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);

        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            if constexpr (kIsMXSF) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                }
                cutlass::arch::fence_view_async_shared();
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                    mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                    if (cute::elect_one_sync())
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
                    __syncwarp();
                }
            }

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

                if constexpr (kIsMXSF) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                        utccp_required_smem_warp_transpose(smem_ptr);
                    }
                    cutlass::arch::fence_view_async_shared();
                }

                if (cute::elect_one_sync()) {
                    if constexpr (kIsMXSF) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
                        }
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                        CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

                        smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

                        if constexpr (kIsMXSF) {
                            DG_STATIC_ASSERT((not kIsFP4 and kHeadDim == 32) or kHeadDim == 64 or kHeadDim == 128, "Invalid head dim");

                            constexpr uint32_t kPackFactor = kIsFP4 ? 2 : 1;
                            constexpr uint32_t kQKSwizzleMode = kHeadDim / kPackFactor;

                            using mma_op_t = cute::conditional_t<kIsFP4, ptx::SM100_MMA_MXF4_SS, ptx::SM100_MMA_MXF8F6F4_SS>;
                            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<qk_dtype_t, qk_dtype_t, float, cutlass::float_ue8m0_t,
                                                                                       UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                            #pragma unroll
                            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                                auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * kPackFactor, k * kPackFactor);
                                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(
                                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(
                                    smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                                mma_op_t::fma(
                                    a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                                    kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
                            }
                        } else {
                            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                                            UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                            auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                            #pragma unroll
                            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                    smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                                ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, k, runtime_instr_desc);
                            }
                        }

                        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                     ::"r"(cute::cast_smem_ptr_to_uint(&smem.full_tmem_barriers[tmem_stage_idx])));
                    }
                }
                __syncwarp();
                if constexpr (kIsMXSF)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_kv_barriers[kv_stage_idx]));
            }
            smem.empty_q_barriers[q_stage_idx].arrive();
        }
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_warpgroup_idx = warpgroup_idx;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;
        DG_STATIC_ASSERT(kNumMathWarpGroups <= kNumTmemStages, "Math warp groups exceed TMEM stages");
        tmem_pipeline.advance(math_warpgroup_idx);

        constexpr bool kIsReduceBF16 = not cute::is_same_v<reduce_dtype_t, float>;
        DG_STATIC_ASSERT(not kIsReduceBF16 or kNumHeads % 2 == 0, "bf16 weights need even kNumHeads");
        DG_STATIC_ASSERT(kNumHeads == 4 or kNumHeads == 8 or kNumHeads == 16 or kNumHeads == 32 or kNumHeads == 64,
                         "Unsupported TMEM load size");
        using weights_reg_dtype_t = cute::conditional_t<kIsReduceBF16, nv_bfloat162, float>;
        constexpr uint32_t kNumWeightsRegPerToken = kIsReduceBF16 ? (kNumHeads / 2) : kNumHeads;
        weights_reg_dtype_t weights[BLOCK_Q][kNumWeightsRegPerToken];
        float accum[kNumHeads];

        auto tmem_load_no_fence = [](auto num_elems_t, const uint32_t& addr, float* load_dst) {
            constexpr uint32_t N = decltype(num_elems_t)::value;
            using Loader = cute::conditional_t<N == 2,  cute::SM100_TMEM_LOAD_32dp32b2x,
                           cute::conditional_t<N == 4,  cute::SM100_TMEM_LOAD_32dp32b4x,
                           cute::conditional_t<N == 8,  cute::SM100_TMEM_LOAD_32dp32b8x,
                           cute::conditional_t<N == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                           cute::conditional_t<N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                                                        cute::SM100_TMEM_LOAD_32dp32b64x>>>>>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(addr, reinterpret_cast<uint32_t*>(load_dst)[Is]...);
            }(cute::make_index_sequence<N>{});
        };

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            const auto process_q_block = [&](auto num_valid_tokens_t) {
                constexpr uint32_t kNumValidTokens = decltype(num_valid_tokens_t)::value;

                #pragma unroll
                for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                    const auto smem_weights_row = smem.smem_weights[q_stage_idx] + i * kNumHeads;
                    if constexpr (kIsReduceBF16) {
                        // Load two bf16 weights at a time as one packed shared u32
                        const auto packed_row = reinterpret_cast<const uint32_t*>(smem_weights_row);
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads / 2; ++ j) {
                            const auto packed = ptx::ld_shared(packed_row + j);
                            weights[i][j] = ptx::exchange(*reinterpret_cast<const nv_bfloat162*>(&packed), 0);
                        }
                    } else {
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads; ++ j)
                            weights[i][j] = ptx::ld_shared(smem_weights_row + j);
                    }
                }

                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                    auto kv_offset = scheduler.get_logits_col(kv_base, kv_split_idx, math_thread_idx);

                    // FP8 consumes the KV pipeline directly to read per-KV scales;
                    // MXFP4 / MXFP8 folds its scale into the MX SF UMMA, no extra scale
                    reduce_dtype_t scale_kv = static_cast<reduce_dtype_t>(0.0f);
                    uint32_t kv_stage_idx = 0;
                    if constexpr (not kIsMXSF) {
                        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx_local, kv_phase);
                        kv_stage_idx = kv_stage_idx_local;
                        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
                        scale_kv = static_cast<reduce_dtype_t>(ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + math_thread_idx));
                    }

                    CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                    smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                    ptx::tcgen05_after_thread_sync();

                    // Release KV smem only after UMMA commits TMEM; earlier release races TMA overwrite
                    if constexpr (not kIsMXSF)
                        smem.empty_kv_barriers[kv_stage_idx].arrive();

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                        if constexpr (kNumHeads == 8) {
                            tmem_load_no_fence(cute::Int<kNumHeads>{}, tmem_addr, accum);
                            cutlass::arch::fence_view_async_tmem_load();
                        } else if constexpr (kNumHeads == 16) {
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr, accum);
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2, accum + kNumHeads / 2);
                            cutlass::arch::fence_view_async_tmem_load();
                        } else {
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr, accum);
                            cutlass::arch::fence_view_async_tmem_load();
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2, accum + kNumHeads / 2);
                            cutlass::arch::fence_view_async_tmem_load();
                        }

                        if (i == kNumValidTokens - 1) {
                            ptx::tcgen05_before_thread_sync();
                            smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                        }

                        reduce_dtype_t reduced;
                        if constexpr (kIsReduceBF16) {
                            auto sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                            auto sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                            const auto transform = [&](const uint32_t& j, const nv_bfloat162& sum) {
                                const auto a = ptx::cvt_relu_bf16x2_f32(make_float2(accum[j], accum[j + 1]));
                                const auto b = weights[i][j / 2];
                                return __hfma2(a, b, sum);
                            };

                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads; j += 4) {
                                sum_0 = transform(j, sum_0);
                                sum_1 = transform(j + 2, sum_1);
                            }

                            auto sum = __hadd2_rn(sum_0, sum_1);
                            reduced = __hadd_rn(sum.x, sum.y);
                        } else {
                            auto sum_0 = make_float2(0, 0);
                            auto sum_1 = make_float2(0, 0);
                            const auto transform = [&](const uint32_t& j, const float2& sum) {
                                auto a_0 = make_float2(accum[j], accum[j + 1]);
                                auto a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));
                                auto b = make_float2(weights[i][j], weights[i][j + 1]);
                                return __ffma2_rn(__fadd2_rn(a_0, a_1), b, sum);
                            };

                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads; j += 4) {
                                sum_0 = transform(j, sum_0);
                                sum_1 = transform(j + 2, sum_1);
                            }

                            auto sum = __fadd2_rn(sum_0, sum_1);
                            reduced = (sum.x + sum.y) / 2;
                        }
                        auto result = static_cast<logits_dtype_t>(kIsMXSF ? reduced : reduced * scale_kv);
                        // Independent R1 emitter; all TMA/MMA/FP32 reduction above is unchanged.
                        emit(scheduler.get_logits_row(q_block_idx, i), kv_offset, result);
                    }
                }
            };

            if constexpr (decltype(scheduler)::kHasPartialBlock)
                dispatch_num_block_tokens<BLOCK_Q>(scheduler.get_num_block_tokens(q_block_idx), process_q_block);
            else
                process_q_block(cute::Int<BLOCK_Q>{});

            smem.empty_q_barriers[q_stage_idx].arrive();
        }

        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}


} // namespace deep_gemm
