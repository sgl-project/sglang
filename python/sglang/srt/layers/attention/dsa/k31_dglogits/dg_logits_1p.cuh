// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Vendored from sgl-deep-gemm 0.2.3.post2 (deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh, and the paged scheduler of
// deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh; DeepGEMM, MIT License,
// Copyright (c) 2025 DeepSeek; the license text follows this note). Additions, marked "K3.1:" (the first two for
// the verify index logits launched with PDL):
//  - the two TMA producer warps execute fence.proxy.async.global after the grid-dependency wait: the index query, head gates
//    and new index-K rows they read with TMA (async proxy) were written by the preceding kernels with plain stores (generic proxy);
//  - thread 0 triggers the dependent launch right after the wait, so the next kernel (top-k, which itself waits for this
//    grid to complete before reading the logits) is scheduled while the logits run. A fence.acq_rel.gpu precedes the trigger
//    in the same asm block: the top-k reads its row lengths before its own wait, so the trigger has to release the stores
//    this grid's wait observed (the row lengths' writer completed before that wait returned). Only a CTA's first trigger has
//    an effect, so one thread suffices; no thread stores to global memory between the CTA barrier before the wait and the
//    trigger. A __syncwarp then reconverges warp 0 before its register reconfiguration.
//  - qk_dtype_t == cutlass::detail::float_e2m1_unpacksmem_t selects an FP8 (e4m3) query against MXFP4
//    index-K (the indexer-cache-FP4 form of the verify logits): the index-K TMA map is the 4-bit padded type (16U4_ALIGN16B,
//    .b4x16_p64), so 64 B rows land as 128 B FP8-layout smem rows, and the block-scaled MMA takes A = e2m1 index-K, B = e4m3
//    query (MXF8F6F4, UMMA_K 32), as DeepGEMM's sm100_fp8_fp4_gemm_1d1d does for its FP4 operand; that copy's transaction
//    count is the packed global bytes (the same rule as sm100_fp8_fp4_gemm_1d1d). MX scales only. FP8 and FP4 instantiations
//    compile unchanged (their smem / MMA types equal qk_dtype_t).
//  - dg_logits_1p.cuh = dg_logits.cuh + a ONE-PASS paged entry sm100_paged_mqa_logits_1p: one Q-block holds
//    all kTokensPerRequest (5..7) query rows of a request (UMMA_N = rows x heads <= 224), so each index-K split is streamed and MMA'd once
//    per request instead of once per 4-row Q-block; scheduler SM100PagedMQALogitsScheduler1P = DeepGEMM's paged scheduler with BLOCK_Q
//    a template argument. In the shared core: TMEM accumulator stages = 3 when 3 x UMMA_N + scale columns fit the 512 columns, else 2
//    (one per math warpgroup; the 4-row instantiations keep 3), and the math warps read each token's FP32 gates from the Q stage per
//    token (FP32 reduce only) instead of caching BLOCK_Q x heads gates in registers. Per (row, token) arithmetic is unchanged, so the
//    logits equal the 4-row kernel's bit for bit. Synchronization: the same barriers / PDL / proxy fences; the Q stage is still released
//    (fence_view_async_shared + empty-Q arrive) only after the last gate read of the Q-block.
//  - the 1-pass math warps handle query rows in pairs: two rows' TMEM loads, one wait::ld, then the two rows'
//    unchanged reductions / stores (bit-identical).
//  - rows per wait::ld per instantiation: the FP8-query form at 6 rows all 6, FP8 at 7 rows groups of 4, else pairs.
// Everything else is unchanged; the common DeepGEMM headers come from dg_include/ (unmodified sgl-deep-gemm 0.2.3.post2
// copies), CuTe and CUTLASS from the installed package.
//
// MIT License (DeepGEMM, applies to the vendored DeepGEMM code in this file):
//
// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/clean_logits.cuh>
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

// Load heads using power-of-two TMEM chunks.
template <uint32_t kNumHeads, uint32_t kNumLoaded = 0>
CUTLASS_DEVICE void load_tmem_heads_decomposed(const uint32_t& tmem_col, float* accum) {
    constexpr uint32_t kNumRemaining = kNumHeads - kNumLoaded;
    if constexpr (kNumRemaining > 0) {
        constexpr uint32_t kChunk = kNumRemaining >= 64 ? 64
                                  : kNumRemaining >= 32 ? 32
                                  : kNumRemaining >= 16 ? 16
                                  : kNumRemaining >= 8  ? 8
                                  : 4;
        ptx::tmem_load_32dp32b<kChunk>(tmem_col + kNumLoaded, reinterpret_cast<uint32_t*>(accum + kNumLoaded));
        load_tmem_heads_decomposed<kNumHeads, kNumLoaded + kChunk>(tmem_col, accum);
    }
}

// Keep these helpers outside the core so existing local lambda types retain
// their identities and excluded templates preserve their generated code.
template <typename scheduler_t, bool kIsMXSF, uint32_t kNumHeads, uint32_t kHeadDim,
          typename logits_dtype_t, typename reduce_dtype_t>
CUTLASS_HOST_DEVICE constexpr bool use_paged_wide_tmem_load() {
    if constexpr (scheduler_t::kIsPaged)
        return scheduler_t::kIsContextLens2DLayout and not scheduler_t::kIsVarlenLayout and
            scheduler_t::kTokensPerRequest == 6 and
            (scheduler_t::kPageKV == 64 or scheduler_t::kPageKV == 128) and
            not kIsMXSF and kNumHeads == 64 and kHeadDim == 32 and
            not cute::is_same_v<reduce_dtype_t, float> and
            cute::is_same_v<logits_dtype_t, cutlass::bfloat16_t>;
    return false;
}

// Keep long-lived math uses local to their register-reconfigured branch.
// Carrying the warp index across setmaxnreg spills with the wide TMEM load.
template <bool kPagedWideLoad, typename WarpIdx>
CUTLASS_DEVICE WarpIdx get_mqa_math_warp_idx(const WarpIdx& warp_idx) {
    if constexpr (kPagedWideLoad) {
        uint32_t thread_idx;
        asm volatile("mov.u32 %0, %%tid.x;" : "=r"(thread_idx));
        return static_cast<WarpIdx>(__shfl_sync(0xffffffff, thread_idx / 32, 0));
    }
    return warp_idx;
}

// Shared device core parameterized by dtype and scheduler geometry/addressing
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF, bool kIsCompressedLogits, bool kCleanLogits,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t, typename MakeScheduler,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_DEVICE void sm100_mqa_logits_core_impl(const uint32_t logits_stride,
                                               logits_dtype_t* logits,
                                               const cute::TmaDescriptor& tensor_map_q,
                                               const cute::TmaDescriptor& tensor_map_sf_q,
                                               const cute::TmaDescriptor& tensor_map_kv,
                                               const cute::TmaDescriptor& tensor_map_sf_kv,
                                               const cute::TmaDescriptor& tensor_map_weights,
                                               const MakeScheduler& make_scheduler) {
    constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;
    // K3.1: FP8 query x MXFP4 index-K (see the header note); smem and the query keep FP8 types
    constexpr bool kIsFP8QFP4KV = cute::is_same_v<qk_dtype_t, cutlass::detail::float_e2m1_unpacksmem_t>;
    using smem_qk_dtype_t = cute::conditional_t<kIsFP8QFP4KV, cutlass::float_e4m3_t, qk_dtype_t>;
    DG_STATIC_ASSERT(not kIsFP8QFP4KV or kIsMXSF, "FP8 query x MXFP4 index-K needs MX scales");

    // For spilling cleaner/non-power-of-two families, keep contiguous CTA-ID reads
    // inside each register-reconfigured branch: carrying the index across setmaxnreg
    // makes NVCC spill it. Preserve other families' existing scheduler expression.
    const auto sm_idx = blockIdx.x;
    constexpr bool kReuseSchedulerIndex = decltype(make_scheduler(0))::kIsPaged or
        not (kCleanLogits or kNumHeads == 12 or kNumHeads == 20);
    using scheduler_t = decltype(make_scheduler(0));
    constexpr bool kPagedWideLoad = use_paged_wide_tmem_load<scheduler_t, kIsMXSF,
        kNumHeads, kHeadDim, logits_dtype_t, reduce_dtype_t>();
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    const auto kNegInf = -cute::numeric_limits<logits_dtype_t>::infinity();

    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t BLOCK_QH = BLOCK_Q * kNumHeads;
    static constexpr uint32_t UMMA_N = math::constexpr_align(BLOCK_QH, 8u);
    static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    static constexpr uint32_t kNumSFQ  = kIsMXSF ? math::constexpr_align(UMMA_N, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kNumSFKV = kIsMXSF ? math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems) : 0;
    // K3.1 1-pass: three accumulator stages when they fit the 512 TMEM columns, else two (one per math warpgroup)
    static constexpr uint32_t kNumTmemStages = (UMMA_N * 3 + kNumSFQ / 32 + kNumSFKV / 32 <= 512) ? 3 : 2;
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_QH * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kNumQKBytesPerToken;
    // K3.1: bytes the index-K TMA copies count on the barrier (packed 4-bit for the padded FP4 map)
    static constexpr uint32_t TMA_KV_SIZE_PER_STAGE = kIsFP8QFP4KV ? SPLIT_KV * kHeadDim / 2 : SMEM_KV_SIZE_PER_STAGE;
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = kIsMXSF ? (BLOCK_QH * sizeof(int)) : 0;
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kIsMXSF ? (kNumSFKV * sizeof(int)) : (SPLIT_KV * sizeof(float));
    static constexpr uint32_t kNumWeightBytesPerRow = math::constexpr_align(
        kNumHeads * static_cast<uint32_t>(sizeof(reduce_dtype_t)), 16u);
    static constexpr uint32_t kNumWeightElementsPerRow =
        kNumWeightBytesPerRow / static_cast<uint32_t>(sizeof(reduce_dtype_t));
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumWeightBytesPerRow;

    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");
    DG_STATIC_ASSERT(not (kIsCompressedLogits and kCleanLogits), "Compressed logits cannot be cleaned in-kernel");

    using SharedStorage = layout::MQALogitsSharedStorage<kNumHeads, kHeadDim, kIsMXSF, BLOCK_Q, SPLIT_KV,
                                                         kNumQStages, kNumKVStages, kNumTmemStages, smem_qk_dtype_t, reduce_dtype_t>;
    extern __shared__ __align__(SharedStorage::kSwizzleAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumTmemStages;
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

    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;

    // Only these cleaner families need 64 producer registers to avoid spills.
    // Rebalance math to 216 within the existing CTA pool; keep other tiles at 56/224.
    constexpr bool kMoreSpecializedRegisters = kCleanLogits and
        (kNumHeads == 12 or kNumHeads == 20 or
         (kIsMXSF and kNumHeads == 64 and not cute::is_same_v<reduce_dtype_t, float>));
    constexpr uint32_t kNumSpecializedRegisters = kMoreSpecializedRegisters ? 64 : 56;
    // Three math warpgroups split the CTA pool of 512 registers per thread-slot four ways
    constexpr uint32_t kNumEntryRegisters = (512 / (kNumMathWarpGroups + 1) / 8) * 8;
    constexpr uint32_t kNumMathRegisters = kNumMathWarpGroups == 2 ?
        (kMoreSpecializedRegisters ? 216 : 224) :
        ((kNumEntryRegisters * (kNumMathWarpGroups + 1) - kNumSpecializedRegisters) / kNumMathWarpGroups / 8) * 8;
    DG_STATIC_ASSERT(kNumMathRegisters * kNumMathWarpGroups + kNumSpecializedRegisters <= 512,
                     "Register reconfiguration exceeds the CTA pool");

    cudaGridDependencySynchronize();
    // K3.1: release the dependent launch at entry; dependents still wait for this grid's completion before reading the logits,
    // but read their own inputs (the top-k's row lengths) before that wait. Thread 0's fence orders what its wait made visible
    // before the trigger, so those early reads see it. Only the CTA's first trigger counts; no other thread triggers.
    if (threadIdx.x == 0) {
        asm volatile("fence.acq_rel.gpu;\n\tgriddepcontrol.launch_dependents;" ::: "memory");
    }
    __syncwarp();

    if (warp_idx == kSpecWarpStart) {
        // K3.1: order this warp's TMA (async proxy) reads after the predecessors' generic-proxy stores.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (cute::elect_one_sync()) {
            auto scheduler = make_scheduler(kReuseSchedulerIndex ? sm_idx : uint32_t(blockIdx.x));
            // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

                const uint32_t q_token_base = scheduler.get_q_tma_token_base(q_block_idx);
                tma::copy<kHeadDim, BLOCK_Q * kNumHeads, 0>(
                    &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                    smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
                if constexpr (kIsMXSF)
                    tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx], smem.smem_sf_q[q_stage_idx], 0, q_token_base);
                tma::copy<kNumWeightElementsPerRow, BLOCK_Q, 0>(&tensor_map_weights, &smem.full_q_barriers[q_stage_idx], smem.smem_weights[q_stage_idx], 0, q_token_base);
                smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        // K3.1: order this warp's TMA (async proxy) reads after the predecessors' generic-proxy stores.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        auto scheduler = make_scheduler(kReuseSchedulerIndex ? sm_idx : uint32_t(blockIdx.x));
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
                            tma::copy<kHeadDim, kPageKV, 0, smem_qk_dtype_t, true>(
                                &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                                smem.smem_kv[kv_stage_idx] + page_idx * kPageKV * kNumQKBytesPerToken,
                                0, 0, 1, page_coords[page_idx]);
                            tma::copy<kPageKV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                                     smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                                                     0, page_coords[page_idx]);
                        }
                        smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(TMA_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                    }
                    __syncwarp();
                } else if (cute::elect_one_sync()) {
                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);

                    const uint32_t kv_tma_offset = scheduler.get_kv_tma_offset(kv_base, kv_split_idx);
                    // A TMA box spans at most 256 rows
                    constexpr uint32_t kKVTmaTokens = SPLIT_KV % 256 == 0 ? 256 : 128;
                    #pragma unroll
                    for (uint32_t i = 0; i < SPLIT_KV / kKVTmaTokens; ++ i) {
                        tma::copy<kHeadDim, kKVTmaTokens, 0>(
                            &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                            smem.smem_kv[kv_stage_idx] + i * kKVTmaTokens * kNumQKBytesPerToken,
                            0, kv_tma_offset + i * kKVTmaTokens);
                        tma::copy<kKVTmaTokens, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                                      smem.smem_sf_kv[kv_stage_idx] + i * kKVTmaTokens,
                                                      kv_tma_offset + i * kKVTmaTokens, 0);
                    }
                    smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(TMA_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
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

        auto scheduler = make_scheduler(kReuseSchedulerIndex ? sm_idx : uint32_t(blockIdx.x));
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

                            constexpr uint32_t kPackFactor = get_smem_pack_factor<smem_qk_dtype_t>();
                            constexpr uint32_t kQKSwizzleMode = kHeadDim / kPackFactor;

                            using mma_op_t = cute::conditional_t<kIsFP4, ptx::SM100_MMA_MXF4_SS, ptx::SM100_MMA_MXF8F6F4_SS>;
                            // K3.1: A = index-K (qk_dtype_t: e2m1 padded for kIsFP8QFP4KV), B = query (smem_qk_dtype_t)
                            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<qk_dtype_t, smem_qk_dtype_t, float, cutlass::float_ue8m0_t,
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

                        ptx::umma_arrive_no_elect(smem.full_tmem_barriers[tmem_stage_idx]);
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

        if constexpr (kCleanLogits) {
            const auto cleaner = epilogue::LogitsCleaner<logits_dtype_t>(lane_idx);

            // Cleaning always runs grid-stride
            auto scheduler = make_scheduler(kReuseSchedulerIndex ? sm_idx : uint32_t(blockIdx.x)).make_cleaner(blockIdx.x);
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                const auto coverage_end = cute::min(kv_base + num_kv_splits * SPLIT_KV, logits_stride);
                #pragma unroll 1
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    const auto row = logits + scheduler.get_logits_row(q_block_idx, i) * static_cast<uint64_t>(logits_stride);
                    cleaner.fill_row(row, 0, kv_base);
                    cleaner.fill_row(row, coverage_end, logits_stride);
                }
            }
        }
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        constexpr bool kLoadSeqBounds = kIsCompressedLogits or kCleanLogits;
        uint32_t seq_k_start[kLoadSeqBounds ? BLOCK_Q : 1];
        uint32_t seq_k_end[kLoadSeqBounds ? BLOCK_Q : 1];
        const auto local_math_warp_idx = get_mqa_math_warp_idx<kPagedWideLoad>(warp_idx);
        const auto math_warpgroup_idx = local_math_warp_idx / 4;
        const auto math_thread_idx = local_math_warp_idx * 32 + lane_idx;
        DG_STATIC_ASSERT(kNumMathWarpGroups <= kNumTmemStages, "Math warp groups exceed TMEM stages");
        tmem_pipeline.advance(math_warpgroup_idx);

        constexpr bool kIsReduceBF16 = not cute::is_same_v<reduce_dtype_t, float>;
        constexpr bool kFenceWeightsEarly = kIsMXSF and not kIsFP4 and kIsCompressedLogits and
            kNumHeads == 32 and kHeadDim == 64 and kIsReduceBF16;
        DG_STATIC_ASSERT(kNumHeads % 4 == 0, "Head count must be a multiple of 4");
        DG_STATIC_ASSERT(8 <= UMMA_N and UMMA_N <= 256, "Invalid UMMA_N for MMA");
        using weights_reg_dtype_t = cute::conditional_t<kIsReduceBF16, nv_bfloat162, float>;
        constexpr uint32_t kNumWeightsRegPerToken = kIsReduceBF16 ? (kNumHeads / 2) : kNumHeads;
        // K3.1 1-pass: one token's FP32 gates, read from the Q stage per token (BLOCK_Q x heads registers would spill)
        DG_STATIC_ASSERT(not kIsReduceBF16, "1-pass core: FP32 gates only");
        weights_reg_dtype_t weights_row[kNumWeightsRegPerToken];
        float accum[kNumHeads];

        auto scheduler = make_scheduler(kReuseSchedulerIndex ? sm_idx : uint32_t(blockIdx.x));
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.template next_q_block<kLoadSeqBounds>(
                q_block_idx, kv_base, num_kv_splits, seq_k_start, seq_k_end)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            const auto process_q_block = [&](auto num_valid_tokens_t) {
                constexpr uint32_t kNumValidTokens = decltype(num_valid_tokens_t)::value;

                // MX math performs no more generic shared data loads after these
                // Q-stage weights. Keep the proxy fence after their reads and before
                // the unchanged empty-Q arrival, outside the output-store critical path.
                if constexpr (kFenceWeightsEarly)
                    cutlass::arch::fence_view_async_shared();

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
                    if constexpr (not kIsMXSF) {
                        cutlass::arch::fence_view_async_shared();
                        smem.empty_kv_barriers[kv_stage_idx].arrive();
                    }

                    // K3.1: rows in PAIRS - both rows' TMEM accumulators (two 16-column loads each) are issued before ONE
                    // tcgen05.wait::ld, then each row's unchanged gate load, head reduction and store; per-(row, token) arithmetic and order are those
                    // of DeepGEMM's per-row loop, so the logits are bit-identical. The stage is released (before_thread_sync + empty-TMEM arrive) after
                    // the wait that covers the stage's last loads, as before (same release form as DeepGEMM's kPagedWideLoad path: loads, one wait,
                    // then the arrive).
                    DG_STATIC_ASSERT(kNumHeads == 32 and not kIsReduceBF16, "1-pass core: 32 index heads (two 16-column TMEM loads per row), FP32 reduce");
                    const auto finish_row = [&](const uint32_t i, const float* acc) {
                        float wr[kNumHeads];
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads; j += 4) {
                            const auto w4 = ptx::ld_shared(reinterpret_cast<const float4*>(
                                smem.smem_weights[q_stage_idx] + i * kNumWeightElementsPerRow + j));
                            wr[j] = w4.x;
                            wr[j + 1] = w4.y;
                            wr[j + 2] = w4.z;
                            wr[j + 3] = w4.w;
                        }
                        auto sum_0 = make_float2(0, 0);
                        auto sum_1 = make_float2(0, 0);
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads; j += 4) {
                            sum_0 = __ffma2_rn(__fadd2_rn(make_float2(acc[j], acc[j + 1]), make_float2(fabsf(acc[j]), fabsf(acc[j + 1]))),
                                               make_float2(wr[j], wr[j + 1]), sum_0);
                            sum_1 = __ffma2_rn(__fadd2_rn(make_float2(acc[j + 2], acc[j + 3]), make_float2(fabsf(acc[j + 2]), fabsf(acc[j + 3]))),
                                               make_float2(wr[j + 2], wr[j + 3]), sum_1);
                        }
                        const auto sum = __fadd2_rn(sum_0, sum_1);
                        const reduce_dtype_t reduced = (sum.x + sum.y) / 2;
                        auto result = static_cast<logits_dtype_t>(kIsMXSF ? reduced : reduced * scale_kv);
                        const auto q_offset = scheduler.get_logits_row(q_block_idx, i) * static_cast<uint64_t>(logits_stride);
                        if constexpr (kIsCompressedLogits) {
                            const uint32_t rel_kv = kv_offset - seq_k_start[i];
                            const uint32_t len = seq_k_end[i] - seq_k_start[i];
                            if (rel_kv < len)
                                logits[q_offset + rel_kv] = result;
                        } else if constexpr (kCleanLogits) {
                            const uint32_t rel_kv = kv_offset - seq_k_start[i];
                            const uint32_t len = seq_k_end[i] - seq_k_start[i];
                            logits[q_offset + kv_offset] = rel_kv < len ? result : kNegInf;
                        } else {
                            logits[q_offset + kv_offset] = result;
                        }
                    };
                    // K3.1: rows per wait::ld chosen per instantiation (same bits): the FP8-query form at 6 rows
                    // loads all 6 rows before one wait (its 2-stage TMEM ring is the bound; the stage frees after one wait), FP8 at 7 rows loads groups of 4,
                    // every other form keeps the pairs below.
                    constexpr uint32_t kRowGroup = (kIsMXSF and not kIsFP4 and BLOCK_Q == 6) ? 6 : ((not kIsMXSF and BLOCK_Q == 7) ? 4 : 2);
                    if constexpr (kRowGroup != 2) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumValidTokens; i += kRowGroup) {
                            float acc[kRowGroup][kNumHeads];
                            const uint32_t ta = tmem_stage_idx * UMMA_N + i * kNumHeads;
                            #pragma unroll
                            for (uint32_t g = 0; g < kRowGroup; ++ g) {
                                if (i + g < kNumValidTokens) {
                                    ptx::tmem_load_32dp32b<kNumHeads / 2>(ta + g * kNumHeads, reinterpret_cast<uint32_t*>(acc[g]));
                                    ptx::tmem_load_32dp32b<kNumHeads / 2>(ta + g * kNumHeads + kNumHeads / 2, reinterpret_cast<uint32_t*>(acc[g] + kNumHeads / 2));
                                }
                            }
                            cutlass::arch::fence_view_async_tmem_load();
                            if (i + kRowGroup >= kNumValidTokens) {
                                ptx::tcgen05_before_thread_sync();
                                smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                            }
                            #pragma unroll
                            for (uint32_t g = 0; g < kRowGroup; ++ g) {
                                if (i + g < kNumValidTokens)
                                    finish_row(i + g, acc[g]);
                            }
                        }
                    } else {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumValidTokens; i += 2) {
                        float acc_a[kNumHeads], acc_b[kNumHeads];
                        const uint32_t ta = tmem_stage_idx * UMMA_N + i * kNumHeads;
                        ptx::tmem_load_32dp32b<kNumHeads / 2>(ta, reinterpret_cast<uint32_t*>(acc_a));
                        ptx::tmem_load_32dp32b<kNumHeads / 2>(ta + kNumHeads / 2, reinterpret_cast<uint32_t*>(acc_a + kNumHeads / 2));
                        if (i + 1 < kNumValidTokens) {
                            ptx::tmem_load_32dp32b<kNumHeads / 2>(ta + kNumHeads, reinterpret_cast<uint32_t*>(acc_b));
                            ptx::tmem_load_32dp32b<kNumHeads / 2>(ta + kNumHeads + kNumHeads / 2, reinterpret_cast<uint32_t*>(acc_b + kNumHeads / 2));
                        }
                        cutlass::arch::fence_view_async_tmem_load();
                        if (i + 2 >= kNumValidTokens) {
                            ptx::tcgen05_before_thread_sync();
                            smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                        }
                        finish_row(i, acc_a);
                        if (i + 1 < kNumValidTokens)
                            finish_row(i + 1, acc_b);
                    }
                    }
                }
            };

            if constexpr (decltype(scheduler)::kHasPartialBlock)
                dispatch_num_block_tokens<BLOCK_Q>(scheduler.get_num_block_tokens(q_block_idx), process_q_block);
            else
                process_q_block(cute::Int<BLOCK_Q>{});

            if constexpr (not kFenceWeightsEarly)
                cutlass::arch::fence_view_async_shared();
            smem.empty_q_barriers[q_stage_idx].arrive();
        }

        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (get_mqa_math_warp_idx<kPagedWideLoad>(warp_idx) == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

// Unified contiguous-KV entry for FP8 / MXFP4 / MXFP8.
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF, bool kIsCompressedLogits, bool kCleanLogits,
          bool kUseSchedule,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_mqa_logits(const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
                      const uint32_t logits_stride,
                      const uint32_t* cu_seq_len_k_start,
                      const uint32_t* cu_seq_len_k_end,
                      const uint32_t* schedule_meta,
                      logits_dtype_t* logits,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto make_scheduler = [&](const uint32_t& sm_idx) {
        return sched::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs, kUseSchedule>(
            sm_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, schedule_meta);
    };

    sm100_mqa_logits_core_impl<kNumHeads, kHeadDim, kIsMXSF, kIsCompressedLogits, kCleanLogits,
                               BLOCK_Q, SPLIT_KV,
                               kNumQStages, kNumKVStages, kNumSMs,
                               kNumSpecializedThreads, kNumMathThreads, qk_dtype_t, logits_dtype_t,
                               reduce_dtype_t, decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

// Long H8/BF16 reduction benefits from the original fully unrolled span scan.
// A separate entry preserves the default kernel symbol and codegen for other modes.
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF, bool kIsCompressedLogits, bool kCleanLogits,
          bool kUseSchedule,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_mqa_logits_unrolled(const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
                      const uint32_t logits_stride,
                      const uint32_t* cu_seq_len_k_start,
                      const uint32_t* cu_seq_len_k_end,
                      const uint32_t* schedule_meta,
                      logits_dtype_t* logits,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto make_scheduler = [&](const uint32_t& sm_idx) {
        constexpr bool kRematerializeSpan = not kUseSchedule and kNumHeads == 8 and kHeadDim == 128 and
            kIsMXSF and not kIsCompressedLogits and not kCleanLogits and BLOCK_Q == 16 and SPLIT_KV == 256 and
            kNumQStages == 3 and kNumKVStages == 5 and kNumSpecializedThreads == 128 and kNumMathThreads == 256 and
            kNumMathWarpGroups == 2 and cute::is_same_v<qk_dtype_t, cutlass::float_e4m3_t> and
            cute::is_same_v<logits_dtype_t, cutlass::bfloat16_t> and cute::is_same_v<reduce_dtype_t, __nv_bfloat16>;
        return sched::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs, kUseSchedule, 16, kRematerializeSpan>(
            sm_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, schedule_meta);
    };

    sm100_mqa_logits_core_impl<kNumHeads, kHeadDim, kIsMXSF, kIsCompressedLogits, kCleanLogits,
                               BLOCK_Q, SPLIT_KV,
                               kNumQStages, kNumKVStages, kNumSMs,
                               kNumSpecializedThreads, kNumMathThreads, qk_dtype_t, logits_dtype_t,
                               reduce_dtype_t, decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

// K3.1 1-pass: DeepGEMM's SM100PagedMQALogitsScheduler (scheduler/sm100_paged_mqa_logits.cuh, 0.2.3.post2) with BLOCK_Q a template
// argument instead of 128 / kNumHeads; every other line is DeepGEMM's. With BLOCK_Q == kNextN a request is one Q-block, so next_q_block
// emits (request, all remaining splits) and each index-K split is loaded once per request.
template <uint32_t kNextN, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumHeads, uint32_t SPLIT_KV, uint32_t PAGE_KV, uint32_t kSplitsPerChunk, uint32_t kBlockQ>
struct SM100PagedMQALogitsScheduler1P : sched::SM100IndicesStorage<kIsVarlen> {
    static constexpr bool kIsPaged = true;
    static constexpr bool kIsVarlenLayout = kIsVarlen;
    static constexpr bool kIsContextLens2DLayout = kIsContextLens2D;
    static constexpr uint32_t kTokensPerRequest = kNextN;
    static constexpr bool kHasPartialBlock = true;
    static constexpr uint32_t kPageKV = PAGE_KV;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    static constexpr uint32_t BLOCK_Q = kBlockQ;

    using Info = sched::RequestInfo<kNextN, kIsContextLens2D, kIsVarlen, BLOCK_Q, SPLIT_KV, PAGE_KV>;

    const uint32_t* context_lens;
    const uint32_t* block_table;
    uint32_t block_table_stride;
    uint32_t num_q_tokens_total;

    Info cur;
    uint32_t cur_kv_split_base;
    uint32_t cur_q_block_in_request;
    uint32_t end_q_token_idx, end_kv_split_idx;
    bool done;

    uint32_t cur_block_table_row;
    uint32_t cur_q_block_token_base;
    uint32_t cur_num_block_tokens;
    uint32_t cur_request_num_kv_pages;

    CUTLASS_DEVICE const uint32_t* get_indices() const {
        if constexpr (kIsVarlen)
            return this->indices;
        return nullptr;
    }

    CUTLASS_DEVICE SM100PagedMQALogitsScheduler1P(const uint32_t& sm_idx,
                                                  const uint32_t* context_lens,
                                                  const uint32_t* schedule_meta,
                                                  const uint32_t* indices,
                                                  const uint32_t* block_table,
                                                  const uint32_t& block_table_stride,
                                                  const uint32_t& num_q_tokens_total) {
        this->context_lens = context_lens;
        this->block_table = block_table;
        this->block_table_stride = block_table_stride;
        this->num_q_tokens_total = num_q_tokens_total;
        if constexpr (kIsVarlen)
            this->indices = indices;

        const auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
        const auto end = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
        end_q_token_idx = end.x;
        end_kv_split_idx = end.y;

        cur_kv_split_base = start.y;
        cur_q_block_in_request = 0;
        done = (start.x >= num_q_tokens_total) or
               (start.x == end_q_token_idx and start.y >= end_kv_split_idx);
        if (not done)
            cur = Info::from_q_token(start.x, num_q_tokens_total, context_lens, get_indices());

        cur_block_table_row = 0;
        cur_q_block_token_base = 0;
        cur_num_block_tokens = 1;
        cur_request_num_kv_pages = 0;
    }

    CUTLASS_DEVICE uint32_t get_cur_kv_split_upper() const {
        return (cur.q_token_start == end_q_token_idx) ? end_kv_split_idx : cur.num_kv_splits;
    }

    template <bool kIsCompressedLogits = false>
    CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_split_base, uint32_t& num_kv_splits,
                                     uint32_t* = nullptr, uint32_t* = nullptr) {
        q_block_idx = 0;
        if (done)
            return false;

        const uint32_t upper = get_cur_kv_split_upper();
        cur_block_table_row = cur.get_block_table_row();
        uint32_t q_block_token_offset, q_block_num_tokens;
        cur.get_q_block_span(cur_q_block_in_request, q_block_token_offset, q_block_num_tokens);
        cur_q_block_token_base = cur.q_token_start + q_block_token_offset;
        cur_num_block_tokens = q_block_num_tokens;
        cur_request_num_kv_pages = cur.num_kv_pages;
        kv_split_base = cur_kv_split_base;
        const uint32_t remaining = upper - cur_kv_split_base;
        num_kv_splits = (cur.num_q_blocks == 1) ? remaining
                                               : (remaining < kSplitsPerChunk ? remaining : kSplitsPerChunk);

        ++ cur_q_block_in_request;
        if (cur_q_block_in_request == cur.num_q_blocks) {
            cur_q_block_in_request = 0;
            cur_kv_split_base += num_kv_splits;
            if (cur_kv_split_base >= upper) {
                if (cur.q_token_start == end_q_token_idx) {
                    done = true;
                } else {
                    const uint32_t next_q_token = cur.q_token_start + cur.num_q_tokens;
                    cur_kv_split_base = 0;
                    if (next_q_token >= num_q_tokens_total)
                        done = true;
                    else {
                        cur = Info::from_q_token(next_q_token, num_q_tokens_total, context_lens, get_indices());
                        if (cur.q_token_start == end_q_token_idx and end_kv_split_idx == 0)
                            done = true;
                    }
                }
            }
        }
        return true;
    }

    CUTLASS_DEVICE uint32_t get_num_block_tokens(const uint32_t&) const {
        return cur_num_block_tokens;
    }

    CUTLASS_DEVICE uint32_t get_q_tma_token_base(const uint32_t&) const {
        return cur_q_block_token_base;
    }

    CUTLASS_DEVICE static uint32_t get_kv_tma_offset(const uint32_t& kv_split_base, const uint32_t& kv_split_idx) {
        return (kv_split_base + kv_split_idx) * SPLIT_KV;
    }

    CUTLASS_DEVICE uint32_t get_kv_page_coord_by_page_offset(const uint32_t& page_offset) const {
        if (page_offset >= cur_request_num_kv_pages)
            return 0;
        const auto block_table_offset = cur_block_table_row * static_cast<uint64_t>(block_table_stride);
        return block_table[block_table_offset + page_offset];
    }

    CUTLASS_DEVICE uint32_t get_logits_row(const uint32_t&, const uint32_t& token_idx) const {
        return cur_q_block_token_base + token_idx;
    }

    CUTLASS_DEVICE uint32_t get_logits_col(const uint32_t& kv_split_base,
                                           const uint32_t& kv_split_idx,
                                           const uint32_t& math_thread_idx) const {
        return (kv_split_base + kv_split_idx) * SPLIT_KV + math_thread_idx;
    }
};

// K3.1 1-pass paged entry: DeepGEMM's paged entry with BLOCK_Q = kTokensPerRequest (one Q-block per request) and the 1P scheduler.
template <uint32_t kTokensPerRequest, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t PAGE_KV,
          bool kIsMXSF, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV, uint32_t kSplitsPerChunk,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_paged_mqa_logits_1p(const uint32_t num_q_tokens_total,
                               const uint32_t logits_stride, const uint32_t block_table_stride,
                               const uint32_t* context_lens, logits_dtype_t* logits,
                               const uint32_t* block_table, const uint32_t* indices,
                               const uint32_t* schedule_meta,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    static constexpr uint32_t BLOCK_Q = kTokensPerRequest;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    DG_STATIC_ASSERT(SPLIT_KV == PAGE_KV * kNumPagesPerSplit, "Invalid split/page size");
    DG_STATIC_ASSERT(not kIsVarlen and BLOCK_Q * kNumHeads <= 256 and BLOCK_Q * kNumHeads > 128, "1-pass: 5..8 rows x 32 heads, non-varlen");

    const auto make_scheduler = [&](const uint32_t& sm_idx) {
        return SM100PagedMQALogitsScheduler1P<kTokensPerRequest, kIsContextLens2D, kIsVarlen,
                                              kNumHeads, SPLIT_KV, PAGE_KV, kSplitsPerChunk, BLOCK_Q>(
            sm_idx, context_lens, schedule_meta, indices,
            block_table, block_table_stride, num_q_tokens_total);
    };

    sm100_mqa_logits_core_impl<kNumHeads, kHeadDim, kIsMXSF, false, false,
                               BLOCK_Q, SPLIT_KV,
                               kNumQStages, kNumKVStages, 0,
                               kNumSpecializedThreads, kNumMathThreads, qk_dtype_t, logits_dtype_t,
                               reduce_dtype_t, decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

} // namespace deep_gemm
