#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;
using namespace deep_gemm::sm100;

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          bool kIsContextLens2D,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
__global__ __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_fp8_paged_mqa_logits(const uint32_t batch_size,
                                const uint64_t logits_stride, const uint64_t block_table_stride,
                                const uint32_t* context_lens, float* logits,
                                const uint32_t* block_table, const uint32_t* schedule_meta,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                                const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto& warpgroup_idx = warp_idx / 4;
    const auto& lane_idx = get_lane_idx();

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextN * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = SPLIT_KV * sizeof(float);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextN * kNumHeads * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q and KV data on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i);
    });
    constexpr auto smem_offset = SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages;
    auto smem_kv_scales = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SCALE_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });

    // Barriers and TMEM pointer on shared memory
    const auto barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers     = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
    auto full_kv_barriers    = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
    auto empty_kv_barriers   = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; });
    const auto umma_barrier_ptr = barrier_ptr + kNumQStages * 2 + kNumKVStages * 2;
    auto full_umma_barriers  = PatternVisitor([&](const uint32_t& i) { return umma_barrier_ptr + i; });
    auto empty_umma_barriers = PatternVisitor([&](const uint32_t& i) { return umma_barrier_ptr + kNumMathWarpGroups + i; });
    auto tmem_ptr_in_smem    = reinterpret_cast<uint32_t*>(umma_barrier_ptr + kNumMathWarpGroups * 2);

    constexpr uint32_t kNumTmemCols = kNextN * kNumHeads * kNumMathWarpGroups;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");
    const bool& is_math_warp = (warp_idx < kNumMathWarpGroups * 4);
    const bool& is_tma_load_warp = (warp_idx == kNumMathWarpGroups * 4);
    const bool& is_umma_warp = (warp_idx == kNumMathWarpGroups * 4 + 1);

    // Initialize barriers
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    if (is_umma_warp) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
                full_umma_barriers[i]->init(1);
                empty_umma_barriers[i]->init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

    // Scheduler
    constexpr uint32_t kNumBlocksPerSplit = SPLIT_KV / BLOCK_KV;
    auto scheduler = PagedMQALogitsScheduler<kNextN, kIsContextLens2D, BLOCK_KV, kNumBlocksPerSplit>(batch_size, blockIdx.x, context_lens, schedule_meta);
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumBlocksPerSplit, "Invalid `SPLIT_KV`");

    // Q and KV pipeline
    const auto& get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1}; // Q pipeline stage and phase
    };
    const auto& get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1}; // KV pipeline stage and phase
    };
    uint32_t q_iter_idx = 0, kv_iter_idx = 0;

    // UMMA settings
    // Construct instruction with layout D
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N = kNextN * kNumHeads;
    DG_STATIC_ASSERT(SPLIT_KV == UMMA_M * kNumMathWarpGroups, "Invalid `SPLIT_KV`");

    if (is_tma_load_warp) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& q_idx) {
            if (cute::elect_one_sync()) {
                tma_copy<kHeadDim, kNextN * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, q_idx * kNextN * kNumHeads);
                tma_copy<kNextN * kNumHeads, 1, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, q_idx);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
        uint32_t q_idx = batch_size, kv_idx, num_kv;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        // Prefetch the first Q
        if ((fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)))
            issue_tma_q(0, next_q_idx), q_iter_idx = 1;

        int kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            // Prefetch next Q when current Q changes
            bool prefetch_q = (q_idx != next_q_idx and scheduler.exist_q_idx(next_q_idx + 1));
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            // Read KV block index
            // TODO: deal with `-1`?
            if (kv_idx == 0 or kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                kv_block_idx_storage = (kv_idx + lane_idx < num_kv ? __ldg(block_table + q_idx * block_table_stride + (kv_idx + lane_idx)) : 0);
            }
            DG_STATIC_ASSERT(32 % kNumBlocksPerSplit == 0, "Invalid `UMMA_M`");

            // Wait Q consumer release and issue TMA Q
            if (prefetch_q) {
                CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_idx + 1);
            }

            int kv_block_idx[kNumBlocksPerSplit];
            #pragma unroll
            for (int i = 0; i < kNumBlocksPerSplit; ++ i)
                kv_block_idx[i] = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr + i);
            kv_block_idx_ptr += kNumBlocksPerSplit;

            // Wait KV consumer release
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            if (cute::elect_one_sync()) {
                #pragma unroll
                for (int i = 0; i < kNumBlocksPerSplit; ++ i) {
                    tma_copy<kHeadDim, BLOCK_KV, 0, __nv_fp8_e4m3, true>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                                         smem_kv[kv_stage_idx] + (BLOCK_KV * kHeadDim) * i,
                                                                         0, 0, 1, kv_block_idx[i]);
                    tma_copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                             smem_kv_scales[kv_stage_idx] + BLOCK_KV * i,
                                             0, kv_block_idx[i]);
                }
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            // Fetch next task
            fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv);
        }
    } else if (is_umma_warp) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;
        uint32_t umma_phase = 1;

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            if (q_idx != next_q_idx) {
                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);
            }

            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                empty_umma_barriers[i]->wait(umma_phase);    
                tcgen05_after_thread_sync();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                    auto a_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                        smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                    auto b_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                        smem_q[q_stage_idx], 0, k * UMMA_K);
                    cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k, runtime_instr_desc);
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i]));
            }
            umma_phase ^= 1;
        }
    } else if (is_math_warp) {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // Offsets
        const auto& tmem_start = __shfl_sync(0xffffffff, warpgroup_idx * UMMA_N, 0);
        const uint32_t thread_idx = threadIdx.x;

        // Weights
        constexpr uint32_t kNumWeightsInReg = (kNextN == 1 ? kNumHeads : cute::min(48, kNumHeads));
        float weights[kNextN][kNumWeightsInReg];
        DG_STATIC_ASSERT(kNumWeightsInReg % 4 == 0, "Invalid number of weights in registers");

        // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;
        uint32_t umma_phase = 0;

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            // Current Q changes
            if (q_idx != next_q_idx) {
                // Release Last Q empty
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                // Wait TMA Q arrival
                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);

                // Read weights
                #pragma unroll
                for (uint32_t i = 0; i < kNextN; ++ i) {
                    for (uint32_t j = 0; j < kNumWeightsInReg; ++ j)
                        weights[i][j] = ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j);
                }
            }

            // Get current Q and KV index
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            // Calculate KV offset in advance
            auto kv_offset = q_idx * kNextN * logits_stride + kv_idx * BLOCK_KV;

            // Compute `[kNextN * kNumHeads, kHeadDim] @ [SPLIT_KV, kHeadDim] -> [kNextN, SPLIT_KV]`
            // Wait TMA KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            // Read per-KV scales
            float scale_kv = ld_shared(smem_kv_scales[kv_stage_idx] + thread_idx);

            // Wait UMMA arrival
            full_umma_barriers[warpgroup_idx]->wait(umma_phase);
            tcgen05_after_thread_sync();
            umma_phase ^= 1;

            // Release KV empty
            empty_kv_barriers[kv_stage_idx]->arrive();

            // Reduce over the head dim and store
            DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
            constexpr uint32_t kNumLDTMElems = kNumHeads * kNextN;
            uint32_t shifted_accum[kNumLDTMElems];
            DG_STATIC_ASSERT(kNumLDTMElems == 32 or kNumLDTMElems == 64 or kNumLDTMElems == 128, "Invalid LDTM");
            auto tmem_load = [&](auto... Is) {
                if constexpr (kNumLDTMElems == 32) {
                    cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start, shifted_accum[Is]...);
                } else if constexpr (kNumLDTMElems == 64) {
                    cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_start, shifted_accum[Is]...);
                } else if constexpr (kNumLDTMElems == 128) {
                    cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_start, shifted_accum[Is]...);
                }
            };
            [&]<size_t... Is>(cute::index_sequence<Is...>) { tmem_load(Is...); }(cute::make_index_sequence<kNumLDTMElems>{});
            cutlass::arch::fence_view_async_tmem_load();

            tcgen05_before_thread_sync();
            empty_umma_barriers[warpgroup_idx]->arrive();

            #pragma unroll
            for (uint32_t i = 0; i < kNextN; ++ i) {
                auto accum = reinterpret_cast<float*>(shifted_accum + i * kNumHeads);

                auto sum_0 = make_float2(0, 0);
                auto sum_1 = make_float2(0, 0);

                const auto& transform_reg = [&](const uint32_t& j, const float2& sum) {
                    auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                    auto b = make_float2(weights[i][j], weights[i][j + 1]);
                    return __ffma2_rn(a, b, sum);
                };

                #pragma unroll
                for (int j = 0; j < kNumWeightsInReg; j += 4) {
                    sum_0 = transform_reg(j, sum_0);
                    sum_1 = transform_reg(j + 2, sum_1);
                }

                const auto& transform_smem = [&](const uint32_t& j, const float2& sum) {
                    auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                    auto b = make_float2(ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j),
                                         ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j + 1));
                    return __ffma2_rn(a, b, sum);
                };

                #pragma unroll
                for (int j = kNumWeightsInReg; j < kNumHeads; j += 4) {
                    sum_0 = transform_smem(j, sum_0);
                    sum_1 = transform_smem(j + 2, sum_1);
                }

                auto sum = __fadd2_rn(sum_0, sum_1);
                float result = scale_kv * (sum.x + sum.y);

                // Store into the global memory
                // NOTES: we have redundant writes here, consider more carefully
                logits[kv_offset + i * logits_stride + thread_idx] = result;
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    }

    // Free tensor memory
    __syncthreads();
    if (is_umma_warp)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
}

} // namespace deep_gemm
