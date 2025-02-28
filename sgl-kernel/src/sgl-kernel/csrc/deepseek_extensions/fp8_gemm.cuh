#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm {

enum class Layout {
    RowMajor,
    ColMajor
};

template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return (block_m == 64 ? 1 : 2) * kNumMathThreadsPerGroup + kNumTMAThreads;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                uint32_t shape_m,
                const __grid_constant__ CUtensorMap tensor_map_a,
                const __grid_constant__ CUtensorMap tensor_map_b,
                const __grid_constant__ CUtensorMap tensor_map_scales_a,
                const __grid_constant__ CUtensorMap tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_N, BLOCK_K) == 1, "Too much B scales in a single block");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Shared memory
    static constexpr int kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_B_SIZE = ceil_div<uint32_t>(SHAPE_K_SCALES * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads) {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages];
    __nv_fp8_e4m3* smem_b[kNumStages];
    float* smem_scales_a[kNumStages];
    float* smem_scales_b;

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

    // Fill shared memory pointers
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_a[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_A_SIZE_PER_STAGE);
    }
    smem_scales_b = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE));

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_b) + SMEM_SCALES_B_SIZE);
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (threadIdx.x == kNumMathThreads) {
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling
    struct DivisibleK {};
    struct NotDivisibleK {};
    auto launch_k_iterations = [](const auto& func) {
        if constexpr (SHAPE_K % kFullKOfAllStages == 0) {
            for (int k_iter = 0; k_iter < kNumIterations; ++ k_iter)
                func(k_iter, DivisibleK{});
        } else {
            for (int k_iter = 0; k_iter < kNumIterations - 1; ++ k_iter)
                func(k_iter, DivisibleK{});
            func(kNumIterations - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast>(shape_m, grouped_layout);

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                launch_k_iterations([&](int k_iter, auto type) {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        // Wait consumer release
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                        // Issue TMA A with broadcasting
                        auto& full_barrier = *full_barriers[s];
                        int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                        tma_copy<kNumTMAMulticast>(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                                   smem_a[s], k_idx, scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx));
                        tma_copy<kNumTMAMulticast>(&tensor_map_scales_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                                   smem_scales_a[s], m_block_idx * BLOCK_M,
                                                   scheduler.get_global_idx(SHAPE_K_SCALES, 1, k_idx / BLOCK_K));

                        // Issue TMA B without broadcasting
                        tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_b[s], k_idx, scheduler.get_global_idx<false>(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx));
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                    }

                    // Wait unaligned cases
                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                        full_barriers[s]->arrive();
                    }
                });
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s)
                    empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + 1) & 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Decide the number of scales B to load
            DG_STATIC_ASSERT(SHAPE_N % 8 == 0, "Invalid shape N");
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(SHAPE_N - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            uint32_t num_scales_b = SHAPE_K_SCALES * (num_former_iters >= num_full_iters ? 1 : 2);

            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
            if (threadIdx.x >= 32) {
                auto num_previous_lines = scheduler.get_global_idx<false>(ceil_div(SHAPE_N, BLOCK_K), 0, 0, m_block_idx);
                auto local_scales_b = scales_b + (num_previous_lines + ((n_block_idx * BLOCK_N) / BLOCK_K)) * SHAPE_K_SCALES;
                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_scales_b; i += kNumMathThreads - 32)
                    st_shared(smem_scales_b + i, __ldg(local_scales_b + i));
            }
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Accumulation for WGMMA or CUDA promotion
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(lane_idx) : void();
                }
            };

            // Launch MMAs
            launch_k_iterations([&](int k_iter, auto type) {
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                #pragma unroll
                for (int s = 0; s < kNumInnerStages; ++ s) {
                    // Read B scales
                    float scale_b_0 = ld_shared(smem_scales_b + k_iter * kNumStages + s), scale_b_1;
                    // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                    if constexpr (not kMustUseUniformedScaleB)
                        scale_b_1 = ld_shared(smem_scales_b + k_iter * kNumStages + s + SHAPE_K_SCALES);

                    // Wait TMA arrivals
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                    // Read A scales
                    // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                    auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0), scale_a_1 = ld_shared(smem_scales_a[s] + r_1);

                    // Commit WGMMA instructions
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                        warpgroup_fence_operand(accum[i]);
                    warpgroup_arrive();
                    #pragma unroll
                    for (int k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    warpgroup_commit_batch();
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                        warpgroup_fence_operand(accum[i]);
                    warpgroup_wait<0>();

                    // Notify barrier arrival
                    empty_barrier_arrive(s);

                    // Promote with scales
                    float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                    float scale_0_1, scale_1_1;
                    if constexpr (not kMustUseUniformedScaleB)
                        scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                        final_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                        final_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                        final_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                        final_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                    }
                }

                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                    empty_barrier_arrive(s);
                }
            });

            // Write back to shared memory using STSM
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 8; ++ i) {
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[i * 8 + 0], final_accum[i * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 2], final_accum[i * 8 + 3]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 4], final_accum[i * 8 + 5]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 6], final_accum[i * 8 + 7]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16)
                );
            }
            if constexpr (WGMMA::kNumAccum % 8 != 0) {
                SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 0], final_accum[WGMMA::kNumAccum / 8 * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 2], final_accum[WGMMA::kNumAccum / 8 * 8 + 3]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + WGMMA::kNumAccum / 8 * 16
                );
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Use TMA store to write back to global memory
            if (threadIdx.x == 0) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d, n_block_idx * BLOCK_N,
                                              scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx));
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType>
class Gemm {
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    Gemm() = default;

    static void run(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                    uint32_t shape_m,
                    const CUtensorMap& tma_a_desc,
                    const CUtensorMap& tma_b_desc,
                    const CUtensorMap& tma_scales_a_desc,
                    const CUtensorMap& tma_d_desc,
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      kNumTMAMulticast, kGemmType>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;

        // Launch
        auto status = cudaLaunchKernelEx(&config, kernel,
                                         gmem_d, scales_b, grouped_layout,
                                         shape_m,
                                         tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_a_desc(T* global_address, uint32_t shape_m) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_K, BLOCK_M, BLOCK_K);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_b_desc(T* global_address) {
        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                SHAPE_K, SHAPE_N * (kGemmType != GemmType::Normal ? kNumGroups : 1), BLOCK_K, BLOCK_N);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_d_desc(T* global_address, uint32_t shape_m) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_N,
                                min(BLOCK_M, shape_m), BLOCK_N,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_scales_a_desc(T* global_address, uint32_t shape_m) {
        // Make TMA aligned to 16 bytes
        constexpr uint32_t kAlignment = 16 / sizeof(T);
        shape_m = ceil_div(shape_m, kAlignment) * kAlignment;

        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                shape_m, ceil_div(SHAPE_K, BLOCK_K) * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), BLOCK_M, 1,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_desc(
            T* global_address, Layout layout,
            uint32_t gmem_rows, uint32_t gmem_cols,
            uint32_t smem_rows, uint32_t smem_cols,
            CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
        if (layout == Layout::RowMajor) {
            uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
            uint32_t smem_dim[2] = {smem_cols, smem_rows};
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_cols * sizeof(T), smem_dim, swizzle_type);
        } else {
            uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
            uint32_t smem_dim[2] = {smem_rows, smem_cols};
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type);
        }
    }
};

};  // namespace deep_gemm

#pragma clang diagnostic pop
