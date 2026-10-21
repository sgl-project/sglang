/*
TopK select kernel bf16 common variant.

Usage caveats:
  - vocab_size must be < 2^23.
  - Do NOT assume that the kernel will select a prefix index on ties.
*/

#pragma once

#include <cutlass/kernel_launch.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <kerutils/kerutils.cuh>

#include "structs.h"
#include "cuda_kernels/utils.cuh"
#include "cuda_kernels/common_parts.cuh"

namespace topk_select_bf16_normal {

template<typename Config>
class TopkSelectKernelBF16Normal : public topk_select_common::TopkSelectKernelBF16Base<Config> {
    using BF16Base = topk_select_common::TopkSelectKernelBF16Base<Config>;
public:
    using ValueT = typename BF16Base::ValueT;
    using OutIdxT = typename BF16Base::OutIdxT;
    using TmaParams = typename BF16Base::TmaParams;
    using SharedMemoryPlanBase = typename BF16Base::SharedMemoryPlanBase;
    using EpilogueT = typename BF16Base::EpilogueT;
    using BF16Base::NUM_THREADS;
    using BF16Base::MAX_TOPK;
    using BF16Base::NUM_ELEMS_PER_SEG;
    using BF16Base::NUM_ELEMS_PER_ROUND;
    using BF16Base::NUM_SEGS_IN_INIT_WINDOW;
    using BF16Base::NUM_INIT_ROUNDS_MAX;
    using BF16Base::NUM_TAIL_ELEMS;
    using BF16Base::NUM_TAIL_SEGS;
    using BF16Base::NUM_SEGS_PER_ROUND;

    static_assert(Config::target_occupancy >= 1 && Config::target_occupancy <= 8);
    static_assert(MAX_TOPK == 512 || MAX_TOPK == 1024 || MAX_TOPK == 4096);
    static constexpr bool HAS_PARTIAL_ROUNDS = NUM_SEGS_PER_ROUND > NUM_TAIL_SEGS;
    struct SharedMemoryPlan : SharedMemoryPlanBase {};

    static __device__ __forceinline__
    void topk_select_kernel_devfunc(const TopkSelectArgs &args, const TmaParams &tma_params) {
        uint32_t batch_idx = blockIdx.x;
        uint32_t end_vocab_idx = args.end_ptr == nullptr ? args.vocab_size : __ldg(args.end_ptr + batch_idx);

        extern __shared__ CUTE_ALIGNAS(1024) char wksp_buf[];
        SharedMemoryPlan &smem = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

        uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
        uint32_t lane_idx = threadIdx.x % 32;

        if (end_vocab_idx <= args.topk) {
            EpilogueT::template topk_select_epilogue<true>(
                (ValueT*)smem.surviving_topk_pairs[0],
                (uint32_t*)(smem.surviving_topk_pairs[1]),
                args,
                batch_idx, end_vocab_idx, warp_idx,
                *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(smem.incoming_topk_pairs)
            );
            return;
        }

        uint32_t survivor_buf_idx = 0;  // Current candidate buffer (A/B, swapped by reconstruct)
        bool have_nan = false;

        BF16Base::init_shared_memory(smem, warp_idx);

        uint32_t num_input_segs = ku::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
        uint32_t num_perm_segs = num_input_segs <= NUM_SEGS_IN_INIT_WINDOW ? 0u : (num_input_segs - 1) / NUM_TAIL_SEGS * NUM_TAIL_SEGS; // Skip permutation if the total number of segs is small enough
        uint32_t num_perm_elems = num_perm_segs * NUM_ELEMS_PER_SEG;
        uint32_t num_tail_elems_padded = num_perm_segs != 0 ? (uint32_t)NUM_TAIL_ELEMS : num_input_segs * NUM_ELEMS_PER_SEG;
        uint32_t num_rounds = ku::ceil_div(num_tail_elems_padded + num_perm_elems, (uint32_t)NUM_ELEMS_PER_ROUND);
        uint32_t num_init_rounds = min(num_rounds, (uint32_t)NUM_INIT_ROUNDS_MAX);

        BF16Base::template scan_segs<false>(
            tma_params, smem,
            batch_idx, end_vocab_idx, args.topk,
            warp_idx, lane_idx,
            num_perm_segs,
            0,
            num_perm_segs,
            num_tail_elems_padded,
            end_vocab_idx - num_perm_elems,
            survivor_buf_idx,
            have_nan,
            [&](uint32_t m) {
                if constexpr (HAS_PARTIAL_ROUNDS) {
                    return (num_init_rounds + m) * NUM_SEGS_PER_ROUND + warp_idx < NUM_TAIL_SEGS + num_perm_segs;
                } else {
                    return true;
                }
            }
        );

        // `__syncthreads_or` is also the barrier that makes the last `reconstruct`'s writes visible before
        // `stage_output_and_epilogue` reads the survivor buffer.
        have_nan = __syncthreads_or(have_nan) != 0;
        if (have_nan) {
            BF16Base::take_action_when_have_nan(args, batch_idx);
            return;
        }

        BF16Base::template stage_output_and_epilogue<Config::sorted_index>(
            smem, *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(smem.incoming_topk_pairs), args,
            batch_idx, end_vocab_idx, survivor_buf_idx, warp_idx, lane_idx);
    }
};

template<typename Kernel>
__launch_bounds__(Kernel::NUM_THREADS, Kernel::TARGET_OCCUPANCY, 1)
__global__ void topk_kernel(__grid_constant__ const TopkSelectArgs args, __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::topk_select_kernel_devfunc(args, tma_params);
}

template<typename Config>
void run_topk_select_kernel(const TopkSelectArgs &args) {
    KU_ASSERT(args.sorted_value == Config::sorted_value, "Dispatch failure");
    KU_ASSERT(args.sorted_index == Config::sorted_index, "Dispatch failure");
    KU_ASSERT(args.return_value == Config::return_value, "Dispatch failure");
    static_assert(cute::is_same_v<typename Config::ValueT, nv_bfloat16>);
    KU_ASSERT(args.vocab_size < MAX_VOCAB_SIZE, "`vocab_size` is too big");

    using Kernel = TopkSelectKernelBF16Normal<Config>;
    KU_ASSERT(args.topk <= Kernel::MAX_TOPK, "topk is too large. Maximum allowed: %d\n", Kernel::MAX_TOPK);
    static_assert(INPUT_STRIDE_ALIGNMENT_REQUIREMENT % 16 == 0);

    auto kernel = topk_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    KU_ASSERT(smem_size * Kernel::TARGET_OCCUPANCY <= args.shared_memory_size_per_sm);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    KU_ASSERT(args.stride_input_batch % 8 == 0, "stride_input_batch must be 16B-aligned");
    typename Kernel::TmaParams tma_params = {Kernel::make_topk_tensor_map(args)};

    ku::launch_kernel(ku::KernelLaunchConfig {
        dim3(args.batch_size),
        dim3(Kernel::NUM_THREADS),
        smem_size,
        args.stream
    }, kernel, args, tma_params);
}

}   // topk_select_bf16_normal
