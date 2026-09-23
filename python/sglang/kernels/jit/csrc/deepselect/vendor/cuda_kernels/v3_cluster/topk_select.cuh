/*
TopK select kernel bf16 cluster variant. Use multiple CTAs to calculate one sequence to avoid wave quantization

Algorithm:
- Each CTA computes its local top-k pairs
- Each CTA sends its local top-k pairs to CTA0
- CTA0 selects top-k from those pairs
*/

#pragma once

#include <cutlass/kernel_launch.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <kerutils/kerutils.cuh>

#include "structs.h"
#include "cuda_kernels/utils.cuh"
#include "cuda_kernels/common_parts.cuh"

namespace topk_select_bf16_cluster {

CUTE_DEVICE
static void st_async_32b(uint32_t dst_addr, const uint32_t& data, transac_bar_t &mbar) {
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile (
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.s32 [%0], {%1}, [%2]; \n"
        :
        : "r"(dst_addr), "r"(data), "r"(mbar_addr)
    );
}

template<typename Config>
class TopkSelectKernelBF16Cluster : public topk_select_common::TopkSelectKernelBF16Base<Config> {
    using BF16Base = topk_select_common::TopkSelectKernelBF16Base<Config>;
public:
    using ValueT = typename BF16Base::ValueT;
    using OutIdxT = typename BF16Base::OutIdxT;
    using TmaParams = typename BF16Base::TmaParams;
    using SharedMemoryPlanBase = typename BF16Base::SharedMemoryPlanBase;
    using EpilogueT = typename BF16Base::EpilogueT;
    using BF16Base::NUM_THREADS;
    using BF16Base::MAX_TOPK;
    using BF16Base::NUM_EXTRA_SLOTS;
    using BF16Base::NUM_ELEMS_PER_SEG;
    using BF16Base::NUM_ELEMS_PER_ROUND;
    using BF16Base::NUM_SEGS_IN_INIT_WINDOW;
    using BF16Base::NUM_INIT_ROUNDS_MAX;
    using BF16Base::NUM_TAIL_ELEMS;
    using BF16Base::NUM_TAIL_SEGS;
    using BF16Base::NUM_SEGS_PER_ROUND;

    static_assert(Config::target_occupancy == 1);
    static_assert(MAX_TOPK == 512 || MAX_TOPK == 1024);
    static_assert(Config::cluster_size == 8 || Config::cluster_size == 16);
    static_assert(NUM_ELEMS_PER_ROUND * sizeof(ValueT) % 1024 == 0);
    static_assert((uint64_t)(MAX_VOCAB_SIZE / NUM_ELEMS_PER_SEG) * (MAX_VOCAB_SIZE / NUM_ELEMS_PER_SEG)
                  + BF16Base::PERM_ADD_BASE <= 0xFFFFFFFFull);
    static constexpr uint32_t CLUSTER_SIZE = Config::cluster_size;
    static constexpr uint32_t NUM_GATHER_PAIRS = Config::cluster_size * MAX_TOPK;
    static constexpr uint32_t NUM_GATHER_UNITS = NUM_GATHER_PAIRS / 2;
    static constexpr uint32_t NUM_UINT32_GATHER_PER_THREAD = ((NUM_GATHER_UNITS + NUM_THREADS - 1) / NUM_THREADS) | 1u;
    struct SharedMemoryPlanBF16Cluster : SharedMemoryPlanBase {
        kerutils::transac_bar_t gather_val_bar;
        kerutils::transac_bar_t gather_bar;
        CUTE_ALIGNAS(16) uint32_t gathered_num_survivors[Config::cluster_size];
    };
    static_assert(NUM_GATHER_UNITS % NUM_THREADS == 0);
    static_assert(NUM_GATHER_PAIRS <= 0xFFFF);
    static_assert(2 * NUM_UINT32_GATHER_PER_THREAD <= 256);

    static __device__ __forceinline__ void topk_select_kernel_devfunc(const TopkSelectArgs &args, const TmaParams &tma_params) {
        uint32_t rank_in_cluster = blockIdx.x;
        uint32_t batch_idx = blockIdx.y;
        uint32_t end_vocab_idx = args.end_ptr == nullptr ? args.vocab_size : __ldg(args.end_ptr + batch_idx);

        extern __shared__ CUTE_ALIGNAS(1024) char wksp_buf[];
        SharedMemoryPlanBF16Cluster &smem = *reinterpret_cast<SharedMemoryPlanBF16Cluster*>(wksp_buf);

        uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
        uint32_t lane_idx = threadIdx.x % 32;

        if (end_vocab_idx <= args.topk) {
            if (rank_in_cluster != 0) {
                return;
            }
            EpilogueT::template topk_select_epilogue<true>(
                (ValueT*)smem.surviving_topk_pairs[0],
                (uint32_t*)(smem.surviving_topk_pairs[1]),
                args,
                batch_idx, end_vocab_idx, warp_idx,
                *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(smem.incoming_topk_pairs)
            );
            return;
        }

        uint32_t survivor_buf_idx = 0;  // current candidate buffer (A/B, swapped by reconstruct)

        BF16Base::init_shared_memory(smem, warp_idx, [&]{
            smem.gather_val_bar.init(1);
            smem.gather_bar.init(1);
        });

        // Still, divide the whole input into the "perm" part and the "tail" part
        // The "tail" part contains the last 1 ~ NUM_TAIL_ELEMS elements and the "perm" part has the other elements.
        uint32_t num_input_segs = ku::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
        uint32_t num_perm_segs = num_input_segs <= NUM_SEGS_IN_INIT_WINDOW ? 0u : (num_input_segs - 1) / NUM_TAIL_SEGS * NUM_TAIL_SEGS;
        uint32_t num_perm_elems = num_perm_segs * NUM_ELEMS_PER_SEG;
        uint32_t num_tail_elems_padded = num_perm_segs != 0 ? (uint32_t)NUM_TAIL_ELEMS : num_input_segs * NUM_ELEMS_PER_SEG;

        // The "tail" part is handled by CTA0 in the cluster
        // "local" means the current CTA (in the cluster)
        uint32_t num_local_tail_elems_padded = rank_in_cluster == 0 ? num_tail_elems_padded : 0u;
        uint32_t num_local_tail_segs = num_local_tail_elems_padded / NUM_ELEMS_PER_SEG;
        uint32_t num_local_tail_elems = rank_in_cluster == 0 ? end_vocab_idx - num_perm_elems : 0u;

        // rank r takes the r-th contiguous interval of the row's visit order, with equal visit elements per rank
        uint32_t num_total_elems_padded = num_tail_elems_padded + num_perm_elems;
        // Rank r's slice of the padded row starts at r * num_total_elems_padded / cluster_size; the tail is
        // rank 0's alone, so clamp that boundary into the perm part (element index within the perm part).
        auto perm_boundary = [&](uint32_t rank) {
            static_assert((uint64_t)Config::cluster_size * (MAX_VOCAB_SIZE + NUM_TAIL_ELEMS) <= 0xFFFFFFFFull);
            uint32_t boundary = rank * num_total_elems_padded / Config::cluster_size;
            return boundary > num_tail_elems_padded ? boundary - num_tail_elems_padded : 0u;
        };
        uint32_t local_start_seg_idx = perm_boundary(rank_in_cluster) / NUM_ELEMS_PER_SEG;
        uint32_t local_end_seg_idx = perm_boundary(rank_in_cluster + 1) / NUM_ELEMS_PER_SEG;
        uint32_t num_local_perm_segs = local_end_seg_idx - local_start_seg_idx;
        uint32_t num_local_elems = num_local_tail_elems_padded + num_local_perm_segs * NUM_ELEMS_PER_SEG;

        uint32_t num_local_rounds = ku::ceil_div(num_local_elems, (uint32_t)NUM_ELEMS_PER_ROUND);
        uint32_t num_local_init_rounds = min(num_local_rounds, (uint32_t)NUM_INIT_ROUNDS_MAX);

        uint32_t num_local_segs = num_local_tail_segs + num_local_perm_segs;
        uint32_t used_segs = num_local_init_rounds * NUM_SEGS_PER_ROUND;
        uint32_t rem_segs = num_local_segs > used_segs ? num_local_segs - used_segs : 0u;
        uint32_t warp_rem_segs = warp_idx < rem_segs ? rem_segs - warp_idx : 0u;
        uint32_t warp_round_limit = ku::ceil_div(warp_rem_segs, (uint32_t)NUM_SEGS_PER_ROUND);

        // Each CTA gets its local top-k elements
        bool nan_seen = false;
        uint32_t num_survivors = BF16Base::template scan_segs<true>(
            tma_params, smem,
            batch_idx, end_vocab_idx, args.topk, warp_idx, lane_idx,
            num_perm_segs, local_start_seg_idx, num_local_perm_segs, num_local_tail_elems_padded, num_local_tail_elems,
            survivor_buf_idx,
            nan_seen,
            [&](uint32_t m) { return m < warp_round_limit; }
        );

        // CTA-wide OR; this is also the barrier that makes the scan's writes to the survivor buffer visible to the gather below
        nan_seen = __syncthreads_or(nan_seen) != 0;

        // The cluster barrier must not complete before every CTA has finished its own scan: the gather
        // below writes into CTA0's shared memory, so a CTA that finished early would otherwise clobber
        // CTA0's `tma_load_buf` / `incoming_topk_pairs` while CTA0 is still scanning them.
        ku::barrier_cluster_arrive_relaxed();   // Paired with the `barrier_cluster_wait_acquire` below
        ku::barrier_cluster_wait_acquire();

        static_assert(NUM_GATHER_UNITS * sizeof(uint32_t) <= sizeof(smem.incoming_topk_pairs));
        void* val_dst = (void*)(int64_t)cute::set_block_rank(
            cute::cast_smem_ptr_to_uint((uint32_t*)smem.incoming_topk_pairs + rank_in_cluster * (MAX_TOPK / 2)), 0);

        static_assert(NUM_GATHER_PAIRS * sizeof(uint64_t) <= sizeof(SharedMemoryPlanBase::tma_load_buf));
        void* dst = (void*)(int64_t)cute::set_block_rank(
            cute::cast_smem_ptr_to_uint(reinterpret_cast<uint64_t*>(smem.tma_load_buf) + rank_in_cluster * MAX_TOPK), 0);

        static_assert(MAX_TOPK*sizeof(ValueT) % 16 == 0);
        constexpr uint32_t NUM_VAL_CHUNKS = MAX_TOPK * sizeof(ValueT) / 16; // 16B store
        constexpr uint32_t NUM_VAL_CHUNKS_PER_THREAD = ku::ceil_div(NUM_VAL_CHUNKS, NUM_THREADS);

        if (warp_idx == 0 && cute::elect_one_sync()) {
            if (rank_in_cluster == 0) {
                smem.gather_val_bar.arrive_and_expect_tx((NUM_VAL_CHUNKS * 16 + sizeof(uint32_t)) * CLUSTER_SIZE);
                smem.gather_bar.arrive_and_expect_tx((MAX_TOPK * (uint32_t)sizeof(uint64_t)) * CLUSTER_SIZE);
            }
            st_async_32b(
                cute::set_block_rank(cute::cast_smem_ptr_to_uint(smem.gathered_num_survivors + rank_in_cluster), 0),
                num_survivors | ((uint32_t)nan_seen << 31),
                smem.gather_val_bar
            );
        }

        // Store values first... so that CTA0 can begin pivot selection earlier
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_VAL_CHUNKS_PER_THREAD; ++i) {
            uint32_t c = i * NUM_THREADS + threadIdx.x;
            if constexpr (NUM_VAL_CHUNKS % NUM_THREADS != 0) {
                if (c >= NUM_VAL_CHUNKS) break;
            }
            uint32_t vw[4];
            CUTE_UNROLL
            for (uint32_t j = 0; j < 4; ++j) {
                uint32_t pair2[4];
                topk_select_common::ld_shared<4>(pair2, (const uint32_t*)(smem.surviving_topk_pairs[survivor_buf_idx] + 8 * c + 2 * j));
                vw[j] = __byte_perm(pair2[1], pair2[3], 0x5410);
            }
            ku::st_async((char*)val_dst + c * sizeof(uint4), make_uint4(vw[0], vw[1], vw[2], vw[3]),
                        smem.gather_val_bar);
        }

        // And then store those (value, index) pairs
        static_assert(MAX_TOPK * sizeof(uint64_t) % 16 == 0);
        constexpr uint32_t NUM_INDEX_VALUE_CHUNKS = MAX_TOPK * sizeof(uint64_t) / 16;
        static_assert(NUM_INDEX_VALUE_CHUNKS % NUM_THREADS == 0);
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_INDEX_VALUE_CHUNKS / NUM_THREADS; ++i) {
            uint32_t s = 2 * (i * NUM_THREADS + threadIdx.x);
            ulonglong2 pp = *reinterpret_cast<const ulonglong2*>(smem.surviving_topk_pairs[survivor_buf_idx] + s);
            ku::st_async((char*)dst + s * sizeof(uint64_t), pp, smem.gather_bar);
        }

        if (rank_in_cluster != 0) {
            return;
        }

        // The following code is CTA0 only
        smem.gather_val_bar.wait(0);
        BF16Base::clear_reconstruct_histograms(smem, threadIdx.x);
        __syncthreads();

        static_assert(Config::cluster_size <= 32);
        uint32_t stored_num_survivors = lane_idx < Config::cluster_size ? smem.gathered_num_survivors[lane_idx] : 0u;
        nan_seen |= (stored_num_survivors >> 31) != 0;

        uint32_t unit_base = threadIdx.x * NUM_UINT32_GATHER_PER_THREAD;
        uint32_t num_my_units = unit_base < NUM_GATHER_UNITS ? min((uint32_t)NUM_UINT32_GATHER_PER_THREAD, NUM_GATHER_UNITS - unit_base) : 0u;

        const uint32_t *gather_vals = (const uint32_t*)smem.incoming_topk_pairs;
        auto num_valid_in_unit = [&](uint32_t unit) {
            uint32_t rank = 2 * unit / MAX_TOPK;
            uint32_t offset = 2 * unit % MAX_TOPK;
            uint32_t count = smem.gathered_num_survivors[rank] & 0x7FFFFFFFu;
            return offset < count ? min(2u, count - offset) : 0u;
        };
        nv_bfloat162 values[NUM_UINT32_GATHER_PER_THREAD];
        uint32_t num_my_padding_elems = 0;
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_UINT32_GATHER_PER_THREAD; i++) {
            if (i == num_my_units) break;
            values[i] = topk_select_common::u32_to_bf16x2(gather_vals[unit_base + i]);
            num_my_padding_elems += 2 - num_valid_in_unit(unit_base + i);
        }
        BF16Base::histogram_radix_msb(smem.reconstruct_bucket_counter[0], values, num_my_units);
        __syncthreads();

        auto [pivot_value_x2_bits, out_prefix, eq_quota, cnt_nan] =
            BF16Base::template compute_pivot_and_quota<true>(args.topk, values, num_my_units, warp_idx, lane_idx, smem, num_my_padding_elems);
        nan_seen |= cnt_nan != 0;

        smem.gather_bar.wait(0);
        __syncthreads();

        uint32_t out_ptr = cute::cast_smem_ptr_to_uint(smem.surviving_topk_pairs[survivor_buf_idx ^ 1]) + out_prefix * (uint32_t)sizeof(uint64_t);
        uint32_t gather_pairs_base = cute::cast_smem_ptr_to_uint(smem.tma_load_buf);
        CUTE_UNROLL
        for (uint32_t m = 0; m < NUM_UINT32_GATHER_PER_THREAD; m++) {
            if (m == num_my_units) break;
            BF16Base::template copy_selected_pairs_to_survivor<true>(out_ptr, eq_quota, values[m], pivot_value_x2_bits, gather_pairs_base + 2 * (unit_base + m) * (uint32_t)sizeof(uint64_t), num_valid_in_unit(unit_base + m));
        }
        survivor_buf_idx ^= 1;

        nan_seen = __syncthreads_or(nan_seen) != 0;
        if (nan_seen) {
            BF16Base::take_action_when_have_nan(args, batch_idx);
            return;
        }

        BF16Base::template stage_output_and_epilogue<Config::sorted_index>(
            smem, *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(smem.incoming_topk_pairs), args,
            batch_idx, end_vocab_idx, survivor_buf_idx, warp_idx, lane_idx);
    }
};

template<typename Kernel>
__launch_bounds__(Kernel::NUM_THREADS, Kernel::TARGET_OCCUPANCY, Kernel::CLUSTER_SIZE)
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

    using Kernel = TopkSelectKernelBF16Cluster<Config>;
    KU_ASSERT(args.topk <= Kernel::MAX_TOPK, "topk is too large. Maximum allowed: %d\n", Kernel::MAX_TOPK);
    static_assert(INPUT_STRIDE_ALIGNMENT_REQUIREMENT % 16 == 0);

    auto kernel = topk_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlanBF16Cluster);
    KU_ASSERT(smem_size * Kernel::TARGET_OCCUPANCY <= args.shared_memory_size_per_sm);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    if constexpr (Kernel::CLUSTER_SIZE > 8) {
        KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    }

    KU_ASSERT(args.stride_input_batch % 8 == 0, "stride_input_batch must be 16B-aligned");
    typename Kernel::TmaParams tma_params = {Kernel::make_topk_tensor_map(args)};

    ku::launch_kernel(ku::KernelLaunchConfig {
        dim3(Kernel::CLUSTER_SIZE, args.batch_size),
        dim3(Kernel::NUM_THREADS),
        smem_size,
        args.stream,
        dim3(Kernel::CLUSTER_SIZE, 1, 1)
    }, kernel, args, tma_params);
}

}   // topk_select_bf16_cluster
