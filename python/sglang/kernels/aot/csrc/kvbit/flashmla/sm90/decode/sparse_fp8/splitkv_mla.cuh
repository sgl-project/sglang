#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <math_constants.h>

#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

#include "attention_math.h"
#include "components/helpers.h"
#include "config.h"
#include "flashmla_utils.h"
#include "packed_int4.cuh"
#include "splitkv_mla.h"

using namespace cute;

namespace sm90::decode::sparse_fp8 {

static constexpr float MAX_INIT_VAL = -1e30;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;

template <typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void scale_softmax(
    Tensor0& rP,
    Tensor1& rS,
    Tensor2& rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    bool is_kv_valid[],
    int idx_in_warpgroup) {
  float scale_for_olds[2];
  CUTE_UNROLL
  for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
    Tensor cur_rP = flatten(rP(make_coord(_, local_row_idx, _), _, _));
    Tensor cur_rS = flatten(rS(make_coord(_, local_row_idx, _), _, _));
    Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));
    float cur_max = -INFINITY;
    CUTE_UNROLL
    for (int i = 0; i < size(cur_rP); ++i) {
      if (!is_kv_valid[(i & 1) + (i / 2) * 8 + (idx_in_warpgroup % 4) * 2]) cur_rP(i) = -INFINITY;
      cur_max = max(cur_max, cur_rP(i));
    }
    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
    cur_max *= scale_softmax_log2;
    const float old_max = rM[local_row_idx];
    rM[local_row_idx] = max(cur_max, old_max);
    const float scale_for_old = exp2f(old_max - rM[local_row_idx]);
    scale_for_olds[local_row_idx] = scale_for_old;
    CUTE_UNROLL
    for (int i = 0; i < size(cur_rO); ++i)
      cur_rO(i) *= scale_for_old;
    float cur_sum = 0;
    CUTE_UNROLL
    for (int i = 0; i < size(cur_rP); ++i) {
      cur_rP(i) = exp2f(cur_rP(i) * scale_softmax_log2 - rM[local_row_idx]);
      cur_rS(i) = (bf16)cur_rP(i);
      cur_sum += cur_rP(i);
    }
    rL[local_row_idx] = rL[local_row_idx] * scale_for_old + cur_sum;
  }
  if (idx_in_warpgroup % 4 == 0) *(float2*)(sScale + 2 * (idx_in_warpgroup / 4)) = *(float2*)(scale_for_olds);
}

template <ModelType MODEL_TYPE, int NUM_HEADS>
template <typename TMAParams>
__device__ void
KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnDecodeParams& params, const TMAParams& tma_params) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
  static_assert(MODEL_TYPE == ModelType::MODEL1 && NUM_HEADS == 64 && CLUSTER_SIZE == 1);
  const int head_block_idx = 0;
  const int s_q_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int warpgroup_idx = cutlass::canonical_warp_group_idx();
  const int idx_in_warpgroup = threadIdx.x % 128;
  const int warp_idx = cutlass::canonical_warp_idx_sync();

  extern __shared__ char wksp_buf[];
  SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
  Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
  Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
  Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
  Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
  float* sM = plan.sM;
  float* sL = plan.sL;
  float* sScale = plan.sScale;
  if (warp_idx == 0 && elect_one_sync()) {
    cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
    plan.bar_q.init(1);
    CUTE_UNROLL
    for (int i = 0; i < NUM_K_BUFS; ++i) {
      plan.bar_k_local_ready[i].init(128);
      plan.bar_k_avail[i].init(256);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  int bar_phase_k = 0;
  const DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
  if (sched_meta.begin_req_idx >= params.b) return;

  if (warp_idx == 0 && elect_one_sync()) {
    Tensor gQ = flat_divide(
        tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, sched_meta.begin_req_idx),
        Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{})(_, _, head_block_idx, _0{});
    launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
    plan.bar_q.arrive_and_expect_tx(BLOCK_M * HEAD_DIM_K * sizeof(bf16));
  }

  struct MainloopArgs {
    int start_block_idx, end_block_idx;
    bool is_no_split;
    int topk_length, extra_topk_length, num_orig_kv_blocks;
  };
  auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
    MainloopArgs args;
    args.topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
    args.extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
    const int orig_topk_padded = max(ku::ceil(args.topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
    const int total_topk_padded = orig_topk_padded + ku::ceil(args.extra_topk_length, (int)TOPK_BLOCK_SIZE);
    args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
    args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
    args.end_block_idx =
        batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
    args.is_no_split = batch_idx == sched_meta.begin_req_idx
                           ? !sched_meta.is_first_req_splitted
                           : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);
    return args;
  };

  if (warpgroup_idx == 0) {
    cutlass::arch::warpgroup_reg_alloc<192>();
    TiledMMA tiled_mma_QK = TiledMMA_QK{};
    ThrMMA thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
    TiledMMA tiled_mma_PV = TiledMMA_PV_LocalP{};
    ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
    float rL[2], rM[2];
    Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V / 2>>{});
    Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
    Tensor rS = make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));
    float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
    if (params.attn_sink != nullptr) {
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        const int head_idx = get_AorC_row_idx(i, idx_in_warpgroup);
        rAttn_sink[i] = __ldg(params.attn_sink + head_idx) * CUDART_L2E_F;
      }
    }
#pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      const MainloopArgs args = get_cur_req_info(batch_idx);
      rL[0] = rL[1] = 0.0f;
      rM[0] = rM[1] = MAX_INIT_VAL;
      cute::fill(rO, 0.);
      plan.bar_q.wait((sched_meta.begin_req_idx - batch_idx) & 1);
      CUTE_NO_UNROLL
      for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
        const int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
        Tensor sK = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutHalfV{});
        plan.bar_k_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
        gemm<true, -1>(tiled_mma_QK, thr_mma_QK.partition_fragment_A(sQ), thr_mma_QK.partition_fragment_B(sK), rP);
        bar_phase_k ^= 1 << buf_idx;
        // Preserve the consumer schedule: overlap the free-barrier with QK.
        // Warpgroups reach their barriers from different control-flow paths.
        if (block_idx != args.start_block_idx)
          NamedBarrier(256, NamedBarriers::sScale_and_sS_free).arrive_and_wait_unaligned();
        cute::warpgroup_wait<0>();
        scale_softmax(
            rP, rS, rO, params.sm_scale_div_log2, sScale, rM, rL, plan.is_kv_valid[buf_idx], idx_in_warpgroup);
        save_rPb_to_sP(rS, sS, idx_in_warpgroup);
        fence_view_async_shared();
        gemm<false, -1>(tiled_mma_PV, rS, thr_mma_PV.partition_fragment_B(sV), rO);
        NamedBarrier(256, NamedBarriers::sScale_and_sS_ready).arrive_unaligned();
        cute::warpgroup_wait<0>();
        plan.bar_k_avail[buf_idx].arrive();
      }
      if (warp_idx == 0 && elect_one_sync()) {
        if (batch_idx != sched_meta.end_req_idx) {
          Tensor gQ = flat_divide(
              tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx + 1),
              Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{})(_, _, head_block_idx, _0{});
          launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
          plan.bar_q.arrive_and_expect_tx(BLOCK_M * HEAD_DIM_K * sizeof(bf16));
        } else {
          cudaTriggerProgrammaticLaunchCompletion();
        }
      }
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        rL[i] += __shfl_xor_sync(0xffffffff, rL[i], 1);
        rL[i] += __shfl_xor_sync(0xffffffff, rL[i], 2);
        if (idx_in_warpgroup % 4 == 0) {
          const int row = get_AorC_row_idx(i, idx_in_warpgroup);
          sL[row] = rL[i];
          sM[row] = rM[i];
        }
      }
      float o_scales[2];
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        if (args.is_no_split) {
          o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i] + exp2f(rAttn_sink[i] - rM[i]));
        } else {
          o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i]);
        }
        if (idx_in_warpgroup % 4 == 0) {
          plan.sOScale[get_AorC_row_idx(i, idx_in_warpgroup)] = o_scales[i];
        }
      }
      NamedBarrier(256, NamedBarriers::oBuf_free_and_sL_ready).arrive_and_wait_unaligned();
      if (args.is_no_split) {
        bf16* o_ptr = params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q;
        Tensor gO = make_tensor(
            make_gmem_ptr(o_ptr),
            make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_h_q, _1{})));
        store_o<true>(
            rO,
            gO,
            sOBuf,
            sOAccumBuf,
            plan,
            o_scales,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            BLOCK_M,
            warpgroup_idx,
            idx_in_warpgroup);
        const int i = threadIdx.x;
        if (i < BLOCK_M) {
          const float sink_log2 = params.attn_sink ? __ldg(params.attn_sink + i) * CUDART_L2E_F : -INFINITY;
          params.lse[batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q + i] =
              kvbit::dsv4::no_split_lse(sL[i], sM[i], sink_log2);
        }
      } else {
        const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        const int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
        float* oaccum_ptr =
            params.o_accum + split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q;
        Tensor gOAccum = make_tensor(
            make_gmem_ptr(oaccum_ptr),
            make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_accum_h_q, _1{})));
        store_o<false>(
            rO,
            gOAccum,
            sOBuf,
            sOAccumBuf,
            plan,
            o_scales,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            BLOCK_M,
            warpgroup_idx,
            idx_in_warpgroup);
        const int i = threadIdx.x;
        if (i < BLOCK_M) {
          params.lse_accum[split_idx * params.stride_lse_accum_split + s_q_idx * params.stride_lse_accum_s_q + i] =
              sL[i] == 0.0f ? -INFINITY : log2f(sL[i]) + sM[i];
        }
      }
      cute::tma_store_wait<0>();
      sync_all_threads_in_cluster();
    }
  } else if (warpgroup_idx == 1) {
    cutlass::arch::warpgroup_reg_dealloc<160>();
    TiledMMA tiled_mma_PV = TiledMMA_PV_RemoteP{};
    ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
    Tensor rO = partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V / 2>>{});
#pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      const MainloopArgs args = get_cur_req_info(batch_idx);
      cute::fill(rO, 0.);
      CUTE_NO_UNROLL
      for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
        const int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
        Tensor sV =
            make_tensor(make_smem_ptr(plan.u.k[buf_idx].data() + (SmemLayoutV{})(_256{}, _0{})), SmemLayoutHalfV{});
        NamedBarrier(256, NamedBarriers::sScale_and_sS_ready).arrive_and_wait_unaligned();
        float cur_scales[2];
        *(float2*)cur_scales = *(float2*)(sScale + (idx_in_warpgroup / 4) * 2);
        CUTE_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
          Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));
          CUTE_UNROLL
          for (int i = 0; i < size(cur_rO); ++i)
            cur_rO(i) *= cur_scales[local_row_idx];
        }
        gemm<false, -1>(tiled_mma_PV, thr_mma_PV.partition_fragment_A(sS), thr_mma_PV.partition_fragment_B(sV), rO);
        cute::warpgroup_wait<0>();
        plan.bar_k_avail[buf_idx].arrive();
        if (block_idx != args.end_block_idx - 1)
          NamedBarrier(256, NamedBarriers::sScale_and_sS_free).arrive_unaligned();
      }
      NamedBarrier(256, NamedBarriers::oBuf_free_and_sL_ready).arrive_and_wait_unaligned();
      float o_scales[2];
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i)
        o_scales[i] = plan.sOScale[get_AorC_row_idx(i, idx_in_warpgroup)];
      if (args.is_no_split) {
        bf16* o_ptr = params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q;
        Tensor gO = make_tensor(
            make_gmem_ptr(o_ptr),
            make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_h_q, _1{})));
        store_o<true>(
            rO,
            gO,
            sOBuf,
            sOAccumBuf,
            plan,
            o_scales,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            BLOCK_M,
            warpgroup_idx,
            idx_in_warpgroup);
      } else {
        const int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        const int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
        float* oaccum_ptr =
            params.o_accum + split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q;
        Tensor gOAccum = make_tensor(
            make_gmem_ptr(oaccum_ptr),
            make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_accum_h_q, _1{})));
        store_o<false>(
            rO,
            gOAccum,
            sOBuf,
            sOAccumBuf,
            plan,
            o_scales,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            BLOCK_M,
            warpgroup_idx,
            idx_in_warpgroup);
      }
      cute::tma_store_wait<0>();
      sync_all_threads_in_cluster();
    }
  } else {
    cutlass::arch::warpgroup_reg_dealloc<152>();
    CUTE_NO_UNROLL
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      const MainloopArgs args = get_cur_req_info(batch_idx);
      CUTE_NO_UNROLL
      for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
        const int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
        const bool extra = block_idx >= args.num_orig_kv_blocks;
        const int relative_block = extra ? block_idx - args.num_orig_kv_blocks : block_idx;
        const int* indices =
            extra ? params.extra_indices + batch_idx * params.stride_extra_indices_b +
                        s_q_idx * params.stride_extra_indices_s_q
                  : params.indices + batch_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;
        const int width = extra ? params.extra_topk : params.topk;
        const int length = extra ? args.extra_topk_length : args.topk_length;
        const int valid_count = max(0, min(TOPK_BLOCK_SIZE, min(width, length) - relative_block * TOPK_BLOCK_SIZE));
        const uint8_t* base =
            reinterpret_cast<const uint8_t*>(extra ? params.extra_packed_kcache_ptr : params.packed_kcache_ptr);
        const int64_t num_rows = extra ? static_cast<int64_t>(params.extra_num_blocks) * params.extra_page_block_size
                                       : static_cast<int64_t>(params.num_blocks) * params.page_block_size;
        plan.bar_k_avail[buf_idx].wait((bar_phase_k >> buf_idx & 1) ^ 1);
        load_int4_tile(
            plan,
            buf_idx,
            indices + relative_block * TOPK_BLOCK_SIZE,
            valid_count,
            num_rows,
            base,
            params.packed_row_bytes,
            idx_in_warpgroup);
        // Every producer publishes its writes before arriving. The consumer
        // waits for all 128 arrivals; no producer-only pointer-table barrier.
        fence_view_async_shared();
        plan.bar_k_local_ready[buf_idx].arrive();
        bar_phase_k ^= 1 << buf_idx;
      }
      sync_all_threads_in_cluster();
    }
  }
#else
  if (cute::thread0()) CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
#endif
}

template <typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE) flash_fwd_splitkv_mla_fp8_sparse_kernel(
    __grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TMAParams tma_params) {
  Kernel::template devfunc<TMAParams>(params, tma_params);
}

template <ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run_impl(const SparseAttnDecodeParams& params) {
  static_assert(MODEL_TYPE == ModelType::MODEL1 && NUM_HEADS == 64);
  KU_ASSERT(params.h_kv == 1 && params.h_q == 64);
  KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
  KU_ASSERT(params.d_qk == HEAD_DIM_K && params.d_v == HEAD_DIM_V);
  KU_ASSERT(params.packed_kcache_ptr != nullptr);
  KU_ASSERT(
      params.packed_row_bytes == kvbit::dsv4::COMPACT_ROW_BYTES ||
      params.packed_row_bytes == kvbit::dsv4::ALIGNED_ROW_BYTES);
  auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q, params.b);
  auto tma_Q = cute::make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(
          make_gmem_ptr((bf16*)params.q),
          make_layout(shape_Q, make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b))),
      SmemLayoutQ{});
  CUtensorMap tensor_map_o;
  {
    uint64_t size[5] = {
        OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V / OBUF_SW, (unsigned long)params.s_q, (unsigned long)params.b};
    uint64_t stride[4] = {
        params.stride_o_h_q * sizeof(bf16),
        OBUF_SW * sizeof(bf16),
        params.stride_o_s_q * sizeof(bf16),
        params.stride_o_b * sizeof(bf16)};
    uint32_t box_size[5] = {OBUF_SW, BLOCK_M, HEAD_DIM_V / OBUF_SW, 1, 1};
    uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
    CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tensor_map_o,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        5,
        params.out,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    KU_ASSERT(res == CUresult::CUDA_SUCCESS);
  }
  TmaParams<decltype(shape_Q), decltype(tma_Q)> tma_params = {shape_Q, tma_Q, tensor_map_o};
  auto mla_kernel =
      &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS>, decltype(tma_params)>;
  constexpr size_t smem_size = sizeof(SharedMemoryPlan);
  KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  cutlass::ClusterLaunchParams launch_params = {
      dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts),
      dim3(NUM_THREADS, 1, 1),
      dim3(CLUSTER_SIZE, 1, 1),
      smem_size,
      params.stream};
  cutlass::launch_kernel_on_cluster(launch_params, (void*)mla_kernel, params, tma_params);
  KU_CHECK_KERNEL_LAUNCH();
}

template <ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run_int4(const SparseAttnDecodeParams& params) {
  run_impl(params);
}

template <ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_int4_sparse_kernel(const SparseAttnDecodeParams& params) {
  KernelTemplate<MODEL_TYPE, NUM_HEADS>::run_int4(params);
}

}  // namespace sm90::decode::sparse_fp8
