/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "kernel.cuh"

namespace sglang {

template <typename Kernel>
__global__ void sparse_mla_q8kv8_etap_kernel(__grid_constant__ const SparseMlaQ8Kv8PrefillParams params);

// Transpose both GEMMs so heads occupy WGMMA's narrow N dimension:
// K Q^T -> [64, H], V^T P^T -> [512, H]. Softmax still reduces over keys.
template <int D_QK, int H, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK, int LOGICAL_H = H>
struct SparseMlaQ8Kv8EtapKernel {
  static_assert(H == 8 || H == 16 || H == 32);
  static_assert(LOGICAL_H >= H && LOGICAL_H % H == 0);
  static constexpr int HEAD_SPLITS = LOGICAL_H / H;
  static constexpr int NUM_THREADS = 256;
  static constexpr int MIN_CTAS = 1;
  static constexpr int NUM_STAGES = 2;
  static constexpr bool COALESCED_O = true;
  static constexpr int D_V = 512;
  static constexpr int B_TOPK = 64;
  static constexpr int NUM_COLS = H / 4;
  using fp8_t = cutlass::float_e4m3_t;
  static __device__ __forceinline__ cutlass::Array<fp8_t, 2> pack_probabilities(float p0, float p1) {
    cutlass::Array<float, 2> values;
    values[0] = p0;
    values[1] = p1;
    return cutlass::NumericArrayConverter<fp8_t, float, 2>::convert(values);
  }

  template <int M, int K>
  using SmemLayout = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<M>, Int<K>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
  using SmemLayoutQ = SmemLayout<H, D_QK>;
  using SmemLayoutK = SmemLayout<B_TOPK, D_QK>;
  using SmemLayoutVt = SmemLayout<D_V, B_TOPK>;
  using SmemLayoutP = SmemLayout<H, B_TOPK>;
  using SmemLayoutO = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<H>, Int<D_V>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
  using MmaAtom = std::conditional_t<
      H == 8,
      GMMA::MMA_64x8x32_F32E4M3E4M3_SS_TN<>,
      std::conditional_t<H == 16, GMMA::MMA_64x16x32_F32E4M3E4M3_SS_TN<>, GMMA::MMA_64x32x32_F32E4M3E4M3_SS_TN<>>>;
  using TiledMMA = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_1, _1, _1>>{}));
  using TransposeV = SmemTransposeFp8_64x64<B_TOPK, D_V>;

  struct SharedMemoryPlan {
    union {
      array_aligned<fp8_t, cosize_v<SmemLayoutQ>> q;
      array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } q_o;
    array_aligned<fp8_t, cosize_v<SmemLayoutK>> k[NUM_STAGES];
    array_aligned<fp8_t, cosize_v<SmemLayoutVt>> vt[NUM_STAGES];
    array_aligned<fp8_t, cosize_v<SmemLayoutP>> p;
    bool valid[NUM_STAGES][B_TOPK];
    float col_max[4][H], col_sum[4][H];
    transac_bar_t q_ready, k_ready[NUM_STAGES], vt_ready[NUM_STAGES], free[NUM_STAGES];
  };

  static __device__ __forceinline__ int get_col(int slot, int tid) {
    return 8 * (slot / 2) + (tid % 4) * 2 + (slot & 1);
  }

  static __device__ __forceinline__ int get_col_slot(int i) {
    return (i % (H / 2)) / 4 * 2 + (i & 1);
  }

  // Match the KV-row packing used by the vectorized FP8 V transpose.
  static __device__ __forceinline__ int get_p_key_idx(int row) {
    return (row & ~14) | ((row & 6) << 1) | ((row & 8) >> 2);
  }

  static __device__ __forceinline__ void devfunc(const SparseMlaQ8Kv8PrefillParams& params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    const int tid = threadIdx.x % 128;
    const int wg = threadIdx.x / 128;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int query = blockIdx.x / HEAD_SPLITS;
    const int head_offset = (blockIdx.x % HEAD_SPLITS) * H;
    extern __shared__ char workspace[];
    auto& plan = *reinterpret_cast<SharedMemoryPlan*>(workspace);
    if (threadIdx.x == 0) {
      plan.q_ready.init(128);
      CUTE_UNROLL
      for (int b = 0; b < NUM_STAGES; ++b) {
        plan.k_ready[b].init(128);
        if constexpr (H == 32) plan.vt_ready[b].init(128);
        plan.free[b].init(128);
      }
      fence_barrier_init();
    }
    __syncthreads();
    const int length = HAVE_TOPK_LENGTH ? __ldg(params.topk_length + query) : params.topk;
    const int num_blocks = ku::ceil_div(length, B_TOPK);

    if (wg == 0) {
      cutlass::arch::warpgroup_reg_alloc<216>();
      const float kv_scale = __ldg(params.kv_scale_ptr);
      const float score_scale = __ldg(params.q_scale_ptr) * kv_scale * params.sm_scale_div_log2;
      Tensor sQ = make_tensor(make_smem_ptr(plan.q_o.q.data()), SmemLayoutQ{});
      Tensor sP = make_tensor(make_smem_ptr(plan.p.data()), SmemLayoutP{});
      Tensor rS = partition_fragment_C(TiledMMA{}, Shape<Int<B_TOPK>, Int<H>>{});
      Tensor rO = partition_fragment_C(TiledMMA{}, Shape<Int<D_V>, Int<H>>{});
      Tensor cO = TiledMMA{}.get_slice(tid).partition_C(make_identity_tensor(Shape<Int<D_V>, Int<H>>{}));
      float rM[NUM_COLS], rL[NUM_COLS];
      CUTE_UNROLL
      for (int s = 0; s < NUM_COLS; ++s) {
        rM[s] = -1e30f;
        rL[s] = 0.0f;
      }
      cute::fill(rO, 0.0f);
      plan.q_ready.wait(0);
      const int row0 = ::sm90::get_AorC_row_idx(0, tid);
      const int row1 = ::sm90::get_AorC_row_idx(1, tid);

      CUTE_NO_UNROLL
      for (int block = 0; block < num_blocks; ++block) {
        const int buf = block % NUM_STAGES;
        const int phase = (block / NUM_STAGES) & 1;
        plan.k_ready[buf].wait(phase);
        Tensor sK = make_tensor(make_smem_ptr(plan.k[buf].data()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(plan.vt[buf].data()), SmemLayoutVt{});
        CUTE_UNROLL
        for (int t = 0; t < D_QK / 64; ++t) {
          Tensor sKt = flat_divide(sK, Tile<Int<B_TOPK>, _64>{})(_, _, _0{}, t);
          Tensor sQt = flat_divide(sQ, Tile<Int<H>, _64>{})(_, _, _0{}, t);
          ::sm90::gemm_ss(t == 0, TiledMMA{}, sKt, sQt, rS, tid);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(rS);
        CUTE_UNROLL
        for (int s = 0; s < NUM_COLS; ++s) {
          const int i = (s / 2) * 4 + (s & 1);
          rS(i) = plan.valid[buf][row0] ? rS(i) * score_scale : -INFINITY;
          rS(i + 2) = plan.valid[buf][row1] ? rS(i + 2) * score_scale : -INFINITY;
          float m = max(rS(i), rS(i + 2));
          m = max(m, __shfl_xor_sync(0xffffffff, m, 4));
          m = max(m, __shfl_xor_sync(0xffffffff, m, 8));
          m = max(m, __shfl_xor_sync(0xffffffff, m, 16));
          if (lane < 4) plan.col_max[warp][get_col(s, tid)] = m;
        }
        NamedBarrier::arrive_and_wait(128, 0);
        float rescale[NUM_COLS];
        CUTE_UNROLL
        for (int s = 0; s < NUM_COLS; ++s) {
          const int head = get_col(s, tid);
          float m = rM[s];
          CUTE_UNROLL
          for (int w = 0; w < 4; ++w)
            m = max(m, plan.col_max[w][head]);
          rescale[s] = exp2f(rM[s] - m);
          rM[s] = m;
          const int i = (s / 2) * 4 + (s & 1);
          const float p0 = exp2f(rS(i) - m);
          const float p1 = exp2f(rS(i + 2) - m);
          rL[s] = rL[s] * rescale[s] + p0 + p1;
          const auto packed = pack_probabilities(p0, p1);
          sP(head, get_p_key_idx(row0)) = packed[0];
          sP(head, get_p_key_idx(row1)) = packed[1];
        }
        CUTE_UNROLL
        for (int i = 0; i < size(rO); ++i)
          rO(i) *= rescale[get_col_slot(i)];
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, 0);
        // V is physically transposed: FP8 WGMMA requires a K-major A operand.
        if constexpr (H == 32) plan.vt_ready[buf].wait(phase);
        ::sm90::gemm_ss(false, TiledMMA{}, sVt, sP, rO, tid);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(rO);
        // All reads of K, Vt and validity have finished before buffer reuse.
        plan.free[buf].arrive();
      }

      CUTE_UNROLL
      for (int s = 0; s < NUM_COLS; ++s) {
        float l = rL[s];
        l += __shfl_xor_sync(0xffffffff, l, 4);
        l += __shfl_xor_sync(0xffffffff, l, 8);
        l += __shfl_xor_sync(0xffffffff, l, 16);
        if (lane < 4) plan.col_sum[warp][get_col(s, tid)] = l;
      }
      NamedBarrier::arrive_and_wait(128, 0);
      float output_scale[NUM_COLS];
      CUTE_UNROLL
      for (int s = 0; s < NUM_COLS; ++s) {
        const int head = get_col(s, tid);
        float l = 0.0f;
        CUTE_UNROLL
        for (int w = 0; w < 4; ++w)
          l += plan.col_sum[w][head];
        if (warp == 0 && lane < 4) {
          params.max_logits[(int64_t)query * LOGICAL_H + head_offset + head] = l > 0 ? rM[s] * CUDART_LN2_F : -INFINITY;
          params.lse[(int64_t)query * LOGICAL_H + head_offset + head] =
              l > 0 ? (rM[s] + log2f(l)) * CUDART_LN2_F : -INFINITY;
        }
        float denom = l;
        if constexpr (HAVE_ATTN_SINK)
          denom += exp2f(__ldg(params.attn_sink + head_offset + head) * CUDART_L2E_F - rM[s]);
        output_scale[s] = l > 0 ? kv_scale / denom : 0.0f;
      }
      if constexpr (H == 32 || COALESCED_O) {
        Tensor sO = make_tensor(make_smem_ptr(plan.q_o.o.data()), SmemLayoutO{});
        CUTE_UNROLL
        for (int i = 0; i < size(rO); ++i) {
          const auto coord = cO(i);
          sO(get<1>(coord), get<0>(coord)) = bf16(rO(i) * output_scale[get_col_slot(i)]);
        }
        NamedBarrier::arrive_and_wait(128, 0);
        // The larger output amortizes the shared transpose and coalesced stores.
        CUTE_UNROLL
        for (int i = tid; i < H * (D_V / 8); i += 128) {
          const int head = i / (D_V / 8), dim = (i % (D_V / 8)) * 8;
          *reinterpret_cast<uint128_t*>(
              params.out + (int64_t)query * LOGICAL_H * D_V + (head_offset + head) * D_V + dim) =
              *reinterpret_cast<const uint128_t*>(&sO(head, dim));
        }
      } else {
        CUTE_UNROLL
        for (int i = 0; i < size(rO); ++i) {
          const auto coord = cO(i);
          params.out[((int64_t)query * LOGICAL_H + head_offset + get<1>(coord)) * D_V + get<0>(coord)] =
              bf16(rO(i) * output_scale[get_col_slot(i)]);
        }
      }
    } else {
      cutlass::arch::warpgroup_reg_dealloc<56>();
      const fp8_t* q = reinterpret_cast<const fp8_t*>(params.q) + (int64_t)query * params.stride_q_s_q +
                       (int64_t)head_offset * params.stride_q_h_q;
      Tensor sQ = make_tensor(make_smem_ptr(plan.q_o.q.data()), SmemLayoutQ{});
      const int64_t cache_policy = ::sm90::createpolicy_evict_last();
      CUTE_UNROLL
      for (int i = tid; i < H * (D_QK / 16); i += 128) {
        const int row = i / (D_QK / 16), col = i % (D_QK / 16) * 16;
        ::sm90::cp_async_cacheglobal_l2_prefetch_256B(
            q + row * params.stride_q_h_q + col, &sQ(row, col), true, cache_policy);
      }
      asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
      fence_view_async_shared();
      NamedBarrier::arrive_and_wait(128, 1);
      plan.q_ready.arrive();
      const fp8_t* kv = reinterpret_cast<const fp8_t*>(params.kv);
      const int* indices = params.indices + (int64_t)query * params.stride_indices_s_q;
      TransposeV transpose;
      CUTE_NO_UNROLL
      for (int block = 0; block < num_blocks; ++block) {
        const int buf = block % NUM_STAGES;
        plan.free[buf].wait(((block / NUM_STAGES) & 1) ^ 1);
        Tensor sK = make_tensor(make_smem_ptr(plan.k[buf].data()), SmemLayoutK{});
        // Four lanes load one 64-byte key tile; every key is owned by one group.
        CUTE_UNROLL
        for (int row = tid / 4; row < B_TOPK; row += 32) {
          const int slot = block * B_TOPK + row;
          const int token = slot < length ? __ldg(indices + slot) : -1;
          const bool valid = token >= 0 && token < params.s_kv;
          if (tid % 4 == 0) plan.valid[buf][row] = valid;
          CUTE_UNROLL
          for (int t = 0; t < D_QK / 64; ++t) {
            const int col = t * 64 + (tid % 4) * 16;
            ::sm90::cp_async_cacheglobal_l2_prefetch_256B(
                kv + (int64_t)(valid ? token : 0) * params.stride_kv_s_kv + col, &sK(row, col), valid, cache_policy);
          }
        }
        asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, 1);
        if constexpr (H == 32) plan.k_ready[buf].arrive();
        Tensor sV = as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(plan.k[buf].data()), typename TransposeV::SmemLayoutTransposeV{}));
        Tensor sVt = as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(plan.vt[buf].data()), typename TransposeV::SmemLayoutTransposeVt{}));
        CUTE_UNROLL
        for (int t = 0; t < D_V / 64; t += 2) {
          transpose.transpose_pair(
              flatten(sV(_, 0, t)), flatten(sVt(_, 0, t)), flatten(sV(_, 0, t + 1)), flatten(sVt(_, 0, t + 1)));
        }
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, 1);
        if constexpr (H == 32) {
          plan.vt_ready[buf].arrive();
        } else {
          plan.k_ready[buf].arrive();
        }
      }
    }
#else
    if (cute::thread0()) CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
#endif
  }

  static void run(const SparseMlaQ8Kv8PrefillParams& params) {
    KU_ASSERT(params.h_kv == 1 && params.h_q == LOGICAL_H);
    KU_ASSERT(params.topk > 0 && params.topk % 128 == 0);
    auto kernel = sparse_mla_q8kv8_etap_kernel<SparseMlaQ8Kv8EtapKernel>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    if constexpr (MIN_CTAS == 2)
      KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    kernel<<<params.s_q * HEAD_SPLITS, NUM_THREADS, smem_size, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, Kernel::MIN_CTAS)
    sparse_mla_q8kv8_etap_kernel(__grid_constant__ const SparseMlaQ8Kv8PrefillParams params) {
  Kernel::devfunc(params);
}

}  // namespace sglang

#include "headsplit.cuh"
#include "row_qk.cuh"
