/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "etap.cuh"

namespace sglang {

template <typename Kernel>
__global__ void sparse_mla_q8kv8_row_qk_kernel(__grid_constant__ const SparseMlaQ8Kv8PrefillParams params);

template <int D_QK, int H, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
struct SparseMlaQ8Kv8RowQKKernel {
  static_assert(H == 32);
  using Parent = SparseMlaQ8Kv8EtapKernel<D_QK, H, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>;
  static constexpr int NUM_THREADS = 256, NUM_STAGES = Parent::NUM_STAGES;
  static constexpr int D_V = 512, B_TOPK = 64, QK_HEADS = 64, NUM_COLS = H / 4;
  static constexpr int CONSUMER_REGS = 216, PRODUCER_REGS = 56;
  static constexpr bool DECOUPLED = false, PACKED_P = false;
  static constexpr int MIN_CTAS = NUM_STAGES == 1 && CONSUMER_REGS + PRODUCER_REGS <= 256 ? 2 : 1;
  using fp8_t = cutlass::float_e4m3_t;
  using SmemLayoutQ = typename Parent::template SmemLayout<QK_HEADS, D_QK>;
  using SmemLayoutK = typename Parent::SmemLayoutK;
  using SmemLayoutVt = typename Parent::SmemLayoutVt;
  using SmemLayoutP = typename Parent::SmemLayoutP;
  using SmemLayoutO = typename Parent::SmemLayoutO;
  using TransposeV = typename Parent::TransposeV;
  using QkMMA = decltype(make_tiled_mma(GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));
  using PvMMA = typename Parent::TiledMMA;

  struct SharedMemoryPlan {
    union {
      array_aligned<fp8_t, cosize_v<SmemLayoutQ>> q;
      array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } q_o;
    array_aligned<fp8_t, cosize_v<SmemLayoutK>> k[NUM_STAGES];
    array_aligned<fp8_t, cosize_v<SmemLayoutVt>> vt[NUM_STAGES];
    array_aligned<fp8_t, cosize_v<SmemLayoutP>> p;
    bool valid[NUM_STAGES][B_TOPK];
    float rescale[H], output_scale[H];
    transac_bar_t q_ready, k_ready[NUM_STAGES], vt_ready[NUM_STAGES];
    transac_bar_t free[NUM_STAGES], k_free[NUM_STAGES];
  };

  static __device__ __forceinline__ void devfunc(const SparseMlaQ8Kv8PrefillParams& params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    const int tid = threadIdx.x % 128, wg = threadIdx.x / 128;
    const int lane = tid % 32, query = blockIdx.x;
    extern __shared__ char workspace[];
    auto& plan = *reinterpret_cast<SharedMemoryPlan*>(workspace);
    if (threadIdx.x == 0) {
      plan.q_ready.init(128);
      CUTE_UNROLL
      for (int b = 0; b < NUM_STAGES; ++b) {
        plan.k_ready[b].init(128);
        plan.vt_ready[b].init(128);
        plan.free[b].init(128);
        plan.k_free[b].init(128);
      }
      fence_barrier_init();
    }
    __syncthreads();
    const int length = HAVE_TOPK_LENGTH ? __ldg(params.topk_length + query) : params.topk;
    const int blocks = ku::ceil_div(length, B_TOPK);
    if (wg == 0) {
      cutlass::arch::warpgroup_reg_alloc<CONSUMER_REGS>();
      const float kv_scale = __ldg(params.kv_scale_ptr);
      const float score_scale = __ldg(params.q_scale_ptr) * kv_scale * params.sm_scale_div_log2;
      Tensor sQ = make_tensor(make_smem_ptr(plan.q_o.q.data()), SmemLayoutQ{});
      Tensor sP = make_tensor(make_smem_ptr(plan.p.data()), SmemLayoutP{});
      Tensor rS = partition_fragment_C(QkMMA{}, Shape<Int<QK_HEADS>, Int<B_TOPK>>{});
      Tensor rO = partition_fragment_C(PvMMA{}, Shape<Int<D_V>, Int<H>>{});
      Tensor cO = PvMMA{}.get_slice(tid).partition_C(make_identity_tensor(Shape<Int<D_V>, Int<H>>{}));
      float rM[2] = {-1e30f, -1e30f}, rL[2] = {0.0f, 0.0f};
      cute::fill(rO, 0.0f);
      plan.q_ready.wait(0);
      CUTE_NO_UNROLL
      for (int block = 0; block < blocks; ++block) {
        const int buf = block % NUM_STAGES, phase = (block / NUM_STAGES) & 1;
        plan.k_ready[buf].wait(phase);
        Tensor sK = make_tensor(make_smem_ptr(plan.k[buf].data()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(plan.vt[buf].data()), SmemLayoutVt{});
        CUTE_UNROLL
        for (int t = 0; t < D_QK / 64; ++t) {
          Tensor sQt = flat_divide(sQ, Tile<Int<QK_HEADS>, _64>{})(_, _, _0{}, t);
          Tensor sKt = flat_divide(sK, Tile<Int<B_TOPK>, _64>{})(_, _, _0{}, t);
          ::sm90::gemm_ss(t == 0, QkMMA{}, sQt, sKt, rS, tid);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(rS);
        CUTE_UNROLL
        for (int row = 0; row < 2; ++row) {
          const int head = ::sm90::get_AorC_row_idx(row, tid);
          if (head < H) {
            float cur_max = -INFINITY;
            CUTE_UNROLL
            for (int i = row * 2; i < size(rS); i += 4) {
              const int key = (i / 4) * 8 + (lane % 4) * 2;
              rS(i) = plan.valid[buf][key] ? rS(i) * score_scale : -INFINITY;
              rS(i + 1) = plan.valid[buf][key + 1] ? rS(i + 1) * score_scale : -INFINITY;
              cur_max = max(cur_max, max(rS(i), rS(i + 1)));
            }
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
            const float m = max(rM[row], cur_max), alpha = exp2f(rM[row] - m);
            rM[row] = m;
            float sum = 0.0f;
            CUTE_UNROLL
            for (int i = row * 2; i < size(rS); i += 4) {
              const int key = (i / 4) * 8 + (lane % 4) * 2;
              const float p0 = exp2f(rS(i) - m), p1 = exp2f(rS(i + 1) - m);
              sum += p0 + p1;
              const auto converted = Parent::pack_probabilities(p0, p1);
              const fp8_t fp0 = converted[0], fp1 = converted[1];
              if constexpr (PACKED_P) {
                const uint16_t pair = uint16_t(fp0.storage) | (uint16_t(fp1.storage) << 8);
                *reinterpret_cast<uint16_t*>(&sP(head, Parent::get_p_key_idx(key))) = pair;
              } else {
                sP(head, Parent::get_p_key_idx(key)) = fp0;
                sP(head, Parent::get_p_key_idx(key + 1)) = fp1;
              }
            }
            rL[row] = rL[row] * alpha + sum;
            if (lane % 4 == 0) plan.rescale[head] = alpha;
          }
        }
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, 0);
        if constexpr (DECOUPLED) plan.k_free[buf].arrive();
        float scales[NUM_COLS];
        CUTE_UNROLL
        for (int s = 0; s < NUM_COLS; ++s)
          scales[s] = plan.rescale[Parent::get_col(s, tid)];
        CUTE_UNROLL
        for (int i = 0; i < size(rO); ++i)
          rO(i) *= scales[Parent::get_col_slot(i)];
        plan.vt_ready[buf].wait(phase);
        ::sm90::gemm_ss(false, PvMMA{}, sVt, sP, rO, tid);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(rO);
        // All readers must finish before the next iteration overwrites rescale and P.
        NamedBarrier::arrive_and_wait(128, 0);
        plan.free[buf].arrive();
      }
      CUTE_UNROLL
      for (int row = 0; row < 2; ++row) {
        const int head = ::sm90::get_AorC_row_idx(row, tid);
        float l = rL[row];
        l += __shfl_xor_sync(0xffffffff, l, 1);
        l += __shfl_xor_sync(0xffffffff, l, 2);
        if (head < H && lane % 4 == 0) {
          float denom = l;
          if constexpr (HAVE_ATTN_SINK) denom += exp2f(__ldg(params.attn_sink + head) * CUDART_L2E_F - rM[row]);
          plan.output_scale[head] = l > 0 ? kv_scale / denom : 0.0f;
          params.max_logits[(int64_t)query * H + head] = l > 0 ? rM[row] * CUDART_LN2_F : -INFINITY;
          params.lse[(int64_t)query * H + head] = l > 0 ? (rM[row] + log2f(l)) * CUDART_LN2_F : -INFINITY;
        }
      }
      NamedBarrier::arrive_and_wait(128, 0);
      float output_scales[NUM_COLS];
      CUTE_UNROLL
      for (int s = 0; s < NUM_COLS; ++s)
        output_scales[s] = plan.output_scale[Parent::get_col(s, tid)];
      Tensor sO = make_tensor(make_smem_ptr(plan.q_o.o.data()), SmemLayoutO{});
      CUTE_UNROLL
      for (int i = 0; i < size(rO); ++i) {
        const auto coord = cO(i);
        sO(get<1>(coord), get<0>(coord)) = bf16(rO(i) * output_scales[Parent::get_col_slot(i)]);
      }
      NamedBarrier::arrive_and_wait(128, 0);
      CUTE_UNROLL
      for (int i = tid; i < H * (D_V / 8); i += 128) {
        const int head = i / (D_V / 8), dim = (i % (D_V / 8)) * 8;
        *reinterpret_cast<uint128_t*>(params.out + (int64_t)query * H * D_V + head * D_V + dim) =
            *reinterpret_cast<const uint128_t*>(&sO(head, dim));
      }
    } else {
      cutlass::arch::warpgroup_reg_dealloc<PRODUCER_REGS>();
      const fp8_t* q = reinterpret_cast<const fp8_t*>(params.q) + (int64_t)query * params.stride_q_s_q;
      Tensor sQ = make_tensor(make_smem_ptr(plan.q_o.q.data()), SmemLayoutQ{});
      const int64_t policy = ::sm90::createpolicy_evict_last();
      CUTE_UNROLL
      for (int i = tid; i < QK_HEADS * (D_QK / 16); i += 128) {
        const int row = i / (D_QK / 16), col = (i % (D_QK / 16)) * 16;
        ::sm90::cp_async_cacheglobal_l2_prefetch_256B(
            q + (row < H ? row : 0) * params.stride_q_h_q + col, &sQ(row, col), row < H, policy);
      }
      asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
      fence_view_async_shared();
      NamedBarrier::arrive_and_wait(128, 1);
      plan.q_ready.arrive();
      const fp8_t* kv = reinterpret_cast<const fp8_t*>(params.kv);
      const int* indices = params.indices + (int64_t)query * params.stride_indices_s_q;
      TransposeV transpose;
      CUTE_NO_UNROLL
      for (int block = 0; block < blocks; ++block) {
        const int buf = block % NUM_STAGES, phase = (block / NUM_STAGES) & 1;
        if constexpr (DECOUPLED)
          plan.k_free[buf].wait(phase ^ 1);
        else
          plan.free[buf].wait(phase ^ 1);
        Tensor sK = make_tensor(make_smem_ptr(plan.k[buf].data()), SmemLayoutK{});
        CUTE_UNROLL
        for (int row = tid / 4; row < B_TOPK; row += 32) {
          const int slot = block * B_TOPK + row, token = slot < length ? __ldg(indices + slot) : -1;
          const bool valid = token >= 0 && token < params.s_kv;
          if (tid % 4 == 0) plan.valid[buf][row] = valid;
          CUTE_UNROLL
          for (int t = 0; t < D_QK / 64; ++t) {
            const int col = t * 64 + (tid % 4) * 16;
            ::sm90::cp_async_cacheglobal_l2_prefetch_256B(
                kv + (int64_t)(valid ? token : 0) * params.stride_kv_s_kv + col, &sK(row, col), valid, policy);
          }
        }
        asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, 1);
        plan.k_ready[buf].arrive();
        if constexpr (DECOUPLED) plan.free[buf].wait(phase ^ 1);
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
        plan.vt_ready[buf].arrive();
      }
    }
#else
    if (cute::thread0()) CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
#endif
  }

  static void run(const SparseMlaQ8Kv8PrefillParams& params) {
    KU_ASSERT(params.h_kv == 1 && params.h_q == H);
    KU_ASSERT(params.topk > 0 && params.topk % 128 == 0);
    auto kernel = sparse_mla_q8kv8_row_qk_kernel<SparseMlaQ8Kv8RowQKKernel>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    if constexpr (MIN_CTAS == 2)
      KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, Kernel::MIN_CTAS)
    sparse_mla_q8kv8_row_qk_kernel(__grid_constant__ const SparseMlaQ8Kv8PrefillParams params) {
  Kernel::devfunc(params);
}

}  // namespace sglang
