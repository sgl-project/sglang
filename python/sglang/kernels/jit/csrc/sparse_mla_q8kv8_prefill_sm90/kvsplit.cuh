/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include "kernel.cuh"
#include <algorithm>
#include <optional>

namespace sglang {

template <typename Kernel>
__global__ void
sparse_mla_q8kv8_kvsplit_kernel(__grid_constant__ const SparseMlaQ8Kv8PrefillParams params, int initial_registers);

// Transpose both GEMMs so heads occupy WGMMA's narrow N dimension:
// K Q^T -> [64, H], V^T P^T -> [512, H]. Softmax still reduces over keys.
template <int D_QK, int H, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK>
struct SparseMlaQ8Kv8KVSplitKernel {
  static_assert(H == 8 || H == 16 || H == 32);
  static constexpr int NUM_THREADS = 256;
  static constexpr int MIN_CTAS = 1;
  static constexpr int NUM_STAGES = 2;
  static constexpr bool COALESCED_O = true;
  static constexpr int D_V = 512;
  static constexpr int B_TOPK = 64;
  static constexpr int NUM_COLS = H / 4;
  using fp8_t = cutlass::float_e4m3_t;

  template <int M, int K>
  using SmemLayout = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<M>, Int<K>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
  using SmemLayoutQ = SmemLayout<H, D_QK>;
  using SmemLayoutK = SmemLayout<B_TOPK, D_QK>;
  using SmemLayoutVt = SmemLayout<D_V, B_TOPK>;
  using SmemLayoutP = SmemLayout<H, B_TOPK>;
  using SmemLayoutSplitO = decltype(coalesce(
      tile_to_shape(GMMA::Layout_K_SW128_Atom<float>{}, Shape<Int<H>, Int<D_V>>{}, Step<_1, _2>{}), Shape<_1, _1>{}));
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
      array_aligned<float, cosize_v<SmemLayoutSplitO>> o;
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

  static __device__ __forceinline__ void devfunc(const SparseMlaQ8Kv8PrefillParams& params, int initial_registers) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    const int tid = threadIdx.x % 128;
    const int wg = threadIdx.x / 128;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int query = blockIdx.x / params.kvsplit_count, part = blockIdx.x % params.kvsplit_count;
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
    const int storage_blocks = ku::ceil_div(params.topk, B_TOPK);
    const int first_block = int((int64_t)storage_blocks * part / params.kvsplit_count);
    const int end_block = int((int64_t)storage_blocks * (part + 1) / params.kvsplit_count);
    const int last_block = end_block < num_blocks ? end_block : num_blocks;
    const int part_blocks = last_block > first_block ? last_block - first_block : 0;

    if (wg == 0) {
      if (216 >= initial_registers)
        cutlass::arch::warpgroup_reg_alloc<216>();
      else
        cutlass::arch::warpgroup_reg_dealloc<216>();
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
      for (int block = 0; block < part_blocks; ++block) {
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
          sP(head, get_p_key_idx(row0)) = fp8_t(p0);
          sP(head, get_p_key_idx(row1)) = fp8_t(p1);
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
      CUTE_UNROLL
      for (int s = 0; s < NUM_COLS; ++s) {
        const int head = get_col(s, tid);
        float l = 0.0f;
        CUTE_UNROLL
        for (int w = 0; w < 4; ++w)
          l += plan.col_sum[w][head];
        if (warp == 0 && lane < 4) {
          const int64_t offset = ((int64_t)query * params.kvsplit_count + part) * H + head;
          params.kvsplit_m[offset] = l > 0 ? rM[s] : -INFINITY;
          params.kvsplit_l[offset] = l;
        }
      }
      Tensor sO = make_tensor(make_smem_ptr(plan.q_o.o.data()), SmemLayoutSplitO{});
      CUTE_UNROLL
      for (int i = 0; i < size(rO); ++i) {
        const auto coord = cO(i);
        sO(get<1>(coord), get<0>(coord)) = rO(i);
      }
      NamedBarrier::arrive_and_wait(128, 0);
      CUTE_UNROLL
      for (int i = tid; i < H * (D_V / 4); i += 128) {
        const int head = i / (D_V / 4), dim = (i % (D_V / 4)) * 4;
        const int64_t offset = (((int64_t)query * params.kvsplit_count + part) * H + head) * D_V + dim;
        *reinterpret_cast<float4*>(params.kvsplit_out + offset) = *reinterpret_cast<const float4*>(&sO(head, dim));
      }
    } else {
      if (56 >= initial_registers)
        cutlass::arch::warpgroup_reg_alloc<56>();
      else
        cutlass::arch::warpgroup_reg_dealloc<56>();
      const fp8_t* q = reinterpret_cast<const fp8_t*>(params.q) + (int64_t)query * params.stride_q_s_q;
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
      for (int block = 0; block < part_blocks; ++block) {
        const int buf = block % NUM_STAGES;
        plan.free[buf].wait(((block / NUM_STAGES) & 1) ^ 1);
        Tensor sK = make_tensor(make_smem_ptr(plan.k[buf].data()), SmemLayoutK{});
        // Four lanes load one 64-byte key tile; every key is owned by one group.
        CUTE_UNROLL
        for (int row = tid / 4; row < B_TOPK; row += 32) {
          const int slot = (first_block + block) * B_TOPK + row;
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
    KU_ASSERT(params.h_kv == 1 && params.h_q == H);
    KU_ASSERT(params.topk > 0 && params.topk % 128 == 0);
    auto kernel = sparse_mla_q8kv8_kvsplit_kernel<SparseMlaQ8Kv8KVSplitKernel>;
    cudaFuncAttributes attributes;
    KU_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    KU_ASSERT(216 + 56 <= 2 * attributes.numRegs);
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    if constexpr (MIN_CTAS == 2)
      KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    kernel<<<params.s_q * params.kvsplit_count, NUM_THREADS, smem_size, params.stream>>>(params, attributes.numRegs);
    KU_CHECK_KERNEL_LAUNCH();
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, Kernel::MIN_CTAS)
    sparse_mla_q8kv8_kvsplit_kernel(__grid_constant__ const SparseMlaQ8Kv8PrefillParams params, int initial_registers) {
  Kernel::devfunc(params, initial_registers);
}

template <int H, bool HAVE_ATTN_SINK>
__global__ void sparse_mla_q8kv8_kvsplit_merge(__grid_constant__ const SparseMlaQ8Kv8PrefillParams params) {
  const int query = blockIdx.x / H, head = blockIdx.x % H;
  __shared__ float weights[16], output_scale;
  if (threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (int part = 0; part < params.kvsplit_count; ++part) {
      const int64_t stat = ((int64_t)query * params.kvsplit_count + part) * H + head;
      if (params.kvsplit_l[stat] > 0.0f) maximum = fmaxf(maximum, params.kvsplit_m[stat]);
    }
    float sum = 0.0f;
    for (int part = 0; part < params.kvsplit_count; ++part) {
      const int64_t stat = ((int64_t)query * params.kvsplit_count + part) * H + head;
      const float l = params.kvsplit_l[stat];
      const float weight = l > 0.0f ? exp2f(params.kvsplit_m[stat] - maximum) : 0.0f;
      weights[part] = weight;
      sum += weight * l;
    }
    float denominator = sum;
    if constexpr (HAVE_ATTN_SINK) {
      if (sum > 0.0f) denominator += exp2f(params.attn_sink[head] * CUDART_L2E_F - maximum);
    }
    output_scale = sum > 0.0f ? *params.kv_scale_ptr / denominator : 0.0f;
    const int64_t stat = (int64_t)query * H + head;
    params.max_logits[stat] = sum > 0.0f ? maximum * CUDART_LN2_F : -INFINITY;
    params.lse[stat] = sum > 0.0f ? (maximum + log2f(sum)) * CUDART_LN2_F : -INFINITY;
  }
  __syncthreads();
  for (int dim = threadIdx.x; dim < 512; dim += blockDim.x) {
    float output = 0.0f;
    for (int part = 0; part < params.kvsplit_count; ++part) {
      const int64_t offset = (((int64_t)query * params.kvsplit_count + part) * H + head) * 512 + dim;
      output += params.kvsplit_out[offset] * weights[part];
    }
    params.out[((int64_t)query * H + head) * 512 + dim] = cutlass::bfloat16_t(output * output_scale);
  }
}

template <int D_QK, int H, bool HAVE_TOPK_LENGTH, bool HAVE_ATTN_SINK, int MAX_SPLITS>
static bool try_sparse_mla_q8kv8_kvsplit(const SparseMlaQ8Kv8PrefillParams& params) {
  static thread_local int cached_device = -1, cached_sms = 0;
  int device;
  KU_CUDA_CHECK(cudaGetDevice(&device));
  if (cached_device != device) {
    KU_CUDA_CHECK(cudaDeviceGetAttribute(&cached_sms, cudaDevAttrMultiProcessorCount, device));
    cached_device = device;
  }
  if (cached_sms < 128 || params.s_q <= 0 || params.s_q > (H == 32 ? cached_sms / 2 : 64) || params.s_kv < 256 ||
      params.topk < 0)
    return false;
  int budget = std::min(MAX_SPLITS, std::min(cached_sms / params.s_q, std::max(1, ku::ceil_div(params.topk, 64) / 2)));
  int parts = 1;
  while (parts * 2 <= budget)
    parts *= 2;
  if (parts == 1) return false;
  const size_t stats = (size_t)params.s_q * parts * H;
  float* workspace = nullptr;
  std::optional<tvm::ffi::Tensor> owner;
  if (TVMFFIEnvGetDLPackManagedTensorAllocator() != nullptr &&
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device)) == params.stream) {
    const int64_t elements = (int64_t)stats * (512 + 2);
    owner.emplace(
        tvm::ffi::Tensor::FromEnvAlloc(
            TVMFFIEnvTensorAlloc, tvm::ffi::Shape{elements}, DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCUDA, device}));
    workspace = static_cast<float*>(owner->data_ptr());
  } else {
    KU_CUDA_CHECK(
        cudaMallocAsync(reinterpret_cast<void**>(&workspace), stats * (512 + 2) * sizeof(float), params.stream));
  }
  auto partial = params;
  partial.kvsplit_count = parts;
  partial.kvsplit_out = workspace;
  partial.kvsplit_m = workspace + stats * 512;
  partial.kvsplit_l = partial.kvsplit_m + stats;
  try {
    SparseMlaQ8Kv8KVSplitKernel<D_QK, H, HAVE_TOPK_LENGTH, HAVE_ATTN_SINK>::run(partial);
    sparse_mla_q8kv8_kvsplit_merge<H, HAVE_ATTN_SINK><<<params.s_q * H, 128, 0, params.stream>>>(partial);
    KU_CHECK_KERNEL_LAUNCH();
  } catch (...) {
    if (!owner) cudaFreeAsync(workspace, params.stream);
    throw;
  }
  if (!owner) KU_CUDA_CHECK(cudaFreeAsync(workspace, params.stream));
  return true;
}

}  // namespace sglang
