// clang-format off
#include <tuple>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>
#include <flashinfer/vec_dtypes.cuh>
// clang-format on

#include "utils.h"

using namespace cute;

static constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_max_all(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    // Broadcast to the whole warp
    return __shfl_sync(0xffffffff, v, 0);
  }

// ---------------------------------------------------------------------------
// 1. SmallWarp kernel — warp‑local, no shared memory
//    • One warp handles one token.
//    • Eight tokens per 256‑thread CTA.
// ---------------------------------------------------------------------------
template <typename T, int kTokensPerCTA = 8, int kVecSize = 16>
__global__ void per_token_quant_fp8_small_kernel(
    const T* __restrict__ input,
    FP8_TYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int warp_id = threadIdx.x / kWarpSize;        // 0‑7  (8 warps)
  const int lane_id = threadIdx.x & (kWarpSize - 1);  // 0‑31
  const int token = blockIdx.x * kTokensPerCTA + warp_id;
  if (token >= num_tokens) return;

  // Global tensors for this token
  auto gmem_in = make_tensor(const_cast<T*>(input) + token * hidden_dim, make_shape(hidden_dim));
  auto gmem_out = make_tensor(output_q + token * hidden_dim, make_shape(hidden_dim));
  auto gmem_s = make_tensor(output_s + token, make_shape(1));

  //
  // Pass-1: compute max across whole token
  //
  float max_value = 0.f;
  using vec_t = flashinfer::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;
    input_vec.cast_load(gmem_in.data() + i * kVecSize);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
    }
  }

  float warp_max = warp_max_all(max_value);

  // Broadcast scale
  if (lane_id == 0) {
    gmem_s(0) = warp_max / FP8_E4M3_MAX;
  }
  const float scale_inv = (warp_max == 0.f) ? 0.f : 1.0f / (warp_max / FP8_E4M3_MAX);

  //
  // Pass-2: quantise and write back
  //
  for (int i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;
    input_vec.cast_load(gmem_in.data() + i * kVecSize);
    FP8_TYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(input_vec[j]) * scale_inv;
      val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
      val = nearbyintf(val);

#ifndef USE_ROCM
      output_arr[j] = static_cast<FP8_TYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }
    uint4 packed;
    memcpy(&packed, output_arr, 16);
    *reinterpret_cast<uint4*>(gmem_out.data() + i * kVecSize) = packed;
  }
}

// ---------------------------------------------------------------------------
// 2. LargeTile kernel — 8 KiB tile with **single‑buffer** CUTE TiledCopy
// ---------------------------------------------------------------------------
template <typename T, int kBlockSize = 256, int kVecSize = 16, int kTileBytes = 8192>
__global__ void per_token_quant_fp8_large_kernel(
    const T* __restrict__ input,
    FP8_TYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    int hidden_dim,
    int num_tokens) {
  const int token = blockIdx.x;
  if (token >= num_tokens) return;
  const int tid = threadIdx.x;

  // Gmem tensors
  auto g_in = make_tensor(const_cast<T*>(input) + token * hidden_dim, make_shape(hidden_dim));
  auto g_out = make_tensor(output_q + token * hidden_dim, make_shape(hidden_dim));
  auto g_s = make_tensor(output_s + token, make_shape(1));

  // Shared memory tile (8 KiB)
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* tile = reinterpret_cast<T*>(smem_raw);
  using AtomT = cutlass::uint128_t;  // 16  B
  constexpr int kTileElems = kTileBytes / sizeof(T);
  constexpr int kAtomElems = sizeof(AtomT) / sizeof(T);  // 4 for float
  static_assert(kTileBytes % sizeof(AtomT) == 0, "");
  constexpr int kAtomsPerTile = kTileBytes / sizeof(AtomT);  // 8192 / 16 = 512

  auto smem_tensor = cute::make_tensor(
      reinterpret_cast<AtomT*>(tile), cute::make_shape(cute::Int<kAtomsPerTile>{})  // vec4 (16 B)
  );

  float local_max = 0.0f;
  using vec_t = flashinfer::vec_t<T, kVecSize>;
  //------------------------------------------------------------------
  // Pass‑1 : compute max
  //------------------------------------------------------------------

  for (int g = 0; g < hidden_dim; g += kTileElems) {
    const int chunk = min(kTileElems, hidden_dim - g);

    if (chunk == kTileElems) {
      // full tile: use TiledCopy + cp.async
      auto src_tile = cute::make_tensor(
          reinterpret_cast<const AtomT*>(g_in.data() + g), cute::make_shape(cute::Int<kAtomsPerTile>{}));
      cute::copy(src_tile, smem_tensor);  // AutoCopyAsync → cp.async
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();
    } else {
      // tail: use vector copy
      for (int idx = tid * kVecSize; idx < chunk; idx += kBlockSize * kVecSize) {
        vec_t v;
        v.cast_load(g_in.data() + g + idx);
        v.cast_store(tile + idx);
      }
      __syncthreads();
    }

    for (int idx = tid * kVecSize; idx < chunk; idx += kBlockSize * kVecSize) {
      vec_t v;
      v.cast_load(tile + idx);
#pragma unroll
      for (int j = 0; j < kVecSize; ++j)
        local_max = fmaxf(local_max, fabsf(float(v[j])));
    }
    __syncthreads();
  }

  float token_max = blockReduceMax(local_max);
  __shared__ float scale;
  if (tid == 0) {
    scale = token_max / FP8_E4M3_MAX;
    g_s(0) = scale;
  }
  __syncthreads();
  const float inv_scale = (scale == 0.0f) ? 0.0f : 1.0f / scale;

  //------------------------------------------------------------------
  // Pass‑2 : quantize & write
  //------------------------------------------------------------------
  for (int g = 0; g < hidden_dim; g += kTileElems) {
    const int chunk = min(kTileElems, hidden_dim - g);

    if (chunk == kTileElems) {
      auto src_tile = cute::make_tensor(
          reinterpret_cast<const AtomT*>(g_in.data() + g), cute::make_shape(cute::Int<kAtomsPerTile>{}));
      cute::copy(src_tile, smem_tensor);
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();
    } else {
      for (int idx = tid * kVecSize; idx < chunk; idx += kBlockSize * kVecSize) {
        vec_t v;
        v.cast_load(g_in.data() + g + idx);
        v.cast_store(tile + idx);
      }
      __syncthreads();
    }

    for (int idx = tid * kVecSize; idx < chunk; idx += kBlockSize * kVecSize) {
      vec_t v;
      v.cast_load(tile + idx);
      FP8_TYPE q[kVecSize];
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        float f = static_cast<float>(v[j]) * inv_scale;
        f = fminf(fmaxf(f, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        q[j] = static_cast<FP8_TYPE>(f);
      }
      *reinterpret_cast<uint4*>(g_out.data() + g + idx) = *reinterpret_cast<const uint4*>(q);
    }
    __syncthreads();
  }
}

void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);

  const auto input_sizes = input.sizes();
  const int64_t num_tokens = input_sizes[0];
  const int64_t hidden_dim = input_sizes[1];

  TORCH_CHECK(hidden_dim % 16 == 0, "Hidden dimension must be divisible by 16, but got ", hidden_dim);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    constexpr int TOKENS_PER_CTA = 8;
    constexpr int THREADS = TOKENS_PER_CTA * WARP_SIZE;  // 256
    dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
    dim3 block(THREADS);

    per_token_quant_fp8_small_kernel<scalar_t, TOKENS_PER_CTA, 16><<<grid, block, 0, stream>>>(
        static_cast<const scalar_t*>(input.data_ptr()),
        static_cast<FP8_TYPE*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);

    // if (hidden_dim <= 512) {
    //   // ---------------- small-path ----------------
    //   constexpr int TOKENS_PER_CTA = 8;
    //   constexpr int THREADS = TOKENS_PER_CTA * WARP_SIZE;  // 256
    //   dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
    //   dim3 block(THREADS);

    //   per_token_quant_fp8_small_kernel<scalar_t, TOKENS_PER_CTA, 16><<<grid, block, 0, stream>>>(
    //       static_cast<const scalar_t*>(input.data_ptr()),
    //       static_cast<FP8_TYPE*>(output_q.data_ptr()),
    //       static_cast<float*>(output_s.data_ptr()),
    //       hidden_dim,
    //       num_tokens);
    // } else {
    //   // ---------------- large-path ----------------
    //   constexpr int BLOCK_SIZE = 256;
    //   constexpr int SMEM_BYTES = 8192;
    //   dim3 grid(num_tokens), block(BLOCK_SIZE);

    //   per_token_quant_fp8_large_kernel<scalar_t, BLOCK_SIZE, 16, SMEM_BYTES><<<grid, block, SMEM_BYTES, stream>>>(
    //       static_cast<const scalar_t*>(input.data_ptr()),
    //       static_cast<FP8_TYPE*>(output_q.data_ptr()),
    //       static_cast<float*>(output_s.data_ptr()),
    //       hidden_dim,
    //       num_tokens);
    // }
    return true;
  });
}
