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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// No support for async
#else

__device__ inline void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}
#endif


// ---------------------------------------------------------------------------
// 1. Warp‑local, no shared memory
//    • One warp handles one token.
//    • Eight tokens per 256‑thread CTA.
// ---------------------------------------------------------------------------
template <typename T, int kTokensPerCTA = 8, int kVecSize = 16>
__global__ void per_token_quant_fp8_kernel(
    const T* __restrict__ input,
    FP8_TYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int warp_id = threadIdx.x / kWarpSize;        // 0‑7  (8 warps)
  const int lane_id = threadIdx.x & (kWarpSize - 1);  // 0‑31
  const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
  if (token_id >= num_tokens) return;

  // Global tensors for this token
  const T* token_input = input + token_id * hidden_dim;
  FP8_TYPE* token_output = output_q + token_id * hidden_dim;
  float* token_scale = output_s + token_id;

  //
  // Pass-1: Perform a warp reduce to find the max_value of a token's hidden_dim
  //
  float max_value = 0.f;
  using vec_t = flashinfer::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
    }
  }

  float warp_max = warpReduceMax(max_value);

  __shared__ float scale;
  scale = warp_max / FP8_E4M3_MAX;
  // Broadcast scale
  if (lane_id == 0) {
    token_scale[0] = scale;
  }
  float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  //
  // Pass-2: quantize and write back
  //
  for (int i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);
    FP8_TYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(input_vec[j]) * scale_inv;
      val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);

#ifndef USE_ROCM
      output_arr[j] = static_cast<FP8_TYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }
    *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
  }
}

// ---------------------------------------------------------------------------
// 2.  Baseline kernel (1 token / CTA, CUB block reduce)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void per_token_quant_fp8_small_batch_kernel(
    const T* __restrict__ input,
    FP8_TYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  const T* token_input = input + token_idx * hidden_dim;
  FP8_TYPE* token_output = output_q + token_idx * hidden_dim;

  float max_value = 0.0f;

  // We want to store 128 bits of data at a time. 16 = 128 / 8 bits
  // Load is already vectorized, so 16 elements work for T.
  const uint32_t VEC_SIZE = 16;
  using vec_t = flashinfer::vec_t<T, VEC_SIZE>;
  const int32_t num_vec_elems = hidden_dim / VEC_SIZE;

  // Find max using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * VEC_SIZE);

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  max_value = blockReduceMax(max_value);

  __shared__ float scale;
  if (tid == 0) {
    scale = max_value / FP8_E4M3_MAX;
    output_s[token_idx] = scale;
  }
  __syncthreads();

  const float scale_inv = 1.0f / scale;

  // Quantize using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * VEC_SIZE);

    FP8_TYPE output_arr[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#ifndef USE_ROCM
      output_arr[j] = static_cast<FP8_TYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }

    *(uint4*)(token_output + i * VEC_SIZE) = *(uint4*)output_arr;
  }
}

// ---------------------------------------------------------------------------
// 3. LargeTile kernel — 8 KiB tile with **single‑buffer** CUTE TiledCopy
// ---------------------------------------------------------------------------
template <typename T, int kBlockSize = 256, int kVecSize = 16, int kTileBytes = 8192, int kStages = 2>
__launch_bounds__(kBlockSize) __global__ void per_token_quant_fp8_large_kernel(
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

  // Shared memory tile (8 KiB)
  extern __shared__ __align__(16) unsigned char smem_raw[];
  using AtomT = cutlass::uint128_t;  // 16  B
  constexpr int kTileElems = kTileBytes / sizeof(T);
  constexpr int kAtomsPerTile = kTileBytes / sizeof(AtomT);  // 4 for float
  AtomT* smem = reinterpret_cast<AtomT*>(smem_raw);

  int g_offset = 0;
#pragma unroll
  for (int stage = 0; stage < kStages; ++stage, g_offset += kTileElems) {
    if (g_offset < hidden_dim) {
      auto src =
          cute::make_tensor(reinterpret_cast<const AtomT*>(g_in.data() + g_offset),
          cute::make_shape(cute::Int<kAtomsPerTile>{}));
      auto dst = cute::make_tensor(
          smem + stage * kAtomsPerTile,
          cute::make_shape(cute::Int<kAtomsPerTile>{}));
      cute::copy(src, dst);
    }
  }
  cp_async_commit_group();
  cp_async_wait_group<1>();

  //--------------------------------------------------------------------
  // Pass-1: reduce max + prefetch next tile
  //--------------------------------------------------------------------
  float local_max = 0.f;
  for (int g_base = 0; g_base < hidden_dim; g_base += kTileElems) {
    int stage = (g_base / kTileElems) % kStages;
    AtomT* tile_atom = smem + stage * kAtomsPerTile;
    T* tile = reinterpret_cast<T*>(tile_atom);

    // update max
    for (int idx = tid * kVecSize; idx <= kTileElems - kVecSize && g_base + idx < hidden_dim; idx += kBlockSize * kVecSize) {
      flashinfer::vec_t<T, kVecSize> v;
      v.cast_load(tile + idx);
#pragma unroll
      for (int i = 0; i < kVecSize; ++i)
        local_max = fmaxf(local_max, fabsf(float(v[i])));
    }

    // prefetch next tile to double buffer
    int g_next = g_base + kStages * kTileElems;
    if (g_next < hidden_dim) {
      auto src = cute::make_tensor(
          reinterpret_cast<const AtomT*>(g_in.data() + g_next), cute::make_shape(cute::Int<kAtomsPerTile>{}));
      auto dst = cute::make_tensor(
        smem + ((stage ^ 1) * kAtomsPerTile),
        cute::make_shape(cute::Int<kAtomsPerTile>{}));
      cute::copy(src, dst);
    }
    cp_async_commit_group();
    cp_async_wait_group<1>();
    __syncthreads();
  }

  // CTA reduce max & broadcast scale
  float token_max = blockReduceMax(local_max);

  __shared__ float scale;
  if (tid == 0) {
    scale = token_max / FP8_E4M3_MAX;
    output_s[token] = scale;
  }
  __syncthreads();
  const float inv_scale = scale ? 1.f / scale : 0.f;

  //--------------------------------------------------------------------
  // Pass-2: quantize write back（reuse data in SMEM in double buffer）
  //--------------------------------------------------------------------
  for (int g_base = 0; g_base < hidden_dim; g_base += kTileElems) {
    int stage = (g_base / kTileElems) % kStages;
    AtomT* tile_atom = smem + stage * kAtomsPerTile;
    T* tile = reinterpret_cast<T*>(tile_atom);

    for (int idx = tid * kVecSize; idx <= kTileElems - kVecSize && g_base + idx < hidden_dim; idx += kBlockSize * kVecSize) {
      flashinfer::vec_t<T, kVecSize> v;
      v.cast_load(tile + idx);
      FP8_TYPE q[kVecSize];
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        float f = fminf(fmaxf(float(v[i]) * inv_scale, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        q[i] = static_cast<FP8_TYPE>(f);
      }
      *reinterpret_cast<uint4*>(g_out.data() + g_base + idx) = *reinterpret_cast<const uint4*>(q);
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
  // Hard-code sm_count
  int sm_count = 132;
  constexpr int TOKENS_PER_CTA = 8;
  const bool use_large_kernel = (hidden_dim > 8192) && (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (use_large_kernel) {
      // -------- cute copy ---------------------------------------------------
      constexpr int BLOCK_SIZE = 256;
      constexpr int SMEM_BYTES = 2 * 8192;
      dim3 grid(num_tokens), block(BLOCK_SIZE);

      AT_CUDA_CHECK(cudaFuncSetAttribute(
          per_token_quant_fp8_large_kernel<scalar_t, BLOCK_SIZE>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          SMEM_BYTES));

      per_token_quant_fp8_large_kernel<scalar_t, BLOCK_SIZE><<<grid, block, SMEM_BYTES, stream>>>(
          static_cast<const scalar_t*>(input.data_ptr()),
          static_cast<FP8_TYPE*>(output_q.data_ptr()),
          static_cast<float*>(output_s.data_ptr()),
          hidden_dim,
          num_tokens);
    } else {
      // -------- baseline -----------------------------------------------------
      constexpr int THREADS = 256;
      dim3 grid(num_tokens);
      dim3 block(THREADS);
      per_token_quant_fp8_small_batch_kernel<scalar_t><<<grid, block, 0, stream>>>(
          static_cast<const scalar_t*>(input.data_ptr()),
          static_cast<FP8_TYPE*>(output_q.data_ptr()),
          static_cast<float*>(output_s.data_ptr()),
          hidden_dim,
          num_tokens);
    }
    return true;
  });
}
