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
// Baseline kernel (1 token / CTA, CUB block reduce)
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

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

#else

template <typename T, int kBlockSize = 256, int kVecSize = 16, int kTileBytes = 8192, int kStages = 2>
__launch_bounds__(kBlockSize) __global__ void per_token_quant_fp8_large_kernel(
    const T* __restrict__ input,
    FP8_TYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    int hidden_dim,
    int num_tokens) {
  static_assert(kBlockSize % 32 == 0, "kBlockSize must be multiple of 32");
  static_assert(kVecSize > 0, "kVecSize must be > 0");
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
  constexpr int kGroupSize = kVecSize * kWarpSize;  // tail length of each warp
  static_assert(kGroupSize > 0, "kGroupSize must be > 0");

  const int token = blockIdx.x;
  if (token >= num_tokens) return;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  auto shape_2d = make_shape(num_tokens, hidden_dim);
  auto stride_2d = make_stride(hidden_dim, _1{});  // row-major
  auto g_in_all = make_tensor(make_gmem_ptr(input), shape_2d, stride_2d);
  auto g_out_all = make_tensor(make_gmem_ptr(output_q), shape_2d, stride_2d);

  auto g_in_vec = g_in_all(token, _);
  auto g_out_vec = g_out_all(token, _);

  // --------- Warp tile  ---------
  auto warp_tiler = make_shape(Int<kGroupSize>{});

  using PackTIn = cute::uint128_t;
  using PackTOut = cute::uint128_t;

  // --------- TiledCopy（gmem <-> reg）---------
  using CopyAtomIn = Copy_Atom<UniversalCopy<PackTIn>, T>;
  using CopyAtomOut = Copy_Atom<UniversalCopy<PackTOut>, FP8_TYPE>;

  auto copy_in = make_tiled_copy(CopyAtomIn{}, Layout<Shape<_32>>{}, Layout<Shape<Int<kVecSize>>>{});
  auto copy_out = make_tiled_copy(CopyAtomOut{}, Layout<Shape<_32>>{}, Layout<Shape<Int<kVecSize>>>{});

  const int num_tiles = ceil_div(hidden_dim, kGroupSize);

  // ----------------------- Pass-1: reduce token max -----------------------
  float local_max = 0.f;

  for (int tile_idx = warp; tile_idx < num_tiles; tile_idx += kWarpsPerBlock) {
    const int base = tile_idx * kGroupSize;
    const int valid = min(kGroupSize, hidden_dim - base);

    if (valid == kGroupSize) {
      // local tile：warp level copy
      auto warp_in = local_tile(g_in_vec, warp_tiler, tile_idx);

      auto thd_in = copy_in.get_slice(lane).partition_S(warp_in);
      auto reg_in = make_tensor_like(thd_in);

      copy(copy_in, thd_in, reg_in);  // gmem -> reg

      CUTE_UNROLL
      for (int i = 0; i < size(reg_in); ++i) {
        local_max = fmaxf(local_max, fabsf(float(reg_in(i))));
      }
    } else {
      // tail tile
      for (int idx = lane; idx < valid; idx += kWarpSize) {
        float v = float(g_in_vec(base + idx));
        local_max = fmaxf(local_max, fabsf(v));
      }
    }
  }

  float token_max = blockReduceMax(local_max);

  __shared__ float scale;
  if (tid == 0) {
    scale = (token_max == 0.f) ? 0.f : (token_max / FP8_E4M3_MAX);
    output_s[token] = scale;
  }
  __syncthreads();

  const float inv_scale = (scale == 0.f) ? 0.f : 1.f / scale;

  // ----------------------- Pass-2: quantize + write back -----------------------
  for (int tile_idx = warp; tile_idx < num_tiles; tile_idx += kWarpsPerBlock) {
    const int base = tile_idx * kGroupSize;
    const int valid = min(kGroupSize, hidden_dim - base);

    if (valid == kGroupSize) {
      auto warp_in = local_tile(g_in_vec, warp_tiler, tile_idx);
      auto warp_out = local_tile(g_out_vec, warp_tiler, tile_idx);

      auto thd_in = copy_in.get_slice(lane).partition_S(warp_in);
      auto thd_out = copy_out.get_slice(lane).partition_D(warp_out);

      auto reg_in = make_tensor_like(thd_in);
      auto reg_out = make_tensor_like(thd_out);

      copy(copy_in, thd_in, reg_in);

      CUTE_UNROLL
      for (int i = 0; i < size(reg_out); ++i) {
        float f = float(reg_in(i)) * inv_scale;
        f = fminf(fmaxf(f, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        reg_out(i) = FP8_TYPE(f);
      }

      copy(copy_out, reg_out, thd_out);  // reg -> gmem
    } else {
      for (int idx = lane; idx < valid; idx += kWarpSize) {
        float f = float(g_in_vec(base + idx)) * inv_scale;
        f = fminf(fmaxf(f, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        g_out_vec(base + idx) = FP8_TYPE(f);
      }
    }
  }
}

#endif

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
  const bool use_large_kernel = (hidden_dim >= 4096) && (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (use_large_kernel) {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
      // -------- warp‑local ---------------------------------------------------
      constexpr int THREADS = TOKENS_PER_CTA * kWarpSize;  // 256
      dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
      dim3 block(THREADS);
      per_token_quant_fp8_kernel<scalar_t, TOKENS_PER_CTA, 16><<<grid, block, 0, stream>>>(
          static_cast<const scalar_t*>(input.data_ptr()),
          static_cast<FP8_TYPE*>(output_q.data_ptr()),
          static_cast<float*>(output_s.data_ptr()),
          hidden_dim,
          num_tokens);
#else
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
#endif
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
