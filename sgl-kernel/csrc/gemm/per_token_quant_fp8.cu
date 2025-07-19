// clang-format off
#include <tuple>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <flashinfer/vec_dtypes.cuh>
// clang-format on

#include "utils.h"

using namespace cute;

static constexpr int kWarpSize = 32;

// ---------------------------------------------------------------------------
// 1. SmallWarp kernel — warp‑local, no shared memory
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
  const int token = blockIdx.x * kTokensPerCTA + warp_id;
  if (token >= num_tokens) return;

  // Global tensors for this token
  auto gmem_in = make_tensor(input + token * hidden_dim, make_shape(hidden_dim));
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

  float warp_max = warpReduceMax(max_value);

  __shared__ float scale;
  scale = warp_max / FP8_E4M3_MAX;
  // Broadcast scale
  if (lane_id == 0) {
    gmem_s(0) = scale;
  }
  float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

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

#ifndef USE_ROCM
      output_arr[j] = static_cast<FP8_TYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }
    *(uint4*)(gmem_out.data() + i * kVecSize) = *(uint4*)output_arr;
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

    per_token_quant_fp8_kernel<scalar_t, TOKENS_PER_CTA, 16><<<grid, block, 0, stream>>>(
        static_cast<const scalar_t*>(input.data_ptr()),
        static_cast<FP8_TYPE*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
    return true;
  });
}
