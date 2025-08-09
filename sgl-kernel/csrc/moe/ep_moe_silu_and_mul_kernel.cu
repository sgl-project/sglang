#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <THC/THCAtomics.cuh>
#include <algorithm>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

using namespace flashinfer;

template <typename scalar_t>
__device__ inline scalar_t silu_quantize(float x);

template <>
__device__ inline float silu_quantize<float>(float x) {
  float y = x / (1.f + __expf(-x));
  return y;
}

template <>
__device__ inline __half silu_quantize<__half>(float x) {
  float y = x / (1.f + __expf(-x));
  return __float2half_rn(y);
}

template <>
__device__ inline __nv_bfloat16 silu_quantize<__nv_bfloat16>(float x) {
  float y = x / (1.f + __expf(-x));
  return __float2bfloat16_rn(y);
}

template <typename scalar_t>
__global__ void ep_moe_act_and_mul_cuda_kernel(
    const scalar_t* __restrict__ gateup_output,
    scalar_t* __restrict__ down_input,
    const int* __restrict__ reorder_topk_ids,
    const float* __restrict__ scales,
    int start_expert_id,
    int end_expert_id,
    int hidden_size) {
  constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
  using vec_t = flashinfer::vec_t<scalar_t, vec_size>;

  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;

  const int half_hidden_size = hidden_size >> 1;
  const int expert_id = reorder_topk_ids[token_idx];

  if (expert_id < start_expert_id || expert_id > end_expert_id) return;
  const scalar_t* gate_output_ptr = gateup_output + static_cast<int64_t>(token_idx) * hidden_size;
  const scalar_t* up_output_ptr = gate_output_ptr + half_hidden_size;
  scalar_t* dst_ptr = down_input + static_cast<int64_t>(token_idx) * half_hidden_size;
  scalar_t scale_q = static_cast<scalar_t>(scales ? (1.f / scales[expert_id - start_expert_id]) : 1.f);

  const uint32_t vec_elements = half_hidden_size / vec_size;
#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < vec_elements; idx += stride) {
    vec_t gate_vec, up_vec, out_vec;
    gate_vec.load(gate_output_ptr + idx * vec_size);
    up_vec.load(up_output_ptr + idx * vec_size);

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float gate_f = static_cast<float>(gate_vec[i]);
      scalar_t gate_q = silu_quantize<scalar_t>(gate_f);
      scalar_t prod = gate_q * up_vec[i] * scale_q;
      out_vec[i] = prod;
    }
    out_vec.store(dst_ptr + idx * vec_size);
  }

  const int64_t scalar_start = static_cast<int64_t>(vec_elements) * vec_size + thread_idx;
#pragma unroll 1
  for (int64_t idx = scalar_start; idx < half_hidden_size; idx += stride) {
    float gate_f = static_cast<float>(gate_output_ptr[idx]);
    scalar_t gate_q = silu_quantize<scalar_t>(gate_f);
    dst_ptr[idx] = gate_q * up_output_ptr[idx] * scale_q;
  }
}

void ep_moe_silu_and_mul(
    torch::Tensor gateup_output,
    torch::Tensor down_input,
    torch::Tensor reorder_topk_ids,
    torch::Tensor scales,
    int64_t start_expert_id,
    int64_t end_expert_id) {
  const int total_tokens = gateup_output.size(0);
  const int hidden_size = gateup_output.size(1);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(gateup_output.scalar_type(), scalar_t, [&] {
    dim3 grid(total_tokens);
    constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
    const int half_hidden_size = hidden_size >> 1;
    uint32_t threads = (half_hidden_size + vec_size - 1) / vec_size;
    threads = std::max<uint32_t>(threads, 256);
    threads = ((threads + 31) & ~31U);
    dim3 block(std::min(threads, 1024U));
    ep_moe_act_and_mul_cuda_kernel<scalar_t><<<grid, block>>>(
        static_cast<scalar_t*>(gateup_output.data_ptr()),
        static_cast<scalar_t*>(down_input.data_ptr()),
        reorder_topk_ids.data_ptr<int>(),
        scales.defined() ? scales.data_ptr<float>() : nullptr,
        static_cast<int>(start_expert_id),
        static_cast<int>(end_expert_id),
        hidden_size);
    return true;
  });
}
