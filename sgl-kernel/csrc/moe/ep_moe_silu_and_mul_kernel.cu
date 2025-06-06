#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

using namespace flashinfer;

template <typename scalar_t, float (*Activation)(const float&)>
__global__ void ep_moe_act_and_mul_cuda_kernel(
    const scalar_t* __restrict__ gateup_output,
    scalar_t* __restrict__ down_input,
    const int64_t* __restrict__ reorder_topk_ids,
    const float* __restrict__ scales,
    int start_expert_id,
    int end_expert_id,
    int hidden_size) {
  constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;

  const int half_hidden_size = hidden_size >> 1;
  const int expert_id = reorder_topk_ids[token_idx];

  if (expert_id < start_expert_id || expert_id > end_expert_id) return;
  const scalar_t* gate_output_ptr = gateup_output + static_cast<int64_t>(token_idx) * hidden_size;
  const scalar_t* up_output_ptr = gate_output_ptr + half_hidden_size;
  scalar_t* dst_ptr = down_input + static_cast<int64_t>(token_idx) * half_hidden_size;
  float scale = 1.0f;
  if (scales != nullptr) {
    scale = 1.0f / scales[expert_id - start_expert_id];
  }

  using vec_t = flashinfer::vec_t<float, vec_size>;
  const uint32_t vec_elements = half_hidden_size / vec_size;
#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < vec_elements; idx += stride) {
    vec_t gate_vec, up_vec, out_vec;
    gate_vec.cast_load(gate_output_ptr + idx * vec_size);
    up_vec.cast_load(up_output_ptr + idx * vec_size);

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = Activation(gate_vec[i]) * up_vec[i] * scale;
    }
    out_vec.cast_store(dst_ptr + idx * vec_size);
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
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
  TORCH_CHECK(hidden_size % 2 == 0, "hidden_size must be even.");
  const int block_size = 512;
  dim3 grid(total_tokens);
  dim3 block(block_size);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(gateup_output.scalar_type(), scalar_t, [&] {
    ep_moe_act_and_mul_cuda_kernel<scalar_t, silu><<<grid, block>>>(
        static_cast<scalar_t*>(gateup_output.data_ptr()),
        static_cast<scalar_t*>(down_input.data_ptr()),
        reorder_topk_ids.data_ptr<int64_t>(),
        scales.defined() ? scales.data_ptr<float>() : nullptr,
        start_expert_id,
        end_expert_id,
        hidden_size);
    return true;
  });
}
