#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

template <typename scalar_t>
__global__ void ep_pre_reorder_cuda_kernel(
    const scalar_t* __restrict__ input_ptr,
    scalar_t* __restrict__ gateup_input_ptr,
    const int* __restrict__ src2dst_ptr,
    const int* __restrict__ topk_ids_ptr,
    const float* __restrict__ a1_scales_ptr,
    int start_expert_id,
    int end_expert_id,
    int topk,
    int hidden_size,
    bool use_per_token_if_dynamic) {
  int token_idx = blockIdx.x;
  int tid = threadIdx.x;

  const scalar_t* src_ptr = input_ptr + int64_t(token_idx) * hidden_size;
  const int* token_src2dst = src2dst_ptr + token_idx * topk;
  const int* token_topk_ids = topk_ids_ptr + token_idx * topk;

  float scale = 1.0f;

  if (a1_scales_ptr != nullptr and use_per_token_if_dynamic) {
    scale = 1.0f / a1_scales_ptr[token_idx];
  }

  for (int k = 0; k < topk; ++k) {
    int expert_id = token_topk_ids[k];
    if (expert_id < start_expert_id || expert_id > end_expert_id) continue;

    if (a1_scales_ptr != nullptr) {
      if (!use_per_token_if_dynamic) {
        scale = 1.0f / a1_scales_ptr[expert_id - start_expert_id];
      }
    }

    int dst_idx = token_src2dst[k];
    scalar_t* dst_ptr = gateup_input_ptr + int64_t(dst_idx) * hidden_size;

    constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
    using vec_t = flashinfer::vec_t<scalar_t, vec_size>;

    int vec_elements = (hidden_size / vec_size) * vec_size;
    for (int idx = tid; idx < hidden_size / vec_size; idx += blockDim.x) {
      vec_t input_vec, output_vec;
      input_vec.cast_load(src_ptr + idx * vec_size);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float val = static_cast<float>(input_vec[i]);
        output_vec[i] = static_cast<scalar_t>(val * scale);
      }
      output_vec.cast_store(dst_ptr + idx * vec_size);
    }

    for (int idx = vec_elements + tid; idx < hidden_size; idx += blockDim.x) {
      float val = static_cast<float>(src_ptr[idx]);
      dst_ptr[idx] = static_cast<scalar_t>(val * scale);
    }
  }
}

template <typename scalar_t>
__global__ void ep_post_reorder_cuda_kernel(
    const scalar_t* __restrict__ down_output_ptr,
    scalar_t* __restrict__ output_ptr,
    const int* __restrict__ src2dst_ptr,
    const int* __restrict__ topk_ids_ptr,
    const scalar_t* __restrict__ topk_weights_ptr,
    int start_expert_id,
    int end_expert_id,
    int topk,
    int hidden_size) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const int* token_src2dst = src2dst_ptr + token_idx * topk;
  const int* token_topk_ids = topk_ids_ptr + token_idx * topk;
  const scalar_t* token_topk_weights = topk_weights_ptr + token_idx * topk;

  scalar_t* dst_ptr = output_ptr + static_cast<int64_t>(token_idx) * hidden_size;

  constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
  using vec_t = flashinfer::vec_t<scalar_t, vec_size>;

  const int vec_iters = hidden_size / vec_size;
  for (int idx = tid; idx < vec_iters; idx += blockDim.x) {
    float acc[vec_size] = {0};

    for (int k = 0; k < topk; ++k) {
      const int expert_id = token_topk_ids[k];
      if (expert_id < start_expert_id || expert_id > end_expert_id) continue;
      const int src_row = token_src2dst[k];
      const scalar_t* src_ptr = down_output_ptr + static_cast<int64_t>(src_row) * hidden_size;
      const float weight = static_cast<float>(token_topk_weights[k]);

      vec_t src_vec;
      src_vec.cast_load(src_ptr + idx * vec_size);

#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        acc[i] += static_cast<float>(src_vec[i]) * weight;
      }
    }
    vec_t out_vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i)
      out_vec[i] = static_cast<scalar_t>(acc[i]);

    out_vec.cast_store(dst_ptr + idx * vec_size);
  }
}

void ep_moe_pre_reorder(
    torch::Tensor input,
    torch::Tensor gateup_input,
    torch::Tensor src2dst,
    torch::Tensor topk_ids,
    torch::Tensor a1_scales,
    int64_t start_expert_id,
    int64_t end_expert_id,
    int64_t topk,
    bool use_per_token_if_dynamic) {
  const int total_blocks = input.size(0);
  const int block_size = 512;
  dim3 grid(total_blocks);
  dim3 block(block_size);
  int hidden_size = input.size(1);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    ep_pre_reorder_cuda_kernel<scalar_t><<<grid, block>>>(
        static_cast<scalar_t*>(input.data_ptr()),
        static_cast<scalar_t*>(gateup_input.data_ptr()),
        src2dst.data_ptr<int>(),
        topk_ids.data_ptr<int>(),
        a1_scales.defined() ? a1_scales.data_ptr<float>() : nullptr,
        start_expert_id,
        end_expert_id,
        topk,
        hidden_size,
        use_per_token_if_dynamic);
    return true;
  });
}

void ep_moe_post_reorder(
    torch::Tensor down_output,
    torch::Tensor output,
    torch::Tensor src2dst,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    int64_t start_expert_id,
    int64_t end_expert_id,
    int64_t topk) {
  const int total_tokens = output.size(0);
  const int block_size = 512;
  dim3 grid(total_tokens);
  dim3 block(block_size);
  const int hidden_size = output.size(1);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(down_output.scalar_type(), scalar_t, [&] {
    ep_post_reorder_cuda_kernel<scalar_t><<<grid, block>>>(
        static_cast<scalar_t*>(down_output.data_ptr()),
        static_cast<scalar_t*>(output.data_ptr()),
        src2dst.data_ptr<int>(),
        topk_ids.data_ptr<int>(),
        static_cast<scalar_t*>(topk_weights.data_ptr()),
        static_cast<int>(start_expert_id),
        static_cast<int>(end_expert_id),
        static_cast<int>(topk),
        hidden_size);
    return true;
  });
}
