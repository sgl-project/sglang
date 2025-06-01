#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>

#include "utils.h"

__global__ void ep_pre_reorder_cuda_kernel(
    const float* __restrict__ input_ptr,
    float* __restrict__ gateup_input_ptr,
    const int* __restrict__ src2dst_ptr,
    const int* __restrict__ topk_ids_ptr,
    const float* __restrict__ a1_scales_ptr,
    int start_expert_id,
    int end_expert_id,
    int topk,
    int hidden_size) {
  int token_idx = blockIdx.x;
  int tid = threadIdx.x;

  const float* src_ptr = input_ptr + token_idx * hidden_size;
  const int* token_src2dst = src2dst_ptr + token_idx * topk;
  const int* token_topk_ids = topk_ids_ptr + token_idx * topk;

  for (int k = 0; k < topk; ++k) {
    int expert_id = token_topk_ids[k];
    if (expert_id < start_expert_id || expert_id > end_expert_id) continue;

    float scale = 1.0f;
    if (a1_scales_ptr != nullptr) {
      scale = 1.0f / a1_scales_ptr[expert_id - start_expert_id];
    }

    int dst_idx = token_src2dst[k];
    float* dst_ptr = gateup_input_ptr + dst_idx * hidden_size;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
      dst_ptr[i] = src_ptr[i] * scale;
    }
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
    int64_t topk) {
  int total_blocks = input.size(0);
  int block_size = 512;
  dim3 grid(total_blocks);
  dim3 block(block_size);
  int hidden_size = input.size(1);
  ep_pre_reorder_cuda_kernel<<<grid, block>>>(
      input.data_ptr<float>(),
      gateup_input.data_ptr<float>(),
      src2dst.data_ptr<int>(),
      topk_ids.data_ptr<int>(),
      a1_scales.defined() ? a1_scales.data_ptr<float>() : nullptr,
      start_expert_id,
      end_expert_id,
      topk,
      hidden_size);
}
