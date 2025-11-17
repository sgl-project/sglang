#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor, 0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}

void cutlass_w4a8_moe_mm_sm90(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size,
    int64_t topk);

void get_cutlass_w4a8_moe_mm_data_caller(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k);

void cutlass_w4a8_moe_mm(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size,
    int64_t topk) {
  cutlass_w4a8_moe_mm_sm90(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      chunk_size,
      topk);
  return;
}

void get_cutlass_w4a8_moe_mm_data(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  get_cutlass_w4a8_moe_mm_data_caller(
      topk_ids,
      expert_offsets,
      problem_sizes1,
      problem_sizes2,
      input_permutation,
      output_permutation,
      num_experts,
      n,
      k);
  return;
}
