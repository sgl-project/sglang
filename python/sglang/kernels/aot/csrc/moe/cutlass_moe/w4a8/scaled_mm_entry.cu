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

void cutlass_mxfp4a8_moe_mm_sm90(
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
    int64_t topk,
    std::optional<torch::Tensor> act_block_scales,
    std::optional<torch::Tensor> as_strides,
    int64_t act_scale_group,
    std::optional<torch::Tensor> expert_ids);

void cutlass_mxfp4a8_fused_moe_mm_sm90(
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
    int64_t topk,
    int64_t swg_config,
    std::optional<torch::Tensor> expert_ids);

void fused_per_token_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    const torch::Tensor& residual,
    const torch::Tensor& expert_offsets,
    int64_t num_experts);

void fused_per_token_quant_fp8_shuffled(
    const torch::Tensor& input,
    const torch::Tensor& permutation,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    const torch::Tensor& residual,
    const torch::Tensor& expert_offsets,
    int64_t num_experts);

bool fused_prepare_moe_input_and_quant_fp8_shuffled(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    torch::Tensor& output_q,
    torch::Tensor& output_s,
    const torch::Tensor& residual,
    torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    int64_t num_experts,
    int64_t intermediate_size);

void fused_swiglu_quant_fp8(
    const at::Tensor& input,
    at::Tensor& output_q,
    at::Tensor& output_s,
    const at::Tensor& residual,
    const at::Tensor& expert_offsets,
    int64_t num_experts,
    double swiglu_limit,
    bool has_swiglu_limit);

void get_cutlass_w4a8_moe_mm_data_with_permutation(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k);

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

void cutlass_mxfp4a8_moe_mm(
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
    int64_t topk,
    std::optional<torch::Tensor> act_block_scales,
    std::optional<torch::Tensor> as_strides,
    int64_t act_scale_group,
    std::optional<torch::Tensor> expert_ids) {
  cutlass_mxfp4a8_moe_mm_sm90(
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
      topk,
      act_block_scales,
      as_strides,
      act_scale_group,
      expert_ids);
  return;
}

void cutlass_mxfp4a8_fused_moe_mm(
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
    int64_t topk,
    int64_t swg_config,
    std::optional<torch::Tensor> expert_ids) {
  cutlass_mxfp4a8_fused_moe_mm_sm90(
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
      topk,
      swg_config,
      expert_ids);
}

void cutlass_mxfp4a8_fused_moe_core(
    torch::Tensor& c1,
    torch::Tensor& c2,
    torch::Tensor const& input,
    torch::Tensor const& topk_ids,
    torch::Tensor& a_map,
    torch::Tensor& c_map,
    torch::Tensor& gateup_input_bf16,
    torch::Tensor& gateup_input,
    torch::Tensor& a1_scale,
    torch::Tensor& intermediate_q,
    torch::Tensor& a2_scale,
    torch::Tensor const& w1,
    torch::Tensor const& w1_scale,
    torch::Tensor const& w1_residual,
    torch::Tensor const& w2,
    torch::Tensor const& w2_scale,
    torch::Tensor const& w2_residual,
    torch::Tensor& expert_offsets,
    torch::Tensor const& gemm_expert_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor const& a_strides1,
    torch::Tensor const& b_strides1,
    torch::Tensor const& c_strides1,
    torch::Tensor const& s_strides1,
    torch::Tensor const& a_strides2,
    torch::Tensor const& b_strides2,
    torch::Tensor const& c_strides2,
    torch::Tensor const& s_strides2,
    int64_t topk,
    int64_t gemm1_config,
    int64_t gemm2_config,
    int64_t num_experts,
    int64_t intermediate_size,
    int64_t hidden_size,
    double swiglu_limit,
    bool has_swiglu_limit,
    bool prepare_inputs,
    std::optional<torch::Tensor> expert_ids) {
  if (prepare_inputs) {
    const bool fused = fused_prepare_moe_input_and_quant_fp8_shuffled(
        input,
        topk_ids,
        gateup_input,
        a1_scale,
        w1_residual,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        intermediate_size);
    if (!fused) {
      // Preserve the existing implementation as the fallback for shapes outside
      // the bounded tiny route/expert domain.
      get_cutlass_w4a8_moe_mm_data_with_permutation(
          topk_ids,
          expert_offsets,
          problem_sizes1,
          problem_sizes2,
          a_map,
          c_map,
          num_experts,
          intermediate_size,
          hidden_size);
      fused_per_token_quant_fp8_shuffled(
          input, a_map, gateup_input, a1_scale, w1_residual, expert_offsets, num_experts);
    }
  } else {
    fused_per_token_quant_fp8_shuffled(input, a_map, gateup_input, a1_scale, w1_residual, expert_offsets, num_experts);
  }
  cutlass_mxfp4a8_fused_moe_mm(
      c1,
      gateup_input,
      w1,
      a1_scale,
      w1_scale,
      gemm_expert_offsets,
      problem_sizes1,
      a_strides1,
      b_strides1,
      c_strides1,
      s_strides1,
      topk,
      gemm1_config,
      expert_ids);
  fused_swiglu_quant_fp8(
      c1, intermediate_q, a2_scale, w2_residual, expert_offsets, num_experts, swiglu_limit, has_swiglu_limit);
  cutlass_mxfp4a8_fused_moe_mm(
      c2,
      intermediate_q,
      w2,
      a2_scale,
      w2_scale,
      gemm_expert_offsets,
      problem_sizes2,
      a_strides2,
      b_strides2,
      c_strides2,
      s_strides2,
      topk,
      gemm2_config,
      expert_ids);
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
