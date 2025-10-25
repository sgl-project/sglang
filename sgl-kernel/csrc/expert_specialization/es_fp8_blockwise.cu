#include <torch/all.h>

#include <tuple>

#include "es_fp8_blockwise_launcher.cuh"

/**
 * @brief Performs blockwise grouped matrix multiplication on FP8 quantized inputs,
 *        with per-block scaling.
 *
 * This function dispatches to hardware-specific implementations (e.g., SM100 FP8)
 * to compute:
 *     C_i = scale_a[i] * A_i * scale_b[i] * B_i
 * for each expert group `i`, using input `problem_sizes` and `expert_offsets`
 * to describe the individual matrix dimensions and their offsets.
 *
 * Input tensors A and B must be quantized to 8-bit formats and dequantized before multiplication.
 * The output tensor is written with bfloat16 or half precision.
 *
 * @param output         Output tensor (must be of type bfloat16 or half).
 * @param a              Input tensor A (must be kFloat8_e4m3fn).
 * @param b              Input tensor B (must be kFloat8_e4m3fn).
 * @param scales_a       Scaling factors for tensor A, float32 per expert group.
 * @param scales_b       Scaling factors for tensor B, float32 per expert group.
 * @param stride_a       Stride information for tensor A (int32).
 * @param stride_b       Stride information for tensor B (int32).
 * @param stride_c       Stride information for output tensor C (int32).
 * @param problem_sizes  2D int32 tensor of shape (num_experts, 3), specifying (M, N, K)
 *                       for each grouped matrix multiplication problem.
 * @param expert_offsets 1D int32 tensor of size (num_experts), used to index into
 *                       the grouped input tensors for dispatch.
 */
void es_fp8_blockwise_scaled_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_d,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0), "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "a must be kFloat8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "b must be kFloat8_e4m3fn");
  TORCH_CHECK(
      output.scalar_type() == torch::kBFloat16 || output.scalar_type() == torch::kHalf,
      "output must be bfloat16 or half");

  int num_experts = (int)problem_sizes.size(0);
  torch::TensorOptions options_int64 = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
  torch::TensorOptions options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(a.device());
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int64);
  torch::Tensor a_ptrs = torch::empty(num_experts, options_int64);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int64);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int64);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int64);

  torch::Tensor layout_sfa = torch::empty({num_experts, 5}, options_int32);
  torch::Tensor layout_sfb = torch::empty({num_experts, 5}, options_int32);

  torch::Tensor lm_problem_sizes = torch::empty({num_experts, 3}, options_int32);
  torch::Tensor mm_problem_sizes = torch::empty({num_experts, 3}, options_int32);
  torch::Tensor hm_problem_sizes = torch::empty({num_experts, 3}, options_int32);

  const std::string H20_device_type_str("NVIDIA H20");
  bool is_h20_device = std::string(at::cuda::getCurrentDeviceProperties()->name) == H20_device_type_str;
  at::cuda::CUDAGuard device_guard{(char)a.get_device()};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

  if (output.dtype() == torch::kBFloat16) {
    expert_specialization::es_sm90_fp8_blockwise_scaled_group_mm_pre_compute<cutlass::bfloat16_t>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        layout_sfa,
        layout_sfb,
        lm_problem_sizes,
        mm_problem_sizes,
        hm_problem_sizes,
        output,
        a,
        b,
        scales_a,
        scales_b,
        problem_sizes,
        expert_offsets,
        is_h20_device,
        stream);
  } else if (output.dtype() == torch::kFloat16) {
    expert_specialization::es_sm90_fp8_blockwise_scaled_group_mm_pre_compute<cutlass::half_t>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        layout_sfa,
        layout_sfb,
        lm_problem_sizes,
        mm_problem_sizes,
        hm_problem_sizes,
        output,
        a,
        b,
        scales_a,
        scales_b,
        problem_sizes,
        expert_offsets,
        is_h20_device,
        stream);
  } else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }

  if (output.dtype() == torch::kBFloat16) {
    expert_specialization::es_sm90_fp8_blockwise_scaled_group_mm_distpatch_out_dtype<cutlass::bfloat16_t>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_d,
        layout_sfa,
        layout_sfb,
        lm_problem_sizes,
        mm_problem_sizes,
        hm_problem_sizes,
        workspace,
        is_h20_device,
        stream);
  } else if (output.dtype() == torch::kFloat16) {
    expert_specialization::es_sm90_fp8_blockwise_scaled_group_mm_distpatch_out_dtype<cutlass::half_t>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_d,
        layout_sfa,
        layout_sfb,
        lm_problem_sizes,
        mm_problem_sizes,
        hm_problem_sizes,
        workspace,
        is_h20_device,
        stream);
  } else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
      can_implement, "No implemented fp8_blockwise_scaled_grouped_mm for current compute capability: ", sm_version);
#endif
}
