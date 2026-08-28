// SPDX-License-Identifier: Apache-2.0

/*
 * KDA provenance: this kernel was automatically optimized by the Humanize2
 * workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
 * (https://github.com/mit-han-lab/kernel-design-agents).
 * Source: https://github.com/BBuf/KDA-Pilot/pull/195 @
 * 516c976cee824a236679adf6eb525275a0a9a120.
 */
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

int launch_cutlass_fp4_gemm(void *output, const void *input, const void *weight,
                            const void *input_scales, const void *weight_scales,
                            const void *alpha, int m, int n, int k,
                            cudaStream_t stream);

void cutlass_fp4_gemm(torch::Tensor output, torch::Tensor input,
                      torch::Tensor weight, torch::Tensor input_scales,
                      torch::Tensor weight_scales, torch::Tensor alpha) {
  TORCH_CHECK(output.is_cuda() && input.is_cuda() && weight.is_cuda());
  TORCH_CHECK(input_scales.is_cuda() && weight_scales.is_cuda() &&
              alpha.is_cuda());
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input.scalar_type() == torch::kUInt8);
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8);
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat32 && alpha.numel() == 1);
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous());
  TORCH_CHECK(output.is_contiguous() && alpha.is_contiguous());
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && output.dim() == 2);

  const int m = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1) * 2);
  const int n = static_cast<int>(weight.size(0));
  TORCH_CHECK(weight.size(1) * 2 == k);
  TORCH_CHECK(output.size(0) == m && output.size(1) == n);

  const int status = launch_cutlass_fp4_gemm(
      output.data_ptr(), input.data_ptr(), weight.data_ptr(),
      input_scales.data_ptr(), weight_scales.data_ptr(), alpha.data_ptr(), m, n,
      k, c10::cuda::getCurrentCUDAStream(input.get_device()));
  TORCH_CHECK(status == 0, "CUTLASS SM120 FP4 GEMM failed with status ",
              status);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cutlass_fp4_gemm", &cutlass_fp4_gemm);
}
