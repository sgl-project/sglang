#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <torch/all.h>

#include <vector>

#include "common.h"

namespace {

template <typename out_t, typename weight_t, typename scale_t>
void fused_scale_cpu_impl(
    at::Tensor& out, const at::Tensor& weight, const at::Tensor& q_scale, double out_scale, int64_t numel) {
  const weight_t* __restrict__ weight_ptr = weight.const_data_ptr<weight_t>();
  const scale_t* __restrict__ q_scale_ptr = q_scale.const_data_ptr<scale_t>();
  out_t* __restrict__ out_ptr = out.mutable_data_ptr<out_t>();
  const float out_scale_f = static_cast<float>(out_scale);

  at::parallel_for(0, numel, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const float value = static_cast<float>(weight_ptr[i]) * out_scale_f * static_cast<float>(q_scale_ptr[i]);
      out_ptr[i] = static_cast<out_t>(value);
    }
  });
}

}  // namespace

at::Tensor fused_scale_cpu(at::Tensor& weight, double out_scale, at::Tensor& q_scale) {
  RECORD_FUNCTION("sgl-kernel::fused_scale_cpu", std::vector<c10::IValue>({weight, q_scale}));

  CHECK_INPUT(weight);
  CHECK_INPUT(q_scale);
  CHECK_DIM(2, weight);
  TORCH_CHECK(q_scale.numel() == weight.numel(), "q_scale must have the same number of elements as weight");
  TORCH_CHECK(
      weight.scalar_type() == at::kFloat || weight.scalar_type() == at::kHalf || weight.scalar_type() == at::kBFloat16,
      "weight must be float32, float16, or bfloat16");
  TORCH_CHECK(q_scale.scalar_type() == at::kFloat, "q_scale must be float32");

  const int64_t B = weight.size(0);
  const int64_t H = weight.size(1);
  const int64_t numel = B * H;
  at::Tensor out = at::empty({B, H, 1}, weight.options().dtype(at::kFloat));

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, weight.scalar_type(), "fused_scale_cpu_weight", [&] {
    fused_scale_cpu_impl<float, scalar_t, float>(out, weight, q_scale, out_scale, numel);
  });

  return out;
}
