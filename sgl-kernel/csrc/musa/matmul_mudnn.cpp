#include <mudnn.h>
#include <torch/all.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/extension.h>
#include <torch_musa/csrc/aten/utils/Context.h>
#include <torch_musa/csrc/aten/utils/Utils.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include <optional>
#include <vector>

using at::musa::GetComputeModeFromCtx;
using at::musa::muTensor;

void mudnn_w8a8_scaled_mm(
    torch::Tensor& c,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(1) && b.size(0) == c.size(1));

  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() && bias->dim() == 1);
  }

  at::musa::OptionalMUSAGuard const device_guard(device_of(a));

  const auto a_type = a.scalar_type();

  muTensor a_mu = at::musa::CreateMUTensor(a);
  muTensor b_mu = at::musa::CreateMUTensor(b);
  muTensor out_mu = at::musa::CreateMUTensor(c);
  torch::Tensor bias_ = bias.value_or(torch::Tensor());
  muTensor bias_mu = at::musa::CreateMUTensor(bias_);

  muTensor a_scales_mu = at::musa::CreateMUTensor(a_scales);
  muTensor b_scales_mu = at::musa::CreateMUTensor(b_scales);

  auto& handle = at::GetMudnnHandle();
  ::musa::dnn::BatchMatMul op;
  ::musa::dnn::MatMulLtParam param;

  CHECK_MUDNN_STATUS(param.SetScale(a_scales_mu, b_scales_mu, muTensor(), muTensor(), 128), "SetScale");
  CHECK_MUDNN_STATUS(op.SetTranspose(false, true), "SetTranspose");
  CHECK_MUDNN_STATUS(op.SetDeterministic(false), "SetDeterministic");
  CHECK_MUDNN_STATUS(op.SetComputeMode(GetComputeModeFromCtx(a_type)), "SetComputeMode");
  CHECK_MUDNN_STATUS(op.SetBeta(0.0), "SetBeta");

  CHECK_MUDNN_STATUS(op.RunLt(handle, out_mu, a_mu, b_mu, out_mu, bias_mu, param, at::musa::InternalMemAlloc), "RunLt");
}
