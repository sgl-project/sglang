#pragma once

#include <Python.h>
#include <torch/all.h>

#include "quantization/extensions/scalar_type.hpp"
#include "quantization/extensions/torch_utils.hpp"
#include "machete_mm_kernel.cuh"

namespace machete {

struct MMArgs {
  torch::Tensor const& A;
  torch::Tensor const& B;
  torch::Tensor& D;
  vllm::ScalarType const& b_type;
  std::optional<at::ScalarType> const& maybe_out_type;
  std::optional<torch::Tensor> const& maybe_group_scales;
  std::optional<torch::Tensor> const& maybe_group_zeros;
  std::optional<int64_t> maybe_group_size;
  std::optional<torch::Tensor> const& maybe_channel_scales;
  std::optional<torch::Tensor> const& maybe_token_scales;
  std::optional<std::string> maybe_schedule;
  std::optional<torch::Tensor> const& maybe_group_layout;
  std::optional<torch::Tensor> const& maybe_valid_len;
  std::optional<int64_t> group_stride;
};

struct SupportedSchedulesArgs {
  at::ScalarType a_type;
  vllm::ScalarType b_type;
  std::optional<at::ScalarType> maybe_group_scales_type;
  std::optional<at::ScalarType> maybe_group_zeros_type;
  std::optional<at::ScalarType> maybe_channel_scales_type;
  std::optional<at::ScalarType> maybe_token_scales_type;
  std::optional<at::ScalarType> maybe_out_type;
};

torch::Tensor mm_dispatch(MMArgs args);

std::vector<std::string> supported_schedules_dispatch(SupportedSchedulesArgs args);

template <typename MacheteKernel>
torch::Tensor run_impl(MMArgs args) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(args.A));

  auto device = args.A.device();
  auto stream = at::cuda::getCurrentCUDAStream(device.index());

  int M = args.A.size(0);
  int N = args.B.size(1);
  int K = args.A.size(1);
  int G = 1;
  if (args.B.dim() == 3) {
    // A: [M,K] B: [G,N,K]
    G = args.B.size(0);
    N = args.B.size(2);
  }

  // Allocate output
  torch::Tensor& D = args.D;

  int64_t group_stride_real = 0;
  if (args.group_stride.has_value()) {
    group_stride_real = args.group_stride.value();
  }
  if (group_stride_real == 0) group_stride_real = 1;
  auto arguments = MacheteKernel::create_arguments(
      stream,  //
      args.A,
      args.B,
      D,
      args.maybe_group_scales,
      args.maybe_group_zeros,
      args.maybe_group_size,
      args.maybe_channel_scales,
      args.maybe_token_scales,
      args.maybe_group_layout,
      args.maybe_valid_len,
      G,
      group_stride_real);
  TORCH_CHECK(MacheteKernel::can_implement(arguments), "Machete kernel cannot be run with these arguments");

  size_t workspace_size = MacheteKernel::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty(workspace_size, torch::TensorOptions().dtype(torch::kU8).device(device));

  MacheteKernel::run(arguments, workspace.mutable_data_ptr(), stream);

  return D;
};

};  // namespace machete
