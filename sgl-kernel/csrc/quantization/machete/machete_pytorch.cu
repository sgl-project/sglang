#include "quantization/extensions/scalar_type.hpp"
#include "machete_mm_launcher.cuh"
#include "machete_prepack_launcher.cuh"

// Adapted form https://github.com/vllm-project/vllm/blob/main/csrc/quantization/machete/machete_pytorch.cu

namespace machete {

using namespace vllm;

std::vector<std::string> supported_schedules(
    at::ScalarType a_type,
    int64_t b_type_id,
    std::optional<at::ScalarType> maybe_group_scales_type,
    std::optional<at::ScalarType> maybe_group_zeros_type,
    std::optional<at::ScalarType> maybe_channel_scales_type,
    std::optional<at::ScalarType> maybe_token_scales_type,
    std::optional<at::ScalarType> maybe_out_type) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  return supported_schedules_dispatch({
      .a_type = a_type,
      .b_type = b_type,
      .maybe_group_scales_type = maybe_group_scales_type,
      .maybe_group_zeros_type = maybe_group_zeros_type,
      .maybe_channel_scales_type = maybe_channel_scales_type,
      .maybe_token_scales_type = maybe_token_scales_type,
      .maybe_out_type = maybe_out_type,
  });
}

torch::Tensor
mm(torch::Tensor const& A,
   torch::Tensor const& B,
   torch::Tensor& D,
   int64_t b_type_id,
   std::optional<at::ScalarType> const& maybe_out_type,
   std::optional<torch::Tensor> const& maybe_group_scales,
   std::optional<torch::Tensor> const& maybe_group_zeros,
   std::optional<int64_t> maybe_group_size,
   std::optional<torch::Tensor> const& maybe_channel_scales,
   std::optional<torch::Tensor> const& maybe_token_scales,
   std::optional<std::string> maybe_schedule,
   std::optional<torch::Tensor> const& maybe_group_layout,
   std::optional<torch::Tensor> const& maybe_valid_len,
   std::optional<int64_t> group_stride) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  return mm_dispatch(
      {.A = A,
       .B = B,
       .D = D,
       .b_type = b_type,
       .maybe_out_type = maybe_out_type,
       .maybe_group_scales = maybe_group_scales,
       .maybe_group_zeros = maybe_group_zeros,
       .maybe_group_size = maybe_group_size,
       .maybe_channel_scales = maybe_channel_scales,
       .maybe_token_scales = maybe_token_scales,
       .maybe_schedule = maybe_schedule,
       .maybe_group_layout = maybe_group_layout,
       .maybe_valid_len = maybe_valid_len,
       .group_stride = group_stride});
}

torch::Tensor prepack_B(
    torch::Tensor const& B,
    at::ScalarType const& a_type,
    int64_t b_type_id,
    std::optional<at::ScalarType> const& maybe_group_scales_type) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  return prepack_B_dispatch(
      {.B = B, .a_type = a_type, .b_type = b_type, .maybe_group_scales_type = maybe_group_scales_type});
}

};  // namespace machete
