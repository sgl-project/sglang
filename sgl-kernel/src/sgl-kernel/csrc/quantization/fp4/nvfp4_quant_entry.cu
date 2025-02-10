#include <torch/all.h>

#if defined ENABLE_NVFP4 && ENABLE_NVFP4
void scaled_fp4_quant_sm100a(torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_sf,
                             torch::Tensor const& input_sf);
#endif

void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_sf,
                      torch::Tensor const& input_sf) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  return scaled_fp4_quant_sm100a(output, input, output_sf, input_sf);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 quantization");
}
