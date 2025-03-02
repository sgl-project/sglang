#include <torch/all.h>

#if defined ENABLE_NVFP4 && ENABLE_NVFP4
void cutlass_scaled_fp4_mm_sm100a(torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B,
                                  torch::Tensor const& A_sf, torch::Tensor const& B_sf, torch::Tensor const& alpha);
#endif

void cutlass_scaled_fp4_mm(torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B, torch::Tensor const& A_sf,
                           torch::Tensor const& B_sf, torch::Tensor const& alpha) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  return cutlass_scaled_fp4_mm_sm100a(D, A, B, A_sf, B_sf, alpha);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 mm kernel.");
}
