#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <torch/all.h>

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

template <
    typename ElementAB,
    typename ElementC,
    typename ElementAccumulator,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ScaleConfig>
__global__ void get_group_gemm_starts(
    int32_t* expert_offsets,
    ElementAB** a_offsets,
    ElementAB** b_offsets,
    ElementC** out_offsets,
    ElementAccumulator** a_scales_offsets,
    ElementAccumulator** b_scales_offsets,
    ElementAB* a_base_as_int,
    ElementAB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    ElementAccumulator* b_scales_base_as_int,
    LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int,
    int* problem_sizes,
    int* problem_sizes_transpose,
    bool transpose = false) {
  int expert_id = threadIdx.x;

  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }

  int m = problem_sizes[expert_id * 3];
  int n = problem_sizes[expert_id * 3 + 1];
  int k = problem_sizes[expert_id * 3 + 2];
  if (transpose) {
    problem_sizes_transpose[expert_id * 3] = n;
    problem_sizes_transpose[expert_id * 3 + 1] = m;
    problem_sizes_transpose[expert_id * 3 + 2] = k;
  }

  int32_t expert_offset = expert_offsets[expert_id];
  int a_stride = 0;
  int b_stride = 0;
  int a_scale_stride = 0;
  int b_scale_stride = 0;
  if (!transpose) {
    a_stride = expert_offset * k;
    b_stride = expert_id * k * n;
    a_scale_stride = expert_offset * k / 128;
    b_scale_stride = expert_id * k * n / 128 / 128;
  } else {
    a_stride = expert_id * k * n;
    b_stride = expert_offset * k;
    a_scale_stride = expert_id * k * n / 128 / 128;
    b_scale_stride = expert_offset * k / 128;
  }
  a_offsets[expert_id] = a_base_as_int + a_stride;
  b_offsets[expert_id] = b_base_as_int + b_stride;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base_as_int + a_scale_stride;
  b_scales_offsets[expert_id] = b_scales_base_as_int + b_scale_stride;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  if (!transpose) {
    *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  } else {
    *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
    *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));
  }
}

#define __CALL_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE, LayoutSFA, LayoutSFB, ScaleConfig)         \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                                                 \
    get_group_gemm_starts<cutlass::float_e4m3_t, C_TYPE, float, LayoutSFA, LayoutSFB, ScaleConfig> \
        <<<1, num_experts, 0, stream>>>(                                                           \
            static_cast<int32_t*>(expert_offsets.data_ptr()),                                      \
            static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),                               \
            static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr()),                               \
            static_cast<C_TYPE**>(out_ptrs.data_ptr()),                                            \
            static_cast<float**>(a_scales_ptrs.data_ptr()),                                        \
            static_cast<float**>(b_scales_ptrs.data_ptr()),                                        \
            static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),                             \
            static_cast<cutlass::float_e4m3_t*>(b_tensors.data_ptr()),                             \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                                          \
            static_cast<float*>(a_scales.data_ptr()),                                              \
            static_cast<float*>(b_scales.data_ptr()),                                              \
            reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),                                   \
            reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),                                   \
            static_cast<int*>(problem_sizes.data_ptr()),                                           \
            static_cast<int*>(problem_sizes_transpose.data_ptr()),                                 \
            transpose);                                                                            \
  }

namespace {
template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
void run_get_group_gemm_starts(
    torch::Tensor const& expert_offsets,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor& out_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& layout_sfa,
    torch::Tensor const& layout_sfb,
    torch::Tensor const& problem_sizes,
    torch::Tensor& problem_sizes_transpose,
    bool transpose = false) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(out_tensors.size(1) % 128 == 0 or out_tensors.size(0) % 128 == 0);
  TORCH_CHECK(a_tensors.size(1) % 128 == 0 or a_tensors.size(0) % 128 == 0);

  int num_experts = (int)expert_offsets.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  if (false) {
  }
  __CALL_GET_STARTS_KERNEL(torch::kBFloat16, cutlass::bfloat16_t, LayoutSFA, LayoutSFB, ScaleConfig)
  __CALL_GET_STARTS_KERNEL(torch::kFloat16, half, LayoutSFA, LayoutSFB, ScaleConfig)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}
}  // namespace
