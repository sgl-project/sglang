#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <torch/all.h>

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
__global__ void int4_fp8_get_group_gemm_starts(
    int32_t* expert_offsets,
    ElementA** a_offsets,
    ElementB** b_offsets,
    ElementC** out_offsets,
    ElementAccumulator** a_scales_offsets,
    cutlass::bfloat16_t** b_scales_offsets,
    ElementA* a_base_as_int,
    ElementB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    cutlass::bfloat16_t* b_scales_base_as_int,
    int64_t n,
    int64_t k,
    bool per_act_token,
    bool per_out_ch,
    // MXFP4A8: optional per-token+per-block activation scale (N-indexed, bf16).
    // When as_offsets/as_base are non-null the per-expert pointer is computed as
    // as_base + expert_offset * (k / act_scale_group), mirroring the token
    // grouping of the activation `a_offsets`. nullptr for int4a8 (per-tensor).
    cutlass::bfloat16_t** as_offsets = nullptr,
    cutlass::bfloat16_t* as_base_as_int = nullptr,
    int64_t act_scale_group = 0,
    // Weight-scale K-wise quant group size. int4a8 uses 128; mxfp4a8 (E8M0
    // block) uses 32. The per-expert weight-scale buffer holds n*k/group bf16
    // elements, so the per-expert pointer must advance by that amount. Defaults
    // to 128 so the int4a8 call site (which passes 128) is byte-identical.
    int64_t weight_scale_group = 128,
    // MXFP4A8: per-expert activation-scale stride array [E, 2] (int64). Element
    // [e][0] = the per-expert token stride of the block-scale buffer, PADDED up
    // to an even count so the TMA scale_k gmem stride (M_pad * sizeof(Array<bf16,4>)
    // = M_pad*8 bytes) is a multiple of 16 (TMA requirement). The activation-scale
    // buffer is per-expert-concatenated with PADDED token blocks, so the per-expert
    // pointer must advance by the exclusive cumsum of these padded strides -- NOT
    // by the (real) expert_offset used for the M-contiguous activation tensor.
    // nullptr for int4a8 (act-scale path disabled).
    int64_t const* as_strides = nullptr) {
  int expert_id = threadIdx.x;
  int32_t expert_offset = expert_offsets[expert_id];

  a_offsets[expert_id] = a_base_as_int + expert_offset * k;
  b_offsets[expert_id] = b_base_as_int + expert_id * k * n / 2;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base_as_int + (per_act_token ? expert_offset : 0);
  b_scales_offsets[expert_id] =
      b_scales_base_as_int + (per_out_ch ? expert_id * n * k / weight_scale_group : expert_id);
  if (as_offsets != nullptr && as_base_as_int != nullptr) {
    // Exclusive cumsum of the PADDED per-expert token strides (as_strides[j][0],
    // laid out as [E,2] so element j is at index j*2). Falls back to the real
    // expert_offset when no stride array is supplied (defensive; the mxfp4a8
    // caller always passes as_strides). E is tiny (<=256) so the serial scan is
    // negligible in this single-block launch.
    int64_t as_tok_off = expert_offset;
    if (as_strides != nullptr) {
      as_tok_off = 0;
      for (int j = 0; j < expert_id; ++j) {
        as_tok_off += as_strides[j * 2];
      }
    }
    as_offsets[expert_id] = as_base_as_int + as_tok_off * (k / act_scale_group);
  }
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
__global__ void int4_fp8_get_group_gemm_starts_3d(
    ElementA** a_offsets,
    ElementB** b_offsets,
    ElementC** out_offsets,
    ElementAccumulator** a_scales_offsets,
    cutlass::bfloat16_t** b_scales_offsets,
    ElementA* a_base_as_int,
    ElementB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    cutlass::bfloat16_t* b_scales_base_as_int,
    int64_t n,
    int64_t m,
    int64_t k,
    bool per_act_token,
    bool per_out_ch,
    int num_experts) {
  int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert_id >= num_experts) return;

  int64_t a_offset = expert_id * m * k;
  int64_t b_offset = expert_id * k * n / 2;
  int64_t out_offset = expert_id * m * n;
  int64_t a_scales_offset = 0;
  int64_t b_scales_offset = per_out_ch ? expert_id * n * 4 * k / 512 : expert_id;

  a_offsets[expert_id] = a_base_as_int + a_offset;
  b_offsets[expert_id] = b_base_as_int + b_offset;
  out_offsets[expert_id] = out_base_as_int + out_offset;
  a_scales_offsets[expert_id] = a_scales_base_as_int + a_scales_offset;
  b_scales_offsets[expert_id] = b_scales_base_as_int + b_scales_offset;
}

#define __CALL_W4A8_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE)                              \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                                        \
    int4_fp8_get_group_gemm_starts<cutlass::float_e4m3_t, cutlass::int8_t, C_TYPE, float> \
        <<<1, num_experts, 0, stream>>>(                                                  \
            static_cast<int32_t*>(expert_offsets.data_ptr()),                             \
            static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),                      \
            static_cast<cutlass::int8_t**>(b_ptrs.data_ptr()),                            \
            static_cast<C_TYPE**>(out_ptrs.data_ptr()),                                   \
            static_cast<float**>(a_scales_ptrs.data_ptr()),                               \
            static_cast<cutlass::bfloat16_t**>(b_scales_ptrs.data_ptr()),                 \
            static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),                    \
            static_cast<cutlass::int8_t*>(b_tensors.data_ptr()),                          \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                                 \
            static_cast<float*>(a_scales.data_ptr()),                                     \
            static_cast<cutlass::bfloat16_t*>(b_scales.data_ptr()),                       \
            out_tensors.size(1),                                                          \
            a_tensors.size(1),                                                            \
            per_act_token,                                                                \
            per_out_ch,                                                                   \
            as_ptrs_raw,                                                                  \
            as_base_raw,                                                                  \
            act_scale_group,                                                              \
            weight_scale_group,                                                           \
            as_strides_raw);                                                              \
  }

#define __CALL_W4A8_GET_STARTS_KERNEL_3D(TENSOR_C_TYPE, C_TYPE)                              \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                                           \
    int4_fp8_get_group_gemm_starts_3d<cutlass::float_e4m3_t, cutlass::int8_t, C_TYPE, float> \
        <<<1, num_experts, 0, stream>>>(                                                     \
            static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),                         \
            static_cast<cutlass::int8_t**>(b_ptrs.data_ptr()),                               \
            static_cast<C_TYPE**>(out_ptrs.data_ptr()),                                      \
            static_cast<float**>(a_scales_ptrs.data_ptr()),                                  \
            static_cast<cutlass::bfloat16_t**>(b_scales_ptrs.data_ptr()),                    \
            static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),                       \
            static_cast<cutlass::int8_t*>(b_tensors.data_ptr()),                             \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                                    \
            static_cast<float*>(a_scales.data_ptr()),                                        \
            static_cast<cutlass::bfloat16_t*>(b_scales.data_ptr()),                          \
            out_tensors.size(2),                                                             \
            a_tensors.size(1),                                                               \
            a_tensors.size(2),                                                               \
            per_act_token,                                                                   \
            per_out_ch,                                                                      \
            num_experts);                                                                    \
  }

namespace {

void run_int4_fp8_get_group_gemm_starts(
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
    // MXFP4A8: optional per-token+per-block activation scale plumbing. When
    // as_ptrs/act_scales are provided, per-expert bf16 activation-scale pointers
    // are emitted (N-indexed like the activation). Left empty for int4a8.
    std::optional<torch::Tensor> as_ptrs = std::nullopt,
    std::optional<torch::Tensor> act_scales = std::nullopt,
    int64_t act_scale_group = 0,
    // Weight-scale K-wise quant group size (int4a8: 128, mxfp4a8: 32). Passed to
    // the launch macro so the per-expert weight-scale pointer advances by the
    // correct n*k/group. Defaults to 128 (int4a8) for byte-identical behaviour.
    int64_t weight_scale_group = 128,
    // MXFP4A8: per-expert activation-scale stride tensor [E, 2] (int64). Used to
    // compute the exclusive cumsum of PADDED per-expert token strides so the
    // per-expert act-scale pointer lands on the correctly-padded (16B-aligned)
    // sub-buffer. Left empty for int4a8.
    std::optional<torch::Tensor> as_strides = std::nullopt) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kBFloat16);

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;

  // MXFP4A8: raw pointers for the optional activation block-scale path.
  cutlass::bfloat16_t** as_ptrs_raw = nullptr;
  cutlass::bfloat16_t* as_base_raw = nullptr;
  int64_t const* as_strides_raw = nullptr;
  if (as_ptrs.has_value() && act_scales.has_value()) {
    TORCH_CHECK(act_scales->dtype() == torch::kBFloat16, "activation block-scale must be bf16");
    as_ptrs_raw = static_cast<cutlass::bfloat16_t**>(as_ptrs->data_ptr());
    as_base_raw = static_cast<cutlass::bfloat16_t*>(act_scales->data_ptr());
    if (as_strides.has_value()) {
      TORCH_CHECK(as_strides->dtype() == torch::kInt64, "as_strides must be int64");
      as_strides_raw = static_cast<int64_t const*>(as_strides->data_ptr());
    }
  }

  auto stream = at::cuda::getCurrentCUDAStream(expert_offsets.device().index());

  if (a_tensors.dim() == 3) {
    if (false) {
    }
    __CALL_W4A8_GET_STARTS_KERNEL_3D(torch::kBFloat16, cutlass::bfloat16_t)
    __CALL_W4A8_GET_STARTS_KERNEL_3D(torch::kFloat16, half)
    else {
      TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
    }
  } else {
    if (false) {
    }
    __CALL_W4A8_GET_STARTS_KERNEL(torch::kBFloat16, cutlass::bfloat16_t)
    __CALL_W4A8_GET_STARTS_KERNEL(torch::kFloat16, half)
    else {
      TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
    }
  }
}

}  // namespace
