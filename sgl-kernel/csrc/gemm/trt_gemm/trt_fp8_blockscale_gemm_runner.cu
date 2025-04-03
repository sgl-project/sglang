// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "trt_fp8_blockscale_gemm_kernel.cuh"
#include "trt_fp8_blockscale_gemm_runner.cuh"
#include "utils.h"

namespace tensorrt_llm::kernels::fp8_blockscale_gemm {

template <typename ElementA, typename ElementB, typename ElementD>
CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::CutlassFp8BlockScaleGemmRunner() {}

template <typename ElementA, typename ElementB, typename ElementD>
CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::~CutlassFp8BlockScaleGemmRunner() {}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::gemm(
    void* mat_d,
    void const* mat_a,
    void const* mat_b,
    int shape_m,
    int shape_n,
    int shape_k,
    cudaStream_t stream,
    float const* scales_a,
    float const* scales_b) {
  constexpr bool internal_quantize_a = !std::is_same_v<ElementA, __nv_fp8_e4m3>;
  constexpr bool internal_quantize_b = !std::is_same_v<ElementB, __nv_fp8_e4m3>;
  __nv_fp8_e4m3* fp8_mat_a;
  __nv_fp8_e4m3* fp8_mat_b;
  float* per_token_per_128c_scales;
  float* per_block_scales;

  auto* ws_ptr = workspace_;
  if constexpr (internal_quantize_a || internal_quantize_b) {
    TORCH_CHECK(ws_ptr != nullptr);
  }

  if constexpr (internal_quantize_a) {
    fp8_mat_a = reinterpret_cast<__nv_fp8_e4m3*>(ws_ptr);
    ws_ptr += max_shape_m_4_align_ * shape_k * sizeof(__nv_fp8_e4m3);
    per_token_per_128c_scales = reinterpret_cast<float*>(ws_ptr);
    ws_ptr += max_shape_m_4_align_ * div_up(shape_k, 128) * sizeof(float);
  }

  if constexpr (internal_quantize_b) {
    fp8_mat_b = reinterpret_cast<__nv_fp8_e4m3*>(ws_ptr);
    ws_ptr += shape_n * shape_k * sizeof(__nv_fp8_e4m3);
    per_block_scales = reinterpret_cast<float*>(ws_ptr);
    ws_ptr += div_up(shape_n, 128) * div_up(shape_k, 128) * sizeof(float);
  }

#ifdef COMPILE_HOPPER_TMA_GEMMS
  if constexpr (internal_quantize_a && internal_quantize_b) {
    fp8_gemm_run(
        reinterpret_cast<__nv_bfloat16 const*>(mat_a),
        fp8_mat_a,
        shape_k,
        per_token_per_128c_scales,
        reinterpret_cast<__nv_bfloat16 const*>(mat_b),
        fp8_mat_b,
        shape_k,
        per_block_scales,
        reinterpret_cast<__nv_bfloat16*>(mat_d),
        shape_n,
        shape_m,
        shape_n,
        shape_k,
        stream,
        internal_quantize_a,
        internal_quantize_b);
  }

  if constexpr (internal_quantize_a && !internal_quantize_b) {
    fp8_gemm_run(
        reinterpret_cast<__nv_bfloat16 const*>(mat_a),
        fp8_mat_a,
        shape_k,
        per_token_per_128c_scales,
        nullptr,
        reinterpret_cast<__nv_fp8_e4m3*>(const_cast<void*>(mat_b)),
        shape_k,
        const_cast<float*>(scales_b),
        reinterpret_cast<__nv_bfloat16*>(mat_d),
        shape_n,
        shape_m,
        shape_n,
        shape_k,
        stream,
        internal_quantize_a,
        internal_quantize_b);
  }
#else   // COMPILE_HOPPER_TMA_GEMMS
  TORCH_CHECK(false, "fp8 blockscale gemm only support Hopper.");
#endif  // COMPILE_HOPPER_TMA_GEMMS
}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::gemm(
    __nv_fp8_e4m3 const* mat_a,
    int ld_a,
    __nv_fp8_e4m3 const* mat_b,
    int ld_b,
    __nv_bfloat16* mat_d,
    int ld_d,
    int shape_m,
    int shape_n,
    int shape_k,
    float const* scales_a,
    float const* scales_b,
    cudaStream_t stream) {
  fp8_gemm_run(
      const_cast<__nv_fp8_e4m3*>(mat_a),
      ld_a,
      const_cast<__nv_fp8_e4m3*>(mat_b),
      ld_b,
      mat_d,
      ld_d,
      shape_m,
      shape_n,
      shape_k,
      const_cast<float*>(scales_a),
      const_cast<float*>(scales_b),
      stream);
}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::moeGemm(
    void* mat_d,
    void const* mat_a,
    void const* mat_b,
    int64_t const* problem_m_offsets,
    size_t num_problems,
    size_t shape_n,
    size_t shape_k,
    cudaStream_t stream,
    float const* scales_a,
    float const* scales_b) {
  constexpr bool internal_quantize_a = !std::is_same_v<ElementA, __nv_fp8_e4m3>;
  constexpr bool internal_quantize_b = !std::is_same_v<ElementB, __nv_fp8_e4m3>;

  __nv_fp8_e4m3* fp8_mat_a;
  float* per_token_per_128c_scales;
  __nv_fp8_e4m3* fp8_mat_b;
  float* per_block_scales;
  int64_t* problem_m_padded_offsets;

  auto* ws_ptr = workspace_;
  if constexpr (internal_quantize_a || internal_quantize_b) {
    TORCH_CHECK(ws_ptr != nullptr);
  }

  if constexpr (internal_quantize_a) {
    fp8_mat_a = reinterpret_cast<__nv_fp8_e4m3*>(ws_ptr);
    ws_ptr += max_shape_m_4_align_ * shape_k * sizeof(__nv_fp8_e4m3);
    per_token_per_128c_scales = reinterpret_cast<float*>(ws_ptr);
    ws_ptr += max_shape_m_32_align_padded_ * div_up(shape_k, 128) * sizeof(float);
  } else {
    fp8_mat_a = reinterpret_cast<__nv_fp8_e4m3*>(const_cast<void*>(mat_a));
    per_token_per_128c_scales = const_cast<float*>(scales_a);
  }

  if constexpr (internal_quantize_b) {
    fp8_mat_b = reinterpret_cast<__nv_fp8_e4m3*>(ws_ptr);
    ws_ptr += num_problems * shape_n * shape_k * sizeof(__nv_fp8_e4m3);
    per_block_scales = reinterpret_cast<float*>(ws_ptr);
  } else {
    for (int i = 0; i < num_problems; i++) {
      fp8_mat_b = reinterpret_cast<__nv_fp8_e4m3*>(const_cast<void*>(mat_b));
      per_block_scales = const_cast<float*>(scales_b);
    }
  }

  problem_m_padded_offsets = reinterpret_cast<int64_t*>(ws_ptr);
  ws_ptr += (num_problems + 1) * sizeof(int64_t);

#ifdef COMPILE_HOPPER_TMA_GEMMS
  if constexpr (std::is_same_v<ElementA, __nv_bfloat16> && std::is_same_v<ElementB, __nv_bfloat16>) {
    fp8_grouped_gemm_run(
        reinterpret_cast<__nv_bfloat16 const*>(mat_a),
        fp8_mat_a,
        per_token_per_128c_scales,
        reinterpret_cast<__nv_bfloat16 const*>(mat_b),
        fp8_mat_b,
        per_block_scales,
        reinterpret_cast<__nv_bfloat16*>(mat_d),
        problem_m_offsets,
        problem_m_padded_offsets,
        num_problems,
        expected_m_,
        max_shape_m_4_align_,
        max_shape_m_32_align_padded_,
        shape_n,
        shape_k,
        stream,
        internal_quantize_a,
        internal_quantize_b);
  } else if constexpr (std::is_same_v<ElementA, __nv_bfloat16> && std::is_same_v<ElementB, __nv_fp8_e4m3>) {
    fp8_grouped_gemm_run(
        reinterpret_cast<__nv_bfloat16 const*>(mat_a),
        fp8_mat_a,
        per_token_per_128c_scales,
        nullptr,
        fp8_mat_b,
        per_block_scales,
        reinterpret_cast<__nv_bfloat16*>(mat_d),
        problem_m_offsets,
        problem_m_padded_offsets,
        num_problems,
        expected_m_,
        max_shape_m_4_align_,
        max_shape_m_32_align_padded_,
        shape_n,
        shape_k,
        stream,
        internal_quantize_a,
        internal_quantize_b);
  } else if constexpr (std::is_same_v<ElementA, __nv_fp8_e4m3> && std::is_same_v<ElementB, __nv_fp8_e4m3>) {
    fp8_grouped_gemm_run(
        nullptr,
        fp8_mat_a,
        per_token_per_128c_scales,
        reinterpret_cast<__nv_bfloat16 const*>(mat_b),
        fp8_mat_b,
        per_block_scales,
        reinterpret_cast<__nv_bfloat16*>(mat_d),
        problem_m_offsets,
        problem_m_padded_offsets,
        num_problems,
        expected_m_,
        max_shape_m_4_align_,
        max_shape_m_32_align_padded_,
        shape_n,
        shape_k,
        stream,
        internal_quantize_a,
        internal_quantize_b);
  } else {
    TORCH_CHECK(false, "fp8 blockscale gemm only support __nv_fp8_e4m3 or bfloat16 as dataType.");
  }
#else
  TORCH_CHECK(false, "fp8 blockscale gemm only support Hopper.");
#endif
}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::strideBatchGemm(
    __nv_bfloat16* mat_d,
    int ld_d,
    int stride_d,
    __nv_fp8_e4m3* mat_a,
    int ld_a,
    int stride_a,
    __nv_fp8_e4m3* mat_b,
    int ld_b,
    int stride_b,
    int num_problems,
    int shape_m,
    int shape_n,
    int shape_k,
    cudaStream_t stream,
    float* scales_a,
    int stride_scales_a,
    float* scales_b) {
  fp8_stride_batch_gemm_run(
      nullptr,
      mat_a,
      scales_a,
      ld_a,
      stride_a,
      stride_scales_a,
      nullptr,
      mat_b,
      scales_b,
      ld_b,
      stride_b,
      mat_d,
      ld_d,
      stride_d,
      num_problems,
      shape_m,
      shape_n,
      shape_k,
      stream,
      false,
      false);
}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::fp8CS1x128(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream) {
  fp8_1x128_cs(mat_quant, scales, mat, shape_x, shape_y, stream);
}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::fp8CS1x128Reshape(
    __nv_fp8_e4m3* mat_quant,
    float* scales,
    __nv_bfloat16 const* mat,
    int shape_x,
    int shape_h,
    int shape_y,
    int stride_x,
    cudaStream_t stream) {
  fp8_1x128_cs_reshape(mat_quant, scales, mat, shape_x, shape_h, shape_y, stride_x, stream);
}

template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::fp8CS128x128(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream) {
  fp8_128x128_cs(mat_quant, scales, mat, shape_x, shape_y, stream);
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getWorkspaceSizeBase(
    size_t max_shape_m, size_t shape_n, size_t shape_k, size_t num_problems) {
  max_shape_m_4_align_ = std::max(max_shape_m_4_align_, int64_t(div_up(max_shape_m, 4) * 4));
  if (expected_m_ == 0) {
    expected_m_ = div_up(max_shape_m_4_align_, num_problems);
  }
  max_shape_m_32_align_padded_ = int64_t(div_up(max_shape_m + num_problems * 31, 32) * 32);

  constexpr bool internal_quantize_a = !std::is_same_v<ElementA, __nv_fp8_e4m3>;
  constexpr bool internal_quantize_b = !std::is_same_v<ElementB, __nv_fp8_e4m3>;
  size_t total_workspace_size = 0;
  if constexpr (internal_quantize_a) {
    // fp8_mat_a
    total_workspace_size += max_shape_m_4_align_ * shape_k * sizeof(__nv_fp8_e4m3);
    // scales_a
    total_workspace_size += max_shape_m_32_align_padded_ * div_up(shape_k, 128) * sizeof(float);
  }

  if constexpr (internal_quantize_b) {
    // fp8_mat_b
    total_workspace_size += num_problems * shape_n * shape_k * sizeof(__nv_fp8_e4m3);
    // scales_b
    total_workspace_size += num_problems * div_up(shape_k, 128) * div_up(shape_n, 128) * sizeof(float);
  }

  total_workspace_size += (num_problems + 1) * sizeof(int64_t);

  return total_workspace_size;
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getWorkspaceSize(
    size_t shape_m, size_t shape_n, size_t shape_k, size_t top_k, size_t num_problems) {
  expected_m_ = shape_m;
  return getWorkspaceSizeBase(shape_m * top_k, shape_n, shape_k, num_problems);
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t
CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getFP8DataSize(int shape_m, int shape_n, bool is_act) {
  int shape_m_4_align = div_up(shape_m, 4) * 4;
  constexpr bool internal_quantize_a = !std::is_same_v<ElementA, __nv_fp8_e4m3>;
  constexpr bool internal_quantize_b = !std::is_same_v<ElementB, __nv_fp8_e4m3>;
  if (is_act && internal_quantize_a) {
    return div_up(shape_m_4_align * shape_n * sizeof(__nv_fp8_e4m3), 128) * 128;
  }

  if ((!is_act) && internal_quantize_b) {
    return div_up(shape_m * shape_n * sizeof(__nv_fp8_e4m3), 128) * 128;
  }
  return 0;
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getActScaleSize(int shape_m, int shape_k) {
  int shape_m_4_align = div_up(shape_m, 4) * 4;
  constexpr bool internal_quantize_a = !std::is_same_v<ElementA, __nv_fp8_e4m3>;
  size_t total_workspace_size = 0;
  if constexpr (internal_quantize_a) {
    // scales_a
    total_workspace_size += div_up(shape_m_4_align * div_up(shape_k, 128) * sizeof(float), 128) * 128;
  }
  return total_workspace_size;
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getWeightScaleSize(int shape_n, int shape_k) {
  constexpr bool internal_quantize_b = !std::is_same_v<ElementB, __nv_fp8_e4m3>;
  size_t total_workspace_size = 0;
  if constexpr (internal_quantize_b) {
    // scales_b
    total_workspace_size += div_up(div_up(shape_k, 128) * div_up(shape_n, 128) * sizeof(float), 128) * 128;
  }

  return total_workspace_size;
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getActWorkspaceSize(int shape_m, int shape_k) {
  return getFP8DataSize(shape_m, shape_k, true) + getActScaleSize(shape_m, shape_k);
}

template <typename ElementA, typename ElementB, typename ElementD>
size_t CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::getWeightWorkspaceSize(int shape_n, int shape_k) {
  return getFP8DataSize(shape_n, shape_k, false) + getWeightScaleSize(shape_n, shape_k);
}

template class CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
template class CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>;
template class CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_bfloat16, __nv_bfloat16>;
template class CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;

}  // namespace tensorrt_llm::kernels::fp8_blockscale_gemm
