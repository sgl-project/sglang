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

#pragma once

#ifdef Layout
#undef Layout
#endif
#define Layout at::Layout
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/cuda/Exceptions.h>
#undef Layout
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <array>
#include <cstdint>
#include <cub/cub.cuh>
#include <optional>
#include <string>
#include <vector>

#include "ada_blockwise_gemm/sm89_fp8_gemm_1d1d.cuh"

#undef CHECK_CONTIGUOUS
#undef CHECK_TH_CUDA
#undef CHECK_INPUT_SM89
#undef CHECK_TYPE
#undef FP8_BLOCK_SCALING_SF_DTYPE

#define CHECK_TYPE(x, st) \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_SM89(x, st) \
  CHECK_TH_CUDA(x);             \
  CHECK_CONTIGUOUS(x);          \
  CHECK_TYPE(x, st)
#define FP8_BLOCK_SCALING_SF_DTYPE torch::ScalarType::Float

void check_input_dtypes(torch::Tensor const& mat, torch::Tensor const& matScale) {
  TORCH_CHECK(
      mat.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "Matrix dtype must be FP8 (the matrix will be dequantized on the fly).");

  CHECK_INPUT_SM89(matScale, FP8_BLOCK_SCALING_SF_DTYPE);
}

template <typename TileShape>
void launch_sm89_fp8_blockwise_scaled_mm(
    const void* mat_a,
    const void* mat_b,
    void* mat_d,
    const float* scales_a,
    const float* scales_b,
    uint32_t shape_m,
    uint32_t shape_n,
    uint32_t shape_k,
    cudaStream_t stream) {
  using ElementInput = cute::float_e4m3_t;
  using ElementOutput = cute::bfloat16_t;
  using ElementAccum = float;
  using ElementBlockScale = float;
  static constexpr int Stages = 3;
  using KT = ada_blockwise_gemm::AdaBlockwiseGemmTraits<
      ElementInput,
      ElementOutput,
      ElementAccum,
      ElementBlockScale,
      Stages,
      TileShape::kM,
      TileShape::kN,
      TileShape::kK>;
  using GemmKernel = ada_blockwise_gemm::AdaBlockwiseGemmKernel<KT>;

  static constexpr int kSmemSize = KT::kSmemSize;
  static constexpr int kThreadCount = KT::kThreadCount;
  int grid_m = (shape_m + KT::kTileM - 1) / KT::kTileM;
  int grid_n = (shape_n + KT::kTileN - 1) / KT::kTileN;
  int grid_k = 1;
  dim3 grid = dim3(grid_m, grid_n, grid_k);
  dim3 block = dim3(kThreadCount, 1, 1);

  auto result = cudaFuncSetAttribute(
      ada_blockwise_gemm::sm89_fp8_gemm_1d1d_impl<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
  TORCH_CHECK(result == cudaSuccess, "sm89 gemm kernel cannot launch: %s", cudaGetErrorString(result));

  ada_blockwise_gemm::sm89_fp8_gemm_1d1d_impl<GemmKernel>
      <<<grid, block, kSmemSize, stream>>>(shape_m, shape_n, shape_k, mat_a, mat_b, mat_d, scales_a, scales_b);

  result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "sm89 gemm kernel runtime error: %s", cudaGetErrorString(result));
}

torch::Tensor cutlass_gemm_blockwise_sm89_fp8_dispatch(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& scales_a, const torch::Tensor& scales_b) {
  check_input_dtypes(a, scales_a);
  check_input_dtypes(b, scales_b);

  TORCH_CHECK(a.dim() == 2, "a must be a matrix");
  TORCH_CHECK(b.dim() == 2, "b must be a matrix");
  TORCH_CHECK(
      a.sizes()[1] == b.sizes()[1],
      "a and b shapes cannot be multiplied (",
      a.sizes()[0],
      "x",
      a.sizes()[1],
      " and ",
      b.sizes()[0],
      "x",
      b.sizes()[1],
      ")");

  auto const m = a.sizes()[0];
  auto const n = b.sizes()[0];
  auto const k = a.sizes()[1];
  TORCH_CHECK(k % 128 == 0, "K must be a multiple of 128, (K=", k, ")");
  TORCH_CHECK(n % 16 == 0, "N must be a multiple of 16, (N=", n, ")");

  at::Tensor out = at::detail::empty_cuda({m, n}, at::ScalarType::BFloat16, a.device(), std::nullopt);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  float const* scales_aPtr = scales_a.data_ptr<float>();
  float const* scales_bPtr = scales_b.data_ptr<float>();

  if (m >= 256) {
    using TileShape = cutlass::gemm::GemmShape<64, 128, 128>;
    launch_sm89_fp8_blockwise_scaled_mm<TileShape>(
        reinterpret_cast<const __nv_fp8_e4m3*>(a.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        scales_aPtr,
        scales_bPtr,
        m,
        n,
        k,
        stream);
  } else {
    using TileShape = cutlass::gemm::GemmShape<32, 128, 128>;
    launch_sm89_fp8_blockwise_scaled_mm<TileShape>(
        reinterpret_cast<const __nv_fp8_e4m3*>(a.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        scales_aPtr,
        scales_bPtr,
        m,
        n,
        k,
        stream);
  }

  return out;
}