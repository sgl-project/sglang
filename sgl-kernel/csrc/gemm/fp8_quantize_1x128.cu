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

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/cuda/Exceptions.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/cuda.h>

#include "utils.h"

#undef CHECK_CONTIGUOUS
#undef CHECK_TH_CUDA
#undef FP8_BLOCK_SCALING_SF_DTYPE

#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define FP8_BLOCK_SCALING_SF_DTYPE torch::ScalarType::Float

namespace sglang::kernels::fp8_blockscale {

inline int getMultiProcessorCount() {
  int nSM{0};
  int deviceID{0};
  CHECK_CUDA_SUCCESS(cudaGetDevice(&deviceID));
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
  return nSM;
}

template <typename T>
__device__ __host__ constexpr T div_up(T a, int b) {
  return (a + b - 1) / b;
}

using TileShape = std::tuple<uint32_t, uint32_t, uint32_t>;

template <typename T>
__forceinline__ __device__ T find_max_elem_in_warp(T value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = T(std::max(float(value), __shfl_down_sync(0xFFFFFFFF, float(value), offset)));
  }
  value = T(__shfl_sync(0xffffffff, float(value), 0));
  return value;
}

size_t getActScaleSize(int shape_m, int shape_k) {
  int shape_m_4_align = div_up(shape_m, 4) * 4;
  size_t total_workspace_size = 0;
  total_workspace_size += div_up(shape_m_4_align * div_up(shape_k, 128) * sizeof(float), 128) * 128;
  return total_workspace_size;
}

template <typename InputType, typename OutputType, typename ScaleType = float, bool USE_UE8M0 = false>
__global__ void
scale_1x128_kernel(OutputType* output, ScaleType* scales, InputType const* const input, int dim_x, int dim_y) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
  size_t scales_along_dim_x = div_up(dim_x, 128);
  size_t scales_along_dim_y = div_up(dim_y, 1);
  size_t stride_scale_dim_y = div_up(dim_y, 4) * 4;
  using Input2Type = typename std::conditional<std::is_same<InputType, half>::value, half2, __nv_bfloat162>::type;
  for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
       warp_idx < scales_along_dim_x * scales_along_dim_y;
       warp_idx += gridDim.x * blockDim.x / 32) {
    int scales_idx_y = warp_idx / scales_along_dim_x;
    int scales_idx_x = warp_idx % scales_along_dim_x;

    InputType const* input_line = input + (size_t)scales_idx_y * dim_x + scales_idx_x * 128;
    InputType input_amax = InputType(0);
    // Each thread reads 2 elements from input_line
    int lane_id = threadIdx.x % 32 * 2;

    Input2Type input_frag2[2] = {Input2Type(0, 0), Input2Type(0, 0)};
#pragma unroll
    for (int i = 0; i < 2; i++) {
      if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x) {
        break;
      } else {
        input_frag2[i] = *((Input2Type*)(input_line) + lane_id / 2);
      }
      input_line += 64;
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x) {
        break;
      } else {
        input_amax = InputType(__hmax(input_amax, __hmax(__habs(input_frag2[i].x), __habs(input_frag2[i].y))));
      }
    }

    InputType amax = find_max_elem_in_warp(input_amax);
    ScaleType quant_scale = amax != InputType(0.f) ? 448.f / ScaleType(amax) : 1.f;
    ScaleType dequant_scale;

    if constexpr (USE_UE8M0) {
      // Round dequant scale to UE8M0 (power of 2)
      ScaleType dequant_scale_raw = 1.f / quant_scale;
      __nv_fp8_e8m0 ue8m0_scale;
      ue8m0_scale.__x = __nv_cvt_float_to_e8m0(float(dequant_scale_raw), __NV_SATFINITE, cudaRoundPosInf);
      // Cast back to float automatically decodes E8M0 format
      dequant_scale = ScaleType(static_cast<float>(ue8m0_scale));
      // Recompute quant scale from rounded dequant scale for consistency
      quant_scale = dequant_scale != ScaleType(0.f) ? 1.f / dequant_scale : 1.f;
    } else {
      dequant_scale = 1.f / quant_scale;
    }

    if (lane_id == 0) {
      scales[(size_t)scales_idx_x * stride_scale_dim_y + scales_idx_y] = dequant_scale;
    }

    OutputType* output_line = output + (size_t)scales_idx_y * dim_x + scales_idx_x * 128;
#pragma unroll
    for (int i = 0; i < 2; i++) {
      if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x) {
        break;
      } else {
        ScaleType value_1 = ScaleType(input_frag2[i].x) * quant_scale;
        ScaleType value_2 = ScaleType(input_frag2[i].y) * quant_scale;
        output_line[lane_id] = OutputType(value_1);
        output_line[lane_id + 1] = OutputType(value_2);
      }
      output_line += 64;
    }
  }
#endif
}

static int kNumDeviceSMs = -1;

void fp8_1x128_cs(
    __nv_fp8_e4m3* mat_quant,
    float* scales,
    __nv_bfloat16 const* mat,
    int shape_x,
    int shape_y,
    cudaStream_t stream,
    bool use_ue8m0 = false) {
  if (kNumDeviceSMs < 0) {
    kNumDeviceSMs = sglang::kernels::fp8_blockscale::getMultiProcessorCount();
  }
  if (use_ue8m0) {
    scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, true>
        <<<kNumDeviceSMs * 8, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
  } else {
    scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3, float, false>
        <<<kNumDeviceSMs * 8, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
  }
}
}  // namespace sglang::kernels::fp8_blockscale

std::tuple<at::Tensor, at::Tensor> fp8_quantize_1x128(at::Tensor const& self, bool use_ue8m0) {
  CHECK_TH_CUDA(self);
  CHECK_CONTIGUOUS(self);

  TORCH_CHECK(self.scalar_type() == at::ScalarType::BFloat16, "Input matrix dtype must be BF16.");
  TORCH_CHECK(self.dim() == 2, "input must be a matrix");

  auto const m = self.sizes()[0];
  auto const n = self.sizes()[1];

  TORCH_CHECK(m <= std::numeric_limits<int32_t>::max(), "M must be within int32");
  TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "N must be within int32");

  // required by the sm90 fp8_block_scaling gemm kernel
  TORCH_CHECK(n % 16 == 0, "self.sizes()[1] must be a multiple of 16, but got ", n);

  auto const m_padded = (m + 4 - 1) / 4 * 4;

  // row major, add padding required by the sm90 fp8_block_scaling gemm kernel
  at::Tensor valueE4M3 =
      at::detail::empty_cuda({m_padded, n}, at::ScalarType::Float8_e4m3fn, self.device(), /* stride */ std::nullopt);
  // int64_t scaleSizeInBytes = mGemmRunner.getActScaleSize(m, n); // 128-byte aligned
  signed long long scaleSizeInBytes = sglang::kernels::fp8_blockscale::getActScaleSize(m, n);  // 128-byte aligned lanxj
  signed long long elementSize = scaleSizeInBytes / torch::elementSize(FP8_BLOCK_SCALING_SF_DTYPE);

  // col major
  at::Tensor scaleFP8SF = at::detail::empty_cuda(
      {elementSize}, FP8_BLOCK_SCALING_SF_DTYPE, self.device(), /* stride */ std::nullopt);  // 1D tensor

  __nv_fp8_e4m3* act_buffer = reinterpret_cast<__nv_fp8_e4m3*>(valueE4M3.data_ptr());
  float* act_scale_buffer = reinterpret_cast<float*>(scaleFP8SF.data_ptr());

  auto stream = at::cuda::getCurrentCUDAStream(self.get_device());

  sglang::kernels::fp8_blockscale::fp8_1x128_cs(
      act_buffer,
      act_scale_buffer,
      reinterpret_cast<__nv_bfloat16 const*>(self.data_ptr()),
      n,  // shape_x = K维度
      m,  // shape_y = M维度
      stream,
      use_ue8m0);

  return {valueE4M3.slice(0, 0, m), scaleFP8SF};
}