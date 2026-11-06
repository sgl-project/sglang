// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace sglang {

namespace device {
namespace rdna4_nvfp4 {

constexpr int kBlockThreads = 256;
constexpr int kNvfp4BlockSize = 16;

/// \brief Decode one packed E2M1 (FP4) nibble to fp32.
SGL_DEVICE inline float decode_e2m1(uint8_t bits) {
  const uint8_t magnitude = bits & 0x7;
  float value;
  switch (magnitude) {
    case 0:
      value = 0.0f;
      break;
    case 1:
      value = 0.5f;
      break;
    case 2:
      value = 1.0f;
      break;
    case 3:
      value = 1.5f;
      break;
    case 4:
      value = 2.0f;
      break;
    case 5:
      value = 3.0f;
      break;
    case 6:
      value = 4.0f;
      break;
    default:
      value = 6.0f;
      break;
  }
  return (bits & 0x8) != 0 ? -value : value;
}

/// \brief Decode one E4M3FN (OCP FP8) byte to fp32.
SGL_DEVICE inline float decode_e4m3fn(uint8_t bits) {
  const uint8_t sign = bits >> 7;
  const uint8_t exponent = (bits >> 3) & 0xf;
  const uint8_t mantissa = bits & 0x7;

  float value;
  if (exponent == 0) {
    value = static_cast<float>(mantissa) * 0x1p-9f;
  } else if (exponent == 0xf && mantissa == 0x7) {
    value = nanf("");
  } else {
    value = ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
  }
  return sign != 0 ? -value : value;
}

/// \brief W4A16 GEMV: one block reduces one output element over K.
/// \tparam DType The activation and output type (bf16_t or fp16_t).
template <typename DType>
__global__ void linear_kernel(
    const DType* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ weight_scale,
    const float* __restrict__ weight_global_scale,
    DType* __restrict__ output,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "The RDNA4 NVFP4 kernel must only be compiled for gfx1201."
#endif
  __shared__ float partial[kBlockThreads];

  const int64_t output_index = static_cast<int64_t>(blockIdx.x);
  const int64_t output_row = output_index / size_n;
  const int64_t output_col = output_index - output_row * size_n;
  const int64_t packed_k = size_k / 2;

  float accumulator = 0.0f;
  for (int64_t packed_index = threadIdx.x; packed_index < packed_k; packed_index += blockDim.x) {
    const uint8_t packed_weight = weight[output_col * packed_k + packed_index];
    const int64_t input_index = packed_index * 2;
    const float scale =
        decode_e4m3fn(weight_scale[output_col * (size_k / kNvfp4BlockSize) + input_index / kNvfp4BlockSize]);
    const float input_low = device::cast<float>(input[output_row * size_k + input_index]);
    const float input_high = device::cast<float>(input[output_row * size_k + input_index + 1]);
    accumulator +=
        (input_low * decode_e2m1(packed_weight & 0xf) + input_high * decode_e2m1(packed_weight >> 4)) * scale;
  }

  partial[threadIdx.x] = accumulator;
  __syncthreads();
  for (int offset = kBlockThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[output_index] = device::cast<DType>(partial[0] * weight_global_scale[0]);
  }
}

}  // namespace rdna4_nvfp4
}  // namespace device

/// \brief Host launcher for the gfx1201 NVFP4 decode GEMV.
///
/// Consumes the canonical ModelOpt checkpoint layout with no repacking:
/// packed E2M1 weights of shape [N, K/2] and E4M3FN block scales of shape
/// [N, K/16], scaled by one fp32 global scale.
///
/// \tparam DType The activation and output type (bf16_t or fp16_t).
template <typename DType>
struct Rdna4Nvfp4LinearKernel {
  /// \brief Launch the kernel; every argument is verified before launch.
  /// \param input Activations, [M, K] with M == 1.
  /// \param weight Packed E2M1 weights, [N, K/2] uint8.
  /// \param weight_scale Block scales, [N, K/16] float8_e4m3fn.
  /// \param weight_global_scale One fp32 scalar.
  /// \param output Destination, [M, N].
  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView weight_scale,
      tvm::ffi::TensorView weight_global_scale,
      tvm::ffi::TensorView output) {
    using namespace host;

    SymbolicSize size_m{"M"};
    SymbolicSize size_n{"N"};
    SymbolicSize size_k{"K"};
    SymbolicSize packed_k{"packed K"};
    SymbolicSize scale_k{"scale K"};
    SymbolicDevice gpu_device;
    gpu_device.set_options<kDLGPU>();

    TensorMatcher({size_m, size_k}).with_dtype<DType>().with_device(gpu_device).verify(input);
    TensorMatcher({size_n, packed_k}).with_dtype<uint8_t>().with_device(gpu_device).verify(weight);
    TensorMatcher({size_n, scale_k}).with_device(gpu_device).verify(weight_scale);
    TensorMatcher({1}).with_dtype<float>().with_device(gpu_device).verify(weight_global_scale);
    TensorMatcher({size_m, size_n}).with_dtype<DType>().with_device(gpu_device).verify(output);

    CHECK_HOST(size_m.unwrap() == 1) << "rdna4_nvfp4: the HIP decode kernel requires M=1, got " << size_m.unwrap();
    CHECK_HOST(size_n.unwrap() > 0) << "rdna4_nvfp4: N must be positive";
    CHECK_HOST(size_k.unwrap() > 0 && size_k.unwrap() % device::rdna4_nvfp4::kNvfp4BlockSize == 0)
        << "rdna4_nvfp4: K must be positive and divisible by " << device::rdna4_nvfp4::kNvfp4BlockSize << ", got "
        << size_k.unwrap();
    CHECK_HOST(packed_k.unwrap() * 2 == size_k.unwrap())
        << "rdna4_nvfp4: packed weight K must equal K/2, got " << packed_k.unwrap() << " for K=" << size_k.unwrap();
    CHECK_HOST(scale_k.unwrap() * device::rdna4_nvfp4::kNvfp4BlockSize == size_k.unwrap())
        << "rdna4_nvfp4: weight scale K must equal K/" << device::rdna4_nvfp4::kNvfp4BlockSize << ", got "
        << scale_k.unwrap() << " for K=" << size_k.unwrap();
    CHECK_HOST(
        weight_scale.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && weight_scale.dtype().bits == 8 &&
        weight_scale.dtype().lanes == 1)
        << "rdna4_nvfp4: weight_scale must be float8_e4m3fn";

    const int64_t output_count = size_m.unwrap() * size_n.unwrap();
    CHECK_HOST(output_count <= std::numeric_limits<uint32_t>::max())
        << "rdna4_nvfp4: M*N exceeds the supported launch range, got " << output_count;

    const auto* input_ptr = static_cast<const DType*>(input.data_ptr());
    const auto* weight_ptr = static_cast<const uint8_t*>(weight.data_ptr());
    const auto* weight_scale_ptr = static_cast<const uint8_t*>(weight_scale.data_ptr());
    const auto* global_scale_ptr = static_cast<const float*>(weight_global_scale.data_ptr());
    auto* output_ptr = static_cast<DType*>(output.data_ptr());

    LaunchKernel(static_cast<uint32_t>(output_count), device::rdna4_nvfp4::kBlockThreads, gpu_device.unwrap())(
        device::rdna4_nvfp4::linear_kernel<DType>,
        input_ptr,
        weight_ptr,
        weight_scale_ptr,
        global_scale_ptr,
        output_ptr,
        size_m.unwrap(),
        size_n.unwrap(),
        size_k.unwrap());
  }
};

}  // namespace sglang
