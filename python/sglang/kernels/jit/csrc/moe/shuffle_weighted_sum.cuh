#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace sglang {
namespace shuffle_weighted_sum {

SGL_DEVICE float accumulate(float value, float factor, float total) {
  float result;
  // Keep the slot-order FMA, including +0 * +0 for inactive slots. No FTZ.
  asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(value), "f"(factor), "f"(total));
  return result;
}

template <
    typename scalar_t,
    typename weight_t,
    typename index_t,
    int64_t kTopK,
    int kThreads,
    bool kCastWeightsToInputDType>
__global__ void reduce(
    const scalar_t* inputs,
    scalar_t* output,
    const index_t* permutation,
    const weight_t* weights,
    int64_t width,
    int64_t rows,
    int64_t input_stride,
    int64_t output_stride,
    int64_t weight_stride0,
    int64_t weight_stride1) {
  constexpr int kItems = 2;
  const int64_t token = blockIdx.x;
  const int64_t first_column = static_cast<int64_t>(blockIdx.y) * kThreads * kItems + threadIdx.x;
  if (first_column >= width) return;
  float totals[kItems] = {0.0f, 0.0f};
  for (int64_t slot = 0; slot < kTopK; ++slot) {
    const int64_t row = permutation[token * kTopK + slot];
    const bool active = row >= 0 && row < rows;
    float factor = 0.0f;
    if (active) {
      factor = device::cast<float>(weights[token * weight_stride0 + slot * weight_stride1]);
      if constexpr (kCastWeightsToInputDType && !std::is_same_v<scalar_t, weight_t>) {
        // Match scores.to(inputs.dtype) before the FP32 slot-order FMA.
        factor = device::cast<float>(device::cast<scalar_t>(factor));
      }
    }
#pragma unroll
    for (int item = 0; item < kItems; ++item) {
      const int64_t column = first_column + item * kThreads;
      float value = 0.0f;
      if (active && column < width) value = device::cast<float>(inputs[row * input_stride + column]);
      totals[item] = accumulate(value, factor, totals[item]);
    }
  }
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int64_t column = first_column + item * kThreads;
    if (column < width) output[token * output_stride + column] = device::cast<scalar_t>(totals[item]);
  }
}

}  // namespace shuffle_weighted_sum

/// \brief Reduce mapped rows in slot order without reading inactive payloads/scores.
/// \param permutation Contiguous int32/int64 map of any rank, one entry per weight.
/// \param weights Potentially strided scores; rows outside the input extent are inactive.
template <
    typename scalar_t,
    typename weight_t,
    typename index_t,
    int64_t kTopK,
    int kThreads,
    bool kCastWeightsToInputDType>
void shuffle_weighted_sum_cuda(
    tvm::ffi::TensorView inputs,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView permutation,
    tvm::ffi::TensorView weights) {
  using namespace host;
  static_assert(std::is_same_v<scalar_t, fp16_t> || std::is_same_v<scalar_t, bf16_t>);
  static_assert(
      std::is_same_v<weight_t, fp16_t> || std::is_same_v<weight_t, bf16_t> || std::is_same_v<weight_t, float>);
  static_assert(std::is_same_v<index_t, int32_t> || std::is_same_v<index_t, int64_t>);
  static_assert(kTopK >= 0 && kThreads > 0 && kThreads <= 1024);
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto rows = SymbolicSize{"input_rows"};
  auto tokens = SymbolicSize{"output_rows"};
  auto width = SymbolicSize{"width"};
  auto input_stride = SymbolicSize{"input_stride"};
  auto output_stride = SymbolicSize{"output_stride"};
  auto weight_stride0 = SymbolicSize{"weight_stride0"};
  auto weight_stride1 = SymbolicSize{"weight_stride1"};
  TensorMatcher({rows, width})
      .with_strides({input_stride, 1})
      .with_dtype<scalar_t>()
      .with_device(device)
      .verify(inputs);
  TensorMatcher({tokens, width})
      .with_strides({output_stride, 1})
      .with_dtype<scalar_t>()
      .with_device(device)
      .verify(output);
  TensorMatcher({tokens, kTopK})
      .with_strides({weight_stride0, weight_stride1})
      .with_dtype<weight_t>()
      .with_device(device)
      .verify(weights);
  auto map_dtype = SymbolicDType{};
  map_dtype.set_options<index_t>();
  map_dtype.verify(permutation.dtype());
  device.verify(permutation.device());
  CHECK_HOST(permutation.is_contiguous() && permutation.numel() == weights.numel());
  if (tokens.unwrap() == 0 || width.unwrap() == 0) return;
  constexpr int64_t kTile = kThreads * 2;
  const int64_t tiles = div_ceil(width.unwrap(), kTile);
  // These are the same x/y grid limits as the Triton fallback.
  CHECK_HOST(tokens.unwrap() <= std::numeric_limits<int32_t>::max() && tiles <= 65535);
  LaunchKernel(dim3(static_cast<uint32_t>(tokens.unwrap()), static_cast<uint32_t>(tiles)), kThreads, device.unwrap())(
      shuffle_weighted_sum::reduce<scalar_t, weight_t, index_t, kTopK, kThreads, kCastWeightsToInputDType>,
      static_cast<const scalar_t*>(inputs.data_ptr()),
      static_cast<scalar_t*>(output.data_ptr()),
      static_cast<const index_t*>(permutation.data_ptr()),
      static_cast<const weight_t*>(weights.data_ptr()),
      width.unwrap(),
      rows.unwrap(),
      input_stride.unwrap(),
      output_stride.unwrap(),
      weight_stride0.unwrap(),
      weight_stride1.unwrap());
}

}  // namespace sglang
