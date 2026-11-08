// SPDX-License-Identifier: Apache-2.0
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstddef>
#include <cstdint>

namespace sglang {

template <bool Vectorized>
__global__ void groupwise_int8_dequant_kernel(
    bf16_t* __restrict__ dst,
    const int8_t* __restrict__ src,
    const float* __restrict__ scales,
    size_t count,
    size_t width,
    size_t groups) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if constexpr (Vectorized) {
    if (i >= count / 8) return;
    using Input = device::AlignedVector<int8_t, 8>;
    using Output = device::AlignedVector<bf16_t, 8>;
    const auto values = device::load_as<Input>(src, i);
    const float scale = scales[i / 16];
    Output result;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      result[j] = __float2bfloat16_rn(__fmul_rn(static_cast<float>(values[j]), scale));
    }
    device::store_as<Output>(dst, result, i);
  } else {
    if (i >= count) return;
    const float scale = scales[(i / width) * groups + (i % width) / 128];
    dst[i] = __float2bfloat16_rn(__fmul_rn(static_cast<float>(src[i]), scale));
  }
}

void groupwise_int8_dequant(
    tvm::ffi::TensorView dst, tvm::ffi::TensorView src, tvm::ffi::TensorView scales) {
  using namespace host;
  SymbolicSize rows = {"rows"}, width = {"width"};
  SymbolicSize src_stride = {"src_stride"}, dst_stride = {"dst_stride"}, scale_stride = {"scale_stride"};
  SymbolicDevice device_;
  TensorMatcher({rows, width})
      .with_strides({src_stride, 1})
      .with_dtype<int8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(src);
  TensorMatcher({rows, width})
      .with_strides({dst_stride, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device_)
      .verify(dst);
  const size_t w = width.unwrap();
  const size_t groups = div_ceil(w, size_t{128});
  TensorMatcher({rows, static_cast<int64_t>(groups)})
      .with_strides({scale_stride, 1})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(scales);
  const size_t count = rows.unwrap() * w;
  if (count == 0) return;
  RuntimeCheck(
      rows.unwrap() <= 1 ||
          (src_stride.unwrap() == w && dst_stride.unwrap() == w && scale_stride.unwrap() == groups),
      "Weights, output and scales must be contiguous");
  auto* dst_ptr = static_cast<bf16_t*>(dst.data_ptr());
  const auto* src_ptr = static_cast<const int8_t*>(src.data_ptr());
  const auto* scale_ptr = static_cast<const float*>(scales.data_ptr());
  const bool vectorized = w % 128 == 0 && reinterpret_cast<uintptr_t>(src_ptr) % 8 == 0 &&
                          reinterpret_cast<uintptr_t>(dst_ptr) % 16 == 0;
  if (vectorized) {
    LaunchKernel(div_ceil(count / 8, size_t{256}), 256, device_.unwrap())(
        groupwise_int8_dequant_kernel<true>, dst_ptr, src_ptr, scale_ptr, count, w, groups);
  } else {
    LaunchKernel(div_ceil(count, size_t{256}), 256, device_.unwrap())(
        groupwise_int8_dequant_kernel<false>, dst_ptr, src_ptr, scale_ptr, count, w, groups);
  }
}
}  // namespace sglang
