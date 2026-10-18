// RMSNorm with an MXFP8 epilogue: one pass emits the bf16 normalized row and its MXFP8 quantization (e4m3 codes plus
// UE8M0 per-32 scales in FlashInfer's 128x4 swizzled layout): RMSNorm followed by FlashInfer's
// mxfp8_quantize(..., is_sf_swizzled_layout=True) of the bf16 row, up to fp32 summation order.
//
// One CTA per row, kDim / 8 threads of one 16 B vector each; a 32-element scale group is 4 adjacent lanes. One more CTA
// zeroes the scales of the padding rows [M, ceil(M / 128) * 128), as mxfp8_quantize does. Built for few rows
// (latency); large batches would want several rows per CTA.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::device::rmsnorm_mxfp8 {

struct Params {
  const bf16_t* x;
  const bf16_t* weight;
  bf16_t* y;  // optional
  fp8_e4m3_t* q;
  uint8_t* scale;  // 128x4 image, ceil(M / 128) * 128 * K / 32 bytes
  int64_t x_stride;
  int64_t y_stride;
  int64_t q_stride;
  uint32_t num_rows;
  float eps;
};

// FlashInfer's positive-rounding UE8M0 of a group amax (subnormals included): the scale byte and the multiplier that
// maps the group into e4m3 range.
SGL_DEVICE uint32_t ue8m0_scale(float amax, float& inv) {
  const float normalized = amax * (1.0f / 448.0f);
  const uint32_t bits = __float_as_uint(normalized);
  const uint32_t exponent = (bits >> 23) & 255u, mantissa = bits & 0x7FFFFFu;
  const bool bump = mantissa != 0 && !(exponent == 0 && mantissa <= 0x400000u);
  const uint32_t sf = normalized <= 0.0f ? 0u : min(exponent + uint32_t(bump), 254u);
  inv = __uint_as_float(sf == 0 ? 0u : (254u - sf) << 23);
  return sf;
}

// Byte offset of (row, group) in the 128x4 swizzled scale image.
SGL_DEVICE uint32_t scale_offset(uint32_t row, uint32_t group, uint32_t num_groups) {
  return (row / 128) * (128 * num_groups) + (group / 4) * 512 + ((row % 32) * 4 + (row / 32) % 4) * 4 + group % 4;
}

template <int kDim, bool kUsePDL, bool kEarlyTrigger>
__global__ __launch_bounds__(kDim / 8) void rmsnorm_mxfp8_kernel(const __grid_constant__ Params p) {
  using vec_t = AlignedVector<bf16x2_t, 4>;  // 8 elements
  constexpr uint32_t kThreads = kDim / 8, kWarps = kThreads / kWarpThreads, kGroups = kDim / 32;
  static_assert(kDim % 256 == 0 && kThreads <= 1024, "whole warps of one 16 B vector each");
  __shared__ float s_sum[kWarps];

  const auto tid = threadIdx.x;
  const auto row = blockIdx.x, num_rows = p.num_rows;
  PDLWaitPrimary<kUsePDL>();
  if constexpr (kEarlyTrigger) PDLTriggerSecondary<kUsePDL>();

  if (row == num_rows) {
    // The padding CTA: a scale word is (row, 4 groups); word i of the last tile is
    // (group / 4) * 128 + (row % 32) * 4 + (row / 32) % 4.
    const uint32_t tile0 = num_rows / 128 * 128;
    const auto words = reinterpret_cast<uint32_t*>(p.scale + tile0 * kGroups);
    for (uint32_t i = tid; i < 128 * kGroups / 4; i += kThreads) {
      if (tile0 + (i % 4) * 32 + (i % 128) / 4 >= num_rows) words[i] = 0u;
    }
    if constexpr (!kEarlyTrigger) PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // The input first, then the weight, whose load hides behind the input's.
  vec_t x, w;
  x.load(pointer::offset<bf16_t>(p.x, row * p.x_stride), tid);
  w.load(p.weight, tid);

  float sq = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const auto [a, b] = cast<fp32x2_t>(x[j]);
    sq = fmaf(a, a, sq);
    sq = fmaf(b, b, sq);
  }
  s_sum[tid / kWarpThreads] = warp::reduce_sum(sq);
  __syncthreads();
  sq = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kWarps; ++i) {
    sq += s_sum[i];
  }

  constexpr float kScale = 1.0f / float(kDim);
  const float inv_rms = math::rsqrt(sq * kScale + p.eps);

  vec_t y;
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const auto [a, b] = cast<fp32x2_t>(x[j]);
    const auto [wa, wb] = cast<fp32x2_t>(w[j]);
    y[j] = cast<bf16x2_t>(fp32x2_t{a * inv_rms * wa, b * inv_rms * wb});
    const auto [ya, yb] = cast<fp32x2_t>(y[j]);
    amax = fmaxf(amax, fmaxf(fabsf(ya), fabsf(yb)));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  float inv;
  const uint32_t sf = ue8m0_scale(amax, inv);

  if (p.y != nullptr) y.store(p.y + row * p.y_stride, tid);
  AlignedVector<fp8x2_e4m3_t, 4> q;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const auto [ya, yb] = cast<fp32x2_t>(y[j]);
    q[j] = cast<fp8x2_e4m3_t>(fp32x2_t{ya * inv, yb * inv});
  }
  if constexpr (!kEarlyTrigger) PDLTriggerSecondary<kUsePDL>();
  q.store(p.q + row * p.q_stride, tid);
  p.scale[scale_offset(row, tid / 4, kGroups)] = static_cast<uint8_t>(sf);
}

}  // namespace sglang::device::rmsnorm_mxfp8

namespace sglang {

/**
 * \brief RMSNorm + MXFP8 quantization of the bf16 normalized row, one CTA per row.
 * \tparam kDim            row width (a multiple of 256, at most 8192).
 * \tparam kEarlyTrigger release the PDL dependents right after griddepcontrol.wait (a successor that prefetches)
 * instead of at the end.
 * \param x       [M, kDim] bf16, unit inner stride.
 * \param weight  [kDim] bf16.
 * \param y       optional [M, kDim] bf16 normalized row.
 * \param q       [M, kDim] e4m3.
 * \param scale   >= ceil(M / 128) * 128 * kDim / 32 bytes, the 128x4 image.
 */
template <int kDim, bool kEarlyTrigger>
inline void rmsnorm_mxfp8(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weight,
    tvm::ffi::Optional<tvm::ffi::TensorView> y,
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView scale,
    double eps) {
  using namespace host;
  auto M = SymbolicSize{"num_rows"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({M, kDim}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device_).ensure_alignment(16).verify(
      x);
  TensorMatcher({kDim}).with_dtype<bf16_t>().with_device(device_).ensure_alignment(16).verify(weight);
  TensorMatcher({M, kDim})
      .with_strides({-1, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device(device_)
      .ensure_alignment(16)
      .verify(q);
  TensorMatcher({-1}).with_dtype<uint8_t>().with_device(device_).ensure_alignment(16).verify(scale);
  if (y.has_value()) {
    TensorMatcher({M, kDim})
        .with_strides({-1, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(y.value());
  }
  const auto num_rows = M.unwrap();
  CHECK_HOST(scale.size(0) >= (num_rows + 127) / 128 * 128 * (kDim / 32)) << scale.size(0);
  if (num_rows == 0) return;
  const auto params = device::rmsnorm_mxfp8::Params{
      .x = static_cast<const bf16_t*>(x.data_ptr()),
      .weight = static_cast<const bf16_t*>(weight.data_ptr()),
      .y = y.has_value() ? static_cast<bf16_t*>(y.value().data_ptr()) : nullptr,
      .q = static_cast<fp8_e4m3_t*>(q.data_ptr()),
      .scale = static_cast<uint8_t*>(scale.data_ptr()),
      .x_stride = x.stride(0),
      .y_stride = y.has_value() ? y.value().stride(0) : 0,
      .q_stride = q.stride(0),
      .num_rows = static_cast<uint32_t>(num_rows),
      .eps = static_cast<float>(eps),
  };
  // The extra CTA zeroes the padding scale rows, when there are any.
  const auto num_ctas = static_cast<uint32_t>(num_rows + (num_rows % 128 != 0 ? 1 : 0));
  constexpr auto kernel = device::rmsnorm_mxfp8::rmsnorm_mxfp8_kernel<kDim, true, kEarlyTrigger>;
  LaunchKernel(num_ctas, kDim / 8, device_.unwrap()).enable_pdl(true)(kernel, params);
}

}  // namespace sglang
