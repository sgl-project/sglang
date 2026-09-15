#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#ifndef USE_ROCM
#include <cuda_fp4.h>
#endif

// FP4 (e2m1) helpers: per-32 UE8M0 for the indexer, per-16 E4M3 for compressed KV.

namespace sglang {

namespace deepseek_v4::fp4 {

/// Largest finite e2m1 value.
constexpr float kMax = 6.0f;
/// `6 * 2^-126`, the amax floor `torch_quant.fake_quant_fp4` clamps to.
constexpr float kAmaxFloor = 6.0f * 1.1754943508222875e-38f;
/// Elements sharing one ue8m0 scale.
constexpr uint32_t kBlockSize = 32;
/// Compressed-KV elements sharing one E4M3 scale.
constexpr uint32_t kCompressedKVBlockSize = 16;

/// \brief Round amax / 6 to a positive finite E4M3 scale, ties to even.
SGL_DEVICE float compressed_kv_scale(float amax) {
  const auto raw = fminf(fmaxf(amax * (1.0f / kMax), 0x1p-9f), 448.0f);
  return static_cast<float>(__nv_fp8_e4m3(raw));
}

/// \brief Quantize compressed KV with its E4M3 scale and return dequantized values.
SGL_DEVICE fp32x2_t fake_quant_compressed_kv_x2(fp32x2_t x, float scale) {
  const fp32x2_t scaled{__fdiv_rn(x.x, scale) + 0.0f, __fdiv_rn(x.y, scale) + 0.0f};
  const auto code = __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);
  const auto grid = device::cast<fp32x2_t>(fp16x2_t{__nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1)});
  return {grid.x * scale, grid.y * scale};
}

/// \brief Per-block ue8m0 scale and its reciprocal, from the block's absmax.
///
/// Both come out of one biased exponent, so the reciprocal costs a subtract
/// rather than a division.
SGL_DEVICE fp32x2_t block_scale(float amax) {
  const auto exponent = fp8::cast_to_ue8m0(fmaxf(amax, kAmaxFloor) * (1.0f / kMax));
  return {__uint_as_float(static_cast<uint32_t>(exponent) << 23), fp8::inv_scale_ue8m0(exponent)};
}

/// \brief Round a pair onto the e2m1 grid and back, through `scale`.
///
/// `cvt.rn.satfinite.e2m1x2.f32` rounds to nearest even and saturates to +-6;
/// every e2m1 value is exact in fp16. Adding `0.0f` during scaling clears negative
/// zero to match `torch.sign(0) == 0` in `torch_quant.round_fp4`.
SGL_DEVICE fp32x2_t fake_quant_x2(fp32x2_t x, float scale, float inv_scale) {
  const fp32x2_t scaled{__fmaf_rn(x.x, inv_scale, 0.0f), __fmaf_rn(x.y, inv_scale, 0.0f)};
  const auto code = __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);
  const auto grid = device::cast<fp32x2_t>(fp16x2_t{__nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1)});
  return {grid.x * scale, grid.y * scale};
}

}  // namespace deepseek_v4::fp4

}  // namespace sglang
