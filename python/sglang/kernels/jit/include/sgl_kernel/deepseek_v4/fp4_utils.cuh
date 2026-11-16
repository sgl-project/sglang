#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>
#include <sgl_kernel/quantization/fp4.cuh>

#ifndef USE_ROCM
#include <cuda_fp4.h>
#endif

// FP4 (e2m1) helpers: per-32 UE8M0 for the indexer, per-16 E4M3 for compressed KV.
// The generic e2m1 / UE8M0 helpers live in sgl_kernel/quantization/fp4.cuh and are
// forwarded below to their old names; remove the forwarding once callers use that header.

namespace sglang {

namespace deepseek_v4::fp4 {

using namespace device::fp4;

/// Elements sharing one ue8m0 scale.
constexpr uint32_t kBlockSize = 32;
/// Compressed-KV elements sharing one E4M3 scale.
constexpr uint32_t kCompressedKVBlockSize = 16;
#ifdef USE_ROCM
/// \brief `float(__nv_fp8_e4m3(raw))` for `raw` in `[2^-9, 448]`: RNE onto the E4M3FN
/// grid, gfx950's `v_cvt_pk_fp8_f32` in hardware, software elsewhere (gfx942's fp8 is FNUZ).
SGL_DEVICE float e4m3_round_rn(float raw) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
  const auto packed = __builtin_amdgcn_cvt_pk_fp8_f32(raw, raw, 0, false);
  return __builtin_amdgcn_cvt_f32_fp8(packed, 0);
#else
  const int exponent = ilogbf(raw);
  const float step = exponent < -6 ? 0x1p-9f : ldexpf(1.0f, exponent - 3);
  return rintf(raw / step) * step;
#endif
}
#endif

/// \brief Round amax / 6 to a positive finite E4M3 scale, ties to even.
SGL_DEVICE float compressed_kv_scale(float amax) {
  const auto raw = fminf(fmaxf(amax * (1.0f / kMax), 0x1p-9f), 448.0f);
#ifdef USE_ROCM
  return e4m3_round_rn(raw);
#else
  return static_cast<float>(__nv_fp8_e4m3(raw));
#endif
}

/// \brief Quantize compressed KV with its E4M3 scale and return dequantized values.
SGL_DEVICE fp32x2_t fake_quant_compressed_kv_x2(fp32x2_t x, float scale) {
  const fp32x2_t scaled{__fdiv_rn(x.x, scale) + 0.0f, __fdiv_rn(x.y, scale) + 0.0f};
#ifdef USE_ROCM
  const fp32x2_t grid{e2m1_value(e2m1_code(scaled.x)), e2m1_value(e2m1_code(scaled.y))};
#else
  const auto code = __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);
  const auto grid = device::cast<fp32x2_t>(fp16x2_t{__nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1)});
#endif
  return {grid.x * scale, grid.y * scale};
}

}  // namespace deepseek_v4::fp4

}  // namespace sglang
