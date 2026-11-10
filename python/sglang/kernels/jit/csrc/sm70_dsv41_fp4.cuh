// SPDX-License-Identifier: Apache-2.0
// SM70 DeepSeek-V4.1 CSA2 FP4/FP8 helpers.
// Rounding matches python/sglang/srt/layers/attention/dsv4/sm70_csa2_reference.py:
//   RoPE in fp32 -> round to fp16 -> quant. E2M1 is torch.round RNE, not midpoint
//   thresholds. Do not use cvt.rn.satfinite.e2m1x2 (SM100) or PDL/TMA/wgmma.
#pragma once

#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang::sm70_dsv41 {

inline constexpr int kHeadDim = 512;
inline constexpr int kRopeDim = 64;
inline constexpr int kIndexDim = 128;
inline constexpr int kKvFp4Block = 16;
inline constexpr int kIndexFp4Block = 32;
inline constexpr int kSwaFp8Block = 32;
inline constexpr int kKvPayloadBytes = kHeadDim / 2;              // 256
inline constexpr int kKvScaleBytes = kHeadDim / kKvFp4Block;      // 32
inline constexpr int kKvRowBytes = kKvPayloadBytes + kKvScaleBytes;  // 288
inline constexpr int kIndexPayloadBytes = kIndexDim / 2;          // 64
inline constexpr int kIndexScaleBytes = kIndexDim / kIndexFp4Block;  // 4
inline constexpr int kIndexRowBytes = kIndexPayloadBytes + kIndexScaleBytes;  // 68
inline constexpr int kSwaPayloadBytes = kHeadDim;                 // 512 e4m3
inline constexpr int kSwaScaleBytes = kHeadDim / kSwaFp8Block;    // 16
inline constexpr int kSwaRowBytes = kSwaPayloadBytes + kSwaScaleBytes;  // 528

inline constexpr float kFp4Max = 6.0f;
inline constexpr float kFp8Max = 448.0f;
inline constexpr float kE4m3MinNormalScale = 0.001953125f;  // 2^-9
inline constexpr float kFp4Ue8m0AmaxFloor = 6.0f * 1.1754943508222875e-38f;  // 6 * 2^-126
inline constexpr float kFp8ActAmaxFloor = 1.0e-4f;

/**
 * \brief Destination / RoPE rows for pack-at (write into a static table).
 *
 * ``dst_row = (pos / row_div) % dst_mod`` (no modulo when ``dst_mod == 0``),
 * ``freq_row = (pos / row_div) * freq_mul``. Positions are non-negative.
 */
template <bool kScatterAt>
SGL_DEVICE void pack_scatter_rows(uint32_t row,
                                  const int64_t* positions,
                                  int32_t row_div,
                                  int32_t freq_mul,
                                  int32_t dst_mod,
                                  int32_t* dst_row,
                                  int64_t* freq_row) {
  if constexpr (kScatterAt) {
    const int64_t grouped = positions[row] / row_div;
    int32_t d = static_cast<int32_t>(grouped);
    if (dst_mod > 0) {
      d %= dst_mod;
    }
    *dst_row = d;
    *freq_row = grouped * freq_mul;
  } else {
    *dst_row = static_cast<int32_t>(row);
    *freq_row = static_cast<int64_t>(row);
  }
}

/**
 * \brief 2**ceil(log2(x)) on the IEEE bits (HF fast_round_scale / ceil_pow2).
 * \param x Positive finite fp32.
 */
SGL_DEVICE float ceil_pow2(float x) {
  const uint32_t bits = __float_as_uint(x);
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
  exponent += static_cast<int32_t>((bits & 0x7FFFFFu) != 0u);
  return __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
}

/**
 * \brief Round fp32 in [-6, 6] onto the E2M1 grid with round-to-nearest-even.
 *
 * Matches torch.round(mag/step)*step*sign (the CSA2 oracle). Midpoint ties go
 * to the even E2M1 code, which is not the same as ax>0.25 threshold coding.
 */
SGL_DEVICE float round_e2m1(float x) {
  const float mag = fabsf(x);
  const float step = mag < 2.0f ? 0.5f : (mag < 4.0f ? 1.0f : 2.0f);
  const float rounded = rintf(mag / step) * step;
  return copysignf(rounded, x);
}

/**
 * \brief E2M1 code: low 3 bits magnitude, bit 3 sign. -0 encodes as +0.
 */
SGL_DEVICE uint8_t e2m1_code(float scaled) {
  const float mag = fabsf(scaled);
  uint8_t idx = 0;
  if (mag == 0.5f) {
    idx = 1;
  } else if (mag == 1.0f) {
    idx = 2;
  } else if (mag == 1.5f) {
    idx = 3;
  } else if (mag == 2.0f) {
    idx = 4;
  } else if (mag == 3.0f) {
    idx = 5;
  } else if (mag == 4.0f) {
    idx = 6;
  } else if (mag == 6.0f) {
    idx = 7;
  }
  if ((scaled < 0.0f) && (idx != 0)) {
    idx = static_cast<uint8_t>(idx | 0x8u);
  }
  return idx;
}

SGL_DEVICE float decode_e2m1(uint8_t code) {
  constexpr float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = mag[code & 0x7u];
  return (code & 0x8u) ? -v : v;
}

/**
 * \brief fp32 -> E4M3FN bits, round-to-nearest-even, satfinite.
 *
 * CUDA's software cvt on SM70 matches torch.float8_e4m3fn (OCP E4M3FN).
 */
SGL_DEVICE uint8_t float_to_e4m3fn(float f) {
  const __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
  return static_cast<uint8_t>(s);
}

SGL_DEVICE float e4m3fn_to_float(uint8_t b) {
  const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 24;
  const uint32_t e = (static_cast<uint32_t>(b) >> 3) & 0xFu;
  const uint32_t m = static_cast<uint32_t>(b) & 0x7u;
  if (e == 0u) {
    if (m == 0u) {
      return __uint_as_float(sign);
    }
    return __uint_as_float(sign | __float_as_uint(static_cast<float>(m) * 0.001953125f));
  }
  if (e == 15u && m == 7u) {
    return __uint_as_float(sign | 0x7FC00000u);
  }
  return __uint_as_float(sign | ((e + 120u) << 23) | (m << 20));
}

SGL_DEVICE float ue8m0_to_float(uint8_t exp) {
  return __uint_as_float(static_cast<uint32_t>(exp) << 23);
}

/**
 * \brief Rotate adjacent fp32 pairs of the last kRopeDim features (HF apply_rotary_emb).
 *
 * \param vals   In/out length-kHeadDim fp32 row.
 * \param freqs  Interleaved real/imag, length kRopeDim (complex64 view_as_real).
 */
SGL_DEVICE void rope_tail_fp32(float* vals, const float* freqs) {
  const int head = kHeadDim - kRopeDim;
#pragma unroll
  for (int j = 0; j < kRopeDim / 2; ++j) {
    const float re = vals[head + 2 * j];
    const float im = vals[head + 2 * j + 1];
    const float fr = freqs[2 * j];
    const float fi = freqs[2 * j + 1];
    vals[head + 2 * j] = re * fr - im * fi;
    vals[head + 2 * j + 1] = re * fi + im * fr;
  }
}

SGL_DEVICE void rope_tail_fp32_dim(float* vals, const float* freqs, int dim, int rope_dim) {
  const int head = dim - rope_dim;
  const int n_pairs = rope_dim / 2;
  for (int j = 0; j < n_pairs; ++j) {
    const float re = vals[head + 2 * j];
    const float im = vals[head + 2 * j + 1];
    const float fr = freqs[2 * j];
    const float fi = freqs[2 * j + 1];
    vals[head + 2 * j] = re * fr - im * fi;
    vals[head + 2 * j + 1] = re * fi + im * fr;
  }
}

}  // namespace sglang::sm70_dsv41
