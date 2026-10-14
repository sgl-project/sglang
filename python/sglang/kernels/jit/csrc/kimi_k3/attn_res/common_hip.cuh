// Row layout and wave64 helpers shared by the Kimi-K3 attention-residual HIP kernels.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

#ifndef USE_ROCM
#error "kimi_k3/attn_res/common_hip.cuh is a wave64 (ROCm) header"
#endif

namespace sglang {
namespace k3_attn_res_read {

inline constexpr int kHidden = 7168;
inline constexpr int kVec = 8;  // bf16 per thread: one 16-byte vector of the row
inline constexpr int kThreads = kHidden / kVec;
inline constexpr int kWave = 64;
inline constexpr int kWaves = kThreads / kWave;
inline constexpr int kMaxNvb = 8;

SGL_DEVICE float bf16_lo(uint32_t w) {
  return __uint_as_float(w << 16);
}

SGL_DEVICE float bf16_hi(uint32_t w) {
  return __uint_as_float(w & 0xffff0000u);
}

// Round to nearest even, as torch and Triton convert fp32 to bf16.
SGL_DEVICE uint32_t bf16_bits(float f) {
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

SGL_DEVICE uint32_t pack_bf16(float lo, float hi) {
  return bf16_bits(lo) | (bf16_bits(hi) << 16);
}

SGL_DEVICE void unpack8(const uint4& v, float (&f)[kVec]) {
  f[0] = bf16_lo(v.x), f[1] = bf16_hi(v.x), f[2] = bf16_lo(v.y), f[3] = bf16_hi(v.y);
  f[4] = bf16_lo(v.z), f[5] = bf16_hi(v.z), f[6] = bf16_lo(v.w), f[7] = bf16_hi(v.w);
}

SGL_DEVICE uint4 pack8(const float (&f)[kVec]) {
  return make_uint4(pack_bf16(f[0], f[1]), pack_bf16(f[2], f[3]), pack_bf16(f[4], f[5]), pack_bf16(f[6], f[7]));
}

template <int kCtrl>
SGL_DEVICE float dpp(float v) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), kCtrl, 0xf, 0xf, false));
}

// Sum over the wave, uniform in every lane: DPP butterflies inside each 16-lane row
// (quad_perm [1,0,3,2], [2,3,0,1], row_half_mirror, row_mirror), then the four row sums
// via readlane. __shfl_xor would run every step through the LDS crossbar.
SGL_DEVICE float wave_sum(float v) {
  v += dpp<0xb1>(v);
  v += dpp<0x4e>(v);
  v += dpp<0x141>(v);
  v += dpp<0x140>(v);
  const int bits = __float_as_int(v);
  return (__int_as_float(__builtin_amdgcn_readlane(bits, 0)) + __int_as_float(__builtin_amdgcn_readlane(bits, 16))) +
         (__int_as_float(__builtin_amdgcn_readlane(bits, 32)) + __int_as_float(__builtin_amdgcn_readlane(bits, 48)));
}

SGL_DEVICE void load_fp32x8(const float4* src, float (&f)[kVec]) {
  const float4 a = src[0], b = src[1];
  f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w;
  f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

}  // namespace k3_attn_res_read
}  // namespace sglang
