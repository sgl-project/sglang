/// \file dsv41/engram.cuh
/// \brief Shared pieces of the DeepSeek-V4.1 Engram kernels.
///
/// Engram updates the hyper-connection residual per token: four gates from the
/// residual streams and the four keys of the wkv row, then
/// ``R[h] += gate[h] * value``. The kernels live in
/// csrc/deepseek_v4/engram_fusion.cuh; what they agree on lives here.
#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/dsv41/mhc.cuh>

#include <cstdint>

namespace sglang {

namespace device::engram {

inline constexpr uint32_t kMaxWorld = 8;

// The table: a row is one MXFP8 head vector, 256 e4m3 bytes + 8 e8m0 bytes.
inline constexpr uint32_t kHeadDim = 256;
inline constexpr uint32_t kScalesPerRow = kHeadDim / 32;
inline constexpr uint32_t kCols = 24;              // hash columns per token (3 n-grams x 8 heads)
inline constexpr uint32_t K = kCols * kHeadDim;    // 6144, wkv's K
inline constexpr uint32_t kSfKTiles = K / 32 / 4;  // 48 swizzle tiles of 4 scale columns

inline constexpr uint32_t D = mhc::kHidden;
inline constexpr uint32_t kHC = mhc::kNumStreams;
inline constexpr uint32_t KV = (kHC + 1) * D;  // 25600, 4 keys then the value

/// One 16B slice of a residual / kv row: thread t owns elements [8t, 8t + 8).
using vec_t = device::AlignedVector<bf16x2_t, 4>;

/// One 16B slice of a table row, the unit the gather moves; measured even with
/// 32B lanes where bandwidth-bound and slightly ahead in the latency corners.
using table_vec_t = device::AlignedVector<uint64_t, 2>;

/// \brief An e8m0 scale as fp32: 2^(e - 127), exactly (0 is the subnormal
/// 2^-127, the padding sentinel; 255 degrades to inf -- tables carry no NaN).
SGL_DEVICE float e8m0_to_float(uint8_t e) {
  return __uint_as_float(e ? (static_cast<uint32_t>(e) << 23) : 0x00400000u);
}

/// \brief The gate of one stream from its three reduced sums: a QK-RMSNorm
/// attention logit, sqrt-compressed and squashed. MUFU approximations -- the
/// training sigmoid is itself ex2.approx, and the clamp keeps rsqrt inputs positive.
SGL_DEVICE float gate_from_sums(float x_sq, float key_sq, float dot, float eps, float clamp) {
  const float inv_d = 1.0f / static_cast<float>(D);
  const float rstd = rsqrtf(x_sq * inv_d + eps) * rsqrtf(key_sq * inv_d + eps);
  const float logit = dot * rstd * rsqrtf(static_cast<float>(D));
  const float a = fmaxf(fabsf(logit), clamp);
  const float z = copysignf(a * rsqrtf(a), logit);  // sqrt(a), one MUFU + FMUL
  return math::sigmoid_fast</*kFastRcp=*/true>(z);
}

}  // namespace device::engram

}  // namespace sglang
