#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>

// DSA indexer helpers for the warp-per-row layout: kHeadDim = 128 over 32 lanes,
// each lane owning one 4-element pack, the leading kRopeDim = 64 dims roped.
namespace sglang::device::dsa_indexer {

// NeoX ("halves") cos/sin for this lane's pack; the cache row is [cos(32) | sin(32)].
template <int64_t kRopeDim>
SGL_DEVICE void load_neox_cos_sin(
    const float* __restrict__ cos_sin_cache,
    int32_t lane_id,
    AlignedVector<float, 4>& fcos,
    AlignedVector<float, 4>& fsin) {
  constexpr int64_t kHalfRopeDim = kRopeDim / 2;
  const int32_t idx = lane_id & static_cast<int32_t>(kHalfRopeDim / 4 - 1);
  fcos.load(cos_sin_cache, idx);
  fsin.load(cos_sin_cache + kHalfRopeDim, idx);
}

// NeoX rope in place; the shuffle needs every lane, so call this on all of them.
template <uint32_t kRopeSize>
SGL_DEVICE void apply_neox_rope(
    AlignedVector<float, 4>& data,
    const AlignedVector<float, 4>& fcos,
    const AlignedVector<float, 4>& fsin,
    int32_t lane_id,
    bool is_rope_lane) {
  AlignedVector<float, 4> other;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    other[i] = __shfl_xor_sync(warp::kFullMask, data[i], kRopeSize / 2, kWarpThreads);
  }
  if (!is_rope_lane) return;
  const bool is_first_half = (lane_id & (kRopeSize / 2)) == 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float t = other[i] * fsin[i];
    data[i] = __fmaf_rn(data[i], fcos[i], is_first_half ? -t : t);
  }
}

// 128-point Hadamard (Sylvester order) scaled by 1/sqrt(128), in place.
SGL_DEVICE void hadamard_128(AlignedVector<float, 4>& data, int32_t lane_id) {
  {
    const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a1;
    data[1] = a0 - a1;
    data[2] = a2 + a3;
    data[3] = a2 - a3;
  }
  {
    const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a2;
    data[1] = a1 + a3;
    data[2] = a0 - a2;
    data[3] = a1 - a3;
  }
#pragma unroll
  for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float other = __shfl_xor_sync(warp::kFullMask, data[i], mask, kWarpThreads);
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }
  const float scale = math::rsqrt(128.0f);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    data[i] *= scale;
}

}  // namespace sglang::device::dsa_indexer
