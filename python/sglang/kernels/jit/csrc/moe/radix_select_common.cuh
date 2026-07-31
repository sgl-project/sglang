#pragma once

// Primitives shared by the two radix top-k selectors in this directory
// (route_radix.cuh and moe_front.cuh). Both carried byte-identical copies -- one
// set prefixed `fgt_` -- so this header is now the single definition.
//
// The bodies below are the originals, moved not rewritten. That matters: two of
// them carry correctness constraints that are easy to lose in a retype --
// biased_to_key must canonicalise -0.0 so equal scores get equal keys, and
// sigmoid_match must stay instruction-identical (__fdividef) so both kernels rank
// and weight identically. A reimplementation of this header changed every
// selected expert id while leaving the weights untouched.
//
// RadixSelectBase is a tag struct: static members and nested types only, so
// deriving from it adds no data member, and MatchBin keeps its fields and
// alignas(16), leaving the Smem blocks that embed it byte-identical.

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

namespace moe::radix {

struct RadixSelectBase {
  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadixSize = 1 << kRadixBits;
  static constexpr uint32_t kRadixRounds = 32 / kRadixBits;

  struct alignas(16) MatchBin {
    uint32_t bin;
    uint32_t above_count;  // active elements in bins strictly above `bin`
    uint32_t equal_count;  // active elements in bin `bin`
  };
};

inline constexpr float kNanFloor = -1e30f;

// Monotone unsigned key: larger biased -> larger key. Caller must have floored
// biased-NaN. Canonicalizes -0.0 -> +0.0 so equal values get equal keys.
SGL_DEVICE uint32_t biased_to_key(float biased) {
  if (biased == 0.0f) biased = 0.0f;
  uint32_t u = __float_as_uint(biased);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// tl.sigmoid(x) = 1/(1+exp(-x)). Must stay instruction-identical to v1's
// sigmoid_match so both kernels rank (and weight) identically.
SGL_DEVICE float sigmoid_match(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}

SGL_DEVICE float nan_floor(float x) {
  return (x == x) ? x : kNanFloor;
}

SGL_DEVICE void bar_sync(uint32_t id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
}

SGL_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

// Exclusive prefix (block-wide, thread-rank order) of `cnt`. Uses
// smem_warp_sum[kNumWarps]; syncs on entry (so the workspace can be reused
// across calls) and before the cross-warp read.
SGL_DEVICE uint32_t block_exclusive_sum(uint32_t cnt, uint32_t lane_id, uint32_t warp_id, uint32_t* smem_warp_sum) {
  const uint32_t inc = warp_inclusive_sum(lane_id, cnt);
  if (lane_id == 31) smem_warp_sum[warp_id] = inc;
  __syncthreads();
  // TODO: replace `__reduce_add_sync` with `warp::reduce_sum`
  const auto base = __reduce_add_sync(0xFFFFFFFF, lane_id < warp_id ? smem_warp_sum[lane_id] : 0u);
  return base + inc - cnt;
}

}  // namespace moe::radix

}  // namespace sglang
