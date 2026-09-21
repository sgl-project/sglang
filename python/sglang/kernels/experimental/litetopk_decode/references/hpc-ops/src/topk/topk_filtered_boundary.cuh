// Copyright (C) 2026 Tencent.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace hpc {
namespace topk {
namespace filtered {

// The 11-bit coarse map is monotone in descending score order.  Sampling still
// uses this exact FP16 projection; CoarseBoundary only removes that projection
// from the subsequent full-row classification pass.
__device__ __forceinline__ uint16_t to_coarse_key(float x) {
  uint16_t bits = __half_as_ushort(__float2half_rn(x));
  uint16_t key = (bits & 0x8000) ? bits : static_cast<uint16_t>((bits ^ 0xFFFF) & 0x7FFF);
  return static_cast<uint16_t>(key >> 5);
}

struct CoarseBoundary {
  float value;
  // tau < 31 contains only positive FP16 NaNs, which are outside the operator
  // contract.  Keep an explicit empty flag so +inf is not admitted there.
  bool empty;
};

// Hot-loop form.  An empty interval is encoded by a NaN sentinel, which makes
// the numeric predicate false for every score in the non-NaN operator domain.
// Keeping value and bits together also preserves the +0/-0 distinction without
// carrying endpoint flags through the row scan.
struct PackedCoarseBoundary {
  float value;
  uint32_t bits;
};

enum class CoarseClass : uint8_t {
  kSelected,
  kCandidate,
  kDropped,
};

// Return the smallest FP32 value whose round-to-nearest FP16 coarse key is at
// most tau.  Midpoints between adjacent FP16 values are exactly representable
// in FP32; when the accepted endpoint has an odd significand, the midpoint
// rounds to the rejected neighbour and the first accepted FP32 value is the
// next representable number toward +inf.
__device__ __forceinline__ CoarseBoundary coarse_boundary(int tau) {
  if (tau < 31) {
    return {CUDART_INF_F, true};
  }
  if (tau >= 2016) {
    return {-CUDART_INF_F, false};
  }

  const uint16_t max_key = static_cast<uint16_t>((tau << 5) | 31);
  const uint16_t accepted_bits =
      max_key < 0x8000 ? static_cast<uint16_t>(0x7FFFu - max_key) : max_key;

  if (accepted_bits == 0) {
    return {0.0f, false};
  }
  // The first finite FP32 value that rounds to +inf under RN-even.
  if (accepted_bits == 0x7C00u) {
    return {65520.0f, false};
  }
  // The rejected neighbour below the most-negative finite FP16 value is -inf.
  // -65520 itself rounds to -inf, so advance one FP32 ULP toward +inf.
  if (accepted_bits == 0xFBFFu) {
    return {__uint_as_float(__float_as_uint(-65520.0f) - 1u), false};
  }

  const uint16_t rejected_bits = accepted_bits & 0x8000u ? static_cast<uint16_t>(accepted_bits + 1)
                                                         : static_cast<uint16_t>(accepted_bits - 1);
  const float accepted = __half2float(__ushort_as_half(accepted_bits));
  const float rejected = __half2float(__ushort_as_half(rejected_bits));
  float boundary = (accepted + rejected) * 0.5f;

  if (accepted_bits & 1u) {
    uint32_t bits = __float_as_uint(boundary);
    bits += (bits & 0x80000000u) ? static_cast<uint32_t>(-1) : 1u;
    boundary = __uint_as_float(bits);
  }
  return {boundary, false};
}

__device__ __forceinline__ PackedCoarseBoundary
pack_coarse_boundary(const CoarseBoundary& boundary) {
  const float value = boundary.empty ? CUDART_NAN_F : boundary.value;
  return {value, __float_as_uint(value)};
}

__device__ __forceinline__ bool coarse_accepts(float x, const PackedCoarseBoundary& boundary) {
  return x > boundary.value || __float_as_uint(x) == boundary.bits;
}

// Classify against an exact coarse threshold without projecting x to FP16.
// Testing the inclusive candidate boundary first keeps the common dropped path
// to one predicate; only the coarse upper tail reaches the strict boundary.
__device__ __forceinline__ CoarseClass
coarse_classify(float x, const PackedCoarseBoundary& select_boundary,
                const PackedCoarseBoundary& candidate_boundary) {
  if (!coarse_accepts(x, candidate_boundary)) {
    return CoarseClass::kDropped;
  }
  return coarse_accepts(x, select_boundary) ? CoarseClass::kSelected : CoarseClass::kCandidate;
}

// Numeric >= with IEEE signed-zero order preserved.  For nonzero boundaries,
// numeric equality already implies identical bits; for the +0 boundary this
// admits +0 and rejects -0, exactly matching the FP16 coarse projection.
__device__ __forceinline__ bool fp32_boundary_accepts(float x, float boundary,
                                                      uint32_t boundary_bits) {
  return x > boundary || __float_as_uint(x) == boundary_bits;
}

}  // namespace filtered
}  // namespace topk
}  // namespace hpc
