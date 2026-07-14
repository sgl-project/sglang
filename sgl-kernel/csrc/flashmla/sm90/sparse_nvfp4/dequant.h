/*
 * Copyright (c) 2026 SGLang Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_bf16.h>

#include <cstdint>

#include "layout.h"

// Reuse the vector type expected by the pinned sparse FlashMLA producer.
#include "sm90/decode/sparse_fp8/components/dequant.h"

namespace sm90::nvfp4 {

__device__ __forceinline__ uint64_t load_packed_e2m1x16(const void* address) {
  uint64_t value;
  asm volatile("ld.global.nc.L1::evict_last.L2::128B.u64 %0, [%1];" : "=l"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ uint8_t load_scale_e4m3_bits(const void* address) {
  uint32_t value;
  asm volatile("ld.global.nc.L1::evict_last.L2::128B.u8 %0, [%1];" : "=r"(value) : "l"(address));
  return static_cast<uint8_t>(value);
}

__device__ __forceinline__ float e4m3_bits_to_float(uint8_t bits) {
  __nv_fp8_e4m3 value;
  value.__x = bits;
  return static_cast<float>(value);
}

struct E2M1Bf16Lut {
  // The low and high bytes of the eight positive E2M1 magnitudes, split into
  // two four-byte PRMT sources.  Keeping the table in registers is much
  // cheaper than synthesizing 16 FP32 values and multiplying them one by one.
  uint32_t low_0_3;
  uint32_t low_4_7;
  uint32_t high_0_3;
  uint32_t high_4_7;
};

struct bf16x16 {
  bf16x8 lo;
  bf16x8 hi;
};

__device__ __forceinline__ uint32_t cvt_bf16x2_bits(float lo, float hi) {
  const __nv_bfloat162 values = __float22bfloat162_rn(make_float2(lo, hi));
  return *reinterpret_cast<const uint32_t*>(&values);
}

__device__ __forceinline__ E2M1Bf16Lut make_e2m1_bf16_lut(float scale) {
  // E2M1's positive magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}.  Computing
  // each scaled BF16 value once reduces sixteen FP32 multiplies/conversions
  // to four packed conversions.  Multiplication is deliberately performed
  // in FP32 before the BF16 rounding, matching the scalar implementation.
  const uint32_t pair_0_1 = cvt_bf16x2_bits(0.0f * scale, 0.5f * scale);
  const uint32_t pair_2_3 = cvt_bf16x2_bits(1.0f * scale, 1.5f * scale);
  const uint32_t pair_4_5 = cvt_bf16x2_bits(2.0f * scale, 3.0f * scale);
  const uint32_t pair_6_7 = cvt_bf16x2_bits(4.0f * scale, 6.0f * scale);

  return {
      __byte_perm(pair_0_1, pair_2_3, 0x6420),
      __byte_perm(pair_4_5, pair_6_7, 0x6420),
      __byte_perm(pair_0_1, pair_2_3, 0x7531),
      __byte_perm(pair_4_5, pair_6_7, 0x7531),
  };
}

__device__ __forceinline__ uint32_t e2m1_signs_0_1(uint32_t packed) {
  // Move nibble sign bits 3 and 7 to the sign bits of two packed BF16s.
  return ((packed & 0x00000008u) << 12) | ((packed & 0x00000080u) << 24);
}

__device__ __forceinline__ uint32_t e2m1_signs_2_3(uint32_t packed) {
  // Move nibble sign bits 11 and 15 to the sign bits of two packed BF16s.
  return ((packed & 0x00000800u) << 4) | ((packed & 0x00008000u) << 16);
}

__device__ __forceinline__ bf16x8 dequant_e2m1x8(uint32_t packed, const E2M1Bf16Lut& lut) {
  bf16x8 result;
  uint32_t* result_pairs = reinterpret_cast<uint32_t*>(&result);

  // A PRMT selector consists of four nibbles.  The E2M1 magnitude is already
  // a three-bit LUT index, so masking away the sign makes the packed source
  // directly usable as a selector for four values at once.
  const uint32_t selector_lo = packed & 0x00007777u;
  const uint32_t low_bytes_lo = __byte_perm(lut.low_0_3, lut.low_4_7, selector_lo);
  const uint32_t high_bytes_lo = __byte_perm(lut.high_0_3, lut.high_4_7, selector_lo);
  result_pairs[0] = __byte_perm(low_bytes_lo, high_bytes_lo, 0x5140) ^ e2m1_signs_0_1(packed);
  result_pairs[1] = __byte_perm(low_bytes_lo, high_bytes_lo, 0x7362) ^ e2m1_signs_2_3(packed);

  packed >>= 16;
  const uint32_t selector_hi = packed & 0x00007777u;
  const uint32_t low_bytes_hi = __byte_perm(lut.low_0_3, lut.low_4_7, selector_hi);
  const uint32_t high_bytes_hi = __byte_perm(lut.high_0_3, lut.high_4_7, selector_hi);
  result_pairs[2] = __byte_perm(low_bytes_hi, high_bytes_hi, 0x5140) ^ e2m1_signs_0_1(packed);
  result_pairs[3] = __byte_perm(low_bytes_hi, high_bytes_hi, 0x7362) ^ e2m1_signs_2_3(packed);
  return result;
}

__device__ __forceinline__ bf16x8 dequant_e2m1x8(uint32_t packed, float scale) {
  const E2M1Bf16Lut lut = make_e2m1_bf16_lut(scale);
  return dequant_e2m1x8(packed, lut);
}

__device__ __forceinline__ bf16x16 dequant_e2m1x16(uint64_t packed, float scale) {
  const E2M1Bf16Lut lut = make_e2m1_bf16_lut(scale);
  return {
      dequant_e2m1x8(static_cast<uint32_t>(packed), lut),
      dequant_e2m1x8(static_cast<uint32_t>(packed >> 32), lut),
  };
}

}  // namespace sm90::nvfp4
