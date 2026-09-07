#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "packed_layout.h"

namespace sm90::decode::sparse_fp8 {

__device__ __forceinline__ bf16x8 decode_int4_word(uint32_t word, float scale) {
  alignas(16) bf16x8 result;
#ifdef SGL_KERNEL_DSV4_INT4_VECTOR_DEQUANT
  // XOR changes signed nibbles into biased integers [0, 15]. BF16 mantissa
  // insertion at exponent 128 is exact, as is subtracting the bias 136.
  // E4M3 scales and signed INT4 codes are exactly representable in BF16;
  // only the final multiply rounds, just as BF16(float(code) * scale) does.
  word ^= 0x88888888u;
  const __nv_bfloat162 bias = __float2bfloat162_rn(136.0f);
  const __nv_bfloat162 scale2 = __float2bfloat162_rn(scale);
  CUTE_UNROLL
  for (int pair = 0; pair < 4; ++pair) {
    __nv_bfloat162_raw raw;
    raw.x = 0x4300u | ((word >> (pair * 8)) & 0xFu);
    raw.y = 0x4300u | ((word >> (pair * 8 + 4)) & 0xFu);
    const __nv_bfloat162 codes = __hsub2(__nv_bfloat162(raw), bias);
    reinterpret_cast<__nv_bfloat162*>(&result)[pair] = __hmul2(codes, scale2);
  }
#else
  CUTE_UNROLL
  for (int j = 0; j < 8; ++j) {
    const int nibble = static_cast<int>((word >> (j * 4)) & 0xFu);
    reinterpret_cast<bf16*>(&result)[j] = bf16(static_cast<float>((nibble ^ 8) - 8) * scale);
  }
#endif
  return result;
}

template <typename Plan>
__device__ __forceinline__ void load_int4_tile(
    Plan& plan,
    int buf_idx,
    const int* indices,
    int valid_count,
    int64_t num_rows,
    const uint8_t* base,
    int row_bytes,
    int idx_in_warpgroup) {
  using namespace kvbit::dsv4;
  constexpr int TILE = 64;
  const int lane = idx_in_warpgroup & 31;
  const int warp = idx_in_warpgroup >> 5;
  CUTE_UNROLL
  for (int round = 0; round < 16; ++round) {
    const int t = warp * 16 + round;
    int index = -1;
    if (lane == 0 && t < valid_count) index = __ldg(indices + t);
    index = __shfl_sync(0xffffffff, index, 0);
    const bool valid = index >= 0 && static_cast<int64_t>(index) < num_rows;
    // The API validates tightly packed pages. Flat addressing eliminates
    // page division/modulo and is valid for SWA, C4, and C128 alike.
    const uint8_t* row = valid ? base + static_cast<int64_t>(index) * row_bytes : nullptr;

    if (lane == 0) plan.is_kv_valid[buf_idx][t] = valid;
    if (lane < ROPE_DIM / 8) {
      alignas(16) bf16x8 rope;
      *reinterpret_cast<uint128_t*>(&rope) = uint128_t();
      if (valid) rope = *reinterpret_cast<const bf16x8*>(row + ROPE_OFFSET + lane * 16);
      bf16* dest = plan.u.k[buf_idx].data() + t * 8 + (NOPE_DIM + lane * 8) * TILE;
      *reinterpret_cast<__int128_t*>(dest) = *reinterpret_cast<__int128_t*>(&rope);
    }

    CUTE_UNROLL
    for (int half = 0; half < 2; ++half) {
      const int dim = half * 256 + lane * 8;
      uint32_t word = 0;
      float scale = 0.0f;
      if (valid && dim < NOPE_DIM) {
        word = __ldg(reinterpret_cast<const uint32_t*>(row + dim / 2));
#ifdef SGL_KERNEL_DSV4_INT4_VECTOR_DEQUANT
        if ((lane & 3) == 0)
#endif
          scale = static_cast<float>(reinterpret_cast<const __nv_fp8_e4m3*>(row + HEADER_OFFSET)[dim / GROUP_SIZE]);
      }
#ifdef SGL_KERNEL_DSV4_INT4_VECTOR_DEQUANT
      // All lanes participate, including the unused tail lanes.
      scale = __shfl_sync(0xffffffff, scale, lane & ~3);
#endif
      alignas(16) const bf16x8 values = decode_int4_word(word, scale);
      CUTE_UNROLL
      for (int phase = 0; phase < 2; ++phase) {
        if (dim < NOPE_DIM && (lane >> 4) == phase) {
          bf16* dest = plan.u.k[buf_idx].data() + t * 8 + dim * TILE;
          *reinterpret_cast<__int128_t*>(dest) = *reinterpret_cast<const __int128_t*>(&values);
        }
      }
    }
  }
}

}  // namespace sm90::decode::sparse_fp8
