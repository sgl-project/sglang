// Warp-level matrix ops: ldmatrix and mma.sync (PTX ISA 9.7.14).

#pragma once

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/ptx/barrier.cuh>
#include <sgl_kernel/ptx/utils.cuh>

#include <cstdint>

namespace sglang::device::ptx {

/// Four 8x8 b16 matrices from shared memory; lane i supplies the row address of row i % 8 of matrix i / 8, and
/// receives in 32-bit word j of `value` its two elements of matrix j (row lane / 4, columns 2 (lane % 4), +1).
/// `V` is any 16 B type, e.g. `uint32_t[4]` or `AlignedVector<bf16x2_t, 4>`.
template <SmemAddress Addr, typename V>
SGL_DEVICE void ldmatrix_x4(Addr addr, V& value) {
  static_assert(sizeof(V) == 16);
  uint32_t r[4];
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(to_shared(addr)));
  __builtin_memcpy(&value, r, sizeof(r));
}

/// d += a * b for one m16n8k16 tile, bf16 inputs (a row-major, b column-major), fp32 accumulators. `A` / `B` are any
/// 16 B / 8 B types holding the fragments (four / two 32-bit words of bf16 pairs).
template <typename A, typename B>
SGL_DEVICE void mma_m16n8k16_bf16(float (&d)[4], const A& a, const B& b) {
  static_assert(sizeof(A) == 16 && sizeof(B) == 8);
  uint32_t ra[4], rb[2];
  __builtin_memcpy(ra, &a, sizeof(ra));
  __builtin_memcpy(rb, &b, sizeof(rb));
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]));
}

}  // namespace sglang::device::ptx
