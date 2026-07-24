/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <cuda_runtime.h>
#include <cstdint>

// ================= common/ptx/smem.cuh =================
// Shared-memory wrappers — only those without a clean C++ equivalent.
// PTX ISA 9.2 §9.7.14.5.16 (stmatrix).
//
// Note on plain vector smem ops:
//   For 16-byte vector loads/stores you do NOT need inline PTX. Use
//   `int4` / `float4` (or any 16-byte aligned `__shared__` aggregate) and
//   nvcc emits `st.shared.v4.b32` / `ld.shared.v4.b32` automatically:
//
//       __shared__ int4 buf[N];
//       buf[i] = make_int4(a, b, c, d);          // → st.shared.v4.b32
//       int4 v = buf[i];                          // → ld.shared.v4.b32
//
//   Inline-PTX wrappers for these would only matter if you needed a
//   specific cache hint (.cs / .cv) or to defeat compiler reordering, which
//   we don't here.


namespace ptx {


// ldmatrix.sync.aligned.m8n8.x4.b16 — warp-collective load of
// 4 × (8×8) BF16 matrices from smem into MMA-fragment registers.
//
// Per-thread inputs (PTX ISA §9.7.14.5.15):
//   row_addr — smem byte address of this thread's "row". Lanes 0–7 supply
//              row-bases for matrix 0, 8–15 matrix 1, 16–23 matrix 2, 24–31
//              matrix 3. Each address points to 8 contiguous BF16 (= 16 B);
//              must be 16-byte aligned.
//
// Per-thread outputs:
//   r0..r3   — 4 packed-BF16 registers (one per matrix). Lane L's r_m holds
//              the BF16 pair at (matrix m, row L/4, cols (L%4)*2 and
//              (L%4)*2+1). The 32 lanes' .b32 outputs together cover all
//              64 cells of each 8×8 matrix.
//
// Mandatory `.sync.aligned`: every lane in the warp must execute the same
// instruction with matching qualifiers.
static SGL_DEVICE void ldmatrix_x4_b16(
        uint32_t row_addr,
        uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(row_addr));
}

// ldmatrix.sync.aligned.m8n8.x2.b16 — warp-collective load of two 8x8
// BF16 matrices from shared memory.
static SGL_DEVICE void ldmatrix_x2_b16(
        uint32_t row_addr, uint32_t& r0, uint32_t& r1) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0, %1}, [%2];"
        : "=r"(r0), "=r"(r1)
        : "r"(row_addr));
}


}  // namespace ptx
