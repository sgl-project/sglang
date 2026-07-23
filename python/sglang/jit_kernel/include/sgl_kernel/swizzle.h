/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ================= common/swizzle.h =================

// CPU-side swizzle math for TMA tile-mode swizzle modes. The TMA hardware
// permutes the (row × col) atom layout when storing the loaded tile into smem,
// so reading a known cell back from smem requires applying the same permutation.
//
// PICKING A SWIZZLE FOR YOUR TENSOR-MAP (the ptx::SmemSwizzle / CU_TENSOR_MAP_SWIZZLE_*
// you pass to cuTensorMapEncodeTiled):
//   K-major MMA feed   → swizzle == K_bytes (mandatory; DeepGEMM mma/sm90.cuh:251).
//                          K=32 FP8 / K=64 FP4 packed → 32B
//                          K=64 FP8 / K=32 BF16 → 64B
//                          K=128 FP8 / K=64 BF16 / K=32 FP32 → 128B
//                          K_bytes > 128 → cap at 128B
//                          K_bytes < 32  → unsupported in K-major MMA path
//   MN-major MMA feed  → min(BLOCK_MN_bytes, 128).
//   Not consumed by MMA → SWIZZLE_NONE (the hardware swizzle's only purpose is
//                                        bank-conflict avoidance for tensor cores).
// Full motivation + empirical verification: ptx/a_tma_2d/README.md.
//
// FORMULA STRUCTURE (atom = 16 bytes; same atomicity across 32B / 64B / 128B):
//   Underlying rule: byte bits 7..(7+log2(swz/16)-1) of the offset within an
//   8-row block determine the XOR shift in atom-units. That's why all three
//   modes have an effective 8-row period but different visual periods (128B
//   shifts every row; 64B every 2 rows; 32B every 4 rows).
//
// References:
//   PTX ISA 9.2 §5.5.7 (Swizzling Modes), Figures 23–37
//   CUDA Driver API: CU_TENSOR_MAP_SWIZZLE_*
//
// All formulas verified empirically on B300 (sm_103a) by a load +
// read-with-formula roundtrip in ptx/a_tma_2d.

namespace swz {


// 128B swizzle. Row stride = 128 bytes (= 8 atoms = 64 BF16 cols).
// smem_atom = logical_atom XOR (r & 7). 8-row period.
//
// Returns the smem column index in BF16 units within the row.
__host__ __device__ inline uint32_t smem_col_128b_bf16(uint32_t r, uint32_t c) {
    return c ^ ((r & 7u) << 3);   // (r & 7) atoms shifted; atom = 8 BF16 cols
}


}  // namespace swz
