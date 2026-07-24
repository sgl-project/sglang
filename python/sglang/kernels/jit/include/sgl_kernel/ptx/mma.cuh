/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ================= common/ptx/mma.cuh =================
// Warp-level mma.sync wrappers. PTX ISA 9.2 §9.7.14
// (refs/sections/9_7_14_warp_level_matrix_multiply_accumulate.txt).
// Distinct from tcgen05 MMA (`ptx/tcgen05_mma_dense.cuh`): mma.sync is the
// warp-register form — co-resides freely (no TMEM, no 1-CTA/SM cap), the
// pick for small/latency-bound and co-resident GEMMs
// (recipes/gemm_design_guide §2.2/§2.3).


namespace ptx {


// mma.sync.aligned.m16n8k16.row.col bf16*bf16→f32, D += A·B. Same fragment
// shape as the f16 form above (recipes/mma_sync_warp_gemm §Fragments).
static __device__ __forceinline__ void mma_m16n8k16_bf16f32(
        float4& d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}


}  // namespace ptx
