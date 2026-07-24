/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <sgl_kernel/ptx/mma_desc.cuh>
#include <sgl_kernel/ptx/tcgen05.cuh>

#include <cstdint>

// ================= common/ptx/tcgen05_mma_dense.cuh =================
// Parametric wrapper unifying `tcgen05.mma kind::f16` (BF16/F16 inputs) and
// `tcgen05.mma kind::f8f6f4` (FP8/FP6/FP4 inputs, dense — no scale_block) under
// one templated entry point. Both kinds share IDENTICAL operand shape and the
// same {alloc, commit, dealloc, relinquish} pipeline; only the `kind::` in the
// PTX op and the instruction-descriptor's atype/btype encoding differ.
//
// Composition pieces used unchanged from existing common headers:
//   * `mma_inst_desc_f16` / `mma_inst_desc_f8f6f4`   (mma_desc.cuh)
//   * `tcgen05_mma_f16{,_2sm}` / `tcgen05_mma_f8f6f4{,_2sm}` (ptx/tcgen05.cuh)
//   * `tcgen05_alloc{,_2sm}`, `tcgen05_commit_arrive{,_2sm,...}`, etc.
//
// What this header adds:
//   1. `MmaDenseKind { F16, F8F6F4 }`    — pick at compile time.
//   2. `DenseAType<KIND>`                — picks the right enum (F16Type vs FP8Type).
//   3. `mma_inst_desc_dense<KIND>(...)`  — single inst-desc builder.
//   4. `tcgen05_mma_dense<KIND, CTA_GROUP>(...)` — single MMA wrapper.
//
// Block-scale variants (kind::mxf8f6f4, kind::mxf4) live in their own modules
// (`ptx/c_tcgen05_mma_mxf8f6f4/`, `ptx/c_tcgen05_mma_mxf4/`) because they add
// SF operands + UTCCP staging that don't compose with the dense pipeline here.
// See `ptx/c_tcgen05_mma_dense/README.md` for the picking decision tree.


namespace ptx {

enum class MmaDenseKind : uint8_t {
    F16    = 0,   // F16 / BF16 × F16 / BF16 → F16 / F32   (PTX kind::f16)
    F8F6F4 = 1,   // FP8 / FP6 / FP4 mix    → F32          (PTX kind::f8f6f4)
    TF32   = 2,   // TF32 × TF32           → F32          (PTX kind::tf32)
};

// Per-kind atype/btype enum. `F16Type` (F16=0, BF16=1) for kind::f16;
// `FP8Type` (E4M3=0, E5M2=1, E2M3=3, E3M2=4, E2M1=5) for kind::f8f6f4. The
// two enums share bit positions in the instruction descriptor but encode
// different values — this trait keeps the kernel-side type generic. kind::tf32
// has no per-operand type choice (atype/btype are fixed at TF32=2), so its
// trait maps to a placeholder enum the builder ignores.
template <MmaDenseKind KIND> struct DenseAType;
template <> struct DenseAType<MmaDenseKind::F16>    { using type = F16Type; };
template <> struct DenseAType<MmaDenseKind::F8F6F4> { using type = FP8Type; };
template <> struct DenseAType<MmaDenseKind::TF32>   { using type = F16Type; };

template <MmaDenseKind KIND>
using DenseAType_t = typename DenseAType<KIND>::type;

// Single inst-desc builder. Dispatches to the kind-specific function in
// `mma_desc.cuh`; the layouts are bit-identical for the M / N / dtype /
// major fields (see Table 44 of the PTX ISA).
template <MmaDenseKind KIND>
__host__ __device__ static __forceinline__ constexpr uint32_t mma_inst_desc_dense(
        uint32_t M, uint32_t N,
        DenseAType_t<KIND> a_type, DenseAType_t<KIND> b_type,
        DType d_type   = DType::F32,
        Major a_major  = Major::K,
        Major b_major  = Major::K,
        bool  negate_a = false,
        bool  negate_b = false) {
    if constexpr (KIND == MmaDenseKind::F16) {
        return mma_inst_desc_f16(M, N, a_type, b_type, d_type, a_major, b_major,
                                 negate_a, negate_b);
    } else if constexpr (KIND == MmaDenseKind::TF32) {
        (void)a_type; (void)b_type;   // kind::tf32 atype/btype fixed at TF32=2
        return mma_inst_desc_tf32(M, N, d_type, a_major, b_major,
                                  negate_a, negate_b);
    } else {
        return mma_inst_desc_f8f6f4(M, N, a_type, b_type, d_type, a_major, b_major,
                                    negate_a, negate_b);
    }
}

// Single MMA-issue wrapper. `CTA_GROUP` is 1 (single-CTA) or 2 (cluster).
// `d` is the local CTA's TMEM base address; for CTA_GROUP=2 the M-dim is
// distributed across the two peer CTAs (HW writes half the M rows into each
// peer's TMEM). Pair with `tcgen05_commit_arrive` (CTA_GROUP=1) or
// `tcgen05_commit_arrive_2sm{,_multicast}` (CTA_GROUP=2).
//
// Valid shapes (dense, per Table 41 of the PTX ISA — off-table shapes hit
// `cudaErrorIllegalInstruction` at the commit, NOT at the MMA op itself):
//   kind::f16:
//     cta_group::1 — M ∈ {64, 128}, N ∈ {8, 16, …, 256} (steps of 8), K = 16.
//     cta_group::2 — M ∈ {128, 256}, N ∈ {16, 32, …, 256} (steps of 16),  K = 16.
//   kind::f8f6f4:
//     cta_group::1 — M ∈ {64, 128}, N ∈ {8, 16, …, 256} (steps of 8), K = 32.
//     cta_group::2 — M ∈ {128, 256}, N ∈ {16, 32, …, 256} (steps of 16),  K = 32.
template <MmaDenseKind KIND, int CTA_GROUP>
static __device__ __forceinline__ void tcgen05_mma_dense(
        uint32_t d, uint64_t desc_a, uint64_t desc_b,
        uint32_t inst_desc_high, uint32_t scale_c) {
    static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
    if constexpr (KIND == MmaDenseKind::F16) {
        if constexpr (CTA_GROUP == 1)
            tcgen05_mma_f16    (d, desc_a, desc_b, inst_desc_high, scale_c);
        else
            tcgen05_mma_f16_2sm(d, desc_a, desc_b, inst_desc_high, scale_c);
    } else if constexpr (KIND == MmaDenseKind::TF32) {
        if constexpr (CTA_GROUP == 1)
            tcgen05_mma_tf32    (d, desc_a, desc_b, inst_desc_high, scale_c);
        else
            tcgen05_mma_tf32_2sm(d, desc_a, desc_b, inst_desc_high, scale_c);
    } else {
        if constexpr (CTA_GROUP == 1)
            tcgen05_mma_f8f6f4    (d, desc_a, desc_b, inst_desc_high, scale_c);
        else
            tcgen05_mma_f8f6f4_2sm(d, desc_a, desc_b, inst_desc_high, scale_c);
    }
}

}  // namespace ptx
