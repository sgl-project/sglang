/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <sgl_kernel/ptx/addr.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_runtime.h>
#include <cstdint>

// ================= common/ptx/tcgen05.cuh =================
// tcgen05 (Blackwell 5th-gen TensorCore) wrappers. PTX ISA 9.2 §9.7.16.
//
// LIFECYCLE (mandatory order, PTX ISA §9.7.16.7.1):
//   1. tcgen05_alloc(smem_for_taddr, n_cols)   — one warp issues; n_cols
//      power of 2 in [32, 512]. Writes the TMEM byte address to smem.
//   2. __syncthreads(); read taddr from smem.
//   3. tcgen05_st / tcgen05_ld for direct register ↔ TMEM movement;
//      tcgen05_cp_* for smem → TMEM (UTCCP); tcgen05_mma_* for compute.
//   4. tcgen05_dealloc(taddr, n_cols)
//   5. tcgen05_relinquish() before kernel exit (mandatory if any alloc happened).
//
// LANE-BAND ACCESS RESTRICTIONS (PTX ISA §9.7.16.8.1):
//   Each warp can access only its own 32-lane band of TMEM:
//     warpgroup_warp_id 0 → TMEM lanes 0..31
//                       1 → TMEM lanes 32..63
//                       2 → TMEM lanes 64..95
//                       3 → TMEM lanes 96..127
//   For shape `.16x256b.x1` (16 lanes per call), use `taddr | 0x00100000` to
//   address the upper half (lane 16) within the warp's band.
//
// MMA RESULT LAYOUT (PTX ISA §9.7.16.10.5, Layouts A–G):
//   The data layout of D in TMEM depends on (M, cta_group, sparsity, .ws):
//     M=64, cta_group::1, no .ws       → Layout F  (4×1, 1/2 datapath utilized)
//     M=64, cta_group::1, .ws          → Layout E  (2×2)
//     M=128, cta_group::1, .ws=any     → Layout D  (4×1, full)
//     M=128, cta_group::2, dense       → Layout B  (2×2)
//     M=256, cta_group::2              → Layout A  (4×1, full)
//   Layout F (M=64) leaves half the lanes empty — naive `.32x32b.x1` drains
//   read zeros for half the rows. Use M=128 (Layout D) for the simple drain
//   path. ptx/c_tcgen05_mma_dense uses this.
//
// SYNCHRONIZATION (PTX ISA §9.7.16.6.4):
//   ld / st  → use `tcgen05_wait_ld()` / `tcgen05_wait_st()` before consuming.
//   mma      → use `tcgen05_commit_arrive(bar)` + `mbar_wait_*` +
//              `tcgen05_fence_after_thread_sync()` before reading the result
//              with `tcgen05_ld_*`. The fence::after_thread_sync is mandatory
//              — without it, register reads after `tcgen05.ld` may see stale
//              values even though the mbarrier signaled "MMA done".
//   cp (UTCCP) → use `tcgen05_wait_st()` (the SF write to TMEM is treated as
//                a store from the caller's perspective).
//
// MMA DESCRIPTORS:
//   See common/mma_desc.cuh for mma_smem_desc / mma_inst_desc_*. Bit layout
//   is documented there (the PTX spec text has errors at bits 46-60 of the
//   smem descriptor — see mma_desc.cuh comments).


namespace ptx {

// ---- TMEM allocation lifecycle ----------------------------------------------

static SGL_DEVICE void tcgen05_alloc(uint32_t smem_addr_for_taddr, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr_for_taddr), "r"(n_cols));
}

static SGL_DEVICE void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols));
}

static SGL_DEVICE void tcgen05_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

// ---- TMEM lifecycle for cta_group::2 (cluster MMA) -------------------------
//
// PTX ISA §9.7.16.7.1: `cta_group::2` requires that ONE warp from EACH peer
// CTA collectively performs the alloc and dealloc (i.e. both CTAs must call
// these wrappers from a designated warp). The resulting TMEM addresses are
// symmetric — each CTA's TMEM is allocated at the same column offset.

static SGL_DEVICE void tcgen05_alloc_2sm(uint32_t smem_addr_for_taddr, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr_for_taddr), "r"(n_cols));
}


// ---- TMEM ↔ register load/store ---------------------------------------------

// WARP-IMPLICIT LANE BAND (the ld/st wrappers below take only `taddr`, but
// which 32 of TMEM's 128 lanes the warp accesses is determined by the
// issuing warp's index):
//
//   For `.32x32b` shape, warp `w` accesses TMEM lanes `[(w%4)*32, (w%4+1)*32)`.
//   The `% 4` is the easy-to-miss part: 4-warp drains (warps 0-3) map cleanly
//   to lane bands 0-3, but if the epilogue uses warps 2-5 (because warps 0
//   and 1 are running TMA / MMA), warps 4 and 5 wrap to bands 0 and 1. Match
//   the smem write rows to the actually-drained band (`(warp_id % 4) * 32`),
//   not to a naive `(warp_id - epilogue_start) * 32` — see fused_gemm/v1
//   rule 6a for the post-mortem.
//
//   The TMEM address can also encode the lane in bits [22:16] (multiples of
//   32 for `.32x32b`) for explicitness. Encoded lane and warp-implicit band
//   must agree.
//
// CHOOSING `.shape.num`:
//   The number of registers per lane is `regs_per_lane(shape) * num_factor(.x?)`
//   per PTX ISA Tables 51/52:
//     base register count: .32x32b → 1, .16x64b → 1, .16x128b → 2, .16x256b → 4
//     `.x?` multiplier:    .x1 → 1, .x2 → 2, .x4 → 4, .x8 → 8, …
//   So .32x32b.x2 = 2 regs/lane = 64 cells/warp (= 2 columns of TMEM).
//      .16x256b.x2 = 8 regs/lane (= 16 lanes × 8 columns × 32 bits = 4096 bits per call).
//
//   Trade-off: higher `.x?` drains more TMEM per instruction (fewer issues for
//   the same data) but uses more per-thread registers. Pick the smallest `.x?`
//   that gives you the needed per-call data without spilling. `.16x256b.x1`
//   (4 regs/lane) is a common choice for BF16 epilogues that drain 4 cols at
//   a time; `.x2` and `.x4` cover wider drains.


// .32x32b.x8: 8 b32 registers per lane = 256 cells/warp = 8 TMEM columns.
// Per-lane 8 FP32 → 4 bf16x2 packs = 16 BF16 bytes = one int4. Natural fit for
// BF16 epilogues that drain a TMEM column band with 16-byte smem stores.
static SGL_DEVICE void tcgen05_ld_32x32b_x8(
        uint32_t taddr,
        uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
        uint32_t& r4, uint32_t& r5, uint32_t& r6, uint32_t& r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                 " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
                   "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
                 : "r"(taddr));
}

static SGL_DEVICE void tcgen05_ld_32x32b_x8(
        uint32_t taddr, uint32_t* dst) {
    tcgen05_ld_32x32b_x8(
        taddr, dst[0], dst[1], dst[2], dst[3],
        dst[4], dst[5], dst[6], dst[7]);
}

static SGL_DEVICE void tcgen05_st_32x32b_x8(
        uint32_t taddr, const uint32_t* src) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 "
                 " [%8], {%0, %1, %2, %3, %4, %5, %6, %7};"
                 :
                 : "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
                   "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]),
                   "r"(taddr));
}


// ---- Sync / commit ----------------------------------------------------------

// `tcgen05.wait::ld` blocks until all PRIOR tcgen05.ld (TMEM->reg drains) of
// this thread have COMPLETED. ptxas lowers it to per-load scoreboard waits on
// the dependent register consumers — but an `mbarrier.arrive` that does NOT
// read the drained registers has no such dependency, so without a barrier ptxas
// will HOIST the arrive ABOVE the last LDTM's completion (observed in SASS:
// the @216 "buffer-free" arrive scheduled between the overlap LDTM issue and
// its drain → the next tile's MMA reuses [220,256) while it is still draining =
// the cross-tile acc-overlap WAR). The "memory" clobber forces ptxas to keep
// every subsequent shared-memory op (the @216 arrive) AFTER this wait, so the
// drain provably retires before "buffer free" is signaled. This is the ordering
// CuteDSL gets from placing `tcgen05.wait::ld.sync.aligned` (cd:1835) right
// before the @216 arrive (cd:1840).
static SGL_DEVICE void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
static SGL_DEVICE void tcgen05_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// Commit prior tcgen05.mma operations to an mbarrier (arrive-on-one).
static SGL_DEVICE void tcgen05_commit_arrive(uint64_t* bar) {
    // Note: spec accepts `.shared::cluster` or no state-space; NOT `.shared::cta`.
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"(to_shared(bar)));
}

// cta_group::2 commit — pairs with cta_group::2 MMA. Signals the mbar in
// generic-proxy. By itself it only signals the LOCAL CTA's mbar; use the
// multicast variant below if you want both peer CTAs notified.
static SGL_DEVICE void tcgen05_commit_arrive_2sm(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [%0];"
                 :: "r"(to_shared(bar)));
}

// cta_group::2 commit with multicast::cluster — signals mbarriers at the
// SAME shared-memory offset in every CTA whose `%cluster_ctarank` bit is
// set in `cta_mask` (16-bit). Spec §9.7.16: "the mbarrier signal is
// multicast to the same offset as mbar in the shared memory of each
// destination CTA." Mbar address: pass a CTA-local pointer; the generic
// addressing mode resolves it to the per-CTA copy.
static SGL_DEVICE void tcgen05_commit_arrive_2sm_multicast(
        uint64_t* bar, uint16_t cta_mask) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                 ".shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :: "r"(to_shared(bar)), "h"(cta_mask));
}


static SGL_DEVICE void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// ---- MMA --------------------------------------------------------------------

// kind::f16 — F16/BF16 × F16/BF16 → F16 or FP32 in TMEM. Same operand
// shape as kind::f8f6f4 (smem-descriptor A, smem-descriptor B,
// instruction-descriptor, scale_c predicate). The MMA-Kind on the
// instruction determines whether the inst-desc atype/btype fields are
// interpreted per Table 44's f16 column (F16=0, BF16=1) vs the f8f6f4
// column.
//
// Valid shapes (cta_group::1, dense): M ∈ {64, 128}, N ∈ {8, 16, …, 256}
// steps of 8, K = 16. See `ptx/c_tcgen05_mma_dense/README.md` for the
// full table and the per-M layout rules. Off-table shapes are NOT
// rejected by ptxas — they hit cudaErrorIllegalInstruction at runtime,
// so consult Table 41 before changing any of (M, N).
static SGL_DEVICE void tcgen05_mma_f16(
        uint32_t d, uint64_t desc_a, uint64_t desc_b,
        uint32_t inst_desc_high, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(d), "l"(desc_a), "l"(desc_b), "r"(inst_desc_high), "r"(scale_c));
}


// kind::f16 with cta_group::2 — same operand shape, but the M dimension is
// distributed across 2 peer CTAs in a cluster (Layout A for M=256, Layout B
// for M=128). Each peer CTA must have called `tcgen05_alloc_2sm` and the
// `taddr` argument is the LOCAL CTA's TMEM base. The HW writes one half of
// the M rows into each peer's TMEM; pair with `tcgen05_commit_arrive_2sm`
// (or its multicast variant) for completion signaling.
//
// Valid shapes (cta_group::2, dense): M ∈ {128, 256}, N ∈ {16, 32, …, 256}
// **steps of 16** (note: NOT steps of 8 like cta_group::1), K = 16. See
// `ptx/c_tcgen05_mma_dense/README.md` (2cta path, "What changes between 1cta
// and 2cta" picking-table) for the full table and details. Off-
// table shapes hit cudaErrorIllegalInstruction at the *commit* (not the
// MMA) — chase the descriptor, not the commit, when debugging.
static SGL_DEVICE void tcgen05_mma_f16_2sm(
        uint32_t d, uint64_t desc_a, uint64_t desc_b,
        uint32_t inst_desc_high, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(d), "l"(desc_a), "l"(desc_b), "r"(inst_desc_high), "r"(scale_c));
}


// kind::f8f6f4 — FP8 (E4M3/E5M2/E3M2/E2M3) × FP8/FP6/FP4 → FP32 in TMEM.
// d        : 32-bit TMEM address (first cell of D).
// desc_a/b : 64-bit shared-memory matrix descriptors.
// inst_desc_high : upper 32 bits of the 64-bit instruction descriptor (the
//                  PTX op uses the upper 32 only).
// scale_c  : 0 = D = A·B; non-zero = D = D + A·B (predicate).
static SGL_DEVICE void tcgen05_mma_f8f6f4(
        uint32_t d, uint64_t desc_a, uint64_t desc_b,
        uint32_t inst_desc_high, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(d), "l"(desc_a), "l"(desc_b), "r"(inst_desc_high), "r"(scale_c));
}

// kind::f8f6f4 with cta_group::2 — same operand shape as the 1cta variant,
// but the M dimension is distributed across 2 peer CTAs in a cluster
// (Layout A for M=256, Layout B for M=128). Each peer must have called
// `tcgen05_alloc_2sm`. Dense (no SF), so no sf_a/sf_b operands. Pair with
// `tcgen05_commit_arrive_2sm` (or its multicast variant) for completion.
static SGL_DEVICE void tcgen05_mma_f8f6f4_2sm(
        uint32_t d, uint64_t desc_a, uint64_t desc_b,
        uint32_t inst_desc_high, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(d), "l"(desc_a), "l"(desc_b), "r"(inst_desc_high), "r"(scale_c));
}

// kind::tf32 — TF32 × TF32 → F32 in TMEM. Operand stored as fp32 (4 B/elem)
// in smem; the MMA reads the truncated 19-bit tf32 mantissa from the fp32
// word (no host cvt). Same operand convention as kind::f16 (smem-desc A,
// smem-desc B, inst-desc, scale_c predicate). K=8 per call.
//
// Valid shapes (cta_group::1, dense): M ∈ {64, 128}, N ∈ {8, 16, …, 256}
// steps of 8, K = 8. Provenance: kernels/qr/studies/inhouse_gemm/tf32_mma.cuh.
static SGL_DEVICE void tcgen05_mma_tf32(
        uint32_t d, uint64_t desc_a, uint64_t desc_b,
        uint32_t inst_desc_high, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(d), "l"(desc_a), "l"(desc_b), "r"(inst_desc_high), "r"(scale_c));
}

// kind::tf32 with cta_group::2 — same operand shape as the 1cta variant, but
// the M dimension is distributed across 2 peer CTAs in a cluster. Each peer
// must have called `tcgen05_alloc_2sm`. Dense (no SF). Pair with
// `tcgen05_commit_arrive_2sm` (or its multicast variant) for completion.
//
// Valid shapes (cta_group::2, dense): M ∈ {128, 256}, N ∈ {16, 32, …, 256}
// **steps of 16**, K = 8.
static SGL_DEVICE void tcgen05_mma_tf32_2sm(
        uint32_t d, uint64_t desc_a, uint64_t desc_b,
        uint32_t inst_desc_high, uint32_t scale_c) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(d), "l"(desc_a), "l"(desc_b), "r"(inst_desc_high), "r"(scale_c));
}


// ---- Sparse mxf4nvf4 MMA wrappers (tcgen05.mma.sp) -------------------------
//
// Sparse MMA where matrix A is 4:8 structured-sparse (§9.7.16.10.8.3 lines
// 3222-3245): each row of A has 50% zeros in pair-wise structured chunks of
// 8 elements (4 zero + 4 non-zero, where zero/non-zero clusters are 2-wide
// sub-chunks). Only the 4 non-zero elements per 8-wide chunk are stored in
// memory, halving A's footprint. The sparse metadata at `[sp-metadata-tmem]`
// encodes the positions of the 2 non-zero sub-chunks per 8-wide chunk via
// 2 two-bit indices (one of {0b0100, 0b1000, 0b1100, 0b1001, 0b1101, 0b0110,
// 0b1110} per the spec — all 8 other 4-bit codes are undefined behavior).
//
// MMA shape consequence: matrix A is logically Mx(K/2) (only the non-zero
// stored), B is KxN, D is MxN. For K=128 sparse mxf4nvf4 the stored A is
// MxK_packed where K_packed = K/2 = 64 elements per row (= 32 bytes per
// row, 2 nibbles/byte; same smem footprint as K=64 dense). FLOPs/MMA are
// computed under the dense-equivalent K = 128 convention (which is what
// NVIDIA marketing-peak sparsity numbers use): FLOPs = 2 * M * N * 128.
//
// Sm support (§9.7.16.10.9.2 lines 3971-3975): tcgen05.mma.sp.kind::mxf4nvf4
// supported on sm_100a / sm_101a (sm_110a) / sm_103a / sm_110a. B300 (sm_103a)
// is GREEN.
//
// Operand convention vs dense:
//   - 4 operands: [d-tmem], a-desc, b-desc, [sp-metadata-tmem], idesc, [scale-A-tmem],
//     [scale-B-tmem], enable-input-d
//   - sp-meta-tmem points to the TMEM cells holding the metadata indices.
//   - idesc bit 2 (Sparsity) MUST be 1 — use `mma_inst_desc_mxf4nvf4_block16(...,
//     sparse=true)`.
//   - For block16 + K=128, SFA_ID and SFB_ID MUST be 0 (Table 58 / Figures 233,
//     242 — all sub-columns are auto-selected; no SF ID offset).

// kind::mxf4nvf4.block_scale.block16 sparse, cta_group::1 (M=128, K=128 sparse).
// `sp_meta` is a TMEM byte address holding the metadata cells.
static SGL_DEVICE void tcgen05_mma_mxf4nvf4_block16_sp(
        uint32_t d, uint64_t desc_a, uint64_t desc_b,
        uint32_t sp_meta, uint32_t inst_desc_high, uint32_t scale_c,
        uint32_t sf_a, uint32_t sf_b) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16"
        "  [%0], %1, %2, [%3], %4, [%6], [%7], p;\n\t}\n"
        :: "r"(d), "l"(desc_a), "l"(desc_b), "r"(sp_meta),
           "r"(inst_desc_high), "r"(scale_c), "r"(sf_a), "r"(sf_b));
}


}  // namespace ptx
