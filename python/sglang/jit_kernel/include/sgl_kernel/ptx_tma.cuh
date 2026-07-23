/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <sgl_kernel/ptx_addr.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// ================= common/ptx_tma.cuh =================
// TMA (Tensor Memory Accelerator) wrappers. PTX ISA 9.2 §9.7.9.25.
//
// CHOOSING THE FORM:
//   1D bulk (cp.async.bulk)         — linear memcpy, no tensor map.
//                                     Use for combine buffers, scratch, packed
//                                     metadata — anything not "tile-shaped".
//   2D tile (cp.async.bulk.tensor.2d) — strided tile of a 2D tensor, optional
//                                       swizzle for tensor-core consumption.
//                                       Use for any tile feeding an MMA.
//
// COMPLETION MECHANISM (asymmetric — load and store differ):
//   load  (gmem→smem): mbarrier-based. Pair with `mbar_arrive_expect_tx(bar, BYTES)`
//                      then `mbar_wait_parity(bar, parity)`. The TMA hw decrements
//                      the mbarrier's tx-count as bytes land. With default `.acquire`
//                      wait, no `fence.proxy.async` needed afterward
//                      (ptx_mbarrier.cuh / ptx/b_fence/README.md explain why).
//   store (smem→gmem): bulk-group based — stores CANNOT use mbarrier (destination
//                      is global, no smem object to attach to). Pair with
//                      `fence_async_smem()` (REQUIRED — see ptx/b_fence/README.md)
//                      then `tma_store_commit()` then `tma_store_wait<N>()`.
//
// SWIZZLE CHOICE (for the 2D form): drives bank-conflict-free smem reads by
// the consumer.
//   K-major MMA feed   → swizzle == K_bytes (mandatory; assertion in DeepGEMM
//                                            mma/sm90.cuh:251).
//   MN-major MMA feed  → min(BLOCK_MN_bytes, 128).
//   Not consumed by MMA → SWIZZLE_NONE (swizzle's only purpose is bank-conflict
//                                       avoidance for tensor cores).
// Decode formulas + empirical verification: see common/swizzle.h and
// ptx/a_tma_2d/README.md.
//
// ALIGNMENT (PTX ISA §9.7.9.25.4.1 + §9.7.9.25.5.2):
//   - 1D bulk: bytes must be multiple of 16; both endpoints aligned to 16.
//   - 2D tile: smem destination 16-byte aligned for no-swizzle, 128-byte for
//              swizzled. Over-align to 128 always — costs nothing.


namespace ptx {

// ---- tensor-map descriptor prefetch ------------------------------------------

// Warm the cache line holding a TMA tensor-map descriptor so the FIRST
// cp.async.bulk.tensor load of the persistent loop doesn't pay the
// descriptor-fetch latency (the steady-state loads then hit cache). cd issues
// this in the producer prologue (cpasync.prefetch_descriptor, one elected lane
// per TMA warp) before entering the mainloop. `tmap` is a generic address into
// the __grid_constant__ CUtensorMap param — no space qualifier → generic
// addressing resolves it to .param (PTX ISA §9.7.9.15, line 1888). Issue once,
// off the hot path. Idempotent / side-effect-free beyond cache state.
static __device__ __forceinline__ void prefetch_tensormap(const void* tmap) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(tmap) : "memory");
}


// ---- 1D bulk (no tensor map) -------------------------------------------------


// ---- 2D tile-mode TMA (with tensor map) -------------------------------------

// COORDINATE CONVENTION (the easy-to-flip part): the tensor map's `globalDim`
// is `(inner, outer)` — dim 0 is the stride-1 axis. For a row-major (rows,
// cols) tensor with `cols` innermost, encode as
// `encode_tiled_2d(global_rows=rows, global_cols=cols, ...)`. The kernel
// load/store calls then take `(x = inner_offset, y = outer_offset)` —
// `x` indexes into cols, `y` into rows. Mismatch and you'll load the
// transposed tile (likely with right-looking magnitudes but scrambled
// per-cell pairing — easy to confuse with a real bug).

// global → shared::cta. tmap is a CUtensorMap by pointer (typically a
// __grid_constant__ kernel arg).
static __device__ __forceinline__ void cp_async_bulk_tensor_2d_load(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst_smem), "l"(tmap), "r"(x), "r"(y), "r"(to_shared(bar))
        : "memory");
}


// 4D single-CTA tile load (no cluster/multicast). Coordinate convention as 3D:
// `x` = innermost (stride-1) offset, `y`/`z`/`w` = the next-outer dims. The
// only pre-existing 4D forms are cta_group::2 cluster loads (multicast / 2sm /
// bit24) — this plain shared::cta form is the D18 form-(a) SWIZZLE_128B
// one-issue hd128 KV feed for kernels/attention/decode (a [1, T_kv, hd/64, 64]
// box over the paged [n_pages, P, hd/64, 64] pool; the fmha_floors PR-3ext
// probe proved it byte-exact against the 128B-swizzle read-back). Guarded so a
// concurrent common/ edit that also adds it does not double-define.
#ifndef PTX_TMA_HAS_4D_SINGLE_CTA_LOAD
#define PTX_TMA_HAS_4D_SINGLE_CTA_LOAD 1

#endif  // PTX_TMA_HAS_4D_SINGLE_CTA_LOAD

// CLUSTER VARIANT (sm_90 form): same as above, but the destination mbar
// (`bar`) may reside in a peer CTA's smem (within the same cluster).
// Mbar address is the local-smem byte offset; the `.shared::cluster`
// qualifier tells the HW to look up the cross-CTA byte at the same offset.
//
// IMPORTANT: on sm_100+ (Blackwell), this sm_90-style instruction does NOT
// propagate the tx-count decrement across CTAs — the decrement stays
// local. Use `cp_async_bulk_tensor_2d_load_2sm` below for cluster MMA on
// sm_100+. This wrapper is here for sm_90 compatibility.
static __device__ __forceinline__ void cp_async_bulk_tensor_2d_load_cluster(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, uint64_t* bar, uint32_t cta_rank) {
    const uint32_t mapped = mapa_shared_cluster(to_shared(bar), cta_rank);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst_smem), "l"(tmap), "r"(x), "r"(y), "r"(mapped)
        : "memory");
}

// 2-CTA CLUSTER VARIANT (sm_100+): the canonical TMA load for cluster MMA
// on Blackwell. Has `.cta_group::2` qualifier — required for the HW to
// route tx-count across CTAs in a 2-CTA cluster.
//
// IMPORTANT — the sm_103a hazard. The sm_90-style `cp_async_bulk_tensor_2d_load_cluster`
// above (`.shared::cluster.tile.mbarrier::complete_tx::bytes`) compiles
// cleanly on sm_103a and the load itself appears to issue, but the
// `mbarrier::complete_tx::bytes` decrement does NOT propagate to the
// addressed peer CTA's mbar — it stays local. The sm_90 form works on
// B200 (sm_100a) but the same kernel hangs on B300 (sm_103a) until you
// switch to this `cta_group::2` form.
//
// Pattern: BOTH peer CTAs issue this instruction. Mbar address is mapped
// into CTA-0's smem via `ptx::mapa_shared_cluster(local_bar, /*cta_rank=*/0)`
// so both CTAs' tx-count decrements land on CTA-0's mbar. Pair with
// `mbar_arrive_expect_tx_cluster`. Mbar init count = CTA_GROUP = 2.
//
// First-principles derivation: see `ptx/a_tma_2d/README.md` "Cluster scope".
//
// `cache_hint` defaults to EVICT_NORMAL (0x10C0_0000_0000_0000).
static __device__ __forceinline__ void cp_async_bulk_tensor_2d_load_2sm(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, uint64_t* bar, uint32_t cta_rank,
        uint64_t cache_hint = 0x0ULL) {
    const uint32_t mapped = mapa_shared_cluster(to_shared(bar), cta_rank);
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(dst_smem), "l"(tmap), "r"(x), "r"(y), "r"(mapped), "l"(cache_hint)
        : "memory");
}


// MULTICAST TMA load (cta_group::2 + multicast::cluster). Loads the SAME
// bytes from gmem to MULTIPLE peers' smem within a cluster (identical
// destination layout). Both peers observe byte-identical smem post-load.
//
// Use case: cluster-scoped pre-reqs that need both peers' smem identical
// (so a subsequent UTCCP `cta_group::2.warpx4` broadcast is idempotent —
// see `ptx/c_tcgen05_tmem/README.md` § "UTCCP `cta_group::2.warpx4` is
// BROADCAST"). DG `sm100_fp8_fp4_mega_moe.cuh:824-839` is the canonical
// pattern (`tma::copy<SF_BLOCK_M, 1, 0>(..., 2)` — trailing `2` =
// num_tma_multicast).
//
// Pattern: ONE leader CTA issues this instruction (single issuer per call);
// `multicast_mask` selects which CTAs in the cluster receive the data.
// Both peers' smem at `dst_smem` (the SAME local-smem byte offset on each
// peer) gets byte-identical bytes. The mbar's bit 24 is cleared
// (`Sm100MmaPeerBitMask = 0xFEFFFFFF`) so the tx-count completion routes
// to CTA-0's mbar — multicast TMA fans data out, consolidates completion
// to one mbar.
//
// `multicast_mask` is a 16-bit bitfield: bit `i` = "include CTA `i` of
// cluster as a multicast target". For 2-CTA cluster: 0b11.
//
// Mbar arrival pattern (from DG):
//   if (is_leader_cta) {
//       mbar->arrive_and_expect_tx(BYTES_FOR_ONE_PEER * NUM_PEERS);
//   } else {
//       mbar->arrive(0u);
//   }
// CTA-0's mbar must have `count = NUM_PEERS` so both peers' arrives land.
// Tx-count totals `BYTES * NUM_PEERS` because the multicast TMA writes
// `BYTES` to each peer's smem (so the HW reports each peer's load as a
// separate tx-decrement on CTA-0's mbar).
//
// Reference: PTX ISA 9.2 §9.7.9.25 (multicast::cluster qualifier).
static __device__ __forceinline__ void cp_async_bulk_tensor_2d_load_multicast(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, uint64_t* bar,
        uint16_t multicast_mask = 0b11,
        uint64_t cache_hint = 0x0ULL) {
    // Clear bit 24 of mbar address — Sm100MmaPeerBitMask routes tx-count
    // completion to leader CTA-0's mbar regardless of which peer issues.
    const uint32_t mbar_addr = to_shared(bar) & 0xFEFFFFFFu;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        " [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"(dst_smem), "l"(tmap), "r"(mbar_addr), "h"(multicast_mask),
           "r"(x), "r"(y), "l"(cache_hint)
        : "memory");
}

// MULTICAST TMA load (cta_group::1 + multicast::cluster). PTX ISA 9.2
// §9.7.9.25 (multicast::cluster, .cta_group::1). Loads the SAME bytes from
// gmem to MULTIPLE peers' smem at the SAME CTA-relative offset; with
// `.cta_group::1` the mbarrier complete-tx signal is ALSO multicast "to the
// same offset as mbar in the shared memory of the destination CTA" (spec
// §9.7.9.25, .cta_group::1 bullet). i.e. EACH destination CTA's local mbar
// at `dst_bar`'s offset receives its OWN `BYTES` tx-decrement.
//
// CONTRAST with `cp_async_bulk_tensor_2d_load_multicast` (cta_group::2):
//   - cta_group::2 CONSOLIDATES all tx-completion onto ONE CTA's mbar
//     (bit-24 cleared → CTA-0). Total tx = BYTES * NUM_PEERS on one mbar.
//   - cta_group::1 DISTRIBUTES: each peer's mbar (same offset) gets BYTES.
//     Total tx = BYTES on each peer's own mbar.
// USE cta_group::1 when each peer keeps a per-CTA-LOCAL mbar that its own
// consumer warp waits on (so the leader's single DRAM read fills both
// peers' smem AND signals both peers' local mbars). This is the pattern
// for SFB in `kernels/fused_gemm_2cta_sf` where each CTA's `tma_mbars`
// are local and self-decremented — the follower keeps its
// `expect_tx += SFB_BYTES` and stops ISSUING the load; the leader's one
// multicast both fills the follower's smem and decrements the follower's
// mbar. Halves the SFB DRAM traffic with NO change to either peer's
// expect_tx accounting.
//
// `dst_smem` / `dst_bar` are the issuer's LOCAL `.shared::cta` byte
// offsets; the HW interprets them in the `.shared::cluster` window at the
// same offset in each ctaMask-selected CTA (do NOT clear bit 24 — that is
// the cta_group::2 consolidation trick and would mis-route the signal).
//
// `multicast_mask`: bit `i` = include CTA `i` of cluster. 2-CTA = 0b11.
// Pattern: ONE leader CTA issues; each peer (leader AND follower) keeps its
// own `mbar_arrive_expect_tx(local_bar, BYTES)`.
static __device__ __forceinline__ void cp_async_bulk_tensor_2d_load_multicast_cg1(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, uint64_t* bar,
        uint16_t multicast_mask = 0b11,
        uint64_t cache_hint = 0x0ULL) {
    const uint32_t mbar_addr = to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::1.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        " [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"(dst_smem), "l"(tmap), "r"(mbar_addr), "h"(multicast_mask),
           "r"(x), "r"(y), "l"(cache_hint)
        : "memory");
}


// ====================================================================
// CuteDSL-faithful (2,4) 8-CTA tiled-multicast TMA (sf S1).
// ====================================================================
// PER-RECIPIENT tx routing (the (2,4) cluster property the consolidating
// wrappers above do NOT give). Spec §9.7.9.25 (cta_group::2.multicast::cluster):
// "the mbarrier signal is multicasted either to all the odd numbered CTAs or
//  the even numbered CTAs within the corresponding CTA-Pair ... based on the
//  CTA's %cluster_ctarank parity of shared memory where the mbarrier object
//  resides." Clearing bit 24 (0xFEFFFFFF) forces the mbar address to EVEN
// parity → the complete-tx signal lands on the EVEN (m_pair==0) CTA of EACH
// pair selected by the mask. In a (2,4) 8-CTA cluster the A mask 0x55 selects
// the 4 even CTAs {0,2,4,6} (one per N-rank), so each of the 4 N-ranks' leader
// CTA receives ITS OWN tx-decrement on its OWN local AB-ready mbar — NOT all
// consolidated onto cluster CTA-0. This is the cd UTMALDG.3D/4D.MULTICAST.2CTA
// routing: 1 DRAM read multicast to N recipients, each counting its own bytes.
// cd cubin: 0xfefffff8 mask on the mbar addr (bit-24 + 8B-align clear). The
// issuing CTA arms its OWN AB-ready mbar with `mbar_arrive_expect_tx` (LOCAL
// .shared::cta) BEFORE the copy — see the S1 producer in kernel.cu.

// (The cd 3D/4D-collapse forms — UTMALDG.3D.2CTA for B, UTMALDG.4D.MULTICAST.2CTA
// for SF — would need 3D/4D HOST tensormaps; the S1 sf feed instead keeps the
// proven 2D packed-straddle gmem layout and realizes the same multicast DRAM
// fan-out with the existing 2D multicast / 2sm_bit24 forms. If a future step
// rebuilds the host tmaps to 3D/4D, add the cta_group::2 3D/4D load wrappers here.)


// Make a 64-bit L2 cache-policy descriptor via PTX `createpolicy.fractional`
// (sm_80+). Fraction defaults to 1.0 — apply the policy to 100% of bytes.
// Returns an opaque uint64 cookie to feed `L2::cache_hint` qualifiers on
// TMA / load / store instructions.
//
// `EVICT_FIRST` is appropriate for one-shot output tiles (D in GEMM): the
// bytes are written once and never re-read by this kernel, so we mark them
// evict-first so they don't push the next tile's A/B tiles out of L2.
// Saves ~0.5-1% on big GEMMs where the D matrix fights A/B for L2 residency.
//
// `EVICT_LAST` would be the right hint for hot data we want to keep
// resident (e.g., a small per-expert weight slab); not used here.
//
// REFERENCE: PTX ISA §9.7.9.4 (createpolicy.fractional).
enum class L2EvictPolicy : int { NORMAL = 0, EVICT_FIRST = 1, EVICT_LAST = 2 };


// ---- bulk-group completion --------------------------------------------------

// Close the per-thread bulk async-group containing all prior bulk_group ops.
static __device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;");
}

// Wait until at most N bulk-groups are pending. tma_store_wait_all() = wait_group 0.
// Use N > 0 for pipelined producer-consumer (next group can issue while older
// groups still in flight); N = 0 to drain everything.
template <int N = 0>
static __device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N));
}

static __device__ __forceinline__ void tma_store_wait_all() {
    asm volatile("cp.async.bulk.wait_group 0;");
}

}  // namespace ptx
