/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <sgl_kernel/ptx/addr.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

// ================= common/ptx/mbarrier.cuh =================
// mbarrier wrappers. PTX ISA 9.2 §9.7.13.15.
//
// Wait variant: this file ONLY wraps `try_wait.parity`. The other two waiter
// axes — state-token (`mbarrier.try_wait`) and busy-spin (`mbarrier.test_wait`,
// `mbarrier.test_wait.parity`) — exist in PTX but are not wrapped here:
//
//   - state-token waits couple the arriver and waiter (the waiter must hold
//     the 64-bit token returned by `mbarrier.arrive`). Every same-thread
//     arrive+wait pattern can be written equivalently with parity tracking
//     and is harder to mis-use that way; mixed state/parity codepaths in the
//     same kernel are a deadlock risk. Caller-side phase counters are
//     cheap.
//   - `test_wait` busy-spins on the issue pipeline, which is only worth it
//     for sub-microsecond waits (rare in real code). `try_wait` may suspend
//     the thread on long waits, freeing the SM for other warps. Required
//     sm_90+; we already target sm_103a.
//     ⚠ MEASURED SASS REALITY (ptxas 13.x, sm_103a; kernels/gemm/LEDGER.md
//     L-sf-power-op-w3b; adversarially reproduced vw3b): the NO-timeHint
//     `try_wait` form compiles to a
//     raw `SYNCS...TRYWAIT + BRA` loop — a FULL-RATE BUSY-SPIN, no suspend at
//     all. Only the timeHint form (`..., 10000000`) emits the parked loop
//     (`TRYWAIT + NANOSLEEP.SYNCS <hint>`). A no-hint spin concurrent with
//     other warps' work (e.g. a teardown wait beside the last epi stores)
//     steals issue slots — prefer the hint form or a nanosleep backoff there.
//
// If you ever need the unwrapped instructions, the inline-asm shapes are:
//     mbarrier.try_wait.shared.b64        p, [bar], state;   // state-token
//     mbarrier.test_wait.parity.shared.b64 p, [bar], parity; // busy parity
//     mbarrier.test_wait.shared.b64        p, [bar], state;  // busy state
//
// Parity bookkeeping: `mbarrier.try_wait.parity bar, parity_arg` returns when
// `current_phase_parity != parity_arg`. After init, parity is 0; each full
// cycle (count arrivals → bar fires → reset) flips it. Caller maintains the
// phase counter (typically `phase ^= (stage == 0)` at the stage-wrap).
//
// Conventions:
//   - wrappers take a `uint64_t*` to the mbar object in smem; `to_shared`
//     is done internally. Pass `&array[i]` or `&single_bar`; for a known-
//     shared array nvcc folds `__cvta_generic_to_shared` to a single PTX
//     instruction (or nothing).
//   - all wrappers are zero-cost (static SGL_DEVICE).
//   - the spin-wait wrapper uses `WAIT_%=` for inline-asm label uniqueness so
//     it can be inlined multiple times in the same kernel without colliding.
//   - default `.sem` for `try_wait` is `.acquire` per the spec — we don't
//     override it, so we get acquire semantics + the standard happens-before
//     with prior `arrive.release` ops.
//
// Cross-proxy visibility (the load-side "do I need fence.proxy.async?" question):
//   When a wait below returns True with default `.acquire` semantics, prior
//   `cp.async.bulk` writes tracked by THIS mbarrier are visible to subsequent
//   generic-proxy reads on the executing thread — no `fence.proxy.async`
//   needed after a TMA load + wait. Spec basis: §9.7.13.15.16 point 3.
//   This guarantee disappears with `.relaxed`; if you ever pass that, add an
//   explicit `fence.proxy.async.shared::cta` after the wait. Full decision
//   matrix (load vs store, acquire vs relaxed) lives in ptx/b_fence/README.md.
//
// Parity initial-value convention (the "what phase do I pass on the first
// wait?" question, easy to flip and deadlock):
//   After `mbar_init(bar, count)` the bar is at parity 0. Phase tracking
//   means each FULL cycle (count arrivals → bar fires → reset) flips parity.
//   The first wait_parity call must pass the phase the caller expects to
//   see when the bar is "currently un-fired" — this depends on whether
//   the caller acts as producer-first or consumer-first:
//     - Consumer-first (waits for an external producer's first signal):
//       init_phase = 0. First wait blocks until the producer fires bar →
//       parity flips to 1, wait returns. Caller then flips to 1 for the
//       second wait. (TMA-consumer warp pattern in fused_gemm.)
//     - Producer-first (waits for the consumer to release a shared
//       resource, with no prior consumer activity yet): init_phase = 1.
//       First wait is a no-op skip (current parity is 0, expected is 1
//       → predicate already true). Subsequent waits track normally.
//       (TMA-producer warp pattern: at stage `s` it waits on `mma_mbar[s]`
//       saying "is the MMA done with this slot?" — for the first visit
//       there's no MMA yet, so init_phase=1 makes the first wait return.)
//   Mismatch deadlocks: a consumer-first wait initialized to phase=1 will
//   skip the producer's first signal and block forever on the second.
//   Caller maintains the phase counter and flips per stage-wrap (or
//   per-cycle).


namespace ptx {

// mbarrier.init [bar], count;
static SGL_DEVICE void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"(to_shared(bar)), "r"(count));
}


// mbarrier.arrive [bar]. Returns the 64-bit phase state token, but the only
// remaining waiter wrapper is parity-based, so the return is typically
// discarded.
static SGL_DEVICE uint64_t mbar_arrive(uint64_t* bar) {
    uint64_t state;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
                 : "=l"(state) : "r"(to_shared(bar)));
    return state;
}

// CLUSTER VARIANT: arrive on an mbar in CTA `cta_rank`'s smem (same cluster).
// `mapa.shared::cluster.u32` (via `mapa_shared_cluster`) translates the
// local smem byte offset to the cluster-mapped address that targets
// `cta_rank`'s copy of that smem location. The `.shared::cluster` qualifier
// on `mbarrier.arrive` is REQUIRED for cross-CTA mbar access — the plain
// `.shared` form targets the local CTA's mbar regardless of address bit 24.
//
// State-token return is sinked (`_`) since cross-CTA waits are parity-based.
//
// First-principles derivation: see `ptx/b_mbarrier/README.md` "Cluster scope".
static SGL_DEVICE void mbar_arrive_cluster(uint64_t* bar, uint32_t cta_rank) {
    const uint32_t mapped = mapa_shared_cluster(to_shared(bar), cta_rank);
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(mapped));
}

// RELEASE VARIANT: cross-CTA arrive with `.release.cta` memory-ordering
// semantics. The plain `mbar_arrive_cluster` above carries NO release — it
// only updates the mbar phase, so prior memory ops (notably a retired
// `tcgen05.ld` TMEM drain on the arriving warp) are NOT guaranteed
// visible-before a *peer* warp's `.acquire` wait returns. When the wait's
// consumer then overwrites that TMEM (e.g. MMA(t+1) reusing a single-buffered
// D accumulator after the epi early-releases), the missing release/acquire
// happens-before is a true memory race (spec §8.8: a release pattern requires
// `mbarrier.arrive.release [M]`, NOT a plain arrive). This is the exact form
// CuteDSL emits for its early-release `accum_empty` arrive
// (`mbarrier.arrive.release.cta.shared::cluster.b64 _, [addr], 1`, PTX 1840),
// paired with the MMA-warp's default-`.acquire` `try_wait`. Use this whenever
// the arriving warp has done TMEM/smem work the waiting peer must observe.
static SGL_DEVICE void mbar_arrive_cluster_release(uint64_t* bar, uint32_t cta_rank) {
    const uint32_t mapped = mapa_shared_cluster(to_shared(bar), cta_rank);
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;" :: "r"(mapped));
}


// mbarrier.arrive.expect_tx [bar], bytes; (combined arrive + set tx-count).
// State is sinked with `_` — caller must use parity-based wait.
static SGL_DEVICE void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 :: "r"(to_shared(bar)), "r"(bytes));
}

// CLUSTER VARIANT: like mbar_arrive_expect_tx but the mbar lives in
// CTA `cta_rank`'s smem (same cluster). Used when accumulating tx-count from
// multiple peer CTAs into one CTA's mbar (typical: 2-CTA TMA where every
// CTA's arrive_expect_tx targets CTA-0's mbar). Pair with the cluster TMA
// load (`cp_async_bulk_tensor_2d_load_cluster`).
static SGL_DEVICE void mbar_arrive_expect_tx_cluster(
        uint64_t* bar, uint32_t cta_rank, uint32_t bytes) {
    const uint32_t mapped = mapa_shared_cluster(to_shared(bar), cta_rank);
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mapped), "r"(bytes));
}

// Wait for phase `parity` (0 or 1) to complete. `try_wait` form: hardware may
// suspend the thread; we loop because the spec allows spurious early wakeups
// (system timeout). The default and only wait wrapper here — see header
// comment for why state-token / test_wait variants are intentionally absent.
static SGL_DEVICE void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT_%=;\n\t}\n"
        :: "r"(to_shared(bar)), "r"(parity));
}


// CLUSTER VARIANT (BROKEN on sm_103a — DO NOT USE).
//
// This wrapper attempts to wait on an mbar that lives in peer CTA
// `cta_rank`'s smem via `mbarrier.try_wait.parity.shared::cluster`.
// Use case would have been: producer TMA reports tx-count to CTA-0's
// mbar, consumers in OTHER peers want to observe completion. ptxas
// (CUDA 13.0, sm_103a) REJECTS the `.shared::cluster` qualifier on
// `mbarrier.try_wait.parity` with "Illegal modifier '::cluster' for
// instruction 'mbarrier.try_wait.parity'". Surfaced by Phase 3.2's
// CLC microbench (`recipes/_scratch/clc_warp0_multiplex_probe/`).
//
// CORRECT PATTERN for cross-CTA mbar wait on sm_103a: use the
// multicast-commit form so the producer writes to BOTH peers' local
// mbars at the same offset (e.g., `tcgen05_commit_arrive_2sm_multicast`
// in fused_gemm_2cta), and each peer waits on its own LOCAL mbar via
// `mbar_wait_parity`. There is no working `try_wait` form that observes
// a peer's mbar.
//
// Wrapper kept (rather than deleted) so future readers find this comment
// before re-attempting the pattern; do not call it.
// Implementation removed: ptxas rejects the `.shared::cluster` qualifier on
// `mbarrier.try_wait.parity`. If you need cross-CTA mbar observation,
// restructure to use multicast-commit (see comment above).

}  // namespace ptx
