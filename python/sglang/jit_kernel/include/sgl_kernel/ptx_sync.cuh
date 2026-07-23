/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ================= common/ptx_sync.cuh =================
// Lightweight sync wrappers: named barrier.cta, fence.proxy.async variants.
// PTX ISA 9.2 §9.7.13.


namespace ptx {


// fence.proxy.async.shared::cta — make generic-proxy smem writes visible to
// the async proxy (TMA store engine). Required before any cp.async.bulk store
// that reads smem written by regular ld/st.
static __device__ __forceinline__ void fence_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;");
}


// ---- Cluster-wide barrier (sm_90a+, sm_100a+, sm_103a) ----------------------
//
// `barrier.cluster.{arrive,wait}` synchronizes ALL threads of ALL CTAs in
// the launching cluster. Use when work in one CTA must observe state
// produced by another CTA in the same cluster (typical: smem writes,
// mbarrier inits, TMEM allocations).
//
// `cluster_sync` is the canonical "fence both ways" pattern. When you only
// need one direction (e.g., signaling work-done before the peer reads),
// the separate arrive/wait wrappers let you interleave other work between.


static __device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
}

// `cluster_sync` with explicit `release` / `acquire` semantics. Use this
// when the cluster_sync is also the publish/observe boundary for memory
// writes done before/after — e.g., between `mbarrier.init` (release) and
// any `.shared::cluster` use of those mbars (acquire). PTX ISA §9.7.13.
static __device__ __forceinline__ void cluster_sync_rel_acq() {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}


// ---- elect.sync — pick one lane in the warp ---------------------------------
//
// Returns true on exactly one lane of the issuing warp; false on the others.
// All 32 lanes execute `elect.sync` (it's a sync instruction); the HW chooses
// one. Use to guard "single-thread-issuer" sites (mbar.init, TMA issue, MMA
// issue, alloc/dealloc) without having to gate on `lane_id == 0`.
//
// PTX ISA 9.2 §9.7.4. sm_90+.
static __device__ __forceinline__ bool elect_one() {
    uint32_t pred;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "elect.sync _|p, 0xffffffff;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}\n"
        : "=r"(pred));
    return pred != 0;
}


// ---- setmaxnreg.{dec,inc} — per-warpgroup register-budget reallocation ------
//
// Reallocates the warp-group register file at runtime. Use to widen the
// epilogue's per-thread reg budget (so it can hold a larger primary array
// without spilling to local memory) at the cost of narrowing the mainloop
// warps, which typically need few regs.
//
// PTX ISA 9.2 §9.7.12.6 (`setmaxnreg`):
//   - `dec` lowers the warp's max-allocatable register count to N; the
//     released physical regs are returned to the per-SM RF pool.
//   - `inc` raises the warp's max-allocatable register count to N; the
//     additional physical regs are pulled from the per-SM RF pool.
//   - Both forms are warp-group-synchronizing (`.sync.aligned`): all 128
//     threads of the issuing warp-group must execute the SAME instruction
//     with the SAME N, and the instruction acts as an aligned barrier
//     across the 4-warp group.
//   - N range: 24 ≤ N ≤ 256, multiple of 8. Per-thread.
//   - The RF cap on B100/B300 is 64 K regs/SM (`64512` is the safe
//     allocatable cap accounting for ~1024 reserved regs). The total
//     budget across all warp-groups in the CTA must satisfy
//     `Σ (warp_group_threads × N) ≤ 64512`. Caller is responsible for
//     budgeting (no compile-time check possible — `N` per warp-group is
//     orthogonal).
//   - Issue site MUST be on a warp-group boundary (warps 0-3 or 4-7 in
//     an 8-warp CTA). Issuing only from warp 0 of a group will hang.
//   - Available on sm_90+ (Hopper+).
//
// Example (8-warp CTA, mainloop warps 0-3, epilogue warps 4-7):
//     if (warp_id < 4) ptx::setmaxnreg_dec<48>();   // mainloop
//     else             ptx::setmaxnreg_inc<208>();  // epilogue
//   Budget check: 4*32*48 + 4*32*208 = 6144 + 26624 = 32768 ≤ 64512.
//
// WHEN TO USE (B100/B300):
//   Use `setmaxnreg_{dec,inc}` for warp-specialized GEMMs with
//   *asymmetric* per-warp-group reg budgets where ptxas can't see the
//   asymmetry from the source (e.g. mainloop warps need 48 regs but
//   epilogue warps need 208; static `__launch_bounds__` would force the
//   whole CTA to 208).
//
//   For *symmetric* reg-cap raising (all warpgroups same budget),
//   `__launch_bounds__(NUM_THREADS, 1)` is cleaner — ptxas already
//   accounts for it without runtime instructions. DG Tech 6 (docs/archive/LESSONS.md
//   line ~590, 2026-05-05 W2.D abort) found that `setmaxnreg.dec/inc`
//   on B100/B300 ptxas does not pay vs `__launch_bounds__` for the
//   symmetric case (regressed -2 to -5%). The asymmetric case still
//   wins on memory-bound epilogues per the example above.

template <int N>
static __device__ __forceinline__ void setmaxnreg_dec() {
    static_assert(N >= 24 && N <= 256, "setmaxnreg N must be in [24, 256]");
    static_assert((N & 7) == 0, "setmaxnreg N must be a multiple of 8");
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N));
}

template <int N>
static __device__ __forceinline__ void setmaxnreg_inc() {
    static_assert(N >= 24 && N <= 256, "setmaxnreg N must be in [24, 256]");
    static_assert((N & 7) == 0, "setmaxnreg N must be a multiple of 8");
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N));
}


}  // namespace ptx
