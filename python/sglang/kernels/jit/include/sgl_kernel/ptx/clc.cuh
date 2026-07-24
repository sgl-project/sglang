/// Extracted reusable utilities from kimi_k3/comm/gemm_ar.cuh.

#pragma once

#include <sgl_kernel/ptx/addr.cuh>
#include <sgl_kernel/ptx/mbarrier.cuh>

#include <cstdint>

// ================= common/ptx/clc.cuh =================
// Cluster Launch Control (CLC) wrappers. PTX ISA 9.2 §9.7.13.17 / §9.7.13.18.
// sm_100+ (verified on sm_103a / B300).
//
// What CLC is. A hardware-side persistent-grid scheduler. The host launches
// a grid containing one CTA (or one cluster) per output tile (a "non-
// persistent" launch shape), and instead of a software persistent-loop
// (`for (slot = bid; slot < total; slot += grid)`), each worker queries
// `clusterlaunchcontrol.try_cancel` to atomically claim the next tile that
// hasn't started yet. The HW scheduler hands back a slot-id (broadcast to
// all CTAs in the cluster, hence "cluster launch control"), or a "decline"
// signal once the grid is exhausted.
//
// Why use it. Two scenarios. (a) Wave-quantization absorption: when the
// trailing wave has fewer tiles than there are SMs, idle SMs at the tail
// can steal late-finishing SMs' work without waiting for the tail wave to
// finish. (b) Runtime-variance absorption: if some tiles take longer than
// others (e.g., ragged-K or per-tile branch divergence), CLC lets fast
// SMs work-steal slow SMs' queued tiles. For our 8K^3 GEMM kernels with
// uniform tile cost and 64×64 = 4096 tiles ≫ 148 SMs, the win regime is
// thin — see `recipes/clc_vs_persistent/` for the empirical picking
// table.
//
// Why NOT use it. (a) On long-running uniform kernels with many tiles
// per SM, a software persistent loop already amortizes dispatch cost,
// and CLC's per-tile cancel-call overhead can dominate. (b) The launch
// shape changes (grid dim = total tiles, not min(total, NUM_SMS)), which
// means launch overhead is paid up-front for all tiles rather than just
// the resident wave. For very small grids (low total tile count) this
// can flip the verdict.
//
// Usage shape:
//   __shared__ __align__(16) uint8_t  clc_response[16];
//   __shared__ __align__(8)  uint64_t clc_mbar;
//   ptx::mbar_init(&clc_mbar, /*count=*/CLUSTER_SIZE);
//   ...
//   if (warp_id == ISSUER_WARP && ptx::elect_one()) {
//       ptx::mbar_arrive_expect_tx(&clc_mbar, 16);
//       ptx::clc_try_cancel(&clc_response, &clc_mbar);
//   }
//   ptx::mbar_wait_parity(&clc_mbar, parity);
//   ptx::ClcResponse r = ptx::clc_query_cancel(&clc_response);
//   if (!r.canceled) break;             // grid exhausted
//   int m = r.ctaid_x; int n = r.ctaid_y; int batch = r.ctaid_z;
//
// Key conventions for THIS wrapper (matches `common/`'s rules):
//   - response buffer is `uint8_t[16]` (or `uint128_t`), 16 B aligned;
//     mbar is `uint64_t`, 8 B aligned. Caller declares both as `__shared__`.
//   - `clc_try_cancel(response, mbar)` takes typed pointers, calls
//     `to_shared` internally for both. Must be issued by exactly one
//     thread per cluster (use `elect_one`).
//   - `clc_query_cancel(response)` parses the b128 response into a small
//     POD struct. Returns canceled bit + 3 ctaid components. Read-only;
//     no mbar.
//   - The mbar must be `mbar_arrive_expect_tx`'d for 16 bytes BEFORE the
//     cancel issue; the HW signals completion via `mbarrier::complete_tx`
//     so the wait succeeds once the response is written. Skipping the
//     expect_tx is the most common deadlock — the response writes 16B
//     of tx but no thread arrives, so the mbar never completes.
//
// Per the spec:
//   - `try_cancel.async.shared::cta.mbarrier::complete_tx::bytes
//      .multicast::cluster::all.b128 [response], [mbar];`
//      issues the cancel; result is broadcast to ALL CTAs in the
//      cluster's matching smem address (.multicast::cluster::all).
//   - `query_cancel.is_canceled.pred.b128 p, response_reg;`
//      true iff the response represents a successful cancel (i.e. a
//      slot was claimed). Inverse means the grid is exhausted; the
//      caller must exit the work loop.
//   - `query_cancel.get_first_ctaid.v4.b32.b128 {x,y,z,_}, response_reg;`
//      extracts the (x,y,z) of the FIRST CTA in the canceled cluster
//      (so for a cluster shape (M,N) the cluster occupies tiles
//      [x..x+M-1] × [y..y+N-1]). For a 1×1 cluster this is just the
//      claimed CTA's coords. The fourth lane (`_`) is reserved.
//
// What we DON'T wrap. The PTX spec exposes a synchronous form
// (`clusterlaunchcontrol.try_cancel.b128`, no `.async`/`.mbarrier`)
// that returns the response directly into registers, but it stalls
// the issuing warp until the HW scheduler responds. The async form
// hides that latency behind the mbar wait while the rest of the warp
// (and all other warps) keep running. Only the async form is wrapped
// here. If you ever need the synchronous form, the inline-asm shape is:
//     clusterlaunchcontrol.try_cancel.b128 {%0_lo, %0_hi}, [smem_b128_buf];
// followed by the same `query_cancel.*` ops on the register pair.


namespace ptx {

// Parsed CLC response. `canceled == true` means the issuing cluster claimed
// a slot at (ctaid_x, ctaid_y, ctaid_z). `canceled == false` means the
// grid is exhausted; ctaid_* are unspecified.
//
// "first ctaid" = the (x,y,z) blockIdx of the FIRST CTA in the claimed
// cluster's bounding box, in cluster-grid space. For cluster shape
// (cm, cn, ck), the cluster occupies CTAs at [x..x+cm-1] × [y..y+cn-1]
// × [z..z+ck-1].
struct ClcResponse {
    uint32_t ctaid_x;
    uint32_t ctaid_y;
    uint32_t ctaid_z;
    bool     canceled;
};

// Per-CTA smem block holding the CLC scheduler's HW-response buffer, the
// completion mbar, and a 2-slot ping-pong queue of decoded slot indices.
//
// Layout is dictated by `clusterlaunchcontrol.try_cancel.async.shared::cta.
// mbarrier::complete_tx::bytes.b128`: a 16 B response buffer 16-B-aligned
// + an 8-B-aligned mbarrier. The 2-slot queue lets warp 0 (TMA issuer) and
// the consumer warps (warp 1 MMA + warp 4-7 epi) operate on tiles
// `t` and `t+1` in parallel — warp 0 writes the next slot at index
// `(t+1) & 1` while consumers read the current slot at `t & 1`. The
// queue is per-CTA (NOT cluster-shared) — even in the 2-CTA cluster
// kernels, each CTA's warp 0 maintains its own copy via the multicast
// CLC response (cite recipes/gemm_epi/transpose_picking/fused_gemm_2cta/kernel.cu — both peers
// declare the same struct in their own dynamic smem).
//
// Consumers (rule-of-three trigger; cite the audit T3 in the
// SESSION_HANDOFF):
//   - kernels/gemm/dense_1cta/kernel.cu  (bf16_path namespace)
//   - recipes/gemm_epi/transpose_picking/fused_gemm_2cta/kernel.cu (top-level)
//   - kernels/fused_gemm_grouped_dense/kernel.cu (bf16_path namespace)
//
// CLC_SLOT_END is the sentinel that means "grid exhausted; stop the work
// loop". `clc_try_cancel`'s response sets `canceled=false`; the kernel
// translates `!r.canceled` into `slot = CLC_SLOT_END` before writing
// to the queue, so downstream consumers can compare a single sentinel
// rather than re-querying the response.
struct ClcSmem {
    alignas(16) uint8_t  response[16];
    alignas(8)  uint64_t mbar;
    alignas(8)  uint32_t slot_q[2];
};
constexpr uint32_t CLC_SLOT_END = 0xFFFFFFFFu;

// Issue the async cancel — NON-MULTICAST form for kernels launched WITHOUT
// thread-block clusters (1×1×1 cluster shape, the default). Each CTA gets
// an independent response; only ONE thread in the CTA must call this.
//
//  - `response`: 16 B aligned shared buffer; the HW writes the b128
//                response here. Must be `__shared__ __align__(16)`.
//  - `mbar`:     8 B aligned shared mbarrier. Caller must
//                `mbar_arrive_expect_tx(mbar, 16)` AFTER (or before — the
//                ordering is robust under the scope=cta/sem=acquire wait)
//                this call; the HW signals completion via tx-count.
//
// The async form returns immediately; the response lands in smem when
// the HW scheduler responds. Wait via mbar_wait_parity on the same mbar.
// Pair with `mbar_arrive_expect_tx(mbar, 16)` so the mbar's tx-count
// matches the 16 B the HW writes.
static __device__ __forceinline__ void clc_try_cancel(
        void* response, uint64_t* mbar) {
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::"
        "complete_tx::bytes.b128 [%0], [%1];"
        :: "r"(to_shared(response)), "r"(to_shared(mbar))
        : "memory");
}


// Parse the b128 response from smem. After the mbar wait returns the
// HW response is sitting in `response`; this loads it once and decodes
// both the canceled-pred and the ctaid_* coords in one asm block (the
// `@p1` predicate guards the get_first_ctaid so we don't read garbage
// when canceled=false).
//
// Read-only; safe to call on every thread that needs the result. The
// `ld.shared.b128` lands in a temporary, so concurrent calls don't race.
static __device__ __forceinline__ ClcResponse clc_query_cancel(
        const void* response) {
    ClcResponse r{};
    uint32_t valid = 0;
    // The `fence.proxy.async.shared::cta` AFTER get_first_ctaid (inside the
    // predicated success branch) is REQUIRED: without it, the response read
    // can race with the next iteration's try_cancel.multicast — manifesting
    // as a hang at high CLC issue cadence (small num_iters + large grid),
    // caught by our gate. The PTX spec §9.7.13.17 example documents a stronger
    // form (`fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster`);
    // the weaker `.shared::cta` form here is the minimal one that passes
    // empirically.
    asm volatile(
        "{\n\t"
        ".reg .pred p1;\n\t"
        ".reg .b128 clc_result;\n\t"
        "ld.shared.b128 clc_result, [%4];\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"
        "selp.u32 %3, 1, 0, p1;\n\t"
        "@!p1 bra.uni DONE_%=;\n\t"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
            "{%0, %1, %2, _}, clc_result;\n\t"
        "fence.proxy.async.shared::cta;\n\t"
        "DONE_%=:\n\t"
        "}\n"
        : "=r"(r.ctaid_x), "=r"(r.ctaid_y), "=r"(r.ctaid_z), "=r"(valid)
        : "r"(to_shared(response))
        : "memory");
    r.canceled = (valid != 0);
    return r;
}

}  // namespace ptx
