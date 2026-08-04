#pragma once

// mbarrier PTX wrappers shared by the Kimi K3 kernels that drive TMA by hand:
// kimi_k3/comm/gemm_ar.cuh and kimi_k3/attn_res/fused_tma.cuh both defined these
// with identical bodies.
//
// The enclosing namespace is the same global `ptx` both files already open, so
// existing `::ptx::mbar_*` call sites need no change.
//
// gemm_ar.cuh keeps mbar_arrive_cluster_release: only it uses that one.
// attention/kda_prefill.cu duplicates a different set (MMA / ldmatrix) but is
// built without a sglang include path and cannot consume this header.

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace ptx {

// Inline-PTX `.shared` instructions take a 32-bit byte offset in the shared
// window, not a generic 64-bit pointer.
template <typename T>
static SGL_DEVICE uint32_t to_shared(T* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// ---- mbarrier (PTX ISA §9.7.13.15) -----------------------------------------
//
// Only the `try_wait.parity` waiter is wrapped: state-token waits couple
// arriver and waiter, and mixing state- and parity-tracked codepaths in one
// kernel is a deadlock risk. The caller owns the phase counter and flips it at
// the stage wrap (`phase ^= (stage == 0)`).
//
// Initial parity, the easy-to-flip part: after `mbar_init` the bar is at
// parity 0, and each full cycle (count arrivals -> fire -> reset) flips it.
//   - consumer-first (waits for an external producer's first signal) -> 0.
//   - producer-first (waits for a consumer to release a slot that no consumer
//     has touched yet) -> 1, so the first wait is a no-op skip.
// A consumer-first wait initialized to 1 skips the producer's first signal and
// blocks forever on the second.
static SGL_DEVICE void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(count));
}

static SGL_DEVICE uint64_t mbar_arrive(uint64_t* bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(to_shared(bar)));
  return state;
}

// Combined arrive + set tx-count, for TMA-load completion.
static SGL_DEVICE void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes));
}

// Wait for phase `parity` to complete. Looped because the spec allows spurious
// early wakeups. Default `.acquire` semantics mean prior `cp.async.bulk` writes
// tracked by this mbarrier are visible to later generic-proxy reads on this
// thread with no `fence.proxy.async` (spec §9.7.13.15.16 point 3).
static SGL_DEVICE void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity));
}

}  // namespace ptx
