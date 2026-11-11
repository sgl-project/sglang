// Barriers: mbarrier, named barriers and the barrier-init fence. `sglang::device::ptx` is the one namespace for
// inline PTX; ptx/async.cuh and ptx/tcgen05.cuh add the copy and tensor-core primitives, distributed/ptx.cuh the
// collective ones.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

namespace device::ptx {

// Inline-PTX `.shared` instructions take a 32-bit byte offset in the shared
// window, not a generic 64-bit pointer.
template <typename T>
SGL_DEVICE uint32_t to_shared(T* ptr) {
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
SGL_DEVICE void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(count));
}

SGL_DEVICE uint64_t mbar_arrive(uint64_t* bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(to_shared(bar)));
  return state;
}

SGL_DEVICE uint64_t mbar_arrive_relaxed(uint64_t* bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(to_shared(bar)));
  return state;
}

// Combined arrive + set tx-count, for TMA-load completion.
SGL_DEVICE void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes));
}

// Sets the transaction count of the current phase without arriving (the arrivals come separately).
SGL_DEVICE void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes) : "memory");
}

// Wait for phase `parity` to complete. Looped because the spec allows spurious
// early wakeups. Default `.acquire` semantics mean prior `cp.async.bulk` writes
// tracked by this mbarrier are visible to later generic-proxy reads on this
// thread with no `fence.proxy.async` (spec §9.7.13.15.16 point 3).
SGL_DEVICE void mbar_wait_parity(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity));
}

// The same wait with the PTX suspendTimeHint (nanoseconds): a thread that finds the phase incomplete may be suspended
// by the hardware until it completes (or the hint elapses) instead of re-polling, which keeps the shared-memory pipe
// free for the warps that work. Default 10 ms, as CUTLASS.
SGL_DEVICE void mbar_wait_parity_suspend(uint64_t* bar, uint32_t parity, uint32_t suspend_ns = 10'000'000u) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity),
      "r"(suspend_ns)
      : "memory");
}

// Makes the mbarrier initializations visible to the async proxy (TMA, tcgen05.commit) and the cluster.
SGL_DEVICE void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// ---- named barriers (PTX ISA §9.7.13.1, bar / barrier) ----------------------
// CTA-level hardware barriers `id` 0-15 (barrier 0 is also __syncthreads). A barrier completes once `num_threads`
// threads arrived; `num_threads` must be a multiple of the warp size, and every use of one id in one phase must pass
// the same count.
//
// kAligned = true (default) is the `.aligned` form; `bar.sync` / `bar.arrive` are its aliases. It requires every
// thread of the warp to execute this same instruction: under a condition, the condition must be warp-uniform (no
// lane may skip it, take another path to a different barrier instruction, or arrive divergently), otherwise the
// behavior is undefined. Use kAligned = false (plain `barrier.sync` / `barrier.arrive`) wherever lanes of a warp may
// reach the barrier divergently.
//
// Memory ordering: when the barrier completes, the prior memory accesses of every arriving thread (sync or arrive)
// are performed relative to all participating threads; a sync additionally issues no new memory access before the
// completion. So arrive is the producer side (it does not wait) and sync the consumer side of a producer / consumer
// handoff.

template <bool kAligned = true>
SGL_DEVICE void bar_sync(uint32_t id, uint32_t num_threads) {
  if constexpr (kAligned) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
  } else {
    asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
  }
}

template <bool kAligned = true>
SGL_DEVICE void bar_arrive(uint32_t id, uint32_t num_threads) {
  if constexpr (kAligned) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
  } else {
    asm volatile("barrier.arrive %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
  }
}

}  // namespace device::ptx

}  // namespace sglang
