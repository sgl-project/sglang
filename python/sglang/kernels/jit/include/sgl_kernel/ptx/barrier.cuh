// Barriers: mbarrier, named barriers, cluster barriers and the barrier-init fence.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/ptx/utils.cuh>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace device::ptx {

// Inline-PTX `.shared` instructions take a 32-bit byte offset in the shared window, not a generic 64-bit pointer.
template <SmemAddress T>
SGL_DEVICE uint32_t to_shared(T addr) {
  if constexpr (std::is_pointer_v<T>) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  } else {
    return static_cast<uint32_t>(addr);
  }
}

// This CTA's rank in its cluster (0 below Hopper, where every CTA is its own cluster).
SGL_DEVICE uint32_t cluster_ctarank() {
#if SGL_ARCH_HOPPER_OR_GREATER
  uint32_t rank;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
#else
  return 0;
#endif
}

// The number of CTAs in this cluster (1 below Hopper).
SGL_DEVICE uint32_t cluster_nctarank() {
#if SGL_ARCH_HOPPER_OR_GREATER
  uint32_t num;
  asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(num));
  return num;
#else
  return 1;
#endif
}

// The shared::cluster address of the same smem location in the CTA of cluster rank `rank`.
template <SmemAddress Addr>
SGL_DEVICE uint32_t mapa_shared(Addr addr, uint32_t rank) {
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(result) : "r"(to_shared(addr)), "r"(rank));
  return result;
}

// ---- mbarrier (PTX ISA 9.7.13.15) ------------------------------------------
// Only parity waits are wrapped; the caller owns the phase bit. A fresh mbarrier waits on parity 0:
// a consumer waiting for the producer's first signal starts at 0, a producer waiting on a never-used slot at 1.
template <MBarAddress Bar>
SGL_DEVICE void mbar_init(Bar bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(count) : "memory");
}

template <MBarAddress Bar>
SGL_DEVICE uint64_t mbar_arrive(Bar bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(to_shared(bar)) : "memory");
  return state;
}

template <MBarAddress Bar>
SGL_DEVICE uint64_t mbar_arrive_relaxed(Bar bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(to_shared(bar)) : "memory");
  return state;
}

// Combined arrive + set tx-count, for TMA-load completion.
template <MBarAddress Bar>
SGL_DEVICE void mbar_arrive_expect_tx(Bar bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes) : "memory");
}

// Sets the transaction count of the current phase without arriving (the arrivals come separately).
template <MBarAddress Bar>
SGL_DEVICE void mbar_expect_tx(Bar bar, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(to_shared(bar)), "r"(bytes) : "memory");
}

// The default `.acquire.cta` makes the `cp.async.bulk` writes tracked by this mbarrier visible to later generic
// reads, with no `fence.proxy.async`.
template <MBarAddress Bar>
SGL_DEVICE void mbar_wait_parity(Bar bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity)
      : "memory");
}

// With a suspendTimeHint (ns): the thread may sleep until the phase completes instead of re-polling smem.
// Default 10 ms, as CUTLASS.
template <MBarAddress Bar>
SGL_DEVICE void mbar_wait_parity_suspend(Bar bar, uint32_t parity, uint32_t suspend_ns = 10'000'000u) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity),
      "r"(suspend_ns)
      : "memory");
}

// Cluster-scope acquire: needed when peer CTAs deliver the data (st.async, remote release arrives).
template <MBarAddress Bar>
SGL_DEVICE void mbar_wait_parity_cluster(Bar bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity)
      : "memory");
}

// Makes the mbarrier initializations visible to the async proxy (TMA, tcgen05.commit) and the cluster.
// Usually pairs with `bar_cluster_arrive_relaxed` + `bar_cluster_wait_acquire`.
SGL_DEVICE void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// ---- named barriers (PTX ISA 9.7.13.1) -------------------------------------
// `id` 0-15 (0 is __syncthreads); `num_threads` is a multiple of 32 and the same for every use of an id in a phase.
// kAligned: the whole warp must execute the same instruction (warp-uniform condition), otherwise UB;
// use kAligned = false where lanes may reach it divergently. Arrive is the producer side, sync the consumer side.

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

// ---- cluster barrier (barrier.cluster), kAligned as above ------------------

template <bool kAligned = true>
SGL_DEVICE void bar_cluster_arrive_relaxed() {
  if constexpr (kAligned) {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  } else {
    asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
  }
}

template <bool kAligned = true>
SGL_DEVICE void bar_cluster_arrive_release() {
  if constexpr (kAligned) {
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  } else {
    asm volatile("barrier.cluster.arrive.release;" ::: "memory");
  }
}

// `barrier.cluster.wait` has only the acquire form.
template <bool kAligned = true>
SGL_DEVICE void bar_cluster_wait_acquire() {
  if constexpr (kAligned) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
    asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
  }
}

// Same as `cg::this_cluster().sync()`; kRelaxed drops the release (e.g. after `fence_barrier_init`).
template <bool kAligned = true, bool kRelaxed = false>
SGL_DEVICE void bar_cluster_sync() {
  if constexpr (kRelaxed) {
    bar_cluster_arrive_relaxed<kAligned>();
  } else {
    bar_cluster_arrive_release<kAligned>();
  }
  bar_cluster_wait_acquire<kAligned>();
}

}  // namespace device::ptx

}  // namespace sglang
