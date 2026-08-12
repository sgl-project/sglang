#pragma once

// System-scope PTX the Kimi K3 collectives need and no shared sglang header
// wraps. These live beside their only consumers (gemm_ar / gemm_ag) rather than
// in distributed/communicator.cuh: that header's Semaphore does not use any of
// them, so putting them there would grow a shared header for one caller's
// benefit.
//
// The `device::distributed` namespace is deliberate -- it is where the rest of
// the collective vocabulary lives, so call sites and `using` declarations read
// the same whichever header supplied the symbol.

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

namespace device::distributed {

// Peer-visible flag increment. `.sys` scope, relaxed: ordering is established by
// the surrounding fence, not by this store.
SGL_DEVICE void red_add_relaxed_sys(uint32_t* ptr, uint32_t val) {
  asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");
}

// Acquire load of a peer-written flag: everything the writer released before
// its matching store is visible to this thread afterwards.
SGL_DEVICE uint32_t load_acquire_sys(const uint32_t* ptr) {
  uint32_t val;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(ptr) : "memory");
  return val;
}

// Publishes every prior write to system scope. Pair with a relaxed flag store so
// a peer's acquire load of that flag also observes the payload.
SGL_DEVICE void fence_release_sys() {
  asm volatile("fence.release.sys;" ::: "memory");
}

// Device-scope arrival counter. acq_rel so the winner of the count also observes
// the losers' payload writes.
SGL_DEVICE uint32_t atomic_add_acq_rel_gpu(uint32_t* ptr, uint32_t val) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(val) : "memory");
  return old;
}

// One store fanned out to every rank in the multicast team.
SGL_DEVICE void multimem_store_relaxed(uint32_t* ptr, uint32_t val) {
  asm volatile("multimem.st.relaxed.sys.global.b32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");
}

SGL_DEVICE void multimem_red_add_relaxed(uint32_t* mc_flag) {
#if SGL_ARCH_HOPPER_OR_GREATER
  asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], 1;" ::"l"(mc_flag) : "memory");
#else
  assert(false && "multimem red is only supported on Hopper or later architecture");
#endif
}

SGL_DEVICE void multimem_red_add_release(uint32_t* mc_flag) {
#if SGL_ARCH_HOPPER_OR_GREATER
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], 1;" ::"l"(mc_flag) : "memory");
#else
  assert(false && "multimem red is only supported on Hopper or later architecture");
#endif
}

}  // namespace device::distributed

}  // namespace sglang
