#pragma once

// Inline PTX the distributed collectives need and no CUDA intrinsic covers:
// system-scope flag traffic, multicast (NVLS) reductions, and the 16B
// vectorized global accesses the reduce loops are built on.
//
// One header rather than a copy per kernel: before this existed the multimem
// flag increments had three identical definitions, and the 16B helpers were
// defined at `namespace sglang` scope inside custom_all_reduce.cuh, so every
// K3 kernel picked them up transitively and broke whenever that file's
// includes changed.
//
// Names mirror their PTX mnemonic (`red.relaxed.sys.global.add.u32` ->
// `red_add_relaxed_sys`) so a call site can be checked against the ISA docs
// without indirection.
//
// These join `sglang::ptx`, the namespace sgl_kernel/mbarrier.cuh opens for
// inline PTX: one vocabulary for every family, split across headers by what
// the instructions are for.

#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>
#include <type_traits>

namespace sglang {

namespace device::ptx {

// ---------------------------------------------------------------------------
// 16B vectorized global access
//
// All six take a base pointer plus a *vector* index, and require `V` to be
// exactly one 16B vector: the reduce loops force 16B accesses to keep register
// pressure down, so the width is a static contract rather than a parameter.
// ---------------------------------------------------------------------------

template <typename V>
SGL_DEVICE void ld_global_16B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
  uint4 val;
  asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_global_16B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const uint4 val = *reinterpret_cast<const uint4*>(&x);
  addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
  asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :  //
               : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}

/// Peer-visible load. Relaxed: ordering comes from the surrounding barrier or
/// from the payload's own marker (lamport polling), not from here.
///
/// Every scoped access here is `.sys`, including the ones that only ever poll
/// memory on this device: load scope measures the same at `.gpu`, so one scope
/// keeps the vocabulary small. Store scope is not interchangeable -- a store a
/// peer must observe has to be `.sys`.
template <typename V>
SGL_DEVICE void ld_relaxed_16B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
  uint4 val;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_relaxed_16B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const uint4 val = *reinterpret_cast<const uint4*>(&x);
  addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
  asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :  //
               : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}

/// One load that sums the corresponding vector across every rank in the
/// multicast team (NVLS). `mc_addr` must be a multicast VA.
template <typename V>
SGL_DEVICE void ld_multimem_16B(V& x, const void* mc_addr, int64_t vec_offset) {
#if SGL_ARCH_HOPPER_OR_GREATER
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  mc_addr = static_cast<const uint8_t*>(mc_addr) + vec_offset * 16;
  if constexpr (std::is_same_v<V, device::AlignedVector<fp32x2_t, 2>>) {
    float4 val;
    asm volatile("multimem.ld_reduce.weak.add.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(mc_addr));
    x = *reinterpret_cast<const V*>(&val);
  } else {
    // Packed f16x2/bf16x2 results live in b32 registers ("=r"); .acc::f32 only
    // raises the accumulation precision, not the result register type -- ptxas
    // rejects .f32 ("=f") destinations with "Arguments mismatch".
    uint4 val;
    if constexpr (std::is_same_v<V, device::AlignedVector<fp16x2_t, 4>>) {
      asm volatile("multimem.ld_reduce.weak.add.acc::f32.v4.f16x2 {%0, %1, %2, %3}, [%4];"
                   : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                   : "l"(mc_addr));
    } else {
      static_assert(std::is_same_v<V, device::AlignedVector<bf16x2_t, 4>>);  // 4x bf16x2
      asm volatile("multimem.ld_reduce.weak.add.acc::f32.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
                   : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                   : "l"(mc_addr));
    }
    x = *reinterpret_cast<const V*>(&val);
  }
#else
  assert(false && "multimem load is only supported on Hopper or later architecture");
#endif
}

/// 8B relaxed pair, for the narrow (2 x fp32) peer accumulators the fused
/// qk-norm exchanges. Replaces `ld.volatile` / `st.volatile`, which PTX defines
/// as relaxed at system scope.
template <typename V>
SGL_DEVICE void ld_relaxed_8B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 8 && sizeof(V) == 8);
  addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
  uint2 val;
  asm volatile("ld.relaxed.sys.global.v2.b32 {%0, %1}, [%2];" : "=r"(val.x), "=r"(val.y) : "l"(addr) : "memory");
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_relaxed_8B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 8 && sizeof(V) == 8);
  const uint2 val = *reinterpret_cast<const uint2*>(&x);
  addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
  asm volatile("st.relaxed.sys.global.v2.b32 [%2], {%0, %1};" ::"r"(val.x), "r"(val.y), "l"(addr) : "memory");
}

/// One store fanned out to every rank in the multicast team.
template <typename V>
SGL_DEVICE void st_multimem_16B(const V& x, void* mc_addr, int64_t vec_offset) {
#if SGL_ARCH_HOPPER_OR_GREATER
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto val = *reinterpret_cast<const float4*>(&x);
  mc_addr = static_cast<uint8_t*>(mc_addr) + vec_offset * 16;
  asm volatile("multimem.st.weak.v4.f32 [%4], {%0, %1, %2, %3};"
               :
               : "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w), "l"(mc_addr));
#else
  assert(false && "multimem store is only supported on Hopper or later architecture");
#endif
}

// ---------------------------------------------------------------------------
// System-scope flag traffic
//
// A flag is a u32 that a peer polls. `red` is a reduction with no result
// register, which is what a fire-and-forget increment wants; `atom` returns the
// old value for callers that need to know who arrived last.
// ---------------------------------------------------------------------------

SGL_DEVICE uint32_t load_relaxed_sys(const uint32_t* ptr) {
  uint32_t val;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(ptr) : "memory");
  return val;
}

/// Acquire load of a peer-written flag: everything the writer released before
/// its matching store is visible to this thread afterwards.
SGL_DEVICE uint32_t load_acquire_sys(const uint32_t* ptr) {
  uint32_t val;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(ptr) : "memory");
  return val;
}

/// Peer-visible flag increment. Relaxed: ordering is established by the
/// surrounding fence, not by this store.
SGL_DEVICE void red_add_relaxed_sys(uint32_t* ptr, uint32_t val) {
  asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");
}

/// Flag increment that also publishes every prior write to system scope.
SGL_DEVICE void red_add_release_sys(uint32_t* ptr, uint32_t val) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");
}

/// Publishes every prior write to system scope. Pair with a relaxed flag store
/// so a peer's acquire load of that flag also observes the payload.
SGL_DEVICE void fence_release_sys() {
  asm volatile("fence.release.sys;" ::: "memory");
}

/// Device-scope arrival counter. acq_rel so the winner of the count also
/// observes the losers' payload writes.
SGL_DEVICE uint32_t atomic_add_acq_rel_gpu(uint32_t* ptr, uint32_t val) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(val) : "memory");
  return old;
}

// ---------------------------------------------------------------------------
// Multicast flag traffic
//
// One instruction updates the flag on every rank in the team, replacing a
// world_size-wide unicast fan-out.
// ---------------------------------------------------------------------------

SGL_DEVICE void multimem_store_relaxed(uint32_t* mc_ptr, uint32_t val) {
  asm volatile("multimem.st.relaxed.sys.global.b32 [%0], %1;" : : "l"(mc_ptr), "r"(val) : "memory");
}

SGL_DEVICE void multimem_red_add_relaxed(uint32_t* mc_ptr, uint32_t val) {
#if SGL_ARCH_HOPPER_OR_GREATER
  asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], %1;" : : "l"(mc_ptr), "r"(val) : "memory");
#else
  assert(false && "multimem red is only supported on Hopper or later architecture");
#endif
}

SGL_DEVICE void multimem_red_add_release(uint32_t* mc_ptr, uint32_t val) {
#if SGL_ARCH_HOPPER_OR_GREATER
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" : : "l"(mc_ptr), "r"(val) : "memory");
#else
  assert(false && "multimem red is only supported on Hopper or later architecture");
#endif
}

}  // namespace device::ptx

}  // namespace sglang
