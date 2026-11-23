#pragma once

// Inline PTX the distributed collectives need and no CUDA intrinsic covers:
// system-scope flag traffic, multicast (NVLS) reductions, the 16B vectorized
// global accesses the reduce loops are built on, and self-tagged 64-bit
// payloads.
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
// These join `sglang::ptx`, the namespace sgl_kernel/ptx/*.cuh open for
// inline PTX: one vocabulary for every family, split across headers by what
// the instructions are for.

#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace sglang {

namespace device::ptx {

#ifndef USE_ROCM

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
    asm volatile("multimem.ld_reduce.weak.global.add.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(mc_addr));
    x = *reinterpret_cast<const V*>(&val);
  } else {
    // Packed f16x2/bf16x2 results live in b32 registers ("=r"); .acc::f32 only
    // raises the accumulation precision, not the result register type -- ptxas
    // rejects .f32 ("=f") destinations with "Arguments mismatch".
    uint4 val;
    if constexpr (std::is_same_v<V, device::AlignedVector<fp16x2_t, 4>>) {
      asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.v4.f16x2 {%0, %1, %2, %3}, [%4];"
                   : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                   : "l"(mc_addr));
    } else {
      static_assert(std::is_same_v<V, device::AlignedVector<bf16x2_t, 4>>);  // 4x bf16x2
      asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
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
  asm volatile("multimem.st.weak.global.v4.f32 [%4], {%0, %1, %2, %3};"
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

/// Rank-local counter increment; nothing on another GPU reads it.
SGL_DEVICE void red_add_relaxed_gpu(uint32_t* ptr, uint32_t val) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");
}

/// 64-bit form of `red_add_relaxed_gpu`.
SGL_DEVICE void red_add_relaxed_gpu(uint64_t* ptr, uint64_t val) {
  asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");
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

/// Owner-side take of a peer-written flag, returning the old value.
SGL_DEVICE uint32_t atomic_add_relaxed_sys(uint32_t* ptr, uint32_t val) {
  uint32_t old;
  asm volatile("atom.relaxed.sys.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(val) : "memory");
  return old;
}

/// Same, and acquires the writes peers released before their increments.
SGL_DEVICE uint32_t atomic_add_acquire_sys(uint32_t* ptr, uint32_t val) {
  uint32_t old;
  asm volatile("atom.acquire.sys.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(val) : "memory");
  return old;
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

// ---------------------------------------------------------------------------
// Self-tagged 64-bit payloads
//
// An entry that carries its own epoch tag next to its payload needs no flag: a
// relaxed 8-byte access is single-copy atomic, so a reader that sees the
// current tag sees the payload written with it.
// ---------------------------------------------------------------------------

SGL_DEVICE uint64_t load_relaxed_sys(const uint64_t* ptr) {
  uint64_t val;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(val) : "l"(ptr) : "memory");
  return val;
}

SGL_DEVICE void store_relaxed_sys(uint64_t* ptr, uint64_t val) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");
}

SGL_DEVICE void multimem_store_relaxed(uint64_t* mc_ptr, uint64_t val) {
  asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" : : "l"(mc_ptr), "l"(val) : "memory");
}

#else  // USE_ROCM

// The same vocabulary on AMD GPUs. `.sys` maps to the system memory scope (the
// "" sync scope), which on gfx942/gfx950 sets `sc0 sc1`. The one exception is
// `ld_relaxed_16B`, which only ever polls this rank's own workspace: that is an
// uncached allocation (see hip_vmm_utils.py), which the owning GPU still caches
// in L2 with peer writes over XGMI invalidating the line by probe, so an agent
// scope load (skip only the L1) sees them without going to HBM on every spin.

namespace details {

using u32x4_t = uint32_t __attribute__((ext_vector_type(4)));
using global_u32x4_t = __attribute__((address_space(1))) u32x4_t;

}  // namespace details

template <typename V>
SGL_DEVICE void ld_global_16B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  x = static_cast<const V*>(addr)[vec_offset];
}

template <typename V>
SGL_DEVICE void st_global_16B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  static_cast<V*>(addr)[vec_offset] = x;
}

/// The signal fence keeps a polling loop from hoisting the load: the builtin
/// alone is not treated as a volatile access.
template <typename V>
SGL_DEVICE void ld_relaxed_16B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto ptr = static_cast<const details::u32x4_t*>(addr) + vec_offset;
  details::u32x4_t val;
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_global_load_b128)
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  val = __builtin_amdgcn_global_load_b128((details::global_u32x4_t*)ptr, "agent");
#else
  val = *reinterpret_cast<const volatile details::u32x4_t*>(ptr);
#endif
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_relaxed_16B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto ptr = static_cast<details::u32x4_t*>(addr) + vec_offset;
  const auto val = *reinterpret_cast<const details::u32x4_t*>(&x);
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_global_store_b128)
  __builtin_amdgcn_global_store_b128((details::global_u32x4_t*)ptr, val, "");
#else
  *reinterpret_cast<volatile details::u32x4_t*>(ptr) = val;
#endif
}

/// Eight `st_relaxed_16B`s to `base` + kOffset + i * kStep bytes, for long
/// runs of stores into a peer's buffer: the builtin form makes the compiler
/// wait (s_wait_xcnt) after every store on gfx1250.
/// One asm statement owns all eight operands until its own trailing wait has
/// seen them read, so the compiler cannot reuse a register too early.
template <int32_t kOffset, int32_t kStep, typename V>
SGL_DEVICE void st_relaxed_x8_16B(const V* x, void* base) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto v = reinterpret_cast<const details::u32x4_t*>(x);
#if defined(__gfx1250__)
  asm volatile(
      "global_store_b128 %0, %1, off offset:%9 scope:SCOPE_SYS\n\t"
      "global_store_b128 %0, %2, off offset:%10 scope:SCOPE_SYS\n\t"
      "global_store_b128 %0, %3, off offset:%11 scope:SCOPE_SYS\n\t"
      "global_store_b128 %0, %4, off offset:%12 scope:SCOPE_SYS\n\t"
      "global_store_b128 %0, %5, off offset:%13 scope:SCOPE_SYS\n\t"
      "global_store_b128 %0, %6, off offset:%14 scope:SCOPE_SYS\n\t"
      "global_store_b128 %0, %7, off offset:%15 scope:SCOPE_SYS\n\t"
      "global_store_b128 %0, %8, off offset:%16 scope:SCOPE_SYS\n\t"
      "s_wait_xcnt 0x0"
      :
      : "v"(base),
        "v"(v[0]),
        "v"(v[1]),
        "v"(v[2]),
        "v"(v[3]),
        "v"(v[4]),
        "v"(v[5]),
        "v"(v[6]),
        "v"(v[7]),
        "i"(kOffset),
        "i"(kOffset + kStep),
        "i"(kOffset + 2 * kStep),
        "i"(kOffset + 3 * kStep),
        "i"(kOffset + 4 * kStep),
        "i"(kOffset + 5 * kStep),
        "i"(kOffset + 6 * kStep),
        "i"(kOffset + 7 * kStep)
      : "memory");
#else
#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    st_relaxed_16B(x[i], static_cast<uint8_t*>(base) + kOffset + i * kStep, 0);
  }
#endif
}

/// Four-store form of `st_relaxed_x8_16B`.
template <int32_t kOffset, int32_t kStep, typename V>
SGL_DEVICE void st_relaxed_x4_16B(const V* x, void* base) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto v = reinterpret_cast<const details::u32x4_t*>(x);
#if defined(__gfx1250__)
  asm volatile(
      "global_store_b128 %0, %1, off offset:%5 scope:SCOPE_SYS\n\t"
      "global_store_b128 %0, %2, off offset:%6 scope:SCOPE_SYS\n\t"
      "global_store_b128 %0, %3, off offset:%7 scope:SCOPE_SYS\n\t"
      "global_store_b128 %0, %4, off offset:%8 scope:SCOPE_SYS\n\t"
      "s_wait_xcnt 0x0"
      :
      : "v"(base),
        "v"(v[0]),
        "v"(v[1]),
        "v"(v[2]),
        "v"(v[3]),
        "i"(kOffset),
        "i"(kOffset + kStep),
        "i"(kOffset + 2 * kStep),
        "i"(kOffset + 3 * kStep)
      : "memory");
#else
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    st_relaxed_16B(x[i], static_cast<uint8_t*>(base) + kOffset + i * kStep, 0);
  }
#endif
}

namespace details {

template <uint32_t kStride, typename V, uint32_t... G>
SGL_DEVICE void st_relaxed_x8_run(const V* x, void* base, std::integer_sequence<uint32_t, G...>) {
  constexpr auto kStep = static_cast<int32_t>(kStride * sizeof(V));
  (st_relaxed_x8_16B<static_cast<int32_t>(G * 8) * kStep, kStep>(x + G * 8, base), ...);
}

}  // namespace details

/// `x[u]` to `base` + u * kStride vectors for u < kCount, as system-scope
/// stores: eight per `st_relaxed_x8_16B`, then four per `st_relaxed_x4_16B`.
template <uint32_t kStride, uint32_t kCount, typename V>
SGL_DEVICE void st_relaxed_run_16B(const V* x, void* base) {
  static_assert(kCount % 4 == 0);
  constexpr auto kStep = static_cast<int32_t>(kStride * sizeof(V));
  constexpr uint32_t kRuns = kCount / 8;
  details::st_relaxed_x8_run<kStride>(x, base, std::make_integer_sequence<uint32_t, kRuns>{});
  if constexpr (kCount % 8 != 0) {
    st_relaxed_x4_16B<static_cast<int32_t>(kRuns * 8) * kStep, kStep>(x + kRuns * 8, base);
  }
}

template <typename V>
SGL_DEVICE void ld_multimem_16B(V&, const void*, int64_t) {
  assert(false && "multimem load is not supported on ROCm");
}

template <typename V>
SGL_DEVICE void ld_relaxed_8B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 8 && sizeof(V) == 8);
  const auto ptr = const_cast<uint64_t*>(static_cast<const uint64_t*>(addr) + vec_offset);
  const uint64_t val = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_relaxed_8B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 8 && sizeof(V) == 8);
  const auto ptr = static_cast<uint64_t*>(addr) + vec_offset;
  __hip_atomic_store(ptr, *reinterpret_cast<const uint64_t*>(&x), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename V>
SGL_DEVICE void st_multimem_16B(const V&, void*, int64_t) {
  assert(false && "multimem store is not supported on ROCm");
}

SGL_DEVICE uint32_t load_relaxed_sys(const uint32_t* ptr) {
  return __hip_atomic_load(const_cast<uint32_t*>(ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

SGL_DEVICE uint32_t load_acquire_sys(const uint32_t* ptr) {
  return __hip_atomic_load(const_cast<uint32_t*>(ptr), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

SGL_DEVICE void red_add_relaxed_sys(uint32_t* ptr, uint32_t val) {
  __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

SGL_DEVICE void red_add_relaxed_gpu(uint32_t* ptr, uint32_t val) {
  __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

SGL_DEVICE void red_add_relaxed_gpu(uint64_t* ptr, uint64_t val) {
  __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

SGL_DEVICE uint32_t atomic_add_relaxed_sys(uint32_t* ptr, uint32_t val) {
  return __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

SGL_DEVICE uint32_t atomic_add_acquire_sys(uint32_t* ptr, uint32_t val) {
  return __hip_atomic_fetch_add(ptr, val, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

SGL_DEVICE void red_add_release_sys(uint32_t* ptr, uint32_t val) {
  __hip_atomic_fetch_add(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

SGL_DEVICE void fence_release_sys() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}

SGL_DEVICE uint32_t atomic_add_acq_rel_gpu(uint32_t* ptr, uint32_t val) {
  return __hip_atomic_fetch_add(ptr, val, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}

SGL_DEVICE void multimem_store_relaxed(uint32_t*, uint32_t) {
  assert(false && "multimem store is not supported on ROCm");
}

SGL_DEVICE void multimem_red_add_relaxed(uint32_t*, uint32_t) {
  assert(false && "multimem red is not supported on ROCm");
}

SGL_DEVICE void multimem_red_add_release(uint32_t*, uint32_t) {
  assert(false && "multimem red is not supported on ROCm");
}

SGL_DEVICE uint64_t load_relaxed_sys(const uint64_t* ptr) {
  return __hip_atomic_load(const_cast<uint64_t*>(ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

SGL_DEVICE void store_relaxed_sys(uint64_t* ptr, uint64_t val) {
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

SGL_DEVICE void multimem_store_relaxed(uint64_t*, uint64_t) {
  assert(false && "multimem store is not supported on ROCm");
}

#endif  // USE_ROCM

}  // namespace device::ptx

}  // namespace sglang
