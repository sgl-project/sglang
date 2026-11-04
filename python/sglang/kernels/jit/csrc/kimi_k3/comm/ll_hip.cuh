// LL (low-latency) push protocol for Kimi-K3's TP8 gathers on ROCm (wave64).
//
// A sender writes each 8-byte payload into the receiver's scratch as a 16-byte line
// {data0, flag, data1, flag} in one transaction; the receiver polls its own scratch until both
// flags of its lines equal the epoch flag. No barrier: a line is rewritten two epochs later, by
// which point its receiver has consumed it (the writer has since received the receiver's
// next-epoch data). Scratch lives in uncached device memory exchanged over IPC (the meta buffer
// tail of a dedicated aiter CustomAllreduce).

#pragma once

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace sglang {
namespace k3_ll {

using llx_v4u = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;

// One 16-byte transaction each way, so a line's payload and flag land together.
// The asm barriers keep the compiler from caching a polled line in registers or
// moving it across the loop (the loads are not volatile).
SGL_DEVICE void ll_store(uint4* dst, uint32_t d0, uint32_t d1, uint32_t flag) {
  llx_v4u v = {d0, flag, d1, flag};
  __builtin_nontemporal_store(v, reinterpret_cast<llx_v4u*>(dst));
  asm volatile("" ::: "memory");
}

SGL_DEVICE uint4 ll_load(const uint4* src) {
  asm volatile("" ::: "memory");
  const llx_v4u v = __builtin_nontemporal_load(reinterpret_cast<const llx_v4u*>(src));
  return make_uint4(v[0], v[1], v[2], v[3]);
}

// Vector v of a row of n vectors travels as lines v and n + v, so each store or load
// instruction of a wave covers one contiguous span (strided 16-byte remote stores are slow).
SGL_DEVICE void ll_push(uint4* row, int v, int n, const uint4& x, uint32_t flag) {
  ll_store(row + v, x.x, x.y, flag);
  ll_store(row + n + v, x.z, x.w, flag);
}

SGL_DEVICE bool ll_ready(const uint4& a, const uint4& b, uint32_t flag) {
  return a.y == flag && a.w == flag && b.y == flag && b.w == flag;
}

SGL_DEVICE uint4 ll_payload(const uint4& a, const uint4& b) {
  return make_uint4(a.x, a.z, b.x, b.z);
}

SGL_DEVICE uint32_t next_flag(const int32_t* epochs, int slot) {
  const uint32_t flag = static_cast<uint32_t>(epochs[slot]) + 1u;
  return flag == 0u ? 1u : flag;  // 0 is the zeroed scratch
}

// Workgroup barrier for LDS only. __syncthreads() also waits out the wave's vmcnt, i.e.
// the acks of its remote LL stores, which take microseconds.
SGL_DEVICE void lds_barrier() {
  asm volatile("" ::: "memory");
  __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0), other counters untouched
  __builtin_amdgcn_s_barrier();
  asm volatile("" ::: "memory");
}

}  // namespace k3_ll
}  // namespace sglang
