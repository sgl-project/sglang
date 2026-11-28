// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang::ipc_a2a {

// what an expired spin_wait leaves in the timeout words (peer not recorded);
// wait_peers stores its caller's site code plus the peer (ipc_a2a_multi.py)
constexpr int kSpinWaitTimeout = 400;

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void spin_wait_kernel(
    volatile int* flag, const int* target, int* timed_out, int* peer_timed_out, unsigned long long budget_ns) {
  int t = *target;
  unsigned long long start = now_ns();
  while (*flag < t) {
    if (now_ns() - start > budget_ns) {
      // both ranks must retire the transport at the same request boundary
      *timed_out = kSpinWaitTimeout;
      *peer_timed_out = kSpinWaitTimeout;
      __threadfence_system();
      return;
    }
  }
  __threadfence_system();
}

__global__ void bump_signal_kernel(int* seq, volatile int* peer_flag) {
  int v = *seq + 1;
  *seq = v;
  __threadfence_system();
  *peer_flag = v;
}

inline void spin_wait(
    tvm::ffi::TensorView flag,
    tvm::ffi::TensorView target,
    tvm::ffi::TensorView timed_out,
    tvm::ffi::TensorView peer_timed_out,
    int64_t budget_ns) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(flag).verify(target).verify(timed_out).verify(
      peer_timed_out);
  LaunchKernel(1, 1, device.unwrap())(
      spin_wait_kernel,
      static_cast<int*>(flag.data_ptr()),
      static_cast<const int*>(target.data_ptr()),
      static_cast<int*>(timed_out.data_ptr()),
      static_cast<int*>(peer_timed_out.data_ptr()),
      static_cast<unsigned long long>(budget_ns));
}

inline void bump_signal(tvm::ffi::TensorView seq, tvm::ffi::TensorView peer_flag) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(seq).verify(peer_flag);
  LaunchKernel(1, 1, device.unwrap())(
      bump_signal_kernel, static_cast<int*>(seq.data_ptr()), static_cast<int*>(peer_flag.data_ptr()));
}

/// \brief Hold the issuing stream until every member's entry (this rank's
/// excepted) of the local int32 [world] array at address `flags` reaches
/// `target` (cyclic compare, so counters may wrap): one launch in place of a
/// stream wait per peer, and unlike a stream wait it can give up.
/// A wait that expires stores `code + member` into `timed_out` unless an
/// earlier one did; once `timed_out` is set, every wait gives up at once (the
/// request fails at its end, so later waits would only add their budgets).
__global__ void wait_peers_kernel(
    int64_t flags, int* timed_out, int world, int rank, int target, int code, unsigned long long budget_ns) {
  volatile int* expired = timed_out;
  if (*expired != 0) {
    return;
  }
  volatile int* arrived = reinterpret_cast<volatile int*>(flags);
  const unsigned long long start = now_ns();
  for (int m = 0; m < world; ++m) {
    if (m == rank) {
      continue;
    }
    while (static_cast<int>(static_cast<unsigned>(arrived[m]) - static_cast<unsigned>(target)) < 0) {
      if (*expired != 0) {
        return;
      }
      if (now_ns() - start > budget_ns) {
        atomicCAS(timed_out, 0, code + m);
        __threadfence_system();
        return;
      }
    }
  }
  __threadfence_system();
}

inline void wait_peers(
    tvm::ffi::TensorView timed_out,
    int64_t flags,
    int64_t world,
    int64_t rank,
    int64_t target,
    int64_t code,
    int64_t budget_ns) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(timed_out);
  LaunchKernel(1, 1, device.unwrap())(
      wait_peers_kernel,
      flags,
      static_cast<int*>(timed_out.data_ptr()),
      static_cast<int>(world),
      static_cast<int>(rank),
      static_cast<int>(target),
      static_cast<int>(code),
      static_cast<unsigned long long>(budget_ns));
}

}  // namespace sglang::ipc_a2a
