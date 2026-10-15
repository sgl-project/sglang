// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang::ipc_a2a {

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
      *timed_out = 1;
      *peer_timed_out = 1;
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

}  // namespace sglang::ipc_a2a
