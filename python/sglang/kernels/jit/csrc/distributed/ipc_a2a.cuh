// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/array.h>

#include <cstdint>
#include <cuda.h>

namespace sglang::ipc_a2a {

inline void check_driver(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    host::RuntimeCheck(false, "Peer copy submission failed: ", name ? name : "unknown");
  }
}

// The caller retains every mapping until the consumer stream has completed.
// Batch host submission only: payloads use CE and waits remain stream ordered.
inline void copy_peer_strided(
    const tvm::ffi::Array<int64_t>& signals,
    const tvm::ffi::Array<int64_t>& copies,
    int64_t stream_ptr,
    int64_t generation) {
  host::RuntimeCheck(signals.size() % 4 == 0 && copies.size() % 6 == 0, "Invalid peer copy commands");
  auto stream = reinterpret_cast<CUstream>(stream_ptr);
  const auto value = static_cast<uint32_t>(generation);
  // Each peer supplies credit-write, credit-wait, arrival-write, arrival-wait.
  for (size_t i = 0; i < signals.size(); i += 4) {
    check_driver(cuStreamWriteValue32(stream, signals[i], value, 0));
  }
  for (size_t i = 0; i < signals.size(); i += 4) {
    check_driver(cuStreamWaitValue32(stream, signals[i + 1], value, 0));
  }
  for (size_t i = 0; i < copies.size(); i += 6) {
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = copies[i];
    copy.dstDevice = copies[i + 1];
    copy.WidthInBytes = copies[i + 2];
    copy.Height = copies[i + 3];
    copy.srcPitch = copies[i + 4];
    copy.dstPitch = copies[i + 5];
    check_driver(cuMemcpy2DAsync(&copy, stream));
  }
  for (size_t i = 0; i < signals.size(); i += 4) {
    check_driver(cuStreamWriteValue32(stream, signals[i + 2], value, 0));
  }
  for (size_t i = 0; i < signals.size(); i += 4) {
    check_driver(cuStreamWaitValue32(stream, signals[i + 3], value, 0));
  }
}

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
