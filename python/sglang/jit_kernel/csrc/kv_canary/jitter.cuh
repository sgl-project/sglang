// CUDA spin-wait kernel for the kv-canary timing-jitter fuzzer.
//
// Single-thread single-block: occupies one warp lane on one SM for a controlled
// number of device cycles so it can be captured into a cuda graph alongside the
// canary HEAD/TAIL launches. The cycle count is read from a device int64 tensor
// every replay, satisfying the same static-buffer protocol as canary plan input.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace canary {

namespace {

__global__ void spin_wait_kernel(const int64_t* __restrict__ cycles_ptr) {
  if (threadIdx.x != 0) {
    return;
  }
  const int64_t target = *cycles_ptr;
  if (target <= 0) {
    return;
  }
  const int64_t start = clock64();
  // volatile sink defeats nvcc dead-store elimination; clock64() is treated as
  // pure by the optimiser otherwise.
  volatile int64_t sink = 0;
  int64_t now = clock64();
  while ((now - start) < target) {
    sink = now;
    now = clock64();
  }
  (void)sink;
}

}  // namespace

inline void spin_wait_step_cuda(tvm::ffi::TensorView cycles) {
  using namespace host;

  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(cycles);

  const DLDevice device = device_.unwrap();
  const int64_t* cycles_ptr = static_cast<const int64_t*>(cycles.data_ptr());

  LaunchKernel(1u, 1u, device)(spin_wait_kernel, cycles_ptr);
}

}  // namespace canary
