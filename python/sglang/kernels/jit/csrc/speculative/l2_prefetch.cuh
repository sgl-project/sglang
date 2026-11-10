#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

// A prefetch is a hint: it returns no data and has no completion to wait on, so it orders against no other access.
__global__ void l2_prefetch_kernel(const int64_t* __restrict__ chunks, uint32_t num_chunks) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_chunks) {
    const uint64_t addr = static_cast<uint64_t>(chunks[2 * i]);
    const uint32_t bytes = static_cast<uint32_t>(chunks[2 * i + 1]);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(addr), "r"(bytes) : "memory");
  }
}

struct L2PrefetchKernel {
  static void run(tvm::ffi::TensorView chunks) {
    using namespace host;
    RuntimeCheck(chunks.ndim() == 2 && chunks.size(1) == 2, "chunks must be an [n, 2] table");
    const int64_t n = chunks.size(0);
    if (n == 0) return;
    constexpr uint32_t kThreads = 256;
    const auto blocks = static_cast<uint32_t>((n + kThreads - 1) / kThreads);
    LaunchKernel(dim3(blocks), dim3(kThreads), chunks.device())(
        l2_prefetch_kernel, static_cast<const int64_t*>(chunks.data_ptr()), static_cast<uint32_t>(n));
  }
};

}  // namespace sglang
