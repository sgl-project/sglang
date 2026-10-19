// L2 prefetch of a static weight range, issued while the GPU waits on a latency-bound step
// (a cross-GPU all-reduce) so the next bandwidth-bound kernel reads part of its weights from
// L2. Bulk prefetches are fire-and-forget hints; no value changes.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace sglang {

/** \brief Bulk L2 prefetches of [base, base + bytes), chunk i issued by CTA i % gridDim.x. */
__global__ void mkd_l2_prefetch_kernel(const char* base, int64_t bytes, int32_t chunk) {
  const int64_t nchunks = (bytes + chunk - 1) / chunk;
  for (int64_t i = blockIdx.x + static_cast<int64_t>(gridDim.x) * threadIdx.x; i < nchunks;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t off = i * chunk;
    const uint32_t size = static_cast<uint32_t>(min(static_cast<int64_t>(chunk), bytes - off));
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + off), "r"(size) : "memory");
  }
}

struct MkdPrefetch {
  static void run(const tvm::ffi::TensorView weight, int64_t bytes, int64_t chunk) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({-1, -1}).with_device(device).ensure_alignment(16).verify(weight);
    CHECK_HOST(chunk % 16 == 0 && bytes % 16 == 0) << "bulk prefetch sizes must be multiples of 16 bytes";
    const auto* base = static_cast<const char*>(weight.data_ptr());
    // Spread the chunks over up to 128 CTAs (tunable).
    const int64_t nchunks = (bytes + chunk - 1) / chunk;
    const uint32_t grid = static_cast<uint32_t>(std::min<int64_t>(128, std::max<int64_t>(1, nchunks)));
    LaunchKernel(grid, 32, device.unwrap())(mkd_l2_prefetch_kernel, base, bytes, static_cast<int32_t>(chunk));
  }
};

}  // namespace sglang
