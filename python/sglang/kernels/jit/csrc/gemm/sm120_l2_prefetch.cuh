#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

// Fire-and-forget L2 warmer: stream a weight blob through L2 with default
// (evict-normal) policy so the consumer kernels a few microseconds later hit
// L2 instead of DRAM. Launched on a low-priority side stream during the
// DRAM-quiet compute phases of the mamba layers.
__global__ void __launch_bounds__(256) sm120_l2_prefetch_kernel(const uint4* __restrict__ p, const size_t n_vec) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint32_t sink = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    const uint4 v = p[i];
    sink ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  asm volatile("" ::"r"(sink));
}

struct Sm120L2PrefetchKernel {
  static void run(const tvm::ffi::TensorView t, int64_t num_blocks) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto nbytes = SymbolicSize{"bytes"};
    TensorMatcher({nbytes}).with_dtype<uint8_t>().with_device(device).verify(t);
    const size_t n_vec = static_cast<size_t>(nbytes.unwrap()) / 16;
    LaunchKernel(static_cast<uint32_t>(num_blocks), 256, device.unwrap())(
        sm120_l2_prefetch_kernel, static_cast<const uint4*>(t.data_ptr()), n_vec);
  }
};

}  // namespace sglang
