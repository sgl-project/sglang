#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

namespace sglang {

template <int kSmem, int kMode>
__global__ void carveout_probe(int32_t *output) {
  device::PDLWaitPrimary<kMode == 2>();
  extern __shared__ int32_t scratch[];
  if constexpr (kSmem > 0) {
    scratch[threadIdx.x] = threadIdx.x;
    __syncthreads();
    output[threadIdx.x] = scratch[255 - threadIdx.x];
  } else {
    output[threadIdx.x] = 255 - threadIdx.x;
  }
}

template <int kSmem, int kMode>
void check_launch_carveout(tvm::ffi::TensorView output) {
  host::TensorMatcher({256})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>()
      .verify(output);
  const auto kernel = carveout_probe<kSmem, kMode>;
  const int device_id = output.device().device_id;
  const tvm::ffi::CUDADeviceGuard guard(device_id);
  int baseline = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&baseline, kernel,
                                                           256, kSmem));
  CHECK_HOST(baseline > 0);
  if constexpr (kMode == 2)
    host::ensure_prefer_l1(kernel, device_id, 256, kSmem);
  auto *ptr = static_cast<int32_t *>(output.data_ptr());
  if constexpr (kMode == 0) {
    host::LaunchKernel(1, 256, output.device(), kSmem)(kernel, ptr);
  } else if constexpr (kMode == 1) {
    host::LaunchKernel(
        1, 256, host::LaunchKernel::resolve_device(output.device()), kSmem)
        .prefer_l1()(kernel, ptr);
  } else {
    host::LaunchKernel(1, 256, output.device(), kSmem)
        .config({.use_pdl = true, .prefer_l1 = true})(kernel, ptr);
  }
  cudaFuncAttributes attrs{};
  CHECK_CUDA(cudaFuncGetAttributes(&attrs, kernel));
  if constexpr (kMode == 0) {
    CHECK_HOST(attrs.preferredShmemCarveout == cudaSharedmemCarveoutDefault);
  } else {
    const auto receipt = host::ensure_prefer_l1(kernel, device_id, 256, kSmem);
    const auto memo = host::ensure_prefer_l1(kernel, device_id, 256, kSmem);
    CHECK_HOST(receipt.carveout_pct >= 0 && receipt.carveout_pct <= 100);
    CHECK_HOST(receipt.blocks_per_sm >= static_cast<uint32_t>(baseline));
    CHECK_HOST(attrs.preferredShmemCarveout == receipt.carveout_pct);
    CHECK_HOST(memo.carveout_pct == receipt.carveout_pct &&
               memo.blocks_per_sm == receipt.blocks_per_sm);
    int actual = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&actual, kernel,
                                                             256, kSmem));
    CHECK_HOST(actual >= baseline);
  }
}

} // namespace sglang
