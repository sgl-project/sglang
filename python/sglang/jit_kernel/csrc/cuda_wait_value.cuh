#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <cuda_runtime_api.h>

#include <cstdint>

namespace {

__global__ void wait_flag_kernel(const int32_t* flag, int32_t target) {
  const volatile int32_t* vflag = (volatile const int32_t*)flag;

  while (*vflag != target) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(100);
#else
    // Note: This falls back to an inefficient busy-wait on pre-Volta architectures.
#endif
  }
}

auto stream_wait_value(const tvm::ffi::TensorView flag, std::int32_t value) -> void {
  using namespace host;

  auto length = SymbolicSize{"length"};
  TensorMatcher({length}).with_dtype<int32_t>().with_device<kDLCUDA>().verify(flag);
  RuntimeCheck(length.unwrap() >= 1, "wait_flag expects a non-empty tensor.");

  auto* ptr = static_cast<std::int32_t*>(flag.data_ptr());
  const auto stream = LaunchKernel::resolve_device(flag.device());

  constexpr int blocks = 1;
  constexpr int threads = 1;
  wait_flag_kernel<<<blocks, threads, 0, stream>>>(ptr, value);
  RuntimeDeviceCheck(cudaGetLastError());
}

}  // namespace
