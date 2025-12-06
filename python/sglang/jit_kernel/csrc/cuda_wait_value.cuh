#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace {

auto stream_wait_value(const tvm::ffi::TensorView flag, std::int32_t value) -> void {
  using namespace host;

  auto length = SymbolicSize{"length"};
  TensorMatcher({length}).with_dtype<int32_t>().with_device<kDLCUDA>().verify(flag);
  RuntimeCheck(length.unwrap() >= 1, "wait_flag expects a non-empty tensor.");

  auto* ptr = static_cast<std::int32_t*>(flag.data_ptr());
  const auto stream = LaunchKernel::resolve_device(flag.device());

  printf("Waiting for flag=%p to become %d\n", ptr, value);
  fflush(stdout);

  CUresult result =
      cuStreamWaitValue32(stream, reinterpret_cast<CUdeviceptr>(ptr), static_cast<cuuint32_t>(value), 0x0);

  printf("cuStreamWaitValue32 returned: %d\n", result);
  fflush(stdout);

  RuntimeCheck(result == CUDA_SUCCESS, "cuStreamWaitValue32 failed");
}

}  // namespace
