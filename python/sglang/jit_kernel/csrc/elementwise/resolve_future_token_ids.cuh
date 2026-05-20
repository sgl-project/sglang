#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <typename T>
__global__ void resolve_future_token_ids_kernel(T* __restrict__ input_ids, const T* __restrict__ future_map, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T val = input_ids[idx];
    if (val < 0) {
      T key = -val;
      if (key < 0) key = 0;  // clamp for overflow
      input_ids[idx] = future_map[key];
    }
  }
}

constexpr size_t kBlockSize = 256;

template <typename T>
struct ResolveFutureTokenIds {
  static void run(tvm::ffi::TensorView input_ids, tvm::ffi::TensorView future_map) {
    using namespace host;

    SymbolicSize N = {"num_tokens"};
    SymbolicSize M = {"map_size"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N}).with_dtype<T>().with_device(device_).verify(input_ids);

    TensorMatcher({M}).with_dtype<T>().with_device(device_).verify(future_map);

    const size_t num_tokens = N.unwrap();
    if (num_tokens == 0) return;

    const size_t grid_size = div_ceil(num_tokens, kBlockSize);
    const DLDevice device = device_.unwrap();

    LaunchKernel(grid_size, kBlockSize, device)(
        resolve_future_token_ids_kernel<T>,
        static_cast<T*>(input_ids.data_ptr()),
        static_cast<const T*>(future_map.data_ptr()),
        num_tokens);
  }
};

}  // namespace
