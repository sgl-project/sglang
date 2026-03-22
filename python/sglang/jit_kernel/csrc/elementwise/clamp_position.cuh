#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <typename T>
__global__ void clamp_position_kernel(T* __restrict__ dst, const T* __restrict__ seq_lens, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T val = seq_lens[idx] - 1;
    dst[idx] = val < 0 ? 0 : val;
  }
}

constexpr size_t kBlockSize = 256;

template <typename T>
struct ClampPosition {
  static void run(tvm::ffi::TensorView dst, tvm::ffi::TensorView seq_lens) {
    using namespace host;

    SymbolicSize N = {"num_elements"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA>();

    TensorMatcher({N})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(dst)
        .verify(seq_lens);

    const size_t num_elements = N.unwrap();
    if (num_elements == 0) return;

    const size_t grid_size = div_ceil(num_elements, kBlockSize);
    const DLDevice device = device_.unwrap();

    LaunchKernel(grid_size, kBlockSize, device)(
        clamp_position_kernel<T>,
        static_cast<T*>(dst.data_ptr()),
        static_cast<const T*>(seq_lens.data_ptr()),
        num_elements);
  }
};

}  // namespace
