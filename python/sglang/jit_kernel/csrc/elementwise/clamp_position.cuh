#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

template <typename T>
__global__ void clamp_position_kernel(T* __restrict__ dst, const T* __restrict__ seq_lens, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T val = seq_lens[idx] - 1;
    dst[idx] = val < 0 ? 0 : val;
  }
}

__global__ void clamp_position_int32_vec4_kernel(
    int32_t* __restrict__ dst,
    const int32_t* __restrict__ seq_lens,
    size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_vec = n / 4;

  if (idx < n_vec) {
    int4 values = reinterpret_cast<const int4*>(seq_lens)[idx];
    values.x = values.x <= 0 ? 0 : values.x - 1;
    values.y = values.y <= 0 ? 0 : values.y - 1;
    values.z = values.z <= 0 ? 0 : values.z - 1;
    values.w = values.w <= 0 ? 0 : values.w - 1;
    reinterpret_cast<int4*>(dst)[idx] = values;
  }

  const size_t tail_idx = n_vec * 4 + idx;
  if (tail_idx < n) {
    int32_t val = seq_lens[tail_idx] - 1;
    dst[tail_idx] = val < 0 ? 0 : val;
  }
}

constexpr size_t kBlockSize = 256;
constexpr size_t kVecMinElements = 1 << 20;

template <typename T>
struct ClampPosition {
  static void run(tvm::ffi::TensorView dst, tvm::ffi::TensorView seq_lens) {
    using namespace host;

    SymbolicSize N = {"num_elements"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({N})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(dst)
        .verify(seq_lens);

    const size_t num_elements = N.unwrap();
    if (num_elements == 0) return;

    const DLDevice device = device_.unwrap();
    auto* dst_ptr = static_cast<T*>(dst.data_ptr());
    const auto* seq_lens_ptr = static_cast<const T*>(seq_lens.data_ptr());

    if constexpr (std::is_same_v<T, int32_t>) {
      const bool is_vec_aligned =
          (reinterpret_cast<uintptr_t>(dst_ptr) % alignof(int4) == 0) &&
          (reinterpret_cast<uintptr_t>(seq_lens_ptr) % alignof(int4) == 0);
      if (num_elements >= kVecMinElements && is_vec_aligned) {
        const size_t vec_work_items = std::max(num_elements / 4, num_elements % 4);
        const size_t grid_size = div_ceil(vec_work_items, kBlockSize);
        LaunchKernel(grid_size, kBlockSize, device)(
            clamp_position_int32_vec4_kernel,
            static_cast<int32_t*>(dst.data_ptr()),
            static_cast<const int32_t*>(seq_lens.data_ptr()),
            num_elements);
        return;
      }
    }

    const size_t grid_size = div_ceil(num_elements, kBlockSize);
    LaunchKernel(grid_size, kBlockSize, device)(
        clamp_position_kernel<T>,
        dst_ptr,
        seq_lens_ptr,
        num_elements);
  }
};

}  // namespace
