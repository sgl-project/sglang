#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For div_ceil, RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx] + kConstant;
  }
}

template <int32_t kConstant>
__global__ void add_constant_vec4_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t length_vec4 = length / 4;

  if (idx < length_vec4) {
    int4 values = reinterpret_cast<const int4*>(src)[idx];
    values.x += kConstant;
    values.y += kConstant;
    values.z += kConstant;
    values.w += kConstant;
    reinterpret_cast<int4*>(dst)[idx] = values;
  }

  const size_t tail_idx = length_vec4 * 4 + idx;
  if (tail_idx < length) {
    dst[tail_idx] = src[tail_idx] + kConstant;
  }
}

constexpr size_t kBlockSize = 256;
constexpr size_t kVec4MinElements = 1 << 20;

// You can also use struct with static method as an alternative
template <int32_t kConstant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

  // 1. Validate input tensors
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  TensorMatcher({N})                  // 1D tensor, must be contiguous
      .with_dtype<int32_t>()          // must be int32
      .with_device<kDLCUDA>(device_)  // must be on CUDA device
      .verify(dst)                    // check tensor dst
      .verify(src);                   // check tensor src

  // 2. Extract required parameters, prepare for kernel launch
  const size_t num_elements = N.unwrap();
  const DLDevice device = device_.unwrap();
  [[maybe_unused]]  // optional, can be omitted
  const size_t dynamic_smem = 0;
  [[maybe_unused]]  // optional, LaunchKernel can auto determine stream from device
  const cudaStream_t stream = LaunchKernel::resolve_device(device);
  // some extra runtime checks using host::RuntimeCheck
  RuntimeCheck(num_elements > 0, "We only support non-empty tensors, got num_elements = ", num_elements);

  const auto* src_ptr = static_cast<const int32_t*>(src.data_ptr());
  auto* dst_ptr = static_cast<int32_t*>(dst.data_ptr());
  const bool is_vec4_aligned = (reinterpret_cast<uintptr_t>(src_ptr) % alignof(int4) == 0) &&
                               (reinterpret_cast<uintptr_t>(dst_ptr) % alignof(int4) == 0);

  // 3. Launch the kernel. Error code will be automatically checked.
  if (num_elements >= kVec4MinElements && is_vec4_aligned) {
    const size_t vec4_work_items = std::max(num_elements / 4, num_elements % 4);
    const size_t grid_size = div_ceil(vec4_work_items, kBlockSize);
    LaunchKernel(grid_size, kBlockSize, device /*, dynamic_smem*/)(
        add_constant_vec4_kernel<kConstant>, dst_ptr, src_ptr, num_elements);
  } else {
    const size_t grid_size = div_ceil(num_elements, kBlockSize);
    LaunchKernel(grid_size, kBlockSize, device /*, dynamic_smem*/)(
        add_constant_kernel<kConstant>, dst_ptr, src_ptr, num_elements);
  }
}

}  // namespace
