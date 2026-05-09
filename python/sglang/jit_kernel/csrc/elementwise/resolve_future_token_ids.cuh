#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

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

__global__ void resolve_future_token_ids_int32_vec4_kernel(
    int32_t* __restrict__ input_ids,
    const int32_t* __restrict__ future_map,
    size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_vec = n / 4;

  if (idx < n_vec) {
    int4 values = reinterpret_cast<const int4*>(input_ids)[idx];
    bool changed = false;
    if (values.x < 0) {
      int32_t key = -values.x;
      values.x = future_map[key < 0 ? 0 : key];
      changed = true;
    }
    if (values.y < 0) {
      int32_t key = -values.y;
      values.y = future_map[key < 0 ? 0 : key];
      changed = true;
    }
    if (values.z < 0) {
      int32_t key = -values.z;
      values.z = future_map[key < 0 ? 0 : key];
      changed = true;
    }
    if (values.w < 0) {
      int32_t key = -values.w;
      values.w = future_map[key < 0 ? 0 : key];
      changed = true;
    }
    if (changed) {
      reinterpret_cast<int4*>(input_ids)[idx] = values;
    }
  }

  const size_t tail_idx = n_vec * 4 + idx;
  if (tail_idx < n) {
    int32_t val = input_ids[tail_idx];
    if (val < 0) {
      int32_t key = -val;
      if (key < 0) key = 0;
      input_ids[tail_idx] = future_map[key];
    }
  }
}

constexpr size_t kBlockSize = 256;
constexpr size_t kVecMinElements = 1 << 20;

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

    const DLDevice device = device_.unwrap();
    auto* input_ids_ptr = static_cast<T*>(input_ids.data_ptr());
    const auto* future_map_ptr = static_cast<const T*>(future_map.data_ptr());

    if constexpr (std::is_same_v<T, int32_t>) {
      const bool is_vec_aligned =
          (reinterpret_cast<uintptr_t>(input_ids_ptr) % alignof(int4) == 0) &&
          (reinterpret_cast<uintptr_t>(future_map_ptr) % alignof(int4) == 0);
      if (num_tokens >= kVecMinElements && is_vec_aligned) {
        const size_t vec_work_items = std::max(num_tokens / 4, num_tokens % 4);
        const size_t grid_size = div_ceil(vec_work_items, kBlockSize);
        LaunchKernel(grid_size, kBlockSize, device)(
            resolve_future_token_ids_int32_vec4_kernel,
            static_cast<int32_t*>(input_ids.data_ptr()),
            static_cast<const int32_t*>(future_map.data_ptr()),
            num_tokens);
        return;
      }
    }

    const size_t grid_size = div_ceil(num_tokens, kBlockSize);
    LaunchKernel(grid_size, kBlockSize, device)(
        resolve_future_token_ids_kernel<T>,
        input_ids_ptr,
        future_map_ptr,
        num_tokens);
  }
};

}  // namespace
