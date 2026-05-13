#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/kvcacheio.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

/// NOTE: for offload to cpu kernel, we use persistent kernel
inline constexpr uint32_t kBlockSize = 1024;
inline constexpr uint32_t kBlockQuota = 4;

#define OFFLOAD_KERNEL __global__ __launch_bounds__(kBlockSize, 1)

struct IOParams {
  void* gpu_cache;
  void* cpu_cache;
  const int64_t* gpu_indices;
  const int64_t* cpu_indices;
  uint32_t num_items;
  uint32_t num_layers;
};

OFFLOAD_KERNEL void offload_to_cpu(const __grid_constant__ IOParams params) {
  using namespace device::hisparse;
  const auto gpu_caches = static_cast<void**>(params.gpu_cache);
  const auto cpu_caches = static_cast<void**>(params.cpu_cache);
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr auto kNumWarps = (kBlockSize / 32) * kBlockQuota;
  for (auto i = global_tid / 32; i < params.num_items; i += kNumWarps) {
    const int32_t gpu_index = params.gpu_indices[i];
    const int32_t cpu_index = params.cpu_indices[i];
    for (auto j = 0u; j < params.num_layers; ++j) {
      const auto gpu_cache = gpu_caches[j];
      const auto cpu_cache = cpu_caches[j];
      transfer_item<TransferDirection::DeviceToHost>(
          /*dst_cache=*/cpu_cache,
          /*src_cache=*/gpu_cache,
          /*dst_index=*/cpu_index,
          /*src_index=*/gpu_index);
    }
  }
}

OFFLOAD_KERNEL void load_to_gpu(const __grid_constant__ IOParams params) {
  using namespace device::hisparse;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr auto kNumWarps = (kBlockSize / 32) * kBlockQuota;
  for (auto i = global_tid / 32; i < params.num_items; i += kNumWarps) {
    const int32_t gpu_index = params.gpu_indices[i];
    const int32_t cpu_index = params.cpu_indices[i];
    transfer_item<TransferDirection::HostToDevice>(
        /*dst_cache=*/params.gpu_cache,
        /*src_cache=*/params.cpu_cache,
        /*dst_index=*/gpu_index,
        /*src_index=*/cpu_index);
  }
}

[[maybe_unused]]
void hisparse_transfer(
    tvm::ffi::TensorView gpu_ptrs,
    tvm::ffi::TensorView cpu_ptrs,
    tvm::ffi::TensorView gpu_indices,
    tvm::ffi::TensorView cpu_indices) {
  using namespace host;
  auto N = SymbolicSize{"num_items"};
  auto L = SymbolicSize{"num_layers"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({L})  // 1D cache pointers
      .with_dtype<uint64_t>()
      .with_device(device_)
      .verify(gpu_ptrs)
      .verify(cpu_ptrs);
  TensorMatcher({N})  // 1D indices
      .with_dtype<int64_t>()
      .with_device(device_)
      .verify(gpu_indices)
      .verify(cpu_indices);
  const auto params = IOParams{
      .gpu_cache = gpu_ptrs.data_ptr(),
      .cpu_cache = cpu_ptrs.data_ptr(),
      .gpu_indices = static_cast<const int64_t*>(gpu_indices.data_ptr()),
      .cpu_indices = static_cast<const int64_t*>(cpu_indices.data_ptr()),
      .num_items = static_cast<uint32_t>(N.unwrap()),
      .num_layers = static_cast<uint32_t>(L.unwrap()),
  };
  LaunchKernel(kBlockQuota, kBlockSize, device_.unwrap())(offload_to_cpu, params);
}

[[maybe_unused]]
void hisparse_load_to_device(
    tvm::ffi::TensorView gpu_cache,
    tvm::ffi::TensorView cpu_cache,
    tvm::ffi::TensorView gpu_indices,
    tvm::ffi::TensorView cpu_indices) {
  using namespace host;
  auto N = SymbolicSize{"num_items"};
  auto device_ = SymbolicDevice{};
  TensorMatcher({-1, -1})  // DSV4 page-padded C4 cache on GPU
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(gpu_cache);
  TensorMatcher({-1, -1})  // token-linear registered host C4 cache
      .with_dtype<uint8_t>()
      .with_device<kDLCPU, kDLCUDAHost>()
      .verify(cpu_cache);
  TensorMatcher({N})  // 1D indices
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(gpu_indices)
      .verify(cpu_indices);
  const auto params = IOParams{
      .gpu_cache = gpu_cache.data_ptr(),
      .cpu_cache = cpu_cache.data_ptr(),
      .gpu_indices = static_cast<const int64_t*>(gpu_indices.data_ptr()),
      .cpu_indices = static_cast<const int64_t*>(cpu_indices.data_ptr()),
      .num_items = static_cast<uint32_t>(N.unwrap()),
      .num_layers = 1,
  };
  LaunchKernel(kBlockQuota, kBlockSize, device_.unwrap())(load_to_gpu, params);
}

}  // namespace
