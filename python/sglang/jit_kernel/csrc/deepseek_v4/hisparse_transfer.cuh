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

struct OffloadParams {
  void** gpu_caches;
  void** cpu_caches;
  const int64_t* gpu_indices;
  const int64_t* cpu_indices;
  uint32_t num_items;
  uint32_t num_layers;
};

OFFLOAD_KERNEL void offload_to_cpu(const __grid_constant__ OffloadParams params) {
  using namespace device::hisparse;
  const auto [gpu_caches, cpu_caches, gpu_indices, cpu_indices, num_items, num_layers] = params;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr auto kNumWarps = (kBlockSize / 32) * kBlockQuota;
  for (auto i = global_tid / 32; i < num_items; i += kNumWarps) {
    const int32_t gpu_index = gpu_indices[i];
    const int32_t cpu_index = cpu_indices[i];
    for (auto j = 0u; j < num_layers; ++j) {
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
  const auto params = OffloadParams{
      .gpu_caches = static_cast<void**>(gpu_ptrs.data_ptr()),
      .cpu_caches = static_cast<void**>(cpu_ptrs.data_ptr()),
      .gpu_indices = static_cast<const int64_t*>(gpu_indices.data_ptr()),
      .cpu_indices = static_cast<const int64_t*>(cpu_indices.data_ptr()),
      .num_items = static_cast<uint32_t>(N.unwrap()),
      .num_layers = static_cast<uint32_t>(L.unwrap()),
  };
  LaunchKernel(kBlockQuota, kBlockSize, device_.unwrap())(offload_to_cpu, params);
}

}  // namespace
