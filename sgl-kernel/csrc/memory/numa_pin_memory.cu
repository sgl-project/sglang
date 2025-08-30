#include <ATen/core/TensorBody.h>
#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/TensorOptions.h>
#include <cuda_runtime.h>
#include <numa.h>

#include <cstdlib>
#include <optional>

auto allocate_pin_memory(
    const int64_t size,
    const at::ScalarType dtype,
    const bool write_combined,
    const std::optional<int64_t> numa_affinity) -> at::Tensor {
  const auto size_bytes = size * at::elementSize(dtype);
  void* data_ptr;
  const auto options = at::TensorOptions(at::kCPU).dtype(dtype).pinned_memory(true);
  if (numa_affinity.has_value()) {
    // only initialize once
    static const int kNumaCount = ::numa_max_node() + 1;
    const auto node = static_cast<int>(numa_affinity.value());
    TORCH_CHECK(node >= 0 && node < kNumaCount, "Invalid NUMA node: ", node);
    TORCH_CHECK(!write_combined, "Write-combine is not compatible with NUMA allocation");
    // allocate on the specified NUMA node
    data_ptr = ::numa_alloc_onnode(size_bytes, node);
    TORCH_CHECK(data_ptr != nullptr, "Failed to allocate memory on NUMA node: ", node);
    const auto result = ::cudaHostRegister(data_ptr, size_bytes, cudaHostRegisterDefault);
    if (result != ::cudaSuccess) {
      ::numa_free(data_ptr, size_bytes);
      TORCH_CHECK(false, "Failed to register pinned memory: ", ::cudaGetErrorString(result));
    }
    return at::from_blob(
        data_ptr,
        {size},
        [size_bytes](void* data_ptr) {
          const auto result = ::cudaHostUnregister(data_ptr);
          ::numa_free(data_ptr, size_bytes);
          TORCH_CHECK(result == ::cudaSuccess, "Failed to unregister pinned memory: ", ::cudaGetErrorString(result));
        },
        options,
        at::kCPU);
  } else {
    const auto flags = write_combined ? cudaHostAllocWriteCombined : cudaHostAllocDefault;
    const auto result = ::cudaHostAlloc(&data_ptr, size_bytes, flags);
    TORCH_CHECK(result == ::cudaSuccess, "Failed to allocate pinned memory: ", ::cudaGetErrorString(result));
    return at::from_blob(
        data_ptr,
        {size},
        [](void* data_ptr) {
          const auto result = ::cudaFreeHost(data_ptr);
          TORCH_CHECK(result == ::cudaSuccess, "Failed to free pinned memory: ", ::cudaGetErrorString(result));
        },
        options,
        at::kCPU);
  }
}
