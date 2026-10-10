#pragma once

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <vector>
#if !defined(USE_ROCM) && !defined(USE_MUSA)
#include <dlfcn.h>
#endif

namespace sglang::direct {
struct Copy {
  uintptr_t src, dst;
  size_t size;
};

inline cudaError_t submit_copies(std::vector<Copy>& copies, cudaStream_t stream) {
  if (copies.empty()) return cudaSuccess;
  // A smallest-first batch faulted in cuMemcpyBatchAsync on GB300 (driver 580.178.04);
  // the same disjoint, in-bounds copies passed individually. Largest-first is an
  // empirical workaround; stable sorting preserves locality for equal sizes.
  const auto larger_first = [](const Copy& a, const Copy& b) { return a.size > b.size; };
  if (!std::is_sorted(copies.begin(), copies.end(), larger_first)) {
    std::stable_sort(copies.begin(), copies.end(), larger_first);
  }
  const auto copy_individually = [&]() {
    for (const auto& copy : copies) {
      const auto result = cudaMemcpyAsync(
          reinterpret_cast<void*>(copy.dst),
          reinterpret_cast<const void*>(copy.src),
          copy.size,
          cudaMemcpyDefault,
          stream);
      if (result != cudaSuccess) return result;
    }
    return cudaSuccess;
  };
#if !defined(USE_ROCM) && !defined(USE_MUSA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  // The batch API rejects the legacy default stream.
  if (stream == nullptr) return copy_individually();
  static const auto batch_fn = dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync");
  if (batch_fn == nullptr) return copy_individually();
  static int driver_version = 0;
  static const auto driver_status = cudaDriverGetVersion(&driver_version);
  if (driver_status != cudaSuccess) return driver_status;
  if (driver_version < 12080) return copy_individually();
  static int runtime_version = 0;
  static const auto runtime_status = cudaRuntimeGetVersion(&runtime_version);
  if (runtime_status != cudaSuccess) return runtime_status;
  cudaError_t result;

  std::vector<void*> srcs, dsts;
  std::vector<size_t> sizes;
  srcs.reserve(copies.size());
  dsts.reserve(copies.size());
  sizes.reserve(copies.size());
  for (const auto& copy : copies) {
    srcs.push_back(reinterpret_cast<void*>(copy.src));
    dsts.push_back(reinterpret_cast<void*>(copy.dst));
    sizes.push_back(copy.size);
  }
  cudaMemcpyAttributes attrs{};
  attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t attrs_index = 0;
  // CUDA 13 removed failIdx; choose the ABI by loaded runtime version.
  if (runtime_version >= 13000) {
    using Fn = cudaError_t (*)(
        void* const*, const void* const*, const size_t*, size_t, cudaMemcpyAttributes*, size_t*, size_t, cudaStream_t);
    result = reinterpret_cast<Fn>(batch_fn)(
        dsts.data(), srcs.data(), sizes.data(), copies.size(), &attrs, &attrs_index, 1, stream);
  } else {
    using Fn =
        cudaError_t (*)(void**, void**, size_t*, size_t, cudaMemcpyAttributes*, size_t*, size_t, size_t*, cudaStream_t);
    size_t fail_index = std::numeric_limits<size_t>::max();
    result = reinterpret_cast<Fn>(batch_fn)(
        dsts.data(), srcs.data(), sizes.data(), copies.size(), &attrs, &attrs_index, 1, &fail_index, stream);
  }
  if (result != cudaErrorNotSupported && result != cudaErrorCallRequiresNewerDriver) return result;
  (void)cudaGetLastError();
#endif
  return copy_individually();
}
}  // namespace sglang::direct
