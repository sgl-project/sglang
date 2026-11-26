#pragma once

#include <sgl_kernel/utils.h>

#include <cstddef>
#include <dlfcn.h>
#include <limits>

namespace sglang::memcpy_batch {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080

/// \brief cudaMemcpyBatchAsync of the cudart loaded in this process, with its ABI.
///
/// The symbol is resolved at run time: the loaded cudart (often torch's) can differ in
/// major version from the toolkit that built this module, and CUDA 13 dropped the failIdx
/// parameter that 12.8/12.9 take before the stream. The version is read from the same
/// library the symbol came from, so the call matches the function actually invoked.
struct Api {
  void* fn = nullptr;
  bool v13 = false;
};

inline const Api& api() {
  static const Api resolved = []() {
    Api a;
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess || driver_version < 12080) return a;
    void* sym = dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync");
    if (sym == nullptr) return a;
    Dl_info info{};
    if (dladdr(sym, &info) == 0 || info.dli_fname == nullptr) return a;
    void* lib = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    if (lib == nullptr) return a;
    using GetVersionFn = cudaError_t (*)(int*);
    auto get_version = reinterpret_cast<GetVersionFn>(dlsym(lib, "cudaRuntimeGetVersion"));
    int runtime_version = 0;
    const bool ok = get_version != nullptr && get_version(&runtime_version) == cudaSuccess;
    dlclose(lib);  // drops the NOLOAD reference only; the library stays loaded
    if (!ok || runtime_version < 12080) return a;
    a.fn = sym;
    a.v13 = runtime_version >= 13000;
    return a;
  }();
  return resolved;
}

inline bool available() {
  return api().fn != nullptr;
}

/// One batched copy with a single attribute set for every descriptor. Requires available().
inline cudaError_t
copy(void** dsts, void** srcs, size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, cudaStream_t stream) {
  size_t attrs_idx = 0;
  if (api().v13) {
    using FnV13 = cudaError_t (*)(
        void* const*, const void* const*, const size_t*, size_t, cudaMemcpyAttributes*, size_t*, size_t, cudaStream_t);
    return reinterpret_cast<FnV13>(api().fn)(dsts, srcs, sizes, count, attrs, &attrs_idx, 1, stream);
  }
  using FnV12 =
      cudaError_t (*)(void**, void**, size_t*, size_t, cudaMemcpyAttributes*, size_t*, size_t, size_t*, cudaStream_t);
  size_t fail_idx = std::numeric_limits<size_t>::max();
  return reinterpret_cast<FnV12>(api().fn)(dsts, srcs, sizes, count, attrs, &attrs_idx, 1, &fail_idx, stream);
}

#endif

}  // namespace sglang::memcpy_batch
