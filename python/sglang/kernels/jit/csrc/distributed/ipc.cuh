#include <sgl_kernel/ffi.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace host::distributed {

struct AllocationRange {
  uintptr_t base;
  size_t size;
  size_t offset;
};

inline auto get_allocation_range(uintptr_t ptr) -> AllocationRange {
  CUdeviceptr base = 0;
  size_t size = 0;
  const CUresult res = cuMemGetAddressRange(&base, &size, ptr);
  if (res != CUDA_SUCCESS) {
    const char* name = nullptr;
    cuGetErrorName(res, &name);
    RuntimeCheck(false, "cuMemGetAddressRange failed: ", name ? name : "unknown");
  }
  const auto b = static_cast<uintptr_t>(base);
  return {.base = b, .size = size, .offset = ptr - b};
}

/**
 * \brief Batched cudaIpc handle exchange for CUDA-graph input pointers.
 *
 * `batch_get_handles` maps local device pointers to (base allocation IPC
 * handle, offset) pairs; `batch_open_handles` opens peer handles (cached per
 * unique handle) and returns absolute peer pointers. Only works for
 * cudaMalloc-backed pointers; VMM-backed pointers take the fabric/posix-fd
 * path in Python instead.
 */
struct IPCManager : public tvm::ffi::Object {
 public:
  using IPCHandle = std::array<char, sizeof(cudaIpcMemHandle_t)>;
  using FFIHandle = tvm::ffi::Array<char>;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.IPCManager", IPCManager, tvm::ffi::Object);
  static constexpr bool _type_mutable = true;

  using BatchGetResult = tvm::ffi::Array<tvm::ffi::Tuple<FFIHandle, size_t>>;
  using BatchGetInputs = tvm::ffi::Array<uintptr_t>;

  IPCManager() = default;
  ~IPCManager() {
    this->destroy();
  }

  void destroy() {
    for (const auto& [handle, base_addr] : m_handle2ptr_cache) {
      if (m_local_handles.count(handle)) continue;
      RuntimeDeviceCheck(cudaIpcCloseMemHandle(reinterpret_cast<void*>(base_addr)));
    }
    m_handle2ptr_cache.clear();
    m_ptr2handle_cache.clear();
    m_local_handles.clear();
  }

  BatchGetInputs batch_open_handles(BatchGetResult handles) {
    tvm::ffi::Array<uintptr_t> result;
    result.reserve(handles.size());
    for (const auto& pair : handles) {
      const auto ipc_handle = to_ipc_handle(get<0>(pair));
      const auto offset = get<1>(pair);
      result.push_back(open_handle(ipc_handle) + offset);
    }
    return result;
  }

  BatchGetResult batch_get_handles(const BatchGetInputs& ptrs) {
    RuntimeCheck(m_ptr2handle_cache.empty(), "Internal error: stale pointer cache");
    BatchGetResult result;
    result.reserve(ptrs.size());
    using Tuple = tvm::ffi::Tuple<FFIHandle, size_t>;
    for (const auto& ptr : ptrs) {
      const auto [ipc_handle, offset] = get_handle(ptr);
      result.emplace_back(Tuple{to_ffi_handle(ipc_handle), offset});
    }
    // We intentionally do NOT cache by base address across calls. The caching
    // allocator (e.g. PyTorch) may free a CUDA allocation and later return a
    // new allocation whose virtual range overlaps with the freed one; a cache
    // keyed on base/size would then hand back a stale `cudaIpcMemHandle_t`
    // that no longer maps to live memory on the peer. Re-querying is cheap.
    m_ptr2handle_cache.clear();
    return result;
  }

 private:
  static IPCHandle to_ipc_handle(const FFIHandle& ffi_handle) {
    IPCHandle ipc_handle;
    RuntimeCheck(ffi_handle.size() == sizeof(cudaIpcMemHandle_t), "Invalid IPC handle size: ", ffi_handle.size());
    for (size_t i = 0; i < sizeof(cudaIpcMemHandle_t); ++i) {
      ipc_handle[i] = static_cast<char>(ffi_handle[i]);
    }
    return ipc_handle;
  }

  static IPCHandle to_ipc_handle(const cudaIpcMemHandle_t& cuda_handle) {
    IPCHandle ipc_handle;
    std::memcpy(ipc_handle.data(), &cuda_handle, sizeof(cudaIpcMemHandle_t));
    return ipc_handle;
  }

  static FFIHandle to_ffi_handle(const IPCHandle& ipc_handle) {
    FFIHandle ffi_handle;
    ffi_handle.reserve(sizeof(cudaIpcMemHandle_t));
    for (size_t i = 0; i < sizeof(cudaIpcMemHandle_t); ++i) {
      ffi_handle.push_back(static_cast<uint8_t>(ipc_handle[i]));
    }
    return ffi_handle;
  }

  std::pair<IPCHandle, size_t> get_handle(uintptr_t ptr) {
    auto it = m_ptr2handle_cache.upper_bound(ptr);
    if (it != m_ptr2handle_cache.begin()) {
      --it;
      const auto& [cached_handle, cached_size] = it->second;
      const auto offset = ptr - it->first;
      if (offset < cached_size) return {cached_handle, offset};
    }
    // Not found in cache, query CUDA and cache the result
    const auto range = get_allocation_range(ptr);
    cudaIpcMemHandle_t handle;
    RuntimeDeviceCheck(cudaIpcGetMemHandle(&handle, reinterpret_cast<void*>(range.base)));
    const auto ipc_handle = to_ipc_handle(handle);
    const auto [_, success] = m_ptr2handle_cache.try_emplace(range.base, ipc_handle, range.size);
    RuntimeCheck(success, "Internal error: base address already exists in cache");
    m_handle2ptr_cache.try_emplace(ipc_handle, range.base);
    m_local_handles.insert(ipc_handle);
    return {ipc_handle, range.offset};
  }

  uintptr_t open_handle(const IPCHandle& handle) {
    const auto it = m_handle2ptr_cache.find(handle);
    if (it != m_handle2ptr_cache.end()) {
      return it->second;
    }
    cudaIpcMemHandle_t cuda_handle;
    std::memcpy(&cuda_handle, handle.data(), sizeof(cudaIpcMemHandle_t));
    void* base_ptr = nullptr;
    RuntimeDeviceCheck(cudaIpcOpenMemHandle(&base_ptr, cuda_handle, cudaIpcMemLazyEnablePeerAccess));
    const auto base_addr = reinterpret_cast<uintptr_t>(base_ptr);
    const auto [_, success] = m_handle2ptr_cache.try_emplace(handle, base_addr);
    RuntimeCheck(success, "Internal error: IPC handle already exists in cache");
    return base_addr;
  }

  struct EqualCUDAIPC {
    bool operator()(const IPCHandle& a, const IPCHandle& b) const {
      return std::memcmp(a.data(), b.data(), sizeof(cudaIpcMemHandle_t)) == 0;
    }
  };

  struct HashCUDAIPC {
    std::size_t operator()(const IPCHandle& handle) const {
      const auto sv = std::string_view{handle.data(), sizeof(cudaIpcMemHandle_t)};
      return std::hash<std::string_view>{}(sv);
    }
  };

  std::unordered_map<IPCHandle, uintptr_t, HashCUDAIPC, EqualCUDAIPC> m_handle2ptr_cache;
  std::unordered_set<IPCHandle, HashCUDAIPC, EqualCUDAIPC> m_local_handles;
  std::map<uintptr_t, std::pair<IPCHandle, size_t>> m_ptr2handle_cache;
};

}  // namespace host::distributed

inline void register_ipc_manager() {
  namespace refl = tvm::ffi::reflection;
  using Class = host::distributed::IPCManager;
  refl::ObjectDef<Class>()
      .def(refl::init<>(), "__init__")
      .def("batch_get_handles", &Class::batch_get_handles)
      .def("batch_open_handles", &Class::batch_open_handles)
      .def("destroy", &Class::destroy);
}
