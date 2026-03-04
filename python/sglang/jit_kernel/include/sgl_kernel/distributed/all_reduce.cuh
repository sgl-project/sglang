#pragma once
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/common.cuh>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace host::distributed {

struct AllReduceData {
  constexpr AllReduceData() {}
  void* __restrict__ input[device::distributed::kMaxNumGPU];
};

using ExternHandle = tvm::ffi::Array<char>;

inline ExternHandle to_extern_handle(void* ptr) {
  using namespace host;
  ExternHandle array;
  cudaIpcMemHandle_t handle;
  RuntimeDeviceCheck(cudaIpcGetMemHandle(&handle, ptr));
  for (size_t i = 0; i < sizeof(handle); ++i) {
    array.push_back(handle.reserved[i]);
  }
  return array;
}

inline void* from_extern_handle(const ExternHandle& array) {
  using namespace host;
  cudaIpcMemHandle_t handle;
  RuntimeCheck(array.size() == sizeof(handle), "Invalid IPC handle size: ", array.size());
  for (size_t i = 0; i < sizeof(handle); ++i) {
    handle.reserved[i] = array[i];
  }
  void* ptr;
  RuntimeDeviceCheck(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  return ptr;
}

struct HandleHash {
  std::size_t operator()(const cudaIpcMemHandle_t& handle) const {
    return std::hash<std::string_view>{}({handle.reserved, sizeof(handle.reserved)});
  }
};

struct HandleEqual {
  bool operator()(const cudaIpcMemHandle_t& a, const cudaIpcMemHandle_t& b) const {
    return std::memcmp(a.reserved, b.reserved, sizeof(a.reserved)) == 0;
  }
};

/**
 * \brief The control plane of the custom all-reduce implementation.
 * It manages the internal state and synchronization of the participating GPUs.
 */
struct CustomAllReduceBase : public tvm::ffi::Object {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.CustomAllReduce", CustomAllReduceBase, tvm::ffi::Object);

  static constexpr bool _type_mutable = true;
  using InputPair = tvm::ffi::Tuple<int64_t, ExternHandle>;  // (offset, ipc handle)

  CustomAllReduceBase(
      uint32_t rank, uint32_t num_gpu, uint32_t max_num_cta, int64_t buffer_size_bytes, int64_t graph_buffer_count)
      : m_buffer_size_bytes(buffer_size_bytes),
        m_graph_buffer_count(graph_buffer_count),
        m_rank(rank),
        m_num_gpu(num_gpu),
        m_max_num_cta(max_num_cta) {
    // The buffer layout is as follows:
    // | SignalArray | AllReduceData (local) |  GraphBuffer (AllReduceData) | buffer data |
    using namespace host;
    const auto needed_bytes = m_payload_bytes() + buffer_size_bytes;
    RuntimeDeviceCheck(cudaMalloc(&m_storage, needed_bytes));
    RuntimeCheck(rank < m_num_gpu, "Invalid rank: ", rank);
    const int64_t kU32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    RuntimeCheck(m_buffer_size_bytes <= kU32Max, "Buffer size is too large: ", m_buffer_size_bytes);
  }

  ExternHandle share_storage() {
    return to_extern_handle(m_storage);
  }

  tvm::ffi::Array<InputPair> share_graph_inputs() {
    using namespace host;
    tvm::ffi::Array<InputPair> result;
    result.reserve(m_graph_capture_inputs.size());
    std::unordered_map<void*, ExternHandle> ipc_cache;
    const auto get_handle = [&](void* ptr) -> ExternHandle {
      auto it = ipc_cache.find(ptr);
      if (it != ipc_cache.end()) return it->second;
      const auto handle = to_extern_handle(ptr);
      ipc_cache.try_emplace(ptr, handle);
      return handle;
    };
    for (const auto ptr : m_graph_capture_inputs) {
      // note: must share the base address of each allocation, or we get wrong address
      void* base_ptr;
      const auto cu_result = cuPointerGetAttribute(&base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr);
      RuntimeCheck(cu_result == CUDA_SUCCESS, "failed to get pointer attr");
      const auto offset = reinterpret_cast<char*>(ptr) - reinterpret_cast<char*>(base_ptr);
      result.push_back(InputPair{offset, get_handle(base_ptr)});
    }
    return result;
  }

  void post_init(tvm::ffi::Array<ExternHandle> ipc_storages) {
    using namespace host;
    RuntimeCheck(ipc_storages.size() == m_num_gpu, "Invalid array size: ", ipc_storages.size());
    m_storage_pointers.resize(m_num_gpu);
    for (const auto i : irange(m_num_gpu)) {
      if (i == m_rank) {
        m_storage_pointers[i] = m_storage;
      } else {
        m_storage_pointers[i] = from_extern_handle(ipc_storages[i]);
      }
    }

    // set signal buffer to zero
    RuntimeDeviceCheck(cudaMemset(m_storage, 0, m_semaphore_bytes()));

    // update the default data pointer
    AllReduceData data;
    for (const auto i : irange(m_num_gpu)) {
      data.input[i] = pointer::offset(m_storage_pointers[i], m_payload_bytes());
    }
    RuntimeDeviceCheck(cudaMemcpy(m_get_data_ptr(), &data, sizeof(AllReduceData), cudaMemcpyHostToDevice));

    // update the controller
    RuntimeCheck(!m_ctrl.has_value(), "Controller is already initialized");
    m_ctrl.emplace(m_storage_pointers.data(), m_num_gpu);
  }

  void register_inputs(tvm::ffi::Array<tvm::ffi::Array<InputPair>> ipc_graph_inputs) {
    using namespace host;
    RuntimeCheck(!m_is_registered, "Inputs have already been registered");
    m_is_registered = true;
    RuntimeCheck(ipc_graph_inputs.size() == m_num_gpu);
    const auto registered_count = m_registered_count();
    if (registered_count == 0) return;  // avoid `m_get_data_ptr(0)` out-of-bounds when no input is registered
    std::vector<AllReduceData> data;
    data.resize(registered_count);
    std::unordered_map<cudaIpcMemHandle_t, void*, HandleHash, HandleEqual> ipc_cache;
    const auto open_cached = [&](const ExternHandle& h) -> void* {
      RuntimeCheck(h.size() == sizeof(cudaIpcMemHandle_t), "Invalid IPC handle size: ", h.size());
      cudaIpcMemHandle_t handle;
      for (size_t i = 0; i < sizeof(handle); ++i)
        handle.reserved[i] = h[i];
      const auto [it, success] = ipc_cache.try_emplace(handle, nullptr);
      if (success) {
        void* ptr;
        RuntimeDeviceCheck(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
        it->second = ptr;
      }
      return it->second;
    };
    for (const auto i : irange(ipc_graph_inputs.size())) {
      const auto& array = ipc_graph_inputs[i];
      RuntimeCheck(int64_t(array.size()) == registered_count);
      if (i == m_rank) {
        for (const auto j : irange(registered_count)) {
          data[j].input[i] = m_graph_capture_inputs[j];
        }
      } else {
        for (const auto j : irange(registered_count)) {
          /// NOTE: structural binding will cause intern compiler error...
          const auto elem = array[j];
          const auto offset = get<0>(elem);
          const auto ipc_handle = get<1>(elem);
          data[j].input[i] = pointer::offset(open_cached(ipc_handle), offset);
        }
      }
    }
    const auto registered_bytes = sizeof(AllReduceData) * registered_count;
    RuntimeDeviceCheck(cudaMemcpy(m_get_data_ptr(0), data.data(), registered_bytes, cudaMemcpyHostToDevice));
  }

  void set_cuda_graph_capture(bool enabled) {
    host::RuntimeCheck(!m_is_registered, "Cannot set graph capture mode after inputs are registered");
    m_is_graph_capturing = enabled;
  }

  void free() {
    host::RuntimeDeviceCheck(cudaFree(m_storage));
  }

 protected:
  AllReduceData* m_allocate_graph_capture_input(void* data_ptr) {
    using namespace host;
    const auto count = m_registered_count();
    RuntimeCheck(count < m_graph_buffer_count, "Graph buffer overflow, increase `graph_buffer_count`!");
    m_graph_capture_inputs.push_back(data_ptr);
    return m_get_data_ptr(count);
  }

  AllReduceData* m_get_data_ptr(int64_t which = -1) {
    using namespace host;
    const auto count = m_registered_count();
    RuntimeCheck(which >= -1 && which < count, "Invalid graph buffer index: ", which, ", count: ", count);
    const auto start = pointer::offset(m_storage, m_semaphore_bytes());
    return static_cast<AllReduceData*>(start) + (1 + which);
  }

  int64_t m_registered_count() const {
    return static_cast<int64_t>(m_graph_capture_inputs.size());
  }

  int64_t m_semaphore_bytes() const {
    return sizeof(device::distributed::Semaphore) * m_max_num_cta;
  }

  int64_t m_payload_bytes() const {
    return m_semaphore_bytes() + sizeof(AllReduceData) * (1 + m_graph_buffer_count);
  }

  const int64_t m_buffer_size_bytes;
  const int64_t m_graph_buffer_count;
  const uint32_t m_rank;
  const uint32_t m_num_gpu;
  const uint32_t m_max_num_cta;
  bool m_is_graph_capturing = false;
  bool m_is_registered = false;
  void* m_storage = nullptr;
  std::vector<void*> m_graph_capture_inputs;
  std::vector<void*> m_storage_pointers;
  std::optional<device::distributed::Controller> m_ctrl;
};

struct CustomAllReduceRef : public tvm::ffi::ObjectRef {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CustomAllReduceRef, tvm::ffi::ObjectRef, CustomAllReduceBase);
};

}  // namespace host::distributed
