#pragma once
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/common.cuh>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace host::distributed {

using device::distributed::PullController, device::distributed::PushController;

struct AllReduceData {
  constexpr AllReduceData() {}
  void* __restrict__ input[device::distributed::kMaxNumGPU];
};

using ExternHandle = tvm::ffi::Array<char>;

inline ExternHandle to_extern_handle(void* ptr) {
  ExternHandle array;
  cudaIpcMemHandle_t handle;
  RuntimeDeviceCheck(cudaIpcGetMemHandle(&handle, ptr));
  for (size_t i = 0; i < sizeof(handle); ++i) {
    array.push_back(handle.reserved[i]);
  }
  return array;
}

inline void* from_extern_handle(const ExternHandle& array) {
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
      uint32_t rank,
      uint32_t num_gpu,
      uint32_t max_num_cta_pull,
      uint32_t max_num_cta_push,
      int64_t pull_buffer_size,
      int64_t push_buffer_size,
      int64_t graph_buffer_count)
      : m_pull_buffer_bytes(pull_buffer_size),
        m_push_buffer_bytes(push_buffer_size),
        m_graph_buffer_count(graph_buffer_count),
        m_rank(rank),
        m_num_gpu(num_gpu),
        m_max_num_cta_pull(max_num_cta_pull),
        m_max_num_cta_push(max_num_cta_push),
        // default config for pull kernel, can be updated by `configure()`
        m_num_cta(max_num_cta_pull),
        m_cta_size(256) {
    RuntimeCheck(pull_buffer_size % 128 == 0, "Pull buffer size should be aligned to 128 bytes");
    RuntimeCheck(push_buffer_size % 128 == 0, "Push buffer size should be aligned to 128 bytes");
    RuntimeCheck(rank < num_gpu, "Invalid rank: ", rank);
    const int64_t kU32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    const int64_t push_buffer_size_all = push_all_ranks_bytes();
    RuntimeCheck(pull_buffer_size <= kU32Max, "Pull buffer size is too large: ", pull_buffer_size);
    RuntimeCheck(push_buffer_size_all <= kU32Max, "Push buffer size is too large: ", push_buffer_size_all);
    RuntimeDeviceCheck(cudaMalloc(&m_storage, storage_bytes()));
  }

  ExternHandle share_storage() {
    return to_extern_handle(m_storage);
  }

  tvm::ffi::Array<InputPair> share_graph_inputs() {
    tvm::ffi::Array<InputPair> result;
    const auto new_inputs_count = registered_count() - m_cum_registered_count;
    RuntimeCheck(new_inputs_count >= 0, "Invalid new count: ", new_inputs_count);
    result.reserve(new_inputs_count);
    std::unordered_map<void*, ExternHandle> ipc_cache;
    const auto get_handle = [&](void* ptr) -> ExternHandle {
      const auto it = ipc_cache.find(ptr);
      if (it != ipc_cache.end()) return it->second;
      const auto handle = to_extern_handle(ptr);
      ipc_cache.try_emplace(ptr, handle);
      return handle;
    };
    for (const auto ptr : std::span(m_graph_capture_inputs).subspan(m_cum_registered_count)) {
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
    RuntimeCheck(ipc_storages.size() == m_num_gpu, "Invalid array size: ", ipc_storages.size());
    m_peer_storage.resize(m_num_gpu);
    for (const auto i : irange(m_num_gpu)) {
      if (i == m_rank) {
        m_peer_storage[i] = m_storage;
      } else {
        m_peer_storage[i] = from_extern_handle(ipc_storages[i]);
      }
    }

    // set signal buffer to zero
    const auto pull_signal = get_pull_signal(m_storage);
    RuntimeDeviceCheck(cudaMemset(pull_signal, 0, pull_signal_bytes()));

    // update the pull controller and data pointer
    RuntimeCheck(!m_pull_ctrl.has_value(), "Controller is already initialized");
    m_pull_ctrl.emplace(m_peer_storage.data(), m_num_gpu);
    AllReduceData data;
    for (const auto i : irange(m_num_gpu)) {
      data.input[i] = get_pull_buffer(m_peer_storage[i]);
    }
    const auto default_data_ptr = get_data_ptr();
    RuntimeDeviceCheck(cudaMemcpy(default_data_ptr, &data, sizeof(AllReduceData), cudaMemcpyHostToDevice));

    // update the push controller and data pointer
    RuntimeCheck(!m_push_ctrl.has_value(), "Controller is already initialized");
    const auto push_signal = get_push_signal(m_storage);
    RuntimeDeviceCheck(cudaMemset(push_signal, 0, push_signal_bytes()));
    m_push_ctrl.emplace(push_signal);
    const auto push_buffer = get_push_buffer(m_storage);
    RuntimeDeviceCheck(cudaMemset(push_buffer, 0, push_all_ranks_bytes()));
  }

  void register_inputs(tvm::ffi::Array<tvm::ffi::Array<InputPair>> ipc_graph_inputs) {
    RuntimeCheck(ipc_graph_inputs.size() == m_num_gpu);
    const auto new_registered_count = registered_count() - m_cum_registered_count;
    RuntimeCheck(new_registered_count >= 0, "Invalid registered count: ", new_registered_count);
    if (new_registered_count == 0) return;  // avoid `m_get_data_ptr()` out-of-bounds
    std::vector<AllReduceData> data;
    data.resize(new_registered_count);
    const auto open_cached = [&](const ExternHandle& h) -> void* {
      RuntimeCheck(h.size() == sizeof(cudaIpcMemHandle_t), "Invalid IPC handle size: ", h.size());
      cudaIpcMemHandle_t handle;
      for (size_t i = 0; i < sizeof(handle); ++i)
        handle.reserved[i] = h[i];
      const auto [it, success] = m_ipc_cache.try_emplace(handle, nullptr);
      if (success) {
        void* ptr;
        RuntimeDeviceCheck(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
        it->second = ptr;
      }
      return it->second;
    };
    for (const auto i : irange(ipc_graph_inputs.size())) {
      const auto& array = ipc_graph_inputs[i];
      RuntimeCheck(int64_t(array.size()) == new_registered_count);
      if (i == m_rank) {
        for (const auto j : irange(new_registered_count)) {
          data[j].input[i] = m_graph_capture_inputs[m_cum_registered_count + j];
        }
      } else {
        for (const auto j : irange(new_registered_count)) {
          /// NOTE: structural binding will cause intern compiler error...
          const auto elem = array[j];
          const auto offset = elem.get<0>();
          const auto ipc_handle = elem.get<1>();
          data[j].input[i] = pointer::offset(open_cached(ipc_handle), offset);
        }
      }
    }

    const auto new_registered_bytes = sizeof(AllReduceData) * new_registered_count;
    const auto dst_ptr = get_data_ptr(m_cum_registered_count);
    m_cum_registered_count += new_registered_count;
    RuntimeDeviceCheck(cudaMemcpy(dst_ptr, data.data(), new_registered_bytes, cudaMemcpyHostToDevice));
  }

  void set_cuda_graph_capture(bool enabled) {
    m_is_graph_capturing = enabled;
  }

  void free_ipc_handles() {
    for (const auto& pair : m_ipc_cache) {
      host::RuntimeDeviceCheck(cudaIpcCloseMemHandle(pair.second));
    }
    m_ipc_cache.clear();
  }

  void free_storage() {
    host::RuntimeDeviceCheck(cudaFree(m_storage));
    m_storage = nullptr;
  }

  tvm::ffi::Tuple<uint32_t, uint32_t> configure_pull(uint32_t num_cta, uint32_t cta_size) {
    using host::RuntimeCheck;
    const auto min_cta_size = m_num_gpu * device::kWarpThreads;
    RuntimeCheck(num_cta > 0 && num_cta <= m_max_num_cta_pull, "Invalid number of CTAs: ", num_cta);
    RuntimeCheck(cta_size >= min_cta_size, "Block size must be at least ", min_cta_size);
    const auto old_num_cta = m_num_cta;
    const auto old_block_size = m_cta_size;
    m_num_cta = num_cta;
    m_cta_size = cta_size;
    return tvm::ffi::Tuple<uint32_t, uint32_t>{old_num_cta, old_block_size};
  }

 protected:
  AllReduceData* allocate_graph_capture_input(void* data_ptr) {
    const auto count = registered_count();
    RuntimeCheck(count < m_graph_buffer_count, "Graph buffer overflow, increase `graph_buffer_count`!");
    m_graph_capture_inputs.push_back(data_ptr);
    return get_data_ptr(count);
  }
  AllReduceData* get_data_ptr(int64_t which = -1) {
    const auto count = registered_count();
    RuntimeCheck(which >= -1 && which < count, "Invalid graph buffer index: ", which, ", count: ", count);
    const auto start = get_pull_params(m_storage);
    return static_cast<AllReduceData*>(start) + (1 + which);
  }
  int64_t registered_count() const {
    return static_cast<int64_t>(m_graph_capture_inputs.size());
  }
  int64_t pull_signal_bytes() const {
    return _align_bytes(sizeof(PullController::SignalType) * m_max_num_cta_pull);
  }
  int64_t push_signal_bytes() const {
    return _align_bytes(sizeof(PushController::SignalType) * m_max_num_cta_push);
  }
  int64_t graph_param_bytes() const {
    return _align_bytes(sizeof(AllReduceData) * (1 + m_graph_buffer_count));  // 1 for default
  }
  int64_t push_all_ranks_bytes() const {
    return _align_bytes(PushController::kNumStages * m_num_gpu * m_push_buffer_bytes);
  }
  int64_t storage_bytes() const {
    return _get_offset_impl(5);
  }
  void* get_pull_signal(void* ptr) const {
    return pointer::offset(ptr, _get_offset_impl(0));
  }
  void* get_push_signal(void* ptr) const {
    return pointer::offset(ptr, _get_offset_impl(1));
  }
  void* get_pull_params(void* ptr) const {
    return pointer::offset(ptr, _get_offset_impl(2));
  }
  void* get_pull_buffer(void* ptr) const {
    return pointer::offset(ptr, _get_offset_impl(3));
  }
  void* get_push_buffer(void* ptr) const {
    return pointer::offset(ptr, _get_offset_impl(4));
  }
  int64_t _get_offset_impl(int64_t which) const {
    // | SignalArray (pull + push) | GraphBuffers (pull params) | Buffers (pull + push) |
    const int64_t offset_map[5] = {
        /*[0]=*/pull_signal_bytes(),
        /*[1]=*/push_signal_bytes(),
        /*[2]=*/graph_param_bytes(),
        /*[3]=*/m_pull_buffer_bytes,
        /*[4]=*/push_all_ranks_bytes(),
    };
    RuntimeCheck(which >= 0 && which <= 5, "Invalid offset index: ", which);
    return std::accumulate(offset_map, offset_map + which, int64_t(0));
  }
  static int64_t _align_bytes(int64_t size) {
    return div_ceil(size, 128) * 128;
  }

  const int64_t m_pull_buffer_bytes;
  const int64_t m_push_buffer_bytes;
  const int64_t m_graph_buffer_count;
  const uint32_t m_rank;
  const uint32_t m_num_gpu;
  const uint32_t m_max_num_cta_pull;
  const uint32_t m_max_num_cta_push;
  // these 2 config should only affect pull kernel
  uint32_t m_num_cta;
  uint32_t m_cta_size;
  // other states
  bool m_is_graph_capturing = false;
  int64_t m_cum_registered_count = 0;
  std::optional<PullController> m_pull_ctrl;
  std::optional<PushController> m_push_ctrl;
  void* m_storage = nullptr;
  std::vector<void*> m_graph_capture_inputs;
  std::vector<void*> m_peer_storage;
  std::unordered_map<cudaIpcMemHandle_t, void*, HandleHash, HandleEqual> m_ipc_cache;
};

struct CustomAllReduceRef : public tvm::ffi::ObjectRef {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CustomAllReduceRef, tvm::ffi::ObjectRef, CustomAllReduceBase);
};

}  // namespace host::distributed

namespace device::distributed {

template <typename DType2, size_t N, uint32_t M>
SGL_DEVICE auto reduce_impl(AlignedVector<DType2, N> (&storage)[M]) -> AlignedVector<DType2, N> {
  fp32x2_t acc[N] = {};
#pragma unroll  // unroll num gpu
  for (uint32_t i = 0; i < M; ++i) {
#pragma unroll  // unroll vec
    for (uint32_t j = 0; j < N; ++j) {
      const auto [x, y] = cast<fp32x2_t>(storage[i][j]);
      auto& [x_acc, y_acc] = acc[j];
      x_acc += x;
      y_acc += y;
    }
  }

  AlignedVector<DType2, N> result;
#pragma unroll
  for (uint32_t j = 0; j < N; ++j) {
    result[j] = cast<DType2>(acc[j]);
  }

  return result;
}

}  // namespace device::distributed
