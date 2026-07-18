#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "quick_all_reduce.cuh"

#define HIP_CHECK(err)                                                                               \
  do {                                                                                               \
    hipError_t err_ = (err);                                                                         \
    if (err_ != hipSuccess) {                                                                        \
      std::printf("HIP error %d at %s:%d. %s\n", err_, __FILE__, __LINE__, hipGetErrorString(err_)); \
      throw std::runtime_error("HIP error");                                                         \
    }                                                                                                \
  } while (0)

namespace quickreduce {
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void allreduce_prototype_twoshot(
    T const* A,
    T* B,
    uint32_t N,
    uint32_t num_blocks,
    int rank,
    uint8_t** dbuffer_list,
    uint32_t data_offset,
    uint32_t* d_flag_counters,
    int64_t data_size_per_phase) {
  int block = blockIdx.x;
  int grid = gridDim.x;

  // Read this block's counter from device memory and bump it here in the
  // kernel. Keeping the value in device memory (instead of a host scalar
  // baked into the launch) lets every CUDA-graph replay see a fresh color.
  uint32_t flag_color = d_flag_counters[blockIdx.x];

  while (block < num_blocks) {
    AllReduceKernel::run(A, B, N, block, rank, dbuffer_list, data_offset, flag_color, data_size_per_phase);
    block += grid;
    flag_color++;
  }
  // The whole block ends up with the same value, so a single writer suffices.
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    d_flag_counters[blockIdx.x] = flag_color;
  }
}

#define TWOSHOT_DISPATCH(__codec)                                         \
  if (world_size == 2) {                                                  \
    using LineCodec = __codec<T, 2>;                                      \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>; \
    hipLaunchKernelGGL(                                                   \
        (allreduce_prototype_twoshot<AllReduceKernel, T>),                \
        dim3(grid),                                                       \
        dim3(kBlockTwoShot),                                              \
        0,                                                                \
        stream,                                                           \
        A,                                                                \
        B,                                                                \
        N,                                                                \
        num_blocks,                                                       \
        rank,                                                             \
        dbuffer_list,                                                     \
        data_offset,                                                      \
        d_flag_counters,                                                  \
        this->kMaxProblemSize);                                           \
  } else if (world_size == 4) {                                           \
    using LineCodec = __codec<T, 4>;                                      \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>; \
    hipLaunchKernelGGL(                                                   \
        (allreduce_prototype_twoshot<AllReduceKernel, T>),                \
        dim3(grid),                                                       \
        dim3(kBlockTwoShot),                                              \
        0,                                                                \
        stream,                                                           \
        A,                                                                \
        B,                                                                \
        N,                                                                \
        num_blocks,                                                       \
        rank,                                                             \
        dbuffer_list,                                                     \
        data_offset,                                                      \
        d_flag_counters,                                                  \
        this->kMaxProblemSize);                                           \
  } else if (world_size == 8) {                                           \
    using LineCodec = __codec<T, 8>;                                      \
    using AllReduceKernel = AllReduceTwoshot<T, LineCodec, cast_bf2half>; \
    hipLaunchKernelGGL(                                                   \
        (allreduce_prototype_twoshot<AllReduceKernel, T>),                \
        dim3(grid),                                                       \
        dim3(kBlockTwoShot),                                              \
        0,                                                                \
        stream,                                                           \
        A,                                                                \
        B,                                                                \
        N,                                                                \
        num_blocks,                                                       \
        rank,                                                             \
        dbuffer_list,                                                     \
        data_offset,                                                      \
        d_flag_counters,                                                  \
        this->kMaxProblemSize);                                           \
  }

enum QuickReduceQuantLevel {
  F16 = 0,
  INT8 = 1,
  INT6 = 2,
  INT4 = 3,
};

class HipDeviceGuard {
 public:
  explicit HipDeviceGuard(int device) {
    HIP_CHECK(hipGetDevice(&previous_device_));
    changed_ = previous_device_ != device;
    if (changed_) HIP_CHECK(hipSetDevice(device));
  }

  ~HipDeviceGuard() {
    if (changed_) (void)hipSetDevice(previous_device_);
  }

 private:
  int previous_device_ = 0;
  bool changed_ = false;
};

struct VmmPeerMapping {
  uint8_t* ptr = nullptr;
  int64_t size = 0;
  hipMemGenericAllocationHandle_t handle = 0;
  bool reserved = false;
  bool mapped = false;
};

struct DeviceComms {
  static constexpr int64_t kDefaultMaxProblemSize = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;

  // Max problem size is 2GB (in bytes) or half of uint32_t max value.
  int64_t kMaxProblemSize = kDefaultMaxProblemSize;

  // Max TP-8
  static int constexpr kMaxWorldSize = 8;

  bool initialized = false;
  uint32_t* d_flag_counters = nullptr;
  int world_size;
  int rank;
  int device_index;

  uint8_t* dbuffer;
  uint8_t** dbuffer_list;
  bool uses_vmm;
  hipIpcMemHandle_t buffer_ipc_handle;
  std::vector<hipIpcMemHandle_t> all_buffer_ipc_handles;
  std::vector<uint8_t*> buffer_list;
  uint32_t data_offset;
  hipMemGenericAllocationHandle_t vmm_local_handle;
  int64_t vmm_local_size;
  bool vmm_local_reserved;
  bool vmm_local_mapped;
  std::vector<VmmPeerMapping> vmm_peer_mappings;

  DeviceComms()
      : initialized(false),
        d_flag_counters(nullptr),
        world_size(1),
        rank(0),
        device_index(-1),
        dbuffer(nullptr),
        dbuffer_list(nullptr),
        uses_vmm(false),
        data_offset(0),
        vmm_local_handle(0),
        vmm_local_size(0),
        vmm_local_reserved(false),
        vmm_local_mapped(false) {}
  ~DeviceComms() noexcept {
    destroy_impl(false);
  }

  void init(int world_size, int rank, std::optional<int64_t> max_problem_size = std::nullopt) {
    destroy();
    this->world_size = world_size;
    this->rank = rank;
    HIP_CHECK(hipGetDevice(&device_index));
    this->kMaxProblemSize = kDefaultMaxProblemSize;
    if (max_problem_size.has_value() && max_problem_size.value() > 0) {
      this->kMaxProblemSize = max_problem_size.value();
    }
    uses_vmm = false;
    try {
      HIP_CHECK(hipExtMallocWithFlags(
          (void**)&dbuffer, get_buffer_size(world_size, max_problem_size), hipDeviceMallocUncached));
      init_common_buffers();
      all_buffer_ipc_handles.resize(world_size);
      HIP_CHECK(hipIpcGetMemHandle(&buffer_ipc_handle, dbuffer));
      initialized = true;
    } catch (...) {
      destroy();
      throw;
    }
  }

  static int64_t get_buffer_size(int world_size, std::optional<int64_t> max_problem_size = std::nullopt) {
    int64_t problem_size = kDefaultMaxProblemSize;
    if (max_problem_size.has_value() && max_problem_size.value() > 0) {
      problem_size = max_problem_size.value();
    }
    int64_t flags_buffer_size = 2LL * world_size * kMaxNumBlocks * sizeof(uint32_t);
    return flags_buffer_size + 2LL * problem_size;
  }

  std::pair<int, int64_t> init_vmm(
      int world_size,
      int rank,
      int device_index,
      std::optional<int64_t> max_problem_size = std::nullopt,
      bool uncached = true) {
    destroy();
    this->world_size = world_size;
    this->rank = rank;
    this->device_index = device_index;
    this->kMaxProblemSize = kDefaultMaxProblemSize;
    if (max_problem_size.has_value() && max_problem_size.value() > 0) {
      this->kMaxProblemSize = max_problem_size.value();
    }
    uses_vmm = true;

    HipDeviceGuard guard(device_index);
    try {
      hipMemAllocationProp prop{};
      prop.type = uncached ? hipMemAllocationTypeUncached : hipMemAllocationTypePinned;
      prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
      prop.location.type = hipMemLocationTypeDevice;
      prop.location.id = device_index;

      size_t granularity = 0;
      HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityRecommended));
      int64_t requested_size = get_buffer_size(world_size, max_problem_size);
      vmm_local_size = ((requested_size + granularity - 1) / granularity) * granularity;
      HIP_CHECK(hipMemCreate(&vmm_local_handle, vmm_local_size, &prop, 0));
      HIP_CHECK(hipMemAddressReserve((void**)&dbuffer, vmm_local_size, granularity, nullptr, 0));
      vmm_local_reserved = true;
      HIP_CHECK(hipMemMap(dbuffer, vmm_local_size, 0, vmm_local_handle, 0));
      vmm_local_mapped = true;

      hipMemAccessDesc access{};
      access.location.type = hipMemLocationTypeDevice;
      access.location.id = device_index;
      access.flags = hipMemAccessFlagsProtReadWrite;
      HIP_CHECK(hipMemSetAccess(dbuffer, vmm_local_size, &access, 1));
      init_common_buffers();

      int export_fd = -1;
      HIP_CHECK(hipMemExportToShareableHandle(&export_fd, vmm_local_handle, hipMemHandleTypePosixFileDescriptor, 0));
      return {export_fd, vmm_local_size};
    } catch (...) {
      destroy();
      throw;
    }
  }

  void init_common_buffers() {
    data_offset = 2 * world_size * kMaxNumBlocks * sizeof(uint32_t);
    HIP_CHECK(hipMemset(dbuffer, 0, data_offset));
    HIP_CHECK(hipMalloc(&d_flag_counters, kMaxNumBlocks * sizeof(uint32_t)));
    std::vector<uint32_t> init_color(kMaxNumBlocks, 1u);
    HIP_CHECK(hipMemcpy(d_flag_counters, init_color.data(), kMaxNumBlocks * sizeof(uint32_t), hipMemcpyHostToDevice));
    buffer_list.assign(world_size, nullptr);
    buffer_list[rank] = dbuffer;
    HIP_CHECK(hipMalloc(&dbuffer_list, world_size * sizeof(uint8_t*)));
  }
  int get_world_size() {
    return world_size;
  }
  int get_rank() {
    return rank;
  }
  bool status() {
    return initialized;
  }
  hipIpcMemHandle_t const get_handle() {
    return buffer_ipc_handle;
  }

  void destroy() {
    destroy_impl(true);
  }

  void destroy_impl(bool throw_on_error) noexcept(false) {
    hipError_t first_error = hipSuccess;
    auto cleanup_call = [&first_error](hipError_t error) {
      if (error != hipSuccess && first_error == hipSuccess) first_error = error;
    };

    int previous_device = -1;
    bool restore_device = false;
    if (device_index >= 0) {
      hipError_t get_device_error = hipGetDevice(&previous_device);
      cleanup_call(get_device_error);
      if (get_device_error == hipSuccess && previous_device != device_index) {
        hipError_t set_device_error = hipSetDevice(device_index);
        cleanup_call(set_device_error);
        restore_device = set_device_error == hipSuccess;
      }
    }

    if (d_flag_counters) {
      cleanup_call(hipFree(d_flag_counters));
      d_flag_counters = nullptr;
    }

    if (uses_vmm) {
      for (auto& mapping : vmm_peer_mappings) {
        if (mapping.mapped) cleanup_call(hipMemUnmap(mapping.ptr, mapping.size));
        if (mapping.reserved) cleanup_call(hipMemAddressFree(mapping.ptr, mapping.size));
        if (mapping.handle) cleanup_call(hipMemRelease(mapping.handle));
      }
      vmm_peer_mappings.clear();
      if (vmm_local_mapped) cleanup_call(hipMemUnmap(dbuffer, vmm_local_size));
      if (vmm_local_reserved) cleanup_call(hipMemAddressFree(dbuffer, vmm_local_size));
      if (vmm_local_handle) cleanup_call(hipMemRelease(vmm_local_handle));
    } else {
      for (int i = 0; i < static_cast<int>(buffer_list.size()); i++) {
        if (i != rank && buffer_list[i]) {
          cleanup_call(hipIpcCloseMemHandle(buffer_list[i]));
        }
      }
      if (dbuffer) cleanup_call(hipFree(dbuffer));
    }
    if (dbuffer_list) cleanup_call(hipFree(dbuffer_list));

    if (restore_device) cleanup_call(hipSetDevice(previous_device));

    dbuffer = nullptr;
    dbuffer_list = nullptr;
    buffer_list.clear();
    all_buffer_ipc_handles.clear();
    data_offset = 0;
    vmm_local_handle = 0;
    vmm_local_size = 0;
    vmm_local_reserved = false;
    vmm_local_mapped = false;
    uses_vmm = false;
    initialized = false;
    device_index = -1;

    if (throw_on_error && first_error != hipSuccess) {
      throw std::runtime_error("QuickAllReduce HIP cleanup failed: " + std::string(hipGetErrorString(first_error)));
    }
  }

  void open_ipc_handles(std::vector<hipIpcMemHandle_t> const& ipc_handles) {
    assert(ipc_handles.size() == all_buffer_ipc_handles.size());
    for (int i = 0; i < world_size; i++) {
      all_buffer_ipc_handles[i] = ipc_handles[i];
    }

    // Open device memory access to the IPC communication buffers.
    // Note: For our own rank, we do not need to open a handle.
    for (int i = 0; i < world_size; i++) {
      if (i != rank) {
        HIP_CHECK(
            hipIpcOpenMemHandle((void**)&buffer_list[i], all_buffer_ipc_handles[i], hipIpcMemLazyEnablePeerAccess));
      } else {
        buffer_list[i] = dbuffer;
      }
    }

    HIP_CHECK(hipMemcpy(dbuffer_list, buffer_list.data(), world_size * sizeof(uint8_t*), hipMemcpyHostToDevice));
  }

  void open_vmm_handles(const std::vector<int64_t>& peer_fds, const std::vector<int64_t>& peer_sizes) {
    if (!uses_vmm || peer_fds.size() != static_cast<size_t>(world_size) ||
        peer_sizes.size() != static_cast<size_t>(world_size)) {
      throw std::invalid_argument("invalid QuickAllReduce VMM peer metadata");
    }

    HipDeviceGuard guard(device_index);
    try {
      for (int i = 0; i < world_size; i++) {
        if (i == rank) continue;
        if (peer_fds[i] < 0 || peer_sizes[i] <= 0) {
          throw std::invalid_argument("invalid QuickAllReduce VMM peer fd or size");
        }
        if (peer_sizes[i] != vmm_local_size) {
          throw std::invalid_argument("QuickAllReduce VMM peer allocation size mismatch");
        }

        vmm_peer_mappings.emplace_back();
        auto& mapping = vmm_peer_mappings.back();
        mapping.size = peer_sizes[i];
        HIP_CHECK(hipMemImportFromShareableHandle(
            &mapping.handle,
            reinterpret_cast<void*>(static_cast<intptr_t>(peer_fds[i])),
            hipMemHandleTypePosixFileDescriptor));
        hipMemAllocationProp peer_prop{};
        HIP_CHECK(hipMemGetAllocationPropertiesFromHandle(&peer_prop, mapping.handle));
        size_t peer_granularity = 0;
        HIP_CHECK(
            hipMemGetAllocationGranularity(&peer_granularity, &peer_prop, hipMemAllocationGranularityRecommended));
        HIP_CHECK(hipMemAddressReserve((void**)&mapping.ptr, mapping.size, peer_granularity, nullptr, 0));
        mapping.reserved = true;
        HIP_CHECK(hipMemMap(mapping.ptr, mapping.size, 0, mapping.handle, 0));
        mapping.mapped = true;

        hipMemAccessDesc access{};
        access.location.type = hipMemLocationTypeDevice;
        access.location.id = device_index;
        access.flags = hipMemAccessFlagsProtReadWrite;
        HIP_CHECK(hipMemSetAccess(mapping.ptr, mapping.size, &access, 1));
        buffer_list[i] = mapping.ptr;
      }
      HIP_CHECK(hipMemcpy(dbuffer_list, buffer_list.data(), world_size * sizeof(uint8_t*), hipMemcpyHostToDevice));
      initialized = true;
    } catch (...) {
      destroy();
      throw;
    }
  }

  template <typename T, bool cast_bf2half>
  void allreduce(T const* A, T* B, uint32_t N, int quant_level, hipStream_t stream) {
    if (world_size != 2 && world_size != 4 && world_size != 8) {
      throw std::runtime_error("All Reduce not supported for world_size = " + std::to_string(world_size));
    }

    // Configuration.
    uint32_t msg_size = N * sizeof(T);
    uint32_t num_blocks = divceil(msg_size, kTileSize);
    uint32_t grid = min(kMaxNumBlocks, num_blocks);
    auto quant_level_ = static_cast<QuickReduceQuantLevel>(quant_level);
    switch (quant_level_) {
      case QuickReduceQuantLevel::INT8:
        TWOSHOT_DISPATCH(CodecQ8)
        break;
      case QuickReduceQuantLevel::INT6:
        TWOSHOT_DISPATCH(CodecQ6)
        break;
      case QuickReduceQuantLevel::INT4:
        TWOSHOT_DISPATCH(CodecQ4)
        break;
      default:
        TWOSHOT_DISPATCH(CodecFP)
        break;
    }
    HIP_CHECK(cudaGetLastError());
    // The color now advances on-device inside the kernel; no host-side bump.
  }
};

}  // namespace quickreduce
