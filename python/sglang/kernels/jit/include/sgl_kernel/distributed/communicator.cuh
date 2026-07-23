#pragma once
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/object.h>

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace device::distributed {

inline constexpr uint32_t kMaxWorldSize = 8;

struct Counter {
 public:
  Counter(const Counter&) = delete;
  SGL_DEVICE uint32_t get() const {
    return m_counter;
  }
  SGL_DEVICE void set(uint32_t val) {
    m_counter = val;
  }
  SGL_DEVICE uint32_t inc(uint32_t val) {
    return ::atomicAdd(&m_counter, val);
  }

 private:
  uint32_t m_counter;
};

struct alignas(128) Semaphore {
 public:
  Semaphore(const Semaphore&) = delete;
  SGL_DEVICE Counter* counter_ptr() {
    return &m_counter;
  }
  SGL_DEVICE uint32_t get_relaxed() const {
    uint32_t val;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(&m_flag) : "memory");
    return val;
  }
  SGL_DEVICE void put_relaxed() {
    asm volatile("red.relaxed.sys.global.add.u32 [%0], 1;" : : "l"(&m_flag) : "memory");
  }
  SGL_DEVICE uint32_t get_acquire() const {
    uint32_t val;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(&m_flag) : "memory");
    return val;
  }
  SGL_DEVICE void put_release() {
    asm volatile("red.release.sys.global.add.u32 [%0], 1;" : : "l"(&m_flag) : "memory");
  }

 private:
  uint32_t m_flag;
  Counter m_counter;
};

}  // namespace device::distributed

namespace host::distributed {

using device::distributed::Counter, device::distributed::Semaphore;
inline constexpr uint32_t kMaxWorldSize = device::distributed::kMaxWorldSize;

/**
 * \brief Storage plane of the custom all-reduce implementation.
 *
 * A thin, kernel-agnostic view over externally owned buffers: per-rank
 * symmetric workspaces, synchronization primitives, and grid-size settings.
 * It performs no allocation and no IPC; the Python side owns the storage
 * (symmetric memory) and its lifetime.
 */
struct CommunicatorObj : public tvm::ffi::Object {
 public:
  using TensorView = tvm::ffi::TensorView;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.Communicator", CommunicatorObj, tvm::ffi::Object);
  static constexpr bool _type_mutable = true;  // config() mutates block counts

  // Defined in csrc/distributed/communicator.cuh (only the registration
  // module needs the implementation).
  CommunicatorObj(
      uint32_t rank,
      uint32_t world_size,
      std::vector<TensorView> push_workspaces,
      std::vector<TensorView> pull_workspaces,
      std::vector<TensorView> pull_semaphores,
      TensorView push_counter,
      std::optional<int64_t> pull_mc_workspace_ptr);

  void config(std::map<std::string, uint32_t> config);

  uint32_t rank;
  uint32_t world_size;
  int64_t push_bytes;  // per-buffer bytes; each rank holds 2 * world_size buffers
  int64_t pull_bytes;
  uint32_t num_push_blocks;  // not configurable (bound to the counter array)
  uint32_t num_pull_blocks;
  uint32_t num_multicast_blocks;
  std::array<uint8_t*, kMaxWorldSize> pull_workspaces;    // symmetric memory
  std::array<uint8_t*, kMaxWorldSize> push_workspaces;    // symmetric memory
  std::array<Semaphore*, kMaxWorldSize> pull_semaphores;  // symmetric memory
  Counter* push_counter;                                  // local memory
  uint8_t* pull_mc_workspace;                             // multicast address of the pull workspace (may be null)

 private:  // upper bounds for config()
  uint32_t total_pull_blocks;
};

struct CommunicatorRef : public tvm::ffi::ObjectRef {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CommunicatorRef, tvm::ffi::ObjectRef, CommunicatorObj);
};

}  // namespace host::distributed
