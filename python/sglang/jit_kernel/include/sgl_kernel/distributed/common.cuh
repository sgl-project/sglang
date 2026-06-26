#pragma once
#include <sgl_kernel/utils.cuh>

namespace device::distributed {

inline constexpr uint32_t kMaxNumGPU = 8;

struct alignas(128) Semaphore {
 public:
  constexpr Semaphore() : m_flag(0), m_counter(0) {}

  template <bool kFence>
  SGL_DEVICE uint32_t get() const {
    uint32_t val;
    if constexpr (kFence) {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(&m_flag));
    } else {
      asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(val) : "l"(&m_flag));
    }
    return val;
  }

  template <bool kFence>
  SGL_DEVICE uint32_t add(uint32_t val) {
    uint32_t old_val;
    if constexpr (kFence) {
      asm volatile("atom.release.sys.global.add.u32 %0, [%1], %2;" : "=r"(old_val) : "l"(&m_flag), "r"(val));
    } else {
      asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(old_val) : "l"(&m_flag), "r"(val));
    }
    return old_val;
  }

  // Only called by the owning GPU - plain load is sufficient
  SGL_DEVICE uint32_t get_counter() const {
    return m_counter;
  }

  // Only called by the owning GPU - plain store is sufficient
  SGL_DEVICE void set_counter(uint32_t val) {
    m_counter = val;
  }

 private:
  uint32_t m_flag;
  uint32_t m_counter;
};

struct PullController {
 public:
  using SignalType = Semaphore;

  PullController(void** signals, uint32_t num_gpu) {
    for (uint32_t i = 0; i < num_gpu; ++i) {
      m_signals[i] = static_cast<Semaphore*>(signals[i]);
    }
  }

  /// Synchronize all GPUs.
  /// When kFence is true, establishes happens-before across GPUs using
  /// release/acquire semantics, ensuring prior writes are visible system-wide.
  template <bool kFence, bool kStart>
  SGL_DEVICE void sync(uint32_t rank, uint32_t num_gpu) const {
    // For fenced sync: ensure all threads in this block have completed their writes,
    // so the signaling thread's release carries them transitively.
    static_assert(!(kFence && kStart), "Start stage does not need to wait fence");
    if constexpr (kFence || !kStart) __syncthreads();
    constexpr auto kStage = kStart ? 1 : 2;
    const auto warp_id = threadIdx.x / kWarpThreads;
    const auto lane_id = threadIdx.x % kWarpThreads;
    if (lane_id == 0 && warp_id < num_gpu) {
      auto& signal = m_signals[warp_id][blockIdx.x];
      signal.add<kFence>(1);
      if (warp_id == rank) {
        const auto target = num_gpu * kStage;
        /// NOTE: correctness here:
        /// - base is only read/updated locally by the owning GPU
        const auto base = signal.get_counter();
        while (signal.get<kFence>() - base < target)
          ;
        if constexpr (!kStart) {
          signal.set_counter(base + target);
        }
      }
    }
    if constexpr (kStart) __syncthreads();
  }

 private:
  Semaphore* __restrict__ m_signals[kMaxNumGPU];
};

struct PushController {
 public:
  using SignalType = uint32_t;
  static constexpr int64_t kNumStages = 2;

  PushController(void* ptr) : m_local_signal(static_cast<SignalType*>(ptr)) {}

  SGL_DEVICE SignalType epoch() const {
    return m_local_signal[blockIdx.x];
  }

  SGL_DEVICE void exit() const {
    __syncthreads();
    if (threadIdx.x == 0) {
      this->exit_unsafe(blockIdx.x);
    }
  }

  SGL_DEVICE void exit_unsafe(uint32_t which) const {
    auto& signal = m_local_signal[which];
    signal = (signal + 1) % kNumStages;
  }

 private:
  SignalType* m_local_signal;
};

}  // namespace device::distributed
