#pragma once
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cstdio>

namespace device::distributed {

inline constexpr uint32_t kMaxNumGPU = 8;

/// Device-wide nanosecond wall clock. Unlike `clock64()` this is independent of
/// the SM clock, so it can be used to express a real time bound.
SGL_DEVICE uint64_t global_timer_ns() {
  uint64_t ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
  return ns;
}

/// A time bound for the cross-rank rendezvous spins below.
///
/// Both custom all-reduce protocols assume that the all-reduces of one
/// communicator are *totally ordered*, i.e. at most one is in flight at a time,
/// which is exactly what issuing them on a single stream provides. The state
/// that encodes "has my peer arrived?" is per communicator and per block -- a
/// two-stage counter here, and in the push kernel the push buffer itself -- so
/// two all-reduces of one communicator that execute concurrently (e.g. issued
/// from two CUDA streams) alias that state: one kernel's poll phase re-arms the
/// buffer slice the other one is still waiting for, and the non-atomic epoch
/// update can leave the ranks permanently out of step. The rendezvous then never
/// completes. Since these loops spin, that used to manifest as a silent,
/// permanent, unkillable 100%-utilization hang (issue #31117).
///
/// The host side now serializes a second stream's issue behind the first, which
/// removes the hazard for every issue it can see. It cannot see two CUDA graphs
/// replayed concurrently on two streams -- a replay makes no host call -- so this
/// deadline is the backstop for that case: on expiry the kernel traps, turning
/// the hang into an immediate, attributable CUDA error. Override the bound with
/// `SGLANG_CUSTOM_ALL_REDUCE_SPIN_TIMEOUT` (seconds; 0 disables it).
struct SpinDeadline {
 public:
  /// Spin iterations between two clock reads. A healthy rendezvous completes in
  /// far fewer than this, so the common path never reads `%globaltimer` at all:
  /// the clock is only started once a wait is already long enough to be
  /// suspicious. (Reading it eagerly, once per thread, was measurable -- ~1% of
  /// the one-shot push kernel at small message sizes.)
  static constexpr uint32_t kCheckInterval = 4096;

  SGL_DEVICE explicit SpinDeadline(uint64_t timeout_ns)
      : m_timeout_ns(timeout_ns), m_start_ns(0), m_countdown(kCheckInterval) {}

  /// Call once per spin iteration. Never returns once the deadline has passed.
  SGL_DEVICE void tick() {
    if (m_timeout_ns == 0) return;   // bound disabled
    if (--m_countdown != 0) return;  // amortize the clock read
    m_countdown = kCheckInterval;
    const uint64_t now = global_timer_ns();
    if (m_start_ns == 0) {
      m_start_ns = now;  // first suspiciously long wait: start the clock
    } else if (now - m_start_ns > m_timeout_ns) {
      expire();
    }
  }

 private:
  /// Cold path: `noinline` + `noreturn` so the hot loop keeps no state alive
  /// across the call. The trap tears the context down, so the message above it
  /// is the only diagnosis the user gets; one line per stuck warp keeps it
  /// readable.
  ///
  /// NOTE: the message deliberately takes no printf arguments. Passing any would
  /// make the kernel allocate a per-thread stack frame for the varargs buffer
  /// (`STACK:8` in `cuobjdump -res-usage`), and setting that local-memory window
  /// up costs ~45ns on *every* launch of this kernel -- ~1.5% of a small
  /// all-reduce, for a buffer only the dead path ever writes.
  [[noreturn]] __noinline__ __device__ void expire() const {
    const uint32_t lane = threadIdx.x % kWarpThreads;
    if (lane == static_cast<uint32_t>(__ffs(__activemask()) - 1)) {
      printf(
          "[sglang] custom all-reduce timed out waiting for its peers. Two all-reduces of the "
          "SAME communicator were in flight at once -- e.g. two CUDA graphs replayed concurrently "
          "on two streams. The protocol supports only one in-flight all-reduce per communicator; "
          "use one communicator per concurrent stream. (Bound: "
          "SGLANG_CUSTOM_ALL_REDUCE_SPIN_TIMEOUT seconds, 0 disables.)\n");
    }
    __trap();
    __builtin_unreachable();
  }

  uint64_t m_timeout_ns;
  uint64_t m_start_ns;
  uint32_t m_countdown;
};

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
  /// `spin_timeout_ns` bounds the wait for the peers (see `SpinDeadline`).
  template <bool kFence, bool kStart>
  SGL_DEVICE void sync(uint32_t rank, uint32_t num_gpu, uint64_t spin_timeout_ns) const {
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
        SpinDeadline deadline(spin_timeout_ns);
        while (signal.get<kFence>() - base < target)
          deadline.tick();
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
