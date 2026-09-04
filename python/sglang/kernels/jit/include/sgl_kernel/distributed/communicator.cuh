#pragma once
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace sglang {

namespace device::distributed {

inline constexpr uint32_t kMaxWorldSize = 16;

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

/// One block's arrival slot: a flag peers increment plus a phase counter.
/// Padded to a cache line so neighbouring blocks never share one.
struct alignas(128) Semaphore {
 public:
  Semaphore(const Semaphore&) = delete;
  SGL_DEVICE Counter* counter_ptr() {
    return &m_counter;
  }
  SGL_DEVICE uint32_t get_relaxed() const {
    return ptx::load_relaxed_sys(&m_flag);
  }
  SGL_DEVICE void put_relaxed() {
    ptx::red_add_relaxed_sys(&m_flag, 1);
  }
  SGL_DEVICE void put_relaxed_multicast() {
    ptx::multimem_red_add_relaxed(&m_flag, 1);
  }
  SGL_DEVICE uint32_t get_acquire() const {
    return ptx::load_acquire_sys(&m_flag);
  }
  SGL_DEVICE void put_release() {
    ptx::red_add_release_sys(&m_flag, 1);
  }
  SGL_DEVICE void put_release_multicast() {
    ptx::multimem_red_add_release(&m_flag, 1);
  }

 private:
  uint32_t m_flag;
  Counter m_counter;
};
static_assert(sizeof(Semaphore) == 128, "must match _SEMAPHORE_BYTES in custom_all_reduce_v2.py");

/// Kernel-facing slice of a push plane: `[offset, offset + size)` within every
/// slot, with `slot_bytes` as the stride from one slot to the next. Trivially
/// copyable, so it drops straight into a `__grid_constant__` params struct.
template <uint32_t kWorldSize>
struct PushWorkSpace {
  std::array<uint8_t*, kWorldSize> workspaces;
  Counter* counter;  // NOTE: this is a local tensor, so no mc
  uint8_t* mc_workspace;
  uint32_t slot_bytes;
};

/// Kernel-facing slice of a pull plane. The semaphores are indexed by block,
/// not by byte, so they pass through the slice unchanged.
template <uint32_t kWorldSize>
struct PullWorkSpace {
  std::array<Semaphore*, kWorldSize> semaphores;
  std::array<uint8_t*, kWorldSize> workspaces;
  Semaphore* mc_semaphore;
  uint8_t* mc_workspace;
};

/// Bit patterns of the lamport "slot empty" marker. A producer rewrites any
/// +0.0 in its payload to -0.0 (numerically identical for the reduction) so a
/// consumer can treat a remaining +0.0 as "not arrived yet".

template <typename T>
struct FloatTrait {};

template <>
struct FloatTrait<bf16_t> {
  using type = uint16_t;
  static constexpr uint16_t kNegZero = 0x8000u;
};

template <>
struct FloatTrait<fp16_t> {
  using type = uint16_t;
  static constexpr uint16_t kNegZero = 0x8000u;
};

template <>
struct FloatTrait<float> {
  using type = uint32_t;
  static constexpr uint32_t kNegZero = 0x80000000u;
};

template <typename T, uint32_t N, uint32_t kAtom = sizeof(T)>
struct LamportTrait {
  static_assert(kAtom >= sizeof(T) && (kAtom == 2 || kAtom == 4 || kAtom == 8));
  static_assert(kAtom % sizeof(T) == 0 && N % (kAtom / sizeof(T)) == 0);
  using Packed = std::conditional_t<kAtom == 2, uint16_t, std::conditional_t<kAtom == 4, uint32_t, uint64_t>>;
  static constexpr Packed kNegZero = FloatTrait<T>::kNegZero;
  static constexpr uint32_t kNumPacked = N / (kAtom / sizeof(T));

  SGL_DEVICE static void clear_pos_zero(void* val) {
    const auto ptr = static_cast<Packed*>(val);
#pragma unroll
    for (uint32_t i = 0; i < kNumPacked; ++i) {
      if (ptr[i] == 0) ptr[i] = kNegZero;
    }
  }

  SGL_DEVICE static bool has_pos_zero(const void* val) {
    const auto ptr = static_cast<const Packed*>(val);
    bool result = false;
#pragma unroll
    for (uint32_t i = 0; i < kNumPacked; ++i) {
      result |= ptr[i] == 0;
    }
    return result;
  }

  SGL_DEVICE static void fill_pos_zero(void* val) {
    const auto ptr = static_cast<Packed*>(val);
#pragma unroll
    for (uint32_t i = 0; i < kNumPacked; ++i) {
      ptr[i] = 0;
    }
  }
};

template <uint32_t kWorldSize>
struct Barrier {
 public:
  SGL_DEVICE Barrier(Semaphore* const* semaphores, uint32_t rank, uint32_t num_arrives)
      : m_counter(0), m_rank(rank), m_semaphores(semaphores) {
    const auto counter = semaphores[rank][blockIdx.x].counter_ptr();
    const auto signal = num_arrives * kWorldSize;
    m_counter = threadIdx.x == rank ? counter->inc(signal) : 0;
  }

  template <bool kNeedFence>
  SGL_DEVICE void arrive(uint32_t n) const {
    if (const auto tx = threadIdx.x; tx < kWorldSize) {
      const auto bx = blockIdx.x;
      const auto semaphore = &m_semaphores[tx][bx];
      const auto current = m_counter + n * kWorldSize;
      if constexpr (kNeedFence) {
        semaphore->put_release();
        if (tx == m_rank) {
          while (semaphore->get_acquire() - current < kWorldSize)
            ;
        }
      } else {
        semaphore->put_relaxed();
        if (tx == m_rank) {
          while (semaphore->get_relaxed() - current < kWorldSize)
            ;
        }
      }
    }
  }

  SGL_DEVICE void arrive_relaxed(uint32_t n) const {
    return this->arrive<false>(n);
  }

  SGL_DEVICE void arrive_rel_acq(uint32_t n) const {
    return this->arrive<true>(n);
  }

 private:
  uint32_t m_counter;
  uint32_t m_rank;
  Semaphore* const* m_semaphores;
};

/// Picks which half of a push plane's `2 * kWorldSize` slots this round owns.
/// The halves alternate because a round leaves its pos-zero markers behind: a
/// peer still draining the previous round must not see them refilled.
template <uint32_t kWorldSize>
struct PushEpoch {
 public:
  SGL_DEVICE PushEpoch(Counter* counter, uint8_t* const* workspaces, uint32_t slot_bytes)
      : m_counter(counter), m_workspaces(workspaces), m_slot_bytes(slot_bytes), m_epoch(m_counter[blockIdx.x].get()) {}

  SGL_DEVICE PushEpoch(const PushWorkSpace<kWorldSize>& ws)
      : PushEpoch(ws.counter, ws.workspaces.data(), ws.slot_bytes) {}

  /// Rank `src`'s slot inside rank `dst`'s workspace, for the current epoch.
  SGL_DEVICE void* slot_ptr(uint32_t dst, uint32_t src = 0) const {
    const auto epoch_stride_bytes = (m_epoch & 1) * m_slot_bytes * kWorldSize;
    return m_workspaces[dst] + src * m_slot_bytes + epoch_stride_bytes;
  }

  SGL_DEVICE uint32_t slot_offset(uint32_t src = 0) const {
    const auto epoch_stride_bytes = (m_epoch & 1) * m_slot_bytes * kWorldSize;
    return src * m_slot_bytes + epoch_stride_bytes;
  }

  SGL_DEVICE void flip() const {
    if (threadIdx.x == 0) m_counter[blockIdx.x].set(m_epoch ^ 1);
  }

  SGL_DEVICE void unsafe_flip_at(uint32_t bx) const {
    m_counter[bx].set(m_epoch ^ 1);
  }

  SGL_DEVICE void unsafe_flip_range(uint32_t start, uint32_t finish) const {
    for (uint32_t idx = start + threadIdx.x; idx < finish; idx += blockDim.x) {
      m_counter[idx].set(m_epoch ^ 1);
    }
  }

 private:
  Counter* m_counter;
  uint8_t* const* m_workspaces;
  uint32_t m_slot_bytes;
  uint32_t m_epoch;
};

/// Same window protocol as `Barrier`, but a single `multimem.red` reaches every
/// peer's row at once instead of a `kWorldSize`-wide unicast fan-out. One
/// thread drives the whole barrier, so `world_size` need not be a constant and
/// nothing here has to be a template.
///
/// Construction only reserves the window; signalling happens in `arrive`. Keep
/// them separate at the call site: the reservation is worth hoisting above a
/// PDL wait (it takes the RMW latency off the post-wait critical path) while
/// the signal must stay after it, since it asserts the producer grid flushed.
struct McBarrier {
 public:
  SGL_DEVICE McBarrier(Semaphore* local, Semaphore* mc, uint32_t world_size, uint32_t num_arrives)
      : m_counter(0), m_world_size(world_size), m_local(local), m_mc(mc) {
    if (threadIdx.x == 0) {
      m_counter = local[blockIdx.x].counter_ptr()->inc(num_arrives * world_size);
    }
  }

  template <bool kNeedFence>
  SGL_DEVICE void arrive(uint32_t n) const {
    return arrive_at<kNeedFence>(m_local, m_mc, m_world_size, m_counter + n * m_world_size);
  }

  SGL_DEVICE void arrive_relaxed(uint32_t n) const {
    return this->arrive<false>(n);
  }

  SGL_DEVICE void arrive_rel_acq(uint32_t n) const {
    return this->arrive<true>(n);
  }

  /// The reserved window base, valid in thread 0. A kernel whose body is
  /// register-hungry enough that keeping this object live would spill can park
  /// this one word in shared memory and finish through `arrive_at` instead.
  SGL_DEVICE uint32_t window() const {
    return m_counter;
  }

  /// `arrive` against a window reserved earlier, without holding the object.
  /// `window` already includes the `n * world_size` offset.
  template <bool kNeedFence>
  SGL_DEVICE static void arrive_at(Semaphore* local, Semaphore* mc, uint32_t world_size, uint32_t window) {
    if (threadIdx.x != 0) return;
    const auto bx = blockIdx.x;
    const auto semaphore = &local[bx];
    const auto mc_semaphore = &mc[bx];
    if constexpr (kNeedFence) {
      mc_semaphore->put_release_multicast();
      while (semaphore->get_acquire() - window < world_size)
        ;
    } else {
      mc_semaphore->put_relaxed_multicast();
      while (semaphore->get_relaxed() - window < world_size)
        ;
    }
  }

 private:
  uint32_t m_counter;  // window base; meaningful in thread 0, the sole poller
  uint32_t m_world_size;
  Semaphore* m_local;
  Semaphore* m_mc;
};

}  // namespace device::distributed

namespace host::distributed {

using device::distributed::Counter, device::distributed::Semaphore;
using device::distributed::PullWorkSpace, device::distributed::PushWorkSpace;
using TensorView = tvm::ffi::TensorView;
template <typename T>
using Optional = tvm::ffi::Optional<T>;

inline constexpr uint32_t kMaxWorldSize = device::distributed::kMaxWorldSize;

/// Multicast VAs are optional, so keep null null instead of forming
/// `nullptr + offset` when slicing a plane that has no multicast mapping.
inline uint8_t* offset_mc(uint8_t* base, int64_t offset) {
  return base != nullptr ? base + offset : nullptr;
}

/// Identity shared by every plane; the constructor is the one place that
/// validates it (defined in csrc/distributed/registry.cuh).
struct BasePlane {
  BasePlane(const BasePlane&) = delete;
  BasePlane(uint32_t rank, uint32_t world_size);
  uint32_t rank;
  uint32_t world_size;
};

/**
 * \brief Lamport push plane: a zero-filled symmetric workspace plus a
 *        rank-local phase counter.
 *
 * Each rank owns `2 * world_size` slots of `slot_bytes` (two phases x one
 * producer slot per peer). Producers store into the destination rank's slot
 * and consumers poll for the pos-zero marker to clear, so the workspace MUST
 * be zero-filled before first use and every kernel MUST restore the marker
 * on its way out. `counter` is rank-local (never read by a peer).
 *
 * Holds no storage: the caller (Python) owns the tensors and their lifetime.
 */
struct PushPlaneObj : public tvm::ffi::Object, BasePlane {
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.distributed.PushPlane", PushPlaneObj, tvm::ffi::Object);

  // Defined in csrc/distributed/registry.cuh (only the registration module needs the implementation).
  PushPlaneObj(
      uint32_t rank,
      uint32_t world_size,
      std::vector<TensorView> workspaces,  // world_size * [2 * world_size][slot_bytes]
      TensorView counter,                  // [num_blocks]
      intptr_t mc_workspace_ptr);

  uint32_t num_blocks;                             // bound to the counter array, hence not tunable
  int64_t slot_bytes;                              // per-slot bytes; each rank holds 2 * world_size slots
  Counter* counter;                                // rank-local memory
  std::array<uint8_t*, kMaxWorldSize> workspaces;  // symmetric memory
  uint8_t* mc_workspace;                           // multicast VA of the local workspace (may be null)

  template <uint32_t N>
  PushWorkSpace<N> get_workspace(int64_t size, int64_t offset = 0) const {
    CHECK_HOST(N == world_size) << "Plane holds " << world_size << " ranks, asked for " << N;
    CHECK_HOST(size >= 0 && offset >= 0 && offset + size <= slot_bytes)
        << "slice [" << offset << ", " << offset + size << ") escapes the " << slot_bytes << "-byte push slot";
    // Device-side phase striding is 32-bit: the largest offset a kernel forms
    // is `(2 * N - 1) * slot_bytes`, so the whole double-buffered plane must fit.
    CHECK_HOST(2 * N * slot_bytes <= std::numeric_limits<uint32_t>::max())
        << 2 * N * slot_bytes << " bytes of push plane exceeds the 32-bit offset range";
    PushWorkSpace<N> ws{{}, counter, offset_mc(mc_workspace, offset), static_cast<uint32_t>(slot_bytes)};
    for (uint32_t i = 0; i < N; ++i) {
      ws.workspaces[i] = workspaces[i] + offset;
    }
    return ws;
  }
};

/**
 * \brief Pull plane: symmetric per-rank buffers plus the per-block barrier
 *        semaphores that guard them.
 *
 * Either half may be absent -- pass a 0-element tensor for the one you do not
 * own, which leaves the corresponding `num_bytes` / `num_blocks` at zero and
 * makes any kernel needing it fail with a clear message:
 *
 *  - workspaces only: nothing today, but the shape a caller who brings its own
 *    symmetric buffers would take.
 *  - semaphores only: the K3 fused collectives, which reduce in place on the
 *    caller's own symmetric input (its multicast VA arrives per call, since it
 *    varies with the slice) and borrow this plane purely to barrier on.
 *  - both: the generic custom all-reduce, whose callers hand it plain tensors,
 *    so it stages them through `workspaces` before reducing.
 *
 * Holds no storage: the caller (Python) owns the tensors and their lifetime.
 */
struct PullPlaneObj : public tvm::ffi::Object, BasePlane {
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.distributed.PullPlane", PullPlaneObj, tvm::ffi::Object);

  // Defined in csrc/distributed/registry.cuh (only the registration module needs the implementation).
  PullPlaneObj(
      uint32_t rank,
      uint32_t world_size,
      std::vector<TensorView> workspaces,  // world_size * [num_bytes]
      std::vector<TensorView> semaphores,  // world_size * [num_blocks]
      intptr_t mc_workspace_ptr,
      intptr_t mc_semaphore_ptr);

  uint32_t num_blocks;                               // semaphore capacity; callers clamp their grid to it
  int64_t num_bytes;                                 // per-rank workspace bytes
  std::array<Semaphore*, kMaxWorldSize> semaphores;  // symmetric memory
  std::array<uint8_t*, kMaxWorldSize> workspaces;    // symmetric memory
  Semaphore* mc_semaphore;                           // multicast VA of the local semaphores (may be null)
  uint8_t* mc_workspace;                             // multicast VA of the local workspace (may be null)

  template <uint32_t N>
  PullWorkSpace<N> get_workspace(int64_t size, int64_t offset = 0) const {
    CHECK_HOST(N == world_size) << "Plane holds " << world_size << " ranks, asked for " << N;
    CHECK_HOST(size >= 0 && offset >= 0 && offset + size <= num_bytes)
        << "slice [" << offset << ", " << offset + size << ") escapes the " << num_bytes << "-byte pull workspace";
    PullWorkSpace<N> ws{{}, {}, mc_semaphore, offset_mc(mc_workspace, offset)};
    for (uint32_t i = 0; i < N; ++i) {
      ws.semaphores[i] = semaphores[i];
      ws.workspaces[i] = workspaces[i] + offset;
    }
    return ws;
  }
};

SGLANG_REGISTER_FFI_REFERENCE_CLASS(PushPlaneRef, PushPlaneObj);
SGLANG_REGISTER_FFI_REFERENCE_CLASS(PullPlaneRef, PullPlaneObj);

/**
 * \brief The planes every kernel in this directory takes, plus the launch
 *        widths that are tuning rather than capacity.
 *
 * A plane is absent when the owner never uses that half (a push-only instance
 * passes `pull=None`), and asking for it then fails with a clear message
 * instead of silently reading a placeholder buffer.
 */
struct CommunicatorObj : public tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.distributed.Communicator", CommunicatorObj, tvm::ffi::Object);
  static constexpr bool _type_mutable = true;  // the set_pull_* knobs below

  // Only the constructors live in csrc/distributed/registry.cuh: kernel
  // modules are separate shared objects that see this header but never link
  // that translation unit, so everything they call has to be inline here.
  CommunicatorObj(Optional<PushPlaneRef> push, Optional<PullPlaneRef> pull);

  uint32_t get_rank() const {
    return m_push.has_value() ? m_push.value()->rank : m_pull.value()->rank;
  }
  uint32_t get_world_size() const {
    return m_push.has_value() ? m_push.value()->world_size : m_pull.value()->world_size;
  }

  bool has_push() const {
    return m_push.has_value();
  }
  bool has_pull() const {
    return m_pull.has_value();
  }
  /// The planes as handles, for callers that hold on to one (Python). Kernels
  /// want `get_*_obj()` below, which skips the refcount traffic.
  Optional<PushPlaneRef> get_push() const {
    return m_push;
  }
  Optional<PullPlaneRef> get_pull() const {
    return m_pull;
  }
  const PushPlaneObj& get_push_obj() const {
    CHECK_HOST(m_push.has_value()) << "This communicator has no push plane";
    return *m_push.value().get();
  }
  const PullPlaneObj& get_pull_obj() const {
    CHECK_HOST(m_pull.has_value()) << "This communicator has no pull plane";
    return *m_pull.value().get();
  }

  /// Both knobs only ever narrow the grid: unset means "the plane's capacity",
  /// and the multicast width additionally never exceeds the pull width, since
  /// extra multicast traffic costs NVLS throughput.
  void set_pull_blocks(std::optional<uint32_t> num_blocks) {
    m_pull_blocks = num_blocks;
  }
  void set_pull_multicast_blocks(std::optional<uint32_t> num_blocks) {
    m_pull_multicast_blocks = num_blocks;
  }
  uint32_t get_pull_blocks() const {
    const auto capacity = get_pull_obj().num_blocks;
    CHECK_HOST(capacity > 0) << "This pull plane has no semaphores to barrier on";
    return m_pull_blocks.has_value() ? std::min(*m_pull_blocks, capacity) : capacity;
  }
  uint32_t get_pull_multicast_blocks() const {
    const auto blocks = get_pull_blocks();
    return m_pull_multicast_blocks.has_value() ? std::min(*m_pull_multicast_blocks, blocks) : blocks;
  }

 private:
  std::optional<uint32_t> m_pull_blocks;
  std::optional<uint32_t> m_pull_multicast_blocks;
  Optional<PushPlaneRef> m_push;
  Optional<PullPlaneRef> m_pull;
};

SGLANG_REGISTER_FFI_REFERENCE_CLASS(CommunicatorRef, CommunicatorObj);

}  // namespace host::distributed

}  // namespace sglang
