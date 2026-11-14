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

/**
 * \brief The push plane's phase word: one u64 shared by every kernel on the plane.
 *
 * Protocol (every kernel that uses the plane, on one stream):
 *  - A grid has N participants 0..N-1; the defaults make every block of a 1D grid
 *    one participant. A 2D/3D grid, or a grid where only some blocks take part
 *    (e.g. one per cluster), must pass idx and num explicitly.
 *  - Each participant commits exactly once, on every exit path, after its last read
 *    of the phase. Participant 0 adds 2^32 - (N - 1), the others add 1, so a grid adds
 *    exactly 2^32: the high half gains 1 and the low half returns to 0. The phase is
 *    bit 0 of the high half, so reading it is a 32-bit load and an `& 1`.
 *  - No partial sum carries into the high half, so the phase cannot change before
 *    the last participant commits. A participant may commit only after every thread
 *    it covers has read the phase: put a block barrier before the commit, and a
 *    cluster barrier when other CTAs of the cluster read but do not commit.
 *  - `load` must come after the PDL wait: only completion of the previous grid makes
 *    its commits visible. A kernel that only reads the phase (a producer whose paired
 *    consumer commits) must sit between the previous commit and that consumer.
 *  - Every rank runs the same sequence of grids on the plane, so phases agree across ranks.
 *
 * Nothing checks these rules: a wrong N, or a skipped or doubled commit, silently
 * desynchronises the phase and corrupts later rounds.
 */
struct EpochCounter {
 public:
  EpochCounter(const EpochCounter&) = delete;
  /// The current phase, 0 or 1. A plain load of the high half: it runs after the PDL wait.
  SGL_DEVICE uint32_t load() const {
    return m_half[1] & 1u;
  }
  /// One call per participant, from a single thread; no thread filtering here.
  SGL_DEVICE void commit(uint32_t idx = blockIdx.x, uint32_t num = gridDim.x) {
    ptx::red_add_relaxed_gpu(&m_word, idx == 0 ? (uint64_t{1} << 32) - (num - 1) : uint64_t{1});
  }

 private:
  union {
    uint64_t m_word;
    uint32_t m_half[2];  // little-endian: [1] is the high half
  };
};
static_assert(sizeof(EpochCounter) == 8, "must match the counter allocation in custom_all_reduce_v2.py");

/**
 * \brief One cross-rank barrier slot: one arrival counter per ordering class, padded
 *        to a cache line because the owner's RMWs contend with neighbouring slots.
 *
 * A counter holds arrivals not yet taken by the owning rank. Every rank adds 1 per
 * arrival (unicast or through the multicast alias); the owner then takes `n` (the
 * world size) at once and, if the balance went negative, waits for it to return to
 * >= 0. A peer is at most one barrier ahead, so a balance stays in [-n, n - 1].
 *
 * Counting cannot tell rounds apart: a peer's next arrival may be taken in place of
 * another peer's in-flight one. Release arrivals have their own counter so that a
 * taken stand-in is always a release, whose acquire still orders the missing peer's
 * writes transitively (its release was acquired by the stand-in's sender first).
 * A relaxed stand-in would break that, and relaxed barriers promise no ordering.
 */
struct alignas(128) Semaphore {
 public:
  Semaphore(const Semaphore&) = delete;
  SGL_DEVICE void put_relaxed() {
    ptx::red_add_relaxed_sys(&m_relaxed, 1);
  }
  SGL_DEVICE void put_relaxed_multicast() {
    ptx::multimem_red_add_relaxed(&m_relaxed, 1);
  }
  /// Also publishes this thread's prior writes to the taker.
  SGL_DEVICE void put_release() {
    ptx::red_add_release_sys(&m_release, 1);
  }
  SGL_DEVICE void put_release_multicast() {
    ptx::multimem_red_add_release(&m_release, 1);
  }

  /// Owner side: take `n` relaxed arrivals, returning once all of them have landed.
  SGL_DEVICE void take_relaxed(uint32_t n) {
    if (static_cast<int32_t>(ptx::atomic_add_relaxed_sys(&m_relaxed, -n) - n) >= 0) return;
    while (static_cast<int32_t>(ptx::load_relaxed_sys(&m_relaxed)) < 0)
      ;
  }
  /// Owner side: take `n` release arrivals and acquire the writes they published.
  SGL_DEVICE void take_acquire(uint32_t n) {
    if (static_cast<int32_t>(ptx::atomic_add_acquire_sys(&m_release, -n) - n) >= 0) return;
    while (static_cast<int32_t>(ptx::load_acquire_sys(&m_release)) < 0)
      ;
  }

 private:
  uint32_t m_relaxed;  // arrivals from put_relaxed*
  uint32_t m_release;  // arrivals from put_release*
};
static_assert(sizeof(Semaphore) == 128, "must match _SEMAPHORE_BYTES in custom_all_reduce_v2.py");

/**
 * \brief Kernel-facing slice of a push plane: `[offset, offset + size)` within every
 *        slot, with `slot_bytes` as the stride from one slot to the next.
 *
 * Trivially copyable, so it drops straight into a `__grid_constant__` params struct.
 */
template <uint32_t kWorldSize>
struct PushWorkSpace {
  std::array<uint8_t*, kWorldSize> workspaces;  // every rank's workspace, already offset
  EpochCounter* counter;                        // rank-local, so no mc
  uint8_t* mc_workspace;                        // multicast VA of the local workspace, or null
  uint32_t slot_bytes;                          // stride between slots, not the slice size
};

/// Kernel-facing slice of a pull plane. The semaphores are indexed by block,
/// not by byte, so they pass through the slice unchanged.
template <uint32_t kWorldSize>
struct PullWorkSpace {
  std::array<Semaphore*, kWorldSize> semaphores;  // every rank's semaphore array
  std::array<uint8_t*, kWorldSize> workspaces;    // every rank's workspace, already offset
  Semaphore* mc_semaphore;                        // multicast VA of the local semaphores, or null
  uint8_t* mc_workspace;                          // multicast VA of the local workspace, or null
};

/// The -0.0 bit pattern per element type, which the Lamport protocol writes in place of +0.0.
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

/**
 * \brief The Lamport "slot empty" marker over a vector of N elements of T.
 *
 * The marker is checked per `kAtom`-byte word: an all-zero word means "not arrived
 * yet". Producers rewrite every all-zero word of their payload to -0.0 in its lowest
 * element, which reduces to the same value, so a written word is never 0. Consumers
 * refill the slot with zeros after reading. A word is written by one store, so it
 * never shows half of a payload; kAtom must not exceed the store width.
 */
template <typename T, uint32_t N, uint32_t kAtom = sizeof(T)>
struct LamportTrait {
  static_assert(kAtom >= sizeof(T) && (kAtom == 2 || kAtom == 4 || kAtom == 8));
  static_assert(kAtom % sizeof(T) == 0 && N % (kAtom / sizeof(T)) == 0);
  using Packed = std::conditional_t<kAtom == 2, uint16_t, std::conditional_t<kAtom == 4, uint32_t, uint64_t>>;
  static constexpr Packed kNegZero = FloatTrait<T>::kNegZero;
  static constexpr uint32_t kNumPacked = N / (kAtom / sizeof(T));

  /// Producer side: make every word non-zero without changing the reduced value.
  SGL_DEVICE static void clear_pos_zero(void* val) {
    const auto ptr = static_cast<Packed*>(val);
#pragma unroll
    for (uint32_t i = 0; i < kNumPacked; ++i) {
      if (ptr[i] == 0) ptr[i] = kNegZero;
    }
  }

  /// Consumer side: whether any word still holds the empty marker.
  SGL_DEVICE static bool has_pos_zero(const void* val) {
    const auto ptr = static_cast<const Packed*>(val);
    bool result = false;
#pragma unroll
    for (uint32_t i = 0; i < kNumPacked; ++i) {
      result |= ptr[i] == 0;
    }
    return result;
  }

  /// The empty marker, for the consumer to restore after reading.
  SGL_DEVICE static void fill_pos_zero(void* val) {
    const auto ptr = static_cast<Packed*>(val);
#pragma unroll
    for (uint32_t i = 0; i < kNumPacked; ++i) {
      ptr[i] = 0;
    }
  }
};

/**
 * \brief Per-block barrier across all ranks, through semaphore `idx` of every rank.
 *
 * Stateless: thread `tx < kWorldSize` signals rank tx's slot and thread `rank` takes
 * kWorldSize arrivals from its own. The block with the same `idx` on every rank must
 * make the same sequence of arrivals, and `idx` must be below the semaphore capacity.
 */
template <uint32_t kWorldSize>
struct Barrier {
 public:
  SGL_DEVICE Barrier(Semaphore* const* semaphores, uint32_t rank, uint32_t idx = blockIdx.x)
      : m_rank(rank), m_idx(idx), m_semaphores(semaphores) {}

  template <bool kNeedFence>
  SGL_DEVICE void arrive() const {
    if (const auto tx = threadIdx.x; tx < kWorldSize) {
      const auto semaphore = &m_semaphores[tx][m_idx];
      if constexpr (kNeedFence) {
        semaphore->put_release();
        if (tx == m_rank) semaphore->take_acquire(kWorldSize);
      } else {
        semaphore->put_relaxed();
        if (tx == m_rank) semaphore->take_relaxed(kWorldSize);
      }
    }
  }

  /// Synchronises only; for an arrival that publishes no data.
  SGL_DEVICE void arrive_relaxed() const {
    return this->arrive<false>();
  }

  /// Also makes this block's prior writes visible to every rank after the barrier.
  SGL_DEVICE void arrive_rel_acq() const {
    return this->arrive<true>();
  }

 private:
  uint32_t m_rank;
  uint32_t m_idx;
  Semaphore* const* m_semaphores;
};

/**
 * \brief Picks which half of a push plane's `2 * kWorldSize` slots this round owns.
 *
 * Rank dst's workspace is `[phase][src][slot_bytes]`: slot (dst, src) receives rank
 * src's data in rank dst. The halves alternate because a round leaves its pos-zero
 * markers behind: a peer still draining the previous round must not see them refilled.
 */
template <uint32_t kWorldSize>
struct PushEpoch {
 public:
  SGL_DEVICE PushEpoch(EpochCounter* counter, uint8_t* const* workspaces, uint32_t slot_bytes)
      : m_counter(counter), m_workspaces(workspaces), m_slot_bytes(slot_bytes), m_epoch(counter->load()) {}

  SGL_DEVICE PushEpoch(const PushWorkSpace<kWorldSize>& ws)
      : PushEpoch(ws.counter, ws.workspaces.data(), ws.slot_bytes) {}

  /// Rank `src`'s slot inside rank `dst`'s workspace, for the current epoch.
  SGL_DEVICE void* slot_ptr(uint32_t dst, uint32_t src = 0) const {
    return m_workspaces[dst] + slot_offset(src);
  }

  /// Byte offset of rank `src`'s slot within any rank's workspace (or its multicast VA).
  SGL_DEVICE uint32_t slot_offset(uint32_t src = 0) const {
    const auto epoch_stride_bytes = m_epoch * m_slot_bytes * kWorldSize;
    return src * m_slot_bytes + epoch_stride_bytes;
  }

  /// `EpochCounter::commit` as a block-collective call: every thread of a
  /// participating block calls it, and thread 0 issues the single commit.
  SGL_DEVICE void commit(uint32_t idx = blockIdx.x, uint32_t num = gridDim.x) const {
    if (threadIdx.x == 0) m_counter->commit(idx, num);
  }

 private:
  EpochCounter* m_counter;
  uint8_t* const* m_workspaces;
  uint32_t m_slot_bytes;
  uint32_t m_epoch;  // 0 or 1
};

/*
 * Same protocol as `Barrier`, but a single `multimem.red` reaches every peer's
 * slot at once instead of a `kWorldSize`-wide unicast fan-out. One thread drives
 * the whole barrier, so `world_size` need not be a constant and nothing here has
 * to be a template. Stateless: constructing one at each call site costs nothing.
 */
struct McBarrier {
 public:
  SGL_DEVICE McBarrier(Semaphore* local, Semaphore* mc, uint32_t world_size, uint32_t idx = blockIdx.x)
      : m_world_size(world_size), m_idx(idx), m_local(local), m_mc(mc) {}

  template <bool kNeedFence>
  SGL_DEVICE void arrive() const {
    if (threadIdx.x != 0) return;
    if constexpr (kNeedFence) {
      m_mc[m_idx].put_release_multicast();
      m_local[m_idx].take_acquire(m_world_size);
    } else {
      m_mc[m_idx].put_relaxed_multicast();
      m_local[m_idx].take_relaxed(m_world_size);
    }
  }

  SGL_DEVICE void arrive_relaxed() const {
    return this->arrive<false>();
  }

  SGL_DEVICE void arrive_rel_acq() const {
    return this->arrive<true>();
  }

 private:
  uint32_t m_world_size;
  uint32_t m_idx;
  Semaphore* m_local;
  Semaphore* m_mc;
};

}  // namespace device::distributed

namespace host::distributed {

using device::distributed::EpochCounter, device::distributed::Semaphore;
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
 *        rank-local phase word.
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
      TensorView counter,                  // [sizeof(EpochCounter)]
      intptr_t mc_workspace_ptr,
      uint32_t default_blocks);

  uint32_t default_blocks;                         // launch width for callers without their own; any grid is legal
  int64_t slot_bytes;                              // per-slot bytes; each rank holds 2 * world_size slots
  EpochCounter* counter;                           // rank-local memory
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
 * own, which leaves the corresponding `num_bytes` / `capacity` at zero and
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
      std::vector<TensorView> semaphores,  // world_size * [capacity]
      intptr_t mc_workspace_ptr,
      intptr_t mc_semaphore_ptr,
      uint32_t default_blocks);

  uint32_t capacity;                                 // semaphore count; a grid may use any idx below it
  uint32_t default_blocks;                           // launch width for callers without their own
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

  /*
   * Only the constructors live in csrc/distributed/registry.cuh: kernel
   * modules are separate shared objects that see this header but never link
   * that translation unit, so everything they call has to be inline here.
   */
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

  /*
   * Unset pull width means the plane's `default_blocks`; both are clamped to
   * the semaphore capacity, and the multicast width additionally never exceeds
   * the pull width, since extra multicast traffic costs NVLS throughput.
   */
  void set_pull_blocks(std::optional<uint32_t> num_blocks) {
    m_pull_blocks = num_blocks;
  }
  void set_pull_multicast_blocks(std::optional<uint32_t> num_blocks) {
    m_pull_multicast_blocks = num_blocks;
  }
  uint32_t get_pull_blocks() const {
    const auto& pull = get_pull_obj();
    CHECK_HOST(pull.capacity > 0) << "This pull plane has no semaphores to barrier on";
    return std::min(m_pull_blocks.value_or(pull.default_blocks), pull.capacity);
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
