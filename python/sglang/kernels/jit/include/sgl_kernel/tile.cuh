/// \file tile.cuh
/// \brief Tiled memory access helpers for coalesced global memory I/O.
///
/// `tile::Memory<T>` represents a contiguous memory region where multiple
/// threads cooperatively load/store elements. The three factory methods
/// determine the thread group:
/// - `thread()` - single thread (no tiling).
/// - `warp()`   - all threads in a warp cooperate.
/// - `cta()`    - all threads in the CTA cooperate.

#pragma once
#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace device::tile {

/**
 * \brief Represents a contiguous memory region for cooperative tiled access.
 *
 * Each instance is parameterized by an element type `T` and bound to a
 * specific thread id (`tid`) within a group of `tsize` threads.
 *
 * \tparam T The storage element type (e.g. `AlignedVector<packed_t<float>, 4>`).
 */
template <typename T>
struct Memory {
 public:
  SGL_DEVICE constexpr Memory(uint32_t tid, uint32_t tsize) : tid(tid), tsize(tsize) {}
  /// \brief Create a Memory accessor for a single thread (no cooperation).
  SGL_DEVICE static constexpr Memory thread() {
    return Memory{0, 1};
  }
  /// \brief Create a Memory accessor distributed across warp threads.
  SGL_DEVICE static Memory warp(int warp_threads = kWarpThreads) {
    return Memory{static_cast<uint32_t>(threadIdx.x % warp_threads), static_cast<uint32_t>(warp_threads)};
  }
  /// \brief Create a Memory accessor distributed across all CTA threads.
  SGL_DEVICE static Memory cta(int cta_threads = blockDim.x) {
    return Memory{static_cast<uint32_t>(threadIdx.x), static_cast<uint32_t>(cta_threads)};
  }
  /// \brief Load one element from `ptr` at the position assigned to this thread.
  /// \param ptr  Base pointer (cast to `const T*`).
  /// \param offset  Optional tile offset (multiplied by `tsize`).
  SGL_DEVICE T load(const void* ptr, int64_t offset = 0) const {
    return static_cast<const T*>(ptr)[tid + offset * tsize];
  }
  /// \brief Store one element to `ptr` at the position assigned to this thread.
  SGL_DEVICE void store(void* ptr, T val, int64_t offset = 0) const {
    static_cast<T*>(ptr)[tid + offset * tsize] = val;
  }
  /// \brief Check whether this thread's element index is within bounds.
  SGL_DEVICE bool in_bound(int64_t element_count, int64_t offset = 0) const {
    return tid + offset * tsize < element_count;
  }

 private:
  uint32_t tid;
  uint32_t tsize;
};

}  // namespace device::tile
