// A collective's output taken row by row while the collective is still running
// (NVLinkComm::all_reduce_stream_pull): the order the rows arrive in, and how a
// warp takes one row. The producer's side is `StreamMarker`.

#pragma once

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>

#include <cstdint>

namespace sglang::device::distributed {

/**
 * \brief The rows of a streamed collective in the order they arrive.
 *
 * Rank r delivers its share, [r * avg + min(r, rem), ...), in ascending order,
 * and all ranks at about the same pace: position p stands for row p / world of
 * rank p % world. Rows that are already there are one rank's.
 */
struct StreamRowOrder {
  /// What `row_at` returns for a position that stands for no row.
  static constexpr uint32_t kNoRow = 0xFFFFFFFFu;

  uint32_t world;
  uint32_t avg;
  uint32_t rem;

  SGL_DEVICE StreamRowOrder(uint32_t num_rows, uint32_t world_size)
      : world(world_size), avg(num_rows / world_size), rem(num_rows % world_size) {}

  /// Positions to walk; with a ragged split some stand for no row.
  SGL_DEVICE uint32_t size() const {
    return (avg + (rem != 0 ? 1 : 0)) * world;
  }

  /// The row at `pos`, or `kNoRow` when its rank has no such row.
  SGL_DEVICE uint32_t row_at(uint32_t pos) const {
    const auto rank = pos % world;
    const auto idx = pos / world;
    if (idx >= avg + (rank < rem ? 1 : 0)) return kNoRow;
    return rank * avg + (rank < rem ? rank : rem) + idx;
  }
};

/**
 * \brief Moves one row of the stream into `dst` (shared memory) and leaves the
 * marker in its place for the next collective.
 *
 * The row may not have arrived yet, or only in part, and only its own vectors
 * say so: poll a few of them, copy the row, and copy again for as long as the
 * copy holds a marker. For a whole warp; lane 0 copies on `bar`, an mbarrier of
 * one arrival that no other warp uses, whose phase is `parity`.
 */
template <uint32_t kRowVecs>
SGL_DEVICE void take_stream_row(void* row, void* dst, uint64_t* bar, uint32_t& parity) {
  using vec_t = AlignedVector<uint32_t, 4>;  // 16 bytes
  static_assert(kRowVecs % kWarpThreads == 0, "each lane owns the same number of row vectors");
  constexpr uint32_t kRowBytes = kRowVecs * sizeof(vec_t);
  const auto lane_id = threadIdx.x % kWarpThreads;
  vec_t vec;
  bool has_empty;
  do {
    do {
      ptx::ld_relaxed_16B(vec, row, lane_id * (kRowVecs / kWarpThreads));
    } while (__any_sync(0xffffffffu, StreamMarker::is_empty(vec.data())));
    has_empty = false;
    if (lane_id == 0) {
      ptx::mbar_arrive_expect_tx(bar, kRowBytes);
      ptx::cp_async_bulk_g2s(dst, row, kRowBytes, bar);
    }
    ptx::mbar_wait_parity(bar, parity);
    for (uint32_t v = lane_id; v < kRowVecs; v += kWarpThreads) {
      vec.load(dst, v);
      has_empty |= StreamMarker::is_empty(vec.data());
    }
    parity ^= 1;
  } while (__any_sync(0xffffffffu, has_empty));
  StreamMarker::fill(vec.data());
  for (uint32_t v = lane_id; v < kRowVecs; v += kWarpThreads) {
    ptx::st_global_16B(vec, row, v);
  }
}

}  // namespace sglang::device::distributed
