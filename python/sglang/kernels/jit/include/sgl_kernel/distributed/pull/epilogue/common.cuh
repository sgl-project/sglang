#pragma once

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/fusion.cuh>

#include <concepts>
#include <cstdint>

namespace sglang::device::distributed::pull {

/// One row of `kHiddenSize` bf16 per CTA, one 16-byte vector per thread: thread `t` holds the row's vector `t`.
template <uint32_t kHiddenSize>
struct RowTile {
  static constexpr uint32_t kRowVecs = kHiddenSize / 8;
  static constexpr uint32_t kNumThreads = kRowVecs;
  static constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;
  static_assert(kHiddenSize % 256 == 0 && kNumThreads <= 1024, "whole warps, at most 1024 threads");
};

/// `kNumRows` consecutive rows of one group: row `row + u` of the all-reduced tensor, which is row `local + u` of this
/// rank's shard.
struct RowGroup {
  uint32_t row;
  uint32_t local;

  /// The 64-bit index of this thread's vector of row `row + u` in a `[rows, kRowVecs]` tensor: one base plus a
  /// constant per row, which a 32-bit `row + u` (it may wrap) does not let ptxas fold.
  template <uint32_t kRowVecs>
  SGL_DEVICE int64_t vec(uint32_t u) const {
    return int64_t{row} * kRowVecs + int64_t{u} * kRowVecs + threadIdx.x;
  }
  template <uint32_t kRowVecs>
  SGL_DEVICE int64_t local_vec(uint32_t u) const {
    return int64_t{local} * kRowVecs + int64_t{u} * kRowVecs + threadIdx.x;
  }
};

/**
 * \brief A pull epilogue consumes groups of all-reduced rows of this rank's shard:
 *   - `Tile`: its `RowTile`; `kNumSums`: how many row sums it takes per row (each one row-uniform);
 *   - `Params`: its tensors, a member of the kernel's `__grid_constant__` parameters (held by reference);
 *   - `T{params}`: loads what no row changes (the weights), once per kernel;
 *   - `kNumRows` is the rows of one call: a full group, or one tail row;
 *   - `prefetch<kNumRows>(group)`: issues the group's own loads (e.g. the residual) and returns them;
 *   - `store<kNumRows>(group, x, inputs, smem, phase)`: `x[u]` is this thread's vector of row `group.row + u`, the
 *     bf16 all-reduced sum; each sum goes through `smem.publish<kNumRows>(values, phase, index)` once per
 *     `index < kNumSums` (one barrier for the whole group), then `smem.total(u, phase, index)` per row where it is
 *     used; outputs that every rank needs are multicast stores.
 */
template <typename T>
concept Epilogue = requires(const T obj, RowGroup group) {
  typename T::Tile;
  typename T::Params;
  { T::kNumSums } -> std::convertible_to<uint32_t>;
  requires std::constructible_from<T, const typename T::Params&>;
  obj.template prefetch<1>(group);
};

}  // namespace sglang::device::distributed::pull
