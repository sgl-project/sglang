#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/fusion.cuh>
#include <sgl_kernel/distributed/push/row_sum.cuh>

#include <concepts>
#include <cstdint>

namespace sglang::device::distributed::push {

/// The shared memory of an epilogue that reduces nothing over the row: nothing to set up.
struct NoSmem {
  SGL_DEVICE void init() {}
};

/**
 * \brief An epilogue consumes the reduced row, one bf16 vector per thread at `(row, col)` (`col` counts 16-byte
 *        vectors within the row):
 *   - `Params`: its tensors, a member of the kernel's `__grid_constant__` parameters (held by reference);
 *   - `Smem<kClusterSize, kNumWarps>`: its `__shared__` state for a row held by a cluster of `kClusterSize` CTAs of
 *     `kNumWarps` warps, `NoSmem` or a `ClusterRowSum`; the kernel calls its `init()` at kernel start (every thread);
 *   - `T{params, row, col, vid}` (`vid = row * kRowVecs + col`, as the prologue's): issues all work that does not
 *     depend on the sum: the input loads (residual, weights, mixes) and any output that no row writes (e.g.
 *     padding); the kernel constructs it after the push, so this overlaps the exchange;
 *   - `store(x, smem)`: `x` is the all-reduced vector, rounded to bf16 as the unfused all-reduce returns it; with a
 *     `ClusterRowSum`, it first calls `smem.wait_peers()` (row-uniform, also on rows that skip the sums), then
 *     `smem.sum(v, i)` or `smem.rms_norm(..., i)` (at most one call per index, row-uniform) reduces over the row's
 *     threads.
 */
template <typename T>
concept Epilogue = requires(const T obj, const VecStorage<bf16_t>& x, typename T::template Smem<1, 1>& smem) {
  requires std::constructible_from<T, typename T::Params, uint32_t, uint32_t, uint32_t>;
  smem.init();
  obj.store(x, smem);
};

}  // namespace sglang::device::distributed::push
