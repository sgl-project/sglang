#pragma once

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/fusion.cuh>

#include <concepts>
#include <cstdint>

namespace sglang::device::distributed::push {

/**
 * \brief A prologue produces this rank's partial, one bf16 vector per thread:
 *   - `Params`: its tensors, a member of the kernel's `__grid_constant__` parameters (held by reference);
 *   - `T{params, row, col, vid}`: the thread's vector `col` of row `row` (`vid = row * kRowVecs + col`, the
 *     vector's index in the whole tensor);
 *   - `load()`: the partial's vector there.
 */
template <typename T>
concept Prologue = requires(const T obj) {
  requires std::constructible_from<T, typename T::Params, uint32_t, uint32_t, uint32_t>;
  { obj.load() } -> std::same_as<typename T::Vec>;
};

}  // namespace sglang::device::distributed::push
