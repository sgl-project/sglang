#pragma once

#include <sgl_kernel/utils.cuh>

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace sglang::device {

template <typename T>
concept GenericAddress = std::is_pointer_v<T>;

template <typename T>
concept SmemAddress = GenericAddress<T> || std::integral<T>;

template <typename T>
concept MBarAddress = SmemAddress<T> && (std::integral<T> || std::same_as<T, uint64_t*>);

}  // namespace sglang::device
