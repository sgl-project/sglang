/// \file vec.cuh
/// \brief Aligned vector types for coalesced global memory access.
///
/// `AlignedVector<T, N>` wraps `N` elements of type `T` in a naturally
/// aligned struct so that the compiler emits wide (vectorized) load/store
/// instructions (e.g. `LDG.128`). The maximum supported vector width is
/// 256 bits (32 bytes), matching CUDA's widest vector load.

#pragma once
#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>

namespace device {

namespace details {

/// \brief Maps byte-width to the corresponding unsigned integer type.
template <std::size_t N>
struct uint_trait {};

template <>
struct uint_trait<1> {
  using type = uint8_t;
};

template <>
struct uint_trait<2> {
  using type = uint16_t;
};

template <>
struct uint_trait<4> {
  using type = uint32_t;
};

template <>
struct uint_trait<8> {
  using type = uint64_t;
};

/// \brief Alias: maps `sizeof(T)` to matching unsigned int type.
template <typename T>
using sized_int = typename uint_trait<sizeof(T)>::type;

}  // namespace details

/// \brief Raw aligned storage for `N` elements of type `T`.
template <typename T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedStorage {
  T data[N];
};

/**
 * \brief Aligned vector for vectorized memory access on GPU.
 *
 * Stores `N` elements of type `T` with natural alignment so that a single
 * `load`/`store` call compiles to a wide memory transaction.
 *
 * \tparam T Element type (e.g. `fp16_t`, `bf16_t`, `float`).
 * \tparam N Number of elements. Must be a power of two and
 *           `sizeof(T) * N <= 32` (256 bits).
 *
 * Example:
 * \code
 *   AlignedVector<fp16_t, 8> vec;  // 16 bytes, 128-bit aligned
 *   vec.load(input_ptr, tid);      // vectorized load
 *   vec[0] = vec[0] + 1;
 *   vec.store(output_ptr, tid);    // vectorized store
 * \endcode
 */
template <typename T, std::size_t N>
struct AlignedVector {
 private:
  /// NOTE: N must be a power of two and sizeof(T) * N <= 32 bytes (256 bits)
  static_assert((N > 0 && (N & (N - 1)) == 0) && sizeof(T) * N <= 32, "CUDA only supports at most 256-bit vector op");
  using element_t = typename details::sized_int<T>;
  using storage_t = AlignedStorage<element_t, N>;

 public:
  /// \brief Vectorized load from `ptr` at the given element `offset`.
  template <typename U>
  SGL_DEVICE void load(const U* ptr, std::size_t offset = 0) {
    static_assert(std::is_same_v<U, T> || std::is_same_v<U, void>);
    m_storage = reinterpret_cast<const storage_t*>(ptr)[offset];
  }
  /// \brief Vectorized store to `ptr` at the given element `offset`.
  template <typename U>
  SGL_DEVICE void store(U* ptr, std::size_t offset = 0) const {
    static_assert(std::is_same_v<U, T> || std::is_same_v<U, void>);
    reinterpret_cast<storage_t*>(ptr)[offset] = m_storage;
  }
  /// \brief Fill all N elements with the same `value`.
  SGL_DEVICE void fill(T value) {
    const auto store_value = *reinterpret_cast<element_t*>(&value);
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      m_storage.data[i] = store_value;
    }
  }

  SGL_DEVICE auto operator[](std::size_t idx) -> T& {
    return reinterpret_cast<T*>(&m_storage)[idx];
  }
  SGL_DEVICE auto operator[](std::size_t idx) const -> T {
    return reinterpret_cast<const T*>(&m_storage)[idx];
  }
  SGL_DEVICE auto data() -> T* {
    return reinterpret_cast<T*>(&m_storage);
  }
  SGL_DEVICE auto data() const -> const T* {
    return reinterpret_cast<const T*>(&m_storage);
  }

 private:
  storage_t m_storage;
};

}  // namespace device
