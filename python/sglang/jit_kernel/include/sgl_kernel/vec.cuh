#pragma once
#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>

namespace device {

namespace details {

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

template <typename T>
using sized_int = typename uint_trait<sizeof(T)>::type;

}  // namespace details

template <typename T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedStorage {
  T data[N];
};

template <typename T, std::size_t N>
struct AlignedVector {
 private:
  /// NOTE: 1. must be pow of two 2. 16 * 8 = 128 byte, which is the max vector size supported by most devices
  static_assert((N > 0 && (N & (N - 1)) == 0) && sizeof(T) * N <= 16, "CUDA only support at most 128B vector op");
  using element_t = typename details::sized_int<T>;
  using storage_t = AlignedStorage<element_t, N>;

 public:
  template <typename U>
  SGL_DEVICE void load(const U* ptr, std::size_t offset = 0) {
    static_assert(std::is_same_v<U, T> || std::is_same_v<U, void>);
    m_storage = reinterpret_cast<const storage_t*>(ptr)[offset];
  }
  template <typename U>
  SGL_DEVICE void store(U* ptr, std::size_t offset = 0) const {
    static_assert(std::is_same_v<U, T> || std::is_same_v<U, void>);
    reinterpret_cast<storage_t*>(ptr)[offset] = m_storage;
  }
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
