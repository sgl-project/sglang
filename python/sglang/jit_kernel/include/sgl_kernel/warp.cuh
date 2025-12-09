#pragma once
#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace device::warp {

namespace details {

template <std::size_t kUnit>
inline constexpr auto get_mem_package() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
  }
}

inline constexpr auto default_unit_size(std::size_t x) -> std::size_t {
  if (x % (16 * kWarpThreads) == 0) return 16;
  if (x % (8 * kWarpThreads) == 0) return 8;
  if (x % (4 * kWarpThreads) == 0) return 4;
  return 0;  // trigger static assert in _get_mem_package
}

template <std::size_t kBytes, std::size_t kUnit>
using mem_package_t = decltype(get_mem_package<kUnit>());

template <typename T, std::size_t N>
struct storage_vec {
  T data[N];
};

__always_inline __device__ auto load_nc(const uint1* __restrict__ src) -> uint1 {
  uint32_t tmp;
  asm volatile("ld.global.cs.b32 %0,[%1];" : "=r"(tmp) : "l"(src));
  return uint1{tmp};
}

__always_inline __device__ auto load_nc(const uint2* __restrict__ src) -> uint2 {
  uint32_t tmp0, tmp1;
  asm volatile("ld.global.cs.v2.b32 {%0,%1},[%2];" : "=r"(tmp0), "=r"(tmp1) : "l"(src));
  return uint2{tmp0, tmp1};
}

__always_inline __device__ auto load_nc(const uint4* __restrict__ src) -> uint4 {
  uint32_t tmp0, tmp1, tmp2, tmp3;
  asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3},[%4];" : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3) : "l"(src));
  return uint4{tmp0, tmp1, tmp2, tmp3};
}

__always_inline __device__ void store_nc(uint1* __restrict__ dst, const uint1& value) {
  uint32_t tmp = value.x;
  asm volatile("st.global.cs.b32 [%0],%1;" ::"l"(dst), "r"(tmp));
}

__always_inline __device__ void store_nc(uint2* __restrict__ dst, const uint2& value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  asm volatile("st.global.cs.v2.b32 [%0],{%1,%2};" ::"l"(dst), "r"(tmp0), "r"(tmp1));
}

__always_inline __device__ void store_nc(uint4* __restrict__ dst, const uint4& value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  uint32_t tmp2 = value.z;
  uint32_t tmp3 = value.w;
  asm volatile("st.global.cs.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(dst), "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
}

}  // namespace details

template <
    std::size_t kBytes,
    std::size_t kUnit = details::default_unit_size(kBytes),
    std::size_t kThreads = ::device::kWarpThreads>
__always_inline __device__ void copy(void* __restrict__ dst, const void* __restrict__ src) {
  using Package = details::mem_package_t<kBytes, kUnit>;
  constexpr auto kBytesPerLoop = sizeof(Package) * kThreads;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0, "kBytes must be multiple of 128 bytes");

  const auto dst_packed = static_cast<Package*>(dst);
  const auto src_packed = static_cast<const Package*>(src);
  const auto lane_id = threadIdx.x % kThreads;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kThreads + lane_id;
    dst_packed[j] = src_packed[j];
  }
}

template <
    std::size_t kBytes,
    std::size_t kUnit = details::default_unit_size(kBytes),
    std::size_t kThreads = ::device::kWarpThreads>
__always_inline __device__ auto load_vec(const void* __restrict__ src) {
  using Package = details::mem_package_t<kBytes, kUnit>;
  constexpr auto kBytesPerLoop = sizeof(Package) * kThreads;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0, "kBytes must be multiple of 128 bytes");

  const auto src_packed = static_cast<const Package*>(src);
  const auto lane_id = threadIdx.x % kThreads;
  details::storage_vec<Package, kLoopCount> vec;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kThreads + lane_id;
    vec.data[i] = details::load_nc(src_packed + j);
  }

  return vec;
}

template <
    std::size_t kBytes,
    std::size_t kUnit = details::default_unit_size(kBytes),
    std::size_t kThreads = ::device::kWarpThreads,
    typename Tp>
__always_inline __device__ void store_vec(void* __restrict__ dst, const Tp& vec) {
  using Package = details::mem_package_t<kBytes, kUnit>;
  constexpr auto kBytesPerLoop = sizeof(Package) * kThreads;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0, "kBytes must be multiple of 128 bytes");
  static_assert(std::is_same_v<Tp, details::storage_vec<Package, kLoopCount>>);

  const auto dst_packed = static_cast<Package*>(dst);
  const auto lane_id = threadIdx.x % kThreads;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kThreads + lane_id;
    details::store_nc(dst_packed + j, vec.data[i]);
  }
}

}  // namespace device::warp
