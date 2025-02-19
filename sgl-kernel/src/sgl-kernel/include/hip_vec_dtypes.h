#pragma once

#if USE_ROCM

#include <hip/hip_common.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

// Adapted from flashinfer

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

namespace flashinfer {

template <typename float_t, size_t vec_size>
struct vec_t;

template <typename srcDtype, typename dstDtype, size_t vec_size>
FLASHINFER_INLINE void cast_load_impl(vec_t<dstDtype, vec_size>& dst, const srcDtype* src);

template <typename float_t, size_t vec_size>
struct vec_t {
  FLASHINFER_INLINE float_t& operator[](size_t i);
  FLASHINFER_INLINE const float_t& operator[](size_t i) const;
  FLASHINFER_INLINE float_t* ptr();

  FLASHINFER_INLINE void load(const float_t* ptr);

  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src);
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr);
};

} // namespace flashinfer

// **** impl *****

namespace flashinfer  {

template <typename srcDtype, typename dstDtype, size_t vec_size>
FLASHINFER_INLINE void cast_load_impl(vec_t<dstDtype, vec_size>& dst, const srcDtype* src_ptr) {
  if constexpr (std::is_same<srcDtype, dstDtype>::value) {
    dst.load(src_ptr);
  } else {
    vec_t<srcDtype, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename float_t, size_t vec_size>
template <typename T>
FLASHINFER_INLINE void vec_t<float_t, vec_size>::cast_load(const T* ptr) {
  cast_load_impl(*this, ptr);
}

} // namespace flashinfer

#include "impl/hip_vec_bf16_impl.h"
#include "impl/hip_vec_half_impl.h"
#include "impl/hip_vec_fp32_impl.h"
#endif
