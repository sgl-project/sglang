#pragma once

#if USE_ROCM

#include <hip/hip_common.h>

// Adapted from flashinfer-rocm [PR#491](https://github.com/flashinfer-ai/flashinfer/pull/491)

namespace flashinfer {

template <>
struct vec_t<float, 1> {
  float data;

  FLASHINFER_INLINE float& operator[](size_t i) {
    return ((float*)(&data))[i];
  }
  FLASHINFER_INLINE const float& operator[](size_t i) const {
    return ((const float*)(&data))[i];
  }
  FLASHINFER_INLINE float* ptr() {
    return reinterpret_cast<float*>(&data);
  }
  FLASHINFER_INLINE void load(const float* ptr);
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
};

FLASHINFER_INLINE void vec_t<float, 1>::load(const float* ptr) {
  data = *ptr;
}

// float x 2

template <>
struct vec_t<float, 2> {
  float2 data;

  FLASHINFER_INLINE float& operator[](size_t i) {
    return ((float*)(&data))[i];
  }
  FLASHINFER_INLINE const float& operator[](size_t i) const {
    return ((const float*)(&data))[i];
  }
  FLASHINFER_INLINE float* ptr() {
    return reinterpret_cast<float*>(&data);
  }
  FLASHINFER_INLINE void load(const float* ptr);
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
};

FLASHINFER_INLINE void vec_t<float, 2>::load(const float* ptr) {
  data = *((float2*)ptr);
}

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  float4 data[vec_size / 4];

  FLASHINFER_INLINE float& operator[](size_t i) {
    return ((float*)(data))[i];
  }
  FLASHINFER_INLINE const float& operator[](size_t i) const {
    return ((const float*)(data))[i];
  }
  FLASHINFER_INLINE float* ptr() {
    return reinterpret_cast<float*>(&data);
  }
  FLASHINFER_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
};

}  // namespace flashinfer

#endif
