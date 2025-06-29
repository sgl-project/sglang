#pragma once

#if USE_ROCM

#include <hip/hip_common.h>

// Adapted from flashinfer-rocm [PR#491](https://github.com/flashinfer-ai/flashinfer/pull/491)

namespace sgl_hip {

template <>
struct vec_t<float, 1> {
  float data;

  SGL_HIP_INLINE float& operator[](size_t i) {
    return ((float*)(&data))[i];
  }
  SGL_HIP_INLINE const float& operator[](size_t i) const {
    return ((const float*)(&data))[i];
  }
  SGL_HIP_INLINE float* ptr() {
    return reinterpret_cast<float*>(&data);
  }
  SGL_HIP_INLINE void load(const float* ptr);
  SGL_HIP_INLINE void store(float* ptr) const;
  template <typename T>
  SGL_HIP_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  SGL_HIP_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  SGL_HIP_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

SGL_HIP_INLINE void vec_t<float, 1>::load(const float* ptr) {
  data = *ptr;
}

SGL_HIP_INLINE void vec_t<float, 1>::store(float* ptr) const {
  *ptr = data;
}

// float x 2

template <>
struct vec_t<float, 2> {
  float2 data;

  SGL_HIP_INLINE float& operator[](size_t i) {
    return ((float*)(&data))[i];
  }
  SGL_HIP_INLINE const float& operator[](size_t i) const {
    return ((const float*)(&data))[i];
  }
  SGL_HIP_INLINE float* ptr() {
    return reinterpret_cast<float*>(&data);
  }
  SGL_HIP_INLINE void load(const float* ptr);
  SGL_HIP_INLINE void store(float* ptr) const;
  template <typename T>
  SGL_HIP_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  SGL_HIP_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  SGL_HIP_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

SGL_HIP_INLINE void vec_t<float, 2>::load(const float* ptr) {
  data = *((float2*)ptr);
}

SGL_HIP_INLINE void vec_t<float, 2>::store(float* ptr) const {
  *((float2*)ptr) = data;
}

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  float4 data[vec_size / 4];

  SGL_HIP_INLINE float& operator[](size_t i) {
    return ((float*)(data))[i];
  }
  SGL_HIP_INLINE const float& operator[](size_t i) const {
    return ((const float*)(data))[i];
  }
  SGL_HIP_INLINE float* ptr() {
    return reinterpret_cast<float*>(&data);
  }
  SGL_HIP_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  SGL_HIP_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  SGL_HIP_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  SGL_HIP_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  SGL_HIP_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

}  // namespace sgl_hip

#endif
