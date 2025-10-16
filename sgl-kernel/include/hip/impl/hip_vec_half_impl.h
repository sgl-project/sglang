#pragma once

#if USE_ROCM

#include <hip/hip_common.h>
#include <hip/hip_fp16.h>

// Adapted from flashinfer-rocm [PR#491](https://github.com/flashinfer-ai/flashinfer/pull/491)

using half = __half;
using half2 = __half2;

namespace sgl_hip {

// half x 1
template <>
struct vec_t<half, 1> {
  half data;

  SGL_HIP_INLINE half& operator[](size_t i) {
    return ((half*)(&data))[i];
  }
  SGL_HIP_INLINE const half& operator[](size_t i) const {
    return ((const half*)(&data))[i];
  }
  SGL_HIP_INLINE half* ptr() {
    return reinterpret_cast<half*>(&data);
  }
  SGL_HIP_INLINE void load(const half* ptr);
  SGL_HIP_INLINE void store(half* ptr) const;
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

SGL_HIP_INLINE void vec_t<half, 1>::load(const half* ptr) {
  data = *ptr;
}

SGL_HIP_INLINE void vec_t<half, 1>::store(half* ptr) const {
  *ptr = data;
}

// half x 2
template <>
struct vec_t<half, 2> {
  half2 data;

  SGL_HIP_INLINE half& operator[](size_t i) {
    return ((half*)(&data))[i];
  }
  SGL_HIP_INLINE const half& operator[](size_t i) const {
    return ((const half*)(&data))[i];
  }
  SGL_HIP_INLINE half* ptr() {
    return reinterpret_cast<half*>(&data);
  }
  SGL_HIP_INLINE void load(const half* ptr);
  SGL_HIP_INLINE void store(half* ptr) const;
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

SGL_HIP_INLINE void vec_t<half, 2>::load(const half* ptr) {
  data = *((half2*)ptr);
}

SGL_HIP_INLINE void vec_t<half, 2>::store(half* ptr) const {
  *((half2*)ptr) = data;
}

// half x 4

template <>
struct vec_t<half, 4> {
  uint2 data;

  SGL_HIP_INLINE half& operator[](size_t i) {
    return ((half*)(&data))[i];
  }
  SGL_HIP_INLINE const half& operator[](size_t i) const {
    return ((const half*)(&data))[i];
  }
  SGL_HIP_INLINE half* ptr() {
    return reinterpret_cast<half*>(&data);
  }
  SGL_HIP_INLINE void load(const half* ptr);
  SGL_HIP_INLINE void store(half* ptr) const;
  template <typename T>
  SGL_HIP_INLINE void cast_from(const vec_t<T, 4>& src) {
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

SGL_HIP_INLINE void vec_t<half, 4>::load(const half* ptr) {
  data = *((uint2*)ptr);
}

SGL_HIP_INLINE void vec_t<half, 4>::store(half* ptr) const {
  *((uint2*)ptr) = data;
}

// half x 8 or more

template <size_t vec_size>
struct vec_t<half, vec_size> {
  uint4 data[vec_size / 8];

  SGL_HIP_INLINE half& operator[](size_t i) {
    return ((half*)data)[i];
  }
  SGL_HIP_INLINE const half& operator[](size_t i) const {
    return ((const half*)data)[i];
  }
  SGL_HIP_INLINE half* ptr() {
    return reinterpret_cast<half*>(&data);
  }
  SGL_HIP_INLINE void load(const half* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  SGL_HIP_INLINE void store(half* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4*)ptr)[i] = data[i];
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
