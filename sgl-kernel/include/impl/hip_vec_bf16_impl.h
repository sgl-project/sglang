#pragma once

#if USE_ROCM

#include <hip/hip_bf16.h>
#include <hip/hip_common.h>

// Adapted from flashinfer-rocm [PR#491](https://github.com/flashinfer-ai/flashinfer/pull/491)

using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;

__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 make_bfloat162(const __hip_bfloat16 x, const __hip_bfloat16 y) {
  __hip_bfloat162 t;
  t.x = x;
  t.y = y;
  return t;
}

namespace sgl_hip {

// nv_bfloat16 x 1
template <>
struct vec_t<nv_bfloat16, 1> {
  nv_bfloat16 data;
  SGL_HIP_INLINE nv_bfloat16& operator[](size_t i) {
    return ((nv_bfloat16*)(&data))[i];
  }
  SGL_HIP_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  SGL_HIP_INLINE nv_bfloat16* ptr() {
    return reinterpret_cast<nv_bfloat16*>(&data);
  }
  SGL_HIP_INLINE void load(const nv_bfloat16* ptr);
  SGL_HIP_INLINE void store(nv_bfloat16* ptr) const;
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

SGL_HIP_INLINE void vec_t<nv_bfloat16, 1>::load(const nv_bfloat16* ptr) {
  data = *ptr;
}

SGL_HIP_INLINE void vec_t<nv_bfloat16, 1>::store(nv_bfloat16* ptr) const {
  *ptr = data;
}

// nv_bfloat16 x 2
template <>
struct vec_t<nv_bfloat16, 2> {
  nv_bfloat162 data;

  SGL_HIP_INLINE nv_bfloat16& operator[](size_t i) {
    return ((nv_bfloat16*)(&data))[i];
  }
  SGL_HIP_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  SGL_HIP_INLINE nv_bfloat16* ptr() {
    return reinterpret_cast<nv_bfloat16*>(&data);
  }
  SGL_HIP_INLINE void load(const nv_bfloat16* ptr);
  SGL_HIP_INLINE void store(nv_bfloat16* ptr) const;
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

SGL_HIP_INLINE void vec_t<nv_bfloat16, 2>::load(const nv_bfloat16* ptr) {
  data = *((nv_bfloat162*)ptr);
}

SGL_HIP_INLINE void vec_t<nv_bfloat16, 2>::store(nv_bfloat16* ptr) const {
  *((nv_bfloat162*)ptr) = data;
}

template <>
struct vec_t<nv_bfloat16, 4> {
  uint2 data;

  SGL_HIP_INLINE nv_bfloat16& operator[](size_t i) {
    return ((nv_bfloat16*)(&data))[i];
  }
  SGL_HIP_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  SGL_HIP_INLINE nv_bfloat16* ptr() {
    return reinterpret_cast<nv_bfloat16*>(&data);
  }
  SGL_HIP_INLINE void load(const nv_bfloat16* ptr);
  SGL_HIP_INLINE void store(nv_bfloat16* ptr) const;
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

SGL_HIP_INLINE void vec_t<nv_bfloat16, 4>::load(const nv_bfloat16* ptr) {
  data = *((uint2*)ptr);
}

SGL_HIP_INLINE void vec_t<nv_bfloat16, 4>::store(nv_bfloat16* ptr) const {
  *((uint2*)ptr) = data;
}

// nv_bfloat16 x 8 or more

template <size_t vec_size>
struct vec_t<nv_bfloat16, vec_size> {
  uint4 data[vec_size / 8];

  SGL_HIP_INLINE nv_bfloat16& operator[](size_t i) {
    return ((nv_bfloat16*)data)[i];
  }
  SGL_HIP_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)data)[i];
  }
  SGL_HIP_INLINE nv_bfloat16* ptr() {
    return reinterpret_cast<nv_bfloat16*>(&data);
  }
  SGL_HIP_INLINE void load(const nv_bfloat16* ptr) {
#pragma unoll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }
  SGL_HIP_INLINE void store(nv_bfloat16* ptr) const {
#pragma unoll
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
