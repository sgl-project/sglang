/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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

template <typename srcDtype, typename dstDtype, size_t vec_size>
FLASHINFER_INLINE void cast_store_impl(dstDtype* dst_ptr,
                                       const vec_t<srcDtype, vec_size>& src);

template <typename float_t, size_t vec_size>
struct vec_t {
  FLASHINFER_INLINE float_t& operator[](size_t i);
  FLASHINFER_INLINE const float_t& operator[](size_t i) const;
  FLASHINFER_INLINE float_t* ptr();

  FLASHINFER_INLINE void load(const float_t* ptr);
  FLASHINFER_INLINE void store(float_t* ptr) const;

  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src);
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr);
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const;
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

template <typename srcDtype, typename dstDtype, size_t vec_size>
FLASHINFER_INLINE void cast_store_impl(dstDtype* dst_ptr,
                                       const vec_t<srcDtype, vec_size>& src) {
  if constexpr (std::is_same<srcDtype, dstDtype>::value) {
    src.store(dst_ptr);
  } else {
    vec_t<dstDtype, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}

template <typename float_t, size_t vec_size>
template <typename T>
FLASHINFER_INLINE void vec_t<float_t, vec_size>::cast_load(const T* ptr) {
  cast_load_impl(*this, ptr);
}

template <typename float_t, size_t vec_size>
template <typename T>
FLASHINFER_INLINE void vec_t<float_t, vec_size>::cast_store(T* ptr) const {
  cast_store_impl(ptr, *this);
}

} // namespace flashinfer

#include "impl/hip_vec_bf16_impl.h"
#include "impl/hip_vec_half_impl.h"
#include "impl/hip_vec_fp32_impl.h"
#endif
