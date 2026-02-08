/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Type traits for common CUDA types
*/

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "cutlass/numeric_types.h"
#include "cutlass/complex.h"

namespace cutlass {
struct half_t;

template <typename T>
struct TypeTraits {
  typedef T host_type;
  typedef T device_type;
  static inline T remove_negative_zero(T x) { return x; }
  static inline T to_print(T x) { return x; }
  static inline device_type to_device(host_type x) { return x; }
};

template <>
struct TypeTraits<int8_t> {
  static cudaDataType_t const cublas_type = CUDA_R_8I;
  typedef int8_t host_type;
  typedef int8_t device_type;
  typedef int8_t integer_type;
  typedef uint8_t unsigned_type;
  static inline int8_t remove_negative_zero(int8_t x) { return x; }
  static inline int to_print(int8_t x) { return (int)x; }
  static inline device_type to_device(host_type x) { return x; }
};

template <>
struct TypeTraits<uint8_t> {
  static cudaDataType_t const cublas_type = CUDA_R_8I;
  typedef uint8_t host_type;
  typedef uint8_t device_type;
  typedef uint8_t integer_type;
  typedef uint8_t unsigned_type;
  static inline uint8_t remove_negative_zero(uint8_t x) { return x; }
  static inline uint32_t to_print(uint8_t x) { return (uint32_t)x; }
  static inline device_type to_device(host_type x) { return x; }
};

template <>
struct TypeTraits<int> {
  static cudaDataType_t const cublas_type = CUDA_R_32I;
  typedef int host_type;
  typedef int device_type;
  typedef int32_t integer_type;
  typedef uint32_t unsigned_type;
  static inline int32_t remove_negative_zero(int32_t x) { return x; }
  static inline int to_print(int x) { return x; }
  static inline device_type to_device(host_type x) { return x; }
};

template <>
struct TypeTraits<unsigned> {
  static cudaDataType_t const cublas_type = CUDA_R_32I;
  typedef unsigned host_type;
  typedef unsigned device_type;
  typedef uint32_t integer_type;
  typedef uint32_t unsigned_type;
  static inline uint32_t remove_negative_zero(uint32_t x) { return x; }
  static inline uint32_t to_print(uint32_t x) { return x; }
  static inline device_type to_device(host_type x) { return x; }
};

template <>
struct TypeTraits<int64_t> {
  static cudaDataType_t const cublas_type = CUDA_R_8I;
  typedef int64_t host_type;
  typedef int64_t device_type;
  typedef int64_t integer_type;
  typedef uint64_t unsigned_type;
  static inline int64_t remove_negative_zero(int64_t x) { return x; }
  static inline int64_t to_print(int64_t x) { return x; }
  static inline device_type to_device(host_type x) { return x; }
};

template <>
struct TypeTraits<uint64_t> {
  static cudaDataType_t const cublas_type = CUDA_R_8I;
  typedef uint64_t host_type;
  typedef uint64_t device_type;
  typedef uint64_t integer_type;
  typedef uint64_t unsigned_type;
  static inline uint64_t remove_negative_zero(uint64_t x) { return x; }
  static inline uint64_t to_print(uint64_t x) { return x; }
  static inline device_type to_device(host_type x) { return x; }
};

template <>
struct TypeTraits<half_t> {
  static cudaDataType_t const cublas_type = CUDA_R_16F;
  typedef half_t host_type;
  typedef half_t device_type;
  typedef int16_t integer_type;
  typedef uint16_t unsigned_type;
  static inline half_t remove_negative_zero(half_t x) {
    return (x.raw() == 0x8000 ? half_t::bitcast(0) : x);
  }
  static inline half_t to_print(half_t x) { return x; }
  static inline device_type to_device(half_t x) { return reinterpret_cast<device_type const &>(x); }
};

template <>
struct TypeTraits<float> {
  static cudaDataType_t const cublas_type = CUDA_R_32F;
  typedef float host_type;
  typedef float device_type;
  typedef int32_t integer_type;
  typedef uint32_t unsigned_type;
  static inline float remove_negative_zero(float x) { return x == -0.f ? 0.f : x; }
  static inline float to_print(float x) { return x; }
  static inline device_type to_device(host_type x) { return x; }
};

template <>
struct TypeTraits<double> {
  static cudaDataType_t const cublas_type = CUDA_R_64F;
  typedef double host_type;
  typedef double device_type;
  typedef int64_t integer_type;
  typedef uint64_t unsigned_type;
  static inline double remove_negative_zero(double x) { return x == -0.0 ? 0.0 : x; }
  static inline double to_print(double x) { return x; }
  static inline device_type to_device(host_type x) { return x; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Complex types
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct TypeTraits<complex<half> > {
  static cudaDataType_t const cublas_type = CUDA_C_16F;
  typedef complex<half_t> host_type;
  typedef complex<half> device_type;
  typedef int16_t integer_type;
  typedef uint16_t unsigned_type;
  static inline device_type to_device(complex<half> x) { return reinterpret_cast<device_type const &>(x); }
};

template <>
struct TypeTraits<complex<half_t> > {
  static cudaDataType_t const cublas_type = CUDA_C_16F;
  typedef complex<half_t> host_type;
  typedef complex<half> device_type;
  typedef int16_t integer_type;
  typedef uint16_t unsigned_type;
  static inline complex<half_t> remove_negative_zero(complex<half_t> x) {
    return complex<half_t>(
      real(x) == -0_hf ? 0_hf : real(x),
      imag(x) == -0_hf ? 0_hf : imag(x)
    );
  }
  static inline complex<half_t> to_print(complex<half_t> x) { return x; }
  static inline device_type to_device(complex<half_t> x) { return reinterpret_cast<device_type const &>(x); }
};

template <>
struct TypeTraits<complex<float> > {

  static cudaDataType_t const cublas_type = CUDA_C_32F;
  typedef complex<float> host_type;
  typedef complex<float> device_type;
  typedef int64_t integer_type;
  typedef uint64_t unsigned_type;

  static inline complex<float> remove_negative_zero(complex<float> x) {
    return complex<float>(
      real(x) == -0.f ? 0.f : real(x),
      imag(x) == -0.f ? 0.f : imag(x)
    );
  }

  static inline complex<float> to_print(complex<float> x) { return x; }
  static inline device_type to_device(complex<float> x) { return reinterpret_cast<device_type const &>(x); }
};

template <>
struct TypeTraits<complex<double> > {
  static cudaDataType_t const cublas_type = CUDA_C_64F;
  typedef complex<double> host_type;
  typedef complex<double> device_type;
  struct integer_type { int64_t real, imag; };
  struct unsigned_type { uint64_t real, imag; };
  static inline complex<double> remove_negative_zero(complex<double> x) {
    return complex<double>(
      real(x) == -0.0 ? 0.0 : real(x),
      imag(x) == -0.0 ? 0.0 : imag(x)
    );
  }
  static inline complex<double> to_print(complex<double> x) { return x; }
  static inline device_type to_device(complex<double> x) { return reinterpret_cast<device_type const &>(x); }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
