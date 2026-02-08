/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Helper macros for the CUTLASS library
*/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////


#ifdef CUTLASS_NAMESPACE
#define concat_tok(a, b) a ## b
#define mkcutlassnamespace(pre, ns) concat_tok(pre, ns)
#define cutlass mkcutlassnamespace(cutlass_, CUTLASS_NAMESPACE)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define CUTLASS_HOST_DEVICE __forceinline__ __device__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE inline
#define CUTLASS_DEVICE inline
#endif

#if ! defined(_MSC_VER)
#define CUTLASS_LAMBDA_FUNC_INLINE __attribute__((always_inline))
#else
#define CUTLASS_LAMBDA_FUNC_INLINE [[msvc::forceinline]]
#endif

#define CUTLASS_HOST __host__
#define CUTLASS_GLOBAL __global__ static

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
CUTLASS_HOST_DEVICE void __CUTLASS_UNUSED(T const &) 
{ }

#if defined(__GNUC__)
  #define CUTLASS_UNUSED(expr) __CUTLASS_UNUSED(expr)
#else
  #define CUTLASS_UNUSED(expr) do { ; } while (&expr != &expr)
#endif

#ifdef _MSC_VER
// Provides support for alternative operators 'and', 'or', and 'not'
#include <ciso646>
#endif // _MSC_VER

#if !defined(__CUDACC_RTC__)
#include <cassert>
#endif

#if defined(__CUDA_ARCH__)
  #if defined(_MSC_VER)
    #define CUTLASS_NOT_IMPLEMENTED() { printf("%s not implemented\n", __FUNCSIG__); asm volatile ("brkpt;\n"); }
  #else
    #define CUTLASS_NOT_IMPLEMENTED() { printf("%s not implemented\n", __PRETTY_FUNCTION__); asm volatile ("brkpt;\n"); }
  #endif
#else
  #if defined(_MSC_VER)
    #define CUTLASS_NOT_IMPLEMENTED() assert(0 && __FUNCSIG__)
  #else
    #define CUTLASS_NOT_IMPLEMENTED() assert(0 && __PRETTY_FUNCTION__)
  #endif
#endif

// CUTLASS_CMATH_NAMESPACE is the namespace where code can find
// <cmath> functions like isnan and log.  Such functions are in
// the std namespace in host code, but in the global namespace
// in device code.
//
// The intended use case for this macro is in "using" declarations
// for making argument-dependent lookup (ADL) work in generic code.
// For example, if T is cutlass::half_t, the following code will
// invoke cutlass::isnan(half_t).  If T is float, it will invoke
// std::isnan on host and ::isnan on device.  (CUTLASS's support
// for NVRTC prevents it from using things in the std namespace
// in device code.)  Correct use of "using" declarations can help
// avoid unexpected implicit conversions, like from half_t to float.
//
// template<class T>
// bool foo(T x) {
//   using CUTLASS_CMATH_NAMESPACE :: isnan;
//   return isnan(x);
// }
//
// Without this macro, one would need to write the following.
//
// template<class T>
// bool foo(T x) {
// #if defined(__CUDA_ARCH__)
//   using ::isnan;
// #else
//   using std::isnan;
// #endif
//   return isnan(x);
// }

#if defined(__CUDA_ARCH__)
#  define CUTLASS_CMATH_NAMESPACE
#else
#  define CUTLASS_CMATH_NAMESPACE std
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {


#ifndef CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED
#define CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED 0
#endif


// CUDA 10.1 introduces the mma instruction
#if !defined(CUTLASS_ENABLE_TENSOR_CORE_MMA)
#define CUTLASS_ENABLE_TENSOR_CORE_MMA 0
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CUTLASS_ASSERT(x) assert(x)

////////////////////////////////////////////////////////////////////////////////////////////////////

// CUTLASS_PRAGMA_(UNROLL|NO_UNROLL) optimization directives for the CUDA compiler.
#if defined(__CUDA_ARCH__) && !defined(__INTELLISENSE__)
  #if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
    #define CUTLASS_PRAGMA_UNROLL _Pragma("unroll")
    #define CUTLASS_PRAGMA_NO_UNROLL _Pragma("unroll 1")
  #else
    #define CUTLASS_PRAGMA_UNROLL #pragma unroll
    #define CUTLASS_PRAGMA_NO_UNROLL #pragma unroll 1
  #endif

  #define CUTLASS_GEMM_LOOP CUTLASS_PRAGMA_NO_UNROLL

#else

    #define CUTLASS_PRAGMA_UNROLL
    #define CUTLASS_PRAGMA_NO_UNROLL
    #define CUTLASS_GEMM_LOOP

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__CUDACC_RTC__)
#define CUTLASS_THREAD_LOCAL thread_local
#else
#define CUTLASS_THREAD_LOCAL
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(_MSVC_LANG)
#  define CUTLASS_CPLUSPLUS _MSVC_LANG
#else
#  define CUTLASS_CPLUSPLUS __cplusplus
#endif

// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/n4762.pdf
// Section 14.8 Predefined macro names
#if (201703L <= CUTLASS_CPLUSPLUS)
#define CUTLASS_CONSTEXPR_IF_CXX17 constexpr
#define CUTLASS_CXX17_OR_LATER 1
#else
#define CUTLASS_CONSTEXPR_IF_CXX17
#define CUTLASS_CXX17_OR_LATER 0
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// __CUDA_ARCH_SPECIFIC__ is introduced in CUDA 12.9
#if !defined(CUDA_ARCH_CONDITIONAL)

#if defined(__CUDA_ARCH_SPECIFIC__)
#define CUDA_ARCH_CONDITIONAL(ARCH_XXYY) (__CUDA_ARCH_SPECIFIC__ == ARCH_XXYY)
#else
#define CUDA_ARCH_CONDITIONAL(ARCH_XXYY) (false)
#endif

#endif

// __CUDA_ARCH_FAMILY_SPECIFIC__ is introduced in CUDA 12.9
#if !defined(CUDA_ARCH_FAMILY)

#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
#define CUDA_ARCH_FAMILY(ARCH_XXYY) (__CUDA_ARCH_FAMILY_SPECIFIC__ == ARCH_XXYY)
#else
#define CUDA_ARCH_FAMILY(ARCH_XXYY) (false)
#endif

#endif

#if !defined(CUDA_ARCH_CONDITIONAL_OR_FAMILY)
#define CUDA_ARCH_CONDITIONAL_OR_FAMILY(ARCH_XXYY) \
  (CUDA_ARCH_CONDITIONAL(ARCH_XXYY) || CUDA_ARCH_FAMILY(ARCH_XXYY))
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

}; // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
