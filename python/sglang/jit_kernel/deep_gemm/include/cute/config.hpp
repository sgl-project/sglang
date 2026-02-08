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
#pragma once

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#  define CUTE_HOST_DEVICE __forceinline__ __host__ __device__
#  define CUTE_DEVICE      __forceinline__          __device__
#  define CUTE_HOST        __forceinline__ __host__
#else
#  define CUTE_HOST_DEVICE inline
#  define CUTE_DEVICE      inline
#  define CUTE_HOST        inline
#endif // CUTE_HOST_DEVICE, CUTE_DEVICE

#if defined(__CUDACC_RTC__)
#  define CUTE_HOST_RTC CUTE_HOST_DEVICE
#else
#  define CUTE_HOST_RTC CUTE_HOST
#endif

#if !defined(__CUDACC_RTC__) && !defined(__clang__) && \
  (defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA))
#  define CUTE_UNROLL    #pragma unroll
#  define CUTE_NO_UNROLL #pragma unroll 1
#elif defined(__CUDACC_RTC__) || defined(__clang__)
#  define CUTE_UNROLL    _Pragma("unroll")
#  define CUTE_NO_UNROLL _Pragma("unroll 1")
#else
#  define CUTE_UNROLL
#  define CUTE_NO_UNROLL
#endif // CUTE_UNROLL

#if defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA)
#  define CUTE_INLINE_CONSTANT                 static const __device__
#else
#  define CUTE_INLINE_CONSTANT                 static constexpr
#endif

// __grid_constant__ was introduced in CUDA 11.7.
#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 7)))
#  define CUTE_GRID_CONSTANT_SUPPORTED
#endif

// __grid_constant__ can be enabled only on SM70+.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
#  define CUTE_GRID_CONSTANT_ENABLED
#endif

#if ! defined(CUTE_GRID_CONSTANT)
#  if defined(CUTE_GRID_CONSTANT_SUPPORTED) && defined(CUTE_GRID_CONSTANT_ENABLED)
#    define CUTE_GRID_CONSTANT __grid_constant__
#  else
#    define CUTE_GRID_CONSTANT
#  endif
#endif

// Some versions of GCC < 11 have trouble deducing that a
// function with "auto" return type and all of its returns in an "if
// constexpr ... else" statement must actually return.  Thus, GCC
// emits spurious "missing return statement" build warnings.
// Developers can suppress these warnings by using the
// CUTE_GCC_UNREACHABLE macro, which must be followed by a semicolon.
// It's harmless to use the macro for other GCC versions or other
// compilers, but it has no effect.
#if ! defined(CUTE_GCC_UNREACHABLE)
#  if defined(__GNUC__)
#    define CUTE_GCC_UNREACHABLE __builtin_unreachable()
#  else
#    define CUTE_GCC_UNREACHABLE
#  endif
#endif

#if defined(_MSC_VER)
// Provides support for alternative operators 'and', 'or', and 'not'
#  include <ciso646>
#endif // _MSC_VER

#if defined(__CUDACC_RTC__)
#  define CUTE_STL_NAMESPACE cuda::std
#  define CUTE_STL_NAMESPACE_IS_CUDA_STD
#else
#  define CUTE_STL_NAMESPACE std
#endif

//
// Assertion helpers
//

#if defined(__CUDACC_RTC__)
#  include <cuda/std/cassert>
#else
#  include <cassert>
#endif

#define CUTE_STATIC_V(x)            decltype(x)::value

#define CUTE_STATIC_ASSERT          static_assert
#define CUTE_STATIC_ASSERT_V(x,...) static_assert(decltype(x)::value, ##__VA_ARGS__)

// Fail and print a message. Typically used for notification of a compiler misconfiguration.
#if defined(__CUDA_ARCH__)
#  define CUTE_INVALID_CONTROL_PATH(x) assert(0 && x); printf(x); __brkpt()
#else
#  define CUTE_INVALID_CONTROL_PATH(x) assert(0 && x); printf(x)
#endif

//
// IO
//

#if !defined(__CUDACC_RTC__)
#  include <cstdio>
#  include <iostream>
#  include <iomanip>
#endif

//
// Type
//

#if defined(__CUDACC_RTC__)
#  include <cuda/std/cstdint>
#else
#  include <cstdint>
#endif

//
// Debugging utilities
//

#include <cute/util/debug.hpp>
