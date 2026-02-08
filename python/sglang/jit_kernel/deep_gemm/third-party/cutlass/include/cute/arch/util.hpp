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

#include <cute/config.hpp>
#include <cute/numeric/integer_sequence.hpp>

#if defined(__clang__) && defined(__CUDA__)
  //  __cvta_generic_to_shared was added in Clang 14: https://reviews.llvm.org/D111665
  #if __clang_major__ >= 14
    #define CUTE_CLANG_SUPPORTS_CVTA_GENERIC_TO_SHARED 1
  #endif

  // __nvvm_get_smem_pointer added in Clang 14: https://reviews.llvm.org/D111665
  // ... but will not work on Windows until Clang 15: https://reviews.llvm.org/D122897
  #if (!defined(_WIN32) && __clang_major__ >= 14) || __clang_major__ >= 15
    #define CUTE_CLANG_SUPPORTS_NVVM_GET_SMEM_POINTER 1
  #endif
#endif

#if defined(__NVCC__) || defined(__CUDACC_RTC__)
  // __cvta_generic_to_shared added in CUDA 11+
  #if __CUDACC_VER_MAJOR__ >= 11
    #define CUTE_NVCC_SUPPORTS_CVTA_GENERIC_TO_SHARED 1
  #endif

  // __nvvm_get_smem_pointer added in CUDA 10.2
  #if __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2
    #define CUTE_NVCC_SUPPORTS_NVVM_GET_SMEM_POINTER 1
  #endif
#endif

#if CUTE_NVCC_SUPPORTS_CVTA_GENERIC_TO_SHARED || CUTE_CLANG_SUPPORTS_CVTA_GENERIC_TO_SHARED
  #define CUTE_CVTA_GENERIC_TO_SHARED_SUPPORTED 1
#endif

#if !defined(CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED) && CUTE_CVTA_GENERIC_TO_SHARED_SUPPORTED && defined(__CUDA_ARCH__)
  #define CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED 1
#endif

#if CUTE_NVCC_SUPPORTS_NVVM_GET_SMEM_POINTER || CUTE_CLANG_SUPPORTS_NVVM_GET_SMEM_POINTER
  #define CUTE_NVVM_GET_SMEM_POINTER_SUPPORTED 1
#endif

#if !defined(CUTE_NVVM_GET_SMEM_POINTER_ACTIVATED) && CUTE_NVVM_GET_SMEM_POINTER_SUPPORTED && defined(__CUDA_ARCH__)
  #define CUTE_NVVM_GET_SMEM_POINTER_ACTIVATED 1
#endif

// Clang 14+ provides a declaration of __nvvm_get_smem_pointer, so we only need
// to provide one for NVCC
#if CUTE_NVCC_SUPPORTS_NVVM_GET_SMEM_POINTER
  extern "C" {
  // This NVVM intrinsic is subject to change in future versions of CUDA.
  // Clients should not call it directly.
  CUTE_DEVICE uint32_t __nvvm_get_smem_pointer(void*);
  }
#endif

namespace cute
{

/// CUTE helper to cast SMEM pointer to unsigned
CUTE_HOST_DEVICE
uint32_t
cast_smem_ptr_to_uint(void const* const ptr)
{
// We prefer to use the new CVTA intrinsics if they are available, otherwise we will fall back to
// the previous internal intrinsics if they are available.
#if CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED
  //
  // This NVVM intrinsic converts an address in shared memory to a plain
  // unsigned integer. This is necessary to pass to shared memory instructions
  // in inline PTX.
  //
  // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only available in 10.2].
  //
  //__device__ size_t __cvta_generic_to_shared(void* ptr);

  /// CUTE helper to get SMEM pointer
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

#elif CUTE_NVVM_GET_SMEM_POINTER_ACTIVATED

  return __nvvm_get_smem_pointer(ptr);

#elif defined(__CUDA_ARCH__)

  uint32_t smem_ptr;

  asm(
  "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
    : "=r"(smem_ptr) : "l"(ptr));

  return smem_ptr;

#else


  (void) ptr;
  printf("ERROR: cast_smem_ptr_to_uint not supported but used.\n");
  return 0;

#endif
}

namespace detail {

//
// Wrapper for MMAOp::fma
//

template <class MmaOp>
struct CallFMA {
  template <class... Args>
  CUTE_HOST_DEVICE constexpr void
  operator()(Args&&... args) const {
    return MmaOp::fma(static_cast<Args&&>(args)...);
  }
};

//
// Wrapper for CopyOp::copy
//

template <class CopyOp>
struct CallCOPY {
  template <class... Args>
  CUTE_HOST_DEVICE constexpr void
  operator()(Args&&... args) const {
    return CopyOp::copy(static_cast<Args&&>(args)...);
  }
};

//
// Utility for exploding pointers/arrays/tensors into functions
//

template <class Fn,
          class PtrA, int... I>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrA&& a, int_sequence<I...>)
{
  return fn(a[I]...);
}

template <class Fn,
          class PtrS, int... Is,
          class PtrD, int... Id>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrS&& s, int_sequence<Is...>,
        PtrD&& d, int_sequence<Id...>)
{
  return fn(s[Is]..., d[Id]...);
}

template <class Fn,
          class PtrA, int... Ia,
          class PtrB, int... Ib,
          class PtrC, int... Ic>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrA&& a, int_sequence<Ia...>,
        PtrB&& b, int_sequence<Ib...>,
        PtrC&& c, int_sequence<Ic...>)
{
  return fn(a[Ia]..., b[Ib]..., c[Ic]...);
}

template <class Fn,
          class PtrD, int... Id,
          class PtrA, int... Ia,
          class PtrB, int... Ib,
          class PtrC, int... Ic>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrD&& d, int_sequence<Id...>,
        PtrA&& a, int_sequence<Ia...>,
        PtrB&& b, int_sequence<Ib...>,
        PtrC&& c, int_sequence<Ic...>)
{
  return fn(d[Id]..., a[Ia]..., b[Ib]..., c[Ic]...);
}

template <class Fn,
          class PtrD, int... Id,
          class PtrA, int... Ia,
          class PtrB, int... Ib,
          class PtrC, int... Ic,
          class PtrE, int... Ie>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrD&& d, int_sequence<Id...>,
        PtrA&& a, int_sequence<Ia...>,
        PtrB&& b, int_sequence<Ib...>,
        PtrC&& c, int_sequence<Ic...>,
        PtrE&& e, int_sequence<Ie...>)
{
  return fn(d[Id]..., a[Ia]..., b[Ib]..., c[Ic]..., e[Ie]...);
}

template <class Fn,
          class PtrD, int... Id,
          class PtrA, int... Ia,
          class PtrB, int... Ib,
          class PtrC, int... Ic,
          class PtrE, int... Ie,
          class PtrF, int... If>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrD&& d, int_sequence<Id...>,
        PtrA&& a, int_sequence<Ia...>,
        PtrB&& b, int_sequence<Ib...>,
        PtrC&& c, int_sequence<Ic...>,
        PtrE&& e, int_sequence<Ie...>,
        PtrF&& f, int_sequence<If...>)
{
  return fn(d[Id]..., a[Ia]..., b[Ib]..., c[Ic]..., e[Ie]..., f[If]...);
}

template <class Fn,
          class PtrD, int... Id,
          class PtrA, int... Ia,
          class PtrB, int... Ib,
          class PtrC, int... Ic,
          class PtrE, int... Ie,
          class PtrF, int... If,
          class PtrG, int... Ig>
CUTE_HOST_DEVICE constexpr
void
explode(Fn fn,
        PtrD&& d, int_sequence<Id...>,
        PtrA&& a, int_sequence<Ia...>,
        PtrB&& b, int_sequence<Ib...>,
        PtrC&& c, int_sequence<Ic...>,
        PtrE&& e, int_sequence<Ie...>,
        PtrF&& f, int_sequence<If...>,
        PtrG&& g, int_sequence<Ig...>)
{
  return fn(d[Id]..., a[Ia]..., b[Ib]..., c[Ic]..., e[Ie]..., f[If]..., g[Ig]...);
}

//
// Utility for exploding tuples into functions
//

template <class Fn,
          class TupleA, int... I>
CUTE_HOST_DEVICE constexpr
void
explode_tuple(Fn fn,
              TupleA&& a, int_sequence<I...>)
{
  return fn(get<I>(a)...);
}

template <class Fn,
          class TupleA, int... Ia,
          class TupleB, int... Ib>
CUTE_HOST_DEVICE constexpr
void
explode_tuple(Fn fn,
              TupleA&& a, int_sequence<Ia...>,
              TupleB&& b, int_sequence<Ib...>)
{
  return fn(get<Ia>(a)..., get<Ib>(b)...);
}

template <class Fn,
          class TupleA, int... Ia,
          class TupleB, int... Ib,
          class TupleC, int... Ic>
CUTE_HOST_DEVICE constexpr
void
explode_tuple(Fn fn,
              TupleA&& a, int_sequence<Ia...>,
              TupleB&& b, int_sequence<Ib...>,
              TupleC&& c, int_sequence<Ic...>)
{
  return fn(get<Ia>(a)..., get<Ib>(b)..., get<Ic>(c)...);
}

} // end namespace detail

} // end namespace cute
