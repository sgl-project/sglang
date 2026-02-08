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
    \brief Architecture-specific operators on memory added for SM80
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/arch/synclog.hpp"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  #define CUDA_CP_ASYNC_ACTIVATED 1
#else
  #define CUDA_CP_ASYNC_ACTIVATED 0
#endif

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Initiates an asynchronous copy from global memory to shared memory.
///
/// cp.async
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async;

/// Initiates an asynchronous copy from global memory to shared memory. Rather than predicate
/// the entire transfer, zeros are written to SMEM if the guard predicate is false.
///
/// cp.async
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async_zfill;

/// Initiates an asynchronous copy from global memory to shared memory. Rather than predicate
/// the entire transfer, nans (0x7eff) are written to SMEM if the guard predicate is false.
///
/// cp.async
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async_nan;

/// Either 0 or 1 are written to SMEM based on input element type
/// Used for diagonal elements of triangular matrix of BLAS3 functions
///
/// st.shared
///
template <
   /// Type of Element
   typename Element,
   /// If the data is for a Hermitian matrix diagonal
   bool IsHermitianData = false>
struct cp_async_diag;

static const uint32_t OOB_NAN_F16 = 0x7eff;
static const uint32_t OOB_NAN_F16x2 = ((OOB_NAN_F16 << 16) | OOB_NAN_F16);

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Always> {

  /// Copy
  CUTLASS_DEVICE
  cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    #if CUDA_CP_ASYNC_ACTIVATED

      // Make sure the size is supported.
      static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
                "Size is not supported");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
          "  @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;\n"
#else
          "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
#endif
          "}\n" ::"r"((int)pred_guard),
          "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));

    #else
      using AccessType  = Array<uint8_t, SizeInBytes>;

      if (pred_guard) {
        *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
      }
    #endif
  }
};

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Always> {

  /// Copy with zero fill
  CUTLASS_DEVICE
  cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    #if CUDA_CP_ASYNC_ACTIVATED

      // Make sure the size is supported.
      static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
                "Size is not supported");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
      int src_in_bytes = (pred_guard ? SizeInBytes : 0);

      asm volatile(
#if CUTLASS_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
#else
        "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
#endif
        "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));

    #else
      using AccessType  = Array<uint8_t, SizeInBytes>;

      if (pred_guard) {
        *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
      }
      else {
        AccessType zeros;
        zeros.clear();
        *static_cast<AccessType *>(smem_ptr) = zeros;
      }
    #endif
  }
};

/// Partial specialization
template <>
struct cp_async_nan<16, CacheOperation::Always> {
  static int const kSizeInBytes = 16;

  /// Copy with nan fill
  CUTLASS_DEVICE
  cp_async_nan(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    #if CUDA_CP_ASYNC_ACTIVATED

      static __constant__ uint4 OOB_NAN_F16x8 = {OOB_NAN_F16x2, OOB_NAN_F16x2,
                                                 OOB_NAN_F16x2, OOB_NAN_F16x2};

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
          "  @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;\n"
#else
          "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
#endif
          "  @!p st.shared.v4.u32 [%1], {%4, %5, %6, %7};\n"
          "}\n"
          :
          : "r"((int)pred_guard), "r"(smem_int_ptr), "l"(global_ptr),
            "n"(kSizeInBytes), "r"(OOB_NAN_F16x8.x), "r"(OOB_NAN_F16x8.y), "r"(OOB_NAN_F16x8.z),
            "r"(OOB_NAN_F16x8.w));

    #else

      CUTLASS_UNUSED(smem_ptr);
      CUTLASS_UNUSED(global_ptr);
      CUTLASS_UNUSED(pred_guard);
      CUTLASS_NOT_IMPLEMENTED();

    #endif
  }
};

/// Partial specialization to write one (1)
template<typename Element_>
struct cp_async_diag <Element_, false> {
  using Element = Element_;

  CUTLASS_DEVICE
  cp_async_diag(void *smem_ptr) {
    #if CUDA_CP_ASYNC_ACTIVATED

      /// Values for the diagonal elements of the triangular input matrix
      static __constant__ uint2 DIAG_DATA_DOUBLE_ONE = {0x3ff00000, 0x00000000};
      static __constant__ uint1 DIAG_DATA_FLOAT_ONE = {0x3f800000};
      static __constant__ uint1 DIAG_DATA_ZERO = {0x00000000};

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      if (platform::is_same<Element, complex<double>>::value) {
        asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                      : :
                      "r"(smem_int_ptr), "r"(DIAG_DATA_DOUBLE_ONE.y), "r"(DIAG_DATA_DOUBLE_ONE.x),
                      "r"(DIAG_DATA_ZERO.x), "r"(DIAG_DATA_ZERO.x));
      } else if (platform::is_same<Element, complex<float>>::value) {
        asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
                      : :
                      "r"(smem_int_ptr), "r"(DIAG_DATA_FLOAT_ONE.x), "r"(DIAG_DATA_ZERO.x));
      } else if (platform::is_same<Element, double>::value) {
        asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
                      : :
                      "r"(smem_int_ptr), "r"(DIAG_DATA_DOUBLE_ONE.y),"r"(DIAG_DATA_DOUBLE_ONE.x));
      } else if (platform::is_same<Element, float>::value) {
        asm volatile("st.shared.u32 [%0], %1;\n"
                      : :
                      "r"(smem_int_ptr), "r"(DIAG_DATA_FLOAT_ONE.x));
      } else {
        CUTLASS_UNUSED(smem_int_ptr);
        CUTLASS_NOT_IMPLEMENTED();
      }
      
    #else

      CUTLASS_UNUSED(smem_ptr);
      CUTLASS_NOT_IMPLEMENTED();

    #endif
  }
};

/// Partial specialization to write zero for the imaginary part of Hermitian data
template<typename Element_>
struct cp_async_diag <Element_, true> {
  using Element = Element_;

  CUTLASS_DEVICE
  cp_async_diag(void *smem_ptr) {
    #if CUDA_CP_ASYNC_ACTIVATED

      /// Values for the diagonal elements of the triangular input matrix
      static __constant__ uint1 DIAG_DATA_ZERO = {0x00000000};

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      if (platform::is_same<Element, complex<double>>::value) {
        asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
                      : :
                      "r"(smem_int_ptr), "r"(DIAG_DATA_ZERO.x), "r"(DIAG_DATA_ZERO.x));
      } else if (platform::is_same<Element, complex<float>>::value) {
        asm volatile("st.shared.u32 [%0], %1;\n"
                      : :
                      "r"(smem_int_ptr), "r"(DIAG_DATA_ZERO.x));
      } else {
        CUTLASS_UNUSED(smem_int_ptr);
        CUTLASS_NOT_IMPLEMENTED();
      }
      
    #else

      CUTLASS_UNUSED(smem_ptr);
      CUTLASS_NOT_IMPLEMENTED();

    #endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Global> {

  /// Copy
  CUTLASS_DEVICE
  cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    #if CUDA_CP_ASYNC_ACTIVATED

      static_assert(SizeInBytes == 16,
        "cp.async only supports CacheOperation::Global when access size is 16B.");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
      cutlass::arch::synclog_emit_cp_async(__LINE__, smem_int_ptr, global_ptr, pred_guard, SizeInBytes);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
          "  @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
#else
          "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
#endif
          "}\n" ::"r"((int)pred_guard),
          "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));

    #else
      using AccessType  = Array<uint8_t, SizeInBytes>;

      if (pred_guard) {
        *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
      }
    #endif
  }
};

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Global> {

  /// Copy with zero fill
  CUTLASS_DEVICE
  cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    #if CUDA_CP_ASYNC_ACTIVATED

      static_assert(SizeInBytes == 16,
        "cp.async only supports CacheOperation::Global when access size is 16B.");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
      int src_in_bytes = (pred_guard ? SizeInBytes : 0);
      cutlass::arch::synclog_emit_cp_async_zfill(__LINE__, smem_int_ptr, global_ptr, pred_guard, SizeInBytes);

      asm volatile(
#if CUTLASS_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
#else
        "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
#endif
        "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));

    #else
      using AccessType  = Array<uint8_t, SizeInBytes>;

      if (pred_guard) {
        *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
      }
      else {
        AccessType zeros;
        zeros.clear();
        *static_cast<AccessType *>(smem_ptr) = zeros;
      }
    #endif
  }
};

/// Partial specialization
template <>
struct cp_async_nan<16, CacheOperation::Global> {
  static int const kSizeInBytes = 16;

  /// Copy with nan fill
  CUTLASS_DEVICE
  cp_async_nan(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    #if CUDA_CP_ASYNC_ACTIVATED

      static __constant__ uint4 OOB_NAN_F16x8 = {OOB_NAN_F16x2, OOB_NAN_F16x2,
                                                 OOB_NAN_F16x2, OOB_NAN_F16x2};

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
      cutlass::arch::synclog_emit_cp_async_nan(__LINE__, smem_int_ptr, global_ptr, pred_guard);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
          "  @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
#else
          "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
#endif
          "  @!p st.shared.v4.u32 [%1], {%4, %5, %6, %7};\n"
          "}\n"
          :
          : "r"((int)pred_guard), "r"(smem_int_ptr), "l"(global_ptr),
            "n"(kSizeInBytes), "r"(OOB_NAN_F16x8.x), "r"(OOB_NAN_F16x8.y), "r"(OOB_NAN_F16x8.z),
            "r"(OOB_NAN_F16x8.w));

    #else

      CUTLASS_UNUSED(smem_ptr);
      CUTLASS_UNUSED(global_ptr);
      CUTLASS_UNUSED(pred_guard);
      CUTLASS_NOT_IMPLEMENTED();

    #endif
  }
};
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
CUTLASS_DEVICE
void cp_async_fence() {
  #if CUDA_CP_ASYNC_ACTIVATED
  asm volatile("cp.async.commit_group;\n" ::);
  cutlass::arch::synclog_emit_cp_async_fence(__LINE__);
  #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template <int N>
CUTLASS_DEVICE void cp_async_wait() {
  #if CUDA_CP_ASYNC_ACTIVATED
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  cutlass::arch::synclog_emit_cp_async_wait(__LINE__, N);
  #endif
}

/// Blocks until all previous cp.async.commit_group operations have committed.
template <>
CUTLASS_DEVICE void cp_async_wait<0>() {
  #if CUDA_CP_ASYNC_ACTIVATED
  asm volatile("cp.async.wait_all;\n" ::);
  cutlass::arch::synclog_emit_cp_async_wait_all(__LINE__);
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace arch
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
