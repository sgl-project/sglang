/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Sparse matrix multiply accumulate for SM89
*/

#pragma once
#include "cutlass/cutlass.h"
#include CUDA_STD_HEADER(cassert)

#include "mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4)
#  define CUTLASS_ARCH_SPARSE_MMA_F32_SM89_SUPPORTED
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#  if defined(CUTLASS_ARCH_SPARSE_MMA_F32_SM89_SUPPORTED)
#    define CUTLASS_ARCH_SPARSE_MMA_F32_SM89_ENABLED
#  endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = fe4m3 * fe4m3 + F32
template <typename Operator_>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  cutlass::float_e4m3_t,
  layout::RowMajor,
  cutlass::float_e4m3_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  Operator_,
  SPFormatType::Thread> {

  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 16>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using FragmentE = uint32_t;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {

#if defined(CUTLASS_ARCH_SPARSE_MMA_F32_SM89_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

      if (id2 == 0) {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
      }
      else {
        assert(0);
      }
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = fe4m3 * fe5m2 + F32
template <typename Operator_>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  cutlass::float_e4m3_t,
  layout::RowMajor,
  cutlass::float_e5m2_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  Operator_,
  SPFormatType::Thread> {

  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e5m2_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 16>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using FragmentE = uint32_t;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {

#if defined(CUTLASS_ARCH_SPARSE_MMA_F32_SM89_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

      if (id2 == 0) {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
      }
      else {
        assert(0);
      }
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = fe5m2 * fe4m3 + F32
template <typename Operator_>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  cutlass::float_e5m2_t,
  layout::RowMajor,
  cutlass::float_e4m3_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  Operator_,
  SPFormatType::Thread> {

  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = cutlass::float_e5m2_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 16>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using FragmentE = uint32_t;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {

#if defined(CUTLASS_ARCH_SPARSE_MMA_F32_SM89_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

      if (id2 == 0) {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
      }
      else {
        assert(0);
      }
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = fe5m2 * fe5m2 + F32
template <typename Operator_>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  cutlass::float_e5m2_t,
  layout::RowMajor,
  cutlass::float_e5m2_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  Operator_,
  SPFormatType::Thread> {

  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = cutlass::float_e5m2_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e5m2_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 16>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using FragmentE = uint32_t;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 1;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c,
    uint32_t const &E,
    int const id2
  ) const {

#if defined(CUTLASS_ARCH_SPARSE_MMA_F32_SM89_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

      if (id2 == 0) {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
      }
      else {
        assert(0);
      }
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
