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
    \brief Sparse matrix multiply accumulate for SM80
*/

#pragma once
#include "cutlass/cutlass.h"
#include CUDA_STD_HEADER(cassert)

#include "mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#if ((__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 1))

#define CUTLASS_ARCH_SPARSE_MMA_SM80_SUPPORTED 1

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED
#endif

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 16832
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct SparseMma<
  gemm::GemmShape<16, 8, 32>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread
> {

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
  uint32_t *D = reinterpret_cast<uint32_t *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
  if (id2 == 0) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "r"(C[0]), "r"(C[1]), "r"(E));
  }
  else if (id2 == 1) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x1;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "r"(C[0]), "r"(C[1]), "r"(E));
  }
  else {
    assert(0);
  }
#else
  if (id2 == 0) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "r"(C[0]), "r"(C[1]), "r"(E));
  }
  else if (id2 == 1) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x1;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "r"(C[0]), "r"(C[1]), "r"(E));
  }
  else {
    assert(0);
  }
#endif

#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct SparseMma<
  gemm::GemmShape<16, 8, 32>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd,
  SPFormatType::Thread
  > {

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
  if (id2 == 0) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
          "r"(E));
  }
  else if (id2 == 1) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
          "r"(E));
  }
  else {
    assert(0);
  }
#else
  if (id2 == 0) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
          "r"(E));
  }
  else if (id2 == 1) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
          "r"(E));
  }
  else {
    assert(0);
  }

#endif

#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 16832 - Float BF16, FP32 accumulation 
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = bf16 * bf16 + F32
template <>
struct SparseMma<gemm::GemmShape<16, 8, 32>, 32, bfloat16_t, layout::RowMajor,
           bfloat16_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd, SPFormatType::Thread> {
  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 2;

  static int const kMaxID2 = 2;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    if (id2 == 0) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
    } else if (id2 == 1) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#else
    if (id2 == 0) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
    } else if (id2 == 1) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#endif

#else

    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 16816 - Float TF32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = tf32 * tf32 + F32
template <>
struct SparseMma<gemm::GemmShape<16, 8, 16>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd, SPFormatType::Thread> {
  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 4>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  static int const kSparse = 2;

  static int const kMetaSizeInBits = 4;

  static int const kMaxID2 = 2;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c, uint32_t const &E, int const id2) const {

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    if (id2 == 0) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
    } else if (id2 == 1) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#else
    if (id2 == 0) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
    } else if (id2 == 1) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x1;\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]), 
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#endif

#else

    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 16864 - S8 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

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

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    if (id2 == 0) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#else
    if (id2 == 0) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#endif

#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

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

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    if (id2 == 0) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#else
    if (id2 == 0) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#endif

#else

    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

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

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    if (id2 == 0) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#else
    if (id2 == 0) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k64.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#endif

#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,64>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,64>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

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

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    if (id2 == 0) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#else
    if (id2 == 0) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k64.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#endif

#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Sparse Matrix Multiply 168128 - S4 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

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

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    if (id2 == 0) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#else
    if (id2 == 0) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#endif

#else

    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

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

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    if (id2 == 0) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.s32.s4.u4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#else
    if (id2 == 0) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.u4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#endif

#else

    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

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

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    if (id2 == 0) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.s32.u4.s4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#else
    if (id2 == 0) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k128.row.col.s32.u4.s4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#endif

#else

    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct SparseMma<
  gemm::GemmShape<16,8,128>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate,
  SPFormatType::Thread> {

  using Shape = gemm::GemmShape<16,8,128>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using FragmentE = uint32_t;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

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

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    if (id2 == 0) {
      asm volatile(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.s32.u4.u4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#else
    if (id2 == 0) {
      asm volatile(
          "mma.sp.sync.aligned.m16n8k128.row.col.s32.u4.u4.s32.satfinite {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
          "{%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
          : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
            "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(E));
    } else {
      assert(0);
    }
#endif

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
