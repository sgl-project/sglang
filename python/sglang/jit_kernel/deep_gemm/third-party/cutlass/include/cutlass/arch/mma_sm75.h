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
    \brief Matrix multiply for SM75
*/

#pragma once
#include "cutlass/cutlass.h"
#include CUDA_STD_HEADER(cassert)

#include "cutlass/arch/wmma.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
// CUDA Toolkit includes for nvcuda::wmma needed for binarized matrix multiply.
#include <mma.h>
#include "cutlass/wmma_array.h"
#endif

// CUTLASS includes
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

////////////////////////////////////////////////////////////////////////////////

#if ((__CUDACC_VER_MAJOR__ > 10) || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))

#define CUTLASS_ARCH_MMA_SM75_SUPPORTED 1

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
#define CUTLASS_ARCH_MMA_SM75_ENABLED
#endif
#endif

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - FP16 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation - F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<16, 8, 8>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;
  
  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 2>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  unsigned const *C = reinterpret_cast<unsigned const *>(&c);
  unsigned *D = reinterpret_cast<unsigned *>(&d);

  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(C[0]), "r"(C[1]));

#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 8>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : 
        "r"(A[0]), "r"(A[1]), 
        "r"(B[0]), 
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );

#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Integer matrix multiply  (8b) with SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Integer matrix multiply  (4b) - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 32>,
  32,
  int4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8, 8, 32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 32>,
  32,
  uint4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8, 8, 32>;

  using ElementA = uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 32>,
  32,
  int4b_t,
  layout::RowMajor,
  uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8, 8, 32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 32>,
  32,
  uint4b_t,
  layout::RowMajor,
  uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8, 8, 32>;

  using ElementA = uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint4b_t, 8>;

  using ElementB = uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));
#else
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_UNUSED(d);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// b1 ^ b1 + s32 => s32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma<
  gemm::GemmShape<8,8,128>,
  32,
  uint1b_t,
  layout::RowMajor,
  uint1b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpXorPopc> {

  using Shape = gemm::GemmShape<8,8,128>;

  using ElementA = uint1b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint1b_t, 32>;

  using ElementB = uint1b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint1b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpXorPopc;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
  using WmmaFragmentA = nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          nvcuda::wmma::experimental::precision::b1,
          nvcuda::wmma::row_major>;

  using WmmaFragmentB = nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          nvcuda::wmma::experimental::precision::b1,
          nvcuda::wmma::col_major>;

  using WmmaFragmentC = nvcuda::wmma::fragment<
          nvcuda::wmma::accumulator,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          int>;
  
  WmmaFragmentA const & A = reinterpret_cast<WmmaFragmentA const &>(a);
  WmmaFragmentB const & B = reinterpret_cast<WmmaFragmentB const &>(b);

  WmmaFragmentC const & C = reinterpret_cast<WmmaFragmentC const &>(c);
  WmmaFragmentC & D = reinterpret_cast<WmmaFragmentC &>(d);

  nvcuda::wmma::bmma_sync(D, A, B, C, nvcuda::wmma::experimental::bmmaBitOpXOR, 
                                          nvcuda::wmma::experimental::bmmaAccumulateOpPOPC);

#else

  CUTLASS_UNUSED(a);
  CUTLASS_UNUSED(b);
  CUTLASS_UNUSED(c);
  CUTLASS_UNUSED(d);
  CUTLASS_NOT_IMPLEMENTED(); // WMMA must be supported to issue binary matrix multiply-accumulate instructions.

#endif // defined(CUTLASS_ARCH_WMMA_ENABLED)

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
