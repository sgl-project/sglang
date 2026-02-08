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
    \brief Matrix multiply
*/

#pragma once
#include "cutlass/cutlass.h"
#include CUDA_STD_HEADER(cassert)

#include "mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/config.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////
/// Matrix Multiply-Add 16x8x4 fp64
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F64 = F64 * F64 + F64
template <>
struct Mma<
  gemm::GemmShape<16,8,4>,
  32,
  double,
  layout::RowMajor,
  double,
  layout::ColumnMajor,
  double,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,4>;

  using ElementA = double;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<double, 2>;

  using ElementB = double;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<double, 1>;

  using ElementC = double;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<double, 4>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm90;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_SM90_F64_MMA_ENABLED)

  double const *A = reinterpret_cast<double const *>(&a);
  double const *B = reinterpret_cast<double const *>(&b);

  double const *C = reinterpret_cast<double const *>(&c);
  double *D = reinterpret_cast<double *>(&d);

  asm volatile("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64.rn {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      : "=d"(D[0]), "=d"(D[1]), "=d"(D[2]), "=d"(D[3])
      : "d"(A[0]), "d"(A[1]),
        "d"(B[0]),
        "d"(C[0]), "d"(C[1]), "d"(C[2]), "d"(C[3]));

#else
    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Matrix Multiply-Add 16x8x8 fp64
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F64 = F64 * F64 + F64
template <>
struct Mma<
  gemm::GemmShape<16,8,8>,
  32,
  double,
  layout::RowMajor,
  double,
  layout::ColumnMajor,
  double,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,8>;

  using ElementA = double;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<double, 4>;

  using ElementB = double;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<double, 2>;

  using ElementC = double;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<double, 4>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm90;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_SM90_F64_MMA_ENABLED)

  double const *A = reinterpret_cast<double const *>(&a);
  double const *B = reinterpret_cast<double const *>(&b);

  double const *C = reinterpret_cast<double const *>(&c);
  double *D = reinterpret_cast<double *>(&d);

  asm volatile("mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=d"(D[0]), "=d"(d[1]), "=d"(d[2]), "=d"(d[3])
      : "d"(A[0]), "d"(A[1]), "d"(A[2]), "d"(A[3]),
        "d"(B[0]), "d"(B[1]),
        "d"(C[0]), "d"(C[1]), "d"(C[2]), "d"(C[3]));

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Matrix Multiply-Add 16x8x16 fp64
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F64 = F64 * F64 + F64
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  double,
  layout::RowMajor,
  double,
  layout::ColumnMajor,
  double,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = double;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<double, 8>;

  using ElementB = double;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<double, 4>;

  using ElementC = double;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<double, 4>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm90;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    
#if defined(CUTLASS_ARCH_MMA_SM90_F64_MMA_ENABLED)

  double const *A = reinterpret_cast<double const *>(&a);
  double const *B = reinterpret_cast<double const *>(&b);

  double const *C = reinterpret_cast<double const *>(&c);
  double *D = reinterpret_cast<double *>(&d);

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, {%16, %17, %18, %19};\n"
      : "=d"(D[0]), "=d"(D[1]), "=d"(D[2]), "=d"(D[3])
      : "d"(A[0]), "d"(A[2]), "d"(A[2]), "d"(A[3]), "d"(A[4]), "d"(A[5]), "d"(A[6]), "d"(A[7]),
        "d"(B[0]), "d"(B[1]), "d"(B[2]), "d"(B[3]), 
        "d"(C[0]), "d"(C[1]), "d"(C[2]), "d"(C[3]));

#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

