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
    \brief Matrix multiply-accumulate specialzied for SM89
*/

#pragma once
#include "cutlass/cutlass.h"
#include CUDA_STD_HEADER(cassert)

#include "mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

////////////////////////////////////////////////////////////////////////////////

#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4)
#  define CUTLASS_ARCH_MMA_F32_SM89_SUPPORTED
#endif

#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
#  define CUTLASS_ARCH_MMA_F16_SM89_SUPPORTED
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#  if defined(CUTLASS_ARCH_MMA_F32_SM89_SUPPORTED)
#    define CUTLASS_ARCH_MMA_F32_SM89_ENABLED
#  endif

#  if defined(CUTLASS_ARCH_MMA_F16_SM89_SUPPORTED)
#    define CUTLASS_ARCH_MMA_F16_SM89_ENABLED
#  endif
#endif

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Whether the Mma uses as SM89 staged accumulation policy
template <class Operator>
static constexpr bool is_sm89_staged_policy_v =
  (
    // ElementA must be FP8
    platform::is_same<typename Operator::ElementA, cutlass::float_e4m3_t>::value ||
    platform::is_same<typename Operator::ElementA, cutlass::float_e5m2_t>::value
  ) &&
  (
    // ElementB must be FP8
    platform::is_same<typename Operator::ElementB, cutlass::float_e4m3_t>::value ||
    platform::is_same<typename Operator::ElementB, cutlass::float_e5m2_t>::value
  ) &&
  (
    // The instruction shape must be 16x8x32
    Operator::ArchMmaOperator::Shape::kM == 16 &&
    Operator::ArchMmaOperator::Shape::kN == 8 &&
    Operator::ArchMmaOperator::Shape::kK == 32
  ) &&
  (
    // The operator must be OpMultiplyAdd (default)
    platform::is_same<typename Operator::MathOperator, OpMultiplyAdd>::value
  );
} // namespace detail

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16832 - Float {E4M3, E5M2}, FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation - F32 = fe4m3 * fe4m3 + F32
template <typename Operator_>
struct Mma<
  gemm::GemmShape<16, 8, 32>,
  32,
  cutlass::float_e4m3_t,
  layout::RowMajor,
  cutlass::float_e4m3_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  Operator_> {
  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_F32_SM89_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

/// Matrix multiply-add operation - F32 = fe4m3 * fe5m2 + F32
template <typename Operator_>
struct Mma<
  gemm::GemmShape<16, 8, 32>,
  32,
  cutlass::float_e4m3_t,
  layout::RowMajor,
  cutlass::float_e5m2_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  Operator_> {
  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e5m2_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_F32_SM89_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

/// Matrix multiply-add operation - F32 = fe5m2 * fe4m3 + F32
template <typename Operator_>
struct Mma<
  gemm::GemmShape<16, 8, 32>,
  32,
  cutlass::float_e5m2_t,
  layout::RowMajor,
  cutlass::float_e4m3_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  Operator_> {
  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = cutlass::float_e5m2_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_F32_SM89_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

/// Matrix multiply-add operation - F32 = fe5m2 * fe5m2 + F32
template <typename Operator_>
struct Mma<
  gemm::GemmShape<16, 8, 32>,
  32,
  cutlass::float_e5m2_t,
  layout::RowMajor,
  cutlass::float_e5m2_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  Operator_> {
  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = cutlass::float_e5m2_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e5m2_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_F32_SM89_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );

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
//
// Matrix Multiply 16832 - Float {E4M3, E5M2}, FP16 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation - F16 = fe4m3 * fe4m3 + F16
template <typename Operator_>
struct Mma<
  gemm::GemmShape<16, 8, 32>,
  32,
  cutlass::float_e4m3_t,
  layout::RowMajor,
  cutlass::float_e4m3_t,
  layout::ColumnMajor,
  cutlass::half_t,
  layout::RowMajor,
  Operator_> {
  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 8>;

  using ElementC = cutlass::half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<cutlass::half_t, 4>;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_F16_SM89_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
  uint32_t *D = reinterpret_cast<uint32_t *>(&d);

  asm(
      "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(D[0]), "=r"(D[1])
      :
        "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1])
  );

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

/// Matrix multiply-add operation - F16 = fe4m3 * fe5m2 + F16
template <typename Operator_>
struct Mma<
  gemm::GemmShape<16, 8, 32>,
  32,
  cutlass::float_e4m3_t,
  layout::RowMajor,
  cutlass::float_e5m2_t,
  layout::ColumnMajor,
  cutlass::half_t,
  layout::RowMajor,
  Operator_> {
  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e5m2_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 8>;

  using ElementC = cutlass::half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<cutlass::half_t, 4>;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_F16_SM89_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
  uint32_t *D = reinterpret_cast<uint32_t *>(&d);

  asm(
      "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(D[0]), "=r"(D[1])
      :
        "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1])
  );

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

/// Matrix multiply-add operation - F16 = fe5m2 * fe4m3 + F16
template <typename Operator_>
struct Mma<
  gemm::GemmShape<16, 8, 32>,
  32,
  cutlass::float_e5m2_t,
  layout::RowMajor,
  cutlass::float_e4m3_t,
  layout::ColumnMajor,
  cutlass::half_t,
  layout::RowMajor,
  Operator_> {
  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = cutlass::float_e5m2_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 8>;

  using ElementC = cutlass::half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<cutlass::half_t, 4>;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_F16_SM89_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
  uint32_t *D = reinterpret_cast<uint32_t *>(&d);

  asm(
      "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e4m3.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(D[0]), "=r"(D[1])
      :
        "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1])
  );

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

/// Matrix multiply-add operation - F16 = fe5m2 * fe5m2 + F16
template <typename Operator_>
struct Mma<
  gemm::GemmShape<16, 8, 32>,
  32,
  cutlass::float_e5m2_t,
  layout::RowMajor,
  cutlass::float_e5m2_t,
  layout::ColumnMajor,
  cutlass::half_t,
  layout::RowMajor,
  Operator_> {
  static_assert(platform::is_same<Operator_, OpMultiplyAdd>::value ||
                platform::is_same<Operator_, OpMultiplyAddFastAccum>::value,
                "Invalid operator for SM89 FP8 instruction");

  using Shape = gemm::GemmShape<16, 8, 32>;

  using ElementA = cutlass::float_e5m2_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 16>;

  using ElementB = cutlass::float_e5m2_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 8>;

  using ElementC = cutlass::half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<cutlass::half_t, 4>;

  using Operator = Operator_;
  using ArchTag = arch::Sm89;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_F16_SM89_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
  uint32_t *D = reinterpret_cast<uint32_t *>(&d);

  asm(
      "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(D[0]), "=r"(D[1])
      :
        "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1])
  );

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

} // namespace arch
} // namespace cutlass
