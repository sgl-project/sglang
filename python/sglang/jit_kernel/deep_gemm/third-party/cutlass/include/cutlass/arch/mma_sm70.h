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

#if ((__CUDACC_VER_MAJOR__ > 10) || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))
#define CUTLASS_ARCH_MMA_SM70_SUPPORTED
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))

#if ((__CUDACC_VER_MAJOR__ > 10) || (__CUDACC_VER_MAJOR__ == 10 &&__CUDACC_VER_MINOR__ >= 1))
#define CUTLASS_ARCH_MMA_SM70_ENABLED
#endif

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Matrix multiply accumulate 884 - FP16 accumulation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<8,8,4>,
  8,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::ColumnMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm70;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_SM70_ENABLED)

    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);

    asm volatile("mma.sync.aligned.m8n8k4.col.col.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3])
    );

#else
    assert(0);
    #if defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
    #endif
#endif
  }
};

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::ColumnMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::RowMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm70;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_SM70_ENABLED)

    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);

    asm volatile("mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3])
    );

#else
    assert(0);
    #if defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
    #endif
#endif
  }
};

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm70;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_SM70_ENABLED)

    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);

    asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3])
    );

#else
    assert(0);
    #if defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
    #endif
#endif
  }
};

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::RowMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm70;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_SM70_ENABLED)

    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const *B = reinterpret_cast<unsigned const *>(&b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);

    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3])
    );

#else
    assert(0);
    #if defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
    #endif
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Matrix multiply accumulate 884 - FP32 accumulation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::ColumnMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm70;

  /// Multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_SM70_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm volatile("mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=f"(D[0]),
        "=f"(D[1]),
        "=f"(D[2]),
        "=f"(D[3]),
        "=f"(D[4]),
        "=f"(D[5]),
        "=f"(D[6]),
        "=f"(D[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7])
  );

#else
    assert(0);
    #if defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
    #endif
#endif
  }
};

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::ColumnMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::RowMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm70;

  /// Multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_SM70_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=f"(D[0]),
        "=f"(D[1]),
        "=f"(D[2]),
        "=f"(D[3]),
        "=f"(D[4]),
        "=f"(D[5]),
        "=f"(D[6]),
        "=f"(D[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7])
  );

#else
    assert(0);
    #if defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
    #endif
#endif
  }
};

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm70;

  /// Multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_SM70_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=f"(D[0]),
        "=f"(D[1]),
        "=f"(D[2]),
        "=f"(D[3]),
        "=f"(D[4]),
        "=f"(D[5]),
        "=f"(D[6]),
        "=f"(D[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7])
  );

#else
    assert(0);
    #if defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
    #endif
#endif
  }
};

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 4>,
  8,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 4>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::RowMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm70;

  /// Multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) {

#if defined(CUTLASS_ARCH_MMA_SM70_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=f"(D[0]),
        "=f"(D[1]),
        "=f"(D[2]),
        "=f"(D[3]),
        "=f"(D[4]),
        "=f"(D[5]),
        "=f"(D[6]),
        "=f"(D[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7])
  );

#else
    assert(0);
    #if defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
    #endif
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation specialized for the entire warp
template <
  typename LayoutA,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename Operator
>
struct Mma<
  gemm::GemmShape<16, 16, 4>,
  32,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  ElementC,
  LayoutC,
  Operator
> : 
  public Mma<
    gemm::GemmShape<8, 8, 4>, 
    8, 
    half_t, 
    LayoutA, 
    half_t, 
    LayoutB,
    ElementC, 
    LayoutC, 
    Operator> {

  using Shape = gemm::GemmShape<16, 16, 4>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
