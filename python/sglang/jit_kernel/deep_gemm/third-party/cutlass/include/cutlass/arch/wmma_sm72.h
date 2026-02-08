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
#include "cutlass/layout/matrix.h"

////////////////////////////////////////////////////////////////////////////////
namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////
//
// WMMA template structure defines nvcuda::wmma::fragments and static assert for
// wmma native instruction sizes supported for int8_t
//
////////////////////////////////////////////////////////////////////////////////
template <
typename Shape_, 
typename LayoutA_, 
typename LayoutB_,
typename LayoutC_>
struct Wmma<
  Shape_,                                   ///< Size of the matrix product (concept: GemmShape)
  int8_t,                                   ///< ElementA
  LayoutA_,                                 ///< LayoutA
  int8_t,                                   ///< ElementB
  LayoutB_,                                 ///< LayoutB
  int32_t,                                  ///< ElementC
  LayoutC_,                                 ///< LayoutC
  cutlass::arch::OpMultiplyAdd              ///< Operator (multiply-add, xor.popc)
> {
#if defined(CUTLASS_ARCH_WMMA_SM72_ENABLED)
  using Shape = Shape_;
  using ElementA = int8_t;
  using LayoutA = LayoutA_;
  using ElementB = int8_t;
  using LayoutB = LayoutB_;
  using ElementC = int32_t;
  using LayoutC = LayoutC_;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using ArchTag = arch::Sm72;

  // check supported wmma shape for the given multiplicand data types
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, Shape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 8, 32, 16>, Shape>::value ||
    platform::is_same<cutlass::gemm::GemmShape<32,  8, 16>, Shape>::value,
    "Supported list of wmma operator shape for s8 multiplicands are: 16x16x16, 8x32x16, and 32x8x16");


  // Wmma Fragment
  using FragmentA = nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          typename CutlassToWmmaDataType<ElementA>::Type,
          typename CutlassToWmmaLayout<LayoutA>::Layout>;

  using FragmentB = nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          typename CutlassToWmmaDataType<ElementB>::Type,
          typename CutlassToWmmaLayout<LayoutB>::Layout>;

  using FragmentC = nvcuda::wmma::fragment<
          nvcuda::wmma::accumulator,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          typename CutlassToWmmaDataType<ElementC>::Type>;

  /// Performs a nvcuda::wmma matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    FragmentA const &A, 
    FragmentB const &B, 
    FragmentC const &C) const {

      nvcuda::wmma::mma_sync(D, A, B, C);
  }

#else
    static_assert(false, "wmma.mma.sync integer type multiplicands is available only for SM72 and beyond");
#endif

};

////////////////////////////////////////////////////////////////////////////////
//
// WMMA template structure defines nvcuda::wmma::fragments and static assert for
// wmma native instruction sizes supported for uint8_t
//
////////////////////////////////////////////////////////////////////////////////
template <
typename Shape_, 
typename LayoutA_, 
typename LayoutB_,
typename LayoutC_>
struct Wmma<
  Shape_,                                   ///< Size of the matrix product (concept: GemmShape)
  uint8_t,                                  ///< ElementA
  LayoutA_,                                 ///< LayoutA
  uint8_t,                                  ///< ElementB
  LayoutB_,                                 ///< LayoutB
  int32_t,                                  ///< ElementC
  LayoutC_,                                 ///< LayoutC
  cutlass::arch::OpMultiplyAdd              ///< Operator (multiply-add, xor.popc)
> {
#if defined(CUTLASS_ARCH_WMMA_SM72_ENABLED)
  using Shape = Shape_;
  using ElementA = uint8_t;
  using LayoutA = LayoutA_;
  using ElementB = uint8_t;
  using LayoutB = LayoutB_;
  using ElementC = int32_t;
  using LayoutC = LayoutC_;
  using Operator = cutlass::arch::OpMultiplyAdd;
  using ArchTag = arch::Sm72;

  // check supported wmma shape for the given multiplicand data types
  static_assert(
    platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, Shape>::value ||
    platform::is_same<cutlass::gemm::GemmShape< 8, 32, 16>, Shape>::value ||
    platform::is_same<cutlass::gemm::GemmShape<32,  8, 16>, Shape>::value,
    "Supported list of wmma operator shape for u8 multiplicands are: 16x16x16, 8x32x16, and 32x8x16");

  // Wmma Fragment
  using FragmentA = nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          typename CutlassToWmmaDataType<ElementA>::Type,
          typename CutlassToWmmaLayout<LayoutA>::Layout>;

  using FragmentB = nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          typename CutlassToWmmaDataType<ElementB>::Type,
          typename CutlassToWmmaLayout<LayoutB>::Layout>;

  using FragmentC = nvcuda::wmma::fragment<
          nvcuda::wmma::accumulator,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          typename CutlassToWmmaDataType<ElementC>::Type>;
  
  /// Performs a nvcuda::wmma matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D, 
    FragmentA const &A, 
    FragmentB const &B, 
    FragmentC const &C) const {

      nvcuda::wmma::mma_sync(D, A, B, C);
  }
  
#else
    static_assert(false, "wmma.mma.sync integer type multiplicands is available only for SM72 and beyond");
#endif

};

} // namespace arch
} // namespace cutlass
