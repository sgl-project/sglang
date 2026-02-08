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
    \brief Templates exposing architecture support for warp matrix multiply-add (WMMA) operations
*/

#pragma once

#if (__CUDACC_VER_MAJOR__ >= 9)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))
#define CUTLASS_ARCH_WMMA_ENABLED
#define CUTLASS_ARCH_WMMA_SM70_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ >= 10)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 720))
#define CUTLASS_ARCH_INTEGER_MATRIX_MULTIPLY_ENABLED
#define CUTLASS_ARCH_WMMA_SM72_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ >= 10)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
#define CUTLASS_SUBBYTE_INTEGER_MATRIX_MULTIPLY_ENABLED
#define CUTLASS_ARCH_WMMA_SM75_ENABLED
#endif
#endif

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

#include <mma.h>
#include "cutlass/arch/mma.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////
/// Statically maps cutlass data types => nvcuda::wmma data types
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Type_>
struct CutlassToWmmaDataType{
  using Type = Type_;
};

/// Statically maps cutlass::half_t => __half
template<>
struct CutlassToWmmaDataType<cutlass::half_t> {
  using Type = __half;
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
template<>
struct CutlassToWmmaDataType<cutlass::bfloat16_t> {
  using Type = __nv_bfloat16;
};
#endif

/// Statically maps int8_t => char
template<>
struct CutlassToWmmaDataType<int8_t> {
  using Type = signed char;
};

/// Statically maps uint8_t => char
template<>
struct CutlassToWmmaDataType<uint8_t> {
  using Type = unsigned char;
};

/// Statically maps int32_t => int
template<>
struct CutlassToWmmaDataType<int32_t> {
  using Type = int;
};

#if defined(CUTLASS_SUBBYTE_INTEGER_MATRIX_MULTIPLY_ENABLED)
/// Statically maps cutlass::int4b_t => experimental::precision::s4
template<>
struct CutlassToWmmaDataType<cutlass::int4b_t> {
  using Type = nvcuda::wmma::experimental::precision::s4;
};

/// Statically maps cutlass::uint4b_t => experimental::precision::s4
template<>
struct CutlassToWmmaDataType<cutlass::uint4b_t> {
  using Type = nvcuda::wmma::experimental::precision::u4;
};

/// Statically maps cutlass::uint1b_t => experimental::precision::b1
template<>
struct CutlassToWmmaDataType<cutlass::uint1b_t> {
  using Type = nvcuda::wmma::experimental::precision::b1;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////
/// Statically maps cutlass::layout => nvcuda::wmma layout tags
////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Layout_>
struct CutlassToWmmaLayout {
};

/// Statically maps cutlass::layout::RowMajor => nvcuda::wmma::row_major layout tags
template <>
struct CutlassToWmmaLayout<cutlass::layout::RowMajor> {
  using Layout = nvcuda::wmma::row_major;
  static nvcuda::wmma::layout_t const value = nvcuda::wmma::layout_t::mem_row_major;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// Statically maps cutlass::layout::RowMajor => nvcuda::wmma::row_major layout tags
////////////////////////////////////////////////////////////////////////////////////////////////
template <>
struct CutlassToWmmaLayout<cutlass::layout::ColumnMajor> {
  using Layout = nvcuda::wmma::col_major;
  static nvcuda::wmma::layout_t const value = nvcuda::wmma::layout_t::mem_col_major;
};
////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
/// Statically maps nvcuda::wmma data types => cutlass data types
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Type_>
struct WmmaToCutlassDataType{
  using Type = Type_;
};

/// Statically maps __half => cutlass::half_t
template<>
struct WmmaToCutlassDataType<__half> {
  using Type = cutlass::half_t;
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDACC_VER_MAJOR__ >= 11)
template<>
struct WmmaToCutlassDataType<__nv_bfloat16> {
  using Type = cutlass::bfloat16_t;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// WMMA template structure defines nvcuda::wmma::fragments and static assertion chaeks
// for a specific template parameterized data type (Element[A|B|C]), layout (Layout[A|B|C]), 
// and native wmma size (Shape)
/////////////////////////////////////////////////////////////////////////////////////////////////
template <  
  typename Shape_,                                   ///< Size of the matrix product (concept: GemmShape)
  typename ElementA_,                                ///< Data type of A elements 
  typename LayoutA_,                                 ///< Layout of A matrix (concept: MatrixLayout)  
  typename ElementB_,                                ///< Data type of B elements
  typename LayoutB_,                                 ///< Layout of B matrix (concept: MatrixLayout)  
  typename ElementC_,                                ///< Element type of C matrix  
  typename LayoutC_,                                 /// Layout of C matrix (concept: MatrixLayout)
  typename Operator_ = cutlass::arch::OpMultiplyAdd   ///< Inner product operator (multiply-add, xor.popc)
>
struct Wmma;
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Specializations for each compute capability
//
#ifdef CUTLASS_ARCH_WMMA_SM70_ENABLED
#include "cutlass/arch/wmma_sm70.h"
#endif

#ifdef CUTLASS_ARCH_WMMA_SM72_ENABLED
#include "cutlass/arch/wmma_sm72.h"
#endif

#ifdef CUTLASS_ARCH_WMMA_SM75_ENABLED
#include "cutlass/arch/wmma_sm75.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif //CUTLASS_ARCH_WMMA_ENABLED
