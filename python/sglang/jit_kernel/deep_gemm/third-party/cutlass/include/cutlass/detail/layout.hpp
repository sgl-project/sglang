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

#include "cute/layout.hpp"
#include "cute/pointer_sparse.hpp"       // cute::is_sparse
#include "cute/swizzle.hpp"              // cute::Swizzle
#include "cute/swizzle_layout.hpp"       // cute::get_swizzle_portion
#include "cute/util/type_traits.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/copy_sm100_tma.hpp"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/collective.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////
// For each cutlass::layout, provides its corresponding cute stride types, 64b by default

template <class L>
struct TagToStrideA {
  using type = L;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<layout::RowMajor> {
  using type = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using tag = layout::RowMajor;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<layout::ColumnMajor> {
  using type = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using tag = layout::ColumnMajor;
};

template <class L>
struct TagToStrideB {
  using type = L;
};

// Maps to modes [N, K, L]
template <>
struct TagToStrideB<layout::RowMajor> {
  using type = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using tag = layout::RowMajor;
};

// Maps to modes [N, K, L]
template <>
struct TagToStrideB<layout::ColumnMajor> {
  using type = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using tag = layout::ColumnMajor;
};

// For each cutlass::layout *, provides its corresponding cute stride types, 64b by default
// Used by pointer array and grouped gemm
// Maps to modes [M, K, L]
template <>
struct TagToStrideA<layout::RowMajor *> {
  using UnderlyingType = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
  using type = UnderlyingType*;
  using tag = layout::RowMajor;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<layout::ColumnMajor *> {
  using UnderlyingType = cute::Stride<cute::Int<1>, int64_t, cute::Int<0>>;
  using type = UnderlyingType*;
  using tag = layout::ColumnMajor;
};

// Maps to modes [N, K, L]
template <>
struct TagToStrideB<layout::RowMajor *> {
  using UnderlyingType = cute::Stride<cute::Int<1>, int64_t, cute::Int<0>>;
  using type = UnderlyingType*;
  using tag = layout::RowMajor;
};

// Maps to modes [N, K, L]
template <>
struct TagToStrideB<layout::ColumnMajor *> {
  using UnderlyingType = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
  using type = UnderlyingType*;
  using tag = layout::ColumnMajor;
};

// Maps to modes [M, N, L]
template <class LayoutTag>
struct TagToStrideC : TagToStrideA<LayoutTag> { };

// Conv: Maps to modes ((P,N), C, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<cutlass::layout::TensorNWC> {
  using type = cute::Stride<cute::Stride<int64_t, int64_t>, cute::Int<1>, cute::Int<0>>;
};

// Conv: Maps to modes ((P,Q,N), C, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<cutlass::layout::TensorNHWC> {
  using type = cute::Stride<cute::Stride<int64_t, int64_t, int64_t>, cute::Int<1>, cute::Int<0>>;
};

// Conv: Maps to modes ((P,Q,Z,N), C, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<cutlass::layout::TensorNDHWC> {
  using type = cute::Stride<cute::Stride<int64_t, int64_t, int64_t, int64_t>, cute::Int<1>, cute::Int<0>>;
};

// Conv: Maps to modes (K, (C,S), _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<cutlass::layout::TensorKCS> {
  using type = cute::Stride<int64_t, cute::Stride<cute::Int<1>, int64_t>, cute::Int<0>>;
};

// Conv: Maps to modes (K, (C,S,R), _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<cutlass::layout::TensorKCSR> {
  using type = cute::Stride<int64_t, cute::Stride<cute::Int<1>, int64_t, int64_t>, cute::Int<0>>;
};

// Conv: Maps to modes (K, (C,S,R,T), _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<cutlass::layout::TensorKCSRT> {
  using type = cute::Stride<int64_t, cute::Stride<cute::Int<1>, int64_t, int64_t, int64_t>, cute::Int<0>>;
};

// Conv: Maps to modes ((C,S), K, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<cutlass::layout::TensorCSK> {
  using type = cute::Stride<cute::Stride<cute::Int<1>, int64_t>, int64_t, cute::Int<0>>;
};

// Conv: Maps to modes ((C,S,R), K, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<cutlass::layout::TensorCSRK> {
  using type = cute::Stride<cute::Stride<cute::Int<1>, int64_t, int64_t>, int64_t, cute::Int<0>>;
};

// Conv: Maps to modes ((C,S,R,T), K, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<cutlass::layout::TensorCSRTK> {
  using type = cute::Stride<cute::Stride<cute::Int<1>, int64_t, int64_t, int64_t>, int64_t, cute::Int<0>>;
};

// Convenience aliases
template<class LayoutTag>
using TagToStrideA_t = typename TagToStrideA<LayoutTag>::type;

template<class LayoutTag>
using TagToStrideB_t = typename TagToStrideB<LayoutTag>::type;

template<class LayoutTag>
using TagToStrideC_t = typename TagToStrideC<LayoutTag>::type;

////////////////////////////////////////////////////////////////////////////////////////////////////
// For 2.x compatibility APIs, provide stride->layout tag mappers

template<int ModeIndex, class Stride>
constexpr bool
is_major(Stride = {}) {
  // Account for stride types with and without batch mode and batch modes with static zero stride
  return cute::is_constant<1, decltype(cute::front(cute::get<ModeIndex>(cute::remove_pointer_t<Stride>{})))>::value;
}

template<int ModeIndex, class Shape, class Stride>
constexpr bool
is_major(cute::Layout<Shape,Stride> = {}) {
  return is_major<ModeIndex>(Stride{});
}

// Note : This method can be used for deducing the Layout Tag of A, C, D Matrices
template<class StrideA>
constexpr
auto
stride_to_layout_tag_A() {
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  if constexpr (cute::is_layout<InternalStrideA>::value) {
    return stride_to_layout_tag_A<decltype(cute::stride(InternalStrideA{}))>();
  }
  else if constexpr (is_major<0, StrideA>()) { // M major
    return layout::ColumnMajor{};
  }
  // Specialize for sparse layout
  else if constexpr (cute::get<0>(InternalStrideA{}) == cute::_2{} &&
                     cute::rank(cute::get<1>(InternalStrideA{})) == 2 &&
                     cute::is_same_v<cute::_1, cute::remove_cvref_t<decltype(cute::get<1,0>(InternalStrideA{}))>>) {
    return layout::ColumnMajor{};
  }
  else { // K major
    return layout::RowMajor{};
  }

  CUTE_GCC_UNREACHABLE;
}

template<class StrideB>
constexpr
auto
stride_to_layout_tag_B() {
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  if constexpr (cute::is_layout<InternalStrideB>::value) {
    return stride_to_layout_tag_B<decltype(cute::stride(InternalStrideB{}))>();
  }
  else if constexpr (is_major<0, StrideB>()) { // N major
    return layout::RowMajor{};
  }
  else { // K major
    return layout::ColumnMajor{};
  }

  CUTE_GCC_UNREACHABLE;
}

template<class StrideC>
constexpr
auto
stride_to_layout_tag_C() {
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  if constexpr (cute::is_layout<InternalStrideC>::value) {
    return stride_to_layout_tag_C<decltype(cute::stride(InternalStrideC{}))>();
  }
  else if constexpr (is_major<0, StrideC>()) { // M major
    return layout::ColumnMajor{};
  }
  else { // N major
    return layout::RowMajor{};
  }

  CUTE_GCC_UNREACHABLE;
}

// Utilities to map Stride back on to their corresponding layout tags
template <class S>
struct StrideToLayoutTagA {
  using type = decltype(detail::stride_to_layout_tag_A<S>());
};

template <class S>
struct StrideToLayoutTagB {
  using type = decltype(detail::stride_to_layout_tag_B<S>());
};

template <class S>
struct StrideToLayoutTagC {
  using type = decltype(detail::stride_to_layout_tag_C<S>());
};

// Convenience aliases
template<class S>
using StrideToLayoutTagA_t = typename StrideToLayoutTagA<S>::type;

template<class S>
using StrideToLayoutTagB_t = typename StrideToLayoutTagB<S>::type;

template<class S>
using StrideToLayoutTagC_t = typename StrideToLayoutTagC<S>::type;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Inspects a tiled copy and whether its copy engine is TMA or not
template<class GmemTiledCopy>
constexpr bool is_tma_copy_engine() {
  if constexpr (cute::is_void_v<GmemTiledCopy>) {
    return false;
  }
  else {
   if constexpr (   cute::is_base_of_v<cute::SM90_TMA_LOAD,                         GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM90_TMA_LOAD_MULTICAST,              GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM90_TMA_LOAD_IM2COL,                 GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM90_TMA_LOAD_IM2COL_MULTICAST,       GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM90_TMA_STORE,                       GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM90_TMA_STORE_IM2COL,                GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM100_TMA_2SM_LOAD,                   GmemTiledCopy>
                  || cute::is_base_of_v<cute::SM100_TMA_2SM_LOAD_MULTICAST,         GmemTiledCopy>
                  ) {
      return true;
    }
  }
  return false;
}

template <class X, class = void>
struct RawDtype { using type = X; };

template <class X>
struct RawDtype<X,cute::void_t<typename X::raw_type>> { using type = typename X::raw_type; };


// Inspects a TiledCopy and returns its alignment in terms of element count
template <class GmemTiledCopy, class Element, class ElementMma = Element>
constexpr int
get_alignment_count_from_gmem_tiled_copy() {

  if constexpr (cute::is_void_v<GmemTiledCopy>) {
    return 1;
  }

  // Account for ElementC = void kernels
  else if constexpr (cute::is_void_v<Element>) {
    return 0;
  }

  else {
    // For TMA tiled copies, we know the alignment has to be 128 bits
    if constexpr (is_tma_copy_engine<GmemTiledCopy>()) {
      if constexpr ( cute::is_same_v<typename RawDtype<ElementMma>::type, cutlass::detail::float_e2m1_unpacksmem_t> ||
                     cute::is_same_v<typename RawDtype<ElementMma>::type, cutlass::detail::float_e3m2_unpacksmem_t> ||
                     cute::is_same_v<typename RawDtype<ElementMma>::type, cutlass::detail::float_e2m3_unpacksmem_t> ||
                     cute::is_same_v<typename RawDtype<ElementMma>::type, cutlass::detail::type_erased_dynamic_float4_unpacksmem_t> ||
                     cute::is_same_v<typename RawDtype<ElementMma>::type, cutlass::detail::type_erased_dynamic_float6_unpacksmem_t> ||
                     cutlass::gemm::collective::detail::is_sm10x_f8f6f4_element<Element>() && cute::is_same_v<typename RawDtype<ElementMma>::type, uint8_t>) {
        return 128;
      }

      // For sparse MMA, alignment in logical elements is increased by sparsity factor
      if constexpr (cute::is_sparse_v<ElementMma>) {
        return 128 / sizeof_bits<Element>::value * ElementMma::sparsity;
      }
      return 128 / sizeof_bits<Element>::value;
    }
    else {
      // For non-TMA tiled copies, TiledCopy holds the alignment count directly in its TiledShape_MN
      return GmemTiledCopy::NumValSrc;
    }
  }
}

// Return alignment bit requirements for the GEMM inputs.
template <
  class ElementType
  , bool IsF8F6F4SubBytes=false
>
constexpr int
get_input_alignment_bits() {
  if constexpr (IsF8F6F4SubBytes && sizeof_bits<ElementType>::value == 4) {
    // 16U4 format: The inner tensor size dimension should be multiple of 64B.
    return 64 * 8;
  }
  else if constexpr (IsF8F6F4SubBytes && sizeof_bits<ElementType>::value == 6) {
    // 16U6 format : The inner tensor size dimension must be a multiple of 96B.
    return 96 * 8;
  }
  // TMA 16B alignment requirement
  return 128;
}

// Return alignment bit requirements for the GEMM outputs.
template <class ElementType>
constexpr int
get_output_alignment_bits() {
  if constexpr (sizeof_bits<ElementType>::value == 6) {
    // 16U6 format : The inner tensor size dimension must be a multiple of 96B.
    return 96 * 8;
  }
  // TMA 16B alignment requirement
  return 128;
}

// Check if tensor layout satisfies a given major alignment
template<int Alignment, class Shape, class Stride>
CUTLASS_HOST_DEVICE constexpr
bool
check_alignment(cute::Layout<Shape,Stride> const& layout) {
  // Condition: shape must divide by Alignment without rounding
  bool shape_check = cute::size(layout.shape()) == Alignment * cute::size(cute::upcast<Alignment>(layout));
  // Condition: every dynamic stride must be a multiple of Alignment
  bool stride_check = cute::all_of(cute::flatten(layout.stride()), [](auto s){ return cute::is_static<decltype(s)>::value || (s % Alignment == 0); });
  return shape_check && stride_check;
}

// Check if tensor layout satisfies a given major alignment
template<int Alignment, class Shape, class Stride>
CUTLASS_HOST_DEVICE constexpr
bool
check_alignment(Shape const& shape, Stride const& stride) {
  return check_alignment<Alignment>(cute::make_layout(shape, stride));
}

template<int B, int M, int S>
CUTLASS_HOST_DEVICE constexpr
size_t
alignment_for_swizzle(cute::Swizzle<B, M, S>) {
  static_assert(B >= 0 and M >= 0);
  return size_t(1) << size_t(B + M + cute::abs(S));
}

template<class Layout>
CUTLASS_HOST_DEVICE constexpr
size_t
alignment_for_swizzle(Layout layout) {
  return alignment_for_swizzle(cute::get_swizzle_portion(layout));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
