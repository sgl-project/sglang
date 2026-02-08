/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Block Wise Scale configs specific for SM100 Blockwise/Groupwise MMA
*/

#pragma once

#include "cutlass/layout/matrix.h"

#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace cutlass::detail{

/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute;

template<int SFVecSizeMN, int SFVecSizeK, UMMA::Major majorSFA = UMMA::Major::MN>
struct Sm100MixedInputBlockwiseScaleConfig {

  using ShapeScale = Shape<Shape<Int<SFVecSizeMN>, int32_t>, Shape<Int<SFVecSizeK>, int32_t>, int32_t>;

  using StrideScale = conditional_t<majorSFA == UMMA::Major::MN, 
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>, 
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using LayoutScale = Layout<ShapeScale, StrideScale>;

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layout_scale() {
    return LayoutScale{};
  }

  template<class CtaShape_MN_K>
  CUTE_HOST_DEVICE
  static constexpr auto
  smem_atom_layout_scale(CtaShape_MN_K cta_shape_mn_k) {
    static_assert(cute::is_static_v<CtaShape_MN_K>, "Expect static CTA shape");

    int constexpr size_MN = cute::get<0>(CtaShape_MN_K{});
    int constexpr size_K = cute::get<1>(CtaShape_MN_K{});

    int constexpr SmemSizeMN = (SFVecSizeMN < size_MN) 
                           ? SFVecSizeMN 
                           : size_MN;

    int constexpr SmemSizeK = (SFVecSizeK < size_K) 
                           ? SFVecSizeK 
                           : size_K;

    int constexpr div_MN = cute::ceil_div(size_MN, SmemSizeMN);
    int constexpr div_K = cute::ceil_div(size_K, SmemSizeK);
    
    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (majorSFA == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, Int<div_MN>{}));
      }
      else {
        return make_stride(make_stride(_0{}, Int<div_K>{}), make_stride(_0{}, _1{}));
      }
    }();

    return make_layout(
      make_shape(make_shape(Int<SmemSizeMN>{}, Int<div_MN>{}),
                 make_shape(Int<SmemSizeK>{}, Int<div_K>{})),
      strides
    );
  }



  // The following function is provided for user fill dynamic problem size to the layout_SFA.
  template <class ScaledInputDim>
  CUTE_HOST_DEVICE
  static constexpr auto 
  tile_atom_to_shape_scale(ScaledInputDim scale_input_dims) {
    const auto scale_input_dims_MNKL = append<3>(scale_input_dims, 1);

    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [MN, K, L] = scale_input_dims_MNKL;
      if constexpr (majorSFA == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(MN, SFVecSizeMN)));
      }
      else {
        return make_stride(make_stride(_0{}, cute::ceil_div(K, SFVecSizeK)), make_stride(_0{}, _1{}));
      }
    }();

    auto [MN, K, L] = scale_input_dims_MNKL;
    auto mk_layout = make_layout(
      make_shape(make_shape(Int<SFVecSizeMN>{}, cute::ceil_div(MN, SFVecSizeMN)),
                 make_shape(Int<SFVecSizeK>{}, cute::ceil_div(K, SFVecSizeK))),
      strides
    );

    return make_layout(append(shape(mk_layout), L), append(stride(mk_layout), size(filter_zeros(mk_layout))));
  }

};

template<UMMA::Major majorScale = UMMA::Major::MN>
struct RuntimeMixedInputBlockwiseScaleConfig {

  using ShapeScale = Shape<Shape<int32_t, int32_t>, Shape<int32_t, int32_t>, int32_t>;

  using StrideScale = conditional_t<majorScale == UMMA::Major::MN, 
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>, 
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using LayoutScale = Layout<ShapeScale, StrideScale>;

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layout_scale() {
    return LayoutScale{};
  }

  // The following function is provided for user fill dynamic problem size to the layout_S.
  template <class ProblemShape, class SFVecShape>
  CUTE_HOST_DEVICE
  static constexpr auto 
  tile_atom_to_shape_scale(ProblemShape problem_shape, SFVecShape sf_vec_shape) {
    auto problem_shape_MNKL = append<3>(problem_shape, 1);

    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [MN, K, L] = problem_shape_MNKL;
      auto [sfmn, sfk] = sf_vec_shape;
      if constexpr (majorScale == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(MN, sfmn)));
      }
      else {
        return make_stride(make_stride(_0{}, cute::ceil_div(K, sfk)), make_stride(_0{}, _1{}));
      }
    }();

    auto [MN, K, L] = problem_shape_MNKL;
    auto [sfmn, sfk] = sf_vec_shape;
    auto mk_layout = make_layout(
      make_shape(make_shape(sfmn, cute::ceil_div(MN, sfmn)),
                 make_shape(sfk, cute::ceil_div(K, sfk))),
      strides
    );

    return make_layout(append(shape(mk_layout), L), append(stride(mk_layout), size(filter_zeros(mk_layout))));
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
