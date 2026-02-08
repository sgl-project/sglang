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
    \brief Blockwise Scale configs specific for Blockwise/Groupwise MMA
*/

#pragma once

#include "cutlass/layout/matrix.h"

#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/arch/mma_sm90.hpp"

namespace cutlass::detail{

/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute;

template<int SFVecSizeM, int SFVecSizeN, int SFVecSizeK, UMMA::Major majorSFA = UMMA::Major::MN, UMMA::Major majorSFB = UMMA::Major::MN>
struct Sm1xxBlockwiseScaleConfig {

  using ShapeSFA = Shape<Shape<Int<SFVecSizeM>, int32_t>, Shape<Int<SFVecSizeK>, int32_t>, int32_t>;
  using ShapeSFB = Shape<Shape<Int<SFVecSizeN>, int32_t>, Shape<Int<SFVecSizeK>, int32_t>, int32_t>;

  using StrideSFA = conditional_t<majorSFA == UMMA::Major::MN, 
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>, 
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using StrideSFB = conditional_t<majorSFB == UMMA::Major::MN, 
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>, 
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using LayoutSFA = Layout<ShapeSFA, StrideSFA>;
  using LayoutSFB = Layout<ShapeSFB, StrideSFB>;

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFA() {
    return LayoutSFA{};
  }

  template<typename CtaShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  smem_atom_layoutSFA(CtaShape_MNK cta_shape_mnk) {
    static_assert(cute::is_static_v<CtaShape_MNK>, "Expect static CTA shape");
    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K] = cta_shape_mnk;
      if constexpr (majorSFA == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, Int<cute::ceil_div(size<0>(CtaShape_MNK{}), SFVecSizeM)>{}));
      }
      else {
        return make_stride(make_stride(_0{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{}), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K] = cta_shape_mnk;
    return make_layout(
      make_shape(make_shape(Int<SFVecSizeM>{}, Int<cute::ceil_div(size<0>(CtaShape_MNK{}), SFVecSizeM)>{}),
                 make_shape(Int<SFVecSizeK>{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{})),
      strides
    );
  }


  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFB() {
    return LayoutSFB{};
  }

  template<typename CtaShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  smem_atom_layoutSFB(CtaShape_MNK cta_shape_mnk) {
    static_assert(cute::is_static_v<CtaShape_MNK>, "Expect static CTA shape");
    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (majorSFA == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, Int<cute::ceil_div(size<1>(CtaShape_MNK{}), SFVecSizeN)>{}));
      }
      else {
        return make_stride(make_stride(_0{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{}), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K] = cta_shape_mnk;
    return make_layout(
      make_shape(make_shape(Int<SFVecSizeN>{}, Int<cute::ceil_div(size<1>(CtaShape_MNK{}), SFVecSizeN)>{}),
                 make_shape(Int<SFVecSizeK>{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{})),
      strides
    );
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFA.
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto 
  tile_atom_to_shape_SFA(ProblemShape problem_shape) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K, L] = problem_shape_MNKL;
      if constexpr (majorSFA == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(M, SFVecSizeM)));
      }
      else {
        return make_stride(make_stride(_0{}, cute::ceil_div(K, SFVecSizeK)), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K, L] = problem_shape_MNKL;
    auto mk_layout = make_layout(
      make_shape(make_shape(Int<SFVecSizeM>{}, cute::ceil_div(M, SFVecSizeM)),
                 make_shape(Int<SFVecSizeK>{}, cute::ceil_div(K, SFVecSizeK))),
      strides
    );

    return make_layout(append(shape(mk_layout), L), append(stride(mk_layout), size(filter_zeros(mk_layout))));
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFB.
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto 
  tile_atom_to_shape_SFB(ProblemShape problem_shape) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K, L] = problem_shape_MNKL;

      if constexpr (majorSFB == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(N, SFVecSizeN)));
      }
      else {
        return make_stride(make_stride(_0{}, cute::ceil_div(K, SFVecSizeK)), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K, L] = problem_shape_MNKL;
    auto nk_layout = make_layout(
      make_shape(make_shape(Int<SFVecSizeN>{}, cute::ceil_div(N, SFVecSizeN)),
                 make_shape(Int<SFVecSizeK>{}, cute::ceil_div(K, SFVecSizeK))),
      strides
    );

    return make_layout(append(shape(nk_layout), L), append(stride(nk_layout), size(filter_zeros(nk_layout))));
  }

};

template<UMMA::Major majorSFA = UMMA::Major::MN, UMMA::Major majorSFB = UMMA::Major::MN>
struct RuntimeBlockwiseScaleConfig {

  using ShapeSFA = Shape<Shape<int32_t, int32_t>, Shape<int32_t, int32_t>, int32_t>;
  using ShapeSFB = Shape<Shape<int32_t, int32_t>, Shape<int32_t, int32_t>, int32_t>;

  using StrideSFA = conditional_t<majorSFA == UMMA::Major::MN, 
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>, 
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using StrideSFB = conditional_t<majorSFB == UMMA::Major::MN, 
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>, 
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using LayoutSFA = Layout<ShapeSFA, StrideSFA>;
  using LayoutSFB = Layout<ShapeSFB, StrideSFB>;

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFA() {
    return LayoutSFA{};
  }

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFB() {
    return LayoutSFB{};
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFA.
  template <class ProblemShape, class SFVecShape>
  CUTE_HOST_DEVICE
  static constexpr auto 
  tile_atom_to_shape_SFA(ProblemShape problem_shape, SFVecShape sf_vec_shape) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K, L] = problem_shape_MNKL;
      auto [sfm, sfn, sfk] = sf_vec_shape;
      if constexpr (majorSFA == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(M, sfm)));
      }
      else {
        return make_stride(make_stride(_0{}, cute::ceil_div(K, sfk)), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K, L] = problem_shape_MNKL;
    auto [sfm, sfn, sfk] = sf_vec_shape;
    auto mk_layout = make_layout(
      make_shape(make_shape(sfm, cute::ceil_div(M, sfm)),
                 make_shape(sfk, cute::ceil_div(K, sfk))),
      strides
    );

    return make_layout(append(shape(mk_layout), L), append(stride(mk_layout), size(filter_zeros(mk_layout))));
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFB.
  template <class ProblemShape, class SFVecShape>
  CUTE_HOST_DEVICE
  static constexpr auto 
  tile_atom_to_shape_SFB(ProblemShape problem_shape, SFVecShape sf_vec_shape) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto [M, N, K, L] = problem_shape_MNKL;
      auto [sfm, sfn, sfk] = sf_vec_shape;

      if constexpr (majorSFB == UMMA::Major::MN) {
        return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(N, sfn)));
      }
      else {
        return make_stride(make_stride(_0{}, cute::ceil_div(K, sfk)), make_stride(_0{}, _1{}));
      }
    }();

    auto [M, N, K, L] = problem_shape_MNKL;
    auto [sfm, sfn, sfk] = sf_vec_shape;
    auto nk_layout = make_layout(
      make_shape(make_shape(sfn, cute::ceil_div(N, sfn)),
                 make_shape(sfk, cute::ceil_div(K, sfk))),
      strides
    );

    return make_layout(append(shape(nk_layout), L), append(stride(nk_layout), size(filter_zeros(nk_layout))));
  }

};

// Sm90 only supports MN major for SFA and SFB for now
template<int SFVecSizeM, int SFVecSizeN, int SFVecSizeK, cute::GMMA::Major majorSFA = cute::GMMA::Major::MN, cute::GMMA::Major majorSFB = cute::GMMA::Major::MN>
using Sm90BlockwiseScaleConfig = Sm1xxBlockwiseScaleConfig<
    SFVecSizeM, 
    SFVecSizeN, 
    SFVecSizeK, 
    majorSFA == cute::GMMA::Major::MN ? UMMA::Major::MN : UMMA::Major::K, 
    majorSFB == cute::GMMA::Major::MN ? UMMA::Major::MN : UMMA::Major::K>;

template<int SFVecSizeM, int SFVecSizeN, int SFVecSizeK, UMMA::Major majorSFA = UMMA::Major::MN, UMMA::Major majorSFB = UMMA::Major::MN>
using Sm100BlockwiseScaleConfig = Sm1xxBlockwiseScaleConfig<SFVecSizeM, SFVecSizeN, SFVecSizeK, majorSFA, majorSFB>;

template<int SFVecSizeM, int SFVecSizeN, int SFVecSizeK, UMMA::Major majorSFA = UMMA::Major::MN, UMMA::Major majorSFB = UMMA::Major::MN>
using Sm120BlockwiseScaleConfig = Sm1xxBlockwiseScaleConfig<SFVecSizeM, SFVecSizeN, SFVecSizeK, majorSFA, majorSFB>;

template<class MmaTileShape_MNK>
constexpr auto sm90_trivial_blockwise_scale_config(MmaTileShape_MNK) {
  return Sm90BlockwiseScaleConfig<size<0>(MmaTileShape_MNK{}), size<1>(MmaTileShape_MNK{}), size<2>(MmaTileShape_MNK{})>{};
}

template<class MmaTileShape_MNK>
constexpr auto sm100_trivial_blockwise_scale_config(MmaTileShape_MNK) {
  return Sm100BlockwiseScaleConfig<size<0>(MmaTileShape_MNK{}), size<1>(MmaTileShape_MNK{}), size<2>(MmaTileShape_MNK{})>{};
}

template<class MmaTileShape_MNK>
constexpr auto sm120_trivial_blockwise_scale_config(MmaTileShape_MNK) {
  return Sm120BlockwiseScaleConfig<size<0>(MmaTileShape_MNK{}), size<1>(MmaTileShape_MNK{}), size<2>(MmaTileShape_MNK{})>{};
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
