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

/*! \file
  \brief Fusion callbacks specializations for the sm100 TMA warp-specialized (ws) epilogue
*/


#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

#include "cutlass/epilogue/fusion/sm100_visitor_compute_tma_warpspecialized.hpp"  
#include "cutlass/epilogue/fusion/sm100_visitor_store_tma_warpspecialized.hpp" 

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Sm100 Tma warp specialized callbacks just alias to their sm90 counterpart
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class Operation,
  class CtaTile_MNK,
  class EpilogueTile_MN,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    Operation,
    CtaTile_MNK,
    EpilogueTile_MN,
    Args...
> : FusionCallbacks<
      epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...
    > {
  using FusionCallbacks<
      epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...>::FusionCallbacks;
};

// Sm100 direct store callbacks alias to sm100 tma callbacks with 0 stages
// Additional copy atom args will be ignored in the 0-stage specializations of aux load/store nodes
template <
  class Operation,
  class CtaTile_MNK,
  class EpilogueTile_MN,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm100NoSmemWarpSpecialized,
    Operation,
    CtaTile_MNK,
    EpilogueTile_MN,
    Args...
> : FusionCallbacks<
      epilogue::Sm100TmaWarpSpecialized<0, 0, 0, false, false>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...
    > {
  using FusionCallbacks<
      epilogue::Sm100TmaWarpSpecialized<0, 0, 0, false, false>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...>::FusionCallbacks;
};

// Sm100 Ptr array tma warp specialized callbacks just alias to their sm90 counterpart
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class Operation,
  class CtaTile_MNK,
  class EpilogueTile_MN,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    Operation,
    CtaTile_MNK,
    EpilogueTile_MN,
    Args...
> : FusionCallbacks<
      epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, 1>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...
    > {
  using FusionCallbacks<
      epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, 1>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...>::FusionCallbacks;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C
// With Row BlockScaleFactor Generation.
template<
  int SFVecsize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinearCombRowBlockScaleFactor =
  Sm90EVT<Sm100BlockScaleFactorRowStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>, // gen scalefactor
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinearCombRowBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100LinearCombRowBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {
          {
            // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}                    // ternary args : multiply_add
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };
  
  // Ctor inheritance
  using Impl::Impl;
};

// D = alpha * acc + beta * C
// With Col BlockScaleFactor Generation.
template<
  int SFVecsize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinearCombColBlockScaleFactor =
  Sm90EVT<Sm100BlockScaleFactorColStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>, // gen scalefactor
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::ColumnMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinearCombColBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100LinearCombColBlockScaleFactor<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::ColumnMajor,  ElementSource, ElementScalar, RoundStyle>;  

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {
          {
            // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}                    // ternary args : multiply_add
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// For Ptr-Array and Grouped GEMM
// D = alpha * acc + beta * C, where alpha and beta can be vectors for each batch/group
// With Row BlockScaleFactor Generation, separate tensors per batch/group.
template<
  int SFVecsize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinearCombRowBlockScaleFactorPtrArray =
  Sm90EVT<Sm100BlockScaleFactorRowStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor *, RoundStyle>, // gen scalefactor
    Sm90LinearCombinationPtrArray<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinearCombRowBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100LinearCombRowBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombBlockScaleFactor<SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;
    ElementScalar const* const* beta_ptr_array = nullptr;
    ElementBlockScaleFactor ** block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    // NormConst is a single device-side constant value, its not per-batch or per-group
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {
          {
            // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {beta_ptr_array}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}                    // ternary args : multiply_add
          },
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };
  
  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// For Ptr-Array and Grouped GEMM
// D = activation(alpha * acc + beta * C), where alpha and beta can be vectors for each batch/group
// With Row BlockScaleFactor Generation, separate tensors per batch/group.
template<
  int SFVecsize,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombEltActRowBlockScaleFactorPtrArray =
  Sm90EVT<Sm100BlockScaleFactorRowStore<SFVecsize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor *, RoundStyle>, // gen scalefactor
    Sm90LinCombEltActPtrArray<ActivationFn, ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // activation(beta * C + (alpha * acc))
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombEltActBlockScaleFactor<ActivationFn, SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombEltActRowBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, ActivationFn, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle> {

  using Impl =  Sm100LinCombEltActRowBlockScaleFactorPtrArray<SFVecSize, EpilogueTile, ActivationFn, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombEltActBlockScaleFactor<ActivationFn, SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;
    ElementScalar const* const* beta_ptr_array = nullptr;
    ElementBlockScaleFactor ** block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {
          {    // unary op: activation(beta * C + (alpha * acc))
            {    // ternary op : beta * C + (alpha * acc)
              {{beta}, {beta_ptr}, {beta_ptr_array}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // binary op : alpha * acc
                {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {}                  // binary args : multiplies
              },                    // end binary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C + per-row bias
//   with row blockScaled generation
template<
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerRowBiasRowBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorRowStore<
      SFVecsize, EpilogueTile, ElementOutput, 
      ElementCompute, ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerRowBias<
      CtaTileShapeMNK, ElementCompute, ElementCompute, 
      ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor,
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerRowBiasRowBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, 
      ElementScalar, 
      AlignmentBias,
       RoundStyle
    > 
{

  using Impl = 
    Sm100LinCombPerRowBiasRowBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerRowBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {
          {  // ternary op : beta * C + (alpha * acc + bias)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                  // ternary args : multiply_add
            },                    // end ternary op
            {} // ternary args : multiply_add
          },  // end ternary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// D = alpha * acc + beta * C + per-row bias
//   with col blockScaled generation
template<
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerRowBiasColBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorColStore<
      SFVecsize, EpilogueTile, ElementOutput, 
      ElementCompute, ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerRowBias<
      CtaTileShapeMNK, ElementCompute, ElementCompute, 
      ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::ColumnMajor,
      ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerRowBiasColBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    > 
{

  using Impl = 
    Sm100LinCombPerRowBiasColBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerRowBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::ColumnMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {
          {  // ternary op : beta * C + (alpha * acc + bias)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                  // ternary args : multiply_add
            },                    // end ternary op
            {} // ternary args : multiply_add
          },  // end ternary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C + per_col bias
//   with row blockScaled generation
template<
  int StagesC,
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerColBiasRowBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorRowStore<
      SFVecsize, EpilogueTile, ElementOutput, 
      ElementCompute, ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerColBias<
      StagesC, CtaTileShapeMNK, EpilogueTile, ElementCompute, ElementCompute, 
      ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerColBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor,
      ElementBias, ElementSource, 
      ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerColBiasRowBlockScaleFactor<
      StagesC, SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    > 
{

  using Impl = 
    Sm100LinCombPerColBiasRowBlockScaleFactor<
      StagesC, SFVecSize, CtaTileShapeMNK, EpilogueTile, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerColBiasBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor,
      ElementBias, ElementSource, 
      ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};


    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {
          {  // ternary op : beta * C + (alpha * acc + bias)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                  // ternary args : multiply_add
            },                    // end ternary op
            {} // ternary args : multiply_add
          },  // end ternary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per-row bias) 
//   with row blockScaled generation
template<
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerRowBiasEltActRowBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorRowStore<
      SFVecsize, EpilogueTile, 
      ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, 
      ElementCompute, ElementCompute, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerRowBiasEltActRowBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    > {

  using Impl = 
    Sm100LinCombPerRowBiasEltActRowBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerRowBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};
    
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {
          {    // unary op : activation(beta * C + (alpha * acc + bias))
            {    // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per-row bias) 
//   with col blockScaled generation
template<
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerRowBiasEltActColBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorColStore<
      SFVecsize, EpilogueTile, 
      ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, 
      ElementCompute, ElementCompute, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::ColumnMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerRowBiasEltActColBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    > {

  using Impl = 
    Sm100LinCombPerRowBiasEltActColBlockScaleFactor<
      SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerRowBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::ColumnMajor, 
      ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};


    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};
    
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {
          {    // unary op : activation(beta * C + (alpha * acc + bias))
            {    // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per_col bias) 
//   with row blockScaled generation
template<
  int StagesC,
  int SFVecsize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor, 
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm100LinCombPerColBiasEltActRowBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorRowStore<
      SFVecsize, EpilogueTile, 
      ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, RoundStyle
    >,
    Sm90LinCombPerColBiasEltAct<
      StagesC, CtaTileShapeMNK, EpilogueTile, ActivationFn, 
      ElementCompute, ElementCompute, ElementBias, 
      ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  int SFVecSize,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerColBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor,
      ElementBias, ElementSource, 
      ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm100LinCombPerColBiasEltActRowBlockScaleFactor<
      StagesC, SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    > {

  using Impl = 
    Sm100LinCombPerColBiasEltActRowBlockScaleFactor<
      StagesC, SFVecSize, CtaTileShapeMNK, EpilogueTile, ActivationFn, 
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, 
      ElementCompute, ElementBlockScaleFactor, ElementBias, ElementSource, ElementScalar, 
      AlignmentBias, RoundStyle
    >;

  using Operation = 
    fusion::LinCombPerColBiasEltActBlockScaleFactor<
      ActivationFn, SFVecSize, ElementOutput, ElementCompute, 
      ElementBlockScaleFactor, cutlass::layout::RowMajor,
      ElementBias, ElementSource, 
      ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor * block_scale_factor_ptr = nullptr;
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    using StrideNormConst = Stride<_0,_0,int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};
    
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {
          {    // unary op : activation(beta * C + (alpha * acc + bias))
            {    // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst} // BlockScaleFactor args
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};


// --------------------------------------------------------------------
//  Sm100PtrArrayNoSmemWarpSpecialized  (direct-store, grouped GEMM)
// --------------------------------------------------------------------
template <
    class Operation,
    class CtaTile_MNK,
    class EpilogueTile_MN,
    class... Args
>
struct FusionCallbacks<
        epilogue::Sm100PtrArrayNoSmemWarpSpecialized,
        Operation,
        CtaTile_MNK,
        EpilogueTile_MN,
        Args...>
  : FusionCallbacks<
        // reuse the ptr-array *TMA* callbacks with 0 stages
        epilogue::Sm100PtrArrayTmaWarpSpecialized<0,0,0,false,false>,
        Operation,
        CtaTile_MNK,
        EpilogueTile_MN,
        Args...> {

  using Base = FusionCallbacks<
      epilogue::Sm100PtrArrayTmaWarpSpecialized<0,0,0,false,false>,
      Operation,
      CtaTile_MNK,
      EpilogueTile_MN,
      Args...>;

  // bring ctors into scope
  using Base::Base;
};

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
