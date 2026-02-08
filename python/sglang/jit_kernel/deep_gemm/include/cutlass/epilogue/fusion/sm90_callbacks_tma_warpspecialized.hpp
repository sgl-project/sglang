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
  \brief Fusion callbacks specializations for the sm90 TMA warp-specialized (ws) epilogue
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"

#include "cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class NodeOp, class... ChildOps>
using Sm90EVT = Sm90TreeVisitor<NodeOp, ChildOps...>;

// D = alpha * acc
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::ScaledAcc<ElementOutput, ElementCompute, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90EVT<Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, 
      Sm90AccFetch
    > {
  using Impl = 
    Sm90EVT<Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>,
      Sm90AccFetch
    >;
  using Operation = fusion::ScaledAcc<ElementOutput, ElementCompute, ElementScalar, RoundStyle>;

  struct Arguments {
    // Give a name and flat ordering to the fusion callback args
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};

    // Conversion to the args expected by the visitor implementation
    // to_underlying_arguments will implicitly call this
    operator typename Impl::Arguments() const {
      return
        {    // binary op : alpha * acc
          {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
          {},                     // leaf args : acc
          {} // binary args : multiplies
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C
template<
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinearCombination =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
    Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
      Sm90AccFetch // acc
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
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + (alpha * acc)
          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // binary op : alpha * acc
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {}                  // binary args : multiplies
          },                    // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C, where beta and alpha can be vectors for each batch
template<
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinearCombinationPtrArray =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
    Sm90ScalarBroadcastPtrArray<ElementScalar, Stride<_0,_0,int64_t>>, // beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
      Sm90ScalarBroadcastPtrArray<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
      Sm90AccFetch // acc
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, 
                                             StagesD, 
                                             FragmentSize, 
                                             ReuseSmemC, 
                                             DelayTmaStore, 
                                             NumEpilogueWarpGroups
                                            >,
    fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinearCombinationPtrArray<typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm90LinearCombinationPtrArray<typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;
    ElementScalar const* const* beta_ptr_array = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + (alpha * acc)
          {{beta}, {beta_ptr}, {beta_ptr_array}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // binary op : alpha * acc
            {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {}                  // binary args : multiplies
          },                    // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C)
template<
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombEltAct =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>, // activation(beta * C + (alpha * acc))
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
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
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombEltAct<ActivationFn, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombEltAct<ActivationFn, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm90LinCombEltAct<ActivationFn, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombEltAct<ActivationFn, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {    // unary op: activation(beta * C + (alpha * acc))
          {    // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {} // ternary args : multiply_add
          },   // end ternary op
          activation // unary args: activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C), where beta and alpha can be vectors for each batch
template<
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombEltActPtrArray =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>, // activation(beta * C + (alpha * acc))
    Sm90LinearCombinationPtrArray<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, 
                                             StagesD, 
                                             FragmentSize, 
                                             ReuseSmemC, 
                                             DelayTmaStore, 
                                             NumEpilogueWarpGroups
                                            >,
    fusion::LinCombEltAct<ActivationFn, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombEltActPtrArray<ActivationFn, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm90LinCombEltActPtrArray<ActivationFn, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombEltAct<ActivationFn, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;
    ElementScalar const* const* beta_ptr_array = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
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
          activation // unary args: activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C + per-row bias
template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerRowBias =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + bias)
    Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
      Sm90AccFetch, // acc
      Sm90ColBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_1,_0,int64_t>, AlignmentBias> // bias
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
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBias<ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombPerRowBias<
      CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle> {
  using Impl = Sm90LinCombPerRowBias<
    CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>;
  using Operation = fusion::LinCombPerRowBias<
    ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {     // ternary op : beta * C + (alpha * acc + bias)
          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // ternary op : alpha * acc + bias
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
            {}                  // ternary args : multiply_add
          },                    // end ternary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = alpha * acc + beta * C + per-column bias
template<
  int StagesC,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerColBias =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + bias)
    Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
      Sm90AccFetch, // acc
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias> // bias
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
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerColBias<ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombPerColBias<
      StagesC, CtaTileShapeMNK, EpilogueTile, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle> {
  using Impl = Sm90LinCombPerColBias<
    StagesC, CtaTileShapeMNK, EpilogueTile, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>;
  using Operation = fusion::LinCombPerColBias<
    ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {     // ternary op : beta * C + (alpha * acc + bias)
          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // ternary op : alpha * acc + bias
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
            {}                  // ternary args : multiply_add
          },                    // end ternary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per-row bias)
template<
  class CtaTileShapeMNK,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerRowBiasEltAct =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
    Sm90LinCombPerRowBias<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>
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
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    > {

  using Impl =
    Sm90LinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::LinCombPerRowBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

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
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per-column bias)
template<
  int StagesC,
  class CtaTileShapeMNK,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerColBiasEltAct =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
    Sm90LinCombPerColBias<StagesC, CtaTileShapeMNK, EpilogueTile, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>
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
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerColBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombPerColBiasEltAct<
      StagesC, CtaTileShapeMNK, EpilogueTile, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    > {

  using Impl =
    Sm90LinCombPerColBiasEltAct<
      StagesC, CtaTileShapeMNK, EpilogueTile, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::LinCombPerColBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

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
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(alpha * acc + beta * C + per-row bias)
// Aux = alpha * acc + beta * C + per-row bias)
template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int Stages,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpR2S,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerRowBiasEltActAux =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
    Sm90EVT<Sm90AuxStore<Stages, EpilogueTile, ElementAux, RoundStyle, StrideAux, SmemLayoutAtom, CopyOpR2S, AlignmentAux>,
      Sm90LinCombPerRowBias<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpR2S
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerRowBiasEltActAux<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpR2S
> : Sm90LinCombPerRowBiasEltActAux<
      CtaTileShapeMNK, EpilogueTile, StagesD, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, SmemLayoutAtom, CopyOpR2S, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    > {

  using Impl =
    Sm90LinCombPerRowBiasEltActAux<
      CtaTileShapeMNK, EpilogueTile, StagesD, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, SmemLayoutAtom, CopyOpR2S, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::LinCombPerRowBiasEltActAux<
      GmemLayoutTagAux, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      return
        {    // unary op : activation(store(beta * C + (alpha * acc + bias)))
          {                 // unary op : store(beta * C + (alpha * acc + bias))
            {                  // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {}               // ternary args : multiply_add
            },                 // end ternary op
            {aux_ptr, dAux} // unary args : store
          },                // end unary op
          activation // unary args : activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// D = activation(alpha * acc + beta * C + per_col bias)
// Aux = alpha * acc + beta * C + per_col bias)
template<
  int StagesC,
  class CtaTileShapeMNK,
  class EpilogueTile,
  int Stages,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpR2S,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerColBiasEltActAux =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
    Sm90EVT<Sm90AuxStore<Stages, EpilogueTile, ElementAux, RoundStyle, StrideAux, SmemLayoutAtom, CopyOpR2S, AlignmentAux>,
      Sm90LinCombPerColBias<StagesC, CtaTileShapeMNK, EpilogueTile, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpR2S
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombPerColBiasEltActAux<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpR2S
> : Sm90LinCombPerColBiasEltActAux<
      StagesC, CtaTileShapeMNK, EpilogueTile, StagesD, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, SmemLayoutAtom, CopyOpR2S, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    > {

  using Impl =
    Sm90LinCombPerColBiasEltActAux<
      StagesC, CtaTileShapeMNK, EpilogueTile, StagesD, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, SmemLayoutAtom, CopyOpR2S, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::LinCombPerColBiasEltActAux<
      GmemLayoutTagAux, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      return
        {    // unary op : activation(store(beta * C + (alpha * acc + bias)))
          {                 // unary op : store(beta * C + (alpha * acc + bias))
            {                  // ternary op : beta * C + (alpha * acc + bias)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // ternary op : alpha * acc + bias
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {}                  // ternary args : multiply_add
              },                    // end ternary op
              {}               // ternary args : multiply_add
            },                 // end ternary op
            {aux_ptr, dAux} // unary args : store
          },                // end unary op
          activation // unary args : activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = per-row alpha * acc + per-row beta * C + per-row bias
template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90PerRowLinCombPerRowBias =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + bias)
    Sm90ColBroadcast<0, CtaTileShapeMNK, ElementScalar, ElementCompute, Stride<bool,_0,int64_t>, AlignmentScalar>, // beta, dynamic scalar/vector broadcast
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90ColBroadcast<0, CtaTileShapeMNK, ElementScalar, ElementCompute, Stride<bool,_0,int64_t>, AlignmentScalar>, // alpha, dynamic scalar/vector broadcast
      Sm90AccFetch, // acc
      Sm90ColBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_1,_0,int64_t>, AlignmentBias> // bias
    >
  >;

// D = activation(per-row alpha * acc + per-row beta * C + per-row bias)
template<
  class CtaTileShapeMNK,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90PerRowLinCombPerRowBiasEltAct =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
    Sm90PerRowLinCombPerRowBias<CtaTileShapeMNK, ElementCompute, ElementCompute,
                                ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle>
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
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  int AlignmentScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::PerRowLinCombPerRowBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90PerRowLinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    > {

  using Impl =
    Sm90PerRowLinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    >;
  using Operation =
    fusion::PerRowLinCombPerRowBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    >;

  struct Arguments {
    using StrideAlpha = Stride<bool,_0,int64_t>;
    using StrideBeta  = Stride<bool,_0,int64_t>;
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    StrideAlpha dAlpha = {bool(1), _0{}, 0};
    StrideBeta  dBeta  = {bool(1), _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {    // unary op : activation(beta * C + (alpha * acc + bias))
          {    // ternary op : beta * C + (alpha * acc + bias)
            {beta_ptr, beta, dBeta}, // leaf args : beta
            {},                      // leaf args : C
            {                        // ternary op : alpha * acc + bias
              {alpha_ptr, alpha, dAlpha}, // leaf args : alpha
              {},                         // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                     // ternary args : multiply_add
            },                       // end ternary op
            {} // ternary args : multiply_add
          },   // end ternary op
          activation // unary args : activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = per-col alpha * acc + per-col beta * C + per-column bias
template<
  int StagesC,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90PerColLinCombPerColBias =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + bias)
    Sm90RowBroadcast<0, CtaTileShapeMNK, ElementScalar, ElementCompute, Stride<_0,bool,int64_t>, AlignmentScalar>, // beta, dynamic scalar/vector broadcast
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementScalar, ElementCompute, Stride<_0,bool,int64_t>, AlignmentScalar>, // alpha, dynamic scalar/vector broadcast
      Sm90AccFetch, // acc
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias> // bias
    >
  >;

// D = activation(per-col alpha * acc + per-col beta * C + per-column bias)
template<
  int StagesC,
  class CtaTileShapeMNK,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90PerColLinCombPerColBiasEltAct =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
    Sm90PerColLinCombPerColBias<StagesC, CtaTileShapeMNK, EpilogueTile, ElementCompute, ElementCompute,
                                ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle>
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
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  int AlignmentScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::PerColLinCombPerColBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90PerColLinCombPerColBiasEltAct<
      StagesC, CtaTileShapeMNK, EpilogueTile, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    > {

  using Impl =
    Sm90PerColLinCombPerColBiasEltAct<
      StagesC, CtaTileShapeMNK, EpilogueTile, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    >;
  using Operation =
    fusion::PerColLinCombPerColBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,bool,int64_t>;
    using StrideBeta  = Stride<_0,bool,int64_t>;
    StrideAlpha dAlpha = {_0{}, bool(1), 0};
    StrideBeta  dBeta  = {_0{}, bool(1), 0};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {    // unary op : activation(beta * C + (alpha * acc + bias))
          {    // ternary op : beta * C + (alpha * acc + bias)
            {beta_ptr, beta, dBeta}, // leaf args : beta
            {},               // leaf args : C
            {                 // ternary op : alpha * acc + bias
              {alpha_ptr, alpha, dAlpha},   // leaf args : alpha
              {},                   // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}              // ternary args : multiply_add
            },                // end ternary op
            {} // ternary args : multiply_add
          },   // end ternary op
          activation // unary args : activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(per-col alpha * acc + per-column bias) + per-col beta * C
template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90PerColResAddPerColBiasEltAct =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + activation(alpha * acc + bias)
    Sm90RowBroadcast<0, CtaTileShapeMNK, ElementScalar, ElementCompute, Stride<_0,bool,int64_t>, AlignmentScalar>, // beta, dynamic scalar/vector broadcast
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // activation(alpha * acc + bias)
      Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
        Sm90RowBroadcast<0, CtaTileShapeMNK, ElementScalar, ElementCompute, Stride<_0,bool,int64_t>, AlignmentScalar>, // alpha, dynamic scalar/vector broadcast
        Sm90AccFetch, // acc
        Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias> // bias
      >
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
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  int AlignmentScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::PerColResAddPerColBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90PerColResAddPerColBiasEltAct<
      CtaTileShapeMNK, EpilogueTile, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    > {

  using Impl =
    Sm90PerColResAddPerColBiasEltAct<
      CtaTileShapeMNK, EpilogueTile, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    >;
  using Operation =
    fusion::PerColResAddPerColBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,bool,int64_t>;
    using StrideBeta  = Stride<_0,bool,int64_t>;
    StrideAlpha dAlpha = {_0{}, bool(1), 0};
    StrideBeta  dBeta  = {_0{}, bool(1), 0};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + activation(alpha * acc + bias)
          {beta_ptr, beta, dBeta}, // leaf args : beta
          {},                      // leaf args : C
          {                        // unary op : activation(alpha * acc + bias)
            {                          // ternary op : alpha * acc + bias
              {alpha_ptr, alpha, dAlpha},        // leaf args : alpha
              {},                                // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {}                       // ternary args : multiply_add
            },                         // end ternary op
            activation             // unary args : activation
          },                       // end unary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T>
constexpr bool is_fp8_v = cute::is_same_v<T,float_e4m3_t> || cute::is_same_v<T,float_e5m2_t>;

// We only apply the scaling factor if output is fp8
template <typename ElementOutput>
struct ScaleOutOp { template <typename T> using Op = cutlass::first<T>; };
template <>
struct ScaleOutOp<float_e4m3_t> { template <typename T> using Op = cutlass::multiplies<T>; };
template <>
struct ScaleOutOp<float_e5m2_t> { template <typename T> using Op = cutlass::multiplies<T>; };

template <typename T>
using amax = cutlass::maximum_absolute_value_reduction<T, true>; // propogate nans

}; // end namespace detail

// D = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-row bias
template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledLinCombPerRowBias =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + bias)
    Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>, 2>, // scale_c * beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>, 3>, // scale_a * scale_b * alpha
      Sm90AccFetch, // acc
      Sm90ColBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_1,_0,int64_t>, AlignmentBias> // bias
    >
  >;

// Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + per-row bias
// if D is fp8 
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
template<
  class CtaTileShapeMNK,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledLinCombPerRowBiasEltAct =
  Sm90EVT<Sm90Compute<detail::ScaleOutOp<ElementOutput>::template Op, ElementOutput, ElementCompute, RoundStyle>, // activation(Z) * scale_d
    Sm90EVT<Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // activation(Z)
      // Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + per-row bias
      Sm90ScaledLinCombPerRowBias<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>
    >,
    Sm90ScalarBroadcast<ElementScalar> // scale_d
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
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::ScaledLinCombPerRowBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90ScaledLinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    > {

  using Impl =
    Sm90ScaledLinCombPerRowBiasEltAct<
      CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::ScaledLinCombPerRowBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    ElementScalar scale_a = ElementScalar(1);
    ElementScalar scale_b = ElementScalar(1);
    ElementScalar scale_c = ElementScalar(1);
    ElementScalar scale_d = ElementScalar(1);
    ElementScalar const* scale_a_ptr = nullptr;
    ElementScalar const* scale_b_ptr = nullptr;
    ElementScalar const* scale_c_ptr = nullptr;
    ElementScalar const* scale_d_ptr = nullptr;

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
        {    // binary op : activation((scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias)) * scale_d
          {    // unary op : activation((scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias))
            {    // ternary op : (scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias)
              {{beta, scale_c},
               {beta_ptr, scale_c_ptr},
               {dBeta, {_0{}, _0{}, 0}}
               },  // leaf args : (scale_c * beta)
              {},  // leaf args : C
              {    // ternary op : (scale_a * scale_b * alpha) * acc + bias
                {{alpha, scale_a, scale_b}, 
                 {alpha_ptr, scale_a_ptr, scale_b_ptr},
                 {dAlpha, {_0{}, _0{}, 0}, {_0{}, _0{}, 0}}
                 },                   // leaf args : (scale_a * scale_b * alpha)
                {},                   // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {} // ternary args : multiply_add
              },   // end ternary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {{scale_d},
           {scale_d_ptr}
           },   // leaf args : scale_d
          {} // binary args : multiplies or first
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-col bias
template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledLinCombPerColBias =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + bias)
    Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>, 2>, // scale_c * beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>, 3>, // scale_a * scale_b * alpha
      Sm90AccFetch, // acc
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias> // bias
    >
  >;

// Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + per-col bias
// if D is fp8 
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
template<
  class CtaTileShapeMNK,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledLinCombPerColBiasEltAct =
  Sm90EVT<Sm90Compute<detail::ScaleOutOp<ElementOutput>::template Op, ElementOutput, ElementCompute, RoundStyle>, // activation(Z) * scale_d
    Sm90EVT<Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // activation(Z)
      // Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + per-row bias
      Sm90ScaledLinCombPerColBias<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>
    >,
    Sm90ScalarBroadcast<ElementScalar> // scale_d
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
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::ScaledLinCombPerColBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90ScaledLinCombPerColBiasEltAct<
      CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    > {

  using Impl =
    Sm90ScaledLinCombPerColBiasEltAct<
      CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::ScaledLinCombPerColBiasEltAct<
      ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    ElementScalar scale_a = ElementScalar(1);
    ElementScalar scale_b = ElementScalar(1);
    ElementScalar scale_c = ElementScalar(1);
    ElementScalar scale_d = ElementScalar(1);
    ElementScalar const* scale_a_ptr = nullptr;
    ElementScalar const* scale_b_ptr = nullptr;
    ElementScalar const* scale_c_ptr = nullptr;
    ElementScalar const* scale_d_ptr = nullptr;

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
        {    // binary op : activation((scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias)) * scale_d
          {    // unary op : activation((scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias))
            {    // ternary op : (scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias)
              {{beta, scale_c},
               {beta_ptr, scale_c_ptr},
               {dBeta, {_0{}, _0{}, 0}}
               },  // leaf args : (scale_c * beta)
              {},  // leaf args : C
              {    // ternary op : (scale_a * scale_b * alpha) * acc + bias
                {{alpha, scale_a, scale_b}, 
                 {alpha_ptr, scale_a_ptr, scale_b_ptr},
                 {dAlpha, {_0{}, _0{}, 0}, {_0{}, _0{}, 0}}
                 },                   // leaf args : (scale_a * scale_b * alpha)
                {},                   // leaf args : acc
                {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
                {} // ternary args : multiply_add
              },   // end ternary op
              {} // ternary args : multiply_add
            },   // end ternary op
            activation // unary args : activation
          },   // end unary op
          {{scale_d},
           {scale_d_ptr}
           },   // leaf args : scale_d
          {} // binary args : multiplies or first
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-row bias
// if D is fp8 
//   amax_d = max(abs(elements in activation(Z)))
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
// if Aux is fp8 
//   amax_aux = max(abs(elements in Z))
//   Aux = scale_aux * Z
// else
//   Aux = Z

// fp8 aux specialization
template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int StagesD,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpR2S,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementAmax = ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledLinCombPerRowBiasEltActAmaxAuxFp8 =
  Sm90SplitTreeVisitor<
    // Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-row bias
    Sm90ScaledLinCombPerRowBias<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>,
    // D = activation(Z) * scale_d, amax_d = max(abs(elements in D))
    Sm90EVT<Sm90Compute<detail::ScaleOutOp<ElementOutput>::template Op, ElementOutput, ElementCompute, RoundStyle>, // activation(Z) * scale_d
      Sm90EVT<Sm90ScalarReduction<detail::amax, atomic_maximum, ElementAmax, ElementCompute, RoundStyle>, // amax_d
        Sm90EVT<Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // activation(Z)
          Sm90SplitTreeFetch // Z
        >
      >,
      Sm90ScalarBroadcast<ElementScalar> // scale_d
    >,
    // Aux = Z * scale_aux, amax_aux = max(abs(elements in Aux))
    Sm90EVT<Sm90AuxStore<StagesD, EpilogueTile, ElementAux, RoundStyle, StrideAux, SmemLayoutAtom, CopyOpR2S, AlignmentAux>, // store(Aux)
      Sm90EVT<Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // Z * scale_aux
        Sm90EVT<Sm90ScalarReduction<detail::amax, atomic_maximum, ElementAmax, ElementCompute, RoundStyle>, // amax_aux
          Sm90SplitTreeFetch // Z
        >,
        Sm90ScalarBroadcast<ElementScalar> // scale_aux
      >
    >
  >;

// non-fp8 aux specialization
// lets us use some EVT specializations such as relu + uint1b_t aux
template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int StagesD,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpR2S,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementAmax = ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledLinCombPerRowBiasEltActAmaxAuxNotFp8 =
  // D = activation(Z) * scale_d, amax_d = max(abs(elements in D))
  Sm90EVT<Sm90Compute<detail::ScaleOutOp<ElementOutput>::template Op, ElementOutput, ElementCompute, RoundStyle>, // activation(Z) * scale_d
    Sm90EVT<Sm90ScalarReduction<detail::amax, atomic_maximum, ElementAmax, ElementCompute, RoundStyle>, // amax_d
      Sm90EVT<Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // activation(Z)
        Sm90EVT<Sm90AuxStore<StagesD, EpilogueTile, ElementAux, RoundStyle, StrideAux, SmemLayoutAtom, CopyOpR2S, AlignmentAux>, // Aux = Z
          // Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-row bias
          Sm90ScaledLinCombPerRowBias<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>
        >
      >
    >,
    Sm90ScalarBroadcast<ElementScalar> // scale_d
  >;

// dispatcher
template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int StagesD,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpR2S,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementAmax = ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledLinCombPerRowBiasEltActAmaxAux = conditional_t<detail::is_fp8_v<ElementAux>,
  Sm90ScaledLinCombPerRowBiasEltActAmaxAuxFp8<
    CtaTileShapeMNK, EpilogueTile, StagesD, StrideAux, SmemLayoutAtom, CopyOpR2S, ActivationFn,
    ElementOutput, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar,AlignmentAux, AlignmentBias, RoundStyle
  >,
  Sm90ScaledLinCombPerRowBiasEltActAmaxAuxNotFp8<
    CtaTileShapeMNK, EpilogueTile, StagesD, StrideAux, SmemLayoutAtom, CopyOpR2S, ActivationFn,
    ElementOutput, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
  >
>;


template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementAmax,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpR2S
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpR2S
> : Sm90ScaledLinCombPerRowBiasEltActAmaxAux<
      CtaTileShapeMNK, EpilogueTile, StagesD, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
      SmemLayoutAtom, CopyOpR2S, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    > {

  using Impl =
    Sm90ScaledLinCombPerRowBiasEltActAmaxAux<
      CtaTileShapeMNK, EpilogueTile, StagesD, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
      SmemLayoutAtom, CopyOpR2S, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    ElementScalar scale_a = ElementScalar(1);
    ElementScalar scale_b = ElementScalar(1);
    ElementScalar scale_c = ElementScalar(1);
    ElementScalar scale_d = ElementScalar(1);
    ElementScalar const* scale_a_ptr = nullptr;
    ElementScalar const* scale_b_ptr = nullptr;
    ElementScalar const* scale_c_ptr = nullptr;
    ElementScalar const* scale_d_ptr = nullptr;

    ElementScalar scale_aux = ElementScalar(1);
    ElementScalar const* scale_aux_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    ElementAmax* amax_D_ptr = nullptr;
    ElementAmax* amax_aux_ptr = nullptr;

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      // Only compute amax_d if D is fp8
      ElementAmax* amax_D_ptr_ = nullptr;
      if constexpr (detail::is_fp8_v<ElementOutput>) {
        amax_D_ptr_ = amax_D_ptr;
      }

      // Aux is fp8 -> DAG arguments
      if constexpr (detail::is_fp8_v<ElementAux>) {
        typename Impl::Arguments args;
        // always use structured binding to unpack DAG args since it may or may not be a tuple
        auto& [Z_args, aux_args, D_args] = args;

        Z_args =
          {    // ternary op : (scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias)
            {{beta, scale_c},
             {beta_ptr, scale_c_ptr},
             {dBeta, {_0{}, _0{}, 0}}
             },  // leaf args : (scale_c * beta)
            {},  // leaf args : C
            {    // ternary op : (scale_a * scale_b * alpha) * acc + bias
              {{alpha, scale_a, scale_b}, 
               {alpha_ptr, scale_a_ptr, scale_b_ptr},
               {dAlpha ,{_0{}, _0{}, 0}, {_0{}, _0{}, 0}}
               },                   // leaf args : (scale_a * scale_b * alpha)
              {},                   // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {} // ternary args : multiply_add
            },   // end ternary op
            {} // ternary args : multiply_add
          };   // end ternary op

        D_args =
          {    // binary op : activation(Z) * scale_d or activation(Z)
            {    // unary op : reduce(activation(Z))
              {             // unary op : activation(Z)
                {},             // leaf args : Z
                activation      // unary args : activation
              },                // end unary op
              {amax_D_ptr_} // unary args : reduce
            },              // end unary op
            {{scale_d},
             {scale_d_ptr}
             },  // leaf args : scale_d
            {} // binary args : multiplies or first
          };   // end binary op

        aux_args =
          {    // unary op : store(Aux)
            {    // binary op : Z * scale_d or Z
              {    // unary op : reduce(Z)
                {},            // leaf args : Z
                {amax_aux_ptr} // unary args : reduce
              },   // end unary op
              {{scale_aux},
               {scale_aux_ptr}
               },  // leaf args : scale_d
              {} // binary args : multiplies
            },   // end binary op
            {aux_ptr, dAux} // unary args : store
          };   // end unary op

        return args;
      }

      // Aux is not fp8 -> Tree arguments
      else {
        return
          {  // binary op : activation(Z) * scale_d or activation(Z)
            {  // unary op : reduce(activation(Z))
              {  // unary op : activation(Z)
                {  // unary op : store(Z)
                  {  // ternary op : (scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias)
                    {{beta, scale_c},
                     {beta_ptr, scale_c_ptr},
                     {dBeta, {_0{}, _0{}, 0}}
                    },                // leaf args : (scale_c * beta)
                    {},               // leaf args : C
                    {                 // ternary op : (scale_a * scale_b * alpha) * acc + bias
                      {{alpha, scale_a, scale_b}, 
                       {alpha_ptr, scale_a_ptr, scale_b_ptr},
                       {dAlpha, {_0{}, _0{}, 0}}
                      },                // leaf args : (scale_a * scale_b * alpha)
                      {},               // leaf args : acc
                      {bias_ptr, ElementBias(0), dBias
                      },                // leaf args : bias
                      {}              // ternary args : multiply_add
                    },                // end ternary op
                    {}              // ternary args : multiply_add
                  },                // end ternary op
                  {aux_ptr, dAux} // unary args : store
                },                // end unary op
                activation     // unary args : activation
              },               // end unary op
              {amax_D_ptr_} // unary args : reduce
            },              // end unary op
            {{scale_d},{scale_d_ptr}}, // leaf args : scale_d
            {} // binary args : multiplies or first
          };   // end binary op
      }
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-col bias
// if D is fp8 
//   amax_d = max(abs(elements in activation(Z)))
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
// if Aux is fp8 
//   amax_aux = max(abs(elements in Z))
//   Aux = scale_aux * Z
// else
//   Aux = Z

// fp8 aux specialization
template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int StagesD,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpR2S,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementAmax = ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledLinCombPerColBiasEltActAmaxAuxFp8 =
  Sm90SplitTreeVisitor<
    // Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-col bias
    Sm90ScaledLinCombPerColBias<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>,
    // D = activation(Z) * scale_d, amax_d = max(abs(elements in D))
    Sm90EVT<Sm90Compute<detail::ScaleOutOp<ElementOutput>::template Op, ElementOutput, ElementCompute, RoundStyle>, // activation(Z) * scale_d
      Sm90EVT<Sm90ScalarReduction<detail::amax, atomic_maximum, ElementAmax, ElementCompute, RoundStyle>, // amax_d
        Sm90EVT<Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // activation(Z)
          Sm90SplitTreeFetch // Z
        >
      >,
      Sm90ScalarBroadcast<ElementScalar> // scale_d
    >,
    // Aux = Z * scale_aux, amax_aux = max(abs(elements in Aux))
    Sm90EVT<Sm90AuxStore<StagesD, EpilogueTile, ElementAux, RoundStyle, StrideAux, SmemLayoutAtom, CopyOpR2S, AlignmentAux>, // store(Aux)
      Sm90EVT<Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // Z * scale_aux
        Sm90EVT<Sm90ScalarReduction<detail::amax, atomic_maximum, ElementAmax, ElementCompute, RoundStyle>, // amax_aux
          Sm90SplitTreeFetch // Z
        >,
        Sm90ScalarBroadcast<ElementScalar> // scale_aux
      >
    >
  >;

// non-fp8 aux specialization
// lets us use some EVT specializations such as relu + uint1b_t aux
template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int StagesD,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpR2S,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementAmax = ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledLinCombPerColBiasEltActAmaxAuxNotFp8 =
  // D = activation(Z) * scale_d, amax_d = max(abs(elements in D))
  Sm90EVT<Sm90Compute<detail::ScaleOutOp<ElementOutput>::template Op, ElementOutput, ElementCompute, RoundStyle>, // activation(Z) * scale_d
    Sm90EVT<Sm90ScalarReduction<detail::amax, atomic_maximum, ElementAmax, ElementCompute, RoundStyle>, // amax_d
      Sm90EVT<Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // activation(Z)
        Sm90EVT<Sm90AuxStore<StagesD, EpilogueTile, ElementAux, RoundStyle, StrideAux, SmemLayoutAtom, CopyOpR2S, AlignmentAux>, // Aux = Z
          // Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-row bias
          Sm90ScaledLinCombPerColBias<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>
        >
      >
    >,
    Sm90ScalarBroadcast<ElementScalar> // scale_d
  >;

// dispatcher
template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int StagesD,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpR2S,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementAmax = ElementCompute,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledLinCombPerColBiasEltActAmaxAux = conditional_t<detail::is_fp8_v<ElementAux>,
  Sm90ScaledLinCombPerColBiasEltActAmaxAuxFp8<
    CtaTileShapeMNK, EpilogueTile, StagesD, StrideAux, SmemLayoutAtom, CopyOpR2S, ActivationFn,
    ElementOutput, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar,AlignmentAux, AlignmentBias, RoundStyle
  >,
  Sm90ScaledLinCombPerColBiasEltActAmaxAuxNotFp8<
    CtaTileShapeMNK, EpilogueTile, StagesD, StrideAux, SmemLayoutAtom, CopyOpR2S, ActivationFn,
    ElementOutput, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
  >
>;


template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementAmax,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpR2S
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::ScaledLinCombPerColBiasEltActAmaxAux<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpR2S
> : Sm90ScaledLinCombPerColBiasEltActAmaxAux<
      CtaTileShapeMNK, EpilogueTile, StagesD, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
      SmemLayoutAtom, CopyOpR2S, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    > {

  using Impl =
    Sm90ScaledLinCombPerColBiasEltActAmaxAux<
      CtaTileShapeMNK, EpilogueTile, StagesD, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
      SmemLayoutAtom, CopyOpR2S, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::ScaledLinCombPerColBiasEltActAmaxAux<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementAmax, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    ElementScalar scale_a = ElementScalar(1);
    ElementScalar scale_b = ElementScalar(1);
    ElementScalar scale_c = ElementScalar(1);
    ElementScalar scale_d = ElementScalar(1);
    ElementScalar const* scale_a_ptr = nullptr;
    ElementScalar const* scale_b_ptr = nullptr;
    ElementScalar const* scale_c_ptr = nullptr;
    ElementScalar const* scale_d_ptr = nullptr;

    ElementScalar scale_aux = ElementScalar(1);
    ElementScalar const* scale_aux_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    ElementAmax* amax_D_ptr = nullptr;
    ElementAmax* amax_aux_ptr = nullptr;

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      // Only compute amax_d if D is fp8
      ElementAmax* amax_D_ptr_ = nullptr;
      if constexpr (detail::is_fp8_v<ElementOutput>) {
        amax_D_ptr_ = amax_D_ptr;
      }

      // Aux is fp8 -> DAG arguments
      if constexpr (detail::is_fp8_v<ElementAux>) {
        typename Impl::Arguments args;
        // always use structured binding to unpack DAG args since it may or may not be a tuple
        auto& [Z_args, aux_args, D_args] = args;

        Z_args =
          {    // ternary op : (scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias)
            {{beta, scale_c},
             {beta_ptr, scale_c_ptr},
             {dBeta, {_0{}, _0{}, 0}}
             },  // leaf args : (scale_c * beta)
            {},  // leaf args : C
            {    // ternary op : (scale_a * scale_b * alpha) * acc + bias
              {{alpha, scale_a, scale_b}, 
               {alpha_ptr, scale_a_ptr, scale_b_ptr},
               {dAlpha, {_0{}, _0{}, 0}, {_0{}, _0{}, 0}}
               },                   // leaf args : (scale_a * scale_b * alpha)
              {},                   // leaf args : acc
              {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
              {} // ternary args : multiply_add
            },   // end ternary op
            {} // ternary args : multiply_add
          };   // end ternary op

        D_args =
          {    // binary op : activation(Z) * scale_d or activation(Z)
            {    // unary op : reduce(activation(Z))
              {             // unary op : activation(Z)
                {},             // leaf args : Z
                activation      // unary args : activation
              },                // end unary op
              {amax_D_ptr_} // unary args : reduce
            },              // end unary op
            {{scale_d},
             {scale_d_ptr}
             },  // leaf args : scale_d
            {} // binary args : multiplies or first
          };   // end binary op

        aux_args =
          {    // unary op : store(Aux)
            {    // binary op : Z * scale_d or Z
              {    // unary op : reduce(Z)
                {},            // leaf args : Z
                {amax_aux_ptr} // unary args : reduce
              },   // end unary op
              {{scale_aux},
               {scale_aux_ptr}
               },  // leaf args : scale_d
              {} // binary args : multiplies
            },   // end binary op
            {aux_ptr, dAux} // unary args : store
          };   // end unary op

        return args;
      }

      // Aux is not fp8 -> Tree arguments
      else {
        return
          {  // binary op : activation(Z) * scale_d or activation(Z)
            {  // unary op : reduce(activation(Z))
              {  // unary op : activation(Z)
                {  // unary op : store(Z)
                  {  // ternary op : (scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias)
                    {{beta, scale_c},
                    {beta_ptr, scale_c_ptr},
                    {dBeta, {_0{}, _0{}, 0}}
                    },  // leaf args : (scale_c * beta)
                    {},               // leaf args : C
                    {                 // ternary op : (scale_a * scale_b * alpha) * acc + bias
                      {{alpha, scale_a, scale_b}, 
                       {alpha_ptr, scale_a_ptr, scale_b_ptr},
                       {dAlpha, {_0{}, _0{}, 0}, {_0{}, _0{}, 0}}
                      },                // leaf args : (scale_a * scale_b * alpha)
                      {},               // leaf args : acc
                      {bias_ptr, ElementBias(0), dBias
                      },                // leaf args : bias
                      {}              // ternary args : multiply_add
                    },                // end ternary op
                    {}              // ternary args : multiply_add
                  },                // end ternary op
                  {aux_ptr, dAux} // unary args : store
                },                // end unary op
                activation     // unary args : activation
              },               // end unary op
              {amax_D_ptr_} // unary args : reduce
            },              // end unary op
            {{scale_d},{scale_d_ptr}}, // leaf args : scale_d
            {} // binary args : multiplies or first
          };   // end binary op
      }
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int Stages,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpS2R,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombDeEltAct =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>, // activation(beta * C + (alpha * acc), aux)
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle>, // beta * C + (alpha * acc)
    Sm90AuxLoad<Stages, EpilogueTile, ElementAux, StrideAux, SmemLayoutAtom, CopyOpS2R, AlignmentAux> // aux
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpS2R
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpS2R
> : Sm90LinCombDeEltAct<
      CtaTileShapeMNK, EpilogueTile, StagesC, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, SmemLayoutAtom, CopyOpS2R, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    > {

  using Impl =
    Sm90LinCombDeEltAct<
      CtaTileShapeMNK, EpilogueTile, StagesC, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, SmemLayoutAtom, CopyOpS2R, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >;
  using Operation =
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux const* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      return
        {    // binary op : activation(beta * C + (alpha * acc), aux)
          {                  // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}               // ternary args : multiply_add
          },                 // end ternary op
          {aux_ptr, ElementAux(0), dAux}, // leaf args : aux
          activation // binary args : activation
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  int Stages,
  class StrideAux,
  class SmemLayoutAtom,
  class CopyOpS2R,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementBias = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentAux = 128 / sizeof_bits_v<ElementAux>,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombDeEltActDePerRowBias =
  Sm90EVT<Sm90Compute<cutlass::epilogue::thread::Identity, ElementOutput, ElementCompute, RoundStyle>, // Identity for final conversion
    Sm90EVT<Sm90ColReduction<plus, plus, plus, 0, CtaTileShapeMNK,
                             ElementBias, ElementCompute, RoundStyle, Stride<_1,_0,int64_t>, AlignmentBias>,
      Sm90LinCombDeEltAct<CtaTileShapeMNK, EpilogueTile, Stages, StrideAux, SmemLayoutAtom, CopyOpS2R, ActivationFn,
                          ElementCompute, ElementCompute, ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle>
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux,
  class ElementBias,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpS2R
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombDeEltActDePerRowBias<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpS2R
> : Sm90LinCombDeEltActDePerRowBias<
      CtaTileShapeMNK, EpilogueTile, StagesC, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, SmemLayoutAtom, CopyOpS2R, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    > {

  using Impl =
    Sm90LinCombDeEltActDePerRowBias<
      CtaTileShapeMNK, EpilogueTile, StagesC, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, SmemLayoutAtom, CopyOpS2R, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;
  using Operation =
    fusion::LinCombDeEltActDePerRowBias<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux const* aux_ptr = nullptr;
    StrideAux dAux = {};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias* dbias_ptr = nullptr;
    StrideBias dDbias = {};

    operator typename Impl::Arguments() const {
      return
      {   // unary op : identity/convert
        {    // unary op : reduce(activation(beta * C + (alpha * acc), aux))
          {    // binary op : activation(beta * C + (alpha * acc), aux)
            {                  // ternary op : beta * C + (alpha * acc)
              {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
              {},                   // leaf args : C
              {                     // binary op : alpha * acc
                {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                {},                     // leaf args : acc
                {}                  // binary args : multiplies
              },                    // end binary op
              {}               // ternary args : multiply_add
            },                 // end ternary op
            {aux_ptr, ElementAux(0), dAux}, // leaf args : aux
            activation // binary args : activation
          },   // end binary op
          {dbias_ptr, ElementCompute(0), dDbias} // unary args : reduce
        },   // end unary op
        {} // unary args : identity/convert
      };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = softmax(top_k(alpha * acc + beta * C))
template<
  int TopK,
  int FragmentSize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombTopKSoftmaxCol =
  Sm90EVT<Sm90TopKSoftmaxColReduction<TopK, FragmentSize, CtaTileShapeMNK, EpilogueTile, ElementOutput, ElementCompute, RoundStyle>, // softmax(top_k(beta * C + (alpha * acc)))
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  int TopK,
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombTopKSoftmaxCol<TopK, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinCombTopKSoftmaxCol<TopK, FragmentSize, CtaTileShapeMNK, EpilogueTile, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm90LinCombTopKSoftmaxCol<TopK, FragmentSize, CtaTileShapeMNK, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombTopKSoftmaxCol<TopK, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    operator typename Impl::Arguments() const {
      return
        {    // unary op: activation(beta * C + (alpha * acc))
          {    // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {} // ternary args : multiply_add
          },   // end ternary op
          {} // unary args: activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Grouped Wgrad Conv
template<
  class GroupsPerTile,
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinearCombinationGroupedWgrad =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
    Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
      Sm90ScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
      Sm90AccFetchGroupedWgrad<GroupsPerTile> // acc
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
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class GroupsPerTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinearCombinationGroupedWgrad<GroupsPerTile, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90LinearCombinationGroupedWgrad<GroupsPerTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle> {

  using Impl = Sm90LinearCombinationGroupedWgrad<GroupsPerTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinearCombinationGroupedWgrad<GroupsPerTile, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    //ElementScalar groups = ElementScalar(1);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + (alpha * acc)
          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // binary op : alpha * acc
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {}                  // binary args : multiplies
          },                    // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {
template <class FusionOpOrCallbacks, class = cute::void_t<>>
struct get_element_aux {
  using type = void;
};

template <class FusionOpOrCallbacks>
struct get_element_aux<FusionOpOrCallbacks, cute::void_t<typename FusionOpOrCallbacks::ElementAux>> {
  using type = typename FusionOpOrCallbacks::ElementAux;
};

template <class NodeOp, class... ChildOps>
struct get_element_aux<Sm90TreeVisitor<NodeOp, ChildOps...>, cute::void_t<>> {
  using type = typename get_element_aux<NodeOp>::type;
};

template <class... Ts>
struct get_element_aux<FusionCallbacks<Ts...>, cute::void_t<typename FusionCallbacks<Ts...>::Operation>> {
 private:
  using Operation = typename FusionCallbacks<Ts...>::Operation;
 public:
  using type = typename get_element_aux<Operation>::type;
};
} // namespace cutlass:epilogue::fusion::detail

template <class Callbacks>
using get_element_aux_t = typename detail::get_element_aux<Callbacks>::type;

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
