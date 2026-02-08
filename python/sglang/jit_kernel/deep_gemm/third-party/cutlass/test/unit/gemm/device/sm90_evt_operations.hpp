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
    \brief Host reference and operations for Sm90 EVT unit test
*/
#pragma once
#include "gemm_testbed_3x_evt.hpp"

//////////////////////////////////////////////////////////////////////////////
/// Host references used for testing
namespace test::gemm::device {
template<class NodeOp, class ...ChildOp>
using HEVT = HostTreeVisitor<NodeOp, ChildOp...>;

template<class EdgeTuple, class ...Ops>
using HDAG = HostTopoVisitor<EdgeTuple, Ops...>;

template<class InputTree, class OutputTree, class... AuxOutTrees>
using HST = HostSplitTreeVisitor<InputTree, OutputTree, AuxOutTrees...>;

/// D = alpha * acc + beta * C + AuxLoad
template<class Gemm, class ElementAux, class LayoutAux>
class HostEVTAuxLoad {
public:
  using ElementC = typename Gemm::GemmKernel::ElementC;
  using LayoutC = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideC>;
  using ElementD = typename Gemm::GemmKernel::ElementC;
  using LayoutD = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideD>;

  using ScalarAlpha = HostScalarBroadcast<1>;
  using AccFetchNode = HostAccumulator<>;
  using AuxLoadNode = HostAuxLoad<ElementAux, LayoutAux, false>;
  using TernaryCompute0 = HEVT<HostCompute<cutlass::homogeneous_multiply_add>, ScalarAlpha, AccFetchNode, AuxLoadNode>;
  using ScalarBeta = HostScalarBroadcast<1>;
  using CLoadNode = HostAuxLoad<ElementC, LayoutC, true>;
  using TernaryCompute1 = HEVT<HostCompute<cutlass::homogeneous_multiply_add>, ScalarBeta, CLoadNode, TernaryCompute0>;
  using EVTModule = HEVT<HostAuxStore<ElementD, LayoutD, true>, TernaryCompute1>;
};

/// D = alpha * acc + beta * C + per-column bias
template<class Gemm, class ElementBias>
class HostPerColBias {
public:
  using ElementC = typename Gemm::GemmKernel::ElementC;
  using LayoutC = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideC>;
  using ElementD = typename Gemm::GemmKernel::ElementC;
  using LayoutD = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideD>;

  using ScalarAlpha = HostScalarBroadcast<1>;
  using AccFetchNode = HostAccumulator<>;
  using RowBroadcastNode = HostRowBroadcast<ElementBias>;
  using TernaryCompute0 = HEVT<HostCompute<cutlass::homogeneous_multiply_add>, ScalarAlpha, AccFetchNode, RowBroadcastNode>;
  using ScalarBeta = HostScalarBroadcast<1>;
  using CLoadNode = HostAuxLoad<ElementC, LayoutC, true>;
  using TernaryCompute1 = HEVT<HostCompute<cutlass::homogeneous_multiply_add>, ScalarBeta, CLoadNode, TernaryCompute0>;
  using EVTModule = HEVT<HostAuxStore<ElementD, LayoutD, true>, TernaryCompute1>;
};

/// D = beta * C + Graph(relu(alpha * acc + aux) + aux)
/// Testing EVT - DAG structure
template<class Gemm>
class HostEVTDAG {
public:
  using ElementC = typename Gemm::GemmKernel::ElementC;
  using LayoutC = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideC>;
  using ElementD = typename Gemm::GemmKernel::ElementC;
  using LayoutD = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideD>;

  using ScalarAlpha = HostScalarBroadcast<1>;
  using AccFetchNode = HostAccumulator<>;
  using AuxLoadNode = HostAuxLoad<cutlass::half_t, cutlass::layout::RowMajor, false>;
  using DAGNode = HDAG<
    float,
    cute::tuple<
      cute::tuple<>, // 0. alpha
      cute::tuple<>, // 1. acc
      cute::tuple<>, // 2. aux load
      cute::tuple<cute::_0, cute::_1, cute::_2>, // 3. alpha * acc + aux load
      cute::tuple<cute::_3>, // relu(alpha * acc + aux load)
      cute::tuple<cute::_2, cute::_4> // relu(alpha * acc + aux load) + aux load
    >,
    ScalarAlpha,
    AccFetchNode,
    AuxLoadNode,
    HostCompute<cutlass::homogeneous_multiply_add>,
    HostCompute<cutlass::epilogue::thread::ReLu>,
    HostCompute<cutlass::plus>
  >;
  using ScalarBeta = HostScalarBroadcast<1>;
  using CLoadNode = HostAuxLoad<ElementC, LayoutC, true>;
  using TernaryCompute1 = HEVT<HostCompute<cutlass::homogeneous_multiply_add>, ScalarBeta, CLoadNode, DAGNode>;
  using EVTModule = HEVT<HostAuxStore<ElementD, LayoutD, true>, TernaryCompute1>;
};

/// EVT = alpha * acc + C
/// D = Graph(maximum(EVT + per-row bias, EVT))
/// Testing DAG - EVT
template<class Gemm>
class HostDAGEVT {
public:
  using ElementC = typename Gemm::GemmKernel::ElementC;
  using LayoutC = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideC>;
  using ElementD = typename Gemm::GemmKernel::ElementC;
  using LayoutD = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideD>;

  using EVTNode = HEVT<
    HostAuxStore<cutlass::half_t, cutlass::layout::RowMajor, false>,
    HEVT<
      HostCompute<cutlass::homogeneous_multiply_add>,
      HostScalarBroadcast<2>,
      HostAccumulator<>,
      HostAuxLoad<ElementC, LayoutC, true>
    >
  >;
  using EVTModule = HEVT<
    HostAuxStore<ElementD, LayoutD, true>,
    HDAG<
      float,
      cute::tuple<
      cute::tuple<>, // 0. EVT
      cute::tuple<>, // 1. per-row bias
      cute::tuple<cute::_0, cute::_1>, // 2. EVT + per-row bias
      cute::tuple<cute::_0, cute::_2> // 3. maximum(EVT + per-row bias, EVT)
      >,
      EVTNode,
      HostColBroadcast<cutlass::half_t, cute::Stride<cute::_1,cute::_0,int>>,
      HostCompute<cutlass::plus>,
      HostCompute<cutlass::maximum_with_default_nan_propagation>
    >
  >;
};

/// Xreduce(alpha * acc + beta * C)
template<class Gemm, class ReduceOp>
class HostReduce {
public:
  using ElementC = typename Gemm::GemmKernel::ElementC;
  using LayoutC = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideC>;
  using ElementD = typename Gemm::GemmKernel::ElementC;
  using LayoutD = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideD>;

  using ScalarAlpha = HostScalarBroadcast<1>;
  using AccFetchNode = HostAccumulator<>;
  using BinaryCompute0 = HEVT<HostCompute<cutlass::multiplies>, ScalarAlpha, AccFetchNode>;
  using ScalarBeta = HostScalarBroadcast<1>;
  using CLoadNode = HostAuxLoad<ElementC, LayoutC, true>;
  using TernaryCompute1 = HEVT<HostCompute<cutlass::homogeneous_multiply_add>, ScalarBeta, CLoadNode, BinaryCompute0>;
  using ReduceNode = HEVT<ReduceOp, TernaryCompute1>;
  using EVTModule = HEVT<HostAuxStore<ElementD, LayoutD, true>, ReduceNode>;
};

// Z = scale_a * scale_b * alpha * acc + beta * scale_c * C + per-row bias
// if D is fp8
//   D = scale_d * activation(Z)
// else
//   D = activation(Z)
template <class Gemm, template <class> class ActivationFn, class ElementD>
class HostScaledLinCombPerRowBiasEltAct {
public:
  using ElementC = typename Gemm::GemmKernel::ElementC;
  using LayoutC = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideC>;
  using LayoutD = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideD>;

  using EVTModule = HEVT<
  HostAuxStore<ElementD, LayoutD, true>,
  HEVT<
    HostCompute<cutlass::epilogue::fusion::detail::ScaleOutOp<ElementD>::template Op>,  // activation(Z) * scaled_d
    HEVT<
      HostCompute<ActivationFn>, // activation(Z)
      HEVT<
        HostCompute<cutlass::homogeneous_multiply_add>,
        HostScalarBroadcast<1, 2, cute::Stride<cute::_0,cute::_0,int64_t>>, // scale_c * beta
        HostAuxLoad<ElementC, LayoutC, true>, // C
        HEVT<
          HostCompute<cutlass::homogeneous_multiply_add>,
          HostScalarBroadcast<1, 3, cute::Stride<cute::_0,cute::_0,int64_t>>, // scale_a * scale_b * alpha
          HostAccumulator<>,
          HostColBroadcast<ElementD, cute::Stride<cute::_1,cute::_0,int64_t>>
        >
      >
    >,
    HostScalarBroadcast<1> // scale_d
  >
  >;
};

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
template <class Gemm, template <class> class ActivationFn, class ElementD, class ElementAux = ElementD>
class HostScaledLinCombPerRowBiasEltActAmaxAux {
public:
  using ElementC = typename Gemm::GemmKernel::ElementC;
  using LayoutC = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideC>;
  using LayoutD = cutlass::detail::StrideToLayoutTagC_t<typename Gemm::GemmKernel::StrideD>;

  template <typename T>
  using amax = cutlass::maximum_absolute_value_reduction<T, true>;
  using EVTModuleAuxFp8 = HEVT<
    HostAuxStore<ElementD, LayoutD, true>,
    HST<float,
      // Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-row bias
      HEVT<
        HostCompute<cutlass::homogeneous_multiply_add>,
        HostScalarBroadcast<1, 2, cute::Stride<cute::_0,cute::_0,int64_t>>, // scale_c * beta
        HostAuxLoad<ElementC, LayoutC, true>, // C
        HEVT<
          HostCompute<cutlass::homogeneous_multiply_add>,
          HostScalarBroadcast<1, 3, cute::Stride<cute::_0,cute::_0,int64_t>>, // scale_a * scale_b * alpha
          HostAccumulator<>,
          HostColBroadcast<ElementD, cute::Stride<cute::_1,cute::_0,int64_t>>
        >
      >,
      // D = activation(Z) * scaled_d, amax_d = max(abs(elements in D))
      HEVT<
        HostCompute<cutlass::epilogue::fusion::detail::ScaleOutOp<ElementD>::template Op>,
        HEVT<
          HostScalarReduce<amax, float>,
          HEVT<
            HostCompute<ActivationFn>, //activation(Z) * scaled_d
            HostAccumulator<> // Z
          >
        >,
        HostScalarBroadcast<1> // scale_d
      >,
      // Aux = Z * scale_aux, amax_aux = max(abs(elements in Aux))
      HEVT<
        HostAuxStore<ElementAux, cutlass::layout::RowMajor, false>,
        HEVT<
          HostCompute<cutlass::multiplies>,
          HEVT<
            HostScalarReduce<amax, float>,
            HostAccumulator<>
            >,
          HostScalarBroadcast<1>
        >
      >
    >
  >;

  using EVTModuleAuxNotFp8 = HEVT<
    // D = activation(Z) * scaled_d, amax_d = max(abs(elements in D))
    HostAuxStore<ElementD, LayoutD, true>,
      HEVT<
        HostCompute<cutlass::epilogue::fusion::detail::ScaleOutOp<ElementD>::template Op>,
        HEVT<
          HostScalarReduce<amax, float>,
          HEVT<
            HostCompute<ActivationFn>, //activation(Z) * scaled_d
            HEVT<
              // Aux = Z
              HostAuxStore<ElementAux, cutlass::layout::RowMajor, false>,
              // Z = scale_a * scale_b * alpha * acc + scale_c * beta * C + per-row bias
              HEVT<
                HostCompute<cutlass::homogeneous_multiply_add>,
                HostScalarBroadcast<1, 2, cute::Stride<cute::_0,cute::_0,int64_t>>, // scale_c * beta
                HostAuxLoad<ElementC, LayoutC, true>, // C
                HEVT<
                  HostCompute<cutlass::homogeneous_multiply_add>,
                  HostScalarBroadcast<1, 3, cute::Stride<cute::_0,cute::_0,int64_t>>, // scale_a * scale_b * alpha
                  HostAccumulator<>,
                  HostColBroadcast<ElementD, cute::Stride<cute::_1,cute::_0,int64_t>>
                >
              >
            >
          >
        >,
        HostScalarBroadcast<1> // scale_d
      >
    >;
      
  using EVTModule = cute::conditional_t<cutlass::epilogue::fusion::detail::is_fp8_v<ElementAux>, EVTModuleAuxFp8, EVTModuleAuxNotFp8>;

};
} // namespace test::gemm::device

//////////////////////////////////////////////////////////////////////////////
namespace cutlass::epilogue {
namespace fusion {

namespace detail {

template <typename T>
struct maximum_with_default_nan_propagation : maximum<T> {};

} // namespace detail

//////////////////////////////////////////////////////////////////////////////
/// D = alpha * acc + beta * C + AuxLoad
template<
  class EpilogueDescriptor,
  class AuxLoadDescriptor,
  class ElementOutput,
  class ElementCompute,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombAuxLoad =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + bias)
    Sm90ScalarBroadcast<ElementScalar>, // beta
    Sm90SrcFetch<ElementOutput>, // C
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90ScalarBroadcast<ElementScalar>, // alpha
      Sm90AccFetch, // acc
      Sm90AuxLoad<
        AuxLoadDescriptor::Stages, typename EpilogueDescriptor::EpilogueTile,
        typename AuxLoadDescriptor::Element,
        typename AuxLoadDescriptor::Stride, typename AuxLoadDescriptor::SmemLayoutAtom,
        typename AuxLoadDescriptor::CopyOpS2R // aux load
      >
    >
  >;

//////////////////////////////////////////////////////////////////////////////
/// D = alpha * acc + beta * C + AuxLoadNoSmem
template<
  class EpilogueDescriptor,
  class ElementAux,
  class StrideAux,
  class ElementOutput,
  class ElementCompute,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombAuxLoadNoSmem =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + bias)
    Sm90ScalarBroadcast<ElementScalar>, // beta
    Sm90SrcFetch<ElementOutput>, // C
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90ScalarBroadcast<ElementScalar>, // alpha
      Sm90AccFetch, // acc
      Sm90AuxLoad<0, void, ElementAux, StrideAux, void, void> // aux load
    >
  >;

//////////////////////////////////////////////////////////////////////////////
/// Example DAG
/// beta * C + Graph(alpha * acc + gamma + acc)
template<
  typename EpilogueDescriptor,
  typename AuxLoadDescriptor,
  class ElementOutput,
  class ElementCompute,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombEVTDAG =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + aux)
    Sm90ScalarBroadcast<ElementScalar>, // beta
    Sm90SrcFetch<ElementOutput>, // C
    Sm90TopologicalVisitor<
      ElementCompute,
      cute::tuple<
        cute::seq<>, // 0. alpha
        cute::seq<>, // 1. acc
        cute::seq<>, // 2. aux load
        cute::seq<1, 0, 2>, // 3. alpha * acc + aux load
        cute::seq<3>, // relu(alpha & acc + aux load)
        cute::seq<2, 4> // relu(alpha * acc + aux load) + aux load
      >,
      Sm90ScalarBroadcast<ElementScalar>, // alpha
      Sm90AccFetch, // acc
      Sm90AuxLoad<
        AuxLoadDescriptor::Stages, typename EpilogueDescriptor::EpilogueTile,
        typename AuxLoadDescriptor::Element, typename AuxLoadDescriptor::Stride,
        typename AuxLoadDescriptor::SmemLayoutAtom, typename AuxLoadDescriptor::CopyOpS2R>,
      Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>,
      Sm90Compute<cutlass::epilogue::thread::ReLu, ElementCompute, ElementCompute, RoundStyle>,
      Sm90Compute<plus, ElementCompute, ElementCompute, RoundStyle>
    >
    >;


//////////////////////////////////////////////////////////////////////////////
/// Example DAG
/// EVT = alpha * acc + C
/// D = Graph(maximum(EVT + per-row bias, EVT))
template<
  class EpilogueDescriptor,
  class AuxStoreDescriptor,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombDAGEVT =
  Sm90TopologicalVisitor<
    ElementCompute,
    cute::tuple<
      cute::seq<>,
      cute::seq<>,
      cute::seq<1, 0>,
      cute::seq<0, 2>
    >,
    Sm90EVT<
      Sm90AuxStore<
        AuxStoreDescriptor::Stages, typename EpilogueDescriptor::EpilogueTile,
        typename AuxStoreDescriptor::Element, RoundStyle, typename AuxStoreDescriptor::Stride,
        typename AuxStoreDescriptor::SmemLayoutAtom, typename AuxStoreDescriptor::CopyOpR2S>,
      Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>,
        Sm90ScalarBroadcast<ElementScalar>,
        Sm90AccFetch,
        Sm90SrcFetch<ElementOutput>
      >
    >,
    Sm90ColBroadcast<0, typename EpilogueDescriptor::TileShape, ElementBias, ElementCompute>,
    Sm90Compute<plus, ElementCompute, ElementCompute, RoundStyle>,
    Sm90Compute<detail::maximum_with_default_nan_propagation, ElementOutput, ElementCompute, RoundStyle>
  >;


//////////////////////////////////////////////////////////////////////////////
/// D = alpha * acc + beta * C + per-column bias
template<
  class EpilogueDescriptor,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerColumnBias =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc + bias)
    Sm90ScalarBroadcast<ElementScalar>, // beta
    Sm90SrcFetch<ElementOutput>, // C
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc + bias
      Sm90ScalarBroadcast<ElementScalar>, // alpha
      Sm90AccFetch, // acc
      Sm90RowBroadcast<0, typename EpilogueDescriptor::TileShape, ElementBias, ElementCompute>
    >
  >;


//////////////////////////////////////////////////////////////////////////////
/// D = per-column reduce(alpha * acc + beta * C)
template<
  template <class> class RegReduceFn,
  template <class> class GmemReduceFn,
  class ElementReduce,
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerColumnReduce =
  Sm90EVT<Sm90RowReduction<RegReduceFn, RegReduceFn, GmemReduceFn, 0, CtaTileShapeMNK, ElementReduce, ElementCompute, RoundStyle>, // per column reduce
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + alpha * acc
      Sm90ScalarBroadcast<ElementScalar>, // beta
      Sm90SrcFetch<ElementOutput>, // C
      Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
        Sm90ScalarBroadcast<ElementScalar>, // alpha
        Sm90AccFetch // acc
      >
    >
  >;


//////////////////////////////////////////////////////////////////////////////
/// D = per-row reduce(alpha * acc + beta * C)
template<
  template <class> class RegReduceFn,
  template <class> class GmemReduceFn,
  class ElementReduce,
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombPerRowReduce =
  Sm90EVT<Sm90ColReduction<RegReduceFn, RegReduceFn, GmemReduceFn, 0, CtaTileShapeMNK, ElementReduce, ElementCompute, RoundStyle>, // per column reduce
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + alpha * acc
      Sm90ScalarBroadcast<ElementScalar>, // beta
      Sm90SrcFetch<ElementOutput>, // C
      Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
        Sm90ScalarBroadcast<ElementScalar>, // alpha
        Sm90AccFetch // acc
      >
    >
  >;


//////////////////////////////////////////////////////////////////////////////
/// D = scalar reduce(alpha * acc + beta * C)
template<
  template <class> class RegReduceFn,
  template <class> class GmemReduceFn,
  class ElementReduce,
  class ElementOutput,
  class ElementCompute,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90LinCombScalarReduce =
  Sm90EVT<Sm90ScalarReduction<RegReduceFn, GmemReduceFn, ElementReduce, ElementCompute, RoundStyle>, // per column reduce
    Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + alpha * acc
      Sm90ScalarBroadcast<ElementScalar>, // beta
      Sm90SrcFetch<ElementOutput>, // C
      Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
        Sm90ScalarBroadcast<ElementScalar>, // alpha
        Sm90AccFetch // acc
      >
    >
  >;
} // namespace fusion

} // namespace cutlass::epilogue
