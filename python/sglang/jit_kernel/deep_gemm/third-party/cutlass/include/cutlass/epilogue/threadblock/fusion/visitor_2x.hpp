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
  \brief Visitor tree operation base implementation to enable composable fusions
         for the CUTLASS 2x epilogue
*/

#pragma once

#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using cute::tuple;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <class... Ops>
struct VisitorImpl2x: fusion::detail::Sm90VisitorImplBase<Ops...> {
  using fusion::detail::Sm90VisitorImplBase<Ops...>::Sm90VisitorImplBase;
  using fusion::detail::Sm90VisitorImplBase<Ops...>::ops;

  template <class CallbacksTuple>
  struct Callbacks {
    // Callbacks can store non-persistent variables (e.g. tensors) or copies of persistent variables
    CallbacksTuple callbacks_tuple;

    /// Called at the start of the epilogue just before iterating over accumulator slices
    CUTLASS_DEVICE void
    begin_epilogue() {
      for_each(callbacks_tuple,
        [] (auto& callbacks) {
          callbacks.begin_epilogue();
        }
      );
    }

    /// Called at the start of one step before starting accumulator exchange
    CUTLASS_DEVICE void
    begin_step(int step_idx) {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.begin_step(step_idx);
        }
      );
    }

    /// Called at the start of a row
    CUTLASS_DEVICE void
    begin_row(int row_idx) {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.begin_row(row_idx);
        }
      );
    }

    /// Called after accumulators have been exchanged for each accumulator vector
    template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc,
          Array<ElementInputs, FragmentSize> const&... frg_inputs) // depends on the N-naryness of the op
      = delete; // Must be implemented for each operation

    /// Called at the start of a row
    CUTLASS_DEVICE void
    end_row(int row_idx) {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.end_row(row_idx);
        }
      );
    }

    /// Called after all accumulator elements have been visited
    CUTLASS_DEVICE void
    end_step(int step_idx) {
      for_each(callbacks_tuple,
        [&] (auto& callbacks) {
          callbacks.end_step(step_idx);
        }
      );
    }

    /// Called after all steps have been completed
    CUTLASS_DEVICE void
    end_epilogue() {
      for_each(callbacks_tuple,
        [] (auto& callbacks) {
          callbacks.end_epilogue();
        }
      );
    }
  };

  // Callbacks factory
  // All operations must redefine this
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    return transform_apply(ops,
      [&] (auto& op) {
        return op.get_callbacks(
          threadblock_tile_offset,
          thread_idx,
          problem_shape);
      },
      [] (auto&&... callbacks) {
        auto callbacks_tuple = cute::make_tuple(callbacks...);
        return Callbacks<decltype(callbacks_tuple)>{callbacks_tuple};
      }
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Convenience aliases
using EmptyCallbacks = VisitorImpl2x<>::Callbacks<cute::tuple<>>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tree visitor
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class NodeOp, class... ChildOps>
struct TreeVisitor2x : VisitorImpl2x<ChildOps..., NodeOp> {

  using VisitorImpl2x<ChildOps..., NodeOp>::VisitorImpl2x;

  template<class CallbacksImpl>
  struct Callbacks : CallbacksImpl {
    CUTLASS_DEVICE
    Callbacks(CallbacksImpl&& impl)
      : CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}

    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      constexpr int Rm1 = sizeof...(ChildOps);
      return cute::detail::tapply(callbacks_tuple,
        [&] (auto& child_callbacks) {
          return child_callbacks.visit(iter_idx, row_idx, column_idx, frg_idx, frg_acc);
        },
        [&] (auto&&... frg_inputs) {
          return get<Rm1>(callbacks_tuple).visit(iter_idx, row_idx, column_idx, frg_idx, frg_acc, frg_inputs...);
        },
        make_seq<Rm1>{}
      );
    }
  };

  // Callbacks factory
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    return Callbacks<
    decltype(VisitorImpl2x<ChildOps..., NodeOp>::
      get_callbacks(
        threadblock_tile_offset,
        thread_idx,
        problem_shape
      ))>(
      VisitorImpl2x<ChildOps..., NodeOp>::
      get_callbacks(
        threadblock_tile_offset,
        thread_idx,
        problem_shape
      )
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template<
  class ElementCompute,
  class EdgeTuple,
  class... Ops
>
struct TopologicalVisitor2x : VisitorImpl2x<Ops...> {
  static_assert(is_static_v<EdgeTuple>);
  static_assert(cute::rank(EdgeTuple{}) == sizeof...(Ops));
  static_assert(sizeof...(Ops) > 1);

  using VisitorImpl2x<Ops...>::VisitorImpl2x;

  template<class CallbacksImpl>
  struct Callbacks : CallbacksImpl {
    CUTLASS_DEVICE
    Callbacks(CallbacksImpl&& impl)
      : CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}

    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      constexpr int Rm1 = sizeof...(Ops) - 1;
      auto frg_compute_tuple = cute::repeat<Rm1>(Array<ElementCompute, FragmentSize>{});

      return cute::detail::tapply(EdgeTuple{}, callbacks_tuple, frg_compute_tuple,
        // Visit the first R-1 ops in topological order
        [&] (auto&& edge_seq, auto& callbacks, auto& frg_compute) {
          frg_compute = cute::detail::apply(frg_compute_tuple,
          // Compute the current op with children inputs
          [&] (auto const&... frg_inputs) {
            auto frg_output = callbacks.visit(iter_idx, row_idx, column_idx, frg_idx, frg_acc, frg_inputs...);
            using ElementOutput = typename decltype(frg_output)::Element;
            using ConvertOutput = NumericArrayConverter<ElementCompute, ElementOutput, FragmentSize>;
            ConvertOutput convert_output{};

            return convert_output(frg_output);
          },
          // Get inputs in the sequence given by the children indices of the current op
          edge_seq
        );
        return frg_compute;
      },
      // Visit the last op
      [&] (auto const&...ops) {
        return cute::detail::apply(frg_compute_tuple,
          // Compute the last op with children inputs
          [&] (auto const&... frg_inputs) {
            return get<Rm1>(callbacks_tuple).visit(iter_idx, row_idx, column_idx, frg_idx, frg_acc, frg_inputs...);
          },
          // Get inputs in the sequence given by the children indices of the last op
          get<Rm1>(EdgeTuple{})
        );
      },
      // Transform to visit R-1 ops, apply to visit last op
      make_seq<Rm1>{}
      );
    }
  };

  // Callbacks factory
  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    return Callbacks<decltype(
      VisitorImpl2x<Ops...>::
      get_callbacks(
        threadblock_tile_offset,
        thread_idx,
        problem_shape
      ))>(
      VisitorImpl2x<Ops...>::
      get_callbacks(
        threadblock_tile_offset,
        thread_idx,
        problem_shape
      )
    );
  }
};


template <class NodeOp, class... ChildOps>
using Sm80EVT = TreeVisitor2x<NodeOp, ChildOps...>;

template<
  class ElementCompute,
  class EdgeTuple,
  class... Ops
>
using Sm80TopologicalVisitor = TopologicalVisitor2x<ElementCompute, EdgeTuple, Ops...>;


using X = Underscore;

/////////////////////////////////////////////////////////////////////////////////////////////////

// OutputTileThreadLayout translate the CUTLASS 2.X OutputTileOptimalThreadMap into cute layout
// used by CUTLASS 3.X Epilogue
template <
  typename ThreadblockShape_,
  typename WarpShape_,
  typename Element_,
  int ElementsPerAccess,
  int Stages_=1
>
struct OutputTileThreadLayout: DefaultThreadMapTensorOp<
  ThreadblockShape_,
  WarpShape_,
  ThreadblockShape_::kK/WarpShape_::kK,
  Element_,
  ElementsPerAccess>::Type {

  using Base = typename DefaultThreadMapTensorOp<
    ThreadblockShape_,
    WarpShape_,
    ThreadblockShape_::kK/WarpShape_::kK,
    Element_,
    ElementsPerAccess>::Type;
  using Base::Base;

  // Software pipeline stages in epilogue
  static_assert(Stages_ <= 2, "Sm80 EVT only support upto 2 Stages.");
  static const int Stages = Stages_;

  using ThreadShape = cute::Shape<
    cute::Int<Base::Detail::kAccessWidth>,                 // lane col idx
    cute::Int<Base::Detail::kAccessRows>,                  // lane row idx
    cute::Int<Base::Detail::kWarpsRemainingForRows>,       // warp row idx
    cute::Int<Base::Shape::kGroup>,                        // group idx
    cute::Int<Base::Shape::kCluster>                       // cluster idx
  >;

  using Shape = typename Base::Shape;
  using Count = typename Base::Count;

  using ThreadMapShape = cute::Shape<
    // Column
    Int<Base::kElementsPerAccess>,                // vector
    Int<Base::Detail::kAccessWidth>,              // lane_col_coord
    Int<Base::Iterations::kColumn>,               // iteration::column
    // Row
    Int<Base::Detail::kAccessRows>,               // lane_row_coord
    Int<Base::Iterations::kRow>,                  // iterations in row
    Int<Base::Detail::kWarpsRemainingForRows>,    // warp_row_coord
    Int<Count::kRow>,                             // iteration::row
    Int<Count::kGroup>,                           // iteration::group
    Int<Shape::kGroup>,                           // group_coord
    Int<Count::kCluster>,                         // iteration::cluster
    Int<Shape::kCluster>                          // cluster_coord
  >;

  // The shape of CTA Tile
  using CtaShapeMNL = cute::Shape<
    Int<
      Shape::kRow * Count::kRow *
      Shape::kGroup * Count::kGroup *
      Shape::kCluster * Count::kCluster
    >,
    Int<Shape::kColumn * Count::kColumn>,
    _1
  >;

  static const int kElementsPerAccess = ElementsPerAccess;

  //
  // Methods
  //

  CUTLASS_DEVICE
  static auto tid2coord(int thread_idx) {
    return cute::idx2crd(thread_idx, ThreadShape{});
  }

  template <class TensorInput>
  CUTLASS_DEVICE
  static auto partition(TensorInput &&xT, int thread_idx, gemm::GemmCoord threadblock_tile_offset) {

    // (BLK_M,BLK_N)
    Tensor bCxT = local_tile(
      xT, CtaShapeMNL{}, make_coord(_,_,_), Step<_1,_1, X>{}
    )(_,_,threadblock_tile_offset.m(),threadblock_tile_offset.n(),threadblock_tile_offset.k());

    auto [lane_col_coord, lane_row_coord, warp_row_coord, group_coord, cluster_coord] = tid2coord(thread_idx);

    // transform to column-major
    Tensor bCxT_nm = make_tensor(
      std::forward<decltype(bCxT)>(bCxT).data(), make_layout(get<1>(bCxT.layout()), get<0>(bCxT.layout()))
    ).compose(make_layout(ThreadMapShape{}));
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    return bCxT_nm(_,lane_col_coord,_,lane_row_coord,_,warp_row_coord,_,_,group_coord,_,cluster_coord);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
