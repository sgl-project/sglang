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
  \brief Visitor tree load operations for the CUTLASS 2x epilogue
*/

#pragma once

#include "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp"
#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;

using X = Underscore;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Fetch Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// returns accumulator
struct VisitorAccFetch : VisitorImpl2x<> {

  using VisitorImpl2x<>::VisitorImpl2x;

  struct Callbacks : EmptyCallbacks {
    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementAccumulator, FragmentSize>
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx, Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      return frg_acc;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    return Callbacks{};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Broadcast Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////
// Scalar broadcast
template<
  class Element,
  class StrideMNL = Stride<_0,_0,_0>,
  int BroadcastCount = 1,
  template <class> class ReductionFn = multiplies
>
struct VisitorScalarBroadcast {
  static_assert(
    (cute::is_same_v<StrideMNL, Stride<_0,_0,_0>>) || // scalar broadcast, e.g. alpha
    (cute::is_same_v<StrideMNL, Stride<_0,_0,_1>>) ||
    (cute::is_same_v<StrideMNL, Stride<_0,_0,int>>));  // batched scalar broadcast, e.g. per-batch alpha

  struct SharedStorage { };

  struct Arguments {
    Element scalars[BroadcastCount] = {};
    Element const* scalar_ptrs[BroadcastCount] = {};
    StrideMNL dScalar = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  CUTLASS_HOST_DEVICE
  VisitorScalarBroadcast() { }

  CUTLASS_HOST_DEVICE
  VisitorScalarBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params) {
    // Get the scalar for non-batched broadcast
    if constexpr (cute::is_same_v<StrideMNL, Stride<_0,_0,_0>>) {
      update_scalar();
    }
  }

  Element scalar;
  Params const* params_ptr;

  struct Callbacks: EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(Element scalar)
      : scalar(scalar) {}

    Element scalar;

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Array<Element, FragmentSize> frg_scalar;
      frg_scalar.fill(scalar);

      return frg_scalar;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    // Get the scalar for batched broadcast
    if constexpr (
      cute::is_same_v<StrideMNL, Stride<_0,_0,_1>> ||
      cute::is_same_v<StrideMNL, Stride<_0,_0,int>>) {
      update_scalar(threadblock_tile_offset.k());
    }
    return Callbacks(scalar);
  }

private:
  CUTLASS_DEVICE void
  update_scalar(int l_coord = 0) {
    int l_offset = l_coord * size<2>(params_ptr->dScalar);

    if (params_ptr->scalar_ptrs[0] != nullptr) {
      scalar = params_ptr->scalar_ptrs[0][l_offset];
    } else {
      // batch stride is ignored for nullptr fallback
      scalar = params_ptr->scalars[0];
    }

    // Do reduction over multiple broadcasts if necessary
    ReductionFn<Element> reduction_fn;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < BroadcastCount; ++i) {
      if (params_ptr->scalar_ptrs[i] != nullptr) {
        scalar = reduction_fn(scalar, params_ptr->scalar_ptrs[i][l_offset]);
      } else {
        // batch stride is ignored for nullptr fallback
        scalar = reduction_fn(scalar, params_ptr->scalars[i]);
      }
    }
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ThreadMap,
  class Element,
  class StrideMNL
>
struct VisitorAuxLoad{

  struct Arguments {
    Element* ptr_aux = nullptr;
    Element null_default = Element(0);
    StrideMNL dAux = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  // Software pipeline stages
  static const int Stages = ThreadMap::Stages;

  struct SharedStorage {};

  // Global load type
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorAuxLoad() { }

  CUTLASS_HOST_DEVICE
  VisitorAuxLoad(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gAux,
      RTensor&& tC_rAux,
      CTensor&& tC_cAux,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gAux(cute::forward<GTensor>(tC_gAux)),
      tC_rAux(cute::forward<RTensor>(tC_rAux)),
      tC_cAux(cute::forward<CTensor>(tC_cAux)),
      problem_shape(problem_shape),
      params_ptr(params_ptr) { }

    GTensor tC_gAux;
    RTensor tC_rAux;
    CTensor tC_cAux;
    Params const* params_ptr;
    ProblemShape problem_shape;

    CUTLASS_DEVICE void
    begin_step(int step_idx) {
      clear(tC_rAux(_,_,_,step_idx%Stages));
      auto src_v = filter(tC_gAux(_,_,_,step_idx));
      auto coord_v = filter(tC_cAux(_,_,_,step_idx));
      auto dst_v = filter(tC_rAux(_,_,_,step_idx%Stages));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = elem_less(coord_v(i), problem_shape);
        cutlass::arch::global_load<VecType, sizeof(VecType)>(dst_v(i), (void const*)&src_v(i), guard);
      }
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux(_,_,_,iter_idx%Stages)));
      return tC_rAux_frg(frg_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mAux = make_tensor(
      make_gmem_ptr(params_ptr->ptr_aux),
      problem_shape,
      params_ptr->dAux);   // (M,N,L)
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    Tensor tC_gAux = recast<VecType>(
      group_modes<3,6>(ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset)));
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, Stages
    Tensor tC_rAux = make_tensor<VecType>(
      make_layout(flatten(make_shape(take<0,3>(tC_gAux.shape()), Int<Stages>{}))));

    // Generate the pred tensor
    Tensor cAux = make_identity_tensor(mAux.shape());
    Tensor tC_cAux = outer_partition(
      group_modes<3,6>(ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset)),
      Shape<Int<VecLength>>{},
      (_0{})
    );

    return Callbacks<
      decltype(tC_gAux), decltype(tC_rAux),
      decltype(tC_cAux), ProblemShape>(
      cute::move(tC_gAux),
      cute::move(tC_rAux),
      cute::move(tC_cAux),
      problem_shape,
      params_ptr
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Row vector broadcast
template<
  class ThreadMap,
  class Element,
  class StrideMNL,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct VisitorRowBroadcast {

  struct Arguments {
    Element const* ptr_row = nullptr;
    Element null_default = Element(0);
    StrideMNL dRow = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  struct SharedStorage {};

  // Global load type
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorRowBroadcast() { }

  CUTLASS_HOST_DEVICE
  VisitorRowBroadcast(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gRow,
      RTensor&& tC_rRow,
      CTensor&& tC_cRow,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gRow(cute::forward<GTensor>(tC_gRow)),
      tC_rRow(cute::forward<RTensor>(tC_rRow)),
      tC_cRow(cute::forward<CTensor>(tC_cRow)),
      n(get<1>(problem_shape)),
      params_ptr(params_ptr) { }

    GTensor tC_gRow;
    RTensor tC_rRow;
    CTensor tC_cRow;
    Params const* params_ptr;
    int n;

    CUTLASS_DEVICE void
    begin_epilogue() {
      if constexpr (EnableNullptr) {
        if (params_ptr->ptr_row == nullptr) {
          auto tC_rRow_vec = recast<Array<Element, VecLength>>(coalesce(tC_rRow));
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tC_rRow_vec); ++i) {
            tC_rRow_vec[i].fill(params_ptr->null_default);
          }
          return;
        }
      }
      clear(tC_rRow);
      auto src_v = filter(tC_gRow);
      auto coord_v = filter(tC_cRow);
      auto dst_v = filter(tC_rRow);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = get<1>(coord_v(i)) < n;
        cutlass::arch::global_load<VecType, sizeof(VecType)>(dst_v(i), (void const *)&src_v(i), guard);
      }
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Tensor rRow_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rRow));
      return rRow_frg(column_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mRow = make_tensor(
      make_gmem_ptr(params_ptr->ptr_row),
      problem_shape,
      params_ptr->dRow);

    // VECTOR, FRAGMENT_COLUMN
    Tensor tC_gRow = recast<VecType>(
      ThreadMap::partition(mRow, thread_idx, threadblock_tile_offset)
    )(_,_,_0{},_0{},_0{},_0{});
    Tensor tC_rRow = make_tensor_like(tC_gRow);

    // Generate the pred tensor
    Tensor cRow = make_identity_tensor(mRow.shape());
    Tensor tC_cRow = outer_partition(
      ThreadMap::partition(cRow, thread_idx, threadblock_tile_offset)(_,_,_0{},_0{},_0{},_0{}),
      Shape<Int<VecLength>>{},
      (_0{})
    );

    return Callbacks<
      decltype(tC_gRow), decltype(tC_rRow),
      decltype(tC_cRow), ProblemShape>(
      cute::move(tC_gRow),
      cute::move(tC_rRow),
      cute::move(tC_cRow),
      problem_shape,
      params_ptr
    );
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Column vector broadcast
template<
  class ThreadMap,
  class Element,
  class StrideMNL = Stride<_1,_0,_0>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct VisitorColBroadcast {

  struct Arguments {
    Element const* ptr_col = nullptr;
    Element null_default = Element(0);
    StrideMNL dCol = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  struct SharedStorage { };

  CUTLASS_HOST_DEVICE
  VisitorColBroadcast() { }

  CUTLASS_HOST_DEVICE
  VisitorColBroadcast(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gCol,
      RTensor&& tC_rCol,
      CTensor&& tC_cCol,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gCol(cute::forward<GTensor>(tC_gCol)),
      tC_rCol(cute::forward<RTensor>(tC_rCol)),
      tC_cCol(cute::forward<CTensor>(tC_cCol)),
      m(get<0>(problem_shape)),
      params_ptr(params_ptr) { }

    GTensor tC_gCol;
    RTensor tC_rCol;
    CTensor tC_cCol;
    Params const* params_ptr;
    int m;

    CUTLASS_DEVICE void
    begin_epilogue() {
      if constexpr (EnableNullptr) {
        if (params_ptr->ptr_col == nullptr) {
          fill(tC_rCol, params_ptr->null_default);
          return;
        }
      }
      clear(tC_rCol);
      Tensor tC_pCol = cute::lazy::transform(tC_cCol, [&] (auto const& c) { return get<0>(c) < m; });
      copy_if(tC_pCol, tC_gCol, tC_rCol);
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Array<Element, FragmentSize> frg_col;
      frg_col.fill(tC_rCol(row_idx,iter_idx));
      return frg_col;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mCol = make_tensor(
      make_gmem_ptr(params_ptr->ptr_col),
      problem_shape,
      params_ptr->dCol);

    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    Tensor tC_gCol = group_modes<1,4>(
      ThreadMap::partition(mCol, thread_idx, threadblock_tile_offset)(_0{},_0{},_,_,_,_));
    Tensor tC_rCol = make_tensor_like(tC_gCol);

    // Generate the pred tensor
    Tensor cCol = make_identity_tensor(mCol.shape());
    Tensor tC_cCol = group_modes<1,4>(
      ThreadMap::partition(cCol, thread_idx, threadblock_tile_offset)(_0{},_0{},_,_,_,_));

    return Callbacks<
      decltype(tC_gCol), decltype(tC_rCol),
      decltype(tC_cCol), ProblemShape>(
      cute::move(tC_gCol),
      cute::move(tC_rCol),
      cute::move(tC_cCol),
      problem_shape,
      params_ptr
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
