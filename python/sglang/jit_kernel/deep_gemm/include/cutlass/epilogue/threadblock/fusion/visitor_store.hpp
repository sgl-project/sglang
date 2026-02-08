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
  \brief Visitor tree store operations for the CUTLASS 2x epilogue
*/

#pragma once

#include "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;
using X = Underscore;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Store Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ThreadMap,
  class Element,
  FloatRoundStyle RoundStyle,
  class StrideMNL
>
struct VisitorAuxStore{

  struct Arguments {
    Element* ptr_aux = nullptr;
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

  struct SharedStorage {};

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorAuxStore() { }

  CUTLASS_HOST_DEVICE
  VisitorAuxStore(Params const& params, SharedStorage const& shared_storage)
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
      clear(tC_rAux);
    }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc,
          Array<ElementInput, FragmentSize> const& frg_input) {
      using ConvertInput = NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux));
      tC_rAux_frg(frg_idx) = convert_input(frg_input);

      return frg_input;
    }

    CUTLASS_DEVICE void
    end_step(int step_idx) {
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_,_,_,step_idx));
      auto dst_v = filter(tC_gAux(_,_,_,step_idx));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = elem_less(coord_v(i), problem_shape);
        cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i), (void*)&dst_v(i), guard);
      }
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
    Tensor tC_gAux = recast<VecType>(group_modes<3,6>(ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset)));
    Tensor tC_rAux = make_tensor_like(take<0,3>(tC_gAux));

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
//
// Reduction Store Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions
template <
  template <class> class ReduceFn,
  int kThreads, class T>
CUTLASS_DEVICE
void intra_warp_row_reduce(T& value) {
  using ReduceInput = ReduceFn<T>;
  ReduceInput reduce_input{};
  constexpr int kHalfThreads = kThreads >> 1;
  CUTLASS_PRAGMA_UNROLL
  for (int i = kHalfThreads; i > 0; i >>= 1) {
    value = reduce_input(value, __shfl_xor_sync(0xFFFFFFFF, value, i));
  }
}

template <
  template <class> class ReduceFn,
  FloatRoundStyle RoundStyle,
  class ElementCompute,
  class ElementFragment, int FragmentSize>
CUTLASS_DEVICE
void fragment_reduce(ElementCompute& value, Array<ElementFragment, FragmentSize> const& frg) {
  using ReduceInput = ReduceFn<ElementCompute>;
  ReduceInput reduce_input{};
  using ConvertInput = NumericConverter<ElementCompute, ElementFragment, RoundStyle>;
  ConvertInput convert_input{};

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < FragmentSize; ++i) {
    value = reduce_input(value, convert_input(frg[i]));
  }
}

template<
  template <class> class AtomicReduceFn,
  FloatRoundStyle RoundStyle,
  class ElementCompute,
  class ElementOutput>
CUTLASS_DEVICE
void atomic_reduce(ElementOutput* ptr, ElementCompute const& value) {
  using ReduceOutput = AtomicReduceFn<ElementOutput>;
  using ConvertOutput = NumericConverter<ElementOutput, ElementCompute, RoundStyle>;
  ReduceOutput reduce_output{};
  ConvertOutput convert_output{};

  reduce_output(ptr, convert_output(value));
}

// Col vector reduction
template <
  template <class> class RegReduceFn,
  template <class> class AtomicReduceFn,
  class ThreadMap,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_1,_0,_0>
>
struct VisitorColReduction {

  struct Arguments {
    ElementOutput* ptr_col = nullptr;
    ElementCompute reduction_identity = 0;
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
  VisitorColReduction() { }

  CUTLASS_HOST_DEVICE
  VisitorColReduction(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gCol,
      CTensor&& tC_cCol,
      ProblemShape problem_shape,
      Params const* params_ptr,
      int thread_idx
    ):
      tC_gCol(cute::forward<GTensor>(tC_gCol)),
      tC_cCol(cute::forward<CTensor>(tC_cCol)),
      m(get<0>(problem_shape)),
      n(get<1>(problem_shape)),
      params_ptr(params_ptr) {
        // The partial reduction results of each warp are further
        // reduced to the first thread in each row.
        // Only the first thread in each row is the writing thread
        is_writing_thread = thread_idx % ThreadMap::Detail::kAccessWidth == 0;
      }

    GTensor tC_gCol;
    CTensor tC_cCol;
    Params const* params_ptr;
    int m;
    int n;
    int curr_iter_idx;
    bool is_writing_thread;

    ElementCompute reduction_accum;

    CUTLASS_DEVICE void
    begin_row(int row_idx) {
      reduction_accum = ElementCompute(params_ptr->reduction_identity);
    }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc,
          Array<ElementInput, FragmentSize> const& frg_input) {

      curr_iter_idx = iter_idx;

      int coord_n = get<1>(tC_cCol(column_idx, row_idx, iter_idx));
      if (coord_n < n) {
        fragment_reduce<RegReduceFn, RoundStyle>(reduction_accum, frg_input);
      }

      // Intra-warp reduction
      if (column_idx + 1 == ThreadMap::Iterations::kColumn) {
        intra_warp_row_reduce<RegReduceFn, ThreadMap::Detail::kAccessWidth>(reduction_accum);
      }

      return frg_input;
    }

    CUTLASS_DEVICE auto
    end_row(int row_idx) {
      bool guard = get<0>(tC_cCol(_0{}, row_idx,curr_iter_idx)) < m;

      if (guard && is_writing_thread) {
        atomic_reduce<AtomicReduceFn, RoundStyle>(&tC_gCol(row_idx,curr_iter_idx), reduction_accum);
      }
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
    // FRAGMENT_ROW, (ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER)
    Tensor tC_gCol = group_modes<1,4>(
      ThreadMap::partition(mCol, thread_idx, threadblock_tile_offset)(_0{},_0{},_,_,_,_));

    // Generate the pred tensor
    Tensor cCol = make_identity_tensor(mCol.shape());
    // FRAGMENT_COL, FRAGMENT_ROW, (ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER)
    Tensor tC_cCol = group_modes<2,5>(
      ThreadMap::partition(cCol, thread_idx, threadblock_tile_offset)(_0{},_,_,_,_,_));

    return Callbacks<
      decltype(tC_gCol), decltype(tC_cCol),
      ProblemShape>(
      cute::move(tC_gCol),
      cute::move(tC_cCol),
      problem_shape,
      params_ptr,
      thread_idx
    );
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
// Row vector reduction
template <
  template <class> class RegReduceFn,
  template <class> class AtomicReduceFn,
  class ThreadMap,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_0,_1,_0>
>
struct VisitorRowReduction {

  struct Arguments {
    ElementOutput* ptr_row = nullptr;
    ElementCompute reduction_identity = 0;
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

  using SharedStorageShape = decltype(select<0,1,2,3,5,8,10>(typename ThreadMap::ThreadMapShape{}));

  struct SharedStorage {
    AlignedArray<ElementCompute, size(SharedStorageShape{}), 16> reduction;
  };

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<ElementOutput>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;

  CUTLASS_HOST_DEVICE
  VisitorRowReduction() { }

  CUTLASS_HOST_DEVICE
  VisitorRowReduction(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params),
      smem_reduce(const_cast<ElementCompute*>(shared_storage.reduction.data())) { }

  Params const* params_ptr;
  ElementCompute* smem_reduce;

  template <
    class RTensorR2S, class STensorR2S, class CTensorR2S,
    class STensorS2R, class RTensorS2R, class CTensorS2R,
    class GTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      // R->S
      RTensorR2S&& tRS_rSrc,
      STensorR2S&& tRS_sRows,
      CTensorR2S&& tRS_cSrc,
      // S->R
      STensorS2R&& tSR_sRows,
      RTensorS2R&& tSR_rRows,
      CTensorS2R&& tSR_cRows,
      // R->G
      GTensor&& tC_gRow,
      CTensor&& tC_cRow,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      // R->S
      tRS_rSrc(cute::forward<RTensorR2S>(tRS_rSrc)),
      tRS_sRows(cute::forward<STensorR2S>(tRS_sRows)),
      tRS_cSrc(cute::forward<CTensorR2S>(tRS_cSrc)),
      // S->R
      tSR_sRows(cute::forward<STensorS2R>(tSR_sRows)),
      tSR_rRows(cute::forward<RTensorS2R>(tSR_rRows)),
      tSR_cRows(cute::forward<CTensorS2R>(tSR_cRows)),
      // R->G
      tC_gRow(cute::forward<GTensor>(tC_gRow)),
      tC_cRow(cute::forward<CTensor>(tC_cRow)),
      m(get<0>(problem_shape)),
      n(get<1>(problem_shape)),
      params_ptr(params_ptr) { }

    // R->S
    RTensorR2S tRS_rSrc;
    STensorR2S tRS_sRows;
    CTensorR2S tRS_cSrc;
    // S->R
    STensorS2R tSR_sRows;
    RTensorS2R tSR_rRows;
    CTensorS2R tSR_cRows;
    // R->G
    GTensor tC_gRow;
    CTensor tC_cRow;

    Params const* params_ptr;
    int n;
    int m;

    CUTLASS_DEVICE void
    begin_epilogue() {
      fill(tRS_rSrc, params_ptr->reduction_identity);
    }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc,
          Array<ElementInput, FragmentSize> const& frg_input) {

      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};
      Tensor tRS_rRow_frg = recast<Array<ElementCompute, FragmentSize>>(coalesce(tRS_rSrc));

      int coord_m = get<0>(tRS_cSrc(column_idx,row_idx,iter_idx));
      if (coord_m < m)
        reduction(tRS_rRow_frg[column_idx], convert_input(frg_input));

      return frg_input;
    }

    CUTLASS_DEVICE void
    end_epilogue() {
      //
      // Store the partially reduced value to SMEM
      //

      // Guard against uses of the existing SMEM tile
      __syncthreads();

      copy(tRS_rSrc, tRS_sRows);

      __syncthreads();

      //
      // Now, threads are assigned several columns of the output. They fetch over all rows from
      // the compacted SMEM tile and perform a reduction.
      //

      fill(tSR_rRows, params_ptr->reduction_identity);

      using ReduceInputReg = RegReduceFn<ElementCompute>;
      ReduceInputReg reduce_input_reg{};

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tSR_rRows); ++j) {
        if (get<0>(tSR_cRows(j)) < get<1>(typename ThreadMap::CtaShapeMNL{}) && get<1>(tC_cRow(j)) < n) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tSR_sRows) / size(tSR_rRows); ++i) {
            tSR_rRows(j) = reduce_input_reg(tSR_rRows(j), tSR_sRows(i + j * size(tSR_sRows) / size(tSR_rRows)));
          }
          atomic_reduce<AtomicReduceFn, RoundStyle>(&tC_gRow(j), tSR_rRows(j));
        }

      }
    }

  private:

    template <int FragmentSize>
    CUTLASS_DEVICE ElementCompute
    reduction(Array<ElementCompute, FragmentSize>& reduce_buffer, Array<ElementCompute, FragmentSize> const& result) {
      using ReduceInput = RegReduceFn<ElementCompute>;
      ReduceInput reduce_input{};
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
            reduce_buffer[i] = reduce_input(reduce_buffer[i], result[i]);
        }
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

    //
    // Step 1: reduce fragment input (Src) into tRS_rSrc
    //

    // VECTOR,FRAGMENT_COL
    Tensor tRS_rSrc = make_tensor<ElementCompute>(select<0,2>(typename ThreadMap::ThreadMapShape{}));

    Tensor cSrc = make_identity_tensor(mRow.shape());
    // FRAGMENT_COLUMN, FRAGMENT_ROW, (ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER)
    Tensor tRS_cSrc = group_modes<2,5>(ThreadMap::partition(cSrc, thread_idx, threadblock_tile_offset)(_0{},_,_,_,_,_));

    //
    // Step 2: copy the partial results in tRS_rSrc to sRows in shared memory
    //

    // VECTOR,ACCESS_WIDTH,FRAGMENT_COL,ACCESS_ROWS,WARPS_PER_ROW,GROUPS,CLUSTERS
    Tensor sRows = make_tensor(
      make_smem_ptr(smem_reduce), SharedStorageShape{}
    );

    auto [lane_col_coord, lane_row_coord, warp_row_coord, group_coord, cluster_coord] = ThreadMap::tid2coord(thread_idx);
    Tensor tRS_sRows = sRows(_,lane_col_coord,_,lane_row_coord,warp_row_coord,group_coord,cluster_coord);

    //
    // Step 3: copy the partial results in sRows to tSR_sRow for reduction
    //

    // VECTOR*ACCESS_WIDTH*FRAGMENT_COL,ACCESS_ROWS*WARPS_PER_ROW*GROUPS*CLUSTERS
    Tensor sRows_nm = coalesce(group_modes<1,5>(group_modes<0,3>(sRows)), Shape<_1,_1>{});
    // SMEM_ROW/THREADS,ACCESS_ROWS*WARPS_PER_ROW*GROUPS*CLUSTERS
    Tensor tSR_sRows = outer_partition(sRows_nm, Shape<Int<ThreadMap::kThreads>,_1>{}, thread_idx);
    // SMEM_ROW/THREADS
    Tensor tSR_rRows = make_tensor_like(tSR_sRows(_,_0{}));
    // Coord
    Tensor cRows_nm = make_identity_tensor(sRows_nm.shape());
    Tensor tSR_cRows = outer_partition(cRows_nm, Shape<Int<ThreadMap::kThreads>,_1>{}, thread_idx)(_,_0{});

    //
    // Step 4: atomically reduce the results to global memory
    //

    Tensor tC_gRow = outer_partition(
      // Cta tile
      local_tile(
        mRow, typename ThreadMap::CtaShapeMNL{}, make_coord(_,_,_),Step<_1,_1, X>{}
      )(_,_,threadblock_tile_offset.m(),threadblock_tile_offset.n(),threadblock_tile_offset.k()),
      // Partition to threads
      Shape<_1,Int<ThreadMap::kThreads>>{}, thread_idx
    )(_0{},_);

    Tensor cRow = make_identity_tensor(mRow.shape());
    Tensor tC_cRow = outer_partition(
      // Cta tile
      local_tile(
        cRow, typename ThreadMap::CtaShapeMNL{}, make_coord(_,_,_), Step<_1,_1, X>{}
      )(_,_,threadblock_tile_offset.m(),threadblock_tile_offset.n(),threadblock_tile_offset.k()),
      // Partition to threads
      Shape<_1,Int<ThreadMap::kThreads>>{}, thread_idx
    )(_0{},_);

    return Callbacks<
      decltype(tRS_rSrc), decltype(tRS_sRows),
      decltype(tRS_cSrc), decltype(tSR_sRows),
      decltype(tSR_rRows), decltype(tSR_cRows),
      decltype(tC_gRow), decltype(tC_cRow),
      ProblemShape>(
      // R->S
      cute::move(tRS_rSrc),
      cute::move(tRS_sRows),
      cute::move(tRS_cSrc),
      // S->R
      cute::move(tSR_sRows),
      cute::move(tSR_rRows),
      cute::move(tSR_cRows),
      // R->G
      cute::move(tC_gRow),
      cute::move(tC_cRow),
      problem_shape,
      params_ptr
    );
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
// Scalar reduction
template <
  template <class> class RegReduceFn,
  template <class> class AtomicReduceFn,
  class ThreadMap,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class StrideMNL = Stride<_0,_0,_0>
>
struct VisitorScalarReduction {
  static_assert(
    (cute::is_same_v<StrideMNL, Stride<_0,_0, _0>>) || // scalar reduction, e.g. tensor max element
    (cute::is_same_v<StrideMNL, Stride<_0,_0, _1>>) || // batched scalar reduction, e.g. per-batch max element
    (cute::is_same_v<StrideMNL, Stride<_0,_0,int>>));

  struct Arguments {
    ElementOutput* ptr_scalar = nullptr;
    ElementCompute reduction_identity = 0;
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

  struct SharedStorage { };

  CUTLASS_HOST_DEVICE
  VisitorScalarReduction(){ };

  CUTLASS_HOST_DEVICE
  VisitorScalarReduction(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class CTensor, class GTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      CTensor&& tC_cSrc,
      GTensor&& tC_gScalar,
      ProblemShape problem_shape,
      Params const* params_ptr,
      int thread_idx
    ):
      tC_cSrc(cute::forward<CTensor>(tC_cSrc)),
      tC_gScalar(cute::forward<GTensor>(tC_gScalar)),
      problem_shape(problem_shape),
      params_ptr(params_ptr) {
        // The partial reduction results of each warp are further
        // reduced to this first thread.
        // Only the first thread of each warp is the writing thread
        is_writing_thread = thread_idx % ThreadMap::kWarpSize == 0;
      }

      GTensor tC_gScalar;
      CTensor tC_cSrc;
      Params const* params_ptr;
      ProblemShape problem_shape;
      bool is_writing_thread;

      ElementCompute reduction_accum;

      CUTLASS_DEVICE void
      begin_epilogue() {
        reduction_accum = ElementCompute(params_ptr->reduction_identity);
      }

      template <class ElementAccumulator, class ElementInput, int FragmentSize>
      CUTLASS_DEVICE auto
      visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
            Array<ElementAccumulator, FragmentSize> const& frg_acc,
            Array<ElementInput, FragmentSize> const& frg_input) {

        auto coord = tC_cSrc(column_idx, row_idx, iter_idx);
        if (elem_less(coord, problem_shape)) {
          fragment_reduce<RegReduceFn, RoundStyle>(reduction_accum, frg_input);
        }

        return frg_input;
      }

      CUTLASS_DEVICE auto
      end_epilogue() {
        // Intra-warp reduction
        intra_warp_row_reduce<RegReduceFn, ThreadMap::kWarpSize>(reduction_accum);

        // Atomically reduce to global memory
        atomic_reduce<AtomicReduceFn, RoundStyle>(&tC_gScalar(_0{},_0{}), reduction_accum);
      }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor cSrc = make_identity_tensor(problem_shape);
    // FRAGMENT_COL, FRAGMENT_ROW, (ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER)
    Tensor tC_cSrc = group_modes<2,5>(
      ThreadMap::partition(cSrc, thread_idx, threadblock_tile_offset)(_0{},_,_,_,_,_)
    );

    Tensor mScalar = make_tensor(
      make_gmem_ptr(params_ptr->ptr_scalar),
      problem_shape,
      params_ptr->dScalar
    );

    Tensor tC_gScalar = mScalar(_,_,threadblock_tile_offset.k());

    return Callbacks<
      decltype(tC_cSrc), decltype(tC_gScalar),
      ProblemShape>(
      cute::move(tC_cSrc),
      cute::move(tC_gScalar),
      problem_shape,
      params_ptr,
      thread_idx
    );
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
