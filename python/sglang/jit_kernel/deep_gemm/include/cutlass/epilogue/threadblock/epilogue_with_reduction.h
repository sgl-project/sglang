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

  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once
#include "cutlass/cutlass.h"
#include CUDA_STD_HEADER(cassert)


#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/functional.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator with reduction over each column 
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename TensorTileIterator_,             ///< Additional tile iterator for tensor-valued operands
  typename ElementVector_,                  ///< Pointer to reduction vector
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_,                       ///< Output operator
  typename ReductionOp_,                    ///< Reduction operator
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  int IterationsUnroll =                    ///< Used to reduce binary size when epilogue op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)
>
class EpilogueWithReduction : 
  public EpilogueBase<
    Shape_, 
    typename WarpMmaOperator_::Shape, 
    PartitionsK, 
    AccumulatorFragmentIterator_, 
    WarpTileIterator_, 
    Padding_> {

public:

  using Base = EpilogueBase<
    Shape_, 
    typename WarpMmaOperator_::Shape, 
    PartitionsK, 
    AccumulatorFragmentIterator_, 
    WarpTileIterator_, 
    Padding_>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using TensorTileIterator = TensorTileIterator_;
  using ElementVector = ElementVector_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using ReductionOp = ReductionOp_;
  using Padding = Padding_;

  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  static bool const kIsSingleSource = true;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename Base::AccumulatorTile;

  /// Accumulator element
  using ElementAccumulator = typename WarpTileIterator::Element;

  /// Compute data type produced by the output op
  using ElementCompute = typename OutputOp::ElementCompute;

  /// Compute fragment
  using FragmentCompute = Array<ElementCompute, OutputTileIterator::Fragment::kElements>;

  /// Thread map used by output tile iterators
  using ThreadMap = typename OutputTileIterator::ThreadMap;

  /// Fragment object used in reduction
  using ReductionFragment = Array<
    ElementAccumulator, 
    ThreadMap::Iterations::kColumn * ThreadMap::kElementsPerAccess>;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Data type of additional tensor
  using ElementTensor = typename TensorTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef = typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Array type used to output
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Array type used by output functor
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>; 

  /// Array type used by output functor
  using ComputeAccessType = Array<ElementCompute, OutputTileIterator::kElementsPerAccess>;

  /// Tensor access type
  using TensorAccessType = Array<ElementTensor, OutputTileIterator::kElementsPerAccess>;
  
  /// Number of warps
  using WarpCount = typename Base::WarpCount;

  /// Shared memory allocation from epilogue base class
  using BaseSharedStorage = typename Base::SharedStorage;

  /// Used for the reduction
  struct ReductionDetail {

    /// If true, accumulator coordinates are computed and out-of-bounds checks are enabled when
    /// performing the reduction.
    static bool const kOobCheck = false;

    /// Number of threads per warp
    static int const kWarpSize = 32;

    /// Number of distinct scalar column indices handled by each thread
    static int const kColumnsPerThread = ThreadMap::Iterations::kColumn * ThreadMap::kElementsPerAccess;

    /// Number of distinct scalar row indices handled by each thread
    static int const kRowsPerThread = ThreadMap::Iterations::kCount / ThreadMap::Iterations::kColumn;

    /// Number of threads per threadblock
    static int const kThreadCount = kWarpSize * WarpCount::kCount;

    /// Number of distinct threads per row of output tile
    static int const kThreadsPerRow = (Shape::kN / kColumnsPerThread);

    /// Number of distinct threads which must be reduced during the final reduction phase within the threadblock.
    static int const kThreadRows = kThreadCount / kThreadsPerRow;

    /// I'm not sure what I meant here.
    static int const kThreadAccessesPerRow = const_max(1, (Shape::kN + kThreadCount - 1) / kThreadCount);

    /// Shape of the shared memory allocation for the epilogue    
    using StorageShape = MatrixShape<
      kThreadRows,
      Shape::kN
    >;

    /// Debug printing
    CUTLASS_DEVICE
    static void print() {
#if 0
      printf("ReductionDetail {\n");
      printf(
        "  kElementsPerAccess:%d\nkColumnsPerThread: %d\nkRowsPerThread: %d\n,kThreadCount: %d\nkThreadsPerRow: %d\n"
        "kThreadRows: %d\nThreadAccessesPerRow: %d\nStorageShape: %d x %d (count: %d)\n",
        kElementsPerAccess,
        kColumnsPerThread,
        kRowsPerThread,
        kThreadCount,
        kThreadsPerRow,
        kThreadRows,
        kThreadAccessesPerRow,
        StorageShape::kRow,
        StorageShape::kColumn,
        StorageShape::kCount
      );
      printf("};\n");
#endif
    }
  };

  /// Shared storage structure (shadows base) with additional SMEM buffer for reduction
  struct SharedStorage {
    union {
      BaseSharedStorage base;
      AlignedArray<ElementAccumulator, ReductionDetail::StorageShape::kCount, 16> reduction;    ///< Shared storage for reduction
    };

    CUTLASS_HOST_DEVICE
    SharedStorage() { }
  };

public:


  static_assert(SharedLoadIterator::Fragment::kElements == OutputTileIterator::Fragment::kElements,
    "Mismatch between shared load iterator and output tile iterator.");

  static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess), 
    "Divisibility");

private:

  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;

  /// Shared memory pointer fo rreduction
  ElementAccumulator *reduction_ptr_;

  /// Thread index within the threadblock
  int thread_idx_;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueWithReduction(
    SharedStorage &shared_storage,                    ///< Shared storage object    
    int thread_idx,                                   ///< ID of a thread within the threadblock
    int warp_idx,                                     ///< ID of warp within threadblock
    int lane_idx                                      ///< Id of thread within warp
  ):
    Base(shared_storage.base, thread_idx, warp_idx, lane_idx),
    shared_load_iterator_(shared_storage.base.reference(), thread_idx),
    reduction_ptr_(shared_storage.reduction.data()),
    thread_idx_(thread_idx)
  {

  }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                        ///< Output operator
    ElementVector * reduction_output_ptr,             ///< Reduction output vector
    OutputTileIterator destination_iterator,          ///< Tile iterator for destination
    AccumulatorTile const &accumulators,              ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator,               ///< Tile iterator for source accumulator matrix
    TensorTileIterator tensor_iterator,               ///< Threadblock tile iterator for additional tensor operand
    MatrixCoord const &problem_size =                 ///< Problem size needed to guard against out-of-bounds accesses
        MatrixCoord(Shape::kM, Shape::kN),
    MatrixCoord const &threadblock_offset =           ///< Threadblock's initial offset within the problem size space
        MatrixCoord()) {
    
    ReductionFragment reduction_fragment;
    reduction_fragment.clear();

    if (!output_op.is_source_needed()) {
      compute_source_not_needed_(
        output_op, 
        reduction_fragment, 
        destination_iterator, 
        accumulators,
        tensor_iterator,
        problem_size,
        threadblock_offset);
    }
    else {
      compute_source_needed_(
        output_op, 
        reduction_fragment, 
        destination_iterator, 
        accumulators, 
        source_iterator,
        tensor_iterator,
        problem_size,
        threadblock_offset);
    }

    if (output_op.participates_in_reduction()) {
      reduction_(problem_size, threadblock_offset, reduction_output_ptr, reduction_fragment);
    }
  }

private:

  /// Perform the reduction
  CUTLASS_DEVICE
  void reduction_(
    MatrixCoord const &problem_size,                  ///< Problem size needed to guard against out-of-bounds accesses
    MatrixCoord const &threadblock_offset,            ///< Problem size needed to guard against out-of-bounds accesses
    ElementVector * reduction_output_ptr,          ///< Reduction output vector
    ReductionFragment const & reduction_fragment) {

    //
    // Store the partially reduced value to SMEM
    //

    // Guard against uses of the existing SMEM tile
    __syncthreads();
    
    using AccessType = AlignedArray<ElementAccumulator, ThreadMap::kElementsPerAccess>;

    //
    // Determine a compacted thread arrangement to store to SMEM.
    //
    int const kThreadsPerRow = Shape::kN / (ThreadMap::Iterations::kColumn * ThreadMap::kElementsPerAccess);

    MatrixCoord thread_offset(
      thread_idx_ / kThreadsPerRow, 
      (thread_idx_ % kThreadsPerRow) * ThreadMap::kElementsPerAccess);
   
    //
    // Each thread store its fragment to a SMEM
    //

    AccessType *aligned_reduction_ptr = reinterpret_cast<AccessType *>(
      &reduction_ptr_[thread_offset.row() * Shape::kN + thread_offset.column()]);

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&reduction_fragment);
    
    CUTLASS_PRAGMA_UNROLL
    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
      int col_idx = column * ThreadMap::Delta::kColumn / ThreadMap::kElementsPerAccess;

      aligned_reduction_ptr[col_idx] = frag_ptr[column];
    }

    __syncthreads();

    //
    // Now, threads are assigned several columns of the output. They fetch over all rows from
    // the compacted SMEM tile and perform a reduction.
    //

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < ReductionDetail::kThreadAccessesPerRow; ++j) {
      int column_idx = thread_idx_ + j * ReductionDetail::kThreadCount;

      ReductionOp reduction_op;
      ElementAccumulator reduction_element = ElementAccumulator();

      int output_column_idx = threadblock_offset.column() + column_idx;

      if (column_idx < Shape::kN && output_column_idx < problem_size.column()) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ReductionDetail::kThreadRows; ++row) {
          if (row) {
            auto frag = reduction_ptr_[row * Shape::kN + column_idx];

            reduction_element = reduction_op(reduction_element, frag);
          }
          else {

            reduction_element = reduction_ptr_[column_idx];
          }
        }

        // Store
        reduction_output_ptr[column_idx] = ElementVector(reduction_element);
      }
    }
  }

  template<class Seq>
  struct acc2smem;

  template <size_t... Seq>
  struct acc2smem<cutlass::index_sequence<Seq...>> {
    template<int Advance>
    CUTLASS_DEVICE
    static void helper(AccumulatorFragmentIterator accum_fragment_iterator,
                       WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      typename AccumulatorFragmentIterator::Fragment accum_fragment;
      accum_fragment_iterator.load(accum_fragment);
      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void push(size_t pos,
                     AccumulatorFragmentIterator const &iterator_begin,
                     WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {(pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
    }
  };

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_not_needed_(
    OutputOp const &output_op,                        ///< Output operator
    ReductionFragment &reduction_fragment,            ///< Fragment containing the accumulated partial reduction over columns
    OutputTileIterator destination_iterator,          ///< Tile iterator for destination
    AccumulatorTile const &accumulators,              ///< Complete warp-level accumulator tile 
    TensorTileIterator tensor_iterator,               ///< Threadblock tile iterator for additioanl tensor operand
    MatrixCoord const &problem_size,                  ///< Problem size needed to guard against out-of-bounds accesses
    MatrixCoord const &threadblock_offset             ///< Threadblock's initial offset within the problem size space
    ) { 

    //
    // Iterator over warp-level accumulator fragment
    //

    typename TensorTileIterator::Fragment tensor_fragment;
    tensor_fragment.clear();

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    // 

    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

      //
      // Convert and store fragment
      //

      tensor_iterator.load(tensor_fragment);
      ++tensor_iterator;
      
      __syncthreads();

      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
          iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];

      shared_load_iterator_.load(aligned_accum_fragment[0]);

      //
      // If the number of k-slices is > 1 - perform a reduction amongst the k-slices
      //
      if (kPartitionsK > 1)
      {
        plus <typename SharedLoadIterator::Fragment> add_fragments;
        const int tile_row_offset = Base::SharedStorage::StorageShape::kRow / PartitionsK;

        CUTLASS_PRAGMA_UNROLL
        for ( int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_tile_offset({tile_row_offset , 0});
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
        }

        shared_load_iterator_.add_tile_offset({-1 * (kPartitionsK-1) * tile_row_offset, 0});
      }

      //
      // Compute the output result
      //
     
      FragmentCompute compute_fragment;

      apply_output_operator_source_not_needed_(
        reduction_fragment,
        compute_fragment, 
        output_op, 
        aligned_accum_fragment[0],
        tensor_fragment,
        destination_iterator);

      //
      // Store the final result
      //
      
      NumericArrayConverter<ElementOutput, ElementCompute, FragmentCompute::kElements> converter;

      typename OutputTileIterator::Fragment output_fragment = converter(compute_fragment);

      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
  }

  
  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_needed_(
    OutputOp const &output_op,                    ///< Output operator
    ReductionFragment &reduction_fragment,        ///< Fragment containing the accumulated partial reduction over columns
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators,          ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator,           ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
    TensorTileIterator tensor_iterator,            ///< Threadblock tile iterator for additioanl tensor operand
    MatrixCoord const &problem_size,                  ///< Problem size needed to guard against out-of-bounds accesses
    MatrixCoord const &threadblock_offset             ///< Threadblock's initial offset within the problem size space
    ) { 
    
    typename OutputTileIterator::Fragment source_fragment;
    source_fragment.clear();

    typename TensorTileIterator::Fragment tensor_fragment;
    tensor_fragment.clear();

    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    // 

    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

      //
      // Load the source
      //

      source_fragment.clear();
      source_iterator.load(source_fragment);
      ++source_iterator;

      tensor_iterator.load(tensor_fragment);
      ++tensor_iterator;

      //
      // Convert and store fragment
      //
      
      __syncthreads();

      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
          iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];

      shared_load_iterator_.load(aligned_accum_fragment[0]);

      // If the number of k-slices is > 1 - perform a reduction amongst the k-slices
      if (kPartitionsK > 1)
      {
        plus <typename SharedLoadIterator::Fragment> add_fragments;
        const int tile_row_offset = Base::SharedStorage::StorageShape::kRow / PartitionsK;

        CUTLASS_PRAGMA_UNROLL
        for ( int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_tile_offset({tile_row_offset , 0});
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
        }

        shared_load_iterator_.add_tile_offset({-1 * (kPartitionsK-1) * tile_row_offset, 0});
      }

      //
      // Compute the output result
      //
     
      FragmentCompute compute_fragment;

      apply_output_operator_(
        reduction_fragment, 
        compute_fragment, 
        output_op, 
        aligned_accum_fragment[0], 
        source_fragment,
        tensor_fragment,
        destination_iterator);

      //
      // Convert and store the final result
      //

      NumericArrayConverter<ElementOutput, ElementCompute, FragmentCompute::kElements> converter;

      typename OutputTileIterator::Fragment output_fragment = converter(compute_fragment);

      destination_iterator.store(output_fragment);      
      ++destination_iterator;
    }
  }

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_(
    ReductionFragment &reduction_fragment,
    FragmentCompute &compute_fragment,
    OutputOp const &output_op,                    ///< Output operator
    typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
    typename OutputTileIterator::Fragment const &source_fragment,
    typename TensorTileIterator::Fragment const &tensor_fragment,
    OutputTileIterator const & destination_iterator) {
      
    ComputeAccessType *compute_frag_ptr = 
      reinterpret_cast<ComputeAccessType *>(&compute_fragment);

    AccumulatorAccessType const *accum_frag_ptr = 
      reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

    OutputAccessType const *source_frag_ptr = 
      reinterpret_cast<OutputAccessType const *>(&source_fragment);

    TensorAccessType const *tensor_frag_ptr =
      reinterpret_cast<TensorAccessType const *>(&tensor_fragment);

    int const kOutputOpIterations = 
      OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {

      // Call the output operator
      compute_frag_ptr[i] = output_op(accum_frag_ptr[i], source_frag_ptr[i], tensor_frag_ptr[i]);
    }

    //
    // Partial reduction over each column
    //

    ReductionOp reduction_op;

    typename OutputTileIterator::Mask mask;
    destination_iterator.get_mask(mask);

    CUTLASS_PRAGMA_UNROLL
    for (int column = 0; column < ReductionDetail::kColumnsPerThread; ++column) {

      int column_vector_idx = column / ThreadMap::kElementsPerAccess;
      bool column_guard = mask.predicates[column_vector_idx];

      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < ReductionDetail::kRowsPerThread; ++row) {

        bool fetch;
        if (ReductionDetail::kOobCheck) {
          int row_idx = (row % ThreadMap::Iterations::kRow);
          int residual = (row / ThreadMap::Iterations::kRow);

          int group_idx = (residual % ThreadMap::Iterations::kGroup);
          residual = (residual / ThreadMap::Iterations::kGroup);

          int cluster_idx = (residual % ThreadMap::Iterations::kCluster);

          int row_offset = row_idx * ThreadMap::Delta::kRow 
            + group_idx * ThreadMap::Delta::kGroup 
            + cluster_idx * ThreadMap::Delta::kCluster;

          int output_row = destination_iterator.thread_start_row() + row_offset;

          fetch = (output_row < destination_iterator.extent_row() && column_guard);
        }
        else {
          fetch = true;
        }

        ElementCompute value = ElementCompute();
        if (fetch) {
          value = compute_fragment[row * ReductionDetail::kColumnsPerThread + column];
        }

        reduction_fragment[column] = reduction_op(
          reduction_fragment[column], 
          value);
      }
    }
  }

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_source_not_needed_(
    ReductionFragment &reduction_fragment,
    FragmentCompute &compute_fragment,
    OutputOp const &output_op,                    ///< Output operator
    typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
    typename TensorTileIterator::Fragment const &tensor_fragment,
    OutputTileIterator const & destination_iterator
  ) {
    
    ComputeAccessType *compute_frag_ptr = 
      reinterpret_cast<ComputeAccessType *>(&compute_fragment);

    AccumulatorAccessType const *accum_frag_ptr = 
      reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

    TensorAccessType const *tensor_frag_ptr =
      reinterpret_cast<TensorAccessType const *>(&tensor_fragment);

    int const kOutputOpIterations = 
      OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {

      // Call the output operator
      compute_frag_ptr[i] = output_op(accum_frag_ptr[i], tensor_frag_ptr[i]);
    }

    //
    // Partial reduction over each column
    //

    ReductionOp reduction_op;

    typename OutputTileIterator::Mask mask;
    destination_iterator.get_mask(mask);

    CUTLASS_PRAGMA_UNROLL
    for (int column = 0; column < ReductionDetail::kColumnsPerThread; ++column) {

      int column_vector_idx = column / ThreadMap::kElementsPerAccess;
      bool column_guard = mask.predicates[column_vector_idx];

      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < ReductionDetail::kRowsPerThread; ++row) {

        bool fetch;
        if (ReductionDetail::kOobCheck) {
          int row_idx = (row % ThreadMap::Iterations::kRow);
          int residual = (row / ThreadMap::Iterations::kRow);

          int group_idx = (residual % ThreadMap::Iterations::kGroup);
          residual = (residual / ThreadMap::Iterations::kGroup);

          int cluster_idx = (residual % ThreadMap::Iterations::kCluster);

          int row_offset = row_idx * ThreadMap::Delta::kRow 
            + group_idx * ThreadMap::Delta::kGroup 
            + cluster_idx * ThreadMap::Delta::kCluster;

          int output_row = destination_iterator.thread_start_row() + row_offset;

          fetch = (output_row < destination_iterator.extent_row() && column_guard);
        }
        else {
          fetch = true;
        }

        ElementCompute value = ElementCompute();
        if (fetch) {
          value = compute_fragment[row * ReductionDetail::kColumnsPerThread + column];
        }

        reduction_fragment[column] = reduction_op(
          reduction_fragment[column], 
          value);
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
