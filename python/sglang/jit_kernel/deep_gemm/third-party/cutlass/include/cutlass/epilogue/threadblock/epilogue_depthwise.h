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
  \brief Epilogue for Depthwise convoltuion

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/reduction_op.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <typename Shape_,                   ///< Shape of threadblock tile (concept: GemmShape)
          typename ThreadOutputShape_,       /// Size of the matrix to load (concept: TensorNHWC)
          typename ThreadBlockOutputShape_,  /// Size of the matrix to load (concept: TensorNHWC)
          typename WarpMmaOperator_,         ///< Warp-level MMA operator (concept:
                                             ///< gemm::warp::MmaTensorOp)
          typename OutputTileIterator_,      ///< Tile iterator reading and writing output tensors
          typename AccumulatorFragmentIterator_,  ///< Fragment iterator selecting accumulators
          typename WarpTileIterator_,    ///< Warp-scoped tile iterator writing accumulators to SMEM
          typename SharedLoadIterator_,  ///< Threadblock-scoped tile iterator loading from SMEM
          typename OutputOp_,            ///< Output operator
          typename Padding_  ///< Padding added to SMEM allocation to avoid bank conflicts (concept:
                             ///< MatrixShape)
          >
class EpilogueDepthwise {
 public:
  using Shape = Shape_;
  using WarpShape = typename WarpMmaOperator_::Shape;
  using ThreadOutputShape = ThreadOutputShape_;
  using ThreadBlockOutputShape = ThreadBlockOutputShape_;
  using WarpMmaOperator = WarpMmaOperator_;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = Padding_;

  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

  /// Accumulator element
  using ElementAccumulator = typename WarpTileIterator::Element;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef = typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Array type used to output
  using OutputAccessType =
      Array<typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Array type used by output functor
  using AccumulatorAccessType =
      Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Number of warps
  using WarpCount =
      gemm::GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN>;

 public:
  static_assert(SharedLoadIterator::Fragment::kElements ==
  OutputTileIterator::Fragment::kElements,
    "Mismatch between shared load iterator and output tile iterator.");

  static_assert(OutputTileIterator::kElementsPerAccess,
                "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess),
                "Divisibility");

  /// Shared storage allocation needed by the epilogue
  struct SharedStorage {
    //
    // Type definitions
    //

    /// Element type of shared memory
    using Element = typename WarpTileIterator::Element;

    /// Tensor reference to shared memory allocation
    using TensorRef = typename WarpTileIterator::TensorRef;

    /// Layout of shared memory allocation
    using Layout = typename WarpTileIterator::Layout;

    /// Logical shape of the shared memory tile written to by all warps.
    using Shape = MatrixShape<ThreadBlockOutputShape::kNHW, ThreadBlockOutputShape::kC>;

    /// Shape of the shared memory allocation for the epilogue
    using StorageShape = MatrixShape<Shape::kRow, Shape::kColumn>;

    //
    // Data members
    //

    AlignedBuffer<Element, StorageShape::kCount> storage;

    //
    // Methods
    //

    /// Returns a pointer to the shared memory buffer
    CUTLASS_DEVICE
    Element *data() { return storage.data(); }

    /// Returns a tensor reference to the shared memory buffer
    CUTLASS_DEVICE
    TensorRef reference() {
      return TensorRef(storage.data(), Layout::packed({StorageShape::kRow, StorageShape::kColumn}));
    }
  };

 private:
  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;

  /// Stores a warp's fragment of accumulators to SMEM
  WarpTileIterator warp_tile_iterator_;

  LongIndex warp_offset;
  int thread_idx;
  int warp_idx;
  int lane_idx;
  int warp_m, warp_n;  // warp coordinates within a cta
  int tid_m, tid_n;    // thread coordinates within a warp

 public:
  /// Constructor
  CUTLASS_DEVICE
  EpilogueDepthwise(SharedStorage &shared_storage,  ///< Shared storage object
                    int thread_idx_,                ///< ID of a thread within the threadblock
                    int warp_idx_,                  ///< ID of warp within threadblock
                    int lane_idx_                   ///< Id of thread within warp
                    )
      : thread_idx(thread_idx_),
        warp_idx(warp_idx_),
        lane_idx(lane_idx_),
        shared_load_iterator_(shared_storage.reference(), thread_idx_),
        warp_tile_iterator_(shared_storage.reference(), thread_idx_, lane_idx_) {}

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(OutputOp const &output_op,                ///< Output operator
                  OutputTileIterator destination_iterator,  ///< Tile iterator for destination
                  AccumulatorTile const &accumulators,  ///< Complete warp-level accumulator tile
                  OutputTileIterator source_iterator,   ///< Threadblock tile coordinate in GEMM (in
                                                        ///< units of threadblock tiles)
                  const int smem_base_offset) {         ///< SMEM base offset for epilogue operation
    // initiate the smem base offset for different output tile.
    warp_tile_iterator_.set_smem_base_address(smem_base_offset);

    shared_load_iterator_.set_smem_base_address(smem_base_offset);

    if (!output_op.is_source_needed()) {
      compute_source_not_needed_(output_op, destination_iterator, accumulators);
    } else {
      compute_source_needed_(output_op, destination_iterator, accumulators, source_iterator);
    }
  }

 private:
  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_needed_(
      OutputOp const &output_op,                ///< Output operator
      OutputTileIterator destination_iterator,  ///< Tile iterator for destination
      AccumulatorTile const &accumulators,      ///< Complete warp-level accumulator tile
      OutputTileIterator source_iterator) {     ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    typename OutputTileIterator::Fragment source_fragment;

    source_fragment.clear();

    source_iterator.load(source_fragment);

    // store to smem
    warp_tile_iterator_.store(accumulators);

    __syncthreads();

    typename SharedLoadIterator::Fragment aligned_accum_fragment;

    // load from smem
    shared_load_iterator_.load(aligned_accum_fragment);

    typename OutputTileIterator::Fragment output_fragment;

    apply_output_operator_(output_fragment, output_op, aligned_accum_fragment, source_fragment);

    // Store to GMEM
    destination_iterator.store(output_fragment);
  }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_not_needed_(
      OutputOp const &output_op,                ///< Output operator
      OutputTileIterator destination_iterator,  ///< Tile iterator for destination
      AccumulatorTile const &accumulators) {    ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    // store to smem
    warp_tile_iterator_.store(accumulators);

    __syncthreads();

    typename SharedLoadIterator::Fragment aligned_accum_fragment;

    // load from smem
    shared_load_iterator_.load(aligned_accum_fragment);

    typename OutputTileIterator::Fragment output_fragment;

    apply_output_operator_source_not_needed_(output_fragment, output_op, aligned_accum_fragment);

    // Store to GMEM
    destination_iterator.store(output_fragment);
  }

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_(
    typename OutputTileIterator::Fragment &output_fragment,
    OutputOp const &output_op,                    ///< Output operator
    typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
    typename OutputTileIterator::Fragment const &source_fragment) {
      
    OutputAccessType *output_frag_ptr = 
      reinterpret_cast<OutputAccessType *>(&output_fragment);

    AccumulatorAccessType const *compute_frag_ptr = 
      reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

    OutputAccessType const *source_frag_ptr = 
      reinterpret_cast<OutputAccessType const *>(&source_fragment);

    int const kOutputOpIterations = 
      OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
      // Call the output operator
      output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
    }
  }

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_source_not_needed_(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,  ///< Output operator
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment) {
    OutputAccessType *output_frag_ptr = reinterpret_cast<OutputAccessType *>(&output_fragment);

    AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

    int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
      // Call the output operator
      output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
