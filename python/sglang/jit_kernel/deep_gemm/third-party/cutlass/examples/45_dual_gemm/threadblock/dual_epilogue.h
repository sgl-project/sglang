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
#include "cutlass/array.h"
#include CUDA_STD_HEADER(cassert)
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/numeric_types.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  ///< Output operator
  typename OutputOp0_,
  typename OutputOp1_,
  typename OutputOp2_,
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  bool StoreD0 = true,
  bool StoreD1 = true,
  int FragmentsPerPartition = 1,            ///< Used to coarsten the epilogue granularity
  int IterationsUnroll =                    ///< Used to reduce binary size when epilogue op is large
    (!IsEpilogueFunctorHeavy<OutputOp0_>::value)
>
class DualEpilogue {

public:

  using Base = EpilogueBase<
    Shape_, 
    typename WarpMmaOperator_::Shape, 
    PartitionsK, 
    AccumulatorFragmentIterator_, 
    WarpTileIterator_, 
    Padding_,
    FragmentsPerPartition>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  static bool constexpr kStoreD0 = StoreD0;
  static bool constexpr kStoreD1 = StoreD1;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp0 = OutputOp0_;
  using OutputOp1 = OutputOp1_;
  using OutputOp2 = OutputOp2_;
  using Padding = Padding_;

  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = typename Base::AccumulatorTile;

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
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Array type used by output functor
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>; 
  
  /// Number of warps
  using WarpCount = typename Base::WarpCount;

  struct SharedStorage {
    using Element = typename WarpTileIterator::Element;

    /// Tensor reference to shared memory allocation
    using TensorRef = typename WarpTileIterator::TensorRef;

    /// Logical shape of the shared memory tile written to by all warps.
    using Shape = typename Base::Shape;

    /// Shape of the shared memory allocation for the epilogue    
    using StorageShape = typename Base::SharedStorage::StorageShape;

    //
    // Data members
    //

    AlignedBuffer<Element, StorageShape::kCount> storage[2];

    //
    // Methods
    //

    /// Returns a tensor reference to the shared memory buffer
    CUTLASS_DEVICE
    TensorRef reference(int i) {
      return TensorRef(
        storage[i].data(), 
        Layout::packed({StorageShape::kRow, StorageShape::kColumn}));
    }
  };

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;
  static int constexpr kSmemPointerOffset = SharedStorage::StorageShape::kCount / kSmemTiles;

public:

  static_assert(SharedLoadIterator::Fragment::kElements == OutputTileIterator::Fragment::kElements,
    "Mismatch between shared load iterator and output tile iterator.");

  static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess), 
    "Divisibility");

private:

  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator0_;
  SharedLoadIterator shared_load_iterator1_;

  /// Stores a warp's fragment of accumulators to SMEM
  WarpTileIterator warp_tile_iterator0_;
  WarpTileIterator warp_tile_iterator1_;

public:

  /// Constructor
  CUTLASS_DEVICE
  DualEpilogue(
    SharedStorage &shared_storage,    ///< Shared storage object    
    int thread_idx,                   ///< ID of a thread within the threadblock
    int warp_idx,                     ///< ID of warp within threadblock
    int lane_idx                     ///< Id of thread within warp
  ):
    shared_load_iterator0_(shared_storage.reference(0), thread_idx),
    shared_load_iterator1_(shared_storage.reference(1), thread_idx),
    warp_tile_iterator0_(shared_storage.reference(0), lane_idx),
    warp_tile_iterator1_(shared_storage.reference(1), lane_idx)
  {
    int warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);
    int warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int warp_m = warp_mn % WarpCount::kM;
    int warp_n = warp_mn / WarpCount::kM;

    MatrixCoord warp_offset{warp_k * WarpCount::kM + warp_m, warp_n};

    warp_tile_iterator0_.add_tile_offset(warp_offset);
    warp_tile_iterator1_.add_tile_offset(warp_offset);
  }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp0 const &output_op0,
    OutputOp1 const &output_op1,
    OutputOp2 const &output_op2,
    OutputTileIterator dest0,
    OutputTileIterator dest1,
    OutputTileIterator dest2,
    AccumulatorTile const &accumulator0,
    AccumulatorTile const &accumulator1,
    OutputTileIterator source_iterator[2],
    bool writeToD2 // true if it's the final split-k
  ) {
    // TODO: Implement when no source is needed

    typename OutputTileIterator::Fragment source_fragment[2];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      source_fragment[i].clear();
    }

    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator[2] = {accumulator0, accumulator1};

    //
    // Iterate over accumulator tile
    // 

    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

      //
      // Load the source
      //

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 2; ++i) {
        source_iterator[i].load(source_fragment[i]);
        ++source_iterator[i];
      }

      //
      // Convert and store fragment
      //
      
      __syncthreads();

      acc2smem_source_needed<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
          iter, accum_fragment_iterator[0], this->warp_tile_iterator0_);
      acc2smem_source_needed<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
          iter, accum_fragment_iterator[1], this->warp_tile_iterator1_);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment aligned_accum_fragment0[kPartitionsK];
      typename SharedLoadIterator::Fragment aligned_accum_fragment1[kPartitionsK];

      shared_load_iterator0_.load(aligned_accum_fragment0[0]);
      shared_load_iterator1_.load(aligned_accum_fragment1[0]);

      // If the number of k-slices is > 1 - perform a reduction amongst the k-slices
      if (kPartitionsK > 1) {

        plus <typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for ( int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator0_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator1_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator0_.load(aligned_accum_fragment0[i]);
          shared_load_iterator1_.load(aligned_accum_fragment1[i]);
          aligned_accum_fragment0[0] = add_fragments(aligned_accum_fragment0[0], aligned_accum_fragment0[i]);
          aligned_accum_fragment1[0] = add_fragments(aligned_accum_fragment1[0], aligned_accum_fragment1[i]);
        }

        shared_load_iterator0_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
        shared_load_iterator1_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
      }

      //
      // Compute the output result
      //
     
      typename OutputTileIterator::Fragment output_fragment[3];

      apply_output_operator_(output_fragment,
        output_op0, output_op1, output_op2,
        aligned_accum_fragment0[0], aligned_accum_fragment1[0],
        source_fragment);


      //
      // Store the final result
      //

      if (kStoreD0) {
        dest0.store(output_fragment[0]);
        ++dest0;
      }
      if (kStoreD1) {
        dest1.store(output_fragment[1]);
        ++dest1;
      }
      if (writeToD2) {
        dest2.store(output_fragment[2]);
        ++dest2;
      }
    }
  }

private:

  static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1, "One of these must be exactly 1.");

  template<class Seq>
  struct acc2smem_source_needed;

  template <size_t... Seq>
  struct acc2smem_source_needed<cutlass::index_sequence<Seq...>> {
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

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_(
    typename OutputTileIterator::Fragment (&output_fragment)[3],
    OutputOp0 const &output_op0,
    OutputOp1 const &output_op1,
    OutputOp2 const &output_op2,
    typename SharedLoadIterator::Fragment const& aligned_accum_fragment0,
    typename SharedLoadIterator::Fragment const& aligned_accum_fragment1,
    typename OutputTileIterator::Fragment const (&source_fragment)[2]) {
      
    OutputAccessType* output_frag_ptr[3] = {
      reinterpret_cast<OutputAccessType *>(&output_fragment[0]),
      reinterpret_cast<OutputAccessType *>(&output_fragment[1]),
      reinterpret_cast<OutputAccessType *>(&output_fragment[2])
    };

    AccumulatorAccessType const *compute_frag_ptr[2] = {
      reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment0),
      reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment1)
    };

    OutputAccessType const *source_frag_ptr[2] = {
      reinterpret_cast<OutputAccessType const *>(&source_fragment[0]),
      reinterpret_cast<OutputAccessType const *>(&source_fragment[1])
    };

    int const kOutputOpIterations = 
      OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {

      // Call the output operators
      output_frag_ptr[0][i] = output_op0(compute_frag_ptr[0][i], source_frag_ptr[0][i]);
      output_frag_ptr[1][i] = output_op1(compute_frag_ptr[1][i], source_frag_ptr[1][i]);
      output_frag_ptr[2][i] = output_op2(output_frag_ptr[0][i], output_frag_ptr[1][i]);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
