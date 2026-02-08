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
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/array_planar_complex.h"
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

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator for planar-complex output representations.
///
/// Note, as with most CUTLASS components for planar complex, the template arguments describe
/// the underlying real data type.
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_,                       ///< Output operator
  typename Padding_                         ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
>
class EpiloguePlanarComplex {
public:
  
  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = Padding_;

  /// Output layout is always row-major
  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// The complete warp-level accumulator tile
  using AccumulatorTile = ArrayPlanarComplex<
    typename WarpMmaOperator::FragmentC::Element, 
    WarpMmaOperator::FragmentC::kElements
  >;

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
  
  /// Shape of each warp-level operation
  using WarpShape = typename WarpMmaOperator::Shape;

  /// Number of warps
  using WarpCount = gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    kPartitionsK
  >;

  /// Shared memory allocation
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
    using Shape = MatrixShape<
      WarpCount::kM * WarpTileIterator::Shape::kRow * WarpCount::kK,
      WarpCount::kN * WarpTileIterator::Shape::kColumn
    >;

    /// Shape of the shared memory allocation for the epilogue    
    using StorageShape = MatrixShape<
      Shape::kRow + Padding::kRow, 
      Shape::kColumn + Padding::kColumn
    >;

    static int const kImaginaryStride = StorageShape::kCount;

    //
    // Data members
    //

    AlignedBuffer<Element, kImaginaryStride * 2> storage;

    //
    // Methods
    //

    /// Returns a pointer to the shared memory buffer
    CUTLASS_DEVICE
    Element *data() {
      return storage.data();
    }

    /// Returns a tensor reference to the shared memory buffer
    CUTLASS_DEVICE
    TensorRef reference() {
      return TensorRef(
        storage.data(), 
        Layout::packed({StorageShape::kRow, StorageShape::kColumn}));
    }
  };

private:

  //
  // Data members
  //

  SharedStorage &shared_storage_;

  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;

  /// Stores a warp's fragment of accumulators to SMEM
  WarpTileIterator warp_tile_iterator_;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpiloguePlanarComplex(
    SharedStorage &shared_storage,    ///< Shared storage object    
    int thread_idx,                   ///< ID of a thread within the threadblock
    int warp_idx,                     ///< ID of warp within threadblock
    int lane_idx                      ///< Id of thread within warp
  ):
    shared_storage_(shared_storage),
    shared_load_iterator_(shared_storage.reference(), thread_idx),
    warp_tile_iterator_(shared_storage.reference(), lane_idx) {

    // Compute warp location within threadblock tile by mapping the warp_id to three coordinates:
    //
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);
    int warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int warp_m = warp_mn % WarpCount::kM;
    int warp_n = warp_mn / WarpCount::kM;

    MatrixCoord warp_offset{warp_k * WarpCount::kM + warp_m, warp_n};

    warp_tile_iterator_.add_tile_offset(warp_offset);
  }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                        ///< Output operator
    OutputTileIterator destination_iterator_real,     ///< Tile iterator for destination
    OutputTileIterator destination_iterator_imag,     ///< Tile iterator for destination
    AccumulatorTile const &accumulators,              ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator_real,          ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
    OutputTileIterator source_iterator_imag) {        ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    typename OutputTileIterator::Fragment source_fragment_real;
    typename OutputTileIterator::Fragment source_fragment_imag;

    if (!output_op.is_source_needed()) {
      source_iterator_real.clear_mask();
      source_iterator_imag.clear_mask();
    }

    source_fragment_real.clear();
    source_fragment_imag.clear();

    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator_real(accumulators.real);
    AccumulatorFragmentIterator accum_fragment_iterator_imag(accumulators.imag);

    //
    // Iterate over accumulator tile
    // 

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

      //
      // Load the source
      //

      source_iterator_real.load(source_fragment_real);
      source_iterator_imag.load(source_fragment_imag);

      ++source_iterator_real;
      ++source_iterator_imag;

      //
      // Convert and store fragment
      //
      
      __syncthreads();

      typename AccumulatorFragmentIterator::Fragment accum_fragment_real;
      typename AccumulatorFragmentIterator::Fragment accum_fragment_imag;

      accum_fragment_iterator_real.load(accum_fragment_real);
      accum_fragment_iterator_imag.load(accum_fragment_imag);
      
      ++accum_fragment_iterator_real;
      ++accum_fragment_iterator_imag;

      this->warp_tile_iterator_.store(accum_fragment_real);
      this->warp_tile_iterator_.store_with_pointer_offset(accum_fragment_imag, SharedStorage::kImaginaryStride);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment aligned_accum_fragment_real[kPartitionsK];
      typename SharedLoadIterator::Fragment aligned_accum_fragment_imag[kPartitionsK];

      shared_load_iterator_.load(aligned_accum_fragment_real[0]);
      shared_load_iterator_.load_with_pointer_offset(aligned_accum_fragment_imag[0], SharedStorage::kImaginaryStride);

      // If the number of k-slices is > 1 - perform a reduction amongst the k-slices
      static_assert(kPartitionsK  == 1, "Sliced-K not supported for planar complex at this time");
    
      //
      // Compute the output result
      //
     
      typename OutputTileIterator::Fragment output_fragment_real;
      typename OutputTileIterator::Fragment output_fragment_imag;

      apply_output_operator_(
        output_fragment_real, 
        output_fragment_imag, 
        output_op, 
        aligned_accum_fragment_real[0],
        aligned_accum_fragment_imag[0], 
        source_fragment_real,
        source_fragment_imag);

      //
      // Store the final result
      //

      destination_iterator_real.store(output_fragment_real);
      destination_iterator_imag.store(output_fragment_imag);

      ++destination_iterator_real;
      ++destination_iterator_imag;
    }
  }

private:

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_(
    typename OutputTileIterator::Fragment &output_fragment_real,
    typename OutputTileIterator::Fragment &output_fragment_imag,
    OutputOp const &output_op,                    ///< Output operator
    typename SharedLoadIterator::Fragment const &aligned_accum_fragment_real,
    typename SharedLoadIterator::Fragment const &aligned_accum_fragment_imag,
    typename OutputTileIterator::Fragment const &source_fragment_real,
    typename OutputTileIterator::Fragment const &source_fragment_imag) {

    OutputAccessType *output_frag_real_ptr = 
      reinterpret_cast<OutputAccessType *>(&output_fragment_real);

    OutputAccessType *output_frag_imag_ptr = 
      reinterpret_cast<OutputAccessType *>(&output_fragment_imag);

    AccumulatorAccessType const *compute_frag_real_ptr = 
      reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment_real);

    AccumulatorAccessType const *compute_frag_imag_ptr = 
      reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment_imag);

    OutputAccessType const *source_frag_real_ptr = 
      reinterpret_cast<OutputAccessType const *>(&source_fragment_real);

    OutputAccessType const *source_frag_imag_ptr = 
      reinterpret_cast<OutputAccessType const *>(&source_fragment_imag);

    int const kOutputOpIterations = 
      OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {

      // Call the output operator
      auto result_fragment = output_op(
        make_ArrayPlanarComplex(compute_frag_real_ptr[i], compute_frag_imag_ptr[i]), 
        make_ArrayPlanarComplex(source_frag_real_ptr[i], source_frag_imag_ptr[i])
      );

      output_frag_real_ptr[i] = result_fragment.real;
      output_frag_imag_ptr[i] = result_fragment.imag;
    }
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
