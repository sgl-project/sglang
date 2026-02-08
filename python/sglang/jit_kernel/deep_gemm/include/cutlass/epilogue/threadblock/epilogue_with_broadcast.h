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

#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(utility)
#else
#include <utility>
#endif

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

#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This base class is meant to define the concept required of the
/// EpilogueWithBroadcast::OutputOp
template <
  typename ElementC_,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementZ_,
  typename ElementT_,
  int ElementsPerAccess,
  bool StoreZ = true,
  bool StoreT = true
>
struct EpilogueWithBroadcastOpBase {
  
  using ElementOutput = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementZ = ElementZ_;
  using ElementT = ElementT_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = Array<ElementOutput, kElementsPerAccess>;
  using FragmentZ = Array<ElementZ, kElementsPerAccess>;
  using FragmentT = Array<ElementT, kElementsPerAccess>;

  /// If true, the 'Z' tensor is stored
  static bool const kStoreZ = StoreZ;

  /// If true, the 'T' tensor is stored
  static bool const kStoreT = StoreT;

  /// Parameters structure - required
  struct Params { };

  //
  // Methods
  //

  /// Constructor from Params
  EpilogueWithBroadcastOpBase(Params const &params_) { }

  /// Determine if the source is needed. May return false if 
  bool is_source_needed() const {
    return true;
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) { }

  /// Applies the operation when is_source_needed() is true
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z, 
    FragmentT &frag_T, 
    FragmentAccumulator const &AB,
    FragmentC const &frag_C1,
    FragmentC const &frag_C2,
    FragmentCompute const &V) const {

  }

  /// Applies the operation when is_source_needed() is false
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z, 
    FragmentT &frag_T, 
    FragmentAccumulator const &AB,
    FragmentCompute const &V) const {

  }
};

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator with bias vector broadcast over columns.
///
/// Computes the following:
///
///
///  Z, T = OutputOp(AB, C, Broadcast)
///
///  if (ElementwiseOp::kStoreZ) {
///    store(converted_u);
///  }  
///
///  if (ElementwiseOp::kStoreT) {
///    store(v);
///  }  
///
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors (z)
  typename TensorTileIterator_,             ///< Additional tile iterator for tensor-valued operands (t)
  typename ElementVector_,                  ///< Pointer to broadcast vector
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_,                       ///< Output operator - concept is EpilogueWithBroadcastOp
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  int FragmentsPerPartition = 1,            ///< Used to coarsten the epilogue granularity
  int IterationsUnroll =                    ///< Used to reduce binary size when epilogue op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value),
  bool IsSingleSource = OutputOp_::kIsSingleSource
>
class EpilogueWithBroadcast;

template <
  typename Shape_,
  typename WarpMmaOperator_,
  int PartitionsK,
  typename OutputTileIterator_,
  typename TensorTileIterator_,
  typename ElementVector_,
  typename AccumulatorFragmentIterator_,
  typename WarpTileIterator_,
  typename SharedLoadIterator_,
  typename OutputOp_,
  typename Padding_,
  int FragmentsPerPartition,
  int IterationsUnroll
>
class EpilogueWithBroadcast<
  Shape_,
  WarpMmaOperator_,
  PartitionsK,
  OutputTileIterator_,
  TensorTileIterator_,
  ElementVector_,
  AccumulatorFragmentIterator_,
  WarpTileIterator_,
  SharedLoadIterator_,
  OutputOp_,
  Padding_,
  FragmentsPerPartition,
  IterationsUnroll,
  false
> : 
  public EpilogueBase<
    Shape_, 
    typename WarpMmaOperator_::Shape, 
    PartitionsK, 
    AccumulatorFragmentIterator_, 
    WarpTileIterator_, 
    Padding_,
    FragmentsPerPartition> {

public:

  using Base = EpilogueBase<
    Shape_, 
    typename WarpMmaOperator_::Shape, 
    PartitionsK, 
    AccumulatorFragmentIterator_, 
    WarpTileIterator_, 
    Padding_,
    FragmentsPerPartition>;

  static bool const kIsSingleSource = false;
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
  using Padding = Padding_;

  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

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

  /// Fragment object used to store the broadcast values
  using BroadcastFragment = Array<
    ElementCompute, 
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

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;
  static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;

  /// Used for the broadcast
  struct BroadcastDetail {

    /// Number of threads per warp
    static int const kWarpSize = 32;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

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
      printf("BroadcastDetail {\n");
      printf(
        "  kColumnsPerThread: %d\nkRowsPerThread: %d\n,kThreadCount: %d\nkThreadsPerRow: %d\n"
        "kThreadRows: %d\nThreadAccessesPerRow: %d\nStorageShape: %d x %d (count: %d)\n",
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

  /// Thread index within the threadblock
  int thread_idx_;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueWithBroadcast(
    SharedStorage &shared_storage,                    ///< Shared storage object    
    int thread_idx,                                   ///< ID of a thread within the threadblock
    int warp_idx,                                     ///< ID of warp within threadblock
    int lane_idx                                      ///< Id of thread within warp
  ):
    Base(shared_storage.base, thread_idx, warp_idx, lane_idx),
    shared_load_iterator_(shared_storage.base.reference(), thread_idx),
    thread_idx_(thread_idx)
  {

  }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                        ///< Output operator
    ElementVector const * broadcast_ptr,              ///< Broadcast vector
    OutputTileIterator destination_iterator,          ///< Tile iterator for destination
    AccumulatorTile const &accumulators,              ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator1,              ///< Tile iterator for first source accumulator matrix
    OutputTileIterator source_iterator2,              ///< Tile iterator for second source accumulator matrix
    TensorTileIterator tensor_iterator,               ///< Threadblock tile iterator for additional tensor operand
    MatrixCoord const &problem_size =                 ///< Problem size needed to guard against out-of-bounds accesses
        MatrixCoord(Shape::kM, Shape::kN),
    MatrixCoord const &threadblock_offset =           ///< Threadblock's initial offset within the problem size space
        MatrixCoord()) {
    
    BroadcastFragment broadcast_fragment;

    load_broadcast_fragment_(broadcast_fragment, broadcast_ptr, problem_size, threadblock_offset);

    if (!output_op.is_source_needed()) {
      compute_source_not_needed_(
        output_op, 
        broadcast_fragment, 
        destination_iterator, 
        accumulators,
        tensor_iterator);
    }
    else {
      compute_source_needed_(
        output_op, 
        broadcast_fragment, 
        destination_iterator, 
        accumulators, 
        source_iterator1,
        source_iterator2,
        tensor_iterator);
    }
  }

private:

  CUTLASS_DEVICE
  void load_broadcast_fragment_(
    BroadcastFragment & broadcast_fragment,      ///< Fragment containing the accumulated partial reduction over columns
    ElementVector const * broadcast_ptr,         ///< Broadcast vector
    MatrixCoord const &problem_size,             ///< Problem size needed to guard against out-of-bounds accesses
    MatrixCoord const &threadblock_offset        ///< Threadblock's initial offset within the problem size space
    ) {

    broadcast_fragment.clear();
    
    // If no pointer is supplied, set with all zeros and avoid memory accesses
    if (!broadcast_ptr) {
      return;
    }

    int thread_initial_column = ThreadMap::initial_offset(thread_idx_).column();

    int thread_column_idx = threadblock_offset.column() + thread_initial_column;
    broadcast_ptr += thread_initial_column;

    NumericArrayConverter<ElementCompute, ElementVector, BroadcastDetail::kElementsPerAccess> converter;
    using AccessType = AlignedArray<ElementVector, BroadcastDetail::kElementsPerAccess>;
    using ComputeFragmentType = Array<ElementCompute, BroadcastDetail::kElementsPerAccess>;

    ComputeFragmentType *frag_ptr = reinterpret_cast<ComputeFragmentType *>(&broadcast_fragment);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < ThreadMap::Iterations::kColumn; ++j) {

      AccessType loaded;

      loaded.clear();

      if (thread_column_idx < problem_size.column()) {
        loaded = *reinterpret_cast<AccessType const *>(broadcast_ptr);
      }

      ComputeFragmentType cvt = converter(loaded);
      frag_ptr[j] = cvt;

      thread_column_idx += ThreadMap::Delta::kColumn;
      broadcast_ptr += ThreadMap::Delta::kColumn;
    }
  }

  template <class Seq>
  struct acc2smem_source_not_needed;

  template <size_t... Seq>
  struct acc2smem_source_not_needed<cutlass::index_sequence<Seq...>> {
    template <int Advance>
    CUTLASS_DEVICE static void helper(AccumulatorFragmentIterator accum_fragment_iterator,
                                      WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {
        typename AccumulatorFragmentIterator::Fragment accum_fragment;

        accum_fragment_iterator.load(accum_fragment);
        ++accum_fragment_iterator;

        warp_tile_iterator.store(accum_fragment);
        if (p < Base::kFragmentsPerIteration - 1) {
          warp_tile_iterator.add_pointer_offset(kSmemPointerOffset);
        }
      }

      if (Base::kFragmentsPerIteration > 1) {
        warp_tile_iterator.add_pointer_offset(kSmemPointerOffset *
                                              (1 - Base::kFragmentsPerIteration));
      }
    }

    CUTLASS_DEVICE
    static void push(size_t pos,
                     AccumulatorFragmentIterator const &iterator_begin,
                     WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {
          (pos == (Seq * Base::kFragmentsPerIteration)) &&
          (helper<Seq * Base::kFragmentsPerIteration>(iterator_begin, warp_tile_iterator), 0)...};

      CUTLASS_UNUSED(dummy[0]);
    }
  };

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_not_needed_(
    OutputOp const &output_op,                        ///< Output operator
    BroadcastFragment const &broadcast_fragment,      ///< Fragment containing the accumulated partial reduction over columns
    OutputTileIterator destination_iterator,          ///< Tile iterator for destination
    AccumulatorTile const &accumulators,              ///< Complete warp-level accumulator tile 
    TensorTileIterator tensor_iterator                ///< Threadblock tile iterator for additioanl tensor operand
    ) { 

    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    // 

    // CUTLASS_PRAGMA_UNROLL
    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations / Base::kFragmentsPerIteration : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; iter += Base::kFragmentsPerIteration) {

      //
      // Convert and store fragment
      //
      

      __syncthreads();

      acc2smem_source_not_needed<
          cutlass::make_index_sequence<OutputTileIterator::kIterations /
                                   Base::kFragmentsPerIteration>>::push(iter,
                                                                        accum_fragment_iterator,
                                                                        this->warp_tile_iterator_);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {


        typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];

        shared_load_iterator_.load(aligned_accum_fragment[0]);

        if (p < Base::kFragmentsPerIteration - 1) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
        }
        else if (kPartitionsK > 1) {

          plus <typename SharedLoadIterator::Fragment> add_fragments;

          CUTLASS_PRAGMA_UNROLL
          for ( int i = 1; i < kPartitionsK; ++i) {
            shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
            shared_load_iterator_.load(aligned_accum_fragment[i]);
            aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
          }

          shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
        }

        //
        // Apply output operation
        //

        typename OutputTileIterator::Fragment frag_Z;
        typename TensorTileIterator::Fragment frag_T;

        apply_output_operator_source_not_needed_(
          frag_Z,
          frag_T,
          output_op,
          aligned_accum_fragment[0],
          broadcast_fragment);

        //
        // Conditionally store fragments
        //

        if (OutputOp::kStoreZ) {
          destination_iterator.store(frag_Z);
          ++destination_iterator;
        }

        if (OutputOp::kStoreT) {
          tensor_iterator.store(frag_T);
          ++tensor_iterator;
        }
      }

      if (Base::kFragmentsPerIteration > 1) {
        shared_load_iterator_.add_pointer_offset(kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
      }
    }
  }


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

  
  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_needed_(
    OutputOp const &output_op,                    ///< Output operator
    BroadcastFragment const &broadcast_fragment,  ///< Fragment containing the accumulated partial reduction over columns
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators,          ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator1,          ///< Tile iterator for first source accumulator matrix
    OutputTileIterator source_iterator2,          ///< Tile iterator for second source accumulator matrix
    TensorTileIterator tensor_iterator            ///< Threadblock tile iterator for additioanl tensor operand
    ) { 
    
    typename OutputTileIterator::Fragment source_fragment1;
    source_fragment1.clear();
    typename OutputTileIterator::Fragment source_fragment2;
    source_fragment2.clear();

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

      source_iterator1.load(source_fragment1);
      ++source_iterator1;

      source_iterator2.load(source_fragment2);
      ++source_iterator2;

      //
      // Convert and store fragment
      //
      
      __syncthreads();

      acc2smem_source_needed<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
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
      // Apply output operation
      //

      typename OutputTileIterator::Fragment frag_Z;
      typename TensorTileIterator::Fragment frag_T;

      apply_output_operator_(
        frag_Z,
        frag_T,
        output_op,
        aligned_accum_fragment[0],
        source_fragment1,
        source_fragment2,
        broadcast_fragment);

      //
      // Conditionally store fragments
      //

      if (OutputOp::kStoreZ) {
        destination_iterator.store(frag_Z);
        ++destination_iterator;
      }

      if (OutputOp::kStoreT) {
        tensor_iterator.store(frag_T);
        ++tensor_iterator;
      }
    }
  }

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_(
    typename OutputTileIterator::Fragment &frag_Z,
    typename TensorTileIterator::Fragment &frag_T,
    OutputOp const &output_op,
    typename SharedLoadIterator::Fragment const &frag_AB,
    typename OutputTileIterator::Fragment const &frag_C1,
    typename OutputTileIterator::Fragment const &frag_C2,
    BroadcastFragment const &frag_Broadcast) {

    using AccessTypeZ = Array<typename OutputTileIterator::Element, kElementsPerAccess>;
    using AccessTypeT = Array<typename TensorTileIterator::Element, kElementsPerAccess>;
    using AccessTypeBroadcast = Array<ElementCompute, kElementsPerAccess>;

    AccessTypeZ *frag_Z_ptr = reinterpret_cast<AccessTypeZ *>(&frag_Z);
    AccessTypeT *frag_T_ptr = reinterpret_cast<AccessTypeT *>(&frag_T);
    
    AccumulatorAccessType const *frag_AB_ptr = 
      reinterpret_cast<AccumulatorAccessType const *>(&frag_AB);

    OutputAccessType const *frag_C1_ptr =
      reinterpret_cast<OutputAccessType const *>(&frag_C1);

    OutputAccessType const *frag_C2_ptr =
      reinterpret_cast<OutputAccessType const *>(&frag_C2);

    AccessTypeBroadcast const *frag_Broadcast_ptr =
      reinterpret_cast<AccessTypeBroadcast const *>(&frag_Broadcast);

    int const kOutputOpIterations = 
      OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
        output_op(
          frag_Z_ptr[i],
          frag_T_ptr[i],
          frag_AB_ptr[i],
          frag_C1_ptr[i],
          frag_C2_ptr[i],
          frag_Broadcast_ptr[i % ThreadMap::Iterations::kColumn]);
    }
  }

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_source_not_needed_(
    typename OutputTileIterator::Fragment &frag_Z,
    typename TensorTileIterator::Fragment &frag_T,
    OutputOp const &output_op,
    typename SharedLoadIterator::Fragment const &frag_AB,
    BroadcastFragment const &frag_Broadcast) {

    using AccessTypeZ = Array<typename OutputTileIterator::Element, kElementsPerAccess>;
    using AccessTypeT = Array<typename TensorTileIterator::Element, kElementsPerAccess>;
    using AccessTypeBroadcast = Array<ElementCompute, kElementsPerAccess>;

    AccessTypeZ *frag_Z_ptr = reinterpret_cast<AccessTypeZ *>(&frag_Z);
    AccessTypeT *frag_T_ptr = reinterpret_cast<AccessTypeT *>(&frag_T);
    
    AccumulatorAccessType const *frag_AB_ptr = 
      reinterpret_cast<AccumulatorAccessType const *>(&frag_AB);

    AccessTypeBroadcast const *frag_Broadcast_ptr =
      reinterpret_cast<AccessTypeBroadcast const *>(&frag_Broadcast);

    int const kOutputOpIterations = 
      OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {

      output_op(
        frag_Z_ptr[i], 
        frag_T_ptr[i], 
        frag_AB_ptr[i], 
        frag_Broadcast_ptr[i % ThreadMap::Iterations::kColumn]);
    }
  }

  public:
    /// Stream-K reduce helper
    CUTLASS_DEVICE
    void reduce(
        int reduce_fragment_idx,                        ///< Reduce fragment index
        OutputOp const &output_op,                      ///< Output operator
        ElementVector const * broadcast_ptr,            ///< Broadcast vector
        OutputTileIterator destination_iterator,        ///< Tile iterator for destination
        OutputTileIterator source_iterator1,            ///< Tile iterator for first  source accumulator matrix
        OutputTileIterator source_iterator2,            ///< Tile iterator for second source accumulator matrix
        TensorTileIterator tensor_iterator,             ///< Threadblock tile iterator for additional tensor operand
        MatrixCoord const &problem_size =               ///< Problem size needed to guard against out-of-bounds accesses
            MatrixCoord(Shape::kM, Shape::kN),
        MatrixCoord const &threadblock_offset =         ///< Threadblock's initial offset within the problem size space
            MatrixCoord()) 
    {

      BroadcastFragment broadcast_fragment;
      load_broadcast_fragment_(broadcast_fragment, broadcast_ptr, problem_size, threadblock_offset);

      // Initialize/load source-fragment data
      typename OutputTileIterator::Fragment source_fragment1;
      source_fragment1.clear();
      typename OutputTileIterator::Fragment source_fragment2;
      source_fragment2.clear();

      if (output_op.is_source_needed())
      {
        source_iterator1 += reduce_fragment_idx;
        source_iterator1.load(source_fragment1);

        source_iterator2 += reduce_fragment_idx;
        source_iterator2.load(source_fragment2);
      }

      // Load fragment from shared memory
      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
      shared_load_iterator_.load(aligned_accum_fragment[0]);

      // Add fragments shared by other k partitions
      if (kPartitionsK > 1)
      {
        plus <typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for ( int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
        }
      }

      //
      // Apply output operation
      //

      typename OutputTileIterator::Fragment frag_Z;
      typename TensorTileIterator::Fragment frag_T;

      if (!output_op.is_source_needed()) {
        apply_output_operator_source_not_needed_(
          frag_Z,
          frag_T,
          output_op,
          aligned_accum_fragment[0],
          broadcast_fragment);
      } else {
        apply_output_operator_(
          frag_Z,
          frag_T,
          output_op,
          aligned_accum_fragment[0],
          source_fragment1,
          source_fragment2,
          broadcast_fragment);
      }

      //
      // Conditionally store fragments
      //

      if (OutputOp::kStoreZ) {
        destination_iterator += reduce_fragment_idx;
        destination_iterator.store(frag_Z);
      }

      if (OutputOp::kStoreT) {
        tensor_iterator += reduce_fragment_idx;
        tensor_iterator.store(frag_T);
      }
    }
};


template <
  typename Shape_,
  typename WarpMmaOperator_,
  int PartitionsK,
  typename OutputTileIterator_,
  typename TensorTileIterator_,
  typename ElementVector_,
  typename AccumulatorFragmentIterator_,
  typename WarpTileIterator_,
  typename SharedLoadIterator_,
  typename OutputOp_,
  typename Padding_,
  int FragmentsPerPartition,
  int IterationsUnroll
>
class EpilogueWithBroadcast<
  Shape_,
  WarpMmaOperator_,
  PartitionsK,
  OutputTileIterator_,
  TensorTileIterator_,
  ElementVector_,
  AccumulatorFragmentIterator_,
  WarpTileIterator_,
  SharedLoadIterator_,
  OutputOp_,
  Padding_,
  FragmentsPerPartition,
  IterationsUnroll,
  true
> : 
  public EpilogueBase<
    Shape_, 
    typename WarpMmaOperator_::Shape, 
    PartitionsK, 
    AccumulatorFragmentIterator_, 
    WarpTileIterator_, 
    Padding_,
    FragmentsPerPartition> {

public:

  using Base = EpilogueBase<
    Shape_, 
    typename WarpMmaOperator_::Shape, 
    PartitionsK, 
    AccumulatorFragmentIterator_, 
    WarpTileIterator_, 
    Padding_,
    FragmentsPerPartition>;

  static bool const kIsSingleSource = true;
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
  using Padding = Padding_;

  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

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

  /// Fragment object used to store the broadcast values
  using BroadcastFragment = Array<
    ElementCompute, 
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

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;
  static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;

  /// Used for the broadcast
  struct BroadcastDetail {

    /// Number of threads per warp
    static int const kWarpSize = 32;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

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
      printf("BroadcastDetail {\n");
      printf(
        "  kColumnsPerThread: %d\nkRowsPerThread: %d\n,kThreadCount: %d\nkThreadsPerRow: %d\n"
        "kThreadRows: %d\nThreadAccessesPerRow: %d\nStorageShape: %d x %d (count: %d)\n",
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

  /// Thread index within the threadblock
  int thread_idx_;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueWithBroadcast(
    SharedStorage &shared_storage,                    ///< Shared storage object    
    int thread_idx,                                   ///< ID of a thread within the threadblock
    int warp_idx,                                     ///< ID of warp within threadblock
    int lane_idx                                      ///< Id of thread within warp
  ):
    Base(shared_storage.base, thread_idx, warp_idx, lane_idx),
    shared_load_iterator_(shared_storage.base.reference(), thread_idx),
    thread_idx_(thread_idx)
  {

  }

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                        ///< Output operator
    ElementVector const * broadcast_ptr,              ///< Broadcast vector
    OutputTileIterator destination_iterator,          ///< Tile iterator for destination
    AccumulatorTile const &accumulators,              ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator,               ///< Tile iterator for source accumulator matrix
    TensorTileIterator tensor_iterator,               ///< Threadblock tile iterator for additional tensor operand
    MatrixCoord const &problem_size =                 ///< Problem size needed to guard against out-of-bounds accesses
        MatrixCoord(Shape::kM, Shape::kN),
    MatrixCoord const &threadblock_offset =           ///< Threadblock's initial offset within the problem size space
        MatrixCoord()) {
    
    BroadcastFragment broadcast_fragment;

    load_broadcast_fragment_(broadcast_fragment, broadcast_ptr, problem_size, threadblock_offset);

    if (!output_op.is_source_needed()) {
      compute_source_not_needed_(
        output_op, 
        broadcast_fragment, 
        destination_iterator, 
        accumulators,
        tensor_iterator);
    }
    else {
      compute_source_needed_(
        output_op, 
        broadcast_fragment, 
        destination_iterator, 
        accumulators, 
        source_iterator,
        tensor_iterator);
    }
  }

private:

  CUTLASS_DEVICE
  void load_broadcast_fragment_(
    BroadcastFragment & broadcast_fragment,      ///< Fragment containing the accumulated partial reduction over columns
    ElementVector const * broadcast_ptr,         ///< Broadcast vector
    MatrixCoord const &problem_size,             ///< Problem size needed to guard against out-of-bounds accesses
    MatrixCoord const &threadblock_offset        ///< Threadblock's initial offset within the problem size space
    ) {

    broadcast_fragment.clear();
    
    // If no pointer is supplied, set with all zeros and avoid memory accesses
    if (!broadcast_ptr) {
      return;
    }

    int thread_initial_column = ThreadMap::initial_offset(thread_idx_).column();

    int thread_column_idx = threadblock_offset.column() + thread_initial_column;
    broadcast_ptr += thread_initial_column;

    NumericArrayConverter<ElementCompute, ElementVector, BroadcastDetail::kElementsPerAccess> converter;
    using AccessType = AlignedArray<ElementVector, BroadcastDetail::kElementsPerAccess>;
    using ComputeFragmentType = Array<ElementCompute, BroadcastDetail::kElementsPerAccess>;

    ComputeFragmentType *frag_ptr = reinterpret_cast<ComputeFragmentType *>(&broadcast_fragment);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < ThreadMap::Iterations::kColumn; ++j) {

      AccessType loaded;

      loaded.clear();

      if (thread_column_idx < problem_size.column()) {
        loaded = *reinterpret_cast<AccessType const *>(broadcast_ptr);
      }

      ComputeFragmentType cvt = converter(loaded);
      frag_ptr[j] = cvt;

      thread_column_idx += ThreadMap::Delta::kColumn;
      broadcast_ptr += ThreadMap::Delta::kColumn;
    }
  }

  template <class Seq>
  struct acc2smem_source_not_needed;

  template <size_t... Seq>
  struct acc2smem_source_not_needed<cutlass::index_sequence<Seq...>> {
    template <int Advance>
    CUTLASS_DEVICE static void helper(AccumulatorFragmentIterator accum_fragment_iterator,
                                      WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {
        typename AccumulatorFragmentIterator::Fragment accum_fragment;

        accum_fragment_iterator.load(accum_fragment);
        ++accum_fragment_iterator;

        warp_tile_iterator.store(accum_fragment);
        if (p < Base::kFragmentsPerIteration - 1) {
          warp_tile_iterator.add_pointer_offset(kSmemPointerOffset);
        }
      }

      if (Base::kFragmentsPerIteration > 1) {
        warp_tile_iterator.add_pointer_offset(kSmemPointerOffset *
                                              (1 - Base::kFragmentsPerIteration));
      }
    }

    CUTLASS_DEVICE
    static void push(size_t pos,
                     AccumulatorFragmentIterator const &iterator_begin,
                     WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {
          (pos == (Seq * Base::kFragmentsPerIteration)) &&
          (helper<Seq * Base::kFragmentsPerIteration>(iterator_begin, warp_tile_iterator), 0)...};

      CUTLASS_UNUSED(dummy[0]);
    }
  };

  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_not_needed_(
    OutputOp const &output_op,                        ///< Output operator
    BroadcastFragment const &broadcast_fragment,      ///< Fragment containing the accumulated partial reduction over columns
    OutputTileIterator destination_iterator,          ///< Tile iterator for destination
    AccumulatorTile const &accumulators,              ///< Complete warp-level accumulator tile 
    TensorTileIterator tensor_iterator                ///< Threadblock tile iterator for additioanl tensor operand
    ) { 

    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    // 

    // CUTLASS_PRAGMA_UNROLL
    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations / Base::kFragmentsPerIteration : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; iter += Base::kFragmentsPerIteration) {

      //
      // Convert and store fragment
      //
      

      __syncthreads();

      acc2smem_source_not_needed<
          cutlass::make_index_sequence<OutputTileIterator::kIterations /
                                   Base::kFragmentsPerIteration>>::push(iter,
                                                                        accum_fragment_iterator,
                                                                        this->warp_tile_iterator_);

      __syncthreads();

      //
      // Load fragments from shared memory
      //

      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {


        typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];

        shared_load_iterator_.load(aligned_accum_fragment[0]);

        if (p < Base::kFragmentsPerIteration - 1) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
        }
        else if (kPartitionsK > 1) {

          plus <typename SharedLoadIterator::Fragment> add_fragments;

          CUTLASS_PRAGMA_UNROLL
          for ( int i = 1; i < kPartitionsK; ++i) {
            shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
            shared_load_iterator_.load(aligned_accum_fragment[i]);
            aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
          }

          shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
        }

        //
        // Apply output operation
        //

        typename OutputTileIterator::Fragment frag_Z;
        typename TensorTileIterator::Fragment frag_T;

        apply_output_operator_source_not_needed_(
          frag_Z,
          frag_T,
          output_op,
          aligned_accum_fragment[0],
          broadcast_fragment);

        //
        // Conditionally store fragments
        //

        if (OutputOp::kStoreZ) {
          destination_iterator.store(frag_Z);
          ++destination_iterator;
        }

        if (OutputOp::kStoreT) {
          tensor_iterator.store(frag_T);
          ++tensor_iterator;
        }
      }

      if (Base::kFragmentsPerIteration > 1) {
        shared_load_iterator_.add_pointer_offset(kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
      }
    }
  }


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

  
  /// Streams the result to global memory
  CUTLASS_DEVICE
  void compute_source_needed_(
    OutputOp const &output_op,                    ///< Output operator
    BroadcastFragment const &broadcast_fragment,  ///< Fragment containing the accumulated partial reduction over columns
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators,          ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator,           ///< Tile iterator for source accumulator matrix
    TensorTileIterator tensor_iterator            ///< Threadblock tile iterator for additioanl tensor operand
    ) { 
    
    typename OutputTileIterator::Fragment source_fragment;
    source_fragment.clear();

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

      source_iterator.load(source_fragment);
      ++source_iterator;

      //
      // Convert and store fragment
      //
      
      __syncthreads();

      acc2smem_source_needed<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
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
      // Apply output operation
      //

      typename OutputTileIterator::Fragment frag_Z;
      typename TensorTileIterator::Fragment frag_T;

      apply_output_operator_(
        frag_Z,
        frag_T,
        output_op,
        aligned_accum_fragment[0],
        source_fragment,
        broadcast_fragment);

      //
      // Conditionally store fragments
      //

      if (OutputOp::kStoreZ) {
        destination_iterator.store(frag_Z);
        ++destination_iterator;
      }

      if (OutputOp::kStoreT) {
        tensor_iterator.store(frag_T);
        ++tensor_iterator;
      }
    }
  }

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_(
    typename OutputTileIterator::Fragment &frag_Z,
    typename TensorTileIterator::Fragment &frag_T,
    OutputOp const &output_op,
    typename SharedLoadIterator::Fragment const &frag_AB,
    typename OutputTileIterator::Fragment const &frag_C,
    BroadcastFragment const &frag_Broadcast) {

    using AccessTypeZ = Array<typename OutputTileIterator::Element, kElementsPerAccess>;
    using AccessTypeT = Array<typename TensorTileIterator::Element, kElementsPerAccess>;
    using AccessTypeBroadcast = Array<ElementCompute, kElementsPerAccess>;

    AccessTypeZ *frag_Z_ptr = reinterpret_cast<AccessTypeZ *>(&frag_Z);
    AccessTypeT *frag_T_ptr = reinterpret_cast<AccessTypeT *>(&frag_T);
    
    AccumulatorAccessType const *frag_AB_ptr = 
      reinterpret_cast<AccumulatorAccessType const *>(&frag_AB);

    OutputAccessType const *frag_C_ptr =
      reinterpret_cast<OutputAccessType const *>(&frag_C);

    AccessTypeBroadcast const *frag_Broadcast_ptr =
      reinterpret_cast<AccessTypeBroadcast const *>(&frag_Broadcast);

    int const kOutputOpIterations = 
      OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {
        output_op(
          frag_Z_ptr[i],
          frag_T_ptr[i],
          frag_AB_ptr[i],
          frag_C_ptr[i],
          frag_Broadcast_ptr[i % ThreadMap::Iterations::kColumn]);
    }
  }

  /// Helper to invoke the output functor over each vector of output
  CUTLASS_DEVICE
  void apply_output_operator_source_not_needed_(
    typename OutputTileIterator::Fragment &frag_Z,
    typename TensorTileIterator::Fragment &frag_T,
    OutputOp const &output_op,
    typename SharedLoadIterator::Fragment const &frag_AB,
    BroadcastFragment const &frag_Broadcast) {

    using AccessTypeZ = Array<typename OutputTileIterator::Element, kElementsPerAccess>;
    using AccessTypeT = Array<typename TensorTileIterator::Element, kElementsPerAccess>;
    using AccessTypeBroadcast = Array<ElementCompute, kElementsPerAccess>;

    AccessTypeZ *frag_Z_ptr = reinterpret_cast<AccessTypeZ *>(&frag_Z);
    AccessTypeT *frag_T_ptr = reinterpret_cast<AccessTypeT *>(&frag_T);
    
    AccumulatorAccessType const *frag_AB_ptr = 
      reinterpret_cast<AccumulatorAccessType const *>(&frag_AB);

    AccessTypeBroadcast const *frag_Broadcast_ptr =
      reinterpret_cast<AccessTypeBroadcast const *>(&frag_Broadcast);

    int const kOutputOpIterations = 
      OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOutputOpIterations; ++i) {

      output_op(
        frag_Z_ptr[i], 
        frag_T_ptr[i], 
        frag_AB_ptr[i], 
        frag_Broadcast_ptr[i % ThreadMap::Iterations::kColumn]);
    }
  }


  public:
    /// Stream-K reduce helper
    CUTLASS_DEVICE
    void reduce(
        int reduce_fragment_idx,                        ///< Reduce fragment index
        OutputOp const &output_op,                      ///< Output operator
        ElementVector const * broadcast_ptr,            ///< Broadcast vector
        OutputTileIterator destination_iterator,        ///< Tile iterator for destination
        OutputTileIterator source_iterator,             ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
        TensorTileIterator tensor_iterator,             ///< Threadblock tile iterator for additional tensor operand
        MatrixCoord const &problem_size =               ///< Problem size needed to guard against out-of-bounds accesses
            MatrixCoord(Shape::kM, Shape::kN),
        MatrixCoord const &threadblock_offset =         ///< Threadblock's initial offset within the problem size space
            MatrixCoord()) 
    {

      BroadcastFragment broadcast_fragment;
      load_broadcast_fragment_(broadcast_fragment, broadcast_ptr, problem_size, threadblock_offset);

      // Initialize/load source-fragment data
      typename OutputTileIterator::Fragment source_fragment;
      source_fragment.clear();

      if (output_op.is_source_needed())
      {
        source_iterator += reduce_fragment_idx;
        source_iterator.load(source_fragment);
      }

      // Load fragment from shared memory
      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
      shared_load_iterator_.load(aligned_accum_fragment[0]);

      // Add fragments shared by other k partitions
      if (kPartitionsK > 1)
      {
        plus <typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for ( int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
        }
      }

      //
      // Apply output operation
      //

      typename OutputTileIterator::Fragment frag_Z;
      typename TensorTileIterator::Fragment frag_T;

      if (!output_op.is_source_needed()) {
        apply_output_operator_source_not_needed_(
          frag_Z,
          frag_T,
          output_op,
          aligned_accum_fragment[0],
          broadcast_fragment);
      } else {
        apply_output_operator_(
          frag_Z,
          frag_T,
          output_op,
          aligned_accum_fragment[0],
          source_fragment,
          broadcast_fragment);
      }

      //
      // Conditionally store fragments
      //

      if (OutputOp::kStoreZ) {
        destination_iterator.store(frag_Z);
        ++destination_iterator;
      }

      if (OutputOp::kStoreT) {
        tensor_iterator.store(frag_T);
        ++tensor_iterator;
      }
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
