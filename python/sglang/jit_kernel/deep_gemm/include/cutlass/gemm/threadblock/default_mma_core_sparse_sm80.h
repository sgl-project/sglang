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
    \brief Defines basic properties needed by CTA-level GEMMs assuming
   expectations about data layout of the global memory fragments, data types,
   and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting sparse
   TensorOp instructions.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"

#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/default_mma_sparse_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"

#include "cutlass/gemm/threadblock/default_mma_core.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "cutlass/gemm/threadblock/mma_sparse_multistage.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Template defininng default matrix multiply operators inferred from threadblock tile size,
/// global memory data layout, and target math instruction.
template <
    /// Shape of threadblock-scoped matrix multiply operator
    typename Shape,
    /// Shape of warp-level matrix multiply operator
    typename WarpShape,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape,
    /// Element data type of A operand
    typename ElementA,
    /// Layout of operand A
    typename LayoutA,
    /// Element data type of B operand
    typename ElementB,
    /// Layout of operand B
    typename LayoutB,
    /// Data type of accumulator
    typename ElementC,
    /// Layout of accumulator
    typename LayoutC,
    /// Indicates type of math operator (arch::OpClassSimt or arch::OpClassTensorOp)
    typename OperatorClass,
    /// Number of stages
    int Stages,
    /// Operation performed by MMA
    typename Operator = typename platform::conditional<
        (platform::is_same<OperatorClass,
                           cutlass::arch::OpClassTensorOp>::value) &&
            (platform::is_same<ElementA, int8_t>::value ||
             platform::is_same<ElementA, int4b_t>::value ||
             platform::is_same<ElementA, uint8_t>::value ||
             platform::is_same<ElementA, uint4b_t>::value),
        cutlass::arch::OpMultiplyAddSaturate,
        cutlass::arch::OpMultiplyAdd>::type,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false
    /// Cache operation of operand A
    , cutlass::arch::CacheOperation::Kind CacheOpA =
        cutlass::arch::CacheOperation::Global,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB =
        cutlass::arch::CacheOperation::Global
>
struct DefaultSparseMmaCore;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major
///   B: row-major
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Operation performed by MMA
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSparseMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::RowMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                      Operator_, false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  static int const kSparse = 2;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;
 
  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kWarpThreadArrangementContiguousA =
      platform::min(Shape::kM / (kAccessSizeInBits / sizeof_bits<ElementA>::value), 8);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  static int const kWarpThreadArrangementContiguousB =
      platform::min(Shape::kN / (kAccessSizeInBits / sizeof_bits<ElementB>::value), 8);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //
  static int const Crosswise_A = platform::min(int(128 / sizeof(ElementA)),
                                               Shape::kM);

  using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<
      sizeof_bits<ElementA>::value, Crosswise_A>;

  // Shared memory layout
  static int const Crosswise_B = platform::min(int(128 / sizeof(ElementB)),
                                               Shape::kN);

  using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<ElementB>::value, Crosswise_B>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kM, Shape::kK / kSparse>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK / kSparse>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Cache operation of operand E
  static cutlass::arch::CacheOperation::Kind const kCacheOpE =
      cutlass::arch::CacheOperation::Global;

  static int const kInterleavedE = MmaTensorOp::kInterleaved;
  static int const kMetaSizeInBits = MmaTensorOp::kMetaSizeInBits;
  static int const kMaxID2 = MmaTensorOp::kMaxID2;
  static int const kElementsPerElementE = MmaTensorOp::kElementsPerElementE;

  using ElementE = typename MmaTensorOp::ElementE;
  using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<kInterleavedE>;

  // Shared memory layout.  Interleaved layout is mapped to PitchLinear layout.
  using SmemLayoutE = typename MmaTensorOp::LayoutE;

  /// ThreadMap of iterator E
  static int const kElementsPerAccessE =
      kAccessSizeInBits / sizeof_bits<ElementE>::value;

  /// E is tiny.  Not all warps are needed.
  static int const kThreadsE =
      (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
           (kAccessSizeInBits / sizeof_bits<ElementE>::value) >
       kThreads)
          ? kThreads
          : (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
             (kAccessSizeInBits / sizeof_bits<ElementE>::value));

  using IteratorThreadMapE = transform::PitchLinearStripminedThreadMap<
      layout::PitchLinearShape<Shape::kM * kInterleavedE,
                               Shape::kK / kSparse / kElementsPerElementE /
                                   kInterleavedE>,
      kThreadsE, kElementsPerAccessE>;

  /// Shared memory iterator to E operand
  using SmemIteratorE = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM * kInterleavedE,
                  Shape::kK / kSparse / kElementsPerElementE / kInterleavedE>,
      ElementE, SmemLayoutE, 0, IteratorThreadMapE>;

  /// Policy used to define MmaPipelined
  using MmaPolicy =
      SparseMmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                      MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: column-major
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Operation performed by MMA
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSparseMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                      Operator_, false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  static int const kSparse = 2;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kWarpThreadArrangementContiguousA =
      Shape::kK / kSparse / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  // crosswise cannot be larger than 1024 bit.
  static int const kCrosswiseB =
      (Shape::kK > (1024 / sizeof_bits<ElementB>::value))
          ? (1024 / sizeof_bits<ElementB>::value)
          : Shape::kK;

  static int const kWarpThreadArrangementContiguousB =
      kCrosswiseB / (kAccessSizeInBits / sizeof_bits<ElementB>::value);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kK / kSparse>;

  // Shared memory layout
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, kCrosswiseB>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK / kSparse, Shape::kM>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK / kSparse>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Cache operation of operand E
  static cutlass::arch::CacheOperation::Kind const kCacheOpE =
      cutlass::arch::CacheOperation::Global;

  static int const kInterleavedE = MmaTensorOp::kInterleaved;
  static int const kMetaSizeInBits = MmaTensorOp::kMetaSizeInBits;
  static int const kMaxID2 = MmaTensorOp::kMaxID2;
  static int const kElementsPerElementE = MmaTensorOp::kElementsPerElementE;

  using ElementE = typename MmaTensorOp::ElementE;
  using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<kInterleavedE>;

  // Shared memory layout.  Interleaved layout is mapped to PitchLinear layout.
  using SmemLayoutE = typename MmaTensorOp::LayoutE;

  /// ThreadMap of iterator E
  static int const kElementsPerAccessE =
      kAccessSizeInBits / sizeof_bits<ElementE>::value;

  /// E is tiny.  Not all warps are needed.
  static int const kThreadsE =
      (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
           (kAccessSizeInBits / sizeof_bits<ElementE>::value) >
       kThreads)
          ? kThreads
          : (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
             (kAccessSizeInBits / sizeof_bits<ElementE>::value));

  using IteratorThreadMapE = transform::PitchLinearStripminedThreadMap<
      layout::PitchLinearShape<Shape::kM * kInterleavedE,
                               Shape::kK / kSparse / kElementsPerElementE /
                                   kInterleavedE>,
      kThreadsE, kElementsPerAccessE>;


  /// Shared memory iterator to E operand
  using SmemIteratorE = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM * kInterleavedE,
                  Shape::kK / kSparse / kElementsPerElementE / kInterleavedE>,
      ElementE, SmemLayoutE, 0, IteratorThreadMapE>;

  /// Policy used to define MmaPipelined
  using MmaPolicy =
      SparseMmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                      MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major
///   B: column-major
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Operation performed by MMA
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSparseMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, Stages,
                      Operator_, false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;

  using LayoutA = layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;

  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  static int const kSparse = 2;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const Crosswise_A = platform::min(int(128 / sizeof(ElementA)),
                                               Shape::kM);

  static int const kWarpThreadArrangementContiguousA =
      platform::min(Shape::kM / (kAccessSizeInBits / sizeof_bits<ElementA>::value), 8);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  // Warp thread arrangement
  // crosswise cannot be larger than 1024 bit.
  static int const kCrosswiseB =
      (Shape::kK > (1024 / sizeof_bits<ElementB>::value))
          ? (1024 / sizeof_bits<ElementB>::value)
          : Shape::kK;

  static int const kWarpThreadArrangementContiguousB =
      kCrosswiseB / (kAccessSizeInBits / sizeof_bits<ElementB>::value);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<
      sizeof_bits<ElementA>::value, Crosswise_A>;

  // Shared memory layout
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, kCrosswiseB>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kM, Shape::kK / kSparse>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK / kSparse>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Cache operation of operand E
  static cutlass::arch::CacheOperation::Kind const kCacheOpE =
      cutlass::arch::CacheOperation::Global;

  static int const kInterleavedE = MmaTensorOp::kInterleaved;
  static int const kMetaSizeInBits = MmaTensorOp::kMetaSizeInBits;
  static int const kMaxID2 = MmaTensorOp::kMaxID2;
  static int const kElementsPerElementE = MmaTensorOp::kElementsPerElementE;

  using ElementE = typename MmaTensorOp::ElementE;
  using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<kInterleavedE>;

  // Shared memory layout.  Interleaved layout is mapped to PitchLinear layout.
  using SmemLayoutE = typename MmaTensorOp::LayoutE;

  /// ThreadMap of iterator E
  static int const kElementsPerAccessE =
      kAccessSizeInBits / sizeof_bits<ElementE>::value;

  /// E is tiny.  Not all warps are needed.
  static int const kThreadsE =
      (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
           (kAccessSizeInBits / sizeof_bits<ElementE>::value) >
       kThreads)
          ? kThreads
          : (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
             (kAccessSizeInBits / sizeof_bits<ElementE>::value));

  using IteratorThreadMapE = transform::PitchLinearStripminedThreadMap<
      layout::PitchLinearShape<Shape::kM * kInterleavedE,
                               Shape::kK / kSparse / kElementsPerElementE /
                                   kInterleavedE>,
      kThreadsE, kElementsPerAccessE>;

  /// Shared memory iterator to E operand
  using SmemIteratorE = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM * kInterleavedE,
                  Shape::kK / kSparse / kElementsPerElementE / kInterleavedE>,
      ElementE, SmemLayoutE, 0, IteratorThreadMapE>;

  /// Policy used to define MmaPipelined
  using MmaPolicy =
      SparseMmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                      MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: row-major
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Operation performed by MMA
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSparseMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  static int const kSparse = 2;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kWarpThreadArrangementContiguousA =
      Shape::kK / kSparse / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  static int const kWarpThreadArrangementContiguousB =
      platform::min(Shape::kN / (kAccessSizeInBits / sizeof_bits<ElementB>::value), 8);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  static int const Crosswise_B = platform::min(int(128 / sizeof(ElementB)),
                                               Shape::kN);


  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kK / kSparse>;

  // Shared memory layout
  using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<ElementB>::value, Crosswise_B>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK / kSparse, Shape::kM>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK / kSparse>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultSparseMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Cache operation of operand E
  static cutlass::arch::CacheOperation::Kind const kCacheOpE =
      cutlass::arch::CacheOperation::Global;

  static int const kInterleavedE = MmaTensorOp::kInterleaved;
  static int const kMetaSizeInBits = MmaTensorOp::kMetaSizeInBits;
  static int const kMaxID2 = MmaTensorOp::kMaxID2;
  static int const kElementsPerElementE = MmaTensorOp::kElementsPerElementE;

  using ElementE = typename MmaTensorOp::ElementE;
  using GmemLayoutE = cutlass::layout::ColumnMajorInterleaved<kInterleavedE>;

  // Shared memory layout.  Interleaved layout is mapped to PitchLinear layout.
  using SmemLayoutE = typename MmaTensorOp::LayoutE;

  /// ThreadMap of iterator E
  static int const kElementsPerAccessE =
      kAccessSizeInBits / sizeof_bits<ElementE>::value;

  /// E is tiny.  Not all warps are needed.
  static int const kThreadsE =
      (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
           (kAccessSizeInBits / sizeof_bits<ElementE>::value) >
       kThreads)
          ? kThreads
          : (Shape::kM * Shape::kK / kSparse / kElementsPerElementE /
             (kAccessSizeInBits / sizeof_bits<ElementE>::value));

  using IteratorThreadMapE = transform::PitchLinearStripminedThreadMap<
      layout::PitchLinearShape<Shape::kM * kInterleavedE,
                               Shape::kK / kSparse / kElementsPerElementE /
                                   kInterleavedE>,
      kThreadsE, kElementsPerAccessE>;

  /// Shared memory iterator to E operand
  using SmemIteratorE = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM * kInterleavedE,
                  Shape::kK / kSparse / kElementsPerElementE / kInterleavedE>,
      ElementE, SmemLayoutE, 0, IteratorThreadMapE>;

  /// Policy used to define MmaPipelined
  using MmaPolicy =
      SparseMmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                      MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
