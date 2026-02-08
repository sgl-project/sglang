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

      Partial specializations for threadblock::Mma operations targeting TensorOp
   instructions.

      SM80 Multi stage kernel expects stage number to be larger or equal to 3
   to use asynchronous copy.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"

#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core.h"
#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for double-precision
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::ColumnMajor, double, layout::ColumnMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = double;
  using LayoutA = layout::ColumnMajor;
  using ElementB = double;
  using LayoutB = layout::ColumnMajor;
  using ElementC = double;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>; 

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  static_assert(WarpCount::kCount > 1,
    "This specialization requires at least two warps.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 64;

  /// Default Operator
  using Operator = Operator_;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous64b;

  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicand64bCrosswise;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpStripedThreadMap<
      layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
      layout::PitchLinearShape<16, 2>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<16, 2>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

/// Partial specialization for double-precision
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::ColumnMajor, double, layout::RowMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = double;
  using LayoutA = layout::ColumnMajor;
  using ElementB = double;
  using LayoutB = layout::RowMajor;
  using ElementC = double;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>; 

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  static_assert(WarpCount::kCount > 1,
    "This specialization requires at least two warps.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 64;

  /// Default Operator
  using Operator = Operator_;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous64b;

  // Shared memory layout
  using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous64b;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpStripedThreadMap<
      layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
      layout::PitchLinearShape<16, 2>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpStripedThreadMap<
      layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
      layout::PitchLinearShape<16, 2>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for double-precision
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::RowMajor, double, layout::ColumnMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = double;
  using LayoutA = layout::RowMajor;
  using ElementB = double;
  using LayoutB = layout::ColumnMajor;
  using ElementC = double;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

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
  static int const kAccessSizeInBits = 64;

  /// Default Operator
  using Operator = Operator_;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicand64bCrosswise;

  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicand64bCrosswise;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<16, 2>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<16, 2>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////
///
/// Partial specialization for double-precision
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::RowMajor, double, layout::RowMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = double;
  using LayoutA = layout::RowMajor;
  using ElementB = double;
  using LayoutB = layout::RowMajor;
  using ElementC = double;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  static_assert(WarpCount::kCount > 1,
    "This specialization requires at least two warps.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 64;

  /// Default Operator
  using Operator = Operator_;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicand64bCrosswise;

  using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous64b;


  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<16, 2>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpStripedThreadMap<
      layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
      layout::PitchLinearShape<16, 2>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for double-precision
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::AffineRank2ColumnMajor, double, layout::AffineRank2ColumnMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = double;
  using LayoutA = layout::AffineRank2ColumnMajor;
  using ElementB = double;
  using LayoutB = layout::AffineRank2ColumnMajor;
  using ElementC = double;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::ColumnMajor,
                              ElementB,
                              layout::ColumnMajor,
                              ElementC,
                              LayoutC,
                              arch::OpClassTensorOp,
                              kStages,
                              Operator,
                              false,
                              kCacheOpA,
                              kCacheOpB>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;
};

/// Partial specialization for double-precision
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::AffineRank2ColumnMajor, double, layout::AffineRank2RowMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = double;
  using LayoutA = layout::AffineRank2ColumnMajor;
  using ElementB = double;
  using LayoutB = layout::AffineRank2RowMajor;
  using ElementC = double;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::ColumnMajor,
                              ElementB,
                              layout::RowMajor,
                              ElementC,
                              LayoutC,
                              arch::OpClassTensorOp,
                              kStages,
                              Operator,
                              false,
                              kCacheOpA,
                              kCacheOpB>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for double-precision
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::AffineRank2RowMajor, double, layout::AffineRank2ColumnMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = double;
  using LayoutA = layout::AffineRank2RowMajor;
  using ElementB = double;
  using LayoutB = layout::AffineRank2ColumnMajor;
  using ElementC = double;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::RowMajor,
                              ElementB,
                              layout::ColumnMajor,
                              ElementC,
                              LayoutC,
                              arch::OpClassTensorOp,
                              kStages,
                              Operator,
                              false,
                              kCacheOpA,
                              kCacheOpB>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;
};

////////////////////////////////////////////////////////////////////////////////
///
/// Partial specialization for double-precision
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, double,
                      layout::AffineRank2RowMajor, double, layout::AffineRank2RowMajor, double,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = double;
  using LayoutA = layout::AffineRank2RowMajor;
  using ElementB = double;
  using LayoutB = layout::AffineRank2RowMajor;
  using ElementC = double;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::RowMajor,
                              ElementB,
                              layout::RowMajor,
                              ElementC,
                              LayoutC,
                              arch::OpClassTensorOp,
                              kStages,
                              Operator,
                              false,
                              kCacheOpA,
                              kCacheOpB>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float-precision
///
///   ElementA: complex<float>
///   ElementB: complex<float>
///   ElementC: complex<float>
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Layout for A operand
    typename LayoutA_,
    /// Layout for B operand
    typename LayoutB_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Operation performed by MMA
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// per-element transformation for elements of A
    ComplexTransform TransformA_,
    /// per-element transformation for elements of B
    ComplexTransform TransformB_
    >
struct DefaultMmaCore<
  Shape_, WarpShape_, GemmShape<16, 8, 8>, 
  complex<float>, LayoutA_, 
  complex<float>, LayoutB_, 
  complex<float>, LayoutC_, 
  arch::OpClassTensorOp, 
  Stages, 
  Operator_, 
  false, 
  CacheOpA, 
  CacheOpB,
  TransformA_, TransformB_, true> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<16, 8, 8>;
  using ElementA = complex<float>;
  using LayoutA = LayoutA_;
  using ElementB = complex<float>;
  using LayoutB = LayoutB_;
  using ElementC = complex<float>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;
  static const ComplexTransform TransformA = TransformA_;
  static const ComplexTransform TransformB = TransformB_;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>; 

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  static_assert(WarpCount::kCount > 1,
    "This specialization requires at least two warps.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Default Operator
  using Operator = Operator_;

  static_assert(
    platform::is_same<Operator, arch::OpMultiplyAddComplex>::value ||
    platform::is_same<Operator, arch::OpMultiplyAddGaussianComplex>::value ||
    platform::is_same<Operator, arch::OpMultiplyAddComplexFastF32>::value,
    "The operator tag must indicate complex multiplication.");

  //
  // Underlying template
  //

  using MmaComplexCore = DefaultMultistageMmaComplexCore<
    Shape, WarpShape, InstructionShape,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    arch::OpClassTensorOp,
    kStages, 
    TransformA,
    TransformB,
    Operator,
    kCacheOpA,
    kCacheOpB
  >;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename MmaComplexCore::SmemLayoutA;

  // Shared memory layout
  using SmemLayoutB = typename MmaComplexCore::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename MmaComplexCore::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename MmaComplexCore::SmemIteratorA;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = typename MmaComplexCore::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename MmaComplexCore::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename MmaComplexCore::MmaTensorOp;

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename MmaComplexCore::MmaPolicy;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for double-precision
///
///   ElementA: complex<double>
///   ElementB: complex<double>
///   ElementC: complex<double>
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
    /// Layout for A operand
    typename LayoutA_,
    /// Layout for B operand
    typename LayoutB_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Operation performed by MMA
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// per-element transformation for elements of A
    ComplexTransform TransformA_,
    /// per-element transformation for elements of B
    ComplexTransform TransformB_
    >
struct DefaultMmaCore<
  Shape_, WarpShape_, InstructionShape_, 
  complex<double>, LayoutA_, 
  complex<double>, LayoutB_, 
  complex<double>, LayoutC_, 
  arch::OpClassTensorOp, 
  Stages, 
  Operator_, 
  false, 
  CacheOpA, 
  CacheOpB,
  TransformA_, TransformB_, true> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = complex<double>;
  using LayoutA = LayoutA_;
  using ElementB = complex<double>;
  using LayoutB = LayoutB_;
  using ElementC = complex<double>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;
  static const ComplexTransform TransformA = TransformA_;
  static const ComplexTransform TransformB = TransformB_;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>; 

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  static_assert(WarpCount::kCount > 1,
    "This specialization requires at least two warps.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 64;

  /// Default Operator
  using Operator = Operator_;

  static_assert(
    platform::is_same<Operator, arch::OpMultiplyAddComplex>::value ||
    platform::is_same<Operator, arch::OpMultiplyAddGaussianComplex>::value,
    "The operator tag must indicate complex multiplication.");

  //
  // Underlying template
  //

  using MmaComplexCore = DefaultMultistageMmaComplexCore<
    Shape, WarpShape, InstructionShape,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    arch::OpClassTensorOp,
    kStages, 
    TransformA,
    TransformB,
    Operator,
    kCacheOpA,
    kCacheOpB
  >;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename MmaComplexCore::SmemLayoutA;

  // Shared memory layout
  using SmemLayoutB = typename MmaComplexCore::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename MmaComplexCore::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename MmaComplexCore::SmemIteratorA;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = typename MmaComplexCore::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename MmaComplexCore::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename MmaComplexCore::MmaTensorOp;

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename MmaComplexCore::MmaPolicy;
};

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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
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
      layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
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
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
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
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  static int const kWarpThreadArrangementContiguousB =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementB>::value);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kK>;

  // Shared memory layout
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, Shape::kK>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
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
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
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
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

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
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, Shape::kK>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
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
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
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
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
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
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  static int const kWarpThreadArrangementContiguousB =
      platform::min(Shape::kN / (kAccessSizeInBits / sizeof_bits<ElementB>::value), 8);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kK>;

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
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                               kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
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
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major-interleaved
///   B: row-major-interleaved
///   Operator: tensor op class
///
/// This uses the default warp-level operator given tile sizes
///
/// Column/RowMajorInterleved<InterleavedK>(m, n) is mapped to Column/RowMajor(m
/// x InterleavedK, n / InterleavedK) so that Column/RowMajor global iterators
/// can be reused. The shared store iterator is the same as the crosswise shared
/// store iterator. So, the only thing we need to do is to swap the coordinates
/// (contiguous <=> strided) used by the global iterator and the shared store
/// iterator.
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
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Number of interleaved K
    int InterleavedK>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajorInterleaved<InterleavedK>, ElementB_,
                      layout::RowMajorInterleaved<InterleavedK>, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, Stages, Operator_,
                      AccumulatorsInRowMajor, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
  using ElementB = ElementB_;
  using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;
  static int const kInterleavedK = InterleavedK;

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
  static int const kElementsPerAccess =
      kAccessSizeInBits / sizeof_bits<ElementA>::value;

  static int const kWarpThreadArrangementContiguous =
      kInterleavedK / kElementsPerAccess;

  static int const kWarpThreadArrangementStrided =
      kWarpSize / kWarpThreadArrangementContiguous;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, kInterleavedK>;

  // Shared memory layout
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, kInterleavedK>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kM * kInterleavedK,
                               Shape::kK / kInterleavedK>,
      kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = transform::TransposePitchLinearThreadMap<
      IteratorThreadMapA,
      layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                               kWarpThreadArrangementStrided>>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      SmemThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kN * kInterleavedK,
                               Shape::kK / kInterleavedK>,
      kThreads, layout::PitchLinearShape<32, 1>, kElementsPerAccess>;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapB = transform::TransposePitchLinearThreadMap<
      IteratorThreadMapB,
      layout::PitchLinearShape<kWarpThreadArrangementContiguous,
                               kWarpThreadArrangementStrided>>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      SmemThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK, AccumulatorsInRowMajor>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
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
    /// Operation performed by Simt
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
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
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;

  // Shared memory layout
  using SmemLayoutB = layout::RowMajor;

  //
  // Iterators to write to shared memory
  //


  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator B 
  using SmemThreadMapB = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      SmemThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = 4;
  static const int WarpNumThreadsN = 8;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static_assert(!((Shape::kK / 32) % LaneN),
                "Padding must be divisible by Lane");

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape, /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,  /// Data type of A elements
    SmemLayoutA,   /// Layout of A matrix (concept: MatrixLayout)
    ElementB,  /// Data type of B elements
    SmemLayoutB,   /// Layout of B matrix (concept: MatrixLayout)
    ElementC,  /// Element type of C matrix
    LayoutC,   /// Layout of C matrix (concept: MatrixLayout)
    Policy     /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    >;         /// Used for partial specialization

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, Shape::kK / 32>,
    WarpCount::kK>;
};

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
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
    /// Operation performed by Simt
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::RowMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
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
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;

  // Shared memory layout
  using SmemLayoutB = layout::RowMajor;

  //
  // Iterators to write to shared memory
  //


  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = 4;
  static const int WarpNumThreadsN = 8;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape, /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,  /// Data type of A elements
    SmemLayoutA,   /// Layout of A matrix (concept: MatrixLayout)
    ElementB,  /// Data type of B elements
    SmemLayoutB,   /// Layout of B matrix (concept: MatrixLayout)
    ElementC,  /// Element type of C matrix
    LayoutC,   /// Layout of C matrix (concept: MatrixLayout)
    Policy     /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    >;         /// Used for partial specialization

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK>;
};

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
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
    /// Operation performed by Simt
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
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
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;

  // Shared memory layout
  using SmemLayoutB = layout::RowMajor;

  //
  // Iterators to write to shared memory
  //


  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      SmemThreadMapA>;

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator B 
  using SmemThreadMapB = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      SmemThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = 4;
  static const int WarpNumThreadsN = 8;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static_assert(!((Shape::kK / 32) % LaneM) && !((Shape::kK / 32) % LaneN),
                "Padding must be divisible by Lane");

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape, /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,  /// Data type of A elements
    SmemLayoutA,   /// Layout of A matrix (concept: MatrixLayout)
    ElementB,  /// Data type of B elements
    SmemLayoutB,   /// Layout of B matrix (concept: MatrixLayout)
    ElementC,  /// Element type of C matrix
    LayoutC,   /// Layout of C matrix (concept: MatrixLayout)
    Policy     /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    >;         /// Used for partial specialization

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<Shape::kK / 32, 0>,
    MatrixShape<0, Shape::kK / 32>,
    WarpCount::kK>;
};

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
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
    /// Operation performed by Simt
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, Stages, Operator_,
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
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                              Shape::kN / WarpShape::kN, 
                              Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Default Operator
  using Operator = Operator_;

  // Warp thread arrangement
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;

  // Shared memory layout
  using SmemLayoutB = layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      SmemThreadMapA>;

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = 4;
  static const int WarpNumThreadsN = 8;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static_assert(!((Shape::kK / 32) % LaneM),
                "Padding must be divisible by Lane");

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cutlass::gemm::warp::MmaSimt<
    WarpShape, /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,  /// Data type of A elements
    SmemLayoutA,   /// Layout of A matrix (concept: MatrixLayout)
    ElementB,  /// Data type of B elements
    SmemLayoutB,   /// Layout of B matrix (concept: MatrixLayout)
    ElementC,  /// Element type of C matrix
    LayoutC,   /// Layout of C matrix (concept: MatrixLayout)
    Policy     /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    >;         /// Used for partial specialization

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<Shape::kK / 32, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK>;
};

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
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
    /// Operation performed by Simt
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::AffineRank2ColumnMajor, ElementB_, layout::AffineRank2RowMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::AffineRank2ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::AffineRank2RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::ColumnMajor,
                              ElementB,
                              layout::RowMajor,
                              ElementC,
                              LayoutC,
                              arch::OpClassSimt,
                              kStages,
                              Operator,
                              false,
                              kCacheOpA,
                              kCacheOpB>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;
};

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
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
    /// Operation performed by Simt
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::AffineRank2RowMajor, ElementB_, layout::AffineRank2ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::AffineRank2RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::AffineRank2ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::RowMajor,
                              ElementB,
                              layout::ColumnMajor,
                              ElementC,
                              LayoutC,
                              arch::OpClassSimt,
                              kStages,
                              Operator,
                              false,
                              kCacheOpA,
                              kCacheOpB>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;
};

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
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
    /// Operation performed by Simt
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::AffineRank2ColumnMajor, ElementB_, layout::AffineRank2ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassSimt, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::AffineRank2ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::AffineRank2ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::ColumnMajor,
                              ElementB,
                              layout::ColumnMajor,
                              ElementC,
                              LayoutC,
                              arch::OpClassSimt,
                              kStages,
                              Operator,
                              false,
                              kCacheOpA,
                              kCacheOpB>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;

};

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
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
    /// Operation performed by Simt
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::AffineRank2RowMajor, ElementB_, layout::AffineRank2RowMajor, ElementC_,
                      LayoutC_, arch::OpClassSimt, Stages, Operator_,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::AffineRank2RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::AffineRank2RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Default Operator
  using Operator = Operator_;

  using Base = DefaultMmaCore<Shape,
                              WarpShape,
                              InstructionShape,
                              ElementA,
                              layout::RowMajor,
                              ElementB,
                              layout::RowMajor,
                              ElementC,
                              LayoutC,
                              arch::OpClassSimt,
                              kStages,
                              Operator,
                              false,
                              kCacheOpA,
                              kCacheOpB>;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = typename Base::IteratorThreadMapA;

  /// Shared memory iterator to A operand
  using SmemIteratorA = typename Base::SmemIteratorA;

  /// Policy of iterator B
  using IteratorThreadMapB = typename Base::IteratorThreadMapB;

  /// Shared memory iterator to B operand
  using SmemIteratorB = typename Base::SmemIteratorB;

  //
  // Warp-level matrix multiply operator
  //

  /// Policy used to define MmaPipelined
  using MmaPolicy = typename Base::MmaPolicy;

};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
