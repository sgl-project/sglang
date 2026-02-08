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
    \brief Defines basic properties needed by CTA-level GEMMs assuming expectations about data
      layout of the global memory fragments, data types, and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting TensorOp instructions.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/platform/platform.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

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
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::RowMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, 2, Operator_
                      > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

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
  using SmemLayoutA = 
    layout::ColumnMajorTensorOpMultiplicandCongruous<
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
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                             kWarpThreadArrangementStridedA>,
    kAccessSizeInBits / sizeof_bits<ElementA>::value
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                             kWarpThreadArrangementStridedB>,
    kAccessSizeInBits / sizeof_bits<ElementB>::value
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
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
    /// Operation performed by MMA
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, 2, Operator_
                      > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

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
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    0,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    1,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
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
    /// Operation performed by MMA
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajor, ElementB_, layout::RowMajor, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, 2, Operator_
                      > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

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
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    ElementA, 
    SmemLayoutA,
    0,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                             kWarpThreadArrangementStridedB>,
    kAccessSizeInBits / sizeof_bits<ElementB>::value
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    ElementB, 
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
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
    /// Operation performed by MMA
    typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajor, ElementB_, layout::ColumnMajor,
                      ElementC_, LayoutC_, arch::OpClassTensorOp, 2, Operator_
                      > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

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
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
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
/// Below is for arch::OpMultiplyAddFastF16

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
    /// Layout of accumulator
    typename LayoutC_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, float,
                      layout::ColumnMajor, float, layout::RowMajor, float,
                      LayoutC_, arch::OpClassTensorOp, 2,
                      arch::OpMultiplyAddFastF16> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = float;
  using LayoutA = layout::ColumnMajor;
  using ElementB = float;
  using LayoutB = layout::RowMajor;
  using ElementC = float;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 256;

  /// Default Operator
  using Operator = arch::OpMultiplyAdd;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<
      sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>;

  // Shared memory layout
  using SmemLayoutB =
      layout::RowMajorTensorOpMultiplicandCongruous<sizeof_bits<half_t>::value,
                                                    int(128 / sizeof(half_t))>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    layout::PitchLinearShape<8, 4>,
    kAccessSizeInBits / sizeof_bits<ElementA>::value
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    half_t, 
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    layout::PitchLinearShape<8, 4>,
    kAccessSizeInBits / sizeof_bits<ElementB>::value
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    half_t, 
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, half_t, SmemLayoutA, half_t, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
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
    /// Layout of accumulator
    typename LayoutC_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, float,
                      layout::RowMajor, float, layout::ColumnMajor, float,
                      LayoutC_, arch::OpClassTensorOp, 2,
                      arch::OpMultiplyAddFastF16> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = float;
  using LayoutA = layout::RowMajor;
  using ElementB = float;
  using LayoutB = layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 256;

  /// Default Operator
  using Operator = arch::OpMultiplyAdd;

  // Warp thread arrangement 
  static int const kWarpThreadArrangementContiguousA =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  static int const kWarpThreadArrangementContiguousB =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA =
      layout::RowMajorTensorOpMultiplicandCrosswise<sizeof_bits<half_t>::value,
                                                    Shape::kK>;

  // Shared memory layout
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<half_t>::value, Shape::kK>;

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
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    half_t, 
    SmemLayoutA,
    0,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    half_t, 
    SmemLayoutB,
    1,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, half_t, SmemLayoutA, half_t, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
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
    /// Layout of accumulator
    typename LayoutC_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, float,
                      layout::RowMajor, float, layout::RowMajor, float,
                      LayoutC_, arch::OpClassTensorOp, 2,
                      arch::OpMultiplyAddFastF16> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = float;
  using LayoutA = layout::RowMajor;
  using ElementB = float;
  using LayoutB = layout::RowMajor;
  using ElementC = float;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 256;

  /// Default Operator
  using Operator = arch::OpMultiplyAdd;

  // Warp thread arrangement 
  static int const kWarpThreadArrangementContiguousA =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<half_t>::value, Shape::kK>;

  // Shared memory layout
  using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>;

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
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>, 
    half_t,
    SmemLayoutA,
    0,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    layout::PitchLinearShape<8, 4>,
    kAccessSizeInBits / sizeof_bits<ElementB>::value
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>, 
    half_t, 
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, half_t, SmemLayoutA, half_t, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined 
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
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
    /// Layout of accumulator
    typename LayoutC_>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, float,
                      layout::ColumnMajor, float, layout::ColumnMajor, float,
                      LayoutC_, arch::OpClassTensorOp, 2,
                      arch::OpMultiplyAddFastF16> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = float;
  using LayoutA = layout::ColumnMajor;
  using ElementB = float;
  using LayoutB = layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

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
  static int const kAccessSizeInBits = 256;

  /// Default Operator
  using Operator = arch::OpMultiplyAdd; 

  // Warp thread arrangement 
  static int const kWarpThreadArrangementContiguousB =
      Shape::kK / (kAccessSizeInBits / sizeof_bits<ElementA>::value);

  static int const kWarpThreadArrangementStridedB =
      kWarpSize / kWarpThreadArrangementContiguousB;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous<
      sizeof_bits<half_t>::value, int(128 / sizeof(half_t))>;

  // Shared memory layout
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<half_t>::value, Shape::kK>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
      layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
      MatrixShape<Shape::kM, Shape::kK>, half_t, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousB,
                               kWarpThreadArrangementStridedB>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
      MatrixShape<Shape::kK, Shape::kN>, half_t, SmemLayoutB, 1,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, half_t, SmemLayoutA, half_t, SmemLayoutB,
      ElementC, LayoutC, Operator, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
                              WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major-interleave
///   B: row-major-interleave
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
    /// Operation performed by MMA
    typename Operator_,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor,
    /// Number of interleaved k
    int InterleavedK>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajorInterleaved<InterleavedK>, ElementB_,
                      layout::RowMajorInterleaved<InterleavedK>, ElementC_,
                      LayoutC_, arch::OpClassTensorOp, 2, Operator_,
                      AccumulatorsInRowMajor> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
  using ElementB = ElementB_;
  using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;
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
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
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
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
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

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
