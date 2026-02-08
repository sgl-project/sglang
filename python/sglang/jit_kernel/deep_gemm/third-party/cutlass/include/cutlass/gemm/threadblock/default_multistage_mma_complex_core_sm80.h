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
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"

#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/gemm/warp/default_mma_complex_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

#include "cutlass/gemm/threadblock/default_multistage_mma_complex_core.h"

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

/// Partial specialization for complex double-precision
///
///   A: column-major
///   B: row-major
///   Operator: arch::OpMultiplyAddComplex or arch::OpMultiplyGaussianComplex
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
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, InstructionShape_, 
    complex<double>, layout::ColumnMajor,
    complex<double>, layout::RowMajor,
    complex<double>, LayoutC_, 
    arch::OpClassTensorOp,
    Stages,
    TransformA, TransformB,
    Operator_,
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = complex<double>;
  using LayoutA = layout::ColumnMajor;
  using ElementB = complex<double>;
  using LayoutB = layout::RowMajor;
  using ElementC = complex<double>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of a threadblock-scoped 128
  static int const kAccessSizeInBits = 128;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous128b;

  using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous128b;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
      layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
      layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
      WarpShape, InstructionShape, 
      ElementA, SmemLayoutA, 
      ElementB, SmemLayoutB,
      ElementC, LayoutC, 
      kTransformA, kTransformB,
      Operator>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};


/// Partial specialization for complex double-precision
///
///   A: column-major
///   B: row-major
///   Operator: arch::OpMultiplyAddComplex or arch::OpMultiplyGaussianComplex
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
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, InstructionShape_, 
    complex<double>, layout::ColumnMajor,
    complex<double>, layout::ColumnMajor,
    complex<double>, LayoutC_, 
    arch::OpClassTensorOp,
    Stages, 
    TransformA, TransformB,
    Operator_, 
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = complex<double>;
  using LayoutA = layout::ColumnMajor;
  using ElementB = complex<double>;
  using LayoutB = layout::ColumnMajor;
  using ElementC = complex<double>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  using Operator = Operator_;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
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

  /// Size of a threadblock-scoped 128
  static int const kAccessSizeInBits = 128;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous128b;
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise128x4;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kM, Shape::kK>, kThreads,
      layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
      WarpShape, InstructionShape, 
      ElementA, SmemLayoutA, 
      ElementB, SmemLayoutB,
      ElementC, LayoutC, 
      kTransformA, kTransformB,
      Operator>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex double-precision
///
///   A: row-major
///   B: column-major
///   Operator: arch::OpMultiplyAddComplex or arch::OpMultiplyGaussianComplex
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
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, InstructionShape_, 
    complex<double>, layout::RowMajor,
    complex<double>, layout::ColumnMajor,
    complex<double>, LayoutC_, 
    arch::OpClassTensorOp,
    Stages,
    TransformA, TransformB,
    Operator_, 
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = complex<double>;
  using LayoutA = layout::RowMajor;
  using ElementB = complex<double>;
  using LayoutB = layout::ColumnMajor;
  using ElementC = complex<double>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of a threadblock-scoped 128
  static int const kAccessSizeInBits = 128;


  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise128x4;
  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise128x4;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
      WarpShape, InstructionShape, 
      ElementA, SmemLayoutA, 
      ElementB, SmemLayoutB,
      ElementC, LayoutC, 
      kTransformA, kTransformB,
      Operator>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};


/// Partial specialization for complex double-precision
///
///   A: row-major
///   B: row-major
///   Operator: arch::OpMultiplyAddComplex or arch::OpMultiplyGaussianComplex
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
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator_,    
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, InstructionShape_, 
    complex<double>, layout::RowMajor,
    complex<double>, layout::RowMajor,
    complex<double>, LayoutC_, 
    arch::OpClassTensorOp,
    Stages, 
    TransformA, TransformB, 
    Operator_,
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = complex<double>;
  using LayoutA = layout::RowMajor;
  using ElementB = complex<double>;
  using LayoutB = layout::RowMajor;
  using ElementC = complex<double>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of a threadblock-scoped 128
  static int const kAccessSizeInBits = 128;


  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise128x4;
  using SmemLayoutB = layout::RowMajorTensorOpMultiplicandCongruous128b;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
      layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
      WarpShape, InstructionShape, 
      ElementA, SmemLayoutA, 
      ElementB, SmemLayoutB,
      ElementC, LayoutC, 
      kTransformA, kTransformB,
      Operator>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex floating-point
///
///   A: column-major
///   B: column-major
///   Operator: arch::OpMultiplyAddComplex
///   Math Instruction: mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, GemmShape<16, 8, 8>, 
    complex<float>, layout::ColumnMajor,
    complex<float>, layout::ColumnMajor,
    complex<float>, LayoutC_, 
    arch::OpClassTensorOp,
    Stages,
    TransformA, TransformB,
    Operator_,
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<16, 8, 8>;
  using ElementA = complex<float>;
  using LayoutA = layout::ColumnMajor;
  using ElementB = complex<float>;
  using LayoutB = layout::ColumnMajor;
  using ElementC = complex<float>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of a threadblock-scoped
  static int const kAccessSizeInBits = 64;

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
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
      WarpShape, InstructionShape, 
      ElementA, SmemLayoutA, 
      ElementB, SmemLayoutB,
      ElementC, LayoutC, 
      kTransformA, kTransformB,
      Operator>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};


/// Partial specialization for complex floating-point
///
///   A: column-major
///   B: row-major
///   Operator: arch::OpMultiplyAddComplex
///   Math Instruction: mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, GemmShape<16, 8, 8>, 
    complex<float>, layout::ColumnMajor,
    complex<float>, layout::RowMajor,
    complex<float>, LayoutC_, 
    arch::OpClassTensorOp,
    Stages,
    TransformA, TransformB,
    Operator_,
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<16, 8, 8>;
  using ElementA = complex<float>;
  using LayoutA = layout::ColumnMajor;
  using ElementB = complex<float>;
  using LayoutB = layout::RowMajor;
  using ElementC = complex<float>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of a threadblock-scoped
  static int const kAccessSizeInBits = 64;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajorTensorOpMultiplicandCongruous64b;

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
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
      WarpShape, InstructionShape, 
      ElementA, SmemLayoutA, 
      ElementB, SmemLayoutB,
      ElementC, LayoutC, 
      kTransformA, kTransformB,
      Operator>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex floating-point
///
///   A: row-major
///   B: column-major
///   Operator: arch::OpMultiplyAddComplex
///   Math Instruction: mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, GemmShape<16, 8, 8>, 
    complex<float>, layout::RowMajor,
    complex<float>, layout::ColumnMajor,
    complex<float>, LayoutC_, 
    arch::OpClassTensorOp,
    Stages,
    TransformA, TransformB,
    Operator_,
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<16, 8, 8>;
  using ElementA = complex<float>;
  using LayoutA = layout::RowMajor;
  using ElementB = complex<float>;
  using LayoutB = layout::ColumnMajor;
  using ElementC = complex<float>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of a threadblock-scoped
  static int const kAccessSizeInBits = 64;

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
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
      WarpShape, InstructionShape, 
      ElementA, SmemLayoutA, 
      ElementB, SmemLayoutB,
      ElementC, LayoutC, 
      kTransformA, kTransformB,
      Operator>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex floating-point
///
///   A: row-major
///   B: row-major
///   Operator: arch::OpMultiplyAddComplex
///   Math Instruction: mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, GemmShape<16, 8, 8>, 
    complex<float>, layout::RowMajor,
    complex<float>, layout::RowMajor,
    complex<float>, LayoutC_, 
    arch::OpClassTensorOp,
    Stages,
    TransformA, TransformB,
    Operator_,
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<16, 8, 8>;
  using ElementA = complex<float>;
  using LayoutA = layout::RowMajor;
  using ElementB = complex<float>;
  using LayoutB = layout::RowMajor;
  using ElementC = complex<float>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of a threadblock-scoped
  static int const kAccessSizeInBits = 64;

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
  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<
      WarpShape, InstructionShape, 
      ElementA, SmemLayoutA, 
      ElementB, SmemLayoutB,
      ElementC, LayoutC, 
      kTransformA, kTransformB,
      Operator>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                                        MatrixShape<0, 0>, WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for complex SIMT operation
///
///   A: column-major
///   B: column-major
///   Operator: arch::OpMultiplyAddComplex or arch::OpMultiplyGaussianComplex
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    typename RealA,
    typename RealB,
    typename RealC,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, GemmShape<1, 1, 1>, 
    complex<RealA>, layout::ColumnMajor,
    complex<RealB>, layout::ColumnMajor,
    complex<RealC>, LayoutC_, 
    arch::OpClassSimt,
    Stages,
    TransformA, TransformB,
    Operator_,
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = complex<RealA>;
  using LayoutA = layout::ColumnMajor;
  using ElementB = complex<RealB>;
  using LayoutB = layout::ColumnMajor;
  using ElementC = complex<RealC>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of access
  static int const kAccessSizeInBits = sizeof_bits<ElementA>::value;

  /// No vectorized accesses
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;

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
    WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,     /// Data type of A elements
    SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
    ElementB,     /// Data type of B elements
    SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
    ElementC,     /// Element type of C matrix
    LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
    Policy,       /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    1,            /// 1 partition along K dimension
    kTransformA,  /// Transform for A
    kTransformB   /// Transform for B
    >;            /// Used for partial specialization

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, Shape::kK / 32>,
    WarpCount::kK>;
};

/// Partial specialization for complex SIMT operation
///
///   A: column-major
///   B: row-major
///   Operator: arch::OpMultiplyAddComplex or arch::OpMultiplyGaussianComplex
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    typename RealA,
    typename RealB,
    typename RealC,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, GemmShape<1, 1, 1>, 
    complex<RealA>, layout::ColumnMajor,
    complex<RealB>, layout::RowMajor,
    complex<RealC>, LayoutC_, 
    arch::OpClassSimt,
    Stages,
    TransformA, TransformB,
    Operator_,
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = complex<RealA>;
  using LayoutA = layout::ColumnMajor;
  using ElementB = complex<RealB>;
  using LayoutB = layout::RowMajor;
  using ElementC = complex<RealC>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of access
  static int const kAccessSizeInBits = sizeof_bits<ElementA>::value;

  /// No vectorized accesses
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;

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
    WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,     /// Data type of A elements
    SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
    ElementB,     /// Data type of B elements
    SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
    ElementC,     /// Element type of C matrix
    LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
    Policy,       /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    1,            /// 1 partition along K dimension
    kTransformA,  /// Transform for A
    kTransformB   /// Transform for B
    >;            /// Used for partial specialization

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,    // or Shape::kK / 32
    WarpCount::kK>;
};

/// Partial specialization for complex SIMT operation
///
///   A: row-major
///   B: column-major
///   Operator: arch::OpMultiplyAddComplex or arch::OpMultiplyGaussianComplex
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    typename RealA,
    typename RealB,
    typename RealC,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, GemmShape<1, 1, 1>, 
    complex<RealA>, layout::RowMajor,
    complex<RealB>, layout::ColumnMajor,
    complex<RealC>, LayoutC_, 
    arch::OpClassSimt,
    Stages,
    TransformA, TransformB,
    Operator_,
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = complex<RealA>;
  using LayoutA = layout::RowMajor;
  using ElementB = complex<RealB>;
  using LayoutB = layout::ColumnMajor;
  using ElementC = complex<RealC>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of access
  static int const kAccessSizeInBits = sizeof_bits<ElementA>::value;

  /// No vectorized accesses
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;

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
    WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,     /// Data type of A elements
    SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
    ElementB,     /// Data type of B elements
    SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
    ElementC,     /// Element type of C matrix
    LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
    Policy,       /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    1,            /// 1 partition along K dimension
    kTransformA,  /// Transform for A
    kTransformB   /// Transform for B
    >;            /// Used for partial specialization

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<Shape::kK / 32, 0>,
    MatrixShape<0, Shape::kK / 32>,
    WarpCount::kK>;
};

/// Partial specialization for complex SIMT operation
///
///   A: row-major
///   B: row-major
///   Operator: arch::OpMultiplyAddComplex or arch::OpMultiplyGaussianComplex
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    typename RealA,
    typename RealB,
    typename RealC,
    /// Layout of accumulator
    typename LayoutC_,
    /// Number of stages
    int Stages,
    /// Complex transformation on operand A
    ComplexTransform TransformA,
    /// Complex transformation on operand B
    ComplexTransform TransformB,
    /// Multiply-add operator (arch::OpMultiplyAddComplex, arch::OpMultiplyGaussianComplex)
    typename Operator_,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMultistageMmaComplexCore<
    Shape_, WarpShape_, GemmShape<1, 1, 1>, 
    complex<RealA>, layout::RowMajor,
    complex<RealB>, layout::RowMajor,
    complex<RealC>, LayoutC_, 
    arch::OpClassSimt,
    Stages,
    TransformA, TransformB,
    Operator_,
    CacheOpA, CacheOpB> {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<1, 1, 1>;
  using ElementA = complex<RealA>;
  using LayoutA = layout::RowMajor;
  using ElementB = complex<RealB>;
  using LayoutB = layout::RowMajor;
  using ElementC = complex<RealC>;
  using LayoutC = LayoutC_;
  static int const kStages = Stages;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  using Operator = Operator_;
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

  /// Size of access
  static int const kAccessSizeInBits = sizeof_bits<ElementA>::value;

  /// No vectorized accesses
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = layout::ColumnMajor;

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
    WarpShape,    /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,     /// Data type of A elements
    SmemLayoutA,  /// Layout of A matrix (concept: MatrixLayout)
    ElementB,     /// Data type of B elements
    SmemLayoutB,  /// Layout of B matrix (concept: MatrixLayout)
    ElementC,     /// Element type of C matrix
    LayoutC,      /// Layout of C matrix (concept: MatrixLayout)
    Policy,       /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    1,            /// 1 partition along K dimension
    kTransformA,  /// Transform for A
    kTransformB   /// Transform for B
    >;            /// Used for partial specialization

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaWarpSimt,
    MatrixShape<Shape::kK / 32, 0>,
    MatrixShape<0, 0>,    // or Shape::kK / 32
    WarpCount::kK>;
};

////////////////////////////////////////////////////////////////////////////////


}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
