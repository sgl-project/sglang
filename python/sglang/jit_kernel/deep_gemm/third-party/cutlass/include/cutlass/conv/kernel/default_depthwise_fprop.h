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
    \brief 
    Default kernel-level Depthwise implicit GEMM convolution definitions combine threadblock-scoped 
      matrix multiply-add with the appropriate threadblock-scoped epilogue.  
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"
#include "cutlass/conv/kernel/direct_convolution.h"

#include "cutlass/conv/threadblock/depthwise_mma_core_with_lane_access_size.h"

#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/depthwise_fprop_pipelined.h"

// Direct Conv Related Header files
#include "cutlass/conv/threadblock/depthwise_fprop_activation_tile_access_iterator_direct_conv_optimized.h"
#include "cutlass/conv/threadblock/depthwise_fprop_activation_tile_access_iterator_direct_conv_fixed_stride_dilation.h"

#include "cutlass/conv/threadblock/depthwise_fprop_filter_tile_access_iterator_direct_conv_optimized.h"
#include "cutlass/conv/threadblock/depthwise_fprop_direct_conv_multistage.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for DepthwiseFprop
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename OperatorClass,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kAnalytic,
  conv::StrideSupport StrideSupport = StrideSupport::kUnity,
  /// Access granularity of A matrix in units of elements
  int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value,
  /// Access granularity of B matrix in units of elements
  int AlignmentB = cutlass::sizeof_bits<ElementB>::value / cutlass::sizeof_bits<ElementB>::value
> struct DefaultDepthwiseFprop;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for DepthwiseFprop with direct convolution algorithm
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename OperatorClass,
  typename ArchTag,
  typename ThreadblockShape,
  typename ThreadBlockOutputShape,
  typename FilterShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kAnalytic,
  conv::StrideSupport StrideSupport = StrideSupport::kUnity,
  // MatrixShape<Height, Width>
  typename StrideShape = cutlass::MatrixShape<-1, -1>,
  // MatrixShape< Height, Width> 
  typename DilationShape =  cutlass::MatrixShape<-1, -1>, 
  /// Access granularity of A matrix in units of elements
  int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value,
  /// Access granularity of B matrix in units of elements
  int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value
> struct DefaultDepthwiseDirect2dConvFprop;

/////////////////////////////////////////////////////////////////////////////////////////////////
//                            OpClassSimt convolutions
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Depthwise specialization for Analytic IteratorAlgorithm
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultDepthwiseFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag, //   cutlass::arch::OpMultiplyAdd
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::conv::threadblock::DepthwiseMmaCoreWithLaneAccessSize<
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      ElementA,
      layout::RowMajor,
      ElementB,
      layout::ColumnMajor,
      ElementAccumulator,
      layout::RowMajor,
      arch::OpClassSimt,
      128,
      sizeof_bits<ElementB>::value,
      2,
      MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB,
        cutlass::conv::GroupMode::kDepthwise
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::DepthwiseFpropPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop,
    Conv2dProblemSize,
    cutlass::conv::GroupMode::kDepthwise
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Depthwise specialization for direct 2d conv implementation, 
/// multiple stage pipeline, and SIMT-based mainloop
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename ThreadBlockOutputShape,
  typename FilterShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  typename StrideShape,
  typename DilationShape,
  int AlignmentA,
  int AlignmentB
>
struct DefaultDepthwiseDirect2dConvFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  ThreadBlockOutputShape,
  FilterShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  StrideShape,
  DilationShape,
  AlignmentA,
  AlignmentB
> {
  // One warp handles the entrie groups per cta.
  static_assert(ThreadblockShape::kN == WarpShape::kN,
                "ThreadblockShape::kN should be same as WarpShape::kN ");
  static_assert(ThreadblockShape::kK == FilterShape::kCount && WarpShape::kK == FilterShape::kCount,
                "ThreadblockShape::kK and WarpShape::kK should be same as filter size");
  static_assert(ThreadblockShape::kM % WarpShape::kM == 0,
                "ThreadblockShape::kM must be divisible by WarpShape shape::kM");
  static_assert(ThreadBlockOutputShape::kN, "ThreadBlockOutputShape::kN should be 1");

  // Define the core components from GEMM
  using MmaCore = typename cutlass::conv::threadblock::DepthwiseDirectConvMmaCoreWithLaneAccessSize<
      ThreadblockShape,
      ThreadBlockOutputShape,
      FilterShape,
      WarpShape,
      InstructionShape,
      ElementA,
      layout::RowMajor,
      ElementB,
      layout::ColumnMajor,
      ElementAccumulator,
      layout::RowMajor,
      arch::OpClassSimt,
      128,
      128,
      Stages,
      MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::DepthwiseFpropActivationDirect2dConvTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM,ThreadblockShape::kN>, // < outputShape:KMNK, groups per cta>
      ThreadBlockOutputShape,
      ElementA, LayoutA,
      ThreadMapA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
      cutlass::conv::threadblock::DepthwiseFpropFilterDirectConvTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kN, FilterShape::kCount>,
        ElementB, LayoutB,
        ThreadMapB
      >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;
  using ThreadOutputShape = typename MmaCore::ThreadOutputShape;
  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultDirectConvEpilogueSimt<
    ThreadblockShape, // < outputShape:KMNK, groups per cta>
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    ThreadOutputShape,
    ThreadBlockOutputShape
  >::Epilogue;

  // Define the Mma
  using Mma = threadblock::DepthwiseFpropDirectConvMultipleStage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    Epilogue
  >;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::DirectConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop,
    Conv2dProblemSize,
    cutlass::conv::GroupMode::kDepthwise,
    ThreadBlockOutputShape
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Depthwise specialization for direct 2d conv implementation, 
/// multiple stage pipeline, and SIMT-based mainloop
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename ThreadBlockOutputShape,
  typename FilterShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  typename StrideShape,
  typename DilationShape,
  int AlignmentA,
  int AlignmentB
>
struct DefaultDepthwiseDirect2dConvFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  ThreadBlockOutputShape,
  FilterShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kFixedStrideDilation,
  StrideSupport,
  StrideShape,
  DilationShape,
  AlignmentA,
  AlignmentB
> {



  // One warp handles the entrie groups per cta.
  static_assert(ThreadblockShape::kN == WarpShape::kN,
                "ThreadblockShape::kN should be same as WarpShape::kN ");
  static_assert(ThreadblockShape::kK == FilterShape::kCount && WarpShape::kK == FilterShape::kCount,
                "ThreadblockShape::kK and WarpShape::kK should be same as filter size");
  static_assert(ThreadblockShape::kM % WarpShape::kM == 0,
                "ThreadblockShape::kM must be divisible by WarpShape shape::kM");
  static_assert(ThreadBlockOutputShape::kN, "ThreadBlockOutputShape::kN should be 1");

  static_assert(StrideShape::kRow >= 0 && StrideShape::kColumn >= 0, "Stride should be fixed");
  static_assert(DilationShape::kRow >= 0 && DilationShape::kColumn >= 0, "Stride should be fixed");

  // Activations loaded by threadblock
  static int const ActivationShapeH = (ThreadBlockOutputShape::kH - 1) * StrideShape::kRow +
                             (FilterShape::kRow - 1) * DilationShape::kRow + 1;

  static int const ActivationShapeW = (ThreadBlockOutputShape::kW - 1) * StrideShape::kColumn +
                             (FilterShape::kColumn - 1) * DilationShape::kColumn + 1;

  using ActivationShape =
      cutlass::conv::TensorNHWCShape<1, ActivationShapeH, ActivationShapeW, ThreadblockShape::kN >;

  // Define the core components from GEMM
  using MmaCore = typename cutlass::conv::threadblock::DepthwiseDirectConvMmaCoreWithLaneAccessSize<
      ThreadblockShape,
      ThreadBlockOutputShape,
      FilterShape,
      WarpShape,
      InstructionShape,
      ElementA,
      layout::RowMajor,
      ElementB,
      layout::ColumnMajor,
      ElementAccumulator,
      layout::RowMajor,
      arch::OpClassSimt,
      128,
      128,
      Stages,
      MathOperatorTag,
      IteratorAlgorithm::kFixedStrideDilation,
      StrideShape,
      DilationShape,
      ActivationShape>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::DepthwiseFpropActivationDirect2dConvTileAccessIteratorFixedStrideDilation<
      cutlass::MatrixShape<ThreadblockShape::kM,ThreadblockShape::kN>, // < outputShape:KMNK, groups per cta>
      ThreadBlockOutputShape,
      StrideShape,
      DilationShape,
      ActivationShape,
      ElementA, LayoutA,
      ThreadMapA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
      cutlass::conv::threadblock::DepthwiseFpropFilterDirectConvTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kN, FilterShape::kCount>,
        ElementB, LayoutB,
        ThreadMapB
      >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;
  using ThreadOutputShape = typename MmaCore::ThreadOutputShape;
  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * AlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultDirectConvEpilogueSimt<
    ThreadblockShape, // < outputShape:KMNK, groups per cta>
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    ThreadOutputShape,
    ThreadBlockOutputShape
  >::Epilogue;

  // Define the Mma
  using Mma = threadblock::DepthwiseFpropDirectConvMultipleStage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    CacheOpA,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages,
    Epilogue,
    IteratorAlgorithm::kFixedStrideDilation
  >;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::DirectConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop,
    Conv2dProblemSize,
    cutlass::conv::GroupMode::kDepthwise,
    ThreadBlockOutputShape
  >;
};

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
