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
   Default kernel-level fused activation's scale+bias+relu and implicit GEMM convolution
   definitions that combine threadblock-scoped matrix multiply-add with the
   appropriate threadblock-scoped epilogue.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"

#include "cutlass/conv/threadblock/conv3d_fprop_activation_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv3d_fprop_filter_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv3d_fprop_activation_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv3d_fprop_filter_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/predicated_scale_bias_vector_access_iterator.h"
#include "cutlass/transform/threadblock/regular_scale_bias_vector_access_iterator.h"
#include "cutlass/gemm/warp/scale_bias_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for fused batch norm and Conv3dFprop
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementScaleBias,
  typename LayoutScaleBias,
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
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kOptimized,
  conv::StrideSupport StrideSupport = StrideSupport::kUnity
> struct DefaultConv3dFpropFusion;

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         OpClassTensorOp convolutions 
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv3dFprop specialzation for Analytic IteratorAlgorithm and multistage 
/// pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementScaleBias,
  typename LayoutScaleBias,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag
>
struct DefaultConv3dFpropFusion <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementScaleBias,
  LayoutScaleBias,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv3dFpropActivationTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      ThreadMapA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv3dFpropFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      ThreadMapB
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  /// Define iterators over tiles from scale/bias vectors
  using IteratorScaleBias =
      cutlass::conv::threadblock::PredicatedScaleBiasVectorAccessIterator<
          cutlass::MatrixShape<1, ThreadblockShape::kK>, ElementScaleBias,
          LayoutScaleBias>;

  using SmemIteratorScaleBias =
      cutlass::transform::threadblock::RegularScaleBiasVectorAccessIterator<
          cutlass::MatrixShape<1, ThreadblockShape::kK>, ElementScaleBias,
          LayoutScaleBias>;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static int const kThreadCount = 32;

  // Warp-level iterators to load scale and bias vectors
  using WarpIteratorScaleBias = cutlass::gemm::warp::ScaleBiasTileIterator<
      MatrixShape<WarpShape::kM, WarpShape::kK>, ElementScaleBias,
      LayoutScaleBias, MatrixShape<InstructionShape::kM, InstructionShape::kK>,
      typename WarpMmaTensorOp::IteratorA::Base::Policy, kThreadCount,
      MmaCore::WarpCount::kK>;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmFpropFusionMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    arch::CacheOperation::Global,
    IteratorScaleBias,
    SmemIteratorScaleBias,
    arch::CacheOperation::Always,
    MmaPolicy,
    WarpIteratorScaleBias,
    Stages 
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionFusion<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop,
    Conv3dProblemSize
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv3dFprop specialzation for Optimized IteratorAlgorithm and 
/// multistage pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementScaleBias,
  typename LayoutScaleBias,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag
>
struct DefaultConv3dFpropFusion <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementScaleBias,
  LayoutScaleBias,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
    ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
    Stages, MathOperatorTag
  >;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv3dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      LayoutA,
      ThreadMapA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand 
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv3dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      LayoutB,
      ThreadMapB
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  /// Define iterators over tiles from scale/bias vectors
  using IteratorScaleBias =
      cutlass::conv::threadblock::PredicatedScaleBiasVectorAccessIterator<
          cutlass::MatrixShape<1, ThreadblockShape::kK>, ElementScaleBias,
          LayoutScaleBias>;

  using SmemIteratorScaleBias =
      cutlass::transform::threadblock::RegularScaleBiasVectorAccessIterator<
          cutlass::MatrixShape<1, ThreadblockShape::kK>, ElementScaleBias,
          LayoutScaleBias>;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static int const kThreadCount = 32;

  // Warp-level iterators to load scale and bias vectors
  using WarpIteratorScaleBias = cutlass::gemm::warp::ScaleBiasTileIterator<
      MatrixShape<WarpShape::kM, WarpShape::kK>, ElementScaleBias,
      LayoutScaleBias, MatrixShape<InstructionShape::kM, InstructionShape::kK>,
      typename WarpMmaTensorOp::IteratorA::Base::Policy, kThreadCount,
      MmaCore::WarpCount::kK>;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmFpropFusionMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    arch::CacheOperation::Global,
    IteratorScaleBias,
    SmemIteratorScaleBias,
    arch::CacheOperation::Always,
    MmaPolicy,
    WarpIteratorScaleBias,
    Stages 
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionFusion<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop,
    Conv3dProblemSize
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
