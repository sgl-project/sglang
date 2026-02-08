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
    \brief Template for a multi-staged Depthwise Convolution kernel.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/platform/platform.h"
#include "cutlass/semaphore.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure
template <typename Mma_,                 ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          conv::Operator ConvOperator,   ///! Convolutional operator (Fprop, Dgrad, Wgrad)
          typename Arguments_,           ///! Kernel Arguments
          typename ConvOutputIteratorParameter_, ///! Output Iterator Params
          typename ConvProblemSize_ = Conv2dProblemSize,  ///! Convolutional operator on 2D or 3D problem
          conv::GroupMode GroupMode_ = conv::GroupMode::kNone,  ///! Group mode
          typename ThreadBlockOutputShape_ = cutlass::conv::TensorNHWCShape<1, 1, 1, 1> >  ///! OutputShape per ThreadBlock
struct DirectConvolutionParams {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using ThreadBlockOutputShape = ThreadBlockOutputShape_;
  static Operator const kConvolutionalOperator = ConvOperator;
  using ConvProblemSize = ConvProblemSize_;
  using Arguments = Arguments_;
  using ConvOutputIteratorParameter = ConvOutputIteratorParameter_;

  using ThreadblockShape = typename Mma::Shape;
  static IteratorAlgorithm const kIteratorAlgorithm = Mma::IteratorA::kIteratorAlgorithm;
  static conv::GroupMode const kGroupMode = GroupMode_;
  static int const kStages = Mma::kStages;

  ConvProblemSize problem_size;
  cutlass::gemm::GemmCoord grid_tiled_shape;
  gemm::GemmCoord implicit_gemm_problem_size;
  int swizzle_log_tile;
  int smem_size_;

  int gemm_k_iterations;
  int gemm_k_iterations_per_channel;
  typename Mma::IteratorA::Params iterator_A;
  typename Mma::IteratorA::Element const *ptr_A;
  typename Mma::IteratorB::Params iterator_B;
  typename Mma::IteratorB::Element const *ptr_B;
  typename Mma::IteratorB::Element *ptr_reordered_B;
  typename Epilogue::OutputTileIterator::Params iterator_C;
  typename Epilogue::OutputTileIterator::Element *ptr_C;
  typename Epilogue::OutputTileIterator::Params iterator_D;
  typename Epilogue::OutputTileIterator::Element *ptr_D;
  typename EpilogueOutputOp::Params output_op;
  int *semaphore;
  SplitKMode split_k_mode;
  int split_k_slices;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  DirectConvolutionParams() : swizzle_log_tile(0), gemm_k_iterations(0) {}

  ///
  CUTLASS_HOST_DEVICE
  DirectConvolutionParams(Arguments const &args, int *semaphore = nullptr)
      : problem_size(args.problem_size),
        implicit_gemm_problem_size(
            cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size)),
        iterator_A(Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
        ptr_A(args.ref_A.data()),
        iterator_B(Mma::IteratorB::getParams(args.problem_size, args.ref_B.layout())),
        ptr_B(args.ref_B.data()),
        ptr_reordered_B(args.ref_reordered_B.data()),
        iterator_C(ConvOutputIteratorParameter::layout(args.ref_C), args.problem_size),
        ptr_C(args.ref_C.data()),
        iterator_D(ConvOutputIteratorParameter::layout(args.ref_D), args.problem_size),
        ptr_D(args.ref_D.data()),
        output_op(args.output_op),
        semaphore(semaphore),
        split_k_mode(args.split_k_mode),
        split_k_slices(args.problem_size.split_k_slices) {
    gemm_k_iterations =
        depthwise_gemm_k_iterations<ThreadBlockOutputShape::kN,
                                    ThreadBlockOutputShape::kH,
                                    ThreadBlockOutputShape::kW>(kConvolutionalOperator,
                                                                ThreadblockShape::kK,
                                                                args.problem_size,
                                                                kIteratorAlgorithm,
                                                                kGroupMode,
                                                                ThreadblockShape::kN);

    gemm_k_iterations_per_channel = implicit_gemm_k_iterations_per_channel(
        kConvolutionalOperator, args.problem_size, kIteratorAlgorithm);

    ThreadblockSwizzle threadblock_swizzle;

    grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);

    swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);

    // Dynamic SMEM usage because stride and dilation are runtime params.
    smem_size_ = (cutlass::platform::max(iterator_A.activation_size, int(sizeof(typename Epilogue::SharedStorage))) * kStages + iterator_B.filter_size);
  }

  CUTLASS_HOST_DEVICE
  int get_smem_size() {
    // Dynamic Smem Size
    return smem_size_;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Params_, typename ElementB_>
struct ReorderKernel {
  using Params = Params_;
  using ElementB = ElementB_;

  union SharedStorage {};

  static unsigned int const kReorderKernelThreadPerCTA = 128;

  CUTLASS_HOST_DEVICE
  ReorderKernel() {}

  CUTLASS_HOST_DEVICE
  static dim3 get_grid_shape(Params const &params) {
    return dim3{static_cast<unsigned int>(
                    (params.problem_size.filter_size() + kReorderKernelThreadPerCTA - 1) /
                    kReorderKernelThreadPerCTA),
                1,
                1};
  }

  CUTLASS_HOST_DEVICE
  static dim3 get_block_shape() { return dim3{kReorderKernelThreadPerCTA, 1, 1}; }

  CUTLASS_HOST_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    int64_t m = static_cast<int64_t>(params.problem_size.groups);
    int64_t n = static_cast<int64_t>(params.problem_size.filter_size() / params.problem_size.K);
    const ElementB *src_with_type = static_cast<const ElementB *>(params.ptr_B);
    ElementB *dst_with_type = static_cast<ElementB *>(params.ptr_reordered_B);

    int64_t linear_index = blockIdx.x * kReorderKernelThreadPerCTA + threadIdx.x;
    int64_t index_m = linear_index / n;
    int64_t index_n = linear_index % n;
    int64_t new_linear_index = index_m + index_n * m;

    if (linear_index < m * n) {
      dst_with_type[new_linear_index] = src_with_type[linear_index];
    }
    return;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,                             ///! Epilogue
  typename ThreadblockSwizzle_,                   ///! Threadblock swizzling function
  conv::Operator ConvOperator,                    ///! Convolutional operator (Fprop, Dgrad, Wgrad)
  typename ConvProblemSize_ = Conv2dProblemSize,  ///! Convolutional operator on 2D or 3D problem
  conv::GroupMode GroupMode_ = conv::GroupMode::kNone,    ///! Group mode
  typename ThreadBlockOutputShape_ = cutlass::conv::TensorNHWCShape<1, 1, 1, 1>
>
struct DirectConvolution {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using ThreadBlockOutputShape = ThreadBlockOutputShape_;
  static Operator const kConvolutionalOperator = ConvOperator;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename EpilogueOutputOp::ElementOutput;

  /// Set output tensor C layout
  using LayoutC = LayoutA;

  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using WarpMmaOperator = typename Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;
  
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename WarpMmaOperator::Shape;
  using InstructionShape = typename cutlass::gemm::GemmShape<1, 1, 1>;

  static int const kStages = Mma::kStages;
  static IteratorAlgorithm const kIteratorAlgorithm = Mma::IteratorA::kIteratorAlgorithm; 
  static StrideSupport const kStrideSupport = Mma::IteratorA::kStrideSupport;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

  /// Check iterator A and B convolution dimension are the same and 
  // set device::ImplicitGemmConvolution::kConvDim
  static_assert(Mma::IteratorA::kConvDim == Mma::IteratorB::kConvDim, 
    "Convolution on different different dimensions is not supported");
  static int const kConvDim = Mma::IteratorA::kConvDim;

  /// Conv dimension and problem size structure (Conv2d or Conv3d)
  using ConvProblemSize = ConvProblemSize_;

  static conv::GroupMode const kGroupMode = GroupMode_;


  //
  //
  //
  using ConvOutputIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutC,
    typename Epilogue::OutputTileIterator::Layout, 
    TensorRefC,
    ConvOperator,
    ConvProblemSize
    >;


  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    ConvProblemSize problem_size;
    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefB ref_reordered_B;
    TensorRefC ref_C;
    TensorRefC ref_D;
    typename EpilogueOutputOp::Params output_op;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }
   
    CUTLASS_HOST_DEVICE 
    Arguments(
      ConvProblemSize const & problem_size
    ):
      problem_size(problem_size) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const & problem_size,
      TensorRefA const & ref_A,
      TensorRefB const & ref_B,
      TensorRefC const & ref_C,
      TensorRefC const & ref_D,
      typename EpilogueOutputOp::Params const & output_op,
      TensorRefB const & ref_reordered_B = nullptr,
      SplitKMode const & split_k_mode = SplitKMode::kSerial
    ):
      problem_size(problem_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_C(ref_C),
      ref_D(ref_D),
      output_op(output_op),
      ref_reordered_B(ref_reordered_B),
      split_k_mode(split_k_mode)
    {

    }

  };

  using Params =
      typename cutlass::conv::kernel::DirectConvolutionParams<Mma,
                                                              Epilogue,
                                                              ThreadblockSwizzle,
                                                              kConvolutionalOperator,
                                                              Arguments,
                                                              ConvOutputIteratorParameter,
                                                              ConvProblemSize,
                                                              kGroupMode,
                                                              ThreadBlockOutputShape>;

  using ReorderKernel = typename cutlass::conv::kernel::ReorderKernel<Params, ElementB>;

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  DirectConvolution() { } 

  /// Executes one ImplicitGEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if threadblock is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {

      return;
    }

    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    int iterator_column_offset = 0;
    int filter_row_offset = 0;
    if (kGroupMode != GroupMode::kNone) {
      if (kGroupMode == GroupMode::kDepthwise) {
        iterator_column_offset += threadblock_tile_idx.n() * Mma::Shape::kN;
      }
    } 

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.iterator_A,
      params.problem_size,
      params.ptr_A,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() + threadblock_tile_idx.k(),
        iterator_column_offset
      )
    );
    
    typename Mma::IteratorB iterator_B(
      params.iterator_B,
      params.problem_size,
      params.ptr_reordered_B,
      thread_idx,
      MatrixCoord(
        filter_row_offset,
        iterator_column_offset
      )
    );

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);
    
    // Compute logical position within grid
    threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);


    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() + threadblock_tile_idx.k(),
      threadblock_tile_idx.n() * Mma::Shape::kN
    );

    // Tile iterator writing to destination tensor
    typename Epilogue::OutputTileIterator iterator_D(
      params.iterator_D,
      params.ptr_D,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );
    
    // Tile iterator reading from source accumulator tensor
    typename Epilogue::OutputTileIterator iterator_C(
      params.iterator_C,
      params.ptr_C,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );


    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);


    // Compute threadblock-scoped matrix multiply-add
    // Epilogue is fused in the mainloop
    mma(params.gemm_k_iterations,
        accumulators,
        iterator_A,
        params.iterator_A,
        iterator_B,
        params.iterator_B,
        accumulators,
        epilogue,
        output_op,
        iterator_D,
        iterator_C,
        params.split_k_slices);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
