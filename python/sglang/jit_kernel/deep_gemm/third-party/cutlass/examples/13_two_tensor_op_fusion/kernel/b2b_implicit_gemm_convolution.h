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
    \brief Template for a pipelined Implicit GEMM kernel.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
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

template <
  typename B2bMma_,                               ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,                             ///! Epilogue
  typename ThreadblockSwizzle_,                   ///! Threadblock swizzling function
  conv::Operator ConvOperator,                    ///! Convolutional operator (Fprop, Dgrad, Wgrad)
  typename ConvProblemSize_ = Conv2dProblemSize   ///! Convolutional operator on 2D or 3D problem
>
struct B2bImplicitGemmConvolution {

  using B2bMma = B2bMma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp0 = typename B2bMma::OutputOp;
  using EpilogueOutputOp1 = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static Operator const kConvolutionalOperator = ConvOperator;

  using ElementA = typename B2bMma::IteratorA0::Element;
  using LayoutA = typename B2bMma::IteratorA0::Layout;
  using ElementB = typename B2bMma::IteratorB0::Element;
  using LayoutB = typename B2bMma::IteratorB0::Layout;
  using ElementC = typename EpilogueOutputOp1::ElementOutput;

  /// Set output tensor C layout
  using LayoutC = LayoutA;

  using ElementAccumulator = typename EpilogueOutputOp0::ElementAccumulator;
  using ElementCompute = typename EpilogueOutputOp0::ElementCompute;

  /// Scale and Bias
  using ElementScaleBias = typename B2bMma::IteratorAccumulatorScaleBias::Element;
  using LayoutScaleBias = typename B2bMma::IteratorAccumulatorScaleBias::Layout;

  using WarpMmaOperator0 = typename B2bMma::Policy0::Operator;
  using WarpMmaOperator1 = typename B2bMma::Policy1::Operator;

  using ArchMmaOperator = typename WarpMmaOperator0::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;
  
  using OperatorClass = typename WarpMmaOperator0::OperatorClass;
  using ArchTag = typename WarpMmaOperator0::ArchTag;

  using ThreadblockShape0 = typename B2bMma::Shape0;
  using ThreadblockShape1 = typename B2bMma::Shape1;
  using WarpShape0 = typename WarpMmaOperator0::Shape;
  using WarpShape1 = typename WarpMmaOperator1::Shape;
  using InstructionShape = typename ArchMmaOperator::Shape;

  static int const kStages = B2bMma::kStages;
  static IteratorAlgorithm const kIteratorAlgorithm = B2bMma::IteratorA0::kIteratorAlgorithm; 
 
  /// Warp count (concept: GemmShape)
  using WarpCount0 = typename B2bMma::WarpCount0;
  static int const kThreadCount = 32 * WarpCount0::kCount;

  using TensorRefA0 = typename B2bMma::IteratorA0::TensorRef;
  using TensorRefB0 = typename B2bMma::IteratorB0::TensorRef;
  using TensorRefScaleBias0 = typename B2bMma::IteratorAccumulatorScaleBias::TensorRef;
  using TensorRefB1 = typename B2bMma::IteratorB1::TensorRef;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

  /// Check iterator A and B convolution dimension are the same and 
  // set device::B2bImplicitGemmConvolution::kConvDim
  static_assert(B2bMma::IteratorA0::kConvDim == B2bMma::IteratorB0::kConvDim, 
    "Convolution on different dimensions is not supported");
  static int const kConvDim = B2bMma::IteratorA0::kConvDim;

  /// Conv dimension and problem size structure (Conv2d or Conv3d)
  using ConvProblemSize = ConvProblemSize_;

  /// Wgrad C stride idx for implicit gemm algorithm 
  // Conv2d row-major matrix C (KxRSC) 
  // Conv3d row-major matrix C (KxTRSC)
  static int const kWgradCStrideIdx = 
    cutlass::platform::is_same<LayoutC, cutlass::layout::TensorNHWC>::value ? 2 : 3;

  /// This chooses the appropriate stride element of the C tensor.
  static int const kTensorCStrideIdx = 
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradCStrideIdx : 0);

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

    ConvProblemSize problem_size_0;
    ConvProblemSize problem_size_1;
    TensorRefA0 ref_A0;
    TensorRefB0 ref_B0;
    TensorRefC ref_C0;
    TensorRefScaleBias0 ref_Scale0;
    TensorRefScaleBias0 ref_Bias0;
    TensorRefB1 ref_B1;
    TensorRefC ref_C1;
    TensorRefC ref_D1;
    typename EpilogueOutputOp0::Params output_op_0;
    typename EpilogueOutputOp1::Params output_op_1;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }
   
    CUTLASS_HOST_DEVICE 
    Arguments(
      ConvProblemSize const & problem_size_0,
      ConvProblemSize const & problem_size_1
    ):
      problem_size_0(problem_size_0),
      problem_size_1(problem_size_1) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const & problem_size_0,
      ConvProblemSize const & problem_size_1,
      TensorRefA0 const & ref_A0,
      TensorRefB0 const & ref_B0,
      TensorRefC const & ref_C0,
      TensorRefScaleBias0 const & ref_Scale0,
      TensorRefScaleBias0 const & ref_Bias0,
      TensorRefB1 const & ref_B1,
      TensorRefC const & ref_C1,
      TensorRefC const & ref_D1,
      typename EpilogueOutputOp0::Params const & output_op_0,
      typename EpilogueOutputOp1::Params const & output_op_1,
      SplitKMode const & split_k_mode = SplitKMode::kSerial
    ):
      problem_size_0(problem_size_0),
      problem_size_1(problem_size_1),
      ref_A0(ref_A0),
      ref_B0(ref_B0),
      ref_C0(ref_C0),
      ref_Scale0(ref_Scale0),
      ref_Bias0(ref_Bias0),
      ref_B1(ref_B1),
      ref_C1(ref_C1),
      ref_D1(ref_D1),
      output_op_0(output_op_0),
      output_op_1(output_op_1),
      split_k_mode(split_k_mode)
    {

    }

  };

  /// Parameters structure
  struct Params {
    ConvProblemSize problem_size_0;
    ConvProblemSize problem_size_1;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    gemm::GemmCoord implicit_gemm_problem_size_0;
    gemm::GemmCoord implicit_gemm_problem_size_1;
    int swizzle_log_tile;
    int gemm_k_iterations_0;
    int gemm_k_iterations_1;
    typename B2bMma::IteratorA0::Params iterator_A0;
    typename B2bMma::IteratorA0::Element const *ptr_A0;
    typename B2bMma::IteratorB0::Params iterator_B0;
    typename B2bMma::IteratorB0::Element const *ptr_B0;
    typename Epilogue::OutputTileIterator::Params iterator_C0;
    typename Epilogue::OutputTileIterator::Element *ptr_C0;
    typename B2bMma::IteratorAccumulatorScaleBias::Element *ptr_Scale0;
    typename B2bMma::IteratorAccumulatorScaleBias::Element *ptr_Bias0;
    typename B2bMma::IteratorB1::Params iterator_B1;
    typename B2bMma::IteratorB1::Element const *ptr_B1;
    typename Epilogue::OutputTileIterator::Params iterator_C1;
    typename Epilogue::OutputTileIterator::Element *ptr_C1;
    typename Epilogue::OutputTileIterator::Params iterator_D1;
    typename Epilogue::OutputTileIterator::Element *ptr_D1;
    typename EpilogueOutputOp0::Params output_op_0;
    typename EpilogueOutputOp1::Params output_op_1;
    int *semaphore;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), gemm_k_iterations_0(0), gemm_k_iterations_1(0) { }

    /// 
    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore = nullptr
    ):
      problem_size_0(args.problem_size_0),
      problem_size_1(args.problem_size_1),
      implicit_gemm_problem_size_0(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size_0)),
      implicit_gemm_problem_size_1(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size_1)),
      iterator_A0(B2bMma::IteratorA0::getParams(args.problem_size_0, args.ref_A0.layout())),
      ptr_A0(args.ref_A0.data()),
      iterator_B0(args.problem_size_0, args.ref_B0.layout()),
      ptr_B0(args.ref_B0.data()),
      iterator_C0(ConvOutputIteratorParameter::layout(args.ref_C0)),
      ptr_C0(args.ref_C0.data()),
      ptr_Scale0(args.ref_Scale0.data()),
      ptr_Bias0(args.ref_Bias0.data()),
      iterator_B1(args.problem_size_1, args.ref_B1.layout()),
      ptr_B1(args.ref_B1.data()),
      iterator_C1(ConvOutputIteratorParameter::layout(args.ref_C1)),
      ptr_C1(args.ref_C1.data()),
      iterator_D1(ConvOutputIteratorParameter::layout(args.ref_D1)),
      ptr_D1(args.ref_D1.data()),
      output_op_0(args.output_op_0),
      output_op_1(args.output_op_1),
      semaphore(semaphore),
      split_k_mode(args.split_k_mode)
    {
      gemm_k_iterations_0 = implicit_gemm_k_iterations(kConvolutionalOperator, ThreadblockShape0::kK, args.problem_size_0);
      gemm_k_iterations_1 = implicit_gemm_k_iterations(kConvolutionalOperator, ThreadblockShape1::kK, args.problem_size_1);

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        implicit_gemm_problem_size_0,
        {ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK},
        args.problem_size_0.split_k_slices);

      swizzle_log_tile = ThreadblockSwizzle().get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename B2bMma::B2bMmaSharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  B2bImplicitGemmConvolution() { } 

  /// Executes one ImplicitGEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {

      return;
    }

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename B2bMma::IteratorA0 iterator_A0(
      params.iterator_A0,
      params.problem_size_0,
      params.ptr_A0,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * B2bMma::Shape0::kM,
        threadblock_tile_idx.k() * B2bMma::Shape0::kK
      )
    );
    
    typename B2bMma::IteratorB0 iterator_B0(
      params.iterator_B0,
      params.problem_size_0,
      params.ptr_B0,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * B2bMma::Shape0::kK,
        threadblock_tile_idx.n() * B2bMma::Shape0::kN
      )
    );

    typename B2bMma::IteratorB1 iterator_B1(
      params.iterator_B1,
      params.problem_size_1,
      params.ptr_B1,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * B2bMma::Shape1::kK,
        threadblock_tile_idx.n() * B2bMma::Shape1::kN
      )
    );


    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    // Construct iterators to accumulator scale/bias vector
    typename B2bMma::IteratorAccumulatorScaleBias iterator_Scale0(
      params.ptr_Scale0,
      {1, params.problem_size_0.K},
      thread_idx,
      warp_idx,
      MatrixCoord(
        0, threadblock_tile_idx.n() * B2bMma::Shape0::kN
      )
    );

    typename B2bMma::IteratorAccumulatorScaleBias iterator_Bias0(
      params.ptr_Bias0,
      {1, params.problem_size_0.K},
      thread_idx,
      warp_idx,
      MatrixCoord(
        0, threadblock_tile_idx.n() * B2bMma::Shape0::kN
      )
    );


    //
    // Main loop
    //

    EpilogueOutputOp0 output_op_0(params.output_op_0);

    // Construct thread-scoped matrix multiply
    B2bMma b2bMma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename B2bMma::FragmentC0 src_accum;
    typename B2bMma::FragmentC1 accumulators;

    src_accum.clear();
    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    b2bMma(params.gemm_k_iterations_0, accumulators, iterator_A0, iterator_B0, 
        iterator_Scale0, iterator_Bias0, iterator_B1, src_accum, output_op_0);

    //
    // Epilogue
    //

    EpilogueOutputOp1 output_op_1(params.output_op_1);

    // Construct the semaphore.
    int block_idx = threadblock_tile_idx.m() + threadblock_tile_idx.n() * params.grid_tiled_shape.m();

    Semaphore semaphore(params.semaphore + block_idx, thread_idx);
    
    // Compute logical position within grid
    threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {
        
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op_1.set_k_partition(threadblock_tile_idx.k(), params.grid_tiled_shape.k());
    }

    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() * B2bMma::Shape1::kM,
      threadblock_tile_idx.n() * B2bMma::Shape1::kN
    );

    // Tile iterator writing to destination tensor
    typename Epilogue::OutputTileIterator iterator_D1(
      params.iterator_D1,
      params.ptr_D1,
      ConvOutputIteratorParameter::extent(params.problem_size_1),
      thread_idx,
      threadblock_offset
    );
    
    // Tile iterator reading from source accumulator tensor
    typename Epilogue::OutputTileIterator iterator_C1(
      params.iterator_C1,
      params.ptr_C1,
      ConvOutputIteratorParameter::extent(params.problem_size_1),
      thread_idx,
      threadblock_offset
    );


    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_idx.k()) {
        iterator_C1 = iterator_D1;
      }

      semaphore.wait(threadblock_tile_idx.k());

      __threadfence();
    }
    // Each split-k-slice writes to a unique tensor location
    else if (params.split_k_mode == SplitKMode::kParallel) {
      iterator_D1.add_pointer_offset(threadblock_tile_idx.k() * 
        cutlass::conv::implicit_gemm_tensor_c_size(ConvOperator, params.problem_size_1));
    }

    // Run efficient epilogue
    epilogue(output_op_1, iterator_D1, accumulators, iterator_C1);
  
    //
    // Release the semaphore
    //

    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) { 

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_idx.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_idx.k() + 1;
      }
      
      semaphore.release(lock);
    }
  } 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

