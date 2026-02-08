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
#include "cutlass/fast_math.h"
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
  typename Mma_,                                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,                             ///! Epilogue
  typename ThreadblockSwizzle_,                   ///! Threadblock swizzling function
  conv::Operator ConvOperator,                    ///! Convolutional operator (Fprop, Dgrad, Wgrad)
  typename ConvProblemSize_ = Conv2dProblemSize   ///! Convolutional operator on 2D or 3D problem
>
struct ImplicitGemmConvolutionStridedDgrad {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
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
  using InstructionShape = typename ArchMmaOperator::Shape;

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

  static conv::GroupMode const kGroupMode = conv::GroupMode::kNone;

  /// Wgrad C stride idx for implicit gemm algorithm 
  // Conv2d row-major matrix C (KxRSC) 
  // Conv3d row-major matrix C (KxTRSC)
  static int const kWgradCStrideIdx = 
    platform::is_same<LayoutC, cutlass::layout::TensorNHWC>::value ? 2 : 3;

  /// This chooses the appropriate stride element of the C tensor.
  static int const kTensorCStrideIdx = 
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradCStrideIdx : 0);

  // Strided dgrad uses a specialized threadblock swizzle for functionality and performance
  static_assert((platform::is_same<ThreadblockSwizzle,
                      threadblock::StridedDgradHorizontalThreadblockSwizzle>::value) ||
                (platform::is_same<ThreadblockSwizzle,
                      threadblock::StridedDgradIdentityThreadblockSwizzle<1>>::value) ||
                (platform::is_same<ThreadblockSwizzle,
                      threadblock::StridedDgradIdentityThreadblockSwizzle<4>>::value) ||
                (platform::is_same<ThreadblockSwizzle,
                      threadblock::StridedDgradIdentityThreadblockSwizzle<8>>::value),
    "Needs ThreadblockSwizzle type specialized for strided dgrad");

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

    ConvProblemSize problem_size{};
    TensorRefA ref_A{};
    TensorRefB ref_B{};
    TensorRefC ref_C{};
    TensorRefC ref_D{};
    typename EpilogueOutputOp::Params output_op{};
    SplitKMode split_k_mode{};

    //
    // Methods
    //

    /// Default ctor
    Arguments() = default;
   
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
      SplitKMode const & split_k_mode = SplitKMode::kSerial
    ):
      problem_size(problem_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_C(ref_C),
      ref_D(ref_D),
      output_op(output_op),
      split_k_mode(split_k_mode)
    {

    }

  };

  /// Parameters structure
  struct Params {
    ConvProblemSize problem_size{};
    cutlass::gemm::GemmCoord grid_tiled_shape{};
    int swizzle_log_tile{0};
    FastDivmod stride_h_divmod{};
    FastDivmod stride_w_divmod{};
    int gemm_k_iterations{0};
    typename Mma::IteratorA::Params iterator_A{};
    typename Mma::IteratorA::Element const *ptr_A = nullptr;
    typename Mma::IteratorB::Params iterator_B{};
    typename Mma::IteratorB::Element const *ptr_B = nullptr;
    typename Epilogue::OutputTileIterator::Params iterator_C{};
    typename Epilogue::OutputTileIterator::Element *ptr_C = nullptr;
    typename Epilogue::OutputTileIterator::Params iterator_D{};
    typename Epilogue::OutputTileIterator::Element *ptr_D = nullptr;
    typename EpilogueOutputOp::Params output_op {};
    int *semaphore = nullptr;
    SplitKMode split_k_mode {};

    //
    // Methods
    //
    Params() = default;

    /// 
    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore = nullptr
    ):
      problem_size(args.problem_size),
      stride_h_divmod(args.problem_size.stride_h),
      stride_w_divmod(args.problem_size.stride_w),
      iterator_A(Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
      ptr_A(args.ref_A.data()),
      iterator_B(args.problem_size, args.ref_B.layout()),
      ptr_B(args.ref_B.data()),
      iterator_C(ConvOutputIteratorParameter::layout(args.ref_C), args.problem_size, ThreadblockShape::kM),
      ptr_C(args.ref_C.data()),
      iterator_D(ConvOutputIteratorParameter::layout(args.ref_D), args.problem_size, ThreadblockShape::kM),
      ptr_D(args.ref_D.data()),
      output_op(args.output_op),
      semaphore(semaphore),
      split_k_mode(args.split_k_mode)
    {
      gemm_k_iterations = implicit_gemm_k_iterations(kConvolutionalOperator, ThreadblockShape::kK, args.problem_size);

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);
      
      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };
  
  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  ImplicitGemmConvolutionStridedDgrad() { } 

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

    // Compute starting filter position for strided dgrad
    int tile_m_per_filter = strided_dgrad_tile_m_per_filter(params.problem_size, 
                                                            ThreadblockShape::kM);
    int filter_tile_m = (threadblock_tile_idx.m() / tile_m_per_filter);
    

    // The subsequent fast_divmod() operations are equivalent to the following logical computation:
    //
    // int start_r = filter_tile_m / (params.problem_size.stride_w);
    // int start_s = filter_tile_m % (params.problem_size.stride_w);

    int start_r, start_s;
    params.stride_w_divmod(start_r, start_s, filter_tile_m);

    int filter_r = start_r;
    int filter_s = start_s;

    if (params.problem_size.mode == Mode::kConvolution) {
      filter_r = (params.problem_size.R - 1 - filter_r);
      filter_s = (params.problem_size.S - 1 - filter_s);
    }

    // Starting h, w positions for filter position in gemm_k=0
    int start_h, start_w;
    strided_dgrad_starting_coords(
      params.problem_size,
      params.stride_h_divmod, params.stride_w_divmod,
      filter_r, filter_s,
      start_h, start_w);

    if (start_h >= params.problem_size.H || start_w >= params.problem_size.W) {
      return;
    }

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    // Check if CTA contributes valid MMA (Dy * w) and accumulator will be non-zero after MMA
    if (start_r < params.problem_size.R && start_s < params.problem_size.S) {
      // Scale gemm_k_iterations for strided dgrad
      int gemm_k_iterations = (params.gemm_k_iterations / (params.problem_size.R * params.problem_size.S)
                              ) * params.problem_size.num_gemm_k_filter_positions(start_r, start_s);
      
      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        params.iterator_A,
        params.problem_size,
        params.ptr_A,
        thread_idx,
        params.stride_h_divmod, params.stride_w_divmod,
        start_r, start_s,
        MatrixCoord(
          threadblock_tile_idx.m() * Mma::Shape::kM,
          threadblock_tile_idx.k() * Mma::Shape::kK
        ) 
      );
      
      typename Mma::IteratorB iterator_B(
        params.iterator_B,
        params.problem_size,
        params.ptr_B,
        thread_idx,
        start_r, start_s,
        MatrixCoord(
          threadblock_tile_idx.k() * Mma::Shape::kK,
          threadblock_tile_idx.n() * Mma::Shape::kN
        )
      );

      //
      // Main loop
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    }

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

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
      output_op.set_k_partition(threadblock_tile_idx.k(), params.grid_tiled_shape.k());
    }

    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() * Mma::Shape::kM,
      threadblock_tile_idx.n() * Mma::Shape::kN
    );

    // Tile iterator writing to destination tensor
    typename Epilogue::OutputTileIterator iterator_D(
      params.iterator_D,
      params.ptr_D,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      params.stride_h_divmod, params.stride_w_divmod,
      start_r, start_s,
      threadblock_offset
    );

    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    if (output_op.is_source_needed())
    {
      // Tile iterator reading from source accumulator tensor
      typename Epilogue::OutputTileIterator iterator_C(
        params.iterator_C,
        params.ptr_C,
        ConvOutputIteratorParameter::extent(params.problem_size),
        thread_idx,
        params.stride_h_divmod, params.stride_w_divmod,
        start_r, start_s,
        threadblock_offset);

      // Wait on the semaphore - this latency may have been covered by iterator construction
      if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {

        // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
        if (threadblock_tile_idx.k()) {
          iterator_C = iterator_D;
        }

        semaphore.wait(threadblock_tile_idx.k());
      }

      // Run epilogue with addend source iterator
      epilogue(output_op, iterator_D, accumulators, iterator_C);
    }
    else
    {
      // Run epilogue without addend source iterator
      epilogue(output_op, iterator_D, accumulators);
    }

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
