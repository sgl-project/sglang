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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include "../threadblock/dual_mma_multistage.h"
#include "../threadblock/dual_epilogue.h"
#include "../dual_gemm_common.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename DualMma_,               ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue0_,             ///! Epilogue
  typename Epilogue1_,             ///! Epilogue
  typename OutputOp2_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial,              ///! If true, code supporting split-K via serial reduction is enabled.
  bool StoreD0,
  bool StoreD1
>
struct DualGemm {

  using DualMma = DualMma_;

  using Epilogue0 = Epilogue0_;
  using Epilogue1 = Epilogue1_;
  using OutputOp0 = typename Epilogue0::OutputOp;
  using OutputOp1 = typename Epilogue1::OutputOp;
  using OutputOp2 = OutputOp2_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static constexpr bool kStoreD0 = StoreD0;
  static constexpr bool kStoreD1 = StoreD1;

  using DualEpilogue = cutlass::epilogue::threadblock::DualEpilogue<
      typename Epilogue0::Shape,
      typename Epilogue0::WarpMmaOperator,
      Epilogue0::kPartitionsK,
      typename Epilogue0::OutputTileIterator,
      typename Epilogue0::AccumulatorFragmentIterator,
      typename Epilogue0::WarpTileIterator,
      typename Epilogue0::SharedLoadIterator,
      OutputOp0,
      OutputOp1,
      OutputOp2,
      typename Epilogue0::Padding,
      kStoreD0,
      kStoreD1,
      Epilogue0::kFragmentsPerIteration,
      true // IterationsUnroll
  >;

  using ElementA = typename DualMma::IteratorA::Element;
  using ElementB = typename DualMma::IteratorB0::Element;
  using ElementC = typename DualEpilogue::OutputTileIterator::Element;

  static bool const kSplitKSerial = SplitKSerial;
  static_assert(!kSplitKSerial || (kStoreD0 && kStoreD1),
    "Split-K serial requires buffers for D0/D1 for reduction");

  /// Warp count (concept: GemmShape)
  using WarpCount0 = typename DualMma::WarpCount;
  static int const kThreadCount = 32 * WarpCount0::kCount;

  /// Parameters structure
  struct Params {
    DualGemmMode mode;
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;

    // Mma0
    typename DualMma::IteratorA::Params params_A0;
    typename DualMma::IteratorA::TensorRef ref_A0;
    typename DualMma::IteratorB0::Params params_B0;
    typename DualMma::IteratorB0::TensorRef ref_B0;
    typename Epilogue0::OutputTileIterator::Params params_C0;
    typename Epilogue0::OutputTileIterator::TensorRef ref_C0;
    typename Epilogue0::OutputTileIterator::Params params_D0;
    typename Epilogue0::OutputTileIterator::TensorRef ref_D0;
    typename OutputOp0::Params output_op_0;

    // Mma1
    typename DualMma::IteratorB1::Params params_B1;
    typename DualMma::IteratorB1::TensorRef ref_B1;
    typename Epilogue1::OutputTileIterator::Params params_C1;
    typename Epilogue1::OutputTileIterator::TensorRef ref_C1;
    typename Epilogue1::OutputTileIterator::Params params_D1;
    typename Epilogue1::OutputTileIterator::TensorRef ref_D1;
    typename OutputOp1::Params output_op_1;

    typename Epilogue1::OutputTileIterator::Params params_D2;
    typename Epilogue1::OutputTileIterator::TensorRef ref_D2;
    typename OutputOp2::Params output_op_2;

    int *semaphore;
    int gemm_k_size;

    int64_t batch_stride_A;
    int64_t batch_stride_B0;
    int64_t batch_stride_B1;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      DualGemmMode mode,
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      // Mma0: D0 = A @ B0 + C0
      typename DualMma::IteratorA::TensorRef ref_A0,
      typename DualMma::IteratorB0::TensorRef ref_B0,
      typename Epilogue0::OutputTileIterator::TensorRef ref_C0,
      typename Epilogue0::OutputTileIterator::TensorRef ref_D0,
      // Mma1: D1 = A @ B1 + C1
      typename DualMma::IteratorB1::TensorRef ref_B1,
      typename Epilogue1::OutputTileIterator::TensorRef ref_C1,
      typename Epilogue1::OutputTileIterator::TensorRef ref_D1,

      typename Epilogue1::OutputTileIterator::TensorRef ref_D2,
      typename OutputOp0::Params output_op_0 = typename OutputOp0::Params(),
      typename OutputOp1::Params output_op_1 = typename OutputOp1::Params(),
      typename OutputOp2::Params output_op_2 = typename OutputOp2::Params(),
      int *workspace = nullptr,
      int64_t batch_stride_A = 1,
      int64_t batch_stride_B0 = 1,
      int64_t batch_stride_B1 = 1,
      int64_t batch_stride_C = 1,
      int64_t batch_stride_D = 1
    ):
      mode(mode),
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      // Mma0
      params_A0(ref_A0.layout()),
      ref_A0(ref_A0),
      params_B0(ref_B0.layout()),
      ref_B0(ref_B0),
      params_C0(ref_C0.layout()),
      ref_C0(ref_C0),
      params_D0(ref_D0.layout()),
      ref_D0(ref_D0),
      // Mma1
      params_B1(ref_B1.layout()),
      ref_B1(ref_B1),
      params_C1(ref_C1.layout()),
      ref_C1(ref_C1),
      params_D1(ref_D1.layout()),
      ref_D1(ref_D1),
      params_D2(ref_D2.layout()),
      ref_D2(ref_D2),
      output_op_0(output_op_0),
      output_op_1(output_op_1),
      output_op_2(output_op_2),
      batch_stride_A(batch_stride_A),
      batch_stride_B0(batch_stride_B0),
      batch_stride_B1(batch_stride_B1),
      batch_stride_C(batch_stride_C),
      batch_stride_D(batch_stride_D) {

      int total_gemm_k_iterations = (problem_size.k() + DualMma::Shape::kK - 1) / DualMma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      gemm_k_size = gemm_k_iterations * DualMma::Shape::kK;

      semaphore = workspace;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename DualMma::SharedStorage main_loop;
    typename DualEpilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  DualGemm() { }

  /// Determines whether kernel satisfies alignment
    static Status can_implement(
      cutlass::gemm::GemmCoord const & problem_size,
      typename DualMma::IteratorA::TensorRef ref_A0,
      typename DualMma::IteratorB0::TensorRef ref_B0,
      typename Epilogue0::OutputTileIterator::TensorRef ref_C0,
      typename Epilogue0::OutputTileIterator::TensorRef ref_D0,
      typename DualMma::IteratorB1::TensorRef ref_B1,
      typename Epilogue1::OutputTileIterator::TensorRef ref_C1,
      typename Epilogue1::OutputTileIterator::TensorRef ref_D1,
      typename Epilogue1::OutputTileIterator::TensorRef ref_D2) {

    static int const kAlignmentA = DualMma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = DualMma::IteratorB0::AccessType::kElements;
    static int const kAlignmentC = Epilogue0::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A0, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B0, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C0, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D0, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B1, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C1, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D1, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D2, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A0 = static_cast<ElementA *>(params.ref_A0.data());
    ElementB *ptr_B0 = static_cast<ElementB *>(params.ref_B0.data());
    ElementB *ptr_B1 = static_cast<ElementB *>(params.ref_B1.data());

    //
    // Fetch pointers based on mode.
    //
    if (params.mode == DualGemmMode::kGemm) {
      if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {
        problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
      }

      offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
    }
    else if (params.mode == DualGemmMode::kBatched) {
      ptr_A0 += threadblock_tile_offset.k() * params.batch_stride_A;
      ptr_B0 += threadblock_tile_offset.k() * params.batch_stride_B0;
      ptr_B1 += threadblock_tile_offset.k() * params.batch_stride_B1;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A0{
      threadblock_tile_offset.m() * DualMma::Shape::kM,
      offset_k,
    };

    cutlass::MatrixCoord tb_offset_B0{
      offset_k,
      threadblock_tile_offset.n() * DualMma::Shape::kN
    };

    cutlass::MatrixCoord tb_offset_B1{
      offset_k,
      threadblock_tile_offset.n() * DualMma::Shape::kN
    };

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename DualMma::IteratorA iterator_A0(
      params.params_A0,
      ptr_A0,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A0);

    typename DualMma::IteratorB0 iterator_B0(
      params.params_B0,
      ptr_B0,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B0);

    typename DualMma::IteratorB1 iterator_B1(
      params.params_B1,
      ptr_B1,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B1);


    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //


    // Construct thread-scoped matrix multiply
    typename DualMma::FragmentC accum0;
    typename DualMma::FragmentC accum1;
    accum0.clear();
    accum1.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - offset_k + DualMma::Shape::kK - 1) / DualMma::Shape::kK;

    DualMma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations,
        accum0, accum1,
        iterator_A0, iterator_B0, iterator_B1,
        accum0, accum1);
    }

    //
    // Epilogue
    //

    OutputOp0 output_op_0(params.output_op_0);
    OutputOp1 output_op_1(params.output_op_1);
    OutputOp2 output_op_2(params.output_op_2);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * DualMma::Shape::kM,
      threadblock_tile_offset.n() * DualMma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    ElementC *ptr_C0 = static_cast<ElementC *>(params.ref_C0.data());
    ElementC *ptr_C1 = static_cast<ElementC *>(params.ref_C1.data());
    ElementC *ptr_D0 = static_cast<ElementC *>(params.ref_D0.data());
    ElementC *ptr_D1 = static_cast<ElementC *>(params.ref_D1.data());
    ElementC *ptr_D2 = static_cast<ElementC *>(params.ref_D2.data());

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if (params.mode == DualGemmMode::kGemm) {
      // If performing a reduction via split-K, fetch the initial synchronization
      if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is currently updating
        output_op_0.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
        output_op_1.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
      }
    }
    else if (params.mode == DualGemmMode::kBatched) {
      ptr_C0 += threadblock_tile_offset.k() * params.batch_stride_C;
      ptr_C1 += threadblock_tile_offset.k() * params.batch_stride_C;
      ptr_D0 += threadblock_tile_offset.k() * params.batch_stride_D;
      ptr_D1 += threadblock_tile_offset.k() * params.batch_stride_D;
      ptr_D2 += threadblock_tile_offset.k() * params.batch_stride_D;
    }

    // Tile iterator loading from source tensor.
    typename Epilogue0::OutputTileIterator iterator_C0(
      params.params_C0,
      ptr_C0,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );
    typename Epilogue1::OutputTileIterator iterator_C1(
      params.params_C1,
      ptr_C1,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue0::OutputTileIterator iterator_D0(
      params.params_D0,
      ptr_D0,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );
    typename Epilogue1::OutputTileIterator iterator_D1(
      params.params_D1,
      ptr_D1,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );
    typename Epilogue1::OutputTileIterator iterator_D2(
      params.params_D2,
      ptr_D2,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    DualEpilogue epilogue(
      shared_storage.epilogue,
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C0 = iterator_D0;
        iterator_C1 = iterator_D1;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }

    // Execute the epilogue operator to update the destination tensor.
    typename Epilogue0::OutputTileIterator source_iters[] = {
      iterator_C0, iterator_C1
    };
    const bool writeToD2 = (!kSplitKSerial || params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1);
    epilogue(
      output_op_0, output_op_1, output_op_2,
      iterator_D0, iterator_D1, iterator_D2,
      accum0, accum1,
      source_iters,
      writeToD2
    );
    
    //
    // Release the semaphore
    //

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      __threadfence();
      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

