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
    \brief Grouped Rank2K kernel.
*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/rank_2k_transpose_operands.h"
#include "cutlass/gemm/kernel/rank_2k_grouped_problem_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma1_,                          ///! Threadblock-scoped matrix multiply-accumulate (A*B^T)
  typename Mma2_,                          ///! Threadblock-scoped matrix multiply-accumulate (B*A^T)
  typename Epilogue_,                      ///! Epilogue
  typename ThreadblockSwizzle_,            ///! Threadblock swizzling function
  ComplexTransform OriginalTransformA_,    ///! Public-facing transformation on A
  ComplexTransform OriginalTransformB_,    ///! Public-facing transformation on B
  FillMode FillModeC_,                     ///! Fill Mode for C (kLower or kUpper)
  BlasMode BlasMode_,                      ///! Blas3 computation mode
  GroupScheduleMode GroupScheduleMode_,    ///! Type of scheduling to perform
  bool Transposed = false
>
struct Rank2KGrouped {
public:

  using Mma1 = Mma1_;
  using Mma2 = Mma2_;

  static_assert(platform::is_same<typename Mma1::LayoutC, cutlass::layout::RowMajor>::value &&
                platform::is_same<typename Mma2::LayoutC, cutlass::layout::RowMajor>::value,
                "Kernel-level grouped Rank2K requires that LayoutC be row major.");

  // Define generic Mma for usecases that use Kernel::Mma
  using Mma = Mma1_;

  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed = Transposed;

  // Public-facing type definitions related to operand element type, layout, and complex conjugate
  // operation. Must interact with the 'kTransposed' notion to reflect the original layout,
  // fill mode, etc. passed in.
  //
  // Recall that a Rank2K operation performs (A x BT) + (B x AT)
  // This is performed via:
  //    Mma1 = (A x BT)
  //    Mma2 = (B x AT)
  //
  // However, if C needs to be transposed, then this is changed to the following:
  //    Mma1 = (B x AT)
  //    Mma2 = (A x BT)
  //
  // The transformation above is achieved by swapping the Layouts/Elements/Transforms/etc.
  // of A and B as they are passed into the instantiations of Mma1 and Mma2.
  //
  // Now, given access to only Mma1 and Mma2, as well as whether a transposition has occurred,
  // we wish to retrieve the original Layouts/Elements/etc. for A and B that were passed into
  // the device-level call.
  //
  // The logic to do this (which is made clearer by referencing the above instantiations) is as follows:
  //   LayoutA = kTransposed ? Mma2::LayoutA : Mma1::LayoutA
  //   LayoutB = kTransposed ? Mma1::LayoutA : Mma2::LayoutA
  //
  // We achieve this swapping by passing Mma1::*A and Mma2::*B to Rank2KMapArguments:
  using MapArgumentsA = kernel::detail::Rank2KMapArguments<
    typename Mma1::IteratorA::Element,
    typename Mma1::IteratorA::Layout,
    Mma1::kTransformA,
    Mma1::IteratorA::AccessType::kElements,
    typename Mma2::IteratorA::Element,
    typename Mma2::IteratorA::Layout,
    Mma2::kTransformA,
    Mma2::IteratorA::AccessType::kElements,
    typename Mma1::LayoutC,
    FillModeC_,
    kTransposed
  >;

  using ElementA = typename MapArgumentsA::ElementA;
  using LayoutA = typename MapArgumentsA::LayoutA;
  static int const kAlignmentA = MapArgumentsA::kAlignmentA;

  using MapArgumentsB = kernel::detail::Rank2KMapArguments<
    typename Mma2::IteratorA::Element,
    typename Mma2::IteratorA::Layout,
    Mma2::kTransformA,
    Mma2::IteratorA::AccessType::kElements,
    typename Mma1::IteratorA::Element,
    typename Mma1::IteratorA::Layout,
    Mma1::kTransformA,
    Mma1::IteratorA::AccessType::kElements,
    typename Mma2::LayoutC,
    FillModeC_,
    kTransposed
  >;

  using ElementB = typename MapArgumentsB::ElementA;
  using LayoutB = typename MapArgumentsB::LayoutA;
  static int const kAlignmentB = MapArgumentsB::kAlignmentA;

  // Use the user-provided TransformA and TransformB, rather than those
  // resulting from MapArguments, because Mma1 and Mma2 may have different
  // complex transforms than those passed in by the user.
  // (See kernel/rank_2k_complex.h for an example of this)
  static cutlass::ComplexTransform const kTransformA = OriginalTransformA_;
  static cutlass::ComplexTransform const kTransformB = OriginalTransformB_;

  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArgumentsA::LayoutC;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;
  static FillMode const kFillModeC = MapArgumentsA::kFillModeC;

  // Common type definitions for Mma1 and Mma2
  using Operator = typename Mma1::Operator;
  using OperatorClass = typename Mma1::Operator::OperatorClass;
  using ThreadblockShape = typename Mma1::Shape;
  using WarpShape = typename Mma1::Operator::Shape;
  using InstructionShape = typename Mma1::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma1::ArchTag;

  static int const kStages = Mma1::kStages;
  static BlasMode const kBlasMode = BlasMode_;

private:
  static FillMode const kInternalFillModeC = FillModeC_;

public:

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma1::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor = Rank2KGroupedProblemVisitor<
                            ThreadblockShape,
                            kGroupScheduleMode,
                            kThreadCount,
                            kThreadCount,
                            kInternalFillModeC>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmUniversalMode mode = GemmUniversalMode::kGemm;
    GemmCoord *problem_sizes = nullptr;
    int problem_count{0};
    int threadblock_count{0};

    typename EpilogueOutputOp::Params epilogue;

    ElementA ** ptr_A = nullptr;
    ElementB ** ptr_B = nullptr;
    ElementC ** ptr_C = nullptr;
    ElementC ** ptr_D = nullptr;

    typename LayoutA::Stride::LongIndex *lda = nullptr;
    typename LayoutB::Stride::LongIndex *ldb = nullptr;
    typename LayoutC::Stride::LongIndex *ldc = nullptr;
    typename LayoutC::Stride::LongIndex *ldd = nullptr;

    // Only used by device-level operator
    GemmCoord *host_problem_sizes = nullptr;

    bool allow_early_exit = false;

    //
    // Methods
    //

    /// Default ctor
    Arguments() = default;

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmUniversalMode mode,
      GemmCoord *problem_sizes,
      int problem_count,
      int threadblock_count,
      typename EpilogueOutputOp::Params epilogue,
      ElementA ** ptr_A,
      ElementB ** ptr_B,
      ElementC ** ptr_C,
      ElementC ** ptr_D,
      typename LayoutA::Stride::LongIndex *lda,
      typename LayoutB::Stride::LongIndex *ldb,
      typename LayoutC::Stride::LongIndex *ldc,
      typename LayoutC::Stride::LongIndex *ldd,
      GemmCoord *host_problem_sizes=nullptr,
      bool allow_early_exit=false
    ):
      mode(mode),
      problem_sizes(problem_sizes),
      problem_count(problem_count),
      threadblock_count(threadblock_count),
      epilogue(epilogue),
      ptr_A(ptr_A),
      ptr_B(ptr_B),
      ptr_C(ptr_C),
      ptr_D(ptr_D),
      lda(lda),
      ldb(ldb),
      ldc(ldc),
      ldd(ldd),
      host_problem_sizes(host_problem_sizes),
      allow_early_exit(allow_early_exit)
    {

    }

  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    typename ProblemVisitor::Params problem_visitor{};
    int threadblock_count = 0;

    typename EpilogueOutputOp::Params output_op{};

    GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
    int batch_count = 0;

    ElementA** ptr_A = nullptr;
    ElementB** ptr_B = nullptr;
    ElementC** ptr_C = nullptr;
    ElementC** ptr_D = nullptr;

    typename LayoutA::Stride::LongIndex* lda = nullptr;
    typename LayoutB::Stride::LongIndex* ldb = nullptr;
    typename LayoutC::Stride::LongIndex* ldc = nullptr;
    typename LayoutC::Stride::LongIndex* ldd = nullptr;

    bool allow_early_exit = false;

    //
    // Methods
    //

    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args, void *workspace = nullptr, int tile_count = 0):
      problem_visitor(args.problem_sizes, args.problem_count, workspace, tile_count),
      threadblock_count(args.threadblock_count),
      output_op(args.epilogue),
      ptr_A(args.ptr_A),
      ptr_B(args.ptr_B),
      ptr_C(args.ptr_C),
      ptr_D(args.ptr_D),
      lda(args.lda),
      ldb(args.ldb),
      ldc(args.ldc),
      ldd(args.ldd),
      allow_early_exit(args.allow_early_exit)
    {

    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr,
      int tile_count = 0) {

      problem_visitor = typename ProblemVisitor::Params(args.problem_sizes, args.problem_count, workspace, tile_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B = args.ptr_B;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
    }
  };

  /// Shared memory storage structure
  struct SharedStorage {
    union {
      typename Mma1::SharedStorage mma1_main_loop;
      typename Mma2::SharedStorage mma2_main_loop;
      typename Epilogue::SharedStorage epilogue;
    } kernel;

    // ProblemVisitor shared storage can't be overlapped with others
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

public:

  //
  // Methods
  //

  Rank2KGrouped() = default;

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Early exit following LAPACK's definition
    if (params.allow_early_exit &&
        (params.output_op.alpha == ElementC(0)) && (params.output_op.beta == ElementC(1))) {
      return;
    }

    //
    // Problem visitor.
    //

    ProblemVisitor problem_visitor(
      params.problem_visitor,
      shared_storage.problem_visitor,
      blockIdx.x);

    // Outer 'persistent' loop to iterate over tiles
    while (problem_visitor.next_tile()) {

      GemmCoord problem_size  = problem_visitor.problem_size();
      int32_t problem_idx     = problem_visitor.problem_index();
      int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

      GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

      cutlass::gemm::GemmCoord threadblock_tile_offset = problem_visitor.threadblock_offset(threadblock_idx);

      //
      // Perform checks to determine whether the results of this threadblock will be needed.
      // An example of an unneeded threadblock is one that is assigned to compute in the upper
      // portion of a Rank2K kernel filled with mode kLower.
      //
      //

      // Early exit if threadblock is out of range
      if (grid_shape.m() <= threadblock_tile_offset.m() ||
          grid_shape.n() <= threadblock_tile_offset.n()) {
        // Next tile
        problem_visitor.advance(gridDim.x);
        continue;
      }

      // Skip this tile if Fill Mode is Lower and
      // if the entire tile is above the main diagonal (bottom-left corner is at or above the diagonal)
      if (kInternalFillModeC == cutlass::FillMode::kLower &&
          (threadblock_tile_offset.m() + 1) * Mma1::Shape::kM <= threadblock_tile_offset.n() * Mma1::Shape::kN) {
        // Next tile
        problem_visitor.advance(gridDim.x);
        continue;
      }

      // Skip this tile if Fill Mode is Upper and
      // if the entire tile is below the main diagonal (top-right corner is at or below the diagonal)
      if (kInternalFillModeC == cutlass::FillMode::kUpper &&
          threadblock_tile_offset.m() * Mma1::Shape::kM >= (threadblock_tile_offset.n() + 1) * Mma1::Shape::kN) {
        // Next tile
        problem_visitor.advance(gridDim.x);
        continue;
      }

      bool tile_on_diagonal = false;
      // Mark tiles that are being crossed by the main diagonal
      // (top-right and bottom-left corners are on either side of the diagonal)
      if ((threadblock_tile_offset.m() + 1) * Mma1::Shape::kM > threadblock_tile_offset.n() * Mma1::Shape::kN
          && threadblock_tile_offset.m() * Mma1::Shape::kM < (threadblock_tile_offset.n() + 1) * Mma1::Shape::kN) {
        tile_on_diagonal = true;
      }

      int offset_k = 0;
      int problem_size_k = problem_size.k();

      //
      // Fetch pointers based on mode.
      //
      if (params.mode == GemmUniversalMode::kGemm ||
          params.mode == GemmUniversalMode::kGemmSplitKParallel) {

        if (threadblock_tile_offset.k() + 1 < grid_shape.k()) {
          problem_size_k = (threadblock_tile_offset.k() + 1) * problem_size.k();
        }

        offset_k = threadblock_tile_offset.k() * problem_size.k();
      }

      ElementA *ptr_A = reinterpret_cast<ElementA *>((kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]));
      typename LayoutA::Stride::LongIndex ldm_A = (kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx]);

      ElementB *ptr_B = reinterpret_cast<ElementB *>((kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]));
      typename LayoutB::Stride::LongIndex ldm_B = (kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx]);

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_MxK{
        threadblock_tile_offset.m() * Mma1::Shape::kM,
        offset_k,
      };

      cutlass::MatrixCoord tb_offset_KxN{
        offset_k,
        threadblock_tile_offset.n() * Mma1::Shape::kN
      };

      // Assume identity swizzle
      MatrixCoord tb_offset(
        threadblock_tile_offset.m() * Mma1::Shape::kM,
        threadblock_tile_offset.n() * Mma1::Shape::kN
      );

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands for Mma1
      typename Mma1::IteratorA iterator_A(
        typename Mma1::IteratorA::Params(ldm_A),
        ptr_A,
        {problem_size.m(), problem_size_k},
        thread_idx,
        tb_offset_MxK);

      typename Mma1::IteratorB iterator_BT(
        typename Mma1::IteratorB::Params(ldm_B),
        ptr_B,
        {problem_size_k, problem_size.n()},
        thread_idx,
        tb_offset_KxN);

      // Construct iterators to A and B operands for Mma2
      typename Mma2::IteratorA iterator_B(
        typename Mma2::IteratorA::Params(ldm_B),
        ptr_B,
        {problem_size.m(), problem_size_k},
        thread_idx,
        tb_offset_MxK);

      typename Mma2::IteratorB iterator_AT(
        typename Mma2::IteratorB::Params(ldm_A),
        ptr_A,
        {problem_size_k, problem_size.n()},
        thread_idx,
        tb_offset_KxN);

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = canonical_warp_idx_sync();

      int lane_idx = threadIdx.x % 32;

      //
      // Main loop
      //

      // Construct thread-scoped matrix multiply for Mma1 (A x BT)
      Mma1 mma1(shared_storage.kernel.mma1_main_loop, thread_idx, warp_idx, lane_idx);

      // Construct thread-scoped matrix multiply for Mma2 (B x AT)
      Mma2 mma2(shared_storage.kernel.mma2_main_loop, thread_idx, warp_idx, lane_idx);

      typename Mma1::FragmentC accumulators;

      accumulators.clear();

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations = (problem_size_k - offset_k + Mma1::Shape::kK - 1) / Mma1::Shape::kK;

      // Wait for all threads to finish their epilogue phases from the previous tile.
      __syncthreads();

      // Compute threadblock-scoped matrix multiply-add (A x BT)
      mma1(
        gemm_k_iterations,
        accumulators,
        iterator_A,
        iterator_BT,
        accumulators);

      // HER2K kernel needs Alpha to be complex and is conj(Alpha) is applied to the second HERK.
      if (kBlasMode == BlasMode::kHermitian) {

        //
        // Epilogue
        //

        EpilogueOutputOp output_op(params.output_op);

        int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * grid_shape.m();

        ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C[problem_idx]);
        ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D[problem_idx]);

        // If TB not on diagonal, FillMode doesn't apply.
        FillMode kFillModeTB = tile_on_diagonal ? kInternalFillModeC : FillMode::kNone;

        // Tile iterator loading from source tensor.
        typename Epilogue::OutputTileIterator iterator_C(
          typename Epilogue::OutputTileIterator::Params(params.ldc[problem_idx]),
          ptr_C,
          problem_size.mn(),
          thread_idx,
          tb_offset,
          kFillModeTB
        );

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D(
          typename Epilogue::OutputTileIterator::Params(params.ldd[problem_idx]),
          ptr_D,
          problem_size.mn(),
          thread_idx,
          tb_offset,
          kFillModeTB
        );

        Epilogue epilogue(
          shared_storage.kernel.epilogue,
          thread_idx,
          warp_idx,
          lane_idx);

        // Execute the epilogue operator to update the destination tensor.
        epilogue(
          output_op,
          iterator_D,
          accumulators,
          iterator_C);

        __syncthreads();

        accumulators.clear();
      }

      // Compute threadblock-scoped matrix multiply-add (B x AT)
      mma2(
        gemm_k_iterations,
        accumulators,
        iterator_B,
        iterator_AT,
        accumulators);

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      /* Needed for HER2K where the second HERK is multiplied by conj(alpha) */
      typename EpilogueOutputOp::Params second_her2k_params(conj(params.output_op.alpha), 1);
      EpilogueOutputOp output_op_her2k(second_her2k_params);

      //
      // Masked tile iterators constructed from members
      //

      int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * grid_shape.m();

      ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C[problem_idx]);

      // HER2K kernel needs Alpha to be complex and is conj(Alpha) is applied to the second HERK.
      if (kBlasMode == BlasMode::kHermitian) {
        ptr_C = static_cast<ElementC *>(params.ptr_D[problem_idx]);
      }

      ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D[problem_idx]);

      // If TB not on diagonal, FillMode doesn't apply.
      FillMode kFillModeTB = tile_on_diagonal ? kInternalFillModeC : FillMode::kNone;

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_C(
        typename Epilogue::OutputTileIterator::Params(params.ldc[problem_idx]),
        ptr_C,
        problem_size.mn(),
        thread_idx,
        tb_offset,
        kFillModeTB
      );

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(
        typename Epilogue::OutputTileIterator::Params(params.ldd[problem_idx]),
        ptr_D,
        problem_size.mn(),
        thread_idx,
        tb_offset,
        kFillModeTB
      );

      Epilogue epilogue(
        shared_storage.kernel.epilogue,
        thread_idx,
        warp_idx,
        lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      if (kBlasMode == BlasMode::kSymmetric) {
        epilogue(
          output_op,
          iterator_D,
          accumulators,
          iterator_C);
      } else {
        epilogue(
          output_op_her2k,
          iterator_D,
          accumulators,
          iterator_C);
      }

      // Next tile
      problem_visitor.advance(gridDim.x);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
