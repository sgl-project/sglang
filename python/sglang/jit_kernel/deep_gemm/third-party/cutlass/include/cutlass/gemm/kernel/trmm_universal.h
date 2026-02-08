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

*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"
#include "cutlass/core_io.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  SideMode SideMode_,             ///! Side Mode for the kernel (kLeft or kRight)
  FillMode FillMode_,             ///! Fill Mode for triangular matrix (kLower or kUpper)
  DiagType DiagType_              ///! Diag Type for triangular matrix (kNonUnit or kUnit)
>
struct TrmmUniversal {
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Epilogue::OutputTileIterator::Layout;
  static SideMode const kSideMode = SideMode_;
  static FillMode const kFillMode = FillMode_;
  static DiagType const kDiagType = DiagType_;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmUniversalMode mode{GemmUniversalMode::kGemm};
    GemmCoord problem_size{};
    int batch_count{1};

    typename EpilogueOutputOp::Params epilogue{};

    void const * ptr_A{nullptr};
    void const * ptr_B{nullptr};
    void * ptr_D{nullptr};

    int64_t batch_stride_A{0};
    int64_t batch_stride_B{0};
    int64_t batch_stride_D{0};

    typename LayoutA::Stride::Index lda{0};
    typename LayoutB::Stride::Index ldb{0};
    typename LayoutC::Stride::Index ldd{0};

    //
    // Methods
    //

    Arguments() = default;

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void * ptr_D,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_D,
      typename LayoutA::Stride::Index lda,
      typename LayoutB::Stride::Index ldb,
      typename LayoutC::Stride::Index ldd
    ):
      mode(mode), 
      problem_size(problem_size),
      batch_count(batch_count),
      epilogue(epilogue), 
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_D(ptr_D), 
      batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B), batch_stride_D(batch_stride_D), 
      lda(lda), ldb(ldb), ldd(ldd) {
      }
    
    /// Returns arguments for the transposed problem sizes
    Arguments transposed_problem_size() const {
      Arguments args(*this);

      std::swap(args.problem_size.m(), args.problem_size.n());

      return args;
    }

    /// Returns arguments for the transposed matrices
    Arguments swapped_matrices() const {
      Arguments args(*this);

      std::swap(args.ptr_A, args.ptr_B);
      std::swap(args.lda, args.ldb);
      std::swap(args.batch_stride_A, args.batch_stride_B);

      return args;
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    cutlass::gemm::GemmCoord problem_size{};
    cutlass::gemm::GemmCoord grid_tiled_shape{};
    int swizzle_log_tile{0};
   
    typename Mma::IteratorA::Params params_A{};
    typename Mma::IteratorB::Params params_B{};
    typename Epilogue::OutputTileIterator::Params params_D{};
    
    typename EpilogueOutputOp::Params output_op{};

    GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
    int batch_count {0};
    int gemm_k_size {0};

    void * ptr_A{nullptr};
    void * ptr_B{nullptr};
    void * ptr_D{nullptr};

    int64_t batch_stride_A {0};
    int64_t batch_stride_B {0};
    int64_t batch_stride_D {0};

    int *semaphore{nullptr};

    //
    // Methods
    //
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      int gemm_k_size,
      void *workspace = nullptr
    ):
      problem_size(args.problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(args.lda),
      params_B(args.ldb),
      params_D(args.ldd),
      output_op(args.epilogue),
      mode(args.mode),
      batch_count(args.batch_count),
      gemm_k_size(gemm_k_size),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_D(args.ptr_D),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_D(args.batch_stride_D),
      semaphore(static_cast<int *>(workspace)) {
    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr) {

      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_D = args.ptr_D;

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_D = args.batch_stride_D;

      output_op = args.epilogue;

      semaphore = static_cast<int *>(workspace);
    }

  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  TrmmUniversal() { } 

  /// Determines whether kernel satisfies alignment
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size) {

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if ((problem_size.m() % kAlignmentA) || (problem_size.k() % kAlignmentA) ||
      (problem_size.n() % kAlignmentB) || (problem_size.k() % kAlignmentB) ||
      (problem_size.m() % kAlignmentC) || (problem_size.n() % kAlignmentC)) {

      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A); 
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    //
    // Fetch pointers based on mode.
    //
    if (params.mode == GemmUniversalMode::kGemm || 
      params.mode == GemmUniversalMode::kGemmSplitKParallel) {

      if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {

        problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size; 
      }

      offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_A += threadblock_tile_offset.k() * params.batch_stride_A;
      ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA * const *>(params.ptr_A)[threadblock_tile_offset.k()];
      ptr_B = static_cast<ElementB * const *>(params.ptr_B)[threadblock_tile_offset.k()];
    }

    __syncthreads();

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      offset_k,
    };

    cutlass::MatrixCoord tb_offset_B{
      offset_k,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();

    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;
    
    /******************************************************************************************************
      First two cases: (Left Side, Lower Fill) and (Right Side, Upper Fill) are transpose of each other
        - (Left Side, Lower Fill): calculate bottom of the CTA tile,  then find the k-iterations 
                                    needed to process all elements till that coordinate.
        - (Right Side, Upper Fill): calculate right end of the CTA tile,  then find the k-iterations 
                                    needed to process all elements till that coordinate.

      Last two cases: (Left Side, Upper Fill) and (Right Side, Lower Fill) are transpose of each other
        - (Left Side, Upper Fill): calculate the top of the CTA tile, then find k-iterations 
                                   that can be skipped for all elements of this tile.
        - (Right Side, Lower Fill): calculate the left start of the CTA tile, then find k-iterations 
                                    that can be skipped for all elements of this tile.
    ********************************************************************************************************/
 
    if (kSideMode == SideMode::kLeft && kFillMode == FillMode::kLower) {

      int k_iterations_till_diagonal = ((threadblock_tile_offset.m() + 1) * Mma::Shape::kM + Mma::Shape::kK - 1) / Mma::Shape::kK;
      if (k_iterations_till_diagonal < gemm_k_iterations) {
        gemm_k_iterations = k_iterations_till_diagonal;
      }

    } else if (kSideMode == SideMode::kRight && kFillMode == FillMode::kUpper) {

      int k_iterations_till_diagonal = ((threadblock_tile_offset.n() + 1) * Mma::Shape::kN + Mma::Shape::kK - 1) / Mma::Shape::kK;
      if (k_iterations_till_diagonal < gemm_k_iterations) {
        gemm_k_iterations = k_iterations_till_diagonal;
      }

    } else if (kSideMode == SideMode::kLeft && kFillMode == FillMode::kUpper) {

      int k_iterations_till_diagonal = ((threadblock_tile_offset.m()) * Mma::Shape::kM) / Mma::Shape::kK;

      if (k_iterations_till_diagonal != 0) {
        tb_offset_A += cutlass::MatrixCoord({0, k_iterations_till_diagonal * Mma::Shape::kK});
        tb_offset_B += cutlass::MatrixCoord({k_iterations_till_diagonal * Mma::Shape::kK, 0});
        gemm_k_iterations -= k_iterations_till_diagonal;
      }

    } else if (kSideMode == SideMode::kRight && kFillMode == FillMode::kLower) {

      int k_iterations_till_diagonal = ((threadblock_tile_offset.n()) * Mma::Shape::kN) / Mma::Shape::kK;

      if (k_iterations_till_diagonal != 0) {
        tb_offset_A += cutlass::MatrixCoord({0, k_iterations_till_diagonal * Mma::Shape::kK});
        tb_offset_B += cutlass::MatrixCoord({k_iterations_till_diagonal * Mma::Shape::kK, 0});
        gemm_k_iterations -= k_iterations_till_diagonal;
      }

    }

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    // Compute threadblock-scoped matrix multiply-add
    mma(
      gemm_k_iterations, 
      accumulators, 
      iterator_A, 
      iterator_B, 
      accumulators);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);

    //
    // Fetch pointers based on mode.
    //
    
    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if (params.mode == GemmUniversalMode::kGemm) {

      // If performing a reduction via split-K, fetch the initial synchronization
      if (params.grid_tiled_shape.k() > 1) {
        
        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is currently updating
        output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
      }
    }
    else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[threadblock_tile_offset.k()];
    }

    
    // Tile iterator loading from source tensor (although irrelevant to this kernel as beta is zero).
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_D,
      ptr_D,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      ptr_D,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }


    // Execute the epilogue operator to update the destination tensor.
    epilogue(
      output_op, 
      iterator_D, 
      accumulators, 
      iterator_C); 
    
    //
    // Release the semaphore
    //

    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) { 

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }
      
      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
