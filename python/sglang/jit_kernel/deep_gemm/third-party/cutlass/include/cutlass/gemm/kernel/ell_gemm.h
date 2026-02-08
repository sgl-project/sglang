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
    \brief Template for a Block-Ell sparse gemm kernel.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/arch/arch.h"

#include "cutlass/transform/threadblock/ell_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial,              ///! If true, code supporting split-K via serial reduction is enabled.
  bool IsASparse                  ///! If true, A is sparse matrix
>
struct EllGemm {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size{};
    cutlass::gemm::GemmCoord grid_tiled_shape{};
    int swizzle_log_tile{0};
    typename Mma::IteratorA::Params params_A{};
    typename Mma::IteratorA::TensorRef ref_A{};
    typename Mma::IteratorB::Params params_B{};
    typename Mma::IteratorB::TensorRef ref_B{};
    typename Epilogue::OutputTileIterator::Params params_C{};
    typename Epilogue::OutputTileIterator::TensorRef ref_C{};
    typename Epilogue::OutputTileIterator::Params params_D{};
    typename Epilogue::OutputTileIterator::TensorRef ref_D{};
    typename OutputOp::Params output_op{};
    int *semaphore = nullptr;
    int gemm_k_iterations{0};
    int gemm_k_size{0};
    const int* ell_idx = nullptr;
    int ell_ncol{0};
    int ell_blocksize{0};
    int ell_base_idx{0};

    //
    // Methods
    //
   Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      const int* ell_idx,
      int ell_ncol,
      int ell_blocksize,
      int ell_base_idx,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *workspace = nullptr
    ):
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A.layout()),
      ref_A(ref_A),
      params_B(ref_B.layout()),
      ref_B(ref_B),
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      output_op(output_op),
      ell_idx(ell_idx),
      ell_ncol(ell_ncol),
      ell_blocksize(ell_blocksize),
      ell_base_idx(ell_base_idx)
    {

      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();

      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

    semaphore = workspace;
    }
  };

  /// Shared memory storage structure
  struct SharedStorage {
    union{
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
    };
    typename cutlass::transform::threadblock::ell::SharedStorage ell;
  };

  //
  // Methods
  //
  EllGemm() = default;

  /// Determines whether kernel satisfies alignment
    static Status can_implement(
      cutlass::gemm::GemmCoord const & problem_size,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorA::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =  (platform::is_same<typename Mma::IteratorB::Layout,
                                                       layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorB::Layout,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if ((problem_size.m() % kAlignmentA) || (problem_size.k() % kAlignmentA) ||
      (problem_size.n() % kAlignmentB) || (problem_size.k() % kAlignmentB) ||
      (problem_size.m() % kAlignmentC) || (problem_size.n() % kAlignmentC)) {

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

    int tile_in_ell_block = (params.ell_blocksize + Mma::Shape::kM - 1 ) / Mma::Shape::kM;
    int ell_block_offset_m = threadblock_tile_offset.m() / tile_in_ell_block;
    int tile_offset_m = threadblock_tile_offset.m() % tile_in_ell_block;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // skip computation if matrix is 0
    if (params.ell_ncol > 0) {

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
        ell_block_offset_m * params.ell_blocksize
        + tile_offset_m * Mma::Shape::kM,
        threadblock_tile_offset.k() * params.gemm_k_size
      };

      cutlass::MatrixCoord tb_offset_B{
        threadblock_tile_offset.k() * params.gemm_k_size,
        threadblock_tile_offset.n() * Mma::Shape::kN
      };

      int ell_idx_start =
        (threadblock_tile_offset.m() / tile_in_ell_block) *
        (params.ell_ncol / params.ell_blocksize);
      const int* ell_idx_ptr = &(params.ell_idx[ell_idx_start]);

      // Problem size is a function of threadblock index in the K dimension
      int problem_size_k = min(
        params.problem_size.k(),
        (threadblock_tile_offset.k() + 1) * params.gemm_k_size);
      problem_size_k = min(problem_size_k, params.ell_ncol);

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations =
        (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        params.params_A,
        params.ref_A.data(),
        {params.problem_size.m(), problem_size_k},
        thread_idx,
        tb_offset_A);

      typename Mma::IteratorB iterator_B(
        params.params_B,
        params.ref_B.data(),
        {problem_size_k, params.problem_size.n()},
        thread_idx,
        tb_offset_B);

      // Define coef for ELL index depending on LayoutB
      int ell_stride = iterator_B.get_stride();

      typename cutlass::transform::threadblock::ell::Iterator ell_iterator(
        shared_storage.ell,
        ell_idx_ptr,
        params.ell_blocksize,
        params.ell_base_idx,
        Mma::Shape::kK,
        problem_size_k,
        ell_stride,
        thread_idx
      );

      //
      // Main loop
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      if (!kSplitKSerial || gemm_k_iterations > 0) {
        // check if index computations can be skipped
        static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
        static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
        static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;
        constexpr bool is_double = (sizeof(Mma::IteratorA::Element) == 8);
        constexpr bool is_multiple_alignment =  
          (kAlignmentA > 1) && (kAlignmentB > 1) && (kAlignmentC > 1);
        const bool is_specialized_blocksize =
          ((params.ell_blocksize) & (params.ell_blocksize-1)) == 0
          && params.ell_blocksize >= Mma::Shape::kK;
        // Compute threadblock-scoped matrix multiply-add
        if ((is_double || is_multiple_alignment) && is_specialized_blocksize) {
          mma.operator()<true, true>(
              gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators, ell_iterator);
        } 
        else {
          mma.operator()<true, false>(
              gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators, ell_iterator);
        }
      }
    } // if (params.ell_ncols > 0)

    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    ell_block_offset_m = threadblock_tile_offset.m() / tile_in_ell_block;
    tile_offset_m = threadblock_tile_offset.m() % tile_in_ell_block;

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      ell_block_offset_m * params.ell_blocksize
      + tile_offset_m * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    //avoid out of bounds
    MatrixCoord threadblock_extent(
      min(params.problem_size.m(),
         ell_block_offset_m * params.ell_blocksize
         + min((tile_offset_m + 1) * Mma::Shape::kM, params.ell_blocksize)),
      min(params.problem_size.n(),
        (threadblock_tile_offset.n()+1) * Mma::Shape::kN)
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      params.ref_C.data(),
      threadblock_extent,
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      threadblock_extent,
      thread_idx,
      threadblock_offset
    );

    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op, iterator_D, accumulators, iterator_C);

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

      semaphore.release(lock);
    }
  }
};

// B is Sparse
template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
>
struct EllGemm<Mma_, Epilogue_, ThreadblockSwizzle_, SplitKSerial, false> {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size{};
    cutlass::gemm::GemmCoord grid_tiled_shape{};
    int swizzle_log_tile{0};
    typename Mma::IteratorA::Params params_A{};
    typename Mma::IteratorA::TensorRef ref_A{};
    typename Mma::IteratorB::Params params_B{};
    typename Mma::IteratorB::TensorRef ref_B{};
    typename Epilogue::OutputTileIterator::Params params_C{};
    typename Epilogue::OutputTileIterator::TensorRef ref_C{};
    typename Epilogue::OutputTileIterator::Params params_D{};
    typename Epilogue::OutputTileIterator::TensorRef ref_D{};
    typename OutputOp::Params output_op{};
    int *semaphore = nullptr;
    int gemm_k_iterations{0};
    int gemm_k_size{0};
    const int* ell_idx = nullptr;
    int ell_ncol{0};
    int ell_blocksize{0};
    int ell_base_idx{0};

    //
    // Methods
    //
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      const int* ell_idx,
      int ell_ncol,
      int ell_blocksize,
      int ell_base_idx,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *workspace = nullptr
    ):
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A.layout()),
      ref_A(ref_A),
      params_B(ref_B.layout()),
      ref_B(ref_B),
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      output_op(output_op),
      ell_idx(ell_idx),
      ell_ncol(ell_ncol),
      ell_blocksize(ell_blocksize),
      ell_base_idx(ell_base_idx)
    {

      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();

      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

    semaphore = workspace;
    }
  };

  /// Shared memory storage structure
  struct SharedStorage {
    union{
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
    };
    typename cutlass::transform::threadblock::ell::SharedStorage ell;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  EllGemm() { }

  /// Determines whether kernel satisfies alignment
    static Status can_implement(
      cutlass::gemm::GemmCoord const & problem_size,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorA::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =  (platform::is_same<typename Mma::IteratorB::Layout,
                                                       layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorB::Layout,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if ((problem_size.m() % kAlignmentA) || (problem_size.k() % kAlignmentA) ||
      (problem_size.n() % kAlignmentB) || (problem_size.k() % kAlignmentB) ||
      (problem_size.m() % kAlignmentC) || (problem_size.n() % kAlignmentC)) {

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

    int tile_in_ell_block = (params.ell_blocksize + Mma::Shape::kN - 1 ) / Mma::Shape::kN;
    int ell_block_offset_n = threadblock_tile_offset.n() / tile_in_ell_block;
    int tile_offset_n = threadblock_tile_offset.n() % tile_in_ell_block;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // skip computation if matrix is 0
    if (params.ell_ncol > 0) {

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.k() * params.gemm_k_size,
      };

      cutlass::MatrixCoord tb_offset_B{
        threadblock_tile_offset.k() * params.gemm_k_size,
        ell_block_offset_n * params.ell_blocksize
        + tile_offset_n * Mma::Shape::kN,
      };

      int ell_idx_start =
        (threadblock_tile_offset.n() / tile_in_ell_block) *
        (params.ell_ncol / params.ell_blocksize);
      const int* ell_idx_ptr = &(params.ell_idx[ell_idx_start]);

      // Problem size is a function of threadblock index in the K dimension
      int problem_size_k = min(
        params.problem_size.k(),
        (threadblock_tile_offset.k() + 1) * params.gemm_k_size);
      problem_size_k = min(problem_size_k, params.ell_ncol);

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations =
        (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        params.params_A,
        params.ref_A.data(),
        {params.problem_size.m(), problem_size_k},
        thread_idx,
        tb_offset_A);

      typename Mma::IteratorB iterator_B(
        params.params_B,
        params.ref_B.data(),
        {problem_size_k, params.problem_size.n()},
        thread_idx,
        tb_offset_B);

      // Define coef for ELL index depending on LayoutA
      int ell_stride = iterator_A.get_stride();

      typename cutlass::transform::threadblock::ell::Iterator ell_iterator(
        shared_storage.ell,
        ell_idx_ptr,
        params.ell_blocksize,
        params.ell_base_idx,
        Mma::Shape::kK,
        problem_size_k,
        ell_stride,
        thread_idx
      );

      //
      // Main loop
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      if (!kSplitKSerial || gemm_k_iterations > 0) {
        // check if index computations can be skipped
        static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
        static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
        static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;
        constexpr bool is_double = (sizeof(typename Mma::IteratorA::Element) == 8);
        constexpr bool is_multiple_alignment =
          (kAlignmentA > 1) && (kAlignmentB > 1) && (kAlignmentC > 1);
        const bool is_specialized_blocksize =
          ((params.ell_blocksize) & (params.ell_blocksize-1)) == 0
          && params.ell_blocksize >= Mma::Shape::kK;
        // Compute threadblock-scoped matrix multiply-add
        if ((is_double || is_multiple_alignment) && is_specialized_blocksize) {
          mma.template operator()<false, true>(
              gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators, ell_iterator);
        }
        else {
          mma.template operator()<false, false>(
              gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators, ell_iterator);
        }
      }
    } // if (params.ell_ncols > 0)

    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    ell_block_offset_n = threadblock_tile_offset.n() / tile_in_ell_block;
    tile_offset_n = threadblock_tile_offset.n() % tile_in_ell_block;

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      ell_block_offset_n * params.ell_blocksize
      + tile_offset_n * Mma::Shape::kN
    );

    //avoid out of bounds
    MatrixCoord threadblock_extent(
      min(params.problem_size.m(),
        (threadblock_tile_offset.m()+1) * Mma::Shape::kM),
      min(params.problem_size.n(),
         ell_block_offset_n * params.ell_blocksize
         + min((tile_offset_n + 1) * Mma::Shape::kN, params.ell_blocksize))
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      params.ref_C.data(),
      threadblock_extent,
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      threadblock_extent,
      thread_idx,
      threadblock_offset
    );

    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op, iterator_D, accumulators, iterator_C);

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

      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

