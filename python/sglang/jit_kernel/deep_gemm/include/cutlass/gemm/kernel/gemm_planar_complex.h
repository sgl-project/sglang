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

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"
#include "cutlass/gemm/kernel/params_universal_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmPlanarComplex {
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
  using Operator = typename Mma::Operator;
  using ArchTag = typename Mma::ArchTag;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(
    128 / sizeof_bits<ElementA>::value, 
    128 / sizeof_bits<ElementB>::value);

  //
  // Additional types needed for reflection
  //

  using ElementAccumulator = typename Mma::Policy::Operator::ElementC;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::Shape;

  static int const kStages = Mma::kStages;
    
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  //
  // Arguments structure
  //

  /// Argument structure
  struct Arguments : UniversalArgumentsBase
  {
    //
    // Data members
    //

    typename EpilogueOutputOp::Params epilogue{};

    void const * ptr_A_real{nullptr};
    void const * ptr_A_imag{nullptr};
    void const * ptr_B_real{nullptr};
    void const * ptr_B_imag{nullptr};
    void const * ptr_C_real{nullptr};
    void const * ptr_C_imag{nullptr};
    void * ptr_D_real{nullptr};
    void * ptr_D_imag{nullptr};

    typename LayoutA::Stride::Index lda_real{};
    typename LayoutA::Stride::Index lda_imag{};
    typename LayoutB::Stride::Index ldb_real{};
    typename LayoutB::Stride::Index ldb_imag{};
    typename LayoutC::Stride::Index ldc_real{};
    typename LayoutC::Stride::Index ldc_imag{};
    typename LayoutC::Stride::Index ldd_real{};
    typename LayoutC::Stride::Index ldd_imag{};
    
    int64_t batch_stride_A{0};
    int64_t batch_stride_A_imag{0};
    int64_t batch_stride_B{0};
    int64_t batch_stride_B_imag{0};
    int64_t batch_stride_C{0};
    int64_t batch_stride_C_imag{0};
    int64_t batch_stride_D_imag{0};

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
      void const * ptr_A_real,
      void const * ptr_A_imag,
      void const * ptr_B_real,
      void const * ptr_B_imag,
      void const * ptr_C_real,
      void const * ptr_C_imag,
      void * ptr_D_real,
      void * ptr_D_imag,
      typename LayoutA::Stride::Index lda_real,
      typename LayoutA::Stride::Index lda_imag,
      typename LayoutB::Stride::Index ldb_real,
      typename LayoutB::Stride::Index ldb_imag,
      typename LayoutC::Stride::Index ldc_real,
      typename LayoutC::Stride::Index ldc_imag,
      typename LayoutC::Stride::Index ldd_real,
      typename LayoutC::Stride::Index ldd_imag,
      int64_t batch_stride_A = 0,
      int64_t batch_stride_A_imag = 0,
      int64_t batch_stride_B = 0,
      int64_t batch_stride_B_imag = 0,
      int64_t batch_stride_C = 0,
      int64_t batch_stride_C_imag = 0,
      int64_t batch_stride_D = 0,
      int64_t batch_stride_D_imag = 0)
    :
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue(epilogue), 
      ptr_A_real(ptr_A_real), 
      ptr_A_imag(ptr_A_imag), 
      ptr_B_real(ptr_B_real),
      ptr_B_imag(ptr_B_imag),
      ptr_C_real(ptr_C_real),
      ptr_C_imag(ptr_C_imag),
      ptr_D_real(ptr_D_real), 
      ptr_D_imag(ptr_D_imag), 
      lda_real(lda_real),
      lda_imag(lda_imag),
      ldb_real(ldb_real),
      ldb_imag(ldb_imag),
      ldc_real(ldc_real),
      ldc_imag(ldc_imag),
      ldd_real(ldd_real),
      ldd_imag(ldd_imag),
      batch_stride_A(batch_stride_A),
      batch_stride_A_imag(batch_stride_A_imag),
      batch_stride_B(batch_stride_B),
      batch_stride_B_imag(batch_stride_B_imag),
      batch_stride_C(batch_stride_C),
      batch_stride_C_imag(batch_stride_C_imag),
      batch_stride_D_imag(batch_stride_D_imag)
    {}

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);
      
      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_A_real, args.ptr_B_real);
      std::swap(args.ptr_A_imag, args.ptr_B_imag);
      std::swap(args.lda_real, args.ldb_real);
      std::swap(args.lda_imag, args.ldb_imag);
      std::swap(args.batch_stride_A, args.batch_stride_B);
      std::swap(args.batch_stride_A_imag, args.batch_stride_B_imag);

      return args;
    }
  };


  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params : UniversalParamsBase<
    ThreadblockSwizzle,
    ThreadblockShape,
    ElementA,
    ElementB,
    ElementC,
    LayoutA,
    LayoutB>
  {
    using ParamsBase = UniversalParamsBase<
      ThreadblockSwizzle,
      ThreadblockShape,
      ElementA,
      ElementB,
      ElementC,
      LayoutA,
      LayoutB>;

    //
    // Data members
    //

    typename Mma::IteratorA::Params params_A_real{};
    typename Mma::IteratorA::Params params_A_imag{};
    typename Mma::IteratorB::Params params_B_real{};
    typename Mma::IteratorB::Params params_B_imag{};
    typename Epilogue::OutputTileIterator::Params params_C_real{};
    typename Epilogue::OutputTileIterator::Params params_C_imag{};
    typename Epilogue::OutputTileIterator::Params params_D_real{};
    typename Epilogue::OutputTileIterator::Params params_D_imag{};
    
    typename EpilogueOutputOp::Params output_op{};

    void * ptr_A_real{nullptr};
    void * ptr_A_imag{nullptr};
    void * ptr_B_real{nullptr};
    void * ptr_B_imag{nullptr};
    void * ptr_C_real{nullptr};
    void * ptr_C_imag{nullptr};
    void * ptr_D_real{nullptr};
    void * ptr_D_imag{nullptr};

    int64_t batch_stride_A{0};
    int64_t batch_stride_B{0};
    int64_t batch_stride_C{0};

    int64_t batch_stride_A_imag{0};
    int64_t batch_stride_B_imag{0};
    int64_t batch_stride_C_imag{0};
    int64_t batch_stride_D_imag{0};

    //
    // Host dispatch API
    //

    /// Default constructor
    Params() = default;

    /// Constructor
    Params(
      Arguments const &args,  /// GEMM application arguments
      int device_sms,         /// Number of SMs on the device
      int sm_occupancy)       /// Kernel SM occupancy (in thread blocks)
    :
      ParamsBase(args, device_sms, sm_occupancy),
      params_A_real(args.lda_real),
      params_A_imag(args.lda_imag),
      params_B_real(args.ldb_real),
      params_B_imag(args.ldb_imag),
      params_C_real(args.ldc_real),
      params_C_imag(args.ldc_imag),
      params_D_real(args.ldd_real),
      params_D_imag(args.ldd_imag),
      output_op(args.epilogue),
      ptr_A_real(const_cast<void *>(args.ptr_A_real)),
      ptr_A_imag(const_cast<void *>(args.ptr_A_imag)),
      ptr_B_real(const_cast<void *>(args.ptr_B_real)),
      ptr_B_imag(const_cast<void *>(args.ptr_B_imag)),
      ptr_C_real(const_cast<void *>(args.ptr_C_real)),
      ptr_C_imag(const_cast<void *>(args.ptr_C_imag)),
      ptr_D_real(args.ptr_D_real),
      ptr_D_imag(args.ptr_D_imag),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C(args.batch_stride_C),
      batch_stride_A_imag(args.batch_stride_A_imag),
      batch_stride_B_imag(args.batch_stride_B_imag),
      batch_stride_C_imag(args.batch_stride_C_imag),
      batch_stride_D_imag(args.batch_stride_D_imag)
    {}

    /// Returns the workspace size (in bytes) needed for this problem geometry
    size_t get_workspace_size() const
    {
      size_t workspace_bytes = ParamsBase::get_workspace_size();
      if (this->mode == GemmUniversalMode::kGemmSplitKParallel)
      {
        // Double the size returned by the base class because we need to
        // accumulate two ElementC components
        workspace_bytes *= 2;
      }

      return workspace_bytes;
    }

    /// Lightweight update given a subset of arguments.
    void update(Arguments const &args)
    {
      ptr_A_real = const_cast<void *>(args.ptr_A_real);
      ptr_A_imag = const_cast<void *>(args.ptr_A_imag);

      ptr_B_real = const_cast<void *>(args.ptr_B_real);
      ptr_B_imag = const_cast<void *>(args.ptr_B_imag);

      ptr_C_real = const_cast<void *>(args.ptr_C_real);
      ptr_C_imag = const_cast<void *>(args.ptr_C_imag);

      ptr_D_real = const_cast<void *>(args.ptr_D_real);
      ptr_D_imag = const_cast<void *>(args.ptr_D_imag);

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      this->batch_stride_D = args.batch_stride_D;
      batch_stride_A_imag = args.batch_stride_A_imag;
      batch_stride_B_imag = args.batch_stride_B_imag;
      batch_stride_C_imag = args.batch_stride_C_imag;
      batch_stride_D_imag = args.batch_stride_D_imag;

      output_op = args.epilogue;
    }
  };


  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

public:

  //
  // Host dispatch API
  //

  /// Determines whether kernel satisfies alignment
  static Status can_implement(Arguments const &args)
  {
    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = args.problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = args.problem_size.m() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = args.problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = args.problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = args.problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = args.problem_size.m() % kAlignmentC;
    }

    if (isAMisaligned || isBMisaligned || isCMisaligned) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

public:

  //
  // Device-only API
  //

  // Factory invocation
  CUTLASS_DEVICE
  static void invoke(
    Params const &params,
    SharedStorage &shared_storage)
  {
    GemmPlanarComplex op;
    op(params, shared_storage);
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

    ElementA *ptr_A_real = static_cast<ElementA *>(params.ptr_A_real);
    ElementA *ptr_A_imag = static_cast<ElementA *>(params.ptr_A_imag);

    ElementB *ptr_B_real = static_cast<ElementB *>(params.ptr_B_real);
    ElementB *ptr_B_imag = static_cast<ElementB *>(params.ptr_B_imag);

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
      ptr_A_real += int64_t(threadblock_tile_offset.k()) * params.batch_stride_A;
      ptr_A_imag += int64_t(threadblock_tile_offset.k()) * params.batch_stride_A_imag;
      ptr_B_real += int64_t(threadblock_tile_offset.k()) * params.batch_stride_B;
      ptr_B_imag += int64_t(threadblock_tile_offset.k()) * params.batch_stride_B_imag;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_A_real = static_cast<ElementA * const *>(params.ptr_A_real)[threadblock_tile_offset.k()];
      ptr_A_imag = static_cast<ElementA * const *>(params.ptr_A_imag)[threadblock_tile_offset.k()];
      ptr_B_real = static_cast<ElementB * const *>(params.ptr_B_real)[threadblock_tile_offset.k()];
      ptr_B_imag = static_cast<ElementB * const *>(params.ptr_B_imag)[threadblock_tile_offset.k()];
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

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A_real(
      params.params_A_real,
      ptr_A_real,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorA iterator_A_imag(
      params.params_A_imag,
      ptr_A_imag,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorB iterator_B_real(
      params.params_B_real,
      ptr_B_real,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    typename Mma::IteratorB iterator_B_imag(
      params.params_B_imag,
      ptr_B_imag,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

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

    // Compute threadblock-scoped matrix multiply-add
    mma(
      gemm_k_iterations, 
      accumulators, 
      iterator_A_real,
      iterator_A_imag,
      iterator_B_real, 
      iterator_B_imag, 
      accumulators);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    ElementC *ptr_C_real = static_cast<ElementC *>(params.ptr_C_real);
    ElementC *ptr_C_imag = static_cast<ElementC *>(params.ptr_C_imag);
    ElementC *ptr_D_real = static_cast<ElementC *>(params.ptr_D_real);
    ElementC *ptr_D_imag = static_cast<ElementC *>(params.ptr_D_imag);

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
      ptr_D_real += threadblock_tile_offset.k() * params.batch_stride_D;
      ptr_D_imag += threadblock_tile_offset.k() * params.batch_stride_D_imag;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C_real += int64_t(threadblock_tile_offset.k()) * params.batch_stride_C;
      ptr_C_imag += int64_t(threadblock_tile_offset.k()) * params.batch_stride_C_imag;
      ptr_D_real += int64_t(threadblock_tile_offset.k()) * params.batch_stride_D;
      ptr_D_imag += int64_t(threadblock_tile_offset.k()) * params.batch_stride_D_imag;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_C_real = static_cast<ElementC * const *>(params.ptr_C_real)[threadblock_tile_offset.k()];
      ptr_C_imag = static_cast<ElementC * const *>(params.ptr_C_imag)[threadblock_tile_offset.k()];
      ptr_D_real = static_cast<ElementC * const *>(params.ptr_D_real)[threadblock_tile_offset.k()];
      ptr_D_imag = static_cast<ElementC * const *>(params.ptr_D_imag)[threadblock_tile_offset.k()];
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C_real(
      params.params_C_real,
      ptr_C_real,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    typename Epilogue::OutputTileIterator iterator_C_imag(
      params.params_C_imag,
      ptr_C_imag,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D_real(
      params.params_D_real,
      ptr_D_real,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    typename Epilogue::OutputTileIterator iterator_D_imag(
      params.params_D_imag,
      ptr_D_imag,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    //
    // Construct epilogue
    //

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C_real = iterator_D_real;
        iterator_C_imag = iterator_D_imag;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }


    // Execute the epilogue operator to update the destination tensor.
    epilogue(
      output_op, 
      iterator_D_real, 
      iterator_D_imag, 
      accumulators, 
      iterator_C_real,
      iterator_C_imag); 
    
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

