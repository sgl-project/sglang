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
struct GemmPlanarComplexArray {
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

    int const *ptr_M{nullptr};
    int const *ptr_N{nullptr};
    int const *ptr_K{nullptr};

    void const * const * ptr_A_real{nullptr};
    void const * const * ptr_A_imag{nullptr};

    void const * const * ptr_B_real{nullptr};
    void const * const * ptr_B_imag{nullptr};

    void const * const * ptr_C_real{nullptr};
    void const * const * ptr_C_imag{nullptr};

    void * const * ptr_D_real{nullptr};
    void * const * ptr_D_imag{nullptr};

    typename LayoutA::Stride::Index lda_real{};
    typename LayoutA::Stride::Index lda_imag{};
    typename LayoutB::Stride::Index ldb_real{};
    typename LayoutB::Stride::Index ldb_imag{};
    typename LayoutC::Stride::Index ldc_real{};
    typename LayoutC::Stride::Index ldc_imag{};
    typename LayoutC::Stride::Index ldd_real{};
    typename LayoutC::Stride::Index ldd_imag{};

    //
    // Methods
    //

    Arguments() = default;

    /// constructs an arguments structure
    Arguments(
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      int const *ptr_M,
      int const *ptr_N,
      int const *ptr_K,
      void const * const * ptr_A_real,
      void const * const * ptr_A_imag,
      void const * const * ptr_B_real,
      void const * const * ptr_B_imag,
      void const * const * ptr_C_real,
      void const * const * ptr_C_imag,
      void * const * ptr_D_real,
      void * const * ptr_D_imag,
      typename LayoutA::Stride::Index lda_real,
      typename LayoutA::Stride::Index lda_imag,
      typename LayoutB::Stride::Index ldb_real,
      typename LayoutB::Stride::Index ldb_imag,
      typename LayoutC::Stride::Index ldc_real,
      typename LayoutC::Stride::Index ldc_imag,
      typename LayoutC::Stride::Index ldd_real,
      typename LayoutC::Stride::Index ldd_imag)
    :
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue(epilogue),
      ptr_M(ptr_M),
      ptr_N(ptr_N),
      ptr_K(ptr_K),
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
      ldd_imag(ldd_imag)
    {}

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);
      
      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_M, args.ptr_N);
      std::swap(args.ptr_A_real, args.ptr_B_real);
      std::swap(args.ptr_A_imag, args.ptr_B_imag);
      std::swap(args.lda_real, args.ldb_real);
      std::swap(args.lda_imag, args.ldb_imag);

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

    int const *ptr_M{nullptr};
    int const *ptr_N{nullptr};
    int const *ptr_K{nullptr};

    void const * const * ptr_A_real{nullptr};
    void const * const * ptr_A_imag{nullptr};
    void const * const * ptr_B_real{nullptr};
    void const * const * ptr_B_imag{nullptr};
    void const * const * ptr_C_real{nullptr};
    void const * const * ptr_C_imag{nullptr};
    void * const * ptr_D_real{nullptr};
    void * const * ptr_D_imag{nullptr};

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
      ptr_M(args.ptr_M),
      ptr_N(args.ptr_N),
      ptr_K(args.ptr_K),
      params_A_real(args.lda_real),
      params_A_imag(args.lda_imag),
      params_B_real(args.ldb_real),
      params_B_imag(args.ldb_imag),
      params_C_real(args.ldc_real),
      params_C_imag(args.ldc_imag),
      params_D_real(args.ldd_real),
      params_D_imag(args.ldd_imag),
      output_op(args.epilogue),
      ptr_A_real(args.ptr_A_real),
      ptr_A_imag(args.ptr_A_imag),
      ptr_B_real(args.ptr_B_real),
      ptr_B_imag(args.ptr_B_imag),
      ptr_C_real(args.ptr_C_real),
      ptr_C_imag(args.ptr_C_imag),
      ptr_D_real(args.ptr_D_real),
      ptr_D_imag(args.ptr_D_imag)
    {}

    /// Lightweight update given a subset of arguments.
    void update(Arguments const &args)
    {
      ptr_M = args.ptr_M;
      ptr_N = args.ptr_N;
      ptr_K = args.ptr_K;

      ptr_A_real = args.ptr_A_real;
      ptr_A_imag = args.ptr_A_imag;

      ptr_B_real = args.ptr_B_real;
      ptr_B_imag = args.ptr_B_imag;

      ptr_C_real = args.ptr_C_real;
      ptr_C_imag = args.ptr_C_imag;

      ptr_D_real = args.ptr_D_real;
      ptr_D_imag = args.ptr_D_imag;

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
  static Status can_implement(Arguments const &args) {

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
    GemmPlanarComplexArray op;
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

    int batch_idx = threadblock_tile_offset.k();

    int problem_size_m = params.problem_size.m();
    int problem_size_n = params.problem_size.n();
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A_real = static_cast<ElementA *>(const_cast<void *>(params.ptr_A_real[batch_idx]));
    ElementA *ptr_A_imag = static_cast<ElementA *>(const_cast<void *>(params.ptr_A_imag[batch_idx]));

    ElementB *ptr_B_real = static_cast<ElementB *>(const_cast<void *>(params.ptr_B_real[batch_idx]));
    ElementB *ptr_B_imag = static_cast<ElementB *>(const_cast<void *>(params.ptr_B_imag[batch_idx]));

    //
    // If pointers for problem sizes are specified, these are loaded from global memory
    //

    if (params.ptr_M) {
      problem_size_m = params.ptr_M[batch_idx];
    }

    if (params.ptr_N) {
      problem_size_n = params.ptr_N[batch_idx];
    }

    if (params.ptr_K) {
      problem_size_k = params.ptr_K[batch_idx];
    }

    int const kBlockCountM = (problem_size_m + Mma::Shape::kM - 1) / Mma::Shape::kM;
    int const kBlockCountN = (problem_size_n + Mma::Shape::kN - 1) / Mma::Shape::kN;
        
    int const kGemmKIterations = (problem_size_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    //
    // Each threadblock loops over the logical problem size which the kernel may have discovered
    // after the grid is launched.
    //

    CUTLASS_PRAGMA_NO_UNROLL
    for (int block_m = threadblock_tile_offset.m(); 
      block_m < kBlockCountM; 
      block_m += params.grid_tiled_shape.m()) {

      CUTLASS_PRAGMA_NO_UNROLL
      for (int block_n = threadblock_tile_offset.n(); 
        block_n < kBlockCountN; 
        block_n += params.grid_tiled_shape.n()) {

        //
        // Compute indices within threadblock and warp.
        //
        int thread_idx = threadIdx.x;

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = canonical_warp_idx_sync();
        int lane_idx = threadIdx.x % 32;
    
        //
        // Proceed with regular GEMM logic.
        //

        // Compute initial location in logical coordinates
        cutlass::MatrixCoord tb_offset_A{ block_m * Mma::Shape::kM, 0};
        cutlass::MatrixCoord tb_offset_B{ 0, block_n * Mma::Shape::kN };

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A_real(
          params.params_A_real,
          ptr_A_real,
          {problem_size_m, problem_size_k},
          thread_idx,
          tb_offset_A);

        typename Mma::IteratorA iterator_A_imag(
          params.params_A_imag,
          ptr_A_imag,
          {problem_size_m, problem_size_k},
          thread_idx,
          tb_offset_A);

        typename Mma::IteratorB iterator_B_real(
          params.params_B_real,
          ptr_B_real,
          {problem_size_k, problem_size_n},
          thread_idx,
          tb_offset_B);
  
        typename Mma::IteratorB iterator_B_imag(
          params.params_B_imag,
          ptr_B_imag,
          {problem_size_k, problem_size_n},
          thread_idx,
          tb_offset_B);

        //
        // Main loop
        //

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentC accumulators;

        accumulators.clear();

        // Compute threadblock-scoped matrix multiply-add
        mma(
          kGemmKIterations, 
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

        //assume identity swizzle
        MatrixCoord threadblock_offset(
          block_m * Mma::Shape::kM,
          block_n * Mma::Shape::kN
        );

        ElementC *ptr_C_real = static_cast<ElementC *>(const_cast<void *>(params.ptr_C_real[batch_idx]));
        ElementC *ptr_C_imag = static_cast<ElementC *>(const_cast<void *>(params.ptr_C_imag[batch_idx]));
        ElementC *ptr_D_real = static_cast<ElementC *>(params.ptr_D_real[batch_idx]);
        ElementC *ptr_D_imag = static_cast<ElementC *>(params.ptr_D_imag[batch_idx]);

        // Tile iterator loading from source tensor.
        typename Epilogue::OutputTileIterator iterator_C_real(
          params.params_C_real,
          ptr_C_real,
          {problem_size_m, problem_size_n},
          thread_idx,
          threadblock_offset
        );

        typename Epilogue::OutputTileIterator iterator_C_imag(
          params.params_C_imag,
          ptr_C_imag,
          {problem_size_m, problem_size_n},
          thread_idx,
          threadblock_offset
        );

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D_real(
          params.params_D_real,
          ptr_D_real,
          {problem_size_m, problem_size_n},
          thread_idx,
          threadblock_offset
        );

        typename Epilogue::OutputTileIterator iterator_D_imag(
          params.params_D_imag,
          ptr_D_imag,
          {problem_size_m, problem_size_n},
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

        // Execute the epilogue operator to update the destination tensor.
        epilogue(
          output_op, 
          iterator_D_real, 
          iterator_D_imag, 
          accumulators, 
          iterator_C_real,
          iterator_C_imag); 


      } // for block_n
    } // for block_m
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

