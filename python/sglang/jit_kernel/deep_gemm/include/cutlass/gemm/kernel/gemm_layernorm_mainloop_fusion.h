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
    \brief Template for a multistage GEMM kernel with layernorm operations fused in mainloop.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"
#include "cutlass/gemm/kernel/params_universal_base.h"

#include "cutlass/layout/matrix.h"

#include "cutlass/trace.h"

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
struct GemmLayernormMainloopFusion {
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

  using ElementScaleBias = typename Mma::IteratorVarMean::Element;
  using LayoutScaleBias = typename Mma::IteratorVarMean::Layout;

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
  struct Arguments : UniversalArgumentsBase
  {
    //
    // Data members
    //

    typename EpilogueOutputOp::Params epilogue{};

    void const * ptr_A{nullptr};
    void const * ptr_B{nullptr};
    void const * ptr_var{nullptr};
    void const * ptr_mean{nullptr};
    void const * ptr_gamma{nullptr};
    void const * ptr_beta{nullptr};
    void const * ptr_C{nullptr};
    void * ptr_D{nullptr};

    int64_t batch_stride_A{0};
    int64_t batch_stride_B{0};
    int64_t batch_stride_var{0};
    int64_t batch_stride_mean{0};
    int64_t batch_stride_gamma{0};
    int64_t batch_stride_beta{0};
    int64_t batch_stride_C{0};

    typename LayoutA::Stride stride_a{};
    typename LayoutB::Stride stride_b{};
    typename LayoutScaleBias::Stride stride_var{};
    typename LayoutScaleBias::Stride stride_mean{};
    typename LayoutScaleBias::Stride stride_gamma{};
    typename LayoutScaleBias::Stride stride_beta{};
    typename LayoutC::Stride stride_c{};
    typename LayoutC::Stride stride_d{};

    typename LayoutA::Stride::LongIndex lda{};
    typename LayoutB::Stride::LongIndex ldb{};
    typename LayoutScaleBias::Stride::LongIndex ld_var{};
    typename LayoutScaleBias::Stride::LongIndex ld_mean{};
    typename LayoutScaleBias::Stride::LongIndex ld_gamma{};
    typename LayoutScaleBias::Stride::LongIndex ld_beta{};
    typename LayoutC::Stride::LongIndex ldc{};
    typename LayoutC::Stride::LongIndex ldd{};

    int const * ptr_gather_A_indices{nullptr};
    int const * ptr_gather_B_indices{nullptr};
    int const * ptr_scatter_D_indices{nullptr};

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
      void const * ptr_var,
      void const * ptr_mean,
      void const * ptr_gamma,
      void const * ptr_beta,
      void const * ptr_C,
      void * ptr_D,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_var,
      int64_t batch_stride_mean,
      int64_t batch_stride_gamma,
      int64_t batch_stride_beta,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      typename LayoutA::Stride stride_a,
      typename LayoutB::Stride stride_b,
      typename LayoutScaleBias::Stride stride_var,
      typename LayoutScaleBias::Stride stride_mean,
      typename LayoutScaleBias::Stride stride_gamma,
      typename LayoutScaleBias::Stride stride_beta,
      typename LayoutC::Stride stride_c,
      typename LayoutC::Stride stride_d,
      int const *ptr_gather_A_indices = nullptr,
      int const *ptr_gather_B_indices = nullptr,
      int const *ptr_scatter_D_indices = nullptr)
    :
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue(epilogue), 
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D),
      ptr_var(ptr_var), ptr_mean(ptr_mean), 
      ptr_gamma(ptr_gamma), ptr_beta(ptr_beta), 
      batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B), batch_stride_C(batch_stride_C),
      batch_stride_var(batch_stride_var), batch_stride_mean(batch_stride_mean),
      batch_stride_gamma(batch_stride_gamma), batch_stride_beta(batch_stride_beta),
      lda(0), ldb(0), ldc(0), ldd(0),
      ld_var(0), ld_mean(0),
      ld_gamma(0), ld_beta(0),
      stride_a(stride_a), stride_b(stride_b), stride_c(stride_c), stride_d(stride_d),
      stride_var(stride_var), stride_mean(stride_mean),
      stride_gamma(stride_gamma), stride_beta(stride_beta),
      ptr_gather_A_indices(ptr_gather_A_indices), ptr_gather_B_indices(ptr_gather_B_indices),
      ptr_scatter_D_indices(ptr_scatter_D_indices)
    {
      CUTLASS_TRACE_HOST("GemmUniversal::Arguments::Arguments() - problem_size: " << problem_size);
    }

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_var,
      void const * ptr_mean,
      void const * ptr_gamma,
      void const * ptr_beta,
      void const * ptr_C,
      void * ptr_D,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_var,
      int64_t batch_stride_mean,
      int64_t batch_stride_gamma,
      int64_t batch_stride_beta,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      typename LayoutA::Stride::LongIndex lda,
      typename LayoutB::Stride::LongIndex ldb,
      typename LayoutScaleBias::Stride::LongIndex ld_var,
      typename LayoutScaleBias::Stride::LongIndex ld_mean,
      typename LayoutScaleBias::Stride::LongIndex ld_gamma,
      typename LayoutScaleBias::Stride::LongIndex ld_beta,
      typename LayoutC::Stride::LongIndex ldc,
      typename LayoutC::Stride::LongIndex ldd,
      int const *ptr_gather_A_indices = nullptr,
      int const *ptr_gather_B_indices = nullptr,
      int const *ptr_scatter_D_indices = nullptr)
    :
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue(epilogue), 
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D),
      ptr_var(ptr_var), ptr_mean(ptr_mean), 
      ptr_gamma(ptr_gamma), ptr_beta(ptr_beta), 
      batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B), batch_stride_C(batch_stride_C),
      batch_stride_var(batch_stride_var), batch_stride_mean(batch_stride_mean),
      batch_stride_gamma(batch_stride_gamma), batch_stride_beta(batch_stride_beta),
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd),
      ld_var(ld_var), ld_mean(ld_mean),
      ld_gamma(ld_gamma), ld_beta(ld_beta),
      ptr_gather_A_indices(ptr_gather_A_indices), ptr_gather_B_indices(ptr_gather_B_indices),
      ptr_scatter_D_indices(ptr_scatter_D_indices)
    {
      stride_a = make_Coord(lda);
      stride_b = make_Coord(ldb);
      stride_c = make_Coord(ldc);
      stride_d = make_Coord(ldd);
      stride_var = make_Coord(ld_var);
      stride_mean = make_Coord(ld_mean);
      stride_gamma = make_Coord(ld_gamma);
      stride_beta = make_Coord(ld_beta);
      CUTLASS_TRACE_HOST("GemmUniversal::Arguments::Arguments() - problem_size: " << problem_size);
    }

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);
      
      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_A, args.ptr_B);
      std::swap(args.lda, args.ldb);
      std::swap(args.stride_a, args.stride_b);
      std::swap(args.batch_stride_A, args.batch_stride_B);
      std::swap(args.ptr_gather_A_indices, args.ptr_gather_B_indices);

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

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    
    typename EpilogueOutputOp::Params output_op;

    void * ptr_A;
    void * ptr_B;
    void * ptr_var;
    void * ptr_mean;
    void * ptr_gamma;
    void * ptr_beta;
    void * ptr_C;
    void * ptr_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_var;
    int64_t batch_stride_mean;
    int64_t batch_stride_gamma;
    int64_t batch_stride_beta;
    int64_t batch_stride_C;

    int * ptr_gather_A_indices;
    int * ptr_gather_B_indices;
    int * ptr_scatter_D_indices;

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
      params_A(args.lda ? make_Coord_with_padding<LayoutA::kStrideRank>(args.lda) : args.stride_a),
      params_B(args.ldb ? make_Coord_with_padding<LayoutB::kStrideRank>(args.ldb) : args.stride_b),
      params_C(args.ldc ? make_Coord_with_padding<LayoutC::kStrideRank>(args.ldc) : args.stride_c),
      params_D(args.ldd ? make_Coord_with_padding<LayoutC::kStrideRank>(args.ldd) : args.stride_d),
      output_op(args.epilogue),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_var(const_cast<void *>(args.ptr_var)),
      ptr_mean(const_cast<void *>(args.ptr_mean)),
      ptr_gamma(const_cast<void *>(args.ptr_gamma)),
      ptr_beta(const_cast<void *>(args.ptr_beta)),
      ptr_C(const_cast<void *>(args.ptr_C)),
      ptr_D(args.ptr_D),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_var(args.batch_stride_var),
      batch_stride_mean(args.batch_stride_mean),
      batch_stride_gamma(args.batch_stride_gamma),
      batch_stride_beta(args.batch_stride_beta),
      batch_stride_C(args.batch_stride_C),
      ptr_gather_A_indices(const_cast<int *>(args.ptr_gather_A_indices)),
      ptr_gather_B_indices(const_cast<int *>(args.ptr_gather_B_indices)),
      ptr_scatter_D_indices(const_cast<int *>(args.ptr_scatter_D_indices))
    {}

    /// Lightweight update given a subset of arguments.
    void update(Arguments const &args)
    {
      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_var = const_cast<void *>(args.ptr_var);
      ptr_mean = const_cast<void *>(args.ptr_mean);
      ptr_gamma = const_cast<void *>(args.ptr_gamma);
      ptr_beta = const_cast<void *>(args.ptr_beta);
      ptr_C = const_cast<void *>(args.ptr_C);
      ptr_D = args.ptr_D;

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      batch_stride_var = args.batch_stride_var;
      batch_stride_mean = args.batch_stride_mean;
      batch_stride_gamma = args.batch_stride_gamma;
      batch_stride_beta = args.batch_stride_beta;
      this->batch_stride_D = args.batch_stride_D;

      ptr_gather_A_indices = const_cast<int *>(args.ptr_gather_A_indices);
      ptr_gather_B_indices = const_cast<int *>(args.ptr_gather_B_indices);
      ptr_scatter_D_indices = const_cast<int *>(args.ptr_scatter_D_indices);

      output_op = args.epilogue;
      
      CUTLASS_TRACE_HOST("GemmUniversal::Params::update()");
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
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size) {

    CUTLASS_TRACE_HOST("GemmUniversal::can_implement()");

    static int const kAlignmentA = (platform::is_same<LayoutA,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<LayoutA,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = (platform::is_same<LayoutB,
                                                      layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<LayoutB,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = (platform::is_same<LayoutC,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<LayoutC,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Epilogue::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value
            || platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    }

    if (isAMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for A operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isBMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for B operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isCMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for C operand");
      return Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
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
    GemmLayernormMainloopFusion op;
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

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A,
      params.ptr_gather_A_indices);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B,
      params.ptr_gather_B_indices);

    // Construct iterators to A var/mean vector
    typename Mma::IteratorVarMean iterator_var_mean(
      params.problem_size.m(),
      static_cast<ElementScaleBias const *>(params.ptr_var),
      static_cast<ElementScaleBias const *>(params.ptr_mean),
      thread_idx,
      MatrixCoord(0, (threadblock_tile_offset.m() * Mma::Shape::kM))
    );

    // Construct iterators to A scale/bias vector
    typename Mma::IteratorGammaBeta iterator_gamma_beta(
      problem_size_k,
      static_cast<ElementScaleBias const *>(params.ptr_gamma),
      static_cast<ElementScaleBias const *>(params.ptr_beta),
      thread_idx,
      MatrixCoord(
        0, (threadblock_tile_offset.k() * Mma::Shape::kK)
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

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(
      gemm_k_iterations, 
      accumulators, 
      iterator_A, 
      iterator_B,
      iterator_var_mean,
      iterator_gamma_beta, 
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

    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C); 
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
      ptr_C += threadblock_tile_offset.k() * params.batch_stride_C;
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_C = static_cast<ElementC * const *>(params.ptr_C)[threadblock_tile_offset.k()];
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[threadblock_tile_offset.k()];
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      ptr_C,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.ptr_scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      ptr_D,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.ptr_scatter_D_indices
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
