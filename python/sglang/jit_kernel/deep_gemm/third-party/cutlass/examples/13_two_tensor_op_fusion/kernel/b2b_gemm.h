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

#include "kernel/b2b_gemm_grouped_problem_visitor.h"
#include "threadblock/grouped_threadblock_swizzle.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

namespace detail {

/// Utility struct for returning the type of the problem visitor used by the swizzling function,
/// if it is a grouped swizzling function, or a default visitor. This is used only for defining
/// the parameters of the problem visitor used in GroupedParams.
template <
  typename B2bMma_,
  typename ThreadblockSwizzle_,
  typename Enable = void
>
struct ProblemVisitorOrDefault;

/// Return a generic problem visitor for GEMM problems
template <
  typename B2bMma_,
  typename ThreadblockSwizzle_
>
struct ProblemVisitorOrDefault<B2bMma_,
                               ThreadblockSwizzle_,
                               typename platform::enable_if<
                                                  ! cutlass::gemm::threadblock::detail::IsGroupedSwizzle<ThreadblockSwizzle_>::value
                                                >::type> {
  using value = B2bGemmGroupedProblemVisitor<typename B2bMma_::Shape,
                                             GroupScheduleMode::kDeviceOnly,
                                             128,
                                             128,
                                             platform::is_same<typename B2bMma_::LayoutC,
                                                               cutlass::layout::ColumnMajor>::value>;
};

/// Return the problem visitor specified by the swizzling function
template <
  typename B2bMma_,
  typename ThreadblockSwizzle_
>
struct ProblemVisitorOrDefault<B2bMma_,
                               ThreadblockSwizzle_,
                               typename platform::enable_if<
                                                  cutlass::gemm::threadblock::detail::IsGroupedSwizzle<ThreadblockSwizzle_>::value
                                                >::type>  {
  using value = typename ThreadblockSwizzle_::ProblemVisitor;
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename B2bMma_,               ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct B2bGemm {

  using B2bMma = B2bMma_;
  using Epilogue = Epilogue_;
  using OutputOp0 = typename B2bMma::OutputOp;
  using OutputOp1 = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA0 = typename B2bMma::IteratorA0::Element;
  using LayoutA0 = typename B2bMma::IteratorA0::Layout;
  using ElementB0 = typename B2bMma::IteratorB0::Element;
  using LayoutB0 = typename B2bMma::IteratorB0::Layout;
  using ElementB1 = typename B2bMma::IteratorB1::Element;
  using LayoutB1 = typename B2bMma::IteratorB1::Layout;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Epilogue::OutputTileIterator::Layout;

  using ScaleBiasData = typename B2bMma::IteratorAccumulatorScaleBias::Element;

  /// Data types needed for higher-level containers. In some cases, a single type must be exposed
  /// despite the B2b GEMM using two GEMMs under the hood. In such cases, we select the values from
  /// the second GEMM (other than for ElementA/ElementB)
  using ElementA = typename B2bMma::IteratorA0::Element;
  using LayoutA = typename B2bMma::IteratorA0::Layout;
  using ElementB = typename B2bMma::IteratorB0::Element;
  using LayoutB = typename B2bMma::IteratorB0::Layout;

  static ComplexTransform const kTransformA = B2bMma::kTransformA;
  static ComplexTransform const kTransformB = B2bMma::kTransformB;
  using Operator = typename B2bMma::Operator0;

  using OperatorClass = typename Operator::OperatorClass;
  using ThreadblockShape = typename B2bMma::Shape0;
  using WarpShape = typename Operator::Shape;
  using InstructionShape = typename Operator::InstructionShape;
  using ArchTag = typename B2bMma::ArchTag;

  static int const kStages = B2bMma::kStages;
  static int const kAlignmentA = B2bMma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = B2bMma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  using Mma = B2bMma;
  using EpilogueOutputOp = OutputOp1;

  /// Warp count (concept: GemmShape)
  using WarpCount0 = typename B2bMma::WarpCount0;
  static int const kThreadCount = 32 * WarpCount0::kCount;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
    GemmCoord problem_size_0{0,0,0};
    GemmCoord problem_size_1{0,0,0};
    typename B2bMma::IteratorA0::TensorRef ref_A0{};
    typename B2bMma::IteratorB0::TensorRef ref_B0{};
    typename Epilogue::OutputTileIterator::TensorRef ref_C0{};
    typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Scale0{};
    typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Bias0{};
    typename B2bMma::IteratorB1::TensorRef ref_B1{};
    typename Epilogue::OutputTileIterator::TensorRef ref_C1{};
    typename Epilogue::OutputTileIterator::TensorRef ref_D1{};
    int64_t batch_stride_A0{0};
    int64_t batch_stride_B0{0};
    int64_t batch_stride_B1{0};
    int64_t batch_stride_C1{0};
    int64_t batch_stride_D1{0};
    int64_t batch_stride_Bias0{0};
    int64_t batch_stride_Scale0{0};
    typename OutputOp0::Params epilogue0 {};
    typename OutputOp1::Params epilogue1 {};
    int batch_count{1};

    //
    // Methods
    //

    /// Default ctor
    Arguments() = default;

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmUniversalMode mode_,
      GemmCoord problem_size_0_,
      GemmCoord problem_size_1_,
      typename B2bMma::IteratorA0::TensorRef ref_A0_,
      typename B2bMma::IteratorB0::TensorRef ref_B0_,
      typename Epilogue::OutputTileIterator::TensorRef ref_C0_,
      typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Scale0_,
      typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Bias0_,
      typename B2bMma::IteratorB1::TensorRef ref_B1_,
      typename Epilogue::OutputTileIterator::TensorRef ref_C1_,
      typename Epilogue::OutputTileIterator::TensorRef ref_D1_,
      int64_t batch_stride_A0_,
      int64_t batch_stride_B0_,
      int64_t batch_stride_B1_,
      int64_t batch_stride_C1_,
      int64_t batch_stride_D1_,
      int64_t batch_stride_Bias0_,
      int64_t batch_stride_Scale0_,
      typename OutputOp0::Params epilogue0_ = typename OutputOp0::Params(),
      typename OutputOp1::Params epilogue1_ = typename OutputOp1::Params(),
      int batch_count_ = 1
    ):
      mode(mode_),
      problem_size_0(problem_size_0_),
      problem_size_1(problem_size_1_),
      ref_A0(ref_A0_),
      ref_B0(ref_B0_),
      ref_C0(ref_C0_),
      ref_Scale0(ref_Scale0_),
      ref_Bias0(ref_Bias0_),
      ref_B1(ref_B1_),
      ref_C1(ref_C1_),
      ref_D1(ref_D1_),
      batch_stride_A0(batch_stride_A0_),
      batch_stride_B0(batch_stride_B0_),
      batch_stride_B1(batch_stride_B1_),
      batch_stride_C1(batch_stride_C1_),
      batch_stride_D1(batch_stride_D1_),
      batch_stride_Bias0(batch_stride_Bias0_),
      batch_stride_Scale0(batch_stride_Scale0_),
      epilogue0(epilogue0_),
      epilogue1(epilogue1_),
      batch_count(batch_count_) {
    }
  };

  // Arguments structure for grouped B2B problems
  struct GroupedArguments {
    GemmCoord* problem_size_0;
    GemmCoord* problem_size_1;
    typename B2bMma::IteratorA0::TensorRef* ref_A0;
    typename B2bMma::IteratorB0::TensorRef* ref_B0;
    typename Epilogue::OutputTileIterator::TensorRef* ref_C0;
    typename B2bMma::IteratorAccumulatorScaleBias::TensorRef* ref_Scale0;
    typename B2bMma::IteratorAccumulatorScaleBias::TensorRef* ref_Bias0;
    typename B2bMma::IteratorB1::TensorRef* ref_B1;
    typename Epilogue::OutputTileIterator::TensorRef* ref_C1;
    typename Epilogue::OutputTileIterator::TensorRef* ref_D1;

    // Epilogue params remain constant across all problems in the group. Thus,
    // the parameter here is not a pointer.
    typename OutputOp0::Params epilogue0;
    typename OutputOp1::Params epilogue1;

    int problem_count;
    int threadblock_count;
    GemmCoord* host_problem_sizes;

    CUTLASS_HOST_DEVICE
    GroupedArguments(
      int problem_count,
      GemmCoord* problem_size_0_,
      GemmCoord* problem_size_1_,
      typename B2bMma::IteratorA0::TensorRef* ref_A0_,
      typename B2bMma::IteratorB0::TensorRef* ref_B0_,
      typename Epilogue::OutputTileIterator::TensorRef* ref_C0_,
      typename B2bMma::IteratorAccumulatorScaleBias::TensorRef* ref_Scale0_,
      typename B2bMma::IteratorAccumulatorScaleBias::TensorRef* ref_Bias0_,
      typename B2bMma::IteratorB1::TensorRef* ref_B1_,
      typename Epilogue::OutputTileIterator::TensorRef* ref_C1_,
      typename Epilogue::OutputTileIterator::TensorRef* ref_D1_,
      typename OutputOp0::Params epilogue0_ = typename OutputOp0::Params(),
      typename OutputOp1::Params epilogue1_ = typename OutputOp1::Params(),
      int threadblock_count = 0
    ) : problem_size_0(problem_size_0_), problem_size_1(problem_size_1_),
        ref_A0(ref_A0_), ref_B0(ref_B0_), ref_C0(ref_C0_),
        ref_Scale0(ref_Scale0_), ref_Bias0(ref_Bias0_), ref_B1(ref_B1_),
        ref_C1(ref_C1_), ref_D1(ref_D1_), epilogue0(epilogue0_), epilogue1(epilogue1_),
        problem_count(problem_count),
        threadblock_count(threadblock_count)
        {}
  };

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;
    cutlass::gemm::GemmCoord problem_size_0{};
    cutlass::gemm::GemmCoord problem_size_1{};
    cutlass::gemm::GemmCoord grid_tiled_shape{};
    int swizzle_log_tile{0};
    typename B2bMma::IteratorA0::Params params_A0{};
    typename B2bMma::IteratorA0::TensorRef ref_A0{};
    typename B2bMma::IteratorB0::Params params_B0{};
    typename B2bMma::IteratorB0::TensorRef ref_B0{};
    typename Epilogue::OutputTileIterator::Params params_C0{};
    typename Epilogue::OutputTileIterator::TensorRef ref_C0{};
    typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Scale0{};
    typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Bias0{};
    typename B2bMma::IteratorB1::Params params_B1{};
    typename B2bMma::IteratorB1::TensorRef ref_B1{};
    typename Epilogue::OutputTileIterator::Params params_C1{};
    typename Epilogue::OutputTileIterator::TensorRef ref_C1{};
    typename Epilogue::OutputTileIterator::Params params_D1{};
    typename Epilogue::OutputTileIterator::TensorRef ref_D1{};
    typename OutputOp0::Params output_op_0{};
    typename OutputOp1::Params output_op_1{};
    int64_t batch_stride_A0{0};
    int64_t batch_stride_B0{0};
    int64_t batch_stride_B1{0};
    int64_t batch_stride_C1{0};
    int64_t batch_stride_D1{0};
    int64_t batch_stride_Bias0{0};
    int64_t batch_stride_Scale0{0};
    int *semaphore = nullptr;
    int gemm_k_iterations_0{0};
    int gemm_k_size_0{0};
    int gemm_k_iterations_1{0};
    int gemm_k_size_1{0};

    //
    // Methods
    //

    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmUniversalMode mode,
      cutlass::gemm::GemmCoord const & problem_size_0,
      cutlass::gemm::GemmCoord const & problem_size_1,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename B2bMma::IteratorA0::TensorRef ref_A0,
      typename B2bMma::IteratorB0::TensorRef ref_B0,
      typename Epilogue::OutputTileIterator::TensorRef ref_C0,
      typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Scale0,
      typename B2bMma::IteratorAccumulatorScaleBias::TensorRef ref_Bias0,
      typename B2bMma::IteratorB1::TensorRef ref_B1,
      typename Epilogue::OutputTileIterator::TensorRef ref_C1,
      typename Epilogue::OutputTileIterator::TensorRef ref_D1,
      int64_t batch_stride_A0,
      int64_t batch_stride_B0,
      int64_t batch_stride_B1,
      int64_t batch_stride_C1,
      int64_t batch_stride_D1,
      int64_t batch_stride_Bias0,
      int64_t batch_stride_Scale0,
      typename OutputOp0::Params output_op_0 = typename OutputOp0::Params(),
      typename OutputOp1::Params output_op_1 = typename OutputOp1::Params(),
      int *workspace = nullptr
    ):
      mode(mode),
      problem_size_0(problem_size_0),
      problem_size_1(problem_size_1),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle::get_log_tile(grid_tiled_shape)),
      params_A0(ref_A0.layout()),
      ref_A0(ref_A0),
      params_B0(ref_B0.layout()),
      ref_B0(ref_B0),
      params_C0(ref_C0.layout()),
      ref_C0(ref_C0),
      ref_Scale0(ref_Scale0),
      ref_Bias0(ref_Bias0),
      params_B1(ref_B1.layout()),
      ref_B1(ref_B1),
      params_C1(ref_C1.layout()),
      ref_C1(ref_C1),
      params_D1(ref_D1.layout()),
      ref_D1(ref_D1),
      batch_stride_A0(batch_stride_A0),
      batch_stride_B0(batch_stride_B0),
      batch_stride_B1(batch_stride_B1),
      batch_stride_C1(batch_stride_C1),
      batch_stride_D1(batch_stride_D1),
      batch_stride_Bias0(batch_stride_Bias0),
      batch_stride_Scale0(batch_stride_Scale0),
      output_op_0(output_op_0),
      output_op_1(output_op_1) {

      int total_gemm_k_iterations_0 = (problem_size_0.k() + B2bMma::Shape0::kK - 1) / B2bMma::Shape0::kK;
      int gemm_k_iterations_0 = (total_gemm_k_iterations_0 + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      gemm_k_size_0 = gemm_k_iterations_0 * B2bMma::Shape0::kK;
      int total_gemm_k_iterations_1 = (problem_size_1.k() + B2bMma::Shape1::kK - 1) / B2bMma::Shape1::kK;
      int gemm_k_iterations_1 = (total_gemm_k_iterations_1 + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      gemm_k_size_1 = gemm_k_iterations_1 * B2bMma::Shape1::kK;

    semaphore = workspace;
    }
  };

  struct GroupedParams {
    cutlass::gemm::GemmCoord* problem_size_0;
    cutlass::gemm::GemmCoord* problem_size_1;
    cutlass::gemm::GemmCoord* grid_tiled_shape;
    typename B2bMma::IteratorA0::TensorRef* ref_A0;
    typename B2bMma::IteratorB0::TensorRef* ref_B0;
    typename Epilogue::OutputTileIterator::TensorRef* ref_C0;
    typename B2bMma::IteratorAccumulatorScaleBias::TensorRef* ref_Scale0;
    typename B2bMma::IteratorAccumulatorScaleBias::TensorRef* ref_Bias0;
    typename B2bMma::IteratorB1::TensorRef* ref_B1;
    typename Epilogue::OutputTileIterator::TensorRef* ref_C1;
    typename Epilogue::OutputTileIterator::TensorRef* ref_D1;

    // Epilogue params remain constant across all problems in the group. Thus,
    // the parameter here is not a pointer.
    typename OutputOp0::Params output_op_0;
    typename OutputOp1::Params output_op_1;

    using ProblemVisitor = typename detail::ProblemVisitorOrDefault<B2bMma, ThreadblockSwizzle>::value;
    typename ProblemVisitor::Params problem_visitor;
    int threadblock_count;
    int* workspace;

    CUTLASS_HOST_DEVICE
    GroupedParams() {}

    CUTLASS_HOST_DEVICE
    GroupedParams(
      GroupedArguments const &args,
      void *workspace = nullptr,
      int tile_count = 0
    ) :
        problem_size_0(args.problem_size_0), problem_size_1(args.problem_size_1),
        ref_A0(args.ref_A0), ref_B0(args.ref_B0), ref_C0(args.ref_C0),
        ref_Scale0(args.ref_Scale0), ref_Bias0(args.ref_Bias0), ref_B1(args.ref_B1), ref_C1(args.ref_C1), ref_D1(args.ref_D1),
        output_op_0(args.epilogue0), output_op_1(args.epilogue1),
        problem_visitor(args.problem_size_0, args.problem_size_1, args.problem_count, workspace, tile_count),
        threadblock_count(args.threadblock_count),
        workspace(reinterpret_cast<int*>(workspace)) {}

    CUTLASS_HOST_DEVICE
    void transpose() {
      // Only row-major outputs are currently supported, so no transpose is performed
    }

    /// Returns non-grouped parameters to be used as input to the kernel-level
    /// operator for the problem indicated by problem_visitor.
    CUTLASS_HOST_DEVICE
    Params to_single_params(const ProblemVisitor& problem_visitor) const {
      GemmCoord problem_size0 = problem_visitor.problem_size0();
      GemmCoord problem_size1 = problem_visitor.problem_size1();
      int32_t idx = problem_visitor.problem_index();
      GemmCoord grid_shape = problem_visitor.grid_shape(problem_size1);

      return Params(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size0,
        problem_size1,
        grid_shape,
        ref_A0[idx],
        ref_B0[idx],
        ref_C0[idx],
        ref_Scale0[idx],
        ref_Bias0[idx],
        ref_B1[idx],
        ref_C1[idx],
        ref_D1[idx],
        0, 0, 0, 0, 0, 0, 0, // Batched B2B GEMMs within the grouped kernel are currently unsupported
        output_op_0,
        output_op_1,
        workspace
      );
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
  B2bGemm() { }

  /// Determines whether kernel satisfies alignment
    static Status can_implement(
      cutlass::gemm::GemmCoord const & problem_size_0,
      cutlass::gemm::GemmCoord const & problem_size_1,
      typename B2bMma::IteratorA0::TensorRef ref_A0,
      typename B2bMma::IteratorB0::TensorRef ref_B0,
      typename Epilogue::OutputTileIterator::TensorRef ref_C0,
      typename B2bMma::IteratorB1::TensorRef ref_B1,
      typename Epilogue::OutputTileIterator::TensorRef ref_C1,
      typename Epilogue::OutputTileIterator::TensorRef ref_D1) {

    static int const kAlignmentA = B2bMma::IteratorA0::AccessType::kElements;
    static int const kAlignmentB = B2bMma::IteratorB0::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A0, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B0, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C0, kAlignmentC)) {
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

    if ((problem_size_0.m() % kAlignmentA) || (problem_size_0.k() % kAlignmentA) ||
      (problem_size_0.n() % kAlignmentB) || (problem_size_0.k() % kAlignmentB) ||
      (problem_size_0.m() % kAlignmentC) || (problem_size_0.n() % kAlignmentC) ||
      (problem_size_1.m() % kAlignmentA) || (problem_size_1.k() % kAlignmentA) ||
      (problem_size_1.n() % kAlignmentB) || (problem_size_1.k() % kAlignmentB) ||
      (problem_size_1.m() % kAlignmentC) || (problem_size_1.n() % kAlignmentC)) {

      return Status::kErrorMisalignedOperand;
    }

    // Determine if fusion sizes are valid
    if(problem_size_0.m() != problem_size_1.m())
      return Status::kErrorInvalidProblem;

    if(problem_size_0.n() != problem_size_1.k())
      return Status::kErrorInvalidProblem;

    if(problem_size_0.n() > B2bMma::Shape0::kN)
      return Status::kErrorInvalidProblem;

    if(problem_size_1.n() > B2bMma::Shape1::kN)
      return Status::kErrorInvalidProblem;

    return Status::kSuccess;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;
    run_with_swizzle(params, shared_storage, threadblock_swizzle);
  }

  /// Executes one GEMM with an externally-provided swizzling function
  CUTLASS_DEVICE
  void run_with_swizzle(Params const &params, SharedStorage &shared_storage, ThreadblockSwizzle& threadblock_swizzle) {

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    ElementA0 *ptr_A0 = static_cast<ElementA0 *>(params.ref_A0.data());
    ElementB0 *ptr_B0 = static_cast<ElementB0 *>(params.ref_B0.data());
    ElementB1 *ptr_B1 = static_cast<ElementB1 *>(params.ref_B1.data());

    ScaleBiasData *ptr_Bias0 = static_cast<ScaleBiasData *>(params.ref_Bias0.data());
    ScaleBiasData *ptr_Scale0 = static_cast<ScaleBiasData *>(params.ref_Scale0.data());

    int offset_k_0 = 0;
    int offset_k_1 = 0;

    int problem_size_k_0 = params.problem_size_0.k();
    int problem_size_k_1 = params.problem_size_1.k();

    if (params.mode == GemmUniversalMode::kGemm) {

      // Problem size is a function of threadblock index in the K dimension
      problem_size_k_0 = min(
        problem_size_k_0,
        (threadblock_tile_offset.k() + 1) * params.gemm_k_size_0);

      // Problem size is a function of threadblock index in the K dimension
      problem_size_k_1 = min(
        problem_size_k_1,
        (threadblock_tile_offset.k() + 1) * params.gemm_k_size_1);

      offset_k_0 = threadblock_tile_offset.k() * params.gemm_k_size_0;
      offset_k_1 = threadblock_tile_offset.k() * params.gemm_k_size_1;
    }

    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_A0 += threadblock_tile_offset.k() * params.batch_stride_A0;
      ptr_B0 += threadblock_tile_offset.k() * params.batch_stride_B0;
      ptr_B1 += threadblock_tile_offset.k() * params.batch_stride_B1;
      ptr_Bias0 += threadblock_tile_offset.k() * params.batch_stride_Bias0;
      ptr_Scale0 += threadblock_tile_offset.k() * params.batch_stride_Scale0;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A0{
      threadblock_tile_offset.m() * B2bMma::Shape0::kM,
      offset_k_0,
    };

    cutlass::MatrixCoord tb_offset_B0{
      offset_k_0,
      threadblock_tile_offset.n() * B2bMma::Shape0::kN
    };

    cutlass::MatrixCoord tb_offset_B1{
      offset_k_1,
      threadblock_tile_offset.n() * B2bMma::Shape1::kN
    };

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations_0 = (problem_size_k_0 - tb_offset_A0.column() + B2bMma::Shape0::kK - 1) / B2bMma::Shape0::kK;

    // Compute threadblock-scoped matrix multiply-add
    // int gemm_k_iterations_1 = (problem_size_k_1 - tb_offset_B1.row() + B2bMma::Shape1::kK - 1) / B2bMma::Shape1::kK;


    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename B2bMma::IteratorA0 iterator_A0(
      params.params_A0,
      ptr_A0,
      {params.problem_size_0.m(), problem_size_k_0},
      thread_idx,
      tb_offset_A0);

    typename B2bMma::IteratorB0 iterator_B0(
      params.params_B0,
      ptr_B0,
      {problem_size_k_0, params.problem_size_0.n()},
      thread_idx,
      tb_offset_B0);

    typename B2bMma::IteratorB1 iterator_B1(
      params.params_B1,
      ptr_B1,
      {problem_size_k_1, params.problem_size_1.n()},
      thread_idx,
      tb_offset_B1);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    // Construct iterators to accumulator scale/bias vector
    typename B2bMma::IteratorAccumulatorScaleBias iterator_Scale0(
      ptr_Scale0,
      {1, params.problem_size_0.n()},
      thread_idx,
      warp_idx,
      MatrixCoord(
        0, threadblock_tile_offset.n() * B2bMma::Shape0::kN
      )
    );

    typename B2bMma::IteratorAccumulatorScaleBias iterator_Bias0(
      ptr_Bias0,
      {1, params.problem_size_0.n()},
      thread_idx,
      warp_idx,
      MatrixCoord(
        0, threadblock_tile_offset.n() * B2bMma::Shape0::kN
      )
    );

    //
    // Main loop
    //

    OutputOp0 output_op_0(params.output_op_0);

    if (cutlass::gemm::threadblock::detail::IsGroupedSwizzle<ThreadblockSwizzle>::value) {
      // Wait for all threads to finish their epilogue phases from the previous tile.
      __syncthreads();
    }

    // Construct thread-scoped matrix multiply
    B2bMma b2bMma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx, params.problem_size_0.n());

    typename B2bMma::FragmentC0 src_accum;
    typename B2bMma::FragmentC1 accumulators;

    src_accum.clear();
    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    b2bMma(gemm_k_iterations_0, accumulators, iterator_A0, iterator_B0,
      iterator_Scale0, iterator_Bias0, iterator_B1, src_accum, output_op_0);

    //
    // Epilogue
    //

    OutputOp1 output_op_1(params.output_op_1);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * B2bMma::Shape1::kM,
      threadblock_tile_offset.n() * B2bMma::Shape1::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    ElementC *ptr_C1 = static_cast<ElementC *>(params.ref_C1.data());
    ElementC *ptr_D1 = static_cast<ElementC *>(params.ref_D1.data());

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if (params.mode == GemmUniversalMode::kGemm) {
      // If performing a reduction via split-K, fetch the initial synchronization

      if (params.grid_tiled_shape.k() > 1) {
        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is currently updating
        output_op_1.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
      }
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C1 += threadblock_tile_offset.k() * params.batch_stride_C1;
      ptr_D1 += threadblock_tile_offset.k() * params.batch_stride_D1;
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C1(
      params.params_C1,
      ptr_C1,
      params.problem_size_1.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D1(
      params.params_D1,
      ptr_D1,
      params.problem_size_1.mn(),
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
        iterator_C1 = iterator_D1;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op_1, iterator_D1, accumulators, iterator_C1);

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

      __threadfence();
      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass
