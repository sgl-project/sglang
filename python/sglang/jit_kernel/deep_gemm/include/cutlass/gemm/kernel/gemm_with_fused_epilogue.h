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
    \brief Gemm kernel with fused reduction operation.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"
#include "cutlass/gemm/kernel/params_universal_base.h"
#include "cutlass/subbyte_reference.h"
#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool IsSingleSource = Epilogue_::kIsSingleSource
>
struct GemmWithFusedEpilogue;

// GemmWithFusedEpilogue with two sources
template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmWithFusedEpilogue<Mma_, Epilogue_, ThreadblockSwizzle_, false> {
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
  static int const kSplitKAlignment = const_max(
    128 / sizeof_bits<ElementA>::value,
    128 / sizeof_bits<ElementB>::value
  );

  //
  // Structures
  //

  /// Argument structure
  struct Arguments : UniversalArgumentsBase{

    //
    // Data members
    //

    typename EpilogueOutputOp::Params epilogue;

    void const * ptr_A;
    void const * ptr_B;
    void const * ptr_C1;
    void const * ptr_C2;
    void * ptr_D;

    void * ptr_Vector;
    void * ptr_Tensor;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C1;
    int64_t batch_stride_C2;
    int64_t batch_stride_Vector;
    int64_t batch_stride_Tensor;

    typename LayoutA::Stride::Index lda;
    typename LayoutB::Stride::Index ldb;
    typename LayoutC::Stride::Index ldc1;
    typename LayoutC::Stride::Index ldc2;
    typename LayoutC::Stride::Index ldd;
    typename LayoutC::Stride::Index ldr;
    typename LayoutC::Stride::Index ldt;

    //
    // Methods
    //

    Arguments():
      ptr_A(nullptr),
      ptr_B(nullptr),
      ptr_C1(nullptr),
      ptr_C2(nullptr),
      ptr_D(nullptr)
    {}

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C1,
      void const * ptr_C2,
      void * ptr_D,
      void * ptr_Vector,
      void * ptr_Tensor,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C1,
      int64_t batch_stride_C2,
      int64_t batch_stride_D,
      int64_t batch_stride_Vector,
      int64_t batch_stride_Tensor,
      typename LayoutA::Stride::Index lda,
      typename LayoutB::Stride::Index ldb,
      typename LayoutC::Stride::Index ldc1,
      typename LayoutC::Stride::Index ldc2,
      typename LayoutC::Stride::Index ldd,
      typename LayoutC::Stride::Index ldr,
      typename LayoutC::Stride::Index ldt)
    :
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue(epilogue),
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C1(ptr_C1), ptr_C2(ptr_C2), ptr_D(ptr_D),
      ptr_Vector(ptr_Vector),
      ptr_Tensor(ptr_Tensor),
      batch_stride_A(batch_stride_A),
      batch_stride_B(batch_stride_B),
      batch_stride_C1(batch_stride_C1),
      batch_stride_C2(batch_stride_C2),
      batch_stride_Vector(batch_stride_Vector),
      batch_stride_Tensor(batch_stride_Tensor),
      lda(lda), ldb(ldb), ldc1(ldc1), ldc2(ldc2), ldd(ldd), ldr(ldr), ldt(ldt)
    {
      CUTLASS_TRACE_HOST("GemmWithFusedEpilogue::Arguments::Arguments() - problem_size: " << problem_size);
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
      CUTLASS_TRACE_HOST("  ldt: " << this->ldt);
    }

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);

      std::swap(args.problem_size.m(), args.problem_size.n());
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
    typename Epilogue::OutputTileIterator::Params params_C1;
    typename Epilogue::OutputTileIterator::Params params_C2;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::TensorTileIterator::Params params_Tensor;
    typename EpilogueOutputOp::Params output_op;

    void * ptr_A;
    void * ptr_B;
    void * ptr_C1;
    void * ptr_C2;
    void * ptr_D;

    void * ptr_Vector;
    typename LayoutC::Stride::Index ldr;

    void * ptr_Tensor;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C1;
    int64_t batch_stride_C2;
    int64_t batch_stride_Vector;
    int64_t batch_stride_Tensor;

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
      params_A(args.lda),
      params_B(args.ldb),
      params_C1(args.ldc1),
      params_C2(args.ldc2),
      params_D(args.ldd),
      params_Tensor(args.ldt),
      output_op(args.epilogue),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C1(const_cast<void *>(args.ptr_C1)),
      ptr_C2(const_cast<void *>(args.ptr_C2)),
      ptr_D(args.ptr_D),
      ptr_Vector(args.ptr_Vector),
      ldr(args.ldr),
      ptr_Tensor(args.ptr_Tensor),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C1(args.batch_stride_C1),
      batch_stride_C2(args.batch_stride_C2),
      batch_stride_Vector(args.batch_stride_Vector),
      batch_stride_Tensor(args.batch_stride_Tensor)
    {
      CUTLASS_TRACE_HOST("GemmWithFusedEpilogue::Params::Params()");
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
      CUTLASS_TRACE_HOST("  ldt: " << args.ldt);
    }

    /// Lightweight update given a subset of arguments.
    CUTLASS_HOST_DEVICE
    void update(Arguments const &args)
    {
      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C1 = const_cast<void *>(args.ptr_C1);
      ptr_C2 = const_cast<void *>(args.ptr_C2);
      ptr_D = args.ptr_D;

      ptr_Vector = args.ptr_Vector;
      ldr = args.ldr;
      ptr_Tensor = args.ptr_Tensor;

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C1 = args.batch_stride_C1;
      batch_stride_C2 = args.batch_stride_C2;
      batch_stride_Vector = args.batch_stride_Vector;
      batch_stride_Tensor = args.batch_stride_Tensor;
      this->batch_stride_D = args.batch_stride_D;

      output_op = args.epilogue;

      CUTLASS_TRACE_HOST("GemmWithFusedEpilogue::Params::update()");
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
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

    CUTLASS_TRACE_HOST("GemmWithFusedEpilogue::can_implement()");

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

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
    GemmWithFusedEpilogue op;
    op(params, shared_storage);
  }

  #define SPLIT_K_ENABLED 1

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


    #if SPLIT_K_ENABLED
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
    #endif

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
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

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

    ElementC *ptr_C1 = static_cast<ElementC *>(params.ptr_C1);
    ElementC *ptr_C2 = static_cast<ElementC *>(params.ptr_C2);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);
    typename Epilogue::ElementTensor *ptr_Tensor = static_cast<typename Epilogue::ElementTensor *>(params.ptr_Tensor);

    // Define the reduction output pointer and move to the appropriate place
    typename Epilogue::ElementVector *ptr_Vector =
      static_cast<typename Epilogue::ElementVector *>(params.ptr_Vector);

    //
    // Fetch pointers based on mode.
    //

    //
    // Special path when split-K not enabled.
    //

    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() == 1) {

      // Tile iterators loading from source tensors.
      typename Epilogue::OutputTileIterator iterator_C1(
        params.params_C1,
        ptr_C1,
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset
      );

      typename Epilogue::OutputTileIterator iterator_C2(
        params.params_C2,
        ptr_C2,
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

      // Additional tensor to load from
      typename Epilogue::TensorTileIterator tensor_iterator(
          params.params_Tensor,
          // Only the final block outputs Tensor
          ptr_Tensor,
          params.problem_size.mn(),
          thread_idx,
          threadblock_offset);

      // Construct the epilogue
      Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx);

      // Move to appropriate location for this output tile
      if (ptr_Vector) {
        ptr_Vector += threadblock_offset.column() + threadblock_tile_offset.m() * params.ldr;
      }

      // Execute the epilogue operator to update the destination tensor.
      epilogue(output_op,
               ptr_Vector,
               iterator_D,
               accumulators,
               iterator_C1,
               iterator_C2,
               tensor_iterator,
               params.problem_size.mn(),
               threadblock_offset);

      return;
    }

    //
    // Slower path when split-K or batching is needed
    //


    #if SPLIT_K_ENABLED
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
      ptr_C1 += threadblock_tile_offset.k() * params.batch_stride_C1;
      if (ptr_C2) {
        ptr_C2 += threadblock_tile_offset.k() * params.batch_stride_C2;
      }
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
      if (ptr_Tensor) {
        ptr_Tensor = ReferenceFactory<typename Epilogue::ElementTensor>::add_pointer_offset(
          ptr_Tensor,
          threadblock_tile_offset.k() * params.batch_stride_Tensor);
      }
      if (ptr_Vector) {
        ptr_Vector += threadblock_tile_offset.k() * params.batch_stride_Vector;
      }
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_C1 = static_cast<ElementC * const *>(params.ptr_C1)[threadblock_tile_offset.k()];
      if (ptr_C2) {
        ptr_C2 = static_cast<ElementC * const *>(params.ptr_C2)[threadblock_tile_offset.k()];
      }
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[threadblock_tile_offset.k()];
      if (ptr_Tensor) {
        ptr_Tensor = static_cast<typename Epilogue::ElementTensor * const *>(params.ptr_Tensor)[threadblock_tile_offset.k()];
      }
      if (ptr_Vector) {
        ptr_Vector = static_cast<typename Epilogue::ElementVector * const *>(params.ptr_Vector)[threadblock_tile_offset.k()];
      }
    }
    #endif

    // Tile iterators loading from source tensors.
    typename Epilogue::OutputTileIterator iterator_C1(
      params.params_C1,
      ptr_C1,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    typename Epilogue::OutputTileIterator iterator_C2(
      params.params_C2,
      ptr_C2,
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

    // Additional tensor to load from
    typename Epilogue::TensorTileIterator tensor_iterator(
        params.params_Tensor,
        // Only the final block outputs Tensor
        ((params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) &&
         (params.grid_tiled_shape.k() != threadblock_tile_offset.k() + 1))
            ? nullptr
            : ptr_Tensor,
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset);

    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    #if SPLIT_K_ENABLED
    // Wait on the semaphore - this latency may have been covered by iterator construction
    if ((params.mode == GemmUniversalMode::kGemm) && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C1 = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

    }
    #endif

    // Move to appropriate location for this output tile
    if (ptr_Vector) {
      ptr_Vector += threadblock_offset.column() + threadblock_tile_offset.m() * params.ldr;
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op,
             // Only the final block uses Vector
             ((params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) &&
              (params.grid_tiled_shape.k() != threadblock_tile_offset.k() + 1))
                 ? nullptr
                 : ptr_Vector,
             iterator_D,
             accumulators,
             iterator_C1,
             iterator_C2,
             tensor_iterator,
             params.problem_size.mn(),
             threadblock_offset);

    //
    // Release the semaphore
    //

    #if SPLIT_K_ENABLED
    if ((params.mode == GemmUniversalMode::kGemm)  && params.grid_tiled_shape.k() > 1) {

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
    #endif
  }
};

// GemmWithFusedEpilogue with one source
template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmWithFusedEpilogue<Mma_, Epilogue_, ThreadblockSwizzle_, true> {
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
  static int const kSplitKAlignment = const_max(
    128 / sizeof_bits<ElementA>::value,
    128 / sizeof_bits<ElementB>::value
  );

  //
  // Structures
  //

  /// Argument structure
  struct Arguments : UniversalArgumentsBase
  {
    //
    // Data members
    //

    typename EpilogueOutputOp::Params epilogue;

    void const * ptr_A;
    void const * ptr_B;
    void const * ptr_C;
    void * ptr_D;

    void * ptr_Vector;
    void * ptr_Tensor;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_Vector;
    int64_t batch_stride_Tensor;

    typename LayoutA::Stride::Index lda;
    typename LayoutB::Stride::Index ldb;
    typename LayoutC::Stride::Index ldc;
    typename LayoutC::Stride::Index ldd;
    typename LayoutC::Stride::Index ldr;
    typename LayoutC::Stride::Index ldt;

    //
    // Methods
    //

    Arguments():
      ptr_A(nullptr),
      ptr_B(nullptr),
      ptr_C(nullptr),
      ptr_D(nullptr)
    {}

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      void * ptr_Vector,
      void * ptr_Tensor,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      int64_t batch_stride_Vector,
      int64_t batch_stride_Tensor,
      typename LayoutA::Stride::Index lda,
      typename LayoutB::Stride::Index ldb,
      typename LayoutC::Stride::Index ldc,
      typename LayoutC::Stride::Index ldd,
      typename LayoutC::Stride::Index ldr,
      typename LayoutC::Stride::Index ldt)
    :
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue(epilogue),
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D),
      ptr_Vector(ptr_Vector),
      ptr_Tensor(ptr_Tensor),
      batch_stride_A(batch_stride_A),
      batch_stride_B(batch_stride_B),
      batch_stride_C(batch_stride_C),
      batch_stride_Vector(batch_stride_Vector),
      batch_stride_Tensor(batch_stride_Tensor),
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd), ldr(ldr), ldt(ldt)
    {
      CUTLASS_TRACE_HOST("GemmWithFusedEpilogue::Arguments::Arguments() - problem_size: " << problem_size);
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
      CUTLASS_TRACE_HOST("  ldt: " << this->ldt);
    }

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);

      std::swap(args.problem_size.m(), args.problem_size.n());
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
    typename Epilogue::TensorTileIterator::Params params_Tensor;

    typename EpilogueOutputOp::Params output_op;

    void * ptr_A;
    void * ptr_B;
    void * ptr_C;
    void * ptr_D;

    void * ptr_Vector;
    typename LayoutC::Stride::Index ldr;

    void * ptr_Tensor;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_Vector;
    int64_t batch_stride_Tensor;

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
      params_A(args.lda),
      params_B(args.ldb),
      params_C(args.ldc),
      params_D(args.ldd),
      params_Tensor(args.ldt),
      output_op(args.epilogue),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C(const_cast<void *>(args.ptr_C)),
      ptr_D(args.ptr_D),
      ptr_Vector(args.ptr_Vector),
      ldr(args.ldr),
      ptr_Tensor(args.ptr_Tensor),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C(args.batch_stride_C),
      batch_stride_Vector(args.batch_stride_Vector),
      batch_stride_Tensor(args.batch_stride_Tensor)
    {
      CUTLASS_TRACE_HOST("GemmWithFusedEpilogue::Params::Params()");
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
      CUTLASS_TRACE_HOST("  ldt: " << args.ldt);
    }

    /// Lightweight update given a subset of arguments.
    CUTLASS_HOST_DEVICE
    void update(Arguments const &args)
    {
      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C = const_cast<void *>(args.ptr_C);
      ptr_D = args.ptr_D;

      ptr_Vector = args.ptr_Vector;
      ldr = args.ldr;
      ptr_Tensor = args.ptr_Tensor;

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      batch_stride_Vector = args.batch_stride_Vector;
      batch_stride_Tensor = args.batch_stride_Tensor;
      this->batch_stride_D = args.batch_stride_D;

      output_op = args.epilogue;

      CUTLASS_TRACE_HOST("GemmWithFusedEpilogue::Params::update()");
      CUTLASS_TRACE_HOST("  ptr_Vector: " << (void *)this->ptr_Vector);
      CUTLASS_TRACE_HOST("  ptr_Tensor: " << (void *)this->ptr_Tensor);
      CUTLASS_TRACE_HOST("  ldr: " << this->ldr);
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

    CUTLASS_TRACE_HOST("GemmWithFusedEpilogue::can_implement()");

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

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
    GemmWithFusedEpilogue op;
    op(params, shared_storage);
  }

  #define SPLIT_K_ENABLED 1

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


    #if SPLIT_K_ENABLED
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
    #endif

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
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
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

    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);
    typename Epilogue::ElementTensor *ptr_Tensor = static_cast<typename Epilogue::ElementTensor *>(params.ptr_Tensor);

    // Define the reduction output pointer and move to the appropriate place
    typename Epilogue::ElementVector *ptr_Vector =
      static_cast<typename Epilogue::ElementVector *>(params.ptr_Vector);

    //
    // Fetch pointers based on mode.
    //

    //
    // Special path when split-K not enabled.
    //

    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() == 1) {

      // Tile iterators loading from source tensors.
      typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        ptr_C,
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

      // Additional tensor to load from
      typename Epilogue::TensorTileIterator tensor_iterator(
          params.params_Tensor,
          // Only the final block outputs Tensor
          ptr_Tensor,
          params.problem_size.mn(),
          thread_idx,
          threadblock_offset);

      // Construct the epilogue
      Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx);

      // Move to appropriate location for this output tile
      if (ptr_Vector) {
        ptr_Vector += threadblock_offset.column() + threadblock_tile_offset.m() * params.ldr;
      }

      // Execute the epilogue operator to update the destination tensor.
      epilogue(output_op,
               ptr_Vector,
               iterator_D,
               accumulators,
               iterator_C,
               tensor_iterator,
               params.problem_size.mn(),
               threadblock_offset);

      return;
    }

    //
    // Slower path when split-K or batching is needed
    //


    #if SPLIT_K_ENABLED
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
      if (ptr_Tensor) {
        ptr_Tensor = ReferenceFactory<typename Epilogue::ElementTensor>::add_pointer_offset(
          ptr_Tensor,
          threadblock_tile_offset.k() * params.batch_stride_Tensor);
      }
      if (ptr_Vector) {
        ptr_Vector += threadblock_tile_offset.k() * params.batch_stride_Vector;
      }
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_C = static_cast<ElementC * const *>(params.ptr_C)[threadblock_tile_offset.k()];
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[threadblock_tile_offset.k()];
      if (ptr_Tensor) {
        ptr_Tensor = static_cast<typename Epilogue::ElementTensor * const *>(params.ptr_Tensor)[threadblock_tile_offset.k()];
      }
      if (ptr_Vector) {
        ptr_Vector = static_cast<typename Epilogue::ElementVector * const *>(params.ptr_Vector)[threadblock_tile_offset.k()];
      }
    }
    #endif

    // Tile iterators loading from source tensors.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      ptr_C,
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

    // Additional tensor to load from
    typename Epilogue::TensorTileIterator tensor_iterator(
        params.params_Tensor,
        // Only the final block outputs Tensor
        ((params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) &&
         (params.grid_tiled_shape.k() != threadblock_tile_offset.k() + 1))
            ? nullptr
            : ptr_Tensor,
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset);

    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    #if SPLIT_K_ENABLED
    // Wait on the semaphore - this latency may have been covered by iterator construction
    if ((params.mode == GemmUniversalMode::kGemm) && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

    }
    #endif

    // Move to appropriate location for this output tile
    if (ptr_Vector) {
      ptr_Vector += threadblock_offset.column() + threadblock_tile_offset.m() * params.ldr;
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op,
             // Only the final block uses Vector
             ((params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) &&
              (params.grid_tiled_shape.k() != threadblock_tile_offset.k() + 1))
                 ? nullptr
                 : ptr_Vector,
             iterator_D,
             accumulators,
             iterator_C,
             tensor_iterator,
             params.problem_size.mn(),
             threadblock_offset);

    //
    // Release the semaphore
    //

    #if SPLIT_K_ENABLED
    if ((params.mode == GemmUniversalMode::kGemm)  && params.grid_tiled_shape.k() > 1) {

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
    #endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
