/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/arch/arch.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/params_universal_base.h"
#include "cutlass/gemm/kernel/gemm_sparse_universal.h"

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
class GemmSparseUniversalWithAbsmax {
public:
  using Base = GemmSparseUniversal<Mma_, Epilogue_, ThreadblockSwizzle_>;

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  static int const kSparse = Mma::kSparse;
  static int const kMetaSizeInBits = Mma::kMetaSizeInBits;
  static int const kMaxID2 = Mma::kMaxID2;
  static int const kElementsPerElementE = Mma::kElementsPerElementE;

  using ElementE = typename Mma::ElementE;
  using LayoutE = typename Mma::LayoutE;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Epilogue::OutputTileIterator::Layout;
  using ElementAux = typename Epilogue::AuxOutputTileIterator::Element;
  using LayoutAux = typename Epilogue::AuxOutputTileIterator::Layout;
  using ElementVector = typename Epilogue::ElementVector;

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
  struct Arguments : detail::SparseUniversalArgumentsBase<
      LayoutA,
      LayoutB,
      LayoutC,
      LayoutE
    > {
    using Base = detail::SparseUniversalArgumentsBase<
      LayoutA,
      LayoutB,
      LayoutC,
      LayoutE
    >;

    void const* ptr_Aux;
    void const* ptr_Vector;
    int64_t batch_stride_Aux;
    int64_t batch_stride_Vector;
    typename LayoutAux::Stride::LongIndex ldaux;
    int64_t ldvector;

    typename EpilogueOutputOp::Params epilogue;

    Arguments() {}

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
      void const * ptr_E,
      void const * ptr_Aux,
      void const * ptr_Vector,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      int64_t batch_stride_E,
      int64_t batch_stride_Aux,
      int64_t batch_stride_Vector,
      typename LayoutA::Stride::LongIndex lda,
      typename LayoutB::Stride::LongIndex ldb,
      typename LayoutC::Stride::LongIndex ldc,
      typename LayoutC::Stride::LongIndex ldd,
      typename LayoutC::Stride::LongIndex lde,
      typename LayoutAux::Stride::LongIndex ldaux,
      int64_t ldvector
      )
    :
      Base(
        mode, problem_size, batch_count,
        ptr_A, ptr_B, ptr_C, ptr_D, ptr_E,
        batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D, batch_stride_E,
        lda, ldb, ldc, ldd, lde
      ),
      ptr_Aux(ptr_Aux),
      ptr_Vector(ptr_Vector),
      batch_stride_Aux(batch_stride_Aux),
      batch_stride_Vector(batch_stride_Vector),
      ldaux(ldaux),
      ldvector(ldvector),
      epilogue(epilogue)
    { }
  };


  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params : detail::SparseUniversalParamsBase<
    Mma,
    Epilogue,
    Arguments,
    ThreadblockSwizzle,
    ThreadblockShape,
    ElementA,
    ElementB,
    ElementC,
    LayoutA,
    LayoutB>
  {
    using ParamsBase = detail::SparseUniversalParamsBase<
      Mma,
      Epilogue,
      Arguments,
      ThreadblockSwizzle,
      ThreadblockShape,
      ElementA,
      ElementB,
      ElementC,
      LayoutA,
      LayoutB>;

    typename Epilogue::AuxOutputTileIterator::Params params_Aux;
    int64_t ldvector;

    void* ptr_Aux;
    void* ptr_Vector;

    int64_t batch_stride_Aux;
    int64_t batch_stride_Vector;
    typename EpilogueOutputOp::Params output_op;

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
      params_Aux(args.ldaux),
      ldvector(args.ldvector),
      ptr_Aux(const_cast<void *>(args.ptr_Aux)),
      ptr_Vector(const_cast<void *>(args.ptr_Vector)),
      batch_stride_Aux(args.batch_stride_Aux),
      batch_stride_Vector(args.batch_stride_Vector),
      output_op(args.epilogue)
    {}

    /// Lightweight update given a subset of arguments.
    void update(Arguments const &args)
    {
      CUTLASS_TRACE_HOST("GemmUniversal::Params::update()");

      // Update input/output pointers
      this->ptr_A = const_cast<void *>(args.ptr_A);
      this->ptr_B = const_cast<void *>(args.ptr_B);
      this->ptr_C = const_cast<void *>(args.ptr_C);
      this->ptr_D = args.ptr_D;
      this->ptr_E = const_cast<void *>(args.ptr_E);
      ptr_Aux = const_cast<void *>(args.ptr_Aux);
      ptr_Vector = const_cast<void *>(args.ptr_Vector);

      this->batch_stride_A = args.batch_stride_A;
      this->batch_stride_B = args.batch_stride_B;
      this->batch_stride_C = args.batch_stride_C;
      this->batch_stride_D = args.batch_stride_D;
      this->batch_stride_E = args.batch_stride_E;
      this->batch_stride_Aux = args.batch_stride_Aux;
      batch_stride_Vector = args.batch_stride_Vector;

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
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size,
    GemmUniversalMode mode,
    int split_k_count) {
    return Base::can_implement(problem_size, mode, split_k_count);
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size, args.mode, args.batch_count);
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
    GemmSparseUniversalWithAbsmax op;
    op(params, shared_storage);
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

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);
    ElementE *ptr_E = static_cast<ElementE *>(params.ptr_E);

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
      ptr_A += threadblock_tile_offset.k() * params.batch_stride_A / kSparse;
      ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
      ptr_E += threadblock_tile_offset.k() * params.batch_stride_E / kSparse;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA * const *>(params.ptr_A)[threadblock_tile_offset.k()];
      ptr_B = static_cast<ElementB * const *>(params.ptr_B)[threadblock_tile_offset.k()];
      ptr_E = static_cast<ElementE * const *>(params.ptr_E)[threadblock_tile_offset.k()];
    }

    __syncthreads();

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      offset_k / kSparse,
    };

    cutlass::MatrixCoord tb_offset_B{
      offset_k,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    cutlass::MatrixCoord tb_offset_E{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      offset_k / kSparse / kElementsPerElementE,
    };

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k / kSparse},
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    typename Mma::IteratorE iterator_E(
      params.params_E,
      ptr_E,
      {params.problem_size.m(), problem_size_k / kSparse / kElementsPerElementE},
      thread_idx,
      tb_offset_E);

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
      iterator_E,
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
    ElementAux * ptr_Aux = static_cast<ElementAux *>(params.ptr_Aux);
    ElementVector * ptr_Vector = static_cast<ElementVector *>(params.ptr_Vector);

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
      if (ptr_Aux) {
        ptr_Aux += threadblock_tile_offset.k() * params.batch_stride_Aux;
      }
      if (ptr_Vector) {
        ptr_Vector += threadblock_tile_offset.k() * params.batch_stride_Vector;
      }
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_C = static_cast<ElementC * const *>(params.ptr_C)[threadblock_tile_offset.k()];
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[threadblock_tile_offset.k()];
      if (ptr_Aux) {
        ptr_Aux = static_cast<ElementAux * const *>(params.ptr_Aux)[threadblock_tile_offset.k()];
      }
      if (ptr_Vector) {
        ptr_Vector = static_cast<ElementVector * const *>(params.ptr_Vector)[threadblock_tile_offset.k()];
      }
    }

    // Move to appropriate location for this output tile
    if (ptr_Vector) {
      ptr_Vector += threadblock_offset.column() + threadblock_tile_offset.m() * params.ldvector;
    }

    // Tile iterator loading from source tensor.
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

    // Tile iterator writing to auxiliary destination tensor.
    typename Epilogue::AuxOutputTileIterator iterator_Aux(
      params.params_Aux,
      // Only the final block writes the auxiliary tensor
      ((params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) &&
          (params.grid_tiled_shape.k() != threadblock_tile_offset.k() + 1))
          ? nullptr
          : ptr_Aux,
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
    }


    // Execute the epilogue operator to update the destination tensor.
    epilogue(
      output_op,
      // Only the final block uses Vector
      ((params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) &&
       (params.grid_tiled_shape.k() != threadblock_tile_offset.k() + 1))
          ? nullptr
          : ptr_Vector,
      iterator_D,
      accumulators,
      iterator_C,
      iterator_Aux,
      params.problem_size.mn(),
      threadblock_offset);

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
