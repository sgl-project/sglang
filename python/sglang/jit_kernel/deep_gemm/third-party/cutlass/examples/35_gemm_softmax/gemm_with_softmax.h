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

/**

*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/threadblock/epilogue_visitor_with_softmax.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/reduction/kernel/reduce_softmax_final.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "gemm_with_epilogue_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Kernel computes partial reduction
//
//
// 2. Sum[m, n'] = sum_n(exp(D[m, n] - N[m, 0]))
//
template <
  typename ElementD_,
  typename ElementNorm_,
  typename ElementSum_,
  typename ElementSoft_,
  typename ElementSoftmaxCompute_,
  int Alignment,
  typename ApplyShape_ = MatrixShape<1, 1024>
>
class ApplySoftmax {
public:

  using ElementD = ElementD_;
  using ElementNorm = ElementNorm_;
  using ElementSum = ElementSum_;
  using ElementSoft = ElementSoft_;
  using ElementSoftmaxCompute = ElementSoftmaxCompute_;

  static int const kAlignment = Alignment;
  using ApplyShape = ApplyShape_;

  using Layout = cutlass::layout::RowMajor;

  using TensorRefD = TensorRef<ElementD, Layout>;
  using TensorRefN = TensorRef<ElementNorm, Layout>;
  using TensorRefSum = TensorRef<ElementSum, Layout>;
  using TensorRefSoft = TensorRef<ElementSoft, Layout>;

  using FragmentSoftmax = Array<ElementSoftmaxCompute, kAlignment>;

  //
  // Arguments
  //

  struct Arguments {

    MatrixCoord     extent;             ///< Extent of D and Softmax matrices
    int             batch_count;        ///< Batch count
    TensorRefD      ref_D;              ///< D matrix computed by GEMM+Max (input)
    TensorRefN      ref_N;              ///< Norm tensor (input)
    TensorRefSum    ref_S;              ///< Sum  tensor (input)
    TensorRefSoft   ref_Soft;           ///< Softmax tensor (output)
    int64_t         batch_stride_D;     ///< Batch stride for D tensor
    int64_t         batch_stride_N;     ///< Batch stride for N tensor
    int64_t         batch_stride_S;     ///< Batch stride for S tensor
    int64_t         batch_stride_Soft;  ///< Batch stride for softmax tensor

    //
    // Methods
    //
    Arguments():
      batch_count(1),
      batch_stride_D(0),
      batch_stride_N(0),
      batch_stride_S(0),
      batch_stride_Soft(0)
    { }

    Arguments(
      MatrixCoord     extent_,             ///< Extent of D and Softmax matrices
      int             batch_count_,        ///< Batch count
      TensorRefD      ref_D_,              ///< D matrix computed by GEMM+PartialReduce
      TensorRefN      ref_N_,              ///< Output parameter for N
      TensorRefSum    ref_S_,              ///< Output parameter for N
      TensorRefSoft   ref_Soft_,           ///< Softmax
      int64_t         batch_stride_D_ = 0,
      int64_t         batch_stride_N_ = 0,
      int64_t         batch_stride_S_ = 0,
      int64_t         batch_stride_Soft_ = 0
    ):
      extent(extent_),
      batch_count(batch_count_),
      ref_D(ref_D_),
      ref_N(ref_N_),
      ref_S(ref_S_),
      ref_Soft(ref_Soft_),
      batch_stride_D(batch_stride_D_),
      batch_stride_N(batch_stride_N_),
      batch_stride_S(batch_stride_S_),
      batch_stride_Soft(batch_stride_Soft_)
    {

    }
  };

  //
  // Params struct
  //

  struct Params {
    Arguments args;

    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args_): args(args_) { }
  };

  //
  // SharedStorage
  //

  struct SharedStorage {

  };

private:

public:

  CUTLASS_DEVICE
  ApplySoftmax() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    apply(params, shared_storage);
  }

private:


  /// Compute Softmax
  CUTLASS_DEVICE
  void apply(Params const &params, SharedStorage &shared_storage) {

    using AccessTypeD = AlignedArray<ElementD, kAlignment>;

    int block_batch = blockIdx.z;
    int block_m = blockIdx.x * ApplyShape::kRow;
    int block_n = 0;

    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x * kAlignment;

    int idx_m = block_m + thread_m;
    int idx_n = block_n + thread_n;

    int batch_offset_norm = block_batch * params.args.batch_stride_N;
    int batch_offset_sum = block_batch * params.args.batch_stride_S;

    // Kill off thread if it is outside the row boundary
    if (params.args.extent.row() <= idx_m) {
      return;
    }

    //
    // Setup pointers to load D again
    //

    using AccessTypeD = AlignedArray<ElementD, kAlignment>;
    using AccessTypeSoft = AlignedArray<ElementSoft, kAlignment>;
    using FragmentSoft = Array<ElementSoft, kAlignment>;
    using ConvertSoftCompute = cutlass::NumericArrayConverter<ElementSoftmaxCompute, ElementD, kAlignment>;
    using ConvertSoftOutput = cutlass::NumericArrayConverter<ElementSoft, ElementSoftmaxCompute, kAlignment>;

    using Mul = cutlass::multiplies<FragmentSoftmax>;
    using Minus = cutlass::minus<FragmentSoftmax>;
    using Exp   = cutlass::fast_exp_op<FragmentSoftmax>;

    ConvertSoftCompute   convert_soft_compute;
    ConvertSoftOutput  convert_soft_output;

    Minus     minus;
    Mul       mul;
    Exp       exponential;

    using ConvertSum = cutlass::NumericConverter<ElementSoftmaxCompute, ElementSum>;
    using ConvertNorm = cutlass::NumericConverter<ElementSoftmaxCompute, ElementNorm>;

    ConvertSum   convert_sum;
    ConvertNorm  convert_norm;

    AccessTypeD *access_d = reinterpret_cast<AccessTypeD *>(
      params.args.ref_D.data() +
      params.args.batch_stride_D * block_batch +
      params.args.ref_D.layout()({idx_m, idx_n}));

    AccessTypeSoft *access_soft = reinterpret_cast<AccessTypeSoft *>(
      params.args.ref_Soft.data() +
      params.args.batch_stride_Soft * block_batch +
      params.args.ref_Soft.layout()({idx_m, idx_n}));

    ElementSum inv_sum = (params.args.ref_S.data())[idx_m + batch_offset_sum];
    ElementNorm norm = (params.args.ref_N.data())[idx_m + batch_offset_norm];

    //
    // Loop
    //
    CUTLASS_PRAGMA_UNROLL
    for (
      int idx = 0;
      idx < params.args.extent.column();
      idx += ApplyShape::kColumn * kAlignment) {

      if (idx_n < params.args.extent.column()) {
        AccessTypeD fetch;
        arch::global_load<AccessTypeD, sizeof(AccessTypeD)>(fetch, access_d, true);

        FragmentSoftmax result = mul(exponential(minus(convert_soft_compute(fetch), convert_norm(norm))),  convert_sum(inv_sum));
        FragmentSoft soft  = convert_soft_output(result);

        arch::global_store<FragmentSoft, sizeof(FragmentSoft)>(soft, access_soft, true);
      }

      access_d += ApplyShape::kColumn;
      access_soft += ApplyShape::kColumn;
      idx_n += ApplyShape::kColumn * kAlignment;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

///
template <
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename ElementCompute_,
  typename OperatorClass_,
  typename ArchTag_,
  typename ThreadblockShape_,
  typename WarpShape_,
  typename InstructionShape_,
  typename EpilogueFunctorOp_,
  int kStages_,
  typename ApplyShape_ = MatrixShape<1, 1024>,
  int AlignmentA_ = 128 / cutlass::sizeof_bits<ElementA_>::value,
  int AlignmentB_ = 128 / cutlass::sizeof_bits<ElementB_>::value,
  int AlignmentSoftmax_ = 128 / cutlass::sizeof_bits<ElementC_>::value,
  typename ElementNorm_ = float,
  typename ElementSum_ = float,
  typename ElementSoftmax_ = ElementC_
>
class GemmSoftmax {
public:

  ///////////////////////////////////////////////////////////////////////////////////////////////

  //
  // Type definitions
  //

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;
  using ElementCompute = ElementCompute_;
  using ElementSum = ElementSum_;
  using ElementSoft = ElementSoftmax_;
  using ElementSoftmaxCompute = float;

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;

  using EpilogueFunctorOp = EpilogueFunctorOp_;
  using ElementNorm = ElementNorm_;

  using ApplyShape = ApplyShape_;

  // These are mandatory layouts.
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutN = cutlass::layout::RowMajor;
  using LayoutS = cutlass::layout::RowMajor;
  using LayoutSoft = cutlass::layout::RowMajor;

  using TensorRefA = TensorRef<ElementA, LayoutA>;
  using TensorRefB = TensorRef<ElementB, LayoutB>;
  using TensorRefC = TensorRef<ElementC, LayoutC>;
  using TensorRefN = TensorRef<ElementNorm, LayoutN>;
  using TensorRefSum = TensorRef<ElementSum, LayoutS>;
  using TensorRefSoft = TensorRef<ElementSoft, LayoutSoft>;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape        = WarpShape_;
  using InstructionShape = InstructionShape_;

  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;

  static int const kStages  = kStages_;
  static int const AlignmentA = AlignmentA_;
  static int const AlignmentB = AlignmentB_;
  static int const AlignmentSoftmax = AlignmentSoftmax_;

  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // basic GEMM kernel
  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    ElementA,
    LayoutA,
    AlignmentA,
    ElementB,
    LayoutB,
    AlignmentB,
    ElementC,
    LayoutC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    ThreadblockSwizzle,
    kStages,
    true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementA, ElementB, ElementC, ElementCompute>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone
  >::GemmKernel;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Epilogue visitor
  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorSoftmax<
    ThreadblockShape,
    DefaultGemmKernel::kThreadCount,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,
    ElementCompute,
    ElementNorm,
    ElementSum,
    ElementSoftmaxCompute,
    EpilogueFunctorOp
  >;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    EpilogueVisitor,
    typename DefaultGemmKernel::Epilogue
  >::Epilogue;

  // GEMM
  using GemmKernel = gemm::kernel::GemmWithEpilogueVisitor<
    typename DefaultGemmKernel::Mma,
    Epilogue,
    ThreadblockSwizzle
  >;

  // Softmax kernel
  using SoftmaxApplyKernel = kernel::ApplySoftmax<
    ElementC,
    ElementNorm,
    ElementSum,
    ElementSoft,
    ElementSoftmaxCompute,
    AlignmentSoftmax,
    ApplyShape
  >;

  using ApplyFinalReductionKernel = cutlass::reduction::kernel::ApplySoftmaxFinalReduction<
    ElementNorm,
    ElementSum,
    ElementSoftmaxCompute,
    ThreadblockShape
  >;

public:

  /// Arguments class
  struct Arguments {

    typename GemmKernel::Arguments         gemm;
    typename SoftmaxApplyKernel::Arguments softmax;
    typename ApplyFinalReductionKernel::Arguments reduction;
    cutlass::gemm::GemmCoord extend;

    //
    // Methods
    //
    Arguments() { }

    Arguments(
      cutlass::gemm::GemmCoord problem_size,
      int32_t    batch_count_,
      TensorRefA ref_A_,
      TensorRefB ref_B_,
      TensorRefC ref_C_,
      TensorRefC ref_D_,
      typename EpilogueFunctorOp::Params linear_scaling,
      TensorRefN ref_N_,
      TensorRefSum ref_S_,
      TensorRefSoft ref_Softmax_,
      int64_t batch_stride_A_ = 0,
      int64_t batch_stride_B_ = 0,
      int64_t batch_stride_C_ = 0,
      int64_t batch_stride_D_ = 0,
      int64_t batch_stride_Max_ = 0,
      int64_t batch_stride_Sum_ = 0,
      int64_t batch_stride_Softmax_ = 0
    ):
      gemm(
        cutlass::gemm::GemmUniversalMode::kBatched,
        problem_size,
        batch_count_,
        ref_A_,
        ref_B_,
        ref_C_,
        ref_D_,
        ref_N_.data(),
        ref_S_.data(),
        batch_stride_A_,
        batch_stride_B_,
        typename EpilogueVisitor::Arguments(
          linear_scaling,
          batch_stride_C_,
          batch_stride_D_,
          batch_stride_Max_,
          batch_stride_Sum_
        )
      ),
      reduction(
        problem_size,
        ref_N_.data(),
        ref_S_.data(),
        batch_stride_Max_,
        batch_stride_Sum_
      ), 
      softmax(
        MatrixCoord(problem_size.m(), problem_size.n()),
        batch_count_,
        ref_D_,
        ref_N_,
        ref_S_,
        ref_Softmax_,
        batch_stride_D_,
        batch_stride_Max_,
        batch_stride_Sum_,
        batch_stride_Softmax_
      ),
      extend(problem_size)
    {

    }
  };

  struct Params {

    typename GemmKernel::Params         gemm;
    typename SoftmaxApplyKernel::Params softmax;
    typename ApplyFinalReductionKernel::Params reduction;
    MatrixCoord extend;
    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args):
      gemm(args.gemm),
      reduction(args.reduction),
      softmax(args.softmax),
      extend(MatrixCoord(args.extend.m(), args.extend.n()))
    {

    }
  };

public:

  // Gemm


  //
  // Methods
  //

private:

  Params params_;

public:

  /// Ctor
  GemmSoftmax() {

  }

  /// Initialize
  Status initialize(Arguments const &args) {

    params_ = Params(args);

    return cutlass::Status::kSuccess;
  }

  /// Run
  Status run(cudaStream_t stream) {

    //
    // Launch the GEMM + max kernel
    //

    dim3 gemm_grid = ThreadblockSwizzle().get_grid_shape(params_.gemm.grid_tiled_shape);
    dim3 gemm_block(GemmKernel::kThreadCount, 1, 1);

    int gemm_smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    cudaError_t result;

    if (gemm_smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    gemm_smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<GemmKernel><<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }


    //
    // Launch the ApplyFinalReductionKernel
    //

    int thread_per_block = 128;
    int block_per_row = (params_.extend.row() + thread_per_block - 1) / thread_per_block;
    if (block_per_row < 4) {
      thread_per_block = 32;
      block_per_row = (params_.extend.row() + thread_per_block - 1) / thread_per_block;
    }

    dim3 final_reduction_grid(block_per_row, 1, params_.softmax.args.batch_count);
    dim3 final_reduction_block(thread_per_block);

    Kernel<ApplyFinalReductionKernel><<<
      final_reduction_grid, final_reduction_block, sizeof(typename ApplyFinalReductionKernel::SharedStorage), stream
    >>>(params_.reduction);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the SoftmaxApplyKernel
    //

    dim3 apply_block(SoftmaxApplyKernel::ApplyShape::kColumn, SoftmaxApplyKernel::ApplyShape::kRow);

    int threadblock_rows = SoftmaxApplyKernel::ApplyShape::kRow;
    int threadblock_columns = SoftmaxApplyKernel::ApplyShape::kColumn * SoftmaxApplyKernel::kAlignment;

    dim3 apply_grid(
      (params_.softmax.args.extent.row() + threadblock_rows - 1) / threadblock_rows,
      (params_.softmax.args.extent.column() + threadblock_columns - 1) / threadblock_columns,
      params_.softmax.args.batch_count);

    Kernel<SoftmaxApplyKernel><<<
      apply_grid, apply_block, sizeof(typename SoftmaxApplyKernel::SharedStorage), stream
    >>>(params_.softmax);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return cutlass::Status::kSuccess;
  }

  /// Function call operator
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
