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
    \brief A file contains all functioning classes needed by GemmLayernorm.

    GemmLayernorm example =  GEMM0 with partial reduction fused in epilogue (EpilogueVisitorLayerNorm)
                          +  lightweight full reduction kernel (ApplyFinalReduction)
                          +  GEMM1 with elemenwise operations fused in mainloop (GemmLayernormMainloopFusion)

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
#include "cutlass/gemm/device/gemm_layernorm_mainloop_fusion.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "gemm_with_epilogue_visitor.h"
#include "helper.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementVariance_,
  typename ElementMean_,
  typename ElementLayernormCompute_,
  typename ElementOutput,
  typename ThreadblockShape_,
  bool IsShiftedVariance_ = false
>
class ApplyFinalReduction {
public:

  using ElementVariance = ElementVariance_;
  using ElementMean = ElementMean_;
  using ElementLayernormCompute = ElementLayernormCompute_;
  using ThreadblockShape = ThreadblockShape_;

  // Pre-processing has ensured the layout equivalent to RowMajor
  using Layout = cutlass::layout::RowMajor;

  using TensorVariance = TensorRef<ElementVariance, Layout>;
  using TensorMean = TensorRef<ElementMean, Layout>;

  static bool const kIsShiftedVariance = IsShiftedVariance_;

  //
  // Arguments
  //

  struct Arguments {

    MatrixCoord     extent;             ///< Extent of D and Layernorm matrices
    TensorVariance  ref_Variance;       ///< Sum Square or Variance tensor (input / output)
    TensorMean      ref_Mean;           ///< Sum or Mean tensor (input / output)
    ElementOutput   *ptr_Shifted_K;     ///< Shifted K tensor pointer

    //
    // Methods
    //
    Arguments(){ }

    Arguments(
      MatrixCoord     extent_,
      TensorVariance  ref_Variance_,
      TensorMean      ref_Mean_,
      ElementOutput   *ptr_Shifted_K_
    ):
      extent(extent_),
      ref_Variance(ref_Variance_),
      ref_Mean(ref_Mean_),
      ptr_Shifted_K(ptr_Shifted_K_)
    {

    }
  };

  struct SharedStorage {


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

private:

public:

  CUTLASS_DEVICE
  ApplyFinalReduction() { }

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    apply(params, shared_storage);
  }

private:

  /// Partial reduction
  CUTLASS_DEVICE
  void apply(Params const &params, SharedStorage &shared_storage) {

    int threadblock_num = (params.args.extent.column() + ThreadblockShape::kM - 1) / ThreadblockShape::kM;

    int block_n = blockIdx.x * blockDim.x;

    int thread_n = threadIdx.x;

    int idx_n = block_n + thread_n;

    if (idx_n >= params.args.extent.row()) {
      return;
    }

    using ConvertVarianceOutput = cutlass::NumericConverter<ElementVariance, ElementLayernormCompute>;
    using ConvertMeanOutput = cutlass::NumericConverter<ElementMean, ElementLayernormCompute>;

    using ConvertVariance = cutlass::NumericConverter<ElementLayernormCompute, ElementVariance>;
    using ConvertMean = cutlass::NumericConverter<ElementLayernormCompute, ElementMean>;

    using ConvertShiftK = cutlass::NumericConverter<ElementLayernormCompute, ElementOutput>;

    ConvertVariance   convert_variance;
    ConvertMean  convert_mean;

    ConvertVarianceOutput   convert_variance_output;
    ConvertMeanOutput  convert_mean_output;

    ElementVariance *access_square = params.args.ref_Variance.data() + idx_n;
    ElementMean *access_mean = params.args.ref_Mean.data() + idx_n;

    ElementVariance *access_square_bak = access_square;
    ElementMean *access_mean_bak = access_mean;

    ElementLayernormCompute frag_square_sum = ElementLayernormCompute(0);
    ElementLayernormCompute frag_element_sum = ElementLayernormCompute(0);
    ElementVariance fetch_square;
    ElementMean fetch_mean;

    CUTLASS_PRAGMA_UNROLL
    for (int idx_m = 0; idx_m < threadblock_num; idx_m++) {
      arch::global_load<ElementVariance, sizeof(ElementVariance)>(fetch_square, access_square, true);
      arch::global_load<ElementMean, sizeof(ElementMean)>(fetch_mean, access_mean, true);
      frag_element_sum += convert_mean(fetch_mean);
      frag_square_sum += convert_variance(fetch_square);
      access_square += params.args.extent.row();
      access_mean += params.args.extent.row();
    }

    ElementLayernormCompute mean = frag_element_sum;
    ElementLayernormCompute square_mean = frag_square_sum;

    ElementLayernormCompute variance;

    if (kIsShiftedVariance && params.args.ptr_Shifted_K != nullptr) {
      ElementOutput *access_shift_k = params.args.ptr_Shifted_K + idx_n;
      ElementOutput fetch_shift_k;
      ConvertShiftK convert_shift_k;
      arch::global_load<ElementOutput, sizeof(ElementOutput)>(fetch_shift_k, access_shift_k, true);
      ElementLayernormCompute shifted_mean =  mean - convert_shift_k(fetch_shift_k);
      variance = cutlass::constants::one<ElementLayernormCompute>() / cutlass::fast_sqrt(square_mean - shifted_mean * shifted_mean + ElementLayernormCompute(1e-6));
    }else{
      variance = cutlass::constants::one<ElementLayernormCompute>() / cutlass::fast_sqrt(square_mean - mean * mean + ElementLayernormCompute(1e-6));
    }

    mean = -mean * variance;

    access_square = access_square_bak;
    access_mean = access_mean_bak;

    access_square[0] = convert_variance_output(variance);
    access_mean[0] = convert_mean_output(mean);

  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ThreadblockShape_,
  int ThreadCount,
  typename OutputTileIterator_,
  typename AccumulatorTile_,
  typename ElementAccumulator_,
  typename ElementVariance_,
  typename ElementMean_,
  typename ElementLayernormCompute_,
  typename ElementwiseFunctor_,
  bool IsShiftedVariance_ = false
>
class EpilogueVisitorLayerNorm {
public:

  using ElementVariance = ElementVariance_;
  using ElementMean = ElementMean_;
  using ElementLayernormCompute = ElementLayernormCompute_;

  using AccumulatorTile = AccumulatorTile_;

  using ThreadblockShape   = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
  static int const kRowIterations = OutputTileIterator::ThreadMap::Iterations::kRow;

  static int const kThreads = OutputTileIterator::ThreadMap::kThreads;

  static bool const kIsShiftedVariance = IsShiftedVariance_;

  using ElementOutput = typename OutputTileIterator::Element;

  static int const kDeltaRow = OutputTileIterator::ThreadMap::Delta::kRow;

  /// Array type used in Shift-K Layernorm
  static int const kRowAccessCount = kIterations * kRowIterations;

  using ConvertedShiftFragment = Array<ElementLayernormCompute, kRowAccessCount>;

  // Conducts manual transpose externally (already supported) for column major
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementAccumulator = ElementAccumulator_;

  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using LayernormFragment = Array<ElementLayernormCompute, kElementsPerAccess>;
  using OutputVector = Array<ElementOutput, kElementsPerAccess>;
  using TensorRefD = TensorRef<ElementOutput, LayoutOutput>;

  static int const kThreadsPerRow = OutputTileIterator::ThreadMap::Detail::RowArrangement::Detail::kShapeWidth;
  static int const kThreadsInColumn = kThreads / kThreadsPerRow;
  static int const kHalfThreadsPerRow = (kThreadsPerRow >> 1);

  /// Argument structure
  struct Arguments {

    typename ElementwiseFunctor::Params   elementwise;
    TensorRefD                            ref_C;
    TensorRefD                            ref_D;
    ElementVariance                       *ptr_Variance;
    ElementMean                           *ptr_Mean;
    ElementOutput                         *ptr_Shifted_K;

    //
    // Methods
    //
    Arguments():
      ptr_Variance(nullptr),
      ptr_Mean(nullptr),
      ptr_Shifted_K(nullptr)
    {

    }

    Arguments(
      typename ElementwiseFunctor::Params   elementwise_,
      TensorRefD                            ref_C_,
      TensorRefD                            ref_D_,
      ElementVariance                       *ptr_Variance,
      ElementMean                           *ptr_Mean_,
      ElementOutput                         *ptr_Shifted_K_ = nullptr
    ):
      elementwise(elementwise_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      ptr_Variance(ptr_Variance),
      ptr_Mean(ptr_Mean_),
      ptr_Shifted_K(ptr_Shifted_K_)
    {

    }
  };

  struct Params {

    typename ElementwiseFunctor::Params   elementwise;
    typename OutputTileIterator::Params   params_C;
    typename OutputTileIterator::Params   params_D;
    typename OutputTileIterator::Element *ptr_C;
    typename OutputTileIterator::Element *ptr_D;
    ElementVariance                       *ptr_Variance;
    ElementMean                           *ptr_Mean;
    ElementOutput                         *ptr_Shifted_K;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params():
      ptr_D(nullptr),
      ptr_Variance(nullptr),
      ptr_Mean(nullptr)
    {

    }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args):
      elementwise(args.elementwise),
      params_C(args.ref_C.layout()),
      params_D(args.ref_D.layout()),
      ptr_C(args.ref_C.data()),
      ptr_D(args.ref_D.data()),
      ptr_Variance(args.ptr_Variance),
      ptr_Mean(args.ptr_Mean),
      ptr_Shifted_K(args.ptr_Shifted_K)
    {

    }
  };

  /// Shared storage
  struct SharedStorage {

  };

private:

  Params const &                        params_;
  SharedStorage &                       shared_storage_;
  MatrixCoord                           extent_;
  ElementwiseFunctor                    elementwise_;

  OutputTileIterator                    iterator_C_;
  OutputTileIterator                    iterator_D_;
  typename OutputTileIterator::Fragment fragment_C_;
  typename OutputTileIterator::Fragment fragment_D_;

  ElementAccumulator                    alpha_;
  ElementAccumulator                    beta_;
  ConvertedShiftFragment                shift_k_frag_;

  ElementLayernormCompute               accum_sum_square_;
  ElementLayernormCompute               accum_sum_element_;

  MatrixCoord                           thread_offset_;

public:

  CUTLASS_DEVICE
  EpilogueVisitorLayerNorm(
    Params const &params,                                         ///< Parameters routed to the epilogue
    SharedStorage &shared_storage,                                ///< Shared storage needed by the functors here
    MatrixCoord const &problem_size0,                              ///< Problem size of the output
    int thread_idx,                                               ///< Thread index within the threadblock
    int warp_idx,                                                 ///< Warp index within the threadblock
    int lane_idx,                                                 ///< Lane index within the warp
    MatrixCoord const &threadblock_offset = MatrixCoord(0, 0)
  ):
    params_(params),
    shared_storage_(shared_storage),
    extent_(problem_size0),
    elementwise_(params.elementwise),
    iterator_C_(params.params_C, params.ptr_C, problem_size0, thread_idx, threadblock_offset),
    iterator_D_(params.params_D, params.ptr_D, problem_size0, thread_idx, threadblock_offset)
  {
    alpha_ = (params.elementwise.alpha_ptr ? *params.elementwise.alpha_ptr : params.elementwise.alpha);
    beta_ =  (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr : params.elementwise.beta);

    if (beta_ == ElementAccumulator()) {
      iterator_C_.clear_mask();
    }
  }

  /// Helper to indicate split-K behavior
  CUTLASS_DEVICE
  void set_k_partition(
    int split_k_index,                                            ///< Index of this threadblock within split-K partitioned scheme
    int split_k_slices) {                                         ///< Total number of split-K slices

  }

  /// Called to set the batch index
  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {

  }

  /// Called at the start of the epilogue just before iterating over accumulator slices
  CUTLASS_DEVICE
  void begin_epilogue() {

    // If shift-K feature is enabled, we load shift-k fragment
    // at the very beginning of an epilogue
    if (kIsShiftedVariance && params_.ptr_Shifted_K != nullptr) {
      shift_k_frag_.clear();
      int thread_offset_row_base = iterator_D_.thread_start_row();

      CUTLASS_PRAGMA_UNROLL
      for (int iter_idx = 0; iter_idx < kIterations; ++iter_idx) {
        int step_offset = iter_idx * OutputTileIterator::Shape::kRow;
        CUTLASS_PRAGMA_UNROLL
        for (int rid = 0; rid < kRowIterations; ++rid) {
          int row_step_offset = rid * kDeltaRow;
          int row_offset = thread_offset_row_base + step_offset + row_step_offset;
          bool is_load = (row_offset < extent_.row());
          shift_k_frag_[iter_idx * kRowIterations + rid] = load_shift_k_(row_offset, is_load);
        }

      }

    }

  }

  /// Called at the start of one step before starting accumulator exchange
  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    fragment_D_.clear();

    if (elementwise_.kScale != cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      fragment_C_.clear();
      iterator_C_.load(fragment_C_);
      ++iterator_C_;
    }
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void begin_row(int row_idx) {

  }

  /// Called after accumulators have been exchanged for each accumulator vector
  CUTLASS_DEVICE
  void visit(
    int iter_idx,
    int row_idx,
    int column_idx,
    int frag_idx,
    AccumulatorFragment const &accum) {

    using Mul = cutlass::multiplies<ElementLayernormCompute>;
    using Minus = cutlass::minus<ElementLayernormCompute>;
    using Exp   = cutlass::fast_exp_op<ElementLayernormCompute>;

    [[maybe_unused]] Minus minus;
    [[maybe_unused]] Mul   mul;
    [[maybe_unused]] Exp   exponential;

    LayernormFragment result;

    thread_offset_ =
      iterator_D_.thread_start() +
      OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

    NumericArrayConverter<ElementLayernormCompute, ElementOutput, kElementsPerAccess> source_converter;
    OutputVector &source_vector = reinterpret_cast<OutputVector *>(&fragment_C_)[frag_idx];

    bool column_guard = (thread_offset_.column() < extent_.column());

    if (elementwise_.kScale == cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      result = source_converter(elementwise_(accum));
    }else{
      result = source_converter(elementwise_(accum, source_vector));
    }


    ElementLayernormCompute inv_scalar = cutlass::constants::one<ElementLayernormCompute>() / ElementLayernormCompute(extent_.column());

    // Fragment is cleared for non-reachable columns so no need to check against column guard
    accum_sum_element_ = element_sum_accumulator_(result);

    // Square sum is different. Non-reachable columns should've been computed for shift-k
    // Otherwise we will incorrectly have some extra k^2 added into square sum.
    if (column_guard) {
      accum_sum_square_ = (kIsShiftedVariance) ? \
                        square_sum_accumulator_(result, shift_k_frag_[iter_idx * kRowIterations + row_idx]) : \
                        square_sum_accumulator_(result);
    }
    else {
      accum_sum_square_ = ElementLayernormCompute(0);
    }

    accum_sum_element_ *= inv_scalar;
    accum_sum_square_ *= inv_scalar;

    // After performing the in-thread reduction, we then perform cross-thread / in-warp reduction
    CUTLASS_PRAGMA_UNROLL
    for (int i = kHalfThreadsPerRow; i > 0; i >>= 1) {
      accum_sum_element_ += __shfl_xor_sync(0xFFFFFFFF, accum_sum_element_, i);
      accum_sum_square_ += __shfl_xor_sync(0xFFFFFFFF, accum_sum_square_, i);
    }

    // Convert to the output
    NumericArrayConverter<ElementOutput, ElementLayernormCompute, kElementsPerAccess> output_converter;
    OutputVector &output = reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx];
    output = output_converter(result);
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void end_row(int row_idx) {

    using ConvertVarianceOutput = cutlass::NumericConverter<ElementVariance, ElementLayernormCompute>;
    using ConvertMeanOutput = cutlass::NumericConverter<ElementMean, ElementLayernormCompute>;

    ConvertVarianceOutput   convert_variance_output;
    ConvertMeanOutput  convert_mean_output;

    bool is_write_thread = (thread_offset_.row() < extent_.row() && (threadIdx.x % kThreadsPerRow) == 0);
    int row_offset = thread_offset_.row() + blockIdx.y * extent_.row();

    ElementVariance *curr_ptr_sum_square = params_.ptr_Variance + row_offset;
    ElementMean *curr_ptr_element_sum = params_.ptr_Mean + row_offset;

    arch::global_store<ElementVariance, sizeof(ElementVariance)>(
              convert_variance_output(accum_sum_square_),
              (void *)curr_ptr_sum_square,
              is_write_thread);

    arch::global_store<ElementMean, sizeof(ElementMean)>(
              convert_mean_output(accum_sum_element_),
              (void *)curr_ptr_element_sum,
              is_write_thread);

  }

  /// Called after all accumulator elements have been visited
  CUTLASS_DEVICE
  void end_step(int step_idx) {

    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  /// Called after all steps have been completed
  CUTLASS_DEVICE
  void end_epilogue() {

  }

private:

  CUTLASS_DEVICE
  ElementLayernormCompute load_shift_k_(int row_offset, bool is_load) {
    using ConvertShiftK = cutlass::NumericConverter<ElementLayernormCompute, ElementOutput>;
    ConvertShiftK convert_shift_k;
    ElementOutput shift_k_val;

    // Computes the address to load shift_k element
    ElementOutput *curr_ptr_shift_k = params_.ptr_Shifted_K + row_offset;
    // Conditionally loads from global memory
    arch::global_load<ElementOutput, sizeof(ElementOutput)>(shift_k_val, (void *)curr_ptr_shift_k, is_load);
    // Converts data type to return
    ElementLayernormCompute converted_shift_k_val = convert_shift_k(shift_k_val);

    return converted_shift_k_val;
  }

  CUTLASS_DEVICE
  ElementLayernormCompute square_sum_accumulator_(LayernormFragment const &accum) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      auto accum_ = accum[i];
      sum_ += accum_ * accum_;
    }

    return sum_;
  }

  CUTLASS_DEVICE
  ElementLayernormCompute square_sum_accumulator_(LayernormFragment const &accum, ElementLayernormCompute shift_k_val) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      auto accum_ = accum[i] - shift_k_val;
      sum_ += accum_ * accum_;
    }

    return sum_;
  }

  CUTLASS_DEVICE
  ElementLayernormCompute element_sum_accumulator_(LayernormFragment const &accum) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      sum_ += accum[i];
    }

    return sum_;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

///
template <
  typename ElementInputA0_,
  typename LayoutInputA0_,
  typename ElementInputB0_,
  typename LayoutInputB0_,
  typename ElementOutput_,
  typename LayoutOutput_,
  typename ElementCompute_,
  typename EpilogueFunctorOp_,
  typename ThreadblockShape_,
  typename WarpShape_,
  typename InstructionShape_,
  int Stages0,
  int Stages1,
  bool IsShiftedVariance_ = false
>
class GemmLayernorm {
public:

  ///////////////////////////////////////////////////////////////////////////////////////////////

  //
  // Type definitions
  //

  static bool const kInternalTranspose = cutlass::platform::is_same<LayoutOutput_, cutlass::layout::ColumnMajor>::value;
  static bool const kIsShiftedVariance = IsShiftedVariance_;

  // These is mandatory layout.
  using LayoutInputScaleBias = cutlass::layout::RowMajor;

  // These are mandatory data types.
  using ElementLayernormCompute = float;
  using ElementInputScaleBias = cutlass::half_t;

  // These are mandatory params required by mainloop fusion
  using OperatorClass       = cutlass::arch::OpClassTensorOp;
  using ArchTag             = cutlass::arch::Sm80;

  // These are mandatory layouts and data types
  // that are inheritated from pre-defined params

  using LayoutSumSqr = LayoutInputScaleBias;
  using LayoutSum = LayoutInputScaleBias;

  using ElementMean = ElementInputScaleBias;
  using ElementVariance = ElementInputScaleBias;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  using LayoutInputA0 = LayoutInputA0_;
  using LayoutInputB0 = LayoutInputB0_;
  using LayoutInputA1 = LayoutOutput_;
  using LayoutInputB1 = LayoutOutput_;
  using LayoutOutputC0 = LayoutOutput_;
  using LayoutOutputC1 = LayoutOutput_;

  using ElementInputA0 = ElementInputA0_;
  using ElementInputB0 = ElementInputB0_;
  using ElementOutputC0 = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementInputB1 = ElementInputB0_;

  using ElementInputA1 = ElementOutputC0;
  using ElementOutputC1 = ElementOutputC0;

  using EpilogueFunctorOp = EpilogueFunctorOp_;

  using TensorRefA = TensorRef<ElementInputA0, LayoutInputA0>;
  using TensorRefB = TensorRef<ElementInputB0, LayoutInputB0>;
  using TensorRefC = TensorRef<ElementOutputC0, LayoutOutputC0>;
  using TensorVariance = TensorRef<ElementVariance, LayoutSumSqr>;
  using TensorMean = TensorRef<ElementMean, LayoutSum>;

  using ThreadblockShape = ThreadblockShape_;
  using WarpShape        = WarpShape_;
  using InstructionShape = InstructionShape_;

  static int const kStages0 = Stages0;
  static int const kStages1 = Stages1;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  using MapArguments = cutlass::gemm::kernel::detail::MapArguments<
    ElementInputA0,
    LayoutInputA0,
    cutlass::ComplexTransform::kNone,
    128 / cutlass::sizeof_bits<ElementInputA0>::value,
    ElementInputB0,
    LayoutInputB0,
    cutlass::ComplexTransform::kNone,
    128 / cutlass::sizeof_bits<ElementInputB0>::value,
    LayoutOutputC0,
    kInternalTranspose
  >;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementOutputC0,
    typename MapArguments::LayoutC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    SwizzleThreadBlock,
    kStages0,
    true,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementInputA0, ElementInputB0, ElementOutputC0, ElementCompute>::Operator,
    cutlass::gemm::SharedMemoryClearOption::kNone
  >::GemmKernel;

  ///////////////////////////////////////////////////////////////////////////////////////////////

  // Epilogue visitor
  using EpilogueVisitor = kernel::EpilogueVisitorLayerNorm<
    ThreadblockShape,
    DefaultGemmKernel::kThreadCount,
    typename DefaultGemmKernel::Epilogue::OutputTileIterator,
    typename DefaultGemmKernel::Epilogue::AccumulatorFragmentIterator::AccumulatorTile,
    ElementCompute,
    ElementVariance,
    ElementMean,
    ElementLayernormCompute,
    EpilogueFunctorOp,
    kIsShiftedVariance
  >;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
    EpilogueVisitor,
    typename DefaultGemmKernel::Epilogue
  >::Epilogue;

  // GEMM
  using GemmEpilogueFusion = gemm::kernel::GemmWithEpilogueVisitor<
    typename DefaultGemmKernel::Mma,
    Epilogue,
    SwizzleThreadBlock
  >;

  using ApplyFinalReductionKernel = kernel::ApplyFinalReduction<
    ElementVariance,
    ElementMean,
    ElementLayernormCompute,
    ElementOutputC0,
    ThreadblockShape,
    kIsShiftedVariance
  >;

using GemmMainloopFusion = typename cutlass::gemm::device::GemmLayernormMainloopFusion<
  ElementInputA1, LayoutInputA1,
  ElementInputB1, LayoutInputB1,
  ElementInputScaleBias, LayoutInputScaleBias,
  ElementOutputC1, LayoutOutputC1,
  ElementCompute,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueFunctorOp,
  SwizzleThreadBlock,
  kStages1
>;

public:

  /// Arguments class
  struct Arguments {

    typename GemmEpilogueFusion::Arguments         gemm0;
    typename GemmMainloopFusion::Arguments         gemm1;
    typename ApplyFinalReductionKernel::Arguments reduction;
    cutlass::gemm::GemmCoord extend;

    //
    // Methods
    //
    Arguments() { }

    Arguments(
      cutlass::gemm::GemmCoord problem_size0,
      cutlass::gemm::GemmCoord problem_size1,
      ElementInputA0 * ptr_A,
      ElementInputB0 * ptr_B,
      ElementOutputC0 * ptr_C,
      ElementOutputC0 * ptr_D,
      ElementOutputC0 * ptr_E,
      ElementOutputC0 * ptr_O,
      int64_t    ldm_A,
      int64_t    ldm_B,
      int64_t    ldm_C,
      int64_t    ldm_D,
      int64_t    ldm_E,
      int64_t    ldm_O,
      typename EpilogueFunctorOp::Params linear_scaling,
      TensorVariance ref_Variance_,
      TensorMean ref_Mean_,
      TensorVariance ref_Gamma_,
      TensorMean ref_Beta_,
      ElementOutputC0 *ptr_Shifted_K = nullptr
    ):
      gemm0(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {kInternalTranspose ? problem_size0.n() : problem_size0.m(),\
         kInternalTranspose ? problem_size0.m() : problem_size0.n(),\
         problem_size0.k()},
        {kInternalTranspose ? ptr_B : ptr_A, \
        kInternalTranspose ? ldm_B : ldm_A},
        {kInternalTranspose ? ptr_A : ptr_B, \
        kInternalTranspose ? ldm_A : ldm_B},
        typename EpilogueVisitor::Arguments(
          linear_scaling,
          {ptr_C, ldm_C},
          {ptr_D, ldm_D},
          ref_Variance_.data(),
          ref_Mean_.data(),
          ptr_Shifted_K
        )
      ),
      reduction(
        MatrixCoord(kInternalTranspose ? problem_size0.n() : problem_size0.m(),\
                    kInternalTranspose ? problem_size0.m() : problem_size0.n()),
        ref_Variance_,
        ref_Mean_,
        ptr_Shifted_K
      ),
      gemm1(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size1,
        1,
        linear_scaling,
        kInternalTranspose ? ptr_E : ptr_D,
        kInternalTranspose ? ptr_D : ptr_E,
        ref_Variance_.data(),
        ref_Mean_.data(),
        ref_Gamma_.data(),
        ref_Beta_.data(),
        ptr_O,
        ptr_O,
        problem_size1.m() * problem_size1.k(),
        problem_size1.n() * problem_size1.k(),
        problem_size1.n(),
        problem_size1.n(),
        problem_size1.k(),
        problem_size1.k(),
        problem_size1.m() * problem_size1.n(),
        problem_size1.m() * problem_size1.n(),
        kInternalTranspose ? ldm_E : ldm_D,
        kInternalTranspose ? ldm_D : ldm_D,
        ref_Variance_.layout().stride(0),
        ref_Mean_.layout().stride(0),
        ref_Gamma_.layout().stride(0),
        ref_Beta_.layout().stride(0),
        ldm_O,
        ldm_O
      ),
      extend(problem_size0)
    {

    }
  };

  struct Params {

    typename GemmEpilogueFusion::Params         gemm0;
    typename ApplyFinalReductionKernel::Params reduction;
    MatrixCoord extend;
    //
    // Methods
    //
    Params() { }

    Params(Arguments const &args):
      gemm0(args.gemm0),
      reduction(args.reduction),
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
  GemmMainloopFusion gemm_fusion_op;

public:

  /// Ctor
  GemmLayernorm() {

  }

  /// Initialize
  Status initialize(Arguments const &args) {

    params_ = Params(args);
    cutlass::Status status;
    size_t workspace_size = gemm_fusion_op.get_workspace_size(args.gemm1);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    status = gemm_fusion_op.can_implement(args.gemm1);
    CUTLASS_CHECK(status);

    status = gemm_fusion_op.initialize(args.gemm1, workspace.get());
    CUTLASS_CHECK(status);

    return cutlass::Status::kSuccess;
  }

  /// Run
  Status run(cudaStream_t stream) {

    //
    // Launch the GEMM + layernorm kernel
    //

    dim3 gemm_grid = SwizzleThreadBlock().get_grid_shape(params_.gemm0.grid_tiled_shape);
    dim3 gemm_block(GemmEpilogueFusion::kThreadCount, 1, 1);

    int gemm_smem_size = int(sizeof(typename GemmEpilogueFusion::SharedStorage));

    cutlass::Kernel<GemmEpilogueFusion><<<gemm_grid, gemm_block, gemm_smem_size, stream>>>(params_.gemm0);

    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the ApplyFinalReductionKernel
    //

    // always performs reduction from leading dimension
    int leading_dim_0 = kInternalTranspose ? params_.extend.row() : params_.extend.column();
    int leading_dim_1 = kInternalTranspose ? params_.extend.column() : params_.extend.row();

    int thread_per_block = 128;
    int block_per_row = (leading_dim_1 + thread_per_block - 1) / thread_per_block;
    if (block_per_row < 4) {
      thread_per_block = 32;
      block_per_row = (leading_dim_1 + thread_per_block - 1) / thread_per_block;
    }

    dim3 final_reduction_block(thread_per_block);
    dim3 final_reduction_grid(block_per_row);

    Kernel<ApplyFinalReductionKernel><<<
      final_reduction_grid, final_reduction_block, sizeof(typename ApplyFinalReductionKernel::SharedStorage), stream
    >>>(params_.reduction);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the GEMM + mainloop fusion kernel
    //

    cutlass::Status status = gemm_fusion_op();
    CUTLASS_CHECK(status);

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
