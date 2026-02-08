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
/* \file
  \brief Defines reference operations for blockwise/groupwise GEMM operation kinds in CUTLASS Library
*/



#pragma once

#include <iostream>
#include <sstream>
#include <cstring>

#include "cutlass/cutlass.h"

#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "cutlass/library/util.h"
#include "cutlass/util/packed_stride.hpp"
#include "library_internal.h"

#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  Provider Provider_,
  typename ElementA_, 
  typename LayoutA_,
  typename LayoutSFA_,
  typename ElementSFA_,
  typename ElementB_,
  typename LayoutB_,
  typename LayoutSFB_,
  typename ElementSFB_,
  typename ElementC_,
  typename LayoutC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  typename ConvertOp_ = NumericConverter<ElementD_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
class BlockwiseGemmReferenceOperation : public Operation {
public:
  static Provider const kProvider = Provider_;

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementSFA = ElementSFA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementSFB = ElementSFB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using ElementD = ElementD_;
  using ElementCompute = ElementCompute_;
  using ElementAccumulator = ElementAccumulator_;
  using ConvertOp = ConvertOp_;
  using InnerProductOp = InnerProductOp_;

protected:

  /// Storage for the name string
  std::string name_;

  ///
  BlockwiseGemmDescription description_;

public:

  /// Constructor
  BlockwiseGemmReferenceOperation(int SFMVecSize_, int SFNVecSize_, int SFKVecSize_)
    : SFMVecSize(SFMVecSize_), SFNVecSize(SFNVecSize_), SFKVecSize(SFKVecSize_) {
    
    // Basic information
    description_.provider = kProvider;
    description_.kind = OperationKind::kBlockwiseGemm;
    description_.gemm_kind = GemmKind::kUniversal;

    // Tensor description
    description_.A = make_TensorDescription<ElementA, LayoutA>();
    description_.SFA = make_TensorDescription<ElementSFA, LayoutSFA_>();
    description_.B = make_TensorDescription<ElementB, LayoutB>();
    description_.SFB = make_TensorDescription<ElementSFB, LayoutSFB_>();
    description_.C = make_TensorDescription<ElementC, LayoutC>();
    description_.D = make_TensorDescription<ElementD, LayoutC>();
    
    // Epilogue compute and accumulator type description
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.tile_description.math_instruction.element_accumulator =
      NumericTypeMap<ElementAccumulator>::kId;

    // Compute capability for gemm reference
    description_.tile_description.minimum_compute_capability = 
      (kProvider == Provider::kReferenceDevice ? 50 : 0);

    description_.tile_description.maximum_compute_capability = 1024;

    description_.SFMVecSize = SFMVecSize;
    description_.SFNVecSize = SFNVecSize;
    description_.SFKVecSize = SFKVecSize;

    // Procedural name
    std::stringstream ss;

    ss << "gemm"  
      << "_reference_" << to_string(description_.provider)
      << "_" << to_string(description_.A.element) << to_string(description_.A.layout)
      << "_" << to_string(description_.SFA.element) << SFMVecSize << "x" << SFKVecSize << to_string(description_.SFA.layout)
      << "_" << to_string(description_.B.element) << to_string(description_.B.layout)
      << "_" << to_string(description_.SFB.element)  << SFNVecSize << "x" << SFKVecSize << to_string(description_.SFB.layout)
      << "_" << to_string(description_.C.element) << to_string(description_.C.layout)
      << "_" << to_string(description_.tile_description.math_instruction.element_accumulator);

    name_ = ss.str();

    description_.name = name_.c_str();

    // Epilogue compute and accumulator type description
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.tile_description.math_instruction.element_accumulator =
      NumericTypeMap<ElementAccumulator>::kId;
  }

  /// Returns the description of the GEMM operation
  virtual OperationDescription const & description() const {
    return description_;
  }

  virtual Status can_implement(
    void const *configuration,
    void const *arguments) const {

    return Status::kSuccess;
  }

  virtual uint64_t get_host_workspace_size(
    void const *configuration) const {

    return sizeof(GemmUniversalConfiguration);
  }

  virtual uint64_t get_device_workspace_size(
    void const *configuration,
    void const *arguments = nullptr) const {

    return 0;
  }

  virtual Status initialize(
    void const *configuration,
    void *host_workspace,
    void *device_workspace = nullptr,
    cudaStream_t stream = nullptr) const {
    return Status::kSuccess;
  }

  virtual Status run(
    void const *arguments,
    void *host_workspace,
    void *device_workspace = nullptr,
    cudaStream_t stream = nullptr) const {
    using namespace cute;

    BlockwiseGemmArguments const &args = *static_cast<BlockwiseGemmArguments const *>(arguments);

    // Construct cute::Tensor A/B/C 

    int M = args.problem_size.m();
    int N = args.problem_size.n();
    int K = args.problem_size.k();
    int L = args.batch_count;

    auto problem_shape_MNKL = cute::make_shape(M, N, K, L);

    auto alpha = *(static_cast<ElementCompute const*>(args.alpha));
    auto beta = *(static_cast<ElementCompute const*>(args.beta));

    using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
    using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
    using StrideD = cutlass::gemm::TagToStrideC_t<LayoutC>;

    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));
    using BlockwiseConfig = cutlass::detail::RuntimeBlockwiseScaleConfig<>;
    auto A = cute::make_tensor(static_cast<ElementA const*>(args.A),
        cute::make_layout(cute::make_shape(M, K, L), stride_a));
    auto SfA = make_tensor(static_cast<ElementSFA const*>(args.SFA), BlockwiseConfig::tile_atom_to_shape_SFA(problem_shape_MNKL, cute::make_tuple(SFMVecSize, SFNVecSize, SFKVecSize)));

    auto B = cute::make_tensor(static_cast<ElementB const*>(args.B),
        cute::make_layout(cute::make_shape(N, K, L), stride_b));
    auto SfB = make_tensor(static_cast<ElementSFB const*>(args.SFB), BlockwiseConfig::tile_atom_to_shape_SFB(problem_shape_MNKL, cute::make_tuple(SFMVecSize, SFNVecSize, SFKVecSize)));

    auto C = [&]() {
      if constexpr (not is_same_v<ElementC, void>) {
        return cute::make_tensor(static_cast<ElementC const*>(args.C),
            cute::make_layout(cute::make_shape(M, N, L), stride_c));
      }
      else {
        return cute::make_tensor(static_cast<ElementD const*>(nullptr),
            cute::make_layout(cute::make_shape(M, N, L), stride_c));
      }
    }();

    auto D = cute::make_tensor(static_cast<ElementD *>(args.D),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));

    cutlass::reference::host::GettBlockScalingMainloopParams<ElementAccumulator, 
        decltype(A), decltype(SfA), 
        decltype(B), decltype(SfB)> 
        mainloop_params{A, SfA, B, SfB};

    //  W/O SF generation
    cutlass::reference::host::GettEpilogueParams<
        ElementCompute, ElementAccumulator, ElementAccumulator, ElementCompute,
        decltype(C), decltype(D)>
        epilogue_params{alpha, beta, C, D};

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    return Status::kSuccess;
  }

private:
  int SFMVecSize;
  int SFNVecSize;
  int SFKVecSize;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_,
  typename ElementSFA_,
  typename ElementB_,
  typename ElementSFB_,
  typename ElementC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  typename ConvertOp_ = NumericConverter<ElementD_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_blockwise_gemm(Manifest &manifest, int SFMVecSize, int SFNVecSize, int SFKVecSize) {
  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));

  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));


  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));

  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));


  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));

  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));


  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));

  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));



  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));
  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));


  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::RowMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));
  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));


  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));
  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));


  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));

  manifest.append(new BlockwiseGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::ColumnMajor,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ConvertOp_,
    InnerProductOp_
  >(SFMVecSize, SFNVecSize, SFKVecSize));


}

template<class ElementC,
         class ElementD>
void initialize_blockwise_gemm_reference_operations_given_C_and_D(Manifest &manifest) {
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 1 , 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 128, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 1, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 128, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 64, 1, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 64, 128, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 32, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 32, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 64, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 64, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 256, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 256, 128);


  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 1 , 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 128, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 1, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 128, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 64, 1 , 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 64, 128, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 32, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 32, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 64, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 64, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 256, 128);
  make_blockwise_gemm<
    float_e4m3_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 256, 128);

  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 1 , 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 128, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 1, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 128, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 64, 1, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 64, 128, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 32, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 32, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 64, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 64, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 256, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e4m3_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 256, 128);

  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 1 , 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 128, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 1, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 128, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 64, 1 , 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 64, 128, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 32, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 32, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 64, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 64, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 128, 256, 128);
  make_blockwise_gemm<
    float_e5m2_t /*A*/, float /*SFA*/, float_e5m2_t /*B*/, float /*SFB*/,
    ElementC /*D*/, float /*Compute*/, float /*Accum*/, ElementD /*D*/
  >(manifest, 1, 256, 128);

}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

