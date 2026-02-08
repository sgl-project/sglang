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
  \brief Defines reference operations for block-scaled GEMM operation kinds in CUTLASS Library
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
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

namespace detail {
template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  Provider Provider_,
  typename ElementA_,
  typename LayoutA_,
  typename ElementSFA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementSFB_,
  typename ElementC_,
  typename LayoutC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  typename ElementSFD_ = void,
  typename LayoutSFD_ = LayoutC_,
  int SFVecSize_ = 32,
  int EpilogueSFVecSize_ = 0,
  typename ConvertOp_ = NumericConverter<ElementD_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
class BlockScaledGemmReferenceOperation : public Operation {
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
  using ElementSFD = ElementSFD_;
  using LayoutSFD = LayoutSFD_;
  using ElementCompute = ElementCompute_;
  using ElementAccumulator = ElementAccumulator_;
  using ConvertOp = ConvertOp_;
  using InnerProductOp = InnerProductOp_;
  constexpr static int SFVecSize = SFVecSize_;
  constexpr static int EpilogueSFVecSize = EpilogueSFVecSize_;

protected:

  /// Storage for the name string
  std::string name_;

  ///
  BlockScaledGemmDescription description_;

public:

  /// Constructor
  BlockScaledGemmReferenceOperation() {

    // Basic information
    description_.provider = kProvider;
    description_.kind = OperationKind::kBlockScaledGemm;
    description_.gemm_kind = GemmKind::kUniversal;

    // Tensor description
    description_.A = make_TensorDescription<ElementA, LayoutA>();
    description_.SFA = make_TensorDescription<ElementSFA, LayoutA>();
    description_.B = make_TensorDescription<ElementB, LayoutB>();
    description_.SFB = make_TensorDescription<ElementSFB, LayoutB>();
    description_.C = make_TensorDescription<ElementC, LayoutC>();
    description_.D = make_TensorDescription<ElementD, LayoutC>();
    description_.SFD = make_TensorDescription<ElementSFD, LayoutSFD>();

    // Epilogue compute and accumulator type description
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.tile_description.math_instruction.element_accumulator =
      NumericTypeMap<ElementAccumulator>::kId;

    // Compute capability for gemm reference
    description_.tile_description.minimum_compute_capability =
      (kProvider == Provider::kReferenceDevice ? 50 : 0);

    description_.tile_description.maximum_compute_capability = 1024;

    description_.SFVecSize = SFVecSize;
    description_.EpilogueSFVecSize = EpilogueSFVecSize;

    // Procedural name
    std::stringstream ss;

    ss << "gemm"
      << "_reference_" << to_string(description_.provider)
      << "_" << to_string(description_.A.element) << to_string(description_.A.layout)
      << "_" << to_string(description_.SFA.element) << to_string(description_.SFA.layout)
      << "_" << to_string(description_.B.element) << to_string(description_.B.layout)
      << "_" << to_string(description_.SFB.element) << to_string(description_.SFB.layout)
      << "_" << to_string(description_.C.element) << to_string(description_.C.layout)
      << "_" << to_string(description_.SFD.element) << to_string(description_.SFD.layout)
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

    BlockScaledGemmArguments const &args = *static_cast<BlockScaledGemmArguments const *>(arguments);

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

    using Sm1xxBlockScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
    auto A = cute::make_tensor(detail::make_iterator(static_cast<ElementA const*>(args.A)),
        cute::make_layout(cute::make_shape(M, K, L), stride_a));
    auto SfA = make_tensor(static_cast<ElementSFA const*>(args.SFA), Sm1xxBlockScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL));

    auto B = cute::make_tensor(detail::make_iterator(static_cast<ElementB const*>(args.B)),
        cute::make_layout(cute::make_shape(N, K, L), stride_b));
    auto SfB = make_tensor(static_cast<ElementSFB const*>(args.SFB), Sm1xxBlockScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL));

    auto C = [&]() {
      if constexpr (not is_same_v<ElementC, void>) {
        return cute::make_tensor(detail::make_iterator(static_cast<ElementC const*>(args.C)),
            cute::make_layout(cute::make_shape(M, N, L), stride_c));
      }
      else {
        return cute::make_tensor(detail::make_iterator(static_cast<ElementD const*>(nullptr)),
            cute::make_layout(cute::make_shape(M, N, L), stride_c));
      }
    }();

    auto D = cute::make_tensor(detail::make_iterator(static_cast<ElementD *>(args.D)),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));

    cutlass::reference::host::GettBlockScalingMainloopParams<ElementAccumulator,
        decltype(A), decltype(SfA),
        decltype(B), decltype(SfB)>
        mainloop_params{A, SfA, B, SfB};

    if constexpr (not is_same_v<ElementSFD, void>) {

      using Sm1xxBlockScaledOutputConfig= cutlass::detail::Sm1xxBlockScaledOutputConfig<
                                              EpilogueSFVecSize
                                            >;

      auto SfD = cute::make_tensor(detail::make_iterator(static_cast<ElementSFD*>(args.SFD)), Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(problem_shape_MNKL));

      cutlass::reference::host::GettBlockScalingEpilogueParams<
          ElementCompute, ElementAccumulator, ElementCompute,
          decltype(C), decltype(D), decltype(SfD), Int<EpilogueSFVecSize>, cutlass::reference::host::SfStrategy::SfDGen>
          epilogue_params{alpha, beta, C, D, SfD, *(static_cast<ElementCompute const*>(args.norm_constant))};

      cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
    }
    else {
      //  W/O SF generation
      auto SfD = cute::make_tensor(static_cast<ElementSFA *>(nullptr),
          cute::make_layout(cute::make_shape(M, N, L))); // not used.
      cutlass::reference::host::GettBlockScalingEpilogueParams<
          ElementCompute, ElementAccumulator, ElementCompute,
          decltype(C), decltype(D), decltype(SfD)>
          epilogue_params{alpha, beta, C, D, SfD};

      cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
    }

    return Status::kSuccess;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_,
  typename ElementSFA_,
  typename ElementB_,
  typename ElementSFB_,
  typename ElementC_,
  typename ElementCompute_,
  typename ElementSFD_ = void,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  int SFVecSize = 32,
  int EpilogueSFVecSize = SFVecSize,
  typename ConvertOp_ = NumericConverter<ElementD_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_block_scaled_gemm_tn(Manifest &manifest) {
#if !defined(CUTLASS_PROFILER_DISABLE_REFERENCE)
  manifest.append(new BlockScaledGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ElementSFD_,
    cutlass::layout::RowMajor,
    SFVecSize,
    EpilogueSFVecSize,
    ConvertOp_,
    InnerProductOp_
  >);
#endif // !defined(CUTLASS_PROFILER_DISABLE_REFERENCE)
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA_,
  typename ElementSFA_,
  typename ElementB_,
  typename ElementSFB_,
  typename ElementC_,
  typename ElementCompute_,
  typename ElementSFD_ = void,
  typename ElementAccumulator_ = ElementCompute_,
  typename ElementD_ = ElementC_,
  int SFVecSize = 32,
  int EpilogueSFVecSize = SFVecSize,
  typename ConvertOp_ = NumericConverter<ElementD_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_block_scaled_gemm(Manifest &manifest) {
  ///
  /// A is Row , B is Col
  ///
  manifest.append(new BlockScaledGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ElementSFD_,
    cutlass::layout::RowMajor,
    SFVecSize,
    EpilogueSFVecSize,
    ConvertOp_,
    InnerProductOp_
  >);
  manifest.append(new BlockScaledGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::RowMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::ColumnMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ElementSFD_,
    cutlass::layout::RowMajor,
    SFVecSize,
    EpilogueSFVecSize,
    ConvertOp_,
    InnerProductOp_
  >);
  ///
  /// A is Col , B is Row
  ///
  manifest.append(new BlockScaledGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::RowMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::RowMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ElementSFD_,
    cutlass::layout::RowMajor,
    SFVecSize,
    EpilogueSFVecSize,
    ConvertOp_,
    InnerProductOp_
  >);
  manifest.append(new BlockScaledGemmReferenceOperation<
    Provider::kReferenceHost,
    ElementA_,
    cutlass::layout::ColumnMajor,
    ElementSFA_,
    ElementB_,
    cutlass::layout::RowMajor,
    ElementSFB_,
    ElementC_,
    cutlass::layout::ColumnMajor,
    ElementCompute_,
    ElementAccumulator_,
    ElementD_,
    ElementSFD_,
    cutlass::layout::RowMajor,
    SFVecSize,
    EpilogueSFVecSize,
    ConvertOp_,
    InnerProductOp_
  >);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

