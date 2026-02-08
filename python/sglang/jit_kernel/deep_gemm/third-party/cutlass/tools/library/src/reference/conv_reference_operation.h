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
  \brief Defines operations for all CONV operation kinds in CUTLASS Library
*/

#pragma once

#include <iostream>
#include <sstream>
#include <cstring>

#include "cutlass/cutlass.h"

#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "cutlass/library/util.h"
#include "library_internal.h"

#include "cutlass/conv/convolution.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/device/convolution.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  Provider kProvider,
  cutlass::conv::Operator ConvolutionalOperator,
  int ConvDim,
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename LayoutC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ConvertOp_ = NumericConverter<ElementC_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
struct ConvReferenceDispatcher;

/// Dispatcher for Conv2d (partially specialized for kConvDim == 2)
template <
  Provider kProvider,
  cutlass::conv::Operator kConvolutionalOperator,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator,
  typename ConvertOp,
  typename InnerProductOp
>
struct ConvReferenceDispatcher<
  kProvider,
  kConvolutionalOperator, 
  2, 
  ElementA, LayoutA, 
  ElementB, LayoutB, 
  ElementC, LayoutC, 
  ElementCompute, 
  ElementAccumulator, 
  ConvertOp, 
  InnerProductOp> {

  static Status dispatch(
    void const *configuration,
    ElementA *ptr_A,
    ElementB *ptr_B,
    ElementC *ptr_C,
    ElementC *ptr_D,
    ElementCompute alpha,
    ElementCompute beta,
    cudaStream_t stream = nullptr
  ) {

    Conv2dConfiguration const &config = 
      *static_cast<Conv2dConfiguration const *>(configuration);

    // TODO: make below code more general.  It is fixed for NHWC now.
    layout::TensorNHWC layout_a;
    layout::TensorNHWC layout_b;
    layout::TensorNHWC layout_c;

    layout_a.stride() =
        make_Coord(int32_t(config.stride_a[0]), 
                   int32_t(config.stride_a[1]), 
                   int32_t(config.stride_a[2]));

    layout_b.stride() =
        make_Coord(int32_t(config.stride_b[0]), 
                   int32_t(config.stride_b[1]), 
                   int32_t(config.stride_b[2]));

    layout_c.stride() =
        make_Coord(int32_t(config.stride_c[0]), 
                   int32_t(config.stride_c[1]), 
                   int32_t(config.stride_c[2]));

    if (kProvider == Provider::kReferenceHost) {

      cutlass::reference::host::Conv2d<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC ,
        LayoutC,
        ElementCompute,
        ElementAccumulator,
        ElementC,
        ConvertOp,
        InnerProductOp
      >(
        kConvolutionalOperator,
        config.problem_size,
        {ptr_A, layout_a},
        {ptr_B, layout_b},
        {ptr_C, layout_c},
        {ptr_D, layout_c},
        alpha,
        beta
      );

      return Status::kSuccess;
    }
    else if (kProvider == Provider::kReferenceDevice) {
      return cutlass::reference::device::Conv2d<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementCompute,
        ElementAccumulator,
        ConvertOp,
        InnerProductOp
      >(
        kConvolutionalOperator,
        config.problem_size,
        {ptr_A, layout_a},
        {ptr_B, layout_b},
        {ptr_C, layout_c},
        {ptr_D, layout_c},
        alpha,
        beta,
        stream
      );
    }
    return Status::kErrorNotSupported;
  }
};

/// Dispatcher for Conv3d (partially specialized for kConvDim == 3)
template <
  Provider kProvider,
  cutlass::conv::Operator kConvolutionalOperator,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator,
  typename ConvertOp,
  typename InnerProductOp
>
struct ConvReferenceDispatcher<
  kProvider,
  kConvolutionalOperator, 
  3, 
  ElementA, LayoutA, 
  ElementB, LayoutB, 
  ElementC, LayoutC, 
  ElementCompute, 
  ElementAccumulator, 
  ConvertOp, 
  InnerProductOp> {

  static Status dispatch(
    void const *configuration,
    ElementA *ptr_A,
    ElementB *ptr_B,
    ElementC *ptr_C,
    ElementC *ptr_D,
    ElementCompute alpha,
    ElementCompute beta,
    cudaStream_t stream = nullptr
  ) {

    Conv3dConfiguration const &config = 
      *static_cast<Conv3dConfiguration const *>(configuration);
    
    ConvKind const conv_kind = ConvKindMap<kConvolutionalOperator>::kId;

    if (kProvider == Provider::kReferenceHost) {
      cutlass::reference::host::Conv3d<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC ,
        LayoutC,
        ElementCompute,
        ElementAccumulator,
        ConvertOp,
        InnerProductOp
      >(
        kConvolutionalOperator,
        config.problem_size,
        {ptr_A, config.layout_a(conv_kind)},
        {ptr_B, config.layout_b(conv_kind)},
        {ptr_C, config.layout_c(conv_kind)},
        {ptr_D, config.layout_c(conv_kind)},
        alpha,
        beta
      );

      return Status::kSuccess;
    }
    else if (kProvider == Provider::kReferenceDevice) {
      return cutlass::reference::device::Conv3d<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC,
        ElementCompute,
        ElementAccumulator,
        ConvertOp,
        InnerProductOp
      >(
        kConvolutionalOperator,
        config.problem_size,
        {ptr_A, config.layout_a(conv_kind)},
        {ptr_B, config.layout_b(conv_kind)},
        {ptr_C, config.layout_c(conv_kind)},
        {ptr_D, config.layout_c(conv_kind)},
        alpha,
        beta,
        stream
      );
    }
    return Status::kErrorNotSupported;
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  Provider Provider_,
  cutlass::conv::Operator ConvolutionalOperator,
  int ConvDim,
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename LayoutC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ConvertOp_ = NumericConverter<ElementC_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
class ConvReferenceOperation : public Operation {
public:
  static Provider const kProvider = Provider_;
  static cutlass::conv::Operator const kConvolutionalOperator = ConvolutionalOperator;
  static int const kConvDim = ConvDim;

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using ElementCompute = ElementCompute_;
  using ElementAccumulator = ElementAccumulator_;
  using ConvertOp = ConvertOp_;
  using InnerProductOp = InnerProductOp_;

protected:

  /// Storage for the name string
  std::string name_;

  ///
  ConvDescription description_;

public:

  /// Constructor
  ConvReferenceOperation() {
    
    // Basic information
    description_.provider = kProvider;
    description_.kind = (kConvDim == 2 ? OperationKind::kConv2d : OperationKind::kConv3d);
    description_.conv_kind = ConvKindMap<kConvolutionalOperator>::kId;
    description_.conv_dim = kConvDim;

    // Tensor description
    description_.A = make_TensorDescription<ElementA, LayoutA>();
    description_.B = make_TensorDescription<ElementB, LayoutB>();
    description_.C = make_TensorDescription<ElementC, LayoutC>();
    
    // Epilogue compute and accumulator type description
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;

    description_.tile_description.math_instruction.element_accumulator =
      NumericTypeMap<ElementAccumulator>::kId;

    // Iterator algorithm for convolution reference
    description_.iterator_algorithm = IteratorAlgorithmID::kNone;
    
    // Compute capability for convolution reference
    description_.tile_description.minimum_compute_capability = 
      (kProvider == Provider::kReferenceDevice ? 50 : 0);

    description_.tile_description.maximum_compute_capability = 1024;

    // Procedural name
    std::stringstream ss;

    ss << "conv" << kConvDim << "d_" << to_string(description_.conv_kind) 
      << "_reference_" << to_string(description_.provider)
      << "_" << to_string(description_.A.element) << to_string(description_.A.layout)
      << "_" << to_string(description_.B.element) << to_string(description_.B.layout)
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

    switch (kConvDim) {
    case 2:
      return sizeof(Conv2dConfiguration);
    case 3:
      return sizeof(Conv3dConfiguration);
    default:
      break;
    }

    return 0;
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

    std::memcpy(host_workspace, configuration, get_host_workspace_size(configuration));

    return Status::kSuccess;
  }

  virtual Status run(
    void const *arguments,
    void *host_workspace,
    void *device_workspace = nullptr,
    cudaStream_t stream = nullptr) const {

    ConvArguments const  &args = *static_cast<ConvArguments const *>(arguments);

    ElementCompute alpha;
    ElementCompute beta;

    alpha = *static_cast<ElementCompute const *>(args.alpha);
    beta = *static_cast<ElementCompute const *>(args.beta);

    // TODO - respect pointer mode

    // Invoke 2D or 3D convolution
    return detail::ConvReferenceDispatcher<
      kProvider,
      kConvolutionalOperator,
      kConvDim,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementCompute,
      ElementAccumulator,
      ConvertOp,
      InnerProductOp
    >::dispatch(
      host_workspace,
      static_cast<ElementA *>(const_cast<void *>(args.A)),
      static_cast<ElementB *>(const_cast<void *>(args.B)),
      static_cast<ElementC *>(const_cast<void *>(args.C)),
      static_cast<ElementC *>(args.D),
      alpha,
      beta,
      stream
    );
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Constructs Fprop reference operators.
template <
  int kConvDim,
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename LayoutC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ConvertOp_ = NumericConverter<ElementC_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_conv_fprop(Manifest &manifest) {
#if !defined(CUTLASS_PROFILER_DISABLE_REFERENCE)
  manifest.append(new ConvReferenceOperation<
    Provider::kReferenceHost,
    cutlass::conv::Operator::kFprop,
    kConvDim,
    ElementA_, LayoutA_,
    ElementB_, LayoutB_,
    ElementC_, LayoutC_,
    ElementCompute_,
    ElementAccumulator_,
    ConvertOp_,
    InnerProductOp_
  >);

  manifest.append(new ConvReferenceOperation<
    Provider::kReferenceDevice,
    cutlass::conv::Operator::kFprop,
    kConvDim,
    ElementA_, LayoutA_,
    ElementB_, LayoutB_,
    ElementC_, LayoutC_,
    ElementCompute_,
    ElementAccumulator_,
    ConvertOp_,
    InnerProductOp_
  >);
#endif // !defined(CUTLASS_PROFILER_DISABLE_REFERENCE)
}

/// Constructs Dgrad and Wgrad reference operators.
template <
  int kConvDim,
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename LayoutC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ConvertOp_ = NumericConverter<ElementC_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_conv_backwards(Manifest &manifest) {
#if !defined(CUTLASS_PROFILER_DISABLE_REFERENCE)
  manifest.append(new ConvReferenceOperation<
    Provider::kReferenceHost,
    cutlass::conv::Operator::kDgrad,
    kConvDim,
    ElementA_, LayoutA_,
    ElementB_, LayoutB_,
    ElementC_, LayoutC_,
    ElementCompute_,
    ElementAccumulator_,
    ConvertOp_,
    InnerProductOp_
  >);

  manifest.append(new ConvReferenceOperation<
    Provider::kReferenceDevice,
    cutlass::conv::Operator::kDgrad,
    kConvDim,
    ElementA_, LayoutA_,
    ElementB_, LayoutB_,
    ElementC_, LayoutC_,
    ElementCompute_,
    ElementAccumulator_,
    ConvertOp_,
    InnerProductOp_
  >);

  manifest.append(new ConvReferenceOperation<
    Provider::kReferenceHost,
    cutlass::conv::Operator::kWgrad,
    kConvDim,
    ElementA_, LayoutA_,
    ElementB_, LayoutB_,
    ElementC_, LayoutC_,
    ElementCompute_,
    ElementAccumulator_,
    ConvertOp_,
    InnerProductOp_
  >);

  manifest.append(new ConvReferenceOperation<
    Provider::kReferenceDevice,
    cutlass::conv::Operator::kWgrad,
    kConvDim,
    ElementA_, LayoutA_,
    ElementB_, LayoutB_,
    ElementC_, LayoutC_,
    ElementCompute_,
    ElementAccumulator_,
    ConvertOp_,
    InnerProductOp_
  >);
#endif // !defined(CUTLASS_PROFILER_DISABLE_REFERENCE)
}

/// Six operators for the price of one.
template <
  int kConvDim,
  typename ElementA_,
  typename LayoutA_,
  typename ElementB_,
  typename LayoutB_,
  typename ElementC_,
  typename LayoutC_,
  typename ElementCompute_,
  typename ElementAccumulator_ = ElementCompute_,
  typename ConvertOp_ = NumericConverter<ElementC_, ElementCompute_>,
  typename InnerProductOp_ = multiply_add<ElementAccumulator_>
>
void make_conv_all(Manifest &manifest) {

  make_conv_fprop<
    kConvDim,
    ElementA_, LayoutA_,
    ElementB_, LayoutB_,
    ElementC_, LayoutC_,
    ElementCompute_,
    ElementAccumulator_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);

  make_conv_backwards<
    kConvDim,
    ElementA_, LayoutA_,
    ElementB_, LayoutB_,
    ElementC_, LayoutC_,
    ElementCompute_,
    ElementAccumulator_,
    ConvertOp_,
    InnerProductOp_
  >(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

