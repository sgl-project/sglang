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

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {

template<
  typename TensorLayout_,                             ///! The original output tensor layout
  typename OutputIteratorLayout_,                     ///! Layout used by epilogue output iterator
  typename TensorRef_,                                ///! Input tensor to epilogue output iterator
  conv::Operator ConvOperator,                        ///! Convolutional operator (Fprop, Dgrad, Wgrad)
  typename ConvProblemSize_                          ///! Convolutional operator on 2D or 3D problem
>
struct ConvOutputIteratorParameter {

  using TensorLayout = TensorLayout_;
  using OutputIteratorLayout = OutputIteratorLayout_;
  using OutputTensorCoord = typename OutputIteratorLayout::TensorCoord;
  using TensorRef = TensorRef_;
  static conv::Operator const kConvolutionalOperator = ConvOperator;
  using ConvProblemSize = ConvProblemSize_;

  /// Wgrad stride idx for implicit gemm algorithm 
  // Conv2d row-major matrix (KxRSC) 
  // Conv3d row-major matrix (KxTRSC)
  static int const kWgradStrideIdx = 
    platform::is_same<TensorLayout, layout::TensorNHWC>::value ? 2 : 3;

  /// This chooses the appropriate stride element of the C tensor.
  static int const kTensorStrideIdx = 
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradStrideIdx : 0);

  CUTLASS_HOST_DEVICE
  static OutputIteratorLayout layout(const TensorRef & ref) {
    return ref.stride(kTensorStrideIdx);
  }

  CUTLASS_HOST_DEVICE
  static OutputTensorCoord extent(ConvProblemSize problem_size) {
    return conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn();
  }
};

template<
  typename TensorRef_,                                ///! Input tensor to epilogue output iterator
  typename ConvProblemSize_                          ///! Convolutional operator on 2D or 3D problem
>
struct ConvOutputIteratorParameter<layout::TensorNHWC, layout::TensorNHWC, TensorRef_, conv::Operator::kFprop, ConvProblemSize_> {

  using TensorLayout = layout::TensorNHWC;
  using OutputIteratorLayout = layout::TensorNHWC;
  using MappedLayout = layout::RowMajor;
  using OutputTensorCoord = typename OutputIteratorLayout::TensorCoord;
  using MappedTensorCoord = typename MappedLayout::TensorCoord;
  using TensorRef = TensorRef_;
  static conv::Operator const kConvolutionalOperator = conv::Operator::kFprop;
  using ConvProblemSize = ConvProblemSize_;

  CUTLASS_HOST_DEVICE
  static OutputIteratorLayout layout(const TensorRef & ref) {
    return ref.stride();
  }

  CUTLASS_HOST_DEVICE
  static MappedTensorCoord extent(ConvProblemSize problem_size) {
    return conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn();
  }
};

template<
  typename TensorRef_,                                ///! Input tensor to epilogue output iterator
  typename ConvProblemSize_                          ///! Convolutional operator on 2D or 3D problem
>
struct ConvOutputIteratorParameter<layout::TensorNHWC, layout::TensorNHWC, TensorRef_, conv::Operator::kDeconv, ConvProblemSize_> {

  using TensorLayout = layout::TensorNHWC;
  using OutputIteratorLayout = layout::TensorNHWC;
  using MappedLayout = layout::RowMajor;
  using OutputTensorCoord = typename OutputIteratorLayout::TensorCoord;
  using MappedTensorCoord = typename MappedLayout::TensorCoord;
  using TensorRef = TensorRef_;
  static conv::Operator const kConvolutionalOperator = conv::Operator::kDeconv;
  using ConvProblemSize = ConvProblemSize_;

  CUTLASS_HOST_DEVICE
  static OutputIteratorLayout layout(const TensorRef & ref) {
    return ref.stride();
  }

  CUTLASS_HOST_DEVICE
  static MappedTensorCoord extent(ConvProblemSize problem_size) {
    return conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn();
  }
};

template<
  typename TensorRef_,                                ///! Input tensor to epilogue output iterator
  typename ConvProblemSize_                          ///! Convolutional operator on 2D or 3D problem
>
struct ConvOutputIteratorParameter<layout::TensorNDHWC, layout::TensorNDHWC, TensorRef_, conv::Operator::kFprop, ConvProblemSize_> {

  using TensorLayout = layout::TensorNDHWC;
  using OutputIteratorLayout = layout::TensorNDHWC;
  using MappedLayout = layout::RowMajor;
  using OutputTensorCoord = typename OutputIteratorLayout::TensorCoord;
  using MappedTensorCoord = typename MappedLayout::TensorCoord;
  using TensorRef = TensorRef_;
  static conv::Operator const kConvolutionalOperator = conv::Operator::kFprop;
  using ConvProblemSize = ConvProblemSize_;

  CUTLASS_HOST_DEVICE
  static OutputIteratorLayout layout(const TensorRef & ref) {
    return ref.stride();
  }

  CUTLASS_HOST_DEVICE
  static MappedTensorCoord extent(ConvProblemSize problem_size) {
    return conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn();
  }
};

template<
  typename TensorRef_,                                ///! Input tensor to epilogue output iterator
  typename ConvProblemSize_                          ///! Convolutional operator on 2D or 3D problem
>
struct ConvOutputIteratorParameter<layout::TensorNDHWC, layout::TensorNDHWC, TensorRef_, conv::Operator::kDeconv, ConvProblemSize_> {

  using TensorLayout = layout::TensorNDHWC;
  using OutputIteratorLayout = layout::TensorNDHWC;
  using MappedLayout = layout::RowMajor;
  using OutputTensorCoord = typename OutputIteratorLayout::TensorCoord;
  using MappedTensorCoord = typename MappedLayout::TensorCoord;
  using TensorRef = TensorRef_;
  static conv::Operator const kConvolutionalOperator = conv::Operator::kDeconv;
  using ConvProblemSize = ConvProblemSize_;

  CUTLASS_HOST_DEVICE
  static OutputIteratorLayout layout(const TensorRef & ref) {
    return ref.stride();
  }

  CUTLASS_HOST_DEVICE
  static MappedTensorCoord extent(ConvProblemSize problem_size) {
    return conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn();
  }
};

template <
  int InterleavedK,
  typename TensorRef_,
  conv::Operator ConvOperator,
  typename ConvProblemSize_
>
struct ConvOutputIteratorParameter<
  layout::TensorNCxHWx<InterleavedK>, 
  layout::TensorNCxHWx<InterleavedK>,
  TensorRef_,
  ConvOperator,
  ConvProblemSize_>
{ 

  using TensorLayout = typename layout::TensorNCxHWx<InterleavedK>;
  using OutputIteratorLayout = typename layout::TensorNCxHWx<InterleavedK>;
  using OutputTensorCoord = typename OutputIteratorLayout::TensorCoord;
  using TensorRef = TensorRef_;
  static conv::Operator const kConvolutionalOperator = ConvOperator;
  using ConvProblemSize = ConvProblemSize_;

  CUTLASS_HOST_DEVICE
  static OutputIteratorLayout layout(const TensorRef & ref) {
    return ref.stride();
  }

  CUTLASS_HOST_DEVICE
  static OutputTensorCoord extent(ConvProblemSize problem_size) {
    return problem_size.output_extent();
  }

};

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass
