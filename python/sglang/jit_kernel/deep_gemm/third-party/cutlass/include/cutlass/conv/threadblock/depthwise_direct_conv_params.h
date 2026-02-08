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
/*! 
  \file 
  \brief Extracts the host-params objects into non-template code.
*/

#pragma once

#define TRACE_CONV_PARAMS_INITIALIZERS_ENABLED 0

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"

#if TRACE_CONV_PARAMS_INITIALIZERS_ENABLED
#include <fstream>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure used for DepthwiseFpropActivationDirect2dConvTileAccessIteratorOptimized
template<typename Layout_ = layout::TensorNHWC >
struct Depthwise2dFpropDirectConvParams;

/// Parameters structure used for DepthwiseFpropActivationDirect2dConvTileAccessIteratorFixedStrideDilation
template<typename Layout_ = layout::TensorNHWC >
struct Depthwise2dFpropDirectConvActivationIteratorFixedStrideDilationParams;

/// Parameters structure used for DepthwiseFpropFilterDirectConvTileAccessIteratorOptimized
template<typename Layout_ = layout::TensorNHWC >
struct Depthwise2dFpropDirectConvFilterIteratorParams;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure used for DepthwiseFpropActivationDirect2dConvTileAccessIteratorOptimized
template<>
struct Depthwise2dFpropDirectConvParams<layout::TensorNHWC> {
  
  using Layout = layout::TensorNHWC;

  Layout layout;

  int32_t activation_tile_h;
  int32_t activation_tile_w;
  int32_t activation_tile_hw;
  FastDivmod activation_tile_w_divmod;
  
  int filter[2];
  int stride[2];
  int dilation[2];
  int inc_next[2];
  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  int activation_load_count;
  int activation_storage_elements;
  int activation_size;
  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Depthwise2dFpropDirectConvParams() { }

  CUTLASS_HOST_DEVICE
  Depthwise2dFpropDirectConvParams(
      Conv2dProblemSize const &problem_size,
      Layout const &layout,             ///< layout object
      MatrixCoord threadblock_shape,    ///< CTA threadblock Shape
      Layout::TensorCoord threadblock_output_shape,  ///< Output tile Shape per threadblock
      const int element_size_bits,      ///< bits of activation element
      const int thread_count,           ///< threads per threadblock
      const int thread_count_contiguous, ///< number of threads for continuous dimension
      const int element_per_load)       ///< element per each load
      : layout(layout) {
          
    filter[0] = problem_size.S;
    filter[1] = problem_size.R;
    
    stride[0] =  problem_size.stride_w;
    stride[1] =  problem_size.stride_h;

    dilation[0] = problem_size.dilation_w;
    dilation[1] = problem_size.dilation_h;

    // Compute activation_tile size per threadblock because stride and dilation are runtime params.
    activation_tile_h = (threadblock_output_shape.h() - 1) * problem_size.stride_h +
                        (problem_size.R - 1) * problem_size.dilation_h + 1;
    activation_tile_w = (threadblock_output_shape.w() - 1) * problem_size.stride_w +
                        (problem_size.S - 1) * problem_size.dilation_w + 1;
    activation_tile_hw = activation_tile_h * activation_tile_w;

    activation_tile_w_divmod = FastDivmod(activation_tile_w);

    /// Below two values could not be templatized because the stride and dilation are runtime params
    activation_load_count = (thread_count_contiguous * activation_tile_hw + (thread_count - 1)) / thread_count;
    activation_storage_elements = activation_load_count * element_per_load * thread_count;
    activation_size =  activation_storage_elements * element_size_bits / 8;

    // Fastdivmod for output P, Q
    int tiles_p =
        (problem_size.P + (threadblock_output_shape.h() - 1)) / (threadblock_output_shape.h());
    int tiles_q = (problem_size.Q + (threadblock_output_shape.w() - 1)) /
                  (threadblock_output_shape.w());

    pq_divmod = FastDivmod(tiles_p * tiles_q);
    q_divmod = FastDivmod(tiles_q);

    // next S
    inc_next[0] = problem_size.dilation_w;
    // next R
    inc_next[1] = (activation_tile_w * problem_size.dilation_h - (problem_size.S - 1) * problem_size.dilation_w);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Parameters structure used for DepthwiseFpropActivationDirect2dConvTileAccessIteratorFixedStrideDilation
template <>
struct Depthwise2dFpropDirectConvActivationIteratorFixedStrideDilationParams<layout::TensorNHWC> {
  using Layout = layout::TensorNHWC;

  Layout layout;

  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  int activation_size;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Depthwise2dFpropDirectConvActivationIteratorFixedStrideDilationParams() {}

  CUTLASS_HOST_DEVICE
  Depthwise2dFpropDirectConvActivationIteratorFixedStrideDilationParams(
      Conv2dProblemSize const &problem_size,
      Layout const &layout,                          ///< Layout object
      MatrixCoord threadblock_shape,                 ///< Threadblock Shape
      Layout::TensorCoord threadblock_output_shape,  ///< Output tile Shape per threadblock
      const int activation_size_                     ///< Activation size loaded by iterator
      )
      : layout(layout),
        activation_size(activation_size_) {
    // Fastdivmod for output P, Q
    int tiles_p =
        (problem_size.P + (threadblock_output_shape.h() - 1)) / (threadblock_output_shape.h());
    int tiles_q =
        (problem_size.Q + (threadblock_output_shape.w() - 1)) / (threadblock_output_shape.w());

    pq_divmod = FastDivmod(tiles_p * tiles_q);
    q_divmod = FastDivmod(tiles_q);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure used for DepthwiseFpropFilterDirectConvTileAccessIteratorOptimized
template <>
struct Depthwise2dFpropDirectConvFilterIteratorParams<layout::TensorNHWC> {
  using Layout = layout::TensorNHWC;

  Layout layout;

  int filter_size;

  bool is_convolution;
  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Depthwise2dFpropDirectConvFilterIteratorParams() {}

  CUTLASS_HOST_DEVICE
  Depthwise2dFpropDirectConvFilterIteratorParams(
      Conv2dProblemSize const &problem_size,
      Layout const &layout,           ///< Layout object
      MatrixCoord threadblock_shape,  ///< Threadblock Shape
      const int filter_size_)         ///< Filter size loaded by iterator
      : layout(layout),
        filter_size(filter_size_),
        is_convolution(problem_size.mode == Mode::kConvolution){}
};

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
