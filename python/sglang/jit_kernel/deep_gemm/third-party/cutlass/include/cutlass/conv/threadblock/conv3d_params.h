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
#include "cutlass/conv/threadblock/conv2d_params.h"
#include "cutlass/conv/conv3d_problem_size.h"

#if TRACE_CONV_PARAMS_INITIALIZERS_ENABLED
#include <fstream>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Params structure used for all Conv3d analytic tile iterators
template< typename Layout_ = layout::TensorNDHWC >
struct Conv3dAnalyticParams {

  using Layout = Layout_;

  Layout layout;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv3dAnalyticParams() { }

  CUTLASS_HOST_DEVICE
  Conv3dAnalyticParams(
    Conv3dProblemSize const &,  // unused; placeholder to match other Params interfaces.
    Layout const &layout
  ): layout(layout) {

  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure used for Conv3dFpropActivationTileIteratorOptimized
template< typename Layout_ = layout::TensorNDHWC >
struct Conv3dFpropActivationIteratorOptimizedParams;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure used for Conv3dFpropActivationTileIteratorOptimized
template<>
struct Conv3dFpropActivationIteratorOptimizedParams<layout::TensorNDHWC> {
  
  using Layout = layout::TensorNDHWC;

  Layout layout;

  int64_t inc_next[4];    // {next S, next R, next T, next C}
  int filter_c_delta;     // number of logical elements to add to filter_c_
  int ZPQ;                // product of Z*P*Q
  int PQ;                 // product of P*Q

  FastDivmod zpq_divmod;
  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv3dFpropActivationIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv3dFpropActivationIteratorOptimizedParams(
    Conv3dProblemSize const &problem_size,
    Layout const &layout,                             ///< layout object
    int element_size_bits,                            ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): 
    layout(layout), 
    PQ(problem_size.P * problem_size.Q),
    ZPQ(problem_size.Z * problem_size.P * problem_size.Q),  
    zpq_divmod(ZPQ),
    pq_divmod(PQ), 
    q_divmod(problem_size.Q) {

    TRACE_CONV_INITIALIZERS("conv3d_fprop", "activation", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);
  

    int conv_sign = (problem_size.mode == Mode::kConvolution ? -1 : 1);

    // next S
    inc_next[0] = conv_sign * (
      int64_t(layout.stride()[0]) * problem_size.dilation_w
    ) * element_size_bits / 8;

    // next R
    inc_next[1] = conv_sign * (
        int64_t(layout.stride()[1]) * problem_size.dilation_h
        - (problem_size.S - 1) * layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // next T
    inc_next[2] = conv_sign * (
      int64_t(layout.stride()[2]) * problem_size.dilation_d
      - (problem_size.R - 1) * layout.stride()[1] * problem_size.dilation_h
      - (problem_size.S - 1) * layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // next C
    inc_next[3] = (
        threadblock_shape.column() * problem_size.split_k_slices
        - conv_sign * int64_t(problem_size.T - 1) * layout.stride()[2] * problem_size.dilation_d
        - conv_sign * int64_t(problem_size.R - 1) * layout.stride()[1] * problem_size.dilation_h
        - conv_sign * int64_t(problem_size.S - 1) * layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_c_delta = threadblock_shape.column() * problem_size.split_k_slices;
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////


template< typename Layout_ = layout::TensorNDHWC >
struct Conv3dFpropFilterIteratorOptimizedParams;

/////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Conv3dFpropFilterIteratorOptimizedParams<layout::TensorNDHWC>
{

  using Layout = layout::TensorNDHWC;

  Layout layout;
  int TRS;
  int filter_c_delta;

  int64_t inc_next_k;         // offset in units of bytes to next K position
  int64_t inc_next_trs;        // offset in units of bytes to next TRS position
  int64_t inc_next_c;         // offset in units of bytes to next C position

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv3dFpropFilterIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv3dFpropFilterIteratorOptimizedParams(
    Conv3dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): 
    layout(layout) {
    
    TRACE_CONV_INITIALIZERS("conv3d_fprop", "filter", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    TRS = problem_size.T * problem_size.R * problem_size.S;

    inc_next_k = (int64_t(layout.stride()[3]) * threadmap_delta.strided() * element_size_bits) / 8;

    inc_next_trs =
      ( int64_t(layout.stride()[0])
        - int64_t(layout.stride()[3]) * (threadmap_iterations.strided() - 1) * threadmap_delta.strided()
      ) * element_size_bits / 8;

    inc_next_c =
      (
        threadblock_shape.row() * problem_size.split_k_slices
        - int64_t(TRS - 1) * layout.stride()[0]
        - int64_t(threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[3]
      ) * element_size_bits / 8;

    filter_c_delta = threadblock_shape.row() * problem_size.split_k_slices;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters object for Conv3d DGRAD OutputGradient (dy) iterator
struct Conv3dDgradOutputGradientIteratorOptimizedParams {

  using Layout = layout::TensorNDHWC;

  Layout layout;

  int64_t inc_next[4];    // {next S, next R, next T, next K}
  int filter_k_delta;     // number of logical elements to add to filter_k_

  FastDivmod dhw_divmod;
  FastDivmod hw_divmod;
  FastDivmod w_divmod;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv3dDgradOutputGradientIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv3dDgradOutputGradientIteratorOptimizedParams(
    Conv3dProblemSize const &problem_size,
    Layout const &layout,                             ///< layout object
    int element_size_bits,                            ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): 
    layout(layout), 
    dhw_divmod(problem_size.D * problem_size.H * problem_size.W),
    hw_divmod(problem_size.H * problem_size.W), 
    w_divmod(problem_size.W) {

    TRACE_CONV_INITIALIZERS("conv3d_dgrad", "output_gradient", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    int conv_sign = (problem_size.mode == Mode::kConvolution ? 1 : -1);

    // next S
    inc_next[0] = conv_sign * (
      int64_t(layout.stride()[0]) * problem_size.dilation_w
    ) * element_size_bits / 8;

    // next R
    inc_next[1] = conv_sign * (
        int64_t(layout.stride()[1]) * problem_size.dilation_h
        - (problem_size.S - 1) * layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // next T
    inc_next[2] = conv_sign * (
      int64_t(layout.stride()[2]) * problem_size.dilation_d
      - (problem_size.R - 1) * layout.stride()[1] * problem_size.dilation_h
      - (problem_size.S - 1) * layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // next K
    inc_next[3] = (
        threadblock_shape.column() * problem_size.split_k_slices
        - conv_sign * int64_t(problem_size.T - 1) * layout.stride()[2] * problem_size.dilation_d
        - conv_sign * int64_t(problem_size.R - 1) * layout.stride()[1] * problem_size.dilation_h
        - conv_sign * int64_t(problem_size.S - 1) * layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_k_delta = threadblock_shape.column() * problem_size.split_k_slices;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters object for Conv2d DGRAD Filter (w) iterator
struct Conv3dDgradFilterIteratorOptimizedParams {

  using Layout = layout::TensorNDHWC;

  Layout layout;
  int TRS;
  int filter_k_delta;

  int64_t inc_next_strided;   // offset in units of bytes to next K coordinate within tile
  int64_t inc_next_trs;       // offset in units of bytes to next TRS position
  int64_t inc_next_k;         // offset in units of bytes to next K position in subsequent tile

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv3dDgradFilterIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv3dDgradFilterIteratorOptimizedParams(
    Conv3dProblemSize const &problem_size,
    Layout const &layout,    
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size, 
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): 
    layout(layout), TRS(problem_size.T * problem_size.R * problem_size.S) {

    TRACE_CONV_INITIALIZERS("conv3d_dgrad", "filter", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    inc_next_strided = ((int64_t)layout.stride()[3] * threadmap_delta.strided() * element_size_bits) / 8;

    inc_next_trs =
      ( (int64_t)layout.stride()[0]
        - (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * (int64_t)layout.stride()[3]
      ) * element_size_bits / 8;

    inc_next_k =
      (
        threadblock_shape.row() * problem_size.split_k_slices * (int64_t)layout.stride()[3]
        - (problem_size.T * problem_size.R * problem_size.S - 1) * (int64_t)layout.stride()[0]
        - (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * (int64_t)layout.stride()[3]
      ) * element_size_bits / 8;

    filter_k_delta = threadblock_shape.row() * problem_size.split_k_slices;
  }
};

/// Parameters object for Conv3d WGRAD OutputGradient iterator
struct Conv3dWgradOutputGradientIteratorOptimizedParams {

  using Layout = layout::TensorNDHWC;
  using LongIndex = typename Layout::LongIndex;

  Layout layout;

  int NZPQ;                // precomputd product of N*Z*P*Q for clearing predicates
  int ZPQ;                 // product of Z*P*Q
  unsigned zpq_mul;        // precomputed quantities for fast computation of div/% by ZPQ
  unsigned zpq_shr;        //    in device code.

  int PQ;                  // product of P*Q
  unsigned pq_mul;         // precomputed quantities for fast computation of div/% by PQ
  unsigned pq_shr;         //    in device code.

  unsigned q_mul;          // precomputed quantities for fast computation of div/% by Q
  unsigned q_shr;          //    in device code.

  LongIndex offset_next_strided;     // offset in units of bytes to next nzpq coordinate within tile
  LongIndex offset_next_contiguous;  // offset in units of bytes to next k coordinate within tile
  LongIndex inc_next_nzpq;           // offset in units of bytes to next nzpq position in subsequent tile

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv3dWgradOutputGradientIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv3dWgradOutputGradientIteratorOptimizedParams(
    Conv3dProblemSize const &problem_size,
    Layout const &layout,    
    int element_size_bits,
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size, 
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): layout(layout) {

  TRACE_CONV_INITIALIZERS("conv3d_wgrad", "output_gradient", 
    element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

  // Incremental offsets in unites of bytes (number of elements) * element_size_bits / 8
  offset_next_strided = (threadmap_delta.strided() * (int64_t)layout.stride()[0])
                      * element_size_bits / 8;

  offset_next_contiguous = (threadmap_delta.contiguous()) 
                          * element_size_bits / 8;

  inc_next_nzpq = (threadblock_shape.column() * problem_size.split_k_slices * (int64_t)layout.stride()[0])
                    * element_size_bits / 8;

  // Precompute several quantities for fast modulo arithmetic.
  NZPQ = problem_size.N * problem_size.Z * problem_size.P * problem_size.Q;
  ZPQ = problem_size.Z * problem_size.P * problem_size.Q;
  find_divisor(zpq_mul, zpq_shr, ZPQ);

  PQ = problem_size.P * problem_size.Q;
  find_divisor(pq_mul, pq_shr, PQ);

  find_divisor(q_mul, q_shr, problem_size.Q);

  }
};

/// Parameters object for Conv3d WGRAD Activation Tile Access Iterator
struct Conv3dWgradActivationIteratorOptimizedParams {

  using Layout = layout::TensorNDHWC;

  Layout layout;

  int RSC;                 // product of R*S*C
  unsigned rsc_mul;        // precomputed quantities for fast computation of div/% by RSC
  unsigned rsc_shr;        //    in device code.

  int SC;                  // product of S*C
  unsigned sc_mul;         // precomputed quantities for fast computation of div/% by SC
  unsigned sc_shr;         //    in device code.

  unsigned c_mul;          // precomputed quantities for fast computation of div/% by C
  unsigned c_shr;          //    in device code.

  int ZPQ;                 // product of Z*P*Q
  unsigned zpq_mul;        // precomputed quantities for fast computation of div/% by ZPQ
  unsigned zpq_shr;        //    in device code.

  int PQ;                  // product of P*Q
  unsigned pq_mul;         // precomputed quantities for fast computation of div/% by PQ
  unsigned pq_shr;         //    in device code.

  unsigned q_mul;          // precomputed quantities for fast computation of div/% by Q
  unsigned q_shr;          //    in device code.

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv3dWgradActivationIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv3dWgradActivationIteratorOptimizedParams(
    Conv3dProblemSize const &problem_size,
    Layout const &layout,    
    int element_size_bits,
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size, 
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): layout(layout) {

  TRACE_CONV_INITIALIZERS("conv3d_wgrad", "activation", 
    element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

  // Precompute several quantities for fast modulo arithmetic.
  RSC = problem_size.R * problem_size.S * problem_size.C;
  find_divisor(rsc_mul, rsc_shr, RSC);

  SC = problem_size.S * problem_size.C;
  find_divisor(sc_mul, sc_shr, SC);
      
  find_divisor(c_mul, c_shr, problem_size.C);

  ZPQ = problem_size.Z * problem_size.P * problem_size.Q;
  find_divisor(zpq_mul, zpq_shr, ZPQ);

  PQ = problem_size.P * problem_size.Q;
  find_divisor(pq_mul, pq_shr, PQ);

  find_divisor(q_mul, q_shr, problem_size.Q);

  }
};

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

