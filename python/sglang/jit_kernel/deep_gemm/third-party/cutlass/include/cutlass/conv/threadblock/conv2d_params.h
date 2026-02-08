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

/// Params structure used for all Conv2d analytic tile iterators
template< typename Layout_ = layout::TensorNHWC >
struct Conv2dAnalyticParams {

  using Layout = Layout_;

  Layout layout;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dAnalyticParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dAnalyticParams(
    Conv2dProblemSize const &,  // unused; placeholder to match other Params interfaces.
    Layout const &layout
  ): layout(layout) {

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Params structure used for all Conv2d analytic tile iterators
template< typename Layout_ = layout::TensorNHWC >
struct Conv2dFewChannelsParams {

  using Layout = Layout_;


  int32_t stride_w;
  int32_t stride_h;
  int32_t stride_n;

  FastDivmod divmod_P;
  FastDivmod divmod_Q;
  FastDivmod divmod_S;
  FastDivmod divmod_C;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dFewChannelsParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dFewChannelsParams(
    Conv2dProblemSize const &problem_size,  // unused; placeholder to match other Params interfaces.
    Layout const &layout
  ):
    stride_w(int32_t(layout.stride()[0])),
    stride_h(int32_t(layout.stride()[1])),
    stride_n(int32_t(layout.stride()[2])),
    divmod_P(problem_size.P),
    divmod_Q(problem_size.Q),
    divmod_S(problem_size.S),
    divmod_C(problem_size.C)
  {

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure used for Conv2dDgradOutputGradientTileAccessIteratorAnalyticParams
struct Conv2dDgradOutputGradientTileAccessIteratorAnalyticParams {
  
  using Layout = layout::TensorNHWC;

  Layout layout;
  int tiled_rows_per_filter;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientTileAccessIteratorAnalyticParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientTileAccessIteratorAnalyticParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,                            ///< layout object
    int element_size_bits,                           ///< size of each element in bits
    MatrixCoord threadblock_shape
  ): layout(layout) {
    
    int tile_m_per_filter = strided_dgrad_tile_m_per_filter(problem_size, threadblock_shape.row());
  
    tiled_rows_per_filter = tile_m_per_filter * threadblock_shape.row();
    
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

#if TRACE_CONV_PARAMS_INITIALIZERS_ENABLED

CUTLASS_HOST_DEVICE
void TraceIteratorParams(
  char const *conv_operator, 
  char const *operand,
  int element_size_bits,
  MatrixCoord threadblock_shape,
  int thread_count,
  int access_size,
  layout::PitchLinearCoord threadmap_iterations,
  layout::PitchLinearCoord threadmap_delta
) {
 
#if !defined(__CUDA_ARCH__)

  char const *fname = "conv_iterator_params.csv";

  std::ifstream test(fname);
  bool file_exists = test.is_open();

  if (file_exists) {
    test.close();
  }
 
  std::ofstream trace("conv_iterator_params.csv", std::ofstream::app);

  if (!file_exists) {
    trace 
      << "Operator,Operand,ElementSize,CtaRows,CtaColumns,ThreadCount,AccessSize,"
      << "IterationsContiguous,IterationsStrided,DeltaContiguous,DeltaStrided\n";
  }

  trace << conv_operator << "," << operand << "," << element_size_bits << "," 
    << threadblock_shape.row() << "," << threadblock_shape.column()
    << "," << thread_count << "," << access_size 
    << "," << threadmap_iterations.contiguous() << "," << threadmap_iterations.strided()
    << "," << threadmap_delta.contiguous() << "," << threadmap_delta.strided() << "\n";
#endif
}

#define TRACE_CONV_INITIALIZERS(conv_op, operand, element_size, cta_shape, thread_count, access_size, iterations, delta) \
  TraceIteratorParams(conv_op, operand, element_size, cta_shape, thread_count, access_size, iterations, delta);

#else

#define TRACE_CONV_INITIALIZERS(conv_op, operand, element_size, cta_shape, thread_count, access_size, iterations, delta) {}

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure used for Conv2dFpropActivationTileIteratorOptimized
template< typename Layout_ = layout::TensorNHWC >
struct Conv2dFpropActivationIteratorOptimizedParams;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters structure used for Conv2dFpropActivationTileIteratorOptimized
template<>
struct Conv2dFpropActivationIteratorOptimizedParams<layout::TensorNHWC> {
  
  using Layout = layout::TensorNHWC;

  Layout layout;

  int64_t inc_next[3];    // {next S, next R, next C}
  int filter_c_delta;     // number of logical elements to add to filter_c_
  int PQ;                 // product of P*Q

  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
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
    pq_divmod(PQ), 
    q_divmod(problem_size.Q) {

    TRACE_CONV_INITIALIZERS("conv2d_fprop", "activation", 
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

    // next C
    inc_next[2] = (
        threadblock_shape.column() * problem_size.split_k_slices
        - conv_sign * int64_t(problem_size.R - 1) * layout.stride()[1] * problem_size.dilation_h
        - conv_sign * int64_t(problem_size.S - 1) * layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_c_delta = threadblock_shape.column() * problem_size.split_k_slices;
  }

#if ENABLE_CONV2D_PARAMS_PRINT
  /// Prints internal state.
  CUTLASS_HOST_DEVICE
  void print() {
    auto stride = layout.stride();
    printf(
      "Conv2dFpropActivationIteratorOptimizedParams:\n"
      "  layout(w: %d, h: %d, n: %d)\n"
      "  inc_next[%ld, %ld, %ld]\n"
      "  filter_c_delta(%d) - PQ(%d)\n"
      "  pq_divmod(divisor: %d, multiplier: %u, shift_right: %u)\n"
      "  q_divmod(divisor: %d, multiplier: %u, shift_right: %u)\n",
      stride[0], stride[1], stride[2],
      inc_next[0], inc_next[1], inc_next[2],
      filter_c_delta,
      PQ,
      pq_divmod.divisor,
      pq_divmod.multiplier,
      pq_divmod.shift_right,
      q_divmod.divisor,
      q_divmod.multiplier,
      q_divmod.shift_right
    );
  }
#endif  
};

/// Parameters structure used for Conv2dFpropActivationTileIteratorOptimized
template <int Interleaved_>
struct Conv2dFpropActivationIteratorOptimizedParams<layout::TensorNCxHWx<Interleaved_>> {
  static int const kInterleaved = Interleaved_;
 
  using Layout = layout::TensorNCxHWx<kInterleaved>;

  Layout layout;

  int64_t inc_next[3];    // {next S, next R, next C}
  int filter_c_delta;     // number of logical elements to add to filter_c_
  int PQ;                 // product of P*Q

  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dFpropActivationIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,                             ///< layout object
    int element_size_bits,                            ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): 
    layout(layout), PQ(problem_size.P * problem_size.Q), pq_divmod(PQ), q_divmod(problem_size.Q) {

    TRACE_CONV_INITIALIZERS("conv2d_fprop", "activation", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    int conv_sign = (problem_size.mode == Mode::kConvolution ? -1 : 1);

    // next S
    inc_next[0] = conv_sign * (kInterleaved * problem_size.dilation_w) * element_size_bits / 8;

    // next R
    inc_next[1] = conv_sign * (
        int64_t(layout.stride()[0]) * problem_size.dilation_h
        - (problem_size.S - 1) * kInterleaved * problem_size.dilation_w
      ) * element_size_bits / 8;

    // next C
    inc_next[2] = (
        threadblock_shape.column() * problem_size.split_k_slices / kInterleaved * int64_t(layout.stride()[1])
        - conv_sign * int64_t(problem_size.R - 1) * layout.stride()[0] * problem_size.dilation_h
        - conv_sign * int64_t(problem_size.S - 1) * kInterleaved * problem_size.dilation_w
      ) * element_size_bits / 8;

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_c_delta = threadblock_shape.column() * problem_size.split_k_slices;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout_ = layout::TensorNHWC >
struct Conv2dFpropFilterIteratorOptimizedParams;

/////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Conv2dFpropFilterIteratorOptimizedParams<layout::TensorNHWC>
{

  using Layout = layout::TensorNHWC;

  Layout layout;
  int RS;
  int filter_c_delta;

  int64_t inc_next_k;         // offset in units of bytes to next K position
  int64_t inc_next_rs;        // offset in units of bytes to next RS position
  int64_t inc_next_c;         // offset in units of bytes to next C position

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): 
    layout(layout) {
    
    TRACE_CONV_INITIALIZERS("conv2d_fprop", "filter", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    RS = problem_size.R * problem_size.S;

    inc_next_k = (int64_t(layout.stride()[2]) * threadmap_delta.strided() * element_size_bits) / 8;

    inc_next_rs =
      ( int64_t(layout.stride()[0])
        - int64_t(layout.stride()[2]) * (threadmap_iterations.strided() - 1) * threadmap_delta.strided()
      ) * element_size_bits / 8;

    inc_next_c =
      (
        threadblock_shape.row() * problem_size.split_k_slices
        - int64_t(RS - 1) * layout.stride()[0]
        - int64_t(threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[2]
      ) * element_size_bits / 8;

    filter_c_delta = threadblock_shape.row() * problem_size.split_k_slices;
  }

#if ENABLE_CONV2D_PARAMS_PRINT
  /// Prints internal state.
  CUTLASS_HOST_DEVICE
  void print() {
    auto stride = layout.stride();
    printf(
      "Conv2dFpropFilterIteratorOptimizedParams:\n"
      "  layout[%d, %d, %d]\n"
      "  RS(%d), filter_c_delta(%d), inc_next(k: %ld, rs: %ld, c: %ld)\n",
      stride[0], stride[1], stride[2],
      RS,
      filter_c_delta,
      inc_next_k, inc_next_rs, inc_next_c
    );
  }
#endif
};

template<int Interleaved_>
struct Conv2dFpropFilterIteratorOptimizedParams<layout::TensorCxRSKx<Interleaved_>>
{
  static int const kInterleaved = Interleaved_;
  using Layout = layout::TensorCxRSKx<kInterleaved>;

  Layout layout;
  int RS;
  int filter_c_delta;

  int64_t inc_next_k;         // offset in units of bytes to next K position
  int64_t inc_next_rs;        // offset in units of bytes to next RS position
  int64_t inc_next_c;         // offset in units of bytes to next C position

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dFpropFilterIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): 
    layout(layout) {
    
    TRACE_CONV_INITIALIZERS("conv2d_fprop", "filter", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    RS = problem_size.R * problem_size.S;

    inc_next_k = (kInterleaved * threadmap_delta.strided() * element_size_bits) / 8;

    inc_next_rs =
      (  int64_t(layout.stride()[0])
        - kInterleaved * (threadmap_iterations.strided() - 1) * threadmap_delta.strided()
      ) * element_size_bits / 8;

    inc_next_c =
      (
        threadblock_shape.row() * problem_size.split_k_slices / kInterleaved * int64_t(layout.stride()[2])
        - int64_t(RS - 1) * layout.stride()[0]
        - int64_t(threadmap_iterations.strided() - 1) * threadmap_delta.strided() * kInterleaved 
      ) * element_size_bits / 8;

    filter_c_delta = threadblock_shape.row() * problem_size.split_k_slices;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Dgrad Optimized Dy params (layout::TensorNHWC)
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Parameters object for Conv2d DGRAD OutputGradient (dy) iterator
struct Conv2dDgradOutputGradientIteratorOptimizedParams {

  using Layout = layout::TensorNHWC;

  Layout layout;

  int64_t inc_next[3];    // {next S, next R, next K}

  int filter_k_delta;     // number of logical elements to add to filter_k_

  int HW;                  // product of H*W

  FastDivmod hw_divmod;
  FastDivmod w_divmod;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dDgradOutputGradientIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): 
    layout(layout), 
    HW(problem_size.H *problem_size.W), 
    hw_divmod(HW), 
    w_divmod(problem_size.W) {
    
    TRACE_CONV_INITIALIZERS("conv2d_dgrad", "output_gradient", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    int conv_sign = (problem_size.mode == Mode::kConvolution ? 1 : -1);

    // next S
    inc_next[0] = conv_sign * (
      (int64_t)layout.stride()[0] * problem_size.dilation_w
    ) * element_size_bits / 8;

    // next R
    inc_next[1] = conv_sign * (
        (int64_t)layout.stride()[1] * problem_size.dilation_h
        - (problem_size.S - 1) * (int64_t)layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // next K
    inc_next[2] = (
        threadblock_shape.column() * problem_size.split_k_slices
        - conv_sign * (problem_size.R - 1) * (int64_t)layout.stride()[1] * problem_size.dilation_h
        - conv_sign * (problem_size.S - 1) * (int64_t)layout.stride()[0] * problem_size.dilation_w
      ) * element_size_bits / 8;

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_k_delta = threadblock_shape.column() * problem_size.split_k_slices;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Strided Dgrad Optimized Dy params (layout::TensorNHWC)
/////////////////////////////////////////////////////////////////////////////////////////////////
struct Conv2dStridedDgradOutputGradientIteratorOptimizedParams {
  
  using Layout = layout::TensorNHWC;

  Layout layout;
  
  int64_t inc_next[3];    // {next S, next R, next K}

  int filter_k_delta;     // number of logical elements to add to filter_k_

  int tiled_rows_per_filter;

  int conv_sign;
  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dStridedDgradOutputGradientIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dStridedDgradOutputGradientIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,                            ///< layout object
    int element_size_bits,                           ///< size of each element in bits
    MatrixCoord threadblock_shape
  ): layout(layout) {
    
    int tile_m_per_filter = strided_dgrad_tile_m_per_filter(problem_size, threadblock_shape.row());
  
    tiled_rows_per_filter = tile_m_per_filter * threadblock_shape.row();

    conv_sign = (problem_size.mode == Mode::kConvolution ? 1 : -1);

    // next S
    inc_next[0] = conv_sign * (
      (int64_t)layout.stride()[0] * problem_size.dilation_w
    ) * element_size_bits / 8;

    // next R
    inc_next[1] = conv_sign * (
        (int64_t)layout.stride()[1] * problem_size.dilation_h
      ) * element_size_bits / 8;

    // next K
    inc_next[2] = (
        threadblock_shape.column() * problem_size.split_k_slices
      ) * element_size_bits / 8;

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_k_delta = threadblock_shape.column() * problem_size.split_k_slices;
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
// Dgrad Optimized w params (layout::TensorNHWC)
/////////////////////////////////////////////////////////////////////////////////////////////////
struct Conv2dDgradFilterIteratorOptimizedParams {

  using Layout = layout::TensorNHWC;

  Layout layout;
  int RS;
  int filter_k_delta;

  int64_t inc_next_strided;   // offset in units of bytes to next K coordinate within tile
  int64_t inc_next_rs;        // offset in units of bytes to next RS position
  int64_t inc_next_k;         // offset in units of bytes to next K position in subsequent tile

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dDgradFilterIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,    
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size, 
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): 
    layout(layout), RS(problem_size.R * problem_size.S) {

    TRACE_CONV_INITIALIZERS("conv2d_dgrad", "filter", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    inc_next_strided = ((int64_t)layout.stride()[2] * threadmap_delta.strided() * element_size_bits) / 8;

    inc_next_rs =
      ( (int64_t)layout.stride()[0]
        - (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * (int64_t)layout.stride()[2]
      ) * element_size_bits / 8;

    inc_next_k =
      (
        threadblock_shape.row() * problem_size.split_k_slices * (int64_t)layout.stride()[2]
        - (problem_size.R * problem_size.S - 1) * (int64_t)layout.stride()[0]
        - (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * (int64_t)layout.stride()[2]
      ) * element_size_bits / 8;

    filter_k_delta = threadblock_shape.row() * problem_size.split_k_slices;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
// StridedDgrad Optimized w params (layout::TensorNHWC)
/////////////////////////////////////////////////////////////////////////////////////////////////
struct Conv2dStridedDgradFilterIteratorOptimizedParams {

  using Layout = layout::TensorNHWC;

  Layout layout;
  int RS;
  int filter_k_delta;

  int64_t inc_next_strided;   // offset in units of bytes to next K coordinate within tile
  int64_t inc_next[3];        // {next S, next R, next K}
  int64_t reset_bytes;        // offset in units of bytes to move back the pointer 
  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv2dStridedDgradFilterIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dStridedDgradFilterIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,    
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size, 
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ): 
    layout(layout), RS(problem_size.R * problem_size.S) {

    TRACE_CONV_INITIALIZERS("conv2d_dgrad", "filter", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    inc_next_strided = (layout.stride()[2] * threadmap_delta.strided() * element_size_bits) / 8;

    // next S
    inc_next[0] =
      ( (int64_t)layout.stride()[0] * problem_size.stride_w
        //- (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[2]
      ) * element_size_bits / 8;

    // next R
    inc_next[1] =
      ( (int64_t)layout.stride()[1] * problem_size.stride_h
        //- (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[2]
      ) * element_size_bits / 8;

    // next K
    inc_next[2] =
      (
        threadblock_shape.row() * problem_size.split_k_slices * (int64_t)layout.stride()[2]
        //- (problem_size.R * problem_size.S - 1) * layout.stride()[0]
        //- (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * layout.stride()[2]
      ) * element_size_bits / 8;

    // offset in units of bytes to move the pointer in backward direction
    reset_bytes = (threadmap_iterations.strided() - 1) * threadmap_delta.strided() * (int64_t)layout.stride()[2]
            * element_size_bits / 8;

    filter_k_delta = threadblock_shape.row() * problem_size.split_k_slices;
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameters object for Conv2d WGRAD Output Gradient (dy) iterator
struct Conv2dWgradOutputGradientIteratorOptimizedParams {

  using Layout = layout::TensorNHWC;

  Layout layout;

  int NPQ;                      // precomputd product of N*P*Q for clearing predicates

  FastDivmod pq_divmod;
  FastDivmod q_divmod;

  int64_t offset_next_strided;    // offset in units of bytes to next npq coordinate within tile
  int64_t offset_next_contiguous; // offset in units of bytes to next k coordinate within tile
  int64_t inc_next_npq;           // offset in units of bytes to next npq position in subsequent tile

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv2dWgradOutputGradientIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dWgradOutputGradientIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,    
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    layout(layout),
    NPQ(problem_size.N * problem_size.P * problem_size.Q),
    pq_divmod(problem_size.P * problem_size.Q),
    q_divmod(problem_size.Q) {
    
    TRACE_CONV_INITIALIZERS("conv2d_wgrad", "output_gradient", 
      element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);

    // Incremental offsets in unites of bytes (number of elements) * sizeof_bits<Element>::value / 8
    offset_next_strided = (threadmap_delta.strided() * (int64_t)layout.stride()[0])
                        * element_size_bits / 8;

    offset_next_contiguous = (threadmap_delta.contiguous())
                            * element_size_bits / 8;

    inc_next_npq = (threadblock_shape.column() * problem_size.split_k_slices * (int64_t)layout.stride()[0])
                      * element_size_bits / 8;
  }
};

struct Conv2dWgradActivationIteratorOptimizedParams {

  using Layout = layout::TensorNHWC;

  Layout layout;

  FastDivmod sc_divmod;
  FastDivmod pq_divmod;
  FastDivmod q_divmod;
  FastDivmod c_divmod;
  FastDivmod s_divmod;
  int small_channel_conv_s_offset;

  //
  // Methods
  //
  CUTLASS_HOST_DEVICE
  Conv2dWgradActivationIteratorOptimizedParams() { }

  CUTLASS_HOST_DEVICE
  Conv2dWgradActivationIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout
  ):
    layout(layout),
    sc_divmod(problem_size.S * problem_size.C),
    pq_divmod(problem_size.P * problem_size.Q),
    q_divmod(problem_size.Q),
    c_divmod(problem_size.C),
    s_divmod(problem_size.S * problem_size.dilation_w),
    small_channel_conv_s_offset((problem_size.S - 1) * problem_size.dilation_w - problem_size.pad_w) {
  }

  CUTLASS_HOST_DEVICE
  Conv2dWgradActivationIteratorOptimizedParams(
    Conv2dProblemSize const &problem_size,
    Layout const &layout,
    int element_size_bits,                        ///< size of each element in bits
    MatrixCoord threadblock_shape,
    int thread_count,
    int access_size,
    layout::PitchLinearCoord threadmap_iterations,
    layout::PitchLinearCoord threadmap_delta
  ):
    Conv2dWgradActivationIteratorOptimizedParams(
      problem_size,
      layout
    ) { 
    
      TRACE_CONV_INITIALIZERS("conv2d_wgrad", "activation", 
        element_size_bits, threadblock_shape, thread_count, access_size, threadmap_iterations, threadmap_delta);
    }
};

struct PredicatedScaleBiasVectorAccessIteratorParams {
  public:
    /// Default ctor
    CUTLASS_HOST_DEVICE
    PredicatedScaleBiasVectorAccessIteratorParams() { }

    // Default ctor
    CUTLASS_HOST_DEVICE
    PredicatedScaleBiasVectorAccessIteratorParams(
      Conv2dProblemSize const &problem_size,
      layout::PitchLinear const &layout) {}

    // Default ctor
    CUTLASS_HOST_DEVICE
    PredicatedScaleBiasVectorAccessIteratorParams(
      Conv2dProblemSize const &problem_size,
      layout::RowMajor const &layout) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
