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
    \brief This file contains definitions and utility functions for describing convolution problem sizes.

  Conv2dProblem desciption:
    activation (NHWC), 
    filter (KRSC), 
    output (NPQK), 
    pading (pad_h, pad_w),
    stride (stride_h, stride_w),
    dilation (dilation_h, dilation_w).
    
  Free functions to map:
    Map tensor extents (Conv2d -> ImplicitGemm)      : implicit_gemm_tensor_[a|b|c]_extent(ConvolutionOperator)
    Map tensor sizes (Conv2d -> ImplicitGemm)        : implicit_gemm_tensor_[a|b|c]_size(ConvolutionOperator)
    Map tensor problem sizes (Conv2d -> ImplicitGemm): implicit_gemm_problem_size(ConvolutionOperator)
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/functional.h"

namespace cutlass {
namespace conv {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Problem size structure
struct Conv2dProblemSize {

  // Conv2d strictly problem size parameters
  int N, H, W, C, P, Q, K, R, S;
  int pad_h, pad_w;
  int stride_h, stride_w;
  int dilation_h, dilation_w;
  Mode mode;

  // Conv2d implementation-related parameters 
  int split_k_slices;
  int groups;

  //
  // Methods
  //

public:
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize():
    N(0), H(0), W(0), C(0), P(0), Q(0), K(0), R(0), S(0),
    pad_h(0), pad_w(0), stride_h(1), stride_w(1), dilation_h(1), dilation_w(1),
    mode(Mode::kConvolution), split_k_slices(1), groups(1) { }
 
  /// Constructor for default padding, stride, dilation, and split-K
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize(
    int N,
    int H,
    int W,
    int C,
    int P,
    int Q,
    int K,
    int R,
    int S,
    Mode mode
  ): 
    N(N), H(H), W(W), C(C), P(P), Q(Q), K(K), R(R), S(S),
    pad_h(R / 2), pad_w(S / 2), stride_h(1), stride_w(1), dilation_h(1), dilation_w(1),
    mode(mode), split_k_slices(1), groups (1) { }
  
  /// Constructor
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize(
    int N,
    int H,
    int W,
    int C,
    int K,
    int R,
    int S,
    int P,
    int Q,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    Mode mode,
    int split_k_slices = 1,
    int groups = 1
  ):
    N(N), H(H), W(W), C(C), P(P), Q(Q), K(K), R(R), S(S),
    pad_h(pad_h), pad_w(pad_w), stride_h(stride_h), stride_w(stride_w),
    dilation_h(dilation_h), dilation_w(dilation_w), 
    mode(mode), split_k_slices(split_k_slices), groups (groups) { }

  /// Constructs convolution problem size from cutlass Tensor4DCoord and MatrixCoord 
  // set user-defined output size and sets P and Q (include all data members in ctor)
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize(
    cutlass::Tensor4DCoord input_size,    // NHWC
    cutlass::Tensor4DCoord filter_size,   // KRSC
    cutlass::Tensor4DCoord padding,       // pad_h, _, pad_w, _
    cutlass::MatrixCoord stride,          // stride_h, stride_w
    cutlass::MatrixCoord dilation,        // dilation_h, dilation_w
    cutlass::Tensor4DCoord output_size,   // NPQK
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation,
    int split_k_slices = 1,
    int groups = 1
  ):
    N(input_size.n()), H(input_size.h()), W(input_size.w()), C(input_size.c()),
    P(output_size.h()), Q(output_size.w()),
    K(filter_size.n()), R(filter_size.h()), S(filter_size.w()),
    pad_h(padding[0]), pad_w(padding[2]),
    stride_h(stride.row()), stride_w(stride.column()),
    dilation_h(dilation.row()), dilation_w(dilation.column()),
    mode(mode), split_k_slices(split_k_slices), groups(groups) {}

  /// Constructs convolution problem size from cutlass Tensor4DCoord and MatrixCoord 
  // computes output size and sets P and Q (skip output from ctor arguments)
  CUTLASS_HOST_DEVICE  
  Conv2dProblemSize(
    cutlass::Tensor4DCoord input_size,   // NHWC
    cutlass::Tensor4DCoord filter_size,  // KRSC
    cutlass::Tensor4DCoord padding,      // pad_h, upper_pad_h, pad_w, upper_pad_w
    cutlass::MatrixCoord stride,         // stride_h, stride_w
    cutlass::MatrixCoord dilation,       // dilation_h, dilation_w
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation,
    int split_k_slices = 1,
    int groups = 1
  ):
    N(input_size.n()), H(input_size.h()), W(input_size.w()), C(input_size.c()),
    K(filter_size.n()), R(filter_size.h()), S(filter_size.w()),
    pad_h(padding[0]), pad_w(padding[2]),
    stride_h(stride.row()), stride_w(stride.column()),
    dilation_h(dilation.row()), dilation_w(dilation.column()),
    mode(mode), split_k_slices(split_k_slices), groups(groups) {
      // set output P and Q
      P = ((H + pad_h + padding[1] - R * dilation_h) / stride_h) + 1;
      Q = ((W + pad_w + padding[3] - S * dilation_w) / stride_w) + 1;
    }

  /// Constructs convolution problem size from cutlass Tensor4DCoord and MatrixCoord 
  // set user-defined output size and sets P and Q (skip padding, striding, and dilation)
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize(
    cutlass::Tensor4DCoord input_size,    // NHWC
    cutlass::Tensor4DCoord filter_size,   // KRSC
    cutlass::Tensor4DCoord output_size,   // NPQK
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation,
    int split_k_slices = 1,
    int groups = 1
  ):
    N(input_size.n()), H(input_size.h()), W(input_size.w()), C(input_size.c()),
    P(output_size.h()), Q(output_size.w()),
    K(filter_size.n()), R(filter_size.h()), S(filter_size.w()),
    pad_h(R / 2), pad_w(S / 2), stride_h(1), stride_w(1),
    dilation_h(1), dilation_w(1),
    mode(mode), split_k_slices(split_k_slices), groups(groups) {}

  // Reset covolution mode in the problem
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize reset_mode(cutlass::conv::Mode mode_) {
    Conv2dProblemSize tmp(*this);
    tmp.mode = mode_; 
    return tmp; 
  }

  // Reset covolution mode in the problem
  CUTLASS_HOST_DEVICE
  Conv2dProblemSize reset_split_k_slices(int split_k_slices_) {
    Conv2dProblemSize tmp(*this);
    tmp.split_k_slices = split_k_slices_; 
    return tmp; 
  }

  /// Equality operator (ignores mode and split_k_slice)
  CUTLASS_HOST_DEVICE
  bool operator==(Conv2dProblemSize const &conv) const {
    return (
      (N == conv.N) && (H == conv.H) && (W == conv.W) && (C == conv.C) &&
      (K == conv.K) && (R == conv.R) && (S == conv.S) &&
      (P == conv.P) && (Q == conv.Q) &&
      (pad_h == conv.pad_h) && (pad_w == conv.pad_w) &&
      (stride_h == conv.stride_h) && (stride_w == conv.stride_w) &&
      (dilation_h == conv.dilation_h) && (dilation_w == conv.dilation_w)
    );  
  }

  /// Inequality operator
  CUTLASS_HOST_DEVICE
  bool operator!=(Conv2dProblemSize const &rhs) const {
    return !(*this == rhs);
  }

  /// Returns activation extent as Tensor4DCoord
  CUTLASS_HOST_DEVICE
  cutlass::Tensor4DCoord activation_extent() const {

    return cutlass::Tensor4DCoord ({N, H, W, C});
  }

  /// Returns filter extent as Tensor4DCoord
  CUTLASS_HOST_DEVICE
  cutlass::Tensor4DCoord filter_extent(bool is_deconv = false) const {

    return is_deconv ? cutlass::Tensor4DCoord ({C, R, S, K / groups})
        : cutlass::Tensor4DCoord ({K, R, S, C / groups});
  }

  /// Returns output extent as Tensor4DCoord
  CUTLASS_HOST_DEVICE
  cutlass::Tensor4DCoord output_extent() const {

    return cutlass::Tensor4DCoord ({N, P, Q, K});
  }

  /// Returns activation size in number of elements
  CUTLASS_HOST_DEVICE
  int64_t activation_size() const {

    return static_cast<int64_t>(N) * static_cast<int64_t>(H) *
           static_cast<int64_t>(W) * static_cast<int64_t>(C);
  }

  /// Returns filter size in number of elements
  CUTLASS_HOST_DEVICE
  int64_t filter_size() const {

    return static_cast<int64_t>(K) * static_cast<int64_t>(R) *
           static_cast<int64_t>(S) * static_cast<int64_t>(C) /
           static_cast<int64_t>(groups);
  }

  /// Returns output size in number of elements
  CUTLASS_HOST_DEVICE
  int64_t output_size() const {

    return static_cast<int64_t>(N) * static_cast<int64_t>(P) *
           static_cast<int64_t>(Q) * static_cast<int64_t>(K);
  }

  /// Returns padding as Tensor4DCoord
  CUTLASS_HOST_DEVICE
  cutlass::Tensor4DCoord padding() const {

    return cutlass::Tensor4DCoord ({pad_h, pad_h, pad_w, pad_w});
  }

  /// Returns stride as MatrixCoord
  CUTLASS_HOST_DEVICE
  cutlass::MatrixCoord stride() const {

    return cutlass::MatrixCoord ({stride_h, stride_w});
  }

  /// Returns dilation as MatrixCoord
  CUTLASS_HOST_DEVICE
  cutlass::MatrixCoord dilation() const {

    return cutlass::MatrixCoord ({dilation_h, dilation_w});
  }

  /////////////////////////////////////////////////////////////////
  //        Methods used for strided dgrad implementation
  /////////////////////////////////////////////////////////////////
  /// Number of filter r positions to accumulate in gemm-k dim
  CUTLASS_HOST_DEVICE
  int num_gemm_k_filter_r(int r) const {
    return ((R - r + stride_h - 1) / stride_h);
  }

  /// Number of filter s positions to accumulate in gemm-k dim
  CUTLASS_HOST_DEVICE
  int num_gemm_k_filter_s(int s) const {
    return ((S - s + stride_w - 1) / stride_w);
  }

  /// Number of filter positions to accumulate in gemm-k dim
  CUTLASS_HOST_DEVICE
  int num_gemm_k_filter_positions(int r, int s) const {
    return num_gemm_k_filter_r(r) * num_gemm_k_filter_s(s);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  ImplicitGemm helper functions                                 //
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Determine the problem size of the implicit GEMM operation
CUTLASS_HOST_DEVICE
cutlass::gemm::GemmCoord implicit_gemm_problem_size(
  Operator conv_operator, 
  Conv2dProblemSize const &problem_size) {
  // Compute problem size
  switch (conv_operator) {
  case Operator::kFprop:
    return gemm::GemmCoord(
      problem_size.N * problem_size.P * problem_size.Q,
      problem_size.K,
      problem_size.R * problem_size.S * problem_size.C / problem_size.groups
    );
  case Operator::kDeconv:
  case Operator::kDgrad:
    return gemm::GemmCoord(
      problem_size.N * problem_size.H * problem_size.W,
      problem_size.C,
      problem_size.R * problem_size.S * problem_size.K
    );
  case Operator::kWgrad:
    return gemm::GemmCoord(
      problem_size.K,
      problem_size.R * problem_size.S * problem_size.C,
      problem_size.N * problem_size.P * problem_size.Q
    );
  default:
    break;
  }
  return gemm::GemmCoord();
}

// Determine the number of gemm_k iterations for conv2d problem using implicit gemm algorithm
CUTLASS_HOST_DEVICE
int implicit_gemm_k_iterations(
  Operator conv_operator, 
  int threadblock_K, 
  Conv2dProblemSize const &problem_size,
  IteratorAlgorithm algorithm = IteratorAlgorithm::kAnalytic,
  GroupMode group_mode = GroupMode::kNone,
  int threadblock_N = 0) {

  int iterations = 0;

  if (group_mode == GroupMode::kNone) {

    if (algorithm == IteratorAlgorithm::kFixedChannels) {

      int positions_per_iteration = threadblock_K / problem_size.C;
      switch (conv_operator) {
      case Operator::kFprop:
        iterations = (problem_size.R * problem_size.S + positions_per_iteration - 1 ) / positions_per_iteration;
        break;

      default:
        break;
      }
    }
    else if (algorithm == IteratorAlgorithm::kFewChannels) {

      switch (conv_operator) {
      case Operator::kFprop:
        iterations = (problem_size.R * problem_size.S * problem_size.C + threadblock_K - 1 ) / threadblock_K;
        break;

      default:
        break;
      }
    }
    else {
      int elements_per_split_k_slice = 0;

      switch (conv_operator) {
      case Operator::kFprop:
        elements_per_split_k_slice = (problem_size.C + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
        iterations = problem_size.R * problem_size.S * ((elements_per_split_k_slice + threadblock_K - 1) / threadblock_K);
        break;

      case Operator::kDeconv:
      case Operator::kDgrad:
        elements_per_split_k_slice = (problem_size.K + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
        iterations = problem_size.R * problem_size.S * ((elements_per_split_k_slice + threadblock_K - 1) / threadblock_K);
        break;

      case Operator::kWgrad:
        elements_per_split_k_slice = (problem_size.N * problem_size.P * problem_size.Q + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
        iterations = (elements_per_split_k_slice + threadblock_K - 1) / threadblock_K;
        break;

      default:
        break;
      }
    }

  } else if (group_mode == GroupMode::kDepthwise) {
    int channels_per_cta = threadblock_N;

    if (algorithm == IteratorAlgorithm::kAnalytic) {
      switch (conv_operator) {
        case Operator::kFprop:
          iterations = problem_size.R * problem_size.S *
                       ((channels_per_cta + threadblock_K - 1) / threadblock_K);
          break;

        default:
          break;
      }
    }
  } else {  // Group conv

    int channels_per_group = problem_size.C / problem_size.groups;
    int k_per_group = problem_size.K / problem_size.groups;

    if (algorithm == IteratorAlgorithm::kAnalytic) {
      switch (conv_operator) {
        case Operator::kFprop:
          iterations = problem_size.R * problem_size.S * ((channels_per_group + threadblock_K - 1) / threadblock_K);
          // In group conv, if k_per_group < threadblock_N, one Threadblock will calculate multiple groups
          if (problem_size.groups != 1) {
            if (k_per_group < threadblock_N) {
              iterations *= threadblock_N / k_per_group;
            }
          }
          break;

        default:
          break;
      }
    } else if (algorithm == IteratorAlgorithm::kOptimized) {
      // Current optimized iterator only support GroupMode::kSingleGroup
      if (group_mode == GroupMode::kSingleGroup) {
        switch (conv_operator) {
          case Operator::kFprop:
            iterations = problem_size.R * problem_size.S * ((channels_per_group + threadblock_K - 1) / threadblock_K);
            break;

          default:
            break;
        }
      }
    }

  }

  return iterations;
}


template <int N = 1, int Output_P = 1, int Output_Q = 1>
CUTLASS_HOST_DEVICE
int depthwise_gemm_k_iterations(
  Operator conv_operator, 
  int threadblock_K, 
  Conv2dProblemSize const &problem_size,
  IteratorAlgorithm algorithm = IteratorAlgorithm::kAnalytic,
  GroupMode group_mode = GroupMode::kNone,
  int threadblock_N = 0) {

    int n =  problem_size.N;
    int p = (problem_size.P + Output_P - 1) /  Output_P;
    int q = (problem_size.Q + Output_Q - 1) /  Output_Q;

    int iterations = (n * p * q + problem_size.split_k_slices - 1) / problem_size.split_k_slices;
    return iterations;
}


CUTLASS_HOST_DEVICE
int implicit_gemm_k_iterations_per_channel(
    Operator conv_operator,
    Conv2dProblemSize const &problem_size,
    IteratorAlgorithm algorithm = IteratorAlgorithm::kAnalytic) {

  int iterations = 0; //0 means not applicable
  if (algorithm == IteratorAlgorithm::kAnalytic || algorithm == IteratorAlgorithm::kOptimized) {
    switch (conv_operator) {
      case Operator::kFprop:
        iterations = problem_size.R * problem_size.S;
        break;

      case Operator::kDeconv:
      case Operator::kDgrad:
        iterations = problem_size.R * problem_size.S;
        break;

      default:
        break;
    }
  }
  return iterations;
}

////////////////////////////////////////////////////////////////////////////////
//  Mapping function (ImplicitGemm A, B, C -> Conv Activation, Filter, Output)
////////////////////////////////////////////////////////////////////////////////
/// Returns ImplicitGemm tensor A extent as Tensor4DCoord
CUTLASS_HOST_DEVICE
cutlass::Tensor4DCoord implicit_gemm_tensor_a_extent(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.activation_extent();
    case cutlass::conv::Operator::kDeconv:
    case cutlass::conv::Operator::kDgrad: return problem_size.output_extent();
    case cutlass::conv::Operator::kWgrad: return problem_size.output_extent();
    default : break;
  }
  return cutlass::Tensor4DCoord();
}

/// Returns ImplicitGemm tensor B extent as Tensor4DCoord
CUTLASS_HOST_DEVICE
cutlass::Tensor4DCoord implicit_gemm_tensor_b_extent(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.filter_extent();
    case cutlass::conv::Operator::kDeconv: return problem_size.filter_extent(true);
    case cutlass::conv::Operator::kDgrad: return problem_size.filter_extent();
    case cutlass::conv::Operator::kWgrad: return problem_size.activation_extent();
    default : break;
  }
  return cutlass::Tensor4DCoord();
}

/// Returns ImplicitGemm tensor C extent as Tensor4DCoord
CUTLASS_HOST_DEVICE
cutlass::Tensor4DCoord implicit_gemm_tensor_c_extent(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.output_extent();
    case cutlass::conv::Operator::kDeconv:
    case cutlass::conv::Operator::kDgrad: return problem_size.activation_extent();
    case cutlass::conv::Operator::kWgrad: return problem_size.filter_extent();
    default : break;
  }
  return cutlass::Tensor4DCoord();
}

/// Returns ImplicitGemm tensor A size in number of elements
CUTLASS_HOST_DEVICE
int64_t implicit_gemm_tensor_a_size(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.activation_size();
    case cutlass::conv::Operator::kDeconv:
    case cutlass::conv::Operator::kDgrad: return problem_size.output_size();
    case cutlass::conv::Operator::kWgrad: return problem_size.output_size();
    default : break;
  }
  return 0;
}

/// Returns ImplicitGemm tensor B size in number of elements
CUTLASS_HOST_DEVICE
int64_t implicit_gemm_tensor_b_size(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.filter_size();
    case cutlass::conv::Operator::kDeconv:
    case cutlass::conv::Operator::kDgrad: return problem_size.filter_size();
    case cutlass::conv::Operator::kWgrad: return problem_size.activation_size();
    default : break;
  }
  return 0;
}

/// Returns ImplicitGemm tensor C size in number of elements
CUTLASS_HOST_DEVICE
int64_t implicit_gemm_tensor_c_size(
  Operator conv_operator,
  Conv2dProblemSize const &problem_size) {
  switch (conv_operator) {
    case cutlass::conv::Operator::kFprop: return problem_size.output_size();
    case cutlass::conv::Operator::kDeconv:
    case cutlass::conv::Operator::kDgrad: return problem_size.activation_size();
    case cutlass::conv::Operator::kWgrad: return problem_size.filter_size();
    default : break;
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  Strided dgrad helper functions                                 //
////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns number of CTAs tile M to cover valid MMAs per starting filter postion
CUTLASS_HOST_DEVICE
int strided_dgrad_tile_m_per_filter(
  Conv2dProblemSize const &problem_size,
  int tile_size_m) {

  // Compute NHW rows in Dx output that needs MMA per starting filter position
  int rows_h_per_filter = (problem_size.H + problem_size.stride_h - 1) / problem_size.stride_h;
  int rows_w_per_filter = (problem_size.W + problem_size.stride_w - 1) / problem_size.stride_w;
  int rows_nhw_per_filter = problem_size.N * rows_h_per_filter * rows_w_per_filter;

  // Number of CTAs tile M to cover valid MMAs per starting filter postion
  int tile_m_per_filter = (rows_nhw_per_filter + tile_size_m - 1) / tile_size_m;

  return tile_m_per_filter;
}

// Computes starting Dx coord (h, w) for given starting filter postion
CUTLASS_HOST_DEVICE
void strided_dgrad_starting_coords(
  Conv2dProblemSize const &problem_size,
  FastDivmod const &stride_h_divmod, FastDivmod const &stride_w_divmod,
  int r, int s,
  int &start_h, int &start_w) {

  // function locals for remainder by fast divmod
  int pad_h_rem_, pad_w_rem_;

  // start_h  = std::abs(problem_size.stride_h - ((problem_size.pad_h % problem_size.stride_h) - r)) % problem_size.stride_h;
  stride_h_divmod.divmod(pad_h_rem_, problem_size.pad_h);
  int r_ = absolute_value(problem_size.stride_h - (pad_h_rem_ - r));
  stride_h_divmod.divmod(start_h, r_);

  //start_w  = std::abs(problem_size.stride_w - ((problem_size.pad_w % problem_size.stride_w) - s)) % problem_size.stride_w;
  stride_w_divmod.divmod(pad_w_rem_, problem_size.pad_w);
  int s_ = absolute_value(problem_size.stride_w - (pad_w_rem_ - s));
  stride_w_divmod.divmod(start_w, s_);
}

} // namespace conv
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
