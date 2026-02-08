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
    \brief Implicit GEMM testbed sizes for Conv2d problem
*/
#pragma once

#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"

namespace test {
namespace conv {
namespace device {

using Conv2dProblemVector = std::vector<cutlass::conv::Conv2dProblemSize>;

//
// Structures to prune items from Conv2dProblemVector
//
// Specification template for pruning items for convolution problem lists
template <typename T> struct Specification
{
  virtual ~Specification() = default;
  virtual bool is_satisfied(T item) const = 0;
};

// input size  (NHWC) specification
struct InputSizeSpecification : Specification<cutlass::conv::Conv2dProblemSize>
{
  cutlass::Tensor4DCoord input_size;

  InputSizeSpecification(cutlass::Tensor4DCoord input_size_) : input_size(input_size_) {}

  bool is_satisfied(cutlass::conv::Conv2dProblemSize item) const override {
    return ((input_size.n() == item.N) && (input_size.h() == item.H) && (input_size.w() == item.W) && (input_size.c() == item.C));
  }
};

// stride (stride_h, stride_w) specification
struct StrideSpecification : Specification<cutlass::conv::Conv2dProblemSize>
{
  cutlass::MatrixCoord stride;

  StrideSpecification(cutlass::MatrixCoord stride_) : stride(stride_) {}

  bool is_satisfied(cutlass::conv::Conv2dProblemSize item) const override {
    return ((stride.row() == item.stride_h) && (stride.column() == item.stride_h));
  }
};

// channel (C,K) specification, must be multiple of minimum channel
struct ChannelDivisibilitySpecification : Specification<cutlass::conv::Conv2dProblemSize>
{
  int channel_multiple;

  ChannelDivisibilitySpecification(int channel_multiple_) : channel_multiple(channel_multiple_) {}

  bool is_satisfied(cutlass::conv::Conv2dProblemSize item) const override {
    return ((item.K % channel_multiple == 0) && (item.C % channel_multiple == 0));
  }
};

//
// Pruning function for items from Conv2dProblemVector based on a Specification
//
inline Conv2dProblemVector prune(Conv2dProblemVector const &items,
                           Specification<cutlass::conv::Conv2dProblemSize> const &spec)
{
  Conv2dProblemVector pruned_list;

  for (auto& p : items)
    if (spec.is_satisfied(p))
      pruned_list.push_back(p);
  return pruned_list;
}


////////////////////////////////////////////////////////////////////////////
/// Structure TestbedConv2dProblemSizes initializes and holds conv default and 
/// important network sizes
////////////////////////////////////////////////////////////////////////////
struct TestbedConv2dProblemSizes {

  //
  // Data members
  //
  int minimum_channel_size;

  Conv2dProblemVector conv2d_default_sizes;
  Conv2dProblemVector conv2d_rigorous_sizes;
  Conv2dProblemVector conv2d_resnet50_sizes;
  Conv2dProblemVector conv2d_resnet50_sizes_perf;

  //
  // Methods
  //
  /// Default ctor
  TestbedConv2dProblemSizes(int minimum_channel_size_ = 64): minimum_channel_size (minimum_channel_size_) { 
    initialize_conv2d_default_sizes();
    initialize_conv2d_rigorous_sizes();
    initialize_conv2d_resnet50_sizes(conv2d_resnet50_sizes, 1 /*batch-size*/);

    initialize_conv2d_resnet50_sizes(conv2d_resnet50_sizes_perf, 34 /*batch-size*/);
    filter_all();
  }

  /// Eliminates some illegal cases
  void filter_all() {

    Conv2dProblemVector *problems_vectors[] = {
      &conv2d_default_sizes,
      &conv2d_rigorous_sizes,
      &conv2d_resnet50_sizes,
      &conv2d_resnet50_sizes_perf
    };

    for (Conv2dProblemVector *problems : problems_vectors) {
      Conv2dProblemVector filtered;

      for (cutlass::conv::Conv2dProblemSize const & problem : *problems) {
        if (!(problem.C % minimum_channel_size)) {
          filtered.push_back(problem);
        }
      }

      *problems = filtered;
    } 
  }

  // Add a few standard convolution problem sizes
  void initialize_conv2d_default_sizes() {

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Small input size x stride (1,1)
    // C < CTA::K and non-multiples of CTA::K. Typical CTA::K = {32, 64}
    ////////////////////////////////////////////////////////////////////////////////////////////
    
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 1, 1, minimum_channel_size},   // input size  (NHWC)
      {8, 1, 1, minimum_channel_size},   // filter size (KRSC)
      {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 1, 8, minimum_channel_size},   // input size  (NHWC)
      {8, 1, 3, minimum_channel_size},   // filter size (KRSC)
      {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 7, 8, minimum_channel_size},   // input size  (NHWC)
      {8, 3, 3, minimum_channel_size},   // filter size (KRSC)
      {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 7, 9, minimum_channel_size},  // input size  (NHWC)
      {8, 4, 4, minimum_channel_size},  // filter size (KRSC)
      {1, 1, 1, 1},                     // padding (pad_h, _, pad_w, _)
      {1, 1},                           // stride (stride_h, stride_w)
      {1, 1}                            // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {2, 7, 9, minimum_channel_size},   // input size  (NHWC)
      {8, 5, 5, minimum_channel_size},   // filter size (KRSC)
      {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {3, 7, 9, minimum_channel_size},   // input size  (NHWC)
      {8, 6, 5, minimum_channel_size},   // filter size (KRSC)
      {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {3, 7, 9, minimum_channel_size},   // input size  (NHWC)
      {8, 6, 6, minimum_channel_size},   // filter size (KRSC)
      {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {3, 7, 9, minimum_channel_size},   // input size  (NHWC)
      {8, 7, 7, minimum_channel_size},   // filter size (KRSC)
      {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Small input size x stride (1,1) asymmetric paddings (1, 0, 1, 0)
    // C < CTA::K and non-multiples of CTA::K. Typical CTA::K = {32, 64}
    ////////////////////////////////////////////////////////////////////////////////////////////
    
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 1, 1, minimum_channel_size},   // input size  (NHWC)
      {8, 1, 1, minimum_channel_size},   // filter size (KRSC)
      {1, 0, 1, 0},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 1, 8, minimum_channel_size},   // input size  (NHWC)
      {8, 1, 3, minimum_channel_size},   // filter size (KRSC)
      {1, 0, 1, 0},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 7, 8, minimum_channel_size},   // input size  (NHWC)
      {8, 3, 3, minimum_channel_size},   // filter size (KRSC)
      {1, 0, 1, 0},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 7, 9, minimum_channel_size},  // input size  (NHWC)
      {8, 4, 4, minimum_channel_size},  // filter size (KRSC)
      {1, 0, 1, 0},                     // padding (pad_h, _, pad_w, _)
      {1, 1},                           // stride (stride_h, stride_w)
      {1, 1}                            // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {2, 7, 9, minimum_channel_size},   // input size  (NHWC)
      {8, 5, 5, minimum_channel_size},   // filter size (KRSC)
      {1, 0, 1, 0},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {3, 7, 9, minimum_channel_size},   // input size  (NHWC)
      {8, 6, 5, minimum_channel_size},   // filter size (KRSC)
      {1, 0, 1, 0},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {3, 7, 9, minimum_channel_size},   // input size  (NHWC)
      {8, 6, 6, minimum_channel_size},   // filter size (KRSC)
      {1, 0, 1, 0},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {3, 7, 9, minimum_channel_size},   // input size  (NHWC)
      {8, 7, 7, minimum_channel_size},   // filter size (KRSC)
      {1, 0, 1, 0},                      // padding (pad_h, _, pad_w, _)
      {1, 1},                            // stride (stride_h, stride_w)
      {1, 1}                             // dilation (dilation_h, dilation_w) 
    ));

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Small input size x stride (2,2)
    // C < CTA::K and non-multiples of CTA::K. Typical CTA::K = {32, 64}
    ////////////////////////////////////////////////////////////////////////////////////////////
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 11, 7, minimum_channel_size},  // input size  (NHWC)
      {8, 1, 1, minimum_channel_size},    // filter size (KRSC)
      {0, 0, 0, 0},                       // padding (pad_h, _, pad_w, _)
      {2, 2},                             // stride (stride_h, stride_w)
      {1, 1}                              // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 11, 7, minimum_channel_size},   // input size  (NHWC)
      {8, 3, 3, minimum_channel_size},     // filter size (KRSC)
      {1, 1, 1, 1},                        // padding (pad_h, _, pad_w, _)
      {2, 2},                              // stride (stride_h, stride_w)
      {1, 1}                               // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 13, 11, minimum_channel_size},   // input size  (NHWC)
      {8, 1, 1, minimum_channel_size},     // filter size (KRSC)
      {1, 1, 1, 1},                        // padding (pad_h, _, pad_w, _)
      {2, 2},                              // stride (stride_h, stride_w)
      {1, 1}                               // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 17, 19, minimum_channel_size},   // input size  (NHWC)
      {16, 2, 2, minimum_channel_size},   // filter size (KRSC)
      {1, 1, 1, 1},    // padding (pad_h, _, pad_w, _)
      {2, 2},          // stride (stride_h, stride_w)
      {1, 1}           // dilation (dilation_h, dilation_w) 
    ));
  
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 23, 5, minimum_channel_size},   // input size  (NHWC)
      {16, 3, 3, minimum_channel_size},   // filter size (KRSC)
      {1, 1, 1, 1},    // padding (pad_h, _, pad_w, _)
      {2, 2},          // stride (stride_h, stride_w)
      {1, 1}           // dilation (dilation_h, dilation_w) 
    ));
  
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize( 
      {1, 13, 17, 8},   // input size  (NHWC)
      {24, 3, 3, 8},   // filter size (KRSC)
      {0, 0, 0, 0},    // padding (pad_h, _, pad_w, _)
      {2, 2},          // stride (stride_h, stride_w)
      {1, 1}           // dilation (dilation_h, dilation_w) 
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 23, 21, 8},     // input size (NHWC)
      {24, 3, 3, 8},     // filter size (KRSC)
      {1, 1, 1, 1},     // padding (pad_h, _, pad_w, _)
      {3, 3},           // stride (stride_h, stride_w)
      {1, 1}            // dilation (dilation_h, dilation_w)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 20, 24, 8},   // input size (NHWC)
      {40, 3, 3, 8},     // filter size (KRSC)
      {3, 3, 3, 3},     // padding (pad_h, _, pad_w, _)
      {3, 3},           // stride (stride_h, stride_w)
      {1, 1}            // dilation (dilation_h, dilation_w)
    ));

    ////////////////////////////////////////////////////////////////////////////////////
    // Medium input size (1x16x16x128), filter size (1x1, 2x2, 3x3, 5x5), stride (1, 1) 
    ////////////////////////////////////////////////////////////////////////////////////
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 15, 19, 160},   // input size  (NHWC)
      {224, 1, 1, 160},   // filter size (KRSC)
      {0, 0, 0, 0},       // padding (pad_h, _, pad_w, _) 
      {1, 1},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w) 
    ));
    
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 19, 37, 160},     // input size  (NHWC)
      {224, 3, 3, 160},     // filter size (KRSC)
      {1, 1, 1, 1},         // padding (pad_h, _, pad_w, _)
      {2, 2},               // stride (stride_h, stride_w)
      {1, 1}                // dilation (dilation_h, dilation_w)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 16, 16, 160},   // input size  (NHWC)
      {224, 2, 3, 160},   // filter size (KRSC)
      {1, 1, 1, 1},       // padding (pad_h, _, pad_w, _) 
      {1, 1},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w) 
    ));
  
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 23, 21, 128},  // input size  (NHWC)
      {224, 3, 3, 128},  // filter size (KRSC)
      {1, 1, 1, 1},      // padding (pad_h, _, pad_w, _)
      {1, 1},            // stride (stride_h, stride_w)
      {1, 1}             // dilation (dilation_h, dilation_w)
    ));
  
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 29, 37, 160},      // input size  (NHWC)
      {224, 5, 5, 160},      // filter size (KRSC)
      {2, 2, 2, 2},          // padding (pad_h, _, pad_w, _)
      {1, 1},                // stride (stride_h, stride_w)
      {1, 1}                 // dilation (dilation_h, dilation_w)
    ));

    ////////////////////////////////////////////////////////////////////////////////////
    // C > CTA::K and non-multiples of CTA::K. Typical CTA::K = {32, 64}
    ////////////////////////////////////////////////////////////////////////////////////
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 15, 19, 32 + minimum_channel_size},     // input size  (NHWC)
      {96, 3, 3, 32 + minimum_channel_size},      // filter size (KRSC)
      {1, 1, 1, 1},                               // padding (pad_h, _, pad_w, _)
      {1, 1},                                     // stride (stride_h, stride_w)
      {1, 1}                                      // dilation (dilation_h, dilation_w)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 16, 24, 64 + minimum_channel_size},     // input size  (NHWC)
      {96, 3, 3, 64 + minimum_channel_size},      // filter size (KRSC)
      {1, 1, 1, 1},                               // padding (pad_h, _, pad_w, _)
      {1, 1},                                     // stride (stride_h, stride_w)
      {1, 1}                                      // dilation (dilation_h, dilation_w)
    ));

    ////////////////////////////////////////////////////////////////////////////////////
    // Medium input size, filter size (1x1, 3,x3, 5x5, 7x7), stride (2, 2)  
    //////////////////////////////////////////////////////////////////////////////////// 
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 13, 16, 288},   // input size  (NHWC)
      {160, 5, 5, 288},   // filter size (KRSC)
      {2, 2, 2, 2},       // padding (pad_h, _, pad_w, _)
      {2, 2},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 55, 51, 256},   // input size (NHWC)
      {512, 1, 1, 256},   // filter size (KRSC)
      {0, 0, 0, 0},       // padding (pad_h, _, pad_w, _)
      {2, 2},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 71, 80, 32},    // input size (NHWC)
      {64, 5, 5, 32},     // filter size (KRSC)
      {2, 2, 2, 2},       // padding (pad_h, _, pad_w, _)
      {2, 2},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 224, 224, 8},   // input size (NHWC)
      {64, 7, 7, 8},      // filter size (KRSC)
      {3, 3, 3, 3},       // padding (pad_h, _, pad_w, _)
      {2, 2},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w)
    ));

    ////////////////////////////////////////////////////////////////////////////////////
    // Medium input size stride (3, 3), filter (3, 3), non-default padding
    ////////////////////////////////////////////////////////////////////////////////////
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 27, 23, 256},     // input size (NHWC)
      {512, 3, 3, 256},     // filter size (KRSC)
      {0, 0, 0, 0},         // padding (pad_h, _, pad_w, _)
      {3, 3},               // stride (stride_h, stride_w)
      {1, 1}                // dilation (dilation_h, dilation_w)
    ));
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Medium input size padding > stride, asymmetric filter, padding and striding
    ////////////////////////////////////////////////////////////////////////////////////
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 27, 31, 256},     // input size (NHWC)
      {512, 3, 3, 256},     // filter size (KRSC)
      {5, 5, 7, 7},         // padding (pad_h, _, pad_w, _)
      {3, 4},               // stride (stride_h, stride_w)
      {1, 1}                // dilation (dilation_h, dilation_w)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 27, 35, 256},     // input size (NHWC)
      {512, 7, 5, 256},     // filter size (KRSC)
      {11, 11, 7, 7},       // padding (pad_h, _, pad_w, _)
      {3, 5},               // stride (stride_h, stride_w)
      {1, 1}                // dilation (dilation_h, dilation_w)
    ));

    ////////////////////////////////////////////////////////////////////////////////////
    // Medium input size *mixed* stride (1, 2) and (2, 1), 
    // filter (3, 3), default padding
    ////////////////////////////////////////////////////////////////////////////////////
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 27, 27, 256},     // input size (NHWC)
      {512, 3, 3, 256},     // filter size (KRSC)
      {1, 1, 1, 1},         // padding (pad_h, _, pad_w, _)
      {1, 2},               // stride (stride_h, stride_w)
      {1, 1}                // dilation (dilation_h, dilation_w)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 27, 27, 256},     // input size (NHWC)
      {512, 3, 3, 256},     // filter size (KRSC)
      {1, 1, 1, 1},         // padding (pad_h, _, pad_w, _)
      {2, 1},               // stride (stride_h, stride_w)
      {1, 1}                // dilation (dilation_h, dilation_w)
    ));

    /////////////////////////////////////////////////////////////////////////////
    // Additional input size 
    /////////////////////////////////////////////////////////////////////////////
    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {3, 28, 28, 256},  // input size  (NHWC)
      {256, 2, 2, 256},  // filter size (KRSC)
      {0, 0, 0, 0},      // padding (pad_h, _, pad_w, _)
      {2, 2},            // stride (stride_h, stride_w)
      {1, 1}             // dilation (dilation_h, dilation_w)
    ));
   
   conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 32, 32, 16},  // input size  (NHWC)
      {32, 3, 3, 16},  // filter size (KRSC)
      {1, 1, 1, 1},      // padding (pad_h, _, pad_w, _)
      {6, 2},            // stride (stride_h, stride_w)
      {1, 1}             // dilation (dilation_h, dilation_w)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {32, 24, 32, 32},  // input size  (NHWC)
      {32, 1, 2, 32},    // filter size (KRSC)
      {0, 0, 0, 0},      // padding (pad_h, _, pad_w, _)
      {1, 1},            // stride (stride_h, stride_w)
      {1, 1}             // dilation (dilation_h, dilation_w)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {4, 4, 5, 128},     // input size  (NHWC)
      {256, 3, 6, 128},   // filter size (KRSC)
      {0, 0, 0, 0},       // padding (pad_h, _, pad_w, _)
      {1, 1},             // stride (stride_h, stride_w)
      {1, 1},             // dilation (dilation_h, dilation_w)
      {4, 3, 3, 256}      // output size (NPQK)
    ));

    conv2d_default_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {4, 2, 3, 256},     // input size  (NHWC)
      {328, 3, 5, 256},   // filter size (KRSC)
      {1, 1, 1, 1},       // padding (pad_h, _, pad_w, _)
      {1, 1},             // stride (stride_h, stride_w)
      {1, 1},             // dilation (dilation_h, dilation_w)
      {4, 1, 1, 328}      // output size (NPQK)
    ));
  }


  // Add a few large and rigorous convolution problem sizes
  void initialize_conv2d_rigorous_sizes() {

#if CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED                  
  conv2d_rigorous_sizes.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 124, 224, 96},    // input size  (NHWC)
    {24, 7, 7, 96},       // filter size (KRSC)
    {1, 229, 129, 32}     // output size (NPQK)
  ));

  conv2d_rigorous_sizes.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 233, 35, 48},     // input size  (NHWC)
    {24, 7, 5, 48},       // filter size (KRSC)
    {1, 233, 35, 24}      // output size (NPQK)
  ));

#endif 

  }


  // Add resent50 layers to unit testing sizes 
  void initialize_conv2d_resnet50_sizes(Conv2dProblemVector &conv2d_problem_vector, int batch_size = 1){

#if 0 // Resnet50 first layer (layer_id = 0) with channel = 3 is not supported in cutlass
    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(   
      [1, 224, 224, 3],           // input size (NHWC)
      [64, 7, 7, 3],              // filter size (KRSC)
      [3, 3, 3, 3],               // padding (pad_h, _, pad_w, _)
      [2, 2],                     // stride (stride_h, stride_w)
      [1, 1],                     // dilation (dilation_h, dilation_w)
    ));
#endif

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 56, 56, 64},   // input size (NHWC)
      {256, 1, 1, 64},            // filter size (KRSC)
      {0, 0, 0, 0},               // padding (pad_h, _, pad_w, _)
      {1, 1},                     // stride (stride_h, stride_w)
      {1, 1}                      // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 56, 56, 64},   // input size (NHWC)
      {64, 1, 1, 64},             // filter size (KRSC)
      {0, 0, 0, 0},               // padding (pad_h, _, pad_w, _)
      {1, 1},                     // stride (stride_h, stride_w)
      {1, 1}                      // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 56, 56, 64},    // input size (NHWC)
      {64, 3, 3, 64},             // filter size (KRSC)
      {1, 1, 1, 1},               // padding (pad_h, _, pad_w, _)
      {1, 1},                     // stride (stride_h, stride_w)
      {1, 1}                      // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 56, 56, 256},   // input size (NHWC)
      {64, 1, 1, 256},             // filter size (KRSC)
      {0, 0, 0, 0},                // padding (pad_h, _, pad_w, _)
      {1, 1},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));

   conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 56, 56, 256},   // input size (NHWC)
      {512, 1, 1, 256},            // filter size (KRSC)
      {0, 0, 0, 0},                // padding (pad_h, _, pad_w, _)
      {2, 2},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 56, 56, 256},   // input size (NHWC)
      {128, 1, 1, 256},            // filter size (KRSC)
      {0, 0, 0, 0},                // padding (pad_h, _, pad_w, _)
      {2, 2},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 28, 28, 128},   // input size (NHWC)
      {128, 3, 3, 128},            // filter size (KRSC)
      {1, 1, 1, 1},                // padding (pad_h, _, pad_w, _)
      {1, 1},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 28, 28, 128},   // input size (NHWC)
      {512, 1, 1, 128},            // filter size (KRSC)
      {0, 0, 0, 0},                // padding (pad_h, _, pad_w, _)
      {1, 1},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 28, 28, 512},   // input size (NHWC)
      {128, 1, 1, 512},            // filter size (KRSC)
      {0, 0, 0, 0},                // padding (pad_h, _, pad_w, _)
      {1, 1},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));
 
    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 28, 28, 512},   // input size (NHWC)
      {1024, 1, 1, 512},           // filter size (KRSC)
      {0, 0, 0, 0},                // padding (pad_h, _, pad_w, _)
      {2, 2},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));
        
    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 28, 28, 512},   // input size (NHWC)
      {256, 1, 1, 512},            // filter size (KRSC)
      {0, 0, 0, 0},                // padding (pad_h, _, pad_w, _)
      {2, 2},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 14, 14, 256},   // input size (NHWC)
      {256, 3, 3, 256},            // filter size (KRSC)
      {1, 1, 1, 1},                // padding (pad_h, _, pad_w, _)
      {1, 1},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 14, 14, 256},   // input size (NHWC)
      {1024, 1, 1, 256},           // filter size (KRSC)
      {0, 0, 0, 0},                // padding (pad_h, _, pad_w, _)
      {1, 1},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 14, 14, 1024},   // input size (NHWC)
      {256, 1, 1, 1024},            // filter size (KRSC)
      {0, 0, 0, 0},                 // padding (pad_h, _, pad_w, _)
      {1, 1},                       // stride (stride_h, stride_w)
      {1, 1}                        // dilation (dilation_h, dilation_w)
    ));

     conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 14, 14, 1024},   // input size (NHWC)
      {2048, 1, 1, 1024},           // filter size (KRSC)
      {0, 0, 0, 0},                 // padding (pad_h, _, pad_w, _)
      {2, 2},                       // stride (stride_h, stride_w)
      {1, 1}                        // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 14, 14, 1024},   // input size (NHWC)
      {512, 1, 1, 1024},            // filter size (KRSC)
      {0, 0, 0, 0},                 // padding (pad_h, _, pad_w, _)
      {2, 2},                       // stride (stride_h, stride_w)
      {1, 1}                        // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 7, 7, 512},     // input size (NHWC)
      {512, 3, 3, 512},            // filter size (KRSC)
      {1, 1, 1, 1},                // padding (pad_h, _, pad_w, _)
      {1, 1},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 7, 7, 512},     // input size (NHWC)
      {2048, 1, 1, 512},           // filter size (KRSC)
      {0, 0, 0, 0},                // padding (pad_h, _, pad_w, _)
      {1, 1},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));

    conv2d_problem_vector.push_back(cutlass::conv::Conv2dProblemSize(
      {batch_size, 7, 7, 2048},    // input size (NHWC)
      {512, 1, 1, 2048},           // filter size (KRSC)
      {0, 0, 0, 0},                // padding (pad_h, _, pad_w, _)
      {1, 1},                      // stride (stride_h, stride_w)
      {1, 1}                       // dilation (dilation_h, dilation_w)
    ));
 }

};


////////////////////////////////////////////////////////////////////////////
/// Structure TestbedGroupConv2dProblemSizes initializes and holds group conv default and
/// important network sizes
////////////////////////////////////////////////////////////////////////////
struct TestbedGroupConv2dProblemSizes {

  //
  // Data members
  //
  int threadblock_n;
  int threadblock_k;
  int minimum_channel_size;

  Conv2dProblemVector default_single_group_sizes;
  Conv2dProblemVector default_multiple_group_sizes;

  //
  // Methods
  //
  /// Default ctor
  TestbedGroupConv2dProblemSizes(
    int threadblock_n_,
    int threadblock_k_,
    int minimum_channel_size_ = 64)
  : threadblock_n (threadblock_n_),
    threadblock_k (threadblock_k_),
    minimum_channel_size (minimum_channel_size_) {
    initialize_group_conv2d_default_sizes();
    filter_all();
  }

  /// Eliminates some illegal cases
  void filter_all() {

    Conv2dProblemVector *problems_vectors[] = {
      &default_single_group_sizes,
      &default_multiple_group_sizes
    };

    for (Conv2dProblemVector *problems : problems_vectors) {
      Conv2dProblemVector filtered;

      for (cutlass::conv::Conv2dProblemSize const & problem : *problems) {
        if (!((problem.C / problem.groups) % minimum_channel_size)) {
          filtered.push_back(problem);
        }
      }

      *problems = filtered;
    }
  }

  // Add a few standard convolution problem sizes
  void initialize_group_conv2d_default_sizes() {

    ////////////////////////////////////////////////////////////////////////////////////
    // One group calculated by one or multiple CTAs: k_per_group % CTA::N = 0
    // One CTA calculates a single group
    ////////////////////////////////////////////////////////////////////////////////////

    for (int cta_per_group_k = 1; cta_per_group_k < 4; ++cta_per_group_k) {
      // groups = 2, 3, 4
      for (int groups = 2; groups < 5; ++groups) {

        int conv_k = cta_per_group_k * threadblock_n * groups;
        default_single_group_sizes.push_back(cutlass::conv::Conv2dProblemSize(
          {1, 8, 8, threadblock_k * 2 * groups},        // input size  (NHWC)
          {conv_k, 3, 3, threadblock_k * 2},            // filter size (KRSC)
          {1, 1, 1, 1},                                 // padding (pad_h, _, pad_w, _)
          {1, 1},                                       // stride (stride_h, stride_w)
          {1, 1},                                       // dilation (dilation_h, dilation_w)
          cutlass::conv::Mode::kCrossCorrelation,
          1,                                            // split_k_slices
          groups                                        // groups
        ));

      } // loop groups
    } // loop cta_per_group_k

    // Partial gemm_k: k_per_group == CTA::N && channels_per_group < CTA::K
    default_single_group_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 8, 8, threadblock_k},                       // input size  (NHWC)
      {threadblock_n * 2, 3, 3, threadblock_k / 2},   // filter size (KRSC)
      {1, 1, 1, 1},                                   // padding (pad_h, _, pad_w, _)
      {1, 1},                                         // stride (stride_h, stride_w)
      {1, 1},                                         // dilation (dilation_h, dilation_w)
      cutlass::conv::Mode::kCrossCorrelation,
      1,                                              // split_k_slices
      2                                               // groups
    ));

    // Larger problem sizes
    
    default_single_group_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 56, 56, 696},                               // input size  (NHWC)
      {768, 3, 3, 232},                               // filter size (KRSC)
      {1, 1, 1, 1},                                   // padding (pad_h, _, pad_w, _)
      {2, 2},                                         // stride (stride_h, stride_w)
      {1, 1},                                         // dilation (dilation_h, dilation_w)
      cutlass::conv::Mode::kCrossCorrelation,
      1,                                              // split_k_slices
      3                                               // groups
    ));
    default_single_group_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 14, 14, 1392},                              // input size  (NHWC)
      {1536, 3, 3, 232},                              // filter size (KRSC)
      {1, 1, 1, 1},                                   // padding (pad_h, _, pad_w, _)
      {1, 1},                                         // stride (stride_h, stride_w)
      {1, 1},                                         // dilation (dilation_h, dilation_w)
      cutlass::conv::Mode::kCrossCorrelation,
      1,                                              // split_k_slices
      3                                               // groups
    ));

    ////////////////////////////////////////////////////////////////////////////////////
    // One CTA calculate multiple groups: CTA::N % k_per_group = 0
    ////////////////////////////////////////////////////////////////////////////////////

    // 2 groups per CTA
    default_multiple_group_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 8, 8, threadblock_k * 4},                   // input size  (NHWC)
      {threadblock_n, 3, 3, threadblock_k * 2},       // filter size (KRSC)
      {1, 1, 1, 1},                                   // padding (pad_h, _, pad_w, _)
      {1, 1},                                         // stride (stride_h, stride_w)
      {1, 1},                                         // dilation (dilation_h, dilation_w)
      cutlass::conv::Mode::kCrossCorrelation,
      1,                                              // split_k_slices
      2                                               // groups
    ));

    // 2 groups per CTA and partial gemm_k
    default_multiple_group_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 8, 8, threadblock_k},                       // input size  (NHWC)
      {threadblock_n, 3, 3, threadblock_k / 2},       // filter size (KRSC)
      {1, 1, 1, 1},                                   // padding (pad_h, _, pad_w, _)
      {1, 1},                                         // stride (stride_h, stride_w)
      {1, 1},                                         // dilation (dilation_h, dilation_w)
      cutlass::conv::Mode::kCrossCorrelation,
      1,                                              // split_k_slices
      2                                               // groups
    ));

    // 4 groups per CTA
    default_multiple_group_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 8, 8, threadblock_k * 8},                   // input size  (NHWC)
      {threadblock_n / 2, 3, 3, threadblock_k * 2},   // filter size (KRSC)
      {1, 1, 1, 1},                                   // padding (pad_h, _, pad_w, _)
      {1, 1},                                         // stride (stride_h, stride_w)
      {1, 1},                                         // dilation (dilation_h, dilation_w)
      cutlass::conv::Mode::kCrossCorrelation,
      1,                                              // split_k_slices
      4                                               // groups
    ));

    // 4 groups per CTA and partial gemm_k
    default_multiple_group_sizes.push_back(cutlass::conv::Conv2dProblemSize(
      {1, 8, 8, threadblock_k * 2},                   // input size  (NHWC)
      {threadblock_n / 2, 3, 3, threadblock_k / 2},   // filter size (KRSC)
      {1, 1, 1, 1},                                   // padding (pad_h, _, pad_w, _)
      {1, 1},                                         // stride (stride_h, stride_w)
      {1, 1},                                         // dilation (dilation_h, dilation_w)
      cutlass::conv::Mode::kCrossCorrelation,
      1,                                              // split_k_slices
      4                                               // groups
    ));
  }

};


} // namespace device
} // namespace conv
} // namespace test
