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

#include "../../common/cutlass_unit_test.h"

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/core_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"

namespace test {
namespace conv {
namespace device {

using Conv3dProblemVector = std::vector<cutlass::conv::Conv3dProblemSize>;

////////////////////////////////////////////////////////////////////////////
/// Structure TestbedConv3dProblemSizes initializes and holds conv default and 
/// important network sizes
////////////////////////////////////////////////////////////////////////////
struct TestbedConv3dProblemSizes {

  //
  // Data members
  //
  int minimum_channel_size;
  Conv3dProblemVector conv3d_default_sizes;
  Conv3dProblemVector conv3d_vnet_medical_sizes;

  //
  // Methods
  //
  /// Default ctor
  TestbedConv3dProblemSizes(int minimum_channel_size_ = 64): minimum_channel_size (minimum_channel_size_) { 

    initialize_conv3d_default_sizes();
    initialize_conv3d_vnet_medical_sizes(conv3d_vnet_medical_sizes, 1 /*batch-size*/);

    filter_all();
  }

  /// Eliminates some illegal cases
  void filter_all() {

    Conv3dProblemVector *problems_vectors[] = {
      &conv3d_default_sizes,
      &conv3d_vnet_medical_sizes
    };

    for (Conv3dProblemVector *problems : problems_vectors) {
      Conv3dProblemVector filtered;

      for (cutlass::conv::Conv3dProblemSize const & problem : *problems) {
        if (!(problem.C % minimum_channel_size)) {
          filtered.push_back(problem);
        }
      }

      *problems = filtered;
    } 
  }

  // Add a few standard convolution problem sizes
  void initialize_conv3d_default_sizes() {

    conv3d_default_sizes.push_back(cutlass::conv::Conv3dProblemSize(
      {1, 1, 3, 3, minimum_channel_size}, // input size  (NDHWC)
      {8, 1, 1, 1, minimum_channel_size}, // filter size (KTRSC)
      cutlass::Coord<3>({0, 0, 0}),       // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),       // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})        // dilation (dilation_d, dilation_h, dilation_w) 
    ));

    conv3d_default_sizes.push_back(cutlass::conv::Conv3dProblemSize(
      {1, 1, 1, 8, minimum_channel_size}, // input size  (NDHWC)
      {8, 1, 1, 3, minimum_channel_size},   // filter size (KTRSC)
      cutlass::Coord<3>({1, 1, 1}),         // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),         // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})          // dilation (dilation_d, dilation_h, dilation_w)
    ));

    conv3d_default_sizes.push_back(cutlass::conv::Conv3dProblemSize(
      {1, 1, 1, 8, minimum_channel_size},   // input size  (NDHWC)
      {8, 1, 1, 3, minimum_channel_size},   // filter size (KTRSC)
      CUTLASS_STL_NAMESPACE::make_tuple(
        cutlass::Coord<3>({1, 1, 1}),       // near padding (pad_d, pad_h, pad_w)
        cutlass::Coord<3>({0, 0, 0})        // far padding (pad_d, pad_h, pad_w)
      ),
      cutlass::Coord<3>({1, 1, 1}),         // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})          // dilation (dilation_d, dilation_h, dilation_w) 
    ));

    conv3d_default_sizes.push_back(cutlass::conv::Conv3dProblemSize(
      {1, 8, 8, 8, minimum_channel_size}, // input size  (NDHWC)
      {8, 3, 3, 3, minimum_channel_size},   // filter size (KTRSC)
      cutlass::Coord<3>({1, 1, 1}),         // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),         // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})          // dilation (dilation_d, dilation_h, dilation_w) 
    ));

    conv3d_default_sizes.push_back(cutlass::conv::Conv3dProblemSize(
      {1, 8, 8, 8, minimum_channel_size},    // input size  (NDHWC)
      {8, 3, 3, 3, minimum_channel_size},    // filter size (KTRSC)
      CUTLASS_STL_NAMESPACE::make_tuple(
        cutlass::Coord<3>({1, 1, 1}),       // near padding (pad_d, pad_h, pad_w)
        cutlass::Coord<3>({0, 0, 0})        // far padding (pad_d, pad_h, pad_w)
      ),
      cutlass::Coord<3>({1, 1, 1}),          // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})           // dilation (dilation_d, dilation_h, dilation_w) 
    ));

    conv3d_default_sizes.push_back(cutlass::conv::Conv3dProblemSize(
      {1, 16, 16, 16, minimum_channel_size}, // input size  (NDHWC)
      {8, 3, 3, 3, minimum_channel_size},   // filter size (KTRSC)
      cutlass::Coord<3>({1, 1, 1}),         // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),         // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})          // dilation (dilation_d, dilation_h, dilation_w) 
    ));

    conv3d_default_sizes.push_back(cutlass::conv::Conv3dProblemSize(
      {1, 1, 15, 19, 160},              // input size  (NDHWC)
      {224, 1, 3, 6, 160},              // filter size (KTRSC)
      cutlass::Coord<3>({0, 0, 0}),     // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),     // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})      // dilation (dilation_d, dilation_h, dilation_w) 
    )); 

    conv3d_default_sizes.push_back(cutlass::conv::Conv3dProblemSize(
      {1, 2, 1, 1, minimum_channel_size},  // input size  (NDHWC)
      {8, 2, 1, 1, minimum_channel_size},  // filter size (KTRSC)
      cutlass::Coord<3>({0, 0, 0}),        // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),        // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})         // dilation (dilation_d, dilation_h, dilation_w) 
    ));

    conv3d_default_sizes.push_back(cutlass::conv::Conv3dProblemSize(
      {1,  1, 7, 7, minimum_channel_size}, // input size  (NDHWC)
      {16, 1, 3, 3, minimum_channel_size}, // filter size (KTRSC)
      cutlass::Coord<3>({0, 0, 0}),        // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),        // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})         // dilation (dilation_d, dilation_h, dilation_w) 
    ));


    conv3d_default_sizes.push_back(cutlass::conv::Conv3dProblemSize(
      {1, 11, 15, 19, 64},              // input size  (NDHWC)
      {32, 4, 3, 6, 64},                // filter size (KTRSC)
      cutlass::Coord<3>({2, 1, 3}),     // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),     // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})      // dilation (dilation_d, dilation_h, dilation_w) 
    ));
  }

  // Add vnet layers to unit testing sizes 
  void initialize_conv3d_vnet_medical_sizes(Conv3dProblemVector &conv3d_problem_vector, int batch_size = 1) {

    conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
      {batch_size, 32, 32, 32, 16},     // input size  (NDHWC)
      {32, 2, 2, 2, 16},              // filter size (KTRSC)
      cutlass::Coord<3>({0, 0, 0}),    // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({2, 2, 2}),    // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})     // dilation (dilation_d, dilation_h, dilation_w) 
    ));
  
  
    conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
      {batch_size, 16, 16, 16, 32},     // input size  (NDHWC)
      {32, 3, 3, 3, 32},              // filter size (KTRSC)
      cutlass::Coord<3>({1, 1, 1}),    // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),    // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})     // dilation (dilation_d, dilation_h, dilation_w) 
    ));
  
  
    conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
      {batch_size, 16, 16, 16, 32},     // input size  (NDHWC)
      {64, 2, 2, 2, 32},              // filter size (KTRSC)
      cutlass::Coord<3>({0, 0, 0}),    // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({2, 2, 2}),    // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})     // dilation (dilation_d, dilation_h, dilation_w) 
    ));
  
  
    conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
      {batch_size, 8, 8, 8, 64},     // input size  (NDHWC)
      {64, 3, 3, 3, 64},              // filter size (KTRSC)
      cutlass::Coord<3>({1, 1, 1}),    // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),    // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})     // dilation (dilation_d, dilation_h, dilation_w) 
    ));
  
  
    conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
      {batch_size, 8, 8, 8, 64},     // input size  (NDHWC)
      {128, 2, 2, 2, 64},              // filter size (KTRSC)
      cutlass::Coord<3>({0, 0, 0}),    // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({2, 2, 2}),    // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})     // dilation (dilation_d, dilation_h, dilation_w) 
    ));
  
  
    conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
      {batch_size, 4, 4, 4, 128},     // input size  (NDHWC)
      {128, 3, 3, 3, 128},              // filter size (KTRSC)
      cutlass::Coord<3>({1, 1, 1}),    // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),    // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})     // dilation (dilation_d, dilation_h, dilation_w) 
    ));
  
  
    conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
      {batch_size, 8, 8, 8, 128},     // input size  (NDHWC)
      {128, 3, 3, 3, 128},              // filter size (KTRSC)
      cutlass::Coord<3>({1, 1, 1}),    // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),    // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})     // dilation (dilation_d, dilation_h, dilation_w) 
    ));
  
  
    conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
      {batch_size, 16, 16, 16, 64},     // input size  (NDHWC)
      {64, 3, 3, 3, 64},              // filter size (KTRSC)
      cutlass::Coord<3>({1, 1, 1}),    // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({1, 1, 1}),    // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})     // dilation (dilation_d, dilation_h, dilation_w) 
    ));
  
  
    conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
      {batch_size, 32, 32, 32, 16},     // input size  (NDHWC)
      {64, 2, 2, 2, 16},              // filter size (KTRSC)
      cutlass::Coord<3>({0, 0, 0}),    // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({2, 2, 2}),    // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})     // dilation (dilation_d, dilation_h, dilation_w) 
    ));
  
  
    conv3d_problem_vector.push_back(cutlass::conv::Conv3dProblemSize(
      {batch_size, 16, 16, 16, 32},     // input size  (NDHWC)
      {128, 2, 2, 2, 32},              // filter size (KTRSC)
      cutlass::Coord<3>({0, 0, 0}),    // padding (pad_d, pad_h, pad_w)
      cutlass::Coord<3>({2, 2, 2}),    // stride (stride_d, stride_h, stride_w)
      cutlass::Coord<3>({1, 1, 1})     // dilation (dilation_d, dilation_h, dilation_w) 
    ));

  }

};

} // namespace device
} // namespace conv
} // namespace test
