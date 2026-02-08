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
    \brief Tests for TensorReduce family of device-wide operators
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/reduction/device/tensor_reduce.h"

#include "cutlass/functional.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This reduces the C dimension, transforming an NHWC tensor into NHWC with C=1.
template <typename TensorReduction, typename ElementCompute = typename TensorReduction::ElementCompute>
bool TestAllReduction_NHWC_reduce_c(ElementCompute reduction_identity = ElementCompute()) {

  using Layout = typename TensorReduction::Layout;
  using ElementOutput = typename TensorReduction::ElementOutput;
  using ElementSource = typename TensorReduction::ElementSource;

  int const kV = TensorReduction::kVectorLength;

  int const N_indices[] = {3, 13};
  int const H_indices[] = {5, 17};
  int const W_indices[] = {7, 19};
  int const C_indices[] = {2049, 2048, 2047, 384, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1};
  
  for (int N : N_indices) {
    for (int H : H_indices) {
      for (int W : W_indices) {
        for (int Cx : C_indices) {

          int C = Cx * kV;

          cutlass::HostTensor<ElementSource, Layout> src_tensor({N, H, W, C});
          cutlass::HostTensor<ElementOutput, Layout> dst_tensor({N, H, W, 1});

          cutlass::reference::host::TensorFillRandomUniform(
            src_tensor.host_view(), 17, 10, -10, 0);

          dst_tensor.sync_device();
          src_tensor.sync_device();

          // Execute a tensor reduction over rank 3 (the 'C' dimension is reduced; NHWC => NHW)
          TensorReduction reduction(src_tensor.extent(), 3);

          cutlass::DeviceAllocation<uint8_t> device_workspace(reduction.workspace_size());

          cutlass::Status status = reduction.reduce(
            dst_tensor.device_ref(),
            src_tensor.device_ref(),
            device_workspace.get(),
            reduction_identity
          );

          EXPECT_EQ(status, cutlass::Status::kSuccess);
          EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
          
          dst_tensor.sync_host();

          typename TensorReduction::ReductionOp reduction_op;

          //
          // Reference check
          //
          for (int n = 0; n < src_tensor.extent().n(); ++n) {
            for (int h = 0; h < src_tensor.extent().h(); ++h) {
              for (int w = 0; w < src_tensor.extent().w(); ++w) {

                ElementCompute c_accum = reduction_identity;

                for (int c = 0; c < src_tensor.extent().c(); ++c) {
                  c_accum = reduction_op(c_accum, ElementCompute(src_tensor.at({n, h, w, c})));
                }

                ElementCompute got = ElementCompute(dst_tensor.at({n, h, w, 0}));

                bool equal = (c_accum == got);

                EXPECT_TRUE(equal);
                if (!equal) {

                  std::cerr 
                    << "Error at location (" << n << ", " << h << ", " << w << ", 0)" << std::endl;

                  std::cerr 
                    << "  expected: " << c_accum << std::endl
                    << "       got: " << got << std::endl;

                  std::cerr 
                    << "Problem: " << src_tensor.extent() << " -> " 
                    << dst_tensor.extent() << std::endl;

                  std::cerr 
                    << "   Grid: " << reduction.reduction_strided.grid_shape 
                    << "\n   Block: " << reduction.reduction_strided.threadblock_shape << std::endl
                    << "  FInal: " << reduction.reduction_strided.grid_final 
                    << "\n   Block: " << reduction.reduction_strided.threadblock_final << "\n";

                  return false;
                }

              } //w
            } // h
          } // n
          
          //
          // Next problem
          //

        } // C
      } // W
    } // H
  } // N

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_reduce_c_f32x1) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  int const kV = 1;
  
  // Define the functor
  using Functor = cutlass::plus<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_reduce_c_f32x1_f16x1) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = float;
  using ElementSource = cutlass::half_t;
  using ElementCompute = float;
  int const kV = 1;
  
  // Define the functor
  using Functor = cutlass::plus<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_reduce_c_f32x2) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  int const kV = 2;
  
  // Define the functor
  using Functor = cutlass::plus<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_reduce_c_f32x2_f16x2) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = float;
  using ElementSource = cutlass::half_t;
  using ElementCompute = float;
  int const kV = 2;
  
  // Define the functor
  using Functor = cutlass::plus<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_reduce_c_f32x4) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  int const kV = 4;
  
  // Define the functor
  using Functor = cutlass::plus<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_reduce_c_f32x4_f16x4) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = float;
  using ElementSource = cutlass::half_t;
  using ElementCompute = float;
  int const kV = 4;
  
  // Define the functor
  using Functor = cutlass::plus<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_maximum_c_f32x4) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  int const kV = 4;
  
  // Define the functor
  using Functor = cutlass::maximum<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>( -std::numeric_limits<float>::max() ));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_minimum_c_f32x4) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  int const kV = 4;
  
  // Define the functor
  using Functor = cutlass::minimum<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>( std::numeric_limits<float>::max() ));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_ANY_c_s32) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = int;
  using ElementSource = int;
  using ElementCompute = int;
  int const kV = 1;
  
  // Define the functor
  using Functor = cutlass::logical_or<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>( ElementCompute(0) ));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_ALL_c_s32) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = int;
  using ElementSource = int;
  using ElementCompute = int;
  int const kV = 1;
  
  // Define the functor
  using Functor = cutlass::logical_and<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>( ElementCompute(1) ));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_ANY_c_f32) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  int const kV = 1;
  
  // Define the functor
  using Functor = cutlass::logical_or<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>( ElementCompute(0) ));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHW
TEST(Reduction_TensorReduce, nhwc_ALL_c_f32) {

  using Layout = cutlass::layout::TensorNHWC;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  int const kV = 1;
  
  // Define the functor
  using Functor = cutlass::logical_and<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;
  
  EXPECT_TRUE(TestAllReduction_NHWC_reduce_c<TensorReduction>( ElementCompute(1) ));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
