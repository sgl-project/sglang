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
#include <limits>

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

/// This reduces the W dimension, transforming an NHWC tensor into NHWC with W=1.
template <
  typename TensorReduction, 
  typename ElementCompute = typename TensorReduction::ElementCompute
>
bool TestAllReduction_NHWC_reduce_w(ElementCompute reduction_identity = ElementCompute()) {

  using Layout = typename TensorReduction::Layout;
  using ElementOutput = typename TensorReduction::ElementOutput;
  using ElementSource = typename TensorReduction::ElementSource;

  int const kV = TensorReduction::kVectorLength;

  int const N_indices[] = {1, 2, 5, 10};
  int const H_indices[] = {1, 3, 9 };
  int const W_indices[] = {1, 5, 19, 40, 224};
  int const C_indices[] = {
    kV, 
    2 * kV, 
    5 * kV, 
    9 * kV, 
    17 * kV, 
    39 * kV, 
    257 * kV, 
    kV * 760
  };

  using Element = int;

  for (int N : N_indices) {
    for (int H : H_indices) {
      for (int W : W_indices) {
        for (int C : C_indices) {

          cutlass::HostTensor<ElementSource, Layout> src_tensor({N, H, W, C});
          cutlass::HostTensor<ElementOutput, Layout> dst_tensor({N, H, 1, C});

          cutlass::reference::host::TensorFillRandomUniform(
            src_tensor.host_view(), 17, 10, -10, 0);

          cutlass::reference::host::BlockFillSequential(
            dst_tensor.host_data(), dst_tensor.capacity());

          dst_tensor.sync_device();
          src_tensor.sync_device();

          // Execute a tensor reduction over rank 2 (the 'W' dimension is reduced; NHWC => NHC)
          TensorReduction reduction(src_tensor.extent(), 2);

          cutlass::DeviceAllocation<uint8_t> device_workspace(reduction.workspace_size());

          cutlass::Status status = reduction.reduce(
            dst_tensor.device_ref(),
            src_tensor.device_ref(),
            device_workspace.get(),
            reduction_identity
          );

          EXPECT_EQ(status, cutlass::Status::kSuccess);
          EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
          // Reference check
          dst_tensor.sync_host();

          typename TensorReduction::ReductionOp reduction_op;

          for (int n = 0; n < src_tensor.extent().n(); ++n) {
            for (int h = 0; h < src_tensor.extent().h(); ++h) {
              for (int c = 0; c < src_tensor.extent().c(); ++c) {

                ElementCompute w_accum = reduction_identity;

                for (int w = 0; w < src_tensor.extent().w(); ++w) {
                  w_accum = reduction_op(w_accum, ElementCompute(src_tensor.at({n, h, w, c})));
                }

                ElementCompute got = ElementCompute(dst_tensor.at({n, h, 0, c}));

                bool equal = (w_accum == got);

                EXPECT_TRUE(equal);
                if (!equal) {

                  std::cerr 
                    << "Error at location (" << n << ", " << h << ", 0, " << c << ")" << std::endl;

                  std::cerr 
                    << "  expected: " << w_accum << std::endl
                    << "       got: " << got << std::endl;

                  std::cerr 
                    << "Problem: " << src_tensor.extent() << " -> " 
                    << dst_tensor.extent() << std::endl;

                  std::cerr 
                    << "   Grid: " << reduction.reduction_strided.grid_shape 
                    << "\n  Block: " << reduction.reduction_strided.threadblock_shape << std::endl
                    << "  Final: " << reduction.reduction_strided.grid_final 
                    << "\n  Block: " << reduction.reduction_strided.threadblock_final << "\n";

                  return false;
                }
              }
            }
          }
        }
      }
    }
  }

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_reduce_w_f32x8_f16x8) {

  int const kV = 8;
  using ElementOutput = float;
  using ElementSource = cutlass::half_t;
  using ElementCompute = float;
  using Layout = cutlass::layout::TensorNHWC;

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

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>());          
}

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_reduce_w_f32x2_f16x2) {

  int const kV = 2;
  using ElementOutput = float;
  using ElementSource = cutlass::half_t;
  using ElementCompute = float;
  using Layout = cutlass::layout::TensorNHWC;

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

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>());          
}

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_reduce_w_f32x1_f16x1) {

  int const kV = 1;
  using ElementOutput = float;
  using ElementSource = cutlass::half_t;
  using ElementCompute = float;
  using Layout = cutlass::layout::TensorNHWC;

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

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>());          
}

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_reduce_w_s32x4) {

  int const kV = 4;
  using Element = int;
  using Layout = cutlass::layout::TensorNHWC;

  // Define the functor
  using Functor = cutlass::plus<Element>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    Element,
    Element,
    Layout,
    Functor,
    kV,
    Element
  >;

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>());          
}

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_reduce_w_cf32) {

  int const kV = 1;
  using ElementOutput = cutlass::complex<float>;
  using ElementSource = cutlass::complex<float>;
  using ElementCompute = cutlass::complex<float>;
  using Layout = cutlass::layout::TensorNHWC;

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

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>());          
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_maximum_w_cf32) {

  int const kV = 1;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  using Layout = cutlass::layout::TensorNHWC;

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

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>( -std::numeric_limits<float>::max() ));          
}

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_minimum_w_cf32) {

  int const kV = 1;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  using Layout = cutlass::layout::TensorNHWC;

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

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>(std::numeric_limits<float>::max()));          
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_XOR_w_u32) {

  int const kV = 1;
  using ElementOutput = int;
  using ElementSource = int;
  using ElementCompute = int;
  using Layout = cutlass::layout::TensorNHWC;

  // Define the functor
  using Functor = cutlass::bit_xor<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>());          
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_AND_w_s32) {

  int const kV = 1;
  using ElementOutput = unsigned;
  using ElementSource = unsigned;
  using ElementCompute = unsigned;
  using Layout = cutlass::layout::TensorNHWC;

  // Define the functor
  using Functor = cutlass::bit_and<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>(0xffffffff));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_OR_w_u32) {

  int const kV = 1;
  using ElementOutput = int;
  using ElementSource = int;
  using ElementCompute = int;
  using Layout = cutlass::layout::TensorNHWC;

  // Define the functor
  using Functor = cutlass::bit_or<ElementCompute>;

  using TensorReduction = cutlass::reduction::device::TensorReduction<
    ElementOutput,
    ElementSource,
    Layout,
    Functor,
    kV,
    ElementCompute
  >;

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>());          
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_ANY_w_s32) {

  int const kV = 1;
  using ElementOutput = int;
  using ElementSource = int;
  using ElementCompute = int;
  using Layout = cutlass::layout::TensorNHWC;

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

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>(ElementCompute(0)));          
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_ALL_w_s32) {

  int const kV = 1;
  using ElementOutput = int;
  using ElementSource = int;
  using ElementCompute = int;
  using Layout = cutlass::layout::TensorNHWC;

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

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>(ElementCompute(1)));          
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_ANY_w_f32) {

  int const kV = 1;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  using Layout = cutlass::layout::TensorNHWC;

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

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>(ElementCompute(0)));          
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test tensor reduction from NHWC to NHC
TEST(Reduction_TensorReduce, nhwc_ALL_w_f32) {

  int const kV = 1;
  using ElementOutput = float;
  using ElementSource = float;
  using ElementCompute = float;
  using Layout = cutlass::layout::TensorNHWC;

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

  EXPECT_TRUE(TestAllReduction_NHWC_reduce_w<TensorReduction>(ElementCompute(1)));          
}

/////////////////////////////////////////////////////////////////////////////////////////////////
