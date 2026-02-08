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
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/reduction/kernel/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace reduction {

template <typename ReductionKernel>
__global__ void kernel_reduce_splitk(typename ReductionKernel::Params params) {

  __shared__ typename ReductionKernel::SharedStorage shared_storage;

  ReductionKernel reduction_op;

  reduction_op(params, shared_storage);
}

template <typename ReductionKernel>
class ReduceSplitKTestbed {
public:

  using ElementAccumulator = typename ReductionKernel::ElementAccumulator;
  using ElementWorkspace = typename ReductionKernel::ElementWorkspace;
  using ElementOutput = typename ReductionKernel::ElementOutput;
  using Layout = cutlass::layout::RowMajor;

public:

  cutlass::Distribution::Kind distribution_workspace;
  cutlass::Distribution::Kind distribution_source;
  uint64_t seed;

public:

  /// Ctor
  ReduceSplitKTestbed(
    cutlass::Distribution::Kind distribution_workspace = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind distribution_source = cutlass::Distribution::Uniform,
    uint64_t seed = 2019
  ):
    distribution_workspace(distribution_workspace),
    distribution_source(distribution_source),
    seed(seed) {

  }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(cutlass::TensorView<Element, Layout> view,
                         cutlass::Distribution::Kind dist_kind, 
                         uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {
      cutlass::reference::host::TensorFillRandomUniform(view, seed, 8, -8, 0);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {
      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5, -1);
    } else if (dist_kind == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(view);
    } else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(),
                                                    view.capacity());
    } else {
      EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
  }

  /// Runs a single problem size
  bool run(
    cutlass::MatrixCoord problem_size, 
    int partitions, 
    ElementAccumulator alpha = 1, 
    ElementAccumulator beta = 0) {

    cutlass::HostTensor<ElementWorkspace, Layout> workspace({
      problem_size.row() * partitions, 
      problem_size.column()
    });

    cutlass::HostTensor<ElementOutput, Layout> source(problem_size);
    cutlass::HostTensor<ElementOutput, Layout> destination(problem_size);
    cutlass::HostTensor<ElementOutput, Layout> destination_reference(problem_size, false);

    //
    // Initialize
    //
    initialize_tensor(workspace.host_view(), distribution_workspace, seed);
    initialize_tensor(source.host_view(), distribution_source, seed + 23);

    cutlass::reference::host::TensorFill(destination.host_view());

    workspace.sync_device();
    source.sync_device();
    destination.sync_device();

    //
    // Launch reduction kernel
    //

    dim3 block = ReductionKernel::block_shape();
    dim3 grid = ReductionKernel::grid_shape(problem_size);

    typename ReductionKernel::Params params(
      problem_size,
      partitions,
      problem_size.row() * problem_size.column(),
      workspace.device_ref(),
      destination.device_ref(),
      source.device_ref(),
      {alpha, beta}
    );

    test::reduction::kernel_reduce_splitk<ReductionKernel><<< grid, block >>>(params);

    cudaError_t result = cudaDeviceSynchronize();

    EXPECT_EQ(result, cudaSuccess)
      << "CUDA error: " << cudaGetErrorString(result);

    destination.sync_host();

    //
    // Compute reference
    //

    for (int m = 0; m < problem_size.row(); ++m) {
      for (int n = 0; n < problem_size.column(); ++n) {

        ElementAccumulator accum = 0;

        for (int k = 0; k < partitions; ++k) {
          accum += ElementAccumulator(workspace.at({m + k * problem_size.row(), n}));
        }

        ElementAccumulator c = ElementAccumulator(source.at({m, n}));

        destination_reference.at({m, n}) = ElementOutput(accum * alpha + beta * c);
      }
    }

    //
    // Compare
    //

    EXPECT_GT(cutlass::reference::host::TensorNorm(destination.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(destination_reference.host_view()), 0);

    bool passed = cutlass::reference::host::TensorEquals(
        destination.host_view(), destination_reference.host_view());

    EXPECT_TRUE(passed)
      << "Workspace =\n" << workspace.host_view() << "\n\n"
      << "\n"
      << "Reference =\n" << destination_reference.host_view() << "\n\n"
      << "Computed =\n" << destination.host_view() << "\n";

    return passed;
  }

  /// Runs through a variety of test cases
  bool run_all() {

    cutlass::MatrixCoord problem_sizes[] = {
      {8, 8},
      {136, 72},
      {248, 232},
    };

    int partition_counts[] = {
      1,3,4,5,11
    };

    bool passed = false;

    for (cutlass::MatrixCoord problem : problem_sizes) {
      for (int partitions : partition_counts) {
        passed = run(problem, partitions);
        if (!passed) {
          return false;
        }
      }
    }

    return passed;
  }
};

} // namespace reduction
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Strictly F32 data
//
TEST(Reduction_ReduceSplitK, f32_f32_f32_1_1x32) {

  using ElementWorkspace = float;
  using ElementAccumulator = float;
  using ElementOutput = float;
  int const kN = 1;
  using Shape = cutlass::MatrixShape<1, 32>;

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kN,
    ElementAccumulator,
    ElementAccumulator
  >;

  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator, 
    ElementWorkspace,
    kN
  >;

  using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
    Shape,
    OutputOp,
    ReductionOp
  >;

  test::reduction::ReduceSplitKTestbed<ReductionKernel> testbed;

  EXPECT_TRUE(testbed.run_all());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Vectorized access
//
TEST(Reduction_ReduceSplitK, f32_f32_f32_2_4x64) {

  using ElementWorkspace = float;
  using ElementAccumulator = float;
  using ElementOutput = float;
  int const kN = 2;
  using Shape = cutlass::MatrixShape<4, 64>;

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kN,
    ElementAccumulator,
    ElementAccumulator
  >;

  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator, 
    ElementWorkspace,
    kN
  >;

  using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
    Shape,
    OutputOp,
    ReductionOp
  >;

  test::reduction::ReduceSplitKTestbed<ReductionKernel> testbed;

  EXPECT_TRUE(testbed.run_all());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Vectorized access
//
TEST(Reduction_ReduceSplitK, f32_f32_f16_2_4x64) {

  using ElementWorkspace = float;
  using ElementAccumulator = float;
  using ElementOutput = cutlass::half_t;
  int const kN = 2;
  using Shape = cutlass::MatrixShape<4, 64>;

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kN,
    ElementAccumulator,
    ElementAccumulator
  >;

  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator, 
    ElementWorkspace,
    kN
  >;

  using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
    Shape,
    OutputOp,
    ReductionOp
  >;

  test::reduction::ReduceSplitKTestbed<ReductionKernel> testbed;

  EXPECT_TRUE(testbed.run_all());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Vectorized access
//
TEST(Reduction_ReduceSplitK, f32_f32_f16_8_4x64) {

  using ElementWorkspace = float;
  using ElementAccumulator = float;
  using ElementOutput = cutlass::half_t;
  int const kN = 8;
  using Shape = cutlass::MatrixShape<4, 64>;

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kN,
    ElementAccumulator,
    ElementAccumulator
  >;

  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator, 
    ElementWorkspace,
    kN
  >;

  using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
    Shape,
    OutputOp,
    ReductionOp
  >;

  test::reduction::ReduceSplitKTestbed<ReductionKernel> testbed;

  EXPECT_TRUE(testbed.run_all());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

