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

/**
In the normal GEMM, the fast changing dimension of a matrix always has stride 
equals to 1, e.g. ColumnMajor and RowMajor matrix.  Affine2 matrix can have 
larger than 1 stride in both dimensions.  To support such layout, we need to 
change to method to visit the global memory:

  1. We can only visit 1 element a time because elements are not stored
     consecutively anymore.  Vectorized load/store is not possible.
  2. One extra multiplication is needed in calculating the global memory
     address
     addr = base_pointer + coord1 * stride1 + coord2 * stride2

The rest part of GEMM which includes shared memory load/store, mma comutation
is the same.

This example uses Ampere fp64 tensore core Affine2 GEMM as an example.  SIMT 
(e.g. sgemm, dgemm) has support Affine2 layout.
*/

#include <iostream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/kernel/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/matrix_coord.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "helper.h"

// The code section below describes datatype for input, output tensors and computation between
// elements 
using ElementAccumulator = double;                 // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // Data type of epilogue computation
using ElementInputA = double;                      // Data type of elements in input tensor
using ElementInputB = double;                      // Data type of elements in input tensor
using ElementOutput = double;                      // Data type of elements in output tensor

// Since Affine2 explicitly lists the strides of both dimensions, it does not really matter if 
// it is columnmajor and rowmajor.  However, it helps CUTLASS to improve the load locality if 
// CUTLASS can know which dimension of A/B operand has smaller stride or more dense.
//
// Affine2 ColumnMajor means the row stride is smaller and Affine2 RowMajor means the column 
// stride is smaller.
//
// The Affine2 epilogue reuses AffineN epilogue so it does not need to specify column majore
// or row major.
using LayoutInputA = cutlass::layout::AffineRank2ColumnMajor;
using LayoutInputB = cutlass::layout::AffineRank2RowMajor;
using LayoutOutput = cutlass::layout::AffineRankN<2>;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;         // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

// Number of pipelines you want to use
constexpr int NumStages = 3;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        // Data type of output matrix.
    1,                                                    // The number of elements per memory
                                                          // access has.  It has to be 1 for
                                                          // affine2. 
    ElementAccumulator,
    ElementComputeEpilogue>;

using Gemm = typename cutlass::gemm::device::GemmUniversal<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  1,
  1,
  cutlass::arch::OpMultiplyAdd
>;

/////////////////////////////////////////////////////////////////////////////////////////////////

int run() {

  // Construct Gemm ProblemSize with user defined output size
  cutlass::gemm::GemmCoord problem_size = {1024, 512, 1024};

  // Stride factor shows the distance between two elements in the different dimensions.  The
  // first data is the logical distance between two rows, the second is between two columns.
  // CUTLASS has a utility tool cutlass::layout::Affine2Layout_Factory<Layout>::layout_factory
  // to help to convert stride_factor to the two strides.
  //
  // It is also totally fine to compute the strides directly without using the utility to 
  // construct the affine2 layout.
  typename LayoutInputA::Stride::Index stride_factor_A[] = {3, 4}; 
  typename LayoutInputB::Stride::Index stride_factor_B[] = {5, 6};
  typename LayoutOutput::Stride::Index stride_factor_C[] = {7, 8};

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk(),
         cutlass::layout::Affine2Layout_Factory<LayoutInputA>::layout_factory(problem_size.mk(),
                                                                              stride_factor_A));
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn(),
         cutlass::layout::Affine2Layout_Factory<LayoutInputB>::layout_factory(problem_size.kn(),
                                                                              stride_factor_B));

  // Create matrix C used to load for bias addition.
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn(),
         cutlass::layout::Affine2Layout_Factory<LayoutOutput>::layout_factory(problem_size.mn(),
                                                                              stride_factor_C));

  // Create matrix D used to store output from CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn(),
         cutlass::layout::Affine2Layout_Factory<LayoutOutput>::layout_factory(problem_size.mn(),
                                                                              stride_factor_C));

  // Create matrix D with dimensions M x N used to store output from reference
  // kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(problem_size.mn(),
         cutlass::layout::Affine2Layout_Factory<LayoutOutput>::layout_factory(problem_size.mn(),
                                                                              stride_factor_C));

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      0);  // <- Fill matrix A on host with uniform-distribution random data

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(4),
      ElementInputB(-4),
      0);  // <- Fill matrix B on host with uniform-distribution random data

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data

  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;

  int batch_count = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
    mode,
    problem_size,
    batch_count,
    {alpha, beta},
    tensor_a.device_ref().data(),              // <- reference to matrix A on device
    tensor_b.device_ref().data(),              // <- reference to matrix B on device
    tensor_c.device_ref().data(),              // <- reference to matrix C on device
    tensor_d.device_ref().data(),              // <- reference to matrix D on device
    tensor_a.layout().capacity(problem_size.mk()),
    tensor_b.layout().capacity(problem_size.kn()),
    tensor_c.layout().capacity(problem_size.mn()),
    tensor_d.layout().capacity(problem_size.mn()),
    tensor_a.layout().stride(),
    tensor_b.layout().stride(),
    tensor_c.layout().stride(),
    tensor_d.layout().stride()
  };                    

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op();

  CUTLASS_CHECK(status);

  //
  // Create instantiation for device reference gemm kernel
  //

  // Launch device reference to compute strictly the product A * B
  cutlass::reference::device::Gemm<
      ElementInputA, 
      LayoutInputA, 
      ElementInputB, 
      LayoutInputB, 
      ElementOutput,
      LayoutOutput, 
      ElementComputeEpilogue, 
      ElementAccumulator> gemm_device;

  gemm_device
    (
      problem_size,
      alpha, 
      tensor_a.device_ref(),
      tensor_b.device_ref(),
      beta, 
      tensor_c.device_ref(), 
      tensor_ref_d.device_ref()
    );

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  bool pass = cutlass::reference::host::TensorEquals(tensor_d.host_view(),
                                                     tensor_ref_d.host_view());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  std::cout << (pass
                    ? "Passed"
                    : "Failed")
            << std::endl;

  CUTLASS_CHECK(status);

  return (pass ? 0 : -1);
}

int main(int argc, char const **args) {

  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 11.0.
  //
  // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv2dFprop examples.
  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (!(props.major >= 8)) {
    std::cerr << "Ampere Tensor Ops must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    return 0;
  }

  return run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
