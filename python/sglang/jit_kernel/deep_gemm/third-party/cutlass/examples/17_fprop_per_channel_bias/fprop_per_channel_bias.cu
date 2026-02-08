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
The convolution version of 12_gemm_bias_relu.  Similarly, we put bias vector in Operand C and the
rest is the same as normal convolution.
*/

#include <iostream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

// The code section below describes datatype for input, output tensors and computation between
// elements 
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // Data type of epilogue computation
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;         // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 4;

// This code section describe iterator algorithm selected is Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,                                        // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // The number of elements per vectorized.
                                                          // memory access. This becomes the vector width of
                                                          // math instructions in the epilogue too.
    ElementAccumulator,                                   // Data type of accumulator
    ElementComputeEpilogue,                               // Data type for alpha in linear combination
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // alpha X C + per channel bias


using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
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
  cutlass::arch::OpMultiplyAdd,
  IteratorAlgorithm
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

int run() {

  // Construct Conv2dProblemSize with user defined output size
  cutlass::conv::Conv2dProblemSize problem_size(      
    {1, 7, 7, 512},                               // activation 
    {512, 3, 3, 512},                             // filter
    {1, 1, 1, 1},                                 // padding
    {1, 1},                                       // striding
    {1, 1},                                       // dilation
    cutlass::conv::Mode::kCrossCorrelation,       // mode (convolution or cross-correlation)
    1                                             // split-k slices
  );

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.activation_extent());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.filter_extent());

  // Create tensor C with dimensions 1x1x1xk which is the bias vector
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c_bias({1, 1, 1, problem_size.K});

  // Create tensor D used to store output from CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.output_extent());
  // Create matrix D with dimensions M x N used to store output from reference
  // kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(problem_size.output_extent());

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      0);  // <- Fill tensor A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(4),
      ElementInputB(-4),
      0);  // <- Fill tensor B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c_bias.host_view(),
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
  tensor_c_bias.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),              // <- reference to tensor A on device
    tensor_b.device_ref(),              // <- reference to tensor B on device
    // tensor C  is treated as the bias vector. We can enable the CONV
    // to project away the N, H, W dimension by setting the stride to zero.
    {tensor_c_bias.device_data(), LayoutOutput::Stride(0)},
    tensor_d.device_ref(),              // <- reference to tensor D on device
    {alpha} };                    

  // Instantiate CUTLASS kernel depending on templates
  ImplicitGemm implicit_gemm_op;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = implicit_gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = implicit_gemm_op();

  CUTLASS_CHECK(status);

  //
  // Create instantiation for device reference conv kernel
  //

  // Launch device reference to compute strictly the product A * B
  cutlass::reference::device::Conv2d<
      ElementInputA, 
      LayoutInputA, 
      ElementInputB, 
      LayoutInputB, 
      ElementOutput,
      LayoutOutput, 
      ElementComputeEpilogue, 
      ElementAccumulator,
      cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>>
    (
      cutlass::conv::Operator::kFprop, 
      problem_size, 
      tensor_a.device_ref(),
      tensor_b.device_ref(), 
      tensor_c_bias.device_ref(), 
      tensor_ref_d.device_ref(),
      alpha, ElementComputeEpilogue(0)
    );

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Compute bias + relu in host code
  for (int n = 0; n < problem_size.N; ++n) {
    for (int p = 0; p < problem_size.P; ++p) {
      for (int q = 0; q < problem_size.Q; ++q) {
        for (int k = 0; k < problem_size.K; ++k) {
          
          tensor_ref_d.at({n, p, q, k}) =
              std::max(ElementOutput(0),
                       ElementOutput(tensor_ref_d.at({n, p, q, k}) +
                                     tensor_c_bias.at({0, 0, 0, k})));
        }
      }
    }
  }

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  std::cout << (cutlass::reference::host::TensorEquals(tensor_d.host_view(),
                                                       tensor_ref_d.host_view())
                    ? "Passed"
                    : "Failed")
            << std::endl;

  CUTLASS_CHECK(status);
  return 0;
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
