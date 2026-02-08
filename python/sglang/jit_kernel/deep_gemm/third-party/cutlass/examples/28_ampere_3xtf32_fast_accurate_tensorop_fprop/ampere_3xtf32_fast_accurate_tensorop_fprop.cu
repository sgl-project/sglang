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

This example adopts example 16 to use 3xTF32 to bring FP32 accuracy with 2x performance
compared with CUDA Cores.  See example 27 for the trick of 3xTF32. 
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output tensors and computation between
// elements 
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = float;                       // Data type of elements in input tensor
using ElementInputB = float;                       // Data type of elements in input tensor
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;         // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 3;

// This code section describe iterator algorithm selected is Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized.
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

// 3xTF32 Fprop
using Conv2dFpropKernel_3xTF32 = typename cutlass::conv::kernel::DefaultConv2dFprop<
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
  // Only thing needs to be changed from normal Fprop
  cutlass::arch::OpMultiplyAddFastF32,
  IteratorAlgorithm
>::Kernel;

// 1xTF32 Fprop
using Conv2dFpropKernel_1xTF32 = typename cutlass::conv::kernel::DefaultConv2dFprop<
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

using ImplicitGemm_3xTF32 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel_3xTF32>;
using ImplicitGemm_1xTF32 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel_1xTF32>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  int iterations;
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    iterations(20),
    save_workspace(false),
    alpha(1),
    beta(0),
    benchmark(false) { }

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of cutlass::half_t (F16) elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 8 elements.
    //
    int const kAlignment = 4;

    if ((input_size.c() % kAlignment) ||
      (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) ||
      (padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    cutlass::Tensor4DCoord input_size,
    cutlass::Tensor4DCoord filter_size) {

    this->input_size = input_size;
    this->filter_size = filter_size;

    padding.n() = filter_size.h() / 2;
    padding.h() = filter_size.h() / 2;
    padding.w() = filter_size.w() / 2;
    padding.c() = filter_size.w() / 2;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.c() = input_size.c(); 

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    if (filter_size.h() == 3 && filter_size.w() == 3) {
      padding = {1, 1, 1, 1};
    }
    else {
      filter_size.h() = 1;
      filter_size.w() = 1;
      padding = {0, 0, 0, 0};
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "28_ampere_3xtf32_fast_accurate_tensorop_fprop example\n\n"
      << "  This example uses Ampere's Tensor Core operators on F16 data types to compute\n"
      << "  forward convolution on tensors of layout NHWC.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n=<int>            Input tensor extent N\n"
      << "  --h=<int>            Input tensor extent H\n"
      << "  --w=<int>            Input tensor extent W\n"
      << "  --c=<int>            Input tensor extent C\n"
      << "  --k=<int>            Filter extent K\n"
      << "  --r=<int>            Filter extent R\n"
      << "  --s=<int>            Filter extent S\n\n"
      << "  --alpha=<float>      Epilogue scalar alpha\n"
      << "  --beta=<float>       Epilogue scalar beta\n\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/28_ampere_3xtf32_fast_accurate_tensorop_fprop/28_ampere_3xtf32_fast_accurate_tensorop_fprop  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
      << "$ ./examples/28_ampere_3xtf32_fast_accurate_tensorop_fprop/28_ampere_3xtf32_fast_accurate_tensorop_fprop  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

    return out;
  }
  
  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    return cutlass::Tensor4DCoord(
      input_size.n(),
      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
      filter_size.n());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;

  double l2_norm_3xtf32_vs_fp64;
  double l2_norm_1xtf32_vs_fp64;
  double l2_norm_fp32_vs_fp64;

  Result(): 
    runtime_ms(0), 
    gflops(0),
    status(cutlass::Status::kSuccess),
    error(cudaSuccess),
    l2_norm_3xtf32_vs_fp64(0),
    l2_norm_1xtf32_vs_fp64(0),
    l2_norm_fp32_vs_fp64(0) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,H,W,C,K,R,S,Runtime,GFLOPs,3xTF32_vs_FP64,1xTF32_vs_FP64,FP32_vs_FP64";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out 
      << "conv_" << idx << ","
      << options.input_size.n() << ","
      << options.input_size.h() << ","
      << options.input_size.w() << ","
      << options.input_size.c() << ","
      << options.filter_size.n() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << runtime_ms << ","
      << gflops << ","
      << l2_norm_3xtf32_vs_fp64 << ","
      << l2_norm_1xtf32_vs_fp64 << ","
      << l2_norm_fp32_vs_fp64;

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs one benchmark
Result profile_convolution(Options const &options) {

  Result result;

  ////////////////////////////////////////////////////////////////////////////////
  /// 1. Initialize F32 Precision input tensors using CUTLASS helper functions
  ////////////////////////////////////////////////////////////////////////////////

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //

  cutlass::HostTensor<float, LayoutInputA> tensor_a_F32(options.input_size);
  cutlass::HostTensor<float, LayoutInputB> tensor_b_F32(options.filter_size);
  cutlass::HostTensor<float, LayoutOutput> tensor_c_F32(options.output_size());
  cutlass::HostTensor<float, LayoutOutput> tensor_d_F32(options.output_size());

  //
  // Initialize tensors
  //

  // Fill tensor A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a_F32.host_view(),
      1,
      ElementInputA(7),
      ElementInputA(-8));

  // Fill tensor B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b_F32.host_view(),
      1,
      ElementInputB(7),
      ElementInputB(-8));

  // Fill tensor C on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c_F32.host_view(),
      1,
      ElementInputB(7),
      ElementInputB(-8));

  // Fill tensor D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_d_F32.host_view());

  // Copy data from host to GPU
  tensor_a_F32.sync_device();
  tensor_b_F32.sync_device();
  tensor_c_F32.sync_device();
  tensor_d_F32.sync_device();

  ////////////////////////////////////////////////////////////////////////////////
  /// 2. Initialize F32 Precision input tensors using CUTLASS helper functions
  ////////////////////////////////////////////////////////////////////////////////

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //

  cutlass::HostTensor<double, LayoutInputA> tensor_a_F64(options.input_size);
  cutlass::HostTensor<double, LayoutInputB> tensor_b_F64(options.filter_size);
  cutlass::HostTensor<double, LayoutOutput> tensor_c_F64(options.output_size());

  cutlass::HostTensor<double, LayoutOutput> tensor_d_F64(options.output_size());
  cutlass::HostTensor<float, LayoutOutput> tensor_d_3xTF32(options.output_size());
  cutlass::HostTensor<float, LayoutOutput> tensor_d_1xTF32(options.output_size());

  // Copy values from the DP tensors
  cutlass::reference::host::TensorCopy(tensor_a_F64.host_view(), tensor_a_F32.host_view());
  cutlass::reference::host::TensorCopy(tensor_b_F64.host_view(), tensor_b_F32.host_view());
  cutlass::reference::host::TensorCopy(tensor_c_F64.host_view(), tensor_c_F32.host_view());
  cutlass::reference::host::TensorCopy(tensor_d_F64.host_view(), tensor_d_F32.host_view());
  cutlass::reference::host::TensorCopy(tensor_d_3xTF32.host_view(), tensor_d_F32.host_view());
  cutlass::reference::host::TensorCopy(tensor_d_1xTF32.host_view(), tensor_d_F32.host_view());
 
  // Copy data from host to GPU
  tensor_a_F64.sync_device();
  tensor_b_F64.sync_device();
  tensor_c_F64.sync_device();
  tensor_d_F64.sync_device();
  tensor_d_3xTF32.sync_device();
  tensor_d_1xTF32.sync_device();

  //
  // Define arguments for CUTLASS Convolution
  //

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Construct Conv2dProblemSize with user defined output size
  cutlass::conv::Conv2dProblemSize problem_size(      
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.output_size(),
      mode,
      split_k_slices
  );

  ////////////////////////////////////////////////////////////////////////////////
  /// 3. Run  3xTF32 kernel within a profiling loop
  ////////////////////////////////////////////////////////////////////////////////

  // Construct ImplicitGemm::Argument structure with conv2d 
  // problem size, data pointers, and epilogue values
  typename ImplicitGemm_3xTF32::Arguments arguments_3xTF32{
    problem_size,
    tensor_a_F32.device_ref(),
    tensor_b_F32.device_ref(),
    tensor_c_F32.device_ref(),
    tensor_d_3xTF32.device_ref(),
    {options.alpha, options.beta},
  };

  //
  // Initialize CUTLASS Convolution
  //

  ImplicitGemm_3xTF32 implicit_gemm_op_3xTF32;

  size_t workspace_size_3xTF32 = implicit_gemm_op_3xTF32.get_workspace_size(arguments_3xTF32);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace_3xTF32(workspace_size_3xTF32);

  result.status = implicit_gemm_op_3xTF32.can_implement(arguments_3xTF32);
  CUTLASS_CHECK(result.status);

  result.status = implicit_gemm_op_3xTF32.initialize(arguments_3xTF32, workspace_3xTF32.get());
  CUTLASS_CHECK(result.status);

  //
  // Launch initialized CUTLASS kernel
  //
  result.status = implicit_gemm_op_3xTF32();

  CUTLASS_CHECK(result.status);
  
  //
  // Performance measurement
  //

  cudaEvent_t events[2];
  
  for (auto & event : events) {
    result.error = cudaEventCreate(&event);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }
  }

  // Record an event at the start of a series of convolution operations.
  result.error = cudaEventRecord(events[0]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
  }

  // Launch a sequence of implicit GEMM operations on the device
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    result.status = implicit_gemm_op_3xTF32();
    CUTLASS_CHECK(result.status);
  }

  // Record an event when the convolutions have been launched.
  result.error = cudaEventRecord(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
  }

  // Wait for work on the device to complete.
  result.error = cudaEventSynchronize(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
  }

  // Measure elapsed runtime
  float runtime_ms = 0;
  result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
    return result;
  }

  // Print average runtime and GFLOPs.
  result.runtime_ms = double(runtime_ms) / double(options.iterations);
  result.gflops = options.gflops(result.runtime_ms / 1000.0);

  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }

  tensor_d_3xTF32.sync_host();

  ////////////////////////////////////////////////////////////////////////////////
  /// 4. Run 1xTF32 kernel within a profiling loop
  ////////////////////////////////////////////////////////////////////////////////

  // Construct ImplicitGemm::Argument structure with conv2d 
  // problem size, data pointers, and epilogue values
  typename ImplicitGemm_1xTF32::Arguments arguments_1xTF32{
    problem_size,
    tensor_a_F32.device_ref(),
    tensor_b_F32.device_ref(),
    tensor_c_F32.device_ref(),
    tensor_d_1xTF32.device_ref(),
    {options.alpha, options.beta},
  };

  //
  // Initialize CUTLASS Convolution
  //

  ImplicitGemm_1xTF32 implicit_gemm_op_1xTF32;

  size_t workspace_size_1xTF32 = implicit_gemm_op_1xTF32.get_workspace_size(arguments_1xTF32);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace_1xTF32(workspace_size_1xTF32);

  result.status = implicit_gemm_op_1xTF32.can_implement(arguments_1xTF32);
  CUTLASS_CHECK(result.status);

  result.status = implicit_gemm_op_1xTF32.initialize(arguments_1xTF32, workspace_1xTF32.get());
  CUTLASS_CHECK(result.status);

  //
  // Launch initialized CUTLASS kernel
  //
  result.status = implicit_gemm_op_1xTF32();

  CUTLASS_CHECK(result.status);
  
  tensor_d_1xTF32.sync_host();

  ////////////////////////////////////////////////////////////////////////////////
  // Run reference kernel (F64)
  ////////////////////////////////////////////////////////////////////////////////

  cutlass::reference::device::Conv2d<
    double,
    LayoutInputA,
    double,
    LayoutInputB,
    double,
    LayoutOutput,
    double,
    double 
  >(
      cutlass::conv::Operator::kFprop,
      problem_size,
      tensor_a_F64.device_ref(),
      tensor_b_F64.device_ref(),
      tensor_c_F64.device_ref(),
      tensor_d_F64.device_ref(),
      options.alpha, 
      options.beta);

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d_F64.sync_host();

  ////////////////////////////////////////////////////////////////////////////////
  // Run reference kernel (F32)
  ////////////////////////////////////////////////////////////////////////////////

  cutlass::reference::device::Conv2d<
    float,
    LayoutInputA,
    float,
    LayoutInputB,
    float,
    LayoutOutput,
    float,
    float 
  >(
      cutlass::conv::Operator::kFprop,
      problem_size,
      tensor_a_F32.device_ref(),
      tensor_b_F32.device_ref(),
      tensor_c_F32.device_ref(),
      tensor_d_F32.device_ref(),
      options.alpha, 
      options.beta);

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d_F32.sync_host();

  ////////////////////////////////////////////////////////////////////////////////
  ///////               Compute l2 norms 
  ////////////////////////////////////////////////////////////////////////////////

  // l2 norm 3xTF32 vs F64
  cutlass::HostTensor<double, LayoutOutput> tensor_d_3xTF32_in_F64(options.output_size());
  cutlass::reference::host::TensorCopy(tensor_d_3xTF32_in_F64.host_view(), tensor_d_3xTF32.host_view());

  result.l2_norm_3xtf32_vs_fp64 = cutlass::reference::host::TensorRelativeErrorMetric(
    tensor_d_3xTF32_in_F64.host_view(), tensor_d_F64.host_view());

  // l2 norm 1xTF32 vs F64
  cutlass::HostTensor<double, LayoutOutput> tensor_d_1xTF32_in_F64(options.output_size());
  cutlass::reference::host::TensorCopy(tensor_d_1xTF32_in_F64.host_view(), tensor_d_1xTF32.host_view());

  result.l2_norm_1xtf32_vs_fp64 = cutlass::reference::host::TensorRelativeErrorMetric(
    tensor_d_1xTF32_in_F64.host_view(), tensor_d_F64.host_view());

  // l2 norm F32 vs F64
  cutlass::HostTensor<double, LayoutOutput> tensor_d_F32_in_F64(options.output_size());
  cutlass::reference::host::TensorCopy(tensor_d_F32_in_F64.host_view(), tensor_d_F32.host_view());

  result.l2_norm_fp32_vs_fp64 = cutlass::reference::host::TensorRelativeErrorMetric(
    tensor_d_F32_in_F64.host_view(), tensor_d_F64.host_view());

  ///////////////////////////////////////////////////////////////////////////////

  if (options.save_workspace) {

    std::stringstream ss;

    ss << "28_ampere_3xtf32_fast_accurate_tensorop_fprop_"
      << options.input_size.n() << "x" << options.input_size.h() << "x" << options.input_size.w() << "x" << options.input_size.c() 
      << "_"
      << options.filter_size.n() << "x" << options.filter_size.h() << "x" << options.filter_size.w() << "x" << options.filter_size.c() 
      << ".dat";

    std::ofstream output_workspace(ss.str());

    output_workspace 
      << "Input = \n" << tensor_a_F32.host_view() << "\n\n"
      << "Filters = \n" << tensor_b_F32.host_view() << "\n\n";

    output_workspace << "TF32x3 = \n" << tensor_d_3xTF32.host_view() << std::endl;
    output_workspace << "TF32x1 = \n" << tensor_d_1xTF32.host_view() << std::endl;
    output_workspace << "FP32 = \n" << tensor_d_F32.host_view() << std::endl;
    output_workspace << "FP64 = \n" << tensor_d_F64.host_view() << "\n\n";

    std::cout << "Results written to '" << ss.str() << "'." << std::endl;
  }

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

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

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.benchmark) {
    // Benchmark several layers

    int batch_sizes[] = {1, 32, 64, 128, 256};

    struct Benchmark {
      int h, w, c, k, r, s;
    } layers[] = {
      {56,  56,   64,   256, 1, 1},
      {56,  56,   64,    64, 1, 1},
      {56,  56,   64,    64, 3, 3},
      {56,  56,  256,    64, 1, 1},
      {56,  56,  256,   512, 1, 1},
      {56,  56,  256,   128, 1, 1},
      {28,  28,  128,   128, 3, 3},
      {28,  28,  128,   512, 1, 1},
      {28,  28,  512,   128, 1, 1},
      {28,  28,  512,  1024, 1, 1},
      {28,  28,  512,   256, 1, 1},
      {14,  14,  256,   256, 3, 3},
      {14,  14,  256,  1024, 1, 1},
      {14,  14,  1024,  256, 1, 1},
      {14,  14,  1024, 2048, 1, 1},
      {14,  14,  1024,  512, 1, 1},
      {7,    7,   512,  512, 3, 3},
    };

    Result::print_header(std::cout, options) << std::endl;

    int idx = 1;

    for (auto const &layer : layers) {
      for (auto N : batch_sizes) {

        options.update({N, layer.h, layer.w, layer.c}, {layer.k, layer.r, layer.s, layer.c});

        Result result = profile_convolution(options);
        result.print(std::cout, idx, options) << std::endl;
      }

      ++idx;
    }
  }
  else {

    // Execute one problem size
    if (!options.valid()) {
      std::cerr << "Invalid problem." << std::endl;
      return -1;
    }

    Result result = profile_convolution(options);

    Result::print_header(std::cout, options) << std::endl;
    result.print(std::cout, 1, options) << std::endl;
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
