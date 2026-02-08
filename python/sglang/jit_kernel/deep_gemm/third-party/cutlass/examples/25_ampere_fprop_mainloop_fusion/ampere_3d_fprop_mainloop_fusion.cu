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

This example shows how to fuse per channel scale+bias+relu of the activations 
into the 3D fprop mainloop.

Compared with original 3D fprop kernel, this example has two more vectors, one for
the scale and one for the bias.  The length of the vectors is the same as the
activation channel number.  This kernel loads the vectors when the associated
activation channels are loaded in the mainloop.  Between reading the 
activations and scale/bias data from the shared memory and calling tensor core
instructions, scale+bias+relu is computed in the register file.

This example is customized for Ampere 16816 fp16 tensor core instruction.
Changing to different data types or different tensor core instruction require
source code changing.  See
include/cutlass/conv/threadblock/implicit_gemm_fprop_fusion_multistage.h for more
technical details.

This example is modified based on 25_ampere_fprop_mainloop_fusion.  The command
line is the same.
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv3d_fprop_fusion.h"
#include "cutlass/conv/device/implicit_gemm_convolution_fusion.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
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
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputScaleBias = cutlass::half_t;     // Data type of elements in input sclae and bias vectors
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNDHWC;
using LayoutInputB = cutlass::layout::TensorNDHWC;
using LayoutInputScaleBias = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::TensorNDHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>; // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;          // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 4;

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

using Conv3dFpropFusionKernel = typename cutlass::conv::kernel::DefaultConv3dFpropFusion<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementInputScaleBias, LayoutInputScaleBias,
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

using ImplicitGemmFusion = cutlass::conv::device::ImplicitGemmConvolutionFusion<Conv3dFpropFusionKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::Tensor5DCoord input_size;
  cutlass::Tensor5DCoord filter_size;
  cutlass::Coord<3> padding;
  cutlass::Coord<3> conv_stride;
  cutlass::Coord<3> dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32, 32),
    filter_size(32, 3, 3, 3, 32),
    padding(cutlass::make_Coord(1, 1, 1)),
    conv_stride(cutlass::make_Coord(1, 1, 1)),
    dilation(cutlass::make_Coord(1, 1, 1)),
    reference_check(true),
    measure_performance(false),
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
    int const kAlignment = 8;

    if ((input_size.c() % kAlignment) ||
      (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding[0] != filter_size.d() / 2) || 
      (padding[1] != filter_size.h() / 2) ||
      (padding[2] != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    cutlass::Tensor5DCoord input_size,
    cutlass::Tensor5DCoord filter_size,
    cutlass::Coord<3> stride) {

    this->input_size = input_size;
    this->filter_size = filter_size;
    conv_stride = stride;

    padding[0] = filter_size.d() / 2;
    padding[1] = filter_size.h() / 2;
    padding[2] = filter_size.w() / 2;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("ref-check")) {
      reference_check = true;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("d", input_size.d());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("t", filter_size.d());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.c() = input_size.c(); 

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    if (filter_size.d() == 3 && filter_size.h() == 3 && filter_size.w() == 3) {
      padding = cutlass::make_Coord(1, 1, 1);
    }
    else {
      filter_size.d() = 1;
      filter_size.h() = 1;
      filter_size.w() = 1;
      padding = cutlass::make_Coord(0, 0, 0);
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "25_ampere_3d_fprop_mainloop_fusion example\n\n"
      << "  This example fuses scale+bias+relu of the activations into Ampere's\n"
      << "  Tensor Core operators on F16 data types to compute\n"
      << "  forward convolution on tensors of layout NDHWC.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n <int>            Input tensor extent N\n"
      << "  --d <int>            Input tensor extent D\n"
      << "  --h <int>            Input tensor extent H\n"
      << "  --w <int>            Input tensor extent W\n"
      << "  --c <int>            Input tensor extent C\n"
      << "  --k <int>            Filter extent K\n"
      << "  --t <int>            Filter extent T\n"
      << "  --r <int>            Filter extent R\n"
      << "  --s <int>            Filter extent S\n\n"
      << "  --alpha <float>      Epilogue scalar alpha\n"
      << "  --beta <float>       Epilogue scalar beta\n\n"
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations <int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag <string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./25_ampere_3d_fprop_mainloop_fusion  --n=32 --d=96 --h=96 --w=96 --c=64 --k=64 --t=1 --r=1 --s=1\n\n"
      << "$ ./25_ampere_3d_fprop_mainloop_fusion  --n=1  --d=224 --h=224 --w=224 --c=32 --k=32 --t=3 --r=3 --s=3 --ref-check\n\n"
      << "$ ./25_ampere_3d_fprop_mainloop_fusion  --n=19 --d=94 --h=96 --w=96 --c=128 --k=128 --t=1 --r=1 --s=1\n\n";

    return out;
  }
  
  /// Computes the output tensor size (NPQK)
  cutlass::Tensor5DCoord output_size() const {
    return cutlass::Tensor5DCoord(
      input_size.n(),
      (input_size.d() + padding[0] + padding[0] - filter_size.d()) / conv_stride[0] + 1,
      (input_size.h() + padding[1] + padding[1] - filter_size.h()) / conv_stride[1] + 1,
      (input_size.w() + padding[2] + padding[2] - filter_size.w()) / conv_stride[2] + 1,
      filter_size.n());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.d() * filter_size.h() * filter_size.w() * filter_size.c());
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cutlass::Status reference_check;
  cudaError_t error;

  Result(): 
    runtime_ms(0), 
    gflops(0),
    status(cutlass::Status::kSuccess),
    reference_check(cutlass::Status::kInvalid),
    error(cudaSuccess) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,D,H,W,C,K,T,R,S,Stride_D,Stride_H,Stride_W,Runtime,GFLOPs";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out 
      << "conv_" << idx << ","
      << options.input_size.n() << ","
      << options.input_size.d() << ","
      << options.input_size.h() << ","
      << options.input_size.w() << ","
      << options.input_size.c() << ","
      << options.filter_size.n() << ","
      << options.filter_size.d() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << options.conv_stride[0] << ","
      << options.conv_stride[1] << ","
      << options.conv_stride[2] << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs one benchmark
Result profile_convolution(Options const &options) {

  Result result;

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.input_size);
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_transformed_a(options.input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.filter_size);
  cutlass::HostTensor<ElementInputScaleBias, LayoutInputScaleBias>
      tensor_a_scale({1, options.input_size.c()});
  cutlass::HostTensor<ElementInputScaleBias, LayoutInputScaleBias>
      tensor_a_bias({1, options.input_size.c()});
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(options.output_size());

  //
  // Initialize tensors
  //

  // Fill tensor A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(3),
      ElementInputA(-4),
      0);

  // Fill scale vector for tensor A on host with uniform-distribution random
  // data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a_scale.host_view(),
      1,
      ElementInputA(3),
      ElementInputA(-4),
      0);

  // Fill bias vector for tensor A on host with uniform-distribution random
  // data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a_bias.host_view(),
      1,
      ElementInputA(3),
      ElementInputA(-4),
      0);

  // Fill tensor B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(7),
      ElementInputB(-8),
      0);

  // Fill tensor C on host with uniform-distribution random data 
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(7),
      ElementOutput(-8),
      0);

  // Fill tensor D for reference on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_a_scale.sync_device();
  tensor_a_bias.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  //
  // Define arguments for CUTLASS Convolution
  //

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Construct Conv3dProblemSize with user defined output size
  cutlass::conv::Conv3dProblemSize problem_size(      
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.output_size(),
      mode,
      split_k_slices
  );

  typename ImplicitGemmFusion::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_a_scale.device_ref(),
    tensor_a_bias.device_ref(),
    tensor_c.device_ref(),
    tensor_d.device_ref(),
    {options.alpha, options.beta},
  };

  //
  // Initialize CUTLASS Convolution
  //

  ImplicitGemmFusion implicit_gemm_fusion_op;

  size_t workspace_size = implicit_gemm_fusion_op.get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  result.status = implicit_gemm_fusion_op.can_implement(arguments);
  CUTLASS_CHECK(result.status);

  result.status = implicit_gemm_fusion_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(result.status);

  //
  // Launch initialized CUTLASS kernel
  //
  result.status = implicit_gemm_fusion_op();

  CUTLASS_CHECK(result.status);

  //
  // Optional reference check
  //
  
  if (options.reference_check) {
    std::cout << "Verification on device...\n";

    // Compute scale + bias + relu in host code
    for (int n = 0; n < options.input_size.n(); ++n) {
      for (int d = 0; d < options.input_size.d(); ++d) {
        for (int h = 0; h < options.input_size.h(); ++h) {
          for (int w = 0; w < options.input_size.w(); ++w) {
            for (int c = 0; c < options.input_size.c(); ++c) {
              tensor_transformed_a.at({n, d, h, w, c}) = std::max(
                  ElementOutput(0), ElementOutput(tensor_a.at({n, d, h, w, c}) *
                                                      tensor_a_scale.at({0, c}) +
                                                  tensor_a_bias.at({0, c})));
            }
          }
        }
      }
    }

    tensor_transformed_a.sync_device();

    // Compute with reference implementation
    cutlass::reference::device::Conv3dFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>
    >(
      problem_size,
      tensor_transformed_a.device_ref(),
      tensor_b.device_ref(),
      tensor_c.device_ref(),
      tensor_ref_d.device_ref(),
      options.alpha,
      options.beta
    );

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    bool passed = cutlass::reference::host::TensorEquals(
      tensor_d.host_view(),
      tensor_ref_d.host_view());

    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - results miscompared.\n";
    }
    else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Passed.\n";
    }
  }
  else {
    result.reference_check = cutlass::Status::kInvalid;
  }

  if (options.save_workspace) {

    std::stringstream ss;

    ss << "25_ampere_3d_fprop_mainloop_fusion"
      << options.input_size.n() << "x" << options.input_size.h() << "x" << options.input_size.w() << "x" << options.input_size.c() 
      << "_"
      << options.filter_size.n() << "x" << options.filter_size.h() << "x" << options.filter_size.w() << "x" << options.filter_size.c() 
      << ".dat";

    std::ofstream output_workspace(ss.str());

    output_workspace 
      << "Input = \n" << tensor_a.host_view() << "\n\n"
      << "Filters = \n" << tensor_b.host_view() << "\n\n";

    if (options.reference_check) {
      output_workspace << "Reference = \n" << tensor_ref_d.host_view() << "\n\n";
    }

    output_workspace << "Computed = \n" << tensor_d.host_view() << std::endl;

    std::cout << "Results written to '" << ss.str() << "'." << std::endl;
  }
 
  //
  // Performance measurement
  //

  if (options.measure_performance) {

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
      result.status = implicit_gemm_fusion_op();
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
  }

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 11.0.
  //
  // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv3dFprop examples.
  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (!(props.major >= 8)) {
    std::cerr << "This test must run on SM80 or above.\n";
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

    int batch_sizes[] = {34, 18};

    struct Benchmark {
      int d, h, w, c, k, t, r, s, stride_d, stride_h, stride_w;
    } layers[] = {
      {56, 56, 56,   64,  256, 1, 1, 1, 1, 1, 1},
      {56, 56, 56,   64,   64, 1, 1, 1, 1, 1, 1},
      {56, 56, 56,   64,   64, 3, 3, 3, 1, 1, 1},
      {56, 56, 56, 256,   64, 1, 1, 1, 1, 1, 1},
      {56, 56, 56,  256,  512, 1, 1, 1, 2, 2, 2},
      {56, 56, 56,  256,  128, 1, 1, 1, 1, 1, 1},
      {56, 56, 56,  128,  128, 3, 3, 3, 2, 2, 2},
      {28, 28, 28, 128,  512, 1, 1, 1, 1, 1, 1},
      {28, 28, 28,  512,  128, 1, 1, 1, 1, 1, 1},
      {28, 28, 28,  128,  128, 3, 3, 3, 1, 1, 1},
      {28, 28, 28,  512, 1024, 1, 1, 1, 2, 2, 2},
      {28, 28, 28, 512,  256, 1, 1, 1, 1, 1, 1},
      {28, 28, 28,  256,  256, 3, 3, 3, 2, 2, 2},
      {14, 14, 14,  256, 1024, 1, 1, 1, 1, 1, 1},
      {14, 14, 14, 1024,  256, 1, 1, 1, 1, 1, 1},
      {14, 14, 14,  256,  256, 3, 3, 3, 1, 1, 1},
      {14, 14, 14, 1024, 2048, 1, 1, 1, 2, 2, 2},
      {14, 14, 14, 1024,  512, 1, 1, 1, 1, 1, 1},
      {14, 14,  14, 512,  512, 3, 3, 3, 2, 2, 2},
      { 7,  7,  7, 512, 2048, 1, 1, 1, 1, 1, 1},
      { 7,  7, 7, 2048,  512, 1, 1, 1, 1, 1, 1},
      { 7,  7, 7, 512,  512, 3, 3, 3, 1, 1, 1},
    };

    Result::print_header(std::cout, options) << std::endl;

    int idx = 1;

    for (auto const &layer : layers) {
      for (auto N : batch_sizes) {
        options.update({N, layer.d, layer.h, layer.w, layer.c},
                       {layer.k, layer.t, layer.r, layer.s, layer.c},
                       cutlass::make_Coord(layer.stride_d, layer.stride_h, layer.stride_w));

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
