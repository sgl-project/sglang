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
This example shows how to run depthwise 2d convolution kernels using functions and data structures
provided by CUTLASS using SIMT instruction;

There are 3 types of implementations of depthwise 2d convoltion
  1. kAnalytic
    Implicit gemm 2d convoltion algorithm.
  2. kOptimized
    An optimized algorithm and supports arbitrary stride and dilation.
  3. kFixedStrideDilation
    An optimized algorithm with fixed stride and dilation to reduce the runtime computation and do
more optimizations.

In general, the perf of kFixedStrideDilation would be better than kOptimized. However, if the filter
size, stride or dilation is large, it would encounter register spilling and may hurt the perf. If
in this case, please use kOptimized.

For kOptimized and kFixedStrideDilation, in order to fully utilize GPU hardware resources and achieve
better perf, when the output tensor size is large, splitk should be enabled to achieve better perf.

In this example, it demonstrates how to construct and run a FixedStrideDilation depthwise 2d
convolution kernel.
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_depthwise_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/device/direct_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

// The code section below describes datatype for input, output tensors and computation between
// elements
using ElementAccumulator = cutlass::half_t;      // Data type of accumulator
using ElementComputeEpilogue = cutlass::half_t;  // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;           // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;           // Data type of elements in input tensor
using ElementOutput = cutlass::half_t;           // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassSimt;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm60;

// This code section describes the groups a thread block will compute
constexpr int groups_per_cta = 64;

// This code section describes the output tile <N, O, P, Q> a thread block will compute
using ThreadBlockOutputShape = cutlass::conv::TensorNHWCShape<1, 8, 8, groups_per_cta>;

// This code section describes the filter shape <R, S>
using FilterShape = cutlass::MatrixShape<3, 3>;

// Threadblock tile shape
using ThreadblockShape =
    cutlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, groups_per_cta, FilterShape::kCount>;

// This code section describes tile size a warp will computes
// WarpShape::kM = P * Q the warps would process
// WarpShape::kN = groups_per_cta that the warps would process
// WarpShape::kK = filter_size that the warps would process
using WarpShape = cutlass::gemm::GemmShape<16, groups_per_cta, FilterShape::kCount>;

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
        1,
        ThreadBlockOutputShape::kN,
        ThreadBlockOutputShape::kH,
        ThreadBlockOutputShape::kW>;

// Number of pipelines you want to use
constexpr int NumStages = 4;

// This code section describe iterator algorithm selected is kFixedStrideDilation
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
    cutlass::conv::IteratorAlgorithm::kFixedStrideDilation;
using StrideShape = cutlass::MatrixShape<1, 1>;
using DilationShape = cutlass::MatrixShape<1, 1>;

constexpr int kEpilogueElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,               // Data type of output matrix.
    kEpilogueElementsPerAccess,  // The number of elements per vectorized.
                                 // memory access. This becomes the vector width of
                                 // math instructions in the epilogue too.
    ElementAccumulator,          // Data type of accumulator
    ElementComputeEpilogue,      // Data type for alpha/beta in linear combination
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;  // Epilogue scaling operation.

using DepthwiseDirect2dConv = typename cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
    ElementInputA,
    LayoutInputA,
    ElementInputB,
    LayoutInputB,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    ThreadBlockOutputShape,
    FilterShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm,
    cutlass::conv::StrideSupport::kFixed,
    StrideShape,
    DilationShape>::Kernel;

using Direct2dConv = cutlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {
  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  int groups;
  int splitk;
  bool reference_check;
  bool measure_performance;
  int iterations;
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  std::string tag;

  Options()
      : help(false),
        input_size(1, 128, 128, 32),
        filter_size(32, 3, 3, 1),
        groups(32),
        padding(1, 1, 1, 1),
        conv_stride(1, 1),
        dilation(1, 1),
        reference_check(false),
        measure_performance(true),
        iterations(20),
        save_workspace(false),
        alpha(1),
        beta(0),
        splitk(1) {}

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {
    //
    // CUTLASS attempts to load 128b vectors of cutlass::half_t (F16) elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 8 elements.
    //
    int const kAlignment = 8;

    if ((input_size.c() % kAlignment) || (filter_size.n() % kAlignment)) {
      // misaligned tensors
      return false;
    }

    // depthwise conv
    if (groups != input_size.c()) {
      return false;
    }

    if (filter_size.n() != groups) {
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) || (padding.w() != filter_size.w() / 2)) {
      return false;
    }

    // Filter size passed through command line does not match filter size template parameter
    if (filter_size.h() != FilterShape::kRow || filter_size.w() != FilterShape::kColumn) {
      std::cerr << "Filter size passed in (" << filter_size.h() << "x" << filter_size.w() << ") "
                << "must match the FilterShape template parameter of the convolution "
                << "(" << FilterShape::kRow << "x" << FilterShape::kColumn << "). "
                << "To use the filter shape passed in, change the FilterShape template "
                << "parameter and recompile this example."
                << std::endl;
      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(cutlass::Tensor4DCoord input_size, cutlass::Tensor4DCoord filter_size) {
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

    if (cmd.check_cmd_line_flag("ref-check")) {
      reference_check = true;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());

    cmd.get_cmd_line_argument("g", groups);

    filter_size.c() = 1;
    filter_size.n() = input_size.c();

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("splitk", splitk);

    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    int32_t padding_h = filter_size.h() / 2;
    int32_t padding_w = filter_size.w() / 2;
    padding = {padding_h, padding_h, padding_w, padding_w};
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {
    out << "46_depthwise_gemm_fprop example\n\n"
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
        << "  --g=<int>            Groups\n\n"
        << "  --alpha=<float>      Epilogue scalar alpha\n"
        << "  --beta=<float>       Epilogue scalar beta\n\n"
        << "  --splitk=<int>       Enable splitK\n\n"
        << "  --ref-check          If set (true), reference check on the host is computed\n"
        << "  --perf-check         If set (true), performance is measured.\n"
        << "  --iterations=<int>   Number of profiling iterations to perform.\n"
        << "  --save-workspace     If set, workspace is written to a text file.\n"
        << "  --tag=<string>       String to replicate across the first column in the results "
           "table\n";

    out << "\n\nExamples:\n\n"
        << "$ ./examples/46_depthwise_simt_conv2dfprop/46_depthwise_simt_conv2dfprop  --n=32 "
           "--h=224 --w=224 --c=128 --k=128 --g=128 --r=3 --s=3\n\n"
        << "$ ./examples/46_depthwise_simt_conv2dfprop/46_depthwise_simt_conv2dfprop  --n=1 "
           "--h=224 --w=224 --c=32 --k=32 --g=32 --r=3 --s=3 --splitk=10 --ref-check\n\n";

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
    int64_t fmas =
        output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());

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

  Result()
      : runtime_ms(0),
        gflops(0),
        status(cutlass::Status::kSuccess),
        reference_check(cutlass::Status::kInvalid),
        error(cudaSuccess) {}

  static std::ostream &print_header(std::ostream &out, Options const &options) {
    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,H,W,C,K,R,S,G,stride_h,stride_w,dilation_h,dilation_w,splitK,Runtime,GFLOPs";

    return out;
  }

  std::ostream &print(std::ostream &out, int idx, Options const &options) {
    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    cutlass::Tensor4DCoord output_size = options.output_size();
    out << "conv_" << idx << "," << options.input_size.n() << "," << options.input_size.h() << ","
        << options.input_size.w() << "," << options.input_size.c() << ","

        << options.filter_size.n() << "," << options.filter_size.h() << ","
        << options.filter_size.w() << ","

        << options.groups << "," << options.conv_stride.row() << "," << options.conv_stride.column()
        << ","

        << options.dilation.row() << "," << options.dilation.column() << ","

        << options.splitk << ","

        << runtime_ms << "," << gflops;

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs one testcase
Result profile_convolution(Options const &options) {
  Result result;

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.filter_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b_transpose(options.filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(options.output_size());

  //
  // Initialize tensors
  //

  // Fill tensor A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(), 1, ElementInputA(5), ElementInputA(-6), 0);

  // Fill tensor B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(), 1, ElementInputB(3), ElementInputB(-6), 0);

  // Fill tensor C on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(), 1, ElementOutput(5), ElementOutput(-6), 0);

  // Fill tensor D on host with zeros
  cutlass::reference::host::TensorFill(tensor_d.host_view());

  // Fill tensor D for reference on host with zeros
  cutlass::reference::host::TensorFill(tensor_ref_d.host_view());

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_b_transpose.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  //
  // Define arguments for CUTLASS Convolution
  //

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Split P*Q into multiple CTA
  int split_k_slices = options.splitk;

  // Construct Conv2dProblemSize with user defined output size
  cutlass::conv::Conv2dProblemSize problem_size(options.input_size,
                                                options.filter_size,
                                                options.padding,
                                                options.conv_stride,
                                                options.dilation,
                                                options.output_size(),
                                                mode,
                                                split_k_slices,
                                                options.groups);

  // Construct Direc2dConv::Argument structure with conv2d
  // problem size, data pointers, and epilogue values
  typename Direct2dConv::Arguments arguments{problem_size,
                                             tensor_a.device_ref(),
                                             tensor_b.device_ref(),
                                             tensor_c.device_ref(),
                                             tensor_d.device_ref(),
                                             {options.alpha, options.beta},
                                             tensor_b_transpose.device_ref()};

  //
  // Initialize CUTLASS Convolution
  //

  Direct2dConv implicit_gemm_op;

  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  result.status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(result.status);

  result.status = implicit_gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(result.status);

  //
  // Launch initialized CUTLASS kernel
  //
  result.status = implicit_gemm_op();

  CUTLASS_CHECK(result.status);

  //
  // Optional reference check
  //

  if (options.reference_check) {
    std::cout << "Verification on host...\n";

    // Compute with reference implementation
    cutlass::reference::host::Conv2dFprop<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementComputeEpilogue,
        ElementAccumulator >(problem_size,
                             tensor_a.host_ref(),
                             tensor_b.host_ref(),
                             tensor_c.host_ref(),
                             tensor_ref_d.host_ref(),
                             options.alpha,
                             options.beta);

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    tensor_d.sync_host();

    bool passed =
        cutlass::reference::host::TensorEquals(tensor_d.host_view(), tensor_ref_d.host_view());

    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - results miscompared.\n";
    } else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Passed.\n";
    }
  } else {
    result.reference_check = cutlass::Status::kInvalid;
  }

  if (options.save_workspace) {
    std::stringstream ss;

    ss << "46_depthwise_simt_conv2dfprop" << options.input_size.n() << "x" << options.input_size.h()
       << "x" << options.input_size.w() << "x" << options.input_size.c() << "_"
       << options.filter_size.n() << "x" << options.filter_size.h() << "x"
       << options.filter_size.w() << "x" << options.filter_size.c() << ".dat";

    std::ofstream output_workspace(ss.str());

    output_workspace << "Input = \n"
                     << tensor_a.host_view() << "\n\n"
                     << "Filters = \n"
                     << tensor_b.host_view() << "\n\n";

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

    for (auto &event : events) {
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
      result.status = implicit_gemm_op();
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
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error)
                << std::endl;
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

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (!(props.major >= 6)) {
    std::cerr << "Run on a machine with compute capability at least 60." << std::endl;
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

  // Execute one problem size
  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  Result result = profile_convolution(options);

  Result::print_header(std::cout, options) << std::endl;
  result.print(std::cout, 1, options) << std::endl;

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
