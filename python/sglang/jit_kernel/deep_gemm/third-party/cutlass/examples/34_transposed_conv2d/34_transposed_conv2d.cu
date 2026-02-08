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

/*
This example shows how to compute 2d transposed convolution, also known as deconvolution, using CUTLASS
conv2d Dgrad kernels. Although two operations are computationaly equivalent, some care is needed to correctly
set up a problem size for CUTLASS.
In deep learning, transposed convolution is sometimes used for upscaling feature maps. This example
demonstrates the 2x upscaling case using the strided Dgrad kernel.
*/

#include <iostream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

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
using cutlass::layout::TensorNHWC;
using cutlass::TensorRef;

using ElementAccumulator = cutlass::half_t;                  // Data type of accumulator
using ElementComputeEpilogue = cutlass::half_t;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementOutput = cutlass::half_t;                       // Data type of elements in output tensor
using ElementC = ElementOutput;
using ElementCompute = ElementComputeEpilogue;
using LayoutInputA = TensorNHWC;
using LayoutInputB = TensorNHWC;
using LayoutOutput = TensorNHWC;

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
using SwizzleThreadBlock = cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>;

// Number of pipelines you want to use
constexpr int NumStages = 3;

// This code section describe iterator algorithm selected is Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementCompute,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementCompute>::value,  // The number of elements per vectorized.
    // memory access. This becomes the vector width of
    // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

using Conv2dDgradKernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementAccumulator, LayoutOutput,
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
    IteratorAlgorithm,
    cutlass::conv::StrideSupport::kStrided  // Use the strided Dgrad specialization
    >::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 16),
    padding(1, 1, 1, 1),
    conv_stride(2, 2),
    dilation(1, 1),
    reference_check(true),
    measure_performance(false),
    iterations(20),
    alpha(1),
    beta(0) {}

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
    if ((padding.h() != filter_size.h() / 2) ||
	(padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("skip-ref-check")) {
      reference_check = false;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    // Filter layout is CRSK
    cmd.get_cmd_line_argument("k", filter_size.c());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.n() = input_size.c();

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

    out << "34_transposed_conv2d example\n\n"
	<< "  This example shows how to compute 2d transposed convolution, also known as\n"
	<< "  deconvolution, using CUTLASS conv2d Dgrad kernels. Although two operations are\n"
	<< "  computationaly equivalent, some care is needed to correctly set up a problem size.\n\n"
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
	<< "  --skip-ref-check     If set (true), skip reference check on the host\n"
	<< "  --perf-check         If set (true), performance is measured.\n"
	<< "  --iterations=<int>   Number of profiling iterations to perform.\n"
	<< "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
	<< "$ ./examples/34_transposed_conv2d/34_transposed_conv2d --n=8 --h=32 --w=32 --c=16 --k=32 --r=3 --s=3\n\n";

    return out;
  }

  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    // Here, out_pad corresponds to "output_padding" of conv2d_transpose op in deep learning frameworks.
    // See for example https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    int out_pad_h = conv_stride.row() > 1 ? 1 : 0;
    int out_pad_w = conv_stride.column() > 1 ? 1 : 0;
    int out_h = (input_size.h() - 1) * conv_stride.row() - 2 * padding.n() + (((filter_size.h() - 1) * dilation.row() + 1)) + out_pad_h;
    int out_w = (input_size.w() - 1) * conv_stride.column() - 2 * padding.w() + (((filter_size.w() - 1) * dilation.column() + 1)) + out_pad_w;
    return cutlass::Tensor4DCoord(input_size.n(), out_h, out_w, filter_size.c());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NHWC * KRS
    // Note that the input with the layout NHWC corresponds to the output from the perspective of dgrad,
    // and that the filter layout is CRSK.
    int64_t fmas = input_size.product() * int64_t(filter_size.h() * filter_size.w() * filter_size.n());

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

    out << "Layer,N,H,W,C,K,R,S,Stride_H,Stride_W,Runtime,GFLOPs";

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
      << options.filter_size.c() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << options.conv_stride.row() << ","
      << options.conv_stride.column() << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

// This is the same as Conv2dDgrad in tools/util/include/cutlass/util/reference/host/convolution.h,
// only variable names have been adapted for transposed conv2d.
void Conv2dTransposeReference(
  cutlass::conv::Conv2dProblemSize problem_size,
  TensorRef<ElementInputA, LayoutInputA> tensor_a,
  TensorRef<ElementInputB, LayoutInputB> tensor_b,
  TensorRef<ElementC, LayoutOutput> tensor_c,
  TensorRef<ElementC, LayoutOutput> tensor_d,
  ElementCompute alpha,
  ElementCompute beta) {

  int H = problem_size.P;
  int W = problem_size.Q;
  int P = problem_size.H;
  int Q = problem_size.W;
  int K = problem_size.C;
  int C = problem_size.K;

  for (int n = 0; n < problem_size.N; ++n) {
    for (int p = 0; p < P; ++p) {
      for (int q = 0; q < Q; ++q) {
        for (int k = 0; k < K; ++k) {

          ElementAccumulator acc = ElementAccumulator();

          for (int r = 0; r < problem_size.R; ++r) {
            for (int s = 0; s < problem_size.S; ++s) {
              for (int c = 0; c < C; ++c) {

                int filter_r = r;
                int filter_s = s;

                int h = p + problem_size.pad_h - filter_r * problem_size.dilation_h;
                int w = q + problem_size.pad_w - filter_s * problem_size.dilation_w;

                if (h >= 0 && (h % problem_size.stride_h) == 0 &&
                    w >= 0 && (w % problem_size.stride_w) == 0) {

                  h = h / problem_size.stride_h;
                  w = w / problem_size.stride_w;

                  if (h < H && w < W) {

                    ElementInputA a = tensor_a.at(cutlass::make_Coord(n, h, w, c));
                    ElementInputB b = tensor_b.at(cutlass::make_Coord(c, r, s, k));

                    acc += ElementAccumulator(a) * ElementAccumulator(b);
                  }
                }

              } // for (C)
            } // for (S)
          } // for (R)

          // Apply Epilogue, compute ElementCompute, convert and store ElementC
          ElementC c_ref = ElementC();

          if (beta != ElementCompute()) {
            c_ref = tensor_c.at(cutlass::make_Coord(n, p, q, k));
          }

          tensor_d.at(cutlass::make_Coord(n, p, q, k)) = alpha * ElementCompute(acc) + beta * ElementCompute(c_ref);

        } // for (K)
      } // for (W)
    } // for (H)
  } // for (N)
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs one benchmark
Result profile_convolution(Options const &options) {

  std::cout << "Output shape: " << options.output_size() << std::endl;

  Result result;

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //

  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_a(options.input_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_b(options.filter_size);
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_c(options.output_size());
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_d(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(options.output_size());

  //
  // Initialize tensors
  //

  // Fill tensor A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(7),
      ElementInputA(-8),
      0);

  // Fill tensor B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(7),
      ElementInputB(-8),
      0);

  // Fill tensor C and D on host with zeros
  cutlass::reference::host::TensorFill(tensor_c.host_view());

  cutlass::reference::host::TensorFill(tensor_d.host_view());

  // Fill tensor D for reference on host with zeros
  cutlass::reference::host::TensorFill(tensor_ref_d.host_view());

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();

  //
  // Define arguments for CUTLASS Convolution
  //

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Construct Conv2dProblemSize with user defined output size
  // The input in transposed conv2d corresponds to the output in the equivalent dgrad.
  // Similarly for the output.
  // Although the filter layout is CRSK from the perspective of conv2d transpose,
  // the filter size does not need to change for setting up the problem size.
  // There is no need to transpose the filter tensor either.

  cutlass::conv::Conv2dProblemSize problem_size(
      options.output_size(),
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.input_size,
      mode
  );

  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_c.device_ref(),
    tensor_d.device_ref(),
    {options.alpha, options.beta}
   };

  //
  // Initialize CUTLASS Convolution
  //

  ImplicitGemm implicit_gemm;

  size_t workspace_size = implicit_gemm.get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  result.status = implicit_gemm.can_implement(arguments);
  CUTLASS_CHECK(result.status);

  result.status = implicit_gemm.initialize(arguments, workspace.get());
  CUTLASS_CHECK(result.status);

  result.status = implicit_gemm();
  CUTLASS_CHECK(result.status);

  // // Skip reference check since there is no reference code for conv2d transpose in cutlass.
  if (options.reference_check) {
    tensor_d.sync_host();
    std::cout << "Verification on host...\n";
    Conv2dTransposeReference(problem_size,
                             tensor_a.host_ref(),
                             tensor_b.host_ref(),
                             tensor_c.host_ref(),
                             tensor_ref_d.host_ref(),
                             options.alpha, options.beta);

    bool passed = cutlass::reference::host::TensorEquals(tensor_d.host_view(), tensor_ref_d.host_view());

    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - results miscompared.\n";
    }
    else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Passed.\n";
    }
  }

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
      result.status = implicit_gemm();
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
