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
This example shows how to compute conv2d gradient with respect to weight (wgrad). In wgrad, the K dimension of
impligit GEMM, corresponding to the sequential reduction loop, is very large (N * P * Q). Split-k with parallel
reduction is highly effective for such cases. Given split_k_slices parameter, it partitions the K loop into
split_k_slices chunks and computes partial reductions in parallel across different blocks. After that,
a parallel reduction kernel is launched to accumulate partial reductions.
In practice, wgrad requires fp32 accumulation to avoid overflow. When the input is fp16, some care is needed
to correctly instantiate the GEMM template.
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad.h"
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
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

#include "helper.h"

// The code section below describes datatype for input, output tensors and computation between
// elements
// In Wgrad, fp32 accumulation is necessary in practice.
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementOutput = cutlass::half_t;                       // Data type of elements in output tensor
using ElementC = ElementOutput;
using ElementCompute = ElementComputeEpilogue;
using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

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
constexpr int NumStages = 3;

// This code section describe iterator algorithm selected is Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// We need two epilogue functors - one for GEMM and another for the final reduction.
// The epilogue for GEMM is not used, but needed to instantiate the CUTLASS kernel template.
// Note that, when the input is fp16 and accumulation is fp32, the output of GEMM needs to be fp32,
// the final reduction is done in fp32, and the reduction epilogue converts fp32 outputs to fp16.
// Therefore, the output type of the GEMM epilogue is ElementCompute, not ElementOutput.

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOpGEMM = cutlass::epilogue::thread::LinearCombination<
    ElementCompute,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementCompute>::value,  // The number of elements per vectorized.
    // memory access. This becomes the vector width of
    // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

// The epilogue functor for reduction. This is the one that is actually used.
using EpilogueOpReduction = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized.
    // memory access. This becomes the vector width of
    // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in lin

using Conv2dWgradKernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementAccumulator, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOpGEMM,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    IteratorAlgorithm
    >::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dWgradKernel>;

using EpilogueOutputOp = EpilogueOpReduction;

/// Reduction kernel
using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
   >;

using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
    cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
    EpilogueOutputOp,
    ReductionOp
   >;

using ReductionDevice = cutlass::reduction::device::ReduceSplitK<ReductionKernel>;
using ReductionStrideIndex = typename ReductionDevice::StrideIndex;

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
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  int split_k_slices;
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(true),
    measure_performance(false),
    iterations(20),
    save_workspace(false),
    alpha(1),
    beta(0),
    split_k_slices(8),
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
    if ((padding.h() != filter_size.h() / 2) ||
	(padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
	      cutlass::Tensor4DCoord input_size,
	      cutlass::Tensor4DCoord filter_size,
	      cutlass::MatrixCoord stride) {

    this->input_size = input_size;
    this->filter_size = filter_size;
    conv_stride = stride;

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
    cmd.get_cmd_line_argument("split-k-slices", split_k_slices);

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

    out << "30_wgrad_split_k example\n\n"
	<< "  This example shows how to compute conv2d gradient with respect to weight (wgrad).\n"
	<< "  In wgrad, the K dimension of impligit GEMM, corresponding to the sequential reduction loop, is very large (N * P * Q).\n"
	<< "  Split-k with parallel reduction is highly effective for such cases.\n\n"
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
	<< "  --split-k-slices=<int>   Split-k factor \n\n"
	<< "  --ref-check          If set (true), reference check on the host is computed\n"
	<< "  --perf-check         If set (true), performance is measured.\n"
	<< "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
	<< "  --iterations=<int>   Number of profiling iterations to perform.\n"
	<< "  --save-workspace     If set, workspace is written to a text file.\n"
	<< "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
	<< "$ ./examples/30_wgrad_split_k/30_wgrad_split_k --n=32 --h=224 --w=224 --c=128 --k=256 --r=3 --s=3 --split-k-slices=8\n\n";

    return out;
  }

  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    return cutlass::Tensor4DCoord(input_size.n(),
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
      << options.filter_size.n() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << options.conv_stride.row() << ","
      << options.conv_stride.column() << ","
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

  // Inputs are the output gradient and the original activation.
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.output_size());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.input_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(options.filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(options.filter_size);

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

  // Fill tensor C, D on host with zeros
  cutlass::reference::host::TensorFill(tensor_c.host_view());

  cutlass::reference::host::TensorFill(tensor_d.host_view());

  // Fill tensor D for reference on host with zeros
  cutlass::reference::host::TensorFill(tensor_ref_d.host_view());

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  //
  // Define arguments for CUTLASS Convolution
  //

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Partition the GEMM K loop into split_k_slices chunks
  int split_k_slices = options.split_k_slices;

  // Construct Conv2dProblemSize with user defined output size
  // Do not forget to pass the last argument.
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

  using cutlass::layout::TensorNHWC;

  cutlass::conv::SplitKMode const split_k_mode = cutlass::conv::SplitKMode::kParallel;

  // Since the epilogue is not computed after GEMM, there is no need to pass the C tensor and
  // alpha and beta can be set to 1 and 0 respectively.
  // Moreover, since the output will be written to the workspace, there is no need to pass
  // the D tensor as well.
  // Do not forget to pass the last argument.
  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    {nullptr, TensorNHWC()},
    {nullptr, TensorNHWC()},
    {ElementCompute(1), ElementCompute(0)},
    split_k_mode
  };

  //
  // Initialize CUTLASS Convolution
  //

  ImplicitGemm implicit_gemm;

  size_t workspace_size = implicit_gemm.get_workspace_size(arguments);

  // Split-K requires non-zero workspace size. The workspace size grows linearly with split_k_slices.
  std::cout << "split-k-slices: " << split_k_slices << std::endl;
  std::cout << "workspace size: " << workspace_size << std::endl;

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  result.status = implicit_gemm.can_implement(arguments);
  CUTLASS_CHECK(result.status);

  // After the workspace is allocated, we point the GEMM destination pointer to the workspace.
  TensorNHWC layout_D{TensorNHWC::packed(options.filter_size)};
  arguments.ref_D.reset(reinterpret_cast<ElementCompute*>(workspace.get()), layout_D);

  result.status = implicit_gemm.initialize(arguments, workspace.get());
  CUTLASS_CHECK(result.status);

  //
  // Launch initialized CUTLASS kernel
  //
  result.status = implicit_gemm();

  CUTLASS_CHECK(result.status);

  if (split_k_mode == cutlass::conv::SplitKMode::kParallel) {
    // Do reduction
    ReductionDevice reduction_op;
    auto& status = result.status;
    static cutlass::conv::Operator const kConvolutionalOperator = ImplicitGemm::kConvolutionalOperator;
    typename ReductionDevice::Arguments reduction_args(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn(),
        problem_size.split_k_slices,
        cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, problem_size),
        // Reduction input
        {
            reinterpret_cast<ElementAccumulator*> (workspace.get()),
            ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
        },
        // Destination
        {
            tensor_d.device_data(),
            ReductionStrideIndex(tensor_d.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
        },
        // Source
        {
            tensor_c.device_data(),
            ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
        },
        {options.alpha, options.beta}
    );

    status = reduction_op.initialize(reduction_args, nullptr);
    status = reduction_op();
  }

  //
  // Optional reference check
  //

  if (options.reference_check) {
    std::cout << "Verification on device...\n";

    // Compute with reference implementation
    cutlass::reference::device::Conv2dWgrad<
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
	  tensor_a.device_ref(),
	  tensor_b.device_ref(),
	  tensor_c.device_ref(),
	  tensor_ref_d.device_ref(),
	  options.alpha,
	  options.beta
	  );

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    tensor_c.sync_host();
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

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
  else {
    result.reference_check = cutlass::Status::kInvalid;
  }

  if (options.save_workspace) {

    std::stringstream ss;

    ss << "30_wgrad_split_k_"
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

    output_workspace << "Computed = \n" << tensor_c.host_view() << std::endl;

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

  if (options.benchmark) {
    // Benchmark several layers

    int batch_sizes[] = {34, 408};

    struct Benchmark {
      int h, w, c, k, r, s, stride_h, stride_w;
    } layers[] = {
	 {56, 56,   64,  256, 1, 1, 1, 1},
	 {56, 56,   64,   64, 1, 1, 1, 1},
	 {56, 56,   64,   64, 3, 3, 1, 1},
	 {56, 56,  256,   64, 1, 1, 1, 1},
	 {56, 56,  256,  512, 1, 1, 2, 2},
	 {56, 56,  256,  128, 1, 1, 1, 1},
	 {56, 56,  128,  128, 3, 3, 2, 2},
	 {28, 28,  128,  512, 1, 1, 1, 1},
	 {28, 28,  512,  128, 1, 1, 1, 1},
	 {28, 28,  128,  128, 3, 3, 1, 1},
	 {28, 28,  512, 1024, 1, 1, 2, 2},
	 {28, 28,  512,  256, 1, 1, 1, 1},
	 {28, 28,  256,  256, 3, 3, 2, 2},
	 {14, 14,  256, 1024, 1, 1, 1, 1},
	 {14, 14, 1024,  256, 1, 1, 1, 1},
	 {14, 14,  256,  256, 3, 3, 1, 1},
	 {14, 14, 1024, 2048, 1, 1, 2, 2},
	 {14, 14, 1024,  512, 1, 1, 1, 1},
	 {14, 14,  512,  512, 3, 3, 2, 2},
	 { 7,  7,  512, 2048, 1, 1, 1, 1},
	 { 7,  7, 2048,  512, 1, 1, 1, 1},
	 { 7,  7,  512,  512, 3, 3, 1, 1},
    };

    Result::print_header(std::cout, options) << std::endl;

    int idx = 1;

    for (auto const &layer : layers) {
      for (auto N : batch_sizes) {
        options.update({N, layer.h, layer.w, layer.c},
                       {layer.k, layer.r, layer.s, layer.c},
                       {layer.stride_h, layer.stride_w});

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
