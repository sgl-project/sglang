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

This example shows how to run CUTLASS's convolution kernels
based on the Implicit GEMM algorithm, that use the Tensor Cores
on an NVIDIA Ampere GPU.

Writing a single high-performance convolution kernel is hard enough,
let alone writing kernels that perform well for multiple problem sizes
and use good software abstractions.
CUTLASS provides simplified abstractions
to compose multiple sections of a convolution kernel.
When used properly, the kernels can reach peak GPU performance.

CUTLASS divides a kernel into hierarchical composable sections
for each level of the GPU hardware hierarchy:
thread, warp, and threadblock.
Each section computes on its own tile shape,
with each higher level's tile shape
being composed from lower-level tile shapes.
Multiple thread tiles (the tile shape each thread computes)
can be used to form warp tiles (the tile shape each warp computes),
and multiple warp tiles can be used to compute threadblock tiles
(the tile shape computed by a threadblock).

In this example, we split variable initialization into two parts.

1. Setting up data properties: describes how tensors are laid out in the memory
   and how the kernel can view them (logical to physical mapping)

2. Setting up computation properties: describes how the above tensors
   will be used to compute the output of convolution

We begin by setting up the data types
of all the input and output elements of a convolution.
A convolution computes
C = alpha * Conv2dFprop(A, B) + beta * C,
so we set up data types for the input tensor A,
weights tensor B, output tensor C,
and the scaling factors alpha and beta.
CUTLASS divides the convolution into two parts:
the "mainloop" that computes X = Conv2dFprop(A, B),
and the "epilogue" that computes C = alpha * X + beta * C.
The epilogue is an element-wise operation on X and C.
In this case, it is a linear combination,
but other epilogues are possible.

In this example, we want

* the scaling factors alpha and beta to be float,

* the elements of A and B to be cutlass::half_t
  (a 16-bit floating-point type),

* the elements of C to be float, and

* intermediate sums to be accumulated in float.

We convey this to the CUTLASS kernel
by setting the following template parameters.

* alpha and beta: ElementComputeEpilogue = float

* Elements of input tensor A: ElementInputA = cutlass::half_t

* Elements of input tensor B: ElementInputB = cutlass::half_t

* Elements of output tensor C: ElementOutput = float

* Accumulation type: ElementAccumulator = float

Next, we describe the layout of the input and output tensors.
We convey this to the CUTLASS kernel
by setting the following template parameters.

* Layout of input tensor A: LayoutInputA = TensorNHWC

* Layout of input tensor B: LayoutInputB = TensorNHWC

* Layout of output tensor C: LayoutOutput = TensorNHWC

After that, we set up rules to compute the epilogue.
The epilogue in this case is a simple linear combination
C = alpha * X + beta * C.
Thus, we set the kernel's template parameter EpilogueOp
to LinearCombination.  LinearCombination itself
has template parameters:

* the element type of the output tensor (ElementOutput),

* the number of elements per vector memory access (8),

* the data type of the accumulator (ElementAccumulator),

* and the data type used to compute the linear combination
  (ElementComputeEpilogue).

We then define the tile shapes
that each level of the computation uses.
We define these as types that encode the tile shapes
as compile-time integer values.
Each shape expresses the dimensions M x N x K.
Here, the letters refer to the dimensions
of a matrix-matrix multiply.

* ThreadblockShape defines the threadblock tile shape
  as 128 x 128 x 64.

* WarpShape defines the warp tile shape as 64 x 64 x 64.

* InstructionShape defines the MMA
  (matrix multiply-accumulate) operation shape
  as 16 x 8 x 16.

These types become template arguments
of the kernel properties type
cutlass::conv::kernel::DefaultConv2dFprop.
The kernel uses these shapes to deduce
the number of threads needed per threadblock,
the required amount of shared memory,
the internal layouts needed to access
shared memory without bank conflicts,
and many other properties that the kernel needs
for good performance.
CUTLASS deduces all these properties automatically,
so that users don't have to.
DefaultConv2dFprop accepts other template parameters
that describe things like the target CUDA SM architecture.

CUTLASS also supports multiple MMA pipelines in a threadblock.
An MMA pipeline constitutes the whole process
of loading input data from global memory to shared memory,
loading data from shared memory to registers,
doing matrix multiplication,
and storing the result to global memory.
The below flow sequence shows a typical MMA multistage pipeline
(see include/cutlass/conv/threadblock/implicit_gemm_multistage.h).

tensor in global memory
--cp_async-->
tile in shared memory
--smem loads-->
registers
--mma-->
registers
--global stores-->
output to global memory

On NVIDIA Ampere, the kernel uses `cp_async`
to build a multistage software pipeline.
This helps it better hide latency.

At this point, we can define the actual CUTLASS kernel type
as the alias ImplicitGemm, a specialization of
cutlass::conv::device::ImplicitGemmConvolution.
The latter accepts the kernel properties type alias
Conv2dFpropKernel as its one template argument.

This example then sets up a test problem
and arguments to the kernel.
We use CUTLASS utilities to allocate
the input and output tensors
and fill them with sample input data.
We then create the kernel arguments
as an instance of ImplicitGemm::Arguments.
The arguments include
the problem size (N = 1, H = 64, W = 64, C = 128),
filter size (K = 64, R = 3, S = 3, C = 128),
padding, strides, dilation, tensors, alpha, beta,
and the split k-dimension factor.
We also query CUTLASS if the kernel we instantiated
requires any memory for scratch space.
If yes, we reserve scratch space and pass it along
with other arguments to initialize the CUTLASS kernel.

After launching the CUTLASS kernel, this example runs
a reference convolution kernel (from CUTLASS utilities)
to check correctness.
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
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

// Data types for input and output tensors
// and computation between elements
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// Whether to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// SM architecture number
using SmArch = cutlass::arch::Sm80;

// Threadblock tile shape
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;

// Warp tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;

// MMA (Tensor Core instruction, in this case) tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// How the kernel schedules threadblocks
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipeline stages to use
constexpr int NumStages = 3;

// Which iterator algorithm to use: Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// Is the output packed or strided
// Use kStride if using strided output
static cutlass::conv::StrideSupport const OutputStride = cutlass::conv::StrideSupport::kUnity;

// The epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

// Kernel properties type
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
  IteratorAlgorithm,
  OutputStride
>::Kernel;

// Type of the actual kernel
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

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
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(false),
    measure_performance(true),
    iterations(20),
    save_workspace(false),
    alpha(1),
    beta(0),
    benchmark(false) { }

  // Verify that the problem size is compatible with CUTLASS's convolution implementation
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

  /// Update input and filter sizes
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

  // Parse command-line arguments
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

  /// Print an explanation of the command-line arguments
  std::ostream & print_usage(std::ostream &out) const {

    out << "16_ampere_tensorop_conv2dfprop example\n\n"
      << "  This example uses Ampere's Tensor Core operators on F16 data types\n"
      << "  to compute forward convolution on tensors of layout NHWC.\n\n"
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
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/16_ampere_tensorop_conv2dfprop/16_ampere_tensorop_conv2dfprop  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
      << "$ ./examples/16_ampere_tensorop_conv2dfprop/16_ampere_tensorop_conv2dfprop  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

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

  /// Compute performance in Gflop/s
  ///
  /// Gflop/s stands for billions (10^9) of
  /// floating-point operations per second (Gflop/s).
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());

    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

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

  static std::ostream& print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,H,W,C,K,R,S,Runtime,GFLOPs";

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
      << gflops;

    return out;
  }
};

/// Runs one benchmark
Result profile_convolution(Options const &options) {

  Result result;

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(options.output_size());

  //
  // Initialize tensors
  //

  // Fill tensor A on host with uniformly distributed random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(7),
      ElementInputA(-8),
      0);

  // Fill tensor B on host with uniformly distributed random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(7),
      ElementInputB(-8),
      0);

  // Fill tensor C on host with uniformly distributed random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(7),
      ElementOutput(-8),
      0);

  // Fill tensor D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());

  // Fill tensor D for reference on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());

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

  // Construct ImplicitGemm::Argument structure with conv2d
  // problem size, data pointers, and epilogue values
  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_c.device_ref(),
    tensor_d.device_ref(),
    {options.alpha, options.beta},
  };

  //
  // Initialize CUTLASS Convolution
  //

  ImplicitGemm implicit_gemm_op;

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
      ElementAccumulator
    >(
      problem_size,
      tensor_a.host_ref(),
      tensor_b.host_ref(),
      tensor_c.host_ref(),
      tensor_ref_d.host_ref(),
      options.alpha,
      options.beta
    );

    // Check if CUTLASS kernel and reference kernel produced the same output
    tensor_d.sync_host();

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

    ss << "16_ampere_workspace_conv2dfprop_"
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

    // Launch a sequence of implicit GEMM operations on the device.
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
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime.
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Print average run time and floating-point throughput (Gflop/s).
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    // Cleanup
    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }
  }

  return result;
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

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.benchmark) {
    // Benchmark several layers

    int batch_sizes[] = {1, 32, 64, 128, 256, 512};

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
