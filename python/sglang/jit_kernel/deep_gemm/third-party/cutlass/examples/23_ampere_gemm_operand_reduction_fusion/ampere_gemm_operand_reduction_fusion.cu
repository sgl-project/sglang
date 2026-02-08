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
The example demonstrates how to reduce one of the operands of the GEMM along the k-dimension when
computing GEMM.  So the output also contains either a Mx1 or 1XN vector.  It only works with Ampere
16x8x16 FP16/BF16 tensor cores, though it is not difficult to apply to other Turing/Ampere tensor
core instructions.

Most of the reduction is done in gemm/warp level, see gemm/warp/mma_with_reduction_tensor_op.h
A few bit of reduction is done in the epilogue before storing the vector, see
epilogue/threadblock/epilogue_gemm_k_reduction.h 
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_with_k_reduction.h"
#include "cutlass/gemm/kernel/default_gemm_with_k_reduction.h"
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
#include "cutlass/util/reference/device/convolution.h"

#include "helper.h"

// The code section below describes datatype for input, output tensors and computation between
// elements 
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // Data type of epilogue computation
using ElementInputA = cutlass::bfloat16_t;         // Data type of elements in input tensor
using ElementInputB = cutlass::bfloat16_t;         // Data type of elements in input tensor
using ElementOutput = cutlass::bfloat16_t;         // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;
// Layout of the output vector
using LayoutGemmKReduction = cutlass::layout::PitchLinear;

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
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

// Number of pipelines you want to use
constexpr int NumStages = 4;

// Reduce A or B operand along the K dimension
constexpr bool ReduceKForA = true;

// Alignment of A operand
constexpr int AlignmentA = 8;

// Alignment of B operand
constexpr int AlignmentB = 8;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // The number of elements per vectorized.
                                                          // memory access. This becomes the vector width of
                                                          // math instructions in the epilogue too.
    ElementAccumulator,                                   // Data type of accumulator
    ElementComputeEpilogue>;

using Gemm = typename cutlass::gemm::device::GemmWithKReduction<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  ReduceKForA,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  AlignmentA,
  AlignmentB,
  cutlass::arch::OpMultiplyAdd,
  cutlass::ComplexTransform::kNone,
  cutlass::ComplexTransform::kNone
>;

// Below is the reduction kernel used in the case of parallel split-k
using ReduceGemmSplitKShape = cutlass::MatrixShape<4, 64>;

using ReduceOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    ElementOutput,
    EpilogueOp::kCount 
  >;

using ReduceGemmSplitKKernel = cutlass::reduction::kernel::ReduceSplitK<
    ReduceGemmSplitKShape,
    EpilogueOp,
    ReduceOp
  >;

using ReduceGemmSplitK = cutlass::reduction::device::ReduceSplitK<ReduceGemmSplitKKernel>;

using ReduceVectorSplitKShape = cutlass::MatrixShape<1, 256>;

// This code section describes the epilogue part of the kernel, we use default value
using DummyEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,     // The number of elements per vectorized.
                                                          // memory access. This becomes the vector width of
                                                          // math instructions in the epilogue too.
    ElementAccumulator,                                   // Data type of accumulator
    ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::Nothing>;

using ReduceVectorSplitKKernel = cutlass::reduction::kernel::ReduceSplitK<
    ReduceVectorSplitKShape,
    DummyEpilogueOp,
    ReduceOp
  >;

using ReduceVectorSplitK = cutlass::reduction::device::ReduceSplitK<ReduceVectorSplitKKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::gemm::GemmCoord problem_size;
  int split_k_slices;
  bool parallel_split_k;
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
    problem_size(1024, 1024, 1024),
    split_k_slices(1),
    parallel_split_k(false),
    reference_check(true),
    measure_performance(false),
    iterations(20),
    save_workspace(false),
    alpha(-1),
    beta(-1),
    benchmark(false) { }

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of cutlass::half_t (F16) elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 8 elements.
    //
    int const kAlignment = 8;

    if ((problem_size.m() % kAlignment) ||
        (problem_size.n() % kAlignment) ||
        (problem_size.k() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    cutlass::gemm::GemmCoord problem_size,
    int split_k_slices,
    bool parallel_split_k) {

    this->problem_size = problem_size;
    this->split_k_slices = split_k_slices;
    this->parallel_split_k = parallel_split_k;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("parallel-split-k")) {
      parallel_split_k = true;
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

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("split-k-slices", split_k_slices);

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "23_ampere_operand_gemm_reduction_fusion\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --m=<int>            GEMM M\n"
      << "  --n=<int>            GEMM N\n"
      << "  --k=<int>            GEMM K\n"
      << "  --split-k-slices=<int> Split K Slices\n"
      << "  --alpha=<float>      Epilogue scalar alpha\n"
      << "  --beta=<float>       Epilogue scalar beta\n\n"
      << "  --parallel-split-k   If set (true), use parallel split K\n"
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several problem sizes.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/23_ampere_gemm_operand_reduction_fusion/23_ampere_gemm_operand_reduction_fusion  --m=1024 --n=1024 --k=1024 \n\n";

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  cutlass::Status status;
  cutlass::Status reference_check;
  cudaError_t error;

  Result(): 
    runtime_ms(0), 
    status(cutlass::Status::kSuccess),
    reference_check(cutlass::Status::kInvalid),
    error(cudaSuccess) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "ID,M,N,K,SplitK-Slices,Parallel-SplitK,Runtime";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out 
      << "gemm_" << idx << ","
      << options.problem_size.m() << ","
      << options.problem_size.n() << ","
      << options.problem_size.k() << ","
      << options.split_k_slices << ","
      << options.parallel_split_k << ","
      << runtime_ms ;

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs one benchmark
Result profile(Options const &options) {

  Result result;

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.problem_size.mk());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.problem_size.kn());


  // Create tensor C with dimensions 1x1x1xk which is the bias vector
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.problem_size.mn());

  // Create tensor D used to store output from CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(options.problem_size.mn());
  // Create matrix D with dimensions M x N used to store output from reference
  // kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(options.problem_size.mn());

  int reduce_vector_length = ReduceKForA ? options.problem_size.m() : options.problem_size.n();

  cutlass::HostTensor<ElementOutput, LayoutGemmKReduction> tensor_reduction({reduce_vector_length, 1});
  cutlass::HostTensor<ElementOutput, LayoutGemmKReduction> tensor_ref_reduction({reduce_vector_length, 1});

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1997,
      ElementInputA(1),
      ElementInputA(-1),
      0);  // <- Fill tensor A on host with uniform-distribution random data

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      2003,
      ElementInputB(1),
      ElementInputB(-1),
      0);  // <- Fill tensor B on host with uniform-distribution random data

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      2017,
      ElementOutput(1),
      ElementOutput(-1),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  cutlass::reference::host::TensorFill(
      tensor_reduction.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_reduction.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();
  tensor_reduction.sync_device();

  // Initialize alpha for dot product computation
  ElementComputeEpilogue alpha = options.parallel_split_k ? ElementComputeEpilogue(1)
                                                          : ElementComputeEpilogue(options.alpha);
  ElementComputeEpilogue beta = options.parallel_split_k ? ElementComputeEpilogue(0)
                                                         : ElementComputeEpilogue(options.beta);

  cutlass::gemm::GemmUniversalMode mode = options.parallel_split_k ? 
                     cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel :
                     cutlass::gemm::GemmUniversalMode::kGemm;

  int batch_count = options.split_k_slices;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments(
    mode,
    options.problem_size,
    batch_count,
    {alpha, beta},
    tensor_a.device_ref().data(),              // <- reference to tensor A on device
    tensor_b.device_ref().data(),              // <- reference to tensor B on device
    tensor_c.device_ref().data(),              // <- reference to matrix C on device
    tensor_d.device_ref().data(),              // <- reference to matrix D on device
    tensor_reduction.device_ref().data(),      // <- reference to reduction tensor on device
    options.problem_size.m() * options.problem_size.k(),
    options.problem_size.n() * options.problem_size.k(),
    options.problem_size.m() * options.problem_size.n(),
    options.problem_size.m() * options.problem_size.n(),
    reduce_vector_length,
    tensor_a.layout().stride(0),
    tensor_b.layout().stride(0),
    tensor_c.layout().stride(0),
    tensor_d.layout().stride(0),
    tensor_reduction.layout().stride(0));

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  result.status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(result.status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  result.status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(result.status);

  // Launch initialized CUTLASS kernel
  result.status = gemm_op();

  CUTLASS_CHECK(result.status);

  if (options.parallel_split_k && batch_count > 1) {
    // reduce gemm

    ElementComputeEpilogue alpha = ElementComputeEpilogue(options.alpha);
    ElementComputeEpilogue beta = ElementComputeEpilogue(options.beta);

    int splitk_gemm_stride = options.problem_size.m();

    cutlass::layout::RowMajor splitk_gemm_layout(splitk_gemm_stride);

    void * workspace_gemm_ptr = workspace.get();
    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> workspace_gemm_tensorref(static_cast<ElementOutput *>(workspace_gemm_ptr), splitk_gemm_layout);

    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> tensor_d_tensorref(tensor_d.device_ref().data(), splitk_gemm_layout);

    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> tensor_c_tensorref(tensor_c.device_ref().data(), splitk_gemm_layout);

    typename ReduceGemmSplitK::Arguments reduce_gemm_splitk_arguments{
      cutlass::MatrixCoord(options.problem_size.n(), options.problem_size.m()),
      batch_count,
      size_t(options.problem_size.m() * options.problem_size.n()),
      workspace_gemm_tensorref,
      tensor_d_tensorref,
      tensor_c_tensorref,
      {alpha, beta} 
    };

    ReduceGemmSplitK reduce_gemm_splitk_op;
   
    result.status = reduce_gemm_splitk_op.initialize(reduce_gemm_splitk_arguments); 
    CUTLASS_CHECK(result.status);

    result.status = reduce_gemm_splitk_op();
    CUTLASS_CHECK(result.status);

    // reduce k vector
    cutlass::layout::RowMajor splitk_vector_layout(reduce_vector_length);
   
    ElementOutput *workspace_vector_ptr = static_cast<ElementOutput *>(workspace_gemm_ptr) + batch_count * options.problem_size.m() * options.problem_size.n();
    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> workspace_vector_tensorref(workspace_vector_ptr, splitk_vector_layout);

    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> tensor_reduction_tensorref(tensor_reduction.device_ref().data(), splitk_vector_layout);

    cutlass::TensorRef<ElementOutput, cutlass::layout::RowMajor> tensor_nullptr_tensorref(nullptr, splitk_vector_layout);

    typename ReduceVectorSplitK::Arguments reduce_vector_splitk_arguments(
      cutlass::MatrixCoord(1, reduce_vector_length),
      batch_count,
      size_t(reduce_vector_length),
      workspace_vector_tensorref,
      tensor_reduction_tensorref,
      tensor_nullptr_tensorref,
      {1.0f, 0.0f});

    ReduceVectorSplitK reduce_vector_splitk_op;
   
    result.status = reduce_vector_splitk_op.initialize(reduce_vector_splitk_arguments); 
    CUTLASS_CHECK(result.status);

    result.status = reduce_vector_splitk_op();
    CUTLASS_CHECK(result.status);
  }

  //
  // Create instantiation for device reference conv kernel
  //
  if (options.reference_check) {
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
        options.problem_size,
        ElementComputeEpilogue(options.alpha),
        tensor_a.device_ref(),
        tensor_b.device_ref(),
        ElementComputeEpilogue(options.beta),
        tensor_c.device_ref(), 
        tensor_ref_d.device_ref()
      );
  
    // Wait for kernels to finish
    cudaDeviceSynchronize();
  
    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();
  
    tensor_reduction.sync_host();
  
    // Reduce K in host code
    if (ReduceKForA) {
      for (int m = 0; m < options.problem_size.m(); ++m) {
        for (int k = 0; k < options.problem_size.k(); ++k) {
          tensor_ref_reduction.at({m, 0}) += 
            tensor_a.at(cutlass::MatrixCoord(m, k));
        }
      }
    } else {
      for (int k = 0; k < options.problem_size.k(); ++k) {
        for (int n = 0; n < options.problem_size.n(); ++n) {
          tensor_ref_reduction.at({n, 0}) += 
            tensor_b.at(cutlass::MatrixCoord(k, n));
        }
      }
    }
  
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool pass = cutlass::reference::host::TensorEquals(tensor_d.host_view(),
                                                       tensor_ref_d.host_view());

    pass &= cutlass::reference::host::TensorEquals(tensor_ref_reduction.host_view(),
                                                   tensor_reduction.host_view());

    if (!pass) {
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

    ss << "23_ampere_gemm_operand_reduction_fusion"
      << options.problem_size.m() << "x" << options.problem_size.n() << "x" << options.problem_size.k()
      << ".dat";

    std::ofstream output_workspace(ss.str());

    output_workspace 
      << "A = \n" << tensor_a.host_view() << "\n\n"
      << "B = \n" << tensor_b.host_view() << "\n\n";

    if (options.reference_check) {
      output_workspace << "Reference D = \n" << tensor_ref_d.host_view() << "\n\n";
      output_workspace << "Reference reduction vector = \n" << tensor_ref_reduction.host_view() << "\n\n";
    }

    output_workspace << "Computed D = \n" << tensor_d.host_view() << std::endl;
    output_workspace << "Computed reduction vector = \n" << tensor_reduction.host_view() << std::endl;

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
      result.status = gemm_op();
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

    struct Benchmark {
      int m, n, k, split_k_slices, parallel_split_k;
    } problem_sizes[] = {
      {4096, 6144, 4096, 1, false},
    };

    Result::print_header(std::cout, options) << "\n";
 
    int idx = 1;

    for (auto const &problem_size : problem_sizes) {
      options.update({problem_size.m, problem_size.n, problem_size.k},
                     problem_size.split_k_slices, problem_size.parallel_split_k);

      Result result = profile(options);
      result.print(std::cout, idx, options) << "\n";

      ++idx;
    }
  } else { 

    // Execute one problem size
    if (!options.valid()) {
      std::cerr << "Invalid problem." << "\n";
      return -1;
    }

    Result result = profile(options);

    Result::print_header(std::cout, options) << "\n";
    result.print(std::cout, 1, options) << "\n";
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
