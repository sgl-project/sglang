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
    \brief Block-Ell sparse gemm example.

    This example performs a Sparse-matrix dense-matrix multiplication (SpMM) operation.
    Matrix A is stored in the Blocked-Ellpack (Blocked-ELL) storage format.
    Details about the Blocked-Ellpack (Blocked-ELL) storage format can be found here:
    https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-spmat-create-blockedell
    Whereas matrix B is a dense matrix.

    Blocked-Ellpack or Blocked-ELL storage format comprises of two matrices.
    First is a packed matrix (ellValue matrix) that stores non-zero values in consecutive blocks,
    represented by tensor_a in this example. Second is a matrix of indices (ellColInd matrix),
    represented by tensor_ell_idx in this example, that represent the column indices of the 
    corresponding non-zero blocks. All rows in the matrices must have the same number of blocks.
    ellColInd can contain -1 values for indicating empty blocks. These matrices store elements in
    row-major order.

    Description of parameters and tensors used to represent the Blocked-Ellpack (ELL) format
    for this example:
      a_rows              - Rows in the sparse matrix.
      a_cols              - Columns in the sparse matrix.
      a_ell_blocksize     - Size of the ELL-Blocks.
      a_ell_num_columns   - Number of columns in the Blocked-Ellpack format (ellValue columns)
      tensor_a            - ellValue matrix, whose size is (a_rows * a_ell_num_columns)
      tensor_ell_idx      - Blocked-ELL Column indices (ellColInd), whose size is
                            (a_rows / a_ell_blocksize) * (a_ell_num_columns / a_ell_blocksize)
      tensor_b            - Input dense matrix whose size is (a_cols * n)
      tensor_c/tensor_d   - Output dense matrix whose size is (a_rows * n)
      {a_rows, n, a_cols} - Problem size
    
*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/ell_gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/host_uncompress.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool reference_check;
  int iterations;
  int cuda_streams;
  int a_rows, n, a_cols;
  int a_ell_num_columns;
  int a_ell_blocksize;
  int a_base;
  float alpha;
  float beta;

  //
  // Methods
  // 

  Options():
    help(false),
    reference_check(true),
    iterations(20),
    cuda_streams(0),
    a_rows(1024),
    n(1024),
    a_cols(1024),
    a_ell_num_columns(512),
    a_ell_blocksize(16),
    a_base(0),
    alpha(1),
    beta()
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);    
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("streams", cuda_streams, 0);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);

    cmd.get_cmd_line_argument("a_rows", a_rows, 1024);
    cmd.get_cmd_line_argument("n", n, 1024);
    cmd.get_cmd_line_argument("a_cols", a_cols, 1024);

    cmd.get_cmd_line_argument("a_ell_num_columns", a_ell_num_columns, 512);
    cmd.get_cmd_line_argument("a_ell_blocksize", a_ell_blocksize, 16);
    cmd.get_cmd_line_argument("a_base", a_base, 0);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "43_ell_block_sparse_gemm\n\n"
      << "  This example profiles the performance of a ELL block sparse GEMM kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --a_rows=<int>              Sets the number of the rows of the sparse matrix.\n"
      << "  --n=<int>                   Sets the N dimension.\n"
      << "  --a_cols=<int>              Sets the number of columns of the sparse matrix.\n"
      << "  --a_ell_num_columns=<int>   Sets the actual number of columns of the Blocked-Ellpack format.\n"
      << "  --a_ell_blocksize=<int>     Sets the size of the ELL-Block.\n"
      << "  --a_base=<int>              Sets the base index.\n"
      << "  --alpha=<f32>               Epilogue scalar alpha (real part)\n"
      << "  --beta=<f32>                Epilogue scalar beta (real part)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n"
      << "  --reference-check=<bool>    If true, performs reference check.\n";

    out << "\n\nExamples:\n\n"

      << "# Runs a 1024x1024x1024 ELL block sparse GEMM with 16x16 block size and actual 512 non-zero columns in A operand\n"
      << "$ ./examples/43_ell_block_sparse_gemm/43_ell_block_sparse_gemm --a_rows=1024 --n=1024 --a_cols=1024 --a_ell_num_columns=512 --a_ell_blocksize=16\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = (int64_t)a_rows * (int64_t)a_cols * (int64_t)n;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
class Testbed {
public:

  //
  // Type definitions
  //

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  using MatrixCoord = typename LayoutC::TensorCoord;

private:

  //
  // Data members
  //

  Options options;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  cutlass::Distribution::Kind init_ELL;
  uint32_t seed;

  cutlass::HostTensor<ElementA, LayoutA> tensor_a;
  cutlass::HostTensor<ElementB, LayoutB> tensor_b;
  cutlass::HostTensor<ElementC, LayoutC> tensor_c;
  cutlass::HostTensor<ElementC, LayoutC> tensor_d;

  cutlass::HostTensor<ElementA, LayoutA> tensor_a_uncompressed;
  cutlass::HostTensor<ElementC, LayoutC> reference_d;

  cutlass::HostTensor<int32_t, LayoutA> tensor_ell_idx;

public:

  //
  // Methods
  //

  Testbed(
    Options const &options_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_ELL_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3080
  ):
    options(options_), init_A(init_A_), init_B(init_B_), init_C(init_C_), init_ELL(init_ELL_), seed(seed_) { }

private:

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  void initialize_tensor_(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint32_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      Element scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        if (cutlass::sizeof_bits<ElementAccumulator>::value <= 16) {
          scope_max = 5;
          scope_min = -5;
        }
        else {
          scope_max = 8;
          scope_min = -8;
        }
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(
        view, seed, Element(), Element(0.5f));
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      // Fill with increasing elements
      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity(), Element(1), Element());
    } else {

      // Fill with all 1s
      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity(), Element(), Element(1));
    }
  }

  /// Initializes data structures
  void initialize_() {
    tensor_a.resize(cutlass::make_Coord(options.a_rows, options.a_ell_num_columns));
    tensor_b.resize(cutlass::make_Coord(options.a_cols, options.n));
    tensor_c.resize(cutlass::make_Coord(options.a_rows, options.n));
    tensor_d.resize(cutlass::make_Coord(options.a_rows, options.n));

    tensor_a_uncompressed.resize(cutlass::make_Coord(options.a_rows, options.a_cols));
    reference_d.resize(cutlass::make_Coord(options.a_rows, options.n));

    tensor_ell_idx.resize(cutlass::make_Coord(options.a_rows / options.a_ell_blocksize,
                          options.a_ell_num_columns / options.a_ell_blocksize));

    //
    // Initialize the problems of the workspace
    //

    initialize_tensor_(tensor_a.host_view(), init_A, seed * 2021);
    initialize_tensor_(tensor_b.host_view(), init_B, seed * 2022);
    initialize_tensor_(tensor_c.host_view(), init_C, seed * 2023);

    if (init_ELL == cutlass::Distribution::Uniform) {
      cutlass::reference::host::TensorFillRandomEllIdx(
          tensor_ell_idx.host_view(), seed,
          options.a_rows / options.a_ell_blocksize,
          options.a_ell_num_columns / options.a_ell_blocksize,
          options.a_cols / options.a_ell_blocksize);

    } else {
      for(int i = 0; i < options.a_rows / options.a_ell_blocksize; ++i) {
        for(int j = 0; j < options.a_ell_num_columns / options.a_ell_blocksize; ++j) {
          tensor_ell_idx.at({i, j}) = j+3;
        }
      }
    }

    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    tensor_ell_idx.sync_device();
  }

  /// Verifies the result is a GEMM
  bool verify_() {

    bool passed = true;

    tensor_d.sync_host();

    cutlass::uncompress_ell_block_sparse(
          tensor_a_uncompressed.host_ref(),
          tensor_a.host_ref(),
          tensor_ell_idx.host_ref(),
          options.a_rows,
          options.a_cols,
          options.a_ell_num_columns,
          options.a_ell_blocksize
    );

    cutlass::reference::host::Gemm<
        typename Gemm::ElementA, typename Gemm::LayoutA,                                             
        typename Gemm::ElementB, typename Gemm::LayoutB,                                             
        typename Gemm::ElementC, typename Gemm::LayoutC,                                             
        ElementCompute,
        ElementAccumulator, typename Gemm::Operator>                                                 
        reference_gemm;                                                                              
    
    reference_gemm(                                                                                  
      {options.a_rows, options.n, options.a_cols},
      options.alpha, 
      tensor_a_uncompressed.host_ref(), 
      tensor_b.host_ref(),
      options.beta,
      reference_d.host_ref(),
      ElementAccumulator(0)
    );

    // Reference check
    passed = cutlass::reference::host::TensorEquals(tensor_d.host_view(), reference_d.host_view());

    if (!passed) {
      std::cerr << "\n***\nError - problem failed the QA check\n***\n" << std::endl;

      std::stringstream fname;

      fname << "error_43_ell_block_sparse_gemm"
            << "mnk_"
            << options.a_rows << "x"
            << options.n << "x"
            << options.a_cols << "_"
            << options.a_ell_num_columns << "_"
            << options.a_ell_blocksize << ".txt";

      std::cout << fname.str() << std::endl;

      std::ofstream results(fname.str());

      results
        << "alpha: " << ElementCompute(options.alpha) << "\n"
        << "beta: "  << ElementCompute(options.beta) << "\n"
        << "block size: " << options.a_ell_blocksize << "\n"
        << "\nA:\n" << tensor_a.host_view() << "\n"
        << "\nA Ell Index:\n" << tensor_ell_idx.host_view() << "\n"
        << "\nB:\n" << tensor_b.host_view() << "\n"
        << "\nC:\n" << tensor_c.host_view() << "\n"
        << "\nD reference:\n" << reference_d.host_view() << "\n"
        << "\nD computed:\n" << tensor_d.host_view() << "\n";


      return passed;
    }
    
    return passed;
  }

public:

  /// Returns the number of threadblocks to launch if the kernel can run on the target
  /// device. Otherwise, returns zero.
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    size_t smem_size = sizeof(typename Gemm::GemmKernel::SharedStorage);

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      return false;
    }

    return true;
  }

  /// Executes a BlockedEll SpMM kernel and measures runtime.
  Result profile() {

    Result result;

    // Early exit
    if (!sufficient()) {
      std::cout << "Active CUDA device lacks hardware resources to run CUTLASS BlockedEll SpMM kernel." << std::endl;
      return result;
    }

    result.passed = false;

    // Initialize the problem
    initialize_();

    // Configure the GEMM arguments
    typename EpilogueOutputOp::Params epilogue_op(options.alpha, options.beta);

    // Configure GEMM arguments
    typename Gemm::Arguments args(
      {options.a_rows, options.n, options.a_cols},
      tensor_a.device_ref(),
      tensor_b.device_ref(),
      tensor_c.device_ref(),
      tensor_d.device_ref(),
      tensor_ell_idx.device_data(),
      options.a_ell_num_columns,
      options.a_ell_blocksize,
      options.a_base,
      epilogue_op 
    );

    // Initialize the GEMM object
    Gemm gemm{};

    result.status = gemm.initialize(args);

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize CUTLASS BlockedEll SpMM kernel." << std::endl;
      return result;
    }

    // Run the BlockedEll SpMM object
    result.status = gemm.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS BlockedEll SpMM kernel." << std::endl;
      return result;
    }

    // Wait for completion
    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Verify correctness
    //
    result.passed = true;

    if (options.reference_check) {
      result.passed = verify_();
    }

    //
    // Warm-up run
    //
    result.status = gemm.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS BlockedEll SpMM kernel." << std::endl;
      return result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
      }
    }

    // Record an event at the start of a series of GEMM operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm();
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMM operations have been launched.
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

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    std::cout << std::endl;
    std::cout << "ELL Block Sparse GEMM (CUTLASS):\n"
      << "====================================================" << std::endl;

    std::cout << std::endl;
    std::cout << "    " << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << " GFLOPs: " << result.gflops << std::endl;

    return result;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  //
  // This example uses mma.sync to directly access Tensor Cores to achieve peak performance.
  //

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
  
    //
    // This example requires an NVIDIA Ampere-architecture GPU.
    //

    std::cout 
      << "CUTLASS's BlockedEll SpMM example requires a GPU of NVIDIA's Ampere Architecture or "
      << "later (compute capability 80 or greater).\n";

    return 0;
  }

  //
  // Parse options
  //

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Define the BlockedEll type
  //

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  constexpr int32_t kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr int32_t kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  constexpr int32_t kStages = 4;
  using Gemm = typename cutlass::gemm::device::EllGemm<
    ElementA, 
    LayoutA, 
    ElementB,
    LayoutB, 
    ElementOutput,
    LayoutC,
    ElementAccumulator, 
    cutlass::arch::OpClassTensorOp, 
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape, 
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 
    kStages, kAlignmentA, kAlignmentB>;

  //
  // Profile it
  //

  Testbed<Gemm> testbed(options);

  if (!testbed.sufficient()) {
    std::cout << "The active CUDA device lacks sufficient hardware resources to execute this kernel.\n";
    return 0;
  }

  Result result = testbed.profile();
  if (!result.passed) {
    std::cout << "Profiling CUTLASS ELL block sparse GEMM has failed.\n";
    std::cout << "\nFailed\n";
    return -1;
  }

  std::cout << "\nPassed\n";

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
