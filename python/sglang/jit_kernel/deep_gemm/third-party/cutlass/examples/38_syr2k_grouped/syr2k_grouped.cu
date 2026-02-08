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
    \brief SYR2K Grouped Example.

    This workload computes a batch of SYR2K operations with distinct problem sizes. This example closely
    follows 24_gemm_grouped.

    Examples:

      # Runs a grouped SYR2K with 100 random problem sizes
      $ ./examples/38_syr2k_grouped/38_syr2k_grouped --groups=100

      # Runs a grouped SYR2K with 100 random problem sizes (with SYR2K-K dimension equal to 1024)
      $ ./examples/38_syr2k_grouped/24_gemm_grouped --groups=100 --k=1024 --verbose=true

      # Runs a grouped SYR2K that is equivalent to a batched SYR2K
      $ ./examples/38_syr2k_grouped/38_syr2k_grouped --groups=100 --n=1024 --k=1024 --verbose=true

      # Execute grouped SYR2K and profile with NSight
      $ nv-nsight-cu-cli ./examples/38_syr2k_grouped/38_syr2k_grouped --n=256 --k=256 --verbose=true \
                                                                    --iterations=1 --reference-check=false

*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "cutlass/blas3.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/rank_2k_grouped.h"
#include "cutlass/gemm/kernel/default_rank_2k_grouped.h"
#include "cutlass/gemm/device/rank_2k_grouped.h"
#include "cutlass/gemm/device/rank_2k.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/rank_2k_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double initialization_time_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double initialization_time_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), initialization_time_ms(initialization_time_ms), gflops(gflops),
    status(status), error(error), passed(true) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;
  bool profile_initialization;
  bool sort_problems;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes;

  int alignment;
  int problem_count;
  int iterations;
  int cuda_streams;
  bool verbose;
  float alpha;
  float beta;
  std::string benchmark_path;

  std::string   output_tag;
  std::ofstream output_file;

  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;
  std::vector<GroupScheduleMode> scheduler_modes;

  std::unordered_map<std::string, GroupScheduleMode>
    str_to_scheduler_mode = {
      {"kDeviceOnly", GroupScheduleMode::kDeviceOnly},
      {"kHostPrecompute", GroupScheduleMode::kHostPrecompute}
    };

  struct GroupScheduleModeHash {
    size_t operator()(GroupScheduleMode m) const {
      return static_cast<size_t>(m);
    }
  };

  std::unordered_map<GroupScheduleMode, std::string, GroupScheduleModeHash>
    scheduler_mode_to_str = {
      {GroupScheduleMode::kDeviceOnly, "kDeviceOnly"},
      {GroupScheduleMode::kHostPrecompute, "kHostPrecompute"}
    };

  std::vector<GroupScheduleMode> all_scheduler_modes = {GroupScheduleMode::kDeviceOnly, GroupScheduleMode::kHostPrecompute};

  //
  // Methods
  //

  Options():
    help(false),
    error(false),
    alignment(8),
    reference_check(true),
    profile_initialization(false),
    sort_problems(false),
    problem_count(5),
    iterations(20),
    cuda_streams(0),
    verbose(false),
    alpha(1),
    beta(),
    scheduler_modes({GroupScheduleMode::kDeviceOnly})
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("alignment", alignment, 8);
    cmd.get_cmd_line_argument("groups", problem_count, 5);
    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("streams", cuda_streams, 0);
    cmd.get_cmd_line_argument("verbose", verbose, false);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
    cmd.get_cmd_line_argument("profile-initialization", profile_initialization, false);
    cmd.get_cmd_line_argument("sort-problems", sort_problems, false);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);

    std::vector<std::string> scheduler_mode_strs;
    cmd.get_cmd_line_arguments("scheduler-modes", scheduler_mode_strs);

    if (!scheduler_mode_strs.empty()) {
      scheduler_modes.clear();
      if (scheduler_mode_strs.size() == 1 && scheduler_mode_strs[0] == "all") {
        scheduler_modes = all_scheduler_modes;
      } else {
        for (std::string precomp_str : scheduler_mode_strs) {
          auto it = str_to_scheduler_mode.find(precomp_str);
          if (it != str_to_scheduler_mode.end()) {
            scheduler_modes.push_back(it->second);
          } else if (precomp_str == "all") {
            std::cerr << "Flag --scheduler-modes=all must not contain other scheduler modes in list." << std::endl;
            error = true;
            return;
          } else {
            std::cerr << "Unrecognized scheduler mode '" << precomp_str << "'" << std::endl;
            error = true;
            return;
          }
        }
      }
    }

    std::string output_path;
    cmd.get_cmd_line_argument("tag", output_tag);
    cmd.get_cmd_line_argument("output_file", output_path);

    if (!output_path.empty()) {

      std::ios_base::openmode open_mode = std::ios_base::out;

      std::ifstream input_file(output_path.c_str());

      if (input_file.good()) {
        open_mode = std::ios_base::app;
        input_file.close();
      }

      output_file.open(output_path.c_str(), open_mode);

      if (output_file.good() && open_mode != std::ios_base::app) {
        output_file << "Tag,Provider,Kind,Groups,Runtime,GFLOPs\n";
      }
    }

    // Decide how to initialize the problems
    if (!benchmark_path.empty()) {
      if (!benchmark_problems()) {
        error = true;
        problem_sizes.clear();
        return;
      }
    }
    else {
      randomize_problems(cmd);
    }
  }

  void randomize_problems(cutlass::CommandLine &cmd) {

    //
    // For now, randomly choose the problem sizes.
    //

    int cmd_line_m = -1;
    int cmd_line_n = -1;
    int cmd_line_k = -1;

    cmd.get_cmd_line_argument("m", cmd_line_m);
    cmd.get_cmd_line_argument("n", cmd_line_n);
    cmd.get_cmd_line_argument("k", cmd_line_k);

    // SYR2K is defined via only N and K.
    if (cmd_line_m != -1) {
      std::cerr << "Parameter M is ignored for SYR2K\n";
      error = true;
      return;
    }

    problem_sizes.reserve(problem_count);

    for (int i = 0; i < problem_count; ++i) {
      int n = cmd_line_n;
      int k = cmd_line_k;

      if (n < 1) {
        n = alignment * ((rand() % 256) + 1);
      }

      if (k < 1) {
        k = alignment * ((rand() % 256) + 1);
      }

      // SYR2K is defined only in terms of N and K. Replicate N into
      // the SYR2K-N dimension.
      cutlass::gemm::GemmCoord problem(n, n, k);

      problem_sizes.push_back(problem);
    }
  }

  /// Load a benchmark
  bool benchmark_problems() {
    std::ifstream file(benchmark_path);
    if (!file.good()) {
      return false;
    }

    while (file.good()) {

      int idx = -1;
      std::string extent_str;

      file >> idx >> extent_str;

      if (idx < 0 || extent_str.empty()) {
        break;
      }

      cutlass::gemm::GemmCoord extent;
      std::vector<std::string> tokens;

      cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        int x = std::atoi(tokens.at(i).c_str());

        // round up
        if (x % alignment) {
          x += (alignment - (x % alignment));
        }

        extent.at(i) = x;
      }

      if (extent.product()) {
        problem_sizes.push_back(extent);
      }
    }

    problem_count = int(problem_sizes.size());
    return true;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "38_syr2k_grouped\n\n"
      << "  This example profiles the performance of a 'grouped' SYR2K kernel. This example closely follows 24_gemm_grouped\n\n"
      << "Options:\n\n"
      << "  --help                           If specified, displays this usage statement.\n\n"
      << "  --benchmark=<str>                Executes a benchmark problem size.\n"
      << "  --output_file=<str>              Path to a CSV file to output results. If it exists already, results are appended.\n"
      << "  --tag=<str>                      String tag to prepend to the CSV file.\n"
      << "  --groups=<int>                   Number of individual SYR2K problems (default: --groups=15)\n"
      << "  --m=<int>                        Sets the M dimension for all groups. Otherwise, it is selected randomly\n"
      << "  --n=<int>                        Sets the N dimension for all groups. Otherwise, it is selected randomly\n"
      << "  --k=<int>                        Sets the K dimension for all groups. Otherwise, it is selected randomly\n"
      << "  --alpha=<f32>                    Epilogue scalar alpha (real part)\n"
      << "  --beta=<f32>                     Epilogue scalar beta (real part)\n"
      << "  --scheduler-modes=<str>          List of scheduler modes to be profile for grouped GEMM scheduler (default: --scheduler_modes=kDeviceOnly)\n"
      << "  --iterations=<int>               Number of profiling iterations to perform.\n"
      << "  --reference-check=<bool>         If true, performs reference check.\n"
      << "  --verbose=<bool>                 If true, prints problem sizes and batching structure.\n"
      << "  --profile-initialization=<bool>  If true, profiles the device-level kernel's initialization.\n"
      << "  --sort-problems=<bool>           If true, sorts problem sizes in descending order of SYR2K-K dimension.\n";

    out << "\n\nExamples:\n\n"

      << "# Runs a grouped SYR2K with 100 random problem sizes\n"
      << "$ ./examples/38_syr2k_grouped/38_syr2k_grouped --groups=100\n\n"

      << "# Runs a grouped SYR2K with 100 random problem sizes (with K dimension equal to 1024)\n"
      << "$ ./examples/38_syr2k_grouped/38_syr2k_grouped --groups=100 --k=1024 --verbose=true\n\n"

      << "# Runs a grouped SYR2K that is equivalent to a batched SYR2K\n"
      << "$ ./examples/38_syr2k_grouped/38_syr2k_grouped --groups=100 --n=1024 --k=1024 --verbose=true\n\n"

      << "# Runs a grouped SYR2K with each different scheduler mode\n"
      << "$ ./examples/38_syr2k_grouped/38_syr2k_grouped --scheduler-modes=all\n\n"

      << "# Runs a grouped SYR2K with each different scheduler mode and profiles host-side initialization time\n"
      << "$ ./examples/38_syr2k_grouped/38_syr2k_grouped --scheduler-modes=all --profile-initialization=true\n\n"

      << "# Runs a grouped SYR2K problem given an externally supplied benchmark file. This is a text file in which\n"
      << "# Each line contains a unique group index and an MxNxK triple indicating problemsize. NOTE that the\n"
      << "# GEMM-M and GEMM-N dimensions must match.\n"
      << "#\n"
      << "# For example, assume the following are the contents of 'problems.txt'\n"
      << "#\n"
      << "# 0 256x256x520\n"
      << "# 1 264x264x1024\n"
      << "# 2 48x48x1024\n"
      << "#\n"
      << "$ ./examples/38_syr2k_grouped/38_syr2k_grouped --benchmark=problems.txt\n\n"

      << "# Execute Grouped SYR2K and profile with NSight\n"
      << "$ nv-nsight-cu-cli ./examples/38_syr2k_grouped/38_syr2k_grouped --n=256 --k=256 --verbose=true --iterations=1 --reference-check=false\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds
    int64_t fmas = int64_t();

    for (auto const & problem : problem_sizes) {
      fmas += problem.product();
    }

    // SYR2K is defined as (A x BT) + (B x AT), so the number of FMAs is twice that in a GEMM
    fmas *= 2;

    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Rank2K>
class BaseTestbed {
public:
  //
  // Type definitions
  //

  using ElementA = typename Rank2K::ElementA;
  using ElementB = typename Rank2K::ElementB;
  using ElementC = typename Rank2K::ElementC;
  using ElementAccumulator = typename Rank2K::ElementAccumulator;

  using EpilogueOutputOp = typename Rank2K::Rank2Kkernel::Epilogue::OutputOp;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using LayoutA = typename Rank2K::LayoutA;
  using LayoutB = typename Rank2K::LayoutB;
  using LayoutC = typename Rank2K::LayoutC;

  using MatrixCoord = typename LayoutC::TensorCoord;

  //
  // Data members
  //

  Options & options;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint32_t seed;

  cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;

  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<int64_t> lda_host;
  std::vector<int64_t> ldb_host;
  std::vector<int64_t> ldc_host;
  std::vector<int64_t> ldd_host;

  cutlass::DeviceAllocation<int64_t> lda;
  cutlass::DeviceAllocation<int64_t> ldb;
  cutlass::DeviceAllocation<int64_t> ldc;
  cutlass::DeviceAllocation<int64_t> ldd;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementC> block_D;

  cutlass::DeviceAllocation<ElementA *> ptr_A;
  cutlass::DeviceAllocation<ElementB *> ptr_B;
  cutlass::DeviceAllocation<ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementC *> ptr_D;

  BaseTestbed(
    Options &options_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3080
  ):
    options(options_), init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  int problem_count() const {
    return options.problem_count;
  }

  /// Helper to initialize a tensor view
  template <typename Element>
  void initialize_tensor(
    Element *ptr,
    size_t capacity,
    cutlass::Distribution::Kind dist_kind,
    uint32_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      Element scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<ElementC>::value;

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

      cutlass::reference::device::BlockFillRandomUniform(
        ptr, capacity, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::device::BlockFillRandomGaussian(
        ptr, capacity, seed, Element(), Element(0.5f));
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      // Fill with increasing elements
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(1), Element());
    }
    else {

      // Fill with all 1s
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(), Element(1));
    }
  }

  /// Allocates device-side data
  void allocate() {
    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;
    int64_t total_elements_D = 0;

    lda_host.resize(problem_count());
    ldb_host.resize(problem_count());
    ldc_host.resize(problem_count());
    ldd_host.resize(problem_count());

    for (int32_t i = 0; i < problem_count(); ++i) {

      auto problem = options.problem_sizes.at(i);

      lda_host.at(i) = LayoutA::packed({problem.n(), problem.k()}).stride(0);
      ldb_host.at(i) = LayoutB::packed({problem.n(), problem.k()}).stride(0);
      ldc_host.at(i) = LayoutC::packed({problem.n(), problem.n()}).stride(0);
      ldd_host.at(i) = LayoutC::packed({problem.n(), problem.n()}).stride(0);

      offset_A.push_back(total_elements_A);
      offset_B.push_back(total_elements_B);
      offset_C.push_back(total_elements_C);
      offset_D.push_back(total_elements_D);

      int64_t elements_A = problem.n() * problem.k();
      int64_t elements_B = problem.n() * problem.k();
      int64_t elements_C = problem.n() * problem.n();
      int64_t elements_D = problem.n() * problem.n();

      total_elements_A += elements_A;
      total_elements_B += elements_B;
      total_elements_C += elements_C;
      total_elements_D += elements_D;
    }

    lda.reset(problem_count());
    ldb.reset(problem_count());
    ldc.reset(problem_count());
    ldd.reset(problem_count());

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);
  }

  /// Initializes device-side data
  void initialize() {
    problem_sizes_device.reset(problem_count());
    problem_sizes_device.copy_from_host(options.problem_sizes.data());

    lda.copy_from_host(lda_host.data());
    ldb.copy_from_host(ldb_host.data());
    ldc.copy_from_host(ldc_host.data());
    ldd.copy_from_host(ldd_host.data());

    //
    // Assign pointers
    //

    std::vector<ElementA *> ptr_A_host(problem_count());
    std::vector<ElementB *> ptr_B_host(problem_count());
    std::vector<ElementC *> ptr_C_host(problem_count());
    std::vector<ElementC *> ptr_D_host(problem_count());

    for (int32_t i = 0; i < problem_count(); ++i) {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    }

    ptr_A.reset(problem_count());
    ptr_A.copy_from_host(ptr_A_host.data());

    ptr_B.reset(problem_count());
    ptr_B.copy_from_host(ptr_B_host.data());

    ptr_C.reset(problem_count());
    ptr_C.copy_from_host(ptr_C_host.data());

    ptr_D.reset(problem_count());
    ptr_D.copy_from_host(ptr_D_host.data());

    //
    // Initialize the problems of the workspace
    //

    initialize_tensor(block_A.get(), block_A.size(), init_A, seed * 2021);
    initialize_tensor(block_B.get(), block_B.size(), init_B, seed * 2022);
    initialize_tensor(block_C.get(), block_C.size(), init_C, seed * 2023);

    cutlass::reference::device::BlockFillSequential(
      block_D.get(), block_D.size(), ElementC(), ElementC());
  }

  /// Verifies the result is a SYR2K
  bool verify() {

    bool passed = true;

    for (int32_t i = 0; i < problem_count(); ++i) {
      cutlass::gemm::GemmCoord problem = options.problem_sizes.at(i);

      LayoutA layout_A(lda_host.at(i));
      LayoutB layout_B(ldb_host.at(i));
      LayoutC layout_C(ldc_host.at(i));
      LayoutC layout_D(ldd_host.at(i));

      cutlass::HostTensor<ElementA, LayoutA> host_A(
        typename LayoutA::TensorCoord(problem.n(), problem.k()), /*device_backed=*/false);
      cutlass::HostTensor<ElementB, LayoutB> host_B(
        typename LayoutB::TensorCoord(problem.n(), problem.k()), /*device_backed=*/false);
      cutlass::HostTensor<ElementC, LayoutC> host_C(
        typename LayoutC::TensorCoord(problem.n(), problem.n()), /*device_backed=*/false);
      cutlass::HostTensor<ElementC, LayoutC> host_D(
        typename LayoutC::TensorCoord(problem.n(), problem.n()), /*device_backed=*/false);

      cutlass::device_memory::copy_to_host(host_A.host_data(), block_A.get() + offset_A.at(i), problem.n() * problem.k());
      cutlass::device_memory::copy_to_host(host_B.host_data(), block_B.get() + offset_B.at(i), problem.n() * problem.k());
      cutlass::device_memory::copy_to_host(host_C.host_data(), block_C.get() + offset_C.at(i), problem.n() * problem.n());
      cutlass::reference::host::BlockFillSequential(
        host_D.host_data(), problem.n() * problem.n(), ElementC(), ElementC());

      MatrixCoord extent_C{problem.n(), problem.n()};

      // Reference Rank2K
      cutlass::reference::host::Rank2KComplex<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementC, ElementAccumulator
      >(
        problem,
        (double)options.alpha,
        host_A.host_view(),
        Rank2K::kTransformA,
        host_B.host_view(),
        Rank2K::kTransformB,
        (double)options.beta,
        host_C.host_view(),
        host_D.host_view(),
        ElementAccumulator(0),
        Rank2K::kFillModeC,
        Rank2K::kBlasMode
      );

      // Copy to host memory
      std::vector<ElementC> matrix_D(layout_D.capacity(extent_C));
      cutlass::device_memory::copy_to_host(matrix_D.data(), block_D.get() + offset_D.at(i), matrix_D.size());

      cutlass::TensorView<ElementC, LayoutC> view_D(matrix_D.data(), layout_D, extent_C);
      cutlass::TensorView<ElementC, LayoutC> view_Ref = host_D.host_view();

      // Reference check
      passed = cutlass::reference::host::TensorEquals(view_D, view_Ref);

      if (!passed) {
        std::cerr << "\n***\nError - problem " << i << " failed the QA check\n***\n" << std::endl;
        return passed;
      }
    }

    return passed;
  }
};

template <typename Rank2K>
class TestbedConventional : BaseTestbed<Rank2K> {
public:
  TestbedConventional(
    Options &options_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3080
  ): BaseTestbed<Rank2K>(options_, init_A_, init_B_, init_C_, seed_) {}

  /// Verbose printing of problem sizes
  void print_problem_sizes() {

    // Print groups
    std::cout << this->problem_count() << " groups:\n";

    int32_t idx = 0;
    int64_t total_tiles = 0;

    for (auto const & problem : this->options.problem_sizes) {
      int tiles =
        ((problem.m() + Rank2K::ThreadblockShape::kM - 1) / Rank2K::ThreadblockShape::kM) *
        ((problem.n() + Rank2K::ThreadblockShape::kN - 1) / Rank2K::ThreadblockShape::kN);

      total_tiles += tiles;

      std::cout << "  [" << idx << "]: "
        << problem.m() << "-by-" << problem.n() << "-by-" << problem.k()
        << " (" << tiles << " threadblock tiles)" << "\n";

      ++idx;
    }
    std::cout << std::endl;
  }

  /// Executes a conventional SYR2K kernel.
  Result profile() {
    std::cout << "Conventional Rank2K:\n"
      << "====================================================" << std::endl;

    Result result;
    result.passed = false;

    // Initialize the problem
    this->allocate();
    this->initialize();

    if (this->options.verbose) {
      print_problem_sizes();
    }

    //
    // Create CUDA streams to maximize concurrency of SYR2K kernels
    //
    int32_t effective_streams = (this->options.cuda_streams ? this->options.cuda_streams : 1);
    std::vector<cudaStream_t> cuda_streams;
    char const *provider = "CUTLASS";

    //
    // Warmup run
    //

    if (this->options.cuda_streams) {
      for (int i = 0; i < this->options.cuda_streams; ++i) {
        cudaStream_t stream;

        result.error = cudaStreamCreate(&stream);
        if (result.error != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream." << std::endl;
          return result;
        }
        cuda_streams.push_back(stream);
      }
    }
    else {
      cuda_streams.push_back(nullptr);
    }

    // Use 'D' for the in/out workspace
    this->block_D.copy_from_device(this->block_C.get());

    for (size_t i = 0; i < this->options.problem_sizes.size(); ++i) {
      cutlass::gemm::GemmCoord const & problem = this->options.problem_sizes[i];
      int32_t batch_count = 1;
      int64_t lda = this->lda_host.at(i);
      int64_t ldb = this->ldb_host.at(i);
      int64_t ldc = this->ldc_host.at(i);
      typename Rank2K::ElementA* ptrA = this->block_A.get() + this->offset_A.at(i);
      typename Rank2K::ElementB* ptrB = this->block_B.get() + this->offset_B.at(i);
      typename Rank2K::ElementC* ptrC = this->block_C.get() + this->offset_C.at(i);
      typename Rank2K::ElementC* ptrD = this->block_D.get() + this->offset_D.at(i);

      //
      // Initialize the CUTLASS SYR2K operator
      //

      // Configure the SYR2K arguments
      typename Rank2K::EpilogueOutputOp::Params epilogue_op(this->options.alpha, this->options.beta);

      typename Rank2K::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        batch_count,
        epilogue_op,
        (void const *)ptrA,
        (void const *)ptrB,
        (void const *)ptrC,
        (void       *)ptrD,
        int64_t(),
        int64_t(),
        int64_t(),
        int64_t(),
        int64_t(lda),
        int64_t(ldb),
        int64_t(ldc),
        int64_t(ldc)
      };

      Rank2K rank2k_op;

      cutlass::Status status = rank2k_op.initialize(arguments);

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
        return result;
      }

      status = rank2k_op();

      if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
        return result;
      }
    }

    //
    // Wait for completion
    //

    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
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

    //
    // Wait for completion
    //

    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    // Record an event at the start of a series of SYR2K operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    size_t last_stream_idx = 0;

    for (int iter = 0; iter < this->options.iterations; ++iter) {
      for (size_t i = 0; i < this->options.problem_sizes.size(); ++i) {
        cutlass::gemm::GemmCoord const & problem = this->options.problem_sizes[i];
        int32_t batch_count = 1;
        int64_t lda = this->lda_host.at(i);
        int64_t ldb = this->ldb_host.at(i);
        int64_t ldc = this->ldc_host.at(i);
        typename Rank2K::ElementA* ptrA = this->block_A.get() + this->offset_A.at(i);
        typename Rank2K::ElementB* ptrB = this->block_B.get() + this->offset_B.at(i);
        typename Rank2K::ElementC* ptrC = this->block_C.get() + this->offset_C.at(i);
        typename Rank2K::ElementC* ptrD = this->block_D.get() + this->offset_D.at(i);

        last_stream_idx = (i % effective_streams);

        //
        // Initialize the CUTLASS SYR2K operator
        //

        // Configure the SYR2K arguments
        typename Rank2K::EpilogueOutputOp::Params epilogue_op(this->options.alpha, this->options.beta);

        typename Rank2K::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          problem,
          batch_count,
          epilogue_op,
          (void const *)ptrA,
          (void const *)ptrB,
          (void const *)ptrC,
          (void       *)ptrD,
          int64_t(),
          int64_t(),
          int64_t(),
          int64_t(),
          int64_t(lda),
          int64_t(ldb),
          int64_t(ldc),
          int64_t(ldc)
        };

        Rank2K rank2k_op;

        cutlass::Status status = rank2k_op.initialize(arguments);

        if (status != cutlass::Status::kSuccess) {
          std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
          return result;
        }

        status = rank2k_op(cuda_streams[last_stream_idx]);

        if (status != cutlass::Status::kSuccess) {
          std::cerr << "CUTLASS error on line " << __LINE__ << std::endl;
          return result;
        }
      }
    }

    //
    // Stop profiling loop
    //

    // Record an event when the SYR2K operations have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Wait for work to be completed
    //

    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
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
    result.runtime_ms = double(runtime_ms) / double(this->options.iterations);
    result.gflops = this->options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    for (auto stream : cuda_streams) {
      if (stream) {
        (void)cudaStreamDestroy(stream);
      }
    }

    std::cout << "    " << this->options.problem_sizes.size() << " conventional Rank2Ks launched" << std::endl;
    std::cout << std::endl;
    std::cout << "    " << "Conventional Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << "Conventional  GFLOPS: " << result.gflops << std::endl;

    if (this->options.output_file.good()) {
      this->options.output_file << this->options.output_tag << "," << provider << ",conventional,"
        << this->problem_count() << "," << result.runtime_ms << "," << result.gflops << std::endl;
    }

    result.passed = true;
    return result;
  }
};

template <typename Rank2K_, cutlass::gemm::kernel::GroupScheduleMode GroupScheduleMode_>
class TestbedGrouped : BaseTestbed<Rank2K_> {
public:
  TestbedGrouped(
    Options &options_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3080
  ) : BaseTestbed<Rank2K_>(options_, init_A_, init_B_, init_C_, seed_) {}

  // Redefine Rank2K with different GroupScheduleMode_
    using Rank2Kkernel = typename cutlass::gemm::kernel::DefaultRank2KGrouped<
    typename Rank2K_::ElementA, typename Rank2K_::LayoutA, Rank2K_::kTransformA, Rank2K_::kAlignmentA,
    typename Rank2K_::ElementB, typename Rank2K_::LayoutB, Rank2K_::kTransformB, Rank2K_::kAlignmentB,
    typename Rank2K_::ElementC, typename Rank2K_::LayoutC, Rank2K_::kFillModeC,
    typename Rank2K_::ElementAccumulator,
    typename Rank2K_::OperatorClass,
    typename Rank2K_::ArchTag,
    typename Rank2K_::ThreadblockShape,
    typename Rank2K_::WarpShape,
    typename Rank2K_::InstructionShape,
    typename Rank2K_::EpilogueOutputOp,
    typename Rank2K_::ThreadblockSwizzle,
    Rank2K_::kStages,
    typename Rank2K_::Operator::ArchMmaOperator::Operator,
    Rank2K_::kBlasMode,
    GroupScheduleMode_>::Rank2Kkernel;

  using Rank2K = cutlass::gemm::device::Rank2KGrouped<Rank2Kkernel>;

  /// Verbose printing of problem sizes
  void print_problem_sizes() {

    // Print groups
    std::cout << this->problem_count() << " groups:\n";

    int32_t idx = 0;
    int64_t total_tiles = 0;

    for (auto const & problem : this->options.problem_sizes) {
      int tiles = Rank2K::problem_tile_count(problem);
      total_tiles += tiles;

      std::cout << "  [" << idx << "]: "
        << problem.m() << "-by-" << problem.n() << "-by-" << problem.k()
        << " (" << tiles << " threadblock tiles)" << "\n";

      ++idx;
    }
    std::cout << std::endl;
  }

  /// Sort problems in descending order of problem-K dimension
  void sort_problems() {
    Rank2K::sort_problems(this->options.problem_count,
                          this->options.problem_sizes.data(),
                          this->lda_host.data(),
                          this->ldb_host.data(),
                          this->ldc_host.data(),
                          this->ldd_host.data(),
                          this->offset_A.data(),
                          this->offset_B.data(),
                          this->offset_C.data(),
                          this->offset_D.data());
  }

  /// Executes a grouped kernel and measures runtime.
  Result profile() {
    std::string sched_mode = this->options.scheduler_mode_to_str.find(GroupScheduleMode_)->second;
    std::cout << std::endl;
    std::cout << "Grouped Rank2K (CUTLASS) with mode " << sched_mode << ":\n"
      << "====================================================" << std::endl;

    Result result;

    int threadblock_count = Rank2K::sufficient(this->options.problem_sizes.data(), this->options.problem_count);

    // Early exit
    if (!threadblock_count) {
      std::cout << "Active CUDA device lacks hardware resources to run CUTLASS Grouped SYR2K kernel." << std::endl;
      return result;
    }

    result.passed = false;

    // Initialize the problem
    this->allocate();
    if (this->options.sort_problems) {
      sort_problems();
    }
    this->initialize();

    if (this->options.verbose) {
      print_problem_sizes();
    }

    // Configure the Rank2K arguments
    typename Rank2K::EpilogueOutputOp::Params epilogue_op(this->options.alpha, this->options.beta);

    // Configure Rank2K arguments
    typename Rank2K::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      this->problem_sizes_device.get(),
      this->problem_count(),
      threadblock_count,
      epilogue_op,
      this->ptr_A.get(),
      this->ptr_B.get(),
      this->ptr_C.get(),
      this->ptr_D.get(),
      this->lda.get(),
      this->ldb.get(),
      this->ldc.get(),
      this->ldd.get(),
      this->options.problem_sizes.data()
    );

    // Initialize the Rank2K object
    Rank2K rank2k{};
    size_t workspace_size = rank2k.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

    result.status = rank2k.initialize(args, workspace.get());

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize CUTLASS Grouped Rank2K kernel." << std::endl;
      return result;
    }

    // Run the grouped Rank2K object
    result.status = rank2k.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Grouped Rank2K kernel." << std::endl;
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
    if (this->options.reference_check) {
      result.passed = this->verify();
    }

    //
    // Warm-up run of the grouped Rank2K object
    //
    result.status = rank2k.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Grouped Rank2K kernel." << std::endl;
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

    // Record an event at the start of a series of SYR2K operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < this->options.iterations; ++iter) {
      rank2k();
    }

    //
    // Stop profiling loop
    //

    // Record an event when the Rank2K operations have been launched.
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
    result.runtime_ms = double(runtime_ms) / double(this->options.iterations);
    result.gflops = this->options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    // Optionally profile initialization
    if (this->options.profile_initialization) {
      // Warm up
      rank2k.initialize(args, workspace.get());

      auto start_time = std::chrono::high_resolution_clock::now();
      for (int32_t i = 0; i < this->options.iterations; ++i) {
        rank2k.initialize(args, workspace.get());
      }
      auto end_time = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> duration = end_time - start_time;
      duration /= double(this->options.iterations);
      result.initialization_time_ms = duration.count();
    }

    int64_t total_tiles = Rank2K::group_tile_count(args);
    std::cout << "    " << total_tiles << " total threadblock tiles." << std::endl;

    std::cout << std::endl;
    std::cout << "    " << "Grouped Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << "Grouped  GFLOPs: " << result.gflops << std::endl;
    if (this->options.profile_initialization) {
      std::cout << "    " << "Init    Runtime: " << result.initialization_time_ms << " ms" << std::endl;
    }

    if (this->options.output_file.good()) {
      this->options.output_file << this->options.output_tag << ",CUTLASS,grouped-" << sched_mode << ","
        << this->problem_count() << "," << result.runtime_ms << "," << result.gflops << std::endl;
    }

    std::cout << "\nPassed\n";

    return result;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

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
      << "CUTLASS's Grouped Rank2K example requires a GPU of NVIDIA's Ampere Architecture or "
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

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Define the Grouped and Conventional Rank2K types
  //

  using ElementA = double;
  using ElementB = double;
  using ElementOutput = double;
  using ElementAccumulator = double;
  const cutlass::FillMode kFillModeC = cutlass::FillMode::kLower;
  const int kAlignmentA = 1;
  const int kAlignmentB = 1;
  const cutlass::ComplexTransform kTransformA = cutlass::ComplexTransform::kNone;
  const cutlass::ComplexTransform kTransformB = cutlass::ComplexTransform::kNone;

  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;

  using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>;

  // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
  // This parameter is passed in at present to match the APIs of other kernels. The parameter
  // is unused within the kernel.
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  const int kStages = 4;
  const bool kSplitKSerial = false;
  using Operator = cutlass::arch::OpMultiplyAdd;
  const cutlass::BlasMode kBlasMode = cutlass::BlasMode::kSymmetric;

  // Define a grouped Rank2K kernel with all template parameters set except
  // for scheduling mode. This will be used as the template for all scheduling
  // modes executed.
  using Rank2Kkernel = typename cutlass::gemm::kernel::DefaultRank2KGrouped<
    ElementA, LayoutA, kTransformA, kAlignmentA,
    ElementB, LayoutB, kTransformB, kAlignmentB,
    ElementOutput, LayoutC, kFillModeC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    Operator,
    kBlasMode>::Rank2Kkernel;

  using Rank2KGrouped = cutlass::gemm::device::Rank2KGrouped<Rank2Kkernel>;

  // Rank2k operator
  using Rank2KConventional = cutlass::gemm::device::Rank2K<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementOutput, LayoutC, kFillModeC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    kAlignmentA,
    kAlignmentB,
    kSplitKSerial,
    Operator,
    kTransformA,
    kTransformB,
    kBlasMode
  >;

  //
  // Profile it
  //

  TestbedConventional<Rank2KConventional> testbed(options);

  Result result = testbed.profile();
  if (!result.passed) {
    std::cout << "Profiling CUTLASS conventional Rank2K has failed.\n";
    std::cout << "\nFailed\n";
    return -1;
  }

  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;
  for (GroupScheduleMode mode : options.scheduler_modes) {
    Result result;
    switch (mode) {
      case GroupScheduleMode::kDeviceOnly:
        {
          TestbedGrouped<Rank2KGrouped, GroupScheduleMode::kDeviceOnly> runner(options);
          result = runner.profile();
          break;
        }
      case GroupScheduleMode::kHostPrecompute:
        {
          TestbedGrouped<Rank2KGrouped, GroupScheduleMode::kHostPrecompute> runner(options);
          result = runner.profile();
          break;
        }
    }

    if (result.error != cudaSuccess) {
      return 1;
    }

    // Override verbose flag to avoid printing duplicate information for each scheduling mode
    options.verbose = false;
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
