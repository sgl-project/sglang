/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file A MLA (Multi-Head Latent Attention) inference kernel sample for the
          NVIDIA Blackwell Architecture.
*/

#include <iostream>
#include <random>
#include <regex>
#include <cmath>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "reference/fmha_mla_reference.hpp"
#include "reference/reference_abs_error.hpp"

#include "device/sm100_mla.hpp"
#include "kernel/sm100_mla_tile_scheduler.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cutlass::fmha::kernel;

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class InitStyle {
  kOne, kLinearStride128, kLinearStride1, kRandom, kRandomLarge, kNone
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help = false;
  bool error = false;

  int b = 1;
  int k = 256;
  int split_kv = -1; // number of split along k dim.
  bool is_var_split_kv = false;
  int max_split_kv = 16;
  int page = -1;
  float spread = 0.2f;
  int iterations = 3;
  bool verify = false;
  bool verbose = false;
  bool is_fused_reduction = false;

  int sm_count = 0;

  std::string kernel_filter;

  InitStyle init_style_q = InitStyle::kRandom;
  InitStyle init_style_c = InitStyle::kRandom;

  static void get_init_style_argument(cutlass::CommandLine& cmd, const char* name, InitStyle& dst, InitStyle const& src) {
    std::string s;
    cmd.get_cmd_line_argument(name, s, s);
    if (s.empty()) {
      dst = src;
    }
    else {
      if (s == "r") {
        dst = InitStyle::kRandom;
      }
      else if (s == "l") {
        dst = InitStyle::kRandomLarge;
      }
      else if (s == "1") {
        dst = InitStyle::kOne;
      }
      else if (s == "d") {
        dst = InitStyle::kLinearStride1;
      }
      else if (s == "s") {
        dst = InitStyle::kLinearStride128;
      }
      else if (s == "n") {
        dst = InitStyle::kNone;
      }
      else {
        std::cout << "Error: " << s << " is not a valid input type.\n";
        std::exit(-1);
      }
    }
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    Options defaults;

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("k", k, -1);
    if (k == -1) k = defaults.k;

    cmd.get_cmd_line_argument("b", b, -1);
    if (b == -1) b = 16384 / k;
    if (b == 0) b = 1;

    cmd.get_cmd_line_argument("split_kv", split_kv, defaults.split_kv);
    if (split_kv == 0) {
      split_kv = 1;
    }
    cmd.get_cmd_line_argument("page", page, defaults.page);
    cmd.get_cmd_line_argument("spread", spread, defaults.spread);
    is_var_split_kv = cmd.check_cmd_line_flag("var_split_kv");
    if (page == -1) {
      is_var_split_kv = false;
    }
    cmd.get_cmd_line_argument("max_split_kv", max_split_kv, defaults.max_split_kv);
    if (is_var_split_kv == true) {
      split_kv = max_split_kv;
    }
    is_fused_reduction = cmd.check_cmd_line_flag("fuse_reduction");
    if (split_kv == 1) {
      is_fused_reduction = false;
    }
    cmd.get_cmd_line_argument("iterations", iterations, defaults.iterations);
    verify = cmd.check_cmd_line_flag("verify");
    verbose = cmd.check_cmd_line_flag("verbose");
    cmd.get_cmd_line_argument("sm-count", sm_count, defaults.sm_count);
    
    get_init_style_argument(cmd, "init-style", init_style_q, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_c, defaults.init_style_c);
    get_init_style_argument(cmd, "init-style-q", init_style_q, init_style_q);
    get_init_style_argument(cmd, "init-style-c", init_style_c, init_style_c);

    cmd.get_cmd_line_argument("kernel-filter", kernel_filter, defaults.kernel_filter);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "77_blackwell_mla\n\n"
      << "  This example showcases the use of CUTLASS for fused multi-head latent\n"
      << "  attention kernels targeting NVIDIA's Blackwell architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --b=<int>                   Sets the B extent\n"
      << "  --k=<int>                   Sets the K extent\n"
      << "  --page=<int>                Enables paging and sets the page size\n"
      << "  --iterations=<int>          Benchmarking iterations\n"
      << "  --spread=<float>            Relative spread away from K for paging\n"
      << "  --split_kv=<int>            Split KV factor\n"
      << "  --fused_reduction           Fuse the reduction operation\n"
      << "  --var_split_kv              Use varying split KV factor\n"
      << "  --verify                    Verify results\n"
      << "  --verbose                   Print smem and execution time per kernel\n"
      << " --sm-count                   Sets SM count rather than querying it\n"
      << " --kernel-filter=<filter>     Sets regexp to match kernel against\n"
      << "\n";

    return out;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
void initialize_block(
    DeviceAllocation<Element>& block,
    uint64_t seed=2023, InitStyle init_style = InitStyle::kRandom) {

  switch (init_style) {
    case InitStyle::kOne: {
      cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, (Element) 1, (Element) 1);
      break;
    }
    case InitStyle::kRandom: {
      cutlass::reference::device::BlockFillRandomGaussian(
        block.get(), block.size(), seed, (Element) -1, (Element) 1);
      break;
    }
    case InitStyle::kRandomLarge: {
      cutlass::reference::device::BlockFillRandomGaussian(
        block.get(), block.size(), seed, (Element) -1, (Element) 100);
      break;
    }
    case InitStyle::kLinearStride1: {
      std::vector<Element> data(block.size());
      for (size_t i = 0; i < block.size() / 128; i ++) {
        for (int j = 0; j < 128; j++) {
          data[j + 128*i] = static_cast<Element>((double) (j % 4));
        }
      }
      block.copy_from_host(data.data(), data.size());
      break;
    }
    case InitStyle::kLinearStride128: {
      std::vector<Element> data(block.size());
      for (size_t i = 0; i < block.size() / 64; i ++) {
        for (int j = 0; j < 64; j++) {
          data[j + 64*i] = static_cast<Element>((double) (i % 9));
        }
      }
      block.copy_from_host(data.data(), data.size());
      break;
    }
    case InitStyle::kNone: {
      break;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct ExampleResult {
  bool passed = false;
  bool verified = false;
  float runtime_ms = 0;
  double tflops_tc_s = 0;
  double tbytes_s = 0;
  size_t smem_size = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

template<bool v>
struct IsPersistent {
  static const bool value = v;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class TileShape,
  class PersistenceOption = IsPersistent<true>
>
struct Runner {

#ifdef FP8
  using Element = cutlass::float_e4m3_t;
#elif FP16
  using Element = cutlass::half_t;
#else
  #error "Must either define FP8 or FP16"
#endif

  using ElementAcc = float;
  using ElementOut = cutlass::half_t;

  using TileShapeH = cute::tuple_element_t<0, TileShape>;
  using TileShapeD = cute::tuple_element_t<2, TileShape>;

  // H K (D_latent D_rope) B
  using ProblemShape = cute::tuple<TileShapeH, int, TileShapeD, int>;
  
  using StrideQ = cute::tuple<int64_t, _1, int64_t>;  // H D B
  using StrideK = cute::tuple<int64_t, _1, int64_t>;  // K D B
  using StrideO = StrideK;                            // H D B
  using StrideLSE = cute::tuple<_1, int>;             // H B

  using TileScheduler = std::conditional_t<
      PersistenceOption::value,
      Sm100MlaPersistentTileScheduler,
      Sm100MlaIndividualTileScheduler
  >;

  using Kernel = cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized<
    TileShape, Element, ElementAcc, ElementOut, ElementAcc, TileScheduler
  >;
  using Operation = cutlass::fmha::device::MLA<Kernel>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q_latent;
  StrideK stride_C_latent;
  StrideQ stride_Q_rope;
  StrideK stride_K_rope;
  StrideO stride_O;
  StrideLSE stride_LSE;
  StrideLSE stride_PT;
  
  uint64_t seed = 0;

  int page_size = -1;
  int page_count = -1;

  // We allocate Q and C as first latent, then rope
  // This means that we offset the pointer by HeadDim_latent to get the rope
  // portion
  DeviceAllocation<Element> block_Q;
  DeviceAllocation<Element> block_C;
  DeviceAllocation<ElementOut> block_O;
  DeviceAllocation<int> block_seq;
  DeviceAllocation<int> block_PT;
  DeviceAllocation<int> block_split_kv;
  DeviceAllocation<int> block_accum_split_len; 
  DeviceAllocation<ElementAcc> block_LSE;
  DeviceAllocation<ElementOut> block_ref_O;
  DeviceAllocation<ElementAcc> block_ref_LSE;
   
  ElementAcc scale;

  //
  // Methods
  //

  bool verify(const ProblemShape& problem_shape) {
    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    int page_K = K;
    int page_B = B;
    if (block_PT.get() != nullptr) {
      page_K = page_size;
      page_B = page_count;
    }

    Tensor mQ_latent = make_tensor(make_gmem_ptr(block_Q.get()),
      cute::make_tuple(H, D_latent, B),
      stride_Q_latent);

    Tensor mQ_rope = make_tensor(make_gmem_ptr(block_Q.get() + D_latent),
      cute::make_tuple(H, D_rope, B),
      stride_Q_rope);

    Tensor mC_latent = make_tensor(make_gmem_ptr(block_C.get()),
      cute::make_tuple(page_K, D_latent, page_B),
      stride_C_latent);

    Tensor mK_rope = make_tensor(make_gmem_ptr(block_C.get() + D_latent),
      cute::make_tuple(page_K, D_rope, page_B),
      stride_K_rope);

    Tensor mO = make_tensor(make_gmem_ptr(block_ref_O.get()),
      cute::make_tuple(H, D_latent, B),
      stride_O);

    Tensor mLSE = make_tensor(make_gmem_ptr(block_ref_LSE.get()),
      cute::make_tuple(H, B),
      stride_LSE);

    Tensor mSeq = make_tensor(make_gmem_ptr(static_cast<int*>(block_seq.get())), make_shape(B));
    Tensor mPT = make_tensor(make_gmem_ptr(static_cast<int*>(block_PT.get())), make_shape(ceil_div(K, page_size), B), stride_PT);

    fmha_mla_reference(problem_shape, mSeq, mPT, mQ_latent, mQ_rope, mC_latent, mK_rope, mO, mLSE, scale);

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    const double kMaxDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-2;
    const double kMeanDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-3;

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    double max_diff = 0;
    double mean_diff = 0;
    reference_abs_diff(block_O, block_ref_O, max_diff, mean_diff);

    bool passed_O = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_O) {
      std::cerr << "failed O: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    bool passed_LSE = true;
    reference_abs_diff(block_LSE, block_ref_LSE, max_diff, mean_diff);

    passed_LSE = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if ( ! passed_LSE) {
      std::cerr << "failed LSE: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    return passed_O && passed_LSE;
  }

  ProblemShape initialize(const Options& options) {
    auto problem_shape = cute::make_tuple(TileShapeH{}, options.k, TileShapeD{}, options.b);

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    // the scale is based on the non-absorbed sizes, change as appropriate
    // we can't determine this parameter from the info we have, it's an input
    int D_non_latent = 128;
    scale = static_cast<decltype(scale)>(1.0 / sqrt(1.0 * (D_non_latent + D_rope)));
    // Shape (H, D, B)
    stride_Q_latent = cute::make_tuple(static_cast<int64_t>(0 + D_latent + D_rope), _1{}, static_cast<int64_t>(H * (0 + D_latent + D_rope)));
    stride_Q_rope = stride_Q_latent;
    stride_O = cute::make_tuple(static_cast<int64_t>(0 + D_latent), _1{}, static_cast<int64_t>(0 + H * D_latent));
    stride_LSE = cute::make_tuple(_1{}, 0 + H);

    block_Q.reset(static_cast<size_t>(options.b) * H * (D_latent + D_rope));
    block_O.reset(static_cast<size_t>(options.b) * H * D_latent);
    block_LSE.reset(static_cast<size_t>(options.b) * H);
    block_ref_O.reset(static_cast<size_t>(options.b) * H * D_latent);
    block_ref_LSE.reset(static_cast<size_t>(options.b) * H);

    if (options.page == -1) {

      stride_C_latent = cute::make_tuple(static_cast<int64_t>(0 + D_latent + D_rope), _1{}, static_cast<int64_t>(options.k) * (D_latent + D_rope));
      stride_K_rope = stride_C_latent;

      block_C.reset(static_cast<size_t>(options.b) * options.k * (D_latent + D_rope));

    }
    else {
      
      float spread = options.spread;
      int max_K = static_cast<int>((1 + spread) * K);
      int min_K = static_cast<int>((1 - spread) * K);
      page_size = options.page;
      page_count = B * ceil_div(max_K, page_size);
      stride_PT = cute::make_stride(_1{}, page_count);

      std::vector<int> host_seq(B);
      std::vector<int> host_PT(page_count * B);

      for (int i = 0; i < B; i++) {
        int seq = min_K + rand() % (max_K - min_K + 1);
        host_seq[i] = seq;
        for (int j = 0; j < ceil_div(seq, page_size); j++) {
          host_PT[page_count * i + j] = i + j * B;
        }
      }

      block_seq.reset(host_seq.size());
      block_seq.copy_from_host(host_seq.data(), host_seq.size());
      block_PT.reset(host_PT.size());
      block_PT.copy_from_host(host_PT.data(), host_PT.size());

      get<1>(problem_shape) = max_K;

      stride_C_latent = cute::make_tuple(static_cast<int64_t>(0 + D_latent + D_rope), _1{}, page_size * static_cast<int64_t>((D_latent + D_rope)));
      stride_K_rope = stride_C_latent;

      block_C.reset(page_count * page_size * static_cast<int64_t>((D_latent + D_rope)));

      if (options.is_var_split_kv == true) {
        std::vector<int> host_split_kv(B);
        for(int i = 0; i < B; ++i) {
          auto len = host_seq[i];
	  int split = ceil_div(options.max_split_kv, ceil_div(max_K, len));
	  host_split_kv[i] = split;
        }
	block_split_kv.reset(B);
        block_split_kv.copy_from_host(host_split_kv.data(), host_split_kv.size());
      } 
    }

    initialize_block(block_Q, seed + 2023, options.init_style_q);
    initialize_block(block_C, seed + 2022, options.init_style_c);

    return problem_shape;
  }

  ExampleResult run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {

    ProblemShape problem_shape = initialize(options);

    auto [H, K, D, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    typename Operation::Arguments arguments{
      problem_shape,
      { scale,
        block_Q.get(), stride_Q_latent,
        block_Q.get() + D_latent, stride_Q_rope,
        block_C.get(), stride_C_latent,
        block_C.get() + D_latent, stride_K_rope,
        block_seq.get(),
        block_PT.get(), stride_PT,
        page_count, page_size},
      { block_O.get(), 
        stride_O,
        block_LSE.get(),
        stride_LSE}, 
      hw_info,
      options.split_kv,
      options.is_var_split_kv ? block_split_kv.get() : nullptr,
      options.is_fused_reduction
    };
    if (options.split_kv < 0 && !options.is_var_split_kv) {
      Operation::set_split_kv(arguments);
    }

    Operation op;

    ExampleResult example_result;

    example_result.smem_size = Operation::Kernel::SharedStorageSize;

    size_t workspace_size = 0;
    workspace_size = Operation::get_workspace_size(arguments);
    DeviceAllocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
    }

    status = op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
    }
    // Run
    status = op.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result = cudaEventCreate(&event);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result) << std::endl;
        return example_result;
      }
    }

    // Record an event at the start of a series of GEMMs
    result = cudaEventRecord(events[0]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    for (int i = 0; i < options.iterations; i++) {
      status = op.run();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: " 
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return example_result;
      } 
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMMs are complete
    result = cudaEventRecord(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Wait for work on the device to complete.
    result = cudaEventSynchronize(events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    runtime_ms /= static_cast<float>(options.iterations);

    double flops = 1.0;
    flops *= B;
    flops *= K;
    flops *= H;
    flops *= 2.0;
    flops *= (2.0 * D_latent + D_rope);

    double bytes_q = sizeof(Element);
    bytes_q *= B;
    bytes_q *= H;
    bytes_q *= (D_latent + D_rope);
    double bytes_c = sizeof(Element);
    bytes_c *= B;
    bytes_c *= options.k;  // K may be max_K here
    bytes_c *= (D_latent + D_rope);
    double bytes_o = sizeof(ElementOut);
    bytes_o *= B;
    bytes_o *= H;
    bytes_o *= D_latent;
    double bytes = bytes_q + bytes_c + bytes_o;

    double tflops_s = flops * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
    double tbytes_s = bytes * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
    example_result.tflops_tc_s = tflops_s;
    example_result.tbytes_s = tbytes_s;
    example_result.runtime_ms = runtime_ms;

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return example_result;
    }

    // Verify that the result is correct
    bool passed = true;
    if (options.verify) {
      passed = verify(problem_shape);
      if (passed) example_result.verified = true;
    }
    
    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
      return example_result;
    }

    example_result.passed = true;

    return example_result;
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main_result = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, ExampleResult result, bool verbose) {
  std::ios fmt(nullptr);
  fmt.copyfmt(std::cout);
  std::cout << (result.passed ? (result.verified ? " [OK]  " : " [--] ") : "[FAIL] ");
  if (! result.passed) {
    main_result = -1;
  }
  std::cout << std::setw(32) << std::left << description;
  std::cout.copyfmt(fmt);
  std::cout << " : " << result.tflops_tc_s << " TFLOPS/s " << result.tbytes_s << " TB/s" << std::endl;
  if (verbose) {
    std::cout << "       t=" << result.runtime_ms * 1e3 << " us, "
        "smem=" << result.smem_size << "b" << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void run_mla(Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, const char* name, auto... kernel_options) {
    if ((! options.kernel_filter.empty()) && (! std::regex_search(name, std::basic_regex(options.kernel_filter)))) {
        return;
    }
    Runner<decltype(shape), decltype(kernel_options)...> runner;
    auto result = runner.run(options, hw_info);
    print_result(name, result, options.verbose);
  };

  using NumHeads = _128;
  using HeadDimLatent = _512;
  using HeadDim = Shape<HeadDimLatent, _64>;

  std::cout << "###### B " << options.b << " MLA H " << 0 + NumHeads{} << " ";
  std::cout << "D_rope " << 0 + get<1>(HeadDim{}) << " D_latent " << 0 + get<0>(HeadDim{}) << " ";
  std::cout << "Q 1 K " << options.k << " Gen None ";
  std::cout << "Split " << options.split_kv << " Gen None ";
  std::cout << "#SM " << hw_info.sm_count << std::endl;

  using Blocking = _128;
  std::string name = std::to_string((int) NumHeads{}) + "x" + std::to_string((int) Blocking{});
  std::string individual = " individual";
  std::string persistent = " persistent";
#if FP8
  name += " fp8";
  // Persistent Tile Scheduler
  run(Shape<NumHeads, Blocking, HeadDim>{}, (name + persistent).c_str(), IsPersistent<true>{});
  // Individual Tile Scheduler
  if (!options.is_fused_reduction || options.split_kv == 1) {
    run(Shape<NumHeads, Blocking, HeadDim>{}, (name + individual).c_str(), IsPersistent<false>{});
  }
#elif FP16
  name += " fp16";
  // Persistent Tile Scheduler
  run(Shape<NumHeads, Blocking, HeadDim>{}, (name + persistent).c_str(), IsPersistent<true>{});
  // Individual Tile Scheduler
  if (!options.is_fused_reduction || options.split_kv == 1) {
    run(Shape<NumHeads, Blocking, HeadDim>{}, (name + individual).c_str(), IsPersistent<false>{});
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////


int main_single(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 12 || props.major != 10) {
    std::cout
      << "This example requires a GPU of NVIDIA's Blackwell Architecture "
      << "(compute capability major 10) and CUDA 12.8 or greater.\n";
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

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of SMs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  if (options.sm_count == 0) {
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  }
  else {
    hw_info.sm_count = options.sm_count;
  }

  run_mla(options, hw_info);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  std::vector<std::string> full_arguments(args, args + argc);

  bool recursed = false;
  for (size_t i = 1; i < full_arguments.size(); i++) {
    if (full_arguments[i].find(',') != std::string::npos) {
      auto arg = full_arguments[i];
      size_t eq_pos = arg.find('=');
      std::string prefix = eq_pos == std::string::npos ? "" : arg.substr(0, eq_pos+1);
      std::string rest = eq_pos == std::string::npos ? arg : arg.substr(eq_pos+1);
      for (;;) {
        size_t comma_pos = rest.find(',');
        std::string current = rest.substr(0, comma_pos);
        full_arguments[i] = prefix + current;
        std::vector<const char*> next_args;
        for (auto& elem : full_arguments) { next_args.push_back(elem.data()); }
        main(argc, next_args.data());
        if (comma_pos == std::string::npos) break;
        rest = rest.substr(comma_pos+1);
      }
      recursed = true;
      break;
    }
  }

  if (! recursed) {
    main_single(argc, args);
  }

  return main_result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
