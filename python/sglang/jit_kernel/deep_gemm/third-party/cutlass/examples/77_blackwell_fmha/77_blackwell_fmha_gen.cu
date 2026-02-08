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
/*! \file
    \brief Example implementation of fused multi-head attention for the NVIDIA Blackwell SM100
    architecture using CUTLASS 3.

    MQA/GQA
    -------

    The head dimension can be represented as a tuple, where the K/V strides in the
    first dimension is zero. This has the effect of MQA or GQA.
    * MHA is (head_size:head_stride).
    * MQA is (head_size:head_stride) in Q and (head_size:_0) in K and V.
    * GQA is (grouped_heads,heads_kv):(head_stride,grouped_heads*head_stride) in Q
      and (grouped_heads,heads_kv):(0,head_stride) in K and V

    Example usage:
      $ ./examples/77_blackell_fmha/77_blackell_fmha_gen_fp8 \
            --b=2048 --h=2048 --d=2048 --k=2048
*/

#define DSHOW(x) print(#x ": "); print(x); print("\n");
#define DSHOWT(x) print(#x ": "); print_tensor(x); print("\n");

#include <iostream>
#include <random>
#include <regex>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "reference/fmha_fwd_gen_reference.hpp"
#include "reference/reference_abs_error.hpp"

#include "device/fmha.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_gen_mainloop_warpspecialized.hpp"
#include "collective/sm100_fmha_gen_epilogue_warpspecialized.hpp"
#include "kernel/sm100_fmha_gen_kernel_warpspecialized.hpp"
#include "kernel/fmha_tile_scheduler.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class InitStyle {
  kZero, kOne, kLinearStride128, kLinearStride1, kRandom, kNone
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help = false;
  bool error = false;

  int b = 1;
  int h = 1;
  int h_k = 1;
  int k = 512;
  int d = 128;
  int iterations = 3;
  bool verify = false;
  bool verbose = false;
  bool remap = false;
  bool varlen = false;
  bool cache_only = false;

  int sm_count = 0;

  std::string kernel_filter;
  bool clear_cache = false;

  InitStyle init_style_q = InitStyle::kRandom;
  InitStyle init_style_cache_k = InitStyle::kRandom;
  InitStyle init_style_cache_v = InitStyle::kRandom;
  InitStyle init_style_new_k = InitStyle::kRandom;
  InitStyle init_style_new_v = InitStyle::kRandom;

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
      else if (s == "0") {
        dst = InitStyle::kZero;
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

    cmd.get_cmd_line_argument("d", d, defaults.d);
    cmd.get_cmd_line_argument("h", h, -1);
    if (h == -1) h = 2048 / d;

    cmd.get_cmd_line_argument("h_k", h_k, -1);
    if (h_k == -1) h_k = h;

    cmd.get_cmd_line_argument("k", k, defaults.k);

    cmd.get_cmd_line_argument("b", b, -1);
    if (b == -1) b = 16384 / k;
    if (b == 0) b = 1;

    cmd.get_cmd_line_argument("iterations", iterations, defaults.iterations);
    verify = cmd.check_cmd_line_flag("verify");
    verbose = cmd.check_cmd_line_flag("verbose");
    varlen = cmd.check_cmd_line_flag("varlen");
    remap = cmd.check_cmd_line_flag("remap");
    cache_only = cmd.check_cmd_line_flag("cache-only");
    cmd.get_cmd_line_argument("sm-count", sm_count, defaults.sm_count);

    get_init_style_argument(cmd, "init-style", init_style_q, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_cache_k, defaults.init_style_cache_k);
    get_init_style_argument(cmd, "init-style", init_style_cache_v, defaults.init_style_cache_v);
    get_init_style_argument(cmd, "init-style", init_style_new_k, defaults.init_style_new_k);
    get_init_style_argument(cmd, "init-style", init_style_new_v, defaults.init_style_new_v);
    get_init_style_argument(cmd, "init-style-q", init_style_q, init_style_q);
    get_init_style_argument(cmd, "init-style-cache-k", init_style_cache_k, init_style_cache_k);
    get_init_style_argument(cmd, "init-style-cache-v", init_style_cache_v, init_style_cache_v);
    get_init_style_argument(cmd, "init-style-new-k", init_style_new_k, init_style_new_k);
    get_init_style_argument(cmd, "init-style-new-v", init_style_new_v, init_style_new_v);

    clear_cache = cmd.check_cmd_line_flag("clear-cache");

    cmd.get_cmd_line_argument("kernel-filter", kernel_filter, defaults.kernel_filter);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "77_blackwell_fmha_gen\n\n"
      << "  This example showcases the use of CUTLASS's collective operation builders to easily construct\n"
      << "  fused multi-head attention forward-pass gen-phase kernels targeting NVIDIA's Blackwell architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --b=<int>                   Sets the B extent\n"
      << "  --h=<int>                   Sets the H extent\n"
      << "  --h_k=<int>                 Sets the H_K/V extent (for GQA/MQA)\n"
      << "  --k=<int>                   Sets the K extent (sampled around this length)\n"
      << "  --d=<int>                   Sets the D extentn"
      << "  --iterations=<int>          Benchmarking iterations\n"
      << "  --verify                    Verify results\n"
      << "  --verbose                   Print smem and execution time per kernel\n"
      << "  --remap                     Enables batch index remapping\n"
      << "  --cache-only                Only use data from KV cache, no reading or inserting new entry\n"
      << "  --varlen                    Varies sequence length between cache entries\n"
      << "  --sm-count                  Sets SM count rather than querying it\n"
      << "  --clear-cache               Clears the cache before benchmarking runs\n"
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
    case InitStyle::kZero: {
      cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, (Element) 0, (Element) 0);
      break;
    }
    case InitStyle::kOne: {
      cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, (Element) 1, (Element) 1);
      break;
    }
    case InitStyle::kRandom: {
      cutlass::reference::device::BlockFillRandomGaussian(
        block.get(), block.size(), seed, (Element) 0, (Element) 1);
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
      for (size_t i = 0; i < block.size() / 128; i ++) {
        for (int j = 0; j < 128; j++) {
          data[j + 128*i] = static_cast<Element>((double) (i % 4));
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
  bool supported = false;
  bool passed = false;
  bool verified = false;
  float runtime_ms = 0;
  double tflops_tc_s = 0;
  double tops_exp2_s = 0;
  double tbytes_s = 0;
  size_t smem_size = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

struct ClearCache {
  const int size = 1024 * 1024 * 1024 / 4;
  DeviceAllocation<float> data;
  bool active = false;

  ClearCache() = default;

  void set_active(bool the_active) {
    active = the_active;
    if (active) {
      data.reset(size);
    }
    else {
      data.reset(0);
    }
  }

  void operator ()() {
    if (active) {
      initialize_block(data, 0x49314, InitStyle::kRandom);
    }
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class KernelType {
  UMMA_P, UMMA_I
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

template<KernelType kKernelType, class TileShape, class ThreadShape>
struct ExampleRunner {

  using Element = cutlass::float_e5m2_t;
  using ElementAcc = float;
  using ElementOut = cutlass::half_t;

  using ProblemShape = Shape<_1, int, int, Shape<Shape<int, int>, int>>;

  using StrideQ = Stride<_0, _1, Stride<Stride<int, int>, int>>;
  using StrideNewK = Stride<_0, _1, Stride<Stride<_0, int>, int>>;
  using StrideCacheK = Stride<int, _1, Stride<Stride<_0, int>, int>>;
  using StrideNewV = StrideNewK;
  using StrideCacheV = StrideCacheK;
  using StrideO = StrideQ;

  using Kernel = 
    cutlass::fmha::kernel::Sm100FmhaGenKernelWarpspecialized<
      ProblemShape,
      cutlass::fmha::collective::Sm100FmhaGenMainloopWarpspecialized<
        Element, ElementAcc, ElementAcc, ElementOut,
        TileShape,
        StrideQ, StrideNewK, StrideNewV,
        StrideCacheK, StrideCacheV, StrideO
      >,
      cutlass::fmha::collective::Sm100FmhaGenEpilogueWarpspecialized<ElementOut, StrideO>,
      std::conditional_t<kKernelType == KernelType::UMMA_P,
        cutlass::fmha::kernel::PersistentTileScheduler,
        cutlass::fmha::kernel::IndividualTileScheduler
      >
    >;
  
  using Operation = cutlass::fmha::device::FMHA<Kernel>;

  StrideQ stride_q;
  StrideNewK stride_new_k;
  StrideNewV stride_new_v;
  StrideCacheK stride_cache_k;
  StrideCacheV stride_cache_v;
  StrideO stride_o;
  uint64_t seed = 0;

  std::vector<int> seqlen_kv;

  DeviceAllocation<int> block_seqlen_kv;
  DeviceAllocation<int> block_cache_batch_idx;
  DeviceAllocation<Element> block_q;
  DeviceAllocation<Element> block_new_k;
  DeviceAllocation<Element> block_new_v;
  DeviceAllocation<Element> block_cache_k;
  DeviceAllocation<Element> block_cache_v;
  DeviceAllocation<ElementOut> block_o;

  DeviceAllocation<Element> block_ref_cache_k;
  DeviceAllocation<Element> block_ref_cache_v;
  DeviceAllocation<ElementOut> block_ref_o;

  ClearCache clear_cache;

  bool verify(const ProblemShape& problem_shape) {

    Tensor mQ = make_tensor(make_gmem_ptr(block_q.get()), select<0,2,3>(problem_shape), stride_q);
    Tensor mNewK = make_tensor(make_gmem_ptr(block_new_k.get()), select<0,2,3>(problem_shape), stride_new_k);
    Tensor mNewV = make_tensor(make_gmem_ptr(block_new_v.get()), select<0,2,3>(problem_shape), stride_new_v);
    Tensor mCacheK = make_tensor(make_gmem_ptr(block_ref_cache_k.get()), select<1,2,3>(problem_shape), stride_cache_k);
    Tensor mCacheV = make_tensor(make_gmem_ptr(block_ref_cache_v.get()), select<1,2,3>(problem_shape), stride_cache_v);
    Tensor mO = make_tensor(make_gmem_ptr(block_ref_o.get()), select<0,2,3>(problem_shape), stride_o);

    fmha_fwd_gen_reference<ElementAcc>(
        problem_shape, block_seqlen_kv.get(), block_cache_batch_idx.get(),
        mQ, mNewK, mNewV, mCacheK, mCacheV, mO);
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
    reference_abs_diff(block_o, block_ref_o, max_diff, mean_diff);
    bool passed_O = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_O) {
      std::cerr << "failed O: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(block_cache_k, block_ref_cache_k, max_diff, mean_diff);
    bool passed_K = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if ( ! passed_K) {
      std::cerr << "failed Cache K: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(block_cache_v, block_ref_cache_v, max_diff, mean_diff);
    bool passed_V = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if ( ! passed_V) {
      std::cerr << "failed Cache V: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    return passed_O && passed_K && passed_V;
  }

  ProblemShape initialize(const Options& options) {

    clear_cache.set_active(options.clear_cache);

    std::vector<int> cache_batch_idx;

    // set up stides and sizes
    if (options.remap) {
      for (int i = 0; i < options.b; i++) {
        cache_batch_idx.push_back(i);
      }
      std::mt19937 rng(0x202305291305ull);
      std::shuffle(cache_batch_idx.begin(), cache_batch_idx.end(), rng);
    }

    seqlen_kv = std::vector<int>(options.b, options.k);
    if (options.varlen) {
      std::mt19937 rng(0x202305151552ull);
      std::normal_distribution<double> dist_kv(options.k, options.k / 2);

      auto generate_positive_int = [](auto& dist, auto& gen) {
        int result = 0;
        do {
          result = static_cast<int>(dist(gen));
        } while (result <= 0);
        return result;
      };

      for (int i = 0; i < options.b; i++) {
        seqlen_kv[i] = generate_positive_int(dist_kv, rng);
      }
    }

    int max_seqlen_kv = 0;
    for (auto e : seqlen_kv) {
      max_seqlen_kv = std::max(e, max_seqlen_kv);
    }

    ProblemShape result = make_shape(_1{}, max_seqlen_kv + 1, options.d, make_shape(make_shape(options.h / options.h_k, options.h_k), options.b));

    stride_q = make_stride(_0{}, _1{}, make_stride(make_stride(options.d, options.d * size<3,0,0>(result)), options.d * size<3,0>(result)));
    stride_new_k = make_stride(_0{}, _1{}, make_stride(make_stride(_0{}, options.d), options.d * size<3,0,1>(result)));
    stride_cache_k = make_stride(options.d * size<3,0,1>(result), _1{}, make_stride(make_stride(_0{}, options.d), options.d * size<3,0,1>(result) * get<1>(result)));

    stride_new_v = stride_new_k;
    stride_cache_v = stride_cache_k;
    stride_o = stride_q;

    block_q.reset(options.b * get<2,1>(stride_q));
    if (! options.cache_only) {
      block_new_k.reset(options.b * get<2,1>(stride_new_k));
      block_new_v.reset(options.b * get<2,1>(stride_new_v));
    }
    block_cache_k.reset(options.b * get<2,1>(stride_cache_k));
    block_cache_v.reset(options.b * get<2,1>(stride_cache_v));
    block_o.reset(options.b * get<2,1>(stride_o));

    block_ref_cache_k.reset(options.b * get<2,1>(stride_cache_k));
    block_ref_cache_v.reset(options.b * get<2,1>(stride_cache_v));
    block_ref_o.reset(options.b * get<2,1>(stride_o));
    
    initialize_block(block_q, seed + 2023, options.init_style_q);
    if (! options.cache_only) {
      initialize_block(block_new_k, seed + 2022, options.init_style_new_k);
      initialize_block(block_new_v, seed + 2021, options.init_style_new_v);
    }

    initialize_block(block_cache_k, seed + 2024 - 2025, options.init_style_cache_k);
    initialize_block(block_cache_v, seed + 2025, options.init_style_cache_v);

    block_ref_cache_k.copy_from_device(block_cache_k.get(), block_cache_k.size());
    block_ref_cache_v.copy_from_device(block_cache_v.get(), block_cache_v.size());
    block_seqlen_kv.reset(seqlen_kv.size());
    block_seqlen_kv.copy_from_host(seqlen_kv.data(), seqlen_kv.size());

    if (! cache_batch_idx.empty()) {
      block_cache_batch_idx.reset(cache_batch_idx.size());
      block_cache_batch_idx.copy_from_host(cache_batch_idx.data(), cache_batch_idx.size());
    }

    return result;
  }

  ExampleResult run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    auto problem_shape = initialize(options);

    typename Operation::Arguments arguments{
      problem_shape,
      block_seqlen_kv.get(), block_cache_batch_idx.get(),
      block_q.get(), stride_q,
      block_new_k.get(), stride_new_k,
      block_new_v.get(), stride_new_v,
      block_cache_k.get(), stride_cache_k,
      block_cache_v.get(), stride_cache_v,
      block_o.get(), stride_o,
      hw_info
    };

    Operation op;

    ExampleResult example_result;

    example_result.smem_size = Operation::Kernel::SharedStorageSize;

    size_t workspace_size = 0;
    workspace_size = Operation::get_workspace_size(arguments);
    DeviceAllocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      // std::cerr << "This kernel is not supported. Last CUDA error is: "
      //           << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return example_result;
    }
    example_result.supported = true;

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

    float total_runtime_ms = 0;

    for (int i = 0; i < options.iterations; i++) {

      clear_cache();

      // Record an event at the start of a series of GEMMs
      result = cudaEventRecord(events[0]);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
        return example_result;
      }

      status = op.run();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return example_result;
      }

      // Record an event when the GEMMs are complete
      result = cudaEventRecord(events[1]);
      if (result != cudaSuccess) {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result) << std::endl;
        return example_result;
      }

      //
      // Stop profiling loop
      //
  
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

      result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize() failed: " << cudaGetErrorString(result) << std::endl;
        return example_result;
      }

      total_runtime_ms += runtime_ms;

    }

    float runtime_ms = total_runtime_ms / static_cast<float>(options.iterations);

    double bytes;
    bytes = 0.0;
    bytes += double(sizeof(Element) * size<3>(problem_shape));  // Q
    bytes += double(sizeof(ElementOut) * size<3>(problem_shape));  // O
    bytes += 2.0 * double(sizeof(Element) * size<3>(problem_shape) / size<3,0,0>(problem_shape));  // NewK, NewV
    double total_seqlen_kv = 0;
    for (auto e : seqlen_kv) {
      total_seqlen_kv += double(e + 1);
    }
    bytes += 2.0 * double(sizeof(Element) * size<3,0,1>(problem_shape) * total_seqlen_kv);  // CacheK, CacheV
    bytes *= static_cast<double>(size<2>(problem_shape));
    double tbytes_s = bytes * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
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

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main_result = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, ExampleResult result, bool verbose) {
  std::ios fmt(nullptr);
  fmt.copyfmt(std::cout);
  std::cout << (result.supported ? (result.passed ? (result.verified ? " [OK]  " : " [--] ") : "[FAIL] ") : "[NSUP] ");
  if (result.supported && ! result.passed) {
    main_result = -1;
  }
  std::cout << std::setw(32) << std::left << description;
  std::cout.copyfmt(fmt);
  std::cout << " : " << result.tbytes_s << " TB/s" << std::endl;
  if (verbose) {
    std::cout << "       t=" << result.runtime_ms << "ms, "
        "smem=" << result.smem_size << "b" << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main_single(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 12 || props.major < 10) {
    std::cout
      << "This example requires a GPU of NVIDIA's Blackwell Architecture or "
      << "later (compute capability 90 or greater) and CUDA 12.0 or greater.\n";
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

  std::cout << "###### B " << options.b << " H " << options.h << " H_K " << options.h_k << " K " << options.k << " D " << options.d << " ";
  std::cout << "Gen" << " " << (options.varlen ? "Variable" : "Uniform") << " " << (options.remap ? "Remap" : "Linear") << " ";
  std::cout << "#SM " << hw_info.sm_count << std::endl;

  using UMMA = true_type;
  using FFMA2 = false_type;
  auto run = [&](const char* name, auto kernel_type, auto tile, auto thr) {
    if ((! options.kernel_filter.empty()) && (! std::regex_search(name, std::basic_regex(options.kernel_filter)))) {
        return;
    }
    ExampleRunner<decltype(kernel_type)::value, decltype(tile), decltype(thr)> runner;
    auto result = runner.run(options, hw_info);
    print_result(name, result, options.verbose);
  };


  #define RUN(MODE, m, n, k, tm, tn, tk) \
    run( \
      #MODE " " #m "x" #n "x" #k " / " #tm "x" #tn "x" #tk, \
      std::integral_constant<KernelType, KernelType::MODE>{}, Shape<_##m, _##n, _##k>{}, Shape<_##tm, _##tn, _##tk>{} \
    )

  if (options.d == 128) {
    RUN(UMMA_I, 128, 64, 128, 1, 1, 1);
    RUN(UMMA_I, 128, 128, 128, 1, 1, 1);
    RUN(UMMA_I, 128, 256, 128, 1, 1, 1);
    RUN(UMMA_P, 128, 64, 128, 1, 1, 1);
    RUN(UMMA_P, 128, 128, 128, 1, 1, 1);
    RUN(UMMA_P, 128, 256, 128, 1, 1, 1);
  }
  else {
    std::cout << "Head Dimension != 128 is not supported for the fmha_gen example\n";
  }
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
