/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Example implementation of fused multi-head attention for Blackwell using CUTLASS 3.

    This example showcases the use of CUTLASS to build backward fused
    multi-head attention (FMHA) collectives from existing CUTLASS collectives targeting
    the NVIDIA Blackwell architecture.

    Background and motivation
    -------------------------
    CUTLASS is a highly flexible library that provides open-source building blocks
    for tensor core programming for GEMM or GEMM-like problems. Fused multi-head
    attention (FMHA) is a foundational kernel for large language models (LLMs) since it
    makes long sequence lengths feasible from a memory-usage perspective. It also
    improves computational efficiency since it transforms an outer-product-like and
    a matrix-vector-like GEMM into a fused operation with much higher arithmetic
    intensity. For more details, see Dao et al, 2022; Dao, 2023.
    Implementing this kernel in CUTLASS enabled easy customization and high
    performance.

    Introduction
    ------------
    The example targets the NVIDIA Blackwell architecture, and takes advantage of
    5th gen tensor cores and the Tensor Memory Accelerator (TMA), just like
    GEMMs do. It provides a backward pass (often abbreviated
    bwd in the code).
    The code is structured into three layers: The runner (and the reference kernels)
    takes care of initialization, measurement, and testing; the device layer
    orchestrates kernel calls and partitions workspace; and the kernel layer (just
    like the CUTLASS kernel layer.

    Support
    ---------

    We support fp16 and fp8 data types with a head dimension of 128.

    Example usage:
    $ ./examples/77_blackwell_fmha/77_blackwell_fmha_bwd_fp16 \
            --b=2048 --h=2048 --d=2048 --q=2048 --k=2048
*/

#include <iostream>
#include <random>
#include <regex>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "reference/fmha_fwd_reference.hpp"
#include "reference/fmha_bwd_reference.hpp"
#include "reference/reference_abs_error.hpp"

#include "collective/fmha_fusion.hpp"
#include "device/fmha_device_bwd.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;

///////////////////////////////////////////////////////////////////////////////////////////////////

enum class InitStyle {
  kOne, kZero, kLinearStride128, kLinearStride1, kRandom, kNone
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help = false;
  bool error = false;

  int b = 16;
  int h = 16;
  int h_k = 1;
  int q = 1024;
  int k = 1024;
  std::vector<int> varlen_q;
  std::vector<int> varlen_k;
  int d = 128;
  int d_vo = 128;
  int iterations = 3;
  bool verify = false;
  bool verbose = false;

  bool causal = false;
  bool residual = false;
  bool varlen = false;
  int sm_count = 0;

  std::string kernel_filter;

  InitStyle init_style_q = InitStyle::kRandom;
  InitStyle init_style_k = InitStyle::kRandom;
  InitStyle init_style_v = InitStyle::kRandom;
  InitStyle init_style_do = InitStyle::kRandom;
  bool skip_reference = false;

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
    cmd.get_cmd_line_argument("d_vo", d_vo, d);
    cmd.get_cmd_line_argument("h", h, -1);
    if (h == -1) h = 2048 / d;

    cmd.get_cmd_line_argument("h_k", h_k, -1);
    if (h_k == -1) h_k = h;

    varlen = cmd.check_cmd_line_flag("varlen");

    cmd.get_cmd_line_argument("q", q, -1);
    cmd.get_cmd_line_argument("k", k, -1);
    cmd.get_cmd_line_argument("b", b, -1);
    std::string varlen_q_str;
    cmd.get_cmd_line_argument("varlen-q", varlen_q_str);
    std::string varlen_k_str;
    cmd.get_cmd_line_argument("varlen-k", varlen_k_str);

    if (varlen && ! varlen_q_str.empty()) {
      varlen_q.clear();
      while (! varlen_q_str.empty()) {
        size_t pos = varlen_q_str.find(':');
        varlen_q.push_back(std::stoi(varlen_q_str.substr(0, pos)));
        if (pos == std::string::npos) {
          break;
        }
        varlen_q_str = varlen_q_str.substr(pos + 1);
      }
      if (b == -1) {
        b = static_cast<int>(varlen_q.size());
      }
      if (b != static_cast<int>(varlen_q.size())) {
        std::cout << "Error: Invalid --varlen-q length\n";
        std::exit(-1);
      }
      int new_q = 0;
      for (auto elem : varlen_q) {
        new_q += elem;
      }
      if (q != -1) {
        std::cout << "Error: Can't provide --q and --varlen-q\n";
        std::exit(-1);
      }
      q = new_q;
    }

    if (varlen && ! varlen_k_str.empty()) {
      varlen_k.clear();
      while (! varlen_k_str.empty()) {
        size_t pos = varlen_k_str.find(':');
        varlen_k.push_back(std::stoi(varlen_k_str.substr(0, pos)));
        if (pos == std::string::npos) {
          break;
        }
        varlen_k_str = varlen_k_str.substr(pos + 1);
      }
      if (b == -1) {
        b = static_cast<int>(varlen_k.size());
      }
      if (b != static_cast<int>(varlen_k.size())) {
        std::cout << " Error: Invalid --varlen-k length\n";
        std::exit(-1);
      }
      int new_k = 0;
      for (auto elem : varlen_k) {
        new_k += elem;
      }
      if (k != -1) {
        std::cout << "Error: Can't provide --k and --varlen-k\n";
        std::exit(-1);
      }
      k = new_k;
    }

    if (q == -1) q = k;
    if (k == -1) k = q;
    if (q == -1 && k == -1) q = k = defaults.q;
    if (b == -1) b = 16384 / k;
    if (b == 0) b = 1;

    cmd.get_cmd_line_argument("iterations", iterations, defaults.iterations);
    verify = cmd.check_cmd_line_flag("verify");
    verbose = cmd.check_cmd_line_flag("verbose");
    std::string mask;
    cmd.get_cmd_line_argument<std::string>("mask", mask, "");
    if (mask == "causal") {
      causal = true;
    }
    else if (mask == "residual") {
      residual = true;
    }
    else {
      causal = defaults.causal;
    }
    if (varlen) {
      residual = true;
    }

    skip_reference = cmd.check_cmd_line_flag("skip-reference");
    cmd.get_cmd_line_argument("sm-count", sm_count, defaults.sm_count);

    get_init_style_argument(cmd, "init-style", init_style_q, defaults.init_style_q);
    get_init_style_argument(cmd, "init-style", init_style_k, defaults.init_style_k);
    get_init_style_argument(cmd, "init-style", init_style_v, defaults.init_style_v);
    get_init_style_argument(cmd, "init-style", init_style_do, defaults.init_style_do);
    get_init_style_argument(cmd, "init-style-q", init_style_q, init_style_q);
    get_init_style_argument(cmd, "init-style-k", init_style_k, init_style_k);
    get_init_style_argument(cmd, "init-style-v", init_style_v, init_style_v);
    get_init_style_argument(cmd, "init-style-do", init_style_v, init_style_do);

    cmd.get_cmd_line_argument("kernel-filter", kernel_filter, defaults.kernel_filter);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "77_blackwell_fmha_bwd\n\n"
      << "  This example showcases the use of CUTLASS's collective operation builders to easily construct\n"
      << "  fused multi-head attention kernels for the backward pass targeting NVIDIA's Blackwell architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --b=<int>                   Sets the B extent\n"
      << "  --h=<int>                   Sets the H extent\n"
      << "  --h_k=<int>                 Sets the H_K/V extent (for GQA/MQA)\n"
      << "  --q=<int>                   Sets the Q extent\n"
      << "  --k=<int>                   Sets the K extent\n"
      << "  --varlen-q=<int>:<int...>   Sets the variable Q extent per batch (colon separated)\n"
      << "  --varlen-k=<int>:<int...>   Sets the variable K extent per batch (colon separated)\n"
      << "  --d=<int>                   Sets the D extent\n"
      << "  --d_vo=<int>                  Sets the D_VO extent\n"
      << "  --iterations=<int>          Benchmarking iterations\n"
      << "  --verify                    Verify results\n"
      << "  --verbose                   Print smem and execution time per kernel\n"
      << "  --mask=<no|residual|causal> Enables masking\n"
      << "  --varlen                    Enables variable sequence length\n"
      << "                              B*Q and B*K become the total sequence length\n"
      << "                              and are split B-ways, alternatingly +10% and -10%\n"
      << "                              with the last batch sized to make it fit\n"
      << "                              implies at least residual masking for correctness\n"
      << "  --sm-count                  Sets SM count rather than querying it\n"
      << "  --kernel-filter=<filter>    Sets regexp to match kernel against\n"
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
    case InitStyle::kZero: {
      cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, (Element) 0, (Element) 0);
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
  bool passed = false;
  bool verified = false;
  float runtime_ms = 0;
  double tflops_tc_s = 0;
  size_t smem_size = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

template<
  bool kIsVarlen,
  bool kIsMla,
  class TileShape,
  class DispatchPolicy,
  class ActiveMask,
  class... KernelOptions
>
struct BwdRunner {

#ifdef FP8
  using Element = cutlass::float_e4m3_t;
#else
  using Element = cutlass::half_t;
#endif
  using ElementAccumulator = float;

  // Q K D D_VO ((H_R, H_K) B)
  using ProblemShape = std::conditional_t<
      kIsVarlen,
      cute::tuple<VariableLength, VariableLength, int, int, cute::tuple<cute::tuple<int, int>, int>>,
      cute::tuple<int, int, int, int, cute::tuple<cute::tuple<int, int>, int>>
  >;
  
  using StrideQ = Stride<int, _1, Stride<Stride<int, int>, int>>; // Q D    ((H_R, H_K), B)
  using StrideK = Stride<int, _1, Stride<Stride<_0, int>, int>>;  // K D    ((H_R, H_K), B)
  using StrideV = StrideK;                                        // K D_VO ((H_R, H_K), B)
  using StrideO = StrideQ;                                        // Q D_VO ((H_R, H_K), B)
  using StrideLSE = Stride<_1, Stride<Stride<int, int>, int>>;    // Q      ((H_R, H_K), B)

  // Backwards specific
  using StrideDQ = StrideQ;
  using StrideDK = StrideK;
  using StrideDV = StrideV;
  using StrideDO = StrideO;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;

  StrideDQ stride_dQ;
  StrideDK stride_dK;
  StrideDV stride_dV;
  StrideDO stride_dO;

  uint64_t seed = 0;

  DeviceAllocation<Element> block_Q;
  DeviceAllocation<Element> block_K;
  DeviceAllocation<Element> block_V;
  DeviceAllocation<Element> block_O;
  DeviceAllocation<ElementAccumulator> block_LSE;

  DeviceAllocation<int> block_cumulative_seqlen_q;
  DeviceAllocation<int> block_cumulative_seqlen_kv;

  DeviceAllocation<Element> block_dQ;
  DeviceAllocation<Element> block_dK;
  DeviceAllocation<Element> block_dV;
  DeviceAllocation<Element> block_dO;

  DeviceAllocation<Element> block_ref_dQ;
  DeviceAllocation<Element> block_ref_dK;
  DeviceAllocation<Element> block_ref_dV;

  //
  // Methods
  //
  bool verify(const ProblemShape& problem_shape) {
    auto [Q, K, D, D_VO, HB] = problem_shape;
    auto [H, B] = HB;

    Tensor mQ = make_tensor(make_gmem_ptr(block_Q.get()), make_shape(Q, D, HB), stride_Q);
    Tensor mK = make_tensor(make_gmem_ptr(block_K.get()), make_shape(K, D, HB), stride_K);
    Tensor mV = make_tensor(make_gmem_ptr(block_V.get()), make_shape(K, D_VO, HB), stride_V);
    Tensor mO = make_tensor(make_gmem_ptr(block_O.get()), make_shape(Q, D_VO, HB), stride_O);
    Tensor mLSE = make_tensor(make_gmem_ptr(block_LSE.get()), make_shape(Q, HB), stride_LSE);
    Tensor mDQ = make_tensor(make_gmem_ptr(block_ref_dQ.get()), make_shape(Q, D, HB), stride_dQ);
    Tensor mDK = make_tensor(make_gmem_ptr(block_ref_dK.get()), make_shape(K, D, HB), stride_dK);
    Tensor mDV = make_tensor(make_gmem_ptr(block_ref_dV.get()), make_shape(K, D_VO, HB), stride_dV);
    Tensor mDO = make_tensor(make_gmem_ptr(block_dO.get()), make_shape(Q, D_VO, HB), stride_dO);

    fmha_bwd_reference(problem_shape, mQ, mK, mV, mO, mLSE, mDO, mDQ, mDK, mDV, ActiveMask{});

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    const double kMaxDiffThresh = sizeof(Element) == 1 ? 1e-0 : 1e-2;
    const double kMeanDiffThresh = sizeof(Element) == 1 ? 1e-1 : 1e-3;

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    double max_diff = 0;
    double mean_diff = 0;
    reference_abs_diff(block_dQ, block_ref_dQ, max_diff, mean_diff);

    bool passed_dQ = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_dQ) {
      std::cerr << "failed dQ: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(block_dK, block_ref_dK, max_diff, mean_diff);

    bool passed_dK = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_dK) {
      std::cerr << "failed dK: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    reference_abs_diff(block_dV, block_ref_dV, max_diff, mean_diff);

    bool passed_dV = (max_diff < kMaxDiffThresh) && (mean_diff < kMeanDiffThresh);
    if (! passed_dV) {
      std::cerr << "failed dV: max diff " << max_diff 
                << " mean " << mean_diff << std::endl;
    }

    return passed_dQ && passed_dK && passed_dV;
  }

  auto initialize_problem_shape(Options const& options) {
    int h_r = options.h / options.h_k;
    assert(options.h % options.h_k == 0);

    if constexpr (kIsVarlen) {
      int num_batches = options.b;

      // generate Q as --b times
      //    gaussian (--Q, --Q / 2) sampled positive
      //    track cumulative 
      std::mt19937 rng(0x202305151552ull);
      std::normal_distribution<double> dist_q(options.q, options.q / 2);
      std::normal_distribution<double> dist_kv(options.k, options.k / 2);

      auto generate_positive_int = [](auto& dist, auto& gen) {
        // "0" is a valid value we test here
        return std::max(0, static_cast<int>(dist(gen)));
      };

      std::vector<int> cumulative_seqlen_q = {0};
      std::vector<int> cumulative_seqlen_kv = {0};

      int total_seqlen_q = 0;
      int total_seqlen_kv = 0;
      int max_seqlen_q = 0;
      int max_seqlen_kv = 0;

      const bool kVarlenSame = false;
      for (int i = 0; i < num_batches; i++) {
        int seqlen_q = (! options.varlen_q.empty()) ? options.varlen_q.at(i) : 
                kVarlenSame ? options.q :
                generate_positive_int(dist_q, rng);
        int seqlen_kv = (! options.varlen_k.empty()) ? options.varlen_k.at(i) :
                kVarlenSame ? options.k :
                generate_positive_int(dist_kv, rng);

        total_seqlen_q += seqlen_q;
        total_seqlen_kv += seqlen_kv;

        max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
        max_seqlen_kv = std::max(max_seqlen_kv, seqlen_kv);

        cumulative_seqlen_q.push_back(cumulative_seqlen_q.back() + seqlen_q);
        cumulative_seqlen_kv.push_back(cumulative_seqlen_kv.back() + seqlen_kv);
      }

      block_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
      block_cumulative_seqlen_q.copy_from_host(cumulative_seqlen_q.data(), cumulative_seqlen_q.size());
      block_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
      block_cumulative_seqlen_kv.copy_from_host(cumulative_seqlen_kv.data(), cumulative_seqlen_kv.size());

      ProblemShape problem_shape{
          {max_seqlen_q, block_cumulative_seqlen_q.get(), total_seqlen_q},
          {max_seqlen_kv, block_cumulative_seqlen_kv.get(), total_seqlen_kv},
          options.d, options.d_vo, {{h_r, options.h_k}, options.b}
      };
      auto tensor_shape = make_shape(total_seqlen_q, total_seqlen_kv, options.d, options.d_vo, make_shape(make_shape(h_r, options.h_k), 1));

      return cute::make_tuple(problem_shape, tensor_shape);
    }
    else {
      ProblemShape problem_shape{options.q, options.k, options.d, options.d_vo, {{h_r, options.h_k}, options.b}};
      return cute::make_tuple(problem_shape, problem_shape);
    }
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  ProblemShape initialize(Options const& options) {
    auto [problem_shape, tensor_shape] = initialize_problem_shape(options);
    auto [Q, K, D, D_VO, HB] = tensor_shape;
    auto [H, B] = HB;
    auto [H_R, H_K] = H;
    D = cutlass::round_up(D, 8);  // Alignment

    // for varlen, Q == total_Q, K == total_K, B = 1
    // but in problem_shape, they've got to be max_Q/max_K, and B = B

    auto shape_Q = make_shape(Q, D, HB);
    auto shape_K = make_shape(K, D, HB);
    auto shape_V = make_shape(K, D_VO, HB);
    auto shape_O = make_shape(Q, D_VO, HB);
    auto shape_LSE = make_shape(Q, HB);

    stride_Q = make_stride(D, _1{}, make_stride(make_stride(D*Q, D*Q*H_R), B == 1 ? 0 : D*Q*H_R*H_K));
    stride_K = make_stride(D, _1{}, make_stride(make_stride(_0{}, D*K), B == 1 ? 0 : D*K*H_K));
    stride_V = make_stride(D_VO, _1{}, make_stride(make_stride(_0{},D_VO*K), B == 1 ? 0 : D_VO*K*H_K));
    stride_O = make_stride(D_VO, _1{}, make_stride(make_stride(D_VO*Q, D_VO*Q*H_R), B == 1 ? 0 : D_VO*Q*H_R*H_K));
    stride_LSE = make_stride(_1{}, make_stride(make_stride(Q, Q*H_R), B == 1 ? 0 : Q*H_R*H_K));

    stride_dQ = stride_Q;
    stride_dK = stride_K;
    stride_dV = stride_V;
    stride_dO = stride_O;

    auto lsize = [](auto shape) {
      return size(make_shape(1ull, shape));
    };

    auto size_K = lsize(K * D * H_K * B);
    auto size_V = lsize(K * D_VO * H_K * B);

    block_Q.reset(lsize(shape_Q));
    block_K.reset(size_K);
    block_V.reset(size_V);
    block_O.reset(lsize(shape_O));
    block_LSE.reset(lsize(shape_LSE));

    block_dQ.reset(lsize(shape_Q));
    block_dK.reset(size_K);
    block_dV.reset(size_V);
    block_dO.reset(lsize(shape_O));

    block_ref_dQ.reset(lsize(shape_Q));
    block_ref_dK.reset(size_K);
    block_ref_dV.reset(size_V);

    initialize_block(block_Q, seed + 2023, options.init_style_q);
    initialize_block(block_K, seed + 2022, options.init_style_k);
    initialize_block(block_V, seed + 2021, options.init_style_v);
    initialize_block(block_dO, seed + 2020, options.init_style_do);

    initialize_block(block_dQ, seed + 2030, InitStyle::kOne);
    initialize_block(block_dK, seed + 2031, InitStyle::kOne);
    initialize_block(block_dV, seed + 2032, InitStyle::kOne);
    initialize_block(block_ref_dQ, seed + 2033);
    initialize_block(block_ref_dK, seed + 2034);
    initialize_block(block_ref_dV, seed + 2035);

    Tensor mQ = make_tensor(make_gmem_ptr(block_Q.get()),
      select<0,2,4>(problem_shape),
      stride_Q);

    Tensor mK = make_tensor(make_gmem_ptr(block_K.get()),
      select<1,2,4>(problem_shape),
      stride_K);

    Tensor mV = make_tensor(make_gmem_ptr(block_V.get()),
      select<1,3,4>(problem_shape),
      stride_V);

    Tensor mO = make_tensor(make_gmem_ptr(block_O.get()),
      select<0,3,4>(problem_shape),
      stride_O);

    Tensor mLSE = make_tensor(make_gmem_ptr(block_LSE.get()),
      select<0,4>(problem_shape),
      stride_LSE);

    if (not options.skip_reference) {
      fmha_reference(problem_shape, mQ, mK, mV, mO, mLSE, ActiveMask{});
    }

    return problem_shape;
  }

  ExampleResult run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    auto problem_shape = initialize(options);

    ElementAccumulator softmax_scale = 1.0f / sqrtf(options.d);

    ExampleResult example_result;

    using Operation = cutlass::fmha::device::Sm100FmhaBwd<ProblemShape, Element, ElementAccumulator, TileShape, kIsMla, ActiveMask>;

    typename Operation::Arguments arguments{
      problem_shape,
      block_Q.get(), stride_Q,
      block_K.get(), stride_K,
      block_V.get(), stride_V,
      block_O.get(), stride_O,
      block_LSE.get(), stride_LSE,
      block_dO.get(), stride_dO,
      block_dQ.get(), stride_dQ,
      block_dK.get(), stride_dK,
      block_dV.get(), stride_dV,
      softmax_scale,
      hw_info
    };

    Operation op;

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

    double flops = 2.0 * (std::is_same_v<ActiveMask, CausalForBackwardMask<false>> || std::is_same_v<ActiveMask, CausalForBackwardMask<true>> ? 0.5 : 1.0);
    flops *= static_cast<double>(get<0>(problem_shape));
    flops *= static_cast<double>(get<1>(problem_shape));
    flops *= (3 * static_cast<double>(get<2>(problem_shape)) + 2 * static_cast<double>(get<3>(problem_shape)));
    flops *= static_cast<double>(get<4,0,0>(problem_shape));
    flops *= static_cast<double>(get<4,0,1>(problem_shape));
    flops *= static_cast<double>(get<4,1>(problem_shape));
    double tflops_s = flops * 1e-12 /*tera*/ / (runtime_ms * 1e-3 /*ms*/);
    example_result.tflops_tc_s = tflops_s;
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
  std::cout << " : " << result.tflops_tc_s << " TFLOPS/s" << std::endl;
  if (verbose) {
    std::cout << "       t=" << result.runtime_ms << "ms, "
        "smem=" << result.smem_size << "b" << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct KernelCoop {};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<class Fn>
auto dispatch_bool(bool value, Fn fn) {
  if (value) {
    return fn(std::true_type{});
  }
  else {
    return fn(std::false_type{});
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

template<class Mask>
void run_bwd_64(Mask fusion, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, auto kernel, const char* name, auto... kernel_options) {
    dispatch_bool(options.varlen, [&](auto is_varlen) {
      BwdRunner<decltype(is_varlen)::value, false,decltype(shape), decltype(kernel), Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    });
  };

  using HeadDim = _64;

  run(Shape<_128, _128, HeadDim, HeadDim>{}, KernelCoop{}, "tma");
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<class Mask>
void run_bwd_128(Mask fusion, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, auto kernel, const char* name, auto... kernel_options) {
    dispatch_bool(options.varlen, [&](auto is_varlen) {
      BwdRunner<decltype(is_varlen)::value, false, decltype(shape), decltype(kernel), Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    });
  };

  using HeadDim = _128;

  run(Shape<_128, _128, HeadDim, HeadDim>{}, KernelCoop{}, "tma");
}

template<class Mask>
void run_bwd_mla_192(Mask fusion, Options const & options, cutlass::KernelHardwareInfo const& hw_info) {
  auto run = [&](auto shape, auto kernel, const char* name, auto... kernel_options) {
    dispatch_bool(options.varlen, [&](auto is_varlen) {
      BwdRunner<decltype(is_varlen)::value, true, decltype(shape), decltype(kernel), Mask, decltype(kernel_options)...> runner;
      auto result = runner.run(options, hw_info);
      print_result(name, result, options.verbose);
    });
  };

  using HeadDim = _192;

  run(Shape<_64, _128, HeadDim, _128>{}, KernelCoop{}, "tma");
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
      << "(compute capability 100a) and CUDA 12.8 or greater.\n";
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

  std::cout << "###### B " << options.b << " H " << options.h << " H_K " << options.h_k << " Q " << options.q << " K " << options.k << " D " << options.d << " D_VO " << options.d_vo << " ";
  std::cout << "Backward" << " " << (options.causal ? "Causal" : "Full") << " ";
  std::cout << "#SM " << hw_info.sm_count << std::endl;

  auto with_causal = [&](auto fn) {
    if (options.causal) {
      fn(CausalForBackwardMask{});
    }
    else if (options.residual) {
      fn(ResidualMaskForBackward{});
    }
    else {
      fn(NoMask{});
    }
  };

  with_causal([&](auto fusion) {
    if (options.d <= 64 && options.d_vo == options.d) {
      run_bwd_64(fusion, options, hw_info);
    }
    else if (options.d <= 128 && options.d_vo == options.d) {
      run_bwd_128(fusion, options, hw_info);
    }
    else if (options.d == 192 && options.d_vo == 128) {
      run_bwd_mla_192(fusion, options, hw_info);
    }
    else {
      std::cout << "No kernel instantiated for d=" << options.d << std::endl;
    }
  });
#endif

  return main_result;
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
