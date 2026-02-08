/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Command line options parsing
using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;
template<typename _ProblemShape>
struct Options {
  using ProblemShape = _ProblemShape;

  bool help = false;

  float alpha = 1.f, beta = 0.f;
  int iterations = 1000;
  int m = 1024, n = 512, k = 1024, groups = 10;
  std::string benchmark_path;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_after_alignment_host;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  int const tma_alignment_bits = 128;
  int const alignment = tma_alignment_bits / cutlass::sizeof_bits<cutlass::float_e4m3_t>::value;
  int const k_alignment = 128;
  int const m_alignment = 128;
  int const n_alignment = 128;

  RasterOrderOptions raster_order;
  int swizzle;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("groups", groups);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster_order = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster_order = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h') {
      raster_order = RasterOrderOptions::Heuristic;
    }

    cmd.get_cmd_line_argument("swizzle", swizzle, 1);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);

    // Decide how to initialize the problems
    if (!benchmark_path.empty()) {
      if (!benchmark_problems()) {
        problem_sizes_after_alignment_host.clear();
        problem_sizes_host.clear();
        return;
      }
    }
    else {
      randomize_problems(cmd);
    }

  }

  void randomize_problems(cutlass::CommandLine &cmd) {
    int cmd_line_m = -1, cmd_line_n = -1, cmd_line_k = -1;
    cmd.get_cmd_line_argument("m", cmd_line_m);
    cmd.get_cmd_line_argument("n", cmd_line_n);
    cmd.get_cmd_line_argument("k", cmd_line_k);

    problem_sizes_after_alignment_host.reserve(groups);
    problem_sizes_host.reserve(groups);
    for (int i = groups; i > 0; i--) {
      int m = cmd_line_m;
      int n = cmd_line_n;
      int k = cmd_line_k;
      if (m < 0) {
        m = m_alignment * (rand() % (64 * alignment / m_alignment));
      }
      if (n < 0) {
        n = n_alignment * (rand() % (64 * alignment / n_alignment));
      }
      if (k < 0) {
        k = k_alignment * (rand() % (32 * alignment / k_alignment));
      }
      problem_sizes_after_alignment_host.push_back({m, n, k});
      problem_sizes_host.push_back({m, n, k});
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

      cutlass::gemm::GemmCoord extent_after_alignment, extent;
      std::vector<std::string> tokens;

      cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        int x = std::atoi(tokens.at(i).c_str());

        extent.at(i) = x;
        // round up
        if (x % alignment) {
          x += (alignment - (x % alignment));
        }

        extent_after_alignment.at(i) = x;
      }

      problem_sizes_after_alignment_host.push_back({extent_after_alignment.m(), extent_after_alignment.n(), extent_after_alignment.k()});
      problem_sizes_host.push_back({extent.m(), extent.n(), extent.k()});
    }
    groups = static_cast<int>(problem_sizes_after_alignment_host.size());

    return true;
  }

  /// Calculate memory bandwidth statistics
  template <class ElementA, 
            class ElementB,
            class ElementC,
            class ElementD,
            class ElementBlockScale,
            class TileShape,
            int ScaleMsPerTile,
            int ScaleNsPerTile>
  auto gbps(double runtime_s) const {
    double total_read_bytes = 0;
    double total_write_bytes = 0;
    
    // Calculate bytes read and written for each problem
    for (int i = 0; i < groups; ++i) {
      auto problem = problem_sizes_host.at(i);
      auto M = cute::get<0>(problem);
      auto N = cute::get<1>(problem);
      auto K = cute::get<2>(problem);
      
      if (M > 0) {  // Only count active problems
        // Matrix A: M*K elements read
        total_read_bytes += M * K * sizeof(ElementA);
        
        // Matrix B: K*N elements read
        total_read_bytes += K * N * sizeof(ElementB);
        
        // Matrix C: M*N elements read (for beta operation)
        total_read_bytes += M * N * sizeof(ElementC);
        
        // Block scales for A and B
        auto blockscale_shape = cute::shape(cute::get<1>(cute::zipped_divide(cute::make_layout(problem), TileShape{})));
        auto blockscale_m = cute::get<0>(blockscale_shape);
        auto blockscale_n = cute::get<1>(blockscale_shape);
        auto blockscale_k = cute::get<2>(blockscale_shape);
        auto groupscale_m = blockscale_m * ScaleMsPerTile;
        auto groupscale_n = blockscale_n * ScaleNsPerTile;
        
        total_read_bytes += groupscale_m * blockscale_k * sizeof(ElementBlockScale);  // A scales
        total_read_bytes += groupscale_n * blockscale_k * sizeof(ElementBlockScale);  // B scales
        
        // Matrix D: M*N elements written
        total_write_bytes += M * N * sizeof(ElementD);
      }
    }

    return (total_read_bytes + total_write_bytes) / 1.0e9 / runtime_s;
  }

  double bandwidth_util(double eff_bandwidth) const {
    int memoryClockRate;
    int memoryBusWidth;
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth , 0);
    double bw = 2.0 * memoryClockRate * (memoryBusWidth / 8) / 1.0e6;
    return eff_bandwidth / bw * 100.0;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling\n\n"
      << "  Hopper FP8 Grouped GEMM using a Warp Specialized kernel with Blockwise Scaling.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --groups=<int>              Sets the number of individual GEMM problems for Grouped GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
      << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
      << "  --benchmark=<str>           Executes a benchmark problem size.\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling" << " --m=1024 --n=512 --k=1024 --groups=10 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = 0ull;

    for (auto const [m, n, k] : problem_sizes_host) {
      fmas += static_cast<uint64_t>(m) *
              static_cast<uint64_t>(n) *
              static_cast<uint64_t>(k);
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};
