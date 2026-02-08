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
    \brief A GEMM example using CUTLASS for the NVIDIA Blackwell SM100 architecture with the Stream-K scheduler.

    Stream-K is a GEMM parallelization technique that attempts to reduce load imbalance across SMs
    by parallelizing certain output tiles across the K mode of the GEMM, without using a static splitting factor.
    For complete details on Stream-K, please see https://arxiv.org/abs/2301.03598.

    CUTLASS's Stream-K scheduler using the CUTLASS 3.x API is capable of supporting various modes of
    decomposing a GEMM (referred to as "decomposition modes" in this example):
      * DataParallel: basic GEMM parallelized spatially via tiling, but without splitting the K mode
      * SplitK: `split_factor` CTAs compute portions of the K mode for a given output tile and reduce their results
      * StreamK: parallelizes work according to the stream-K load balancing method described in https://arxiv.org/abs/2301.03598
      * Heuristic: applies an internal heuristic in attempt to choose the most performant among the three preceding decomposition modes

    Additionally, the Stream-K scheduler supports two different means of performing reductions for
    decomposition modes that require reduction (SplitK, StreamK, and Heuristic):
      * Deterministic: Participating CTAs perform reduction in a turnstile fashion in order of the K mode
                       covered by each CTA. This requires a lock to be held exclusively by the CTA that is
                       currently accumulating.
      * Nondeterministic: Participating CTAs perform reduction atomically to the same workspace (mostly) without locking.
                          Locks are used only to wait for the first CTA to write its partial values (to initialize the
                          workspace), and for all but the final CTA to have accumulated (so that the final CTA can load
                          the accumulated value and accumulate it into registers on top of which the epilogue will
                          be performed). Due to the nondeterminsitic ordering of accumulation, deterministic numeric
                          behavior cannot be guaranteed with this mode (e.g., floating-point rounding error will depend
                          on the order of accumulation)

    This example allows one to try out different decomposition modes, reduction modes, and (when using Split-K) splitting factors.
    Here are a few examples of usage:
       # Heuristic mode with deterministic reduction
      ./74_blackwell_gemm_streamk" --m=256 --n=256 --k=16384 --decomposition=Heuristic --reduction=Deterministic

      # Stream-K mode with deterministic reduction
      ./74_blackwell_gemm_streamk" --m=256 --n=256 --k=16384 --decomposition=StreamK --reduction=Deterministic

      # Split-K mode with a splitting factor of 2 and deterministic reduction
      ./74_blackwell_gemm_streamk" --m=256 --n=256 --k=16384 --decomposition=SplitK --reduction=Deterministic --splits=2

      # Stream-K mode with nondeterministic reduction
      ./74_blackwell_gemm_streamk" --m=256 --n=256 --k=16384 --decomposition=StreamK --reduction=Nondeterministic
*/



#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = half_t;                                         // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = half_t;                                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = float;                                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag

// MMA and Cluster Tile Shapes
// Shape of the tile computed by tcgen05 MMA, could be across 2 SMs if Cluster Shape % 2 == 0
using MmaTileShape_MNK = Shape<_256,_128,_64>;
// Shape of the cluster set to <int,int,_1> to indicate dynamic cluster shape
using ClusterShape_MNK = Shape<int,int,_1>;
// When dynamic cluster is used, KernelScheduleAuto always selects mainloop dispatch policy that 
// lowers to tcgen05 MMA cta_group = 1 as we don't know if the dynamic cluster M dimension will be a multiple of 2
// To use tcgen05 MMA cta_group = 2, users must explicitly use 2sm builder schedules
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int, int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler // <--- Change needed to enable the stream-K scheduler
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  int preferred_cluster_m, preferred_cluster_n, fallback_cluster_m, fallback_cluster_n;
  using DecompositionMode = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
  using ReductionMode = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::ReductionMode;
  DecompositionMode decomposition_mode;
  ReductionMode reduction_mode;
  int splits;

  std::unordered_map<DecompositionMode, std::vector<std::string>> dec_mappings = {
    {DecompositionMode::Heuristic,    {"Heuristic", "heuristic", "h", "H", ""}},
    {DecompositionMode::SplitK,       {"SplitK", "split-k", "split-K", "Split-K", "Split-k", "splitk", "Splitk", "splitK", "spk", "SpK", "spK"}},
    {DecompositionMode::StreamK,      {"StreamK", "stream-k", "stream-K", "Stream-K", "Stream-k", "streamk", "Streamk", "streamK", "stk", "StK", "stK"}},
    {DecompositionMode::DataParallel, {"DataParallel", "data-parallel", "dataparallel", "dp", "DP"}}
  };

  std::unordered_map<ReductionMode, std::vector<std::string>> red_mappings = {
    {ReductionMode::Deterministic,    {"Deterministic", "deterministic", "d", "D", ""}},
    {ReductionMode::Nondeterministic, {"Nondeterministic", "nondeterministic", "n", "N"}}
  };

  Options():
    help(false),
    m(256), n(256), k(16384),
    alpha(1.f), beta(0.f),
    iterations(10),
    preferred_cluster_m(4),
    preferred_cluster_n(4),
    fallback_cluster_m(2),
    fallback_cluster_n(1),
    decomposition_mode(DecompositionMode::Heuristic),
    reduction_mode(ReductionMode::Deterministic),
    splits(1)
  { }

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
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("splits", splits, 1);
    cmd.get_cmd_line_argument("preferred_cluster_m", preferred_cluster_m, 4);
    cmd.get_cmd_line_argument("preferred_cluster_n", preferred_cluster_n, 4);
    cmd.get_cmd_line_argument("fallback_cluster_m", fallback_cluster_m, 2);
    cmd.get_cmd_line_argument("fallback_cluster_n", fallback_cluster_n, 1);

    // Parse decompsition mode
    std::string decomp_mode;
    cmd.get_cmd_line_argument("decomposition", decomp_mode);
    bool found = parse_from_options_map(decomp_mode, dec_mappings, decomposition_mode);
    if (!found) {
      std::cout << "--decomposition must be one of Heuristic, SplitK, StreamK, or DataParallel" << std::endl;
      help = true;
      return;
    }

    // Parse reduction mode
    std::string red_mode;
    cmd.get_cmd_line_argument("reduction", red_mode);
    found = parse_from_options_map(red_mode, red_mappings, reduction_mode);
    if (!found) {
      std::cout << "--reduction must be one of Deterministic and Nondeterministic" << std::endl;
      help = true;
      return;
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "74_blackwell_gemm_streamk\n\n"
      << "  Blackwell FP16 GEMM using a stream-K kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --preferred_cluster_m=<str> Sets the M extent of preferred cluster shape\n"
      << "  --preferred_cluster_n=<str> Sets the N extent of preferred cluster shape\n"
      << "  --fallback_cluster_m=<str>  Sets the M extent of fallback cluster shape\n"
      << "  --fallback_cluster_n=<str>  Sets the N extent of fallback cluster shape\n"
      << "  --decomposition=<str>       Mode in which the stream-K kernel should decompose the problem. Options: Heuristic (default), SplitK, StreamK, DataParallel\n"
      << "  --reduction=<str>           Mode in which the stream-K kernel's reduction should be performed. Options: Deterministic (default), Nondeterministic\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "74_blackwell_gemm_streamk" << " --m=256 --n=256 --k=16384 --decomposition=Heuristic --reduction=Deterministic \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }

  std::string decomposition_mode_str() const {
    return dec_mappings.at(decomposition_mode).at(0);
  }

  std::string reduction_mode_str() const {
    return red_mappings.at(reduction_mode).at(0);
  }

 private:
  template <class T>
  bool parse_from_options_map(std::string val, std::unordered_map<T, std::vector<std::string>> options, T& result) const {
    for (const auto & [key, values] : options) {
      if (std::find(values.begin(), values.end(), val) != values.end()) {
        result = key;
        return true;
      }
    }
    return false;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(cutlass::DeviceAllocation<Element>& block, uint64_t seed=2023) {
  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options) {
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D}
  };

  arguments.hw_info.cluster_shape = dim3(options.preferred_cluster_m, options.preferred_cluster_n,1);
  arguments.hw_info.cluster_shape_fallback = dim3(options.fallback_cluster_m, options.fallback_cluster_n,1);

  arguments.scheduler.splits = options.splits;
  arguments.scheduler.decomposition_mode = options.decomposition_mode;
  arguments.scheduler.reduction_mode = options.reduction_mode;

  return arguments;
}

bool verify(const Options &options) {
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;

  // Launch device reference gemm kernel
  gemm_reference(
    {options.m, options.n, options.k},
    ElementAccumulator(options.alpha),
    ref_A,
    ref_B,
    ElementAccumulator(options.beta),
    ref_C,
    ref_D);

  // Wait for kernel to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

  return passed;
}

/// Execute a given example GEMM computation
int run(Options &options) {

  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "Stream-K GEMM with"
            << " Problem Size: " << options.m << 'x' << options.n << 'x' << options.k
            << " Preferred Cluster = (" << options.preferred_cluster_m << ", " << options.preferred_cluster_n << ", 1)"
            << " Fallback Cluster = (" << options.fallback_cluster_m << ", " << options.fallback_cluster_n << ", 1)\n"
            << " Decomposition_mode=" << options.decomposition_mode_str()
            << " Split_count=" << options.splits
            << " Reduction_mode=" << options.reduction_mode_str()
            << std::endl;

  std::cout << "--------------------------------------------------------------------------------" << std::endl;

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.8 Toolkit or newer to run this example
  // and must have compute capability at least 100.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (__CUDACC_VER_MAJOR__ < 13) {
    if (props.major != 10 || (props.minor != 0 && props.minor != 1 && props.minor != 3)) {
      std::cerr << "This example requires a GPU with compute capability 100a|f, 101a|f, or 103a|f)." << std::endl;
      return 0;
    } 
  }
  else {
    if ((props.major != 10 || props.major != 11) && props.minor != 0) {
      std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 100 or 110)." << std::endl;
      return 0;
    }
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
  // Evaluate CUTLASS kernels
  //
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  run(options);
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
