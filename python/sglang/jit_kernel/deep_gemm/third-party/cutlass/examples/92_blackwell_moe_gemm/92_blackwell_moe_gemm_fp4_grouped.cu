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
  \brief Example of Blackwell MoE-style grouped NVFP4 GEMM implementation using TMA to load A and CPASYNC to load B.

  This example demonstrates an implementation of GEMM using mixed TMA+CPASYNC to load input matrices.
  In the decoding stage of Mixture of Experts (MoE) models, the number of tokens in different experts 
  can varies a lot, which requires frequently updates of TMA descriptors in TMA-based implementation.
  This examples uses CPASYNC to load activation (B) matrix to avoid the overhead of updating TMA descriptors.

  Usage:
  $ ./examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_fp4_grouped
  --m=28672 --n=4 --k=4096 --l=8 --benchmark=benchmark.txt

*/

#include <iostream>
#include <fstream>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
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
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"


#include "helper.h"


using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool verification;

  int m, n, k, l;

  int iterations;
  
  std::string benchmark_path;

  Options():
    help(false),
    error(false),
    verification(true),
    m(2048), n(2048), k(2048), l(1),
    iterations(10)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 2048);
    cmd.get_cmd_line_argument("n", n, 2048);
    cmd.get_cmd_line_argument("k", k, 2048);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 10);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);


    if (cmd.check_cmd_line_flag("no_verif")) {
      verification = false;
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "92_blackwell_moe_gemm_fp4_grouped\n\n"
      << "  Blackwell MoE-style grouped NVFP4 GEMM implementation using TMA to load A and CPASYNC to load B\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --iterations=<int>          Set the number of profiling iterations to perform\n"
      << "  --benchmark=<file>          Executes a benchmark problem size\n"
      << "  --no_verif                  Do not run verification kernels\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Element, class Layout>
bool initialize_block(
    cutlass::TensorView<Element, Layout> view,
    uint64_t seed) {

  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t> || cute::is_same_v<Element, cutlass::float_ue4m3_t>) {
      scope_max = 4;
      scope_min = 1;
    }
    else {
      scope_max = 1;
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;
    scope_min = -4;
  }

  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

template <class T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct ExampleRunner {
  // Type of kernel schedule to generate
  using MainloopScheduleType = cutlass::gemm::KernelMixedTmaCpAsyncWarpSpecialized1SmBlockScaledSm100;
  // Type of epilogue schedule to generate
  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  static constexpr bool FuseQuantization = false;

  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::ColumnMajor;
  using LayoutDTag = cutlass::layout::ColumnMajor;
  using LayoutSFDTag = LayoutDTag;                                    // Layout type for SFD should be same as D matrix operand

  using ElementInput = cutlass::float_e2m1_t;                                // Element type for Input matrix operands
  using ElementSF    = cutlass::float_ue4m3_t;                               // Element type for SF matrix operands

  using ElementA = cutlass::nv_float4_t<ElementInput>;
  using ElementB = cutlass::nv_float4_t<ElementInput>;
  using ElementC = cutlass::half_t;
  using ElementD = cute::conditional_t<FuseQuantization, ElementInput, ElementC>;
  using ElementSFD = ElementSF;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;

  

  using ClusterShapeMNK = Shape<_1,_1,_1>;
  using MmaTileMNK    = Shape<_128,_64,_256>;  // use tile size of N=64 to match real use cases (N is typically very small in decoding stage)

  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  static constexpr int OutputSFVectorSize = 16;

  // D = alpha * acc + beta * C
  //      With BlockScaleFactor generation.
  using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      OutputSFVectorSize,
      ElementD,
      ElementCompute,
      ElementSFD, LayoutSFDTag,
      ElementC>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTileMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      EpilogueScheduleType,
      cute::conditional_t<
        FuseQuantization, 
        FusionOperation, 
        cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      MmaTileMNK, ClusterShapeMNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopScheduleType
    >::CollectiveOp;

  using ProblemShapeGroup = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
  using ProblemShapeMax = Shape<int,int,int,int>; // max <M,N,K,L>
  using ProblemShape = cutlass::gemm::MoEProblemShape<ProblemShapeGroup, ProblemShapeMax>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA   = typename Gemm::GemmKernel::StrideA;
  using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
  using StrideB   = typename Gemm::GemmKernel::StrideB;
  using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
  using StrideC   = typename Gemm::GemmKernel::StrideC;
  using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
  using StrideD   = typename Gemm::GemmKernel::StrideD;
  using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

  using FusionOp = typename Gemm::EpilogueOutputOp;
  static constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;
  using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
  using LayoutSFD = typename SfdOutputCfg::LayoutSF;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  LayoutA layout_A;
  LayoutSFA layout_SFA;
  StrideB stride_B;
  LayoutB layout_B;
  LayoutSFB layout_SFB;
  StrideC stride_C;
  LayoutC layout_C;
  StrideD stride_D;
  LayoutD layout_D;
  LayoutSFD layout_SFD;
  uint64_t seed = 0;

  cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
  cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
  cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
  cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
  cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout> block_SFD;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;
  cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout> block_reference_SFD;
  cutlass::HostTensor<ElementCompute, cutlass::layout::PackedVectorLayout> block_Normconst;

  cutlass::DeviceAllocation<typename ProblemShapeGroup::UnderlyingProblemShape> problem_sizes;

  //
  // Methods
  //

  bool verify(ProblemShape const& problem_size, float alpha, float beta) {
    // Create the arguments for host reference implementation
    Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
    Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
    Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
    Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);

    // think about how to simplify the gemm3x interface.
    cutlass::reference::host::GettBlockScalingMainloopParams<
        ElementAccumulator,                   // ElementAccumulator
        decltype(tensor_A),                   // TensorA
        decltype(tensor_SFA),                 // TensorSfA
        decltype(tensor_B),                   // TensorB
        decltype(tensor_SFB)                  // TensorSfB
      > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

    Tensor tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
    Tensor tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);
    Tensor tensor_SFD = make_tensor(block_reference_SFD.host_data(), layout_SFD);

    if constexpr (FuseQuantization) {
      cutlass::reference::host::GettBlockScalingEpilogueParams<
          ElementCompute,                       // ElementScalar
          ElementAccumulator,                   // ElementAccumulator
          ElementCompute,                       // ElementCompute
          decltype(tensor_C),                   // TensorC
          decltype(tensor_D),                   // TensorD
          decltype(tensor_SFD),                 // TensorSfD
          cute::Int<OutputSFVectorSize>,
          cutlass::reference::host::SfStrategy::SfDGen
        > epilogue_params {alpha, beta, tensor_C, tensor_D, tensor_SFD, block_Normconst.at(cutlass::make_Coord(0))};

      cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
    } 
    else {
      cutlass::reference::host::GettBlockScalingEpilogueParams<
          ElementCompute,                       // ElementScalar
          ElementAccumulator,                   // ElementAccumulator
          ElementCompute,                       // ElementCompute
          decltype(tensor_C),                   // TensorC
          decltype(tensor_D)                   // TensorD
        > epilogue_params {alpha, beta, tensor_C, tensor_D };

      cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
    }

    bool passed = true;

    // Comparison
    block_D.sync_host();

    auto [maxM, maxN, maxK, L] = problem_size.max_problem_shape;
    for (int i = 0; i < problem_size.problem_shape.num_groups; i++) {
      auto problem = problem_size.problem_shape.get_host_problem_shape(i);
      auto [M, N, K] = problem;

      // assume all M == maxM
      auto refD_view = block_reference_D.host_view().subview(cutlass::make_Coord(M * N), cutlass::make_Coord(i * maxN * maxM));
      auto D_view = block_D.host_view().subview(cutlass::make_Coord(M * N), cutlass::make_Coord(i * maxN * maxM));

      passed &= cutlass::reference::host::TensorEquals(refD_view, D_view);
    }

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(ProblemShape const& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size.max_problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    // For SFA and SFB tensors layouts
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    // For SFD tensor layout
    using Sm1xxBlockScaledOutputConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    // printf("\nStrideC = ");
    // print(StrideC{});

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, L});
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, L});
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, L});
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, L});

    // printf("\nstride_C = ");
    // print(stride_C);

    layout_A = make_layout(make_shape(M, K, L), stride_A);
    layout_B = make_layout(make_shape(N, K, L), stride_B);
    layout_C = make_layout(make_shape(M, N, L), stride_C);
    layout_D = make_layout(make_shape(M, N, L), stride_D);
    layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
    layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));
    layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape(M, N, K, L));

    // printf("\nlayout_A = ");
    // print(layout_A);
    // printf("\nlayout_B = ");
    // print(layout_B);
    // printf("\nlayout_C = ");
    // print(layout_C);

    // printf("\nsize(layout_A)=%lld", (long long)size(layout_A));
    // printf("\n");

    block_A.reset(cutlass::make_Coord(size(layout_A)));
    block_B.reset(cutlass::make_Coord(size(layout_B)));
    block_C.reset(cutlass::make_Coord(size(layout_C)));
    block_D.reset(cutlass::make_Coord(size(layout_D)));
    block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
    block_reference_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
    block_Normconst.reset(cutlass::make_Coord(1));

    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
    block_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));

    initialize_block(block_A.host_view(), seed + 2021);
    initialize_block(block_B.host_view(), seed + 2022);
    initialize_block(block_C.host_view(), seed + 2023);
    initialize_block(block_SFA.host_view(), seed + 2024);
    initialize_block(block_SFB.host_view(), seed + 2025);
    block_Normconst.at(cutlass::make_Coord(0)) = 2;

    block_A.sync_device();
    block_B.sync_device();
    block_C.sync_device();
    block_D.sync_device();
    block_SFA.sync_device();
    block_SFB.sync_device();
    block_SFD.sync_device();
    block_Normconst.sync_device();
  }

  /// Load a benchmark
  std::vector<ProblemShapeGroup::UnderlyingProblemShape> benchmark_problems(std::string const& benchmark_path) {
    std::vector<ProblemShapeGroup::UnderlyingProblemShape> problem_sizes_host;

    std::ifstream file(benchmark_path);
    if (!file.good()) {
      return {};
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
        extent.at(i) = std::atoi(tokens.at(i).c_str());
      }
      problem_sizes_host.push_back({extent.m(), extent.n(), extent.k()});
    }

    return problem_sizes_host;
  }

  bool run(Options const& options, cutlass::KernelHardwareInfo const& hw_info) {
    auto problem_sizes_host = benchmark_problems(options.benchmark_path);
    if (problem_sizes_host.empty()) {
      return false;
    }

    problem_sizes.reset(problem_sizes_host.size());
    problem_sizes.copy_from_host(problem_sizes_host.data());

    ProblemShape problem_size;
    problem_size.max_problem_shape = ProblemShapeMax{options.m, options.n, options.k, options.l};
    problem_size.problem_shape.num_groups = problem_sizes_host.size();
    problem_size.problem_shape.problem_shapes = problem_sizes.get();
    problem_size.problem_shape.host_problem_shapes = problem_sizes_host.data();

    initialize(problem_size);

    typename Gemm::Arguments arguments {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      problem_size,
      { // Mainloop arguments
        block_A.device_data(), stride_A,
        block_B.device_data(), stride_B,
        block_SFA.device_data(), layout_SFA,
        block_SFB.device_data(), layout_SFB
      },
      { // Epilogue arguments
        {},
        block_C.device_data(), stride_C,
        block_D.device_data(), stride_D
      },
      hw_info
    };

    auto f = [&](auto blockscale) {
      auto impl = [this](auto& arguments) {
        arguments.epilogue.thread.block_scale_factor_ptr = block_SFD.device_data();
        arguments.epilogue.thread.norm_constant_ptr      = block_Normconst.device_data();
      };
      if constexpr (decltype(blockscale)::value) {
        impl(arguments);
      }
    };
    f(std::bool_constant<IsBlockScaleSupported>());

    // arguments.scheduler.max_swizzle_size = options.swizzle;
    
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    // Run the GEMM
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    if (options.verification) {
      // Verify that the result is correct
      bool passed = verify(problem_size, 1.0f, 0.0f);

      std::cout << "  Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

      if (!passed) {
        exit(-1);
        return false;
      }
    }

    // Run profiling loop
    if (options.iterations > 0)
    {
      GpuTimer timer;
      timer.start();
      for (int iter = 0; iter < options.iterations; ++iter) {
        CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
        CUTLASS_CHECK(gemm_op.run());
      }
      timer.stop();

      // Compute average setup and runtime and FLOPs.
      float elapsed_ms       = timer.elapsed_millis();
      double avg_runtime_ms  = double(elapsed_ms) / double(options.iterations);
      double flops           = double(int64_t(2) * options.m * options.n * options.k * options.l) / (avg_runtime_ms / 1000.0);

      std::cout << "  Avg runtime : " << avg_runtime_ms << " ms" << std::endl;
      std::cout << "  TFLOPS      : " << flops / 1e12 << std::endl;
    }

    return true;
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

  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
  
  if (!(props.major == 10 && props.minor == 0)) {
    std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 100)." << std::endl;
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
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  std::cout << "Running kernel with mixed TMA+CPASYNC load:" << std::endl;
  ExampleRunner runner_mixed_tma_cpasync;
  runner_mixed_tma_cpasync.run(options, hw_info);

#endif

  return 0;
}
