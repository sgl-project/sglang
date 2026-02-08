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
  \brief Example of a Hopper gather+GEMM+scatter kernel fusion.

  This example fuses gather before GEMM and scatter after GEMM into the same
  GEMM kernel. Gather and scatter operation is controlled by an index vector
  to select rows or columns from A, B, C or D matrices.

  Gather/scatter operations are always performed along a strided dimension 
  in order to preserve vectorized loads/stores. Thus the index vector is 
  applied to rows of row-major matrices and columns of column-major matrices.

  Note that the index vector must contain integers in range [0,X) where
  X is one of (M,N,K), depending on selected gather dimension. The problem
  shape given to the GEMM kernel must consist of matrix sizes AFTER gather
  and BEFORE scatter operations are applied.
*/

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
#include "gather_gemm.hpp"
#include "gather_kernel.cuh"
#include "scatter_epilogue.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;

namespace example {

// Command line options parsing
struct Options {

  bool help = false;

  cutlass::gemm::BatchedGemmCoord problem_size = {2048, 2048, 2048, 1};
  int index_size = 1024;
  int mode = 1; // N-mode gather/scatter by default

  float alpha = 1.0f;
  float beta  = 0.0f;

  bool reference_check = true;
  int iterations = 20;

  bool valid() const {
    return problem_size.m() > 0
        && problem_size.n() > 0
        && problem_size.k() > 0
        && problem_size.batch() > 0
        && 0 <= mode && mode < 3
        && index_size <= problem_size.at(mode)
        && iterations > 0;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("batch_size", problem_size.batch());
    cmd.get_cmd_line_argument("index_size", index_size);

    char const modes[] = {'m', 'n', 'k'};
    char mode_input = modes[mode];
    cmd.get_cmd_line_argument("mode", mode_input);
    mode = int(std::distance(std::begin(modes), std::find(std::begin(modes), std::end(modes), mode_input)));

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

    cmd.get_cmd_line_argument("check", reference_check, true);
    cmd.get_cmd_line_argument("iterations", iterations);

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << 
      "52_hopper_gather_scatter_fusion example\n"
      "\n"
      "  This example uses the CUTLASS Library to fuse gather/scatter of input/output tensors with GEMM.\n"
      "  It validates and benchmarks the fused kernel against an unfused implementation that executes\n"
      "  gather+GEMM+scatter in sequence and writes intermediate (gathered) tensors to memory.\n"
      "  For the unfused implementation two GEMM kernels are considered: default one that uses the same\n"
      "  schedule and instruction set as the fused one, and an optimized one that utilizes advanced\n"
      "  features (such as TMA units) that cannot be used by the fused kernel due to hardware constraints."
      "\n"
      "Options:\n"
      "  --help                      If specified, displays this usage statement.\n"
      "  --m=<int>                   GEMM M dimension\n"
      "  --n=<int>                   GEMM N dimension\n"
      "  --k=<int>                   GEMM K dimension\n"
      "  --batch_size=<int>          GEMM batch size\n"
      "  --index_size=<int>          Size of N dimension gather/scatter index\n"
      "  --mode=<m,n,k>              Gather mode (M, N, or K)\n"
      "  --alpha=<float>             GEMM alpha parameter\n"
      "  --beta=<float>              GEMM beta parameter\n"
      "  --iterations=<int>          Number of profiling iterations to perform.\n"
      "\n"
      "Examples:\n"
      "\n"
      "$ ./examples/52_hopper_gather_scatter_fusion/52_hopper_gather_scatter_fusion --m=1024 --n=2048 --k=1024 --mode=n --index_size=1024\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template<class ElementA, class LayoutA, class GatherA,
         class ElementB, class LayoutB, class GatherB,
         class ElementC, class LayoutC, class GatherC,
         class ElementD, class LayoutD, class ScatterD,
         class ElementAccumulator, class ElementComputeEpilogue>
struct ExampleRunner
{
  // Useful aliases

  using ProblemShape = Shape<int,int,int,int>;

  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
  using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
  using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

  // Alias to for the epilogue type that supports gather/scatter
  using Epilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::EpilogueGatherScatter<
      StrideC, StrideD,
      cutlass::epilogue::thread::LinearCombination<
        ElementD, 1,
        ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::Default,
        cutlass::FloatRoundStyle::round_to_nearest, ElementC
      >,
      cutlass::gemm::EpilogueDefault,
      GatherC,
      ScatterD
    >
  >;

  // Alias to for the mainloop type
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
    ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementB>::value,
    ElementAccumulator,
    Shape<_128,_128,_64>,
    Shape<_1,_1,_1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelCpAsyncWarpSpecialized
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmGather<
    ProblemShape,
    Mainloop,
    Epilogue,
    void,
    GatherA,
    GatherB
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static constexpr bool DoGatherA  = not cutlass::platform::is_same<GatherA,  NoGather>::value;
  static constexpr bool DoGatherB  = not cutlass::platform::is_same<GatherB,  NoGather>::value;
  static constexpr bool DoGatherC  = not cutlass::platform::is_same<GatherC,  NoGather>::value;
  static constexpr bool DoScatterD = not cutlass::platform::is_same<ScatterD, NoGather>::value;

  static constexpr bool GatherAonM  = DoGatherA  && cutlass::platform::is_same<LayoutA,cutlass::layout::RowMajor>::value;
  static constexpr bool GatherAonK  = DoGatherA  && cutlass::platform::is_same<LayoutA,cutlass::layout::ColumnMajor>::value;
  static constexpr bool GatherBonN  = DoGatherB  && cutlass::platform::is_same<LayoutB,cutlass::layout::ColumnMajor>::value;
  static constexpr bool GatherBonK  = DoGatherB  && cutlass::platform::is_same<LayoutB,cutlass::layout::RowMajor>::value;
  static constexpr bool GatherConM  = DoGatherC  && cutlass::platform::is_same<LayoutC,cutlass::layout::RowMajor>::value;
  static constexpr bool GatherConN  = DoGatherC  && cutlass::platform::is_same<LayoutC,cutlass::layout::ColumnMajor>::value;
  static constexpr bool ScatterDonM = DoScatterD && cutlass::platform::is_same<LayoutD,cutlass::layout::RowMajor>::value;
  static constexpr bool ScatterDonN = DoScatterD && cutlass::platform::is_same<LayoutD,cutlass::layout::ColumnMajor>::value;

  static constexpr bool GatherModeM = GatherAonM || GatherConM || ScatterDonM;
  static constexpr bool GatherModeN = GatherBonN || GatherConN || ScatterDonN;
  static constexpr bool GatherModeK = GatherAonK || GatherBonK;

  static_assert( GatherModeM && !GatherModeN && !GatherModeK ||
                !GatherModeM &&  GatherModeN && !GatherModeK ||
                !GatherModeM && !GatherModeN &&  GatherModeK,
                "Only one gather mode (M, N or K) is supported by example runner");

  // Construct a reference (non-gather) GEMM kernel type

  using MainloopRef = Mainloop;

  using EpilogueRef = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::DefaultEpilogue<
      ElementC, StrideC, StrideD,
      typename Epilogue::ThreadEpilogueOp,
      typename Epilogue::EpilogueSchedule
    >
  >;

  using KernelRef = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    MainloopRef,
    EpilogueRef,
    void
  >;

  using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<KernelRef>;

  // Construct an optimized reference GEMM kernel type (using TMA)

  using EpilogueOpt = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128,_128,_64>,
    Shape<_2,_2,_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementComputeEpilogue,
    ElementC, LayoutC, 128 / cutlass::sizeof_bits<ElementC>::value,
    ElementD, LayoutD, 128 / cutlass::sizeof_bits<ElementD>::value,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using MainloopOpt = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
    ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementB>::value,
    ElementAccumulator,
    Shape<_128,_128,_64>,
    Shape<_2,_2,_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename EpilogueOpt::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using KernelOpt = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    MainloopOpt,
    EpilogueOpt,
    void
  >;

  using GemmOpt = cutlass::gemm::device::GemmUniversalAdapter<KernelOpt>;

  // Data members

  cutlass::gemm::BatchedGemmCoord problem_size_orig;
  cutlass::gemm::BatchedGemmCoord problem_size;
  ProblemShape problem_shape_orig;
  ProblemShape problem_shape;
  cutlass::KernelHardwareInfo hw_info;

  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  StrideA stride_A_orig;
  StrideB stride_B_orig;
  StrideC stride_C_orig;
  StrideD stride_D_orig;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  cutlass::device_memory::allocation<ElementA> tensor_a;
  cutlass::device_memory::allocation<ElementB> tensor_b;
  cutlass::device_memory::allocation<ElementC> tensor_c;
  cutlass::device_memory::allocation<ElementD> tensor_d;

  cutlass::device_memory::allocation<int> gather_indices;

  cutlass::device_memory::allocation<ElementA> tensor_a_gathered;
  cutlass::device_memory::allocation<ElementB> tensor_b_gathered;
  cutlass::device_memory::allocation<ElementC> tensor_c_gathered;
  cutlass::device_memory::allocation<ElementD> tensor_d_gathered;
  cutlass::device_memory::allocation<ElementD> tensor_d_reference;

  cutlass::gemm::GemmUniversalMode gemm_mode;

  Gemm gemm;
  typename Gemm::Arguments arguments;
  cutlass::device_memory::allocation<uint8_t> workspace;

  GemmRef gemm_ref;
  typename GemmRef::Arguments arguments_ref;
  cutlass::device_memory::allocation<uint8_t> workspace_ref;

  GemmOpt gemm_opt;
  typename GemmOpt::Arguments arguments_opt;
  cutlass::device_memory::allocation<uint8_t> workspace_opt;

  ExampleRunner(Options const &options, cutlass::KernelHardwareInfo const &hw_info)
  : problem_size_orig(options.problem_size),
    problem_size(GatherModeM ? options.index_size : problem_size_orig.m(),
                 GatherModeN ? options.index_size : problem_size_orig.n(),
                 GatherModeK ? options.index_size : problem_size_orig.k(),
                 problem_size_orig.batch()),
    problem_shape_orig(problem_size_orig.m(), problem_size_orig.n(), problem_size_orig.k(), problem_size_orig.batch()),
    problem_shape(problem_size.m(), problem_size.n(), problem_size.k(), problem_size.batch()),
    hw_info(hw_info),
    alpha(options.alpha),
    beta(options.beta),
    stride_A_orig(cutlass::make_cute_packed_stride(
        StrideA{}, make_shape(problem_size_orig.m(), problem_size_orig.k(), problem_size_orig.batch()))),
    stride_B_orig(cutlass::make_cute_packed_stride(
        StrideB{}, make_shape(problem_size_orig.n(), problem_size_orig.k(), problem_size_orig.batch()))),
    stride_C_orig(cutlass::make_cute_packed_stride(
        StrideC{}, make_shape(problem_size_orig.m(), problem_size_orig.n(), problem_size_orig.batch()))),
    stride_D_orig(cutlass::make_cute_packed_stride(
        StrideD{}, make_shape(problem_size_orig.m(), problem_size_orig.n(), problem_size_orig.batch()))),
    stride_A(cutlass::make_cute_packed_stride(
        StrideA{}, make_shape(problem_size.m(), problem_size.k(), problem_size.batch()))),
    stride_B(cutlass::make_cute_packed_stride(
        StrideB{}, make_shape(problem_size.n(), problem_size.k(), problem_size.batch()))),
    stride_C(cutlass::make_cute_packed_stride(
        StrideC{}, make_shape(problem_size.m(), problem_size.n(), problem_size.batch()))),
    stride_D(cutlass::make_cute_packed_stride(
        StrideD{}, make_shape(problem_size.m(), problem_size.n(), problem_size.batch()))),
    tensor_a(problem_size_orig.m() * problem_size_orig.k() * problem_size_orig.batch()),
    tensor_b(problem_size_orig.k() * problem_size_orig.n() * problem_size_orig.batch()),
    tensor_c(problem_size_orig.m() * problem_size_orig.n() * problem_size_orig.batch()),
    tensor_d(problem_size_orig.m() * problem_size_orig.n() * problem_size_orig.batch()),
    gather_indices(options.index_size),
    tensor_a_gathered(problem_size.m() * problem_size.k() * problem_size_orig.batch()),
    tensor_b_gathered(problem_size.k() * problem_size.n() * problem_size_orig.batch()),
    tensor_c_gathered(problem_size.m() * problem_size.n() * problem_size_orig.batch()),
    tensor_d_gathered(problem_size.m() * problem_size.n() * problem_size_orig.batch()),
    tensor_d_reference(problem_size_orig.m() * problem_size_orig.n() * problem_size_orig.batch()),
    gemm_mode(problem_size.batch() > 1 ?  cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm),
    gemm(),
    // When constructing arguments for gather/scatter gemm, we must pass stride arguments
    // made for the original (non-gathered) problem size, because they are used to access
    // tensors of the original shape. However we still use the reduced (gathered) problem
    // shape since it corresponds to the logical indexing in reduced size GEMM.
    arguments{
      gemm_mode,
      problem_shape,
      {
        tensor_a.get(),
        stride_A_orig,
        tensor_b.get(),
        stride_B_orig
      },
      { 
        { alpha, beta },
        tensor_c.get(), stride_C_orig,
        tensor_d.get(), stride_D_orig,
        typename Epilogue::GatherC {gather_indices.get()},
        typename Epilogue::ScatterD{gather_indices.get()}
      },
      hw_info,
      {},
      typename Kernel::GatherA{gather_indices.get()},
      typename Kernel::GatherB{gather_indices.get()}
    },
    workspace(Gemm::get_workspace_size(arguments)),
    gemm_ref(),
    arguments_ref{
      gemm_mode,
      problem_shape,
      {
        DoGatherA ? tensor_a_gathered.get() : tensor_a.get(),
        stride_A,
        DoGatherB ? tensor_b_gathered.get() : tensor_b.get(),
        stride_B
      },
      { 
        { alpha, beta },
        DoGatherC  ? tensor_c_gathered.get() : tensor_c.get(),
        stride_C, 
        DoScatterD ? tensor_d_gathered.get() : tensor_d_reference.get(),
        stride_D
      },
      hw_info
    },
    workspace_ref(GemmRef::get_workspace_size(arguments_ref)),
    gemm_opt(),
    arguments_opt{
      gemm_mode,
      problem_shape,
      {
        DoGatherA ? tensor_a_gathered.get() : tensor_a.get(),
        stride_A,
        DoGatherB ? tensor_b_gathered.get() : tensor_b.get(),
        stride_B
      },
      { 
        { alpha, beta },
        DoGatherC  ? tensor_c_gathered.get() : tensor_c.get(),
        stride_C, 
        DoScatterD ? tensor_d_gathered.get() : tensor_d_reference.get(),
        stride_D
      },
      hw_info
    },
    workspace_opt(GemmOpt::get_workspace_size(arguments_opt))
  {
    // Fill input and output matrices on host using CUTLASS helper functions
    cutlass::reference::device::BlockFillRandomUniform(tensor_a.get(), tensor_a.size(), 1, ElementA(7), ElementA(-8), 0);
    cutlass::reference::device::BlockFillRandomUniform(tensor_b.get(), tensor_b.size(), 1, ElementB(7), ElementB(-8), 0);
    cutlass::reference::device::BlockFillRandomUniform(tensor_c.get(), tensor_c.size(), 1, ElementC(7), ElementC(-8), 0);
    cutlass::reference::device::BlockFillSequential(tensor_d.get(), tensor_d.size(), ElementD(0), ElementD(0));

    // <- Fill gather_indices with unique random integers in range [0,n)
    int index_range = GatherModeM ? problem_size_orig.m() : (GatherModeN ? problem_size_orig.n() : problem_size_orig.k());
    std::vector<int> indices(index_range);
    std::iota(indices.begin(), indices.end(), 0);
    { // std::random_shuffle was deprecated in C++14 and removed in C++17
      std::random_device make_seed;
      std::mt19937 source_of_randomness(make_seed());
      std::shuffle(indices.begin(), indices.end(), source_of_randomness);
    }
    gather_indices.copy_from_host(indices.data());

    auto const gemm_init = [](auto & gemm, auto const & arguments, auto & workspace)
    {
      cutlass::Status status = gemm.can_implement(arguments);
      CUTLASS_CHECK(status);
      status = gemm.initialize(arguments, workspace.get());
      CUTLASS_CHECK(status);
    };

    gemm_init(gemm,     arguments,     workspace    );
    gemm_init(gemm_ref, arguments_ref, workspace_ref);
    gemm_init(gemm_opt, arguments_opt, workspace_opt);
  }

  void debug_output(std::ostream & os)
  {
    auto print_tensor = [](std::ostream &os, char const * name, auto const & data, auto shape, auto stride)
    {
      std::vector<remove_cvref_t<decltype(*data.get())>> h_data(data.size());
      data.copy_to_host(h_data.data());
      Tensor t = make_tensor(h_data.data(), shape, stride);
      os << "\n" << name << ": " << std::setw(4) << t << std::endl;
    };
    {
      auto [M,N,K,L] = problem_shape_orig;
      print_tensor(os, "A", tensor_a, make_shape(M,K,L), stride_A_orig);
      print_tensor(os, "B", tensor_b, make_shape(N,K,L), stride_B_orig);
      print_tensor(os, "C", tensor_c, make_shape(M,N,L), stride_C_orig);
      print_tensor(os, "D", tensor_d, make_shape(M,N,L), stride_D_orig);
      print_tensor(os, "D reference", tensor_d_reference, make_shape(M,N,L), stride_D_orig);
      print_tensor(os, "indices", gather_indices, make_shape(gather_indices.size()), make_stride(_1{}));
    }
  }

  template<class Gemm2>
  static void run_gemm(Gemm2 &gemm)
  {
    cutlass::Status status = gemm.run();
    CUTLASS_CHECK(status);
  }

  template<class Gemm2>
  void run_reference(Gemm2 &gemm)
  {
    // Convenience wrapper around calls to separate gather/scatter kernels
    auto run_gather = [this](auto call, auto const & input, auto & output, auto gather_func, auto batch_size, auto stride)
    {
      [[maybe_unused]] auto idx = find_if(stride, [](auto x){ return not is_constant<1, decltype(x)>{}; });
      constexpr int I = decltype(idx)::value;
      call(input.get(), 
           output.get(),
           gather_func,
           batch_size,
           static_cast<int>(input.size() / batch_size),
           static_cast<int>(output.size() / batch_size),
           static_cast<int>(get<I>(stride)),
           hw_info);
    };

    // Forward calls via lambda to avoid specifying template arguments
    auto gather_call = [](auto&&... args){ gather(static_cast<decltype(args)&&>(args)...); };
    // MSVC doesn't count use inside a false "if constexpr" branch.
    [[maybe_unused]] auto scatter_call = [](auto&&... args){ scatter(static_cast<decltype(args)&&>(args)...); };

    if constexpr (DoGatherA) {
      run_gather(gather_call, tensor_a, tensor_a_gathered, arguments.gather_A, problem_size.batch(), stride_A);
    }
    if constexpr (DoGatherB) {
      run_gather(gather_call, tensor_b, tensor_b_gathered, arguments.gather_B, problem_size.batch(), stride_B);
    }
    if constexpr (DoGatherC) {
      if (beta != ElementComputeEpilogue(0)) {
        run_gather(gather_call, tensor_c, tensor_c_gathered, arguments.epilogue.gather_C, problem_size.batch(), stride_C);
      }
    }

    run_gemm(gemm);

    if constexpr (DoScatterD) {
      run_gather(scatter_call, tensor_d_gathered, tensor_d_reference, arguments.epilogue.scatter_D, problem_size.batch(), stride_D);
    }
  }

  bool verify()
  {
    run_gemm(gemm);
    run_reference(gemm_ref);
    cudaDeviceSynchronize();
    return cutlass::reference::device::BlockCompareEqual(tensor_d.get(), tensor_d_reference.get(), tensor_d.size());
  }

  bool run(Options const &options)
  {
    if (options.reference_check) {
      if (!verify()) {
        std::cout << "Failed validation" << std::endl;
#if 0
        debug_output(std::cout);
#endif
        return false;
      } 
      else {
        std::cout << "Passed validation" << std::endl;
      }
    }

    //
    // Run profiling loop
    //

    auto const benchmark = [&](auto name, auto func)
    {
      GpuTimer timer;
      timer.start();
      for (int iter = 0; iter < options.iterations; ++iter) {
        func();
      }
      timer.stop();

      double runtime = timer.elapsed_millis() / double(options.iterations);
      double gflops = 2 * double(problem_size.product()) / 1e6 / runtime; // Two flops per multiply-add

      std::cout << name << ":\n";
      std::cout << "  Runtime: " << runtime << " ms\n";
      std::cout << "   GFLOPs: " << gflops  << "\n";
    };

    benchmark("Fused", [&](){ run_gemm(gemm); });
    benchmark("Unfused default", [&](){ run_reference(gemm_ref); });
    benchmark("Unfused optimized", [&](){ run_reference(gemm_opt); });

    return true;
  }
};

} // namespace example

int main(int argc, const char ** argv) {

  bool notSupported = false;

  // CUDA 12 minimum required
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA Toolkit version 12 or later.\n";
    notSupported = true;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (props.major < 9) {
    std::cerr << "This example requires a device with compute capability 90 or higher.\n";
    notSupported = true;
  }
  
  else if (props.major != 9 || props.minor != 0) {
    std::cerr << "This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n";
    notSupported = true;
  }
  

  if (notSupported) {
    return EXIT_SUCCESS; // Do not fail CI checks on unsupported systems
  }

  example::Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << "\n";
    return EXIT_SUCCESS;
  }

  if (!options.valid()) {
    std::cerr << "Invalid arguments." << "\n";
    return EXIT_FAILURE;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  bool result = true;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  switch (options.mode) {
    using namespace example;
    case 0: {
      std::cout << "Gather A,C + scatter D on M mode:" << std::endl;
      using Runner = ExampleRunner<
        cutlass::half_t, cutlass::layout::RowMajor,    IndexedGather<int>, // A
        cutlass::half_t, cutlass::layout::ColumnMajor, NoGather,           // B
        cutlass::half_t, cutlass::layout::RowMajor,    IndexedGather<int>, // C
        cutlass::half_t, cutlass::layout::RowMajor,    IndexedGather<int>, // D
        float, float>;
      result &= Runner(options, hw_info).run(options);
      break;
    }
    case 1: {
      std::cout << "Gather B,C + scatter D on N mode:" << std::endl;
      using Runner = ExampleRunner<
        cutlass::half_t, cutlass::layout::RowMajor,    NoGather,           // A
        cutlass::half_t, cutlass::layout::ColumnMajor, IndexedGather<int>, // B
        cutlass::half_t, cutlass::layout::ColumnMajor, IndexedGather<int>, // C
        cutlass::half_t, cutlass::layout::ColumnMajor, IndexedGather<int>, // D
        float, float>;
      result &= Runner(options, hw_info).run(options);
      break;
    }
    case 2: {
      std::cout << "Gather A,B on K mode:" << std::endl;
      using Runner = ExampleRunner<
        cutlass::half_t, cutlass::layout::ColumnMajor, IndexedGather<int>, // A
        cutlass::half_t, cutlass::layout::RowMajor,    IndexedGather<int>, // B
        cutlass::half_t, cutlass::layout::RowMajor,    NoGather,           // C
        cutlass::half_t, cutlass::layout::RowMajor,    NoGather,           // D
        float, float>;
      result &= Runner(options, hw_info).run(options);
      break;
    }
  }
#endif

  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
