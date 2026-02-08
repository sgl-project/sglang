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

/*! \file
    \brief Hopper GEMM example to create a GEMM kernel with custom Collectives

    The following example shows how to assemble a custom GEMM kernel that spells out the Collectives
    directly instead of using a builder and, in the process, instance a more efficient Epilogue
    (from `cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp`) instead of using the default epilogue.

    The GemmUniversal API takes 3 main template arguments:
      (1) the problem shape / extents
      (2) the collective mainloop type
      (3) the collective epilogue type

    While the collecive mainloop can be stamped out using a CollectiveBuilder interface, it is
    possible to build a custom collective mainloop directly as well. Furthermore, since epilogues
    do not yet have a builder interface, this example shows how to instantiate a more-efficient
    epilogue alongside the collective mainloop.

    Note: there are several ways to implement the GEMM epilogue in Hopper - each with its own set
    of trade-offs. So it is recommended that users look at the options available under
    cutlass/epilogue/collective and evaluate for their particular scenario.

    Please refer to examples 48, 49 to learn more about kernel schedules and other CuTe examples
    present in `test/unit/cute` to famialiarize with the basics of CuTe.

    Examples:

      $ ./examples/50_hopper_gemm_with_epilogue_swizzle/50_hopper_gemm_with_epilogue_swizzle
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l;
  int alpha, beta;

  Options():
    help(false),
    error(false),
    m(2048), n(2048), k(2048), l(1),
    alpha(1), beta(0)
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
    cmd.get_cmd_line_argument("alpha", alpha, 1);
    cmd.get_cmd_line_argument("beta", beta, 0);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "50_hopper_gemm_with_epilogue_swizzle\n\n"
      << "Hopper GEMM Example with Epilogue Swizzle.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<s32>               Epilogue scalar alpha\n"
      << "  --beta=<s32>                Epilogue scalar beta\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else {
    scope_max = 8;
    scope_min = -8;
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// Wrapper to run and verify a GEMM.
template <
  class Gemm
>
struct ExampleRunner {

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAcc = typename Gemm::ElementAccumulator;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, int32_t alpha, int32_t beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          ElementCompute(alpha),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementCompute(beta),
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(M * K * L);
    block_B.reset(K * N * L);
    block_C.reset(M * N * L);
    block_D.reset(M * N * L);
    block_ref_D.reset(M * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  bool run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

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

    // Verify that the result is correct
    bool passed = verify(problem_size, options.alpha, options.beta);
    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
    }

    return passed;
  }

};

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 12 || props.major < 9) {
    std::cout
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater) and CUDA 12.0 or greater.\n";
      return 0;
  }
  
  else if (__CUDACC_VER_MAJOR__ < 12 || props.major != 9 || props.minor != 0) {
    std::cout
      << "This example requires a GPU of NVIDIA's Hopper Architecture "
      << "(compute capability 90) and CUDA 12.0 or greater.\n";
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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

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

  bool passed;

  // Problem configuration
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementAcc = int32_t;
  using ElementOutput = int8_t;

  // Note : Only TN WGMMA Gemm is supported currently in 3.0
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  // Tiling configuration selection
  using TileShape = Shape<_128,_64,_128>;

  // Choosing a thread block cluster larger than 1 allows us to Multicast data across thread blocks
  using ClusterShape = Shape<_1,_2,_1>;

  //
  // Assembling the CollectiveMainloop type
  //

  // Pipeline Depth to be used i.e number of A, B buffers in shared memory
  constexpr int PipelineStages = 8;

  // Let's choose a Warp-Specialized Mainloop implementation which uses TMA
  // Note : This requires / assumes the tensors to be 16B aligned
  using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape,
                           cutlass::gemm::KernelTmaWarpSpecialized>;

  // TN => K Major for both A & B
  static constexpr cute::GMMA::Major GmmaMajorA = cute::GMMA::Major::K;
  static constexpr cute::GMMA::Major GmmaMajorB = cute::GMMA::Major::K;

  // We use the SS op selector as both A, B operands are read directly from SMEM (for TN WGMMA)
  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementA, ElementB, ElementAcc, TileShape, GmmaMajorA, GmmaMajorB>()));

  // A loads can be optimized with multicast if cluster-n > 1
  using GmemTiledCopyA = std::conditional< cute::size(shape<1>(ClusterShape{})) == 1,
                           cute::SM90_TMA_LOAD,
                           cute::SM90_TMA_LOAD_MULTICAST>::type;

  // B loads can be optimized with multicast if cluster-m > 1
  using GmemTiledCopyB = std::conditional< cute::size(shape<0>(ClusterShape{})) == 1,
                           cute::SM90_TMA_LOAD,
                           cute::SM90_TMA_LOAD_MULTICAST>::type;

  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
      GmmaMajorA, ElementA, decltype(cute::get<0>(TileShape{})), decltype(cute::get<2>(TileShape{}))
    >());

  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
      GmmaMajorB, ElementB, decltype(cute::get<1>(TileShape{})), decltype(cute::get<2>(TileShape{}))
    >());

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      void, // Does not need a SmemCopyAtom, since A is read directly from SMEM
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      void, // Does not need a SmemCopyAtom, since B is read directly from SMEM
      cute::identity
    >;

  //
  // Assembling the Collective Epilogue Type
  //

  // Break the 128 along TILE_M into chunks of 32, to get a 128B leading dimension
  using PreSwizzleLayout = Layout< Shape< Shape <_32,_4   >,_64>,
                                   Stride<Stride< _1,_2048>,_32>>;

  // 128 threads loading 16 elements each (to get vectorized global stores)
  using TileShapeS2R = Shape<_128,_16>;

  // Layout to ensure bank-conflict free loads & stores
  using SmemLayout = ComposedLayout<
                       Swizzle<3,4,3>,
                       smem_ptr_flag_bits<sizeof_bits<ElementAcc>::value>,
                       PreSwizzleLayout>;

  // Tiled copy from Smem to Registers
  // Note : CuTe will vectorize this copy if the tiling + swizzling above were right
  using TiledCopyS2R = TiledCopy<
                         Copy_Atom<DefaultCopy, ElementAcc>,
                         Layout< Shape<_128,_16>, 
                                 Stride<_16,_1>>,
                         TileShapeS2R>;

  using Epilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
      SmemLayout,
      Copy_Atom<DefaultCopy, ElementAcc>,
      TiledCopyS2R,
      Copy_Atom<DefaultCopy, ElementOutput>>>;

  //
  // Assembling the GemmKernel
  //

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      Epilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  
  ExampleRunner<Gemm> runner;

  passed = runner.run(options, hw_info);

  std::cout << "WGMMA GEMM with Epilogue Swizzle : " << (passed ? "Passed" : "Failed") << std::endl;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
