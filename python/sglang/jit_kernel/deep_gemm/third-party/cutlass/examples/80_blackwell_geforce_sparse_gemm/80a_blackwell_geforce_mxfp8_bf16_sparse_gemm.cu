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
    \brief A GEMM example using CUTLASS for the NVIDIA Blackwell SM120 architecture.

    This example demonstrates a simple way to instantiate and run a narrow precision blockscaled sparse GEMM on the NVIDIA Blackwell SM120 architecture.
    This kernel is optimized for the GeForce RTX 50 series GPUs.

    The Blackwell SM120 CUTLASS kernel uses the new Block Scaled Sparse Tensor Core MMA Instructions:
      * mma.sync.aligned.kind::mxf8f6f4.sp::ordered_metadata.block_scale.
    Please see more detail in https://docs.nvidia.com/cuda/parallel-thread-execution.

    The kernel leverages:
    1. Warp-Specialized persistent kernel design that supports cooperative scheduler introduced in Hopper.
    2. The new SW controlled dynamic scheduler based on cluster launch control (See https://docs.nvidia.com/cuda/parallel-thread-execution).
    3. Block Scaled Sparse Tensor Core MMA Instructions

    Note that GeForce RTX 50 series GPUs do not support:
    1. Multicast feature of TMA load. Cluster shape has to be 1x1x1.
    2. Dynamic datatypes.

    Usage:
      $ ./examples/80_blackwell_geforce_sparse_gemm/80a_blackwell_geforce_mxfp8_bf16_sparse_gemm --m=2048 --n=2048 --k=2048
*/
#include <iostream>
#include "cutlass/cutlass.h"
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
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/transform/device/transform_universal_adapter.hpp"

#include "helper.h"
using namespace cute;
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
// A matrix configuration
using         ElementA    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;     // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                       // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                              // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;     // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                    // Layout type for B matrix operand
constexpr int AlignmentB  = 16;                                              // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                             // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                             // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                       // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                       // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;     // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;     // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// E matrix configuration. Note, E is used to represent metadata tensor.
using         ElementE    = uint8_t;                                         // Element type for E matrix operand
// Kernel functional config
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledSparseTensorOp; // Operator class tag
using KernelScheduleType =  cutlass::gemm::KernelSparseTmaWarpSpecializedMxf8f6f4Acc2x4Sm120;   // Kernel schedule policy
// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_256>;                           // Threadblock's tile size
using ClusterShape        = Shape<_1,_1,_1>;                                 // Shape of the threadblocks in a cluster
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
  >::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelScheduleType                                                       // Mainloop schedule policy
  >::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                  // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));
using LayoutE   = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
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
LayoutE layout_E;
uint64_t seed;
// The HostTensors are only used for allocating memory on host and device, and transferring data between host and device
// Use cute::Tensor and cute::Layout for iterating thru the matrix elements
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A_Decompressed;
cutlass::HostTensor<ElementE, cutlass::layout::PackedVectorLayout> block_E;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
// Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
// Reference Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;
#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////
// Command line options parsing
struct Options {
  bool help;
  float alpha, beta;
  int iterations;
  int m, n, k;
  Options():
    help(false),
    m(1024), n(1024), k(1024),
    alpha(1.f), beta(0.f),
    iterations(10)
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
  }
  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {
    out << "80a_blackwell_geforce_mxfp8_bf16_sparse_gemm\n\n"
      << "  Blackwell MXFP8 Sparse GEMM is a warp specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";
    out << "\n\nExamples:\n\n"
      << "$ " << "./examples/80_blackwell_geforce_sparse_gemm/80a_blackwell_geforce_mxfp8_bf16_sparse_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";
    return out;
  }
  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
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
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper to initialize a block of device data
template <typename Element, typename Layout>
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
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
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
/// Initialize blocks that released to sparse Matrix A and its metadata E
bool initialize_sparse_blocks(const Options &options) {
  auto workload = make_shape(options.m,
                             options.n,
                             options.k,
                             1);
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  /// Alias SparseConfig and Compressor
  using SparseConfig = typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;
  using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
                              cute::Shape<int, int, int, int>,
                              ElementA::DataType,
                              LayoutATag,
                              SparseConfig>;
  using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
                              cute::Shape<int, int, int, int>,
                              ElementA::DataType,
                              LayoutATag,
                              SparseConfig,
                              cutlass::arch::Sm120>;
  using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;
  /// Declare compressor_utility to randomly fill zero in Matrix A to match sparsity needs
  CompressorUtility compressor_utility(workload, stride_A);
  // Aligned M K dimension size for A and E
  int aligned_m_e = compressor_utility.get_metadata_m_physical();
  int aligned_k_e = compressor_utility.get_metadata_k_physical();
  int aligned_m_a = compressor_utility.get_tensorA_m_physical();
  int aligned_k_a = compressor_utility.get_tensorA_k_physical();
  /// Layout A and E
  layout_A = SparseConfig::fill_layoutA(workload);
  layout_E = SparseConfig::fill_layoutE(workload);

  block_A.reset(cutlass::make_Coord(aligned_m_a * aligned_k_a));
  block_E.reset(cutlass::make_Coord(aligned_m_e * aligned_k_e));
  block_A_Decompressed.reset(cutlass::make_Coord(options.m * options.k));
  initialize_block(block_A_Decompressed.host_view(), seed + 2020);
  compressor_utility.structure_sparse_zero_mask_fill(
          block_A_Decompressed.host_data(), static_cast<int>(seed + 2021));
  block_A_Decompressed.sync_device();

  /// Use compressor kernel to generate compressed Matrix A and E
  cutlass::Status status { cutlass::Status::kSuccess };
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  typename Compressor::Arguments arguments{
    {options.m, options.n, options.k, 1},
    {block_A_Decompressed.device_data(),
      stride_A,
      block_A.device_data(),
      block_E.device_data()},
    {hw_info}
  };

  // Compress A and E
  Compressor compressor_op;
  size_t workspace_size = Compressor::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  status = compressor_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  status = compressor_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  status = compressor_op.run();
  auto result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    return false;
  }

  block_A.sync_host();
  block_E.sync_host();
  return true;
}
/// Initialize operands to be used in the GEMM and reference GEMM
bool initialize(const Options &options) {
  using namespace cute;

  // Initial A, E(metadata) and A_compressed blocks
  if(!initialize_sparse_blocks(options)) return false;

  // Define B, C and D blocks
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});
  layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
  layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
  layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
  // Define SFA and SFB tensors layouts
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, 1));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  initialize_block(block_B.host_view(), seed + 2022);
  initialize_block(block_C.host_view(), seed + 2023);
  initialize_block(block_SFA.host_view(), seed + 2024);
  initialize_block(block_SFB.host_view(), seed + 2025);
  block_B.sync_device();
  block_C.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  return true;
}
// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    { // Mainloop arguments
      block_A.device_data(), layout_A,
      block_B.device_data(), stride_B,
      block_E.device_data(), layout_E,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB
    },
    { // Epilogue arguments
      {options.alpha, options.beta},
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D
    }
  };
  return arguments;
}
bool verify(const Options &options) {
  using namespace cute;
  // Create the arguments for host reference implementation
  Tensor tensor_A = make_tensor(make_iterator(block_A_Decompressed.host_data()), layout_A);
  Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
  Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
  Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);
  Tensor tensor_E = make_tensor(make_iterator(block_E.host_data()), layout_E);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,                 // ElementAccumulator
      decltype(tensor_A),                 // TensorA
      decltype(tensor_SFA),               // TensorSfA
      decltype(tensor_B),                 // TensorB
      decltype(tensor_SFB)                // TensorSfB
    > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};
  auto tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
  auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);

  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator,                   // ElementScalar
      ElementAccumulator,                   // ElementAccumulator
      ElementAccumulator,                   // ElementCompute
      decltype(tensor_C),                   // TensorC
      decltype(tensor_D)                    // TensorD
    > epilogue_params{options.alpha, options.beta, tensor_C, tensor_D};
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
  // Comparison
  block_D.sync_host();

  bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_reference_D.host_view());
  passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
  passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);
  return passed;
}
/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  // Initialization
  if(!initialize(options))
  {
    std::cerr << " Initialization failed! " << std::endl;
    exit(-1);
  }

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
  cudaDeviceSynchronize();
  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);
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
#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.8 or higher Toolkit for SM120 support,
  // or CUDA 12.9 or higher for SM121 support.
  // Must have compute capability at least 120.
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer for SM120 support." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#elif defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 9)) {
    std::cerr << "This example requires CUDA 12.9 or newer for SM121 support." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#endif
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));

  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {
    std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 120 or 121)." << std::endl;
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
  // Evaluate CUTLASS kernels
  //
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  run<Gemm>(options);
#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
