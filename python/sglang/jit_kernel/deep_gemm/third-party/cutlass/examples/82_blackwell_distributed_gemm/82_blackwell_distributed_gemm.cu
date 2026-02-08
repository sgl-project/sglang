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
    \brief Distributed GEMM (DistGEMM) for Blackwell.

    This example runs Tensor Parallel GEMMs using the (experimental) Distributed GEMM API in 
    CUTLASS. For more information, please refer to README.md.

    Note that Distributed GEMM assumes an any-to-any NVLink network topology.
    To check whether your device is compatible, run:

      $ nvidia-smi topo -m

    and make sure there's an any-to-any NVLink topology. It would look like this:

                GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
        GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18
        GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18
        GPU2    NV18    NV18     X      NV18    NV18    NV18    NV18    NV18
        GPU3    NV18    NV18    NV18     X      NV18    NV18    NV18    NV18
        GPU4    NV18    NV18    NV18    NV18     X      NV18    NV18    NV18
        GPU5    NV18    NV18    NV18    NV18    NV18     X      NV18    NV18
        GPU6    NV18    NV18    NV18    NV18    NV18    NV18     X      NV18
        GPU7    NV18    NV18    NV18    NV18    NV18    NV18    NV18     X

    You should also additionally check if the driver enables peer to peer access:

      $ nvidia-smi topo -p2p r

    Output should be something like this:

               GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
        GPU0   X       OK      OK      OK      OK      OK      OK      OK
        GPU1   OK      X       OK      OK      OK      OK      OK      OK
        GPU2   OK      OK      X       OK      OK      OK      OK      OK
        GPU3   OK      OK      OK      X       OK      OK      OK      OK
        GPU4   OK      OK      OK      OK      X       OK      OK      OK
        GPU5   OK      OK      OK      OK      OK      X       OK      OK
        GPU6   OK      OK      OK      OK      OK      OK      X       OK
        GPU7   OK      OK      OK      OK      OK      OK      OK      X

    It is recommended to build this target with the following flag to enable 
    Grid Dependency Control instructions (GDC) in CUTLASS:
      - CUTLASS_ENABLE_GDC_FOR_SM100

    Example:

      $ mkdir build && cd build

      $ cmake .. -DCUTLASS_NVCC_ARCHS="100a" -DCUTLASS_ENABLE_GDC_FOR_SM100=1

      $ cd examples/82_blackwell_distributed_gemm

      $ make

      $ ./82_blackwell_distributed_gemm
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"

// Distributed GEMM headers
#include "cutlass/experimental/distributed/device/dist_gemm_universal_wrapper.hpp"
#include "cutlass/experimental/distributed/kernel/dist_gemm_kernel_wrapper.hpp"
#include "cutlass/experimental/distributed/schedules/dist_gemm_1d_schedules.hpp"

#include "helper.h"

// Distributed GEMM helpers
#include "dist_gemm_helpers.h"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Distributed GEMM configuration
/////////////////////////////////////////////////////////////////////////////////////////////////

// TP size (= number of processors/GPUs)
using TP = _8;
static constexpr int TP_ = TP{};

#if defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && \
  (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))

// Distributed GEMM tiling/sharding schedule
// Choices:
//
// * All Gather + GEMM:
//   * AllGather1D_TilingCD_RotatingA
//   * AllGather1D_TilingCD_RotatingB
//
// * GEMM + Reduce Scatter:
//   * ReduceScatter1D_TilingA_RotatingC
//   * ReduceScatter1D_TilingB_RotatingC

using DistSchedule = cutlass::distributed::schedules::AllGather1D_TilingCD_RotatingA<TP>;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::RowMajor;                      // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = cutlass::float_e4m3_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

using         ElementD    = cutlass::float_e4m3_t;                          // Element type for C and D matrix operands
using         LayoutD     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)

// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ArchTag             = cutlass::arch::Sm100;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag

// MMA and Cluster Tile Shapes
// Shape of the tile computed by tcgen05 MMA, could be across 2 SMs if Cluster Shape %2 == 0 
using MmaTileShape_MNK = Shape<_256,_256,_128>;                          
// Shape of the threadblocks in a cluster
using ClusterShape_MNK = Shape<_2,_1,_1>;
// Shape of the tile computed by each SM
using PerSmTileShape_MNK = Shape<_128, _256, _128>;

// Build the epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, 
    PerSmTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Build the mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100
  >::CollectiveOp;

// Compose into a kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int, int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;                   // Default to ClusterLaunchControl (CLC) based tile scheduler 

// We're going to use the single-device GEMM as reference
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Instantiate Distributed GEMM kernel
using DistGemmKernel = cutlass::distributed::kernel::DistributedGemmKernelWrapper<
  GemmKernel,
  DistSchedule
>;
using DistGemm = cutlass::distributed::device::DistributedGemmUniversalAdapter<DistGemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

using HostTensorA = typename cutlass::HostTensor<ElementA, LayoutA>;
using HostTensorB = typename cutlass::HostTensor<ElementB, LayoutB>;
using HostTensorC = typename cutlass::HostTensor<ElementC, LayoutC>;
using HostTensorD = typename cutlass::HostTensor<ElementD, LayoutD>;

// Reference GEMM tensors
HostTensorA tensor_A;
HostTensorB tensor_B;
HostTensorC tensor_C;
HostTensorD tensor_D;
HostTensorD tensor_ref_D;

// DistGEMM tensors (multi-device)
HostTensorA tensor_A_arr[TP_];
HostTensorB tensor_B_arr[TP_];
HostTensorD tensor_C_arr[TP_];
HostTensorD tensor_D_arr[TP_];

#endif // (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) &&
       // (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;

  float alpha = 1.f, beta = 0.f;
  int iterations = 100;
  int warmup_iterations = 10;
  int m = 16384, n = 106496, k = 16384, l = 1;
  float eps = 0.f;

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
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("warmup-iterations", warmup_iterations);
    cmd.get_cmd_line_argument("eps", eps);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "82_blackwell_distributed_gemm\n\n"
      << "  Blackwell Distributed GEMM (DistGEMM). \n"
      << "  For more details please refer to the source file.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch) of the GEMM (default: 1)\n"
      << "  --alpha=<f32>               Epilogue scalar alpha (default: 1.0)\n"
      << "  --beta=<f32>                Epilogue scalar beta (default: 0.0)\n"
      << "  --iterations=<int>          Number of profiling iterations to perform (default: 100)\n"
      << "  --warmup-iterations=<int>   Number of warmup iterations prior to profiling (default: 10)\n"
      << "  --eps=<f32>                 Threshold for error compared to reference " 
      << "GEMM (default: 0.0)\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "82_blackwell_distributed_gemm" << " --m=16384 --n=106496 --k=16384 \n\n";

    return out;
  }

  /// Compute performance in TFLOP/s
  double tflops(double runtime_s) const {

    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k * l / TP_;
    double tflop = double(flop) / double(1.0e12);
    return tflop / runtime_s;
  }
};

/// Result structure
struct Result {
  double avg_runtime_ms;
  double tflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double tflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), tflops(tflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && \
  (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed,
  bool is_device_tensor = false) {

  double scope_max, scope_min;
  int bits = cutlass::sizeof_bits<Element>::value;

  if (bits == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if (bits <= 16) {
    scope_max = 2;
    scope_min = -2;
  }
  else {
    scope_max = 8;
    scope_min = -8;
  }

  if (is_device_tensor) {
    using Real = typename cutlass::RealType<Element>::Type;
    cutlass::reference::device::TensorFillRandomUniform(
      view, seed, static_cast<Real>(scope_max), static_cast<Real>(scope_min), 0);
    cudaDeviceSynchronize();
  } else {
    cutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min, 0);
  }

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
  auto problem_shape = cute::make_tuple(options.m, options.n, options.k, options.l);

  // Setup (reference) GEMM tensors
  auto shape_A = cute::select<0,2,3>(problem_shape);
  auto shape_B = cute::select<1,2,3>(problem_shape);
  auto shape_C = cute::select<0,1,3>(problem_shape);
  auto shape_D = cute::select<0,1,3>(problem_shape);

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_C);
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_D);

  auto a_coord = cutlass::make_Coord(size(shape_A), 1);
  auto b_coord = cutlass::make_Coord(size(shape_B), 1);
  auto c_coord = cutlass::make_Coord(size(shape_C), 1);

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);

  initialize_tensor(tensor_A.device_view(), seed + 2022, /* is_device_tensor = */ true);
  initialize_tensor(tensor_B.device_view(), seed + 2023, /* is_device_tensor = */ true);
  initialize_tensor(tensor_C.device_view(), seed + 2024, /* is_device_tensor = */ true);

  tensor_A.sync_host();
  tensor_B.sync_host();
  tensor_C.sync_host();
  tensor_D.sync_host();
  tensor_ref_D.sync_host();

  // Set up DistGEMM tensors
  auto local_shape_A = DistSchedule::get_local_a_shape(problem_shape);
  auto local_shape_B = DistSchedule::get_local_b_shape(problem_shape);
  auto local_shape_C = DistSchedule::get_local_c_shape(problem_shape);
  auto local_shape_D = DistSchedule::get_local_d_shape(problem_shape);

  auto a_coord_device = cutlass::make_Coord(size(local_shape_A), 1);
  auto b_coord_device = cutlass::make_Coord(size(local_shape_B), 1);
  auto c_coord_device = cutlass::make_Coord(size(local_shape_C), 1);

  int primary_device_idx;
  CUDA_CHECK(cudaGetDevice(&primary_device_idx));

  // Enable any-to-any access
  for (int device_idx = 0; device_idx < TP_; ++device_idx) {
    int can_access;
    CUDA_CHECK(cudaSetDevice(device_idx));
    for (int peer_idx = 0; peer_idx < TP_; ++peer_idx) {
      if (peer_idx != device_idx) {
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device_idx, peer_idx));
        if (not can_access) {
          std::cerr << "FAILURE: Device " << device_idx << " can't access device " << peer_idx << "." <<
            std::endl;
          exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaDeviceEnablePeerAccess(peer_idx, 0));
      }
    }

    tensor_A_arr[device_idx].resize(a_coord_device);
    tensor_B_arr[device_idx].resize(b_coord_device);
    tensor_C_arr[device_idx].resize(c_coord_device);
    tensor_D_arr[device_idx].resize(c_coord_device);
  }
  CUDA_CHECK(cudaSetDevice(primary_device_idx));
}

/// Commandline options -> Gemm/DistGemm Arguments
using GemmArguments = typename Gemm::Arguments;
GemmArguments gemm_args_from_options(const Options &options) {
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {tensor_A.device_data(), stride_A, tensor_B.device_data(), stride_B},
    {
      {static_cast<ElementCompute>(options.alpha), static_cast<ElementCompute>(options.beta)},
      tensor_C.device_data(), stride_C,
      tensor_ref_D.device_data(), stride_D
    }
  };

  return arguments;
}

using DistGemmArguments = typename DistGemm::Arguments;
DistGemmArguments dist_gemm_args_from_options(
    const Options &options,
    int device_idx,
    cudaStream_t stream) {

  auto problem_shape = cute::make_tuple(options.m, options.n, options.k, options.l);

  auto global_A = cute::make_tensor(tensor_A.device_data(),
      cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
  auto global_B = cute::make_tensor(tensor_B.device_data(),
      cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
  auto global_C = cute::make_tensor(tensor_C.device_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_C));

  auto global_A_device_slice = DistSchedule::get_device_slice_A(global_A, device_idx);
  auto global_B_device_slice = DistSchedule::get_device_slice_B(global_B, device_idx);
  auto global_C_device_slice = DistSchedule::get_device_slice_C(global_C, device_idx);

  auto local_shape_A = DistSchedule::get_local_a_shape(problem_shape);
  auto local_shape_B = DistSchedule::get_local_b_shape(problem_shape);
  auto local_shape_C = DistSchedule::get_local_c_shape(problem_shape);
  auto local_shape_D = DistSchedule::get_local_d_shape(problem_shape);

  auto local_stride_A = cutlass::make_cute_packed_stride(StrideA{}, local_shape_A);
  auto local_stride_B = cutlass::make_cute_packed_stride(StrideB{}, local_shape_B);
  auto local_stride_C = cutlass::make_cute_packed_stride(StrideC{}, local_shape_C);
  auto local_stride_D = cutlass::make_cute_packed_stride(StrideD{}, local_shape_D);

  auto local_A = cute::make_tensor(
      tensor_A_arr[device_idx].device_data(),
      make_layout(local_shape_A, local_stride_A));
  auto local_B = cute::make_tensor(
      tensor_B_arr[device_idx].device_data(),
      make_layout(local_shape_B, local_stride_B));
  auto local_C = cute::make_tensor(
      tensor_C_arr[device_idx].device_data(),
      make_layout(local_shape_C, local_stride_C));
  auto local_D = cute::make_tensor(
      tensor_D_arr[device_idx].device_data(),
      make_layout(local_shape_D, local_stride_D));

  // Copy over tensor tiles for the first iteration
  cutlass::device_copy(global_A_device_slice, local_A, stream);
  cutlass::device_copy(global_B_device_slice, local_B, stream);
  cutlass::device_copy(global_C_device_slice, local_C, stream);

  DistGemmArguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,                                       // mode
    problem_shape,                                                                 // problem shape
    {
      reinterpret_cast<const ElementA*>(local_A.data()),
      local_A.stride(),
      reinterpret_cast<const ElementB*>(local_B.data()),
      local_B.stride()
    },                                                                             // mainloop
    {
      {                                                                            // epilogue.thread
        static_cast<ElementCompute>(options.alpha),
        static_cast<ElementCompute>(options.beta)
      },
      reinterpret_cast<const ElementC*>(local_C.data()),
      local_C.stride(),
      reinterpret_cast<ElementD*>(local_D.data()),
      local_D.stride(),
    },                                                                             // epilogue
    {},                                                                            // hw_info
    {}                                                                             // scheduler
  };

  return arguments;
}

// Gathers results, moves back to the original full-sized D tensor on the primary device.
void gather_results(const Options &options, int device_idx, cudaStream_t stream = nullptr) {

  auto problem_shape = cute::make_tuple(options.m, options.n, options.k, options.l);

  // Global dest
  auto global_D = cute::make_tensor(tensor_D.device_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_D));
  auto global_D_device_slice = DistSchedule::get_device_slice_D(global_D, device_idx);

  // Device_idx local dest
  auto local_shape_D = DistSchedule::get_local_d_shape(problem_shape);
  auto local_stride_D = cutlass::make_cute_packed_stride(StrideD{}, local_shape_D);
  auto local_D = cute::make_tensor(
      tensor_D_arr[device_idx].device_data(),
      make_layout(local_shape_D, local_stride_D)
  );

  // Copy to global dest
  cutlass::device_copy(local_D, global_D_device_slice, stream);
}

bool verify(const Options &options) {
  tensor_D.sync_host();
  tensor_ref_D.sync_host();

  bool passed = false;
  if (options.eps == 0.f) {
    passed = cutlass::reference::host::TensorEquals(tensor_ref_D.host_view(), tensor_D.host_view());
  } else {
    double err = cutlass::reference::host::TensorRelativeErrorMetric(
      tensor_D.host_view(),
      tensor_ref_D.host_view());
    passed = err < 1e-5;
  }

  if (options.m <= 64 && options.n <= 64) {
    std::cout << "GEMM output:\n" << tensor_D.host_view() << "\n\n";
    std::cout << "Reference output:\n" << tensor_ref_D.host_view() << "\n\n";
  }

  return passed;
}

/// Execute a given example GEMM computation
int run(Options &options) {

  int primary_device_idx;
  cudaError_t device_get_result = cudaGetDevice(&primary_device_idx);
  if (device_get_result != cudaSuccess) {
    throw std::runtime_error("cudaGetDevice() failed");
  }

  initialize(options);

  // Reference single-GPU GEMM
  Gemm reference_gemm;
  cutlass::device_memory::allocation<uint8_t> reference_workspace;

  auto reference_arguments = gemm_args_from_options(options);
  size_t reference_workspace_size = Gemm::get_workspace_size(reference_arguments);
  reference_workspace = cutlass::device_memory::allocation<uint8_t>(reference_workspace_size);

  CUTLASS_CHECK(reference_gemm.can_implement(reference_arguments));
  CUTLASS_CHECK(reference_gemm.initialize(reference_arguments, reference_workspace.get()));
  CUTLASS_CHECK(reference_gemm.run());

  using ElementBarrier = typename DistGemm::ElementBarrier;
  using ElementFlag = typename DistGemmKernel::ElementFlag;

  // Set up per-device streams
  cudaStream_t stream_arr[TP_];

  for (int device_idx = 0; device_idx < TP_; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream_arr[device_idx]));
  }

  // Instantiate DistGEMM
  DistGemm dist_gemm_arr[TP_];  // Distributed GEMM array for multiple devices

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace_arr[TP_];
  cutlass::device_memory::allocation<uint8_t> exclusive_workspace_arr[TP_];

  // Cross-device workspace pointer array for gemm.initialize()
  void * workspace_ptr_arr[TP_];
  void * exclusive_workspace_ptr_arr[TP_];

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  DistGemmArguments arguments_[TP_];

  for (int device_idx = 0; device_idx < TP_; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));

    arguments_[device_idx] = dist_gemm_args_from_options(options, device_idx, stream_arr[device_idx]);

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = DistGemm::get_workspace_size(arguments_[device_idx]);
    size_t exclusive_workspace_size = DistGemm::get_exclusive_workspace_size();

    workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(workspace_size);
    exclusive_workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(exclusive_workspace_size);

    // Throw workspace pointers into arrays for gemm.initialize()
    workspace_ptr_arr[device_idx] = workspace_arr[device_idx].get();
    exclusive_workspace_ptr_arr[device_idx] = exclusive_workspace_arr[device_idx].get();

    // Zero out exclusive workspace
    cudaMemsetAsync(exclusive_workspace_ptr_arr[device_idx], 0, exclusive_workspace_size, stream_arr[device_idx]);

    cudaDeviceSynchronize();
  }

  for (int device_idx = 0; device_idx < TP_; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));

    // Check if the problem size is supported or not
    CUTLASS_CHECK(dist_gemm_arr[device_idx].can_implement(arguments_[device_idx]));

#if defined(CUTLASS_ENABLE_GDC_FOR_SM100)
    bool launch_with_pdl = true;
#else
    bool launch_with_pdl = false;
#endif

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(dist_gemm_arr[device_idx].initialize(
          arguments_,
          workspace_ptr_arr,
          exclusive_workspace_ptr_arr,
          device_idx,
          stream_arr[device_idx],
          launch_with_pdl
          ));

    cudaDeviceSynchronize();
  }

  // Correctness / Warmup iteration
  std::cout << std::endl << "  running DistGEMM..." << std::endl;

  for (int device_idx = 0; device_idx < TP_; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
  }
  for (int device_idx = 0; device_idx < TP_; ++device_idx) {
    CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    CUDA_CHECK(cudaGetLastError());
    gather_results(options, device_idx);
  }

  std::cout << "  running DistGEMM finished without runtime errors" << std::endl;

  //// Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;

  result.passed = verify(options);

  std::cout << std::endl << "  Disposition (eps: " << options.eps << "): " << 
    (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0) {
    float elapsed_ms = 0.f;

    // Warmup
    std::cout << "  Warming up for " << options.warmup_iterations << " iterations." << std::endl;
    for (int warmup_iter = 0; warmup_iter < options.warmup_iterations; ++warmup_iter) {
      for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
      }
    }

    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
      CUDA_CHECK(cudaSetDevice(device_idx));
      CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    }

    CUDA_CHECK(cudaSetDevice(primary_device_idx));

    // Benchmark
    std::cout << "  Profiling for " << options.iterations << " iterations." << std::endl;
    using AtomicBoolean = cuda::atomic<bool>;
    AtomicBoolean* atomic_flag_ptr;
    CUDA_CHECK(cudaHostAlloc(&atomic_flag_ptr, sizeof(AtomicBoolean), cudaHostAllocPortable));
    atomic_flag_ptr->store(false);

    cutlass::DistGpuTimer<TP_> timer;

    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
      CUDA_CHECK(cudaSetDevice(device_idx));
      cutlass::delay_kernel<<<1, 1, 0, stream_arr[device_idx]>>>(atomic_flag_ptr);
      CUDA_CHECK(cudaGetLastError());
    }

    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
      timer.start(device_idx, stream_arr[device_idx]);
    }

    atomic_flag_ptr->store(true);

    for (int profile_iter = 0; profile_iter < options.iterations; ++profile_iter) {
      for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
      }
    }

    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
      CUDA_CHECK(cudaSetDevice(device_idx));
      timer.stop(device_idx, stream_arr[device_idx]);
    }

    CUDA_CHECK(cudaSetDevice(primary_device_idx));

    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
      elapsed_ms = max(elapsed_ms, timer.elapsed_millis(device_idx));
    }

    // Compute average runtime and TFLOPs.
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    double avg_runtime_s = (double)(result.avg_runtime_ms / 1000.0);
    result.tflops = options.tflops(avg_runtime_s);

    auto [local_M, local_N, local_K, local_L] = DistSchedule::get_local_gemm_shape(
        cute::make_tuple(options.m, options.n, options.k, options.l));

    std::cout << std::endl;
    std::cout << "  TP: " << TP::value << std::endl;
    std::cout << "  Problem Size: " << 
      options.m << " x " << 
      options.n << " x " << 
      options.k << " x " << 
      options.l << std::endl;
    std::cout << "  Local GEMM Problem Size: " << 
      local_M << " x " << 
      local_N << " x " << 
      local_K << " x " << 
      local_L<< std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  TFLOPS: " << result.tflops << std::endl;
  }

  return 0;
}

#endif // (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) &&
       // (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA Toolkit 12.8 or newer to run Blackwell kernels.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  if (num_devices < TP_) {
    std::cerr << "Distributed GEMM is compiled with TP = " << TP::value << ", but " << 
      "found only " << num_devices << " devices." <<
      std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 10 || props.minor != 0) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Blackwell Architecture "
      << "(compute capability 100), " 
      << "got compute capability " << props.major * 10 + props.minor << "." 
      << std::endl;
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

#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)))
  run(options);
#else
    std::cerr
      << "This example must be compiled with `sm100a` and CUDA Toolkit 12.8 or later." << std::endl;
    return 0;
#endif

  return 0;
}
