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
    \brief Grouped scale Hopper FP8 Grouped GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture
    This example demonstrates a grouped scaled FP8 Grouped GEMM using the new CUTLASS 3.0.
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:
    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.
    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster. This example also showcases on-the-fly modification of TMA
    descriptors to move between groups/problem_count (represented by groups).
    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).
    4. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can
    improve performance.
    5. This example is tuned specifically for the sparse groups case, where the number of active groups (groups
    with non-zero problem count) is much smaller than the total number of groups.
    Examples:
      $ ./examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling_with_sparse_groups  \
        --m=2816 --n=3072 --k=16384 --save_aux=false --save_amax=false \
        --raster=h --swizzle=2 --benchmark=./test_benchmark.txt

      Where the test_benchmark.txt may look as such:
        0 256x512x128
        1 256x512x512
        2 512x256x128
        3 256x256x128
        4 256x512x1024
        5 1024x512x128 and so on
*/

#include <iostream>
#include <optional>
#include <fstream>
#include <sstream>
#include <vector>
#include <cfloat>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"

// Includes from examples directory
#include "helper.h"
#include "hopper_fp8_commandline.hpp"

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = cutlass::float_e4m3_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation

using ArchTag       = cutlass::arch::Sm90;                          // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;               // Operator class tag
using TileShape     = Shape<_128,_128,_128>;                        // Threadblock-level tile size
using ClusterShape  = Shape<_1,_1,_1>;                              // Shape of the threadblocks in a cluster

static constexpr int ScaleGranularityM = 1;
static constexpr int ScaleGranularityN = 128;
static constexpr int ScaleGranularityK = 128;
static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;

using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK, cute::GMMA::Major::MN, cute::GMMA::Major::K>;

using LayoutSFA     = decltype(ScaleConfig::deduce_layoutSFA());    // Layout type for SFA matrix operand
using LayoutSFB     = decltype(ScaleConfig::deduce_layoutSFB());    // Layout type for SFB matrix operand


using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using EpilogueTileType  = cutlass::epilogue::collective::EpilogueTileAuto;
using FusionOperation   = cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag, OperatorClass,
  TileShape, ClusterShape,
  EpilogueTileType,
  ElementAccumulator, ElementCompute,
  ElementC, LayoutC *, AlignmentC,
  ElementD, LayoutD *, AlignmentD,
  EpilogueSchedule,
  FusionOperation
>::CollectiveOp;

using CollectiveMainloopWithGroupWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag, OperatorClass,
  ElementA, cute::tuple<LayoutA *, LayoutSFA *>, AlignmentA,
  ElementB, cute::tuple<LayoutB *, LayoutSFB *>, AlignmentB,
  ElementAccumulator,
  TileShape, ClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
  >,
  KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloopWithGroupWiseScaling,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename Gemm::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;
using ActivationFunctor = typename EpilogueOutputOp::ActivationFn;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

/// Initialization

cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;
std::vector<int64_t> offset_blockscale_A;
std::vector<int64_t> offset_blockscale_B;

std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;
std::vector<StrideD> stride_D_host;
std::vector<LayoutSFA> layout_SFA_host;
std::vector<LayoutSFB> layout_SFB_host;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

uint64_t seed;

cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<ElementD> block_D;
cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_A;
cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_B;

cutlass::DeviceAllocation<const ElementA *> ptr_A;
cutlass::DeviceAllocation<const ElementB *> ptr_B;
cutlass::DeviceAllocation<const ElementC *> ptr_C;
cutlass::DeviceAllocation<ElementD *> ptr_D;
cutlass::DeviceAllocation<ElementD *> ptr_ref_D;
cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_A;
cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_B;

cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<StrideC> stride_C;
cutlass::DeviceAllocation<StrideD> stride_D;
cutlass::DeviceAllocation<LayoutSFA> layout_SFA;
cutlass::DeviceAllocation<LayoutSFB> layout_SFB;

cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
cutlass::DeviceAllocation<ElementAccumulator> block_beta;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED) 

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  double gbps;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    double gbps = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), gbps(gbps), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element, class ScopeMin = std::nullopt_t, class ScopeMax = std::nullopt_t>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023,
  ScopeMin scope_min = std::nullopt, ScopeMax scope_max = std::nullopt) {

  double _scope_max, _scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  if (bits_input == 1) {
    _scope_max = 2;
    _scope_min = 0;
  } else if (bits_input <= 8) {
    _scope_max = 2;
    _scope_min = -2;
  } else if (bits_input == 16) {
    _scope_max = 5;
    _scope_min = -5;
  } else {
    _scope_max = 8;
    _scope_min = -8;
  }
  if constexpr (!std::is_same_v<ScopeMax, std::nullopt_t>) {
    _scope_max = scope_max;
  }
  if constexpr (!std::is_same_v<ScopeMin, std::nullopt_t>) {
    _scope_min = scope_min;
  }
  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, (Element) _scope_max, (Element) _scope_min, 0);

  return true;
}

/// Allocates device-side data
template <typename OptionType>
void allocate(const OptionType &options) {

  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;
  int64_t total_elements_blockscale_A = 0;
  int64_t total_elements_blockscale_B = 0;

  offset_A.clear();
  offset_B.clear();
  offset_C.clear();
  offset_D.clear();
  offset_blockscale_A.clear();
  offset_blockscale_B.clear();
  stride_A_host.clear();
  stride_B_host.clear();
  stride_C_host.clear();
  stride_D_host.clear();
  
  for (int32_t i = 0; i < options.groups; ++i) {

    auto problem = options.problem_sizes_after_alignment_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    auto group_layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto group_layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);
    offset_blockscale_A.push_back(total_elements_blockscale_A);
    offset_blockscale_B.push_back(total_elements_blockscale_B);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;
    int64_t elements_blockscale_A = size(filter_zeros(group_layout_SFA));
    int64_t elements_blockscale_B = size(filter_zeros(group_layout_SFB));

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
    total_elements_blockscale_A += elements_blockscale_A;
    total_elements_blockscale_B += elements_blockscale_B;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
    layout_SFA_host.push_back(group_layout_SFA);
    layout_SFB_host.push_back(group_layout_SFB);

  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
  blockscale_block_A.reset(total_elements_blockscale_A);
  blockscale_block_B.reset(total_elements_blockscale_B);
}

/// Initialize operands to be used in the GEMM and reference GEMM
template <typename OptionType>
void initialize(const OptionType &options) {

  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_after_alignment_host.data());

  std::vector<ElementA *> ptr_A_host(options.groups);
  std::vector<ElementB *> ptr_B_host(options.groups);
  std::vector<ElementC *> ptr_C_host(options.groups);
  std::vector<ElementD *> ptr_D_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);
  std::vector<ElementBlockScale *> ptr_blockscale_A_host(options.groups);
  std::vector<ElementBlockScale *> ptr_blockscale_B_host(options.groups);

  alpha_host.clear();
  beta_host.clear();

  for (int i = 0; i < options.groups; i++) {
    // If the current group's matrix has size 0, set the pointer to nullptr
    if (i < options.groups - 1 && offset_A.at(i) == offset_A.at(i + 1)) {
      ptr_A_host.at(i) = nullptr;
    } else {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    }
    if (i < options.groups - 1 && offset_B.at(i) == offset_B.at(i + 1)) {
      ptr_B_host.at(i) = nullptr;
    } else {
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    }
    if (i < options.groups - 1 && offset_C.at(i) == offset_C.at(i + 1)) {
      ptr_C_host.at(i) = nullptr;
    } else {
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    }
    if (i < options.groups - 1 && offset_D.at(i) == offset_D.at(i + 1)) {
      ptr_D_host.at(i) = nullptr;
    } else {
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    }
    if (i < options.groups - 1 && offset_blockscale_A.at(i) == offset_blockscale_A.at(i + 1)) {
      ptr_blockscale_A_host.at(i) = nullptr;
    } else {
      ptr_blockscale_A_host.at(i) = blockscale_block_A.get() + offset_blockscale_A.at(i);
    }
    if (i < options.groups - 1 && offset_blockscale_B.at(i) == offset_blockscale_B.at(i + 1)) {
      ptr_blockscale_B_host.at(i) = nullptr;
    } else {
      ptr_blockscale_B_host.at(i) = blockscale_block_B.get() + offset_blockscale_B.at(i);
    }
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  ptr_A.reset(options.groups);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  ptr_blockscale_A.reset(options.groups);
  ptr_blockscale_A.copy_from_host(ptr_blockscale_A_host.data());

  ptr_blockscale_B.reset(options.groups);
  ptr_blockscale_B.copy_from_host(ptr_blockscale_B_host.data());

  stride_A.reset(options.groups);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(options.groups);
  stride_B.copy_from_host(stride_B_host.data());

  stride_C.reset(options.groups);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(options.groups);
  stride_D.copy_from_host(stride_D_host.data());

  layout_SFA.reset(options.groups);
  layout_SFA.copy_from_host(layout_SFA_host.data());

  layout_SFB.reset(options.groups);
  layout_SFB.copy_from_host(layout_SFB_host.data());

  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  initialize_block(block_A, seed + 2022);
  initialize_block(block_B, seed + 2023);
  initialize_block(block_C, seed + 2024);
  initialize_block(blockscale_block_A, seed + 2025, -1, 1);
  initialize_block(blockscale_block_B, seed + 2026, -1, 1);

  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());

}

/// Populates a Gemm::Arguments structure from the given commandline options
template<typename GemmArguments, typename OptionType>
GemmArguments args_from_options(const OptionType &options, bool host_problem_shapes_available = true)
{
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  int device_id = 0;
  cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(device_id);

  GemmArguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {options.groups, problem_sizes.get(), host_problem_shapes_available ? options.problem_sizes_after_alignment_host.data() : (decltype(options.problem_sizes_after_alignment_host.data())) nullptr},
    {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(),
     ptr_blockscale_A.get(), layout_SFA.get(),
     ptr_blockscale_B.get(), layout_SFB.get()
    },
    {
      {}, // epilogue.thread
      ptr_C.get(), stride_C.get(),
      ptr_D.get(), stride_D.get()
    },
    kernel_hw_info
  };

  auto &fusion_args = arguments.epilogue.thread;
  if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
    // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
    fusion_args.alpha = options.alpha;
    fusion_args.beta = options.beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // Single alpha and beta for all groups
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
  }
  else {
    // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
    fusion_args.alpha = 0;
    fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = alpha_device.get();
    fusion_args.beta_ptr_array = beta_device.get();
    // One alpha and beta per each group
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
  }

  arguments.scheduler.raster_order = options.raster_order;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

template <typename OptionType>
bool verify(const OptionType &options) {

  //
  // Compute reference output
  //

  std::vector<ElementA> block_A_host(block_A.size());
  std::vector<ElementB> block_B_host(block_B.size());
  std::vector<ElementC> block_C_host(block_C.size());
  std::vector<ElementD> block_D_host_kernel(block_D.size());
  std::vector<ElementD> block_D_host_ref(block_D.size());
  std::vector<ElementBlockScale> blockscale_block_A_host(blockscale_block_A.size());
  std::vector<ElementBlockScale> blockscale_block_B_host(blockscale_block_B.size());

  block_A.copy_to_host(block_A_host.data());
  block_B.copy_to_host(block_B_host.data());
  block_C.copy_to_host(block_C_host.data());
  block_D.copy_to_host(block_D_host_kernel.data());
  blockscale_block_A.copy_to_host(blockscale_block_A_host.data());
  blockscale_block_B.copy_to_host(blockscale_block_B_host.data());

  bool passed = true;
  std::cout << "  Running host reference kernel - may run for a while for large problems." << std::endl;
  for (int group_idx = 0; group_idx < options.groups; group_idx++) {
    // Group scaling tensors shapes based `ScaleGranularityM`, CTA Block (TileShape) and GEMM Problem shape
    auto [m, n, k] = options.problem_sizes_after_alignment_host.at(group_idx);

    // Create instantiation for device reference gemm kernel
    auto A = cute::make_tensor(block_A_host.data() + offset_A.at(group_idx),
                              cute::make_layout(
                                  cute::make_shape(m, k, 1),
                                  stride_A_host.at(group_idx)
                                )
                              );
    auto B = cute::make_tensor(block_B_host.data() + offset_B.at(group_idx),
                              cute::make_layout(
                                cute::make_shape(n, k, 1),
                                stride_B_host.at(group_idx)
                                )
                              );
    auto C = cute::make_tensor(block_C_host.data() + offset_C.at(group_idx),
                              cute::make_layout(
                                  cute::make_shape(m, n, 1),
                                  stride_C_host.at(group_idx)
                                )
                              );
    auto D = cute::make_tensor(block_D_host_ref.data() + offset_D.at(group_idx),
                              cute::make_layout(
                                  cute::make_shape(m, n, 1),
                                  stride_D_host.at(group_idx)
                                )
                              );

    auto SFA = cute::make_tensor(blockscale_block_A_host.data() + offset_blockscale_A.at(group_idx),
                                 layout_SFA_host.at(group_idx));
    auto SFB = cute::make_tensor(blockscale_block_B_host.data() + offset_blockscale_B.at(group_idx),
                                 layout_SFB_host.at(group_idx));

    using unused_t = decltype(D);

    cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,
      decltype(A), 
      decltype(SFA), 
      decltype(B),
      decltype(SFB)
    > mainloop_params{A, SFA, B, SFB};

    cutlass::reference::host::GettEpilogueParams<
        ElementScalar,
        ElementScalar,
        ElementAccumulator,
        ElementCompute,
        decltype(C),
        decltype(D)
    > epilogue_params;

    epilogue_params.C = C;
    epilogue_params.D = D;
    epilogue_params.alpha = alpha_host.at(group_idx);
    epilogue_params.beta = beta_host.at(group_idx);

    // get reference result
    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    auto this_group_passed = std::equal(
      // std::execution::par_unseq,
      block_D_host_ref.data() + offset_D.at(group_idx),
      block_D_host_ref.data() + offset_D.at(group_idx) + m * n,
      block_D_host_kernel.data() + offset_D.at(group_idx)
    );
    
    passed &= this_group_passed;

#if 0
    std::cout << "Group: " << group_idx << " M: " << m << " N: " << n << " K: " << k << " Status: " << this_group_passed << std::endl;
#endif

  }

  return passed;
}

/// Execute a given example GEMM computation
template <typename OptionType>
int run(OptionType &options, bool host_problem_shapes_available = true)
{
  allocate(options);
  initialize(options);

  std::cout << "  Problem Sizes, Alpha, Beta " << std::endl;
  for (int32_t i = 0; i < options.groups; ++i) {
    std::cout << "    " << options.problem_sizes_host.at(i);
    std::cout << ", " << alpha_host.at(i) << ", " << beta_host.at(i) << std::endl;
  }
  std::cout << "  Groups      : " << options.groups  << std::endl;
  std::cout << "  Tile shape (M, N, K): " << size<0>(TileShape{}) << ", " << size<1>(TileShape{}) << ", " << size<2>(TileShape{}) << std::endl;
  std::cout << "  ScaleGranularityM: " << ScaleGranularityM << " (ScaleMsPerTile: " << ScaleMsPerTile << ")" << std::endl;
  std::cout << "  ScaleGranularityN: " << ScaleGranularityN << " (ScaleNsPerTile: " << ScaleNsPerTile << ")" << std::endl;
  std::string raster = "Heuristic";
  if (options.raster_order == RasterOrderOptions::AlongN) {
    raster = "Along N";
  }
  else if (options.raster_order == RasterOrderOptions::AlongM) {
    raster = "Along M";
  }
  std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<typename Gemm::Arguments>(options, host_problem_shapes_available);

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

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
   exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0) {
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
    result.gbps = options.template gbps<ElementA, 
                                        ElementB, 
                                        ElementC, 
                                        ElementD, 
                                        ElementBlockScale, 
                                        TileShape, 
                                        ScaleMsPerTile, 
                                        ScaleNsPerTile>(result.avg_runtime_ms / 1000.0);

    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
    std::cout << "  GBPS:   " << result.gbps << std::endl;
    fflush(stdout);
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 3)) {
    std::cerr << "This example requires CUDA 12.3 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

  //
  // Parse options
  //

  Options<ProblemShape> options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

  std::cout << "Running tests with host problem shapes:" << std::endl;
  run(options, true);
  std::cout << "Running tests without host problem shapes:" << std::endl;
  run(options, false);

#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
