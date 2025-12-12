#pragma once

/**
 * @file w4a8_grouped_mm_c3x.cuh
 * @brief Implementation of grouped GEMM operation with int4 and fp8 mixed
 * precision
 *
 * This file implements a grouped GEMM operation that multiplies FP8 matrices
 * (A) with quantized INT4 matrices (B), applying per-block scaling factors.
 * The implementation is optimized for NVIDIA Hopper GPUs, leveraging Tensor
 * Cores for mixed precision arithmetic.
 *
 * Key features:
 * - Supports grouped GEMM operations with multiple experts
 * - Uses FP8 (e4m3) for matrix A
 * - Uses INT4 quantization for matrix B with per-block scaling
 * - Implements preprocessing for INT4 encoding and scale packing
 * - Optimized for Hopper architecture with Tensor Core operations
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp"
#include "w4a8_get_group_starts.cuh"

using namespace cute;

namespace {

// Type definitions
using MmaType = cutlass::float_e4m3_t;     // FP8 e4m3 type
using QuantType = cutlass::int4b_t;        // 4-bit integer type
using ElementAccumulator = float;          // Accumulator type
using ElementScale = cutlass::bfloat16_t;  // Scale type
using ElementC = cutlass::bfloat16_t;      // Output type
using ElementD = ElementC;                 // Output type
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

// Architecture-specific configurations
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
// constexpr int TileShapeK = 512;
// using TileShape = Shape<_128, _32, cute::Int<TileShapeK>>;
// using ClusterShape = Shape<_1, _1, _1>;

// Layout configurations
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;

// Transposed layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

// Alignments
static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

template <typename TileShape, typename ClusterShape, typename KernelSchedule, typename EpilogueSchedule>
struct cutlass_3x_w4a8_group_gemm {
  static constexpr int GroupSize = 128;
  static constexpr int PackedScalesNum = get<2>(TileShape{}) / GroupSize;
  using ElementScalePacked = cutlass::Array<ElementScale, PackedScalesNum>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC_Transpose*,
      AlignmentC,
      ElementD,
      LayoutD_Transpose*,
      AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
      ArchTag,
      OperatorClass,
      cute::tuple<QuantType, ElementScalePacked>,
      LayoutB_Transpose*,
      AlignmentB,
      MmaType,
      LayoutA_Transpose*,
      AlignmentA,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  // Define the final kernel and GEMM operation types
  using GemmKernelScaleOnly =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopScaleOnly, CollectiveEpilogue>;

  using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

  using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
  using StrideC = typename GemmKernelScaleOnly::InternalStrideC;
  using StrideD = typename GemmKernelScaleOnly::InternalStrideD;
  using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;
};

/**
 * @brief Main function to run int4 * fp8 grouped GEMM from PyTorch
 *
 * This function performs multiple GEMM operations in parallel where each
 * operation multiplies an FP8 matrix (A) with a quantized INT4 matrix (B),
 * applying per-channel scaling factors. It's designed for efficient execution
 * on NVIDIA Hopper GPUs, leveraging Tensor Cores for optimal performance with
 * mixed precision arithmetic.
 *
 * The function includes preprocessing steps for both INT4 tensors and scale
 * factors to ensure optimal performance and correct operation.
 *
 * @param d_tensors Output tensor D with shape [total_m, total_n]
 * @param a_tensors Tensor containing all A matrices (fp8_e4m3) with shape
 * [total_m, K]
 * @param b_tensors Tensor containing all B matrices (int4 packed as int8) with
 * shape [E, N, K/2]
 * @param a_scales Tensor containing A matrix scale factors
 * @param b_scales Tensor containing B matrix scale factors with shape [E,
 * K//512, N*4]
 * @param expert_offsets Tensor containing expert offsets for determining group
 * boundaries (int32)
 * @param problem_sizes Tensor containing problem sizes with shape [num_experts,
 * 3] (M, N, K for each group) (int32)
 * @param a_strides Stride information for A tensors
 * @param b_strides Stride information for B tensors
 * @param d_strides Stride information for D tensors
 * @param s_strides Stride information for scale tensors
 * @param chunk_size Size of each chunk for scales (K / number of scale chunks)
 */
// template <typename TileShape, typename ClusterShape, typename KernelSchedule, typename EpilogueSchedule>
template <typename Gemm>
void cutlass_w4a8_group_gemm_caller(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size) {
  //   using Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
  using Args = typename Gemm::GemmScaleOnly::Arguments;

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;

  // Check inputs
  TORCH_CHECK(a_tensors.dim() == 2, "A tensor must be 2D");
  TORCH_CHECK(b_tensors.dim() == 3, "B tensor must be 3D [E, N, K/2]");
  TORCH_CHECK(b_scales.dim() == 3, "Scale tensor must be 3D [E, K//512, N*4]");
  TORCH_CHECK(a_scales.dim() == 1, "A Scale tensor must be 1D [1]");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a 1D tensor");
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");

  // Check tensor shapes
  TORCH_CHECK(problem_sizes.size(0) == num_experts, "problem_sizes must have num_experts rows");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have 3 columns (N, M, K)");
  TORCH_CHECK(b_tensors.size(0) == num_experts, "B tensor first dimension must match number of groups");
  TORCH_CHECK(b_scales.size(0) == num_experts, "Scale tensor first dimension must match number of groups");
  TORCH_CHECK(b_tensors.size(2) * 2 == a_tensors.size(1), "B tensor K/2 dimension must match A tensor K dimension");

  // Check tensor types
  TORCH_CHECK(a_tensors.scalar_type() == torch::kFloat8_e4m3fn, "A tensor must be fp8 (float_e4m3_t) type");
  TORCH_CHECK(b_tensors.scalar_type() == torch::kInt8, "B tensor must contain packed int4 values (stored as int8)");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "Expert offsets must be int32 type");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32, "Problem sizes must be int32 type");

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());
  auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a_tensors.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  Args arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 0;
  fusion_args.beta = 0;
  fusion_args.alpha_ptr = a_scales.data_ptr<float>();
  ;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr());

  run_int4_fp8_get_group_gemm_starts(
      expert_offsets,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      a_tensors,
      b_tensors,
      d_tensors,
      a_scales,
      b_scales);

  arguments = Args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      {static_cast<const QuantType**>(b_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideB*>(b_strides.data_ptr()),
       static_cast<const MmaType**>(a_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideA*>(a_strides.data_ptr()),
       static_cast<const typename Gemm::ElementScalePacked**>(b_scales_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideS*>(s_strides.data_ptr()),
       static_cast<int>(chunk_size)},
      {fusion_args,
       nullptr,
       nullptr,
       static_cast<ElementD**>(out_ptrs.data_ptr()),
       static_cast<typename Gemm::StrideD*>(d_strides.data_ptr())},
      hw_info};

  // Instantiate and run GEMM
  typename Gemm::GemmScaleOnly gemm;
  size_t workspace_size = Gemm::GemmScaleOnly::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a_tensors.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "GEMM implementation not supported");
  }

  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  if (status != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "GEMM initialization failed");
  }

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    TORCH_CHECK(false, "GEMM execution failed");
  }
}

}  // namespace
