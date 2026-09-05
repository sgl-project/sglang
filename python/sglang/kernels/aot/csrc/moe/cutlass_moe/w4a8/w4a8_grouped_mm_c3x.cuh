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

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass_extensions/epilogue/collective/default_epilogue_array_per_token_scale.hpp"
#include "cutlass_extensions/epilogue/collective/default_epilogue_array_swiglu_quant.hpp"
#include "cutlass_extensions/epilogue/collective/sm90_epilogue_array_tma_warpspecialized_mixed_input.hpp"
#include "cutlass_extensions/epilogue/fusion/sm90_ptr_array_per_token_scale_callbacks_tma_warpspecialized.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_single_warpgroup_persistent.hpp"
#include "w4a8_get_group_starts.cuh"
#include "w4a8_swg_precomputed_work_map.cuh"

using namespace cute;

namespace sgl_kernel::w4a8_detail {

// Internal selector tag for config333's register-to-global GEMM2 epilogue.
// The selected collective itself advertises CUTLASS's no-SMEM policy.
struct WarpShuffleGemm2Epilogue {};
struct WarpShufflePackedStoreGemm2Epilogue {};
struct WarpShufflePackedStoreMaxRegsGemm2Epilogue {};

// Type definitions
using MmaType = cutlass::float_e4m3_t;            // FP8 e4m3 type
using QuantType = cutlass::int4b_t;               // 4-bit integer type (default, int4a8)
using ElementAccumulator = float;                 // Accumulator type
using DefaultElementScale = cutlass::bfloat16_t;  // Scale type
using ElementC = cutlass::bfloat16_t;             // Output type
using ElementD = ElementC;                        // Output type
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

template <
    bool UseSingleWarpgroup,
    bool UsePreMmaE8M0,
    bool FuseSwiGLUQuant,
    typename TileShape,
    typename ClusterShape,
    typename EpilogueSchedule>
struct W4A8EpilogueSelector;

template <typename TileShape, typename ClusterShape, typename EpilogueSchedule>
struct W4A8EpilogueSelector<false, false, false, TileShape, ClusterShape, EpilogueSchedule> {
  using Type = typename cutlass::epilogue::collective::CollectiveBuilder<
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
};

template <typename TileShape, typename ClusterShape, typename EpilogueSchedule>
struct W4A8EpilogueSelector<false, true, false, TileShape, ClusterShape, EpilogueSchedule> {
  using FusionOperation =
      cutlass::epilogue::fusion::PtrArrayPerTokenScaledAcc<ElementD, ElementAccumulator, ElementAccumulator>;
  using Type = typename tensorrt_llm::cutlass_extensions::epilogue::collective::MixedInputSm90TmaEpilogueBuilder<
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
      EpilogueSchedule,
      FusionOperation>::CollectiveOp;
};

template <typename TileShape, typename ClusterShape>
struct W4A8EpilogueSelector<false, true, false, TileShape, ClusterShape, WarpShuffleGemm2Epilogue> {
  using Epilogue = cutlass::epilogue::collective::WarpShuffleEpilogueArrayPerTokenScale<
      TileShape,
      ElementC,
      cutlass::detail::TagToStrideC_t<LayoutC_Transpose*>,
      ElementD,
      cutlass::detail::TagToStrideC_t<LayoutD_Transpose*>,
      ElementAccumulator,
      ElementAccumulator>;
  using Type = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<Epilogue>;
};

template <typename TileShape, typename ClusterShape>
struct W4A8EpilogueSelector<false, true, false, TileShape, ClusterShape, WarpShufflePackedStoreMaxRegsGemm2Epilogue> {
  using Epilogue = cutlass::epilogue::collective::WarpShuffleEpilogueArrayPerTokenScale<
      TileShape,
      ElementC,
      cutlass::detail::TagToStrideC_t<LayoutC_Transpose*>,
      ElementD,
      cutlass::detail::TagToStrideC_t<LayoutD_Transpose*>,
      ElementAccumulator,
      ElementAccumulator,
      true,
      true>;
  using Base = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<Epilogue>;
  struct Type : Base {
    using Base::Base;
    static constexpr bool PreferMaxMmaRegisters = true;
  };
};

template <typename TileShape, typename ClusterShape>
struct W4A8EpilogueSelector<false, true, false, TileShape, ClusterShape, WarpShufflePackedStoreGemm2Epilogue> {
  using Epilogue = cutlass::epilogue::collective::WarpShuffleEpilogueArrayPerTokenScale<
      TileShape,
      ElementC,
      cutlass::detail::TagToStrideC_t<LayoutC_Transpose*>,
      ElementD,
      cutlass::detail::TagToStrideC_t<LayoutD_Transpose*>,
      ElementAccumulator,
      ElementAccumulator,
      true>;
  using Type = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<Epilogue>;
};

template <typename TileShape, typename ClusterShape, typename EpilogueSchedule>
struct W4A8EpilogueSelector<true, true, false, TileShape, ClusterShape, EpilogueSchedule> {
  using Epilogue = cutlass::epilogue::collective::SmemEpilogueArrayPerTokenScale<
      TileShape,
      ElementC,
      cutlass::detail::TagToStrideC_t<LayoutC_Transpose*>,
      ElementD,
      cutlass::detail::TagToStrideC_t<LayoutD_Transpose*>,
      ElementAccumulator,
      ElementAccumulator>;
  using Type = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<Epilogue>;
};

template <typename TileShape, typename ClusterShape, typename EpilogueSchedule>
struct W4A8EpilogueSelector<true, true, true, TileShape, ClusterShape, EpilogueSchedule> {
  using Epilogue = cutlass::epilogue::collective::SmemEpilogueArraySwiGLUQuant<
      TileShape,
      ElementC,
      cutlass::detail::TagToStrideC_t<LayoutC_Transpose*>,
      ElementD,
      cutlass::detail::TagToStrideC_t<LayoutD_Transpose*>,
      ElementAccumulator,
      ElementAccumulator>;
  using Type = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<Epilogue>;
};

template <typename TileShape, typename ClusterShape, typename EpilogueSchedule>
struct W4A8EpilogueSelector<false, true, true, TileShape, ClusterShape, EpilogueSchedule> {
  using Epilogue = cutlass::epilogue::collective::SmemEpilogueArraySwiGLUQuant<
      TileShape,
      ElementC,
      cutlass::detail::TagToStrideC_t<LayoutC_Transpose*>,
      ElementD,
      cutlass::detail::TagToStrideC_t<LayoutD_Transpose*>,
      ElementAccumulator,
      ElementAccumulator>;
  using Type = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<Epilogue>;
};

template <
    typename TileShape,
    typename ClusterShape,
    typename KernelSchedule,
    typename EpilogueSchedule,
    // MXFP4A8: the 4-bit weight element type. Defaults to int4b_t so all existing
    // int4a8 instantiations are byte-identical; pass cutlass::float_e2m1_t for mxfp4a8.
    typename QuantTypeB = QuantType,
    // K-wise quant group size. int4a8 uses 128; mxfp4a8 (E8M0 block) uses 32.
    int GroupSizeK = 128,
    bool UseSingleWarpgroup = false,
    bool UsePreMmaE8M0 = false,
    bool ChunkMajorWorkMap = true,
    int ExpertRowPolicyValue = 0,
    bool FuseSwiGLUQuant = false>
struct cutlass_3x_w4a8_group_gemm {
  static constexpr bool UseSingleWarpgroupKernel = UseSingleWarpgroup;
  static constexpr bool UsePreMmaE8M0Scale = UsePreMmaE8M0;
  static constexpr bool UseChunkMajorWorkMap = ChunkMajorWorkMap;
  static constexpr bool UseWarpShuffleGemm2Epilogue =
      std::is_same_v<EpilogueSchedule, WarpShuffleGemm2Epilogue> ||
      std::is_same_v<EpilogueSchedule, WarpShufflePackedStoreGemm2Epilogue> ||
      std::is_same_v<EpilogueSchedule, WarpShufflePackedStoreMaxRegsGemm2Epilogue>;
  static constexpr auto ExpertRows = static_cast<sgl_kernel::swg_detail::ExpertRowPolicy>(ExpertRowPolicyValue);
  static constexpr bool FuseSwiGLUQuantEpilogue = FuseSwiGLUQuant;
  static constexpr int GroupSize = GroupSizeK;
  static constexpr int PackedScalesNum = get<2>(TileShape{}) / GroupSize;
  using ElementScale = std::conditional_t<UsePreMmaE8M0Scale, cutlass::float_ue8m0_t, DefaultElementScale>;
  using ElementScalePacked =
      std::conditional_t<UsePreMmaE8M0Scale, ElementScale, cutlass::Array<ElementScale, PackedScalesNum>>;
  // Alignment for the 4-bit weight operand (int4b_t / float_e2m1_t are both 4-bit).
  static constexpr int AlignmentQuantB = 128 / cutlass::sizeof_bits<QuantTypeB>::value;
  static_assert(!UseSingleWarpgroup || std::is_same_v<QuantTypeB, cutlass::float_e2m1_t>);
  static_assert(!UseSingleWarpgroup || GroupSize == 32);
  static_assert(!UseSingleWarpgroup || UsePreMmaE8M0Scale);
  static_assert(!FuseSwiGLUQuant || UsePreMmaE8M0);
  static_assert(!UseWarpShuffleGemm2Epilogue || UsePreMmaE8M0Scale);
  static_assert(!UseWarpShuffleGemm2Epilogue || !UseSingleWarpgroup);
  static_assert(!UseWarpShuffleGemm2Epilogue || !FuseSwiGLUQuant);
  static_assert(!UseWarpShuffleGemm2Epilogue || !UseChunkMajorWorkMap);
  static_assert(
      !UseWarpShuffleGemm2Epilogue ||
      std::is_same_v<KernelSchedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>);
  static_assert(
      !UseWarpShuffleGemm2Epilogue ||
      (cute::size<0>(TileShape{}) == 128 && cute::size<1>(TileShape{}) == 32 && cute::size<2>(TileShape{}) == 512));
  static_assert(!UseWarpShuffleGemm2Epilogue || cute::size(ClusterShape{}) == 1);
  static_assert(!UsePreMmaE8M0Scale || std::is_same_v<QuantTypeB, cutlass::float_e2m1_t>);
  static_assert(!UsePreMmaE8M0Scale || GroupSize == 32);
  static_assert(
      !UsePreMmaE8M0Scale || std::is_same_v<KernelSchedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>);
  static_assert(!UseSingleWarpgroup || cute::size(ClusterShape{}) == 1);

  using CollectiveEpilogue = typename W4A8EpilogueSelector<
      UseSingleWarpgroupKernel,
      UsePreMmaE8M0Scale,
      FuseSwiGLUQuantEpilogue,
      TileShape,
      ClusterShape,
      EpilogueSchedule>::Type;

  using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
      ArchTag,
      OperatorClass,
      cute::tuple<QuantTypeB, ElementScalePacked>,
      LayoutB_Transpose*,
      AlignmentQuantB,
      MmaType,
      LayoutA_Transpose*,
      AlignmentA,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      std::conditional_t<
          UseSingleWarpgroupKernel,
          cutlass::gemm::collective::StageCount<3>,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>>,
      KernelSchedule,
      UsePreMmaE8M0Scale ? cutlass::gemm::collective::MixedInputScaleMode::kPreMmaE8M0
                         : cutlass::gemm::collective::MixedInputScaleMode::kPostMma>::CollectiveOp;

  // Expose the weight quant type so the caller can cast device pointers correctly.
  using ElementQuantB = QuantTypeB;

  // Define the final kernel and GEMM operation types
  static constexpr int SingleWarpgroupTileM = cute::size<0>(TileShape{});
  static constexpr int SingleWarpgroupTileN = cute::size<1>(TileShape{});
  static_assert(
      !UseSingleWarpgroup || (SingleWarpgroupTileN == 8 || SingleWarpgroupTileN == 16 || SingleWarpgroupTileN == 32 ||
                              SingleWarpgroupTileN == 40));
  static constexpr int SingleWarpgroupCtasPerSm =
      SingleWarpgroupTileN == 8 ? 6 : (SingleWarpgroupTileN == 16 ? 5 : (SingleWarpgroupTileN == 32 ? 4 : 3));

  using PrecomputedTileScheduler =
      cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90GroupPrecomputed<ProblemShape, 8, UseChunkMajorWorkMap>;
  using GemmKernelScaleOnly = std::conditional_t<
      UseSingleWarpgroupKernel,
      cutlass::gemm::kernel::SingleWarpgroupPersistentGemm<
          ProblemShape,
          CollectiveMainloopScaleOnly,
          CollectiveEpilogue,
          SingleWarpgroupCtasPerSm,
          3,
          cutlass::gemm::kernel::SingleWarpgroupPipelineMode::RollingRefill>,
      std::conditional_t<
          UsePreMmaE8M0Scale,
          cutlass::gemm::kernel::GemmUniversalPrecomputedScheduler<
              ProblemShape,
              CollectiveMainloopScaleOnly,
              CollectiveEpilogue,
              PrecomputedTileScheduler>,
          cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopScaleOnly, CollectiveEpilogue>>>;

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
    int64_t chunk_size,
    // MXFP4A8: optional per-token+per-block activation scale (N-indexed, bf16
    // [total_m, K/act_group]) and its per-expert stride array. When provided the
    // mainloop's EnableActBlockScale path is fed; for int4a8 these stay nullopt
    // so the aggregate leaves ptr_AS/dAS default-null and the kernel is byte-identical.
    std::optional<torch::Tensor> act_block_scales = std::nullopt,
    std::optional<torch::Tensor> as_strides = std::nullopt,
    int64_t act_scale_group = 0,
    std::optional<torch::Tensor> expert_ids = std::nullopt,
    std::optional<torch::Tensor> fused_output_q = std::nullopt,
    std::optional<torch::Tensor> fused_output_s = std::nullopt,
    std::optional<torch::Tensor> fused_row_amax = std::nullopt,
    std::optional<torch::Tensor> fused_row_arrivals = std::nullopt,
    std::optional<torch::Tensor> fused_expert_residual = std::nullopt,
    double swiglu_limit = 0.0,
    bool has_swiglu_limit = false) {
  //   using Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
  using Args = typename Gemm::GemmScaleOnly::Arguments;

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != b_tensors.size(0);

  // Check inputs
  TORCH_CHECK(a_tensors.dim() == 2 or a_tensors.dim() == 3, "A tensor must be 2D/3D");
  TORCH_CHECK(b_tensors.dim() == 3, "B tensor must be 3D [E, N, K/2]");
  if constexpr (Gemm::UsePreMmaE8M0Scale) {
    TORCH_CHECK(b_scales.is_contiguous(), "prescale weight scales must be folded and contiguous");
    TORCH_CHECK(
        b_scales.numel() == b_tensors.size(0) * b_tensors.size(1) * b_tensors.size(2) * 2 / Gemm::GroupSize,
        "prescale weight scales must contain E*N*K/32 raw E8M0 elements");
  } else {
    TORCH_CHECK(b_scales.dim() == 3, "Scale tensor must be 3D [E, K//512, N*4]");
  }
  TORCH_CHECK(a_scales.dim() == 1, "A Scale tensor must be 1D [1]");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a 1D tensor");
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");

  // Check tensor shapes
  TORCH_CHECK(problem_sizes.size(0) == num_experts, "problem_sizes must have num_experts rows");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have 3 columns (N, M, K)");
  if (expert_ids.has_value()) {
    TORCH_CHECK(expert_ids->dim() == 1, "expert_ids must be a 1D tensor");
    TORCH_CHECK(expert_ids->size(0) == num_experts, "expert_ids must match problem_sizes rows");
    TORCH_CHECK(expert_ids->scalar_type() == torch::kInt32, "expert_ids must be int32");
    TORCH_CHECK(expert_ids->is_contiguous(), "expert_ids must be contiguous");
    TORCH_CHECK(expert_ids->device() == b_tensors.device(), "expert_ids must be on the weight device");
  } else {
    TORCH_CHECK(b_tensors.size(0) == num_experts, "B tensor first dimension must match number of groups");
    if constexpr (!Gemm::UsePreMmaE8M0Scale) {
      TORCH_CHECK(b_scales.size(0) == num_experts, "Scale tensor first dimension must match number of groups");
    }
  }
  TORCH_CHECK(
      b_tensors.size(2) * 2 == a_tensors.size(1) or b_tensors.size(2) * 2 == a_tensors.size(2),
      "B tensor K/2 dimension must match A tensor K dimension");

  // Check tensor types
  TORCH_CHECK(a_tensors.scalar_type() == torch::kFloat8_e4m3fn, "A tensor must be fp8 (float_e4m3_t) type");
  TORCH_CHECK(b_tensors.scalar_type() == torch::kInt8, "B tensor must contain packed int4 values (stored as int8)");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "Expert offsets must be int32 type");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32, "Problem sizes must be int32 type");
  if constexpr (Gemm::UseSingleWarpgroupKernel || Gemm::FuseSwiGLUQuantEpilogue || Gemm::UseWarpShuffleGemm2Epilogue) {
    TORCH_CHECK(
        d_tensors.dim() == 2 && d_tensors.is_contiguous(), "Specialized GEMM epilogue requires contiguous 2D D");
  }
  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());
  auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
  // MXFP4A8: per-expert activation block-scale pointer array (only used when
  // act_block_scales is provided; int4a8 leaves this empty).
  torch::Tensor as_scales_ptrs;
  bool use_act_block_scale = !Gemm::UsePreMmaE8M0Scale && act_block_scales.has_value() && as_strides.has_value();
  if (use_act_block_scale) {
    as_scales_ptrs = torch::empty(num_experts, options_int);
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a_tensors.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  if constexpr (Gemm::UseSingleWarpgroupKernel) {
    hw_info.sm_count *= Gemm::SingleWarpgroupCtasPerSm;
  }
  Args arguments;
  sgl_kernel::swg_detail::SwgPrecomputedWorkMap swg_work_map;

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr());

  if constexpr (!Gemm::UseSingleWarpgroupKernel) {
    run_int4_fp8_get_group_gemm_starts<typename Gemm::ElementScale>(
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
        b_scales,
        use_act_block_scale ? std::optional<torch::Tensor>(as_scales_ptrs) : std::nullopt,
        use_act_block_scale ? act_block_scales : std::nullopt,
        use_act_block_scale ? act_scale_group : 0,
        // Weight-scale group size for this instantiation (int4a8=128, mxfp4a8=32),
        // so the per-expert weight-scale pointer advances by n*k/GroupSize.
        static_cast<int64_t>(Gemm::GroupSize),
        // MXFP4A8: per-expert act-scale stride tensor [E,2] so get_group_starts can
        // advance the act-scale pointer by the PADDED (16B-aligned) cumsum.
        use_act_block_scale ? as_strides : std::nullopt,
        expert_ids);
  }

  decltype(arguments.epilogue.thread) fusion_args;
  if constexpr (Gemm::UsePreMmaE8M0Scale) {
    fusion_args.token_scale_default = ElementAccumulator(1);
    fusion_args.token_scale_ptr_array =
        per_act_token ? static_cast<float const* const*>(a_scales_ptrs.data_ptr()) : nullptr;
    if constexpr (Gemm::FuseSwiGLUQuantEpilogue) {
      TORCH_CHECK(fused_output_q.has_value(), "fused GEMM1 requires output_q");
      TORCH_CHECK(fused_output_s.has_value(), "fused GEMM1 requires output_s");
      TORCH_CHECK(fused_row_amax.has_value(), "fused GEMM1 requires row_amax");
      TORCH_CHECK(fused_row_arrivals.has_value(), "fused GEMM1 requires row_arrivals");
      TORCH_CHECK(fused_expert_residual.has_value(), "fused GEMM1 requires expert residual");
      TORCH_CHECK(expert_ids == std::nullopt, "fused GEMM1 does not support compact experts");
      TORCH_CHECK(d_tensors.dim() == 2, "fused GEMM1 staging must be 2D");
      TORCH_CHECK(
          b_tensors.size(1) == d_tensors.size(1) * 2,
          "fused GEMM1 staging width must be half the physical output width");
      fusion_args.output_fp8 = static_cast<cutlass::float_e4m3_t*>(fused_output_q->data_ptr());
      fusion_args.output_scale = static_cast<float*>(fused_output_s->data_ptr());
      fusion_args.row_amax = static_cast<float*>(fused_row_amax->data_ptr());
      fusion_args.row_arrivals = static_cast<int32_t*>(fused_row_arrivals->data_ptr());
      fusion_args.expert_residual = static_cast<float const*>(fused_expert_residual->data_ptr());
      fusion_args.total_rows = d_tensors.size(0);
      fusion_args.logical_channel_extent = d_tensors.size(1);
      fusion_args.swiglu_limit = static_cast<float>(swiglu_limit);
      fusion_args.has_swiglu_limit = has_swiglu_limit;
    }
  } else {
    fusion_args.alpha = 0;
    fusion_args.beta = 0;
    fusion_args.alpha_ptr = a_scales.data_ptr<float>();
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
  }
  if constexpr (Gemm::UseSingleWarpgroupKernel || Gemm::FuseSwiGLUQuantEpilogue || Gemm::UseWarpShuffleGemm2Epilogue) {
    arguments = Args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts, problem_sizes_as_shapes, nullptr},
        {static_cast<const typename Gemm::ElementQuantB**>(b_ptrs.data_ptr()),
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
         static_cast<typename Gemm::StrideD*>(d_strides.data_ptr()),
         static_cast<ElementD*>(d_tensors.data_ptr()),
         d_tensors.size(1),
         d_tensors.size(1),
         ElementAccumulator(0)},
        hw_info};
  } else {
    arguments = Args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts, problem_sizes_as_shapes, nullptr},
        {static_cast<const typename Gemm::ElementQuantB**>(b_ptrs.data_ptr()),
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
  }
  if constexpr (Gemm::UsePreMmaE8M0Scale) {
    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
    arguments.scheduler.max_swizzle_size = sgl_kernel::swg_detail::kSwgSchedulerMaxSwizzle;
    arguments.scheduler.raster_order = RasterOrderOptions::AlongM;
    auto const swg_grid_shape = Gemm::GemmScaleOnly::get_grid_shape(arguments);
    swg_work_map = sgl_kernel::swg_detail::build_swg_precomputed_work_map<Gemm>(
        problem_sizes_as_shapes,
        num_experts,
        static_cast<uint64_t>(d_tensors.size(0)),
        static_cast<uint64_t>(b_tensors.size(1)),
        swg_grid_shape,
        false,
        a_tensors.device());
    arguments.scheduler.precomputed_work_tiles = static_cast<uint64_t const*>(swg_work_map.storage.data_ptr());
    if constexpr (Gemm::UseChunkMajorWorkMap) {
      arguments.scheduler.precomputed_work_tiles_per_worker = swg_work_map.tiles_per_worker;
    }
    arguments.mainloop.ptr_A_prebuilt_tma_desc =
        static_cast<cute::TmaDescriptor const*>(swg_work_map.prebuilt_tma_desc_a.data_ptr());
    arguments.mainloop.ptr_B_prebuilt_tma_descs =
        static_cast<cute::TmaDescriptor const*>(swg_work_map.prebuilt_tma_desc_b.data_ptr());
  }

  // MXFP4A8: feed the activation block-scale into the mainloop's optional path.
  // These members default to nullptr, so the int4a8 path is unaffected.
  if constexpr (!Gemm::UsePreMmaE8M0Scale) {
    if (use_act_block_scale) {
      arguments.mainloop.ptr_AS = static_cast<const typename Gemm::ElementScalePacked**>(as_scales_ptrs.data_ptr());
      arguments.mainloop.dAS = static_cast<typename Gemm::StrideS*>(as_strides->data_ptr());
    }
  }

  // Instantiate and run GEMM
  typename Gemm::GemmScaleOnly gemm;
  size_t workspace_size = gemm.get_workspace_size(arguments);
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

  if constexpr (Gemm::UsePreMmaE8M0Scale) {
    sgl_kernel::swg_detail::launch_swg_precomputed_work_map<Gemm>(
        swg_work_map,
        problem_sizes_as_shapes,
        num_experts,
        gemm.params().mainloop,
        expert_offsets,
        expert_ids,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a_tensors,
        b_tensors,
        d_tensors,
        a_scales,
        b_scales,
        false,
        stream);
  }

  status = gemm.run(stream, nullptr, true);
  if (status != cutlass::Status::kSuccess) {
    cudaError_t ce = cudaGetLastError();
    if constexpr (Gemm::UsePreMmaE8M0Scale) {
      auto const grid = Gemm::GemmScaleOnly::get_grid_shape(arguments);
      auto const block = Gemm::GemmKernelScaleOnly::get_block_shape();
      int const max_active_blocks = Gemm::GemmScaleOnly::maximum_active_blocks();
      TORCH_CHECK(
          false,
          "SWG GEMM execution failed: status=",
          cutlassGetStatusString(status),
          " cuda=",
          cudaGetErrorString(ce),
          " grid=(",
          grid.x,
          ",",
          grid.y,
          ",",
          grid.z,
          ") block=(",
          block.x,
          ",",
          block.y,
          ",",
          block.z,
          ") smem=",
          Gemm::GemmKernelScaleOnly::SharedStorageSize,
          " max_active_blocks=",
          max_active_blocks);
    }
    TORCH_CHECK(
        false, "GEMM execution failed: status=", cutlassGetStatusString(status), " cuda=", cudaGetErrorString(ce));
  }
}

}  // namespace sgl_kernel::w4a8_detail
