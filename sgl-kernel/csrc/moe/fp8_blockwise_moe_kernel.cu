#include <cutlass/arch/arch.h>
#include <torch/all.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass_moe_helper.cu"
#include "utils.h"

using namespace cute;

template <
    typename OutType,
    typename ScheduleConfig,
    int ScaleGranularityM = 1,
    int ScaleGranularityN = 128,
    int ScaleGranularityK = 128,
    typename LayoutD = cutlass::layout::RowMajor>
void launch_sm100_fp8_blockwise_scaled_group_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    bool transpose = false) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = OutType;
  using ElementD = ElementC;
  using ElementAccumulator = float;
  // Layout definitions
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = LayoutD;

  // Alignment constraints
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  // Architecture definitions
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  // For fp8 block scale.
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      cute::UMMA::Major::K,
      cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutC*,
      AlignmentC,
      typename ScheduleConfig::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA*, LayoutSFA*>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB*, LayoutSFB*>,
      AlignmentB,
      ElementAccumulator,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  int num_experts = (int)expert_offsets.size(0);

  auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(a.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor problem_sizes_transpose = torch::empty(num_experts * 3, options_int);
  if (!transpose) {
    run_get_group_gemm_starts<LayoutSFA, LayoutSFB, ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a,
        b,
        output,
        scales_a,
        scales_b,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose);
  } else {
    run_get_group_gemm_starts<LayoutSFA, LayoutSFB, ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        b,
        a,
        output,
        scales_b,
        scales_a,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose,
        true);
  }

  auto problem_sizes_ptr = static_cast<const int32_t*>(problem_sizes.data_ptr());

  // Use the expert offsets tensor directly
  auto expert_offsets_ptr = static_cast<const int32_t*>(expert_offsets.data_ptr());

  // Create an instance of the GEMM
  Gemm gemm_op;

  // Initialize problem_sizes_as_shapes correctly
  UnderlyingProblemShape* problem_sizes_as_shapes =
      transpose ? static_cast<UnderlyingProblemShape*>(problem_sizes_transpose.data_ptr())
                : static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
      static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;

  hw_info.device_id = a.get_device();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  // Currently, we are only able to do broadcast on either all or none a_scales
  // and on either all or none b_scales
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<StrideC*>(stride_c.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(stride_c.data_ptr())};

  // Use prob_shape in the GEMM arguments
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  // Run the GEMM
  auto status = gemm_op.initialize(args, workspace.get());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void sm100_fp8_blockwise_group_mm_dispatch_shape(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets) {
  struct MMASmallConfig {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _16, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  };

  struct MMALargeConfig {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  };
  if (a.size(0) <= 512) {
    // Swaps the input tensors for performance.
    output = output.t();
    torch::Tensor a_t = a.t();
    torch::Tensor b_t = b.transpose(1, 2);
    torch::Tensor scales_a_t = scales_a.t();
    torch::Tensor scales_b_t = scales_b.transpose(1, 2);
    launch_sm100_fp8_blockwise_scaled_group_mm<OutType, MMASmallConfig, 128, 1, 128, cutlass::layout::ColumnMajor>(
        output,
        a_t,
        b_t,
        scales_a_t,
        scales_b_t,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        true);
    output = output.t();
  } else {
    launch_sm100_fp8_blockwise_scaled_group_mm<OutType, MMALargeConfig, 1, 128, 128>(
        output,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets);
  }
}

void fp8_blockwise_scaled_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets) {
  // Input validation
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0), "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");

  // Get output shapes and create output tensors
  bool can_implement = false;
  auto sm_version = getSMVersion();

#if defined(CUTLASS_ARCH_MMA_SM100A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#if defined CUDA_VERSION && CUDA_VERSION >= 12080
  if (sm_version == 100) {
    if (output.scalar_type() == torch::kBFloat16) {
      sm100_fp8_blockwise_group_mm_dispatch_shape<cutlass::bfloat16_t>(
          output,
          a,
          b,
          scales_a,
          scales_b,
          stride_a,
          stride_b,
          stride_c,
          layout_sfa,
          layout_sfb,
          problem_sizes,
          expert_offsets);
    } else {
      sm100_fp8_blockwise_group_mm_dispatch_shape<cutlass::half_t>(
          output,
          a,
          b,
          scales_a,
          scales_b,
          stride_a,
          stride_b,
          stride_c,
          layout_sfa,
          layout_sfb,
          problem_sizes,
          expert_offsets);
    }
    can_implement = true;
  }
#endif
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      can_implement, "No implemented fp8_blockwise_scaled_mm for current compute capability: ", sm_version);
}