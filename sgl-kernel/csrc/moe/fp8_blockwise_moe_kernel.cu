#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
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

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

template <typename OutType, typename ScheduleConfig, typename LayoutD>
void launch_sm90_fp8_blockwise_scaled_group_mm(
    torch::Tensor& out_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = void;
  using ElementD = OutType;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = LayoutD;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using CustomEVTIdentity =  // acc
      cutlass::epilogue::fusion::Sm90EVT<
          cutlass::epilogue::fusion::
              Sm90Compute<cutlass::epilogue::thread::Identity, ElementD, ElementAccumulator, RoundStyle>,
          cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,  // Use void to avoid load Matrix C
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutC*,
      AlignmentC,
      typename ScheduleConfig::EpilogueSchedule,
      CustomEVTIdentity>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA*, typename ScheduleConfig::LayoutSFA*>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB*, typename ScheduleConfig::LayoutSFB*>,
      AlignmentB,
      ElementAccumulator,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  int num_experts = (int)expert_offsets.size(0);
  Gemm gemm_op;

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
      static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFB*>(layout_sfb.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<StrideC*>(stride_c.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(stride_c.data_ptr())};

  UnderlyingProblemShape* problem_sizes_as_shapes = static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  at::cuda::CUDAGuard device_guard{(char)a_ptrs.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a_ptrs.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType, typename ScheduleConfig, typename LayoutD>
void launch_sm100_fp8_blockwise_scaled_group_mm(
    torch::Tensor& out_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = OutType;
  using ElementD = ElementC;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = LayoutD;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      void,
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
      cute::tuple<LayoutA*, typename ScheduleConfig::LayoutSFA*>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB*, typename ScheduleConfig::LayoutSFB*>,
      AlignmentB,
      ElementAccumulator,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  int num_experts = (int)expert_offsets.size(0);
  // Create an instance of the GEMM
  Gemm gemm_op;

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
      static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFB*>(layout_sfb.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;

  hw_info.device_id = 0;
  // sm_count is the number of SMs on the current device, since we only support SM100 blackwell, so we set it to 148
  hw_info.sm_count = 148;
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<StrideC*>(stride_c.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(stride_c.data_ptr())};

  UnderlyingProblemShape* problem_sizes_as_shapes = static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  at::cuda::CUDAGuard device_guard{(char)a_ptrs.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a_ptrs.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void sm100_fp8_blockwise_group_mm_dispatch_shape(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
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
    const torch::Tensor& workspace) {
  // Check the first matrix size to decide on the configuration
  // Assuming all matrices in the group have similar size characteristics
  // bool use_small_config = a[0].size(0) <= 128;
  struct MmaConfig1 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_256, _32, _128>;
    using ClusterShape = Shape<_2, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
    using ScaleConfig =
        cutlass::detail::Sm100BlockwiseScaleConfig<128, 1, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  struct MmaConfig2 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig =
        cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  struct MmaConfig3 {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_64, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;  // Layout type for SFB matrix operand
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig =
        cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };
  int num_experts = (int)expert_offsets.size(0);
  torch::TensorOptions options_int = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
  torch::Tensor problem_sizes_transpose = torch::empty(num_experts * 3, options_int);
  torch::Tensor output_t = output.t();
  torch::Tensor a_t = a.t();
  torch::Tensor b_t = b.transpose(1, 2);
  torch::Tensor scales_a_t = scales_a.t();
  torch::Tensor scales_b_t = scales_b.transpose(1, 2);

  if (a.size(0) <= 2048 && a.size(1) >= 2048) {
    run_get_group_gemm_starts<MmaConfig1::LayoutSFA, MmaConfig1::LayoutSFB, MmaConfig1::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        b_t,
        a_t,
        output_t,
        scales_b_t,
        scales_a_t,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose,
        true);
    launch_sm100_fp8_blockwise_scaled_group_mm<OutType, MmaConfig1, cutlass::layout::ColumnMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes_transpose,
        expert_offsets,
        workspace);
    output = output_t.t();
  } else if (a.size(0) > 2048 && a.size(1) >= 2048) {
    run_get_group_gemm_starts<MmaConfig2::LayoutSFA, MmaConfig2::LayoutSFB, MmaConfig2::ScaleConfig>(
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
    launch_sm100_fp8_blockwise_scaled_group_mm<OutType, MmaConfig2, cutlass::layout::RowMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace);
  } else {
    run_get_group_gemm_starts<MmaConfig3::LayoutSFA, MmaConfig3::LayoutSFB, MmaConfig3::ScaleConfig>(
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
    launch_sm100_fp8_blockwise_scaled_group_mm<OutType, MmaConfig3, cutlass::layout::RowMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
        workspace);
  }
}

template <typename OutType>
void sm90_fp8_blockwise_group_mm_dispatch_shape(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
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
    const torch::Tensor& workspace) {
  struct MmaConfigSmallM {
    // Swap A/B
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _32, _128>;
    using ClusterShape = Shape<_2, _1, _1>;
    // TODO: Check Pingpong or Cooperative
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    using ScaleConfig =
        cutlass::detail::Sm90BlockwiseScaleConfig<128, 1, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };

  struct MmaConfigH20LargeK {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_64, _128, _128>;
    using ClusterShape = Shape<_2, _1, _1>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8Blockwise;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
    using ScaleConfig =
        cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };

  struct MmaConfigHx00AndH20SmallK {
    using ElementA = cutlass::float_e4m3_t;
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    using ScaleConfig =
        cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  };

  int num_experts = (int)expert_offsets.size(0);
  torch::TensorOptions options_int = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
  torch::Tensor problem_sizes_transpose = torch::empty(num_experts * 3, options_int);
  torch::Tensor output_t = output.t();
  torch::Tensor a_t = a.t();
  torch::Tensor b_t = b.transpose(1, 2);
  torch::Tensor scales_a_t = scales_a.t();
  torch::Tensor scales_b_t = scales_b.transpose(1, 2);

  const std::string H20_device_type_str("NVIDIA H20");
  bool is_h20_device = std::string(at::cuda::getCurrentDeviceProperties()->name) == H20_device_type_str;

  if (a.size(0) <= 2048) {
    run_get_group_gemm_starts<MmaConfigSmallM::LayoutSFA, MmaConfigSmallM::LayoutSFB, MmaConfigSmallM::ScaleConfig>(
        expert_offsets,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        b_t,
        a_t,
        output_t,
        scales_b_t,
        scales_a_t,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        problem_sizes_transpose,
        true);
    launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfigSmallM, cutlass::layout::ColumnMajor>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes_transpose,
        expert_offsets,
        workspace);
    output = output_t.t();
  } else {
    if (is_h20_device && a.size(1) > 128) {
      // For H20 with K > 128, use Pingpong Schedule
      run_get_group_gemm_starts<
          MmaConfigH20LargeK::LayoutSFA,
          MmaConfigH20LargeK::LayoutSFB,
          MmaConfigH20LargeK::ScaleConfig>(
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
      launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfigH20LargeK, cutlass::layout::RowMajor>(
          out_ptrs,
          a_ptrs,
          b_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
          stride_a,
          stride_b,
          stride_c,
          layout_sfa,
          layout_sfb,
          problem_sizes,
          expert_offsets,
          workspace);
    } else {
      // For H20 with K <= 128, and H100 & H200 & H800, use Cooperative Schedule
      run_get_group_gemm_starts<
          MmaConfigHx00AndH20SmallK::LayoutSFA,
          MmaConfigHx00AndH20SmallK::LayoutSFB,
          MmaConfigHx00AndH20SmallK::ScaleConfig>(
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
      launch_sm90_fp8_blockwise_scaled_group_mm<OutType, MmaConfigHx00AndH20SmallK, cutlass::layout::RowMajor>(
          out_ptrs,
          a_ptrs,
          b_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
          stride_a,
          stride_b,
          stride_c,
          layout_sfa,
          layout_sfb,
          problem_sizes,
          expert_offsets,
          workspace);
    }
  }
}

/**
 * @brief Performs blockwise grouped matrix multiplication on FP8 quantized inputs,
 *        with per-block scaling.
 *
 * This function dispatches to hardware-specific implementations (e.g., SM100 FP8)
 * to compute:
 *     C_i = scale_a[i] * A_i * scale_b[i] * B_i
 * for each expert group `i`, using input `problem_sizes` and `expert_offsets`
 * to describe the individual matrix dimensions and their offsets.
 *
 * Input tensors A and B must be quantized to 8-bit formats and dequantized before multiplication.
 * The output tensor is written with bfloat16 or half precision.
 *
 * @param output         Output tensor (must be of type bfloat16 or half).
 * @param a              Input tensor A (must be kFloat8_e4m3fn).
 * @param b              Input tensor B (must be kFloat8_e4m3fn).
 * @param scales_a       Scaling factors for tensor A, float32 per expert group.
 * @param scales_b       Scaling factors for tensor B, float32 per expert group.
 * @param stride_a       Stride information for tensor A (int32).
 * @param stride_b       Stride information for tensor B (int32).
 * @param stride_c       Stride information for output tensor C (int32).
 * @param layout_sfa     Layout descriptor for A (int32), e.g., row-major/column-major.
 * @param layout_sfb     Layout descriptor for B (int32).
 * @param problem_sizes  2D int32 tensor of shape (num_experts, 3), specifying (M, N, K)
 *                       for each grouped matrix multiplication problem.
 * @param expert_offsets 1D int32 tensor of size (num_experts), used to index into
 *                       the grouped input tensors for dispatch.
 *  @note Performance Optimization:
 *       If the batch size (a.size(0)) is smaller than 512, the implementation
 *       will internally transpose input matrices to align with the optimal memory access
 *       pattern for better GPU efficiency. This transformation is done within the kernel.
 */
void fp8_blockwise_scaled_grouped_mm(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
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
    const torch::Tensor& workspace) {
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0), "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "a must be kFloat8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "b must be kFloat8_e4m3fn");
  TORCH_CHECK(
      output.scalar_type() == torch::kBFloat16 || output.scalar_type() == torch::kHalf,
      "output must be bfloat16 or half");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be float32");
  TORCH_CHECK(stride_a.scalar_type() == torch::kInt64, "stride_a must be int64");
  TORCH_CHECK(stride_b.scalar_type() == torch::kInt64, "stride_b must be int64");
  TORCH_CHECK(stride_c.scalar_type() == torch::kInt64, "stride_c must be int64");
  TORCH_CHECK(layout_sfa.scalar_type() == torch::kInt32, "layout_sfa must be int32");
  TORCH_CHECK(layout_sfb.scalar_type() == torch::kInt32, "layout_sfb must be int32");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32, "expert_offsets must be int32");

  TORCH_CHECK(output.dim() == 2, "output must be 2D tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D tensor");
  TORCH_CHECK(b.dim() == 3, "b must be 3D tensor");
  TORCH_CHECK(scales_a.dim() == 2, "scales_a must be 2D tensor");
  TORCH_CHECK(scales_b.dim() == 3, "scales_b must be 3D tensor");
  TORCH_CHECK(stride_a.dim() == 1, "stride_a must be 1D tensor");
  TORCH_CHECK(stride_b.dim() == 1, "stride_b must be 1D tensor");
  TORCH_CHECK(stride_c.dim() == 1, "stride_c must be 1D tensor");
  TORCH_CHECK(layout_sfa.dim() == 2, "layout_sfa must be 1D tensor");
  TORCH_CHECK(layout_sfb.dim() == 2, "layout_sfb must be 1D tensor");
  TORCH_CHECK(a_ptrs.dim() == 1, "a_ptrs must be 1D tensor");
  TORCH_CHECK(b_ptrs.dim() == 1, "b_ptrs must be 1D tensor");
  TORCH_CHECK(out_ptrs.dim() == 1, "out_ptrs must be 1D tensor");
  TORCH_CHECK(a_scales_ptrs.dim() == 1, "a_scales_ptrs must be 1D tensor");
  TORCH_CHECK(b_scales_ptrs.dim() == 1, "b_scales_ptrs must be 1D tensor");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be 1D tensor");
  TORCH_CHECK(workspace.dim() == 1, "workspace must be 1D tensor");

  bool can_implement = false;
  auto sm_version = getSMVersion();

#if defined(CUTLASS_ARCH_MMA_SM100A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#if defined CUDA_VERSION && CUDA_VERSION >= 12080
  if (sm_version == 100
#if CUDA_VERSION >= 12090
      || sm_version == 103
#endif
  ) {
    if (output.scalar_type() == torch::kBFloat16) {
      sm100_fp8_blockwise_group_mm_dispatch_shape<cutlass::bfloat16_t>(
          output,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
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
          expert_offsets,
          workspace);
    } else {
      sm100_fp8_blockwise_group_mm_dispatch_shape<cutlass::half_t>(
          output,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
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
          expert_offsets,
          workspace);
    }
    can_implement = true;
  }
#endif
#endif

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  if (sm_version == 90) {
    if (output.scalar_type() == torch::kBFloat16) {
      sm90_fp8_blockwise_group_mm_dispatch_shape<cutlass::bfloat16_t>(
          output,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
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
          expert_offsets,
          workspace);
    } else {
      sm90_fp8_blockwise_group_mm_dispatch_shape<cutlass::half_t>(
          output,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
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
          expert_offsets,
          workspace);
    }
    can_implement = true;
  }
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      can_implement, "No implemented fp8_blockwise_scaled_grouped_mm for current compute capability: ", sm_version);
}
