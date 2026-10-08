#pragma once

// Misc
#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_size.h"

// Collective Builder
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

// Integration
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

namespace expert_specialization {

using namespace cute;

// Different configs for 1SM and 2SM MMA kernel
struct MMA2SMConfig {
  using MmaTileShape = Shape<_256, _128, _128>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
const dim3 MMA2SMConfig::preferred_cluster(4, 1, 1);
const dim3 MMA2SMConfig::fallback_cluster(2, 1, 1);

template <typename _MMAConfig, typename OutputDtype>
struct ExpertSpecializationSm100MXFP8BlockscaledMoeGroupGemmTraits {
  using MMAConfig = _MMAConfig;
  using ElementInput = cutlass::float_e4m3_t;
  using ElementOutput = OutputDtype;
  using ProblemShape = cutlass::gemm::MoEProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

  // A matrix configuration
  using ElementA = cutlass::mx_float8_t<ElementInput>;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr static int AlignmentA = 16;

  // B matrix configuration
  using ElementB = cutlass::mx_float8_t<ElementInput>;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr static int AlignmentB = 16;

  // C/D matrix configuration
  using ElementC = void;
  using ElementD = ElementOutput;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;
  constexpr static int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;
  constexpr static int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using CustomEVTIdentity =  // acc
      cutlass::epilogue::fusion::Sm90EVT<
          cutlass::epilogue::fusion::
              Sm90Compute<cutlass::epilogue::thread::Identity, ElementD, ElementAccumulator, RoundStyle>,
          cutlass::epilogue::fusion::Sm90AccFetch>;

  // Core kernel configurations
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;

  // Runtime Cluster Shape
  using ClusterShape = Shape<int32_t, int32_t, _1>;

  // Define Epilogue
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename MMAConfig::MmaTileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutD*,
      AlignmentD,
      typename MMAConfig::EpilogueSchedule,
      CustomEVTIdentity>::CollectiveOp;

  // Define Mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA,
      AlignmentA,
      ElementB,
      LayoutB*,
      AlignmentB,
      ElementAccumulator,
      typename MMAConfig::MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename MMAConfig::KernelSchedule>::CollectiveOp;

  // Define GemmKernel
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using ElementSF = typename GemmKernel::ElementSF;
};

}  // namespace expert_specialization
