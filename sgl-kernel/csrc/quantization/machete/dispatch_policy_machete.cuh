#pragma once

#include "cute/atom/copy_traits_sm100.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"  // cute::false_type
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"
namespace cutlass {
namespace gemm {
struct KernelTmaWarpSpecializedCooperativeMachete {
  static constexpr int SchedulerPipelineStageCount = 0;
};
}  // namespace gemm
}  // namespace cutlass
