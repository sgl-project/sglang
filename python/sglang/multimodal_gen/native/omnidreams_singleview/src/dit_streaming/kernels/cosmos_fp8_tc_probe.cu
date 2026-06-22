// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cosmos_fp8_tc_probe.cuh"

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"

#include <cstdlib>
#include <string>

namespace omnidreams_singleview {
namespace {

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using namespace cute;

using ProbeClusterShape = Shape<_1, _1, _1>;

void* g_probe_workspace = nullptr;
size_t g_probe_workspace_size = 0;

cudaError_t ensure_probe_workspace(size_t needed, void** workspace) {
  if (!workspace) return cudaErrorInvalidValue;
  if (needed == 0) {
    *workspace = nullptr;
    return cudaSuccess;
  }
  if (needed > g_probe_workspace_size) {
    if (g_probe_workspace) {
      cudaError_t free_err = cudaFree(g_probe_workspace);
      if (free_err != cudaSuccess) return free_err;
      g_probe_workspace = nullptr;
      g_probe_workspace_size = 0;
    }
    cudaError_t alloc_err = cudaMalloc(&g_probe_workspace, needed);
    if (alloc_err != cudaSuccess) return alloc_err;
    g_probe_workspace_size = needed;
  }
  *workspace = g_probe_workspace;
  return cudaSuccess;
}

static bool env_equals(const char* name, const char* value) {
  const char* actual = std::getenv(name);
  return actual && std::string(actual) == value;
}

template <
    class LayoutB,
    class MmaTileShape,
    class BuilderScheduleTag,
    int StageCountOverride>
struct Sm120Fp8ProbeGemm {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ScaleConfig = decltype(cutlass::detail::sm120_trivial_blockwise_scale_config(MmaTileShape{}));
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      MmaTileShape,
      ProbeClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using MainloopStageCount = cute::conditional_t<
      StageCountOverride == 0,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::StageCount<StageCountOverride>>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      ElementA,
      cute::tuple<LayoutA, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ProbeClusterShape,
      MainloopStageCount,
      BuilderScheduleTag>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <
    class LayoutB,
    class MmaTileShape,
    class BuilderScheduleTag,
    int StageCountOverride>
cudaError_t run_probe_gemm(
    const cutlass::float_e4m3_t* a,
    const cutlass::float_e4m3_t* b,
    const float* a_scale,
    const float* b_scale,
    const cutlass::bfloat16_t* c_scratch,
    cutlass::bfloat16_t* out,
    int m,
    int n,
    int k,
    int batch_count,
    cudaStream_t stream) {
  if (!a || !b || !a_scale || !b_scale || !c_scratch || !out) {
    return cudaErrorInvalidValue;
  }
  if (m <= 0 || n <= 0 || k <= 0 || batch_count <= 0) {
    return cudaErrorInvalidValue;
  }
  if ((m % 128) != 0 || (n % 128) != 0 || (k % 128) != 0) {
    return cudaErrorInvalidValue;
  }

  using Probe = Sm120Fp8ProbeGemm<LayoutB, MmaTileShape, BuilderScheduleTag, StageCountOverride>;
  using ScaleConfig = typename Probe::ScaleConfig;
  using Gemm = typename Probe::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto shape_a = cute::make_shape(m, k, batch_count);
  auto shape_b = cute::make_shape(n, k, batch_count);
  auto shape_c = cute::make_shape(m, n, batch_count);
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, shape_a);
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, shape_b);
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, shape_c);
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, shape_c);
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, batch_count));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, batch_count));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, batch_count},
      {a, stride_a, b, stride_b, a_scale, layout_sfa, b_scale, layout_sfb},
      {
          {},
          c_scratch,
          stride_c,
          out,
          stride_d,
      }};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  cudaError_t err = ensure_probe_workspace(workspace_size, &workspace);
  if (err != cudaSuccess) {
    return err;
  }
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaGetLastError();
}

template <class LayoutB>
cudaError_t run_probe_gemm_autotuned(
    const cutlass::float_e4m3_t* a,
    const cutlass::float_e4m3_t* b,
    const float* a_scale,
    const float* b_scale,
    const cutlass::bfloat16_t* c_scratch,
    cutlass::bfloat16_t* out,
    int m,
    int n,
    int k,
    int batch_count,
    cudaStream_t stream) {
  const bool use_m128n64k128 = env_equals("OMNIDREAMS_DIT_FP8_TC_TILE", "m128n64k128");
  const bool use_pingpong = env_equals("OMNIDREAMS_DIT_FP8_TC_SCHEDULE", "pingpong");
  const bool use_stage2 = env_equals("OMNIDREAMS_DIT_FP8_TC_STAGE", "2");

  if (use_m128n64k128) {
    if (use_stage2 && use_pingpong) {
      return run_probe_gemm<
          LayoutB,
          Shape<_128, _64, _128>,
          cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120,
          2>(
              a, b, a_scale, b_scale, c_scratch, out, m, n, k, batch_count, stream);
    }
    if (use_stage2) {
      return run_probe_gemm<
          LayoutB,
          Shape<_128, _64, _128>,
          cutlass::gemm::collective::KernelScheduleAuto,
          2>(
              a, b, a_scale, b_scale, c_scratch, out, m, n, k, batch_count, stream);
    }
    if (use_pingpong) {
      return run_probe_gemm<
          LayoutB,
          Shape<_128, _64, _128>,
          cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120,
          0>(
              a, b, a_scale, b_scale, c_scratch, out, m, n, k, batch_count, stream);
    }
    return run_probe_gemm<
        LayoutB,
        Shape<_128, _64, _128>,
        cutlass::gemm::collective::KernelScheduleAuto,
        0>(
            a, b, a_scale, b_scale, c_scratch, out, m, n, k, batch_count, stream);
  }

  if (use_stage2 && use_pingpong) {
    return run_probe_gemm<
        LayoutB,
        Shape<_128, _128, _128>,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120,
        2>(
            a, b, a_scale, b_scale, c_scratch, out, m, n, k, batch_count, stream);
  }
  if (use_stage2) {
    return run_probe_gemm<
        LayoutB,
        Shape<_128, _128, _128>,
        cutlass::gemm::collective::KernelScheduleAuto,
        2>(
            a, b, a_scale, b_scale, c_scratch, out, m, n, k, batch_count, stream);
  }
  if (use_pingpong) {
    return run_probe_gemm<
        LayoutB,
        Shape<_128, _128, _128>,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120,
        0>(
            a, b, a_scale, b_scale, c_scratch, out, m, n, k, batch_count, stream);
  }
  return run_probe_gemm<
      LayoutB,
      Shape<_128, _128, _128>,
      cutlass::gemm::collective::KernelScheduleAuto,
      0>(
          a, b, a_scale, b_scale, c_scratch, out, m, n, k, batch_count, stream);
}

#endif

}  // namespace

cudaError_t run_cosmos_fp8_tc_probe_qk(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const float* q_scale,
    const float* k_scale,
    const cutlass::bfloat16_t* c_scratch,
    cutlass::bfloat16_t* out,
    int Mq,
    int Mk,
    int D,
    cudaStream_t stream) {
  return run_cosmos_fp8_tc_probe_qk_batched(
      q, k, q_scale, k_scale, c_scratch, out, Mq, Mk, D, 1, stream);
}

cudaError_t run_cosmos_fp8_tc_probe_qk_batched(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const float* q_scale,
    const float* k_scale,
    const cutlass::bfloat16_t* c_scratch,
    cutlass::bfloat16_t* out,
    int Mq,
    int Mk,
    int D,
    int batch_count,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return run_probe_gemm_autotuned<cutlass::layout::ColumnMajor>(
      q, k, q_scale, k_scale, c_scratch, out, Mq, Mk, D, batch_count, stream);
#else
  (void)q;
  (void)k;
  (void)q_scale;
  (void)k_scale;
  (void)c_scratch;
  (void)out;
  (void)Mq;
  (void)Mk;
  (void)D;
  (void)batch_count;
  (void)stream;
  return cudaErrorNotSupported;
#endif
}

cudaError_t run_cosmos_fp8_tc_probe_pv(
    const cutlass::float_e4m3_t* probs,
    const cutlass::float_e4m3_t* v,
    const float* probs_scale,
    const float* v_scale,
    const cutlass::bfloat16_t* c_scratch,
    cutlass::bfloat16_t* out,
    int Mq,
    int Mk,
    int D,
    cudaStream_t stream) {
  return run_cosmos_fp8_tc_probe_pv_batched(
      probs, v, probs_scale, v_scale, c_scratch, out, Mq, Mk, D, 1, stream);
}

cudaError_t run_cosmos_fp8_tc_probe_pv_batched(
    const cutlass::float_e4m3_t* probs,
    const cutlass::float_e4m3_t* v,
    const float* probs_scale,
    const float* v_scale,
    const cutlass::bfloat16_t* c_scratch,
    cutlass::bfloat16_t* out,
    int Mq,
    int Mk,
    int D,
    int batch_count,
    cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return run_probe_gemm_autotuned<cutlass::layout::ColumnMajor>(
      probs, v, probs_scale, v_scale, c_scratch, out, Mq, D, Mk, batch_count, stream);
#else
  (void)probs;
  (void)v;
  (void)probs_scale;
  (void)v_scale;
  (void)c_scratch;
  (void)out;
  (void)Mq;
  (void)Mk;
  (void)D;
  (void)batch_count;
  (void)stream;
  return cudaErrorNotSupported;
#endif
}

}  // namespace omnidreams_singleview
