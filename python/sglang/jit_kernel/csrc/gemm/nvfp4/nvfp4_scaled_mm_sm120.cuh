/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "nvfp4_scaled_mm_common.cuh"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

struct sm120_fp4_config_small_m {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _256>;
  using PerSmTileShape_MNK = Shape<_128, _128, _256>;
};

struct sm120_fp4_config_M256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

struct sm120_fp4_config_default {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

template <typename Config, typename OutType>
struct Fp4GemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape_MNK,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      void,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
typename Gemm::Arguments args_from_options_sm120(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int M,
    int N,
    int K) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()),
       layout_SFB},
      {{}, nullptr, stride_D, static_cast<ElementD*>(D.data_ptr()), stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());

  return arguments;
}

template <typename Gemm>
void runGemmSm120(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  Gemm gemm;

  auto arguments = args_from_options_sm120<Gemm>(D, A, B, A_sf, B_sf, alpha, M, N, K);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace_tensor = alloc_workspace_tensor(workspace_size, A.device());
  void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));

  CUTLASS_CHECK(gemm.run(arguments, workspace, stream));
}

void cutlass_fp4_bf16_gemm_dispatch_sm120(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 32) {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_small_m, cutlass::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else if (mp2 <= 256) {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_M256, cutlass::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_default, cutlass::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

void cutlass_fp4_f16_gemm_dispatch_sm120(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 32) {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_small_m, cutlass::half_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else if (mp2 <= 256) {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_M256, cutlass::half_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_default, cutlass::half_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

#endif  // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
