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

#include "fp8_blockwise_scaled_mm_common.cuh"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

template <
    typename OutType,
    typename MmaTileShape,
    typename PerSmTileShape,
    typename EpilogueTileShape,
    typename ScalesPerTile,
    int TileSizeM_ = 128,
    class ClusterShape = Shape<_1, _1, _1>>
void launch_sm120_fp8_blockwise_scaled_mm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  using ElementBlockScale = float;

  // A matrix configuration
  using ElementA = cutlass::float_e4m3_t;        // Element type for A matrix operand
  using LayoutATag = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A matrix in units of
                                                    // elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = cutlass::float_e4m3_t;           // Element type for B matrix operand
  using LayoutBTag = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B matrix in units of
                                                    // elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementD = OutType;                      // Element type for D matrix operand
  using ElementC = void;                         // Element type for C matrix operand
  using LayoutCTag = cutlass::layout::RowMajor;  // Layout type for C matrix operand
  using LayoutDTag = cutlass::layout::RowMajor;  // Layout type for D matrix operand
  constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<ElementD>::value;  // Memory access granularity/alignment of C matrix in units of
                                                    // elements (up to 16 bytes)
  constexpr int AlignmentC =
      AlignmentD;  // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // Kernel functional config
  using ElementAccumulator = float;      // Element type for internal accumulation
  using ArchTag = cutlass::arch::Sm120;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag - changed from OpClassBlockScaledTensorOp

  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});

  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      cute::UMMA::Major::MN,
      cute::UMMA::Major::K>;
  // FP8 Block-wise scaling configuration
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());  // Layout type for SFA matrix operand
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());  // Layout type for SFB matrix operand

  constexpr bool kCanUsePingpong = (64 % ScaleGranularityM == 0);

  int m = a.size(0);
  int k = a.size(1);
  int n = b.size(1);

  auto a_ptr = static_cast<ElementA*>(a.data_ptr());
  auto b_ptr = static_cast<ElementB*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  auto scales_a_ptr = static_cast<ElementBlockScale*>(scales_a.data_ptr());
  auto scales_b_ptr = static_cast<ElementBlockScale*>(scales_b.data_ptr());

  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto run_gemm = [&](auto tag) -> cutlass::Status {
    using GemmKernel = decltype(tag);
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    Gemm gemm_op;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideD;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        a_ptr, stride_a, b_ptr, stride_b, scales_a_ptr, layout_SFA, scales_b_ptr, layout_SFB};

    typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, stride_c, c_ptr, stride_c};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        mainloop_args,
        epilogue_args,
    };

    auto can_implement = gemm_op.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
      return can_implement;
    }

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto workspace_tensor = alloc_workspace_tensor(workspace_size, a.device());
    void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

    auto init_status = gemm_op.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
      return init_status;
    }

    return gemm_op.run(stream);
  };

  using CooperativeCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CooperativeStageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CooperativeCollectiveEpilogue::SharedStorage))>;

  using CooperativeCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutATag, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutBTag, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      CooperativeStageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using CooperativeGemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, CooperativeCollectiveMainloop, CooperativeCollectiveEpilogue, void>;

  cutlass::Status status = cutlass::Status::kSuccess;
  if constexpr (kCanUsePingpong) {
    using PingpongMmaTileShape_MNK = Shape<_64, _128, _128>;
    using PingpongCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        PerSmTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC,
        LayoutCTag,
        AlignmentC,
        ElementD,
        LayoutDTag,
        AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using PingpongStageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename PingpongCollectiveEpilogue::SharedStorage))>;

    using PingpongCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        cute::tuple<LayoutATag, LayoutSFA>,
        AlignmentA,
        ElementB,
        cute::tuple<LayoutBTag, LayoutSFB>,
        AlignmentB,
        ElementAccumulator,
        PingpongMmaTileShape_MNK,
        ClusterShape,
        PingpongStageCount,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>::CollectiveOp;

    using PingpongGemmKernel = cutlass::gemm::kernel::
        GemmUniversal<Shape<int, int, int, int>, PingpongCollectiveMainloop, PingpongCollectiveEpilogue, void>;

    if (m <= 64) {
      status = run_gemm(PingpongGemmKernel{});
      if (status != cutlass::Status::kSuccess) {
        status = run_gemm(CooperativeGemmKernel{});
      }
    } else {
      status = run_gemm(CooperativeGemmKernel{});
    }
  } else {
    status = run_gemm(CooperativeGemmKernel{});
  }

  CUTLASS_CHECK(status);
}

// swapAB variant for small token counts (small M). Instead of tiling the small
// token dimension on M (min tile 128 due to the 128-wide 2D weight scale block),
// we compute the transposed GEMM D^T = Wgemm(weight, activation) so tokens land on
// the N axis, where the per-token 1D activation scale (granularity 1) allows a small
// tile N (16/32/64). Mapping:
//   A' = weight  (row-major [N, K], aliases the column-major [K, N] weight)
//   B' = activation (column-major [K, M], aliases the row-major [M, K] activation)
//   D' = out^T  (column-major [N, M], aliases the row-major [M, N] output)
// Scales swap too: weight scale becomes SFA (K-major, 128x128), activation scale
// becomes SFB (MN-major, per-token 1x128). Only the cooperative schedule is used
// since the M' axis (= weight N) is large.
template <
    typename OutType,
    typename MmaTileShape,
    typename PerSmTileShape,
    typename EpilogueTileShape,
    typename ScalesPerTile,
    class ClusterShape = Shape<_1, _1, _1>>
void launch_sm120_fp8_blockwise_scaled_mm_swapab(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  using ElementBlockScale = float;

  using ElementA = cutlass::float_e4m3_t;        // A' = weight
  using LayoutATag = cutlass::layout::RowMajor;  // weight [N, K] is row-major
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::float_e4m3_t;           // B' = activation
  using LayoutBTag = cutlass::layout::ColumnMajor;  // activation as [K, M] column-major
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::ColumnMajor;  // D' = out^T is column-major
  using LayoutDTag = cutlass::layout::ColumnMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});

  // Operands are swapped, so the scale majors swap relative to the non-swap path:
  // SFA (weight) is K-major; SFB (per-token activation) is MN-major.
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      cute::UMMA::Major::K,
      cute::UMMA::Major::MN>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  int m = a.size(0);  // original tokens  -> swapped N'
  int k = a.size(1);
  int n = b.size(1);  // original weight cols -> swapped M'

  auto weight_ptr = static_cast<ElementA*>(b.data_ptr());
  auto act_ptr = static_cast<ElementB*>(a.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  auto weight_scale_ptr = static_cast<ElementBlockScale*>(scales_b.data_ptr());
  auto act_scale_ptr = static_cast<ElementBlockScale*>(scales_a.data_ptr());

  // Swapped problem shape (M', N', K) = (n, m, k).
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, 1));

  auto run_gemm = [&](auto tag) -> cutlass::Status {
    using GemmKernel = decltype(tag);
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    Gemm gemm_op;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideD;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(n, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(m, k, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        weight_ptr, stride_a, act_ptr, stride_b, weight_scale_ptr, layout_SFA, act_scale_ptr, layout_SFB};

    typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, stride_c, c_ptr, stride_c};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {n, m, k, 1},
        mainloop_args,
        epilogue_args,
    };

    auto can_implement = gemm_op.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
      return can_implement;
    }

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto workspace_tensor = alloc_workspace_tensor(workspace_size, a.device());
    void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

    auto init_status = gemm_op.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
      return init_status;
    }

    return gemm_op.run(stream);
  };

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutATag, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutBTag, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  CUTLASS_CHECK(run_gemm(GemmKernel{}));
}

// kForceNoSwap=true forces the legacy non-swapped 128-tile path even for small M
// (used for old-vs-new tile benchmarking); default false enables swapAB for small M.
template <typename OutType, bool kForceNoSwap = false>
void sm120_fp8_blockwise_dispatch_shape(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  const int m = a.size(0);  // token count (NOT padded; may be any value)

  // Use the swapAB path when M is small OR not a multiple of 4. swapAB puts the
  // tokens on a small N tile (tile N=32) where the per-token 1D activation scale
  // (granularity 1 in N) allows the small tile; it also has no M-alignment
  // requirement, so it handles M % 4 != 0 that the non-swap path (which needs
  // 4-aligned M) cannot. This replaces the previous Python-side row padding.
  //
  // A single tile N=32 is used for all swapAB (matching vLLM). Cold-L2 CUPTI
  // benchmarks showed tile N=32 beats tile N=64 for M in (32,64] on typical
  // shapes (up to ~1.2x, e.g. out_proj/gate_up_proj) and only ties on very large
  // N (down_proj). tile N=16 is NOT supported by the SM120 blockwise collective
  // (TMA-store epilogue needs EPI_TILE_N=32 | CTA_N, and the mainloop B LDSM copy
  // needs N >= 32), so 32 is the floor.
  using EpilogueTileShape = Shape<_128, _64>;  // vestigial on SM120 (epilogue tile is auto)
  if constexpr (!kForceNoSwap) {
  if (m <= 64 || (m % 4 != 0)) {
    launch_sm120_fp8_blockwise_scaled_mm_swapab<
        OutType,
        Shape<_128, _32, _128>,
        Shape<_128, _32, _128>,
        EpilogueTileShape,
        Shape<_1, _32, _1>>(out, a, b, scales_a, scales_b, stream);
    return;
  }
  }  // if constexpr (!kForceNoSwap)

  // 4-aligned larger M (or forced): standard (non-swapped) path, 128x128 MMA tile.
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape = Shape<_128, _128, _128>;
  using ScalesPerTile = Shape<_128, _1, _1>;
  launch_sm120_fp8_blockwise_scaled_mm<OutType, MmaTileShape, PerSmTileShape, EpilogueTileShape, ScalesPerTile>(
      out, a, b, scales_a, scales_b, stream);
}

// Non-templated entry: validate inputs and dispatch on the output dtype.
inline void fp8_blockwise_scaled_mm_sm120(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    bool force_noswap = false) {
  RuntimeCheck(mat_a.device().device_type == kDLCUDA, "mat_a must be a CUDA tensor");
  RuntimeCheck(mat_b.device().device_type == kDLCUDA, "mat_b must be a CUDA tensor");

  RuntimeCheck(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  RuntimeCheck(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  RuntimeCheck(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  RuntimeCheck(mat_b.stride(0) == 1, "mat_b must be a column major tensor");
  RuntimeCheck(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");

  RuntimeCheck(
      (mat_a.size(1) * (mat_a.dtype().bits / 8)) % 16 == 0,
      "mat_a must be multiple of 16 bytes for memory alignment");
  RuntimeCheck(
      (mat_b.size(0) * (mat_b.dtype().bits / 8)) % 16 == 0,
      "mat_b must be multiple of 16 bytes for memory alignment");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_a.dtype()), "mat_a must be Float8_e4m3fn");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_b.dtype()), "mat_b must be Float8_e4m3fn");

  RuntimeCheck(mat_a.size(0) == scales_a.size(0), "size of scales_a is not matched");
  RuntimeCheck(mat_a.size(1) / 128 == scales_a.size(1), "size of scales_a is not matched");
  RuntimeCheck(mat_b.size(0) / 128 == scales_b.size(0), "size of scales_b is not matched");
  RuntimeCheck(mat_b.size(1) / 128 == scales_b.size(1), "size of scales_b is not matched");
  RuntimeCheck(host::is_type<float>(scales_a.dtype()), "scales_a must be Float32");
  RuntimeCheck(host::is_type<float>(scales_b.dtype()), "scales_b must be Float32");

  RuntimeCheck(
      (out.size(1) * (out.dtype().bits / 8)) % 16 == 0,
      "out must be multiple of 16 bytes for memory alignment");

  const cudaStream_t stream = LaunchKernel::resolve_device(mat_a.device());

  if (force_noswap) {
    if (host::is_type<bf16_t>(out.dtype())) {
      sm120_fp8_blockwise_dispatch_shape<cutlass::bfloat16_t, true>(out, mat_a, mat_b, scales_a, scales_b, stream);
    } else if (host::is_type<fp16_t>(out.dtype())) {
      sm120_fp8_blockwise_dispatch_shape<cutlass::half_t, true>(out, mat_a, mat_b, scales_a, scales_b, stream);
    } else {
      Panic("out_dtype must be Half or BFloat16");
    }
  } else {
    if (host::is_type<bf16_t>(out.dtype())) {
      sm120_fp8_blockwise_dispatch_shape<cutlass::bfloat16_t, false>(out, mat_a, mat_b, scales_a, scales_b, stream);
    } else if (host::is_type<fp16_t>(out.dtype())) {
      sm120_fp8_blockwise_dispatch_shape<cutlass::half_t, false>(out, mat_a, mat_b, scales_a, scales_b, stream);
    } else {
      Panic("out_dtype must be Half or BFloat16");
    }
  }
}

// Benchmark/debug entry: force a fixed swapAB tile N (32 or 64) regardless of M.
template <int TILE_N>
inline void fp8_blockwise_scaled_mm_sm120_swapab_fixed(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b) {
  const cudaStream_t stream = LaunchKernel::resolve_device(mat_a.device());
  using Mma = Shape<_128, cute::Int<TILE_N>, _128>;
  using ScalesPerTile = Shape<_1, cute::Int<TILE_N>, _1>;
  using Epi = Shape<_128, _64>;
  if (host::is_type<bf16_t>(out.dtype())) {
    launch_sm120_fp8_blockwise_scaled_mm_swapab<cutlass::bfloat16_t, Mma, Mma, Epi, ScalesPerTile>(
        out, mat_a, mat_b, scales_a, scales_b, stream);
  } else if (host::is_type<fp16_t>(out.dtype())) {
    launch_sm120_fp8_blockwise_scaled_mm_swapab<cutlass::half_t, Mma, Mma, Epi, ScalesPerTile>(
        out, mat_a, mat_b, scales_a, scales_b, stream);
  } else {
    Panic("out_dtype must be Half or BFloat16");
  }
}

#endif  // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
