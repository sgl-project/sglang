// Adapted from
// https://github.com/vllm-project/vllm/blob/16bff144be6739c9f773968ace0b9cd239f67f19/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm90_fp8_dispatch.cuh
// and int8_gemm_kernel.cu in sgl-kernel

#pragma once

// torch include first so this header is self-contained: the shared
// scaled_mm_epilogues_c3x.hpp below references torch::Tensor, and clang-format
// may sort this header's include ahead of the .cu's own torch includes.
#include <torch/all.h>

#include "cutlass_extensions/common.hpp"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "cutlass_extensions/gemm/cutlass_gemm_caller.cuh"

using namespace cute;

// Generic CUTLASS 3.x SM90 GEMM structure for both int8 and fp8 (dtype-generic
// via ElementAcc conditional at line 25). Copied from fp8_gemm_sm90_dispatch.cuh
// and renamed to *_int8 to avoid ODR violation.
template <
    typename ElementAB_,
    typename ElementD_,
    template <typename, typename, typename> typename Epilogue_,
    typename TileShape,
    typename ClusterShape,
    typename KernelSchedule,
    typename EpilogueSchedule,
    bool swap_ab_ = false>
struct cutlass_3x_gemm_sm90_int8 {
  using ElementAB = ElementAB_;
  using ElementC = ElementD_;
  using ElementD = ElementD_;
  using ElementAcc = typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t, float>::type;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;

  using EVTCompute = typename Epilogue::EVTCompute;

  static constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Compile-time swap_ab flag
  static constexpr bool swap_ab = swap_ab_;

  // -----------------------------------------------------------
  // Layout definitions
  // -----------------------------------------------------------
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;

  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;

  // -----------------------------------------------------------
  // Collective epilogue (conditionally swap operands and layouts)
  // -----------------------------------------------------------
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc,
      float,
      ElementC,
      conditional_t<swap_ab, LayoutC_Transpose, LayoutC>,
      AlignmentCD,
      ElementD,
      conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
      AlignmentCD,
      EpilogueSchedule,
      EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize = sizeof(typename CollectiveEpilogue::SharedStorage);

  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(CEStorageSize)>;

  // -----------------------------------------------------------
  // Collective mainloop (conditionally swap operands and layouts)
  // -----------------------------------------------------------
  using CollectiveMainloop = conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          ElementAB,
          LayoutB_T,
          AlignmentAB,  // Swapped B (as A)
          ElementAB,
          LayoutA_T,
          AlignmentAB,  // Swapped A (as B)
          ElementAcc,
          TileShape,
          ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassTensorOp,
          ElementAB,
          LayoutA,
          AlignmentAB,
          ElementAB,
          LayoutB,
          AlignmentAB,
          ElementAcc,
          TileShape,
          ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp>;

  // -----------------------------------------------------------
  // Kernel definition
  // -----------------------------------------------------------
  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_int8_config_default {
  // M in (128, inf)
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;

  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogueBias,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule>,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogue,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule>>;
};

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller_sm90_int8(
    torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b, EpilogueArgs&&... epilogue_params) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;

  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  auto prob_shape = swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args =
      swap_ab ? typename GemmKernel::MainloopArguments{b_ptr, b_stride, a_ptr, a_stride}
              : typename GemmKernel::MainloopArguments{a_ptr, a_stride, b_ptr, b_stride};

  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(std::forward<EpilogueArgs>(epilogue_params)...), c_ptr, c_stride, c_ptr, c_stride};

  cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args, epilogue_args);
}

// Canonical-order caller: every dispatch site passes (a_scales, b_scales) in
// the original (non-swapped) order; this wrapper does the swap internally
// based on the Gemm config's swap_ab flag. Prevents the latent footgun of
// having to remember to re-order scales per-bucket at the call site.
template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller_sm90_int8_scaled(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    EpilogueArgs&&... epilogue_extras) {
  if constexpr (Gemm::swap_ab) {
    return cutlass_gemm_caller_sm90_int8<Gemm>(
        out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(epilogue_extras)...);
  } else {
    return cutlass_gemm_caller_sm90_int8<Gemm>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(epilogue_extras)...);
  }
}

template <typename InType, typename OutType, bool EnableBias>
struct sm90_int8_config_M128_largeN {
  // M in (64, 128], N > 4096 (large-N path; small-N routes to M128_smallN below)
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogueBias,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule>,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogue,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule>>;
};

// Fallback for M in (64, 128] when N <= 4096: the swap dispatch's M128_largeN
// tile falls back to the original int8 non-swap tile for small N to avoid
// regression in this region.
template <typename InType, typename OutType, bool EnableBias>
struct sm90_int8_config_M128_smallN {
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_2, _1, _1>;

  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogueBias,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule>,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogue,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule>>;
};

// Non-swap fallback for M in (16, 64] when N <= kNThreshold. At small N the
// original non-swap tile is already efficient, and the swap M64 tile
// (<64,16,256>) regressed ~12% on H20 here; routing to non-swap restores parity
// (no win, but no regression -- M<=16 still swaps and wins ~1.6x). Tile
// <64,64,128> + cluster <1,4,1> matches the original int8 small-M tile; it is
// distinct from M128_smallN below (same tile but cluster <2,1,1>, which serves
// the (64,128] range). Non-swap, so it uses ScaledEpilogue(Bias) (row bias),
// not the swap configs' ScaledEpilogueColumnBias.
template <typename InType, typename OutType, bool EnableBias>
struct sm90_int8_config_M64_smallN_noswap {
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_1, _4, _1>;

  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogueBias,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule>,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogue,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule>>;
};

// Small-M swap buckets start with FP8 swap tile seeds.
template <typename InType, typename OutType, bool EnableBias>
struct sm90_int8_config_M64_largeN {
  // M in (32, 64], N > 1280
  // Swap-AB path with tile <64,64,256>.
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogueColumnBias,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule,
          true>,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogue,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule,
          true>>;
};

// Dedicated bucket for M_orig in (16, 32], N > 1280. In swap mode, kernel-N
// equals M_orig, so the M64_largeN tile (kernel-N=64) leaves half the N-tile
// padded for M_orig=32 (50% compute waste). This config sets kernel-N=32 to
// match M_orig=32 exactly, recovering the wasted half.
template <typename InType, typename OutType, bool EnableBias>
struct sm90_int8_config_M32_largeN {
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _32, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogueColumnBias,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule,
          true>,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogue,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule,
          true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_int8_config_M16_smallN {
  // M in [1, 16], N in [1, 1280]
  // Swap-AB path with tile <64,16,256>.
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _16, _256>;
  using ClusterShape = Shape<_1, _2, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogueColumnBias,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule,
          true>,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogue,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule,
          true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_int8_config_M16_largeN {
  // M in [1, 16], N > 1280
  // Swap-AB path with tile <64,16,256>.
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _16, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogueColumnBias,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule,
          true>,
      cutlass_3x_gemm_sm90_int8<
          InType,
          OutType,
          c3x::ScaledEpilogue,
          TileShape,
          ClusterShape,
          KernelSchedule,
          EpilogueSchedule,
          true>>;
};

template <typename InType, typename OutType, bool EnableBias, typename... EpilogueArgs>
inline void cutlass_gemm_sm90_int8_dispatch(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, int8_t>());
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  using Cutlass3xGemmDefault = typename sm90_int8_config_default<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM128_largeN = typename sm90_int8_config_M128_largeN<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM128_smallN = typename sm90_int8_config_M128_smallN<InType, OutType, EnableBias>::Cutlass3xGemm;

  using Cutlass3xGemmM64_smallN_noswap =
      typename sm90_int8_config_M64_smallN_noswap<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM64_largeN = typename sm90_int8_config_M64_largeN<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM32_largeN = typename sm90_int8_config_M32_largeN<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM16_smallN = typename sm90_int8_config_M16_smallN<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM16_largeN = typename sm90_int8_config_M16_largeN<InType, OutType, EnableBias>::Cutlass3xGemm;

  uint32_t const m = a.size(0);
  uint32_t const n = b.size(1);

  // Threshold separating "smallN" from "largeN" config variants for the M16
  // and M64 buckets (small N ~= attention/KV proj; large N ~= MLP). Seeded from
  // the FP8 swap-AB dispatch; validated on H20 to give no regression across the
  // benchmark matrix, but not exhaustively int8-tuned (may shift on H100/H200).
  static constexpr uint32_t kNThreshold = 1280;

  // Threshold splitting the M128 bucket into a small-N fallback (non-swap,
  // original int8 tile) vs the larger-N path with swap. 4096 is the empirical
  // crossover; below it the smaller-tile fallback avoids regression.
  static constexpr uint32_t kM128NThreshold = 4096;

  // All dispatch sites pass scales in the canonical (a_scales, b_scales) order;
  // cutlass_gemm_caller_sm90_int8_scaled handles swap-AB internally based on the
  // Gemm config's swap_ab flag.
  if (m <= 16) {
    // m in [1, 16]
    if (n <= kNThreshold) {
      return cutlass_gemm_caller_sm90_int8_scaled<Cutlass3xGemmM16_smallN>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    }
    return cutlass_gemm_caller_sm90_int8_scaled<Cutlass3xGemmM16_largeN>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  } else if (m <= 64) {
    // m in (16, 64]
    if (n <= kNThreshold) {
      // Small N is already efficient with the original non-swap tile; the swap
      // M64_smallN tile (<64,16,256>) regresses ~12% here on H20, so fall back
      // to the non-swap config for (16, 64] x small-N. (M<=16 still swaps.)
      return cutlass_gemm_caller_sm90_int8_scaled<Cutlass3xGemmM64_smallN_noswap>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    }
    if (m <= 32) {
      // M_orig=32 with kernel-N=64 wastes 50% of N-tile; route to M32 tile
      // (kernel-N=32) instead.
      return cutlass_gemm_caller_sm90_int8_scaled<Cutlass3xGemmM32_largeN>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    }
    return cutlass_gemm_caller_sm90_int8_scaled<Cutlass3xGemmM64_largeN>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  } else if (m <= 128) {
    // m in (64, 128]
    if (n <= kM128NThreshold) {
      // small-N: fall back to non-swap tile (recovers regression in this region)
      return cutlass_gemm_caller_sm90_int8_scaled<Cutlass3xGemmM128_smallN>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    }
    return cutlass_gemm_caller_sm90_int8_scaled<Cutlass3xGemmM128_largeN>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  } else {
    // m in (128, inf)
    return cutlass_gemm_caller_sm90_int8_scaled<Cutlass3xGemmDefault>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  }
}

template <bool EnableBias, typename... EpilogueArgs>
void cutlass_scaled_mm_sm90_int8_epilogue(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::bfloat16_t, EnableBias>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::half_t, EnableBias>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_mm_sm90_int8(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(), "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm90_int8_epilogue<true>(out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm90_int8_epilogue<false>(out, a, b, a_scales, b_scales);
  }
}
