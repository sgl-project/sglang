#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include <cstdlib>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "w4a8_grouped_mm_c3x.cuh"

using namespace cute;
using sgl_kernel::w4a8_detail::cutlass_3x_w4a8_group_gemm;
using sgl_kernel::w4a8_detail::cutlass_w4a8_group_gemm_caller;

namespace {

enum class Sched { PP, CO };

// Weight quant format selector. INT4 keeps the original int4a8 path byte-identical;
// MXFP4 selects the E2M1 weight element with an E8M0 (block=32) group size.
enum class WType { INT4, MXFP4 };

template <WType W>
struct QuantTraits {
  using Element = cutlass::int4b_t;
  static constexpr int GroupSize = 128;
};
template <>
struct QuantTraits<WType::MXFP4> {
  using Element = cutlass::float_e2m1_t;
  static constexpr int GroupSize = 32;
};

template <int M, int N, int K, int A, int B, int C, Sched S, WType W = WType::INT4>
struct SM90W4A8Config {
  using KernelSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;

  using EpilogueSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;

  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;
  using Cutlass3xW4A8Gemm = cutlass_3x_w4a8_group_gemm<
      TileShape,
      ClusterShape,
      KernelSchedule,
      EpilogueSchedule,
      typename QuantTraits<W>::Element,
      QuantTraits<W>::GroupSize>;
};

template <int M, int N, int K, int A, int B, int C>
using SM90_PP = SM90W4A8Config<M, N, K, A, B, C, Sched::PP>;

template <int M, int N, int K, int A, int B, int C>
using SM90_CO = SM90W4A8Config<M, N, K, A, B, C, Sched::CO>;

// MXFP4 variants (E2M1 weight, E8M0 block=32 group size).
template <int M, int N, int K, int A, int B, int C>
using SM90_PP_MXFP4 = SM90W4A8Config<M, N, K, A, B, C, Sched::PP, WType::MXFP4>;

template <int M, int N, int K, int A, int B, int C>
using SM90_CO_MXFP4 = SM90W4A8Config<M, N, K, A, B, C, Sched::CO, WType::MXFP4>;

template <int N>
struct SM90_SWG_MXFP4 {
  using TileShape = cute::Shape<cute::Int<128>, cute::Int<N>, cute::Int<128>>;
  using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
  using Cutlass3xW4A8Gemm = cutlass_3x_w4a8_group_gemm<
      TileShape,
      ClusterShape,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
      typename QuantTraits<WType::MXFP4>::Element,
      QuantTraits<WType::MXFP4>::GroupSize,
      true,
      true>;
};

// General two-consumer-warpgroup fused MXFP4A8 tactic. It keeps the pre-MMA E8M0
// mainloop and the existing chunk-major precomputed work map, while using the
// regular ping-pong kernel for larger token tiles.
template <
    int M,
    int N,
    int K,
    int ClusterM = 1,
    int ClusterN = 1,
    bool ChunkMajorWorkMap = true,
    sgl_kernel::swg_detail::ExpertRowPolicy RowPolicy = sgl_kernel::swg_detail::ExpertRowPolicy::All>
struct SM90_PRECOMPUTED_MXFP4 {
  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<ClusterM>, cute::Int<ClusterN>, cute::Int<1>>;
  using Cutlass3xW4A8Gemm = cutlass_3x_w4a8_group_gemm<
      TileShape,
      ClusterShape,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
      typename QuantTraits<WType::MXFP4>::Element,
      QuantTraits<WType::MXFP4>::GroupSize,
      false,
      true,
      ChunkMajorWorkMap,
      static_cast<int>(RowPolicy)>;
};

// config334: register warp-transpose GEMM2 epilogue with adjacent BF16 values
// packed into one aligned 32-bit global store per lane. 128x32x512, cluster-1,
// expert-major scheduler (config320 base).
struct SM90_PRECOMPUTED_MXFP4_WARP_SHUFFLE_PACKED_GEMM2 {
  using TileShape = cute::Shape<cute::Int<128>, cute::Int<32>, cute::Int<512>>;
  using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
  using Cutlass3xW4A8Gemm = cutlass_3x_w4a8_group_gemm<
      TileShape,
      ClusterShape,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      sgl_kernel::w4a8_detail::WarpShufflePackedStoreGemm2Epilogue,
      typename QuantTraits<WType::MXFP4>::Element,
      QuantTraits<WType::MXFP4>::GroupSize,
      false,
      true,
      false>;
};

template <typename Config>
inline void invoke_gemm(
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
    // MXFP4A8: optional per-token+per-block activation scale. Defaulted to
    // nullopt so all int4a8 call sites are unchanged.
    std::optional<torch::Tensor> act_block_scales = std::nullopt,
    std::optional<torch::Tensor> as_strides = std::nullopt,
    int64_t act_scale_group = 0,
    std::optional<torch::Tensor> expert_ids = std::nullopt) {
  using GemmT = typename Config::Cutlass3xW4A8Gemm;
  cutlass_w4a8_group_gemm_caller<GemmT>(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      chunk_size,
      act_block_scales,
      as_strides,
      act_scale_group,
      expert_ids);
}

// Helper macro to reduce code duplication.
// Note: Config must be wrapped in parentheses when it contains commas (e.g., template parameters).
// This uses a helper macro to strip the parentheses from the template parameter
#define INVOKE_GEMM_WITH_CONFIG_HELPER(...) \
  invoke_gemm<__VA_ARGS__>(                 \
      d_tensors,                            \
      a_tensors,                            \
      b_tensors,                            \
      a_scales,                             \
      b_scales,                             \
      expert_offsets,                       \
      problem_sizes,                        \
      a_strides,                            \
      b_strides,                            \
      d_strides,                            \
      s_strides,                            \
      chunk_size)
#define INVOKE_GEMM_WITH_CONFIG(Config) INVOKE_GEMM_WITH_CONFIG_HELPER Config

// MXFP4A8 variant that also threads the activation block-scale.
#define INVOKE_GEMM_WITH_CONFIG_AS_HELPER(...) \
  invoke_gemm<__VA_ARGS__>(                    \
      d_tensors,                               \
      a_tensors,                               \
      b_tensors,                               \
      a_scales,                                \
      b_scales,                                \
      expert_offsets,                          \
      problem_sizes,                           \
      a_strides,                               \
      b_strides,                               \
      d_strides,                               \
      s_strides,                               \
      chunk_size,                              \
      act_block_scales,                        \
      as_strides,                              \
      act_scale_group,                         \
      expert_ids)
#define INVOKE_GEMM_WITH_CONFIG_AS(Config) INVOKE_GEMM_WITH_CONFIG_AS_HELPER Config

inline int get_forced_mxfp4_config(char const* name) {
  char const* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return -1;
  }
  return std::atoi(value);
}

void dispatch_w4a8_moe_mm_sm90(
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
    int64_t topk) {
#if defined(SGL_KERNEL_DISABLE_W4A8_INT4)
  TORCH_CHECK(false, "INT4 W4A8 grouped GEMM kernels are disabled in this build");
#else
  uint32_t const m = a_tensors.size(0) / topk;
  uint32_t const n = d_tensors.size(1);
  uint32_t const k = a_tensors.size(1);

  if (n == 4096 && k == 7168) {
    // group gemm 1
    if (m <= 4) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 512, 2, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 2, 1, 1>));
    } else if (m <= 256) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 2, 1, 1>));
    } else if (m <= 4096) {
      // Optimized for prefill: seq_len up to 4096 (m=4096 with topk=1)
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 2, 1, 1>));
    } else {
      // Optimized for prefill: seq_len up to 8192 (m=8192 with topk=1)
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 7168 && k == 2048) {
    // group gemm 2
    if (m <= 8) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 16, 512, 1, 1, 1>));
    } else if (m <= 512) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 1, 1, 1>));
    } else if (m <= 4096) {
      // Optimized for prefill: larger cluster for better throughput
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 2, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 512 && k == 7168) {
    // group gemm 1 for tp
    if (m <= 4) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 512, 2, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 2, 1, 1>));
    } else if (m <= 256) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 2, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 7168 && k == 256) {
    // group gemm 2 for tp
    if (m <= 8) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 16, 128, 1, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 1, 1, 1>));
    } else if (m <= 512) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 2, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 64, 128, 1, 1, 1>));
    }
  } else {
    if (k % 512 == 0) {
      // For large m (prefill), prefer larger cluster
      if (m <= 32) {
        // Decode: target batch size (16-32) - use cluster size 1 for better latency
        INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>));
      } else if (m <= 1024) {
        // Decode: large batch or small prefill
        INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 1, 1, 1>));
      } else {
        // Prefill: large sequence length - prefer larger cluster
        INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
      }
    } else {
      if (m <= 32) {
        // Decode: target batch size (16-32) - use larger tile for better throughput
        INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 1, 1, 1>));
      } else {
        // Prefill: larger sequence length
        INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 64, 128, 1, 1, 1>));
      }
    }
  }
#endif
}

void dispatch_w4a8_mxfp4_moe_mm_sm90(
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
    int64_t topk,
    // MXFP4A8: per-token+per-block activation scale (bf16 [total_m, K/32]) and
    // its per-expert stride array. When present the mainloop uses the mxfp8
    // activation path; otherwise it falls back to per-tensor (epilogue alpha).
    std::optional<torch::Tensor> act_block_scales,
    std::optional<torch::Tensor> as_strides,
    int64_t act_scale_group,
    std::optional<torch::Tensor> expert_ids = std::nullopt) {
  uint32_t const m = a_tensors.size(0) / topk;
  uint32_t const n = d_tensors.size(1);
  uint32_t const k = a_tensors.size(1);
  bool const compact_groups = expert_ids.has_value();

  // MXFP4A8 TMA constraint: the group scale is loaded as one packed element
  // Array<bf16, TileK / GroupSize>. SM90 TMA supports at most a 64-bit element,
  // so GroupSize=32 hard-locks TileK=128 and PackedScalesNum=4. TileK=256 would
  // instantiate Array<bf16,8> (uint128_t), for which no SM90 TMA format exists.
  // Host-side weight and activation scale packing must therefore remain 4-wide.
  if (n == 4096 && k == 4096) {
    // Acceptance MXFP4A8 GEMM1 (hidden=4096, 2*inter=4096, E=256, topk=6).
    int const forced_config = get_forced_mxfp4_config("SGL_MXFP4A8_GEMM1_CONFIG");
    if (forced_config >= 0) {
      switch (forced_config) {
        case 0:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 16, 128, 1, 1, 1>));
          return;
        case 1:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 16, 128, 2, 1, 1>));
          return;
        case 2:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 32, 128, 1, 1, 1>));
          return;
        case 3:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 32, 128, 2, 1, 1>));
          return;
        case 4:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 64, 128, 1, 1, 1>));
          return;
        case 5:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 64, 128, 2, 1, 1>));
          return;
        case 6:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 128, 128, 1, 1, 1>));
          return;
        case 7:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 128, 128, 2, 1, 1>));
          return;
        case 100:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_SWG_MXFP4<8>));
          return;
        case 101:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_SWG_MXFP4<16>));
          return;
        default:
          TORCH_CHECK(false, "Unsupported SGL_MXFP4A8_GEMM1_CONFIG=", forced_config);
      }
    }
    if (compact_groups && m <= 256) {
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 16, 128, 1, 1, 1>));
    } else if (m <= 256) {
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 16, 128, 1, 1, 1>));
    } else if (m <= 1024) {
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 32, 128, 2, 1, 1>));
    } else if (m <= 2048) {
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 64, 128, 1, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 64, 128, 1, 1, 1>));
    }
  } else if (n == 4096 && k == 2048) {
    // Acceptance MXFP4A8 GEMM2 (inter=2048, hidden=4096, E=256, topk=6).
    int const forced_config = get_forced_mxfp4_config("SGL_MXFP4A8_GEMM2_CONFIG");
    if (forced_config >= 0) {
      switch (forced_config) {
        case 0:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 16, 128, 1, 1, 1>));
          return;
        case 1:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 16, 128, 2, 1, 1>));
          return;
        case 2:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 32, 128, 1, 1, 1>));
          return;
        case 3:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 32, 128, 2, 1, 1>));
          return;
        case 4:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 64, 128, 1, 1, 1>));
          return;
        case 5:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 64, 128, 2, 1, 1>));
          return;
        case 6:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 128, 128, 1, 1, 1>));
          return;
        case 7:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 128, 128, 2, 1, 1>));
          return;
        case 100:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_SWG_MXFP4<8>));
          return;
        case 101:
          INVOKE_GEMM_WITH_CONFIG_AS((SM90_SWG_MXFP4<16>));
          return;
        default:
          TORCH_CHECK(false, "Unsupported SGL_MXFP4A8_GEMM2_CONFIG=", forced_config);
      }
    }
    if (compact_groups && m <= 256) {
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 32, 128, 1, 1, 1>));
    } else if (m <= 256) {
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 16, 128, 1, 1, 1>));
    } else if (m <= 1024) {
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 32, 128, 2, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 64, 128, 2, 1, 1>));
    }
  } else {
    if (k % 128 == 0) {
      // TileK=128 uses the legal 64-bit Array<bf16,4> TMA scale element.
      if (compact_groups && m <= 256) {
        INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 32, 128, 1, 1, 1>));
      } else if (m <= 32) {
        INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 16, 128, 1, 1, 1>));
      } else if (m <= 1024) {
        INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 32, 128, 1, 1, 1>));
      } else {
        INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 64, 128, 1, 1, 1>));
      }
    } else {
      if (m <= 32) {
        // Decode: target batch size (16-32) - use larger tile for better throughput
        INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 32, 128, 1, 1, 1>));
      } else {
        // Prefill: larger sequence length
        INVOKE_GEMM_WITH_CONFIG_AS((SM90_CO_MXFP4<128, 64, 128, 1, 1, 1>));
      }
    }
  }
}

void dispatch_mxfp4a8_fused_moe_mm_sm90(
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
    int64_t topk,
    int64_t swg_config,
    std::optional<torch::Tensor> expert_ids) {
  constexpr int64_t chunk_size = QuantTraits<WType::MXFP4>::GroupSize;
  std::optional<torch::Tensor> act_block_scales = std::nullopt;
  std::optional<torch::Tensor> as_strides = std::nullopt;
  constexpr int64_t act_scale_group = 0;

  TORCH_CHECK(b_tensors.scalar_type() == torch::kInt8, "fused MXFP4A8 interleaved weight must be int8");
  TORCH_CHECK(b_tensors.is_contiguous(), "fused MXFP4A8 interleaved weight must be contiguous");
  TORCH_CHECK(b_scales.scalar_type() == torch::kUInt8, "fused MXFP4A8 folded offset must be uint8");
  TORCH_CHECK(b_scales.is_contiguous(), "fused MXFP4A8 folded offset must be contiguous");
  TORCH_CHECK(a_scales.scalar_type() == torch::kFloat32, "fused MXFP4A8 per-row scale must be float32");
  TORCH_CHECK(a_scales.dim() == 1, "fused MXFP4A8 per-row scale must be 1D");
  TORCH_CHECK(a_scales.is_contiguous(), "fused MXFP4A8 per-row scale must be contiguous");
  TORCH_CHECK(
      a_scales.numel() == a_tensors.numel() / a_tensors.size(-1),
      "fused MXFP4A8 per-row scale must contain one value per activation row");
  TORCH_CHECK(topk > 0, "topk must be positive");

  switch (swg_config) {
    case 100:
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_SWG_MXFP4<8>));
      return;
    case 101:
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_SWG_MXFP4<16>));
      return;
    case 204:
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_PRECOMPUTED_MXFP4<64, 32, 512, 1, 1, false>));
      return;
    case 205:
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_PRECOMPUTED_MXFP4<64, 64, 512, 2, 1, false>));
      return;
    case 313:
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_PRECOMPUTED_MXFP4<64, 64, 512, 1, 1, false>));
      return;
    case 320:
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_PRECOMPUTED_MXFP4<128, 32, 512, 1, 1, false>));
      return;
    case 322:
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_PRECOMPUTED_MXFP4<128, 32, 512>));
      return;
    case 334:
      INVOKE_GEMM_WITH_CONFIG_AS((SM90_PRECOMPUTED_MXFP4_WARP_SHUFFLE_PACKED_GEMM2));
      return;
    default:
      TORCH_CHECK(
          false,
          "Unsupported fused MXFP4A8 config=",
          swg_config,
          "; expected one of 100, 101, 204, 205, 313, 320, 322, 334");
  }
}

}  // namespace

void cutlass_w4a8_moe_mm_sm90(
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
    int64_t topk) {
  dispatch_w4a8_moe_mm_sm90(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      chunk_size,
      topk);
}

// MXFP4A8 entry: identical calling convention to cutlass_w4a8_moe_mm_sm90, but
// the weight operand is MXFP4 (E2M1) with an E8M0 block=32 group scale that has
// been pre-expanded to bf16 on the host side, so the kernel post-MMA scale path
// is reused unchanged.
void cutlass_mxfp4a8_moe_mm_sm90(
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
    int64_t topk,
    std::optional<torch::Tensor> act_block_scales,
    std::optional<torch::Tensor> as_strides,
    int64_t act_scale_group,
    std::optional<torch::Tensor> expert_ids) {
  dispatch_w4a8_mxfp4_moe_mm_sm90(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      chunk_size,
      topk,
      act_block_scales,
      as_strides,
      act_scale_group,
      expert_ids);
}

void cutlass_mxfp4a8_fused_moe_mm_sm90(
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
    int64_t topk,
    int64_t swg_config,
    std::optional<torch::Tensor> expert_ids) {
  dispatch_mxfp4a8_fused_moe_mm_sm90(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      topk,
      swg_config,
      expert_ids);
}
