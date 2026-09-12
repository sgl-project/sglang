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

/**
 * \brief SM90 grouped GEMM with MXFP4 weights and bf16 activations, for MoE experts.
 *
 * Same vendored mixed-input collective the W4A8 MoE path uses, with MmaType = bfloat16_t instead of
 * fp8 e4m3: the wgmma pipeline keeps its lead over marlin without the activation quantize /
 * dequantize glue around the GEMM. Weights are E2M1 nibble pairs in the checkpoint layout, scales
 * are one E8M0 byte per 32 K-elements folded into the A fragment.
 */

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cuda_runtime.h>

// clang-format off
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp"
// clang-format on

namespace sglang {

using namespace host;
using namespace cute;

#define MXFP4_A16_CUTLASS_CHECK(status)                                              \
  {                                                                                  \
    cutlass::Status error = status;                                                  \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

namespace mxfp4_a16_sm90 {

using ElementAccumulator = float;
using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::float_e2m1_t;
// E8M0 biased exponent, 127 == 1.0. The mainloop decodes E2M1 codes without renormalizing, which
// leaves value * 2^-126, so the loader bakes the missing +126 into these bytes; a true exponent
// above 128 would reach byte 255, which is Inf/NaN in bf16.
using ElementScale = uint8_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = ElementC;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

// The collective computes D^T = B^T A^T, so TileShape M is the weight's output dim and TileShape N
// is the token dim. Weight-dim 128 is the cooperative schedule's minimum; TileK is capped at 256 by
// group-32 scales (the operand is TMA-loaded as Array<uint8_t, TileK / ScaleGroupSize>, and cute has
// no TMA format for a 16-byte element).
static constexpr int ScaleGroupSize = 32;
static constexpr int kTileWeightN = 128;
static constexpr int kTileK = 128;
using ClusterShape = Shape<_1, _1, _1>;
static constexpr int PackedScalesNum = kTileK / ScaleGroupSize;
using ElementScalePacked = cutlass::Array<ElementScale, PackedScalesNum>;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// One instantiation per token-tile width. Only TileShape differs, and the packed-scale layout
// depends on TileK alone, so every width reads the same repacked weights and scales.
template <int kTileTokens>
struct TileTraits {
  using TileShape = Shape<cute::Int<kTileWeightN>, cute::Int<kTileTokens>, cute::Int<kTileK>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
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

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
      ArchTag,
      OperatorClass,
      cute::tuple<QuantType, ElementScalePacked>,
      LayoutB_Transpose*,
      AlignmentB,
      MmaType,
      LayoutA_Transpose*,
      AlignmentA,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
using StrideD = typename TileTraits<kTileWeightN>::GemmKernel::InternalStrideD;
using StrideS = typename TileTraits<kTileWeightN>::CollectiveMainloop::StrideScale;

// One set of per-expert stride arrays is filled before the width is chosen, so the stride types must
// not depend on the token tile.
template <int kTileTokens>
static constexpr bool kStridesMatch =
    cute::is_same_v<StrideD, typename TileTraits<kTileTokens>::GemmKernel::InternalStrideD> &&
    cute::is_same_v<StrideS, typename TileTraits<kTileTokens>::CollectiveMainloop::StrideScale>;
static_assert(
    kStridesMatch<16> && kStridesMatch<32> && kStridesMatch<64>, "token tile changed the operand stride types");

/// \brief A rank-3 cute stride whose every dynamic mode is \p ld.
///
/// Which of the three modes is a compile-time constant differs per operand and moved between
/// CUTLASS 4.2 and 4.5, so naming the modes positionally does not compile against both. Every
/// operand here is 2D with leading dimension \p ld, and the batch mode is never read because the
/// grouped GEMM runs each expert at L == 1, so filling all dynamic modes alike is exact.
template <class Stride>
CUTE_HOST_DEVICE constexpr Stride make_ld_stride(int64_t ld) {
  return cute::transform_leaf(Stride{}, [&](auto mode) {
    if constexpr (cute::is_static_v<decltype(mode)>) {
      return mode;
    } else {
      return static_cast<decltype(mode)>(ld);
    }
  });
}

// Fills the per-expert pointer and stride arrays the grouped GEMM indexes by group id. Activations
// and output are token-major over the expert-sorted rows, so they advance by expert_offsets; the
// weights and their scales are per-expert blocks. The weight block is addressed in bytes, two E2M1
// codes each.
__global__ void fill_group_arguments(
    int32_t const* __restrict__ expert_offsets,
    MmaType const* a_base,
    uint8_t const* b_base,
    ElementD* d_base,
    ElementScale const* s_base,
    MmaType const** a_ptrs,
    QuantType const** b_ptrs,
    ElementD** d_ptrs,
    ElementScalePacked const** s_ptrs,
    StrideA* a_strides,
    StrideB* b_strides,
    StrideD* d_strides,
    StrideS* s_strides,
    int num_experts,
    int64_t n,
    int64_t k) {
  int const e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_experts) return;

  int64_t const row = expert_offsets[e];
  a_ptrs[e] = a_base + row * k;
  b_ptrs[e] = reinterpret_cast<QuantType const*>(b_base + int64_t(e) * n * k / 2);
  d_ptrs[e] = d_base + row * n;
  s_ptrs[e] = reinterpret_cast<ElementScalePacked const*>(s_base + int64_t(e) * n * (k / ScaleGroupSize));

  a_strides[e] = make_ld_stride<StrideA>(k);
  b_strides[e] = make_ld_stride<StrideB>(k);
  d_strides[e] = make_ld_stride<StrideD>(n);
  // One row of the packed scale operand holds `n` ElementScalePacked, one per output element.
  s_strides[e] = make_ld_stride<StrideS>(n);
}

// Bump allocator over one workspace blob; every section starts 16B-aligned.
struct WorkspaceCursor {
  uint8_t* base = nullptr;
  size_t offset = 0;

  template <typename T>
  T* take(size_t count) {
    offset = (offset + 15) & ~size_t(15);
    T* out = base == nullptr ? nullptr : reinterpret_cast<T*>(base + offset);
    offset += sizeof(T) * count;
    return out;
  }
};

// Pointer and stride arrays for `num_experts` groups, sized without allocating. The take sequence
// must mirror the one below exactly: each take re-aligns to 16B, so folding the four pointer
// arrays into one under-sizes the blob whenever `num_experts` is odd.
inline size_t group_arguments_bytes(int num_experts) {
  WorkspaceCursor cursor;
  cursor.take<void*>(num_experts);
  cursor.take<void*>(num_experts);
  cursor.take<void*>(num_experts);
  cursor.take<void*>(num_experts);
  cursor.take<StrideA>(num_experts);
  cursor.take<StrideB>(num_experts);
  cursor.take<StrideD>(num_experts);
  cursor.take<StrideS>(num_experts);
  return cursor.offset;
}

// Everything `launch_grouped_gemm` needs that is independent of the token tile.
struct GroupedGemmArgs {
  int num_experts = 0;
  ProblemShape::UnderlyingProblemShape* problem_sizes = nullptr;
  MmaType const** a_ptrs = nullptr;
  QuantType const** b_ptrs = nullptr;
  ElementD** d_ptrs = nullptr;
  ElementScalePacked const** s_ptrs = nullptr;
  StrideA* a_strides = nullptr;
  StrideB* b_strides = nullptr;
  StrideD* d_strides = nullptr;
  StrideS* s_strides = nullptr;
  DLDevice device{};
};

template <int kTileTokens>
void launch_grouped_gemm(GroupedGemmArgs const& g, cudaStream_t stream) {
  using Gemm = typename TileTraits<kTileTokens>::Gemm;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = g.device.device_id;
  hw_info.sm_count = host::runtime::get_sm_count(hw_info.device_id);

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.dBeta = {_0{}, _0{}, 0};

  // The collective only feeds chunk_size to can_implement (it must divide the K tile); the scale
  // group size that the mainloop actually reads is the template parameter.
  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {g.num_experts, g.problem_sizes, nullptr},
      {g.b_ptrs, g.b_strides, g.a_ptrs, g.a_strides, g.s_ptrs, g.s_strides, kTileK},
      {fusion_args, nullptr, nullptr, g.d_ptrs, g.d_strides},
      hw_info};

  Gemm gemm;
  size_t const workspace_bytes = Gemm::get_workspace_size(arguments);
  auto workspace = host::ffi::alloc_workspace_tensor(workspace_bytes, g.device);
  void* workspace_ptr = workspace_bytes == 0 ? nullptr : workspace.data_ptr();

  MXFP4_A16_CUTLASS_CHECK(gemm.can_implement(arguments));
  MXFP4_A16_CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr, stream));
  MXFP4_A16_CUTLASS_CHECK(gemm.run(stream));
}

}  // namespace mxfp4_a16_sm90

/**
 * \brief Grouped GEMM over MoE experts: bf16 activations x MXFP4 weights, bf16 output.
 *
 * \param out [m_total, n] bf16, rows sorted by expert.
 * \param a [m_total, k] bf16, rows sorted by expert.
 * \param b_q [num_experts, n, k / 2] uint8; two E2M1 codes per byte, low nibble is the even k.
 * \param b_scales [num_experts, k / (32 * PackedScalesNum), n * PackedScalesNum] uint8 E8M0, biased
 *                 by +126 offline, interleaved so one row holds PackedScalesNum groups per output.
 * \param expert_offsets [num_experts] int32, first row of each expert in `a` / `out`.
 * \param problem_sizes [num_experts, 3] int32 per-expert (n, tokens, k).
 */
void mxfp4_a16_moe_mm_sm90(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b_q,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::TensorView expert_offsets,
    tvm::ffi::TensorView problem_sizes) {
  using namespace mxfp4_a16_sm90;

  RuntimeCheck(a.device().device_type == kDLCUDA, "activations must be a CUDA tensor");
  RuntimeCheck(is_type<bf16_t>(a.dtype()), "activations must be bf16");
  RuntimeCheck(is_type<bf16_t>(out.dtype()), "output must be bf16");
  RuntimeCheck(is_type<uint8_t>(b_q.dtype()), "MXFP4 weights must be packed in uint8");
  RuntimeCheck(is_type<uint8_t>(b_scales.dtype()), "MXFP4 scales must be E8M0 bytes");
  RuntimeCheck(
      is_type<int32_t>(expert_offsets.dtype()) && is_type<int32_t>(problem_sizes.dtype()), "int32 metadata expected");
  RuntimeCheck(b_q.dim() == 3 && b_scales.dim() == 3, "weights and scales are per-expert 3-D blocks");
  RuntimeCheck(a.dim() == 2 && out.dim() == 2, "activations and output are 2-D");
  RuntimeCheck(a.stride(1) == 1 && out.stride(1) == 1, "activations and output must be row major");

  int const num_experts = static_cast<int>(b_q.size(0));
  int64_t const n = b_q.size(1);
  int64_t const k = a.size(1);
  constexpr int kScaleTileK = ScaleGroupSize * PackedScalesNum;
  static_assert(kScaleTileK == kTileK, "one K tile must hold exactly PackedScalesNum scale groups");

  RuntimeCheck(b_q.size(2) == k / 2, "weight K must match the activation K");
  RuntimeCheck(out.size(1) == n && out.size(0) == a.size(0), "output shape must be [m_total, n]");
  RuntimeCheck(k % kTileK == 0, "K must be a multiple of the K tile ", kTileK, ", got ", k);
  // fill_group_arguments derives every operand's leading dimension from n / k alone, so a padded
  // row pitch would silently offset the expert's rows.
  RuntimeCheck(a.stride(0) == k, "activations must be tightly packed, got row stride ", a.stride(0));
  RuntimeCheck(out.stride(0) == n, "output must be tightly packed, got row stride ", out.stride(0));
  RuntimeCheck(expert_offsets.size(0) == num_experts, "one offset per expert");
  RuntimeCheck(problem_sizes.size(0) == num_experts && problem_sizes.size(1) == 3, "problem sizes are [E, 3]");
  RuntimeCheck(
      b_scales.size(0) == num_experts && b_scales.size(1) == k / kScaleTileK && b_scales.size(2) == n * PackedScalesNum,
      "scales must be [E, K/",
      kScaleTileK,
      ", N*",
      PackedScalesNum,
      "]");

  cudaStream_t const stream = LaunchKernel::resolve_device(a.device());

  auto group_args = host::ffi::alloc_workspace_tensor(group_arguments_bytes(num_experts), a.device());
  WorkspaceCursor cursor{static_cast<uint8_t*>(group_args.data_ptr()), 0};
  auto** a_ptrs = cursor.take<MmaType const*>(num_experts);
  auto** b_ptrs = cursor.take<QuantType const*>(num_experts);
  auto** d_ptrs = cursor.take<ElementD*>(num_experts);
  auto** s_ptrs = cursor.take<ElementScalePacked const*>(num_experts);
  auto* a_strides = cursor.take<StrideA>(num_experts);
  auto* b_strides = cursor.take<StrideB>(num_experts);
  auto* d_strides = cursor.take<StrideD>(num_experts);
  auto* s_strides = cursor.take<StrideS>(num_experts);

  constexpr int kFillBlock = 128;
  LaunchKernel((num_experts + kFillBlock - 1) / kFillBlock, kFillBlock, stream)(
      fill_group_arguments,
      static_cast<int32_t const*>(expert_offsets.data_ptr()),
      static_cast<MmaType const*>(a.data_ptr()),
      static_cast<uint8_t const*>(b_q.data_ptr()),
      static_cast<ElementD*>(out.data_ptr()),
      static_cast<ElementScale const*>(b_scales.data_ptr()),
      a_ptrs,
      b_ptrs,
      d_ptrs,
      s_ptrs,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      num_experts,
      n,
      k);

  GroupedGemmArgs group{
      num_experts,
      static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr()),
      a_ptrs,
      b_ptrs,
      d_ptrs,
      s_ptrs,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      a.device()};

  // A token tile wider than the expert's row count spends the difference on padded mma, and at
  // decode there are only a few rows per expert; a narrower tile than the row count instead re-reads
  // the expert's weights once per tile, so round the average up to the next width.
  int64_t const rows_per_expert = (a.size(0) + num_experts - 1) / num_experts;
  if (rows_per_expert <= 16) {
    launch_grouped_gemm<16>(group, stream);
  } else if (rows_per_expert <= 32) {
    launch_grouped_gemm<32>(group, stream);
  } else if (rows_per_expert <= 64) {
    launch_grouped_gemm<64>(group, stream);
  } else {
    launch_grouped_gemm<128>(group, stream);
  }
}

}  // namespace sglang
