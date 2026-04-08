// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cutlass/cutlass.h>

#include <cutlass/gemm/collective/collective_builder.hpp>

#include "kda/sm90/collective/common.hpp"
#include "kda/sm90/collective/load_predicated.hpp"
#include "kda/sm90/collective/load_tma.hpp"
#include "kda/sm90/collective/named_barriers.hpp"
#include "kda/sm90/collective/store_tma.hpp"
#include "kda/sm90/kernel/options.hpp"
#include "kda/sm90/utils/debug.hpp"
#include "kda/sm90/utils/math_order_barrier.hpp"
#include "kda/sm90/utils/unused.hpp"
#include "kerutils/kerutils.cuh"

// #define INLINE_LAMBDA [[gnu::always_inline]]
#define INLINE_LAMBDA __attribute__((always_inline))
// #define INLINE_LAMBDA [[msvc::forceinline]]

#define WORKAROUND_WGMMA_PERFORMANCE_LOSS() \
  if (thread_idx > 8192) {                  \
    __syncwarp();                           \
  }

namespace kda::sm90::collective {

struct KdaNamedBarriers : FlatSharedNamedBarriers {
  static constexpr int StateMath = FlatSharedNamedBarriers::NumBarriersUsed + 0;
  static constexpr int AuxMath = FlatSharedNamedBarriers::NumBarriersUsed + 1;
  static constexpr int StateMathWG0 = FlatSharedNamedBarriers::NumBarriersUsed + 2;
  // NOTE: only for debug
  // used for subchunk MMA with two groups, each group has 2 warps
  // static constexpr int AuxMathWarp0 = FlatSharedNamedBarriers::NumBarriersUsed + 3;
  // static constexpr int AuxMathWarp1 = FlatSharedNamedBarriers::NumBarriersUsed + 4;
};

using ku::alignment_for_swizzle;
using ku::select_layout;
using ku::select_tensor;
using namespace cute;
using kda::sm90::kernel::find_option_t;
using kda::sm90::kernel::Tag;

template <
    class Element_,
    class ElementAccumulatorQK_,
    class ElementAccumulatorKV_,
    class TileShape_,  // (seqlen_q, seqlen_kv, d)
    class LayoutQ_,
    class LayoutK_,
    class LayoutV_,
    class LayoutO_,  // (seqlen_q/k, d, h)
    class Options>
struct FlatMainloopTmaWarpSpecializedKdaFwd {
  using Element = Element_;
  using ElementAccumulatorQK = ElementAccumulatorQK_;
  using ElementAccumulatorO = ElementAccumulatorQK;
  using ElementAccumulatorKV = ElementAccumulatorKV_;
  using ElementO = Element;
  using ElementAlpha = float;
  // TODO: support bf16 beta
  using ElementBeta = float;
  using ElementGatedMMA = cutlass::tfloat32_t;

  using TileShape = TileShape_;

  using LayoutQ = LayoutQ_;      // (seqlen_q, d, h)
  using LayoutK = LayoutK_;      // (seqlen_k, d, h)
  using LayoutV = LayoutV_;      // (seqlen_k, d, h)
  using LayoutO = LayoutO_;      // (seqlen_k, d, h)
  using LayoutAlpha = LayoutQ_;  // (seqlen_q, d, h)

  // Options
  static constexpr bool kIsPersistent = find_option_t<Tag::kIsPersistent, false_type, Options>::value;

  static constexpr bool kInitStateFromInput = find_option_t<Tag::kInitStateFromInput, false_type, Options>::value;

  static constexpr int NumLoadWarpGroups = 1;
  static constexpr int NumStateMmaWarpGroups = 2;
  static constexpr int NumAuxMmaWarpGroups = 1;

  static constexpr int StageCountQ = find_option_t<Tag::kStagesQ, Int<2>, Options>::value;
  static constexpr int StageCountK = find_option_t<Tag::kStagesK, Int<2>, Options>::value;
  static constexpr int StageCountV = find_option_t<Tag::kStagesV, Int<1>, Options>::value;

  static constexpr int NeedsAlpha = find_option_t<Tag::kNeedsAlpha, cute::true_type, Options>::value;
  static constexpr int NeedsBeta = find_option_t<Tag::kNeedsBeta, cute::true_type, Options>::value;
  static_assert(NeedsAlpha && NeedsBeta, "Alpha and Beta are both used in KDA.");

  static constexpr int SafeGate = true;  // only support safe_gate=true

  static constexpr int NumLoadThreads = NumLoadWarpGroups * 128;
  static constexpr int NumStateMmaThreads = NumStateMmaWarpGroups * 128;
  static constexpr int NumAuxMmaThreads = NumAuxMmaWarpGroups * 128;

  static constexpr uint32_t OrderedBarrierId0 = uint32_t(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
  static constexpr uint32_t OrderedBarrierId1 = uint32_t(cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

  using OrderedMathBarriers = std::conditional_t<
      NumStateMmaWarpGroups == 2,
      OrderedNamedBarriers</*UseReservedNB=*/true, OrderedBarrierId0, OrderedBarrierId1>,
      OrderedNamedBarriers</*UseReservedNB=*/true, OrderedBarrierId0>>;

  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using StagesK = cutlass::gemm::collective::StageCount<StageCountK>;
  using StagesV = cutlass::gemm::collective::StageCount<StageCountV>;
  using StagesQ_K_Scaled = cutlass::gemm::collective::StageCount<2>;
  using StagesO = cutlass::gemm::collective::StageCount<1>;
  using ClusterShape = Shape<_1, _1, _1>;

  using StagesQK = cutlass::gemm::collective::StageCount<2>;
  using StagesKK = cutlass::gemm::collective::StageCount<2>;

  using StagesAlpha = cutlass::gemm::collective::StageCount<2>;
  using StagesBeta = cutlass::gemm::collective::StageCount<2>;

  static constexpr int Alignment = 16 / sizeof(Element);

  static constexpr auto BlkSeqQ = get<0>(TileShape{});   // Blk_Q
  static constexpr auto BlkSeqKV = get<1>(TileShape{});  // Blk_K/V
  static constexpr auto HeadSize = get<2>(TileShape{});  // D (Dq, Dk, Dv all equal)
  static constexpr auto HeadSizeQK = HeadSize;
  static constexpr auto HeadSizeV = HeadSize;
  using HeadSizeHalf = _64;
  using HeadSizeQuar = _32;

  using TileShapeQK = decltype(make_shape(BlkSeqQ, BlkSeqKV, HeadSizeQK));
  // used for element-wise in compute_aux prologue, to reduce register usage
  using TileShapeQK_Half = decltype(make_shape(BlkSeqQ, BlkSeqKV, HeadSizeHalf{}));
  using TileShapeQK_Quar = decltype(make_shape(BlkSeqQ, BlkSeqKV, HeadSizeQuar{}));
  using TileShapeKK = decltype(make_shape(BlkSeqKV, BlkSeqKV, HeadSizeQK));
  using TileShapeKV = decltype(make_shape(HeadSizeV, HeadSizeQK, BlkSeqKV));
  static_assert(std::is_same_v<TileShapeQK, TileShapeKK>);

  using TileShapeO2 = decltype(make_shape(HeadSizeV, BlkSeqQ, BlkSeqKV));
  using TileShapeO1 = decltype(make_shape(HeadSizeV, BlkSeqQ, HeadSizeQK));

  static_assert(BlkSeqQ % 64 == 0);
  static_assert(BlkSeqQ == 64 || BlkSeqQ == 128);
  static_assert(BlkSeqQ == BlkSeqKV);
  static constexpr bool IsQKCooperative = BlkSeqQ == 128;
  static constexpr bool IsKKCooperative = IsQKCooperative;

  using DummyStages = cutlass::gemm::collective::StageCount<2>;
  ;
  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      LayoutQ,
      Alignment,
      Element,
      LayoutK,
      Alignment,
      ElementAccumulatorQK,
      TileShapeQK,
      ClusterShape,
      DummyStages,
      std::conditional_t<
          IsQKCooperative,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative,
          cutlass::gemm::KernelTmaWarpSpecialized>>::CollectiveOp;

  // dummy TiledMmaQK RS for S2R/R2S layout consistency
  using AtomLayoutQK = Layout<Shape<Int<BlkSeqQ / 64>, _1, _1>>;
  using TiledMmaQK_RS = decltype(make_tiled_mma(
      decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccumulatorQK, TileShapeQK>()){}, AtomLayoutQK{}));
  static_assert(size(TiledMmaQK_RS{}) == NumAuxMmaThreads);
  using TiledMmaQK_RS_Quar = decltype(make_tiled_mma(
      decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccumulatorQK, TileShapeQK_Quar>()){},
      AtomLayoutQK{}));
  static_assert(size(TiledMmaQK_RS_Quar{}) == NumAuxMmaThreads);

  using CollectiveMmaKV_G2S = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      decltype(select<1, 0, 2>(LayoutV{})),
      Alignment,  // direct TMA copy for GMEM -> SMEM
      Element,
      decltype(select<1, 0, 2>(LayoutK{})),
      Alignment,
      ElementAccumulatorKV,
      TileShapeKV,
      ClusterShape,
      DummyStages,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using SmemLayoutAlphaAtom = GMMA::Layout_K_SW128_Atom<ElementAlpha>;
  using SmemLayoutAlpha_SD = decltype(tile_to_shape(
      SmemLayoutAlphaAtom{},
      make_shape(
          shape<1>(TileShapeQK{}),
          shape<2>(TileShapeQK{}),
          Int<StagesAlpha::value>{})));                     // (blk_kv, head_size), (64, 128)
  using GmemShapeAlpha = Shape<int64_t, int32_t, int32_t>;  // (seqlen_k, d, h)
  using GmemStrideAlpha = Stride<int64_t, _1, int32_t>;
  using GmemLayoutAlpha = Layout<GmemShapeAlpha, GmemStrideAlpha>;
  using GmemTiledCopyAlpha = cute::SM90_TMA_LOAD;
  using TMA_Alpha = decltype(make_tma_copy(
      GmemTiledCopyAlpha{},
      make_tensor(make_gmem_ptr(static_cast<float const*>(nullptr)), GmemLayoutAlpha{}),
      take<0, 2>(SmemLayoutAlpha_SD{}),
      select<1, 2>(TileShapeQK{}),
      size<0>(ClusterShape{})));

  // raw layout for copy
  using SmemLayoutQ_SD = decltype(unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StagesQ::value>{}));
  using SmemLayoutQ_K_Scaled_SD =
      decltype(unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StagesQ_K_Scaled::value>{}));
  using SmemLayoutK_DS =
      decltype(unstage_smem_layout(typename CollectiveMmaKV_G2S::SmemLayoutB{}, Int<StagesK::value>{}));
  using SmemLayoutQ_K_Scaled_DS =
      decltype(unstage_smem_layout(typename CollectiveMmaKV_G2S::SmemLayoutB{}, Int<StagesQ_K_Scaled::value>{}));
  using SmemLayoutV_DS =
      decltype(unstage_smem_layout(typename CollectiveMmaKV_G2S::SmemLayoutA{}, Int<StagesV::value>{}));

  // Layout for V^T
  using RefLayoutV = decltype(make_layout(select<0, 2>(TileShapeKV{}), LayoutRight{}));
  using CollectiveMmaKV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      RefLayoutV,
      Alignment,  // needs a S2R transposition for MMA
      Element,
      decltype(select<1, 0, 2>(LayoutK{})),
      Alignment,
      ElementAccumulatorKV,
      TileShapeKV,
      ClusterShape,
      DummyStages,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using RefLayoutKV = decltype(make_layout(select<0, 1>(TileShapeKV{}), LayoutRight{}));  // (dv, dk)
  using CollectiveMmaO1 = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      RefLayoutKV,
      Alignment,  // NOTE: S (KV) as operand A
      Element,
      LayoutQ,
      Alignment,
      ElementAccumulatorO,
      TileShapeO1,
      ClusterShape,
      DummyStages,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  // (blk_q,blk_k) to align with O2 mma, LayoutRight to align with QK mma output
  using DesiredLayoutQK = decltype(make_layout(select<0, 1>(TileShapeQK{}), LayoutRight{}));
  using CollectiveMmaO2 = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      RefLayoutV,
      Alignment,  // V^T
      Element,
      DesiredLayoutQK,
      Alignment,
      ElementAccumulatorO,
      TileShapeO2,
      ClusterShape,
      DummyStages,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;  // Q@K^t
  using TiledMmaKV = decltype(convert_to_gmma_rs(typename CollectiveMmaKV::TiledMma{}));
  using TiledMmaO1 = decltype(convert_to_gmma_rs(typename CollectiveMmaO1::TiledMma{}));
  using TiledMmaO2 = decltype(convert_to_gmma_rs(typename CollectiveMmaO2::TiledMma{}));

  static_assert(size(TiledMmaQK{}) == NumAuxMmaThreads);

  static_assert(size(TiledMmaKV{}) == NumStateMmaThreads);
  static_assert(size(TiledMmaO1{}) == NumStateMmaThreads);
  static_assert(size(TiledMmaO2{}) == NumStateMmaThreads);

  using CollectiveStoreO = CollectiveStoreTma<
      TileShapeO1,
      ClusterShape,
      ElementO,
      ElementAccumulatorO,
      /*Seme*/ ElementO,
      decltype(select<1, 0, 2>(LayoutO{})),
      StagesO::value>;

  // layout for compute
  using QKSmemLayoutQ = SmemLayoutQ_SD;
  using QKSmemLayoutK = decltype(select_layout<1, 0, 2>(SmemLayoutK_DS{}));
  using QKScaledSmemLayoutQ = SmemLayoutQ_K_Scaled_SD;

  using KVSmemLayoutK = SmemLayoutK_DS;
  using KVSmemLayoutV = SmemLayoutV_DS;
  using QKScaledSmemLayoutKt = SmemLayoutQ_K_Scaled_DS;

  using QKQSmemLayoutAlpha = SmemLayoutAlpha_SD;
  using QKKSmemLayoutAlpha = decltype(select_layout<1, 0, 2>(SmemLayoutAlpha_SD{}));
  using KKTKSmemLayoutAlpha = SmemLayoutAlpha_SD;
  using KKTKTSmemLayoutAlpha = decltype(select_layout<1, 0, 2>(SmemLayoutAlpha_SD{}));

  // layout for compute output
  using SmemLayoutQK = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<Element>{},
      flatten(make_shape(select<0, 1>(TileShapeQK{}), Int<StagesQK::value>{})),
      Step<_1, _2, _3>{}));
  using SmemLayoutO = typename CollectiveStoreO::SmemLayoutO;

  using SmemLayoutKK = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<Element>{},
      flatten(make_shape(select<0, 1>(TileShapeQK{}), Int<StagesQK::value>{})),
      Step<_1, _2, _3>{}));

  using InverseType = cutlass::half_t;
  using CollectiveInverse = ku::CollectiveInverse<InverseType, true, false>;

  using ElementAccumulatorSK = float;
  using TileShapeSK = decltype(make_shape(HeadSizeV, BlkSeqKV, HeadSizeQK));
  using CollectiveMmaSK = typename cutlass::gemm::collective::CollectiveBuilder<  // basically the same as O1
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        RefLayoutKV,
        Alignment,
        Element,
        LayoutK,
        Alignment,
        ElementAccumulatorSK,
        TileShapeSK,
        ClusterShape,
        DummyStages,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using ElementAccumulatorNewV = float;
  using TileShapeNewV = decltype(make_shape(HeadSizeV, BlkSeqKV, BlkSeqKV));
  using RefLayoutSK = decltype(make_layout(select<0, 2>(TileShapeNewV{}), LayoutRight{}));      // (dv, Blk)
  using DesiredLayoutKK = decltype(make_layout(select<1, 2>(TileShapeNewV{}), LayoutRight{}));  //
  using CollectiveMmaNewV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      RefLayoutSK,
      Alignment,
      Element,
      DesiredLayoutKK,
      Alignment,
      ElementAccumulatorKV,
      TileShapeNewV,
      ClusterShape,
      DummyStages,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  // FIXME: K@K^t are not exactly the same as Q@K^t, but similar enough (what does this mean??)
  using TiledMmaKK = typename CollectiveMmaQK::TiledMma;  // T = inv(I + strict_lower_triangular(K@K^t))
  using TiledMmaSK = decltype(convert_to_gmma_rs(typename CollectiveMmaSK::TiledMma{}));      // ??   = -S@K^t + V^t
  using TiledMmaNewV = decltype(convert_to_gmma_rs(typename CollectiveMmaNewV::TiledMma{}));  // NewV = ??@T^t

  static_assert(size(TiledMmaKK{}) == NumAuxMmaThreads);

  using GmemStrideBeta = Stride<int64_t, int32_t>;
  using GmemLayoutBeta = Layout<Shape<int64_t, int32_t>, GmemStrideBeta>;  // (seq, head)

  // only store the last row in Alpha
  using SmemLayoutAlphaLast = decltype(make_layout(make_shape(HeadSize, Int<StagesAlpha::value>{})));
  using SmemLayoutBeta = decltype(make_layout(make_shape(BlkSeqQ, Int<StagesBeta::value>{})));

  using MainloopQPipeline = cutlass::PipelineTmaAsync<StagesQ::value>;
  using MainloopKPipeline = cutlass::PipelineTmaAsync<StagesK::value>;
  using MainloopVPipeline = cutlass::PipelineTmaAsync<StagesV::value>;
  using MainloopAlphaPipeline = std::conditional_t<NeedsAlpha, cutlass::PipelineTmaAsync<StagesAlpha::value>, Unused>;
  using MainloopOPipeline = typename CollectiveStoreO::Pipeline;

  using MainloopQKPipeline = cutlass::PipelineAsync<StagesQK::value>;
  using MainloopKKPipeline = cutlass::PipelineAsync<StagesKK::value>;

  using MainloopAlphaLastPipeline = std::conditional_t<NeedsAlpha, cutlass::PipelineAsync<StagesAlpha::value>, Unused>;

  using MainloopBetaPipeline = std::conditional_t<NeedsBeta, cutlass::PipelineAsync<StagesBeta::value>, Unused>;

  using QPipelineState = typename cutlass::PipelineState<MainloopQPipeline::Stages>;
  using KPipelineState = typename cutlass::PipelineState<MainloopKPipeline::Stages>;
  using VPipelineState = typename cutlass::PipelineState<MainloopVPipeline::Stages>;
  using OPipelineState = typename CollectiveStoreO::PipelineState;

  using QKPipelineState = cutlass::PipelineState<MainloopQKPipeline::Stages>;
  using KKPipelineState = cutlass::PipelineState<MainloopKKPipeline::Stages>;

  using AlphaLastPipelineState =
      std::conditional_t<NeedsAlpha, cutlass::PipelineState<MainloopAlphaLastPipeline::Stages>, Unused>;

  using AlphaPipelineState =
      std::conditional_t<NeedsAlpha, cutlass::PipelineState<MainloopAlphaPipeline::Stages>, Unused>;
  using BetaPipelineState = std::conditional_t<NeedsBeta, cutlass::PipelineState<MainloopBetaPipeline::Stages>, Unused>;

  using BetaProcessor = Unused;

  static constexpr int LoadQBytes = size(QKSmemLayoutQ{}(_, _, _0{})) * sizeof(Element);
  static constexpr int LoadKBytes = size(KVSmemLayoutK{}(_, _, _0{})) * sizeof(Element);
  static constexpr int LoadVBytes = size(KVSmemLayoutV{}(_, _, _0{})) * sizeof(Element);
  static constexpr int LoadAlphaBytes = size(QKQSmemLayoutAlpha{}(_, _, _0{})) * sizeof(ElementAlpha);
  static constexpr int StoreOBytes = CollectiveStoreO::TmaTransactionBytes;

  using SharedStorageO = typename CollectiveStoreO::SharedStorage;

  struct SharedStorage {
    alignas(alignment_for_swizzle(QKSmemLayoutQ{})) cute::array_aligned<Element, cute::cosize_v<QKSmemLayoutQ>> smem_q;
    alignas(alignment_for_swizzle(KVSmemLayoutK{})) cute::array_aligned<Element, cute::cosize_v<KVSmemLayoutK>> smem_k;
    alignas(alignment_for_swizzle(KVSmemLayoutV{})) cute::array_aligned<Element, cute::cosize_v<KVSmemLayoutV>> smem_v;
    alignas(alignment_for_swizzle(
        QKQSmemLayoutAlpha{})) cute::array_aligned<ElementAlpha, cute::cosize_v<QKQSmemLayoutAlpha>> smem_alpha;
    alignas(alignment_for_swizzle(SmemLayoutQK{})) cute::array_aligned<Element, cute::cosize_v<SmemLayoutQK>> smem_qk;
    alignas(
        alignment_for_swizzle(SmemLayoutKK{})) cute::array_aligned<InverseType, cute::cosize_v<SmemLayoutKK>> smem_kk;
    // smemq_k_scaled for exp(alpha) * Q and exp(alpha) * K in QS and KS, computed in Math WG2/3
    alignas(alignment_for_swizzle(
        QKScaledSmemLayoutQ{})) cute::array_aligned<Element, cute::cosize_v<QKScaledSmemLayoutQ>> smem_q_k_scaled;

    SharedStorageO smem_o;

    cute::array_aligned<ElementBeta, cute::cosize_v<SmemLayoutBeta>> smem_beta;
    // store last row in Alpha separately, used for S'=K^T NewV's epilogue and S+=decay(S') (one fused epilogue)
    cute::array_aligned<ElementAlpha, cute::cosize_v<SmemLayoutAlphaLast>> smem_alpha_last;
  };

  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaKV_G2S::Params::TMA_B;
  using TMA_V = typename CollectiveMmaKV_G2S::Params::TMA_A;
  using TMA_O = typename CollectiveStoreO::Params::TMA_O;

  using LoadQ = CollectiveLoadTma<LoadKind::kQ, MainloopQPipeline, Element, QKSmemLayoutQ, TMA_Q>;
  using LoadK = CollectiveLoadTma<LoadKind::kK, MainloopKPipeline, Element, KVSmemLayoutK, TMA_K>;
  using LoadV = CollectiveLoadTma<LoadKind::kV, MainloopVPipeline, Element, KVSmemLayoutV, TMA_V>;
  using LoadAlpha =
      CollectiveLoadTma<LoadKind::kAlpha, MainloopAlphaPipeline, ElementAlpha, QKQSmemLayoutAlpha, TMA_Alpha>;

  using LoadBeta = CollectiveLoadVector<
      LoadKindVector::kBeta,
      MainloopBetaPipeline,
      ElementBeta,
      GmemLayoutBeta,
      ElementBeta,
      SmemLayoutBeta,
      BetaProcessor>;

  struct Arguments {  // clang-format off
    Element const* ptr_Q; LayoutQ dQ;
    Element const* ptr_K; LayoutK dK;
    Element const* ptr_V; LayoutV dV;
    Element*       ptr_O; LayoutO dO;
    float   const* ptr_Alpha; LayoutAlpha dAlpha;
    float*        ptr_output_state; // layout fixed (kdim, vdim, num_heads, num_seqs):LayoutLeft{}
    float const*  ptr_input_state;
    float scale;
    ElementBeta const* beta_ptr;  GmemStrideBeta beta_stride;
  };  // clang-format on

  struct Params {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;
    TMA_Alpha tma_load_alpha;
    TMA_O tma_store_o;
    void* tensormaps;
    float scale;

    float* ptr_output_state;
    float const* ptr_input_state;

    ElementBeta const* beta_ptr;
    GmemLayoutBeta beta_layout;
  };

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_size, Arguments const& args) {
    return true && (problem_size.head_size <= get<2>(TileShape{})) && ((problem_size.head_size % Alignment) == 0);
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_size, Arguments const& args, void* workspace) {
    int64_t s = problem_size.total_seqlen;
    int64_t t = problem_size.total_seqlen;
    int32_t d = problem_size.head_size;

    auto params_qk = CollectiveMmaQK::to_underlying_arguments(
        make_shape(s, t, d, problem_size.num_heads),
        typename CollectiveMmaQK::Arguments{
            args.ptr_Q, args.dQ, args.ptr_K, args.dK,  // never used, dummy
        },
        /*workspace=*/nullptr);

    auto params_kv_k = CollectiveMmaKV_G2S::to_underlying_arguments(
        make_shape(d, d, s, problem_size.num_heads),
        typename CollectiveMmaKV_G2S::Arguments{
            args.ptr_V,
            select<1, 0, 2>(args.dV),  // not used
            args.ptr_K,
            select<1, 0, 2>(args.dK),  // used as G2S for K
        },
        /*workspace=*/nullptr);

    auto alpha_shape = make_shape(s, d, problem_size.num_heads);
    auto alpha_stride = make_stride(
        get<0>(args.dAlpha),  // seqlen stride
        get<1>(args.dAlpha),  // head_dim stride
        get<2>(args.dAlpha)   // head stride
    );
    Tensor mAlpha = make_tensor(make_gmem_ptr(args.ptr_Alpha), make_layout(alpha_shape, alpha_stride));
    TMA_Alpha tma_load_alpha = make_tma_copy(
        GmemTiledCopyAlpha{},
        mAlpha,
        take<0, 2>(SmemLayoutAlpha_SD{}),
        select<1, 2>(TileShapeQK{}),
        size<0>(ClusterShape{}));

    auto params_kv_v = CollectiveMmaKV_G2S::to_underlying_arguments(
        make_shape(d, d, s, problem_size.num_heads),
        typename CollectiveMmaKV_G2S::Arguments{
            args.ptr_V,
            select<1, 0, 2>(args.dV),  // used as G2S for V
            args.ptr_K,
            select<1, 0, 2>(args.dK),  // not used
        },
        /*workspace=*/nullptr);

    auto params_o = CollectiveStoreO::to_underlying_arguments(
        make_shape(d, s, d, problem_size.num_heads),  // in O1
        // make_shape(d, s, s, problem_size.num_heads),  // in O2
        typename CollectiveStoreO::Arguments{args.ptr_O, select<1, 0, 2>(args.dO), workspace},
        workspace);

    return Params{
        .tma_load_q = params_qk.tma_load_a,
        .tma_load_k = params_kv_k.tma_load_b,
        .tma_load_v = params_kv_v.tma_load_a,
        .tma_load_alpha = tma_load_alpha,
        .tma_store_o = params_o.tma_store_o,
        .tensormaps = params_o.tensormaps,
        .scale = args.scale,

        .ptr_output_state = args.ptr_output_state,
        .ptr_input_state = args.ptr_input_state,

        // TODO: refactor all name to varname_vartype
        .beta_ptr = args.beta_ptr,
        .beta_layout = make_layout(make_shape(s, problem_size.num_heads), args.beta_stride),
    };
  }

  static size_t get_workspace_size(Arguments const& args, int sm_count) {
    return CollectiveStoreO::get_workspace_size(sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream) {
    return CollectiveStoreO::initialize_workspace(problem_shape, workspace, stream);
  }

  CUTE_DEVICE static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_alpha.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
  }

  template <typename ProblemShape, typename LoadTileShape, typename WorkDesc>
  CUTE_DEVICE void load_qkv(
      Params const& params,
      ProblemShape const& problem_size,
      LoadTileShape const& load_tile_shape,
      WorkDesc const& work_desc,
      MainloopQPipeline& q_pipeline,
      QPipelineState& q_smem_pipe_write,
      MainloopKPipeline& k_pipeline,
      KPipelineState& k_smem_pipe_write,
      MainloopVPipeline& v_pipeline,
      VPipelineState& v_smem_pipe_write,
      MainloopAlphaPipeline& alpha_pipeline,
      AlphaPipelineState& alpha_smem_pipe_write,
      SharedStorage& storage) {
    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    uint32_t lane_predicate = cute::elect_one_sync();

    auto q_collective_load = LoadQ(params.tma_load_q, q_pipeline, storage.smem_q);
    auto k_collective_load = LoadK(params.tma_load_k, k_pipeline, storage.smem_k);
    auto v_collective_load = LoadV(params.tma_load_v, v_pipeline, storage.smem_v);
    auto alpha_collective_load = LoadAlpha{params.tma_load_alpha, alpha_pipeline, storage.smem_alpha};

    auto q_src_dst = q_collective_load.partition_SD(problem_size, load_tile_shape, work_desc);
    auto k_src_dst = k_collective_load.partition_SD(problem_size, load_tile_shape, work_desc);
    auto v_src_dst = v_collective_load.partition_SD(problem_size, load_tile_shape, work_desc);
    auto alpha_src_dst = alpha_collective_load.partition_SD(problem_size, load_tile_shape, work_desc);

    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks; ++blk) {
      alpha_collective_load.step(alpha_src_dst, blk, alpha_smem_pipe_write, lane_predicate);
      q_collective_load.step(q_src_dst, blk, q_smem_pipe_write, lane_predicate);
      k_collective_load.step(k_src_dst, blk, k_smem_pipe_write, lane_predicate);
      v_collective_load.step(v_src_dst, blk, v_smem_pipe_write, lane_predicate);
    }
  }

  template <typename ProblemShape, typename TileShape, typename WorkDesc>
  CUTE_DEVICE void load_beta(
      Params const& params,
      ProblemShape const& problem_size,
      TileShape const& tile_shape,
      WorkDesc const& work_desc,
      MainloopBetaPipeline& pipeline,
      BetaPipelineState& smem_pipe_write,
      SharedStorage& storage) {
    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));

    // fuse post inverse diag(beta) into diagonal of IKK
    // auto collective_load = LoadBeta{params.beta_ptr, params.beta_layout, /*oob_value=*/1.0f, pipeline,
    // storage.smem_beta};
    auto collective_load =
        LoadBeta{params.beta_ptr, params.beta_layout, /*oob_value=*/0.0f, pipeline, storage.smem_beta};
    auto src_dst = collective_load.partition_SD(problem_size, tile_shape, work_desc);

    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks - 1; ++blk) {
      collective_load.step</*IsTail=*/false>(src_dst, blk, smem_pipe_write, num_blocks);
    }
    collective_load.step</*IsTail=*/true>(src_dst, num_blocks - 1, smem_pipe_write, num_blocks);
  }

  template <typename ProblemShape, typename TileShape, typename WorkDesc>
  CUTE_DEVICE void extract_alpha_last(
      Params const& params,
      ProblemShape const& problem_size,
      TileShape const& tile_shape,
      WorkDesc const& work_desc,
      MainloopAlphaPipeline& alpha_pipeline,
      AlphaPipelineState& alpha_smem_pipe_read,
      MainloopAlphaLastPipeline& alpha_last_pipeline,
      AlphaLastPipelineState& alpha_last_smem_pipe_write,
      SharedStorage& storage) {
    int thread_idx = threadIdx.x % cutlass::NumThreadsPerWarp;

    Tensor sAqkq = make_tensor(make_smem_ptr(storage.smem_alpha.data()), QKQSmemLayoutAlpha{});
    Tensor sAlast = make_tensor(make_smem_ptr(storage.smem_alpha_last.data()), SmemLayoutAlphaLast{});

    auto extract_loop_body = [&](int blk, auto is_final_block_) INLINE_LAMBDA {
      constexpr bool is_final_block = decltype(is_final_block_)::value;

      int B = is_final_block ? valid_seq_len(work_desc, blk) : BlkSeqKV;

      auto sAqkq_curr = sAqkq(_, _, alpha_smem_pipe_read.index());
      Tensor sAlast_out = sAlast(_, alpha_last_smem_pipe_write.index());

      alpha_pipeline.consumer_wait(alpha_smem_pipe_read);
      alpha_last_pipeline.producer_acquire(alpha_last_smem_pipe_write);

      // each thread copy 4 elements, total 128 elements with one warp
      CUTE_UNROLL
      for (int t = thread_idx; t < HeadSize; t += 32) {
        sAlast_out(t) = sAqkq_curr(B - 1, t);
      }

      cutlass::arch::fence_view_async_shared();
      alpha_last_pipeline.producer_commit(alpha_last_smem_pipe_write);
      ++alpha_last_smem_pipe_write;
      alpha_pipeline.consumer_release(alpha_smem_pipe_read);
      ++alpha_smem_pipe_read;
    };

    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks - 1; ++blk) {
      extract_loop_body(blk, /*is_final_block_=*/cute::false_type{});
    }
    extract_loop_body(num_blocks - 1, /*is_final_block_=*/cute::true_type{});
  }

  template <typename ProblemSize, typename StoreTileShape, typename WorkDesc, typename PipelineState>
  CUTE_DEVICE void store(
      TMA_O const& tma_store,
      void* tensormaps,
      ProblemSize const& problem_size,
      StoreTileShape const& store_tile_shape,
      WorkDesc const& work_desc,
      MainloopOPipeline& pipeline,
      PipelineState& smem_pipe_read,
      SharedStorageO& storage) {
    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    uint32_t lane_predicate = cute::elect_one_sync();

    auto collective_store = CollectiveStoreO{tma_store, pipeline, storage, tensormaps};
    auto src_dst = collective_store.partition_SD(problem_size, store_tile_shape, work_desc);

    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks; ++blk) {
      DPRINTF0_W(
          "O collective_store.step smem_pipe_read:%d -> blk_idx:%d, num_blocks:%d\n",
          smem_pipe_read.index(),
          blk,
          num_blocks);
      collective_store.step(problem_size, work_desc, src_dst, smem_pipe_read, blk, num_blocks, lane_predicate);
    }
  }

  template <class ProblemShape, class WorkDesc>
  CUTE_DEVICE void compute(
      Params const& params,
      ProblemShape const& problem_size,
      WorkDesc const& work_desc,
      MainloopQPipeline& q_pipeline,
      QPipelineState& q_smem_pipe_read,
      MainloopKPipeline& k_pipeline,
      KPipelineState& k_smem_pipe_read,
      MainloopVPipeline& v_pipeline,
      VPipelineState& v_smem_pipe_read,
      MainloopOPipeline& o_pipeline,
      OPipelineState& o_smem_pipe_write,
      MainloopQKPipeline& qk_pipeline,
      QKPipelineState& qk_smem_pipe_read,
      MainloopKKPipeline& kk_pipeline,
      KKPipelineState& kk_smem_pipe_read,
      MainloopAlphaPipeline& alpha_pipeline,
      AlphaPipelineState& alpha_smem_pipe_read,
      MainloopBetaPipeline& beta_pipeline,
      BetaPipelineState& beta_smem_pipe_read,
      MainloopAlphaLastPipeline& alpha_last_pipeline,
      AlphaLastPipelineState& alpha_last_smem_pipe_read,
      OrderedMathBarriers& math_barriers,
      SharedStorage& storage) {
    // MAKE NVCC HAPPY!
    constexpr auto zero = Element{};

    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    DPRINTF0_WG("num_blocks: %d\n", num_blocks);

    int thread_idx = int(threadIdx.x) - NumLoadThreads;
    int warpgroup_idx = thread_idx / cutlass::NumThreadsPerWarpGroup;
    int thread_idx_in_wg = thread_idx % cutlass::NumThreadsPerWarpGroup;

    float scale = params.scale;

    Tensor Beta = make_tensor(make_smem_ptr(storage.smem_beta.data()), SmemLayoutBeta{});
    Tensor AlphaLast = make_tensor(make_smem_ptr(storage.smem_alpha_last.data()), SmemLayoutAlphaLast{});

    Tensor sQqk = make_tensor(make_smem_ptr(storage.smem_q.data()), QKSmemLayoutQ{});
    Tensor sKqk = make_tensor(make_smem_ptr(storage.smem_k.data()), QKSmemLayoutK{});
    Tensor sAqkq = make_tensor(make_smem_ptr(storage.smem_alpha.data()), QKQSmemLayoutAlpha{});
    Tensor sVkv = make_tensor(make_smem_ptr(storage.smem_v.data()), KVSmemLayoutV{});
    Tensor sQK = make_tensor(make_smem_ptr(storage.smem_qk.data()), SmemLayoutQK{});
    Tensor sO = make_tensor(make_smem_ptr(storage.smem_o.data()), SmemLayoutO{});

    static_assert(sizeof(InverseType) == sizeof(Element));
    Tensor sKK_inv = make_tensor(make_smem_ptr(storage.smem_kk.data()), SmemLayoutKK{});
    Tensor sKK_opd = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(storage.smem_kk.data())), SmemLayoutKK{});

    Tensor sQ_K_scaled = make_tensor(make_smem_ptr(storage.smem_q_k_scaled.data()), QKScaledSmemLayoutQ{});
    Tensor sQ_K_scaled_Kt = make_tensor(make_smem_ptr(storage.smem_q_k_scaled.data()), QKScaledSmemLayoutKt{});

    ///////////////////////////////////////////////////////////////////////////
    // Q@S, K@S, Q/K prologue
    // each WG process 32 at a time, reduce peak register usage
    // each WG process half head dim (64) at all
    auto qk_tiled_mma_rs_quar = TiledMmaQK_RS_Quar{};
    auto qk_thr_mma_rs_quar = qk_tiled_mma_rs_quar.get_thread_slice(thread_idx_in_wg);
    constexpr auto tiler_alpha = Shape<_64, Shape<_32, _1>>{};
    constexpr auto tiler_qk = Shape<_64, Shape<_32, _1>>{};
    constexpr auto tiler_alpha_last = Shape<_32>{};
    // used for Alpha S2R (float)
    using CopyAlphaAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAlpha>;
    // used for Q/K S2R and R2S (fp16/bf16)
    using CopyOpS2R = SM75_U32x4_LDSM_N;
    using CopyOpR2S = SM90_U32x4_STSM_N;
    auto tiled_load_qk_quar = make_tiled_copy_A(Copy_Atom<CopyOpS2R, Element>{}, qk_thr_mma_rs_quar);
    auto thr_load_qk_quar = tiled_load_qk_quar.get_thread_slice(thread_idx_in_wg);
    auto tiled_store_qk_quar = make_tiled_copy_A(Copy_Atom<CopyOpR2S, Element>{}, qk_thr_mma_rs_quar);
    auto thr_store_qk_quar = tiled_store_qk_quar.get_thread_slice(thread_idx_in_wg);

    auto cMq_quar = make_identity_tensor(select<0, 2>(TileShapeQK_Quar{}));  // (QTok, HeadDim / 2)
    auto tQcMq_quar = qk_thr_mma_rs_quar.partition_A(cMq_quar);              // (idx) -> (tok_q, head_dim / 2)

    ///////////////////////////////////////////////////////////////////////////
    // K@K  (basically I + strict_lower_triangular(K K^T)
    auto kk_tiled_mma = TiledMmaKK{};
    auto kk_thr_mma = kk_tiled_mma.get_thread_slice(thread_idx_in_wg);
    Tensor tKKsK = kk_thr_mma.partition_B(sKqk);
    Tensor tKKrA = kk_thr_mma.make_fragment_A(tKKsK);
    auto cMqk = make_identity_tensor(select<0, 1>(TileShapeQK{}));  // (QTok, KTok)
    auto const& cMkk = cMqk;
    auto tKKcMkk = kk_thr_mma.partition_C(cMkk);

    // S@K  (-S K^T  +  V^T)
    auto sk_tiled_mma = TiledMmaSK{};
    auto sk_thr_mma = sk_tiled_mma.get_thread_slice(thread_idx);

    // tSKrV adds to tSKrSK (acc)
    using SK_V_S2R = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    auto tSKrV_tiled_copy = make_tiled_copy_C(SK_V_S2R{}, sk_tiled_mma);
    auto tSKrV_thr_copy = tSKrV_tiled_copy.get_thread_slice(thread_idx);

    Tensor tSKsK = sk_thr_mma.partition_B(sQ_K_scaled);
    Tensor tSKrK = sk_thr_mma.make_fragment_B(tSKsK);

    ///////////////////////////////////////////////////////////////////////////
    // NewV = (S@K result) @ T^t
    auto newv_tiled_mma = TiledMmaNewV{};
    auto newv_thr_mma = newv_tiled_mma.get_thread_slice(thread_idx);

    Tensor tNewVsB = newv_thr_mma.partition_B(sKK_opd);
    Tensor tNewVrB = newv_thr_mma.make_fragment_B(tNewVsB);

    ///////////////////////////////////////////////////////////////////////////
    // K@V
    auto kv_tiled_mma = TiledMmaKV{};  // (V, Blk_k) @ (Blk_k, K) = (V, K)
    auto kv_thr_mma = kv_tiled_mma.get_thread_slice(thread_idx);

    Tensor tKVrKV = partition_fragment_C(kv_thr_mma, select<0, 1>(TileShapeKV{}));

    // Tensor tKVrV    = kv_thr_mma.partition_fragment_A(sVkv(_, _, _0{}));  // mma src
    // Tensor tKVrV_cv = tKVrV_thr_copy.retile_D(tKVrV);                     // copy view dst
    // Tensor tKVsV    = tKVrV_thr_copy.partition_S(sVkv);                   // copy view src

    auto const cV = make_identity_tensor(Shape<Int<HeadSizeV>, Int<BlkSeqKV>>{});
    Tensor tKVcV = kv_thr_mma.partition_A(cV);
    auto const cS = make_identity_tensor(Shape<Int<HeadSizeV>, Int<HeadSizeQK>>{});
    Tensor tKVcS = kv_thr_mma.partition_C(cS);

    ///////////////////////////////////////////////////////////////////////////
    // Q@K@V
    auto o1_tiled_mma = TiledMmaO1{};
    auto o1_thr_mma = o1_tiled_mma.get_thread_slice(thread_idx);
    auto o2_tiled_mma = TiledMmaO2{};
    auto o2_thr_mma = o2_tiled_mma.get_thread_slice(thread_idx);

    // A1 for Q@(KV)
    // Tensor tOrKV = make_acc_into_op<Element>(tKVrKV, typename TiledMmaO1::LayoutA_TV{});
    // B1 for Q@(KV)
    Tensor tOsQ = o1_thr_mma.partition_B(sQ_K_scaled);
    Tensor tOrQ = o1_thr_mma.make_fragment_B(tOsQ);

    // A2 for QK@V
    // Tensor tOsV = o2_thr_mma.partition_A(sVkv);
    // Tensor tOrV = o2_thr_mma.make_fragment_A(tOsV);
    // B2 for QK@V
    Tensor tOsQK = o2_thr_mma.partition_B(sQK);
    Tensor tOrQK = o2_thr_mma.make_fragment_B(tOsQK);

    using O_R2S = typename CollectiveStoreO::CopyAtomR2S;
    auto tiled_copy_o = make_tiled_copy_C(O_R2S{}, o1_tiled_mma);
    auto thr_copy_o = tiled_copy_o.get_thread_slice(thread_idx);
    auto tOsO = thr_copy_o.partition_D(sO);

    auto const cO = make_identity_tensor(Shape<Int<HeadSizeQK>, Int<BlkSeqQ>>{});
    Tensor tOcO = o1_thr_mma.partition_C(cO);

    auto const seq_idx = work_desc.seq_idx;
    auto const q_head_idx = work_desc.q_head_idx();
    auto const k_head_idx = work_desc.k_head_idx();
    auto const v_head_idx = work_desc.v_head_idx();

    auto sk_load_v = [&](int pipe_idx) INLINE_LAMBDA {
      Tensor tSKrV = make_fragment_like<Element>(partition_fragment_C(sk_thr_mma, sVkv(_, _, _0{})));  // mma acc
      Tensor tSKrV_cv = tSKrV_thr_copy.retile_D(tSKrV);                                                // copy view dst
      Tensor tSKsV = tSKrV_thr_copy.partition_S(sVkv);                                                 // copy view src
      copy(tSKrV_tiled_copy, tSKsV(_, _, _, pipe_idx), tSKrV_cv);
      return tSKrV;
    };

    auto kv_load = [&](auto& tKVrKV) INLINE_LAMBDA {
      DPRINTF0_WG("[%d,%d,%d,%d]>> load tKVgKV -> tKVrKV\n", seq_idx, q_head_idx, k_head_idx, v_head_idx);
      int num_state_heads = problem_size.num_heads;
      int state_head_idx = work_desc.o_head_idx();
      auto gKV = make_tensor(
          make_gmem_ptr(params.ptr_input_state),
          make_layout(make_shape(Int<HeadSizeQK>{}, Int<HeadSizeV>{}, num_state_heads, problem_size.num_seqs)))(
          _, _, state_head_idx, seq_idx);  // (KDim, VDim), K-contiguous
      // NOTE: load S in transposed GMEM
      // because in GDN's equation, S = NewV^T @ K, while in KDA, S = K^T @ NewV
      auto gKV_trans = make_tensor(
          make_gmem_ptr(gKV.data()),
          make_layout(
              make_shape(get<1>(gKV.layout().shape()), get<0>(gKV.layout().shape())),
              make_stride(get<1>(gKV.layout().stride()), get<0>(gKV.layout().stride()))));

      auto tiled_copy_kv = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, Element>{}, kv_tiled_mma);
      auto thr_copy_kv = tiled_copy_kv.get_thread_slice(thread_idx);

      auto tKVgKV = thr_copy_kv.partition_S(select_tensor<1, 0>(gKV_trans));
      copy(tiled_copy_kv, tKVgKV, tKVrKV);
    };

    auto kv_store = [&]() INLINE_LAMBDA {  // tKVrKV is carried over whole mainloop
      DPRINTF0_WG("[%d,%d,%d,%d]>> save tKVrKV -> tKVgKV\n", seq_idx, q_head_idx, k_head_idx, v_head_idx);
      int num_state_heads = problem_size.num_heads;
      int state_head_idx = work_desc.o_head_idx();
      auto gKV = make_tensor(
          make_gmem_ptr(params.ptr_output_state),
          make_layout(make_shape(Int<HeadSizeQK>{}, Int<HeadSizeV>{}, num_state_heads, problem_size.num_seqs)))(
          _, _, state_head_idx, seq_idx);  // (KDim, VDim), K-contiguous
      // NOTE: store S in transposed GMEM
      // because in GDN's equation, S = NewV^T @ K, while in KDA, S = K^T @ NewV
      auto gKV_trans = make_tensor(
          make_gmem_ptr(gKV.data()),
          make_layout(
              make_shape(get<1>(gKV.layout().shape()), get<0>(gKV.layout().shape())),
              make_stride(get<1>(gKV.layout().stride()), get<0>(gKV.layout().stride()))));

      auto tiled_copy_kv = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, Element>{}, kv_tiled_mma);
      auto thr_copy_kv = tiled_copy_kv.get_thread_slice(thread_idx);

      auto tKVgKV = thr_copy_kv.partition_D(select_tensor<1, 0>(gKV_trans));
      copy(tiled_copy_kv, tKVrKV, tKVgKV);
    };

    auto s_decay = [&](auto& tKVrKV, auto const& alpha_last_smem_pipe_read) INLINE_LAMBDA {
      Tensor alpha_last_curr = AlphaLast(_, alpha_last_smem_pipe_read.index());
      for_each(make_int_sequence<size(tKVcS)>{}, [&](auto i) {
        auto coord = tKVcS(i);
        auto [s, t] = coord;  // (head_size_v, head_size_k)
        tKVrKV(i) *= exp2f(alpha_last_curr(t));
      });
    };

    auto o1_epi = [&](auto& tOrO1) INLINE_LAMBDA {
      CUTE_UNROLL
      for (int i = 0; i < size(tOrO1); ++i) {
        tOrO1(i) = scale * tOrO1(i);
      }
    };

    auto o_store = [&](auto tOrO) INLINE_LAMBDA {
      auto tOrO_cvt = make_fragment_like<ElementO>(tOrO);
      copy(tOrO, tOrO_cvt);

      DPRINTF0_WG("compute: o_pipeline.producer_wait: smem_pipe_write:%d\n", o_smem_pipe_write.index());
      o_pipeline.producer_acquire(o_smem_pipe_write);
      Tensor tOrO_cvt_cv = thr_copy_o.retile_S(tOrO_cvt);
      cutlass::arch::fence_view_async_shared();
      copy(tiled_copy_o, tOrO_cvt_cv, tOsO(_, _, _, o_smem_pipe_write.index()));
      cutlass::arch::fence_view_async_shared();
      o_pipeline.producer_commit(o_smem_pipe_write);
      ++o_smem_pipe_write;
    };

    auto kk_inv = [&](auto const& kk_smem_pipe_read) INLINE_LAMBDA {
      auto sKK_inv_pipe_slice = sKK_inv(_, _, kk_smem_pipe_read.index());
      static_assert(sizeof(Element) == 2);
      using CopyOpR2S = SM90_U32x4_STSM_N;
      auto tiled_store_kk = make_tiled_copy_C(Copy_Atom<CopyOpR2S, InverseType>{}, kk_tiled_mma);
      auto thr_store_kk = tiled_store_kk.get_thread_slice(thread_idx);
      auto tKKsKK = thr_store_kk.partition_D(sKK_inv_pipe_slice);
      // TODO: use tKKcMkk? no more allocating fragments
      auto tKKrKK = kk_thr_mma.partition_fragment_C(sKK_inv_pipe_slice);
      auto tKKrKK_cv = thr_store_kk.retile_S(tKKrKK);
      auto collective_inverse = CollectiveInverse(KdaNamedBarriers::StateMathWG0);
      collective_inverse.compute(sKK_inv_pipe_slice);
      // FIXME: we can ignore core matrices above diagonal
      if constexpr (NeedsBeta || !std::is_same_v<InverseType, Element>) {
        cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, KdaNamedBarriers::StateMathWG0);
        using CopyOpS2R = SM75_U32x4_LDSM_N;
        auto tiled_load_kk = make_tiled_copy_C(Copy_Atom<CopyOpS2R, InverseType>{}, kk_tiled_mma);
        auto thr_load_kk = tiled_load_kk.get_thread_slice(thread_idx);
        auto tKKrKK_cpy = make_fragment_like<InverseType>(tKKrKK_cv);
        auto tKKrKK_cvt = make_fragment_like<Element>(tKKrKK_cv);
        auto tKKcMkk_cv = thr_load_kk.retile_D(tKKcMkk);
        copy(tiled_load_kk, thr_load_kk.partition_S(sKK_inv_pipe_slice), tKKrKK_cpy);
        cute::transform(tKKrKK_cpy, tKKcMkk_cv, tKKrKK_cvt, [&](auto val, auto coord) {
          auto [_, t] = coord;
          if constexpr (NeedsBeta) {
            return Element(float(val) * Beta(t, beta_smem_pipe_read.index()));
          } else {
            return Element(val);
          }
        });
        copy(tiled_store_kk, tKKrKK_cvt, recast<Element>(tKKsKK));
      }
    };

    auto compute_loop_body = [&](int blk, auto is_first_block_, auto is_final_block_) INLINE_LAMBDA {
      constexpr bool is_first_block = decltype(is_first_block_)::value;
      constexpr bool is_final_block = decltype(is_final_block_)::value;
      int B = is_final_block ? valid_seq_len(work_desc, blk) : BlkSeqKV;

      auto sQqk_curr = sQqk(_, _, q_smem_pipe_read.index());
      auto sKqk_curr = sKqk(_, _, k_smem_pipe_read.index());
      auto sQ_scaled_curr = sQ_K_scaled(_, _, _0{});
      auto sK_scaled_curr = sQ_K_scaled(_, _, _1{});
      auto sAlast_curr = AlphaLast(_, alpha_last_smem_pipe_read.index());
      auto sAqkq_curr = sAqkq(_, _, alpha_smem_pipe_read.index());
      auto sQqk_slice = flat_divide(sQqk_curr, tiler_qk);
      auto sKqk_slice = flat_divide(sKqk_curr, tiler_qk);
      auto sQ_scaled_slice = flat_divide(sQ_scaled_curr, tiler_qk);
      auto sK_scaled_slice = flat_divide(sK_scaled_curr, tiler_qk);
      auto sAqkq_slice = flat_divide(sAqkq_curr, tiler_alpha);
      auto sAlast_slice = flat_divide(sAlast_curr, tiler_alpha_last);

      if constexpr (NeedsAlpha) {
        alpha_pipeline.consumer_wait(alpha_smem_pipe_read);
      }
      DPRINTF0_WG("compute: q_pipeline.consumer_wait: smem_pipe_read:%d\n", q_smem_pipe_read.index());
      q_pipeline.consumer_wait(q_smem_pipe_read);
      DPRINTF0_WG("compute: k_pipeline.consumer_wait: smem_pipe_read:%d\n", k_smem_pipe_read.index());
      k_pipeline.consumer_wait(k_smem_pipe_read);

      // load alpha and exp2(alpha) only once
      // and reuse these registers in exp(alpha) * Q/K prologue
      if constexpr (!is_first_block) {
        // make sure sQ_K_scaled is already consumed for previous K^@V
        cutlass::arch::NamedBarrier::arrive_and_wait(NumStateMmaThreads, KdaNamedBarriers::StateMath);
        // Each WG iterates over 2 slices of 32 elements each.
        // WG0 (thread_idx < 128): wg_idx=0, processes alpha indices {0,1}, Q/K dim1=0
        // WG1 (thread_idx >= 128): wg_idx=1, processes alpha indices {2,3}, Q/K dim1=1
        {
          int wg_idx = thread_idx / 128;  // 0 or 1
          int alpha_base = wg_idx * 2;    // 0 or 2

          // Allocate Q/K register fragments once (reused across slices)
          // Only shape/layout matters for partition_fragment_A, use compile-time indices
          auto tQKrQ_wg = qk_thr_mma_rs_quar.partition_fragment_A(sQqk_slice(_, _, _0{}, make_coord(_0{}, _0{})));
          auto tQKrK_wg = qk_thr_mma_rs_quar.partition_fragment_A(sKqk_slice(_, _, _0{}, make_coord(_0{}, _0{})));
          auto tArA = make_fragment_like<ElementAlpha>(tQKrQ_wg);

          for (int s = 0; s < 2; ++s) {
            // S2R Alpha: alpha_col = wg_idx * 2 + s
            int alpha_col = alpha_base + s;
            auto sA_cur = sAqkq_slice(_, _, _0{}, make_coord(0, alpha_col));
            auto tAsA_cur = qk_thr_mma_rs_quar.partition_A(sA_cur);
            copy(CopyAlphaAtom{}, tAsA_cur, tArA);

            cute::transform(tArA, [](auto g) { return exp2f(g); });

            // S2R Q
            auto sQqk_cur = sQqk_slice(_, _, _0{}, make_coord(s, wg_idx));
            auto tQKsQ_cur = thr_load_qk_quar.partition_S(sQqk_cur);
            auto tQKrQ_cv = thr_load_qk_quar.retile_D(tQKrQ_wg);
            copy(tiled_load_qk_quar, tQKsQ_cur, tQKrQ_cv);

            // element-wise exp(alpha) * Q
            cute::transform(tQKrQ_wg, tArA, tQKrQ_wg, [&](auto q, auto alpha) {
              Element dst = Element(alpha * float(q));
              return dst;
            });

            // R2S Q -> stage 0
            auto sQ_scaled_cur = sQ_scaled_slice(_, _, _0{}, make_coord(s, wg_idx));
            auto tQKsQ_out = thr_store_qk_quar.partition_D(sQ_scaled_cur);
            auto tQKrQ_out_cv = thr_store_qk_quar.retile_S(tQKrQ_wg);
            copy(tiled_store_qk_quar, tQKrQ_out_cv, tQKsQ_out);

            // S2R K
            auto sKqk_cur = sKqk_slice(_, _, _0{}, make_coord(s, wg_idx));
            auto tQKsK_cur = thr_load_qk_quar.partition_S(sKqk_cur);
            auto tQKrK_cv = thr_load_qk_quar.retile_D(tQKrK_wg);
            copy(tiled_load_qk_quar, tQKsK_cur, tQKrK_cv);

            // element-wise exp(alpha) * K
            cute::transform(tQKrK_wg, tArA, tQKrK_wg, [&](auto k, auto alpha) {
              Element dst = Element(alpha * float(k));
              return dst;
            });

            // R2S K -> stage 1
            auto sK_scaled_cur = sK_scaled_slice(_, _, _0{}, make_coord(s, wg_idx));
            auto tQKsK_out = thr_store_qk_quar.partition_D(sK_scaled_cur);
            auto tQKrK_out_cv = thr_store_qk_quar.retile_S(tQKrK_wg);
            copy(tiled_store_qk_quar, tQKrK_out_cv, tQKsK_out);
          }
        }
        cutlass::arch::NamedBarrier::arrive_and_wait(NumStateMmaThreads, KdaNamedBarriers::StateMath);
        // fence to produce data for WGMMA async proxy
        cutlass::arch::fence_view_async_shared();
        // if (blk <= 1 && thread_idx == 0) {
        //   printf("After Q/K prologue: exp(alpha) * Q at stage 0, exp(alpha) * K at stage 1\n");
        //   cute::print_tensor(sQ_K_scaled_curr);
        // }
      }

      // 2.1 Q @ KV, NOTE: use old KV here

      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch O WGMMA\n", seq_idx, q_head_idx, k_head_idx, v_head_idx);
      auto tOrO = partition_fragment_C(o1_thr_mma, select<0, 1>(TileShapeO1{}));
      if constexpr (is_first_block) {
        DPRINTF0_WG("compute: q_pipeline.consumer_release: smem_pipe_read:%d\n", q_smem_pipe_read.index());
        q_pipeline.consumer_release(q_smem_pipe_read);
        ++q_smem_pipe_read;
      } else {
        Tensor tOrKV = make_acc_into_op<Element>(tKVrKV, typename TiledMmaO1::LayoutA_TV{});
        warpgroup_fence_operand(tOrKV);
        warpgroup_fence_operand(tOrO);
        // ======DEBUG=======
        // if (blk <= 6 && thread_idx == 0) {
        //   printf("=======Before Q@S, block_idx: %d, thread_idx: %d=======\n", blk, thread_idx);
        //   cute::print_tensor(tOrKV);
        //   cute::print_tensor(sQ_K_scaled_slice);
        // }

        math_barriers.ordered_or_wait(warpgroup_idx);
        warpgroup_arrive();
        gemm_zero_acc(o1_thr_mma, tOrKV, tOrQ(_, _, _, 0), tOrO);
        warpgroup_commit_batch();  // q@kv batch
        math_barriers.notify_next_blocked(warpgroup_idx);
      }
      if constexpr (!is_first_block) {
        warpgroup_wait<0>();  // q@kv batch
        // ======DEBUG=======
        // if (blk <= 1 && thread_idx == 0) {
        //   printf("\n");
        //   printf("=======O_inter after Q@S, block_idx: %d, thread_idx: %d=======\n", blk, thread_idx);
        //   cute::print_tensor(tOrO);
        // }
        DPRINTF0_WG("compute: q_pipeline.consumer_release: smem_pipe_read:%d\n", q_smem_pipe_read.index());
        q_pipeline.consumer_release(q_smem_pipe_read);
        ++q_smem_pipe_read;
        o1_epi(tOrO);
      }

      auto tSKrSK = partition_fragment_C(sk_thr_mma, sVkv(_, _, _0{}));
      if constexpr (!is_first_block) {
        auto tSKrS = make_acc_into_op<Element>(tKVrKV, typename TiledMmaSK::LayoutA_TV{});
        warpgroup_fence_operand(tSKrSK);
        warpgroup_fence_operand(tSKrS);
        math_barriers.ordered_or_wait(warpgroup_idx);
        warpgroup_arrive();

        // ======DEBUG=======
        // if (blk <= 6 && thread_idx == 0) {
        //   printf("=======Before K@S, block_idx: %d, thread_idx: %d=======\n", blk, thread_idx);
        //   cute::print_tensor(tSKrS);
        // }

        // SK: K_scaled is in stage 1 of sQ_K_scaled
        gemm_zero_acc(sk_tiled_mma, tSKrS, tSKrK(_, _, _, 1), tSKrSK);
        warpgroup_commit_batch();
        math_barriers.notify_next_blocked(warpgroup_idx);
        warpgroup_wait<0>();
      }
      // ======DEBUG=======
      // if (blk <= 6 && thread_idx == 0) {
      //   printf("=======After K@S, block_idx: %d, thread_idx: %d=======\n", blk, thread_idx);
      //   cute::print_tensor(tSKrSK);
      // }

      DPRINTF0_WG("compute: v_pipeline.consumer_wait: smem_pipe_read:%d\n", v_smem_pipe_read.index());
      v_pipeline.consumer_wait(v_smem_pipe_read);
      auto tSKrV = sk_load_v(v_smem_pipe_read.index());
      if constexpr (!is_first_block) {
        // sk_epi(tSKrSK, alpha_smem_pipe_read);
        // V' = V - SK
        transform(tSKrV, tSKrSK, tSKrV, [](auto v, auto sk) { return v - Element(sk); });
      }

      kk_pipeline.consumer_wait(kk_smem_pipe_read);
      beta_pipeline.consumer_wait(beta_smem_pipe_read);
      cutlass::arch::fence_view_async_shared();
      // KK inverse
      if (warpgroup_idx == 0) {
        kk_inv(kk_smem_pipe_read);
      }
      // wait for KK inverse ready
      cutlass::arch::NamedBarrier::arrive_and_wait(NumStateMmaThreads, KdaNamedBarriers::StateMath);

      auto tNewVrA = make_acc_into_op<Element>(tSKrV, typename TiledMmaNewV::LayoutA_TV{});
      auto tNewVrC = partition_fragment_C(newv_thr_mma, select<0, 1>(TileShapeNewV{}));
      warpgroup_fence_operand(tNewVrA);
      warpgroup_fence_operand(tNewVrC);
      math_barriers.ordered_or_wait(warpgroup_idx);
      warpgroup_arrive();
      // if constexpr (is_final_block) {
      //   if (thread_idx == 0) {
      //     printf("\n");
      //     printf("=======tNewVrA, tNewVrB before V'@T, block_idx: %d, thread_idx: %d=======\n", blk,
      //     thread_idx); printf("tNewVrA\n"); cute::print_tensor(tNewVrA); printf("sKK_opd\n");
      //     cute::print_tensor(sKK_opd);
      //     printf("=======tNewVrA, tNewVrB before V'@T, block_idx: %d, thread_idx: %d=======\n", blk,
      //     thread_idx); printf("\n");
      //   }
      // }
      // NewV = V'T
      gemm_zero_acc(o1_thr_mma, tNewVrA, tNewVrB(_, _, _, kk_smem_pipe_read.index()), tNewVrC);
      warpgroup_commit_batch();  // new_v batch
      math_barriers.notify_next_blocked(warpgroup_idx);
      warpgroup_wait<0>();  // new_v batch
      // if constexpr (is_final_block) {
      //   if (thread_idx == 0) {
      //     printf("\n");
      //     printf("=======tNewVrC after V'@T, block_idx: %d, thread_idx: %d=======\n", blk, thread_idx);
      //     cute::print_tensor(tNewVrC);
      //   }
      // }
      DPRINTF0_WG("compute: v_pipeline.consumer_release: smem_pipe_read:%d\n", v_smem_pipe_read.index());
      ++v_smem_pipe_read;  // NOTE: if we delay this increment after consumer_release, race condition happens,
                           // why?
      v_pipeline.consumer_release(v_smem_pipe_read);

      kk_pipeline.consumer_release(kk_smem_pipe_read);
      ++kk_smem_pipe_read;
      beta_pipeline.consumer_release(beta_smem_pipe_read);
      ++beta_smem_pipe_read;

      /////////////////////////////////////////////////////////////////////////
      // 2. compute qkv
      // 2.2 QK @ V, NOTE: use old KV here and QK is scaled
      qk_pipeline.consumer_wait(qk_smem_pipe_read);
      auto tOrV_or_tKVrV = make_acc_into_op<Element>(tNewVrC, typename TiledMmaKV::LayoutA_TV{});
      warpgroup_fence_operand(tOrV_or_tKVrV);
      warpgroup_fence_operand(tOrO);
      math_barriers.ordered_or_wait(warpgroup_idx);
      warpgroup_arrive();
      // (V_new)^T @ QK
      if constexpr (is_first_block) {
        gemm_zero_acc(o2_tiled_mma, tOrV_or_tKVrV, tOrQK(_, _, _, qk_smem_pipe_read.index()), tOrO);
      } else {
        gemm(o2_tiled_mma, tOrV_or_tKVrV, tOrQK(_, _, _, qk_smem_pipe_read.index()), tOrO);
      }
      warpgroup_commit_batch();  // qk@v batch
      math_barriers.notify_next_blocked(warpgroup_idx);
      warpgroup_wait<0>();  // qk@v batch
      // if (blk <= 6 && thread_idx == 0) {
      //   printf("\n");
      //   printf("=======O_intra after NewV@QK, block_idx: %d, thread_idx: %d=======\n", blk, thread_idx);
      //   cute::print_tensor(tOrO);
      // }
      qk_pipeline.consumer_release(qk_smem_pipe_read);
      ++qk_smem_pipe_read;
      o_store(tOrO);

      /////////////////////////////////////////////////////////////////////////
      // 3. update KV
      Tensor tKVsK = kv_thr_mma.partition_B(sQ_K_scaled_Kt);
      Tensor tKVrK = kv_thr_mma.make_fragment_B(tKVsK);

      if constexpr (NeedsAlpha) {
        alpha_last_pipeline.consumer_wait(alpha_last_smem_pipe_read);
        cutlass::arch::fence_view_async_shared();
      }
      s_decay(tKVrKV, alpha_last_smem_pipe_read);

      // synchronize 2 WGs before rewriting sQ_K_scaled
      cutlass::arch::NamedBarrier::arrive_and_wait(NumStateMmaThreads, KdaNamedBarriers::StateMath);
      // exp(alpha_last - alpha) * K
      // Each WG iterates over 2 slices of 32 elements each.
      // WG0 (thread_idx < 128): wg_idx=0, alpha_last indices {0,1}, K/output dim1=0
      // WG1 (thread_idx >= 128): wg_idx=1, alpha_last indices {2,3}, K/output dim1=1
      {
        int wg_idx = thread_idx / 128;  // 0 or 1
        int alpha_base = wg_idx * 2;    // 0 or 2

        // Allocate K/Alpha register fragments once (reused across slices)
        auto tQKrK_wg = qk_thr_mma_rs_quar.partition_fragment_A(sKqk_slice(_, _, _0{}, make_coord(_0{}, _0{})));
        auto tArA_wg = make_fragment_like<ElementAlpha>(tQKrK_wg);

        for (int s = 0; s < 2; ++s) {
          // S2R Alpha
          int alpha_col = alpha_base + s;
          auto sA_cur = sAqkq_slice(_, _, _0{}, make_coord(0, alpha_col));
          auto tAsA_cur = qk_thr_mma_rs_quar.partition_A(sA_cur);
          copy(CopyAlphaAtom{}, tAsA_cur, tArA_wg);

          // S2R K
          auto sKqk_cur = sKqk_slice(_, _, _0{}, make_coord(s, wg_idx));
          auto tQKsK_cur = thr_load_qk_quar.partition_S(sKqk_cur);
          auto tQKrK_cv = thr_load_qk_quar.retile_D(tQKrK_wg);
          copy(tiled_load_qk_quar, tQKsK_cur, tQKrK_cv);

          // element-wise: exp(alpha_last - alpha) * K
          int alast_idx = alpha_base + s;
          auto alpha_last_cur = sAlast_slice(_, alast_idx);
          for_each(make_int_sequence<size(tQcMq_quar)>{}, [&](auto i) {
            auto coord = tQcMq_quar(i);
            auto [seq, t] = coord;
            auto alpha = tArA_wg(i);
            auto k = tQKrK_wg(i);
            auto alpha_last = alpha_last_cur(t);
            auto k_scaled = Element(exp2f(alpha_last - alpha) * float(k));
            tQKrK_wg(i) = k_scaled;
            if constexpr (is_final_block) {
              if (seq >= B) {
                tQKrK_wg(i) = Element(0.0f);
              }
            }
          });

          // R2S K -> stage 0 (reuse for KV update)
          auto sQ_scaled_cur = sQ_scaled_slice(_, _, _0{}, make_coord(s, wg_idx));
          auto tQKsK_out = thr_store_qk_quar.partition_D(sQ_scaled_cur);
          auto tQKrK_out_cv = thr_store_qk_quar.retile_S(tQKrK_wg);
          copy(tiled_store_qk_quar, tQKrK_out_cv, tQKsK_out);
        }
      }
      // wait for smemq_k_scaled ready
      cutlass::arch::NamedBarrier::arrive_and_wait(NumStateMmaThreads, KdaNamedBarriers::StateMath);
      // fence to produce data for WGMMA async proxy
      cutlass::arch::fence_view_async_shared();

      if constexpr (NeedsAlpha) {
        alpha_last_pipeline.consumer_release(alpha_last_smem_pipe_read);
        ++alpha_last_smem_pipe_read;
      }

      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch KV WGMMA\n", seq_idx, q_head_idx, k_head_idx, v_head_idx);
      warpgroup_fence_operand(tOrV_or_tKVrV);
      warpgroup_fence_operand(tKVrKV);

      math_barriers.ordered_or_wait(warpgroup_idx);
      warpgroup_arrive();
      gemm(kv_tiled_mma, tOrV_or_tKVrV, tKVrK(_, _, _, 0), tKVrKV);
      warpgroup_commit_batch();  // k@v batch
      math_barriers.notify_next_blocked(warpgroup_idx);
      warpgroup_wait<0>();
      // if constexpr (is_final_block) {
      //   if (thread_idx == 0) {
      //     printf("\n");
      //     printf("=======After K^T@NewV, block_idx: %d, thread_idx: %d=======\n", blk, thread_idx);
      //     printf("tKVrKV\n");
      //     cute::print_tensor(tKVrKV);
      //   }
      // }

      if constexpr (NeedsAlpha) {
        alpha_pipeline.consumer_release(alpha_smem_pipe_read);
        ++alpha_smem_pipe_read;
      }

      DPRINTF0_WG("compute: k_pipeline.consumer_release: smem_pipe_read:%d\n", k_smem_pipe_read.index());
      k_pipeline.consumer_release(k_smem_pipe_read);
      ++k_smem_pipe_read;

      // if (blk <= 6 && thread_idx == 0) {
      //   printf("\n");
      //   printf("=======After S epilogue, block_idx: %d, thread_idx: %d, head_idx: %d=======\n", blk,
      //   thread_idx, q_head_idx); printf("tKVrKV\n"); cute::print_tensor(tKVrKV);
      // }
    };

    if constexpr (!kInitStateFromInput) {
      clear(tKVrKV);
      compute_loop_body(0, /*is_first_block_=*/cute::true_type{}, /*is_final_block_=*/cute::false_type{});
    } else {
      kv_load(tKVrKV);  // GMEM -> Register, only once at the beginning
      compute_loop_body(0, /*is_first_block_=*/cute::false_type{}, /*is_final_block_=*/cute::false_type{});
    }
    CUTE_NO_UNROLL
    for (int blk = 1; blk < num_blocks - 1; ++blk) {
      compute_loop_body(blk, /*is_first_block_=*/cute::false_type{}, /*is_final_block_=*/cute::false_type{});
    }
    if (num_blocks != 1) {
      compute_loop_body(
          num_blocks - 1,
          /*is_first_block_=*/cute::false_type{},
          /*is_final_block_=*/cute::true_type{});
    }
    kv_store();
  }

  template <class ProblemShape, class WorkDesc>
  CUTE_DEVICE void compute_aux_safe(
      Params const& params,
      ProblemShape const& problem_size,
      WorkDesc const& work_desc,
      MainloopQPipeline& q_pipeline,
      QPipelineState& q_smem_pipe_read,
      MainloopKPipeline& k_pipeline,
      KPipelineState& k_smem_pipe_read,
      MainloopQKPipeline& qk_pipeline,
      QKPipelineState& qk_smem_pipe_write,
      MainloopKKPipeline& kk_pipeline,
      KKPipelineState& kk_smem_pipe_write,
      MainloopAlphaPipeline& alpha_pipeline,
      AlphaPipelineState& alpha_smem_pipe_read,
      MainloopBetaPipeline& beta_pipeline,
      BetaPipelineState& beta_smem_pipe_read,
      MainloopAlphaLastPipeline& alpha_last_pipeline,
      AlphaLastPipelineState& alpha_last_smem_pipe_write,
      SharedStorage& storage) {
    using TileShape_SubChunk = Shape<_16, _16, _32>;
    int thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    float scale = params.scale;

    Tensor Beta = make_tensor(make_smem_ptr(storage.smem_beta.data()), SmemLayoutBeta{});

    Tensor sQqk = make_tensor(make_smem_ptr(storage.smem_q.data()), QKSmemLayoutQ{});
    Tensor sKqk = make_tensor(make_smem_ptr(storage.smem_k.data()), QKSmemLayoutK{});

    Tensor sAqkq = make_tensor(make_smem_ptr(storage.smem_alpha.data()), QKQSmemLayoutAlpha{});
    Tensor sAqkk = make_tensor(make_smem_ptr(storage.smem_alpha.data()), QKKSmemLayoutAlpha{});
    Tensor sAlast = make_tensor(make_smem_ptr(storage.smem_alpha_last.data()), SmemLayoutAlphaLast{});

    Tensor sKkv = make_tensor(make_smem_ptr(storage.smem_k.data()), KVSmemLayoutK{});
    Tensor sVkv = make_tensor(make_smem_ptr(storage.smem_v.data()), KVSmemLayoutV{});
    Tensor sQK = make_tensor(make_smem_ptr(storage.smem_qk.data()), SmemLayoutQK{});
    Tensor sO = make_tensor(make_smem_ptr(storage.smem_o.data()), SmemLayoutO{});

    static_assert(sizeof(InverseType) == sizeof(Element));
    Tensor sKK_inv = make_tensor(make_smem_ptr(storage.smem_kk.data()), SmemLayoutKK{});
    Tensor sKK_opd = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(storage.smem_kk.data())), SmemLayoutKK{});

    constexpr int BK = 32;  // should be same as TileShape_SubChunk
    constexpr int NK = 128 / BK;

    ///////////////////////////////////////////////////////////////////////////
    // Q@K
    auto qk_tiled_mma_rs = TiledMmaQK_RS{};
    auto qk_thr_mma_rs = qk_tiled_mma_rs.get_thread_slice(thread_idx);

    auto cMqk = make_identity_tensor(select<0, 1>(TileShapeQK{}));  // (QTok, KTok)
    auto tQKcMqk = qk_thr_mma_rs.partition_C(cMqk);                 // (idx) -> (tok_q, tok_k)
    auto cMq = make_identity_tensor(select<0, 2>(TileShapeQK{}));   // (QTok, HeadDim)
    auto tQcMq = qk_thr_mma_rs.partition_A(cMq);                    // (idx) -> (tok_q, head_dim)

    auto const seq_idx = work_desc.seq_idx;
    auto const q_head_idx = work_desc.q_head_idx();
    auto const k_head_idx = work_desc.k_head_idx();
    auto const v_head_idx = work_desc.v_head_idx();

    auto qk_kk_subchunk_mma_and_store = [&](int blk) INLINE_LAMBDA {
      using CopyOp_R2S = SM90_U32x2_STSM_N;
      using CopyAlphaAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAlpha>;
      // Q/K S2R: use BF16 MMA's LDSM tiled copy for efficient shared memory loads,
      // then convert register layout to TF32 MMA layout via warp shuffles.
      // This replaces the previous AutoVectorizingCopy<16> which caused 50% more smem traffic.
      using CopyQKAtom_LDSM = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
      // TF32 MMA: float(Element) → tf32 → MMA, better precision than fp16/bf16 MMA
      using MMA = SM80_16x8x8_F32TF32TF32F32_TN;
      using TiledMma_SubChunk = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _2, _1>>{}, TileShape_SubChunk{}));
      // BF16 MMA (same shape 16x8x8) used only for creating LDSM-compatible tiled copies
      using MMA_BF16 = SM80_16x8x8_F32BF16BF16F32_TN;
      using TiledMma_BF16_SubChunk =
          decltype(make_tiled_mma(MMA_BF16{}, Layout<Shape<_1, _2, _1>>{}, TileShape_SubChunk{}));

      int local_thread_idx = thread_idx % 64;
      auto tiledmma_subchunk = TiledMma_SubChunk{};
      auto thr_mma_subchunk = tiledmma_subchunk.get_thread_slice(local_thread_idx);
      auto tiledmma_bf16_subchunk = TiledMma_BF16_SubChunk{};
      auto thr_mma_bf16_subchunk = tiledmma_bf16_subchunk.get_thread_slice(local_thread_idx);

      // Alpha S2R: load in BF16 MMA layout so gating happens before the layout shuffle,
      // reducing register pressure (alpha can be freed before the shuffle).
      // BF16-layout alpha copies for operand A and B (for element-wise gating)
      auto alpha_Q_bf16_tiled_copy = make_tiled_copy_A(CopyAlphaAtom{}, tiledmma_bf16_subchunk);
      auto alpha_Kt_bf16_tiled_copy = make_tiled_copy_B(CopyAlphaAtom{}, tiledmma_bf16_subchunk);
      // Q/K S2R: LDSM copies using BF16 MMA layout for efficient ldmatrix loads
      auto Q_tiled_copy = make_tiled_copy_A(CopyQKAtom_LDSM{}, tiledmma_bf16_subchunk);
      auto Kt_tiled_copy = make_tiled_copy_B(CopyQKAtom_LDSM{}, tiledmma_bf16_subchunk);
      // R2S copies for accumulators (C layout, same for all MMA types with same output)
      auto O_tiled_copy = make_tiled_copy_C(Copy_Atom<CopyOp_R2S, Element>{}, tiledmma_subchunk);
      auto O_tiled_copy_kk = make_tiled_copy_C(Copy_Atom<CopyOp_R2S, InverseType>{}, tiledmma_subchunk);

      auto alpha_Q_bf16_thr_copy = alpha_Q_bf16_tiled_copy.get_thread_slice(local_thread_idx);
      auto alpha_Kt_bf16_thr_copy = alpha_Kt_bf16_tiled_copy.get_thread_slice(local_thread_idx);
      auto Q_thr_copy = Q_tiled_copy.get_thread_slice(local_thread_idx);
      auto Kt_thr_copy = Kt_tiled_copy.get_thread_slice(local_thread_idx);
      auto O_thr_copy = O_tiled_copy.get_thread_slice(local_thread_idx);
      auto O_thr_copy_kk = O_tiled_copy_kk.get_thread_slice(local_thread_idx);

      // index tensor
      auto cMqk_subchunk = make_identity_tensor(select<0, 1>(TileShape_SubChunk{}));
      auto tQKcMqk_subchunk = thr_mma_subchunk.partition_C(cMqk_subchunk);

      // do MMA at the granularity of 16x16x64 with two warps
      constexpr auto tiler_subchunk_alpha = Shape<_16, Shape<_32, _1>>{};
      constexpr auto tiler_subchunk_qk = Shape<_16, Shape<_32, _1>>{};
      constexpr auto tiler_subchunk_beta = Shape<_16>{};
      auto sQqk_curr = sQqk(_, _, q_smem_pipe_read.index());
      auto sKqk_curr = sKqk(_, _, k_smem_pipe_read.index());
      auto sAqkq_curr = sAqkq(_, _, alpha_smem_pipe_read.index());
      Tensor sBeta_curr = Beta(_, beta_smem_pipe_read.index());

      // (_16,(_32,_1),_4,(_2,_2)):(_64,(_1,_0),_1024,(_32,_4096))
      auto sQqk_slice = flat_divide(sQqk_curr, tiler_subchunk_qk);
      auto sKqk_slice = flat_divide(sKqk_curr, tiler_subchunk_qk);
      // (_16,(_32,_1),_4,(_1,_4)):(_32,(_1,_0),_512,(_0,_2048))
      auto sAqkq_slice = flat_divide(sAqkq_curr, tiler_subchunk_alpha);
      auto sBeta_slice = flat_divide(sBeta_curr, tiler_subchunk_beta);

      // Acc results
      constexpr auto tiler_acc_qk_kk = Shape<_16, _16>{};
      static_assert(sizeof(Element) == 2);
      auto sQK_curr = sQK(_, _, qk_smem_pipe_write.index());
      auto sQK_slice = flat_divide(sQK_curr, tiler_acc_qk_kk);
      auto sKK_inv_curr = sKK_inv(_, _, kk_smem_pipe_write.index());
      auto sKK_inv_slice = flat_divide(sKK_inv_curr, tiler_acc_qk_kk);

      // used for make_fragment_like in Alpha and S2R (TF32 MMA layout)
      Tensor sQqk_1_0 = sQqk_slice(_, _, _1{}, make_coord(_0{}, _0{}));
      Tensor sKqk_1_0 = sKqk_slice(_, _, _1{}, make_coord(_0{}, _0{}));
      Tensor tQKrQ_1_0 = thr_mma_subchunk.partition_fragment_A(sQqk_1_0);
      Tensor tQKrKt_1_0 = thr_mma_subchunk.partition_fragment_B(sKqk_1_0);
      auto tv_layout_mma_A = tQKrQ_1_0.layout();
      auto tv_layout_mma_B = tQKrKt_1_0.layout();

      // BF16 MMA fragment layouts for LDSM-based S2R loads (same shape, different TV mapping)
      Tensor tQKrQ_bf16_1_0 = thr_mma_bf16_subchunk.partition_fragment_A(sQqk_1_0);
      Tensor tQKrKt_bf16_1_0 = thr_mma_bf16_subchunk.partition_fragment_B(sKqk_1_0);
      auto tv_layout_bf16_mma_A = tQKrQ_bf16_1_0.layout();
      auto tv_layout_bf16_mma_B = tQKrKt_bf16_1_0.layout();

      // S2R Q/K/G for operand A at row r, head dim slice j, and element-wise compute.
      // Loads alpha once in BF16 MMA layout, derives g_first via warp shuffle (8 shuffles,
      // replaces 1 S2R load), gates Q/K before the BF16→TF32 layout conversion.
      // Also extracts g_first in operand B layout for free (broadcast → register copy).
      // j0 = j % 2, j1 = j / 2: precomputed by caller to avoid redundant div/mod.
      // returns (tQKrQ, tQKrK, tArAfirst_kt) = (Q * exp2(g - g_first), K * exp2(g - g_first), g_first in B
      // layout)
      auto s2r_compute_subchunk_operandA = [&](auto r_, int j, int j0, int j1) INLINE_LAMBDA {
        // S2R g_r_j in BF16 MMA operand A layout (single load)
        Tensor sAqkq_r_j = sAqkq_slice(_, _, r_, make_coord(_0{}, j));
        Tensor tAsA_r_j = alpha_Q_bf16_thr_copy.partition_S(sAqkq_r_j);
        Tensor tArA_r_j = make_fragment_like<ElementAlpha>(tv_layout_bf16_mma_A);
        Tensor tArA_r_j_cv = alpha_Q_bf16_thr_copy.retile_D(tArA_r_j);
        copy(alpha_Q_bf16_tiled_copy, tAsA_r_j, tArA_r_j_cv);

        // Derive g_first (alpha[row=0, :]) from tArA_r_j via warp shuffle,
        // directly into operand B layout (8 values instead of 16).
        // g_first is broadcast (all M rows identical), so operand B only needs the
        // v1=0 subset of operand A. We shuffle v1=0 values from t1=0 thread and
        // output directly as operand B fragment, saving 8 float registers.
        Tensor tArAfirst_r_j_kt = make_fragment_like<ElementAlpha>(tv_layout_bf16_mma_B);
        broadcast_row0_operandA_to_operandB_bf16_layout(tArA_r_j, tArAfirst_r_j_kt, local_thread_idx);

        // gqn_r_j = exp2(g_r_j - g_r_j_first[None, :]) in BF16 MMA A layout.
        // g_first per k-iter is in tArAfirst_r_j_kt: frag_B(2j)=K_lo, frag_B(2j+1)=K_hi.
        // In A layout: v1=0 indices (4j+0, 4j+1) have same K as v1=1 indices (4j+2, 4j+3),
        // so g_first for index 4j+{0,2} = frag_B(2j), for 4j+{1,3} = frag_B(2j+1).
        CUTE_UNROLL
        for (int k = 0; k < 4; k++) {
          auto gf_lo = tArAfirst_r_j_kt(2 * k);                      // g_first at K = 2*t0
          auto gf_hi = tArAfirst_r_j_kt(2 * k + 1);                  // g_first at K = 2*t0+1
          tArA_r_j(4 * k + 0) = exp2f(tArA_r_j(4 * k + 0) - gf_lo);  // v0=0, v1=0
          tArA_r_j(4 * k + 1) = exp2f(tArA_r_j(4 * k + 1) - gf_hi);  // v0=1, v1=0
          tArA_r_j(4 * k + 2) = exp2f(tArA_r_j(4 * k + 2) - gf_lo);  // v0=0, v1=1
          tArA_r_j(4 * k + 3) = exp2f(tArA_r_j(4 * k + 3) - gf_hi);  // v0=1, v1=1
        }

        Tensor sQqk_r_j = sQqk_slice(_, _, r_, make_coord(j0, j1));
        Tensor sKqk_r_j = sKqk_slice(_, _, r_, make_coord(j0, j1));

        // --- Process Q ---
        // S2R Q in BF16 MMA layout
        Tensor tQKrQ_r_j_bf16 = make_fragment_like<Element>(tv_layout_bf16_mma_A);
        Tensor tQKsQ_r_j = Q_thr_copy.partition_S(sQqk_r_j);
        Tensor tQKrQ_r_j_bf16_cv = Q_thr_copy.retile_D(tQKrQ_r_j_bf16);
        copy(Q_tiled_copy, tQKsQ_r_j, tQKrQ_r_j_bf16_cv);
        // gate: Q * exp2(g - g_first) in BF16 MMA layout, producing float
        Tensor tQKrQ_r_j_float = make_fragment_like<float>(tv_layout_bf16_mma_A);
        cute::transform(tQKrQ_r_j_bf16, tArA_r_j, tQKrQ_r_j_float, [&](auto q, auto g) { return float(q) * g; });
        // convert BF16 MMA layout → TF32 MMA layout in-place via warp shuffles
        convert_bf16_to_tf32_operandA_layout(tQKrQ_r_j_float, local_thread_idx);
        // NOTE: triton tl.dot also lets MMA hardware for truncation
        // recast float storage as tf32 view (zero cost, same 32-bit registers; MMA hw truncates)
        auto tQKrQ_r_j = recast<ElementGatedMMA>(tQKrQ_r_j_float);

        // --- Process K (sequential, after Q is done to reduce peak reg usage) ---
        Tensor tQKrK_r_j_bf16 = make_fragment_like<Element>(tv_layout_bf16_mma_A);
        Tensor tQKsK_r_j = Q_thr_copy.partition_S(sKqk_r_j);
        Tensor tQKrK_r_j_bf16_cv = Q_thr_copy.retile_D(tQKrK_r_j_bf16);
        copy(Q_tiled_copy, tQKsK_r_j, tQKrK_r_j_bf16_cv);
        Tensor tQKrK_r_j_float = make_fragment_like<float>(tv_layout_bf16_mma_A);
        cute::transform(tQKrK_r_j_bf16, tArA_r_j, tQKrK_r_j_float, [&](auto k, auto g) { return float(k) * g; });
        // convert BF16 MMA layout → TF32 MMA layout in-place via warp shuffles
        convert_bf16_to_tf32_operandA_layout(tQKrK_r_j_float, local_thread_idx);
        auto tQKrK_r_j = recast<ElementGatedMMA>(tQKrK_r_j_float);

        return cute::make_tuple(tQKrQ_r_j, tQKrK_r_j, tArAfirst_r_j_kt);
      };

      // S2R K/G for operand B at column c, head dim slice j, and element-wise compute
      // Loads alpha in BF16 MMA B layout, gates K in BF16 MMA B layout (before shuffle),
      // then converts gated result to TF32 MMA B layout.
      // tArAfirst_kt: pre-loaded g_first register tensor (BF16 MMA B layout) for computing gktn = exp2(g_first -
      // g_c) returns tQKrKt = K_c * exp2(g_first - g_c)
      auto s2r_compute_subchunk_operandB = [&](auto c_, int j, int j0, int j1, auto const& tArAfirst_kt) INLINE_LAMBDA {
        // S2R g_c_j in BF16 MMA operand B layout
        Tensor sAqkq_c_j = sAqkq_slice(_, _, c_, make_coord(_0{}, j));
        Tensor tAsA_c_j = alpha_Kt_bf16_thr_copy.partition_S(sAqkq_c_j);
        Tensor tArA_c_j = make_fragment_like<ElementAlpha>(tv_layout_bf16_mma_B);
        Tensor tArA_c_j_cv = alpha_Kt_bf16_thr_copy.retile_D(tArA_c_j);
        copy(alpha_Kt_bf16_tiled_copy, tAsA_c_j, tArA_c_j_cv);

        // compute gktn_c_j = exp2(g_first - g_c_j) in BF16 MMA B layout
        cute::transform(tArA_c_j, tArAfirst_kt, tArA_c_j, [&](auto g, auto g_first) { return exp2f(g_first - g); });

        // S2R k_c_j using BF16 LDSM
        Tensor sKqk_c_j = sKqk_slice(_, _, c_, make_coord(j0, j1));
        Tensor tQKrKt_c_j_bf16 = make_fragment_like<Element>(tv_layout_bf16_mma_B);
        Tensor tQKsKt_c_j = Kt_thr_copy.partition_S(sKqk_c_j);
        Tensor tQKrKt_c_j_bf16_cv = Kt_thr_copy.retile_D(tQKrKt_c_j_bf16);
        copy(Kt_tiled_copy, tQKsKt_c_j, tQKrKt_c_j_bf16_cv);

        // convert bf16 → float in BF16 MMA B layout
        Tensor tQKrKt_c_j_float = make_fragment_like<float>(tv_layout_bf16_mma_B);
        // gate in BF16 MMA B layout (alpha and K are in the same layout)
        cute::transform(tQKrKt_c_j_bf16, tArA_c_j, tQKrKt_c_j_float, [&](auto k, auto g) { return float(k) * g; });

        // convert BF16 MMA layout → TF32 MMA layout in-place via warp shuffles
        convert_bf16_to_tf32_operandB_layout(tQKrKt_c_j_float, local_thread_idx);
        auto tQKrKt_c_j = recast<ElementGatedMMA>(tQKrKt_c_j_float);

        return tQKrKt_c_j;
      };

      // R2S (register to shared memory) store for subchunk accumulator results
      // Stores both tQKrQK (QK accumulator, fp32 -> Element) and tKKrKK (KK accumulator, fp32 -> InverseType)
      // into their respective shared memory tiles at position (r_, c_)
      auto r2s_subchunk_acc = [&](auto r_, auto c_, auto const& tQKrQK, auto const& tKKrKK) INLINE_LAMBDA {
        // R2S KK
        Tensor sKK_inv_r_c = sKK_inv_slice(_, _, r_, c_);
        Tensor tKKsKK_r_c = O_thr_copy_kk.partition_D(sKK_inv_r_c);
        Tensor tKKrKK_cv = O_thr_copy_kk.retile_S(tKKrKK);
        auto tKKrKK_cvt_cv = make_fragment_like<InverseType>(tKKrKK_cv);
        cute::transform(tKKrKK_cv, tKKrKK_cvt_cv, [](auto v) { return InverseType(v); });
        copy(O_tiled_copy_kk, tKKrKK_cvt_cv, tKKsKK_r_c);

        // R2S QK
        Tensor sQK_r_c = sQK_slice(_, _, r_, c_);
        Tensor tQKsQK_r_c = O_thr_copy.partition_D(sQK_r_c);
        Tensor tQKrQK_cv = O_thr_copy.retile_S(tQKrQK);
        auto tQKrQK_cvt_cv = make_fragment_like<Element>(tQKrQK_cv);
        cute::transform(tQKrQK_cv, tQKrQK_cvt_cv, [](auto v) { return Element(v); });
        copy(O_tiled_copy, tQKrQK_cvt_cv, tQKsQK_r_c);
      };

      // do tensor core GEMM with single 16x16x128
      // NOTE: should use safe_gate with lower_bound >= -5, otherwise overflow issues
      auto gemm_tensor_core_1x16x16x128 =
          [&](auto r_, auto c_, auto is_diagonal_, auto is_first_subchunk_) INLINE_LAMBDA {
            constexpr bool is_first_subchunk = decltype(is_first_subchunk_)::value;

            // allocate acc_r_c [16, 16]
            Tensor tQKrQK_r_c = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
            Tensor tKKrKK_r_c = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
            clear(tQKrQK_r_c);
            clear(tKKrKK_r_c);
            // wait for data ready
            if constexpr (is_first_subchunk) {
              q_pipeline.consumer_wait(q_smem_pipe_read);
              k_pipeline.consumer_wait(k_smem_pipe_read);
              if constexpr (NeedsAlpha) {
                alpha_pipeline.consumer_wait(alpha_smem_pipe_read);
              }
            }

            // for loop head dim
            CUTE_NO_UNROLL
            for (int j = 0; j < NK; ++j) {
              int j0 = j % 2, j1 = j / 2;
              // S2R Q/K/G/g_first for operand A and element-wise compute
              auto [tQKrQ_r_j, tQKrK_r_j, tArAfirst_r_j_kt] = s2r_compute_subchunk_operandA(r_, j, j0, j1);
              // S2R K/G for operand B and element-wise compute
              auto tQKrKt_c_j = s2r_compute_subchunk_operandB(c_, j, j0, j1, tArAfirst_r_j_kt);

              // q_r_j/k_r_j @ k_c_j, accumulate acc_r_c
              gemm(tiledmma_subchunk, tQKrQ_r_j, tQKrKt_c_j, tQKrQK_r_c);
              gemm(tiledmma_subchunk, tQKrK_r_j, tQKrKt_c_j, tKKrKK_r_c);
            }

            // S2R beta_j (maybe resident in register?)
            // epilogue: qk^t * scale
            cute::transform(tQKrQK_r_c, [scale](auto v) { return v * scale; });
            // epilogue: kk^t * beta_r
            if constexpr (is_first_subchunk) {
              beta_pipeline.consumer_wait(beta_smem_pipe_read);
              cutlass::arch::fence_view_async_shared();
            }
            Tensor sBeta_r = sBeta_slice(_, r_);
            for_each(make_int_sequence<size(tQKcMqk_subchunk)>{}, [&](auto i) {
              auto coord = tQKcMqk_subchunk(i);
              auto [s, t] = coord;
              tKKrKK_r_c(i) *= sBeta_r(s);
            });

            // R2S qk_r_c, kk_r_c, wait for current QK/KK free
            if constexpr (is_first_subchunk) {
              kk_pipeline.producer_acquire(kk_smem_pipe_write);
            }
            if constexpr (is_first_subchunk) {
              qk_pipeline.producer_acquire(qk_smem_pipe_write);
            }
            r2s_subchunk_acc(r_, c_, tQKrQK_r_c, tKKrKK_r_c);
          };

      // zero fill for upper triangular of QK and KK, because smem is randomly initialized
      auto zero_fill = [&](int row, int col) INLINE_LAMBDA {
        auto sQK_r_c = sQK_slice(_, _, row, col);
        auto sKK_r_c = sKK_inv_slice(_, _, row, col);
        // allocate regs
        Tensor tQKrQK_r_c = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tQKsQK_r_c = O_thr_copy.partition_D(sQK_r_c);
        Tensor tQKrQK_r_c_cv = O_thr_copy.retile_S(tQKrQK_r_c);
        Tensor tKKsKK_r_c = O_thr_copy_kk.partition_D(sKK_r_c);
        Tensor tKKrKK_r_c_cv = O_thr_copy_kk.retile_S(tQKrQK_r_c);
        auto tQKrQK_r_c_cvt_cv = make_fragment_like<Element>(tQKrQK_r_c_cv);
        auto tKKrKK_r_c_cvt_cv = make_fragment_like<InverseType>(tKKrKK_r_c_cv);
        // zero fill
        clear(tQKrQK_r_c_cvt_cv);
        clear(tKKrKK_r_c_cvt_cv);
        // R2S
        copy(O_tiled_copy, tQKrQK_r_c_cvt_cv, tQKsQK_r_c);
        copy(O_tiled_copy_kk, tKKrKK_r_c_cvt_cv, tKKsKK_r_c);
      };

      // g_i_j/q_i_j/k_i_j: the j-th head dim slice of the i-th subchunk
      if (thread_idx < 64) {
        // Q/K0@K0, Q/K3@K3, Q/K3@K0, Q/K3@K1, Q/K3@K2

        // NOTE: tensor core MMA for safe gate with lower_bound >= -5
        gemm_tensor_core_1x16x16x128(
            Int<0>{},
            Int<0>{},
            /*is_diagonal_=*/cute::true_type{},
            /*is_first_subchunk_=*/cute::true_type{});

        // Q/K3@K0, Q/K3@K1, Q/K3@K2
        // allocate acc_3_0, acc_3_1, acc_3_2 [16, 16]
        Tensor tQKrQK_3_0 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tKKrKK_3_0 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tQKrQK_3_1 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tKKrKK_3_1 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tQKrQK_3_2 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tKKrKK_3_2 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tQKrQK_3_3 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tKKrKK_3_3 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        clear(tQKrQK_3_0);
        clear(tKKrKK_3_0);
        clear(tQKrQK_3_1);
        clear(tKKrKK_3_1);
        clear(tQKrQK_3_2);
        clear(tKKrKK_3_2);
        clear(tQKrQK_3_3);
        clear(tKKrKK_3_3);

        // for loop head dim
        CUTE_NO_UNROLL
        for (int j = 0; j < NK; ++j) {
          int j0 = j % 2, j1 = j / 2;
          // S2R Q/K/G/g_first for operand A (row 3) and element-wise compute
          auto [tQKrQ_3_j, tQKrK_3_j, tArAfirst_3_j_kt] = s2r_compute_subchunk_operandA(_3{}, j, j0, j1);

          // S2R K/G for operand B (col 0) and element-wise compute
          auto tQKrKt_0_j = s2r_compute_subchunk_operandB(_0{}, j, j0, j1, tArAfirst_3_j_kt);
          // q_3_j/k_3_j @ k_0_j, accumulate acc_3_0
          gemm(tiledmma_subchunk, tQKrQ_3_j, tQKrKt_0_j, tQKrQK_3_0);
          gemm(tiledmma_subchunk, tQKrK_3_j, tQKrKt_0_j, tKKrKK_3_0);

          // S2R K/G for operand B (col 1) and element-wise compute
          auto tQKrKt_1_j = s2r_compute_subchunk_operandB(_1{}, j, j0, j1, tArAfirst_3_j_kt);
          // q_3_j/k_3_j @ k_1_j, accumulate acc_3_1
          gemm(tiledmma_subchunk, tQKrQ_3_j, tQKrKt_1_j, tQKrQK_3_1);
          gemm(tiledmma_subchunk, tQKrK_3_j, tQKrKt_1_j, tKKrKK_3_1);

          // S2R K/G for operand B (col 2) and element-wise compute
          auto tQKrKt_2_j = s2r_compute_subchunk_operandB(_2{}, j, j0, j1, tArAfirst_3_j_kt);
          // q_3_j/k_3_j @ k_2_j, accumulate acc_3_2
          gemm(tiledmma_subchunk, tQKrQ_3_j, tQKrKt_2_j, tQKrQK_3_2);
          gemm(tiledmma_subchunk, tQKrK_3_j, tQKrKt_2_j, tKKrKK_3_2);

          // S2R K/G for operand B (col 3) and element-wise compute
          auto tQKrKt_3_j = s2r_compute_subchunk_operandB(_3{}, j, j0, j1, tArAfirst_3_j_kt);
          // q_3_j/k_3_j @ k_3_j, accumulate acc_3_3
          gemm(tiledmma_subchunk, tQKrQ_3_j, tQKrKt_3_j, tQKrQK_3_3);
          gemm(tiledmma_subchunk, tQKrK_3_j, tQKrKt_3_j, tKKrKK_3_3);
        }

        // S2R beta (maybe resident in register?)
        // epilogue: qk^t * scale
        cute::transform(tQKrQK_3_0, [scale](auto v) { return v * scale; });
        cute::transform(tQKrQK_3_1, [scale](auto v) { return v * scale; });
        cute::transform(tQKrQK_3_2, [scale](auto v) { return v * scale; });
        cute::transform(tQKrQK_3_3, [scale](auto v) { return v * scale; });
        // epilogue: kk^t * beta_3
        Tensor sBeta_3 = sBeta_slice(_, _3{});
        for_each(make_int_sequence<size(tQKcMqk_subchunk)>{}, [&](auto i) {
          auto coord = tQKcMqk_subchunk(i);
          auto [s, t] = coord;
          auto b = sBeta_3(s);
          tKKrKK_3_0(i) *= b;
          tKKrKK_3_1(i) *= b;
          tKKrKK_3_2(i) *= b;
          tKKrKK_3_3(i) *= b;
        });

        // R2S qk_3_0, kk_3_0, wait for current QK/KK free
        r2s_subchunk_acc(_3{}, _0{}, tQKrQK_3_0, tKKrKK_3_0);
        // R2S qk_3_1, kk_3_1
        r2s_subchunk_acc(_3{}, _1{}, tQKrQK_3_1, tKKrKK_3_1);
        // R2S qk_3_2, kk_3_2
        r2s_subchunk_acc(_3{}, _2{}, tQKrQK_3_2, tKKrKK_3_2);
        // R2S qk_3_3, kk_3_3
        r2s_subchunk_acc(_3{}, _3{}, tQKrQK_3_3, tKKrKK_3_3);
      } else {
        // Q/K1@K0, Q/K2@K0, Q/K2@K1, Q/K2@K2, Q/K1@K1

        // Q/K1@K0
        // allocate acc_1_0 [16, 16]
        Tensor tQKrQK_1_0 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tKKrKK_1_0 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tQKrQK_1_1 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tKKrKK_1_1 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        clear(tQKrQK_1_0);
        clear(tKKrKK_1_0);
        clear(tQKrQK_1_1);
        clear(tKKrKK_1_1);
        // wait for data ready
        q_pipeline.consumer_wait(q_smem_pipe_read);
        k_pipeline.consumer_wait(k_smem_pipe_read);
        if constexpr (NeedsAlpha) {
          alpha_pipeline.consumer_wait(alpha_smem_pipe_read);
        }

        // for loop head dim
        CUTE_NO_UNROLL
        for (int j = 0; j < NK; ++j) {
          int j0 = j % 2, j1 = j / 2;
          // S2R Q/K/G/g_first for operand A (row 1) and element-wise compute
          auto [tQKrQ_1_j, tQKrK_1_j, tArAfirst_1_j_kt] = s2r_compute_subchunk_operandA(_1{}, j, j0, j1);

          // S2R K/G for operand B (col 0) and element-wise compute
          auto tQKrKt_0_j = s2r_compute_subchunk_operandB(_0{}, j, j0, j1, tArAfirst_1_j_kt);
          // q_1_j/k_1_j @ k_0_j, accumulate acc_1_0
          gemm(tiledmma_subchunk, tQKrQ_1_j, tQKrKt_0_j, tQKrQK_1_0);
          gemm(tiledmma_subchunk, tQKrK_1_j, tQKrKt_0_j, tKKrKK_1_0);

          // S2R K/G for operand B (col 1) and element-wise compute
          auto tQKrKt_1_j = s2r_compute_subchunk_operandB(_1{}, j, j0, j1, tArAfirst_1_j_kt);
          // q_1_j/k_1_j @ k_1_j, accumulate acc_1_1
          gemm(tiledmma_subchunk, tQKrQ_1_j, tQKrKt_1_j, tQKrQK_1_1);
          gemm(tiledmma_subchunk, tQKrK_1_j, tQKrKt_1_j, tKKrKK_1_1);
        }

        // S2R beta_j (maybe resident in register?)
        // epilogue: qk^t * scale
        cute::transform(tQKrQK_1_0, [scale](auto v) { return v * scale; });
        cute::transform(tQKrQK_1_1, [scale](auto v) { return v * scale; });
        // epilogue: kk^t * beta_1
        beta_pipeline.consumer_wait(beta_smem_pipe_read);
        cutlass::arch::fence_view_async_shared();
        Tensor sBeta_1 = sBeta_slice(_, _1{});
        for_each(make_int_sequence<size(tQKcMqk_subchunk)>{}, [&](auto i) {
          auto coord = tQKcMqk_subchunk(i);
          auto [s, t] = coord;
          tKKrKK_1_0(i) *= sBeta_1(s);
          tKKrKK_1_1(i) *= sBeta_1(s);
        });

        // R2S qk_1_0, kk_1_0, wait for current QK/KK free
        kk_pipeline.producer_acquire(kk_smem_pipe_write);
        qk_pipeline.producer_acquire(qk_smem_pipe_write);

        r2s_subchunk_acc(_1{}, _0{}, tQKrQK_1_0, tKKrKK_1_0);
        // R2S qk_1_1, kk_1_1
        r2s_subchunk_acc(_1{}, _1{}, tQKrQK_1_1, tKKrKK_1_1);

        // Q/K2@K0, Q/K2@K1
        // allocate acc_2_0, acc_2_1 [16, 16]
        Tensor tQKrQK_2_0 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tKKrKK_2_0 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tQKrQK_2_1 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tKKrKK_2_1 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tQKrQK_2_2 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        Tensor tKKrKK_2_2 = partition_fragment_C(tiledmma_subchunk, select<0, 1>(TileShape_SubChunk{}));
        clear(tQKrQK_2_0);
        clear(tKKrKK_2_0);
        clear(tQKrQK_2_1);
        clear(tKKrKK_2_1);
        clear(tQKrQK_2_2);
        clear(tKKrKK_2_2);

        // for loop head dim
        CUTE_NO_UNROLL
        for (int j = 0; j < NK; ++j) {
          int j0 = j % 2, j1 = j / 2;
          // S2R Q/K/G/g_first for operand A (row 2) and element-wise compute
          auto [tQKrQ_2_j, tQKrK_2_j, tArAfirst_2_j_kt] = s2r_compute_subchunk_operandA(_2{}, j, j0, j1);

          // S2R K/G for operand B (col 0) and element-wise compute
          auto tQKrKt_0_j = s2r_compute_subchunk_operandB(_0{}, j, j0, j1, tArAfirst_2_j_kt);
          // q_2_j/k_2_j @ k_0_j, accumulate acc_2_0
          gemm(tiledmma_subchunk, tQKrQ_2_j, tQKrKt_0_j, tQKrQK_2_0);
          gemm(tiledmma_subchunk, tQKrK_2_j, tQKrKt_0_j, tKKrKK_2_0);

          // S2R K/G for operand B (col 1) and element-wise compute
          auto tQKrKt_1_j = s2r_compute_subchunk_operandB(_1{}, j, j0, j1, tArAfirst_2_j_kt);
          // q_2_j/k_2_j @ k_1_j, accumulate acc_2_1
          gemm(tiledmma_subchunk, tQKrQ_2_j, tQKrKt_1_j, tQKrQK_2_1);
          gemm(tiledmma_subchunk, tQKrK_2_j, tQKrKt_1_j, tKKrKK_2_1);

          // S2R K/G for operand B (col 2) and element-wise compute
          auto tQKrKt_2_j = s2r_compute_subchunk_operandB(_2{}, j, j0, j1, tArAfirst_2_j_kt);
          // q_2_j/k_2_j @ k_2_j, accumulate acc_2_2
          gemm(tiledmma_subchunk, tQKrQ_2_j, tQKrKt_2_j, tQKrQK_2_2);
          gemm(tiledmma_subchunk, tQKrK_2_j, tQKrKt_2_j, tKKrKK_2_2);
        }

        // S2R beta (maybe resident in register?)
        // epilogue: qk^t * scale
        cute::transform(tQKrQK_2_0, [scale](auto v) { return v * scale; });
        cute::transform(tQKrQK_2_1, [scale](auto v) { return v * scale; });
        cute::transform(tQKrQK_2_2, [scale](auto v) { return v * scale; });
        // epilogue: kk^t * beta_2
        Tensor sBeta_2 = sBeta_slice(_, 2);
        for_each(make_int_sequence<size(tQKcMqk_subchunk)>{}, [&](auto i) {
          auto coord = tQKcMqk_subchunk(i);
          auto [s, t] = coord;
          auto b = sBeta_2(s);
          tKKrKK_2_0(i) *= b;
          tKKrKK_2_1(i) *= b;
          tKKrKK_2_2(i) *= b;
        });

        // R2S qk_2_0, kk_2_0, wait for current QK/KK free
        r2s_subchunk_acc(_2{}, _0{}, tQKrQK_2_0, tKKrKK_2_0);
        // R2S qk_2_1, kk_2_1
        r2s_subchunk_acc(_2{}, _1{}, tQKrQK_2_1, tKKrKK_2_1);
        // R2S qk_2_2, kk_2_2
        r2s_subchunk_acc(_2{}, _2{}, tQKrQK_2_2, tKKrKK_2_2);
      }
    };

    auto qk_and_kk_epi = [&](auto is_final_block_, auto B /*valid seqlen*/) INLINE_LAMBDA {
      using CopyOpS2R_Chunk = SM75_U32x4_LDSM_N;
      using CopyOpR2S_Chunk = SM90_U32x4_STSM_N;
      // S2R QK/KK
      auto sQK_curr = sQK(_, _, qk_smem_pipe_write.index());
      auto sKK_inv_curr = sKK_inv(_, _, kk_smem_pipe_write.index());
      Tensor tQKrQK_ref = partition_fragment_C(TiledMmaQK_RS{}, select<0, 1>(TileShapeQK{}));
      auto tiled_load_qk = make_tiled_copy_C(Copy_Atom<CopyOpS2R_Chunk, Element>{}, qk_tiled_mma_rs);
      auto thr_load_qk = tiled_load_qk.get_thread_slice(thread_idx);
      auto tiled_load_kk = make_tiled_copy_C(Copy_Atom<CopyOpS2R_Chunk, InverseType>{}, qk_tiled_mma_rs);
      auto thr_load_kk = tiled_load_kk.get_thread_slice(thread_idx);

      auto tQKrQK_cv = thr_load_qk.retile_D(tQKrQK_ref);
      auto tQKrQK = make_fragment_like<Element>(tQKrQK_cv);
      auto tKKrKK = make_fragment_like<InverseType>(tQKrQK_cv);
      copy(tiled_load_qk, thr_load_qk.partition_S(sQK_curr), tQKrQK);
      copy(tiled_load_kk, thr_load_kk.partition_S(sKK_inv_curr), tKKrKK);

      // triangular mask and boundary mask
      constexpr bool is_final_block = decltype(is_final_block_)::value;
      for_each(make_int_sequence<size(tQKcMqk)>{}, [&](auto i) {
        auto coord = tQKcMqk(i);
        auto [s, t] = coord;
        bool pred = s >= t;
        tQKrQK(i) = pred ? tQKrQK(i) : Element(0.0f);
        tKKrKK(i) = pred ? tKKrKK(i) : InverseType(0.0f);  // diagonal is garbage filled, will process during inversion
        if constexpr (is_final_block) {
          bool pred = s < B && t < B;
          tQKrQK(i) = pred ? tQKrQK(i) : Element(0.0f);
          tKKrKK(i) = pred ? tKKrKK(i) : InverseType(0.0f);
        }
      });

      // R2S QK/KK
      auto tiled_store_qk = make_tiled_copy_C(Copy_Atom<CopyOpR2S_Chunk, Element>{}, qk_tiled_mma_rs);
      auto thr_store_qk = tiled_store_qk.get_thread_slice(thread_idx);
      auto tiled_store_kk = make_tiled_copy_C(Copy_Atom<CopyOpR2S_Chunk, InverseType>{}, qk_tiled_mma_rs);
      auto thr_store_kk = tiled_store_kk.get_thread_slice(thread_idx);

      copy(tiled_store_qk, thr_store_qk.retile_S(tQKrQK), thr_store_qk.partition_D(sQK_curr));
      copy(tiled_store_kk, thr_store_kk.retile_S(tKKrKK), thr_store_kk.partition_D(sKK_inv_curr));
    };

    auto compute_aux_loop_body = [&](int blk, auto is_final_block_) INLINE_LAMBDA {
      constexpr bool is_final_block = decltype(is_final_block_)::value;

      int B = is_final_block ? valid_seq_len(work_desc, blk) : BlkSeqKV;

      // ====DEBUG=====
      // maintain pipeline correctness while removing subchunk
      // wait for data ready
      // q_pipeline.consumer_wait(q_smem_pipe_read);
      // k_pipeline.consumer_wait(k_smem_pipe_read);
      // if constexpr (NeedsAlpha) { alpha_pipeline.consumer_wait(alpha_smem_pipe_read); }
      // beta_pipeline.consumer_wait(beta_smem_pipe_read);
      // cutlass::arch::fence_view_async_shared();

      // qk_pipeline.producer_acquire(qk_smem_pipe_write);
      // kk_pipeline.producer_acquire(kk_smem_pipe_write);

      // SubChunk MMA for QK^T and KK^T for numerical stability
      // FIXME: use g_half as anchor in the diagonal subchunk to align with FLA for smaller numerical differences
      qk_kk_subchunk_mma_and_store(blk);
      // wait for QK/KK ready
      cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, KdaNamedBarriers::AuxMath);
      qk_and_kk_epi(is_final_block_, B);
      // =====DEBUG======
      // cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup, KdaNamedBarriers::AuxMath);
      // if constexpr (is_final_block) {
      //   if (thread_idx == 0) {
      //     printf("sQK\n");
      //     cute::print_tensor(sQK(_, _, qk_smem_pipe_write.index()));
      //     printf("sKK_inv\n");
      //     cute::print_tensor(sKK_inv(_, _, kk_smem_pipe_write.index()));
      //   }
      // }

      // QK/KK is ready to consume
      cutlass::arch::fence_view_async_shared();
      qk_pipeline.producer_commit(qk_smem_pipe_write);
      ++qk_smem_pipe_write;
      kk_pipeline.producer_commit(kk_smem_pipe_write);
      ++kk_smem_pipe_write;

      k_pipeline.consumer_release(k_smem_pipe_read);
      ++k_smem_pipe_read;
      q_pipeline.consumer_release(q_smem_pipe_read);
      ++q_smem_pipe_read;
      if constexpr (NeedsAlpha) {
        alpha_pipeline.consumer_release(alpha_smem_pipe_read);
        ++alpha_smem_pipe_read;
      }

      if constexpr (NeedsBeta) {
        beta_pipeline.consumer_release(beta_smem_pipe_read);
        ++beta_smem_pipe_read;
      }
    };

    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks - 1; ++blk) {
      compute_aux_loop_body(blk, /*is_final_block_=*/cute::false_type{});
    }
    compute_aux_loop_body(num_blocks - 1, /*is_final_block_=*/cute::true_type{});
  }

  template <typename WorkDesc>
  CUTE_DEVICE int valid_seq_len(WorkDesc work_desc, int blk_idx) {
    int remain_len = work_desc.seq_len - BlkSeqKV * blk_idx;
    return remain_len <= BlkSeqKV ? remain_len : BlkSeqKV;
  }
};

}  // namespace kda::sm90::collective
