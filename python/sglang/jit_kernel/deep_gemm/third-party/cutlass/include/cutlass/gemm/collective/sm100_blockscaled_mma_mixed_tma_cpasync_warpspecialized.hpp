/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/arch/memory.h"

#include "cute/algorithm/functional.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/collective/collective_mma_decl.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
// Both DMA Load and MMA methods of this class must be run by a single thread that's picked by elect_one
template <
  int Stages,
  int SchedulerPipelineStageCount,
  int AccumulatorPipelineStageCount,
  class ClusterShape,   // Static cluster shape or dynamic (int, int, _1)
  class TileShape_,     // (MmaAtomShapeM, MmaAtomShapeN, TileK)
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomPairA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomPairB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm100UmmaMixedTmaCpAsyncWarpSpecializedBlockScaled<
      Stages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomPairA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomPairB_,
    SmemCopyAtomB_,
    TransformB_> {
  using TiledMma = TiledMma_;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;

  // Statically asserting to ensure only 1x1x1 cluster shape & 1sm setup is received
  static_assert(size(AtomThrShapeMNK{}) == 1, "Lower alignment SM100 GEMM only supports 1SM MMA");
  static_assert(size(ClusterShape{}) == 1, "CPASYNC does not support multicast so the cluster shape is restricted to 1, 1, 1");

  static_assert(size(typename TiledMma::AtomThrID{}) == 1);

  using DispatchPolicy = MainloopSm100UmmaMixedTmaCpAsyncWarpSpecializedBlockScaled<
                          Stages,
                          SchedulerPipelineStageCount,
                          AccumulatorPipelineStageCount,
                          ClusterShape>;
  // TileShape refers to MmaTileShape to adapt for runtime cluster
  using TileShape = TileShape_;
  using TiledMma_SF = TiledMMA<MMA_Atom<typename TiledMma::MMA_ScaleFactor>,
                                        Layout<Shape<_1,_1,_1>>,
                                        Tile<Underscore,Underscore,Underscore>>;

  static constexpr int SFVecSize = TiledMma::SFVecSize;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;
  static_assert(!IsOverlappingAccum, "TMA+CPASYNC kernel currently only supports non-overlapping accum.");

  CUTE_STATIC_ASSERT_V(evenly_divides(TileShape{}, tile_shape(TiledMma{})),
                       "Static cluster shape used: TileShape should be evenly divided by TiledMma");

  // Define A and B block shapes
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));
  // using LoadShapeA_MK = decltype(select<0,2>(TileShape{}));
  using LoadShapeB_NK = decltype(select<1,2>(TileShape{}));

  // CtaShape_MNK is queried from collective in all kernel layers
  using CtaShape_MNK = TileShape;
  static_assert(shape<1>(CtaShape_MNK{}) == 192 or shape<1>(CtaShape_MNK{}) == 64 or
      shape<1>(CtaShape_MNK{}) == 128 or shape<1>(CtaShape_MNK{}) == 256,
      "Cta N should be one of 64/128/192/256");

  using ClusterTileShape = decltype(make_shape(get<0>(TileShape{})*get<0>(ClusterShape{}),get<1>(TileShape{})*get<1>(ClusterShape{}),get<2>(TileShape{})*get<2>(ClusterShape{})));
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
  using Blk_MN = typename Sm1xxBlkScaledConfig::Blk_MN;
  static constexpr int IsCtaN192 = shape<1>(CtaShape_MNK{}) == 192;
  static constexpr int IsCtaN64 = shape<1>(CtaShape_MNK{}) == 64;
  static int constexpr CTA_N_SF = cutlass::ceil_div(size<1>(CtaShape_MNK{}), Blk_MN{}) * Blk_MN{};
  // Tile shape used for partitioning Scale Factor B.
  // The M-dim does not affect the SFB, so just set it as the original TileShape;
  using TileShape_SF = decltype(make_shape(get<0>(CtaShape_MNK{}),
                                           Int<CTA_N_SF>{} * shape<2>(typename TiledMma::ThrLayoutVMNK()),
                                           get<2>(TileShape{})));

  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;
  using SmemLayoutAtomPairA = SmemLayoutAtomPairA_;
  using SmemLayoutAtomPairB = SmemLayoutAtomPairB_;
  static_assert(cute::is_same_v<remove_cvref_t<decltype(get<1>(ElementPairA{}))>,
                              remove_cvref_t<decltype(get<1>(ElementPairB{}))>>, "SFA and SFB data types should be the same");


  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using ElementAMma = typename TiledMma::ValTypeA;
  using StrideA = remove_cvref_t<decltype(get<0>(StridePairA{}))>;
  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StrideB = remove_cvref_t<decltype(get<0>(StridePairB{}))>;

  static constexpr bool IsRuntimeDataTypeA = cute::is_same_v<ElementA, cutlass::type_erased_dynamic_float8_t>;
  static constexpr bool IsRuntimeDataTypeB = cute::is_same_v<ElementB, cutlass::type_erased_dynamic_float8_t>;

  static_assert(IsRuntimeDataTypeA == IsRuntimeDataTypeB,
                "ElementA and ElementB should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

  // SFA and SFB
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using LayoutSFA = remove_cvref_t<decltype(get<1>(StridePairA{}))>;
  using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;

  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyPairB = GmemTiledCopyPairB_;

  using GmemTiledCopyA    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopySFA  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopyB    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB{}))>;
  using GmemTiledCopySFB  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB{}))>;

  using SmemLayoutAtomA   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomPairA{}))>;
  using SmemLayoutAtomSFA = remove_cvref_t<decltype(get<1>(SmemLayoutAtomPairA{}))>;
  using SmemLayoutAtomB   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomPairB{}))>;
  using SmemLayoutAtomSFB = remove_cvref_t<decltype(get<1>(SmemLayoutAtomPairB{}))>;

  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipelineTMA = cutlass::PipelineTmaUmmaAsync<DispatchPolicy::Stages, ClusterShape, AtomThrShapeMNK>;
  using MainloopPipelineTMAState = typename MainloopPipelineTMA::PipelineState;

  using MainloopPipelineCpAsync = cutlass::PipelineUmmaConsumerAsync<DispatchPolicy::Stages, AtomThrShapeMNK>;
  using MainloopPipelineCpAsyncState = typename MainloopPipelineCpAsync::PipelineState;

  // static_assert(size(GmemTiledCopyA{}) == size(GmemTiledCopyB{}), "A and B GmemTiledCopy should share the same thread count");
  static constexpr int NumLoadThreadsCpAsync = size(GmemTiledCopyB{});

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtomA must be rank 2 (M,K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtomA must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtomA must evenly divide tile shape.");
  static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtomB must be rank 2 (N,K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtomB must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtomB must evenly divide tile shape.");
  static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // (MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE)
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));


  using MmaSmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using LoadSmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      append(LoadShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  // SmemLayoutAtomSFA and SmemLayoutAtomSFB are for whole CTA tiles. We add the number of pipeline stages here.
  // The number of pipeline stages is the same as the number of pipeline stages from AB Load <-> MainLoop
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");

  static constexpr bool IsF8F6F4 = detail::is_sm100_mma_f8f6f4<TiledMma, ElementA, ElementB>();

  using TmaInternalElementA = cute::conditional_t<IsF8F6F4, ElementAMma, ElementA>;

  using SmemAllocTypeA = cute::conditional_t<IsF8F6F4 && cute::sizeof_bits_v<ElementAMma> < 8, uint8_t, ElementAMma>;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4 && cute::sizeof_bits_v<ElementBMma> < 8, uint8_t, ElementBMma>;

  using BitTypeElementA = cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>;
  using BitTypeElementB = cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>;

  using ArrayElementA = cute::conditional_t<IsRuntimeDataTypeA, BitTypeElementA, ElementA>;
  using ArrayElementB = cute::conditional_t<IsRuntimeDataTypeB, BitTypeElementB, ElementB>;

  using RuntimeDataTypeA = typename detail::sm10x_block_scale_runtime_input_t<ElementAMma, IsRuntimeDataTypeA>::Type;
  using RuntimeDataTypeB = typename detail::sm10x_block_scale_runtime_input_t<ElementBMma, IsRuntimeDataTypeB>::Type;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<LoadSmemLayoutB>> smem_B;
      cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;

    using PipelineStorageTMA = typename MainloopPipelineTMA::SharedStorage;
    using PipelineStorageCpAsync = typename MainloopPipelineCpAsync::SharedStorage;

    struct PipelineStorage : cute::aligned_struct<16, _0> {
      alignas(16) PipelineStorageTMA tma;
      alignas(16) PipelineStorageCpAsync cpasync;
    } pipelines;
  };

  // Expose shared storage for tensors/pipelines separately to allow kernel layer to reorder them.
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr uint32_t SFTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>);
  static constexpr uint32_t ATmaTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>);
  static constexpr uint32_t TmaTransactionBytes = ATmaTransactionBytes + SFTransactionBytes;

  template <class AccTensor, class SfaTensor, class SfbTensor>
  struct TmemStorage {
    AccTensor accumulators;
    SfaTensor tCtSFA;
    SfbTensor tCtSFB;
  };

  // Host side kernel arguments
  struct Arguments {
    ArrayElementA const* ptr_A{nullptr};
    StrideA dA{};
    ArrayElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementSF const* ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    ElementSF const* ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
    RuntimeDataTypeA runtime_data_type_a{};
    RuntimeDataTypeB runtime_data_type_b{};
  };

  // Device side kernel params
  struct Params {
    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(ClusterShape{}),
                                                     make_tile(typename TiledMma::AtomThrID{})));
    using ClusterLayoutSfb_VMNK = decltype(tiled_divide(make_layout(ClusterShape{}),
                                                     make_tile(typename TiledMma_SF::AtomThrID{})));

    using TMA_A = decltype(make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );
    using TMA_SFA = decltype(make_tma_atom_A_sm100<uint16_t>(
        GmemTiledCopySFA{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFA{}),
        SmemLayoutSFA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );
    using TMA_SFB = decltype(make_tma_atom_B_sm100<uint16_t>(
        GmemTiledCopySFB{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFB{}),
        SmemLayoutSFB{}(_,_,_,cute::Int<0>{}),
        TileShape_SF{},
        TiledMma_SF{},
        ClusterLayoutSfb_VMNK{})
      );

    TMA_A tma_load_a;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;

    ArrayElementB const* ptr_B{nullptr};
    StrideB dB{};

    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;

    RuntimeDataTypeA runtime_data_type_a;
    RuntimeDataTypeB runtime_data_type_b;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params)
    : layout_SFA_(params.layout_SFA)
    , layout_SFB_(params.layout_SFB)
    , runtime_data_type_a_(params.runtime_data_type_a)
    , runtime_data_type_b_(params.runtime_data_type_b) {
    
    observed_tma_load_a_ = &params.tma_load_a;
    observed_tma_load_sfa_ = &params.tma_load_sfa;
    observed_tma_load_sfb_ = &params.tma_load_sfb;
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace,
      cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_B = recast_ptr<ElementBMma>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_sfa = make_tensor(args.ptr_SFA, args.layout_SFA);
    Tensor tensor_sfb = make_tensor(args.ptr_SFB, args.layout_SFB);

    auto cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape{}), make_tile(typename TiledMma::AtomThrID{}));
    auto cluster_layout_sfb_vmnk = tiled_divide(make_layout(ClusterShape{}), make_tile(typename TiledMma_SF::AtomThrID{}));

    typename Params::TMA_A tma_load_a = make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);
    typename Params::TMA_SFA tma_load_sfa = make_tma_atom_A_sm100<uint16_t>(
        GmemTiledCopySFA{},
        tensor_sfa,
        SmemLayoutSFA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);
    typename Params::TMA_SFB tma_load_sfb = make_tma_atom_B_sm100<uint16_t>(
        GmemTiledCopySFB{},
        tensor_sfb,
        SmemLayoutSFB{}(_,_,_,cute::Int<0>{}),
        TileShape_SF{},
        TiledMma_SF{},
        cluster_layout_sfb_vmnk);

    return {
      tma_load_a,
      tma_load_sfa,
      tma_load_sfb,
      args.ptr_B,
      args.dB,
      args.layout_SFA,
      args.layout_SFB,
      args.runtime_data_type_a,
      args.runtime_data_type_b
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    // static constexpr bool IsF8F6F4 = detail::is_sm100_mma_f8f6f4<TiledMma, ElementA, ElementB>();
    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cute::sizeof_bits<ElementA>::value;

    bool implementable = true;

    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }

    implementable = implementable && cutlass::detail::check_alignment<GmemTiledCopyB::NumValSrc>(cute::make_shape(N,K,L), StrideB{});
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for CpAsync.\n");
    }

    // Check for SFA SFB layout requirement
    const auto layout_sfa_ref = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
    const auto layout_sfb_ref = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);
    implementable = implementable && (layout_sfa_ref == args.layout_SFA);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: layout_SFA mismatch, layout_SFA needs to be K-major\n");
    }

    implementable = implementable && (layout_sfb_ref == args.layout_SFB);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: layout_SFB mismatch, layout_SFB needs to be K-major\n");
    }

    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE void
  prefetch_tma_descriptors() {
    cute::prefetch_tma_descriptor(observed_tma_load_a_->get_tma_descriptor());
    cute::prefetch_tma_descriptor(observed_tma_load_sfa_->get_tma_descriptor());
    cute::prefetch_tma_descriptor(observed_tma_load_sfb_->get_tma_descriptor());
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE static 
  auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));  // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

    return acc_shape;
  }

  template <class TmemStorage>
  CUTLASS_DEVICE static
  auto
  slice_accumulator(TmemStorage tmem_storage, int stage) {
    return cute::make_tuple(tmem_storage.accumulators(_,_,_,stage));
  }

  template <class EpilogueTile, bool IsOverlappingAccum = false>
  CUTLASS_DEVICE static
  auto
  init_tmem_tensors(EpilogueTile epi_tile) {
    TiledMma tiled_mma;
    auto acc_shape = partition_accumulator_shape();
    // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N,ACC_PIPE) where ACC_PIPE=2 so we can double buffer our accumulators for mainloop and epilogue.
    Tensor accumulators = cutlass::detail::make_sm100_accumulator<AccumulatorPipelineStageCount, IsOverlappingAccum>(
        tiled_mma, acc_shape, EpilogueTile{});
    Tensor tCtSFA = make_tensor<typename TiledMma::FrgTypeSFA>(shape(SmemLayoutAtomSFA{}));
    Tensor tCtSFB = make_tensor<typename TiledMma::FrgTypeSFB>(shape(SmemLayoutAtomSFB{}));

    TmemStorage<decltype(accumulators), decltype(tCtSFA), decltype(tCtSFB)> tmem_storage;
    tmem_storage.accumulators = accumulators;
    tmem_storage.tCtSFA = tCtSFA;
    tmem_storage.tCtSFB = tCtSFB;

    return tmem_storage;
  }

  template <class TmemStorage>
  CUTLASS_DEVICE static
  void
  set_tmem_offsets(TmemStorage& tmem_storage, uint32_t tmem_base_addr) {
    tmem_storage.accumulators.data() = tmem_base_addr;
    tmem_storage.tCtSFA.data() = tmem_storage.accumulators.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tmem_storage.accumulators);
    tmem_storage.tCtSFB.data() = tmem_storage.tCtSFA.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tmem_storage.tCtSFA);
  }


  /// Set up the data needed by this collective for load.
  /// Return tuple element contain
  /// gA_mkl - The tiled tensor for input A
  /// gB_nkl - The tiled tensor for input B
  /// tAsA - partitioned smem tensor for A
  /// tBsB - partitioned smem tensor for B
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init_tma(
      ProblemShape_MNKL const& problem_shape_MNKL,
      TensorStorage& shared_tensors) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K,L));
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});    // (BLK_M, BLK_K, m, k, l)

    // Represent the full tensor of Scale factors
    Tensor mSFA_mkl = observed_tma_load_sfa_->get_tma_tensor(shape(layout_SFA_));
    auto mSFB_nkl = [=](){
      if constexpr (IsCtaN192) {
        Tensor mSFB_tmp = observed_tma_load_sfb_->get_tma_tensor(shape(layout_SFB_));
        auto x = stride<0,1>(mSFB_tmp);
        auto y = ceil_div(shape<0,1>(mSFB_tmp), 4);
        auto  new_shape =  make_shape (make_shape( shape<0,0>(mSFB_tmp),
                                       make_shape( make_shape(_2{}, _2{}),   y)),  shape<1>(mSFB_tmp), shape<2>(mSFB_tmp));
        auto new_stride = make_stride(make_stride(stride<0,0>(mSFB_tmp),
                                      make_stride(make_stride(   x,    x), x*3)), stride<1>(mSFB_tmp), stride<2>(mSFB_tmp));
        return make_tensor(mSFB_tmp.data(), make_layout(new_shape, new_stride));
      }
      else if constexpr (IsCtaN64) {
        Tensor mSFB_tmp = observed_tma_load_sfb_->get_tma_tensor(shape(layout_SFB_));
        auto new_shape = make_shape(make_shape(shape<0,0>(mSFB_tmp),
                                    make_shape(_2{} , shape<0,1>(mSFB_tmp))), shape<1>(mSFB_tmp), shape<2>(mSFB_tmp));
        auto new_stride = make_stride(make_stride(stride<0,0>(mSFB_tmp),
                                      make_stride(_0{}, stride<0,1>(mSFB_tmp))), stride<1>(mSFB_tmp), stride<2>(mSFB_tmp));
        return make_tensor(mSFB_tmp.data(), make_layout(new_shape, new_stride));
      }
      else {
        return observed_tma_load_sfb_->get_tma_tensor(shape(layout_SFB_));
      }
    }();

    Tensor gSFA_mkl = local_tile(mSFA_mkl, TileShape{},    make_coord(_,_,_), Step<_1, X,_1>{});  // (TILE_M,TILE_K,m,k,l)
    Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape_SF{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (TILE_N,TILE_K,n,k,l)


    ThrMMA cta_mma = TiledMma{}.get_slice(0);
    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);          // (MMA, MMA_M, MMA_K, m, k, l)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});  // (MMA,MMA_M,MMA_K,PIPE)

    ThrMMA cta_mma_sfb = TiledMma_SF{}.get_slice(blockIdx.x % size(typename TiledMma_SF::AtomThrID{}));
    Tensor tCgSFA_mkl = cta_mma.partition_A(gSFA_mkl);          // (MMA, MMA_M, MMA_K, m, k, l)
    Tensor tCgSFB_nkl = cta_mma_sfb.partition_B(gSFB_nkl);          // (MMA, MMA_N, MMA_K, n, k, l)

    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

    // Define the CTA-in-cluster Layout and Coord
    Layout cta_layout_mnk  = make_layout(ClusterShape{});
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(0);
    Layout cta_layout_sfb_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma_SF::AtomThrID{}));
    auto cta_coord_sfb_vmnk  = cta_layout_sfb_vmnk.get_flat_coord(0);

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));
    // Project the cta_layout for tma_a along the n-modes
    auto [tAgSFA_mkl, tAsSFA] = tma_partition(*observed_tma_load_sfa_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sSFA), group_modes<0,3>(tCgSFA_mkl));
    // Project the cta_layout for tma_b along the m-modes
    auto [tBgSFB_nkl, tBsSFB] = tma_partition(*observed_tma_load_sfb_,
                                      get<1>(cta_coord_sfb_vmnk), make_layout(size<1>(cta_layout_sfb_vmnk)),
                                      group_modes<0,3>(sSFB), group_modes<0,3>(tCgSFB_nkl));
                                      
    return cute::make_tuple(
      shape<3>(gA_mkl),      // for scheduler
      tAgA_mkl, tAsA,        // for input tensor values
      tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB // for input scale factor tensor values
    );
  }

  template <class ProblemShape_MNKL, class TileScheduler>
  CUTLASS_DEVICE auto
  load_init_cpasync(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_tensors,
      TileScheduler const& scheduler,
      typename TileScheduler::WorkTileInfo const& work_tile_info) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // convert to subptr iterator if necessary
    auto ptr_B = recast_ptr<ElementBMma>(params.ptr_B);
    // Represent the full tensors
    Tensor mB_nkl = make_tensor(make_gmem_ptr(ptr_B), make_shape(N,K,L), params.dB); //(n,k,l)
    // Partition for cpasync
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{}); // (BLK_N,BLK_K,n,k,l)

    // Build the coordinate tensors with the same shape as input matrices
    Tensor cB_nk  = make_identity_tensor(make_shape(N,K));
    // Slice the coordinate tensors in the same way as A/B tensor partitioning
    Tensor cgB_nk = local_tile(cB_nk, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{}); // (BLK_N,BLK_K,n,k)

    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), LoadSmemLayoutB{});

    GmemTiledCopyB gmem_to_smem_b_tiled_copy;

    int thread_idx = threadIdx.x % NumLoadThreadsCpAsync;
    auto thr_copy_b = gmem_to_smem_b_tiled_copy.get_slice(thread_idx);

    return cute::make_tuple(
      gB_nkl, cgB_nk, sB, 
      // problem_shape_MNKL, 
      gmem_to_smem_b_tiled_copy, thr_copy_b);
  }

  /// Set up the data needed by this collective for mma compute.
  template <class TmemStorage>
  CUTLASS_DEVICE auto
  mma_init(
      Params const& params,
      TmemStorage tmem_storage,
      // [[maybe_unused]] cute::tuple<cute::Tensor<FrgEngine, FrgLayout>, cute::Tensor<FrgEngine, FrgLayout>> const& accumulators_pair,
      TensorStorage& shared_tensors) const {
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), MmaSmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA = TiledMma::make_fragment_A(sA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = TiledMma::make_fragment_B(sB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sA));                                     // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sB));

    //
    // Scale Factor
    //
    Tensor tCtSFA = tmem_storage.tCtSFA;
    Tensor tCtSFB = tmem_storage.tCtSFB;
    // Setup smem descriptors for UTCCP
    Tensor tCsSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor tCsSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

    // Make SMEM and TMEM tensors compact removing the zero strides to eliminate unnecessary copy instructions.
    auto tCsSFA_compact = make_tensor(tCsSFA.data(), filter_zeros(tCsSFA.layout()));
    auto tCtSFA_compact = make_tensor(tCtSFA.data(), filter_zeros(tCtSFA.layout()));
    auto tCsSFB_compact = make_tensor(tCsSFB.data(), filter_zeros(tCsSFB.layout()));
    auto tCtSFB_compact = make_tensor(tCtSFB.data(), filter_zeros(tCtSFB.layout()));

    // Create the SMEM to TMEM copy operations based on the MMA atom used (1CTA vs 2CTA)
    using AtomThrID = typename TiledMma::AtomThrID;
    using UtccpOp = cute::conditional_t<(decltype(cute::size(AtomThrID{}) == Int<2>{})::value),
      SM100_UTCCP_4x32dp128bit_2cta, SM100_UTCCP_4x32dp128bit_1cta>;
    auto tiled_copy_s2t_SFA = make_utccp_copy(UtccpOp{}, tCtSFA_compact);
    auto tiled_copy_s2t_SFB = make_utccp_copy(UtccpOp{}, tCtSFB_compact);

    auto thr_copy_s2t_SFA = tiled_copy_s2t_SFA.get_slice(0);
    auto thr_tCsSFA_compact_s2t_ = thr_copy_s2t_SFA.partition_S(tCsSFA_compact);
    // SMEM to TMEM copy operation requires source SMEM operand to be an SMEM descriptor
    auto thr_tCsSFA_compact_s2t = get_utccp_smem_desc_tensor<UtccpOp>(thr_tCsSFA_compact_s2t_);
    auto thr_tCtSFA_compact_s2t = thr_copy_s2t_SFA.partition_D(tCtSFA_compact);

    auto thr_copy_s2t_SFB = tiled_copy_s2t_SFB.get_slice(0);
    auto thr_tCsSFB_compact_s2t_ = thr_copy_s2t_SFB.partition_S(tCsSFB_compact);
    // SMEM to TMEM copy operation requires source SMEM operand to be an SMEM descriptor
    auto thr_tCsSFB_compact_s2t = get_utccp_smem_desc_tensor<UtccpOp>(thr_tCsSFB_compact_s2t_);
    auto thr_tCtSFB_compact_s2t = thr_copy_s2t_SFB.partition_D(tCtSFB_compact);

    TiledMma tiled_mma;

    if constexpr (IsRuntimeDataType) {
      // Update instruction descriptor according to runtime argument.
      // Applying bitmask (0b111) to help compiler deduce that the conversion and assignment are safe.
      tiled_mma.idesc_.a_format_ = uint8_t(params.runtime_data_type_a) & 0b111;
      tiled_mma.idesc_.b_format_ = uint8_t(params.runtime_data_type_b) & 0b111;
    }

    return cute::make_tuple(
      tiled_mma, 
      tCrA, tCrB,
      tCtSFA, tCtSFB,
      tiled_copy_s2t_SFA, thr_tCsSFA_compact_s2t, thr_tCtSFA_compact_s2t,
      tiled_copy_s2t_SFB, thr_tCsSFB_compact_s2t, thr_tCtSFB_compact_s2t
      
      // debug
      // , sA, sB, tCsSFA, tCsSFB
    );
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    // class KTileCount,
    // class GTensorPartitionedA,
    // class STensorA,
    class TileCoordMNKL,
    class KTileIterator,
    class... TLoadParams  // see load_init_tma
  >
  CUTLASS_DEVICE auto
  load_tma(
    MainloopPipelineTMA mainloop_pipeline,
    MainloopPipelineTMAState mainloop_pipe_producer_state,
    cute::tuple<TLoadParams...> const& load_inputs,
    TileCoordMNKL const& cta_coord_mnkl,
    KTileIterator k_tile_iter, int k_tile_count) {
    
    // Unpack from load_inputs
    // KTileCount k_tiles = get<0>(load_inputs);
    // GTensorPartitionedA tAgA_mkl = get<1>(load_inputs);
    // STensorA tAsA = get<2>(load_inputs);

    auto [k_tiles,
          tAgA_mkl, tAsA,
          tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB] = load_inputs;

    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tAgSFA = tAgSFA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgSFB = tBgSFB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
    
    auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);

      using BarrierType = typename MainloopPipelineTMA::ProducerBarrierType;
      BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

      int write_stage = mainloop_pipe_producer_state.index();
      ++mainloop_pipe_producer_state;
      barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

      if (cute::elect_one_sync()) {
        copy(observed_tma_load_a_->with(*tma_barrier), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
        copy(observed_tma_load_sfa_->with(*tma_barrier), tAgSFA(_,*k_tile_iter), tAsSFA(_,write_stage));
        copy(observed_tma_load_sfb_->with(*tma_barrier), tBgSFB(_,*k_tile_iter), tBsSFB(_,write_stage));
      }

      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
  }


  template <
    // class GTensorB,
    // class CTensorB,
    // class STensorB,
    // class ProblemShape_MNKL,
    // class TiledCopyB,
    // class ThreadCopyB,
    class TileCoordMNKL,
    class KTileIterator,
    class ProblemShape_MNKL,
    class... TParams
  >
  CUTLASS_DEVICE auto
  load_cpasync(
    Params const& params,
    MainloopPipelineCpAsync mainloop_pipeline,
    MainloopPipelineCpAsyncState mainloop_pipe_producer_state,
    cute::tuple<TParams...> const& load_inputs,
    TileCoordMNKL const& cta_coord_mnkl,
    KTileIterator k_tile_iter, int k_tile_count,
    ProblemShape_MNKL effective_shape
  ) {

    // Unpack from load_inputs
    // GTensorB tBgB_nkl = get<0>(load_inputs);
    // CTensorB cgB_nk = get<1>(load_inputs);
    // STensorB sB = get<2>(load_inputs);
    // ProblemShape_MNKL problem_shape_MNKL = get<3>(load_inputs);
    // TiledCopyB gmem_to_smem_b_tiled_copy = get<4>(load_inputs);
    // ThreadCopyB thr_copy_b = get<5>(load_inputs);

    // auto [M,N,K,L] = problem_shape_MNKL;

    auto [
      tBgB_nkl, cgB_nk, sB, 
      // problem_shape_MNKL, 
      gmem_to_smem_b_tiled_copy, thr_copy_b] = load_inputs;

    auto [M,N,K,L] = effective_shape;

    // Slice out the work coord from partitioned tensors
    Tensor gB_in = tBgB_nkl(_, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
    // Repeat slicing out coordinate tensor exactly the same as input tensor does
    Tensor cgB_nk_in = cgB_nk(_, _, get<1>(cta_coord_mnkl), _);

    auto k_residue    = K - size<1>(gB_in) * size<2>(gB_in);  // K - BLK_K * k is negative

    Tensor gB = gB_in;
    Tensor cB = cgB_nk_in;

    auto tBgB = thr_copy_b.partition_S(gB);
    auto tBsB = thr_copy_b.partition_D(sB);

    // Allocate predicate tensors for n
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});
    Tensor tBcB_nk = thr_copy_b.partition_S(cgB_nk_in);
    Tensor tBcB = thr_copy_b.partition_S(cB);

    // Copy gmem to smem for *k_tile_iter, predicating for k residue
    Tensor tBgBk = tBgB(_,_,_,*k_tile_iter);

    // Repeating on predicators with the same operations on tBgB
    Tensor tBcBk = tBcB(_,_,_,*k_tile_iter);

    // Set predicates for n bounds
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<0>(tBpB); ++n) {
      tBpB(n,0) = elem_less(get<0>(tBcBk(0,n,0)), N);  // blk_n coord < N
    }

    // we will process the last tile after the mainloop
    if (k_residue != 0) {
      --k_tile_count;
    }

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {

      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state);
      int write_stage = mainloop_pipe_producer_state.index();

      copy_if(gmem_to_smem_b_tiled_copy, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

      mainloop_pipeline.producer_commit(mainloop_pipe_producer_state, cutlass::arch::cpasync_barrier_arrive);
      --k_tile_count;
      ++k_tile_iter;
      ++mainloop_pipe_producer_state;
    }
    
    // last tile with predication on k to account for residue
    // For performance consideration,
    // this predicated block for K-tail is only activated when there is k-residue
    if (k_residue != 0)  {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state);
      int write_stage = mainloop_pipe_producer_state.index();

      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tBsB); ++k) {
        if (int(get<1>(tBcBk(0,0,k))) >= 0) {      // blk_k coord < K
          copy_if(gmem_to_smem_b_tiled_copy, tBpB(_,k), tBgB(_,_,k,*k_tile_iter), tBsB(_,_,k,write_stage));
        }
        else {
          clear(tBsB(_,_,k,write_stage));
        }
      }
      ++k_tile_iter;
      --k_tile_count;

      // UNLOCK mainloop_pipe_producer_state
      mainloop_pipeline.producer_commit(mainloop_pipe_producer_state, cutlass::arch::cpasync_barrier_arrive);

      // Advance mainloop_pipe_producer_state
      ++mainloop_pipe_producer_state;
    }

    return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
  CUTLASS_DEVICE void
  load_tail_tma(MainloopPipelineTMA mainloop_pipeline, MainloopPipelineTMAState mainloop_pipe_producer_state) {
    // Issue the epilogue waits
    // This helps avoid early exit of ctas in Cluster
    // Waits for all stages to either be released (all
    // Consumer UNLOCKs), or if the stage was never used
    // then would just be acquired since the phase was
    // still inverted from make_producer_start_state
    mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
  }
  CUTLASS_DEVICE void
  load_tail_cpasync(MainloopPipelineCpAsync mainloop_pipeline, MainloopPipelineCpAsyncState mainloop_pipe_producer_state) {
    mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class AccumulatorPipeline,
    class FrgEngine, class FrgLayout,
    class CtaTileCoord,
    class... TMmaParams
  >
  CUTLASS_DEVICE auto
  mma(cute::tuple<MainloopPipelineTMA,
                  MainloopPipelineCpAsync,
                  AccumulatorPipeline> pipelines,
      cute::tuple<MainloopPipelineTMAState,
                  MainloopPipelineCpAsyncState,
                  typename AccumulatorPipeline::PipelineState> pipeline_states,
      cute::tuple<cute::Tensor<FrgEngine, FrgLayout>> const& accumulators_pair,
      cute::tuple<TMmaParams...> const& mma_inputs,
      CtaTileCoord cta_tile_coord,
      int k_tile_count
  ) {
    static_assert(is_tmem<FrgEngine>::value, "Accumulator must be tmem resident.");
    static_assert(rank(FrgLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");
    auto accumulators = get<0>(accumulators_pair);
    auto [tiled_mma, tCrA, tCrB, tCtSFA, tCtSFB,
          tiled_copy_s2t_SFA, thr_tCsSFA_s2t,
          thr_tCtSFA_s2t, tiled_copy_s2t_SFB,
          thr_tCsSFB_s2t, thr_tCtSFB_s2t

          // debug
          // , sA, sB, tCsSFA, tCsSFB
        ] = mma_inputs;

    auto [mainloop_pipeline_tma, mainloop_pipeline_cpasync, accumulator_pipeline] = pipelines;
    auto [mainloop_pipe_tma_consumer_state, mainloop_pipe_cpasync_consumer_state, accumulator_pipe_producer_state] = pipeline_states;

    auto tCtSFB_mma = [tCtSFB = tCtSFB, cta_tile_coord]() {
      if constexpr (IsCtaN192) {
        // If this is an ODD tile, shift the TMEM start address for N=192 case by two words (ignores first 64 columns of SFB)
        auto tCtSFB_tmp = tCtSFB;
        if (size<1>(cta_tile_coord) % 2 == 1) {
          tCtSFB_tmp.data() = tCtSFB_tmp.data().get() + 2;
        }
        return tCtSFB_tmp;
      }
      else if constexpr (IsCtaN64) {
        // Move in increments of 64 columns of SFB
        auto tCtSFB_tmp = tCtSFB;
        tCtSFB_tmp.data() = tCtSFB_tmp.data().get() + (size<1>(cta_tile_coord) % 2) * 2;
        return tCtSFB_tmp;
      }
      else {
        return tCtSFB;
      }
    }();

    // Wait for tmem accumulator buffer to become empty with a flipped phase
    accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);

    //
    // PIPELINED MAIN LOOP
    //
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      mainloop_pipeline_tma.consumer_wait(mainloop_pipe_tma_consumer_state);
      mainloop_pipeline_cpasync.consumer_wait(mainloop_pipe_cpasync_consumer_state);

      int read_stage_tma = mainloop_pipe_tma_consumer_state.index();
      int read_stage_cpasync = mainloop_pipe_cpasync_consumer_state.index();

      if (cute::elect_one_sync()) {
        copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_,_,_,_,read_stage_tma), thr_tCtSFA_s2t);
        copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_,_,_,_,read_stage_tma), thr_tCtSFB_s2t);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
                                  tCtSFA(_,_,k_block),
                                  tCtSFB_mma(_,_,k_block)), 
            tCrA(_,_,k_block,read_stage_tma), 
            tCrB(_,_,k_block,read_stage_cpasync), 
            accumulators);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }

      mainloop_pipeline_tma.consumer_release(mainloop_pipe_tma_consumer_state);
      mainloop_pipeline_cpasync.consumer_release(mainloop_pipe_cpasync_consumer_state);
      --k_tile_count;
      ++mainloop_pipe_tma_consumer_state;
      ++mainloop_pipe_cpasync_consumer_state;
    }

    return cute::make_tuple(mainloop_pipe_tma_consumer_state, mainloop_pipe_cpasync_consumer_state);
  }

protected:

  typename Params::TMA_A const* observed_tma_load_a_{nullptr};
  typename Params::TMA_SFA const* observed_tma_load_sfa_{nullptr};
  typename Params::TMA_SFB const* observed_tma_load_sfb_{nullptr};

  LayoutSFA layout_SFA_;
  LayoutSFB layout_SFB_;

  RuntimeDataTypeA runtime_data_type_a_{};
  RuntimeDataTypeB runtime_data_type_b_{};

  // ClusterShape cluster_shape_;
  // uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
