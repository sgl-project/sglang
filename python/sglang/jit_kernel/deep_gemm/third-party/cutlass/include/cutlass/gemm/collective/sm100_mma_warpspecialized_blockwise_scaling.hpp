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
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

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
  class ElementA_,
  class StridePairA_,
  class ElementB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling<
      Stages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape>,
    TileShape_,
    ElementA_,
    StridePairA_,
    ElementB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using TiledMma = TiledMma_;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;

  using DispatchPolicy = MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling<
                          Stages,
                          SchedulerPipelineStageCount,
                          AccumulatorPipelineStageCount,
                          ClusterShape>;
  using TileShape = TileShape_;

  using ElementA = ElementA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using StrideA = cute::remove_cvref_t<decltype(get<0>(StridePairA_{}))>;
  using LayoutSFA = cute::remove_cvref_t<decltype(get<1>(StridePairA_{}))>;
  using ElementSFA = typename TiledMma::ValTypeC;
  using ElementB = ElementB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StrideB = cute::remove_cvref_t<decltype(get<0>(StridePairB_{}))>;
  using LayoutSFB = cute::remove_cvref_t<decltype(get<1>(StridePairB_{}))>;
  using ElementSFB = typename TiledMma::ValTypeC;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  static constexpr int ScaleGranularityM = size<0,0>(LayoutSFA{});
  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static_assert(size<0>(TileShape{}) % ScaleGranularityM == 0 and ScaleGranularityM <= size<0>(TileShape{}), "Scale Granularity M must divide Tile Shape");

  static constexpr int ScaleGranularityN = size<0,0>(LayoutSFB{});
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;
  static_assert(size<1>(TileShape{}) % ScaleGranularityN == 0 and ScaleGranularityN <= size<1>(TileShape{}), "Scale Granularity N must divide Tile Shape");

  static_assert(size<1, 0>(LayoutSFA{}) == size<1, 0>(LayoutSFB{}), "Vector size K must be equal for SFA and SFB");

  static constexpr int ScaleGranularityK = size<1, 0>(LayoutSFA{});
  static constexpr int ScaleKsPerTile = size<2>(TileShape{}) / ScaleGranularityK;
  static_assert(size<2>(TileShape{}) % ScaleGranularityK == 0 and ScaleGranularityK <= size<2>(TileShape{}), "Scale Granularity K must divide Tile Shape");
  static_assert(ScaleGranularityK % size<2>(typename TiledMma::AtomShape_MNK{}) == 0, "Scale Granularity K must be divisible by MMA_K");

  static constexpr int K_BLOCK_MMAS_PER_SCALE_K = ScaleGranularityK / size<2>(typename TiledMma::AtomShape_MNK{});

  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      size<0,1>(LayoutSFA{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K,
      size<0,1>(LayoutSFB{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K>;

  CUTE_STATIC_ASSERT_V(evenly_divides(TileShape{}, tile_shape(TiledMma{})),
                       "Static cluster shape used: TileShape should be evenly divided by TiledMma");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  static_assert(size<0>(CtaShape_MNK{}) >= ScaleGranularityM, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<1>(CtaShape_MNK{}) >= ScaleGranularityN, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<2>(CtaShape_MNK{}) >= ScaleGranularityK, "Scale Granularity must be smaller than or equal to the tile shape");

  using SmemLayoutAtomSFA = decltype(ScaleConfig::smem_atom_layoutSFA(CtaShape_MNK{}));
  using SmemLayoutAtomSFB = decltype(ScaleConfig::smem_atom_layoutSFB(CtaShape_MNK{}));

  // Define A and B block shapes for reduced size TMA_LOADs
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();

  static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementB>();

  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
                (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
                "ElementA and ElementB should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = cute::remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA_{}))>;
  using GmemTiledCopySFA = cute::remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA_{}))>;
  using GmemTiledCopyB = cute::remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB_{}))>;
  using GmemTiledCopySFB = cute::remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB_{}))>;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopABPipeline = cutlass::PipelineTmaUmmaAsync<
                                DispatchPolicy::Stages,
                                ClusterShape,
                                AtomThrShapeMNK>;
  using MainloopABPipelineState = typename MainloopABPipeline::PipelineState;

  using MainloopSFPipeline = cutlass::PipelineAsync<DispatchPolicy::Stages>;
  using MainloopSFPipelineState = typename MainloopSFPipeline::PipelineState;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
                                  AccumulatorPipelineStageCount,
                                  AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  static constexpr int CopyAlignmentSFA = GmemTiledCopySFA::AtomNumVal::value * sizeof(typename GmemTiledCopySFA::ValType) / sizeof(ElementAccumulator);
  static constexpr int CopyAlignmentSFB = GmemTiledCopySFB::AtomNumVal::value * sizeof(typename GmemTiledCopySFB::ValType) / sizeof(ElementAccumulator);

  static constexpr int AlignmentSFA = CopyAlignmentSFA * (GmemTiledCopySFA::AtomNumVal::value > 1 ?
      (size<0,1>(LayoutSFA{}.stride()) == 1 ? ScaleGranularityM : ScaleGranularityK) : 1);
  static constexpr int AlignmentSFB = CopyAlignmentSFB * (GmemTiledCopySFB::AtomNumVal::value > 1 ?
      (size<0,1>(LayoutSFB{}.stride()) == 1 ? ScaleGranularityN : ScaleGranularityK) : 1);


  // Two arrivals per thread in the warp (1 arrival and 1 arrival through cp.async.mbarrier)
  static constexpr int NumMainloopSFProducerThreadEvents = 64;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtomA must be rank 2 (M,K)");
  static_assert(((size<0,0>(MmaShapeA_MK{}) * size<1>(MmaShapeA_MK{})) % size<0>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(((size<0,1>(MmaShapeA_MK{}) * size<2>(MmaShapeA_MK{})) % size<1>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtomB must be rank 2 (N,K)");
  static_assert(((size<0,0>(MmaShapeB_NK{}) * size<1>(MmaShapeB_NK{})) % size<0>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(((size<0,1>(MmaShapeB_NK{}) * size<2>(MmaShapeB_NK{})) % size<1>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  // (MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE)
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  // (MMA_TILE_N,MMA_TILE_K),MMA_N,MMA_K,PIPE)
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(
      (size(AtomThrShapeMNK{}) == 1 &&
        (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>)) ||
      (size(AtomThrShapeMNK{}) == 2 &&
        (cute::is_same_v<GmemTiledCopyA, SM100_TMA_2SM_LOAD> || cute::is_same_v<GmemTiledCopyA, SM100_TMA_2SM_LOAD_MULTICAST>)),
      "GmemTiledCopy - invalid TMA copy atom specified.");
  static_assert(
      (size(AtomThrShapeMNK{}) == 1 &&
        (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>)) ||
      (size(AtomThrShapeMNK{}) == 2 &&
        (cute::is_same_v<GmemTiledCopyB, SM100_TMA_2SM_LOAD> || cute::is_same_v<GmemTiledCopyB, SM100_TMA_2SM_LOAD_MULTICAST>)),
      "GmemTiledCopy -  invalid TMA copy atom specified.");

  using TmaInternalElementA = cute::conditional_t<cute::is_same_v<ElementA, float>, cutlass::tfloat32_t, ElementAMma>;
  using TmaInternalElementB = cute::conditional_t<cute::is_same_v<ElementB, float>, cutlass::tfloat32_t, ElementBMma>;

  using SmemAllocTypeA = cute::conditional_t<cute::sizeof_bits_v<ElementAMma> < 8, uint8_t, ElementAMma>;
  using SmemAllocTypeB = cute::conditional_t<cute::sizeof_bits_v<ElementBMma> < 8, uint8_t, ElementBMma>;

  using BitTypeElementA = cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>;
  using BitTypeElementB = cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>;

  using ArrayElementA = cute::conditional_t<IsRuntimeDataTypeA, BitTypeElementA, ElementA>;
  using ArrayElementB = cute::conditional_t<IsRuntimeDataTypeB, BitTypeElementB, ElementB>;

  using RuntimeDataTypeA = cute::conditional_t<IsRuntimeDataTypeA, cute::UMMA::MXF8F6F4Format, void*>;
  using RuntimeDataTypeB = cute::conditional_t<IsRuntimeDataTypeB, cute::UMMA::MXF8F6F4Format, void*>;

  using SmemLayoutScaleA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));
  using SmemLayoutScaleB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::ArrayEngine<ElementAccumulator, cute::cosize_v<SmemLayoutScaleA>> smem_SFA;
      cute::ArrayEngine<ElementAccumulator, cute::cosize_v<SmemLayoutScaleB>> smem_SFB;
    } tensors;

    using PipelineABStorage = typename MainloopABPipeline::SharedStorage;
    using PipelineSFStorage = typename MainloopSFPipeline::SharedStorage;
    using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;

    struct PipelineStorage {
      alignas(16) PipelineABStorage pipeline_ab;
      alignas(16) PipelineSFStorage pipeline_sf;
      alignas(16) AccumulatorPipelineStorage pipeline_accum;
    };
  };

  // Expose shared storage for tensors/pipelines separately to allow kernel layer to reorder them.
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Only one thread issues the TMA and updates the barriers in a 2SM MMA, adjust bytes accordingly
  static constexpr uint32_t TmaTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>) +
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>);

  template<class AccTensor>
  struct TmemStorage {
    AccTensor accumulators;
  };

  template<
    class KTileCount,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB
  >
  struct LoadABParams {
    // for scheduler
    KTileCount k_tiles;
    // for input tensor values
    GTensorPartitionedA tAgA_mkl;
    GTensorPartitionedB tBgB_nkl;
    STensorA tAsA;
    STensorB tBsB;

    // the TMA multicast masks
    uint16_t mcast_mask_a;
    uint16_t mcast_mask_b;

    CUTLASS_DEVICE
    LoadABParams (
        KTileCount k_tiles_,
        GTensorPartitionedA tAgA_mkl_, GTensorPartitionedB tBgB_nkl_,
        STensorA tAsA_, STensorB tBsB_,
        uint16_t mcast_mask_a_, uint16_t mcast_mask_b_)
    : k_tiles(k_tiles_)
    , tAgA_mkl(tAgA_mkl_), tBgB_nkl(tBgB_nkl_)
    , tAsA(tAsA_), tBsB(tBsB_)
    , mcast_mask_a(mcast_mask_a_), mcast_mask_b(mcast_mask_b_) {}
  };

  template<
    class KTileCount,
    class GTensorScaleA, class GTensorScaleB,
    class IdentTensorScaleA, class IdentTensorScaleB,
    class STensorScaleA, class STensorScaleB
  >
  struct LoadSFParams {
    // for scheduler
    KTileCount k_tiles;

    GTensorScaleA gSFA_mkl;
    GTensorScaleB gSFB_nkl;
    IdentTensorScaleA identSFA_mkl;
    IdentTensorScaleB identSFB_nkl;
    STensorScaleA sSFA;
    STensorScaleB sSFB;

    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;

    CUTLASS_DEVICE
    LoadSFParams (
        KTileCount k_tiles_,
        GTensorScaleA gSFA_mkl_, GTensorScaleB gSFB_nkl_,
        IdentTensorScaleA identSFA_mkl_, IdentTensorScaleB identSFB_nkl_,
        STensorScaleA sSFA_, STensorScaleB sSFB_,
        LayoutSFA layout_SFA_, LayoutSFB layout_SFB_)
    : k_tiles(k_tiles_)
    , gSFA_mkl(gSFA_mkl_), gSFB_nkl(gSFB_nkl_)
    , identSFA_mkl(identSFA_mkl_), identSFB_nkl(identSFB_nkl_)
    , sSFA(sSFA_), sSFB(sSFB_)
    , layout_SFA(layout_SFA_), layout_SFB(layout_SFB_) {}
  };

  template<class FragmentA, class FragmentB>
  struct MmaParams {
    TiledMma tiled_mma;
    FragmentA tCrA;
    FragmentB tCrB;

    CUTLASS_DEVICE
    MmaParams (
        TiledMma tiled_mma_,
        FragmentA tCrA_, FragmentB tCrB_)
    : tiled_mma(tiled_mma_)
    , tCrA(tCrA_), tCrB(tCrB_) {}
  };

  template<
    class STensorScaleA, class STensorScaleB
  >
  struct AccumTransformParams {
    // for scheduler

    STensorScaleA sSFA;
    STensorScaleB sSFB;

    CUTLASS_DEVICE
    AccumTransformParams (
        STensorScaleA sSFA_, STensorScaleB sSFB_)
    :  sSFA(sSFA_), sSFB(sSFB_) {}
  };


  // Host side kernel arguments
  struct Arguments {
    ArrayElementA const* ptr_A{nullptr};
    StrideA dA{};
    ArrayElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementAccumulator const* ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    ElementAccumulator const* ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
    RuntimeDataTypeA runtime_data_type_a{};
    RuntimeDataTypeB runtime_data_type_b{};
  };

  // Device side kernel params
  struct Params {
    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
                                                     make_tile(typename TiledMma::AtomThrID{})));

    using TMA_A = decltype(make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    using TMA_B = decltype(make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_A tma_load_a_fallback;
    TMA_B tma_load_b_fallback;
    dim3 cluster_shape_fallback;
    RuntimeDataTypeA runtime_data_type_a;
    RuntimeDataTypeB runtime_data_type_b;

    ElementAccumulator const* ptr_SFA;
    LayoutSFA layout_SFA;
    ElementAccumulator const* ptr_SFB;
    LayoutSFB layout_SFB;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
    : cluster_shape_(cluster_shape)
    , block_rank_in_cluster_(block_rank_in_cluster)
    , runtime_data_type_a_(params.runtime_data_type_a)
    , runtime_data_type_b_(params.runtime_data_type_b) {
    if constexpr (IsDynamicCluster) {
      const bool is_fallback_cluster = (cute::size<0>(cluster_shape_) == params.cluster_shape_fallback.x &&
                                        cute::size<1>(cluster_shape_) == params.cluster_shape_fallback.y);
      observed_tma_load_a_ = is_fallback_cluster ? &params.tma_load_a_fallback : &params.tma_load_a;
      observed_tma_load_b_ = is_fallback_cluster ? &params.tma_load_b_fallback : &params.tma_load_b;
    }
    else {
      observed_tma_load_a_ = &params.tma_load_a;
      observed_tma_load_b_ = &params.tma_load_b;
    }
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
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);

    // Cluster layout for TMA construction
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));
    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));
    typename Params::TMA_A tma_load_a = make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_B tma_load_b = make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_A tma_load_a_fallback = make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    typename Params::TMA_B tma_load_b_fallback = make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    return {
      tma_load_a,
      tma_load_b,
      tma_load_a_fallback,
      tma_load_b_fallback,
      hw_info.cluster_shape_fallback,
      args.runtime_data_type_a,
      args.runtime_data_type_b,
      args.ptr_SFA,
      args.layout_SFA,
      args.ptr_SFB,
      args.layout_SFB
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    static constexpr bool IsF8F6F4 = detail::is_sm100_mma_f8f6f4<TiledMma, ElementA, ElementB>();
    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cute::sizeof_bits<ElementA>::value;

    bool implementable = true;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cute::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }

    bool implementable_sf = cutlass::detail::check_alignment<CopyAlignmentSFA>(args.layout_SFA);
    implementable_sf = implementable_sf && cutlass::detail::check_alignment<CopyAlignmentSFB>(args.layout_SFB);

    if (!implementable_sf) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for Scale Factors.\n");
    }

    return implementable && implementable_sf;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE void
  prefetch_tma_descriptors() {
    cute::prefetch_tma_descriptor(observed_tma_load_a_->get_tma_descriptor());
    cute::prefetch_tma_descriptor(observed_tma_load_b_->get_tma_descriptor());
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE static
  auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));     // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

    return acc_shape;
  }

  template <class TmemStorage>
  CUTLASS_DEVICE static
  auto
  slice_accumulator(TmemStorage tmem_storage, int stage) {
    return cute::make_tuple(tmem_storage.accumulators(_,_,_,stage));
  }

  template<class EpilogueTile, bool IsOverlappingAccum = false>
  CUTLASS_DEVICE static
  auto
  init_tmem_tensors(EpilogueTile epi_tile) {
    TiledMma tiled_mma;
    auto acc_shape = partition_accumulator_shape();
    // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N,ACC_PIPE) where ACC_PIPE=2 so we can double buffer our accumulators for mainloop and epilogue.
    Tensor accumulators = cutlass::detail::make_sm100_accumulator<AccumulatorPipelineStageCount, IsOverlappingAccum>(
        tiled_mma, acc_shape, EpilogueTile{});
    TmemStorage<decltype(accumulators)> tmem_storage;
    tmem_storage.accumulators = accumulators;
    return tmem_storage;
  }

  template<class AccTensor>
  CUTLASS_DEVICE static
  void
  set_tmem_offsets(TmemStorage<AccTensor>& tmem_storage, uint32_t tmem_base_addr) {
    tmem_storage.accumulators.data() = tmem_base_addr;
  }

  /// Set up the data needed by this collective for load.
  /// Return load params containing
  /// gA_mkl - The tiled tma tensor for input A
  /// gB_nkl - The tiled tma tensor for input B
  /// tAsA - partitioned smem tensor for A
  /// tBsB - partitioned smem tensor for B
  /// mcast_mask_a - tma multicast mask for A
  /// mcast_mask_b - tma multicast mask for B
  template <class ProblemShape_MNKL,
            class MainloopParams>
  CUTLASS_DEVICE auto
  load_ab_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K,L));
    Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N,K,L));

    // Tile the tensors and defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});     // (BLK_M, BLK_K, m, k, l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N, BLK_K, n, k, l)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);                                       // (MMA, MMA_M, MMA_K, m, k, l)
    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);                                       // (MMA, MMA_N, MMA_K, n, k, l)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});      // (MMA,MMA_M,MMA_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});      // (MMA,MMA_N,MMA_K,PIPE)

    // Define the CTA-in-cluster Layout and Coord
    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgB_nkl, tBsB] = tma_partition(*observed_tma_load_b_,
                                      get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                      group_modes<0,3>(sB), group_modes<0,3>(tCgB_nkl));

    // TMA Multicast Masks
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    LoadABParams load_params {
      shape<3>(gA_mkl),                               // for scheduler
      tAgA_mkl, tBgB_nkl, tAsA, tBsB,                 // for input tensor values
      mcast_mask_a, mcast_mask_b,                     // multicast masks
    };
    return load_params;
  }

  /// Set up the data needed by this collective for load.
  /// Return load params containing
  /// tSFAgSFA_mkl - partitioned gmem tensor for SFA
  /// tSFBgSFB_nkl - partitioned gmem tensor for SFB
  /// tSFAIdentSFA_mkl - partitioned identity tensor for SFA in gmem
  /// tSFBIdentSFB_nkl - partitioned identity tensor for SFB in gmem
  /// tSFAsSFA - partitioned smem tensor for SFA
  /// tSFBsSFB - partitioned smem tensor for SFB
  /// layout_SFA - layout of SFA in gmem
  /// layout_SFB - layout of SFB in gmem
  template <class ProblemShape_MNKL,
            class MainloopParams>
  CUTLASS_DEVICE auto
  load_sf_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor mSFA_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_SFA), mainloop_params.layout_SFA);    // (m,k,l)
    Tensor mSFB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_SFB), mainloop_params.layout_SFB);    // (n,k,l)

    Tensor SFA_mkl_ident = make_identity_tensor(shape(mainloop_params.layout_SFA));

    Tensor SFB_nkl_ident = make_identity_tensor(shape(mainloop_params.layout_SFB));

    // Tile the tensors and defer the slice
    Tensor gSFA_mkl = local_tile(mSFA_mkl, CtaShape_MNK{},
        make_coord(_,_,_), Step<_1, X,_1>{});                                                 // (BLK_M, BLK_K, m, k, l)
    Tensor gSFB_nkl = local_tile(mSFB_nkl, CtaShape_MNK{},
        make_coord(_,_,_), Step< X,_1,_1>{});                                                 // (BLK_N, BLK_K, n, k, l)

    Tensor identSFA_mkl = local_tile(SFA_mkl_ident, CtaShape_MNK{},
        make_coord(_,_,_), Step<_1, X,_1>{});                                                 // (BLK_M, BLK_K, m, k, l)
    Tensor identSFB_nkl = local_tile(SFB_nkl_ident, CtaShape_MNK{},
        make_coord(_,_,_), Step< X,_1,_1>{});                                                 // (BLK_N, BLK_K, n, k, l)

    static_assert(rank(decltype(gSFA_mkl){}) == 5);
    static_assert(rank(decltype(gSFB_nkl){}) == 5);

    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()),
        SmemLayoutScaleA{});                                                                          // (CTA_M,CTA_K,P)
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()),
        SmemLayoutScaleB{});                                                                          // (CTA_M,CTA_K,P)

    LoadSFParams load_params {
      size<3>(gSFA_mkl),
      gSFA_mkl, gSFB_nkl,                             // for input scale tensor values
      identSFA_mkl, identSFB_nkl,                     // for predicating scale tensor copies
      sSFA, sSFB,                                     // for scale tensor values
      mainloop_params.layout_SFA,                     // for predicating scale tensor copies
      mainloop_params.layout_SFB                      // for predicating scale tensor copies
    };
    return load_params;
  }


  /// Set up the data needed by this collective for mma compute.
  template <class AccTensor>
  CUTLASS_DEVICE auto
  mma_init(
      [[maybe_unused]] TmemStorage<AccTensor> tmem_tensors,
      TensorStorage& shared_tensors) const {
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA_ = TiledMma::make_fragment_A(sA);                                              // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB_ = TiledMma::make_fragment_B(sB);                                              // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(rank(tCrA_) == _4{});

    auto mma_tile_shape_A = make_shape(get<0>(shape(tCrA_.layout())),
                                       get<1>(shape(tCrA_.layout())),
                                       Int<K_BLOCK_MMAS_PER_SCALE_K>{},
                                       _1{});

    auto mma_tile_shape_B = make_shape(get<0>(shape(tCrB_.layout())),
                                       get<1>(shape(tCrB_.layout())),
                                       Int<K_BLOCK_MMAS_PER_SCALE_K>{},
                                       _1{});

    Tensor tCrA = flat_divide(tCrA_,
        mma_tile_shape_A)(_,_,_,_0{},_0{},_0{},_,_);                      // (MMA,MMA_M,MMA_K_PER_SCALE,MMA_K_REST,PIPE)

    Tensor tCrB = flat_divide(tCrB_,
        mma_tile_shape_B)(_,_,_,_0{},_0{},_0{},_,_);                      // (MMA,MMA_N,MMA_K_PER_SCALE,MMA_K_REST,PIPE)


    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sA));                                          // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sB));

    TiledMma tiled_mma;

    if constexpr (IsRuntimeDataType) {
      // Update instruction descriptor according to runtime argument.
      // Applying bitmask (0b111) to help compiler deduce that the conversion and assignment are safe.
      tiled_mma.idesc_.a_format_ = uint8_t(runtime_data_type_a_) & 0b111;
      tiled_mma.idesc_.b_format_ = uint8_t(runtime_data_type_b_) & 0b111;
    }
    MmaParams<decltype(tCrA), decltype(tCrB)> mma_params {
      tiled_mma,
      tCrA, tCrB
    };
    return mma_params;
  }

  /// Set up the data needed by this collective for transform.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  accum_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor sSFA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFA.begin()),
        SmemLayoutScaleA{});                                                        // (ScaleMsPerTile,ScakeKsPerTile,P)
    Tensor sSFB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFB.begin()),
        SmemLayoutScaleB{});                                                        // (ScaleNsPerTile,ScaleKsPerTile,P)


    AccumTransformParams transform_params {
      sSFA, sSFB                        // for input tensor values
    };
    return transform_params;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class LoadABParams,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load_ab(
      MainloopABPipeline mainloop_pipeline,
      MainloopABPipelineState mainloop_pipe_producer_state,
      LoadABParams const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_k_tiles,
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b] = load_inputs;

    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);

      using BarrierType = typename MainloopABPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

      int write_stage = mainloop_pipe_producer_state.index();
      auto curr_mainloop_pipe_producer_state = mainloop_pipe_producer_state;
      ++mainloop_pipe_producer_state;
      barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

      if (cute::elect_one_sync()) {
        copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
        copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
      }

      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
  CUTLASS_DEVICE void
  load_ab_tail(
      MainloopABPipeline mainloop_pipeline,
      MainloopABPipelineState mainloop_pipe_producer_state) {
    // Issue the epilogue waits
    // This helps avoid early exit of ctas in Cluster
    // Waits for all stages to either be released (all
    // Consumer UNLOCKs), or if the stage was never used
    // then would just be acquired since the phase was
    // still inverted from make_producer_start_state
    mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
  }

  /// Perform a collective-scoped transform
  /// Load producer Perspective
  template <
    class LoadSFParams,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load_sf(
      MainloopSFPipeline mainloop_sf_pipeline,
      MainloopSFPipelineState mainloop_sf_pipe_producer_state,
      LoadSFParams const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_k_tiles,
          gSFA_mkl, gSFB_nkl,
          identSFA_mkl, identSFB_nkl,
          sSFA, sSFB,
          layout_SFA, layout_SFB] = load_inputs;

    // slice out the work coord from partitioned tensors
    GmemTiledCopySFA scale_copy_a{};
    GmemTiledCopySFB scale_copy_b{};

    Tensor gSFA_k_compact = filter_zeros(
      gSFA_mkl(_, _, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl)));               // (BLK_M_CPT, BLK_K_CPT, k_cpt)
    Tensor gSFB_k_compact = filter_zeros(
      gSFB_nkl(_, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl)));               // (BLK_N_CPT, BLK_K_CPT, k_cpt)

    Tensor identSFA_k_compact = filter_zeros(
        identSFA_mkl(_, _, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl)), 
        gSFA_k_compact.stride());                                                       // (BLK_M_CPT, BLK_K_CPT, k_cpt)
    Tensor identSFB_k_compact = filter_zeros(
        identSFB_nkl(_, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl)), 
        gSFB_k_compact.stride());                                                       // (BLK_N_CPT, BLK_K_CPT, k_cpt)

    Tensor sSFA_compact = filter_zeros(sSFA);                                               // (BLK_M_CPT, BLK_K_CPT, P)
    Tensor sSFB_compact = filter_zeros(sSFB);                                               // (BLK_N_CPT, BLK_K_CPT, P)

    ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(threadIdx.x % size(scale_copy_a));
    ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(threadIdx.x % size(scale_copy_b));

    Tensor tSFAgSFA_k_compact = thr_scale_copy_a.partition_S(gSFA_k_compact);                  // (CPY, BLK_M, BLK_K, k)
    Tensor tSFAIdentSFA_k_compact = thr_scale_copy_a.partition_S(identSFA_k_compact);          // (CPY, BLK_M, BLK_K, k)

    Tensor tSFAsSFA_compact = thr_scale_copy_a.partition_D(sSFA_compact);

    Tensor tSFBgSFB_k_compact = thr_scale_copy_b.partition_S(gSFB_k_compact);                  // (CPY, BLK_N, BLK_K, k)
    Tensor tSFBIdentSFB_k_compact = thr_scale_copy_b.partition_S(identSFB_k_compact);          // (CPY, BLK_N, BLK_K, k)
    Tensor tSFBsSFB_compact = thr_scale_copy_b.partition_D(sSFB_compact);

    Tensor thr_tile_pSFA = make_fragment_like<bool>(tSFAgSFA_k_compact(_0{},_,_,_0{}));
    Tensor thr_tile_pSFB = make_fragment_like<bool>(tSFBgSFB_k_compact(_0{},_,_,_0{}));

    // Issue the loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK pipe_producer_state for _writing_
      mainloop_sf_pipeline.producer_acquire(mainloop_sf_pipe_producer_state);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(thr_tile_pSFA); ++i) {
        Tensor tSFAIdentSFA_compact = tSFAIdentSFA_k_compact(_0{},_,_,*k_tile_iter);
        thr_tile_pSFA(i) = elem_less(tSFAIdentSFA_compact(i), 
            shape(filter_zeros(layout_SFA))) && threadIdx.x % 32 < size(scale_copy_a);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(thr_tile_pSFB); ++i) {
        Tensor tSFBIdentSFB_compact = tSFBIdentSFB_k_compact(_0{},_,_,*k_tile_iter);
        thr_tile_pSFB(i) = elem_less(tSFBIdentSFB_compact(i), 
            shape(filter_zeros(layout_SFB))) && threadIdx.x % 32 < size(scale_copy_b);
      }

      copy_if(scale_copy_a, thr_tile_pSFA, tSFAgSFA_k_compact(_,_,_,*k_tile_iter), 
          tSFAsSFA_compact(_,_,_,mainloop_sf_pipe_producer_state.index()));
      copy_if(scale_copy_b, thr_tile_pSFB, tSFBgSFB_k_compact(_,_,_,*k_tile_iter), 
          tSFBsSFB_compact(_,_,_,mainloop_sf_pipe_producer_state.index()));
      mainloop_sf_pipeline.producer_commit(mainloop_sf_pipe_producer_state, cutlass::arch::cpasync_barrier_arrive_noinc);

      __syncwarp();

      ++mainloop_sf_pipe_producer_state;
      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_sf_pipe_producer_state, k_tile_iter);
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
  CUTLASS_DEVICE void
  load_sf_tail(
      MainloopSFPipeline mainloop_sf_pipeline,
      MainloopSFPipelineState mainloop_sf_pipe_producer_state) {
    // Issue the epilogue waits
    // This helps avoid early exit of ctas in Cluster
    // Waits for all stages to either be released (all
    // Consumer UNLOCKs), or if the stage was never used
    // then would just be acquired since the phase was
    // still inverted from make_producer_start_state
    mainloop_sf_pipeline.producer_tail(mainloop_sf_pipe_producer_state);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class TmemStorage,
    class MmaParams,
    class CtaTileCoord
  >
  CUTLASS_DEVICE auto
  mma(
      cute::tuple<MainloopABPipeline,
                  AccumulatorPipeline> pipelines,
      cute::tuple<MainloopABPipelineState,
                  AccumulatorPipelineState> pipeline_states,
      TmemStorage tmem_storage,
      MmaParams const& mma_inputs,
      CtaTileCoord cta_tile_coord,
      int k_tile_count) {
    auto [tiled_mma, tCrA, tCrB] = mma_inputs;

    auto [mainloop_pipeline,
          accumulator_pipeline] = pipelines;

    auto [mainloop_pipe_consumer_state,
          accumulator_pipe_producer_state] = pipeline_states;

    uint32_t skip_wait = k_tile_count <= 0;
    auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

    //
    // PIPELINED MAIN LOOP
    //
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // WAIT on mainloop_pipe_consumer_state until its data are available
      // (phase bit flips from mainloop_pipe_consumer_state.phase() value)
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state);

      // Compute on k_tile
      int read_stage = mainloop_pipe_consumer_state.index();
      // Save current mainlop pipeline read state
      auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

      // Advance mainloop_pipe
      ++mainloop_pipe_consumer_state;
      --k_tile_count;
      skip_wait = k_tile_count <= 0;
      // Peek at next iteration
      barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

      CUTLASS_PRAGMA_UNROLL
      for (int scale_k_iter = 0; scale_k_iter < size<3>(tCrA); ++scale_k_iter) {
        accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);

        auto acc = get<0>(slice_accumulator(tmem_storage, accumulator_pipe_producer_state.index()));
        static_assert(is_tmem<remove_cvref_t<decltype(acc)>>::value, "Accumulator must be tmem resident.");
        static_assert(rank(remove_cvref_t<decltype(acc)>{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

        // for each set of scale_k_blocks we zero the accumulator
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        // Unroll the K mode manually so we can set scale C to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          // (V,M) x (V,N) => (V,M,N)
          cute::gemm(tiled_mma,
                     tCrA(_,_,k_block,scale_k_iter,read_stage),
                     tCrB(_,_,k_block,scale_k_iter,read_stage),
                     acc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
        ++accumulator_pipe_producer_state;
      }
      mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);

    }

    return make_tuple(mainloop_pipe_consumer_state, accumulator_pipe_producer_state);
  }

  /// Transform
  template <
    class AccumTransformParams,
    class TmemStorage,
    class CtaTileCoord,
    class CopyOpT2R,
    class EpilogueTile
  >
  CUTLASS_DEVICE auto
  accum(
      cute::tuple<AccumulatorPipeline, MainloopSFPipeline> pipelines,
      cute::tuple<AccumulatorPipelineState, MainloopSFPipelineState> consumer_states,
      TmemStorage tmem_storage,
      AccumTransformParams const& transform_inputs,
      CtaTileCoord cta_tile_coord,
      CopyOpT2R,
      EpilogueTile,
      int k_tile_count) {

    static_assert(size<0>(EpilogueTile{}) <= size<0>(CtaShape_MNK{}), "Restrict epilogue tile to be smaller than or equal to CTA Tile");
    static_assert(size<1>(EpilogueTile{}) <= size<1>(CtaShape_MNK{}), "Restrict epilogue tile to be smaller than or equal to CTA Tile");


    //
    // PIPELINED Transform
    //

    Tensor acc = get<0>(slice_accumulator(tmem_storage, _0{}));

    Tensor tAcc = acc(make_coord(_,_),_0{},_0{});

    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                          // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    // Append N with a stride of 0 to SFA
    Tensor sSFA_ = transform_inputs.sSFA;
    Tensor sSFA = make_tensor(sSFA_.data(), make_layout(
      make_shape(get<0>(sSFA_.shape()), get<1>(CtaShape_MNK{}), get<1>(sSFA_.shape()), get<2>(sSFA_.shape())),
      make_stride(get<0>(sSFA_.stride()), _0{}, get<1>(sSFA_.stride()), get<2>(sSFA_.stride()))
    ));

    CUTE_STATIC_ASSERT_V(size<0>(sSFA) == size<0>(tAcc));
    CUTE_STATIC_ASSERT_V(size<1>(sSFA) == size<1>(tAcc));

    Tensor sSFA_epi = flat_divide(sSFA, EpilogueTile{});

    // Append M with a stride of 0 to SFB
    Tensor sSFB_ = transform_inputs.sSFB;
    Tensor sSFB = make_tensor(sSFB_.data(), make_layout(
      make_shape(get<0>(CtaShape_MNK{}), get<0>(sSFB_.shape()), get<1>(sSFB_.shape()), get<2>(sSFB_.shape())),
      make_stride(_0{}, get<0>(sSFB_.stride()), get<1>(sSFB_.stride()), get<2>(sSFB_.stride()))
    ));

    CUTE_STATIC_ASSERT_V(size<0>(sSFB) == size<0>(tAcc));
    CUTE_STATIC_ASSERT_V(size<1>(sSFB) == size<1>(tAcc));

    Tensor sSFB_epi = flat_divide(sSFB, EpilogueTile{});

    TiledCopy tiled_t2r_epi = make_tmem_copy(CopyOpT2R{}, tAcc_epi(_,_,_0{},_0{}));

    int thread_idx = threadIdx.x % size(tiled_t2r_epi);

    ThrCopy thread_t2r_epi = tiled_t2r_epi.get_slice(thread_idx);

    Tensor acc_ident_epi = make_identity_tensor(shape(tAcc_epi));

    Tensor tTR_rAcc_epi = thread_t2r_epi.partition_D(acc_ident_epi);                // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)

    Tensor tTR_sSFA_epi = thread_t2r_epi.partition_D(sSFA_epi);                     // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    Tensor tTR_sSFB_epi = thread_t2r_epi.partition_D(sSFB_epi);                     // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)

    static_assert(rank(decltype(tTR_sSFA_epi){}) == 7);

    Tensor tTR_FullAcc = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi));
    Tensor tTR_PartAcc = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi(_,_,_,_0{},_0{})));

    Tensor tTR_rSFA_compact = make_fragment_like<ElementAccumulator>(filter_zeros(tTR_sSFA_epi(_,_,_,_,_,_,_0{})));
    Tensor tTR_rSFB_compact = make_fragment_like<ElementAccumulator>(filter_zeros(tTR_sSFB_epi(_,_,_,_,_,_,_0{})));

    Layout tTR_rSFA_layout = make_layout(tTR_sSFA_epi(_,_,_,_,_,_,_0{}).shape(), tTR_rSFA_compact.stride());
    Layout tTR_rSFB_layout = make_layout(tTR_sSFB_epi(_,_,_,_,_,_,_0{}).shape(), tTR_rSFB_compact.stride());

    // Zero our accumulator
    clear(tTR_FullAcc);

    auto [accumulator_pipeline, mainloop_sf_pipeline] = pipelines;
    auto [accumulator_pipe_state, mainloop_sf_pipe_state] = consumer_states;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {

      mainloop_sf_pipeline.consumer_wait(mainloop_sf_pipe_state);
      int read_idx = mainloop_sf_pipe_state.index();

      copy(filter_zeros(tTR_sSFA_epi(_,_,_,_,_,_,read_idx)), tTR_rSFA_compact);
      copy(filter_zeros(tTR_sSFB_epi(_,_,_,_,_,_,read_idx)), tTR_rSFB_compact);

      CUTE_STATIC_ASSERT_V(cosize(tTR_rSFA_layout) == size(tTR_rSFA_compact));
      CUTE_STATIC_ASSERT_V(cosize(tTR_rSFB_layout) == size(tTR_rSFB_compact));

      Tensor tTR_rSFA = make_tensor(tTR_rSFA_compact.data(), tTR_rSFA_layout);
      Tensor tTR_rSFB = make_tensor(tTR_rSFB_compact.data(), tTR_rSFB_layout);

      mainloop_sf_pipeline.consumer_release(mainloop_sf_pipe_state);
      ++mainloop_sf_pipe_state;

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < ScaleKsPerTile; ++k_block) {

        accumulator_pipeline.consumer_wait(accumulator_pipe_state);

        Tensor acc = get<0>(slice_accumulator(tmem_storage, accumulator_pipe_state.index()));
        Tensor tAcc = acc(make_coord(_,_),_0{},_0{});
        Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                   // (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        Tensor tTR_tAcc = thread_t2r_epi.partition_S(tAcc_epi);                     // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)

        CUTLASS_PRAGMA_UNROLL
        for (int epi_m = 0; epi_m < size<2>(tAcc_epi); ++epi_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int epi_n = 0; epi_n < size<3>(tAcc_epi); ++epi_n) {

            auto scale_a = tTR_rSFA(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);
            auto scale_b = tTR_rSFB(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);

            Tensor full_acc = tTR_FullAcc(_,_,_,epi_m,epi_n);
            // Compute tmem load predication if necessary
            copy(tiled_t2r_epi, tTR_tAcc(_,_,_,epi_m,epi_n), tTR_PartAcc);
            cutlass::arch::fence_view_async_tmem_load();

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(full_acc); ++i) {
              ElementAccumulator scale = scale_a(i) * scale_b(i);
              full_acc(i) += scale * tTR_PartAcc(i);
            }
          }
        }
        cutlass::arch::fence_view_async_tmem_load();
        accumulator_pipeline.consumer_release(accumulator_pipe_state);
        // release acc
        ++accumulator_pipe_state;
      }

      --k_tile_count;
    }

    return cute::make_tuple(tTR_FullAcc, tiled_t2r_epi, cute::make_tuple(accumulator_pipe_state, mainloop_sf_pipe_state));
 }

protected:

  typename Params::TMA_A const* observed_tma_load_a_{nullptr};
  typename Params::TMA_B const* observed_tma_load_b_{nullptr};

  RuntimeDataTypeA runtime_data_type_a_{};
  RuntimeDataTypeB runtime_data_type_b_{};

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
