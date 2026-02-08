/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/gemm/collective/builders/sm1xx_sparse_config.inl"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"

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
  class ElementPairA_,
  class LayoutPairA_,
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
    MainloopSm100TmaUmmaWarpSpecializedBlockScaledSparse<
      Stages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape>,
    TileShape_,
    ElementPairA_,
    LayoutPairA_,
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
    TransformB_>
{
  //
  // Type Aliases
  //
  using TiledMma = TiledMma_;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;

  using DispatchPolicy = MainloopSm100TmaUmmaWarpSpecializedBlockScaledSparse<
                          Stages,
                          SchedulerPipelineStageCount,
                          AccumulatorPipelineStageCount,
                          ClusterShape>;
  using TileShape = TileShape_;
  using TiledMMA_SF = TiledMMA<MMA_Atom<typename TiledMma::MMA_ScaleFactor>,
                                        Layout<Shape<_1,_1,_1>>,
                                        Tile<Underscore,Underscore,Underscore>>;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  static constexpr int SFVecSize = TiledMma::SFVecSize;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;

  CUTE_STATIC_ASSERT_V(evenly_divides(TileShape{}, tile_shape(TiledMma{})),
                       "Static cluster shape used: TileShape should be evenly divided by TiledMma");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));
  static_assert(shape<1>(CtaShape_MNK{}) == 192 or shape<1>(CtaShape_MNK{}) == 128 or shape<1>(CtaShape_MNK{}) == 256,
      "Cta N should be one of 128/192/256");

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

  // CtaK needs to be multiplier of SFAtomK
  using SfAtom = typename Sm1xxBlkScaledConfig::SfAtom;
  using SfAtomK = cute::Int<cute::size<1>(SfAtom{})>;
  static_assert( shape<2>(CtaShape_MNK{}) % SfAtomK{} == 0, "CtaK needs to be multiplier of SFAtomK");

  // Define A and B block shapes for reduced size TMA_LOADs
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));
  static_assert(get<0,0>(MmaShapeA_MK{}) == 128 &&
                (get<2>(MmaShapeA_MK{}) == 2 || get<2>(MmaShapeA_MK{}) == 4),
                "This kernel only support MmaShape=128 and 2/4 kphase.");

  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using LayoutPairA = LayoutPairA_;
  using StridePairB = StridePairB_;
  static_assert(cute::is_same_v<remove_cvref_t<decltype(get<1>(ElementPairA{}))>,
                                remove_cvref_t<decltype(get<1>(ElementPairB{}))>>, "SFA and SFB data types should be the same");

  // A, B, and E matrices
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementAMmaRaw = typename ElementAMma::raw_type;
  using LayoutA =  remove_cvref_t<decltype(get<0>(LayoutPairA{}))>;
  static constexpr int ElementAMmaSparsity = ElementAMma::sparsity;
  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();

  using ElementEMma = typename TiledMma::ValTypeE;
  using ElementE = typename ElementEMma::raw_type;
  using LayoutE =  remove_cvref_t<decltype(get<1>(LayoutPairA{}))>;
  static constexpr int ElementEMmaSparsity = ElementEMma::sparsity;

  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB = remove_cvref_t<decltype(get<0>(StridePairB{}))>;
  using ElementBMma = typename TiledMma::ValTypeB;
  static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementB>();

  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
                (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
                "ElementA and ElementB should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;

  // SFA and SFB
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using LayoutSFA = remove_cvref_t<decltype(get<2>(LayoutPairA{}))>;
  using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;

  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyPairB = GmemTiledCopyPairB_;
  using GmemTiledCopyA    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopySFA  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopyB    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB{}))>;
  using GmemTiledCopySFB  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB{}))>;

  using SmemLayoutAtomPairA = SmemLayoutAtomPairA_;
  using SmemLayoutAtomPairB = SmemLayoutAtomPairB_;
  using SmemLayoutAtomA   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomPairA{}))>;
  using SmemLayoutAtomSFA = remove_cvref_t<decltype(get<1>(SmemLayoutAtomPairA{}))>;
  using SmemLayoutAtomB   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomPairB{}))>;
  using SmemLayoutAtomSFB = remove_cvref_t<decltype(get<1>(SmemLayoutAtomPairB{}))>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(is_sparse<ElementAMma>::value, "ElementAMma is sparse");
  static_assert(!is_sparse<ElementA>::value, "ElementA is not sparse");
  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) || (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
                "ElementA and ElementB should be both runtime or both static.");

  // LayoutA is nested in the stride due to the sparsity.
  static constexpr bool is_A_mn_major = cute::is_same_v<decltype(stride<0>(LayoutA{})), Int<ElementAMmaSparsity>>;

  using SparseConfig = cutlass::Sm1xxGemmSparseConfig<ElementAMma,
                                                      cute::conditional_t<is_A_mn_major, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>,
                                                      ElementEMma>;
  static constexpr int ElementASparsity = 2; // typename SparseConfig::ElementASparsity{};

  // The offline permutation for the metadata.
  using SmemLayoutAtomE_ = typename SparseConfig::TensorEAtom;
  using SmemLayoutAtomE  = ComposedLayout<Swizzle<0,4,3>,
                                          smem_sparse_ptr_flag_bits<ElementEMmaSparsity, sizeof_bits_v<ElementE>>,
                                          SmemLayoutAtomE_>;

  // Metadata pathways
  using GmemCopyAtomE = GmemTiledCopyA;

  using MainloopPipeline = cutlass::PipelineTmaSparseUmmaAsync<
                             DispatchPolicy::Stages,
                             ClusterShape,
                             AtomThrShapeMNK>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;

  static constexpr int UtccpReuseCnt = ((size<2>(TileShape{}) / typename SparseConfig::TensorEAtomK{}) == 0) ?
                                        typename SparseConfig::TensorEAtomK{} / size<2>(TileShape{}) : 1;
  static_assert(UtccpReuseCnt == 1 || UtccpReuseCnt == 2, "UTCCP reuse count can only be either one or two");
  // (TileM, TileN, TileK) TileK is adjusted according to the reuse.
  using TileShapeE = decltype(replace<2>(TileShape{}, cute::lcm(size<2>(TileShape{}), typename SparseConfig::TensorEAtomK{})));
  using MmaShapeE_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShapeE{}), size<2>(TileShapeE{}))));

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtomA must evenly divide the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtomA must evenly divide the tile shape.");
  static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtomB must evenly divide the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtomB must evenly divide the tile shape.");
  static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  static_assert(rank(SmemLayoutAtomE{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomE{})) == 0, "SmemLayoutAtomE must evenly divide the tile shape.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  // (MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE)
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<is_A_mn_major, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  // (MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE) that one UTCCP instruction can provide
  using SmemLayoutE = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomE{},
      append(MmaShapeE_MK{}, Int<DispatchPolicy::Stages>{})));
  // (MMA_TILE_N,MMA_TILE_K),MMA_N,MMA_K,PIPE)
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

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

  static_assert(rank(SmemLayoutAtomE{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomE{})) == 0, "SmemLayoutAtomE must evenly divide tile shape.");

  static constexpr bool IsF8F6F4 = detail::is_sm100_sparse_f8f6f4<TiledMma, ElementA, ElementB>();

  using TmaInternalElementA = cute::sparse_elem<ElementASparsity,
                                                cute::conditional_t<IsF8F6F4, ElementAMmaRaw, ElementA>>;
  using TmaInternalElementB = cute::conditional_t<IsF8F6F4, ElementBMma, ElementB>;

  using SmemAllocTypeA = cute::sparse_elem<ElementAMmaSparsity,
                                           cute::conditional_t<IsF8F6F4 && cute::sizeof_bits_v<ElementAMmaRaw> < 8,
                                                               uint8_t,
                                                               ElementAMmaRaw>>;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4 && cute::sizeof_bits_v<ElementBMma> < 8, uint8_t, ElementBMma>;

  // Kernel Input Data Type that consider runtime dtype
  using ArrayElementA = cute::conditional_t<IsRuntimeDataTypeA,
                                            cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>,
                                            ElementA>;
  using ArrayElementB = cute::conditional_t<IsRuntimeDataTypeB,
                                            cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>,
                                            ElementB>;

  using RuntimeDataTypeA = cute::conditional_t<IsRuntimeDataTypeA,
                                               cute::conditional_t<IsF8F6F4,
                                                                   cute::UMMA::MXF8F6F4Format,
                                                                   cute::UMMA::MXF4Format>,
                                               void*>;

  using RuntimeDataTypeB = cute::conditional_t<IsRuntimeDataTypeB,
                                               cute::conditional_t<IsF8F6F4,
                                                                   cute::UMMA::MXF8F6F4Format,
                                                                   cute::UMMA::MXF4Format>,
                                               void*>;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::ArrayEngine<ElementEMma, cute::cosize_v<SmemLayoutE>> smem_E;
      cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };

  // Expose shared storage for tensors/pipelines separately to allow kernel layer to reorder them.
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Only one thread issues the TMA and updates the barriers in a 2SM MMA, adjust bytes accordingly
  static constexpr uint32_t SFTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>);
  static constexpr uint32_t ABTmaTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutA{})) * cute::sizeof_bits_v<TmaInternalElementA>) +
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutB{})) * cute::sizeof_bits_v<TmaInternalElementB>);
  static constexpr uint32_t MetadataTmaTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutE{})) * cute::sizeof_bits_v<ElementEMma>);
  static constexpr uint32_t MainLoadTmaTransactionBytes = SFTransactionBytes + ABTmaTransactionBytes;

  template <
    class AccTensor,
    class ETensor, class SfaTensor, class SfbTensor
  >
  struct TmemStorage {
    AccTensor accumulators;
    ETensor tCtE;
    SfaTensor tCtSFA;
    SfbTensor tCtSFB;
  };

  template <
    class KTileCount,
    class GTensorPartitionedA, class GTensorPartitionedB, class GTensorPartitionedE,
    class STensorA, class STensorB, class STensorE,
    class GTensorPartitionedSFA, class GTensorPartitionedSFB,
    class STensorSFA, class STensorSFB
  >
  struct LoadParams {
    // for scheduler
    KTileCount k_tiles;
    // for input tensor values
    GTensorPartitionedA tAgA_mkl;
    GTensorPartitionedB tBgB_nkl;
    GTensorPartitionedE tEgE_nkl;
    STensorA tAsA;
    STensorB tBsB;
    STensorE tEsE;
    GTensorPartitionedSFA tAgSFA_mkl;
    GTensorPartitionedSFB tBgSFB_nkl;
    STensorSFA tAsSFA;
    STensorSFB tBsSFB;
    // the TMA multicast masks
    uint16_t mcast_mask_a;
    uint16_t mcast_mask_b;
    uint16_t mcast_mask_e;
    uint16_t mcast_mask_sfa;
    uint16_t mcast_mask_sfb;

    CUTLASS_DEVICE
    LoadParams (
        KTileCount k_tiles_,
        GTensorPartitionedA tAgA_mkl_, GTensorPartitionedB tBgB_nkl_, GTensorPartitionedE tEgE_nkl_,
        STensorA tAsA_, STensorB tBsB_, STensorE tEsE_,
        GTensorPartitionedSFA tAgSFA_mkl_, GTensorPartitionedSFB tBgSFB_nkl_,
        STensorSFA tAsSFA_, STensorSFB tBsSFB_,
        uint16_t mcast_mask_a_, uint16_t mcast_mask_b_, uint16_t mcast_mask_e_,
        uint16_t mcast_mask_sfa_, uint16_t mcast_mask_sfb_)
    : k_tiles(k_tiles_)
    , tAgA_mkl(tAgA_mkl_), tBgB_nkl(tBgB_nkl_), tEgE_nkl(tEgE_nkl_)
    , tAsA(tAsA_), tBsB(tBsB_), tEsE(tEsE_)
    , tAgSFA_mkl(tAgSFA_mkl_), tBgSFB_nkl(tBgSFB_nkl_)
    , tAsSFA(tAsSFA_), tBsSFB(tBsSFB_)
    , mcast_mask_a(mcast_mask_a_), mcast_mask_b(mcast_mask_b_), mcast_mask_e(mcast_mask_e_)
    , mcast_mask_sfa(mcast_mask_sfa_), mcast_mask_sfb(mcast_mask_sfb_) {}
  };

  template <
    class TiledMma,
    class FragmentA, class FragmentB,
    class FragmentE,   class ETiledCopy,   class SmemFrgE,   class TmemFrgE,
    class FragmentSFA, class SFATiledCopy, class SmemFrgSFA, class TmemFrgSFA,
    class FragmentSFB, class SFBTiledCopy, class SmemFrgSFB, class TmemFrgSFB
  >
  struct MmaParams {
    TiledMma tiled_mma;
    // A
    FragmentA tCrA;
    // B
    FragmentB tCrB;
    // E
    FragmentE tCtE;
    ETiledCopy tiled_copy_s2t_E;
    SmemFrgE thr_tCsE_s2t;
    TmemFrgE thr_tCtE_s2t;
    // SFA
    FragmentSFA tCtSFA;
    SFATiledCopy tiled_copy_s2t_SFA;
    SmemFrgSFA thr_tCsSFA_s2t;
    TmemFrgSFA thr_tCtSFA_s2t;
    // SFB
    FragmentSFB tCtSFB;
    SFBTiledCopy tiled_copy_s2t_SFB;
    SmemFrgSFB thr_tCsSFB_s2t;
    TmemFrgSFB thr_tCtSFB_s2t;

    CUTLASS_DEVICE
    MmaParams (
        TiledMma tiled_mma_,
        FragmentA tCrA_, FragmentB tCrB_,
        FragmentE tCtE_, ETiledCopy tiled_copy_s2t_E_,
        SmemFrgE thr_tCsE_s2t_, TmemFrgE thr_tCtE_s2t_,
        FragmentSFA tCtSFA_, SFATiledCopy tiled_copy_s2t_SFA_,
        SmemFrgSFA thr_tCsSFA_s2t_, TmemFrgSFA thr_tCtSFA_s2t_,
        FragmentSFB tCtSFB_, SFBTiledCopy tiled_copy_s2t_SFB_,
        SmemFrgSFB thr_tCsSFB_s2t_, TmemFrgSFB thr_tCtSFB_s2t_)
    : tiled_mma(tiled_mma_)
    , tCrA(tCrA_), tCrB(tCrB_)
    , tCtE(tCtE_), tiled_copy_s2t_E(tiled_copy_s2t_E_)
    , thr_tCsE_s2t(thr_tCsE_s2t_), thr_tCtE_s2t(thr_tCtE_s2t_)
    , tCtSFA(tCtSFA_), tiled_copy_s2t_SFA(tiled_copy_s2t_SFA_)
    , thr_tCsSFA_s2t(thr_tCsSFA_s2t_), thr_tCtSFA_s2t(thr_tCtSFA_s2t_)
    , tCtSFB(tCtSFB_), tiled_copy_s2t_SFB(tiled_copy_s2t_SFB_)
    , thr_tCsSFB_s2t(thr_tCsSFB_s2t_), thr_tCtSFB_s2t(thr_tCtSFB_s2t_) {}
  };

  // Host side kernel arguments
  struct Arguments {
    // A is A Compressed, not raw tensorA
    ArrayElementA const* ptr_A{nullptr};
    LayoutA layout_a{};
    ArrayElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementE const* ptr_E{nullptr};
    LayoutE layout_e{};
    ElementSF const* ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    ElementSF const* ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
    RuntimeDataTypeA runtime_data_type_a{};
    RuntimeDataTypeB runtime_data_type_b{};
  };

  // Device side kernel params
  struct Params {
    using ClusterLayout_VMNK =
      decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}),
                                                                              ClusterShape{})), make_tile(typename TiledMma::AtomThrID{})));

    using ClusterLayoutSfb_VMNK =
      decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}),
                                                                              ClusterShape{})), make_tile(typename TiledMMA_SF::AtomThrID{})));

    using TMA_A = decltype(make_tma_atom_A_sm100<typename TmaInternalElementA::raw_type>(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), LayoutA{}),
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    using TMA_E = decltype(make_tma_atom_A_sm100<uint64_t>( // use uint64_t to get the largest loading box.
        GmemCopyAtomE{},
        make_tensor(recast_ptr<ElementEMma>(nullptr), LayoutE{}),
        SmemLayoutE{}(_,_,_,cute::Int<0>{}),
        TileShapeE{},
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
        TiledMMA_SF{},
        ClusterLayoutSfb_VMNK{})
      );

    TMA_A tma_load_a;
    TMA_E tma_load_e;
    TMA_B tma_load_b;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;
    TMA_A tma_load_a_fallback;
    TMA_E tma_load_e_fallback;
    TMA_B tma_load_b_fallback;
    TMA_SFA tma_load_sfa_fallback;
    TMA_SFB tma_load_sfb_fallback;
    LayoutA layout_a;
    LayoutE layout_e;
    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    dim3 cluster_shape_fallback;
    RuntimeDataTypeA runtime_data_type_a;
    RuntimeDataTypeB runtime_data_type_b;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
    : cluster_shape_(cluster_shape)
    , block_rank_in_cluster_(block_rank_in_cluster)
    , layout_a_(params.layout_a)
    , layout_e_(params.layout_e)
    , layout_SFA_(params.layout_SFA)
    , layout_SFB_(params.layout_SFB)
    , runtime_data_type_a_(params.runtime_data_type_a)
    , runtime_data_type_b_(params.runtime_data_type_b) {
    if constexpr (IsDynamicCluster) {
      const bool is_fallback_cluster = (cute::size<0>(cluster_shape_) == params.cluster_shape_fallback.x &&
                                        cute::size<1>(cluster_shape_) == params.cluster_shape_fallback.y);
      observed_tma_load_a_ = is_fallback_cluster ? &params.tma_load_a_fallback : &params.tma_load_a;
      observed_tma_load_e_ = is_fallback_cluster ? &params.tma_load_e_fallback : &params.tma_load_e;
      observed_tma_load_b_ = is_fallback_cluster ? &params.tma_load_b_fallback : &params.tma_load_b;
      observed_tma_load_sfa_ = is_fallback_cluster ? &params.tma_load_sfa_fallback : &params.tma_load_sfa;
      observed_tma_load_sfb_ = is_fallback_cluster ? &params.tma_load_sfb_fallback : &params.tma_load_sfb;
    }
    else {
      observed_tma_load_a_ = &params.tma_load_a;
      observed_tma_load_e_ = &params.tma_load_e;
      observed_tma_load_b_ = &params.tma_load_b;
      observed_tma_load_sfa_ = &params.tma_load_sfa;
      observed_tma_load_sfb_ = &params.tma_load_sfb;
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
    auto ptr_E = recast_ptr<ElementEMma>(args.ptr_E);

    Tensor tensor_a = make_tensor(ptr_A, args.layout_a);
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));
    Tensor tensor_e = make_tensor(ptr_E, args.layout_e);
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);

    // Cluster layout for TMA construction
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));
    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));
    Tensor tensor_sfa = make_tensor(args.ptr_SFA, args.layout_SFA);
    Tensor tensor_sfb = make_tensor(args.ptr_SFB, args.layout_SFB);

    // Cluster layout for TMA construction of SFB
    auto cluster_layout_sfb_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMMA_SF::AtomThrID{}));
    auto cluster_layout_sfb_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMMA_SF::AtomThrID{}));

    typename Params::TMA_A tma_load_a = make_tma_atom_A_sm100<typename TmaInternalElementA::raw_type>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_E tma_load_e = make_tma_atom_A_sm100<uint64_t>( // use uint64_t to get the largest loading box.
        GmemCopyAtomE{},
        tensor_e,
        SmemLayoutE{}(_,_,_,cute::Int<0>{}),
        TileShapeE{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_B tma_load_b = make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_A tma_load_a_fallback = make_tma_atom_A_sm100<typename TmaInternalElementA::raw_type>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    typename Params::TMA_E tma_load_e_fallback = make_tma_atom_A_sm100<uint64_t>( // use uint64_t to get the largest loading box.
        GmemCopyAtomE{},
        tensor_e,
        SmemLayoutE{}(_,_,_,cute::Int<0>{}),
        TileShapeE{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    typename Params::TMA_B tma_load_b_fallback = make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

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
        TiledMMA_SF{},
        cluster_layout_sfb_vmnk);

    typename Params::TMA_SFA tma_load_sfa_fallback = make_tma_atom_A_sm100<uint16_t>(
        GmemTiledCopySFA{},
        tensor_sfa,
        SmemLayoutSFA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    typename Params::TMA_SFB tma_load_sfb_fallback = make_tma_atom_B_sm100<uint16_t>(
        GmemTiledCopySFB{},
        tensor_sfb,
        SmemLayoutSFB{}(_,_,_,cute::Int<0>{}),
        TileShape_SF{},
        TiledMMA_SF{},
        cluster_layout_sfb_vmnk_fallback);

    return {
      tma_load_a,
      tma_load_e,
      tma_load_b,
      tma_load_sfa,
      tma_load_sfb,
      tma_load_a_fallback,
      tma_load_e_fallback,
      tma_load_b_fallback,
      tma_load_sfa_fallback,
      tma_load_sfb_fallback,
      args.layout_a,
      args.layout_e,
      args.layout_SFA,
      args.layout_SFB,
      hw_info.cluster_shape_fallback,
      args.runtime_data_type_a,
      args.runtime_data_type_b
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {

    // Check for Alignment Requirement
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cute::sizeof_bits_v<ElementA>;

    bool implementable = true;
    // Check Alignment A
    if constexpr (is_A_mn_major) {
      implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,     K/2, L),
                                                                                                    cute::make_stride(_1{}, M,   M*K/2));
    }
    else { // If A is K-major
      implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,    K/2,  L),
                                                                                                    cute::make_stride(K/2, _1{}, M*K/2));
    }
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA on tensorA\n");
    }

    // Check Alignment B
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cute::sizeof_bits_v<ElementB>;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA on tensorB\n");
    }

    // Check for AB layout requirement
    const auto layout_a_ref = SparseConfig::fill_layoutA(problem_shape_MNKL);
    const auto layout_e_ref = SparseConfig::fill_layoutE(problem_shape_MNKL);
    implementable = implementable && (layout_a_ref == args.layout_a);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: layout_a mismatch\n");
    }

    implementable = implementable && (layout_e_ref == args.layout_e);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: layout_e mismatch\n");
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

    if constexpr (IsRuntimeDataType && detail::is_sm10x_mxf4nvf4_input<ElementAMma>() && detail::is_sm10x_mxf4nvf4_input<ElementBMma>()) {
      bool is_compatible = (SFVecSize == 32 ||
                           (SFVecSize == 64 && is_same_v<ElementSF, cutlass::float_ue8m0_t>
                                            && args.runtime_data_type_a == cute::UMMA::MXF4Format::E2M1
                                            && args.runtime_data_type_b == cute::UMMA::MXF4Format::E2M1));
      if (!is_compatible) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: 2x mode (VectorSize=64) only supports float_e2m1_t for a/b types and ue8m0_t for sf type.\n");
      }
      implementable &= is_compatible;
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE void
  prefetch_tma_descriptors() {
    cute::prefetch_tma_descriptor(observed_tma_load_a_->get_tma_descriptor());
    cute::prefetch_tma_descriptor(observed_tma_load_b_->get_tma_descriptor());
    cute::prefetch_tma_descriptor(observed_tma_load_e_->get_tma_descriptor());
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
    Tensor tCtE   = make_tensor<typename TiledMma::FrgTypeE>(take<0,3>(shape(SmemLayoutE{})));

    TmemStorage<decltype(accumulators), decltype(tCtE), decltype(tCtSFA), decltype(tCtSFB)> tmem_storage;
    tmem_storage.accumulators = accumulators;
    tmem_storage.tCtSFA = tCtSFA;
    tmem_storage.tCtSFB = tCtSFB;
    tmem_storage.tCtE = tCtE;

    return tmem_storage;
  }

  template <class TmemStorage>
  CUTLASS_DEVICE static
  void
  set_tmem_offsets(TmemStorage& tmem_storage, uint32_t tmem_base_addr) {
    tmem_storage.accumulators.data() = tmem_base_addr;
    tmem_storage.tCtE.data()         = tmem_base_addr + cutlass::detail::find_tmem_tensor_col_offset(tmem_storage.accumulators);
    tmem_storage.tCtSFA.data()       = tmem_storage.tCtE.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tmem_storage.tCtE);
    tmem_storage.tCtSFB.data()       = tmem_storage.tCtSFA.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tmem_storage.tCtSFA);
  }

  /// Set up the data needed by this collective for load.
  /// Return tuple element contain
  /// gA_mkl - The tiled tma tensor for input A
  /// gB_nkl - The tiled tma tensor for input B
  /// tAgA_mkl - partitioned gmem tensor for A
  /// tBgB_nkl - partitioned gmem tensor for B
  /// tAsA - partitioned smem tensor for A
  /// tBsB - partitioned smem tensor for B
  /// tAgSFA_mkl - partitioned gmem tensor for SFA
  /// tBgSFB_nkl - partitioned gmem tensor for SFB
  /// tAsSFA - partitioned tmem tensor for SFA
  /// tAsSFB - partitioned tmem tensor for SFB
  /// mcast_mask_a - tma multicast mask for A
  /// mcast_mask_b - tma multicast mask for B
  /// mcast_mask_sfa - tma multicast mask for SFA
  /// mcast_mask_sfb - tma multicast mask for SFB
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(layout_a_.shape());
    Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N,K,L));
    Tensor mE_mkl = observed_tma_load_e_->get_tma_tensor(layout_e_.shape());

    // Tile the tensors and defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});    // (BLK_M, BLK_K, m, k, l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});    // (BLK_N, BLK_K, n, k, l)
    Tensor gE_mkl = local_tile(mE_mkl, TileShapeE{}, make_coord(_,_,_), Step<_1, X,_1>{});    // (BLK_M, BLK_K, m, k, l)

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

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);          // (MMA, MMA_M, MMA_K, m, k, l)
    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);          // (MMA, MMA_N, MMA_K, n, k, l)
    Tensor tCgE_mkl = cta_mma.partition_A(gE_mkl);          // (MMA, MMA_M, MMA_K, m, k, l)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});  // (MMA,MMA_N,MMA_K,PIPE)
    Tensor sE = make_tensor(make_smem_ptr(shared_tensors.smem_E.begin()), SmemLayoutE{});  // (MMA,MMA_M,MMA_K,PIPE)

    ThrMMA cta_mma_sfb = TiledMMA_SF{}.get_slice(blockIdx.x % size(typename TiledMMA_SF::AtomThrID{}));
    Tensor tCgSFA_mkl = cta_mma.partition_A(gSFA_mkl);          // (MMA, MMA_M, MMA_K, m, k, l)
    Tensor tCgSFB_nkl = cta_mma_sfb.partition_B(gSFB_nkl);          // (MMA, MMA_N, MMA_K, n, k, l)

    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

    // Define the CTA-in-cluster Layout and Coord
    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    Layout cta_layout_sfb_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMMA_SF::AtomThrID{}));
    auto cta_coord_sfb_vmnk  = cta_layout_sfb_vmnk.get_flat_coord(block_rank_in_cluster_);

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgB_nkl, tBsB] = tma_partition(*observed_tma_load_b_,
                                      get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                      group_modes<0,3>(sB), group_modes<0,3>(tCgB_nkl));

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgSFA_mkl, tAsSFA] = tma_partition(*observed_tma_load_sfa_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sSFA), group_modes<0,3>(tCgSFA_mkl));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgSFB_nkl, tBsSFB] = tma_partition(*observed_tma_load_sfb_,
                                      get<1>(cta_coord_sfb_vmnk), make_layout(size<1>(cta_layout_sfb_vmnk)),
                                      group_modes<0,3>(sSFB), group_modes<0,3>(tCgSFB_nkl));

    // Project the cta_layout for tma_a along the n-modes
    auto [tEgE_mkl, tEsE] = tma_partition(*observed_tma_load_e_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sE), group_modes<0,3>(tCgE_mkl));

    // TMA Multicast Masks
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_sfa = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_sfb = create_tma_multicast_mask<1>(cta_layout_sfb_vmnk, cta_coord_sfb_vmnk);
    uint16_t mcast_mask_e = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);

    return LoadParams{
      size<3>(gA_mkl),                                // for scheduler
      tAgA_mkl, tBgB_nkl, tEgE_mkl, tAsA, tBsB, tEsE, // for input tensor values
      tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB,         // for input scale factor tensor values
      mcast_mask_a, mcast_mask_b, mcast_mask_e, mcast_mask_sfa, mcast_mask_sfb}; // multicast masks
  }

  /// Set up the data needed by this collective for mma compute.
  template <class TmemStorage>
  CUTLASS_DEVICE auto
  mma_init(
    TmemStorage tmem_storage,
    TensorStorage& shared_tensors) const {

    // Allocate "fragments/descriptors" for A B E matrices
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
    Tensor sE = make_tensor(make_smem_ptr(shared_tensors.smem_E.begin()), SmemLayoutE{});  // (MMA,MMA_M,MMA_K,PIPE) that one UTCCP can provide

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA = TiledMma::make_fragment_A(sA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = TiledMma::make_fragment_B(sB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sA));                                     // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sB));                                     // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sE));                                     // PIPE

    Tensor tCtE = tmem_storage.tCtE;
    using AtomThrID = typename TiledMma::AtomThrID;
    using UtccpEOp = cute::conditional_t<(decltype(cute::size(AtomThrID{}) == Int<2>{})::value),
      cute::SM100_UTCCP_128dp128bit_2cta, cute::SM100_UTCCP_128dp128bit_1cta>;
    auto tiled_copy_s2t_E = make_utccp_copy(UtccpEOp{}, recast<ElementE>(tCtE));

    auto thr_copy_s2t_E = tiled_copy_s2t_E.get_slice(0);
    Tensor thr_tCsE_s2t_ = thr_copy_s2t_E.partition_S(recast<ElementE>(sE));
    // SMEM to TMEM copy operation requires source SMEM operand to be an SMEM descriptor
    Tensor thr_tCsE_s2t = get_utccp_smem_desc_tensor<UtccpEOp>(thr_tCsE_s2t_);
    Tensor thr_tCtE_s2t = thr_copy_s2t_E.partition_D(recast<ElementE>(tCtE));

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
    using UtccpOp = cute::conditional_t<(decltype(cute::size(AtomThrID{}) == Int<2>{})::value),
      SM100_UTCCP_4x32dp128bit_2cta, SM100_UTCCP_4x32dp128bit_1cta>;
    auto tiled_copy_s2t_SFA = make_utccp_copy(UtccpOp{}, tCtSFA_compact);
    auto tiled_copy_s2t_SFB = make_utccp_copy(UtccpOp{}, tCtSFB_compact);

    auto thr_copy_s2t_SFA = tiled_copy_s2t_SFA.get_slice(0);
    auto thr_tCsSFA_s2t_ = thr_copy_s2t_SFA.partition_S(tCsSFA_compact);
    // SMEM to TMEM copy operation requires source SMEM operand to be an SMEM descriptor
    auto thr_tCsSFA_s2t = get_utccp_smem_desc_tensor<UtccpOp>(thr_tCsSFA_s2t_);
    auto thr_tCtSFA_s2t = thr_copy_s2t_SFA.partition_D(tCtSFA_compact);

    auto thr_copy_s2t_SFB = tiled_copy_s2t_SFB.get_slice(0);
    auto thr_tCsSFB_s2t_ = thr_copy_s2t_SFB.partition_S(tCsSFB_compact);
    // SMEM to TMEM copy operation requires source SMEM operand to be an SMEM descriptor
    auto thr_tCsSFB_s2t = get_utccp_smem_desc_tensor<UtccpOp>(thr_tCsSFB_s2t_);
    auto thr_tCtSFB_s2t = thr_copy_s2t_SFB.partition_D(tCtSFB_compact);

    TiledMma tiled_mma;

    if constexpr (IsRuntimeDataType) {
      // Update instruction descriptor according to runtime argument.
      // Applying bitmask (0b111) to help compiler deduce that the conversion and assignment are safe.
      tiled_mma.idesc_.a_format_ = uint8_t(runtime_data_type_a_) & 0b111;
      tiled_mma.idesc_.b_format_ = uint8_t(runtime_data_type_b_) & 0b111;
    }

    return MmaParams{
      tiled_mma,
      tCrA, tCrB,
      tCtE,   tiled_copy_s2t_E,   thr_tCsE_s2t,   thr_tCtE_s2t,
      tCtSFA, tiled_copy_s2t_SFA, thr_tCsSFA_s2t, thr_tCtSFA_s2t,
      tCtSFB, tiled_copy_s2t_SFB, thr_tCsSFB_s2t, thr_tCtSFB_s2t};
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class LoadParams,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load(
    MainloopPipeline mainloop_pipeline,
    MainloopPipelineState mainloop_pipe_producer_state,
    LoadParams const& load_inputs,
    TileCoordMNKL const& cta_coord_mnkl,
    KTileIterator k_tile_iter, int k_tile_count) {

    auto [k_tiles,
          tAgA_mkl, tBgB_nkl, tEgE_mkl, tAsA, tBsB, tEsE,
          tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB,
          mcast_mask_a, mcast_mask_b, mcast_mask_e,
          mcast_mask_sfa, mcast_mask_sfb] = load_inputs;

    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tEgE = tEgE_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
    Tensor tAgSFA = tAgSFA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgSFB = tBgSFB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);
      // Note: We don't synchronize the sf_pipeline for "Buffer_Empty". We use mainloop pipeline
      // to do the synchronization at once.

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

      int write_stage = mainloop_pipe_producer_state.index();
      ++mainloop_pipe_producer_state;
      barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

      if (cute::elect_one_sync()) {
        copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
        copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
        copy(observed_tma_load_sfa_->with(*tma_barrier, mcast_mask_sfa), tAgSFA(_,*k_tile_iter), tAsSFA(_,write_stage));
        copy(observed_tma_load_sfb_->with(*tma_barrier, mcast_mask_sfb), tBgSFB(_,*k_tile_iter), tBsSFB(_,write_stage));
        copy(observed_tma_load_e_->with(*tma_barrier, mcast_mask_e), tEgE(_,*k_tile_iter), tEsE(_,write_stage));
      }

      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline mainloop_pipeline, MainloopPipelineState mainloop_pipe_producer_state) {
    // Issue the epilogue waits
    // This helps avoid early exit of ctas in Cluster
    // Waits for all stages to either be released (all
    // Consumer UNLOCKs), or if the stage was never used
    // then would just be acquired since the phase was
    // still inverted from make_producer_start_state
    mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class AccumulatorPipeline,
    class FrgEngine, class FrgLayout,
    class MmaParams,
    class CtaTileCoord
  >
  CUTLASS_DEVICE auto
  mma(cute::tuple<MainloopPipeline,
                  AccumulatorPipeline> pipelines,
      cute::tuple<MainloopPipelineState,
                  typename AccumulatorPipeline::PipelineState> pipeline_states,
      cute::tuple<cute::Tensor<FrgEngine, FrgLayout>> const& accumulators_pair,
      MmaParams const& mma_inputs,
      CtaTileCoord cta_tile_coord,
      int k_tile_count
  ) {
    static_assert(is_tmem<FrgEngine>::value, "Accumulator must be tmem resident.");
    static_assert(rank(FrgLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

    auto accumulators = get<0>(accumulators_pair);
    auto [tiled_mma,
          tCrA, tCrB,
          tCtE,   tiled_copy_s2t_E,   thr_tCsE_s2t, thr_tCtE_s2t,
          tCtSFA, tiled_copy_s2t_SFA, thr_tCsSFA_s2t, thr_tCtSFA_s2t,
          tCtSFB, tiled_copy_s2t_SFB, thr_tCsSFB_s2t, thr_tCtSFB_s2t] = mma_inputs;

    auto [mainloop_pipeline, accumulator_pipeline] = pipelines;
    auto [mainloop_pipe_consumer_state, accumulator_pipe_producer_state] = pipeline_states;

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

    uint32_t skip_wait = k_tile_count <= 0;
    auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

    //
    // PIPELINED MAIN LOOP
    //
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    if constexpr (IsOverlappingAccum) {
      // first iteration manual unroll for tmem overlap kernel
      if (k_tile_count > 0) {
        // WAIT on mainloop_pipe_consumer_state until its data are available
        // (phase bit flips from mainloop_pipe_consumer_state.phase() value)
        mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);

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

        if (cute::elect_one_sync()) {
          copy(tiled_copy_s2t_E,   thr_tCsE_s2t(_,_,_,_,read_stage),   thr_tCtE_s2t);
          copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_,_,_,_,read_stage), thr_tCtSFA_s2t);
          copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_,_,_,_,read_stage), thr_tCtSFB_s2t);
        }

        // Wait for tmem accumulator buffer to become empty with a flipped phase
        accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);

        // Unroll the K mode manually so we can set scale C to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          // (V,M) x (V,N) => (V,M,N)
          cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
                                    tCtE(_,_,k_block),
                                    tCtSFA(_,_,k_block),
                                    tCtSFB_mma(_,_,k_block)),
              tCrA(_,_,k_block,read_stage),
              tCrB(_,_,k_block,read_stage),
              accumulators);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }

        mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
      }
    }
    else {
      // Wait for tmem accumulator buffer to become empty with a flipped phase
      accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
    }

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // WAIT on mainloop_pipe_consumer_state until its data are available
      // (phase bit flips from mainloop_pipe_consumer_state.phase() value)
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);

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

      if (cute::elect_one_sync()) {
        copy(tiled_copy_s2t_E,   thr_tCsE_s2t(_,_,_,_,read_stage),   thr_tCtE_s2t);
        copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_,_,_,_,read_stage), thr_tCtSFA_s2t);
        copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_,_,_,_,read_stage), thr_tCtSFB_s2t);
      }

      // Unroll the K mode manually so we can set scale C to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
                                  tCtE(_,_,k_block),
                                  tCtSFA(_,_,k_block),
                                  tCtSFB_mma(_,_,k_block)),
            tCrA(_,_,k_block,read_stage),
            tCrB(_,_,k_block,read_stage),
            accumulators);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }

      mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
    }

    return mainloop_pipe_consumer_state;
  }

protected:

  typename Params::TMA_A const* observed_tma_load_a_{nullptr};
  typename Params::TMA_E const* observed_tma_load_e_{nullptr};
  typename Params::TMA_B const* observed_tma_load_b_{nullptr};
  typename Params::TMA_SFA const* observed_tma_load_sfa_{nullptr};
  typename Params::TMA_SFB const* observed_tma_load_sfb_{nullptr};

  LayoutA layout_a_;
  LayoutE layout_e_;
  LayoutSFA layout_SFA_;
  LayoutSFB layout_SFB_;
  RuntimeDataTypeA runtime_data_type_a_{};
  RuntimeDataTypeB runtime_data_type_b_{};

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
