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
#include "cutlass/detail/sm103_blockscaled_layout.hpp"
#include "cutlass/detail/collective/sm103_kernel_type.hpp"
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
  int LoadABPipelineStageCount,
  int LoadSFPipelineStageCount,
  int SchedulerPipelineStageCount,
  int AccumulatorPipelineStageCount,
  class ClusterShape,   // Static cluster shape or dynamic (int, int, int)
  cutlass::sm103::detail::KernelPrefetchType PrefetchType,
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
    MainloopSm103ArrayTmaUmmaWarpSpecializedBlockScaled<
      LoadABPipelineStageCount,
      LoadSFPipelineStageCount,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape,
      PrefetchType>,
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
    TransformB_>
{
  //
  // Type Aliases
  //
  using TiledMma = TiledMma_;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;

  using DispatchPolicy = MainloopSm103ArrayTmaUmmaWarpSpecializedBlockScaled<
                          LoadABPipelineStageCount,
                          LoadSFPipelineStageCount,
                          SchedulerPipelineStageCount,
                          AccumulatorPipelineStageCount,
                          ClusterShape,
                          PrefetchType>;

  using TileShape = TileShape_;
  // Due to an MSVC bug, we can't use decltype(make_tiled_mma()) interface.
  using TiledMMA_SF = TiledMMA<MMA_Atom<typename TiledMma::MMA_ScaleFactor>,
                                        Layout<Shape<_1,_1,_1>>,
                                        Tile<Underscore,Underscore,Underscore>>;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  static constexpr int SFVecSize = TiledMma::SFVecSize;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;

  // Assert that TiledMma and TileShape should be weakly compatible
  CUTE_STATIC_ASSERT_V(evenly_divides(TileShape{}, tile_shape(TiledMma{})),
                       "Static cluster shape used: TiledMma and TileShape should be weakly compatible");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));
  static_assert(shape<1>(CtaShape_MNK{}) == 192 or shape<1>(CtaShape_MNK{}) == 128 or shape<1>(CtaShape_MNK{}) == 256,
      "Cta N should be one of 128/192/256");

  using ClusterTileShape = decltype(make_shape(get<0>(TileShape{})*get<0>(ClusterShape{}),get<1>(TileShape{})*get<1>(ClusterShape{}),get<2>(TileShape{})*get<2>(ClusterShape{})));
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm103BlockScaledConfig<SFVecSize>;
  using Blk_MN = typename Sm1xxBlkScaledConfig::Blk_MN;
  static constexpr int IsCtaN192 = shape<1>(CtaShape_MNK{}) == 192;
  static int constexpr CTA_N_SF = cutlass::round_up(size<1>(CtaShape_MNK{}), Blk_MN{});
  // Tile shape used for partitioning Scale Factor B.
  // The M-dim does not affect the SFB, so just set it as the original TileShape;
  using TileShape_SF = decltype(make_shape(get<0>(CtaShape_MNK{}),
                                           Int<CTA_N_SF>{} * shape<2>(typename TiledMma::ThrLayoutVMNK()),
                                           get<2>(TileShape{})));

  static int constexpr SF_BUFFERS_PER_TILE_K = SFVecSize == 16 ? 4 : 2;
  using MMA_SF_Tiler = decltype(make_tile(shape<0>(CtaShape_MNK{}), Int<CTA_N_SF>{}, Int<shape<2>(CtaShape_MNK{})/SF_BUFFERS_PER_TILE_K>{}));

  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;
  using SmemLayoutAtomPairA = SmemLayoutAtomPairA_;
  using SmemLayoutAtomPairB = SmemLayoutAtomPairB_;
  static_assert(cute::is_same_v<remove_cvref_t<decltype(get<1>(ElementPairA{}))>,
                                remove_cvref_t<decltype(get<1>(ElementPairB{}))>>, "SFA and SFB data types should be the same");

  // A and B matrices
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA  = remove_cvref_t<decltype(get<0>(StridePairA{}))>;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB  = remove_cvref_t<decltype(get<0>(StridePairB{}))>;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();

  static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementB>();

  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
                (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
                "ElementA and ElementB should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

  // SFA and SFB
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using LayoutSFA = remove_cvref_t<decltype(get<1>(StridePairA{}))>;
  using InternalLayoutSFA = cute::remove_pointer_t<LayoutSFA>;
  using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;
  using InternalLayoutSFB = cute::remove_pointer_t<LayoutSFB>;
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

  using MainloopABPipeline = cutlass::PipelineTmaUmmaAsync<
                             DispatchPolicy::LoadABPipelineStageCount,
                             ClusterShape,
                             AtomThrShapeMNK>;
  using MainloopABPipelineState = typename MainloopABPipeline::PipelineState;

  using MainloopSFPipeline = cutlass::PipelineTmaUmmaAsync<
                             DispatchPolicy::LoadSFPipelineStageCount,
                             ClusterShape,
                             AtomThrShapeMNK>;
  using MainloopSFPipelineState = typename MainloopSFPipeline::PipelineState;

  static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM103 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM103 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  // (MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE)
 using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
    SmemLayoutAtomA{},
    append(make_shape(make_shape(shape<0>(CtaShape_MNK{}), _16{}), _1{}, _8{}), Int<DispatchPolicy::LoadABPipelineStageCount>{} /*PIPE*/),
    cute::conditional_t<cutlass::gemm::detail::is_mn_major<InternalStrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));     // ((CTA_MMA_M,16bytes),1,8,NUM_PIPES)
  using SmemLayoutA_tma = decltype(UMMA::tile_to_mma_shape(
    SmemLayoutAtomA{},
    append(make_shape(make_shape(shape<0>(CtaShape_MNK{}), _16{}), _1{}, _8{}), Int<3>{}  /*Per mainloop iteration */),
    cute::conditional_t<cutlass::gemm::detail::is_mn_major<InternalStrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));     // ((CTA_MMA_M,16bytes),1,8,3)

  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
    SmemLayoutAtomB{},
    append(make_shape(make_shape(shape<1>(CtaShape_MNK{}) / size(typename TiledMma::AtomThrID{}), _16{}), _1{}, _8{}), Int<DispatchPolicy::LoadABPipelineStageCount>{} /*PIPE*/),
    cute::conditional_t<cutlass::gemm::detail::is_mn_major<InternalStrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));     // ((CTA_MMA_N,16bytes),1,8,NUM_PIPES)
  using SmemLayoutB_tma = decltype(UMMA::tile_to_mma_shape(
    SmemLayoutAtomB{},
    append(make_shape(make_shape(shape<1>(CtaShape_MNK{}) / size(typename TiledMma::AtomThrID{}), _16{}), _1{}, _8{}), Int<3>{} /*Per mainloop iteration */),
    cute::conditional_t<cutlass::gemm::detail::is_mn_major<InternalStrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));     // ((CTA_MMA_N,16bytes),1,8,3)


  // SmemLayoutAtomSFA and SmemLayoutAtomSFB are for whole CTA tiles. We add the number of pipeline stages here.
  // The number of pipeline stages is the same as the number of pipeline stages from AB Load <-> MainLoop
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::LoadSFPipelineStageCount>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::LoadSFPipelineStageCount>{}),
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

  static constexpr bool IsF8F6F4 = detail::is_sm100_mma_f8f6f4<TiledMma, ElementA, ElementB>();
  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  using TmaInternalElementA = uint8_t;
  using TmaInternalElementB = uint8_t;

  using SmemAllocTypeA = uint8_t;
  using SmemAllocTypeB = uint8_t;

  using BitTypeElementA = cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>;
  using BitTypeElementB = cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>;

  using ArrayElementA = cute::conditional_t<IsRuntimeDataTypeA, BitTypeElementA, ElementA>;
  using ArrayElementB = cute::conditional_t<IsRuntimeDataTypeB, BitTypeElementB, ElementB>;

  using RuntimeDataTypeA = typename detail::sm10x_block_scale_runtime_input_t<ElementAMma, IsRuntimeDataTypeA>::Type;
  using RuntimeDataTypeB = typename detail::sm10x_block_scale_runtime_input_t<ElementBMma, IsRuntimeDataTypeB>::Type;

  using SmemPrefetchType = uint8_t;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::ArrayEngine<SmemAllocTypeA,   cute::cosize_v<SmemLayoutA>> smem_A;
      cute::ArrayEngine<SmemAllocTypeB,   cute::cosize_v<SmemLayoutB>> smem_B;
      cute::ArrayEngine<ElementSF,        cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      cute::ArrayEngine<ElementSF,        cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;

    struct TensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
      cute::TmaDescriptor smem_tensormap_SFA;
      cute::TmaDescriptor smem_tensormap_SFB;
    } tensormaps;

    using PipelineABStorage = typename MainloopABPipeline::SharedStorage;
    using PipelineSFStorage = typename MainloopSFPipeline::SharedStorage;
    struct PipelineStorage {
      PipelineABStorage pipeline_ab;
      PipelineSFStorage pipeline_sf;
    };
  };

  // Expose shared storage for tensors/pipelines separately to allow kernel layer to reorder them.
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr uint32_t SFTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>);
  // Only one thread issues the TMA and updates the barriers in a 2SM MMA, adjust bytes accordingly
  static constexpr uint32_t ABTmaTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutA{})) * cute::sizeof_bits_v<TmaInternalElementA>) +
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutB{})) * cute::sizeof_bits_v<TmaInternalElementB>);

  // Host side kernel arguments
  struct Arguments {
    ArrayElementA const** ptr_A{nullptr};
    StrideA dA{};
    ArrayElementB const** ptr_B{nullptr};
    StrideB dB{};
    ElementSF const** ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    ElementSF const** ptr_SFB{nullptr};
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

    using TMA_A = decltype(make_tma_atom<uint8_t>(
        GmemTiledCopyA{},
        recast<uint8_t>(make_tensor(recast_ptr<ElementA>(nullptr), repeat_like(InternalStrideA{}, int32_t(0)), InternalStrideA{})),
        SmemLayoutA_tma{},
        make_tile(size<1,0>(typename TiledMma::ALayout{}), _384{}),
        size<1>(ClusterShape{}))
      );

    using TMA_B = decltype(make_tma_atom<uint8_t>(
        GmemTiledCopyB{},
        recast<uint8_t>(make_tensor(recast_ptr<ElementB>(nullptr), repeat_like(InternalStrideB{}, int32_t(0)), InternalStrideB{})),
        SmemLayoutB_tma{},
        make_tile(size<1,0>(typename TiledMma::BLayout{}), _384{}),
        size<0>(ClusterShape{})/size(typename TiledMma::AtomThrID{}))
      );

    using TMA_SFA = decltype(make_tma_atom<uint8_t>( // using legacy sm90 make_tma_atom
        GmemTiledCopySFA{},
        make_tensor(static_cast<ElementSF const*>(nullptr), InternalLayoutSFA{}),
        SmemLayoutSFA{}(_,_,_,cute::Int<0>{}),
        make_shape(get<0>(MMA_SF_Tiler{}), get<2>(MMA_SF_Tiler{})),
        size<1>(ClusterShape{}))
      );

    using TMA_SFB = decltype(make_tma_atom<uint8_t>( // using legacy sm90 make_tma_atom
        GmemTiledCopySFB{},
        make_tensor(static_cast<ElementSF const*>(nullptr), InternalLayoutSFB{}),
        SmemLayoutSFB{}(_,_,_,cute::Int<0>{}),
        make_shape(get<1>(MMA_SF_Tiler{}), get<2>(MMA_SF_Tiler{})),
        size<0>(ClusterShape{})/size(typename TiledMMA_SF::AtomThrID{}))
      );

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;
    TMA_A tma_load_a_fallback;
    TMA_B tma_load_b_fallback;
    TMA_SFA tma_load_sfa_fallback;
    TMA_SFB tma_load_sfb_fallback;
    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    dim3 cluster_shape_fallback;
    RuntimeDataTypeA runtime_data_type_a;
    RuntimeDataTypeB runtime_data_type_b;
    cute::TmaDescriptor* tensormaps;
    ArrayElementA const** ptr_A;
    StrideA dA;
    ArrayElementB const** ptr_B;
    StrideB dB;
    ElementSF const** ptr_SFA;
    ElementSF const** ptr_SFB;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params) {
    if constexpr (IsDynamicCluster) {
      dim3 cs = cute::cluster_shape();
      const bool is_fallback_cluster = (cs.x == params.cluster_shape_fallback.x && cs.y == params.cluster_shape_fallback.y);
      observed_tma_load_a_ = is_fallback_cluster ? &params.tma_load_a_fallback : &params.tma_load_a;
      observed_tma_load_b_ = is_fallback_cluster ? &params.tma_load_b_fallback : &params.tma_load_b;
      observed_tma_load_sfa_ = is_fallback_cluster ? &params.tma_load_sfa_fallback : &params.tma_load_sfa;
      observed_tma_load_sfb_ = is_fallback_cluster ? &params.tma_load_sfb_fallback : &params.tma_load_sfb;

    }
    else {
      observed_tma_load_a_ = &params.tma_load_a;
      observed_tma_load_b_ = &params.tma_load_b;
      observed_tma_load_sfa_ = &params.tma_load_sfa;
      observed_tma_load_sfb_ = &params.tma_load_sfb;
    }
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
    ProblemShape const& problem_shapes,
    Arguments const& args,
    [[maybe_unused]] void* workspace,
    cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_M = int32_t(size<0>(TileShape{}));
    auto init_N = int32_t(size<1>(TileShape{}));
    auto init_K = int32_t(size<2>(TileShape{}));
    auto init_L = 1;

    // Tensor pointers will be fixed before the first access
    ElementA const* ptr_A_first_batch = nullptr;
    ElementB const* ptr_B_first_batch = nullptr;

    InternalStrideA stride_a;
    InternalStrideB stride_b;
    InternalLayoutSFA layout_SFA;
    InternalLayoutSFB layout_SFB;

    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      stride_a = InternalStrideA{};
      stride_b = InternalStrideB{};
      layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(init_M, init_N, init_K, 1));
      layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(init_M, init_N, init_K, 1));
    }
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      stride_a = args.dA;
      stride_b = args.dB;
      layout_SFA = args.layout_SFA;
      layout_SFB = args.layout_SFB;
    }

    // Batches/Groups are managed by using appropriate pointers to input matrices.
    Tensor tensor_a = recast<TmaInternalElementA>(make_tensor(ptr_A_first_batch, make_layout(make_shape(init_M,init_K,init_L), stride_a)));
    Tensor tensor_b = recast<TmaInternalElementB>(make_tensor(ptr_B_first_batch, make_layout(make_shape(init_N,init_K,init_L), stride_b)));

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);

    // Cluster layout for TMA construction
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));
    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));

    // Tensor pointers will be fixed before the first access
    ElementSF const* ptr_SFA_first_batch = nullptr;
    ElementSF const* ptr_SFB_first_batch = nullptr;

    Tensor tensor_sfa = make_tensor(ptr_SFA_first_batch, layout_SFA);
    Tensor tensor_sfb = make_tensor(ptr_SFB_first_batch, layout_SFB);

    typename Params::TMA_A tma_load_a = make_tma_atom<uint8_t>(
      GmemTiledCopyA{},
      tensor_a,
      SmemLayoutA_tma{},
      make_tile(size<1,0>(typename TiledMma::ALayout{}), _384{}),
      size<1>(cluster_shape)
    );

    typename Params::TMA_B tma_load_b = make_tma_atom<uint8_t>(
      GmemTiledCopyB{},
      tensor_b,
      SmemLayoutB_tma{},
      make_tile(size<1,0>(typename TiledMma::BLayout{}), _384{}),
      size<0>(cluster_shape)/size(typename TiledMma::AtomThrID{})
    );

    typename Params::TMA_A tma_load_a_fallback =  make_tma_atom<uint8_t>(
      GmemTiledCopyA{},
      tensor_a,
      SmemLayoutA_tma{},
      make_tile(size<1,0>(typename TiledMma::ALayout{}), _384{}),
      size<1>(cluster_shape_fallback)
    );

    typename Params::TMA_B tma_load_b_fallback = make_tma_atom<uint8_t>(
      GmemTiledCopyB{},
      tensor_b,
      SmemLayoutB_tma{},
      make_tile(size<1,0>(typename TiledMma::BLayout{}), _384{}),
      size<0>(cluster_shape_fallback)/size(typename TiledMma::AtomThrID{})
    );

    typename Params::TMA_SFA tma_load_sfa = make_tma_atom<uint8_t>(
      GmemTiledCopySFA{},
      tensor_sfa,
      SmemLayoutSFA{}(_,_,_,cute::Int<0>{}),
      make_shape(get<0>(MMA_SF_Tiler{}), get<2>(MMA_SF_Tiler{})),
      size<1>(cluster_shape)
    );

    typename Params::TMA_SFB tma_load_sfb = make_tma_atom<uint8_t>(
      GmemTiledCopySFB{},
      tensor_sfb,
      SmemLayoutSFB{}(_,_,_,cute::Int<0>{}),
      make_shape(get<1>(MMA_SF_Tiler{}), get<2>(MMA_SF_Tiler{})),
      size<0>(cluster_shape)/size(typename TiledMMA_SF::AtomThrID{})
    );

    typename Params::TMA_SFA tma_load_sfa_fallback = make_tma_atom<uint8_t>(
      GmemTiledCopySFA{},
      tensor_sfa,
      SmemLayoutSFA{}(_,_,_,cute::Int<0>{}),
      make_shape(get<0>(MMA_SF_Tiler{}), get<2>(MMA_SF_Tiler{})),
      size<1>(cluster_shape_fallback)
    );

    typename Params::TMA_SFB tma_load_sfb_fallback = make_tma_atom<uint8_t>(
      GmemTiledCopySFB{},
      tensor_sfb,
      SmemLayoutSFB{}(_,_,_,cute::Int<0>{}),
      make_shape(get<1>(MMA_SF_Tiler{}), get<2>(MMA_SF_Tiler{})),
      size<0>(cluster_shape_fallback)/size(typename TiledMMA_SF::AtomThrID{})
    );

    #if 0
    print("tma_load_a:\n");
    print(tma_load_a);
    print("tma_load_a.tma_desc:\n"); print(tma_load_a.tma_desc_);          print("\n");

    print("tma_load_b:\n");
    print(tma_load_b);
    print("tma_load_b.tma_desc:\n"); print(tma_load_b.tma_desc_);          print("\n");

    print("layout_SFA:      "); print(args.layout_SFA); print("\n");
    print("tma_load_sfa:\n");
    print(tma_load_sfa);
    print("tma_load_sfa.tma_desc:\n"); print(tma_load_sfa.tma_desc_);      print("\n");

    print("layout_SFB:      "); print(args.layout_SFB); print("\n");
    print("tma_load_sfb:\n");
    print(tma_load_sfb);
    print("tma_load_sfb.tma_desc:\n"); print(tma_load_sfb.tma_desc_);      print("\n");

    print("layout_sfa:      "); print(args.layout_SFA); print("\n");
    print("tma_load_sfa_fallback:\n");
    print(tma_load_sfa_fallback);
    print("tma_load_sfa_fallback.tma_desc:\n"); print(tma_load_sfa_fallback.tma_desc_);      print("\n");

    print("layout_sfb:      "); print(args.layout_SFB); print("\n");
    print("tma_load_sfb_fallback:\n");
    print(tma_load_sfb_fallback);
    print("tma_load_sfb_fallback.tma_desc:\n"); print(tma_load_sfb_fallback.tma_desc_);      print("\n");
    #endif

    return {
      tma_load_a,
      tma_load_b,
      tma_load_sfa,
      tma_load_sfb,
      tma_load_a_fallback,
      tma_load_b_fallback,
      tma_load_sfa_fallback,
      tma_load_sfb_fallback,
      args.layout_SFA,
      args.layout_SFB,
      hw_info.cluster_shape_fallback,
      args.runtime_data_type_a,
      args.runtime_data_type_b,
      reinterpret_cast<cute::TmaDescriptor*>(workspace),
      reinterpret_cast<ArrayElementA const**>(args.ptr_A),
      args.dA,
      reinterpret_cast<ArrayElementB const**>(args.ptr_B),
      args.dB,
      reinterpret_cast<ElementSF const**>(args.ptr_SFA),
      reinterpret_cast<ElementSF const**>(args.ptr_SFB)
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    constexpr uint32_t NumInputTensors = 4;
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies
    return (NumInputTensors * SizeOfCuTensorMap * sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cute::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cute::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), InternalStrideA{});
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), InternalStrideB{});
      }
    }

    if constexpr (IsRuntimeDataType && detail::is_sm10x_mxf4nvf4_input<ElementAMma>() && detail::is_sm10x_mxf4nvf4_input<ElementBMma>()) {
      bool is_compatible = (SFVecSize == 16 ||
                           (SFVecSize == 32 && is_same_v<ElementSF, cutlass::float_ue8m0_t>
                                            && args.runtime_data_type_a == cute::UMMA::MXF4Format::E2M1
                                            && args.runtime_data_type_b == cute::UMMA::MXF4Format::E2M1));
      if (!is_compatible) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: 2x mode (VectorSize=32) only supports float_e2m1_t for a/b types and ue8m0_t for sf type.\n");
      }
      implementable &= is_compatible;
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));  // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

    return acc_shape;
  }

  template <class FrgEngine, class FrgLayout>
  CUTLASS_DEVICE auto
  slice_accumulator(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, int stage) {
    return accumulators(_,_,_,stage);
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  get_mkl_shape_tensor (
      ProblemShape_MNKL const& problem_shape_MNKL) const {
    auto [M,N,K,L] = problem_shape_MNKL;
    const int32_t mock_L = 1;
    int K_recast = (K*cute::sizeof_bits_v<ElementA>/8);

    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K_recast,mock_L));
    Tensor gA_mkl = local_tile(mA_mkl, replace<2>(TileShape{}, _384{}), make_coord(_,_,_), Step<_1, X,_1>{});
    return gA_mkl;
  }

  /// Set up the data needed by this collective for load.
  /// Return tuple element contain
  /// gA_mkl - The tiled tma tensor for input A
  /// gB_nkl - The tiled tma tensor for input B
  /// tAgA_mkl - partitioned gmem tensor for A
  /// tBgB_nkl - partitioned gmem tensor for B
  /// tAsA - partitioned smem tensor for A
  /// tBsB - partitioned smem tensor for B
  /// mcast_mask_a - tma multicast mask for A
  /// mcast_mask_b - tma multicast mask for B
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_ab_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_tensors,
      TensorMapStorage& shared_tensormaps,
      int32_t const sm_count, int32_t const sm_idx) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    const int32_t mock_L = 1;
    int K_recast = (K*cute::sizeof_bits_v<ElementA>/8);

    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K_recast,mock_L));
    Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N,K_recast,mock_L));

    // Tile the tensors and defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, replace<2>(TileShape{}, _384{}), make_coord(_,_,_), Step<_1, X,_1>{});    // (BLK_M, BLK_K, m, k, l)
    Tensor gB_nkl = local_tile(mB_nkl, replace<2>(TileShape{}, _384{}), make_coord(_,_,_), Step< X,_1,_1>{});    // (BLK_N, BLK_K, n, k, l)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mkl_tmp = cta_mma.partition_A(gA_mkl);                                       // ((CTA_MMA_M,96),Rest_MMA_M,Rest_MMA_K, m, k, l)
    Tensor cta_tCgA = make_tensor(tCgA_mkl_tmp.data(), make_layout(coalesce(make_layout(cute::layout<0,0>(tCgA_mkl_tmp), cute::layout<1>(tCgA_mkl_tmp))),
                                                                   coalesce(make_layout(cute::layout<0,1>(tCgA_mkl_tmp), cute::layout<2>(tCgA_mkl_tmp))),
                                                                   cute::layout<3>(tCgA_mkl_tmp), cute::layout<4>(tCgA_mkl_tmp), cute::layout<5>(tCgA_mkl_tmp)));   // (CTA_M,CTA_K,m,k,l)

    Tensor tCgA_mkl = make_tensor(cta_tCgA.data(), tiled_divide(cta_tCgA.layout(),
                                                                make_tile(size<1,0>(typename TiledMma::ALayout{}) /*MMA_M for SM100*/,
                                                                _128{} /*128bytes*/)));      // ((CTA_MMA_M,256),Rest_MMA_M,Rest_MMA_K, m, k, l)

    Tensor tCgB_nkl_tmp = cta_mma.partition_B(gB_nkl);                                       // ((MMA_ATOM_M,96),Rest_MMA_M,Rest_MMA_K, n, k, l)
    Tensor cta_tCgB = make_tensor(tCgB_nkl_tmp.data(), make_layout(coalesce(make_layout(cute::layout<0,0>(tCgB_nkl_tmp), cute::layout<1>(tCgB_nkl_tmp))),
                                                                   coalesce(make_layout(cute::layout<0,1>(tCgB_nkl_tmp), cute::layout<2>(tCgB_nkl_tmp))),
                                                                  cute::layout<3>(tCgB_nkl_tmp), cute::layout<4>(tCgB_nkl_tmp), cute::layout<5>(tCgB_nkl_tmp)));   // (CTA_M,CTA_K,m,k,l)
    Tensor tCgB_nkl = make_tensor(cta_tCgB.data(), tiled_divide(cta_tCgB.layout(),
                                                                make_tile(size<1,0>(typename TiledMma::BLayout{}) /*MMA_M for SM100*/,
                                                                _128{} /*128bytes*/)));      // ((CTA_MMA_M,256),Rest_MMA_M, Rest_MMA_K, m, k, l)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});    // ((CTA_MMA_M,32),Rest_MMA_M,8,NUM_PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});    // ((CTA_MMA_N,32),Rest_MMA_N,8,NUM_PIPE)


    Layout cta_layout_mnk  = make_layout(cutlass::detail::select_cluster_shape(ClusterShape{}, cute::cluster_shape()));
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    int block_rank_in_cluster = cute::block_rank_in_cluster();
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

    Layout cta_layout_sfb_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMMA_SF::AtomThrID{}));
    auto cta_coord_sfb_vmnk  = cta_layout_sfb_vmnk.get_flat_coord(block_rank_in_cluster);

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sA), group_modes<0,1>(tCgA_mkl));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgB_nkl, tBsB] = tma_partition(*observed_tma_load_b_,
                                      get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                      group_modes<0,3>(sB), group_modes<0,1>(tCgB_nkl));

    // TMA Multicast Masks
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);
    // Fetch a copy of tensormaps for the CTA from Params
    auto input_tensormaps = tensormaps_init_ab(params, shared_tensormaps, sm_count, sm_idx);

    return cute::make_tuple(
      gA_mkl, gB_nkl,                         // for scheduler
      tAgA_mkl, tBgB_nkl, tAsA, tBsB,         // for input tensor values
      mcast_mask_a, mcast_mask_b,            // multicast masks
      input_tensormaps);                      // for tma descriptor modification (per-CTA tensormap copy)
  }


  /// Set up the data needed by this collective for load.
  /// Return tuple element contain
  /// tAgA_mkl - partitioned gmem tensor for A
  /// tBgB_nkl - partitioned gmem tensor for B
  /// mcast_mask_sfa - tma multicast mask for SFA
  /// mcast_mask_sfb - tma multicast mask for SFB
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_sf_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_tensors,
      TensorMapStorage& shared_tensormaps,
      int32_t const sm_count, int32_t const sm_idx,
      int32_t init_group) const {
    using X = Underscore;

    // Separate out problem shape for convenience

    InternalLayoutSFA layout_SFA{};
    InternalLayoutSFB layout_SFB{};
    if constexpr (IsGroupedGemmKernel) {
      layout_SFA = params.layout_SFA[init_group];
      layout_SFB = params.layout_SFB[init_group];
    }
    else {
      layout_SFA = params.layout_SFA;
      layout_SFB = params.layout_SFB;
    }

    // Represent the full tensor of Scale factors
    Tensor mSFA_mkl = observed_tma_load_sfa_->get_tma_tensor(shape(layout_SFA));
    auto mSFB_nkl = [=](){
      if constexpr (IsCtaN192) {
        Tensor mSFB_tmp = observed_tma_load_sfb_->get_tma_tensor(shape(layout_SFB));
        auto x = stride<0,1>(mSFB_tmp);
        auto y = ceil_div(shape<0,1>(mSFB_tmp), 4);
        auto  new_shape =  make_shape (make_shape( shape<0,0>(mSFB_tmp),
                                       make_shape( make_shape(_2{}, _2{}),   y)),  shape<1>(mSFB_tmp), shape<2>(mSFB_tmp));
        auto new_stride = make_stride(make_stride(stride<0,0>(mSFB_tmp),
                                      make_stride(make_stride(   x,    x), x*3)), stride<1>(mSFB_tmp), stride<2>(mSFB_tmp));
        return make_tensor(mSFB_tmp.data(), make_layout(new_shape, new_stride));
      }
      else {
        return observed_tma_load_sfb_->get_tma_tensor(shape(layout_SFB));
      }
    }();

    // Partition for this CTA
    Tensor gSFA_mkl = local_tile(mSFA_mkl, MMA_SF_Tiler{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (TILE_M,TILE_K,m,k,l)
    Tensor gSFB_nkl = local_tile(mSFB_nkl, MMA_SF_Tiler{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (TILE_N,TILE_K,n,k,l)

    Tensor tCgSFA_mkl = make_tensor(gSFA_mkl.data(), tiled_divide(gSFA_mkl.layout(), make_tile(get<0>(MMA_SF_Tiler{}), get<2>(MMA_SF_Tiler{})))); // ((MMA_M,MMA_K),Rest_MMA_M,Rest_MMA_K, m, k, l)
    Tensor tCgSFB_nkl = make_tensor(gSFB_nkl.data(), tiled_divide(gSFB_nkl.layout(), make_tile(get<1>(MMA_SF_Tiler{}), get<2>(MMA_SF_Tiler{})))); // ((MMA_N,MMA_K),Rest_MMA_N,Rest_MMA_K, n, k, l)

    Tensor tCsSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor tCsSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

    Layout cta_layout_mnk  = make_layout(cutlass::detail::select_cluster_shape(ClusterShape{}, cute::cluster_shape()));
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    int block_rank_in_cluster = cute::block_rank_in_cluster();
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

    Layout cta_layout_sfb_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMMA_SF::AtomThrID{}));
    auto cta_coord_sfb_vmnk  = cta_layout_sfb_vmnk.get_flat_coord(block_rank_in_cluster);
    // Project the cta_layout for tma_a along the n-modes
    auto [tAgSFA_mkl, tAsSFA] = tma_partition(*observed_tma_load_sfa_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(tCsSFA), group_modes<0,3>(tCgSFA_mkl));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgSFB_nkl, tBsSFB] = tma_partition(*observed_tma_load_sfb_,
                                      get<1>(cta_coord_sfb_vmnk), make_layout(size<1>(cta_layout_sfb_vmnk)),
                                      group_modes<0,3>(tCsSFB), group_modes<0,3>(tCgSFB_nkl));

    // TMA Multicast Masks
    uint16_t mcast_mask_sfa = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_sfb = create_tma_multicast_mask<1>(cta_layout_sfb_vmnk, cta_coord_sfb_vmnk);

    auto input_tensormaps = tensormaps_init_sf(params, shared_tensormaps, sm_count, sm_idx);

    return cute::make_tuple(
      tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB, // for input scale factor tensor values
      mcast_mask_sfa, mcast_mask_sfb,         // multicast masks
      input_tensormaps);                        // for tma descriptor modification (per-CTA tensormap copy)
  }

  /// Set up the data needed by this collective for mma compute.
  CUTLASS_DEVICE auto
  mma_init(
    Params const& params,
    TensorStorage& shared_tensors,
    uint32_t const tmem_offset) const {

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});    // ((CTA_MMA_M,32),Rest_MMA_M,8,NUM_PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});    // ((CTA_MMA_M,32),Rest_MMA_M,8,NUM_PIPE)

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA = make_tensor<typename TiledMma::FrgTypeA>(sA);;
    Tensor tCrB = make_tensor<typename TiledMma::FrgTypeB>(sB);;

    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::LoadABPipelineStageCount>{} == size<3>(sA));                                     // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::LoadABPipelineStageCount>{} == size<3>(sB));                                     // PIPE

    //
    // Scale Factor
    //
    Tensor tCtSFA = make_tensor<typename TiledMma::FrgTypeSFA>(take<0,3>(shape(SmemLayoutAtomSFA{})));
    // TMEM allocations for SFA and SFB will always start at DP 0.
    tCtSFA.data() = tmem_offset;
    Tensor tCtSFB = make_tensor<typename TiledMma::FrgTypeSFB>(take<0,3>(shape(SmemLayoutAtomSFB{})));

    tCtSFB.data() = tCtSFA.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tCtSFA);

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
    auto tCtSFA_compact_copy = make_tensor(tCtSFA_compact.data(), append<3>(tCtSFA_compact(_,_0{},_0{}).layout()));
    auto tCtSFB_compact_copy = make_tensor(tCtSFB_compact.data(), append<3>(tCtSFB_compact(_,_0{},_0{}).layout()));
    auto tiled_copy_s2t_SFA = make_utccp_copy(UtccpOp{}, tCtSFA_compact_copy);
    auto tiled_copy_s2t_SFB = make_utccp_copy(UtccpOp{}, tCtSFB_compact_copy);

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
      tiled_mma.idesc_.a_format_ = uint8_t(params.runtime_data_type_a) & 0b111;
      tiled_mma.idesc_.b_format_ = uint8_t(params.runtime_data_type_b) & 0b111;
    }

    // using MMA_SF_Tiler = decltype(make_tile(shape<0>(CtaShape_MNK{}), Int<CTA_N_SF>{}, Int<shape<2>(CtaShape_MNK{})/2>{}));  // 128x128x384
    // MMA shapes are ((_128,_96),_1,_8) which makes the MMA_SFA_Shape ((128, (16,3)), 1, 8/3)
    // The number is not divisible by 4 in K dimension which is needed for TMEM allocation.
    // To be able to iterate thru the SFs for MMA, we model this as (MMA), MMA_M, MMA_K: ((128, (16,1)), 1, 24)
    // with this layout we can iterate thru the SFs by incrementing MMA_K mode by 3/6 for this example (Vs=16 vs Vs=32).
    constexpr int MMA_M = size<0>(CtaShape_MNK{});
    constexpr int MMA_N_SF = CTA_N_SF;
    constexpr int MMA_K_SF = shape<2>(CtaShape_MNK{}) / 2;
    auto mnBasicBlockShape  =  make_shape(_32{}, _4{});
    auto kBasicBlockShape_single   = make_shape(Int<SFVecSize>{}, Int<1>{});
    auto mma_iter_SFA_shape  = make_shape( prepend(Int<MMA_M/128>{},  mnBasicBlockShape),  kBasicBlockShape_single);
    auto sSFA_iter_shape  =   make_shape(mma_iter_SFA_shape,  _1{},  Int<MMA_K_SF/SFVecSize>{});
    auto mma_iter_SFB_shape  = make_shape( prepend(Int<MMA_N_SF/128>{},  mnBasicBlockShape),  kBasicBlockShape_single);
    auto sSFB_iter_shape  =   make_shape(mma_iter_SFB_shape,  _1{},  Int<MMA_K_SF/SFVecSize>{});

    // Used for MMAs
    using MmaIterShapeSFA = decltype(sSFA_iter_shape);  // ((32,4),(SFVecSize,1), MMA_M/128, SF_MMA_K/SfVecSize
    using MmaIterShapeSFB = decltype(sSFB_iter_shape);  // ((32,4),(SFVecSize,1), MMA_N/128, SF_MMA_K/SfVecSize

    Tensor tCtSFA_mma = make_tensor<typename TiledMma::FrgTypeSFA>(MmaIterShapeSFA{});
    tCtSFA_mma.data() = tCtSFA.data();
    Tensor tCtSFB_mma = make_tensor<typename TiledMma::FrgTypeSFB>(MmaIterShapeSFB{});
    tCtSFB_mma.data() = tCtSFB.data();

    return cute::make_tuple(
      tiled_mma,
      tCrA, tCrB, tCtSFA, tCtSFB, tCtSFA_mma, tCtSFB_mma,
      tiled_copy_s2t_SFA, thr_tCsSFA_compact_s2t, thr_tCtSFA_compact_s2t,
      tiled_copy_s2t_SFB, thr_tCsSFB_compact_s2t, thr_tCtSFB_compact_s2t);
  }

// Helper function to handle both prefetch types
  template <int BuffersPerKtile, typename TmaPrefetchFn, typename KTileIterator>
  CUTLASS_DEVICE void issue_prefetch(
      int& prefetch_k_tile_count,
      int& prefetch_buf_idx,
      KTileIterator& prefetch_k_tile,
      TmaPrefetchFn&& tma_prefetch_fn)
  {
    if (prefetch_k_tile_count > 0) {
      if constexpr (PrefetchType == cutlass::sm103::detail::KernelPrefetchType::TmaPrefetch) {
        tma_prefetch_fn();
      }

      prefetch_buf_idx = (prefetch_buf_idx + 1) % BuffersPerKtile;
      if(prefetch_buf_idx == 0) {
        ++prefetch_k_tile;
        --prefetch_k_tile_count;
      }
    }
  }


  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class GTensorA, class GTensorB,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB,
    class TensorMapA, class TensorMapB,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load_ab(
    Params const& params,
    MainloopABPipeline pipeline,
    MainloopABPipelineState mainloop_pipe_producer_state,
    cute::tuple<GTensorA, GTensorB,
                GTensorPartitionedA, GTensorPartitionedB,
                STensorA, STensorB,
                uint16_t, uint16_t,
                cute::tuple<TensorMapA, TensorMapB>> const& load_inputs,
    TileCoordMNKL const& cta_coord_mnkl,
    KTileIterator k_tile_iter, int k_tile_count, 
    bool did_batch_change, int prefetch_k_tile_count = 0) {

    auto tAgA_mkl = get<2>(load_inputs);
    auto tBgB_nkl = get<3>(load_inputs);
    auto tAsA = get<4>(load_inputs);
    auto tBsB = get<5>(load_inputs);
    auto mcast_mask_a = get<6>(load_inputs);
    auto mcast_mask_b = get<7>(load_inputs);
    auto input_tensormaps = get<8>(load_inputs);

    if (did_batch_change) {
      tensormaps_fence_acquire(get<0>(input_tensormaps));
      tensormaps_fence_acquire(get<1>(input_tensormaps));
    }
    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mkl(_, _, _, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgB = tBgB_nkl(_, _, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    auto barrier_token = pipeline.producer_try_acquire(mainloop_pipe_producer_state);
    constexpr int BuffersPerKtile = 3;
    auto prefetch_k_tile = k_tile_iter;
    auto prefetch_buf_idx = 0;
    auto tile_k_advance = LoadABPipelineStageCount / BuffersPerKtile;

    if constexpr (PrefetchType != cutlass::sm103::detail::KernelPrefetchType::Disable) {
      prefetch_buf_idx = LoadABPipelineStageCount % BuffersPerKtile;
      CUTLASS_PRAGMA_UNROLL
      for (int i=0;i<tile_k_advance;i++) {
        ++prefetch_k_tile;
        --prefetch_k_tile_count;
      }
    }

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      using BarrierType = typename MainloopABPipeline::ProducerBarrierType;
      // In total, we will load 3 buffers per k_tile_iter. Unrolled.
      CUTLASS_PRAGMA_UNROLL
      for(int buffer = 0; buffer < BuffersPerKtile; buffer++) {
        pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);
        BarrierType* tma_barrier = pipeline.producer_get_barrier(mainloop_pipe_producer_state);
        int write_stage = mainloop_pipe_producer_state.index();
        ++mainloop_pipe_producer_state;
        barrier_token = pipeline.producer_try_acquire(mainloop_pipe_producer_state);

        auto tma_copy_traits_a = observed_tma_load_a_->with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a);
        auto tma_copy_traits_b = observed_tma_load_b_->with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b);

        if (cute::elect_one_sync()) {
          copy(tma_copy_traits_a, group_modes<0,2>(tAgA(_,_,buffer,*k_tile_iter)), tAsA(_,write_stage));
          copy(tma_copy_traits_b, group_modes<0,2>(tBgB(_,_,buffer,*k_tile_iter)), tBsB(_,write_stage));
        }

        if constexpr (PrefetchType != cutlass::sm103::detail::KernelPrefetchType::Disable) {
          issue_prefetch <BuffersPerKtile>(
            prefetch_k_tile_count,
            prefetch_buf_idx,
            prefetch_k_tile,
            [&]() {
              prefetch(tma_copy_traits_a, group_modes<0,2>(tAgA(_,_,prefetch_buf_idx,*prefetch_k_tile)));
              prefetch(tma_copy_traits_b, group_modes<0,2>(tBgB(_,_,prefetch_buf_idx,*prefetch_k_tile)));
            }
          );
        }
      }

      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
  }


  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class GTensorPartitionedSFA, class GTensorPartitionedSFB,
    class STensorSFA, class STensorSFB,
    class TensorMapSFA, class TensorMapSFB,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load_sf(
    Params const& params,
    MainloopSFPipeline pipeline,
    MainloopSFPipelineState mainloop_sf_pipe_producer_state,
    cute::tuple<GTensorPartitionedSFA, GTensorPartitionedSFB,
                STensorSFA, STensorSFB,
                uint16_t, uint16_t,
                cute::tuple<TensorMapSFA, TensorMapSFB>> const& load_inputs,
    TileCoordMNKL const& cta_coord_mnkl,
    KTileIterator k_tile_iter, int k_tile_count, 
    bool did_batch_change, int prefetch_k_tile_count = 0) {

    auto tAgSFA_mkl = get<0>(load_inputs);
    auto tBgSFB_nkl = get<1>(load_inputs);
    auto tAsSFA = get<2>(load_inputs);
    auto tBsSFB = get<3>(load_inputs);
    auto mcast_mask_sfa = get<4>(load_inputs);
    auto mcast_mask_sfb = get<5>(load_inputs);
    auto input_tensormaps_sf = get<6>(load_inputs);
    // slice out the work coord from partitioned tensors
    Tensor tAgSFA = tAgSFA_mkl(_, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
    Tensor tBgSFB = tBgSFB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    // Check to see if tensormaps have been replaced in gmem
    if (did_batch_change) {
      tensormaps_fence_acquire(get<0>(input_tensormaps_sf));
      tensormaps_fence_acquire(get<1>(input_tensormaps_sf));
    }

    auto barrier_token = pipeline.producer_try_acquire(mainloop_sf_pipe_producer_state);

    using BarrierType = typename MainloopSFPipeline::ProducerBarrierType;
    auto tAsSFA_compact = make_tensor(tAsSFA.data(), filter_zeros(tAsSFA.layout()));
    auto tBsSFB_compact = make_tensor(tBsSFB.data(), filter_zeros(tBsSFB.layout()));
    auto prefetch_k_tile = k_tile_iter;
    auto prefetch_buf_idx = 0;
    auto tile_k_advance = LoadSFPipelineStageCount / SF_BUFFERS_PER_TILE_K;

    if constexpr (PrefetchType != cutlass::sm103::detail::KernelPrefetchType::Disable) {
      prefetch_buf_idx = LoadSFPipelineStageCount % SF_BUFFERS_PER_TILE_K;
      CUTLASS_PRAGMA_UNROLL
      for (int i=0;i<tile_k_advance;i++) {
        ++prefetch_k_tile;
        --prefetch_k_tile_count;
      }
    }

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // In total, we will load 2 or 4 buffers per k_tile_iter. Unrolled.
      CUTLASS_PRAGMA_UNROLL
      for(int buffer = 0; buffer < SF_BUFFERS_PER_TILE_K; buffer++) {
        pipeline.producer_acquire(mainloop_sf_pipe_producer_state, barrier_token);
        BarrierType* tma_barrier = pipeline.producer_get_barrier(mainloop_sf_pipe_producer_state);

        int write_stage = mainloop_sf_pipe_producer_state.index();
        ++mainloop_sf_pipe_producer_state;
        barrier_token = pipeline.producer_try_acquire(mainloop_sf_pipe_producer_state);
        auto tAgSFA_compact = make_tensor(tAgSFA(_,*k_tile_iter*SF_BUFFERS_PER_TILE_K + buffer).data(), filter_zeros(tAgSFA(_,*k_tile_iter*SF_BUFFERS_PER_TILE_K + buffer).layout()));
        auto tBgSFB_compact = make_tensor(tBgSFB(_,*k_tile_iter*SF_BUFFERS_PER_TILE_K + buffer).data(), filter_zeros(tBgSFB(_,*k_tile_iter*SF_BUFFERS_PER_TILE_K + buffer).layout()));

        auto tma_copy_traits_sfa = observed_tma_load_sfa_->with(get<0>(input_tensormaps_sf), *tma_barrier, mcast_mask_sfa);
        auto tma_copy_traits_sfb = observed_tma_load_sfb_->with(get<1>(input_tensormaps_sf), *tma_barrier, mcast_mask_sfb);

        if (cute::elect_one_sync()) {
          copy(observed_tma_load_sfa_->with(get<0>(input_tensormaps_sf), *tma_barrier, mcast_mask_sfa), tAgSFA_compact, tAsSFA_compact(_,write_stage));
          copy(observed_tma_load_sfb_->with(get<1>(input_tensormaps_sf), *tma_barrier, mcast_mask_sfb), tBgSFB_compact, tBsSFB_compact(_,write_stage));
        }

        auto tAgSFA_compact_prefetch = make_tensor(tAgSFA(_,*prefetch_k_tile*SF_BUFFERS_PER_TILE_K + prefetch_buf_idx).data(), filter_zeros(tAgSFA(_,*prefetch_k_tile*SF_BUFFERS_PER_TILE_K + prefetch_buf_idx).layout()));
        auto tBgSFB_compact_prefetch = make_tensor(tBgSFB(_,*prefetch_k_tile*SF_BUFFERS_PER_TILE_K + prefetch_buf_idx).data(), filter_zeros(tBgSFB(_,*prefetch_k_tile*SF_BUFFERS_PER_TILE_K + prefetch_buf_idx).layout()));
        if constexpr (PrefetchType != cutlass::sm103::detail::KernelPrefetchType::Disable) {
          issue_prefetch <SF_BUFFERS_PER_TILE_K>(
            prefetch_k_tile_count,
            prefetch_buf_idx,
            prefetch_k_tile,
            [&]() {
              prefetch(tma_copy_traits_sfa, tAgSFA_compact_prefetch);
              prefetch(tma_copy_traits_sfb, tBgSFB_compact_prefetch);
            }
          );
        }
      }

      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_sf_pipe_producer_state, k_tile_iter);
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
    template <
    class MainloopPipeline, class MainloopPipelineState
  >
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, MainloopPipelineState mainloop_pipe_producer_state) {
    // Issue the epilogue waits
    // This helps avoid early exit of ctas in Cluster
    // Waits for all stages to either be released (all
    // Consumer UNLOCKs), or if the stage was never used
    // then would just be acquired since the phase was
    // still inverted from make_producer_start_state
    pipeline.producer_tail(mainloop_pipe_producer_state);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class AccumulatorPipeline,
    class FrgEngine, class FrgLayout,
    class FragmentA, class FragmentB,
    class FragmentSFA, class FragmentSFB,
    class MmaFragmentSFA, class MmaFragmentSFB,
    class CtaTileCoord,
    class SFATiledCopy, class SmemFrgSFA, class TmemFrgSFA,
    class SFBTiledCopy, class SmemFrgSFB, class TmemFrgSFB
  >
  CUTLASS_DEVICE auto
  mma(cute::tuple<MainloopABPipeline,MainloopSFPipeline,AccumulatorPipeline> pipelines,
      cute::tuple<MainloopABPipelineState,MainloopSFPipelineState, typename AccumulatorPipeline::PipelineState> pipeline_states,
      cute::Tensor<FrgEngine, FrgLayout>& accumulators,
      cute::tuple<TiledMma,
                  FragmentA, FragmentB,
                  FragmentSFA, FragmentSFB, MmaFragmentSFA, MmaFragmentSFB,
                  SFATiledCopy, SmemFrgSFA, TmemFrgSFA,
                  SFBTiledCopy, SmemFrgSFB, TmemFrgSFB> const& mma_inputs,
      CtaTileCoord cta_tile_coord,
      int k_tile_count
  ) {
    static_assert(is_tmem<FrgEngine>::value, "Accumulator must be tmem resident.");
    static_assert(rank(FrgLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");
    auto pipeline_ab = get<0>(pipelines);
    auto pipeline_sf = get<1>(pipelines);
    auto accumulator_pipeline = get<2>(pipelines);
    auto mainloop_pipe_ab_consumer_state = get<0>(pipeline_states);
    auto mainloop_pipe_sf_consumer_state = get<1>(pipeline_states);
    auto accumulator_pipe_producer_state = get<2>(pipeline_states);
    auto tiled_mma  = get<0>(mma_inputs);
    auto tCrA       = get<1>(mma_inputs);
    auto tCrB       = get<2>(mma_inputs);
    auto tCtSFA     = get<3>(mma_inputs);
    auto tCtSFB     = get<4>(mma_inputs);
    auto tCtSFA_mma = get<5>(mma_inputs);
    auto tCtSFB_mma = get<6>(mma_inputs);
    auto tiled_copy_s2t_SFA = get<7>(mma_inputs);
    auto tCsSFA_s2t     = get<8>(mma_inputs);
    auto tCtSFA_s2t     = get<9>(mma_inputs);
    auto tiled_copy_s2t_SFB = get<10>(mma_inputs);
    auto tCsSFB_s2t     = get<11>(mma_inputs);
    auto tCtSFB_s2t     = get<12>(mma_inputs);

    tCtSFB_mma = [tCtSFB_mma = tCtSFB_mma, cta_tile_coord]() {
      if constexpr (IsCtaN192) {
        // If this is an ODD tile, shift the TMEM start address for N=192 case by two words (ignores first 64 columns of SFB)
        auto tCtSFB_tmp = tCtSFB_mma;
        if (get<1>(cta_tile_coord) % 2 == 1) {
          tCtSFB_tmp.data() = tCtSFB_tmp.data().get() + 2;
        }
        return tCtSFB_tmp;
      }
      else {
        return tCtSFB_mma;
      }
    }();

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    constexpr int sf_stride = TiledMma::SFVecSize == 16 ? 6 : 3;
    auto barrier_token_ab = pipeline_ab.consumer_try_wait(mainloop_pipe_ab_consumer_state);
    auto barrier_token_sf = pipeline_sf.consumer_try_wait(mainloop_pipe_sf_consumer_state);
    constexpr int MmasPerSfBuffer = 8 / SF_BUFFERS_PER_TILE_K;

    auto sf_load_fn = [&](const int kphase, const int k_tile_count) {
      if (kphase % MmasPerSfBuffer == 0) {
        pipeline_sf.consumer_wait(mainloop_pipe_sf_consumer_state, barrier_token_sf);
        int read_stage_sf_buffer0 = mainloop_pipe_sf_consumer_state.index();
        if (cute::elect_one_sync()) {
          copy(tiled_copy_s2t_SFA, tCsSFA_s2t(_,_,_,_,read_stage_sf_buffer0), tCtSFA_s2t);
          copy(tiled_copy_s2t_SFB, tCsSFB_s2t(_,_,_,_,read_stage_sf_buffer0), tCtSFB_s2t);
        }
        auto buffer0_mainloop_pipe_sf_consumer_state = mainloop_pipe_sf_consumer_state;
        ++mainloop_pipe_sf_consumer_state;
        barrier_token_sf = pipeline_sf.consumer_try_wait(mainloop_pipe_sf_consumer_state, (kphase == 8 - MmasPerSfBuffer) && k_tile_count <= 1); // only skip wait for the last one.
        pipeline_sf.consumer_release(buffer0_mainloop_pipe_sf_consumer_state);
      }
    };

    bool is_first_iteration = true;
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // MMA 0
      sf_load_fn(0, k_tile_count);
      pipeline_ab.consumer_wait(mainloop_pipe_ab_consumer_state, barrier_token_ab);
      int read_stage_ab_buffer0 = mainloop_pipe_ab_consumer_state.index();
      auto buffer0_mainloop_pipe_ab_consumer_state = mainloop_pipe_ab_consumer_state;
      ++mainloop_pipe_ab_consumer_state;
      barrier_token_ab = pipeline_ab.consumer_try_wait(mainloop_pipe_ab_consumer_state);

      // delay the acc acquire to unblock tmem copy.
      if constexpr (IsOverlappingAccum) {
        if(is_first_iteration) {
          accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
          is_first_iteration = false;
        }
      };

      cute::gemm(tiled_mma,
      make_zip_tensor(tCrA(_,_,0,read_stage_ab_buffer0),  // A buffer: Points to buffer[0]
                      tCrA(_,_,0,read_stage_ab_buffer0),  // Next A buffer for circular buffers: Points to buffer[0]
                      tCtSFA_mma(_, _, 0 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFA
      make_zip_tensor(tCrB(_,_,0,read_stage_ab_buffer0),  // B buffer: Points to buffer[0]
                      tCrB(_,_,0,read_stage_ab_buffer0),  // Next B buffer for circular buffers: Points to buffer[0]
                      tCtSFB_mma(_, _, 0 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFB
      accumulators);   // (V,M) x (V,N) => (V,M,N)

      tiled_mma.accumulate_ = UMMA::ScaleOut::One;

      // MMA 1
      sf_load_fn(1, k_tile_count);
      cute::gemm(tiled_mma,
        make_zip_tensor(tCrA(_,_,3,read_stage_ab_buffer0),  // A buffer: Points to buffer[0] + 48 bytes. Note the 3.
                        tCrA(_,_,0,read_stage_ab_buffer0),  // Next A buffer for circular buffers: Points to buffer[0]
                        tCtSFA_mma(_, _, 1 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFA
        make_zip_tensor(tCrB(_,_,3,read_stage_ab_buffer0),  // B buffer: Points to buffer[0] + 48 bytes. Note the 3.
                        tCrB(_,_,0,read_stage_ab_buffer0),  // Next B buffer for circular buffers: Points to buffer[0]
                        tCtSFB_mma(_, _, 1 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFB
        accumulators);   // (V,M) x (V,N) => (V,M,N)


      // MMA 2
      sf_load_fn(2, k_tile_count);
      pipeline_ab.consumer_wait(mainloop_pipe_ab_consumer_state, barrier_token_ab);
      int read_stage_ab_buffer1 = mainloop_pipe_ab_consumer_state.index();
      auto buffer1_mainloop_pipe_ab_consumer_state = mainloop_pipe_ab_consumer_state;
      ++mainloop_pipe_ab_consumer_state;
      barrier_token_ab = pipeline_ab.consumer_try_wait(mainloop_pipe_ab_consumer_state);

      cute::gemm(tiled_mma,
        make_zip_tensor(tCrA(_,_,6,read_stage_ab_buffer0),  // A buffer: Points to buffer[0] + 96 bytes. Note the 6.
                        tCrA(_,_,0,read_stage_ab_buffer1),  // Next A buffer for circular buffers: Points to buffer[1].
                        tCtSFA_mma(_, _, 2 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFA
        make_zip_tensor(tCrB(_,_,6,read_stage_ab_buffer0),  // B buffer: Points to buffer[0] + 96 bytes. Note the 6.
                        tCrB(_,_,0,read_stage_ab_buffer1),  // Next B buffer for circular buffers: Points to buffer[1].
                        tCtSFB_mma(_, _, 2 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFB
        accumulators);   // (V,M) x (V,N) => (V,M,N)

      pipeline_ab.consumer_release(buffer0_mainloop_pipe_ab_consumer_state);


      // MMA 3
      sf_load_fn(3, k_tile_count);
      cute::gemm(tiled_mma,
        make_zip_tensor(tCrA(_,_,1,read_stage_ab_buffer1),  // A buffer: Points to buffer[1] + 16 bytes. Note the 1.
                        tCrA(_,_,0,read_stage_ab_buffer1),  // Next A buffer for circular buffers: Points to buffer[1].
                        tCtSFA_mma(_, _, 3 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFA
        make_zip_tensor(tCrB(_,_,1,read_stage_ab_buffer1),  // B buffer: Points to buffer[1] + 16 bytes. Note the 1.
                        tCrB(_,_,0,read_stage_ab_buffer1),  // Next B buffer for circular buffers: Points to buffer[1].
                        tCtSFB_mma(_, _, 3 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFB
        accumulators);   // (V,M) x (V,N) => (V,M,N)

      // MMA 4
        sf_load_fn(4, k_tile_count);
      cute::gemm(tiled_mma,
        make_zip_tensor(tCrA(_,_,4,read_stage_ab_buffer1),  // A buffer: Points to buffer[1] + 64 bytes. Note the 1.
                        tCrA(_,_,0,read_stage_ab_buffer1),  // Next A buffer for circular buffers: Points to buffer[1].
                        tCtSFA_mma(_, _, 4 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFA
        make_zip_tensor(tCrB(_,_,4,read_stage_ab_buffer1),  // B buffer: Points to buffer[1] + 64 bytes. Note the 1.
                        tCrB(_,_,0,read_stage_ab_buffer1),  // Next B buffer for circular buffers: Points to buffer[1].
                        tCtSFB_mma(_, _, 4 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFB
        accumulators);   // (V,M) x (V,N) => (V,M,N)

      // MMA 5
      sf_load_fn(5, k_tile_count);
      pipeline_ab.consumer_wait(mainloop_pipe_ab_consumer_state, barrier_token_ab);
      int read_stage_ab_buffer2 = mainloop_pipe_ab_consumer_state.index();
      auto buffer2_mainloop_pipe_ab_consumer_state = mainloop_pipe_ab_consumer_state;
      ++mainloop_pipe_ab_consumer_state;
      barrier_token_ab = pipeline_ab.consumer_try_wait(mainloop_pipe_ab_consumer_state, k_tile_count <= 1);

      cute::gemm(tiled_mma,
        make_zip_tensor(tCrA(_,_,7,read_stage_ab_buffer1),  // A buffer: Points to buffer[1] + 112 bytes. Note the 7.
                        tCrA(_,_,0,read_stage_ab_buffer2),  // Next A buffer for circular buffers: Points to buffer[2].
                        tCtSFA_mma(_, _, 5 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFA
        make_zip_tensor(tCrB(_,_,7,read_stage_ab_buffer1),  // B buffer: Points to buffer[1] + 112 bytes. Note the 7.
                        tCrB(_,_,0,read_stage_ab_buffer2),  // Next B buffer for circular buffers: Points to buffer[2].
                        tCtSFB_mma(_, _, 5 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFB
        accumulators);   // (V,M) x (V,N) => (V,M,N)

      pipeline_ab.consumer_release(buffer1_mainloop_pipe_ab_consumer_state);

      // MMA 6
      sf_load_fn(6, k_tile_count);
      cute::gemm(tiled_mma,
        make_zip_tensor(tCrA(_,_,2,read_stage_ab_buffer2),  // A buffer: Points to buffer[1] + 32 bytes. Note the 2.
                        tCrA(_,_,0,read_stage_ab_buffer2),  // Next A buffer for circular buffers: Points to buffer[2].
                        tCtSFA_mma(_, _, 6 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFA
        make_zip_tensor(tCrB(_,_,2,read_stage_ab_buffer2),  // B buffer: Points to buffer[1] + 32 bytes. Note the 2.
                        tCrB(_,_,0,read_stage_ab_buffer2),  // Next B buffer for circular buffers: Points to buffer[2].
                        tCtSFB_mma(_, _, 6 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFB
        accumulators);   // (V,M) x (V,N) => (V,M,N)
      // MMA 7
      sf_load_fn(7, k_tile_count);
      cute::gemm(tiled_mma,
        make_zip_tensor(tCrA(_,_,5,read_stage_ab_buffer2),  // A buffer: Points to buffer[1] + 80 bytes. Note the 5.
                        tCrA(_,_,0,read_stage_ab_buffer2),  // Next A buffer for circular buffers: Points to buffer[2].
                        tCtSFA_mma(_, _, 7 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFA
        make_zip_tensor(tCrB(_,_,5,read_stage_ab_buffer2),  // B buffer: Points to buffer[1] + 80 bytes. Note the 5.
                        tCrB(_,_,0,read_stage_ab_buffer2),  // Next B buffer for circular buffers: Points to buffer[2].
                        tCtSFB_mma(_, _, 7 % MmasPerSfBuffer * sf_stride)),   // Tmem tensors for SFB
        accumulators);   // (V,M) x (V,N) => (V,M,N)

      pipeline_ab.consumer_release(buffer2_mainloop_pipe_ab_consumer_state);
      --k_tile_count;
    }
    return cute::make_tuple(mainloop_pipe_ab_consumer_state, mainloop_pipe_sf_consumer_state);
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //
  CUTLASS_DEVICE auto
  tensormaps_init_ab(
      Params const& mainloop_params,
      TensorMapStorage& shared_tensormaps,
      int32_t const sm_count,
      int32_t const sm_idx) const {
    cute::TmaDescriptor* gmem_tensormap = mainloop_params.tensormaps;

    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];

    if (cute::elect_one_sync()) {
      // Bringing tensormaps from params to smem for modification later
      Tensor pA_tensormap = make_tensor(observed_tma_load_a_->get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
      Tensor pB_tensormap = make_tensor(observed_tma_load_b_->get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});

      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));

    }
    __syncwarp();

    return cute::make_tuple(tma_desc_a, tma_desc_b);
  }

  // Replace address for the global tensor (to be done by single thread)
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address_ab(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                    mainloop_params.ptr_A[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                    mainloop_params.ptr_B[next_batch]);
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties_ab(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    const uint32_t K = get<2>(problem_shape_mnkl);
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};

    ElementA const* ptr_A = nullptr;
    Tensor tensor_a = recast<TmaInternalElementA>(make_tensor(ptr_A, make_shape(M,K,Int<1>{}), mainloop_params.dA[next_group]));

    ElementB const* ptr_B = nullptr;
    Tensor tensor_b = recast<TmaInternalElementB>(make_tensor(ptr_B, make_shape(N,K,Int<1>{}), mainloop_params.dB[next_group]));

    cute::detail::fill_tma_gmem_shape_stride(*observed_tma_load_a_, tensor_a,
                                             prob_shape_A, prob_stride_A);
    cute::detail::fill_tma_gmem_shape_stride(*observed_tma_load_b_, tensor_b,
                                             prob_shape_B, prob_stride_B);

    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_A) {
      stride = (stride * sizeof_bits_v<TmaInternalElementA>) / 8;
    }
    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<TmaInternalElementB>) / 8;
    }
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                            prob_shape_A,
                                                            prob_stride_A);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                            prob_shape_B,
                                                            prob_stride_B);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class TensorMapA, class TensorMapB, class ProblemShape>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update_ab(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TensorMapA, TensorMapB> const& input_ab_tensormaps,
      ProblemShape problem_shape,
      int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address_ab(shared_tensormaps, mainloop_params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        auto problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(next_batch), 1);
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties_ab(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_MNKL);
      }
    }
    // Ensure warp is converged before issuing tensormap fence release
    __syncwarp();
    // Entire warp must do this (ie its aligned)
    tensormaps_cp_fence_release_ab(shared_tensormaps, input_ab_tensormaps);
  }

  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release_ab (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TensorMapA, TensorMapB> const& input_ab_tensormaps) {
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_ab_tensormaps), shared_tensormaps.smem_tensormap_A);
    tma_descriptor_cp_fence_release(get<1>(input_ab_tensormaps), shared_tensormaps.smem_tensormap_B);

  }

  // SF tensormap ops
  CUTLASS_DEVICE auto
  tensormaps_init_sf(
      Params const& mainloop_params,
      TensorMapStorage& shared_tensormaps,
      int32_t const sm_count,
      int32_t const sm_idx) const {
    cute::TmaDescriptor* gmem_tensormap = mainloop_params.tensormaps;

    cute::TmaDescriptor* tma_desc_sfa = &gmem_tensormap[sm_idx + 2 * sm_count];
    cute::TmaDescriptor* tma_desc_sfb = &gmem_tensormap[sm_idx + 3 * sm_count];

    if (cute::elect_one_sync()) {
      // Bringing tensormaps from params to smem for modification later
      Tensor pSFA_tensormap = make_tensor(observed_tma_load_sfa_->get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sSFA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_SFA), Int<1>{}, Int<1>{});
      Tensor pSFB_tensormap = make_tensor(observed_tma_load_sfb_->get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sSFB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_SFB), Int<1>{}, Int<1>{});

      copy(recast<uint128_t>(pSFA_tensormap), recast<uint128_t>(sSFA_tensormap));
      copy(recast<uint128_t>(pSFB_tensormap), recast<uint128_t>(sSFB_tensormap));
    }
    __syncwarp();

    return cute::make_tuple(tma_desc_sfa, tma_desc_sfb);
  }

  // Replace address for the global tensor (to be done by single thread)
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address_sf(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_SFA,
                                                    mainloop_params.ptr_SFA[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_SFB,
                                                    mainloop_params.ptr_SFB[next_batch]);
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties_sf(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    const uint32_t K = get<2>(problem_shape_mnkl);
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_SFA  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_SFA = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_SFB  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_SFB = {0,0,0,0,0};

    ElementSF const* ptr_SF = nullptr;
    Tensor tensor_sfa = make_tensor(ptr_SF, mainloop_params.layout_SFA[next_group]);

    Tensor tensor_sfb = make_tensor(ptr_SF, mainloop_params.layout_SFB[next_group]);

    cute::detail::fill_tma_gmem_shape_stride(*observed_tma_load_sfa_, tensor_sfa,
                                             prob_shape_SFA, prob_stride_SFA);
    cute::detail::fill_tma_gmem_shape_stride(*observed_tma_load_sfb_, tensor_sfb,
                                             prob_shape_SFB, prob_stride_SFB);

    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_SFA) {
      stride = (stride * sizeof_bits_v<ElementSF>) / 8;
    }
    for (uint64_t& stride : prob_stride_SFB) {
      stride = (stride * sizeof_bits_v<ElementSF>) / 8;
    }

    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_SFA,
                                                            prob_shape_SFA,
                                                            prob_stride_SFA);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_SFB,
                                                            prob_shape_SFB,
                                                            prob_stride_SFB);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class TensorMapSFA, class TensorMapSFB, class ProblemShape>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update_sf(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TensorMapSFA, TensorMapSFB> const& input_tensormaps_sf,
      ProblemShape problem_shape,
      int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address_sf(shared_tensormaps, mainloop_params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        auto problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(next_batch), 1);
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties_sf(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_MNKL);
      }
    }
    // Ensure warp is converged before issuing tensormap fence release
    __syncwarp();
    // Entire warp must do this (ie its aligned)
    tensormaps_cp_fence_release_sf(shared_tensormaps, input_tensormaps_sf);
  }

  template <class TensorMapSFA, class TensorMapSFB>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release_sf (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TensorMapSFA, TensorMapSFB> const& input_tensormaps_sf) {
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps_sf), shared_tensormaps.smem_tensormap_SFA);
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps_sf), shared_tensormaps.smem_tensormap_SFB);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::TmaDescriptor const* input_tma_desc) {
    cute::tma_descriptor_fence_acquire(input_tma_desc);
  }

protected:
  typename Params::TMA_A const* observed_tma_load_a_{nullptr};
  typename Params::TMA_B const* observed_tma_load_b_{nullptr};
  typename Params::TMA_SFA const* observed_tma_load_sfa_{nullptr};
  typename Params::TMA_SFB const* observed_tma_load_sfb_{nullptr};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
