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
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/cuda_host_adapter.hpp"

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
    MainloopSm100ArrayTmaUmmaWarpSpecializedBlockwiseScaling<
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

  using DispatchPolicy = MainloopSm100ArrayTmaUmmaWarpSpecializedBlockwiseScaling<
                          Stages,
                          SchedulerPipelineStageCount,
                          AccumulatorPipelineStageCount,
                          ClusterShape>;
  using TileShape = TileShape_;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  CUTE_STATIC_ASSERT_V(evenly_divides(TileShape{}, tile_shape(TiledMma{})),
                       "Static cluster shape used: TileShape should be evenly divided by TiledMma");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  // Define A and B block shapes for reduced size TMA_LOADs
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

  using ElementA = ElementA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using StrideA = cute::remove_cvref_t<decltype(get<0>(StridePairA_{}))>;
  using LayoutSFA = cute::remove_cvref_t<decltype(get<1>(StridePairA_{}))>;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using InternalLayoutSFA = cute::remove_pointer_t<LayoutSFA>;
  using ElementB = ElementB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StrideB = cute::remove_cvref_t<decltype(get<0>(StridePairB_{}))>;
  using LayoutSFB = cute::remove_cvref_t<decltype(get<1>(StridePairB_{}))>;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  using InternalLayoutSFB = cute::remove_pointer_t<LayoutSFB>;

  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();

  static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementB>();

  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
                (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
                "ElementA and ElementB should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

  static constexpr int ScaleGranularityM = size<0,0>(InternalLayoutSFA{});

  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static_assert(size<0>(TileShape{}) % ScaleGranularityM == 0 and ScaleGranularityM <= size<0>(TileShape{}), "Scale Granularity M must divide Tile Shape");

  static constexpr int ScaleGranularityN = size<0,0>(InternalLayoutSFB{});
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;
  static_assert(size<1>(TileShape{}) % ScaleGranularityN == 0 and ScaleGranularityN <= size<1>(TileShape{}), "Scale Granularity N must divide Tile Shape");

  static_assert(size<1, 0>(InternalLayoutSFA{}) == size<1, 0>(InternalLayoutSFB{}), "Vector size K must be equal for SFA and SFB");

  static constexpr int ScaleGranularityK = size<1, 0>(InternalLayoutSFA{});
  static constexpr int ScaleKsPerTile = size<2>(TileShape{}) / ScaleGranularityK;
  static_assert(size<2>(TileShape{}) % ScaleGranularityK == 0 and ScaleGranularityK <= size<2>(TileShape{}), "Scale Granularity K must divide Tile Shape");
  static_assert(ScaleGranularityK % size<2>(typename TiledMma::AtomShape_MNK{}) == 0, "Scale Granularity K must be divisible by MMA_K");

  static constexpr int K_BLOCK_MMAS_PER_SCALE_K = ScaleGranularityK / size<2>(typename TiledMma::AtomShape_MNK{});

  static_assert(size<0>(CtaShape_MNK{}) >= ScaleGranularityM, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<1>(CtaShape_MNK{}) >= ScaleGranularityN, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<2>(CtaShape_MNK{}) >= ScaleGranularityK, "Scale Granularity must be smaller than or equal to the tile shape");

  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      size<0,1>(InternalLayoutSFA{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K,
      size<0,1>(InternalLayoutSFB{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K>;


  using SmemLayoutAtomSFA = decltype(ScaleConfig::smem_atom_layoutSFA(CtaShape_MNK{}));
  using SmemLayoutAtomSFB = decltype(ScaleConfig::smem_atom_layoutSFB(CtaShape_MNK{}));


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

  static constexpr int CopyAlignmentSFA = GmemTiledCopySFA::AtomNumVal::value * sizeof(typename GmemTiledCopySFA::ValType) / sizeof(ElementAccumulator);
  static constexpr int CopyAlignmentSFB = GmemTiledCopySFB::AtomNumVal::value * sizeof(typename GmemTiledCopySFB::ValType) / sizeof(ElementAccumulator);

  static constexpr int AlignmentSFA = CopyAlignmentSFA * (GmemTiledCopySFA::AtomNumVal::value > 1 ?
      (size<0,1>(InternalLayoutSFA{}.stride()) == 1 ? ScaleGranularityM : ScaleGranularityK) : 1);
  static constexpr int AlignmentSFB = CopyAlignmentSFB * (GmemTiledCopySFB::AtomNumVal::value > 1 ?
      (size<0,1>(InternalLayoutSFB{}.stride()) == 1 ? ScaleGranularityN : ScaleGranularityK) : 1);


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
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<InternalStrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  // (MMA_TILE_N,MMA_TILE_K),MMA_N,MMA_K,PIPE)
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<InternalStrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

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

  using BitTypeElementA = uint_bit_t<cute::sizeof_bits_v<ElementA>>;
  using BitTypeElementB = uint_bit_t<cute::sizeof_bits_v<ElementB>>;

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

    struct TensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
    } tensormaps;

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
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Only one thread issues the TMA and updates the barriers in a 2SM MMA, adjust bytes accordingly
  static constexpr uint32_t TmaTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>) +
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>);

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  // Host side kernel arguments
  struct Arguments {
    ArrayElementA const** ptr_A{nullptr};
    StrideA dA{};
    ArrayElementB const** ptr_B{nullptr};
    StrideB dB{};
    ElementAccumulator const** ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    ElementAccumulator const** ptr_SFB{nullptr};
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
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(InternalStrideA{}, int32_t(0)), InternalStrideA{}),
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    using TMA_B = decltype(make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(InternalStrideB{}, int32_t(0)), InternalStrideB{}),
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
    cute::TmaDescriptor* tensormaps;
    ArrayElementA const** ptr_A;
    StrideA dA;
    ArrayElementB const** ptr_B;
    StrideB dB;

    ElementAccumulator const** ptr_SFA;
    LayoutSFA layout_SFA;
    ElementAccumulator const** ptr_SFB;
    LayoutSFB layout_SFB;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
    : cluster_shape_(cluster_shape)
    , block_rank_in_cluster_(block_rank_in_cluster) {
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
    ProblemShape problem_shapes,
    Arguments const& args,
    void* workspace,
    cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_shape = repeat_like(append<4>(typename ProblemShape::UnderlyingProblemShape{}, 1), int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);
    auto init_L = get<3>(init_shape);

    // Tensor pointers will be fixed before the first access
    TmaInternalElementA const* ptr_A_first_batch = nullptr;
    TmaInternalElementB const* ptr_B_first_batch = nullptr;

    InternalStrideA stride_a;
    InternalStrideB stride_b;
    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      stride_a = InternalStrideA{};
      stride_b = InternalStrideB{};
    }
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      stride_a = args.dA;
      stride_b = args.dB;
    }

    // Batches/Groups are managed by using appropriate pointers to input matrices.
    Tensor tensor_a = make_tensor(ptr_A_first_batch, make_layout(make_shape(init_M,init_K,init_L), stride_a));
    Tensor tensor_b = make_tensor(ptr_B_first_batch, make_layout(make_shape(init_N,init_K,init_L), stride_b));

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
      reinterpret_cast<cute::TmaDescriptor*>(workspace),
      reinterpret_cast<ArrayElementA const**>(args.ptr_A),
      args.dA,
      reinterpret_cast<ArrayElementB const**>(args.ptr_B),
      args.dB,
      args.ptr_SFA,
      args.layout_SFA,
      args.ptr_SFB,
      args.layout_SFB
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    constexpr uint32_t NumInputTensors = 2;
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies
    return (NumInputTensors * SizeOfCuTensorMap * sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      [[maybe_unused]] Arguments const& args) {
    static constexpr bool IsF8F6F4 = detail::is_sm100_mma_f8f6f4<TiledMma, ElementA, ElementB>();
    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cute::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cute::sizeof_bits<ElementB>::value;

    bool implementable = true;
    bool implementable_sf = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), InternalStrideA{});
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), InternalStrideB{});
        implementable_sf = implementable_sf && cutlass::detail::check_alignment<CopyAlignmentSFA>(ScaleConfig::tile_atom_to_shape_SFA(problem_shape_MNKL));
        implementable_sf = implementable_sf && cutlass::detail::check_alignment<CopyAlignmentSFB>(ScaleConfig::tile_atom_to_shape_SFB(problem_shape_MNKL));
        if (!implementable_sf) {
          CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for Scale Factors.\n");
        }
      }
    }
    else {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Ignoring check to can implement because host problem shape is not available.\n");
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    implementable = implementable && implementable_sf;
    return implementable;
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));     // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

    return acc_shape;
  }

  template <class FrgEngine, class FrgLayout>
  CUTLASS_DEVICE auto
  slice_accumulator(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, int stage) {
    return accumulators(_,_,_,stage);
  }

  /// Set up the data needed by this collective for load.
  /// Return tuple element contain
  /// gA_mkl - The tiled tma tensor for input A
  /// gB_nkl - The tiled tma tensor for input B
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
      int32_t const sm_count, int32_t const sm_idx,
      [[maybe_unused]] int32_t init_group) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    // Problem Shape and therefore strides that we construct are [M,N,K,L], but since here for the TMA loads
    // we are managing TMA descriptors to change batches, we need to neglect the L mode
    const int32_t mock_L = 1;

    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K,mock_L));
    Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N,K,mock_L));

    // Tile the tensors and defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});     // (BLK_M, BLK_K, m, k, l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N, BLK_K, n, k, l)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);                                       // (MMA, MMA_M, MMA_K, m, k, l)
    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);                                       // (MMA, MMA_N, MMA_K, n, k, l)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});      // (MMA,MMA_M,MMA_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});      // (MMA,MMA_N,MMA_K,PIPE)

    // Define the CTA-in-Cluster Layout and Coord
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

    // Fetch a copy of tensormaps for the CTA from Params
    auto input_tensormaps = tensormaps_init(params, shared_tensormaps, sm_count, sm_idx);

    return cute::make_tuple(
        gA_mkl, gB_nkl,                        // for scheduler
        tAgA_mkl, tBgB_nkl, tAsA, tBsB,        // for input tensor values
        mcast_mask_a, mcast_mask_b,            // multicast masks
        input_tensormaps);                     // for tma descriptor modification (per-CTA tensormap copy)
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_sf_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_tensors,
      int current_group) const {
    return load_sf_update(problem_shape_MNKL, params, shared_tensors, current_group);
  }

  /// Set up the data needed by this collective for load.
  /// Return tuple element contain
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_sf_update(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_tensors,
      int current_group) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    // Problem Shape and therefore strides that we construct are [M,N,K,L], but since here for the TMA loads
    // we are managing TMA descriptors to change batches, we need to neglect the L mode
    const int32_t mock_L = 1;

    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K,mock_L));
    // Tile the tensors and defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});    // (BLK_M, BLK_K, m, k, l)

    auto layout_SFA = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (IsGroupedGemmKernel) {
        return params.layout_SFA[current_group];
      }
      else {
        return params.layout_SFA;
      }
    }();

    auto layout_SFB = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (IsGroupedGemmKernel) {
        return params.layout_SFB[current_group];
      }
      else {
        return params.layout_SFB;
      }
    }();

    Tensor mSFA_mkl = make_tensor(make_gmem_ptr(params.ptr_SFA[current_group]), layout_SFA);                  // (m,k,l)

    Tensor mSFB_nkl = make_tensor(make_gmem_ptr(params.ptr_SFB[current_group]), layout_SFB);                  // (n,k,l)

    Tensor SFA_mkl_ident = make_identity_tensor(shape(layout_SFA));

    Tensor SFB_nkl_ident = make_identity_tensor(shape(layout_SFB));

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

    // 1 thread copies entire set of scalar
    GmemTiledCopySFA scale_copy_a{};
    GmemTiledCopySFB scale_copy_b{};

    ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(threadIdx.x % size(scale_copy_a));
    ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(threadIdx.x % size(scale_copy_b));

    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()),
        SmemLayoutScaleA{});                                                                          // (CTA_M,CTA_K,P)
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()),
        SmemLayoutScaleB{});                                                                          // (CTA_M,CTA_K,P)

    Tensor tSFAgSFA_mkl = thr_scale_copy_a.partition_S(gSFA_mkl);                        // (CPY, BLK_M, BLK_K, m, k, l)
    Tensor tSFAIdentSFA_mkl = thr_scale_copy_a.partition_S(identSFA_mkl);                // (CPY, BLK_M, BLK_K, m, k, l)

    Tensor tSFAsSFA = thr_scale_copy_a.partition_D(sSFA);

    Tensor tSFBgSFB_nkl = thr_scale_copy_b.partition_S(gSFB_nkl);                        // (CPY, BLK_N, BLK_K, m, k, l)
    Tensor tSFBIdentSFB_nkl = thr_scale_copy_b.partition_S(identSFB_nkl);                // (CPY, BLK_N, BLK_K, m, k, l)
    Tensor tSFBsSFB = thr_scale_copy_b.partition_D(sSFB);

    static_assert(rank(decltype(tSFAgSFA_mkl){}) == 6);
    static_assert(rank(decltype(tSFBgSFB_nkl){}) == 6);

    return cute::make_tuple(gA_mkl,
                            tSFAgSFA_mkl, tSFBgSFB_nkl,
                            tSFAsSFA, tSFBsSFB,
                            tSFAIdentSFA_mkl, tSFBIdentSFB_nkl,
                            layout_SFA, layout_SFB);
  }

  /// Setup data needed for transform
  CUTLASS_DEVICE auto
  accum_init(
      TensorStorage& shared_tensors) const {
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()),
        SmemLayoutScaleA{});                                                                          // (CTA_M,CTA_K,P)
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()),
        SmemLayoutScaleB{});                                                                          // (CTA_M,CTA_K,P)

    return cute::make_tuple(sSFA, sSFB);
  }

  /// Set up the data needed by this collective for mma compute.
  template <class FrgEngine, class FrgLayout>
  CUTLASS_DEVICE auto
  mma_init(
      Params const& params,
      [[maybe_unused]] cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TensorStorage& shared_tensors,
      [[maybe_unused]] uint32_t const tmem_nonaccum_offset) const {
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
      tiled_mma.idesc_.a_format_ = uint8_t(params.runtime_data_type_a) & 0b111;
      tiled_mma.idesc_.b_format_ = uint8_t(params.runtime_data_type_b) & 0b111;
    }

    return cute::make_tuple(tiled_mma, tCrA, tCrB);
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
    MainloopABPipeline mainloop_ab_pipeline,
    MainloopABPipelineState mainloop_ab_pipe_producer_state,
    cute::tuple<GTensorA, GTensorB,
                GTensorPartitionedA, GTensorPartitionedB,
                STensorA, STensorB,
                uint16_t, uint16_t,
                cute::tuple<TensorMapA, TensorMapB>> const& load_inputs,
    TileCoordMNKL const& cta_coord_mnkl,
    KTileIterator k_tile_iter, int k_tile_count,
    bool did_batch_change) {

    auto [unused_gA, unused_gB,
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b,
          input_tensormaps] = load_inputs;

    // Check to see if tensormaps have been replaced in gmem
    if (did_batch_change) {
      tensormaps_fence_acquire(input_tensormaps);
    }

    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    auto barrier_token = mainloop_ab_pipeline.producer_try_acquire(mainloop_ab_pipe_producer_state);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_ab_pipeline.producer_acquire(mainloop_ab_pipe_producer_state, barrier_token);

      using BarrierType = typename MainloopABPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = mainloop_ab_pipeline.producer_get_barrier(mainloop_ab_pipe_producer_state);

      int write_stage = mainloop_ab_pipe_producer_state.index();
      ++mainloop_ab_pipe_producer_state;
      barrier_token = mainloop_ab_pipeline.producer_try_acquire(mainloop_ab_pipe_producer_state);

      if (cute::elect_one_sync()) {
        copy(observed_tma_load_a_->with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
        copy(observed_tma_load_b_->with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
      }
      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_ab_pipe_producer_state, k_tile_iter);
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
  CUTLASS_DEVICE void
  load_ab_tail(MainloopABPipeline mainloop_ab_pipeline, MainloopABPipelineState mainloop_ab_pipe_producer_state) {
    // Issue the epilogue waits
    // This helps avoid early exit of ctas in Cluster
    // Waits for all stages to either be released (all
    // Consumer UNLOCKs), or if the stage was never used
    // then would just be acquired since the phase was
    // still inverted from make_producer_start_state
    mainloop_ab_pipeline.producer_tail(mainloop_ab_pipe_producer_state);
  }

  /// Perform a collective-scoped transform
  /// Producer Perspective
  template <
    class UnusedGTensorA,
    class GTensorPartitionedSFA, class GTensorPartitionedSFB,
    class STensorSFA, class STensorSFB,
    class IdentPartitionedSFA, class IdentPartitionedSFB,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load_sf(
    MainloopSFPipeline mainloop_sf_pipeline,
    MainloopSFPipelineState mainloop_sf_pipe_producer_state,
    cute::tuple<UnusedGTensorA,
                GTensorPartitionedSFA, GTensorPartitionedSFB,
                STensorSFA, STensorSFB,
                IdentPartitionedSFA,
                IdentPartitionedSFB,
                InternalLayoutSFA,
                InternalLayoutSFB> const& mainloop_sf_inputs,
    TileCoordMNKL const& cta_coord_mnkl,
    KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused, tSFAgSFA_mkl, tSFBgSFB_nkl,
          tSFAsSFA, tSFBsSFB,
          tSFAIdentSFA_mkl, tSFBIdentSFB_nkl,
          layout_SFA, layout_SFB] = mainloop_sf_inputs;

    // slice out the work coord from partitioned tensors
    GmemTiledCopySFA scale_copy_a{};
    GmemTiledCopySFB scale_copy_b{};

    Tensor tSFAgSFA = tSFAgSFA_mkl(_, _, _, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    Tensor tSFBgSFB = tSFBgSFB_nkl(_, _, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    Tensor thr_tile_SFA_k = tSFAIdentSFA_mkl(_0{}, _, _, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
    Tensor thr_tile_pSFA = make_tensor<bool>(shape(filter_zeros(thr_tile_SFA_k(_,_,_0{}), tSFAgSFA(_0{},_,_,_0{}).stride())));
    Tensor thr_tile_SFB_k = tSFBIdentSFB_nkl(_0{}, _, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    Tensor thr_tile_pSFB = make_tensor<bool>(shape(filter_zeros(thr_tile_SFB_k(_,_,_0{}), tSFBgSFB(_0{},_,_,_0{}).stride())));

    // Issue the loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK pipe_producer_state for _writing_
      mainloop_sf_pipeline.producer_acquire(mainloop_sf_pipe_producer_state);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(thr_tile_pSFA); ++i) {
        Tensor thr_tile_SFA = filter_zeros(thr_tile_SFA_k(_,_,*k_tile_iter), tSFAgSFA(_0{},_,_,_0{}).stride());
        thr_tile_pSFA(i) = elem_less(thr_tile_SFA(i), shape(filter_zeros(layout_SFA))) && threadIdx.x % 32 < size(scale_copy_a);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(thr_tile_pSFB); ++i) {
        Tensor thr_tile_SFB = filter_zeros(thr_tile_SFB_k(_,_,*k_tile_iter), tSFBgSFB(_0{},_,_,_0{}).stride());
        thr_tile_pSFB(i) = elem_less(thr_tile_SFB(i), shape(filter_zeros(layout_SFB))) && threadIdx.x % 32 < size(scale_copy_b);
      }

      copy_if(scale_copy_a, thr_tile_pSFA, filter_zeros(tSFAgSFA(_,_,_,*k_tile_iter)), filter_zeros(tSFAsSFA(_,_,_,mainloop_sf_pipe_producer_state.index())));
      copy_if(scale_copy_b, thr_tile_pSFB, filter_zeros(tSFBgSFB(_,_,_,*k_tile_iter)), filter_zeros(tSFBsSFB(_,_,_,mainloop_sf_pipe_producer_state.index())));
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
    class FrgEngine, class FrgLayout,
    class FragmentA, class FragmentB,
    class CtaTileCoord
  >
  CUTLASS_DEVICE auto
  mma(cute::tuple<MainloopABPipeline,
                  AccumulatorPipeline> pipelines,
      cute::tuple<MainloopABPipelineState,
                  AccumulatorPipelineState> pipeline_states,
      cute::Tensor<FrgEngine, FrgLayout>& accumulators,
      cute::tuple<TiledMma, FragmentA, FragmentB> const& mma_inputs,
      CtaTileCoord cta_tile_coord,
      int k_tile_count) {
    static_assert(is_tmem<FrgEngine>::value, "Accumulator must be tmem resident.");
    static_assert(rank(FrgLayout{}) == 4, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N, P)");
    auto [tiled_mma, tCrA, tCrB] = mma_inputs;

    auto [mainloop_pipeline, accumulator_pipeline] = pipelines;
    auto [mainloop_pipe_consumer_state, accumulator_pipe_producer_state] = pipeline_states;

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

        auto acc = slice_accumulator(accumulators, accumulator_pipe_producer_state.index());
        static_assert(is_tmem<remove_cvref_t<decltype(acc)>>::value, "Accumulator must be tmem resident.");
        static_assert(rank(remove_cvref_t<decltype(acc)>{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

        // for each set of scale_k_iter we zero the accumulator
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
    class FrgEngine,
    class FrgLayout,
    class TensorsSFA,
    class TensorsSFB,
    class CtaTileCoord,
    class CopyOpT2R,
    class EpilogueTile
  >
  CUTLASS_DEVICE auto
  accum(
      cute::tuple<AccumulatorPipeline, MainloopSFPipeline> pipelines,
      cute::tuple<AccumulatorPipelineState, MainloopSFPipelineState> consumer_states,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      cute::tuple<TensorsSFA, TensorsSFB> const& transform_inputs,
      CtaTileCoord cta_tile_coord,
      CopyOpT2R,
      EpilogueTile,
      int k_tile_count) {

    static_assert(size<0>(EpilogueTile{}) <= size<0>(CtaShape_MNK{}), "Restrict epilogue tile to be smaller than or equal to CTA Tile");
    static_assert(size<1>(EpilogueTile{}) <= size<1>(CtaShape_MNK{}), "Restrict epilogue tile to be smaller than or equal to CTA Tile");


    //
    // PIPELINED Transform
    //

    Tensor acc = slice_accumulator(accumulators, _0{});
    Tensor tAcc = acc(make_coord(_,_),_0{},_0{});
    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                          // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    auto [sSFA_, sSFB_] = transform_inputs;

    // Append N with a stride of 0 to SFA
    Tensor sSFA = make_tensor(sSFA_.data(), make_layout(
      make_shape(get<0>(sSFA_.shape()), get<1>(CtaShape_MNK{}), get<1>(sSFA_.shape()), get<2>(sSFA_.shape())),
      make_stride(get<0>(sSFA_.stride()), _0{}, get<1>(sSFA_.stride()), get<2>(sSFA_.stride()))
    ));

    CUTE_STATIC_ASSERT_V(size<0>(sSFA) == size<0>(tAcc));
    CUTE_STATIC_ASSERT_V(size<1>(sSFA) == size<1>(tAcc));

    Tensor sSFA_epi = flat_divide(sSFA, EpilogueTile{});

    // Append M with a stride of 0 to SFB
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

        Tensor acc = slice_accumulator(accumulators, accumulator_pipe_state.index());
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

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //

  CUTLASS_DEVICE auto
  tensormaps_init(
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
  tensormaps_replace_global_address(
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
  tensormaps_replace_global_tensor_properties(
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

    TmaInternalElementA const* ptr_A = nullptr;
    Tensor tensor_a = make_tensor(ptr_A, make_shape(M,K,Int<1>{}), mainloop_params.dA[next_group]);

    TmaInternalElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(ptr_B, make_shape(N,K,Int<1>{}), mainloop_params.dB[next_group]);

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
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps,
      ProblemShape problem_shape,
      int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        auto problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(next_batch), 1);
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_MNKL);
      }
    }
    // Ensure warp is converged before issuing tensormap fence release
    __syncwarp();
    // Entire warp must do this (ie its aligned)
    tensormaps_cp_fence_release(shared_tensormaps, input_tensormaps);
  }

  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class TensorMapA, class TensorMapB>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::tuple<TensorMapA, TensorMapB> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
  }

private:

  typename Params::TMA_A const* observed_tma_load_a_{nullptr};
  typename Params::TMA_B const* observed_tma_load_b_{nullptr};

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
