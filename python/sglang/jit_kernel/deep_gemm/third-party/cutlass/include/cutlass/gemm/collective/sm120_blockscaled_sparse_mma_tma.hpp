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
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/collective/builders/sm1xx_sparse_config.inl"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// CollectiveMma for A/B with different or same stages based on asymmetric DMA.

template <
  int StagesA,
  int StagesB,
  int StagesE,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class TileShape_,
  class ElementPairA_,
  class LayoutPairsA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomsA_,
  class SmemCopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomsB_,
  class SmemCopyAtomsB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm120TmaWarpSpecializedSparseBlockScaled<StagesA, StagesB, StagesE, SchedulerPipelineStageCount, ClusterShape>,
    TileShape_,
    ElementPairA_,
    LayoutPairsA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomsA_,
    SmemCopyAtomsA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomsB_,
    SmemCopyAtomsB_,
    TransformB_> {
  //
  // Type Aliases
  //
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using LayoutPairsA = LayoutPairsA_;
  using StridePairB = StridePairB_;
  using SmemCopyAtomsA = SmemCopyAtomsA_;
  using SmemCopyAtomsB = SmemCopyAtomsB_;

  using TiledMma = TiledMma_;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;
  using DispatchPolicy = MainloopSm120TmaWarpSpecializedSparseBlockScaled<StagesA, StagesB, StagesE, SchedulerPipelineStageCount, ClusterShape>;
  using TileShape = TileShape_;
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementAMmaRaw = typename ElementAMma::raw_type;
  using LayoutA =  remove_cvref_t<decltype(get<0>(LayoutPairsA{}))>;
  using LayoutE =  remove_cvref_t<decltype(get<1>(LayoutPairsA{}))>;
  using StrideA =  remove_cvref_t<decltype(get<3>(LayoutPairsA{}))>;
  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB = remove_cvref_t<decltype(get<0>(StridePairB{}))>;
  using ElementBMma = typename TiledMma::ValTypeB;
  using ElementEMma = typename TiledMma::ValTypeE;
  using ElementE = typename ElementEMma::raw_type;
  using RegisterE = typename remove_extent<typename TiledMma::MMA_Op::ERegisters>::type;
  using ArrayElementA = ElementA;
  using ArrayElementB = ElementB;

  // SFA, SFB and metadata config
  static_assert(cute::is_same_v<remove_cvref_t<decltype(get<1>(ElementPairA{}))>,
                                remove_cvref_t<decltype(get<1>(ElementPairB{}))>>,
                                "SFA and SFB data types should be the same");
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using LayoutSFA = remove_cvref_t<decltype(get<2>(LayoutPairsA{}))>;
  using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;
  static constexpr int SFVecSize = TiledMma::Traits::SFVecSize;
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA_{}))>;
  using GmemTiledCopyB = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB_{}))>;;
  using SmemCopyAtomA = remove_cvref_t<decltype(get<0>(SmemCopyAtomsA{}))>;
  using SmemCopyAtomE = remove_cvref_t<decltype(get<1>(SmemCopyAtomsA{}))>;
  using SmemCopyAtomB = remove_cvref_t<decltype(get<0>(SmemCopyAtomsB{}))>;
  using SmemLayoutAtomA = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsA_{}))>;
  using SmemLayoutAtomB = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsB_{}))>;
  using SmemLayoutAtomSFA = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsA_{}))>;
  using SmemLayoutAtomSFB = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsB_{}))>;
  using SmemCopyAtomSFA = remove_cvref_t<decltype(get<2>(SmemCopyAtomsA{}))>;
  using SmemCopyAtomSFB = remove_cvref_t<decltype(get<1>(SmemCopyAtomsB{}))>;
  using GmemTiledCopySFA = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA_{}))>;
  using GmemTiledCopySFB = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB_{}))>;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using GmemTiledCopyE = GmemTiledCopyA;

  // Asymmetric buffering
  // Tensor A/B could have different buffering, with TILEK, and STAGEs.
  //    It let AsymmetricKRatio equals TILEK_A / TILEK_B, to make sure A/B's
  //    pipeline keep same steps when produce / consume data.
  // Currently, AsymmetricKRatio = {1, 2} is the only support.
  static constexpr int AsymmetricKRatio = DispatchPolicy::StagesA != DispatchPolicy::StagesB ? 2 : 1;

  // Construct TileShape for SFB load from GMEM to SMEM.
  // It is required to keep consistency with BlockScaled granularity defined in Sm1xxBlkScaledConfig.
  // So that TileShape for scaling factor needs to be defined as a multiple of Blk_MN.
  using Blk_MN      = typename Sm1xxBlkScaledConfig::Blk_MN;
  using TileShapeSF = decltype(make_shape(ceil_div(size<0>(CtaShape_MNK{}), Blk_MN{}) * Blk_MN{},
                                           ceil_div(size<1>(CtaShape_MNK{}), Blk_MN{}) * Blk_MN{},
                                           shape<2>(CtaShape_MNK{})));
  using TileShapeB = decltype(make_shape(size<0>(TileShape{}),
                                         size<1>(TileShape{}),
                                         ceil_div(size<2>(TileShape{}), Int<AsymmetricKRatio>{})));

  static constexpr int ThreadCount = size(TiledMma{});
  static constexpr int IsCtaN64 = shape<1>(CtaShape_MNK{}) == 64;
  static constexpr int TensorAMmaSparsity = ElementAMma::sparsity;
  static constexpr int TensorEMmaSparsity = ElementEMma::sparsity;

  // Use two MainloopPipeline for A and B separately.
  using MainloopPipelineMK = cutlass::PipelineTmaAsync<DispatchPolicy::StagesA>;
  using MainloopPipelineNK = cutlass::PipelineTmaAsync<DispatchPolicy::StagesB>;
  using PipelineStateMK  = typename cutlass::PipelineState<DispatchPolicy::StagesA>;
  using PipelineStateNK  = typename cutlass::PipelineState<DispatchPolicy::StagesB>;
  using PipelineParams = typename MainloopPipelineMK::Params;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(not cute::is_void_v<SmemCopyAtomA>,
    "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");
  static_assert(not cute::is_void_v<SmemCopyAtomB>,
    "SM120 mainloop must specify a copy atom for B operand smem->rmem reads.");

  // Tile along modes in a way that maximizes the TMA box size.
  // Note: SmemA, SmemSFA and SmemSFB are with same stages, while SmemB is with another stage number.
  // SmemSFB is not with same stages as SmemB, as it will not design 1.5x stages if Smem not enough.
  // These different stages setting could maximize capacity of latency hide, while keep data in SMEM.
  // Metadata may kept in SMEM, or in GMEM/L2, if under SMEM limitation.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::StagesA>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShapeB{}), shape<2>(TileShapeB{}), Int<DispatchPolicy::StagesB>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::StagesA>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::StagesA>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

  static_assert(DispatchPolicy::StagesA >= 2, "Specialization requires StagesA set to value 2 or more.");
  static_assert(DispatchPolicy::StagesB >= 2, "Specialization requires StagesB set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operands from rmem for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>,
                  "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>,
                  "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  static constexpr bool IsF8F6F4 = detail::is_sm100_sparse_f8f6f4<TiledMma, ElementA, ElementB>();

  // Is E kept in SMEM or GMEM
  static constexpr bool UseSmemE = DispatchPolicy::StagesE != 0;

  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  using TmaInternalElementA = cute::conditional_t<not IsF8F6F4,
                                                  ElementA,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m3_t>,
                                                cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e3m2_t>,
                                                cutlass::detail::float_e3m2_unpacksmem_t,
                                                uint_bit_t<sizeof_bits_v<ElementA>>>>>>;
  using TmaSourceElementA = cute::conditional_t<IsF8F6F4, ElementA, uint8_t>;

  using TmaInternalElementB = cute::conditional_t<not IsF8F6F4,
                                                  ElementB,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m3_t>,
                                                cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e3m2_t>,
                                                cutlass::detail::float_e3m2_unpacksmem_t,
                                                uint_bit_t<sizeof_bits_v<ElementB>>>>>>;

  // Set shared memory layout
  using SmemAllocTypeA = cute::conditional_t<IsF8F6F4, sparse_elem<TensorAMmaSparsity, uint8_t>, ElementAMma>;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4, uint8_t, ElementBMma>;

  static constexpr bool is_A_mn_major = cute::is_same_v<decltype(stride<0>(LayoutA{})), Int<TensorAMmaSparsity>>;
  using SparseConfig = cutlass::Sm1xxGemmSparseConfig<ElementAMma,
                                                      cute::conditional_t<is_A_mn_major, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>,
                                                      ElementEMma>;
  using SmemLayoutAtomE_ = typename SparseConfig::TensorEAtom;
  using SmemLayoutAtomE  = ComposedLayout<Swizzle<0,4,3>,
                                          smem_sparse_ptr_flag_bits<TensorEMmaSparsity, sizeof_bits_v<ElementE>>,
                                          SmemLayoutAtomE_>;
  using SmemLayoutE = decltype(tile_to_shape(
                  SmemLayoutAtomE{},
                  make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::StagesE>{}),
                  conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  static constexpr int SmemSizeE  = UseSmemE ? cosize(SmemLayoutE{}) : 0;
  static constexpr int StageSizeE = UseSmemE ? cosize(take<0,2>(SmemLayoutE{})) : 0;
  // Check if metetata fetching needs predication
  using TensorEAtomM = typename SparseConfig::TensorEAtomM;
  using TensorEAtomK = typename SparseConfig::TensorEAtomK;
  static constexpr bool IsELoadPred = not (TensorEAtomM{} == size<0>(TileShape{}) && TensorEAtomK{} == size<2>(TileShape{}));

  static_assert(rank(SmemLayoutAtomE{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomE{})) == 0, "SmemLayoutAtomE must evenly divide tile shape.");

  // Set the bytes transferred in this TMA transaction
  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementAMma>) +
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(StageSizeE * cute::sizeof_bits_v<ElementEMma>));
  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>));
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128> {
      alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
      cute::ArrayEngine<ElementEMma, Int<SmemSizeE>{}> smem_E;
    } tensors;

    using PipelineStorageMK = typename MainloopPipelineMK::SharedStorage;
    using PipelineStorageNK = typename MainloopPipelineNK::SharedStorage;
    alignas(16) PipelineStorageMK pipeline_storage_mk;
    alignas(16) PipelineStorageNK pipeline_storage_nk;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorageMK = typename SharedStorage::PipelineStorageMK;
  using PipelineStorageNK = typename SharedStorage::PipelineStorageNK;

  struct Arguments {
    ElementA const* ptr_A{nullptr};
    LayoutA layout_a{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementE const* ptr_E{nullptr};
    LayoutE layout_e{};
    ElementSF const* ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    ElementSF const* ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy<TmaInternalElementA>(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<sparse_elem<TensorAMmaSparsity,TmaSourceElementA>>(nullptr), LayoutA{}),
        SmemLayoutA{}(_,_,0),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,0),
        make_shape(shape<1>(TileShapeB{}), shape<2>(TileShapeB{})),
        _1{}));
    using TMA_E = decltype(make_tma_copy<ElementE>(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<ElementEMma>(nullptr), LayoutE{}),
        SmemLayoutE{}(_,_,0),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));
    using TMA_SFA = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySFA{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFA{}),
        SmemLayoutSFA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));
    using TMA_SFB = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySFB{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFB{}),
        SmemLayoutSFB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShapeSF{}), shape<2>(TileShapeSF{})),
        _1{}));
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_E tma_load_e;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;
    LayoutA layout_a;
    LayoutE layout_e;
    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    ElementE const* ptr_E{nullptr};
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<sparse_elem<TensorAMmaSparsity, TmaSourceElementA>>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);
    auto ptr_E = recast_ptr<ElementEMma>(args.ptr_E);

    Tensor tensor_a = make_tensor(ptr_A, args.layout_a);
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));
    Tensor tensor_e = make_tensor(ptr_E, args.layout_e);
    Tensor tensor_sfa = make_tensor(args.ptr_SFA, args.layout_SFA);
    Tensor tensor_sfb = make_tensor(args.ptr_SFB, args.layout_SFB);

    typename Params::TMA_A tma_load_a = make_tma_copy<TmaInternalElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{});
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShapeB{})),
        _1{});
    typename Params::TMA_E tma_load_e = make_tma_copy<ElementE>(
        GmemTiledCopyE{},
        tensor_e,
        SmemLayoutE{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{});
    typename Params::TMA_SFA tma_load_sfa = make_tma_copy<uint16_t>(
        GmemTiledCopySFA{},
        tensor_sfa,
        SmemLayoutSFA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{});
    typename Params::TMA_SFB tma_load_sfb = make_tma_copy<uint16_t>(
        GmemTiledCopySFB{},
        tensor_sfb,
        SmemLayoutSFB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShapeSF{}), shape<2>(TileShapeSF{})),
        _1{});
    return {
      tma_load_a,
      tma_load_b,
      tma_load_e,
      tma_load_sfa,
      tma_load_sfb,
      args.layout_a,
      args.layout_e,
      args.layout_SFA,
      args.layout_SFB,
      args.ptr_E
    };
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::upcast<2>(make_layout(make_shape(M, K, L), StrideA{})));
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_sfa.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_sfb.get_tma_descriptor());
    if constexpr (UseSmemE) {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_e.get_tma_descriptor());
    }
  }

  /// Create fragment for metadata. The function is referred from thrfrg_A(...)
  template <class Tensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_E(Tensor&& tensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(tensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutE_TV = typename Atom::Traits::ELayout;

    auto t_tile = make_tile(get<0>(TiledPerm{}),
                            get<2>(TiledPerm{}));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto t_tensor = logical_divide(tensor, t_tile);

    // Tile the tensor for the Atom
    auto e_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto e_tensor = zipped_divide(t_tensor, e_tile);                                   // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = e_tensor.compose(AtomLayoutE_TV{},_);                               // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);                  // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    // Fragment layout
    return thr_tensor;
  }

  /// get metadata TV
  template<class TiledMma>
  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutE_TV(TiledMma& mma)
  {
      // (M,K) -> (M,K)
      auto tile_shape_mnk = tile_shape(mma);
      auto ref_E = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

      // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
      auto etile = make_tile(_,
                            make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                  make_stride(               Int<1>{} ,                Int<0>{} )),
                                      _));

      // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
      auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
      // (thr_idx,val) -> (M,K)
      return thrfrg_E(ref_E, mma).compose(etile, _).compose(thridx_2_thrid, _);
  }

  /// Partitioning for metadata.
  template <class Tensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_E(Tensor&& tensor, ThrMma& thread_mma) {
    auto thr_tensor = make_tensor(static_cast<Tensor&&>(tensor).data(), thrfrg_E(tensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;

    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ThrMma::Atom::Traits::ValTypeE>(partition.layout());
  }

  // Temporary adhoc partitioning for scaling factors.
  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
  {
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(permutation_mnk),
                            get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfatensor, t_tile);                                                 // (PermM,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                                   // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);                             // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);                  // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
  {
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<1>(permutation_mnk),
                            get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfbtensor, t_tile);                                                 // (PermN,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                                   // ((AtomN,AtomK),(RestN,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{},_);                             // ((ThrV,FrgV),(RestN,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);                  // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
    return thr_tensor;
  }

  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma)
  {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA =  thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFA);
  }

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma)
  {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB =  thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFB);
  }

  template<class TiledMma>
  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutSFA_TV(TiledMma& mma)
  {
    // (M,K) -> (M,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto atile = make_tile(_,
                          make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                make_stride(               Int<1>{} ,                Int<0>{} )),
                                    _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
  }

  template<class TiledMma>
  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutSFB_TV(TiledMma& mma)
  {
    // (N,K) -> (N,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto btile = make_tile(_,
                          make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                make_stride(               Int<0>{} ,                Int<1>{} )),
                                    _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(mainloop_params.layout_a.shape());             // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)
    Tensor mE_mkl = mainloop_params.tma_load_e.get_tma_tensor(mainloop_params.layout_e.shape());             // (m,k,l)
    Tensor mSFA_mkl = mainloop_params.tma_load_sfa.get_tma_tensor(shape(mainloop_params.layout_SFA));
    auto mSFB_nkl = [=](){
      if constexpr (IsCtaN64) {
        Tensor mSFB_tmp = mainloop_params.tma_load_sfb.get_tma_tensor(shape(mainloop_params.layout_SFB));
        auto x = stride<0,1>(mSFB_tmp);
        auto y = ceil_div(shape<0,1>(mSFB_tmp), _2{});
        auto  new_shape =  make_shape (make_shape( shape<0,0>(mSFB_tmp),
                                       make_shape( make_shape(_2{}),   y)),  shape<1>(mSFB_tmp), shape<2>(mSFB_tmp));
        auto new_stride = make_stride(make_stride(stride<0,0>(mSFB_tmp),
                                      make_stride(make_stride(_0{}),   x)), stride<1>(mSFB_tmp), stride<2>(mSFB_tmp));
        return make_tensor(mSFB_tmp.data(), make_layout(new_shape, new_stride));
      }
      else {
        return mainloop_params.tma_load_sfb.get_tma_tensor(shape(mainloop_params.layout_SFB));
      }
    }();

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // ( BLK_M, BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShapeB{}, make_coord(_,_,_), Step< X,_1,_1>{});       // ( BLK_N, BLK_K,n,k,l)
    Tensor gE_mkl = local_tile(mE_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // ( BLK_N, BLK_K,n,k,l)
    Tensor gSFA_mkl = local_tile(mSFA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});    // (TILE_M,TILE_K,m,k,l)
    Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShapeSF{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (TILE_N,TILE_K,n,k,l)
    return cute::make_tuple(gA_mkl, gB_nkl, gE_mkl, gSFA_mkl, gSFB_nkl);
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  template<class MainloopPipeline, class PipelineState>
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  // Issues loads for A/E/SF only (used when DMA warp is split).
  template <
    class TensorA, class TensorB, class TensorE,
    class TensorSFA, class TensorSFB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load_MK(
      Params const& params,
      MainloopPipelineMK pipeline,
      PipelineStateMK smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorE, TensorSFA, TensorSFB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});         // (BLK_M,BLK_K,PIPE)
    Tensor sE = make_tensor(make_smem_ptr(shared_tensors.smem_E.begin()), SmemLayoutE{});         // (BLK_M,BLK_K,PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});   // (BLK_M,BLK_K,PIPE)
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});   // (BLK_N,BLK_K,PIPE)

    //
    // Prepare the TMA loads for A and E
    //

    Tensor gA_mkl = get<0>(load_inputs);                                                             // (BLK_M,BLK_K,k)
    Tensor gE_mkl = get<2>(load_inputs);                                                             // (BLK_M,BLK_K,k)
    Tensor gSFA_mkl = get<3>(load_inputs);                                                           // (BLK_M,BLK_K,k)
    Tensor gSFB_nkl = get<4>(load_inputs);                                                           // (BLK_N,BLK_K,k)

    auto block_tma_a = params.tma_load_a.get_slice(0);
    auto block_tma_e = params.tma_load_e.get_slice(0);
    auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
    auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                       // (BLK_M,BLK_K,k)
    Tensor gE = gE_mkl(_,_,m_coord,_,l_coord);                                                       // (BLK_M,BLK_K,k)
    Tensor gSFA = gSFA_mkl(_,_,m_coord,_,l_coord);                                                   // (BLK_M,BLK_K,k)
    Tensor gSFB = gSFB_nkl(_,_,n_coord,_,l_coord);                                                   // (BLK_N,BLK_K,k)

    // Partition source and destination tensors for tma copies
    Tensor tAgA = block_tma_a.partition_S(gA);                                                // (TMA,TMA_M,TMA_K,   k)
    Tensor tAsA = block_tma_a.partition_D(sA);                                                // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tEgE = block_tma_e.partition_S(gE);                                                // (TMA,TMA_M,TMA_K,   k)
    Tensor tEsE = block_tma_e.partition_D(sE);                                                // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);                                          // (TMA,TMA_M,TMA_K,   k)
    Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);                                          // (TMA,TMA_M,TMA_K,PIPE)
    Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);                                          // (TMA,TMA_N,TMA_K,   k)
    Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);                                          // (TMA,TMA_N,TMA_K,PIPE)

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      //
      // Copy gmem to smem for *k_tile_iter
      //
      using BarrierType = typename MainloopPipelineMK::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      if (cute::elect_one_sync()) {
        copy(params.tma_load_a.with(*tma_barrier), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(params.tma_load_sfa.with(*tma_barrier), tAgSFA(_,_,_,*k_tile_iter), tAsSFA(_,_,_,write_stage));
        copy(params.tma_load_sfb.with(*tma_barrier), tBgSFB(_,_,_,*k_tile_iter), tBsSFB(_,_,_,write_stage));
        if constexpr (UseSmemE) {
          copy(params.tma_load_e.with(*tma_barrier), tEgE(_,_,_,*k_tile_iter), tEsE(_,_,_,write_stage));
        }
      }

      if constexpr (!UseSmemE) {
        // Prefetch 1 stage of E data to L2 in advance
        auto blk_coord_mkl = make_coord(get<0>(blk_coord), *k_tile_iter, get<3>(blk_coord));         // (BLK_M,BLK_K,L)
        prefetch(make_local_E(params, blk_coord_mkl));
      }

      // Advance smem_pipe_write
      ++k_tile_iter;
      ++smem_pipe_write;
    }
  }

  // Issues loads for B/SF only (used when DMA warp is split).
  template <
    class TensorA, class TensorB, class TensorE,
    class TensorSFA, class TensorSFB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load_NK(
      Params const& params,
      MainloopPipelineNK pipeline,
      PipelineStateNK smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorE, TensorSFA, TensorSFB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {

    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)

    //
    // Prepare the TMA loads for B
    //

    Tensor gB_nkl = get<1>(load_inputs);
    auto block_tma_b = params.tma_load_b.get_slice(0);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gB =   gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

    // Partition source and destination tensors for tma copies
    Tensor tBgB = block_tma_b.partition_S(gB);                                                // (TMA,TMA_N,TMA_K,   k)
    Tensor tBsB = block_tma_b.partition_D(sB);                                                // (TMA,TMA_N,TMA_K,PIPE)

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      //
      // Copy gmem to smem for *k_tile_iter
      //
      using BarrierType = typename MainloopPipelineNK::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      if (cute::elect_one_sync()) {
        copy(params.tma_load_b.with(*tma_barrier), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
      }
      // Advance smem_pipe_write
      ++k_tile_iter;
      ++smem_pipe_write;
    }
  }

  // Local tile E from global memory.
  template<class BlockCoord>
  CUTLASS_DEVICE auto
  make_local_E(Params const& mainloop_params,
               BlockCoord const& blk_coord) {
    // E layout
    auto layoutE = mainloop_params.layout_e;
    // E data pointer as sparse datatype
    auto ptr_E = recast_ptr<ElementEMma>(mainloop_params.ptr_E);

    // Global gmem E
    Tensor gE = make_tensor(make_gmem_ptr(ptr_E), layoutE);                                      // (BLK_M,BLK_K,BLK_L)
    // Local tile E
    return local_tile(gE, select<0,2>(TileShape{}), blk_coord);                                        // (BLK_M,BLK_K)
  }

  // Load E from global memory to registers.
  template<bool IsF8F6F4, class BlockCoord, class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_E(Params const& mainloop_params,
         BlockCoord const& blk_coord,
         ProblemShape_MNKL const& problem_shape_MNKL,
         int thread_idx) {
    // Workload
    auto [M, N, K, L] = problem_shape_MNKL;
    auto [m_coord, k_coord, l_coord] = blk_coord;
    auto Shape_MK = cute::make_tuple(M, K);

    // Tiled mma and thread mma
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    // Tile shape
    auto tile_shape_mnk = tile_shape(tiled_mma);
    // Re-sue copy atom E from SmemCopyAtomE
    using GmemCopyAtomeE = SmemCopyAtomE;
    // Gmem tile copy
    auto gmem_tiled_copy_E = make_tiled_copy_impl(GmemCopyAtomeE{},
                                                  get_layoutE_TV(tiled_mma),
                                                  make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    // Gmem thread copy
    auto gmem_thr_copy_E = gmem_tiled_copy_E.get_thread_slice(thread_idx);
    // Gmem local E
    auto gE_mkl = make_local_E(mainloop_params, blk_coord);
    // Tiled gmem E
    Tensor tCgE = gmem_thr_copy_E.partition_S(gE_mkl);                                             // (CPY,CPY_M,CPY_K)
    // Tiled register E and copy view
    Tensor tCrE = partition_fragment_E(gE_mkl, thread_mma);                                        // (MMA,MMA_M,MMA_K)
    Tensor tCrE_copy_view = gmem_thr_copy_E.retile_D(tCrE);                                        // (CPY,CPY_M,CPY_K)

    if constexpr (IsF8F6F4) {
      auto get_copy_atom_and_common_vec = [&]() {
        using ValType = typename decltype(tCrE)::value_type;
        // Get maximum copy vector size (logically)
        auto common_layout = max_common_layout(tCgE, tCrE);
        auto vec_elem = cute::min(size(common_layout), Int<128 / sizeof_bits_v<ValType>>{});
        auto common_vec = composition(common_layout, vec_elem);
        // Compose a Copy_Atom
        using VecType = uint_bit_t<vec_elem * sizeof_bits_v<ValType>>;
        using cpy = Copy_Atom<UniversalCopy<VecType>, ValType>;
        return cute::make_tuple(cpy{}, common_vec);
      };

      // Copy depends on whether predication is needed
      if constexpr (IsELoadPred) {
        // Get predication based on logical element coordinates.
        Tensor cE_mk = local_tile(
                make_identity_tensor(Shape_MK),
                make_shape(get<0>(TileShape{}), get<2>(TileShape{})),
                make_shape(m_coord, k_coord));                                                          // (BLK_M, BLK_K)
        Tensor tCcE = gmem_thr_copy_E.partition_S(cE_mk);                                            // (CPY,CPY_M,CPY_K)
        auto [atom, vec] = get_copy_atom_and_common_vec();
        // Coordinate comparison for out of bound (OOB) predication
        Tensor tZpE = cute::lazy::transform(zipped_divide(tCcE, vec), [&](auto const& c){ return cute::elem_less(c, Shape_MK); });
        // Copy
        cute::copy_if(atom, tZpE, zipped_divide(tCgE, vec), zipped_divide(tCrE_copy_view, vec));
      }
      else {
        // Copy
        cute::copy(cute::AutoVectorizingCopyWithAssumedAlignment<32>{}, tCgE, tCrE_copy_view);
      }
    }
    return tCrE;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC,
    class KTileIterator,
    class CtaTileCoord,
    class ProblemShape_MNKL
  >
  CUTLASS_DEVICE void
  mma(MainloopPipelineMK pipeline_mk,
      PipelineStateMK smem_pipe_read_mk,
      MainloopPipelineNK pipeline_nk,
      PipelineStateNK smem_pipe_read_nk,
      FrgTensorC& accum,
      KTileIterator k_tile_iter,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params,
      CtaTileCoord const& cta_tile_coord,
      ProblemShape_MNKL const& problem_shape_MNKL) {
    using namespace cute;

    CUTE_STATIC_ASSERT(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});         // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)
    Tensor sE = make_tensor(make_smem_ptr(shared_tensors.smem_E.begin()), SmemLayoutE{});         // (BLK_M,BLK_K,PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});   // (BLK_M,BLK_K,PIPE)
    auto SmemLayoutSFB_Ld = [SLayoutSFB = SmemLayoutSFB{}]() {
      if constexpr (IsCtaN64) {
        auto SLayoutSFB_tmp = SLayoutSFB;
        auto  new_shape =  make_shape (make_shape(make_shape(shape<0,0,0>(SLayoutSFB_tmp),
                                    shape<0,0,1>(SLayoutSFB_tmp) / _2{}), shape<0,1>(SLayoutSFB_tmp)),
                                    shape<1>(SLayoutSFB_tmp), shape<2>(SLayoutSFB_tmp));
        auto new_stride = stride(SLayoutSFB_tmp);
        return make_layout(new_shape, new_stride);
      }
      else {
        return SLayoutSFB;
      }
    }();
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()) +
                (IsCtaN64 && get<1>(cta_tile_coord) % 2 == 1 ? 8 : 0), SmemLayoutSFB_Ld);         // (BLK_N,BLK_K,PIPE)

    //
    // Define A/B/E partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate fragments and descriptors
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));                               // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));                               // (MMA,MMA_N,MMA_K)
    Tensor tCrE = partition_fragment_E(sE(_,_,Int<0>{}), thread_mma);                              // (MMA,MMA_M,MMA_K)
    Tensor tCrSFA = partition_fragment_SFA(sSFA(_,_,Int<0>{}), thread_mma);                        // (MMA,MMA_M,MMA_K)
    Tensor tCrSFB = partition_fragment_SFB(sSFB(_,_,Int<0>{}), thread_mma);                        // (MMA,MMA_N,MMA_K)

    //
    // Copy Atom A, B and E retiling
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(
          as_position_independent_swizzle_tensor(sA));                                        // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                                  //      (CPY,CPY_M,CPY_K)

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(
         as_position_independent_swizzle_tensor(sB));                                         // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                                  //      (CPY,CPY_N,CPY_K)

    auto tile_shape_mnk    = tile_shape(tiled_mma);
    auto smem_tiled_copy_E = make_tiled_copy_impl(SmemCopyAtomE{},
                                                  get_layoutE_TV(tiled_mma),
                                                  make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_E   = smem_tiled_copy_E.get_thread_slice(thread_idx);
    Tensor tCsE            = smem_thr_copy_E.partition_S(
                                  as_position_independent_swizzle_tensor(sE));                // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrE_copy_view  = smem_thr_copy_E.retile_D(tCrE);                                  //      (CPY,CPY_M,CPY_K)

    // SFA
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                    get_layoutSFA_TV(tiled_mma),
                                                    make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                  );
    auto smem_thr_copy_SFA   = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA            = smem_thr_copy_SFA.partition_S(
        as_position_independent_swizzle_tensor(sSFA));                                        // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrSFA_copy_view  = smem_thr_copy_SFA.retile_D(tCrSFA);                            //      (CPY,CPY_M,CPY_K)

    // SFB
    auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                    get_layoutSFB_TV(tiled_mma),
                                                    make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                  );
    auto smem_thr_copy_SFB   = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB            = smem_thr_copy_SFB.partition_S(
      as_position_independent_swizzle_tensor(sSFB));                                          // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrSFB_copy_view  = smem_thr_copy_SFB.retile_D(tCrSFB);                            //      (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCsE) == size<1>(tCrE_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB) * Int<AsymmetricKRatio>{});
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == Int<DispatchPolicy::StagesA>{});
    CUTE_STATIC_ASSERT_V(size<3>(tCsB) == Int<DispatchPolicy::StagesB>{});
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::StagesA>{} == size<2>(sA));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::StagesB>{} == size<2>(sB));

    CUTE_STATIC_ASSERT_V(size<1>(tCsSFA) == size<1>(tCrSFA_copy_view));                       // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCrSFA_copy_view));                       // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFA) == size<1>(accum));                                  // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFB) == size<2>(accum));                                  // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCsSFB));                                 // CPY_K
    CUTE_STATIC_ASSERT_V(size<3>(tCsSFA) == size<3>(tCsSFB));                                 // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sA)     == size<2>(sSFA));                                   // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sSFB)   == Int<DispatchPolicy::StagesA>{});                  // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sB)     == Int<DispatchPolicy::StagesB>{});                  // PIPE

    if constexpr (UseSmemE) {
      CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::StagesA>{} == size<2>(sE));
    }

    //
    // DEFINE FUNCTIONS FOR PIPELINED MAIN LOOP
    //

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineStateMK smem_pipe_release_mk = smem_pipe_read_mk;
    PipelineStateNK smem_pipe_release_nk = smem_pipe_read_nk;

    // Wait consumer barrier MK
    auto wait_barrier_mk = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto barrier_token_mk = pipeline_mk.consumer_try_wait(smem_pipe_read_mk);
      pipeline_mk.consumer_wait(smem_pipe_read_mk, barrier_token_mk);
    };

    // Wait consumer barrier NK
    auto wait_barrier_nk = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      auto barrier_token_nk = pipeline_nk.consumer_try_wait(smem_pipe_read_nk);
      pipeline_nk.consumer_wait(smem_pipe_read_nk, barrier_token_nk);
    };

    // Release consumer barrier MK, and move forward
    auto release_advance_mk = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      pipeline_mk.consumer_release(smem_pipe_release_mk);
      ++smem_pipe_read_mk;
      ++smem_pipe_release_mk;
    };

    // Release consumer barrier NK, and move forward
    auto release_advance_nk = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      pipeline_nk.consumer_release(smem_pipe_release_nk);
      ++smem_pipe_read_nk;
      ++smem_pipe_release_nk;
    };

    // Copy A from SMEM to register, and do transform if needed
    auto copy_transform_A = [&](auto m_block, auto k_block) CUTLASS_LAMBDA_FUNC_INLINE {
      // copy smem->rmem for A operand
      copy(smem_tiled_copy_A, tCsA(_,m_block,k_block,smem_pipe_read_mk.index()), tCrA_copy_view(_,m_block,k_block));
      // Perform transform if needed.
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_,m_block,k_block));
    };

    // Copy B from SMEM to register, and do transform if needed
    auto copy_transform_B = [&](auto n_block, auto k_block) CUTLASS_LAMBDA_FUNC_INLINE {
      // copy smem->rmem for B operand
      copy(smem_tiled_copy_B, tCsB(_,n_block,k_block,smem_pipe_read_nk.index()), tCrB_copy_view(_,n_block,k_block));
      // Perform transform if needed.
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,n_block,k_block));
    };

    // Copy SFA from SMEM to register
    auto copy_SFA = [&](auto m_block, auto k_block) CUTLASS_LAMBDA_FUNC_INLINE {
      // Copy smem->rmem for SFA operand
      copy(tCsSFA(_,m_block,k_block,smem_pipe_read_mk.index()), tCrSFA_copy_view(_,m_block,k_block));
    };

    // Copy SFB of all Ns from SMEM to register
    auto copy_SFBs = [&](auto k_block) CUTLASS_LAMBDA_FUNC_INLINE {
      // Copy smem->rmem for SFB operand
      copy(tCsSFB(_,_,k_block,smem_pipe_read_mk.index()), tCrSFB_copy_view(_,_,k_block));
    };

    // Copy E from SMEM to register
    auto copy_E = [&](auto m_block, auto k_block) CUTLASS_LAMBDA_FUNC_INLINE {
      // copy smem->rmem for E operand
      copy( recast<RegisterE>(tCsE(_,m_block,k_block,smem_pipe_read_mk.index())),
            recast<RegisterE>(tCrE_copy_view(_,m_block,k_block)));
    };

    constexpr auto M_BLOCK_MAX = size<1>(tCrA);
    constexpr auto N_BLOCK_MAX = size<1>(tCrB);
    constexpr auto K_BLOCK_MAX = size<2>(tCrA);
    constexpr auto K_BLOCK_STEP = K_BLOCK_MAX / Int<AsymmetricKRatio>{};

    // Perform mainloop gemm, when E is in SMEM.
    auto gemm_loop_with_SmemE = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      // WAIT on smem_pipe_read until data is available
      wait_barrier_mk();
      wait_barrier_nk();

      // Load A/B/E/SFA/SFB, then do gemm.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        // Copy smem->rmem for A/B/E operand
        copy_transform_A(_, k_block);
        copy_transform_B(_, k_block);
        copy_E(_, k_block);

        // Copy smem->rmem for SFA/SFB operand
        copy_SFA(_, k_block);
        copy_SFBs(k_block);

        // Gemm
        cute::gemm(tiled_mma,
                  make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block), tCrE(_,_,k_block)),
                  make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)),
                  accum);

      });

      cutlass::arch::NamedBarrier::sync(
        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);

      // Advance consumer pipeline mk/nk
      release_advance_mk();
      release_advance_nk();
    };

    // Perform mainloop gemm, when E is in GMEM.
    auto gemm_loop_with_GmemE = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      // Copy gmem->rmem for E operand
      auto blk_coord = make_coord(get<0>(cta_tile_coord), *k_tile_iter, get<3>(cta_tile_coord));     // (BLK_M,BLK_K,L)
      Tensor tCrE = load_E<IsF8F6F4>(mainloop_params, blk_coord, problem_shape_MNKL, thread_idx);
      ++k_tile_iter;

      // WAIT on smem_pipe_read until data is available
      wait_barrier_mk();
      wait_barrier_nk();

      for_each(make_int_sequence<K_BLOCK_STEP>{}, [&] (auto k_block) {
        // Copy smem->rmem for SFB operand. SFB needs to be copied with all N_BLOCK_MAX,
        //   as each LDS loads several groups of data needed by one MMA instruction.
        copy_SFBs(k_block);

        for_each(make_int_sequence<N_BLOCK_MAX>{}, [&] (auto n_block) {
          // Copy smem->rmem for B operand
          copy_transform_B(n_block, k_block);

          for_each(make_int_sequence<M_BLOCK_MAX>{}, [&] (auto m_block) {
            // Copy smem->rmem for A operand
            copy_transform_A(m_block, k_block);
            copy_SFA(m_block, k_block);

            // Gemm
            cute::gemm(tiled_mma,
                      make_zip_tensor(tCrA(_,m_block,k_block), tCrSFA(_,m_block,k_block), tCrE(_,m_block,k_block)),
                      make_zip_tensor(tCrB(_,n_block,k_block), tCrSFB(_,n_block,k_block)),
                      accum(_,m_block,n_block));
          });
        });
      });

      cutlass::arch::NamedBarrier::sync(
        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);

      // Advance consumer pipeline_nk
      release_advance_nk();
      // Wait next buffer
      wait_barrier_nk();

      for_each(make_int_sequence<K_BLOCK_STEP>{}, [&] (auto k_block) {
        auto k_block_a = k_block + K_BLOCK_STEP;

        // Copy smem->rmem for SFB operand. SFB needs to be copied with all N_BLOCK_MAX,
        //   as each LDS loads several groups of data needed by one MMA instruction.
        copy_SFBs(k_block_a);

        for_each(make_int_sequence<N_BLOCK_MAX>{}, [&] (auto n_block) {
          // Copy smem->rmem for B operand
          copy_transform_B(n_block, k_block);

          for_each(make_int_sequence<M_BLOCK_MAX>{}, [&] (auto m_block) {
            // Copy smem->rmem for A operand
            copy_transform_A(m_block, k_block_a);
            copy_SFA(m_block, k_block_a);

            // Gemm
            cute::gemm(tiled_mma,
                      make_zip_tensor(tCrA(_,m_block,k_block_a), tCrSFA(_,m_block,k_block_a), tCrE(_,m_block,k_block_a)),
                      make_zip_tensor(tCrB(_,n_block,k_block), tCrSFB(_,n_block,k_block_a)),
                      accum(_,m_block,n_block));
          });
        });
      });

      cutlass::arch::NamedBarrier::sync(
        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);

      // Advance consumer pipeline mk/nk
      release_advance_mk();
      release_advance_nk();
    };

    //
    // PIPELINED MAIN LOOP
    //

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // Case when A/B with same stages, and keep E in SMEM.
      if constexpr (UseSmemE) {
        gemm_loop_with_SmemE();
      }
      // Case when A/B with different stages, and keep E in GMEM.
      else {
        gemm_loop_with_GmemE();
      } // end if

    }
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipelineMK, PipelineStateMK, MainloopPipelineNK, PipelineStateNK, int) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
