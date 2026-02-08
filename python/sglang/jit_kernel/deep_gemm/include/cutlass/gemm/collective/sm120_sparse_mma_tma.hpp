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

template <
  int StagesA,
  int StagesB,
  int StagesE,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class TileShape_,
  class ElementA_,
  class LayoutPairAE_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomPairA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm120TmaWarpSpecializedSparse<StagesA, StagesB, StagesE, SchedulerPipelineStageCount, ClusterShape>,
    TileShape_,
    ElementA_,
    LayoutPairAE_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomPairA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> {
  //
  // Type Aliases
  //
  using TiledMma = TiledMma_;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;
  using DispatchPolicy = MainloopSm120TmaWarpSpecializedSparse<StagesA, StagesB, StagesE, SchedulerPipelineStageCount, ClusterShape>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementAMmaRaw = typename ElementAMma::raw_type;
  using LayoutPairAE = LayoutPairAE_;
  using LayoutA =  remove_cvref_t<decltype(get<0>(LayoutPairAE{}))>;
  using LayoutE =  remove_cvref_t<decltype(get<1>(LayoutPairAE{}))>;
  using StrideA =  remove_cvref_t<decltype(get<2>(LayoutPairAE{}))>;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using ElementEMma = typename TiledMma::ValTypeE;
  using ElementE = typename ElementEMma::raw_type;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = remove_cvref_t<decltype(get<0>(SmemCopyAtomPairA_{}))>;
  using SmemCopyAtomE = remove_cvref_t<decltype(get<1>(SmemCopyAtomPairA_{}))>;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using GmemTiledCopyE = GmemTiledCopyA_;
  using ArrayElementA = ElementA;
  using ArrayElementB = ElementB;
  using RegisterE = typename remove_extent<typename TiledMma::MMA_Op::ERegisters>::type;

  using RuntimeDataTypeA = void*;
  using RuntimeDataTypeB = void*;

  static constexpr int ThreadCount = size(TiledMma{});
  static constexpr int ElementAMmaSparsity = ElementAMma::sparsity;
  static constexpr int ElementEMmaSparsity = ElementEMma::sparsity;

  // Asymmetric buffering
  // Tensor A/B could have different buffering, with TILEK, and STAGEs.
  //    It let AsymmetricKRatio equals TILEK_A / TILEK_B, to make sure A/B's
  //    pipeline keep same steps when produce / consume data.
  static constexpr int AsymmetricKRatio = DispatchPolicy::StagesA != DispatchPolicy::StagesB ? 2 : 1;

  using TileShapeB = decltype(make_shape(size<0>(TileShape{}),
                                         size<1>(TileShape{}),
                                         ceil_div(size<2>(TileShape{}), Int<AsymmetricKRatio>{})));

  // Use two MainloopPipeline for A and B separately.
  using MainloopPipelineMK = cutlass::PipelineTmaAsync<DispatchPolicy::StagesA>;
  using MainloopPipelineNK = cutlass::PipelineTmaAsync<DispatchPolicy::StagesB>;

  using PipelineParams = typename MainloopPipelineMK::Params;
  using PipelineStateMK  = typename cutlass::PipelineState<DispatchPolicy::StagesA>;
  using PipelineStateNK  = typename cutlass::PipelineState<DispatchPolicy::StagesB>;

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

  static_assert(DispatchPolicy::StagesA >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(DispatchPolicy::StagesB >= 2, "Specialization requires Stages set to value 2 or more.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::StagesA>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShapeB{}), shape<2>(TileShapeB{}), Int<DispatchPolicy::StagesB>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

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
  using SmemAllocTypeA = cute::conditional_t<IsF8F6F4, sparse_elem<ElementAMmaSparsity, uint8_t>, ElementAMma>;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4, uint8_t, ElementBMma>;

  static constexpr bool is_A_mn_major = cute::is_same_v<decltype(stride<0>(LayoutA{})), Int<ElementAMmaSparsity>>;
  using SparseConfig = cutlass::Sm1xxGemmSparseConfig<
                                    ElementAMma,
                                    cute::conditional_t<is_A_mn_major, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>,
                                    ElementEMma>;
  using SmemLayoutAtomE_ = typename SparseConfig::TensorEAtom;
  using SmemLayoutAtomE  = ComposedLayout<Swizzle<0,4,3>,
                                          smem_sparse_ptr_flag_bits<ElementEMmaSparsity, sizeof_bits_v<ElementE>>,
                                          SmemLayoutAtomE_>;
  using SmemLayoutE = decltype(tile_to_shape(
                  SmemLayoutAtomE{},
                  make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::StagesE>{}),
                  conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  static constexpr int SmemSizeE  = UseSmemE ? cosize(SmemLayoutE{}) : 0;
  static constexpr int StageSizeE = UseSmemE ? cosize(take<0,2>(SmemLayoutE{})) : 0;
  // Check if metetata fetching needs predicator
  using TensorEAtomM = typename SparseConfig::TensorEAtomM;
  using TensorEAtomK = typename SparseConfig::TensorEAtomK;
  static constexpr bool IsELoadPred = not (TensorEAtomM{} == size<0>(TileShape{}) && TensorEAtomK{} == size<2>(TileShape{}));

  static_assert(rank(SmemLayoutAtomE{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomE{})) == 0, "SmemLayoutAtomE must evenly divide tile shape.");

  // Set the bytes transferred in this TMA transaction
  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementAMma>) +
    cutlass::bits_to_bytes(StageSizeE * cute::sizeof_bits_v<ElementEMma>));
  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>));
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
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
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy<TmaInternalElementA>(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<sparse_elem<ElementAMmaSparsity,ElementA>>(nullptr), LayoutA{}),
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
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_E tma_load_e;
    LayoutA layout_a;
    LayoutE layout_e;
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

    auto ptr_A = recast_ptr<sparse_elem<ElementAMmaSparsity, ElementA>>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);
    auto ptr_E = recast_ptr<ElementEMma>(args.ptr_E);

    Tensor tensor_a = make_tensor(ptr_A, args.layout_a);
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));
    Tensor tensor_e = make_tensor(ptr_E, args.layout_e);
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
        make_shape(shape<1>(TileShapeB{}), shape<2>(TileShapeB{})),
        _1{});
    typename Params::TMA_E tma_load_e = make_tma_copy<ElementE>(
        GmemTiledCopyE{},
        tensor_e,
        SmemLayoutE{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{});
    return {
      tma_load_a,
      tma_load_b,
      tma_load_e,
      args.layout_a,
      args.layout_e,
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

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{},  make_coord(_,_,_), Step<_1, X,_1>{});       // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShapeB{}, make_coord(_,_,_), Step< X,_1,_1>{});       // (BLK_N,BLK_K,n,k,l)
    Tensor gE_mkl = local_tile(mE_mkl, TileShape{},  make_coord(_,_,_), Step<_1, X,_1>{});       // (BLK_N,BLK_K,n,k,l)
    return cute::make_tuple(gA_mkl, gB_nkl, gE_mkl);
  }

  /// Issues loads for A/E only (used when DMA warp is split).
  template <
    class TensorA, class TensorB, class TensorE,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load_MK(
      Params const& mainloop_params,
      MainloopPipelineMK pipeline,
      PipelineStateMK smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorE> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});         // (BLK_M,BLK_K,PIPE)
    Tensor sE = make_tensor(make_smem_ptr(shared_tensors.smem_E.begin()), SmemLayoutE{});         // (BLK_M,BLK_K,PIPE)

    // Prepare the TMA loads for A and B
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gE_mkl = get<2>(load_inputs);
    auto block_tma_a = mainloop_params.tma_load_a.get_slice(0);
    auto block_tma_e = mainloop_params.tma_load_e.get_slice(0);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,  k)
    Tensor gE = gE_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,  k)

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);                                                // (TMA,TMA_M,TMA_K,   k)
    Tensor tAsA = block_tma_a.partition_D(sA);                                                // (TMA,TMA_M,TMA_K,PIPE)
    Tensor tEgE = block_tma_e.partition_S(gE);                                                // (TMA,TMA_M,TMA_K,   k)
    Tensor tEsE = block_tma_e.partition_D(sE);                                                // (TMA,TMA_M,TMA_K,PIPE)

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
        copy(mainloop_params.tma_load_a.with(*tma_barrier), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        if constexpr (UseSmemE) {
          copy(mainloop_params.tma_load_e.with(*tma_barrier), tEgE(_,_,_,*k_tile_iter), tEsE(_,_,_,write_stage));
        }
      }

      if constexpr (!UseSmemE) {
        auto blk_coord_mkl = make_coord(get<0>(blk_coord), *k_tile_iter, get<3>(blk_coord));         // (BLK_M,BLK_K,L)
        prefetch(make_local_E(mainloop_params, blk_coord_mkl));
      }

      // Advance smem_pipe_write
      ++k_tile_iter;
      ++smem_pipe_write;
    }
  }

  /// Issues loads for B only (used when DMA warp is split).
  template <
    class TensorA, class TensorB, class TensorE,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load_NK(
      Params const& mainloop_params,
      MainloopPipelineNK pipeline,
      PipelineStateNK smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorE> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {

    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});     //     (BLK_N,BLK_K,PIPE)

    // Prepare the TMA loads for A and B
    Tensor gB_nkl = get<1>(load_inputs);
    auto block_tma_b = mainloop_params.tma_load_b.get_slice(0);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                //     (BLK_N,BLK_K,   k)

    // Applies the mapping from block_tma_a
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
        copy(mainloop_params.tma_load_b.with(*tma_barrier), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
      }

      // Advance smem_pipe_write
      ++k_tile_iter;
      ++smem_pipe_write;
    }
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
      auto get_copy_atom_and_common_vec = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
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

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});         // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)
    Tensor sE = make_tensor(make_smem_ptr(shared_tensors.smem_E.begin()), SmemLayoutE{});         // (BLK_M,BLK_K,PIPE)

    //
    // Define A/B/E partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate fragments and descriptors
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));                               // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));                               // (MMA,MMA_N,MMA_K)
    Tensor tCrE = partition_fragment_E(sE(_,_,Int<0>{}), thread_mma);                              // (MMA,MMA_M,MMA_K)

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
      fp4_shift_A(MMAOp{}, tCrA(_,m_block,k_block));
    };

    // Copy B from SMEM to register, and do transform if needed
    auto copy_transform_B = [&](auto n_block, auto k_block) CUTLASS_LAMBDA_FUNC_INLINE {
      // copy smem->rmem for B operand
      copy(smem_tiled_copy_B, tCsB(_,n_block,k_block,smem_pipe_read_nk.index()), tCrB_copy_view(_,n_block,k_block));
      // Perform transform if needed.
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_B(MMAOp{}, tCrB(_,n_block,k_block));
    };

    // Copy E from SMEM to register
    auto copy_E = [&](auto m_block, auto k_block) CUTLASS_LAMBDA_FUNC_INLINE {
      // copy smem->rmem for E operand
      copy( recast<RegisterE>(tCsE(_,m_block,k_block,smem_pipe_read_mk.index())),
            recast<RegisterE>(tCrE_copy_view(_,m_block,k_block)));
    };

    // TILE M/N/K for one TILE block
    constexpr auto M_BLOCK_MAX = size<1>(tCrA);
    constexpr auto N_BLOCK_MAX = size<1>(tCrB);
    constexpr auto K_BLOCK_MAX = size<2>(tCrA);
    constexpr auto K_BLOCK_STEP = K_BLOCK_MAX / Int<AsymmetricKRatio>{};

    // Perform mainloop gemm, when E is in SMEM.
    auto gemm_loop_with_SmemE = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      // WAIT on smem_pipe_read until data is available
      wait_barrier_mk();
      wait_barrier_nk();

      // Load A/B/E, then do gemm.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        for_each(make_int_sequence<N_BLOCK_MAX>{}, [&] (auto n_block) {
          // Copy smem->rmem for B operand
          copy_transform_B(n_block, k_block);

          for_each(make_int_sequence<M_BLOCK_MAX>{}, [&] (auto m_block) {
            // Copy smem->rmem for A operand
            copy_transform_A(m_block, k_block);
            copy_E(m_block, k_block);

            // Gemm
            cute::gemm(tiled_mma,
                      make_zip_tensor(tCrA(_,m_block,k_block), tCrE(_,m_block,k_block)),
                      tCrB(_,n_block,k_block),
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
        for_each(make_int_sequence<N_BLOCK_MAX>{}, [&] (auto n_block) {
          // Copy smem->rmem for B operand
          copy_transform_B(n_block, k_block);

          for_each(make_int_sequence<M_BLOCK_MAX>{}, [&] (auto m_block) {
            // Copy smem->rmem for A operand
            copy_transform_A(m_block, k_block);

            // Gemm
            cute::gemm(tiled_mma,
                      make_zip_tensor(tCrA(_,m_block,k_block), tCrE(_,m_block,k_block)),
                      tCrB(_,n_block,k_block),
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
        for_each(make_int_sequence<N_BLOCK_MAX>{}, [&] (auto n_block) {
          // Copy smem->rmem for B operand
          copy_transform_B(n_block, k_block);

          for_each(make_int_sequence<M_BLOCK_MAX>{}, [&] (auto m_block) {
            // Copy smem->rmem for A operand
            copy_transform_A(m_block, k_block_a);

            // Gemm
            cute::gemm(tiled_mma,
                      make_zip_tensor(tCrA(_,m_block,k_block_a), tCrE(_,m_block,k_block_a)),
                      tCrB(_,n_block,k_block),
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

    } // end loop k_tile_count
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipelineMK, PipelineStateMK, MainloopPipelineNK, PipelineStateNK, int) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
