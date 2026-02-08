/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/gemm/collective/builders/sm90_sparse_config.inl"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class LayoutPairAE_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90TmaGmmaWarpSpecializedSparse<Stages, ClusterShape, KernelSchedule>,
    TileShape_,
    ElementA_,
    LayoutPairAE_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedSparse<Stages, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using TiledMma = TiledMma_;
  using ElementA = ElementA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementAMmaRaw = typename ElementAMma::raw_type;
  using LayoutPairAE = LayoutPairAE_;
  using LayoutA = remove_cvref_t<decltype(get<0>(LayoutPairAE{}))>;
  using LayoutE = remove_cvref_t<decltype(get<1>(LayoutPairAE{}))>;
  using StrideA = decltype(cute::stride(LayoutA{}));
  using ElementB = ElementB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StrideB = StrideB_;
  using ElementEMma = typename TiledMma::ValTypeE;
  using ElementE = typename ElementEMma::raw_type;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using ArrayElementA = ElementA;
  using ArrayElementB = ElementB;

  static_assert(is_sparse<ElementAMma>::value, "ElementAMma is sparse");
  static_assert(!is_sparse<ElementA>::value, "ElementA is not sparse");

  static constexpr int ElementAMmaSparsity = ElementAMma::sparsity;
  static constexpr int ElementEMmaSparsity = ElementEMma::sparsity;

  // LayoutA is nested in the stride due to the sparsity.
  static constexpr bool is_A_mn_major = cute::is_same_v<decltype(get<0>(LayoutA{}.stride())), Int<ElementAMmaSparsity>>;
  static constexpr bool is_B_mn_major = cutlass::gemm::detail::is_major<0,StrideB>();

  using SparseConfig = cutlass::Sm90GemmSparseConfig<ElementAMma,
                                                     (is_A_mn_major ? GMMA::Major::MN : GMMA::Major::K),
                                                     ElementEMma,
                                                     decltype(cute::min(size<2>(TileShape{}),_128{}))>;

  // The offline permutation for the metadata.
  using SmemLayoutAtomE_ = typename SparseConfig::TensorEAtom;
  using SmemLayoutAtomE  = ComposedLayout<Swizzle<0,4,3>,
                                          smem_sparse_ptr_flag_bits<ElementEMmaSparsity, sizeof_bits_v<ElementE>>,
                                          SmemLayoutAtomE_>;

  // Metadata pathways
  using SmemCopyAtomE = AutoVectorizingCopy;
  using GmemCopyAtomE = GmemTiledCopyA;

  using CtaShape_MNK = TileShape;
  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;

  // One threads per CTA are producers (1 for operand tile)
  static constexpr int NumProducerThreadEvents = 1;

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M,K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (N,K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<is_A_mn_major, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutE = decltype(tile_to_shape(
      SmemLayoutAtomE{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<is_B_mn_major, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  static_assert(cute::is_void_v<SmemCopyAtomA>,
    "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
  static_assert(cute::is_void_v<SmemCopyAtomB>,
    "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  using TmaInternalElementA = cute::sparse_elem<ElementAMmaSparsity,
                                                cute::conditional_t<cute::is_same_v<ElementA, float>,
                                                                    cutlass::tfloat32_t,
                                                                    uint_bit_t<sizeof_bits_v<ElementAMmaRaw>>>>;
  using TmaInternalElementB = cute::conditional_t<cute::is_same_v<float, ElementB>,
                                                  tfloat32_t,
                                                  uint_bit_t<sizeof_bits_v<ElementBMma>>>;

  struct SharedStorage
  {
    struct TensorStorage {
      alignas(128) cute::ArrayEngine<ElementAMma, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(128) cute::ArrayEngine<ElementBMma, cute::cosize_v<SmemLayoutB>> smem_B;
      alignas(128) cute::ArrayEngine<ElementEMma, cute::cosize_v<SmemLayoutE>> smem_E;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 0;

  static constexpr uint32_t TmaTransactionBytesMK =
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementAMma>) +
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutE{})) * cute::sizeof_bits_v<ElementEMma>);

  static constexpr uint32_t TmaTransactionBytesNK =
        cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementBMma>);

  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{};
    LayoutA layout_a{};
    ElementB const* ptr_B{};
    StrideB dB{};
    ElementE const* ptr_E{};
    LayoutE layout_e{};
  };

  // Device side kernel params
  struct Params {

    using TMA_A = decltype(make_tma_copy_A_sm90<typename TmaInternalElementA::raw_type>(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), LayoutA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}));  // mcast along N mode for this M load, if any

    using TMA_E = decltype(make_tma_copy_A_sm90<uint64_t>( // use uint64_t to get the largest loading box.
        GmemCopyAtomE{},
        make_tensor(recast_ptr<ElementEMma>(nullptr), LayoutE{}),
        SmemLayoutE{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}));  // mcast along N mode for this M load, if any

    using TMA_B = decltype(make_tma_copy_B_sm90<TmaInternalElementB>(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}));  // mcast along M mode for this N load, if any

    TMA_A tma_load_a;
    TMA_E tma_load_e;
    TMA_B tma_load_b;
    LayoutA layout_a;
    LayoutE layout_e;
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
    auto [M,N,K,L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_E = recast_ptr<ElementEMma>(args.ptr_E);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, args.layout_a);
    Tensor tensor_e = make_tensor(ptr_E, args.layout_e);
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90<typename TmaInternalElementA::raw_type>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}); // mcast along N mode for this M load, if any

    typename Params::TMA_E tma_load_e = make_tma_copy_A_sm90<uint64_t>( // use uint64_t to get the largest loading box.
        GmemCopyAtomE{},
        tensor_e,
        SmemLayoutE{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}); // mcast along N mode for this M load, if any

    typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}); // mcast along M mode for this N load, if any

    uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk;

    return {
      tma_load_a,
      tma_load_e,
      tma_load_b,
      args.layout_a,
      args.layout_e,
      transaction_bytes
    };
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool size_check = true;
    // Check Alignment A
    if constexpr (is_A_mn_major) {
      size_check = size_check && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K/2,L), cute::make_stride(_1{}, M, M*K/2));
    }
    else { // If A is K-major
      size_check = size_check && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K/2,L), cute::make_stride(K/2, _1{}, M*K/2));
    }
    size_check = size_check && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!size_check) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }

    // Check if layout_a and layout_e is filled correctly
    auto layout_a_ref = SparseConfig::fill_layoutA(problem_shape_MNKL);
    auto layout_e_ref = SparseConfig::fill_layoutE(problem_shape_MNKL);
    bool layout_check = true;
    layout_check = layout_check && (layout_a_ref == args.layout_a);
    layout_check = layout_check && (layout_e_ref == args.layout_e);

    if (!layout_check) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Layout_a/e mismatch.\n");
    }

    return size_check && layout_check;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_e.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
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
    auto [M,N,K,L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(mainloop_params.layout_a.shape());                      // (m,k,l)
    Tensor mE_mkl = mainloop_params.tma_load_e.get_tma_tensor(mainloop_params.layout_e.shape());                      // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gE_mkl = local_tile(mE_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl, gE_mkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB, class TensorE,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorE> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sE = make_tensor(make_smem_ptr(shared_tensors.smem_E.begin()), SmemLayoutE{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      auto [gA_mkl, gB_nkl, gE_mkl] = load_inputs;

      // Define the CTA-in-cluster Layout and Coord
      Layout cta_layout_mnk = make_layout(ClusterShape{});
      auto cta_coord_mnk = cta_layout_mnk.get_flat_coord(block_rank_in_cluster);

      // TMA Multicast Masks
      uint16_t mcast_mask_a = create_tma_multicast_mask<1>(cta_layout_mnk, cta_coord_mnk);
      uint16_t mcast_mask_e = create_tma_multicast_mask<1>(cta_layout_mnk, cta_coord_mnk);
      uint16_t mcast_mask_b = create_tma_multicast_mask<0>(cta_layout_mnk, cta_coord_mnk);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(get<1>(cta_coord_mnk));
      auto block_tma_e = mainloop_params.tma_load_e.get_slice(get<1>(cta_coord_mnk));
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(get<0>(cta_coord_mnk));

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gE = gE_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tEgE = block_tma_e.partition_S(gE);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tEsE = block_tma_e.partition_D(sE);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count)
      {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_e.with(*tma_barrier, mcast_mask_e), tEgE(_,_,_,*k_tile_iter), tEsE(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
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

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutE{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)
    Tensor sE = as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(shared_tensors.smem_E.begin()), SmemLayoutE{}));                   // (BLK_M,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and
                  stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    auto copy_atom_E = Copy_Atom<SmemCopyAtomE, uint32_t>{};

    Tensor tCsE = partition_E(thread_mma, sE(_,_,Int<0>{}));            // (MMA,MMA_M,MMA_K)
    Tensor tCrE = make_fragment_like<ElementEMma>(tCsE);                // (MMA,MMA_M,MMA_K)

    auto smem_tiled_copy_E = make_tiled_copy_E(copy_atom_E, tiled_mma);
    auto smem_thr_copy_E   = smem_tiled_copy_E.get_thread_slice(thread_idx);

    Tensor tEsE  = smem_thr_copy_E.partition_S(sE);                     // (ECPY,ECPY_M,ECPY_K)
    Tensor tErE  = smem_thr_copy_E.retile_D(tCrE);                      // (ECPY,ECPY_M,ECPY_K)

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0; --k_tile_prologue)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      int read_stage = smem_pipe_read.index();

      // Load metadata smem->rmem for one stage
      copy(smem_tiled_copy_E, tEsE(_,_,_,read_stage), tErE);

      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        cute::gemm(tiled_mma, make_zip_tensor(tCrA(_,_,k_block,read_stage), tErE(_,_,k_block)), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      int read_stage = smem_pipe_read.index();

      // Load metadata smem->rmem for one stage
      copy(smem_tiled_copy_E, tEsE(_,_,_,read_stage), tErE);

      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        cute::gemm(tiled_mma, make_zip_tensor(tCrA(_,_,k_block,read_stage), tErE(_,_,k_block)), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

private:

  template <class MMA_Atom,
            class AtomLayoutMNK,
            class PermutationMNK,
            class ETensor>
  CUTE_HOST_DEVICE static constexpr
  auto
  thrfrg_E(TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK> const& mma, ETensor&& etensor)
  {
    using TiledMma = TiledMMA<MMA_Atom, AtomLayoutMNK, PermutationMNK>;

    CUTE_STATIC_ASSERT_V(rank(etensor) >= Int<2>{});

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(PermutationMNK{}),
                            get<2>(PermutationMNK{}));
    auto t_tensor = logical_divide(etensor, t_tile);                 // (PermM,PermK)

    // Tile the tensor for the Atom
    auto e_tile = make_tile(make_layout(size<0>(typename TiledMma::AtomShape_MNK{})),
                            make_layout(size<2>(typename TiledMma::AtomShape_MNK{})));
    auto e_tensor = zipped_divide(t_tensor, e_tile);                 // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    using AtomLayoutE_TV = typename TiledMma::Atom::Traits::ELayout;
    auto tv_tensor = e_tensor.compose(AtomLayoutE_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(mma.thr_layout_vmnk_)),
                                        make_layout(size<3>(mma.thr_layout_vmnk_))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  template<class... MArgs>
  CUTE_HOST_DEVICE static constexpr
  auto
  get_layoutE_TV(TiledMMA<MArgs...> const& mma)
  {
    // (M,K) -> (M,K)
    auto ref_E = make_layout(make_shape(tile_size<0>(mma), tile_size<2>(mma)));
    // (ethrid,val) -> (M,K)
    auto layoutE_TV = thrfrg_E(mma, ref_E);

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto etile = make_tile(_,
                            make_tile(make_layout(make_shape (size<1>(mma.thr_layout_vmnk_), size<2>(mma.thr_layout_vmnk_)),
                                                  make_stride(               Int<1>{} ,                Int<0>{} )),
                                      _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(mma.thr_layout_vmnk_);

    // (thr_idx,val) -> (M,K)
    return layoutE_TV.compose(etile, _).compose(thridx_2_thrid, _);
  }

  template <class... MArgs, class ETensor>
  CUTE_HOST_DEVICE static constexpr
  auto
  partition_E(ThrMMA<MArgs...> const& thr_mma, ETensor&& etensor)
  {
    auto thr_tensor = make_tensor(static_cast<ETensor&&>(etensor).data(), thrfrg_E(thr_mma, etensor.layout()));

    auto thr_vmk = make_coord(get<0>(thr_mma.thr_vmnk_), make_coord(get<1>(thr_mma.thr_vmnk_), get<3>(thr_mma.thr_vmnk_)));
    return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }

  template <class... CArgs, class... MArgs>
  CUTE_HOST_DEVICE static constexpr
  auto
  make_tiled_copy_E(Copy_Atom<CArgs...> const& copy_atom,
                    TiledMMA<MArgs...>  const& mma)
  {
    return make_tiled_copy_impl(copy_atom, get_layoutE_TV(mma), make_shape(tile_size<0>(mma),tile_size<2>(mma)));
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
