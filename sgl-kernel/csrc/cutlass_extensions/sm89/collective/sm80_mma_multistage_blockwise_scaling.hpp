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

 // Inspired by: sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp

#pragma once

#include "cutlass/cutlass.h"
// #include "cutlass/gemm/dispatch_policy.hpp"
#include <cutlass_extensions/sm89/dispatch_policy_extention.hpp>
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/detail/blockwise_scale_layout.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class ClusterShape_,
  class TileShape_,
  class ElementA_,
  class StridePairA_,
  class ElementB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_
>
struct CollectiveMma<
    MainloopSm80CpAsyncBlockScalingExtension<
      Stages,
      ClusterShape_>,
    TileShape_,
    ElementA_,
    StridePairA_,
    ElementB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_
   >
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm80CpAsyncBlockScalingExtension<
                          Stages,
                          ClusterShape_>;
  using TileShape = TileShape_;
  // Follow the change in TestSmall: TileShape switch to CtaShape
  // In legacy arch, it should be same
  using CtaShape_MNK = TileShape;
  using ElementA = ElementA_;
  using StrideA = cute::tuple_element_t<0,StridePairA_>;
  using LayoutSFA = cute::tuple_element_t<1,StridePairA_>;
  using ElementB = ElementB_;
  using StrideB = cute::tuple_element_t<0,StridePairB_>;
  using LayoutSFB = cute::tuple_element_t<1,StridePairB_>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ElementBlockScale = ElementAccumulator;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int ScaleGranularityM = size<0,0>(LayoutSFA{});
  static constexpr int ScaleGranularityN = size<0,0>(LayoutSFB{});
  static constexpr int ScaleGranularityK = size<1,0>(LayoutSFA{});

  static_assert(size<2>(TileShape{}) % ScaleGranularityK == 0, "BLK_K must be divisible by ScaleGranularityK");
  static_assert(ScaleGranularityK % size<2>(typename TiledMma::AtomShape_MNK{}) == 0,
    "ScaleGranularityK must be divisible by shape K of the MMA atom.");
  static constexpr int ScalePromotionInterval = ScaleGranularityK / size<2>(typename TiledMma::AtomShape_MNK{});
  static_assert(ScalePromotionInterval >= size<2>(TileShape{}) / tile_size<2>(TiledMma{}),
    "ScalePromotionInterval must be greater than or equal to the number of stages of the MMA atom.");
  static_assert(ScalePromotionInterval % (size<2>(TileShape{}) / tile_size<2>(TiledMma{})) == 0,
    "ScalePromotionInterval must be a multiple of the number of stages of the MMA atom.");
  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;

  using ScaleConfig = ::cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
  using SmemLayoutAtomSFA = decltype(ScaleConfig::smem_atom_layoutSFA(TileShape{}));
  using SmemLayoutAtomSFB = decltype(ScaleConfig::smem_atom_layoutSFB(TileShape{}));

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  static_assert(DispatchPolicy::Stages >= 2, "CpAsync mainloop must have at least 2 stages in the pipeline.");

  // Block scaling gmem-to-smem copy atom
  //  we can have partial tiles in M or N, so don't vectorize those loads
  using CopyAtomSFA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;
  using CopyAtomSFB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;
  using GmemTiledCopySFA = decltype(make_tiled_copy(
    CopyAtomSFA{},
    Layout<Shape<Int<32>>>{},
    Layout<Shape<_1>>{}));
  using GmemTiledCopySFB = decltype(make_tiled_copy(
    CopyAtomSFB{},
    Layout<Shape<Int<32>>>{},
    Layout<Shape<_1>>{}));

  static constexpr int AlignmentSFA = 1;
  static constexpr int AlignmentSFB = 1;

  // Block scaling smem layout
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));

  struct SharedStorage
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
    cute::array_aligned<ElementBlockScale, cute::cosize_v<SmemLayoutSFA>> smem_SFA; // ScaleMsPerTile x PIPE_K
    cute::array_aligned<ElementBlockScale, cute::cosize_v<SmemLayoutSFB>> smem_SFB; // ScaleNsPerTile x PIPE_K
  };

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementBlockScale const* ptr_SFA;
    LayoutSFA layout_SFA;
    ElementBlockScale const* ptr_SFB;
    LayoutSFB layout_SFB;
  };

  // Device side kernel params
  using Params = Arguments;

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& _, Arguments const& args, void* workspace) {
    (void) workspace;
    return args;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class FrgTensorD,
    class TensorA,
    class TensorB,
    class TensorSFA,
    class TensorSFB,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,                   // (BLK_M, BLK_K, K_TILES)
      TensorB gB,                   // (BLK_N, BLK_K, K_TILES)
      TensorSFA gSFA,               // (SCLAE_M, BLK_K, K_TILES)
      TensorSFB gSFB,               // (SCLAE_N, BLK_K, K_TILES)
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_gmem<TensorA>::value,    "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value,    "B tensor must be gmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(is_gmem<TensorSFA>::value,    "A scale tensor must be gmem resident.");
    static_assert(is_gmem<TensorSFB>::value,    "B scale tensor must be gmem resident.");

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(storage.smem_SFA.data()), SmemLayoutSFA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sSFB = make_tensor(make_smem_ptr(storage.smem_SFB.data()), SmemLayoutSFB{}); // (BLK_N,BLK_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sSFA));      // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sSFB));      // PIPE

    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyA gmem_tiled_copy_A;
    GmemTiledCopyB gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    GmemTiledCopySFA gmem_tiled_copy_SFA;
    GmemTiledCopySFB gmem_tiled_copy_SFB;
    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_slice(thread_idx);
    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_slice(thread_idx);

    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);
    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    //
    // PREDICATES
    //

    // Allocate predicate tensors for m and n
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});

    // Construct identity layout for sA and sB
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tAcA = gmem_thr_copy_A.partition_S(cA);                             // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tBcB = gmem_thr_copy_B.partition_S(cB);                             // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Set predicates for m bounds
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(tApA); ++m) {
      tApA(m,0) = get<0>(tAcA(0,m,0)) < get<0>(residue_mnk);  // blk_m coord < residue_m
    }
    // Set predicates for n bounds
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<0>(tBpB); ++n) {
      tBpB(n,0) = get<0>(tBcB(0,n,0)) < get<1>(residue_mnk);  // blk_n coord < residue_n
    }

    // Allocate predicate tensors for scale a and b
    Tensor tSFApSFA = make_tensor<bool>(shape(filter_zeros(tSFAsSFA(_,_,_,_0{})))); // (CPY,CPY_M,CPY_K)
    Tensor tSFBpSFB = make_tensor<bool>(shape(filter_zeros(tSFBsSFB(_,_,_,_0{})))); // (CPY,CPY_M,CPY_K)

    Tensor cSFA = make_identity_tensor(make_shape(get<0>(shape(sSFA)), get<1>(shape(sSFA))));
    Tensor cSFB = make_identity_tensor(make_shape(get<0>(shape(sSFB)), get<1>(shape(sSFB))));

    Tensor tSFAcSFA = gmem_thr_copy_SFA.partition_S(cSFA);
    Tensor tSFBcSFB = gmem_thr_copy_SFB.partition_S(cSFB);
    Tensor tSFAcSFA_compact = filter_zeros(tSFAcSFA, tSFAsSFA(_,_,_,_0{}).stride());
    Tensor tSFBcSFB_compact = filter_zeros(tSFBcSFB, tSFBsSFB(_,_,_,_0{}).stride());

    // Since scale granularity K is multiple of BLK_K we do not have to consider if that is OOB
    bool load_sfa = thread_idx < cute::min(32, ScaleMsPerTile);
    bool load_sfb = thread_idx < cute::min(32, ScaleNsPerTile);
    // auto residue_sfm = get<0>(residue_mnk) / ScaleGranularityM;
    auto residue_sf = cute::shape_div(residue_mnk,
                        ResidueMNK{ScaleGranularityM, ScaleGranularityN, ScaleGranularityK});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tSFApSFA); ++i) {
      tSFApSFA(i) = load_sfa && elem_less(get<0, 1>(tSFAcSFA_compact(i)),get<0>(residue_sf));
    }
    // bool load_sfb = thread_idx < cute::min(32, ScaleNsPerTile);
    // auto residue_sfn = get<1>(residue_mnk) / ScaleGranularityN;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tSFBpSFB); ++i) {
      tSFBpSFB(i) = load_sfb && elem_less(get<0, 1>(tSFBcSFB_compact(i)), get<1>(residue_sf));
    }

    //
    // PREFETCH
    //

    // Clear the smem tiles to account for predicated off loads
    clear(tAsA);
    clear(tBsB);
    // clear(tSFAsSFA);
    // clear(tSFBsSFB);
     if (load_sfa) {
      clear(tSFAsSFA);
    }
    if (load_sfb) {
      clear(tSFBsSFB);
    }
    // Start async loads, no k-residue handling needed
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      if (k_tile_count <= 0) {
        clear(tApA);
        clear(tBpB);
        clear(tSFApSFA);
        clear(tSFBpSFB);
      }
      copy_if(gmem_tiled_copy_A, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe));  // CpAsync
      copy_if(gmem_tiled_copy_B, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe));  // CpAsync
      copy_if(gmem_tiled_copy_SFA, tSFApSFA, filter_zeros(tSFAgSFA(_,_,_,*k_tile_iter)), filter_zeros(tSFAsSFA(_,_,_,k_pipe)));
      copy_if(gmem_tiled_copy_SFB, tSFBpSFB, filter_zeros(tSFBgSFB(_,_,_,*k_tile_iter)), filter_zeros(tSFBsSFB(_,_,_,k_pipe)));
      cp_async_fence();
      ++k_tile_iter;
      --k_tile_count;
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA  = thr_mma.partition_fragment_A(sA(_,_,0));                    // (MMA,MMA_M,MMA_K)
    Tensor tCrB  = thr_mma.partition_fragment_B(sB(_,_,0));                    // (MMA,MMA_N,MMA_K)

    Tensor sSFA_mnk_view = make_tensor(sSFA.data(), make_layout(
        make_shape(get<0>(shape(SmemLayoutSFA{})),
                   get<1>(TileShape{}),
                   make_shape(get<1>(shape(SmemLayoutSFA{})),
                   get<2>(shape(SmemLayoutSFA{})))),
        make_stride(get<0>(stride(SmemLayoutSFA{})),
                    _0{},
                    make_stride(get<1>(stride(SmemLayoutSFA{})),
                      get<2>(stride(SmemLayoutSFA{})))))
    );
    Tensor sSFB_mnk_view = make_tensor(sSFB.data(), make_layout(
        make_shape(get<0>(TileShape{}),
                   get<0>(shape(SmemLayoutSFB{})),
                   make_shape(get<1>(shape(SmemLayoutSFB{})),
                    get<2>(shape(SmemLayoutSFB{})))),
        make_stride(_0{},
                    get<0>(stride(SmemLayoutSFB{})),
                    make_stride(get<1>(stride(SmemLayoutSFB{})),
                      get<2>(stride(SmemLayoutSFB{})))))
    );

    Tensor tCsSFA = thr_mma.partition_C(sSFA_mnk_view);                 // (MMA,MMA_M,MMA_N,(MMA_K,PIPE))
    Tensor tCsSFB = thr_mma.partition_C(sSFB_mnk_view);                 // (MMA,MMA_M,MMA_N,(MMA_K,PIPE))
    // Per block scale values for operand A and B
    // Since scale factors always broadcast across MMA_K we slice that away
    Tensor tCrSFA = make_tensor_like<ElementBlockScale>(tCsSFA(_, _, _, _0{}));                     // (MMA,MMA_M,MMA_N)
    Tensor tCrSFB = make_tensor_like<ElementBlockScale>(tCsSFB(_, _, _, _0{}));                     // (MMA,MMA_M,MMA_N)

    Tensor tCrAccum = cute::make_fragment_like(accum);              // (MMA_M,MMA_N)
    clear(tCrAccum);

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K
    CUTE_STATIC_ASSERT_V(size(tCrSFA) == size(accum));                         // MMA_M * MMA_N
    CUTE_STATIC_ASSERT_V(size(tCrSFB) == size(accum));                         // MMA_M * MMA_N

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_A   = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A     = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA           = smem_thr_copy_A.partition_S(sA);                   // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);                    // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB              = smem_thr_copy_B.partition_S(sB);                // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view    = smem_thr_copy_B.retile_D(tCrB);                 // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
      // Load per block scale values from shared memory to registers
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(DispatchPolicy::Stages-1); --k_tile_count)
    {
      // Pipeline the outer products with a static for loop.
      //
      // Note, the for_each() function is required here to ensure `k_block` is of type Int<N>.
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
      {
        if (k_block == K_BLOCK_MAX - 1)
        {
          // Slice the smem_pipe_read smem
          tCsA_p = tCsA(_,_,_,smem_pipe_read);
          tCsB_p = tCsB(_,_,_,smem_pipe_read);

          // Commit the smem for smem_pipe_read
          cp_async_wait<DispatchPolicy::Stages-2>();
          __syncthreads();
        }

        // Load A, B shmem->regs for k_block+1
        auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
        copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        // Copy gmem to smem before computing gemm on each k-pipe
        if (k_block == 0)
        {
          
          copy(tCsSFA(_,_,_,make_coord(_0{}, smem_pipe_read)), tCrSFA);
          copy(tCsSFB(_,_,_,make_coord(_0{}, smem_pipe_read)), tCrSFB);
          // Set all predicates to false if we are going to overshoot bounds
          if (k_tile_count <= 0) {
            clear(tApA);
            clear(tBpB);
            clear(tSFApSFA);
            clear(tSFBpSFB);
          }
          copy_if(gmem_tiled_copy_A, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));
          copy_if(gmem_tiled_copy_B, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));
          copy_if(gmem_tiled_copy_SFA, tSFApSFA, filter_zeros(tSFAgSFA(_,_,_,*k_tile_iter)),
            filter_zeros(tSFAsSFA(_,_,_,smem_pipe_write)));
          copy_if(gmem_tiled_copy_SFB, tSFBpSFB, filter_zeros(tSFBgSFB(_,_,_,*k_tile_iter)),
            filter_zeros(tSFBsSFB(_,_,_,smem_pipe_write)));
          cp_async_fence();
          ++k_tile_iter;

          // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
        }

        // Transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});
        // Thread-level register gemm for k_block
        cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrAccum);
      });
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrSFA(_0{}) * tCrSFB(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          accum(i) += tCrAccum(i) * scale_ab;
          tCrAccum(i) = 0;
        }
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrSFB(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
          filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          accum(i) += tCrAccum(i) * tCrSFA(i);
          tCrAccum(i) = 0;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementBlockScale scale_a = tCrSFA(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
          filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          accum(i) += tCrAccum(i) * tCrSFB(i);
          tCrAccum(i) = 0;
        }
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          accum(i) += tCrAccum(i) * tCrSFA(i) * tCrSFB(i);
          tCrAccum(i) = 0;
        }
      }
      // Load per block scale values from shared memory to registers
      // copy(tCsSFA(_,_,_,make_coord(_0{}, smem_pipe_read)), tCrSFA);
      // copy(tCsSFB(_,_,_,make_coord(_0{}, smem_pipe_read)), tCrSFB);
    }

    cp_async_wait<0>();
    __syncthreads();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////