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
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/transform/collective/sm90_wgmma_transpose.hpp"
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
  class ClusterShape_,
  class TileShape_,
  class KernelSchedule,
  class ElementA_,
  class StrideA_,
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
    MainloopSm90CpAsyncGmmaRmemAWarpSpecialized<Stages,ClusterShape_,KernelSchedule>,
    TileShape_,
    ElementA_,
    StrideA_,
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
  using DispatchPolicy = MainloopSm90CpAsyncGmmaRmemAWarpSpecialized<Stages,ClusterShape_,KernelSchedule>;
  using TileShape = TileShape_;
  using ClusterShape = ClusterShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  // Swap and transpose A/B for A k-major layout and B mn-major layout since WGMMA is k-major only (e.g. tf32, Fp32, Int8, Fp8 WGMMA)
  static constexpr bool IsLayoutAkBmn =
    cute::is_same_v<gemm::detail::StrideToLayoutTagA_t<StrideA>, layout::RowMajor> &&
    cute::is_same_v<gemm::detail::StrideToLayoutTagB_t<StrideB>, layout::RowMajor>;

  static constexpr bool IsInputSizeTwoBytes = sizeof(ElementA) == 2 && sizeof(ElementB) == 2;
  static constexpr bool SwapAB =  !IsInputSizeTwoBytes && IsLayoutAkBmn;
  using InternalGmemTiledCopyA = cute::conditional_t<!SwapAB, GmemTiledCopyA, GmemTiledCopyB>;
  using InternalGmemTiledCopyB = cute::conditional_t<!SwapAB, GmemTiledCopyB, GmemTiledCopyA>;
  using InternalSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using InternalSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using InternalSmemCopyAtomA   = cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using InternalSmemCopyAtomB   = cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  using InternalElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  using InternalElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;
  using InternalStrideA  = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using InternalStrideB  = cute::conditional_t<!SwapAB, StrideB, StrideA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineAsync<DispatchPolicy::Stages>;
  using PipelineState    = typename MainloopPipeline::PipelineState;
  using PipelineParams   = typename MainloopPipeline::Params;

  static_assert(cute::rank(InternalSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(InternalSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      InternalSmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  // If A mn-layout and B mn-layout, transposing B matrix since WGMMA is k-major only (e.g. tf32, fp32, fp8, int8).
  static constexpr bool IsLayoutAmnBmn =
    cute::is_same_v<gemm::detail::StrideToLayoutTagA_t<StrideA>, layout::ColumnMajor> &&
    cute::is_same_v<gemm::detail::StrideToLayoutTagB_t<StrideB>, layout::RowMajor>;
  static constexpr bool TransposeB = !IsInputSizeTwoBytes && IsLayoutAmnBmn;
  using TransposeOperandB = decltype(cutlass::transform::collective::detail::make_transpose_operand_b(
                                      0, 0, TiledMma{}, SmemLayoutB{}, InternalSmemLayoutAtomB{},
                                      InternalElementB{}, cute::bool_constant<TransposeB>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                    cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source A from rmem and B operand from smem_desc for this mainloop.");

  using GmmaSmemLayoutAtomB = decltype(transform::collective::detail::gmma_smem_transpose_or_passthrough<
      TransposeB, InternalSmemLayoutAtomB, InternalElementB>());

  // SmemLayoutB for GMMA is different from SmemLayoutB for TMA if TransposeB
  using GmmaSmemLayoutB = decltype(tile_to_shape(
      GmmaSmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  static_assert(!SwapAB || !TransposeB, "Cannot SwapAB and TransposeB at the same time.");
  static_assert(TransposeB xor (cute::is_same_v<SmemLayoutB, GmmaSmemLayoutB>),
    "Should be same layout if not TransposeB.");
  static_assert(!TransposeB || (cutlass::bits_to_bytes(size<1>(SmemLayoutB{}) * sizeof_bits<InternalElementB>::value)) == 128,
    "SmemLayoutB K must be 128bytes to be transposed.");
  static_assert(!transform::collective::detail::use_universal_transposition<InternalSmemLayoutAtomB, InternalElementB>(),
    "Warp specialized ARF kernels have not supported universal B transposition yet.");

  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<256, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>, 256> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>, 256> smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A = nullptr;
    StrideA dA{};
    ElementB const* ptr_B = nullptr;
    StrideB dB{};
    uint32_t mma_promotion_interval = 4;
  };

  // Device side kernel params
  struct Params {
    InternalElementA const* ptr_A = nullptr;
    InternalStrideA dA{};
    InternalElementB const* ptr_B = nullptr;
    InternalStrideB dB{};
    uint32_t mma_promotion_interval = 4;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
    [[maybe_unused]] ProblemShape const& problem_shape,
    Arguments const& args,
    [[maybe_unused]] void* workspace) {
    if constexpr (not SwapAB) {
      return {
        reinterpret_cast<InternalElementA const*>(args.ptr_A),
        args.dA,
        reinterpret_cast<InternalElementB const*>(args.ptr_B),
        args.dB
      };
    }
    else {
      return {
        reinterpret_cast<InternalElementA const*>(args.ptr_B),
        args.dB,
        reinterpret_cast<InternalElementB const*>(args.ptr_A),
        args.dA
      };
    }
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    implementable = implementable && cutlass::detail::check_alignment<GmemTiledCopyA::NumValSrc>(cute::make_shape(M,K,L), StrideA{});
    implementable = implementable && cutlass::detail::check_alignment<GmemTiledCopyB::NumValSrc>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA,
    class TensorB,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  load(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      TensorA const& gA_in,
      TensorB const& gB_in,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      TensorStorage& shared_tensors)
  {
    using namespace cute;

    static_assert(is_gmem<TensorA>::value, "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value, "B tensor must be gmem resident.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

    // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
    // This aligns the tensor with BLK_K for all but the 0th k_tile
    Tensor gA = domain_offset(make_coord(0, get<2>(residue_mnk), 0), gA_in);
    Tensor gB = domain_offset(make_coord(0, get<2>(residue_mnk), 0), gB_in);

    // Partition the copying of A and B tiles across the threads
    InternalGmemTiledCopyA gmem_tiled_copy_a;
    InternalGmemTiledCopyB gmem_tiled_copy_b;
    auto gmem_thr_copy_a = gmem_tiled_copy_a.get_slice(thread_idx);
    auto gmem_thr_copy_b = gmem_tiled_copy_b.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_a.partition_S(gA);                        // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_a.partition_D(sA);                        // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_b.partition_S(gB);                        // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_b.partition_D(sB);                        // (BCPY,BCPY_N,BCPY_K,PIPE)

    // Allocate predicate tensors for m and n
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});

    // Construct identity layout for sA and sB
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tAcA = gmem_thr_copy_a.partition_S(cA);                             // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tBcB = gmem_thr_copy_b.partition_S(cB);                             // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

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

    // 0-th stage with predication on k to account for residue
    {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);
      int write_stage = smem_pipe_write.index();

      // Copy gmem to smem for *k_tile_iter, predicating for k residue
      Tensor tAgAk = tAgA(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tAsA); ++k) {
        if (get<1>(tAcA(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gA shifted)
          copy_if(gmem_tiled_copy_a, tApA(_,k), tAgAk(_,_,k), tAsA(_,_,k,write_stage));
        }
        else {
          clear(tAsA(_,_,k,write_stage));
        }
      }
      Tensor tBgBk = tBgB(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tBsB); ++k) {
        if (get<1>(tBcB(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gB shifted)
          copy_if(gmem_tiled_copy_b, tBpB(_,k), tBgBk(_,_,k), tBsB(_,_,k,write_stage));
        }
        else {
          clear(tBsB(_,_,k,write_stage));
        }
      }

      ++k_tile_iter;
      --k_tile_count;

      // UNLOCK smem_pipe_write
      pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);
      int write_stage = smem_pipe_write.index();

      // Copy gmem to smem for *k_tile_iter
      copy_if(gmem_tiled_copy_a, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
      copy_if(gmem_tiled_copy_b, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
      ++k_tile_iter;

      // UNLOCK smem_pipe_write
      pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write) {
    // Issue the epilogue waits
    /* This helps avoid early exit of blocks in Cluster
     * Waits for all stages to either be released (all
     * Consumer UNLOCKs), or if the stage was never used
     * then would just be acquired since the phase was
     * still inverted from make_producer_start_state
     */
    pipeline.producer_tail(smem_pipe_write);
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
      Params const& mainloop_params)
  {
    using namespace cute;
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(InternalSmemLayoutAtomA{}) == 2, "InternalSmemLayoutAtomA must be rank 2.");
    static_assert(cute::rank(InternalSmemLayoutAtomB{}) == 2, "InternalSmemLayoutAtomB must be rank 2.");
    static_assert(!cute::is_void_v<InternalSmemCopyAtomA>,
      "SM90 GMMA mainloops must specify a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<InternalSmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    // Obtain warp index
    int warp_idx = canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});         // (BLK_M,BLK_K,PIPE)
    Tensor sA  = as_position_independent_swizzle_tensor(sA_);                                     // (BLK_M,BLK_K,PIPE)
    Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)
    Tensor sB  = as_position_independent_swizzle_tensor(sB_);                                     // (BLK_M,BLK_K,PIPE)

    // If TransposeB, GMMA will read from transposed B layout SMEM
    Tensor gmma_sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), GmmaSmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);
    auto mma_warpgroup_slice = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    // Allocate fragments and descriptors
    Tensor tCsA = mma_thread_slice.partition_A(sA);
    Tensor tCrA = mma_thread_slice.partition_fragment_A(sA(_,_,Int<0>{}));                    // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = mma_warpgroup_slice.partition_B(gmma_sB);                                   // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = mma_warpgroup_slice.make_fragment_B(tCsB);                                  // (MMA,MMA_N,MMA_K,PIPE)

    //
    // Copy Atom A retiling
    //


    auto smem_tiled_copy_A = make_tiled_copy_A(InternalSmemCopyAtomA{}, tiled_mma);

    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);

    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                                       // (CPY,CPY_M,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                                            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                                            // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                                                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    TransposeOperandB transpose = cutlass::transform::collective::detail::make_transpose_operand_b(
                                    warp_idx, warp_group_thread_idx, tiled_mma, SmemLayoutB{},
                                    InternalSmemLayoutAtomB{}, InternalElementB{},
                                    cute::bool_constant<TransposeB>{});

    warpgroup_fence_operand(accum);
    // first k tile
    {
      pipeline.consumer_wait(smem_pipe_read);

      int read_stage = smem_pipe_read.index();

      ++smem_pipe_read;

      bool skip_wait = (pipeline.consumer_try_wait(smem_pipe_read) == BarrierStatus::WaitDone);

      // copy smem->rmem for A operand
      copy(smem_tiled_copy_A, tCsA(_,_,0,read_stage), tCrA_copy_view(_,_,0));
      // transpose B operand in SMEM
      transpose(sB, gmma_sB, read_stage, 0);

      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA) - 1; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_,_,k_block + 1,read_stage), tCrA_copy_view(_,_,k_block + 1));
        if (k_block == 0) {
          transpose(sB, gmma_sB, read_stage, 1);
          transpose.synchronize();
        }
        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_commit_batch();
      }

      warpgroup_wait<2>();


      if (k_tile_count - 1 > 0) {
        if (!skip_wait) {
          pipeline.consumer_wait(smem_pipe_read);
        }
        copy(smem_tiled_copy_A, tCsA(_,_,0,smem_pipe_read.index()), tCrA_copy_view(_,_,0));
        transpose(sB, gmma_sB, smem_pipe_read.index(), 0);
      }

      warpgroup_arrive();
      // (V,M) x (V,N) => (V,M,N)
      cute::gemm(tiled_mma, tCrA(_,_,size<2>(tCrA) - 1), tCrB(_,_,size<2>(tCrA) - 1,read_stage), accum);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      warpgroup_commit_batch();
      warpgroup_wait<2>();
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    --k_tile_count;
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      ++smem_pipe_read;
      bool skip_wait = (pipeline.consumer_try_wait(smem_pipe_read) == BarrierStatus::WaitDone);

      warpgroup_fence_operand(accum);
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        if (k_block == size<2>(tCrA) - 1) {
          if (!skip_wait) {
            pipeline.consumer_wait(smem_pipe_read);
          }
          copy(smem_tiled_copy_A, tCsA(_,_,0,smem_pipe_read.index()), tCrA_copy_view(_,_,0));
          // transpose B operand in SMEM
          transpose(sB, gmma_sB, smem_pipe_read.index(), 0);
        } else {
          copy(smem_tiled_copy_A, tCsA(_,_,k_block + 1,read_stage), tCrA_copy_view(_,_,k_block + 1));
          // transpose B operand in SMEM
          if (k_block < 2) {
            transpose.synchronize(k_block);                                      // make transpose of k_block available
          }
          if (k_block == 0) {
            transpose(sB, gmma_sB, read_stage, 1);
          }
        }

        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_commit_batch();
        warpgroup_wait<2>();
        if (k_block == 1) {
          // release prior barrier
          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it
          ++smem_pipe_release;
        }
      }
      warpgroup_fence_operand(accum);

    }

    warpgroup_fence_operand(accum);

    if (k_tile_count > 0) {
      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      warpgroup_fence_operand(accum);
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA) - 1; ++k_block) {
        copy(smem_tiled_copy_A, tCsA(_,_,k_block + 1,read_stage), tCrA_copy_view(_,_,k_block + 1));
        if (k_block < 2) {
          transpose.synchronize(k_block);                                           // make k_block transpose available
        }
        if (k_block == 0) {
          transpose(sB, gmma_sB, read_stage, 1);
        }
        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_commit_batch();
        warpgroup_wait<2>();
        if (k_block == 1) {
          // release prior barrier
          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it
          ++smem_pipe_release;
        }
      }

      warpgroup_arrive();
      // (V,M) x (V,N) => (V,M,N)
      cute::gemm(tiled_mma, tCrA(_,_,size<2>(tCrA) - 1), tCrB(_,_,size<2>(tCrA) - 1,read_stage), accum);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      warpgroup_commit_batch();
      warpgroup_wait<2>();
      warpgroup_fence_operand(accum);
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
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
