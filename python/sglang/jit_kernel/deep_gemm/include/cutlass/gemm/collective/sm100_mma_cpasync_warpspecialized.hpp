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
#include "cutlass/detail/cluster.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/arch/memory.h"

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
    MainloopSm100UmmaCpAsyncWarpSpecialized<
      Stages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape>,
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
  using TiledMma = TiledMma_;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;

  // Statically asserting to ensure only 1x1x1 cluster shape & 1sm setup is received
  static_assert(size(AtomThrShapeMNK{}) == 1, "Lower alignment SM100 GEMM only supports 1SM MMA");
  static_assert(size(ClusterShape{}) == 1, "CPASYNC does not support multicast so the cluster shape is restricted to 1, 1, 1");

  using DispatchPolicy = MainloopSm100UmmaCpAsyncWarpSpecialized<
                          Stages,
                          SchedulerPipelineStageCount,
                          AccumulatorPipelineStageCount,
                          ClusterShape>;
  // TileShape refers to MmaTileShape to adapt for runtime cluster shape
  using TileShape = TileShape_;

  CUTE_STATIC_ASSERT_V(evenly_divides(TileShape{}, tile_shape(TiledMma{})),
                       "Static cluster shape used: TileShape should be evenly divided by TiledMma");

  // Define A and B block shapes
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));
  using LoadShapeA_MK = decltype(select<0,2>(TileShape{}));
  using LoadShapeB_NK = decltype(select<1,2>(TileShape{}));

  // CtaShape_MNK is queried from collective in all kernel layers
  using CtaShape_MNK = TileShape;

  using ElementA = ElementA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StrideB = StrideB_;

  static constexpr bool IsRuntimeDataTypeA = cute::is_same_v<ElementA, cutlass::type_erased_dynamic_float8_t>;
  static constexpr bool IsRuntimeDataTypeB = cute::is_same_v<ElementB, cutlass::type_erased_dynamic_float8_t>;

  static_assert(IsRuntimeDataTypeA == IsRuntimeDataTypeB,
                "ElementA and ElementB should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;


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

  using MainloopPipeline = cutlass::PipelineUmmaConsumerAsync<DispatchPolicy::Stages, AtomThrShapeMNK>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;

  static_assert(size(GmemTiledCopyA{}) == size(GmemTiledCopyB{}), "A and B GmemTiledCopy should share the same thread count");
  static constexpr int NumLoadThreads = size(GmemTiledCopyA{});

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
  // (MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE)
  using MmaSmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using LoadSmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      append(LoadShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using MmaSmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using LoadSmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      append(LoadShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));


  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");

  using SmemAllocTypeA = cute::conditional_t<cute::sizeof_bits_v<ElementAMma> < 8, uint8_t, ElementAMma>;
  using SmemAllocTypeB = cute::conditional_t<cute::sizeof_bits_v<ElementBMma> < 8, uint8_t, ElementBMma>;

  using BitTypeElementA = cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>;
  using BitTypeElementB = cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>;

  using ArrayElementA = cute::conditional_t<IsRuntimeDataTypeA, BitTypeElementA, ElementA>;
  using ArrayElementB = cute::conditional_t<IsRuntimeDataTypeB, BitTypeElementB, ElementB>;

  using RuntimeDataTypeA = cute::conditional_t<IsRuntimeDataTypeA, cute::UMMA::MXF8F6F4Format, void*>;
  using RuntimeDataTypeB = cute::conditional_t<IsRuntimeDataTypeB, cute::UMMA::MXF8F6F4Format, void*>;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<SmemAllocTypeA, cute::cosize_v<LoadSmemLayoutA>> smem_A;
      cute::array_aligned<SmemAllocTypeB, cute::cosize_v<LoadSmemLayoutB>> smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };

  // Expose shared storage for tensors/pipelines separately to allow kernel layer to reorder them.
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ArrayElementA const* ptr_A{nullptr};
    StrideA dA{};
    ArrayElementB const* ptr_B{nullptr};
    StrideB dB{};
    RuntimeDataTypeA runtime_data_type_a{};
    RuntimeDataTypeB runtime_data_type_b{};
  };

  // Device side kernel params
  struct Params {
    ArrayElementA const* ptr_A{nullptr};
    StrideA dA{};
    ArrayElementB const* ptr_B{nullptr};
    StrideB dB{};
    RuntimeDataTypeA runtime_data_type_a;
    RuntimeDataTypeB runtime_data_type_b;
  };

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
    auto ptr_A = recast_ptr<ElementAMma>(args.ptr_A);
    auto ptr_B = recast_ptr<ElementBMma>(args.ptr_B);

    return {
      args.ptr_A,
      args.dA,
      args.ptr_B,
      args.dB,
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
    bool implementable = true;
    implementable = implementable && cutlass::detail::check_alignment<GmemTiledCopyA::NumValSrc>(cute::make_shape(M,K,L), StrideA{});
    implementable = implementable && cutlass::detail::check_alignment<GmemTiledCopyB::NumValSrc>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for CpAsync.\n");
    }
    return implementable;
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));  // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

    return acc_shape;
  }

  /// Set up the data needed by this collective for load.
  /// Return tuple element contain
  /// gA_mkl - The tiled tensor for input A
  /// gB_nkl - The tiled tensor for input B
  /// tAsA - partitioned smem tensor for A
  /// tBsB - partitioned smem tensor for B
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_tensors) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // Represent the full tensors
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.ptr_A), make_shape(M,K,L), params.dA); //(m,k,l)
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.ptr_B), make_shape(N,K,L), params.dB); //(n,k,l)
    // Partition for cpasync
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{}); // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{}); // (BLK_N,BLK_K,n,k,l)

    // Build the coordinate tensors with the same shape as input matrices
    Tensor cA_mk  = make_identity_tensor(make_shape(M,K));
    Tensor cB_nk  = make_identity_tensor(make_shape(N,K));

    // Slice the coordinate tensors in the same way as A/B tensor partitioning
    Tensor cgA_mk = local_tile(cA_mk, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{}); // (BLK_M,BLK_K,m,k)
    Tensor cgB_nk = local_tile(cB_nk, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{}); // (BLK_N,BLK_K,n,k)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), LoadSmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), LoadSmemLayoutB{});

    GmemTiledCopyA gmem_to_smem_a_tiled_copy;
    GmemTiledCopyB gmem_to_smem_b_tiled_copy;

    int thread_idx = threadIdx.x % NumLoadThreads;
    auto thr_copy_a = gmem_to_smem_a_tiled_copy.get_slice(thread_idx);
    auto thr_copy_b = gmem_to_smem_b_tiled_copy.get_slice(thread_idx);

    return cute::make_tuple(
        gA_mkl, gB_nkl, // gmem
        cgA_mk, cgB_nk, // crd
        sA, sB,         // smem
        problem_shape_MNKL, 
        gmem_to_smem_a_tiled_copy, gmem_to_smem_b_tiled_copy, 
        thr_copy_a, thr_copy_b);
  }

  /// Set up the data needed by this collective for mma compute.
  template <class FrgEngine, class FrgLayout>
  CUTLASS_DEVICE auto
  mma_init(
      Params const& params,
      [[maybe_unused]] cute::tuple<cute::Tensor<FrgEngine, FrgLayout>, cute::Tensor<FrgEngine, FrgLayout>> const& accumulators_pair,
      TensorStorage& shared_tensors) const {
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), MmaSmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), MmaSmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA = TiledMma::make_fragment_A(sA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = TiledMma::make_fragment_B(sB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sA));                                     // PIPE
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
    class CTensorA, class CTensorB,
    class STensorA, class STensorB,
    class ProblemShape_MNKL,
    class TiledCopyA, class TiledCopyB,
    class ThreadCopyA, class ThreadCopyB,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load(
    Params const& params,
    MainloopPipeline mainloop_pipeline,
    MainloopPipelineState mainloop_pipe_producer_state,
    cute::tuple<GTensorA, GTensorB,
                CTensorA, CTensorB,
                STensorA, STensorB,
                ProblemShape_MNKL,
                TiledCopyA, TiledCopyB,
                ThreadCopyA, ThreadCopyB> const& load_inputs,
    TileCoordMNKL const& cta_coord_mnkl,
    KTileIterator k_tile_iter, int k_tile_count) {
    // Unpack from load_inputs
    GTensorA tAgA_mkl = get<0>(load_inputs);
    GTensorB tBgB_nkl = get<1>(load_inputs);
    CTensorA cgA_mk = get<2>(load_inputs);
    CTensorB cgB_nk = get<3>(load_inputs);
    STensorA sA = get<4>(load_inputs);
    STensorB sB = get<5>(load_inputs);
    ProblemShape_MNKL problem_shape_MNKL = get<6>(load_inputs);
    TiledCopyA gmem_to_smem_a_tiled_copy = get<7>(load_inputs);
    TiledCopyB gmem_to_smem_b_tiled_copy = get<8>(load_inputs);
    ThreadCopyA thr_copy_a = get<9>(load_inputs);
    ThreadCopyB thr_copy_b = get<10>(load_inputs);
    auto [M,N,K,L] = problem_shape_MNKL;

    // Slice out the work coord from partitioned tensors
    Tensor gA_in = tAgA_mkl(_, _, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
    Tensor gB_in = tBgB_nkl(_, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    // Repeat slicing out coordinate tensor exactly the same as input tensor does
    Tensor cgA_mk_in = cgA_mk(_, _, get<0>(cta_coord_mnkl), _);
    Tensor cgB_nk_in = cgB_nk(_, _, get<1>(cta_coord_mnkl), _);

    auto k_residue    = K - size<1>(gB_in) * size<2>(gA_in);

    // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
    // This aligns the tensor with BLK_K for all but the 0th k_tile
    Tensor gA = domain_offset(make_coord(0, k_residue, 0), gA_in);
    Tensor gB = domain_offset(make_coord(0, k_residue, 0), gB_in);

    Tensor cA = domain_offset(make_coord(0, k_residue, 0), cgA_mk_in);
    Tensor cB = domain_offset(make_coord(0, k_residue, 0), cgB_nk_in);

    auto tAgA = thr_copy_a.partition_S(gA);
    auto tAsA = thr_copy_a.partition_D(sA);

    auto tBgB = thr_copy_b.partition_S(gB);
    auto tBsB = thr_copy_b.partition_D(sB);

    // Allocate predicate tensors for m and n
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});

    Tensor tAcA = thr_copy_a.partition_S(cA);
    Tensor tBcB = thr_copy_b.partition_S(cB);

    // Copy gmem to smem for *k_tile_iter, predicating for k residue
    Tensor tAgAk = tAgA(_,_,_,*k_tile_iter);
    Tensor tBgBk = tBgB(_,_,_,*k_tile_iter);

    // Repeating on predicators with the same operations on tAgA and tBgB
    Tensor tAcAk = tAcA(_,_,_,*k_tile_iter);
    Tensor tBcBk = tBcB(_,_,_,*k_tile_iter);

    // Set predicates for m bounds
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(tApA); ++m) {
      tApA(m,0) = elem_less(get<0>(tAcAk(0,m,0)), M);  // blk_m coord < M
    }
    // Set predicates for n bounds
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<0>(tBpB); ++n) {
      tBpB(n,0) = elem_less(get<0>(tBcBk(0,n,0)), N);  // blk_n coord < N
    }

    // 0-th stage with predication on k to account for residue
    // For performance consideration,
    // this predicated block for K-tail is only activated when there is k-residue
    if (k_residue != 0 && k_tile_count > 0)  {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state);
      int write_stage = mainloop_pipe_producer_state.index();

      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tAsA); ++k) {
        if ( int(get<1>(tAcAk(0,0,k))) >= 0) {      // blk_k coord < K
          copy_if(gmem_to_smem_a_tiled_copy, tApA(_,k), tAgAk(_,_,k), tAsA(_,_,k,write_stage));
        }
        else {
          clear(tAsA(_,_,k,write_stage));
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tBsB); ++k) {
        if (int(get<1>(tBcBk(0,0,k))) >= 0) {      // blk_k coord < K
          copy_if(gmem_to_smem_b_tiled_copy, tBpB(_,k), tBgBk(_,_,k), tBsB(_,_,k,write_stage));
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

    auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      auto mainloop_pipe_producer_state_curr = mainloop_pipe_producer_state;
      ++mainloop_pipe_producer_state;
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state_curr, barrier_token);
      barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);
      int write_stage = mainloop_pipe_producer_state_curr.index();

      copy_if(gmem_to_smem_a_tiled_copy, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
      copy_if(gmem_to_smem_b_tiled_copy, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

      mainloop_pipeline.producer_commit(mainloop_pipe_producer_state_curr, cutlass::arch::cpasync_barrier_arrive);
      
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
    class FrgEngine, class FrgLayout,
    class FragmentA, class FragmentB
  >
  CUTLASS_DEVICE auto
  mma(MainloopPipeline mainloop_pipeline,
      MainloopPipelineState mainloop_pipe_consumer_state,
      cute::tuple<cute::Tensor<FrgEngine, FrgLayout>, cute::Tensor<FrgEngine, FrgLayout>> const& accumulators_pair,
      cute::tuple<TiledMma, FragmentA, FragmentB> const& mma_inputs,
      int k_tile_count
  ) {
    static_assert(is_tmem<FrgEngine>::value, "Accumulator must be tmem resident.");
    static_assert(rank(FrgLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");
    auto accumulators = get<0>(accumulators_pair);
    auto [tiled_mma, tCrA, tCrB] = mma_inputs;

    //
    // PIPELINED MAIN LOOP
    //
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state);

      int read_stage = mainloop_pipe_consumer_state.index();

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulators);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }

      mainloop_pipeline.consumer_release(mainloop_pipe_consumer_state);
      --k_tile_count;
      ++mainloop_pipe_consumer_state;
  }

    return mainloop_pipe_consumer_state;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
