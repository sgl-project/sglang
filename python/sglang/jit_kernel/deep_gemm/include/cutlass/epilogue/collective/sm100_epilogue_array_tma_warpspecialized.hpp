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

/*! \file
  \brief Functor performing elementwise operations used by Ptr-Array and Grouped Gemm epilogue.
*/



#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/trace.h"

#include "cute/tensor.hpp"
#include "cutlass/cuda_host_adapter.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_,
  class CtaTileShape_, // (CTA_M,CTA_N,CTA_K)
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpT2R_,
  class CopyOpG2S_,
  class SmemLayoutAtomC_,
  class CopyOpS2R_,
  class CopyOpS2G_,
  class SmemLayoutAtomD_,
  class CopyOpR2S_,
  class CopyOpR2R_
>
class CollectiveEpilogue<
    Sm100PtrArrayTmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_>,
    CtaTileShape_,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    FusionCallbacks_,
    CopyOpT2R_,
    CopyOpG2S_,
    SmemLayoutAtomC_,
    CopyOpS2R_,
    CopyOpS2G_,
    SmemLayoutAtomD_,
    CopyOpR2S_,
    CopyOpR2R_
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = Sm100PtrArrayTmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_>;
  using CtaTileShape = CtaTileShape_;
  using EpilogueTile = EpilogueTile_;
  using FusionCallbacks = FusionCallbacks_;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;
  using CopyOpT2R = CopyOpT2R_;
  using CopyOpG2S = CopyOpG2S_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using CopyOpS2R = CopyOpS2R_;
  using CopyOpS2G = CopyOpS2G_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using CopyOpR2S = CopyOpR2S_;
  using CopyOpR2R = CopyOpR2R_;

  using ThreadEpilogueOp = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
  using GmemTiledCopyC = CopyOpG2S;
  using GmemTiledCopyD = CopyOpS2G;

  constexpr static int ThreadCount = 128;

  static_assert(!is_layout<EpilogueTile>::value && is_tuple<EpilogueTile>::value, "EpilogueTile must be a cute::Tile or cute::Shape");
  static_assert(rank(EpilogueTile{}) == 2, "EpilogueTile must be rank-2: [EPI_TILE_M, EPI_TILE_N]");

private:

  constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
  constexpr static bool is_destination_supported = not cute::is_void_v<ElementD>;
  using GmemElementD = cute::conditional_t<is_destination_supported, ElementD, fusion::get_element_aux_t<FusionCallbacks>>;
  using GmemElementC = cute::conditional_t<is_source_supported, ElementC, GmemElementD>; // prevents void ref breakages
  static_assert(not cute::is_void_v<GmemElementD>, "GmemElementD is void");

  using SmemElementD = typename cutlass::detail::get_unpacked_element_type<GmemElementD>::type;
  using SmemElementC = typename cutlass::detail::get_unpacked_element_type<GmemElementC>::type;
  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  static_assert(StagesC >= 1, "StagesC must be >= 1");
  static_assert(StagesD >= 1, "StagesD must be >= 1");
  
  constexpr static bool ReuseSmemC = ReuseSmemC_ && is_destination_supported;

  constexpr static bool is_m_major_C = detail::is_m_major<InternalStrideC>();
  constexpr static bool is_m_major_D = detail::is_m_major<InternalStrideD>();

  using SmemLayoutStageC = decltype(tile_to_shape(SmemLayoutAtomC{}, product_each(shape(EpilogueTile{})),
      cute::conditional_t<is_m_major_C, Step<_2,_1>, Step<_1,_2>>{} ));
  using SmemLayoutStageD = decltype(tile_to_shape(SmemLayoutAtomD{}, product_each(shape(EpilogueTile{})),
      cute::conditional_t<is_m_major_D, Step<_2,_1>, Step<_1,_2>>{} ));

  constexpr static int StageCBits = cosize_v<SmemLayoutStageC> * sizeof_bits_v<SmemElementC>;
  constexpr static int StageDBits = cosize_v<SmemLayoutStageD> * sizeof_bits_v<SmemElementD>;
  constexpr static int MaxStageBits = cute::max(StageCBits, StageDBits);
  constexpr static int StrideStageC = (ReuseSmemC ? MaxStageBits : StageCBits) / sizeof_bits_v<SmemElementC>;
  constexpr static int StrideStageD = (ReuseSmemC ? MaxStageBits : StageDBits) / sizeof_bits_v<SmemElementD>;

  using SmemLayoutC = decltype(cute::append<3>(SmemLayoutStageC{}, Layout<Int<StagesC>,                        Int<StrideStageC>>{}));
  using SmemLayoutD = decltype(cute::append<3>(SmemLayoutStageD{}, Layout<Int<ReuseSmemC ? StagesC : StagesD>, Int<StrideStageD>>{}));

  constexpr static bool support_smem_reuse = is_source_supported && is_destination_supported && StagesD <= StagesC
                                              && MaxStageBits % sizeof_bits_v<SmemElementC> == 0
                                              && MaxStageBits % sizeof_bits_v<SmemElementD> == 0;
  static_assert(not (ReuseSmemC && not support_smem_reuse), "Smem reuse requirements not met");

  constexpr static size_t SmemAlignmentC = cutlass::detail::alignment_for_swizzle(SmemLayoutC{});
  constexpr static size_t SmemAlignmentD = cutlass::detail::alignment_for_swizzle(SmemLayoutD{});
  constexpr static size_t MaxSmemAlignment = cute::max(SmemAlignmentC, SmemAlignmentD);

  // Not unroll epi subtile loop when the activation op is heavy to reduce instruction size and register pressure.
  constexpr static bool UnrollEpiLoop =
    not cutlass::epilogue::thread::kIsHeavy_member_or_false<typename ThreadEpilogueOp::ActivationFn>::value;
  // TMA store delay only benefits with loop unrolling
  constexpr static bool DelayTmaStore = DelayTmaStore_ and UnrollEpiLoop;

  struct CollectiveStorageWithC {
    alignas(SmemAlignmentC) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageWithoutC {
    cute::array<SmemElementC, 0> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageReuseC {
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

public:
  // TMA pipeline for loading C
  using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>;
  using LoadPipelineState = cutlass::PipelineState<StagesC>;
  constexpr static uint32_t TmaTransactionBytes = StageCBits / 8;
  constexpr static uint32_t MinTensorMapWorkspaceAlignment = 64;

  // TMA pipeline for storing D
  using StorePipeline = cute::conditional_t<ReuseSmemC,
                          cutlass::PipelineTmaStore<StagesC, StagesD-1>,
                          cutlass::PipelineTmaStore<StagesD>>;
  using StorePipelineState = cutlass::PipelineState<ReuseSmemC ? StagesC : StagesD>;

  struct SharedStorage {
    struct TensorStorage {
      using CollectiveStorage = cute::conditional_t<not is_source_supported, CollectiveStorageWithoutC,
                                  cute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
      CollectiveStorage collective;

      using FusionStorage = typename FusionCallbacks::SharedStorage;
      FusionStorage thread;
    } tensors;

    struct TensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_C;
      cute::TmaDescriptor smem_tensormap_D;
    } tensormaps;

    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Planar complex kernels have two accumulator copies for the real and imaginary tensors.
  constexpr static int NumAccumulatorMtxs = 1;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideD, StrideD>;

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC{};
    ElementD** ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  struct Params {
    using TensorShapeC = decltype(repeat_like(append<3>(StrideC{}, _1{}), int32_t(0)));
    using TensorShapeD = decltype(repeat_like(append<3>(StrideD{}, _1{}), int32_t(0)));
    using TMA_C = decltype(make_tma_copy(
        CopyOpG2S{},
        make_tensor(
            make_gmem_ptr(static_cast<GmemElementC const*>(nullptr)),
            TensorShapeC{},
            append<3>(InternalStrideC{}, _0{})),
        SmemLayoutStageC{},
        EpilogueTile{},
        _1{}));
    using TMA_D = decltype(make_tma_copy(
        CopyOpS2G{},
        make_tensor(
            make_gmem_ptr(static_cast<GmemElementD*>(nullptr)),
            TensorShapeD{},
            append<3>(InternalStrideD{}, _0{})),
        SmemLayoutStageD{},
        EpilogueTile{},
        _1{}));

    typename FusionCallbacks::Params thread{};
    TMA_C tma_load_c;
    TMA_D tma_store_d;
    cute::TmaDescriptor* tensormaps;
    ElementC const** ptr_C;
    StrideC dC;
    ElementD** ptr_D;
    StrideD dD;
  };

  //
  // Gemm Host Functions
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_M = int32_t(size<0>(CtaTileShape{}));
    auto init_N = int32_t(size<1>(CtaTileShape{}));
    auto init_L = 1;

    InternalStrideC stride_c;
    InternalStrideD stride_d;
    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      stride_c = InternalStrideC{};
      stride_d = InternalStrideD{};
    } 
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNKL = append<4>(problem_shape.get_host_problem_shape(0), 1);
      init_M = get<0>(problem_shape_MNKL);
      init_N = get<1>(problem_shape_MNKL);

      stride_c = args.dC;
      stride_d = args.dD;
    }

    typename Params::TMA_C tma_load_c{};
    if constexpr (is_source_supported) {
      // Tensor pointers will be fixed before the first access
      ElementC const* ptr_C_first_batch = nullptr;
      Tensor tensor_c = make_tensor(ptr_C_first_batch, make_layout(make_shape(init_M,init_N,init_L), append<3>(stride_c, _0{})));
      tma_load_c = make_tma_copy(CopyOpG2S{}, tensor_c, SmemLayoutStageC{}, EpilogueTile{}, _1{});
    }

    typename Params::TMA_D tma_store_d{};
    if constexpr (is_destination_supported) {
      // Tensor pointers will be fixed before the first access
      ElementD* ptr_D_first_batch = nullptr;
      Tensor tensor_d = make_tensor(ptr_D_first_batch, make_layout(make_shape(init_M,init_N,init_L), append<3>(stride_d, _0{})));
      tma_store_d = make_tma_copy(CopyOpS2G{}, tensor_d, SmemLayoutStageD{}, EpilogueTile{}, _1{});
    }

    auto fusion_workspace = static_cast<char*>(workspace);
    auto fusion_workspace_size = round_nearest(FusionCallbacks::get_workspace_size(problem_shape, args.thread), MinTensorMapWorkspaceAlignment);
    auto tma_descriptor_workspace = reinterpret_cast<cute::TmaDescriptor*>(
                                        static_cast<char*>(workspace) + fusion_workspace_size);

    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, fusion_workspace),
      tma_load_c,
      tma_store_d,
      tma_descriptor_workspace,
      args.ptr_C,
      args.dC,
      args.ptr_D,
      args.dD
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    constexpr uint32_t NumInputTensors = cute::is_void_v<ElementC> ? 1 : 2;
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies
    return (NumInputTensors * SizeOfCuTensorMap * sm_count) + (round_nearest(FusionCallbacks::get_workspace_size(problem_shape, args.thread), MinTensorMapWorkspaceAlignment));
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return FusionCallbacks::initialize_workspace(problem_shape, args.thread, workspace, stream, cuda_adapter);
  }

  template <class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shape,
      [[maybe_unused]] Arguments const& args) {
    bool implementable = true;
    bool fusion_implementable = true;

    if (problem_shape.is_host_problem_shape_available()) {
      for (int i = 0; i < problem_shape.groups(); ++i) {
        auto problem_shape_MNKL = append<4>(problem_shape.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;

        if constexpr (is_destination_supported) {
          constexpr int tma_alignment_bits_D = cutlass::detail::get_output_alignment_bits<ElementD>();
          constexpr int min_tma_aligned_elements_D = tma_alignment_bits_D / cutlass::sizeof_bits<ElementD>::value;
          implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_D>(cute::make_shape(M,N,L), InternalStrideD{});
        }

        if constexpr (is_source_supported) {
          constexpr int tma_alignment_bits_C = cutlass::detail::get_input_alignment_bits<ElementC>();
          constexpr int min_tma_aligned_elements_C = tma_alignment_bits_C / cutlass::sizeof_bits<ElementC>::value;
          implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_C>(cute::make_shape(M,N,L), InternalStrideC{});
        }

        fusion_implementable = fusion_implementable && FusionCallbacks::can_implement(problem_shape_MNKL, args.thread);
      }
    }
    else {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Ignoring check to can implement because host problem shape is not available.\n");
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }

    if (!fusion_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum requirements for FusionCallbacks.\n");
    }

    bool beta_implementable = true;

    if (cute::is_void_v<ElementC> || args.ptr_C == nullptr) {
      if constexpr (detail::has_beta<Arguments>::value) {
        beta_implementable = args.thread.beta == 0.0;
      }
      if constexpr (detail::has_beta_ptr<Arguments>::value) {
        beta_implementable = beta_implementable && args.thread.beta_ptr == nullptr;
      }
      if constexpr (detail::has_beta_ptr_array<Arguments>::value) {
        beta_implementable = beta_implementable && args.thread.beta_ptr_array == nullptr;
      }
    }

    if (!beta_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Beta/beta pointer was set, but epilogue is sourceless (void-C).\n");
    }

    return implementable && fusion_implementable && beta_implementable;
  }

  //
  // Static Device Functions
  //

  template<class CtaTileMNK>
  CUTLASS_DEVICE
  static constexpr int
  get_load_pipe_increment(CtaTileMNK const& cta_tile_mnk) {
    // Compute number of epilogue subtiles
    return size<1>(zipped_divide(make_layout(take<0,2>(cta_tile_mnk)), EpilogueTile{}));
  }

  template<class CtaTileMNK>
  CUTLASS_DEVICE
  static constexpr int
  get_store_pipe_increment(CtaTileMNK const& cta_tile_mnk) {
    return get_load_pipe_increment(cta_tile_mnk);
  }

  //
  // Constructor and Data Members
  //
  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params_, TensorStorage& shared_tensors)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread) {}

private:
  Params const& params;
  FusionCallbacks fusion_callbacks;

  //
  // Non-static Device Functions
  //
public:
  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return fusion_callbacks.is_producer_load_needed();
  }

  CUTLASS_DEVICE auto
  load_init(
      Params const& params,
      TensorMapStorage& shared_tensormap,
      int32_t const sm_count,
      int32_t const sm_idx) const {
    // Fetch a copy of tensormaps for the CTA from Params
    constexpr bool IsEpiLoad = true;
    auto load_tensormap = tensormaps_init<IsEpiLoad>(params, shared_tensormap, sm_count, sm_idx);
    return cute::make_tuple(load_tensormap);
  }

  template<
    bool ReuseTmem = false,
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    class TensorMapC
  >
  CUTLASS_DEVICE auto
  load(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaTileMNK cta_tile_mnk,
      CtaCoordMNKL cta_coord_mnkl,
      MmaTileMNK mma_tile_mnk,
      TiledMma tiled_mma,
      TensorStorage& shared_tensors,
      cute::tuple<TensorMapC, bool> load_tensormap_info,
      bool reverse_epi_n = false) {
    using namespace cute;

    // Check to see if tensormaps have been replaced in gmem
    if (get<1>(load_tensormap_info) /* did_batch_change */) {
      tensormaps_fence_acquire<true /* IsEpiLoad */>(get<0>(load_tensormap_info));
    }

    int lane_idx = canonical_lane_idx();
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;

    auto coord_shape = append<3>(make_shape(m_coord, n_coord),Int<0>{});

    // Represent the full source tensor, slice to get the tile this CTA is currently responsible for
    Tensor mC_mn = params.tma_load_c.get_tma_tensor(append<3>(make_shape(M,N),Int<1>{}));              //       (M,N,L)
    Tensor mC = coalesce(mC_mn, take<0,2>(cta_tile_mnk));
    Tensor gC = local_tile(mC, take<0,2>(cta_tile_mnk), coord_shape);                                  // (CTA_M,CTA_N)

    // Apply epilogue subtile, get matching smem tensor
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    Tensor gC_epi = flat_divide(gC, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sC_epi = make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{});           //      (EPI_TILE_M,EPI_TILE_N,PIPE_C)

    // Prepare the thread(b)lock's (G)mem to (S)mem TMA tiled copy (bGS_)
    ThrCopy thrblk_g2s = params.tma_load_c.get_slice(Int<0>{});
    Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);                                    // (TMA,TMA_M,TMA_N,EPI_M,EPI_N)
    Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi);                                    // (TMA,TMA_M,TMA_N,PIPE_C)

    // Get the fusion callbacks for the producer load warp
    auto pld_args = cutlass::epilogue::fusion::detail::ProducerLoadArgs{
                      problem_shape_mnkl,
                      cta_tile_mnk,
                      cta_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      lane_idx
                    };
    auto pld_callbacks = fusion_callbacks.get_producer_load_callbacks(pld_args);
    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();

    // Predication for TMA load (one thread issues TMA load)
    bool issue_tma_load = cute::elect_one_sync();

    // Pre-loop fusion callback entry point
    pld_callbacks.begin();

    CUTLASS_PRAGMA_UNROLL
    for (int iter_n = 0; iter_n < size<3>(gC_epi); ++iter_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter_m = 0; iter_m < size<2>(gC_epi); ++iter_m) {
        int epi_m = iter_m, epi_n = iter_n;
        if constexpr (ReuseTmem) {
          if (reverse_epi_n) {
            epi_n = size<3>(gC_epi) - 1 - iter_n;
          }
        }
        // Acquire the lock for this stage
        constexpr uint16_t mcast_mask = 0;
        uint64_t* tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);
        load_pipeline.producer_acquire(load_pipe_producer_state);

        // Execute the TMA load for C if needed
        if (issue_tma_load && is_C_load_needed) {
          copy(params.tma_load_c.with(get<0>(load_tensormap_info), *tma_barrier, mcast_mask),
              bGS_gC(_,_,_,epi_m,epi_n), bGS_sC(_,_,_,load_pipe_producer_state.index()));
          load_pipeline.producer_expect_transaction(load_pipe_producer_state);
        }

        // Loop fusion callback entry point
        pld_callbacks.step(tma_barrier, epi_m, epi_n, load_pipe_producer_state.count(), issue_tma_load);

        // Commit TMA loads for this stage and release the lock
        load_pipeline.producer_commit(load_pipe_producer_state);
        ++load_pipe_producer_state;
      }
    }

    // Post-loop fusion callback entry point
    pld_callbacks.end();

    return load_pipe_producer_state;
  }

  CUTLASS_DEVICE void
  load_tail(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      [[maybe_unused]] StorePipelineState store_pipe_producer_state) {
    load_pipeline.producer_tail(load_pipe_producer_state);
  }

  CUTLASS_DEVICE auto
  store_init(
      Params const& params,
      TensorMapStorage& shared_tensormap,
      int32_t const sm_count,
      int32_t const sm_idx) const {
    // Fetch a copy of tensormaps for the CTA from Params
    constexpr bool IsEpiLoad = false;
    cute::TmaDescriptor* store_tensormap = nullptr;
    int thread_idx = threadIdx.x % ThreadCount;
    int warp_idx = thread_idx / NumThreadsPerWarp;
    // Only the first epilogue warp needs to perform TMA related operations
    if (warp_idx == 0) {
      store_tensormap = tensormaps_init<IsEpiLoad>(params, shared_tensormap, sm_count, sm_idx);
    }
    return cute::make_tuple(store_tensormap);
  }

  template<
    bool ReuseTmem = false,
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    class AccEngine,
    class AccLayout,
    class TensorMapD
  >
  CUTLASS_DEVICE auto
  store(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState acc_pipe_consumer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaTileMNK cta_tile_mnk,
      CtaCoordMNKL cta_coord_mnkl,
      MmaTileMNK mma_tile_mnk,
      TiledMma tiled_mma,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      TensorStorage& shared_tensors,
      cute::tuple<TensorMapD, bool> store_tensormap_info
      ) {
    using namespace cute;
    using ElementAccumulator = typename AccEngine::value_type;
    using ElementCompute_ = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::ElementCompute;
    using ElementCompute = cute::conditional_t<cute::is_void_v<ElementCompute_>,ElementAccumulator,ElementCompute_>;

    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");
    static_assert(rank(accumulators) == 3, "Accumulators must be MMA-partitioned: [MMA, MMA_M, MMA_N]");
    static_assert(size<1>(accumulators) == 1 && size<2>(accumulators) == 1, "TiledMMA must match partitioned ShapeMN");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(rank(CtaCoordMNKL{}) == 4, "CoordMNKL must be rank 4");

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
    int thread_idx = threadIdx.x % ThreadCount;
    int warp_idx = thread_idx / NumThreadsPerWarp;
    [[maybe_unused]] int lane_idx = thread_idx % NumThreadsPerWarp;

    // Check to see if tensormaps have been replaced in gmem
    // Only the first epilogue warp needs to perform TMA related operations
    if (get<1>(store_tensormap_info) /* did_batch_change */ && warp_idx == 0) {
      tensormaps_fence_acquire<false /* IsEpiLoad */>(get<0>(store_tensormap_info));
    }

    auto coord_shape = append<3>(make_shape(m_coord, n_coord),Int<0>{});

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    Tensor mD_mn = params.tma_store_d.get_tma_tensor(append<3>(make_shape(M,N),Int<1>{}));             //       (M,N,L)
    Tensor mD = coalesce(mD_mn, take<0,2>(cta_tile_mnk));
    Tensor gD = local_tile(mD, take<0,2>(cta_tile_mnk), coord_shape);                                  // (CTA_M,CTA_N)

    Tensor tAcc = accumulators(make_coord(_,_),_0{},_0{});                                             // (CTA_M,CTA_N)

    // Apply epilogue subtiling
    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                         // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor gD_epi   = flat_divide(  gD, EpilogueTile{});                         // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    Tensor sC_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    Tensor sD_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sD), SmemLayoutD{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_D)

    // (t)hread-partition for (t)mem to (r)egister copy (tTR_)
    TiledCopy tiled_t2r = make_tmem_copy(CopyOpT2R{}, tAcc_epi(_,_,_0{},_0{}));
    ThrCopy thread_t2r = tiled_t2r.get_slice(thread_idx);
    Tensor tTR_tAcc = thread_t2r.partition_S(tAcc_epi);                                // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
    Tensor tTR_sD   = thread_t2r.partition_D(sD_epi(_,_,_0{}));                        // (T2R,T2R_M,T2R_N)

    // Allocate D and accumulator registers
    // Does directly store the visitor into smem.
    constexpr bool IsDirectR2S = cute::is_same_v<CopyOpR2R, AutoVectorizingCopyWithAssumedAlignment<128>>;
    using RegisterElementD = cute::conditional_t<!IsDirectR2S, ElementCompute, SmemElementD>;
    Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tTR_sD));                              // (T2R,T2R_M,T2R_N)
    Tensor tTR_rD   = make_tensor<RegisterElementD>(shape(tTR_sD));                                // (T2R,T2R_M,T2R_N)

    // Vectorized fragment view
    constexpr int FragmentSize = DispatchPolicy::FragmentSize;
    Tensor tTR_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc));               // (EPI_V)
    Tensor tTR_rD_frg   = recast<Array<RegisterElementD, FragmentSize>>(coalesce(tTR_rD));                   // (EPI_V)
    CUTE_STATIC_ASSERT(size(tTR_rAcc) % DispatchPolicy::FragmentSize == 0, "Fragment size does not vectorize properly");

    // (t)hread-partition for (s)mem to (r)egister copy (tSR_)
    TiledCopy tiled_s2r = make_tiled_copy_D(Copy_Atom<CopyOpS2R, SmemElementC>{}, tiled_t2r);
    ThrCopy thread_s2r = tiled_s2r.get_slice(thread_idx);
    Tensor tSR_sC        = thread_s2r.partition_S(sC_epi);                                  // (S2R,S2R_M,S2R_N,PIPE_C)
    Layout tSR_rC_layout = thread_s2r.retile_D(tTR_rD).layout();                            // (S2R,S2R_M,S2R_N)

    // Allocate C registers
    // If C smem load is a non-vectorized dst(i) = src(i) then we can allocate C registers directly in the compute type
    // to eliminate some redundant pack+unpack instruction sequences for sub-word types
    constexpr bool IsDirectS2R = cute::is_same_v<CopyOpS2R, AutoVectorizingCopyWithAssumedAlignment<128>>
                                && decltype(max_common_vector(tSR_rC_layout, tSR_sC.layout()))::value <= 1;
    using RegisterElementC = cute::conditional_t<IsDirectS2R, ElementCompute, SmemElementC>;
    Tensor tTR_rC = make_tensor<RegisterElementC>(shape(tTR_sD));                                  // (T2R,T2R_M,T2R_N)
    Tensor tSR_rC = thread_s2r.retile_D(tTR_rC);                                                   // (S2R,S2R_M,S2R_N)

    // (t)hread-partition for (r)egister to (r)egister copy (tRR_)
    TiledCopy tiled_r2r = make_tiled_copy_D(Copy_Atom<CopyOpR2R, RegisterElementD>{}, tiled_t2r);
    ThrCopy thread_r2r = tiled_r2r.get_slice(thread_idx);
    Tensor tRR_rD_src = thread_r2r.retile_S(tTR_rD);                                   // (R2R,R2R_M,R2R_N,EPI_M,EPI_N)
    Tensor tRR_rD_dst = thread_r2r.retile_D(tTR_rD);                                   // (R2R,R2R_M,R2R_N,EPI_M,EPI_N)

    // (t)hread-partition for (r)egister to (s)mem copy (tRS_)
    TiledCopy tiled_r2s = make_tiled_copy_D(Copy_Atom<CopyOpR2S, SmemElementD>{}, tiled_r2r);
    ThrCopy thread_r2s = tiled_r2s.get_slice(thread_idx);
    Tensor tRS_sD = thread_r2s.partition_D(sD_epi);                                         // (R2S,R2S_M,R2S_N,PIPE_D)
    Tensor tRS_rD = [&]() {
      if constexpr (!IsDirectR2S) {
        return make_tensor<SmemElementD>(shape(tRS_sD(_,_,_,_0{})));
      }
      else{
        return thread_r2s.retile_S(tTR_rD);                                                 // (R2S,R2S_M,R2S_N)
      }
    }();

    Tensor tRR_rD_dst_frg = recast<Array<RegisterElementD, FragmentSize>>(coalesce(tRR_rD_dst));
    Tensor tRS_rD_frg     = recast<Array<SmemElementD, FragmentSize>>(coalesce(tRS_rD));

    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
    ThrCopy thrblk_s2g = params.tma_store_d.get_slice(Int<0>{});
    Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);                                    // (S2G,S2G_M,S2G_N,PIPE_D)
    Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(cta_tile_mnk), make_coord(m_coord, n_coord));        // (CTA_M,CTA_N)
    Tensor tTR_cD_mn = thread_t2r.partition_D(flat_divide(cD_mn, EpilogueTile{}));     // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
    // Relative coordinate tensors (static)
    Tensor cD = make_coord_tensor(cD_mn.layout());                                                  // (CTA_M,CTA_N)
    Tensor tTR_cD = make_coord_tensor(tTR_cD_mn.layout());                          // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
    // Subtract the global "bottom right" corner from the local "top left" corner to get the max relative coordinate
    auto residue_cD = make_coord(M,N) - cD_mn(_0{});                                                           // (m,n)
    auto residue_tTR_cD = make_coord(M,N) - tTR_cD_mn(_0{});                                                   // (m,n)

    // Get the fusion callbacks for the consumer store warps
    constexpr bool RefSrc = false; // Register tensors reference T2R copy dst layout
    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      cta_tile_mnk,
                      cta_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      tiled_t2r,
                      cD,
                      residue_cD,
                      tTR_cD,
                      residue_tTR_cD,
                      tTR_rC,
                      thread_idx
                    };

    // Thread synchronizer for previously issued waits or fences
    // to ensure visibility of smem reads/writes to threads or TMA unit
    auto synchronize = [] () CUTLASS_LAMBDA_FUNC_INLINE { cutlass::arch::NamedBarrier::sync(ThreadCount, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };

    // Predication for sub-128 thread T2R tiled copy
    Layout tmem_warp_layout = typename decltype(make_tmem_warp_partitioner(tAcc_epi(_,_,0,0)))::TiledLayout_TV{};
    constexpr bool predicate_tmem_load = size(tmem_warp_layout) != cosize(tmem_warp_layout);
    bool issue_tmem_load = true;

    // If tmem doesn't have enough capacity to support double buffering, a portion of tmem (a column of epilogue tiles)
    // is overlapped between 2 pseudo-buffers. The shared tmem portion corresponds to the last epilogue tile column of
    // tmem accumulator buffer 0, and the first epilogue tile column of tmem accumulator 1.
    // Thus, whenever we are processing tmem accumulator buffer 0, we process the epilogue tiles with reversed column order.
    // Once the last epilogue tile column is loaded from tmem, the acc_pipeline is released.
    // Then, the next accumulation stage for buffer 1 can start.
    [[maybe_unused]] bool reverse_epi_n = ReuseTmem && acc_pipe_consumer_state.phase() == 0;
    static_assert(not (ReuseTmem && AccumulatorPipeline::Stages != 1), "Tmem reuse requires 1 accumulator stage");

    // Predication for TMA store (a single thread from one warp issues TMA store)
    bool issue_tma_store = (warp_idx == 0) && cute::elect_one_sync();

    // In the reuse smem configuration we have StagesC smem buffers and at most StagesD committed TMA stores in flight.
    // The TMA store pipeline producer acquire returns when at most StagesD-1 committed stores are in-flight, so we can
    // only guarantee store completion after StagesD iterations, then we can begin issuing releases on the smem buffer locks.
    // store_pipe_producer_state tracks the acquire and load_pipe_consumer_state tracks the release, in circular buffer fashion.
    // If TMA store supported async transaction mbarriers we would not need this synchronous release behavior.
    LoadPipelineState load_wait_state = load_pipe_consumer_state;
    if constexpr (ReuseSmemC) {
      load_wait_state = store_pipe_producer_state;
      load_wait_state.phase_ ^= 1;
    }

    // We can delay issue of TMA store by one iteration to achieve better interleaving of non-TMA instructions
    // Sync requirements of smem reuse may preclude this optimization
    // Delayed stores cause delayed stage releases which causes deadlock when StagesC == StagesD
    [[maybe_unused]] int epi_m_prev = 0;
    [[maybe_unused]] int epi_n_prev = 0;
    static_assert(not (DelayTmaStore and ReuseSmemC and StagesC <= StagesD), "This TMA epilogue configuration will deadlock");

    // The Epilogue Loop
    auto epi_loop_fn = [&] (auto& cst_callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
      bool is_producer_load_needed = fusion_callbacks.is_producer_load_needed();
      bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();

      // The TMA store sequence for one epilogue loop iteration
      auto tma_store_fn = [&] (int epi_m, int epi_n) CUTLASS_LAMBDA_FUNC_INLINE {
        // Write the tile from smem to gmem with TMA
        cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
        synchronize(); // ensure all threads have issued their async fence

        if constexpr (is_destination_supported) {
          if (issue_tma_store) {
            copy(params.tma_store_d.with(get<0>(store_tensormap_info)), bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
          }
        }
  
        // Post async fence, pre TMA commit callback entry point
        cst_callbacks.tma_store(epi_m, epi_n, store_pipe_producer_state.count(), issue_tma_store);
  
        // Commit the TMA stores for this stage
        if (issue_tma_store) {
          store_pipeline.producer_commit(store_pipe_producer_state);
        }
        ++store_pipe_producer_state;
  
        // Wait for the next smem buffer to be available
        if (issue_tma_store) {
          store_pipeline.producer_acquire(store_pipe_producer_state);
        }
        synchronize();
  
        if constexpr (ReuseSmemC) {
          // producer_acquire returns when at most StagesD-1 committed stores are pending
          bool store_finished = store_pipe_producer_state.count() > StorePipeline::UnacquiredStages;
          // Let dma warp know earliest smem buffer is consumed and empty after StagesD producer commits
          if (store_finished) {
            if (is_producer_load_needed) {
              load_pipeline.consumer_release(load_pipe_consumer_state);
            }
            ++load_pipe_consumer_state;
          }
        }
      }; // tma_store_fn

      // Begin the wait for the producer load results
      ConsumerToken load_wait_token{BarrierStatus::WaitDone};
      if (is_producer_load_needed) {
        load_wait_token = load_pipeline.consumer_try_wait(load_wait_state);
      }
      // Begin the wait for the accumulator results
      ConsumerToken acc_wait_token = acc_pipeline.consumer_try_wait(acc_pipe_consumer_state);

      cst_callbacks.begin();
      if (cst_callbacks.begin_sync_needed()) {
        synchronize();
      }
      // For each epilogue subtile within the CTA tile
      constexpr int NumEpiSubtilesN = CUTE_STATIC_V(size<3>(gD_epi));
      constexpr int NumEpiSubtilesM = CUTE_STATIC_V(size<2>(gD_epi));
      #pragma unroll(UnrollEpiLoop ? NumEpiSubtilesN : 1)
      for (int iter_n = 0; iter_n < NumEpiSubtilesN; ++iter_n) {
        #pragma unroll(UnrollEpiLoop ? NumEpiSubtilesM : 1)
        for (int iter_m = 0; iter_m < NumEpiSubtilesM; ++iter_m) {
          int epi_m = iter_m, epi_n = iter_n;
          bool is_first_iteration = iter_m == 0 && iter_n == 0;
          bool is_last_iteration = iter_m == size<2>(gD_epi)-1 && iter_n == size<3>(gD_epi)-1;
          bool do_acc_release = is_last_iteration;

          // Reverse subtile order for tmem reuse if necessary
          if constexpr (ReuseTmem) {
            if (reverse_epi_n) {
              epi_n = size<3>(gD_epi) - 1 - iter_n;
            }
            do_acc_release = iter_m == size<2>(gD_epi)-1 && iter_n == 0;
          }

          cst_callbacks.begin_loop(epi_m, epi_n);

          if (is_producer_load_needed) {
            // Wait for the producer load to fill smem
            load_pipeline.consumer_wait(load_wait_state, load_wait_token);

            if (is_C_load_needed) {
              // Copy source tile from smem to register
              copy(tiled_s2r, tSR_sC(_,_,_,load_wait_state.index()), tSR_rC);
              // Ensure smem loads are complete before reusing smem for mixed types/layouts
              if constexpr (ReuseSmemC && not (SmemLayoutC{} == SmemLayoutD{})) {
                synchronize();
              }
            }
          }

          // First loop fusion callback entry point
          cst_callbacks.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed);

          if (is_producer_load_needed) {
            // Let producer load warp know smem buffers are consumed and empty
            if constexpr (not ReuseSmemC) {
              cutlass::arch::fence_view_async_shared();
              load_pipeline.consumer_release(load_pipe_consumer_state);
              ++load_pipe_consumer_state;
            }
            ++load_wait_state;
          }

          if (is_first_iteration) {
            // Wait for mma warp to fill tmem buffer with accumulator results
            acc_pipeline.consumer_wait(acc_pipe_consumer_state, acc_wait_token);
          }

          // The current tile in tmem
          Tensor tTR_tAcc_mn = tTR_tAcc(_,_,_,epi_m,epi_n);

          // Compute tmem load predication if necessary
          if constexpr (predicate_tmem_load) {
            // Issue tmem load if this tile's tmem subpartition is accessible by this warp
            int subpart_idx = (tTR_tAcc_mn.data().dp_ / 32) % 4;
            issue_tmem_load = warp_idx == subpart_idx;
          }

          // Copy accumulator tile from tmem to register
          if (issue_tmem_load) {
            copy(tiled_t2r, tTR_tAcc_mn, tTR_rAcc);
          }

          // After the last tmem load, signal that tmem buffer is consumed and empty
          if (do_acc_release) {
            cutlass::arch::fence_view_async_tmem_load();
            acc_pipeline.consumer_release(acc_pipe_consumer_state);
            ++acc_pipe_consumer_state;
          }

          // Vectorized fragment loop with visitor callback entry point
          CUTLASS_PRAGMA_UNROLL
          for (int epi_v = 0; epi_v < size(tTR_rD_frg); ++epi_v) {
            tTR_rD_frg(epi_v) = cst_callbacks.visit(tTR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
          }

          // The latest we can delay the TMA store is right before the smem store of the next iteration
          // since the current TMA store needs to be committed before we can acquire the next smem buffer
          if constexpr (DelayTmaStore) {
            // Issue TMA stores for the previous subtile
            if (not is_first_iteration) {
              tma_store_fn(epi_m_prev, epi_n_prev);
            }
            epi_m_prev = epi_m;
            epi_n_prev = epi_n;
          }
          
          if constexpr (!IsDirectR2S) {
            // At present, only FP4 col output with scalefactor generation fusion would go into these branch
            copy(tiled_r2r, tRR_rD_src, tRR_rD_dst);
          }
          tRS_rD_frg(_0{}) = cutlass::NumericArrayConverter<SmemElementD, RegisterElementD, FragmentSize>{}(tRR_rD_dst_frg(_0{}));

          // Smem reduction callback entry point using current store buffer for workspace
          Tensor reduction_buffer = make_tensor(raw_pointer_cast(sD_epi(_,_,store_pipe_producer_state.index()).data()),
                                                make_layout(stride<2>(get_nonswizzle_portion(SmemLayoutD{})), _1{}));
          cst_callbacks.reduce(reduction_buffer, synchronize, epi_m, epi_n, is_last_iteration, tRS_rD_frg);

          // Copy output tile from register to smem
          bool issue_smem_store = issue_tmem_load;
          if constexpr (is_destination_supported) {
            if (issue_smem_store) {
              copy(tiled_r2s, tRS_rD, tRS_sD(_,_,_,store_pipe_producer_state.index()));
            }
          }

          // Post reduction, pre TMA store callback entry point
          cst_callbacks.postreduce(epi_m, epi_n, store_pipe_producer_state.count(), issue_smem_store);

          if constexpr (not DelayTmaStore) {
            // Issue TMA stores for this subtile
            tma_store_fn(epi_m, epi_n);
          }

          cst_callbacks.end_loop(epi_m, epi_n);

          if (is_producer_load_needed) {
            // Begin the wait for the next subtile producer load
            load_wait_token = load_pipeline.consumer_try_wait(load_wait_state, is_last_iteration);
          }
        } // for epi_m
      } // for epi_n

      if constexpr (DelayTmaStore) {
        // Issue TMA stores for the last subtile
        tma_store_fn(epi_m_prev, epi_n_prev);
      }

      cst_callbacks.end();
    };

    //
    // BEGIN EPILOGUE
    //
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);
    epi_loop_fn(cst_callbacks);
    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state, acc_pipe_consumer_state);
  }

  // API with Global Accumulator in registers for FastFP32 (emulated MMA) kernels.
  // The accumulator in TMEM periodically loaded into the registers so that the MMA can clear out the TMEM accumulator
  // values for better accuracy. This epilogue accepts the accumulator in registers and take TiledCopy for the
  // TMEM->Reg as a parameter to be used in partitioning GMEM tensors C and D.
  template<
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    class AccEngine,
    class AccLayout,
    class TiledCopyT2R,
    class TensorMapD
  >
  CUTLASS_DEVICE auto
  store(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaTileMNK cta_tile_mnk,
      CtaCoordMNKL cta_coord_mnkl,
      MmaTileMNK mma_tile_mnk,
      TiledMma tiled_mma,
      cute::Tensor<AccEngine, AccLayout>& tTR_rAcc,                                     // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
      TensorStorage& shared_tensors,
      TensorMapD store_tensormap,
      TiledCopyT2R tiled_t2r
      ) {
    using namespace cute;
    using ElementAccumulator = typename AccEngine::value_type;
    using ElementCompute_ = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::ElementCompute;
    using ElementCompute = cute::conditional_t<cute::is_void_v<ElementCompute_>,ElementAccumulator,ElementCompute_>;

    static_assert(is_rmem<AccEngine>::value, "Accumulator must be Register resident.");
    static_assert(rank(AccLayout{}) == 5, "Accumulators must be copy-partitioned:  (T2R,T2R_M,T2R_N,EPI_M,EPI_N)");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(rank(CtaCoordMNKL{}) == 4, "CoordMNKL must be rank 4");

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
    int thread_idx = threadIdx.x % ThreadCount;
    int warp_idx = thread_idx / NumThreadsPerWarp;
    [[maybe_unused]] int lane_idx = thread_idx % NumThreadsPerWarp;

    auto coord_shape = append<3>(make_shape(m_coord, n_coord),Int<0>{});

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    Tensor mD_mn = params.tma_store_d.get_tma_tensor(append<3>(make_shape(M,N),Int<1>{}));             //       (M,N,L)
    Tensor mD = coalesce(mD_mn, take<0,2>(cta_tile_mnk));
    Tensor gD = local_tile(mD, take<0,2>(cta_tile_mnk), coord_shape);                                  // (CTA_M,CTA_N)

    // Apply epilogue subtiling
    Tensor gD_epi = flat_divide(  gD, EpilogueTile{});                           // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    Tensor sC_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    Tensor sD_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sD), SmemLayoutD{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_D)

    // (t)hread-partition for (t)mem to (r)egister copy (tTR_)
    ThrCopy thread_t2r = tiled_t2r.get_slice(thread_idx);
    Tensor tTR_sD = thread_t2r.partition_D(sD_epi(_,_,_0{}));                                      // (T2R,T2R_M,T2R_N)

    // Allocate D and accumulator registers
    Tensor tTR_rD = make_tensor<SmemElementD>(shape(tTR_sD));                                      // (T2R,T2R_M,T2R_N)

    // Vectorized fragment view
    constexpr int FragmentSize = DispatchPolicy::FragmentSize;
    Tensor tTR_rD_frg = recast<Array<SmemElementD, FragmentSize>>(coalesce(tTR_rD));                         // (EPI_V)

    // (t)hread-partition for (s)mem to (r)egister copy (tSR_)
    TiledCopy tiled_s2r  = make_tiled_copy_D(Copy_Atom<CopyOpS2R, SmemElementC>{}, tiled_t2r);
    ThrCopy thread_s2r   = tiled_s2r.get_slice(thread_idx);
    Tensor tSR_sC        = thread_s2r.partition_S(sC_epi);                                  // (S2R,S2R_M,S2R_N,PIPE_C)
    Layout tSR_rC_layout = thread_s2r.retile_D(tTR_rD).layout();                                   // (S2R,S2R_M,S2R_N)

    // Allocate C registers
    // If C smem load is a non-vectorized dst(i) = src(i) then we can allocate C registers directly in the compute type
    // to eliminate some redundant pack+unpack instruction sequences for sub-word types
    constexpr bool IsDirectS2R = cute::is_same_v<CopyOpS2R, AutoVectorizingCopyWithAssumedAlignment<128>>
                                && decltype(max_common_vector(tSR_rC_layout, tSR_sC.layout()))::value <= 1;
    using RegisterElementC = cute::conditional_t<IsDirectS2R, ElementCompute, SmemElementC>;
    Tensor tTR_rC = make_tensor<RegisterElementC>(shape(tTR_sD));                                  // (T2R,T2R_M,T2R_N)
    Tensor tSR_rC = thread_s2r.retile_D(tTR_rC);                                                   // (S2R,S2R_M,S2R_N)

    // (t)hread-partition for (r)egister to (s)mem copy (tRS_)
    TiledCopy tiled_r2s = make_tiled_copy_D(Copy_Atom<CopyOpR2S,SmemElementD>{}, tiled_t2r);
    ThrCopy thread_r2s = tiled_r2s.get_slice(thread_idx);
    Tensor tRS_rD = thread_r2s.retile_S(tTR_rD);                                                   // (R2S,R2S_M,R2S_N)
    Tensor tRS_sD = thread_r2s.partition_D(sD_epi);                                         // (R2S,R2S_M,R2S_N,PIPE_D)

    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
    ThrCopy thrblk_s2g = params.tma_store_d.get_slice(Int<0>{});
    Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);                                         // (S2G,S2G_M,S2G_N,PIPE_D)
    Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(cta_tile_mnk), make_coord(m_coord, n_coord));          // (CTA_M,CTA_N)
    Tensor tTR_cD_mn = thread_t2r.partition_D(flat_divide(cD_mn, EpilogueTile{}));     // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
    // Relative coordinate tensors (static)
    Tensor cD = make_coord_tensor(cD_mn.layout());                                                  // (CTA_M,CTA_N)
    Tensor tTR_cD = make_coord_tensor(tTR_cD_mn.layout());                          // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
    // Subtract the global "bottom right" corner from the local "top left" corner to get the max relative coordinate
    auto residue_cD = make_coord(M,N) - cD_mn(_0{});                                                           // (m,n)
    auto residue_tTR_cD = make_coord(M,N) - tTR_cD_mn(_0{});                                                   // (m,n)

    // Get the fusion callbacks for the consumer store warps
    constexpr bool RefSrc = false; // Register tensors reference T2R copy dst layout
    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      cta_tile_mnk,
                      cta_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      tiled_t2r,
                      cD,
                      residue_cD,
                      tTR_cD,
                      residue_tTR_cD,
                      tTR_rC,
                      thread_idx
        };

    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);
    bool is_producer_load_needed = fusion_callbacks.is_producer_load_needed();
    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();

    // Thread synchronizer for previously issued waits or fences
    // to ensure visibility of smem reads/writes to threads or TMA unit
    auto synchronize = [] () { cutlass::arch::NamedBarrier::sync(ThreadCount, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };

    // Predication for TMA store (one warp issues TMA store)
    bool issue_tma_store = warp_idx == 0;

    // In the reuse smem configuration we have StagesC smem buffers and at most StagesD committed TMA stores in flight.
    // The TMA store pipeline producer acquire returns when at most StagesD-1 committed stores are in-flight, so we can
    // only guarantee store completion after StagesD iterations, then we can begin issuing releases on the smem buffer locks.
    // store_pipe_producer_state tracks the acquire and load_pipe_consumer_state tracks the release, in circular buffer fashion.
    // If TMA store supported async transaction mbarriers we would not need this synchronous release behavior.
    LoadPipelineState load_wait_state = load_pipe_consumer_state;
    if constexpr (ReuseSmemC) {
      load_wait_state = store_pipe_producer_state;
      load_wait_state.phase_ ^= 1;
    }

    // We can delay issue of TMA store by one iteration to achieve better interleaving of non-TMA instructions
    // Sync requirements of smem reuse may preclude this optimization
    // Delayed stores cause delayed stage releases which causes deadlock when StagesC == StagesD
    int epi_m_prev = 0, epi_n_prev = 0;
    static_assert(not (DelayTmaStore and ReuseSmemC and StagesC <= StagesD), "This TMA epilogue configuration will deadlock");

    // The TMA store sequence for one subtile iteration
    auto tma_store_fn = [&] (int epi_m, int epi_n) {
      // Write the tile from smem to gmem with TMA
      cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
      synchronize(); // ensure all threads have issued their async fence
      if (issue_tma_store) {
        copy(params.tma_store_d.with(store_tensormap), bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
      }

      // Post async fence, pre TMA commit callback entry point
      cst_callbacks.tma_store(epi_m, epi_n, store_pipe_producer_state.count(), issue_tma_store);

      // Commit the TMA stores for this stage
      if (issue_tma_store) {
        store_pipeline.producer_commit(store_pipe_producer_state);
      }
      ++store_pipe_producer_state;

      // Wait for the next smem buffer to be available
      if (issue_tma_store) {
        store_pipeline.producer_acquire(store_pipe_producer_state);
      }
      synchronize();

      if constexpr (ReuseSmemC) {
        // producer_acquire returns when at most StagesD-1 committed stores are pending
        bool store_finished = store_pipe_producer_state.count() > StorePipeline::UnacquiredStages;
        // Let dma warp know earliest smem buffer is consumed and empty after StagesD producer commits
        if (store_finished) {
          if (is_producer_load_needed) {
            load_pipeline.consumer_release(load_pipe_consumer_state);
          }
          ++load_pipe_consumer_state;
        }
      }
    };

    //
    // BEGIN EPILOGUE
    //

    // Begin the wait for the producer load results
    ConsumerToken load_wait_token{BarrierStatus::WaitDone};
    if (is_producer_load_needed) {
      load_wait_token = load_pipeline.consumer_try_wait(load_wait_state);
    }

    cst_callbacks.begin();
    if (cst_callbacks.begin_sync_needed()) {
      synchronize();
    }

    // For each epilogue subtile within the CTA tile
    constexpr int NumEpiSubtilesN = CUTE_STATIC_V(size<3>(gD_epi));
    constexpr int NumEpiSubtilesM = CUTE_STATIC_V(size<2>(gD_epi));
    #pragma unroll(UnrollEpiLoop ? NumEpiSubtilesN : 1)
    for (int iter_n = 0; iter_n < NumEpiSubtilesN; ++iter_n) {
      #pragma unroll(UnrollEpiLoop ? NumEpiSubtilesM : 1)
      for (int iter_m = 0; iter_m < NumEpiSubtilesM; ++iter_m) {
        int epi_m = iter_m, epi_n = iter_n;
        bool is_first_iteration = iter_m == 0 && iter_n == 0;
        bool is_last_iteration = iter_m == size<2>(gD_epi)-1 && iter_n == size<3>(gD_epi)-1;

        cst_callbacks.begin_loop(epi_m, epi_n);

        if (is_producer_load_needed) {
          // Wait for the producer load to fill smem
          load_pipeline.consumer_wait(load_wait_state, load_wait_token);

          if (is_C_load_needed) {
            // Copy source tile from smem to register
            copy(tiled_s2r, tSR_sC(_,_,_,load_wait_state.index()), tSR_rC);
            // Ensure smem loads are complete before reusing smem for mixed types/layouts
            if constexpr (ReuseSmemC && not (SmemLayoutC{} == SmemLayoutD{})) {
              synchronize();
            }
          }
        }

        // First loop fusion callback entry point
        cst_callbacks.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed);

        if (is_producer_load_needed) {
          // Let producer load warp know smem buffers are consumed and empty
          if constexpr (not ReuseSmemC) {
            cutlass::arch::fence_view_async_shared();
            load_pipeline.consumer_release(load_pipe_consumer_state);
            ++load_pipe_consumer_state;
          }
          ++load_wait_state;
        }

        bool issue_smem_store = true;
        Tensor tTR_rAcc_epi_tile = tTR_rAcc(_,_,_,epi_m,epi_n);
        Tensor tTR_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc_epi_tile));     // (EPI_V)        

        // Vectorized fragment loop with visitor callback entry point
        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size(tTR_rD_frg); ++epi_v) {
          tTR_rD_frg(epi_v) = cst_callbacks.visit(tTR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
        }

        // The latest we can delay the TMA store is right before the smem store of the next iteration
        // since the current TMA store needs to be committed before we can acquire the next smem buffer
        if constexpr (DelayTmaStore) {
          // Issue TMA stores for the previous subtile
          if (not is_first_iteration) {
            tma_store_fn(epi_m_prev, epi_n_prev);
          }
          epi_m_prev = epi_m;
          epi_n_prev = epi_n;
        }

        // Smem reduction callback entry point using current store buffer for workspace
        Tensor reduction_buffer = make_tensor(raw_pointer_cast(sD_epi(_,_,store_pipe_producer_state.index()).data()),
                                              make_layout(stride<2>(get_nonswizzle_portion(SmemLayoutD{})), _1{}));
        cst_callbacks.reduce(reduction_buffer, synchronize, epi_m, epi_n, is_last_iteration, tTR_rD_frg);

        // Copy output tile from register to smem
        if (issue_smem_store) {
          copy(tiled_r2s, tRS_rD, tRS_sD(_,_,_,store_pipe_producer_state.index()));
        }

        // Post reduction, pre TMA store callback entry point
        cst_callbacks.postreduce(epi_m, epi_n, store_pipe_producer_state.count(), issue_smem_store);

        if constexpr (not DelayTmaStore) {
          // Issue TMA stores for this subtile
          tma_store_fn(epi_m, epi_n);
        }

        cst_callbacks.end_loop(epi_m, epi_n);

        if (is_producer_load_needed) {
          // Begin the wait for the next subtile producer load
          load_wait_token = load_pipeline.consumer_try_wait(load_wait_state, is_last_iteration);
        }
      } // for epi_m
    } // for epi_n

    if constexpr (DelayTmaStore) {
      // Issue TMA stores for the last subtile
      tma_store_fn(epi_m_prev, epi_n_prev);
    }

    cst_callbacks.end();

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  template <class CtaTileMNK>
  CUTLASS_DEVICE void
  store_tail(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      CtaTileMNK cta_tile_mnk) {
    if constexpr (ReuseSmemC) {
      if (fusion_callbacks.is_producer_load_needed()) {
        // wait for all TMA stores to complete
        store_pipeline.producer_tail(store_pipe_producer_state);

        // Issue releases on up to StagesD-1 previously issued TMA stores
        constexpr int release_stages = cute::min(StorePipeline::UnacquiredStages, get_load_pipe_increment(cta_tile_mnk));
        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < release_stages; ++stage) {
          load_pipeline.consumer_release(load_pipe_consumer_state);
          ++load_pipe_consumer_state;
        }
      }
    }
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //

  template <bool IsLoad>
  CUTLASS_DEVICE auto
  tensormaps_init(Params const& params,
      TensorMapStorage& shared_tensormap,
      int32_t const sm_count,
      int32_t const sm_idx) const {
    cute::TmaDescriptor* tma_desc = nullptr;
    cute::TmaDescriptor* gmem_tensormap = params.tensormaps;
    if constexpr (IsLoad) {
      if (is_source_supported) {
        tma_desc = &gmem_tensormap[sm_idx];
        if (cute::elect_one_sync()) {
          // Bringing tensormaps from params to smem for modification later
          Tensor pC_tensormap = make_tensor(params.tma_load_c.get_tma_descriptor(), Int<1>{}, Int<1>{});
          Tensor sC_tensormap = make_tensor(make_smem_ptr(&shared_tensormap.smem_tensormap_C), Int<1>{}, Int<1>{});
          copy(recast<uint128_t>(pC_tensormap), recast<uint128_t>(sC_tensormap));
        }
        __syncwarp();
      }
    } else if constexpr (is_destination_supported) {
      int const offset_Ddesc = cute::is_void_v<ElementC> ? 0 : sm_count;
      tma_desc = &gmem_tensormap[sm_idx + offset_Ddesc];
      if (cute::elect_one_sync()) {
        // Bringing tensormaps from params to smem for modification later
        Tensor pD_tensormap = make_tensor(params.tma_store_d.get_tma_descriptor(), Int<1>{}, Int<1>{});
        Tensor sD_tensormap = make_tensor(make_smem_ptr(&shared_tensormap.smem_tensormap_D), Int<1>{}, Int<1>{});
        copy(recast<uint128_t>(pD_tensormap), recast<uint128_t>(sD_tensormap));
      }
      __syncwarp();
    }

    return tma_desc;
  }

  // Replace address for the global tensor (to be done by single thread)
  template <bool IsLoad>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address(
      TensorMapStorage& shared_tensormap,
      Params const& params,
      int32_t next_batch) {
    // Replacing global_address for the next batch
    if constexpr (IsLoad) {
      if constexpr (is_source_supported) {
        if (params.ptr_C != nullptr) {
          cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormap.smem_tensormap_C,
                                                          params.ptr_C[next_batch]);
        }
      }
    } else if constexpr (is_destination_supported) {
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormap.smem_tensormap_D,
                                                      params.ptr_D[next_batch]);
    }
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <bool IsLoad, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride = {0,0,0,0,0};

    if constexpr (IsLoad) {
      if constexpr (is_source_supported) {
        if (params.dC != nullptr) {
          ElementC const* ptr_C = nullptr;
          Tensor tensor_c = make_tensor(ptr_C, make_layout(make_shape(M,N,Int<1>{}), params.dC[next_group]));

          cute::detail::fill_tma_gmem_shape_stride(params.tma_load_c, tensor_c, 
                                                  prob_shape, prob_stride);
          // Convert strides to byte strides
          for (uint64_t& stride : prob_stride) {
            stride = (stride * sizeof_bits_v<ElementC>) / 8;
          }
          cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_C,
                                                                  prob_shape,
                                                                  prob_stride);
        }
      }
    }
    else if constexpr (is_destination_supported) {
      ElementD const* ptr_D = nullptr;
      Tensor tensor_d = make_tensor(ptr_D, make_layout(make_shape(M,N,Int<1>{}), params.dD[next_group]));

      cute::detail::fill_tma_gmem_shape_stride(params.tma_store_d, tensor_d, 
                                               prob_shape, prob_stride);
      // Convert strides to byte strides
      for (uint64_t& stride : prob_stride) {
        stride = (stride * sizeof_bits_v<ElementD>) / 8;
      }

      cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_D,
                                                              prob_shape,
                                                              prob_stride);
    }
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <bool IsLoad, class ProblemShape>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormap,
      Params const& params,
      cute::TmaDescriptor const* tensormap,
      ProblemShape problem_shape,
      int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address<IsLoad>(shared_tensormap, params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        auto problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(next_batch), 1);
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties<IsLoad>(
            shared_tensormap, params, next_batch, problem_shape_MNKL);
      }
    }
    // Ensure warp is converged before issuing tensormap fence release
    __syncwarp();
    // Entire warp must do this (ie its aligned)
    tensormaps_cp_fence_release<IsLoad>(shared_tensormap, tensormap);
  }

  template <bool IsLoad>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release(
      TensorMapStorage& shared_tensormap,
      cute::TmaDescriptor const* tensormap) {
    // Commit and wait for all TMA load/store instructions before updating the tensormap in gmem.
    // This operation only happens when the group/batch changes between consecutive tiles.
    // If there are no uncommitted instructions then tma_desc_commit_group results in an empty bulk async-group.
    auto tma_desc_wait_all_fn = [] () CUTLASS_LAMBDA_FUNC_INLINE {
      if (cute::elect_one_sync()) {
        cute::tma_desc_commit_group();
        cute::tma_desc_wait_group();
      }
    };
    // Entire warp must do this (ie its aligned)
    if constexpr (IsLoad) {
      if (is_source_supported) {
        tma_desc_wait_all_fn();
        tma_descriptor_cp_fence_release(tensormap, shared_tensormap.smem_tensormap_C);
      }
    } else if constexpr (is_destination_supported) {
      tma_desc_wait_all_fn();
      tma_descriptor_cp_fence_release(tensormap, shared_tensormap.smem_tensormap_D);
    }
  }

  template <bool IsLoad>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::TmaDescriptor const* tensormap) {
    if constexpr (IsLoad) {
      if (is_source_supported) {
        cute::tma_descriptor_fence_acquire(tensormap);
      }
    } else if constexpr (is_destination_supported) {
      cute::tma_descriptor_fence_acquire(tensormap);
    }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
