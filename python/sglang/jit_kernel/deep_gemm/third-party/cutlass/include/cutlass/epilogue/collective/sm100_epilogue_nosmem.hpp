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
  \brief Functor performing elementwise operations used by epilogues.
*/



#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/conv/detail.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

template<class T>
struct IsDefaultFusionOp {
  static constexpr bool value = false;
};

template<
  class ElementD, class ElementCompute,
  class ElementC, FloatRoundStyle RoundStyle
>
struct IsDefaultFusionOp<
  epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute, RoundStyle>
> {
  static constexpr bool value = true;
};

template<
  class ElementOutput, int Count, class ElementAccumulator,
  class ElementCompute, epilogue::thread::ScaleType::Kind Scale,
  FloatRoundStyle Round, class ElementSource
>
struct IsDefaultFusionOp<
  epilogue::thread::LinearCombination<
    ElementOutput, Count, ElementAccumulator,
    ElementCompute, Scale, Round, ElementSource>
> {
  static constexpr bool value = true;
};

// Legacy direct store sm100 epilogue using thread::LinearCombination, do not expect this to be stable
template <
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class CopyOpT2R_,
  class AlignmentC_,
  class AlignmentD_
>
class CollectiveEpilogue<
    Sm100NoSmem,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    ThreadEpilogueOp_,
    CopyOpT2R_,
    AlignmentC_,
    AlignmentD_,
    cute::enable_if_t<IsDefaultFusionOp<ThreadEpilogueOp_>::value>
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = Sm100NoSmem;
  using EpilogueTile = EpilogueTile_;
  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementBias = typename detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::type;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using CopyOpT2R = CopyOpT2R_;
  using AlignmentC = AlignmentC_;
  using AlignmentD = AlignmentD_;
  using GmemElementC = cute::conditional_t<cute::is_void_v<ElementC>,ElementD,ElementC>; // prevents void ref breakages

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  constexpr static int ThreadCount = 128;
  constexpr static int kOutputAlignment = ThreadEpilogueOp::kCount;
  constexpr static bool isEpilogueBiasSupported = detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::value;
  constexpr static bool isSourceNeeded = not cute::is_void_v<ElementC>;

  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;
  constexpr static uint32_t TmaTransactionBytes = 0;

  struct SharedStorage { };

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <conv::Operator ConvOp, int NumDims>
  static bool
  can_implement(cutlass::conv::ConvProblemShape<ConvOp,NumDims> const& problem_shape, Arguments const& args) {
    return can_implement(cutlass::conv::detail::get_transformed_problem_shape_MNKL(problem_shape), args);
  }

  template <class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;
    auto shape = cute::make_shape(M,N,L);

    bool implementable = true;
    implementable = implementable && cutlass::detail::check_alignment<AlignmentD{}>(shape, StrideD{});
    if constexpr (isSourceNeeded) {
      implementable = implementable && cutlass::detail::check_alignment<AlignmentC{}>(shape, StrideC{});
    }
    return implementable;  
  }

  //
  // Constructor and Data Members
  //
  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params, SharedStorage&) : params(params) { };

protected:
  Params const& params;

  //
  // Non-static Device Methods
  //
public:
  template<
    bool ReuseTmem = false,
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout
  >
  CUTLASS_DEVICE auto
  operator()(
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState acc_pipe_consumer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK cta_tile_shape_mnk,
      TileCoordMNKL cta_coord_mnkl,
      cute::Tensor<AccEngine, AccLayout> const& accumulators,                                      // (MMA,MMA_M,MMA_N)
      [[maybe_unused]] SharedStorage&) {

    using namespace cute;
    using X = Underscore;

    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

    auto problem_shape_mnl = select<0,1,3>(problem_shape_mnkl);
    auto cta_coord_mnl = select<0,1,3>(cta_coord_mnkl);
    auto cta_tiler = take<0,2>(cta_tile_shape_mnk);

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    Tensor mC = make_tensor(make_gmem_ptr<GmemElementC>(params.ptr_C), problem_shape_mnl, append<3>(params.dC,_0{}));      // (M,N,L)
    Tensor mD = make_tensor(make_gmem_ptr(params.ptr_D), problem_shape_mnl, append<3>(params.dD,_0{}));      // (M,N,L)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord_mnl);                                              // (CTA_M,CTA_N)
    Tensor gD = local_tile(mD, cta_tiler, cta_coord_mnl);                                              // (CTA_M,CTA_N)

    // Partition source and destination tiles according to tmem copy T2R partitioning (tTR_)
    auto tiled_t2r = make_tmem_copy(CopyOpT2R{}, tensor<0>(accumulators));
    auto thread_idx = threadIdx.x % size(tiled_t2r);

    auto thread_t2r = tiled_t2r.get_slice(thread_idx);
    Tensor tTR_gC   = thread_t2r.partition_D(gC);                                                  // (T2R,T2R_M,T2R_N)
    Tensor tTR_gD   = thread_t2r.partition_D(gD);                                                  // (T2R,T2R_M,T2R_N)
    Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tTR_gD));                              // (T2R,T2R_M,T2R_N)

    Tensor tTR_rC = make_tensor<GmemElementC>(shape(tTR_gC));                                          // (T2R,T2R_M,T2R_N)

    Tensor coordCD = make_identity_tensor(problem_shape_mnl);                                     // (M,N,L) -> (m,n,l)
    Tensor cCD = local_tile(coordCD, cta_tiler, cta_coord_mnl);                             // (CTA_M,CTA_N) -> (m,n,l)
    Tensor tTR_cCD = thread_t2r.partition_D(cCD);                                       // (T2R,T2R_M,T2R_N) -> (m,n,l)

    constexpr auto mclD = decltype(max_common_layout(tTR_rAcc.layout(), tTR_gD.layout())){};
    constexpr int VD = cute::min(AlignmentD{}, size(mclD));
    Tensor tTR_rD_frag = make_tensor<ElementD>(shape(tTR_rAcc));
    Tensor tTR_rD_src = recast<Array<ElementD, VD>>(coalesce(tTR_rD_frag));
    Tensor tR2G_rD_dst = recast<Array<ElementD, VD>>(coalesce(tTR_gD));

    Tensor tTR_cD_mn_frg = tensor<1>(zipped_divide(coalesce(tTR_cCD), mclD.compose(Int<VD>{})));
    Tensor tDpD = make_tensor<bool>(shape(tR2G_rD_dst));

    CUTLASS_PRAGMA_UNROLL
    for (int t = 0; t < size(tDpD); t++) {
      tDpD(t) = elem_less(tTR_cD_mn_frg(t), problem_shape_mnl);
    }

    constexpr auto mclC = decltype(max_common_layout(tTR_rAcc.layout(), tTR_gC.layout())){};
    constexpr int VC = cute::min(AlignmentC{}, size(mclC));

    Tensor tTR_cC_mn_frg = tensor<1>(zipped_divide(coalesce(tTR_cCD), mclC.compose(Int<VC>{})));
    Tensor tG2R_rC_dst = recast<Array<GmemElementC, VC>>(coalesce(tTR_gC));
    Tensor tCpC = make_tensor<bool>(shape(tG2R_rC_dst));

    CUTLASS_PRAGMA_UNROLL
    for (int t = 0; t < size(tCpC); t++) {
      tCpC(t) = elem_less(tTR_cC_mn_frg(t), problem_shape_mnl);
    }
    Tensor tTR_rC_src = recast<Array<GmemElementC, VC>>(coalesce(tTR_gC));
    Tensor tTR_rC_dst = recast<Array<GmemElementC, VC>>(coalesce(tTR_rC));

    // Detect interleaved complex fp32 kernels
    [[maybe_unused]] Tensor accs = accumulators;
    using ElementTmem = typename decltype(accs)::value_type;
    constexpr bool is_interleaved_complex_f32 = is_complex<ElementAccumulator>::value && cute::is_same_v<ElementTmem, float>;

    // 1. Load accumulators into register from tmem
    // Tmem -> rmem and transformation for interleaved complex kernels
    if constexpr (is_interleaved_complex_f32) {
      using ElementComputeAccumulator = float;

      Tensor tAccReal = accumulators(make_coord(_,_),_0{},_0{},_0{});                                  // (CTA_M,CTA_N)
      Tensor tAccImag = accumulators(make_coord(_,_),_0{},_0{},_1{});                                  // (CTA_M,CTA_N)
      Tensor tTR_tAccReal = thread_t2r.partition_S(tAccReal);                                      // (T2R,T2R_M,T2R_N)
      Tensor tTR_tAccImag = thread_t2r.partition_S(tAccImag);                                      // (T2R,T2R_M,T2R_N)
      Tensor tTR_rAccReal = make_tensor<ElementComputeAccumulator>(shape(tTR_gD));                 // (T2R,T2R_M,T2R_N)
      Tensor tTR_rAccImag = make_tensor<ElementComputeAccumulator>(shape(tTR_gD));                 // (T2R,T2R_M,T2R_N)

      copy(tiled_t2r, tTR_tAccReal, tTR_rAccReal);
      copy(tiled_t2r, tTR_tAccImag, tTR_rAccImag);

      // 1.1. Transform accumulators in registers
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rAccReal); i++) {
        tTR_rAcc(i) = {tTR_rAccReal(i), tTR_rAccImag(i)};
      }
    }

    // Standard tmem -> rmem epilogue
    else {
      Tensor tAcc = accumulators(make_coord(_,_),_0{},_0{});                                           // (CTA_M,CTA_N)
      Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);                                              // (T2R,T2R_M,T2R_N)

      copy(tiled_t2r, tTR_tAcc, tTR_rAcc);
    }

    cutlass::arch::fence_view_async_tmem_load();
    acc_pipeline.consumer_release(acc_pipe_consumer_state);
    ++acc_pipe_consumer_state;

    // 2. Apply element-wise operation and store to gmem
    ThreadEpilogueOp epilogue_op{params.thread};
    // source is needed
    if (epilogue_op.is_source_needed()) {
      copy_if(tCpC, tTR_rC_src, tTR_rC_dst);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rAcc); i++) {
        tTR_rD_frag(i) = epilogue_op(tTR_rAcc(i), tTR_rC(i));
      }

      copy_if(tDpD, tTR_rD_src, tR2G_rD_dst);
    }
    // source is not needed, avoid load
    else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rAcc); i++) {
        tTR_rD_frag(i) = epilogue_op(tTR_rAcc(i));
      }

      copy_if(tDpD, tTR_rD_src, tR2G_rD_dst);
    }

    return cute::make_tuple(acc_pipe_consumer_state);
  }


  // API with Global Accumulator in registers for FastFP32 (emulated MMA) kernels.
  // The accumulator in TMEM periodically loaded into the registers so that the MMA can clear out the TMEM accumulator
  // values for better accuracy. This epilogue accepts the accumulator in registers and take TiledCopy for the
  // TMEM->Reg as a parameter to be used in partitioning GMEM tensors C and D.
  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledCopy
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK cta_tile_shape_mnk,
      TileCoordMNKL cta_coord_mnkl,
      cute::Tensor<AccEngine, AccLayout>& tTR_rGlobAcc,                                      // (MMA,MMA_M,MMA_N)
      [[maybe_unused]] SharedStorage&,
      TiledCopy tiled_t2r) {

    using namespace cute;
    using X = Underscore;

    static_assert(is_rmem<AccEngine>::value, "Accumulator must be Register resident.");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(rank(AccLayout{}) == 5, "Accumulators must be copy-partitioned:  (T2R,T2R_M,T2R_N,EPI_M,EPI_N)");
    static_assert(rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

    auto problem_shape_mnl = select<0,1,3>(problem_shape_mnkl);
    auto cta_coord_mnl = select<0,1,3>(cta_coord_mnkl);
    auto cta_tiler = take<0,2>(cta_tile_shape_mnk);

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    Tensor mC = make_tensor(make_gmem_ptr<GmemElementC>(params.ptr_C), problem_shape_mnl, append<3>(params.dC,_0{})); // (M,N,L)
    Tensor mD = make_tensor(make_gmem_ptr(params.ptr_D), problem_shape_mnl, append<3>(params.dD,_0{}));      // (M,N,L)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord_mnl);                                              // (CTA_M,CTA_N)
    Tensor gD = local_tile(mD, cta_tiler, cta_coord_mnl);                                              // (CTA_M,CTA_N)


    // Partition source and destination tiles according to tmem copy T2R partitioning (tTR_)
    auto thread_t2r = tiled_t2r.get_slice(threadIdx.x % size(tiled_t2r));
    Tensor tTR_gC   = thread_t2r.partition_D(gC);                                                  // (T2R,T2R_M,T2R_N)
    Tensor tTR_gD   = thread_t2r.partition_D(gD);                                                  // (T2R,T2R_M,T2R_N)


    Tensor coordCD = make_identity_tensor(problem_shape_mnl);                                     // (M,N,L) -> (m,n,l)
    Tensor cCD = local_tile(coordCD, cta_tiler, cta_coord_mnl);                             // (CTA_M,CTA_N) -> (m,n,l)
    Tensor tTR_cCD = thread_t2r.partition_D(cCD);                                       // (T2R,T2R_M,T2R_N) -> (m,n,l)

    // 2. Apply element-wise operation and store to gmem
    ThreadEpilogueOp epilogue_op{params.thread};
    // source is needed
    if (epilogue_op.is_source_needed()) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rGlobAcc); ++i) {
        if (elem_less(tTR_cCD(i), problem_shape_mnl)) {
          tTR_gD(i) = epilogue_op(tTR_rGlobAcc(i), tTR_gC(i));
        }
      }
    }
    // source is not needed, avoid load
    else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rGlobAcc); ++i) {
        if (elem_less(tTR_cCD(i), problem_shape_mnl)) {
          tTR_gD(i) = epilogue_op(tTR_rGlobAcc(i));
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Direct store sm100 epilogue supporting EVT
template <
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpT2R_,
  class AlignmentC_,
  class AlignmentD_
>
class CollectiveEpilogue<
    Sm100NoSmem,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    FusionCallbacks_,
    CopyOpT2R_,
    AlignmentC_,
    AlignmentD_,
    cute::enable_if_t<not IsDefaultFusionOp<FusionCallbacks_>::value>
> {
public:
  //
  // Type Aliases
  //
  // Required by the gemm::kernel
  using DispatchPolicy = Sm100NoSmem;
  using ElementC = ElementC_;
  using ElementD = ElementD_;
  using GmemElementC = cute::conditional_t<cute::is_void_v<ElementC>,ElementD,ElementC>; // prevents void ref breakages
  using StrideC = StrideC_;
  using StrideD = StrideD_;
  using EpilogueTile = EpilogueTile_;
  using CopyOpT2R = CopyOpT2R_;
  using FusionCallbacks = FusionCallbacks_;
  using ThreadEpilogueOp = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

private:
  constexpr static bool IsReductionBufferNeeded = ThreadEpilogueOp::IsDePerRowBiasSupported
                                               || is_same_v<ThreadEpilogueOp, epilogue::fusion::FusionOperation>; // alloc reduction buffer for custom EVTs
  constexpr static size_t ImplicitSharedStorageSize = IsReductionBufferNeeded ? size(EpilogueTile{}) : 0;

  // Not unroll epi subtile loop when the activation op is heavy to reduce instruction size and register pressure.
  constexpr static bool UnrollEpiLoop =
    not cutlass::epilogue::thread::kIsHeavy_member_or_false<typename ThreadEpilogueOp::ActivationFn>::value;

public:
  constexpr static int ThreadCount = 128;
  constexpr static uint32_t TmaTransactionBytes = 0;

  struct SharedStorage {
    using FusionStorage = typename FusionCallbacks::SharedStorage;
    FusionStorage thread;
    array_aligned<uint8_t, ImplicitSharedStorageSize> buffer;
  };

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC = {};
    ElementD* ptr_D = nullptr;
    StrideD dD = {};
  };

  // Device side epilogue params
  struct Params {
    typename FusionCallbacks::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC = {};
    ElementD* ptr_D = nullptr;
    StrideD dD = {};
  };

  //
  // Constructor and Data Members
  //
  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params_, SharedStorage& shared_tensors)
  : fusion_callbacks(params_.thread, shared_tensors.thread)
  , smem_buffer_ptr(shared_tensors.buffer.data())
  , params(params_) {};

protected:
  FusionCallbacks fusion_callbacks;
  uint8_t* smem_buffer_ptr;
  Params const& params;

public:

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace),
      args.ptr_C,
      args.dC,
      args.ptr_D,
      args.dD
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return FusionCallbacks::get_workspace_size(problem_shape, args.thread);
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
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {

    bool fusion_implementable = FusionCallbacks::can_implement(problem_shape, args.thread);
    if (!fusion_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum requirements for FusionCallbacks.\n");
    }
    return fusion_implementable;
  }


  template<
    bool ReuseTmem = false,
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class AccEngine, class AccLayout
  >
  CUTLASS_DEVICE auto
  operator()(
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState acc_pipe_consumer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaTileMNK cta_tile_mnk,
      CtaCoordMNKL cta_coord_mnkl,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      [[maybe_unused]] SharedStorage&
  ) {
    using ElementAccumulator = typename AccEngine::value_type;
    using ElementCompute_ = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::ElementCompute;
    using ElementCompute = cute::conditional_t<cute::is_void_v<ElementCompute_>,ElementAccumulator,ElementCompute_>;

    // Wait for mma warp to fill tmem buffer with accumulator results
    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(rank(CtaCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");
    static_assert(cute::sizeof_bits_v<ElementD> != 6, "Output element requires smem");

    auto [M, N, K, L] = problem_shape_mnkl;
    auto problem_shape_mnl = select<0,1,3>(problem_shape_mnkl);
    auto cta_coord_mnl = select<0,1,3>(cta_coord_mnkl);
    auto cta_tiler = take<0,2>(cta_tile_mnk);

    int thread_idx = threadIdx.x % ThreadCount;

    Tensor tAcc = accumulators(make_coord(_,_),_0{},_0{});                                             // (CTA_M,CTA_N)
    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                         // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    TiledCopy tiled_t2r = make_tmem_copy(CopyOpT2R{}, tAcc_epi(_,_,_0{},_0{}));
    ThrCopy thread_t2r = tiled_t2r.get_slice(thread_idx);
    Tensor tTR_tAcc = thread_t2r.partition_S(tAcc_epi);                                // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)

    constexpr int FragmentSize = size(EpilogueTile{}) / ThreadCount;

    Tensor coordD = make_identity_tensor(problem_shape_mnl);                                      // (M,N,L) -> (m,n,l)
    Tensor cD = local_tile(coordD, cta_tiler, cta_coord_mnl);                               // (CTA_M,CTA_N) -> (m,n,l)
    Tensor cD_epi = flat_divide(cD, EpilogueTile{});
    Tensor tTR_cD = thread_t2r.partition_D(cD_epi);                                     // (T2R,T2R_M,T2R_N) -> (m,n,l)

    Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tTR_cD(_,_,_,_0{},_0{})));

    // Construct the EVT consumer callbacks
    auto residue_cD = make_coord(M,N) - cD(_0{});
    auto residue_tTR_cD = make_coord(M,N) - tTR_cD(_0{});
    Tensor cD_ = make_coord_tensor(cD.layout());
    Tensor tTR_cD_ = make_coord_tensor(tTR_cD.layout());
    constexpr bool RefSrc = false;

    Tensor mC = make_tensor(make_gmem_ptr<GmemElementC>(params.ptr_C), make_shape(M,N,L), params.dC);

    Tensor tTR_gC = cutlass::epilogue::fusion::sm90_partition_for_epilogue<RefSrc>(
                      mC, cta_tile_mnk, cta_coord_mnkl, EpilogueTile{}, tiled_t2r, thread_idx);

    Tensor mD = make_tensor(make_gmem_ptr(recast_ptr<ElementD>(params.ptr_D)), make_shape(M,N,L), params.dD);

    Tensor tTR_gD = cutlass::epilogue::fusion::sm90_partition_for_epilogue<RefSrc>(
                      mD, cta_tile_mnk, cta_coord_mnkl, EpilogueTile{}, tiled_t2r, thread_idx);

    // Register Tensor
    Tensor tTR_rD = make_tensor<ElementD>(take<0,3>(shape(tTR_gD)));

    Tensor coord_cCD = make_identity_tensor(problem_shape_mnl);
    Tensor tTR_cCD = cutlass::epilogue::fusion::sm90_partition_for_epilogue<RefSrc>(
                      coord_cCD, cta_tile_mnk, cta_coord_mnkl, EpilogueTile{}, tiled_t2r, thread_idx);
    constexpr auto mclD = decltype(max_common_layout(tTR_gD(_,_,_,_0{},_0{}), tTR_rD)){};
    constexpr int VD = cute::min(AlignmentD_{}, size(mclD));

    auto tCrC = make_tensor<GmemElementC>(take<0,3>(shape(tTR_gC)));
    constexpr auto mclC = decltype(max_common_layout(tTR_gC(_,_,_,_0{},_0{}), tCrC)){};
    constexpr int VC = cute::min(AlignmentC_{}, size(mclC));

    Tensor tTR_rD_frg = recast<Array<ElementD, FragmentSize>>(coalesce(tTR_rD));

    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
      problem_shape_mnkl,
      cta_tile_mnk,
      cta_coord_mnkl,
      int(0),
      EpilogueTile{},
      tiled_t2r,
      cD_,
      residue_cD,
      tTR_cD_,
      residue_tTR_cD,
      tCrC,
      thread_idx
    };

    auto synchronize = [] () CUTLASS_LAMBDA_FUNC_INLINE { cutlass::arch::NamedBarrier::sync(ThreadCount, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };

    // The Epilogue Loop
    auto epi_loop_fn = [&] (auto& cst_callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
      bool is_C_load_needed = fusion_callbacks.is_C_load_needed();

      // Ensure there are no threads from the previous wave writing to shared memory being utilized for the current wave.
      synchronize();
      cst_callbacks.begin();
      if (cst_callbacks.begin_sync_needed()) {
        synchronize();
      }

      // If tmem doesn't have enough capacity to support double buffering, a portion of tmem (a column of epilogue tiles)
      // is overlapped between 2 pseudo-buffers. The shared tmem portion corresponds to the last epilogue tile column of
      // tmem accumulator buffer 0, and the first epilogue tile column of tmem accumulator 1.
      // Thus, whenever we are processing tmem accumulator buffer 0, we process the epilogue tiles with reversed column order.
      // Once the last epilogue tile column is loaded from tmem, the acc_pipeline is released.
      // Then, the next accumulation stage for buffer 1 can start.
      [[maybe_unused]] bool reverse_epi_n = ReuseTmem && acc_pipe_consumer_state.phase() == 0;
      static_assert(not (ReuseTmem && AccumulatorPipeline::Stages != 1), "Tmem reuse requires 1 accumulator stage");

      // For each epilogue subtile within the CTA tile
      constexpr int NumEpiSubtilesN = CUTE_STATIC_V(size<4>(tTR_tAcc));
      constexpr int NumEpiSubtilesM = CUTE_STATIC_V(size<3>(tTR_tAcc));
      #pragma unroll(UnrollEpiLoop ? NumEpiSubtilesN : 1)
      for (int iter_n = 0; iter_n < NumEpiSubtilesN; ++iter_n) {
        #pragma unroll(UnrollEpiLoop ? NumEpiSubtilesM : 1)
        for (int iter_m = 0; iter_m < NumEpiSubtilesM; ++iter_m) {
          int epi_m = iter_m, epi_n = iter_n;

          bool is_last_iteration = iter_m == size<3>(tTR_tAcc)-1 && iter_n == size<4>(tTR_tAcc)-1;
          bool do_acc_release = is_last_iteration;

          // Reverse subtile order for tmem reuse if necessary
          if constexpr (ReuseTmem) {
            if (reverse_epi_n) {
              epi_n = size<4>(tTR_tAcc) - 1 - iter_n;
            }
            do_acc_release = iter_m == size<3>(tTR_tAcc)-1 && iter_n == 0;
          }

          Tensor tTR_cCD_mn = tTR_cCD(_,_,_,epi_m,epi_n);
          Tensor tTR_pCD_mn = cute::lazy::transform(tTR_cCD_mn, [&] (auto const& c) CUTLASS_LAMBDA_FUNC_INLINE { return elem_less(c, problem_shape_mnl); });
          cst_callbacks.begin_loop(epi_m, epi_n);

          if constexpr (not cute::is_void_v<ElementC>) {
            if (is_C_load_needed) {
              using CVecType = uint_bit_t<VC * sizeof_bits_v<ElementC>>;

              if constexpr (!is_same_v<CVecType, uint256_t>) {
                Tensor tTR_gC_frg = recast<CVecType>(coalesce(tTR_gC(_,_,_,epi_m,epi_n)));
                Tensor tTR_rC_frg = recast<CVecType>(coalesce(tCrC));
                Tensor tTR_pC_frg = tensor<1>(zipped_divide(coalesce(tTR_pCD_mn), mclC.compose(Int<VC>{})));
                copy_if(tTR_pC_frg, tTR_gC_frg, tTR_rC_frg);
              }
              else {
                auto tiled_g2r = make_tiled_copy_D(Copy_Atom<SM100_LOAD_256bit_CACHE_NOALLOCATION, ElementC>{}, tiled_t2r);
                auto thr_g2r = tiled_g2r.get_slice(threadIdx.x);
                Tensor c_src = thr_g2r.retile_S(tTR_gC(_,_,_,epi_m,epi_n));
                Tensor c_dst = thr_g2r.retile_D(tCrC);
                Tensor c_prd = thr_g2r.retile_D(tTR_pCD_mn);
                copy_if(tiled_g2r, c_prd, c_src, c_dst);
              }
            }
          }

          // Copy accumulator tile from tmem to register
          // The current tile in tmem
          Tensor tTR_tAcc_mn = tTR_tAcc(_,_,_,epi_m,epi_n);

          Tensor tTR_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc));

          copy(tiled_t2r, tTR_tAcc_mn, tTR_rAcc);

          // After the last tmem load, signal that tmem buffer is consumed and empty
          if (do_acc_release) {
            cutlass::arch::fence_view_async_tmem_load();
            acc_pipeline.consumer_release(acc_pipe_consumer_state);
            ++acc_pipe_consumer_state;
          }

          CUTLASS_PRAGMA_UNROLL
          for (int epi_v = 0; epi_v < size(tTR_rAcc_frg); ++epi_v) {
            tTR_rD_frg(epi_v) = cst_callbacks.visit(tTR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
          }

          Tensor reduction_buffer = make_tensor(
            raw_pointer_cast(make_smem_ptr(smem_buffer_ptr)), make_layout(Shape<Int<ImplicitSharedStorageSize>>{}));

          cst_callbacks.reduce(reduction_buffer, synchronize, epi_m, epi_n, is_last_iteration, tTR_rAcc /*not used*/);

          cst_callbacks.end_loop(epi_m, epi_n);

          using VecType = uint_bit_t<VD * sizeof_bits_v<ElementD>>;
          if constexpr (!is_same_v<VecType, uint256_t>) {
            Tensor tTR_gD_frg = recast<VecType>(coalesce(tTR_gD(_,_,_,epi_m,epi_n)));
            Tensor tTR_rD_frg = recast<VecType>(coalesce(tTR_rD));
            Tensor tTR_pD_frg = tensor<1>(zipped_divide(coalesce(tTR_pCD_mn), mclD.compose(Int<VD>{})));
            copy_if(tTR_pD_frg, tTR_rD_frg, tTR_gD_frg);
          }
          else {
            auto tiled_r2g = make_tiled_copy_D(Copy_Atom<SM100_STORE_256bit_CACHE_NOALLOCATION, ElementD>{}, tiled_t2r);
            auto thr_r2g = tiled_r2g.get_slice(threadIdx.x);
            Tensor src = thr_r2g.retile_S(tTR_rD);
            Tensor dst = thr_r2g.retile_D(tTR_gD(_,_,_,epi_m,epi_n));
            Tensor prd = thr_r2g.retile_D(tTR_pCD_mn);
            copy_if(tiled_r2g, prd, src, dst);
          }

        } // for epi_m
      } // for epi_n

      cst_callbacks.end();
    };

    //
    // BEGIN EPILOGUE
    //
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);
    epi_loop_fn(cst_callbacks);
    return cute::make_tuple(acc_pipe_consumer_state);
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////
// For sm100 kernels requiring warp specialized epilogues
template <
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class CopyOpT2R_,
  class AlignmentC_,
  class AlignmentD_
>
class CollectiveEpilogue<
    Sm100NoSmemWarpSpecialized,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    ThreadEpilogueOp_,
    CopyOpT2R_,
    AlignmentC_,
    AlignmentD_
> : public detail::Sm100TmaWarpSpecializedAdapter<CollectiveEpilogue<
      Sm100NoSmem,
      EpilogueTile_,
      ElementC_,
      StrideC_,
      ElementD_,
      StrideD_,
      ThreadEpilogueOp_,
      CopyOpT2R_,
      AlignmentC_,
      AlignmentD_,
      void>>
{
public:
  // ctor inheritance
  using detail::Sm100TmaWarpSpecializedAdapter<CollectiveEpilogue<
      Sm100NoSmem,
      EpilogueTile_,
      ElementC_,
      StrideC_,
      ElementD_,
      StrideD_,
      ThreadEpilogueOp_,
      CopyOpT2R_,
      AlignmentC_,
      AlignmentD_,
      void>>::Sm100TmaWarpSpecializedAdapter;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
