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
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"

#include "cute/tensor.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/util/type_traits.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

namespace detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Stride>
constexpr bool
is_m_major() {
  return cutlass::gemm::detail::is_major<0,Stride>();
}

template <class Stride>
constexpr bool
is_n_major() {
  return cutlass::gemm::detail::is_major<1,Stride>();
}

template <class Stride>
constexpr bool
is_im2col() {
  return cute::is_same_v<Stride, cutlass::detail::TagToStrideC_t<cutlass::layout::TensorNWC>>
      || cute::is_same_v<Stride, cutlass::detail::TagToStrideC_t<cutlass::layout::TensorNHWC>>
      || cute::is_same_v<Stride, cutlass::detail::TagToStrideC_t<cutlass::layout::TensorNDHWC>>;
}

template<class Schedule>
struct sm90_is_ptr_array_tma : cute::false_type {};

template<>
struct sm90_is_ptr_array_tma<PtrArrayTmaWarpSpecializedCooperative> : cute::true_type {};

template<>
struct sm90_is_ptr_array_tma<PtrArrayTmaWarpSpecializedPingpong> : cute::true_type {};

template<>
struct sm90_is_ptr_array_tma<PtrArrayTmaWarpSpecialized> : cute::true_type {};

template<class Schedule>
static constexpr bool sm90_is_ptr_array_tma_v = sm90_is_ptr_array_tma<Schedule>::value;

template<class Schedule>
struct sm90_is_ptr_array_tma_cooperative : cute::false_type {};

template<>
struct sm90_is_ptr_array_tma_cooperative<PtrArrayTmaWarpSpecializedCooperative> : cute::true_type {};

template<class Schedule>
static constexpr bool sm90_is_ptr_array_tma_cooperative_v = sm90_is_ptr_array_tma_cooperative<Schedule>::value;

template<class Schedule>
struct sm90_is_ptr_array_tma_pingpong : cute::false_type {};

template<>
struct sm90_is_ptr_array_tma_pingpong<PtrArrayTmaWarpSpecializedPingpong> : cute::true_type {};

template<class Schedule>
static constexpr bool sm90_is_ptr_array_tma_pingpong_v = sm90_is_ptr_array_tma_pingpong<Schedule>::value;

template<class DispatchPolicy>
struct sm90_is_ptr_array_tma_dispatch_policy : cute::false_type {};

template<
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups
>
struct sm90_is_ptr_array_tma_dispatch_policy<
    Sm90PtrArrayTmaWarpSpecialized<StagesC, 
                                   StagesD, 
                                   FragmentSize,
                                   ReuseSmemC, 
                                   DelayTmaStore, 
                                   NumEpilogueWarpGroups>> 
    : cute::true_type {};

template<
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups
>
struct sm90_is_ptr_array_tma_dispatch_policy<
    Sm120PtrArrayTmaWarpSpecialized<StagesC, 
                                   StagesD, 
                                   FragmentSize,
                                   ReuseSmemC, 
                                   DelayTmaStore, 
                                   NumEpilogueWarpGroups>> 
    : cute::true_type {};

template<class DispatchPolicy>
static constexpr bool sm90_is_ptr_array_tma_dispatch_policy_v = sm90_is_ptr_array_tma_dispatch_policy<DispatchPolicy>::value;

using cutlass::atomic_maximum;

template <class T>
static constexpr int elements_per_access_v = cutlass::sizeof_bits<uint32_t>::value / cutlass::sizeof_bits<T>::value;

template <class EpilogueSchedule>
static constexpr bool sm90_is_cooperative_v =
  cute::is_base_of_v<cutlass::epilogue::TmaWarpSpecializedCooperative, EpilogueSchedule> ||
  sm90_is_ptr_array_tma_cooperative_v<EpilogueSchedule>;

template <class EpilogueSchedule>
static constexpr bool sm90_is_warp_specialized_v =
  (!sm90_is_ptr_array_tma_cooperative_v<EpilogueSchedule> && sm90_is_ptr_array_tma_v<EpilogueSchedule>) ||
  cute::is_base_of_v<cutlass::epilogue::TmaWarpSpecialized, EpilogueSchedule>;

template <class GmemLayoutTag>
static constexpr bool is_im2col_mode =
  cute::is_same_v<GmemLayoutTag, cutlass::layout::TensorNWC> ||
  cute::is_same_v<GmemLayoutTag, cutlass::layout::TensorNHWC> ||
  cute::is_same_v<GmemLayoutTag, cutlass::layout::TensorNDHWC>;

template <class T>
struct EmptyStorage {
  CUTLASS_HOST_DEVICE
  T* data() { return nullptr; }
};

template<class EpilogueSchedule, class Stride>
CUTLASS_HOST_DEVICE
auto get_epilogue_stride(Stride stride){
  if constexpr (cute::is_base_of_v<cutlass::gemm::EpilogueTransposed, EpilogueSchedule>||
                cute::is_base_of_v<cutlass::epilogue::PtrArrayNoSmemWarpSpecializedTransposed, EpilogueSchedule>) {
    return cute::make_stride(cute::get<1>(stride), cute::get<0>(stride), cute::get<2>(stride));
  }
  else {
    return stride;
  }
}

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithBias { 
  static constexpr bool value = false; 
  using type = typename ThreadEpilogueOp::ElementCompute; 
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithBias <ThreadEpilogueOp, cute::void_t<typename ThreadEpilogueOp::ElementBias>> { 
  static constexpr bool value = true; 
  using type = typename ThreadEpilogueOp::ElementBias; 
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithPerChannelScaling {
  static constexpr bool value = false;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithPerChannelScaling <ThreadEpilogueOp, cute::enable_if_t<ThreadEpilogueOp::IsPerChannelScalingSupported>> {
  static constexpr bool value = true;
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithResidualAdd {
  static constexpr bool value = false;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithResidualAdd <ThreadEpilogueOp, cute::void_t<decltype(ThreadEpilogueOp::IsResidualSupported)>> {
  static constexpr bool value = ThreadEpilogueOp::IsResidualSupported;
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithActivation {
  static constexpr bool value = false;
  using type = void;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithActivation <ThreadEpilogueOp, cute::enable_if_t<ThreadEpilogueOp::IsEltActSupported>> {
  static constexpr bool value = true;
  using type = typename ThreadEpilogueOp::ActivationFn;
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithPerChannelScaled {
  static constexpr bool value = false;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithPerChannelScaled <ThreadEpilogueOp, cute::void_t<decltype(ThreadEpilogueOp::IsPerRowScaleSupported)>> {
  static constexpr bool value = ThreadEpilogueOp::IsPerRowScaleSupported || ThreadEpilogueOp::IsPerColScaleSupported;
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithElementwiseArguments : cute::false_type {};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithElementwiseArguments<
        ThreadEpilogueOp,
        cute::void_t<typename ThreadEpilogueOp::ElementwiseOp::Arguments>> : cute::true_type {};

// Check if ActivationFn has 'Arguments' type defined
template <class ActivationFn, class = void>
struct sm100_act_has_arguments : cute::false_type {};

template <class ActivationFn>
struct sm100_act_has_arguments<ActivationFn, cute::void_t<typename ActivationFn::Arguments> > : cute::true_type {};

template<typename EpilogueOp, typename = void>
struct Sm100EpilogueOpNumAccumulatorMtxs {
  static constexpr int value = 1;
};

template<typename EpilogueOp>
struct Sm100EpilogueOpNumAccumulatorMtxs<EpilogueOp, cute::void_t<decltype(EpilogueOp::NumAccumulatorMtxs)>> {
  static constexpr int value = EpilogueOp::NumAccumulatorMtxs;
};


// Wrapper class to use operator-style epilogues in sm90 TMA warp-specialized kernels
template <class EpilogueOp>
class Sm90TmaWarpSpecializedAdapter : public EpilogueOp {
public:
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  using LoadPipeline = cutlass::PipelineTransactionAsync<0>;
  using LoadPipelineState = cutlass::PipelineState<0>;
  constexpr static uint32_t TmaTransactionBytes = 0;
  constexpr static bool RequiresTransactionBytes = false;

  using StorePipeline = cutlass::PipelineTmaStore<0>;
  using StorePipelineState = cutlass::PipelineState<0>;

  using TensorStorage = typename EpilogueOp::SharedStorage;
  using TensorMapStorage = typename EpilogueOp::SharedStorage;
  using PipelineStorage = typename LoadPipeline::SharedStorage;

  template<class CtaTileMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_load_pipe_increment(CtaTileMNK) {
    return 1;
  }

  template<class CtaTileMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_store_pipe_increment(CtaTileMNK) {
    return 1;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors([[maybe_unused]] typename EpilogueOp::Params const&) {
  }

  // ctor inheritance
  using EpilogueOp::EpilogueOp;

  CUTLASS_HOST_DEVICE
  Sm90TmaWarpSpecializedAdapter(
      typename EpilogueOp::Params const& params,
      [[maybe_unused]] TensorStorage& shared_tensors)
    : EpilogueOp(params) { }

  CUTLASS_DEVICE
  bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE auto
  load_init(
    [[maybe_unused]] typename EpilogueOp::Params const& params,
    [[maybe_unused]] TensorMapStorage& shared_tensormaps,
    [[maybe_unused]] int32_t sm_count,
    [[maybe_unused]] int32_t sm_idx) {
    return cute::make_tuple(nullptr);
  }

  template<
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  load(
      [[maybe_unused]] LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      [[maybe_unused]] ProblemShapeMNKL problem_shape_mnkl,
      [[maybe_unused]] CtaTileMNK cta_tile_mnk,
      [[maybe_unused]] CtaCoordMNKL cta_coord_mnkl,
      [[maybe_unused]] TiledMma tiled_mma,
      [[maybe_unused]] int thread_idx,
      [[maybe_unused]] TensorStorage& shared_tensors,
      [[maybe_unused]] int subtile_idx=-1)
  {
    return load_pipe_producer_state;
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class TiledMma,
    class TensorMapC
  >
  CUTLASS_DEVICE auto
  load(
      [[maybe_unused]] LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      [[maybe_unused]] ProblemShapeMNKL problem_shape_mnkl,
      [[maybe_unused]] TileShapeMNK tile_shape_MNK,
      [[maybe_unused]] TileCoordMNKL tile_coord_mnkl,
      [[maybe_unused]] TiledMma tiled_mma,
      [[maybe_unused]] int thread_idx,
      [[maybe_unused]] TensorStorage& shared_tensors,
      [[maybe_unused]] TensorMapC const& load_tensormap,
      [[maybe_unused]] int subtile_idx=-1,
      [[maybe_unused]] bool wait = false)
  {
    return load_pipe_producer_state;
  }

  CUTLASS_DEVICE auto
  load_tail(
      [[maybe_unused]] LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state)
  {
    return load_pipe_producer_state;
  }

  CUTLASS_DEVICE auto
  store_init(
    [[maybe_unused]] typename EpilogueOp::Params const& params,
    [[maybe_unused]] TensorMapStorage& shared_tensormaps,
    [[maybe_unused]] int32_t sm_count,
    [[maybe_unused]] int32_t sm_idx,
    [[maybe_unused]] int32_t warp_group_idx) {
    return cute::make_tuple(nullptr);
  }

  template<
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  store(
      [[maybe_unused]] LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaTileMNK cta_tile_mnk,
      CtaCoordMNKL cta_coord_mnkl,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& shared_tensors,
      int subtile_index = -1)
  {
    constexpr int BLK_M_RANK = cute::rank<0>(cta_tile_mnk);
    auto m_max_coord = unwrap(cute::transform(make_seq<BLK_M_RANK>{}, [&](auto i) {
        return get<0,i>(problem_shape_mnkl) - get<0,i>(cta_tile_mnk) * get<0,i>(cta_coord_mnkl);
      }));

    constexpr int BLK_N_RANK = cute::rank<1>(cta_tile_mnk);
    auto n_max_coord = unwrap(cute::transform(make_seq<BLK_N_RANK>{}, [&](auto i) {
        return get<1,i>(problem_shape_mnkl) - get<1,i>(cta_tile_mnk) * get<1,i>(cta_coord_mnkl);
      }));

    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, Int<0>{});

    (*this)(
        problem_shape_mnkl,
        cta_tile_mnk,
        cta_coord_mnkl,
        accumulators,
        tiled_mma,
        residue_mnk,
        thread_idx,
        reinterpret_cast<char*>(&shared_tensors));

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledMma,
    class TensorMapD
  >
  CUTLASS_DEVICE auto
  store(
      [[maybe_unused]] LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& shared_tensors,
      [[maybe_unused]] TensorMapD const& store_tensormap,
      int subtile_index = -1)
  {
    constexpr int BLK_M_RANK = cute::rank<0>(tile_shape_MNK);
    auto m_max_coord = unwrap(cute::transform(make_seq<BLK_M_RANK>{}, [&](auto i) {
        return get<0,i>(problem_shape_mnkl) - get<0,i>(tile_shape_MNK) * get<0,i>(tile_coord_mnkl);
      }));

    constexpr int BLK_N_RANK = cute::rank<1>(tile_shape_MNK);
    auto n_max_coord = unwrap(cute::transform(make_seq<BLK_N_RANK>{}, [&](auto i) {
        return get<1,i>(problem_shape_mnkl) - get<1,i>(tile_shape_MNK) * get<1,i>(tile_coord_mnkl);
      }));

    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, Int<0>{});

    (*this)(
        problem_shape_mnkl,
        tile_shape_MNK,
        tile_coord_mnkl,
        accumulators,
        tiled_mma,
        residue_mnk,
        thread_idx,
        reinterpret_cast<char*>(&shared_tensors));

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  CUTLASS_DEVICE auto
  store_tail(
      [[maybe_unused]] LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state) {
    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  // Dummy methods to perform different parts of TMA/Tensormap modifications

  template <bool IsLoad,
            class ProblemShapeMNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      [[maybe_unused]] TensorMapStorage& shared_tensormaps,
      [[maybe_unused]] typename EpilogueOp::Params const& params,
      [[maybe_unused]] cute::TmaDescriptor const* tensormap,
      [[maybe_unused]] ProblemShapeMNKL problem_shape,
      [[maybe_unused]] int32_t next_batch,
      [[maybe_unused]] int32_t warp_group_idx) { }

  template <bool IsLoad>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release(
      [[maybe_unused]] TensorMapStorage& shared_tensormaps,
      [[maybe_unused]] cute::TmaDescriptor const* tensormap,
      [[maybe_unused]] int32_t warp_group_idx) { }

  template <bool IsLoad>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire([[maybe_unused]] cute::TmaDescriptor const* tensormap) { }
};


// Wrapper class to use operator-style epilogues in sm100 TMA warp-specialized kernels
template <class EpilogueOp>
class Sm100TmaWarpSpecializedAdapter : public EpilogueOp {
public:
  using LoadPipeline = cutlass::PipelineTransactionAsync<0>; // 0 stage to disable smem alloc
  using LoadPipelineState = cutlass::PipelineState<0>;

  using StorePipeline = cutlass::PipelineTmaStore<1>; // tma store pipe has no smem alloc
  using StorePipelineState = cutlass::PipelineState<1>;

  using TensorStorage = typename EpilogueOp::SharedStorage;
  using TensorMapStorage = typename EpilogueOp::SharedStorage;
  using PipelineStorage = typename LoadPipeline::SharedStorage;

  static constexpr int NumAccumulatorMtxs = Sm100EpilogueOpNumAccumulatorMtxs<EpilogueOp>::value;

  template<class CtaTileMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_load_pipe_increment(CtaTileMNK) {
    return 1;
  }

  template<class CtaTileMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_store_pipe_increment(CtaTileMNK) {
    return 1;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors([[maybe_unused]] typename EpilogueOp::Params const&) {
  }

  CUTLASS_DEVICE
  bool
  is_producer_load_needed() const {
    return false;
  }

  // ctor inheritance
  using EpilogueOp::EpilogueOp;

  CUTLASS_DEVICE auto
  load_init(
      [[maybe_unused]] typename EpilogueOp::Params const& params,
      [[maybe_unused]] TensorMapStorage& shared_tensormap,
      [[maybe_unused]] int32_t const sm_count,
      [[maybe_unused]] int32_t const sm_idx) const {
    return cute::make_tuple(nullptr);
  }

  template<
    bool ReuseTmem = false,
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma
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
      bool reverse_epi_n = false)
  {
    // C load is performed in epilogue operator
    return load_pipe_producer_state;
  }

  // with Tensormap
  template<
    bool ReuseTmem = false,
    class ProblemShapeMNKL,
    class CtaTileShapeMNK,
    class CtaTileCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    class TensorMap
  >
  CUTLASS_DEVICE auto
  load(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaTileShapeMNK tile_shape_mnk,
      CtaTileCoordMNKL cta_coord_mnkl,
      MmaTileMNK mma_tile_mnk,
      TiledMma tiled_mma,
      TensorStorage& shared_tensors,
      [[maybe_unused]] cute::tuple<TensorMap, bool> const& load_tensormap_info,
      bool reverse_epi_n = false)
  {
    // C load is performed in epilogue operator
    return load_pipe_producer_state;
  }

  CUTLASS_DEVICE void
  load_tail(
      [[maybe_unused]] LoadPipeline load_pipeline,
      [[maybe_unused]] LoadPipelineState load_pipe_producer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      [[maybe_unused]] StorePipelineState store_pipe_producer_state)
  {
  }

  CUTLASS_DEVICE auto
  store_init(
      [[maybe_unused]] typename EpilogueOp::Params const& params,
      [[maybe_unused]] TensorMapStorage& shared_tensormap,
      [[maybe_unused]] int32_t const sm_count,
      [[maybe_unused]] int32_t const sm_idx) const {
    return cute::make_tuple(nullptr);
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
    class AccLayout
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
      TensorStorage& shared_tensors
      )
  {
    // Wait for mma warp to fill tmem buffer with accumulator results
    acc_pipeline.consumer_wait(acc_pipe_consumer_state);

    auto [acc_state_next] = (*this).template operator()<ReuseTmem>(
        acc_pipeline,
        acc_pipe_consumer_state,
        problem_shape_mnkl,
        cta_tile_mnk,
        cta_coord_mnkl,
        accumulators,
        shared_tensors);

    // Let mma warp know tmem buffer is consumed and empty
    ++load_pipe_consumer_state;
    ++store_pipe_producer_state;

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state, acc_state_next);
  }

  // FastF32 API
  template<
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    class AccEngine,
    class AccLayout,
    class TiledCopyT2R
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
    cute::Tensor<AccEngine, AccLayout>& tTR_rAcc,
    TensorStorage& shared_tensors,
    TiledCopyT2R tiled_t2r)
  {
    (*this)(
      problem_shape_mnkl,
      cta_tile_mnk,
      cta_coord_mnkl,
      tTR_rAcc,
      shared_tensors,
      tiled_t2r);
    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

    // FastF32 API with Tensor Map
  template<
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    class AccEngine,
    class AccLayout,
    class TiledCopyT2R,
    class TensorMap
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
    cute::Tensor<AccEngine, AccLayout>& tTR_rAcc,
    TensorStorage& shared_tensors,
    TensorMap tensormap,
    TiledCopyT2R tiled_t2r) {
    (*this)(
      problem_shape_mnkl,
      cta_tile_mnk,
      cta_coord_mnkl,
      tTR_rAcc,
      shared_tensors,
      tiled_t2r);
    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  template<
    bool ReuseTmem = false,
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class TileCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    class AccEngine,
    class AccLayout,
    class TensorMap
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
      TileCoordMNKL cta_coord_mnkl,
      MmaTileMNK mma_tile_mnk,
      TiledMma tiled_mma,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      TensorStorage& shared_tensors,
      TensorMap tensormap
      )
  {
    // Wait for mma warp to fill tmem buffer with accumulator results
    acc_pipeline.consumer_wait(acc_pipe_consumer_state);

    auto [acc_state_next] = (*this).template operator()<ReuseTmem>(
        acc_pipeline,
        acc_pipe_consumer_state,
        problem_shape_mnkl,
        cta_tile_mnk,
        cta_coord_mnkl,
        accumulators,
        shared_tensors);

    // Let mma warp know tmem buffer is consumed and empty
    ++load_pipe_consumer_state;
    ++store_pipe_producer_state;

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state, acc_state_next);
  }

  template <class CtaTileMNK>
  CUTLASS_DEVICE void
  store_tail(
      [[maybe_unused]] LoadPipeline load_pipeline,
      [[maybe_unused]] LoadPipelineState load_pipe_consumer_state,
      [[maybe_unused]] StorePipeline store_pipeline,
      [[maybe_unused]] StorePipelineState store_pipe_producer_state,
      [[maybe_unused]] CtaTileMNK cta_tile_mnk)
  {
  }

  // Dummy methods to perform different parts of TMA/Tensormap modifications

  template <bool IsLoad, class ProblemShape>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      [[maybe_unused]] TensorMapStorage& shared_tensormap,
      [[maybe_unused]] typename EpilogueOp::Params const& params,
      [[maybe_unused]] cute::TmaDescriptor const* tensormap,
      [[maybe_unused]] ProblemShape problem_shape,
      [[maybe_unused]] int32_t next_batch) { }

  template <bool IsLoad>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release(
      [[maybe_unused]] TensorMapStorage& shared_tensormap,
      [[maybe_unused]] cute::TmaDescriptor const* tensormap) { }

  template <bool IsLoad>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire([[maybe_unused]] cute::TmaDescriptor const* tensormap) { }
};


// SFINAE helpers for detecting beta/beta_ptr/beta_ptr_array in EVT arguments.
template <class Arguments, class = void>
struct has_beta {
  static constexpr bool value = false;
};

template <class Arguments>
struct has_beta<Arguments, cute::void_t<decltype(Arguments{}.thread.beta)>> {
  static constexpr bool value = true;
};

template <class Arguments, class = void>
struct has_beta_ptr {
  static constexpr bool value = false;
};

template <class Arguments>
struct has_beta_ptr<Arguments, cute::void_t<decltype(Arguments{}.thread.beta_ptr)>> {
  static constexpr bool value = true;
};

template <class Arguments, class = void>
struct has_beta_ptr_array {
  static constexpr bool value = false;
};

template <class Arguments>
struct has_beta_ptr_array<Arguments, cute::void_t<decltype(Arguments{}.thread.beta_ptr_array)>> {
  static constexpr bool value = true;
};

} // namespace detail
} // namespace collective
} // namespace epilogue
} // namespace cutlass
