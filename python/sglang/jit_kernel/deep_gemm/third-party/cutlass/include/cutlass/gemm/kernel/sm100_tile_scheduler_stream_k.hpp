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

#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel::detail {

// Persistent Thread Block (TB) scheduler leveraging stream-K decomposition
template <
  class TileShape,
  class ClusterShape,
  uint32_t Stages_
>
class PersistentTileSchedulerSm100StreamK {
  using UnderlyingScheduler = PersistentTileSchedulerSm100<ClusterShape, Stages_>;
  using UnderlyingStreamKScheduler = PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>;
  using InternalWorkTileInfo = typename UnderlyingScheduler::WorkTileInfo;
  using InternalParams = typename UnderlyingScheduler::Params;
  // Shapediv failures currently occur with tile shape N of 192
  static constexpr bool ForceDataParallel = size<1>(TileShape{}) == 192;

public:
  static constexpr uint32_t Stages = Stages_;

  using CLCResponse = typename UnderlyingScheduler::CLCResponse;
  using WorkTileInfo = typename UnderlyingStreamKScheduler::WorkTileInfo;
  using Arguments = typename UnderlyingStreamKScheduler::Arguments;

  using Params = PersistentTileSchedulerSm100StreamKParams;
  using RasterOrder = PersistentTileSchedulerSm90Params::RasterOrder;
  using RasterOrderOptions = PersistentTileSchedulerSm90Params::RasterOrderOptions;

  using SharedStorage = typename UnderlyingScheduler::SharedStorage;
  using Pipeline = typename UnderlyingScheduler::Pipeline;
  using ThrottlePipeline = typename UnderlyingScheduler::ThrottlePipeline;

  static constexpr bool IsDynamicPersistent = true;

  // Number of sub blocks in the kernel epilogue
  static constexpr int EpilogueSubtiles = 1;

  CUTLASS_HOST_DEVICE
  PersistentTileSchedulerSm100StreamK() { }

  CUTLASS_DEVICE
  PersistentTileSchedulerSm100StreamK(Params const& params)
    : sm100_scheduler_(params.sm100_params_)
    , params_(params)
    , block_id_in_cluster_(cute::block_id_in_cluster()) {
    // Set the current linear idx to be equal to the linear idx of the first work tile to be computed
    auto cs = make_shape(
      params.sm100_params_.divmod_cluster_shape_m_.divisor,
      params.sm100_params_.divmod_cluster_shape_n_.divisor,
      Int<1>{});
  }

  CUTLASS_DEVICE
  PersistentTileSchedulerSm100StreamK(CLCResponse* clc_response_ptr, Params const& params, dim3 block_id_in_cluster)
    : sm100_scheduler_(clc_response_ptr, params.sm100_params_, block_id_in_cluster),
      params_(params),
      block_id_in_cluster_(block_id_in_cluster) {
    // Set the current linear idx to be equal to the linear idx of the first work tile to be computed
    auto cs = make_shape(
      params.sm100_params_.divmod_cluster_shape_m_.divisor,
      params.sm100_params_.divmod_cluster_shape_n_.divisor,
      Int<1>{});
  }

  template <class ProblemShape, class TileShapeMNK>
  CUTLASS_DEVICE
  PersistentTileSchedulerSm100StreamK(CLCResponse* clc_response_ptr, Params const& params,
    ProblemShape problem_shape_mnkl, TileShapeMNK tile_shape, dim3 block_id_in_cluster)
    : PersistentTileSchedulerSm100StreamK(clc_response_ptr, params, block_id_in_cluster) { }

  template <class ProblemShape>
  static Params
  to_underlying_arguments(
      ProblemShape problem_shape,
      TileShape tile_shape,
      [[maybe_unused]] ClusterShape cluster_shape,
      KernelHardwareInfo const& hw_info,
      Arguments const& args,
      void* workspace,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      uint32_t ktile_start_alignment_count = 1u) {

    auto cs = cutlass::detail::select_cluster_shape(cluster_shape, hw_info.cluster_shape);
    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cs);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    Params params;
    params.initialize(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(cs),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.reduction_mode,
      ForceDataParallel ? Params::DecompositionMode::DataParallel : args.decomposition_mode,
      workspace,
      ktile_start_alignment_count
    );
    return params;
  }

  template <class ProblemShape, class TileShapeMNK, class AtomThrShape>
  static Params
  to_underlying_arguments(
      ProblemShape problem_shape_mnkl,
      TileShapeMNK tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo const& hw_info,
      Arguments const& args,
      void* workspace = nullptr,
      uint32_t ktile_start_alignment_count = 1u
      ) {

    auto cs = cutlass::detail::select_cluster_shape(cluster_shape_mnk, hw_info.cluster_shape);
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, cs);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    Params params;
    params.initialize(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(cs),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.reduction_mode,
      ForceDataParallel ? Params::DecompositionMode::DataParallel : args.decomposition_mode,
      workspace,
      ktile_start_alignment_count
    );

    return params;
  }

  static bool
  can_implement(Arguments const& args) {
    return UnderlyingStreamKScheduler::can_implement(args);
  }

  CUTLASS_DEVICE
  PipelineState<Stages> 
  advance_to_next_work(Pipeline& clc_pipeline, PipelineState<Stages> clc_pipe_producer_state) const {
    return sm100_scheduler_.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
 }

  // Given the inputs, computes the total number of output blocks this problem will compute over
  template<class ProblemShape>
  CUTLASS_HOST_DEVICE
  static dim3
  get_tiled_cta_shape_mnl(ProblemShape problem_shape_mnkl, TileShape blk_shape, ClusterShape cluster_shape) {
    return UnderlyingScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, blk_shape, cluster_shape);
  }

  template<class ProblemShape, class TileShapeMNK, class AtomThrShape>
  CUTLASS_HOST_DEVICE
  static dim3
  get_tiled_cta_shape_mnl(ProblemShape problem_shape_mnkl,
                          TileShapeMNK tile_shape_mnk,
                          AtomThrShape atom_thr_shape_mnk,
                          ClusterShape cluster_shape_mnk) {
    return UnderlyingScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, cluster_shape_mnk);
  }

  // Given the inputs, computes the physical grid we should launch.
  template <class ProblemShape>
  CUTLASS_HOST_DEVICE
  static dim3
  get_grid_shape(
    Params const& params,
    ProblemShape problem_shape,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info,
    [[maybe_unused]] Arguments arguments) {
    
    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
    return params.get_grid_shape(problem_blocks, to_gemm_coord(cluster_shape));
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShape, class TileShapeMNK, class AtomThrShape>
  CUTLASS_HOST_DEVICE
  static dim3
  get_grid_shape(
    Params const& params,
    ProblemShape problem_shape_mnkl,
    TileShapeMNK tile_shape_mnk,
    AtomThrShape atom_thr_shape_mnk,
    ClusterShape cluster_shape_mnk,
    KernelHardwareInfo hw_info) {

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, cluster_shape_mnk);
    return params.get_grid_shape(problem_blocks, to_gemm_coord(cluster_shape_mnk));
  }


  // Returns the initial work tile info that will be computed over
  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info(ClusterShape cluster_shape) {
    InternalWorkTileInfo work_tile_info = sm100_scheduler_.initial_work_tile_info(cluster_shape);
    work_tile_info.is_valid_tile = false;
    return convert_work(work_tile_info);
  }

  // Returns a CTA-tiled coordinate for the provided work tile info
  CUTLASS_DEVICE
  auto
  work_tile_to_cta_coord(WorkTileInfo const& work_tile_info) {
    if (is_dp_only()) {
      // For data-parallel decompositions, simply default to the
      // underlying SM100 scheduler.
      auto underlying_work_tile = to_underlying_work_tile_info(work_tile_info);
      return sm100_scheduler_.work_tile_to_cta_coord(underlying_work_tile);
    }
    else {
      // The SM90 stream-K scheduler already operates only at CTA level,
      // so the returned work tile info already contains CTA offsets within
      // each cluster tile.
      return cute::make_coord(
        work_tile_info.M_idx,
        work_tile_info.N_idx,
        _,
        work_tile_info.L_idx
      );
    }
  }

  // Returns whether the current work_tile_info passed in should continue to be used.
  CUTLASS_DEVICE
  bool
  continue_current_work(WorkTileInfo& work_tile_info) const {
    return UnderlyingStreamKScheduler::continue_current_work_for_linear_idx(
      current_work_linear_idx_, unit_iter_start_, block_id_in_cluster_, work_tile_info, params_.sk_params_);
  }

  // Kernel helper function to get next CLC ID and whether to advance the CLC pipeline state.
  template <class CLCPipeline, class CLCPipelineState>
  CUTLASS_DEVICE
  cute::tuple<WorkTileInfo, bool>
  fetch_next_work(
    WorkTileInfo work_tile_info,
    CLCPipeline& clc_pipeline,
    CLCPipelineState clc_pipe_consumer_state) {
    // Check whether we should continue on with the current work unit. If this is the case,
    // the work unit will have been updated in continue_current_work to reflect the new
    // tile to be computed. Return `false` to indicate that the CLC pipeline state
    // need not be advanced.
    if (continue_current_work(work_tile_info)) {
      return cute::make_tuple(work_tile_info, false);
    }

    auto [work_tile, _] = sm100_scheduler_.fetch_next_work(InternalWorkTileInfo{}, clc_pipeline, clc_pipe_consumer_state);
    if (!work_tile.is_valid()) {
      return cute::make_tuple(invalid_work_tile(), true);
    }

    auto converted_work_tile = convert_work(work_tile);

    // Return true to indicate that the CLC pipeline state should be advanced
    return cute::make_tuple(converted_work_tile, true);
  }

  CUTLASS_DEVICE
  cute::tuple<WorkTileInfo, bool>
  fetch_next_work(WorkTileInfo work_tile_info) {
    return cute::make_tuple(work_tile_info, true);
  }

  // Set data SMEM ptr 
  CUTLASS_DEVICE
  void
  set_data_ptr(CLCResponse* clc_response_ptr) {
    sm100_scheduler_.set_data_ptr(clc_response_ptr);
  }

  CUTLASS_DEVICE
  static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const& work_tile_info) {
    return true;
  }

  CUTLASS_DEVICE
  static bool
  requires_separate_reduction(Params const& params) {
    return false;
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the case of stream-K, this should only occur if the work is marked as the final split.
  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const& work_tile_info, Params const& params) {
    return UnderlyingStreamKScheduler::compute_epilogue(work_tile_info, params.sk_params_);
  }

  // Non-static variant of compute_epilogue. Used in cases where passing
  // in Params is inconvenient.
  CUTLASS_HOST_DEVICE
  bool
  compute_epilogue(WorkTileInfo const& work_tile_info) const {
    return UnderlyingStreamKScheduler::compute_epilogue(work_tile_info, params_.sk_params_);
  }

  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(
    Arguments const& args,
    ProblemShape problem_shape,
    KernelHardwareInfo const& hw_info,
    uint32_t reduction_warp_groups,
    [[maybe_unused]] const uint32_t epilogue_subtile = 1,
    uint32_t num_accumulator_mtxs = 1,
    uint32_t ktile_start_alignment_count = 1) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);

    auto cs = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    TileShape tile_shape;

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cs);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    return Params::get_workspace_size(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(tile_shape),
      to_gemm_coord(cs),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      ForceDataParallel ? Params::DecompositionMode::DataParallel : args.decomposition_mode,
      args.reduction_mode,
      reduction_warp_groups,
      sizeof_bits<typename UnderlyingStreamKScheduler::BarrierType>::value,
      sizeof_bits<ElementAccumulator>::value,
      EpilogueSubtiles,
      num_accumulator_mtxs,
      ktile_start_alignment_count
    );
  }

  template <class ElementAccumulator, class ProblemShape, class TileShapeMNK, class AtomThrShape>
  static size_t
  get_workspace_size(
      Arguments const& args,
      ProblemShape problem_shape,
      TileShapeMNK tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo const& hw_info,
      uint32_t reduction_warp_groups,
      uint32_t num_accumulator_mtxs = 1,
      uint32_t ktile_start_alignment_count = 1) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);

    auto cs = cutlass::detail::select_cluster_shape(cluster_shape_mnk, hw_info.cluster_shape);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, cs);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    auto cta_tile_shape_mnk = shape_div(tile_shape_mnk, atom_thr_shape_mnk);

    return Params::get_workspace_size(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(cta_tile_shape_mnk),
      to_gemm_coord(cs),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      ForceDataParallel ? Params::DecompositionMode::DataParallel : args.decomposition_mode,
      args.reduction_mode,
      reduction_warp_groups,
      sizeof_bits<typename UnderlyingStreamKScheduler::BarrierType>::value,
      sizeof_bits<ElementAccumulator>::value,
      EpilogueSubtiles,
      num_accumulator_mtxs,
      ktile_start_alignment_count
    );
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(
    Arguments const& args,
    void* workspace,
    cudaStream_t stream,
    ProblemShape const& problem_shape,
    KernelHardwareInfo const& hw_info,
    uint32_t reduction_warp_groups,
    [[maybe_unused]] const uint32_t epilogue_subtile = 1,
    uint32_t num_accumulator_mtxs = 1,
    CudaHostAdapter *cuda_adapter = nullptr,
    uint32_t ktile_start_alignment_count = 1) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);

    auto cs = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    TileShape tile_shape;

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cs);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    return Params::initialize_workspace(
      workspace,
      stream,
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(tile_shape),
      to_gemm_coord(cs),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      ForceDataParallel ? Params::DecompositionMode::DataParallel : args.decomposition_mode,
      args.reduction_mode,
      reduction_warp_groups,
      sizeof_bits<typename UnderlyingStreamKScheduler::BarrierType>::value,
      sizeof_bits<ElementAccumulator>::value,
      EpilogueSubtiles,
      num_accumulator_mtxs,
      cuda_adapter,
      ktile_start_alignment_count
    );
  }

  template <class ElementAccumulator, class ProblemShape, class TileShapeMNK, class AtomThrShape>
  static cutlass::Status
  initialize_workspace(
      Arguments const& args,
      void* workspace,
      cudaStream_t stream,
      ProblemShape const& problem_shape,
      TileShapeMNK tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo const& hw_info,
      uint32_t reduction_warp_groups,
      uint32_t num_accumulator_mtxs = 1,
      CudaHostAdapter *cuda_adapter = nullptr,
      uint32_t ktile_start_alignment_count = 1) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);

    auto cs = cutlass::detail::select_cluster_shape(cluster_shape_mnk, hw_info.cluster_shape);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, cs);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    auto cta_tile_shape_mnk = shape_div(tile_shape_mnk, atom_thr_shape_mnk);

    return Params::initialize_workspace(
      workspace,
      stream,
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(cta_tile_shape_mnk),
      to_gemm_coord(cs),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      ForceDataParallel ? Params::DecompositionMode::DataParallel : args.decomposition_mode,
      args.reduction_mode,
      reduction_warp_groups,
      sizeof_bits<typename UnderlyingStreamKScheduler::BarrierType>::value,
      sizeof_bits<ElementAccumulator>::value,
      EpilogueSubtiles,
      num_accumulator_mtxs,
      cuda_adapter,
      ktile_start_alignment_count
    );
  }

  template <class ProblemShape, class TileShapeMNK>
  CUTLASS_HOST_DEVICE
  static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape, TileShapeMNK) {
    return work_tile_info.k_tile_count;
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const& work_tile_info) {
    return work_tile_info.K_idx;
  }

  template <class ProblemShape, class TileShapeMNK, class Shape>
  CUTLASS_DEVICE
  auto
  get_k_tile_iterator(WorkTileInfo const& work_tile_info, ProblemShape problem_shape, TileShapeMNK tile_shape, Shape) {
    // Get the shape of k tiles instead of the counter.  Otherwise, if the problem shape has
    // multiple k modes, the DMA loop would need to decompose the iterator onto every mode
    // every time global loading happens.  This would incur extra overhead.
    auto k_tiles = cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape));
    auto k_tile_start = get_work_k_tile_start(work_tile_info);
    // Iterate start from current k tile start over the k tiles shape.
    return cute::make_coord_iterator(idx2crd(k_tile_start, k_tiles), k_tiles);
  }

  // Returns whether fixup is needed for `work_tile_info`.
  CUTLASS_HOST_DEVICE
  bool
  requires_fixup(WorkTileInfo const work_tile_info) const {
    return UnderlyingStreamKScheduler::requires_fixup(params_.sk_params_, work_tile_info);
  }

  // Performs the reduction across splits for a given output tile.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  void
  fixup(
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx,
    uint32_t num_accumulator_mtxs = 1) const {

    using BarrierManager = SyncManager<cutlass::detail::SyncwarpSync, NumThreadsPerWarp>;

    UnderlyingStreamKScheduler s;
    return s.template fixup_helper<FrgTensorC, BarrierManager>(
      params_.sk_params_, work_tile_info, accumulators, num_barriers, barrier_idx, num_accumulator_mtxs);
  }


  // Performs the reduction across splits for a given output tile.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  fixup(
      Params const& params,
      WorkTileInfo const& work_tile_info,
      FrgTensorC& accumulators,
      uint32_t num_barriers,
      uint32_t barrier_idx) {
    UnderlyingStreamKScheduler::fixup(params.sk_params_, work_tile_info, accumulators, num_barriers, barrier_idx);
  }

  // Performs reduction across splits for a given output tile
  template <
    bool IsComplex,
    class TiledMma,
    class AccEngine,
    class AccLayout,
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class CopyOpT2R
  >
  CUTLASS_DEVICE
  AccumulatorPipelineState
  fixup(
      TiledMma const& tiled_mma,
      WorkTileInfo const& work_tile_info,
      cute::Tensor<AccEngine, AccLayout>& accumulators,
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState acc_pipe_consumer_state,
      CopyOpT2R) const {
    using namespace cute;
    static_assert(cute::is_rmem_v<AccEngine> || cute::is_tmem_v<AccEngine>, "Accumulator must be in either TMEM or RF");

    if constexpr (ForceDataParallel) {
      return acc_pipe_consumer_state;
    }
    else {
      if (!requires_fixup(work_tile_info)) {
        if constexpr (cute::is_tmem_v<AccEngine>) {
          if (!work_tile_info.is_valid()) {
            // The first work tile can be invalid, but still must release TMEM
            acc_pipeline.consumer_wait(acc_pipe_consumer_state);
            acc_pipeline.consumer_release(acc_pipe_consumer_state);
            ++acc_pipe_consumer_state;
          }
        }
        return acc_pipe_consumer_state;
      }

      if constexpr (cute::is_tmem_v<AccEngine>) {
        // When accumulators reside in TMEM, perform TMEM -> RF loads before performing fixup,
        // and perform RF -> TMEM stores after fixup (when the split must compute the epilogue)
        if constexpr (IsComplex) {
          constexpr uint32_t NumAccumulatorMtx = 2;
          Tensor accumulators_real = accumulators(_,_,_,0);
          tmem_fixup(
            tiled_mma,
            work_tile_info,
            accumulators_real,
            acc_pipeline,
            acc_pipe_consumer_state,
            CopyOpT2R{},
            NumAccumulatorMtx,
            0 /*idx_accumulator_mtx*/
          );

          Tensor accumulators_imag = accumulators(_,_,_,1);
          return tmem_fixup(
            tiled_mma,
            work_tile_info,
            accumulators_imag,
            acc_pipeline,
            acc_pipe_consumer_state,
            CopyOpT2R{},
            NumAccumulatorMtx,
            1 /*idx_accumulator_mtx*/
          );
        }
        else {
          return tmem_fixup(
            tiled_mma,
            work_tile_info,
            accumulators,
            acc_pipeline,
            acc_pipe_consumer_state,
            CopyOpT2R{}
          );
        }
      }
      else {
        // Simply perform fixup without TMEM loads when accumulators reside in RF
        constexpr uint32_t ThreadsForFixup = NumThreadsPerWarpGroup;
        constexpr uint32_t Offset = static_cast<int>(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
        constexpr uint32_t MaxNumNamedBarriers = 1;
        constexpr uint32_t BarrierIdx = 0;
        using BarrierManager = NamedBarrierManager<ThreadsForFixup, Offset, MaxNumNamedBarriers>;
        constexpr int NumAccumulatorMtx = IsComplex ? 2 : 1;

        UnderlyingStreamKScheduler::template fixup_helper<cute::remove_cvref_t<decltype(accumulators)>, BarrierManager>(
          params_.sk_params_, work_tile_info, accumulators, MaxNumNamedBarriers, BarrierIdx, NumAccumulatorMtx);
        return acc_pipe_consumer_state;
      }
    }
  }

  // Convert CTA-level work tile info to cluster-level tile coord
  CUTLASS_DEVICE
  auto
  work_tile_to_cluster_coord_mnkl(WorkTileInfo work_tile_info) const {
    typename UnderlyingScheduler::WorkTileInfo tmp{
      work_tile_info.M_idx,
      work_tile_info.N_idx,
      work_tile_info.L_idx,
      work_tile_info.is_valid()
    };
    return sm100_scheduler_.work_tile_to_cluster_coord_mnkl(tmp);
  }

private:
  CUTLASS_HOST_DEVICE
  WorkTileInfo invalid_work_tile() const {
    // Mark the work tile as invalid based on its having a 0 K tiles to comptue.
    // Set the M, N, and L indices to be outside of the range of valid tiles for the problem.
    return {
      static_cast<int32_t>(params_.sm100_params_.problem_tiles_m_) * params_.sm100_params_.divmod_cluster_shape_m_.divisor,
      static_cast<int32_t>(params_.sm100_params_.problem_tiles_n_) * params_.sm100_params_.divmod_cluster_shape_n_.divisor,
      0, // K_idx
      static_cast<int32_t>(params_.sm100_params_.problem_tiles_l_),
      0  // k_tile_count
    };
  }

  // Converts the work tile info returned by the SM100 scheduler to a linear index
  CUTLASS_DEVICE
  uint64_t
  to_linear_idx(
    InternalWorkTileInfo const& work_tile_info,
    Params const& params) {
    // The InternalWorkTileInfo returned from CLC query gives all CTAs in a cluster
    // the tile offset corresponding to the first CTA tile in the cluster tile assigned
    // to the cluster. Since the SM90 tile scheduler operates at CTA level, we must assign
    // each CTA its own tile when computing the linear ID to be used by the SM90
    // stream-K scheduler.
    auto start_cta_m_preferred_cluster = params.sk_params_.truncate_to_cluster_size_m(work_tile_info.M_idx);
    auto start_cta_n_preferred_cluster = params.sk_params_.truncate_to_cluster_size_n(work_tile_info.N_idx);
    uint64_t cluster_idx = gridDim.y * start_cta_m_preferred_cluster + start_cta_n_preferred_cluster;
    uint64_t sm_count = gridDim.x * gridDim.y;
    uint64_t wave_idx = work_tile_info.L_idx;

    auto cluster_start_linear_id = sm_count * wave_idx + cluster_idx;

    // Determine the offset of this CTA in the preferred cluster shape.
    // This calculation aims to accommodate both cases in which this CTA is part of a preferred cluster
    // and those in which it is part of a fallback cluster.
    //
    // The calculation is performed by computing the starting M and N index of the preferred cluster that
    // this CTA would be in, and then subtracting these from the true CTA M and N indexes.
    //
    // In the case where this CTA is part of a preferred cluster, the resulting offsets are equivalent
    // to those returned by cute::block_id_in_cluster();
    uint64_t cta_m_in_preferred_cluster = work_tile_info.M_idx - start_cta_m_preferred_cluster;
    uint64_t cta_n_in_preferred_cluster = work_tile_info.N_idx - start_cta_n_preferred_cluster;

    if (params.sk_params_.raster_order_ == RasterOrder::AlongN) {
      return cluster_start_linear_id + (params.sk_params_.divmod_cluster_shape_minor_.divisor * cta_n_in_preferred_cluster) + cta_m_in_preferred_cluster;
    }
    else {
      return cluster_start_linear_id + (params.sk_params_.divmod_cluster_shape_minor_.divisor * cta_m_in_preferred_cluster) + cta_n_in_preferred_cluster;
    }
  }

  // Converts the work tile info returned by the SM100 scheduler to a stream-K work tile info
  CUTLASS_DEVICE
  WorkTileInfo
  convert_work(InternalWorkTileInfo const& work_tile_info) {
    if (has_sk_work()) {
      current_work_linear_idx_ = to_linear_idx(work_tile_info, params_);
      auto work = UnderlyingStreamKScheduler::get_current_work_for_linear_idx(unit_iter_start_, current_work_linear_idx_, block_id_in_cluster_, params_.sk_params_);
      if (!work.is_valid()) {
        return invalid_work_tile();
      }
      return work;
    }
    else if (is_split_k()) {
      // Split-K offsets are returned directly by CLC query (rather than being
      // returned by the SM90 stream-K tile scheduler). CLC query returns
      // the first CTA tile of work for each CTA in a cluster, but later use of the
      // split-K work tile for fixup expect a CTA-offset tile. Thus, we need to offset
      // each CTA's M and N index by the CTA offset in the cluster.
      int32_t M_idx = work_tile_info.M_idx;
      int32_t N_idx = work_tile_info.N_idx;

      int L_idx, Split_idx;
      params_.sk_params_.divmod_splits_(L_idx, Split_idx, work_tile_info.L_idx);

      int additional_k_tiles = 0;
      int split_start_offset = params_.sk_params_.big_units_;

      if (Split_idx < params_.sk_params_.big_units_) {
        // Offsets for "big" units. One additional k iteration is performed,
        // and each split preceding us was a big unit, so we must increase
        // our split starting offset by our split ID (Split_idx).
        additional_k_tiles = 1;
        split_start_offset = Split_idx;
      }

      // Set up k iteration count and split starting iteration assuming the
      // iteration space is evenly split.
      uint32_t k_tiles = params_.sk_params_.divmod_k_tiles_per_sk_unit_.divisor;
      uint32_t K_idx = Split_idx * k_tiles;

      // Apply any fixup needed to handle residuals
      K_idx += split_start_offset;
      k_tiles += additional_k_tiles;

      // K_idx is even for each cta.
      //
      // * Example
      // 53 k_tiles per output tile
      // 10 k_tiles for normal size split
      // 11 k_tiles for start three big unit
      //
      // split 0 : K_idx = [0,  10], k_tiles = 11 -> K_idx = [0,  11], k_tiles = 12
      // split 1 : K_idx = [11, 21], k_tiles = 11 -> K_idx = [12, 21], k_tiles = 10
      // split 2 : K_idx = [22, 32], k_tiles = 11 -> K_idx = [22, 33], k_tiles = 12
      // split 3 : K_idx = [33, 42], k_tiles = 10 -> K_idx = [34, 42], k_tiles = 9 -> K_idx = [34, 43], k_tiles = 10
      // split 4 : K_idx = [43, 52], k_tiles = 10 -> K_idx = [44, 52], k_tiles = 9
      if (params_.sk_params_.ktile_start_alignment_count_ == 2u && K_idx % 2 != 0) {
        // If current cta K_idx not start from even, give up one k_tile
        K_idx += 1;
        k_tiles -= 1;
      }
      if (params_.sk_params_.ktile_start_alignment_count_ == 2u &&
          (K_idx + k_tiles) % 2 != 0 &&
          (K_idx + k_tiles) < params_.sk_params_.divmod_tiles_per_output_tile_.divisor) {
        // If next cta K_idx not start from even, acquire one k_tile
        k_tiles += 1;
      }

      return {
        M_idx,
        N_idx,
        static_cast<int32_t>(K_idx),
        static_cast<int32_t>(L_idx),
        k_tiles,
        k_tiles  // remaining iterations
      };
    }
    else {
      // Data-parallel case
      return {
        static_cast<int32_t>(work_tile_info.M_idx),
        static_cast<int32_t>(work_tile_info.N_idx),
        static_cast<int32_t>(0),                   // K_idx
        static_cast<int32_t>(work_tile_info.L_idx),
        static_cast<uint32_t>(params_.sk_params_.divmod_tiles_per_output_tile_.divisor),
        static_cast<uint32_t>(params_.sk_params_.divmod_tiles_per_output_tile_.divisor)
      };
    }
  }

  // Converts a WorkTileInfo struct to the WorkTileInfo representation
  // of the underlying SM100 scheduler.
  CUTLASS_HOST_DEVICE static
  InternalWorkTileInfo
  to_underlying_work_tile_info(WorkTileInfo const& work_tile_info) {
    return {
      work_tile_info.M_idx,
      work_tile_info.N_idx,
      work_tile_info.L_idx,
      work_tile_info.is_valid()
    };
  }

  // Returns whether the current parameters contain only data-parallel tiles
  CUTLASS_HOST_DEVICE
  bool
  is_dp_only() const {
    return params_.sk_params_.sk_units_ == 0 && params_.sk_params_.divmod_splits_.divisor == 1;
  }

  // Returns whether the current parameters are for a split-K decomposition
  CUTLASS_HOST_DEVICE
  bool
  is_split_k() const {
    return params_.sk_params_.divmod_splits_.divisor > 1;
  }

  // Returns whether the current parameters contain any stream-K work
  CUTLASS_HOST_DEVICE
  bool
  has_sk_work() const {
    return params_.sk_params_.sk_units_ > 0;
  }

  // Performs reduction across splits for a given output tile
  template <
    class TiledMma,
    class AccEngine,
    class AccLayout,
    class AccumulatorPipeline,
    class AccumulatorPipelineState,
    class CopyOpT2R
  >
  CUTLASS_DEVICE
  AccumulatorPipelineState
  tmem_fixup(
      TiledMma const& tiled_mma,
      WorkTileInfo const& work_tile_info,
      cute::Tensor<AccEngine, AccLayout>& accumulators,
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState acc_pipe_consumer_state,
      CopyOpT2R,
      uint32_t num_accumulator_mtx = 1,
      uint32_t idx_accumulator_mtx = 0) const {
    using namespace cute;
    static_assert(cute::is_tmem_v<AccEngine>, "Accumulator must be in TMEM");

    using ElementAccumulator = typename AccEngine::element_type;

    constexpr uint32_t ThreadsForFixup = NumThreadsPerWarpGroup;
    constexpr uint32_t Offset = static_cast<int>(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
    constexpr uint32_t MaxNumNamedBarriers = 1;
    constexpr uint32_t BarrierIdx = 0;
    using BarrierManager = NamedBarrierManager<ThreadsForFixup, Offset, MaxNumNamedBarriers>;

    // When accumulators reside in TMEM, perform TMEM -> RF loads before performing fixup,
    // and perform RF -> TMEM stores after fixup (when the split must compute the epilogue)
    auto dummy_gmem_workspace = make_tensor(
      make_gmem_ptr<ElementAccumulator>(nullptr),
      make_layout(take<0,2>(TileShape{}), GenRowMajor{})); // (TILE_M,TILE_N)

    auto dummy_gmem_buffer = tiled_mma.get_slice(0).partition_C(dummy_gmem_workspace); // (MMA,MMA_M,MMA_N)

    auto tmem_load = make_tmem_copy(CopyOpT2R{}, accumulators);
    auto tmem_store = make_tmem_copy(cute::TMEM::tmem_load_to_store(CopyOpT2R{}), accumulators);

    auto thr_tmem_load = tmem_load.get_slice(threadIdx.x % ThreadsForFixup);
    auto thr_tmem_store = tmem_store.get_slice(threadIdx.x % ThreadsForFixup);

    Tensor tCtAcc = thr_tmem_load.partition_S(accumulators);      // (TMEM_LOAD,TMEM_LOAD_MMA,TMEM_LOAD_M,TMEM_LOAD_N)
    Tensor tCgAcc = thr_tmem_load.partition_D(dummy_gmem_buffer); // (TMEM_LOAD,TMEM_LOAD_MMA,TMEM_LOAD_M,TMEM_LOAD_N)
    auto tCrAcc = make_tensor<ElementAccumulator>(shape(tCgAcc)); // (TMEM_LOAD,TMEM_LOAD_MMA,TMEM_LOAD_M,TMEM_LOAD_N)

    acc_pipeline.consumer_wait(acc_pipe_consumer_state);

    // Copy accumulators from tmem to rmem for reduction
    copy(tmem_load, tCtAcc, tCrAcc);

    bool should_compute_epilogue = compute_epilogue(work_tile_info);
    if (!should_compute_epilogue && (idx_accumulator_mtx == (num_accumulator_mtx - 1))) {
      // Splits that do not compute the epilogue must advance the accumulator pipeline
      cutlass::arch::fence_view_async_tmem_load();
      acc_pipeline.consumer_release(acc_pipe_consumer_state);
      ++acc_pipe_consumer_state;
    }

    // Perform fixup
    UnderlyingStreamKScheduler::template fixup_helper<decltype(tCrAcc), BarrierManager>(
      params_.sk_params_, work_tile_info, tCrAcc, MaxNumNamedBarriers, BarrierIdx, num_accumulator_mtx, idx_accumulator_mtx);

    if (should_compute_epilogue) {
      // Splits that compute the epilogue copy the reduced accumulators back to tmem for
      // the epilogue to compute on it
      copy(tmem_store, tCrAcc, tCtAcc);
    }

    return acc_pipe_consumer_state;
  }


  //
  // Members
  //

  UnderlyingScheduler sm100_scheduler_;
  Params params_;
  dim3 block_id_in_cluster_;
  uint64_t current_work_linear_idx_ = 0;
  uint32_t unit_iter_start_ = 0;

  // This might not be needed
  bool is_fallback_cluster_ = false;
};

///////////////////////////////////////////////////////////////////////////////

} // end namespace cutlass::gemm::kernel::detail
