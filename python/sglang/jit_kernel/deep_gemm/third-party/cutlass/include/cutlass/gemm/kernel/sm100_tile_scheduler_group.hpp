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
#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel::detail {

//////////////////// Blackwell Grouped Static Scheduler /////////////////////////

// This tile scheduler is a SM100 wrapper for scheduling by the SM90 Group tile scheduler.
// This helps to enable reusing SM90 group tile scheduling capability for SM100 kernels
// (e.g., support for CTA rasterization).

// For Grouped GEMM, most common use case have Problem Shapes for all groups only on device.
// Therefore, we don't how many tiles there will be for the scheduler to hand out.
// Hence, we have a SM90 style static group scheduler that launches the largest grid possible.
// If we had access to host-side problem shapes, one could to use it to figure out the grid shape
// and thereafter use CLC query (which can then be linearized and mapped to an appropriate tile coord).

template<class GroupProblemShape, int SchedulerPipelineStageCount>
class PersistentTileSchedulerSm100Group {

public:
  using UnderlyingScheduler = PersistentTileSchedulerSm90Group<GroupProblemShape, SchedulerPipelineStageCount>;
  using Params = PersistentTileSchedulerSm100GroupParams<GroupProblemShape>;
  using WorkTileInfo = typename UnderlyingScheduler::WorkTileInfo;
  using Arguments = typename UnderlyingScheduler::Arguments;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;

  using CLCResponse = WorkTileInfo;
  
  static constexpr bool IsDynamicPersistent = UnderlyingScheduler::IsDynamicPersistent;

private:
  UnderlyingScheduler scheduler_sm90;

public:
  template <class TileShape, class AtomThrShape, class ClusterShape>
  static Params
  to_underlying_arguments(
    GroupProblemShape problem_shapes,
    TileShape tile_shape_mnk,
    AtomThrShape atom_thr_shape_mnk,
    ClusterShape cluster_shape_mnk,
    KernelHardwareInfo const& hw_info,
    Arguments const& args,
    void* workspace = nullptr) {

    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);

    auto selected_cluster_shape = cutlass::detail::select_cluster_shape(cluster_shape_mnk, hw_info.cluster_shape);
    auto cta_shape = shape_div(tile_shape_mnk, atom_thr_shape_mnk); // For 2SM kernels, use CTA tile shape for the underlying scheduler

    dim3 problem_blocks = get_tiled_cta_shape_mnl(
      problem_shapes,
      hw_info,
      cta_shape, selected_cluster_shape);

    Params params;
    params.initialize(
      problem_blocks,
      problem_shapes,
      to_gemm_coord(cta_shape),
      to_gemm_coord(selected_cluster_shape),
      hw_info,
      args.max_swizzle_size,
      args.raster_order
    );

    return params;
  }

  static bool
  can_implement(Arguments const& args) {
    return true;
  }

  CUTLASS_DEVICE
  PersistentTileSchedulerSm100Group() { }

  CUTLASS_DEVICE
  PersistentTileSchedulerSm100Group(CLCResponse* clc_response_ptr, Params const& params)
    : scheduler_params(params),
      scheduler_sm90(params.params_sm90_, clc_response_ptr) { }

  CUTLASS_DEVICE
  PersistentTileSchedulerSm100Group(CLCResponse* clc_response_ptr, Params const& params, dim3 /* block_id_in_cluster */)
    : scheduler_params(params),
      scheduler_sm90(params.params_sm90_, clc_response_ptr) { }

  // Returns the initial work tile info that will be computed over
  template <typename ClusterShape>
  CUTLASS_DEVICE
  auto
  initial_work_tile_info(ClusterShape cluster_shape) {
    return scheduler_sm90.initial_work_tile_info(cluster_shape);
  }

  template<class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(GroupProblemShape const &problem_shapes, KernelHardwareInfo hw_info, BlockShape cta_shape, ClusterShape cluster_shape) {
    return UnderlyingScheduler::get_tiled_cta_shape_mnl(problem_shapes, hw_info, cta_shape, cluster_shape);
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class BlockShape, class AtomThrShape, class ClusterShape>
  CUTLASS_HOST_DEVICE
  static dim3
  get_grid_shape(
      Params const& params,
      GroupProblemShape const& problem_shapes,
      BlockShape cta_shape,
      [[maybe_unused]] AtomThrShape atom_thr_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo hw_info) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(
      problem_shapes,
      hw_info,
      cta_shape,
      cluster_shape);

    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    Arguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.params_sm90_.log_swizzle_size_;
    }
    args.raster_order = params.params_sm90_.raster_order_ == RasterOrder::AlongN ? RasterOrderOptions::AlongN : RasterOrderOptions::AlongM;

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      args.max_swizzle_size,
      args.raster_order,
      /* truncate_by_problem_size = */true,
      cute::is_static_v<ClusterShape> ? true : false
    );
  }

  CUTLASS_DEVICE
  static auto
  work_tile_to_cta_coord(WorkTileInfo work_tile_info) {
    // SM90 static scheduler implicitly handles CTA coord in a Cluster
    return make_coord(
      work_tile_info.M_idx,
      work_tile_info.N_idx,
      _,
      work_tile_info.L_idx
    );
  }

  template <typename CLCPipeline, typename CLCPipelineState>
  CUTLASS_DEVICE
  auto
  advance_to_next_work(
    CLCPipeline& clc_pipeline,
    CLCPipelineState clc_pipe_producer_state,
    uint32_t advance_count = 1) {

    return scheduler_sm90.advance_to_next_work(clc_pipeline, clc_pipe_producer_state, advance_count);
  }

  //
  // K Tile API
  //
  template <class ProblemShape, class TileShape, class Shape>
  CUTLASS_DEVICE
  auto
  get_k_tile_iterator(WorkTileInfo const& work_tile_info, ProblemShape problem_shape_MNKL, TileShape tile_shape, Shape) {
    auto k_tiles = cute::ceil_div(cute::get<2>(problem_shape_MNKL), cute::get<2>(tile_shape));
    return cute::make_coord_iterator(k_tiles);
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the Group tile scheduler, this is always true.
  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&, Params const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&) {
    return true;
  }

  // Returns whether fixup is needed for `work_tile_info`. None of the work units returned by
  // this scheduler require fixup, since none of the work units partition the reduction extent.
  CUTLASS_HOST_DEVICE
  static bool
  requires_fixup(Params const& params, WorkTileInfo const work_tile_info) {
    return false;
  }

  // Performs the reduction across splits for a given output tile. No fixup is required for
  // work units returned by this scheduler.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  void
  fixup(WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t, uint32_t = 1) const { }

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
      TiledMma const& ,
      WorkTileInfo const&,
      cute::Tensor<AccEngine, AccLayout>&,
      AccumulatorPipeline,
      AccumulatorPipelineState acc_pipe_consumer_state,
      CopyOpT2R) const {
    return acc_pipe_consumer_state;
  }

  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(Arguments const& args, ProblemShape problem_shape, KernelHardwareInfo const& hw_info, uint32_t, uint32_t = 1, uint32_t = 1) {
    return 0;
  }

  template <class ElementAccumulator, class ProblemShape, class TileShapeMNK, class AtomThrShape, class ClusterShape>
  static size_t
  get_workspace_size(Arguments const& args, ProblemShape problem_shape, TileShapeMNK, AtomThrShape, ClusterShape, KernelHardwareInfo const& hw_info,
      uint32_t reduction_warp_groups, uint32_t num_accumulator_mtxs = 1) {
    return 0;
  }

  template <class ProblemShape, class TileShape>
  CUTLASS_HOST_DEVICE
  static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape problem_shape_MNKL, TileShape tile_shape) {
    // All work units returned by this scheduler cover the entire K iteration
    // space of the output tile assigned to the work unit.
    return cute::size(cute::ceil_div(cute::get<2>(problem_shape_MNKL), cute::get<2>(tile_shape)));
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const&) {
    // All work units returned by this scheduler start from K tile 0
    return 0u;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(Arguments const&, void*, cudaStream_t, ProblemShape const&, KernelHardwareInfo const&, uint32_t, uint32_t = 1, uint32_t = 1, CudaHostAdapter *cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ElementAccumulator, class ProblemShape, class TileShapeMNK, class AtomThrShape, class ClusterShape>
  static cutlass::Status
  initialize_workspace(Arguments const&, void*, cudaStream_t, ProblemShape const&, TileShapeMNK, AtomThrShape, ClusterShape, KernelHardwareInfo const&,
      uint32_t, uint32_t = 1, CudaHostAdapter *cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  // Kernel helper function to get next CLC ID
  template <class CLCPipeline, class CLCPipelineState>
  CUTLASS_DEVICE
  auto
  fetch_next_work(
    WorkTileInfo work_tile_info,
    CLCPipeline& clc_pipeline,
    CLCPipelineState clc_pipe_consumer_state) {

    return scheduler_sm90.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
  }

private:
  //
  // Methods
  //
  [[nodiscard]] CUTLASS_DEVICE
  static CLCResponse
  load_query_response(uint32_t smem_ptr) {
    return UnderlyingScheduler::load_query_response(smem_ptr);
  }
  //
  // Storage
  //
  Params scheduler_params;
};

///////////////////////////////////////////////////////////////////////////////

} // end namespace cutlass::gemm::kernel::detail
