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

#include "cutlass/fast_math.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"

namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////

// Persistent Thread Block (TB) scheduler
template <class GroupProblemShape, int SchedulerPipelineStageCount>
class PersistentTileSchedulerSm90Group {
  //
  // Data members
  //

private:
  uint64_t current_work_linear_idx_ = 0;
  uint64_t total_grid_size_ = 0;

  // Tracking current group, its starting linear idx and total tiles
  struct GroupInfo {
    int group_idx = 0;
    uint64_t start_linear_idx = 0;
    uint64_t total_tiles = 0;
    uint64_t problem_blocks_along_raster_order = 0;
  } current_group_info_;

public:
  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    int32_t is_valid_tile = 0;

    CUTLASS_HOST_DEVICE
    bool
    is_valid() const {
      return is_valid_tile != 0;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo
    invalid_work_tile() {
      return {-1, -1, -1, 0};
    }

    CUTLASS_HOST_DEVICE
    bool
    is_final_split(uint32_t k_tiles_per_output_tile) const {
      return true;
    }

    CUTLASS_HOST_DEVICE
    int32_t
    reduction_subtile_idx() const {
      return -1;
    }
  };

  using ProblemShape = typename GroupProblemShape::UnderlyingProblemShape;
  using Params = PersistentTileSchedulerSm90GroupParams<GroupProblemShape>;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  static constexpr bool IsDynamicPersistent = false;

  // We need to hard code the number of stages here since the scheduling is static
  // and it can benefit from a larger number of stages without worrying about imbalances.

  using Pipeline = PipelineAsync<SchedulerPipelineStageCount>;

  // Call out the types here to work around a bug in MSVC.

  // using PipelineStorage = typename Pipeline::SharedStorage;
  // using PipelineState = typename Pipeline::PipelineState;
  using PipelineStorage = cutlass::PipelineDetail::PipelineAsyncSharedStorage<SchedulerPipelineStageCount>;
  using PipelineState = cutlass::PipelineDetail::PipelineAsyncPipelineState<SchedulerPipelineStageCount>;

  using ThrottlePipeline = PipelineEmpty;
  using ThrottlePipelineStorage = typename PipelineEmpty::SharedStorage;
  using SchedulerResponse = WorkTileInfo;

  class SharedStorage {
  public:
    CUTLASS_DEVICE PipelineStorage pipeline() { return pipeline_; }
    // Pipeline throttle is not needed here as the scheduling is not dynamic.
    CUTLASS_DEVICE ThrottlePipelineStorage throttle_pipeline() { return ThrottlePipelineStorage{}; }
    CUTLASS_DEVICE SchedulerResponse* data() { return data_; }

  private: 
    alignas(16) PipelineStorage pipeline_;
    alignas(16) SchedulerResponse data_[SchedulerPipelineStageCount];
  };

  struct Arguments {
    int max_swizzle_size = 1;
    // Not applying Heuristics for Grouped problems, since largest dimension can change per group
    RasterOrderOptions raster_order = RasterOrderOptions::AlongM;
  };

  // Sink scheduler params as a member
  Params scheduler_params;
  SchedulerResponse *response_ptr_ = nullptr;
  ProblemShape cached_problem_shapes_[2];

  //
  // Methods
  //

  template <class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
    GroupProblemShape problem_shapes,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo const& hw_info,
    Arguments const& arguments,
    [[maybe_unused]] void* workspace=nullptr,
    [[maybe_unused]] const uint32_t epilogue_subtile = 1,
    [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u
    ) {

    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(
      problem_shapes,
      hw_info,
      tile_shape, cluster_shape);

    Params params;
    params.initialize(
      problem_blocks,
      problem_shapes,
      to_gemm_coord(tile_shape),
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size, 
      arguments.raster_order
    );

    return params;
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class TileShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    [[maybe_unused]] Params const& params,
    GroupProblemShape const& problem_shapes,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info,
    Arguments arguments,
    bool truncate_by_problem_size=true) {

    dim3 problem_blocks = get_tiled_cta_shape_mnl(
      problem_shapes,
      hw_info,
      tile_shape, cluster_shape);

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order,
      /* truncate_by_problem_size = */true
    );
  }

  // Given the inputs, computes the total number of output blocks this problem will compute over
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template<class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(GroupProblemShape const& problem_shapes, KernelHardwareInfo hw_info, BlockShape cta_shape, ClusterShape cluster_shape) {
    int groups = problem_shapes.groups();
    uint32_t total_ctas = 0;
    uint32_t cta_in_N_dim = 1; // We linearize the blocks across all the problems here

    // If host problem shapes are not provided.
    if (!problem_shapes.is_host_problem_shape_available()) {
      total_ctas = hw_info.sm_count;
    }
    // If host problem shapes are provided, make a better decision about possibility to launch smaller grid.
    else {
      for (int group = 0; group < groups; group++) {
        auto ctas_along_m = cute::size(cute::ceil_div(cute::shape<0>(problem_shapes.get_host_problem_shape(group)), cute::shape<0>(cta_shape)));
        auto ctas_along_n = cute::size(cute::ceil_div(cute::shape<1>(problem_shapes.get_host_problem_shape(group)), cute::shape<1>(cta_shape)));
        auto problem_blocks_m = round_up(ctas_along_m, cute::get<0>(cluster_shape));
        auto problem_blocks_n = round_up(ctas_along_n, cute::get<1>(cluster_shape));
        total_ctas += problem_blocks_m * problem_blocks_n;
      }
    }

    return Params::get_tiled_cta_shape_mnl(
      to_gemm_coord(cluster_shape),
      total_ctas, cta_in_N_dim
    );
  }

  static bool
  can_implement(Arguments const& args) {
    return true;
  }

  PersistentTileSchedulerSm90Group() = default;

  CUTLASS_DEVICE explicit PersistentTileSchedulerSm90Group(Params const& params_, SchedulerResponse* response_ptr) : scheduler_params(params_), response_ptr_(response_ptr) {
    // MSVC requires protecting use of CUDA-specific nonstandard syntax,
    // like blockIdx and gridDim, with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
    if (scheduler_params.raster_order_ == RasterOrder::AlongN) {
      current_work_linear_idx_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * uint64_t(gridDim.x);
    }
    else {
      current_work_linear_idx_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y) + uint64_t(blockIdx.y);
    }

    int lane_idx = canonical_lane_idx();
    if (lane_idx < params_.problem_shapes_.groups()) {
      cached_problem_shapes_[1] = params_.problem_shapes_.get_problem_shape(lane_idx);
    }

    total_grid_size_ = uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z);
    uint64_t ctas_along_m, ctas_along_n;
    ProblemShape problem_shape = params_.problem_shapes_.get_problem_shape(0);
    if (is_tuple<decltype(cute::shape<0>(problem_shape))>::value ||
        is_tuple<decltype(cute::shape<1>(problem_shape))>::value) {
      ctas_along_m = cute::size(cute::ceil_div(cute::shape<0>(problem_shape), scheduler_params.cta_shape_.m()));
      ctas_along_n = cute::size(cute::ceil_div(cute::shape<1>(problem_shape), scheduler_params.cta_shape_.n()));
    }
    else {
      ctas_along_m = scheduler_params.divmod_cta_shape_m_.divide(cute::shape<0>(problem_shape) +  scheduler_params.divmod_cta_shape_m_.divisor - 1);
      ctas_along_n = scheduler_params.divmod_cta_shape_n_.divide(cute::shape<1>(problem_shape) +  scheduler_params.divmod_cta_shape_n_.divisor - 1);
    }
    auto problem_blocks_m = round_up(ctas_along_m, (1 << params_.log_swizzle_size_) * params_.cluster_shape_.m());
    auto problem_blocks_n = round_up(ctas_along_n, (1 << params_.log_swizzle_size_) * params_.cluster_shape_.n());
    current_group_info_.total_tiles = problem_blocks_m * problem_blocks_n;
    current_group_info_.problem_blocks_along_raster_order = params_.raster_order_ == RasterOrder::AlongN ? problem_blocks_n : problem_blocks_m;

#else
    CUTLASS_ASSERT(false && "This line should never be reached");
#endif
  }

  // get work_idx_m, work_idx_n from linear_idx while applying swizzle
  template<class WorkTileInfo, class GroupInfo, class ProblemShape, class RasterOrder>
  static
  CUTLASS_DEVICE
  WorkTileInfo
  get_work_idx_m_and_n(
      uint64_t linear_idx,
      GroupInfo& group_info,
      GroupProblemShape &problem_shapes,
      ProblemShape (&cached_problem_shapes)[2],
      GemmCoord cta_shape,
      GemmCoord cluster_shape,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cta_shape_m,
      FastDivmodU64 const& divmod_cta_shape_n,
      int32_t log_swizzle_size, 
      RasterOrder raster_order) {

    int32_t valid_tile = 1;

    // Use a warp to "speculatively" check if the work tile maps to the next 32 groups
    int lane_idx = canonical_lane_idx();
    int total_problem_groups = problem_shapes.groups();

    if (linear_idx >= group_info.total_tiles + group_info.start_linear_idx) {
      group_info.group_idx += lane_idx;
      for ( ; ; group_info.group_idx += NumThreadsPerWarp) {
        cached_problem_shapes[0] = cached_problem_shapes[1];
        if (group_info.group_idx + NumThreadsPerWarp < total_problem_groups) {
          cached_problem_shapes[1] = problem_shapes.get_problem_shape(group_info.group_idx + NumThreadsPerWarp);
        }
        if (group_info.group_idx < total_problem_groups) {
          uint64_t ctas_along_m, ctas_along_n;
          if (is_tuple<decltype(cute::shape<0>(cached_problem_shapes[0]))>::value ||
              is_tuple<decltype(cute::shape<1>(cached_problem_shapes[0]))>::value) {
            ctas_along_m = cute::size(cute::ceil_div(cute::shape<0>(cached_problem_shapes[0]), cta_shape.m()));
            ctas_along_n = cute::size(cute::ceil_div(cute::shape<1>(cached_problem_shapes[0]), cta_shape.n()));
          }
          else {
            ctas_along_m = divmod_cta_shape_m.divide(cute::shape<0>(cached_problem_shapes[0]) +  divmod_cta_shape_m.divisor - 1);
            ctas_along_n = divmod_cta_shape_n.divide(cute::shape<1>(cached_problem_shapes[0]) +  divmod_cta_shape_n.divisor - 1);
          }
          auto problem_blocks_m = round_up(ctas_along_m, (1 << log_swizzle_size) * cluster_shape.m());
          auto problem_blocks_n = round_up(ctas_along_n, (1 << log_swizzle_size) * cluster_shape.n());
          group_info.problem_blocks_along_raster_order = raster_order == RasterOrder::AlongN ? problem_blocks_n : problem_blocks_m;
          group_info.total_tiles = problem_blocks_m * problem_blocks_n;
        } else {
          group_info.total_tiles = INT_MAX;
        }

        auto curr_total_tiles = group_info.total_tiles;

        // Calculate prefix sum for start_linear_idx.
        #pragma unroll
        for (int i = 1; i < NumThreadsPerWarp; i *= 2) {
          auto n = __shfl_up_sync(0xffffffff, curr_total_tiles, i);
          curr_total_tiles = lane_idx >= i ? curr_total_tiles + n : curr_total_tiles;
        }
        group_info.start_linear_idx += curr_total_tiles - group_info.total_tiles;

        uint32_t thread_succeed = __ballot_sync(0xffffffff, linear_idx < group_info.start_linear_idx + group_info.total_tiles);
        if (thread_succeed) {
          // Use the first succeeding thread.
          int first_succeeding_thread = __ffs(thread_succeed) - 1;
          group_info.group_idx = __shfl_sync(0xffffffff, group_info.group_idx, first_succeeding_thread);
          group_info.start_linear_idx = __shfl_sync(0xffffffff, group_info.start_linear_idx, first_succeeding_thread);
          group_info.total_tiles = __shfl_sync(0xffffffff, group_info.total_tiles, first_succeeding_thread);
          group_info.problem_blocks_along_raster_order = __shfl_sync(0xffffffff, group_info.problem_blocks_along_raster_order, first_succeeding_thread);
          if (group_info.group_idx + lane_idx < total_problem_groups) {
            cached_problem_shapes[1] = problem_shapes.get_problem_shape(group_info.group_idx + lane_idx);
          }
          break;
        }
        // Update the start_linear_idx for all threads so that they're ready for the next iteration.
        group_info.start_linear_idx = __shfl_sync(0xffffffff, group_info.start_linear_idx + group_info.total_tiles, NumThreadsPerWarp - 1);
      }
    }

    if (group_info.group_idx >= total_problem_groups) {
      return WorkTileInfo::invalid_work_tile();
    }

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    uint64_t blk_per_grid_dim = divmod_cluster_shape_minor.divide(linear_idx - group_info.start_linear_idx);
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);

    // With static schedulers, we launch grid such that all cluster are linear (1-D) order, i.e., 
    // there can only be one cluster in the minor dimension. get_grid_shape() in scheduler params
    // put cluster_shape.m/n() as the minor dimension based on raster order AlongN/M resp.
    // Therefore, the offset of a CTA (inside a cluster) in the minor dimension can be directly be 
    // inferred by the blockIdx along the minor dimension.
    if (raster_order == RasterOrder::AlongN) {
      cluster_minor_offset = blockIdx.x;
    }
    else {
      cluster_minor_offset = blockIdx.y;
    }

    uint64_t cluster_idx_minor, cluster_idx_major;
    
    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    uint64_t curr_group_cluster_blk_major = divmod_cluster_shape_major.divide(group_info.problem_blocks_along_raster_order);

    cluster_idx_minor_div_swizzle = extra / curr_group_cluster_blk_major;
    cluster_idx_major = extra % curr_group_cluster_blk_major;

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;

    auto minor_work_idx = static_cast<int32_t>(cluster_idx_minor * divmod_cluster_shape_minor.divisor + 
                                               cluster_minor_offset);
    auto major_work_idx = static_cast<int32_t>(cluster_idx_major * divmod_cluster_shape_major.divisor + 
                                               cluster_major_offset);

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx, group_info.group_idx, valid_tile};
    }
    else {
      return {major_work_idx, minor_work_idx, group_info.group_idx, valid_tile}; 
    }
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work_for_linear_idx(uint64_t linear_idx) {
    if (scheduler_params.pre_processed_problem_shapes && linear_idx >= scheduler_params.blocks_across_problem_) {
      return WorkTileInfo::invalid_work_tile();
    }
    return get_work_idx_m_and_n<WorkTileInfo>(
              linear_idx,
              current_group_info_,
              scheduler_params.problem_shapes_,
              cached_problem_shapes_,
              scheduler_params.cta_shape_,
              scheduler_params.cluster_shape_,
              scheduler_params.divmod_cluster_shape_major_,
              scheduler_params.divmod_cluster_shape_minor_,
              scheduler_params.divmod_cta_shape_m_,
              scheduler_params.divmod_cta_shape_n_,
              scheduler_params.log_swizzle_size_, 
              scheduler_params.raster_order_);
  }
  template <typename TileSchedulerPipeline, typename TileSchedulerPipelineState>
  CUTLASS_DEVICE
  auto
  advance_to_next_work(
    TileSchedulerPipeline& scheduler_pipeline,
    TileSchedulerPipelineState scheduler_pipe_producer_state,
    uint32_t advance_count = 1) {

    current_work_linear_idx_ += total_grid_size_ * uint64_t(advance_count);
    auto work_tile = get_current_work_for_linear_idx(current_work_linear_idx_);
    scheduler_pipeline.producer_acquire(scheduler_pipe_producer_state);
    if (cute::elect_one_sync()) {
      response_ptr_[scheduler_pipe_producer_state.index()] = work_tile;
      cutlass::arch::fence_view_async_shared();
      scheduler_pipeline.producer_commit(scheduler_pipe_producer_state);
    }
    return cute::make_tuple(work_tile, true);
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the basic tile scheduler, this is always true.
  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const&, Params const&) {
    return true;
  }

  // Performs the reduction across splits for a given output tile. Since this scheduler does
  // not split output tiles, no reduction is needed.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  fixup(Params const&, WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t) {}

  // Returns whether the current WorkTileInfo passed in should continue to be used. Since
  // this scheduler only schedules work in units of single, full output tiles, the WorkTileInfo
  // passed in should not be used after having been processed.
  CUTLASS_DEVICE
  static bool
  continue_current_work(WorkTileInfo&) {
    return false;
  }

  // The basic tile scheduler does not require any additional workspace
  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1, uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(Arguments const&, void*, cudaStream_t, ProblemShape, KernelHardwareInfo const&,
    uint32_t, const uint32_t = 1, uint32_t = 1, CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  template <class ProblemShape_MNKL, class TileShape>
  CUTLASS_HOST_DEVICE
  static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape_MNKL problem_shape, TileShape tile_shape) {
    // All work units returned by this scheduler cover the entire K iteration
    // space of the output tile assigned to the work unit.
    return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const&) {
    // All work units returned by this scheduler start from K tile 0
    return 0u;
  }

  CUTLASS_DEVICE
  static bool
  need_separate_reduction(Params const& params) {
    return false;
  }

  CUTLASS_DEVICE
  bool
  is_work_tile_for_reduction(WorkTileInfo const& work_tile_info, Params const& params) {
    return false;
  }

  CUTLASS_DEVICE
  uint32_t
  epilgoue_subtile_idx(WorkTileInfo const& work_tile_info, Params const& params) const {
    return 0;
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE
  void
  separate_reduction(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx) {
  }

  // Shares the accumulator set with peers in the global workspace
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  share(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx) {
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

  // Kernel helper function to get next work tile
  template <typename TileSchedulerPipeline, typename TileSchedulerPipelineState>
  CUTLASS_DEVICE
  auto
  fetch_next_work(
    WorkTileInfo work_tile_info,
    TileSchedulerPipeline& scheduler_pipeline,
    TileSchedulerPipelineState scheduler_pipe_consumer_state) {

    if (continue_current_work(work_tile_info)) {
      return cute::make_tuple(work_tile_info, true);
    }
    scheduler_pipeline.consumer_wait(scheduler_pipe_consumer_state);
    auto work_tile = response_ptr_[scheduler_pipe_consumer_state.index()];
    cutlass::arch::fence_view_async_shared();
    scheduler_pipeline.consumer_release(scheduler_pipe_consumer_state);

    return cute::make_tuple(work_tile, true);
  }
  
  // Returns the initial work tile info that will be computed over
  template <class ClusterShape>
  CUTLASS_DEVICE
  auto
  initial_work_tile_info(ClusterShape) {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }
};

} // namespace cutlass::gemm::kernel::detail
