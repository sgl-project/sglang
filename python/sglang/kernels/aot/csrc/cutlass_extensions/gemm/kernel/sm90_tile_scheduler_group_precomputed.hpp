/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cute/arch/cluster_sm90.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/pipeline/pipeline.hpp"

namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////

struct PrecomputedGroupWorkTile {
  static constexpr uint64_t ChannelBits = 20;
  static constexpr uint64_t TokenBits = 24;
  static constexpr uint64_t ExpertBits = 19;
  static constexpr uint64_t ChannelMask = (uint64_t(1) << ChannelBits) - 1;
  static constexpr uint64_t TokenMask = (uint64_t(1) << TokenBits) - 1;
  static constexpr uint64_t ExpertMask = (uint64_t(1) << ExpertBits) - 1;
  static constexpr uint64_t TokenShift = ChannelBits;
  static constexpr uint64_t ExpertShift = ChannelBits + TokenBits;
  static constexpr uint64_t Invalid = ~uint64_t(0);

  CUTLASS_HOST_DEVICE
  static bool fits(uint64_t channel_idx, uint64_t token_idx, uint64_t expert_idx) {
    return channel_idx <= ChannelMask && token_idx <= TokenMask && expert_idx <= ExpertMask;
  }

  CUTLASS_DEVICE
  static uint64_t pack(uint64_t channel_idx, uint64_t token_idx, uint64_t expert_idx) {
    if (!fits(channel_idx, token_idx, expert_idx)) {
      asm volatile("trap;");
    }

    return uint64_t(channel_idx) | (uint64_t(token_idx) << TokenShift) | (uint64_t(expert_idx) << ExpertShift);
  }

  CUTLASS_DEVICE
  static bool is_invalid(uint64_t packed) {
    return packed == Invalid;
  }

  CUTLASS_DEVICE
  static int32_t channel_idx(uint64_t packed) {
    return static_cast<int32_t>(packed & ChannelMask);
  }

  CUTLASS_DEVICE
  static int32_t token_idx(uint64_t packed) {
    return static_cast<int32_t>((packed >> TokenShift) & TokenMask);
  }

  CUTLASS_DEVICE
  static int32_t expert_idx(uint64_t packed) {
    return static_cast<int32_t>((packed >> ExpertShift) & ExpertMask);
  }
};

template <bool ChunkMajorWorkMap>
struct PrecomputedWorkMapStride {};

template <>
struct PrecomputedWorkMapStride<true> {
  uint32_t precomputed_work_tiles_per_worker = 0;
};

// Persistent Thread Block (TB) scheduler
template <class GroupProblemShape, int SchedulerPipelineStageCount, bool ChunkMajorWorkMap = false>
class PersistentTileSchedulerSm90GroupPrecomputed {
  //
  // Data members
  //

 private:
  using WorkLinearIdx = uint32_t;
  WorkLinearIdx current_work_linear_idx_ = 0;
  WorkLinearIdx total_grid_size_ = 0;

 public:
  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    int32_t is_valid_tile = 0;

    CUTLASS_HOST_DEVICE
    bool is_valid() const {
      return is_valid_tile != 0;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo invalid_work_tile() {
      return {-1, -1, -1, 0};
    }

    CUTLASS_HOST_DEVICE
    bool is_final_split(uint32_t k_tiles_per_output_tile) const {
      return true;
    }

    CUTLASS_HOST_DEVICE
    int32_t reduction_subtile_idx() const {
      return -1;
    }
  };

  using ProblemShape = typename GroupProblemShape::UnderlyingProblemShape;
  using ParamsBase = PersistentTileSchedulerSm90GroupParams<GroupProblemShape>;
  struct Params : ParamsBase, PrecomputedWorkMapStride<ChunkMajorWorkMap> {
    uint64_t const* precomputed_work_tiles_ = nullptr;

    void initialize_precomputed(
        dim3 problem_blocks,
        GemmCoord cluster_shape,
        KernelHardwareInfo const& hw_info,
        int max_swizzle_size,
        typename ParamsBase::RasterOrderOptions raster_order_option) {
      CUTLASS_UNUSED(hw_info);

      auto problem_blocks_m = round_up(problem_blocks.x, cluster_shape.m());
      auto problem_blocks_n = round_up(problem_blocks.y, cluster_shape.n());

      this->blocks_across_problem_ = problem_blocks.x * problem_blocks.y * problem_blocks.z;
      this->pre_processed_problem_shapes = true;
      CUTLASS_UNUSED(max_swizzle_size);
      this->raster_order_ =
          ParamsBase::get_rasterization_order(problem_blocks_m, problem_blocks_n, raster_order_option);

      this->cluster_shape_ = cluster_shape;
    }
  };
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
    CUTLASS_DEVICE PipelineStorage pipeline() {
      return pipeline_;
    }
    // Pipeline throttle is not needed here as the scheduling is not dynamic.
    CUTLASS_DEVICE ThrottlePipelineStorage throttle_pipeline() {
      return ThrottlePipelineStorage{};
    }
    CUTLASS_DEVICE SchedulerResponse* data() {
      return data_;
    }

   private:
    alignas(16) PipelineStorage pipeline_;
    alignas(16) SchedulerResponse data_[SchedulerPipelineStageCount];
  };

  struct Arguments : PrecomputedWorkMapStride<ChunkMajorWorkMap> {
    int max_swizzle_size = 1;
    // Not applying Heuristics for Grouped problems, since largest dimension can change per group
    RasterOrderOptions raster_order = RasterOrderOptions::AlongM;
    uint64_t const* precomputed_work_tiles = nullptr;
  };

  // Sink scheduler params as a member
  Params scheduler_params;
  void* response_ptr_ = nullptr;

  //
  // Methods
  //

  template <class TileShape, class ClusterShape>
  static Params to_underlying_arguments(
      GroupProblemShape problem_shapes,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo const& hw_info,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace = nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {
    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shapes, hw_info, tile_shape, cluster_shape);

    CUTLASS_ASSERT(arguments.precomputed_work_tiles != nullptr);
    Params params;
    params.initialize_precomputed(
        problem_blocks, to_gemm_coord(cluster_shape), hw_info, arguments.max_swizzle_size, RasterOrderOptions::AlongM);
    params.precomputed_work_tiles_ = arguments.precomputed_work_tiles;
    if constexpr (ChunkMajorWorkMap) {
      params.precomputed_work_tiles_per_worker = arguments.precomputed_work_tiles_per_worker;
    }

    return params;
  }

  // Given the inputs, computes the physical grid we should launch.
  template <class TileShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3 get_grid_shape(
      [[maybe_unused]] Params const& params,
      GroupProblemShape const& problem_shapes,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo hw_info,
      Arguments arguments,
      bool truncate_by_problem_size = true) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shapes, hw_info, tile_shape, cluster_shape);

    return Params::get_grid_shape(
        problem_blocks,
        to_gemm_coord(cluster_shape),
        hw_info,
        arguments.max_swizzle_size,
        RasterOrderOptions::AlongM,
        /* truncate_by_problem_size = */ true);
  }

  // Given the inputs, computes the total number of output blocks this problem will compute over
  // Note that this is only the logical size of our grid, not the physical grid we will actually
  // launch.
  template <class BlockShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3 get_tiled_cta_shape_mnl(
      GroupProblemShape const& problem_shapes,
      KernelHardwareInfo hw_info,
      BlockShape cta_shape,
      ClusterShape cluster_shape) {
    int groups = problem_shapes.groups();
    uint32_t total_ctas = 0;
    uint32_t cta_in_N_dim = 1;  // We linearize the blocks across all the problems here

    // If host problem shapes are not provided.
    if (!problem_shapes.is_host_problem_shape_available()) {
      total_ctas = hw_info.sm_count;
    }
    // If host problem shapes are provided, make a better decision about possibility to launch
    // smaller grid.
    else {
      for (int group = 0; group < groups; group++) {
        auto ctas_along_m = cute::size(
            cute::ceil_div(cute::shape<0>(problem_shapes.get_host_problem_shape(group)), cute::shape<0>(cta_shape)));
        auto ctas_along_n = cute::size(
            cute::ceil_div(cute::shape<1>(problem_shapes.get_host_problem_shape(group)), cute::shape<1>(cta_shape)));
        if (ctas_along_m <= 0) ctas_along_m = 1;
        if (ctas_along_n <= 0) ctas_along_n = 1;
        auto problem_blocks_m = round_up(ctas_along_m, cute::get<0>(cluster_shape));
        auto problem_blocks_n = round_up(ctas_along_n, cute::get<1>(cluster_shape));
        total_ctas += problem_blocks_m * problem_blocks_n;
      }
    }

    return Params::get_tiled_cta_shape_mnl(to_gemm_coord(cluster_shape), total_ctas, cta_in_N_dim);
  }

  static bool can_implement(Arguments const& args) {
    bool implementable = args.precomputed_work_tiles != nullptr;
    if constexpr (ChunkMajorWorkMap) {
      implementable &= args.precomputed_work_tiles_per_worker > 0;
    }
    return implementable;
  }

  PersistentTileSchedulerSm90GroupPrecomputed() = default;

  // Note: constructing this tile scheduler can touch global memory that was
  // written to by the prior kernel.
  CUTLASS_DEVICE explicit PersistentTileSchedulerSm90GroupPrecomputed(Params const& params_)
      : scheduler_params(params_) {
    // MSVC requires protecting use of CUDA-specific nonstandard syntax,
    // like blockIdx and gridDim, with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
    CUTLASS_ASSERT(scheduler_params.precomputed_work_tiles_ != nullptr);
    WorkLinearIdx const worker_idx = WorkLinearIdx(blockIdx.x) * WorkLinearIdx(gridDim.y) + WorkLinearIdx(blockIdx.y) +
                                     WorkLinearIdx(blockIdx.z) * WorkLinearIdx(gridDim.x) * WorkLinearIdx(gridDim.y);
    if constexpr (ChunkMajorWorkMap) {
      current_work_linear_idx_ = worker_idx * scheduler_params.precomputed_work_tiles_per_worker;
      total_grid_size_ = 1;
    } else {
      current_work_linear_idx_ = worker_idx;
      total_grid_size_ = WorkLinearIdx(gridDim.x) * WorkLinearIdx(gridDim.y) * WorkLinearIdx(gridDim.z);
    }

#else
    CUTLASS_ASSERT(false && "This line should never be reached");
#endif
  }

  CUTLASS_DEVICE explicit PersistentTileSchedulerSm90GroupPrecomputed(
      Params const& params_, SchedulerResponse* response_ptr)
      : scheduler_params(params_), response_ptr_(response_ptr) {
    // MSVC requires protecting use of CUDA-specific nonstandard syntax,
    // like blockIdx and gridDim, with __CUDA_ARCH__.
#if defined(__CUDA_ARCH__)
    CUTLASS_ASSERT(scheduler_params.precomputed_work_tiles_ != nullptr);
    WorkLinearIdx const worker_idx = WorkLinearIdx(blockIdx.x) * WorkLinearIdx(gridDim.y) + WorkLinearIdx(blockIdx.y) +
                                     WorkLinearIdx(blockIdx.z) * WorkLinearIdx(gridDim.x) * WorkLinearIdx(gridDim.y);
    if constexpr (ChunkMajorWorkMap) {
      current_work_linear_idx_ = worker_idx * scheduler_params.precomputed_work_tiles_per_worker;
      total_grid_size_ = 1;
    } else {
      current_work_linear_idx_ = worker_idx;
      total_grid_size_ = WorkLinearIdx(gridDim.x) * WorkLinearIdx(gridDim.y) * WorkLinearIdx(gridDim.z);
    }

#else
    CUTLASS_ASSERT(false && "This line should never be reached");
#endif
  }

  CUTLASS_DEVICE
  WorkTileInfo get_current_work() {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE
  WorkTileInfo get_current_work_for_linear_idx(WorkLinearIdx linear_idx) {
    return get_precomputed_work_tile(linear_idx, scheduler_params.precomputed_work_tiles_);
  }

  CUTLASS_DEVICE
  static WorkTileInfo get_precomputed_work_tile(uint64_t linear_idx, uint64_t const* precomputed_work_tiles) {
    uint64_t const packed = __ldg(precomputed_work_tiles + linear_idx);
    if (PrecomputedGroupWorkTile::is_invalid(packed)) {
      return WorkTileInfo::invalid_work_tile();
    }

    return {
        PrecomputedGroupWorkTile::channel_idx(packed),
        PrecomputedGroupWorkTile::token_idx(packed),
        PrecomputedGroupWorkTile::expert_idx(packed),
        1};
  }

  template <
      typename TileSchedulerPipeline,
      typename TileSchedulerPipelineState,
      typename CallbackBeforeCommit = WorkTileInfo (*)(WorkTileInfo)>
  CUTLASS_DEVICE auto advance_to_next_work(
      TileSchedulerPipeline& scheduler_pipeline,
      TileSchedulerPipelineState scheduler_pipe_producer_state,
      uint32_t advance_count = 1,
      CallbackBeforeCommit callback_before_commit = [](WorkTileInfo info) { return info; }) {
    current_work_linear_idx_ += total_grid_size_ * WorkLinearIdx(advance_count);
    auto work_tile = get_current_work_for_linear_idx(current_work_linear_idx_);
    using WorkTileWithCallbackInfo = decltype(callback_before_commit(work_tile));
    WorkTileWithCallbackInfo work_tile_with_callback_info = work_tile;
    scheduler_pipeline.producer_acquire(scheduler_pipe_producer_state);
    if (work_tile_with_callback_info.is_valid()) {
      work_tile_with_callback_info = callback_before_commit(work_tile);
    }

    if (cute::elect_one_sync()) {
      reinterpret_cast<WorkTileWithCallbackInfo*>(response_ptr_)[scheduler_pipe_producer_state.index()] =
          work_tile_with_callback_info;
      cutlass::arch::fence_view_async_shared();
      scheduler_pipeline.producer_commit(scheduler_pipe_producer_state);
    }
    return cute::make_tuple(work_tile_with_callback_info, true);
  }

  CUTLASS_DEVICE
  void advance_to_next_work() {
    current_work_linear_idx_ += total_grid_size_;
  }

  CUTLASS_DEVICE
  void advance_to_next_work(uint32_t advance_count) {
    current_work_linear_idx_ += total_grid_size_ * WorkLinearIdx(advance_count);
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the basic tile scheduler, this is always true.
  CUTLASS_HOST_DEVICE
  static bool compute_epilogue(WorkTileInfo const&, Params const&) {
    return true;
  }

  // Performs the reduction across splits for a given output tile. Since this scheduler does
  // not split output tiles, no reduction is needed.
  template <class FrgTensorC>
  CUTLASS_DEVICE static void fixup(Params const&, WorkTileInfo const&, FrgTensorC&, uint32_t, uint32_t) {}

  // Returns whether the current WorkTileInfo passed in should continue to be used. Since
  // this scheduler only schedules work in units of single, full output tiles, the WorkTileInfo
  // passed in should not be used after having been processed.
  CUTLASS_DEVICE
  static bool continue_current_work(WorkTileInfo&) {
    return false;
  }

  // The basic tile scheduler does not require any additional workspace
  template <class ProblemShape, class ElementAccumulator>
  static size_t get_workspace_size(
      Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1, uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status initialize_workspace(
      Arguments const&,
      void*,
      cudaStream_t,
      ProblemShape,
      KernelHardwareInfo const&,
      uint32_t,
      const uint32_t = 1,
      uint32_t = 1,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  template <class ProblemShape_MNKL, class TileShape>
  CUTLASS_HOST_DEVICE static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape_MNKL problem_shape, TileShape tile_shape) {
    // All work units returned by this scheduler cover the entire K iteration
    // space of the output tile assigned to the work unit.
    return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
  }

  CUTLASS_HOST_DEVICE
  static uint32_t get_work_k_tile_start(WorkTileInfo const&) {
    // All work units returned by this scheduler start from K tile 0
    return 0u;
  }

  CUTLASS_DEVICE
  static bool need_separate_reduction(Params const& params) {
    return false;
  }

  CUTLASS_DEVICE
  bool is_work_tile_for_reduction(WorkTileInfo const& work_tile_info, Params const& params) {
    return false;
  }

  CUTLASS_DEVICE
  uint32_t epilgoue_subtile_idx(WorkTileInfo const& work_tile_info, Params const& params) const {
    return 0;
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE void separate_reduction(
      Params const& params,
      WorkTileInfo const& work_tile_info,
      FrgTensorC& accumulators,
      uint32_t num_barriers,
      uint32_t barrier_idx) {}

  // Shares the accumulator set with peers in the global workspace
  template <class FrgTensorC>
  CUTLASS_DEVICE static void share(
      Params const& params,
      WorkTileInfo const& work_tile_info,
      FrgTensorC& accumulators,
      uint32_t num_barriers,
      uint32_t barrier_idx) {}

  CUTLASS_DEVICE
  static bool valid_warpgroup_in_work_tile(WorkTileInfo const& work_tile_info) {
    return true;
  }

  CUTLASS_DEVICE
  static bool requires_separate_reduction(Params const& params) {
    return false;
  }

  // Kernel helper function to get next work tile
  template <typename WorkTileWithCallbackInfo, typename TileSchedulerPipeline, typename TileSchedulerPipelineState>
  CUTLASS_DEVICE auto fetch_next_work(
      WorkTileWithCallbackInfo work_tile_with_callback_info,
      TileSchedulerPipeline& scheduler_pipeline,
      TileSchedulerPipelineState scheduler_pipe_consumer_state) {
    if (continue_current_work(work_tile_with_callback_info)) {
      return cute::make_tuple(work_tile_with_callback_info, true);
    }
    scheduler_pipeline.consumer_wait(scheduler_pipe_consumer_state);
    work_tile_with_callback_info =
        reinterpret_cast<WorkTileWithCallbackInfo*>(response_ptr_)[scheduler_pipe_consumer_state.index()];
    cutlass::arch::fence_view_async_shared();
    scheduler_pipeline.consumer_release(scheduler_pipe_consumer_state);

    return cute::make_tuple(work_tile_with_callback_info, true);
  }

  CUTLASS_DEVICE
  auto fetch_next_work(WorkTileInfo work_tile_info) {
    if (continue_current_work(work_tile_info)) {
      return cute::make_tuple(work_tile_info, true);
    }

    advance_to_next_work();
    return cute::make_tuple(get_current_work(), true);
  }

  // Returns the initial work tile info that will be computed over
  template <class ClusterShape, typename CallbackBeforeCommit = WorkTileInfo (*)(WorkTileInfo)>
  CUTLASS_DEVICE auto initial_work_tile_info(
      ClusterShape, CallbackBeforeCommit callback_before_commit = [](WorkTileInfo response) { return response; }) {
    auto work_tile = get_current_work_for_linear_idx(current_work_linear_idx_);
    using WorkTileWithCallbackInfo = decltype(callback_before_commit(work_tile));
    WorkTileWithCallbackInfo work_tile_with_callback_info = work_tile;
    if (work_tile_with_callback_info.is_valid()) {
      work_tile_with_callback_info = callback_before_commit(work_tile);
    }
    return work_tile_with_callback_info;
  }
};

}  // namespace cutlass::gemm::kernel::detail
