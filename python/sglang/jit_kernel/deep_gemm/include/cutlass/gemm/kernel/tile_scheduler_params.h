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

/*! \file
    \brief Parameters structures for persistent tile schedulers
*/

#include "cutlass/coord.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/workspace.h"
#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/gemm/kernel/tile_scheduler_detail.hpp"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {
namespace detail {

////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
static uint32_t
get_max_cta_occupancy(int max_sm_per_gpc, GemmCoord cluster_shape, int sm_count) {
  // Provided SM count could possibly be less than the assumed maximum SMs per GPC
  auto cluster_size = cluster_shape.m() * cluster_shape.n();
  int const min_num_gpc = sm_count < max_sm_per_gpc ? 1 : sm_count / max_sm_per_gpc;
  int const max_cta_occupancy_per_gpc = max_sm_per_gpc - (max_sm_per_gpc % cluster_size);
  int cta_per_device = min_num_gpc * max_cta_occupancy_per_gpc;
  // Suppose max_sm_per_gpc = 20, cluster_size = 8, sm_count = 148
  // min_num_gpc = 148 / 20 = 7
  // max_cta_occupancy_per_gpc = 20 - (20 % 8) = 16
  // cta_per_device = 7 * 16 = 112
  // num_gpc_residual = 148 % 20 = 8
  // max_cta_occupancy_per_residual_gpc = 8 - (8 % 8) = 8
  // cta_per_device += 8 = 120
  // cta_per_device = 120 < 148 ? 148 : 120 = 148

  // The calculation below allows for larger grid size launch for different GPUs.
  int const num_gpc_residual = sm_count < max_sm_per_gpc ? 0 : sm_count % max_sm_per_gpc;
  int const max_cta_occupancy_per_residual_gpc = num_gpc_residual - (num_gpc_residual % cluster_size);
  cta_per_device += max_cta_occupancy_per_residual_gpc;

  cta_per_device = sm_count < cta_per_device ? sm_count : cta_per_device;
  return cta_per_device;
}

////////////////////////////////////////////////////////////////////////////////

//
// Parameters for SM90 tile schedulers
//

// Parameters for SM90 persistent tile scheduler
struct PersistentTileSchedulerSm90Params {
  using RasterOrder = cutlass::gemm::kernel::detail::RasterOrder;
  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  FastDivmodU64Pow2 divmod_cluster_shape_major_{};
  FastDivmodU64Pow2 divmod_cluster_shape_minor_{};
  FastDivmodU64 divmod_batch_{};
  FastDivmodU64 divmod_cluster_blk_major_{};

  uint64_t blocks_per_problem_ = 0;
  int32_t log_swizzle_size_ = 0;
  RasterOrder raster_order_ = RasterOrder::AlongN;

  uint32_t problem_tiles_m_ = 0;
  uint32_t problem_tiles_n_ = 0;
  uint32_t problem_tiles_l_ = 0;
  uint32_t cluster_shape_m_ = 0;
  uint32_t cluster_shape_n_ = 0;

  // Initializes members. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  void
  initialize(
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option
  ) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape, tile_shape, cluster_shape);
    return initialize(
      problem_blocks,
      cluster_shape,
      hw_info,
      max_swizzle_size,
      raster_order_option
    );
  }

  // Version of initialize that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  void
  initialize(
    dim3 problem_blocks,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option
  ) {

    CUTLASS_UNUSED(hw_info);

    // Round up to nearest multiple of swizzle_size along each mode
    auto log_swizzle_size = get_log_swizzle_size(problem_blocks.x, problem_blocks.y, max_swizzle_size);
    auto problem_blocks_m = round_up(problem_blocks.x, (1 << log_swizzle_size) * cluster_shape.m());
    auto problem_blocks_n = round_up(problem_blocks.y, (1 << log_swizzle_size) * cluster_shape.n());

    problem_tiles_m_ = problem_blocks_m / cluster_shape.m();
    problem_tiles_n_ = problem_blocks_n / cluster_shape.n();
    problem_tiles_l_ = problem_blocks.z;
    cluster_shape_m_ = cluster_shape.m();
    cluster_shape_n_ = cluster_shape.n();

    RasterOrder raster_order = get_rasterization_order(
      problem_blocks_m,
      problem_blocks_n,
      raster_order_option
    );

    //
    // Set members
    //

    blocks_per_problem_ = problem_blocks_m * problem_blocks_n * problem_blocks.z;
    log_swizzle_size_ = log_swizzle_size;
    raster_order_ = raster_order;
    divmod_batch_ = FastDivmodU64(problem_blocks_m * problem_blocks_n);

    if (raster_order == RasterOrder::AlongN) {
      divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.n());
      divmod_cluster_shape_minor_ = FastDivmodU64Pow2(cluster_shape.m());
      divmod_cluster_blk_major_ = FastDivmodU64(problem_blocks_n / cluster_shape.n());
    }
    else {
      divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.m());
      divmod_cluster_shape_minor_ = FastDivmodU64Pow2(cluster_shape.n());
      divmod_cluster_blk_major_ = FastDivmodU64(problem_blocks_m / cluster_shape.m());
    }
  }

  // Given the inputs, computes the physical grid we should launch.
  // This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    BatchedGemmCoord problem_shape,
    GemmCoord cta_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option,
    bool truncate_by_problem_size=true,
    bool bypass_sm90_occupancy_calculation=false 
    ) {

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape, cta_shape, cluster_shape);
    return get_grid_shape(
      problem_blocks,
      cluster_shape,
      hw_info,
      max_swizzle_size,
      raster_order_option,
      truncate_by_problem_size,
      bypass_sm90_occupancy_calculation 
    );
  }

  // Version of get_grid_shape that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    dim3 problem_blocks,
    GemmCoord cluster_shape,
    KernelHardwareInfo hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option,
    bool truncate_by_problem_size=true,
    bool bypass_sm90_occupancy_calculation=false 
    ) {

    int const sm_count = hw_info.sm_count;
    int const max_active_clusters = hw_info.max_active_clusters;

    // Round up to nearest multiple of swizzle_size along each mode
    auto log_swizzle_size = get_log_swizzle_size(problem_blocks.x, problem_blocks.y, max_swizzle_size);
    auto problem_blocks_m = round_up(problem_blocks.x, (1 << log_swizzle_size) * cluster_shape.m());
    auto problem_blocks_n = round_up(problem_blocks.y, (1 << log_swizzle_size) * cluster_shape.n());

    int problem_blocks_total = problem_blocks_m * problem_blocks_n * problem_blocks.z;

    RasterOrder raster_order = get_rasterization_order(
      problem_blocks_m,
      problem_blocks_n,
      raster_order_option
    );

    dim3 launch_grid;

    if (raster_order == RasterOrder::AlongN) {
      launch_grid = dim3(cluster_shape.m(), 1, 1);
    }
    else {
      launch_grid = dim3(1, cluster_shape.n(), 1);
    }

    auto possibly_truncate = [&](int x, int y) {
      if (truncate_by_problem_size) {
        return platform::min(x, y);
      }
      else {
        return x;
      }
    };

    // The else path is generic, however, we can avoid some divs if we know cluster size is 1
    auto cluster_size = cluster_shape.m() * cluster_shape.n();
    if (cluster_size == 1) {
      if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = possibly_truncate(sm_count, problem_blocks_total);
      }
      else {
        launch_grid.x = possibly_truncate(sm_count, problem_blocks_total);
      }
    }
    // In case the maximum number of clusters that could co-exist on the target device is
    // already calculated using cudaOccupancyMaxActiveClusters
    else if (max_active_clusters != 0 && max_active_clusters * cluster_size <= sm_count) {
      if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = possibly_truncate(
            max_active_clusters * cluster_shape.n(),
            problem_blocks_total / cluster_shape.m());

      }
      else {
        launch_grid.x = possibly_truncate(
            max_active_clusters * cluster_shape.m(),
            problem_blocks_total / cluster_shape.n());
      }
      CUTLASS_TRACE_HOST("get_grid_shape(): Proposed GridDims by the scheduler using cudaOccupancyMaxActiveClusters = "
          "(" << launch_grid.x << ", " << launch_grid.y << ", " << launch_grid.z << ")\n");
    }
    else {
      int cta_per_device = sm_count;
      if (!bypass_sm90_occupancy_calculation) { 
        /*
        * Optimal grid size calculation is based on
        * GH100: 8 GPCs, 72 TPCs (9 TPCs/GPC), 2 SMs/TPC, 144 SMs per full GPU
        * Hence, maximum SMs per GPC = 18
        */
        constexpr int max_sm_per_gpc = 18;
        cta_per_device = get_max_cta_occupancy(max_sm_per_gpc, cluster_shape, sm_count);
      } 

      if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = possibly_truncate(
            cta_per_device       / cluster_shape.m(),
            problem_blocks_total / cluster_shape.m());
      }
      else {
        launch_grid.x = possibly_truncate(
            cta_per_device       / cluster_shape.n(),
            problem_blocks_total / cluster_shape.n());
      }
      CUTLASS_TRACE_HOST("get_grid_shape(): Proposed GridDims by the scheduler using heuristics = "
          "(" << launch_grid.x << ", " << launch_grid.y << ", " << launch_grid.z << ")\n");
    }
    return launch_grid;
  }

  CUTLASS_HOST_DEVICE
  static int32_t
  get_log_swizzle_size(int problem_ctas_m, int problem_ctas_n, int max_swizzle_size) {
    int min_cta_dim = platform::min(problem_ctas_m, problem_ctas_n);
    if (max_swizzle_size >= 8 && min_cta_dim >= 6) {
      return 3;
    }
    else if (max_swizzle_size >= 4 && min_cta_dim >= 3) {
      return 2;
    }
    else if (max_swizzle_size >= 2 && min_cta_dim >= 2) {
      return 1;
    }
    else {
      return 0;
    }
  }

  CUTLASS_HOST_DEVICE
  static RasterOrder
  get_rasterization_order(
    uint32_t tiles_m,
    uint32_t tiles_n,
    RasterOrderOptions raster_order_option
  ) {

    if (raster_order_option == RasterOrderOptions::Heuristic) {
      if (tiles_n > tiles_m) {
        return RasterOrder::AlongM;
      }
      else {
        return RasterOrder::AlongN;
      }
    }
    else {
      switch (raster_order_option) {
        case RasterOrderOptions::AlongN:
          return RasterOrder::AlongN;
          break;
        default:
          return RasterOrder::AlongM;
      }
    }
  }

  // Get the number of CTA tiles in this problem. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  CUTLASS_HOST_DEVICE
  static dim3
  get_tiled_cta_shape_mnl(BatchedGemmCoord problem_shape, GemmCoord cta_shape, GemmCoord cluster_shape) {
    auto cta_m = (problem_shape.m() + cta_shape.m() - 1) / cta_shape.m();
    auto cta_n = (problem_shape.n() + cta_shape.n() - 1) / cta_shape.n();

    return get_tiled_cta_shape_mnl(problem_shape, cluster_shape, cta_m, cta_n);
  }

  // Version of get_tiled_cta_shape_mnl that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE
  static dim3
  get_tiled_cta_shape_mnl(BatchedGemmCoord problem_shape, GemmCoord cluster_shape, uint32_t cta_m, uint32_t cta_n) {

    // Round up to nearest multiple of cluster dim along each mode
    auto problem_blocks_m = ((cta_m + cluster_shape.m() - 1) / cluster_shape.m()) * cluster_shape.m();
    auto problem_blocks_n = ((cta_n + cluster_shape.n() - 1) / cluster_shape.n()) * cluster_shape.n();

    return {
      static_cast<uint32_t>(problem_blocks_m),
      static_cast<uint32_t>(problem_blocks_n),
      static_cast<uint32_t>(problem_shape.batch())
    };
  }
};

////////////////////////////////////////////////////////////////////////////////

// Parameters for SM90 persistent stream-K scheduler
struct PersistentTileSchedulerSm90StreamKParams {
  using ReductionMode = cutlass::gemm::kernel::detail::ReductionMode;
  using DecompositionMode = cutlass::gemm::kernel::detail::DecompositionMode;


  using UnderlyingParams = PersistentTileSchedulerSm90Params;
  using RasterOrder = cutlass::gemm::kernel::detail::RasterOrder;
  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  // Cluster dimensions are typically always a power of 2, so use
  // the power-of-two variants of FastDivmod for these.
  FastDivmodU64Pow2 divmod_cluster_shape_major_{};
  FastDivmodU64Pow2 divmod_cluster_shape_minor_{};

  FastDivmodU64 divmod_batch_{};
  FastDivmodU64 divmod_cluster_blk_major_{};

  // Total number of cluster-sized output tiles (i.e., not including any
  // splitting factors). This is primarily used for split-K decompositions,
  // and may be overridden in other decompositions.
  FastDivmodU64 divmod_clusters_mnl_{};

  // We divide up the number of stream-K tiles amongst G groups of stream-K units.
  // The stream-K units within a group collaborate to compute over the `sk_tiles / G`
  // tiles assigned to that group. Non-unit group sizes can help to preserve L2 locality of
  // partial chunks computed by stream-K units -- units 0 in each group will compute identical K extents
  // of tiles that would be assigned in the same wave according to the rasterization order of the
  // data-parallel formulation of the problem.
  FastDivmodU64 divmod_sk_groups_{};

  // Number of stream-K units in each group
  FastDivmodU64 divmod_sk_units_per_group_{};

  uint64_t units_per_problem_ = 0;
  FastDivmod divmod_tiles_per_output_tile_{};
  int32_t log_swizzle_size_ = 0;
  RasterOrder raster_order_ = RasterOrder::AlongN;

  // The splitting factor to be used in a split-K decomposition of the problem.
  // If this is set to a value greater than 1, stream-K decomposition logic
  // is bypassed in favor of a split-K decomposition.
  FastDivmod divmod_splits_{};

  // Number of stream-K or split-K work units that compute an extra k iteration.
  // This is done to handle residuals in dividing up the k iteration space.
  // For stream-K, since the actual assignment of work to stream-K units will be done
  // at the granularity of a cluster, we store only the number of big clusters.
  uint32_t big_units_ = 0;

  // The number of groups of stream-K units that will process an extra stream-K tile cluster.
  uint32_t big_groups_ = 0;

  // Workspace for holding partial accumulators to be reduced across stream-K/split-K units
  void* reduction_workspace_ = nullptr;

  // Number of tiles covered by stream-K work units
  uint32_t sk_tiles_ = 0;

  // Number of work units computing stream-K tiles
  uint32_t sk_units_ = 0;

  // Number of tiled k iterations computed by each stream-K work unit. This
  // can potentially cover more than one output tile.
  FastDivmod divmod_k_tiles_per_sk_unit_{};
  // Number of tiled k iterations computed by each "big" stream-K units, which
  // processes one more K chunk than a "normal" stream-K unit.
  FastDivmod divmod_k_tiles_per_sk_big_unit_{};

  // Strategy to use when reducing between collaborating CTAs
  ReductionMode reduction_mode_ = ReductionMode::Deterministic;

  // The number of sub blocks in the kernel epilogue
  FastDivmodU64 divmod_epilogue_subtile_{};

  // The number of blocks that launched for doing separate reduction
  uint32_t separate_reduction_units_ = 0;

  // Minimum number of k tiles that can be assigned to a stream-K unit
  static constexpr uint32_t min_iters_per_sk_unit_ = 8u;

  // Maximum number of groups of stream-K units
  static constexpr uint32_t max_sk_groups_ = 8u;

  // ktile start from even for each cta
  uint32_t ktile_start_alignment_count_ { 1u };

  // Divides dividend by the cluster size
  CUTLASS_HOST_DEVICE
  uint64_t
  div_cluster_size(uint64_t dividend) const {
    // Use each underlying fast divmod rather than performing integer division
    // by the multiplication of major.divisor * minor.divisor
    return divmod_cluster_shape_minor_.divide(
      divmod_cluster_shape_major_.divide(dividend)
    );
  }

  
  // Divides dividend by the cluster size in the M dimension
  CUTLASS_HOST_DEVICE
  uint64_t
  truncate_to_cluster_size_m(uint64_t dividend) const {
    if (raster_order_ == RasterOrder::AlongN) {
      return divmod_cluster_shape_minor_.divide(dividend) * divmod_cluster_shape_minor_.divisor;
    }
    else {
      return divmod_cluster_shape_major_.divide(dividend) * divmod_cluster_shape_major_.divisor;
    }
  }

  // Divides dividend by the cluster size in the N dimension
  CUTLASS_HOST_DEVICE
  uint64_t
  truncate_to_cluster_size_n(uint64_t dividend) const {
    if (raster_order_ == RasterOrder::AlongM) {
      return divmod_cluster_shape_minor_.divide(dividend) * divmod_cluster_shape_minor_.divisor;
    }
    else {
      return divmod_cluster_shape_major_.divide(dividend) * divmod_cluster_shape_major_.divisor;
    }
  }
  

  CUTLASS_HOST_DEVICE
  uint64_t
  get_cluster_size() const {
    return divmod_cluster_shape_minor_.divisor * divmod_cluster_shape_major_.divisor;
  }

  // Returns whether the kernel uses separate reduction
  CUTLASS_HOST_DEVICE
  bool
  requires_separate_reduction() const {
    return separate_reduction_units_ > 0;
  }

  // Returns the maximum number of peers that can collaborate on a given output tile
  CUTLASS_HOST_DEVICE
  static uint32_t
  max_peers_per_tile(uint64_t sk_units, uint64_t sk_tiles) {
    // When we can divide up our SK units to SK tiles evenly, the number of peers
    // per SK tile is exactly (sk_units_ / sk_tiles_). In cases where this division
    // is not exact, some tiles will need to be covered by additional SK units. Because
    // the extra work can occur at both the beginning and the end of the SK tile, at
    // most 2 extra peers will be needed.
    return static_cast<uint32_t>(sk_units / sk_tiles + 2);
  }

  // Initializes members. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  void
  initialize(
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    ReductionMode reduction_mode,
    DecompositionMode decomposition_mode,
    void* workspace,
    const uint32_t epilogue_subtile = 1u,
    uint32_t ktile_start_alignment_count = 1u,
    bool bypass_sm90_occupancy_calculation=false
  ) {
    dim3 problem_blocks = UnderlyingParams::get_tiled_cta_shape_mnl(
      problem_shape, tile_shape, cluster_shape);

    // Number of k tiles in each output tile
    uint32_t k_tiles_per_output_tile = (problem_shape.k() + tile_shape.k() - 1) / tile_shape.k();

    initialize(
      problem_blocks,
      k_tiles_per_output_tile,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle,
      raster_order_option,
      reduction_mode,
      decomposition_mode,
      workspace,
      epilogue_subtile,
      ktile_start_alignment_count,
      bypass_sm90_occupancy_calculation
    );
  }

  // Version of initialize that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  void
  initialize(
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord cluster_shape,
    KernelHardwareInfo hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    ReductionMode reduction_mode,
    DecompositionMode decomposition_mode,
    void* workspace,
    const uint32_t epilogue_subtile = 1,
    uint32_t ktile_start_alignment_count = 1u,
    bool bypass_sm90_occupancy_calculation=false
  ) {

    #if !defined(__CUDACC_RTC__)
    if (hw_info.sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      hw_info.sm_count = KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    }
    #endif // !defined(__CUDACC_RTC__) 

    ktile_start_alignment_count_ = ktile_start_alignment_count; 
    UnderlyingParams underlying_params;
    underlying_params.initialize(
      problem_blocks,
      cluster_shape,
      hw_info,
      max_swizzle,
      raster_order_option
    );

    // Set basic parameters that not affected by any heuristics in advance.
    set_params_base(underlying_params, workspace);

    // Call for internal streamk heuristic to setup streamk related params
    stream_k_heuristic(
      underlying_params,
      problem_blocks,
      k_tiles_per_output_tile,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle,
      raster_order_option,
      decomposition_mode,
      reduction_mode,
      epilogue_subtile,
      ktile_start_alignment_count,
      bypass_sm90_occupancy_calculation
    ); 
  }
  
  // max_sk_groups_ unless this extends beyond the extent of the dimension over
  // which the problem is rasterized. For example, if the tiled problem shape
  // (in CTA_M x CTA_N representation) when using 1x1 clusters is 4x16,
  // and we rasterize along the M dimension, we choose 4 groups, rather than 8.
  // If the cluster shape is 2x1, we choose 2 groups (CTA_M / CLUSTER_M).
  uint32_t calculate_groups(
    UnderlyingParams underlying_params,
    ReductionMode reduction_mode,
    uint32_t problem_blocks_m,
    uint32_t problem_blocks_n,
    GemmCoord cluster_shape,
    uint64_t cluster_size,
    uint32_t sk_tiles,
    uint64_t sk_cluster_tiles,
    uint64_t sk_units,
    uint32_t k_tiles_per_output_tile,
    bool do_separate_reduction) {

    uint32_t max_groups_problem;
    if (underlying_params.raster_order_ == RasterOrder::AlongM) {
      max_groups_problem = problem_blocks_m / cluster_shape.m();
    }
    else {
      max_groups_problem = problem_blocks_n / cluster_shape.n();
    }

    // Select the number of groups that will be use. We start with the maximum
    // number of potential groups, and iterate down looking for a group size that
    // evenly divides the stream-K units and tiles, and for which the resulting
    // number of K tiles per stream-K unit remains above min_iters_per_sk_unit_

    uint32_t groups = platform::min(max_groups_problem, uint32_t(max_sk_groups_));
    // Grouping is disabled when separate reduction is used because grouping is primarily an attempt
    // to improve L2 locality, and L2-locality optimizations are unnecessary when the the kernel
    // is a single wave (which is the case for separate reduction).
    if (
      do_separate_reduction
      ) {
      groups = 1;
    }

    uint32_t fallback_groups = 0;
    auto sk_cluster_units = sk_units / cluster_size;

    auto sk_splits_too_small = [&](uint32_t g) {
      // Check whether the number of K tiles computed per stream-K unit is less
      // than min_iters_per_sk_unit_
      auto total_sk_cluster_tiles = (sk_cluster_tiles / g) * cluster_size;
      auto total_sk_k_tiles = total_sk_cluster_tiles * k_tiles_per_output_tile;
      auto k_tiles_per_sk_unit = total_sk_k_tiles / (sk_units / g);
      return k_tiles_per_sk_unit < min_iters_per_sk_unit_;
    };

    auto is_ideal_grouping = [&](uint32_t g) {
      // An ideal grouping will evenly divide stream-K clusters, evenly divide
      // stream-K tiles, and not result in stream-K splits that are too small.
      return (sk_cluster_units % g == 0) && (sk_cluster_tiles % g == 0) && !sk_splits_too_small(g);
    };

    auto is_valid_grouping = [&](uint32_t g) {
      // A grouping is valid, but not ideal, if it evenly divides the
      // stream-K clusters and does not result in stream-K splits that are
      // too small. Such a setting can be used as a fallback option in the
      // case that an ideal grouping is not achievable
      return sk_cluster_units % g == 0 && !sk_splits_too_small(g);
    };

    while (groups > 1 && !is_ideal_grouping(groups)) {
      if (fallback_groups == 0 && is_valid_grouping(groups)) {
        // Set fallback groups once in preference for a larger number of groups.
        fallback_groups = groups;
      }
      --groups;
    }

    // If groups == 1, we did not find a group count that satisfies all criteria. If we have
    // found a fallback group count, use this instead.
    if (groups == 1 && fallback_groups > 0) {
      groups = fallback_groups;
    }
    return groups;
  }

  // Stream-K kernel use below function to set stream-K feature related parameters to choose
  // optimal/customized decomposition mode.
  void stream_k_heuristic(
      UnderlyingParams underlying_params,
      dim3 problem_blocks,
      uint32_t k_tiles_per_output_tile,
      GemmCoord cluster_shape,
      KernelHardwareInfo hw_info,
      int splits,
      int max_swizzle,
      RasterOrderOptions raster_order_option,
      DecompositionMode decomposition_mode,
      ReductionMode reduction_mode,
      const uint32_t epilogue_subtile = 1,
      uint32_t ktile_start_alignment_count = 1u,
      bool bypass_sm90_occupancy_calculation=false) {
    uint32_t groups = 0;
    uint32_t sk_tiles = 0;
    uint64_t sk_units = 0;
    uint64_t cluster_size = 0;
    uint64_t dp_units = 0;
    uint64_t k_tiles_per_group = 0;
    uint64_t k_tiles_per_sk_unit = 0;
    uint64_t sk_big_groups = 0;
    uint32_t sk_splits = 1;
    // Self calculated optimal heuristic mode
    DecompositionMode heuristic_mode =
      select_decomposition_mode(
        groups,
        sk_tiles,
        sk_units,
        cluster_size,
        dp_units,
        k_tiles_per_group,
        k_tiles_per_sk_unit,
        sk_big_groups,
        sk_splits,
        underlying_params,
        problem_blocks,
        k_tiles_per_output_tile,
        cluster_shape,
        hw_info,
        splits,
        max_swizzle,
        raster_order_option,
        decomposition_mode,
        reduction_mode,
        epilogue_subtile,
        ktile_start_alignment_count,
        bypass_sm90_occupancy_calculation
      );

    // Given heuristic_mode returned from the heuristic() method, set params fields.
    // Here, we decouple the params that have no relation with
    // decomposition mode from the params that are decided within heuristic().
    set_params(
      heuristic_mode,
      groups,
      sk_tiles,
      sk_units,
      cluster_size,
      dp_units,
      k_tiles_per_group,
      k_tiles_per_sk_unit,
      sk_big_groups,
      sk_splits,
      underlying_params,
      problem_blocks,
      k_tiles_per_output_tile,
      cluster_shape,
      splits,
      epilogue_subtile,
      reduction_mode,
      ktile_start_alignment_count
    );
  }

  // Return the optimal decomposition result by heuristic.
  DecompositionMode select_decomposition_mode(
    uint32_t &groups,
    uint32_t &sk_tiles,
    uint64_t &sk_units,
    uint64_t &cluster_size,
    uint64_t &dp_units,
    uint64_t &k_tiles_per_group,
    uint64_t &k_tiles_per_sk_unit,
    uint64_t &sk_big_groups,
    uint32_t &sk_splits,
    UnderlyingParams underlying_params,
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord cluster_shape,
    KernelHardwareInfo hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    ReductionMode reduction_mode,
    uint32_t epilogue_subtile,
    uint32_t ktile_start_alignment_count,
    bool bypass_sm90_occupancy_calculation=false
  ) {

    // Get block numbers in m, n and l dimensions
    if (decomposition_mode == DecompositionMode::SplitK ||
        (decomposition_mode == DecompositionMode::Heuristic && splits > 1)) {
      // Short circuit to basic split-K decomposition
      uint32_t adapted_splits = adjust_split_count(
        splits, hw_info.sm_count, k_tiles_per_output_tile
        , ktile_start_alignment_count 
      );
      sk_splits = adapted_splits;
      return DecompositionMode::SplitK;
    }
    else {
      // Calculate the maximum number of blocks from clusters of shape cluster_shape that we
      // can fit within sm_count SMs.
      // Get block numbers in m, n and l dimensions
      auto problem_blocks_l = problem_blocks.z;
      auto problem_blocks_m = round_up(problem_blocks.x, (1 << underlying_params.log_swizzle_size_) * cluster_shape.m());
      auto problem_blocks_n = round_up(problem_blocks.y, (1 << underlying_params.log_swizzle_size_) * cluster_shape.n());
      uint64_t output_tiles = problem_blocks_m * problem_blocks_n * problem_blocks_l;
      dim3 grid = get_grid_shape(
        problem_blocks,
        cluster_shape,
        hw_info,
        max_swizzle,
        raster_order_option,
        bypass_sm90_occupancy_calculation
      );
      uint64_t ctas_per_wave = grid.x * grid.y;
      cluster_size = cluster_shape.m() * cluster_shape.n();
      uint64_t ctas_per_wave_in_full_clusters = (ctas_per_wave / cluster_size) * cluster_size; 

      // The number of output tiles to be computed in stream-K and data-parallel fashion, respectively.
      sk_tiles = get_num_sk_tiles(
        output_tiles,
        ctas_per_wave,
        cluster_size,
        k_tiles_per_output_tile,
        decomposition_mode,
        ctas_per_wave_in_full_clusters 
      );
      uint64_t dp_tiles = output_tiles - sk_tiles;
      // Calculate the number of work units covering the data-parallel and stream-K tiles.
      // A "work unit" is a single index in the linearized ID space used by the scheduler.
      // We distinguish it from a "block," which is typically tied to a hardware unit
      // (e.g., the callers into this scheduler will be persistent thread blocks).
      // A work unit can encompass multiple output tiles worth of work (as will be the
      // case for stream-K blocks).
      // Since splitting is not required for data-parallel tiles, only one data-parallel unit
      // is needed per data-parallel tile.
      dp_units = dp_tiles;

      uint64_t ctas_per_sk_wave = ctas_per_wave;
      ctas_per_sk_wave = ctas_per_wave_in_full_clusters; 
      sk_units = get_num_sk_units(cluster_shape, ctas_per_sk_wave, sk_tiles, k_tiles_per_output_tile);

      if (decomposition_mode == DecompositionMode::DataParallel ||
          (decomposition_mode == DecompositionMode::Heuristic && sk_tiles == 0) ||
          sk_units == 0) {
        // Short circuit to basic data-parallel decomposition
        return DecompositionMode::DataParallel;
      }
      else {
        bool do_separate_reduction = should_perform_separate_reduction(
          epilogue_subtile, sk_units, sk_tiles, dp_tiles, ctas_per_wave);
        
        uint64_t sk_cluster_tiles = sk_tiles / cluster_size;

        groups = calculate_groups(underlying_params, reduction_mode, problem_blocks_m, problem_blocks_n, cluster_shape,
          cluster_size, sk_tiles, sk_cluster_tiles, sk_units, k_tiles_per_output_tile, do_separate_reduction);

        auto sk_units_per_group = sk_units / groups;

        // sk_tiles is guaranteed to be divisible by cluster_size because it is calculated as:
        //    sk_tiles = (waves <= 2) ? total_tiles : (sm_count + (total_tiles % sm_count))
        // Both total_tiles and sm_count are multiples of cluster size due to padding added
        // prior to kernel launch.
        uint64_t sk_cluster_tiles_per_group = sk_cluster_tiles / groups;
        uint64_t sk_tiles_per_group = sk_cluster_tiles_per_group * cluster_size;

        // Groups that will process an extra stream-K tile cluster. These differ from "big_units," which
        // are stream-K units within a group that process an extra K chunk.
        sk_big_groups = sk_cluster_tiles % groups;

        k_tiles_per_group = k_tiles_per_output_tile * sk_tiles_per_group;

        // Number of k tiles computed per stream-K unit
        k_tiles_per_sk_unit = k_tiles_per_group / sk_units_per_group;

        DecompositionMode heuristic_mode;
        if (decomposition_mode == DecompositionMode::Heuristic && sk_tiles < sk_units && sk_units % sk_tiles == 0) {
          // If the number of stream-K units is a multiple of the number of stream-K tiles, then
          // the problem can leverage a basic split-K decomposition for the stream-K tiles.
          // This case happens when separate reduction is disable.
          sk_splits = static_cast<uint32_t>(sk_units / sk_tiles);
          heuristic_mode = DecompositionMode::SplitK;
        }
        else {
          // Rest scenario is streamk
          heuristic_mode = DecompositionMode::StreamK;
        }
        // Refresh heuristic_mode using analytical model before choosing streamk/separate_reduction decomposition,
        // ideally it's to get the final decomposition more accuracy. Comment it as it is place holder at this moment.
        #if 0
        uint32_t total_waves = static_cast<uint32_t>((output_tiles + ctas_per_wave - 1) / ctas_per_wave);
        analytical_model(heuristic_mode, k_tiles_per_output_tile, k_tiles_per_sk_unit,
          sk_splits, epilogue_subtile, total_waves);
        #endif
        return heuristic_mode;
      }
    }
  }

  // Given decomposition mode output from heuristic, set all fields of params.
  void set_params(
    DecompositionMode heuristic_mode,
    uint32_t groups,
    uint32_t sk_tiles,
    uint64_t sk_units,
    uint64_t cluster_size,
    uint64_t dp_units,
    uint64_t k_tiles_per_group,
    uint64_t k_tiles_per_sk_unit,
    uint64_t sk_big_groups,
    uint32_t sk_splits,
    UnderlyingParams underlying_params,
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord cluster_shape,
    uint32_t splits,
    uint32_t epilogue_subtile,
    ReductionMode reduction_mode
    , uint32_t ktile_start_alignment_count 
    ) {
    // The highest priority when customers set as splitk mode, may set
    // with a adapted splits value rather than the original splits
    // even it does not make sense
    if (splits > 1 && heuristic_mode == DecompositionMode::SplitK) {
      set_params_basic(
        underlying_params,
        problem_blocks,
        cluster_shape,
        sk_splits, // split-k set by customers
        k_tiles_per_output_tile,
        reduction_mode
      );
    }
    else if (heuristic_mode == DecompositionMode::DataParallel) {
      set_params_basic(
        underlying_params,
        problem_blocks,
        cluster_shape,
        1, // fast path to fall back to the mode without any split scheme
        k_tiles_per_output_tile,
        reduction_mode
      );
    }
    else if (heuristic_mode == DecompositionMode::SplitK) {
      set_params_basic(
        underlying_params,
        problem_blocks,
        cluster_shape,
        sk_splits, // splits calculated by heuristic
        k_tiles_per_output_tile,
        reduction_mode
      );
    }
    else {
      // streamk
      set_params_stream_k(
        underlying_params,
        k_tiles_per_output_tile,
        groups,
        sk_tiles,
        sk_units,
        cluster_size,
        dp_units,
        k_tiles_per_group,
        k_tiles_per_sk_unit,
        sk_big_groups,
        reduction_mode,
        1, /*epilogue_subtile*/
        0  /*reduction_units*/
      );
    }
  }

  // Given the inputs, computes the physical grid we should launch.
  // This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  CUTLASS_HOST_DEVICE
  static dim3
  get_grid_shape(
    BatchedGemmCoord problem_shape,
    GemmCoord cta_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option,
    bool bypass_sm90_occupancy_calculation=false
  ) {

    dim3 problem_blocks = UnderlyingParams::get_tiled_cta_shape_mnl(problem_shape, cta_shape, cluster_shape);

    return get_grid_shape(
      problem_blocks,
      cluster_shape,
      hw_info,
      max_swizzle_size,
      raster_order_option,
      bypass_sm90_occupancy_calculation
    );
  }

  // Version of get_grid_shape that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE
  static dim3
  get_grid_shape(
    dim3 problem_blocks,
    GemmCoord cluster_shape,
    KernelHardwareInfo hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option,
    bool bypass_sm90_occupancy_calculation=false
  ) {

    // Call into the underlying get_grid_shape method, but do not allow the grid shape returned
    // to be truncated based on the number of output tiles in the problem.
    return UnderlyingParams::get_grid_shape(
      problem_blocks,
      cluster_shape,
      hw_info,
      max_swizzle_size,
      raster_order_option,
      /* truncate_by_problem_size = */false,
      bypass_sm90_occupancy_calculation 
    );
  }

  // Returns the number of stream-K tiles that will be computed amongst `output_tiles` total
  // output tiles on a device with `ctas_per_wave` CTAs in each wave.
  static uint32_t
  get_num_sk_tiles(
    uint64_t output_tiles,
    uint64_t ctas_per_wave,
    uint64_t cluster_size,
    uint32_t k_tiles_per_output_tile,
    DecompositionMode decomposition_mode
    , uint64_t ctas_per_wave_in_full_clusters 
  ) {
    uint32_t full_waves = static_cast<uint32_t>(output_tiles / ctas_per_wave);
    uint32_t total_waves = static_cast<uint32_t>((output_tiles + ctas_per_wave - 1) / ctas_per_wave);

    if (decomposition_mode == DecompositionMode::DataParallel ||
        decomposition_mode == DecompositionMode::SplitK) {
      return 0;
    }

    // If there is wave quantization, assign the first two waves worth of tiles to be
    // covered by stream-K work and the remainder to be data-parallel. Since we know
    // that full_waves == total_waves - 1 in this case, the number of data-parallel
    // waves is simply full_waves-1 (unless full_waves == 0).
    uint32_t dp_waves = full_waves > 1 ? full_waves - 1 : 0;
    uint64_t dp_tiles = dp_waves * ctas_per_wave;
    uint64_t sk_tiles = output_tiles - dp_tiles;

    if (full_waves == total_waves || k_tiles_per_output_tile <= min_iters_per_sk_unit_) {
      // All tiles will be data-parallel tiles if there is either no quantization
      // or if there is no work to be split.
      return 0;
    }

    //
    // The final wave is not full. Perform some stream-K work.
    //
    if (decomposition_mode == DecompositionMode::Heuristic) {
      // Rudimentary heuristic: prefer data-parallel decomposition if we have more than
      // one wave and the tail wave is more than half full. This is subject to change.
      uint64_t tail_tiles = output_tiles - (full_waves * ctas_per_wave);
      if (2 * tail_tiles >= ctas_per_wave) {
        return 0;
      }
    }
    // Ensure that the number of SK tiles is divisible by cluster size so that it can be evenly
    // divided among SK clusters.
    sk_tiles = (sk_tiles / cluster_size) * cluster_size;

    return static_cast<uint32_t>(sk_tiles);
  }

  CUTLASS_HOST_DEVICE
  static uint64_t
  get_num_sk_units(GemmCoord cluster_shape, uint64_t ctas_per_sk_wave, uint32_t sk_tiles, uint32_t k_tiles_per_output_tile) {
    // If there are stream-K tiles to compute and a sufficiently large number of k iterations
    // across them, they will be covered by a single wave of persistent threadblocks. Thus, there
    // will be as many work units as there are threadblocks in a single wave.
    //
    // When the total k iterations across stream-K tiles is too small to justify distributing
    // across an entire wave of blocks, we instead distribute the iterations over a smaller
    // set of blocks.

    // Calculate the number of stream-K units that would be needed if each stream-K unit
    // computed the minimum allowable k iterations. Truncate this to be in units of clusters.

    // Number of k iterations computed by the stream-K units as a whole
    uint64_t k_tiles_sk_total = k_tiles_per_output_tile * sk_tiles;

    // Calculate the number of stream-K units that would be needed if each stream-K unit
    // computed the minimum allowable k iterations. Truncate this to be in units of clusters.
    auto cluster_size = cluster_shape.m() * cluster_shape.n();
    uint64_t min_sized_sk_units = (k_tiles_sk_total / min_iters_per_sk_unit_);
    min_sized_sk_units = (min_sized_sk_units / cluster_size) * cluster_size;

    uint64_t sk_units = platform::min(ctas_per_sk_wave, min_sized_sk_units);
    return sk_units;
  }

  // Calculates the size of the workspace needed for holding reduction barriers
  CUTLASS_HOST_DEVICE
  static size_t
  get_barrier_workspace_size(uint64_t num_tiles, uint32_t mma_warp_groups, uint32_t barrier_bits) {
    size_t workspace_bits = num_tiles * static_cast<size_t>(mma_warp_groups) * static_cast<size_t>(barrier_bits);
    return round_up_to_l2_alignment(bits_to_bytes<size_t>(workspace_bits));
  }

  // Calculates the size of the workspace needed for holding partial outputs from splits
  CUTLASS_HOST_DEVICE
  static size_t
  get_reduction_workspace_size(uint64_t num_tiles, GemmCoord tile_shape, uint32_t accumulator_bits, uint32_t num_accumulator_mtxs = 1) {
    size_t output_tile_size = tile_shape.m() * tile_shape.n();
    size_t workspace_bits = accumulator_bits * output_tile_size * num_tiles * num_accumulator_mtxs;
    return round_up_to_l2_alignment(bits_to_bytes<size_t>(workspace_bits));
  }

  #if !defined(__CUDACC_RTC__)
  static void
  get_workspace_component_sizes(
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    size_t& barrier_workspace_size,
    size_t& reduction_workspace_size,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    ReductionMode reduction_mode,
    uint32_t mma_warp_groups,
    uint32_t barrier_bits,
    uint32_t accumulator_bits,
    uint32_t epilogue_subtile = 1,
    uint32_t num_accumulator_mtxs = 1,
    uint32_t ktile_start_alignment_count = 1,
    bool bypass_sm90_occupancy_calculation=false) {

    auto log_swizzle_size = UnderlyingParams::get_log_swizzle_size(problem_blocks.x, problem_blocks.y, max_swizzle);
    problem_blocks.x = round_up(problem_blocks.x, (1 << log_swizzle_size) * cluster_shape.m());
    problem_blocks.y = round_up(problem_blocks.y, (1 << log_swizzle_size) * cluster_shape.n());

    // Workspace is needed only for output tiles that will be split. Thus, we first determine the number
    // of output tiles that will be split, and then calculate the workspace needed to cover these.
    uint64_t output_tiles = problem_blocks.x * problem_blocks.y * problem_blocks.z;

    if (decomposition_mode == DecompositionMode::DataParallel) {
      barrier_workspace_size = 0;
      reduction_workspace_size = 0;
    }
    else {
      KernelHardwareInfo new_hw_info;
      new_hw_info.device_id = hw_info.device_id;
      new_hw_info.sm_count = hw_info.sm_count;
      new_hw_info.max_active_clusters = hw_info.max_active_clusters;
      if (new_hw_info.sm_count <= 0) {
        CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
            "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
        new_hw_info.sm_count = KernelHardwareInfo::query_device_multiprocessor_count(new_hw_info.device_id);
      }

      dim3 grid = get_grid_shape(
        problem_blocks,
        cluster_shape,
        new_hw_info,
        max_swizzle,
        raster_order_option,
        bypass_sm90_occupancy_calculation
      );
      uint64_t ctas_per_wave = grid.x * grid.y;
      uint64_t cluster_size = cluster_shape.m() * cluster_shape.n();
      uint64_t ctas_per_wave_in_full_clusters = (ctas_per_wave / cluster_size) * cluster_size; 
      uint32_t sk_tiles = get_num_sk_tiles(
        output_tiles,
        ctas_per_wave,
        cluster_size,
        static_cast<uint32_t>(k_tiles_per_output_tile),
        decomposition_mode
        , ctas_per_wave_in_full_clusters 
      );
      uint64_t ctas_per_sk_wave = ctas_per_wave;
      ctas_per_sk_wave = ctas_per_wave_in_full_clusters; 
      uint64_t sk_units = get_num_sk_units(cluster_shape, ctas_per_sk_wave, sk_tiles, k_tiles_per_output_tile);
      uint64_t dp_tiles = output_tiles - sk_tiles;

      if (decomposition_mode == DecompositionMode::SplitK ||
         (decomposition_mode == DecompositionMode::Heuristic && splits > 1)) {
        splits = adjust_split_count(
          splits, new_hw_info.sm_count, k_tiles_per_output_tile
          , ktile_start_alignment_count 
        );
      }

      bool split_k_required = splits > 1 && (decomposition_mode == DecompositionMode::SplitK || decomposition_mode == DecompositionMode::Heuristic);
      bool split_k_selected = !split_k_required &&
                              decomposition_mode == DecompositionMode::Heuristic &&
                              sk_units > sk_tiles &&
                              sk_tiles != 0 &&
                              sk_units % sk_tiles == 0;

      if (split_k_required || split_k_selected) {
        // Basic split-K variant requires workspace for all output tiles
        barrier_workspace_size = get_barrier_workspace_size(output_tiles, mma_warp_groups, barrier_bits);
        reduction_workspace_size = get_reduction_workspace_size(output_tiles, tile_shape, accumulator_bits, num_accumulator_mtxs);
      }
      else {
        uint64_t reduction_tiles = sk_tiles;
        if (
          should_perform_separate_reduction(epilogue_subtile, sk_units, sk_tiles, dp_tiles, ctas_per_wave)
          ) {
          // In separate reduction, each peer writes to its own location in scratch space.
          // Thus, for separate reduction, we need as many reduction tiles per output tile
          // as there are the maximum number of peers that can collaborate on an output tile.
          reduction_tiles *= max_peers_per_tile(sk_units, sk_tiles);
        }

        // Though separate reduction requires a larger reduction workspace, only one barrier
        // is needed per output tile. Each peer will increment the barrier by one once the peer has
        // written its accumulator to scratch space. The separate reduction unit will only begin
        // performing the reduction when the barrier has reached the number of peers for the output tile.
        barrier_workspace_size = get_barrier_workspace_size(sk_tiles, mma_warp_groups, barrier_bits);
        reduction_workspace_size = get_reduction_workspace_size(reduction_tiles, tile_shape, accumulator_bits, num_accumulator_mtxs);
      }
    }
  }
  #endif // !defined(__CUDACC_RTC__)

  // Returns whether the kernel is configured in a manner for which separate reduction should be used
  CUTLASS_HOST_DEVICE
  static bool
  should_perform_separate_reduction(uint32_t, uint64_t, uint64_t, uint64_t, uint64_t) {
    // Separate reduction is temporarily disabled, pending fixes
    return false;
  }

  // Get the amount of scratch workspace needed for the kernel. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  static size_t
  get_workspace_size(
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    ReductionMode reduction_mode,
    uint32_t mma_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile,
    uint32_t num_accumulator_mtxs,
    uint32_t ktile_start_alignment_count = 1) {

    dim3 problem_blocks = UnderlyingParams::get_tiled_cta_shape_mnl(problem_shape, tile_shape, cluster_shape);
    uint32_t k_tiles_per_output_tile = (problem_shape.k() + tile_shape.k() - 1) / tile_shape.k();

    return get_workspace_size(
      problem_blocks,
      k_tiles_per_output_tile,
      tile_shape,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle,
      raster_order_option,
      decomposition_mode,
      reduction_mode,
      mma_warp_groups,
      barrier_bits,
      element_accumulator_bits,
      epilogue_subtile,
      num_accumulator_mtxs,
      ktile_start_alignment_count
    );
  }

  // Version of get_workspace_size that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  static size_t
  get_workspace_size(
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    ReductionMode reduction_mode,
    uint32_t mma_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile = 1,
    uint32_t num_accumulator_mtxs = 1,
    uint32_t ktile_start_alignment_count = 1,
    bool bypass_sm90_occupancy_calculation=false) {

    size_t barrier_workspace_size = 0;
    size_t reduction_workspace_size = 0;

    #if !defined(__CUDACC_RTC__)
      get_workspace_component_sizes(
        problem_blocks,
        k_tiles_per_output_tile,
        tile_shape,
        cluster_shape,
        barrier_workspace_size,
        reduction_workspace_size,
        hw_info,
        splits,
        max_swizzle,
        raster_order_option,
        decomposition_mode,
        reduction_mode,
        mma_warp_groups,
        barrier_bits,
        element_accumulator_bits,
        epilogue_subtile,
        num_accumulator_mtxs,
        ktile_start_alignment_count,
        bypass_sm90_occupancy_calculation
      );
    #endif

    return barrier_workspace_size + reduction_workspace_size;
  }

  // Initialize the workspace to be used for the kernel. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  static cutlass::Status
  initialize_workspace(
    void* workspace,
    cudaStream_t stream,
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    ReductionMode reduction_mode,
    uint32_t mma_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile,
    CudaHostAdapter* cuda_adapter = nullptr,
    uint32_t ktile_start_alignment_count = 1) {

    dim3 problem_blocks = UnderlyingParams::get_tiled_cta_shape_mnl(problem_shape, tile_shape, cluster_shape);
    uint32_t k_tiles_per_output_tile = (problem_shape.k() + tile_shape.k() - 1) / tile_shape.k();

    return initialize_workspace(
      workspace,
      stream,
      problem_blocks,
      k_tiles_per_output_tile,
      tile_shape,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle,
      raster_order_option,
      decomposition_mode,
      reduction_mode,
      mma_warp_groups,
      barrier_bits,
      element_accumulator_bits,
      epilogue_subtile,
      1,
      cuda_adapter,
      ktile_start_alignment_count
    );
  }

  // Version of initialize_workspace that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  static cutlass::Status
  initialize_workspace(
    void* workspace,
    cudaStream_t stream,
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    ReductionMode reduction_mode,
    uint32_t mma_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile = 1,
    uint32_t num_accumulator_mtxs = 1,
    CudaHostAdapter* cuda_adapter = nullptr,
    uint32_t ktile_start_alignment_count = 1,
    bool bypass_sm90_occupancy_calculation=false) {

    #if !defined(__CUDACC_RTC__)
      uint64_t barrier_workspace_size = 0;
      uint64_t reduction_workspace_size = 0;

      get_workspace_component_sizes(
        problem_blocks,
        k_tiles_per_output_tile,
        tile_shape,
        cluster_shape,
        barrier_workspace_size,
        reduction_workspace_size,
        hw_info,
        splits,
        max_swizzle,
        raster_order_option,
        decomposition_mode,
        reduction_mode,
        mma_warp_groups,
        barrier_bits,
        element_accumulator_bits,
        epilogue_subtile,
        num_accumulator_mtxs,
        ktile_start_alignment_count,
        bypass_sm90_occupancy_calculation
      );

      if (barrier_workspace_size > 0) {
        if (workspace == nullptr) {
          return Status::kErrorWorkspaceNull;
        }

        // Only the barrier workspace needs to be cleared for stream-K.
        // Barrier workspace follows reduction workspace.
        uint8_t* barrier_workspace = reinterpret_cast<uint8_t*>(workspace) + reduction_workspace_size;
        return zero_workspace(static_cast<void*>(barrier_workspace), barrier_workspace_size, stream, cuda_adapter);
      }
    #endif // !defined(__CUDACC_RTC__)

    return Status::kSuccess;
  }

  // Set params for basic parameters, which will not affected by different decompositions.
  void
  set_params_base(UnderlyingParams const& underlying_params, void* reduction_workspace) {
    divmod_cluster_shape_major_ = underlying_params.divmod_cluster_shape_major_;
    divmod_cluster_shape_minor_ = underlying_params.divmod_cluster_shape_minor_;
    divmod_cluster_blk_major_ = underlying_params.divmod_cluster_blk_major_;
    log_swizzle_size_ = underlying_params.log_swizzle_size_;
    raster_order_ = underlying_params.raster_order_;
    reduction_workspace_ = reduction_workspace;
  }

  void
  set_params_basic(
    UnderlyingParams const& underlying_params,
    dim3 problem_blocks,
    GemmCoord cluster_shape,
    uint32_t splits,
    uint32_t k_tiles_per_output_tile,
    ReductionMode reduction_mode) {

    auto blocks_l = problem_blocks.z;
    auto blocks_m = round_up(problem_blocks.x,
                             (1 << underlying_params.log_swizzle_size_) * cluster_shape.m());
    auto blocks_n = round_up(problem_blocks.y,
                             (1 << underlying_params.log_swizzle_size_) * cluster_shape.n());

    divmod_batch_ = FastDivmodU64(blocks_m * blocks_n);
    divmod_tiles_per_output_tile_ = FastDivmod(k_tiles_per_output_tile);
    divmod_sk_groups_ = FastDivmodU64(1u);
    auto cluster_size = underlying_params.divmod_cluster_shape_major_.divisor *
                        underlying_params.divmod_cluster_shape_minor_.divisor;
    divmod_clusters_mnl_ = FastDivmodU64((blocks_m * blocks_n * blocks_l) / cluster_size);
    divmod_splits_ = FastDivmod(splits);
    units_per_problem_ = blocks_m * blocks_n * blocks_l;
    big_units_ = k_tiles_per_output_tile % splits;
    reduction_mode_ = reduction_mode;
    divmod_k_tiles_per_sk_unit_ = FastDivmod(k_tiles_per_output_tile / splits);
    divmod_k_tiles_per_sk_big_unit_ = FastDivmod(k_tiles_per_output_tile / splits + 1);

    // No stream-K work is performed for "basic" data-parallel and split-K decompositions
    sk_tiles_ = 0;
    sk_units_ = 0;
    divmod_sk_units_per_group_ = FastDivmodU64(1u);
    separate_reduction_units_ = 0;
  }

  // Set params for streamk(streamk, separate-reduction included) decomposition.
  void
  set_params_stream_k(
    UnderlyingParams const& underlying_params,
    uint32_t k_tiles_per_output_tile,
    uint32_t groups,
    uint32_t sk_tiles,
    uint64_t sk_units,
    uint64_t cluster_size,
    uint64_t dp_units,
    uint64_t k_tiles_per_group,
    uint64_t k_tiles_per_sk_unit,
    uint64_t sk_big_groups,
    ReductionMode reduction_mode,
    uint32_t epilogue_subtile,
    uint32_t reduction_units) {
    // stream-k and separate-reduction decompostions
    divmod_batch_ = underlying_params.divmod_batch_;
    divmod_tiles_per_output_tile_ = FastDivmod(k_tiles_per_output_tile);
    divmod_sk_groups_ = FastDivmodU64(static_cast<uint64_t>(groups));
    divmod_sk_units_per_group_ = FastDivmodU64(static_cast<uint64_t>(sk_units / groups));

    // Override divmod_clusters_mnl_ to be the number of cluster-sized stream-K units.
    // This setting ensures that the use of this divmod for stream-K decompositions
    // is essentially a no-op.
    divmod_clusters_mnl_ = FastDivmodU64(sk_units / cluster_size);
    divmod_splits_ = FastDivmod(1);
    units_per_problem_ = static_cast<uint32_t>(dp_units + sk_units);

    // Assign big_units_ assuming that group count == 1. This is unused by stream-K
    // when group count > 1.
    auto big_units_in_ctas = k_tiles_per_group % sk_units;

    // Store big_units in terms of clusters. big_units_in_ctas is guaranteed to be divisible
    // by cluster_size because both k_tiles_per_group and k_tiles_per_sk_unit must be a multiple
    // of cluster_size.
    auto big_units_in_clusters = big_units_in_ctas / cluster_size;
    big_units_ = static_cast<uint32_t>(big_units_in_clusters);

    big_groups_ = static_cast<uint32_t>(sk_big_groups);
    sk_tiles_ = sk_tiles;
    sk_units_ = static_cast<uint32_t>(sk_units);
    divmod_k_tiles_per_sk_unit_ = FastDivmod(static_cast<uint32_t>(k_tiles_per_sk_unit));
    divmod_k_tiles_per_sk_big_unit_ = FastDivmod(static_cast<uint32_t>(k_tiles_per_sk_unit + 1));
    reduction_mode_ = reduction_mode;
    divmod_epilogue_subtile_ = FastDivmodU64(epilogue_subtile);
    separate_reduction_units_ = reduction_units;
  }

  private:
  // Round up number of bytes to the nearest multiple of L2 cache line alignment
  CUTLASS_HOST_DEVICE
  static size_t
  round_up_to_l2_alignment(size_t bytes) {
    constexpr size_t L2CacheLineSizeBytes = 128u;
    return (bytes + L2CacheLineSizeBytes - 1) / L2CacheLineSizeBytes * L2CacheLineSizeBytes;
  }

  CUTLASS_HOST_DEVICE
  static int adjust_split_count(
      int splits,
      int sm_count,
      uint32_t k_tiles_per_output_tile
      , uint32_t ktile_start_alignment_count 
      ) {
    // Don't split by more than the available number of SMs
    if (splits > sm_count) {
      splits = sm_count;
    }

    // Don't split by more than the K tile iterations
    if (static_cast<uint32_t>(splits) > k_tiles_per_output_tile) {
      splits = k_tiles_per_output_tile;
    }

    // If k_tiles_per_output_tiles / splits == 1, there will be one k_tile per cta
    //   and this violate k_tile start from even requirements. Thus we need to
    //   reduce the number of splits.
    if (ktile_start_alignment_count > 1u && 
          splits > 1 &&
          k_tiles_per_output_tile / static_cast<uint32_t>(splits) == 1) {
      splits = k_tiles_per_output_tile / ktile_start_alignment_count;
    } 
    return splits;
  }
};


////////////////////////////////////////////////////////////////////////////////

// Parameters for SM90 persistent group scheduler (only used for Grouped Gemms)
template<class GroupProblemShape>
struct PersistentTileSchedulerSm90GroupParams {
  using RasterOrder = cutlass::gemm::kernel::detail::RasterOrder;
  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  FastDivmodU64Pow2 divmod_cluster_shape_major_{};
  FastDivmodU64Pow2 divmod_cluster_shape_minor_{};
  FastDivmodU64 divmod_cta_shape_m_{};
  FastDivmodU64 divmod_cta_shape_n_{};

  uint64_t blocks_across_problem_ = 0;
  bool pre_processed_problem_shapes = true;
  int32_t log_swizzle_size_ = 0;
  RasterOrder raster_order_ = RasterOrder::AlongN;

  GroupProblemShape problem_shapes_;
  GemmCoord cta_shape_;
  GemmCoord cluster_shape_;

  // Version of initialize that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  void
  initialize(
    dim3 problem_blocks,
    GroupProblemShape problem_shapes,
    GemmCoord cta_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option
  ) {

    CUTLASS_UNUSED(hw_info);

    // Round up to nearest multiple of swizzle_size along each mode
    auto log_swizzle_size = get_log_swizzle_size(problem_blocks.x, problem_blocks.y, max_swizzle_size);
    auto problem_blocks_m = round_up(problem_blocks.x, (1 << log_swizzle_size) * cluster_shape.m());
    auto problem_blocks_n = round_up(problem_blocks.y, (1 << log_swizzle_size) * cluster_shape.n());

    RasterOrder raster_order = get_rasterization_order(
      problem_blocks_m,
      problem_blocks_n,
      raster_order_option
    );

    //
    // Set members
    //
    problem_shapes_ = problem_shapes;
    cta_shape_ = cta_shape;
    cluster_shape_ = cluster_shape;

    blocks_across_problem_ = problem_blocks.x * problem_blocks.y * problem_blocks.z;
    pre_processed_problem_shapes = problem_shapes.is_host_problem_shape_available();
    log_swizzle_size_ = log_swizzle_size;
    raster_order_ = raster_order;

    if (raster_order == RasterOrder::AlongN) {
      divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.n());
      divmod_cluster_shape_minor_ = FastDivmodU64Pow2(cluster_shape.m());
    }
    else {
      divmod_cluster_shape_major_ = FastDivmodU64Pow2(cluster_shape.m());
      divmod_cluster_shape_minor_ = FastDivmodU64Pow2(cluster_shape.n());
    }

    divmod_cta_shape_m_ = FastDivmodU64(cta_shape_.m());
    divmod_cta_shape_n_ = FastDivmodU64(cta_shape_.n());
  }

  // Version of get_tiled_cta_shape_mnl that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE
  static dim3
  get_tiled_cta_shape_mnl(GemmCoord cluster_shape, uint32_t cta_m, uint32_t cta_n) {
    // Round up to nearest multiple of cluster dim along each mode
    auto problem_blocks_m = ((cta_m + cluster_shape.m() - 1) / cluster_shape.m()) * cluster_shape.m();
    auto problem_blocks_n = ((cta_n + cluster_shape.n() - 1) / cluster_shape.n()) * cluster_shape.n();

    return {
      static_cast<uint32_t>(cta_m),
      static_cast<uint32_t>(cta_n),
      static_cast<uint32_t>(1) // Only a single batch per group is currently supported
    };
  }

  // Version of get_grid_shape that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    dim3 problem_blocks,
    GemmCoord cluster_shape,
    KernelHardwareInfo hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option,
    bool truncate_by_problem_size=true) {

    int const sm_count = hw_info.sm_count;
    int const max_active_clusters = hw_info.max_active_clusters;

    // Round up to nearest multiple of swizzle_size along each mode
    auto log_swizzle_size = get_log_swizzle_size(problem_blocks.x, problem_blocks.y, max_swizzle_size);
    auto problem_blocks_m = round_up(problem_blocks.x, (1 << log_swizzle_size) * cluster_shape.m());
    auto problem_blocks_n = round_up(problem_blocks.y, (1 << log_swizzle_size) * cluster_shape.n());

    int problem_blocks_total = problem_blocks_m * problem_blocks_n * problem_blocks.z;

    RasterOrder raster_order = get_rasterization_order(
      problem_blocks_m,
      problem_blocks_n,
      raster_order_option
    );

    dim3 launch_grid;

    if (raster_order == RasterOrder::AlongN) {
      launch_grid = dim3(cluster_shape.m(), 1, 1);
    }
    else {
      launch_grid = dim3(1, cluster_shape.n(), 1);
    }

    auto possibly_truncate = [&](int x, int y) {
      if (truncate_by_problem_size) {
        return platform::min(x, y);
      }
      else {
        return x;
      }
    };

    // The else path is generic, however, we can avoid some divs if we know cluster size is 1
    auto cluster_size = cluster_shape.m() * cluster_shape.n();
    if (cluster_size == 1) {
      if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = possibly_truncate(sm_count, problem_blocks_total);
      }
      else {
        launch_grid.x = possibly_truncate(sm_count, problem_blocks_total);
      }
    }
    // In case the maximum number of clusters that could co-exist on the target device is
    // already calculated using cudaOccupancyMaxActiveClusters
    else if (max_active_clusters != 0 && max_active_clusters * cluster_size <= sm_count) {
      if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = max_active_clusters * cluster_shape.n();
      }
      else {
        launch_grid.x = max_active_clusters * cluster_shape.m();
      }
      CUTLASS_TRACE_HOST("get_grid_shape(): Proposed GridDims by the scheduler using cudaOccupancyMaxActiveClusters = "
          "(" << launch_grid.x << ", " << launch_grid.y << ", " << launch_grid.z << ")\n");
    }
    else {
      // Optimal grid size calculation is based on
      // GH100: 8 GPCs, 72 TPCs (9 TPCs/GPC), 2 SMs/TPC, 144 SMs per full GPU
      // Hence, maximum SMs per GPC = 18
      constexpr int max_sm_per_gpc = 18;
      int cta_per_device = get_max_cta_occupancy(max_sm_per_gpc, cluster_shape, sm_count);

      if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = possibly_truncate(
            cta_per_device       / cluster_shape.m(),
            problem_blocks_total / cluster_shape.m());
      }
      else {
        launch_grid.x = possibly_truncate(
            cta_per_device       / cluster_shape.n(),
            problem_blocks_total / cluster_shape.n());
      }
      CUTLASS_TRACE_HOST("get_grid_shape(): Proposed GridDims by the scheduler using heuristics = "
          "(" << launch_grid.x << ", " << launch_grid.y << ", " << launch_grid.z << ")\n");
    }
    return launch_grid;
  }

  CUTLASS_HOST_DEVICE
  static int32_t
  get_log_swizzle_size(int problem_ctas_m, int problem_ctas_n, int max_swizzle_size) {
    int min_cta_dim = platform::min(problem_ctas_m, problem_ctas_n);
    if (max_swizzle_size >= 8 && min_cta_dim >= 6) {
      return 3;
    }
    else if (max_swizzle_size >= 4 && min_cta_dim >= 3) {
      return 2;
    }
    else if (max_swizzle_size >= 2 && min_cta_dim >= 2) {
      return 1;
    }
    else {
      return 0;
    }
  }

  CUTLASS_HOST_DEVICE
  static RasterOrder
  get_rasterization_order(
    uint32_t tiles_m,
    uint32_t tiles_n,
    RasterOrderOptions raster_order_option
  ) {

    if (raster_order_option == RasterOrderOptions::Heuristic) {
      if (tiles_n > tiles_m) {
        return RasterOrder::AlongM;
      }
      else {
        return RasterOrder::AlongN;
      }
    }
    else {
      switch (raster_order_option) {
        case RasterOrderOptions::AlongN:
          return RasterOrder::AlongN;
          break;
        default:
          return RasterOrder::AlongM;
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////


//
// Parameters for SM100 tile schedulers
//

// Parameters for SM100 persistent tile scheduler
struct PersistentTileSchedulerSm100Params {

  using UnderlyingParams = PersistentTileSchedulerSm90Params;

  using RasterOrder = UnderlyingParams::RasterOrder;
  using RasterOrderOptions = UnderlyingParams::RasterOrderOptions;

  uint32_t problem_tiles_m_ = 0;
  uint32_t problem_tiles_n_ = 0;
  uint32_t problem_tiles_l_ = 0;
  FastDivmod divmod_cluster_shape_m_{};
  FastDivmod divmod_cluster_shape_n_{};
  FastDivmod divmod_swizzle_size_{};
  RasterOrder raster_order_ = RasterOrder::AlongM;
  int32_t log_swizzle_size_ = 0;
  // Initializes members. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  void
  initialize(
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option
  ) {
    dim3 problem_blocks = UnderlyingParams::get_tiled_cta_shape_mnl(problem_shape, tile_shape, cluster_shape);
    initialize(
      problem_blocks,
      cluster_shape,
      hw_info,
      max_swizzle_size,
      raster_order_option
    );
  }

  void initialize_swizzle(
      dim3 problem_blocks,
      GemmCoord cluster_shape,
      KernelHardwareInfo const& hw_info,
      int max_swizzle_size,
      RasterOrderOptions raster_order_option) {

    raster_order_ = UnderlyingParams::get_rasterization_order(problem_tiles_m_, problem_tiles_n_, raster_order_option);
    if (raster_order_option == RasterOrderOptions::Heuristic && raster_order_ == RasterOrder::AlongN) {
      // The current implementation of AlongN rasterization for B100 requires swapping the number of clusters along the
      // X and Y dimensions of the grid. However, since the grid Y dimension has a smaller range of allowed values
      // than the grid X dimension, we must check whether the swapped grid would exceed the grid Y limit. If the
      // swapped grid would exceed this limit, simply rever to AlongM mode.
      //
      // Overflow in the swapped X dimension is not possible. At worst, there will be ((1 << 16) - 1) clusters
      // along the original Y dimension of the grid. Even if the cluster M mode is 16, the new grid X value
      // will be at most ((1 << 16) - 1) * 16, which is less than the grid X limit of ((1 << 31) - 1).
      uint32_t new_grid_y = problem_tiles_m_ * static_cast<uint32_t>(cluster_shape.n());

      if (new_grid_y > (1 << 16) - 1) {
        raster_order_ = RasterOrder::AlongM;
      }
    }

    if (max_swizzle_size <= 1) {
      // Set divisors directly to be zero to mark as unused
      divmod_swizzle_size_.divisor = 0;
    }
    else {
      divmod_swizzle_size_ = FastDivmod(max_swizzle_size);
    }
  }

  // Version of initialize that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  void
  initialize(
      dim3 problem_blocks,
      GemmCoord cluster_shape,
      KernelHardwareInfo const& hw_info,
      int max_swizzle_size,
      RasterOrderOptions raster_order_option
  ) {

    // Cluster counters in m, n and l dimensions of the problem tiles
    problem_tiles_m_ = problem_blocks.x / cluster_shape.m();
    problem_tiles_n_ = problem_blocks.y / cluster_shape.n();
    problem_tiles_l_ = problem_blocks.z;
    divmod_cluster_shape_m_ = FastDivmod(cluster_shape.m());
    divmod_cluster_shape_n_ = FastDivmod(cluster_shape.n());

    initialize_swizzle(problem_blocks, cluster_shape, hw_info, max_swizzle_size, raster_order_option);
  }

  // Given the inputs, computes the physical grid we should launch.
  // This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    BatchedGemmCoord problem_shape,
    GemmCoord cta_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option
  ) {

    CUTLASS_UNUSED(cluster_shape);
    CUTLASS_UNUSED(hw_info);
    CUTLASS_UNUSED(max_swizzle_size);
    CUTLASS_UNUSED(raster_order_option);

    return get_tiled_cta_shape_mnl(problem_shape, cta_shape, cluster_shape);
  }

  // Get the number of CTA tiles in this problem. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  CUTLASS_HOST_DEVICE
  static dim3
  get_tiled_cta_shape_mnl(
    BatchedGemmCoord problem_shape,
    GemmCoord cta_shape,
    GemmCoord cluster_shape) {

    return UnderlyingParams::get_tiled_cta_shape_mnl(problem_shape, cta_shape, cluster_shape);
  }

  // Get the amount of scratch workspace needed for the kernel. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  static size_t
  get_workspace_size(
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int max_swizzle,
    RasterOrderOptions raster_order_option
  ) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape, tile_shape, cluster_shape);
    return get_workspace_size(
      problem_blocks,
      cluster_shape,
      hw_info,
      max_swizzle,
      raster_order_option
    );
  }

  // Version of get_workspace_size that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  static size_t
  get_workspace_size(
    dim3 problem_blocks,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int max_swizzle,
    RasterOrderOptions raster_order_option
  ) {

    CUTLASS_UNUSED(problem_blocks);
    CUTLASS_UNUSED(cluster_shape);
    CUTLASS_UNUSED(hw_info);
    CUTLASS_UNUSED(max_swizzle);
    CUTLASS_UNUSED(raster_order_option);

    return 0;
  }

  // Initialize the workspace to be used for the kernel. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  static cutlass::Status
  initialize_workspace(
    void* workspace,
    cudaStream_t stream,
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    CudaHostAdapter *cuda_adapter = nullptr
  ) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape, tile_shape, cluster_shape);
    return initialize_workspace(
      workspace,
      stream,
      problem_blocks,
      cluster_shape,
      hw_info,
      max_swizzle,
      raster_order_option,
      cuda_adapter
    );
  }

  // Version of initialize_workspace that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  static cutlass::Status
  initialize_workspace(
    void* workspace,
    cudaStream_t stream,
    dim3 problem_blocks,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    CudaHostAdapter *cuda_adapter = nullptr
  ) {

    CUTLASS_UNUSED(workspace);
    CUTLASS_UNUSED(stream);
    CUTLASS_UNUSED(problem_blocks);
    CUTLASS_UNUSED(cluster_shape);
    CUTLASS_UNUSED(hw_info);
    CUTLASS_UNUSED(max_swizzle);
    CUTLASS_UNUSED(raster_order_option);

    return cutlass::Status::kSuccess;
  }
};

////////////////////////////////////////////////////////////////////////////////

// Parameters for SM100 persistent stream-K tile scheduler
struct PersistentTileSchedulerSm100StreamKParams {
  using UnderlyingParams = PersistentTileSchedulerSm100Params;
  using UnderlyingStreamKParams = PersistentTileSchedulerSm90StreamKParams;
  using RasterOrderOptions = UnderlyingParams::RasterOrderOptions;
  using ReductionMode = UnderlyingStreamKParams::ReductionMode;
  using DecompositionMode = UnderlyingStreamKParams::DecompositionMode;

  using RasterOrder = UnderlyingParams::RasterOrder;
  RasterOrder raster_order_ = RasterOrder::AlongM;
  int32_t log_swizzle_size_ = 0;

  UnderlyingStreamKParams sk_params_{};
  UnderlyingParams sm100_params_{};

  // Initializes members. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  void
  initialize(
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option,
    ReductionMode reduction_mode,
    DecompositionMode decomposition_mode,
    void* workspace,
    uint32_t ktile_start_alignment_count = 1u
  ) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape, tile_shape, cluster_shape);

    // Number of k tiles in each output tile
    uint32_t k_tiles_per_output_tile = (problem_shape.k() + tile_shape.k() - 1) / tile_shape.k();

    initialize(
      problem_blocks,
      k_tiles_per_output_tile,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle_size,
      raster_order_option,
      reduction_mode,
      decomposition_mode,
      workspace,
      ktile_start_alignment_count
    );
  }

  // Version of initialize that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  void
  initialize(
    dim3 problem_blocks,
    uint32_t k_tile_per_output_tile,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option,
    ReductionMode reduction_mode,
    DecompositionMode decomposition_mode,
    void* workspace,
    uint32_t ktile_start_alignment_count = 1u
  ) {
    sk_params_.initialize(
      problem_blocks,
      k_tile_per_output_tile,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle_size,
      raster_order_option,
      reduction_mode,
      decomposition_mode,
      workspace,
      /*epilogue_subtile=*/1,
      ktile_start_alignment_count,
      /*bypass_sm90_occupancy_calculation=*/true
    );

    log_swizzle_size_ = sk_params_.log_swizzle_size_;
    raster_order_ = sk_params_.raster_order_;

    sm100_params_.initialize(
      problem_blocks,
      cluster_shape,
      hw_info,
      0, // Override max_swizzle_size to be 0, since the SM100 stream-K scheduler handles swizzling on its own
      RasterOrderOptions::AlongM // Override raster_order to be AlongM, since the SM100 stream-K scheduler does not require grid swapping for raster order selection
    );
  }

  // Get the number of CTA tiles in this problem.
  CUTLASS_HOST_DEVICE
  static dim3
  get_tiled_cta_shape_mnl(
    BatchedGemmCoord problem_shape,
    GemmCoord cta_shape,
    GemmCoord cluster_shape) {

    return UnderlyingParams::get_tiled_cta_shape_mnl(problem_shape, cta_shape, cluster_shape);
  }

  // Given the inputs, computes the physical grid we should launch.
  // This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  CUTLASS_HOST_DEVICE
  dim3
  get_grid_shape(BatchedGemmCoord problem_shape, GemmCoord cta_shape, GemmCoord cluster_shape) const {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape, cta_shape, cluster_shape);

    return get_grid_shape(problem_blocks, cluster_shape);
  }

  // Version of get_grid_shape that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE
  dim3
  get_grid_shape(dim3 problem_blocks, GemmCoord cluster_shape) const {
    if (sk_params_.sk_units_ > 0) {
      // For stream-K cases, we would, ideally, launch a linear grid of size `sk_params_.units_per_problem_`.
      // However doing so raises two potential issues:
      //   (a) the total number of tiles in the kernel may exceed the amount that can fit in a single
      //       returned value of a CLC query
      //   (b) the launched grid would not respect cluster-size divisibility requirements
      //
      // To circumvent these issues, we must distribute the `sk_params_.units_per_problem_` units of work
      // across the X, Y, and Z dimensions of the grid, while ensuring that the X and Y dimensions are
      // divisible by cluster size (we ignore Z, as all CUTLASS kernels currently use a cluster shape
      // of 1 in the Z dimension).
      //
      // For convenience, we launch this as "waves" of `sk_params_.sk_units_` CTAs, with the wave count being
      // the Z dimension of the grid, and the `sk_params_.sk_units_` CTAs per wave being distributed across
      // the X and Y dimensions of the grid in a way that alingns with cluster divisibility requirements.
      //
      // Thus, the grid that is launched looks like:
      //   grid = dim3(sk_units_ / cluster.y, cluster.y, waves)
      //
      // We place sk_units_ / cluster.y in the X dimension of the grid because the CLC query feature
      // allocates more bits for the X index values returned in the query.
      //

      // For most cases, `sk_params_.sk_units_` will equal the number of available SMs, so this grid will
      // naturally represent waves in the true hardware sense.
      //
      // However, there are some corner cases in which fewer stream-K units are used than the full SM count
      // (e.g., if using the full SM count would result in stream-K units that are assigned fewer than the
      // minimum number of K tile iterations). In these cases, `sk_params_.units_per_problem_` may not be
      // divisible by `sk_params_.sk_units_`, since any data-parallel work performed alongside stream-K
      // work is always done in terms of waves of CTAs of number equal to the number of available SMs.
      // Therefore, we take the ceiling of the division when determining wave count, and allow the underlying
      // stream-K scheduler to determine which indices are in bounds.
      uint32_t waves = static_cast<uint32_t>(
        (sk_params_.units_per_problem_ + sk_params_.sk_units_ - 1) / sk_params_.sk_units_);

      return dim3(
        sk_params_.sk_units_ / cluster_shape.n(),
        cluster_shape.n(),
        waves
      );
    }
    else {
      // Grid launch for data-parallel and basic split-K decomposition. When data-parallel
      // mode is used, params.sk_params_.splits = 1.
      return dim3(problem_blocks.x, problem_blocks.y, problem_blocks.z * sk_params_.divmod_splits_.divisor);
    }
  }

  // Get the amount of scratch workspace needed for the kernel. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  static size_t
  get_workspace_size(
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    ReductionMode reduction_mode,
    uint32_t reduction_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t ktile_start_alignment_count = 1
  ) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape, tile_shape, cluster_shape);
    uint32_t k_tiles_per_output_tile = (problem_shape.k() + tile_shape.k() - 1) / tile_shape.k();

    return get_workspace_size(
      problem_blocks,
      k_tiles_per_output_tile,
      tile_shape,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle,
      raster_order_option,
      decomposition_mode,
      reduction_mode,
      reduction_warp_groups,
      barrier_bits,
      element_accumulator_bits,
      ktile_start_alignment_count
    );
  }

  // Version of get_workspace_size that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  static size_t
  get_workspace_size(
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    ReductionMode reduction_mode,
    uint32_t reduction_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile = 1,
    uint32_t num_accumulator_mtxs = 1,
    uint32_t ktile_start_alignment_count = 1
  ) {
    return UnderlyingStreamKParams::get_workspace_size(
      problem_blocks,
      k_tiles_per_output_tile,
      tile_shape,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle,
      raster_order_option,
      decomposition_mode,
      reduction_mode,
      reduction_warp_groups,
      barrier_bits,
      element_accumulator_bits,
      epilogue_subtile,
      num_accumulator_mtxs,
      ktile_start_alignment_count,
      /*bypass_sm90_occupancy_calculation=*/true
    );
  }

  // Initialize the workspace to be used for the kernel. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  static cutlass::Status
  initialize_workspace(
    void* workspace,
    cudaStream_t stream,
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    ReductionMode reduction_mode,
    uint32_t reduction_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile = 1,
    uint32_t num_accumulator_mtxs = 1,
    CudaHostAdapter *cuda_adapter = nullptr,
    uint32_t ktile_start_alignment_count = 1
  ) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape, tile_shape, cluster_shape);
    uint32_t k_tiles_per_output_tile = (problem_shape.k() + tile_shape.k() - 1) / tile_shape.k();

    return initialize_workspace(
      workspace,
      stream,
      problem_blocks,
      k_tiles_per_output_tile,
      tile_shape,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle,
      raster_order_option,
      decomposition_mode,
      reduction_mode,
      reduction_warp_groups,
      barrier_bits,
      element_accumulator_bits,
      epilogue_subtile,
      num_accumulator_mtxs,
      cuda_adapter,
      ktile_start_alignment_count
    );
  }

  // Version of initialize_workspace that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  static cutlass::Status
  initialize_workspace(
    void* workspace,
    cudaStream_t stream,
    dim3 problem_blocks,
    uint32_t k_tiles_per_output_tile,
    GemmCoord tile_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int splits,
    int max_swizzle,
    RasterOrderOptions raster_order_option,
    DecompositionMode decomposition_mode,
    ReductionMode reduction_mode,
    uint32_t reduction_warp_groups,
    uint32_t barrier_bits,
    uint32_t element_accumulator_bits,
    uint32_t epilogue_subtile = 1,
    uint32_t num_accumulator_mtxs = 1,
    CudaHostAdapter *cuda_adapter = nullptr,
    uint32_t ktile_start_alignment_count = 1
  ) {
    return UnderlyingStreamKParams::initialize_workspace(
      workspace,
      stream,
      problem_blocks,
      k_tiles_per_output_tile,
      tile_shape,
      cluster_shape,
      hw_info,
      splits,
      max_swizzle,
      raster_order_option,
      decomposition_mode,
      reduction_mode,
      reduction_warp_groups,
      barrier_bits,
      element_accumulator_bits,
      epilogue_subtile,
      num_accumulator_mtxs,
      cuda_adapter,
      ktile_start_alignment_count,
      /*bypass_sm90_occupancy_calculation=*/true
    );
  }
};

////////////////////////////////////////////////////////////////////////////////
// Parameters for SM100 persistent group scheduler (only used for Grouped Gemms)
template<class GroupProblemShape>
struct PersistentTileSchedulerSm100GroupParams {

  using UnderlyingSm90Params = PersistentTileSchedulerSm90GroupParams<GroupProblemShape>;
  using RasterOrder = cutlass::gemm::kernel::detail::RasterOrder;
  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  UnderlyingSm90Params params_sm90_{};

  // Version of initialize that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  void
  initialize(
    dim3 problem_blocks,
    GroupProblemShape problem_shapes,
    GemmCoord cta_shape,
    GemmCoord cluster_shape,
    KernelHardwareInfo const& hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option
  ) {

    params_sm90_.initialize(
      problem_blocks,
      problem_shapes,
      cta_shape,
      cluster_shape,
      hw_info,
      max_swizzle_size,
      raster_order_option
    );
  }

  // Version of get_tiled_cta_shape_mnl that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE
  static dim3
  get_tiled_cta_shape_mnl(GemmCoord cluster_shape, uint32_t cta_m, uint32_t cta_n) {
    return UnderlyingSm90Params::get_tiled_cta_shape_mnl(cluster_shape, cta_m, cta_n);
  }

  // Version of get_grid_shape that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using CuTe algebra for calculating tile shapes is easiest.
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    dim3 problem_blocks,
    GemmCoord cluster_shape,
    KernelHardwareInfo hw_info,
    int max_swizzle_size,
    RasterOrderOptions raster_order_option,
    bool truncate_by_problem_size = true,
    bool is_static_cluster_shape = false) {

    int const sm_count = hw_info.sm_count;
    int const max_active_clusters = hw_info.max_active_clusters;

    // Round up to nearest multiple of swizzle_size along each mode
    auto log_swizzle_size = get_log_swizzle_size(problem_blocks.x, problem_blocks.y, max_swizzle_size);
    auto problem_blocks_m = round_up(problem_blocks.x, (1 << log_swizzle_size) * cluster_shape.m());
    auto problem_blocks_n = round_up(problem_blocks.y, (1 << log_swizzle_size) * cluster_shape.n());

    int problem_blocks_total = problem_blocks_m * problem_blocks_n * problem_blocks.z;

    RasterOrder raster_order = get_rasterization_order(
      problem_blocks_m,
      problem_blocks_n,
      raster_order_option
    );

    dim3 launch_grid;

    if (raster_order == RasterOrder::AlongN) {
      launch_grid = dim3(cluster_shape.m(), 1, 1);
    }
    else {
      launch_grid = dim3(1, cluster_shape.n(), 1);
    }

    auto possibly_truncate = [&](int x, int y) {
      if (truncate_by_problem_size) {
        return platform::min(x, y);
      }
      else {
        return x;
      }
    };
    
    if (is_static_cluster_shape) {
      // The else path is generic, however, we can avoid some divs if we know cluster size is 1
      auto cluster_size = cluster_shape.m() * cluster_shape.n();
      if (cluster_size == 1) {
        if (raster_order == RasterOrder::AlongN) {
          launch_grid.y = possibly_truncate(sm_count, problem_blocks_total);
        }
        else {
          launch_grid.x = possibly_truncate(sm_count, problem_blocks_total);
        }
      }
      // In case the maximum number of clusters that could co-exist on the target device is
      // already calculated using cudaOccupancyMaxActiveClusters
      else if (max_active_clusters != 0 && max_active_clusters * cluster_size <= sm_count) {
        if (raster_order == RasterOrder::AlongN) {
          launch_grid.y = max_active_clusters * cluster_shape.n();
        }
        else {
          launch_grid.x = max_active_clusters * cluster_shape.m();
        }
        CUTLASS_TRACE_HOST("get_grid_shape(): Proposed GridDims by the scheduler using cudaOccupancyMaxActiveClusters = "
            "(" << launch_grid.x << ", " << launch_grid.y << ", " << launch_grid.z << ")\n");
      }
      else {
        constexpr int max_sm_per_gpc = 20;
        int cta_per_device = get_max_cta_occupancy(max_sm_per_gpc, cluster_shape, sm_count);
        if (raster_order == RasterOrder::AlongN) {
          launch_grid.y = possibly_truncate(
              cta_per_device       / cluster_shape.m(),
              problem_blocks_total / cluster_shape.m());
        }
        else {
          launch_grid.x = possibly_truncate(
              cta_per_device       / cluster_shape.n(),
              problem_blocks_total / cluster_shape.n());
        }
        CUTLASS_TRACE_HOST("get_grid_shape(): Proposed GridDims by the scheduler using heuristics = "
            "(" << launch_grid.x << ", " << launch_grid.y << ", " << launch_grid.z << ")\n");
      }
    }
    else {
      // With preferred clusters, we can launch the largest possible persistent grid (rounded up to cluster dims) 
      if (raster_order == RasterOrder::AlongN) {
        launch_grid.y = ((possibly_truncate(sm_count, problem_blocks_total) / cluster_shape.m()) / cluster_shape.n()) * cluster_shape.n();
      }
      else {
        launch_grid.x = ((possibly_truncate(sm_count, problem_blocks_total) / cluster_shape.n()) / cluster_shape.m()) * cluster_shape.m();
      }
      CUTLASS_TRACE_HOST("get_grid_shape(): Proposed GridDims by the scheduler using preferred clusters = "
          "(" << launch_grid.x << ", " << launch_grid.y << ", " << launch_grid.z << ")\n");
    }
    return launch_grid;
  }

  CUTLASS_HOST_DEVICE
  static int32_t
  get_log_swizzle_size(int problem_ctas_m, int problem_ctas_n, int max_swizzle_size) {
    return UnderlyingSm90Params::get_log_swizzle_size(problem_ctas_m, problem_ctas_n, max_swizzle_size);
  }

  CUTLASS_HOST_DEVICE
  static RasterOrder
  get_rasterization_order(
    uint32_t tiles_m,
    uint32_t tiles_n,
    RasterOrderOptions raster_order_option
  ) {
    return UnderlyingSm90Params::get_rasterization_order(tiles_m, tiles_n, raster_order_option);
  }
};

////////////////////////////////////////////////////////////////////////////////


} // namespace detail
} // namespace kernel
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
