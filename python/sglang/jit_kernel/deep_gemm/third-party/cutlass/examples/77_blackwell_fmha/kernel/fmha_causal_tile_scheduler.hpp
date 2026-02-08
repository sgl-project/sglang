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
#include "cutlass/fast_math.h"

namespace cutlass::fmha::kernel {

////////////////////////////////////////////////////////////////////////////////

// Swizzle Q tile and H tile to improve L2 cache hit rate, 
// and launch the longest main loop first to keep most SMs busy.

struct CausalIndividualTileScheduler {
  
  static constexpr int TileQ = 16;
  static constexpr int TileH = 8;
  static constexpr int TileSize = TileQ * TileH;

  struct Params {
    dim3 grid;
    int tile_max_q;
    FastDivmod divmod_tile_col;
    FastDivmod divmod_tile_size;
    FastDivmod divmod_tile_head;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  CausalIndividualTileScheduler(Params const& params) : params(params) {}

  template<class ProblemSize, class ClusterShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size, KernelHardwareInfo hw_info,
      ClusterShape const& cluster_shape, TileShape const& tile_shape) {
    using namespace cute;

    dim3 grid(size<3,0>(problem_size), round_up(ceil_div(size<0>(problem_size), size<0>(tile_shape)), size<0>(cluster_shape)), size<3,1>(problem_size));
    // gridDim.x must multiple of TileH
    const int tile_col_count = grid.x / TileH;
    const int tile_max_q = grid.y / TileQ * TileQ;
    return Params{ grid , tile_max_q, tile_col_count, TileSize, TileH};
  }

  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;

    int tile_idx, tile_tail;
    params.divmod_tile_size(tile_idx, tile_tail, block_idx);

    int tile_row_idx, tile_col_idx;
    params.divmod_tile_col(tile_row_idx,tile_col_idx, tile_idx);

    int row_offset_in_tail, col_offset_in_tail;
    params.divmod_tile_head(row_offset_in_tail,col_offset_in_tail, tile_tail);

    const int row_idx = tile_row_idx * TileQ + row_offset_in_tail;
    const int col_idx = tile_col_idx * TileH + col_offset_in_tail;
    
    // last q tile launch first
    if(blockIdx.y >= params.tile_max_q) {
      return make_coord(int(gridDim.y - 1 - blockIdx.y), _0{}, make_coord(int(blockIdx.x), int(blockIdx.z)));
    } 

    return make_coord(int(gridDim.y) - 1 - row_idx, _0{}, make_coord(col_idx, int(blockIdx.z)));
  }

  CUTLASS_DEVICE
  CausalIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////

// Launch order: H Q B
struct CausalPersistentTileScheduler {

  struct Params {
    int num_blocks;
    FastDivmod divmod_h;
    FastDivmod divmod_m_block;
    FastDivmod divmod_b;

    KernelHardwareInfo hw_info;
  };

  int block_idx = 0;
  Params params;

  CUTLASS_DEVICE
  CausalPersistentTileScheduler(Params const& params) : block_idx(blockIdx.x), params(params) {}

  template<class ProblemSize, class ClusterShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size, KernelHardwareInfo hw_info,
      ClusterShape const& cluster_shape, TileShape const& tile_shape) {
    using namespace cute;
    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    hw_info.sm_count = sm_count;

    int num_m_blocks = cutlass::round_up(ceil_div(size<0>(problem_size), size<0>(tile_shape)), size<0>(cluster_shape));
    int num_blocks = num_m_blocks * size<3,0>(problem_size) * size<3,1>(problem_size);

    return Params {
      num_blocks,
      { size<3,0>(problem_size) }, { num_m_blocks}, { size<3,1>(problem_size) },
      hw_info
    };
  }

  static dim3 get_grid_shape(Params const& params) {
    dim3 grid(std::min(params.num_blocks, params.hw_info.sm_count), 1, 1);
    return grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return block_idx < params.num_blocks;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    int block_decode = block_idx;
    int m_block, bidb, bidh;
    params.divmod_h(block_decode, bidh, block_decode);
    params.divmod_m_block(block_decode, m_block, block_decode);
    params.divmod_b(block_decode, bidb, block_decode);
    return make_coord(m_block, _0{}, make_coord(bidh, bidb));
  }

  CUTLASS_DEVICE
  CausalPersistentTileScheduler& operator++() {
    block_idx += gridDim.x;
    return *this;
  }
};
////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::fmha::kernel
