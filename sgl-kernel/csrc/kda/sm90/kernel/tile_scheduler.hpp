// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/kernel_hardware_info.h>

namespace kda::sm90::kernel {

using namespace cute;

struct WorkDesc {
  // coord
  int32_t seq_idx;
  int32_t head_idx;
  int64_t tok_offset;  // offset to the start of the start

  // shape
  int64_t seq_len;

  // update by mainloop
  int32_t tile_idx = 0;

  template <typename Params>
  CUTE_DEVICE bool is_valid(Params const& params) {
    return seq_idx >= 0 && seq_idx < params.num_seqs;
  }

  CUTE_DEVICE int32_t q_head_idx() const {
    return head_idx;
  }
  CUTE_DEVICE int32_t k_head_idx() const {
    return head_idx;
  }
  CUTE_DEVICE int32_t v_head_idx() const {
    return head_idx;
  }
  CUTE_DEVICE int32_t o_head_idx() const {
    return head_idx;
  }

  // compatible interface, for work without ChunkWiseParallel, chunk_len equals to seq_len
  CUTE_DEVICE int32_t chunk_len() const {
    return seq_len;
  }
};

struct IndividualTileScheduler {
  struct Params {
    dim3 grid;
    int32_t num_seqs;
    int32_t num_heads;
  };

  bool scheduled = false;  // a once flag

  CUTE_DEVICE
  IndividualTileScheduler(Params const& params) {}

  template <typename ProblemSize, typename ClusterShape, typename TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size,
      cutlass::KernelHardwareInfo const& hw_info,
      ClusterShape const& cluster_shape,
      TileShape const& tile_shape) {
    dim3 grid(0, 1, 1);
    grid.x = problem_size.num_seqs * problem_size.num_heads;
    DPRINTF(
        "to_underlying_arguments: grid:{.x:%d, .y:%d, .z:%d}, num_seqs:%d, num_heads:%d\n",
        grid.x,
        grid.y,
        grid.z,
        problem_size.num_seqs,
        problem_size.num_heads);
    return {
        .grid = grid,
        .num_seqs = problem_size.num_seqs,
        .num_heads = problem_size.num_heads,
    };
  }

  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  template <typename ProblemSize>
  CUTE_DEVICE WorkDesc get_next_work(Params params, ProblemSize const& problem_size) {
    int32_t seq_idx = blockIdx.x / params.num_heads;
    int32_t head_idx = blockIdx.x % params.num_heads;

    int32_t s = problem_size.cu_seqlens[seq_idx];
    int32_t e = problem_size.cu_seqlens[seq_idx + 1];
    int32_t seq_len = e - s;

    if (scheduled) {
      seq_idx = -1;
    } else {
      scheduled = true;
      DPRINTF0_W(
          "get_next_work: this_work={seq_idx:%d head_idx:%d tok_offset:%lld seq_len:%lld}\n",
          seq_idx,
          head_idx,
          s,
          seq_len);
    }

    return {
        .seq_idx = seq_idx,
        .head_idx = head_idx,
        .tok_offset = s,
        .seq_len = seq_len,
    };
  }
};

}  // namespace kda::sm90::kernel
