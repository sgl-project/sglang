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
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

struct DefaultFusion {
  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return ceil_div(get<3>(problem_size), get<1>(tile_shape));
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_masked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return get_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return 0;
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void before_softmax(
    AccQK& acc_qk,
    IndexQK const& index_qk,
    ProblemSize const& problem_size

  ) {
    return;
  }
};

struct ResidualFusion : DefaultFusion {

  using Base = DefaultFusion;

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_masked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return 1;
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return get_trip_count(blk_coord, tile_shape, problem_size) - 1;
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void before_softmax(
    AccQK& acc_qk,
    IndexQK const& index_qk,
    ProblemSize const& problem_size
  ) {
    // This is useful is seqlen_k % kBlockN != 0 since it masks
    // the remaining elements out from softmax.
    // d % kHeadDim != 0 or seqlen_q % kBlockM do not suffer from similar
    // issues as they are transparently taken care of by TMA and the
    // epilogue, if it is instantiated with predication support.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      if (get<1>(pos) >= get<3>(problem_size)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

struct CausalFusion : DefaultFusion {

  using Base = DefaultFusion;

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    // See note below on different ways to think about causal attention
    // Again, we'd add the offset_q into the max_blocks_q calculation
    int max_blocks_k = Base::get_trip_count(blk_coord, tile_shape, problem_size);
    int max_blocks_q = ceil_div((get<0>(blk_coord) + 1) * get<0>(tile_shape), get<1>(tile_shape));
    return std::min(max_blocks_k, max_blocks_q);
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_masked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return ceil_div(get<0>(tile_shape), get<1>(tile_shape));
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_unmasked_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return get_trip_count(blk_coord, tile_shape, problem_size) - get_masked_trip_count(blk_coord, tile_shape, problem_size);
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void before_softmax(
    AccQK& acc_qk,
    IndexQK const& index_qk,
    ProblemSize const& problem_size
  ) {
    // There are two ways to do causal if N_Q != N_K
    // (1) is to assume that the Q is at the beginning of the matrix
    //    - this is what we demonstrate here
    // (2) is that it is at the end of the matrix
    //    - this is usually what we want for inference settings
    //      where we only compute the next row and use cache for the rest
    //    - if you'd like this, you only need to add an offset like so:
    //      get<0>(pos) + offset_q < get<1>(pos)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      if (get<0>(pos) < get<1>(pos)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }

};

template<class Base>
struct FusionBwdAdapter {
  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return Base{}.get_trip_count(select<1,0,2>(blk_coord), select<1,0,2>(tile_shape), select<0,1,3,2,4>(problem_size));
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void before_softmax(
    AccQK& acc_qk,
    IndexQK const& index_qk,
    ProblemSize const& problem_size
  ) {
    auto index_base = index_qk(_0{});
    auto index_shape = shape(index_qk);
    auto index_stride = transform_leaf(stride(index_qk), [](auto elem) {
      if constexpr (is_scaled_basis<decltype(elem)>::value) {
        if constexpr(decltype(elem.mode() == _0{})::value) {
          return ScaledBasis<decltype(elem.value()), 1>(elem.value());
        } else {
          return ScaledBasis<decltype(elem.value()), 0>(elem.value());
        }
      } else {
        return elem;
      }
    });
    auto index_qk_bwd = make_tensor(make_inttuple_iter(select<1,0>(index_base)), make_layout(index_shape, index_stride));
    Base{}.before_softmax(acc_qk, index_qk_bwd, problem_size);
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  bool is_contributing(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return true;
  }
};

template<>
struct FusionBwdAdapter<CausalFusion> {
  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  int get_trip_count(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    return get<2>(problem_size) / get<0>(TileShape{});
  }

  template<class AccQK, class IndexQK, class ProblemSize>
  CUTLASS_DEVICE
  void before_softmax(
    AccQK& acc_qk,
    IndexQK const& index_qk,
    ProblemSize const& problem_size

  ) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto pos = index_qk(i);
      if (get<1>(pos) < get<0>(pos)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }

  template<class BlkCoord, class TileShape, class ProblemSize>
  CUTLASS_DEVICE
  bool is_contributing(
    BlkCoord const& blk_coord,
    TileShape const& tile_shape,
    ProblemSize const& problem_size
  ) {
    int max_q = get<0>(blk_coord) * get<0>(tile_shape) + get<0>(tile_shape);
    int min_k = get<1>(blk_coord) * get<1>(tile_shape);
    return min_k <= max_q;
  }
};

}  // namespace cutlass::fmha::collective
