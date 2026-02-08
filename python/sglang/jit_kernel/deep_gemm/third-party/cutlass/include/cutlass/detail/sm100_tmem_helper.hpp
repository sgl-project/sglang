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



/*! \file
    \brief TMEM Accumulator Helpers for SM100
*/

#pragma once

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"


namespace cutlass::detail{
constexpr uint32_t TmemColMask = 0x0000'FFFF;

template <class TmemTensor>
CUTE_HOST_DEVICE
static constexpr auto find_tmem_tensor_col_offset(TmemTensor tensor) {
  using namespace cute;
  return cosize(recast<uint32_t>(tensor).layout()) & TmemColMask;
}

template <int AccumulatorPipelineStageCount, bool IsOverlappingAccum,
          class TiledMma, class AccumulatorShape,
          class EpilogueTile>
CUTE_HOST_DEVICE 
static constexpr auto make_sm100_accumulator(TiledMma tiled_mma, AccumulatorShape acc_shape, EpilogueTile epilogue_tile) {
  using namespace cute;
  static_assert(rank(acc_shape) == 3 || (rank(acc_shape) == 4 && IsOverlappingAccum == false), 
    "Expect a rank >= 3 accumulator shape compatible with an SM100 tiled mma, Overlapping accumulators is only available for non-complex kernels");
  if constexpr (IsOverlappingAccum) {
    Tensor accumulators_tmp = TiledMma::make_fragment_C(append(acc_shape, Int<2>{}));
    return make_tensor(
        accumulators_tmp.data(),
        shape(accumulators_tmp),
        replace<3>(
            stride(accumulators_tmp),
            Int<(256 - size<1>(EpilogueTile{})) * stride<0, 1>(accumulators_tmp.layout())>{}));
  } else {
    return TiledMma::make_fragment_C(append(
        acc_shape,
        Int<AccumulatorPipelineStageCount>{}));  // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N,ACC_PIPE)
  }
}
} // namespace cutlass::detail
