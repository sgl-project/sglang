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
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(type_traits)
#else
#include <type_traits>
#endif

#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace cute {

//
// A generic tiling of thread-value layouts
//

template <class Layout_TV_,    // (tid,vid) -> coord   [Need not be 2D...]
          class Tiler_MN_>     // coord space
struct TV_Tiler
{
  using Tiler_MN       = Tiler_MN_;
  using TiledLayout_TV = Layout_TV_;

  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   ((ThrV,FrgV),(RestM,RestN,...))
  // where
  //   ThrV:  The threads local to a tile.
  //   FrgV:  The values local to a tile.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class Tensor>
  CUTE_HOST_DEVICE constexpr static
  auto
  apply(Tensor&& tensor)
  {
    // If Layout_TV and Tiler_MN were composable in general, then this won't be needed!

    // ((thr_id,val_id),(RestM,RestN,...))
    return zipped_divide(tensor, Tiler_MN{}).compose(TiledLayout_TV{}, _);
  }

  template <class SliceCoord>
  struct TV_Partitioner
  {
    SliceCoord coord_;

    template <class TargetTensor>
    CUTE_HOST_DEVICE
    auto
    partition(TargetTensor&& target) {
      Tensor thr_tensor = make_tensor(static_cast<TargetTensor&&>(target).data(), apply(target.layout()));
      return thr_tensor(coord_, repeat<rank_v<TargetTensor>>(_));
    }
  };

  template <class SliceCoord>
  CUTE_HOST_DEVICE static
  auto
  get_slice(SliceCoord const& coord)
  {
    return TV_Partitioner<SliceCoord>{coord};
  }
};

template <class Layout_TV,
          class Tiler_MN>
CUTE_HOST_DEVICE
auto
make_tiler_impl(Layout_TV const&,
                Tiler_MN  const&)
{
  return TV_Tiler<Layout_TV, Tiler_MN>{};
}

}
