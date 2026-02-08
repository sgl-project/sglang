/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
\brief Defies functors for mapping blockIdx to partitions of the batched reduction computation.
*/
#pragma once
#include "cutlass/coord.h"

namespace cutlass {
namespace reduction {
struct DefaultBlockSwizzle {
  /// Ctor
  CUTLASS_HOST_DEVICE DefaultBlockSwizzle() {}

  /// Swizzle the block index.
  CUTLASS_DEVICE dim3 swizzle() { return blockIdx; }

  /// 
  CUTLASS_HOST_DEVICE dim3 get_grid_layout(Coord<3> const &problem_size,
                                           Coord<3> const &OutputTile) {
    assert(OutputTile[0] == 1 && OutputTile[1] == 1);
    assert((problem_size[0] * problem_size[1] * problem_size[2]) % OutputTile[2] == 0);
    dim3 grid;
    grid.x = problem_size[0] * problem_size[1] * problem_size[2]
      / OutputTile[2] ;
    return grid;
  }

  ///
  CUTLASS_DEVICE Coord<3> get_threadblock_offset(Coord<3> const &SubTile) {
    assert(SubTile[0] == 1 && SubTile[1] == 1);
    dim3 block = swizzle();
    Coord<3> threadblock_offset =
      make_Coord(0, 0, block.x * SubTile[2]);
    return threadblock_offset;
  }
};
} // namespace reduction
} // namespace cutlass
