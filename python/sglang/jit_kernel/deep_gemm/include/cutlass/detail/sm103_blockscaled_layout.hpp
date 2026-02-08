/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Blocked Scale configs specific for SM103 BlockScaled MMA
*/

#pragma once

#include "cutlass/layout/matrix.h"

#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace cutlass::detail{

/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute;

template <int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct Sm103BlockScaledBasicChunk {

  using Blk_MN    = _128;
  using Blk_SF    =   _4; 

  using SfKMajorAtom  =  Layout< Shape< Shape< _8, _4, _4>,  Shape<Int<SFVecSize>, _4>>, 
                               Stride<Stride<_16,_128, _4>, Stride<            _0, _1>>>;
  using SfMNMajorAtom = Layout< Shape< Shape<Int<SFVecSize>, _4>,  Shape<_8,   _4, _4>>, 
                               Stride<Stride<            _0, _1>, Stride<_16,_128, _4>>>;
  using SfAtom    = cute::conditional_t<major == UMMA::Major::K, SfKMajorAtom, SfMNMajorAtom>;
};

template <int SFVecSize_>
struct Sm103BlockScaledConfig {
  // We are creating the SFA and SFB tensors' layouts in the collective since they always have the same layout.
  // k-major order
  static constexpr int SFVecSize = SFVecSize_;
  using Sm103BlkScaledChunk = Sm103BlockScaledBasicChunk<SFVecSize>;
  using Blk_MN = typename Sm103BlkScaledChunk::Blk_MN;
  using Blk_SF = typename Sm103BlkScaledChunk::Blk_SF; 
  using SfAtom = typename Sm103BlkScaledChunk::SfAtom;

  using LayoutSF = decltype(tile_to_shape(SfAtom{}, make_shape(int(0),int(0),int(0)),Step<_2,_1,_3>{}));

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFA() {
    return LayoutSF{};
  }

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFB() {
    return LayoutSF{};
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFA.
  template < class ProblemShape, class LayoutSFA = LayoutSF>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFA(ProblemShape problem_shape, LayoutSFA layout_sfa = LayoutSFA{}) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    return tile_to_shape(SfAtom{}, make_shape(M,K,L), Step<_2,_1,_3>{});
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFB.
  template <class ProblemShape, class LayoutSFB = LayoutSF>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFB(ProblemShape problem_shape, LayoutSFB layout_sfb = LayoutSFB{}) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    return tile_to_shape(SfAtom{}, make_shape(N,K,L), Step<_2,_1,_3>{});
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
