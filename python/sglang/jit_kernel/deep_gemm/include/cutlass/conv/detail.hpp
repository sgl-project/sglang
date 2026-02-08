
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

#include "cutlass/conv/convnd_problem_shape.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

  // Helper function to get the problem shape
template <typename T, class ProblemShape>
auto get_problem_shape_MNKL_helper(ProblemShape const& problem_shape, cute::true_type) {
  return T::get_problem_shape_MNKL(problem_shape);
}

template <typename T, class ProblemShape>
ProblemShape get_problem_shape_MNKL_helper(ProblemShape const& problem_shape, cute::false_type) {
  return problem_shape;
}

// Get problem shape MNKL according to following table:
// |               |   Fprop   |   Dgrad         |   Wgrad   |
// |   ----        | --------- | --------        | --------  |
// |   Shape_M     | (Q,P,Z,N) | (W/V,H/U,D/O,N) | (K)       |
// |   Shape_N     | (K)       | (C)             | (C,S,R,T) |
// |   Shape_K     | (C,S,R,T) | (K,S,R,T)       | (Q,P,Z,N) |
// |   Shape_L     | _1        | (V,U,O)         | _1        |

template <class ProblemShape>
CUTLASS_HOST_DEVICE
constexpr auto
get_transformed_problem_shape_MNKL(ProblemShape const& problem_shape) {
  return problem_shape;
}


template <conv::Operator ConvOp, int SpatialDim>
CUTLASS_HOST_DEVICE
constexpr auto
get_transformed_problem_shape_MNKL(ConvProblemShape<ConvOp, SpatialDim> const& problem_shape) {
  using cute::insert;
  using cute::make_shape;
  using cute::reverse;
  using cute::take;

  constexpr int RankT = SpatialDim + 2;

  if constexpr (ConvOp == conv::Operator::kWgrad) {
    auto M_xformed = problem_shape.shape_C[0];
    auto N_xformed = reverse(take<1, RankT>(problem_shape.shape_C));
    auto K_xformed = reverse(take<0, RankT - 1>(problem_shape.shape_A));
    auto L_xformed = cute::Int<1>{};

    return make_shape(M_xformed, N_xformed, K_xformed, L_xformed);
  }
  else if constexpr (ConvOp == conv::Operator::kFprop){
    auto M_xformed = reverse(take<0, RankT - 1>(problem_shape.shape_C));
    auto N_xformed = problem_shape.shape_C[RankT - 1];
    auto K_xformed = reverse(take<1, RankT>(problem_shape.shape_B));
    auto L_xformed = cute::Int<1>{};

    return make_shape(M_xformed, N_xformed, K_xformed, L_xformed);
  }
  else if constexpr (ConvOp == conv::Operator::kDgrad) {
    auto L_xformed = reverse(problem_shape.traversal_stride); // (V,U,O)
    auto M_xformed = ceil_div(reverse(take<0,RankT - 1>(problem_shape.shape_C)), L_xformed);
    auto N_xformed = problem_shape.shape_C[RankT - 1];
    // shape_B: [K,T,R,S,C], K_xformed: [K,S,R,T]
    auto K_xformed = insert<0>(
                (reverse(take<1,RankT - 1>(problem_shape.shape_B))),
                problem_shape.shape_B[0]);

    return make_shape(M_xformed, N_xformed, K_xformed, L_xformed);
  }
}

// Assuming im2col linearization
// Get problem shape MNKL according to following table:
// |               |   Fprop   |   Dgrad               |   Wgrad   |
// |   ----        | --------- | --------              | --------  |
// |   Shape_M     | (Q*P*Z*N) | ([W/V]*[H/U]*[D/O]*N) | (K)       |
// |   Shape_N     | (K)       | (C)                   | (C,S,R,T) |
// |   Shape_K     | (C,S,R,T) | (K,S,R,T)             | (Q*P*Z*N) |
// |   Shape_L     | _1        | (V*U*O)               | _1        |
template <conv::Operator ConvOp, int SpatialDim>
CUTLASS_HOST_DEVICE
constexpr auto
get_linearized_problem_shape_MNKL(ConvProblemShape<ConvOp, SpatialDim> const& problem_shape) {

  auto [M, N, K, L] = get_transformed_problem_shape_MNKL(problem_shape);

  if constexpr (ConvOp == conv::Operator::kFprop || ConvOp == conv::Operator::kDgrad) {
    return cute::make_shape(cute::product(M), N, K, cute::product(L));
  }
  else if constexpr (ConvOp == conv::Operator::kWgrad) {
    return cute::make_shape(M, N, cute::product(K), L);
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::detail

////////////////////////////////////////////////////////////////////////////////////////////////////
