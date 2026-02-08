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
/*! \file
    \brief This file contains definitions and utility functions for describing problem shapes 
           for 3.x Ptr-Array GEMMs and Grouped GEMMs.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_coord.h"

#include "cute/container/array.hpp"

#if ! defined(__CUDACC_RTC__)
#include <initializer_list>
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_>
struct GroupProblemShape {
  using UnderlyingProblemShape = ProblemShape_;
  int32_t num_groups = 1;
  UnderlyingProblemShape* problem_shapes = nullptr;
  UnderlyingProblemShape const* host_problem_shapes = nullptr;

  CUTLASS_HOST_DEVICE
  int32_t groups() const { return num_groups; }

  CUTLASS_HOST_DEVICE
  UnderlyingProblemShape const
  get_problem_shape(int32_t group_idx) const {
    return problem_shapes[group_idx];
  }

  CUTLASS_HOST_DEVICE
  UnderlyingProblemShape const
  get_host_problem_shape(int32_t group_idx) const {
    return host_problem_shapes != nullptr ? host_problem_shapes[group_idx] : UnderlyingProblemShape{};
  }

  CUTLASS_HOST_DEVICE
  bool
  is_host_problem_shape_available() const {
    return host_problem_shapes != nullptr;
  }
};

template <class ProblemShape_, class MaxProblemShape_>
struct MoEProblemShape {
  using UnderlyingProblemShape = ProblemShape_;
  using MaxProblemShape = MaxProblemShape_;

  UnderlyingProblemShape problem_shape;
  MaxProblemShape max_problem_shape;
};


template <class ProblemShape_>
class ArrayProblemShape {
public:
  using UnderlyingProblemShape = ProblemShape_;

  ArrayProblemShape() = default;
  ArrayProblemShape(UnderlyingProblemShape ps) : problem_shape_(ps) {}

  // Num of groups for Ptr-Array GEMM always remain one, just the number of batches (l) can vary
  // This is just to maintain uniformity with GroupProblemShape
  constexpr int32_t groups() const { return 1; }

  UnderlyingProblemShape* problem_shapes() const {
    return &problem_shape_;
  }
  UnderlyingProblemShape const* host_problem_shapes() const {
    return &problem_shape_;
  }

  // This is just to maintain uniformity with GroupProblemShape
  CUTLASS_HOST_DEVICE
  UnderlyingProblemShape const
  get_problem_shape(int32_t /* unused */ = 0) const {
    return problem_shape_;
  }

  CUTLASS_HOST_DEVICE
  UnderlyingProblemShape const
  get_host_problem_shape(int32_t /* unused */ = 0) const {
    return problem_shape_;
  }

  CUTLASS_HOST_DEVICE
  bool
  is_host_problem_shape_available() const {
    return true;
  }
private:
  UnderlyingProblemShape problem_shape_{};
};


namespace detail {
  
template<class T>
struct is_moe_problem_shape : cute::false_type {};
template<class T, class U>
struct is_moe_problem_shape<cutlass::gemm::MoEProblemShape<T,U>> : cute::true_type {}; 

}

} // namespace cutlass::gemm 
