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



#include "cute/container/tuple.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/trace.h"
#include "cute/layout.hpp" // cute::make_shape
#include "cutlass/trace.h" // CUTLASS_TRACE_HOST

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::detail {

// Returns either ClusterShape, if it is static, or a Shape<int,int,Int<1>> populated with the
// x and y dimensions of `dynamic_cluster_shape`.
template <class ClusterShape>
CUTLASS_HOST_DEVICE
static auto
select_cluster_shape(ClusterShape cluster_shape, dim3 dynamic_cluster_shape) {
  return cute::conditional_return<not cute::is_static_v<ClusterShape>>(
    make_shape(static_cast<int>(dynamic_cluster_shape.x), static_cast<int>(dynamic_cluster_shape.y), cute::Int<1>{}),
    cluster_shape);
}

template <class ClusterShape>
CUTLASS_DEVICE
static auto
select_cluster_shape(ClusterShape cluster_shape) {
  if constexpr (cute::is_static_v<ClusterShape>) {
    return cluster_shape;
  }
  else {
    dim3 dynamic_cluster_shape = cute::cluster_shape();
    return make_shape(static_cast<int>(dynamic_cluster_shape.x), static_cast<int>(dynamic_cluster_shape.y), cute::Int<1>{});
  }
}

// Dynamic cluster shape can_implement rule
template <class AtomThrShapeMNK>
CUTLASS_HOST_DEVICE
bool
preferred_cluster_can_implement(dim3 cluster_shape, dim3 cluster_shape_fallback) {
  bool implementable{true};

  // Runtime cluster shape should satisfy MMA requirements
  auto AtomThrShapeM = cute::size<0>(AtomThrShapeMNK{});
  implementable &= (cluster_shape.x > 0 && cluster_shape.y > 0 && cluster_shape.z > 0);
  implementable &= (cluster_shape.x % AtomThrShapeM == 0);

  implementable &= (cluster_shape_fallback.x > 0 && cluster_shape_fallback.y > 0 && cluster_shape_fallback.z > 0);
  implementable &= (cluster_shape_fallback.x % AtomThrShapeM == 0);

  // Only support pow2 runtime cluster shape for now
  implementable &= ispow2(cluster_shape.x) &&
                   ispow2(cluster_shape.y) &&
                   ispow2(cluster_shape.z);

  implementable &= ispow2(cluster_shape_fallback.x) &&
                   ispow2(cluster_shape_fallback.y) &&
                   ispow2(cluster_shape_fallback.z);

  return implementable;
}

} // namespace cutlass::detail

/////////////////////////////////////////////////////////////////////////////////////////////////
