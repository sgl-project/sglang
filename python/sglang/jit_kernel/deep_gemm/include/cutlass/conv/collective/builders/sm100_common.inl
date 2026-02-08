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
//

//
#pragma once

#include "cutlass/layout/tensor.h"
#include "cute/atom/copy_traits_sm100_im2col.hpp"
#include "cutlass/arch/mma.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/conv/collective/builders/sm90_common.inl"
#include "cutlass/gemm/collective/builders/sm100_common.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Collective tile traits struct that serves as a type list containing a tensor's mem layouts and atoms
template<
  class GmemTiledCopy_,
  class SmemLayoutAtom_,
  class TmemLayoutAtom_ = void
>
struct Sm100ImplicitGemmTileTraits {
  using GmemTiledCopy = GmemTiledCopy_;
  using SmemLayoutAtom = SmemLayoutAtom_;
  using TmemLayoutAtom = TmemLayoutAtom_;
};

template <class ClusterShapeMNK, class AtomThrId>
constexpr auto
sm100_cluster_shape_to_im2col_tma_atom_A(ClusterShapeMNK cluster_shape_mnk, AtomThrId atom_thr_id) {
  static_assert(cute::rank(cluster_shape_mnk) == 3);
  constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShapeMNK>;

  if constexpr (cute::size(atom_thr_id) == 2) {
    if constexpr (!IsDynamicCluster) {
      static_assert(cute::size<0>(cluster_shape_mnk) % 2 == 0, "Cluster shape not divisible by MMA size");
      if constexpr (cute::size<1>(cluster_shape_mnk) == 1) {
        return cute::SM100_TMA_2SM_LOAD_IM2COL{};
      }
      else {
        return cute::SM100_TMA_2SM_LOAD_IM2COL_MULTICAST{};
      }
    }
    else {
      return cute::SM100_TMA_2SM_LOAD_IM2COL_MULTICAST{};
    }
  }
  else if constexpr (size(atom_thr_id) == 1) {
    if constexpr (!IsDynamicCluster) {
      return detail::sm90_cluster_shape_to_im2col_tma_atom(cute::size<1>(cluster_shape_mnk));
    }
    else {
      // In the case of dynamic cluster, multicast decision is not known at compile time.
      // A multicast instruction is forced by passing a cute::Int<2>{} to this helper. 
      return detail::sm90_cluster_shape_to_im2col_tma_atom(cute::Int<2>{});
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<ClusterShapeMNK>,
        "Unsupported Configuration for SM100 TMA");
  }
}

template <class ClusterShapeMNK, class AtomThrId>
constexpr auto
sm100_cluster_shape_to_im2col_tma_atom_B(ClusterShapeMNK cluster_shape_mnk, AtomThrId atom_thr_id) {
  static_assert(cute::rank(cluster_shape_mnk) == 3);
  constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShapeMNK>;

  if constexpr (cute::size(atom_thr_id) == 2) {
    if constexpr (!IsDynamicCluster) {
      static_assert(cute::size<0>(cluster_shape_mnk) % 2 == 0, "Cluster shape not divisible by MMA size");
      if constexpr (cute::size<0>(cluster_shape_mnk) == 2) {
        return cute::SM100_TMA_2SM_LOAD_IM2COL{};
      }
      else {
        return cute::SM100_TMA_2SM_LOAD_IM2COL_MULTICAST{};
      }
    }
    else {
      return cute::SM100_TMA_2SM_LOAD_IM2COL_MULTICAST{};
    }
  } else if constexpr (size(atom_thr_id) == 1) {
    if constexpr (!IsDynamicCluster) {
      return detail::sm90_cluster_shape_to_im2col_tma_atom(cute::size<0>(cluster_shape_mnk));
    }
    else {
      // In the case of dynamic cluster, multicast decision is not known at compile time.
      // A multicast instruction is forced by passing a cute::Int<2>{} to this helper. 
      return detail::sm90_cluster_shape_to_im2col_tma_atom(cute::Int<2>{});
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<ClusterShapeMNK>,
        "Unsupported Configuration for SM100 TMA");
  }
}

template<
  class ElementA,
  class ElementB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  class KernelScheduleType
>
constexpr auto
sm100_make_tiled_mma() {
  // MMA_2SM requested
  if constexpr (cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecialized2SmSm100>) {
    return cutlass::gemm::collective::detail::sm100_make_2sm_trivial_tiled_mma<
                ElementA, ElementB, ElementAccumulator,
                TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB>();
  }
  // MMA_1SM requested
  else if constexpr (cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecialized1SmSm100>) {
    return cutlass::gemm::collective::detail::sm100_make_1sm_trivial_tiled_mma<
                ElementA, ElementB, ElementAccumulator,
                TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB>();
  }
  // Auto scheduling requested
  else if constexpr (cute::is_same_v<KernelScheduleType, KernelScheduleAuto>) {
    // Static cluster
    if constexpr (cute::is_static_v<ClusterShape_MNK>) {
      // For MMA_2SM we need a cluster shape that is multiple of 2x1
      // and only M=128 and M=256 are supported, otherwise, fall back to MMA_1SM
      if constexpr (cute::size<0>(ClusterShape_MNK{}) % 2 == 0 &&
                    cute::size<0>(TileShape_MNK{}) % 128 == 0) {
        return cutlass::gemm::collective::detail::sm100_make_2sm_trivial_tiled_mma<
                  ElementA, ElementB, ElementAccumulator,
                  TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB>();
      }
      else {
        return cutlass::gemm::collective::detail::sm100_make_1sm_trivial_tiled_mma<
                  ElementA, ElementB, ElementAccumulator,
                  TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB>();
      }
    // Dynamic cluster shape means we cannot assume we can use 2SM MMA 
    }
    else {
        return cutlass::gemm::collective::detail::sm100_make_1sm_trivial_tiled_mma<
                  ElementA, ElementB, ElementAccumulator,
                  TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB>();
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementA>,
        "Unsupported policy for SM100 collective builder.");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective::detail

/////////////////////////////////////////////////////////////////////////////////////////////////
