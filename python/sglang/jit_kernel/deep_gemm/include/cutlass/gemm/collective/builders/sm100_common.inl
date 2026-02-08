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

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/gemm/dispatch_policy.hpp" // KernelSchedule1Sm, KernelSchedule2Sm
#include "cutlass/gemm/collective/builders/sm90_common.inl" // detail::sm90_cluster_shape_to_tma_atom()
#include "cutlass/numeric_types.h" // all numeric types
#include "cutlass/detail/dependent_false.hpp" // detail::dependent_false
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/layout.hpp" // cutlass::detail::get_input_alignment_bits()
#include "cutlass/layout/matrix.h" // cutlass::layout::RowMajor, cutlass::layout::ColumnMajor
#include "cutlass/fast_math.h" // cutlass::round_up, cutlass::const_max
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/collective/builders/sm1xx_common.inl"

#include "cute/atom/mma_traits_sm100.hpp" // UMMA::Layout_MN_SW*
#include "cute/atom/copy_traits_sm100_tma.hpp" // SM100_TMA_*SM_LOAD_*
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/arch/mma_sm100_desc.hpp" // cute::UMMA::Major
#include "cute/arch/mma_sm100_umma.hpp" // SM100_*MMA_SS_*
#include "cute/numeric/integral_constant.hpp" // is_static_v, cute::integral_constant
#include "cute/util/type_traits.hpp" // cute::alignment_of_v

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

// Forward Declaration
struct KernelScheduleAuto;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

//
// Some named constants
//
constexpr int sm100_smem_capacity_bytes = cutlass::arch::sm100_smem_capacity_bytes;
constexpr int CLCResponseSize =
    sizeof(typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<Shape<_1,_1,_1>,1>::CLCResponse{});


// Helper for SS UMMA smem selection that considers a tensor TileShape:
//   (BLK_MN, BLK_K)
//   or hierarchically
//   ((BLK_MN0,BLK_MN1,...),(BLK_K0,BLK_K1,...))
//   and returns the largest UMMA::Layout that fits BLK_MN0 and BLK_K0
template <cute::UMMA::Major major, class ElementType, class BLK_MN, class BLK_K>
CUTE_HOST_DEVICE constexpr
auto
sm100_smem_selector() {
  auto BLK_MN0 = size<0>(BLK_MN{});
  auto BLK_K0  = size<0>(BLK_K{});

  static_assert(BLK_MN0 % 8 == 0, "BLK_MN0 must be a multiple of 8.");
  static_assert(BLK_K0 % 8 == 0,  "BLK_K0 must be a multiple of 8.");

  if constexpr (major == cute::UMMA::Major::MN) {
    // Handle the special case for F32 NT kernels
    if constexpr ((sizeof(ElementType) == 4)) {
       static_assert(BLK_MN0 % size<0>(UMMA::Layout_MN_SW128_32B_Atom<ElementType>{}) == 0, "for mn-major tf32 operands, SW128_32B is the only available smem layout");
      return UMMA::Layout_MN_SW128_32B_Atom<ElementType>{};
    }
    else {
      // All other data types are handled as SM90
      if constexpr      (BLK_MN0 % size<0>(UMMA::Layout_MN_SW128_Atom<ElementType>{}) == 0) {
        return UMMA::Layout_MN_SW128_Atom<ElementType>{};
      }
      else if constexpr (BLK_MN0 % size<0>(UMMA::Layout_MN_SW64_Atom<ElementType>{}) == 0) {
        return UMMA::Layout_MN_SW64_Atom<ElementType>{};
      }
      else if constexpr (BLK_MN0 % size<0>(UMMA::Layout_MN_SW32_Atom<ElementType>{}) == 0) {
        return UMMA::Layout_MN_SW32_Atom<ElementType>{};
      }
      else if constexpr (BLK_MN0 % size<0>(UMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0) {
        return UMMA::Layout_MN_INTER_Atom<ElementType>{};
      }
      else {
        static_assert(BLK_MN0 % size<0>(UMMA::Layout_MN_INTER_Atom<ElementType>{}) == 0,
                      "BLK_MN0 must be a multiple of size<0>(UMMA::Layout_MN_INTER_Atom<ElementType>{})");
      }
    }
  }
  else if constexpr (major == cute::UMMA::Major::K) {
    if constexpr      (BLK_K0 % size<1>(UMMA::Layout_K_SW128_Atom<ElementType>{}) == 0) {
      return UMMA::Layout_K_SW128_Atom<ElementType>{};
    }
    else if constexpr (BLK_K0 % size<1>(UMMA::Layout_K_SW64_Atom<ElementType>{}) == 0) {
      return UMMA::Layout_K_SW64_Atom<ElementType>{};
    }
    else if constexpr (BLK_K0 % size<1>(UMMA::Layout_K_SW32_Atom<ElementType>{}) == 0) {
      return UMMA::Layout_K_SW32_Atom<ElementType>{};
    }
    else if constexpr (BLK_K0 % size<1>(UMMA::Layout_K_INTER_Atom<ElementType>{}) == 0) {
      return UMMA::Layout_K_INTER_Atom<ElementType>{};
    }
    else {
      static_assert(BLK_K0 % size<1>(UMMA::Layout_K_INTER_Atom<ElementType>{}) == 0,
                    "BLK_K0 must be a multiple of size<1>(UMMA::Layout_K_INTER_Atom<ElementType>{})");
    }
  }
}

// Helper for SS UMMA smem selection that considers a tensor TileShape:
//   (BLK_MN, BLK_K)
//   or hierarchically
//   ((BLK_MN0,BLK_MN1,...),(BLK_K0,BLK_K1,...))
//   and returns the largest UMMA::Layout that fits BLK_MN0 and BLK_K0
template <cute::UMMA::Major major, class ElementType, class BLK_MN, class BLK_K, class Sparsity>
CUTE_HOST_DEVICE constexpr
auto
sm100_smem_selector_sparse()
{
  auto BLK_MN0 = size<0>(BLK_MN{});
  auto BLK_K0  = size<0>(BLK_K{});

  static_assert(BLK_MN0 % 8 == 0, "BLK_MN0 must be a multiple of 8.");
  static_assert(BLK_K0 % 8 == 0,  "BLK_K0 must be a multiple of 8.");

  if constexpr (major == cute::UMMA::Major::MN) {

    // Handle the special case for F32 NT kernels
    if constexpr ((sizeof(ElementType) == 4 && (BLK_MN0 % size<0>(UMMA::Layout_MN_SW128_32B_SpAtom<ElementType, Sparsity{}>{}) == 0))) {
      return UMMA::Layout_MN_SW128_32B_SpAtom<ElementType, Sparsity{}>{};
    }
    else {
      // All other data types are handled as SM90
      if constexpr      (BLK_MN0 % size<0>(UMMA::Layout_MN_SW128_SpAtom<ElementType, Sparsity{}>{}) == 0) {
        return UMMA::Layout_MN_SW128_SpAtom<ElementType, Sparsity{}>{};
      }
      else if constexpr (BLK_MN0 % size<0>(UMMA::Layout_MN_SW64_SpAtom<ElementType, Sparsity{}>{}) == 0) {
        return UMMA::Layout_MN_SW64_SpAtom<ElementType, Sparsity{}>{};
      }
      else if constexpr (BLK_MN0 % size<0>(UMMA::Layout_MN_SW32_SpAtom<ElementType, Sparsity{}>{}) == 0) {
        return UMMA::Layout_MN_SW32_SpAtom<ElementType, Sparsity{}>{};
      }
      else if constexpr (BLK_MN0 % size<0>(UMMA::Layout_MN_INTER_SpAtom<ElementType, Sparsity{}>{}) == 0) {
        return UMMA::Layout_MN_INTER_SpAtom<ElementType, Sparsity{}>{};
      }
      else {
        static_assert(BLK_MN0 % size<0>(UMMA::Layout_MN_INTER_SpAtom<ElementType, Sparsity{}>{}) == 0,
                      "BLK_MN0 must be a multiple of size<0>(UMMA::Layout_MN_INTER_SpAtom<ElementType, Sparsity{}>{})");
      }
    }
  }
  else if constexpr (major == cute::UMMA::Major::K) {
    if constexpr      (BLK_K0 % size<1>(UMMA::Layout_K_SW128_SpAtom<ElementType, Sparsity{}>{}) == 0) {
      return UMMA::Layout_K_SW128_SpAtom<ElementType, Sparsity{}>{};
    }
    else if constexpr (BLK_K0 % size<1>(UMMA::Layout_K_SW64_SpAtom<ElementType, Sparsity{}>{}) == 0) {
      return UMMA::Layout_K_SW64_SpAtom<ElementType, Sparsity{}>{};
    }
    else if constexpr (BLK_K0 % size<1>(UMMA::Layout_K_SW32_SpAtom<ElementType, Sparsity{}>{}) == 0) {
      return UMMA::Layout_K_SW32_SpAtom<ElementType, Sparsity{}>{};
    }
    else if constexpr (BLK_K0 % size<1>(UMMA::Layout_K_INTER_SpAtom<ElementType, Sparsity{}>{}) == 0) {
      return UMMA::Layout_K_INTER_SpAtom<ElementType, Sparsity{}>{};
    }
    else {
      static_assert(BLK_K0 % size<1>(UMMA::Layout_K_INTER_SpAtom<ElementType, Sparsity{}>{}) == 0,
                    "BLK_K0 must be a multiple of size<1>(UMMA::Layout_K_INTER_SpAtom<ElementType, Sparsity{}>{})");
    }
  }
}

template <class ClusterShapeMNK, class AtomThrId>
constexpr auto
sm100_cluster_shape_to_tma_atom_A(ClusterShapeMNK cluster_shape_mnk, AtomThrId atom_thr_id) {
  static_assert(cute::rank(cluster_shape_mnk) == 3);
  constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShapeMNK>;

  if constexpr (cute::size(atom_thr_id) == 2) {
    if constexpr (!IsDynamicCluster) {
      static_assert(cute::size<0>(cluster_shape_mnk) % 2 == 0, "Cluster shape not divisible by MMA size");
      if constexpr (cute::size<1>(cluster_shape_mnk) == 1) {
        return cute::SM100_TMA_2SM_LOAD{};
      }
      else {
        return cute::SM100_TMA_2SM_LOAD_MULTICAST{};
      }
    }
    else {
      return cute::SM100_TMA_2SM_LOAD_MULTICAST{};
    }
  }
  else if constexpr (size(atom_thr_id) == 1) {
    if constexpr (!IsDynamicCluster) {
      return detail::sm90_cluster_shape_to_tma_atom(cute::size<1>(cluster_shape_mnk));
    }
    else {
      // In the case of dynamic cluster, multicast decision is not known at compile time.
      // A multicast instruction is forced by passing a cute::Int<2>{} to this helper. 
      return detail::sm90_cluster_shape_to_tma_atom(cute::Int<2>{});
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<ClusterShapeMNK>,
        "Unsupported Configuration for SM100 TMA");
  }
}

template <class ClusterShapeMNK, class AtomThrId>
constexpr auto
sm100_cluster_shape_to_tma_atom_B(ClusterShapeMNK cluster_shape_mnk, AtomThrId atom_thr_id) {
  static_assert(cute::rank(cluster_shape_mnk) == 3);
  constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShapeMNK>;

  if constexpr (cute::size(atom_thr_id) == 2) {
    if constexpr (!IsDynamicCluster) {
      static_assert(cute::size<0>(cluster_shape_mnk) % 2 == 0, "Cluster shape not divisible by MMA size");
      if constexpr (cute::size<0>(cluster_shape_mnk) == 2) {
        return cute::SM100_TMA_2SM_LOAD{};
      }
      else {
        return cute::SM100_TMA_2SM_LOAD_MULTICAST{};
      }
    }
    else {
      return cute::SM100_TMA_2SM_LOAD_MULTICAST{};
    }
  } else if constexpr (size(atom_thr_id) == 1) {
    if constexpr (!IsDynamicCluster) {
      return detail::sm90_cluster_shape_to_tma_atom(cute::size<0>(cluster_shape_mnk));
    }
    else {
      // In the case of dynamic cluster, multicast decision is not known at compile time.
      // A multicast instruction is forced by passing a cute::Int<2>{} to this helper. 
      return detail::sm90_cluster_shape_to_tma_atom(cute::Int<2>{});
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<ClusterShapeMNK>,
        "Unsupported Configuration for SM100 TMA");
  }
}


template <class ClusterShapeMNK, class AtomThrId>
constexpr auto
sm100_cluster_shape_to_tma_atom_SFB(ClusterShapeMNK cluster_shape_mnk, AtomThrId atom_thr_id) {
  static_assert(cute::rank(cluster_shape_mnk) == 3);
  if constexpr (cute::size(atom_thr_id) == 2) {
    // Always could use multicast feature for SFB with 2cta MMA.
    return cute::SM100_TMA_2SM_LOAD_MULTICAST{};
  }
  else if constexpr (size(atom_thr_id) == 1) {
    constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShapeMNK>;
    if constexpr (!IsDynamicCluster) {
      return detail::sm90_cluster_shape_to_tma_atom(cute::size<0>(cluster_shape_mnk));
    }
    else {
      // In the case of dynamic cluster, multicast decision is not known at compile time.
      // A multicast instruction is forced by passing a cute::Int<2>{} to this helper. 
      return detail::sm90_cluster_shape_to_tma_atom(cute::Int<2>{});
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<ClusterShapeMNK>,
        "Unsupported Configuration for SM100 TMA");
  }
}

template<
  class ElementAMma,
  class ElementBMma,
  class ElementAMmaccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  UMMA::ScaleIn ANeg = UMMA::ScaleIn::One,
  UMMA::ScaleIn BNeg = UMMA::ScaleIn::One
>
constexpr auto
sm100_make_1sm_trivial_tiled_mma() {

  constexpr int M = cute::size<0>(TileShape_MNK{});
  static_assert(M == 64 || M == 128, "Invalid TileShape_M.");

  // Do not allow a tiled MMA N mode > 1, as that is not reasonable.
  constexpr int N = cute::size<1>(TileShape_MNK{});
  static_assert(N % 8 == 0 && N <= 256, "Invalid TileShape_N.");

  if constexpr     (cute::is_same_v<ElementAMma, cutlass::tfloat32_t>) {
    static_assert(cute::is_same_v<ElementAMma, ElementBMma>, "ElementAMma and ElementBMma must match.");
    return make_tiled_mma(cute::SM100_MMA_TF32_SS<ElementAMma, ElementBMma, ElementAMmaccumulator,
                                              M, N, UmmaMajorA, UmmaMajorB, ANeg, BNeg>{});
  }
  else if constexpr (cute::is_same_v<ElementAMma,     cutlass::half_t> ||
                     cute::is_same_v<ElementAMma, cutlass::bfloat16_t>) {
    static_assert(cute::is_same_v<ElementAMma, ElementBMma>, "ElementAMma and ElementBMma must match.");
    return make_tiled_mma(cute::SM100_MMA_F16BF16_SS<ElementAMma, ElementBMma, ElementAMmaccumulator,
                                              M, N, UmmaMajorA, UmmaMajorB, ANeg, BNeg>{});
  }
  else if constexpr (cute::is_same_v<ElementAMma,  int8_t> ||
                     cute::is_same_v<ElementAMma, uint8_t>) {
    return make_tiled_mma(cute::SM100_MMA_S8_SS<ElementAMma, ElementBMma, ElementAMmaccumulator,
                                              M, N, UmmaMajorA, UmmaMajorB>{});
  }
  else if constexpr (cute::is_same_v<ElementAMma, cutlass::type_erased_dynamic_float8_t> 
                    || cute::is_same_v<ElementAMma, cutlass::detail::type_erased_dynamic_float6_unpacksmem_t> 
                    || cute::is_same_v<ElementAMma, cutlass::detail::type_erased_dynamic_float4_unpacksmem_t> 
                    || cute::is_same_v<ElementAMma, cutlass::float_e4m3_t>
                    || cute::is_same_v<ElementAMma, cutlass::float_e5m2_t>
                    || cute::is_same_v<ElementAMma, cutlass::detail::float_e2m3_unpacksmem_t>
                    || cute::is_same_v<ElementAMma, cutlass::detail::float_e3m2_unpacksmem_t>
                    || cute::is_same_v<ElementAMma, cutlass::detail::float_e2m1_unpacksmem_t>
                    ) {
  
      return make_tiled_mma(
        cute::MMA_Traits<
          cute::SM100_MMA_F8F6F4_SS,
          ElementAMma,
          ElementBMma, 
          ElementAMmaccumulator, 
          cute::C<M>, 
          cute::C<N>, 
          cute::integral_constant<UMMA::Major, UmmaMajorA>,
          cute::integral_constant<UMMA::Major, UmmaMajorB>,
          cute::integral_constant<UMMA::ScaleIn, ANeg>,
          cute::integral_constant<UMMA::ScaleIn, BNeg>
        >{}
      );
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAMma>,
        "Unsupported configuration for SM100 collective builder.");
  }
}

template<
  class ElementAMma,
  class ElementBMma,
  class ElementAMmaccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  UMMA::ScaleIn ANeg = UMMA::ScaleIn::One,
  UMMA::ScaleIn BNeg = UMMA::ScaleIn::One
>
constexpr auto
sm100_make_2sm_trivial_tiled_mma() {

  constexpr int M = cute::size<0>(TileShape_MNK{});
  static_assert(M == 128 || M == 256, "Invalid TileShape_M.");

  // Do not allow a tiled MMA N mode > 1, as that is not reasonable.
  constexpr int N = cute::size<1>(TileShape_MNK{});
  static_assert(N % 8 == 0 && N <= 256, "Invalid TileShape_N.");

  if constexpr     (cute::is_same_v<ElementAMma, cutlass::tfloat32_t>) {
    static_assert(cute::is_same_v<ElementAMma, ElementBMma>, "ElementAMma and ElementBMma must match.");
    return make_tiled_mma(cute::SM100_MMA_TF32_2x1SM_SS<ElementAMma, ElementBMma, ElementAMmaccumulator,
                                                     M, N, UmmaMajorA, UmmaMajorB, ANeg, BNeg>{});
  }
  else if constexpr (cute::is_same_v<ElementAMma,     cutlass::half_t> ||
                     cute::is_same_v<ElementAMma, cutlass::bfloat16_t>) {
    static_assert(cute::is_same_v<ElementAMma, ElementBMma>, "ElementAMma and ElementBMma must match.");
    return make_tiled_mma(cute::SM100_MMA_F16BF16_2x1SM_SS<ElementAMma, ElementBMma, ElementAMmaccumulator,
                                                    M, N, UmmaMajorA, UmmaMajorB, ANeg, BNeg>{});
  }
  else if constexpr (cute::is_same_v<ElementAMma, int8_t> ||
                     cute::is_same_v<ElementAMma, uint8_t>) {
    return make_tiled_mma(cute::SM100_MMA_S8_2x1SM_SS<ElementAMma, ElementBMma, ElementAMmaccumulator,
                                                    M, N, UmmaMajorA, UmmaMajorB>{});
  }
  else if constexpr (cute::is_same_v<ElementAMma, cutlass::type_erased_dynamic_float8_t> 
                    || cute::is_same_v<ElementAMma, cutlass::detail::type_erased_dynamic_float6_unpacksmem_t> 
                    || cute::is_same_v<ElementAMma, cutlass::detail::type_erased_dynamic_float4_unpacksmem_t> 
                    || cute::is_same_v<ElementAMma, cutlass::float_e4m3_t>
                    || cute::is_same_v<ElementAMma, cutlass::float_e5m2_t>
                    || cute::is_same_v<ElementAMma, cutlass::detail::float_e2m3_unpacksmem_t>
                    || cute::is_same_v<ElementAMma, cutlass::detail::float_e3m2_unpacksmem_t>
                    || cute::is_same_v<ElementAMma, cutlass::detail::float_e2m1_unpacksmem_t>
                    ) {

    return make_tiled_mma(
      cute::MMA_Traits<
        cute::SM100_MMA_F8F6F4_2x1SM_SS, 
        ElementAMma,
        ElementBMma,
        ElementAMmaccumulator, 
        cute::C<M>, 
        cute::C<N>, 
        cute::integral_constant<UMMA::Major, UmmaMajorA>,
        cute::integral_constant<UMMA::Major, UmmaMajorB>,
        cute::integral_constant<UMMA::ScaleIn, ANeg>,
        cute::integral_constant<UMMA::ScaleIn, BNeg>
      >{}
    );

  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAMma>,
        "Unsupported configuration for SM100 collective builder.");
  }
}

// For new MMA construction and partitioning that supports both dynamic and static cluster shape.
// Used in conjunction with make_tma_atom_(A|B)_sm100
// TileShape_MNK is always static and has shape (MmaAtomShapeM, MmaAtomShapeN, TileK)
// ClusterShape_MNK can be dynamic or static.
template<
  class ElementAMma,
  class ElementBMma,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  class BuilderScheduleTag,
  UMMA::ScaleIn ANeg = UMMA::ScaleIn::One,
  UMMA::ScaleIn BNeg = UMMA::ScaleIn::One
>
constexpr auto
sm100_make_trivial_tiled_mma() {
  // MMA_2SM requested
  if constexpr (cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> ) {
    return sm100_make_2sm_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                    TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ANeg, BNeg>();
  }
  // MMA_1SM requested
  else if constexpr (cute::is_base_of_v<KernelSchedule1Sm, BuilderScheduleTag> ) {
    return sm100_make_1sm_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                    TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ANeg, BNeg>();
  }
  // Auto scheduling requested
  else if constexpr (cute::is_same_v<BuilderScheduleTag, KernelScheduleAuto>) {
    // Static cluster
    if constexpr (cute::is_static_v<ClusterShape_MNK>) {
      // For MMA_2SM we need a cluster shape that is multiple of 2x1
      // and only M=128 and M=256 are supported, otherwise, fall back to MMA_1SM
      if constexpr (cute::size<0>(ClusterShape_MNK{}) % 2 == 0 &&
                    cute::size<0>(TileShape_MNK{}) % 128 == 0) {
        return sm100_make_2sm_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                        TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ANeg, BNeg>();
      }
      else {
        return sm100_make_1sm_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                        TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ANeg, BNeg>();
      }
    // Dynamic cluster shape means we cannot assume we can use 2SM MMA 
    }
    else {
        return sm100_make_1sm_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                        TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, ANeg, BNeg>();
    }
  }
}

template<
  class ElementAMma,
  class ElementBMma,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  int Scale,
  class BuilderScheduleTag
>
constexpr auto
sm100_make_trivial_fastFP32_tiled_mma() {
  // MMA_2SM requested
  if constexpr (cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> ) {
    using AtomLayout_MNK = decltype(make_layout(shape_div(ClusterShape_MNK{}, Shape<_2,_1,_1>{})));
    constexpr int M = cute::size<0>(TileShape_MNK{});
    constexpr int N = cute::size<1>(TileShape_MNK{});
    if constexpr (UmmaMajorA == cute::UMMA::Major::K && !cute::is_base_of_v<KernelTmaWarpSpecializedFastFP32SmemSm100, BuilderScheduleTag>) {
      return make_tiled_mma(cute::SM100_MMA_F16BF16_2x1SM_TS_SCALED<ElementAMma, ElementBMma, ElementAccumulator,
                                                     M, N,  UmmaMajorA,  UmmaMajorB, Scale>{});
    }
    else { // If A needs to be transposed by MMA, fall back to SMEM from A MMA instructions
      return make_tiled_mma(cute::SM100_MMA_F16BF16_2x1SM_SS_SCALED<ElementAMma, ElementBMma, ElementAccumulator,
                                                     M, N,  UmmaMajorA,  UmmaMajorB, Scale>{});
    }
  }
  // MMA_1SM requested
  else if constexpr (cute::is_base_of_v<KernelSchedule1Sm, BuilderScheduleTag> ) {
    // using AtomLayout_MNK = Layout<ClusterShape_MNK>;
    constexpr int M = cute::size<0>(TileShape_MNK{});
    constexpr int N = cute::size<1>(TileShape_MNK{});
    if constexpr (UmmaMajorA == cute::UMMA::Major::K  && !cute::is_base_of_v<KernelTmaWarpSpecializedFastFP32SmemSm100, BuilderScheduleTag>) {
      return make_tiled_mma(cute::SM100_MMA_F16BF16_TS_SCALED<ElementAMma, ElementBMma, ElementAccumulator,
                                                      M, N,  UmmaMajorA,  UmmaMajorB, Scale>{});
    }
    else { // If A needs to be transposed by MMA, fall back to SMEM from A MMA instructions
      return make_tiled_mma(cute::SM100_MMA_F16BF16_SS_SCALED<ElementAMma, ElementBMma, ElementAccumulator,
                                                      M, N,  UmmaMajorA,  UmmaMajorB, Scale>{});
    }
  }
  else if constexpr (cute::is_same_v<BuilderScheduleTag, KernelScheduleSm100FastFP32Gemm> ||
                     cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecializedFastFP32SmemSm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelScheduleSm100PtrArrayFastFP32Gemm> ||
                     cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecializedPtrArrayFastFP32SmemSm100>) {
    // Static cluster
    if constexpr (cute::is_static_v<ClusterShape_MNK>) {
      // For MMA_2SM we need a cluster shape that is multiple of 2x1
      // and only M=128 and M=256 are supported, otherwise, fall back to MMA_1SM
      if constexpr (cute::get<0>(ClusterShape_MNK{}) % 2 == 0 &&
                  (cute::get<0>(TileShape_MNK{}) / cute::get<0>(ClusterShape_MNK{})) % 64 == 0) {
        if constexpr (!cute::is_base_of_v<KernelTmaWarpSpecializedFastFP32SmemSm100, BuilderScheduleTag>) {
          return sm100_make_trivial_fastFP32_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK,
                                            ClusterShape_MNK, UmmaMajorA, UmmaMajorB, Scale, KernelTmaWarpSpecialized2SmFastFP32Sm100>();
        }
        else {
          return sm100_make_trivial_fastFP32_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK,
                                            ClusterShape_MNK, UmmaMajorA, UmmaMajorB, Scale, KernelTmaWarpSpecialized2SmFastFP32SmemSm100>();
        }
      }
      else {
        if constexpr (!cute::is_base_of_v<KernelTmaWarpSpecializedFastFP32SmemSm100, BuilderScheduleTag>) {
          return sm100_make_trivial_fastFP32_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK,
                                              ClusterShape_MNK, UmmaMajorA, UmmaMajorB, Scale, KernelTmaWarpSpecialized1SmFastFP32Sm100>();
        }
        else {
          return sm100_make_trivial_fastFP32_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK,
                                            ClusterShape_MNK, UmmaMajorA, UmmaMajorB, Scale, KernelTmaWarpSpecialized1SmFastFP32SmemSm100>();
        }
      }
    }
    // Dynamic cluster shape means we cannot assume we can use 2SM MMA 
    else {
      if constexpr (!cute::is_base_of_v<KernelTmaWarpSpecializedFastFP32SmemSm100, BuilderScheduleTag>) {
        return sm100_make_trivial_fastFP32_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK,
                                            ClusterShape_MNK, UmmaMajorA, UmmaMajorB, Scale, KernelTmaWarpSpecialized1SmFastFP32Sm100>();
      }
      else {
        return sm100_make_trivial_fastFP32_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK,
                                          ClusterShape_MNK, UmmaMajorA, UmmaMajorB, Scale, KernelTmaWarpSpecialized1SmFastFP32SmemSm100>();
      }
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<TileShape_MNK> == 0,
        "Unsupported policy for SM100 collective builder.");
  }
}

//Setting mma for Mixed input gemm. Here, ElementAMma should be TACompute
template<
  class ElementAMma,
  class ElementBMma,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  class KernelScheduleType
>
constexpr auto
sm100_make_trivial_mixed_input_tiled_mma() {
  constexpr int M = cute::size<0>(TileShape_MNK{});
  constexpr int N = cute::size<1>(TileShape_MNK{});
  //MMA 1Sm requested
  if constexpr (cute::is_base_of_v<KernelSchedule1Sm, KernelScheduleType> ) {
    if constexpr (UmmaMajorA == cute::UMMA::Major::K  && !cute::is_base_of_v<KernelTmaWarpSpecializedMixedInputSmemSm100, KernelScheduleType>) {
      if constexpr     (cute::is_same_v<ElementBMma, cutlass::half_t> || cute::is_same_v<ElementBMma, cutlass::bfloat16_t>) {
        return make_tiled_mma(cute::SM100_MMA_F16BF16_TS<ElementAMma, ElementBMma, ElementAccumulator,
                                                      M, N,  UmmaMajorA,  UmmaMajorB>{});
      }
      if constexpr     (cute::is_same_v<ElementBMma, cutlass::float_e4m3_t>) {
        return make_tiled_mma(cute::SM100_MMA_F8F6F4_TS<ElementAMma, ElementBMma, ElementAccumulator,
                                                      M, N,  UmmaMajorA,  UmmaMajorB>{});
      }
    }
    else { // If A needs to be transposed by MMA, fall back to SMEM from A MMA instructions
      if constexpr     (cute::is_same_v<ElementBMma, cutlass::half_t> || cute::is_same_v<ElementBMma, cutlass::bfloat16_t>) {
        return make_tiled_mma(cute::SM100_MMA_F16BF16_SS<ElementAMma, ElementBMma, ElementAccumulator,
                                                      M, N,  UmmaMajorA,  UmmaMajorB>{});
      }
      if constexpr     (cute::is_same_v<ElementBMma, cutlass::float_e4m3_t>) {
        return make_tiled_mma(
          cute::MMA_Traits<
            cute::SM100_MMA_F8F6F4_SS,
            ElementAMma,
            ElementBMma, 
            ElementAccumulator, 
            cute::C<M>, 
            cute::C<N>, 
            cute::integral_constant<UMMA::Major, UmmaMajorA>,
            cute::integral_constant<UMMA::Major, UmmaMajorB>,
            cute::integral_constant<UMMA::ScaleIn, cute::UMMA::ScaleIn::One>,
            cute::integral_constant<UMMA::ScaleIn, cute::UMMA::ScaleIn::One>>{});
      }
    }
  }
  //MMA 2Sm requested
  else if constexpr (cute::is_base_of_v<KernelSchedule2Sm, KernelScheduleType>) {
    if constexpr (UmmaMajorA == cute::UMMA::Major::K  && !cute::is_base_of_v<KernelTmaWarpSpecializedMixedInputSmemSm100, KernelScheduleType>) {
      if constexpr     (cute::is_same_v<ElementBMma, cutlass::half_t> || cute::is_same_v<ElementBMma, cutlass::bfloat16_t>) {
        return make_tiled_mma(cute::SM100_MMA_F16BF16_2x1SM_TS<ElementAMma, ElementBMma, ElementAccumulator,
                                                      M, N,  UmmaMajorA,  UmmaMajorB>{});
      }
      if constexpr     (cute::is_same_v<ElementBMma, cutlass::float_e4m3_t>) {
        return make_tiled_mma(cute::SM100_MMA_F8F6F4_2x1SM_TS<ElementAMma, ElementBMma, ElementAccumulator,
                                                      M, N,  UmmaMajorA,  UmmaMajorB>{});
      }
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<TileShape_MNK> == 0,
        "Unsupported policy for SM100 collective builder.");
  }
}

template<
  class CtaShape_MNK
>
constexpr auto
sm100_simt_f32_warp_shape_mnk_selector() {
  using namespace cute;

  constexpr int CtaShape_M = cute::size<0>(CtaShape_MNK{});
  constexpr int CtaShape_N = cute::size<1>(CtaShape_MNK{});
  constexpr int CtaShape_K = cute::size<2>(CtaShape_MNK{});

  // CTA tile shape M and N are supposed to be divisible by 32.
  static_assert(CtaShape_M % 32 == 0, "CtaShape_M needs to be divisible by 32.");
  static_assert(CtaShape_N % 32 == 0, "CtaShape_N needs to be divisible by 32.");

  // WarpShape_MNK configuration
  // We assume WarpShape_K is always 1 in our SM100 SIMT SGEMM implementation.
  if constexpr (CtaShape_M >= CtaShape_N) {
    if constexpr (CtaShape_M == 256 && CtaShape_N == 128) {
      return cute::Shape<_4, _2, _1>{};
    }
    else if constexpr ((CtaShape_M == 64 || CtaShape_M == 32) && CtaShape_N == 32) {
      return cute::Shape<_1, _2, _1>{};
    }
    else {
      return cute::Shape<_2, _2, _1>{};
    }
  }
  else {
    if constexpr (CtaShape_M == 128 && CtaShape_N == 256) {
      return cute::Shape<_2, _4, _1>{};
    }
    else if constexpr (CtaShape_M == 32 && CtaShape_N == 64) {
      return cute::Shape<_1, _2, _1>{};
    }
    else {
      return cute::Shape<_1, _4, _1>{};
    }
  }
}


template <
  class ElementPairA,
  class ElementPairB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  detail::blockscaled::BlockScaledInstr Instr,
  class BuilderScheduleTag
>
constexpr auto
sm100_make_blockscaled_1sm_trivial_tiled_mma() {
  // For MMA_1sm atoms, the MMA's AtomLayout is same as the ClusterShape
  using AtomLayout_MNK = Layout<ClusterShape_MNK>;
  constexpr int M = cute::size<0>(TileShape_MNK{});
  static_assert(M == 128, "Invalid TileShape_M.");

  // Do not allow a tiled MMA N mode > 1, as that is not reasonable.
  constexpr int N = cute::size<1>(TileShape_MNK{});
  static_assert(N == 64 || N == 128 || N == 192 || N == 256, "Invalid TileShape_N.");

  using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
  using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
  using ElementA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type;
  using ElementB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type;
  constexpr uint32_t SfVectorSizeA = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::SfVectorSize;
  [[maybe_unused]] constexpr uint32_t SfVectorSizeB = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::SfVectorSize;

  using ElementAMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA, Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB, Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8>());

  using ElementSF = ElementSFA;

  if constexpr (Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8) {
    if constexpr (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ) {
      return make_tiled_mma(cute::SM100_MMA_MXF8F6F4_SS_SPARSE<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                          M, N, UmmaMajorA, UmmaMajorB>{});
    }
    else {
      return make_tiled_mma(cute::SM100_MMA_MXF8F6F4_SS<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                  M, N, UmmaMajorA, UmmaMajorB>{});
    }
  }
  else if constexpr (Instr == detail::blockscaled::BlockScaledInstr::MXF4_NVF4) {
    constexpr int SfVectorSize = SfVectorSizeA;
    if constexpr (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ) {
      return make_tiled_mma(cute::SM100_MMA_MXF4NVF4_SS_SPARSE<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                          M, N, SfVectorSize, UmmaMajorA, UmmaMajorB>{});
    }
    else {
      return make_tiled_mma(cute::SM100_MMA_MXF4_SS<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                  M, N, SfVectorSize, UmmaMajorA, UmmaMajorB>{});
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAMma>,
        "Unsupported configuration for SM100 collective builder.");
  }
}

template <
  class ElementPairA,
  class ElementPairB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  detail::blockscaled::BlockScaledInstr Instr,
  class BuilderScheduleTag
>
constexpr auto
sm100_make_blockscaled_2sm_trivial_tiled_mma() {

  constexpr int M = cute::size<0>(TileShape_MNK{});
  static_assert(M == 256, "Invalid TileShape_M.");

  // Do not allow a tiled MMA N mode > 1, as that is not reasonable.
  constexpr int N = cute::size<1>(TileShape_MNK{});
  static_assert(N == 64 || N == 128 || N == 192 || N == 256, "Invalid TileShape_N.");

  using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
  using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
  using ElementA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type;
  using ElementB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type;
  constexpr uint32_t SfVectorSizeA = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::SfVectorSize;
  [[maybe_unused]] constexpr uint32_t SfVectorSizeB = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::SfVectorSize;
  using ElementAMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA, Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB, Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8>());

  using ElementSF = ElementSFA;

  if constexpr (Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8) {
    if constexpr (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ) {
      return make_tiled_mma(cute::SM100_MMA_MXF8F6F4_2x1SM_SS_SPARSE<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                                M, N, UmmaMajorA, UmmaMajorB>{});
    }
    else {
      return make_tiled_mma(cute::SM100_MMA_MXF8F6F4_2x1SM_SS<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                        M, N, UmmaMajorA, UmmaMajorB>{});
    }
  }
  else if constexpr (Instr == detail::blockscaled::BlockScaledInstr::MXF4_NVF4) {
    constexpr int SfVectorSize = SfVectorSizeA > SfVectorSizeB ? SfVectorSizeA : SfVectorSizeB;
    if constexpr (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ) {
      return make_tiled_mma(cute::SM100_MMA_MXF4NVF4_2x1SM_SS_SPARSE<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                                M, N, SfVectorSize, UmmaMajorA, UmmaMajorB>{});
    }
    else {
      return make_tiled_mma(cute::SM100_MMA_MXF4_2x1SM_SS<ElementAMma, ElementBMma, ElementAccumulator, ElementSF,
                                                        M, N, SfVectorSize, UmmaMajorA, UmmaMajorB>{});
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAMma>,
        "Unsupported configuration for SM100 collective builder.");
  }
}

template <
  class ElementPairA,
  class ElementPairB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  detail::blockscaled::BlockScaledInstr Instr,
  class BuilderScheduleTag,
  bool Is2SM
>
struct TrivialBlockscaledMma {};

template <
  class ElementPairA,
  class ElementPairB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  detail::blockscaled::BlockScaledInstr Instr,
  class BuilderScheduleTag
>
struct TrivialBlockscaledMma <
  ElementPairA,
  ElementPairB,
  ElementAccumulator,
  TileShape_MNK,
  ClusterShape_MNK,
  UmmaMajorA,
  UmmaMajorB,
  Instr,
  BuilderScheduleTag,
  true /*Is2SM*/> {
    using type = decltype(sm100_make_blockscaled_2sm_trivial_tiled_mma<ElementPairA, ElementPairB, ElementAccumulator,
                                                        TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, Instr, BuilderScheduleTag>());
  };

template <
  class ElementPairA,
  class ElementPairB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  detail::blockscaled::BlockScaledInstr Instr,
  class BuilderScheduleTag
>
struct TrivialBlockscaledMma<
  ElementPairA,
  ElementPairB,
  ElementAccumulator,
  TileShape_MNK,
  ClusterShape_MNK,
  UmmaMajorA,
  UmmaMajorB,
  Instr,
  BuilderScheduleTag,
  false /*Is2SM*/> {
    using type = decltype(sm100_make_blockscaled_1sm_trivial_tiled_mma<ElementPairA, ElementPairB, ElementAccumulator,
                                                        TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, Instr, BuilderScheduleTag>());
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective
