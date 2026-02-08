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


#pragma once

#include "cutlass/gemm/collective/builders/sm1xx_common.inl"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/arch/arch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int sm120_smem_capacity_bytes = cutlass::arch::sm120_smem_capacity_bytes;
// Helper for selecting the shared memory copy atom to use for operand A
template <
  class ElementA,
  class ElementB,
  bool UseF8f6f4
>
CUTLASS_HOST_DEVICE constexpr
auto
sm120_rr_smem_copy_selector_A() {
  if constexpr (UseF8f6f4) {
    if constexpr (sizeof_bits_v<ElementA> == 6) {
      return SM100_SU6_DU8x16_x4_LDSM_N{};
    }
    else if constexpr (sizeof_bits_v<ElementA> == 4) {
      return SM100_SU4_DU8x16_x4_LDSM_N{};
    }
    else {
      return SM75_U32x4_LDSM_N{};
    }
  }
  else {
    return SM75_U32x4_LDSM_N{};
  }
}

// Helper for selecting the shared memory copy atom to use for operand B
template <
  class ElementA,
  class ElementB,
  bool UseF8f6f4
>
CUTLASS_HOST_DEVICE constexpr
auto
sm120_rr_smem_copy_selector_B() {
  if constexpr (UseF8f6f4) {
    if constexpr (sizeof_bits_v<ElementB> == 6) {
      return SM100_SU6_DU8x16_x4_LDSM_N{};
    }
    else if constexpr (sizeof_bits_v<ElementB> == 4) {
      return SM100_SU4_DU8x16_x4_LDSM_N{};
    }
    else {
      return SM75_U32x4_LDSM_N{};
    }
  } 
  else {
    return SM75_U32x4_LDSM_N{};
  }
}

template <class ElementType, class MajorSize>
CUTLASS_HOST_DEVICE constexpr
auto
sm120_rr_smem_selector() {
  static_assert(cutlass::sizeof_bits<ElementType>::value <= 8, "Unsupported element size.");
 
  if constexpr      (MajorSize{} % size<1>(UMMA::Layout_K_SW128_Atom<ElementType>{}) == 0) {
    return UMMA::Layout_K_SW128_Atom<ElementType>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_SW64_Atom<ElementType>{}) == 0) {
    return UMMA::Layout_K_SW64_Atom<ElementType>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_SW32_Atom<ElementType>{}) == 0) {
    return UMMA::Layout_K_SW32_Atom<ElementType>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_INTER_Atom<ElementType>{}) == 0) {
    return UMMA::Layout_K_INTER_Atom<ElementType>{};
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementType>, "No shared memory copy atom can be selected.");
  }
}

template <class ElementType, class MajorSize, class Sparsity>
CUTLASS_HOST_DEVICE constexpr
auto
sm120_rr_smem_selector_sparse() {
  static_assert(cutlass::sizeof_bits<ElementType>::value <= 8, "Unsupported element size.");

   if constexpr      (MajorSize{} % size<1>(UMMA::Layout_K_SW128_SpAtom<ElementType, Sparsity{}>{}) == 0) {
    return UMMA::Layout_K_SW128_SpAtom<ElementType, Sparsity{}>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_SW64_SpAtom<ElementType, Sparsity{}>{}) == 0) {
    return UMMA::Layout_K_SW64_SpAtom<ElementType, Sparsity{}>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_SW32_SpAtom<ElementType, Sparsity{}>{}) == 0) {
    return UMMA::Layout_K_SW32_SpAtom<ElementType, Sparsity{}>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_INTER_SpAtom<ElementType, Sparsity{}>{}) == 0) {
    return UMMA::Layout_K_INTER_SpAtom<ElementType, Sparsity{}>{};
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementType>, "No shared memory copy atom can be selected.");
  }
}

template <int SFVectorSize>
CUTLASS_HOST_DEVICE constexpr
auto
sm120_tile_n_permute_selector() {
  // VS = 16
  if constexpr (SFVectorSize == 16) {
    // Permute in the N mode to allow a warp to own all the elements needed for SF reduction
    return cute::Layout<cute::Shape<_8,_2,_2>, cute::Stride<_1, _16,_8>>{};
  }
  // VS = 32
  else if constexpr (SFVectorSize == 32) {
    return cute::Layout<cute::Shape<_8,_2,_2>, cute::Stride<_1, _16,_8>>{};
  }
  else {
    static_assert(cutlass::detail::dependent_false<cute::C<SFVectorSize>>,
      "Unsupported SFVectorSize for SM120 collective builder.");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective::detail
