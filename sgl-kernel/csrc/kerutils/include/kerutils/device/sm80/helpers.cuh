// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/common.h"

namespace kerutils {

// ============================================================
// Global / shared memory store intrinsics
// ============================================================

CUTE_DEVICE
void _store_256b(
    uint32_t const& src0,
    uint32_t const& src1,
    uint32_t const& src2,
    uint32_t const& src3,
    uint32_t const& src4,
    uint32_t const& src5,
    uint32_t const& src6,
    uint32_t const& src7,
    void* gmem_addr) {
  asm volatile(
      "st.global.L1::no_allocate.v8.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n" ::"l"(gmem_addr),
      "r"(src0),
      "r"(src1),
      "r"(src2),
      "r"(src3),
      "r"(src4),
      "r"(src5),
      "r"(src6),
      "r"(src7));
}

// reference: https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm100.hpp#L70
CUTE_DEVICE
void store_256b(void* src, void* dst) {
  uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src);
  _store_256b(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], src_ptr[5], src_ptr[6], src_ptr[7], dst);
}

template <typename T>
CUTE_DEVICE void store_128b(void* smem_ptr, const T& data) {
  static_assert(sizeof(T) == 16);
  *(__int128*)smem_ptr = *(__int128*)&data;
}

// Predicated copy helper (generic, works on SM80+)
// Supports predication on both MN and K dimensions with optional OOB clearing.
template <
    bool Is_even_MN = true,
    bool Is_even_K = true,
    bool Clear_OOB_MN = false,
    bool Clear_OOB_K = true,
    class CopyAtom,
    class TV,
    class Tiler,
    typename Engine0,
    typename Layout0,
    typename Engine1,
    typename Layout1,
    typename Engine2,
    typename Layout2,
    typename Engine3,
    typename Layout3>
CUTLASS_DEVICE void copy_pred(
    cute::TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
    cute::Tensor<Engine0, Layout0> const& S,
    cute::Tensor<Engine1, Layout1>& D,
    cute::Tensor<Engine2, Layout2> const& identity_MN,
    cute::Tensor<Engine3, Layout3> const& predicate_K,
    const int max_MN = 0) {
  using namespace cute;
  // Decay TiledCopy to CopyAtom
  auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K
  // There's no case where !Clear_OOB_K && Clear_OOB_MN
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
  auto has_with_bool =
      cute::is_valid([](auto t) -> void_t<decltype(declval<typename decltype(t)::Traits>().with(true))> {}, copy_atom);
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    bool predicate_mn = Is_even_MN || get<0>(identity_MN(_0{}, m, _0{})) < max_MN;
    // NOTE: currently only this predicate is true because we set Clear_OOB_MN=false
    if constexpr (Is_even_MN || !Clear_OOB_MN) {
      if (Is_even_MN || predicate_mn) {
#pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          if constexpr (Is_even_K || !Clear_OOB_K) {
            if (Is_even_K || predicate_K(k)) {
              cute::copy(copy_atom, S(_, m, k), D(_, m, k));
            }
          } else {  // Clear_OOB_K == true && Is_even_K == false
            // If copy traits can be transformed with a predicate value, do it, otherwise branch here
            if constexpr (has_with_bool) {
              cute::copy(copy_atom.with(predicate_K(k)), S(_, m, k), D(_, m, k));
            } else {
              if (predicate_K(k)) {
                cute::copy(copy_atom, S(_, m, k), D(_, m, k));
              } else {
                cute::clear(D(_, m, k));
              }
            }
          }
        }
      }
    } else {  // Clear_OOB_MN == true && Is_even_MN == false, also implies Clear_OOB_K == true
      if constexpr (!has_with_bool) {
        if (predicate_mn) {
#pragma unroll
          for (int k = 0; k < size<2>(S); ++k) {
            if (Is_even_K || predicate_K(k)) {
              cute::copy(copy_atom, S(_, m, k), D(_, m, k));
            } else if (Clear_OOB_K) {
              cute::clear(D(_, m, k));
            }
          }
        } else {
          cute::clear(D(_, m, _));
        }
      } else {  // combine the mn predicate with the k predicate
#pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          cute::copy(copy_atom.with(predicate_mn && (Is_even_K || predicate_K(k))), S(_, m, k), D(_, m, k));
        }
      }
    }
  }
}

}  // namespace kerutils
