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

#include "cute/layout.hpp"     // cute::Shape
#include "cute/numeric/numeric_types.hpp" // cute::sizeof_bits_v
#include "cutlass/arch/mma.h"  // cutlass::arch::OpClassTensorOp, cutlass::OpClassSparseTensorOp
#include "cute/atom/copy_traits_sm100.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/util/type_traits.hpp" // cute::is_same_v

#include "cutlass/detail/dependent_false.hpp" // cutlass::detail::dependent_false
#include "cutlass/detail/layout.hpp"
#include "cutlass/numeric_size.h" // cutlass::bytes_to_bits
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/epilogue/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/operations.hpp" // detail::is_sfd_epilogue_v
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"
#include "cutlass/cutlass.h"

#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(type_traits)
#else
#include <type_traits>
#endif

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

///////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the smem layout atom to be used for C or D matrix
template<class GmemStrideType, class Element, class EpilogueTile_MN>
constexpr auto
sm100_get_epilogue_smem_swizzle_layout_atom() {
  using namespace cute;

  // Get the max contiguous tile usable by TMA
  [[maybe_unused]] auto tma_tile = cute::transform(EpilogueTile_MN{},
    [](auto const& epi_tile) {
        // assumes get<0>(epi_tile) is coalesced and unit stride
        return size<0>(coalesce(right_inverse(make_layout(get<0>(epi_tile)))));
    });

  // ColMajor C/D (M-major)
  if constexpr (cutlass::detail::is_major<0>(GmemStrideType{})) {
    return cutlass::gemm::collective::detail::sm100_smem_selector<
      UMMA::Major::MN, Element, decltype(get<0>(tma_tile)), decltype(get<1>(tma_tile))
    >();
  }
  // RowMajor C/D (N-major)
  else if constexpr (cutlass::detail::is_major<1>(GmemStrideType{})) {
    return cutlass::gemm::collective::detail::sm100_smem_selector<
      UMMA::Major::K , Element, decltype(get<0>(tma_tile)), decltype(get<1>(tma_tile))
    >();
  }
  else {
    static_assert(cutlass::detail::dependent_false<GmemStrideType>, "Unsupported gmem layout.");
  }
}

namespace sparse {

template <
  class CtaTileShape_MNK,
  class EpilogueTileType,
  class TmemWarpShape_MN,
  class ElementC,
  class StrideC,
  class ElementD,
  class StrideD,
  class EpilogueScheduleType,
  class FusionOp
>
constexpr auto
sm100_sparse_compute_tile_shape_or_override() {
  if constexpr (cute::is_same_v<EpilogueTileType, EpilogueTileAuto>) {
    constexpr int CtaM = size<0>(CtaTileShape_MNK{});
    constexpr int CtaN = size<1>(CtaTileShape_MNK{});
    constexpr int CtaK = size<2>(CtaTileShape_MNK{});
    constexpr int WarpM = size<0>(TmemWarpShape_MN{});
    constexpr int WarpN = size<1>(TmemWarpShape_MN{});
    constexpr bool DisableSource = cute::is_void_v<ElementC>;

    // For SM100 SP BSSP kernel, we always have EpiTileM = CtaM
    constexpr int EpiTileM = CtaM;

    constexpr bool Is1Sm = cute::is_base_of_v<TmaWarpSpecialized1Sm, EpilogueScheduleType>;
    constexpr bool Is2Sm = cute::is_base_of_v<TmaWarpSpecialized2Sm, EpilogueScheduleType>;

    constexpr bool IsBsspMxf8f6f4 = cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized1SmMxf8f6f4, EpilogueScheduleType> ||
                                    cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized2SmMxf8f6f4, EpilogueScheduleType>;
    constexpr bool IsBsspNvf4 = cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized1SmNvf4, EpilogueScheduleType> ||
                                cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized2SmNvf4, EpilogueScheduleType>;
    constexpr bool IsBsspMxf4 = cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized1SmMxf4, EpilogueScheduleType> ||
                                cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized2SmMxf4, EpilogueScheduleType>;
    constexpr bool IsBssp = (IsBsspMxf8f6f4 || IsBsspNvf4 || IsBsspMxf4);
    constexpr bool IsSp = not IsBssp;

    constexpr auto compute_epi_tile_n = [&](int epi_smem_size_kb, int num_epi_stage, int element_bit_size) constexpr -> int {
      // Use Epi Smem + Num Epi Stage to compute Epi Tile N
      return cutlass::bytes_to_bits(epi_smem_size_kb * 1024) / num_epi_stage / EpiTileM / element_bit_size;
    };

    // Row major SFD, EpiTileN = SFD_VS multiplier
    constexpr bool is_sfd_row_major = (not cute::is_void_v<typename FusionOp::GmemLayoutTagScalefactor>) && 
                                      cute::is_same_v<typename FusionOp::GmemLayoutTagScalefactor, cutlass::layout::RowMajor>;
    constexpr bool is_sfd_row_major_vs64 = is_sfd_row_major ? (FusionOp::SFVecSize == 64) : false;

    constexpr auto EpiTileN = [&]() constexpr -> int {
      // VoidC Kernel
      if (DisableSource) {
        auto d_bits = cute::sizeof_bits_v<ElementD>;
        if (IsSp) {
          if (d_bits == 32) {
            if (Is1Sm && CtaN ==  64) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is2Sm && CtaN ==  64) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is2Sm && CtaN == 128) {
              bool Is4KBlock = (CtaK == 64 || CtaK == 256);
              if (Is4KBlock)  { return compute_epi_tile_n(16, 2, d_bits); }
                                return compute_epi_tile_n(32, 2, d_bits);
            }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is2Sm && CtaN == 256) {
              bool Is4KBlock = (CtaK == 64 || CtaK == 256);
              if (Is4KBlock)  { return compute_epi_tile_n(16, 2, d_bits); }
                                return compute_epi_tile_n(32, 2, d_bits);
            }
          }
          if (d_bits == 16) {
            if (Is1Sm && CtaN ==  64) { return compute_epi_tile_n(16, 2, d_bits); }
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is2Sm && CtaN ==  64) { return compute_epi_tile_n(16, 2, d_bits); }
            if (Is2Sm && CtaN == 128) {
              // Prioritize Mxf8f6f4 kernel
              bool Is4KBlock = (CtaK == 256);
              if (Is4KBlock)  { return compute_epi_tile_n(16, 2, d_bits); }
                                return compute_epi_tile_n(32, 2, d_bits);
            }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is2Sm && CtaN == 256) {
              // Prioritize Mxf8f6f4 kernel
              bool IsHmma2KBlock = (CtaK == 64);

              if (IsHmma2KBlock)  { return compute_epi_tile_n(32, 2, d_bits); } 
                                    return compute_epi_tile_n(16, 2, d_bits);
            }
          }
          if (d_bits == 8) {
            if (Is1Sm && CtaN ==  64) { return compute_epi_tile_n( 8, 2, d_bits); }
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(16, 2, d_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(16, 2, d_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(16, 2, d_bits); }
            if (Is2Sm && CtaN ==  64) { return compute_epi_tile_n( 8, 2, d_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(16, 2, d_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(24, 3, d_bits); }
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(16, 2, d_bits); }
          }
        }
        if (IsBsspMxf8f6f4) {
          if (d_bits == 32) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(64, 4, d_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(16, 2, d_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(64, 4, d_bits); }
          }
          if (d_bits == 16) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(16, 2, d_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(64, 4, d_bits); }
          }
          if (d_bits == 8) {
            if (Is1Sm && CtaN == 128) { 
              // SFD VS64 require EpiTileN to be multiplier of 64
              if (is_sfd_row_major_vs64) { return compute_epi_tile_n(24, 3, d_bits); }
              else                       { return compute_epi_tile_n(12, 3, d_bits); }}
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(16, 2, d_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(32, 4, d_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(16, 2, d_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(24, 3, d_bits); }
            if (Is2Sm && CtaN == 256) {
              // SFD VS64 require EpiTileN to be multiplier of 64
              if (is_sfd_row_major_vs64) { return compute_epi_tile_n(16, 2, d_bits); }
              else                       { return compute_epi_tile_n( 8, 2, d_bits); }}
          }
        }
        if (IsBsspNvf4) {
          if (d_bits == 32) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is1Sm && CtaN == 256) {
              bool Is4KBlock = (CtaK == 512);
              if (Is4KBlock) { return compute_epi_tile_n(64, 2, d_bits); }
                               return compute_epi_tile_n(32, 2, d_bits);
            }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is2Sm && CtaN == 256) {
              bool Is4KBlock = (CtaK == 512);
              if (Is4KBlock) { return compute_epi_tile_n(64, 2, d_bits); }
                               return compute_epi_tile_n(48, 3, d_bits);
            }
          }
          if (d_bits == 16) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, d_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, d_bits); }
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, d_bits); }
          }
          if (d_bits == 4) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n( 8, 2, d_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(12, 3, d_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(16, 4, d_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n( 8, 2, d_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(12, 3, d_bits); }
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(16, 4, d_bits); }
          }
        }
        if (IsBsspMxf4) {
          if (d_bits == 32) {
            if (CtaN == 256) {
              return compute_epi_tile_n(32, 2, d_bits);
            }
          }
        }
        // Fallback
        return compute_epi_tile_n(16, 2, d_bits);
      }
      // NonVoidC Kernel
      if (not DisableSource) {
        auto d_bits = cute::sizeof_bits_v<ElementD>;
        auto c_bits = cute::sizeof_bits_v<ElementC>;
        if (IsSp) {
          if (c_bits == 32 && d_bits == 32) {
            if (Is1Sm && CtaN ==  64) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(64, 4, c_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is2Sm && CtaN ==  64) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is2Sm && CtaN == 128) {
              bool Is4KBlock = (CtaK == 64 || CtaK == 256);
              if (Is4KBlock) { return compute_epi_tile_n(32, 4, c_bits); }
                               return compute_epi_tile_n(64, 4, c_bits);
            }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is2Sm && CtaN == 256) {
              bool IsTfmma2KBlock = (CtaK == 32);
              if (IsTfmma2KBlock) { return compute_epi_tile_n(32, 4, c_bits); }
                                    return compute_epi_tile_n(64, 4, c_bits);
            }
          }
          if (c_bits == 16 && (d_bits == 16 || d_bits == 8)) {
            if (Is1Sm && CtaN ==  64) { return compute_epi_tile_n(16, 4, c_bits); }
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is2Sm && CtaN ==  64) { return compute_epi_tile_n(16, 4, c_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(64, 4, c_bits); }
          }
          if (c_bits == 8 && d_bits == 8) {
            // 8 bit C assume no SMEM reuse between C and D. Smem size mentioned below is ONLY for C.
            if (Is1Sm && CtaN ==  64) { return compute_epi_tile_n( 8, 2, c_bits); }
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(16, 2, c_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(24, 3, c_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is2Sm && CtaN ==  64) { return compute_epi_tile_n( 8, 2, c_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(16, 2, c_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(24, 3, c_bits); }
            if (Is2Sm && CtaN == 256) {
              bool Is4KBlock = (CtaK == 256);
              if (Is4KBlock) { return compute_epi_tile_n(32, 4, c_bits); }
                               return compute_epi_tile_n(16, 2, c_bits);
            }
          }
        }
        if (IsBsspMxf8f6f4) {
          if (c_bits == 32 && d_bits == 32) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(64, 4, c_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(64, 4, c_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(64, 4, c_bits); }
          }
          if (c_bits == 16 && d_bits == 16) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(64, 4, c_bits); }
          }
          if (c_bits == 16 && d_bits == 8) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, c_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, c_bits); }
            if (Is2Sm && CtaN == 192) {
              // SFD VS64 require EpiTileN to be multiplier of 64
              if (is_sfd_row_major_vs64) { return compute_epi_tile_n(64, 4, c_bits); }
              else                       { return compute_epi_tile_n(32, 4, c_bits); }}
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(64, 4, c_bits); }
          }
        }
        if (IsBsspNvf4) {
          if (c_bits == 32 && d_bits == 32) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(64, 4, c_bits); }
            if (Is1Sm && CtaN == 256) {
              bool Is4KBlock = (CtaK == 512);
              if (Is4KBlock) { return compute_epi_tile_n(64, 2, c_bits); }
                               return compute_epi_tile_n(64, 4, c_bits);
            }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(64, 4, c_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is2Sm && CtaN == 256) {
              bool Is4KBlock = (CtaK == 512);
              if (Is4KBlock) { return compute_epi_tile_n(64, 2, c_bits); }
                               return compute_epi_tile_n(48, 3, c_bits);
            }
          }
          if (c_bits == 16 && d_bits == 16) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is1Sm && CtaN == 256) {
              bool Is4KBlock = (CtaK == 512);
              if (Is4KBlock) { return compute_epi_tile_n(64, 4, c_bits); }
                               return compute_epi_tile_n(32, 4, c_bits);
            }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(32, 4, c_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, c_bits); }
          }
          if (c_bits == 16 && d_bits == 4) {
            if (Is1Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, c_bits); }
            if (Is1Sm && CtaN == 192) { return compute_epi_tile_n(32, 2, c_bits); }
            if (Is1Sm && CtaN == 256) { return compute_epi_tile_n(32, 2, c_bits); }
            if (Is2Sm && CtaN == 128) { return compute_epi_tile_n(32, 2, c_bits); }
            if (Is2Sm && CtaN == 192) { return compute_epi_tile_n(48, 3, c_bits); }
            if (Is2Sm && CtaN == 256) { return compute_epi_tile_n(48, 3, c_bits); }
          }
        }
        if (IsBsspMxf4) {
          if (c_bits == 32 && d_bits == 32) {
            if (CtaN == 256) {
              return compute_epi_tile_n(64, 4, d_bits);
            }
          }
        }
        // Fallback
        return compute_epi_tile_n(32, 4, c_bits);
      }
    }();

    // stride by tmem warp layout and return a by-mode tiler
    auto tile_m = Layout<Int<EpiTileM>>{};
    auto tile_n = Layout<Shape <Int<EpiTileN / WarpN>,Int<        WarpN>>,
                         Stride<Int<               1>,Int<CtaN / WarpN>>>{};

    return make_tile(tile_m, coalesce(tile_n));
  }
  else if constexpr (cute::is_tuple<EpilogueTileType>::value) {
    return EpilogueTileType{};
  }
  else {
    static_assert(cutlass::detail::dependent_false<EpilogueTileType>, "Invalid type for EpilogueTileType.");
  }
}

template <
  class CtaTileShape_MNK,
  class EpilogueTile_MN,
  class ElementC,
  class ElementD,
  class EpilogueScheduleType
>
constexpr auto
sm100_sparse_get_tma_dispatch_policy() {
  using EpilogueTileShape_MN = decltype(product_each(shape(EpilogueTile_MN{})));
  constexpr int EpiTiles = size(shape_div(take<0,2>(CtaTileShape_MNK{}), EpilogueTileShape_MN{}));
  constexpr int FragmentSize = size(EpilogueTileShape_MN{}) / NumThreadsPerWarpGroup;
  constexpr int CtaN = cute::size<1>(CtaTileShape_MNK{});
  constexpr int CtaK = cute::size<2>(CtaTileShape_MNK{});

  constexpr bool IsBsspMxf8f6f4 = cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized1SmMxf8f6f4, EpilogueScheduleType> ||
                                  cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized2SmMxf8f6f4, EpilogueScheduleType>;
  constexpr bool IsBsspNvf4 = cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized1SmNvf4, EpilogueScheduleType> ||
                              cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized2SmNvf4, EpilogueScheduleType>;
  constexpr bool IsBsspMxf4 = cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized1SmMxf4, EpilogueScheduleType> ||
                              cute::is_same_v<cutlass::epilogue::TmaWarpSpecialized2SmMxf4, EpilogueScheduleType>;
  constexpr bool IsBssp = (IsBsspMxf8f6f4 || IsBsspNvf4 || IsBsspMxf4);
  constexpr bool IsSp = not IsBssp;
  constexpr bool Is1Sm = cute::is_base_of_v<TmaWarpSpecialized1Sm, EpilogueScheduleType>;
  constexpr bool Is2Sm = cute::is_base_of_v<TmaWarpSpecialized2Sm, EpilogueScheduleType>;

  // 8b residuals load fast and consume little smem, so the perf cost of waiting on stores to finish outweighs the cost of extra allocation
  constexpr bool ReuseSmem = sizeof_bits_v<ElementC> > 8;

  // TMA store delay performs worse with residual loads
  constexpr bool DelayTmaStore = is_void_v<ElementC>;

  constexpr auto ExpectedStagesD = [&]() constexpr -> int {
    auto d_bits = cute::sizeof_bits_v<ElementD>;
    auto c_bits = cute::sizeof_bits_v<ElementC>;
    // None void_c kernel pick 2stageD here, in reality it may choose reuse smemC
    if (not cute::is_void_v<ElementC>) {
      return 2;
    }
    // Void_C kernel have fine tunned stageD
    else {
      if (IsSp) {
        if (((d_bits == 32 || d_bits == 16) && ((Is1Sm && CtaN == 192) ||
                                                (Is1Sm && CtaN == 256) ||
                                                (Is2Sm && CtaN == 192))) ||
            (d_bits == 8 && Is2Sm && CtaN == 192)) {
          return 3;
        }
        return 2;
      }
      if (IsBsspMxf8f6f4) {
        if (((d_bits == 32 || d_bits == 16) && Is1Sm && CtaN == 256) ||
            (d_bits == 8 && ((Is1Sm && CtaN == 128) ||
                             (Is2Sm && CtaN == 192)))) {
          return 3;
        }
        if ((d_bits == 32 && Is1Sm && CtaN == 128) ||
            (d_bits == 32 && Is2Sm && CtaN == 256) ||
            (d_bits == 16 && Is2Sm && CtaN == 256) ||
            (d_bits ==  8 && Is1Sm && CtaN == 256)) {
          return 4;
        }
        return 2;
      }
      if (IsBsspNvf4) {
        if ((d_bits == 32 && ((Is1Sm && CtaN == 128) ||
                              (Is2Sm && CtaN == 192) ||
                              (Is2Sm && CtaN == 256 && CtaK == 256))) ||
            (d_bits == 16 && ((Is2Sm && CtaN == 192) ||
                              (Is2Sm && CtaN == 256))) ||
            (d_bits ==  4 && ((Is1Sm && CtaN == 192) ||
                              (Is2Sm && CtaN == 192)))) {
          return 3;
        }
        if ((d_bits == 4 && ((Is1Sm && CtaN == 256) ||
                             (Is2Sm && CtaN == 256)))) {
          return 4;
        }
        return 2;
      }
      return 2;
    }
  }();

  constexpr auto ExpectedStagesC = [&]() constexpr -> int {
    auto d_bits = cute::sizeof_bits_v<ElementD>;
    auto c_bits = cute::sizeof_bits_v<ElementC>;
    // Void_c kernel only use smemD. StageC doesn't matter
    if (cute::is_void_v<ElementC>) {
      return 4;
    }
    // None VoidC kernel have fine tunned stageC
    else {
      if (IsSp) {
        if ((((c_bits == 32 && d_bits == 32) ||
              (c_bits == 16 && d_bits == 16) ||
              (c_bits == 16 && d_bits  == 8)) && ((Is1Sm && CtaN == 192) ||
                                                  (Is1Sm && CtaN == 256) ||
                                                  (Is2Sm && CtaN == 192))) ||
            (c_bits == 8 && d_bits == 8 && ((Is1Sm && CtaN == 192) ||
                                            (Is2Sm && CtaN == 192)))) {
          return 3;
        }
        if (c_bits == 8 && d_bits == 8 && ((Is1Sm && CtaN ==  64) ||
                                           (Is1Sm && CtaN == 128) ||
                                           (Is2Sm && CtaN ==  64) ||
                                           (Is2Sm && CtaN == 128) ||
                                           (Is2Sm && CtaN == 256 && CtaK == 128))) {
          return 2;
        }
        return 4;
      }
      if (IsBsspMxf8f6f4) {
        if ((c_bits == 32 && d_bits == 32 && Is1Sm && CtaN == 256) ||
            (c_bits == 16 && d_bits == 16 && ((Is1Sm && CtaN == 192) ||
                                              (Is1Sm && CtaN == 256))) ||
            (c_bits == 16 && d_bits ==  8 && ((Is1Sm && CtaN == 128) ||
                                              (Is1Sm && CtaN == 256) ||
                                              (Is1Sm && CtaN == 128) ||
                                              (Is2Sm && CtaN == 128)))) {
          return 3;
        }
        return 4;
      }
      if (IsBsspNvf4) {
        if ((c_bits == 32 && d_bits == 32 && ((Is1Sm && CtaN == 128) ||
                                              (Is2Sm && CtaN == 192) ||
                                              (Is2Sm && CtaN == 256))) ||
            (c_bits == 16 && d_bits == 16 && ((Is1Sm && CtaN == 192) ||
                                              (Is2Sm && CtaN == 192) ||
                                              (Is2Sm && CtaN == 256))) ||
            (c_bits == 16 && d_bits == 4 && ((Is2Sm && CtaN == 192) ||
                                             (Is2Sm && CtaN == 256)))) {
          return 3;
        }
        if ((c_bits == 32 && d_bits == 32 && CtaN == 256 && CtaK == 512 ) ||
            (c_bits == 16 && d_bits == 4 && ((Is1Sm && CtaN == 128) ||
                                             (Is1Sm && CtaN == 192) ||
                                             (Is1Sm && CtaN == 256) ||
                                             (Is2Sm && CtaN == 128)))) {
          return 2;
        }
        return 4;
      }
      return 4;
    }
  }();

  constexpr int StagesD = cute::min(EpiTiles, ExpectedStagesD);
  constexpr int StagesC = cute::min(EpiTiles, ExpectedStagesC);

  using DispatchPolicy = Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmaStore>;
  return DispatchPolicy{};
}

} // namespace sparse

/*
 * Returns the TMEM_LOAD copy op to be used for the epilogue
 * Returned TMEM_LOAD op is such that the thread-value ownership matches the widest available
 * smem storage vectorization, subject to the constraints of data types and gmem layout
 * Selected op also maximizes the TMEM_LOAD shape in order to minimize TMEM_LOADs issued,
 * subject to the constraint of the provided per-warp tmem subpartition shape
**/
template<
  class GmemStrideTypeD,
  class ElementAccumulator,
  class ElementD,
  class TmemShape_MN,
  bool IsBlockScaleSupported
>
constexpr auto
sm100_get_tmem_load_op() {
  using namespace cute;

  // Number of datapaths (dp) available in this warp's tmem subpartition.
  // If only 16dp are available then we must use 16dp TMEM_LOAD variants
  // otherwise we prefer 32dp variants as those have higher throughput

  // For those fused patterns which have RowReduction or RowBroadcast
  // 16dp tmem load op can effectively reduce the usage of registers & shuffle instrs
  // Compared to TMEM_LOAD throughput, it's more critical
  constexpr int num_dp = size<0>(TmemShape_MN{});
  static_assert(num_dp == 16 || num_dp == 32, "Unsupported tmem datapath count");

  // Number of columns in this tmem subpartition, in bits
  // Used to select the widest cross variant TMEM_LOAD available
  constexpr int num_col_bits = size<1>(TmemShape_MN{}) * sizeof_bits_v<ElementAccumulator>;

  // Layout information, determines max available smem store vectorization
  // For M-major layouts we tend to target stmatrix_t (UMMA stores tmem accumulator in N-Major)
  constexpr bool is_m_major = cutlass::detail::is_major<0>(GmemStrideTypeD{});
  constexpr bool is_n_major = cutlass::detail::is_major<1>(GmemStrideTypeD{});
  static_assert(is_m_major || is_n_major, "Unsupported gmem layout");

  // dispatch on data types as this determines the correspondence
  // between TMEM_LOAD thread-bit ownership patterns and logical values
  if constexpr (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 32) {
    if constexpr (num_dp == 16) {
      if constexpr (is_m_major) {
        return TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, num_col_bits>(); // 32b stores to smem
      }
      else {
        return TMEM::op_repeater<SM100_TMEM_LOAD_16dp128b1x, num_col_bits>(); // stmatrix_n
        // return TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, num_col_bits>(); // 64b stores to smem
        // return TMEM::op_repeater<SM100_TMEM_LOAD_16dp32b1x, num_col_bits>(); // 128b stores to smem
      }
    }
    else {
      return TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x, num_col_bits>(); // 32b or 128b stores to smem
    }
  }

  else if constexpr (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 16) {
    if constexpr (num_dp == 16) {
      if constexpr (is_m_major) {
        return TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, num_col_bits>(); // stmatrix_t
      }
      else {
        return TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, num_col_bits>(); // stmatrix_n
        // return TMEM::op_repeater<SM100_TMEM_LOAD_16dp32b1x, num_col_bits>(); // 128b stores to smem
      }
    }
    else {
      if constexpr (is_m_major) {
        return TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, num_col_bits>(); // stmatrix_t
      }
      else {
        return TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x, num_col_bits>(); // 128b stores to smem
      }
    }
  }

  // For int8 kernels where accumulation is 32b but result store may be back to int8
  else if constexpr (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 8) {
    if constexpr (num_dp == 16) {
      if constexpr (is_m_major) {
        return TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, num_col_bits>(); // stmatrix_t m16n8
      }
      else {
        // return TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, num_col_bits>(); // 16b stores to smem
        return TMEM::op_repeater<SM100_TMEM_LOAD_16dp32b1x, num_col_bits>(); // 128b stores to smem
      }
    }
    else {
      // To use the HW instruction to find amax along the row/column of acc, the TMEM_LOAD pattern needs to be 32dp32bit.
        return TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x, num_col_bits>(); // 128b stores to smem
    }
  }

  // For 16b accumulation we use pack16b TMEM_LOAD variants as UMMA stores these values sparsely in tmem
  else if constexpr (sizeof_bits_v<ElementAccumulator> == 16 && sizeof_bits_v<ElementD> == 16) {
    if constexpr (num_dp == 16) {
      if constexpr (is_m_major) {
        return TMEM::op_repeater<SM100_TMEM_LOAD_16dp128b1x_16b, num_col_bits>(); // stmatrix_t
      }
      else {
        return TMEM::op_repeater<SM100_TMEM_LOAD_16dp128b1x_16b, num_col_bits>(); // stmatrix_n
        // return TMEM::op_repeater<SM100_TMEM_LOAD_16dp32b1x_16b, num_col_bits>(); // 128b stores to smem
      }
    }
    else {
      if constexpr (is_m_major) {
        return TMEM::op_repeater<SM100_TMEM_LOAD_16dp128b1x_16b, num_col_bits>(); // stmatrix_t
      }
      else {
        return TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x_16b, num_col_bits>(); // 128b stores to smem
      }
    }
  }
  // For complex TF32 kernels
  else if constexpr (sizeof_bits_v<ElementAccumulator> == 64 && sizeof_bits_v<ElementD> == 64) {
    if constexpr (num_dp == 16) {
      return TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, num_col_bits/2>();
    }
    else {
      return TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x, num_col_bits/2>();
    }
  }
  // For narrow precision output
  else if constexpr (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 6) {
    static_assert(num_dp == 32);
    return TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x, num_col_bits>();
  }
  else if constexpr (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 4) {
    static_assert(num_dp == 32);
    return TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x, num_col_bits>();
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAccumulator, ElementD>, "Unsupported data types");
  }
}

// Selects the largest vectorized smem store atom available
// subject to constraint of gmem layout and chosen TMEM_LOAD's thread-value ownership
template <class GmemStrideTypeD, class ElementD, class ElementAccumulator, class AccLoadOp>
constexpr auto
sm100_get_smem_store_op() {
  using namespace cute;

  [[maybe_unused]] constexpr bool is_m_major = cutlass::detail::is_major<0>(GmemStrideTypeD{});
  [[maybe_unused]] constexpr bool is_n_major = cutlass::detail::is_major<1>(GmemStrideTypeD{});
  static_assert(is_m_major || is_n_major, "Unsupported gmem layout");

  // Check for TMEM_LOAD layouts that match the thread-value ownership pattern of stmatrix
  constexpr bool use_stmatrix_m8n8_4x =
    (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 32 && is_n_major &&
      ( cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b2x>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b4x>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b8x>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b16x> ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b32x> ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b64x>   )    ) ||
    (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 16 &&
      ( cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b2x>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b4x>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b8x>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b16x> ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b32x>   )    ) ||
    (sizeof_bits_v<ElementAccumulator> == 16 && sizeof_bits_v<ElementD> == 16 &&
      ( cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b2x_16b>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b4x_16b>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b8x_16b>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b16x_16b> ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b32x_16b> ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b64x_16b>   ));
  [[maybe_unused]] constexpr bool use_stmatrix_m16n8_4x =
    (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 8 && is_m_major &&
      ( cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b4x>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b8x>  ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b16x> ||
        cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b32x>   )    );

  // 1x TMEM_LOAD doesn't have enough values to use largest stmatrix variants
  [[maybe_unused]]  constexpr bool use_stmatrix_m8n8_2x =
    (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 32 && is_n_major &&
      cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b1x>           ) ||
    (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 16 &&
      cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b1x>           ) ||
    (sizeof_bits_v<ElementAccumulator> == 16 && sizeof_bits_v<ElementD> == 16 &&
      cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp128b1x_16b>       );
  [[maybe_unused]] constexpr bool use_stmatrix_m16n8_2x =
    (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 8 && is_m_major &&
      cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b2x>           );
  [[maybe_unused]] constexpr bool use_stmatrix_m16n8_1x =
    (sizeof_bits_v<ElementAccumulator> == 32 && sizeof_bits_v<ElementD> == 8 && is_m_major &&
      cute::is_same_v<AccLoadOp, SM100_TMEM_LOAD_16dp256b1x>           );

  if constexpr (use_stmatrix_m8n8_4x) {
    if constexpr (is_n_major) {
      return SM90_U32x4_STSM_N{};
    }
    else if constexpr (is_m_major) {
      return SM90_U16x8_STSM_T{};
    }
  }
  else if constexpr (use_stmatrix_m8n8_2x) {
    if constexpr (is_n_major) {
      return SM90_U32x2_STSM_N{};
    }
    else if constexpr (is_m_major) {
      return SM90_U16x4_STSM_T{};
    }
  }
  else if constexpr (use_stmatrix_m16n8_4x) {
    return SM100_U8x16_STSM_T{};
  }
  else if constexpr (use_stmatrix_m16n8_2x) {
    return SM100_U8x8_STSM_T{};
  }
  else if constexpr (use_stmatrix_m16n8_1x) {
    return SM100_U8x4_STSM_T{};
  }
  else {
    // auto-vectorizing store
    return AutoVectorizingCopyWithAssumedAlignment<128>{};
  }
}



// Selects the largest vectorized smem load atom available
// subject to constraint of gmem layout and chosen TMEM_LOAD's thread-value ownership
template <class GmemStrideTypeC, class ElementC, class ElementAccumulator, class AccLoadOp>
constexpr auto
sm100_get_smem_load_op() {
  using namespace cute;

  // Reuse the logic from smem store selector
  using SmemStoreOp = decltype(sm100_get_smem_store_op<
      GmemStrideTypeC, ElementC, ElementAccumulator, AccLoadOp>());

  if constexpr (cute::is_same_v<SmemStoreOp, SM90_U32x4_STSM_N>) {
    return SM75_U32x4_LDSM_N{};
  }
  else if constexpr (cute::is_same_v<SmemStoreOp, SM90_U16x8_STSM_T>) {
    return SM75_U16x8_LDSM_T{};
  }
  else if constexpr (cute::is_same_v<SmemStoreOp, SM90_U32x2_STSM_N>) {
    return SM75_U32x2_LDSM_N{};
  }
  else if constexpr (cute::is_same_v<SmemStoreOp, SM90_U16x4_STSM_T>) {
    return SM75_U16x4_LDSM_T{};
  }
  else if constexpr (cute::is_same_v<SmemStoreOp, SM100_U8x16_STSM_T>) {
    return SM100_U8x16_LDSM_T{};
  }
  else if constexpr (cute::is_same_v<SmemStoreOp, SM100_U8x8_STSM_T>) {
    return SM100_U8x8_LDSM_T{};
  }
  else {
    // auto-vectorizing load
    return AutoVectorizingCopyWithAssumedAlignment{};
  }
}

// aux fusion callbacks builder for sm100 tma epilogue
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class FusionOp,
  class CtaTileShape_MNK,
  class EpilogueTile_MN,
  class ElementAccumulator,
  class AccLoadOp
>
struct CallbacksBuilder<
  Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  FusionOp,
  CtaTileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  AccLoadOp,
  cute::enable_if_t<(FusionOp::IsAuxOutSupported ^ FusionOp::IsAuxInSupported) // only one aux tensor
              && not cute::is_subbyte_v<typename FusionOp::ElementAux>>
> {
  using GmemStrideTypeAux = gemm::TagToStrideC_t<typename FusionOp::GmemLayoutTagAux>;
  using SmemLayoutAtomAux = decltype(detail::sm100_get_epilogue_smem_swizzle_layout_atom<
    GmemStrideTypeAux, typename FusionOp::ElementAux, EpilogueTile_MN>());
  using CopyOpR2S = decltype(detail::sm100_get_smem_store_op<
    GmemStrideTypeAux, typename FusionOp::ElementAux, ElementAccumulator, AccLoadOp>());
  using CopyOpS2R = decltype(detail::sm100_get_smem_load_op<
    GmemStrideTypeAux, typename FusionOp::ElementAux, ElementAccumulator, AccLoadOp>());
  using SmemCopyOpAux = cute::conditional_t<FusionOp::IsAuxOutSupported, CopyOpR2S, CopyOpS2R>;

  using Callbacks = fusion::FusionCallbacks<
    Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOp, CtaTileShape_MNK, EpilogueTile_MN,
    SmemLayoutAtomAux, SmemCopyOpAux
  >;
};

// ptr array aux fusion callbacks builder for sm100 tma epilogue
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class FusionOp,
  class CtaTileShape_MNK,
  class EpilogueTile_MN,
  class ElementAccumulator,
  class AccLoadOp
>
struct CallbacksBuilder<
  Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  FusionOp,
  CtaTileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  AccLoadOp,
  cute::enable_if_t<(FusionOp::IsAuxOutSupported ^ FusionOp::IsAuxInSupported) // only one aux tensor
              && not cute::is_subbyte_v<typename FusionOp::ElementAux>>
> {
  using GmemStrideTypeAux = gemm::TagToStrideC_t<typename FusionOp::GmemLayoutTagAux>;
  using SmemLayoutAtomAux = decltype(detail::sm100_get_epilogue_smem_swizzle_layout_atom<
    GmemStrideTypeAux, typename FusionOp::ElementAux, EpilogueTile_MN>());
  using CopyOpR2S = decltype(detail::sm100_get_smem_store_op<
    GmemStrideTypeAux, typename FusionOp::ElementAux, ElementAccumulator, AccLoadOp>());
  using CopyOpS2R = decltype(detail::sm100_get_smem_load_op<
    GmemStrideTypeAux, typename FusionOp::ElementAux, ElementAccumulator, AccLoadOp>());
  using SmemCopyOpAux = cute::conditional_t<FusionOp::IsAuxOutSupported, CopyOpR2S, CopyOpS2R>;

  using Callbacks = fusion::FusionCallbacks<
    Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOp, CtaTileShape_MNK, EpilogueTile_MN,
    SmemLayoutAtomAux, SmemCopyOpAux
  >;
};

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class FusionOp,
  class CtaTileShape_MNK,
  class EpilogueTile_MN,
  class ElementAccumulator,
  class AccLoadOp
>
struct CallbacksBuilder<
  Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  FusionOp,
  CtaTileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  AccLoadOp,
  cute::enable_if_t<(FusionOp::IsAuxOutSupported ^ FusionOp::IsAuxInSupported) // only one aux tensor
              && sizeof_bits_v<typename FusionOp::ElementAux> == 1>
> {
  using Callbacks = fusion::FusionCallbacks<
    Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOp, CtaTileShape_MNK, EpilogueTile_MN,
    Layout<_1,_0>, DefaultCopy // aux bit tensor doesn't use smem
  >;
};

// aux fusion callbacks builder for sm100 direct store epilogue
template <
  class FusionOp,
  class TileShape_MNK,
  class EpilogueTile_MN,
  class AccLoadOp,
  class ElementAccumulator
>
struct CallbacksBuilder<
  Sm100NoSmemWarpSpecialized,
  FusionOp,
  TileShape_MNK,
  EpilogueTile_MN,
  ElementAccumulator,
  AccLoadOp,
  cute::enable_if_t<(FusionOp::IsAuxOutSupported ^ FusionOp::IsAuxInSupported)> // only one aux tensor
> {
  using Callbacks = fusion::FusionCallbacks<
    Sm100NoSmemWarpSpecialized, FusionOp, TileShape_MNK, EpilogueTile_MN,
    Layout<_1,_0>, DefaultCopy // aux tensor doesn't use tma
  >;
};

// Attempts to compute a reasonably performant epilogue tile or allows the user to provide one.
template<
  class OpClass,
  class CtaTileShape_MNK,
  class EpilogueTileType,
  class TmemWarpShape_MN,
  class ElementC_,
  class GmemStrideTypeC,
  class ElementD,
  class GmemStrideTypeD,
  bool IsPerColScaleSupported
>
static constexpr auto
sm100_dense_compute_tile_shape_or_override() {
  using namespace cute;
  static_assert(!cute::is_same_v<OpClass, arch::OpClassSparseTensorOp> && !cute::is_same_v<OpClass, arch::OpClassBlockScaledSparseTensorOp>);

  constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD, ElementC_>;
  
  if constexpr (is_same_v<OpClass, arch::OpClassBlockScaledTensorOp> && 
                is_same_v<EpilogueTileType, EpilogueTileAuto> && 
                size<1>(CtaTileShape_MNK{}) == 256) {
    constexpr int CtaM = size<0>(CtaTileShape_MNK{});
    constexpr int WarpM = size<0>(TmemWarpShape_MN{});
    constexpr int DpFull = 32;
    constexpr int M = cute::min(CtaM, DpFull * WarpM); // target 32dp tmem load
    // Note: 
    // Set Epi_Tile_N to 128 support OverlappingAccum for the largest tile.
    // This is a general workable epi_tile_N which does not promise best perf.
    return make_tile(Int<M>{}, Int<128>{}); 
  }
  else if constexpr (is_same_v<EpilogueTileType, EpilogueTileAuto>) {
    constexpr int CtaM = size<0>(CtaTileShape_MNK{});
    constexpr int CtaN = size<1>(CtaTileShape_MNK{});
    constexpr int WarpM = size<0>(TmemWarpShape_MN{});
    constexpr int WarpN = size<1>(TmemWarpShape_MN{});
    constexpr int MaxBits = cute::max(sizeof_bits_v<ElementC>, sizeof_bits_v<ElementD>);

    constexpr int DpFull = 32; // tmem datapaths in 1 subpartition
    constexpr int M = cute::min(CtaM, DpFull * WarpM); // target 32dp tmem load
    constexpr int N_perf = [&]() constexpr { // Known subtile sizes tested for perf
      // Epilogues w/o residual load are less sensitive to smem allocation
      // Target a fixed amount of compute per epilogue iteration
      if (DisableSource) {
        if (MaxBits == 4) {
          // Make epilogue tile larger to reduce the epilogue iterations.
          // 64 is the experimental value. It will minimize epilogue iterations but keep the number of A/B buffers the same.
          constexpr int ComputeElts = 8192;
          return ComputeElts / M;
        }
        constexpr int ComputeElts = 4096;
        return ComputeElts / M;
      }
      // Epilogues w/ residual load are more sensitive to smem allocation
      // Target optimal smem distribution between epilogue+mainloop based on datatype+tilesize
      else {
        if (MaxBits == 32) {
          return (CtaM > 64 && CtaN <= 128) ? 16 : 32;
        }
        // Per-column scaling is high register pressure, reduce tile to prevent spills
        else if (IsPerColScaleSupported) {
          return 32;
        }
        else if (MaxBits == 16) {
          return (CtaN <= 128) ? 32 : 64;
        }
        else {
          return 64;
        }
      }
    }();
    constexpr int N_min_C = (DisableSource || detail::is_m_major<GmemStrideTypeC>()) ? 8 * WarpN
                              : (sizeof_bits_v<ElementC> == 6) ? 128 * WarpN // TMA store only supports SW128B for FP6 data type
                                                              : 128 / sizeof_bits_v<ElementC> * WarpN;
    constexpr int N_min_D = (detail::is_m_major<GmemStrideTypeD>()) ? 8 * WarpN
                              : (sizeof_bits_v<ElementD> == 6) ? 128 * WarpN // TMA store only supports SW128B for FP6 data type
                                                              : 128 / sizeof_bits_v<ElementD> * WarpN;
    constexpr int N = cute::min(CtaN, cute::max(N_perf, N_min_C, N_min_D));
    static_assert(CtaN >= N_min_C && CtaN >= N_min_D, "CTA tile too small");

    // stride by tmem warp layout and return a by-mode tiler
    auto tile_m = Layout<Int<M>>{};
    auto tile_n = Layout<Shape <Int<N / WarpN>,Int<        WarpN>>,
                        Stride<Int<         1>,Int<CtaN / WarpN>>>{};

    return make_tile(tile_m, coalesce(tile_n));
  }
  else {
    static_assert(cute::is_tuple<EpilogueTileType>::value && not is_layout<EpilogueTileType>::value,
                    "EpilogueTile must be a cute::Tile or cute::Shape");

    EpilogueTileType epi_tile;
    constexpr int M = size<0>(shape(epi_tile));
    constexpr int N = size<1>(shape(epi_tile));
    static_assert(N % 8 == 0, "Unsupported tile shape");

    return epi_tile;
  }
}

template<
  bool Is2SmMma,
  class MmaTileShape_MNK
>
static constexpr auto
sm100_tmem_warps() {
  if constexpr (Is2SmMma && size<0>(MmaTileShape_MNK{}) == 128) {
    return Shape<_2,_2>{};
  }
  else {
    return Shape<_4,_1>{};
  }
}

template<
  bool Is2SmMma,
  class MmaTileShape_MNK
>
static constexpr auto
sm100_cta_tile_shape() {
  if constexpr (Is2SmMma) { // 2x1 threadblock shape
    auto [mma_tile_m, mma_tile_n, mma_tile_k] = MmaTileShape_MNK{};
    auto cta_tile_m = reverse(shape_div(reverse(mma_tile_m), _2{})); // first MmaTile_M/2 elements, preserve multimode
    return make_shape(cta_tile_m, mma_tile_n, mma_tile_k);
  }
  else { // 1x1 threadblock shape
    return MmaTileShape_MNK{};
  }
}

template<
  class EpilogueScheduleType,
  class ElementC_,
  class ElementD,
  int EpiTiles,
  int FragmentSize
>
static constexpr auto
sm100_dense_dispatch_policy() {
  // 8b residuals load fast and consume little smem, so the perf cost of waiting on stores to finish outweighs the cost of extra allocation
  constexpr bool ReuseSmem = sizeof_bits_v<ElementC_> > 8;
  // TMA store delay performs worse with residual loads
  constexpr bool DelayTmaStore = is_void_v<ElementC_>;

  constexpr int StagesD = cute::min(EpiTiles, 2);
  constexpr int StagesC = ReuseSmem ? cute::max(cute::min(EpiTiles, 4), StagesD+1)
                                    : cute::min(EpiTiles, 4);

  if constexpr (is_base_of_v<PtrArrayNoSmemWarpSpecialized1Sm, EpilogueScheduleType> ||
                is_base_of_v<PtrArrayNoSmemWarpSpecialized2Sm, EpilogueScheduleType>) {
    return Sm100PtrArrayNoSmemWarpSpecialized{};
  }
  else if constexpr (is_base_of_v<NoSmemWarpSpecialized1Sm, EpilogueScheduleType> || is_base_of_v<NoSmemWarpSpecialized2Sm, EpilogueScheduleType>) {
    return Sm100NoSmemWarpSpecialized{};
  }
  else if constexpr (is_same_v<EpilogueScheduleType, PtrArrayTmaWarpSpecialized1Sm> ||
                     is_same_v<EpilogueScheduleType, PtrArrayTmaWarpSpecialized2Sm>) {
    constexpr bool DelayTmaStore_ = false; // TMA store delay complicates tensormap updates for Ptr-Array GEMMs
    return Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmaStore_>{};
  }
  else {
    return Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmem, DelayTmaStore>{};
  }
}

// Helper for building TMA warp-specialized collective epilogues, specialized by
// the fusion operation performed and the dispatch policy to use.
template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD_,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class FusionOpOrCallbacks
>
struct Sm100TmaBuilderImpl {
private:
  static constexpr bool Is1SmMma = is_base_of_v<TmaWarpSpecialized1Sm, Schedule>;
  static constexpr bool Is2SmMma = is_base_of_v<TmaWarpSpecialized2Sm, Schedule>;
  static_assert(Is1SmMma ^ Is2SmMma, "unsupported schedule");
  static_assert(not (Is2SmMma && size<0>(ClusterShape_MNK{}) % 2 == 1), "schedule + cluster mismatch");
  // C/D should meet TMA alignment requirement if not void
  static_assert(detail::is_aligned<ElementC_, AlignmentC, ElementD_, AlignmentD>(),
                "C/D Should meet TMA alignment requirement\n");

  static constexpr bool DisableDestination = cute::is_void_v<ElementD_>;
  using ElementD = cute::conditional_t<DisableDestination,fusion::get_element_aux_t<FusionOpOrCallbacks>,ElementD_>; // prevents void ref breakages

  // Passing void C disables source load + smem allocation
  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource,ElementD,ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<DisableSource,GmemLayoutTagD,GmemLayoutTagC_>;

  using InternalSmemElementC = typename cutlass::detail::get_unpacked_element_type<ElementC>::type;
  using InternalSmemElementD = typename cutlass::detail::get_unpacked_element_type<ElementD>::type;

  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  // TMA builder allows for passing callbacks directly, which is either a fusion::FusionCallbacks
  // instance or a direct visitor implementation, e.g. fusion::Sm90LinearCombination
  static constexpr bool IsTaggedFusionOp = is_base_of_v<epilogue::fusion::FusionOperation, FusionOpOrCallbacks>;
  using FusionOp = conditional_t<IsTaggedFusionOp, FusionOpOrCallbacks, epilogue::fusion::FusionOperation>;

  static constexpr auto
  cta_tile_shape() {
    if constexpr (Is2SmMma) { // 2x1 threadblock shape
      auto [mma_tile_m, mma_tile_n, mma_tile_k] = MmaTileShape_MNK{};
      auto cta_tile_m = reverse(shape_div(reverse(mma_tile_m), _2{})); // first MmaTile_M/2 elements, preserve multimode
      return make_shape(cta_tile_m, mma_tile_n, mma_tile_k);
    }
    else { // 1x1 threadblock shape
      return MmaTileShape_MNK{};
    }
  }
  using CtaTileShape_MNK = decltype(cta_tile_shape());
  using TmemWarpShape_MN = decltype(detail::sm100_tmem_warps<Is2SmMma, MmaTileShape_MNK>());

  // Attempts to compute a reasonably performant epilogue tile or allows the user to provide one.
  static constexpr auto
  epilogue_tile() {
    using namespace cute;
    
    if constexpr (is_same_v<OpClass, arch::OpClassSparseTensorOp> ||
                  is_same_v<OpClass, arch::OpClassBlockScaledSparseTensorOp>) {
      return detail::sparse::sm100_sparse_compute_tile_shape_or_override<
                        CtaTileShape_MNK, EpilogueTileType, TmemWarpShape_MN,
                        ElementC_, GmemStrideTypeC, ElementD, GmemStrideTypeD, Schedule,
                        FusionOp>();
    }
    else {
      return sm100_dense_compute_tile_shape_or_override<
        OpClass, CtaTileShape_MNK, EpilogueTileType, TmemWarpShape_MN,
        ElementC_, GmemStrideTypeC, ElementD, GmemStrideTypeD, FusionOp::IsPerColScaleSupported>();
    }
  }
  using EpilogueTile_MN = decltype(epilogue_tile());

  using EpilogueTileShape_MN = decltype(product_each(shape(EpilogueTile_MN{})));
  static constexpr int EpiTiles = size(shape_div(take<0,2>(CtaTileShape_MNK{}), EpilogueTileShape_MN{}));
  static constexpr int FragmentSize = size(EpilogueTileShape_MN{}) / NumThreadsPerWarpGroup;

  using EpilogueWarpTileShape_MN = decltype(shape_div(EpilogueTileShape_MN{}, TmemWarpShape_MN{}));
  using AccLoadOp = decltype(detail::sm100_get_tmem_load_op<
      GmemStrideTypeD, ElementAccumulator, ElementD, EpilogueWarpTileShape_MN,
      FusionOp::IsBlockScaleSupported
      >());

  static constexpr auto
  dispatch_policy() {
    if constexpr (is_same_v<OpClass, arch::OpClassSparseTensorOp> ||
                  is_same_v<OpClass, arch::OpClassBlockScaledSparseTensorOp>) {
      return detail::sparse::sm100_sparse_get_tma_dispatch_policy<CtaTileShape_MNK, EpilogueTile_MN, ElementC_, ElementD, Schedule>();
    }
    else {
      return detail::sm100_dense_dispatch_policy<Schedule, ElementC_, ElementD, EpiTiles, FragmentSize>();
    }
  }

  static constexpr auto
  fusion_callbacks() {
    {
      return typename CallbacksBuilder<
                        decltype(dispatch_policy()),
                        FusionOpOrCallbacks,
                        CtaTileShape_MNK,
                        EpilogueTile_MN,
                        ElementAccumulator,
                        AccLoadOp
                      >::Callbacks({},{});
    }
  }

  static constexpr auto
  gmem_load_op() {
    if constexpr (detail::is_im2col_mode<GmemLayoutTagC>) {
      return SM90_TMA_LOAD_IM2COL{};
    }
    else {
      return SM90_TMA_LOAD{};
    } 
  }

  static constexpr auto
  gmem_store_op() {
    if constexpr (detail::is_im2col_mode<GmemLayoutTagD>) {
      return SM90_TMA_STORE_IM2COL{};
    }
    else {
      return SM90_TMA_STORE{};
    } 
  }

  static constexpr auto
  register_shuffle_op() {
    using namespace cute;

    [[maybe_unused]] constexpr bool is_m_major = cutlass::detail::is_major<0>(GmemStrideTypeD{});
    [[maybe_unused]] constexpr bool is_n_major = cutlass::detail::is_major<1>(GmemStrideTypeD{});
    static_assert(is_m_major || is_n_major, "Unsupported gmem layout");

    if constexpr (sizeof_bits_v<InternalSmemElementD> == 4 && is_m_major) {
      return SM50_Shuffle_U32_2x2Trans_XOR1{};
    }
    else {
      return AutoVectorizingCopyWithAssumedAlignment<128>{};
    }
  }

public:
  using CollectiveOp =
    cutlass::epilogue::collective::CollectiveEpilogue<
      decltype(dispatch_policy()),
      CtaTileShape_MNK,
      EpilogueTile_MN,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeD,
      decltype(fusion_callbacks()),
      AccLoadOp,
      decltype(gmem_load_op()),
      decltype(detail::sm100_get_epilogue_smem_swizzle_layout_atom<GmemStrideTypeC, InternalSmemElementC, EpilogueTile_MN>()),
      decltype(detail::sm100_get_smem_load_op<GmemStrideTypeC, InternalSmemElementC, ElementAccumulator, AccLoadOp>()),
      decltype(gmem_store_op()),
      decltype(detail::sm100_get_epilogue_smem_swizzle_layout_atom<GmemStrideTypeD, InternalSmemElementD, EpilogueTile_MN>()),
      decltype(detail::sm100_get_smem_store_op<GmemStrideTypeD, InternalSmemElementD, ElementAccumulator, AccLoadOp>()),
      decltype(register_shuffle_op())
    >;
};

template<
  class OpClass,
  class MmaTileShape_MNK,
  class EpilogueTileType, 
  class ElementAccumulator_,
  class ElementC,
  class ElementD,
  class Schedule,
  class GmemStrideTypeC,
  class GmemStrideTypeD,
  bool IsPerColScaleSupported,
  bool IsBlockScaleSupported
>
struct Sm100EpilogueDescriptor {
  using ElementAccumulator = ElementAccumulator_;

  static constexpr bool Is2SmMma = is_base_of_v<TmaWarpSpecialized2Sm, Schedule> || is_base_of_v<NoSmemWarpSpecialized2Sm, Schedule>;
  using CtaTileShape_MNK = decltype(sm100_cta_tile_shape<Is2SmMma, MmaTileShape_MNK>());
  using TileShape = CtaTileShape_MNK;

  using TmemWarpShape_MN = decltype(detail::sm100_tmem_warps<Is2SmMma, MmaTileShape_MNK>());

  using EpilogueTile = decltype(
    sm100_dense_compute_tile_shape_or_override<OpClass, CtaTileShape_MNK, EpilogueTileType,
      TmemWarpShape_MN, ElementC, GmemStrideTypeC, ElementD, GmemStrideTypeD, IsPerColScaleSupported>()
  );

  using EpilogueTileShape_MN = decltype(product_each(shape(EpilogueTile{})));
  static constexpr int EpiTiles = size(shape_div(take<0,2>(CtaTileShape_MNK{}), EpilogueTileShape_MN{}));
  static constexpr int FragmentSize = size(EpilogueTileShape_MN{}) / NumThreadsPerWarpGroup;

  using DispatchPolicy = decltype(sm100_dense_dispatch_policy<Schedule, ElementC, ElementD, EpiTiles, FragmentSize>());

  constexpr static int StagesC = DispatchPolicy::StagesC;
  constexpr static int StagesD = DispatchPolicy::StagesD;

  using EpilogueWarpTileShape_MN = decltype(shape_div(EpilogueTileShape_MN{}, TmemWarpShape_MN{}));
  using AccLoadOp = decltype(detail::sm100_get_tmem_load_op<
      GmemStrideTypeD, ElementAccumulator, ElementD, EpilogueWarpTileShape_MN,
      IsBlockScaleSupported
      >());
};

// Get Stride, SmemLayout, and CopyOpS2R for AuxLoad node
template<
  typename EpilogueDescriptor,
  typename StrideOrLayoutTag,
  typename ElementAux
>
struct Sm100AuxLoadDescriptor {
  constexpr static int Stages = EpilogueDescriptor::StagesC;
  using EpilogueTile = typename EpilogueDescriptor::EpilogueTile;
  using Element = ElementAux;
  using Stride = cutlass::detail::TagToStrideC_t<StrideOrLayoutTag>;

  using SmemLayoutAtom = decltype(detail::sm100_get_epilogue_smem_swizzle_layout_atom<
    Stride, ElementAux, EpilogueTile>());

  using CopyOpS2R = decltype(detail::sm100_get_smem_load_op<
    Stride, ElementAux, typename EpilogueDescriptor::ElementAccumulator, typename EpilogueDescriptor::AccLoadOp>());
};

// Get Stride, SmemLayout, and CopyOpS2R for AuxStore node
template<
  typename EpilogueDescriptor,
  typename StrideOrLayoutTag,
  typename ElementAux
>
struct Sm100AuxStoreDescriptor {
  constexpr static int Stages = EpilogueDescriptor::StagesD;
  using EpilogueTile = typename EpilogueDescriptor::EpilogueTile;
  using Element = ElementAux;
  using Stride = cutlass::detail::TagToStrideC_t<StrideOrLayoutTag>;

  using SmemLayoutAtom = decltype(detail::sm100_get_epilogue_smem_swizzle_layout_atom<
    Stride, ElementAux, EpilogueTile>());

  using CopyOpR2S = decltype(detail::sm100_get_smem_store_op<
    Stride, ElementAux, typename EpilogueDescriptor::ElementAccumulator, typename EpilogueDescriptor::AccLoadOp>());
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////

// No smem builder
template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class EpilogueScheduleType,
  class FusionOpOrCallbacks
>
struct CollectiveBuilder<
    arch::Sm100,
    OpClass,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleType,
    FusionOpOrCallbacks,
    cute::enable_if_t<is_base_of_v<NoSmemWarpSpecialized1Sm, EpilogueScheduleType> ||
                      is_base_of_v<NoSmemWarpSpecialized2Sm, EpilogueScheduleType> >
> {
private:
  static_assert(cute::sizeof_bits_v<ElementD> != 6, "Output element requires TMA");

  static constexpr bool Is1SmMma = is_base_of_v<NoSmemWarpSpecialized1Sm, EpilogueScheduleType>;
  static constexpr bool Is2SmMma = is_base_of_v<NoSmemWarpSpecialized2Sm, EpilogueScheduleType>;
  static constexpr bool IsInterleavedComplex = is_complex<ElementAccumulator>::value;
  static constexpr bool IsFastF32Schedule = is_same_v<FastF32NoSmemWarpSpecialized1Sm, EpilogueScheduleType> || 
                                    is_same_v<FastF32NoSmemWarpSpecialized2Sm, EpilogueScheduleType> ||
                                    is_same_v<PtrArrayFastF32NoSmemWarpSpecialized1Sm, EpilogueScheduleType> ||
                                    is_same_v<PtrArrayFastF32NoSmemWarpSpecialized2Sm, EpilogueScheduleType>;
  static constexpr bool IsBlockwiseSchedule = is_same_v<BlockwiseNoSmemWarpSpecialized1Sm, EpilogueScheduleType> || 
                                    is_same_v<BlockwiseNoSmemWarpSpecialized2Sm, EpilogueScheduleType> ||
                                    is_same_v<PtrArrayBlockwiseNoSmemWarpSpecialized1Sm, EpilogueScheduleType> ||
                                    is_same_v<PtrArrayBlockwiseNoSmemWarpSpecialized2Sm, EpilogueScheduleType>;
  // Transform kernels - when dispatching to sm100 nosmem epilogue, go through the default path without EVT support.
  static constexpr bool IsTransformSchedule = IsInterleavedComplex || IsFastF32Schedule || IsBlockwiseSchedule;
  static_assert(Is1SmMma ^ Is2SmMma, "unsupported schedule");
  static_assert(not (Is2SmMma && size<0>(ClusterShape_MNK{}) % 2 == 1), "schedule + cluster mismatch");

  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD, ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<DisableSource, GmemLayoutTagD, GmemLayoutTagC_>;
  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  static constexpr bool IsTaggedFusionOp = is_base_of_v<epilogue::fusion::FusionOperation, FusionOpOrCallbacks>;
  using FusionOp = conditional_t<IsTaggedFusionOp, FusionOpOrCallbacks, epilogue::fusion::FusionOperation>;

  static constexpr auto
  cta_tile_shape() {
    if constexpr (Is2SmMma) { // 2x1 threadblock shape
      auto [mma_tile_m, mma_tile_n, mma_tile_k] = MmaTileShape_MNK{};
      auto cta_tile_m = reverse(shape_div(reverse(mma_tile_m), _2{})); // first MmaTile_M/2 elements, preserve multimode
      return make_shape(cta_tile_m, mma_tile_n, mma_tile_k);
    }
    else { // 1x1 threadblock shape
      return MmaTileShape_MNK{};
    }
  }
  using CtaTileShape_MNK = decltype(cta_tile_shape());
  using TmemWarpShape_MN = decltype(detail::sm100_tmem_warps<Is2SmMma, MmaTileShape_MNK>());

  static constexpr auto
  epilogue_tile() {
    using namespace cute;
    if constexpr (not is_same_v<EpilogueTileType, EpilogueTileAuto>) {
      static_assert(is_tuple_v<EpilogueTileType>, "Shape or Tile");
      return EpilogueTileType{};
    }
    else if constexpr (is_same_v<OpClass,arch::OpClassBlockScaledTensorOp> || not IsTransformSchedule) {
      // Save register usage for sm103 blockscaled kernels and sm100 cpasync kernels
      // to avoid register spilling.
      constexpr int EpiM = size<0>(CtaTileShape_MNK{});
      constexpr int EpiN = cute::min(_64{}, size<1>(CtaTileShape_MNK{}));
      return Shape<Int<EpiM>, Int<EpiN>>{};
    }
    else {
      return take<0,2>(CtaTileShape_MNK{});
    }
  }
  using EpilogueTile = decltype(epilogue_tile());

  using EpilogueWarpTileShape_MN = decltype(shape_div(EpilogueTile{}, TmemWarpShape_MN{}));
  using AccLoadOp = decltype(detail::sm100_get_tmem_load_op<
      GmemStrideTypeD, ElementAccumulator, ElementD, EpilogueWarpTileShape_MN,
      FusionOp::IsBlockScaleSupported
      >());
  static constexpr int FragmentSize = size(EpilogueTile{}) / NumThreadsPerWarpGroup;

  using EpilogueTileShape_MN = decltype(product_each(shape(EpilogueTile{})));
  static constexpr int EpiTiles = size(shape_div(take<0,2>(CtaTileShape_MNK{}), EpilogueTileShape_MN{}));

  using DispatchPolicy = decltype(detail::sm100_dense_dispatch_policy<EpilogueScheduleType, ElementC_, ElementD, EpiTiles, FragmentSize>());

  static constexpr auto
  fusion_callbacks() {
    constexpr thread::ScaleType::Kind ScaleType =
      DisableSource ? thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;
    if constexpr (IsDefaultFusionOp<FusionOp>::value &&\
                  not is_same_v<OpClass, arch::OpClassBlockScaledTensorOp> && \
                 (IsTransformSchedule || \
                  is_same_v<EpilogueScheduleType, PtrArrayNoSmemWarpSpecialized1Sm> || \
                  is_same_v<EpilogueScheduleType, PtrArrayNoSmemWarpSpecialized2Sm>)
                 ) {
      // Legacy codepath using thread::LinearCombination, do not expect this to be stable
      return thread::LinearCombination<
                ElementD, 1, ElementAccumulator, ElementCompute, ScaleType, FusionOp::RoundStyle, ElementC>({});
    }
    else {
      return typename detail::CallbacksBuilder<
                DispatchPolicy,
                FusionOpOrCallbacks,
                CtaTileShape_MNK,
                EpilogueTile,
                ElementAccumulator,
                AccLoadOp
              >::Callbacks({},{});
    }
  }

public:
  using CollectiveOp = 
    cutlass::epilogue::collective::CollectiveEpilogue<
      DispatchPolicy,
      EpilogueTile,
      ElementC_,
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      decltype(fusion_callbacks()),
      AccLoadOp,
      Int<AlignmentC>,
      Int<AlignmentD>
    >;
};

// TMA epilogue builder
template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class EpilogueScheduleType,
  class FusionOp
>
struct CollectiveBuilder<
    arch::Sm100,
    OpClass,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleType,
    FusionOp,
    cute::enable_if_t<
      // Only support TensorOp kernels
      not cute::is_same_v<OpClass, arch::OpClassSimt> &&
      (cute::is_base_of_v<TmaWarpSpecialized1Sm, EpilogueScheduleType> ||
       cute::is_base_of_v<TmaWarpSpecialized2Sm, EpilogueScheduleType>)
    >
>
 {
public:
  using CollectiveOp =
    typename detail::Sm100TmaBuilderImpl<
      OpClass,
      MmaTileShape_MNK,
      ClusterShape_MNK,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      GmemLayoutTagC,
      AlignmentC,
      ElementD,
      GmemLayoutTagD,
      AlignmentD,
      EpilogueScheduleType,
      FusionOp
    >::CollectiveOp;
};

// Auto epilogue builder for TensorOp kernels
template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOp
>
struct CollectiveBuilder<
    arch::Sm100,
    OpClass,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleAuto,
    FusionOp,
    // only for TensorOp kernels
    cute::enable_if_t<not cute::is_same_v<OpClass, arch::OpClassSimt>>
>
 {
private:
  static constexpr bool
  is_2sm() {
    using namespace cute;
    constexpr int MmaTileM = size<0>(MmaTileShape_MNK{});
    constexpr int ClusterM = size<0>(ClusterShape_MNK{});
    constexpr bool StaticClusterM = is_static_v<decltype(get<0>(ClusterShape_MNK{}))>;
    constexpr bool EvenClusterM = StaticClusterM && ClusterM % 2 == 0;
    if constexpr (not EvenClusterM) {
      return false;
    }
    else if constexpr (is_same_v<OpClass,arch::OpClassBlockScaledTensorOp>) {
      return MmaTileM == 256;
    }
    else {
      return MmaTileM == 256 || MmaTileM == 128;
    }
  }
  using EpilogueSchedule = cute::conditional_t<is_2sm(), TmaWarpSpecialized2Sm, TmaWarpSpecialized1Sm>;

public:
  static_assert(cute::is_same_v<EpilogueTileType, EpilogueTileAuto>, "Don't specify epilogue tile with auto schedule");
  using CollectiveOp =
    typename CollectiveBuilder<
      arch::Sm100,
      OpClass,
      MmaTileShape_MNK,
      ClusterShape_MNK,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      GmemLayoutTagC,
      AlignmentC,
      ElementD,
      GmemLayoutTagD,
      AlignmentD,
      EpilogueSchedule,
      FusionOp
    >::CollectiveOp;
};

template <
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class EpilogueScheduleType,
  class FusionOp
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassSimt,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleType,
    FusionOp,
    cute::enable_if_t<
      cute::is_same_v<EpilogueScheduleType, EpilogueSimtVectorized> ||
      cute::is_same_v<EpilogueScheduleType, EpiloguePtrArraySimtVectorized> ||
      cute::is_same_v<EpilogueScheduleType, EpilogueScheduleAuto> >> {
  using CtaTileShape_MNK = MmaTileShape_MNK; // cluster MMA not supported

  // Passing void C disables source load
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,
      ElementD, ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,
      GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = cute::is_void_v<ElementC_> ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  using ThreadOp = cute::conditional_t<
    IsDefaultFusionOp<FusionOp>::value,
    thread::LinearCombination<
      ElementD, AlignmentD, ElementAccumulator, ElementCompute,
      ScaleType, FloatRoundStyle::round_to_nearest, ElementC>
    ,
    thread::LinearCombinationBiasElementwise<
      ElementC, ElementAccumulator, ElementCompute, ElementD, ElementD, AlignmentD,
      typename FusionOp::ActivationFn, cutlass::plus<ElementCompute>,
      false, typename FusionOp::ElementBias>
  >;
  static_assert(not (cute::is_same_v<EpilogueScheduleType, EpiloguePtrArraySimtVectorized> && not IsDefaultFusionOp<FusionOp>::value), "unsupported schedule + fusion");

  using WarpShape_MNK = decltype(cutlass::gemm::collective::detail::sm100_simt_f32_warp_shape_mnk_selector<CtaTileShape_MNK>());
  static constexpr int ThreadCount = cute::size(WarpShape_MNK{}) * NumThreadsPerWarp;
  static constexpr int WarpShape_M = cute::size<0>(WarpShape_MNK{});
  static constexpr int WarpShape_N = cute::size<1>(WarpShape_MNK{});

  // For 32 threads in 1 warp, we use [8 x 4] thread layouts and each thread will hold [4 x 4] accumulator value layouts.
  // Then totally each warp will hold [32 x 16] accumulator value layouts.
  // We separate the whole epilogue calculation to multi steps,
  // each step will calculate 1x [32 x 16] for each warp to reduce register pressure (mainly for C register allocation for beta 1!= 0 case).
  // So EpiTileM = WarpShape_M * 32 and EpiTileN = WarpShape_N * 16.
  using EpiTileM = Int<WarpShape_M * 32>;
  using EpiTileN = Int<WarpShape_N * 16>;

  using SmemLayout = cute::conditional_t<cutlass::detail::is_major<0>(GmemStrideTypeD{}),
                                         cute::Layout<cute::Shape<EpiTileM, EpiTileN>, cute::Stride<_1, EpiTileM>>,
                                         cute::Layout<cute::Shape<EpiTileM, EpiTileN>, cute::Stride<EpiTileN, _1>>>;

  using CopyAtomR2S = Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccumulator>;

  using CopyAtomS2R = Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentD * sizeof_bits_v<ElementAccumulator>>, ElementAccumulator>;

  using TiledCopyS2R = decltype(
        cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
          CopyAtomS2R, ThreadCount, AlignmentD, GmemStrideTypeD, EpiTileM, EpiTileN>());

  using Schedule = cute::conditional_t<is_same_v<EpilogueScheduleType, EpilogueScheduleAuto>,
                                       EpilogueSimtVectorized,
                                       EpilogueScheduleType>;
  using CopyAtomR2G = Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentD * sizeof_bits_v<ElementD>>, ElementD>;
  using CollectiveOp = cutlass::epilogue::collective::Epilogue<
      GmemStrideTypeC,
      GmemStrideTypeD,
      ThreadOp,
      SmemLayout,
      CopyAtomR2S,
      TiledCopyS2R,
      CopyAtomR2G,
      Schedule>;
};
///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective
