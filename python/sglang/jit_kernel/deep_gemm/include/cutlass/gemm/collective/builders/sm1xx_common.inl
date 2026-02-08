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

// Maps input element to umma element
template <class Element, bool IsF8F6F4 = true>
constexpr auto
sm1xx_kernel_input_element_to_mma_input_element() {
  if constexpr (cute::is_same_v<Element, float>) {
    return cutlass::tfloat32_t{};
  }
  else if constexpr (cute::is_same_v<Element, cutlass::float_e2m1_t> && IsF8F6F4) {
    return cutlass::detail::float_e2m1_unpacksmem_t{};
  }
  else if constexpr (cute::is_same_v<Element, cutlass::float_e3m2_t> && IsF8F6F4) {
    return cutlass::detail::float_e3m2_unpacksmem_t{};
  }
  else if constexpr (cute::is_same_v<Element, cutlass::float_e2m3_t> && IsF8F6F4) {
    return cutlass::detail::float_e2m3_unpacksmem_t{};
  }
  else if constexpr (cute::is_same_v<Element, cutlass::type_erased_dynamic_float4_t> && IsF8F6F4) {
    return cutlass::detail::type_erased_dynamic_float4_unpacksmem_t{};
  }
  else if constexpr (cute::is_same_v<Element, cutlass::type_erased_dynamic_float6_t> && IsF8F6F4) {
    return cutlass::detail::type_erased_dynamic_float6_unpacksmem_t{};
  }
  else {
    return Element{};
  }
}

// Maps 2.x A matrix layout tag to respective UMMA major mode enum
template <class Layout>
constexpr cute::UMMA::Major
tag_to_umma_major_A() {
  using LayoutA = cute::remove_pointer_t<Layout>;
  if constexpr (cute::is_same_v<LayoutA, cutlass::layout::RowMajor>) {
    return cute::UMMA::Major::K;
  }
  else if constexpr (cute::is_same_v<LayoutA, cutlass::layout::ColumnMajor>) {
    return cute::UMMA::Major::MN;
  }
  else if constexpr (cutlass::detail::is_major<0, LayoutA>()) {
    return cute::UMMA::Major::MN;
  }
  else if constexpr (cutlass::detail::is_major<1, LayoutA>()) {
    return cute::UMMA::Major::K;
  }
  else {
    static_assert(sizeof(LayoutA) == 0, "Invalid layout.");
  }
}

// Maps 2.x B matrix layout tag to respective UMMA major mode enum
template <class Layout>
constexpr cute::UMMA::Major
tag_to_umma_major_B() {
  using LayoutB = cute::remove_pointer_t<Layout>;
  if constexpr (cute::is_same_v<LayoutB, cutlass::layout::RowMajor>) {
    return cute::UMMA::Major::MN;
  }
  else if constexpr (cute::is_same_v<LayoutB, cutlass::layout::ColumnMajor>) {
    return cute::UMMA::Major::K;
  }
  else if constexpr (cutlass::detail::is_major<0, LayoutB>()) {
    return cute::UMMA::Major::MN;
  }
  else if constexpr (cutlass::detail::is_major<1, LayoutB>()) {
    return cute::UMMA::Major::K;
  }
  else {
    static_assert(sizeof(LayoutB) == 0, "Invalid layout.");
  }
}

template<class BuilderScheduleTag>
constexpr uint32_t find_vector_size() {
  if constexpr (cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized1SmNvf4Sm100> ||
                cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized2SmNvf4Sm100> ||
                cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100> ||
                cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100> ||
                cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecializedNvf4Sm120> ||
                cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecializedPingpongNvf4Sm120>
                || cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103>
                || cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103>
                || cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103DisablePrefetch>
                || cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103DisablePrefetch>
                || cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103TmaPrefetch>
                || cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103TmaPrefetch>
                || cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103>
                || cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103>
                || cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103TmaPrefetch>
                || cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103TmaPrefetch>
                || cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103DisablePrefetch>
                || cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103DisablePrefetch>
              ) {
    return 16;
  }
  else if constexpr (cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecialized1SmNvf4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecialized2SmNvf4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized1SmMxf4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized2SmMxf4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized1SmMxf8f6f4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized2SmMxf8f6f4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized1SmMxf4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized2SmMxf4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecializedNvf4Sm120>) {           
    return 32;
  }
  else if constexpr (cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecialized1SmMxf8f6f4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecialized2SmMxf8f6f4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecialized1SmMxf4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecialized2SmMxf4Sm100> ||
                     cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecializedMxf8f6f4Sm120> ||
                     cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecializedMxf8f6f4Acc2x4Sm120> ||
                     cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecializedMxf4Sm120>) {
    return 64;
  }
  else {
    return 32;
  }
}

/**
 * @brief Check for F8F6F4 alignment requirement
 * 
 * @tparam TileShape_MNK (MmaAtomShape_M, MmaAtomShape_N, TileShape_K)
 * @tparam ClusterShape_MNK (cluster_M, cluster_N, cluster_K)
 * @tparam BuilderScheduleTag Builder tag
 */
template<
  class ElementAMma,
  class ElementBMma,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class LayoutA,
  class LayoutB,
  bool IsSparse,
  bool Is2sm = false
>
constexpr bool sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement(){
  // * 1SM Dense
  //    * A_K(t) : TileShape_K % 128 == 0
  //    * A_M(n) : TileShape_M % 128 == 0
  //    * B_N(t) : TileSize_N % 128 == 0
  //    * B_K(n) : TileSize_K % 128 == 0
  //
  // * 2SM Dense
  //    * A_K(t) : TileShape_K % 128 == 0
  //    * A_M(n) : TileShape_M % 128 == 0
  //    * B_N(t) : TileSize_N % 256 == 0
  //        each sm load half the data along tile_n (split vertically), each sm needs to be 128 elts aligned.
  //        full tile_n needs to be 256 elts aligned
  //    * B_K(n) : TileShape_K % 128 == 0
  //
  // * 1SM Sparse
  //    * A_K(t) : TileShape_K % 256 == 0
  //        num of physical elems needs to be 128 elts aligned
  //        num of logical elems needs to be 256 elts aligned
  //    * A_M(n) : TileShape_M % 128 == 0
  //    * B_N(t) : TileSize_N % 128 == 0
  //    * B_K(n) : TileSize_K % 128 == 0
  //
  // * 2SM Sparse
  //    * A_K(t) : TileShape_K % 256 == 0
  //        num of physical elems needs to be 128 elts aligned
  //        num of logical elems needs to be 256 elts aligned
  //    * A_M(n) : TileShape_M % 128 == 0
  //    * B_N(t) : TileSize_N % 256 == 0
  //        each sm load half the data along tile_n (split vertically), each sm needs to be 128 elts aligned.
  //        full tile_n needs to be 256 elts aligned
  //    * B_K(n) : TileShape_K % 128 == 0
  //
  // * Valid TileShape_MNK Dense
  //    * Notation: 
  //          mma_instruction_tile_shape-cta_tile_shape
  //    * s128x128x64
  //          s128x128x32_128x128x128_nn YES
  //          s128x128x32_128x128x128_nt YES
  //          s128x128x32_128x128x128_tn YES
  //          s128x128x32_128x128x128_tt YES
  //    * s128x256x64
  //          s128x256x32_128x256x128_nn YES
  //          s128x256x32_128x256x128_nt YES
  //          s128x256x32_128x256x128_tn YES
  //          s128x256x32_128x256x128_tt YES
  //    * s256x128x64
  //          s256x128x32_256x128x128_nn YES
  //          s256x128x32_256x128x128_nt NO (2SM B_N TileSize_N % 256 != 0)
  //          s256x128x32_256x128x128_tn YES
  //          s256x128x32_256x128x128_tt NO (2SM B_N TileSize_N % 256 != 0)
  //    * s256x256x64
  //          s256x256x32_256x256x128_nn YES
  //          s256x256x32_256x256x128_nt YES
  //          s256x256x32_256x256x128_tn YES
  //          s256x256x32_256x256x128_tt YES
  //
  // * Valid TileShape_MNK Sparse
  //    * s128x128x64
  //          s128x128x64_128x128x128_nn YES
  //          s128x128x64_128x128x128_nt YES
  //          s128x128x64_128x128x128_tn NO (A_K TileShape_K % 256 != 0)
  //          s128x128x64_128x128x128_tt NO (A_K TileShape_K % 256 != 0)
  //          s128x128x64_128x128x256_nn YES
  //          s128x128x64_128x128x256_nt YES
  //          s128x128x64_128x128x256_tn YES
  //          s128x128x64_128x128x256_tt YES
  //    * s128x256x64
  //          s128x256x64_128x256x128_nn YES
  //          s128x256x64_128x256x128_nt YES
  //          s128x256x64_128x256x128_tn NO (A_K TileShape_K % 256 != 0)
  //          s128x256x64_128x256x128_tt NO (A_K TileShape_K % 256 != 0)
  //          s128x256x64_128x256x256_nn YES
  //          s128x256x64_128x256x256_nt YES
  //          s128x256x64_128x256x256_tn YES
  //          s128x256x64_128x256x256_tt YES
  //    * s256x128x64
  //          s256x128x64_128x128x128_nn YES
  //          s256x128x64_128x128x128_nt NO (2SM B_N TileSize_N % 256 != 0)
  //          s256x128x64_128x128x128_tn NO (A_K TileShape_K % 256 != 0)
  //          s256x128x64_128x128x128_tt NO (A_K TileShape_K % 256 != 0)
  //          s256x128x64_128x128x256_nn YES
  //          s256x128x64_128x128x256_nt NO (2SM B_N TileSize_N % 256 != 0)
  //          s256x128x64_128x128x256_tn YES
  //          s256x128x64_128x128x256_tt NO (2SM B_N TileSize_N % 256 != 0)
  //    * s256x256x64
  //          s256x256x64_128x256x128_nn YES
  //          s256x256x64_128x256x128_nt YES
  //          s256x256x64_128x256x128_tn NO (A_K TileShape_K % 256 != 0)
  //          s256x256x64_128x256x128_tt NO (A_K TileShape_K % 256 != 0)
  //          s256x256x64_128x256x256_nn YES
  //          s256x256x64_128x256x256_nt YES
  //          s256x256x64_128x256x256_tn YES
  //          s256x256x64_128x256x256_tt YES

  [[maybe_unused]] constexpr int TileShape_M = Is2sm ? size<0>(TileShape_MNK{}) / 2 : size<0>(TileShape_MNK{});
  [[maybe_unused]] constexpr int TileShape_N = size<1>(TileShape_MNK{});
  [[maybe_unused]] constexpr int TileShape_K = size<2>(TileShape_MNK{});

  constexpr bool is_b_unpack_f4_f6 = cute::is_same_v<ElementBMma, cutlass::detail::float_e2m1_unpacksmem_t> ||
                                     cute::is_same_v<ElementBMma, cutlass::detail::float_e3m2_unpacksmem_t> ||
                                     cute::is_same_v<ElementBMma, cutlass::detail::float_e2m3_unpacksmem_t> ||
                                     cute::is_same_v<ElementBMma, cutlass::detail::type_erased_dynamic_float4_unpacksmem_t> ||
                                     cute::is_same_v<ElementBMma, cutlass::detail::type_erased_dynamic_float6_unpacksmem_t>;
  // For sparse, this is ElementAMmaRaw
  constexpr bool is_a_unpack_f4_f6 = cute::is_same_v<ElementAMma, cutlass::detail::float_e2m1_unpacksmem_t> ||
                                     cute::is_same_v<ElementAMma, cutlass::detail::float_e3m2_unpacksmem_t> ||
                                     cute::is_same_v<ElementAMma, cutlass::detail::float_e2m3_unpacksmem_t> ||
                                     cute::is_same_v<ElementAMma, cutlass::detail::type_erased_dynamic_float4_unpacksmem_t> ||
                                     cute::is_same_v<ElementAMma, cutlass::detail::type_erased_dynamic_float6_unpacksmem_t>;

  [[maybe_unused]] constexpr bool is_b_n_major = cute::is_same_v<LayoutB, cutlass::layout::RowMajor>;
  [[maybe_unused]] constexpr bool is_b_k_major = not is_b_n_major;
  [[maybe_unused]] constexpr bool is_a_m_major = cute::is_same_v<LayoutA, cutlass::layout::ColumnMajor>;
  [[maybe_unused]] constexpr bool is_a_k_major = not is_a_m_major;

  // 2SM
  if constexpr (Is2sm) {
    if constexpr (IsSparse) {
      constexpr bool valid_a = !is_a_unpack_f4_f6 || (is_a_k_major ?
                                                    TileShape_K % 256 == 0:
                                                    TileShape_M % 128 == 0);

      constexpr bool valid_b = !is_b_unpack_f4_f6 || (is_b_n_major ?
                                                    TileShape_N % 256 == 0: 
                                                    TileShape_K % 128 == 0);
      return valid_a && valid_b;
    }
    else {
      constexpr bool valid_a = !is_a_unpack_f4_f6 || (is_a_k_major ?
                                                    TileShape_K % 128 == 0 :
                                                    TileShape_M % 128 == 0);

      constexpr bool valid_b = !is_b_unpack_f4_f6 || (is_b_n_major ?
                                                    TileShape_N % 256 == 0: 
                                                    TileShape_K % 128 == 0);
      return valid_a && valid_b;
    }
  }
  // 1SM
  else {
    if constexpr (IsSparse) {
      constexpr bool valid_a = !is_a_unpack_f4_f6 || (is_a_k_major ?
                                                    TileShape_K % 256 == 0:
                                                    TileShape_M % 128 == 0);

      constexpr bool valid_b = !is_b_unpack_f4_f6 || (is_b_n_major ? 
                                                    TileShape_N % 128 == 0 : 
                                                    TileShape_K % 128 == 0);
      return valid_a && valid_b;
    }
    else {
      constexpr bool valid_a = !is_a_unpack_f4_f6 || (is_a_k_major ?
                                                    TileShape_K % 128 == 0 :
                                                    TileShape_M % 128 == 0);

      constexpr bool valid_b = !is_b_unpack_f4_f6 || (is_b_n_major ? 
                                                    TileShape_N % 128 == 0 : 
                                                    TileShape_K % 128 == 0);
      return valid_a && valid_b;
    }
  }
}

template <class ElementA, int AlignmentA, class ElementB, int AlignmentB, class BuilderScheduleTag>
constexpr bool
sm1xx_gemm_is_aligned() {
  // Only support dense gemm alignment check
  constexpr bool is_f8f6f4_subbytes = cute::sizeof_bits_v<ElementA> < 8 || cute::sizeof_bits_v<ElementB> < 8;

  return ((cute::sizeof_bits_v<ElementA> * AlignmentA) % cutlass::detail::get_input_alignment_bits<ElementA, is_f8f6f4_subbytes>() == 0) &&
         ((cute::sizeof_bits_v<ElementB> * AlignmentB) % cutlass::detail::get_input_alignment_bits<ElementB, is_f8f6f4_subbytes>() == 0);
}

template <class ElementA, int AlignmentA, class ElementB, int AlignmentB, class BuilderScheduleTag>
constexpr bool
sm1xx_blockscaled_gemm_is_aligned() {
  // Only support blocksscaled gemm alignment check
  constexpr bool is_mxf8f6f4_subbytes = (cute::sizeof_bits_v<ElementA> < 8 || cute::sizeof_bits_v<ElementB> < 8) &&
                                    (cute::is_base_of_v<KernelScheduleMxf8f6f4Sm100, BuilderScheduleTag> ||
                                     cute::is_base_of_v<KernelScheduleMxf8f6f4Sm120, BuilderScheduleTag> );

  return ((cute::sizeof_bits_v<ElementA> * AlignmentA) % cutlass::detail::get_input_alignment_bits<ElementA, is_mxf8f6f4_subbytes>() == 0) &&
         ((cute::sizeof_bits_v<ElementB> * AlignmentB) % cutlass::detail::get_input_alignment_bits<ElementB, is_mxf8f6f4_subbytes>() == 0);
}

template <class ElementA, int AlignmentA, class GmemLayoutATag, class ElementB, int AlignmentB, class BuilderScheduleTag>
constexpr bool
sm1xx_sparse_gemm_is_aligned() {
  // Only support sparse gemm alignment check
  constexpr bool is_f8f6f4_subbytes = cute::sizeof_bits_v<ElementA> < 8 || cute::sizeof_bits_v<ElementB> < 8;
  constexpr int a_k_major_compress_factor = cutlass::gemm::detail::is_k_major_A<GmemLayoutATag>() ? 2 : 1;

  return ((cute::sizeof_bits_v<ElementA> * AlignmentA / a_k_major_compress_factor) % 
          cutlass::detail::get_input_alignment_bits<ElementA, is_f8f6f4_subbytes>() == 0) &&
         ((cute::sizeof_bits_v<ElementB> * AlignmentB) % cutlass::detail::get_input_alignment_bits<ElementB, is_f8f6f4_subbytes>() == 0);
}

template <class ElementA, int AlignmentA, class GmemLayoutATag, class ElementB, int AlignmentB, class BuilderScheduleTag>
constexpr bool
sm1xx_blockscaled_sparse_gemm_is_aligned() {
  // Only support blocksscaled sparse gemm alignment check
  constexpr bool is_mxf8f6f4_subbytes = (cute::sizeof_bits_v<ElementA> < 8 || cute::sizeof_bits_v<ElementB> < 8) &&
                                    (cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm100, BuilderScheduleTag> ||
                                     cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm120, BuilderScheduleTag>);
  constexpr int a_k_major_compress_factor = cutlass::gemm::detail::is_k_major_A<GmemLayoutATag>() ? 2 : 1;

  return ((cute::sizeof_bits_v<ElementA> * AlignmentA / a_k_major_compress_factor) % 
          cutlass::detail::get_input_alignment_bits<ElementA, is_mxf8f6f4_subbytes>() == 0) &&
         ((cute::sizeof_bits_v<ElementB> * AlignmentB) % cutlass::detail::get_input_alignment_bits<ElementB, is_mxf8f6f4_subbytes>() == 0);
}

template<class CollectiveEpilogue>
constexpr int
compute_carveout_from_epi() {
  constexpr int tensor_alignment = cutlass::const_max(128, cute::alignment_of_v<typename CollectiveEpilogue::TensorStorage>);
  constexpr int pipeline_alignment = 16;

  return cutlass::round_up(sizeof(typename CollectiveEpilogue::TensorStorage), tensor_alignment) +
         cutlass::round_up(sizeof(typename CollectiveEpilogue::PipelineStorage), pipeline_alignment);
}

namespace blockscaled {

enum class BlockScaledInstr {
  MXF4_NVF4,
  MXF4F6F8
};

template <class BuilderScheduleTag, class T>
struct blockscaled_type {};

template <class BuilderScheduleTag, class T, class SF>
struct blockscaled_type<BuilderScheduleTag, cute::tuple<T,SF>> {
  using sf_type = SF;
  using data_type = T;
  static constexpr uint32_t SfVectorSize = detail::find_vector_size<BuilderScheduleTag>();
};

template <class BuilderScheduleTag, class T, class SF, int SfVectorSize_>
struct blockscaled_type<BuilderScheduleTag, cute::tuple<T,SF, cute::Int<SfVectorSize_>>> {
  using sf_type = SF;
  using data_type = T;
  static constexpr uint32_t SfVectorSize = SfVectorSize_;
};

template <class BuilderScheduleTag, class T>
struct blockscaled_type<BuilderScheduleTag, cutlass::mx_float6_t<T>> {
  using sf_type = cutlass::float_ue8m0_t;
  using data_type = T;
  static constexpr uint32_t SfVectorSize =
    (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ||
     cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm120, BuilderScheduleTag>) ? 64 : 32;
};

template <class BuilderScheduleTag, class T>
struct blockscaled_type<BuilderScheduleTag, cutlass::mx_float4_t<T>> {
  using sf_type = cutlass::float_ue8m0_t;
  using data_type = T;
  static constexpr uint32_t SfVectorSize =
    (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ||
     cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm120, BuilderScheduleTag>) ? 64 : 32;
};

template <class BuilderScheduleTag, class T>
struct blockscaled_type<BuilderScheduleTag, nv_float4_t<T>> {
  using sf_type = cutlass::float_ue4m3_t;
  using data_type = T;
  static constexpr uint32_t SfVectorSize =
    (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ||
     cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm120, BuilderScheduleTag>) ? 32 : 16;
};
template <class BuilderScheduleTag, class T>
struct blockscaled_type<BuilderScheduleTag, cutlass::mx_float8_t<T>> {
  using sf_type = cutlass::float_ue8m0_t;
  using data_type = T;
  static constexpr uint32_t SfVectorSize =
    (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ||
     cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm120, BuilderScheduleTag>) ? 64 : 32;
};

template <
  class BuilderScheduleTag,
  class ElementPairA, class ElementPairB,
  UMMA::Major UmmaMajorA, UMMA::Major UmmaMajorB
>
CUTLASS_HOST_DEVICE
static constexpr bool
check_input_datatypes() {
  using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
  using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
  using ElementA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type;
  using ElementB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type;
  constexpr uint32_t SfVectorSizeA = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::SfVectorSize;
  constexpr uint32_t SfVectorSizeB = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::SfVectorSize;

  auto is_auto_instr_selection_policy = [&]() {
    return ((cute::is_same_v<BuilderScheduleTag, KernelScheduleAuto>)
            // SM100 BS
            || (cute::is_same_v<BuilderScheduleTag, KernelScheduleBlockScaledGemmSm100>)
            || (cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized1SmBlockScaledSm100>)
            || (cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecialized2SmBlockScaledSm100>)
            || (cute::is_same_v<BuilderScheduleTag, KernelMixedTmaCpAsyncWarpSpecialized1SmBlockScaledSm100>)
            // SM100 BS ptr_array
            || (cute::is_same_v<BuilderScheduleTag, KernelSchedulePtrArrayBlockScaledGemmSm100>)
            || (cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized1SmBlockScaledSm100>)
            || (cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecialized2SmBlockScaledSm100>)
            // SM100 BSSP
            || (cute::is_same_v<BuilderScheduleTag, KernelScheduleBlockScaledSparseGemmSm100>)
            || (cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecialized1SmBlockScaledSm100>)
            || (cute::is_same_v<BuilderScheduleTag, KernelSparseTmaWarpSpecialized2SmBlockScaledSm100>)
            // SM120 BS
            || (cute::is_same_v<BuilderScheduleTag, KernelScheduleBlockScaledGemmSm120>)
            || (cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecializedPingpong>)
            || (cute::is_same_v<BuilderScheduleTag, KernelTmaWarpSpecializedCooperative>)
            // SM120 BS ptr_array
            || (cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecializedPingpong>)
            || (cute::is_same_v<BuilderScheduleTag, KernelPtrArrayTmaWarpSpecializedCooperative>)
            // SM120 BSSP
            || (cute::is_same_v<BuilderScheduleTag, KernelScheduleBlockScaledSparseGemmSm120>)
            );
  };

  static_assert(cute::is_same_v<ElementSFA, ElementSFB>, "Scale factor types for A and B should be the same.");
  static_assert((SfVectorSizeA == SfVectorSizeB), "Scale factor vector size for A and B should be the same.");
  if constexpr ((SfVectorSizeA == 0) || (SfVectorSizeB == 0)) {
     static_assert(!is_auto_instr_selection_policy(), "Auto instr selection isn't valid if scale factor vector size can't be determined from the types");
  }

  static_assert(cute::is_same_v<ElementSFA, cutlass::float_ue8m0_t> 
                || cute::is_same_v<ElementSFA, cutlass::float_ue4m3_t>, "Incorrect scale factor type");

    if constexpr (((sizeof_bits_v<ElementA> == 4 || sizeof_bits_v<ElementA> == 6 || sizeof_bits_v<ElementA> == 8) &&
                   (sizeof_bits_v<ElementB> == 4 || sizeof_bits_v<ElementB> == 6 || sizeof_bits_v<ElementB> == 8)    ) &&  // A and B are 4, 6, or 8 bit types and
                  (!(sizeof_bits_v<ElementA> == 4 && sizeof_bits_v<ElementB> == 4)                                   )     // A and B are not both 4 bit types
                 ) {
      ///////////////////////////////////////////////////////////////////////
      // Mixed Precision FP4, FP6, FP8 case. -> MX_F4F6F8 instructions
      ///////////////////////////////////////////////////////////////////////
      // 1. Check Scale factor data type
      static_assert(cute::is_same_v<ElementSFA, cutlass::float_ue8m0_t>, "MX_F4F6F8 only supports ue8m0 SF type");
      // 2. Check whether A and B type combinations are valid or not
      static_assert(
        ( // If runtime datatypes are used, then both A and B should be runtime data type
          (
           cute::is_same_v<ElementA, cutlass::type_erased_dynamic_float8_t> ||
           cute::is_same_v<ElementA, cutlass::type_erased_dynamic_float6_t> ||
           cute::is_same_v<ElementA, cutlass::type_erased_dynamic_float4_t>
          ) &&
          (
           cute::is_same_v<ElementB, cutlass::type_erased_dynamic_float8_t> ||
           cute::is_same_v<ElementB, cutlass::type_erased_dynamic_float6_t> ||
           cute::is_same_v<ElementB, cutlass::type_erased_dynamic_float4_t>
          )
        ) ||
        ( // Valid (explicit) A and B type pairs
          (
           cute::is_same_v<ElementA, cutlass::float_e2m1_t> ||
           cute::is_same_v<ElementA, cutlass::float_e2m3_t> ||
           cute::is_same_v<ElementA, cutlass::float_e3m2_t> ||
           cute::is_same_v<ElementA, cutlass::float_e4m3_t> ||
           cute::is_same_v<ElementA, cutlass::float_e5m2_t> 
          ) &&
          (
           cute::is_same_v<ElementB, cutlass::float_e2m1_t> ||
           cute::is_same_v<ElementB, cutlass::float_e2m3_t> ||
           cute::is_same_v<ElementB, cutlass::float_e3m2_t> ||
           cute::is_same_v<ElementB, cutlass::float_e4m3_t> ||
           cute::is_same_v<ElementB, cutlass::float_e5m2_t> 
          )
        ), "Incorrect types for A and B for MX_F4F6F8"
      );
      // 3. Check Scale factor vector size is valid. 
      //   SfVectorSize = 32 for blockscaled dense gemm and ptr array blockscaled dense gemm
      //   SfVectorSize = 64 for blockscaled sparse gemm
      static_assert(
        ((SfVectorSizeA == 32 && cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>)
      || (SfVectorSizeA == 32 && cute::is_same_v<KernelTmaWarpSpecializedPingpong, BuilderScheduleTag>)
      || (SfVectorSizeA == 32 && cute::is_same_v<KernelTmaWarpSpecializedCooperative, BuilderScheduleTag>)
      || (SfVectorSizeA == 32 && cute::is_same_v<KernelPtrArrayTmaWarpSpecializedPingpong, BuilderScheduleTag>)
      || (SfVectorSizeA == 32 && cute::is_same_v<KernelPtrArrayTmaWarpSpecializedCooperative, BuilderScheduleTag>)
      || (SfVectorSizeA == 32 && cute::is_base_of_v<KernelScheduleBlockScaledGemmSm100, BuilderScheduleTag>)
      || (SfVectorSizeA == 32 && cute::is_base_of_v<KernelSchedulePtrArrayBlockScaledGemmSm100, BuilderScheduleTag>)
      || (SfVectorSizeA == 64 && cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag>)
      || (SfVectorSizeA == 32 && cute::is_base_of_v<KernelScheduleBlockScaledGemmSm120, BuilderScheduleTag>)
      || (SfVectorSizeA == 64 && cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm120, BuilderScheduleTag>)
        ), "Incorrect SfVectorSize for MX_F4F6F8 is deduced.");

      // 4. Check the kernel policy. Kernel policy should be either auto or *MXf8f6f4*
      static_assert((cute::is_base_of_v<KernelScheduleMxf8f6f4Sm100, BuilderScheduleTag>
                  || cute::is_base_of_v<KernelSchedulePtrArrayMxf8f6f4Sm100, BuilderScheduleTag>
                  || cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm100, BuilderScheduleTag>
                  || cute::is_base_of_v<KernelScheduleMxf8f6f4Sm120, BuilderScheduleTag>
                  || cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm120, BuilderScheduleTag>
                  || is_auto_instr_selection_policy()), "Incorrect Kernel Schedule Policy for Mx_F4F6F8 type inputs.");

      return true;
    }
    else if constexpr ((sizeof_bits_v<ElementA> == 4 && sizeof_bits_v<ElementB> == 4)) {
      ///////////////////////////////////////////////////////////////////////
      // A and B are both 4 bit types
      // There are multiple block scaled tcgen05.mma instructions supporting F4 types.
      ///////////////////////////////////////////////////////////////////////

      // 1. Check Scale factor data type
      static_assert(cute::is_same_v<ElementSFA, cutlass::float_ue8m0_t> 
                      || cute::is_same_v<ElementSFA, cutlass::float_ue4m3_t>
                      , "MXNV_F4 supports ue8m0 and ue4m3 SF types");
      // 2. Check whether A and B type combinations are valid or not
      static_assert(
         ( // If runtime datatypes are used, then both A and B should be runtime data type
          cute::is_same_v<ElementA, cutlass::type_erased_dynamic_float4_t> && 
          cute::is_same_v<ElementB, cutlass::type_erased_dynamic_float4_t>
         ) ||
         ( // Valid (explicit) A and B type pairs
          (
            cute::is_same_v<ElementA, cutlass::float_e2m1_t>
          ) &&
          (
            cute::is_same_v<ElementB, cutlass::float_e2m1_t>
          )
         ), "Incorrect types for A and B for MXNV_F4");
        // 3. Skip checking the scale factor vector size. Will be checked later for specific Kernel Schedule policies.
        // 4. Check the kernel policy.
        static_assert((cute::is_base_of_v<KernelScheduleMxf8f6f4Sm100, BuilderScheduleTag>          ||
                       cute::is_base_of_v<KernelScheduleMxNvf4Sm100, BuilderScheduleTag>            ||
                       cute::is_base_of_v<KernelSchedulePtrArrayMxf8f6f4Sm100, BuilderScheduleTag>  ||
                       cute::is_base_of_v<KernelSchedulePtrArrayMxNvf4Sm100, BuilderScheduleTag>    ||
                       cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm100, BuilderScheduleTag>    ||
                       cute::is_base_of_v<KernelScheduleSparseMxNvf4Sm100, BuilderScheduleTag>      ||
                       cute::is_base_of_v<KernelScheduleMxf8f6f4Sm120, BuilderScheduleTag>          ||
                       cute::is_base_of_v<KernelScheduleMxNvf4Sm120, BuilderScheduleTag>            ||
                       cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm120, BuilderScheduleTag>    ||
                       cute::is_base_of_v<KernelScheduleSparseMxNvf4Sm120, BuilderScheduleTag>      ||
                       is_auto_instr_selection_policy()), "Incorrect Kernel Schedule Policy for F4 type inputs.");

        // If a policy is specified, do more checks
        if constexpr (cute::is_base_of_v<KernelScheduleMxf8f6f4Sm100, BuilderScheduleTag>
                   || cute::is_base_of_v<KernelSchedulePtrArrayMxf8f6f4Sm100, BuilderScheduleTag>
                   || cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm100, BuilderScheduleTag>
                   || cute::is_base_of_v<KernelScheduleMxf8f6f4Sm120, BuilderScheduleTag>
                   || cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm120, BuilderScheduleTag>) {
          // Perform additional checks. Only subset of FP4 and scale factor types are supported.
          static_assert(cute::is_same_v<ElementSFA, cutlass::float_ue8m0_t>, "MX_F4F6F8 only supports ue8m0 SF type");
          static_assert((cute::is_same_v<ElementA, cutlass::type_erased_dynamic_float4_t> &&
                         cute::is_same_v<ElementB, cutlass::type_erased_dynamic_float4_t>) ||
                        (cute::is_same_v<ElementA, cutlass::float_e2m1_t> &&
                         cute::is_same_v<ElementB, cutlass::float_e2m1_t>), "Incorrect types for A and B for MX_F4F6F8");
          static_assert(detail::find_vector_size<BuilderScheduleTag>() == SfVectorSizeA,
                        "Kernel Schedule policy doesn't match the scale factor vector size.");
          return true;
        }
        else if constexpr (cute::is_base_of_v<KernelScheduleMxNvf4Sm100, BuilderScheduleTag>
                        || cute::is_base_of_v<KernelSchedulePtrArrayMxNvf4Sm100, BuilderScheduleTag>
                        || cute::is_base_of_v<KernelScheduleSparseMxNvf4Sm100, BuilderScheduleTag>
                        || cute::is_base_of_v<KernelScheduleMxNvf4Sm120, BuilderScheduleTag>
                        || cute::is_base_of_v<KernelScheduleSparseMxNvf4Sm120, BuilderScheduleTag>) {
            static_assert((UmmaMajorA == UMMA::Major::K && UmmaMajorB == UMMA::Major::K), "MX/NV_F4 only supports RowMajor A, and ColMajorB");
            static_assert(detail::find_vector_size<BuilderScheduleTag>() == SfVectorSizeA,
                          "Kernel Schedule policy doesn't match the scale factor vector size.");
          return true;
        }
        else { // auto policy
          // If the scale factor type is ue4m3 or the scale factor vector size is 16 -> only MXF4_NVF4 instruction can support it
          // For MXF4_NVF4, the layouts should be RowMajor A, and ColMajorB
          static_assert(is_auto_instr_selection_policy(), "Kernel Schedule policy should be auto");
          if constexpr (SfVectorSizeA == 16 || SfVectorSizeB == 16
                        || cute::is_same_v<ElementSFA, cutlass::float_ue4m3_t>
                       ) { // Only MXF4NVF4 can support these types
            static_assert((UmmaMajorA == UMMA::Major::K && UmmaMajorB == UMMA::Major::K), "NV_F4 only supports RowMajor A, and ColMajorB");
            return true;
          }
          return true;
        }
    }
    else {
      return false;
    }
  return false;
}

template <
  class TileShape_MNK, // (MmaAtomShape_M, MmaAtomShape_N, CtaTileShapeK)
  class ClusterShape_MNK,
  class BuilderScheduleTag
>
CUTLASS_HOST_DEVICE
static constexpr bool
is_2sm() {
  // 2SM kernel schedule is requested
  if constexpr (cute::is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag>) { return true; }
  // 1SM kernel schedule is requested
  else if constexpr (cute::is_base_of_v<KernelSchedule1Sm, BuilderScheduleTag>) { return false; }
  // auto schedule is used.
  else {
    if constexpr (!cute::is_static_v<ClusterShape_MNK>) {
      // If the cluster shape is dynamic, we can't guarantee 2x1. Default to 1sm.
      // If tile shape M is 256, throw an error. M=256 is only supported by 2SM instructions.
      static_assert(get<0>(TileShape_MNK{}) != 256, "If M=256, auto policy can't create 2sm kernels. Specify a 2SM policy");
      return false;
    }
    else if constexpr (cute::is_static_v<ClusterShape_MNK> && cute::get<0>(ClusterShape_MNK{}) % 2 == 0) {
      // We need to check the TileShape
      if constexpr (get<0>(TileShape_MNK{}) == 256) {
        return true;
      }
      else if constexpr (get<0>(TileShape_MNK{}) == 128) {
        return false;
      }
      else {
        static_assert(get<0>(TileShape_MNK{}) == 0, "Unsupported M dimension for TileShape_MNK.");
      }
    }
    else { return false;}
  }
}

template <
  class ElementPairA,
  class ElementPairB,
  class ElementAccumulator,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  class BuilderScheduleTag
>
CUTLASS_HOST_DEVICE
static constexpr auto
select_instr() {
  using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
  using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
  using ElementA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type;
  using ElementB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type;
  constexpr uint32_t SfVectorSizeA = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::SfVectorSize;
  constexpr uint32_t SfVectorSizeB = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::SfVectorSize;
  constexpr int SfVectorSize = SfVectorSizeA > SfVectorSizeB ? SfVectorSizeA : SfVectorSizeB;
  using ElementSF = ElementSFA;

  if constexpr (cute::is_base_of_v<KernelScheduleMxf8f6f4Sm100, BuilderScheduleTag>
             || cute::is_base_of_v<KernelSchedulePtrArrayMxf8f6f4Sm100, BuilderScheduleTag>
             || cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm100, BuilderScheduleTag>
             || cute::is_base_of_v<KernelScheduleMxf8f6f4Sm120, BuilderScheduleTag>
             || cute::is_base_of_v<KernelScheduleSparseMxf8f6f4Sm120, BuilderScheduleTag>) {
    return detail::blockscaled::BlockScaledInstr::MXF4F6F8;
  }
  else if constexpr (cute::is_base_of_v<KernelScheduleMxNvf4Sm100, BuilderScheduleTag>
                  || cute::is_base_of_v<KernelSchedulePtrArrayMxNvf4Sm100, BuilderScheduleTag>
                  || cute::is_base_of_v<KernelScheduleSparseMxNvf4Sm100, BuilderScheduleTag>
                  || cute::is_base_of_v<KernelScheduleMxNvf4Sm120, BuilderScheduleTag>
                  || cute::is_base_of_v<KernelScheduleSparseMxNvf4Sm120, BuilderScheduleTag>) {
    return detail::blockscaled::BlockScaledInstr::MXF4_NVF4;
  }
  else {
    // Auto scheduling
    if constexpr ((sizeof_bits_v<ElementA> >= 6 && sizeof_bits_v<ElementA> <= 8) &&
                  (sizeof_bits_v<ElementB> >= 6 && sizeof_bits_v<ElementB> <= 8)) {
      // These types can only be supported by MX_F8F6F4 instruction
      static_assert(
         (SfVectorSize == 32 && cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>)
      || (SfVectorSize == 32 && cute::is_base_of_v<KernelScheduleBlockScaledGemmSm100, BuilderScheduleTag>)
      || (SfVectorSize == 32 && cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, BuilderScheduleTag>)
      || (SfVectorSize == 32 && cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, BuilderScheduleTag>)
      || (SfVectorSize == 32 && cute::is_base_of_v<KernelSchedulePtrArrayBlockScaledGemmSm100, BuilderScheduleTag>)
      || (SfVectorSize == 64 && cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag>
      || (SfVectorSize == 32 && cute::is_base_of_v<KernelScheduleBlockScaledGemmSm120, BuilderScheduleTag>)
      || (SfVectorSize == 64 && cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm120, BuilderScheduleTag>)
        ), "Incorrect SfVectorSize for MX_F4F6F8 is deduced.");
      return detail::blockscaled::BlockScaledInstr::MXF4F6F8;
    }
    else if constexpr (( sizeof_bits_v<ElementA> == 4 && (sizeof_bits_v<ElementB> == 6 || sizeof_bits_v<ElementB> == 8)) ||
                      ((sizeof_bits_v<ElementA> == 6 || sizeof_bits_v<ElementA> == 8) && sizeof_bits_v<ElementB> == 4)) {
      // Fp4 can be mixed with FP6, Fp8 with MMA.MXF8F6F4 only
      return detail::blockscaled::BlockScaledInstr::MXF4F6F8;
    }
    else if constexpr (sizeof_bits_v<ElementA> == 4 && sizeof_bits_v<ElementB> == 4) {
      // Both A and B are 4bits
      if constexpr (UmmaMajorA == UMMA::Major::K && UmmaMajorB == UMMA::Major::K) {
        // MXF4_NVF4 possible
        return detail::blockscaled::BlockScaledInstr::MXF4_NVF4;
      }
      else {
        static_assert(
        ((SfVectorSize == 32 && cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>)
      || (SfVectorSize == 32 && cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, BuilderScheduleTag>)
      || (SfVectorSize == 32 && cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, BuilderScheduleTag>)
      || (SfVectorSize == 32 && cute::is_base_of_v<KernelScheduleBlockScaledGemmSm100, BuilderScheduleTag>)
      || (SfVectorSize == 32 && cute::is_base_of_v<KernelSchedulePtrArrayBlockScaledGemmSm100, BuilderScheduleTag>)
      || (SfVectorSize == 64 && cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag>)
      || (SfVectorSize == 32 && cute::is_base_of_v<KernelScheduleBlockScaledGemmSm120, BuilderScheduleTag>)
      || (SfVectorSize == 64 && cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm120, BuilderScheduleTag>)
          ), "Incorrect SfVectorSize for MX_F4F6F8 is deduced.");

        static_assert(cute::is_same_v<ElementSF, cutlass::float_ue8m0_t> &&
                      (cute::is_same_v<ElementA, cutlass::float_e2m1_t> &&
                       cute::is_same_v<ElementB, cutlass::float_e2m1_t> ||
                       cute::is_same_v<ElementA, cutlass::type_erased_dynamic_float4_t> &&
                       cute::is_same_v<ElementB, cutlass::type_erased_dynamic_float4_t>),
                      "Only MXF4 support with non-TN and MMA.MXF8F6F4.");
        return detail::blockscaled::BlockScaledInstr::MXF4F6F8;
      }
    }
  }
}

} // namespace blockscaled

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective
