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

#include "cutlass/gemm/collective/builders/sm100_common.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<
  class LayoutA,
  int AlignmentA,
  class LayoutB,
  int AlignmentB,
  class CtaShape_MNK,
  class WarpShape_MNK
>
constexpr auto
sm100_make_simt_f32_tiled_mma() {
  using namespace cute;

  constexpr int CtaShape_M = cute::size<0>(CtaShape_MNK{});
  constexpr int CtaShape_N = cute::size<1>(CtaShape_MNK{});
  constexpr int CtaShape_K = cute::size<2>(CtaShape_MNK{});

  constexpr int WarpShape_M = cute::size<0>(WarpShape_MNK{});
  constexpr int WarpShape_N = cute::size<1>(WarpShape_MNK{});
  constexpr int WarpShape_K = cute::size<2>(WarpShape_MNK{});

  // Use Permutation to achieve a [4 x 4] value layout for each thread.
  // Ideally, we want the tiled mma to be such that loads from shared memory are 128 bit wide.
  // While as we are using CtaShape_K = 16, when A and B are K-major, we use tranpose + 8 byte padding to avoid smem bank conflict,
  // so we could only use 64 bit smem load.
  // When A and B are MN-major, we use 128 bit smem load.
  using PermutationA = Layout<Shape<_2, Int<WarpShape_M * 8>, _2>, Stride< _1, _4, _2>>;
  using PermutationB = Layout<Shape<Int<WarpShape_N * 4>, _4>, Stride< _4, _1>>;

  // For 32 threads in 1 warp, we use [8 x 4] thread layouts and each thread will hold [4 x 4] value layouts.
  // Then totally each warp will hold [32 x 16] value layouts.
  // So WarpShape_M needs to be equal or smaller than CtaShape_M / 32 and WarpShape_N needs to be equal or smaller than CtaShape_N / 16.
  static_assert(WarpShape_M <= CtaShape_M / 32, "WarpShape_M is too large, it needs to be equal or smaller than CtaShape_M / 32.");
  static_assert(WarpShape_N <= CtaShape_N / 16, "WarpShape_N is too large, it needs to be equal or smaller than CtaShape_N / 16.");

  constexpr int WarpStride_M = (WarpShape_M != 1) * NumThreadsPerWarp;
  constexpr int WarpStride_N = WarpShape_M * NumThreadsPerWarp;

  // We first introduce a [8 x 4] thread layouts in 1 warp.
  // And inside this [8 x 4] thread layouts, each 4 threads will be arranged as [2 x 2].
  // Then we could set different WarpShape to finalize how many warps we use in our tiled mma.
  // For example :
  // With 128 threads in the tiled mma, we could set the WarpShapeMNK as [2 x 2 x 1], [1 x 4 x 1] and [4 x 1 x 1].
  // With 64 threads in the tiled mma, we could set the WarpShapeMNK as [1 x 2 x 1] and [2 x 1 x 1].
  return make_tiled_mma(
    MMA_Atom<SM100_2x1x1_F32F32F32F32>{},
    Layout<Shape < Shape <_2, _4,  Int<WarpShape_M>>, Shape <_2, _2,  Int<WarpShape_N>>, _1>,
           Stride< Stride<_1, _8, Int<WarpStride_M>>, Stride<_2, _4, Int<WarpStride_N>>, _1>>{},
    Tile<
      PermutationA,
      PermutationB,
      Underscore>{});
}

} // namespace detail

template <
  class GmemLayoutATag,
  int AlignmentA,
  class GmemLayoutBTag,
  int AlignmentB,
  class CtaShape_MNK,
  class ClusterShape_MNK,
  int stages,
  class BuilderScheduleTag>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassSimt,
    float,
    GmemLayoutATag,
    AlignmentA,
    float,
    GmemLayoutBTag,
    AlignmentB,
    float,
    CtaShape_MNK,
    ClusterShape_MNK,
    StageCount<stages>,
    BuilderScheduleTag,
    cute::enable_if_t<
      (cute::is_same_v<BuilderScheduleTag, KernelMultistage> ||
       cute::is_same_v<BuilderScheduleTag, KernelPtrArrayMultistage> ||
       cute::is_same_v<BuilderScheduleTag, KernelScheduleAuto>) &&
      ((sizeof(float) * AlignmentA) % detail::cp_async_min_alignment_bytes == 0) &&
      ((sizeof(float) * AlignmentB) % detail::cp_async_min_alignment_bytes == 0) >> {
  static_assert(cute::size<2>(CtaShape_MNK{}) == 16, "SM100 SIMT SGEMM Kernels only support TileShape_K = 16.");

  // This kernel is specialized for F32 data type.
  using ElementA = float;
  using ElementB = float;

  using M = decltype(cute::size<0>(CtaShape_MNK{}));
  using N = decltype(cute::size<1>(CtaShape_MNK{}));
  using K = decltype(cute::size<2>(CtaShape_MNK{}));

  using WarpShape_MNK = decltype(detail::sm100_simt_f32_warp_shape_mnk_selector<CtaShape_MNK>());

  static constexpr int ThreadCount = cute::size(WarpShape_MNK{}) * NumThreadsPerWarp;

  using TiledMma = decltype(
    detail::sm100_make_simt_f32_tiled_mma<
      GmemLayoutATag,
      AlignmentA,
      GmemLayoutBTag,
      AlignmentB,
      CtaShape_MNK,
      WarpShape_MNK>());

  // for K major layouts, add a smem alignment offset to avoid bank conflicts
  static constexpr int SmemAlignmentOffsetA = cutlass::gemm::detail::is_mn_major_A<GmemLayoutATag>() ? 0 : 2;
  static constexpr int SmemAlignmentOffsetB = cutlass::gemm::detail::is_mn_major_B<GmemLayoutBTag>() ? 0 : 2;
  static constexpr int CtaShape_M = cute::size<0>(CtaShape_MNK{});
  static constexpr int CtaShape_N = cute::size<1>(CtaShape_MNK{});

  // Shared memory layout is [M x K] in M-major
  using SmemLayoutAtomA = cute::Layout<cute::Shape< M, K>,
                                       cute::Stride<_1, Int<CtaShape_M + SmemAlignmentOffsetA>>>;
  // A M-major use 128bit smem load.
  // A K-major needs to do tranpose and 8 byte padding to make smem bank conflict free, then we can only use 64bit smem load.
  using SmemCopyAtomA = std::conditional_t<cutlass::gemm::detail::is_mn_major_A<GmemLayoutATag>(),
                                            cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, ElementA>,
                                            cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<64>, ElementA>>;

  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * AlignmentA>;
  using GmemCopyAtomA = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeA>, ElementA>;
  using GmemTiledCopyA = decltype(
    detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomA, ThreadCount, AlignmentA, TagToStrideA_t<GmemLayoutATag>, M, K>());

  // Shared memory layout is [N x K] in N-major
  using SmemLayoutAtomB = cute::Layout<cute::Shape< N, K>,
                                       cute::Stride<_1, Int<CtaShape_N + SmemAlignmentOffsetB>>>;
  // B N-major use 128bit smem load.
  // B K-major needs to do tranpose and 8 byte padding to make smem bank conflict free, then we can only use 64bit smem load.
  using SmemCopyAtomB = std::conditional_t<cutlass::gemm::detail::is_mn_major_B<GmemLayoutBTag>(),
                                            cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, ElementB>,
                                            cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<64>, ElementB>>;

  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * AlignmentB>;
  using GmemCopyAtomB = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeB>, ElementB>;
  using GmemTiledCopyB = decltype(
    detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomB, ThreadCount, AlignmentB, TagToStrideB_t<GmemLayoutBTag>, N, K>());

  static constexpr bool IsArrayOfPointersGemm = cute::is_same_v<BuilderScheduleTag, KernelPtrArrayMultistage>;
  using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
      cutlass::gemm::MainloopSm80ArrayCpAsync<stages,
                                              ClusterShape_MNK>,
      cutlass::gemm::MainloopSm80CpAsync<stages,
                                         ClusterShape_MNK>
    >;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      CtaShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
