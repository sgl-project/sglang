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
    \brief Blocked Scale configs specific for SM100 BlockScaled MMA
*/

#pragma once

#include "cutlass/layout/matrix.h"

#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace cutlass::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute;

template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct Sm1xxBlockScaledBasicChunk {

  using Blk_MN    = _128;
  using Blk_SF    =   _4; 

  using SfKMajorAtom  = Layout< Shape< Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>, 
                               Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
  using SfMNMajorAtom = Layout< Shape< Shape<Int<SFVecSize>, _4>,  Shape<_32,_4>>, 
                               Stride<Stride<            _0, _1>, Stride<_16,_4>>>;
  using SfAtom    = cute::conditional_t<major == UMMA::Major::K, SfKMajorAtom, SfMNMajorAtom>;
};

template<int SFVecSize_>
struct Sm1xxBlockScaledConfig {
  // We are creating the SFA and SFB tensors' layouts in the collective since they always have the same layout.
  // k-major order
  static constexpr int SFVecSize = SFVecSize_;
  using Sm1xxBlkScaledChunk = Sm1xxBlockScaledBasicChunk<SFVecSize>;
  using Blk_MN = typename Sm1xxBlkScaledChunk::Blk_MN;
  using Blk_SF = typename Sm1xxBlkScaledChunk::Blk_SF; 
  using SfAtom = typename Sm1xxBlkScaledChunk::SfAtom;

  using LayoutSF = decltype(blocked_product(SfAtom{}, make_layout( make_shape(int32_t(0), int32_t(0), int32_t(0)),
                                                                  make_stride(int32_t(0),       _1{}, int32_t(0)))));

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFA() {
    return LayoutSF{};
  }

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFB() {
    return LayoutSF{};
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFA.
  template < class ProblemShape, class LayoutSFA = LayoutSF>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFA(ProblemShape problem_shape, LayoutSFA layout_sfa = LayoutSFA{}) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    return tile_to_shape(SfAtom{}, make_shape(M,K,L), Step<_2,_1,_3>{});
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFB.
  template <class ProblemShape, class LayoutSFB = LayoutSF>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFB(ProblemShape problem_shape, LayoutSFB layout_sfb = LayoutSFB{}) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    return tile_to_shape(SfAtom{}, make_shape(N,K,L), Step<_2,_1,_3>{});
  }

  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFA(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {

    constexpr int MMA_NSF = TiledMma::K / SFVecSize;
    // Basic storage block for new Scaling Factor Layouts
    using mnBasicBlockShape  =  Shape<_32,_4>;
    using mnBasicBlockStride = Stride<_16,_4>;
    using kBasicBlockShape  = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
    using kBasicBlockStride = Stride<_0, _1>;

    // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
    using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                          cute::size<2>(TileShape_MNK{}))));
    // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
    using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                          cute::size<2>(TileShape_MNK{}))));
    // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
    // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size 
    using Blk_MN    = typename Sm1xxBlkScaledChunk::Blk_MN;
    using Blk_SF    = typename Sm1xxBlkScaledChunk::Blk_SF; 
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

    using TL_VMNK = typename TiledMma::ThrLayoutVMNK;
    constexpr TL_VMNK tl_vmnk{};
    constexpr int MMA_M = cute::size<0>(TileShape_MNK{}) / cute::size<0>(tl_vmnk);
    using mma_SFA_shape  = decltype( make_shape( prepend(Int<MMA_M>{}/Blk_MN{},  mnBasicBlockShape{}),  kBasicBlockShape{}));
    using mma_SFA_stride = decltype(make_stride( prepend(          Blk_Elems{}, mnBasicBlockStride{}), kBasicBlockStride{}));
    using sSFA_shape     = decltype( make_shape( mma_SFA_shape{}, _1{},   make_shape( Blk_SF{}/Int<MMA_NSF>{}, Int<size<2>(TileShape_MNK{}) / SFVecSize / Blk_SF{}>{})));
    using sSFA_stride    = decltype(make_stride(mma_SFA_stride{}, _0{},  make_stride(          Int<MMA_NSF>{},                   Int<MMA_M /Blk_MN{} * Blk_Elems{}>{})));
    using SmemLayoutAtomSFA = decltype(make_layout(sSFA_shape{}, sSFA_stride{}));
    return SmemLayoutAtomSFA{};
  }

  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFB(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {

    constexpr int MMA_NSF = TiledMma::K / SFVecSize;
    // Basic storage block for new Scaling Factor Layouts
    using mnBasicBlockShape  =  Shape<_32,_4>;
    using mnBasicBlockStride = Stride<_16,_4>;
    using kBasicBlockShape  = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
    using kBasicBlockStride = Stride<_0, _1>;

    // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
    using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                          cute::size<2>(TileShape_MNK{}))));
    // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
    using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                          cute::size<2>(TileShape_MNK{}))));
    // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
    // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size 
    using Blk_MN    = typename Sm1xxBlkScaledChunk::Blk_MN;
    using Blk_SF    = typename Sm1xxBlkScaledChunk::Blk_SF; 
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

    using TL_VMNK = typename TiledMma::ThrLayoutVMNK;
    constexpr TL_VMNK tl_vmnk{};
    constexpr int MMA_N = cute::size<1>(TileShape_MNK{});
    // If MMA_N is 192, we need to operate at MMA_N = 256 granularity for UTCCP to work for ScaleFactorB.
    // Both TMA and UTCCP will transfer scale factor B as if we have 256 columns in B matrix.
    constexpr int MMA_N_SFB = cutlass::ceil_div(MMA_N, Blk_MN{}) * Blk_MN{};
    using mma_SFB_shape  = decltype(make_shape( prepend(   Int<MMA_N_SFB>{}/Blk_MN{},  mnBasicBlockShape{}),  kBasicBlockShape{}));
    using mma_SFB_stride = decltype(make_stride(prepend(                 Blk_Elems{}, mnBasicBlockStride{}), kBasicBlockStride{}));
    using sSFB_shape     = decltype( make_shape( mma_SFB_shape{}, _1{},  make_shape( Blk_SF{}/Int<MMA_NSF>{}, Int<size<2>(TileShape_MNK{}) / SFVecSize / Blk_SF{}>{})));
    using sSFB_stride    = decltype(make_stride(mma_SFB_stride{}, _0{}, make_stride(         Int<MMA_NSF>{},               Int<MMA_N_SFB / Blk_MN{} * Blk_Elems{}>{})));
    using SmemLayoutAtomSFB = decltype(make_layout(sSFB_shape{}, sSFB_stride{}));
    return SmemLayoutAtomSFB{};
  }
};


template<int SFVecSize_, UMMA::Major major = UMMA::Major::K>
struct Sm1xxBlockScaledOutputConfig {
  // We are creating the SFD tensors' layouts in the collective.
  // k-major order
  static constexpr int SFVecSize = SFVecSize_;
  using Sm1xxBlkScaledChunk = cutlass::detail::Sm1xxBlockScaledBasicChunk<SFVecSize, major>;
  using Blk_MN = typename Sm1xxBlkScaledChunk::Blk_MN;
  using Blk_SF = typename Sm1xxBlkScaledChunk::Blk_SF; 
  using SfAtom = typename Sm1xxBlkScaledChunk::SfAtom;

  using LayoutKMajorSF  = decltype(blocked_product(SfAtom{}, make_layout(make_shape (int32_t(0), int32_t(0), int32_t(0)),
                                                                         make_stride(int32_t(0),       _1{}, int32_t(0)))));

  using LayoutMNMajorSF = decltype(blocked_product(SfAtom{}, make_layout(make_shape (int32_t(0), int32_t(0), int32_t(0)),
                                                                         make_stride(      _1{}, int32_t(0), int32_t(0)))));

  using LayoutSF = cute::conditional_t<major == UMMA::Major::K, LayoutKMajorSF, LayoutMNMajorSF>;

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFD() {
    return LayoutSF{};
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFC.
  template <class ProblemShape, class LayoutSFD = LayoutSF>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFD(ProblemShape problem_shape, LayoutSFD layout_sfc = LayoutSFD{}) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    if constexpr (major == UMMA::Major::K) {
      return tile_to_shape(SfAtom{}, make_shape(M,N,L), Step<_2,_1,_3>{});
    } 
    else { 
      return tile_to_shape(SfAtom{}, make_shape(M,N,L), Step<_1,_2,_3>{});
    }
  }
};

//// Describe the Scalefactor Tensor without VectorSize
struct Sm1xxBlockScaledTensorConfig {
  // k-major order
  // The blockscaled tensor does not need to know vectorsize
  using Blk_M = _128;
  using Blk_N =   _4; 
  using SfAtom = Layout< Shape< Shape<_32,_4>,  Shape<_4>>, 
                        Stride<Stride<_16,_4>, Stride<_1>>>;

  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape(ProblemShape problem_shape) {
    auto problem_shape_MNL = append<3>(problem_shape, 1);
    auto [M, N, L] = problem_shape_MNL;
    return tile_to_shape(SfAtom{}, make_shape(M,N,L), Step<_2,_1,_3>{});
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
