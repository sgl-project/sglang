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

/*! \file
  \brief im2col make_tma_copy
  
*/

#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/atom/copy_traits_sm90_im2col.hpp"
#include "cute/tensor.hpp"

namespace cute {

struct SM100_TMA_2SM_LOAD_IM2COL_OP : SM100_TMA_2SM_LOAD_IM2COL {};

/// @brief Non-executable specialization of Copy_Traits for SM100
///   im2col TMA load, with TMA descriptor but no barrier.
///
/// Use `.with(memory_barrier)` to construct an executable version.
template <class NumBitsPerTMA, class TMATensor>
struct Copy_Traits<SM100_TMA_2SM_LOAD_IM2COL, NumBitsPerTMA, TMATensor>
{
  using ThrID = Layout<_2>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_2, NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_2, NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  Im2ColTmaDescriptor tma_desc_;
  TMATensor tma_tensor_;

  CUTE_HOST_DEVICE constexpr
  Im2ColTmaDescriptor const*
  get_tma_descriptor() const
  {
    return &tma_desc_;
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  TMATensor const
  get_tma_tensor(GShape const&) const
  {
    return tma_tensor_;
  }

  /// @brief Get an executable specialization.
  ///
  /// Copy_Traits specializations with SM100_TMA_2SM_LOAD_IM2COL are not
  /// directly executable.  Instead, call this "with" member function
  /// to get an executable specialization.  "Executable" means that
  /// @c copy_unpack works.
  ///
  /// @param tma_mbar Memory barrier for synchronization
  ///
  /// @param multicast_mask Multicast mask (unused; only exists
  ///   for consistency with the actual multicast Copy_Traits
  ///   specialization)
  ///
  /// @return Executable specialization of @c Copy_Traits
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_IM2COL_OP, NumBitsPerTMA>
  with(uint64_t& tma_mbar, [[maybe_unused]] uint16_t const& multicast_mask = 0) const
  {
    return {{}, {&tma_desc_, &tma_mbar}};
  }

  // Copy_Traits specializations with SM100_TMA_2SM_LOAD_IM2COL
  // are not directly executable.  Instead, call .with
  // to get an executable specialization.
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

///   TMA load, with TMA descriptor and barrier.
template <class NumBitsPerTMA>
struct Copy_Traits<SM100_TMA_2SM_LOAD_IM2COL_OP, NumBitsPerTMA>
     : TMA_LOAD_IM2COL_Unpack<SM100_TMA_2SM_LOAD_IM2COL_OP>
{
  using ThrID = Layout<_2>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_2, NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_2, NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM100_TMA_2SM_LOAD_IM2COL arguments
  tuple<
  Im2ColTmaDescriptor const*,
  uint64_t* // smem mbarrier
  > const opargs_;
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_LOAD_MULTICAST /////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct SM100_TMA_2SM_LOAD_IM2COL_MULTICAST_OP : SM100_TMA_2SM_LOAD_IM2COL_MULTICAST {};

/// @brief Non-executable specialization of Copy_Traits for SM100
///   im2col TMA load, with TMA descriptor but no barrier or multicast
///   mask.
///
/// Use `.with(memory_barrier)` to construct an executable version.
template <class NumBitsPerTMA, class TMATensor>
struct Copy_Traits<SM100_TMA_2SM_LOAD_IM2COL_MULTICAST, NumBitsPerTMA, TMATensor>
{
  using ThrID = Layout<_2>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_2, NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_2, NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  Im2ColTmaDescriptor tma_desc_;
  TMATensor tma_tensor_;

  CUTE_HOST_DEVICE constexpr
  Im2ColTmaDescriptor const*
  get_tma_descriptor() const
  {
    return &tma_desc_;
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  TMATensor const
  get_tma_tensor(GShape const&) const
  {
    return tma_tensor_;
  }

  /// @brief Get an executable specialization.
  ///
  /// Copy_Traits specializations with SM100_TMA_2SM_LOAD_IM2COL_MULTICAST
  /// are not directly executable.  Instead, call this "with" member
  /// function to get an executable specialization.  "Executable"
  /// means that @c copy_unpack works.
  ///
  /// @param tma_mbar Memory barrier for synchronization
  ///
  /// @param multicast_mask Multicast mask (defaults to a single CTA)
  ///
  /// @return Executable specialization of @c Copy_Traits
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_IM2COL_MULTICAST_OP, NumBitsPerTMA>
  with(uint64_t& tma_mbar, uint16_t const& multicast_mask) const
  {
    return {{}, {&tma_desc_, &tma_mbar, multicast_mask}};
  }

  // Copy_Traits specializations with SM100_TMA_LOAD_IM2COL_MULTICAST
  // are not directly executable.  Instead, call .with to get an
  // executable specialization.
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

/// @brief Executable specialization of Copy_Traits for SM100 multicast
///   im2col TMA load, with TMA descriptor, barrier, and multicast mask.
template <class NumBitsPerTMA>
struct Copy_Traits<SM100_TMA_2SM_LOAD_IM2COL_MULTICAST_OP, NumBitsPerTMA>
     : TMA_LOAD_IM2COL_Unpack<SM100_TMA_2SM_LOAD_IM2COL_MULTICAST_OP>
{
  using ThrID = Layout<_2>;
  // Map from (src-thr,src-val) to bit.
  using SrcLayout = Layout<Shape<_2, NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_2, NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM100_TMA_2SM_LOAD_IM2COL_MULTICAST arguments
  tuple<
  Im2ColTmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint16_t   // multicast mask
  > const opargs_;
};

////////////////////////////////////
// Make TMA
///////////////////////////////////

#if !defined(__CUDACC_RTC__)
/** Make a CuTe CTA-collective TiledCopy for a TMA operation.
 *
 * @param CopyOp The target copy operation: SM100_TMA_2SM_LOAD
 * @param gtensor The GMEM Tensor to be involved in the TMA.
 * @param slayout The SMEM Layout to be involved in the TMA.
 * @param cluster_tile The Cluster-local tile that each Cluster will be tiling GMEM with.
 *                     This is often the cluster_tile_shape that is used to tile the GMEM:
 *                       local_tile(gtensor, cluster_tile_shape, cluster_coord)
 *                         -> Cluster-local tile of GMEM
 * @param mma The TiledMMA that defines the Cluster-Tile to Block-Tile partitioning.
 *
 * This code attempts to maximize the TMA box size. It does this by tracing
 * the SMEM "vector" -- the inverse of the smem layout -- to find the largest
 * contiguous array of smem that can be written to/from global memory given
 * the constraints that the TMA instruction imposes.
 *
 * This is accomplished by assigning "basis" strides to the GMEM to track which
 * modes of SMEM map to which modes of GMEM, then reordering the modes of GMEM according
 * to the SMEM vector, and then using those GMEM/SMEM modes to fill in the desc.
 *
 * Examples:
 */
template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Tile,
          class... Args,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST
auto
make_im2col_tma_copy_A_sm100(CopyOp                    const& copy_op,
                             Tensor<GEngine,GLayout>   const& gtensor,        // (M,K,...)
                             SLayout                   const& slayout,        // (MMA, MMA_M, MMA_K)
                             Cluster_Tile              const& cluster_tile,   // (TILE_M,TILE_N,TILE_K)
                             TiledMMA<Args...>         const& mma,
                             LowerCornerStride         const& lower_corner_whd,
                             UpperCornerStride         const& upper_corner_whd,
                             LowerPaddingStride        const& lower_padding_whd,
                             UpperPaddingStride        const& upper_padding_whd,
                             TraversalStride           const& stride_whd,
                             LowerSRTStride            const& lower_srt,
                             DilationStride            const& stride_srt,
                             TMA::DescriptorAuxParams  const& aux_params = {})
{
  constexpr int R = GLayout::rank;
  // Keep only MK modes from MNK
  auto cluster_tile_shape = append<R>(make_shape(get<0>(cluster_tile), get<2>(cluster_tile)), Int<1>{});
  auto cluster_layout = make_identity_layout(cluster_tile_shape);
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_A(cluster_layout))(_, repeat<R>(_));

  auto cta_t_vmnk_strides = [](){
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL_MULTICAST> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_IM2COL_MULTICAST>) {
      return Stride<_0,_0,_1,_0>{};                    // VMNK: Use only the N-CTAs in the Multicast
    } else
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_IM2COL>) {
      return Stride<_0,_0,_0,_0>{};                    // VMNK: Use no CTAs in Non-Multicast
    } else {
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  auto cta_t_shape = shape(mma.get_thr_layout_vmnk());
  // cta rank -> logical cta idx
  auto cta_t_map  = make_layout(cta_t_shape, compact_col_major(cta_t_shape, cta_t_vmnk_strides));

  return detail::make_tma_copy_im2col(copy_op, gtensor, slayout,
                                      cta_t_map, cta_v_tile,
                                      lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd,
                                      lower_srt, stride_srt, aux_params);
}

template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Tile,
          class... Args,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST
auto
make_im2col_tma_copy_B_sm100(CopyOp                    const& copy_op,
                             Tensor<GEngine,GLayout>   const& gtensor,        // (N,K,...)
                             SLayout                   const& slayout,        // (MMA, MMA_N, MMA_K)
                             Cluster_Tile              const& cluster_tile,   // (TILE_M,TILE_N,TILE_K)
                             TiledMMA<Args...>         const& mma,
                             LowerCornerStride         const& lower_corner_whd,
                             UpperCornerStride         const& upper_corner_whd,
                             LowerPaddingStride        const& lower_padding_whd,
                             UpperPaddingStride        const& upper_padding_whd,
                             TraversalStride           const& stride_whd,
                             LowerSRTStride            const& lower_srt,
                             DilationStride            const& stride_srt,
                             TMA::DescriptorAuxParams  const& aux_params = {})
{
  constexpr int R = GLayout::rank;
  // Keep only NK modes from MNK
  auto cluster_tile_shape = append<R>(make_shape(get<1>(cluster_tile), get<2>(cluster_tile)), Int<1>{});
  auto cluster_layout = make_identity_layout(cluster_tile_shape);
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_B(cluster_layout))(_, repeat<R>(_));

  auto cta_t_vmnk_strides = [](){
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL_MULTICAST> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_IM2COL_MULTICAST>) {
      return Stride<_0,_1,_0,_0>{};                    // VMNK: Use only the M-CTAs in the Multicast
    } else
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_IM2COL>) {
      return Stride<_0,_0,_0,_0>{};                    // VMNK: Use no CTAs in Non-Multicast
    } else {
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  auto cta_t_shape = shape(mma.get_thr_layout_vmnk());
  // cta rank -> logical cta idx
  auto cta_t_map  = make_layout(cta_t_shape, compact_col_major(cta_t_shape, cta_t_vmnk_strides));

  return detail::make_tma_copy_im2col(copy_op, gtensor, slayout,
                                      cta_t_map, cta_v_tile,
                                      lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd,
                                      lower_srt, stride_srt, aux_params);
}

/////////////////////////////////////
// Experimental Make Im2col TMA Atom
/////////////////////////////////////

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class MMA_Tiler,
          class... Args,
          class ClusterShapeVMNK,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST
auto
make_im2col_tma_atom_A_sm100(CopyOp                    const& copy_op,
                             Tensor<GEngine,GLayout>   const& gtensor,           // (M, K, ...)
                             SLayout                   const& slayout,           // (MMA, MMA_M, MMA_K, ...)
                             MMA_Tiler                 const& mma_tiler,         // (TILE_M, TILE_N, TILE_K, ...)
                             TiledMMA<Args...>         const& mma,
                             ClusterShapeVMNK          const& cluster_shape,     // (CTA_V, CTA_M, CTA_N, CTA_K)
                             LowerCornerStride         const& lower_corner_whd,
                             UpperCornerStride         const& upper_corner_whd,
                             LowerPaddingStride        const& lower_padding_whd,
                             UpperPaddingStride        const& upper_padding_whd,
                             TraversalStride           const& stride_whd,
                             LowerSRTStride            const& lower_srt,
                             DilationStride            const& stride_srt,
                             TMA::DescriptorAuxParams  const& aux_params = {})
{
  constexpr int R = GLayout::rank;
  // Keep only MK modes from MNK
  auto cluster_tile_shape = append<R>(make_shape(get<0>(mma_tiler), get<2>(mma_tiler)), Int<1>{});
  auto cluster_layout = make_identity_layout(cluster_tile_shape);
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_A(cluster_layout))(_, repeat<R>(_));

  // The size of the multicasting
  auto num_multicast = [&](){
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL_MULTICAST> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_IM2COL_MULTICAST>) {
      return size<2>(cluster_shape);                   // VMNK: Use only the N-CTAs in the Multicast
    } else
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL>  ||
                  is_same_v<CopyOp, SM90_TMA_STORE_IM2COL> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_IM2COL>) {
      return Int<1>{};                                 // VMNK: Use no CTAs in Non-Multicast
    } else {
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  return detail::make_tma_atom_im2col(copy_op, gtensor, slayout, num_multicast, cta_v_tile,
                                      lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd,
                                      stride_whd, lower_srt, stride_srt, aux_params);
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class MMA_Tiler,
          class... Args,
          class ClusterShapeVMNK,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST
auto
make_im2col_tma_atom_B_sm100(CopyOp                    const& copy_op,
                             Tensor<GEngine,GLayout>   const& gtensor,           // (N, K, ...)
                             SLayout                   const& slayout,           // (MMA, MMA_N, MMA_K, ...)
                             MMA_Tiler                 const& mma_tiler,         // (TILE_M, TILE_N, TILE_K, ...)
                             TiledMMA<Args...>         const& mma,
                             ClusterShapeVMNK          const& cluster_shape,     // (CTA_V, CTA_M, CTA_N, CTA_K)
                             LowerCornerStride         const& lower_corner_whd,
                             UpperCornerStride         const& upper_corner_whd,
                             LowerPaddingStride        const& lower_padding_whd,
                             UpperPaddingStride        const& upper_padding_whd,
                             TraversalStride           const& stride_whd,
                             LowerSRTStride            const& lower_srt,
                             DilationStride            const& stride_srt,
                             TMA::DescriptorAuxParams  const& aux_params = {})
{
  constexpr int R = GLayout::rank;
  // Keep only NK modes from MNK
  auto cluster_tile_shape = append<R>(make_shape(get<1>(mma_tiler), get<2>(mma_tiler)), Int<1>{});
  auto cluster_layout = make_identity_layout(cluster_tile_shape);
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_B(cluster_layout))(_, repeat<R>(_));

  // The size of the multicasting
  auto num_multicast = [&](){
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL_MULTICAST> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_IM2COL_MULTICAST>) {
      return size<1>(cluster_shape);                   // VMNK: Use only the M-CTAs in the Multicast
    } else
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_IM2COL>  ||
                  is_same_v<CopyOp, SM90_TMA_STORE_IM2COL> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_IM2COL>) {
      return Int<1>{};                                 // VMNK: Use no CTAs in Non-Multicast
    } else {
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  return detail::make_tma_atom_im2col(copy_op, gtensor, slayout, num_multicast, cta_v_tile,
                                           lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd,
                                           stride_whd, lower_srt, stride_srt, aux_params);
}
#endif // !defined(__CUDACC_RTC__)

} // end namespace cute
