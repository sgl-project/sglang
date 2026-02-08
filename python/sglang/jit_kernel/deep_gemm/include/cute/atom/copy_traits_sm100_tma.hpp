/***************************************************************************************************
 * Copyright (c) 2021 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if !defined(__CUDACC_RTC__)
#include <cuda.h>
#endif

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/atom/copy_traits.hpp>

namespace cute
{

//////////////////////////////////////////////////////////////////////////////
////////////////////////////// TMA_LOAD ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct SM100_TMA_2SM_LOAD_OP : SM100_TMA_2SM_LOAD {};

// The non-executable SM100_TMA_2SM_LOAD with tma_desc and no tma_mbar
// Use .with(tma_mbar) to construct an executable version
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM100_TMA_2SM_LOAD, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_2>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_2,NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_2,NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM100_TMA_2SM_LOAD arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Construct an executable SM100_TMA_2SM_LOAD with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_OP, NumBitsPerTMA>
  with(
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm100 const& cache_hint = TMA::CacheHintSm100::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {{}, {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)}};
  }

  // Construct an executable SM100_TMA_2SM_LOAD with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_OP, NumBitsPerTMA>
  with(
    TmaDescriptor const* new_tma_desc,
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm100 const& cache_hint = TMA::CacheHintSm100::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {{}, {new_tma_desc, &tma_mbar, static_cast<uint64_t>(cache_hint)}};
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // Don't try to execute a copy with SM100_TMA_2SM_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

// The executable SM100_TMA_2SM_LOAD with tma_desc and tma_mbar
template <class NumBitsPerTMA>
struct Copy_Traits<SM100_TMA_2SM_LOAD_OP, NumBitsPerTMA>
     : TMA_LOAD_Unpack<SM100_TMA_2SM_LOAD_OP, NumBitsPerTMA>
{
  using ThrID     = Layout<_2>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_2,NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_2,NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM100_TMA_2SM_LOAD arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint64_t   // cache hint
  > const opargs_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return get<0>(opargs_);
  }
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_LOAD_MULTICAST /////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct SM100_TMA_2SM_LOAD_MULTICAST_OP : SM100_TMA_2SM_LOAD_MULTICAST {};

template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM100_TMA_2SM_LOAD_MULTICAST, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_2>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_2,NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_2,NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM100_TMA_2SM_LOAD_MULTICAST_OP arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Construct an executable SM100_TMA_2SM_LOAD_MULTICAST_OP with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_MULTICAST_OP, NumBitsPerTMA>
  with(
    uint64_t& tma_load_mbar,
    uint16_t const& multicast_mask,
    TMA::CacheHintSm100 const& cache_hint = TMA::CacheHintSm100::EVICT_NORMAL) const {
    return {{}, {&tma_desc_, &tma_load_mbar, multicast_mask, static_cast<uint64_t>(cache_hint)}};
  }

  // Construct an executable SM100_TMA_2SM_LOAD_MULTICAST_OP with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM100_TMA_2SM_LOAD_MULTICAST_OP, NumBitsPerTMA>
  with(
    TmaDescriptor const* new_tma_desc,
    uint64_t& tma_load_mbar,
    uint16_t const& multicast_mask,
    TMA::CacheHintSm100 const& cache_hint = TMA::CacheHintSm100::EVICT_NORMAL) const {
    return {{}, {new_tma_desc, &tma_load_mbar, multicast_mask, static_cast<uint64_t>(cache_hint)}};
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // Don't try to execute a copy with SM100_TMA_2SM_LOAD_MULTICAST_OP before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template <class NumBitsPerTMA>
struct Copy_Traits<SM100_TMA_2SM_LOAD_MULTICAST_OP, NumBitsPerTMA>
     : TMA_LOAD_Unpack<SM100_TMA_2SM_LOAD_MULTICAST_OP, NumBitsPerTMA>
{
  using ThrID     = Layout<_2>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_2,NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_2,NumBitsPerTMA>, Stride<NumBitsPerTMA,_1>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM100_TMA_2SM_LOAD_MULTICAST_OP arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint16_t,  // multicast mask
  uint64_t   // cache hint
  > const opargs_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return get<0>(opargs_);
  }
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
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Tiler,
          class... Args>
CUTE_HOST
auto
make_tma_copy_A_sm100(CopyOp                  const& copy_op,
                      Tensor<GEngine,GLayout> const& gtensor,        // (M, K, ...)
                      SLayout                 const& slayout,        // (MMA, MMA_M, MMA_K, ...)
                      Cluster_Tiler           const& cluster_tiler,  // (TILER_M, TILER_N, TILER_K, ...)
                      TiledMMA<Args...>       const& mma)
{
  // Keep only MK modes from MNK
  auto cluster_tiler_mk = remove<1>(cluster_tiler);
  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(cluster_tiler_mk);     // (TILE_M, TILE_K, ...)
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_A(g_tile))(_, repeat<rank(g_tile)>(_));    // (MMA, MMA_M, MMA_K, ...)

  auto cta_t_vmnk_strides = [](){
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_MULTICAST> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_MULTICAST>) {
      return Stride<_0,_0,_1,_0>{};                    // VMNK: Use only the N-CTAs in the Multicast
    } else
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD>  ||
                  is_same_v<CopyOp, SM90_TMA_STORE> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD>) {
      return Stride<_0,_0,_0,_0>{};                    // VMNK: Use no CTAs in Non-Multicast
    } else {
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  auto cta_t_shape = shape(mma.get_thr_layout_vmnk());
  // cta rank -> logical cta idx
  auto cta_t_map  = coalesce(make_layout(cta_t_shape, compact_col_major(cta_t_shape, cta_t_vmnk_strides)));

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_map, cta_v_tile);
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Tiler,
          class... Args>
CUTE_HOST
auto
make_tma_copy_B_sm100(CopyOp                  const& copy_op,
                      Tensor<GEngine,GLayout> const& gtensor,        // (N, K, ...)
                      SLayout                 const& slayout,        // (MMA, MMA_N, MMA_K, ...)
                      Cluster_Tiler           const& cluster_tiler,  // (TILE_M, TILE_N, TILE_K, ...)
                      TiledMMA<Args...>       const& mma)
{
  // Keep only NK modes from MNK
  auto cluster_tiler_nk = remove<0>(cluster_tiler);
  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(cluster_tiler_nk);     // (TILE_N, TILE_K, ...)
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_B(g_tile))(_, repeat<rank(g_tile)>(_));    // (MMA, MMA_N, MMA_K, ...)

  auto cta_t_vmnk_strides = [](){
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_MULTICAST> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_MULTICAST>) {
      return Stride<_0,_1,_0,_0>{};                    // VMNK: Use only the M-CTAs in the Multicast
    } else
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD>  ||
                  is_same_v<CopyOp, SM90_TMA_STORE> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD>) {
      return Stride<_0,_0,_0,_0>{};                    // VMNK: Use no CTAs in Non-Multicast
    } else {
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  auto cta_t_shape = shape(mma.get_thr_layout_vmnk());
  // cta rank -> logical cta idx
  auto cta_t_map  = coalesce(make_layout(cta_t_shape, compact_col_major(cta_t_shape, cta_t_vmnk_strides)));

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_map, cta_v_tile);
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Tiler,
          class... Args>
CUTE_HOST
auto
make_tma_copy_C_sm100(CopyOp                  const& copy_op,
                      Tensor<GEngine,GLayout> const& gtensor,        // (M, N, ...)
                      SLayout                 const& slayout,        // (MMA, MMA_M, MMA_N, ...)
                      Cluster_Tiler           const& cluster_tiler,  // (TILE_M, TILE_N, TILE_K, ...)
                      TiledMMA<Args...>       const& mma)
{
  // Keep only MN modes from MNK
  auto cluster_tiler_mn = remove<2>(cluster_tiler);
  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(cluster_tiler_mn);     // (TILE_M, TILE_N, ...)
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_C(g_tile))(_, repeat<rank(g_tile)>(_));    // (MMA, MMA_M, MMA_N, ...)

  static_assert(is_same_v<CopyOp, SM90_TMA_LOAD>  ||
                is_same_v<CopyOp, SM90_TMA_STORE> ||
                is_same_v<CopyOp, SM100_TMA_2SM_LOAD>,
                "Unsupported TMA Op, expected a non-multicast TMA");

  // No multicast, so only 1 CTA involved
  auto cta_t_map = Layout<_1,_0>{};

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_map, cta_v_tile);
}

////////////////////////////////////
// Experimental Make TMA Atom
///////////////////////////////////

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class MMA_Tiler,
          class... Args,
          class ClusterShapeVMNK>
CUTE_HOST
auto
make_tma_atom_A_sm100(CopyOp                  const& copy_op,
                      Tensor<GEngine,GLayout> const& gtensor,        // (M, K, ...)
                      SLayout                 const& slayout,        // (MMA, MMA_M, MMA_K, ...)
                      MMA_Tiler               const& mma_tiler,      // (TILE_M, TILE_N, TILE_K, ...)
                      TiledMMA<Args...>       const& mma,
                      ClusterShapeVMNK        const& cluster_shape)  // (CTA_V, CTA_M, CTA_N, CTA_K)
{
  // Keep only MK modes from MNK
  auto mma_tiler_mk = remove<1>(mma_tiler);

  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(mma_tiler_mk);         // (TILE_M, TILE_K, ...)

  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_A(g_tile))(_, repeat<rank(g_tile)>(_));    // (MMA, MMA_M, MMA_K, ...)

#if 0
  print("(tma_a) slayout:      "); print(slayout);      print("\n");
  print("(tma_a) mma_tiler_nk: "); print(mma_tiler_nk); print("\n");
  print("(tma_a) g_tile:       "); print(g_tile);       print("\n");
  print("(tma_a) mma_tiler:    "); print(mma_tiler);    print("\n");
  print("(tma_a) cta_v_tile:   "); print(cta_v_tile);   print("\n");
#endif

  // The size of the multicasting
  auto num_multicast = [&](){
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_MULTICAST> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_MULTICAST>) {
      return size<2>(cluster_shape);                   // VMNK: Use only the N-CTAs in the Multicast
    } else
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD>  ||
                  is_same_v<CopyOp, SM90_TMA_STORE> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD>) {
      return Int<1>{};                                 // VMNK: Use no CTAs in Non-Multicast
    } else {
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_atom<TmaType>(copy_op, gtensor, slayout, num_multicast, cta_v_tile);
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class MMA_Tiler,
          class... Args,
          class ClusterShapeVMNK>
CUTE_HOST
auto
make_tma_atom_B_sm100(CopyOp                  const& copy_op,
                      Tensor<GEngine,GLayout> const& gtensor,        // (N, K, ...)
                      SLayout                 const& slayout,        // (MMA, MMA_N, MMA_K, ...)
                      MMA_Tiler               const& mma_tiler,      // (TILE_M, TILE_N, TILE_K, ...)
                      TiledMMA<Args...>       const& mma,
                      ClusterShapeVMNK        const& cluster_shape)  // (CTA_V, CTA_M, CTA_N, CTA_K)
{
  // Keep only NK modes from MNK
  auto mma_tiler_nk = remove<0>(mma_tiler);
  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(mma_tiler_nk);         // (TILE_N, TILE_K, ...)
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_B(g_tile))(_, repeat<rank(g_tile)>(_));    // (MMA, MMA_N, MMA_K, ...)

#if 0
  print("(tma_b) slayout:      "); print(slayout);      print("\n");
  print("(tma_b) mma_tiler_nk: "); print(mma_tiler_nk); print("\n");
  print("(tma_b) g_tile:       "); print(g_tile);       print("\n");
  print("(tma_b) mma_tiler:    "); print(mma_tiler);    print("\n");
  print("(tma_b) cta_v_tile:   "); print(cta_v_tile);   print("\n");
#endif

  // The size of the multicasting
  auto num_multicast = [&](){
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD_MULTICAST> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_MULTICAST>) {
      return size<1>(cluster_shape);                   // VMNK: Use only the M-CTAs in the Multicast
    } else
    if constexpr (is_same_v<CopyOp, SM90_TMA_LOAD>  ||
                  is_same_v<CopyOp, SM90_TMA_STORE> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD>) {
      return Int<1>{};                                 // VMNK: Use no CTAs in Non-Multicast
    } else {
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_atom<TmaType>(copy_op, gtensor, slayout, num_multicast, cta_v_tile);
}

#endif // !defined(__CUDACC_RTC__)

} // end namespace cute
