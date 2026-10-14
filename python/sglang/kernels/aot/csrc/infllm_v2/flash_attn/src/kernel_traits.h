/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"

using namespace cute;

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = cutlass::half_t>
struct Flash_kernel_traits {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using Element = elem_type;
  static constexpr bool Has_cp_async = true;
#else
  using Element = cutlass::half_t;
  static constexpr bool Has_cp_async = false;
#endif

  using ElementAccum = float;
  using index_t = int64_t;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using MMA_Atom_Arch = std::conditional_t<
      std::is_same_v<elem_type, cutlass::half_t>,
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
#else
  using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
  using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
  using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    bool Is_Q_in_regs_ = false,
    bool Share_Q_K_smem_ = false,
    typename elem_type = cutlass::half_t,
    typename Base = Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Flash_fwd_kernel_traits : public Base {
  using Element = typename Base::Element;
  using ElementAccum = typename Base::ElementAccum;
  using index_t = typename Base::index_t;
  static constexpr bool Has_cp_async = Base::Has_cp_async;
  using SmemCopyAtom = typename Base::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

  static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
  static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

  // The number of threads.
  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * 32;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

  using TiledMma = TiledMMA<
      typename Base::MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,  // 4x1x1 or 8x1x1 thread group
      Tile<Int<16 * kNWarps>, _16, _16>>;

  using SmemLayoutAtomQ = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

  // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
  using SmemLayoutVtransposed =
      decltype(composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  using SmemLayoutAtomO = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{}, Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
  using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;

  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
  // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
  // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
  // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
  // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
  // to the same banks.
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
  static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom =
      Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>;

  // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
  // from the same address by the same threadblock. This is slightly faster.
  using Gmem_copy_struct = std::conditional_t<
      Has_cp_async,
      SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
      AutoVectorizingCopyWithAssumedAlignment<128>>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

  using GmemLayoutAtomOaccum = std::conditional_t<
      kBlockKSmem == 32,
      Layout<
          Shape<_16, _8>,  // Thread layout, 8 threads per row
          Stride<_8, _1>>,
      Layout<
          Shape<_8, _16>,  // Thread layout, 16 threads per row
          Stride<_16, _1>>>;
  using GmemTiledCopyOaccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      GmemLayoutAtomOaccum{},
      Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per store
  using GmemLayoutAtomRotcossin = GmemLayoutAtom;
  using GmemTiledCopyRotcossin = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<uint64_t>, Element>{},
      GmemLayoutAtomRotcossin{},
      Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per load
  using GmemTiledCopyRotcossinCont = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      GmemLayoutAtomRotcossin{},
      Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per load
};

////////////////////////////////////////////////////////////////////////////////////////////////////
