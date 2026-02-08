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

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace cutlass {
namespace gemm {
namespace device {
using namespace cute;

// This type is only intended to demonstrate porting 2.x kernels to 3.0
template<
  class OperatorClass, class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types {
  static_assert(sizeof(ElementA) == 0, "No valid DefaultGemmConfigurationToCutlass3Types configuration exists.");
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandA;

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB;

//
// F16: 128-by-128-by-64
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpSm80_OperandA<half_t, layout::RowMajor, 8, 64>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape < _8,_64>,
                       Stride<_64, _1>>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_8>>{}));
};

/// Operand A - Column-major (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpSm80_OperandA<half_t, layout::ColumnMajor, 8, SizeK>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape <_64, _8>,
                       Stride< _1,_64>>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},
                    Layout<Shape < _8, _1>>{}));
};

// Because the F32F16 TiledMMA is A-B symmetric, we can reuse the DefaultOperands

// Operand B - Column-Major (K-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<half_t, layout::ColumnMajor, Alignment, SizeK>
     : DefaultGemm_TensorOpSm80_OperandA<half_t, layout::RowMajor,    Alignment, SizeK>
{};

// Operand B - Row-Major (N-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<half_t, layout::RowMajor,    Alignment, SizeK>
     : DefaultGemm_TensorOpSm80_OperandA<half_t, layout::ColumnMajor, Alignment, SizeK>
{};

//
// F16: 128-by-128-by-32 (small k-block)
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpSm80_OperandA<half_t, layout::RowMajor, 8, 32>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<2,3,3>{},
                Layout<Shape < _8,_32>,
                       Stride<_32, _1>>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_32,_4>,
                           Stride< _4,_1>>{},
                    Layout<Shape < _1,_8>>{}));
};

}

///////////////////////////////////////////////////////////////////////////////

// Ampere MMA F32F16
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::Sm80,
    half_t, LayoutA,
    half_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 128;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
      Tile<_32,_32,_16>>;       // 32x32x16 MMA for LDSM, 1x2x1 value group

  // A
  static constexpr int kAlignmentA = 8;
  using DefaultOperandA = detail::DefaultGemm_TensorOpSm80_OperandA<
    half_t, LayoutA, kAlignmentA, 32>;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 8;
  using DefaultOperandB = detail::DefaultGemm_TensorOpSm80_OperandB<
    half_t, LayoutB, kAlignmentB, 32>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    half_t, TagToStrideA_t<LayoutA>,
    half_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    float,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

//
// TF32: 128-by-128-by-kblock (kBlock = 16, 32)
//

/// Operand A - Row-major  (K-major) (kBlock = 32)
template <>
struct DefaultGemm_TensorOpSm80_OperandA<tfloat32_t, layout::RowMajor, 4, 32>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,2,3>{},
                Layout<Shape < _8,_32>,
                       Stride<_32, _1>>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, tfloat32_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, tfloat32_t>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));
};

/// Operand A - Row-major  (K-major) (kBlock = 16)
template <>
struct DefaultGemm_TensorOpSm80_OperandA<tfloat32_t, layout::RowMajor, 4, 16>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<2,2,3>{},
                Layout<Shape < _8,_16>,
                       Stride<_16, _1>>{}));
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, tfloat32_t>;
  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, tfloat32_t>{},
                    Layout<Shape <_32,_4>,
                           Stride< _4,_1>>{},
                    Layout<Shape < _1,_4>>{}));
};

/// Operand A - Column-major  (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpSm80_OperandA<tfloat32_t, layout::ColumnMajor, 4, SizeK>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,2,3>{},
                Layout<Shape <_32, _8>,
                       Stride< _1,_32>>{}));
  using SmemCopyAtom = Copy_Atom<UniversalCopy<tfloat32_t>, tfloat32_t>;
  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, tfloat32_t>{},
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},
                    Layout<Shape < _4, _1>>{}));
};

// Because the TF32 TiledMMA is A-B symmetric, we can reuse the DefaultOperands

// Operand B - Column-Major  (K-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<tfloat32_t, layout::ColumnMajor, Alignment, SizeK>
     : DefaultGemm_TensorOpSm80_OperandA<tfloat32_t, layout::RowMajor,    Alignment, SizeK>
{};

// Operand B - Row-Major  (N-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<tfloat32_t, layout::RowMajor,    Alignment, SizeK>
     : DefaultGemm_TensorOpSm80_OperandA<tfloat32_t, layout::ColumnMajor, Alignment, SizeK>
{};

}

///////////////////////////////////////////////////////////////////////////////

// Ampere MMA F32TF32
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::Sm80,
    tfloat32_t, LayoutA,
    tfloat32_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 128;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
      Layout<Shape<_2,_2,_1>, Stride<_2, _1, _1>>, // 2x2x1 thread group
      Tile<_32,_32,_8>>;                           // 32x32x8 MMA for LDSM, 1x2x1 value group

  // A
  static constexpr int kAlignmentA = 4;
  using DefaultOperandA = detail::DefaultGemm_TensorOpSm80_OperandA<
    tfloat32_t, LayoutA, kAlignmentA, 32>;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 4;
  using DefaultOperandB = detail::DefaultGemm_TensorOpSm80_OperandB<
    tfloat32_t, LayoutB, kAlignmentB, 32>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    tfloat32_t, TagToStrideA_t<LayoutA>,
    tfloat32_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    float,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////
template <typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::Sm80,
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _64>;
  static constexpr int ThreadCount = 128;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
      Layout<Shape<_2,_2,_1>>,   // 2x2x1 thread group
      Tile<_32,_32,_32>>;        // 16x16x32 MMA for LDSM, 1x2x1 value group

  // A (M,K)  K-major
  using SmemLayoutAtomA = decltype(
    composition(
      Swizzle<2,4,3>{},
      Layout<Shape <_16,_64>,
             Stride<_64, _1>>{}));
  static constexpr int kAlignmentA = 16;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int8_t>{},
                    Layout<Shape <_32,_4>,
                           Stride< _4,_1>>{},
                    Layout<Shape<_1,Int<kAlignmentA>>>{}));
  // LDS.32- or LDSM-based copy atom
  // using SmemCopyAtomA = Copy_Atom<DefaultCopy, uint8_t>;
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;  // LDSM works

  // B (N,K)  K-major
  using SmemLayoutAtomB = decltype(
    composition(
      Swizzle<2,4,3>{},
      Layout<Shape <_16,_64>,
             Stride<_64, _1>>{}));
  static constexpr int kAlignmentB = 16;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int8_t>{},
                    Layout<Shape <_32,_4>,
                           Stride< _4,_1>>{},
                    Layout<Shape<_1,Int<kAlignmentB>>>{}));

  // LDS.32- or LDSM-based copy atom
  // using SmemCopyAtomB = Copy_Atom<DefaultCopy, uint32_t>;
  using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;  // LDSM works

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    int8_t, TagToStrideA_t<cutlass::layout::RowMajor>,
    int8_t, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    int32_t,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////////// SIMT TWO STAGE ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, typename Layout, int ThreadCount, int ShapeM, int ShapeK>
struct DefaultGemm_Simt_OperandA;

///////////////////////////////////////////////////////////////////////////////

template <typename Element>
struct DefaultGemm_Simt_OperandA<Element, layout::ColumnMajor, 256, 128, 8>
{
  using SmemLayoutAtom = Layout<Shape <_128,  _8>,
                                Stride<  _1,_128>>;

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape<_1,_1>>{}));
};

template <typename Element>
struct DefaultGemm_Simt_OperandA<Element, layout::RowMajor, 256, 128, 8>
{
  using SmemLayoutAtom = Layout<Shape <_128,          _8>,
                                Stride<  _1,Int<128 + 4>>>;   // Padded

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                    Layout<Shape <_32, _8>,
                           Stride< _8, _1>>{},
                    Layout<Shape<_1,_1>>{}));

};

template <typename Element, typename Layout, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB;

template <typename Element, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK>
     : DefaultGemm_Simt_OperandA<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK> {};

template <typename Element, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK>
     : DefaultGemm_Simt_OperandA<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK> {};

} // end namespace detail

// SIMT Two Stage
template <
  class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _8>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm70TwoStage;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>>;

  // A
  static constexpr int kAlignmentA = 1;
  using DefaultOperandA = detail::DefaultGemm_Simt_OperandA<ElementA, LayoutA, ThreadCount, 128, 8>;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
  using SmemCopyAtomA   = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA  = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 1;
  using DefaultOperandB = detail::DefaultGemm_Simt_OperandB<ElementB, LayoutB, ThreadCount, 128, 8>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
  using SmemCopyAtomB   = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB  = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<LayoutA>,
    ElementB, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    ElementC,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;
};


//
// DP4A - int8    Proof-of-concept
//

// SIMT Two Stage TN - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm70TwoStage;
  // NOTE: permuting MMA M mode lets us generate 128b smem loads (LDS.128) but has worst case bank conflicts
  using TiledMma = TiledMMA<
      MMA_Atom<SM61_DP4A>,
      Layout<Shape<_16,_16,_1>>>;  // Tile of atoms (threads)

  // A (M,K)  K-major
  using ElementA = int8_t;
  // 40% from regular M and N major layout
  // using SmemLayoutAtomA = Layout<Shape <_128,_32>,
  //                                Stride<  _1,_128>>;
  // 80% from interleaved layouts
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;

  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 4;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint32_t>, ElementA>{},
                    Layout<Shape <_32,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));

  // B (N,K)  K-major
  using ElementB = int8_t;
  // 40% from regular M and N major layout
  // using SmemLayoutAtomB = Layout<Shape <_128,_32>,
  //                                Stride<  _1,_128>>;
  // 80% from interleaved layouts
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;

  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 4;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint32_t>, ElementB>{},
                    Layout<Shape <_32,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    ElementC,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage NN - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, cutlass::layout::ColumnMajor,
    int8_t, cutlass::layout::ColumnMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;

  using DispatchPolicy = MainloopSm70TwoStage;

  using TiledMma = TiledMMA<
      MMA_Atom<SM61_DP4A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  M-major
  using ElementA = int8_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint8_t>, ElementA>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // B (N,K)  K-major
  using ElementB = int8_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 4;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint32_t>, ElementB>{},
                    Layout<Shape <_32,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    ElementC,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage NT - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, cutlass::layout::ColumnMajor,
    int8_t, cutlass::layout::RowMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm70TwoStage;
  using TiledMma = TiledMMA<
      MMA_Atom<SM61_DP4A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  M-major
  using ElementA = int8_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint8_t>, ElementA>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // B (N,K)  N-major
  using ElementB = int8_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint8_t>, ElementB>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    ElementC,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage TT - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::RowMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm70TwoStage;
  using TiledMma = TiledMMA<
      MMA_Atom<SM61_DP4A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  K-major
  using ElementA = int8_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 4;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint32_t>, ElementA>{},
                    Layout<Shape <_32,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));

  // B (N,K)  N-major
  using ElementB = int8_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint8_t>, ElementB>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape < _1, _1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    ElementC,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// SIMT MULTI STAGE //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// SIMT Multi Stage NT
template <
  class ElementA,
  class ElementB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, arch::Sm80,
    ElementA, cutlass::layout::ColumnMajor,
    ElementB, cutlass::layout::RowMajor,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _16>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>,                            // 16x16x1 thread group
      Tile<Layout<Shape<_16,_2>,Stride<_2,_1>>,               // 32x32x1 MMA with perm for load vectorization
           Layout<Shape<_16,_2>,Stride<_2,_1>>,Underscore>>;

  // A (M,K)  M-major
  using SmemLayoutAtomA = Layout<Shape<_128,_16>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 2;
  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * kAlignmentA>;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<AlignmentTypeA>, ElementA>{},
                    Layout<Shape<_32,_8>>{},
                    Layout<Shape< _2,_1>>{}));

  // B (N,K)  N-major
  using SmemLayoutAtomB = Layout<Shape<_128,_16>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 2;
  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * kAlignmentB>;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<AlignmentTypeB>, ElementB>{},
                    Layout<Shape<_32,_8>>{},
                    Layout<Shape< _2,_1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    ElementC,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Multi Stage TN
template <
  class ElementA,
  class ElementB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, arch::Sm80,
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _16>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  K-major
  using SmemLayoutAtomA = Layout<Shape <_128,          _16>,
                                 Stride<  _1, Int<128 + 1>>>;  // Padded by kAlignmentA
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementA>, ElementA>{},
                    Layout<Shape <_16,_16>,
                           Stride<_16, _1>>{}));

  // B (N,K)  K-major
  using SmemLayoutAtomB = Layout<Shape <_128,          _16>,
                                 Stride<  _1, Int<128 + 1>>>;  // Padded by kAlignmentB
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementB>, ElementB>{},
                    Layout<Shape <_16,_16>,
                           Stride<_16, _1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    ElementC,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Multi Stage NN
template <
  class ElementA,
  class ElementB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, arch::Sm80,
    ElementA, cutlass::layout::ColumnMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _16>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>,                                      // 16x16x1 thread group
      Tile<Layout<Shape<_16,_2>,Stride<_2,_1>>,Underscore,Underscore>>; // 32x16x1 MMA with perm for load vectorization

  // A (M,K)  M-major
  using SmemLayoutAtomA = Layout<Shape<_128,_16>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 2;
  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * kAlignmentA>;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<AlignmentTypeA>, ElementA>{},
                    Layout<Shape<_32,_8>>{},
                    Layout<Shape< _2,_1>>{}));

  // B (N,K)  K-major
  using SmemLayoutAtomB = Layout<Shape <_128,          _16>,
                                 Stride<  _1, Int<128 + 1>>>;  // Padded by kAlignmentB
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementB>, ElementB>{},
                    Layout<Shape <_16,_16>,
                           Stride<_16, _1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    ElementC,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Multi Stage TT
template <
  class ElementA,
  class ElementB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassSimt, arch::Sm80,
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::RowMajor,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _16>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>,                                      // 16x16x1 thread group
      Tile<Underscore,Layout<Shape<_16,_2>,Stride<_2,_1>>,Underscore>>; // 16x32x1 MMA with perm for load vectorization

  // A (M,K)  K-major
  using SmemLayoutAtomA = Layout<Shape <_128,          _16>,
                                 Stride<  _1, Int<128 + 1>>>;  // Padded by kAlignmentA
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementA>, ElementA>{},
                    Layout<Shape <_16,_16>,
                           Stride<_16, _1>>{}));

  // B (N,K)  N-major
  using SmemLayoutAtomB = Layout<Shape <_128,_16>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 2;
  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * kAlignmentB>;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<AlignmentTypeB>, ElementB>{},
                    Layout<Shape<_32,_8>>{},
                    Layout<Shape< _2,_1>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<cutlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    ElementC,
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// Ampere fp64 MMA TN (K-Major A and K-Major B)
template <>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::Sm80,
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
    double>
{
  using TileShape = Shape<_128, _64, _16>;
  static constexpr int ThreadCount = 128;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,            // Atom
      Layout<Shape<_2,_2,_1>>,                         // Atom layout
      Tile<Layout<Shape<_16,_2>,Stride<_2,_1>>,        // 32x32x4 MMA with perm for load vectorization
           Layout<Shape<_16,_2>,Stride<_2,_1>>,
           Underscore>>;

  // A  (M,K)  K-Major
  using SmemLayoutAtomA = decltype(
      composition(Swizzle<2,0,4>{},
                  Layout<Shape <_4,_16>,
                         Stride<_1, _4>>{})); // M, K
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, double>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<double>, double>{}, // CopyAtom
                    Layout<Shape < _8,_16>,
                           Stride<_16, _1>>{},                           // ThrLayout for CopyAtom
                    Layout<Shape<_1,_1>>{}));                            // Value layout: 1x1 doubles

  // B  (N,K)  K-Major
  using SmemLayoutAtomB = decltype(
      composition(Swizzle<2,0,4>{},
                  Layout<Shape <_4,_16>,
                         Stride<_1, _4>>{})); // N, K
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, double>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<double>, double>{}, // CopyAtom
                    Layout<Shape < _8,_16>,
                           Stride<_16, _1>>{},                           // ThrLayout for CopyAtom
                    Layout<Shape<_1,_1>>{}));                            // Value layout: 1x1 doubles

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    double, TagToStrideA_t<cutlass::layout::RowMajor>,
    double, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    double,
    TagToStrideC_t<cutlass::layout::ColumnMajor>,
    TagToStrideC_t<cutlass::layout::ColumnMajor>,
    epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::EpilogueDefault>;

/*
  using EpilogueOutputOp = epilogue::collective::Epilogue<
      epilogue::thread::LinearCombination<double, 1, double, double>,
      Layout<Shape <_64,_32>,
             Stride< _1,_64>>,                                           // SMEM layout
      Copy_Atom<UniversalCopy<double>,double>,                           // R2S with tiled_mma layout
      decltype(make_tiled_copy(Copy_Atom<UniversalCopy<double>,double>{},// S2R
                               Layout<Shape <_16,_16>,
                                      Stride< _1,_16>>{},                // Thread layout
                               Layout<Shape<_2,_1>>{})),                 // Value layout
      Copy_Atom<UniversalCopy<double>,double>                            // R2G with S2R_dst layout
      >;
*/
};

///////////////////////////////////////////////////////////////////////////////

// Ampere fp64 MMA NN (M-Major A and K-Major B)
template <>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::Sm80,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
    double>
{
  using TileShape = Shape<_128, _64, _16>;
  static constexpr int ThreadCount = 128;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,            // Atom
      Layout<Shape<_2,_2,_1>>,                         // Atom layout
      Tile<Layout<Shape<_16,_2>,Stride<_2,_1>>,        // 32x32x4 MMA with perm for load vectorization
           Layout<Shape<_16,_2>,Stride<_2,_1>>,
           Underscore>>;

  // A  (M,K)  M-Major
  using SmemLayoutAtomA = decltype(
      composition(Swizzle<2,2,2>{},
                  Layout<Shape <_16, _4>,
                         Stride< _1,_16>>{})); // M, K
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, double>;
  static constexpr int kAlignmentA = 2;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{}, // CopyAtom
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},                           // ThrLayout for CopyAtom
                    Layout<Shape<_2,_1>>{}));                            // Value layout: 2x1 doubles

  // B  (N,K)  K-Major
  using SmemLayoutAtomB = decltype(
      composition(Swizzle<2,0,4>{},
                  Layout<Shape <_4,_16>,
                         Stride<_1, _4>>{}));// N, K
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, double>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<double>, double>{}, // CopyAtom
                    Layout<Shape < _8,_16>,
                           Stride<_16, _1>>{},                           // ThrLayout for CopyAtom
                    Layout<Shape<_1,_1>>{}));                            // Value layout: 1x1 doubles

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    double, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    double, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    double,
    TagToStrideC_t<cutlass::layout::ColumnMajor>,
    TagToStrideC_t<cutlass::layout::ColumnMajor>,
    epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// Ampere fp64 MMA NT (M-Major A and N-Major B)
template <>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::Sm80,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::ColumnMajor,
    double>
{
  using TileShape = Shape<_128, _64, _16>;
  static constexpr int ThreadCount = 128;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,            // Atom
      Layout<Shape<_2,_2,_1>>,                         // Atom layout
      Tile<Layout<Shape<_16,_2>,Stride<_2,_1>>,        // 32x32x4 MMA with perm for load vectorization
           Layout<Shape<_16,_2>,Stride<_2,_1>>,
           Underscore>>;

  // A  (M,K)  M-Major
  using SmemLayoutAtomA = decltype(
      composition(Swizzle<2,2,2>{},
                  Layout<Shape <_16, _4>,
                         Stride< _1,_16>>{})); // M, K
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, double>;
  static constexpr int kAlignmentA = 2;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{}, // CopyAtom
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},                           // ThrLayout for CopyAtom
                    Layout<Shape<_2,_1>>{}));                            // Value layout: 2x1 doubles

  // B  (N,K)  N-Major
  using SmemLayoutAtomB = decltype(
      composition(Swizzle<2,2,2>{},
                  Layout<Shape <_16, _4>,
                         Stride< _1,_16>>{})); // N, K
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, double>;
  static constexpr int kAlignmentB = 2;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{}, // CopyAtom
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},                           // ThrLayout for CopyAtom
                    Layout<Shape<_2,_1>>{}));                            // Value layout: 2x1 doubles

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    double, TagToStrideA_t<cutlass::layout::ColumnMajor>,
    double, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    double,
    TagToStrideC_t<cutlass::layout::ColumnMajor>,
    TagToStrideC_t<cutlass::layout::ColumnMajor>,
    epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// Ampere fp64 MMA TT (K-Major A and N-Major B)
template <>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::Sm80,
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::ColumnMajor,
    double>
{
  using TileShape = Shape<_128, _64, _16>;
  static constexpr int ThreadCount = 128;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,            // Atom
      Layout<Shape<_2,_2,_1>>,                         // Atom layout
      Tile<Layout<Shape<_16,_2>,Stride<_2,_1>>,        // 32x32x4 MMA with perm for load vectorization
           Layout<Shape<_16,_2>,Stride<_2,_1>>,
           Underscore>>;

  // A  (M,K)  K-Major
  using SmemLayoutAtomA = decltype(
      composition(Swizzle<2,0,4>{},
                  Layout<Shape <_4,_16>,
                         Stride<_1, _4>>{})); // M, K
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, double>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<double>, double>{}, // CopyAtom
                    Layout<Shape < _8,_16>,
                           Stride<_16, _1>>{},                           // ThrLayout for CopyAtom
                    Layout<Shape<_1,_1>>{}));                            // Value layout: 1x1 doubles

  // B  (N,K)  N-Major
  using SmemLayoutAtomB = decltype(
      composition(Swizzle<2,2,2>{},
                  Layout<Shape <_16, _4>,
                         Stride< _1,_16>>{})); // N, K
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, double>;
  static constexpr int kAlignmentB = 2;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{}, // CopyAtom
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},                           // ThrLayout for CopyAtom
                    Layout<Shape<_2,_1>>{}));                            // Value layout: 2x1 doubles

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    double, TagToStrideA_t<cutlass::layout::RowMajor>,
    double, TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    double,
    TagToStrideC_t<cutlass::layout::ColumnMajor>,
    TagToStrideC_t<cutlass::layout::ColumnMajor>,
    epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// Hopper fp64 MMA TN
template <>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::Sm90,
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
    double>
{
  using TileShape = Shape<_128, _64, _16>;
  static constexpr int ThreadCount = 128;
  using DispatchPolicy = MainloopSm80CpAsync<3>;
  using TiledMma = TiledMMA<
      MMA_Atom<SM90_16x8x16_F64F64F64F64_TN>,
      Layout<Shape<_2,_2,_1>>>;

  // A (M,K)  K-major
  using SmemLayoutAtomA = decltype(
    make_ordered_layout(Shape<_128,_16>{},
                        Step <  _2, _1>{})); // M, K
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, double>;
  static constexpr int kAlignmentA = 2;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_2>>{}));

  // B (N,K)  K-major
  using SmemLayoutAtomB = decltype(
    make_ordered_layout(Shape<_64,_16>{},
                        Step < _2, _1>{}));                       // N, K
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, double>;
  static constexpr int kAlignmentB = 2;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_2>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    double, TagToStrideA_t<cutlass::layout::RowMajor>,
    double, TagToStrideB_t<cutlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, Shape<_1,_1,_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    double, double,
    double, cutlass::layout::ColumnMajor, 1,
    double, cutlass::layout::ColumnMajor, 1,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

};

///////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass
