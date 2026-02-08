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

#include "cute/config.hpp"                     // CUTE_STATIC_ASSERT
#include "cute/layout.hpp"                     // cute::Layout, cute::Shape, cute::Stride
#include "cute/numeric/integral_constant.hpp"  // cute::Int
#include "cute/numeric/numeric_types.hpp"      // cute::sizeof_bits_v
#include "cute/pointer_sparse.hpp"             // cute::is_sparse
#include "cute/util/type_traits.hpp"           // cute::is_same_v, cute::conditional_t
#include "cutlass/fast_math.h"                 // cutlass::round_up
#include "cutlass/layout/matrix.h"             // cutlass::layout::RowMajor

namespace cutlass {

using namespace cute;

template<
  class ElementAMma_,
  class LayoutA,
  class ElementEMma_
>
struct Sm1xxGemmSparseConfig {
  /// ElementAMma Check
  static_assert(cute::is_sparse<ElementAMma_>::value, "ElementAMma MUST be sparse elem");
  static_assert(cute::is_sparse<ElementEMma_>::value, "ElementEMma MUST be sparse elem");

  /// A
  using ElementAMma         = ElementAMma_;
  using ElementAMmaRaw      = typename ElementAMma::raw_type;
  using ElementAMmaSparsity = Int<ElementAMma::sparsity>;

  /// MetaData (E)
  using ElementEMma         = ElementEMma_;
  using ElementEMmaRaw      = typename ElementEMma::raw_type;
  using ElementEMmaSparsity = Int<ElementEMma::sparsity>;

  /// Instruction Type
  static constexpr bool IsF4 =(cute::is_same_v<ElementAMmaRaw, uint8_t> && ElementAMmaSparsity{} == _4{}) &&
                              (cute::is_same_v<ElementEMmaRaw, uint8_t> && ElementEMmaSparsity{} == _16{});
  static constexpr bool IsF8F6F4 =(cute::is_same_v<ElementAMmaRaw, detail::float_e2m1_unpacksmem_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, detail::type_erased_dynamic_float4_unpacksmem_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, detail::float_e3m2_unpacksmem_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, detail::float_e2m3_unpacksmem_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, detail::type_erased_dynamic_float6_unpacksmem_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, float_e2m1_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, type_erased_dynamic_float4_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, float_e3m2_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, float_e2m3_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, type_erased_dynamic_float6_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, float_e4m3_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, float_e5m2_t> && ElementAMmaSparsity{} == _2{} ||
                                   cute::is_same_v<ElementAMmaRaw, type_erased_dynamic_float8_t> && ElementAMmaSparsity{} == _2{}) &&
                                  (cute::is_same_v<ElementEMmaRaw, uint8_t> && ElementEMmaSparsity{} == _8{});
  static constexpr bool IsI8   =(cute::is_same_v<ElementAMmaRaw, int8_t> && ElementAMmaSparsity{} == _2{}) &&
                                (cute::is_same_v<ElementEMmaRaw, uint8_t> && ElementEMmaSparsity{} == _8{});
  static constexpr bool IsF16BF16 =(cute::is_same_v<ElementAMmaRaw, half_t> && ElementAMmaSparsity{} == _2{} ||
                                    cute::is_same_v<ElementAMmaRaw, bfloat16_t> && ElementAMmaSparsity{} == _2{}) &&
                                   (cute::is_same_v<ElementEMmaRaw, uint8_t> && ElementEMmaSparsity{} == _8{});
  static constexpr bool IsTF32 =(cute::is_same_v<ElementAMmaRaw, tfloat32_t> && ElementAMmaSparsity{} == _2{} || 
                                 cute::is_same_v<ElementAMmaRaw, float> && ElementAMmaSparsity{} == _2{}) &&
                                (cute::is_same_v<ElementEMmaRaw, uint8_t> && ElementEMmaSparsity{} == _4{});
  static_assert(int(IsF4) + int(IsF8F6F4) + int(IsI8) + int(IsF16BF16) + int(IsTF32) == 1, "Ambiguous Input Type Config (failed to choose Instruction type)");
  static constexpr bool IsF4F6 = IsF8F6F4 && not 
                                    (cute::is_same_v<ElementAMmaRaw, float_e4m3_t> && ElementAMmaSparsity{} == _2{} ||
                                     cute::is_same_v<ElementAMmaRaw, float_e5m2_t> && ElementAMmaSparsity{} == _2{} ||
                                     cute::is_same_v<ElementAMmaRaw, type_erased_dynamic_float8_t> && ElementAMmaSparsity{} == _2{});

  /// Number of ElementARaw stored in ElementAMmaRaw
  using ElemsARawPerElementAMmaRaw = cute::conditional_t<IsF4, _2, _1>;

  /// ElementA Sparsity Ratio
  using ElementASparsity = _2;
  static_assert(ElementASparsity{} == _2{}, "ElementASparsity must be 2 for Blackwell Sparse Gemm");

  // Logical/Physical ElementA per Chunk
  using LogicalElemsAPerChunk = cute::conditional_t<IsF4, _8,
                                cute::conditional_t<IsTF32, _2,
                                _4>>;
  using PhysicalElemsAPerChunk = Int<LogicalElemsAPerChunk{} / ElementASparsity{}>;

  /// Metadata Bits
  using ElementEBitsPerChunk = _4;
  using ElementEBitsPerElementAMma = cute::conditional_t<IsTF32, _4, _2>;

  /// Metadata Layout
  using TensorEAtom_MMA_F4 = Layout<Shape<_128,  _256>,
                                  Stride<_256, _1>>;

  using TensorEAtom_MMA_S8orF8F6F4 = Layout<Shape<_128, _128>,
                                     Stride<_128, _1>>;

  using TensorEAtom_MMA_F16 =  Layout<Shape <Shape <  _8, _2,   _8>, Shape <_16,   _2, _4>>,
                                   Stride<Stride<_128,_16,_2048>, Stride< _1,_1024,_32>>>;

  using TensorEAtom_MMA_TF32 =  Layout<Shape <Shape < _8,_2,   _8>, Shape <_8,  _2,_4>>,
                                    Stride<Stride<_64,_8,_1024>, Stride<_1,_512,_16>>>;

  using TensorEAtom = cute::conditional_t<(IsF8F6F4 || IsI8), TensorEAtom_MMA_S8orF8F6F4,
                      cute::conditional_t<IsF4, TensorEAtom_MMA_F4,
                      cute::conditional_t<IsTF32, TensorEAtom_MMA_TF32,
                      TensorEAtom_MMA_F16>>>;

  // Logical elems that construct the atomK for tensorE/A.  
  using TensorEAtomK = Int<size<1>(TensorEAtom{})>;
  using TensorEAtomM = Int<size<0>(TensorEAtom{})>;

  /// TensorA TensorE Alignment Requirements
  using TensorEAlignmentM = TensorEAtomM;
  using TensorEAlignmentK = TensorEAtomK;
  static_assert(TensorEAlignmentK{} / ElementEMmaSparsity{} == _16{}, "TensorE must be 16B aligned in the K-dim.");

  // When A is MN major, TensorAAlignmentK needs to be multiplier of chunk size
  // When A is K major, TensorAAlignmentK needs to be multiplier of TMA requirements times tensorA sparsity
  //   this is b.c. TensorACompressed needs to satisfy TMA requirements.
  //   (LogicalElemsAPerChunk is always smaller than TMA in this case.)
  // NOTE: TensorAAlignmentK already contains the 2x sparsity factor when k-major
  static constexpr bool IsKMajor = cute::is_same_v<LayoutA, cutlass::layout::RowMajor>;
  using TensorAAlignmentK = cute::conditional_t<not IsKMajor,
                                                LogicalElemsAPerChunk,
                                                cute::conditional_t<IsF4F6,
                                                                    Int<128 * ElementASparsity{}>,
                                                                    Int<128 / cute::sizeof_bits_v<ElementAMma>>>>;

  // When A is MN Major, TensorAAlignmentM needs to be multiplier of TMA requirements
  // When A is K Major, no requirements on TensorAAlignmentM.
  using TensorAAlignmentM = cute::conditional_t<not IsKMajor,
                                                cute::conditional_t<IsF4F6,
                                                         Int<128>,
                                                         Int<128 / cute::sizeof_bits_v<ElementAMmaRaw> * ElemsARawPerElementAMmaRaw{}>>,
                                                _1>;

  // The following two functions are provided for users to determine the static layout types
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutA() {
    using LayoutMMajor = Layout<Shape<int32_t,
                                      Shape<ElementASparsity, int32_t>,
                                      int32_t>,
                                Stride<ElementASparsity,
                                       Stride<_1, int64_t>,
                                       int64_t>>;

    using LayoutKMajor = Layout<Shape<int32_t,
                                      Shape<ElementASparsity, int32_t>,
                                      int32_t>,
                                Stride<int64_t,
                                       Stride<_1, ElementASparsity>,
                                       int64_t>>;

    if constexpr (IsKMajor) {
      return LayoutKMajor{};
    }
    else {
      return LayoutMMajor{};
    }
  }

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutE() {
    return make_layout(
      make_shape(make_shape(shape<0>(TensorEAtom{}), int32_t(0)),
                 make_shape(shape<1>(TensorEAtom{}), int32_t(0)),
                 int32_t(0)),
      make_stride(make_stride(stride<0>(TensorEAtom{}), cute::Int<cute::cosize(TensorEAtom{})>{}),
                  make_stride(stride<1>(TensorEAtom{}), int64_t(0)),
                  int64_t(0))
    );
  }

  // This function is used to revert a CuTe layout to a Cutlass layout tag (RowMajor/ColumnMajor)
  template <
    typename ShapeA,
    typename StrideA
  >
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutA_tag(Layout<ShapeA, StrideA> layout_a) {
    /*
      (m, (2, k/2), l) : (2, (1, m*2), m*k) M-major
      (m, (2, k/2), l) : (k, (1, 2), m*k) K-major
    */
    // Check if the given layout_a is possibly a sparse tensorA layout.
    static_assert(rank_v<ShapeA> == 3 && depth_v<ShapeA> == 2, "Rank and depth mismatch with the sparse tensorA's layout.");
    static_assert(rank(get<1>(ShapeA{})) == 2 && rank(flatten(ShapeA{})) == 4,
                  "Not likely to be a sparse tensorA's layout.");
    static_assert(get<1,0>(StrideA{}) == 1 && get<1,0>(ShapeA{}) == ElementASparsity{},
                  "Not likely to be a sparse tensorA's layout.");
    static_assert(get<0>(StrideA{}) == ElementASparsity{} || get<1,1>(StrideA{}) == ElementASparsity{},
                  "Not likely to be a sparse tensorA's layout.");

    if constexpr (get<0>(StrideA{}) == ElementASparsity{}) {
      return cutlass::layout::ColumnMajor{};
    }
    else {
      return  cutlass::layout::RowMajor{};
    }
  }

  // The following two functions are provided for user fill dynamic problem size to the layout_a/e.
  template <
    class ProblemShape
  >
  CUTE_HOST_DEVICE
  static constexpr auto
  fill_layoutA(ProblemShape problem_shape) {
    // * Purpose of this function
    // This function is sparse gemm equivalent of 
    //
    //   ```cpp
    //   using LayoutATag = cutlass::layout::RowMajor;
    //   using StrideA = cutlass::gemm::TagToStrideA_t<GmemLayoutBTag>; // ( cute::Stride<int64_t, cute::Int<1>, int64_t> )
    //   auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L)); // (M, cute::Int<1>, L)
    //   auto layout_a = cute::make_layout(cute::make_shape(M, K, L), stride_a);
    //   ```
    //
    // Unlike dense gemm where we can simply call `TagToStrideA_t` resp. `make_cute_packed_stride`
    // to get the shape and stride, sparse gemm needs to consider the cute::sparse_elem<> representation.
    // Thus, it's easier to construct the layout directly.
    //
    // * NOTE
    // 1. Returned layout should be used with `cute::sparse_elem<>` pointer, instead of raw element A ptr
    // 2. `TensorAAlignmentK` already include 2x sparsity factor along K dim.
    //    e.g. MMA FP4 A is K major, TensorAAlignmentK = 256. When used with `cute::sparse_elem<2, f4>`, TensorAComp will
    //         have 128 element alignment along k-dim
    const auto [M, N, K, L] = problem_shape;

    // Round up to satisfy TensorA Alignment requirement
    const auto M_AlignedAC = cutlass::round_up(M, TensorAAlignmentM{});
    const auto K_AlignedAC = cutlass::round_up(K, TensorAAlignmentK{});

    if constexpr (IsKMajor) {
      return make_layout(
        make_shape(int32_t(M_AlignedAC),
                   make_shape(ElementASparsity{}, int32_t(K_AlignedAC / ElementASparsity{})),
                   int32_t(L)),
        make_stride(int64_t(K_AlignedAC),
                    make_stride(_1{}, ElementASparsity{}),
                    (L == 1) ? int64_t(0) : int64_t(M_AlignedAC * K_AlignedAC))
      );
    }
    else {
      return make_layout(
        make_shape(int32_t(M_AlignedAC),
                   make_shape(ElementASparsity{}, int32_t(K_AlignedAC) / ElementASparsity{}),
                   int32_t(L)),
        make_stride(ElementASparsity{},
                    make_stride(_1{}, int64_t(M_AlignedAC) * ElementASparsity{}),
                    (L == 1) ? int64_t(0) : int64_t(M_AlignedAC * K_AlignedAC))
      );
    }
  }

  template <
    class ProblemShape
  >
  CUTE_HOST_DEVICE
  static constexpr auto
  fill_layoutE(ProblemShape problem_shape) {
    const auto [M, N, K, L] = problem_shape;

    // Round up to satisfy TensorEAlignment requirement
    const auto M_AlignedE = cutlass::round_up(M, TensorEAlignmentM{});
    const auto K_AlignedE = cutlass::round_up(K, TensorEAlignmentK{});

    // TensorEAtom first along m-dim, then along k-dim, then along batch
    static_assert(TensorEAlignmentM{} == TensorEAtomM{}, "below shape assume tensorEAlignmentM eq TensorEAtomM");
    static_assert(TensorEAlignmentK{} == TensorEAtomK{}, "below shape assume tensorEAlignmentK eq TensorEAtomK");

    return make_layout(
      make_shape(make_shape(shape<0>(TensorEAtom{}), int32_t(M_AlignedE / TensorEAtomM{})),
                 make_shape(shape<1>(TensorEAtom{}), int32_t(K_AlignedE / TensorEAtomK{})),
                 int32_t(L)),
      make_stride(make_stride(stride<0>(TensorEAtom{}), cute::Int<cute::cosize(TensorEAtom{})>{}),
                  make_stride(stride<1>(TensorEAtom{}), int64_t(M_AlignedE * TensorEAtomK{})),
                  (L == 1) ? int64_t(0) : int64_t(M_AlignedE * K_AlignedE))
    );
  }
};

} // namespace cutlass
