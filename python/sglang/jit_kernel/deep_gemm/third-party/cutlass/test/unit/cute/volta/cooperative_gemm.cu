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

#include "cutlass_unit_test.h"

#include <cute/tensor.hpp>
#include <cute/swizzle.hpp> // cute::Swizzle
#include <cute/swizzle_layout.hpp> // cute::compose(cute::Swizzle)

#include "../cooperative_gemm_common.hpp"

using namespace cute;

TEST(SM70_CuTe_Volta, CooperativeGemm1_FloatFMA) {

  constexpr uint32_t thread_block_size = 128;
  using value_type = float;

  auto shape_mnk = make_shape(_64{}, _32{}, _16{});
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<UniversalFMA<value_type, value_type, value_type, value_type>>,
        Layout<Shape<_16, _8, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM70_CuTe_Volta, CooperativeGemm1_FloatFMA_Predication) {

  constexpr uint32_t thread_block_size = 128;
  using value_type = float;

  auto shape_mnk = make_shape(C<88>{}, C<20>{}, C<12>{});
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<UniversalFMA<value_type, value_type, value_type, value_type>>,
        Layout<Shape<_2, _64, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM70_CuTe_Volta, CooperativeGemm1_FloatFMA_Predication2) {

  constexpr uint32_t thread_block_size = 128;
  using value_type = float;

  auto shape_mnk = make_shape(C<88>{}, C<36>{}, C<24>{});
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<UniversalFMA<value_type, value_type, value_type, value_type>>,
        Layout<Shape<_4, _32, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM70_CuTe_Volta, CooperativeGemm1_FloatFMA_Predication3) {
  constexpr uint32_t thread_block_size = 128;
  using value_type = float;

  auto shape_mnk = make_shape(C<67>{}, C<13>{}, C<11>{});
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<UniversalFMA<value_type, value_type, value_type, value_type>>,
        Layout<Shape<_1, _128, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM70_CuTe_Volta, CooperativeGemm2_DoubleFMA) {
  constexpr uint32_t thread_block_size = 128;
  using value_type = double;

  auto shape_mnk = make_shape(C<16>{}, C<32>{}, C<32>{});
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<UniversalFMA<value_type, value_type, value_type, value_type>>,
        Layout<Shape<_16, _8, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM70_CuTe_Volta, CooperativeGemm3_Float_FMA_CustomPermutationMNK) {

  constexpr uint32_t thread_block_size = 256;
  using value_type = float;

  auto shape_mnk = make_shape(_32{}, _32{}, _32{});
  auto tiled_mma = TiledMMA<
    MMA_Atom<
      UniversalFMA<value_type, value_type, value_type, value_type>
    >,
    Layout<
      Shape<_16, _16, _1>
    >,
    Tile<
      Layout<
        Shape<_16,_2>, Stride<_2,_1>
      >,               // 32x32x1 MMA with perm for load vectorization
      Layout<
        Shape<_16,_2>, Stride<_2,_1>
      >,
      Underscore
    >
  >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM70_CuTe_Volta, CooperativeGemm4_Half_MMA) {
  constexpr uint32_t thread_block_size = 128;
  using value_type = cutlass::half_t;

  auto shape_mnk = make_shape(_32{}, _32{}, _32{});
  auto tiled_mma = TiledMMA<
    MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>,
    Layout<Shape<_4, _4, _1>>
  >{};

  auto smem_a_atom_layout = typename decltype(tiled_mma)::AtomLayoutB_TV{};
  auto smem_b_atom_layout = typename decltype(tiled_mma)::AtomLayoutA_TV{};
  auto smem_c_atom_layout = make_layout(select<0, 1>(shape_mnk));

  test_cooperative_gemm_col_major_layout<thread_block_size,
                                         value_type>
    (smem_a_atom_layout,
     smem_b_atom_layout,
     smem_c_atom_layout,
     shape_mnk,
     tiled_mma);
}

TEST(SM70_CuTe_Volta, CooperativeGemm5_Half_MMA) {

  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 128;
  using value_type = cutlass::half_t;

  auto shape_mnk = make_shape(_32{}, _32{}, _32{});
  auto tiled_mma = TiledMMA<
    MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>,
    Layout<Shape<_4, _4, _1>>
  >{};

  auto gmem_a_layout = make_layout(select<0, 2>(shape_mnk));
  auto gmem_b_layout = make_layout(select<1, 2>(shape_mnk), GenColMajor{});
  auto gmem_c_layout = make_layout(select<0, 1>(shape_mnk));

  auto smem_a_layout = make_layout(select<0, 2>(shape_mnk));
  auto smem_b_layout = make_layout(select<1, 2>(shape_mnk), GenColMajor{});
  auto smem_c_layout = make_layout(select<0, 1>(shape_mnk));

  test_cooperative_gemm<thread_block_size,
                        max_vec_bits,
                        value_type,
                        value_type,
                        value_type>
    (gmem_a_layout,
     gmem_b_layout,
     gmem_c_layout,
     smem_a_layout,
     smem_b_layout,
     smem_c_layout,
     tiled_mma);
}

TEST(SM70_CuTe_Volta, CooperativeGemm5_Half_MMA_Predicated) {

  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 16;
  using value_type = cutlass::half_t;

  auto shape_mnk = make_shape(C<31>{}, C<27>{}, C<17>{});
  auto tiled_mma = TiledMMA<
    MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>,
    Layout<Shape<_4, _4, _1>>
  >{};

  auto gmem_a_layout = make_layout(select<0, 2>(shape_mnk));
  auto gmem_b_layout = make_layout(select<1, 2>(shape_mnk), GenColMajor{});
  auto gmem_c_layout = make_layout(select<0, 1>(shape_mnk));

  auto smem_a_layout = make_layout(select<0, 2>(shape_mnk));
  auto smem_b_layout = make_layout(select<1, 2>(shape_mnk), GenColMajor{});
  auto smem_c_layout = make_layout(select<0, 1>(shape_mnk));

  test_cooperative_gemm<thread_block_size,
                        max_vec_bits,
                        value_type,
                        value_type,
                        value_type>
    (gmem_a_layout,
     gmem_b_layout,
     gmem_c_layout,
     smem_a_layout,
     smem_b_layout,
     smem_c_layout,
     tiled_mma);
}

TEST(SM70_CuTe_Volta, CooperativeGemm6_Half_MAA_SwizzledSmemLayouts) {

  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 128;
  using value_type = cutlass::half_t;

  auto shape_mnk = make_shape(_128{}, _128{}, _64{});
  auto tiled_mma = TiledMMA<
    MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>,
    Layout<Shape<_4, _4, _1>>
  >{};

  auto smem_a_atom_layout = composition(Swizzle<3,3,3>{}, Layout<Shape < _8,_64>, Stride<_64, _1>>{});
  auto smem_b_atom_layout = composition(Swizzle<3,3,3>{}, Layout<Shape <_64, _8>, Stride< _1,_64>>{});
  auto smem_c_atom_layout = make_layout(select<0, 1>(shape_mnk), GenRowMajor{});

  auto gmem_a_layout = make_layout(select<0, 2>(shape_mnk), GenRowMajor{});
  auto gmem_b_layout = make_layout(select<1, 2>(shape_mnk), GenColMajor{});
  auto gmem_c_layout = make_layout(select<0, 1>(shape_mnk), GenRowMajor{});

  auto smem_a_layout = tile_to_shape(
      smem_a_atom_layout,
      make_shape(shape<0>(gmem_a_layout), shape<1>(gmem_a_layout)));

  auto smem_b_layout = tile_to_shape(
      smem_b_atom_layout,
      make_shape(shape<0>(gmem_b_layout), shape<1>(gmem_b_layout)));

  auto smem_c_layout = tile_to_shape(
      smem_c_atom_layout,
      make_shape(shape<0>(gmem_c_layout), shape<1>(gmem_c_layout)));

  test_cooperative_gemm<thread_block_size,
                        max_vec_bits,
                        value_type,
                        value_type,
                        value_type>
    (gmem_a_layout,
     gmem_b_layout,
     gmem_c_layout,
     smem_a_layout,
     smem_b_layout,
     smem_c_layout,
     tiled_mma);
}

TEST(SM70_CuTe_Volta, CooperativeGemm7_TransformNegate_FMA) {
  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 64;
  using TA = float;
  using TB = float;
  using TC = double;

  auto shape_mnk = make_shape(_32{}, _32{}, _32{});
  auto tiled_mma = TiledMMA<
    MMA_Atom<UniversalFMA<TC, TA, TB, TC>>,
    Layout<Shape<_16, _8, _1>>
  >{};

  auto aload  = cute::negate {};
  auto bload  = cute::negate {};
  auto cload  = cute::negate {};
  auto cstore = cute::negate {};

  test_cooperative_gemm_col_major_layout<thread_block_size, max_vec_bits, TA, TB, TC>(
      shape_mnk, tiled_mma, aload, bload, cload, cstore);
}

TEST(SM70_CuTe_Volta, CooperativeGemm7_TransformNegate_MMA) {

  constexpr uint32_t thread_block_size = 128;
  using value_type = cutlass::half_t;

  auto shape_mnk = make_shape(_32{}, _32{}, _32{});
  auto tiled_mma = TiledMMA<
    MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>,
    Layout<Shape<_4, _4, _1>>
  >{};

  auto aload  = cute::negate {};
  auto bload  = cute::negate {};
  auto cload  = cute::negate {};
  auto cstore = cute::negate {};

  test_cooperative_gemm_col_major_layout<thread_block_size, value_type>(
      shape_mnk, tiled_mma, aload, bload, cload, cstore);
}

template<class ConstantType>
struct increment_by_x {
  ConstantType x;

  template <class T>
  CUTE_HOST_DEVICE constexpr
  T operator()(const T& arg) const {
    return arg + x;
  }
};

template<class From, class To>
struct convert_to {
  CUTE_HOST_DEVICE constexpr
  To operator()(const From& arg) const {
    return static_cast<To>(arg);
  }
};

TEST(SM70_CuTe_Volta, CooperativeGemm7_TransformCustomOp_FMA) {

  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 64;

  using TA = float;
  using TB = float;
  using TC = double;

  auto shape_mnk = make_shape(_32{}, _32{}, _32{});
  auto tiled_mma = TiledMMA<
    MMA_Atom<UniversalFMA<TC, TA, TB, TC>>,
    Layout<Shape<_16, _8, _1>>
  >{};

  auto aload  = increment_by_x<float>{1.111f};
  auto bload  = convert_to<float, double> {};
  auto cload  = cute::negate {};
  auto cstore = cute::negate {};

  test_cooperative_gemm_col_major_layout<thread_block_size, max_vec_bits, TA, TB, TC>(
      shape_mnk, tiled_mma, aload, bload, cload, cstore);
}
