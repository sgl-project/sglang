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

TEST(SM80_CuTe_Ampere, CooperativeGemm1_Half_MMA) {
  constexpr uint32_t thread_block_size = 128;
  using value_type = cutlass::half_t;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm2_Double_MMA) {
  constexpr uint32_t thread_block_size = 128;
  using value_type = double;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,
         Layout<Shape<_2,_2,_1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm3_Half_MMA_CustomSmemLayouts) {
  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 128;
  using value_type = cutlass::half_t;

  auto shape_mnk = Shape<_128, _128, _128>{};
  auto tiled_mma =
    TiledMMA<
      MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
      Layout<Shape<_2, _2, _1>>, // 2x2x1 thread group
      Tile<_32, _32, _16> // 32x32x16 MMA for LDSM, 1x2x1 value group`
    >{};

  auto smem_a_atom_layout = Layout<Shape<_64, _8>, Stride< _1,_64>>{};
  auto smem_b_atom_layout = Layout<Shape< _8,_32>, Stride<_32, _1>>{};
  auto smem_c_atom_layout = make_layout(select<0,1>(shape_mnk));

  test_cooperative_gemm_col_major_layout<thread_block_size,
                                         max_vec_bits,
                                         value_type,
                                         value_type,
                                         value_type>
    (smem_a_atom_layout,
    smem_b_atom_layout,
    smem_c_atom_layout,
    shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm4_Half_MMA_SwizzledSmemLayouts) {
  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 128;
  using value_type = cutlass::half_t;

  auto shape_mnk = Shape<_128, _128, _128>{};
  auto tiled_mma =
    TiledMMA<
      MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
      Layout<Shape<_2, _2, _1>>, // 2x2x1 thread group
      Tile<_32, _32, _16> // 32x32x16 MMA for LDSM, 1x2x1 value group`
    >{};

  // RowMajor
  auto smem_a_atom_layout =
    composition(Swizzle<3,3,3>{},
                Layout<Shape < _8,_64>,
                       Stride<_64, _1>>{});
  // ColMajor
  auto smem_b_atom_layout =
    composition(Swizzle<3,3,3>{},
                Layout<Shape <_64, _8>,
                       Stride< _1,_64>>{});

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
     tiled_mma,
     cute::identity{}, // TransformLoadA
     cute::identity{}, // TransformLoadB
     cute::identity{}, // TransformLoadC
     cute::identity{}, // TransformStoreC
     SM75_U32x4_LDSM_N{}, // A
     SM75_U16x8_LDSM_T{}, // B
     AutoVectorizingCopyWithAssumedAlignment<128>{}); // C
}

TEST(SM80_CuTe_Ampere, CooperativeGemm5_Double_MMA_SwizzledSmemLayouts) {
  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 128;
  using value_type = double;

  auto shape_mnk = Shape<_128, _64, _16>{};
  auto tiled_mma =
      TiledMMA<MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,        // Atom
               Layout<Shape<_2, _2, _1>>,                   // Atom layout
               Tile<Layout<Shape<_16, _2>, Stride<_2, _1>>, // 32x32x4 MMA with perm for load vectorization
                    Layout<Shape<_16, _2>, Stride<_2, _1>>,
                    Underscore>>{};

  auto smem_a_atom_layout =
      composition(Swizzle<2,2,2>{},
                  Layout<Shape <_16, _4>,
                         Stride< _1,_16>>{}); // M, K
  auto smem_b_atom_layout =
      composition(Swizzle<2,2,2>{},
                  Layout<Shape <_16, _4>,
                         Stride< _1,_16>>{}); // N, K

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

TEST(SM80_CuTe_Ampere, CooperativeGemm6_MixedPrecisionFP16FP32_MMA) {
  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 128;
  using TA = cutlass::half_t;
  using TB = cutlass::half_t;
  using TC = float;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, max_vec_bits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm7_MixedPrecisionBF16FP32_MMA) {
  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 128;
  using TA = cutlass::bfloat16_t;
  using TB = cutlass::bfloat16_t;
  using TC = float;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F32BF16BF16F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, max_vec_bits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm8_MixedPrecisionTF32FP32_MMA) {
  constexpr uint32_t thread_block_size = 128;
  constexpr uint32_t max_vec_bits = 128;
  using TA = cutlass::tfloat32_t;
  using TB = cutlass::tfloat32_t;
  using TC = float;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, max_vec_bits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm9_C64C64C64_MMA_Dynamic) {
  constexpr uint32_t thread_block_size = 256;
  constexpr int MaxVecBits = 128;
  using TA = cutlass::complex<double>;
  using TB = cutlass::complex<double>;
  using TC = cutlass::complex<double>;

  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_8x8x4_C64C64C64C64_TN>,
        Layout<Shape<_4, _4, _1>, Stride<_1, _4, _0>>,
        Tile<Underscore, Underscore, Underscore>
      >{};

  auto a_layout = make_layout(Shape<Int<13>,Int<35>>{}, make_stride(44, 1));
  auto b_layout = make_layout(Shape< Int<7>, Int<35>>{}, make_stride(44, 1));
  auto c_layout = make_layout(Shape<Int<13>,  Int<7>>{}, make_stride(1, 30));

  test_cooperative_gemm<thread_block_size,
                        MaxVecBits,
                        TA, TB, TC>
    (a_layout,
     b_layout,
     c_layout,
     a_layout,
     b_layout,
     c_layout,
     tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm9_C64C64C64_MMA) {
  constexpr uint32_t thread_block_size = 256;
  constexpr int MaxVecBits = 128;
  using TA = cutlass::complex<double>;
  using TB = cutlass::complex<double>;
  using TC = cutlass::complex<double>;

  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_8x8x4_C64C64C64C64_TN>,
        Layout<Shape<_4, _4, _1>, Stride<_1, _4, _0>>,
        Tile<Underscore, Underscore, Underscore>
      >{};

  auto a_layout = Layout<Shape<Int<13>,Int<35>>, Stride<Int<44>, Int<1> >>{};
  auto b_layout = Layout<Shape< Int<7>, Int<35>>, Stride<Int<44>, Int<1> >>{};
  auto c_layout = Layout<Shape<Int<13>,  Int<7>>, Stride< Int<1>, Int<30>>>{};

  test_cooperative_gemm<thread_block_size,
                        MaxVecBits,
                        TA, TB, TC>
    (a_layout,
     b_layout,
     c_layout,
     a_layout,
     b_layout,
     c_layout,
     tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm10_F16F64F16_FMA) {

  constexpr uint32_t thread_block_size = 256;
  constexpr int MaxVecBits = 128;
  using TA = cutlass::half_t;
  using TB = double;
  using TC = cutlass::half_t;

  auto tiled_mma =
      TiledMMA<
        MMA_Atom<UniversalFMA<half_t, half_t, double, half_t>>,
        Layout<Shape<_16, _16, _1>, Stride<_1, _16, _0>>,
        Tile<Underscore, Underscore, Underscore>
      >{};

  auto a_layout = Layout<Shape<Int<64>,Int<64>>, Stride<Int<64>, Int< 1>>>{};
  auto b_layout = Layout<Shape<Int<64>,Int<64>>, Stride<Int< 1>, Int<64>>>{};
  auto c_layout = Layout<Shape<Int<64>,Int<64>>, Stride<Int< 1>, Int<64>>>{};

  test_cooperative_gemm<thread_block_size,
                        MaxVecBits,
                        TA,
                        TB,
                        TC>
    (a_layout,
     b_layout,
     c_layout,
     a_layout,
     b_layout,
     c_layout,
     tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemmComposedStride) {

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 16;
  using T = cute::half_t;

  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
        Layout<Shape<_2, _2, _1>, Stride<_1, _2, _0>>,
        Tile<Underscore, Underscore, Underscore>
      >{};

  auto swizzle = cute::Swizzle<3, 3, 3>{};
  auto offset = cute::_0{};
  auto atom_tile_right = cute::make_layout(cute::Shape<cute::_8, cute::_64>{}, cute::LayoutRight{});
  auto FP16AtomLayoutRight = cute::composition(swizzle, offset, atom_tile_right);

  auto shape = cute::Shape<cute::Int<128>, cute::Int<128>>{};
  auto global_a_layout = cute::make_layout(shape, cute::LayoutRight{});
  auto global_b_layout = cute::make_layout(shape, cute::LayoutLeft{});
  auto global_c_layout = cute::make_layout(shape, cute::LayoutRight{});

  // This is for A row major, B col major according to CUTLASS default configs
  auto a_layout = cute::tile_to_shape(FP16AtomLayoutRight, global_a_layout);
  auto b_layout = cute::tile_to_shape(FP16AtomLayoutRight, global_b_layout);
  auto c_layout = global_c_layout;

  test_cooperative_gemm<thread_block_size,
                        MaxVecBits,
                        T, T, T>
    (a_layout,
     b_layout,
     c_layout,
     a_layout,
     b_layout,
     c_layout,
     tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm8_MixedPrecisionTF32FP32_Transform) {
  constexpr uint32_t thread_block_size = 64;
  constexpr uint32_t max_vec_bits = 16;
  using TA = cutlass::tfloat32_t;
  using TB = cutlass::tfloat32_t;
  using TC = float;

  auto shape_mnk = Shape<C<9>, C<9>, C<9>>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
        Layout<Shape<_1, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, max_vec_bits, TA, TB, TC>
    (shape_mnk, tiled_mma, cute::negate{}, cute::negate{}, cute::negate{}, cute::negate{});
}

TEST(SM80_CuTe_Ampere, CooperativeGemm8_MixedPrecisionTF32FP32_TransformPrecision) {
  constexpr uint32_t thread_block_size = 64;
  constexpr uint32_t max_vec_bits = 16;
  using InputTA = cutlass::half_t;
  using InputTB = cutlass::half_t;
  using InputTC = cutlass::half_t;

  using ComputeTA = cutlass::tfloat32_t;
  using ComputeTB = cutlass::tfloat32_t;
  using ComputeTC = float;

  auto shape_mnk = Shape<C<9>, C<9>, C<9>>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
        Layout<Shape<_1, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, max_vec_bits, InputTA, InputTB, InputTC>
    (shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm8_MixedPrecisionTF32FP32_TransformPrecisionReg) {
  constexpr uint32_t thread_block_size = 64;
  constexpr uint32_t max_vec_bits = 16;
  using InputTA = cutlass::half_t;
  using InputTB = cutlass::half_t;
  using InputTC = cutlass::half_t;

  using ComputeTA = cutlass::tfloat32_t;
  using ComputeTB = cutlass::tfloat32_t;
  using ComputeTC = float;

  auto shape_mnk = Shape<C<9>, C<9>, C<9>>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
        Layout<Shape<_1, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout_rmem_c<thread_block_size, max_vec_bits, InputTA, InputTB, InputTC>
    (shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm1_Half_MMA_Reg) {
  using value_type = cutlass::half_t;

  auto shape_mnk = Shape<_64, _64, _64>{};

  constexpr uint32_t thread_block_size = 128;

  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout_rmem_c<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm2_Double_MMA_Reg) {
  constexpr uint32_t thread_block_size = 128;
  using value_type = double;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,
         Layout<Shape<_2,_2,_1>>
      >{};

  test_cooperative_gemm_col_major_layout_rmem_c<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemm2_Double_MMA_Predicated_Reg) {
  constexpr uint32_t thread_block_size = 128;
  using value_type = double;

  auto shape_mnk = Shape<C<62>, C<62>, C<62>>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>,
         Layout<Shape<_2,_2,_1>>
      >{};

  test_cooperative_gemm_col_major_layout_rmem_c<thread_block_size, value_type>(shape_mnk, tiled_mma);
}

TEST(SM80_CuTe_Ampere, CooperativeGemmLDSMx2) {

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;
  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = float;

  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<_2, _2, _1>, Stride<_1, _2, _0>>,
        Tile<_32, _16, _16>
      >{};

  auto global_a_layout = make_layout(Shape<_32, _32>{}, LayoutRight{});
  auto global_b_layout = make_layout(Shape<_16, _32>{}, LayoutRight{});
  auto global_c_layout = make_layout(Shape<_32, _16>{}, LayoutRight{});

  test_cooperative_gemm<thread_block_size,
                        MaxVecBits,
                        TA, TB, TC>
    (global_a_layout,
     global_b_layout,
     global_c_layout,
     global_a_layout,
     global_b_layout,
     global_c_layout,
     tiled_mma, 
     identity{}, 
     identity{},
     identity{},
     identity{},
     SM75_U32x4_LDSM_N{},
     SM75_U32x2_LDSM_N{});
}

TEST(SM89_CuTe_Ada, CooperativeGemm_e4m3e4m3f32_MMA) {
  using TA = cutlass::float_e4m3_t;
  using TB = cutlass::float_e4m3_t;
  using TC = float;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM89_CuTe_Ada, CooperativeGemm_e4m3e5m2f32_MMA) {
  using TA = cutlass::float_e4m3_t;
  using TB = cutlass::float_e5m2_t;
  using TC = float;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F32E4M3E5M2F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM89_CuTe_Ada, CooperativeGemm_e5m2e4m3f32_MMA) {
  using TA = cutlass::float_e5m2_t;
  using TB = cutlass::float_e4m3_t;
  using TC = float;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F32E5M2E4M3F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM89_CuTe_Ada, CooperativeGemm_e5m2e5m2f32_MMA) {
  using TA = cutlass::float_e5m2_t;
  using TB = cutlass::float_e5m2_t;
  using TC = float;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F32E5M2E5M2F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)

TEST(SM89_CuTe_Ada, CooperativeGemm_e4m3e4m3f16_MMA) {
  using TA = cutlass::float_e4m3_t;
  using TB = cutlass::float_e4m3_t;
  using TC = cute::half_t;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F16E4M3E4M3F16_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM89_CuTe_Ada, CooperativeGemm_e4m3e5m2f16_MMA) {
  using TA = cutlass::float_e4m3_t;
  using TB = cutlass::float_e5m2_t;
  using TC = cute::half_t;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F16E4M3E5M2F16_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM89_CuTe_Ada, CooperativeGemm_e5m2e4m3f16_MMA) {
  using TA = cutlass::float_e5m2_t;
  using TB = cutlass::float_e4m3_t;
  using TC = cute::half_t;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F16E5M2E4M3F16_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM89_CuTe_Ada, CooperativeGemm_e5m2e5m2f16_MMA) {
  using TA = cutlass::float_e5m2_t;
  using TB = cutlass::float_e5m2_t;
  using TC = cute::half_t;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F16E5M2E5M2F16_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

#endif
