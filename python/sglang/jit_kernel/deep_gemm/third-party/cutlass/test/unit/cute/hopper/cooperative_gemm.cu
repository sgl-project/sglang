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

#define USE_FP8 1

#if USE_FP8
TEST(SM90_CuTe_Hopper, CooperativeGemmTilingF8) {

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 16;
  using TA = uint8_t;
  using TB = uint8_t;
  using TC = uint32_t;

  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
        Layout<Shape<_2, _2, _1>, Stride<_1, _2, _0>>,
        Tile<_32, _32, _32>
      >{};

  auto swizzle = Swizzle<2, 4, 3>{};

  // This is for A row major, B col major according to CUTLASS default configs
  auto a_layout = composition(swizzle, Layout<Shape<_64, _64>, Stride<_64, _1>>{});
  auto b_layout = composition(swizzle, Layout<Shape<_64, _64>, Stride<_1, _64>>{});
  auto c_layout = make_layout(Shape<_64, _64>{}, LayoutLeft{});

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

#else

TEST(SM90_CuTe_Hopper, CooperativeGemmTilingF16) {

  constexpr uint32_t thread_block_size = 64;
  constexpr int max_vec_bits = 16;
  using TA = half_t;
  using TB = half_t;
  using TC = half_t;

  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
        Layout<Shape<_2, _1, _1>, Stride<_1, _0, _0>>,
        Tile<_32, _32, _32>
      >{};

  // This is for A row major, B col major according to CUTLASS default configs
  auto swizzle = Swizzle<3, 3, 3>{};
  auto ALayout = composition(swizzle{}, Layout<Shape<_64, _64>, Stride<_64, _1>>{});
  auto BLayout = composition(swizzle{}, Layout<Shape<_64, _64>, Stride<_1, _64>>{});
  auto CLayout = make_layout(Shape<_64, _64>{}, LayoutLeft{});

  test_cooperative_gemm<thread_block_size,
                        max_vec_bits,
                        TA,
                        TB,
                        TC>

    (ALayout,
     BLayout,
     CLayout,
     ALayout,
     BLayout,
     CLayout,
     tiled_mma);
}

#endif

#if defined(CUTE_ARCH_STSM_SM90_ENABLED)

TEST(SM90_CuTe_Hopper, CooperativeGemmSTSM) {

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;
  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;

  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>,
        Layout<Shape<_2, _2, _1>, Stride<_1, _2, _0>>,
        Tile<_32, _32, _16>
      >{};

  auto global_a_layout = make_layout(Shape<_64, _64>{}, LayoutRight{});
  auto global_b_layout = make_layout(Shape<_64, _64>{}, LayoutRight{});
  auto global_c_layout = make_layout(Shape<_64, _64>{}, LayoutRight{});

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
     SM75_U32x4_LDSM_N{},
     SM75_U32x4_LDSM_N{},
     SM90_U32x4_STSM_N{});
}

#endif
