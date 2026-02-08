/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../hopper/tma_store_testbed.hpp"

using namespace cute;
using namespace cutlass::test;

#if CUDA_12_0_SM90_FEATURES_SUPPORTED

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout, class CTA_Tile>
void
test_tma_store(GMEM_Layout const& gmem_layout,
               SMEM_Layout const& smem_layout,
               CTA_Tile    const& cta_tile)
{
  using namespace cute;
  return test_tma_store<T, TmaType>(SM90_TMA_STORE{}, gmem_layout, smem_layout, cta_tile);
}

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout>
void
test_tma_store(GMEM_Layout const& gmem_layout,
               SMEM_Layout const& smem_layout)
{
  using namespace cute;
  return test_tma_store<T, TmaType>(gmem_layout, smem_layout, product_each(shape(smem_layout)));
}

TEST(SM90_CuTe_Hopper, Tma_Load_1D)
{
  Layout smem_layout = Layout<_256, _1>{};
  {
  Layout gmem_layout = smem_layout;
  test_tma_store<int8_t>(gmem_layout, smem_layout);
  test_tma_store<half_t>(gmem_layout, smem_layout);
  test_tma_store< float>(gmem_layout, smem_layout);
  test_tma_store<double>(gmem_layout, smem_layout);
  }

  {
  Layout gmem_layout = make_layout(128, GenColMajor{});
  test_tma_store<int8_t>(gmem_layout, smem_layout);
  test_tma_store<half_t>(gmem_layout, smem_layout);
  test_tma_store< float>(gmem_layout, smem_layout);
  test_tma_store<double>(gmem_layout, smem_layout);
  }
}

TEST(SM90_CuTe_Hopper, Tma_Store_32x32_Col)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  {
  Layout gmem_layout = smem_layout;
  test_tma_store<int8_t>(gmem_layout, smem_layout);
  test_tma_store<half_t>(gmem_layout, smem_layout);
  test_tma_store< float>(gmem_layout, smem_layout);
  test_tma_store<double>(gmem_layout, smem_layout);
  }

  {
  Layout gmem_layout = make_layout(make_shape(32,32), GenColMajor{});
  test_tma_store<int8_t>(gmem_layout, smem_layout);
  test_tma_store<half_t>(gmem_layout, smem_layout);
  test_tma_store< float>(gmem_layout, smem_layout);
  test_tma_store<double>(gmem_layout, smem_layout);
  }

  {
  Layout gmem_layout = make_layout(make_shape(32,32), make_stride(Int<1>{}, 1024));
  test_tma_store<int8_t>(gmem_layout, smem_layout);
  test_tma_store<half_t>(gmem_layout, smem_layout);
  test_tma_store< float>(gmem_layout, smem_layout);
  test_tma_store<double>(gmem_layout, smem_layout);
  }
}

TEST(SM90_CuTe_Hopper, Tma_Store_32x32_Row)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_32,_1>>{};
  {
  Layout gmem_layout = smem_layout;
  test_tma_store<int8_t>(gmem_layout, smem_layout);
  test_tma_store<half_t>(gmem_layout, smem_layout);
  test_tma_store< float>(gmem_layout, smem_layout);
  test_tma_store<double>(gmem_layout, smem_layout);
  }

  {
  Layout gmem_layout = make_layout(make_shape(32,32), GenRowMajor{});
  test_tma_store<int8_t>(gmem_layout, smem_layout);
  test_tma_store<half_t>(gmem_layout, smem_layout);
  test_tma_store< float>(gmem_layout, smem_layout);
  test_tma_store<double>(gmem_layout, smem_layout);
  }

  {
  Layout gmem_layout = make_layout(make_shape(32,32), make_stride(1024, Int<1>{}));
  test_tma_store<int8_t>(gmem_layout, smem_layout);
  test_tma_store<half_t>(gmem_layout, smem_layout);
  test_tma_store< float>(gmem_layout, smem_layout);
  test_tma_store<double>(gmem_layout, smem_layout);
  }
}

template <class T, template <typename> typename SWIZZLE_ATOM>
void
test_tma_store_swizzle_atom_mn()
{
  auto   smem_layout = SWIZZLE_ATOM<T>{};
  Layout gmem_layout = make_layout(make_shape(2*size<0>(smem_layout), 2*size<1>(smem_layout)), GenColMajor{});
  return test_tma_store<T>(gmem_layout, smem_layout);
}

template <class T, template <typename> typename SWIZZLE_ATOM>
void
test_tma_store_swizzle_atom_k()
{
  auto   smem_layout = SWIZZLE_ATOM<T>{};
  Layout gmem_layout = make_layout(make_shape(2*size<0>(smem_layout), 2*size<1>(smem_layout)), GenRowMajor{});
  return test_tma_store<T>(gmem_layout, smem_layout);
}

TEST(SM90_CuTe_Hopper, Tma_Store_Swizzle_Atoms)
{
  test_tma_store_swizzle_atom_mn<int8_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_store_swizzle_atom_mn<half_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_store_swizzle_atom_mn< float, GMMA::Layout_MN_SW128_Atom>();
  test_tma_store_swizzle_atom_mn<double, GMMA::Layout_MN_SW128_Atom>();

  test_tma_store_swizzle_atom_mn<int8_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_store_swizzle_atom_mn<half_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_store_swizzle_atom_mn< float, GMMA::Layout_MN_SW64_Atom>();
  test_tma_store_swizzle_atom_mn<double, GMMA::Layout_MN_SW64_Atom>();

  test_tma_store_swizzle_atom_mn<int8_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_store_swizzle_atom_mn<half_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_store_swizzle_atom_mn< float, GMMA::Layout_MN_SW32_Atom>();
  test_tma_store_swizzle_atom_mn<double, GMMA::Layout_MN_SW32_Atom>();

  test_tma_store_swizzle_atom_mn<int8_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_store_swizzle_atom_mn<half_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_store_swizzle_atom_mn< float, GMMA::Layout_MN_INTER_Atom>();
  test_tma_store_swizzle_atom_mn<double, GMMA::Layout_MN_INTER_Atom>();

  test_tma_store_swizzle_atom_k<int8_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_store_swizzle_atom_k<half_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_store_swizzle_atom_k< float, GMMA::Layout_K_SW128_Atom>();
  test_tma_store_swizzle_atom_k<double, GMMA::Layout_K_SW128_Atom>();

  test_tma_store_swizzle_atom_k<int8_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_store_swizzle_atom_k<half_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_store_swizzle_atom_k< float, GMMA::Layout_K_SW64_Atom>();
  test_tma_store_swizzle_atom_k<double, GMMA::Layout_K_SW64_Atom>();

  test_tma_store_swizzle_atom_k<int8_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_store_swizzle_atom_k<half_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_store_swizzle_atom_k< float, GMMA::Layout_K_SW32_Atom>();
  test_tma_store_swizzle_atom_k<double, GMMA::Layout_K_SW32_Atom>();

  test_tma_store_swizzle_atom_k<int8_t, GMMA::Layout_K_INTER_Atom>();
  test_tma_store_swizzle_atom_k<half_t, GMMA::Layout_K_INTER_Atom>();
  test_tma_store_swizzle_atom_k< float, GMMA::Layout_K_INTER_Atom>();
  test_tma_store_swizzle_atom_k<double, GMMA::Layout_K_INTER_Atom>();
}

template <class T, template <typename> typename SWIZZLE_ATOM>
void
test_tma_store_swizzle_tile_mn()
{
  auto   smem_layout = tile_to_shape(SWIZZLE_ATOM<T>{}, Shape<_128,_128>{});
  Layout gmem_layout = make_layout(make_shape(2*size<0>(smem_layout), 2*size<1>(smem_layout)), GenColMajor{});
  return test_tma_store<T>(gmem_layout, smem_layout);
}

template <class T, template <typename> typename SWIZZLE_ATOM>
void
test_tma_store_swizzle_tile_k()
{
  auto   smem_layout = tile_to_shape(SWIZZLE_ATOM<T>{}, Shape<_128,_128>{});
  Layout gmem_layout = make_layout(make_shape(2*size<0>(smem_layout), 2*size<1>(smem_layout)), GenRowMajor{});
  return test_tma_store<T>(gmem_layout, smem_layout);
}

TEST(SM90_CuTe_Hopper, Tma_Store_Swizzle_Tiles)
{
  // Other T-types use too much smem
  test_tma_store_swizzle_tile_mn<int8_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_store_swizzle_tile_mn<half_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_store_swizzle_tile_mn<int8_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_store_swizzle_tile_mn<half_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_store_swizzle_tile_mn<int8_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_store_swizzle_tile_mn<half_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_store_swizzle_tile_mn<int8_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_store_swizzle_tile_mn<half_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_store_swizzle_tile_k<int8_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_store_swizzle_tile_k<half_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_store_swizzle_tile_k<int8_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_store_swizzle_tile_k<half_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_store_swizzle_tile_k<int8_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_store_swizzle_tile_k<half_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_store_swizzle_tile_k<int8_t, GMMA::Layout_K_INTER_Atom>();
  test_tma_store_swizzle_tile_k<half_t, GMMA::Layout_K_INTER_Atom>();
}

// Tensor by-mode
TEST(SM90_CuTe_Hopper, Tma_Store_Tensor)
{
  // 3-mode TMA
  {
  Layout gmem_layout = make_layout(make_shape(128, 64, 5));
  auto cta_tile      = Shape<_64, _32>{};                    // GMEM Tiling:
                                                             //   Take 64-elem from m
                                                             //   Take 32-elem from k
  auto smem_layout = make_layout(Shape<_64,_32>{});
  test_tma_store<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  // 4-mode TMA
  {
  Layout gmem_layout = make_layout(make_shape(make_shape(80,40),make_shape(32,12)));
  auto cta_tile      = Shape<Shape<_16,_8>,Shape<_32,_2>>{}; // GMEM Tiling:
                                                             //   Take 16-elem from m0, 8-elem from m1,
                                                             //   Take 32-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(Shape<_128,_64>{});
  test_tma_store<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  // 5-mode TMA
  {
  Layout gmem_layout = make_layout(make_shape(make_shape(32,32,32),make_shape(32,12)));
  auto cta_tile      = Shape<Shape<_16,_4,_2>,Shape<_16,_2>>{}; // GMEM Tiling:
                                                             //   Take 4-elem from m0, 4-elem from m1, 5-elem from m2
                                                             //   Take 32-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(Shape<_128,_32>{});
  test_tma_store<half_t>(gmem_layout, smem_layout, cta_tile);
  }
}

// Tensor Multimode -- TMA with more than 5 modes in GMEM (packs residual modes into last TMA mode)
TEST(SM90_CuTe_Hopper, Tma_Store_Tensor_Multimode)
{
  {
  Layout gmem_layout = make_layout(make_shape(make_shape(32,3,2,2),make_shape(32,4,2)));
  auto cta_tile      = Shape<Shape<_32>, Shape<_32,_2>>{};    // GMEM Tiling:
                                                              //  Take 32-elem from m0
                                                              //  Take 32-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(Shape<_32,_64>{});
  test_tma_store<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  {
  Layout gmem_layout = make_layout(make_shape(make_shape(64,3,2,2),make_shape(32,4,2)));
  auto cta_tile      = Shape<Shape<_32,_3>, Shape<_32,_2>>{}; // GMEM Tiling:
                                                              //  Take 32-elem from m0, 3-elem from m1
                                                              //  Take 32-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(Shape<_96,_64>{});
  test_tma_store<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  {
  Layout gmem_layout = make_layout(make_shape(make_shape(64,3,2,3,2),make_shape(32,4,2,2)));
  auto cta_tile      = Shape<Shape<_32>, Shape<_16,_2>>{};    // GMEM Tiling:
                                                              //  Take 32-elem from m0
                                                              //  Take 16-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(Shape<_32,_32>{});
  test_tma_store<half_t>(gmem_layout, smem_layout, cta_tile);
  }
}

#endif
