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

#include "../hopper/tma_load_testbed.hpp"

using namespace cute;
using namespace cutlass::test;

#if CUDA_12_0_SM90_FEATURES_SUPPORTED

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout, class CTA_Tile>
auto
test_tma_load(GMEM_Layout const& gmem_layout,
              SMEM_Layout const& smem_layout,
              CTA_Tile    const& cta_tile)
{
  return test_tma_load<T, TmaType>(SM90_TMA_LOAD{}, gmem_layout, smem_layout, cta_tile);
}

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout>
auto
test_tma_load(GMEM_Layout const& gmem_layout,
              SMEM_Layout const& smem_layout)
{
  return test_tma_load<T, TmaType>(gmem_layout, smem_layout, product_each(shape(smem_layout)));
}

TEST(SM90_CuTe_Hopper, Tma_Load_1D)
{
  {
    Layout smem_layout = Layout<_256, _1>{};
    {
    Layout gmem_layout = smem_layout;
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load< float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
    }

    {
    Layout gmem_layout = make_layout(128, GenColMajor{});
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load< float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
    }

    {
    Layout gmem_layout = make_layout(384, GenColMajor{});
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load< float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
    }
  }

  {
    Layout smem_layout = Layout<Shape<_8,_8>, Stride<_1,_8>>{};
    {
    Layout gmem_layout = smem_layout;
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load< float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
    }

    // This doesn't result in a 1D TMA, even though it could/should...
    {
    Layout gmem_layout = tile_to_shape(smem_layout, Shape<_16,_16>{});
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load< float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
    }
  }
}

TEST(SM90_CuTe_Hopper, Tma_Load_32x32_Col)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  {
  Layout gmem_layout = smem_layout;
  test_tma_load<int8_t>(gmem_layout, smem_layout);
  test_tma_load<half_t>(gmem_layout, smem_layout);
  test_tma_load< float>(gmem_layout, smem_layout);
  test_tma_load<double>(gmem_layout, smem_layout);
  }

  {
  Layout gmem_layout = make_layout(make_shape(32,32), GenColMajor{});
  test_tma_load<int8_t>(gmem_layout, smem_layout);
  test_tma_load<half_t>(gmem_layout, smem_layout);
  test_tma_load< float>(gmem_layout, smem_layout);
  test_tma_load<double>(gmem_layout, smem_layout);
  }

  {
  Layout gmem_layout = make_layout(make_shape(32,32), make_stride(Int<1>{}, 1024));
  test_tma_load<int8_t>(gmem_layout, smem_layout);
  test_tma_load<half_t>(gmem_layout, smem_layout);
  test_tma_load< float>(gmem_layout, smem_layout);
  test_tma_load<double>(gmem_layout, smem_layout);
  }
}

TEST(SM90_CuTe_Hopper, Tma_Load_32x32_Row)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_32,_1>>{};
  {
  Layout gmem_layout = smem_layout;
  test_tma_load<int8_t>(gmem_layout, smem_layout);
  test_tma_load<half_t>(gmem_layout, smem_layout);
  test_tma_load< float>(gmem_layout, smem_layout);
  test_tma_load<double>(gmem_layout, smem_layout);
  }

  {
  Layout gmem_layout = make_layout(make_shape(32,32), GenRowMajor{});
  test_tma_load<int8_t>(gmem_layout, smem_layout);
  test_tma_load<half_t>(gmem_layout, smem_layout);
  test_tma_load< float>(gmem_layout, smem_layout);
  test_tma_load<double>(gmem_layout, smem_layout);
  }

  {
  Layout gmem_layout = make_layout(make_shape(32,32), make_stride(1024, Int<1>{}));
  test_tma_load<int8_t>(gmem_layout, smem_layout);
  test_tma_load<half_t>(gmem_layout, smem_layout);
  test_tma_load< float>(gmem_layout, smem_layout);
  test_tma_load<double>(gmem_layout, smem_layout);
  }
}

template <class T, template <typename> typename SWIZZLE_ATOM>
void
test_tma_load_swizzle_atom_mn()
{
  auto smem_layout = SWIZZLE_ATOM<T>{};
  { // Static gmem
  //Layout gmem_layout = make_layout(shape(smem_layout), GenColMajor{});
  //test_tma_load<T>(gmem_layout, smem_layout);
  }
  { // Dynamic gmem
  Layout gmem_layout = make_layout(make_shape(2*uint32_t(size<0>(smem_layout)), 2*uint32_t(size<1>(smem_layout))),
                                   GenColMajor{});
  test_tma_load<T>(gmem_layout, smem_layout);
  }
}

template <class T, template <typename> typename SWIZZLE_ATOM>
void
test_tma_load_swizzle_atom_k()
{
  auto smem_layout = SWIZZLE_ATOM<T>{};
  { // Static gmem
  //Layout gmem_layout = make_layout(shape(smem_layout), GenRowMajor{});
  //test_tma_load<T>(gmem_layout, smem_layout);
  }
  { // Dynamic gmem
  Layout gmem_layout = make_layout(make_shape(2*uint32_t(size<0>(smem_layout)), 2*uint32_t(size<1>(smem_layout))),
                                   GenRowMajor{});
  test_tma_load<T>(gmem_layout, smem_layout);
  }
}

TEST(SM90_CuTe_Hopper, Tma_Load_Swizzle_Atoms)
{
  test_tma_load_swizzle_atom_mn<int8_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_load_swizzle_atom_mn<half_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_load_swizzle_atom_mn< float, GMMA::Layout_MN_SW128_Atom>();
  test_tma_load_swizzle_atom_mn<double, GMMA::Layout_MN_SW128_Atom>();

  test_tma_load_swizzle_atom_mn<int8_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_load_swizzle_atom_mn<half_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_load_swizzle_atom_mn< float, GMMA::Layout_MN_SW64_Atom>();
  test_tma_load_swizzle_atom_mn<double, GMMA::Layout_MN_SW64_Atom>();

  test_tma_load_swizzle_atom_mn<int8_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_load_swizzle_atom_mn<half_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_load_swizzle_atom_mn< float, GMMA::Layout_MN_SW32_Atom>();
  test_tma_load_swizzle_atom_mn<double, GMMA::Layout_MN_SW32_Atom>();

  test_tma_load_swizzle_atom_mn<int8_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_load_swizzle_atom_mn<half_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_load_swizzle_atom_mn< float, GMMA::Layout_MN_INTER_Atom>();
  test_tma_load_swizzle_atom_mn<double, GMMA::Layout_MN_INTER_Atom>();

  test_tma_load_swizzle_atom_k<int8_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_load_swizzle_atom_k<half_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_load_swizzle_atom_k< float, GMMA::Layout_K_SW128_Atom>();
  test_tma_load_swizzle_atom_k<double, GMMA::Layout_K_SW128_Atom>();

  test_tma_load_swizzle_atom_k<int8_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_atom_k<half_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_atom_k< float, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_atom_k<double, GMMA::Layout_K_SW64_Atom>();

  test_tma_load_swizzle_atom_k<int8_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_atom_k<half_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_atom_k< float, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_atom_k<double, GMMA::Layout_K_SW32_Atom>();

  test_tma_load_swizzle_atom_k<int8_t, GMMA::Layout_K_INTER_Atom>();
  test_tma_load_swizzle_atom_k<half_t, GMMA::Layout_K_INTER_Atom>();
  test_tma_load_swizzle_atom_k< float, GMMA::Layout_K_INTER_Atom>();
  test_tma_load_swizzle_atom_k<double, GMMA::Layout_K_INTER_Atom>();
}

template <class T, template <typename> typename SWIZZLE_ATOM>
auto
test_tma_load_swizzle_tile_mn()
{
  auto   smem_layout = tile_to_shape(SWIZZLE_ATOM<T>{}, Shape<_128,_128>{});
  Layout gmem_layout = make_layout(make_shape(int(size<0>(smem_layout)), int(size<1>(smem_layout))), GenColMajor{});
  return test_tma_load<T>(gmem_layout, smem_layout);
}

template <class T, template <typename> typename SWIZZLE_ATOM>
auto
test_tma_load_swizzle_tile_k()
{
  auto   smem_layout = tile_to_shape(SWIZZLE_ATOM<T>{}, Shape<_128,_128>{});
  Layout gmem_layout = make_layout(make_shape(int(size<0>(smem_layout)), int(size<1>(smem_layout))), GenRowMajor{});
  return test_tma_load<T>(gmem_layout, smem_layout);
}

TEST(SM90_CuTe_Hopper, Tma_Load_Swizzle_Tiles)
{
  // Other T-types use too much smem
  test_tma_load_swizzle_tile_mn<int8_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_load_swizzle_tile_mn<half_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_load_swizzle_tile_mn<int8_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_load_swizzle_tile_mn<half_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_load_swizzle_tile_mn<int8_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_load_swizzle_tile_mn<half_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_load_swizzle_tile_mn<int8_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_load_swizzle_tile_mn<half_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_load_swizzle_tile_k<int8_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_load_swizzle_tile_k<half_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_load_swizzle_tile_k<int8_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_tile_k<half_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_tile_k<int8_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_tile_k<half_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_tile_k<int8_t, GMMA::Layout_K_INTER_Atom>();
  test_tma_load_swizzle_tile_k<half_t, GMMA::Layout_K_INTER_Atom>();
}

// Tensor by-mode
TEST(SM90_CuTe_Hopper, Tma_Load_Tensor)
{
  // 3-mode TMA
  {
  Layout gmem_layout = make_layout(make_shape(128, 64, 5));
  auto cta_tile      = Shape<_64, _32>{};                    // GMEM Tiling:
                                                             //   Take 64-elem from m
                                                             //   Take 32-elem from k
  auto smem_layout = make_layout(Shape<_64,_32>{});
  test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  // 4-mode TMA
  {
  Layout gmem_layout = make_layout(make_shape(make_shape(80,40),make_shape(32,12)));
  auto cta_tile      = Shape<Shape<_16,_8>,Shape<_32,_2>>{}; // GMEM Tiling:
                                                             //   Take 16-elem from m0, 8-elem from m1,
                                                             //   Take 32-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(Shape<_128,_64>{});
  test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  // 5-mode TMA
  {
  Layout gmem_layout = make_layout(make_shape(make_shape(32,32,32),make_shape(32,12)));
  auto cta_tile      = Shape<Shape<_16,_4,_2>,Shape<_16,_2>>{}; // GMEM Tiling:
                                                             //   Take 4-elem from m0, 4-elem from m1, 5-elem from m2
                                                             //   Take 32-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(Shape<_128,_32>{});
  test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }
}

// Tensor Multimode -- TMA with more than 5 modes in GMEM (packs residual modes into last TMA mode)
TEST(SM90_CuTe_Hopper, Tma_Load_Tensor_Multimode)
{
  {
  Layout gmem_layout = make_layout(make_shape(make_shape(32,3,2,2),make_shape(32,4,2)));
  auto cta_tile      = Shape<Shape<_32>, Shape<_32,_2>>{};    // GMEM Tiling:
                                                              //  Take 32-elem from m0
                                                              //  Take 32-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(Shape<_32,_64>{});
  test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  {
  Layout gmem_layout = make_layout(make_shape(make_shape(64,3,2,2),make_shape(32,4,2)));
  auto cta_tile      = Shape<Shape<_32,_3>, Shape<_32,_2>>{}; // GMEM Tiling:
                                                              //  Take 32-elem from m0, 3-elem from m1
                                                              //  Take 32-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(Shape<_96,_64>{});
  test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  {
  Layout gmem_layout = make_layout(make_shape(make_shape(64,3,2,3,2),make_shape(32,4,2,2)));
  auto cta_tile      = Shape<Shape<_32>, Shape<_16,_2>>{};    // GMEM Tiling:
                                                              //  Take 32-elem from m0
                                                              //  Take 16-elem from k0, 2-elem from k1
  auto smem_layout = make_layout(Shape<_32,_32>{});
  test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }
}

TEST(SM90_CuTe_Hopper, Tma_Load_Coalesce)
{
  // Interleaved ColMajor
  {
  Layout gmem_layout = make_layout(make_shape (  128, make_shape (_4{},  128)),
                                   make_stride( _4{}, make_stride(_1{},  512)));
  auto   smem_layout = make_layout(make_shape (_32{}, make_shape (_4{},  _32{})),
                                   make_stride( _4{}, make_stride(_1{}, _128{})));

  // By default, uses cta_tile = Shape<_32,_128>
  auto tma = test_tma_load<int8_t>(gmem_layout, smem_layout);
  // Check the TMA rank
  EXPECT_EQ(rank(tma.get_tma_tensor(shape(gmem_layout))(0)), 2);
  }

  // Interleaved RowMajor
  {
  Layout gmem_layout = make_layout(make_shape (make_shape (_4{},   128),   128),
                                   make_stride(make_stride(_1{},   512),   _4{}));
  auto   smem_layout = make_layout(make_shape (make_shape (_4{},  _32{}), _32{}),
                                   make_stride(make_stride(_1{}, _128{}),  _4{}));

  // By default, uses cta_tile = Shape<_128,_32>
  auto tma = test_tma_load<int8_t>(gmem_layout, smem_layout);
  // Check the TMA rank
  EXPECT_EQ(rank(tma.get_tma_tensor(shape(gmem_layout))(0)), 2);
  }

  // Account for stride-0 modes within the TMA tile
  {
  Layout gmem_layout = make_layout(make_shape (  128, make_shape (_32{},   4)),
                                   make_stride( _1{}, make_stride( _0{}, 128)));
  auto   smem_layout = make_layout(make_shape (_64{}, make_shape (_32{}     )),
                                   make_stride( _1{}, make_stride( _0{}     )));

  // By default, uses cta_tile = Shape<_64,_32>
  auto tma = test_tma_load<uint16_t>(gmem_layout, smem_layout);
  // Check the TMA rank
  EXPECT_EQ(rank(tma.get_tma_tensor(shape(gmem_layout))(0)), 2);
  }

  // Coalesce many modes and account for stride-0 modes within the TMA tile
  {
  Layout gmem_layout = make_layout(make_shape (make_shape (_32{},_4{},     4), _32{}, make_shape (_4{},      4)),
                                   make_stride(make_stride(_16{},_4{},  2048),  _0{}, make_stride(_1{}, _512{})));
  auto   smem_layout = make_layout(make_shape (make_shape (_32{},_4{}       ), _32{}, make_shape (_4{}        )),
                                   make_stride(make_stride(_16{},_4{}       ),  _0{}, make_stride(_1{}        )));

  // By default, uses cta_tile = Shape<_128,_32,_4>
  auto tma = test_tma_load<int8_t>(gmem_layout, smem_layout);
  // Check the TMA rank (Could be 3 instead of 4 with even better coalescing...?)
  EXPECT_EQ(rank(tma.get_tma_tensor(shape(gmem_layout))(0)), 4);
  }
}

TEST(SM90_CuTe_Hopper, Tma_Load_InternalType)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = make_layout(make_shape(64, 64));

  // Downcasted tensors to smaller TmaTypes
  {
  test_tma_load<int8_t, uint8_t>(gmem_layout, smem_layout);
  test_tma_load<half_t, uint8_t>(gmem_layout, smem_layout);
  test_tma_load< float, uint8_t>(gmem_layout, smem_layout);
  test_tma_load<double, uint8_t>(gmem_layout, smem_layout);
  }

  // Upcasted tensors to larger TmaTypes
  {
  test_tma_load<int8_t, uint64_t>(gmem_layout, smem_layout);
  test_tma_load<half_t, uint64_t>(gmem_layout, smem_layout);
  test_tma_load< float, uint64_t>(gmem_layout, smem_layout);
  test_tma_load<double, uint64_t>(gmem_layout, smem_layout);
  }

  // Complex<double> is 128bit, which the TMA has no concept of
  {
  test_tma_load<complex<double>, uint64_t>(gmem_layout, smem_layout);
  test_tma_load<complex<double>, uint32_t>(gmem_layout, smem_layout);
  }
}

#endif
