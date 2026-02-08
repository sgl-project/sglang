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

#include <iostream>
#include <iomanip>
#include <utility>
#include <type_traits>
#include <vector>
#include <numeric>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/swizzle.hpp> // cute::Swizzle

#include "tiled_cp_async_testbed.hpp"

using namespace cute;

TEST(SM80_CuTe_tiled_cp_async, no_swizzle_mn_single_tile)
{
  {
  using copy_atom = decltype(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{});
  using thr_layout = decltype(Layout<Shape <_16, _8>, Stride< _1,_16>>{});
  using val_layout = decltype(Layout<Shape<_2,_1>>{});
  using tiled_copy = decltype(make_tiled_copy(copy_atom{}, thr_layout{}, val_layout{}));
  using smem_layout_atom = decltype(Layout<Shape <_16, _4>, Stride< _1,_16>>{});
  using gmem_stride_type = decltype(LayoutLeft{});
  test_cp_async_no_swizzle<double, cute::Int<64>, cute::Int<16>, gmem_stride_type, smem_layout_atom, tiled_copy>();
  }

  {
  using copy_atom = decltype(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{});
  using thr_layout = decltype(Layout<Shape <_16, _8>, Stride< _1,_16>>{});
  using val_layout = decltype(Layout<Shape<_2,_1>>{});
  using tiled_copy = decltype(make_tiled_copy(copy_atom{}, thr_layout{}, val_layout{}));
  using smem_layout_atom = decltype(Layout<Shape <_16, _4>, Stride< _1,_16>>{});
  using gmem_stride_type = decltype(LayoutLeft{});
  test_cp_async_no_swizzle<double, cute::Int<128>, cute::Int<16>, gmem_stride_type, smem_layout_atom, tiled_copy>();
  }
}

TEST(SM80_CuTe_tiled_cp_async, no_swizzle_k_single_tile)
{
  {
  using copy_atom = decltype(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{});
  using thr_layout = decltype(Layout<Shape <_16, _8>, Stride< _8,_1>>{});
  using val_layout = decltype(Layout<Shape<_1,_2>>{});
  using tiled_copy = decltype(make_tiled_copy(copy_atom{}, thr_layout{}, val_layout{}));
  using smem_layout_atom = decltype(make_ordered_layout(Shape<_128,_16>{}, Step <_2, _1>{}));
  using gmem_stride_type = decltype(LayoutRight{});
  test_cp_async_no_swizzle<double, cute::Int<128>, cute::Int<16>, gmem_stride_type, smem_layout_atom, tiled_copy>();
  }
}

TEST(SM80_CuTe_tiled_cp_async, swizzle_mn_single_tile)
{
  {
  using copy_atom = decltype(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{});
  using thr_layout = decltype(Layout<Shape <_16, _8>, Stride< _1,_16>>{});
  using val_layout = decltype(Layout<Shape<_2,_1>>{});
  using tiled_copy = decltype(make_tiled_copy(copy_atom{}, thr_layout{}, val_layout{}));
  using swizzle_atom = decltype(Swizzle<2,2,2>{});
  using smem_layout_atom = decltype(Layout<Shape <_16, _4>, Stride< _1,_16>>{});
  using gmem_stride_type = decltype(LayoutLeft{});
  test_cp_async_with_swizzle<double, cute::Int<64>, cute::Int<16>, gmem_stride_type, swizzle_atom, smem_layout_atom, tiled_copy>();
  }

  {
  using copy_atom = decltype(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{});
  using thr_layout = decltype(Layout<Shape <_16, _8>, Stride< _1,_16>>{});
  using val_layout = decltype(Layout<Shape<_2,_1>>{});
  using tiled_copy = decltype(make_tiled_copy(copy_atom{}, thr_layout{}, val_layout{}));
  using swizzle_atom = decltype(Swizzle<2,2,2>{});
  using smem_layout_atom = decltype(Layout<Shape <_16, _4>, Stride< _1,_16>>{});
  using gmem_stride_type = decltype(LayoutLeft{});
  test_cp_async_with_swizzle<double, cute::Int<128>, cute::Int<16>, gmem_stride_type, swizzle_atom, smem_layout_atom, tiled_copy>();
  }
}

TEST(SM80_CuTe_tiled_cp_async, swizzle_k_single_tile)
{
  {
  using copy_atom = decltype(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<double>, double>{});
  using thr_layout = decltype(Layout<Shape < _8,_16>, Stride<_16, _1>>{});
  using val_layout = decltype(Layout<Shape<_1,_1>>{});
  using tiled_copy = decltype(make_tiled_copy(copy_atom{}, thr_layout{}, val_layout{}));
  using swizzle_atom = decltype(Swizzle<2,0,4>{});
  using smem_layout_atom = decltype(Layout<Shape <_4,_16>, Stride<_1, _4>>{});
  using gmem_stride_type = decltype(LayoutRight{});
  test_cp_async_with_swizzle<double, cute::Int<128>, cute::Int<16>, gmem_stride_type, swizzle_atom, smem_layout_atom, tiled_copy>();
  }

  {
  using copy_atom = decltype(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, tfloat32_t>{});
  using thr_layout = decltype(Layout<Shape <_16,_8>, Stride< _8,_1>>{});
  using val_layout = decltype(Layout<Shape < _1,_4>>{});
  using tiled_copy = decltype(make_tiled_copy(copy_atom{}, thr_layout{}, val_layout{}));
  using swizzle_atom = decltype(Swizzle<3,2,3>{});
  using smem_layout_atom = decltype(Layout<Shape < _8,_32>, Stride<_32, _1>>{});
  using gmem_stride_type = decltype(LayoutRight{});
  test_cp_async_with_swizzle<tfloat32_t, cute::Int<128>, cute::Int<32>, gmem_stride_type, swizzle_atom, smem_layout_atom, tiled_copy>();
  }
}
