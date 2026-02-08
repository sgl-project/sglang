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

#include "../hopper/tma_mcast_load_testbed.hpp"

using namespace cute;
using namespace cutlass::test;

#if CUDA_12_0_SM90_FEATURES_SUPPORTED

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout, class CTA_Tile, class Cluster_Size = Int<1>>
auto
test_tma_load(GMEM_Layout  const& gmem_layout,
              SMEM_Layout  const& smem_layout,
              CTA_Tile     const& cta_tile,
              Cluster_Size const& cluster_size = {})
{
  return test_tma_load<T, TmaType>(SM90_TMA_LOAD_MULTICAST{}, gmem_layout, smem_layout, cta_tile, cluster_size);
}

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout>
auto
test_tma_load(GMEM_Layout const& gmem_layout,
              SMEM_Layout const& smem_layout)
{
  return test_tma_load<T, TmaType>(gmem_layout, smem_layout, product_each(shape(smem_layout)));
}

TEST(SM90_CuTe_Hopper, Tma_Load_32x32_Col_MCast)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  {
  Layout gmem_layout = make_layout(make_shape(32,32), GenColMajor{});
  test_tma_load<int8_t>(gmem_layout, smem_layout, shape(smem_layout), Int<2>{});
  test_tma_load<half_t>(gmem_layout, smem_layout, shape(smem_layout), Int<2>{});
  test_tma_load< float>(gmem_layout, smem_layout, shape(smem_layout), Int<2>{});
  test_tma_load<double>(gmem_layout, smem_layout, shape(smem_layout), Int<2>{});

  test_tma_load<int8_t>(gmem_layout, smem_layout, shape(smem_layout), 2);
  test_tma_load<half_t>(gmem_layout, smem_layout, shape(smem_layout), 2);
  test_tma_load< float>(gmem_layout, smem_layout, shape(smem_layout), 2);
  test_tma_load<double>(gmem_layout, smem_layout, shape(smem_layout), 2);
  }
}

#endif
