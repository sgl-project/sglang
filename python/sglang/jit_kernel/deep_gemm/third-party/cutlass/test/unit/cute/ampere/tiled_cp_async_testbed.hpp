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

using namespace cute;

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
};

template <class T, class TiledCopy, class GmemLayout, class SmemLayout>
__global__ void
test_tiled_cp_async_device_cute(T const* g_in, T* g_out,
                     TiledCopy const tiled_copy,
                     GmemLayout gmem_layout, SmemLayout smem_layout)
{
  using namespace cute;

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  auto thr_copy = tiled_copy.get_slice(threadIdx.x);
  Tensor gA = make_tensor(make_gmem_ptr(g_in), gmem_layout);
  Tensor gB = make_tensor(make_gmem_ptr(g_out), gmem_layout);

  // Construct SMEM tensor
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout);  

  auto tAgA = thr_copy.partition_S(gA);
  auto tAsA = thr_copy.partition_D(sA);

#if 0
  if (thread0()) {
    print("gA  : "); print(gA.layout());   print("\n");
    print("sA  : "); print(sA.layout());   print("\n");
    print("tAgA: "); print(tAgA.layout()); print("\n");
    print("tAsA: "); print(tAsA.layout()); print("\n");
  }
#endif

  copy(tiled_copy, tAgA, tAsA);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // Store trivially smem -> gmem

  if (thread0()) {
    copy(sA, gB);
  }

}

template <class T, class TiledCopy, class GMEM_Layout, class SMEM_Layout>
void
test_tiled_cp_async(
               TiledCopy const tiled_copy,
               GMEM_Layout const& gmem_layout,
               SMEM_Layout const& smem_layout)
{
  using namespace cute;

  // Allocate and initialize host test data
  size_t N = ceil_div(cosize(gmem_layout) * sizeof_bits<T>::value, 8);
  thrust::host_vector<T> h_in(N);
  Tensor hA_in  = make_tensor(recast_ptr<T>(h_in.data()), gmem_layout);
  for (int i = 0; i < size(hA_in); ++i) { hA_in(i) = static_cast<T>(i % 13); }

  // Allocate and initialize device test data
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(), T(-1));

  // Launch
  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  test_tiled_cp_async_device_cute<<<1, 128, smem_size>>>(
    reinterpret_cast<T const*>(raw_pointer_cast(d_in.data())),
    reinterpret_cast<T*>      (raw_pointer_cast(d_out.data())),
    tiled_copy,
    gmem_layout,
    smem_layout);

  // Copy results back to host
  thrust::host_vector<T> h_out = d_out;
  Tensor hA_out = make_tensor(recast_ptr<T>(h_out.data()), gmem_layout);

  // Validate the results. Print only the first 3 errors.
  int count = 3;
  for (int i = 0; i < size(hA_out) && count > 0; ++i) {
    EXPECT_EQ(hA_in(i), hA_out(i));
    if (hA_in(i) != hA_out(i)) {
      --count;
    }
  }
}

template <typename T, typename M, typename N, typename GMEM_STRIDE_TYPE, typename SMEM_LAYOUT, typename TILED_COPY>
void test_cp_async_no_swizzle() {
  using namespace cute;
  auto smem_atom = SMEM_LAYOUT{};
  auto smem_layout = tile_to_shape(smem_atom, Shape<M, N>{});
  auto gmem_layout = make_layout(make_shape(M{}, N{}), GMEM_STRIDE_TYPE{});
  test_tiled_cp_async<T>(TILED_COPY{}, gmem_layout, smem_layout);
}

template <typename T, typename M, typename N, typename GMEM_STRIDE_TYPE, typename SWIZZLE_ATOM, typename SMEM_LAYOUT, typename TILED_COPY>
void test_cp_async_with_swizzle() {
  using namespace cute;
  auto swizzle_atom = SWIZZLE_ATOM{};
  auto smem_atom = composition(swizzle_atom, SMEM_LAYOUT{});
  auto smem_layout = tile_to_shape(smem_atom, Shape<M, N>{});
  auto gmem_layout = make_layout(make_shape(M{}, N{}), GMEM_STRIDE_TYPE{});
  test_tiled_cp_async<T>(TILED_COPY{}, gmem_layout, smem_layout);
}
