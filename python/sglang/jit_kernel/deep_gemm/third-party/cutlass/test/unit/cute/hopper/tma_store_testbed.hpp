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

#pragma once

#include "cutlass_unit_test.h"

#include <iostream>
#include <cstdint>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

namespace cutlass::test {

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
};

#if CUDA_12_0_SM90_FEATURES_SUPPORTED

template <class T, class TiledCopy, class CTA_Tiler, class GmemLayout, class SmemLayout>
__global__ void
tma_test_device_cute(T const* g_in, T* g_out,
                     CUTE_GRID_CONSTANT TiledCopy const tma, CTA_Tiler cta_tiler,
                     GmemLayout gmem_layout, SmemLayout smem_layout)
{
  using namespace cute;
  CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));

  // Use Shared Storage structure to allocate and distribute aligned SMEM addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Construct SMEM tensor
  Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout);  // (CTA_TILE_M,CTA_TILE_N,...)

  // TMA requires special handling of strides to deal with coord codomain mapping
  // Represent the full tensors -- get these from TMA
  Tensor mA = make_tensor(make_gmem_ptr<T>(g_in), gmem_layout);
  Tensor mB = tma.get_tma_tensor(shape(gmem_layout));

  constexpr int R = rank_v<CTA_Tiler>;
  Tensor gA = flat_divide(mA, cta_tiler);                 // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
  Tensor gB = flat_divide(mB, cta_tiler);                 // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)

  //
  // Prepare the TMA_STORE
  //

  auto cta_tma = tma.get_slice(Int<0>{});                            // CTA slice
  Tensor tBsB_x = cta_tma.partition_S(sB);                           // (TMA,TMA_M,TMA_N)
  Tensor tBgB_x = cta_tma.partition_D(gB);                           // (TMA,TMA_M,TMA_N,REST_M,REST_N)

#if 0
  if (thread0()) {
    print(tma);
    print("TILE  :  "); print(cta_tiler); print("\n");
    print("  mB  :  "); print(  mB.data());   print(" o "); print(  mB.layout());   print("\n");
    print("  gB  :  "); print(  gB.data());   print(" o "); print(  gB.layout());   print("\n");
    print("tBgB_x:  "); print(tBgB_x.data()); print(" o "); print(tBgB_x.layout()); print("\n");
    print("  sB  :  "); print(  sB.data());   print(" o "); print(  sB.layout());   print("\n");
    print("tBsB_x:  "); print(tBsB_x.data()); print(" o "); print(tBsB_x.layout()); print("\n");
  }
#endif

  //
  // Perform the TMA_STORE
  //

  // INPUT: Group the CTA_TILE_X modes and REST_X modes for input
  Tensor tAgA = group_modes<0,R>(group_modes<R,rank(gA)>(gA));       // (CTA_TILE, REST)

  // OUTPUT: Group the REST_X modes and the TMA_X modes to easily iterate through the tiles
  Tensor tBgB = group_modes<1,rank(tBgB_x)>(tBgB_x);                 // (TMA,REST)
  Tensor tBsB = group_modes<1,rank(tBsB_x)>(tBsB_x);                 // (TMA,REST)
  static_assert(size<1>(tBsB) == 1);

#if 0
  if (thread0()) {
    print("tAgA  :  "); print(tAgA.data()); print(" o "); print(tAgA.layout()); print("\n");
    print("tBsB  :  "); print(tBsB.data()); print(" o "); print(tBsB.layout()); print("\n");
    print("tBgB  :  "); print(tBgB.data()); print(" o "); print(tBgB.layout()); print("\n");
  }
#endif

  // Test L2 prefetch
  cooperative_prefetch<128>(threadIdx.x, gA);

  // Loop over the TMA stages, using smem as our buffer
  for (int stage = 0; stage < size<1>(tBgB); ++stage)
  {
    //
    // Read in trivially gmem -> smem
    //
    // Subbyte elements could cause race conditions, so be even more conservative
    if (thread0()) {
      copy(tAgA(_,stage), sB);
    }

    __syncthreads();
    cute::cp_async_wait<0>();

    //
    // Perform the TMA_STORE
    //

    if (threadIdx.x == 0) {
      copy(tma, tBsB(_,0), tBgB(_,stage));
    }

    tma_store_wait<0>();
    __syncthreads();
  }
}

template <class T, class TmaType = T, class CopyOp, class GMEM_Layout, class SMEM_Layout, class CTA_Tile>
void
test_tma_store(CopyOp      const& copy_op,
               GMEM_Layout const& gmem_layout,
               SMEM_Layout const& smem_layout,
               CTA_Tile    const& cta_tile)
{
  using namespace cute;

  // Allocate and initialize host test data
  size_t N = ceil_div(cosize(gmem_layout) * sizeof_bits<T>::value, 8);
  thrust::host_vector<uint8_t> h_in(N);
  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = uint8_t(i % 13);
  }
  Tensor hA_in  = make_tensor(recast_ptr<T>(h_in.data()), gmem_layout);

  // Allocate and initialize device test data
  thrust::device_vector<uint8_t> d_in = h_in;
  thrust::device_vector<uint8_t> d_out(h_in.size(), uint8_t(-1)); // overflow uint

  // Create TMA for this device Tensor
  Tensor gA = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_out.data())), gmem_layout);
  auto tma = make_tma_copy<TmaType>(copy_op, gA, smem_layout, cta_tile, Int<1>{});
  //print(tma);

  // Launch
  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
    reinterpret_cast<T const*>(raw_pointer_cast(d_in.data())),
    reinterpret_cast<T*>      (raw_pointer_cast(d_out.data())),
    tma, cta_tile,
    gmem_layout,
    smem_layout);

  // Copy results back to host
  thrust::host_vector<uint8_t> h_out = d_out;
  Tensor hA_out = make_tensor(recast_ptr<T>(h_out.data()), gmem_layout);

  // Validate the results. Print only the first 3 errors.
  int count = 3;
  for (int i = 0; i < int(size(hA_out)) && count > 0; ++i) {
    EXPECT_EQ(hA_in(i), hA_out(i));
    if (hA_in(i) != hA_out(i)) {
      --count;
    }
  }
}

#endif

} // end namespace cutlass::test
