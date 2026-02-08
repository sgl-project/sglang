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

/*! \file
  \brief Basic tests for BULK_COPY usage with various layouts.
*/

#include "cutlass_unit_test.h"

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

using namespace cute;

template <class ElementType, class SmemLayout>
struct SharedStorage {
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayout>> smem;
};

#if CUDA_12_0_SM90_FEATURES_SUPPORTED
template <class T, class GmemLayout, class SmemLayout>
__global__ void
bulk_copy_test_device_cute(T const* g_in,
                           T      * g_out,
                           GmemLayout gmem_layout,
                           SmemLayout smem_layout)
{
  // Use Shared Storage structure to allocate and distribute aligned SMEM addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Construct SMEM tensor
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.data()), smem_layout);
  // Construct the GMEM tensor
  Tensor gA = make_tensor(make_gmem_ptr(g_in), gmem_layout);

  //
  // Read in trivially
  //

  // Input gmem -> smem
  for (int i = threadIdx.x; i < size(sA); i += blockDim.x) {
    sA(i) = gA(i);
  }

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  //
  // Perform the BULK_COPY store
  //

#if 0
  if (thread0()) {
    print("sA: "); print(sA.data()); print(" o "); print(sA.layout()); print("\n");
    print("gA: "); print(gA.data()); print(" o "); print(gA.layout()); print("\n");
  }
#endif

  Tensor gA_out = make_tensor(make_gmem_ptr(g_out), gmem_layout);

  auto blkcp = Copy_Traits<SM90_BULK_COPY_AUTO>{};

  copy(blkcp, sA, gA_out);
  // Bulk Copy store requires the same sync as TMA store.
  tma_store_arrive();
  tma_store_wait<0>();
}

template <class T, class GLayout, class SLayout>
void run_and_validate(GLayout gmem_layout,
                      SLayout smem_layout)
{
  thrust::host_vector<T> h_in(cosize(gmem_layout));
  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = static_cast<T>(int(i));
  }

  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(d_in.size(), T(-1));

  int32_t smem_size = static_cast<int32_t>(sizeof(SharedStorage<T, decltype(smem_layout)>));
  bulk_copy_test_device_cute<<<1, 128, smem_size>>>(thrust::raw_pointer_cast(d_in.data()),
                                                    thrust::raw_pointer_cast(d_out.data()),
                                                    gmem_layout,
                                                    smem_layout);
  // Transfering results back to host
  thrust::host_vector<T> h_out = d_out;

  // Validate the results
  for (int i = 0; i < cute::size(gmem_layout); ++i) {
    int k = gmem_layout(i);
    EXPECT_EQ(int(h_in[k]), int(h_out[k]));
  }
}

// }  // namespace

TEST(SM90_CuTe_BLKCP, ColMajor)
{
  auto smem_layout = make_layout(Shape<_32,_32>{}, GenColMajor{});
  auto gmem_layout = smem_layout;
  run_and_validate<    int8_t>(gmem_layout, smem_layout);
  run_and_validate<    half_t>(gmem_layout, smem_layout);
  run_and_validate<tfloat32_t>(gmem_layout, smem_layout);
}

TEST(SM90_CuTe_BLKCP, RowMajor)
{
  auto smem_layout = make_layout(Shape<_32,_32>{}, GenRowMajor{});
  auto gmem_layout = smem_layout;
  run_and_validate<    int8_t>(gmem_layout, smem_layout);
  run_and_validate<    half_t>(gmem_layout, smem_layout);
  run_and_validate<tfloat32_t>(gmem_layout, smem_layout);
}

TEST(SM90_CuTe_BLKCP, NonCompact)
{
  {
  auto smem_layout = make_layout(Shape<_32,_32>{}, Stride<_1,Int<48>>{});
  auto gmem_layout = smem_layout;
  run_and_validate<    int8_t>(gmem_layout, smem_layout);
  run_and_validate<    half_t>(gmem_layout, smem_layout);
  run_and_validate<tfloat32_t>(gmem_layout, smem_layout);
  }
  {
  auto smem_layout = make_layout(Shape<_32,_32>{}, Stride<_1,Int<48>>{});
  auto gmem_layout = make_layout(Shape<Shape<_16,_2>, Shape<_4,_8>>{}, Stride<Stride<_1,_64>,Stride<_16,_128>>{});
  run_and_validate<    int8_t>(gmem_layout, smem_layout);
  run_and_validate<    half_t>(gmem_layout, smem_layout);
  run_and_validate<tfloat32_t>(gmem_layout, smem_layout);
  }
  {
  auto smem_layout = make_layout(Shape<_32,_32>{}, Stride<_64,_1>{});
  auto gmem_layout = smem_layout;
  run_and_validate<    int8_t>(gmem_layout, smem_layout);
  run_and_validate<    half_t>(gmem_layout, smem_layout);
  run_and_validate<tfloat32_t>(gmem_layout, smem_layout);
  }
}
#endif // #if CUDA_12_0_SM90_FEATURES_SUPPORTED
