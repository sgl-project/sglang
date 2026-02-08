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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cute/atom/copy_traits_sm75.hpp>

using namespace cute;

__global__ void
movm_test_device(uint16_t* g_in, uint16_t* g_out)
{
  int tid = threadIdx.x;

  // load input gmem -> register
  uint32_t reg = reinterpret_cast<uint32_t*>(g_in)[tid];

  // do two movmatrix calls (transpose twice => identity)
  uint32_t tmp = 0;
  uint32_t dst = 0;
  SM75_U32x1_MOVM_T::copy(reg, tmp);
  SM75_U32x1_MOVM_T::copy(tmp, dst);

  // store result
  reinterpret_cast<uint32_t*>(g_out)[tid] = dst;
}

template <class TiledCopy, class GmemLayout>
__global__ void
movm_test_device_cute(uint16_t* g_in, uint16_t* g_out,
                      TiledCopy tiled_copy, GmemLayout gmem_layout)
{
  using namespace cute;

  auto t_g_in  = make_tensor(make_gmem_ptr(reinterpret_cast<uint32_t*>(g_in)),  gmem_layout);
  auto t_g_out = make_tensor(make_gmem_ptr(reinterpret_cast<uint32_t*>(g_out)), gmem_layout);

  int tid = threadIdx.x;

  auto thr_copy = tiled_copy.get_thread_slice(tid);

  auto tXgS = thr_copy.partition_S(t_g_in);
  auto tXgD = thr_copy.partition_D(t_g_out);

  // Register tensors for intermediate and output data
  auto tXrS = make_tensor<uint32_t>(shape(tXgS)); // src
  auto tXrT = make_tensor<uint32_t>(shape(tXgS)); // tmp
  auto tXrD = make_tensor<uint32_t>(shape(tXgD)); // dst
  clear(tXrS);
  clear(tXrT);
  clear(tXrD);

  // Load gmem -> registers
  for (int i = 0; i < size(tXrS); ++i) {
    tXrS(i) = tXgS(i);
  }

  // do two movmatrix calls for identity
  copy(tiled_copy, tXrS, tXrT);
  copy(tiled_copy, tXrT, tXrD);

  // Store registers -> gmem
  for (int i = 0; i < size(tXrD); ++i) {
    tXgD(i) = tXrD(i);
  }
}

TEST(SM75_CuTe_Turing, Movm)
{
  constexpr int count = 1024;

  thrust::host_vector<uint16_t> h_in(count);
  for (int i = 0; i < count; ++i) {
    h_in[i] = uint16_t(i);
  } 
  thrust::device_vector<uint16_t> d_in = h_in;

  //
  // Direct MOVM
  //

  {
  thrust::device_vector<uint16_t> d_out(count);
  movm_test_device<<<1, 32>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()));
  thrust::host_vector<uint16_t> h_out = d_out;
  // applied movmatrix twice so result should equal input
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("MOVM movm_test_device SUCCESS\n");
  }

  //
  // CuTe MOVM
  //

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto gmem_layout = Layout<Shape <_32, _1>, 
                            Stride< _1,_32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x1_MOVM_T, uint32_t>{},
                                    Layout<Shape<_32, _1>>{}, 
                                    Layout<Shape< _1, _1>>{});

  movm_test_device_cute<<<1, int(size(tiled_copy))>>>(                              
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    gmem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < (size(gmem_layout)*2); ++i) {
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe MOVM SUCCESS\n");
  }

  CUTLASS_TRACE_HOST("PASS");
}
