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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cute/atom/copy_traits_sm75.hpp>


using namespace cute;

template <class T>
__global__ void
ldsm_test_device(uint16_t* g_in, uint16_t* g_out)
{
  constexpr int count = sizeof(T) / 4;
  int tid = threadIdx.x;
  int stride = blockDim.x;

  // load input gmem -> smem
  __shared__ uint32_t smem[32 * count];
  for (int i = 0; i < count; ++i) {
    smem[tid + (stride * i)] = reinterpret_cast<uint32_t*>(g_in)[tid + (stride * i)];
  }

  __syncthreads();

  uint32_t reg[count];
  for (int i = 0; i < count; ++i) {
    reg[i] = 0;
  }

  // load smem -> rmem using LDSM
  uint128_t* smem_ptr = reinterpret_cast<uint128_t*>(smem) + tid;
  T*         rmem_ptr = reinterpret_cast<T*>(reg);
  cute::copy_ldsm(smem_ptr, rmem_ptr);

  // store output rmem -> gmem
  for (int i = 0; i < count; ++i) {
    reinterpret_cast<uint32_t*>(g_out)[tid + (stride * i)] = reg[i];
  }
}

template <class TiledCopy, class SmemLayout>
__global__ void
ldsm_test_device_cute(uint16_t* g_in, uint16_t* g_out,
                      TiledCopy tiled_copy, SmemLayout smem_layout)
{
  using namespace cute;

  __shared__ uint16_t smem[size(smem_layout)];

  auto t_g_in  = make_tensor(make_gmem_ptr(g_in),  smem_layout);
  auto t_g_out = make_tensor(make_gmem_ptr(g_out), smem_layout);
  auto t_smem  = make_tensor(make_smem_ptr(smem),  smem_layout);

  int tid = threadIdx.x;

  // Load input gmem -> smem
  for (int i = tid; i < size(t_smem); i += size(tiled_copy)) {
    t_smem(i) = t_g_in(i);
  }

  __syncthreads();

  auto thr_copy = tiled_copy.get_thread_slice(tid);

  auto tXsX = thr_copy.partition_S(t_smem);   // (V,M,N)
  auto tXgX = thr_copy.partition_D(t_g_out);  // (V,M,N)

  auto tXrX = make_tensor<uint16_t>(shape(tXgX)); // (V,M,N)
  clear(tXrX);  // Just to make sure

/*
  if (thread0()) {
    print("tXsX: " ); print(tXsX.layout()); print("\n");
    print("tXgX: " ); print(tXgX.layout()); print("\n");
    print("tXrX: " ); print(tXrX.layout()); print("\n");
  }
*/

  // Copy smem -> rmem via tiled_copy (LDSM, LDS)
  copy(tiled_copy, tXsX, tXrX);

  // Output rmem -> gmem
  copy(tXrX, tXgX);
}


TEST(SM80_CuTe_Ampere, Ldsm)
{
  constexpr int count = 1024;

  thrust::host_vector<uint16_t> h_in(count);
  for (int i = 0; i < count; ++i) {
    h_in[i] = uint16_t(i);
  }
  thrust::device_vector<uint16_t> d_in = h_in;

  //
  // LDSM 1x (32b)
  //

  {
  thrust::device_vector<uint16_t> d_out(count);
  ldsm_test_device<uint32_t><<<1, 32>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()));
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("LDSM 1x ldsm_test_device SUCCESS\n");
  }

  //
  // LDSM 2x (64b)
  //

  {
  thrust::device_vector<uint16_t> d_out(count);
  ldsm_test_device<uint64_t><<<1, 32>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()));
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < 64; ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("LDSM 2x ldsm_test_device SUCCESS\n");
  }

  //
  // LDSM 4x (128b)
  //

  {
  thrust::device_vector<uint16_t> d_out(count);
  ldsm_test_device<uint128_t><<<1, 32>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()));
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < 128; ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("LDSM 4x ldsm_test_device SUCCESS\n");
  }

  //
  // CuTe LDSM
  //

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                            Stride< _2,Stride<_1,_64>>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x1_LDSM_N, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x8 interleaved U32x1_LDSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                            Stride< _2,Stride<_1,_64>>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x2_LDSM_N, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x8 interleaved U32x2_LDSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                            Stride< _2,Stride<_1,_64>>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x4_LDSM_N, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x8 interleaved U32x4_LDSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                            Stride< _2,Stride<_1,_64>>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i] , h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x8 interleaved LDS.U16 SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride< _1,_32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x1_LDSM_N, uint16_t>{},
                                    Layout<Shape<_16,_2>>{},
                                    Layout<Shape< _2,_4>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U32x1_LDSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride< _1,_32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x2_LDSM_N, uint16_t>{},
                                    Layout<Shape<_16,_2>>{},
                                    Layout<Shape< _2,_4>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U32x2_LDSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride< _1,_32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U32x4_LDSM_N, uint16_t>{},
                                    Layout<Shape<_16,_2>>{},
                                    Layout<Shape< _2,_4>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U32x4_LDSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride< _1,_32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, uint16_t>{},
                                    Layout<Shape<_16,_2>>{},
                                    Layout<Shape< _2,_4>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 LDS.U16 SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride<_32, _1>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U16x2_LDSM_T, uint16_t>{},
                                    Layout<Shape<_4,_8>>{},
                                    Layout<Shape<_2,_1>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i],  h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U16x2_LDSM_T SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride<_32, _1>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U16x4_LDSM_T, uint16_t>{},
                                    Layout<Shape<_4,_8>>{},
                                    Layout<Shape<_4,_1>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i],  h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U16x4_LDSM_T SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride<_32, _1>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM75_U16x8_LDSM_T, uint16_t>{},
                                    Layout<Shape<_4,_8>>{},
                                    Layout<Shape<_8,_1>>{});

  ldsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U16x8_LDSM_T SUCCESS\n");
  }

  CUTLASS_TRACE_HOST("PASS");
}
