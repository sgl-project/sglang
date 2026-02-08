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
#include <cute/arch/copy_sm90.hpp>

using namespace cute;

template<class T>
__global__ void
stsm_test_device(uint16_t* g_in, uint16_t* g_out)
{
  constexpr int count = sizeof(T) / 4;
  int tid = threadIdx.x;
  int stride = blockDim.x;

  // load input gmem -> rmem
  uint32_t reg[count];
  for (int i = 0; i < (sizeof(T) / 4); i++) {
    reg[i] = reinterpret_cast<uint32_t*>(g_in)[tid + (stride * i)];
  }

  __shared__ uint32_t smem[32 * count];

  // load rmem -> smem using STSM
  uint128_t* smem_ptr = reinterpret_cast<uint128_t*>(smem) + tid;
  T*         rmem_ptr = reinterpret_cast<T*>(reg);
  cute::copy_stsm(rmem_ptr, smem_ptr);

  __syncthreads();

  // store output smem -> gmem
  for (int i = 0; i < (sizeof(T) / 4); i++) {
    reinterpret_cast<uint32_t*>(g_out)[tid + (stride * i)] = smem[tid + (stride * i)];
  }
}

template <class TiledCopy, class SmemLayout>
__global__ void
stsm_test_device_cute(uint16_t* g_in, uint16_t* g_out,
                      TiledCopy tiled_copy, SmemLayout smem_layout)
{
  using namespace cute;

  __shared__ uint16_t smem[size(smem_layout)];

  Tensor t_g_in  = make_tensor(make_gmem_ptr(g_in),  smem_layout);
  Tensor t_g_out = make_tensor(make_gmem_ptr(g_out), smem_layout);
  Tensor t_smem  = make_tensor(make_smem_ptr(smem),  smem_layout);

  int tid = threadIdx.x;

  auto thr_copy = tiled_copy.get_thread_slice(tid);

  Tensor tXgX = thr_copy.partition_S(t_g_in);   // (V,M,N)
  Tensor tXsX = thr_copy.partition_D(t_smem);   // (V,M,N)

  Tensor tXrX = make_tensor<uint16_t>(shape(tXgX)); // (V,M,N)
  clear(tXrX);    // Just to make sure

/*
  if (thread0()) {
    print("tXsX: " ); print(tXsX.layout()); print("\n");
    print("tXgX: " ); print(tXgX.layout()); print("\n");
    print("tXrX: " ); print(tXrX.layout()); print("\n");
  }
*/

  // Load input gmem -> rmem
  copy(tXgX, tXrX);

  // Copy rmem -> smem via tiled_copy (STSM, STS)
  copy(tiled_copy, tXrX, tXsX);

  // Output smem -> gmem
  for (int i = tid; i < size(t_smem); i += size(tiled_copy)) {
    t_g_out(i) = t_smem(i);
  }
}

#if CUDA_12_0_SM90_FEATURES_SUPPORTED
TEST(SM90_CuTe_Hopper, Stsm)
{
  constexpr int count = 1024;

  thrust::host_vector<uint16_t> h_in(count);
  for (int i = 0; i < count; ++i) {
    h_in[i] = uint16_t(i);
  }
  thrust::device_vector<uint16_t> d_in = h_in;

  //
  // STSM 1x (32b)
  //

  {
  thrust::device_vector<uint16_t> d_out(count);
  stsm_test_device<uint32_t><<<1, 32>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()));
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < 32; ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("STSM 1x stsm_test_device SUCCESS\n");
  }

  //
  // STSM 2x (64b)
  //

  {
  thrust::device_vector<uint16_t> d_out(count);
  stsm_test_device<uint64_t><<<1, 32>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()));
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < 64; ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("STSM 2x stsm_test_device SUCCESS\n");
  }

  //
  // STSM 4x (128b)
  //

  {
  thrust::device_vector<uint16_t> d_out(count);
  stsm_test_device<uint128_t><<<1, 32>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()));
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < 128; ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("STSM 4x stsm_test_device SUCCESS\n");
  }

  //
  // CuTe STSM
  //

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                            Stride< _2,Stride<_1,_64>>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM90_U32x1_STSM_N, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x8 interleaved U32x1_STSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                            Stride< _2,Stride<_1,_64>>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM90_U32x2_STSM_N, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x8 interleaved U32x2_STSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                            Stride< _2,Stride<_1,_64>>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM90_U32x4_STSM_N, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x8 interleaved U32x4_STSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,Shape <_2, _4>>,
                            Stride< _2,Stride<_1,_64>>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, uint16_t>{},
                                    Layout<Shape<_32,_1>>{},
                                    Layout<Shape< _1,_8>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x8 interleaved STSM.U16 SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride< _1,_32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM90_U32x1_STSM_N, uint16_t>{},
                                    Layout<Shape<_16,_2>>{},
                                    Layout<Shape< _2,_4>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U32x1_STSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride< _1,_32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM90_U32x2_STSM_N, uint16_t>{},
                                    Layout<Shape<_16,_2>>{},
                                    Layout<Shape< _2,_4>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U32x2_STSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride< _1,_32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM90_U32x4_STSM_N, uint16_t>{},
                                    Layout<Shape<_16,_2>>{},
                                    Layout<Shape< _2,_4>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U32x4_STSM_N SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride< _1,_32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, uint16_t>{},
                                    Layout<Shape<_16,_2>>{},
                                    Layout<Shape< _2,_4>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 STSM.U16 SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride<_32, _1>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM90_U16x2_STSM_T, uint16_t>{},
                                    Layout<Shape<_4,_8>>{},
                                    Layout<Shape<_2,_1>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U16x2_STSM_T SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride<_32, _1>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM90_U16x4_STSM_T, uint16_t>{},
                                    Layout<Shape<_4,_8>>{},
                                    Layout<Shape<_4,_1>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U16x4_STSM_T SUCCESS\n");
  }

  {
  thrust::device_vector<uint16_t> d_out(count);

  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride<_32, _1>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<SM90_U16x8_STSM_T, uint16_t>{},
                                    Layout<Shape<_4,_8>>{},
                                    Layout<Shape<_8,_1>>{});

  stsm_test_device_cute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  thrust::host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    //printf("%d  %d\n", int(h_in[i]), int(h_out[i]));
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  CUTLASS_TRACE_HOST("CuTe 32x32 U16x8_STSM_T SUCCESS\n");
  }

  CUTLASS_TRACE_HOST("PASS");
}
#endif
