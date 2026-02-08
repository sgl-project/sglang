
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

template <class GmemTensor, class RmemTiler, class CopyPolicy>
__global__
void
kernel(GmemTensor gC, RmemTiler tiler, CopyPolicy policy)
{
  Tensor tCgC = local_tile(gC, tiler, 0);

  Tensor rC = make_tensor_like(tCgC);
  using T = typename GmemTensor::value_type;
  for (int i = 0; i < size(rC); ++i) {
    rC(i) = T(i % 13);
  }

#if 0
  print("  gC : "); print(  gC); print("\n");
  print("tCgC : "); print(tCgC); print("\n");
  print("  rC : "); print(  rC); print("\n");
#endif

  // NOTE: only 1 thread, this thread produce a block of 8x8 output. The fringe will not be touched.
  //copy(rC, tCgC);           // Enable auto-vectorization if static
  copy(policy, rC, tCgC);     // Use a policy to establish vectorization assumptions
}

template <class T, class CopyPolicy, class GmemLayout, class RmemTiler>
void
test_copy_vectorization(CopyPolicy policy, GmemLayout gmem_layout, RmemTiler rmem_tiler)
{
  thrust::host_vector<T> h_in(cosize(gmem_layout), T(0));

  thrust::device_vector<T> d_in = h_in;
  Tensor m_in = make_tensor(make_gmem_ptr(raw_pointer_cast(d_in.data())), gmem_layout);

  kernel<<<1,1>>>(m_in, rmem_tiler, policy);

  thrust::host_vector<T> h_out = d_in;
  Tensor result = make_tensor(h_out.data(), gmem_layout);

  thrust::host_vector<T> h_true = h_in;
  Tensor ref = make_tensor(h_true.data(), gmem_layout);

  // Set the values directly in the reference tensor, no copy
  Tensor ref_tile = local_tile(ref, rmem_tiler, 0);
  for (int i = 0; i < size(ref_tile); ++i) {
    ref_tile(i) = T(i % 13);
  }

  // Compare the reference and the result. Print only the first 3 errors.
  // print_tensor(result);
  int count = 3;
  for (int i = 0; i < size(ref) && count > 0; ++i) {
    EXPECT_EQ(result(i), ref(i));
    if (result(i) != ref(i)) {
      --count;
    }
  }
}

template <class T, class GmemLayout, class RmemTiler>
void
test_copy_vectorization(GmemLayout gmem_layout, RmemTiler rmem_tiler)
{
  test_copy_vectorization<T>(DefaultCopy{}, gmem_layout, rmem_tiler);
}

TEST(SM70_CuTe_Volta, SimpleVec)
{
  // Fully static layouts are assumed to be aligned -- these will be vectorized
  test_copy_vectorization<float>(make_layout(make_shape(Int<8>{}, Int<8>{})), Shape<_8,_8>{});
  test_copy_vectorization<float>(make_layout(make_shape(Int<12>{}, Int<12>{})), Shape<_8,_8>{});
  // Fails in vectorization recast due to misalignment and static assertions
  //test_copy_vectorization<float>(make_layout(make_shape(Int<9>{}, Int<9>{})), Shape<_8,_8>{});

  // Dynamic layouts are not assumed to be aligned -- these will not be vectorized
  test_copy_vectorization<float>(make_layout(make_shape(12,12)), Shape<_8,_8>{});
  test_copy_vectorization<float>(make_layout(make_shape( 9, 9)), Shape<_8,_8>{});

  // Dynamic layouts that are assumed to be aligned -- these will be vectorized
  test_copy_vectorization<float>(AutoVectorizingCopyWithAssumedAlignment<128>{}, make_layout(make_shape( 8, 8)), Shape<_8,_8>{});
  test_copy_vectorization<float>(AutoVectorizingCopyWithAssumedAlignment<128>{}, make_layout(make_shape(12,12)), Shape<_8,_8>{});
  // Fails -- bad alignment assumption
  //test_copy_vectorization<float>(AutoVectorizingCopyWithAssumedAlignment<128>{}, make_layout(make_shape( 9, 9)), Shape<_8,_8>{});
}
