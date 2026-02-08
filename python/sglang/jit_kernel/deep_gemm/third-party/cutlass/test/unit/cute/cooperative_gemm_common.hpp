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

#pragma once

#include "cutlass/relatively_equal.h"
#include "cutlass_unit_test.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

using namespace cute;

template<typename T>
struct fp64_tester {
  using value_type = double;
};

template<typename T>
struct fp64_tester<complex<T>> {
  using value_type = complex<double>;
};

template<class TA,
         class TB,
         class TC,
         class ALayout, // logical shape (M, K)
         class BLayout, // logical shape (N, K)
         class CLayout> // logical shape (M, N)
auto host_generate_gemm_inputs(
  ALayout a_layout,
  BLayout b_layout,
  CLayout c_layout
) {
  thrust::host_vector<TA> h_a(cosize(a_layout));
  thrust::host_vector<TB> h_b(cosize(b_layout));
  thrust::host_vector<TC> h_c(cosize(c_layout));
  thrust::host_vector<TC> h_c_out(cosize(c_layout));

  auto h_a_tensor = make_tensor(h_a.data(), a_layout);
  auto h_b_tensor = make_tensor(h_b.data(), b_layout);
  auto h_c_tensor = make_tensor(h_c.data(), c_layout);
  size_t max_size   = std::max<size_t>({static_cast<size_t>(size(a_layout)),
                                        static_cast<size_t>(size(b_layout)),
                                        static_cast<size_t>(size(c_layout))});
  for (size_t i = 0; i < max_size; ++i) {
    double di = static_cast<double>(i);
    if(i < size(a_layout)) {
      h_a_tensor(i) = static_cast<TA>(di / size(a_layout));
    }
    if(i < size(b_layout)) {
      h_b_tensor(i) = static_cast<TB>(di / size(a_layout));
    }
    if(i < size(c_layout)) {
      h_c_tensor(i) = static_cast<TC>((di*di) / size(a_layout));
    }
  }

  return std::make_tuple(h_a, h_b, h_c, h_c_out);
}

template<class Alpha, class EngineA, class ALayout,
         class EngineB, class BLayout,
         class Beta, class EngineC, class CLayout,
         class ALoadTransform  = cute::identity,
         class BLoadTransform  = cute::identity,
         class CLoadTransform  = cute::identity,
         class CStoreTransform = cute::identity>
thrust::host_vector<typename EngineC::value_type>
host_reference_gemm(Alpha                           alpha,
                    Tensor<EngineA, ALayout> const& h_a_tensor,
                    Tensor<EngineB, BLayout> const& h_b_tensor,
                    Beta                            beta,
                    Tensor<EngineC, CLayout> const& h_c_tensor,
                    ALoadTransform           const& a_load_transform = {},
                    BLoadTransform           const& b_load_transform = {},
                    CLoadTransform           const& c_load_transform = {},
                    CStoreTransform          const& c_store_transform = {})
  {
  // Cannot use ::value_type because it propagates to complex::value_type,
  // so ViewEngine<complex<double>>::value_type == double
  using TA = remove_cv_t<typename EngineA::element_type>;
  using TB = remove_cv_t<typename EngineB::element_type>;
  using TC = remove_cv_t<typename EngineC::element_type>;

  using tester = fp64_tester<TC>;
  using ABC_64 = typename tester::value_type;

  static_assert(std::is_same_v<typename fp64_tester<TA>::value_type, typename fp64_tester<TB>::value_type>);
  static_assert(std::is_same_v<typename fp64_tester<TB>::value_type, typename fp64_tester<TC>::value_type>);

  thrust::host_vector<TC> h_c_ref(cosize(h_c_tensor.layout()), static_cast<TC>(0.0));
  auto h_c_ref_tensor = make_tensor(h_c_ref.data(), h_c_tensor.layout());
  // A * B
  for (int k = 0; k < size<1>(h_a_tensor); k++) {
    for (int m = 0; m < size<0>(h_a_tensor); m++) {
      for (int n = 0; n < size<0>(h_b_tensor); n++) {
          const auto a_value      = a_load_transform(h_a_tensor(m, k));
          const auto b_value      = b_load_transform(h_b_tensor(n, k));
          const auto a_value_fp64 = static_cast<ABC_64>(a_value);
          const auto b_value_fp64 = static_cast<ABC_64>(b_value);
          h_c_ref_tensor(m, n) += static_cast<TC>(a_value_fp64 * b_value_fp64);
      }
    }
  }
  // C = A*B + C
  for (int i = 0; i < size(h_c_ref_tensor); i++) {
    const auto ab_value_fp64 = static_cast<ABC_64>(h_c_ref_tensor(i));
    const auto c_value_fp64  = static_cast<ABC_64>(c_load_transform(h_c_tensor(i)));
    h_c_ref_tensor(i)        = c_store_transform(static_cast<TC>(alpha * ab_value_fp64 + beta * c_value_fp64));
  }

  return h_c_ref;
}

template<class EngineC, class CLayout>
void verify_gemm_correctness(cute::Tensor<EngineC, CLayout> const& h_c_out_tensor,
                             cute::Tensor<EngineC, CLayout> const& h_c_ref_tensor)
{
  // Cannot use ::value_type because it propagates to complex::value_type,
  // so ViewEngine<complex<double>>::value_type == double
  using TC = remove_cv_t<typename EngineC::element_type>;

  using tester = fp64_tester<TC>;
  using ABC_64 = typename tester::value_type;

  for (int i = 0; i < size(h_c_ref_tensor); i++) {
    ABC_64 h_c_ref_i = h_c_ref_tensor(i);
    ABC_64 h_c_out_i = h_c_out_tensor(i);
    double epsilon(0.1f);
    double nonzero_floor(std::numeric_limits<double>::min());
    bool passed = cutlass::relatively_equal(h_c_out_i, h_c_ref_i, epsilon, nonzero_floor);
    ASSERT_TRUE(passed) << i << " - result:" << h_c_out_i << " expected:" << h_c_ref_i;
  }
}


template<uint32_t ThreadBlockSize,
         uint32_t CopyMaxVecBits,
         class GMemALayout,
         class GMemBLayout,
         class GMemCLayout,
         class SMemALayout,
         class SMemBLayout,
         class SMemCLayout,
         class TA,
         class TB,
         class TC,
         class Alpha,
         class Beta,
         class TiledMma,
         class ALoadTransform,
         class BLoadTransform,
         class CLoadTransform,
         class CStoreTransform,
         class SMemCopyOpA,
         class SMemCopyOpB,
         class SMemCopyLdOpC,
         class SMemCopyStOpC>
__launch_bounds__(ThreadBlockSize) __global__ void
cooperative_gemm_kernel(GMemALayout gmem_a_layout,
                        GMemBLayout gmem_b_layout,
                        GMemCLayout gmem_c_layout,
                        SMemALayout smem_a_layout,
                        SMemBLayout smem_b_layout,
                        SMemCLayout smem_c_layout,
                        TA       const* a,
                        TB       const* b,
                        TC       const* c,
                        TC            * c_out,
                        Alpha    const  alpha,
                        Beta     const  beta,
                        TiledMma        tiled_mma,
                        ALoadTransform  a_load_transform,
                        BLoadTransform  b_load_transform,
                        CLoadTransform  c_load_transform,
                        CStoreTransform c_store_transform,
                        SMemCopyOpA     a_copy_op,
                        SMemCopyOpB     b_copy_op,
                        SMemCopyLdOpC   c_copy_ld_op,
                        SMemCopyStOpC   c_copy_st_op)
{
    using namespace cute;

    Tensor g_a_tensor     = make_tensor(make_gmem_ptr(a), gmem_a_layout);
    Tensor g_b_tensor     = make_tensor(make_gmem_ptr(b), gmem_b_layout);
    Tensor g_c_tensor     = make_tensor(make_gmem_ptr(c), gmem_c_layout);
    Tensor g_c_out_tensor = make_tensor(make_gmem_ptr(c_out), gmem_c_layout);

    constexpr uint32_t copy_max_vec_bytes = CopyMaxVecBits / 8;

    extern __shared__ float4 smem_buf[];
    auto* smem_ptr = reinterpret_cast<unsigned char*>(smem_buf);
    auto* smem_ptr_a = smem_ptr;
    auto* smem_ptr_b = smem_ptr_a + round_up((sizeof(TA) * cosize(smem_a_layout)), copy_max_vec_bytes);
    auto* smem_ptr_c = smem_ptr_b + round_up((sizeof(TB) * cosize(smem_b_layout)), copy_max_vec_bytes);

    Tensor s_a_tensor = make_tensor(make_smem_ptr<TA>(smem_ptr_a), smem_a_layout);
    Tensor s_b_tensor = make_tensor(make_smem_ptr<TB>(smem_ptr_b), smem_b_layout);
    Tensor s_c_tensor = make_tensor(make_smem_ptr<TC>(smem_ptr_c), smem_c_layout);

    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_a_tensor, s_a_tensor);
    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_b_tensor, s_b_tensor);
    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_c_tensor, s_c_tensor);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    cooperative_gemm(
      threadIdx.x, tiled_mma,
      alpha, s_a_tensor, s_b_tensor, beta, s_c_tensor,
      a_load_transform, b_load_transform, c_load_transform, c_store_transform,
      a_copy_op, b_copy_op, c_copy_ld_op, c_copy_st_op
    );
    __syncthreads();

    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, s_c_tensor, g_c_out_tensor);
}

template<uint32_t ThreadBlockSize,
         uint32_t CopyMaxVecBits,
         class GMemALayout,
         class GMemBLayout,
         class GMemCLayout,
         class SMemALayout,
         class SMemBLayout,
         class TA,
         class TB,
         class TC,
         class TiledMma,
         class ALoadTransform,
         class BLoadTransform,
         class CLoadTransform,
         class CStoreTransform,
         class SMemCopyOpA,
         class SMemCopyOpB>
__launch_bounds__(ThreadBlockSize) __global__ void
cooperative_gemm_kernel_rmem_c(GMemALayout gmem_a_layout,
                               GMemBLayout gmem_b_layout,
                               GMemCLayout gmem_c_layout,
                               SMemALayout smem_a_layout,
                               SMemBLayout smem_b_layout,
                               TA        const* a,
                               TB        const* b,
                               TC        const* c,
                               TC             * c_out,
                               TiledMma         tiled_mma,
                               ALoadTransform   a_load_transform,
                               BLoadTransform   b_load_transform,
                               CLoadTransform   c_load_transform,
                               CStoreTransform  c_store_transform,
                               SMemCopyOpA      a_copy_op,
                               SMemCopyOpB      b_copy_op)
  {
    using namespace cute;

    Tensor g_a_tensor     = make_tensor(make_gmem_ptr(a), gmem_a_layout);
    Tensor g_b_tensor     = make_tensor(make_gmem_ptr(b), gmem_b_layout);
    Tensor g_c_tensor     = make_tensor(make_gmem_ptr(c), gmem_c_layout);
    Tensor g_c_out_tensor = make_tensor(make_gmem_ptr(c_out), gmem_c_layout);

    constexpr uint32_t copy_max_vec_bytes = CopyMaxVecBits / 8;

    extern __shared__ float4 smem_buf[];
    auto* smem_ptr = reinterpret_cast<unsigned char*>(smem_buf);
    auto* smem_ptr_a = smem_ptr;
    auto* smem_ptr_b = smem_ptr_a + round_up((sizeof(TA) * cosize(smem_a_layout)), copy_max_vec_bytes);

    Tensor s_a_tensor = make_tensor(make_smem_ptr<TA>(smem_ptr_a), smem_a_layout);
    Tensor s_b_tensor = make_tensor(make_smem_ptr<TB>(smem_ptr_b), smem_b_layout);

    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_a_tensor, s_a_tensor);
    cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(threadIdx.x, g_b_tensor, s_b_tensor);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Create C fragment for storing intermediate results
    auto thr_mma = TiledMma().get_thread_slice(threadIdx.x);
    Tensor g_c_partition = thr_mma.partition_C(g_c_tensor);
    Tensor g_c_out_partition = thr_mma.partition_C(g_c_out_tensor);
    Tensor r_c_partition = thr_mma.make_fragment_C(g_c_partition);

    // Create indexing help for predicated GEMMs
    Tensor cC   = make_identity_tensor(shape(gmem_c_layout));
    Tensor tCcC = thr_mma.partition_C(cC);

    // Load C from global
    // (always loading in predicated way)
    CUTE_UNROLL
    for (int i = 0; i < size(r_c_partition); ++i)
    {
      if (elem_less(tCcC(i), shape(g_c_tensor)))
      {
        r_c_partition(i) = c_load_transform(g_c_partition(i));
      }
    }

    cooperative_gemm(
      threadIdx.x, tiled_mma, s_a_tensor, s_b_tensor, r_c_partition,
      a_load_transform, b_load_transform, a_copy_op, b_copy_op
    );

    __syncthreads();

    // Store C to global
    // (always storing in predicated way)
    CUTE_UNROLL
    for (int i = 0; i < size(r_c_partition); ++i)
    {
      if (elem_less(tCcC(i), shape(g_c_tensor)))
      {
        g_c_out_partition(i) = c_store_transform(r_c_partition(i));
      }
    }
}

template<uint32_t ThreadBlockSize,
         uint32_t CopyMaxVecBits,
         class TA,
         class TB,
         class TC,
         class GMemALayout, // logical shape (M, K)
         class GMemBLayout, // logical shape (N, K)
         class GMemCLayout, // logical shape (M, N)
         class SMemALayout, // logical shape (M, K)
         class SMemBLayout, // logical shape (N, K)
         class SMemCLayout, // logical shape (M, N)
         class TiledMma,
         class ALoadTransform = cute::identity,
         class BLoadTransform = cute::identity,
         class CLoadTransform = cute::identity,
         class CStoreTransform = cute::identity,
         class ASMemCopyOp = AutoVectorizingCopyWithAssumedAlignment<CopyMaxVecBits>,
         class BSMemCopyOp = AutoVectorizingCopyWithAssumedAlignment<CopyMaxVecBits>,
         class CSMemCopyLdOp = AutoVectorizingCopyWithAssumedAlignment<CopyMaxVecBits>,
         class CSMemCopyStOp = AutoVectorizingCopyWithAssumedAlignment<CopyMaxVecBits>>
void test_cooperative_gemm(GMemALayout     gmem_a_layout,
                           GMemBLayout     gmem_b_layout,
                           GMemCLayout     gmem_c_layout,
                           SMemALayout     smem_a_layout,
                           SMemBLayout     smem_b_layout,
                           SMemCLayout     smem_c_layout,
                           TiledMma        tiled_mma,
                           ALoadTransform  a_load_transform  = {},
                           BLoadTransform  b_load_transform  = {},
                           CLoadTransform  c_load_transform  = {},
                           CStoreTransform c_store_transform = {},
                           ASMemCopyOp     a_smem_copy_op = {},
                           BSMemCopyOp     b_smem_copy_op = {},
                           CSMemCopyLdOp   c_smem_copy_ld_op = {},
                           CSMemCopyStOp   c_smem_copy_st_op = {})
{
  static_assert(std::is_same_v<typename fp64_tester<TA>::value_type, typename fp64_tester<TB>::value_type>);
  static_assert(std::is_same_v<typename fp64_tester<TB>::value_type, typename fp64_tester<TC>::value_type>);

  static_assert(size<0>(gmem_a_layout) == size<0>(gmem_c_layout));  // AM == CM
  static_assert(size<0>(gmem_b_layout) == size<1>(gmem_c_layout));  // BN == CN
  static_assert(size<1>(gmem_a_layout) == size<1>(gmem_b_layout));  // AK == BK

  static_assert(size<0>(smem_a_layout) == size<0>(smem_c_layout));  // AM == CM
  static_assert(size<0>(smem_b_layout) == size<1>(smem_c_layout));  // BN == CN
  static_assert(size<1>(smem_a_layout) == size<1>(smem_b_layout));  // AK == BK

  static_assert(cute::size(gmem_a_layout) == cute::size(smem_a_layout));
  static_assert(cute::size(gmem_b_layout) == cute::size(smem_b_layout));
  static_assert(cute::size(gmem_c_layout) == cute::size(smem_c_layout));

#if 0
  print("   "); print("gmem:    "); print(gmem_layout); print("\n");
  print("   "); print("smem:    "); print(smem_layout); print("\n");
  print("   "); print("threads: "); print(ThreadBlockSize); print("\n");
#endif

  const auto alpha = static_cast<TC>(1.1);
  const auto beta  = static_cast<TC>(1.2);

  // Generate inputs
  auto [h_a, h_b, h_c, h_c_out] = host_generate_gemm_inputs<TA, TB, TC>(gmem_a_layout, gmem_b_layout, gmem_c_layout);

  thrust::device_vector<TA> d_a(h_a);
  thrust::device_vector<TB> d_b(h_b);
  thrust::device_vector<TC> d_c(h_c);
  thrust::device_vector<TC> d_c_out(h_c_out.size(), TC(float(-1)));

  constexpr uint32_t copy_max_vec_bytes = CopyMaxVecBits / 8;

  const size_t shared_memory_size = round_up(sizeof(TA) * h_a.size(), copy_max_vec_bytes) +
                                    round_up(sizeof(TB) * h_b.size(), copy_max_vec_bytes) +
                                    sizeof(TC) * h_c.size();


  auto kernel = cooperative_gemm_kernel<
    ThreadBlockSize, CopyMaxVecBits,
    GMemALayout, GMemBLayout, GMemCLayout,
    SMemALayout, SMemBLayout, SMemCLayout,
    TA, TB, TC, decltype(alpha), decltype(beta),
    TiledMma,
    ALoadTransform, BLoadTransform, CLoadTransform, CStoreTransform,
    ASMemCopyOp, BSMemCopyOp, CSMemCopyLdOp, CSMemCopyStOp
  >;

  ASSERT_EQ(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_memory_size)), 0);

  kernel<<<1, ThreadBlockSize, shared_memory_size>>>(
    gmem_a_layout,
    gmem_b_layout,
    gmem_c_layout,
    smem_a_layout,
    smem_b_layout,
    smem_c_layout,
    thrust::raw_pointer_cast(d_a.data()),
    thrust::raw_pointer_cast(d_b.data()),
    thrust::raw_pointer_cast(d_c.data()),
    thrust::raw_pointer_cast(d_c_out.data()),
    alpha,
    beta,
    tiled_mma,
    a_load_transform,
    b_load_transform,
    c_load_transform,
    c_store_transform,
    a_smem_copy_op,
    b_smem_copy_op,
    c_smem_copy_ld_op,
    c_smem_copy_st_op
  );

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    cudaError_t error = cudaGetLastError();
    FAIL() << "Error at kernel sync: " << cudaGetErrorString(error) << "\n";
  }

  // Reference gemm
  auto h_c_ref = host_reference_gemm(alpha,
                                     make_tensor(h_a.data(), gmem_a_layout),
                                     make_tensor(h_b.data(), gmem_b_layout),
                                     beta,
                                     make_tensor(h_c.data(), gmem_c_layout),
                                     a_load_transform,
                                     b_load_transform,
                                     c_load_transform,
                                     c_store_transform);

  // Copy result data
  h_c_out = d_c_out;

  // Verify correctness
  verify_gemm_correctness(make_tensor(h_c_out.data(), gmem_c_layout),
                          make_tensor(h_c_ref.data(), gmem_c_layout));
}

template<uint32_t ThreadBlockSize,
         uint32_t CopyMaxVecBits,
         class TA,
         class TB,
         class TC,
         class GMemALayout, // logical shape (M, K)
         class GMemBLayout, // logical shape (N, K)
         class GMemCLayout, // logical shape (M, N)
         class SMemALayout, // logical shape (M, K)
         class SMemBLayout, // logical shape (N, K)
         class TiledMma,
         class ALoadTransform = cute::identity,
         class BLoadTransform = cute::identity,
         class CLoadTransform = cute::identity,
         class CStoreTransform = cute::identity,
         class ASMemCopyOp = AutoVectorizingCopyWithAssumedAlignment<CopyMaxVecBits>,
         class BSMemCopyOp = AutoVectorizingCopyWithAssumedAlignment<CopyMaxVecBits>>
void test_cooperative_gemm_rmem_c(GMemALayout     gmem_a_layout,
                                  GMemBLayout     gmem_b_layout,
                                  GMemCLayout     gmem_c_layout,
                                  SMemALayout     smem_a_layout,
                                  SMemBLayout     smem_b_layout,
                                  TiledMma        tiled_mma,
                                  ALoadTransform  a_load_transform  = {},
                                  BLoadTransform  b_load_transform  = {},
                                  CLoadTransform  c_load_transform  = {},
                                  CStoreTransform c_store_transform = {},
                                  ASMemCopyOp     a_smem_copy_op    = {},
                                  BSMemCopyOp     b_smem_copy_op    = {})
{
  static_assert(size<0>(gmem_a_layout) == size<0>(gmem_c_layout));  // AM == CM
  static_assert(size<0>(gmem_b_layout) == size<1>(gmem_c_layout));  // BN == CN
  static_assert(size<1>(gmem_a_layout) == size<1>(gmem_b_layout));  // AK == BK

  static_assert(size<1>(smem_a_layout) == size<1>(smem_b_layout));  // AK == BK

  static_assert(cute::size(gmem_a_layout) == cute::size(smem_a_layout));
  static_assert(cute::size(gmem_b_layout) == cute::size(smem_b_layout));

#if 0
  print("   "); print("gmem:    "); print(gmem_layout); print("\n");
  print("   "); print("smem:    "); print(smem_layout); print("\n");
  print("   "); print("threads: "); print(ThreadBlockSize); print("\n");
#endif

  const auto alpha = static_cast<TC>(1.0);
  const auto beta  = static_cast<TC>(1.0);

  // Generate inputs
  auto [h_a, h_b, h_c, h_c_out] =
    host_generate_gemm_inputs<TA, TB, TC>(gmem_a_layout, gmem_b_layout, gmem_c_layout);

  thrust::device_vector<TA> d_a(h_a);
  thrust::device_vector<TB> d_b(h_b);
  thrust::device_vector<TC> d_c(h_c);
  thrust::device_vector<TC> d_c_out(h_c_out.size(), static_cast<TC>(-1));

  constexpr uint32_t copy_max_vec_bytes = CopyMaxVecBits / 8;

  const size_t shared_memory_size = round_up(sizeof(TA) * h_a.size(), copy_max_vec_bytes) +
                                    round_up(sizeof(TB) * h_b.size(), copy_max_vec_bytes);


  auto kernel = cooperative_gemm_kernel_rmem_c<
    ThreadBlockSize, CopyMaxVecBits,
    GMemALayout, GMemBLayout, GMemCLayout,
    SMemALayout, SMemBLayout,
    TA, TB, TC,
    TiledMma,
    ALoadTransform, BLoadTransform, CLoadTransform, CStoreTransform,
    ASMemCopyOp, BSMemCopyOp
  >;

  ASSERT_EQ(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_memory_size)), 0);

  kernel<<<1, ThreadBlockSize, shared_memory_size>>>(
    gmem_a_layout,
    gmem_b_layout,
    gmem_c_layout,
    smem_a_layout,
    smem_b_layout,
    thrust::raw_pointer_cast(d_a.data()),
    thrust::raw_pointer_cast(d_b.data()),
    thrust::raw_pointer_cast(d_c.data()),
    thrust::raw_pointer_cast(d_c_out.data()),
    tiled_mma,
    a_load_transform, b_load_transform, c_load_transform, c_store_transform,
    a_smem_copy_op, b_smem_copy_op
  );

  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    cudaError_t error = cudaGetLastError();
    FAIL() << "Error at kernel sync: " << cudaGetErrorString(error) << "\n";
  }

  // Copy result data
  h_c_out = d_c_out;

  // Reference gemm
  auto h_c_ref = host_reference_gemm(alpha,
                                     make_tensor(h_a.data(), gmem_a_layout),
                                     make_tensor(h_b.data(), gmem_b_layout),
                                     beta,
                                     make_tensor(h_c.data(), gmem_c_layout),
                                     a_load_transform,
                                     b_load_transform,
                                     c_load_transform,
                                     c_store_transform);

  // Verify correctness
  verify_gemm_correctness(make_tensor(h_c_out.data(), gmem_c_layout),
                          make_tensor(h_c_ref.data(), gmem_c_layout));
}

template<uint32_t ThreadBlockSize,
         uint32_t CopyMaxVecBits,
         class TA,
         class TB,
         class TC,
         class ShapeMNK,
         class TiledMma,
         class ... Ops>
void test_cooperative_gemm_col_major_layout(ShapeMNK shape_mnk,
                                            TiledMma tiled_mma,
                                            Ops ... ops)
{
  auto a_layout = make_layout(select<0, 2>(shape_mnk));
  auto b_layout = make_layout(select<1, 2>(shape_mnk), GenRowMajor{});
  auto c_layout = make_layout(select<0, 1>(shape_mnk));

  test_cooperative_gemm<ThreadBlockSize,
                        CopyMaxVecBits,
                        TA, TB, TC>
    (a_layout,
     b_layout,
     c_layout,
     a_layout,
     b_layout,
     c_layout,
     tiled_mma,
     ops...);
}


template<uint32_t ThreadBlockSize,
         uint32_t CopyMaxVecBits,
         class TA,
         class TB,
         class TC,
         class SMemAtomLayoutA,
         class SMemAtomLayoutB,
         class SMemAtomLayoutC,
         class ShapeMNK,
         class TiledMma,
         class ... Ops>
std::enable_if_t<std::conjunction_v<cute::is_layout<SMemAtomLayoutA>,
                                    cute::is_layout<SMemAtomLayoutB>,
                                    cute::is_layout<SMemAtomLayoutC>>>
test_cooperative_gemm_col_major_layout(SMemAtomLayoutA smem_atom_layout_a,
                                       SMemAtomLayoutB smem_atom_layout_b,
                                       SMemAtomLayoutC smem_atom_layout_c,
                                       ShapeMNK        shape_mnk,
                                       TiledMma        tiled_mma,
                                       Ops&&    ...    ops)
{
  auto gmem_a_layout = make_layout(select<0, 2>(shape_mnk));
  auto gmem_b_layout = make_layout(select<1, 2>(shape_mnk), GenRowMajor{});
  auto gmem_c_layout = make_layout(select<0, 1>(shape_mnk));

  auto smem_a_layout = tile_to_shape(
      smem_atom_layout_a,
      make_shape(shape<0>(gmem_a_layout), shape<1>(gmem_a_layout)));

  auto smem_b_layout = tile_to_shape(
      smem_atom_layout_b,
      make_shape(shape<0>(gmem_b_layout), shape<1>(gmem_b_layout)));

  auto smem_c_layout = tile_to_shape(
      smem_atom_layout_c,
      make_shape(shape<0>(gmem_c_layout), shape<1>(gmem_c_layout)));

  test_cooperative_gemm<ThreadBlockSize,
                        CopyMaxVecBits,
                        TA, TB, TC>
    (gmem_a_layout,
     gmem_b_layout,
     gmem_c_layout,
     smem_a_layout,
     smem_b_layout,
     smem_c_layout,
     tiled_mma,
     ops...);
}


template<uint32_t ThreadBlockSize,
         uint32_t CopyMaxVecBits,
         class TA,
         class TB,
         class TC,
         class ShapeMNK,
         class TiledMma,
         class ... Ops>
void test_cooperative_gemm_col_major_layout_rmem_c(ShapeMNK    shape_mnk,
                                                   TiledMma    tiled_mma,
                                                   Ops ... ops)
{
  auto a_layout = make_layout(select<0, 2>(shape_mnk));
  auto b_layout = make_layout(select<1, 2>(shape_mnk), GenRowMajor{});
  auto c_layout = make_layout(select<0, 1>(shape_mnk));


  test_cooperative_gemm_rmem_c<ThreadBlockSize,
                               CopyMaxVecBits,
                               TA, TB,TC>
    (a_layout,
     b_layout,
     c_layout,
     a_layout,
     b_layout,
     tiled_mma,
     ops...);
}

template<uint32_t ThreadBlockSize,
         uint32_t CopyMaxVecBits,
         class TA,
         class TB,
         class TC,
         class SMemAtomLayoutA,
         class SMemAtomLayoutB,
         class ShapeMNK,
         class TiledMma,
         class ... Ops>
std::enable_if_t<std::conjunction_v<cute::is_layout<SMemAtomLayoutA>,
                                    cute::is_layout<SMemAtomLayoutB>>>
test_cooperative_gemm_col_major_layout_rmem_c(SMemAtomLayoutA smem_atom_layout_a,
                                              SMemAtomLayoutB smem_atom_layout_b,
                                              ShapeMNK        shape_mnk,
                                              TiledMma        tiled_mma,
                                              Ops      ...    ops)
{
  auto gmem_a_layout = make_layout(select<0, 2>(shape_mnk));
  auto gmem_b_layout = make_layout(select<1, 2>(shape_mnk), GenRowMajor{});
  auto gmem_c_layout = make_layout(select<0, 1>(shape_mnk));

  auto smem_a_layout = tile_to_shape(
      smem_atom_layout_a,
      make_shape(shape<0>(gmem_a_layout), shape<1>(gmem_a_layout)));

  auto smem_b_layout = tile_to_shape(
      smem_atom_layout_b,
      make_shape(shape<0>(gmem_b_layout), shape<1>(gmem_b_layout)));

  test_cooperative_gemm_rmem_c<ThreadBlockSize, CopyMaxVecBits,
                               TA, TB, TC>
    (gmem_a_layout,
     gmem_b_layout,
     gmem_c_layout,
     smem_a_layout,
     smem_b_layout,
     tiled_mma,
     ops...);
}

template<uint32_t ThreadBlockSize,
         typename T,
         class ... Args>
void test_cooperative_gemm_col_major_layout_rmem_c(Args&& ... args)
{
  test_cooperative_gemm_col_major_layout_rmem_c<ThreadBlockSize,
                                                cute::sizeof_bits_v<T>,
                                                T, T, T>
    (static_cast<Args&&>(args)...);
}

template<uint32_t ThreadBlockSize,
         class T,
         class ... Args>
void test_cooperative_gemm_col_major_layout(Args&& ... args)
{
  test_cooperative_gemm_col_major_layout<ThreadBlockSize,
                                         cute::sizeof_bits_v<T>,
                                         T, T, T>
    (static_cast<Args&&>(args)...);
}
