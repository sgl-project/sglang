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
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/cluster_launch.hpp>

namespace cutlass::test {

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
  alignas(16) cute::uint64_t tma_load_mbar[1];
};

#if CUDA_12_0_SM90_FEATURES_SUPPORTED

template <class T, class GmemLayout, class SmemLayout,
          class CopyAtom, class CTA_Tiler, class Cluster_Size>
__global__ void
tma_test_device_cute(T const* g_in, T* g_out, GmemLayout gmem_layout, SmemLayout smem_layout,
                     CUTE_GRID_CONSTANT CopyAtom const tma, CTA_Tiler cta_tiler, Cluster_Size cluster_size)
{
  using namespace cute;
  CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));

  // Use Shared Storage structure to allocate and distribute aligned SMEM addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Construct SMEM tensor
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout);  // (CTA_TILE_M,CTA_TILE_N,...)
  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t* tma_load_mbar = shared_storage.tma_load_mbar;

  // TMA requires special handling of strides to deal with coord codomain mapping
  // Represent the full tensors -- get these from TMA
  Tensor mA = tma.get_tma_tensor(shape(gmem_layout));
  Tensor mB = make_tensor(make_gmem_ptr<T>(g_out), gmem_layout);

  Tensor gA = zipped_divide(mA, cta_tiler);               // ((CTA_TILE_M,CTA_TILE_N,...),(REST_M,REST_N,...))
  Tensor gB = zipped_divide(mB, cta_tiler);               // ((CTA_TILE_M,CTA_TILE_N,...),(REST_M,REST_N,...))

#if 1
  if (thread0()) {
    print(tma);
    print("TILE  :  "); print(cta_tiler); print("\n");
    print("  mA  :  "); print(  mA);   print("\n");
    print("  mB  :  "); print(  mB);   print("\n");
    print("  gA  :  "); print(  gA);   print("\n");
    print("  gB  :  "); print(  gB);   print("\n");
    print("  sA  :  "); print(  sA);   print("\n");
  } __syncthreads(); cute::cluster_sync();
#endif

  //
  // Prepare the TMA_LOAD
  //

  Tensor sA_x = make_tensor(sA.data(), make_layout(sA.layout(), Layout<_1>{}));  // ((CTA_TILE_M,CTA_TILE_N,...),_1)
  Tensor tBgB = gB;                                                              // ((CTA_TILE_M,CTA_TILE_N,...),(REST_M,REST_N,...))

  int cta_rank_in_cluster  = cute::block_rank_in_cluster();
  auto [tAgA, tAsA] = tma_partition(tma, cta_rank_in_cluster, make_layout(cluster_size), sA_x, gA);

#if 1
  if (thread0()) {
    print("sA_x  :  "); print(sA_x); print("\n");
    print("tBgB  :  "); print(tBgB); print("\n");
    print("tAgA  :  "); print(tAgA); print("\n");
    print("tAsA  :  "); print(tAsA); print("\n");
  } __syncthreads(); cute::cluster_sync();
#endif

  //
  // TMA Multicast Masks -- Get a mask of the active ctas in each TMA
  //


  int elected_cta_rank = 0;
  bool elect_one_cta = (elected_cta_rank == cta_rank_in_cluster);
  bool elect_one_thr = cute::elect_one_sync();

  uint16_t tma_mcast_mask = ((uint16_t(1) << cluster_size) - 1);

#if 1
  if (thread0()) {
    print("tma_mcast_mask :  "); print(tma_mcast_mask); print("\n");
  } __syncthreads(); cute::cluster_sync();
#endif

  //
  // Perform the TMA_LOAD
  //

  if (elect_one_thr) {
    // Initialize TMA barrier
    cute::initialize_barrier(tma_load_mbar[0], /* num_threads */ 1);
  }
  int tma_phase_bit = 0;
  // Ensures all CTAs in the Cluster have initialized
  __syncthreads();
  cute::cluster_sync();

  // Loop over the TMA stages, using smem as our buffer
  for (int stage = 0; stage < size<1>(tAgA); ++stage)
  {
    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    constexpr int kTmaTransactionBytes = sizeof(ArrayEngine<T, CUTE_STATIC_V(size(filter_zeros(sA)))>);

    if (elect_one_thr)
    {
      cute::set_barrier_transaction_bytes(tma_load_mbar[0], kTmaTransactionBytes);

      copy(tma.with(tma_load_mbar[0], tma_mcast_mask), tAgA(_,stage), tAsA(_,0));
    }
    __syncthreads();

    /// Wait on the shared memory barrier until the phase bit flips from tma_phase_bit value
    cute::wait_barrier(tma_load_mbar[0], tma_phase_bit);
    tma_phase_bit ^= 1;

    //
    // Write out trivially smem -> gmem
    //

    // Subbyte elements could cause race conditions, so be even more conservative
    if (elect_one_cta && elect_one_thr) {
      copy(sA, tBgB(_,stage));
    }

    __syncthreads();
    cute::cluster_sync();
  }
}

template <class T, class TmaType = T, class CopyOp,
          class GMEM_Layout, class SMEM_Layout,
          class CTA_Tiler, class Cluster_Size>
auto
test_tma_load(CopyOp       const& copy_op,
              GMEM_Layout  const& gmem_layout,
              SMEM_Layout  const& smem_layout,
              CTA_Tiler    const& cta_tiler,
              Cluster_Size const& cluster_size)
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
  Tensor gA = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_in.data())), gmem_layout);
  auto tma = make_tma_atom<TmaType>(copy_op, gA, smem_layout, cta_tiler, cluster_size);
  //print(tma);

  // Launch

  dim3 dimBlock(32);
  dim3 dimCluster(size(cluster_size));
  dim3 dimGrid = dimCluster;
  int smem_size = sizeof(SharedStorage<T, SMEM_Layout>);

  void* kernel_ptr = (void*) &tma_test_device_cute<T, GMEM_Layout, SMEM_Layout,
                                                   decltype(tma), CTA_Tiler, Cluster_Size>;

  cutlass::launch_kernel_on_cluster({dimGrid, dimBlock, dimCluster, smem_size},
                                    kernel_ptr,
                                    reinterpret_cast<T const*>(raw_pointer_cast(d_in.data())),
                                    reinterpret_cast<T      *>(raw_pointer_cast(d_out.data())),
                                    gmem_layout,
                                    smem_layout,
                                    tma, cta_tiler, cluster_size);

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

  return tma;
}

#endif

} // end namespace cutlass::test
