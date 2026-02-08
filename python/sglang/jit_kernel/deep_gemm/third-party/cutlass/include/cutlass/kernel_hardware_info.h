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

#include "cutlass/device_kernel.h"
#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif
#include <cute/int_tuple.hpp>

namespace cutlass {

struct KernelHardwareInfo {
  //
  // Data members
  //

  // Hardware properties
  int device_id = 0;
  int sm_count  = 0;

  // Kernel properties
  int max_active_clusters = 0;              // Maximum number of clusters that could co-exist on the target device.
  dim3 cluster_shape = {0,0,0};             
  dim3 cluster_shape_fallback = {0,0,0};    

  //
  // Methods
  //

#if !defined(__CUDACC_RTC__)
  static inline int
  query_device_multiprocessor_count(int device_id = 0) {
    cudaError_t result = cudaGetDevice(&device_id);
    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST(
        "  cudaGetDevice() returned error "
        << cudaGetErrorString(result));
      return 0;
    }
    int multiprocessor_count;
    result = cudaDeviceGetAttribute(&multiprocessor_count,
      cudaDevAttrMultiProcessorCount, device_id);
    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST(
        "  cudaDeviceGetAttribute() returned error "
        << cudaGetErrorString(result));
      return 0;
    }
    return multiprocessor_count;
  }

  // Query maximum number of active clusters that could co-exist on the target device
  // based on kernel properties such as cluster dims and threadblock dims
  static inline int
  query_device_max_active_clusters(
      dim3 cluster_dims,
      uint32_t threads_per_block,
      void const* kernel_ptr) {
    int max_active_clusters = 0;
#if defined(CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED)
    ClusterLauncher::LaunchConfig cluster_launch_config = ClusterLauncher::make_cluster_launch_config(
                                                            cluster_dims /* minimum grid dim */, cluster_dims, {threads_per_block, 1, 1});
    // Given the kernel function and launch configuration, return the maximum number of clusters that could co-exist on the target device.
    cudaError_t result = cudaOccupancyMaxActiveClusters(&max_active_clusters, kernel_ptr, &cluster_launch_config.launch_config);
    if (result != cudaSuccess) {
      CUTLASS_TRACE_HOST(
        "  cudaGetDevice() returned error "
        << cudaGetErrorString(result));
      return 0;
    }
    CUTLASS_TRACE_HOST("cudaOccupancyMaxActiveClusters: maximum number of clusters that could co-exist on the target device = "
        << max_active_clusters << "\n");
    return max_active_clusters;
#else
    CUTLASS_TRACE_HOST("ClusterLauncher: CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED not defined! Aborting cluster occupancy query.");
    return max_active_clusters;
#endif
  }

  // Simpler version of the above query function that fetches relevant information from the Kernel 
  template <typename Kernel>
  static inline int
  query_device_max_active_clusters() {
    dim3 cluster_dims(cute::size<0>(typename Kernel::ClusterShape{}),
                      cute::size<1>(typename Kernel::ClusterShape{}),
                      cute::size<2>(typename Kernel::ClusterShape{}));
    uint32_t threads_per_block = Kernel::MaxThreadsPerBlock;
    void const* kernel_ptr = (void*)(device_kernel<Kernel>);
    return query_device_max_active_clusters(cluster_dims, threads_per_block, kernel_ptr);
  }

  template <typename Kernel>
  static inline KernelHardwareInfo
  make_kernel_hardware_info(int const device_id = 0, int sm_count = 0, int max_active_clusters = 0) {
    if (sm_count == 0) {
      sm_count = query_device_multiprocessor_count(device_id);
    }
    if (max_active_clusters == 0) {
      max_active_clusters = query_device_max_active_clusters<Kernel>();
    }
    return {device_id, sm_count, max_active_clusters};
  }
#endif
};

} // namespace cutlass
