/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Benchmark helpers for Distributed GEMM

    A delay kernel to gate all GEMMs across devices, controlled by a flag that
    the host will set off once it launches DistGEMM across all devices.

    DistGpuTimer extends cutlass's existing cudaEvent-based timer to multiple devices.
*/

#pragma once
#include "cutlass/cutlass.h"
#include <iostream>
#include <cuda/atomic>
#include CUDA_STD_HEADER(atomic)

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cuda_host_adapter.hpp"


namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Delay kernel
/////////////////////////////////////////////////////////////////////////////////////////////////

using AtomicBoolean = cuda::atomic<bool>;

__global__ void delay_kernel(const AtomicBoolean* atomic_flag_ptr) {
  while (not atomic_flag_ptr->load()) {
    __nanosleep(40);
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Distributed GPU Timer
/// Sets up cuda events for multiple processors.
/////////////////////////////////////////////////////////////////////////////////////////////////
template <int NP>
struct DistGpuTimer {
  int _primary_device;
  cudaEvent_t _start[NP];
  cudaEvent_t _stop[NP];

  /// Constructor
  DistGpuTimer()
  {
    CUDA_CHECK(cudaGetDevice(&_primary_device));
    for (int device = 0; device < NP; ++device) {
      CUDA_CHECK(cudaSetDevice(device));
      CUDA_CHECK(cudaEventCreate(&_start[device]));
      CUDA_CHECK(cudaEventCreate(&_stop[device]));
    }
    CUDA_CHECK(cudaSetDevice(_primary_device));
  }

  /// Destructor
  ~DistGpuTimer()
  {
    for (int device = 0; device < NP; ++device) {
      CUDA_CHECK(cudaSetDevice(device));
      CUDA_CHECK(cudaEventDestroy(_start[device]));
      CUDA_CHECK(cudaEventDestroy(_stop[device]));
    }
    CUDA_CHECK(cudaSetDevice(_primary_device));
  }

  /// Start the timer for a given stream (defaults to the default stream)
  void start(int device, cudaStream_t stream) {
    assert(device >= 0 && device < NP);
    CUDA_CHECK(cudaEventRecord(_start[device], stream));
  }

  /// Stop the timer
  void stop(int device, cudaStream_t stream) {
    assert(device >= 0 && device < NP);
    CUDA_CHECK(cudaEventRecord(_stop[device], stream));
  }

  /// Return the elapsed time (in milliseconds)
  float elapsed_millis(int device) {
    assert(device >= 0 && device < NP);
    float elapsed = 0.0;
    CUDA_CHECK(cudaEventSynchronize(_stop[device]));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start[device], _stop[device]));
    return elapsed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Generic device-to-device data movement kernel based for CuTe tensors.
///
///   NOTE: this kernel assigns one element copy to every thread, and is by no means
///   an efficient way of copying tensors. It should only be used for convenience in
///   reference checks.
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TensorSource, typename TensorDestination>
void device_copy(TensorSource      tensor_source,
                 TensorDestination tensor_destination,
                 cudaStream_t stream);


template <typename TensorSource, typename TensorDestination>
__global__ void device_copy_kernel(TensorSource const tensor_source, 
                                   TensorDestination tensor_destination) {
  auto linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  using ElementSrc = typename TensorSource::value_type;
  using ElementDst = typename TensorDestination::value_type;
  NumericConverter<ElementDst, ElementSrc> converter;
  if (linear_idx < size(tensor_source)) {
    tensor_destination(linear_idx) = converter(tensor_source(linear_idx));
  }
}

template <typename TensorSource, typename TensorDestination>
void device_copy(TensorSource      tensor_source,
                 TensorDestination tensor_destination,
                 cudaStream_t stream) {
  
  assert(tensor_source.size() == tensor_destination.size());

  auto numel = tensor_source.size();
  static constexpr int NumThreads = 128;
  auto grid_size = cute::ceil_div(numel, NumThreads);

  dim3 grid(grid_size);
  dim3 block(NumThreads);
  device_copy_kernel<<<grid, block, 0, stream>>>(tensor_source, tensor_destination);
}

} //namespace cutlass
