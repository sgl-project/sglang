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



#pragma once

#include <math.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct DeviceAllocation {
  T* ptr_ = nullptr;
  size_t offset_ = 0;
  size_t size_ = 0;

  DeviceAllocation(DeviceAllocation const&) = delete;
  DeviceAllocation& operator=(DeviceAllocation const&) = delete;

  DeviceAllocation() = default;
  DeviceAllocation(size_t size) { reset(size); }
  ~DeviceAllocation() { reset(); }

  void reset(size_t size, size_t offset=0) {
    reset();
    auto ret = cudaMalloc(&ptr_, sizeof(T) * (size + offset));
    assert(ret == cudaSuccess);
    size_ = size;
    offset_ = offset;
  }

  T* get() {
    return ptr_ + offset_;
  }

  const T* get() const {
    return ptr_ + offset_;
  }

  void reset() {
    if (ptr_ != nullptr) {
      auto ret = cudaFree(ptr_);
      assert(ret == cudaSuccess);
    }
  }

  size_t size() const { return size_; }

  size_t get_storage_size() const { return (size_ + offset_) * sizeof(T); }

  void copy_from_host(const T* ptr, size_t sz) {
    auto ret = cudaMemcpy(ptr_, ptr, sz * sizeof(T), cudaMemcpyDefault);
    assert(ret == cudaSuccess);
  }

  void copy_from_device(const T* ptr, size_t sz) {
    auto ret = cudaMemcpy(ptr_, ptr, sz * sizeof(T), cudaMemcpyDefault);
    assert(ret == cudaSuccess);
  }
};

template<typename Element>
__global__ void reference_abs_diff_kernel(
    Element* data, Element* data_ref, size_t count,
    double* max_diff, double* sum_diff,
    bool print_diff ) {

    double thread_max_diff = 0;
    double thread_sum_diff = 0;

    __shared__ double block_max_diff;
    __shared__ double block_sum_diff;

    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
      if (data[i] == data_ref[i]) {
        continue;
      }

      double diff = fabs(data[i] - data_ref[i]);
      if (print_diff) if (not isfinite(diff) || diff > 0.01f) printf("difference at %lld: %f ... %f vs %f\n", static_cast<long long int>(i), diff, (double)data[i], (double)data_ref[i]);
      thread_max_diff = fmax(diff, thread_max_diff);
      thread_sum_diff += diff;
    }

    for (int i = 0; i < blockDim.x; i++) {
      if (i == threadIdx.x) {
        if (i == 0) {
          block_max_diff = thread_max_diff;
          block_sum_diff = thread_sum_diff;
        }
        else {
          block_max_diff = fmax(block_max_diff, thread_max_diff);
          block_sum_diff += thread_sum_diff;
        }
      }
      __syncthreads();
   }

   if (threadIdx.x == 0) {
     atomicAdd(sum_diff, block_sum_diff);

     for (;;) {
       unsigned long long prev = *reinterpret_cast<unsigned long long*>(max_diff);
       double prev_diff = reinterpret_cast<double const&>(prev);
       double new_max_diff = fmax(block_max_diff, prev_diff);
       unsigned long long found = atomicCAS(reinterpret_cast<unsigned long long*>(max_diff), prev, reinterpret_cast<unsigned long long const&>(new_max_diff));
       if (found == prev) break;
    }
   }
}

template<typename Element>
void reference_abs_diff(
    DeviceAllocation<Element> const& data,
    DeviceAllocation<Element> const& data_ref,
    double& max_diff, double& mean_diff) {

  static bool kPrintDiff = getenv("REF_PRINT_DIFF") && atoi(getenv("REF_PRINT_DIFF")) == 1;

  DeviceAllocation<double> result;
  result.reset(2);
  assert(data.size() == data_ref.size());

  cudaError_t err = cudaMemset(result.get(), 0, result.size() * sizeof(double));
  if (err != cudaSuccess) {
    std::cerr << "Memset failed. Last CUDA error: "
              << cudaGetErrorString(err) << std::endl;
    max_diff = mean_diff = 1e20;
    return;
  }

  dim3 block(256, 1, 1);
  dim3 grid(1024, 1, 1);
  reference_abs_diff_kernel<<<block, grid>>>(
      data.get(), data_ref.get(), data.size(),
      result.get(), result.get() + 1, kPrintDiff);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Difference kernel failed. Last CUDA error: "
              << cudaGetErrorString(err) << std::endl;
    max_diff = mean_diff = 1e20;
    return;
  }

  double result_host[2];
  err = cudaMemcpy(result_host, result.get(), result.size() * sizeof(double), cudaMemcpyDefault);
  if (err != cudaSuccess) {
    std::cerr << "Copy failed. Last CUDA error: "
              << cudaGetErrorString(err) << std::endl;
    max_diff = mean_diff = 1e20;
    return;
  }

  max_diff = result_host[0];
  mean_diff = result_host[1] / static_cast<double>(data.size());
}

template<typename Element>
__global__ void reference_rel_diff_kernel(
    Element* data, Element* data_ref, size_t count,
    double* max_diff, double* sum_diff,
    bool print_diff ) {

    double thread_max_diff = 0;
    double thread_sum_diff = 0;

    __shared__ double block_max_diff;
    __shared__ double block_sum_diff;

    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
      if (data[i] == data_ref[i]) {
        continue;
      }
      double diff = fabs(data[i] - data_ref[i]) / fabs(data_ref[i]);
      if (print_diff) if (not isfinite(diff) || diff > 0.01f) printf("difference at %lld: %f ... %f vs %f\n", static_cast<long long int>(i), diff, (double)data[i], (double)data_ref[i]);
      thread_max_diff = fmax(diff, thread_max_diff);
      thread_sum_diff += diff;
    }

    for (int i = 0; i < blockDim.x; i++) {
      if (i == threadIdx.x) {
        if (i == 0) {
          block_max_diff = thread_max_diff;
          block_sum_diff = thread_sum_diff;
        }
        else {
          block_max_diff = fmax(block_max_diff, thread_max_diff);
          block_sum_diff += thread_sum_diff;
        }
      }
      __syncthreads();
   }

   if (threadIdx.x == 0) {
     atomicAdd(sum_diff, block_sum_diff);

     for (;;) {
       unsigned long long prev = *reinterpret_cast<unsigned long long*>(max_diff);
       double prev_diff = reinterpret_cast<double const&>(prev);
       double new_max_diff = fmax(block_max_diff, prev_diff);
       unsigned long long found = atomicCAS(reinterpret_cast<unsigned long long*>(max_diff), prev, reinterpret_cast<unsigned long long const&>(new_max_diff));
       if (found == prev) break;
    }
   }
}

template<typename Element>
void reference_rel_diff(
    DeviceAllocation<Element> const& data,
    DeviceAllocation<Element> const& data_ref,
    double& max_diff, double& mean_diff) {

  static bool kPrintDiff = getenv("REF_PRINT_DIFF") && atoi(getenv("REF_PRINT_DIFF")) == 1;

  DeviceAllocation<double> result;
  result.reset(2);
  assert(data.size() == data_ref.size());

  cudaError_t err = cudaMemset(result.get(), 0, result.size() * sizeof(double));
  if (err != cudaSuccess) {
    std::cerr << "Memset failed. Last CUDA error: "
              << cudaGetErrorString(err) << std::endl;
    max_diff = mean_diff = 1e20;
    return;
  }

  dim3 block(256, 1, 1);
  dim3 grid(1024, 1, 1);
  reference_rel_diff_kernel<<<block, grid>>>(
      data.get(), data_ref.get(), data.size(),
      result.get(), result.get() + 1, kPrintDiff);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Difference kernel failed. Last CUDA error: "
              << cudaGetErrorString(err) << std::endl;
    max_diff = mean_diff = 1e20;
    return;
  }

  double result_host[2];
  err = cudaMemcpy(result_host, result.get(), result.size() * sizeof(double), cudaMemcpyDefault);
  if (err != cudaSuccess) {
    std::cerr << "Copy failed. Last CUDA error: "
              << cudaGetErrorString(err) << std::endl;
    max_diff = mean_diff = 1e20;
    return;
  }

  max_diff = result_host[0];
  mean_diff = result_host[1] / static_cast<double>(data.size());
}
