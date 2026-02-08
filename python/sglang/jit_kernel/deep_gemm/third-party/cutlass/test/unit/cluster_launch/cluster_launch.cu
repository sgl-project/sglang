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
    \brief Unit test for the launch_on_cluster function
*/

#include "../common/cutlass_unit_test.h"
#include "cutlass/cluster_launch.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include <cassert>
#include <memory>
#include <type_traits>

#if defined(CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED)

namespace { // (anonymous)

// Using a struct instead of a lambda makes it possible
// to name the deleter type without std::function
// (which type-erases).
struct scalar_deleter {
  void operator() (float* p) {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
};

using scalar_device_pointer = std::unique_ptr<float, scalar_deleter>;

// Each test needs to initialize this anew,
// from a scalar instance that is in scope during the test.
__device__ float* scalar_ptr_gpu;

// A single scalar value on device.
// The constructor allocates space on device for one value,
// copies the value to device, and sets the global pointer
// `scalar_ptr_gpu` (see above) to point to it.
// sync_to_host() copies that value back to host.
//
// This class exists only for the tests in this file.
// In order to know whether a kernel that launch_on_cluster
// claimed to launch actually got launched, each kernel
// performs a side effect: it modifies the scalar value
// through the scalar_ptr_gpu value.
// It performs a side effect through a global,
// rather than through an argument,
// so that we can test kernel launch
// with kernels that take zero parameters.
class scalar {
private:
  static constexpr std::size_t num_bytes = sizeof(float);

public:
  scalar(float value) : value_host_(value)
  {
    float* ptr_gpu_raw = nullptr;
    auto err = cudaMalloc(&ptr_gpu_raw, num_bytes);
    assert(err == cudaSuccess);

    scalar_device_pointer ptr_gpu{ptr_gpu_raw, scalar_deleter{}};
    err = cudaMemcpy(ptr_gpu.get(), &value_host_,
                     num_bytes, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    ptr_gpu_ = std::move(ptr_gpu);
    upload_device_pointer();
  }

  float sync_to_host()
  {
    auto err = cudaMemcpy(&value_host_, ptr_gpu_.get(),
                          num_bytes, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    return value_host_;
  }

private:
  void upload_device_pointer()
  {
    float* ptr_raw = ptr_gpu_.get();
    auto err = cudaMemcpyToSymbol(scalar_ptr_gpu, &ptr_raw, sizeof(float*));
    assert(err == cudaSuccess);
  }

  float value_host_ = 0.0;
  scalar_device_pointer ptr_gpu_;
};

template<int cluster_x, int cluster_y, int cluster_z>
CUTE_DEVICE void check_cluster_shape() {
  [[maybe_unused]] const dim3 cluster_shape = cute::cluster_shape();
  assert(cluster_shape.x == cluster_x);
  assert(cluster_shape.y == cluster_y);
  assert(cluster_shape.z == cluster_z);
}

template<int cluster_x, int cluster_y, int cluster_z>
__global__ void kernel_0()
{
  check_cluster_shape<cluster_x, cluster_y, cluster_z>();

  // Write to global memory, so that we know
  // whether the kernel actually ran.
  const dim3 block_id = cute::block_id_in_cluster();
  if (threadIdx.x == 0 && block_id.x == 0 && block_id.y == 0 && block_id.z == 0) {
    *scalar_ptr_gpu = 0.1f;
  }
}

template<int cluster_x, int cluster_y, int cluster_z,
         int expected_p0>
__global__ void kernel_1(int p0)
{
  check_cluster_shape<cluster_x, cluster_y, cluster_z>();
  assert(p0 == expected_p0);

  // Write to global memory, so that we know
  // whether the kernel actually ran.
  const dim3 block_id = cute::block_id_in_cluster();
  if (threadIdx.x == 0 && block_id.x == 0 && block_id.y == 0 && block_id.z == 0) {
    *scalar_ptr_gpu = 1.2f;
  }
}

template<int cluster_x, int cluster_y, int cluster_z,
         int expected_p0,
         int expected_p2>
__global__ void kernel_2(int p0, void* p1, int p2)
{
  check_cluster_shape<cluster_x, cluster_y, cluster_z>();
  assert(p0 == expected_p0);
  assert(p1 == nullptr);
  assert(p2 == expected_p2);

  // Write to global memory, so that we know
  // whether the kernel actually ran.
  const dim3 block_id = cute::block_id_in_cluster();
  if (threadIdx.x == 0 && block_id.x == 0 && block_id.y == 0 && block_id.z == 0) {
    *scalar_ptr_gpu = 2.3f;
  }
}

struct OverloadedOperatorAmpersand {
  struct tag_t {};

  // Test that kernel launch uses the actual address,
  // instead of any overloaded operator& that might exist.
  CUTE_HOST_DEVICE tag_t operator& () const {
    return {};
  }

  int x = 0;
  int y = 0;
  int z = 0;
  int w = 0;
};

static_assert(sizeof(OverloadedOperatorAmpersand) == 4 * sizeof(int));

template<int cluster_x, int cluster_y, int cluster_z,
         int expected_p0,
         int expected_p1_x,
         int expected_p1_y,
         int expected_p1_z,
         int expected_p1_w,
         std::uint64_t expected_p2>
__global__ void kernel_3(int p0, OverloadedOperatorAmpersand p1, std::uint64_t p2)
{
  check_cluster_shape<cluster_x, cluster_y, cluster_z>();
  assert(p0 == expected_p0);
  assert(p1.x == expected_p1_x);
  assert(p1.y == expected_p1_y);
  assert(p1.z == expected_p1_z);
  assert(p1.w == expected_p1_w);
  assert(p2 == expected_p2);

  // Write to global memory, so that we know
  // whether the kernel actually ran.
  const dim3 block_id = cute::block_id_in_cluster();
  if (threadIdx.x == 0 && block_id.x == 0 && block_id.y == 0 && block_id.z == 0) {
    *scalar_ptr_gpu = 3.4f;
  }
}

} // namespace (anonymous)

TEST(SM90_ClusterLaunch, Kernel_0)
{
  scalar global_value(-1.0f);

  const dim3 grid_dims{2, 1, 1};
  const dim3 block_dims{1, 1, 1};
  const dim3 cluster_dims{grid_dims.x * block_dims.x, 1, 1};
  const int smem_size_in_bytes = 0;
  cutlass::ClusterLaunchParams params{
    grid_dims, block_dims, cluster_dims, smem_size_in_bytes};

  void const* kernel_ptr = reinterpret_cast<void const*>(&kernel_0<2, 1, 1>);
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params,
    kernel_ptr);
  ASSERT_EQ(status, cutlass::Status::kSuccess);

  cudaError_t result = cudaDeviceSynchronize();
  if (result == cudaSuccess) {
    CUTLASS_TRACE_HOST("Kernel launch succeeded\n");
  }
  else {
    CUTLASS_TRACE_HOST("Kernel launch FAILED\n");
    cudaError_t error = cudaGetLastError();
    EXPECT_EQ(result, cudaSuccess) << "Error at kernel sync: "
      << cudaGetErrorString(error) << "\n";
  }

  ASSERT_EQ(global_value.sync_to_host(), 0.1f);
}

TEST(SM90_ClusterLaunch, Kernel_1)
{
  scalar global_value(-1.0f);

  const dim3 grid_dims{2, 1, 1};
  const dim3 block_dims{1, 1, 1};
  const dim3 cluster_dims{grid_dims.x * block_dims.x, 1, 1};
  const int smem_size_in_bytes = 0;
  cutlass::ClusterLaunchParams params{
    grid_dims, block_dims, cluster_dims, smem_size_in_bytes};

  constexpr int expected_p0 = 42;
  void const* kernel_ptr = reinterpret_cast<void const*>(&kernel_1<2, 1, 1, expected_p0>);
  const int p0 = expected_p0;
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params,
    kernel_ptr, p0);
  ASSERT_EQ(status, cutlass::Status::kSuccess);

  cudaError_t result = cudaDeviceSynchronize();
  if (result == cudaSuccess) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("Kernel launch succeeded\n");
#endif
  }
  else {
    CUTLASS_TRACE_HOST("Kernel launch FAILED\n");
    cudaError_t error = cudaGetLastError();
    EXPECT_EQ(result, cudaSuccess) << "Error at kernel sync: "
      << cudaGetErrorString(error) << "\n";
  }

  ASSERT_EQ(global_value.sync_to_host(), 1.2f);
}

TEST(SM90_ClusterLaunch, Kernel_2)
{
  scalar global_value(-1.0f);

  const dim3 grid_dims{2, 1, 1};
  const dim3 block_dims{1, 1, 1};
  const dim3 cluster_dims{grid_dims.x * block_dims.x, 1, 1};
  const int smem_size_in_bytes = 0;
  cutlass::ClusterLaunchParams params{
    grid_dims, block_dims, cluster_dims, smem_size_in_bytes};

  constexpr int expected_p0 = 42;
  constexpr int expected_p2 = 43;

  int p0 = expected_p0;
  int* p1 = nullptr;
  int p2 = expected_p2;

  void const* kernel_ptr = reinterpret_cast<void const*>(
    &kernel_2<2, 1, 1, expected_p0, expected_p2>);
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params,
    kernel_ptr, p0, p1, p2);
  ASSERT_EQ(status, cutlass::Status::kSuccess);

  cudaError_t result = cudaDeviceSynchronize();
  if (result == cudaSuccess) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("Kernel launch succeeded\n");
#endif
  }
  else {
    CUTLASS_TRACE_HOST("Kernel launch FAILED\n");
    cudaError_t error = cudaGetLastError();
    EXPECT_EQ(result, cudaSuccess) << "Error at kernel sync: "
      << cudaGetErrorString(error) << "\n";
  }

  ASSERT_EQ(global_value.sync_to_host(), 2.3f);
}

TEST(SM90_ClusterLaunch, Kernel_3)
{
  scalar global_value(-1.0f);

  const dim3 grid_dims{2, 1, 1};
  const dim3 block_dims{1, 1, 1};
  const dim3 cluster_dims{grid_dims.x * block_dims.x, 1, 1};
  const int smem_size_in_bytes = 0;
  cutlass::ClusterLaunchParams params{
    grid_dims, block_dims, cluster_dims, smem_size_in_bytes};

  constexpr int expected_p0 = 42;
  constexpr int expected_p1_x = 1;
  constexpr int expected_p1_y = 2;
  constexpr int expected_p1_z = 3;
  constexpr int expected_p1_w = 4;
  constexpr std::uint64_t expected_p2 = 1'000'000'000'000uLL;

  int p0 = expected_p0;
  OverloadedOperatorAmpersand p1{expected_p1_x,
    expected_p1_y, expected_p1_z, expected_p1_w};
  // Verify that operator& is overloaded for this type.
  static_assert(! std::is_same_v<decltype(&p1),
                    OverloadedOperatorAmpersand*>);
  std::uint64_t p2 = expected_p2;

  void const* kernel_ptr = reinterpret_cast<void const*>(
    &kernel_3<2, 1, 1, expected_p0, expected_p1_x,
      expected_p1_y, expected_p1_z, expected_p1_w,
      expected_p2>);
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params,
    kernel_ptr, p0, p1, p2);
  ASSERT_EQ(status, cutlass::Status::kSuccess);

  cudaError_t result = cudaDeviceSynchronize();
  if (result == cudaSuccess) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("Kernel launch succeeded\n");
#endif
  }
  else {
    CUTLASS_TRACE_HOST("Kernel launch FAILED\n");
    cudaError_t error = cudaGetLastError();
    EXPECT_EQ(result, cudaSuccess) << "Error at kernel sync: "
      << cudaGetErrorString(error) << "\n";
  }

  ASSERT_EQ(global_value.sync_to_host(), 3.4f);
}

#endif // CUTLASS_SM90_CLUSTER_LAUNCH_ENABLED
