#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {
namespace {

template <uint32_t kWorldSize>
__global__ void
shared_kv_publish_kernel(uint32_t* const* __restrict__ peer_rows, uint32_t* __restrict__ epoch, uint32_t rank) {
  static_assert(kWorldSize >= 2 && kWorldSize <= 8);
  __shared__ uint32_t expected;
  if (threadIdx.x == 0) {
    expected = *epoch + 1u;
    *epoch = expected;
  }
  __syncthreads();

  if (threadIdx.x < kWorldSize) {
    const uint32_t peer = threadIdx.x;
    uint32_t* remote = peer_rows[peer] + rank;
    asm volatile("st.release.sys.global.u32 [%0], %1;" : : "l"(remote), "r"(expected) : "memory");

    uint32_t observed;
    uint32_t* local = peer_rows[rank] + peer;
    do {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(observed) : "l"(local) : "memory");
    } while (static_cast<int32_t>(observed - expected) < 0);
  }
  __syncthreads();
}

template <uint32_t kWorldSize>
__global__ void shared_kv_publish_status_kernel(
    uint32_t* const* __restrict__ peer_rows,
    uint32_t* __restrict__ epoch,
    uint32_t* __restrict__ result,
    uint32_t rank,
    uint32_t local_success) {
  static_assert(kWorldSize >= 2 && kWorldSize <= 8);
  __shared__ uint32_t expected;
  __shared__ uint32_t all_success;
  if (threadIdx.x == 0) {
    expected = *epoch + 1u;
    *epoch = expected;
    all_success = 1u;
  }
  __syncthreads();

  if (threadIdx.x < kWorldSize) {
    const uint32_t peer = threadIdx.x;
    uint32_t* remote_row = peer_rows[peer];
    asm volatile("st.release.sys.global.u32 [%0], %1;"
                 :
                 : "l"(remote_row + kWorldSize + rank), "r"(local_success)
                 : "memory");
    asm volatile("st.release.sys.global.u32 [%0], %1;" : : "l"(remote_row + rank), "r"(expected) : "memory");

    uint32_t observed;
    uint32_t* local_row = peer_rows[rank];
    do {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(observed) : "l"(local_row + peer) : "memory");
    } while (static_cast<int32_t>(observed - expected) < 0);

    uint32_t peer_success;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(peer_success)
                 : "l"(local_row + kWorldSize + peer)
                 : "memory");
    if (peer_success == 0u) {
      atomicExch(&all_success, 0u);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *result = all_success;
  }
}

template <uint32_t kWorldSize>
void shared_kv_publish(
    tvm::ffi::TensorView flags, tvm::ffi::TensorView peer_ptrs, tvm::ffi::TensorView epoch, int64_t rank) {
  using namespace host;
  SymbolicDevice device;
  device.set_options<kDLCUDA>();
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(flags);
  TensorMatcher({kWorldSize}).with_dtype<int64_t>().with_device(device).verify(peer_ptrs);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(epoch);
  RuntimeCheck(rank >= 0 && rank < kWorldSize, "rank out of range: ", rank);

  auto** rows = reinterpret_cast<uint32_t**>(peer_ptrs.data_ptr());
  auto* epoch_ptr = static_cast<uint32_t*>(epoch.data_ptr());
  const DLDevice launch_device = device.unwrap();
  LaunchKernel(1, 32, launch_device)(
      shared_kv_publish_kernel<kWorldSize>, rows, epoch_ptr, static_cast<uint32_t>(rank));
}

template <uint32_t kWorldSize>
void shared_kv_publish_status(
    tvm::ffi::TensorView flags,
    tvm::ffi::TensorView peer_ptrs,
    tvm::ffi::TensorView epoch,
    tvm::ffi::TensorView result,
    int64_t rank,
    bool local_success) {
  using namespace host;
  SymbolicDevice device;
  device.set_options<kDLCUDA>();
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(flags);
  TensorMatcher({kWorldSize}).with_dtype<int64_t>().with_device(device).verify(peer_ptrs);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(epoch);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(result);
  RuntimeCheck(rank >= 0 && rank < kWorldSize, "rank out of range: ", rank);

  auto** rows = reinterpret_cast<uint32_t**>(peer_ptrs.data_ptr());
  const DLDevice launch_device = device.unwrap();
  LaunchKernel(1, 32, launch_device)(
      shared_kv_publish_status_kernel<kWorldSize>,
      rows,
      static_cast<uint32_t*>(epoch.data_ptr()),
      static_cast<uint32_t*>(result.data_ptr()),
      static_cast<uint32_t>(rank),
      static_cast<uint32_t>(local_success));
}

}  // namespace
}  // namespace sglang
