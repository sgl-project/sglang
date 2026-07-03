#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// Single-launch page free for the device-resident free list
// (DeviceFreeListPagedAllocator). One block per free_index slot: thread 0
// elects at most one winner block per touched page via a monotone epoch
// atomicMax (page-granular dedup with no assumptions about input structure)
// and appends the page to the self ring. Winner blocks then scan their page's
// full_to_swa mapping slots in parallel (one thread per slot), append live
// swa pages to the swa ring under a second epoch claim, and zero the mapping.
// Non-winner blocks exit after one atomic. Ring append order is
// nondeterministic (atomic), which only affects which physical page a later
// alloc picks, never any KV content.
template <int64_t kPageSize>
__global__ void free_dual_pool_kernel(
    const int64_t* __restrict__ free_index,
    size_t n_slots,
    int32_t* __restrict__ self_epoch,
    int32_t self_cur_epoch,
    int64_t* __restrict__ self_ring,
    int64_t self_cap,
    unsigned long long* __restrict__ self_tail,
    int64_t mark_self,
    int64_t* __restrict__ mapping,
    int32_t* __restrict__ swa_epoch,
    int32_t swa_cur_epoch,
    int64_t* __restrict__ swa_ring,
    int64_t swa_cap,
    unsigned long long* __restrict__ swa_tail,
    int64_t scan_swa) {
  const size_t slot_i = blockIdx.x;
  if (slot_i >= n_slots) {
    return;
  }
  const int64_t fp = free_index[slot_i] / kPageSize;
  __shared__ bool winner;
  if (threadIdx.x == 0) {
    winner = atomicMax(&self_epoch[fp], self_cur_epoch) < self_cur_epoch;
    if (winner && mark_self) {
      const unsigned long long pos = atomicAdd(self_tail, 1ull);
      self_ring[pos % self_cap] = fp;
    }
  }
  __syncthreads();
  if (!winner || !scan_swa) {
    return;
  }
  // Page-granular free: the winner clears the whole page's mapping, including
  // slots not present in free_index.
  for (int64_t j = threadIdx.x; j < kPageSize; j += blockDim.x) {
    const int64_t idx = fp * kPageSize + j;
    const int64_t v = mapping[idx];
    mapping[idx] = 0;
    if (v > 0) {
      const int64_t sp = v / kPageSize;
      if (atomicMax(&swa_epoch[sp], swa_cur_epoch) < swa_cur_epoch) {
        const unsigned long long pos = atomicAdd(swa_tail, 1ull);
        swa_ring[pos % swa_cap] = sp;
      }
    }
  }
}

template <int64_t kPageSize>
void free_dual_pool(
    tvm::ffi::TensorView free_index,
    tvm::ffi::TensorView self_epoch,
    int64_t self_cur_epoch,
    tvm::ffi::TensorView self_ring,
    int64_t self_cap,
    tvm::ffi::TensorView self_tail,
    int64_t mark_self,
    tvm::ffi::TensorView mapping,
    tvm::ffi::TensorView swa_epoch,
    int64_t swa_cur_epoch,
    tvm::ffi::TensorView swa_ring,
    int64_t swa_cap,
    tvm::ffi::TensorView swa_tail,
    int64_t scan_swa) {
  using namespace host;

  SymbolicSize N = {"n_slots"};
  SymbolicDevice device_;
  TensorMatcher({N}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(free_index);
  const size_t n_slots = N.unwrap();
  const DLDevice device = device_.unwrap();
  RuntimeCheck(n_slots > 0, "free_dual_pool expects a non-empty free_index, got n_slots = ", n_slots);

  constexpr size_t kBlock = kPageSize < 256 ? static_cast<size_t>(kPageSize) : 256;
  static_assert(kPageSize >= 1, "page_size must be positive");
  LaunchKernel(n_slots, kBlock < 32 ? 32 : kBlock, device)(
      free_dual_pool_kernel<kPageSize>,
      static_cast<const int64_t*>(free_index.data_ptr()),
      n_slots,
      static_cast<int32_t*>(self_epoch.data_ptr()),
      static_cast<int32_t>(self_cur_epoch),
      static_cast<int64_t*>(self_ring.data_ptr()),
      self_cap,
      reinterpret_cast<unsigned long long*>(self_tail.data_ptr()),
      mark_self,
      static_cast<int64_t*>(mapping.data_ptr()),
      static_cast<int32_t*>(swa_epoch.data_ptr()),
      static_cast<int32_t>(swa_cur_epoch),
      static_cast<int64_t*>(swa_ring.data_ptr()),
      swa_cap,
      reinterpret_cast<unsigned long long*>(swa_tail.data_ptr()),
      scan_swa);
}

}  // namespace
