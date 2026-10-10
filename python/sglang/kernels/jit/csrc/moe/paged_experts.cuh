// Paged experts (srt/layers/moe/paged_experts): the residency decision and the expert page-in on
// the GPU, so a decode step pages experts without a host sync and can be captured in a CUDA graph.
//
// decide: for a step whose routed (token, expert) entries fit the K slots, keep every resident
// expert the step uses, give each missing expert the least recently used slot the step does not
// need, and write the page-in plan (src expert -> dst slot, count) to device buffers.
// gather: copy the planned experts from the pinned host store into their slots, for every paged
// tensor of a layer in one launch. The count is read on the device, so a replayed graph moves
// exactly the experts its decide chose.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <climits>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

// One warp. The evictions depend on each other and are committed one at a time by lane 0; each
// victim search and the per-entry bookkeeping are spread over the warp.
__global__ void paged_experts_decide_kernel(
    const int32_t* topk,  // [T] routed expert ids of the step (negative: padding)
    int T,
    int E,
    int K,
    int32_t* step,          // [1] step counter, advanced on the device so replays age the LRU
    int32_t* slot_expert,   // [K] expert in each slot (-1: empty)
    int32_t* expert_slot,   // [E] slot of each expert (-1: not resident)
    int32_t* slot_lastuse,  // [K] step each slot was last used
    int32_t* src,           // [K] out: experts to page in
    int32_t* dst,           // [K] out: their slots
    int32_t* count) {       // [1] out: number of page-ins
  const int lane = threadIdx.x;
  int now = 0;
  if (lane == 0) now = ++(*step);
  now = __shfl_sync(0xffffffff, now, 0);
  for (int i = lane; i < T; i += 32) {
    const int e = topk[i];
    if (e < 0 || e >= E) continue;
    const int s = expert_slot[e];
    if (s >= 0) slot_lastuse[s] = now;
  }
  __syncwarp();
  int n = 0;
  for (int i = 0; i < T; ++i) {
    const int e = topk[i];
    if (e < 0 || e >= E || expert_slot[e] >= 0) continue;  // the same for every lane
    // Victim: the least recently used slot holding no expert of this step, lowest slot on
    // ties. Every slot used this step carries `now`, so one is always left while T <= K.
    int oldest = INT_MAX, victim = K;
    for (int s = lane; s < K; s += 32) {
      const int lastuse = slot_lastuse[s];
      if (lastuse != now && lastuse < oldest) {
        oldest = lastuse;
        victim = s;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const int other_oldest = __shfl_down_sync(0xffffffff, oldest, offset);
      const int other_victim = __shfl_down_sync(0xffffffff, victim, offset);
      if (other_oldest < oldest || (other_oldest == oldest && other_victim < victim)) {
        oldest = other_oldest;
        victim = other_victim;
      }
    }
    victim = __shfl_sync(0xffffffff, victim, 0);
    if (victim == K) continue;
    if (lane == 0) {
      const int old = slot_expert[victim];
      if (old >= 0) expert_slot[old] = -1;
      slot_expert[victim] = e;
      expert_slot[e] = victim;
      slot_lastuse[victim] = now;
      src[n] = e;
      dst[n] = victim;
    }
    ++n;
    __syncwarp();  // the next search reads this eviction
  }
  if (lane == 0) *count = n;
}

template <typename T>
__device__ void
copy_rows(const T* host, T* gpu, long row, int n, const int32_t* src, const int32_t* dst, long tid, long stride) {
  for (long j = tid; j < n * row; j += stride) {
    const long i = j / row, off = j % row;
    gpu[dst[i] * row + off] = host[src[i] * row + off];
  }
}

__global__ void paged_experts_gather_kernel(
    const int64_t* hosts,  // [P] UVA device pointer of each paged tensor's host store
    const int64_t* gpus,   // [P] each paged tensor's K-slot GPU table
    const int64_t* words,  // [P] 4-byte words per expert of each tensor
    int P,
    const int32_t* src,
    const int32_t* dst,
    const int32_t* count) {
  const int n = *count;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int p = 0; p < P; ++p) {
    const long row = words[p];
    if (row % 4 == 0) {  // 16-byte rows: copy in float4, which the host link needs
      copy_rows(
          reinterpret_cast<const float4*>(hosts[p]),
          reinterpret_cast<float4*>(gpus[p]),
          row / 4,
          n,
          src,
          dst,
          tid,
          stride);
    } else {  // small per-expert scalars
      copy_rows(
          reinterpret_cast<const int32_t*>(hosts[p]),
          reinterpret_cast<int32_t*>(gpus[p]),
          row,
          n,
          src,
          dst,
          tid,
          stride);
    }
  }
}

void paged_experts_decide(
    tvm::ffi::TensorView topk,
    tvm::ffi::TensorView step,
    tvm::ffi::TensorView slot_expert,
    tvm::ffi::TensorView expert_slot,
    tvm::ffi::TensorView slot_lastuse,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView count) {
  using namespace sglang::host;

  SymbolicSize E = {"num_experts"}, K = {"num_slots"}, T = {"num_entries"}, One = {"one"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({E}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(expert_slot);
  TensorMatcher({K})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(slot_expert)
      .verify(slot_lastuse)
      .verify(src)
      .verify(dst);
  TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(step).verify(count);
  RuntimeCheck(T.unwrap() <= K.unwrap(), "paged_experts_decide needs at most K routed entries, got ", T.unwrap());

  LaunchKernel(1, 32, device_.unwrap())(
      paged_experts_decide_kernel,
      static_cast<const int32_t*>(topk.data_ptr()),
      static_cast<int>(T.unwrap()),
      static_cast<int>(E.unwrap()),
      static_cast<int>(K.unwrap()),
      static_cast<int32_t*>(step.data_ptr()),
      static_cast<int32_t*>(slot_expert.data_ptr()),
      static_cast<int32_t*>(expert_slot.data_ptr()),
      static_cast<int32_t*>(slot_lastuse.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(count.data_ptr()));
}

void paged_experts_gather(
    tvm::ffi::TensorView hosts,
    tvm::ffi::TensorView gpus,
    tvm::ffi::TensorView words,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView count) {
  using namespace sglang::host;

  SymbolicSize P = {"num_tensors"}, K = {"num_slots"}, One = {"one"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({P}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(hosts).verify(gpus).verify(words);
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(src).verify(dst);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(count);

  // Decode pages few experts per step, so launch cost matters more than link saturation: a
  // 2048x512 grid saturates the link better but measured 5% slower end to end (Qwen3-30B, K=90).
  // An empty plan exits at once.
  LaunchKernel(256, 256, device_.unwrap())(
      paged_experts_gather_kernel,
      static_cast<const int64_t*>(hosts.data_ptr()),
      static_cast<const int64_t*>(gpus.data_ptr()),
      static_cast<const int64_t*>(words.data_ptr()),
      static_cast<int>(P.unwrap()),
      static_cast<const int32_t*>(src.data_ptr()),
      static_cast<const int32_t*>(dst.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()));
}

int64_t paged_experts_host_device_pointer(tvm::ffi::TensorView pinned) {
  void* ptr = nullptr;
  const cudaError_t err = cudaHostGetDevicePointer(&ptr, pinned.data_ptr(), 0);
  sglang::host::RuntimeCheck(err == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  return reinterpret_cast<int64_t>(ptr);
}

}  // namespace
