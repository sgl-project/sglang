// Paged experts (srt/layers/moe/paged_experts) on the GPU: a decode step pages experts with no
// host sync, so the whole step can be captured in a CUDA graph.
// decide: keep the resident experts the step uses, evict LRU slots it does not need, and write
// both the page-in plan and the runner's slot/weight inputs.
// gather: copy the planned experts of a layer from the pinned host store in one launch; the
// count is read on the device, so a replayed graph moves exactly what its decide chose.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <climits>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

// The decision is latency bound: it moves the residency state (4.5 KB at E=512/K=400) and the T
// routed ids, so stage them block-wide rather than one dependent load per 32 scalars on warp 0.
// Graph replay at E=512/K=400/T=80: 14.88 us at 1 warp, 4.36 at 8; 16 buys nothing.
constexpr int kDecideThreads = 256;

// The evictions are serial and committed one at a time by lane 0 of warp 0; each victim search is
// spread over that warp. The scan over the routed entries is a ballot compaction, and the search
// reads the staged last-use copy instead of paying a global round trip per page-in.
__global__ void paged_experts_decide_kernel(
    const int32_t* topk,   // [T] routed expert ids of the step (negative: padding)
    const float* weights,  // [T] their router weights
    int T,
    int E,
    int K,
    int32_t* step,          // [1] step counter, advanced on the device so replays age the LRU
    int32_t* slot_expert,   // [K] expert in each slot (-1: empty)
    int32_t* expert_slot,   // [E] slot of each expert (-1: not resident)
    int32_t* slot_lastuse,  // [K] step each slot was last used
    int32_t* src,           // [K] out: experts to page in
    int32_t* dst,           // [K] out: their slots
    int32_t* count,         // [1] out: number of page-ins
    int32_t* run_ids,       // [T] out: slot of each entry (0 for padding)
    float* run_w) {         // [T] out: weight of each entry (0 for padding)
  extern __shared__ int s_state[];
  int32_t* s_expert_slot = s_state;                    // [E]
  int32_t* s_slot_expert = s_state + E;                // [K]
  int32_t* s_lastuse = s_state + E + K;                // [K]
  int32_t* s_topk = s_state + E + 2 * K;               // [T]
  int32_t* s_miss = s_state + E + 2 * K + T;           // [T] entry positions whose expert is not resident
  int32_t* s_now = s_state + E + 2 * K + 2 * T;        // [1] the step, broadcast from thread 0
  int32_t* s_count = s_state + E + 2 * K + 2 * T + 1;  // [1] the plan size, broadcast from warp 0

  const int tid = threadIdx.x;
  const int lane = tid & 31;

  for (int i = tid; i < K; i += blockDim.x) {
    s_slot_expert[i] = slot_expert[i];
    s_lastuse[i] = slot_lastuse[i];
  }
  for (int i = tid; i < E; i += blockDim.x)
    s_expert_slot[i] = expert_slot[i];
  for (int i = tid; i < T; i += blockDim.x)
    s_topk[i] = topk[i];
  __syncthreads();

  if (tid == 0) *s_now = ++(*step);
  __syncthreads();
  const int now = *s_now;
  for (int i = tid; i < T; i += blockDim.x) {
    const int e = s_topk[i];
    if (e < 0 || e >= E) continue;
    const int s = s_expert_slot[e];
    if (s >= 0) s_lastuse[s] = now;
  }
  __syncthreads();

  // Warp 0: the compaction keeps the entry order, and the evictions are serial by definition.
  if (tid < 32) {
    int nmiss = 0;
    for (int base = 0; base < T; base += 32) {
      const int i = base + lane;
      const int e = (i < T) ? s_topk[i] : -1;
      const bool miss = (e >= 0 && e < E && s_expert_slot[e] < 0);
      const unsigned bits = __ballot_sync(0xffffffff, miss);
      if (miss) s_miss[nmiss + __popc(bits & ((1u << lane) - 1u))] = i;
      nmiss += __popc(bits);
    }
    __syncwarp();

    int n = 0;
    for (int j = 0; j < nmiss; ++j) {
      const int e = s_topk[s_miss[j]];
      if (e < 0 || e >= E || s_expert_slot[e] >= 0) continue;  // the same for every lane
      // Victim: LRU slot unused this step, lowest slot on ties (one is left while T <= K).
      int oldest = INT_MAX, victim = K;
      for (int s = lane; s < K; s += 32) {
        const int lastuse = s_lastuse[s];
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
        const int old = s_slot_expert[victim];
        if (old >= 0) s_expert_slot[old] = -1;
        s_slot_expert[victim] = e;
        s_expert_slot[e] = victim;
        s_lastuse[victim] = now;
        src[n] = e;
        dst[n] = victim;
      }
      ++n;
      __syncwarp();  // the next search reads this eviction
    }
    if (lane == 0) {
      *count = n;
      *s_count = n;
    }
  }
  __syncthreads();  // the slots of the experts paged in above, and the plan size
  const int n = *s_count;
  for (int i = tid; i < T; i += blockDim.x) {
    const int e = s_topk[i];
    const int s = (e < 0 || e >= E) ? -1 : s_expert_slot[e];
    run_ids[i] = s >= 0 ? s : 0;
    run_w[i] = s >= 0 ? weights[i] : 0.f;
  }
  __syncthreads();  // every reader of the staged state has finished with it
  // Without a page-in both slot tables are exactly as they were staged: only the last-use stamps
  // of the resident experts moved. (Writing just the touched slots instead is not cheaper: 80
  // scattered stores against 400 coalesced ones.)
  if (n > 0) {
    for (int i = tid; i < K; i += blockDim.x)
      slot_expert[i] = s_slot_expert[i];
    for (int i = tid; i < E; i += blockDim.x)
      expert_slot[i] = s_expert_slot[i];
  }
  for (int i = tid; i < K; i += blockDim.x)
    slot_lastuse[i] = s_lastuse[i];
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
    const int64_t* hosts,      // [P] UVA device pointer of each paged tensor's host store
    const int64_t* gpus,       // [P] each paged tensor's K-slot GPU table
    const int64_t* row_bytes,  // [P] bytes per expert of each tensor, even
    int P,
    const int32_t* src,
    const int32_t* dst,
    const int32_t* count) {
  const int n = *count;
  if (n == 0) return;  // most decode steps page nothing
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int p = 0; p < P; ++p) {
    const long bytes = row_bytes[p];
    if (bytes % 16 == 0) {  // copy in float4, which the host link needs
      copy_rows(
          reinterpret_cast<const float4*>(hosts[p]),
          reinterpret_cast<float4*>(gpus[p]),
          bytes / 16,
          n,
          src,
          dst,
          tid,
          stride);
    } else if (bytes % 4 == 0) {  // small per-expert scalars
      copy_rows(
          reinterpret_cast<const int32_t*>(hosts[p]),
          reinterpret_cast<int32_t*>(gpus[p]),
          bytes / 4,
          n,
          src,
          dst,
          tid,
          stride);
    } else {  // 16-bit per-expert scalars (e.g. an NVFP4 Marlin global scale)
      copy_rows(
          reinterpret_cast<const int16_t*>(hosts[p]),
          reinterpret_cast<int16_t*>(gpus[p]),
          bytes / 2,
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
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView step,
    tvm::ffi::TensorView slot_expert,
    tvm::ffi::TensorView expert_slot,
    tvm::ffi::TensorView slot_lastuse,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView run_ids,
    tvm::ffi::TensorView run_w) {
  using namespace sglang::host;
  using sglang::kDLGPU;

  SymbolicSize E = {"num_experts"}, K = {"num_slots"}, T = {"num_entries"}, One = {"one"};
  SymbolicDevice device_;
  device_.set_options<kDLGPU>();
  TensorMatcher({E}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(expert_slot);
  TensorMatcher({K})
      .with_dtype<int32_t>()
      .with_device<kDLGPU>(device_)
      .verify(slot_expert)
      .verify(slot_lastuse)
      .verify(src)
      .verify(dst);
  TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(topk).verify(run_ids);
  TensorMatcher({T}).with_dtype<float>().with_device<kDLGPU>(device_).verify(weights).verify(run_w);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(step).verify(count);
  RuntimeCheck(T.unwrap() <= K.unwrap(), "paged_experts_decide needs at most K routed entries, got ", T.unwrap());

  // The kernel stages expert_slot[E], slot_expert[K], slot_lastuse[K], topk[T], miss[T], step[1], count[1].
  const int smem_bytes = static_cast<int>((E.unwrap() + 2 * K.unwrap() + 2 * T.unwrap() + 2) * sizeof(int32_t));
#ifndef USE_ROCM
  if (smem_bytes > 48 * 1024) {
    // Past the default 48 KB reservation: opt in once per device to the device maximum, not the
    // first launch's size, because auto K can give later layers a larger K (the launch bound then
    // enforces the actual size).
    runtime::set_max_dynamic_smem_per_device<paged_experts_decide_kernel>(
        device_.unwrap(), runtime::get_max_smem_per_block(device_.unwrap().device_id));
  }
#endif
  LaunchKernel(1, kDecideThreads, device_.unwrap(), smem_bytes)(
      paged_experts_decide_kernel,
      static_cast<const int32_t*>(topk.data_ptr()),
      static_cast<const float*>(weights.data_ptr()),
      static_cast<int>(T.unwrap()),
      static_cast<int>(E.unwrap()),
      static_cast<int>(K.unwrap()),
      static_cast<int32_t*>(step.data_ptr()),
      static_cast<int32_t*>(slot_expert.data_ptr()),
      static_cast<int32_t*>(expert_slot.data_ptr()),
      static_cast<int32_t*>(slot_lastuse.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(count.data_ptr()),
      static_cast<int32_t*>(run_ids.data_ptr()),
      static_cast<float*>(run_w.data_ptr()));
}

void paged_experts_gather(
    tvm::ffi::TensorView hosts,
    tvm::ffi::TensorView gpus,
    tvm::ffi::TensorView row_bytes,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView count) {
  using namespace sglang::host;
  using sglang::kDLGPU;

  SymbolicSize P = {"num_tensors"}, K = {"num_slots"}, One = {"one"};
  SymbolicDevice device_;
  device_.set_options<kDLGPU>();
  TensorMatcher({P}).with_dtype<int64_t>().with_device<kDLGPU>(device_).verify(hosts).verify(gpus).verify(row_bytes);
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(src).verify(dst);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(count);

  // Decode pages few experts per step, so launch cost beats link saturation: 2048x512 measured 5%
  // slower end to end (Qwen3-30B, K=90).
  constexpr int kGatherBlocks = 256;
  constexpr int kGatherThreads = 256;
  LaunchKernel(kGatherBlocks, kGatherThreads, device_.unwrap())(
      paged_experts_gather_kernel,
      static_cast<const int64_t*>(hosts.data_ptr()),
      static_cast<const int64_t*>(gpus.data_ptr()),
      static_cast<const int64_t*>(row_bytes.data_ptr()),
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
