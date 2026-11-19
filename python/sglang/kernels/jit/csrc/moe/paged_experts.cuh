// Paged experts (srt/layers/moe/paged_experts): the residency decision and the expert page-in on
// the GPU, so a decode step pages experts without a host sync and can be captured in a CUDA graph.
//
// decide: for a step whose routed (token, expert) entries fit the K slots, keep every resident
// expert the step uses, give each missing expert the least recently used slot the step does not
// need, and write the page-in plan (src expert -> dst slot, count) to device buffers. It also
// writes the runner's inputs: each routed entry's slot and router weight, slot 0 and weight 0 for
// padding, so no further kernel is needed to build them.
// gather: copy the planned experts from the pinned host store into their slots, for every paged
// tensor of a layer in one launch. The count is read on the device, so a replayed graph moves
// exactly the experts its decide chose.
// tiered_plan / rows_to_slots: with the experts in pinned host rows (the rest on disk), copy a
// planned expert from its row into its slot; the row keeps it (inclusive rows: nothing is written
// back). Experts only on disk are flagged for the host, which reads them while the captured step
// waits for them (await_rows), or after the step.
// precopy_plan: from the next layer's router logits on this layer's
// input, the next layer's predicted experts that are in none of its slots, given that layer's least
// recently used slots ahead of its decide; the gather (at a narrow grid, on a side stream) copies them.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
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

// Pre-copy: one block, one thread per expert (E <= 1024), on a side stream after this layer's
// gather. ``logits`` [E] (bf16): the next layer's softmax router on this layer's input. Its top_k + 1
// best (selection as the router's top-k, ties to the lower id), each one's margin in softmax
// probability over the (top_k + 1)-th; then, best first, each predicted expert whose margin beats ``margin`` and that
// sits in none of the next layer's slots, at most ``max_copies``, gets the least recently used slot holding no
// predicted expert (any margin) and not picked already, lowest slot on ties. The next layer's residency
// moves as its decide would move it, the slot stamped with its step counter as it is (its last
// decide's, so the decide that runs next sees the slot as used before this step). The plan goes to
// ``src`` / ``dst`` / ``count`` for ``paged_experts_gather``. ``last`` [K] (the experts the previous plan copied, -1:
// none) with ``last_stamp`` [1] and
// ``counts`` [4] (planned, used, wasted, predicted but below the margin or resident: cumulative) tally
// the previous plan first: a copied expert is used if its slot was stamped after the copy.
__global__ void paged_experts_precopy_plan_kernel(
    const __nv_bfloat16* logits,
    int top_k,
    float margin,
    int max_copies,
    int E,
    int K,
    const int32_t* step,
    int32_t* slot_expert,
    int32_t* expert_slot,
    int32_t* slot_lastuse,
    int32_t* src,
    int32_t* dst,
    int32_t* count,
    int32_t* last,
    int32_t* last_stamp,
    int32_t* counts) {
  // The plan runs on a side stream beside compute; every global read is issued once, by all
  // threads in parallel, so the kernel is a few round trips long (a serial loop over K slots
  // measured 45 us per call on Qwen3-Next, K=101).
  __shared__ float warp_val[32];
  __shared__ int warp_idx[32];
  __shared__ float best_val[33];
  __shared__ int best_idx[33];
  __shared__ float pred_margin[32];
  __shared__ int pred_slot[32];
  __shared__ float denom;
  __shared__ int s_lastuse[1024];
  __shared__ unsigned char s_keep[1024];
  __shared__ int tally[2];
  const int i = threadIdx.x, lane = i % 32, warp = i / 32, warps = (blockDim.x + 31) / 32;
  const int now = *step;
  const int then = *last_stamp;
  if (i < 2) tally[i] = 0;
  for (int t = i; t < K; t += blockDim.x) {
    s_lastuse[t] = slot_lastuse[t];
    s_keep[t] = 0;
  }
  __syncthreads();
  // Tally the previous plan (the next layer's decide ran since: used ones were stamped after it).
  for (int t = i; t < K; t += blockDim.x) {
    const int e = last[t];
    if (e < 0) continue;
    const int sl = (e < E) ? expert_slot[e] : -1;
    atomicAdd(&tally[(sl >= 0 && s_lastuse[sl] > then) ? 0 : 1], 1);
    last[t] = -1;
  }
  const float x = i < E ? __bfloat162float(logits[i]) : -INFINITY;
  float v = x;
  int idx = i < E ? i : INT_MAX;
  for (int r = 0; r <= top_k; ++r) {
    float bv = v;
    int bi = idx;
    for (int o = 16; o > 0; o /= 2) {
      const float ov = __shfl_xor_sync(0xffffffff, bv, o);
      const int oi = __shfl_xor_sync(0xffffffff, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if (lane == 0) warp_val[warp] = bv, warp_idx[warp] = bi;
    __syncthreads();
    if (i == 0) {
      for (int w = 1; w < warps; ++w)
        if (warp_val[w] > bv || (warp_val[w] == bv && warp_idx[w] < bi)) bv = warp_val[w], bi = warp_idx[w];
      best_val[r] = bv;
      best_idx[r] = bi;
    }
    __syncthreads();
    if (idx == best_idx[r]) v = -INFINITY, idx = INT_MAX;  // taken
  }
  {  // softmax: the denominator over every expert
    const float sum = sglang::device::warp::reduce_sum(i < E ? expf(x - best_val[0]) : 0.f);
    if (lane == 0) warp_val[warp] = sum;
    __syncthreads();
    if (i == 0) {
      float t = 0.f;
      for (int w = 0; w < warps; ++w)
        t += warp_val[w];
      denom = t;
    }
    __syncthreads();
  }
  if (i < top_k) {
    const float a = best_val[i], b = best_val[top_k];
    pred_margin[i] = (expf(a - best_val[0]) - expf(b - best_val[0])) / denom;
    const int e = best_idx[i];
    const int sl = (e >= 0 && e < E) ? expert_slot[e] : -1;
    pred_slot[i] = sl;
    if (sl >= 0) s_keep[sl] = 1;  // a predicted expert's slot is never a victim
  }
  __syncthreads();
  if (warp != 0) return;
  int n = 0, skipped = 0;
  for (int j = 0; j < top_k && n < max_copies; ++j) {
    const int e = best_idx[j];
    if (e < 0 || e >= E) continue;
    if (!(pred_margin[j] > margin) || pred_slot[j] >= 0) {  // the same for every lane
      ++skipped;
      continue;
    }
    // Victim: the least recently used slot holding no predicted expert and not picked already.
    int oldest = INT_MAX, victim = K;
    for (int sl = lane; sl < K; sl += 32) {
      const int lastuse = s_lastuse[sl];
      if (!s_keep[sl] && lastuse < oldest) {
        oldest = lastuse;
        victim = sl;
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
    if (victim == K) break;  // every slot holds a predicted expert
    if (lane == 0) {
      const int old = slot_expert[victim];
      if (old >= 0) expert_slot[old] = -1;
      slot_expert[victim] = e;
      expert_slot[e] = victim;
      slot_lastuse[victim] = now;
      s_lastuse[victim] = now;
      last[n] = e;
      s_keep[victim] = 1;
      src[n] = e;
      dst[n] = victim;
    }
    ++n;
    __syncwarp();  // the next search reads this pick
  }
  if (lane == 0) {
    *count = n;
    *last_stamp = now;
    counts[0] += n;
    counts[1] += tally[0];
    counts[2] += tally[1];
    counts[3] += skipped;
  }
}

// Calls ``op(host, gpu, row)`` with each paged tensor's rows typed by the widest load their size
// allows: 16 bytes, which the host link needs, else 4 (small per-expert scalars) or 2 (e.g. an
// NVFP4 Marlin global scale). ``row`` is in elements of that type.
template <typename Op>
__device__ void for_each_tensor(const int64_t* hosts, const int64_t* gpus, const int64_t* row_bytes, int P, Op op) {
  for (int p = 0; p < P; ++p) {
    const long bytes = row_bytes[p];
    if (bytes % 16 == 0) {
      op(reinterpret_cast<float4*>(hosts[p]), reinterpret_cast<float4*>(gpus[p]), bytes / 16);
    } else if (bytes % 4 == 0) {
      op(reinterpret_cast<int32_t*>(hosts[p]), reinterpret_cast<int32_t*>(gpus[p]), bytes / 4);
    } else {
      op(reinterpret_cast<int16_t*>(hosts[p]), reinterpret_cast<int16_t*>(gpus[p]), bytes / 2);
    }
  }
}

__device__ inline long grid_stride() {
  return static_cast<long>(gridDim.x) * blockDim.x;
}

__device__ inline long grid_tid() {
  return static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
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
  const long stride = grid_stride();
  for_each_tensor(hosts, gpus, row_bytes, P, [&](auto* host, auto* gpu, long row) {
    for (long j = grid_tid(); j < n * row; j += stride) {
      const long i = j / row, off = j % row;
      gpu[dst[i] * row + off] = host[src[i] * row + off];
    }
  });
}

// Copies host row ``rows[i]`` into GPU slot ``slots[i]`` element by element, one way: the row
// keeps its expert and nothing is written back (inclusive host rows). Rows below zero are skipped.
__global__ void paged_experts_rows_to_slots_kernel(
    const int64_t* hosts,
    const int64_t* gpus,
    const int64_t* row_bytes,
    int P,
    const int32_t* rows,
    const int32_t* slots,
    const int32_t* count) {
  const int n = *count;
  for_each_tensor(hosts, gpus, row_bytes, P, [&](auto* host, auto* gpu, long row) {
    const long stride = grid_stride();
    for (long j = grid_tid(); j < n * row; j += stride) {
      const long i = j / row, off = j % row;
      const int r = rows[i];
      if (r < 0) continue;  // not in host memory: the host pages it in
      gpu[slots[i] * row + off] = host[r * row + off];
    }
  });
}

// One block, one thread per planned entry: record what the copy will leave on disk (doorbell rows:
// missing, src, dst), each entry's host row (``rows_out``, -1: on disk or unused) and
// stamp the rows the copy reads. The maps keep every expert where it is (inclusive rows). Then,
// with ``publish_row``, write the layer's state row to the host and the new clock as its flag.
__global__ void paged_experts_tiered_update_kernel(
    const int32_t* src,
    const int32_t* dst,
    const int32_t* count,
    const int32_t* expert_row,
    int32_t* row_stamp,
    int32_t* clock,
    int32_t* doorbell,  // [3, K]
    int K,
    int32_t* rows_out,               // [K]
    volatile int32_t* publish_row,   // null, or the layer's mailbox row, UVA
    volatile int32_t* publish_flag,  // ... and its flag
    int row_len) {                   // the state row: doorbell [3K] first, the clock last
  // The step's stamp, read once: with more than one warp, a thread reading ``clock`` after thread 0
  // advanced it would stamp its rows one ahead of the published clock.
  __shared__ int step_stamp;
  if (threadIdx.x == 0) step_stamp = *clock + 1;
  __syncthreads();
  const int n = *count;
  const int stamp = step_stamp;
  int e = -1, s = 0, r = -1;
  const int i = threadIdx.x;
  if (i < K) {
    if (i < n) {
      e = src[i];
      s = dst[i];
      r = expert_row[e];
    }
    doorbell[i] = (i < n && r < 0) ? 1 : 0;
    doorbell[K + i] = e;
    doorbell[2 * K + i] = s;
    rows_out[i] = r;
  }
  if (i < K && i < n && r >= 0) row_stamp[r] = stamp;
  if (i == 0) *clock = stamp;
  if (publish_row == nullptr) return;
  // The state row to the host, then the clock as its flag (the host polls the flag).
  __syncthreads();
  for (int j = threadIdx.x; j < row_len; j += blockDim.x)
    publish_row[j] = doorbell[j];
  __threadfence_system();  // each thread's part of the row before the flag
  __syncthreads();
  if (i == 0) publish_flag[0] = stamp;
}

// A captured step that left experts of a layer on disk waits here for the host instead of
// running on with the slots' old experts: one thread spins on the layer's reply row in pinned
// host memory until its ticket equals the step's ``clock`` (the host echoes the clock it saw in
// the published row), then the block copies the host's (row, slot) plan to ``plan`` ``[1 + 2K]``
// for the copy launched after it, writes the host's row maps into ``maps``, clears the missing
// flags and acks the ticket. A layer that left nothing on disk does not wait. After
// ``timeout_ns`` the step runs on unchanged (an empty plan, the flags kept): the replay that
// follows serves the layer as before.
// Reply row: [ticket, n, rows[K], slots[K], maps[E + 2R + 1]].
__device__ __forceinline__ unsigned long long global_timer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void paged_experts_await_rows_kernel(
    int32_t* doorbell,  // [3, K]
    int32_t* maps,      // [E + 2R + 1]
    int maps_len,
    const int32_t* clock,
    const volatile int32_t* reply,
    volatile int32_t* ack,
    int32_t* plan,  // [1 + 2K]: n, rows, slots
    int K,
    long long timeout_ns) {
  __shared__ int served, ticket;
  if (threadIdx.x == 0) {
    int missing = 0;
    for (int i = 0; i < K; ++i)
      missing |= doorbell[i];
    served = 0;
    plan[0] = 0;
    ticket = *clock;  // ``maps`` ends with the clock: read before the copy replaces it
    if (missing) {
      const unsigned long long start = global_timer_ns();
      while (true) {
        if (reply[0] == ticket) {
          served = 1;
          break;
        }
        if (static_cast<long long>(global_timer_ns() - start) > timeout_ns) break;
        __nanosleep(1000);
      }
    }
    __threadfence_system();  // the reply's payload after its ticket
  }
  __syncthreads();
  if (!served) return;
  const int n = reply[1];
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    plan[1 + i] = i < n ? reply[2 + i] : -1;
    plan[1 + K + i] = i < n ? reply[2 + K + i] : 0;
    doorbell[i] = 0;
  }
  for (int j = threadIdx.x; j < maps_len; j += blockDim.x)
    maps[j] = reply[2 + 2 * K + j];
  __syncthreads();
  if (threadIdx.x == 0) {
    plan[0] = n;
    __threadfence_system();
    ack[0] = ticket;
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
    tvm::ffi::TensorView count,
    int64_t blocks) {
  using namespace sglang::host;
  using sglang::kDLGPU;

  SymbolicSize P = {"num_tensors"}, K = {"num_slots"}, One = {"one"};
  SymbolicDevice device_;
  device_.set_options<kDLGPU>();
  TensorMatcher({P}).with_dtype<int64_t>().with_device<kDLGPU>(device_).verify(hosts).verify(gpus).verify(row_bytes);
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(src).verify(dst);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(count);

  // Decode pages few experts per step, so launch cost matters more than link saturation: a
  // 2048x512 grid saturates the link better but measured 5% slower end to end (Qwen3-30B, K=90, RTX 5070 Ti).
  // An empty plan exits at once. A copy beside compute (pre-copy) takes a narrow grid: 32 blocks
  // overlap a layer's compute 0.88-0.95 at -1.6 % bandwidth, 256 blocks 0.32 (RTX 5070 Ti).
  RuntimeCheck(blocks >= 1 && blocks <= 4096, "paged_experts_gather: blocks out of range: ", blocks);
  LaunchKernel(static_cast<int>(blocks), 256, device_.unwrap())(
      paged_experts_gather_kernel,
      static_cast<const int64_t*>(hosts.data_ptr()),
      static_cast<const int64_t*>(gpus.data_ptr()),
      static_cast<const int64_t*>(row_bytes.data_ptr()),
      static_cast<int>(P.unwrap()),
      static_cast<const int32_t*>(src.data_ptr()),
      static_cast<const int32_t*>(dst.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()));
}

void paged_experts_rows_to_slots(
    tvm::ffi::TensorView hosts,
    tvm::ffi::TensorView gpus,
    tvm::ffi::TensorView row_bytes,
    tvm::ffi::TensorView rows,
    tvm::ffi::TensorView slots,
    tvm::ffi::TensorView count) {
  using namespace sglang::host;
  using sglang::kDLGPU;

  SymbolicSize P = {"num_tensors"}, N = {"num_pairs"}, One = {"one"};
  SymbolicDevice device_;
  device_.set_options<kDLGPU>();
  TensorMatcher({P}).with_dtype<int64_t>().with_device<kDLGPU>(device_).verify(hosts).verify(gpus).verify(row_bytes);
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(rows).verify(slots);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(count);

  LaunchKernel(256, 256, device_.unwrap())(
      paged_experts_rows_to_slots_kernel,
      static_cast<const int64_t*>(hosts.data_ptr()),
      static_cast<const int64_t*>(gpus.data_ptr()),
      static_cast<const int64_t*>(row_bytes.data_ptr()),
      static_cast<int>(P.unwrap()),
      static_cast<const int32_t*>(rows.data_ptr()),
      static_cast<const int32_t*>(slots.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()));
}

void paged_experts_tiered_plan(
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView expert_row,
    tvm::ffi::TensorView row_expert,
    tvm::ffi::TensorView row_stamp,
    tvm::ffi::TensorView clock,
    tvm::ffi::TensorView doorbell,
    tvm::ffi::TensorView rows_out,
    int64_t publish_row_address,
    int64_t publish_flag_address,
    int64_t row_len) {
  using namespace sglang::host;
  using sglang::kDLGPU;

  SymbolicSize K = {"num_slots"}, One = {"one"}, E = {"num_experts"}, R = {"num_rows"}, Three = {"three"};
  SymbolicDevice device_;
  device_.set_options<kDLGPU>();
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(src).verify(dst).verify(rows_out);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(count).verify(clock);
  TensorMatcher({E}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(expert_row);
  TensorMatcher({R}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(row_expert).verify(row_stamp);
  TensorMatcher({Three, K}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(doorbell);
  const int k = static_cast<int>(K.unwrap());
  RuntimeCheck(k <= 1024, "paged experts: more slots than one block has threads");
  RuntimeCheck(
      publish_row_address == 0 || (publish_flag_address != 0 && row_len >= 3 * k),
      "paged experts: a publish needs the flag and the whole state row");
  // The missing flags and the rows first, so the step can publish its disk misses before it copies
  // its host-row hits (paged_experts_rows_to_slots over rows_out).
  LaunchKernel(1, (k + 31) / 32 * 32, device_.unwrap())(
      paged_experts_tiered_update_kernel,
      static_cast<const int32_t*>(src.data_ptr()),
      static_cast<const int32_t*>(dst.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      static_cast<const int32_t*>(expert_row.data_ptr()),
      static_cast<int32_t*>(row_stamp.data_ptr()),
      static_cast<int32_t*>(clock.data_ptr()),
      static_cast<int32_t*>(doorbell.data_ptr()),
      k,
      static_cast<int32_t*>(rows_out.data_ptr()),
      reinterpret_cast<volatile int32_t*>(publish_row_address),
      reinterpret_cast<volatile int32_t*>(publish_flag_address),
      static_cast<int>(row_len));
}

void paged_experts_await_rows(
    tvm::ffi::TensorView doorbell,
    tvm::ffi::TensorView maps,
    tvm::ffi::TensorView clock,
    int64_t reply_address,
    int64_t ack_address,
    tvm::ffi::TensorView plan,
    int64_t timeout_ns) {
  using namespace sglang::host;
  using sglang::kDLGPU;

  SymbolicSize K = {"num_slots"}, M = {"maps"}, One = {"one"}, Three = {"three"}, Plan = {"plan"};
  SymbolicDevice device_;
  device_.set_options<kDLGPU>();
  TensorMatcher({Three, K}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(doorbell);
  TensorMatcher({M}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(maps);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(clock);
  TensorMatcher({Plan}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(plan);
  const int k = static_cast<int>(K.unwrap());
  RuntimeCheck(static_cast<int>(Plan.unwrap()) == 1 + 2 * k, "paged experts: the plan holds 1 + 2K entries");

  LaunchKernel(1, 128, device_.unwrap())(
      paged_experts_await_rows_kernel,
      static_cast<int32_t*>(doorbell.data_ptr()),
      static_cast<int32_t*>(maps.data_ptr()),
      static_cast<int>(M.unwrap()),
      static_cast<const int32_t*>(clock.data_ptr()),
      reinterpret_cast<const volatile int32_t*>(reply_address),
      reinterpret_cast<volatile int32_t*>(ack_address),
      static_cast<int32_t*>(plan.data_ptr()),
      k,
      static_cast<long long>(timeout_ns));
}

void paged_experts_precopy_plan(
    tvm::ffi::TensorView logits,
    int64_t top_k,
    double margin,
    int64_t max_copies,
    tvm::ffi::TensorView step,
    tvm::ffi::TensorView slot_expert,
    tvm::ffi::TensorView expert_slot,
    tvm::ffi::TensorView slot_lastuse,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView last,
    tvm::ffi::TensorView last_stamp,
    tvm::ffi::TensorView counts) {
  using namespace sglang::host;
  using sglang::kDLGPU;

  SymbolicSize E = {"num_experts"}, K = {"num_slots"}, One = {"one"}, Four = {"four"};
  SymbolicDevice device_;
  device_.set_options<kDLGPU>();
  TensorMatcher({E}).with_dtype<sglang::bf16_t>().with_device<kDLGPU>(device_).verify(logits);
  TensorMatcher({E}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(expert_slot);
  TensorMatcher({K})
      .with_dtype<int32_t>()
      .with_device<kDLGPU>(device_)
      .verify(slot_expert)
      .verify(slot_lastuse)
      .verify(src)
      .verify(dst)
      .verify(last);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(step).verify(count).verify(last_stamp);
  TensorMatcher({Four}).with_dtype<int32_t>().with_device<kDLGPU>(device_).verify(counts);
  RuntimeCheck(Four.unwrap() == 4, "paged experts pre-copy: counts holds 4 counters");
  const int64_t e = E.unwrap(), k = K.unwrap();
  RuntimeCheck(e >= 1 && e <= 1024, "paged experts pre-copy: at most 1024 experts, got ", e);
  RuntimeCheck(top_k >= 1 && top_k <= 32 && top_k < e, "paged experts pre-copy: top_k out of range: ", top_k);
  RuntimeCheck(max_copies >= 0 && max_copies <= k, "paged experts pre-copy: max_copies out of range: ", max_copies);
  RuntimeCheck(k <= 1024, "paged experts pre-copy: at most 1024 slots, got ", k);
  const int threads = static_cast<int>((e + 31) / 32 * 32);
  LaunchKernel(1, threads, device_.unwrap())(
      paged_experts_precopy_plan_kernel,
      static_cast<const __nv_bfloat16*>(logits.data_ptr()),
      static_cast<int>(top_k),
      static_cast<float>(margin),
      static_cast<int>(max_copies),
      static_cast<int>(e),
      static_cast<int>(k),
      static_cast<const int32_t*>(step.data_ptr()),
      static_cast<int32_t*>(slot_expert.data_ptr()),
      static_cast<int32_t*>(expert_slot.data_ptr()),
      static_cast<int32_t*>(slot_lastuse.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(count.data_ptr()),
      static_cast<int32_t*>(last.data_ptr()),
      static_cast<int32_t*>(last_stamp.data_ptr()),
      static_cast<int32_t*>(counts.data_ptr()));
}

int64_t paged_experts_host_device_pointer(tvm::ffi::TensorView pinned) {
  void* ptr = nullptr;
  const cudaError_t err = cudaHostGetDevicePointer(&ptr, pinned.data_ptr(), 0);
  sglang::host::RuntimeCheck(err == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  return reinterpret_cast<int64_t>(ptr);
}

}  // namespace
