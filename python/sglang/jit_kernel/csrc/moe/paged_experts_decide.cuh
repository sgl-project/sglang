// On-device residency decision for Paged Experts (srt/layers/moe/paged_experts).
//
// Replaces the host-side keep-warm/LRU decision so the per-decode-step paging plan is computed entirely
// on the GPU — no host sync (`.tolist()`), which is what makes the decode step CUDA-graph-capturable.
// The kernels only *decide* (which experts to page in, which slots to evict, and the logical->slot remap);
// the actual weight movement is done by the gather kernels below (capturable — they read the plan count
// on-device). One WARP per decide: the eviction loop has sequential dependencies (each eviction depends
// on the prior assignments) and stays on lane 0, but the bookkeeping passes around it — recency bumps,
// the full-E map snapshot, the needed[] scan — are embarrassingly parallel and dominate the serial time
// at real sizes (E=64..256, K up to E), so they fan out across the warp.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <cuda_runtime.h>  // For cudaHostGetDevicePointer (UVA device pointer of the pinned store)
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <cstdint>

namespace {

// Victim = a non-needed resident slot, chosen by eviction policy: LFU (lfu != 0) -> minimum use count of
// the resident expert, LRU recency as tiebreak; LRU (lfu == 0) -> minimum last-use step. Empty slots
// (slot_expert < 0) get freq key 0 so they are filled first. Mirrors the host path's ResidencyPolicy.
// ``log2hot`` (nullable) makes the choice TIER-AWARE for the windowed store: re-fetching a hot-tier
// (pinned-window) expert is a cheap in-graph gather while re-fetching a cold-tier resident costs a
// deferred replay/break round — so among non-needed slots, hot-tier residents (and empty slots) are
// strictly preferred as victims; a cold-tier resident is evicted only when no hot victim exists.
SGL_DEVICE int pick_victim(
    const int32_t* topk,
    int topk_n,
    int K,
    int lfu,
    const int32_t* slot_expert,
    const int32_t* slot_lastuse,
    const int32_t* freq,
    const int32_t* log2hot) {
  int victim = -1, best_cold = INT_MAX, best_f = INT_MAX, best_lu = INT_MAX;
  for (int s = 0; s < K; ++s) {
    const int se = slot_expert[s];
    bool needed = false;
    for (int j = 0; j < topk_n; ++j) {
      if (topk[j] == se) {
        needed = true;
        break;
      }
    }
    if (needed) continue;  // never evict a slot this step still needs
    const int cold = (log2hot != nullptr && se >= 0 && log2hot[se] < 0) ? 1 : 0;
    const int f = (lfu && se >= 0) ? freq[se] : 0;  // LRU: f == 0 always -> falls through to lastuse
    const int lu = slot_lastuse[s];
    if (cold < best_cold || (cold == best_cold && (f < best_f || (f == best_f && lu < best_lu)))) {
      best_cold = cold;
      best_f = f;
      best_lu = lu;
      victim = s;
    }
  }
  return victim;
}

// Keep-warm + LRU/LFU decision (distinct active experts <= K). Mutates the residency state in place and
// emits the page-in plan: for each distinct active expert not resident, evict a non-needed slot and assign
// it. src[0..n)/dst[0..n) are the (expert -> slot) page-ins; n_out is their count; idx[e] is the updated
// logical->slot map (-1 == not resident) that the forward remap reads.
__global__ void decide_kernel(
    const int32_t* topk,
    int topk_n,
    int E,
    int K,
    int lfu,
    int32_t* step_ctr,       // [1] monotonic step counter, incremented on-device (capture-safe)
    int32_t* slot_expert,    // [K] slot -> expert id (-1 == empty), mutated
    int32_t* expert_slot,    // [E] expert -> slot (-1 == not resident), mutated
    int32_t* slot_lastuse,   // [K] last step each slot was used, mutated
    int32_t* freq,           // [E] per-expert use count (LFU key), mutated
    int32_t* src,            // [>=K] out: page-in source experts
    int32_t* dst,            // [>=K] out: page-in destination slots
    int32_t* n_out,          // [1]  out: number of page-ins
    int32_t* idx) {          // [E]  out: logical -> slot map snapshot
  if (blockIdx.x || threadIdx.x >= 32) return;
  const int lane = threadIdx.x;
  // The step counter lives on-device and is bumped here so a captured graph advances LRU recency on every
  // replay (a host-scalar step would be frozen at capture time).
  int step = 0;
  if (lane == 0) step = ++(*step_ctr);
  step = __shfl_sync(0xffffffff, step, 0);
  // pass 1 (warp-parallel): bump per-expert use count (LFU key) and resident-hit recency. atomicAdd keeps
  // the duplicate-occurrence semantics of the serial version; lastuse writers all store the same step.
  for (int i = lane; i < topk_n; i += 32) {
    const int e = topk[i];
    if (e < 0 || e >= E) continue;
    atomicAdd(&freq[e], 1);
    const int s = expert_slot[e];
    if (s >= 0) slot_lastuse[s] = step;
  }
  __syncwarp();
  // pass 2 (lane 0, serial): each miss evicts a non-needed slot per the policy and pages its expert in.
  // Sequential by nature — every eviction changes the state the next pick depends on.
  if (lane == 0) {
    int n = 0;
    for (int i = 0; i < topk_n; ++i) {
      const int e = topk[i];
      if (e < 0 || e >= E) continue;
      if (expert_slot[e] >= 0) continue;  // resident (or just assigned this step)
      const int victim =
          pick_victim(topk, topk_n, K, lfu, slot_expert, slot_lastuse, freq, nullptr);
      if (victim < 0) continue;  // pool too small (should not happen: distinct <= K)
      const int old = slot_expert[victim];
      if (old >= 0) expert_slot[old] = -1;
      slot_expert[victim] = e;
      expert_slot[e] = victim;
      slot_lastuse[victim] = step;
      src[n] = e;
      dst[n] = victim;
      ++n;
    }
    *n_out = n;
  }
  __syncwarp();
  // map snapshot (warp-parallel)
  for (int e = lane; e < E; e += 32) idx[e] = expert_slot[e];
}

// Bounded keep-warm + LRU/LFU decision for the pinned-WINDOW store (distinct active experts <= K). Same
// residency logic as ``decide_kernel`` but partitions the page-in plan by window membership so the captured
// gather only ever reads the pinned hot block. ``log2hot[e]`` = hot-block index if expert e is in the
// pinned window, else -1.
//   * window hit (hot) -> (src=hot index, dst=slot) in the windowed plan -> on-device gather from host_hot.
//   * cold miss: the expert isn't gatherable in-graph (pageable RAM / disk). Record its LOGICAL id in
//     ``cold_log`` and leave it UNRESIDENT (idx stays -1 -> masked this replay) with NO eviction (don't
//     displace a window hit for an expert we can't gather). The host stages it into the window
//     out-of-graph — at the post-replay refill (replay-twice) or an in-layer eager break (BCG).
// ``needed[s]`` = 1 iff slot s holds an expert needed THIS step (logical id in topk): the refill
// must not evict these, else a still-needed expert re-misses and the loop never converges.
__global__ void decide_bounded_kernel(
    const int32_t* topk,
    int topk_n,
    int E,
    int K,
    int lfu,
    const int32_t* log2hot,  // [E] hot-block index if e in window, else -1
    int32_t* step_ctr,       // [1] monotonic step counter, incremented on-device (capture-safe)
    int32_t* slot_expert,    // [K] slot -> expert id (-1 == empty), mutated
    int32_t* expert_slot,    // [E] expert -> slot (-1 == not resident), mutated
    int32_t* slot_lastuse,   // [K] last step each slot was used, mutated
    int32_t* freq,           // [E] per-expert use count (LFU key), mutated
    int32_t* src,            // [>=K] out: windowed page-in source (hot-block index)
    int32_t* dst,            // [>=K] out: windowed page-in destination slots
    int32_t* n_out,          // [1]  out: number of windowed page-ins
    int32_t* cold_log,       // [>=K] out: deferred cold misses (logical expert ids)
    int32_t* cold_n,         // [1]  out: number of cold entries
    int64_t doorbell,        // 0, or a MAPPED PINNED host address: the cold count is written there
                             // host-visibly so the BCG break can spin on plain memory instead of a
                             // per-layer stream sync (the host resets it to -1 before each replay)
    int32_t* idx,            // [E]  out: logical -> slot map snapshot
    int32_t* needed) {       // [K]  out: 1 iff slot holds an expert needed this step
  if (blockIdx.x || threadIdx.x >= 32) return;
  const int lane = threadIdx.x;
  int step = 0;
  if (lane == 0) step = ++(*step_ctr);
  step = __shfl_sync(0xffffffff, step, 0);
  // pass 1 (warp-parallel): bump per-expert use count (LFU key) and resident-hit recency
  for (int i = lane; i < topk_n; i += 32) {
    const int e = topk[i];
    if (e < 0 || e >= E) continue;
    atomicAdd(&freq[e], 1);
    const int s = expert_slot[e];
    if (s >= 0) slot_lastuse[s] = step;
  }
  __syncwarp();
  // pass 2 (lane 0, serial): each miss is split by window membership (hot -> in-graph gather via a
  // tier-aware eviction, cold -> defer)
  if (lane == 0) {
    int nw = 0, nc = 0;
    for (int i = 0; i < topk_n; ++i) {
      const int e = topk[i];
      if (e < 0 || e >= E) continue;
      if (expert_slot[e] >= 0) continue;  // resident (or just assigned this step)
      const int hi = log2hot[e];
      if (hi < 0) {
        // Cold miss: not gatherable in-graph -> record logical id, no eviction, stays masked this replay.
        cold_log[nc] = e;
        ++nc;
        continue;
      }
      const int victim =
          pick_victim(topk, topk_n, K, lfu, slot_expert, slot_lastuse, freq, log2hot);
      if (victim < 0) continue;  // pool too small (should not happen: distinct <= K)
      const int old = slot_expert[victim];
      if (old >= 0) expert_slot[old] = -1;
      slot_expert[victim] = e;
      expert_slot[e] = victim;
      slot_lastuse[victim] = step;
      // windowed hit -> on-device gather from the pinned hot block
      src[nw] = hi;
      dst[nw] = victim;
      ++nw;
    }
    *n_out = nw;
    *cold_n = nc;
    if (doorbell != 0) {
      __threadfence_system();  // device writes above are visible before the host sees the count
      *reinterpret_cast<volatile int32_t*>(doorbell) = nc;
    }
  }
  __syncwarp();
  // map snapshot + needed[] scan (warp-parallel over E and K; inner topk scan is short)
  for (int e = lane; e < E; e += 32) idx[e] = expert_slot[e];
  for (int s = lane; s < K; s += 32) {
    const int se = slot_expert[s];
    int nd = 0;
    if (se >= 0) {
      for (int i = 0; i < topk_n; ++i) {
        if (topk[i] == se) {
          nd = 1;
          break;
        }
      }
    }
    needed[s] = nd;
  }
}

// Static fixed-wave decision (distinct active experts > K, e.g. prefill / batched decode). Expert e has a
// STATIC home: wave floor(e/K), slot e%K. For wave w this emits the page-in plan for the distinct in-wave
// experts present in topk (src=e, dst=e-w*K) and writes idx[e] = (e in [w*K, (w+1)*K)) ? e-w*K : -1. The
// caller runs ceil(E/K) waves; each active expert is served in exactly its wave, so summing the per-wave
// GEMM partials reconstructs the full MoE output (lossless). No eviction, no state mutation, no host sync.
__global__ void decide_wave_kernel(
    const int32_t* topk,
    int topk_n,
    int E,
    int K,
    int w,
    int32_t* src,
    int32_t* dst,
    int32_t* n_out,
    int32_t* idx) {
  if (blockIdx.x || threadIdx.x >= 32) return;
  const int lane = threadIdx.x;
  const int lo = w * K, hi = lo + K;
  for (int e = lane; e < E; e += 32) idx[e] = (e >= lo && e < hi) ? (e - lo) : -1;
  if (lane == 0) {
    int n = 0;
    for (int i = 0; i < topk_n; ++i) {
      const int e = topk[i];
      if (e < lo || e >= hi) continue;  // not this wave's group
      bool seen = false;
      for (int k = 0; k < n; ++k) {
        if (src[k] == e) {
          seen = true;
          break;
        }
      }
      if (!seen) {  // distinct in-wave hit -> its home slot
        src[n] = e;
        dst[n] = e - lo;
        ++n;
      }
    }
    *n_out = n;
  }
}

// Fused remap + weight mask: replaces the per-layer python chain
//   remap = idx[topk]; safe_ids = where(remap >= 0, remap, 0); masked_tw = where(remap >= 0, tw, 0)
// (a gather + 2x where + 2x zeros_like = 5 elementwise launches) with ONE capturable launch. Reads the
// LIVE idx map, so it can run after an in-graph staging break (BCG) and see the just-staged experts.
__global__ void remap_mask_kernel(
    const int32_t* topk,   // [T] flattened logical expert ids (negative = padding)
    int topk_n,
    int E,
    const int32_t* idx,    // [E] logical -> slot (-1 == not resident / masked)
    const float* tw,       // [T] routing weights (float32)
    int32_t* safe_ids,     // [T] out: slot id, masked -> 0 (slot-0 output x 0 = exact 0)
    float* masked_tw) {    // [T] out: routing weight, masked -> 0
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= topk_n) return;
  const int e = topk[i];
  const int s = (e >= 0 && e < E) ? idx[e] : -1;
  safe_ids[i] = s >= 0 ? s : 0;
  masked_tw[i] = s >= 0 ? tw[i] : 0.0f;
}

// Gather: copy n experts (src[i] -> dst[i]) from the pinned host store into the GPU slot pool, float4.
// The page-in count *n is read ON-DEVICE, so under CUDA-graph capture each replay moves exactly the
// experts the decide kernel chose this step (transfer_kv would move a fixed src_indices.numel() instead).
// ``store`` is the pinned host buffer addressed through its UVA device pointer; ``e16`` = per-expert
// bytes / 16. Copy-only — marlin int4 / bf16 rows travel packed; no dequant.
__global__ void gather_kernel(
    const float4* store, float4* slot, const int32_t* src, const int32_t* dst, const int32_t* n, long e16) {
  const long M = static_cast<long>(*n) * e16;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long j = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; j < M; j += stride) {
    const long s = j / e16, off = j % e16;
    slot[static_cast<long>(dst[s]) * e16 + off] = store[static_cast<long>(src[s]) * e16 + off];
  }
}

// Fused multi-tensor gather: one launch pages ALL of a layer's paged tensors (w13/w2 x qweight/scales/...)
// instead of one launch per tensor — for a marlin-int4 layer that is 6 launches -> 1 (hundreds of mostly-
// empty graph nodes per token at 48 layers). ``stores``/``slots`` are per-tensor base pointers (int64,
// resolved once at setup — UVA devptr for the pinned store, data_ptr for the GPU pool) and ``e16s`` the
// per-tensor expert-block sizes / 16. Blocks iterate the tensors serially and grid-stride within each;
// the copy is PCIe-bound, so a modest grid saturates it and empty plans retire cheaply.
__global__ void gather_multi_kernel(
    const int64_t* stores,   // [ntens] per-tensor pinned-store base (UVA device pointer)
    const int64_t* slots,    // [ntens] per-tensor GPU slot-pool base
    const int64_t* e16s,     // [ntens] per-tensor per-expert bytes / 16
    int ntens,
    const int32_t* src,
    const int32_t* dst,
    const int32_t* n) {
  const int cnt = *n;
  if (cnt == 0) return;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  const long tid = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  for (int t = 0; t < ntens; ++t) {
    const float4* store = reinterpret_cast<const float4*>(stores[t]);
    float4* slot = reinterpret_cast<float4*>(slots[t]);
    const long e16 = e16s[t];
    const long M = static_cast<long>(cnt) * e16;
    for (long j = tid; j < M; j += stride) {
      const long s = j / e16, off = j % e16;
      slot[static_cast<long>(dst[s]) * e16 + off] = store[static_cast<long>(src[s]) * e16 + off];
    }
  }
}

// Fused multi-tensor scatter: the refill's inverse of ``gather_multi`` — copy ``n`` staged rows
// (contiguous per tensor in a DEVICE staging area) into their (arbitrary) victim slots, for all of a
// layer's paged tensors, in ONE launch. Replaces per-tensor-per-expert micro-copies (4*n launches, two
// of which move <1 KB fp8 scale rows) with a single kernel; ``n`` is host-known at refill time.
__global__ void scatter_multi_kernel(
    const int64_t* stages,   // [ntens] per-tensor device staging base (n contiguous rows each)
    const int64_t* slots,    // [ntens] per-tensor GPU slot-pool base
    const int64_t* e16s,     // [ntens] per-tensor per-expert bytes / 16
    int ntens,
    const int32_t* dst,      // [>=n] destination slot per staged row
    int n) {
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  const long tid = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
  for (int t = 0; t < ntens; ++t) {
    const float4* stage = reinterpret_cast<const float4*>(stages[t]);
    float4* slot = reinterpret_cast<float4*>(slots[t]);
    const long e16 = e16s[t];
    const long M = static_cast<long>(n) * e16;
    for (long j = tid; j < M; j += stride) {
      const long s = j / e16, off = j % e16;
      slot[static_cast<long>(dst[s]) * e16 + off] = stage[s * e16 + off];
    }
  }
}

// ---- launchers -------------------------------------------------------------------------------------

void decide(
    tvm::ffi::TensorView topk,
    tvm::ffi::TensorView step_ctr,
    tvm::ffi::TensorView slot_expert,
    tvm::ffi::TensorView expert_slot,
    tvm::ffi::TensorView slot_lastuse,
    tvm::ffi::TensorView freq,
    int64_t lfu,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView n_out,
    tvm::ffi::TensorView idx) {
  using namespace host;

  // All operands are int32 CUDA tensors on the same device. Bind E to expert_slot and K to slot_expert,
  // then verify the rest against those symbolic sizes so a shape mismatch is caught here.
  SymbolicSize E = {"num_experts"}, K = {"num_slots"}, T = {"topk_n"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({E}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(expert_slot).verify(freq).verify(idx);
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(slot_expert).verify(slot_lastuse).verify(src).verify(dst);
  TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk);

  const int e = static_cast<int>(E.unwrap());
  const int k = static_cast<int>(K.unwrap());
  const int t = static_cast<int>(T.unwrap());
  const DLDevice device = device_.unwrap();

  LaunchKernel(1, 32, device)(
      decide_kernel,
      static_cast<const int32_t*>(topk.data_ptr()),
      t,
      e,
      k,
      static_cast<int>(lfu),
      static_cast<int32_t*>(step_ctr.data_ptr()),
      static_cast<int32_t*>(slot_expert.data_ptr()),
      static_cast<int32_t*>(expert_slot.data_ptr()),
      static_cast<int32_t*>(slot_lastuse.data_ptr()),
      static_cast<int32_t*>(freq.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(n_out.data_ptr()),
      static_cast<int32_t*>(idx.data_ptr()));
}

void decide_bounded(
    tvm::ffi::TensorView topk,
    int64_t lfu,
    tvm::ffi::TensorView log2hot,
    tvm::ffi::TensorView step_ctr,
    tvm::ffi::TensorView slot_expert,
    tvm::ffi::TensorView expert_slot,
    tvm::ffi::TensorView slot_lastuse,
    tvm::ffi::TensorView freq,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView n_out,
    tvm::ffi::TensorView cold_log,
    tvm::ffi::TensorView cold_n,
    int64_t doorbell,
    tvm::ffi::TensorView idx,
    tvm::ffi::TensorView needed) {
  using namespace host;

  // E bound to expert_slot, K to slot_expert; the per-expert maps (freq/idx/log2hot) are [E], the
  // per-slot ones (slot_lastuse/needed) and page-in plans (src/dst/cold_log) are [K], topk is [T].
  SymbolicSize E = {"num_experts"}, K = {"num_slots"}, T = {"topk_n"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({E}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(expert_slot).verify(freq).verify(idx).verify(log2hot);
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(slot_expert).verify(slot_lastuse).verify(src).verify(dst).verify(cold_log).verify(needed);
  TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk);

  const int e = static_cast<int>(E.unwrap());
  const int k = static_cast<int>(K.unwrap());
  const int t = static_cast<int>(T.unwrap());
  const DLDevice device = device_.unwrap();

  LaunchKernel(1, 32, device)(
      decide_bounded_kernel,
      static_cast<const int32_t*>(topk.data_ptr()),
      t,
      e,
      k,
      static_cast<int>(lfu),
      static_cast<const int32_t*>(log2hot.data_ptr()),
      static_cast<int32_t*>(step_ctr.data_ptr()),
      static_cast<int32_t*>(slot_expert.data_ptr()),
      static_cast<int32_t*>(expert_slot.data_ptr()),
      static_cast<int32_t*>(slot_lastuse.data_ptr()),
      static_cast<int32_t*>(freq.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(n_out.data_ptr()),
      static_cast<int32_t*>(cold_log.data_ptr()),
      static_cast<int32_t*>(cold_n.data_ptr()),
      doorbell,
      static_cast<int32_t*>(idx.data_ptr()),
      static_cast<int32_t*>(needed.data_ptr()));
}

void decide_wave(
    tvm::ffi::TensorView topk,
    int64_t num_experts,
    int64_t num_slots,
    int64_t wave,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView n_out,
    tvm::ffi::TensorView idx) {
  using namespace host;

  SymbolicSize K = {"num_slots"}, T = {"topk_n"}, Eidx = {"num_experts"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(src).verify(dst);
  TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk);
  TensorMatcher({Eidx}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(idx);

  const int t = static_cast<int>(T.unwrap());
  const DLDevice device = device_.unwrap();

  LaunchKernel(1, 32, device)(
      decide_wave_kernel,
      static_cast<const int32_t*>(topk.data_ptr()),
      t,
      static_cast<int>(num_experts),
      static_cast<int>(num_slots),
      static_cast<int>(wave),
      static_cast<int32_t*>(src.data_ptr()),
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(n_out.data_ptr()),
      static_cast<int32_t*>(idx.data_ptr()));
}

// Resolve the UVA device pointer of a pinned host tensor, once at setup (NOT inside the captured
// region). Returned as int64 and passed back to ``gather`` so no host CUDA call happens during replay.
int64_t host_devptr(tvm::ffi::TensorView pinned) {
  void* d = nullptr;
  cudaError_t e = cudaHostGetDevicePointer(&d, pinned.data_ptr(), 0);
  host::RuntimeCheck(e == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(e));
  return reinterpret_cast<int64_t>(d);
}

void gather(
    int64_t store_devptr,
    tvm::ffi::TensorView slot,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView n_out,
    int64_t item_bytes) {
  using namespace host;

  SymbolicSize Nsrc = {"n_src"}, One = {"one"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({Nsrc}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(src).verify(dst);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(n_out);
  const DLDevice device = device_.unwrap();
  RuntimeCheck(
      item_bytes % 16 == 0,
      "paged_experts gather needs 16-byte-aligned per-expert blocks (float4); got ",
      item_bytes);

  LaunchKernel(2048, 256, device)(
      gather_kernel,
      reinterpret_cast<const float4*>(store_devptr),
      reinterpret_cast<float4*>(slot.data_ptr()),
      static_cast<const int32_t*>(src.data_ptr()),
      static_cast<const int32_t*>(dst.data_ptr()),
      static_cast<const int32_t*>(n_out.data_ptr()),
      static_cast<long>(item_bytes / 16));
}

void gather_multi(
    tvm::ffi::TensorView stores,  // [ntens] int64 CUDA: per-tensor pinned-store UVA base pointers
    tvm::ffi::TensorView slots,   // [ntens] int64 CUDA: per-tensor GPU slot-pool base pointers
    tvm::ffi::TensorView e16s,    // [ntens] int64 CUDA: per-tensor per-expert bytes / 16
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView n_out) {
  using namespace host;

  SymbolicSize Nt = {"n_tensors"}, Nsrc = {"n_src"}, One = {"one"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({Nt}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(stores).verify(slots).verify(e16s);
  TensorMatcher({Nsrc}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(src).verify(dst);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(n_out);
  const int nt = static_cast<int>(Nt.unwrap());
  const DLDevice device = device_.unwrap();

  // The copy is PCIe-bound; 256x256 float4 streams saturate the link while keeping the empty-plan
  // early-exit cheap (the per-tensor 16-byte alignment is validated at setup, python side).
  LaunchKernel(256, 256, device)(
      gather_multi_kernel,
      static_cast<const int64_t*>(stores.data_ptr()),
      static_cast<const int64_t*>(slots.data_ptr()),
      static_cast<const int64_t*>(e16s.data_ptr()),
      nt,
      static_cast<const int32_t*>(src.data_ptr()),
      static_cast<const int32_t*>(dst.data_ptr()),
      static_cast<const int32_t*>(n_out.data_ptr()));
}

void scatter_multi(
    tvm::ffi::TensorView stages,  // [ntens] int64 CUDA: per-tensor device staging base pointers
    tvm::ffi::TensorView slots,   // [ntens] int64 CUDA: per-tensor GPU slot-pool base pointers
    tvm::ffi::TensorView e16s,    // [ntens] int64 CUDA: per-tensor per-expert bytes / 16
    tvm::ffi::TensorView dst,     // [>=n] int32 CUDA: destination slots
    int64_t n) {
  using namespace host;

  SymbolicSize Nt = {"n_tensors"}, Nd = {"n_dst"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({Nt}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(stages).verify(slots).verify(e16s);
  TensorMatcher({Nd}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(dst);
  const int nt = static_cast<int>(Nt.unwrap());
  const DLDevice device = device_.unwrap();

  LaunchKernel(256, 256, device)(
      scatter_multi_kernel,
      static_cast<const int64_t*>(stages.data_ptr()),
      static_cast<const int64_t*>(slots.data_ptr()),
      static_cast<const int64_t*>(e16s.data_ptr()),
      nt,
      static_cast<const int32_t*>(dst.data_ptr()),
      static_cast<int>(n));
}

void remap_mask(
    tvm::ffi::TensorView topk,
    tvm::ffi::TensorView idx,
    tvm::ffi::TensorView tw,
    tvm::ffi::TensorView safe_ids,
    tvm::ffi::TensorView masked_tw) {
  using namespace host;

  SymbolicSize E = {"num_experts"}, T = {"topk_n"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({E}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(idx);
  TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk).verify(safe_ids);
  TensorMatcher({T}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(tw).verify(masked_tw);
  const int e = static_cast<int>(E.unwrap());
  const int t = static_cast<int>(T.unwrap());
  const DLDevice device = device_.unwrap();

  LaunchKernel((t + 127) / 128, 128, device)(
      remap_mask_kernel,
      static_cast<const int32_t*>(topk.data_ptr()),
      t,
      e,
      static_cast<const int32_t*>(idx.data_ptr()),
      static_cast<const float*>(tw.data_ptr()),
      static_cast<int32_t*>(safe_ids.data_ptr()),
      static_cast<float*>(masked_tw.data_ptr()));
}

}  // namespace
