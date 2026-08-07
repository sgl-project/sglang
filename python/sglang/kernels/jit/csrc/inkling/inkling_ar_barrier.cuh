// Cross-GPU barrier primitives shared by the Inkling custom all-reduce kernels
// (inkling_all_reduce.cuh) and the fused AR+sconv+norm decode kernel
// (inkling_ar_fused_decode.cuh). Two designs are provided: a single-leader
// grid barrier and a per-block variant.
//
// Resources (see inkling_all_reduce.py):
//   * flags: DEDICATED symmetric uint32 buffer, zero-initialized at setup:
//     kNumGPU single-leader slots (one per peer), then
//     kNumGPU * kMaxBarrierBlocks per-(writer, block) slots.
//   * state: device-LOCAL uint32 buffer: [arrival0, arrival1, release0,
//     release1, xepoch] padded to kLeaderStateWords, then kMaxBarrierBlocks
//     per-block epochs. All epochs are monotonic (mod 2^32, wrap-safe compares)
//     and advance under CUDA-graph replay, so flags never go stale.

#pragma once

#include <cstdint>

namespace inkling_ar {

constexpr uint32_t kLeaderStateWords = 8;
constexpr uint32_t kMaxBarrierBlocks = 256;

// Grid-level system barrier across all ranks. Two levels:
//   1. Grid: every block arrives at a self-resetting device counter
//      (atomicInc wraps at gridDim.x-1); the last arriver is the leader.
//   2. Cross-GPU: ONLY the leader block does the peer release/acquire
//      signal/wait, so that O(1) cost is independent of gridDim.x (the reason
//      the old per-block barrier was slow for many-block launches).
// The leader then bumps a release counter; followers spin on it (device scope).
//
// `st` is a device-local uint32 state buffer: [arrival0, arrival1, release0,
// release1, xepoch]. idx 0/1 selects the entry/exit instances (distinct grid
// counters so the two barriers in one kernel don't collide). xepoch is a single
// monotonic cross-GPU epoch (entry uses e, exit uses e+1) -- consistent across
// ranks (SPMD) and advancing under CUDA-graph replay, so flags never go stale.
// s_prev is read BEFORE arriving, and the leader (last arriver) bumps release
// only after all blocks arrived, so no follower can miss the bump (no deadlock).
template <uint32_t kNumGPU>
__device__ __forceinline__ void grid_system_barrier(
    uint32_t* __restrict__ st, void* const* __restrict__ flag_ptrs, uint32_t rank, uint32_t idx, bool publish_writes) {
  // publish_writes=true (EXIT barriers): every CTA flushes its just-written
  // reduced/broadcast slices to SYSTEM scope BEFORE it signals arrival, so the
  // single leader's `st.release.sys` publishes ALL blocks' stores rather than
  // only the leader thread's own. Without this, a multi-block launch (the tuned
  // v2/v3 configs) lets a peer leave the exit barrier and read a slice a
  // non-leader CTA wrote but never system-published. ONE fence per CTA suffices:
  // the __syncthreads below orders every thread's stores before thread 0's
  // fence (CTA-scope happens-before), and `fence.sys + relaxed arrival` is a
  // release pattern, so the arrival publishes the whole CTA's stores. ENTRY
  // barriers pass false: the data they gate on was written by a prior kernel
  // and is already uniformly visible, which the leader's release then promotes
  // for free. (The solo path needs no fence either way: its st.release.sys
  // signals below are themselves release ops ordered after the __syncthreads.)
  uint32_t* xepoch = st + 4;
  __shared__ uint32_t s_e;
  __shared__ uint32_t s_prev;
  __shared__ int s_leader;
  const bool solo = (gridDim.x == 1u);  // token=1 etc.: the sole block IS the grid
  __syncthreads();
  if (threadIdx.x == 0) {
    if (solo) {
      s_leader = 1;  // skip the grid arrival/release bookkeeping entirely
    } else {
      if (publish_writes) __threadfence_system();               // release pattern with the arrive below
      s_prev = *static_cast<volatile uint32_t*>(st + 2 + idx);  // pre-barrier release
      // Self-resetting arrive (atomicInc semantics: wrap at gridDim.x-1).
      // acq_rel: the release side pairs with the fence above (publishing this
      // CTA's stores); the acquire side lets the last arriver (leader) inherit
      // every earlier CTA's release pattern, so its st.release.sys to the peers
      // covers the whole grid's writes.
      uint32_t old;
      asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                   : "=r"(old)
                   : "l"(st + idx), "r"(gridDim.x - 1u)
                   : "memory");
      s_leader = (old == gridDim.x - 1u) ? 1 : 0;
    }
  }
  __syncthreads();
  if (s_leader) {
    if (threadIdx.x == 0) {
      const uint32_t e = *xepoch + 1u;
      *xepoch = e;
      s_e = e;
    }
    __syncthreads();
    const uint32_t e = s_e;
    // Cross-GPU arrive+wait with release/acquire at system scope. The release
    // store publishes THIS (leader) thread's system-visible writes and the
    // acquire spin makes the peer's visible -- far cheaper than a full
    // threadfence_system here. Data written by OTHER (non-leader) CTAs is made
    // system-visible by the publish_writes=true fence they each ran before
    // arriving (see top), so the leader's single release covers the whole grid.
    if (threadIdx.x < kNumGPU) {
      const uint32_t peer = threadIdx.x;
      uint32_t* remote = static_cast<uint32_t*>(flag_ptrs[peer]) + rank;
      asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(remote), "r"(e) : "memory");
      uint32_t* mine = static_cast<uint32_t*>(flag_ptrs[rank]) + peer;
      uint32_t got;
      do {
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(got) : "l"(mine) : "memory");
      } while (static_cast<int32_t>(got - e) < 0);  // wrap-safe: epoch is mod-2^32
    }
    __syncthreads();
    if (!solo && threadIdx.x == 0) {
      // Release-ordered bump: pairs with the followers' ld.acquire.gpu so the
      // leader's acquired peer state (and its xepoch store above) is visible to
      // them -- a relaxed atomicAdd would leave that handoff formally unordered.
      asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(st + 2 + idx), "r"(1u) : "memory");
    }
  } else {
    if (threadIdx.x == 0) {
      // `release` is this rank's LOCAL counter -> device-scope acquire suffices.
      uint32_t* rel = st + 2 + idx;
      uint32_t got;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(got) : "l"(rel) : "memory");
      } while (static_cast<int32_t>(got - s_prev) <= 0);  // wrap-safe
    }
    __syncthreads();
  }
}

// Device-LOCAL grid sync (no cross-GPU traffic): all blocks arrive at a
// self-resetting counter (state word 5), the last arriver bumps a release
// counter (word 6), followers spin on it -- the grid level of
// grid_system_barrier without the peer handshake. Words 5/6 are spare in the
// kLeaderStateWords block. Requires all blocks co-resident (the launch cap the
// fused kernels already apply). Used by the two-phase {AR + scattered sconv}
// kernel to publish its local scratch between the reduce and conv phases.
__device__ __forceinline__ void grid_local_sync(uint32_t* __restrict__ st) {
  __syncthreads();
  if (gridDim.x > 1u) {
    if (threadIdx.x == 0) {
      uint32_t* arrive = st + 5;
      uint32_t* release = st + 6;
      const uint32_t prev = *static_cast<volatile uint32_t*>(release);
      uint32_t old;
      asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                   : "=r"(old)
                   : "l"(arrive), "r"(gridDim.x - 1u)
                   : "memory");
      if (old == gridDim.x - 1u) {
        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(release), "r"(1u) : "memory");
      } else {
        uint32_t got;
        do {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(got) : "l"(release) : "memory");
        } while (static_cast<int32_t>(got - prev) <= 0);  // wrap-safe
      }
    }
    __syncthreads();
  }
}

// Per-block cross-GPU barrier (no grid funnel): block b handshakes ONLY with
// block b on each peer -- one NVLink round trip per block, all blocks in
// parallel, no arrival/release atomics and no leader serialization. Valid
// whenever the consumer phase reads exactly the ranges its blockIdx-matched
// producers wrote (true for the push one-shot: its push and reduce loops use
// the same grid-stride mapping, and every rank launches the same grid). The
// signal is a release store, which covers the CTA's prior (multicast) stores
// via the preceding __syncthreads -- no explicit fence needed. Epochs live in
// per-block device-local slots (monotonic across launches and CUDA-graph
// replays, like xepoch).
template <uint32_t kNumGPU>
__device__ __forceinline__ void
block_system_barrier(uint32_t* __restrict__ st, void* const* __restrict__ flag_ptrs, uint32_t rank) {
  __shared__ uint32_t s_e;
  __syncthreads();  // CTA stores done before the release signals below
  if (threadIdx.x == 0) {
    uint32_t* epoch = st + kLeaderStateWords + blockIdx.x;
    const uint32_t e = *epoch + 1u;
    *epoch = e;
    s_e = e;
  }
  __syncthreads();
  const uint32_t e = s_e;
  if (threadIdx.x < kNumGPU) {
    const uint32_t peer = threadIdx.x;
    uint32_t* remote = static_cast<uint32_t*>(flag_ptrs[peer]) + kNumGPU + rank * kMaxBarrierBlocks + blockIdx.x;
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(remote), "r"(e) : "memory");
    uint32_t* mine = static_cast<uint32_t*>(flag_ptrs[rank]) + kNumGPU + peer * kMaxBarrierBlocks + blockIdx.x;
    uint32_t got;
    do {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(got) : "l"(mine) : "memory");
    } while (static_cast<int32_t>(got - e) < 0);  // wrap-safe
  }
  __syncthreads();
}

}  // namespace inkling_ar
