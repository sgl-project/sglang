// ring_simple, CustomAllReduceV2's ring all-reduce for ROCm gfx1250 (MI450): each
// chunk goes to the next rank only, through a fixed-size FIFO behind 64-bit
// counters, in 2 * (world_size - 1) hops per loop. It runs over a ring plane of
// its own (`RingPlaneObj` below), not over the shared communicator.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#ifndef USE_ROCM
#error "ring_simple is ROCm only"
#endif

namespace sglang {

namespace ring_simple {

/// One ring's state on one rank, padded to a cache line. All counts are 64-bit
/// so no comparison ever wraps.
struct alignas(128) RingSemaphore {
  uint64_t drained;  // packets the successor has drained from this rank; only the successor writes it
  uint64_t packet;   // packets this rank's ring has sent (and received); rank-local
  uint64_t ready;    // packets the predecessor has published here; only the predecessor writes it
};
static_assert(sizeof(RingSemaphore) == 128, "must match _SEMAPHORE_BYTES in ring_simple_all_reduce.py");

/// Kernel-facing view of a ring plane from one rank: its own receive FIFOs
/// (the predecessor pushes into them), the successor's FIFOs (this rank pushes
/// into them), and the semaphore rows of this rank and both neighbours.
/// `credits` (local) holds what the neighbours report; `prev_credits` is where
/// this rank reports what it drained, `next_credits` what it published.
struct RingWorkSpace {
  uint8_t* recv;
  uint8_t* send;
  RingSemaphore* credits;
  RingSemaphore* prev_credits;
  RingSemaphore* next_credits;
  uint32_t slot_bytes;
  uint32_t num_slots;
};

/// A 64-bit sequence flag: payload the writer stored before the flag is
/// visible once this load sees it.
SGL_DEVICE uint64_t load_acquire_sys(const uint64_t* ptr) {
  return __hip_atomic_load(const_cast<uint64_t*>(ptr), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

/// Publishes this wave's system-scope stores (`st_relaxed_16B`) before a later
/// flag store, without the cache writeback `fence_release_sys` adds for plain
/// stores: those stores bypass the cache, so completing them is enough. Only
/// gfx12 counts stores apart from loads; elsewhere this is the full fence.
SGL_DEVICE void wait_stores_sys() {
#if defined(__gfx1250__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#else
  device::ptx::fence_release_sys();
#endif
}

struct Params {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t num_vecs;
  uint32_t rank;
  RingWorkSpace ws;
};

/// Threads per ring_simple channel, and 16-byte packs each thread keeps in
/// flight per pass.
inline constexpr uint32_t kRingSimpleThreads = 256;
inline constexpr uint32_t kRingSimpleUnroll = 32;

/// What one ring hop does with each vector: receive it, add this rank's
/// input, store it to the output, send it on.
template <bool kRecv_, bool kReduce_, bool kStoreOut_, bool kSend_>
struct RingHop {
  static constexpr bool kRecv = kRecv_, kReduce = kReduce_, kStoreOut = kStoreOut_, kSend = kSend_;
};

/// The ring all-reduce: one channel per block,
/// a FIFO of `num_slots` chunk-sized slices per channel in the next rank's
/// memory, and per channel three 64-bit counters:
/// `ready` (slices the predecessor published here), `drained` (slices the
/// successor consumed) and `packet` (this rank's own count). A slice is plain
/// data, published once all its stores have completed, so the receiver polls
/// one counter and reads the payload once. FIFO payload moves as device-scope
/// loads and batched system-scope stores; the input and output take plain
/// accesses.
///
/// A channel is one block of 8 waves, at most 2 per SIMD; `amdgpu_waves_per_eu`
/// says so, or the compiler budgets registers for more.
template <typename T, uint32_t kWorldSize, bool kUsePDL>
__global__ __launch_bounds__(kRingSimpleThreads, 1) __attribute__((
    amdgpu_waves_per_eu(1, 2))) void all_reduce_ring_simple_kernel(const __grid_constant__ Params params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  constexpr uint32_t kNumPackets = 2 * (kWorldSize - 1);
  constexpr uint32_t kWarpVecs = kWarpThreads * kRingSimpleUnroll;
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  const auto& ws = params.ws;
  const auto r = params.rank;
  const auto num_vecs = params.num_vecs;
  const auto ch = blockIdx.x;
  const auto num_channels = gridDim.x;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto num_warps = blockDim.x / kWarpThreads;
  const auto slot_mask = ws.num_slots - 1;
  const auto slot_vecs = ws.slot_bytes / 16;
  const auto fifo_offset = ch * ws.num_slots * ws.slot_bytes;
  const auto recv_fifo = ws.recv + fifo_offset;
  const auto send_fifo = ws.send + fifo_offset;
  auto& sem = ws.credits[ch];
  auto& prev_sem = ws.prev_credits[ch];
  auto& next_sem = ws.next_credits[ch];
  const auto slot_of = [&](uint8_t* fifo, uint64_t p) { return fifo + (p & slot_mask) * ws.slot_bytes; };

  PDLWaitPrimary<kUsePDL>();
  uint64_t packet = sem.packet;
  // polled by thread 0 only; the barrier after each wait publishes the result
  uint64_t ready = 0, drained = 0;
  const auto wait_until = [&](uint64_t& seen, const uint64_t* ptr, uint64_t target) {
    while (seen < target) {
      seen = load_acquire_sys(ptr);
      if (seen < target) __builtin_amdgcn_s_sleep(1);
    }
  };
  // Packet p may be received once the predecessor published it, and sent
  // once the successor drained the packet that last held its slot.
  const auto wait = [&](bool recv, uint64_t recv_p, bool send, uint64_t send_p) {
    if (threadIdx.x == 0) {
      if (recv) wait_until(ready, &sem.ready, recv_p + 1);
      if (send && send_p >= ws.num_slots) wait_until(drained, &sem.drained, send_p + 1 - ws.num_slots);
    }
    __syncthreads();
  };
  // Once every wave's stores (and so the loads feeding them) completed: tell
  // the predecessor `recv_p` is drained and the successor `send_p` is ready.
  const auto post = [&](bool recv, uint64_t recv_p, bool send, uint64_t send_p) {
    wait_stores_sys();
    __syncthreads();
    if (threadIdx.x == 0) {
      if (recv) ptx::store_relaxed_sys(&prev_sem.drained, recv_p + 1);
      if (send) ptx::store_relaxed_sys(&next_sem.ready, send_p + 1);
    }
  };
  // One slice: warps take contiguous
  // hunks of kWarpVecs, each lane a pack per kWarpThreads vectors, so a lane
  // has kRingSimpleUnroll loads per source in flight. Only whole hunks take the
  // unrolled path: a bounds check per pack splits it into branches the
  // compiler fences with a full load wait each, serializing the loads.
  const auto copy_packs =
      [&](auto unroll, auto mode, const uint8_t* recv_slot, uint8_t* send_slot, uint32_t start, uint32_t j) {
        constexpr uint32_t kUnroll = decltype(unroll)::value;
        constexpr uint32_t kStride = kUnroll == 1 ? 1 : kWarpThreads;
        constexpr bool kRecv = decltype(mode)::kRecv, kReduce = decltype(mode)::kReduce;
        constexpr bool kStoreOut = decltype(mode)::kStoreOut, kSend = decltype(mode)::kSend;
        // Per-lane base pointers with constant pack offsets, which fold into the
        // instructions' immediate offsets: an index recomputed per pack reuses
        // its VGPR, and each store then waits (s_wait_xcnt) for the last to read it.
        const auto in = static_cast<const vec_t*>(params.input) + start + j;
        const auto out = static_cast<vec_t*>(params.output) + start + j;
        const auto send_at = reinterpret_cast<vec_t*>(send_slot) + j;
        vec_t vecs[kUnroll];
        vec_t inputs[kUnroll];
    // every pack of the local input, then every pack of the FIFO
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          if constexpr (!kRecv || kReduce) inputs[u] = in[u * kStride];
        }
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          if constexpr (kRecv) ptx::ld_relaxed_16B(vecs[u], recv_slot, j + u * kStride);
        }
        // keep every load ahead of the first use: left alone, the scheduler pulls
        // each reduction up behind its two loads and waits there, to save registers
        __builtin_amdgcn_sched_barrier(0);
    // results land in `vecs`; then the stores, one destination at a time
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          if constexpr (kRecv && kReduce) {
            vec_t pair[2];
            pair[0] = vecs[u];
            pair[1] = inputs[u];
            vecs[u] = reduce_vec(pair);
          } else if constexpr (!kRecv) {
            vecs[u] = inputs[u];
          }
        }
        if constexpr (kStoreOut) {
#pragma unroll
          for (uint32_t u = 0; u < kUnroll; ++u)
            out[u * kStride] = vecs[u];
        }
        if constexpr (kSend) {
          if constexpr (kUnroll % 8 == 0) {
            ptx::st_relaxed_run_16B<kStride, kUnroll>(vecs, send_at);
          } else {
#pragma unroll
            for (uint32_t u = 0; u < kUnroll; ++u)
              ptx::st_relaxed_16B(vecs[u], send_at, u * kStride);
          }
        }
      };
  const auto copy = [&](auto mode, const uint8_t* recv_slot, uint8_t* send_slot, uint32_t start, uint32_t n) {
    const auto full = n / kWarpVecs * kWarpVecs;
    for (uint32_t base = warp_id * kWarpVecs; base < full; base += num_warps * kWarpVecs) {
      copy_packs(
          std::integral_constant<uint32_t, kRingSimpleUnroll>{}, mode, recv_slot, send_slot, start, base + lane_id);
    }
    for (uint32_t j = full + threadIdx.x; j < n; j += blockDim.x) {
      copy_packs(std::integral_constant<uint32_t, 1>{}, mode, recv_slot, send_slot, start, j);
    }
  };

  // each channel takes a contiguous share of the tensor, then loops over it in
  // kWorldSize chunks of at most one slot. Shares are whole hunks per chunk:
  // a partial hunk runs as passes of the one-pack path, each costing about as
  // much as a full pass, so the last channels may get less or nothing instead.
  constexpr uint32_t kShareVecs = kWorldSize * kWarpVecs;
  const auto channel_vecs = div_ceil(div_ceil(num_vecs, num_channels), kShareVecs) * kShareVecs;
  const auto channel_start = min(ch * channel_vecs, num_vecs);
  const auto channel_len = min(channel_vecs, num_vecs - channel_start);
  const auto loop_vecs = kWorldSize * slot_vecs;
  for (uint32_t base = 0; base < channel_len; base += loop_vecs) {
    const auto len = min(loop_vecs, channel_len - base);
    const auto chunk_vecs = div_ceil(len, kWorldSize);
    const auto chunk_start = [&](uint32_t c) { return channel_start + base + c * chunk_vecs; };
    const auto chunk_len = [&](uint32_t c) { return min(chunk_vecs, len - min(len, c * chunk_vecs)); };

    // first hop: this rank's own chunk
    wait(false, 0, true, packet);
    copy(RingHop<false, false, false, true>{}, nullptr, slot_of(send_fifo, packet), chunk_start(r), chunk_len(r));
    post(false, 0, true, packet);

    // hop q receives chunk r - 1 - q: a partial for q < kWorldSize - 1, which
    // this rank completes at q = kWorldSize - 2, then a reduced chunk
    for (uint32_t q = 0; q < kNumPackets; ++q, ++packet) {
      const bool forward = q + 1 < kNumPackets;
      const auto c = (r + 2 * kWorldSize - 1 - q) % kWorldSize;
      const auto recv_slot = slot_of(recv_fifo, packet);
      const auto send_slot = slot_of(send_fifo, packet + 1);
      const auto start = chunk_start(c);
      const auto n = chunk_len(c);
      wait(true, packet, forward, packet + 1);
      if (q + 2 < kWorldSize) {
        copy(RingHop<true, true, false, true>{}, recv_slot, send_slot, start, n);
      } else if (q + 2 == kWorldSize) {
        copy(RingHop<true, true, true, true>{}, recv_slot, send_slot, start, n);
      } else if (forward) {
        copy(RingHop<true, false, true, true>{}, recv_slot, send_slot, start, n);
      } else {
        copy(RingHop<true, false, true, false>{}, recv_slot, send_slot, start, n);
      }
      post(true, packet, forward, packet + 1);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  if (threadIdx.x == 0) sem.packet = packet;
}

/// 16-byte packs a hier ring thread keeps in flight per source. Its reducing
/// steps load three sources, so the flat ring's 32 do not fit in registers:
/// 16 runs 5-15% faster than 32 or 8 from 64 MB at TP4 and TP8.
inline constexpr uint32_t kHierRingUnroll = 16;
/// Warps of a hier ring block that run the cross step; the rest run the pair
/// copies of the neighbouring steps at the same time. Run in sequence by every
/// warp, the pair copies cost TP8 a fifth of its time at 256 MB; 5 of 8 warps
/// beat 4 and 6 by up to 4% from 64 MB.
inline constexpr uint32_t kHierCrossWarps = 5;
static_assert(0 < kHierCrossWarps && kHierCrossWarps < kRingSimpleThreads / device::kWarpThreads);

/// One hier ring's state on one rank, padded to a cache line. Each counter has
/// one writer: the cross-ring neighbour named, the DPX peer, or this rank.
struct alignas(128) HierRingSemaphore {
  uint64_t x_drained;    // cross packets the cross successor drained from this rank
  uint64_t x_ready;      // cross packets the cross predecessor published here
  uint64_t in_drained;   // pair-in packets the peer drained from this rank
  uint64_t in_ready;     // pair-in packets the peer published here
  uint64_t out_drained;  // pair-out packets the peer drained from this rank
  uint64_t out_ready;    // pair-out packets the peer published here
  uint64_t x_packet;     // cross packets this rank's ring has sent (and received); rank-local
  uint64_t in_packet;    // pair-in packets exchanged with the peer, each way; rank-local
  uint64_t out_packet;   // pair-out packets exchanged with the peer, each way; rank-local
};
static_assert(sizeof(HierRingSemaphore) == 128, "must match _SEMAPHORE_BYTES in ring_simple_all_reduce.py");

/// The three FIFOs of a hier ring channel, in this order in each rank's row.
enum HierFifo : uint32_t { kCrossFifo = 0, kPairInFifo = 1, kPairOutFifo = 2, kNumHierFifos = 3 };

/// Kernel-facing view of a hier ring plane from one rank: its own FIFOs and
/// semaphores, the cross successor's and the DPX peer's FIFOs, and the
/// semaphores of the cross neighbours and the peer.
struct HierRingWorkSpace {
  uint8_t* own;     // [num_rings][kNumHierFifos][num_slots][slot_bytes]
  uint8_t* x_next;  // the cross successor's rows
  uint8_t* peer;    // the DPX peer's rows
  HierRingSemaphore* credits;
  HierRingSemaphore* x_prev_credits;
  HierRingSemaphore* x_next_credits;
  HierRingSemaphore* peer_credits;
  uint32_t slot_bytes;
  uint32_t num_slots;
};

struct HierParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t num_vecs;
  uint32_t half;  // this rank's index in its DPX pair: the half its cross ring reduces
  uint32_t soc;   // this rank's SoC's position in the cross ring
  HierRingWorkSpace ws;
};

/// What one hier step does with each vector: which of the cross FIFO, the
/// pair FIFO and the input it sums, and where the result goes.
template <bool kRecvX_, bool kRecvP_, bool kInput_, bool kStoreOut_, bool kSendX_, bool kSendP_>
struct HierHop {
  static constexpr bool kRecvX = kRecvX_, kRecvP = kRecvP_, kInput = kInput_;
  static constexpr bool kStoreOut = kStoreOut_, kSendX = kSendX_, kSendP = kSendP_;
};

/// ring_simple over DPX pairs: each SoC's two partitions split every chunk in
/// halves, and half h runs a ring over partition h of every SoC, so each SoC
/// sends 2 * (m - 1) / m of the tensor across the fabric (m SoCs) instead of
/// 2 * (2m - 1) / 2m. The pair exchange rides in the same steps: while the ring
/// reduces, each rank pushes the peer the input chunk the peer reduces next
/// (pair-in FIFO), and every reduced chunk also goes to the peer (pair-out
/// FIFO). One loop is 2m - 1 steps; at step s the rank, its peer and every
/// rank of both rings work on the same chunk index, so each step waits only
/// on steps before it and on the peer's same step.
template <typename T, uint32_t kCrossSize, bool kUsePDL>
__global__ __launch_bounds__(kRingSimpleThreads, 1) __attribute__((
    amdgpu_waves_per_eu(1, 2))) void all_reduce_hier_ring_kernel(const __grid_constant__ HierParams params) {
  using namespace device;
  static_assert(kCrossSize >= 2, "a hier ring needs at least two SoCs");
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  constexpr uint32_t kNumCross = 2 * (kCrossSize - 1);  // cross packets per loop
  constexpr uint32_t kNumSteps = kNumCross + 1;
  constexpr uint32_t kWarpVecs = kWarpThreads * kHierRingUnroll;
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  const auto& ws = params.ws;
  const auto h = params.half;
  const auto k = params.soc;
  const auto num_vecs = params.num_vecs;
  const auto ch = blockIdx.x;
  const auto num_channels = gridDim.x;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto num_warps = blockDim.x / kWarpThreads;
  const auto slot_mask = ws.num_slots - 1;
  const auto slot_vecs = ws.slot_bytes / 16;
  const auto fifo_bytes = ws.num_slots * ws.slot_bytes;
  const auto row = ch * kNumHierFifos * fifo_bytes;
  const auto slot_of = [&](uint8_t* rows, HierFifo fifo, uint64_t p) {
    return rows + row + fifo * fifo_bytes + (p & slot_mask) * ws.slot_bytes;
  };
  auto& sem = ws.credits[ch];
  auto& x_prev_sem = ws.x_prev_credits[ch];
  auto& x_next_sem = ws.x_next_credits[ch];
  auto& peer_sem = ws.peer_credits[ch];

  PDLWaitPrimary<kUsePDL>();
  uint64_t x_packet = sem.x_packet;
  uint64_t in_packet = sem.in_packet;
  uint64_t out_packet = sem.out_packet;
  // polled by thread 0 only; the barrier after each wait publishes the result
  uint64_t x_ready = 0, x_drained = 0, in_ready = 0, in_drained = 0, out_ready = 0, out_drained = 0;
  const auto wait_until = [&](uint64_t& seen, const uint64_t* ptr, uint64_t target) {
    while (seen < target) {
      seen = load_acquire_sys(ptr);
      if (seen < target) __builtin_amdgcn_s_sleep(1);
    }
  };
  // packet p of a FIFO may be sent once the receiver drained the packet that
  // last held its slot
  const auto free_target = [&](uint64_t p) { return p >= ws.num_slots ? p + 1 - ws.num_slots : 0; };

  const auto copy_packs = [&](auto unroll,
                              auto mode,
                              const uint8_t* x_recv,
                              const uint8_t* p_recv,
                              uint8_t* x_send,
                              uint8_t* p_send,
                              uint32_t in_start,
                              uint32_t out_start,
                              uint32_t j) {
    constexpr uint32_t kUnroll = decltype(unroll)::value;
    constexpr uint32_t kStride = kUnroll == 1 ? 1 : kWarpThreads;
    using M = decltype(mode);
    constexpr uint32_t kTerms = M::kRecvX + M::kRecvP + M::kInput;
    static_assert(kTerms > 0);
    const auto in = static_cast<const vec_t*>(params.input) + in_start + j;
    const auto out = static_cast<vec_t*>(params.output) + out_start + j;
    vec_t terms[kUnroll][kTerms];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if constexpr (M::kInput) terms[u][kTerms - 1] = in[u * kStride];
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if constexpr (M::kRecvX) ptx::ld_relaxed_16B(terms[u][0], x_recv, j + u * kStride);
      if constexpr (M::kRecvP) ptx::ld_relaxed_16B(terms[u][M::kRecvX], p_recv, j + u * kStride);
    }
    __builtin_amdgcn_sched_barrier(0);
    vec_t vecs[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if constexpr (kTerms == 1) {
        vecs[u] = terms[u][0];
      } else {
        vecs[u] = reduce_vec(terms[u]);
      }
    }
    if constexpr (M::kStoreOut) {
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u)
        out[u * kStride] = vecs[u];
    }
    const auto store_run = [&](uint8_t* slot) {
      const auto at = reinterpret_cast<vec_t*>(slot) + j;
      if constexpr (kUnroll % 8 == 0) {
        ptx::st_relaxed_run_16B<kStride, kUnroll>(vecs, at);
      } else {
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u)
          ptx::st_relaxed_16B(vecs[u], at, u * kStride);
      }
    };
    if constexpr (M::kSendX) store_run(x_send);
    if constexpr (M::kSendP) store_run(p_send);
  };
  // one copy split over `count` warps of the block, `w` being this warp's index
  // among them
  const auto copy = [&](auto mode,
                        const uint8_t* x_recv,
                        const uint8_t* p_recv,
                        uint8_t* x_send,
                        uint8_t* p_send,
                        uint32_t in_start,
                        uint32_t out_start,
                        uint32_t n,
                        uint32_t w,
                        uint32_t count) {
    const auto full = n / kWarpVecs * kWarpVecs;
    for (uint32_t base = w * kWarpVecs; base < full; base += count * kWarpVecs) {
      copy_packs(
          std::integral_constant<uint32_t, kHierRingUnroll>{},
          mode,
          x_recv,
          p_recv,
          x_send,
          p_send,
          in_start,
          out_start,
          base + lane_id);
    }
    for (uint32_t j = full + w * kWarpThreads + lane_id; j < n; j += count * kWarpThreads) {
      copy_packs(std::integral_constant<uint32_t, 1>{}, mode, x_recv, p_recv, x_send, p_send, in_start, out_start, j);
    }
  };
  // every wave's stores (and so the loads feeding them) completed
  const auto publish = [&] {
    wait_stores_sys();
    __syncthreads();
  };

  // each channel takes a contiguous share of whole hunks per chunk, then loops
  // over it in 2 * kCrossSize chunks of at most one slot: chunks [0, m) are
  // half 0, [m, 2m) half 1
  constexpr uint32_t kShareVecs = 2 * kCrossSize * kWarpVecs;
  const auto channel_vecs = div_ceil(div_ceil(num_vecs, num_channels), kShareVecs) * kShareVecs;
  const auto channel_start = min(ch * channel_vecs, num_vecs);
  const auto channel_len = min(channel_vecs, num_vecs - channel_start);
  const auto loop_vecs = 2 * kCrossSize * slot_vecs;
  for (uint32_t base = 0; base < channel_len; base += loop_vecs) {
    const auto len = min(loop_vecs, channel_len - base);
    const auto chunk_vecs = div_ceil(len, 2 * kCrossSize);
    const auto chunk_off = [&](uint32_t half, uint32_t c) { return (half * kCrossSize + c) * chunk_vecs; };
    const auto chunk_start = [&](uint32_t half, uint32_t c) { return channel_start + base + chunk_off(half, c); };
    const auto chunk_len = [&](uint32_t half, uint32_t c) {
      return min(chunk_vecs, len - min(len, chunk_off(half, c)));
    };

    // step 0 sends this SoC's own chunk; step 1 + q receives chunk k - 1 - q,
    // a partial up to the last reducing step kCrossSize - 1, then a reduced one
    const auto chunk_at = [&](uint32_t s) { return s == 0 ? k : (k + 2 * kCrossSize - s) % kCrossSize; };
    // the input chunk the peer reduces at step s, to pair-in packet p
    const auto send_pair_in = [&](uint32_t s, uint64_t p, uint32_t w, uint32_t count) {
      const auto c = chunk_at(s);
      copy(
          HierHop<false, false, true, false, false, true>{},
          nullptr,
          nullptr,
          nullptr,
          slot_of(ws.peer, kPairInFifo, p),
          chunk_start(h ^ 1, c),
          0,
          chunk_len(h ^ 1, c),
          w,
          count);
    };
    // the peer's reduced chunk of the other half from step s, in pair-out packet p
    const auto recv_pair_out = [&](uint32_t s, uint64_t p, uint32_t w, uint32_t count) {
      const auto c = chunk_at(s);
      copy(
          HierHop<false, true, false, true, false, false>{},
          nullptr,
          slot_of(ws.own, kPairOutFifo, p),
          nullptr,
          nullptr,
          0,
          chunk_start(h ^ 1, c),
          chunk_len(h ^ 1, c),
          w,
          count);
    };

    if (threadIdx.x == 0) wait_until(in_drained, &sem.in_drained, free_target(in_packet));
    __syncthreads();
    send_pair_in(0, in_packet, warp_id, num_warps);
    publish();
    if (threadIdx.x == 0) ptx::store_relaxed_sys(&peer_sem.in_ready, in_packet + 1);

    // Step s also sends the peer its pair-in packet for step s + 1 and takes the
    // peer's pair-out packet from step s - 1, so each step publishes once and
    // waits only on packets sent a step earlier, or this step by the cross ring.
    // The first kHierCrossWarps warps run the cross step, and the others the
    // pair copies alongside it.
    for (uint32_t s = 0; s < kNumSteps; ++s) {
      const auto c = chunk_at(s);
      const bool reducing = s < kCrossSize;
      const bool next_reducing = s + 1 < kCrossSize;
      const bool recv_x = s > 0;
      const bool send_x = s + 1 < kNumSteps;
      const bool done = s + 1 >= kCrossSize;  // this step's chunk is fully reduced
      const bool prev_done = s >= kCrossSize;
      const auto x_recv_p = x_packet + s - 1;
      const auto x_send_p = x_packet + s;

      if (threadIdx.x == 0) {
        if (next_reducing) wait_until(in_drained, &sem.in_drained, free_target(in_packet + 1));
        if (reducing) wait_until(in_ready, &sem.in_ready, in_packet + 1);
        if (recv_x) wait_until(x_ready, &sem.x_ready, x_recv_p + 1);
        if (send_x) wait_until(x_drained, &sem.x_drained, free_target(x_send_p));
        if (done) wait_until(out_drained, &sem.out_drained, free_target(out_packet));
        if (prev_done) wait_until(out_ready, &sem.out_ready, out_packet);
      }
      __syncthreads();
      if (warp_id < kHierCrossWarps) {
        const auto x_recv = slot_of(ws.own, kCrossFifo, x_recv_p);
        const auto p_recv = slot_of(ws.own, kPairInFifo, in_packet);
        const auto x_send = slot_of(ws.x_next, kCrossFifo, x_send_p);
        const auto p_send = slot_of(ws.peer, kPairOutFifo, out_packet);
        const auto start = chunk_start(h, c);
        const auto n = chunk_len(h, c);
        const auto w = warp_id;
        constexpr auto kW = kHierCrossWarps;
        if (s == 0) {
          copy(
              HierHop<false, true, true, false, true, false>{},
              nullptr,
              p_recv,
              x_send,
              nullptr,
              start,
              start,
              n,
              w,
              kW);
        } else if (!done) {
          copy(
              HierHop<true, true, true, false, true, false>{}, x_recv, p_recv, x_send, nullptr, start, start, n, w, kW);
        } else if (reducing) {
          copy(HierHop<true, true, true, true, true, true>{}, x_recv, p_recv, x_send, p_send, start, start, n, w, kW);
        } else if (send_x) {
          copy(HierHop<true, false, false, true, true, true>{}, x_recv, nullptr, x_send, p_send, 0, start, n, w, kW);
        } else {
          copy(HierHop<true, false, false, true, false, true>{}, x_recv, nullptr, nullptr, p_send, 0, start, n, w, kW);
        }
      } else {
        const auto w = warp_id - kHierCrossWarps;
        const auto count = num_warps - kHierCrossWarps;
        if (next_reducing) send_pair_in(s + 1, in_packet + 1, w, count);
        if (prev_done) recv_pair_out(s - 1, out_packet - 1, w, count);
      }
      publish();
      if (threadIdx.x == 0) {
        if (next_reducing) ptx::store_relaxed_sys(&peer_sem.in_ready, in_packet + 2);
        if (reducing) ptx::store_relaxed_sys(&peer_sem.in_drained, in_packet + 1);
        if (recv_x) ptx::store_relaxed_sys(&x_prev_sem.x_drained, x_recv_p + 1);
        if (send_x) ptx::store_relaxed_sys(&x_next_sem.x_ready, x_send_p + 1);
        if (done) ptx::store_relaxed_sys(&peer_sem.out_ready, out_packet + 1);
        if (prev_done) ptx::store_relaxed_sys(&peer_sem.out_drained, out_packet);
      }
      in_packet += reducing;
      out_packet += done;
    }

    // the peer's last reduced chunk
    if (threadIdx.x == 0) wait_until(out_ready, &sem.out_ready, out_packet);
    __syncthreads();
    recv_pair_out(kNumSteps - 1, out_packet - 1, warp_id, num_warps);
    publish();
    if (threadIdx.x == 0) ptx::store_relaxed_sys(&peer_sem.out_drained, out_packet);
    x_packet += kNumCross;
  }

  PDLTriggerSecondary<kUsePDL>();
  if (threadIdx.x == 0) {
    sem.x_packet = x_packet;
    sem.in_packet = in_packet;
    sem.out_packet = out_packet;
  }
}

}  // namespace ring_simple

namespace host::distributed {

/**
 * \brief Ring plane: symmetric FIFOs per rank plus one semaphore per ring.
 *
 * Each block of the launch runs one ring (channel), so each rank holds
 * `[num_rings][num_slots][slot_bytes]`: ring i of the predecessor streams
 * packets into slot `packet % num_slots` of row i. The counters in the
 * semaphores persist across launches, so the semaphores MUST be zero-filled
 * before first use.
 *
 * Holds no storage: the caller (Python) owns the tensors and their lifetime.
 * Everything is inline: the registration module and every kernel module are
 * separate shared objects that only share this header.
 */
struct RingPlaneObj : public tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.distributed.RingPlane", RingPlaneObj, tvm::ffi::Object);

  RingPlaneObj(
      uint32_t rank,
      uint32_t world_size,
      std::vector<tvm::ffi::TensorView> workspaces,  // world_size * [num_rings][num_slots][slot_bytes]
      std::vector<tvm::ffi::TensorView> semaphores)  // world_size * [num_rings][sizeof(RingSemaphore)]
      : rank(rank), world_size(world_size), num_rings(), num_slots(), slot_bytes(), semaphores{}, workspaces{} {
    CHECK_HOST(1 < world_size && world_size <= kMaxWorldSize) << "Invalid world size: " << world_size;
    CHECK_HOST(rank < world_size) << "Invalid rank " << rank << " for world size " << world_size;
    CHECK_HOST(workspaces.size() == world_size) << "Bad ring workspace count";
    CHECK_HOST(semaphores.size() == world_size) << "Bad ring semaphore count";
    auto R = SymbolicSize{"num_rings"};
    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"slot_bytes"};
    auto device_sym = SymbolicDevice{};
    device_sym.set_options<kDLGPU>();
    for (uint32_t i = 0; i < world_size; ++i) {
      TensorMatcher({R, S, N})  //
          .with_dtype<uint8_t>()
          .with_device(device_sym)
          .verify(workspaces[i]);
      TensorMatcher({R, static_cast<int64_t>(sizeof(ring_simple::RingSemaphore))})  //
          .with_dtype<uint8_t>()
          .with_device(device_sym)
          .verify(semaphores[i]);
    }
    const auto rings = R.unwrap(), slots = S.unwrap(), bytes = N.unwrap();
    CHECK_HOST(rings > 0 && bytes > 0 && bytes % 16 == 0) << "Bad ring slot: " << rings << " x " << bytes;
    CHECK_HOST(slots >= 2 && (slots & (slots - 1)) == 0) << "Ring slot count must be a power of two >= 2: " << slots;
    // the kernel forms FIFO offsets in 32 bits
    CHECK_HOST(rings * slots * bytes <= std::numeric_limits<uint32_t>::max())
        << rings * slots * bytes << " bytes of ring FIFO exceeds the 32-bit offset range";

    this->num_rings = static_cast<uint32_t>(rings);
    this->num_slots = static_cast<uint32_t>(slots);
    this->slot_bytes = bytes;
    for (uint32_t i = 0; i < world_size; ++i) {
      this->semaphores[i] = static_cast<ring_simple::RingSemaphore*>(semaphores[i].data_ptr());
      this->workspaces[i] = static_cast<uint8_t*>(workspaces[i].data_ptr());
    }
  }

  uint32_t rank;
  uint32_t world_size;
  uint32_t num_rings;                                                 // one per block of the launch
  uint32_t num_slots;                                                 // FIFO depth per ring, a power of two
  int64_t slot_bytes;                                                 // one packet
  std::array<ring_simple::RingSemaphore*, kMaxWorldSize> semaphores;  // symmetric memory
  std::array<uint8_t*, kMaxWorldSize> workspaces;                     // symmetric memory

  ring_simple::RingWorkSpace get_workspace() const {
    const auto next = (rank + 1) % world_size;
    const auto prev = (rank + world_size - 1) % world_size;
    return ring_simple::RingWorkSpace{
        workspaces[rank],
        workspaces[next],
        semaphores[rank],
        semaphores[prev],
        semaphores[next],
        static_cast<uint32_t>(slot_bytes),
        num_slots,
    };
  }
};

SGLANG_REGISTER_FFI_REFERENCE_CLASS(RingPlaneRef, RingPlaneObj);

/**
 * \brief Hier ring plane: per rank and ring, a cross, a pair-in and a pair-out
 * FIFO plus one semaphore, over a group made of DPX pairs.
 *
 * Each rank holds `[num_rings][3][num_slots][slot_bytes]`. Ranks are group
 * ranks; the caller resolves the DPX peer and the cross-ring neighbours (the
 * same partition of the previous and next SoC). The counters persist across
 * launches, so the semaphores MUST be zero-filled before first use.
 *
 * Holds no storage: the caller (Python) owns the tensors and their lifetime.
 */
struct HierRingPlaneObj : public tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.distributed.HierRingPlane", HierRingPlaneObj, tvm::ffi::Object);

  HierRingPlaneObj(
      uint32_t rank,
      uint32_t world_size,
      uint32_t peer,
      uint32_t x_prev,
      uint32_t x_next,
      uint32_t half,
      uint32_t soc,
      uint32_t num_socs,
      std::vector<tvm::ffi::TensorView> workspaces,  // world_size * [num_rings][3][num_slots][slot_bytes]
      std::vector<tvm::ffi::TensorView> semaphores)  // world_size * [num_rings][sizeof(HierRingSemaphore)]
      : rank(rank),
        world_size(world_size),
        peer(peer),
        x_prev(x_prev),
        x_next(x_next),
        half(half),
        soc(soc),
        num_socs(num_socs),
        num_rings(),
        num_slots(),
        slot_bytes(),
        semaphores{},
        workspaces{} {
    CHECK_HOST(3 < world_size && world_size <= kMaxWorldSize && world_size % 2 == 0)
        << "Invalid world size for a hier ring: " << world_size;
    CHECK_HOST(rank < world_size && peer < world_size && x_prev < world_size && x_next < world_size)
        << "Invalid hier ring ranks";
    CHECK_HOST(peer != rank && half < 2 && num_socs * 2 == world_size && soc < num_socs) << "Invalid hier ring layout";
    CHECK_HOST(workspaces.size() == world_size) << "Bad hier ring workspace count";
    CHECK_HOST(semaphores.size() == world_size) << "Bad hier ring semaphore count";
    auto R = SymbolicSize{"num_rings"};
    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"slot_bytes"};
    auto device_sym = SymbolicDevice{};
    device_sym.set_options<kDLGPU>();
    for (uint32_t i = 0; i < world_size; ++i) {
      TensorMatcher({R, static_cast<int64_t>(ring_simple::kNumHierFifos), S, N})  //
          .with_dtype<uint8_t>()
          .with_device(device_sym)
          .verify(workspaces[i]);
      TensorMatcher({R, static_cast<int64_t>(sizeof(ring_simple::HierRingSemaphore))})  //
          .with_dtype<uint8_t>()
          .with_device(device_sym)
          .verify(semaphores[i]);
    }
    const auto rings = R.unwrap(), slots = S.unwrap(), bytes = N.unwrap();
    CHECK_HOST(rings > 0 && bytes > 0 && bytes % 16 == 0) << "Bad hier ring slot: " << rings << " x " << bytes;
    CHECK_HOST(slots >= 2 && (slots & (slots - 1)) == 0) << "Ring slot count must be a power of two >= 2: " << slots;
    // the kernel forms FIFO offsets in 32 bits
    CHECK_HOST(rings * ring_simple::kNumHierFifos * slots * bytes <= std::numeric_limits<uint32_t>::max())
        << "hier ring FIFOs exceed the 32-bit offset range";

    this->num_rings = static_cast<uint32_t>(rings);
    this->num_slots = static_cast<uint32_t>(slots);
    this->slot_bytes = bytes;
    for (uint32_t i = 0; i < world_size; ++i) {
      this->semaphores[i] = static_cast<ring_simple::HierRingSemaphore*>(semaphores[i].data_ptr());
      this->workspaces[i] = static_cast<uint8_t*>(workspaces[i].data_ptr());
    }
  }

  uint32_t rank;
  uint32_t world_size;
  uint32_t peer;      // the other DPX partition of this rank's SoC
  uint32_t x_prev;    // the same partition of the previous SoC
  uint32_t x_next;    // the same partition of the next SoC
  uint32_t half;      // this rank's index in its pair
  uint32_t soc;       // this rank's SoC's position in the cross ring
  uint32_t num_socs;  // cross ring size
  uint32_t num_rings;
  uint32_t num_slots;
  int64_t slot_bytes;
  std::array<ring_simple::HierRingSemaphore*, kMaxWorldSize> semaphores;  // symmetric memory
  std::array<uint8_t*, kMaxWorldSize> workspaces;                         // symmetric memory

  ring_simple::HierRingWorkSpace get_workspace() const {
    return ring_simple::HierRingWorkSpace{
        workspaces[rank],
        workspaces[x_next],
        workspaces[peer],
        semaphores[rank],
        semaphores[x_prev],
        semaphores[x_next],
        semaphores[peer],
        static_cast<uint32_t>(slot_bytes),
        num_slots,
    };
  }
};

SGLANG_REGISTER_FFI_REFERENCE_CLASS(HierRingPlaneRef, HierRingPlaneObj);

}  // namespace host::distributed

inline void register_ring_plane() {
  namespace refl = tvm::ffi::reflection;
  namespace dist = host::distributed;
  using Tensors = std::vector<tvm::ffi::TensorView>;

  refl::ObjectDef<dist::RingPlaneObj>()
      .def(refl::init<uint32_t, uint32_t, Tensors, Tensors>(), "__init__")
      .def_ro("rank", &dist::RingPlaneObj::rank)
      .def_ro("world_size", &dist::RingPlaneObj::world_size)
      .def_ro("num_rings", &dist::RingPlaneObj::num_rings)
      .def_ro("num_slots", &dist::RingPlaneObj::num_slots)
      .def_ro("slot_bytes", &dist::RingPlaneObj::slot_bytes);
  refl::ObjectDef<dist::HierRingPlaneObj>()
      .def(
          refl::
              init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, Tensors, Tensors>(),
          "__init__")
      .def_ro("rank", &dist::HierRingPlaneObj::rank)
      .def_ro("world_size", &dist::HierRingPlaneObj::world_size)
      .def_ro("num_socs", &dist::HierRingPlaneObj::num_socs)
      .def_ro("num_rings", &dist::HierRingPlaneObj::num_rings)
      .def_ro("num_slots", &dist::HierRingPlaneObj::num_slots)
      .def_ro("slot_bytes", &dist::HierRingPlaneObj::slot_bytes);
}

template <typename T, uint32_t kWorldSize, bool kUsePDL>
struct RingSimpleKernel {
  using Tensor = tvm::ffi::Tensor;

  /// All-reduce `in` over the ring plane, one ring per block.
  static Tensor run(const host::distributed::RingPlaneRef ring_ref, const Tensor in) {
    using namespace host;
    const auto& ring = *ring_ref.get();
    CHECK_HOST(ring.world_size == kWorldSize) << ring.world_size;
    CHECK_HOST(in.IsContiguous() && is_type<T>(in.dtype()) && in.device().device_type == kDLGPU);
    const auto num_elems = in.numel();
    const auto nbytes = static_cast<int64_t>(num_elems * sizeof(T));
    CHECK_HOST(reinterpret_cast<intptr_t>(in.data_ptr()) % 16 == 0 && nbytes % 16 == 0);
    const auto num_vecs = static_cast<uint32_t>(nbytes / 16);
    CHECK_HOST(int64_t{num_vecs} * 16 == nbytes) << nbytes;

    Tensor out = ffi::empty_like(in);
    const auto params = ring_simple::Params{
        .input = in.data_ptr(),
        .output = out.data_ptr(),
        .num_vecs = num_vecs,
        .rank = ring.rank,
        .ws = ring.get_workspace(),
    };
    const auto stream = LaunchKernel::resolve_device(in.device());
    LaunchKernel(ring.num_rings, ring_simple::kRingSimpleThreads, stream)  //
        .enable_pdl(kUsePDL)(ring_simple::all_reduce_ring_simple_kernel<T, kWorldSize, kUsePDL>, params);
    return out;
  }
};

template <typename T, uint32_t kCrossSize, bool kUsePDL>
struct HierRingSimpleKernel {
  using Tensor = tvm::ffi::Tensor;

  /// All-reduce `in` over the hier ring plane, one ring per block.
  static Tensor run(const host::distributed::HierRingPlaneRef ring_ref, const Tensor in) {
    using namespace host;
    const auto& ring = *ring_ref.get();
    CHECK_HOST(ring.num_socs == kCrossSize) << ring.num_socs;
    CHECK_HOST(in.IsContiguous() && is_type<T>(in.dtype()) && in.device().device_type == kDLGPU);
    const auto num_elems = in.numel();
    const auto nbytes = static_cast<int64_t>(num_elems * sizeof(T));
    CHECK_HOST(reinterpret_cast<intptr_t>(in.data_ptr()) % 16 == 0 && nbytes % 16 == 0);
    const auto num_vecs = static_cast<uint32_t>(nbytes / 16);
    CHECK_HOST(int64_t{num_vecs} * 16 == nbytes) << nbytes;

    Tensor out = ffi::empty_like(in);
    const auto params = ring_simple::HierParams{
        .input = in.data_ptr(),
        .output = out.data_ptr(),
        .num_vecs = num_vecs,
        .half = ring.half,
        .soc = ring.soc,
        .ws = ring.get_workspace(),
    };
    const auto stream = LaunchKernel::resolve_device(in.device());
    LaunchKernel(ring.num_rings, ring_simple::kRingSimpleThreads, stream)  //
        .enable_pdl(kUsePDL)(ring_simple::all_reduce_hier_ring_kernel<T, kCrossSize, kUsePDL>, params);
    return out;
  }
};

}  // namespace sglang
