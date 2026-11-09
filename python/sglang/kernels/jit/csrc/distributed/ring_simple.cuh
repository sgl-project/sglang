// ring_simple, CustomAllReduceV2's ring all-reduce for ROCm gfx1250 (MI450): each
// chunk goes to the next rank only, through a fixed-size FIFO behind 64-bit
// counters, in 2 * (world_size - 1) hops per loop. It runs over the
// communicator's ring plane.
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

#include <cstdint>

#ifndef USE_ROCM
#error "ring_simple is ROCm only"
#endif

namespace sglang {

namespace ring_simple {

using device::distributed::RingWorkSpace;

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
/// one counter and reads the payload once. FIFO payload moves as streaming
/// loads and system-scope stores; the input and output take plain accesses.
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
      seen = ptx::load_acquire_sys(ptr);
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
    ptx::wait_stores_sys();
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
          if constexpr (kRecv) ptx::ld_nt_16B(vecs[u], recv_slot, j + u * kStride);
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
            ptx::st_relaxed_nt_run_16B<kStride, kUnroll>(vecs, send_at);
          } else {
#pragma unroll
            for (uint32_t u = 0; u < kUnroll; ++u)
              ptx::st_relaxed_nt_16B(vecs[u], send_at, u * kStride);
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
  // kWorldSize chunks of at most one slot
  const auto channel_vecs = div_ceil(num_vecs, num_channels);
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

}  // namespace ring_simple

template <typename T, uint32_t kWorldSize, bool kUsePDL>
struct RingSimpleKernel {
  using Tensor = tvm::ffi::Tensor;

  /// All-reduce `in` over the communicator's ring plane, one ring per block.
  static Tensor run(const host::distributed::CommunicatorRef comm_ref, const Tensor in) {
    using namespace host;
    const auto& comm = *comm_ref.get();
    CHECK_HOST(comm.get_world_size() == kWorldSize) << comm.get_world_size();
    CHECK_HOST(in.IsContiguous() && is_type<T>(in.dtype()) && in.device().device_type == kDLGPU);
    const auto num_elems = in.numel();
    const auto nbytes = static_cast<int64_t>(num_elems * sizeof(T));
    CHECK_HOST(reinterpret_cast<intptr_t>(in.data_ptr()) % 16 == 0 && nbytes % 16 == 0);
    const auto num_vecs = static_cast<uint32_t>(nbytes / 16);
    CHECK_HOST(int64_t{num_vecs} * 16 == nbytes) << nbytes;

    const auto& ring = comm.get_ring_simple_obj();
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

}  // namespace sglang
