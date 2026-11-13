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

}  // namespace sglang
