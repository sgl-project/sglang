// Two-shot (reduce-scatter + all-gather) all-reduce over a torch
// symmetric-memory buffer.
//
// It operates IN PLACE on the peer symm buffers: the producer (e.g. the wo_ud /
// MoE-combine GEMM) writes its local shard straight into THIS rank's symm buffer
// (via get_ar_buffer), so there is no stage-in copy; the reduced result is left
// in the buffer and handed back to Python as a view, so there is no copy-out.
//
// Correctness (two-shot is race-safe in place): rank r owns the disjoint vec
// slice [local_vec_start, local_vec_finish); it reads every peer's slice, sums,
// and broadcasts the sum back to every peer's slice. Only rank r ever writes
// slice S_r (in any buffer), so there is no write-write conflict, and each
// per-element load completes before its store (data dependency).
//
// Two variants:
//   * ..._kernel  (v1): no in-kernel sync; the caller fences with the symm-mem
//     handle's barrier() on each side (3 launches total).
//   * ..._fused_kernel (v2): an in-kernel per-block system barrier (entry:
//     producers done + visible; exit: broadcasts done + visible), so the whole
//     all-reduce is a single launch. The barrier uses a DEDICATED symmetric
//     flags buffer (independent of torch's signal pad, so no interference with
//     multimem) and a device-resident monotonic epoch counter per block, which
//     keeps advancing across launches -- including CUDA-graph replays -- so
//     flags never go stale (spin is `flag < epoch`, epoch strictly increasing).
//
// Fusion seam: the reduced `result` Storage below is where an epilogue (RMSNorm
// / short-conv / bias) plugs in -- applied in registers before the broadcast
// store, so the normed/conv'd result never makes an extra HBM round trip.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include "inkling_ar_barrier.cuh"
#include <bit>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace {

template <typename DType, uint32_t kNumGPU>
struct InklingAllReduceTrait {
  static constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  static constexpr uint32_t kElemsPerVec = kVecSize * 2;
  using DType2 = packed_t<DType>;
  using Storage = device::AlignedVector<DType2, kVecSize>;
  static_assert(sizeof(Storage) == 16 && alignof(Storage) == 16, "Storage must be 16B");
  static_assert(std::has_single_bit(kNumGPU), "kNumGPU must be a power of two");
};

// Register-level fused add of two vecs (fp32 math, ONE round to DType) -- the
// exact numerics of torch.add on two bf16 tensors, so fusing the shared-expert
// partials stays bit-identical to the unfused {torch.add -> AR} chain.
template <typename DType>
__device__ __forceinline__ typename InklingAllReduceTrait<DType, 2>::Storage add_vec_rn(
    const typename InklingAllReduceTrait<DType, 2>::Storage& a,
    const typename InklingAllReduceTrait<DType, 2>::Storage& b) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, 2>;  // kNumGPU-independent
  using DType2 = typename Trait::DType2;
  typename Trait::Storage out;
#pragma unroll
  for (uint32_t j = 0; j < Trait::kVecSize; ++j) {
    const fp32x2_t x = cast<fp32x2_t>(a[j]);
    const fp32x2_t y = cast<fp32x2_t>(b[j]);
    fp32x2_t s;
    s.x = x.x + y.x;
    s.y = x.y + y.y;
    out[j] = cast<DType2>(s);
  }
  return out;
}

// Fused-shared PROLOGUE for the pull-based kernels (v2/v3/v3b/v4): fold this
// rank's LOCAL shared-expert partials into its own symm input region before
// the ENTRY barrier, so every peer's ld_reduce / peer-read sums
// (routed_r + shared_r) across ranks. The entry barrier must then run in
// publish mode (grid_system_barrier, publish_writes=true): these are in-kernel
// stores by ALL CTAs, not prior-kernel stores, so each CTA has to
// system-publish them before the leader's release. (The per-block barrier
// cannot order this: block b's fold range is not the range peer block b
// reads.) The push-based kernels (v5 & the fused decode family) instead fold
// in registers at the push -- see the shared branch in the push loop.
template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ void
fold_shared_local(DType* __restrict__ buf, const DType* __restrict__ shared, uint32_t num_items) {
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  using Storage = typename Trait::Storage;
  const uint32_t total_vec = num_items / Trait::kElemsPerVec;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vec; v += stride) {
    Storage a, b;
    a.load(buf, v);
    b.load(shared, v);
    add_vec_rn<DType>(a, b).store(buf, v);
  }
}

// Two-shot partition: contiguous, warp-aligned vec slice per rank. Returns
// {start, count} in vec units (empty for trailing ranks when the range is small).
template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ uint2 rank_vec_slice(uint32_t rank, uint32_t num_items) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  const uint32_t total_vec = num_items / Trait::kElemsPerVec;
  const uint32_t vec_per_rank = div_ceil(div_ceil(total_vec, kNumGPU), kWarpThreads) * kWarpThreads;
  const uint32_t start = min(rank * vec_per_rank, total_vec);
  const uint32_t finish = min(start + vec_per_rank, total_vec);
  return {start, finish - start};
}

// Offset each peer pointer to this rank's slice, and return the slice's local
// vec count.
template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ uint32_t
slice_setup(DType* (&input)[kNumGPU], void* const* peer_ptrs, uint32_t rank, uint32_t num_items) {
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  const uint2 slice = rank_vec_slice<DType, kNumGPU>(rank, num_items);
  const uint32_t base = slice.x * Trait::kElemsPerVec;
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i)
    input[i] = static_cast<DType*>(peer_ptrs[i]) + base;
  return slice.y;  // local vec count
}

template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ void two_shot_reduce_local(DType* (&input)[kNumGPU], uint32_t local_vecs) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  using Storage = typename Trait::Storage;
  using DType2 = typename Trait::DType2;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < local_vecs; v += stride) {
    Storage s[kNumGPU];
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i)
      s[i].load(input[i], v);
    Storage result;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      fp32x2_t acc = cast<fp32x2_t>(s[0][j]);
#pragma unroll
      for (uint32_t i = 1; i < kNumGPU; ++i) {
        const fp32x2_t x = cast<fp32x2_t>(s[i][j]);
        acc.x += x.x;
        acc.y += x.y;
      }
      result[j] = cast<DType2>(acc);  // <-- EPILOGUE SEAM
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i)
      result.store(input[i], v);
  }
}

// v1: no in-kernel barrier (caller fences via hdl.barrier()).
template <typename DType, uint32_t kNumGPU>
__global__ __launch_bounds__(1024, 1) void inkling_two_shot_all_reduce_kernel(
    void* const* __restrict__ peer_ptrs, const uint32_t rank, const uint32_t num_items) {
  DType* input[kNumGPU];
  const uint32_t local_vecs = slice_setup<DType, kNumGPU>(input, peer_ptrs, rank, num_items);
  two_shot_reduce_local<DType, kNumGPU>(input, local_vecs);
}

// v2: single-launch, fused entry + exit system barrier. `shared` (optional):
// this rank's LOCAL shared-expert partials, folded into its own buffer before
// the entry barrier (which then must publish -- see fold_shared_local).
template <typename DType, uint32_t kNumGPU>
__global__ __launch_bounds__(1024, 1) void inkling_two_shot_all_reduce_fused_kernel(
    void* const* __restrict__ peer_ptrs,
    void* const* __restrict__ flag_ptrs,
    uint32_t* __restrict__ state,
    const DType* __restrict__ shared,
    const uint32_t rank,
    const uint32_t num_items) {
  DType* input[kNumGPU];
  const uint32_t local_vecs = slice_setup<DType, kNumGPU>(input, peer_ptrs, rank, num_items);
  if (shared != nullptr) {
    fold_shared_local<DType, kNumGPU>(static_cast<DType*>(peer_ptrs[rank]), shared, num_items);
  }
  // ENTRY: producers done + visible (publish the fold's in-kernel stores too).
  inkling_ar::grid_system_barrier<kNumGPU>(state, flag_ptrs, rank, 0, /*publish_writes=*/shared != nullptr);
  two_shot_reduce_local<DType, kNumGPU>(input, local_vecs);
  inkling_ar::grid_system_barrier<kNumGPU>(
      state, flag_ptrs, rank, 1, /*publish_writes=*/true);  // EXIT: broadcasts done + visible
}

// Multimem one-shot all-reduce: uses the NVLink multicast ld_reduce/st hardware
// instructions on the symm buffer's multicast pointer -- the same in-switch
// reduce torch's multimem_all_reduce_ uses -- so it matches multimem for the
// tiny, latency-bound decode messages where two-shot's N peer reads lose. Reduce
// is one transaction (hardware sums all GPUs); scatter partition keeps the store
// traffic minimal. bf16-only (multimem.add supports .bf16x2 on sm90/sm100).
// kPerBlockBarrier swaps both barriers for block_system_barrier (per-block
// peer handshake, no grid funnel). Correct for the two-shot too: any peer
// block's ENTRY signal proves that peer's producer kernel completed (kernel
// serialization on its stream), and kernel end is a grid-wide join, so my
// per-block EXIT waits compose into "every peer block's broadcasts done"
// before my consumer can run. The two calls share the per-block epoch slot
// (it just advances twice per launch).
template <typename DType, uint32_t kNumGPU, bool kPerBlockBarrier>
__global__ __launch_bounds__(1024, 1) void inkling_multimem_one_shot_fused_kernel(
    DType* __restrict__ mc_ptr,     // multicast base pointer (covers all peers)
    DType* __restrict__ local_ptr,  // this rank's LOCAL base of the same buffer
    void* const* __restrict__ flag_ptrs,
    uint32_t* __restrict__ state,
    const DType* __restrict__ shared,  // optional LOCAL shared-expert partials
    const uint32_t rank,
    const uint32_t num_items) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  static_assert(std::is_same_v<DType, bf16_t>, "multimem.add path is bf16-only");
  constexpr uint32_t kElemsPerVec = Trait::kElemsPerVec;  // 8 bf16 = 16 B

  const uint2 slice = rank_vec_slice<DType, kNumGPU>(rank, num_items);
  const uint32_t local_vecs = slice.y;
  DType* mc = mc_ptr + slice.x * kElemsPerVec;

  if (shared != nullptr) {
    // Fold covers the FULL range while each peer ld_reduces only its slice, so
    // the per-block handshake cannot order it -- use the publishing grid
    // barrier for entry even in v3b (exit stays per-block).
    fold_shared_local<DType, kNumGPU>(local_ptr, shared, num_items);
    inkling_ar::grid_system_barrier<kNumGPU>(state, flag_ptrs, rank, 0, /*publish_writes=*/true);
  } else if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(state, flag_ptrs, rank);  // ENTRY
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(
        state, flag_ptrs, rank, 0, /*publish_writes=*/false);  // ENTRY: producers done + visible
  }
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < local_vecs; v += stride) {
    DType* addr = mc + v * kElemsPerVec;  // 16 B, 16-B aligned
    uint32_t r0, r1, r2, r3;
    // hardware reduce across all GPUs mapped to the multicast region.
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(addr));
    // <-- EPILOGUE SEAM (norm / sconv / bias on {r0..r3} before broadcast)
    // broadcast the reduced slice to every GPU.
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
        : "memory");
  }
  if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(state, flag_ptrs, rank);  // EXIT (release signal publishes)
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(
        state, flag_ptrs, rank, 1, /*publish_writes=*/true);  // EXIT: broadcasts done + visible
  }
}

// One-shot PUSH all-reduce (v5): each rank multicast-STORES its full input into
// its per-rank slot of a symmetric staging area (the NVSwitch replicates the
// slot to every GPU), ONE grid barrier waits for all pushes to land, then each
// rank reduces the N staged shards LOCALLY (fp32 accum) into a LOCAL output.
// Single barrier total: the push needs no entry barrier (it publishes only this
// rank's own producer data, stream-ordered locally) and the local output needs
// no exit barrier. Staging reuse is caller-managed (A/B rotation, like v4's
// input; the next AR's barrier proves peers consumed the old buffer).
//
// vs v3/mm (two-shot): drops one full cross-GPU barrier round trip -- wins the
// latency-bound band. vs v4 (full one-shot ld_reduce): switch REPLICATION is
// cheap where the switch's reduce engine serializes N redundant full-range
// reduces, so this scales past v4's 2-row ceiling. Fabric cost: n egress,
// (N-1)*n ingress per GPU; local HBM/L2: N*n read + n write. Like v4, each rank
// holds the FULL row at the epilogue seam (natural RMSNorm-fusion base).
// bf16-only (multimem.st .bf16x2).
//
// kPerBlockBarrier selects block_system_barrier (per-block peer handshake, no
// grid funnel -- the multi-block latency winner) over the single-leader grid
// barrier. Safe here because the reduce loop reads exactly the vec ranges the
// blockIdx-matched pushes wrote.
template <typename DType, uint32_t kNumGPU, bool kPerBlockBarrier>
__global__ __launch_bounds__(1024, 1) void inkling_multimem_push_oneshot_kernel(
    const DType* __restrict__ in_ptr,     // local input (producer's partial sums)
    DType* __restrict__ mc_stage_ptr,     // multicast staging base (slot r at r*num_items)
    const DType* __restrict__ stage_ptr,  // this GPU's LOCAL view of the staging base
    DType* __restrict__ out_ptr,          // local output
    void* const* __restrict__ flag_ptrs,
    uint32_t* __restrict__ state,
    const DType* __restrict__ shared,  // optional LOCAL shared-expert partials
    const uint32_t rank,
    const uint32_t num_items) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  static_assert(std::is_same_v<DType, bf16_t>, "multimem path is bf16-only");
  constexpr uint32_t kElemsPerVec = Trait::kElemsPerVec;  // 8 bf16 = 16 B
  const uint32_t total_vec = num_items / kElemsPerVec;
  const uint32_t stride = gridDim.x * blockDim.x;

  // Phase 1: push. One multicast store per vec; the switch fans it out to every
  // GPU's replica of slot `rank` (including our own). With `shared`, the
  // shared-expert partials fold into the pushed value in registers (fp32 add,
  // one bf16 round -- torch.add numerics) at ZERO extra fabric or HBM traffic;
  // both barrier flavors stay valid because push/reduce mappings are unchanged.
  DType* slot = mc_stage_ptr + rank * num_items;
  if (shared != nullptr) {
    using Storage = typename Trait::Storage;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vec; v += stride) {
      Storage a, b;
      a.load(in_ptr, v);
      b.load(shared, v);
      const Storage s = add_vec_rn<DType>(a, b);
      const uint4 d = *reinterpret_cast<const uint4*>(&s);
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(slot + v * kElemsPerVec),
                   "r"(d.x),
                   "r"(d.y),
                   "r"(d.z),
                   "r"(d.w)
                   : "memory");
    }
  } else {
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vec; v += stride) {
      const uint4 d = *reinterpret_cast<const uint4*>(in_ptr + v * kElemsPerVec);
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(slot + v * kElemsPerVec),
                   "r"(d.x),
                   "r"(d.y),
                   "r"(d.z),
                   "r"(d.w)
                   : "memory");
    }
  }

  // Single barrier: publish our pushes and wait until every rank's pushes for
  // OUR ranges have landed in this GPU's local staging copy.
  if constexpr (kPerBlockBarrier) {
    inkling_ar::block_system_barrier<kNumGPU>(state, flag_ptrs, rank);
  } else {
    inkling_ar::grid_system_barrier<kNumGPU>(state, flag_ptrs, rank, 0, /*publish_writes=*/true);
  }

  // Phase 2: local reduce of the N staged shards -- all-local reads (the pushes
  // just landed in L2), fp32 accumulation.
  using Storage = typename Trait::Storage;
  using DType2 = typename Trait::DType2;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vec; v += stride) {
    Storage s[kNumGPU];
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i)
      s[i].load(stage_ptr + i * num_items, v);
    Storage result;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      fp32x2_t acc = cast<fp32x2_t>(s[0][j]);
#pragma unroll
      for (uint32_t i = 1; i < kNumGPU; ++i) {
        const fp32x2_t x = cast<fp32x2_t>(s[i][j]);
        acc.x += x.x;
        acc.y += x.y;
      }
      result[j] = cast<DType2>(acc);  // <-- EPILOGUE SEAM (full row on-rank)
    }
    result.store(out_ptr, v);
  }
}

// Full one-shot: every rank ld_reduces the ENTIRE range (multicast hardware sum
// -> full result), writing it to a LOCAL output buffer. No broadcast and NO exit
// barrier -- the result is complete on this rank, and input-buffer reuse is the
// caller's responsibility (double-buffer the input). Halving the barrier count
// wins for tiny, latency-bound (decode) messages. bf16-only.
template <typename DType, uint32_t kNumGPU>
__global__ __launch_bounds__(1024, 1) void inkling_multimem_full_oneshot_kernel(
    DType* __restrict__ mc_ptr,        // multicast input base (covers all peers)
    DType* __restrict__ local_in_ptr,  // this rank's LOCAL base of the input
    DType* __restrict__ out_ptr,       // local output base
    void* const* __restrict__ flag_ptrs,
    uint32_t* __restrict__ state,
    const DType* __restrict__ shared,  // optional LOCAL shared-expert partials
    const uint32_t rank,
    const uint32_t num_items) {
  using namespace device;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  static_assert(std::is_same_v<DType, bf16_t>, "multimem.add path is bf16-only");
  constexpr uint32_t kElemsPerVec = Trait::kElemsPerVec;  // 8 bf16 = 16 B
  const uint32_t total_vec = num_items / kElemsPerVec;

  if (shared != nullptr) {
    // Fold into this rank's (double-buffered) input region; the publishing
    // entry barrier then orders it for every peer's ld_reduce. v4 fires only
    // for 1-2 rows, so the extra local pass is negligible next to the
    // torch.add launch it replaces.
    fold_shared_local<DType, kNumGPU>(local_in_ptr, shared, num_items);
  }
  inkling_ar::grid_system_barrier<kNumGPU>(
      state, flag_ptrs, rank, 0, /*publish_writes=*/shared != nullptr);  // ENTRY only (single barrier)
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_vec; v += stride) {
    DType* in = mc_ptr + v * kElemsPerVec;
    uint32_t r0, r1, r2, r3;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(in));
    // <-- EPILOGUE SEAM (norm / sconv / bias on {r0..r3} before the local store)
    *reinterpret_cast<uint4*>(out_ptr + v * kElemsPerVec) = make_uint4(r0, r1, r2, r3);
  }
  // NO exit barrier: result is local & complete; input reuse is caller-managed.
}

// Blocks needed to cover this rank's two-shot slice (v1/v2/v3 partition).
template <typename DType, uint32_t kNumGPU>
uint32_t work_num_blocks(uint32_t n, uint32_t block_size) {
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  const uint32_t total_vec = n / Trait::kElemsPerVec;
  const uint32_t vec_per_rank =
      host::div_ceil(host::div_ceil(total_vec, kNumGPU), device::kWarpThreads) * device::kWarpThreads;
  return max(1u, host::div_ceil(vec_per_rank, block_size));
}

// Blocks needed to cover the FULL vec range (the full one-shot kernel reads
// the entire range on every rank, not a per-rank slice).
template <typename DType>
uint32_t full_range_num_blocks(uint32_t n, uint32_t block_size) {
  constexpr uint32_t kElemsPerVec = InklingAllReduceTrait<DType, 2>::kElemsPerVec;  // kNumGPU-independent
  return max(1u, host::div_ceil(n / kElemsPerVec, block_size));
}

// Max blocks that are simultaneously resident for `kernel` at `block_size`.
// The grid-level barrier REQUIRES all launched blocks to be co-resident (the
// leader waits for every block to arrive); launching more would deadlock, so
// the fused kernels cap their grid at this. Small messages need far fewer.
// Cached per (kernel, block_size, device): the occupancy query costs ~a few us
// on every eager launch of a latency-bound AR otherwise.
template <typename Kernel>
uint32_t max_resident_blocks(Kernel kernel, uint32_t block_size, DLDevice device) {
  using namespace host;
  static std::mutex mu;
  static std::unordered_map<uint64_t, uint32_t> cache;
  const uint64_t key = (std::bit_cast<uint64_t>(reinterpret_cast<void*>(kernel)) << 12) ^
                       (static_cast<uint64_t>(block_size) << 8) ^ static_cast<uint64_t>(device.device_id);
  {
    std::lock_guard<std::mutex> lk(mu);
    if (auto it = cache.find(key); it != cache.end()) return it->second;
  }
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device.device_id);
  RuntimeCheck(sm_count > 0, "failed to query multiProcessorCount");
  const uint32_t bps = runtime::get_blocks_per_sm(kernel, block_size);
  RuntimeCheck(bps > 0, "kernel has zero occupancy at block_size ", block_size);
  const uint32_t result = static_cast<uint32_t>(sm_count) * bps;
  std::lock_guard<std::mutex> lk(mu);
  cache.emplace(key, result);
  return result;
}

// Optional shared-expert partials: numel == 0 -> disabled (nullptr); else a
// LOCAL contiguous tensor covering num_items, folded in-kernel.
template <typename DType>
const DType* shared_ptr_or_null(tvm::ffi::TensorView shared, int64_t num_items) {
  using namespace host;
  if (shared.numel() == 0) return nullptr;
  RuntimeCheck(shared.IsContiguous(), "shared must be contiguous");
  RuntimeCheck(is_type<DType>(shared.dtype()), "shared dtype mismatch");
  RuntimeCheck(shared.numel() >= num_items, "shared smaller than num_items");
  RuntimeCheck(std::bit_cast<intptr_t>(shared.data_ptr()) % 16 == 0, "shared not 16B aligned");
  return reinterpret_cast<const DType*>(shared.data_ptr());
}

template <typename DType, uint32_t kNumGPU>
void validate(tvm::ffi::TensorView buf, int64_t peer_ptrs_dev, int64_t rank, int64_t num_items, uint32_t& n) {
  using namespace host;
  using Trait = InklingAllReduceTrait<DType, kNumGPU>;
  n = static_cast<uint32_t>(num_items);
  RuntimeCheck(buf.IsContiguous(), "buffer must be contiguous");
  RuntimeCheck(buf.device().device_type == kDLCUDA, "buffer must be on a CUDA device");
  RuntimeCheck(is_type<DType>(buf.dtype()), "buffer dtype mismatch");
  RuntimeCheck(static_cast<int64_t>(n) == num_items, "num_items exceeds 4G");
  RuntimeCheck(buf.numel() >= num_items, "buffer smaller than num_items");
  RuntimeCheck(n % Trait::kElemsPerVec == 0, "num_items must be a multiple of ", Trait::kElemsPerVec);
  RuntimeCheck(std::bit_cast<intptr_t>(buf.data_ptr()) % 16 == 0, "buffer not 16B aligned");
  RuntimeCheck(peer_ptrs_dev != 0, "peer_ptrs_dev is null");
  RuntimeCheck(rank >= 0 && rank < kNumGPU, "rank out of range");
}

template <typename DType, uint32_t kNumGPU>
void inkling_two_shot_all_reduce(
    tvm::ffi::TensorView local_buffer, int64_t peer_ptrs_dev, int64_t rank, int64_t num_items) {
  using namespace host;
  uint32_t n;
  validate<DType, kNumGPU>(local_buffer, peer_ptrs_dev, rank, num_items, n);
  const auto device = local_buffer.device();
  const uint32_t num_blocks = work_num_blocks<DType, kNumGPU>(n, 1024u);  // no in-kernel barrier -> uncapped
  const auto stream = LaunchKernel::resolve_device(device);
  LaunchKernel(num_blocks, 1024u, stream)(
      inkling_two_shot_all_reduce_kernel<DType, kNumGPU>,
      reinterpret_cast<void* const*>(peer_ptrs_dev),
      static_cast<uint32_t>(rank),
      n);
}

template <typename DType, uint32_t kNumGPU>
void inkling_two_shot_all_reduce_fused(
    tvm::ffi::TensorView local_buffer,
    int64_t data_ptrs_dev,
    int64_t flag_ptrs_dev,
    int64_t state_ptr,
    int64_t rank,
    int64_t num_items,
    int64_t nb_override,
    int64_t bs_override,
    tvm::ffi::TensorView shared) {
  using namespace host;
  uint32_t n;
  validate<DType, kNumGPU>(local_buffer, data_ptrs_dev, rank, num_items, n);
  RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
  RuntimeCheck(state_ptr != 0, "state_ptr is null");
  const DType* shared_ptr = shared_ptr_or_null<DType>(shared, num_items);
  const auto device = local_buffer.device();
  const auto kernel = inkling_two_shot_all_reduce_fused_kernel<DType, kNumGPU>;
  const uint32_t block_size = bs_override > 0 ? static_cast<uint32_t>(bs_override) : 1024u;
  const uint32_t cap = max_resident_blocks(kernel, block_size, device);
  const uint32_t num_blocks = nb_override > 0 ? min(static_cast<uint32_t>(nb_override), cap)
                                              : min(work_num_blocks<DType, kNumGPU>(n, block_size), cap);
  const auto stream = LaunchKernel::resolve_device(device);
  LaunchKernel(num_blocks, block_size, stream)(
      kernel,
      reinterpret_cast<void* const*>(data_ptrs_dev),
      reinterpret_cast<void* const*>(flag_ptrs_dev),
      reinterpret_cast<uint32_t*>(state_ptr),
      shared_ptr,
      static_cast<uint32_t>(rank),
      n);
}

template <typename DType, uint32_t kNumGPU>
void inkling_multimem_one_shot_fused(
    tvm::ffi::TensorView local_buffer,
    int64_t multicast_ptr,
    int64_t flag_ptrs_dev,
    int64_t state_ptr,
    int64_t rank,
    int64_t num_items,
    int64_t nb_override,
    int64_t bs_override,
    int64_t per_block_barrier,
    tvm::ffi::TensorView shared) {
  using namespace host;
  uint32_t n;
  // validate uses the local buffer view only for device/dtype/shape; the kernel
  // operates on the multicast pointer (plus the local view for the shared fold).
  validate<DType, kNumGPU>(local_buffer, multicast_ptr, rank, num_items, n);
  RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
  RuntimeCheck(state_ptr != 0, "state_ptr is null");
  RuntimeCheck(multicast_ptr % 16 == 0, "multicast_ptr not 16B aligned");
  const DType* shared_ptr = shared_ptr_or_null<DType>(shared, num_items);
  const auto device = local_buffer.device();
  const auto kernel = per_block_barrier ? inkling_multimem_one_shot_fused_kernel<DType, kNumGPU, true>
                                        : inkling_multimem_one_shot_fused_kernel<DType, kNumGPU, false>;
  const uint32_t block_size = bs_override > 0 ? static_cast<uint32_t>(bs_override) : 1024u;
  uint32_t cap = max_resident_blocks(kernel, block_size, device);
  if (per_block_barrier) cap = min(cap, inkling_ar::kMaxBarrierBlocks);
  const uint32_t num_blocks = nb_override > 0 ? min(static_cast<uint32_t>(nb_override), cap)
                                              : min(work_num_blocks<DType, kNumGPU>(n, block_size), cap);
  const auto stream = LaunchKernel::resolve_device(device);
  LaunchKernel(num_blocks, block_size, stream)(
      kernel,
      reinterpret_cast<DType*>(multicast_ptr),
      reinterpret_cast<DType*>(local_buffer.data_ptr()),
      reinterpret_cast<void* const*>(flag_ptrs_dev),
      reinterpret_cast<uint32_t*>(state_ptr),
      shared_ptr,
      static_cast<uint32_t>(rank),
      n);
}

template <typename DType, uint32_t kNumGPU>
void inkling_multimem_push_oneshot(
    tvm::ffi::TensorView in_buffer,
    tvm::ffi::TensorView out_buffer,
    int64_t mc_stage_ptr,
    int64_t local_stage_ptr,
    int64_t flag_ptrs_dev,
    int64_t state_ptr,
    int64_t rank,
    int64_t num_items,
    int64_t nb_override,
    int64_t bs_override,
    int64_t per_block_barrier,
    tvm::ffi::TensorView shared) {
  using namespace host;
  uint32_t n;
  // in_buffer is any LOCAL contiguous bf16 tensor (need not be a symm buffer);
  // validate() covers contiguity/dtype/alignment; mc_stage stands in for the
  // pointer null check.
  validate<DType, kNumGPU>(in_buffer, mc_stage_ptr, rank, num_items, n);
  const DType* shared_ptr = shared_ptr_or_null<DType>(shared, num_items);
  RuntimeCheck(out_buffer.IsContiguous(), "out must be contiguous");
  RuntimeCheck(is_type<DType>(out_buffer.dtype()), "out dtype mismatch");
  RuntimeCheck(out_buffer.numel() >= num_items, "out smaller than num_items");
  RuntimeCheck(std::bit_cast<intptr_t>(out_buffer.data_ptr()) % 16 == 0, "out not 16B aligned");
  RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
  RuntimeCheck(state_ptr != 0, "state_ptr is null");
  RuntimeCheck(local_stage_ptr != 0, "local_stage_ptr is null");
  RuntimeCheck(mc_stage_ptr % 16 == 0, "mc_stage_ptr not 16B aligned");
  RuntimeCheck(local_stage_ptr % 16 == 0, "local_stage_ptr not 16B aligned");
  const auto device = in_buffer.device();
  const auto kernel = per_block_barrier ? inkling_multimem_push_oneshot_kernel<DType, kNumGPU, true>
                                        : inkling_multimem_push_oneshot_kernel<DType, kNumGPU, false>;
  const uint32_t block_size = bs_override > 0 ? static_cast<uint32_t>(bs_override) : 1024u;
  uint32_t cap = max_resident_blocks(kernel, block_size, device);
  // The per-block barrier has kMaxBarrierBlocks flag/epoch slots per rank.
  if (per_block_barrier) cap = min(cap, inkling_ar::kMaxBarrierBlocks);
  const uint32_t num_blocks = nb_override > 0 ? min(static_cast<uint32_t>(nb_override), cap)
                                              : min(full_range_num_blocks<DType>(n, block_size), cap);
  const auto stream = LaunchKernel::resolve_device(device);
  LaunchKernel(num_blocks, block_size, stream)(
      kernel,
      reinterpret_cast<const DType*>(in_buffer.data_ptr()),
      reinterpret_cast<DType*>(mc_stage_ptr),
      reinterpret_cast<const DType*>(local_stage_ptr),
      reinterpret_cast<DType*>(out_buffer.data_ptr()),
      reinterpret_cast<void* const*>(flag_ptrs_dev),
      reinterpret_cast<uint32_t*>(state_ptr),
      shared_ptr,
      static_cast<uint32_t>(rank),
      n);
}

template <typename DType, uint32_t kNumGPU>
void inkling_multimem_full_oneshot(
    tvm::ffi::TensorView in_buffer,
    tvm::ffi::TensorView out_buffer,
    int64_t multicast_ptr,
    int64_t flag_ptrs_dev,
    int64_t state_ptr,
    int64_t rank,
    int64_t num_items,
    int64_t nb_override,
    int64_t bs_override,
    tvm::ffi::TensorView shared) {
  using namespace host;
  uint32_t n;
  validate<DType, kNumGPU>(in_buffer, multicast_ptr, rank, num_items, n);
  RuntimeCheck(out_buffer.IsContiguous(), "out must be contiguous");
  RuntimeCheck(is_type<DType>(out_buffer.dtype()), "out dtype mismatch");
  RuntimeCheck(out_buffer.numel() >= num_items, "out smaller than num_items");
  RuntimeCheck(std::bit_cast<intptr_t>(out_buffer.data_ptr()) % 16 == 0, "out not 16B aligned");
  RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
  RuntimeCheck(state_ptr != 0, "state_ptr is null");
  RuntimeCheck(multicast_ptr % 16 == 0, "multicast_ptr not 16B aligned");
  const DType* shared_ptr = shared_ptr_or_null<DType>(shared, num_items);
  const auto device = in_buffer.device();
  const auto kernel = inkling_multimem_full_oneshot_kernel<DType, kNumGPU>;
  const uint32_t block_size = bs_override > 0 ? static_cast<uint32_t>(bs_override) : 1024u;
  const uint32_t cap = max_resident_blocks(kernel, block_size, device);
  const uint32_t num_blocks = nb_override > 0 ? min(static_cast<uint32_t>(nb_override), cap)
                                              : min(full_range_num_blocks<DType>(n, block_size), cap);
  const auto stream = LaunchKernel::resolve_device(device);
  LaunchKernel(num_blocks, block_size, stream)(
      kernel,
      reinterpret_cast<DType*>(multicast_ptr),
      reinterpret_cast<DType*>(in_buffer.data_ptr()),
      reinterpret_cast<DType*>(out_buffer.data_ptr()),
      reinterpret_cast<void* const*>(flag_ptrs_dev),
      reinterpret_cast<uint32_t*>(state_ptr),
      shared_ptr,
      static_cast<uint32_t>(rank),
      n);
}

}  // namespace
