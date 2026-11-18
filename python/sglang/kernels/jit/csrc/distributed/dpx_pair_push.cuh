// The pair stages of CustomAllReduceV2's hier_2shot_push, for ROCm gfx1250 (MI450)
// DPX: each SoC's two partitions reduce-scatter between themselves, every
// partition all-reduces its half with its peers on the other SoCs (a regular
// 2shot_push), and the pair all-gathers the result. This file holds the pair
// reduce-scatter / all-gather; it runs over a 2-rank push plane of its own.
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

#ifdef USE_ROCM
#define cudaGetDevice hipGetDevice
#endif

namespace sglang {

namespace dpx_pair {

using device::distributed::PushWorkSpace;

struct Params {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t num_vecs;  // 16-byte vectors of the full tensor
  uint32_t rank;
  PushWorkSpace<2> ws;
};

/// One half of a 2-rank 2shot_push. Chunk c of the full tensor belongs to rank c.
///   reduce-scatter: `input` is the full tensor, `output` this rank's reduced chunk.
///   all-gather:     `input` is this rank's chunk, `output` the full tensor.
/// Same Lamport markers and epochs as 2shot_push, with a one-chunk slot: each
/// launch both pushes to and polls (and clears) the peer, so the two stages
/// alternate epochs in step on both ranks and may share the slot.
template <typename T, bool kAllGather, bool kUsePDL>
__global__ __launch_bounds__(1024, 1) void pair_push_kernel(const __grid_constant__ Params params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  using Lamport = distributed::LamportTrait<T, kVecSize * 2, /*kAtom=*/4>;
  const auto r = params.rank;
  const auto peer = r ^ 1;
  const auto chunk_vecs = params.num_vecs / 2;
  const auto num_threads = blockDim.x * gridDim.x;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<2>{params.ws};
  const auto push_ptr = epoch.slot_ptr(/*dst=*/peer, /*src=*/r);
  const auto poll_ptr = epoch.slot_ptr(/*dst=*/r, /*src=*/peer);

  for (auto j = global_tid; j < chunk_vecs; j += num_threads) {
    vec_t vec;
    vec.load(params.input, kAllGather ? j : peer * chunk_vecs + j);
    Lamport::clear_pos_zero(vec.data());
    ptx::st_relaxed_16B(vec, push_ptr, j);
    // remap before the local store too, so both ranks hold identical bits
    if constexpr (kAllGather) vec.store(params.output, r * chunk_vecs + j);
  }

  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());
  for (auto j = global_tid; j < chunk_vecs; j += num_threads) {
    // vec[0] arrives from the peer, vec[1] is local: a two-term sum is the same
    // in either order
    vec_t vec[2];
    do {
      ptx::ld_relaxed_16B(vec[0], poll_ptr, j);
    } while (Lamport::has_pos_zero(vec[0].data()));
    if constexpr (kAllGather) {
      vec[0].store(params.output, peer * chunk_vecs + j);
    } else {
      vec[1].load(params.input, r * chunk_vecs + j);
      reduce_vec(vec).store(params.output, j);
    }
    ptx::st_global_16B(pos_zero_vec, poll_ptr, j);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();  // every thread has read the phase before this block commits
  epoch.commit();
}

#ifdef USE_ROCM
/// Threads per unrolled pair-stage block, and 16-byte packs each lane keeps in flight.
inline constexpr uint32_t kUnrolledThreads = 256;
inline constexpr uint32_t kUnroll = 8;
/// Smaller chunks run faster on the per-vector kernel (7-19% up to 1 MB at TP16 across
/// two hosts, same crossover at TP8).
inline constexpr int64_t kUnrolledMinChunkBytes = int64_t{2} << 20;

/// `pair_push_kernel` with the unrolled 2shot_push memory pipeline: warps take
/// contiguous hunks of `kWarpThreads * kUnroll` vectors and issue every load of a
/// hunk before using any, and peer payload goes out as batched system-scope stores.
/// A late peer is re-polled one hunk at a time with a sleep per spin. The tail past
/// the last whole hunk takes the per-vector path.
template <typename T, bool kAllGather, bool kUsePDL>
__global__ __launch_bounds__(kUnrolledThreads, 1)
    __attribute__((amdgpu_waves_per_eu(1, 2))) void pair_push_unrolled_kernel(const __grid_constant__ Params params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  constexpr uint32_t kHunk = kWarpThreads * kUnroll;
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  using Lamport = distributed::LamportTrait<T, kVecSize * 2, /*kAtom=*/4>;
  const auto r = params.rank;
  const auto peer = r ^ 1;
  const auto chunk_vecs = params.num_vecs / 2;
  const auto full = chunk_vecs / kHunk * kHunk;
  const auto input = static_cast<const vec_t*>(params.input);
  const auto output = static_cast<vec_t*>(params.output);
  const auto num_threads = blockDim.x * gridDim.x;
  const auto num_warps = num_threads / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * (threadIdx.x / kWarpThreads);
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<2>{params.ws};
  const auto push_ptr = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/peer, /*src=*/r));
  const auto poll_ptr = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/r, /*src=*/peer));
  // reduce-scatter pushes the peer's chunk and keeps its own; all-gather pushes its chunk
  const auto src = kAllGather ? input : input + peer * chunk_vecs;
  const auto mine = input + r * chunk_vecs;
  const auto local_out = output + r * chunk_vecs;
  const auto peer_out = output + peer * chunk_vecs;

  for (uint32_t base = global_warp_id * kHunk; base < full; base += num_warps * kHunk) {
    const auto j = base + lane_id;
    vec_t vecs[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      vecs[u] = src[j + u * kWarpThreads];
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      Lamport::clear_pos_zero(vecs[u].data());
    }
    ptx::st_relaxed_run_16B<kWarpThreads, kUnroll>(vecs, push_ptr + j);
    // remap before the local store too, so both ranks hold identical bits
    if constexpr (kAllGather) {
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        local_out[j + u * kWarpThreads] = vecs[u];
      }
    }
  }
  for (auto j = full + global_tid; j < chunk_vecs; j += num_threads) {
    vec_t vec = src[j];
    Lamport::clear_pos_zero(vec.data());
    ptx::st_relaxed_16B(vec, push_ptr, j);
    if constexpr (kAllGather) local_out[j] = vec;
  }

  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());
  for (uint32_t base = global_warp_id * kHunk; base < full; base += num_warps * kHunk) {
    const auto j = base + lane_id;
    vec_t local[kUnroll];
    if constexpr (!kAllGather) {
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        local[u] = mine[j + u * kWarpThreads];
      }
    }
    vec_t vecs[kUnroll];
    while (true) {
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        ptx::ld_relaxed_16B(vecs[u], poll_ptr, j + u * kWarpThreads);
      }
      bool has_zero = false;
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        has_zero |= Lamport::has_pos_zero(vecs[u].data());
      }
      if (!has_zero) break;
      __builtin_amdgcn_s_sleep(1);
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if constexpr (kAllGather) {
        peer_out[j + u * kWarpThreads] = vecs[u];
      } else {
        // a two-term sum is the same in either order
        vec_t pair[2] = {vecs[u], local[u]};
        output[j + u * kWarpThreads] = reduce_vec(pair);
      }
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      ptx::st_global_16B(pos_zero_vec, poll_ptr + j, u * kWarpThreads);
    }
  }
  for (auto j = full + global_tid; j < chunk_vecs; j += num_threads) {
    vec_t vec[2];
    do {
      ptx::ld_relaxed_16B(vec[0], poll_ptr, j);
    } while (Lamport::has_pos_zero(vec[0].data()));
    if constexpr (kAllGather) {
      peer_out[j] = vec[0];
    } else {
      vec[1] = mine[j];
      output[j] = reduce_vec(vec);
    }
    ptx::st_global_16B(pos_zero_vec, poll_ptr, j);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();  // every thread has read the phase before this block commits
  epoch.commit();
}
#endif

// The smallest block that keeps the grid within one block per SM.
inline auto choose_block_size(uint32_t num_threads) -> uint32_t {
  static const uint32_t kNumSM = [] {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    return host::runtime::get_sm_count(device);
  }();
  for (const uint32_t block_size : {128u, 256u, 512u}) {
    if (host::div_ceil(num_threads, block_size) <= kNumSM) return block_size;
  }
  return 1024u;
}

}  // namespace dpx_pair

template <typename T, bool kUsePDL>
struct DpxPairPushKernel {
  using Tensor = tvm::ffi::Tensor;

  /// Reduce-scatter (`all_gather` false: the full tensor in, this rank's reduced
  /// half out) or all-gather (this rank's half in, the full tensor out) over a
  /// 2-rank communicator's push plane.
  static Tensor run(const host::distributed::CommunicatorRef comm_ref, const Tensor in, const bool all_gather) {
    using namespace host;
    const auto& comm = *comm_ref.get();
    CHECK_HOST(comm.get_world_size() == 2) << "a pair stage needs a 2-rank communicator: " << comm.get_world_size();
    CHECK_HOST(in.IsContiguous() && is_type<T>(in.dtype()) && in.device().device_type == kDLGPU);
    const auto num_elems = in.numel();
    const auto nbytes = static_cast<int64_t>(num_elems * sizeof(T));
    CHECK_HOST(reinterpret_cast<intptr_t>(in.data_ptr()) % 16 == 0 && nbytes % 16 == 0);
    const auto in_vecs = static_cast<uint32_t>(nbytes / 16);
    CHECK_HOST(int64_t{in_vecs} * 16 == nbytes) << nbytes;
    // both stages see the full tensor's vector count
    const auto full_vecs = all_gather ? 2 * in_vecs : in_vecs;
    CHECK_HOST(full_vecs % 2 == 0) << "a pair reduce-scatter needs an even vector count: " << full_vecs;

    const auto& push = comm.get_push_obj();
    const auto out_elems = all_gather ? 2 * num_elems : num_elems / 2;
    Tensor out = ffi::empty({out_elems}, in.dtype(), in.device());
    const auto chunk_bytes = int64_t{full_vecs / 2} * 16;
    const auto params = dpx_pair::Params{
        .input = in.data_ptr(),
        .output = out.data_ptr(),
        .num_vecs = full_vecs,
        .rank = push.rank,
        .ws = push.get_workspace<2>(chunk_bytes),
    };
    const auto stream = LaunchKernel::resolve_device(in.device());
#ifdef USE_ROCM
    if (chunk_bytes >= dpx_pair::kUnrolledMinChunkBytes) {
      const auto kernel = all_gather ? dpx_pair::pair_push_unrolled_kernel<T, true, kUsePDL>
                                     : dpx_pair::pair_push_unrolled_kernel<T, false, kUsePDL>;
      LaunchKernel(push.default_blocks, dpx_pair::kUnrolledThreads, stream)  //
          .enable_pdl(kUsePDL)(kernel, params);
      return out;
    }
#endif
    const auto kernel =
        all_gather ? dpx_pair::pair_push_kernel<T, true, kUsePDL> : dpx_pair::pair_push_kernel<T, false, kUsePDL>;
    LaunchKernel(push.default_blocks, dpx_pair::choose_block_size(full_vecs / 2), stream)  //
        .enable_pdl(kUsePDL)(kernel, params);
    return out;
  }
};

}  // namespace sglang
