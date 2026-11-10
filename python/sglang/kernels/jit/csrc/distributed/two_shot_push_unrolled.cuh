// gfx1250 (MI450) 2shot_push for large tensors, included by custom_all_reduce.cuh on
// ROCm: the same algorithm as `all_reduce_2shot_push_kernel` with a deeper memory
// pipeline. `AllReduceKernel` picks it by size.
#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <cstdint>

namespace sglang {

/// Threads per unrolled 2shot_push block, and 16-byte packs each lane keeps in
/// flight in its scatter and gather phases. The reduce phase holds a pack from
/// every rank per unroll step, so it unrolls less as the world grows.
inline constexpr uint32_t kTwoShotPushThreads = 256;
inline constexpr uint32_t kTwoShotPushUnroll = 8;
template <uint32_t kWorldSize>
inline constexpr uint32_t kTwoShotPushReduceUnroll = kWorldSize <= 4 ? 8 : 4;
/// Below these sizes the per-vector kernel wins: one vector per thread spreads a
/// chunk over more warps than whole hunks do (TP2/TP4 unrolled are 10-14% slower
/// at 128 KB - 1 MB), and with one rank arriving late TP2/TP4 stay 3-10% slower
/// up to 4 MB.
template <uint32_t kWorldSize>
inline constexpr int64_t kTwoShotPushUnrolledMinBytes =
    int64_t{
        kWorldSize <= 4  ? 8
        : kWorldSize < 8 ? 2
                         : 1}
    << 20;

/// `all_reduce_2shot_push_kernel` with the ring_simple memory pipeline: same
/// slots, markers and epochs, but warps take contiguous hunks of
/// `kWarpThreads * unroll` vectors and issue every load of a hunk before using
/// any, and slot payload moves as device-scope loads and batched system-scope
/// stores while the input and output take plain accesses. The
/// tail past the last whole hunk takes the per-vector path. Single node only
/// (at most 8 ranks).
template <typename T, uint32_t kWorldSize, bool kUsePDL, typename Params>
__global__ __launch_bounds__(kTwoShotPushThreads, 1) __attribute__((
    amdgpu_waves_per_eu(1, 2))) void all_reduce_2shot_push_unrolled_kernel(const __grid_constant__ Params params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  constexpr uint32_t kFan = kWorldSize - 1;
  constexpr uint32_t kUnroll = kTwoShotPushUnroll;
  constexpr uint32_t kReduceUnroll = kTwoShotPushReduceUnroll<kWorldSize>;
  constexpr uint32_t kHunk = kWarpThreads * kUnroll;
  constexpr uint32_t kReduceHunk = kWarpThreads * kReduceUnroll;
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  using Lamport = distributed::LamportTrait<T, kVecSize * 2, /*kAtom=*/4>;
  const auto r = params.rank;
  const auto num_vecs = params.num_vecs;
  const auto chunk_vecs = div_ceil(num_vecs, kWorldSize);
  const auto get_chunk_len = [&](uint32_t c) {  //
    return min(chunk_vecs, num_vecs - min(num_vecs, c * chunk_vecs));
  };
  const auto input = static_cast<const vec_t*>(params.input);
  const auto output = static_cast<vec_t*>(params.output);
  const auto num_threads = blockDim.x * gridDim.x;
  const auto num_warps = num_threads / kWarpThreads;
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;

  // peer will cover 0..kWorldSize - 1, except for local rank `r`
  const auto peer = (r + 1 + global_warp_id % kFan) % kWorldSize;
  const auto group_warp = global_warp_id / kFan;
  const auto group_warps = num_warps / kFan + (global_warp_id % kFan < num_warps % kFan ? 1 : 0);
  const auto group_tid = group_warp * kWarpThreads + lane_id;
  const auto group_stride = group_warps * kWarpThreads;
  const auto peer_len = get_chunk_len(peer);
  const auto peer_full = peer_len / kHunk * kHunk;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  // reduce-scatter: my partial of the peer's chunk to its owner
  {
    const auto src = input + peer * chunk_vecs;
    const auto push_ptr = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/peer, /*src=*/r));
    for (uint32_t base = group_warp * kHunk; base < peer_full; base += group_warps * kHunk) {
      const auto j = base + lane_id;
      const auto in = src + j;
      vec_t vecs[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        vecs[u] = in[u * kWarpThreads];
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        Lamport::clear_pos_zero(vecs[u].data());
      }
      ptx::st_relaxed_run_16B<kWarpThreads, kUnroll>(vecs, push_ptr + j);
    }
    for (auto j = peer_full + group_tid; j < peer_len; j += group_stride) {
      vec_t vec = src[j];
      Lamport::clear_pos_zero(vec.data());
      ptx::st_relaxed_16B(vec, push_ptr, j);
    }
  }

  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());

  vec_t* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/r, /*src=*/i));
  }

  // reduce my chunk, keep it and fan it out
  {
    const auto my_len = get_chunk_len(r);
    const auto my_full = my_len / kReduceHunk * kReduceHunk;
    const auto my_in = input + r * chunk_vecs;
    const auto my_out = output + r * chunk_vecs;
    for (uint32_t base = global_warp_id * kReduceHunk; base < my_full; base += num_warps * kReduceHunk) {
      const auto j = base + lane_id;
      vec_t vecs[kReduceUnroll][kWorldSize];
      vec_t mine[kReduceUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
        mine[u] = my_in[j + u * kWarpThreads];
      }
      // every rank's slot, the unused (r, r) one too, so no branch on r splits
      // the batch; this rank's own partial is the input, swapped in below
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          ptx::ld_relaxed_16B(vecs[u][i], poll_ptrs[i], j + u * kWarpThreads);
        }
      }
      // A late peer is re-polled alone, with a sleep per spin: re-polling every
      // rank's packs floods the memory the peers' pushes land in.
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        while (true) {
          bool has_zero = false;
#pragma unroll
          for (uint32_t u = 0; u < kReduceUnroll; ++u) {
            has_zero |= ((i != r) & Lamport::has_pos_zero(vecs[u][i].data()));
          }
          if (!has_zero) break;
          __builtin_amdgcn_s_sleep(1);
#pragma unroll
          for (uint32_t u = 0; u < kReduceUnroll; ++u) {
            ptx::ld_relaxed_16B(vecs[u][i], poll_ptrs[i], j + u * kWarpThreads);
          }
        }
      }
      vec_t outs[kReduceUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          if (i == r) vecs[u][i] = mine[u];
        }
        outs[u] = reduce_vec(vecs[u]);
        // remap before the local store too, so every rank holds identical bits
        Lamport::clear_pos_zero(outs[u].data());
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) {
          const auto send_ptr = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/i, /*src=*/r));
          ptx::st_relaxed_run_16B<kWarpThreads, kReduceUnroll>(outs, send_ptr + chunk_vecs + j);
        }
      }
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
        my_out[j + u * kWarpThreads] = outs[u];
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) {
#pragma unroll
          for (uint32_t u = 0; u < kReduceUnroll; ++u) {
            ptx::st_global_16B(pos_zero_vec, poll_ptrs[i] + j, u * kWarpThreads);
          }
        }
      }
    }
    for (auto j = my_full + global_tid; j < my_len; j += num_threads) {
      vec_t vec[kWorldSize];
      const vec_t mine = my_in[j];
      do {
        bool has_zero = false;
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          if (i != r) ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], j);
        }
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          has_zero |= ((i != r) & Lamport::has_pos_zero(vec[i].data()));
        }
        if (!has_zero) break;
      } while (true);
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i == r) vec[i] = mine;
      }
      auto out_vec = reduce_vec(vec);
      Lamport::clear_pos_zero(out_vec.data());
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) ptx::st_relaxed_16B(out_vec, epoch.slot_ptr(/*dst=*/i, /*src=*/r), chunk_vecs + j);
      }
      out_vec.store(my_out, j);
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) ptx::st_global_16B(pos_zero_vec, poll_ptrs[i], j);
      }
    }
  }

  // all-gather: drain the peer's reduced chunk
  {
    const auto poll_ptr = poll_ptrs[peer] + chunk_vecs;
    const auto dst = output + peer * chunk_vecs;
    for (uint32_t base = group_warp * kHunk; base < peer_full; base += group_warps * kHunk) {
      const auto j = base + lane_id;
      const auto at = poll_ptr + j;
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
        dst[j + u * kWarpThreads] = vecs[u];
      }
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        ptx::st_global_16B(pos_zero_vec, at, u * kWarpThreads);
      }
    }
    for (auto j = peer_full + group_tid; j < peer_len; j += group_stride) {
      vec_t vec;
      do {
        ptx::ld_relaxed_16B(vec, poll_ptr, j);
      } while (Lamport::has_pos_zero(vec.data()));
      dst[j] = vec;
      ptx::st_global_16B(pos_zero_vec, poll_ptr, j);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();  // every thread has read the phase before this block commits
  epoch.commit();
}

}  // namespace sglang
