// K3 SP-MoE row-sharded collectives (bf16):
//
//   reduce_scatter_res:
//     [world * rows, hidden] -> [rows, hidden], optionally adding the
//     destination rank's residual rows in the reduction epilogue.
//     Every input vector is written exactly once, to the rank that owns its
//     row shard; the destination polls and reduces the world producer slots.
//
//   all_gather:
//     [rows, hidden] -> [world * rows, hidden].  Every rank multicast-stores
//     its local shard once, then every peer polls the rank slots and copies
//     them into rank-concatenated row order.
//
// Both kernels reuse CustomAllReduceV2's double-buffered push workspace and its
// phase word, so calls remain protocol-compatible with all-reduce and gemm_ag.

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/extra/stl.h>

namespace sglang {

namespace sp_collective {

using device::distributed::EpochCounter;
using device::distributed::Semaphore;

struct Params {
  const uint8_t* input;
  uint8_t* output;
  const uint8_t* residual;
  uint8_t* push_workspaces[device::distributed::kMaxWorldSize];
  uint8_t* push_ws_mc;
  EpochCounter* counter;
  Semaphore* sem_local;
  Semaphore* sem_mc;
  uint8_t* input_mc;
  uint8_t* output_mc;
  int64_t stride_bytes;
  uint32_t rank;
  uint32_t local_vecs;
  uint32_t residual_is_local;
};

/// The 16 B staging vector, viewed as the 4 u32 words the lamport marker
/// protocol tests. See LamportTrait in distributed/communicator.cuh.
using Lamport = device::distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;

template <typename Vec>
SGL_DEVICE Vec empty_vec() {
  Vec vec;
  Lamport::fill_pos_zero(vec.data());
  return vec;
}

/// kPushed: the producer already wrote its rows into every owner's slot (a fused producer such as the MoE finalize
/// with `push` set), so only the polling reduction runs.
template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL, bool kPushed = false>
__global__ void reduce_scatter_res_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B

  device::PDLWaitPrimary<kUsePDL>();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t num_threads = gridDim.x * blockDim.x;
  const uint32_t phase = params.counter->load();
  const auto phase_offset = phase * kWorldSize * params.stride_bytes;
  const auto producer_offset = phase_offset + params.rank * params.stride_bytes;

  // Each vector goes only to the rank that owns its row shard.
  for (uint32_t vid = tid; !kPushed && vid < kWorldSize * params.local_vecs; vid += num_threads) {
    const uint32_t dst_rank = vid / params.local_vecs;
    const uint32_t local_vid = vid - dst_rank * params.local_vecs;
    vec_t vec;
    device::ptx::ld_global_16B(vec, params.input, vid);
    Lamport::clear_pos_zero(vec.data());
    device::ptx::st_relaxed_16B(vec, params.push_workspaces[dst_rank] + producer_offset, local_vid);
  }

  device::PDLTriggerSecondary<kUsePDL>();

  // Poll this rank's shard from all producer slots and reduce locally.
  const auto poll_base = params.push_workspaces[params.rank] + phase_offset;
  const auto residual_base = params.residual + (params.residual_is_local ? 0 : params.rank * params.local_vecs * 16);
  const auto zero = empty_vec<vec_t>();
  for (uint32_t vid = tid; vid < params.local_vecs; vid += num_threads) {
    vec_t vec[kWorldSize + kHasResidual];
    if constexpr (kHasResidual) {
      device::ptx::ld_global_16B(vec[kWorldSize], residual_base, vid);
    }
    do {
      bool empty = false;
#pragma unroll
      for (uint32_t rank = 0; rank < kWorldSize; ++rank) {
        device::ptx::ld_relaxed_16B(vec[rank], poll_base + rank * params.stride_bytes, vid);
        empty |= Lamport::has_pos_zero(vec[rank].data());
      }
      if (!empty) break;
    } while (true);
    const auto out = device::reduce_vec(vec);
    device::ptx::st_global_16B(out, params.output, vid);
#pragma unroll
    for (uint32_t rank = 0; rank < kWorldSize; ++rank) {
      device::ptx::st_global_16B(zero, poll_base + rank * params.stride_bytes, vid);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) params.counter->commit();
}

template <uint32_t kWorldSize, bool kUsePDL>
__global__ void all_gather_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B

  device::PDLWaitPrimary<kUsePDL>();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t num_threads = gridDim.x * blockDim.x;
  const uint32_t phase = params.counter->load();
  const auto phase_offset = phase * kWorldSize * params.stride_bytes;
  const auto producer_offset = phase_offset + params.rank * params.stride_bytes;

  // One multicast store places this rank's shard in the same slot on peers.
  for (uint32_t vid = tid; vid < params.local_vecs; vid += num_threads) {
    vec_t vec;
    device::ptx::ld_global_16B(vec, params.input, vid);
    Lamport::clear_pos_zero(vec.data());
    device::ptx::st_multimem_16B(vec, params.push_ws_mc + producer_offset, vid);
  }

  device::PDLTriggerSecondary<kUsePDL>();

  const auto poll_base = params.push_workspaces[params.rank] + phase_offset;
  const auto zero = empty_vec<vec_t>();
  for (uint32_t vid = tid; vid < kWorldSize * params.local_vecs; vid += num_threads) {
    const uint32_t src_rank = vid / params.local_vecs;
    const uint32_t local_vid = vid - src_rank * params.local_vecs;
    const auto src = poll_base + src_rank * params.stride_bytes;
    vec_t vec;
    do {
      device::ptx::ld_relaxed_16B(vec, src, local_vid);
    } while (Lamport::has_pos_zero(vec.data()));
    device::ptx::st_global_16B(vec, params.output, vid);
    device::ptx::st_global_16B(zero, src, local_vid);
  }

  __syncthreads();
  if (threadIdx.x == 0) params.counter->commit();
}

// Direct variant: output is multicast-bound symmetric memory.  Each producer
// writes its rank slice straight into every peer's final output, avoiding the
// staging read/copy/clear.  The two pull-semaphore barriers preserve protocol
// compatibility with CustomAllReduceV2 and make the remote writes visible
// before any rank leaves the kernel.
template <uint32_t kWorldSize, bool kUsePDL>
__global__ void all_gather_push_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B

  device::PDLWaitPrimary<kUsePDL>();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t num_threads = gridDim.x * blockDim.x;
  const uint32_t phase = params.counter->load();
  const auto phase_offset = phase * kWorldSize * params.stride_bytes;
  const auto producer_offset = phase_offset + params.rank * params.stride_bytes;

  // Unicast twin of all_gather_kernel for planes without multicast: every 16 B of the shard goes to each peer.
  for (uint32_t vid = tid; vid < kWorldSize * params.local_vecs; vid += num_threads) {
    const uint32_t dst_rank = vid / params.local_vecs;
    const uint32_t local_vid = vid - dst_rank * params.local_vecs;
    vec_t vec;
    device::ptx::ld_global_16B(vec, params.input, local_vid);
    Lamport::clear_pos_zero(vec.data());
    device::ptx::st_relaxed_16B(vec, params.push_workspaces[dst_rank] + producer_offset, local_vid);
  }

  device::PDLTriggerSecondary<kUsePDL>();

  const auto poll_base = params.push_workspaces[params.rank] + phase_offset;
  const auto zero = empty_vec<vec_t>();
  for (uint32_t vid = tid; vid < kWorldSize * params.local_vecs; vid += num_threads) {
    const uint32_t src_rank = vid / params.local_vecs;
    const uint32_t local_vid = vid - src_rank * params.local_vecs;
    const auto src = poll_base + src_rank * params.stride_bytes;
    vec_t vec;
    do {
      device::ptx::ld_relaxed_16B(vec, src, local_vid);
    } while (Lamport::has_pos_zero(vec.data()));
    device::ptx::st_global_16B(vec, params.output, vid);
    device::ptx::st_global_16B(zero, src, local_vid);
  }

  __syncthreads();
  if (threadIdx.x == 0) params.counter->commit();
}

#ifdef USE_ROCM
/// gfx1250 forms of `reduce_scatter_res_kernel` / `all_gather_push_kernel` with the
/// 2shot_push_unrolled memory pipeline (distributed/two_shot_push_unrolled.cuh): warps split into
/// one group per peer, so all links carry traffic at once instead of every rank sending to the
/// same destination rank in turn; warps move contiguous hunks of kWarpThreads * unroll vectors
/// with every load of a hunk issued before any store; a late peer is re-polled alone with a
/// sleep per spin; and the rank's own rows never pass through its slot.
inline constexpr uint32_t kUnrolledThreads = 512;
inline constexpr uint32_t kUnrolledUnroll = 8;
inline constexpr uint32_t kUnrolledReduceUnroll = 4;

struct PeerGroup {
  uint32_t lane, warp, num_warps, tid, num_threads, peer, group_warp, group_warps, group_tid, group_stride;

  template <uint32_t kWorldSize>
  SGL_DEVICE static PeerGroup make(uint32_t rank) {
    constexpr uint32_t kFan = kWorldSize - 1;
    PeerGroup g;
    g.lane = threadIdx.x % device::kWarpThreads;
    g.warp = blockIdx.x + gridDim.x * (threadIdx.x / device::kWarpThreads);
    g.num_threads = blockDim.x * gridDim.x;
    g.num_warps = g.num_threads / device::kWarpThreads;
    g.tid = g.warp * device::kWarpThreads + g.lane;
    g.peer = (rank + 1 + g.warp % kFan) % kWorldSize;
    g.group_warp = g.warp / kFan;
    g.group_warps = g.num_warps / kFan + (g.warp % kFan < g.num_warps % kFan ? 1 : 0);
    g.group_tid = g.group_warp * device::kWarpThreads + g.lane;
    g.group_stride = g.group_warps * device::kWarpThreads;
    return g;
  }
};

/// `src[0, len)` to a peer's slot, as Lamport payload.
template <typename vec_t>
SGL_DEVICE void push_rows_unrolled(const PeerGroup& g, const vec_t* src, vec_t* dst, uint32_t len) {
  constexpr uint32_t kHunk = device::kWarpThreads * kUnrolledUnroll;
  const auto full = len / kHunk * kHunk;
  for (uint32_t base = g.group_warp * kHunk; base < full; base += g.group_warps * kHunk) {
    const auto j = base + g.lane;
    vec_t vecs[kUnrolledUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnrolledUnroll; ++u) {
      vecs[u] = src[j + u * device::kWarpThreads];
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (uint32_t u = 0; u < kUnrolledUnroll; ++u) {
      Lamport::clear_pos_zero(vecs[u].data());
    }
    device::ptx::st_relaxed_run_16B<device::kWarpThreads, kUnrolledUnroll>(vecs, dst + j);
  }
  for (auto j = full + g.group_tid; j < len; j += g.group_stride) {
    vec_t vec = src[j];
    Lamport::clear_pos_zero(vec.data());
    device::ptx::st_relaxed_16B(vec, dst, j);
  }
}

/// kPushed: as `reduce_scatter_res_kernel`'s, the producer filled every slot (its own rank's too) and only the
/// reduction runs.
template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL, bool kPushed = false>
__global__
__launch_bounds__(kUnrolledThreads, 1) void reduce_scatter_res_unrolled_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B
  constexpr uint32_t kUnroll = kUnrolledReduceUnroll;
  constexpr uint32_t kHunk = device::kWarpThreads * kUnroll;
  constexpr uint32_t kSlots = kWorldSize + kHasResidual;

  device::PDLWaitPrimary<kUsePDL>();

  const auto r = params.rank;
  const auto len = params.local_vecs;
  const auto g = PeerGroup::make<kWorldSize>(r);
  const uint32_t phase = params.counter->load();
  const auto phase_offset = phase * kWorldSize * params.stride_bytes;
  const auto input = reinterpret_cast<const vec_t*>(params.input);

  // My partial of the peer's row shard goes to the peer's slot for this rank.
  if constexpr (!kPushed) {
    push_rows_unrolled(
        g,
        input + g.peer * len,
        reinterpret_cast<vec_t*>(params.push_workspaces[g.peer] + phase_offset + r * params.stride_bytes),
        len);
  }

  device::PDLTriggerSecondary<kUsePDL>();

  const vec_t* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = reinterpret_cast<const vec_t*>(params.push_workspaces[r] + phase_offset + i * params.stride_bytes);
  }
  const auto mine = input + r * len;
  const auto residual =
      reinterpret_cast<const vec_t*>(params.residual + (params.residual_is_local ? 0 : int64_t{r} * len * 16));
  const auto output = reinterpret_cast<vec_t*>(params.output);
  const auto zero = empty_vec<vec_t>();

  // Sum this rank's shard in rank order (then the residual), as `reduce_scatter_res_kernel` does.
  const auto full = len / kHunk * kHunk;
  for (uint32_t base = g.warp * kHunk; base < full; base += g.num_warps * kHunk) {
    const auto j = base + g.lane;
    vec_t vecs[kUnroll][kSlots];
    vec_t own[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if constexpr (!kPushed) own[u] = mine[j + u * device::kWarpThreads];
      if constexpr (kHasResidual) {
        vecs[u][kWorldSize] = residual[j + u * device::kWarpThreads];
      }
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        device::ptx::ld_relaxed_16B(vecs[u][i], poll_ptrs[i], j + u * device::kWarpThreads);
      }
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      while (true) {
        bool has_zero = false;
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          has_zero |= ((kPushed || i != r) & Lamport::has_pos_zero(vecs[u][i].data()));
        }
        if (!has_zero) break;
        __builtin_amdgcn_s_sleep(1);
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          device::ptx::ld_relaxed_16B(vecs[u][i], poll_ptrs[i], j + u * device::kWarpThreads);
        }
      }
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (!kPushed && i == r) vecs[u][i] = own[u];
      }
      output[j + u * device::kWarpThreads] = device::reduce_vec(vecs[u]);
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      if (kPushed || i != r) {
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          device::ptx::st_global_16B(zero, const_cast<vec_t*>(poll_ptrs[i]) + j, u * device::kWarpThreads);
        }
      }
    }
  }
  for (auto j = full + g.tid; j < len; j += g.num_threads) {
    vec_t vec[kSlots];
    if constexpr (kHasResidual) {
      vec[kWorldSize] = residual[j];
    }
    vec_t own;
    if constexpr (!kPushed) own = mine[j];
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (kPushed || i != r) device::ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], j);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        has_zero |= ((kPushed || i != r) & Lamport::has_pos_zero(vec[i].data()));
      }
      if (!has_zero) break;
    } while (true);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      if (!kPushed && i == r) vec[i] = own;
    }
    output[j] = device::reduce_vec(vec);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      if (kPushed || i != r) device::ptx::st_global_16B(zero, const_cast<vec_t*>(poll_ptrs[i]), j);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) params.counter->commit();
}

#endif  // USE_ROCM

template <uint32_t kWorldSize, bool kUsePDL>
__global__ void all_gather_direct_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B

  const auto barrier = device::distributed::McBarrier(params.sem_local, params.sem_mc, kWorldSize);
  device::PDLWaitPrimary<kUsePDL>();
  barrier.arrive_relaxed();
  __syncthreads();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t step = gridDim.x * blockDim.x;
  const uint32_t dst_bias = params.rank * params.local_vecs;
  for (uint32_t vid = tid; vid < params.local_vecs; vid += step) {
    vec_t vec;
    device::ptx::ld_global_16B(vec, params.input, vid);
    device::ptx::st_multimem_16B(vec, params.output_mc, dst_bias + vid);
  }

  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq();
}

// NVLS pull variant: o_proj writes its TP-partial result into multicast-bound
// symmetric memory. Each rank reduces only its owned row shard directly from
// the multicast alias, so no staging or second broadcast is needed.
template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL>
__global__ void reduce_scatter_pull_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B
  using SumOp = device::ReductionTrait<device::ReductionOp::SUM, bf16x2_t>;

  const auto barrier = device::distributed::McBarrier(params.sem_local, params.sem_mc, kWorldSize);
  device::PDLWaitPrimary<kUsePDL>();
  barrier.arrive_relaxed();
  __syncthreads();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t step = gridDim.x * blockDim.x;
  const auto* input_mc = params.input_mc + params.rank * params.local_vecs * 16;
  const auto* residual =
      kHasResidual ? params.residual + (params.residual_is_local ? 0 : params.rank * params.local_vecs * 16) : nullptr;
  for (uint32_t vid = tid; vid < params.local_vecs; vid += step) {
    vec_t vec;
    device::ptx::ld_multimem_16B(vec, input_mc, vid);
    if constexpr (kHasResidual) {
      vec_t res;
      device::ptx::ld_global_16B(res, residual, vid);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        vec[j] = SumOp::reduce(vec[j], res[j]);
      }
    }
    device::ptx::st_global_16B(vec, params.output, vid);
  }

  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq();
}

}  // namespace sp_collective
using host::distributed::CommunicatorRef;

template <uint32_t kWorldSize, bool kUsePDL>
struct SPCollectiveKernel {
  using TensorView = tvm::ffi::TensorView;

  /// Tensor validation shared by every variant. Plane pointers are bound
  /// separately: the push variants stage through the push plane and never
  /// touch a semaphore, the pull variants barrier on the pull plane's
  /// semaphores and never touch the push workspace.
  static sp_collective::Params make_params(
      const host::distributed::CommunicatorObj& comm,
      TensorView input,
      TensorView output,
      std::optional<TensorView> residual,
      bool residual_is_local) {
    using namespace host;
    auto input_elems = SymbolicSize{"input_elems"};
    auto local_elems = SymbolicSize{"local_elems"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({input_elems}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({local_elems}).with_dtype<bf16_t>().with_device(device).verify(output);
    if (residual.has_value()) {
      if (residual_is_local) {
        TensorMatcher({local_elems}).with_dtype<bf16_t>().with_device(device).verify(residual.value());
      } else {
        TensorMatcher({input_elems}).with_dtype<bf16_t>().with_device(device).verify(residual.value());
      }
    }
    CHECK_HOST(comm.get_world_size() == kWorldSize);
    CHECK_HOST(local_elems.unwrap() > 0);
    CHECK_HOST(input_elems.unwrap() == local_elems.unwrap() * kWorldSize);
    CHECK_HOST(local_elems.unwrap() % 8 == 0) << "local shard bytes must be 16B aligned";

    return sp_collective::Params{
        .input = static_cast<const uint8_t*>(input.data_ptr()),
        .output = static_cast<uint8_t*>(output.data_ptr()),
        .residual = residual.has_value() ? static_cast<const uint8_t*>(residual.value().data_ptr()) : nullptr,
        .push_workspaces = {},
        .push_ws_mc = nullptr,
        .counter = nullptr,
        .sem_local = nullptr,
        .sem_mc = nullptr,
        .input_mc = nullptr,
        .output_mc = nullptr,
        .stride_bytes = 0,
        .rank = comm.get_rank(),
        .local_vecs = static_cast<uint32_t>(local_elems.unwrap() * sizeof(bf16_t) / 16),
        .residual_is_local = static_cast<uint32_t>(residual_is_local),
    };
  }

  static void bind_push(sp_collective::Params& params, const host::distributed::PushPlaneObj& push) {
    CHECK_HOST(int64_t(params.local_vecs) * 16 <= push.slot_bytes) << "local shard exceeds a push slot";
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      params.push_workspaces[i] = push.workspaces[i];
    }
    params.push_ws_mc = push.mc_workspace;
    params.counter = push.counter;
    params.stride_bytes = push.slot_bytes;
  }

  static void bind_pull(sp_collective::Params& params, const host::distributed::PullPlaneObj& pull) {
    CHECK_HOST(pull.mc_semaphore != nullptr) << "the pull path needs a multicast-capable pull plane";
    params.sem_local = pull.semaphores[pull.rank];
    params.sem_mc = pull.mc_semaphore;
  }

  static void check_push_launch(int64_t num_blocks, int64_t block_size) {
    CHECK_HOST(num_blocks > 0);
    // The RS reduction keeps one 16B vector per producer in registers.
    // 1024-thread CTAs exceed the GB300 launch resource limit.
    CHECK_HOST(block_size >= 32 && block_size <= 512 && block_size % 32 == 0);
  }

  static void check_pull_launch(uint32_t max_blocks, int64_t num_blocks, int64_t block_size) {
    CHECK_HOST(num_blocks > 0 && num_blocks <= max_blocks);
    CHECK_HOST(block_size >= 32 && block_size <= 1024 && block_size % 32 == 0);
  }

  static void reduce_scatter_res(
      CommunicatorRef ref,
      TensorView input,
      TensorView output,
      std::optional<TensorView> residual,
      bool residual_is_local,
      int64_t num_blocks,
      int64_t block_size) {
    const auto& push = ref.get()->get_push_obj();
    check_push_launch(num_blocks, block_size);
    auto params = make_params(*ref.get(), input, output, residual, residual_is_local);
    bind_push(params, push);
    const auto kernel = residual.has_value() ? sp_collective::reduce_scatter_res_kernel<kWorldSize, true, kUsePDL>
                                             : sp_collective::reduce_scatter_res_kernel<kWorldSize, false, kUsePDL>;
    host::LaunchKernel(num_blocks, block_size, input.device()).enable_pdl(kUsePDL)(kernel, params);
  }

#ifdef USE_ROCM
  /// The unrolled kernels' blocks poll for pushes that other blocks of the same grid make, so the whole grid must
  /// be resident at once: clamp it to the kernel's occupancy times the CU count.
  template <typename Kernel>
  static int64_t unrolled_blocks(Kernel kernel, int64_t num_blocks, int64_t block_size, int device_id) {
    CHECK_HOST(block_size >= 32 && block_size <= sp_collective::kUnrolledThreads && block_size % 32 == 0);
    const auto resident = int64_t{host::runtime::get_blocks_per_sm(kernel, static_cast<int32_t>(block_size))} *
                          host::runtime::get_sm_count(device_id);
    num_blocks = std::min(num_blocks, resident);
    // every peer group needs at least one warp
    CHECK_HOST(num_blocks * block_size / 32 >= int64_t{kWorldSize} - 1);
    return num_blocks;
  }

  static void reduce_scatter_res_unrolled(
      CommunicatorRef ref,
      TensorView input,
      TensorView output,
      std::optional<TensorView> residual,
      bool residual_is_local,
      int64_t num_blocks,
      int64_t block_size) {
    const auto& push = ref.get()->get_push_obj();
    auto params = make_params(*ref.get(), input, output, residual, residual_is_local);
    bind_push(params, push);
    const auto kernel = residual.has_value()
                            ? sp_collective::reduce_scatter_res_unrolled_kernel<kWorldSize, true, kUsePDL>
                            : sp_collective::reduce_scatter_res_unrolled_kernel<kWorldSize, false, kUsePDL>;
    num_blocks = unrolled_blocks(kernel, num_blocks, block_size, input.device().device_id);
    host::LaunchKernel(num_blocks, block_size, input.device()).enable_pdl(kUsePDL)(kernel, params);
  }
#endif

  /// The reduction half of `reduce_scatter_res` for rows a producer pushed itself (`reduce_scatter_res_kernel` with
  /// kPushed): `output` = this rank's shard, every rank's slot of the current phase holding its partial rows.
  static void
  reduce_scatter_pushed(CommunicatorRef ref, TensorView output, int64_t num_blocks, int64_t block_size, bool unrolled) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();
    check_push_launch(num_blocks, block_size);
    auto elems = SymbolicSize{"local_elems"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({elems}).with_dtype<bf16_t>().with_device(device).verify(output);
    CHECK_HOST(ref.get()->get_world_size() == kWorldSize);
    CHECK_HOST(elems.unwrap() > 0 && elems.unwrap() % 8 == 0) << "local shard bytes must be 16B aligned";
    sp_collective::Params params{};
    params.output = static_cast<uint8_t*>(output.data_ptr());
    params.rank = ref.get()->get_rank();
    params.local_vecs = static_cast<uint32_t>(elems.unwrap() * sizeof(bf16_t) / 16);
    bind_push(params, push);
#ifdef USE_ROCM
    if (unrolled) {
      const auto kernel = sp_collective::reduce_scatter_res_unrolled_kernel<kWorldSize, false, kUsePDL, true>;
      num_blocks = unrolled_blocks(kernel, num_blocks, block_size, output.device().device_id);
      host::LaunchKernel(num_blocks, block_size, output.device()).enable_pdl(kUsePDL)(kernel, params);
      return;
    }
#endif
    host::LaunchKernel(num_blocks, block_size, output.device())
        .enable_pdl(kUsePDL)(sp_collective::reduce_scatter_res_kernel<kWorldSize, false, kUsePDL, true>, params);
  }

  static void
  all_gather(CommunicatorRef ref, TensorView input, TensorView output, int64_t num_blocks, int64_t block_size) {
    const auto& push = ref.get()->get_push_obj();
    check_push_launch(num_blocks, block_size);
    // Reuse the RS matcher by swapping input/output roles conceptually.
    auto params = make_params(*ref.get(), output, input, std::nullopt, false);
    bind_push(params, push);
    CHECK_HOST(params.push_ws_mc != nullptr) << "all-gather requires a multicast-capable push plane";
    params.input = static_cast<const uint8_t*>(input.data_ptr());
    params.output = static_cast<uint8_t*>(output.data_ptr());
    host::LaunchKernel(num_blocks, block_size, input.device())
        .enable_pdl(kUsePDL)(sp_collective::all_gather_kernel<kWorldSize, kUsePDL>, params);
  }

  static void
  all_gather_push(CommunicatorRef ref, TensorView input, TensorView output, int64_t num_blocks, int64_t block_size) {
    const auto& push = ref.get()->get_push_obj();
    check_push_launch(num_blocks, block_size);
    auto params = make_params(*ref.get(), output, input, std::nullopt, false);
    bind_push(params, push);
    params.input = static_cast<const uint8_t*>(input.data_ptr());
    params.output = static_cast<uint8_t*>(output.data_ptr());
    host::LaunchKernel(num_blocks, block_size, input.device())
        .enable_pdl(kUsePDL)(sp_collective::all_gather_push_kernel<kWorldSize, kUsePDL>, params);
  }

  static void all_gather_direct(
      CommunicatorRef ref,
      TensorView input,
      TensorView output,
      int64_t output_mc_ptr,
      int64_t num_blocks,
      int64_t block_size) {
    const auto& pull = ref.get()->get_pull_obj();
    CHECK_HOST(output_mc_ptr != 0) << "direct all-gather needs symmetric output";
    check_pull_launch(ref.get()->get_pull_blocks(), num_blocks, block_size);
    auto params = make_params(*ref.get(), output, input, std::nullopt, false);
    bind_pull(params, pull);
    params.input = static_cast<const uint8_t*>(input.data_ptr());
    params.output = static_cast<uint8_t*>(output.data_ptr());
    params.output_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(output_mc_ptr));
    host::LaunchKernel(num_blocks, block_size, input.device())
        .enable_pdl(kUsePDL)(sp_collective::all_gather_direct_kernel<kWorldSize, kUsePDL>, params);
  }

  static void reduce_scatter_pull(
      CommunicatorRef ref,
      TensorView input,
      TensorView output,
      std::optional<TensorView> residual,
      bool residual_is_local,
      int64_t input_mc_ptr,
      int64_t num_blocks,
      int64_t block_size) {
    const auto& pull = ref.get()->get_pull_obj();
    CHECK_HOST(input_mc_ptr != 0) << "pull RS needs symmetric input";
    check_pull_launch(ref.get()->get_pull_blocks(), num_blocks, block_size);
    auto params = make_params(*ref.get(), input, output, residual, residual_is_local);
    bind_pull(params, pull);
    params.input_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(input_mc_ptr));
    const auto kernel = residual.has_value() ? sp_collective::reduce_scatter_pull_kernel<kWorldSize, true, kUsePDL>
                                             : sp_collective::reduce_scatter_pull_kernel<kWorldSize, false, kUsePDL>;
    host::LaunchKernel(num_blocks, block_size, input.device()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
