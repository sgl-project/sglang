#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <cuda/cmath>
#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/object.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>

namespace sglang {

using device::distributed::PushWorkSpace;
using device::distributed::Semaphore;

using fast_mod_div_u32_t = cuda::fast_mod_div<uint32_t>;

template <uint32_t kWorldSize>
struct NVLinkCommPushParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t dst_offset;  // AR = rank slot stride; AG = packed token prefix
  uint32_t rank;
  uint32_t num_push_vecs;
  uint32_t num_poll_vecs;
  uint32_t num_vecs_per_token;
  // Ragged split, reduce-scatter only: rank r owns `avg + (r < rem)` tokens of
  // the input starting at `r * avg + min(r, rem)`.
  uint32_t tokens_avg;
  uint32_t tokens_rem;
  fast_mod_div_u32_t vecs_per_token_div;
  PushWorkSpace<kWorldSize> ws;
};

struct NVLinkCommPullParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  uint32_t num_vecs;
  // multicast buffer
  uint8_t* input_mc;
  uint8_t* output_mc;
  Semaphore* sem_local;
  Semaphore* sem_mc;
  uint32_t rank;
  uint32_t world_size;
};

/// Peers drained per polling round in the push reduce. Each group is one
/// sequential wait, so wider is faster -- at world size 8, dropping from 8 to 4
/// costs 4-7% -- and the only thing holding it back is the staging array, at 4
/// registers per peer in the group. This takes the widest value measured
/// spill-free at a 1024-thread CTA: 8 is clean through world size 12 and again
/// at 32, but nicks the stack at 16/24 (8 B) and 72 (24 B), where 6 and 4 are
/// clean. Re-measure both axes when the vector width or CTA size changes.

inline constexpr uint32_t get_poll_group(uint32_t world_size) {
  if (world_size <= 8) return world_size;
  return 8;
}

inline constexpr uint32_t kPushCTASize = 1024;  // max value
inline constexpr uint32_t kPullCTASize = 512;   // max value

#define PUSH_KERNEL __global__ __launch_bounds__(kPushCTASize, 1)
#define PULL_KERNEL __global__ __launch_bounds__(kPullCTASize, 1)

enum Primitive {
  RS = 0b01,     // Reduce-Scatter
  AG = 0b10,     // All-Gather
  AR = RS | AG,  // All-Reduce = RS + AG
};

/**
 * \brief Layout:
 * 1. `AG`/`RS`: each rank push to its own slot
 *  [rank0] | [rank 1] | [rank 2] | ...
 * 2. `AG`: each rank push to a contiguous region
 *  [rank0, rank1, rank2, ...]
 *
 * `RS` use swizzle layout for push kernel \n
 * `AG` use normal linear layout for push kernel
 */
template <typename T, Primitive kPrim, uint32_t kWorldSize, bool kUsePDL>
PUSH_KERNEL void nvlink_push_kernel(const __grid_constant__ NVLinkCommPushParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / sizeof(T);  // 16 bytes per vector
  using vec_t = device::AlignedVector<packed_t<T>, kVecSize / 2>;
  using Lamport = distributed::LamportTrait<T, kVecSize, /*kAtom=*/4>;
  constexpr uint32_t kGroup = get_poll_group(kWorldSize);

  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto num_threads = blockDim.x * gridDim.x;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  void* push_ptrs[kWorldSize];
  /// NOTE: broadcast write is only fast when world size is large
  if constexpr (kWorldSize < 8 && (kPrim & Primitive::AG)) {
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      // Same address arithmetic as the multicast branch below, so the two agree
      // on where a sender's shard lands. `dst_offset` is a slot stride for the
      // all-reduce but a packed token prefix for the gather, whose consumer
      // reads the plane linearly; `slot_ptr(i, rank)` would put the gather's
      // senders `slot_bytes` apart and the poll loop would never see them.
      push_ptrs[i] = static_cast<uint8_t*>(epoch.slot_ptr(/*dst=*/i)) + params.dst_offset;
    }
  }

  const auto dst_ptr_mc = params.ws.mc_workspace + params.dst_offset + epoch.slot_offset();
  const auto vpt = params.num_vecs_per_token;

  for (auto vid = global_tid; vid < params.num_push_vecs; vid += num_threads) {
    if constexpr (kPrim & Primitive::AG) {
      vec_t vec;
      vec.load(params.input, vid);
      Lamport::clear_pos_zero(vec.data());
      if constexpr (kWorldSize < 8) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          ptx::st_relaxed_16B(vec, push_ptrs[i], vid);
        }
      } else {
        ptx::st_multimem_16B(vec, dst_ptr_mc, vid);
      }
    } else /* reduce-scatter only */ {
      const auto token_id = vid / params.vecs_per_token_div;
      const auto offset = vid % params.vecs_per_token_div;
      // Both by a compile-time constant, so this is a mask and a shift.
      const auto dst_rank = token_id % kWorldSize;
      const auto dst_token_id = token_id / kWorldSize;
      // The walk is round-robin so neighbouring work lands on different peers
      // and every link stays busy instead congestion on 1 rank
      const auto avg_tokens = params.tokens_avg;
      const auto rem_tokens = params.tokens_rem;
      const auto rank_prefix = dst_rank * avg_tokens + std::min(dst_rank, rem_tokens);
      const auto src_token = rank_prefix + dst_token_id;
      vec_t vec;
      vec.load(params.input, src_token * vpt + offset);
      const auto dst_ptr = epoch.slot_ptr(dst_rank, params.rank);
      Lamport::clear_pos_zero(vec.data());
      ptx::st_relaxed_16B(vec, dst_ptr, dst_token_id * vpt + offset);
    }
  }

  // Poll addresses are linear in the source rank -- one base, `slot_bytes`
  // apart -- so a base plus a vector-index bias replaces a kWorldSize-wide
  // pointer table: 2 registers instead of 2 per peer. (The push side cannot do
  // this; `workspaces[i]` genuinely varies per peer.)
  const auto poll_base = epoch.slot_ptr(params.rank);
  const auto slot_vecs = params.ws.slot_bytes / sizeof(vec_t);
  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());

  for (auto vid = global_tid; vid < params.num_poll_vecs; vid += num_threads) {
    if constexpr (kPrim & Primitive::RS) {
      constexpr uint32_t kNumPairs = kVecSize / 2;
      fp32x2_t acc[kNumPairs];
      vec_t out_vec;

      if constexpr (kGroup >= kWorldSize) {
        vec_t vec[kWorldSize];
        do {
          bool has_zero = false;
#pragma unroll
          for (uint32_t i = 0; i < kWorldSize; ++i) {
            ptx::ld_relaxed_16B(vec[i], poll_base, i * slot_vecs + vid);
          }
#pragma unroll
          for (uint32_t i = 0; i < kWorldSize; ++i) {
            has_zero |= Lamport::has_pos_zero(vec[i].data());
          }
          if (!has_zero) break;
        } while (true);
        out_vec = reduce_vec(vec);
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          ptx::st_global_16B(pos_zero_vec, poll_base, i * slot_vecs + vid);
        }
      } else /* > 1 group: divide into chunks */ {
        constexpr uint32_t kNumGroups = div_ceil(kWorldSize, kGroup);
        vec_t vec[kGroup];

#pragma unroll
        for (uint32_t g = 0; g < kNumGroups; ++g) {
          const auto for_each = [&](auto&& fn) {
#pragma unroll
            for (uint32_t j = 0; j < kGroup; ++j) {
              const auto i = g * kGroup + j;
              if (i >= kWorldSize) continue;
              fn(i, j);
            }
          };

          do {
            bool has_zero = false;
            for_each([&](uint32_t i, uint32_t j) {
              // load all the vectors
              ptx::ld_relaxed_16B(vec[j], poll_base, i * slot_vecs + vid);
            });
            for_each([&](uint32_t, uint32_t j) {
              // check for zeros
              has_zero |= Lamport::has_pos_zero(vec[j].data());
            });
            if (!has_zero) break;
          } while (true);

          for_each([&](uint32_t i, uint32_t j) {
#pragma unroll
            for (uint32_t k = 0; k < kNumPairs; ++k) {
              const auto [x, y] = cast<fp32x2_t>(vec[j][k]);
              acc[k].x = i == 0 ? x : acc[k].x + x;
              acc[k].y = i == 0 ? y : acc[k].y + y;
            }
            ptx::st_global_16B(pos_zero_vec, poll_base, i * slot_vecs + vid);
          });
        }
#pragma unroll
        for (uint32_t k = 0; k < kNumPairs; ++k) {
          out_vec[k] = cast<packed_t<T>>(acc[k]);
        }
      }

      out_vec.store(params.output, vid);
    } else /* all-gather only */ {
      vec_t vec;
      do {
        ptx::ld_relaxed_16B(vec, poll_base, vid);
      } while (Lamport::has_pos_zero(vec.data()));
      vec.store(params.output, vid);
      ptx::st_global_16B(pos_zero_vec, poll_base, vid);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.flip();
}

inline constexpr uint32_t kPullUnroll = 4;

template <typename T, Primitive kPrim, bool kUsePDL>
PULL_KERNEL void nvlink_pull_kernel(const __grid_constant__ NVLinkCommPullParams params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / sizeof(T);  // 16 bytes per vector
  using vec_t = device::AlignedVector<packed_t<T>, kVecSize / 2>;
  constexpr uint32_t kNumWarpVecs = kPullUnroll * kWarpThreads;

  // Round-robin chunks to blocks rather than giving each block a contiguous
  // run: the global warp index runs block-fastest, so neighbouring chunks are
  // driven by different CTAs.
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto num_warps = gridDim.x * (kPullCTASize / kWarpThreads);

  PDLWaitPrimary<kUsePDL>();
  const auto barrier = distributed::McBarrier{params.sem_local, params.sem_mc, params.world_size, 2};
  barrier.arrive_relaxed(/*n=*/0);
  __syncthreads();

  const auto num_whole_chunks = params.num_vecs / kNumWarpVecs;
  // warp uniform unrolled path, 0 predicate
  for (auto chunk = global_warp_id; chunk < num_whole_chunks; chunk += num_warps) {
    vec_t vecs[kPullUnroll];
    const auto base = chunk * kNumWarpVecs + lane_id;

#pragma unroll
    for (uint32_t i = 0; i < kPullUnroll; ++i) {
      if constexpr (kPrim & Primitive::RS) {
        ptx::ld_multimem_16B(vecs[i], params.input_mc, base + i * kWarpThreads);
      } else {
        ptx::ld_global_16B(vecs[i], params.input, base + i * kWarpThreads);
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < kPullUnroll; ++i) {
      if constexpr (kPrim & Primitive::AG) {
        ptx::st_multimem_16B(vecs[i], params.output_mc, base + i * kWarpThreads);
      } else {
        ptx::st_global_16B(vecs[i], params.output, base + i * kWarpThreads);
      }
    }
  }

  const auto chunk_offset = num_whole_chunks * kNumWarpVecs;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;
  const auto global_threads = num_warps * kWarpThreads;
  for (auto vid = chunk_offset + global_tid; vid < params.num_vecs; vid += global_threads) {
    vec_t vec;
    if constexpr (kPrim & Primitive::RS) {
      ptx::ld_multimem_16B(vec, params.input_mc, vid);
    } else {
      ptx::ld_global_16B(vec, params.input, vid);
    }
    if constexpr (kPrim & Primitive::AG) {
      ptx::st_multimem_16B(vec, params.output_mc, vid);
    } else {
      ptx::st_global_16B(vec, params.output, vid);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  if constexpr (kPrim & Primitive::AG) {
    barrier.arrive_rel_acq(/*n=*/1);
  } else {  // no store multimem, only local store
    barrier.arrive_relaxed(/*n=*/1);
  }
}

template <bool kUsePDL>
__global__ void nvlink_barrier_kernel(Semaphore* sem_local, Semaphore* sem_mc, uint32_t world_size) {
  using device::distributed::McBarrier;
  device::PDLWaitPrimary<true>();
  const auto barrier = McBarrier{sem_local, sem_mc, world_size, 1};
  barrier.arrive_relaxed(0);
  device::PDLTriggerSecondary<true>();
}

/// Block size for the push kernel: the smallest that still spreads the work
/// over every SM, capped at the launch bound.
inline auto choose_push_block_size(uint32_t num_vecs) -> uint32_t {
  static const uint32_t kNumSM = [] {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    return host::runtime::get_sm_count(device);
  }();
  for (const uint32_t block_size : {128u, 256u, 384u, 512u}) {
    if (host::div_ceil(num_vecs, block_size) <= kNumSM) return block_size;
  }
  return 1024u;
}

/// Host side of the push collectives. `Primitive` stays in C++: each entry
/// point pins its own, so the Python surface is three named ops rather than
/// one op plus a mode flag.
///
/// The workspace contract each primitive assumes -- these follow from the
/// kernel's addressing, and are the part to re-check against the design:
///
///   all_reduce      rank r multicasts its whole input into slot r of every
///                   rank's push workspace, then each rank folds the
///                   world_size slots locally. Needs `nbytes <= slot_bytes`.
///   reduce_scatter  input is [num_tokens, hidden]; token t is unicast to rank
///                   t % W at local index t / W, into the sender's slot there.
///                   Each rank folds the slots covering the tokens it owns.
///   all_gather      input is this rank's [my_tokens, hidden] shard, multicast
///                   into a contiguous prefix of every workspace; the poll then
///                   walks the gathered extent linearly, which works because
///                   slot i sits at `i * slot_bytes`. Assumes an equal split.
template <typename T, bool kUsePDL>
struct NVLinkComm {
 private:
  using TensorView = tvm::ffi::TensorView;
  using PushPlaneObj = host::distributed::PushPlaneObj;
  using PullPlaneObj = host::distributed::PullPlaneObj;
  using CommunicatorObj = host::distributed::CommunicatorObj;
  using CommunicatorRef = host::distributed::CommunicatorRef;
  static constexpr uint32_t kVecBytes = 16;
  static constexpr uint32_t kVecSize = kVecBytes / sizeof(T);

 public:
  struct RouteInfo {
    uint32_t prefix_tokens;    // exclusive prefix sum
    uint32_t num_rank_tokens;  // current rank
  };

  static RouteInfo get_routing(uint32_t num_tokens, uint32_t rank, uint32_t world_size) {
    const auto avg = num_tokens / world_size;
    const auto rem = num_tokens % world_size;
    return {rank * avg + std::min(rank, rem), avg + (rank < rem ? 1 : 0)};
  }

  struct HostParams {
    int64_t hidden_size;
    DLDevice device;
  };

  static HostParams check_params(const TensorView in, const TensorView out, host::DebugInfo info = {}) {
    using namespace host;
    auto D = SymbolicSize{"hidden_size"};
    auto device_ = SymbolicDevice{};
    auto dtype_ = SymbolicDType{};
    if constexpr (!std::is_same_v<T, void>) dtype_.set_options<T>();
    device_.set_options<kDLCUDA>();
    TensorMatcher({-1, D})  //
        .with_dtype(dtype_)
        .with_device(device_)
        .verify(in, info);
    TensorMatcher({-1, D})  //
        .with_dtype(dtype_)
        .with_device(device_)
        .verify(out, info);
    return {D.unwrap(), device_.unwrap()};
  }

 private:
  template <Primitive kPrim, uint32_t kWorldSize>
  static void run_push(const PushPlaneObj& push, const TensorView in, const TensorView out) {
    CHECK_HOST(push.world_size == kWorldSize) << push.world_size << " != " << kWorldSize;
    const auto [hidden_size, device] = check_params(in, out);
    const auto rank = push.rank;
    const auto num_vecs_per_token = static_cast<uint32_t>(hidden_size / kVecSize);
    const auto num_push_vecs = static_cast<uint32_t>(in.numel() / kVecSize);
    const auto num_poll_vecs = static_cast<uint32_t>(out.numel() / kVecSize);
    const auto slot_bytes = static_cast<int64_t>(push.slot_bytes);
    const auto out_nbytes = static_cast<int64_t>(out.numel() * sizeof(T));
    const auto num_tokens = static_cast<uint32_t>(in.size(0));
    const auto out_tokens = static_cast<uint32_t>(out.size(0));
    const auto total_tokens = kPrim == Primitive::AG ? out_tokens : num_tokens;
    const auto routing = get_routing(total_tokens, rank, kWorldSize);

    uint32_t dst_offset = 0;
    if constexpr (kPrim == Primitive::AR) {
      CHECK_HOST(num_tokens == out_tokens);
      CHECK_HOST(out_nbytes <= push.slot_bytes);
      dst_offset = static_cast<uint32_t>(rank * slot_bytes);
    } else if constexpr (kPrim == Primitive::RS) {
      CHECK_HOST(out_tokens == routing.num_rank_tokens);
      CHECK_HOST(out_nbytes <= push.slot_bytes);
      // dst_offset is not used for this case
    } else {
      static_assert(kPrim == Primitive::AG);
      CHECK_HOST(num_tokens == routing.num_rank_tokens);
      CHECK_HOST(out_nbytes <= slot_bytes * kWorldSize);
      dst_offset = routing.prefix_tokens * static_cast<uint32_t>(num_vecs_per_token * kVecBytes);
    }

    // Slice the whole plane: the kernel reaches every slot, not just this rank's.
    const auto block_size = choose_push_block_size(std::max(num_push_vecs, num_poll_vecs));
    CHECK_HOST(num_vecs_per_token > 0) << "fast div-mod rejects a zero divisor";
    const auto in_tokens_total = static_cast<uint32_t>(in.size(0));
    const auto params = NVLinkCommPushParams<kWorldSize>{
        .input = in.data_ptr(),
        .output = out.data_ptr(),
        .dst_offset = dst_offset,
        .rank = rank,
        .num_push_vecs = num_push_vecs,
        .num_poll_vecs = num_poll_vecs,
        .num_vecs_per_token = num_vecs_per_token,
        .tokens_avg = in_tokens_total / kWorldSize,
        .tokens_rem = in_tokens_total % kWorldSize,
        .vecs_per_token_div = fast_mod_div_u32_t{num_vecs_per_token},
        .ws = push.get_workspace<kWorldSize>(/*size=*/0),
    };
    host::LaunchKernel(push.num_blocks, block_size, device)
        .enable_pdl(kUsePDL)(nvlink_push_kernel<T, kPrim, kWorldSize, kUsePDL>, params);
  }

  template <Primitive kPrim>
  static void run_pull(
      const PullPlaneObj& pull,
      const TensorView in,
      const TensorView out,
      uintptr_t in_mc_ptr,
      uintptr_t out_mc_ptr,
      uint32_t num_blocks_hint = 0) {
    CHECK_HOST(pull.mc_semaphore != nullptr);
    const auto [hidden_size, device] = check_params(in, out);
    const auto rank = pull.rank;
    const auto world_size = pull.world_size;
    const auto num_tokens = static_cast<uint32_t>(in.size(0));
    const auto out_tokens = static_cast<uint32_t>(out.size(0));
    const auto total_tokens = kPrim == Primitive::AG ? out_tokens : num_tokens;
    const auto routing = get_routing(total_tokens, rank, world_size);
    const auto num_vecs_per_token = static_cast<uint32_t>(hidden_size / kVecSize);
    const auto bytes_per_token = static_cast<uint32_t>(num_vecs_per_token * kVecBytes);
    const auto prefix_bytes = static_cast<uint32_t>(routing.prefix_tokens * bytes_per_token);

    // 0 = no hint, autotune; > 0 always use hint but clip to upper bound
    if constexpr (kPrim == Primitive::AR) {
      CHECK_HOST(num_tokens == out_tokens && in_mc_ptr != 0 && out_mc_ptr != 0);
      // each rank simply perform RS + AG on 1 shard
      in_mc_ptr += prefix_bytes;
      out_mc_ptr += prefix_bytes;
      if (num_blocks_hint == 0) num_blocks_hint = host::div_ceil(64u, world_size);
    } else if constexpr (kPrim == Primitive::RS) {
      CHECK_HOST(out_tokens == routing.num_rank_tokens && in_mc_ptr != 0);
      in_mc_ptr += prefix_bytes;
      if (num_blocks_hint == 0) num_blocks_hint = pull.num_blocks;  // use all the blocks for RS
    } else {
      static_assert(kPrim == Primitive::AG);
      // The gather reads its shard locally and stores through the output alias,
      // so it is `out_mc_ptr` that has to be there -- `in_mc_ptr` is passed 0.
      CHECK_HOST(num_tokens == routing.num_rank_tokens && out_mc_ptr != 0);
      out_mc_ptr += prefix_bytes;
      if (num_blocks_hint == 0) num_blocks_hint = host::div_ceil(32u, world_size);
    }

    const auto params = NVLinkCommPullParams{
        .input = in.data_ptr(),
        .output = out.data_ptr(),
        .num_vecs = static_cast<uint32_t>(routing.num_rank_tokens * num_vecs_per_token),
        .input_mc = std::bit_cast<uint8_t*>(in_mc_ptr),
        .output_mc = std::bit_cast<uint8_t*>(out_mc_ptr),
        .sem_local = pull.semaphores[rank],
        .sem_mc = pull.mc_semaphore,
        .rank = rank,
        .world_size = pull.world_size,
    };

    // NOTE: the final num_blocks resolution must be world unified, otherwise may deadlock
    const auto max_vecs_in_world = host::div_ceil(total_tokens, world_size) * num_vecs_per_token;
    const auto max_num_blocks = host::div_ceil(max_vecs_in_world, kPullUnroll * kPullCTASize);
    const auto num_blocks = std::max(1u, std::min(max_num_blocks, num_blocks_hint));
    host::LaunchKernel(num_blocks, kPullCTASize, device)
        .enable_pdl(kUsePDL)(nvlink_pull_kernel<T, kPrim, kUsePDL>, params);
  }

 public:
  // specialized for each world size
  template <uint32_t kWorldSize>
  static void all_reduce_push(CommunicatorRef comm, TensorView in, TensorView out) {
    return run_push<Primitive::AR, kWorldSize>(comm->get_push_obj(), in, out);
  }
  template <uint32_t kWorldSize>
  static void all_gather_push(CommunicatorRef comm, TensorView in, TensorView out) {
    return run_push<Primitive::AG, kWorldSize>(comm->get_push_obj(), in, out);
  }
  template <uint32_t kWorldSize>
  static void reduce_scatter_push(CommunicatorRef comm, TensorView in, TensorView out) {
    return run_push<Primitive::RS, kWorldSize>(comm->get_push_obj(), in, out);
  }

  // only compiler once for each world size
  static void all_reduce_pull(
      CommunicatorRef comm,  // only pull is needed
      TensorView in,
      TensorView out,
      int64_t in_mc_ptr,
      int64_t out_mc_ptr,
      uint32_t num_blocks_hint) {
    return run_pull<Primitive::AR>(comm->get_pull_obj(), in, out, in_mc_ptr, out_mc_ptr, num_blocks_hint);
  }
  static void all_gather_pull(
      CommunicatorRef comm,  // only pull is needed
      TensorView in,
      TensorView out,
      int64_t out_mc_ptr,
      uint32_t num_blocks_hint) {
    return run_pull<Primitive::AG>(comm->get_pull_obj(), in, out, 0, out_mc_ptr, num_blocks_hint);
  }
  static void reduce_scatter_pull(
      CommunicatorRef comm,  // only pull is needed
      TensorView in,
      TensorView out,
      int64_t in_mc_ptr,
      uint32_t num_blocks_hint) {
    return run_pull<Primitive::RS>(comm->get_pull_obj(), in, out, in_mc_ptr, 0, num_blocks_hint);
  }
};

/// The stream comes from the caller rather than the FFI environment: tvm-ffi
/// only publishes the framework stream when a call carries a DLPack tensor, and
/// this one carries none. Resolving it from the environment instead put the
/// launch on a stale stream, so under graph capture the barrier ran once at
/// capture time and every replay silently skipped it.
template <bool kUsePDL>
void nvlink_barrier(host::distributed::CommunicatorRef comm, int64_t stream_id) {
  const auto& pull = comm->get_pull_obj();
  CHECK_HOST(pull.mc_semaphore);
  const auto stream = std::bit_cast<cudaStream_t>(stream_id);
  const auto sem_local = pull.semaphores[pull.rank];
  host::LaunchKernel(1, device::kWarpThreads, stream)  //
      .enable_pdl(kUsePDL)(nvlink_barrier_kernel<kUsePDL>, sem_local, pull.mc_semaphore, pull.world_size);
}

template <bool kUsePDL>
void all_gather_copy_engine(
    const host::distributed::CommunicatorRef comm,
    const tvm::ffi::TensorView in,
    const tvm::ffi::TensorView out,
    const int64_t out_mc_ptr) {
  using Impl = NVLinkComm<void, kUsePDL>;
  const auto& pull = comm->get_pull_obj();
  const auto [hidden_size, device] = Impl::check_params(in, out);
  const auto total_tokens = out.size(0);
  const auto routing = Impl::get_routing(total_tokens, pull.rank, pull.world_size);
  CHECK_HOST(in.size(0) == routing.num_rank_tokens);
  const auto element_bytes = host::dtype_bytes(in.dtype());
  const auto dst_ptr = out_mc_ptr + routing.prefix_tokens * hidden_size * element_bytes;
  const auto stream = host::LaunchKernel::resolve_device(device);
  const auto sem_local = pull.semaphores[pull.rank];
  const auto launch_barrier = [&] {
    host::LaunchKernel(1, device::kWarpThreads, stream)  //
        .enable_pdl(kUsePDL)(nvlink_barrier_kernel<kUsePDL>, sem_local, pull.mc_semaphore, pull.world_size);
  };

  launch_barrier();
  CHECK_CUDA(cudaMemcpyAsync(
      /*dst=*/std::bit_cast<void*>(dst_ptr),
      /*src=*/in.data_ptr(),
      /*count=*/in.numel() * element_bytes,
      /*kind=*/cudaMemcpyDeviceToDevice,
      /*stream=*/stream));
  launch_barrier();
}

}  // namespace sglang
