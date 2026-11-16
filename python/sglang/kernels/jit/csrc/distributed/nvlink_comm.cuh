#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/object.h>

#include <algorithm>
#include <bit>
#include <cstdint>

namespace sglang {

using device::distributed::PushWorkSpace;
using device::distributed::Semaphore;

template <uint32_t kWorldSize>
struct NVLinkCommPushParams {
  const void* __restrict__ input;
  const void* __restrict__ residual;
  void* __restrict__ output;
  uint32_t dst_offset;  // packed token prefix
  uint32_t rank;
  uint32_t num_push_vecs;
  uint32_t num_poll_vecs;
  PushWorkSpace<kWorldSize> ws;
};

struct NVLinkCommPullParams {
  const void* __restrict__ input;
  const void* __restrict__ residual;
  void* __restrict__ output;
  uint32_t num_vecs;
  // multicast buffer
  uint8_t* output_mc;
  Semaphore* sem_local;
  Semaphore* sem_mc;
  uint32_t rank;
  uint32_t world_size;
};

inline constexpr uint32_t kPushCTASize = 1024;  // max value
inline constexpr uint32_t kPullCTASize = 512;   // max value

#define PUSH_KERNEL __global__ __launch_bounds__(kPushCTASize, 1)
#define PULL_KERNEL __global__ __launch_bounds__(kPullCTASize, 1)

template <typename vec_t>
SGL_DEVICE vec_t reduce_vec(vec_t x, vec_t y) {
  vec_t arr[2] = {x, y};
  return device::reduce_vec(arr);
}

/// \brief All-gather push: every rank pushes its shard into one contiguous region
/// `[rank0, rank1, rank2, ...]` of each peer's plane, then polls its own plane.
template <typename T, bool kHasResidual, uint32_t kWorldSize, bool kUsePDL>
PUSH_KERNEL void nvlink_push_kernel(const __grid_constant__ NVLinkCommPushParams<kWorldSize> params) {
  using namespace device;
  enable_smem_spilling();
  constexpr uint32_t kVecSize = 16 / sizeof(T);  // 16 bytes per vector
  using vec_t = device::AlignedVector<packed_t<T>, kVecSize / 2>;
  using Lamport = distributed::LamportTrait<T, kVecSize, /*kAtom=*/4>;

  // Round-robin warps to blocks: the poll domain is this rank's shard, so a
  // block-major index would park all of it on the first few CTAs and idle the
  // rest of the SMs.
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;
  const auto num_threads = blockDim.x * gridDim.x;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  void* push_ptrs[kWorldSize];
  /// NOTE: broadcast write is only fast when world size is large
  if constexpr (kWorldSize < 8) {
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      // `dst_offset` is a packed token prefix: the consumer reads the plane linearly;
      // this must stay the same address arithmetic as the multicast branch below.
      push_ptrs[i] = static_cast<uint8_t*>(epoch.slot_ptr(/*dst=*/i)) + params.dst_offset;
    }
  }

  const auto dst_ptr_mc = params.ws.mc_workspace + params.dst_offset + epoch.slot_offset();

  for (auto vid = global_tid; vid < params.num_push_vecs; vid += num_threads) {
    vec_t vec;
    vec.load(params.input, vid);
    if constexpr (kHasResidual) {
      vec_t res;
      res.load(params.residual, vid);
      vec = reduce_vec(vec, res);
    }
    Lamport::clear_pos_zero(vec.data());
    if constexpr (kWorldSize < 8) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::st_relaxed_16B(vec, push_ptrs[i], vid);
      }
    } else {
      ptx::st_multimem_16B(vec, dst_ptr_mc, vid);
    }
  }

  const auto poll_base = epoch.slot_ptr(params.rank);
  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());
  PDLTriggerSecondary<kUsePDL>();

  for (auto vid = global_tid; vid < params.num_poll_vecs; vid += num_threads) {
    vec_t vec;
    do {
      ptx::ld_relaxed_16B(vec, poll_base, vid);
    } while (Lamport::has_pos_zero(vec.data()));
    vec.store(params.output, vid);
    ptx::st_global_16B(pos_zero_vec, poll_base, vid);
  }

  __syncthreads();
  epoch.commit();
}

template <typename T, bool kHasResidual, bool kUsePDL, uint32_t kPullUnroll>
PULL_KERNEL void nvlink_pull_kernel(const __grid_constant__ NVLinkCommPullParams params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / sizeof(T);  // 16 bytes per vector
  using vec_t = device::AlignedVector<packed_t<T>, kVecSize / 2>;
  constexpr uint32_t kNumWarpVecs = kPullUnroll * kWarpThreads;

  // Round-robin chunks to blocks: the global warp index runs block-fastest, so
  // neighbouring chunks are driven by different CTAs.
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto num_warps = gridDim.x * (kPullCTASize / kWarpThreads);

  PDLWaitPrimary<kUsePDL>();
  const auto barrier = distributed::McBarrier{params.sem_local, params.sem_mc, params.world_size};
  barrier.arrive_relaxed();
  __syncthreads();

  const auto num_whole_chunks = params.num_vecs / kNumWarpVecs;
  // warp uniform unrolled path, 0 predicate
  for (auto chunk = global_warp_id; chunk < num_whole_chunks; chunk += num_warps) {
    vec_t vecs[kPullUnroll];
    const auto base = chunk * kNumWarpVecs + lane_id;

#pragma unroll
    for (uint32_t i = 0; i < kPullUnroll; ++i) {
      const auto vid = base + i * kWarpThreads;
      ptx::ld_global_16B(vecs[i], params.input, vid);
    }

    if constexpr (kHasResidual) {
      vec_t residuals[kPullUnroll];
#pragma unroll
      for (uint32_t i = 0; i < kPullUnroll; ++i) {
        residuals[i].load(params.residual, base + i * kWarpThreads);
      }
#pragma unroll
      for (uint32_t i = 0; i < kPullUnroll; ++i) {
        vecs[i] = reduce_vec(vecs[i], residuals[i]);
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < kPullUnroll; ++i) {
      ptx::st_multimem_16B(vecs[i], params.output_mc, base + i * kWarpThreads);
    }
  }

  const auto chunk_offset = num_whole_chunks * kNumWarpVecs;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;
  const auto global_threads = num_warps * kWarpThreads;
  for (auto vid = chunk_offset + global_tid; vid < params.num_vecs; vid += global_threads) {
    vec_t vec;
    ptx::ld_global_16B(vec, params.input, vid);
    if constexpr (kHasResidual) {
      vec_t res;
      res.load(params.residual, vid);
      vec = reduce_vec(vec, res);
    }
    ptx::st_multimem_16B(vec, params.output_mc, vid);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq();
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

  /// \brief Base pointer of the residual, shifted onto this rank's slice when
  /// the caller hands over the whole tensor.
  ///
  /// A shard-shaped residual passes straight through; a full tensor is sliced
  /// via `get_routing`, not a uniform stride, so ragged splits keep working.
  static const void* get_residual_ptr(
      const tvm::ffi::Optional<TensorView>& residual,
      uint32_t domain_tokens,
      uint32_t total_tokens,
      uint32_t prefix_bytes) {
    if (!residual.has_value()) return nullptr;
    const auto tokens = static_cast<uint32_t>(residual.value().size(0));
    const auto* base = static_cast<const uint8_t*>(residual.value().data_ptr());
    if (tokens == domain_tokens) return base;
    CHECK_HOST(tokens == total_tokens) << "residual has " << tokens << " tokens, expected " << domain_tokens
                                       << " (this rank's shard) or " << total_tokens << " (the whole tensor)";
    return base + prefix_bytes;
  }

  static HostParams check_params(
      const TensorView in,
      const TensorView out,
      const tvm::ffi::Optional<TensorView>& residual = {},
      host::DebugInfo info = {}) {
    using namespace host;
    auto D = SymbolicSize{"hidden_size"};
    auto device_ = SymbolicDevice{};
    auto dtype_ = SymbolicDType{};
    if constexpr (!std::is_same_v<T, void>) dtype_.set_options<T>();
    device_.set_options<kDLGPU>();
    TensorMatcher({-1, D})  //
        .with_dtype(dtype_)
        .with_device(device_)
        .verify(in, info);
    TensorMatcher({-1, D})  //
        .with_dtype(dtype_)
        .with_device(device_)
        .verify(out, info);
    if (residual.has_value()) {
      TensorMatcher({-1, D})  //
          .with_dtype(dtype_)
          .with_device(device_)
          .verify(residual.value(), info);
    }
    return {D.unwrap(), device_.unwrap()};
  }

 private:
  template <uint32_t kWorldSize>
  static void run_push(
      const PushPlaneObj& push,
      const TensorView in,
      const TensorView out,
      const tvm::ffi::Optional<TensorView> residual) {
    CHECK_HOST(push.world_size == kWorldSize) << push.world_size << " != " << kWorldSize;
    const auto [hidden_size, device] = check_params(in, out, residual);
    const auto rank = push.rank;
    const auto num_vecs_per_token = static_cast<uint32_t>(hidden_size / kVecSize);
    const auto num_push_vecs = static_cast<uint32_t>(in.numel() / kVecSize);
    const auto num_poll_vecs = static_cast<uint32_t>(out.numel() / kVecSize);
    const auto slot_bytes = static_cast<int64_t>(push.slot_bytes);
    const auto out_nbytes = static_cast<int64_t>(out.numel() * sizeof(T));
    const auto num_tokens = static_cast<uint32_t>(in.size(0));
    const auto out_tokens = static_cast<uint32_t>(out.size(0));
    const auto total_tokens = out_tokens;
    const auto routing = get_routing(total_tokens, rank, kWorldSize);

    CHECK_HOST(num_tokens == routing.num_rank_tokens);
    CHECK_HOST(out_nbytes <= slot_bytes * kWorldSize);
    const uint32_t dst_offset = routing.prefix_tokens * static_cast<uint32_t>(num_vecs_per_token * kVecBytes);

    // Slice the whole plane: the kernel reaches every slot, not just this rank's.
    const auto block_size = choose_push_block_size(std::max(num_push_vecs, num_poll_vecs));
    CHECK_HOST(num_vecs_per_token > 0) << "hidden_size " << hidden_size << " is below one 16-byte vector";
    // The gather works on this rank's shard, so a full-length residual is sliced.
    const auto residual_ptr = get_residual_ptr(
        residual,
        routing.num_rank_tokens,
        total_tokens,
        routing.prefix_tokens * static_cast<uint32_t>(num_vecs_per_token * kVecBytes));
    const auto params = NVLinkCommPushParams<kWorldSize>{
        .input = in.data_ptr(),
        .residual = residual_ptr,
        .output = out.data_ptr(),
        .dst_offset = dst_offset,
        .rank = rank,
        .num_push_vecs = num_push_vecs,
        .num_poll_vecs = num_poll_vecs,
        .ws = push.get_workspace<kWorldSize>(/*size=*/0),
    };
    const auto kernel = residual.has_value() ? nvlink_push_kernel<T, true, kWorldSize, kUsePDL>
                                             : nvlink_push_kernel<T, false, kWorldSize, kUsePDL>;
    host::LaunchKernel(push.default_blocks, block_size, device).enable_pdl(kUsePDL)(kernel, params);
  }

  template <uint32_t kPullUnroll>
  static void run_pull(
      const PullPlaneObj& pull,
      const TensorView in,
      const TensorView out,
      const tvm::ffi::Optional<TensorView> residual,
      uintptr_t out_mc_ptr,
      uint32_t num_blocks_hint) {
    CHECK_HOST(pull.mc_semaphore != nullptr);
    const auto [hidden_size, device] = check_params(in, out, residual);
    const auto rank = pull.rank;
    const auto world_size = pull.world_size;
    const auto num_tokens = static_cast<uint32_t>(in.size(0));
    const auto out_tokens = static_cast<uint32_t>(out.size(0));
    const auto total_tokens = out_tokens;
    const auto routing = get_routing(total_tokens, rank, world_size);
    const auto num_vecs_per_token = static_cast<uint32_t>(hidden_size / kVecSize);
    const auto bytes_per_token = static_cast<uint32_t>(num_vecs_per_token * kVecBytes);
    const auto prefix_bytes = static_cast<uint32_t>(routing.prefix_tokens * bytes_per_token);

    // 0 = no hint, autotune; > 0 always use hint but clip to upper bound
    CHECK_HOST(num_tokens == routing.num_rank_tokens && out_mc_ptr != 0);
    out_mc_ptr += prefix_bytes;
    if (num_blocks_hint == 0) num_blocks_hint = host::div_ceil(128u, kPullUnroll * world_size);
    /// NOTE: hard limit upper bound is `pull.capacity`
    num_blocks_hint = std::min(num_blocks_hint, pull.capacity);

    // The pull works on this rank's shard, so a full-length residual is sliced onto it.
    const auto residual_ptr = get_residual_ptr(residual, routing.num_rank_tokens, total_tokens, prefix_bytes);
    const auto params = NVLinkCommPullParams{
        .input = in.data_ptr(),
        .residual = residual_ptr,
        .output = out.data_ptr(),
        .num_vecs = static_cast<uint32_t>(routing.num_rank_tokens * num_vecs_per_token),
        .output_mc = std::bit_cast<uint8_t*>(out_mc_ptr),
        .sem_local = pull.semaphores[rank],
        .sem_mc = pull.mc_semaphore,
        .rank = rank,
        .world_size = pull.world_size,
    };

    /// NOTE: the final num_blocks resolution must be world unified, otherwise may deadlock
    const auto max_vecs_in_world = host::div_ceil(total_tokens, world_size) * num_vecs_per_token;
    const auto max_num_blocks = host::div_ceil(max_vecs_in_world, kPullUnroll * kPullCTASize);
    const auto num_blocks = std::max(1u, std::min(max_num_blocks, num_blocks_hint));
    const auto kernel = residual.has_value() ? nvlink_pull_kernel<T, true, kUsePDL, kPullUnroll>
                                             : nvlink_pull_kernel<T, false, kUsePDL, kPullUnroll>;
    host::LaunchKernel(num_blocks, kPullCTASize, device).enable_pdl(kUsePDL)(kernel, params);
  }

 public:
  // specialized for each world size
  template <uint32_t kWorldSize>
  static void
  all_gather_push(CommunicatorRef comm, TensorView in, TensorView out, tvm::ffi::Optional<TensorView> residual) {
    return run_push<kWorldSize>(comm->get_push_obj(), in, out, residual);
  }

  // only compile once for each world size
  template <uint32_t kPullUnroll>
  static void all_gather_pull(
      CommunicatorRef comm,  // only pull is needed
      TensorView in,
      TensorView out,
      tvm::ffi::Optional<TensorView> residual,
      int64_t out_mc_ptr,
      uint32_t num_blocks_hint) {
    return run_pull<kPullUnroll>(comm->get_pull_obj(), in, out, residual, out_mc_ptr, num_blocks_hint);
  }
};

}  // namespace sglang
