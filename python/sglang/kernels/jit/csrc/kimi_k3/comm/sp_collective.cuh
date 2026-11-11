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

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/extra/stl.h>

#include <type_traits>

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

template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL>
__global__ void reduce_scatter_res_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B

  device::PDLWaitPrimary<kUsePDL>();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t num_threads = gridDim.x * blockDim.x;
  const uint32_t phase = params.counter->load();
  const auto phase_offset = phase * kWorldSize * params.stride_bytes;
  const auto producer_offset = phase_offset + params.rank * params.stride_bytes;

  // Each vector goes only to the rank that owns its row shard.
  for (uint32_t vid = tid; vid < kWorldSize * params.local_vecs; vid += num_threads) {
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
template <uint32_t kWorldSize, uint32_t kUnroll, bool kUsePDL>
__global__ void all_gather_direct_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B

  const auto barrier = device::distributed::McBarrier(params.sem_local, params.sem_mc, kWorldSize);
  device::PDLWaitPrimary<kUsePDL>();
  barrier.arrive_relaxed();
  __syncthreads();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t step = gridDim.x * blockDim.x;
  const uint32_t dst_bias = params.rank * params.local_vecs;
  for (uint32_t base = tid; base < params.local_vecs; base += step * kUnroll) {
    vec_t vec[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t vid = base + u * step;
      if (vid < params.local_vecs) device::ptx::ld_global_16B(vec[u], params.input, vid);
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t vid = base + u * step;
      if (vid < params.local_vecs) device::ptx::st_multimem_16B(vec[u], params.output_mc, dst_bias + vid);
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq();
}

// NVLS pull variant: o_proj writes its TP-partial result into multicast-bound
// symmetric memory. Each rank reduces only its owned row shard directly from
// the multicast alias, so no staging or second broadcast is needed.
template <uint32_t kWorldSize, bool kHasResidual, uint32_t kUnroll, bool kUsePDL>
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
  for (uint32_t base = tid; base < params.local_vecs; base += step * kUnroll) {
    vec_t vec[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t vid = base + u * step;
      if (vid < params.local_vecs) device::ptx::ld_multimem_16B(vec[u], input_mc, vid);
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t vid = base + u * step;
      if (vid >= params.local_vecs) continue;
      if constexpr (kHasResidual) {
        vec_t res;
        device::ptx::ld_global_16B(res, residual, vid);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          vec[u][j] = SumOp::reduce(vec[u][j], res[j]);
        }
      }
      device::ptx::st_global_16B(vec[u], params.output, vid);
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq();
}

inline constexpr uint32_t kMaxGatherSegments = 4;

struct MultiParams {
  const uint8_t* input[kMaxGatherSegments];
  uint8_t* output_mc[kMaxGatherSegments];
  uint32_t local_vecs[kMaxGatherSegments];
  Semaphore* sem_local;
  Semaphore* sem_mc;
  uint32_t num_segments;
  uint32_t rank;
};

template <uint32_t kWorldSize, uint32_t kUnroll, bool kUsePDL>
__global__ void all_gather_multi_kernel(const __grid_constant__ MultiParams params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B

  const auto barrier = device::distributed::McBarrier(params.sem_local, params.sem_mc, kWorldSize);
  device::PDLWaitPrimary<kUsePDL>();
  barrier.arrive_relaxed();
  __syncthreads();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t step = gridDim.x * blockDim.x;
#pragma unroll 1
  for (uint32_t s = 0; s < params.num_segments; ++s) {
    const auto* input = params.input[s];
    auto* output_mc = params.output_mc[s];
    const uint32_t local_vecs = params.local_vecs[s];
    const uint32_t dst_bias = params.rank * local_vecs;
    for (uint32_t base = tid; base < local_vecs; base += step * kUnroll) {
      vec_t vec[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        const uint32_t vid = base + u * step;
        if (vid < local_vecs) device::ptx::ld_global_16B(vec[u], input, vid);
      }
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        const uint32_t vid = base + u * step;
        if (vid < local_vecs) device::ptx::st_multimem_16B(vec[u], output_mc, dst_bias + vid);
      }
    }
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

  template <typename F>
  static void with_unroll(int64_t unroll, F&& launch) {
    switch (unroll) {
      case 1:
        return launch(std::integral_constant<uint32_t, 1>{});
      case 2:
        return launch(std::integral_constant<uint32_t, 2>{});
      case 4:
        return launch(std::integral_constant<uint32_t, 4>{});
      case 8:
        return launch(std::integral_constant<uint32_t, 8>{});
      default:
        CHECK_HOST(false) << "unroll must be 1, 2, 4, or 8";
    }
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

  static void all_gather_direct(
      CommunicatorRef ref,
      TensorView input,
      TensorView output,
      int64_t output_mc_ptr,
      int64_t num_blocks,
      int64_t block_size,
      int64_t unroll) {
    const auto& pull = ref.get()->get_pull_obj();
    CHECK_HOST(output_mc_ptr != 0) << "direct all-gather needs symmetric output";
    check_pull_launch(ref.get()->get_pull_blocks(), num_blocks, block_size);
    auto params = make_params(*ref.get(), output, input, std::nullopt, false);
    bind_pull(params, pull);
    params.input = static_cast<const uint8_t*>(input.data_ptr());
    params.output = static_cast<uint8_t*>(output.data_ptr());
    params.output_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(output_mc_ptr));
    with_unroll(unroll, [&](auto u) {
      constexpr uint32_t kUnroll = decltype(u)::value;
      host::LaunchKernel(num_blocks, block_size, input.device())
          .enable_pdl(kUsePDL)(sp_collective::all_gather_direct_kernel<kWorldSize, kUnroll, kUsePDL>, params);
    });
  }

  static void reduce_scatter_pull(
      CommunicatorRef ref,
      TensorView input,
      TensorView output,
      std::optional<TensorView> residual,
      bool residual_is_local,
      int64_t input_mc_ptr,
      int64_t num_blocks,
      int64_t block_size,
      int64_t unroll) {
    const auto& pull = ref.get()->get_pull_obj();
    CHECK_HOST(input_mc_ptr != 0) << "pull RS needs symmetric input";
    check_pull_launch(ref.get()->get_pull_blocks(), num_blocks, block_size);
    auto params = make_params(*ref.get(), input, output, residual, residual_is_local);
    bind_pull(params, pull);
    params.input_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(input_mc_ptr));
    with_unroll(unroll, [&](auto u) {
      constexpr uint32_t kUnroll = decltype(u)::value;
      const auto kernel = residual.has_value()
                              ? sp_collective::reduce_scatter_pull_kernel<kWorldSize, true, kUnroll, kUsePDL>
                              : sp_collective::reduce_scatter_pull_kernel<kWorldSize, false, kUnroll, kUsePDL>;
      host::LaunchKernel(num_blocks, block_size, input.device()).enable_pdl(kUsePDL)(kernel, params);
    });
  }

  static void all_gather_multi(
      CommunicatorRef ref,
      TensorView device_ref,
      int64_t in0,
      int64_t in1,
      int64_t in2,
      int64_t in3,
      int64_t mc0,
      int64_t mc1,
      int64_t mc2,
      int64_t mc3,
      int64_t bytes0,
      int64_t bytes1,
      int64_t bytes2,
      int64_t bytes3,
      int64_t num_segments,
      int64_t num_blocks,
      int64_t block_size,
      int64_t unroll) {
    const auto& pull = ref.get()->get_pull_obj();
    CHECK_HOST(pull.mc_semaphore != nullptr) << "the pull path needs a multicast-capable pull plane";
    CHECK_HOST(num_segments >= 1 && num_segments <= int64_t(sp_collective::kMaxGatherSegments));
    check_pull_launch(ref.get()->get_pull_blocks(), num_blocks, block_size);
    const int64_t ins[] = {in0, in1, in2, in3};
    const int64_t mcs[] = {mc0, mc1, mc2, mc3};
    const int64_t bytes[] = {bytes0, bytes1, bytes2, bytes3};
    sp_collective::MultiParams params{};
    for (int64_t s = 0; s < num_segments; ++s) {
      CHECK_HOST(ins[s] != 0 && mcs[s] != 0 && bytes[s] > 0 && bytes[s] % 16 == 0) << "segment " << s;
      params.input[s] = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ins[s]));
      params.output_mc[s] = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(mcs[s]));
      params.local_vecs[s] = static_cast<uint32_t>(bytes[s] / 16);
    }
    params.sem_local = pull.semaphores[pull.rank];
    params.sem_mc = pull.mc_semaphore;
    params.num_segments = static_cast<uint32_t>(num_segments);
    params.rank = ref.get()->get_rank();
    with_unroll(unroll, [&](auto u) {
      constexpr uint32_t kUnroll = decltype(u)::value;
      host::LaunchKernel(num_blocks, block_size, device_ref.device())
          .enable_pdl(kUsePDL)(sp_collective::all_gather_multi_kernel<kWorldSize, kUnroll, kUsePDL>, params);
    });
  }
};

}  // namespace sglang
