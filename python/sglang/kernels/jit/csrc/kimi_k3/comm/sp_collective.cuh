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
// Both kernels reuse CustomAllReduceV2's double-buffered push workspace and
// phase counters.  A bumper block advances counters outside the tuned work
// grid, so calls remain protocol-compatible with all-reduce and gemm_ag.

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include "../../distributed/custom_all_reduce.cuh"

namespace sglang {

namespace sp_collective {

using device::distributed::Counter;
using device::distributed::Semaphore;

struct Params {
  const uint8_t* input;
  uint8_t* output;
  const uint8_t* residual;
  uint8_t* push_workspaces[device::distributed::kMaxWorldSize];
  uint8_t* push_ws_mc;
  Counter* counter;
  Semaphore* sem_local;
  uint8_t* sem_mc;
  uint8_t* input_mc;
  uint8_t* output_mc;
  int64_t stride_bytes;
  uint32_t num_counters;
  uint32_t rank;
  uint32_t local_vecs;
  uint32_t residual_is_local;
};

template <typename Vec>
SGL_DEVICE void make_nonzero(Vec& vec) {
  constexpr uint32_t kNegZeroPair = 0x8000u;
  auto& bits = *reinterpret_cast<uint4*>(&vec);
  if (bits.x == 0) bits.x = kNegZeroPair;
  if (bits.y == 0) bits.y = kNegZeroPair;
  if (bits.z == 0) bits.z = kNegZeroPair;
  if (bits.w == 0) bits.w = kNegZeroPair;
}

SGL_DEVICE uint32_t* sem_mc_flag(uint8_t* sem_mc, uint32_t block) {
  static_assert(sizeof(Semaphore) == 128);
  return reinterpret_cast<uint32_t*>(sem_mc + block * sizeof(Semaphore));
}

SGL_DEVICE void sem_arrive_relaxed(uint32_t* flag) {
#if SGL_ARCH_HOPPER_OR_GREATER
  asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], 1;" ::"l"(flag) : "memory");
#else
  assert(false && "multimem red requires Hopper or later");
#endif
}

SGL_DEVICE void sem_arrive_release(uint32_t* flag) {
#if SGL_ARCH_HOPPER_OR_GREATER
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], 1;" ::"l"(flag) : "memory");
#else
  assert(false && "multimem red requires Hopper or later");
#endif
}

template <typename Vec>
SGL_DEVICE bool has_empty_marker(const Vec& vec) {
  const auto bits = *reinterpret_cast<const uint4*>(&vec);
  return bits.x == 0 || bits.y == 0 || bits.z == 0 || bits.w == 0;
}

template <typename Vec>
SGL_DEVICE Vec zero_vec() {
  Vec zero;
  zero.fill(bf16x2_t{get_pos_zero<bf16_t>(), get_pos_zero<bf16_t>()});
  return zero;
}

template <uint32_t kWorldSize>
SGL_DEVICE bool bumper_block(const Params& params) {
  const auto bx = blockIdx.x;
  if (bx + 1 != gridDim.x) return false;
  const auto phase = params.counter[bx].get() & 1;
  __syncthreads();
  for (uint32_t i = bx + threadIdx.x; i < params.num_counters; i += blockDim.x) {
    params.counter[i].set(phase ^ 1);
  }
  return true;
}

template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL>
__global__ void reduce_scatter_res_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B

  device::PDLWaitPrimary<kUsePDL>();
  if (bumper_block<kWorldSize>(params)) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const uint32_t bx = blockIdx.x;
  const uint32_t tid = bx * blockDim.x + threadIdx.x;
  const uint32_t num_threads = (gridDim.x - 1) * blockDim.x;
  const uint32_t phase = params.counter[bx].get() & 1;
  const auto phase_offset = phase * kWorldSize * params.stride_bytes;
  const auto producer_offset = phase_offset + params.rank * params.stride_bytes;

  // Each vector goes only to the rank that owns its row shard.
  for (uint32_t vid = tid; vid < kWorldSize * params.local_vecs; vid += num_threads) {
    const uint32_t dst_rank = vid / params.local_vecs;
    const uint32_t local_vid = vid - dst_rank * params.local_vecs;
    vec_t vec;
    ld_global_16B(vec, params.input, vid);
    make_nonzero(vec);
    st_relaxed_16B(vec, params.push_workspaces[dst_rank] + producer_offset, local_vid);
  }

  device::PDLTriggerSecondary<kUsePDL>();

  // Poll this rank's shard from all producer slots and reduce locally.
  const auto poll_base = params.push_workspaces[params.rank] + phase_offset;
  const auto residual_base = params.residual + (params.residual_is_local ? 0 : params.rank * params.local_vecs * 16);
  const auto zero = zero_vec<vec_t>();
  for (uint32_t vid = tid; vid < params.local_vecs; vid += num_threads) {
    vec_t vec[kWorldSize + kHasResidual];
    if constexpr (kHasResidual) {
      ld_global_16B(vec[kWorldSize], residual_base, vid);
    }
    do {
      bool empty = false;
#pragma unroll
      for (uint32_t rank = 0; rank < kWorldSize; ++rank) {
        ld_relaxed_16B(vec[rank], poll_base + rank * params.stride_bytes, vid);
        empty |= has_empty_marker(vec[rank]);
      }
      if (!empty) break;
    } while (true);
    const auto out = reduce(vec);
    st_global_16B(out, params.output, vid);
#pragma unroll
    for (uint32_t rank = 0; rank < kWorldSize; ++rank) {
      st_global_16B(zero, poll_base + rank * params.stride_bytes, vid);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) params.counter[bx].set(phase ^ 1);
}

template <uint32_t kWorldSize, bool kUsePDL>
__global__ void all_gather_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B

  device::PDLWaitPrimary<kUsePDL>();
  if (bumper_block<kWorldSize>(params)) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const uint32_t bx = blockIdx.x;
  const uint32_t tid = bx * blockDim.x + threadIdx.x;
  const uint32_t num_threads = (gridDim.x - 1) * blockDim.x;
  const uint32_t phase = params.counter[bx].get() & 1;
  const auto phase_offset = phase * kWorldSize * params.stride_bytes;
  const auto producer_offset = phase_offset + params.rank * params.stride_bytes;

  // One multicast store places this rank's shard in the same slot on peers.
  for (uint32_t vid = tid; vid < params.local_vecs; vid += num_threads) {
    vec_t vec;
    ld_global_16B(vec, params.input, vid);
    make_nonzero(vec);
    st_multimem_16B(vec, params.push_ws_mc + producer_offset, vid);
  }

  device::PDLTriggerSecondary<kUsePDL>();

  const auto poll_base = params.push_workspaces[params.rank] + phase_offset;
  const auto zero = zero_vec<vec_t>();
  for (uint32_t vid = tid; vid < kWorldSize * params.local_vecs; vid += num_threads) {
    const uint32_t src_rank = vid / params.local_vecs;
    const uint32_t local_vid = vid - src_rank * params.local_vecs;
    const auto src = poll_base + src_rank * params.stride_bytes;
    vec_t vec;
    do {
      ld_relaxed_16B(vec, src, local_vid);
    } while (has_empty_marker(vec));
    st_global_16B(vec, params.output, vid);
    st_global_16B(zero, src, local_vid);
  }

  __syncthreads();
  if (threadIdx.x == 0) params.counter[bx].set(phase ^ 1);
}

// Direct variant: output is multicast-bound symmetric memory.  Each producer
// writes its rank slice straight into every peer's final output, avoiding the
// staging read/copy/clear.  The two pull-semaphore barriers preserve protocol
// compatibility with CustomAllReduceV2 and make the remote writes visible
// before any rank leaves the kernel.
template <uint32_t kWorldSize, bool kUsePDL>
__global__ void all_gather_direct_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B

  uint32_t exit_base = 0;
  if (threadIdx.x == 0) {
    auto* semaphore = &params.sem_local[blockIdx.x];
    const auto reserved = semaphore->counter_ptr()->inc(2 * kWorldSize);
    exit_base = reserved + kWorldSize;
    device::PDLWaitPrimary<kUsePDL>();
    sem_arrive_relaxed(sem_mc_flag(params.sem_mc, blockIdx.x));
    while (semaphore->get_relaxed() - reserved < kWorldSize)
      ;
  }
  __syncthreads();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t step = gridDim.x * blockDim.x;
  const uint32_t dst_bias = params.rank * params.local_vecs;
  for (uint32_t vid = tid; vid < params.local_vecs; vid += step) {
    vec_t vec;
    ld_global_16B(vec, params.input, vid);
    st_multimem_16B(vec, params.output_mc, dst_bias + vid);
  }

  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  if (threadIdx.x == 0) {
    auto* semaphore = &params.sem_local[blockIdx.x];
    sem_arrive_release(sem_mc_flag(params.sem_mc, blockIdx.x));
    while (semaphore->get_acquire() - exit_base < kWorldSize)
      ;
  }
}

// NVLS pull variant: o_proj writes its TP-partial result into multicast-bound
// symmetric memory. Each rank reduces only its owned row shard directly from
// the multicast alias, so no staging or second broadcast is needed.
template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL>
__global__ void reduce_scatter_pull_kernel(const __grid_constant__ Params params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;  // 16 B
  using SumOp = device::ReductionTrait<device::ReductionOp::SUM, bf16x2_t>;

  uint32_t exit_base = 0;
  if (threadIdx.x == 0) {
    auto* semaphore = &params.sem_local[blockIdx.x];
    const auto reserved = semaphore->counter_ptr()->inc(2 * kWorldSize);
    exit_base = reserved + kWorldSize;
    device::PDLWaitPrimary<kUsePDL>();
    sem_arrive_relaxed(sem_mc_flag(params.sem_mc, blockIdx.x));
    while (semaphore->get_relaxed() - reserved < kWorldSize)
      ;
  }
  __syncthreads();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t step = gridDim.x * blockDim.x;
  const auto* input_mc = params.input_mc + params.rank * params.local_vecs * 16;
  const auto* residual =
      kHasResidual ? params.residual + (params.residual_is_local ? 0 : params.rank * params.local_vecs * 16) : nullptr;
  for (uint32_t vid = tid; vid < params.local_vecs; vid += step) {
    vec_t vec;
    ld_multimem_16B(vec, input_mc, vid);
    if constexpr (kHasResidual) {
      vec_t res;
      ld_global_16B(res, residual, vid);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        vec[j] = SumOp::reduce(vec[j], res[j]);
      }
    }
    st_global_16B(vec, params.output, vid);
  }

  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  if (threadIdx.x == 0) {
    auto* semaphore = &params.sem_local[blockIdx.x];
    sem_arrive_release(sem_mc_flag(params.sem_mc, blockIdx.x));
    while (semaphore->get_acquire() - exit_base < kWorldSize)
      ;
  }
}

}  // namespace sp_collective
using host::distributed::CommunicatorRef;

template <uint32_t kWorldSize, bool kUsePDL>
struct SPCollectiveKernel {
  using TensorView = tvm::ffi::TensorView;

  static sp_collective::Params make_params(
      const host::distributed::CommunicatorObj& data,
      TensorView input,
      TensorView output,
      std::optional<TensorView> residual,
      bool residual_is_local,
      int64_t ws_mc_base) {
    using namespace host;
    auto input_elems = SymbolicSize{"input_elems"};
    auto local_elems = SymbolicSize{"local_elems"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({input_elems}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({local_elems}).with_dtype<bf16_t>().with_device(device).verify(output);
    if (residual.has_value()) {
      if (residual_is_local) {
        TensorMatcher({local_elems}).with_dtype<bf16_t>().with_device(device).verify(residual.value());
      } else {
        TensorMatcher({input_elems}).with_dtype<bf16_t>().with_device(device).verify(residual.value());
      }
    }
    CHECK_HOST(data.world_size == kWorldSize);
    CHECK_HOST(local_elems.unwrap() > 0);
    CHECK_HOST(input_elems.unwrap() == local_elems.unwrap() * kWorldSize);
    CHECK_HOST(local_elems.unwrap() % 8 == 0) << "local shard bytes must be 16B aligned";
    CHECK_HOST(local_elems.unwrap() * sizeof(bf16_t) <= data.push_bytes) << "local shard exceeds a push slot";

    sp_collective::Params params{
        .input = static_cast<const uint8_t*>(input.data_ptr()),
        .output = static_cast<uint8_t*>(output.data_ptr()),
        .residual = residual.has_value() ? static_cast<const uint8_t*>(residual.value().data_ptr()) : nullptr,
        .push_workspaces = {},
        .push_ws_mc = reinterpret_cast<uint8_t*>(ws_mc_base),
        .counter = data.push_counter,
        .sem_local = data.pull_semaphores[data.rank],
        .sem_mc = nullptr,
        .input_mc = nullptr,
        .output_mc = nullptr,
        .stride_bytes = data.push_bytes,
        .num_counters = data.num_push_blocks,
        .rank = data.rank,
        .local_vecs = static_cast<uint32_t>(local_elems.unwrap() * sizeof(bf16_t) / 16),
        .residual_is_local = static_cast<uint32_t>(residual_is_local),
    };
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      params.push_workspaces[i] = data.push_workspaces[i];
    }
    return params;
  }

  static void check_launch(const host::distributed::CommunicatorObj& data, int64_t num_blocks, int64_t block_size) {
    CHECK_HOST(num_blocks > 0 && num_blocks < data.num_push_blocks);
    // The RS reduction keeps one 16B vector per producer in registers.
    // 1024-thread CTAs exceed the GB300 launch resource limit.
    CHECK_HOST(block_size >= 32 && block_size <= 512 && block_size % 32 == 0);
  }

  static void reduce_scatter_res(
      CommunicatorRef ref,
      TensorView input,
      TensorView output,
      std::optional<TensorView> residual,
      bool residual_is_local,
      int64_t num_blocks,
      int64_t block_size) {
    const auto& data = *ref.get();
    check_launch(data, num_blocks, block_size);
    auto params = make_params(data, input, output, residual, residual_is_local, 0);
    const auto kernel = residual.has_value() ? sp_collective::reduce_scatter_res_kernel<kWorldSize, true, kUsePDL>
                                             : sp_collective::reduce_scatter_res_kernel<kWorldSize, false, kUsePDL>;
    host::LaunchKernel(num_blocks + 1, block_size, input.device()).enable_pdl(kUsePDL)(kernel, params);
  }

  static void all_gather(
      CommunicatorRef ref,
      TensorView input,
      TensorView output,
      int64_t ws_mc_base,
      int64_t num_blocks,
      int64_t block_size) {
    const auto& data = *ref.get();
    CHECK_HOST(ws_mc_base != 0) << "all-gather requires multicast workspace";
    check_launch(data, num_blocks, block_size);
    // Reuse the RS matcher by swapping input/output roles conceptually.
    auto params = make_params(data, output, input, std::nullopt, false, ws_mc_base);
    params.input = static_cast<const uint8_t*>(input.data_ptr());
    params.output = static_cast<uint8_t*>(output.data_ptr());
    host::LaunchKernel(num_blocks + 1, block_size, input.device())
        .enable_pdl(kUsePDL)(sp_collective::all_gather_kernel<kWorldSize, kUsePDL>, params);
  }

  static void all_gather_direct(
      CommunicatorRef ref,
      TensorView input,
      TensorView output,
      int64_t output_mc_ptr,
      int64_t sem_mc_ptr,
      int64_t num_blocks,
      int64_t block_size) {
    const auto& data = *ref.get();
    CHECK_HOST(output_mc_ptr != 0) << "direct all-gather needs symmetric output";
    CHECK_HOST(sem_mc_ptr != 0) << "direct all-gather needs multicast semaphores";
    CHECK_HOST(num_blocks > 0 && num_blocks <= data.num_pull_blocks);
    CHECK_HOST(block_size >= 32 && block_size <= 1024 && block_size % 32 == 0);
    auto params = make_params(data, output, input, std::nullopt, false, 0);
    params.input = static_cast<const uint8_t*>(input.data_ptr());
    params.output = static_cast<uint8_t*>(output.data_ptr());
    params.output_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(output_mc_ptr));
    params.sem_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(sem_mc_ptr));
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
      int64_t sem_mc_ptr,
      int64_t num_blocks,
      int64_t block_size) {
    const auto& data = *ref.get();
    CHECK_HOST(input_mc_ptr != 0) << "pull RS needs symmetric input";
    CHECK_HOST(sem_mc_ptr != 0) << "pull RS needs multicast semaphores";
    CHECK_HOST(num_blocks > 0 && num_blocks <= data.num_pull_blocks);
    CHECK_HOST(block_size >= 32 && block_size <= 1024 && block_size % 32 == 0);
    auto params = make_params(data, input, output, residual, residual_is_local, 0);
    params.input_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(input_mc_ptr));
    params.sem_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(sem_mc_ptr));
    const auto kernel = residual.has_value() ? sp_collective::reduce_scatter_pull_kernel<kWorldSize, true, kUsePDL>
                                             : sp_collective::reduce_scatter_pull_kernel<kWorldSize, false, kUsePDL>;
    host::LaunchKernel(num_blocks, block_size, input.device()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
