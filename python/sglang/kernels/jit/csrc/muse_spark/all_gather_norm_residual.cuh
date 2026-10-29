// Fused bf16 all-gather + post-norm + gated residual + pre-norm (MuseSpark v1.2 FFN boundary).
#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "norm_residual.cuh"
#include <algorithm>
#include <cstdint>

namespace sglang {

namespace muse_ag_norm {

using device::distributed::Counter;

struct Params {
  uint8_t* ws_local;      // LOCAL VA of the push workspace base (poll + reset)
  uint8_t* ws_mc;         // multicast VA of the push workspace base (push)
  Counter* counter;       // per-block phase counters
  uint32_t num_counters;  // PushPlane::num_blocks
  uint32_t half_bytes;    // world_size * slot_bytes
  uint32_t rank;
  uint32_t num_rows;
  const bf16_t* local;  // [M, H / world]
  NormResidualParams norm;
};

inline constexpr uint32_t kVec = 8;  // bf16 per 16B
using BF16Vec = device::AlignedVector<bf16x2_t, device::norm_residual::kPairs>;

// Reads this block's phase and advances the counters no block of this grid owns.
template <int kNumThreads>
SGL_DEVICE uint32_t begin_phase(const Params& params) {
  const uint32_t phase = params.counter[blockIdx.x].get() & 1;
  if (blockIdx.x == 0) {
    for (uint32_t i = gridDim.x + threadIdx.x; i < params.num_counters; i += kNumThreads) {
      params.counter[i].set(phase ^ 1);
    }
  }
  return phase;
}

template <uint32_t kNLocal, int kNumThreads>
SGL_DEVICE void push_local_slice(const Params& params, uint32_t half_off) {
  const uint32_t num_local = params.num_rows * kNLocal;
  const uint32_t worker = blockIdx.x * kNumThreads + threadIdx.x;
  const uint32_t stride = gridDim.x * kNumThreads * kVec;
  for (uint32_t elem = worker * kVec; elem < num_local; elem += stride) {
    uint4 raw = *reinterpret_cast<const uint4*>(params.local + elem);
    uint32_t* w = reinterpret_cast<uint32_t*>(&raw);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      if ((w[j] & 0xffffu) == 0) w[j] |= 0x8000u;
      if ((w[j] >> 16) == 0) w[j] |= 0x80000000u;
    }
    const uint32_t byte_off = half_off + (params.rank * num_local + elem) * sizeof(bf16_t);
    device::ptx::st_multimem_16B(raw, params.ws_mc + byte_off, 0);
  }
}

SGL_DEVICE bool vector_pending(uint4 raw) {
  return (raw.x & 0xffffu) == 0 || (raw.x >> 16) == 0 || (raw.y & 0xffffu) == 0 || (raw.y >> 16) == 0 ||
         (raw.z & 0xffffu) == 0 || (raw.z >> 16) == 0 || (raw.w & 0xffffu) == 0 || (raw.w >> 16) == 0;
}

// Spins until every rank's copy of the vector at (row, col) arrived, then clears the slot.
template <uint32_t kNLocal>
SGL_DEVICE uint4 take_gathered(bf16_t* gathered, uint32_t num_rows, uint32_t row, uint32_t col) {
  bf16_t* src = gathered + ((col / kNLocal) * num_rows + row) * kNLocal + col % kNLocal;
  uint4 raw;
  bool pending;
  do {
    device::ptx::ld_relaxed_16B(raw, src, 0);
    pending = vector_pending(raw);
  } while (pending);
  device::AlignedVector<uint32_t, 4> zero;
  zero.fill(0);
  zero.store(src);
  return raw;
}

template <int kPacks, int kVectorStride, uint32_t kNLocal>
SGL_DEVICE uint4
receive_pack(bf16_t* gathered, uint32_t num_rows, uint32_t row, uint32_t col, int pack, uint4 (&cache)[2]) {
  if (pack % 2 == 0) {
    if (pack + 1 < kPacks) {
      const uint32_t next_col = col + kVectorStride * kVec;
      bf16_t* src0 = gathered + ((col / kNLocal) * num_rows + row) * kNLocal + col % kNLocal;
      bf16_t* src1 = gathered + ((next_col / kNLocal) * num_rows + row) * kNLocal + next_col % kNLocal;
      bool pending;
      do {
        device::ptx::ld_relaxed_16B(cache[0], src0, 0);
        device::ptx::ld_relaxed_16B(cache[1], src1, 0);
        pending = vector_pending(cache[0]) | vector_pending(cache[1]);
      } while (pending);
      device::AlignedVector<uint32_t, 4> zero;
      zero.fill(0);
      zero.store(src0);
      zero.store(src1);
    } else {
      cache[0] = take_gathered<kNLocal>(gathered, num_rows, row, col);
    }
  }
  return cache[pack % 2];
}

}  // namespace muse_ag_norm

// C > 1: one cluster per row through norm_residual_cluster_row.
template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    int kClusterSize,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_spark_all_gather_norm_residual_cluster_kernel(const __grid_constant__ muse_ag_norm::Params params) {
  constexpr uint32_t kNLocal = static_cast<uint32_t>(kHidden) / kWorld;
  static_assert(kNLocal % muse_ag_norm::kVec == 0, "a vector must never cross a rank block");
  const uint32_t row = blockIdx.x / kClusterSize;

  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = muse_ag_norm::begin_phase<kNumThreads>(params);
  const uint32_t half_off = phase * params.half_bytes;
  muse_ag_norm::push_local_slice<kNLocal, kNumThreads>(params, half_off);
  device::PDLTriggerSecondary<kUsePDL>();

  const auto gathered = reinterpret_cast<bf16_t*>(params.ws_local + half_off);
  norm_residual_cluster_row<
      kHidden,
      kNumThreads,
      kClusterSize,
      kUsePDL,
      kPrefetchWeights,
      kPostNormIsInputScale,
      false,
      kPostWeight,
      kGate>(params.norm, row, [&](int, int index) {
    const uint4 raw =
        muse_ag_norm::take_gathered<kNLocal>(gathered, params.num_rows, row, uint32_t(index) * muse_ag_norm::kVec);
    return *reinterpret_cast<const muse_ag_norm::BF16Vec*>(&raw);
  });

  // Every slot this block consumed was reset above; publish the next phase.
  __syncthreads();
  if (threadIdx.x == 0) params.counter[blockIdx.x].set(phase ^ 1);
}

// C == 1: persistent kernels, each owning rows bx, bx + grid, ... through norm_residual_row.
template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_spark_all_gather_norm_residual_row_kernel(const __grid_constant__ muse_ag_norm::Params params) {
  constexpr int kPacks = kHidden / (kNumThreads * muse_ag_norm::kVec);
  constexpr uint32_t kNLocal = static_cast<uint32_t>(kHidden) / kWorld;
  static_assert(kNLocal % muse_ag_norm::kVec == 0, "a vector must never cross a rank block");

  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t phase = muse_ag_norm::begin_phase<kNumThreads>(params);
  const uint32_t half_off = phase * params.half_bytes;
  muse_ag_norm::push_local_slice<kNLocal, kNumThreads>(params, half_off);
  device::PDLTriggerSecondary<kUsePDL>();

  const auto gathered = reinterpret_cast<bf16_t*>(params.ws_local + half_off);
  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    uint4 receive_cache[2];
    norm_residual_row<
        kHidden,
        kNumThreads,
        kUsePDL,
        kPrefetchWeights,
        kPostNormIsInputScale,
        false,
        kPostWeight,
        kGate>(params.norm, row, [&](int pack, int idx) {
      const uint4 raw = muse_ag_norm::receive_pack<kPacks, kNumThreads, kNLocal>(
          gathered, params.num_rows, row, uint32_t(idx) * muse_ag_norm::kVec, pack, receive_cache);
      return *reinterpret_cast<const muse_ag_norm::BF16Vec*>(&raw);
    });
  }

  __syncthreads();
  if (threadIdx.x == 0) params.counter[blockIdx.x].set(phase ^ 1);
}

using host::distributed::CommunicatorRef;

template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    int kClusterSize,
    bool kUsePDL,
    bool kPostNormIsInputScale,
    bool kPrefetchWeights,
    bool kHasPostWeight,
    bool kHasGate>
struct MuseSparkAllGatherNormResidual {
  static_assert(kClusterSize == 1 || kClusterSize == 2 || kClusterSize == 4 || kClusterSize == 8 || kClusterSize == 16);
  static constexpr int64_t kNLocal = kHidden / kWorld;

  static void
  run(CommunicatorRef ref,
      const tvm::ffi::TensorView local,
      const tvm::ffi::TensorView stream,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView new_stream,
      const tvm::ffi::Optional<tvm::ffi::TensorView> post_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> pre_weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> alpha,
      const tvm::ffi::Optional<tvm::ffi::TensorView> beta,
      double post_eps,
      double pre_eps) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();

    auto M = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto G = SymbolicSize{"gate_dim"};
    auto device_ = SymbolicDevice{};
    H.set_value(kHidden);
    device_.set_options<kDLCUDA>();

    TensorMatcher({M, kNLocal}).with_dtype<bf16_t>().with_device(device_).verify(local);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(output);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(new_stream);
    if (post_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(post_weight.value());
    }
    if (pre_weight.has_value()) {
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(pre_weight.value());
    }
    const bool gate = alpha.has_value();
    RuntimeCheck(gate == beta.has_value(), "alpha and beta must both be present or both absent");
    // One instantiation per module keeps JIT builds cheap; the caller picks it.
    RuntimeCheck(post_weight.has_value() == kHasPostWeight, "post_weight presence must match kHasPostWeight");
    RuntimeCheck(gate == kHasGate, "alpha/beta presence must match kHasGate");
    bool gate_channel_wise = false;
    if (gate) {
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(alpha.value());
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(beta.value());
      const auto n = G.unwrap();
      RuntimeCheck(n == 1 || n == kHidden, "alpha/beta must have 1 or hidden_size elements");
      gate_channel_wise = (n == kHidden);
    }

    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(num_rows >= 1, "fused all-gather norm requires M >= 1");
    RuntimeCheck(push.world_size == kWorld, "the kernel is compiled for TP", kWorld);
    RuntimeCheck(push.mc_workspace != nullptr, "requires a multicast-capable push plane");
    RuntimeCheck(
        int64_t(num_rows) * kHidden * 2 <= int64_t(push.slot_bytes), "gathered rows exceed the push slot size");
    uint32_t grid = num_rows * kClusterSize;
    if constexpr (kClusterSize == 1) {
      grid = std::min<uint32_t>(num_rows, push.num_blocks);
    } else {
      RuntimeCheck(
          int64_t(num_rows) * kClusterSize <= int64_t(push.num_blocks),
          "M * cluster_size exceeds the push plane's block counters");
    }

    const auto params = muse_ag_norm::Params{
        .ws_local = push.workspaces[push.rank],
        .ws_mc = push.mc_workspace,
        .counter = push.counter,
        .num_counters = push.num_blocks,
        .half_bytes = static_cast<uint32_t>(push.slot_bytes * push.world_size),
        .rank = push.rank,
        .num_rows = num_rows,
        .local = static_cast<const bf16_t*>(local.data_ptr()),
        .norm =
            NormResidualParams{
                .branch = nullptr,
                .stream = stream.data_ptr(),
                .output = output.data_ptr(),
                .new_stream = new_stream.data_ptr(),
                .router_out = nullptr,
                .post_weight = post_weight.has_value() ? post_weight.value().data_ptr() : nullptr,
                .pre_weight = pre_weight.has_value() ? pre_weight.value().data_ptr() : nullptr,
                .alpha = gate ? alpha.value().data_ptr() : nullptr,
                .beta = gate ? beta.value().data_ptr() : nullptr,
                .post_eps = static_cast<float>(post_eps),
                .pre_eps = static_cast<float>(pre_eps),
                .num_rows = static_cast<int>(num_rows),
                .gate_channel_wise = gate_channel_wise,
                .has_pre_weight = pre_weight.has_value(),
                .pre_norm_bf16 = false,
            },
    };
    const auto device = device_.unwrap();
    if constexpr (kClusterSize == 1) {
      LaunchKernel(grid, kNumThreads, device)
          .config({.use_pdl = kUsePDL})(
              muse_spark_all_gather_norm_residual_row_kernel<
                  kHidden,
                  kWorld,
                  kNumThreads,
                  kUsePDL,
                  kPrefetchWeights,
                  kPostNormIsInputScale,
                  kHasPostWeight,
                  kHasGate>,
              params);
    } else {
      constexpr auto kernel = muse_spark_all_gather_norm_residual_cluster_kernel<
          kHidden,
          kWorld,
          kNumThreads,
          kClusterSize,
          kUsePDL,
          kPrefetchWeights,
          kPostNormIsInputScale,
          kHasPostWeight,
          kHasGate>;
      [[maybe_unused]] static const bool cluster_configured = [kernel] {
        if constexpr (kClusterSize > 8) {
          RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        }
        cudaLaunchConfig_t occupancy_config{};
        occupancy_config.gridDim = dim3(kClusterSize);
        occupancy_config.blockDim = dim3(kNumThreads);
        int max_cluster_size = 0;
        RuntimeDeviceCheck(cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, kernel, &occupancy_config));
        RuntimeCheck(
            max_cluster_size >= kClusterSize,
            "Requested cluster size ",
            kClusterSize,
            " exceeds this kernel/device limit ",
            max_cluster_size);
        return true;
      }();
      LaunchKernel(grid, kNumThreads, device)
          .config({
              .use_pdl = kUsePDL,
              .cluster_dim = dim3(kClusterSize, 1, 1),
          })(kernel, params);
    }
  }
};

}  // namespace sglang
