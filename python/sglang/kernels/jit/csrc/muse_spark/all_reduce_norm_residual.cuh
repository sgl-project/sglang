// Fused bf16 push all-reduce + post-norm + gated residual + pre-norm (MuseSpark attention boundary).
// All-reduces the o_proj partial with 1-shot or 2-shot push on the CustomAllReduceV2 plane and feeds the reduced row
// to the row functions of norm_residual.cuh; outputs are bitwise those of 1shot_push / 2shot_push + the norm kernel.
#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "norm_residual.cuh"
#include <algorithm>
#include <cstdint>

namespace sglang {

using host::distributed::CommunicatorRef;

namespace muse_ar_norm {

template <uint32_t kWorld>
struct Params {
  device::distributed::PushWorkSpace<kWorld> ws;
  uint32_t num_counters;
  uint32_t rank;
  uint32_t num_rows;
  const bf16_t* partial;
  NormResidualParams norm;
};

using BF16Vec = device::AlignedVector<bf16x2_t, device::norm_residual::kPairs>;
using Lamport = device::distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;

SGL_DEVICE void st_multimem_relaxed_16B(const BF16Vec& x, void* mc_addr, int64_t vec_offset) {
  const auto val = *reinterpret_cast<const uint4*>(&x);
  mc_addr = static_cast<uint8_t*>(mc_addr) + vec_offset * 16;
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(mc_addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
               : "memory");
}

template <int N>
inline constexpr uint32_t kLowBits = N == 32 ? ~0u : (1u << N) - 1;

/// Spin until every vector v[i] with bit i of `mask` set has arrived at slots[i] + vid[i]; all loads of a round are
/// issued before any is tested, so one round trip covers every source.
template <int N>
SGL_DEVICE void poll_all(BF16Vec (&v)[N], void* const (&slots)[N], const int64_t (&vid)[N], uint32_t mask) {
  static_assert(N <= 32, "one mask bit per polled vector");
  bool pending;
  do {
#pragma unroll
    for (int i = 0; i < N; ++i)
      if (mask >> i & 1) device::ptx::ld_relaxed_16B(v[i], slots[i], vid[i]);
    pending = false;
#pragma unroll
    for (int i = 0; i < N; ++i)
      pending |= (mask >> i & 1) && Lamport::has_pos_zero(v[i].data());
  } while (pending);
}

/// The plane's global epoch parity: every counter holds the same value; a block reads and flips its own, and block 0
/// advances the counters of blocks this grid does not launch.
template <uint32_t kWorld>
struct Epoch {
  uint8_t* const* workspaces;
  uint32_t slot_bytes;
  uint32_t value;
  uint8_t* inbox;  // this rank's slots of this epoch, so polls index no parameter array
  SGL_DEVICE Epoch(const device::distributed::PushWorkSpace<kWorld>& ws, uint32_t rank)
      : workspaces(ws.workspaces.data()),
        slot_bytes(ws.slot_bytes),
        value(ws.counter[blockIdx.x].get()),
        inbox(ws.workspaces[rank] + (value & 1) * kWorld * slot_bytes) {}
  SGL_DEVICE uint32_t slot_offset(uint32_t src) const {
    return (src + (value & 1) * kWorld) * slot_bytes;
  }
  SGL_DEVICE void* slot_ptr(uint32_t dst, uint32_t src) const {
    return workspaces[dst] + slot_offset(src);
  }
  SGL_DEVICE void* in_slot(uint32_t src) const {
    return inbox + src * slot_bytes;
  }
};

/// Per-thread exchange of one row: pack p of a thread covers vector `vec_index(p)` of the row, the same map as the norm
/// kernels (cluster rank major within a pack).
template <int64_t kHidden, uint32_t kWorld, int kThreads, int kCluster, bool kTwoShot, bool kMulticast>
struct Exchange {
  static constexpr int kTotal = kThreads * kCluster;
  static constexpr int kPacks = kHidden / (kTotal * 8);
  static constexpr int kRowVecs = kHidden / 8;
  static constexpr int kSliceVecs = kRowVecs / kWorld;
  static_assert(kHidden % (kTotal * 8) == 0 && kRowVecs % kWorld == 0);

  const Params<kWorld>& params;
  Epoch<kWorld> epoch;
  int cluster_rank;

  SGL_DEVICE int vec_index(int p) const {
    return cluster_rank * kThreads + threadIdx.x + p * kTotal;
  }
  SGL_DEVICE int64_t gather_base() const {
    return int64_t(params.num_rows) * kSliceVecs;
  }

  /// Phase 1: this rank's partial of the row to every rank (1-shot) or to each slice owner (2-shot).
  SGL_DEVICE void push(uint32_t row) const {
    const auto r = params.rank;
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = vec_index(p);
      // Load this rank's partial from global memory; send +0 as -0 so it is not read as "not arrived".
      BF16Vec v;
      v.load(params.partial + int64_t(row) * kHidden, idx);
      Lamport::clear_pos_zero(v.data());
      if constexpr (!kTwoShot) {
        // 1-shot: store into every peer's slot for this rank (one multicast store, or one store per peer).
        const int64_t vid = int64_t(row) * kRowVecs + idx;
        if constexpr (kMulticast) {
          st_multimem_relaxed_16B(v, static_cast<uint8_t*>(params.ws.mc_workspace) + epoch.slot_offset(r), vid);
        } else {
#pragma unroll
          for (uint32_t dst = 0; dst < kWorld; ++dst)
            if (dst != r) device::ptx::st_relaxed_16B(v, epoch.slot_ptr(dst, r), vid);
        }
      } else {
        // 2-shot: store into the slot of the rank that owns this column slice; the owner keeps its own.
        const uint32_t owner = idx / kSliceVecs;
        if (owner != r) {
          const int64_t vid = int64_t(row) * kSliceVecs + idx % kSliceVecs;
          device::ptx::st_relaxed_16B(v, epoch.slot_ptr(owner, r), vid);
        }
      }
    }
  }

  /// 2-shot phase 2 for one row: reduce this rank's slice and send it to every rank's gather half. Spread over all
  /// `num_threads` threads of the block (cluster); the slots decouple it from the norm's map.
  SGL_DEVICE void reduce_slice(uint32_t row, int thread, int num_threads) const {
    if (row >= params.num_rows) return;
    const auto r = params.rank;
    BF16Vec zero;
    Lamport::fill_pos_zero(zero.data());
    for (int j = thread; j < kSliceVecs; j += num_threads) {
      const int64_t vid = int64_t(row) * kSliceVecs + j;
      // Own slice from global memory (L2-hit by push: it loaded it, under the norm's thread map, not this one).
      BF16Vec own;
      own.load(params.partial + int64_t(row) * kHidden, int(r) * kSliceVecs + j);
      BF16Vec vec[kWorld];
      void* slots[kWorld];
      int64_t vids[kWorld];
#pragma unroll
      for (uint32_t i = 0; i < kWorld; ++i)
        slots[i] = epoch.in_slot(i), vids[i] = vid;
      // Wait for every peer's copy of the vector, then sum all ranks in rank order in fp32.
      poll_all(vec, slots, vids, kLowBits<kWorld> & ~(1u << r));
#pragma unroll
      for (uint32_t i = 0; i < kWorld; ++i)
        if (i == r) vec[i] = own;
      auto out = device::reduce_vec(vec);
      // Send the reduced vector to every rank's gather half.
      Lamport::clear_pos_zero(out.data());
      const int64_t gid = gather_base() + vid;
      if constexpr (kMulticast) {
        st_multimem_relaxed_16B(out, static_cast<uint8_t*>(params.ws.mc_workspace) + epoch.slot_offset(r), gid);
      } else {
#pragma unroll
        for (uint32_t dst = 0; dst < kWorld; ++dst)
          device::ptx::st_relaxed_16B(out, epoch.slot_ptr(dst, r), gid);
      }
      // Re-arm the slots read for the next launch.
#pragma unroll
      for (uint32_t i = 0; i < kWorld; ++i)
        if (i != r) device::ptx::st_global_16B(zero, slots[i], vid);
    }
  }

  /// The reduced row for this thread's packs; the slots read are re-armed. Every pack's (and source's) load is in
  /// flight at once. 1-shot unicast takes this rank's own partial from `partial` (remapped like a pushed copy).
  SGL_DEVICE void gather(uint32_t row, BF16Vec (&out)[kPacks]) const {
    const auto r = params.rank;
    BF16Vec zero;
    Lamport::fill_pos_zero(zero.data());
    if constexpr (!kTwoShot) {
      constexpr int N = kPacks * kWorld;
      BF16Vec vec[N], own[kPacks];
      void* slots[N];
      int64_t vids[N];
      uint32_t mask = 0;
#pragma unroll
      for (int p = 0; p < kPacks; ++p) {
#pragma unroll
        for (uint32_t i = 0; i < kWorld; ++i) {
          slots[p * kWorld + i] = epoch.in_slot(i);
          vids[p * kWorld + i] = int64_t(row) * kRowVecs + vec_index(p);
          if (kMulticast || i != r) mask |= 1u << (p * kWorld + i);
        }
        // Unicast skips the self push: take this rank's partial from global memory (L2-hot from push).
        if constexpr (!kMulticast) {
          own[p].load(params.partial + int64_t(row) * kHidden, vec_index(p));
          Lamport::clear_pos_zero(own[p].data());
        }
      }
      // Wait for every rank's copy of every pack, then sum each pack in rank order and re-arm its slots.
      poll_all(vec, slots, vids, mask);
      if constexpr (!kMulticast) {
#pragma unroll
        for (int p = 0; p < kPacks; ++p)
#pragma unroll
          for (uint32_t i = 0; i < kWorld; ++i)
            if (i == r) vec[p * kWorld + i] = own[p];
      }
#pragma unroll
      for (int p = 0; p < kPacks; ++p) {
        BF16Vec srcs[kWorld];
#pragma unroll
        for (uint32_t i = 0; i < kWorld; ++i)
          srcs[i] = vec[p * kWorld + i];
        out[p] = device::reduce_vec(srcs);
#pragma unroll
        for (uint32_t i = 0; i < kWorld; ++i)
          if (kMulticast || i != r) device::ptx::st_global_16B(zero, slots[p * kWorld + i], vids[p * kWorld + i]);
      }
    } else {
      void* slots[kPacks];
      int64_t gids[kPacks];
#pragma unroll
      for (int p = 0; p < kPacks; ++p) {
        const int idx = vec_index(p);
        slots[p] = epoch.in_slot(idx / kSliceVecs);
        gids[p] = gather_base() + int64_t(row) * kSliceVecs + idx % kSliceVecs;
      }
      // 2-shot: wait for the owners' reduced vectors, then re-arm the slots.
      poll_all(out, slots, gids, kLowBits<kPacks>);
#pragma unroll
      for (int p = 0; p < kPacks; ++p)
        device::ptx::st_global_16B(zero, slots[p], gids[p]);
    }
  }

  SGL_DEVICE void finish() const {
    __syncthreads();
    if (threadIdx.x == 0) params.ws.counter[blockIdx.x].set(epoch.value ^ 1);
    if (blockIdx.x == 0)
      for (uint32_t i = gridDim.x + threadIdx.x; i < params.num_counters; i += blockDim.x)
        params.ws.counter[i].set(epoch.value ^ 1);
  }
};

}  // namespace muse_ar_norm

// C = 1: rows strided over the grid (bound to the plane's block counters), each through norm_residual_row.
template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    bool kTwoShot,
    bool kMulticast,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kHasRouter,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_spark_all_reduce_norm_residual_kernel(const __grid_constant__ muse_ar_norm::Params<kWorld> params) {
  using Ex = muse_ar_norm::Exchange<kHidden, kWorld, kNumThreads, 1, kTwoShot, kMulticast>;
  // Wait for the o_proj GEMM that writes `partial`.
  device::PDLWaitPrimary<kUsePDL>();
  // Read this block's epoch parity: which half of the double-buffered slots this launch uses.
  const Ex ex{params, muse_ar_norm::Epoch<kWorld>{params.ws, params.rank}, 0};
  // Phase 1: push every row this block owns to its peers.
  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x)
    ex.push(row);
  // 2-shot phase 2 for the first row. The slice of row k + 1 is reduced before row k is gathered, so the norm's HBM
  // traffic for row k overlaps the network wait for row k + 1.
  if constexpr (kTwoShot) ex.reduce_slice(blockIdx.x, threadIdx.x, kNumThreads);
  // Every input is read or pushed: let the next kernel launch.
  device::PDLTriggerSecondary<kUsePDL>();
  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    // 2-shot phase 2 for the next row.
    if constexpr (kTwoShot) ex.reduce_slice(row + gridDim.x, threadIdx.x, kNumThreads);
    // Wait for the reduced row and keep it in registers.
    muse_ar_norm::BF16Vec branch[Ex::kPacks];
    ex.gather(row, branch);
    // Post-norm + gated residual + pre-norm on the reduced row.
    norm_residual_row<
        kHidden,
        kNumThreads,
        kUsePDL,
        kPrefetchWeights,
        kPostNormIsInputScale,
        kHasRouter,
        kPostWeight,
        kGate>(params.norm, row, [&branch](int pack, int) { return branch[pack]; });
  }
  // Flip the epoch parity for the next launch on the plane.
  ex.finish();
}

// C > 1: one cluster per row, through norm_residual_cluster_row.
template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    int kClusterSize,
    bool kTwoShot,
    bool kMulticast,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kHasRouter,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_spark_all_reduce_norm_residual_cluster_kernel(const __grid_constant__ muse_ar_norm::Params<kWorld> params) {
  using Ex = muse_ar_norm::Exchange<kHidden, kWorld, kNumThreads, kClusterSize, kTwoShot, kMulticast>;
  const int cluster_rank = cooperative_groups::this_cluster().block_rank();
  const uint32_t row = blockIdx.x / kClusterSize;
  // Wait for the o_proj GEMM that writes `partial`.
  device::PDLWaitPrimary<kUsePDL>();
  // Read this block's epoch parity: which half of the double-buffered slots this launch uses.
  const Ex ex{params, muse_ar_norm::Epoch<kWorld>{params.ws, params.rank}, cluster_rank};
  // Phase 1: push this block's part of the row to its peers.
  ex.push(row);
  // 2-shot phase 2: the cluster reduces this rank's slice of the row and sends it to every rank.
  if constexpr (kTwoShot) {
    ex.reduce_slice(row, cluster_rank * kNumThreads + threadIdx.x, kNumThreads * kClusterSize);
  }
  // Every input is read or pushed: let the next kernel launch.
  device::PDLTriggerSecondary<kUsePDL>();
  // Wait for the reduced row and keep it in registers.
  muse_ar_norm::BF16Vec branch[Ex::kPacks];
  ex.gather(row, branch);
  // Post-norm + gated residual + pre-norm on the reduced row, statistics shared across the cluster.
  norm_residual_cluster_row<
      kHidden,
      kNumThreads,
      kClusterSize,
      kUsePDL,
      kPrefetchWeights,
      kPostNormIsInputScale,
      kHasRouter,
      kPostWeight,
      kGate>(params.norm, row, [&branch](int pack, int) { return branch[pack]; });
  // Flip the epoch parity for the next launch on the plane.
  ex.finish();
}

template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    int kClusterSize,
    bool kTwoShot,
    bool kMulticast,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kHasRouter>
struct MuseSparkAllReduceNormResidual {
  static void
  run(CommunicatorRef ref,
      const tvm::ffi::TensorView partial,
      const tvm::ffi::TensorView stream,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView new_stream,
      const tvm::ffi::Optional<tvm::ffi::TensorView> router_out,
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
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(partial);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(stream);
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device_).verify(output);
    TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(new_stream);
    RuntimeCheck(router_out.has_value() == kHasRouter, "router_out presence must match kHasRouter");
    if constexpr (kHasRouter)
      TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device_).verify(router_out.value());
    if (post_weight.has_value())
      TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(post_weight.value());
    if (pre_weight.has_value()) TensorMatcher({H}).with_dtype<bf16_t>().with_device(device_).verify(pre_weight.value());
    const bool gate = alpha.has_value();
    RuntimeCheck(gate == beta.has_value(), "alpha and beta must both be present or both absent");
    bool gate_channel_wise = false;
    if (gate) {
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(alpha.value());
      TensorMatcher({G}).with_dtype<fp32_t>().with_device(device_).verify(beta.value());
      const auto n = G.unwrap();
      RuntimeCheck(n == 1 || n == kHidden, "alpha/beta must have 1 or hidden_size elements");
      gate_channel_wise = (n == kHidden);
    }
    RuntimeCheck(!kMulticast || push.mc_workspace != nullptr, "multicast needs a push plane with an mc workspace");

    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    if (num_rows == 0) return;
    RuntimeCheck(push.world_size == kWorld, "compiled for world ", kWorld);
    const int64_t slot = kTwoShot ? int64_t(2) * num_rows * (kHidden / kWorld) * 2 : int64_t(num_rows) * kHidden * 2;

    const auto params = muse_ar_norm::Params<kWorld>{
        .ws = push.get_workspace<kWorld>(slot),
        .num_counters = push.num_blocks,
        .rank = push.rank,
        .num_rows = num_rows,
        .partial = static_cast<const bf16_t*>(partial.data_ptr()),
        .norm =
            NormResidualParams{
                .branch = nullptr,
                .stream = stream.data_ptr(),
                .output = output.data_ptr(),
                .new_stream = new_stream.data_ptr(),
                .router_out = kHasRouter ? router_out.value().data_ptr() : nullptr,
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

    const auto launch = [&]<bool kHasPostWeight, bool kHasGate>() {
      if constexpr (kClusterSize == 1) {
        constexpr auto kernel = muse_spark_all_reduce_norm_residual_kernel<
            kHidden,
            kWorld,
            kNumThreads,
            kTwoShot,
            kMulticast,
            kUsePDL,
            kPrefetchWeights,
            kPostNormIsInputScale,
            kHasRouter,
            kHasPostWeight,
            kHasGate>;
        // Every block must be resident at once: a block polls the same block on every peer.
        static const uint32_t max_blocks = [kernel] {
          int per_sm = 0, dev = 0, sms = 0;
          RuntimeDeviceCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kNumThreads, 0));
          RuntimeDeviceCheck(cudaGetDevice(&dev));
          RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
          return uint32_t(per_sm * sms);
        }();
        LaunchKernel(std::min({num_rows, max_blocks, push.num_blocks}), kNumThreads, device)
            .config({.use_pdl = kUsePDL})(kernel, params);
      } else {
        constexpr auto kernel = muse_spark_all_reduce_norm_residual_cluster_kernel<
            kHidden,
            kWorld,
            kNumThreads,
            kClusterSize,
            kTwoShot,
            kMulticast,
            kUsePDL,
            kPrefetchWeights,
            kPostNormIsInputScale,
            kHasRouter,
            kHasPostWeight,
            kHasGate>;
        // Every cluster must be resident at once: a block polls the same block on every peer.
        static const int max_clusters = [kernel] {
          if constexpr (kClusterSize > 8) {
            RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
          }
          cudaLaunchConfig_t config{};
          cudaLaunchAttribute attr{};
          attr.id = cudaLaunchAttributeClusterDimension;
          attr.val.clusterDim = {kClusterSize, 1, 1};
          config.gridDim = dim3(kClusterSize);
          config.blockDim = dim3(kNumThreads);
          config.attrs = &attr;
          config.numAttrs = 1;
          int n = 0;
          RuntimeDeviceCheck(cudaOccupancyMaxActiveClusters(&n, kernel, &config));
          return n;
        }();
        RuntimeCheck(
            int(num_rows) <= max_clusters, "rows exceed co-resident clusters: ", num_rows, " > ", max_clusters);
        RuntimeCheck(num_rows * kClusterSize <= push.num_blocks, "M * cluster_size exceeds the plane's block counters");
        LaunchKernel(num_rows * kClusterSize, kNumThreads, device)
            .config({.use_pdl = kUsePDL, .cluster_dim = dim3(kClusterSize, 1, 1)})(kernel, params);
      }
    };
    if (post_weight.has_value()) {
      if (gate)
        launch.template operator()<true, true>();
      else
        launch.template operator()<true, false>();
    } else {
      if (gate)
        launch.template operator()<false, true>();
      else
        launch.template operator()<false, false>();
    }
  }
};

}  // namespace sglang
