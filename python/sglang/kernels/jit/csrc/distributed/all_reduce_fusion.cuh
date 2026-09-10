// Fused deferred-MoE finalize -> 1shot lamport push all-reduce [-> RMSNorm]
// over the CustomAllReduceV2 push plane, for decode-sized batches (bf16).
//
// A generalisation of the K3 `finalize_push_norm` kernel
// (csrc/kimi_k3/comm/ar_fusion.cuh): the hidden width, top_k and cluster
// geometry are template parameters chosen from Python, the shared-expert add
// and the RMSNorm epilogue are optional, and the result goes to a separate
// output tensor. Per token row t:
//
//   local[t] = sum_k expert_weights[t, k] * gemm2_out[idx[t * top_k + k]]
//              (+ shared_output[t])                     -- stage 1, registers only
//   out[t]   = sum over ranks of local[t]               -- stage 2
//   out[t]   = out[t] * rsqrt(mean(out[t]^2) + eps) * w -- kNorm only
//
// `idx == -1` marks a dropped slot (EP: the token was routed to an expert
// that is not local) and contributes nothing. Accumulation is fp32; the bf16
// rounding points are exactly the unfused path's: the routed combine (what
// TRT-LLM's finalize returns), the `+ shared` (torch's bf16 add) and the
// all-reduce output, so the plain (kNorm=false) result equals TRT-LLM finalize
// -> `shared.add_(routed)` -> fp32-accumulating bf16 all-reduce in rank order,
// and the staged vector is bit-identical to what the unfused path reduces.
//
// The rank-local finalize never materializes in global memory: each thread
// computes one 16B vector of it and pushes it straight into every peer's push
// slot with unicast `st.relaxed.sys` stores, exactly like the generic
// `all_reduce_1shot_push_kernel`, so no multicast mapping is required.
//
// Push-plane protocol (see include/sgl_kernel/distributed/communicator.cuh):
//   * every rank owns 2 phases x kWorldSize slots of `slot_bytes`; a round
//     uses phase `counter & 1`, producer r writes slot r of every peer, the
//     consumer polls its own kWorldSize slots until no +0.0 marker remains,
//     reduces, and restores the +0.0 markers before it exits;
//   * +0.0 payload words are remapped to -0.0 (numerically identical) so a
//     written word is never 0 and `word == 0` means "not arrived yet";
//   * the phase counters are per block of the GENERIC push kernel, which
//     launches `num_blocks` blocks and flips one counter each. This kernel
//     uses one counter per row cluster (flipped by the cluster's leader block
//     after a cluster barrier, since every block of the cluster reads it) and
//     a trailing "bumper" cluster flips every remaining one, so the whole
//     array keeps one parity and the two kernel families can share the plane
//     freely (single-stream calls are serialized);
//   * every rank must call with the same num_tokens / hidden / top_k / epilogue:
//     slots are addressed by 16B vector index of the [T, hidden] row view.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/extra/stl.h>

#include <cooperative_groups.h>
#include <cstdint>
#include <optional>

namespace sglang {

using device::distributed::PushWorkSpace;
using host::distributed::CommunicatorRef;

/// One 16B staging vector (8 bf16) viewed as the 4 u32 words the lamport marker
/// protocol tests, matching the generic push kernel's `LamportTrait<T, 8, 4>`.
using Lamport = device::distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;
using StageVec = device::AlignedVector<bf16x2_t, 4>;

SGL_DEVICE void barrier_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

SGL_DEVICE void barrier_cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

template <uint32_t kWorldSize>
struct FinalizeAllReduceParams {
  bf16_t* out;                // [num_tokens, kHiddenDim], output-only
  const bf16_t* gemm2;        // [P, kHiddenDim], permuted / padded rows
  const int32_t* idx;         // [num_tokens * kTopK], -1 = dropped slot
  const bf16_t* weights;      // [num_tokens, kTopK], scaling already folded in
  const bf16_t* shared;       // [num_tokens, kHiddenDim] (kHasShared only)
  const bf16_t* norm_weight;  // [kHiddenDim] (kNorm only)
  float norm_eps;             // kNorm only
  // Caller's promise that everything this kernel reads before its PDL wait is
  // complete when the preceding kernel merely *triggers*: no all-reduce on this
  // plane right before it, and the routing metadata's producers finished (not
  // just the immediate predecessor -- PDL completion is not transitive through
  // early-triggering kernels). False (the safe default) waits first.
  bool prefetch_metadata;
  uint32_t rank;
  uint32_t num_tokens;
  uint32_t num_push_counters;  // full counter array size (bumper range end)
  PushWorkSpace<kWorldSize> ws;
};

/// Row geometry: one 16B vector per thread, one cluster per row, so the block
/// size follows from the hidden width and the cluster size. The cluster size
/// is the tuning knob (dims per block = kHiddenDim / kClusterSize).
template <uint32_t kHiddenDim, uint32_t kClusterSize>
struct AllReduceNormTrait {
  static constexpr uint32_t kRowVecs = kHiddenDim / 8;             // 16B vectors per row
  static constexpr uint32_t kBlockSize = kRowVecs / kClusterSize;  // threads per block
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static_assert(kHiddenDim % 8 == 0, "hidden must be a whole number of 16B vectors");
  static_assert(1 <= kClusterSize && kClusterSize <= 8, "portable cluster sizes only");
  static_assert(kRowVecs % kClusterSize == 0, "cluster size must divide the row's vector count");
  static_assert(kBlockSize % device::kWarpThreads == 0, "block must be whole warps");
  static_assert(kBlockSize <= 1024, "block too large: raise the cluster size");
};

// --- stage 1: the deferred finalize of one 16B vector ------------------------
// The shared-expert vector is loaded first so that load is in flight while the
// routing rows and the kTopK gathers are fetched; the routed combine is then
// accumulated in ascending k from zero, rounded to bf16, and the shared vector
// is added with one more bf16 rounding (see the header: the unfused path's
// numerics, reproduced so the fusion does not move the greedy output).
// Threads of the same token read the same kTopK indices / weights (a broadcast
// load per warp). All arithmetic is on bf16 pairs: one cast converts two
// elements.
template <uint32_t kHiddenDim, uint32_t kTopK, bool kHasShared, bool kUsePDL, uint32_t kWorldSize>
SGL_DEVICE StageVec finalize_vec(const FinalizeAllReduceParams<kWorldSize>& params, uint32_t token, uint32_t hvec) {
  using namespace device;
  const auto* idx = params.idx + static_cast<int64_t>(token) * kTopK;
  const auto* weights = params.weights + static_cast<int64_t>(token) * kTopK;
  int32_t rows[kTopK];
  bf16_t w[kTopK];
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    rows[k] = idx[k];
    w[k] = weights[k];
  }

  // delay PDL wait until here
  PDLWaitPrimary<kUsePDL>();

  StageVec shared_in;
  if constexpr (kHasShared) {
    shared_in.load(params.shared + static_cast<int64_t>(token) * kHiddenDim, hvec);
  }

  StageVec in[kTopK];
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    if (rows[k] >= 0) in[k].load(params.gemm2 + static_cast<int64_t>(rows[k]) * kHiddenDim, hvec);
  }

  fp32x2_t acc[4];
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    acc[j] = fp32x2_t{0.0f, 0.0f};
  }

#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    if (rows[k] < 0) continue;
#if SGL_ARCH_BLACKWELL_OR_GREATER
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      acc[j].x = math::fma_f32_bf16(in[k][j].x, w[k], acc[j].x);
      acc[j].y = math::fma_f32_bf16(in[k][j].y, w[k], acc[j].y);
    }
#else
    const auto w_fp32 = cast<fp32_t>(w[k]);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const auto [x, y] = cast<fp32x2_t>(in[k][j]);
      acc[j].x = fmaf(x, w_fp32, acc[j].x);
      acc[j].y = fmaf(y, w_fp32, acc[j].y);
    }
#endif
  }
  // Same rounding as the unfused path: TRT-LLM's finalize returns the routed
  // combine rounded to bf16, and `shared.add_(routed)` then rounds the bf16 +
  // bf16 sum once more (torch adds in fp32). Reproducing both roundings keeps
  // the staged vector bit-identical to what `dev` all-reduces, so the greedy
  // output does not move with the fusion.
  StageVec out;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    if constexpr (kHasShared) {
      const auto routed = cast<fp32x2_t>(cast<bf16x2_t>(acc[j]));
      const auto sh = cast<fp32x2_t>(shared_in[j]);
      out[j] = cast<bf16x2_t>(fp32x2_t{routed.x + sh.x, routed.y + sh.y});
    } else {
      out[j] = cast<bf16x2_t>(acc[j]);
    }
  }
  return out;
}

// --- the kernel --------------------------------------------------------------
// Grid: dim3(num_tokens [+ 1], kClusterSize) with the cluster laid along y, so
// blockIdx.x is the token row (and its phase counter: PushEpoch's default) and
// blockIdx.y the rank inside the cluster. When rows do not own every counter
// of the plane, one extra cluster (blockIdx.x == num_tokens) is the bumper: it
// only flips the counters [num_tokens, num_push_counters) and exits; with
// num_tokens == num_push_counters no bumper is launched. The plane holds
// num_sm counters, so decode batches always fit.
template <
    uint32_t kWorldSize,
    uint32_t kHiddenDim,
    uint32_t kTopK,
    uint32_t kClusterSize,
    bool kUsePDL,
    bool kHasShared,
    bool kNorm>
__global__ __launch_bounds__(AllReduceNormTrait<kHiddenDim, kClusterSize>::kBlockSize)
    __cluster_dims__(1, kClusterSize, 1) void moe_finalize_all_reduce_kernel(
        const __grid_constant__ FinalizeAllReduceParams<kWorldSize> params) {
  namespace cg = cooperative_groups;
  using namespace device;
  using T = AllReduceNormTrait<kHiddenDim, kClusterSize>;
  constexpr uint32_t kRowVecs = T::kRowVecs;
  constexpr uint32_t kBlockSize = T::kBlockSize;
  constexpr uint32_t kNumWarps = T::kNumWarps;

  const auto tx = threadIdx.x;
  const auto row_idx = blockIdx.x;
  const auto cluster_rank = blockIdx.y;
  // this thread's vector within a row: cluster rank picks the block's chunk
  const auto hvec = cluster_rank * kBlockSize + tx;

  // Under PDL this grid may start while the preceding kernel is still running.
  // If that kernel is an all-reduce on this plane, it is still flipping the
  // phase counters and resetting slot markers: reading the epoch now would see
  // a half-done state and the poll below would never complete. Only a caller
  // who knows the predecessor is a compute kernel (the MoE GEMM in the model)
  // may defer the wait to finalize_vec, past the routing-metadata prefetch;
  // the second wait there is then a no-op.
  if (!params.prefetch_metadata) PDLWaitPrimary<kUsePDL>();

  if (row_idx == params.num_tokens) {
    PDLWaitPrimary<kUsePDL>();
    if (cluster_rank == 0) {
      const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};
      __syncthreads();
      epoch.unsafe_flip_range(row_idx, params.num_push_counters);
    }
    return PDLTriggerSecondary<kUsePDL>();
  }

  // this cluster's epoch: the counter at blockIdx.x, one per row cluster
  // (every block of the cluster reads the same one)
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};
  const auto r = params.rank;
  // my slot (`src = r`) inside every peer's workspace, and every peer's slot
  // inside mine (`dst = r`), for this epoch
  void* push_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    push_ptrs[i] = epoch.slot_ptr(/*dst=*/i, /*src=*/r);
  }

  // stage 1: finalize this row's vector in registers and push it to every peer
  const auto vid = row_idx * kRowVecs + hvec;
  {
    auto vec = finalize_vec<kHiddenDim, kTopK, kHasShared, kUsePDL>(params, row_idx, hvec);
    Lamport::clear_pos_zero(vec.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_relaxed_16B(vec, push_ptrs[i], vid);
    }
  }

  // ensure epoch is consumed, so flipping it won't lead to error
  if constexpr (!kNorm) barrier_cluster_arrive_relaxed();

  // stage 2: poll own slots, reduce across ranks, [norm], write, reset markers
  void* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = epoch.slot_ptr(/*dst=*/r, /*src=*/i);
  }
  StageVec vec[kWorldSize];
  do {
    bool has_zero = false;
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], vid);
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      // the producer remapped +0.0 words, so a written word is never 0:
      // word == 0 <=> the slot still holds the empty marker
      has_zero |= Lamport::has_pos_zero(vec[i].data());
    }
    if (!has_zero) break;
  } while (true);

  if constexpr (!kNorm) {
    const auto red = reduce_vec(vec);
    ptx::st_global_16B(red, params.out, vid);
    // ensure epoch is consumed, so flipping it won't lead to error
    barrier_cluster_wait();
  } else {
    // push to peer
    __shared__ float smem_sq[kClusterSize][kNumWarps];
    const auto red = reduce_vec(vec);
    StageVec w;
    w.load(params.norm_weight, hvec);
    const auto cluster = cg::this_cluster();
    fp32x2_t acc[4];
    float sq = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      acc[j] = cast<fp32x2_t>(red[j]);
      sq = fmaf(acc[j].x, acc[j].x, sq);
      sq = fmaf(acc[j].y, acc[j].y, sq);
    }
    sq = warp::reduce_sum(sq);
    const auto lane = tx % kWarpThreads;
    const auto warp = tx / kWarpThreads;
    if (lane < kClusterSize) {
      *cluster.map_shared_rank(&smem_sq[cluster_rank][warp], lane) = sq;
    }
    cluster.sync();
    float total = 0.0f;
#pragma unroll
    for (uint32_t c = 0; c < kClusterSize; ++c) {
#pragma unroll
      for (uint32_t wp = 0; wp < kNumWarps; ++wp) {
        total += smem_sq[c][wp];
      }
    }
    const auto factor = math::rsqrt(total / static_cast<float>(kHiddenDim) + params.norm_eps);
    StageVec out;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const auto [wa, wb] = cast<fp32x2_t>(w[j]);
      out[j] = cast<bf16x2_t>(fp32x2_t{acc[j].x * factor * wa, acc[j].y * factor * wb});
    }
    ptx::st_global_16B(out, params.out, vid);
  }
  PDLTriggerSecondary<kUsePDL>();

  // re-establish the empty markers for the next same-phase round
  StageVec zero_vec;
  Lamport::fill_pos_zero(zero_vec.data());
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    ptx::st_global_16B(zero_vec, poll_ptrs[i], vid);
  }

  if (cluster_rank == 0) epoch.flip();
}

// --- host --------------------------------------------------------------------

template <uint32_t kWorldSize, uint32_t kHiddenDim, uint32_t kTopK, uint32_t kClusterSize, bool kUsePDL>
struct MoeFinalizeAllReduceKernel {
 private:
  using TensorView = tvm::ffi::TensorView;
  using Params = FinalizeAllReduceParams<kWorldSize>;
  using Trait = AllReduceNormTrait<kHiddenDim, kClusterSize>;

  template <bool kHasShared, bool kNorm>
  static constexpr auto kernel =
      moe_finalize_all_reduce_kernel<kWorldSize, kHiddenDim, kTopK, kClusterSize, kUsePDL, kHasShared, kNorm>;

 public:
  /// out = [allreduce over ranks of] finalize(gemm2_out, idx, weights) [+ shared] [-> RMSNorm(norm_weight, eps)].
  /// `out` ([T, kHiddenDim] bf16) is output-only. `shared_output` and `norm_weight`
  /// select the epilogue at runtime (four kernel instantiations per module).
  static void
  run(CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      std::optional<TensorView> norm_weight,
      double eps,
      bool prefetch_metadata) {
    using namespace host;
    const auto& comm = *ref.get();
    const auto& push = comm.get_push_obj();
    CHECK_HOST(push.world_size == kWorldSize)
        << "communicator holds " << push.world_size << " ranks, kernel built for " << kWorldSize;

    auto T = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"num_permuted_rows"};
    auto TK = SymbolicSize{"num_expanded"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({T, kHiddenDim})
        .with_strides({kHiddenDim, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(out);
    TensorMatcher({P, kHiddenDim})
        .with_strides({kHiddenDim, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(gemm2_out);
    TensorMatcher({T, kTopK})
        .with_strides({kTopK, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(expert_weights);
    TK.set_value(T.unwrap() * kTopK);
    TensorMatcher({TK}).with_strides({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(permuted_idx);
    if (shared_output.has_value()) {
      TensorMatcher({T, kHiddenDim})
          .with_strides({kHiddenDim, 1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(shared_output.value());
    }
    if (norm_weight.has_value()) {
      TensorMatcher({kHiddenDim})
          .with_strides({1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(norm_weight.value());
    }
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    CHECK_HOST(num_tokens > 0) << "num_tokens must be positive";
    CHECK_HOST(reinterpret_cast<uintptr_t>(gemm2_out.data_ptr()) % 16 == 0) << "gemm2_out must be 16B aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0) << "out must be 16B aligned";

    // the whole [T, hidden] row view is staged by vector index, so it must fit
    // one push slot; the generic push kernel's callers pick the slot size
    const int64_t nbytes = num_tokens * int64_t(kHiddenDim) * sizeof(bf16_t);
    CHECK_HOST(nbytes <= push.slot_bytes) << "num_tokens * hidden * 2 = " << nbytes << " bytes exceeds the "
                                          << push.slot_bytes << "-byte push slot (reduce the batch or enlarge "
                                          << "max_push_size)";
    // one cluster (and phase counter) per row; the bumper cluster is launched
    // only when counters are left over for it to flip (num_tokens < num_blocks),
    // so a batch that owns every counter runs without it
    CHECK_HOST(num_tokens <= push.num_blocks)
        << "num_tokens = " << num_tokens << " exceeds the " << push.num_blocks << " push phase counters of the plane";
    const uint32_t num_clusters = num_tokens + (num_tokens < push.num_blocks ? 1 : 0);

    const auto params = Params{
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .gemm2 = static_cast<const bf16_t*>(gemm2_out.data_ptr()),
        .idx = static_cast<const int32_t*>(permuted_idx.data_ptr()),
        .weights = static_cast<const bf16_t*>(expert_weights.data_ptr()),
        .shared = shared_output.has_value() ? static_cast<const bf16_t*>(shared_output.value().data_ptr()) : nullptr,
        .norm_weight = norm_weight.has_value() ? static_cast<const bf16_t*>(norm_weight.value().data_ptr()) : nullptr,
        .norm_eps = static_cast<float>(eps),
        .prefetch_metadata = prefetch_metadata,
        .rank = push.rank,
        .num_tokens = num_tokens,
        .num_push_counters = push.num_blocks,
        .ws = push.get_workspace<kWorldSize>(nbytes),
    };

    const auto has_shared = shared_output.has_value();
    const auto has_norm = norm_weight.has_value();
    const auto kern = has_shared ? (has_norm ? kernel<true, true> : kernel<true, false>)
                                 : (has_norm ? kernel<false, true> : kernel<false, false>);
    // __cluster_dims__(1, kClusterSize, 1) is compiled in, so a plain launch
    // already forms the clusters along y
    LaunchKernel(dim3(num_clusters, kClusterSize), Trait::kBlockSize, out.device()).enable_pdl(kUsePDL)(kern, params);
  }
};

}  // namespace sglang
