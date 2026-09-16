// Fused deferred-MoE finalize -> 1shot lamport push all-reduce [-> RMSNorm]
// over the CustomAllReduceV2 push plane, for decode-sized batches (bf16). The
// hidden width, top_k and cluster geometry are template parameters; the
// shared-expert add and the RMSNorm epilogue are optional.
//
// `idx == -1` marks a dropped slot (EP: the token was routed to an expert that
// is not local) and contributes nothing. Accumulation is fp32 and the bf16
// rounding points are exactly the unfused path's (moe_runner/flashinfer_trtllm.py
// finalize -> `shared.add_(routed)` -> fp32-accumulating bf16 all-reduce in rank
// order), so the kNorm=false result is bit-identical to it.
//
// The rank-local finalize never materializes in global memory: each thread
// computes one 16B vector of it and pushes it straight into every peer's push
// slot with unicast `st.relaxed.sys` stores, so no multicast mapping is needed.
//
// Push-plane protocol (see include/sgl_kernel/distributed/communicator.cuh):
//   * every rank owns 2 phases x kWorldSize slots of `slot_bytes`; a round
//     uses phase `counter & 1`, producer r writes slot r of every peer, the
//     consumer polls its own kWorldSize slots until no +0.0 marker remains,
//     reduces, and restores the +0.0 markers before it exits;
//   * +0.0 payload words are remapped to -0.0 (numerically identical) so a
//     written word is never 0 and `word == 0` means "not arrived yet";
//   * the generic push kernel owns one phase counter per block; this kernel
//     uses one per row cluster (flipped by the cluster's leader block after a
//     cluster barrier) plus a trailing "bumper" cluster that flips every
//     remaining one, so the whole array keeps one parity and both kernel
//     families can share the plane;
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
#include <type_traits>

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

template <uint32_t kWorldSize, typename WeightT>
struct FinalizeAllReduceParams {
  bf16_t* out;                // [num_tokens, kHiddenDim], output-only
  const bf16_t* gemm2;        // [P, kHiddenDim], permuted / padded rows
  const int32_t* idx;         // [num_tokens * kTopK], -1 = dropped slot
  const WeightT* weights;     // [num_tokens, kTopK], scaling already folded in
  const bf16_t* shared;       // [num_tokens, kHiddenDim] (kHasShared only)
  const bf16_t* norm_weight;  // [kHiddenDim] (kNorm only)
  float norm_eps;             // kNorm only
  // Caller's promise that everything read before the PDL wait is complete when
  // the predecessor merely *triggers*: no all-reduce on this plane right before
  // it, and the routing metadata's producers finished (PDL completion is not
  // transitive through early-triggering kernels). False (the default) waits first.
  bool prefetch_metadata;
  uint32_t rank;
  uint32_t num_tokens;
  uint32_t num_push_counters;  // full counter array size (bumper range end)
  PushWorkSpace<kWorldSize> ws;
  bf16_t* mhc_out = nullptr;
  const bf16_t* residual = nullptr;
  const float* post = nullptr;
  const float* comb = nullptr;
  const float* pre = nullptr;
  bf16_t* normalized = nullptr;
  fp8_e4m3_t* quantized = nullptr;
  uint8_t* scales = nullptr;
};

template <uint32_t kHiddenDim, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE void mhc_quant_vec(
    const FinalizeAllReduceParams<kWorldSize, WeightT>& params, const StageVec& value, uint32_t token, uint32_t hvec) {
  using namespace device;
  fp32x2_t v[4];
  float amax = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    v[j] = cast<fp32x2_t>(value[j]);
    amax = fmaxf(amax, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 4));
  const float normalized = amax * (1.0f / 448.0f);
  const uint32_t bits = __float_as_uint(normalized);
  const uint32_t exponent = (bits >> 23) & 255;
  const uint32_t mantissa = bits & 0x7fffff;
  const bool bump = mantissa != 0 && !(exponent == 0 && mantissa <= 0x400000);
  const uint32_t sf = normalized <= 0 ? 0 : min(exponent + uint32_t(bump), 254u);
  const float inv_scale = __uint_as_float(sf == 0 ? 0 : (254 - sf) << 23);
  AlignedVector<fp8x2_e4m3_t, 4> q;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    q[j] = cast<fp8x2_e4m3_t>(
        fp32x2_t{fminf(fmaxf(v[j].x * inv_scale, -448.0f), 448.0f), fminf(fmaxf(v[j].y * inv_scale, -448.0f), 448.0f)});
  }
  q.store(params.quantized + static_cast<int64_t>(token) * kHiddenDim, hvec);
  if (hvec % 4 == 0) {
    const uint32_t g = hvec / 4;
    const uint32_t off = (g / 4) * 512 + ((token % 32) * 4 + (token / 32) % 4) * 4 + g % 4;
    params.scales[off] = sf;
  }
}

/// HC=4 post mixing of an already BF16-rounded all-reduce vector, in
/// mhc_post_split_h's order: round comb[0]*residual[0], FMA post*x, then the
/// remaining three residual streams.
template <uint32_t kHiddenDim, bool kCollapse = false, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE StageVec mhc_post_vec(
    const FinalizeAllReduceParams<kWorldSize, WeightT>& params, const StageVec& red, uint32_t token, uint32_t hvec) {
  using namespace device;
  StageVec residual[4];
  fp32x2_t collapsed[4] = {};
#pragma unroll
  for (uint32_t c = 0; c < 4; ++c) {
    residual[c].load(params.residual + (static_cast<int64_t>(token) * 4 + c) * kHiddenDim, hvec);
  }
#pragma unroll
  for (uint32_t c = 0; c < 4; ++c) {
    const float post = params.post[token * 4 + c];
    float comb[4];
#pragma unroll
    for (uint32_t r = 0; r < 4; ++r)
      comb[r] = params.comb[token * 16 + r * 4 + c];
    StageVec out;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const auto x = cast<fp32x2_t>(red[j]);
      const auto r0 = cast<fp32x2_t>(residual[0][j]);
      fp32x2_t acc{fmaf(post, x.x, __fmul_rn(comb[0], r0.x)), fmaf(post, x.y, __fmul_rn(comb[0], r0.y))};
#pragma unroll
      for (uint32_t r = 1; r < 4; ++r) {
        const auto v = cast<fp32x2_t>(residual[r][j]);
        acc.x = fmaf(comb[r], v.x, acc.x);
        acc.y = fmaf(comb[r], v.y, acc.y);
      }
      out[j] = cast<bf16x2_t>(acc);
      if constexpr (kCollapse) {
        const auto rounded = cast<fp32x2_t>(out[j]);
        const float pre = params.pre[token * 4 + c];
        collapsed[j].x = fmaf(rounded.x, pre, collapsed[j].x);
        collapsed[j].y = fmaf(rounded.y, pre, collapsed[j].y);
      }
    }
    out.store(params.mhc_out + (static_cast<int64_t>(token) * 4 + c) * kHiddenDim, hvec);
  }
  StageVec result;
  if constexpr (kCollapse) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j)
      result[j] = cast<bf16x2_t>(collapsed[j]);
  }
  return result;
}

/// Row geometry: one 16B vector per thread, one cluster per row, so the block
/// size follows from the hidden width and the cluster size (the tuning knob).
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
// routing rows and the kTopK gathers are fetched.
template <uint32_t kHiddenDim, uint32_t kTopK, bool kHasShared, bool kUsePDL, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE StageVec
finalize_vec(const FinalizeAllReduceParams<kWorldSize, WeightT>& params, uint32_t token, uint32_t hvec) {
  using namespace device;
  const auto* idx = params.idx + static_cast<int64_t>(token) * kTopK;
  const auto* weights = params.weights + static_cast<int64_t>(token) * kTopK;
  int32_t rows[kTopK];
  WeightT w[kTopK];
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
    if constexpr (std::is_same_v<WeightT, bf16_t>) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        acc[j].x = math::fma_f32_bf16(in[k][j].x, w[k], acc[j].x);
        acc[j].y = math::fma_f32_bf16(in[k][j].y, w[k], acc[j].y);
      }
    } else
#endif
    {
      const auto w_fp32 = cast<fp32_t>(w[k]);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const auto [x, y] = cast<fp32x2_t>(in[k][j]);
        acc[j].x = fmaf(x, w_fp32, acc[j].x);
        acc[j].y = fmaf(y, w_fp32, acc[j].y);
      }
    }
  }
  // Two deliberate roundings -- the routed combine, then the bf16 + bf16 add --
  // keep the staged vector bit-identical to the unfused rank-local result.
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
// Grid dim3(num_tokens [+ 1], kClusterSize) with the cluster along y, so
// blockIdx.x is the token row (and its phase counter) and blockIdx.y the rank
// inside the cluster. The extra cluster (blockIdx.x == num_tokens) is the
// bumper: it only flips the leftover counters [num_tokens, num_push_counters).
template <
    uint32_t kWorldSize,
    uint32_t kHiddenDim,
    uint32_t kTopK,
    uint32_t kClusterSize,
    bool kUsePDL,
    bool kHasShared,
    bool kNorm,
    typename WeightT,
    bool kMhc = false,
    bool kQuant = false>
__global__ __launch_bounds__(AllReduceNormTrait<kHiddenDim, kClusterSize>::kBlockSize)
    __cluster_dims__(1, kClusterSize, 1) void moe_finalize_all_reduce_kernel(
        const __grid_constant__ FinalizeAllReduceParams<kWorldSize, WeightT> params) {
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

  // Reading the epoch before the PDL wait can see a predecessor all-reduce
  // mid-flip on this plane; prefetch_metadata defers the wait to finalize_vec.
  if (!params.prefetch_metadata) PDLWaitPrimary<kUsePDL>();

  if (row_idx == params.num_tokens) {
    PDLWaitPrimary<kUsePDL>();
    if constexpr (kQuant) {
      // The SF buffer is padded to 128 rows. Active rows are written by the
      // norm epilogue; the existing bumper zeros only the disjoint padding.
      for (uint32_t off = hvec; off < (kHiddenDim / 32) * 128; off += kRowVecs) {
        const uint32_t swizzled_row = (off % 512) / 4;
        const uint32_t row = swizzled_row / 4 + (swizzled_row % 4) * 32;
        if (row >= params.num_tokens) params.scales[off] = 0;
      }
    }
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
    if constexpr (kMhc) mhc_post_vec<kHiddenDim>(params, red, row_idx, hvec);
    // ensure epoch is consumed, so flipping it won't lead to error
    barrier_cluster_wait();
  } else {
    // push to peer
    __shared__ float smem_sq[kClusterSize][kNumWarps];
    auto red = reduce_vec(vec);
    if constexpr (kMhc) {
      ptx::st_global_16B(red, params.out, vid);
      red = mhc_post_vec<kHiddenDim, true>(params, red, row_idx, hvec);
    }
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
    ptx::st_global_16B(out, kMhc ? params.normalized : params.out, vid);
    if constexpr (kQuant) mhc_quant_vec<kHiddenDim>(params, out, row_idx, hvec);
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

template <
    uint32_t kWorldSize,
    uint32_t kHiddenDim,
    uint32_t kTopK,
    uint32_t kClusterSize,
    bool kUsePDL,
    typename WeightT,
    bool kMhc = false,
    bool kQuant = false>
struct MoeFinalizeAllReduceKernel {
 private:
  static_assert(std::is_same_v<WeightT, bf16_t> || std::is_same_v<WeightT, fp32_t>);
  using TensorView = tvm::ffi::TensorView;
  using Params = FinalizeAllReduceParams<kWorldSize, WeightT>;
  using Trait = AllReduceNormTrait<kHiddenDim, kClusterSize>;

  template <bool kHasShared, bool kNorm>
  static constexpr auto kernel = moe_finalize_all_reduce_kernel<
      kWorldSize,
      kHiddenDim,
      kTopK,
      kClusterSize,
      kUsePDL,
      kHasShared,
      kNorm,
      WeightT,
      kMhc,
      kQuant>;

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
    static_assert(!kMhc);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        eps,
        prefetch_metadata,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt);
  }

  /// Finalize + all-reduce + HC=4 post; original reduced output is retained.
  static void run_mhc(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb) {
    static_assert(kMhc);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        std::nullopt,
        0.0,
        false,
        mhc_out,
        residual,
        post,
        comb);
  }

  static void run_mhc_norm(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb,
      TensorView pre,
      TensorView norm_weight,
      double eps,
      TensorView normalized) {
    static_assert(kMhc);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        eps,
        false,
        mhc_out,
        residual,
        post,
        comb,
        pre,
        normalized);
  }

  static void run_mhc_quant(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb,
      TensorView pre,
      TensorView norm_weight,
      double eps,
      TensorView normalized,
      TensorView quantized,
      TensorView scales) {
    static_assert(kMhc && kQuant);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        eps,
        false,
        mhc_out,
        residual,
        post,
        comb,
        pre,
        normalized,
        quantized,
        scales);
  }

 private:
  static void run_impl(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      std::optional<TensorView> norm_weight,
      double eps,
      bool prefetch_metadata,
      std::optional<TensorView> mhc_out,
      std::optional<TensorView> residual,
      std::optional<TensorView> post,
      std::optional<TensorView> comb,
      std::optional<TensorView> pre = std::nullopt,
      std::optional<TensorView> normalized = std::nullopt,
      std::optional<TensorView> quantized = std::nullopt,
      std::optional<TensorView> scales = std::nullopt) {
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
        .with_dtype<WeightT>()
        .template with_device<kDLCUDA>(device)
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
    if constexpr (kQuant) {
      CHECK_HOST(num_tokens <= 8);
      CHECK_HOST(norm_weight.has_value());
      TensorMatcher({T, kHiddenDim}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(device).verify(quantized.value());
      TensorMatcher({(kHiddenDim / 32) * 128})
          .with_dtype<uint8_t>()
          .with_device<kDLCUDA>(device)
          .verify(scales.value());
    }
    if constexpr (kMhc) {
      static_assert(kHiddenDim == 5120);
      if (norm_weight.has_value()) {
        TensorMatcher({T, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(pre.value());
        TensorMatcher({T, kHiddenDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(normalized.value());
      }
      TensorMatcher({T, 4, kHiddenDim})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(mhc_out.value())
          .verify(residual.value());
      TensorMatcher({T, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(post.value());
      TensorMatcher({T, 4, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(comb.value());
    }
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
    // only when counters are left over for it to flip
    CHECK_HOST(num_tokens <= push.num_blocks)
        << "num_tokens = " << num_tokens << " exceeds the " << push.num_blocks << " push phase counters of the plane";
    const uint32_t num_clusters = num_tokens + (num_tokens < push.num_blocks ? 1 : 0);

    const auto params = Params{
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .gemm2 = static_cast<const bf16_t*>(gemm2_out.data_ptr()),
        .idx = static_cast<const int32_t*>(permuted_idx.data_ptr()),
        .weights = static_cast<const WeightT*>(expert_weights.data_ptr()),
        .shared = shared_output.has_value() ? static_cast<const bf16_t*>(shared_output.value().data_ptr()) : nullptr,
        .norm_weight = norm_weight.has_value() ? static_cast<const bf16_t*>(norm_weight.value().data_ptr()) : nullptr,
        .norm_eps = static_cast<float>(eps),
        .prefetch_metadata = prefetch_metadata,
        .rank = push.rank,
        .num_tokens = num_tokens,
        .num_push_counters = push.num_blocks,
        .ws = push.get_workspace<kWorldSize>(nbytes),
        .mhc_out = mhc_out.has_value() ? static_cast<bf16_t*>(mhc_out.value().data_ptr()) : nullptr,
        .residual = residual.has_value() ? static_cast<const bf16_t*>(residual.value().data_ptr()) : nullptr,
        .post = post.has_value() ? static_cast<const float*>(post.value().data_ptr()) : nullptr,
        .comb = comb.has_value() ? static_cast<const float*>(comb.value().data_ptr()) : nullptr,
        .pre = pre.has_value() ? static_cast<const float*>(pre.value().data_ptr()) : nullptr,
        .normalized = normalized.has_value() ? static_cast<bf16_t*>(normalized.value().data_ptr()) : nullptr,
        .quantized = quantized.has_value() ? static_cast<fp8_e4m3_t*>(quantized.value().data_ptr()) : nullptr,
        .scales = scales.has_value() ? static_cast<uint8_t*>(scales.value().data_ptr()) : nullptr,
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
