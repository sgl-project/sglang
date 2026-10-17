// Low-latency DeepSeek-V4.1 decode consumer of wo_b_fp8.cuh's push: TP all-reduce of the attention output fused with
// the HC = 4 mHC post-mix, the pre-collapse and the RMSNorm (the attention -> MoE boundary, BF16 out).
//
//   reduced     = bf16(sum over ranks of wo_b's BF16 rows, fp32 in rank order)
//   mhc_out[c]  = bf16(post[c] * reduced + sum_s comb[s][c] * residual[s])   (the next residual streams)
//   normalized  = RMSNorm(bf16(sum_c pre[c] * mhc_out[c])) * norm_weight     (the MoE input)
//
// Same arithmetic as all_reduce_fusion.cuh's mHC + combine-norm epilogue (all_reduce_mhc_post_combine_norm), except
// the mean square is a multiply by 1/5120 instead of an IEEE divide (normalized may differ by a bf16 ulp), and the
// same push-plane protocol (sgl_kernel/distributed/communicator.cuh): the producer pushed every row into slot `rank`
// of every peer's workspace in phase `counter & 1`, +0 words remapped to -0; this kernel polls its own kWorldSize
// slots until no +0 word is left, then restores the +0 markers and flips its row's counter. One 5-CTA cluster per
// token (a thread per 16 B vector), plus a trailing bumper cluster flipping the counters of the rows not launched, so
// the whole counter array keeps one parity and other kernels on the plane stay in step.
//
// The counters are read before griddepcontrol.wait: no predecessor back to the last kernel that waits before it
// triggers (the attention) touches them.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::device::all_reduce_mhc_ll {

constexpr uint32_t kHidden = 5120;
constexpr uint32_t kHC = 4;
constexpr uint32_t kClusterSize = 5;
constexpr uint32_t kRowVecs = kHidden / 8;              // 16 B vectors per row
constexpr uint32_t kThreads = kRowVecs / kClusterSize;  // 128: a vector per thread
constexpr uint32_t kWarps = kThreads / kWarpThreads;
static_assert(kRowVecs % kClusterSize == 0 && kThreads % kWarpThreads == 0);

using Vec = AlignedVector<bf16x2_t, 4>;

namespace ptx {
using device::ptx::ld_relaxed_16B;
using device::ptx::st_relaxed_16B;
using device::ptx::to_shared;
// A phase counter, read at system scope: never a stale L1 line from an earlier kernel on this SM.
SGL_DEVICE uint32_t load_counter(const distributed::Counter* counter) {
  return device::ptx::load_relaxed_sys(reinterpret_cast<const uint32_t*>(counter));
}
SGL_DEVICE void st_shared_cluster_f32(void* smem, uint32_t cta, float v) {
  asm volatile("{\n\t.reg .b32 a;\n\tmapa.shared::cluster.u32 a, %0, %1;\n\tst.shared::cluster.f32 [a], %2;\n\t}" ::"r"(
                   to_shared(smem)),
               "r"(cta),
               "f"(v)
               : "memory");
}
SGL_DEVICE void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
}  // namespace ptx
using Lamport = distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;

template <uint32_t kWorldSize>
struct Params {
  bf16_t* mhc_out;            // [T, 4, 5120]
  bf16_t* normalized;         // [T, 5120]
  const bf16_t* residual;     // [T, 4, 5120]
  const float* post;          // [T, 4]
  const float* comb;          // [T, 4 (source), 4 (target)]
  const float* pre;           // [T, 4]
  const bf16_t* norm_weight;  // [5120]
  float eps;
  uint32_t rank;
  uint32_t num_tokens;
  uint32_t num_counters;  // the plane's counter array: the bumper flips [num_tokens, num_counters)
  distributed::PushWorkSpace<kWorldSize> ws;
};

template <uint32_t kWorldSize, bool kUsePDL>
__global__ __launch_bounds__(kThreads)
    __cluster_dims__(1, kClusterSize, 1) void all_reduce_mhc_ll_kernel(const __grid_constant__ Params<kWorldSize> p) {
  const auto tx = threadIdx.x;
  const auto token = blockIdx.x;
  const auto cluster_rank = blockIdx.y;
  const auto hvec = cluster_rank * kThreads + tx;

  if (token == p.num_tokens) {
    if (cluster_rank == 0) {
      const auto epoch = ptx::load_counter(p.ws.counter + token);
      __syncthreads();  // every thread read it before any flips it
      for (uint32_t i = token + tx; i < p.num_counters; i += kThreads)
        p.ws.counter[i].set(epoch ^ 1);
    }
    return PDLTriggerSecondary<kUsePDL>();
  }

  // ---- before the wait: this row's phase, and every operand that is not the producer's ----
  const auto epoch = ptx::load_counter(p.ws.counter + token);
  const auto phase_base = p.ws.workspaces[p.rank] + (epoch & 1) * p.ws.slot_bytes * kWorldSize;
  void* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = phase_base + i * p.ws.slot_bytes;  // source rank i's slot
  }
  const auto vid = token * kRowVecs + hvec;
  Vec residual[kHC];
#pragma unroll
  for (uint32_t s = 0; s < kHC; ++s) {
    residual[s].load(p.residual + (size_t(token) * kHC + s) * kHidden, hvec);
  }
  Vec weight;
  weight.load(p.norm_weight, hvec);
  float post[kHC], pre[kHC], comb[kHC][kHC];
#pragma unroll
  for (uint32_t c = 0; c < kHC; ++c) {
    post[c] = p.post[token * kHC + c];
    pre[c] = p.pre[token * kHC + c];
#pragma unroll
    for (uint32_t s = 0; s < kHC; ++s)
      comb[s][c] = p.comb[(token * kHC + s) * kHC + c];
  }

  PDLWaitPrimary<kUsePDL>();

  // ---- poll: every rank's slot at once, until no +0 word (the empty marker) is left ----
  Vec parts[kWorldSize];
  while (true) {
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::ld_relaxed_16B(parts[i], poll_ptrs[i], vid);
    }
    bool empty = false;
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      empty |= Lamport::has_pos_zero(parts[i].data());
    }
    if (!empty) break;
  }
  const auto reduced = reduce_vec(parts);

  // ---- mHC post (all_reduce_fusion.cuh's mhc_post_vec order) and the pre-collapse ----
  fp32x2_t collapsed[4] = {};
#pragma unroll
  for (uint32_t c = 0; c < kHC; ++c) {
    Vec mixed;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const auto x = cast<fp32x2_t>(reduced[j]);
      const auto r0 = cast<fp32x2_t>(residual[0][j]);
      fp32x2_t acc{fmaf(post[c], x.x, comb[0][c] * r0.x), fmaf(post[c], x.y, comb[0][c] * r0.y)};
#pragma unroll
      for (uint32_t s = 1; s < kHC; ++s) {
        const auto v = cast<fp32x2_t>(residual[s][j]);
        acc.x = fmaf(comb[s][c], v.x, acc.x);
        acc.y = fmaf(comb[s][c], v.y, acc.y);
      }
      mixed[j] = cast<bf16x2_t>(acc);
      const auto rounded = cast<fp32x2_t>(mixed[j]);
      collapsed[j].x = fmaf(rounded.x, pre[c], collapsed[j].x);
      collapsed[j].y = fmaf(rounded.y, pre[c], collapsed[j].y);
    }
    mixed.store(p.mhc_out + (size_t(token) * kHC + c) * kHidden, hvec);
  }

  // ---- RMSNorm over the row: warp sums, then every CTA's warp sums over DSMEM ----
  __shared__ float s_sq[kClusterSize][kWarps];
  fp32x2_t x[4];
  float sq = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    x[j] = cast<fp32x2_t>(cast<bf16x2_t>(collapsed[j]));
    sq = fmaf(x[j].x, x[j].x, sq);
    sq = fmaf(x[j].y, x[j].y, sq);
  }
  sq = warp::reduce_sum(sq);
  const auto lane = tx % kWarpThreads, warp_id = tx / kWarpThreads;
  if (lane < kClusterSize) ptx::st_shared_cluster_f32(&s_sq[cluster_rank][warp_id], lane, sq);
  // Also every CTA of the cluster has read counter[token] by now, so the flip below is safe.
  ptx::cluster_sync();
  float total = 0.0f;
#pragma unroll
  for (uint32_t c = 0; c < kClusterSize; ++c) {
#pragma unroll
    for (uint32_t w = 0; w < kWarps; ++w) {
      total += s_sq[c][w];
    }
  }
  const auto factor = math::rsqrt(total * (1.0f / kHidden) + p.eps);
  Vec out;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    const auto [wa, wb] = cast<fp32x2_t>(weight[j]);
    out[j] = cast<bf16x2_t>(fp32x2_t{x[j].x * factor * wa, x[j].y * factor * wb});
  }
  out.store(p.normalized + size_t(token) * kHidden, hvec);
  PDLTriggerSecondary<kUsePDL>();

  // ---- re-arm this round's slots for the next same-phase round, then flip the row's phase ----
  Vec zero;
  Lamport::fill_pos_zero(zero.data());
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    ptx::st_relaxed_16B(zero, poll_ptrs[i], vid);
  }
  if (cluster_rank == 0 && tx == 0) p.ws.counter[token].set(epoch ^ 1);
}

}  // namespace sglang::device::all_reduce_mhc_ll

namespace sglang {

/**
 * \brief All-reduce of wo_b_fp8.cuh's pushed rows + mHC post + pre-collapse + RMSNorm, BF16 out.
 *
 * Launched as the PDL secondary of the pushing wo_b, on the same communicator and token count.
 * \tparam kWorldSize  2 / 4, the communicator's world size.
 * \param comm         the CustomAllReduceV2 communicator wo_b pushed into.
 * \param residual     [T, 4, 5120] bf16, the hc residual streams.
 * \param post         [T, 4] fp32.
 * \param comb         [T, 4, 4] fp32, (source, target).
 * \param pre          [T, 4] fp32.
 * \param norm_weight  [5120] bf16.
 * \param mhc_out      [T, 4, 5120] bf16, the post-mixed residual streams.
 * \param normalized   [T, 5120] bf16, the normalized pre-collapse.
 */
template <uint32_t kWorldSize, bool kUsePDL>
inline void dsv41_all_reduce_mhc_ll(
    host::distributed::CommunicatorRef comm,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView post,
    tvm::ffi::TensorView comb,
    tvm::ffi::TensorView pre,
    tvm::ffi::TensorView norm_weight,
    tvm::ffi::TensorView mhc_out,
    tvm::ffi::TensorView normalized,
    double eps) {
  using namespace host;
  namespace K = device::all_reduce_mhc_ll;
  static_assert(kWorldSize == 2 || kWorldSize == 4);
  const auto& push = comm->get_push_obj();
  CHECK_HOST(push.world_size == kWorldSize) << push.world_size;

  auto T = SymbolicSize{"num_tokens"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({T, K::kHC, K::kHidden}).with_dtype<bf16_t>().with_device(device_).verify(residual).verify(mhc_out);
  TensorMatcher({T, K::kHC}).with_dtype<fp32_t>().with_device(device_).verify(post).verify(pre);
  TensorMatcher({T, K::kHC, K::kHC}).with_dtype<fp32_t>().with_device(device_).verify(comb);
  TensorMatcher({K::kHidden}).with_dtype<bf16_t>().with_device(device_).verify(norm_weight);
  TensorMatcher({T, K::kHidden}).with_dtype<bf16_t>().with_device(device_).verify(normalized);
  const auto num_tokens = static_cast<uint32_t>(T.unwrap());
  CHECK_HOST(num_tokens > 0 && num_tokens <= push.num_blocks) << num_tokens << " " << push.num_blocks;
  const int64_t num_bytes = int64_t(num_tokens) * K::kHidden * sizeof(bf16_t);
  CHECK_HOST(num_bytes <= push.slot_bytes) << num_bytes << " " << push.slot_bytes;

  const auto params = K::Params<kWorldSize>{
      .mhc_out = static_cast<bf16_t*>(mhc_out.data_ptr()),
      .normalized = static_cast<bf16_t*>(normalized.data_ptr()),
      .residual = static_cast<const bf16_t*>(residual.data_ptr()),
      .post = static_cast<const float*>(post.data_ptr()),
      .comb = static_cast<const float*>(comb.data_ptr()),
      .pre = static_cast<const float*>(pre.data_ptr()),
      .norm_weight = static_cast<const bf16_t*>(norm_weight.data_ptr()),
      .eps = static_cast<float>(eps),
      .rank = push.rank,
      .num_tokens = num_tokens,
      .num_counters = push.num_blocks,
      .ws = push.get_workspace<kWorldSize>(num_bytes),
  };
  // The bumper cluster only when counters are left over for it to flip.
  const uint32_t num_clusters = num_tokens + (num_tokens < push.num_blocks ? 1 : 0);
  constexpr auto kernel = K::all_reduce_mhc_ll_kernel<kWorldSize, kUsePDL>;
  LaunchKernel(dim3(num_clusters, K::kClusterSize), K::kThreads, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
}

}  // namespace sglang
