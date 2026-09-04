// K3 MNNVL fused all-reduce (bf16-only), two zero-copy algorithm families:
//
//   - push (1shot): lamport-style push, but the data staging is a SINGLE
//     multicast store into slot `rank` of every peer's push workspace
//     (replacing the 7 unicast stores of the generic kernel), followed by a
//     local zero-marker polling reduce. Input is read in place and the
//     result is written back in place — no staging copies. Works for any
//     input tensor; reuses the CustomAllReduceV2 push workspace + counter.
//     Best for small messages.
//
//   - pull (2shot): low-SM NVLS pull directly ON the input tensor, which
//     therefore MUST live in (multicast-bound) symmetric memory: each rank
//     `multimem.ld_reduce`s its shard from the input's multicast address and
//     `multimem.st`s the result back — in-place, zero copies. Two
//     NCCL-style ideas keep a handful of blocks (tuned 1~16, 4 at the large
//     end) at the fabric limit:
//
//       * deep pipelining: the copy loop manually keeps `unroll` multimem
//         loads in flight per thread before the first store, cascading
//         through halving widths for mid-size tails.
//       * multicast barriers on the CustomAllReduceV2 pull semaphores: the
//         generic pull kernels' reservation protocol with each arrival sent
//         as ONE `multimem.red.add` on the semaphore flag's multicast alias
//         instead of per-peer unicast reds — identical memory effects, so
//         both kernel families share the slots freely (single-stream calls
//         are serialized).
//
// Both families fuse either a residual add (`out = allreduce(x) + r`; the
// residual must be identical on every rank — a fully reduced tensor such as
// the attn-res prefix sum — or absent) or the RMSNorm epilogue over the
// latent of the K3 latent|shared MoE buffer (*_norm variants).
//
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cooperative_groups.h>

// TODO: remove dependency on the custom_all_reduce, move out common utilities
#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/extra/stl.h>

namespace sglang {

// The shared vocabulary this file is written in, pulled in by name so the call
// sites below stay unqualified: see include/sgl_kernel/distributed/ptx.cuh and
// .../communicator.cuh.
using device::distributed::Counter;
using device::distributed::Semaphore;
using host::distributed::CommunicatorRef;

/// The 16 B staging vector, viewed as the 4 u32 words the lamport marker
/// protocol tests. See LamportTrait in distributed/communicator.cuh.
using Lamport = device::distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;

struct FusionParams {
  uint8_t* input;              // tensor pointer (in place)
  const uint8_t* residual;     // may be null (compile-time kHasResidual selects)
  uint8_t* push_ws_mc;         // multicast VA of the push workspace base
  uint8_t* push_ws_local;      // local push workspace base (poll side)
  Counter* push_counter;       // per-block phase counters (local memory)
  int64_t push_buffer_stride;  // per-buffer bytes (2 * world_size buffers)
  uint32_t rank;
  uint32_t num_vecs;  // 16B vectors
  // *_norm variants only: RMSNorm epilogue over the first num_norm_rows
  // rows of the [num_vecs / kNormRowVecs, kNormDim] row view
  const uint8_t* norm_weight;
  float norm_eps;
  uint32_t num_norm_rows;
  uint32_t num_push_counters;  // cluster variant only: full counter array size
  // finalize_push_norm only: trtllm-gen deferred-finalize inputs (kimi_k3.py
  // deferred finalize path); `input` is then output-only ([T, kNormDim])
  const uint8_t* fin_gemm2;    // [P, kNormDim] bf16, permuted rows
  const uint8_t* fin_idx;      // [T * kFinTopK] int32, -1 = dropped slot
  const uint8_t* fin_weights;  // [T, kFinTopK] bf16
};

// The *_norm variants view the input as rows of the K3 latent width (3584
// bf16 = 448 16B vectors per row) and give the first num_norm_rows an
// RMSNorm epilogue (K3: the [N latent | 2N shared] MoE buffer with N normed
// rows, or a latent-only [N, 3584] tensor normed in full).
constexpr uint32_t kNormDim = 3584;
constexpr uint32_t kNormRowVecs = kNormDim / 8;                       // 448
constexpr uint32_t kNormWarps = kNormRowVecs / device::kWarpThreads;  // 14

template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL>
__global__ __launch_bounds__(1024, 1) void all_reduce_push_res_kernel(const __grid_constant__ FusionParams params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;

  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  const auto global_tid = bx * blockDim.x + tx;
  const auto num_threads = blockDim.x * gridDim.x;
  const auto num_vecs = params.num_vecs;

  // prologue: the previous phase flip (counter inc) must be visible
  device::PDLWaitPrimary<kUsePDL>();
  const auto phase = params.push_counter[bx].get() & 1;
  const auto r = params.rank;
  const auto stride_bytes = params.push_buffer_stride;
  const auto phase_stride_bytes = (phase * kWorldSize) * stride_bytes;
  // one multicast store lands this rank's data in slot r of EVERY peer
  const auto push_ptr = params.push_ws_mc + r * stride_bytes + phase_stride_bytes;
  const auto poll_ptr = params.push_ws_local + phase_stride_bytes;

  // stage 1: multicast-push local data, remapping all-zero bf16x2 pairs
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    device::ptx::ld_global_16B(vec, params.input, vid);
    Lamport::clear_pos_zero(vec.data());
    device::ptx::st_multimem_16B(vec, push_ptr, vid);
  }

  // launch pdl early for low latency case
  device::PDLTriggerSecondary<kUsePDL>();

  // stage 2: poll all slots, reduce (+ residual), write back in place,
  // re-establish the empty markers for the next same-phase round
  vec_t zero_vec;
  Lamport::fill_pos_zero(zero_vec.data());
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec[kWorldSize + kHasResidual];
    if constexpr (kHasResidual) vec[kWorldSize].load(params.residual, vid);
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        device::ptx::ld_relaxed_16B(vec[i], poll_ptr + i * stride_bytes, vid);
        // the producer remapped all-zero pairs, so a written atom is never 0:
        // atom == 0 <=> the slot still holds the empty marker
        has_zero |= Lamport::has_pos_zero(vec[i].data());
      }
      if (!has_zero) break;
    } while (true);
    const auto out_vec = device::reduce_vec(vec);  // fp32 accumulation over 8(+1) inputs
    device::ptx::st_global_16B(out_vec, params.input, vid);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      device::ptx::st_global_16B(zero_vec, poll_ptr + i * stride_bytes, vid);
    }
  }

  // epilogue: flip this block's phase
  __syncthreads();
  if (tx == 0) params.push_counter[bx].set(phase ^ 1);
}

// --- deferred-finalize staging (finalize_push_norm) ------------------------
// The trtllm-gen MoE with do_finalize=False hands back its finalize inputs
// (see FlashInfer's TRT-LLM-gen MoE); the fused kernel computes the finalize
// during the push staging pass, so the rank-local latent never materializes.

constexpr uint32_t kFinTopK = 16;

// One 16B vector of the deferred MoE finalize (latent width fixed to kNormDim):
//   local[t] = sum_k fin_weights[t, k] * fin_gemm2[fin_idx[t*16 + k]]
// All 16 gathers issue before the FMA chain; threads of the same token
// broadcast-load the same routing rows.
SGL_DEVICE device::AlignedVector<bf16x2_t, 4> finalize_vec(const FusionParams& params, uint32_t vid) {
  using namespace device;
  constexpr uint32_t kIdxVecSize = kMaxVecBytes / sizeof(int32_t);
  constexpr uint32_t kWVecSize = kMaxVecBytes / sizeof(bf16_t);
  constexpr uint32_t kIdxVecs = kFinTopK / kIdxVecSize;  // 2 on SM100+
  constexpr uint32_t kWVecs = kFinTopK / kWVecSize;      // 1 on SM100+

  const uint32_t token = vid / kNormRowVecs;
  const uint32_t hvec = vid % kNormRowVecs;

  AlignedVector<int32_t, kIdxVecSize> idx[kIdxVecs];
#pragma unroll
  for (uint32_t j = 0; j < kIdxVecs; ++j) {
    idx[j].load(params.fin_idx, token * kIdxVecs + j);
  }
  AlignedVector<bf16_t, kWVecSize> weight[kWVecs];
#pragma unroll
  for (uint32_t j = 0; j < kWVecs; ++j) {
    weight[j].load(params.fin_weights, token * kWVecs + j);
  }

  const auto* g2 = reinterpret_cast<const bf16_t*>(params.fin_gemm2);
  AlignedVector<bf16_t, 8> in[kFinTopK];
#pragma unroll
  for (uint32_t k = 0; k < kFinTopK; ++k) {
    const int32_t row = idx[k / kIdxVecSize][k % kIdxVecSize];
    if (row >= 0) {
      in[k].load(g2 + static_cast<int64_t>(row) * kNormDim, hvec);
    }
  }
  float acc[8] = {};
#pragma unroll
  for (uint32_t k = 0; k < kFinTopK; ++k) {
    const int32_t row = idx[k / kIdxVecSize][k % kIdxVecSize];
    if (row < 0) continue;
    const bf16_t w_k = weight[k / kWVecSize][k % kWVecSize];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      acc[i] = device::math::fma_f32_bf16(in[k][i], w_k, acc[i]);
    }
  }
  AlignedVector<bf16x2_t, 4> out;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    out[j] = cast<bf16x2_t>(fp32x2_t{acc[2 * j], acc[2 * j + 1]});
  }
  return out;
}

template <typename T2, size_t N, size_t M>
SGL_DEVICE float reduce_sqr(device::AlignedVector<T2, N>& out_vec, device::AlignedVector<T2, N> (&vec)[M]) {
  fp32x2_t acc_vec[N];
#pragma unroll
  for (size_t i = 0; i < M; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      const auto [x, y] = device::cast<fp32x2_t>(vec[i][j]);
      auto& [acc_x, acc_y] = acc_vec[j];
      acc_x = i == 0 ? x : acc_x + x;
      acc_y = i == 0 ? y : acc_y + y;
    }
  }
  float sum_eq = 0.0f;
#pragma unroll
  for (size_t j = 0; j < N; ++j) {
    sum_eq += acc_vec[j].x * acc_vec[j].x;
    sum_eq += acc_vec[j].y * acc_vec[j].y;
    out_vec[j] = device::cast<T2>(acc_vec[j]);
  }
  return sum_eq;
}

// kFinalize: stage 1 computes the deferred MoE finalize per vector instead of
// reading a staged input tensor; `input` is then output-only. The host sets
// num_norm_rows to the full row count (every reduced row is normed).
template <uint32_t kWorldSize, uint32_t kClusterSize, bool kUsePDL, bool kFinalize = false>
__global__ __launch_bounds__(kNormRowVecs / kClusterSize) __cluster_dims__(kClusterSize, 1, 1)  //
    void all_reduce_push_norm_cluster_kernel(const __grid_constant__ FusionParams params) {
  namespace cg = cooperative_groups;
  using namespace device;
  using vec_t = AlignedVector<bf16x2_t, 4>;
  constexpr uint32_t kBlockSize = kNormRowVecs / kClusterSize;
  constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;

  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kNormRowVecs % kClusterSize == 0);
  static_assert(kNumWarps >= 1);

  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  const auto global_tid = bx * kBlockSize + tx;
  const auto num_vecs = params.num_vecs;
  const auto num_rows = num_vecs / kNormRowVecs;

  const auto row_idx = bx / kClusterSize;
  const auto num_row_clusters = gridDim.x / kClusterSize - 1;  // last one is the bumper
  // stage-1 grid-stride is over the ROW clusters only: the bumper early-returns
  // and never stages, so it must not be counted or its share of vids is dropped
  const auto num_threads = kBlockSize * num_row_clusters * kClusterSize;

  PDLWaitPrimary<kUsePDL>();

  // special case: the bumper cluster flips every remaining counter so the
  // whole array stays globally in phase. Only ONE block does it (threads
  // grid-stride the counters) — each counter must be inc'd exactly once,
  // independent of whether kClusterSize is odd or even.
  if (row_idx == num_row_clusters) {
    if (bx % kClusterSize == 0) {
      for (uint32_t r = num_row_clusters + tx; r < params.num_push_counters; r += kBlockSize) {
        params.push_counter[r].inc(1);
      }
    }
    return PDLTriggerSecondary<kUsePDL>();
  }

  const auto phase = params.push_counter[row_idx].get() & 1;
  const auto r = params.rank;
  const auto stride_bytes = params.push_buffer_stride;
  const auto phase_stride_bytes = (phase * kWorldSize) * stride_bytes;
  const auto push_ptr = params.push_ws_mc + r * stride_bytes + phase_stride_bytes;
  const auto poll_ptr = params.push_ws_local + phase_stride_bytes;

  // stage 1: multicast staging (grid-stride); kFinalize computes each vector
  // in place of the load
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    if constexpr (kFinalize) {
      vec = finalize_vec(params, vid);
    } else {
      ptx::ld_global_16B(vec, params.input, vid);
    }
    Lamport::clear_pos_zero(vec.data());
    ptx::st_multimem_16B(vec, push_ptr, vid);
  }

  // stage 2: one row per cluster pass (the bumper cluster owns no rows)
  const auto cluster = cg::this_cluster();
  const auto cluster_rank = bx % kClusterSize;
  vec_t w;
  w.load(params.norm_weight, cluster_rank * kBlockSize + tx);
  vec_t zero_vec;
  Lamport::fill_pos_zero(zero_vec.data());
  __shared__ alignas(8) float smem_raw[2][kClusterSize][kNumWarps];
  uint32_t parity = 0;

  // NOTE: launch PDL earlier for low latency case
  PDLTriggerSecondary<kUsePDL>();

  for (auto row = row_idx; row < num_rows; row += num_row_clusters) {
    const auto vid = row * kNormRowVecs + cluster_rank * kBlockSize + tx;
    vec_t vec[kWorldSize];
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::ld_relaxed_16B(vec[i], poll_ptr + i * stride_bytes, vid);
        has_zero |= Lamport::has_pos_zero(vec[i].data());
      }
      if (!has_zero) break;
    } while (true);

    vec_t out_vec;
    if (row < params.num_norm_rows) {  // cluster-uniform branch
      auto& smem = smem_raw[parity];
      parity ^= 1;
      // push each warp's partial to EVERY peer's slot for this block: lane p
      // (p < kClusterSize) targets peer p, warp w selects the [.][w] slot. So
      // after the barrier every block holds all kClusterSize*kNumWarps
      // partials in its own smem and reduces them locally — the read side
      // never touches remote DSMEM, so no post-read barrier is needed to guard
      // against a peer CTA exiting (parity double-buffers across rows).
      const auto lane = tx % kWarpThreads;
      const auto warp = tx / kWarpThreads;
      const auto warp_sqr = warp::reduce_sum(reduce_sqr(out_vec, vec));
      if (lane < kClusterSize) {
        float* dst = cluster.map_shared_rank(&smem[cluster_rank][warp], lane);
        *dst = warp_sqr;
      }
      cluster.sync();
      // load local
      float total = 0.0f;
#pragma unroll
      for (uint32_t r = 0; r < kClusterSize; ++r) {
        using vec_t = AlignedVector<float, kNumWarps>;
        vec_t remote_value;
        remote_value.load(smem[r]);
#pragma unroll
        for (uint32_t w = 0; w < kNumWarps; ++w) {
          total += remote_value[w];
        }
      }
      const auto norm_factor = math::rsqrt(total / kNormDim + params.norm_eps);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const auto [a, b] = cast<fp32x2_t>(out_vec[j]);
        const auto [wa, wb] = cast<fp32x2_t>(w[j]);
        out_vec[j] = cast<bf16x2_t>(fp32x2_t{a * norm_factor * wa, b * norm_factor * wb});
      }
    } else {
      out_vec = device::reduce_vec(vec);
    }

    ptx::st_global_16B(out_vec, params.input, vid);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_global_16B(zero_vec, poll_ptr + i * stride_bytes, vid);
    }
  }

  // epilogue: each row cluster flips its own counter; the bumper cluster
  // flips every remaining one so the whole array stays globally uniform
  if (cluster_rank == 0 && tx == 0) {
    params.push_counter[row_idx].set(phase ^ 1);
  }
}

// Pull family: low-SM NVLS, reusing the CustomAllReduceV2 pull semaphores

inline constexpr uint32_t kPullBlockSize = 512;

struct PullParams {
  uint8_t* input_mc;        // multicast VA of the symmetric input
  const uint8_t* residual;  // may be null (compile-time kHasResidual selects)
  Semaphore* sem_local;     // this rank's v2 pull semaphores (poll side)
  Semaphore* sem_mc;        // multicast VA of the pull-semaphore region
  uint32_t rank;
  uint32_t world_size;
  uint32_t num_vecs;  // 16B vectors
  // pull_norm only: RMSNorm epilogue over the first num_norm_rows rows of
  // the [num_vecs / kNormRowVecs, kNormDim] row view
  const uint8_t* norm_weight;
  float norm_eps;
  uint32_t num_norm_rows;
};

// K3 pull barriers reuse the v2 pull-semaphore slots with EXACTLY the
// generic kernels' reservation protocol — reserve a 2 * world_size flag
// window on the local m_counter, signal arrival on m_flag, wait for the
// window to fill — except that each arrival is ONE `multimem.red.add` on
// the m_flag's multicast alias instead of world_size per-peer unicast reds
// (same aggregate effect: every rank's flag gains world_size arrivals per
// phase). Identical memory effects per call, so both kernel families share
// the slots freely (single-stream calls are serialized).

// One pipelined pass at width kWidth (kWidth multimem loads in flight per
// thread), then recurse to kWidth/2 for the remainder, down to a plain
// unroll-1 tail. The cascade matters for mid sizes: a shard smaller than
// kUnroll * step would otherwise skip the main loop entirely and run the
// whole range with a single request in flight.
template <uint32_t kWidth, bool kHasResidual>
SGL_DEVICE void
pull_reduce_pass(uint32_t& vid, const uint32_t num_vecs, const uint32_t step, uint8_t* mc_ptr, const uint8_t* res_ptr) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;
  using SumOp = device::ReductionTrait<device::ReductionOp::SUM, bf16x2_t>;
  for (; vid + (kWidth - 1) * step < num_vecs; vid += kWidth * step) {
    vec_t vec[kWidth];
#pragma unroll
    for (uint32_t u = 0; u < kWidth; ++u) {
      device::ptx::ld_multimem_16B(vec[u], mc_ptr, vid + u * step);
    }
    if constexpr (kHasResidual) {
#pragma unroll
      for (uint32_t u = 0; u < kWidth; ++u) {
        vec_t res_vec;
        res_vec.load(res_ptr, vid + u * step);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          vec[u][j] = SumOp::reduce(vec[u][j], res_vec[j]);
        }
      }
    }
#pragma unroll
    for (uint32_t u = 0; u < kWidth; ++u) {
      device::ptx::st_multimem_16B(vec[u], mc_ptr, vid + u * step);
    }
  }
  if constexpr (kWidth > 1) {
    pull_reduce_pass<kWidth / 2, kHasResidual>(vid, num_vecs, step, mc_ptr, res_ptr);
  }
}

template <uint32_t kUnroll, bool kHasResidual, bool kUsePDL>
__global__
__launch_bounds__(kPullBlockSize, 1) void all_reduce_pull_res_kernel(const __grid_constant__ PullParams params) {
  static_assert(1 <= kUnroll && kUnroll <= 16 && (kUnroll & (kUnroll - 1)) == 0);

  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  // Reserve the window before the PDL wait, signal after it: the reservation's
  // RMW latency stays off the post-wait critical path, while the signal must
  // follow the wait because it asserts the producer grid has flushed.
  const auto barrier =
      device::distributed::McBarrier(params.sem_local, params.sem_mc, params.world_size, /*num_arrives=*/2);
  device::PDLWaitPrimary<kUsePDL>();
  barrier.arrive_relaxed(/*n=*/0);
  __syncthreads();

  // this rank's shard of the 16B-vector range
  const auto r = params.rank;
  const auto avg_vecs = params.num_vecs / params.world_size;
  const auto rem_vecs = params.num_vecs % params.world_size;
  const auto vec_bias = int64_t(avg_vecs) * r + min(r, rem_vecs);
  const auto num_vecs = avg_vecs + (r < rem_vecs ? 1 : 0);
  const auto mc_ptr = params.input_mc + vec_bias * 16;
  const auto res_ptr = kHasResidual ? params.residual + vec_bias * 16 : nullptr;

  // deep-pipelined body: issue up to kUnroll multimem loads back-to-back
  // before the first (residual add and) store, keeping kUnroll requests in
  // flight per thread to hide the NVLink latency with very few blocks; the
  // remainder cascades through halving widths down to 1.
  const auto step = kPullBlockSize * gridDim.x;
  auto vid = bx * kPullBlockSize + tx;
  pull_reduce_pass<kUnroll, kHasResidual>(vid, num_vecs, step, mc_ptr, res_ptr);

  // exit barrier: every peer has finished reading my buffer (and, for 2shot,
  // its broadcast into it is visible) before my next kernel may touch it.
  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq(/*n=*/1);
}

// Fused RMSNorm over the latent of the K3 latent|shared MoE buffer: the
// row-structured counterpart of the res kernel with kUnroll ROWS in flight
// per block. One block pass covers kUnroll consecutive rows: 448 threads
// each ld_reduce one 16B vector per row back-to-back, warp partials of the
// normed rows go to smem (one __syncthreads per group), non-norm rows store
// immediately. smem is parity double-buffered across groups so the next
// group's partial writes can't race this group's reads (the WAR pair is two
// groups apart and separated by the intervening group's barrier). Works on
// this rank's row shard, in place.
template <uint32_t kUnroll, bool kUsePDL>
__global__
__launch_bounds__(kNormRowVecs, 1) void all_reduce_pull_norm_kernel(const __grid_constant__ PullParams params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;
  using namespace device;

  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  // Reserve the window before the PDL wait, signal after it: the reservation's
  // RMW latency stays off the post-wait critical path, while the signal must
  // follow the wait because it asserts the producer grid has flushed.
  const auto barrier =
      device::distributed::McBarrier(params.sem_local, params.sem_mc, params.world_size, /*num_arrives=*/2);
  device::PDLWaitPrimary<kUsePDL>();
  barrier.arrive_relaxed(/*n=*/0);
  __syncthreads();

  // this rank's shard of the row range
  const auto num_rows = params.num_vecs / kNormRowVecs;
  const auto r = params.rank;
  const auto avg_rows = num_rows / params.world_size;
  const auto rem_rows = num_rows % params.world_size;
  const auto row_bias = avg_rows * r + min(r, rem_rows);
  const auto my_rows = avg_rows + (r < rem_rows ? 1 : 0);

  vec_t wvec;
  wvec.load(params.norm_weight, tx);
  __shared__ float smem[2][kUnroll][kNormWarps];
  const auto warp = tx / kWarpThreads;
  const auto lane = tx % kWarpThreads;
  uint32_t parity = 0;

  const auto row_step = gridDim.x * kUnroll;
  for (auto row0 = bx * kUnroll; row0 < my_rows; row0 += row_step) {
    const auto cnt = min(kUnroll, my_rows - row0);
    const auto vid0 = int64_t(row_bias + row0) * kNormRowVecs + tx;
    vec_t vec[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if (u < cnt) ptx::ld_multimem_16B(vec[u], params.input_mc, vid0 + u * kNormRowVecs);
    }
    // norm rows: push this warp's partial sum of squares to smem; non-norm
    // rows (the shared 2/3) don't wait for the barrier — store right away
    auto& sm = smem[parity];
    parity ^= 1;
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if (u >= cnt) continue;  // block-uniform
      if (row_bias + row0 + u < params.num_norm_rows) {
        float sum_of_squares = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          const auto [a, b] = cast<fp32x2_t>(vec[u][j]);
          sum_of_squares += a * a + b * b;
        }
        sum_of_squares = warp::reduce_sum(sum_of_squares);
        if (lane == 0) sm[u][warp] = sum_of_squares;
      } else {
        ptx::st_multimem_16B(vec[u], params.input_mc, vid0 + u * kNormRowVecs);
      }
    }
    __syncthreads();
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if (u >= cnt || row_bias + row0 + u >= params.num_norm_rows) continue;  // block-uniform
      float total = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kNormWarps; ++w) {
        total += sm[u][w];
      }
      const auto norm_factor = math::rsqrt(total / kNormDim + params.norm_eps);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const auto [a, b] = cast<fp32x2_t>(vec[u][j]);
        const auto [wa, wb] = cast<fp32x2_t>(wvec[j]);
        vec[u][j] = cast<bf16x2_t>(fp32x2_t{a * norm_factor * wa, b * norm_factor * wb});
      }
      ptx::st_multimem_16B(vec[u], params.input_mc, vid0 + u * kNormRowVecs);
    }
  }

  // exit barrier: every peer has finished reading my buffer (and, for 2shot,
  // its broadcast into it is visible) before my next kernel may touch it.
  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq(/*n=*/1);
}

inline auto choose_block_size(uint32_t num_threads) -> uint32_t {
  static const uint32_t kNumSM = [] {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    return host::runtime::get_sm_count(device);
  }();
  for (const uint32_t block_size : {128u, 256u, 512u}) {
    if (host::div_ceil(num_threads, block_size) <= kNumSM) return block_size;
  }
  return 1024u;
}

template <uint32_t kWorldSize, bool kUsePDL>
struct AllReduceFusionKernel {
 private:
  using TensorView = tvm::ffi::TensorView;

  template <bool kHasResidual>
  static constexpr auto res_push_kernel = all_reduce_push_res_kernel<kWorldSize, kHasResidual, kUsePDL>;

  static FusionParams
  make_params(const host::distributed::CommunicatorObj& comm, TensorView input, std::optional<TensorView> residual) {
    using namespace host;
    const auto& push = comm.get_push_obj();
    SymbolicSize N = {"num_elements"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    if (residual.has_value()) {
      TensorMatcher({N})  //
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(input)
          .verify(residual.value());
    } else {
      TensorMatcher({N})  //
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(input);
    }
    const auto num_elems = N.unwrap();
    CHECK_HOST(push.world_size == kWorldSize);
    CHECK_HOST(num_elems > 0 && num_elems % 8 == 0);
    CHECK_HOST(push.mc_workspace != nullptr) << "the fused push needs a multicast-capable push plane";
    FusionParams params{};
    params.input = static_cast<uint8_t*>(input.data_ptr());
    params.residual = residual.has_value() ? static_cast<const uint8_t*>(residual.value().data_ptr()) : nullptr;
    params.push_ws_mc = push.mc_workspace;
    params.push_ws_local = push.workspaces[push.rank];
    params.push_counter = push.counter;
    params.push_buffer_stride = push.slot_bytes;
    params.rank = push.rank;
    params.num_vecs = static_cast<uint32_t>(num_elems / 8);
    return params;
  }

  /// The input viewed as rows of the norm width (3584 bf16 each); the first
  /// num_norm_rows get the RMSNorm epilogue, the rest are a plain allreduce
  /// (K3 uses [N latent rows | 2N shared rows] with num_norm_rows = N, or a
  /// latent-only [N, 3584] tensor with num_norm_rows = N).
  static FusionParams make_params_norm(
      const host::distributed::CommunicatorObj& comm,
      TensorView input,
      TensorView weight,
      float eps,
      int64_t num_norm_rows) {
    using namespace host;
    auto params = make_params(comm, input, std::nullopt);
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({kNormDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(weight);
    CHECK_HOST(params.num_vecs % kNormRowVecs == 0)
        << "numel must be a multiple of " << kNormDim << ", got " << int64_t(params.num_vecs) * 8;
    const auto num_rows = params.num_vecs / kNormRowVecs;
    CHECK_HOST(0 <= num_norm_rows && num_norm_rows <= num_rows)
        << "num_norm_rows " << num_norm_rows << " out of range [0, " << num_rows << "]";
    params.norm_weight = static_cast<const uint8_t*>(weight.data_ptr());
    params.norm_eps = static_cast<float>(eps);
    params.num_norm_rows = static_cast<uint32_t>(num_norm_rows);
    params.num_push_counters = comm.get_push_obj().num_blocks;
    return params;
  }

  // Shared pull validation; the reduce is in place on the symmetric input.
  static PullParams make_pull_params(
      const host::distributed::CommunicatorObj& comm,
      TensorView input,
      std::optional<TensorView> residual,
      int64_t input_mc_ptr) {
    using namespace host;
    const auto& pull = comm.get_pull_obj();
    SymbolicSize N = {"num_elements"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    if (residual.has_value()) {
      TensorMatcher({N})  //
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(input)
          .verify(residual.value());
    } else {
      TensorMatcher({N})  //
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(input);
    }
    const auto num_elems = N.unwrap();
    CHECK_HOST(pull.world_size == kWorldSize);
    CHECK_HOST(num_elems > 0 && num_elems % 8 == 0) << "numel must be a positive multiple of 8, got " << num_elems;
    // headroom below 2^32 so the unrolled loop's `vid + (kUnroll-1)*step`
    // arithmetic can never wrap around u32
    CHECK_HOST(num_elems / 8 < (int64_t(1) << 31)) << "numel exceeds the 16B-vector limit";
    CHECK_HOST(input_mc_ptr != 0) << "pull requires the input's multicast address";
    CHECK_HOST(pull.mc_semaphore != nullptr) << "pull requires a multicast-capable pull plane";
    PullParams params{};
    params.input_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(input_mc_ptr));
    params.residual = residual.has_value() ? static_cast<const uint8_t*>(residual.value().data_ptr()) : nullptr;
    params.sem_local = pull.semaphores[pull.rank];
    params.sem_mc = pull.mc_semaphore;
    params.rank = pull.rank;
    params.world_size = pull.world_size;
    params.num_vecs = static_cast<uint32_t>(num_elems / 8);
    return params;
  }

  // Runtime unroll dispatch: every supported width is compiled into the
  // module so the tuned per-size unroll needs no extra JIT builds.
  template <bool kHasResidual>
  static void launch_pull_res(const PullParams& params, int64_t num_blocks, int64_t unroll, DLDevice device) {
    const auto run = [&](auto kernel) {
      host::LaunchKernel(static_cast<uint32_t>(num_blocks), kPullBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
    };
    switch (unroll) {
      case 2:
        return run(all_reduce_pull_res_kernel<2, kHasResidual, kUsePDL>);
      case 4:
        return run(all_reduce_pull_res_kernel<4, kHasResidual, kUsePDL>);
      case 8:
        return run(all_reduce_pull_res_kernel<8, kHasResidual, kUsePDL>);
      case 16:
        return run(all_reduce_pull_res_kernel<16, kHasResidual, kUsePDL>);
      default:
        CHECK_HOST(false) << "unsupported unroll " << unroll << " (must be 2, 4, 8, or 16)";
    }
  }

  static void launch_pull_norm(const PullParams& params, int64_t num_blocks, int64_t unroll, DLDevice device) {
    const auto run = [&](auto kernel) {
      host::LaunchKernel(static_cast<uint32_t>(num_blocks), kNormRowVecs, device).enable_pdl(kUsePDL)(kernel, params);
    };
    switch (unroll) {
      case 2:
        return run(all_reduce_pull_norm_kernel<2, kUsePDL>);
      case 4:
        return run(all_reduce_pull_norm_kernel<4, kUsePDL>);
      case 8:
        return run(all_reduce_pull_norm_kernel<8, kUsePDL>);
      case 16:
        return run(all_reduce_pull_norm_kernel<16, kUsePDL>);
      default:
        CHECK_HOST(false) << "unsupported unroll " << unroll << " (must be 2, 4, 8, or 16)";
    }
  }

 public:
  static void push_res(CommunicatorRef ref, TensorView input, std::optional<TensorView> residual) {
    const auto& push = ref.get()->get_push_obj();
    auto params = make_params(*ref.get(), input, residual);
    const int64_t nbytes = int64_t(params.num_vecs) * 16;
    CHECK_HOST(nbytes <= push.slot_bytes) << "input size " << nbytes << " exceeds push slot size " << push.slot_bytes;
    const auto kernel = residual.has_value() ? res_push_kernel<true> : res_push_kernel<false>;
    host::LaunchKernel(push.num_blocks, choose_block_size(params.num_vecs), input.device())
        .enable_pdl(kUsePDL)(kernel, params);
  }

  static void push_norm(CommunicatorRef ref, TensorView input, TensorView weight, float eps, int64_t num_norm_rows) {
    constexpr auto kClusterSize = 7;
    const auto& push = ref.get()->get_push_obj();
    auto params = make_params_norm(*ref.get(), input, weight, eps, num_norm_rows);
    const int64_t nbytes = int64_t(params.num_vecs) * 16;
    CHECK_HOST(nbytes <= push.slot_bytes) << "input size " << nbytes << " exceeds push slot size " << push.slot_bytes;
    const auto num_rows = params.num_vecs / kNormRowVecs;
    constexpr uint32_t kMaxClusters = 96;
    const auto num_row_clusters = std::max<uint32_t>(std::min(num_rows, kMaxClusters), 1);
    CHECK_HOST(num_row_clusters < push.num_blocks);
    host::LaunchKernel((num_row_clusters + 1) * kClusterSize, kNormRowVecs / kClusterSize, input.device())
        .enable_pdl(kUsePDL)(all_reduce_push_norm_cluster_kernel<kWorldSize, kClusterSize, kUsePDL>, params);
  }

  /// Deferred MoE finalize + 1shot push all-reduce + RMSNorm over EVERY row.
  /// `out` (flattened [num_tokens * kNormDim] bf16) is output-only: each
  /// rank's partial latent is computed from the trtllm-gen deferred-finalize
  /// triple during the staging pass and never materializes in global memory.
  static void finalize_push_norm(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      TensorView weight,
      float eps) {
    using namespace host;
    constexpr auto kClusterSize = 7;
    const auto& push = ref.get()->get_push_obj();
    // every row of the latent-only output is normed
    auto params = make_params_norm(*ref.get(), out, weight, eps, out.size(0) / kNormDim);
    const auto num_tokens = params.num_vecs / kNormRowVecs;

    auto P = SymbolicSize{"num_permuted_rows"};
    auto T = SymbolicSize{"num_tokens"};
    T.set_value(num_tokens);
    auto K = SymbolicSize{"top_k"};
    auto TK = SymbolicSize{"num_expanded"};
    TK.set_value(static_cast<int64_t>(num_tokens) * kFinTopK);
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({P, kNormDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(gemm2_out);
    TensorMatcher({T, K}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(expert_weights);
    TensorMatcher({TK}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(permuted_idx);
    CHECK_HOST(K.unwrap() == kFinTopK) << "finalize_push_norm is specialized for top_k = " << kFinTopK;

    const int64_t nbytes = int64_t(params.num_vecs) * 16;
    CHECK_HOST(nbytes <= push.slot_bytes) << "output size " << nbytes << " exceeds push slot size " << push.slot_bytes;
    params.fin_gemm2 = static_cast<const uint8_t*>(gemm2_out.data_ptr());
    params.fin_idx = static_cast<const uint8_t*>(permuted_idx.data_ptr());
    params.fin_weights = static_cast<const uint8_t*>(expert_weights.data_ptr());

    constexpr uint32_t kMaxClusters = 96;
    const auto num_row_clusters = std::max<uint32_t>(std::min(num_tokens, kMaxClusters), 1);
    CHECK_HOST(num_row_clusters < push.num_blocks);
    host::LaunchKernel((num_row_clusters + 1) * kClusterSize, kNormRowVecs / kClusterSize, out.device())
        .enable_pdl(kUsePDL)(
            all_reduce_push_norm_cluster_kernel<kWorldSize, kClusterSize, kUsePDL, /*kFinalize=*/true>, params);
  }

  /// Low-SM NVLS pull (+ optional residual): in-place reduce-scatter +
  /// broadcast on the symmetric input, whose multicast VA the caller passes
  /// (it varies per call, unlike the barrier plane's own multicast base).
  /// num_blocks -- which must be uniform across ranks per call -- is clamped to
  /// the barrier plane's capacity.
  static void pull_res(
      CommunicatorRef ref,
      TensorView input,
      std::optional<TensorView> residual,
      int64_t input_mc_ptr,
      int64_t num_blocks,
      int64_t unroll) {
    const auto params = make_pull_params(*ref.get(), input, residual, input_mc_ptr);
    CHECK_HOST(num_blocks >= 1) << "invalid num_blocks: " << num_blocks;
    num_blocks = std::min<int64_t>(num_blocks, ref.get()->get_pull_blocks());
    if (residual.has_value()) {
      launch_pull_res<true>(params, num_blocks, unroll, input.device());
    } else {
      launch_pull_res<false>(params, num_blocks, unroll, input.device());
    }
  }

  /// Low-SM NVLS pull + RMSNorm over the latent of the K3 latent|shared MoE
  /// buffer ([num_tokens, 3584] latent then [num_tokens, 7168] shared);
  /// num_tokens and the normed row range are derived from the element count.
  /// Same semaphore / num_blocks semantics as pull_res.
  static void pull_norm(
      CommunicatorRef ref,
      TensorView input,
      TensorView weight,
      double eps,
      int64_t num_norm_rows,
      int64_t input_mc_ptr,
      int64_t num_blocks,
      int64_t unroll) {
    auto params = make_pull_params(*ref.get(), input, std::nullopt, input_mc_ptr);
    CHECK_HOST(num_blocks >= 1) << "invalid num_blocks: " << num_blocks;
    num_blocks = std::min<int64_t>(num_blocks, ref.get()->get_pull_blocks());
    using namespace host;
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({kNormDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(weight);
    CHECK_HOST(params.num_vecs % kNormRowVecs == 0)
        << "numel must be a multiple of " << kNormDim << ", got " << int64_t(params.num_vecs) * 8;
    const auto num_rows = params.num_vecs / kNormRowVecs;
    CHECK_HOST(0 <= num_norm_rows && num_norm_rows <= num_rows)
        << "num_norm_rows " << num_norm_rows << " out of range [0, " << num_rows << "]";
    params.norm_weight = static_cast<const uint8_t*>(weight.data_ptr());
    params.norm_eps = static_cast<float>(eps);
    params.num_norm_rows = static_cast<uint32_t>(num_norm_rows);
    launch_pull_norm(params, num_blocks, unroll, input.device());
  }
};

}  // namespace sglang
