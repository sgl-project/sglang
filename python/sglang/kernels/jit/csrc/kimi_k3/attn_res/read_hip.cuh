// Kimi-K3 TP8 all-gather of the next stream fused with its attention-residual read (ROCm, wave64).
//
// One workgroup per token row; each thread owns 8 consecutive elements of the row. Every rank
// finishes its [T, H / 8] column slice of the stream from the column-parallel latent-MoE
// tail's split-K partials,
//     slice = bf16(bf16(sum_k part[k] / rms(latent)) + shared [+ prefix])
// (kimi_k3_latent_tail_hip.k3_latent_tail on 1/8 of the up_proj rows). The kernel gathers the
// slices into the stream and reads it,
//     out = bf16(rmsnorm(softmax mix of bank[0..nvb) and stream) * ow)
// optionally snapshotting the stream into bank[:, nvb]: attn_res_hip's _agg_kernel, with the
// scores' transcendentals run once per row and DPP wave reductions.
//
// The all-gather is LL: each rank pushes its slice into every peer's scratch as 16-byte lines of
// 8 payload bytes + an epoch flag, and the receiver polls its own scratch, so there is no
// barrier. The bank rows are loaded and scored while the peers' lines are in flight.
//
// Scratch (receiver-owned, the post-Signal tail of a dedicated aiter CustomAllreduce's meta
// buffer): [2 epochs][kWorld src ranks][kMaxRows rows][lines]. Row t is always handled by
// workgroup t, whose flag epochs[t] + 1 is rank-local; every rank launches the same row counts,
// so the flags agree. A line is rewritten two epochs later, by which point its receiver has
// consumed it (the writer has since received the receiver's next-epoch row).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "../comm/ll_hip.cuh"
#include "common_hip.cuh"
#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER csrc/include: CustomAllreduce, Signal

#ifndef USE_ROCM
#error "kimi_k3/attn_res/read_hip.cuh is a wave64 (ROCm) kernel"
#endif

namespace sglang {
namespace k3_attn_res_read {

using namespace k3_ll;

inline constexpr int kWorld = 8;
inline constexpr int kMaxRows = 64;
inline constexpr int kSlice = kHidden / kWorld;  // all-gather columns per rank
inline constexpr int kSliceThreads = kSlice / kVec;
// 8 payload bytes per 16-byte line.
inline constexpr int kAgLinesPerRow = kSlice * 2 / 8;
inline constexpr int64_t kAgEpochLines = int64_t{kWorld} * kMaxRows * kAgLinesPerRow;
// The first region of the gather scratch (comm/front_gather_hip.cuh owns the second) and the
// first kMaxRows of its flags.
inline constexpr int kEpochs = 2 * kMaxRows;
// The column-parallel latent-MoE tail the all-gather can finish: its GEMM's fp32 partials over
// kTailSplits chunks of the kTailK latent dim (kimi_k3_latent_tail_hip.k3_latent_tail_partials).
inline constexpr int kTailK = 3584;
inline constexpr int kTailSplits = 7;
// The attention output's one-shot all-reduce fused with the FFN-side read (k3_ar_attn_res_kernel):
// every rank pushes its whole partial row, two lines per 16-byte vector, into its own slot of
// each peer's region, which follows the MoE front gather's (comm/front_gather_hip.cuh).
inline constexpr int kArMaxRows = 16;
inline constexpr int kArLinesPerRow = 2 * kThreads;
inline constexpr int64_t kArEpochLines = int64_t{kWorld} * kArMaxRows * kArLinesPerRow;
inline constexpr int64_t kArRegionLine = 2 * kAgEpochLines + 2 * int64_t{kWorld} * kMaxRows * 144;
inline constexpr int kArPushers = kWorld - 1;  // pusher workgroups per row, one per peer
static_assert(kThreads % kWave == 0 && kSlice % kVec == 0, "whole waves and vectors");

struct Params {
  bf16_t* __restrict__ bank;       // [T, NB, H] bf16
  const float4* __restrict__ cw;   // [H] fp32, score norm weight * score proj weight
  const float4* __restrict__ ow;   // [H] fp32, output RMSNorm weight
  uint4* __restrict__ prefix_out;  // [T, H] bf16
  uint4* __restrict__ out;         // [T, H] bf16
  // Every rank's scratch base. Index it with constants only: a runtime index turns the
  // kernarg read into a vector load, whose wait then also waits for in-flight remote stores.
  uint4* scratch[kWorld];
  const uint4* self_scratch;     // scratch[rank]
  int32_t* __restrict__ epochs;  // [kEpochs], rank-local
  int64_t bank_stride_t;         // elements
  int64_t bank_stride_b;
  // AttnResRead<0>: the bank's softmax partials for this read (bank_partial_hip.cuh).
  const float4* __restrict__ bank_acc;  // [T, H] fp32
  const float2* __restrict__ bank_ms;   // [T] (m, sum)
  int32_t nvb_write;                    // bank row the prefix is snapshotted into, or -1
  int32_t rank;
  float score_eps;
  float out_eps;
  // Tail source: this rank's slice = bf16(bf16(sum_k part[k] / rms) + shared [+ prefix]).
  const float* __restrict__ tail_part;     // [kTailSplits, T, H / 8] fp32
  const float* __restrict__ tail_sq;       // [kTailSplits, T] fp32, sums of squares of the latent
  const bf16_t* __restrict__ tail_shared;  // [T, H] bf16 (row stride tail_stride), full columns
  const bf16_t* __restrict__ tail_prefix;  // [T, H] bf16 (row stride tail_stride), or null
  int64_t tail_stride;
  float tail_eps;
  // All-reduce read: this rank's attention partial and the pending residual, [T, H] bf16 each
  // (residual null at a block-write layer's FFN read); per-row flags and, for each row, the
  // count of this rank's pushers that have taken the row's flag.
  const uint4* __restrict__ partial;
  const uint4* __restrict__ residual;
  int32_t* __restrict__ ar_epochs;  // [kArMaxRows], rank-local
  int32_t* __restrict__ ar_pushed;  // [kArMaxRows], rank-local, left at 0
};

// The attention-residual read of one row: bank rows + cw are this thread's vectors.
template <int kNvb>
struct AttnResRead {
  static constexpr int kSlots = 2 * kNvb + 2;  // per bank row (dot, sumsq), then the prefix's

  struct Shared {
    // The bank rows wait out the collective here, not in registers (128 VGPRs at kThreads).
    uint4 bank[kNvb][kThreads];
    float part[kWaves][kSlots];
    float weight[kNvb + 1];
    float sq[kWaves];
  };

  uint4 bank[kNvb];
  float cw[kVec];
  float ow[kVec];

  SGL_DEVICE void load(const Params& p, int t, int tid) {
    const bf16_t* row = p.bank + t * p.bank_stride_t + tid * kVec;
#pragma unroll
    for (int r = 0; r < kNvb; ++r)
      bank[r] = *reinterpret_cast<const uint4*>(row + r * p.bank_stride_b);
    load_fp32x8(p.cw + 2 * tid, cw);
    // Loaded up front: a load after the collective's stores would wait for their acks
    // (vmcnt retires in order).
    load_fp32x8(p.ow + 2 * tid, ow);
  }

  SGL_DEVICE void store_partials(Shared& s, int slot, const float (&f)[kVec], int wave, int lane) const {
    float dot = 0.f, sq = 0.f;
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      dot += f[e] * cw[e], sq += f[e] * f[e];
    dot = wave_sum(dot);
    sq = wave_sum(sq);
    if (lane == 0) s.part[wave][2 * slot] = dot, s.part[wave][2 * slot + 1] = sq;
  }

  // Before the collective completes: the bank rows' score partials. Each thread
  // reads back only its own LDS vectors, so the stash needs no barrier.
  SGL_DEVICE void score_bank(Shared& s, int tid, int wave, int lane) const {
#pragma unroll
    for (int r = 0; r < kNvb; ++r) {
      float f[kVec];
      unpack8(bank[r], f);
      store_partials(s, r, f, wave, lane);
      s.bank[r][tid] = bank[r];
    }
  }

  // After it: score the prefix row, mix, normalize; returns this thread's output vector.
  SGL_DEVICE uint4 finish(Shared& s, const Params& p, const float (&prefix)[kVec], int tid, int wave, int lane) const {
    store_partials(s, kNvb, prefix, wave, lane);
    lds_barrier();
    // Wave 0 scores the candidates, lane r candidate r (bank rows, then the prefix), and
    // publishes the softmax weights: the transcendentals run once, not on every thread.
    static_assert(kNvb + 1 <= 16, "candidates fit one 16-lane group");
    if (wave == 0) {
      const bool live = lane <= kNvb;
      float score = -INFINITY;
      if (live) {
        float dot = 0.f, sq = 0.f;
#pragma unroll
        for (int w = 0; w < kWaves; ++w)
          dot += s.part[w][2 * lane], sq += s.part[w][2 * lane + 1];
        score = dot / sqrtf(sq / kHidden + p.score_eps);
      }
      float m = score;
#pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1)
        m = fmaxf(m, __shfl_xor(m, offset, kWave));
      const float e = live ? __expf(score - m) : 0.f;
      float denom = e;
#pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1)
        denom += __shfl_xor(denom, offset, kWave);
      if (live) s.weight[lane] = e * (1.0f / denom);
    }
    lds_barrier();

    float mix[kVec];
    const float wp = s.weight[kNvb];
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      mix[e] = prefix[e] * wp;
#pragma unroll
    for (int r = 0; r < kNvb; ++r) {
      float f[kVec];
      unpack8(s.bank[r][tid], f);
      const float w = s.weight[r];
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        mix[e] += w * f[e];
    }

    float sq = 0.f;
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      sq += mix[e] * mix[e];
    sq = wave_sum(sq);
    if (lane == 0) s.sq[wave] = sq;
    lds_barrier();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w)
      total += s.sq[w];
    const float scale = 1.0f / sqrtf(total / kHidden + p.out_eps);

#pragma unroll
    for (int e = 0; e < kVec; ++e)
      mix[e] = mix[e] * scale * ow[e];
    return pack8(mix);
  }
};

// The read with the bank's softmax partials computed ahead (bank_acc, bank_ms): only the
// stream is scored once the collective completes, then merged as one more candidate.
template <>
struct AttnResRead<0> {
  struct Shared {
    float part[kWaves][2];  // the stream's (dot, sumsq)
    float weight[2];        // the partial's scale and the stream's weight
    float sq[kWaves];
  };

  float acc[kVec];
  float2 ms;
  float cw[kVec];
  float ow[kVec];

  SGL_DEVICE void load(const Params& p, int t, int tid) {
    load_fp32x8(p.bank_acc + (static_cast<int64_t>(t) * kThreads + tid) * 2, acc);
    ms = p.bank_ms[t];
    load_fp32x8(p.cw + 2 * tid, cw);
    // Loaded up front: a load after the collective's stores would wait for their acks.
    load_fp32x8(p.ow + 2 * tid, ow);
  }

  // Nothing to score, but the kernels' push and poll sides must stay apart: a convergent
  // point keeps a wave's pushing lanes from being held back behind its polling lanes.
  SGL_DEVICE void score_bank(Shared&, int, int, int) const {
    __builtin_amdgcn_wave_barrier();
  }

  SGL_DEVICE uint4 finish(Shared& s, const Params& p, const float (&prefix)[kVec], int tid, int wave, int lane) const {
    float dot = 0.f, sq = 0.f;
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      dot += prefix[e] * cw[e], sq += prefix[e] * prefix[e];
    dot = wave_sum(dot);
    sq = wave_sum(sq);
    if (lane == 0) s.part[wave][0] = dot, s.part[wave][1] = sq;
    lds_barrier();
    if (tid == 0) {
      float d = 0.f, q = 0.f;
#pragma unroll
      for (int w = 0; w < kWaves; ++w)
        d += s.part[w][0], q += s.part[w][1];
      const float score = d / sqrtf(q / kHidden + p.score_eps);
      const float m = fmaxf(ms.x, score);
      const float bank_scale = __expf(ms.x - m);
      const float stream = __expf(score - m);
      const float inv = 1.0f / (ms.y * bank_scale + stream);
      s.weight[0] = bank_scale * inv;
      s.weight[1] = stream * inv;
    }
    lds_barrier();

    float mix[kVec];
    const float wb = s.weight[0], wp = s.weight[1];
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      mix[e] = prefix[e] * wp + acc[e] * wb;

    float total_sq = 0.f;
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      total_sq += mix[e] * mix[e];
    total_sq = wave_sum(total_sq);
    if (lane == 0) s.sq[wave] = total_sq;
    lds_barrier();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w)
      total += s.sq[w];
    const float scale = 1.0f / sqrtf(total / kHidden + p.out_eps);

#pragma unroll
    for (int e = 0; e < kVec; ++e)
      mix[e] = mix[e] * scale * ow[e];
    return pack8(mix);
  }
};

// This thread's vector of the rank's tail slice, as k3_latent_tail computes it.
template <bool kHasPrefix>
SGL_DEVICE uint4 finish_tail(const Params& p, int t, int rows, int slice_vec) {
  float acc[kVec] = {};
  float sq = 0.f;
#pragma unroll
  for (int k = 0; k < kTailSplits; ++k) {
    const float4* part =
        reinterpret_cast<const float4*>(p.tail_part + (static_cast<int64_t>(k) * rows + t) * kSlice + slice_vec * kVec);
    const float4 a = part[0], b = part[1];
    acc[0] += a.x, acc[1] += a.y, acc[2] += a.z, acc[3] += a.w;
    acc[4] += b.x, acc[5] += b.y, acc[6] += b.z, acc[7] += b.w;
    sq += p.tail_sq[k * rows + t];
  }
  const int64_t col = static_cast<int64_t>(t) * p.tail_stride + p.rank * kSlice + slice_vec * kVec;
  const uint4 sv = *reinterpret_cast<const uint4*>(p.tail_shared + col);
  uint4 rv = make_uint4(0, 0, 0, 0);
  if constexpr (kHasPrefix) rv = *reinterpret_cast<const uint4*>(p.tail_prefix + col);
  const float inv_rms = 1.0f / sqrtf(sq / kTailK + p.tail_eps);
  float proj[kVec], shared[kVec], prefix[kVec];
#pragma unroll
  for (int e = 0; e < kVec; ++e)
    proj[e] = acc[e] * inv_rms;
  unpack8(pack8(proj), proj);  // the projection is bf16 before the adds see it
  unpack8(sv, shared);
  unpack8(rv, prefix);
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    proj[e] += shared[e];
    if constexpr (kHasPrefix) proj[e] += prefix[e];
  }
  return pack8(proj);
}

template <int kNvb, bool kWriteBank, bool kHasPrefix>
__global__ __launch_bounds__(kThreads, 1) void k3_ag_attn_res_kernel(const Params p) {
  using Read = AttnResRead<kNvb>;
  __shared__ typename Read::Shared smem;
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid / kWave;
  const int lane = tid % kWave;
  const uint32_t flag = next_flag(p.epochs, t);
  const int64_t epoch_off = static_cast<int64_t>(flag & 1u) * kAgEpochLines;
  const int64_t row_vec = static_cast<int64_t>(t) * kThreads + tid;
  // The rank whose slice holds this thread's vector, and the vector's place in it.
  const int src = tid / kSliceThreads;
  const int slice_vec = tid % kSliceThreads;
  const bool mine = src == p.rank;

  uint4 pv = make_uint4(0, 0, 0, 0);
  if (mine) pv = finish_tail<kHasPrefix>(p, t, static_cast<int>(gridDim.x), slice_vec);
  Read read;
  read.load(p, t, tid);

  // Pushers and pollers share waves, and a wave runs one side of a branch at a time: the push
  // comes before the poll in program order, or a polling wave would hold back its own lanes'
  // pushes that the peer is waiting for.
  if (mine) {
    const int64_t row_line = epoch_off + (static_cast<int64_t>(p.rank) * kMaxRows + t) * kAgLinesPerRow;
#pragma unroll
    for (int peer = 0; peer < kWorld; ++peer) {
      if (peer != p.rank) ll_push(p.scratch[peer] + row_line, slice_vec, kSliceThreads, pv, flag);
    }
  }

  read.score_bank(smem, tid, wave, lane);

  if (!mine) {
    const uint4* s =
        p.self_scratch + epoch_off + (static_cast<int64_t>(src) * kMaxRows + t) * kAgLinesPerRow + slice_vec;
    uint4 a, b;
    do {
      a = ll_load(s);
      b = ll_load(s + kSliceThreads);
    } while (!ll_ready(a, b, flag));
    pv = ll_payload(a, b);
  }
  p.prefix_out[row_vec] = pv;
  if constexpr (kWriteBank) {
    *reinterpret_cast<uint4*>(p.bank + t * p.bank_stride_t + p.nvb_write * p.bank_stride_b + tid * kVec) = pv;
  }
  float row[kVec];
  unpack8(pv, row);
  p.out[row_vec] = read.finish(smem, p, row, tid, wave, lane);
  if (tid == 0) p.epochs[t] = static_cast<int32_t>(flag);
}

// A peer's scratch base, read with a constant kernarg index (see Params::scratch).
SGL_DEVICE uint4* peer_scratch(const Params& p, int peer) {
  uint4* base = p.scratch[0];
#pragma unroll
  for (int i = 1; i < kWorld; ++i)
    if (peer == i) base = p.scratch[i];
  return base;
}

// One-shot all-reduce of the attention output's TP partials, its residual add and the FFN-side
// attention-residual read: out = bf16(rmsnorm(mix(bank[0..nvb), prefix)) * ow) with
// prefix = bf16(residual + bf16(sum over ranks of partial)).
// Workgroups [0, T) each read one row; workgroup T + 7 t + j pushes row t to the j-th next peer,
// so no CU pushes a whole row seven times. The sum runs in rank order, identical on every rank.
// A row's flag is ar_epochs[t] + 1 for its reader and its pushers alike; the reader publishes
// it only after its rank's pushers of the row have all taken it (ar_pushed), as a late pusher
// would otherwise read the next flag.
template <int kNvb, bool kHasResidual>
__global__ __launch_bounds__(kThreads, 1) void k3_ar_attn_res_kernel(const Params p) {
  using Read = AttnResRead<kNvb>;
  __shared__ typename Read::Shared smem;
  const int rows = static_cast<int>(gridDim.x) / kWorld;
  const int tid = threadIdx.x;
  if (static_cast<int>(blockIdx.x) >= rows) {
    const int j = static_cast<int>(blockIdx.x) - rows;
    const int t = j / kArPushers;
    const int peer = (p.rank + 1 + j % kArPushers) % kWorld;
    const uint32_t flag = next_flag(p.ar_epochs, t);
    const int64_t epoch_off = kArRegionLine + static_cast<int64_t>(flag & 1u) * kArEpochLines;
    const uint4 v = p.partial[static_cast<int64_t>(t) * kThreads + tid];
    const int64_t row_line = epoch_off + (static_cast<int64_t>(p.rank) * kArMaxRows + t) * kArLinesPerRow;
    ll_push(peer_scratch(p, peer) + row_line, tid, kThreads, v, flag);
    // Every lane has issued its stores, so every lane has read the flag.
    lds_barrier();
    if (tid == 0) atomicAdd(p.ar_pushed + t, 1);
    return;
  }

  const int t = static_cast<int>(blockIdx.x);
  const int wave = tid / kWave;
  const int lane = tid % kWave;
  const uint32_t flag = next_flag(p.ar_epochs, t);
  const int64_t epoch_off = kArRegionLine + static_cast<int64_t>(flag & 1u) * kArEpochLines;
  const int64_t row_vec = static_cast<int64_t>(t) * kThreads + tid;
  const uint4 own = p.partial[row_vec];
  uint4 res = make_uint4(0, 0, 0, 0);
  if constexpr (kHasResidual) res = p.residual[row_vec];
  Read read;
  read.load(p, t, tid);
  read.score_bank(smem, tid, wave, lane);

  float sum[kVec] = {};
#pragma unroll
  for (int src = 0; src < kWorld; ++src) {
    uint4 v = own;
    if (src != p.rank) {
      const uint4* s = p.self_scratch + epoch_off + (static_cast<int64_t>(src) * kArMaxRows + t) * kArLinesPerRow + tid;
      uint4 a, b;
      do {
        a = ll_load(s);
        b = ll_load(s + kThreads);
      } while (!ll_ready(a, b, flag));
      v = ll_payload(a, b);
    }
    float f[kVec];
    unpack8(v, f);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      sum[e] += f[e];
  }
  // The all-reduce's bf16 output, then the residual add rounded as the unfused read does.
  float prefix[kVec];
  uint4 pv = pack8(sum);
  unpack8(pv, prefix);
  if constexpr (kHasResidual) {
    float r[kVec];
    unpack8(res, r);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      prefix[e] = r[e] + prefix[e];
    pv = pack8(prefix);
    unpack8(pv, prefix);
  }
  p.prefix_out[row_vec] = pv;
  p.out[row_vec] = read.finish(smem, p, prefix, tid, wave, lane);
  if (tid == 0) {
    while (__hip_atomic_load(p.ar_pushed + t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < kArPushers) {
    }
    p.ar_pushed[t] = 0;
    p.ar_epochs[t] = static_cast<int32_t>(flag);
  }
}

template <int kNvb, bool kHasPrefix>
void launch_gather(const Params& params, bool write_bank, int rows, DLDevice device) {
  if (write_bank) {
    host::LaunchKernel(rows, kThreads, device)(k3_ag_attn_res_kernel<kNvb, true, kHasPrefix>, params);
  } else {
    host::LaunchKernel(rows, kThreads, device)(k3_ag_attn_res_kernel<kNvb, false, kHasPrefix>, params);
  }
}

template <int kNvb>
void launch(const Params& params, bool has_residual, bool write_bank, int rows, DLDevice device) {
  if (has_residual) {
    launch_gather<kNvb, true>(params, write_bank, rows, device);
  } else {
    launch_gather<kNvb, false>(params, write_bank, rows, device);
  }
}

// The bank partials a read takes instead of the bank rows: acc [T, H] and ms [T, 2], fp32.
inline void verify_bank_partial(
    const tvm::ffi::TensorView acc,
    const tvm::ffi::TensorView ms,
    host::SymbolicSize& T,
    host::SymbolicDevice& device) {
  using namespace host;
  TensorMatcher({T, kHidden}).with_dtype<float>().with_device(device).verify(acc);
  TensorMatcher({T, 2}).with_dtype<float>().with_device(device).verify(ms);
}

struct K3AttnResReadKernel {
  // The all-gather read over `comm`, the dedicated aiter CustomAllreduce* whose meta buffers
  // hold the LL scratch. Each rank's slice is finished from the column tail's partials:
  // tail_part [7, T, H / 8], tail_sq [7, T], tail_shared [T, H] and, when has_residual,
  // residual = the tail's prefix [T, H] (same row stride). write_bank snapshots the stream
  // into bank[:, nvb].
  static void
  run(int64_t comm,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView prefix_out,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView epochs,
      const tvm::ffi::TensorView tail_part,
      const tvm::ffi::TensorView tail_sq,
      const tvm::ffi::TensorView tail_shared,
      const tvm::ffi::TensorView bank_acc,
      const tvm::ffi::TensorView bank_ms,
      int64_t nvb,
      bool has_residual,
      bool write_bank,
      bool bank_partial,
      double score_eps,
      double out_eps,
      double tail_eps) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto NB = SymbolicSize{"num_blocks"};
    auto S = SymbolicSize{"tail_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLROCM>();
    TensorMatcher({T, kHidden}).with_dtype<bf16_t>().with_device(device).verify(prefix_out).verify(out);
    TensorMatcher({T, NB, kHidden}).with_dtype<bf16_t>().with_device(device).with_strides({-1, -1, 1}).verify(bank);
    TensorMatcher({kHidden}).with_dtype<float>().with_device(device).verify(cw).verify(ow);
    TensorMatcher({kTailSplits, T, kSlice}).with_dtype<float>().with_device(device).verify(tail_part);
    TensorMatcher({kTailSplits, T}).with_dtype<float>().with_device(device).verify(tail_sq);
    TensorMatcher({T, kHidden}).with_dtype<bf16_t>().with_device(device).with_strides({S, 1}).verify(tail_shared);
    if (has_residual) {
      TensorMatcher({T, kHidden}).with_dtype<bf16_t>().with_device(device).with_strides({S, 1}).verify(residual);
    }
    RuntimeCheck(S.unwrap() % kVec == 0, "tail rows must be 16-byte aligned");

    const int64_t rows = T.unwrap();
    RuntimeCheck(rows >= 1 && rows <= kMaxRows, "the all-gather read takes 1-", kMaxRows, " rows");
    RuntimeCheck(1 <= nvb && nvb <= kMaxNvb && nvb <= NB.unwrap(), "k3 attn-res read takes 1-8 bank rows");
    RuntimeCheck(!write_bank || nvb < NB.unwrap(), "bank write needs a free row");
    RuntimeCheck(bank.stride(0) % kVec == 0 && bank.stride(1) % kVec == 0, "bank rows must be 16-byte aligned");
    if (bank_partial) verify_bank_partial(bank_acc, bank_ms, T, device);

    Params params{
        .bank = static_cast<bf16_t*>(bank.data_ptr()),
        .cw = static_cast<const float4*>(cw.data_ptr()),
        .ow = static_cast<const float4*>(ow.data_ptr()),
        .prefix_out = static_cast<uint4*>(prefix_out.data_ptr()),
        .out = static_cast<uint4*>(out.data_ptr()),
        .scratch = {},
        .self_scratch = nullptr,
        .epochs = static_cast<int32_t*>(epochs.data_ptr()),
        .bank_stride_t = bank.stride(0),
        .bank_stride_b = bank.stride(1),
        .bank_acc = bank_partial ? static_cast<const float4*>(bank_acc.data_ptr()) : nullptr,
        .bank_ms = bank_partial ? static_cast<const float2*>(bank_ms.data_ptr()) : nullptr,
        .nvb_write = write_bank ? static_cast<int32_t>(nvb) : -1,
        .rank = 0,
        .score_eps = static_cast<float>(score_eps),
        .out_eps = static_cast<float>(out_eps),
        .tail_part = static_cast<const float*>(tail_part.data_ptr()),
        .tail_sq = static_cast<const float*>(tail_sq.data_ptr()),
        .tail_shared = static_cast<const bf16_t*>(tail_shared.data_ptr()),
        .tail_prefix = has_residual ? static_cast<const bf16_t*>(residual.data_ptr()) : nullptr,
        .tail_stride = S.unwrap(),
        .tail_eps = static_cast<float>(tail_eps),
    };
    TensorMatcher({kEpochs}).with_dtype<int32_t>().with_device(device).verify(epochs);
    auto* communicator = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(communicator->world_size_ == kWorld, "the all-gather read requires TP", kWorld);
    params.rank = communicator->rank_;
    for (int i = 0; i < kWorld; ++i) {
      params.scratch[i] =
          reinterpret_cast<uint4*>(reinterpret_cast<char*>(communicator->sg_.signals[i]) + sizeof(aiter::Signal));
    }
    params.self_scratch = params.scratch[communicator->rank_];
    const DLDevice dev = device.unwrap();
    const int n = static_cast<int>(rows);
    if (bank_partial) return launch<0>(params, has_residual, write_bank, n, dev);
    switch (nvb) {
      case 1:
        return launch<1>(params, has_residual, write_bank, n, dev);
      case 2:
        return launch<2>(params, has_residual, write_bank, n, dev);
      case 3:
        return launch<3>(params, has_residual, write_bank, n, dev);
      case 4:
        return launch<4>(params, has_residual, write_bank, n, dev);
      case 5:
        return launch<5>(params, has_residual, write_bank, n, dev);
      case 6:
        return launch<6>(params, has_residual, write_bank, n, dev);
      case 7:
        return launch<7>(params, has_residual, write_bank, n, dev);
      default:
        return launch<8>(params, has_residual, write_bank, n, dev);
    }
  }
};

template <int kNvb>
void launch_ar(const Params& params, bool has_residual, int rows, DLDevice device) {
  if (has_residual) {
    host::LaunchKernel(rows * kWorld, kThreads, device)(k3_ar_attn_res_kernel<kNvb, true>, params);
  } else {
    host::LaunchKernel(rows * kWorld, kThreads, device)(k3_ar_attn_res_kernel<kNvb, false>, params);
  }
}

struct K3AttnResArReadKernel {
  // The attention output's all-reduce over `comm` (as K3AttnResReadKernel) fused with the FFN-side
  // read: partial [T, H] is this rank's TP partial, residual [T, H] the pending residual when
  // has_residual, bank [T, NB, H] read over rows [0, nvb). prefix_out receives the stream.
  static void
  run(int64_t comm,
      const tvm::ffi::TensorView partial,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView prefix_out,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView ar_epochs,
      const tvm::ffi::TensorView ar_pushed,
      const tvm::ffi::TensorView bank_acc,
      const tvm::ffi::TensorView bank_ms,
      int64_t nvb,
      bool has_residual,
      bool bank_partial,
      double score_eps,
      double out_eps) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto NB = SymbolicSize{"num_blocks"};
    auto device = SymbolicDevice{};
    device.set_options<kDLROCM>();
    TensorMatcher({T, kHidden}).with_dtype<bf16_t>().with_device(device).verify(partial).verify(prefix_out).verify(out);
    if (has_residual) TensorMatcher({T, kHidden}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({T, NB, kHidden}).with_dtype<bf16_t>().with_device(device).with_strides({-1, -1, 1}).verify(bank);
    TensorMatcher({kHidden}).with_dtype<float>().with_device(device).verify(cw).verify(ow);
    TensorMatcher({kArMaxRows}).with_dtype<int32_t>().with_device(device).verify(ar_epochs).verify(ar_pushed);
    const int64_t rows = T.unwrap();
    RuntimeCheck(rows >= 1 && rows <= kArMaxRows, "the all-reduce read takes 1-", kArMaxRows, " rows");
    RuntimeCheck(1 <= nvb && nvb <= kMaxNvb && nvb <= NB.unwrap(), "k3 attn-res read takes 1-8 bank rows");
    RuntimeCheck(bank.stride(0) % kVec == 0 && bank.stride(1) % kVec == 0, "bank rows must be 16-byte aligned");
    if (bank_partial) verify_bank_partial(bank_acc, bank_ms, T, device);
    auto* communicator = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(communicator->world_size_ == kWorld, "the all-reduce read requires TP", kWorld);
    Params params{
        .bank = static_cast<bf16_t*>(bank.data_ptr()),
        .cw = static_cast<const float4*>(cw.data_ptr()),
        .ow = static_cast<const float4*>(ow.data_ptr()),
        .prefix_out = static_cast<uint4*>(prefix_out.data_ptr()),
        .out = static_cast<uint4*>(out.data_ptr()),
        .scratch = {},
        .self_scratch = nullptr,
        .epochs = nullptr,
        .bank_stride_t = bank.stride(0),
        .bank_stride_b = bank.stride(1),
        .bank_acc = bank_partial ? static_cast<const float4*>(bank_acc.data_ptr()) : nullptr,
        .bank_ms = bank_partial ? static_cast<const float2*>(bank_ms.data_ptr()) : nullptr,
        .nvb_write = -1,
        .rank = communicator->rank_,
        .score_eps = static_cast<float>(score_eps),
        .out_eps = static_cast<float>(out_eps),
        .partial = static_cast<const uint4*>(partial.data_ptr()),
        .residual = has_residual ? static_cast<const uint4*>(residual.data_ptr()) : nullptr,
        .ar_epochs = static_cast<int32_t*>(ar_epochs.data_ptr()),
        .ar_pushed = static_cast<int32_t*>(ar_pushed.data_ptr()),
    };
    for (int i = 0; i < kWorld; ++i) {
      params.scratch[i] =
          reinterpret_cast<uint4*>(reinterpret_cast<char*>(communicator->sg_.signals[i]) + sizeof(aiter::Signal));
    }
    params.self_scratch = params.scratch[communicator->rank_];
    const DLDevice dev = device.unwrap();
    const int n = static_cast<int>(rows);
    if (bank_partial) return launch_ar<0>(params, has_residual, n, dev);
    switch (nvb) {
      case 1:
        return launch_ar<1>(params, has_residual, n, dev);
      case 2:
        return launch_ar<2>(params, has_residual, n, dev);
      case 3:
        return launch_ar<3>(params, has_residual, n, dev);
      case 4:
        return launch_ar<4>(params, has_residual, n, dev);
      case 5:
        return launch_ar<5>(params, has_residual, n, dev);
      case 6:
        return launch_ar<6>(params, has_residual, n, dev);
      case 7:
        return launch_ar<7>(params, has_residual, n, dev);
      default:
        return launch_ar<8>(params, has_residual, n, dev);
    }
  }
};

}  // namespace k3_attn_res_read
}  // namespace sglang
