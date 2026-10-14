// K3 ROCm (TP8): a TP collective fused with the attention-residual aggregation
// that consumes it.
//
// Why not "collective, then one block per token": the aggregation itself is
// per-CU throughput bound at decode token counts (one block streams nvb + 1
// rows of 14 KB: 4.8 us at nvb 1, 7.5 us at nvb 8 for M = 8), so fusing it
// behind the collective's barrier saves only the launch (measured: no gain).
// The aggregation has to be spread over many CUs, which needs a cross-CTA
// reduction of the score / norm statistics. A cross-GPU barrier already sits
// in the middle of the collective, so the statistics ride it:
//
//   grid = T tokens x split blocks (split = ceil(H / 8 / 256) = 4 at H = 7168). Block (t, 0) of rank r is the PRODUCER
//   of column slice r (H / 8 columns) of token t:
//     kAR  start barrier (AITER per-block slot t), reduce slice r over the 8
//          peers' registered o_proj partials in AITER 2-stage order (fp32 sum
//          starting at the rank that owns the element in AITER's flat
//          partition), add the pending prefix -> the final row slice.
//     kAG  the slice is local: bf16(bf16(y_r + b) [+ c]) (AITER
//          all_gather_lastdim_add math on this rank's up_proj columns), no
//          start barrier (see "buffers" below).
//   It stores the bf16 slice and the slice's partial statistics
//     dot_i = <x_i, cw>, G_ij = <x_i, x_j>   (rows i, j in bank[0..nvb) + row)
//   (fp32, (nvb + 1) + (nvb + 1)(nvb + 2) / 2 values) in its own tmp area and
//   flags all 8 x split consumer blocks of token t (one remote store each).
//   Every block (t, k) is a CONSUMER of 1/split of token t's columns: it waits
//   for the 8 producers' flags, sums the 8 slices' statistics in rank order
//   (identical on every rank), takes score_i = dot_i / rms_i, the global-max
//   softmax, ||mix||^2 = w^T G w for the output RMSNorm, gathers its columns
//   of the row from the producers' tmp areas, mixes with its bank columns and
//   writes prefix_out / bank snapshot / out (normed) / E4M3 / pre-norm stream.
//
// Buffers: a private 1 MiB region at the tail of AITER's 2 x max_size tmp area
// (AITER kernels use its head), double-buffered by launch parity (data,
// statistics, flags). Flags are reset by their consumer. Launch n writes
// parity n % 2 only after every rank has started launch n - 1 (its consumers
// waited for all ranks' producers), so every reader of launch n - 2 is done;
// kAR additionally starts with AITER's start barrier (its peers' inputs). A
// TP collective between two consecutive launches with equal parity (graph
// replay boundary) is required and always present in K3 (MoE all-reduce).
// No end barrier anywhere: kAR inputs are read before the producer signals;
// kAG reads no peer input at all (no registration, no keep-alive).
//
// Numerics: prefix_out (the AR / add3 row) and the bank snapshot are
// bit-identical to the unfused path. Scores and the norm use the same fp32
// formulas with a different reduction order, ||mix||^2 via the Gram form and
// hardware rsq / exp2 / rcp, so out / E4M3 / stream match attn_res_smallm up
// to rare 1-ulp bf16 flips (~1e-4 of elements). Every rank sums the 8
// slices' statistics in rank order: outputs are identical on all ranks.
//
// Latency notes (8x MI355X, M = 8): consumers poll their flags with relaxed
// loads and take one acquire fence after (acquire-per-spin invalidates caches
// every iteration: 2-3x slower at M = 64); local operands are loaded before
// the kAR start barrier (computing on them before the wait was slower); the
// weights use packed / hardware-approximate math (a wave64 IEEE div / sqrt /
// exp chain over nvb + 1 rows cost ~0.8 us).
#pragma once

#ifndef USE_ROCM
#error "ar_agg_hip.cuh is ROCm only"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER: CustomAllreduce, RankData, Signal

namespace sglang {
namespace k3_ar_agg {

constexpr int kNG = 8;
constexpr int kVec = 8;
constexpr int kThreads = 256;
constexpr int kMaxT = aiter::kMaxBlocks;  // start-barrier slots
constexpr int64_t kRegionBytes = 1 << 20;
constexpr int64_t kDataOff = 0, kDataPar = 256 << 10;         // [T][slice] packs
constexpr int64_t kStatOff = 512 << 10, kStatPar = 64 << 10;  // [T][64] fp32
constexpr int64_t kFlagOff = 768 << 10, kFlagPar = 64 << 10;  // [T * split][8] u32
constexpr int kStatStride = 64;

using vec_t = device::AlignedVector<bf16_t, kVec>;
using OP = opus::vector_t<opus::bf16_t, kVec>;
using OA = opus::vector_t<opus::fp32_t, kVec>;

enum Mode : int { kAR = 0, kAG = 1 };

struct Params {
  aiter::RankData* dp;  // kAR: peer addresses of the registered partials
  aiter::RankSignals sg;
  aiter::Signal* self_sg;
  int rank, T, split, parity;
  int64_t region_off;                 // byte offset of the private region in the tmp area
  const bf16_t* __restrict__ y;       // kAG: this rank's [T, H/8] slice (contiguous)
  const bf16_t* __restrict__ prefix;  // kAR: optional pending prefix [T, H]
  const bf16_t* __restrict__ add_b;   // kAG [T, H]
  const bf16_t* __restrict__ add_c;   // kAG optional [T, H]
  bf16_t* __restrict__ prefix_out;    // [T, H]
  bf16_t* __restrict__ bank;          // [T, NB, H]
  const float* __restrict__ cw;       // [H]
  const bf16_t* __restrict__ ow;      // [H] or null
  bf16_t* __restrict__ out;           // [T, H]
  uint8_t* __restrict__ out8;         // [T, H] or null
  bf16_t* __restrict__ stream;        // [T, H] or null
  int64_t stride_p, stride_b, stride_c, stride_po, stride_bm, stride_bb, stride_o, stride_o8, stride_s;
  int32_t hidden, n_vecs, slice_vecs, chunk_vecs;
  int32_t write_bank;
  float score_eps, out_eps;
};

__device__ __forceinline__ char* region(const Params& p, int q) {
  return reinterpret_cast<char*>(aiter::get_tmp_buf<char>(p.sg.signals[q])) + p.region_off;
}

// AITER start_sync on an explicit slot (only the producer blocks take part),
// split into arrive (post this rank's flag) and wait (all peers posted) so
// local work can overlap the barrier.
__device__ __forceinline__ uint32_t start_arrive(const Params& p, int slot) {
  const uint32_t flag = p.self_sg->_flag[slot] + 1;
  if (threadIdx.x < kNG)
    __scoped_atomic_store_n(
        &p.sg.signals[threadIdx.x]->start[slot][p.rank], flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  return flag;
}

__device__ __forceinline__ void start_wait(const Params& p, int slot, uint32_t flag) {
  if (threadIdx.x < kNG) {
    while (__scoped_atomic_load_n(&p.self_sg->start[slot][threadIdx.x], __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < flag)
      ;
  }
  __syncthreads();
  if (threadIdx.x == 0) p.self_sg->_flag[slot] = flag;
}

__device__ __forceinline__ uint8_t to_e4m3(float x) {
  return static_cast<uint8_t>(__hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3));
}

typedef float f2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ void unpack2(const vec_t& v, f2_t (&f)[kVec / 2]) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i)
    f[i] = f2_t{__uint_as_float(u[i] << 16), __uint_as_float(u[i] & 0xffff0000u)};
}

__device__ __forceinline__ void unpack(const vec_t& v, float (&f)[kVec]) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) {
    f[2 * i] = __uint_as_float(u[i] << 16);
    f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
  }
}

template <int kMode, int kNVB>
__global__ void __launch_bounds__(kThreads) ar_agg_kernel(const Params __grid_constant__ p) {
  using namespace device;
  constexpr int kR = kNVB + 1;  // bank rows + the row (index kNVB)
  constexpr int kG = kR * (kR + 1) / 2;
  constexpr int kS = kR + kG;  // dot_i, then G_ij (i <= j, row-major)
  static_assert(kS <= kStatStride, "statistics layout");
  __shared__ float red[kS * 128];
  constexpr int kS4 = (kS + 3) / 4;
  __shared__ __attribute__((aligned(16))) float stat[kS4 * 4];

  const int t = blockIdx.x / p.split;
  const int k = blockIdx.x - t * p.split;
  const int sv = p.slice_vecs;
  const int64_t dpar = kDataOff + p.parity * kDataPar;
  const int64_t spar = kStatOff + p.parity * kStatPar;
  const int64_t fpar = kFlagOff + p.parity * kFlagPar;
  const bf16_t* bank_t = p.bank + static_cast<int64_t>(t) * p.stride_bm;

  // consumer columns: local operands are loaded up front (their latency hides
  // behind the producer work / the cross-GPU wait)
  const int v = k * p.chunk_vecs + threadIdx.x;
  const bool act = threadIdx.x < p.chunk_vecs && v < p.n_vecs;
  vec_t rows[kR];
  vec_t owv;
  if (act) {
#pragma unroll
    for (int r = 0; r < kNVB; ++r) {
      rows[r].load(bank_t + r * p.stride_bb, v);
    }
    if (p.ow != nullptr) owv.load(p.ow, v);
  }

  // ======================= producer: slice `rank` of token t =======================
  if (k == 0) {
    char* mine = region(p, p.rank);
    const int i = threadIdx.x;
    const bool pact = i < sv;
    const int pv_idx = p.rank * sv + i;
    // local operands first (before the start barrier)
    vec_t bk[kNVB];
    float4 cw0, cw1;
    vec_t av, bv, cv;
    if (pact) {
#pragma unroll
      for (int r = 0; r < kNVB; ++r) {
        bk[r].load(bank_t + r * p.stride_bb, pv_idx);
      }
      cw0 = reinterpret_cast<const float4*>(p.cw)[2 * pv_idx];
      cw1 = reinterpret_cast<const float4*>(p.cw)[2 * pv_idx + 1];
      if constexpr (kMode == kAR) {
        if (p.prefix != nullptr) av.load(p.prefix + t * p.stride_p, pv_idx);
      } else {
        av.load(p.y + static_cast<int64_t>(t) * sv * kVec, i);
        bv.load(p.add_b + t * p.stride_b, pv_idx);
        if (p.add_c != nullptr) cv.load(p.add_c + t * p.stride_c, pv_idx);
      }
    }
    // kAR: post this rank's start flag now, wait after the local loads were
    // issued (the bank / cw / prefix loads above overlap the barrier)
    uint32_t sflag = 0;
    if constexpr (kMode == kAR) sflag = start_arrive(p, t);
    if constexpr (kMode == kAR) start_wait(p, t, sflag);
    // kAR: issue the peer loads first; the bank statistics below overlap them
    OP xin[kNG];
    int flat = 0;
    if constexpr (kMode == kAR) {
      if (pact) {
        flat = t * p.n_vecs + pv_idx;
        const int owner = flat / (p.T * sv);  // AITER 2-stage flat partition
#pragma unroll
        for (int q = 0; q < kNG; ++q)
          xin[q] = reinterpret_cast<const OP*>(p.dp->ptrs[(owner + q) & (kNG - 1)])[flat];
      }
    }
    // packed fp32 math (v_pk_fma_f32): element pairs
    f2_t pp[kS];
#pragma unroll
    for (int s = 0; s < kS; ++s)
      pp[s] = f2_t{0.f, 0.f};
    f2_t x[kR][kVec / 2];
    f2_t cwv[kVec / 2];
    if (pact) {
#pragma unroll
      for (int r = 0; r < kNVB; ++r)
        unpack2(bk[r], x[r]);
      cwv[0] = f2_t{cw0.x, cw0.y};
      cwv[1] = f2_t{cw0.z, cw0.w};
      cwv[2] = f2_t{cw1.x, cw1.y};
      cwv[3] = f2_t{cw1.z, cw1.w};
#pragma unroll
      for (int e = 0; e < kVec / 2; ++e) {
#pragma unroll
        for (int a = 0; a < kNVB; ++a)
          pp[a] = x[a][e] * cwv[e] + pp[a];
        int idx = kR;
#pragma unroll
        for (int a = 0; a < kR; ++a)
#pragma unroll
          for (int b = a; b < kR; ++b) {
            if (b < kNVB) pp[idx] = x[a][e] * x[b][e] + pp[idx];
            ++idx;
          }
      }
    }
    if (pact) {
      vec_t row;
      if constexpr (kMode == kAR) {
        OA acc = aiter::upcast(xin[0]);
#pragma unroll
        for (int q = 1; q < kNG; ++q) {
#pragma unroll
          for (int e = 0; e < kVec; ++e)
            acc[e] += aiter::upcast_s(xin[q][e]);
        }
        const OP ar = aiter::downcast<OP>(acc);
        row = *reinterpret_cast<const vec_t*>(&ar);
        if (p.prefix != nullptr) {
          // attn_res_smallm: bf16(prefix + addend), addend = the AR result
#pragma unroll
          for (int e = 0; e < kVec; ++e)
            row[e] = cast<bf16_t>(cast<fp32_t>(av[e]) + cast<fp32_t>(row[e]));
        }
      } else {
        // AITER all_gather_lastdim_add: bf16(bf16(y + b) [+ c])
        OA acc = aiter::upcast(*reinterpret_cast<const OP*>(&av));
        aiter::packed_assign_add<opus::fp32_t, kVec>(acc, aiter::upcast(*reinterpret_cast<const OP*>(&bv)));
        OP o = aiter::downcast<OP>(acc);
        if (p.add_c != nullptr) {
          OA acc2 = aiter::upcast(o);
          aiter::packed_assign_add<opus::fp32_t, kVec>(acc2, aiter::upcast(*reinterpret_cast<const OP*>(&cv)));
          o = aiter::downcast<OP>(acc2);
        }
        row = *reinterpret_cast<const vec_t*>(&o);
      }
      row.store(mine + dpar, static_cast<int64_t>(t) * sv + i);

      // row partials: <row, cw>, <bank_a, row>, <row, row>
      unpack2(row, x[kNVB]);
#pragma unroll
      for (int e = 0; e < kVec / 2; ++e) {
        pp[kNVB] = x[kNVB][e] * cwv[e] + pp[kNVB];
        int idx = kR;
#pragma unroll
        for (int a = 0; a < kR; ++a) {
          idx += kNVB - a;  // (a, kNVB) is the last pair of row a
          pp[idx] = x[a][e] * x[kNVB][e] + pp[idx];
          ++idx;
        }
      }
    }
    float part[kS];
#pragma unroll
    for (int s = 0; s < kS; ++s)
      part[s] = pp[s].x + pp[s].y;
    // block reduction of the kS partials over the first kRedT threads (the
    // slice has sv <= kRedT vectors): LDS transpose, 4 lanes per value
    constexpr int kRedT = 128, kLanes = 4;
    static_assert(kS * kLanes <= kThreads, "reduction layout");
    if (threadIdx.x < kRedT) {
#pragma unroll
      for (int s = 0; s < kS; ++s)
        red[s * kRedT + threadIdx.x] = part[s];
    }
    __syncthreads();
    {
      const int s = threadIdx.x / kLanes, l = threadIdx.x % kLanes;
      float sum = 0.f;
      if (s < kS) {
#pragma unroll
        for (int j = 0; j < kRedT / kLanes; ++j)
          sum += red[s * kRedT + j * kLanes + l];
      }
      sum += __shfl_xor(sum, 1, 64);
      sum += __shfl_xor(sum, 2, 64);
      if (s < kS && l == 0) reinterpret_cast<float*>(mine + spar)[t * kStatStride + s] = sum;
    }
    __syncthreads();  // all slice / statistics stores issued and waited for
    if (threadIdx.x < kNG * p.split) {
      const int q = threadIdx.x / p.split, kk = threadIdx.x - q * p.split;
      uint32_t* f = reinterpret_cast<uint32_t*>(region(p, q) + fpar) + (t * p.split + kk) * kNG + p.rank;
      __scoped_atomic_store_n(f, 1u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
  }

  // ======================= consumer: columns [k*chunk, (k+1)*chunk) of token t =======================
  {
    uint32_t* f = reinterpret_cast<uint32_t*>(region(p, p.rank) + fpar) + blockIdx.x * kNG;
    if (threadIdx.x < kNG) {
      // relaxed polling (an acquire load per spin invalidates the caches every
      // iteration and slows every other block down), one acquire fence after
      while (__scoped_atomic_load_n(f + threadIdx.x, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) == 0u)
        ;
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      __scoped_atomic_store_n(f + threadIdx.x, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }
    __syncthreads();
  }
  // statistics (rank-ordered sum: identical on every rank) and this block's
  // row columns; wave 0 turns the statistics into the mixing weights and the
  // output-norm scale with one value per lane (lane s holds statistic s)
  if (act) {
    const int q = v / sv;
    rows[kNVB].load(region(p, q) + dpar, static_cast<int64_t>(t) * sv + (v - q * sv));
  }
  if (threadIdx.x < kS) {
    float sv8[kNG];
#pragma unroll
    for (int q = 0; q < kNG; ++q)
      sv8[q] = reinterpret_cast<const float*>(region(p, q) + spar)[t * kStatStride + threadIdx.x];
    float st = 0.f;
#pragma unroll
    for (int q = 0; q < kNG; ++q)
      st += sv8[q];
    stat[threadIdx.x] = st;
  }
  __syncthreads();

  // scores, global-max softmax, ||mix||^2 = w^T G w: redundant per thread,
  // from registers (one vectorized LDS read of all statistics)
  float sr[kS4 * 4];
#pragma unroll
  for (int i = 0; i < kS4; ++i) {
    const float4 f4 = reinterpret_cast<const float4*>(stat)[i];
    sr[4 * i] = f4.x;
    sr[4 * i + 1] = f4.y;
    sr[4 * i + 2] = f4.z;
    sr[4 * i + 3] = f4.w;
  }
  const float inv_h = 1.0f / static_cast<float>(p.hidden);
  float w[kR];
  {
    float score[kR];
    int d = kR;
#pragma unroll
    for (int a = 0; a < kR; ++a) {
      score[a] = sr[a] * __builtin_amdgcn_rsqf(sr[d] * inv_h + p.score_eps);  // G_aa
      d += kR - a;
    }
    float m = score[kNVB];
#pragma unroll
    for (int r = 0; r < kNVB; ++r)
      m = fmaxf(m, score[r]);
    float den = 0.f;
#pragma unroll
    for (int r = 0; r < kNVB; ++r) {
      w[r] = __builtin_amdgcn_exp2f((score[r] - m) * 1.4426950408889634f);
      den += w[r];
    }
    w[kNVB] = __builtin_amdgcn_exp2f((score[kNVB] - m) * 1.4426950408889634f);
    const float inv = __builtin_amdgcn_rcpf(den + w[kNVB]);
#pragma unroll
    for (int r = 0; r < kR; ++r)
      w[r] *= inv;
  }
  float scale = 1.f;
  if (p.ow != nullptr) {
    float asq = 0.f;
    int idx = kR;
#pragma unroll
    for (int a = 0; a < kR; ++a) {
      asq += w[a] * w[a] * sr[idx++];
      float cross = 0.f;
#pragma unroll
      for (int b = a + 1; b < kR; ++b)
        cross += w[b] * sr[idx++];
      asq += 2.f * w[a] * cross;
    }
    scale = __builtin_amdgcn_rsqf(fmaxf(asq, 0.f) * inv_h + p.out_eps);
  }
  if (!act) return;

  rows[kNVB].store(p.prefix_out + t * p.stride_po, v);
  if (p.write_bank) rows[kNVB].store(p.bank + t * p.stride_bm + kNVB * p.stride_bb, v);

  float acc[kVec];
  {
    float xb[kVec];
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      acc[e] = 0.f;
#pragma unroll
    for (int r = 0; r < kNVB; ++r) {
      unpack(rows[r], xb);
#pragma unroll
      for (int e = 0; e < kVec; ++e)
        acc[e] = w[r] * xb[e] + acc[e];
    }
    unpack(rows[kNVB], xb);
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      acc[e] = xb[e] * w[kNVB] + acc[e];
  }
  if (p.stream != nullptr) {
    vec_t o;
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      o[e] = cast<bf16_t>(acc[e]);
    o.store(p.stream + t * p.stride_s, v);
  }
  vec_t o;
  if (p.ow != nullptr) {
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      o[e] = cast<bf16_t>(acc[e] * scale * cast<fp32_t>(owv[e]));
  } else {
#pragma unroll
    for (int e = 0; e < kVec; ++e)
      o[e] = cast<bf16_t>(acc[e]);
  }
  o.store(p.out + t * p.stride_o, v);
  if (p.out8 != nullptr) {
    uint32_t q8[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t packed = 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = fminf(fmaxf(cast<fp32_t>(o[h * 4 + e]), -448.f), 448.f);
        packed |= static_cast<uint32_t>(to_e4m3(x)) << (8 * e);
      }
      q8[h] = packed;
    }
    *reinterpret_cast<uint2*>(p.out8 + t * p.stride_o8 + v * kVec) = make_uint2(q8[0], q8[1]);
  }
}

struct ArAgg {
  template <int kMode, int kNVB>
  static void launch(const Params& params, hipStream_t stream) {
    hipLaunchKernelGGL((ar_agg_kernel<kMode, kNVB>), dim3(params.T * params.split), dim3(kThreads), 0, stream, params);
  }

  template <int kMode>
  static void dispatch(const Params& params, int64_t nvb, hipStream_t stream) {
    switch (nvb) {
      case 1:
        launch<kMode, 1>(params, stream);
        break;
      case 2:
        launch<kMode, 2>(params, stream);
        break;
      case 3:
        launch<kMode, 3>(params, stream);
        break;
      case 4:
        launch<kMode, 4>(params, stream);
        break;
      case 5:
        launch<kMode, 5>(params, stream);
        break;
      case 6:
        launch<kMode, 6>(params, stream);
        break;
      case 7:
        launch<kMode, 7>(params, stream);
        break;
      case 8:
        launch<kMode, 8>(params, stream);
        break;
      default:
        host::RuntimeCheck(false, "nvb must be in [1, 8]");
    }
  }

  // mode 0 (AR): x = this rank's TP-partial [T, H]; a = optional pending
  //   prefix [T, H]; b, c unused.
  // mode 1 (AG): x = this rank's up_proj column slice y [T, H/8]; b = add_b
  //   [T, H]; c = optional add_c [T, H]; a unused.
  // flags: bit0 a, bit1 c, bit2 write_bank, bit3 out norm, bit4 out8, bit5 stream.
  // reg_inp_ptr != 0 (AR, eager / copy-in capture): x is first staged into
  // AITER's registered input pool at that address, as AITER's all_reduce does.
  static void
  run(int64_t comm,
      int64_t mode,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView c,
      const tvm::ffi::TensorView prefix_out,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView out8,
      const tvm::ffi::TensorView stream_out,
      int64_t nvb,
      int64_t flags,
      double score_eps,
      double out_eps,
      int64_t reg_inp_ptr,
      int64_t reg_inp_bytes,
      int64_t region_off,
      int64_t parity,
      int64_t split) {
    using namespace host;
    auto* ca = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(ca->world_size_ == kNG, "ar_agg is TP8 only");
    const int64_t T = prefix_out.size(0);
    const int64_t H = prefix_out.size(1);
    RuntimeCheck(T >= 1 && T <= kMaxT, "ar_agg: 1 <= T <= 80");
    RuntimeCheck(H % (kVec * kNG) == 0, "ar_agg: H must be a multiple of 64");
    const int64_t n_vecs = H / kVec, sv = n_vecs / kNG;
    const int64_t chunk = (n_vecs + split - 1) / split;
    RuntimeCheck(sv <= 128 && chunk <= kThreads, "ar_agg: H too large");
    RuntimeCheck(
        T * sv * 16 <= kDataPar && T * kStatStride * 4 <= kStatPar && T * split * kNG * 4 <= kFlagPar,
        "ar_agg: region too small");
    RuntimeCheck(kNG * split <= kThreads, "ar_agg: split too large");
    RuntimeCheck(
        prefix_out.stride(1) == 1 && bank.stride(2) == 1 && out.stride(1) == 1, "inner dim must be contiguous");
    RuntimeCheck(x.size(0) == T && x.stride(1) == 1, "x shape");
    if (mode == kAR)
      RuntimeCheck(x.size(1) == H && x.stride(0) == H, "AR input must be a contiguous [T, H]");
    else
      RuntimeCheck(x.size(1) == H / kNG && x.stride(0) == H / kNG, "AG input must be a contiguous [T, H/8]");
    const bool write_bank = flags & 4;
    RuntimeCheck(bank.size(1) > nvb || !write_bank, "bank has no free row for the snapshot");

    const auto stream = LaunchKernel::resolve_device(prefix_out.device());
    Params p{};
    if (mode == kAR) {
      void* inp = x.data_ptr();
      if (reg_inp_ptr != 0) {
        const int64_t bytes = T * H * 2;
        RuntimeCheck(bytes <= reg_inp_bytes, "registered buffer too small");
        RuntimeDeviceCheck(
            hipMemcpyAsync(reinterpret_cast<void*>(reg_inp_ptr), inp, bytes, hipMemcpyDeviceToDevice, stream));
        inp = reinterpret_cast<void*>(reg_inp_ptr);
      }
      p.dp = ca->get_buffer_RD(stream, inp);
      if (flags & 1) {
        p.prefix = static_cast<const bf16_t*>(a.data_ptr());
        p.stride_p = a.stride(0);
      }
    } else {
      p.y = static_cast<const bf16_t*>(x.data_ptr());
      p.add_b = static_cast<const bf16_t*>(b.data_ptr());
      p.stride_b = b.stride(0);
      if (flags & 2) {
        p.add_c = static_cast<const bf16_t*>(c.data_ptr());
        p.stride_c = c.stride(0);
      }
    }
    p.sg = ca->sg_;
    p.self_sg = ca->self_sg_;
    p.rank = ca->rank_;
    p.T = static_cast<int>(T);
    p.split = static_cast<int>(split);
    p.parity = static_cast<int>(parity & 1);
    p.region_off = region_off;
    p.prefix_out = static_cast<bf16_t*>(prefix_out.data_ptr());
    p.stride_po = prefix_out.stride(0);
    p.bank = static_cast<bf16_t*>(bank.data_ptr());
    p.stride_bm = bank.stride(0);
    p.stride_bb = bank.stride(1);
    p.cw = static_cast<const float*>(cw.data_ptr());
    if (flags & 8) p.ow = static_cast<const bf16_t*>(ow.data_ptr());
    p.out = static_cast<bf16_t*>(out.data_ptr());
    p.stride_o = out.stride(0);
    if (flags & 16) {
      p.out8 = static_cast<uint8_t*>(out8.data_ptr());
      p.stride_o8 = out8.stride(0);
    }
    if (flags & 32) {
      p.stream = static_cast<bf16_t*>(stream_out.data_ptr());
      p.stride_s = stream_out.stride(0);
    }
    p.hidden = static_cast<int32_t>(H);
    p.n_vecs = static_cast<int32_t>(n_vecs);
    p.slice_vecs = static_cast<int32_t>(sv);
    p.chunk_vecs = static_cast<int32_t>(chunk);
    p.write_bank = write_bank ? 1 : 0;
    p.score_eps = static_cast<float>(score_eps);
    p.out_eps = static_cast<float>(out_eps);
    if (mode == kAR)
      dispatch<kAR>(p, nvb, stream);
    else
      dispatch<kAG>(p, nvb, stream);
    RuntimeDeviceCheck(hipGetLastError());
  }
};

}  // namespace k3_ar_agg
}  // namespace sglang
