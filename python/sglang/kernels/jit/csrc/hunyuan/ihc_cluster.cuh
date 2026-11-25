// Hy4 iHC post + pre + RMSNorm for small token counts, one token row per thread-block cluster.
//
// Per token (bf16 roundings where the unfused model path stores bf16):
//   o    = bf16(out * rsqrt(mean(out^2) + out_eps) * w_out)       (kOutNorm; else out as is)
//   x_m  = bf16(post_m * o + residual_m)                           (kHasOut; else x = residual)
//   mix  = (x . hc_fn^T) * rsqrt(mean(x^2) + norm_eps)
//   pre  = sigmoid(mix[:HC] * scale[0] + base[:HC]) + hc_eps
//   post = magnitude * sigmoid(mix[HC:] * scale[1] + base[HC:]) + hc_eps
//   r    = bf16(sum_m pre_m * x_m);  y = bf16(r * rsqrt(mean(r^2) + rms_eps) * w)
//   kQuant: y also as fp8 e4m3 with one fp32 scale per 128 columns, bit-identical to per_token_group_quant.
//
// Grid (kSplit, ceil(T / kTB)), cluster (kSplit, 1, 1): cluster y handles tokens [y * kTB, y * kTB + kTB) and rank r
// owns the 128-column groups [r * kGPC, (r + 1) * kGPC) of every stream. A 16-lane half-warp owns one group of one
// stream, 8 columns per lane. Both row reductions (sum x^2 with the hc_fn dots, then sum r^2) are exchanged through
// distributed shared memory: every CTA sends its per-group partials to every rank with st.async, and every CTA sums
// all kG group partials with the same fixed tree. A group's partial does not depend on kSplit or kTB, so a row's
// result is bitwise independent of the launch configuration and of the other tokens in the batch.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/ptx/barrier.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

namespace device::hy4_ihc::ptx {

SGL_DEVICE void st_async(uint32_t remote_addr, float x, uint32_t remote_bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(remote_addr),
               "f"(x),
               "r"(remote_bar)
               : "memory");
}

SGL_DEVICE void st_async(uint32_t remote_addr, float4 v, uint32_t remote_bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(remote_addr),
      "f"(v.x),
      "f"(v.y),
      "f"(v.z),
      "f"(v.w),
      "r"(remote_bar)
      : "memory");
}

// Phase 0 of an mbarrier that peer CTAs complete with st.async: a relaxed cluster-scope wait plus an acquire fence
// restricted to shared::cluster, which orders the peers' writes without invalidating L1.
SGL_DEVICE void mbar_wait_peers(uint64_t* bar) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 p, [%0], 0;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(device::ptx::to_shared(bar))
      : "memory");
  asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
}

// per_token_group_quant is built with --use_fast_math; these reproduce its two scale instructions.
SGL_DEVICE float div_approx_ftz(float a, float b) {
  float r;
  asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

SGL_DEVICE float mul_ftz(float a, float b) {
  float r;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

}  // namespace device::hy4_ihc::ptx

namespace device::hy4_ihc {

inline constexpr int kGroup = 128;  // columns per half-warp task and per fp8 scale
inline constexpr int kLanes = 16;
inline constexpr int kVec = 8;  // bf16 columns per lane
using V = AlignedVector<bf16x2_t, kVec / 2>;

SGL_DEVICE float bf16_round(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

SGL_DEVICE void unpack(const V& v, float (&f)[kVec]) {
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k) {
    const auto pair = cast<fp32x2_t>(v[k]);
    f[2 * k] = pair.x;
    f[2 * k + 1] = pair.y;
  }
}

SGL_DEVICE V pack(const float (&f)[kVec]) {
  V v;
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k)
    v[k] = cast<bf16x2_t>(fp32x2_t{f[2 * k], f[2 * k + 1]});
  return v;
}

// Pairwise sum of p[0], p[stride], ..., p[(N - 1) * stride] in a shape fixed by N alone.
template <int N>
SGL_DEVICE float tree_sum(const float* p, int stride) {
  if constexpr (N == 1) {
    return p[0];
  } else {
    constexpr int kHalf = N / 2;
    return __fadd_rn(tree_sum<kHalf>(p, stride), tree_sum<N - kHalf>(p + kHalf * stride, stride));
  }
}

// Only this thread's half-warp: the other half may take a different branch.
SGL_DEVICE uint32_t halfwarp_mask() {
  return 0xffffu << (threadIdx.x & kLanes);
}

SGL_DEVICE float halfwarp_sum(float x) {
#pragma unroll
  for (int o = kLanes / 2; o > 0; o /= 2)
    x = __fadd_rn(x, __shfl_xor_sync(halfwarp_mask(), x, o));
  return x;
}

SGL_DEVICE float halfwarp_max(float x) {
#pragma unroll
  for (int o = kLanes / 2; o > 0; o /= 2)
    x = fmaxf(x, __shfl_xor_sync(halfwarp_mask(), x, o));
  return x;
}

template <int kOffset, int N>
SGL_DEVICE void halfwarp_reduce_scatter_step(float (&v)[N], bool upper) {
#pragma unroll
  for (int i = 0; i < N; i += kLanes) {
#pragma unroll
    for (int k = 0; k < kOffset; ++k) {
      const float lo = v[i + k];
      const float hi = v[i + k + kOffset];
      v[i + k] = __fadd_rn(upper ? hi : lo, __shfl_xor_sync(halfwarp_mask(), upper ? lo : hi, kOffset));
    }
  }
}

// Sums v over the 16 lanes of a half-warp, scattered: afterwards v[16 * i] of lane l holds the total of
// v[16 * i + l]. 15 shuffles per 16 values instead of 64 for a butterfly of each.
template <int N>
SGL_DEVICE void halfwarp_reduce_scatter(float (&v)[N], uint32_t lane) {
  static_assert(N % kLanes == 0);
  halfwarp_reduce_scatter_step<8>(v, lane & 8);
  halfwarp_reduce_scatter_step<4>(v, lane & 4);
  halfwarp_reduce_scatter_step<2>(v, lane & 2);
  halfwarp_reduce_scatter_step<1>(v, lane & 1);
}

}  // namespace device::hy4_ihc

struct Hy4IhcClusterParams {
  const bf16_t* residual;         // [T, HC, H]
  const bf16_t* out;              // [T, H] (kHasOut)
  const float* post;              // [T, HC] (kHasOut)
  const bf16_t* out_norm_weight;  // [H] (kOutNorm)
  const float* hc_fn;             // [2 HC, HC H]
  const float* hc_scale;          // [2]
  const float* hc_base;           // [2 HC]
  const bf16_t* rms_weight;       // [H]
  bf16_t* new_residual;           // [T, HC, H] (kHasOut)
  bf16_t* y;                      // [T, H]
  float* next_post;               // [T, HC]
  fp8_e4m3_t* y_q;                // [T, H] (kQuant)
  float* y_s;                     // [T, H / 128] at the strides below (kQuant)
  int64_t y_s_token_stride;
  int64_t y_s_group_stride;
  uint32_t num_tokens;
  float out_eps, norm_eps, hc_eps, rms_eps, magnitude;
};

template <int kHC, int kH, int kSplit, int kTB>
struct Hy4IhcClusterShape {
  static constexpr int kG = kH / device::hy4_ihc::kGroup;  // groups per stream
  static constexpr int kGPC = kG / kSplit;                 // groups per CTA per stream
  static constexpr int kHalfWarps = kHC * kGPC;
  static constexpr int kThreads = kHalfWarps * device::hy4_ihc::kLanes;
  static constexpr int kNG = 2 * kHC;               // hc_fn dot products
  static constexpr int kNP = kNG + 1;               // and the sum of squares
  static constexpr int kNQ = (kNP + 3) / 4 * 4;     // exchanged record, whole float4s
  static constexpr int kNV = (kNP + 15) / 16 * 16;  // reduce-scatter width
  static constexpr int kTasks2 = kTB * kGPC;        // (token, group) tasks of the combine
  static constexpr int kIters2 = (kTasks2 + kHalfWarps - 1) / kHalfWarps;
  static constexpr int kTasksOut = kTB * kG;  // (token, group) tasks of the out-norm (whole row in every CTA)
  static constexpr int kItersOut = (kTasksOut + kHalfWarps - 1) / kHalfWarps;
  // The hc_fn slice stays in registers when it fits (kNG * 8 floats per lane); larger hc_mult reloads it per use.
  static constexpr bool kPreloadFn = kNG <= 8;

  struct Smem {
    alignas(16) bf16_t x[kTB][kHC][kGPC * device::hy4_ihc::kGroup];
    alignas(16) float part[kTB][kHC][kGPC][kNQ];
    alignas(16) float pool1[kTB][kG][kNQ];
    float pool2[kTB][kG];
    float out_part[kTB][kG];
    float inv_out[kTB];
    float pre[kTB][kHC];
    uint64_t bar[2];
  };

  static constexpr bool kValid = kH % device::hy4_ihc::kGroup == 0 && kG % kSplit == 0 && kThreads % 32 == 0 &&
                                 kThreads <= 1024 && sizeof(Smem) <= 48 * 1024 && kSplit >= 1 && kSplit <= 8;
};

template <int kHC, int kH, int kSplit, int kTB, bool kHasOut, bool kOutNorm, bool kQuant, bool kUsePDL>
__global__ __launch_bounds__(Hy4IhcClusterShape<kHC, kH, kSplit, kTB>::kThreads, 1) void hy4_ihc_cluster_kernel(
    const __grid_constant__ Hy4IhcClusterParams p) {
  using namespace device;
  using namespace device::hy4_ihc;
  using S = Hy4IhcClusterShape<kHC, kH, kSplit, kTB>;
  static_assert(S::kValid);
  static_assert(kHasOut || !kOutNorm);
  constexpr int kG = S::kG, kGPC = S::kGPC, kNG = S::kNG, kNP = S::kNP, kNQ = S::kNQ;
  constexpr int64_t kK = int64_t{kHC} * kH;
  __shared__ typename S::Smem s;

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % kLanes;
  const uint32_t hw = tid / kLanes;
  const uint32_t rank = blockIdx.x;
  const uint32_t first_token = blockIdx.y * kTB;

  if (tid == 0) {
    device::ptx::mbar_init(&s.bar[0], 1);
    device::ptx::mbar_init(&s.bar[1], 1);
    device::ptx::mbar_arrive_expect_tx(&s.bar[0], kTB * kG * kNQ * sizeof(float));
    device::ptx::mbar_arrive_expect_tx(&s.bar[1], kTB * kG * sizeof(float));
    device::ptx::fence_barrier_init();
  }
  device::ptx::bar_cluster_arrive_relaxed();

  // Tail tokens recompute the last row and store nothing; every CTA still sends all of its partials.
  const auto token = [&](int b) -> int64_t { return min(first_token + b, p.num_tokens - 1); };
  const auto valid = [&](int b) { return first_token + b < p.num_tokens; };

  // Combine-and-reduce ownership: stream m, group (rank * kGPC + gi), 8 columns from col.
  const int m = hw / kGPC;
  const int gi = hw % kGPC;
  const int vid = (rank * kGPC + gi) * (kGroup / kVec) + lane;  // V index within a stream row

  // Parameters only, nothing the previous kernel writes: loaded before the PDL wait.
  float4 fn[S::kPreloadFn ? kNG : 1][2];
  const auto fn_ptr = [&](int n) { return reinterpret_cast<const float4*>(p.hc_fn + n * kK + m * kH) + vid * 2; };
  if constexpr (S::kPreloadFn) {
#pragma unroll
    for (int n = 0; n < kNG; ++n) {
      fn[n][0] = __ldg(fn_ptr(n));
      fn[n][1] = __ldg(fn_ptr(n) + 1);
    }
  }
  V w_out;
  if constexpr (kOutNorm) w_out.load(p.out_norm_weight, vid);
  V w_rms[S::kIters2];
#pragma unroll
  for (int it = 0; it < S::kIters2; ++it) {
    const int task = hw + it * S::kHalfWarps;
    if (task < S::kTasks2) w_rms[it].load(p.rms_weight, (rank * kGPC + task % kGPC) * (kGroup / kVec) + lane);
  }

  PDLWaitPrimary<kUsePDL>();

  V res[kTB], out[kTB];
  float gate[kTB];
#pragma unroll
  for (int b = 0; b < kTB; ++b) {
    res[b].load(p.residual + token(b) * kK + m * kH, vid);
    if constexpr (kHasOut) {
      out[b].load(p.out + token(b) * kH, vid);
      gate[b] = p.post[token(b) * kHC + m];
    }
  }

  if constexpr (kOutNorm) {
    V row[S::kItersOut];
#pragma unroll
    for (int it = 0; it < S::kItersOut; ++it) {
      const int task = hw + it * S::kHalfWarps;
      if (task < S::kTasksOut) row[it].load(p.out + token(task / kG) * kH, (task % kG) * (kGroup / kVec) + lane);
    }
#pragma unroll
    for (int it = 0; it < S::kItersOut; ++it) {
      const int task = hw + it * S::kHalfWarps;
      if (task < S::kTasksOut) {
        float f[kVec];
        unpack(row[it], f);
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          sum = __fmaf_rn(f[k], f[k], sum);
        sum = halfwarp_sum(sum);
        if (lane == 0) s.out_part[task / kG][task % kG] = sum;
      }
    }
    __syncthreads();
    if (tid < kTB)
      s.inv_out[tid] = rsqrtf(__fadd_rn(__fdiv_rn(tree_sum<kG>(s.out_part[tid], 1), float(kH)), p.out_eps));
    __syncthreads();
  }

  // ---- new residual slice, its partial sum of squares and hc_fn dots, reduced per (stream, group)
#pragma unroll
  for (int b = 0; b < kTB; ++b) {
    float x[kVec];
    if constexpr (kHasOut) {
      float o[kVec], r[kVec], w[kVec];
      unpack(out[b], o);
      unpack(res[b], r);
      if constexpr (kOutNorm) {
        unpack(w_out, w);
        const float inv = s.inv_out[b];
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          o[k] = bf16_round(__fmul_rn(__fmul_rn(o[k], inv), w[k]));
      }
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        x[k] = __fadd_rn(__fmul_rn(gate[b], o[k]), r[k]);
      const V packed = pack(x);
      unpack(packed, x);
      if (valid(b)) packed.store(p.new_residual + token(b) * kK + m * kH, vid);
      packed.store(&s.x[b][m][0], gi * (kGroup / kVec) + lane);
    } else {
      unpack(res[b], x);
      res[b].store(&s.x[b][m][0], gi * (kGroup / kVec) + lane);
    }

    float v[S::kNV];
#pragma unroll
    for (int n = 0; n < S::kNV; ++n)
      v[n] = 0.f;
#pragma unroll
    for (int n = 0; n < kNG; ++n) {
      float4 w0, w1;
      if constexpr (S::kPreloadFn) {
        w0 = fn[n][0];
        w1 = fn[n][1];
      } else {
        w0 = __ldg(fn_ptr(n));
        w1 = __ldg(fn_ptr(n) + 1);
      }
      const float w[kVec] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        a = __fmaf_rn(w[k], x[k], a);
      v[n] = a;
    }
#pragma unroll
    for (int k = 0; k < kVec; ++k)
      v[kNG] = __fmaf_rn(x[k], x[k], v[kNG]);
    halfwarp_reduce_scatter(v, lane);
#pragma unroll
    for (int i = 0; i < S::kNV; i += kLanes) {
      const int q = i + lane;
      if (q < kNQ) s.part[b][m][gi][q] = q < kNP ? v[i] : 0.f;
    }
  }
  __syncthreads();

  // ---- exchange 1: per group, the stream-summed record to every rank
  device::ptx::bar_cluster_wait_acquire();  // every rank's mbarriers are initialized and armed
  constexpr int kQuads = kNQ / 4;
  constexpr int kSends1 = kTB * kGPC * kQuads * kSplit;
  // Each CTA must send every record unconditionally: a missing byte leaves its peers waiting forever.
#pragma unroll
  for (int task = tid; task < kSends1; task += S::kThreads) {
    const int dst = task % kSplit;
    const int j = task / kSplit % kQuads;
    const int g = task / (kSplit * kQuads) % kGPC;
    const int b = task / (kSplit * kQuads * kGPC);
    float4 sum = *reinterpret_cast<const float4*>(&s.part[b][0][g][4 * j]);
#pragma unroll
    for (int mm = 1; mm < kHC; ++mm) {
      const float4 t = *reinterpret_cast<const float4*>(&s.part[b][mm][g][4 * j]);
      sum = make_float4(__fadd_rn(sum.x, t.x), __fadd_rn(sum.y, t.y), __fadd_rn(sum.z, t.z), __fadd_rn(sum.w, t.w));
    }
    hy4_ihc::ptx::st_async(
        device::ptx::mapa_shared(&s.pool1[b][rank * kGPC + g][4 * j], dst),
        sum,
        device::ptx::mapa_shared(&s.bar[0], dst));
  }
  hy4_ihc::ptx::mbar_wait_peers(&s.bar[0]);

  // ---- gates: identical in every CTA; rank 0 stores next_post
  if (tid < kTB * kNG) {
    const int b = tid / kNG;
    const int q = tid % kNG;
    const float mix = tree_sum<kG>(&s.pool1[b][0][q], kNQ);
    const float sumsq = tree_sum<kG>(&s.pool1[b][0][kNG], kNQ);
    const float rs = rsqrtf(__fadd_rn(__fdiv_rn(sumsq, float(kK)), p.norm_eps));
    const float z = __fadd_rn(__fmul_rn(__fmul_rn(mix, rs), p.hc_scale[q < kHC ? 0 : 1]), p.hc_base[q]);
    const float sig = 1.0f / (1.0f + expf(-z));
    if (q < kHC) {
      s.pre[b][q] = __fadd_rn(sig, p.hc_eps);
    } else if (rank == 0 && valid(b)) {
      p.next_post[token(b) * kHC + (q - kHC)] = __fadd_rn(__fmul_rn(p.magnitude, sig), p.hc_eps);
    }
  }
  __syncthreads();

  // ---- combine, partial sum of r^2 per group to every rank
  float r[S::kIters2][kVec];
#pragma unroll
  for (int it = 0; it < S::kIters2; ++it) {
    const int task = hw + it * S::kHalfWarps;
    if (task < S::kTasks2) {
      const int b = task / kGPC;
      const int g = task % kGPC;
      float acc[kVec];
#pragma unroll
      for (int mm = 0; mm < kHC; ++mm) {
        V xv;
        xv.load(&s.x[b][mm][0], g * (kGroup / kVec) + lane);
        float xf[kVec];
        unpack(xv, xf);
        const float pre = s.pre[b][mm];
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          acc[k] = mm == 0 ? __fmul_rn(pre, xf[k]) : __fadd_rn(acc[k], __fmul_rn(pre, xf[k]));
      }
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        r[it][k] = bf16_round(acc[k]);
        sum = __fmaf_rn(r[it][k], r[it][k], sum);
      }
      sum = halfwarp_sum(sum);
      if (lane < kSplit)
        hy4_ihc::ptx::st_async(
            device::ptx::mapa_shared(&s.pool2[b][rank * kGPC + g], lane),
            sum,
            device::ptx::mapa_shared(&s.bar[1], lane));
    }
  }
  hy4_ihc::ptx::mbar_wait_peers(&s.bar[1]);

  // ---- RMSNorm, optional fp8 group quant
#pragma unroll
  for (int it = 0; it < S::kIters2; ++it) {
    const int task = hw + it * S::kHalfWarps;
    if (task < S::kTasks2) {
      const int b = task / kGPC;
      const int g = rank * kGPC + task % kGPC;
      const float inv = rsqrtf(__fadd_rn(__fdiv_rn(tree_sum<kG>(s.pool2[b], 1), float(kH)), p.rms_eps));
      float w[kVec], yv[kVec];
      unpack(w_rms[it], w);
#pragma unroll
      for (int k = 0; k < kVec; ++k)
        yv[k] = __fmul_rn(__fmul_rn(r[it][k], inv), w[k]);
      const V packed = pack(yv);
      if (valid(b)) packed.store(p.y + token(b) * kH, g * (kGroup / kVec) + lane);
      if constexpr (kQuant) {
        unpack(packed, yv);
        float amax = 0.f;
#pragma unroll
        for (int k = 0; k < kVec; ++k)
          amax = fmaxf(amax, fabsf(yv[k]));
        amax = fmaxf(halfwarp_max(amax), 1e-10f);
        const float quant_scale = hy4_ihc::ptx::div_approx_ftz(448.0f, amax);
        AlignedVector<fp8x2_e4m3_t, kVec / 2> q;
#pragma unroll
        for (int k = 0; k < kVec / 2; ++k)
          q[k] = fp8x2_e4m3_t(
              float2{hy4_ihc::ptx::mul_ftz(yv[2 * k], quant_scale), hy4_ihc::ptx::mul_ftz(yv[2 * k + 1], quant_scale)});
        if (valid(b)) {
          q.store(p.y_q + token(b) * kH, g * (kGroup / kVec) + lane);
          if (lane == 0)
            p.y_s[token(b) * p.y_s_token_stride + g * p.y_s_group_stride] = hy4_ihc::ptx::mul_ftz(amax, 1.0f / 448.0f);
        }
      }
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Hy4 iHC post + pre + RMSNorm (+ out-norm, + fp8 group quant) on a thread-block cluster of kSplit CTAs
 *        per kTB tokens. Supports hidden sizes that are multiples of 128 split evenly over kSplit.
 *
 * \tparam kHC       hc_mult (number of residual streams)
 * \tparam kH        hidden size
 * \tparam kSplit    CTAs per cluster; must divide kH / 128
 * \tparam kTB       tokens per cluster
 * \tparam kHasOut   fold the previous block's iHC post (`output`, `post` given)
 * \tparam kOutNorm  apply the previous block's output RMSNorm to `output` first
 * \tparam kQuant    also write `y` as fp8 e4m3 with fp32 per-128-column scales
 * \tparam kUsePDL   launch with programmatic dependent launch
 */
template <int kHC, int kH, int kSplit, int kTB, bool kHasOut, bool kOutNorm, bool kQuant, bool kUsePDL>
struct Hy4IhcCluster {
  using S = Hy4IhcClusterShape<kHC, kH, kSplit, kTB>;
  static constexpr auto kernel = hy4_ihc_cluster_kernel<kHC, kH, kSplit, kTB, kHasOut, kOutNorm, kQuant, kUsePDL>;

  static cudaLaunchConfig_t config(uint32_t num_clusters, cudaLaunchAttribute* attr) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(kSplit, num_clusters, 1);
    cfg.blockDim = dim3(S::kThreads, 1, 1);
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = kSplit;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cfg;
  }

  /// \brief How many clusters of this configuration the current device holds at once.
  static int64_t max_active_clusters() {
    cudaLaunchAttribute attr[1];
    const auto cfg = config(1024, attr);
    int clusters = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveClusters(&clusters, kernel, &cfg));
    return clusters;
  }

  static void
  run(tvm::ffi::TensorView residual,
      tvm::ffi::Optional<tvm::ffi::TensorView> output,
      tvm::ffi::Optional<tvm::ffi::TensorView> post,
      tvm::ffi::Optional<tvm::ffi::TensorView> out_norm_weight,
      tvm::ffi::TensorView hc_fn,
      tvm::ffi::TensorView hc_scale,
      tvm::ffi::TensorView hc_base,
      tvm::ffi::TensorView rms_weight,
      tvm::ffi::Optional<tvm::ffi::TensorView> new_residual,
      tvm::ffi::TensorView y,
      tvm::ffi::TensorView next_post,
      tvm::ffi::Optional<tvm::ffi::TensorView> y_q,
      tvm::ffi::Optional<tvm::ffi::TensorView> y_s,
      float out_eps,
      float norm_eps,
      float hc_eps,
      float rms_eps,
      float magnitude) {
    using namespace host;
    auto T = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({T, kHC, kH})  //
        .with_dtype<bf16_t>()
        .with_device(dev)
        .ensure_alignment(16)
        .verify(residual);
    TensorMatcher({T, kH})  //
        .with_dtype<bf16_t>()
        .with_device(dev)
        .ensure_alignment(16)
        .verify(y);
    TensorMatcher({T, kHC})  //
        .with_dtype<fp32_t>()
        .with_device(dev)
        .verify(next_post);
    TensorMatcher({2 * kHC, kHC * kH})  //
        .with_dtype<fp32_t>()
        .with_device(dev)
        .ensure_alignment(16)
        .verify(hc_fn);
    TensorMatcher({2})  //
        .with_dtype<fp32_t>()
        .with_device(dev)
        .verify(hc_scale);
    TensorMatcher({2 * kHC})  //
        .with_dtype<fp32_t>()
        .with_device(dev)
        .verify(hc_base);
    TensorMatcher({kH})  //
        .with_dtype<bf16_t>()
        .with_device(dev)
        .ensure_alignment(16)
        .verify(rms_weight);
    CHECK_HOST(output.has_value() == kHasOut && post.has_value() == kHasOut && new_residual.has_value() == kHasOut)
        << "output, post and new_residual must be given exactly when the module folds the iHC post";
    CHECK_HOST(out_norm_weight.has_value() == kOutNorm) << "out_norm_weight must match the module's out-norm flag";
    CHECK_HOST(y_q.has_value() == kQuant && y_s.has_value() == kQuant) << "y_q / y_s must match the quant flag";

    const int64_t num_tokens = T.unwrap();
    Hy4IhcClusterParams params{};
    params.residual = static_cast<const bf16_t*>(residual.data_ptr());
    params.hc_fn = static_cast<const float*>(hc_fn.data_ptr());
    params.hc_scale = static_cast<const float*>(hc_scale.data_ptr());
    params.hc_base = static_cast<const float*>(hc_base.data_ptr());
    params.rms_weight = static_cast<const bf16_t*>(rms_weight.data_ptr());
    params.y = static_cast<bf16_t*>(y.data_ptr());
    params.next_post = static_cast<float*>(next_post.data_ptr());
    if constexpr (kHasOut) {
      TensorMatcher({T, kH})  //
          .with_dtype<bf16_t>()
          .with_device(dev)
          .ensure_alignment(16)
          .verify(output.value());
      TensorMatcher({T, kHC})  //
          .with_dtype<fp32_t>()
          .with_device(dev)
          .verify(post.value());
      TensorMatcher({T, kHC, kH})  //
          .with_dtype<bf16_t>()
          .with_device(dev)
          .ensure_alignment(16)
          .verify(new_residual.value());
      params.out = static_cast<const bf16_t*>(output.value().data_ptr());
      params.post = static_cast<const float*>(post.value().data_ptr());
      params.new_residual = static_cast<bf16_t*>(new_residual.value().data_ptr());
    }
    if constexpr (kOutNorm) {
      TensorMatcher({kH})  //
          .with_dtype<bf16_t>()
          .with_device(dev)
          .ensure_alignment(16)
          .verify(out_norm_weight.value());
      params.out_norm_weight = static_cast<const bf16_t*>(out_norm_weight.value().data_ptr());
    }
    if constexpr (kQuant) {
      auto token_stride = SymbolicSize{"scale_token_stride"};
      auto group_stride = SymbolicSize{"scale_group_stride"};
      TensorMatcher({T, kH})  //
          .with_dtype<fp8_e4m3_t>()
          .with_device(dev)
          .ensure_alignment(8)
          .verify(y_q.value());
      TensorMatcher({T, kH / device::hy4_ihc::kGroup})  //
          .with_strides({token_stride, group_stride})
          .with_dtype<fp32_t>()
          .with_device(dev)
          .verify(y_s.value());
      params.y_q = static_cast<fp8_e4m3_t*>(y_q.value().data_ptr());
      params.y_s = static_cast<float*>(y_s.value().data_ptr());
      params.y_s_token_stride = token_stride.unwrap();
      params.y_s_group_stride = group_stride.unwrap();
    }
    if (num_tokens == 0) return;
    const int64_t num_clusters = (num_tokens + kTB - 1) / kTB;
    CHECK_HOST(num_clusters <= 65535) << "token blocks go in grid.y, got " << num_clusters;
    params.num_tokens = static_cast<uint32_t>(num_tokens);
    params.out_eps = out_eps;
    params.norm_eps = norm_eps;
    params.hc_eps = hc_eps;
    params.rms_eps = rms_eps;
    params.magnitude = magnitude;
    LaunchKernel(dim3(kSplit, static_cast<uint32_t>(num_clusters)), S::kThreads, dev.unwrap())
        .config({.use_pdl = kUsePDL, .cluster_dim = dim3(kSplit, 1, 1)})(kernel, params);
  }
};

}  // namespace sglang
