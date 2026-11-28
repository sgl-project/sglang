// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <cmath>
#include <cstdint>
#include <limits>

#if !defined(USE_ROCM) && defined(SGL_CUDA_ARCH) && (SGL_CUDA_ARCH < 1200 || SGL_CUDA_ARCH >= 1300)
#error "norm_ctrl_sm120.cuh requires SM12x (compute capability 12.x)"
#endif

namespace sglang::inkling2_norm_ctrl_sm120 {

namespace {

constexpr int32_t kD = 2304;
constexpr float kInvD = 1.0f / static_cast<float>(kD);
constexpr int32_t kRank = 64;
constexpr int32_t kBlock = 256;
constexpr int32_t kWarps = 8;
constexpr int32_t kElems = 9;
constexpr size_t kSm12xMaxSmemBytes = 101376;

struct Ptr4 {
  bf16_t const* p[4];
};

struct alignas(32) u32x8 {
  uint32_t v[8];
};

SGL_DEVICE float rsqrt_approx_f32(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE float compute_rstd_nofma(float const (&x)[kElems], float eps, float s_red8[kWarps]) {
  int32_t const tid = static_cast<int32_t>(threadIdx.x);
  int32_t const lane = static_cast<int32_t>(device::get_lane_id());
  int32_t const warp = tid >> 5;
  float sq = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i) {
    sq = __fadd_rn(sq, __fmul_rn(x[i], x[i]));
  }
  sq = device::warp::reduce_sum(sq);
  if (lane == 0) s_red8[warp] = sq;
  __syncthreads();
  if (warp == 0) {
    float const v = device::warp::reduce_sum((lane < kWarps) ? s_red8[lane] : 0.0f);
    if (lane == 0) {
      s_red8[0] = rsqrt_approx_f32(__fadd_rn(__fdiv_rn(v, static_cast<float>(kD)), eps));
    }
  }
  __syncthreads();
  float const rstd = s_red8[0];
  __syncthreads();
  return rstd;
}

SGL_DEVICE u32x8 ld_w_v8(void const* ptr) {
  u32x8 r{};
#if defined(__CUDA_ARCH__)
#if SGL_ARCH_BLACKWELL_OR_GREATER
  asm volatile(
      "ld.global.nc.L1::no_allocate.L2::256B.v8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(ptr));
#else
  auto const* p = static_cast<uint8_t const*>(ptr);
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
               : "l"(p));
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
               : "l"(p + 16));
#endif
#endif
  return r;
}

SGL_DEVICE u32x8 ld_x_v8(void const* ptr) {
  u32x8 r{};
#if defined(__CUDA_ARCH__)
#if SGL_ARCH_BLACKWELL_OR_GREATER
  asm volatile(
      "ld.global.L1::evict_last.L2::evict_last.v8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(ptr));
#else
  auto const* p = static_cast<uint8_t const*>(ptr);
  asm volatile("ld.global.ca.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
               : "l"(p));
  asm volatile("ld.global.ca.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
               : "l"(p + 16));
#endif
#endif
  return r;
}

SGL_DEVICE void mma_v8(float (&acc)[4], u32x8 const& x0, u32x8 const& x1, u32x8 const& w) {
#if defined(__CUDA_ARCH__)
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(x0.v[2 * i]),
          "r"(x1.v[2 * i]),
          "r"(x0.v[2 * i + 1]),
          "r"(x1.v[2 * i + 1]),
          "r"(w.v[2 * i]),
          "r"(w.v[2 * i + 1]));
  }
#endif
}

template <bool HAS_ADDEND, int32_t N_STACK>
SGL_DEVICE void attn_res_norm_impl(
    Ptr4 stack_ptrs,
    bf16_t const* __restrict__ partial,
    bf16_t const* __restrict__ addend,
    float const* __restrict__ query,
    bf16_t const* __restrict__ norm_w,
    bf16_t* __restrict__ out_normed,
    int32_t t,
    float eps,
    float norm_eps,
    float s_red[kWarps][10],
    float s_weights[6]) {
  constexpr int32_t N_SRC = N_STACK + 1;
  int32_t const tid = static_cast<int32_t>(threadIdx.x);
  int32_t const lane = static_cast<int32_t>(device::get_lane_id());
  int32_t const warp = tid >> 5;
  int64_t const row_off = static_cast<int64_t>(t) * kD;

  device::PDLWaitPrimary<true>();

  float qv0[kElems];
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i) {
    qv0[i] = __ldg(&query[tid + i * kBlock]);
  }

  bf16x2_t s_pair[N_STACK > 0 ? N_STACK : 1][4];
  bf16_t s_tail[N_STACK > 0 ? N_STACK : 1];
  float last_x[kElems];

  if constexpr (N_STACK > 0) {
#pragma unroll
    for (int32_t n = 0; n < N_STACK; ++n) {
      bf16_t const* src_n = stack_ptrs.p[n] + row_off;
#pragma unroll
      for (int32_t p = 0; p < 4; ++p) {
        s_pair[n][p].x = src_n[tid + (2 * p) * kBlock];
        s_pair[n][p].y = src_n[tid + (2 * p + 1) * kBlock];
      }
      s_tail[n] = src_n[tid + 8 * kBlock];
    }
  }
  {
    bf16_t const* p_ptr = partial + row_off;
    bf16_t const* a_ptr = HAS_ADDEND ? (addend + row_off) : nullptr;
#pragma unroll
    for (int32_t i = 0; i < kElems; ++i) {
      int32_t const d = tid + i * kBlock;
      float x = __bfloat162float(p_ptr[d]);
      if constexpr (HAS_ADDEND) x = __fadd_rn(x, __bfloat162float(a_ptr[d]));
      last_x[i] = x;
    }
  }

  float sum_sq[N_SRC];
  float dots[N_SRC];
  if constexpr (N_STACK > 0) {
#pragma unroll
    for (int32_t n = 0; n < N_STACK; ++n) {
      float sq = 0.0f, dp = 0.0f;
#pragma unroll
      for (int32_t p = 0; p < 4; ++p) {
        float2 const f2 = __bfloat1622float2(s_pair[n][p]);
        sq += f2.x * f2.x;
        dp += qv0[2 * p] * f2.x;
        sq += f2.y * f2.y;
        dp += qv0[2 * p + 1] * f2.y;
      }
      float const x8 = __bfloat162float(s_tail[n]);
      sq += x8 * x8;
      dp += qv0[8] * x8;
      sum_sq[n] = sq;
      dots[n] = dp;
    }
  }
  {
    float sq = 0.0f, dp = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kElems; ++i) {
      sq += last_x[i] * last_x[i];
      dp += qv0[i] * last_x[i];
    }
    sum_sq[N_STACK] = sq;
    dots[N_STACK] = dp;
  }

#pragma unroll
  for (int32_t n = 0; n < N_SRC; ++n) {
    float const vs = device::warp::reduce_sum(sum_sq[n]);
    float const vd = device::warp::reduce_sum(dots[n]);
    if (lane == 0) {
      s_red[warp][n] = vs;
      s_red[warp][N_SRC + n] = vd;
    }
  }
  __syncthreads();

  if (warp == 0) {
    float inv_rms[N_SRC];
    float logits[N_SRC];
#pragma unroll
    for (int32_t n = 0; n < N_SRC; ++n) {
      float const vs = device::warp::reduce_sum((lane < kWarps) ? s_red[lane][n] : 0.0f);
      inv_rms[n] = rsqrt_approx_f32(__fadd_rn(__fmul_rn(vs, kInvD), eps));
      float const vd = device::warp::reduce_sum((lane < kWarps) ? s_red[lane][N_SRC + n] : 0.0f);
      logits[n] = vd * inv_rms[n];
    }
    if (lane == 0) {
      float max_l = logits[0];
#pragma unroll
      for (int32_t n = 1; n < N_SRC; ++n)
        max_l = fmaxf(max_l, logits[n]);
      float exps[N_SRC], sum_e = 0.0f;
#pragma unroll
      for (int32_t n = 0; n < N_SRC; ++n) {
        exps[n] = expf(logits[n] - max_l);
        sum_e += exps[n];
      }
#pragma unroll
      for (int32_t n = 0; n < N_SRC; ++n) {
        s_weights[n] = exps[n] * inv_rms[n];
      }
      s_weights[N_SRC] = 1.0f / sum_e;
    }
  }
  __syncthreads();

  float mix[kElems];
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i)
    mix[i] = 0.0f;
  if constexpr (N_STACK > 0) {
#pragma unroll
    for (int32_t n = 0; n < N_STACK; ++n) {
      float const wn = s_weights[n];
#pragma unroll
      for (int32_t p = 0; p < 4; ++p) {
        float2 const f2 = __bfloat1622float2(s_pair[n][p]);
        mix[2 * p] += wn * f2.x;
        mix[2 * p + 1] += wn * f2.y;
      }
      mix[8] += wn * __bfloat162float(s_tail[n]);
    }
  }
  float const wl = s_weights[N_STACK];
  float const inv_sum = s_weights[N_SRC];
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i) {
    mix[i] = (mix[i] + wl * last_x[i]) * inv_sum;
  }

  float sq = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i)
    sq += mix[i] * mix[i];
  float const v = device::warp::reduce_sum(sq);
  if (lane == 0) s_red[warp][0] = v;
  __syncthreads();
  if (warp == 0) {
    float const sum_sq_out = device::warp::reduce_sum((lane < kWarps) ? s_red[lane][0] : 0.0f);
    if (lane == 0) {
      s_weights[0] = rsqrt_approx_f32(__fadd_rn(__fdiv_rn(sum_sq_out, static_cast<float>(kD)), norm_eps));
    }
  }
  __syncthreads();
  device::PDLTriggerSecondary<true>();

  float const rstd = s_weights[0];
  bf16_t* dst = out_normed + row_off;
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i) {
    int32_t const d = tid + i * kBlock;
    dst[d] = __float2bfloat16_rn(mix[i] * rstd * __bfloat162float(__ldg(&norm_w[d])));
  }
}

template <bool HAS_ADDEND, int32_t N_STACK>
SGL_DEVICE void attn_res_mix_1q_s1_impl(
    Ptr4 stack_ptrs,
    bf16_t const* __restrict__ partial,
    bf16_t const* __restrict__ addend,
    float const* __restrict__ q_ptr,
    float* __restrict__ out_mix_row,
    bf16_t* __restrict__ cat0_row,
    bf16_t* __restrict__ cat1_row,
    bf16_t* __restrict__ cat2_row,
    bool broadcast_3x,
    int32_t t,
    float eps,
    float s_red[kWarps][20],
    float s_weights[6]) {
  constexpr int32_t N_SRC = N_STACK + 1;
  int32_t const tid = static_cast<int32_t>(threadIdx.x);
  int32_t const lane = static_cast<int32_t>(device::get_lane_id());
  int32_t const warp = tid >> 5;
  int64_t const row_off = static_cast<int64_t>(t) * kD;

  device::PDLWaitPrimary<true>();

  float qv[kElems];
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i)
    qv[i] = __ldg(&q_ptr[tid + i * kBlock]);

  bf16x2_t s_pair[N_STACK][4];
  bf16_t s_tail[N_STACK];
  float last_x[kElems];
#pragma unroll
  for (int32_t n = 0; n < N_STACK; ++n) {
    bf16_t const* src_n = stack_ptrs.p[n] + row_off;
#pragma unroll
    for (int32_t p = 0; p < 4; ++p) {
      s_pair[n][p].x = src_n[tid + (2 * p) * kBlock];
      s_pair[n][p].y = src_n[tid + (2 * p + 1) * kBlock];
    }
    s_tail[n] = src_n[tid + 8 * kBlock];
  }
  {
    bf16_t const* p_ptr = partial + row_off;
    bf16_t const* a_ptr = HAS_ADDEND ? (addend + row_off) : nullptr;
#pragma unroll
    for (int32_t i = 0; i < kElems; ++i) {
      int32_t const d = tid + i * kBlock;
      float x = __bfloat162float(p_ptr[d]);
      if constexpr (HAS_ADDEND) x = __fadd_rn(x, __bfloat162float(a_ptr[d]));
      last_x[i] = x;
    }
  }

  float sum_sq[N_SRC], dots[N_SRC];
#pragma unroll
  for (int32_t n = 0; n < N_STACK; ++n) {
    float sq = 0.0f, dp = 0.0f;
#pragma unroll
    for (int32_t p = 0; p < 4; ++p) {
      float2 const f2 = __bfloat1622float2(s_pair[n][p]);
      sq += f2.x * f2.x;
      dp += qv[2 * p] * f2.x;
      sq += f2.y * f2.y;
      dp += qv[2 * p + 1] * f2.y;
    }
    float const x8 = __bfloat162float(s_tail[n]);
    sq += x8 * x8;
    dp += qv[8] * x8;
    sum_sq[n] = sq;
    dots[n] = dp;
  }
  {
    float sq = 0.0f, dp = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kElems; ++i) {
      sq += last_x[i] * last_x[i];
      dp += qv[i] * last_x[i];
    }
    sum_sq[N_STACK] = sq;
    dots[N_STACK] = dp;
  }
#pragma unroll
  for (int32_t n = 0; n < N_SRC; ++n) {
    float const vs = device::warp::reduce_sum(sum_sq[n]);
    float const vd = device::warp::reduce_sum(dots[n]);
    if (lane == 0) {
      s_red[warp][n] = vs;
      s_red[warp][N_SRC + n] = vd;
    }
  }
  __syncthreads();
  if (warp == 0) {
    float inv_rms[N_SRC], logits[N_SRC];
#pragma unroll
    for (int32_t n = 0; n < N_SRC; ++n) {
      float const vs = device::warp::reduce_sum((lane < kWarps) ? s_red[lane][n] : 0.0f);
      inv_rms[n] = rsqrt_approx_f32(__fadd_rn(__fmul_rn(vs, kInvD), eps));
      float const vd = device::warp::reduce_sum((lane < kWarps) ? s_red[lane][N_SRC + n] : 0.0f);
      logits[n] = vd * inv_rms[n];
    }
    if (lane == 0) {
      float max_l = logits[0];
#pragma unroll
      for (int32_t n = 1; n < N_SRC; ++n)
        max_l = fmaxf(max_l, logits[n]);
      float exps[N_SRC], sum_e = 0.0f;
#pragma unroll
      for (int32_t n = 0; n < N_SRC; ++n) {
        exps[n] = expf(logits[n] - max_l);
        sum_e += exps[n];
      }
#pragma unroll
      for (int32_t n = 0; n < N_SRC; ++n) {
        s_weights[n] = exps[n] * inv_rms[n];
      }
      s_weights[N_SRC] = 1.0f / sum_e;
    }
  }
  __syncthreads();

  float mix[kElems];
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i)
    mix[i] = 0.0f;
#pragma unroll
  for (int32_t n = 0; n < N_STACK; ++n) {
    float const wn = s_weights[n];
#pragma unroll
    for (int32_t p = 0; p < 4; ++p) {
      float2 const f2 = __bfloat1622float2(s_pair[n][p]);
      mix[2 * p] += wn * f2.x;
      mix[2 * p + 1] += wn * f2.y;
    }
    mix[8] += wn * __bfloat162float(s_tail[n]);
  }
  float const wl = s_weights[N_STACK];
  float const inv_sum = s_weights[N_SRC];
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i) {
    mix[i] = (mix[i] + wl * last_x[i]) * inv_sum;
    out_mix_row[tid + i * kBlock] = mix[i];
  }

  float sq_m = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i)
    sq_m = __fadd_rn(sq_m, __fmul_rn(mix[i], mix[i]));
  sq_m = device::warp::reduce_sum(sq_m);
  if (lane == 0) s_red[warp][0] = sq_m;
  __syncthreads();
  if (warp == 0) {
    float const v = device::warp::reduce_sum((lane < kWarps) ? s_red[lane][0] : 0.0f);
    if (lane == 0) {
      s_red[0][0] = rsqrt_approx_f32(__fadd_rn(__fdiv_rn(v, static_cast<float>(kD)), eps));
    }
  }
  __syncthreads();
  float const rstd = s_red[0][0];
  device::PDLTriggerSecondary<true>();
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i) {
    int32_t const d = tid + i * kBlock;
    bf16_t const y = __float2bfloat16_rn(__fmul_rn(mix[i], rstd));
    cat0_row[d] = y;
    if (broadcast_3x) {
      cat1_row[d] = y;
      cat2_row[d] = y;
    }
  }
}

inline int32_t verify_stack_blocks(
    tvm::ffi::Optional<tvm::ffi::TensorView> const* blocks,
    int32_t max_blocks,
    host::SymbolicSize& T,
    host::SymbolicDevice& device,
    Ptr4& out_ptrs) {
  int32_t n_stack = 0;
  bool seen_none = false;
  for (int32_t i = 0; i < max_blocks; ++i) {
    if (blocks[i].has_value()) {
      CHECK_HOST(!seen_none) << "stack blocks must be packed in slots 0..n_stack-1";
      host::TensorMatcher({T, kD})  //
          .with_dtype<bf16_t>()
          .with_device(device)
          .ensure_alignment(16)
          .verify(blocks[i].value());
      out_ptrs.p[n_stack++] = static_cast<bf16_t const*>(blocks[i].value().data_ptr());
    } else {
      seen_none = true;
    }
  }
  for (int32_t i = n_stack; i < 4; ++i)
    out_ptrs.p[i] = nullptr;
  return n_stack;
}

}  // namespace

template <bool HAS_ADDEND, int32_t N_STACK>
__global__ void __launch_bounds__(256, 3) attn_res_norm_1q_kernel(
    Ptr4 stack_ptrs,
    bf16_t const* __restrict__ partial,
    bf16_t const* __restrict__ addend,
    float const* __restrict__ queries,
    bf16_t const* __restrict__ norm_w,
    bf16_t* __restrict__ out_normed,
    int64_t out_q_stride,
    int32_t T,
    float eps,
    float norm_eps) {
  int32_t const t = static_cast<int32_t>(blockIdx.x);
  int32_t const q = static_cast<int32_t>(blockIdx.y);
  if (t >= T) return;
  __shared__ float s_red[kWarps][10];
  __shared__ float s_weights[6];
  static_assert(sizeof(s_red) + sizeof(s_weights) <= kSm12xMaxSmemBytes);
  attn_res_norm_impl<HAS_ADDEND, N_STACK>(
      stack_ptrs,
      partial,
      addend,
      queries + static_cast<int64_t>(q) * kD,
      norm_w + static_cast<int64_t>(q) * kD,
      out_normed + static_cast<int64_t>(q) * out_q_stride,
      t,
      eps,
      norm_eps,
      s_red,
      s_weights);
}

template <int32_t N_STACK>
__global__ void __launch_bounds__(256, 4) pre_attn_mix_cross_s1_lowm_kernel(
    Ptr4 stack_ptrs,
    bf16_t const* __restrict__ partial,
    float const* __restrict__ attn_q,
    bf16_t const* __restrict__ prev0,
    bf16_t const* __restrict__ prev1,
    float const* __restrict__ cross_q,
    float* __restrict__ mix_qkv,
    float* __restrict__ msg_buf,
    bf16_t* __restrict__ cat_3x,
    int64_t out_q_stride,
    bool is_swa,
    int32_t T,
    float eps) {
  int32_t const t = static_cast<int32_t>(blockIdx.x);
  int32_t const by = static_cast<int32_t>(blockIdx.y);
  if (t >= T) return;
  int32_t const tid = static_cast<int32_t>(threadIdx.x);
  int32_t const lane = static_cast<int32_t>(device::get_lane_id());
  int32_t const warp = tid >> 5;
  int64_t const row_off = static_cast<int64_t>(t) * kD;
  int64_t const cat_stream_stride = static_cast<int64_t>(T) * (2 * kD);
  int64_t const cat_row_off = static_cast<int64_t>(t) * (2 * kD);

  if (by < 4) {
    device::PDLWaitPrimary<true>();
    float const* q_ptr = cross_q + static_cast<int64_t>(by) * kD;
    float qv[kElems];
#pragma unroll
    for (int32_t i = 0; i < kElems; ++i)
      qv[i] = __ldg(&q_ptr[tid + i * kBlock]);

    bf16_t const* p0 = prev0 + row_off;
    bf16_t const* p1 = prev1 + row_off;
    float s0[kElems], s1[kElems];
    float sq0 = 0.0f, sq1 = 0.0f, a = 0.0f, b = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < kElems; ++i) {
      int32_t const d = tid + i * kBlock;
      s0[i] = __bfloat162float(p0[d]);
      s1[i] = __bfloat162float(p1[d]);
      sq0 += s0[i] * s0[i];
      sq1 += s1[i] * s1[i];
      a += qv[i] * s0[i];
      b += qv[i] * s1[i];
    }
    __shared__ float s_red4[kWarps][4];
    __shared__ float s_w[3];
    sq0 = device::warp::reduce_sum(sq0);
    sq1 = device::warp::reduce_sum(sq1);
    a = device::warp::reduce_sum(a);
    b = device::warp::reduce_sum(b);
    if (lane == 0) {
      s_red4[warp][0] = sq0;
      s_red4[warp][1] = sq1;
      s_red4[warp][2] = a;
      s_red4[warp][3] = b;
    }
    __syncthreads();
    if (warp == 0) {
      float const v0 = device::warp::reduce_sum((lane < kWarps) ? s_red4[lane][0] : 0.0f);
      float const v1 = device::warp::reduce_sum((lane < kWarps) ? s_red4[lane][1] : 0.0f);
      float const inv0 = rsqrt_approx_f32(__fadd_rn(__fmul_rn(v0, kInvD), eps));
      float const inv1 = rsqrt_approx_f32(__fadd_rn(__fmul_rn(v1, kInvD), eps));
      float const l0 = device::warp::reduce_sum((lane < kWarps) ? s_red4[lane][2] : 0.0f) * inv0;
      float const l1 = device::warp::reduce_sum((lane < kWarps) ? s_red4[lane][3] : 0.0f) * inv1;
      if (lane == 0) {
        float const ml = fmaxf(l0, l1);
        float const e0 = expf(l0 - ml);
        float const e1 = expf(l1 - ml);
        s_w[0] = e0 * inv0;
        s_w[1] = e1 * inv1;
        s_w[2] = 1.0f / (e0 + e1);
      }
    }
    __syncthreads();
    float const w0 = s_w[0];
    float const w1 = s_w[1];
    float const inv_s = s_w[2];
    float* dst_msg = msg_buf + static_cast<int64_t>(by) * out_q_stride + row_off;
    float mix[kElems];
#pragma unroll
    for (int32_t i = 0; i < kElems; ++i) {
      float const m = (w0 * s0[i] + w1 * s1[i]) * inv_s;
      mix[i] = m;
      dst_msg[tid + i * kBlock] = m;
    }
    if (by < 3) {
      __shared__ float s_r8[kWarps];
      float const rstd = compute_rstd_nofma(mix, eps, s_r8);
      device::PDLTriggerSecondary<true>();
      bf16_t* cat_q = cat_3x + static_cast<int64_t>(by) * cat_stream_stride + cat_row_off + kD;
#pragma unroll
      for (int32_t i = 0; i < kElems; ++i) {
        cat_q[tid + i * kBlock] = __float2bfloat16_rn(__fmul_rn(mix[i], rstd));
      }
    } else {
      device::PDLTriggerSecondary<true>();
    }
    return;
  }

  int32_t const q = by - 4;
  __shared__ float s_red[kWarps][20];
  __shared__ float s_weights[6];
  bf16_t* c0 = cat_3x + (is_swa ? 0 : static_cast<int64_t>(q) * cat_stream_stride) + cat_row_off;
  bf16_t* c1 = cat_3x + cat_stream_stride + cat_row_off;
  bf16_t* c2 = cat_3x + 2 * cat_stream_stride + cat_row_off;
  attn_res_mix_1q_s1_impl<false, N_STACK>(
      stack_ptrs,
      partial,
      nullptr,
      attn_q + static_cast<int64_t>(q) * kD,
      mix_qkv + static_cast<int64_t>(q) * out_q_stride + row_off,
      c0,
      c1,
      c2,
      is_swa,
      t,
      eps,
      s_red,
      s_weights);
}

template <bool HAS_ADDEND, int32_t N_STACK>
__global__ void __launch_bounds__(256, 4) pre_mlp_mix_s1_kernel(
    Ptr4 stack_ptrs,
    bf16_t const* __restrict__ partial,
    bf16_t const* __restrict__ addend,
    float const* __restrict__ mlp_q,
    float const* __restrict__ msg_f32,
    float* __restrict__ mix_mlp,
    bf16_t* __restrict__ cat_bf16,
    int32_t T,
    float eps) {
  int32_t const t = static_cast<int32_t>(blockIdx.x);
  int32_t const by = static_cast<int32_t>(blockIdx.y);
  if (t >= T) return;
  int32_t const tid = static_cast<int32_t>(threadIdx.x);
  int64_t const row_off = static_cast<int64_t>(t) * kD;
  int64_t const cat_off = static_cast<int64_t>(t) * (2 * kD);

  if (by == 1) {
    device::PDLWaitPrimary<true>();
    float const* src = msg_f32 + row_off;
    float x[kElems];
#pragma unroll
    for (int32_t i = 0; i < kElems; ++i)
      x[i] = src[tid + i * kBlock];
    __shared__ float s_r8[kWarps];
    float const rstd = compute_rstd_nofma(x, eps, s_r8);
    device::PDLTriggerSecondary<true>();
    bf16_t* dst = cat_bf16 + cat_off + kD;
#pragma unroll
    for (int32_t i = 0; i < kElems; ++i) {
      dst[tid + i * kBlock] = __float2bfloat16_rn(__fmul_rn(x[i], rstd));
    }
    return;
  }

  __shared__ float s_red[kWarps][20];
  __shared__ float s_weights[6];
  attn_res_mix_1q_s1_impl<HAS_ADDEND, N_STACK>(
      stack_ptrs,
      partial,
      addend,
      mlp_q,
      mix_mlp + row_off,
      cat_bf16 + cat_off,
      nullptr,
      nullptr,
      false,
      t,
      eps,
      s_red,
      s_weights);
}

template <int32_t M_TILES, bool M_LE_8>
__global__ void __launch_bounds__(256, 4) ctrl_down_v8_kernel(
    bf16_t const* __restrict__ cat,
    bf16_t const* __restrict__ W_down,
    bf16_t* __restrict__ h_out,
    int32_t M,
    int32_t R,
    int32_t K) {
  int32_t const tile = static_cast<int32_t>(blockIdx.x);
  int32_t const s = static_cast<int32_t>(blockIdx.y);
  constexpr int32_t BN = 4;
  constexpr int32_t WARPS_K = 8;
  constexpr int32_t K_STEP = 64 * WARPS_K;
  int32_t const tid = static_cast<int32_t>(threadIdx.x);
  int32_t const warp = tid >> 5;
  int32_t const lane = static_cast<int32_t>(device::get_lane_id());
  int32_t const g = lane >> 2;
  int32_t const t = lane & 3;
  int32_t const n_base = tile * BN;
  int32_t const n_row = n_base + g;
  bool const w_ok = (g < BN) && (n_row < R);

  bf16_t const* A = cat + static_cast<int64_t>(s) * M * K;
  bf16_t const* W = W_down + static_cast<int64_t>(s) * R * K;
  bf16_t* C = h_out + static_cast<int64_t>(s) * M * R;
  bf16_t const* w_base = w_ok ? (W + static_cast<int64_t>(n_row) * K + t * 16) : nullptr;
  bf16_t const* a_base = A + t * 16;
  int32_t const k0 = warp * 64;

  device::PDLWaitPrimary<true>();
  device::PDLTriggerSecondary<true>();
  u32x8 w_cur = (w_ok && k0 < K) ? ld_w_v8(w_base + k0) : u32x8{};

  float acc[M_TILES][4];
#pragma unroll
  for (int32_t mt = 0; mt < M_TILES; ++mt) {
    acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.0f;
  }

  for (int32_t k = k0; k < K; k += K_STEP) {
    int32_t const k_next = k + K_STEP;
    u32x8 const w_nxt = (w_ok && k_next < K) ? ld_w_v8(w_base + k_next) : u32x8{};
#pragma unroll
    for (int32_t mt = 0; mt < M_TILES; ++mt) {
      int32_t const m0 = mt * 16 + g;
      u32x8 const x0 = (m0 < M) ? ld_x_v8(a_base + static_cast<int64_t>(m0) * K + k) : u32x8{};
      u32x8 x1{};
      if constexpr (!M_LE_8) {
        int32_t const m1 = m0 + 8;
        if (m1 < M) x1 = ld_x_v8(a_base + static_cast<int64_t>(m1) * K + k);
      }
      mma_v8(acc[mt], x0, x1, w_cur);
    }
    w_cur = w_nxt;
  }

  __shared__ float4 s_red[WARPS_K - 1][M_TILES][32];
  static_assert(sizeof(s_red) <= kSm12xMaxSmemBytes);
  if (warp > 0) {
#pragma unroll
    for (int32_t mt = 0; mt < M_TILES; ++mt) {
      s_red[warp - 1][mt][lane] = make_float4(acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]);
    }
  }
  __syncthreads();

  if (warp == 0 && (2 * t < BN)) {
    int32_t const col0 = n_base + 2 * t;
#pragma unroll
    for (int32_t mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
      for (int32_t w = 0; w < WARPS_K - 1; ++w) {
        float4 const v = s_red[w][mt][lane];
        acc[mt][0] += v.x;
        acc[mt][1] += v.y;
        acc[mt][2] += v.z;
        acc[mt][3] += v.w;
      }
      int32_t const m0 = mt * 16 + g;
      if (m0 < M) {
        *reinterpret_cast<bf16x2_t*>(C + static_cast<int64_t>(m0) * R + col0) =
            __floats2bfloat162_rn(acc[mt][0], acc[mt][1]);
      }
      if constexpr (!M_LE_8) {
        int32_t const m1 = m0 + 8;
        if (m1 < M) {
          *reinterpret_cast<bf16x2_t*>(C + static_cast<int64_t>(m1) * R + col0) =
              __floats2bfloat162_rn(acc[mt][2], acc[mt][3]);
        }
      }
    }
  }
}

template <int32_t M_TILES, bool M_LE_8>
__global__ void __launch_bounds__(256, 4) ctrl_up_k64_kernel(
    bf16_t const* __restrict__ h,
    bf16_t const* __restrict__ W_forget,
    bf16_t const* __restrict__ W_gain,
    float* __restrict__ fg_out,
    int32_t M) {
  int32_t const warp = static_cast<int32_t>(threadIdx.x >> 5);
  int32_t const lane = static_cast<int32_t>(device::get_lane_id());
  int32_t const g = lane >> 2;
  int32_t const t = lane & 3;
  int32_t const n_base = static_cast<int32_t>(blockIdx.x) * 64 + warp * 8;
  int32_t const is_gain = static_cast<int32_t>(blockIdx.y);
  int32_t const s = static_cast<int32_t>(blockIdx.z);

  device::PDLWaitPrimary<true>();
  device::PDLTriggerSecondary<true>();
  bf16_t const* W = (is_gain ? W_gain : W_forget) + static_cast<int64_t>(s) * kD * kRank;
  u32x8 const w = ld_w_v8(W + static_cast<int64_t>(n_base + g) * kRank + t * 16);

  bf16_t const* A = h + static_cast<int64_t>(s) * M * kRank + t * 16;
  float* C = fg_out + static_cast<int64_t>(s) * M * (2 * kD) + (is_gain ? kD : 0);
  int32_t const col0 = n_base + 2 * t;
#pragma unroll
  for (int32_t mt = 0; mt < M_TILES; ++mt) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int32_t const m0 = mt * 16 + g;
    u32x8 const x0 = (m0 < M) ? ld_x_v8(A + static_cast<int64_t>(m0) * kRank) : u32x8{};
    u32x8 x1{};
    if constexpr (!M_LE_8) {
      int32_t const m1 = m0 + 8;
      if (m1 < M) x1 = ld_x_v8(A + static_cast<int64_t>(m1) * kRank);
    }
    mma_v8(acc, x0, x1, w);
    if (m0 < M) {
      C[static_cast<int64_t>(m0) * (2 * kD) + col0] = acc[0];
      C[static_cast<int64_t>(m0) * (2 * kD) + col0 + 1] = acc[1];
    }
    if constexpr (!M_LE_8) {
      int32_t const m1 = m0 + 8;
      if (m1 < M) {
        C[static_cast<int64_t>(m1) * (2 * kD) + col0] = acc[2];
        C[static_cast<int64_t>(m1) * (2 * kD) + col0 + 1] = acc[3];
      }
    }
  }
}

__global__ void __launch_bounds__(256, 4) ctrl_merge_rmsnorm_kernel(
    float const* __restrict__ in_f32,
    float const* __restrict__ msg_f32,
    float const* __restrict__ fg_f32,
    bf16_t const* __restrict__ norm_w,
    bf16_t* __restrict__ out_bf16,
    int64_t in_stride,
    int64_t msg_stride,
    int64_t out_stride,
    int64_t norm_stride,
    int32_t T,
    float eps) {
  int32_t const t = static_cast<int32_t>(blockIdx.x);
  int32_t const s = static_cast<int32_t>(blockIdx.y);
  if (t >= T) return;
  int32_t const tid = static_cast<int32_t>(threadIdx.x);
  int32_t const lane = static_cast<int32_t>(device::get_lane_id());
  int32_t const warp = tid >> 5;

  float const* in_row = in_f32 + static_cast<int64_t>(s) * in_stride + static_cast<int64_t>(t) * kD;
  float const* msg_row = msg_f32 + static_cast<int64_t>(s) * msg_stride + static_cast<int64_t>(t) * kD;
  float const* fg_row = fg_f32 + (static_cast<int64_t>(s) * T + t) * (2 * kD);
  bf16_t const* nw = norm_w + static_cast<int64_t>(s) * norm_stride;
  bf16_t* out_row = out_bf16 + static_cast<int64_t>(s) * out_stride + static_cast<int64_t>(t) * kD;

  device::PDLWaitPrimary<true>();
  device::PDLTriggerSecondary<true>();

  float in_v[kElems];
  float msg_v[kElems];
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i) {
    int32_t const d = tid + i * kBlock;
    in_v[i] = in_row[d];
    msg_v[i] = msg_row[d];
  }

  float merged[kElems];
  float sq = 0.0f;
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i) {
    int32_t const d = tid + i * kBlock;
    float const wf = tanhf(__ldcs(&fg_row[d]));
    float const wg = tanhf(__ldcs(&fg_row[kD + d]));
    float const m = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, wf), in_v[i]), __fmul_rn(wg, msg_v[i]));
    merged[i] = m;
    sq = __fadd_rn(sq, __fmul_rn(m, m));
  }
  __shared__ float s_red8[kWarps];
  sq = device::warp::reduce_sum(sq);
  if (lane == 0) s_red8[warp] = sq;
  __syncthreads();
  if (warp == 0) {
    float const v = device::warp::reduce_sum((lane < kWarps) ? s_red8[lane] : 0.0f);
    if (lane == 0) {
      s_red8[0] = rsqrt_approx_f32(__fadd_rn(__fdiv_rn(v, static_cast<float>(kD)), eps));
    }
  }
  __syncthreads();

  float const rstd = s_red8[0];
#pragma unroll
  for (int32_t i = 0; i < kElems; ++i) {
    int32_t const d = tid + i * kBlock;
    out_row[d] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(merged[i], rstd), __bfloat162float(__ldg(&nw[d]))));
  }
}

/**
 * \brief Fused Block Attention Residuals (normalize_values=True) + RMSNorm to BF16.
 * \param b0..b3 Optional completed block tensors [T, 2304] BF16 packed in slots 0..n_stack-1.
 * \param partial Current partial block [T, 2304] BF16.
 * \param addend Optional residual addend [T, 2304] BF16 added to partial.
 * \param queries Pre-scaled AttnRes queries [num_q, 2304] FP32 (num_q in {1, 3}).
 * \param norm_w RMSNorm weights [num_q, 2304] BF16.
 * \param out Output [num_q, T, 2304] BF16.
 * \param eps AttnRes RMS epsilon.
 * \param norm_eps Output RMSNorm epsilon.
 */
inline void attn_res_norm_bf16(
    tvm::ffi::Optional<tvm::ffi::TensorView> b0,
    tvm::ffi::Optional<tvm::ffi::TensorView> b1,
    tvm::ffi::Optional<tvm::ffi::TensorView> b2,
    tvm::ffi::Optional<tvm::ffi::TensorView> b3,
    tvm::ffi::TensorView partial,
    tvm::ffi::Optional<tvm::ffi::TensorView> addend,
    tvm::ffi::TensorView queries,
    tvm::ffi::TensorView norm_w,
    tvm::ffi::TensorView out,
    double eps,
    double norm_eps) {
  auto T = host::SymbolicSize{"tokens"};
  auto num_q_sym = host::SymbolicSize{"num_queries"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();

  host::TensorMatcher({T, kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(partial);
  if (addend.has_value()) {
    host::TensorMatcher({T, kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(addend.value());
  }
  host::TensorMatcher({num_q_sym, kD}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(queries);
  host::TensorMatcher({num_q_sym, kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(norm_w);
  host::TensorMatcher({num_q_sym, T, kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out);

  tvm::ffi::Optional<tvm::ffi::TensorView> const blocks[4] = {b0, b1, b2, b3};
  Ptr4 sp{};
  int32_t const n_stack = verify_stack_blocks(blocks, 4, T, device, sp);

  int64_t const tokens = T.unwrap();
  int64_t const num_q = num_q_sym.unwrap();
  CHECK_HOST(tokens >= 0 && tokens <= std::numeric_limits<int32_t>::max())
      << "attn_res_norm_bf16: invalid token count " << tokens;
  CHECK_HOST(num_q == 1 || num_q == 3) << "attn_res_norm_bf16: requires num_q in {1, 3}, got " << num_q;
  if (tokens == 0) return;

  int64_t const out_q_stride = tokens * kD;
  bf16_t const* add_p = addend.has_value() ? static_cast<bf16_t const*>(addend.value().data_ptr()) : nullptr;

  dim3 const grid(static_cast<uint32_t>(tokens), static_cast<uint32_t>(num_q), 1);
  auto go = [&](auto fn) {
    host::LaunchKernel(grid, kBlock, device.unwrap())
        .enable_pdl(true)(
            fn,
            sp,
            static_cast<bf16_t const*>(partial.data_ptr()),
            add_p,
            static_cast<float const*>(queries.data_ptr()),
            static_cast<bf16_t const*>(norm_w.data_ptr()),
            static_cast<bf16_t*>(out.data_ptr()),
            out_q_stride,
            static_cast<int32_t>(tokens),
            static_cast<float>(eps),
            static_cast<float>(norm_eps));
  };
  if (add_p == nullptr) {
    if (n_stack == 0)
      go(attn_res_norm_1q_kernel<false, 0>);
    else if (n_stack == 1)
      go(attn_res_norm_1q_kernel<false, 1>);
    else if (n_stack == 2)
      go(attn_res_norm_1q_kernel<false, 2>);
    else if (n_stack == 3)
      go(attn_res_norm_1q_kernel<false, 3>);
    else
      go(attn_res_norm_1q_kernel<false, 4>);
  } else {
    if (n_stack == 0)
      go(attn_res_norm_1q_kernel<true, 0>);
    else if (n_stack == 1)
      go(attn_res_norm_1q_kernel<true, 1>);
    else if (n_stack == 2)
      go(attn_res_norm_1q_kernel<true, 2>);
    else if (n_stack == 3)
      go(attn_res_norm_1q_kernel<true, 3>);
    else
      go(attn_res_norm_1q_kernel<true, 4>);
  }
}

/**
 * \brief Stage 1 of looped pre-attention: AttnRes mix + cross-pass message bank + pre-norm concat.
 * \param b0..b2 Optional completed stack blocks [T, 2304] BF16 (1 to 3 blocks).
 * \param partial Current partial block [T, 2304] BF16.
 * \param attn_q Pre-scaled attention queries [1 or 3, 2304] FP32.
 * \param prev0, prev1 Previous-pass message bank blocks [T, 2304] BF16.
 * \param cross_q Pre-scaled cross-pass queries [4, 2304] FP32 (attn, k, v, mlp).
 * \param mix_qkv Output [1 or 3, T, 2304] FP32 mixed destination streams.
 * \param msg_buf Output [4, T, 2304] FP32 cross-pass messages.
 * \param cat_3x Output [3, T, 4608] BF16 normalized [dest, msg] pairs for controller down GEMM.
 * \param is_swa Non-zero for local SWA layers (1 attention query broadcast to 3 controller streams).
 * \param eps RMSNorm epsilon.
 */
inline void pre_attn_mix_cross_s1(
    tvm::ffi::Optional<tvm::ffi::TensorView> b0,
    tvm::ffi::Optional<tvm::ffi::TensorView> b1,
    tvm::ffi::Optional<tvm::ffi::TensorView> b2,
    tvm::ffi::TensorView partial,
    tvm::ffi::TensorView attn_q,
    tvm::ffi::TensorView prev0,
    tvm::ffi::TensorView prev1,
    tvm::ffi::TensorView cross_q,
    tvm::ffi::TensorView mix_qkv,
    tvm::ffi::TensorView msg_buf,
    tvm::ffi::TensorView cat_3x,
    int64_t is_swa,
    double eps) {
  auto T = host::SymbolicSize{"tokens"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();
  int64_t const q_streams = is_swa ? 1 : 3;

  host::TensorMatcher({T, kD})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(partial)
      .verify(prev0)
      .verify(prev1);
  host::TensorMatcher({q_streams, kD}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(attn_q);
  host::TensorMatcher({4, kD}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(cross_q);
  host::TensorMatcher({q_streams, T, kD}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(mix_qkv);
  host::TensorMatcher({4, T, kD}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(msg_buf);
  host::TensorMatcher({3, T, 2 * kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(cat_3x);

  tvm::ffi::Optional<tvm::ffi::TensorView> const blocks[3] = {b0, b1, b2};
  Ptr4 sp{};
  int32_t const n_stack = verify_stack_blocks(blocks, 3, T, device, sp);
  CHECK_HOST(n_stack >= 1 && n_stack <= 3) << "pre_attn_mix_cross_s1: requires 1 <= n_stack <= 3, got " << n_stack;

  int64_t const tokens = T.unwrap();
  CHECK_HOST(tokens >= 0 && tokens <= 32) << "pre_attn_mix_cross_s1: requires 0 <= tokens <= 32, got " << tokens;
  if (tokens == 0) return;

  dim3 const grid(static_cast<uint32_t>(tokens), is_swa ? 5u : 7u, 1u);
  auto go = [&](auto fn) {
    host::LaunchKernel(grid, kBlock, device.unwrap())
        .enable_pdl(true)(
            fn,
            sp,
            static_cast<bf16_t const*>(partial.data_ptr()),
            static_cast<float const*>(attn_q.data_ptr()),
            static_cast<bf16_t const*>(prev0.data_ptr()),
            static_cast<bf16_t const*>(prev1.data_ptr()),
            static_cast<float const*>(cross_q.data_ptr()),
            static_cast<float*>(mix_qkv.data_ptr()),
            static_cast<float*>(msg_buf.data_ptr()),
            static_cast<bf16_t*>(cat_3x.data_ptr()),
            tokens * kD,
            is_swa != 0,
            static_cast<int32_t>(tokens),
            static_cast<float>(eps));
  };
  if (n_stack == 1)
    go(pre_attn_mix_cross_s1_lowm_kernel<1>);
  else if (n_stack == 2)
    go(pre_attn_mix_cross_s1_lowm_kernel<2>);
  else
    go(pre_attn_mix_cross_s1_lowm_kernel<3>);
}

/**
 * \brief Stage 1 of looped pre-MLP: AttnRes mix + message pre-norm concat.
 * \param b0..b2 Optional completed stack blocks [T, 2304] BF16 (1 to 3 blocks).
 * \param partial Current partial block [T, 2304] BF16.
 * \param addend Optional residual addend [T, 2304] BF16.
 * \param mlp_q Pre-scaled MLP query [1, 2304] FP32.
 * \param msg_mlp Cross-pass MLP message [T, 2304] FP32 from pre_attn_mix_cross_s1.
 * \param mix_mlp Output [T, 2304] FP32 mixed destination stream.
 * \param cat_1x Output [1, T, 4608] BF16 normalized [dest, msg] pair.
 * \param eps RMSNorm epsilon.
 */
inline void pre_mlp_mix_s1(
    tvm::ffi::Optional<tvm::ffi::TensorView> b0,
    tvm::ffi::Optional<tvm::ffi::TensorView> b1,
    tvm::ffi::Optional<tvm::ffi::TensorView> b2,
    tvm::ffi::TensorView partial,
    tvm::ffi::Optional<tvm::ffi::TensorView> addend,
    tvm::ffi::TensorView mlp_q,
    tvm::ffi::TensorView msg_mlp,
    tvm::ffi::TensorView mix_mlp,
    tvm::ffi::TensorView cat_1x,
    double eps) {
  auto T = host::SymbolicSize{"tokens"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();

  host::TensorMatcher({T, kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(partial);
  if (addend.has_value()) {
    host::TensorMatcher({T, kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(addend.value());
  }
  host::TensorMatcher({1, kD}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(mlp_q);
  host::TensorMatcher({T, kD})  //
      .with_dtype<fp32_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(msg_mlp)
      .verify(mix_mlp);
  host::TensorMatcher({1, T, 2 * kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(cat_1x);

  tvm::ffi::Optional<tvm::ffi::TensorView> const blocks[3] = {b0, b1, b2};
  Ptr4 sp{};
  int32_t const n_stack = verify_stack_blocks(blocks, 3, T, device, sp);
  CHECK_HOST(n_stack >= 1 && n_stack <= 3) << "pre_mlp_mix_s1: requires 1 <= n_stack <= 3, got " << n_stack;

  int64_t const tokens = T.unwrap();
  CHECK_HOST(tokens >= 0 && tokens <= 32) << "pre_mlp_mix_s1: requires 0 <= tokens <= 32, got " << tokens;
  if (tokens == 0) return;

  bf16_t const* add_p = addend.has_value() ? static_cast<bf16_t const*>(addend.value().data_ptr()) : nullptr;
  dim3 const grid(static_cast<uint32_t>(tokens), 2u, 1u);
  auto go = [&](auto fn) {
    host::LaunchKernel(grid, kBlock, device.unwrap())
        .enable_pdl(true)(
            fn,
            sp,
            static_cast<bf16_t const*>(partial.data_ptr()),
            add_p,
            static_cast<float const*>(mlp_q.data_ptr()),
            static_cast<float const*>(msg_mlp.data_ptr()),
            static_cast<float*>(mix_mlp.data_ptr()),
            static_cast<bf16_t*>(cat_1x.data_ptr()),
            static_cast<int32_t>(tokens),
            static_cast<float>(eps));
  };
  if (add_p == nullptr) {
    if (n_stack == 1)
      go(pre_mlp_mix_s1_kernel<false, 1>);
    else if (n_stack == 2)
      go(pre_mlp_mix_s1_kernel<false, 2>);
    else
      go(pre_mlp_mix_s1_kernel<false, 3>);
  } else {
    if (n_stack == 1)
      go(pre_mlp_mix_s1_kernel<true, 1>);
    else if (n_stack == 2)
      go(pre_mlp_mix_s1_kernel<true, 2>);
    else
      go(pre_mlp_mix_s1_kernel<true, 3>);
  }
}

/**
 * \brief Stages 2-4 of looped controller: down GEMM, forget/gain up GEMMs, tanh merge + RMSNorm.
 * \param cat Normalized concatenated inputs [S, M, 4608] BF16 (32B aligned).
 * \param down Down-projection weights [S, 64, 4608] BF16 (32B aligned).
 * \param gain Gain-projection weights [S, 2304, 64] BF16 (32B aligned).
 * \param forget Forget-projection weights [S, 2304, 64] BF16 (32B aligned).
 * \param in_f32 Destination stream(s) [in_S, M, 2304] FP32 (in_S in {1, S}).
 * \param msg_f32 Cross-pass message streams [S, M, 2304] FP32.
 * \param norm_w Output RMSNorm weights [norm_S, 2304] BF16 (norm_S in {1, S}).
 * \param h_buf Scratch bottleneck buffer [S, M, 64] BF16 (32B aligned).
 * \param fg_buf Scratch forget/gain logits [S, M, 4608] FP32.
 * \param out_bf16 Output normalized streams [S, M, 2304] BF16.
 * \param eps Output RMSNorm epsilon.
 */
inline void ctrl_bottleneck_rmsnorm(
    tvm::ffi::TensorView cat,
    tvm::ffi::TensorView down,
    tvm::ffi::TensorView gain,
    tvm::ffi::TensorView forget,
    tvm::ffi::TensorView in_f32,
    tvm::ffi::TensorView msg_f32,
    tvm::ffi::TensorView norm_w,
    tvm::ffi::TensorView h_buf,
    tvm::ffi::TensorView fg_buf,
    tvm::ffi::TensorView out_bf16,
    double eps) {
  auto S = host::SymbolicSize{"streams"};
  auto M = host::SymbolicSize{"tokens"};
  auto in_S = host::SymbolicSize{"in_streams"};
  auto norm_S = host::SymbolicSize{"norm_streams"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();

  host::TensorMatcher({S, M, 2 * kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(cat);
  host::TensorMatcher({S, kRank, 2 * kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(down);
  host::TensorMatcher({S, kD, kRank})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(32)
      .verify(gain)
      .verify(forget);
  host::TensorMatcher({in_S, M, kD}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(in_f32);
  host::TensorMatcher({S, M, kD}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(msg_f32);
  host::TensorMatcher({norm_S, kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(norm_w);
  host::TensorMatcher({S, M, kRank}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(h_buf);
  host::TensorMatcher({S, M, 2 * kD}).with_dtype<fp32_t>().with_device(device).ensure_alignment(16).verify(fg_buf);
  host::TensorMatcher({S, M, kD}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out_bf16);

  int64_t const streams = S.unwrap();
  int64_t const tokens = M.unwrap();
  int64_t const in_streams = in_S.unwrap();
  int64_t const norm_streams = norm_S.unwrap();
  CHECK_HOST(streams == 1 || streams == 3) << "ctrl_bottleneck_rmsnorm: requires S in {1, 3}, got " << streams;
  CHECK_HOST(tokens >= 0 && tokens <= 32) << "ctrl_bottleneck_rmsnorm: requires 0 <= M <= 32, got " << tokens;
  CHECK_HOST(in_streams == 1 || in_streams == streams) << "ctrl_bottleneck_rmsnorm: in_f32 stream mismatch";
  CHECK_HOST(norm_streams == 1 || norm_streams == streams) << "ctrl_bottleneck_rmsnorm: norm_w stream mismatch";
  if (tokens == 0) return;

  int64_t const in_stride = (in_streams == 1) ? 0 : (tokens * kD);
  int64_t const norm_stride = (norm_streams == 1) ? 0 : kD;
  auto const dev = device.unwrap();
  auto const s_u32 = static_cast<uint32_t>(streams);
  auto const m_i32 = static_cast<int32_t>(tokens);

  if (tokens <= 8) {
    host::LaunchKernel(dim3(kRank / 4, s_u32, 1), 256, dev)
        .enable_pdl(true)(
            ctrl_down_v8_kernel<1, true>,
            static_cast<bf16_t const*>(cat.data_ptr()),
            static_cast<bf16_t const*>(down.data_ptr()),
            static_cast<bf16_t*>(h_buf.data_ptr()),
            m_i32,
            kRank,
            2 * kD);
    host::LaunchKernel(dim3(kD / 64, 2, s_u32), 256, dev)
        .enable_pdl(true)(
            ctrl_up_k64_kernel<1, true>,
            static_cast<bf16_t const*>(h_buf.data_ptr()),
            static_cast<bf16_t const*>(forget.data_ptr()),
            static_cast<bf16_t const*>(gain.data_ptr()),
            static_cast<float*>(fg_buf.data_ptr()),
            m_i32);
  } else if (tokens <= 16) {
    host::LaunchKernel(dim3(kRank / 4, s_u32, 1), 256, dev)
        .enable_pdl(true)(
            ctrl_down_v8_kernel<1, false>,
            static_cast<bf16_t const*>(cat.data_ptr()),
            static_cast<bf16_t const*>(down.data_ptr()),
            static_cast<bf16_t*>(h_buf.data_ptr()),
            m_i32,
            kRank,
            2 * kD);
    host::LaunchKernel(dim3(kD / 64, 2, s_u32), 256, dev)
        .enable_pdl(true)(
            ctrl_up_k64_kernel<1, false>,
            static_cast<bf16_t const*>(h_buf.data_ptr()),
            static_cast<bf16_t const*>(forget.data_ptr()),
            static_cast<bf16_t const*>(gain.data_ptr()),
            static_cast<float*>(fg_buf.data_ptr()),
            m_i32);
  } else {
    host::LaunchKernel(dim3(kRank / 4, s_u32, 1), 256, dev)
        .enable_pdl(true)(
            ctrl_down_v8_kernel<2, false>,
            static_cast<bf16_t const*>(cat.data_ptr()),
            static_cast<bf16_t const*>(down.data_ptr()),
            static_cast<bf16_t*>(h_buf.data_ptr()),
            m_i32,
            kRank,
            2 * kD);
    host::LaunchKernel(dim3(kD / 64, 2, s_u32), 256, dev)
        .enable_pdl(true)(
            ctrl_up_k64_kernel<2, false>,
            static_cast<bf16_t const*>(h_buf.data_ptr()),
            static_cast<bf16_t const*>(forget.data_ptr()),
            static_cast<bf16_t const*>(gain.data_ptr()),
            static_cast<float*>(fg_buf.data_ptr()),
            m_i32);
  }
  host::LaunchKernel(dim3(static_cast<uint32_t>(tokens), s_u32, 1), 256, dev)
      .enable_pdl(true)(
          ctrl_merge_rmsnorm_kernel,
          static_cast<float const*>(in_f32.data_ptr()),
          static_cast<float const*>(msg_f32.data_ptr()),
          static_cast<float const*>(fg_buf.data_ptr()),
          static_cast<bf16_t const*>(norm_w.data_ptr()),
          static_cast<bf16_t*>(out_bf16.data_ptr()),
          in_stride,
          tokens * kD,
          tokens * kD,
          norm_stride,
          m_i32,
          static_cast<float>(eps));
}

}  // namespace sglang::inkling2_norm_ctrl_sm120
