// Stand-alone decode hop launches built on sgl_kernel/distributed/q35_hop.cuh:
//   publish       push this rank's bf16 partial [M, 4096] (the all-reduce input)
//   publish_fin   finalize the routed MoE rows (top-k weighted, plus the gated shared output)
//                 and push the result
//   consume       pull + rank-order sum + residual + RMSNorm (+ static FP8), re-arm, advance
// Outputs are bit-identical to the FlashInfer MNNVL LL kernels plus the static FP8 quant.
// Built as a shared library and driven through ctypes (see srt/layers/q35_hop.py).

#include "sgl_kernel/distributed/q35_hop.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

using namespace q35hop;

struct Args {
  Site site;
  const uint16_t* partial;      // [M, H] bf16 (publish)
  const uint16_t* routed;       // [M*K, H] bf16 (publish_fin), rows indexed by permuted
  const uint16_t* expert_w;     // [M, K] bf16
  const int32_t* permuted;      // [M, K]; -1 skips a slot
  const uint16_t* shared;       // [M, H] bf16 gated shared expert output, or null
  const uint16_t* residual_in;  // [M, H] bf16
  const uint16_t* gamma;        // [H] bf16, the applied multiplier
  const float* fp8_scale;       // [1], or null
  uint16_t* residual_out;       // [M, H] bf16
  uint16_t* norm_out;           // [M, H] bf16, or null
  uint8_t* fp8_out;             // [M, H] e4m3, or null
  int m, top_k;
  float eps;
  float weight_bias;  // draft consume modes only: multiplier = weight_bias + gamma
  int32_t delay_ns;   // draft consume modes only, diagnostic: hold the output stores this long
};

constexpr int kMaxTopK = 16;

__device__ __forceinline__ uint4 ld_v4(const void* p) {
  return *static_cast<const uint4*>(p);
}

__device__ __forceinline__ size_t row_elem(int token, int frag) {
  return static_cast<size_t>(token) * kHidden + static_cast<size_t>(frag) * kVec;
}

// sum_k w_k * routed[perm_k] + shared, in k order with separate multiply and add, as the
// reference finalize does; every row load is issued before the sum.
__device__ __forceinline__ uint4
finalize_frag(const Args& a, int token, int frag, const int32_t* rows, const float* wts) {
  uint4 src[kMaxTopK];
#pragma unroll
  for (int k = 0; k < kMaxTopK; ++k)
    if (k < a.top_k && rows[k] >= 0) src[k] = ld_v4(a.routed + row_elem(rows[k], frag));
  uint4 sh = make_uint4(0, 0, 0, 0);
  if (a.shared != nullptr) sh = ld_v4(a.shared + row_elem(token, frag));
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e)
    acc[e] = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxTopK; ++k) {
    if (k < a.top_k && rows[k] >= 0) {
      float f[8];
      unpack8(src[k], f);
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[e] = __fadd_rn(acc[e], __fmul_rn(f[e], wts[k]));
    }
  }
  if (a.shared != nullptr) {
    float f[8];
    unpack8(sh, f);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      acc[e] = __fadd_rn(acc[e], f[e]);
  }
  return pack8(acc);
}

// Finalize-publish with 4 bf16 per thread (8 CTAs x 128 threads per token): twice the loads in
// flight of the fragment-wide layout; the per-element sum order is unchanged.
constexpr int kQuadCtas = 8;

// TOPK > 0 fixes top-k at compile time: every routed row is loaded into registers before any
// accumulation (TOPK = 0 reads top_k at run time).
template <int TOPK>
__global__ void __launch_bounds__(kRowThreads) publish_fin_quad_kernel(Args a) {
  constexpr int kSlots = TOPK > 0 ? TOPK : kMaxTopK;
  __shared__ int32_t rows_s[kSlots];
  __shared__ float wts_s[kSlots];
  const int top_k = TOPK > 0 ? TOPK : a.top_k;
  const int token = blockIdx.x;
  const int quad = blockIdx.y * kRowThreads + threadIdx.x;
  if (threadIdx.x < top_k) {
    const int i = token * top_k + threadIdx.x;
    const int r = a.permuted[i];
    rows_s[threadIdx.x] = r;
    wts_s[threadIdx.x] = r < 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(a.expert_w[i]) << 16);
  }
  __syncthreads();
  const size_t col = static_cast<size_t>(quad) * 4;
  pdl_wait();
  const uint32_t stage = stage_peek(a.site);
  // Read after the wait: the side-stream join that produces it may be captured as a programmatic
  // edge, which lets this launch start before the producer has finished.
  uint2 sh = make_uint2(0, 0);
  if (a.shared != nullptr) sh = *reinterpret_cast<const uint2*>(a.shared + static_cast<size_t>(token) * kHidden + col);
  int rows[kSlots];
#pragma unroll
  for (int k = 0; k < kSlots; ++k)
    rows[k] = k < top_k ? rows_s[k] : -1;
  uint2 src[kSlots];
#pragma unroll
  for (int k = 0; k < kSlots; ++k)
    src[k] = rows[k] >= 0 ? *reinterpret_cast<const uint2*>(a.routed + static_cast<size_t>(rows[k]) * kHidden + col)
                          : make_uint2(0, 0);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < kSlots; ++k) {
    if (rows[k] >= 0) {
      const float w = wts_s[k];
      const float f[4] = {bf_lo(src[k].x), bf_hi(src[k].x), bf_lo(src[k].y), bf_hi(src[k].y)};
#pragma unroll
      for (int e = 0; e < 4; ++e)
        acc[e] = __fadd_rn(acc[e], __fmul_rn(f[e], w));
    }
  }
  if (a.shared != nullptr) {
    const float f[4] = {bf_lo(sh.x), bf_hi(sh.x), bf_lo(sh.y), bf_hi(sh.y)};
#pragma unroll
    for (int e = 0; e < 4; ++e)
      acc[e] = __fadd_rn(acc[e], f[e]);
  }
  push_quad(a.site, stage, token, quad, make_uint2(pack_bf16x2(acc[0], acc[1]), pack_bf16x2(acc[2], acc[3])));
  pdl_trigger();
}

template <bool FIN>
__global__ void __launch_bounds__(kRowThreads) publish_kernel(Args a) {
  __shared__ int32_t rows[kMaxTopK];
  __shared__ float wts[kMaxTopK];
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kRowThreads + threadIdx.x;
  if (FIN) {
    // Routing metadata comes from the router, several launches upstream; it is staged before
    // the PDL wait, as the reference finalize does.
    if (threadIdx.x < a.top_k) {
      const int i = token * a.top_k + threadIdx.x;
      const int r = a.permuted[i];
      rows[threadIdx.x] = r;
      wts[threadIdx.x] = r < 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(a.expert_w[i]) << 16);
    }
    __syncthreads();
  }
  pdl_wait();
  const uint32_t stage = stage_peek(a.site);
  const uint4 mine = FIN ? finalize_frag(a, token, frag, rows, wts) : ld_v4(a.partial + row_elem(token, frag));
  push_frag(a.site, stage, token, frag, mine);
  pdl_trigger();
}

template <int W>
__global__ void __launch_bounds__(kRowThreads) consume_kernel(Args a) {
  __shared__ RowSmem rs;
  const int token = blockIdx.x;
  const int frag = blockIdx.y * kRowThreads + threadIdx.x;
  pdl_trigger();
  float gam[8];
  unpack8(ld_v4(a.gamma + static_cast<size_t>(frag) * kVec), gam);
  const float inv_s = a.fp8_out != nullptr ? inv_scale_of(*a.fp8_scale) : 1.f;
  row_init(rs);
  pdl_wait();
  const Hop h = hop_enter(a.site);
  const uint4 res = ld_v4(a.residual_in + row_elem(token, frag));
  float acc[8];
  const bool ok = pull_frag<W>(a.site, h.stage, token, frag, acc);
  float pre[8];
  const uint4 res_out = residual_add(acc, res, pre);
  const float total = row_sumsq_fold(sumsq8(pre), rs);
  uint4 norm = norm_apply(pre, inv_rms(total, a.eps), gam);
  if (!ok) norm = make_uint4(kNanPair, kNanPair, kNanPair, kNanPair);
  *reinterpret_cast<uint4*>(a.residual_out + row_elem(token, frag)) = res_out;
  if (a.norm_out != nullptr) *reinterpret_cast<uint4*>(a.norm_out + row_elem(token, frag)) = norm;
  if (a.fp8_out != nullptr) *reinterpret_cast<uint2*>(a.fp8_out + row_elem(token, frag)) = quant_fp8(norm, inv_s);
  rearm_frag<W>(a.site, h.stage, token, frag);
  hop_leave(a.site, h, gridDim.x * gridDim.y);
}

// ---------------------------------------------------------------- draft-side consume
// The draft keeps the rounding of the stock kernels these replace; only the transport differs.
// kDraftMnnvl: FlashInfer's legacy MNNVL one-shot all-reduce + residual + RMSNorm as it launches for
//   M <= 16 (clusters of 8 x 64 threads, thread t of a row owning columns [8t, 8t + 8)): fp32 rank sum
//   rounded to bf16, bf16 residual add, bf16 squares summed in fp32 per thread, a 16 -> 1 butterfly per
//   warp, the two warps of each 64-thread block, then the 8 blocks in order from 0.
// kDraftAddNorm: sglang's one-shot push all-reduce (fp32 rank sum rounded to bf16), then FlashInfer's
//   CuTe DSL FusedAddRMSNorm for hidden 4096 (64 threads per row, thread t owning the 8-column chunks
//   64b + t, b = 0..7): fp32 residual add, squares accumulated by fma per thread in (b, column) order
//   from 0, a 1 -> 16 butterfly per 32 threads, then the two warps.
constexpr int kDraftMnnvl = 1;
constexpr int kDraftAddNorm = 2;
// Test variant: FusedAddRMSNorm in FlashInfer's CUDA kernel order (512 threads x 8 columns).
constexpr int kDraftAddNormCuda = 3;

__device__ __forceinline__ float add_ftz(float a, float b) {
  float r;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rsqrt_ftz(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

struct DraftSmem {
  // Add-norm: the fp32 values of this CTA's 16 CuTe threads, [chunk b][thread t % 16][column].
  alignas(16) float xv[8][16][kVec];
  float xs[kRowCtas][kRowWarps];  // one value per warp of the row, from every CTA
  float hs[kRowCtas];             // add-norm: every CTA's tree over its 16 CuTe threads
  uint64_t mbar, mbar_x, mbar_h;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t peer_addr(uint32_t local, uint32_t cta) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local), "r"(cta));
  return r;
}
__device__ __forceinline__ void st_async_f32(uint32_t addr, float v, uint32_t mbar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(addr), "f"(v), "r"(mbar)
      : "memory");
}
__device__ __forceinline__ void st_async_v4(uint32_t addr, float a, float b, float c, float d, uint32_t mbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(addr),
               "f"(a),
               "f"(b),
               "f"(c),
               "f"(d),
               "r"(mbar)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(const uint64_t* mbar) {
  asm volatile(
      "{\n\t.reg .pred P;\n\tWAIT_DRAFT:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], 0;\n\t"
      "@!P bra WAIT_DRAFT;\n\t}" ::"r"(smem_u32(mbar))
      : "memory");
}

// Prologue of every CTA of the row cluster, before griddepcontrol.wait: arm the barriers this mode
// receives on and make them visible to the peers before any of them can send.
template <int MODE>
__device__ __forceinline__ void draft_init(DraftSmem& ds) {
  if (threadIdx.x == 0) {
    const auto arm = [](uint64_t* mbar, uint32_t bytes) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(mbar)) : "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(mbar)), "r"(bytes)
                   : "memory");
    };
    if (MODE == kDraftAddNorm) {
      arm(&ds.mbar_x, static_cast<uint32_t>(sizeof(ds.xv)));
      arm(&ds.mbar_h, static_cast<uint32_t>(sizeof(ds.hs)));
    } else {
      arm(&ds.mbar, static_cast<uint32_t>(sizeof(ds.xs)));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Every CTA of the row cluster gets this warp's value v in xs[cta][warp].
__device__ __forceinline__ void row_exchange(float v, DraftSmem& ds) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane < kRowCtas)
    st_async_f32(peer_addr(smem_u32(&ds.xs[cluster_ctarank()][warp]), lane), v, peer_addr(smem_u32(&ds.mbar), lane));
  mbar_wait(&ds.mbar);
}

// Sum of squares in the CuTe DSL FusedAddRMSNorm order. Thread frag = 64b + t sends its 8 values to the
// CTA owning CuTe threads [16c, 16c + 16), where 16 lanes each accumulate one thread's 64 squares; the
// butterfly then pairs (0 + 1) + (2 + 3) ... within those 16, the CTAs as (0 + 1) + (2 + 3).
__device__ __forceinline__ float cute_sumsq(const float x[8], int frag, DraftSmem& ds) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int b = frag >> 6, t = frag & 63;
  const uint32_t dst = static_cast<uint32_t>(t >> 4);
  const uint32_t slot = peer_addr(smem_u32(&ds.xv[b][t & 15][0]), dst);
  const uint32_t mbar = peer_addr(smem_u32(&ds.mbar_x), dst);
  st_async_v4(slot, x[0], x[1], x[2], x[3], mbar);
  st_async_v4(slot + 16, x[4], x[5], x[6], x[7], mbar);
  mbar_wait(&ds.mbar_x);
  if (warp == 0) {
    float sum = 0.f;
    if (lane < 16) {
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float4 lo = *reinterpret_cast<const float4*>(&ds.xv[c][lane][0]);
        const float4 hi = *reinterpret_cast<const float4*>(&ds.xv[c][lane][4]);
        sum = __fmaf_rn(lo.x, lo.x, sum);
        sum = __fmaf_rn(lo.y, lo.y, sum);
        sum = __fmaf_rn(lo.z, lo.z, sum);
        sum = __fmaf_rn(lo.w, lo.w, sum);
        sum = __fmaf_rn(hi.x, hi.x, sum);
        sum = __fmaf_rn(hi.y, hi.y, sum);
        sum = __fmaf_rn(hi.z, hi.z, sum);
        sum = __fmaf_rn(hi.w, hi.w, sum);
      }
    }
#pragma unroll
    for (int off = 1; off < 16; off <<= 1)
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, off));
    if (lane < kRowCtas)
      st_async_f32(peer_addr(smem_u32(&ds.hs[cluster_ctarank()]), lane), sum, peer_addr(smem_u32(&ds.mbar_h), lane));
  }
  mbar_wait(&ds.mbar_h);
  return __fadd_rn(__fadd_rn(ds.hs[0], ds.hs[1]), __fadd_rn(ds.hs[2], ds.hs[3]));
}

// EARLY releases the dependents at entry, as consume_kernel does; they read only static data before
// their own griddepcontrol.wait, which returns once this grid has completed.
template <int W, int MODE, bool EARLY = false>
__global__ void __launch_bounds__(kRowThreads) consume_draft_kernel(Args a) {
  __shared__ DraftSmem ds;
  const int token = blockIdx.x;
  const int frag = static_cast<int>(blockIdx.y) * kRowThreads + threadIdx.x;
  if (EARLY) pdl_trigger();
  float gam[8];
  unpack8(ld_v4(a.gamma + static_cast<size_t>(frag) * kVec), gam);
  draft_init<MODE>(ds);
  pdl_wait();
  const Hop h = hop_enter(a.site);
  const uint4 res = ld_v4(a.residual_in + row_elem(token, frag));
  float acc[8];
  const bool ok = pull_frag<W>(a.site, h.stage, token, frag, acc);
  float res_f[8];
  unpack8(res, res_f);
  float x[8];
  uint4 res_out;
  float total;
  if (MODE == kDraftMnnvl) {
    float part = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const __nv_bfloat16 r = __hadd(__float2bfloat16_rn(acc[e]), __float2bfloat16_rn(res_f[e]));
      x[e] = __bfloat162float(r);
      part = add_ftz(part, __bfloat162float(__hmul(r, r)));
    }
    res_out = pack8(x);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      part = add_ftz(part, __shfl_xor_sync(0xffffffffu, part, off));
    row_exchange(part, ds);
    // Global warp 2k + i is xs[k / 2][(k % 2) * 2 + i]; 64-thread block k sums its two warps.
    total = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k)
      total = add_ftz(total, add_ftz(ds.xs[k >> 1][(k & 1) * 2], ds.xs[k >> 1][(k & 1) * 2 + 1]));
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e)
      x[e] = __fadd_rn(__bfloat162float(__float2bfloat16_rn(acc[e])), res_f[e]);
    res_out = pack8(x);
    if (MODE == kDraftAddNorm) {
      total = cute_sumsq(x, frag, ds);
    } else {
      float part = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        part = __fmaf_rn(x[e], x[e], part);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, off));
      row_exchange(part, ds);
      float v[16];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        v[i] = ds.xs[i >> 2][i & 3];
#pragma unroll
      for (int w = 8; w > 0; w >>= 1)
#pragma unroll
        for (int i = 0; i < w; ++i)
          v[i] = __fadd_rn(v[i], v[i + w]);
      total = v[0];
    }
  }
  const float inv = rsqrt_ftz(add_ftz(mul_ftz(total, 1.f / static_cast<float>(kHidden)), a.eps));
  float out[8];
#pragma unroll
  for (int e = 0; e < 8; ++e)
    out[e] = __fmul_rn(__fmul_rn(x[e], inv), __fadd_rn(a.weight_bias, gam[e]));
  uint4 norm = pack8(out);
  if (!ok) norm = make_uint4(kNanPair, kNanPair, kNanPair, kNanPair);
  if (a.delay_ns > 0) {
    const unsigned long long t0 = gtimer();
    while (gtimer() - t0 < static_cast<unsigned long long>(a.delay_ns)) {
    }
  }
  *reinterpret_cast<uint4*>(a.residual_out + row_elem(token, frag)) = res_out;
  *reinterpret_cast<uint4*>(a.norm_out + row_elem(token, frag)) = norm;
  if (!EARLY) pdl_trigger();
  rearm_frag<W>(a.site, h.stage, token, frag);
  hop_leave(a.site, h, gridDim.x * gridDim.y);
}

// ---------------------------------------------------------------- out_proj + push
// Row-parallel FP8 out_proj (M <= 8, N = 4096, K = 8 * S * 32) whose epilogue pushes the bf16 partial
// into the hop mailbox. The MMA core follows the o_proj kernel of the fused attention epilogue: CTA c owns
// output rows [32c, 32c + 32) over the full K, warp w takes k-steps [w*S, (w+1)*S), weights are
// packed as mma.m16n8k32 e4m3 A fragments and loaded to registers before the PDL wait, and the
// warp partials are summed in warp order.
struct OprojArgs {
  Site site;
  const uint4* w;    // packed [128 cta][8 warp][S step][2 block][32 lane] x 16 B
  const uint8_t* x;  // [M, K] e4m3 activation
  const float* sx;   // input scale
  const float* sw;   // weight scale
  uint16_t* out;     // [M, 4096] bf16 copy of the partial, or null
  int m;
};

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg_nc_v4(const uint4* q) {
  uint4 r;
  // Weights are static for the life of the process.
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(q));
  return r;
}

template <int S>
__global__ void __launch_bounds__(256, 1) oproj_push_kernel(OprojArgs o) {
  constexpr int kK = 8 * S * 32;
  __shared__ __align__(16) float red[8][32][8];
  const int cta = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  uint4 a[S][2];
  {
    const uint4* wb = o.w + (static_cast<size_t>(cta) * 8 + warp) * (S * 2 * 32) + lane;
#pragma unroll
    for (int st = 0; st < S; ++st)
#pragma unroll
      for (int b = 0; b < 2; ++b)
        a[st][b] = ldg_nc_v4(wb + (st * 2 + b) * 32);
  }
  const float alpha = __fmul_rn(*o.sx, *o.sw);
  pdl_wait();
  const int m = o.m;
  const bool tok_ok = g < m;
  const uint8_t* xr = o.x + static_cast<size_t>(g) * kK + warp * S * 32 + 4 * t;
  float ch[2][2][4] = {};
#pragma unroll
  for (int st = 0; st < S; ++st) {
    const uint32_t b0 = tok_ok ? *reinterpret_cast<const uint32_t*>(xr + st * 32) : 0u;
    const uint32_t b1 = tok_ok ? *reinterpret_cast<const uint32_t*>(xr + st * 32 + 16) : 0u;
    mma_e4m3(ch[0][st & 1], a[st][0], b0, b1);
    mma_e4m3(ch[1][st & 1], a[st][1], b0, b1);
  }
  {
    float4* dst = reinterpret_cast<float4*>(&red[warp][lane][0]);
    dst[0] = make_float4(
        __fadd_rn(ch[0][0][0], ch[0][1][0]),
        __fadd_rn(ch[0][0][1], ch[0][1][1]),
        __fadd_rn(ch[0][0][2], ch[0][1][2]),
        __fadd_rn(ch[0][0][3], ch[0][1][3]));
    dst[1] = make_float4(
        __fadd_rn(ch[1][0][0], ch[1][1][0]),
        __fadd_rn(ch[1][0][1], ch[1][1][1]),
        __fadd_rn(ch[1][0][2], ch[1][1][2]),
        __fadd_rn(ch[1][0][3], ch[1][1][3]));
  }
  __syncthreads();
  // Thread (tok, j) owns rows [8j, 8j + 8) of token tok. Accumulator element (row r, token k) sits at
  // lane (r % 8) * 4 + k / 2, entry (r / 16) * 4 + ((r % 16) / 8) * 2 + k % 2 of the mma layout.
  const int tok = tid >> 2, j = tid & 3;
  if (tok < m) {
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const int r = 8 * j + q;
      const int l = (r % 8) * 4 + tok / 2;
      const int e = (r / 16) * 4 + ((r % 16) / 8) * 2 + (tok % 2);
      float x = red[0][l][e];
#pragma unroll
      for (int w = 1; w < 8; ++w)
        x = __fadd_rn(x, red[w][l][e]);
      v[q] = __fmul_rn(x, alpha);
    }
    const uint4 frag = pack8(v);
    const int fr = cta * 4 + j;
    push_frag(o.site, stage_peek(o.site), tok, fr, frag);
    if (o.out != nullptr) *reinterpret_cast<uint4*>(o.out + row_elem(tok, fr)) = frag;
  }
  pdl_trigger();
}

template <typename K>
int launch(K kernel, const Args& a, cudaStream_t stream, bool cluster, int ctas = kRowCtas) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(a.m, ctas, 1);
  cfg.blockDim = dim3(kRowThreads, 1, 1);
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  int n = 0;
  attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[n].val.programmaticStreamSerializationAllowed = 1;
  ++n;
  if (cluster) {
    attrs[n].id = cudaLaunchAttributeClusterDimension;
    attrs[n].val.clusterDim.x = 1;
    attrs[n].val.clusterDim.y = kRowCtas;
    attrs[n].val.clusterDim.z = 1;
    ++n;
  }
  cfg.attrs = attrs;
  cfg.numAttrs = n;
  return static_cast<int>(cudaLaunchKernelEx(&cfg, kernel, a));
}

template <int W>
int launch_draft(int which, const Args& a, cudaStream_t s) {
  if (which == 3) return launch(consume_draft_kernel<W, kDraftMnnvl>, a, s, true);
  if (which == 4) return launch(consume_draft_kernel<W, kDraftAddNorm>, a, s, true);
  if (which == 7) return launch(consume_draft_kernel<W, kDraftMnnvl, true>, a, s, true);
  if (which == 8) return launch(consume_draft_kernel<W, kDraftAddNorm, true>, a, s, true);
  return launch(consume_draft_kernel<W, kDraftAddNormCuda>, a, s, true);
}

}  // namespace

// which: 0 publish, 1 publish_fin, 2 consume, 3 / 4 draft consume with MNNVL / add-norm rounding, 7 / 8 the
// same releasing their dependents at entry; 6 (test) the add-norm in the CUDA FusedAddRMSNorm order.
// The C entry takes an opaque pointer: Args lives in an unnamed namespace, and a C-linkage
// function whose signature names it would not be exported.
extern "C" int q35hop_launch(int which, const void* args, void* stream) {
  const Args& a = *static_cast<const Args*>(args);
  if (a.m < 1 || a.m > a.site.m_cap || a.top_k > kMaxTopK) return -2;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (which) {
    case 0:
      return launch(publish_kernel<false>, a, s, false);
    case 1:
      return a.top_k == 10 ? launch(publish_fin_quad_kernel<10>, a, s, false, kQuadCtas)
                           : launch(publish_fin_quad_kernel<0>, a, s, false, kQuadCtas);
    case 2:
      switch (a.site.world) {
        case 2:
          return launch(consume_kernel<2>, a, s, true);
        case 4:
          return launch(consume_kernel<4>, a, s, true);
        case 8:
          return launch(consume_kernel<8>, a, s, true);
        default:
          return -3;
      }
    case 3:
    case 4:
    case 6:
    case 7:
    case 8:
      if (a.norm_out == nullptr || a.fp8_out != nullptr) return -4;
      switch (a.site.world) {
        case 2:
          return launch_draft<2>(which, a, s);
        case 4:
          return launch_draft<4>(which, a, s);
        case 8:
          return launch_draft<8>(which, a, s);
        default:
          return -3;
      }
    default:
      return -1;
  }
}

extern "C" int q35hop_oproj(int k_steps, const void* args, void* stream) {
  const OprojArgs& o = *static_cast<const OprojArgs*>(args);
  if (o.m < 1 || o.m > 8 || o.m > o.site.m_cap) return -2;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kHidden / 32, 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.stream = static_cast<cudaStream_t>(stream);
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  switch (k_steps) {
    case 4:
      return static_cast<int>(cudaLaunchKernelEx(&cfg, oproj_push_kernel<4>, o));
    case 8:
      return static_cast<int>(cudaLaunchKernelEx(&cfg, oproj_push_kernel<8>, o));
    default:
      return -3;
  }
}

extern "C" int q35hop_args_size() {
  return static_cast<int>(sizeof(Args));
}
extern "C" int q35hop_oproj_args_size() {
  return static_cast<int>(sizeof(OprojArgs));
}
