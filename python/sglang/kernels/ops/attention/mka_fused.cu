// Fused full-attention prologue / epilogue for Qwen3.5 bs1 verify (M <= 8
// tokens), SM100+.
//
// prologue: normed hidden (bf16) -> static FP8 quant -> qkv(+gate) GEMM -> q/k
// Gemma RMSNorm -> partial
//           NeoX M-RoPE -> FP8 Q, FP8 K/V into the paged cache (one KV head per
//           rank), bf16 gate.
// epilogue: attention output (bf16) * sigmoid(gate) -> bf16 -> static FP8 ->
// o_proj GEMM -> bf16.
//
// Both kernels give one 8-CTA cluster 256 output rows and split K over the
// cluster ranks. Weights are pre-packed as mma.m16n8k32 e4m3 A fragments and
// loaded into registers before griddepcontrol.wait, so the weight stream
// overlaps the predecessor. Rank r owns token r: every rank pushes its partial
// rows of token r into rank r's inbox (st.async + mbarrier), which keeps a
// whole head local for the q/k norm. Rounding steps copy the stock Triton
// kernels (static quant, ATTN_FRONT, ATTN_GATE_QUANT) instruction for
// instruction; only the FP32 summation order of the two GEMMs and of the norm
// differs.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <sgl_kernel/distributed/q35_hop.cuh>

namespace {

constexpr int kThreads = 256;
constexpr int kCluster = 8;
constexpr int kHeadDim = 256;
constexpr int kHalfRot = 32; // rotary_dim 64

__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t smem_u32(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(out)
               : "r"(addr), "r"(rank));
  return out;
}
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(count)
               : "memory");
}
// Publishes the mbarrier init to the cluster before peers' st.async can credit
// it.
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t *bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 "
      "p, [%0], %1;\n @!p bra WAIT_%=;\n}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void st_async_v4(uint32_t dst, float a, float b,
                                            float c, float d, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
               "[%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
               "f"(a), "f"(b), "f"(c), "f"(d), "r"(bar)
               : "memory");
}
__device__ __forceinline__ uint4 ldg_nc_v4(const uint4 *p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float rsqrt_approx_ftz(float a) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float ex2_approx(float a) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float bf16_round(float a) {
  return __bfloat162float(__float2bfloat16_rn(a));
}
// low byte = e4m3(lo), high byte = e4m3(hi)
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint8_t e4m3(float v) {
  return (uint8_t)(e4m3x2(v, 0.f) & 0xff);
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint4 &a,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// _static_quant_fp8: y * div.full(1, s), clamp +-448, cvt.rn.satfinite.e4m3
__device__ __forceinline__ uint32_t quant4_static(const __nv_bfloat16 *p,
                                                  float inv) {
  const uint2 raw = *reinterpret_cast<const uint2 *>(p);
  const __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162 *>(&raw.x);
  const __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162 *>(&raw.y);
  float f[4] = {__low2float(v01), __high2float(v01), __low2float(v23),
                __high2float(v23)};
#pragma unroll
  for (int i = 0; i < 4; ++i)
    f[i] = fminf(fmaxf(__fmul_rn(f[i], inv), -448.f), 448.f);
  return e4m3x2(f[0], f[1]) | (e4m3x2(f[2], f[3]) << 16);
}

// ATTN_GATE_QUANT: sig = div.full(1, 1 + ex2.approx(g * -log2e)); bf16(sig * o)
// * div.full(1, s); clamp
__device__ __forceinline__ float gate_quant1(float o, float g, float inv) {
  const float e = ex2_approx(__fmul_rn(g, __int_as_float(0xBFB8AA3B)));
  const float sig = div_full(1.0f, __fadd_rn(e, 1.0f));
  const float y = __fmul_rn(inv, bf16_round(__fmul_rn(sig, o)));
  return copysignf(fminf(fabsf(y), 448.f), y);
}
__device__ __forceinline__ uint32_t gate_quant4(const __nv_bfloat16 *o,
                                                const __nv_bfloat16 *g,
                                                float inv) {
  const uint2 ro = *reinterpret_cast<const uint2 *>(o);
  const uint2 rg = *reinterpret_cast<const uint2 *>(g);
  const __nv_bfloat162 o01 = *reinterpret_cast<const __nv_bfloat162 *>(&ro.x);
  const __nv_bfloat162 o23 = *reinterpret_cast<const __nv_bfloat162 *>(&ro.y);
  const __nv_bfloat162 g01 = *reinterpret_cast<const __nv_bfloat162 *>(&rg.x);
  const __nv_bfloat162 g23 = *reinterpret_cast<const __nv_bfloat162 *>(&rg.y);
  return e4m3x2(gate_quant1(__low2float(o01), __low2float(g01), inv),
                gate_quant1(__high2float(o01), __high2float(g01), inv)) |
         (e4m3x2(gate_quant1(__low2float(o23), __low2float(g23), inv),
                 gate_quant1(__high2float(o23), __high2float(g23), inv))
          << 16);
}

// Inbox slot s of a 256-row tile -> row: warp w = s/32 holds rows 32w + {g,
// g+8, g+16, g+24}.
__device__ __forceinline__ int slot_row(int s) {
  return 32 * (s >> 5) + ((s >> 2) & 7) + 8 * (s & 3);
}

struct ProParams {
  const __nv_bfloat16 *x; // [M, 4096]
  const uint4 *w;         // packed qkv(+gate) weights
  const float *sx;
  const float *sw;
  const __nv_bfloat16 *qnw;
  const __nv_bfloat16 *knw;
  const float *cos_sin; // [max_pos, 64] fp32
  const int64_t *pos;   // [3, M]
  const int64_t *axis;  // [32]
  const void *loc;      // [M] cache slots, int64 or int32
  int loc_i32;
  uint8_t *k_cache; // [slots, 256] e4m3
  uint8_t *v_cache;
  const float *k_scale;
  const float *v_scale;
  uint8_t *q_out;          // [M, nq*256] e4m3
  __nv_bfloat16 *gate_out; // [M, nq*256]
  int m;
  int nq;
  int pos_stride;
  int cs_stride;
  float eps;
  int split;       // 1: this launch skips the gate units, which mka_gate_kernel
                   // computes
  const uint4 *wg; // split: gate rows packed 32 rows x full K per CTA
};

// grid = (2*nq + 2) units x 8 ranks; unit u = qkv rows [256u, 256u + 256) (q_h,
// gate_h per head, then k, v); rank r = K chunk [512r, 512r + 512) for the GEMM
// and token r for the epilogue.
template <bool kEarlyTrigger>
__global__ void __launch_bounds__(kThreads, 1)
    mka_prologue_kernel(const ProParams p) {
  __shared__ __align__(16) float inbox[kCluster][256];
  __shared__ __align__(16) uint2 bfr[16][32];
  __shared__ float red8[8];
  __shared__ float nrm[256];
  __shared__ __align__(8) uint64_t ibar;
  const int rank = (int)cluster_ctarank();
  const int nq = p.nq;
  // split launches cover q_0..q_{nq-1}, k, v; natural units interleave q_h and
  // gate_h, then k, v
  const int u = blockIdx.x / kCluster;
  const int unit = !p.split ? u : (u < nq ? 2 * u : 2 * nq + (u - nq));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  if (tid == 0) {
    mbar_init(&ibar, 1);
    mbar_init_fence();
  }
  __syncthreads();
  cluster_arrive_relaxed();

  uint4 a[16][2];
  {
    const uint4 *wb = p.w +
                      (size_t)(unit * kCluster + rank) * (8 * 16 * 2 * 32) +
                      warp * 1024 + lane;
#pragma unroll
    for (int s = 0; s < 16; ++s)
#pragma unroll
      for (int b = 0; b < 2; ++b)
        a[s][b] = ldg_nc_v4(wb + (s * 2 + b) * 32);
  }
  // Static operands only before the wait, and nothing consumes them there.
  const bool is_q = unit < 2 * nq && (unit & 1) == 0;
  const bool is_gate = unit < 2 * nq && (unit & 1) == 1;
  const bool is_k = unit == 2 * nq;
  const bool needs_norm = is_q || is_k;
  const int f = slot_row(tid);
  const bool rot = needs_norm && f < 2 * kHalfRot;
  const int rl = f & (kHalfRot - 1);
  const float sx = __ldg(p.sx), sw = __ldg(p.sw);
  __nv_bfloat16 nwr = __float2bfloat16_rn(0.f);
  if (needs_norm)
    nwr = (is_q ? p.qnw : p.knw)[f];
  int64_t ax = 0;
  if (rot)
    ax = __ldg(p.axis + rl);
  float kvs = 1.f;
  if (is_k)
    kvs = __ldg(p.k_scale);
  if (!needs_norm && !is_gate)
    kvs = __ldg(p.v_scale);

  pdl_wait();
  if (kEarlyTrigger)
    pdl_trigger();
  const float inv_sx = div_full(1.0f, sx);
  const float alpha = __fmul_rn(sx, sw);
  const int m = p.m;
  const bool owner = rank < m;
  int64_t slot = 0;
  float cs_c = 0.f, cs_s = 0.f;
  if (owner) {
    slot = p.loc_i32
               ? (int64_t)__ldg(static_cast<const int32_t *>(p.loc) + rank)
               : __ldg(static_cast<const int64_t *>(p.loc) + rank);
    if (rot) {
      const int64_t ps = __ldg(p.pos + ax * p.pos_stride + rank);
      cs_c = __ldg(p.cos_sin + ps * p.cs_stride + rl);
      cs_s = __ldg(p.cos_sin + ps * p.cs_stride + kHalfRot + rl);
    }
  }
  {
    const bool tok_ok = g < m;
    const __nv_bfloat16 *xr = p.x + (size_t)g * 4096 + rank * 512 + 4 * t;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int s = 2 * warp + i;
      bfr[s][lane] = tok_ok
                         ? make_uint2(quant4_static(xr + s * 32, inv_sx),
                                      quant4_static(xr + s * 32 + 16, inv_sx))
                         : make_uint2(0u, 0u);
    }
  }
  __syncthreads();
  float ch[2][4][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ch[b][c][i] = 0.f;
#pragma unroll
  for (int s = 0; s < 16; ++s) {
    const uint2 bb = bfr[s][lane];
    mma_e4m3(ch[0][s & 3], a[s][0], bb.x, bb.y);
    mma_e4m3(ch[1][s & 3], a[s][1], bb.x, bb.y);
  }
  float acc[2][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int i = 0; i < 4; ++i)
      acc[b][i] = __fadd_rn(__fadd_rn(ch[b][0][i], ch[b][1][i]),
                            __fadd_rn(ch[b][2][i], ch[b][3][i]));

  if (tid == 0 && owner)
    mbar_expect_tx(&ibar, kCluster * 256 * 4);
  cluster_wait();
  {
    const uint32_t src = smem_u32(&inbox[rank][warp * 32 + g * 4]);
    const uint32_t bar = smem_u32(&ibar);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int tok = 2 * t + j;
      if (tok < m)
        st_async_v4(mapa(src, tok), acc[0][j], acc[0][2 + j], acc[1][j],
                    acc[1][2 + j], mapa(bar, tok));
    }
  }
  if (!owner) {
    if (!kEarlyTrigger)
      pdl_trigger();
    return;
  }
  mbar_wait(&ibar,
            0); // also keeps this CTA alive until every peer's push has landed

  float v = 0.f;
#pragma unroll
  for (int r = 0; r < kCluster; ++r)
    v = __fadd_rn(v, inbox[r][tid]);
  v = bf16_round(__fmul_rn(v, alpha));
  const int tok = rank;
  if (is_gate) {
    p.gate_out[(size_t)tok * nq * kHeadDim + (unit >> 1) * kHeadDim + f] =
        __float2bfloat16_rn(v);
  } else if (!needs_norm) {
    p.v_cache[(size_t)slot * kHeadDim + f] =
        e4m3(__fmul_rn(__fdiv_rn(1.0f, kvs), v));
  } else {
    float ss = __fmul_rn(v, v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if (lane == 0)
      red8[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      tot = __fadd_rn(tot, red8[i]);
    const float inv = rsqrt_approx_ftz(__fadd_rn(div_full(tot, 256.0f), p.eps));
    float n = bf16_round(
        __fmul_rn(__fadd_rn(__bfloat162float(nwr), 1.0f), __fmul_rn(inv, v)));
    nrm[f] = n;
    __syncthreads();
    if (rot) {
      // first half: fma(cos, x1, -(sin * x2)); second half: fma(x1, sin, cos *
      // x2)
      const float x1 = f < kHalfRot ? n : nrm[f - kHalfRot];
      const float x2 = f < kHalfRot ? nrm[f + kHalfRot] : n;
      n = f < kHalfRot ? bf16_round(__fmaf_rn(cs_c, x1, -__fmul_rn(cs_s, x2)))
                       : bf16_round(__fmaf_rn(x1, cs_s, __fmul_rn(cs_c, x2)));
    }
    if (is_k)
      p.k_cache[(size_t)slot * kHeadDim + f] =
          e4m3(__fmul_rn(__fdiv_rn(1.0f, kvs), n));
    else
      p.q_out[(size_t)tok * nq * kHeadDim + (unit >> 1) * kHeadDim + f] =
          e4m3(n);
  }
  if (!kEarlyTrigger)
    pdl_trigger();
}

// Gate rows for split launches: no cluster, CTA c owns gate rows [32c, 32c +
// 32) (head c/8) over the full K = 4096; warp w takes K [512w, 512w + 512) and
// the 8 warp partials are summed in warp order. It runs beside the prologue:
// launched when the prologue triggers (after the hop's output is visible), it
// reads x through L2 without its own griddepcontrol.wait and waits for the
// prologue only at the end, so a consumer of this launch also sees the
// prologue's Q / KV.
__global__ void __launch_bounds__(kThreads, 1)
    mka_gate_kernel(const ProParams p) {
  __shared__ __align__(16) float red[8][32][8];
  const int cta = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  pdl_trigger();
  uint4 a[16][2];
  {
    const uint4 *wb = p.wg + ((size_t)cta * 8 + warp) * (16 * 2 * 32) + lane;
#pragma unroll
    for (int s = 0; s < 16; ++s)
#pragma unroll
      for (int b = 0; b < 2; ++b)
        a[s][b] = ldg_nc_v4(wb + (s * 2 + b) * 32);
  }
  const float sx = __ldg(p.sx), sw = __ldg(p.sw);
  const float inv_sx = div_full(1.0f, sx);
  const float alpha = __fmul_rn(sx, sw);
  const int m = p.m;
  const bool tok_ok = g < m;
  const __nv_bfloat16 *xr = p.x + (size_t)g * 4096 + warp * 512 + 4 * t;
  uint2 raw[16][2];
#pragma unroll
  for (int s = 0; s < 16; ++s)
#pragma unroll
    for (int h = 0; h < 2; ++h)
      raw[s][h] =
          tok_ok ? __ldcg(reinterpret_cast<const uint2 *>(xr + s * 32 + h * 16))
                 : make_uint2(0u, 0u);
  float ch[2][4][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ch[b][c][i] = 0.f;
#pragma unroll
  for (int s = 0; s < 16; ++s) {
    const uint32_t b0 = quant4_static(
        reinterpret_cast<const __nv_bfloat16 *>(&raw[s][0]), inv_sx);
    const uint32_t b1 = quant4_static(
        reinterpret_cast<const __nv_bfloat16 *>(&raw[s][1]), inv_sx);
    mma_e4m3(ch[0][s & 3], a[s][0], b0, b1);
    mma_e4m3(ch[1][s & 3], a[s][1], b0, b1);
  }
  {
    float4 *dst = reinterpret_cast<float4 *>(&red[warp][lane][0]);
#pragma unroll
    for (int b = 0; b < 2; ++b)
      dst[b] = make_float4(__fadd_rn(__fadd_rn(ch[b][0][0], ch[b][1][0]),
                                     __fadd_rn(ch[b][2][0], ch[b][3][0])),
                           __fadd_rn(__fadd_rn(ch[b][0][1], ch[b][1][1]),
                                     __fadd_rn(ch[b][2][1], ch[b][3][1])),
                           __fadd_rn(__fadd_rn(ch[b][0][2], ch[b][1][2]),
                                     __fadd_rn(ch[b][2][2], ch[b][3][2])),
                           __fadd_rn(__fadd_rn(ch[b][0][3], ch[b][1][3]),
                                     __fadd_rn(ch[b][2][3], ch[b][3][3])));
  }
  __syncthreads();
  {
    // thread -> fragment lane l = tid/8, element e = tid%8: block e/4, row
    // 16*(e/4) + g' + 8*((e%4)/2), token 2t' + (e%2)
    const int l = tid >> 3, e = tid & 7;
    float v = red[0][l][e];
#pragma unroll
    for (int w = 1; w < 8; ++w)
      v = __fadd_rn(v, red[w][l][e]);
    const int row = 16 * (e >> 2) + (l >> 2) + 8 * ((e & 3) >> 1);
    const int tok = 2 * (l & 3) + (e & 1);
    const int h = cta >> 3, r = (cta & 7) * 32 + row;
    if (tok < m)
      p.gate_out[(size_t)tok * p.nq * kHeadDim + h * kHeadDim + r] =
          __float2bfloat16_rn(__fmul_rn(v, alpha));
  }
  pdl_wait(); // completes only after the prologue grid has completed
}

struct EpiParams {
  const __nv_bfloat16 *attn; // [M, nq*256]
  const __nv_bfloat16 *gate; // [M, nq*256]
  const uint4 *w;            // packed o_proj weights
  const float *sx;
  const float *sw;
  __nv_bfloat16 *out; // [M, 4096]
  int m;
  int push; // 1: also push the bf16 partial into the q35 hop mailbox (q35 hop
            // push)
  q35hop::Site site;
};

// grid = 16 units x 8 ranks; NQ = q heads per rank (K = NQ*256, K chunk NQ*32
// per rank).
template <int NQ, bool kEarlyTrigger>
__global__ void __launch_bounds__(kThreads, 1)
    mka_epilogue_kernel(const EpiParams p) {
  constexpr int kSteps = NQ;
  constexpr int kK = NQ * kHeadDim;
  constexpr int kWarpU4 = kSteps * 2 * 32;
  __shared__ __align__(16) float inbox[kCluster][256];
  __shared__ __align__(16) uint2 bfr[kSteps][32];
  __shared__ __align__(16) __nv_bfloat16 ob[256];
  __shared__ __align__(8) uint64_t ibar;
  const int rank = (int)cluster_ctarank();
  const int unit = blockIdx.x / kCluster;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  if (tid == 0) {
    mbar_init(&ibar, 1);
    mbar_init_fence();
  }
  __syncthreads();
  cluster_arrive_relaxed();

  uint4 a[kSteps][2];
  {
    const uint4 *wb = p.w + (size_t)(unit * kCluster + rank) * (8 * kWarpU4) +
                      warp * kWarpU4 + lane;
#pragma unroll
    for (int s = 0; s < kSteps; ++s)
#pragma unroll
      for (int b = 0; b < 2; ++b)
        a[s][b] = ldg_nc_v4(wb + (s * 2 + b) * 32);
  }
  const float sx = __ldg(p.sx), sw = __ldg(p.sw);

  pdl_wait();
  if (kEarlyTrigger)
    pdl_trigger();
  const float inv_sx = div_full(1.0f, sx);
  const float alpha = __fmul_rn(sx, sw);
  const int m = p.m;
  {
    const bool tok_ok = g < m;
    const size_t off = (size_t)g * kK + rank * (kSteps * 32) + 4 * t;
    uint32_t *bw = reinterpret_cast<uint32_t *>(&bfr[0][0]);
#pragma unroll
    for (int i = warp; i < 2 * kSteps; i += 8) {
      const int s = i >> 1, h = i & 1;
      bw[(s * 32 + lane) * 2 + h] =
          tok_ok ? gate_quant4(p.attn + off + s * 32 + h * 16,
                               p.gate + off + s * 32 + h * 16, inv_sx)
                 : 0u;
    }
  }
  __syncthreads();
  constexpr int kCh = kSteps < 4 ? kSteps : 4;
  float ch[2][kCh][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int c = 0; c < kCh; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ch[b][c][i] = 0.f;
#pragma unroll
  for (int s = 0; s < kSteps; ++s) {
    const uint2 bb = bfr[s][lane];
    mma_e4m3(ch[0][s % kCh], a[s][0], bb.x, bb.y);
    mma_e4m3(ch[1][s % kCh], a[s][1], bb.x, bb.y);
  }
  float acc[2][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float x = ch[b][0][i];
#pragma unroll
      for (int c = 1; c < kCh; ++c)
        x = __fadd_rn(x, ch[b][c][i]);
      acc[b][i] = x;
    }
  const bool owner = rank < m;
  if (tid == 0 && owner)
    mbar_expect_tx(&ibar, kCluster * 256 * 4);
  cluster_wait();
  {
    const uint32_t src = smem_u32(&inbox[rank][warp * 32 + g * 4]);
    const uint32_t bar = smem_u32(&ibar);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int tok = 2 * t + j;
      if (tok < m)
        st_async_v4(mapa(src, tok), acc[0][j], acc[0][2 + j], acc[1][j],
                    acc[1][2 + j], mapa(bar, tok));
    }
  }
  if (!owner) {
    if (!kEarlyTrigger)
      pdl_trigger();
    return;
  }
  mbar_wait(&ibar, 0);
  float v = 0.f;
#pragma unroll
  for (int r = 0; r < kCluster; ++r)
    v = __fadd_rn(v, inbox[r][tid]);
  const __nv_bfloat16 o = __float2bfloat16_rn(__fmul_rn(v, alpha));
  if (p.push) {
    // fragments are 8 consecutive columns, so regroup the row through smem;
    // push before the store
    ob[slot_row(tid)] = o;
    __syncthreads();
    if (tid < 32)
      q35hop::push_frag(p.site, q35hop::stage_peek(p.site), rank,
                        unit * 32 + tid,
                        *reinterpret_cast<const uint4 *>(&ob[8 * tid]));
  }
  p.out[(size_t)rank * 4096 + unit * kHeadDim + slot_row(tid)] = o;
  if (!kEarlyTrigger)
    pdl_trigger();
}

template <typename Kernel, typename Params>
void launch(Kernel kernel, int grid, const Params &params) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = kCluster;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, params));
}

template <typename Kernel, typename Params>
void launch_plain(Kernel kernel, int grid, const Params &params) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, params));
}

} // namespace

void prologue(torch::Tensor x, torch::Tensor w, torch::Tensor sx,
              torch::Tensor sw, torch::Tensor qnw, torch::Tensor knw,
              torch::Tensor cos_sin, torch::Tensor pos, torch::Tensor axis,
              torch::Tensor loc, torch::Tensor k_cache, torch::Tensor v_cache,
              torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor q_out,
              torch::Tensor gate_out, torch::Tensor w_gate, int64_t nq,
              double eps, bool early_trigger) {
  const at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.dim() == 2 && x.size(0) <= 8 && x.size(1) == 4096 &&
              x.is_contiguous());
  TORCH_CHECK(pos.dim() == 2 && pos.size(0) == 3 && pos.stride(1) == 1);
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat &&
              axis.scalar_type() == at::kLong);
  TORCH_CHECK(pos.scalar_type() == at::kLong);
  TORCH_CHECK(k_cache.stride(0) == kHeadDim && v_cache.stride(0) == kHeadDim,
              "one KV head of 256 per slot");
  TORCH_CHECK(loc.scalar_type() == at::kLong || loc.scalar_type() == at::kInt);
  ProParams p;
  p.x = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
  p.w = reinterpret_cast<const uint4 *>(w.data_ptr());
  p.sx = sx.data_ptr<float>();
  p.sw = sw.data_ptr<float>();
  p.qnw = reinterpret_cast<const __nv_bfloat16 *>(qnw.data_ptr());
  p.knw = reinterpret_cast<const __nv_bfloat16 *>(knw.data_ptr());
  p.cos_sin = cos_sin.data_ptr<float>();
  p.pos = pos.data_ptr<int64_t>();
  p.axis = axis.data_ptr<int64_t>();
  p.loc = loc.data_ptr();
  p.loc_i32 = loc.scalar_type() == at::kInt;
  p.k_cache = reinterpret_cast<uint8_t *>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t *>(v_cache.data_ptr());
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.q_out = reinterpret_cast<uint8_t *>(q_out.data_ptr());
  p.gate_out = reinterpret_cast<__nv_bfloat16 *>(gate_out.data_ptr());
  p.m = (int)x.size(0);
  p.nq = (int)nq;
  p.pos_stride = (int)pos.stride(0);
  p.cs_stride = (int)cos_sin.stride(0);
  p.eps = (float)eps;
  p.split = w_gate.numel() > 0;
  p.wg = p.split ? reinterpret_cast<const uint4 *>(w_gate.data_ptr()) : nullptr;
  if (!p.split) {
    const int grid = (2 * (int)nq + 2) * kCluster;
    if (early_trigger)
      launch(mka_prologue_kernel<true>, grid, p);
    else
      launch(mka_prologue_kernel<false>, grid, p);
    return;
  }
  // The gate launch must start while the prologue runs, so the prologue
  // triggers right after its wait.
  launch(mka_prologue_kernel<true>, ((int)nq + 2) * kCluster, p);
  launch_plain(mka_gate_kernel, (int)nq * kHeadDim / 32, p);
}

void epilogue(torch::Tensor attn, torch::Tensor gate, torch::Tensor w,
              torch::Tensor sx, torch::Tensor sw, torch::Tensor out, int64_t nq,
              bool early_trigger, std::vector<int64_t> site) {
  const at::cuda::CUDAGuard guard(attn.device());
  TORCH_CHECK(attn.is_contiguous() && gate.is_contiguous() &&
              out.is_contiguous());
  TORCH_CHECK(attn.size(0) <= 8 && attn.size(1) == nq * kHeadDim);
  EpiParams p;
  p.attn = reinterpret_cast<const __nv_bfloat16 *>(attn.data_ptr());
  p.gate = reinterpret_cast<const __nv_bfloat16 *>(gate.data_ptr());
  p.w = reinterpret_cast<const uint4 *>(w.data_ptr());
  p.sx = sx.data_ptr<float>();
  p.sw = sw.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr());
  p.m = (int)attn.size(0);
  // site = [mc, uc, ctrl, rank, world, m_cap, timeout_ns] of the q35 hop
  // workspace, or empty
  p.push = !site.empty();
  p.site = {};
  if (p.push) {
    TORCH_CHECK(site.size() == 7 && p.m <= site[5]);
    p.site.mc = reinterpret_cast<void *>(site[0]);
    p.site.uc = reinterpret_cast<void *>(site[1]);
    p.site.ctrl = reinterpret_cast<q35hop::Ctrl *>(site[2]);
    p.site.rank = (int)site[3];
    p.site.world = (int)site[4];
    p.site.m_cap = (int)site[5];
    p.site.timeout_ns = (unsigned long long)site[6];
  }
  const int grid = 16 * kCluster;
  if (nq == 4) {
    if (early_trigger)
      launch(mka_epilogue_kernel<4, true>, grid, p);
    else
      launch(mka_epilogue_kernel<4, false>, grid, p);
  } else if (nq == 8) {
    if (early_trigger)
      launch(mka_epilogue_kernel<8, true>, grid, p);
    else
      launch(mka_epilogue_kernel<8, false>, grid, p);
  } else {
    TORCH_CHECK(false, "mka epilogue: nq must be 4 (TP8) or 8 (TP4)");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prologue", &prologue);
  m.def("epilogue", &epilogue);
}
