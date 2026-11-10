// SM70 decode-focused FP8 (E4M3) W8A16 GEMM: y[M,N] = x[M,K] @ W[K,N]
//
// Adapted from v100-skinny's QPN8 kernel (https://github.com/dnv2003/v100-skinny,
// MIT License) by the SGLang V100 team, 2026. The QPN8 execution architecture
// streams packed FP8 weights straight from HBM and decodes them into the fp16
// register operands mma.sync.m8n8k4 requires (no persistent fp16 copy, no
// shared-memory staging in the main loop, one cross-warp K-reduce at output).
//
// Differences from upstream: this checkpoint family (Qwen3.8-27B-FP8) uses
// block-wise [N/128][K/128] fp32 scales rather than ModelOpt's per-slice
// scalar scales, so the per-K-128-group scale is applied inside the K loop
// instead of a single epilogue scale per 32-col tile. Everything else follows
// the upstream kernel: [tile=N/32][group=K/16][lane=32][16B] fragment-order
// codes produced by the same permutation (_KORDER8), four quadpairs on N
// sharing one activation tile, NACC accumulators, split-K across WARPS.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#define DEV_INLINE __device__ __forceinline__

// Byte-pair e4m3 -> two fp16 lanes. Reproduces value/256 with a fixed
// exponent re-bias folded into the caller's per-group scale (sg*256).
DEV_INLINE half2 fp8x8_to_half2x4(unsigned char b0, unsigned char b1) {
  const unsigned h0 = ((b0 & 0x80u) << 8) | ((b0 & 0x7Fu) << 7);
  const unsigned h1 = ((b1 & 0x80u) << 8) | ((b1 & 0x7Fu) << 7);
  const unsigned p = h0 | (h1 << 16);
  return *reinterpret_cast<const half2 *>(&p);
}

// Word-parallel e4m3 -> fp16 decode (FASTDEC): pairs (x byte i, y byte i) so
// the (i, i+4) interleave of _KORDER8 cancels inside the fragment.
DEV_INLINE void fp8x8_to_half2x4_fast(const uint2 q, half2 out[4]) {
  constexpr unsigned S = 0x80008000u, EM = 0x3F803F80u;
  unsigned p[4];
  p[0] = __byte_perm(q.x, q.y, 0x0400);
  p[1] = __byte_perm(q.x, q.y, 0x0501);
  p[2] = __byte_perm(q.x, q.y, 0x0602);
  p[3] = __byte_perm(q.x, q.y, 0x0703);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const unsigned v = ((p[i] << 8) & S) | ((p[i] << 7) & EM);
    out[i] = *reinterpret_cast<const half2 *>(&v);
  }
}

#define MMA_8N8K4(C, A0, A1, B0, B1)                                        \
  asm volatile(                                                             \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                        \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),         \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                                  \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// gscales: half[K/128][N/32], one fp16 (scale * 256) per (K-128-group, N-32-tile).
template <int SPLITK, int NACC, bool FASTDEC = false>
__global__ void sm70_fp8_qpn8_k(const uint8_t *__restrict__ bcodes,
                                const half *__restrict__ gscales,
                                const half *__restrict__ x,
                                half *__restrict__ y, int N, int K, int M) {
  __shared__ float cs[SPLITK > 1 ? SPLITK : 1][SPLITK > 1 ? 256 : 1];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq;
  const int tiles = N / 32;
  const uint4 *cb = reinterpret_cast<const uint4 *>(bcodes) +
                    (size_t)tile * G * 32 + lane;
  const half *gs = gscales + tile;  // stride (N/32) per k-block

  float c[NACC][8];
#pragma unroll
  for (int a = 0; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[a][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint4 q4 = __ldcs(cb + (size_t)g * 32);
    half2 b[8];
    if (FASTDEC) {
      fp8x8_to_half2x4_fast(make_uint2(q4.x, q4.y), b + 0);
      fp8x8_to_half2x4_fast(make_uint2(q4.z, q4.w), b + 4);
    } else {
      const unsigned char *bb = reinterpret_cast<const unsigned char *>(&q4);
#pragma unroll
      for (int i = 0; i < 4; i++) b[i] = fp8x8_to_half2x4(bb[i], bb[4 + i]);
#pragma unroll
      for (int i = 0; i < 4; i++)
        b[4 + i] = fp8x8_to_half2x4(bb[8 + i], bb[12 + i]);
    }
    const half sg = __ldg(gs + (size_t)(g >> 3) * tiles);
    const half2 sg2 = make_half2(sg, sg);
#pragma unroll
    for (int i = 0; i < 8; i++) b[i] = __hmul2(b[i], sg2);
    const unsigned *B = reinterpret_cast<const unsigned *>(b);
    uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
    if (r < M) {
      const half *xrow = x + (size_t)r * K;
      a01 = *reinterpret_cast<const uint4 *>(xrow + g * 16);
      a23 = *reinterpret_cast<const uint4 *>(xrow + g * 16 + 8);
    }
    const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
    const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
    MMA_8N8K4(c[0], A0[0], A0[1], B[0], B[1]);
    MMA_8N8K4(c[1 % NACC], A0[2], A0[3], B[2], B[3]);
    MMA_8N8K4(c[2 % NACC], A1[0], A1[1], B[4], B[5]);
    MMA_8N8K4(c[3 % NACC], A1[2], A1[3], B[6], B[7]);
  }

#pragma unroll
  for (int a = 1; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[0][i] += c[a][i];

  if (SPLITK == 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (row < M)
        y[(size_t)row * N + (size_t)tile * 32 + qp * 8 + col] = __float2half(c[0][i]);
    }
    return;
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    cs[warp][row * 32 + qp * 8 + col] = c[0][i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += blockDim.x) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < SPLITK; w++) v += cs[w][e];
    const int row = e >> 5, col = e & 31;
    if (row < M)
      y[(size_t)row * N + (size_t)tile * 32 + col] = __float2half(v);
  }
}

// Decode-only gate/up projection for Qwen's SiLU MLP. Each CTA owns matching
// 32-column gate and up tiles, so the activation fragment is loaded once and
// the 2*N projection is never materialized in HBM. The reduction deliberately
// rounds gate/up to fp16 before SiLU and rounds SiLU before multiplication to
// preserve the ordinary fp16 linear -> silu -> multiply contract.
template <int SPLITK>
__global__ void sm70_fp8_qpn8_gated_silu_k(
    const uint8_t *__restrict__ bcodes, const half *__restrict__ gscales,
    const half *__restrict__ x, half *__restrict__ y, int OUT_N, int K,
    int M) {
  __shared__ float cs[2][SPLITK][256];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gate_tile = blockIdx.x;
  const int out_tiles = OUT_N / 32;
  const int up_tile = gate_tile + out_tiles;
  const int total_tiles = out_tiles * 2;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int G = K >> 4, Gq = G / SPLITK;
  const int g0 = warp * Gq;
  const uint4 *codes4 = reinterpret_cast<const uint4 *>(bcodes);
  const uint4 *cb_gate = codes4 + (size_t)gate_tile * G * 32 + lane;
  const uint4 *cb_up = codes4 + (size_t)up_tile * G * 32 + lane;
  const half *gs_gate = gscales + gate_tile;
  const half *gs_up = gscales + up_tile;

  float cg[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float cu[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint4 qg = __ldcs(cb_gate + (size_t)g * 32);
    const uint4 qu = __ldcs(cb_up + (size_t)g * 32);
    half2 bg[8], bu[8];
    fp8x8_to_half2x4_fast(make_uint2(qg.x, qg.y), bg + 0);
    fp8x8_to_half2x4_fast(make_uint2(qg.z, qg.w), bg + 4);
    fp8x8_to_half2x4_fast(make_uint2(qu.x, qu.y), bu + 0);
    fp8x8_to_half2x4_fast(make_uint2(qu.z, qu.w), bu + 4);
    const half s_gate = __ldg(gs_gate + (size_t)(g >> 3) * total_tiles);
    const half s_up = __ldg(gs_up + (size_t)(g >> 3) * total_tiles);
    const half2 s_gate2 = make_half2(s_gate, s_gate);
    const half2 s_up2 = make_half2(s_up, s_up);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      bg[i] = __hmul2(bg[i], s_gate2);
      bu[i] = __hmul2(bu[i], s_up2);
    }
    const unsigned *BG = reinterpret_cast<const unsigned *>(bg);
    const unsigned *BU = reinterpret_cast<const unsigned *>(bu);
    uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
    if (r < M) {
      const half *xrow = x + (size_t)r * K;
      a01 = *reinterpret_cast<const uint4 *>(xrow + g * 16);
      a23 = *reinterpret_cast<const uint4 *>(xrow + g * 16 + 8);
    }
    const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
    const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
    MMA_8N8K4(cg, A0[0], A0[1], BG[0], BG[1]);
    MMA_8N8K4(cu, A0[0], A0[1], BU[0], BU[1]);
    MMA_8N8K4(cg, A0[2], A0[3], BG[2], BG[3]);
    MMA_8N8K4(cu, A0[2], A0[3], BU[2], BU[3]);
    MMA_8N8K4(cg, A1[0], A1[1], BG[4], BG[5]);
    MMA_8N8K4(cu, A1[0], A1[1], BU[4], BU[5]);
    MMA_8N8K4(cg, A1[2], A1[3], BG[6], BG[7]);
    MMA_8N8K4(cu, A1[2], A1[3], BU[6], BU[7]);
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    const int e = row * 32 + qp * 8 + col;
    cs[0][warp][e] = cg[i];
    cs[1][warp][e] = cu[i];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 256; e += blockDim.x) {
    float gate = 0.f, up = 0.f;
#pragma unroll
    for (int w = 0; w < SPLITK; w++) {
      gate += cs[0][w][e];
      up += cs[1][w][e];
    }
    const int row = e >> 5, col = e & 31;
    if (row < M) {
      const half gate_h = __float2half_rn(gate);
      const half up_h = __float2half_rn(up);
      const float gate_f = __half2float(gate_h);
      const half silu_h =
          __float2half_rn(gate_f / (1.f + expf(-gate_f)));
      y[(size_t)row * OUT_N + (size_t)gate_tile * 32 + col] =
          __hmul(silu_h, up_h);
    }
  }
}

torch::Tensor sm70_fp8_qpn8_linear(torch::Tensor x, torch::Tensor codes,
                                   torch::Tensor gscales, int64_t n,
                                   int64_t k, int64_t splitk, int64_t nacc) {
  const int64_t m = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
              x.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == torch::kUInt8 &&
              codes.is_contiguous());
  TORCH_CHECK(gscales.is_cuda() && gscales.scalar_type() == torch::kFloat16 &&
              gscales.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 8, "qpn8 supports M 1..8, got ", m);
  TORCH_CHECK(k % 64 == 0 && (k / 16) % splitk == 0, "K/SPLITK");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(codes.numel() == n * k, "qpn8 codes size");
  TORCH_CHECK(gscales.numel() == (k / 128) * (n / 32), "qpn8 group scales size");
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_QPN8_F(SPv, NAv)                                               \
  sm70_fp8_qpn8_k<SPv, NAv, true>                                             \
      <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                   \
          codes.data_ptr<uint8_t>(),                                           \
          reinterpret_cast<const half *>(gscales.data_ptr<at::Half>()),        \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
          (int)m)

#define LAUNCH_QPN8(SPv, NAv)                                                 \
  sm70_fp8_qpn8_k<SPv, NAv>                                                   \
      <<<dim3((int)(n / 32)), dim3(32 * SPv), 0, stream>>>(                   \
          codes.data_ptr<uint8_t>(),                                           \
          reinterpret_cast<const half *>(gscales.data_ptr<at::Half>()),        \
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
          (int)m)

  const int key = (int)(splitk * 10 + nacc);
  switch (key) {
    case 43: LAUNCH_QPN8_F(4, 1); break;
    case 83: LAUNCH_QPN8_F(8, 1); break;
    case 84: LAUNCH_QPN8_F(8, 2); break;
    case 163: LAUNCH_QPN8_F(16, 1); break;
    case 164: LAUNCH_QPN8_F(16, 2); break;
    case 323: LAUNCH_QPN8_F(32, 1); break;
    case 324: LAUNCH_QPN8_F(32, 2); break;
    case 41: LAUNCH_QPN8(4, 1); break;
    case 42: LAUNCH_QPN8(4, 2); break;
    case 81: LAUNCH_QPN8(8, 1); break;
    case 82: LAUNCH_QPN8(8, 2); break;
    case 161: LAUNCH_QPN8(16, 1); break;
    case 162: LAUNCH_QPN8(16, 2); break;
    case 321: LAUNCH_QPN8(32, 1); break;
    case 322: LAUNCH_QPN8(32, 2); break;
    default:
      TORCH_CHECK(false, "qpn8 splitk in {4,8,16,32}, nacc in {1,2} (+2 = fast decoder)");
  }
#undef LAUNCH_QPN8
#undef LAUNCH_QPN8_F
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor sm70_fp8_qpn8_gated_silu(torch::Tensor x, torch::Tensor codes,
                                       torch::Tensor gscales, int64_t out_n,
                                       int64_t k) {
  const int64_t m = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 &&
              x.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == torch::kUInt8 &&
              codes.is_contiguous());
  TORCH_CHECK(gscales.is_cuda() && gscales.scalar_type() == torch::kFloat16 &&
              gscales.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 8, "qpn8 gated SiLU supports M 1..8, got ", m);
  TORCH_CHECK(k % 256 == 0, "qpn8 gated SiLU requires K % 256 == 0");
  TORCH_CHECK(out_n % 32 == 0, "qpn8 gated SiLU requires OUT_N % 32 == 0");
  TORCH_CHECK(codes.numel() == 2 * out_n * k,
              "qpn8 gated SiLU codes size");
  TORCH_CHECK(gscales.numel() == (k / 128) * (2 * out_n / 32),
              "qpn8 gated SiLU group scales size");
  auto y = torch::empty({m, out_n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  sm70_fp8_qpn8_gated_silu_k<16>
      <<<dim3((int)(out_n / 32)), dim3(32 * 16), 0, stream>>>(
          codes.data_ptr<uint8_t>(),
          reinterpret_cast<const half *>(gscales.data_ptr<at::Half>()),
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)out_n,
          (int)k, (int)m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qpn8_linear", &sm70_fp8_qpn8_linear,
        "SM70 decode FP8 E4M3 GEMM (QPN8, M<=8, per-K-128-group scales)");
  m.def("qpn8_gated_silu", &sm70_fp8_qpn8_gated_silu,
        "SM70 decode fused FP8 gate/up GEMM and SiLU multiply (M<=8)");
}
