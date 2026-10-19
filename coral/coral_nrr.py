"""Residual add + Gemma RMSNorm + MoE router GEMV + top-k routing table + shared-expert gate in ONE CUDA launch
(M = 1..8 tokens). Replaces the fused norm+router Triton launch AND the 1-warp route launch per MoE layer.

64 CTAs x 8 warps. Every CTA recomputes add+norm for all M tokens (y kept in smem as bf16, exactly the operand the
router reads) and computes 4 router rows for all tokens (warp = (row, K-half)). Logits are rounded to bf16 (what the
separate route kernel reads). The last CTA to arrive (atomic counter) builds the routing table: one warp per token,
softmax over the E=256 logits, 8 rounds of warp-shuffle argmax (ties -> lower expert id, like torch.topk), weights
renormalized over the top-k, plus sigmoid(y . shared_gate_w) in slot TOPK -- the layout coral_kernels'
_moe_up_ns_kernel / _moe_down_tab_kernel consume. CTA 0 writes h = x + residual and y."""
import os
import torch
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

constexpr int K = 2048, E = 256, TOPK = 8, NS = TOPK + 1, ROWS = 4, WARPS = 8, MAXM = 8;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__global__ void __launch_bounds__(WARPS * 32) nrr_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ res, const __nv_bfloat16* __restrict__ wn,
    const __nv_bfloat16* __restrict__ wr, const __nv_bfloat16* __restrict__ sg, __nv_bfloat16* __restrict__ y_out,
    __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ lg, int* __restrict__ cnt, int* __restrict__ eid,
    float* __restrict__ wt, int M, float eps) {
  __shared__ __align__(16) __nv_bfloat16 ys[MAXM][K];
  __shared__ float red[WARPS][MAXM];
  __shared__ float part[2][ROWS][MAXM];
  __shared__ int is_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  // ---- phase 1: h = x + res, rstd, y (8 contiguous elements per thread per token)
  const int k8 = tid * 8;
  float wv[8];
  {
    uint4 u = *reinterpret_cast<const uint4*>(wn + k8);
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
    for (int i = 0; i < 8; ++i) wv[i] = 1.f + __bfloat162float(b[i]);
  }
  float hv[MAXM][8];
#pragma unroll
  for (int m = 0; m < MAXM; ++m) {
    float ss = 0.f;
    if (m < M) {
      uint4 ux = *reinterpret_cast<const uint4*>(x + (size_t)m * K + k8);
      uint4 ur = *reinterpret_cast<const uint4*>(res + (size_t)m * K + k8);
      const __nv_bfloat16* bx = reinterpret_cast<const __nv_bfloat16*>(&ux);
      const __nv_bfloat16* br = reinterpret_cast<const __nv_bfloat16*>(&ur);
#pragma unroll
      for (int i = 0; i < 8; ++i) { hv[m][i] = __bfloat162float(bx[i]) + __bfloat162float(br[i]); ss += hv[m][i] * hv[m][i]; }
    }
    ss = warp_sum(ss);
    if (lane == 0) red[warp][m] = ss;
  }
  __syncthreads();
#pragma unroll
  for (int m = 0; m < MAXM; ++m) {
    if (m >= M) break;
    float tot = 0.f;
#pragma unroll
    for (int q = 0; q < WARPS; ++q) tot += red[q][m];
    const float rstd = rsqrtf(tot / K + eps);
    uint4 oy, oh;
    __nv_bfloat16* by = reinterpret_cast<__nv_bfloat16*>(&oy);
    __nv_bfloat16* bh = reinterpret_cast<__nv_bfloat16*>(&oh);
#pragma unroll
    for (int i = 0; i < 8; ++i) { by[i] = __float2bfloat16(hv[m][i] * rstd * wv[i]); bh[i] = __float2bfloat16(hv[m][i]); }
    *reinterpret_cast<uint4*>(&ys[m][k8]) = oy;
    if (blockIdx.x == 0) {
      *reinterpret_cast<uint4*>(y_out + (size_t)m * K + k8) = oy;
      *reinterpret_cast<uint4*>(h_out + (size_t)m * K + k8) = oh;
    }
  }
  __syncthreads();
  // ---- phase 2: router rows r0..r0+3 for all tokens; warp = (row = warp & 3, half = warp >> 2)
  {
    const int row = r0 + (warp & 3), half = warp >> 2;
    const int kb = half * (K / 2) + lane * 32;  // 32 contiguous k per lane
    float w32[32];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      uint4 u = *reinterpret_cast<const uint4*>(wr + (size_t)row * K + kb + c * 8);
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
      for (int i = 0; i < 8; ++i) w32[c * 8 + i] = __bfloat162float(b[i]);
    }
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
      if (m >= M) break;
      float d = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint4 u = *reinterpret_cast<const uint4*>(&ys[m][kb + c * 8]);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
        for (int i = 0; i < 8; ++i) d += w32[c * 8 + i] * __bfloat162float(b[i]);
      }
      d = warp_sum(d);
      if (lane == 0) part[half][warp & 3][m] = d;
    }
  }
  __syncthreads();
  if (tid < ROWS * MAXM) {
    const int rr = tid / MAXM, m = tid % MAXM;
    if (m < M) lg[(size_t)m * E + r0 + rr] = __float2bfloat16(part[0][rr][m] + part[1][rr][m]);
  }
  // ---- phase 3: last CTA builds the routing table
  __threadfence();
  __syncthreads();
  if (tid == 0) is_last = (atomicAdd(cnt, 1) == (int)gridDim.x - 1);
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  if (warp < M) {
    const int m = warp;
    const volatile __nv_bfloat16* lp = lg + (size_t)m * E;
    float p[8]; int id[8];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      id[i] = i * 32 + lane;
      __nv_bfloat16 v = *(const __nv_bfloat16*)(lp + id[i]);
      p[i] = __bfloat162float(v); mx = fmaxf(mx, p[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { p[i] = __expf(p[i] - mx); sum += p[i]; }
    sum = warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 8; ++i) p[i] /= sum;
    float sel_w[TOPK]; int sel_i[TOPK]; float wsum = 0.f;
#pragma unroll
    for (int t = 0; t < TOPK; ++t) {
      float bv = -1.f; int bi = 1 << 30;
#pragma unroll
      for (int i = 0; i < 8; ++i) if (p[i] > bv || (p[i] == bv && id[i] < bi)) { bv = p[i]; bi = id[i]; }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      sel_w[t] = bv; sel_i[t] = bi; wsum += bv;
#pragma unroll
      for (int i = 0; i < 8; ++i) if (id[i] == bi) p[i] = -1.f;
    }
    // shared-expert gate: sigmoid(y . sg)
    float d = 0.f;
    for (int k = lane; k < K; k += 32) d += __bfloat162float(ys[m][k]) * __bfloat162float(sg[k]);
    d = warp_sum(d);
    if (lane < TOPK) {
      float w = 0.f; int ii = 0;
#pragma unroll
      for (int t = 0; t < TOPK; ++t) if (t == lane) { w = sel_w[t]; ii = sel_i[t]; }
      eid[m * NS + lane] = ii;
      wt[m * NS + lane] = w / wsum;
    }
    if (lane == 0) { eid[m * NS + TOPK] = 0; wt[m * NS + TOPK] = 1.f / (1.f + __expf(-d)); }
  }
  __syncthreads();
  if (tid == 0) *cnt = 0;
}

void nrr(torch::Tensor x, torch::Tensor res, torch::Tensor wn, torch::Tensor wr, torch::Tensor sg, torch::Tensor y,
         torch::Tensor h, torch::Tensor lg, torch::Tensor cnt, torch::Tensor eid, torch::Tensor wt, double eps) {
  const int M = x.size(0);
  TORCH_CHECK(M >= 1 && M <= MAXM && x.size(1) == K && wr.size(0) == E && wr.size(1) == K, "bad shape");
  auto bp = [](const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  auto bw = [](torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); };
  nrr_kernel<<<E / ROWS, WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      bp(x), bp(res), bp(wn), bp(wr), bp(sg), bw(y), bw(h), bw(lg), cnt.data_ptr<int>(), eid.data_ptr<int>(),
      wt.data_ptr<float>(), M, (float)eps);
}
"""
_CPP = ("void nrr(torch::Tensor x, torch::Tensor res, torch::Tensor wn, torch::Tensor wr, torch::Tensor sg, "
        "torch::Tensor y, torch::Tensor h, torch::Tensor lg, torch::Tensor cnt, torch::Tensor eid, torch::Tensor wt, "
        "double eps);")
_ext = None
_CNT = {}


def _load():
    global _ext
    if _ext is None:
        d = os.environ.get("CORAL_EXT_DIR_NRR", "/data/ext_build/coral_nrr")
        os.makedirs(d, exist_ok=True)
        _ext = load_inline("coral_nrr", _CPP, cuda_sources=_SRC, functions=["nrr"], build_directory=d,
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"])
    return _ext


def add_norm_router_route_cuda(x, residual, norm_w, router_w, shared_gate_w, eps):
    """-> (y, h, logits bf16 [M, 256], eid int32 [M*9], wt fp32 [M*9]). New output tensors; x/residual untouched."""
    M, K = x.shape
    dev = x.device
    cnt = _CNT.get(dev)
    if cnt is None:
        cnt = _CNT[dev] = torch.zeros(1, dtype=torch.int32, device=dev)
    y = torch.empty_like(x)
    h = torch.empty_like(residual)
    lg = torch.empty(M, router_w.shape[0], dtype=torch.bfloat16, device=dev)
    eid = torch.empty(M * 9, dtype=torch.int32, device=dev)
    wt = torch.empty(M * 9, dtype=torch.float32, device=dev)
    _load().nrr(x.contiguous(), residual.contiguous(), norm_w, router_w, shared_gate_w, y, h, lg, cnt, eid, wt, eps)
    return y, h, lg, eid, wt
