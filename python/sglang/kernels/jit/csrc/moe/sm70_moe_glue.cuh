// SPDX-License-Identifier: Apache-2.0
// Decode-time MoE glue on Volta: the fp32 router projection from fp16 operands,
// and the clamped SwiGLU between the routed gate_up and down GEMVs.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace sglang::sm70_moe_glue {

inline constexpr int kWarp = 32;
inline constexpr int kVec = 8;
inline constexpr int kLogitsWarps = 8;
inline constexpr int kMaxTokens = 4;

// out[m, n] = sum_k x[m, k] * w[n, k] in fp32; fp16 products are exact in fp32.
// One warp per expert row, eight accumulators per lane, fixed-order reduction.
template <int kTokens>
__global__ __launch_bounds__(kLogitsWarps* kWarp) void router_logits_kernel(
    float* __restrict__ out, const __half* __restrict__ x, const __half* __restrict__ w, int num_experts, int hidden) {
  const int row = blockIdx.x * kLogitsWarps + static_cast<int>(threadIdx.x) / kWarp;
  const int lane = static_cast<int>(threadIdx.x) % kWarp;
  if (row >= num_experts) {
    return;
  }
  float acc[kTokens][kVec] = {};
  const __half* wr = w + static_cast<int64_t>(row) * hidden;
#pragma unroll 4
  for (int k = lane * kVec; k < hidden; k += kWarp * kVec) {
    const int4 wv = *reinterpret_cast<const int4*>(wr + k);
    const __half* hw = reinterpret_cast<const __half*>(&wv);
#pragma unroll
    for (int m = 0; m < kTokens; ++m) {
      const int4 xv = *reinterpret_cast<const int4*>(x + static_cast<int64_t>(m) * hidden + k);
      const __half* hx = reinterpret_cast<const __half*>(&xv);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        acc[m][i] = fmaf(__half2float(hx[i]), __half2float(hw[i]), acc[m][i]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < kTokens; ++m) {
    float sum = acc[m][0];
#pragma unroll
    for (int i = 1; i < kVec; ++i) {
      sum = __fadd_rn(sum, acc[m][i]);
    }
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1) {
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, off));
    }
    if (lane == 0) {
      out[static_cast<int64_t>(m) * num_experts + row] = sum;
    }
  }
}

SGL_DEVICE __half clamp_max_nan(__half v, __half hi) {
  return __hisnan(v) ? v : (__hlt(hi, v) ? hi : v);
}

SGL_DEVICE __half clamp_nan(__half v, __half lo, __half hi) {
  return __hisnan(v) ? v : clamp_max_nan(__hlt(v, lo) ? lo : v, hi);
}

// Matches swiglu_limit_func: fp16 clamps, torch's silu (x / (1 + expf(-x)) in
// fp32, rounded to fp16), then the fp16 product rounded once.
__global__ void swiglu_clamp_kernel(
    __half* __restrict__ out, const __half* __restrict__ gate_up, int64_t rows, int d, __half limit, bool clamp) {
  const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVec;
  if (idx >= rows * d) {
    return;
  }
  const int64_t r = idx / d;
  const int c = static_cast<int>(idx % d);
  const int4 gv = *reinterpret_cast<const int4*>(gate_up + r * 2 * d + c);
  const int4 uv = *reinterpret_cast<const int4*>(gate_up + r * 2 * d + d + c);
  const __half* hg = reinterpret_cast<const __half*>(&gv);
  const __half* hu = reinterpret_cast<const __half*>(&uv);
  const __half neg_limit = __hneg(limit);
  int4 ov;
  __half* ho = reinterpret_cast<__half*>(&ov);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const __half g = clamp ? clamp_max_nan(hg[i], limit) : hg[i];
    const __half u = clamp ? clamp_nan(hu[i], neg_limit, limit) : hu[i];
    const float gf = __half2float(g);
    const float s = __fdiv_rn(gf, __fadd_rn(1.0f, expf(-gf)));
    ho[i] = __float2half_rn(__fmul_rn(__half2float(__float2half_rn(s)), __half2float(u)));
  }
  *reinterpret_cast<int4*>(out + idx) = ov;
}

// Matches moe_sum_reduce_triton (fp32 sum over routes in order, times the
// scale, rounded to fp16) followed by torch's fp16 `routed += shared`.
__global__ void sum_routes_add_shared_kernel(
    __half* __restrict__ out, const __half* __restrict__ rows, int tokens, int topk, int hidden, float routed_scale) {
  const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVec;
  if (idx >= static_cast<int64_t>(tokens) * hidden) {
    return;
  }
  const int t = static_cast<int>(idx / hidden);
  const int h = static_cast<int>(idx % hidden);
  float acc[kVec] = {};
  for (int i = 0; i < topk; ++i) {
    const int4 v = *reinterpret_cast<const int4*>(rows + (static_cast<int64_t>(t) * topk + i) * hidden + h);
    const __half* hv = reinterpret_cast<const __half*>(&v);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      acc[j] = __fadd_rn(acc[j], __half2float(hv[j]));
    }
  }
  const int4 sv =
      *reinterpret_cast<const int4*>(rows + (static_cast<int64_t>(tokens) * topk + t) * hidden + h);
  const __half* hs = reinterpret_cast<const __half*>(&sv);
  int4 ov;
  __half* ho = reinterpret_cast<__half*>(&ov);
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    const __half routed = __float2half_rn(__fmul_rn(acc[j], routed_scale));
    ho[j] = __float2half_rn(__fadd_rn(__half2float(routed), __half2float(hs[j])));
  }
  *reinterpret_cast<int4*>(out + idx) = ov;
}

/// \brief fp32 router logits for decode batches.
/// \param out fp32 [M, N]; x fp16 [M, K]; w fp16 [N, K]; all contiguous, 1 <= M <= 4, K % 8 == 0.
void router_logits(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView w) {
  using namespace host;
  SymbolicSize m = {"num_tokens"};
  SymbolicSize n = {"num_experts"};
  SymbolicSize k = {"hidden"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({m, k}).with_dtype<fp16_t>().with_device(device_).verify(x);
  TensorMatcher({n, k}).with_dtype<fp16_t>().with_device(device_).verify(w);
  TensorMatcher({m, n}).with_dtype<fp32_t>().with_device(device_).verify(out);
  const int64_t tokens = m.unwrap();
  const int64_t hidden = k.unwrap();
  RuntimeCheck(tokens >= 1 && tokens <= kMaxTokens, "router_logits covers 1..4 tokens");
  RuntimeCheck(hidden % kVec == 0, "router_logits needs K % 8 == 0");
  RuntimeCheck(x.is_contiguous() && w.is_contiguous() && out.is_contiguous(), "router_logits needs contiguous tensors");
  RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
               "router_logits operands must be 16-byte aligned");
  const int experts = static_cast<int>(n.unwrap());
  const auto blocks = static_cast<uint32_t>((experts + kLogitsWarps - 1) / kLogitsWarps);
  auto* op = static_cast<float*>(out.data_ptr());
  const auto* xp = static_cast<const __half*>(x.data_ptr());
  const auto* wp = static_cast<const __half*>(w.data_ptr());
  const int kk = static_cast<int>(hidden);
  const DLDevice device = device_.unwrap();
  switch (tokens) {
    case 1:
      LaunchKernel(blocks, kLogitsWarps * kWarp, device)(router_logits_kernel<1>, op, xp, wp, experts, kk);
      break;
    case 2:
      LaunchKernel(blocks, kLogitsWarps * kWarp, device)(router_logits_kernel<2>, op, xp, wp, experts, kk);
      break;
    case 3:
      LaunchKernel(blocks, kLogitsWarps * kWarp, device)(router_logits_kernel<3>, op, xp, wp, experts, kk);
      break;
    default:
      LaunchKernel(blocks, kLogitsWarps * kWarp, device)(router_logits_kernel<4>, op, xp, wp, experts, kk);
      break;
  }
}

/// \brief out = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit); no clamp when limit <= 0.
/// \param out fp16 [R, D]; gate_up fp16 [R, 2 * D] (gate first), contiguous, D % 8 == 0.
void swiglu_clamp(tvm::ffi::TensorView out, tvm::ffi::TensorView gate_up, double limit) {
  using namespace host;
  SymbolicSize r = {"rows"};
  SymbolicSize d2 = {"2d"};
  SymbolicSize d = {"d"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({r, d2}).with_dtype<fp16_t>().with_device(device_).verify(gate_up);
  TensorMatcher({r, d}).with_dtype<fp16_t>().with_device(device_).verify(out);
  const int64_t rows = r.unwrap();
  const int64_t dim = d.unwrap();
  RuntimeCheck(d2.unwrap() == 2 * dim && dim % kVec == 0, "swiglu_clamp needs gate_up [R, 2D] with D % 8 == 0");
  RuntimeCheck(gate_up.is_contiguous() && out.is_contiguous(), "swiglu_clamp needs contiguous tensors");
  if (rows == 0) {
    return;
  }
  constexpr int kThreads = 256;
  const int64_t vecs = rows * dim / kVec;
  const auto blocks = static_cast<uint32_t>((vecs + kThreads - 1) / kThreads);
  // torch.clamp converts the scalar bound to the tensor dtype first.
  const __half lim = __float2half_rn(static_cast<float>(limit));
  LaunchKernel(blocks, kThreads, device_.unwrap())(
      swiglu_clamp_kernel,
      static_cast<__half*>(out.data_ptr()),
      static_cast<const __half*>(gate_up.data_ptr()),
      rows,
      static_cast<int>(dim),
      lim,
      limit > 0.0);
}

/// \brief out[t] = fp16(sum_i rows[t * topk + i] * routed_scale) + rows[tokens * topk + t].
/// \param out fp16 [T, H]; rows fp16 [T * topk + T, H] (routed rows, then one shared row per token),
///        contiguous, H % 8 == 0.
void sum_routes_add_shared(tvm::ffi::TensorView out, tvm::ffi::TensorView rows, int64_t topk, double routed_scale) {
  using namespace host;
  SymbolicSize t = {"tokens"};
  SymbolicSize r = {"rows"};
  SymbolicSize h = {"hidden"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({t, h}).with_dtype<fp16_t>().with_device(device_).verify(out);
  TensorMatcher({r, h}).with_dtype<fp16_t>().with_device(device_).verify(rows);
  const int64_t tokens = t.unwrap();
  const int64_t hidden = h.unwrap();
  RuntimeCheck(topk >= 1 && r.unwrap() == tokens * (topk + 1), "rows must be [T * (topk + 1), H]");
  RuntimeCheck(hidden % kVec == 0, "sum_routes_add_shared needs H % 8 == 0");
  RuntimeCheck(out.is_contiguous() && rows.is_contiguous(), "sum_routes_add_shared needs contiguous tensors");
  if (tokens == 0) {
    return;
  }
  constexpr int kThreads = 128;
  const int64_t vecs = tokens * hidden / kVec;
  const auto blocks = static_cast<uint32_t>((vecs + kThreads - 1) / kThreads);
  LaunchKernel(blocks, kThreads, device_.unwrap())(
      sum_routes_add_shared_kernel,
      static_cast<__half*>(out.data_ptr()),
      static_cast<const __half*>(rows.data_ptr()),
      static_cast<int>(tokens),
      static_cast<int>(topk),
      static_cast<int>(hidden),
      static_cast<float>(routed_scale));
}

}  // namespace sglang::sm70_moe_glue
