// SPDX-License-Identifier: Apache-2.0
// gfx950 Kimi-K3 KDA decode: f_b projection, conv1d update, gated delta-rule
// recurrence and gated RMSNorm fused into one workgroup per (token, head).
//
// Each 256-thread workgroup (12 heads of 128 per rank):
//   1. issues the f_b rows, conv window, gate and norm loads, then the fp32 state
//   2. f_b projection (16 lanes per row) and depthwise conv1d + SiLU of q, k, v
//   3. q / k L2 norms, gate decay and the delta-rule update of the 128 x 128 state
//   4. gated RMSNorm of the output row
// conv_state and state at state_indices are updated in place. Slot 0 is the
// padding slot of the mamba pool: its rows are written as zeros and its state
// is left untouched.

#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHeads = 12;
constexpr int kDim = 128;
constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 64;
constexpr float kQScale = 0.08838834764831845f;  // 1 / sqrt(kDim)

// Slot strides of the hybrid cache may be padded, so state rows are only
// guaranteed to be dword aligned.
typedef float f32x4 __attribute__((ext_vector_type(4), aligned(4)));
typedef float f32x4a __attribute__((ext_vector_type(4)));
typedef uint16_t u16x8 __attribute__((ext_vector_type(8)));
typedef uint16_t u16x8u __attribute__((ext_vector_type(8), aligned(2)));

__device__ __forceinline__ float bf2f(uint16_t x) {
  return __builtin_bit_cast(float, static_cast<uint32_t>(x) << 16);
}

__device__ __forceinline__ uint16_t f2bf(float x) {
  return __builtin_bit_cast(uint16_t, static_cast<__bf16>(x));
}

__device__ __forceinline__ float round_bf16(float x) {
  return bf2f(f2bf(x));
}

__device__ __forceinline__ float sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float dot4(f32x4a a, f32x4a b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

template <int kCtrl>
__device__ __forceinline__ float dpp(float x) {
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), kCtrl, 0xf, 0xf, false));
}

// Every lane receives the sum over its aligned group of 8 (16) lanes.
__device__ __forceinline__ float sum8(float x) {
  x += dpp<0xb1>(x);         // quad_perm(1, 0, 3, 2)
  x += dpp<0x4e>(x);         // quad_perm(2, 3, 0, 1)
  return x + dpp<0x141>(x);  // row_half_mirror
}

__device__ __forceinline__ float sum16(float x) {
  x = sum8(x);
  return x + dpp<0x140>(x);  // row_mirror
}

__device__ __forceinline__ float conv_silu(f32x4a w, uint16_t c0, uint16_t c1, uint16_t c2, uint16_t x) {
  float acc = bf2f(c0) * w[0];
  acc += bf2f(c1) * w[1];
  acc += bf2f(c2) * w[2];
  acc += bf2f(x) * w[3];
  return round_bf16(acc / (1.0f + __expf(-acc)));
}

__global__ void __launch_bounds__(kThreads) kda_decode_gfx950_kernel(
    const uint16_t* f_a,
    int64_t f_a_stride,
    const uint16_t* __restrict__ f_b,
    const uint16_t* __restrict__ x,
    int64_t x_stride,
    const float* __restrict__ conv_weight,
    uint16_t* __restrict__ conv_state,
    int64_t conv_slot_stride,
    int64_t conv_channel_stride,
    int64_t conv_width_stride,
    const uint16_t* __restrict__ raw_beta,
    int64_t beta_stride,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ state,
    int64_t state_slot_stride,
    const int32_t* __restrict__ state_indices,
    const uint16_t* __restrict__ output_gate,
    int64_t gate_stride,
    const uint16_t* __restrict__ norm_weight,
    uint16_t* __restrict__ out,
    int64_t out_stride,
    float lower_bound,
    float norm_eps) {
  __shared__ __attribute__((aligned(16))) float q_s[kDim];
  __shared__ __attribute__((aligned(16))) float k_s[kDim];
  __shared__ __attribute__((aligned(16))) float g_s[kDim];
  __shared__ float v_s[kDim];
  __shared__ float o_s[kDim];
  __shared__ float sq_s[kWaves];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wave = tid >> 6;
  const int batch = blockIdx.x / kHeads;
  const int head = blockIdx.x % kHeads;
  const int64_t slot = state_indices[batch];
  if (slot <= 0) {
    if (tid < kDim) out[batch * out_stride + head * kDim + tid] = 0;
    return;
  }

  // Front loads are issued before the state so their waits do not cover it.
  // Projection: 16 lanes read one 128-wide f_b row, 8 bf16 per lane.
  const int proj_k = (lane & 15) * 8;
  const int proj_row = wave * 32 + (lane >> 4);
  u16x8 w[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    w[i] = *reinterpret_cast<const u16x8*>(f_b + (head * kDim + proj_row + 4 * i) * kDim + proj_k);
  }
  const u16x8 fa = *reinterpret_cast<const u16x8u*>(f_a + batch * f_a_stride + proj_k);

  // Convolution: threads 0..127 own q and v channels, 128..255 own k channels.
  // All threads load the v channel so no divergent branch delays the state
  // loads below; only the owners use and update it.
  const int c = tid & (kDim - 1);
  const int ch_a = (tid < kDim ? 0 : kHeads * kDim) + head * kDim + c;
  const int ch_v = 2 * kHeads * kDim + head * kDim + c;
  uint16_t* cs_a = conv_state + slot * conv_slot_stride + ch_a * conv_channel_stride;
  uint16_t* cs_v = conv_state + slot * conv_slot_stride + ch_v * conv_channel_stride;
  const f32x4a w_a = *reinterpret_cast<const f32x4a*>(conv_weight + ch_a * 4);
  const uint16_t a0 = cs_a[0];
  const uint16_t a1 = cs_a[conv_width_stride];
  const uint16_t a2 = cs_a[2 * conv_width_stride];
  const uint16_t x_a = x[batch * x_stride + ch_a];
  const f32x4a w_v = *reinterpret_cast<const f32x4a*>(conv_weight + ch_v * 4);
  const uint16_t v0 = cs_v[0];
  const uint16_t v1 = cs_v[conv_width_stride];
  const uint16_t v2 = cs_v[2 * conv_width_stride];
  const uint16_t x_v = x[batch * x_stride + ch_v];

  // Recurrence layout: 8 lanes split one 128-wide state row (4 floats x 4),
  // each wave holds 8 rows per step and 4 steps cover the 128 V rows.
  const int lk = lane & 7;
  const int row = wave * 8 + (lane >> 3);
  f32x4a dt[4];
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    dt[ki] = *reinterpret_cast<const f32x4a*>(dt_bias + head * kDim + lk * 4 + 32 * ki);
  }
  const float a_log = A_log[head];
  const uint16_t beta_raw = raw_beta[batch * beta_stride + head];
  const uint16_t gate_raw = output_gate[batch * gate_stride + head * kDim + c];
  const uint16_t norm_raw = norm_weight[c];

  // vmcnt retires loads in issue order, so the front loads must issue first.
  asm volatile("" ::: "memory");
  float* st = state + slot * state_slot_stride + head * kDim * kDim + row * kDim + lk * 4;
  f32x4a s[4][4];
#pragma unroll
  for (int vi = 0; vi < 4; ++vi) {
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      s[vi][ki] = *reinterpret_cast<const f32x4*>(st + vi * 32 * kDim + ki * 32);
    }
  }
  __builtin_amdgcn_sched_barrier(0);

  float fa_f[8];
#pragma unroll
  for (int e = 0; e < 8; ++e)
    fa_f[e] = bf2f(fa[e]);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float acc = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      acc += fa_f[e] * bf2f(w[i][e]);
    acc = sum16(acc);
    if ((lane & 15) == 0) g_s[proj_row + 4 * i] = round_bf16(acc);
  }

  const float y_a = conv_silu(w_a, a0, a1, a2, x_a);
  cs_a[0] = a1;
  cs_a[conv_width_stride] = a2;
  cs_a[2 * conv_width_stride] = x_a;
  if (tid < kDim) {
    q_s[c] = y_a;
    v_s[c] = conv_silu(w_v, v0, v1, v2, x_v);
    cs_v[0] = v1;
    cs_v[conv_width_stride] = v2;
    cs_v[2 * conv_width_stride] = x_v;
  } else {
    k_s[c] = y_a;
  }
  __syncthreads();

  const float a = __expf(a_log);
  f32x4a q[4], k[4], decay[4];
  float qq = 0.0f, kk = 0.0f;
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    const int idx = lk * 4 + 32 * ki;
    q[ki] = *reinterpret_cast<const f32x4a*>(q_s + idx);
    k[ki] = *reinterpret_cast<const f32x4a*>(k_s + idx);
    const f32x4a g = *reinterpret_cast<const f32x4a*>(g_s + idx);
#pragma unroll
    for (int e = 0; e < 4; ++e)
      decay[ki][e] = __expf(lower_bound * sigmoid(a * (g[e] + dt[ki][e])));
    qq += dot4(q[ki], q[ki]);
    kk += dot4(k[ki], k[ki]);
  }
  const float q_scale = rsqrtf(sum8(qq) + 1e-6f) * kQScale;
  const float k_scale = rsqrtf(sum8(kk) + 1e-6f);
  float kq = 0.0f;
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    q[ki] *= q_scale;
    k[ki] *= k_scale;
    kq += dot4(k[ki], q[ki]);
  }
  kq = sum8(kq);
  const float beta = sigmoid(bf2f(beta_raw));

  float o_sq = 0.0f;
#pragma unroll
  for (int vi = 0; vi < 4; ++vi) {
    float hk = 0.0f, hq = 0.0f;
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      s[vi][ki] *= decay[ki];
      hk += dot4(s[vi][ki], k[ki]);
      hq += dot4(s[vi][ki], q[ki]);
    }
    hk = sum8(hk);
    hq = sum8(hq);
    const int r = row + 32 * vi;
    const float v_new = beta * (v_s[r] - hk);
    const float o = round_bf16(hq + v_new * kq);
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      s[vi][ki] += v_new * k[ki];
      *reinterpret_cast<f32x4*>(st + vi * 32 * kDim + ki * 32) = s[vi][ki];
    }
    if (lk == 0) {
      o_s[r] = o;
      o_sq += o * o;
    }
  }
  o_sq = sum16(o_sq);
  o_sq += __shfl_xor(o_sq, 16);
  o_sq += __shfl_xor(o_sq, 32);
  if (lane == 0) sq_s[wave] = o_sq;
  __syncthreads();

  if (tid < kDim) {
    const float inv_rms = rsqrtf((sq_s[0] + sq_s[1] + sq_s[2] + sq_s[3]) * (1.0f / kDim) + norm_eps);
    out[batch * out_stride + head * kDim + tid] = f2bf(o_s[tid] * inv_rms * bf2f(norm_raw) * sigmoid(bf2f(gate_raw)));
  }
}

}  // namespace

void kda_decode_gfx950(
    torch::Tensor f_a,
    torch::Tensor f_b_weight,
    torch::Tensor mixed_qkv,
    torch::Tensor conv_weight,
    torch::Tensor conv_state,
    torch::Tensor raw_beta,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor state,
    torch::Tensor state_indices,
    torch::Tensor output_gate,
    torch::Tensor norm_weight,
    torch::Tensor out,
    double lower_bound,
    double norm_eps) {
  const int64_t batch = f_a.size(0);
  if (batch == 0) return;
  hipLaunchKernelGGL(
      kda_decode_gfx950_kernel,
      dim3(static_cast<unsigned>(batch * kHeads)),
      dim3(kThreads),
      0,
      at::hip::getCurrentHIPStream(),
      reinterpret_cast<const uint16_t*>(f_a.data_ptr()),
      f_a.stride(0),
      reinterpret_cast<const uint16_t*>(f_b_weight.data_ptr()),
      reinterpret_cast<const uint16_t*>(mixed_qkv.data_ptr()),
      mixed_qkv.stride(0),
      conv_weight.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(conv_state.data_ptr()),
      conv_state.stride(0),
      conv_state.stride(1),
      conv_state.stride(2),
      reinterpret_cast<const uint16_t*>(raw_beta.data_ptr()),
      raw_beta.stride(1),
      A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(),
      state.data_ptr<float>(),
      state.stride(0),
      state_indices.data_ptr<int32_t>(),
      reinterpret_cast<const uint16_t*>(output_gate.data_ptr()),
      output_gate.stride(0),
      reinterpret_cast<const uint16_t*>(norm_weight.data_ptr()),
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      out.stride(1),
      static_cast<float>(lower_bound),
      static_cast<float>(norm_eps));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kda_decode_gfx950", &kda_decode_gfx950, "gfx950 Kimi-K3 fused KDA decode");
}
