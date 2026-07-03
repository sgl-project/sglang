// CUDA fast path for LTX2 Q/K RMSNorm + split RoPE.
//
// Developed with MIT HAN Lab Kernel Design Agents:
// https://github.com/mit-han-lab/kernel-design-agents
//
// This mirrors the LTX2 eager oracle:
//   torch.nn.RMSNorm(input) returns fp32 under bf16 autocast, then split RoPE
//   runs in fp32 and rounds once to bf16 at the final attention input.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel and CUDA dtype aliases

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang_ltx2_qknorm_split_rope {

namespace {

constexpr int kThreads = 128;

inline const char* data_ptr(const tvm::ffi::TensorView& t) {
  return static_cast<const char*>(t.data_ptr()) + t.byte_offset();
}

inline char* mutable_data_ptr(const tvm::ffi::TensorView& t) {
  return static_cast<char*>(t.data_ptr()) + t.byte_offset();
}

SGL_DEVICE float compute_rstd(
    const bf16_t* __restrict__ xrow,
    int64_t hidden_size,
    float eps,
    int tid,
    int lane,
    int warp_id,
    float* warp_sum,
    float* s_rstd) {
  float local = 0.f;
  const int64_t n_vec = hidden_size >> 2;
  for (int64_t i = tid; i < n_vec; i += kThreads) {
    const int64_t base = i << 2;
    const float v0 = __bfloat162float(xrow[base + 0]);
    const float v1 = __bfloat162float(xrow[base + 1]);
    const float v2 = __bfloat162float(xrow[base + 2]);
    const float v3 = __bfloat162float(xrow[base + 3]);
    local = fmaf(v0, v0, local);
    local = fmaf(v1, v1, local);
    local = fmaf(v2, v2, local);
    local = fmaf(v3, v3, local);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xffffffffu, local, offset);
  }
  if (lane == 0) {
    warp_sum[warp_id] = local;
  }
  __syncthreads();

  if (tid == 0) {
    const float total = (warp_sum[0] + warp_sum[2]) + (warp_sum[1] + warp_sum[3]);
    *s_rstd = rsqrtf(total / static_cast<float>(hidden_size) + eps);
  }
  __syncthreads();
  return *s_rstd;
}

SGL_DEVICE float norm_value(float x, float weight, float rstd) {
  return weight * (rstd * x);
}

SGL_DEVICE void rope_pair(float x0, float x1, float cos, float sin, float& y0, float& y1) {
  const float p0 = x0 * cos;
  const float p1 = x1 * cos;
  y0 = fmaf(-sin, x1, p0);
  y1 = fmaf(sin, x0, p1);
}

__global__ void ltx2_qknorm_split_rope_kernel(
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ cos,
    const bf16_t* __restrict__ sin,
    const bf16_t* __restrict__ weight,
    bf16_t* __restrict__ out,
    float eps,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t stride_cos_b,
    int64_t stride_cos_h,
    int64_t stride_cos_t,
    int64_t stride_sin_b,
    int64_t stride_sin_h,
    int64_t stride_sin_t) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = row / seq_len;
  const int64_t token = row - batch * seq_len;
  const int64_t hidden_size = num_heads * head_dim;
  const int64_t half_dim = head_dim >> 1;
  const auto* __restrict__ xrow = x + row * hidden_size;
  auto* __restrict__ outrow = out + row * hidden_size;
  const int tid = threadIdx.x + threadIdx.y * 32;
  const int lane = threadIdx.x;
  const int warp_id = threadIdx.y;

  __shared__ float warp_sum[4];
  __shared__ float s_rstd;
  const float rstd = compute_rstd(xrow, hidden_size, eps, tid, lane, warp_id, warp_sum, &s_rstd);

  const int64_t num_pairs = num_heads * half_dim;
  for (int64_t pair = tid; pair < num_pairs; pair += kThreads) {
    const int64_t head = pair / half_dim;
    const int64_t offset = pair - head * half_dim;
    const int64_t idx0 = head * head_dim + offset;
    const int64_t idx1 = idx0 + half_dim;
    const float n0 = norm_value(__bfloat162float(xrow[idx0]), __bfloat162float(weight[idx0]), rstd);
    const float n1 = norm_value(__bfloat162float(xrow[idx1]), __bfloat162float(weight[idx1]), rstd);
    const int64_t cos_offset = batch * stride_cos_b + head * stride_cos_h + token * stride_cos_t + offset;
    const int64_t sin_offset = batch * stride_sin_b + head * stride_sin_h + token * stride_sin_t + offset;

    float y0;
    float y1;
    rope_pair(n0, n1, __bfloat162float(cos[cos_offset]), __bfloat162float(sin[sin_offset]), y0, y1);
    outrow[idx0] = __float2bfloat16_rn(y0);
    outrow[idx1] = __float2bfloat16_rn(y1);
  }
}

inline void launch_one(
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& cos,
    const tvm::ffi::TensorView& sin,
    const tvm::ffi::TensorView& weight,
    const tvm::ffi::TensorView& out,
    float eps,
    int64_t num_rows,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t stride_cos_b,
    int64_t stride_cos_h,
    int64_t stride_cos_t,
    int64_t stride_sin_b,
    int64_t stride_sin_h,
    int64_t stride_sin_t,
    DLDevice device) {
  if (num_rows == 0) {
    return;
  }
  host::RuntimeCheck(num_rows <= static_cast<int64_t>(UINT32_MAX), "LTX2 QKNorm split-RoPE grid is too large");
  host::LaunchKernel(dim3(static_cast<uint32_t>(num_rows)), dim3(32, 4), device)(
      ltx2_qknorm_split_rope_kernel,
      reinterpret_cast<const bf16_t*>(data_ptr(x)),
      reinterpret_cast<const bf16_t*>(data_ptr(cos)),
      reinterpret_cast<const bf16_t*>(data_ptr(sin)),
      reinterpret_cast<const bf16_t*>(data_ptr(weight)),
      reinterpret_cast<bf16_t*>(mutable_data_ptr(out)),
      eps,
      seq_len,
      num_heads,
      head_dim,
      stride_cos_b,
      stride_cos_h,
      stride_cos_t,
      stride_sin_b,
      stride_sin_h,
      stride_sin_t);
}

}  // namespace

struct LTX2QKNormSplitRopeKernel {
  static void
  run(tvm::ffi::TensorView q_out,
      tvm::ffi::TensorView k_out,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView q_cos,
      tvm::ffi::TensorView q_sin,
      tvm::ffi::TensorView q_weight,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView k_cos,
      tvm::ffi::TensorView k_sin,
      tvm::ffi::TensorView k_weight,
      double eps,
      int64_t num_heads,
      int64_t head_dim) {
    using namespace host;

    RuntimeCheck(num_heads > 0, "num_heads must be positive");
    RuntimeCheck(head_dim > 0, "head_dim must be positive");
    RuntimeCheck(head_dim % 2 == 0, "head_dim must be even");
    const int64_t hidden_size = num_heads * head_dim;
    RuntimeCheck(hidden_size % 4 == 0, "hidden size must be divisible by 4");

    auto batch = SymbolicSize{"batch"};
    auto q_seq_len = SymbolicSize{"q_seq_len"};
    auto k_seq_len = SymbolicSize{"k_seq_len"};
    auto heads = SymbolicSize{"num_heads"};
    auto half_dim = SymbolicSize{"half_dim"};
    auto device = SymbolicDevice{};
    heads.set_value(num_heads);
    half_dim.set_value(head_dim / 2);
    device.set_options<kDLCUDA>();

    TensorMatcher({batch, q_seq_len, hidden_size}).with_dtype<bf16_t>().with_device(device).verify(q).verify(q_out);
    TensorMatcher({batch, k_seq_len, hidden_size}).with_dtype<bf16_t>().with_device(device).verify(k).verify(k_out);
    TensorMatcher({hidden_size}).with_dtype<bf16_t>().with_device(device).verify(q_weight);
    TensorMatcher({hidden_size}).with_dtype<bf16_t>().with_device(device).verify(k_weight);
    TensorMatcher({batch, heads, q_seq_len, half_dim})
        .with_strides({-1, -1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_cos);
    TensorMatcher({batch, heads, q_seq_len, half_dim})
        .with_strides({-1, -1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_sin);
    TensorMatcher({batch, heads, k_seq_len, half_dim})
        .with_strides({-1, -1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(k_cos);
    TensorMatcher({batch, heads, k_seq_len, half_dim})
        .with_strides({-1, -1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(k_sin);

    const int64_t batch_size = batch.unwrap();
    const DLDevice dl_device = device.unwrap();
    launch_one(
        q,
        q_cos,
        q_sin,
        q_weight,
        q_out,
        static_cast<float>(eps),
        batch_size * q_seq_len.unwrap(),
        q_seq_len.unwrap(),
        num_heads,
        head_dim,
        q_cos.stride(0),
        q_cos.stride(1),
        q_cos.stride(2),
        q_sin.stride(0),
        q_sin.stride(1),
        q_sin.stride(2),
        dl_device);
    launch_one(
        k,
        k_cos,
        k_sin,
        k_weight,
        k_out,
        static_cast<float>(eps),
        batch_size * k_seq_len.unwrap(),
        k_seq_len.unwrap(),
        num_heads,
        head_dim,
        k_cos.stride(0),
        k_cos.stride(1),
        k_cos.stride(2),
        k_sin.stride(0),
        k_sin.stride(1),
        k_sin.stride(2),
        dl_device);
  }
};

}  // namespace sglang_ltx2_qknorm_split_rope
