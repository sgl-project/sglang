// CUDA fast path for LTX2 Q/K RMSNorm + split RoPE.
//
// Developed with MIT HAN Lab Kernel Design Agents:
// https://github.com/mit-han-lab/kernel-design-agents
//
// This mirrors the LTX2 eager oracle:
//   torch.nn.RMSNorm(input) returns fp32 under bf16 autocast, then split RoPE
//   runs in fp32 and rounds once to bf16 at the final attention input.

#pragma once

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sglang_ltx2_qknorm_split_rope {

namespace ffi = tvm::ffi;

namespace {

constexpr int kThreads = 128;

inline const char* data_ptr(const ffi::TensorView& t) {
  return static_cast<const char*>(t.data_ptr()) + t.byte_offset();
}

inline char* mutable_data_ptr(const ffi::TensorView& t) {
  return static_cast<char*>(t.data_ptr()) + t.byte_offset();
}

__device__ inline float compute_rstd(
    const __nv_bfloat16* __restrict__ xrow,
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

__device__ inline float norm_value(float x, float weight, float rstd) {
  return weight * (rstd * x);
}

__device__ inline void rope_pair(float x0, float x1, float cos, float sin, float& y0, float& y1) {
  const float p0 = x0 * cos;
  const float p1 = x1 * cos;
  y0 = fmaf(-sin, x1, p0);
  y1 = fmaf(sin, x0, p1);
}

__global__ void ltx2_qknorm_split_rope_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
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
    const int64_t cos_offset = static_cast<int64_t>(batch) * stride_cos_b + static_cast<int64_t>(head) * stride_cos_h +
                               static_cast<int64_t>(token) * stride_cos_t + offset;
    const int64_t sin_offset = static_cast<int64_t>(batch) * stride_sin_b + static_cast<int64_t>(head) * stride_sin_h +
                               static_cast<int64_t>(token) * stride_sin_t + offset;

    float y0;
    float y1;
    rope_pair(n0, n1, __bfloat162float(cos[cos_offset]), __bfloat162float(sin[sin_offset]), y0, y1);
    outrow[idx0] = __float2bfloat16_rn(y0);
    outrow[idx1] = __float2bfloat16_rn(y1);
  }
}

inline void launch_one(
    const ffi::TensorView& x,
    const ffi::TensorView& cos,
    const ffi::TensorView& sin,
    const ffi::TensorView& weight,
    const ffi::TensorView& out,
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
    cudaStream_t stream) {
  ltx2_qknorm_split_rope_kernel<<<dim3(static_cast<unsigned>(num_rows)), dim3(32, 4), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(data_ptr(x)),
      reinterpret_cast<const __nv_bfloat16*>(data_ptr(cos)),
      reinterpret_cast<const __nv_bfloat16*>(data_ptr(sin)),
      reinterpret_cast<const __nv_bfloat16*>(data_ptr(weight)),
      reinterpret_cast<__nv_bfloat16*>(mutable_data_ptr(out)),
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

void ltx2_qknorm_split_rope_pair(
    ffi::TensorView q,
    ffi::TensorView q_cos,
    ffi::TensorView q_sin,
    ffi::TensorView q_weight,
    ffi::TensorView q_out,
    ffi::TensorView k,
    ffi::TensorView k_cos,
    ffi::TensorView k_sin,
    ffi::TensorView k_weight,
    ffi::TensorView k_out,
    double eps,
    int64_t q_num_rows,
    int64_t q_seq_len,
    int64_t k_num_rows,
    int64_t k_seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t q_stride_cos_b,
    int64_t q_stride_cos_h,
    int64_t q_stride_cos_t,
    int64_t q_stride_sin_b,
    int64_t q_stride_sin_h,
    int64_t q_stride_sin_t,
    int64_t k_stride_cos_b,
    int64_t k_stride_cos_h,
    int64_t k_stride_cos_t,
    int64_t k_stride_sin_b,
    int64_t k_stride_sin_h,
    int64_t k_stride_sin_t,
    int64_t stream_ptr) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  launch_one(
      q,
      q_cos,
      q_sin,
      q_weight,
      q_out,
      static_cast<float>(eps),
      q_num_rows,
      q_seq_len,
      num_heads,
      head_dim,
      q_stride_cos_b,
      q_stride_cos_h,
      q_stride_cos_t,
      q_stride_sin_b,
      q_stride_sin_h,
      q_stride_sin_t,
      stream);
  launch_one(
      k,
      k_cos,
      k_sin,
      k_weight,
      k_out,
      static_cast<float>(eps),
      k_num_rows,
      k_seq_len,
      num_heads,
      head_dim,
      k_stride_cos_b,
      k_stride_cos_h,
      k_stride_cos_t,
      k_stride_sin_b,
      k_stride_sin_h,
      k_stride_sin_t,
      stream);
}

}  // namespace sglang_ltx2_qknorm_split_rope
