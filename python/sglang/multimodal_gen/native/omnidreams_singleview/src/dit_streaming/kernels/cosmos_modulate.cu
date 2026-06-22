// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// cosmos_modulate.cu — fused element-wise primitives for the Cosmos DiT block.
//
// Two kernels are exposed:
//
//   cosmos_layernorm_modulate_kernel<ElementT>
//       fused (no-affine LayerNorm) + (1 + scale) + shift in one launch.
//       Mirrors the WAN `layernorm_modulate_row_rm_to_row_half_kernel` shape but
//       sources the per-block (shift, scale) tensors externally instead of
//       indexing into a static `scale_shift_table` -- adaln-LoRA produces
//       fresh values per call.
//
//   cosmos_residual_gate_kernel<ElementT>
//       fused gated residual: dest = src + gate * y, where `gate` is a
//       per-row [B, K] tensor produced by adaln-LoRA. Mirrors the WAN
//       `residual_gate_add_kernel_*` family.
//
// Both kernels are templated on the activation type so the cosmos forward
// can stay end-to-end bf16 (matches FlashDreams' CosmosDiTNetwork dtype).

#include "cosmos_block.cuh"
#include "dtype_utils.cuh"

#include <cstdlib>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

namespace omnidreams_singleview {

namespace {

// Default-on toggle for the 1-warp-per-row D=128 RMSNorm-to-FP8 path.
// Set OMNIDREAMS_DIT_RMSNORM_V2=0 (or false/no) to fall back to the 2-warp variant
// that uses __syncthreads. The v2 path is independent of attention backend
// (Sage3 / cuDNN) because RMSNorm runs before attention.
static bool cosmos_rmsnorm_d128_v2_enabled() {
  const char* env = std::getenv("OMNIDREAMS_DIT_RMSNORM_V2");
  if (!env || !env[0]) return true;
  return env[0] != '0' &&
         env[0] != 'f' && env[0] != 'F' &&
         env[0] != 'n' && env[0] != 'N';
}

__device__ __forceinline__ float cosmos_mod_warp_sum(float x) {
  for (int off = 16; off > 0; off >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, off);
  }
  return x;
}

__device__ __forceinline__ float cosmos_mod_block_sum_2warps(float x) {
  x = cosmos_mod_warp_sum(x);
  __shared__ float warp_sums[2];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = x;
  }
  __syncthreads();
  float total = (threadIdx.x < 2) ? warp_sums[lane] : 0.0f;
  total = (warp == 0) ? cosmos_mod_warp_sum(total) : 0.0f;
  if (threadIdx.x == 0) {
    warp_sums[0] = total;
  }
  __syncthreads();
  return warp_sums[0];
}

}  // namespace

// ---------------------------------------------------------------------------
// 1) layernorm + modulate (no-affine LN, then * (1 + scale) + shift)
//
// X         [M, K]            row-major activations
// scale     [B, K] or [1, K]  per-row modulation scale (broadcast on M-axis when stride=0)
// shift     [B, K] or [1, K]  per-row modulation shift
// row_to_b  optional mapping M -> B (nullptr means M==1 broadcast or per-row identical scale)
//
// scale_row_stride, shift_row_stride: stride between rows in `scale`/`shift`.
//   - 0  -> broadcast a single [K] vector across all M rows (Cosmos streaming
//           uses one timestep per batch -> one (shift,scale) per batch).
//   - K  -> per-row vectors; row index used directly.
//
// In the streaming forward we always have B=1 or B=2 (CFG) and M = L_new for
// the entire layer, so the scale/shift are broadcast (stride = 0) -> the
// kernel is purely memory-bound on X.
// ---------------------------------------------------------------------------
template <typename ElementT>
__global__ void cosmos_layernorm_modulate_kernel(
    const ElementT* __restrict__ X,
    const ElementT* __restrict__ shift,    // [B, K] or [K]
    const ElementT* __restrict__ scale,    // [B, K] or [K]
    ElementT* __restrict__ Y,
    int M, int K, float eps,
    int shift_row_stride,                  // 0 = broadcast, K = per-row
    int scale_row_stride,                  // 0 = broadcast, K = per-row
    int rows_per_b)                        // M / B; rows [r..r+rows_per_b) use shift/scale row r/rows_per_b
{
  int row = blockIdx.x;
  if (row >= M) return;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warp_count = (blockDim.x + 31) >> 5;
  extern __shared__ float smem[];
  float* ssum = smem;
  float* ssum2 = smem + warp_count;

  // Pass 1: mean + variance.
  float acc1 = 0.f, acc2 = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = omnidreams_singleview::to_float(X[size_t(row) * K + j]);
    acc1 += v;
    acc2 += v * v;
  }
  acc1 = cosmos_mod_warp_sum(acc1);
  acc2 = cosmos_mod_warp_sum(acc2);
  if (lane == 0) {
    ssum[warp] = acc1;
    ssum2[warp] = acc2;
  }
  __syncthreads();
  float total1 = (tid < warp_count) ? ssum[tid] : 0.0f;
  float total2 = (tid < warp_count) ? ssum2[tid] : 0.0f;
  total1 = (warp == 0) ? cosmos_mod_warp_sum(total1) : 0.0f;
  total2 = (warp == 0) ? cosmos_mod_warp_sum(total2) : 0.0f;
  if (tid == 0) {
    ssum[0] = total1;
    ssum2[0] = total2;
  }
  __syncthreads();
  float mean = ssum[0]  / float(K);
  float var  = ssum2[0] / float(K) - mean * mean;
  float inv  = rsqrtf(var + eps);

  // Determine which (shift, scale) row this M-row consumes.
  // rows_per_b == M when broadcast (B == 1 effectively; row_in_b always = 0).
  int b_row = (rows_per_b > 0) ? (row / rows_per_b) : 0;

  // Pass 2: normalize + modulate.
  for (int j = tid; j < K; j += blockDim.x) {
    float v = (omnidreams_singleview::to_float(X[size_t(row) * K + j]) - mean) * inv;
    float sh = omnidreams_singleview::to_float(shift[size_t(b_row) * shift_row_stride + j]);
    float sc = omnidreams_singleview::to_float(scale[size_t(b_row) * scale_row_stride + j]);
    v = v * (1.f + sc) + sh;
    Y[size_t(row) * K + j] = omnidreams_singleview::from_float<ElementT>(v);
  }
}

template <typename ElementT>
__global__ void cosmos_layernorm_modulate_to_fp8_kernel(
    const ElementT* __restrict__ X,
    const ElementT* __restrict__ shift,
    const ElementT* __restrict__ scale,
    ElementT* __restrict__ Y,
    cutlass::float_e4m3_t* __restrict__ Y_fp8,
    int M, int K, float eps,
    int shift_row_stride,
    int scale_row_stride,
    int rows_per_b)
{
  int row = blockIdx.x;
  if (row >= M) return;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warp_count = (blockDim.x + 31) >> 5;
  extern __shared__ float smem[];
  float* ssum = smem;
  float* ssum2 = smem + warp_count;

  float acc1 = 0.f, acc2 = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = omnidreams_singleview::to_float(X[size_t(row) * K + j]);
    acc1 += v;
    acc2 += v * v;
  }
  acc1 = cosmos_mod_warp_sum(acc1);
  acc2 = cosmos_mod_warp_sum(acc2);
  if (lane == 0) {
    ssum[warp] = acc1;
    ssum2[warp] = acc2;
  }
  __syncthreads();
  float total1 = (tid < warp_count) ? ssum[tid] : 0.0f;
  float total2 = (tid < warp_count) ? ssum2[tid] : 0.0f;
  total1 = (warp == 0) ? cosmos_mod_warp_sum(total1) : 0.0f;
  total2 = (warp == 0) ? cosmos_mod_warp_sum(total2) : 0.0f;
  if (tid == 0) {
    ssum[0] = total1;
    ssum2[0] = total2;
  }
  __syncthreads();
  float mean = ssum[0]  / float(K);
  float var  = ssum2[0] / float(K) - mean * mean;
  float inv  = rsqrtf(var + eps);
  int b_row = (rows_per_b > 0) ? (row / rows_per_b) : 0;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;

  for (int j = tid; j < K; j += blockDim.x) {
    size_t idx = size_t(row) * K + j;
    float v = (omnidreams_singleview::to_float(X[idx]) - mean) * inv;
    float sh = omnidreams_singleview::to_float(shift[size_t(b_row) * shift_row_stride + j]);
    float sc = omnidreams_singleview::to_float(scale[size_t(b_row) * scale_row_stride + j]);
    v = v * (1.f + sc) + sh;
    Y[idx] = omnidreams_singleview::from_float<ElementT>(v);
    Y_fp8[idx] = to_fp8(v);
  }
}

template <typename ElementT>
__global__ void cosmos_layernorm_modulate_to_fp8_only_kernel(
    const ElementT* __restrict__ X,
    const ElementT* __restrict__ shift,
    const ElementT* __restrict__ scale,
    cutlass::float_e4m3_t* __restrict__ Y_fp8,
    int M, int K, float eps,
    int shift_row_stride,
    int scale_row_stride,
    int rows_per_b)
{
  int row = blockIdx.x;
  if (row >= M) return;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warp_count = (blockDim.x + 31) >> 5;
  extern __shared__ float smem[];
  float* ssum = smem;
  float* ssum2 = smem + warp_count;

  float acc1 = 0.f, acc2 = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = omnidreams_singleview::to_float(X[size_t(row) * K + j]);
    acc1 += v;
    acc2 += v * v;
  }
  acc1 = cosmos_mod_warp_sum(acc1);
  acc2 = cosmos_mod_warp_sum(acc2);
  if (lane == 0) {
    ssum[warp] = acc1;
    ssum2[warp] = acc2;
  }
  __syncthreads();
  float total1 = (tid < warp_count) ? ssum[tid] : 0.0f;
  float total2 = (tid < warp_count) ? ssum2[tid] : 0.0f;
  total1 = (warp == 0) ? cosmos_mod_warp_sum(total1) : 0.0f;
  total2 = (warp == 0) ? cosmos_mod_warp_sum(total2) : 0.0f;
  if (tid == 0) {
    ssum[0] = total1;
    ssum2[0] = total2;
  }
  __syncthreads();
  float mean = ssum[0]  / float(K);
  float var  = ssum2[0] / float(K) - mean * mean;
  float inv  = rsqrtf(var + eps);
  int b_row = (rows_per_b > 0) ? (row / rows_per_b) : 0;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;

  for (int j = tid; j < K; j += blockDim.x) {
    size_t idx = size_t(row) * K + j;
    float v = (omnidreams_singleview::to_float(X[idx]) - mean) * inv;
    float sh = omnidreams_singleview::to_float(shift[size_t(b_row) * shift_row_stride + j]);
    float sc = omnidreams_singleview::to_float(scale[size_t(b_row) * scale_row_stride + j]);
    v = v * (1.f + sc) + sh;
    Y_fp8[idx] = to_fp8(v);
  }
}

// Host launcher
template <typename ElementT>
cudaError_t cosmos_layernorm_modulate(
    const ElementT* X,
    const ElementT* shift,
    const ElementT* scale,
    ElementT* Y,
    int M, int K,
    int B,
    float eps,
    cudaStream_t stream)
{
  if (M <= 0 || K <= 0) return cudaSuccess;

  // shift/scale are [B, K] from adaln-LoRA. When M > B we broadcast each
  // row of shift/scale to (M / B) consecutive rows of X.
  int rows_per_b = (B > 0) ? (M / B) : M;
  int shift_row_stride = K;
  int scale_row_stride = K;
  if (B == 1) {
    // Single batch -> single shift/scale row consumed by all M rows. We still
    // pass row_stride=K (b_row always lands at 0) for simplicity.
    rows_per_b = M;
  }

  int threads = 256;
  size_t smem = 2 * ((threads + 31) / 32) * sizeof(float);
  cosmos_layernorm_modulate_kernel<ElementT><<<M, threads, smem, stream>>>(
      X, shift, scale, Y, M, K, eps, shift_row_stride, scale_row_stride, rows_per_b);
  return cudaGetLastError();
}

template <typename ElementT>
cudaError_t cosmos_layernorm_modulate_to_fp8(
    const ElementT* X,
    const ElementT* shift,
    const ElementT* scale,
    ElementT* Y,
    cutlass::float_e4m3_t* Y_fp8,
    int M, int K,
    int B,
    float eps,
    cudaStream_t stream)
{
  if (M <= 0 || K <= 0) return cudaSuccess;
  if (!X || !shift || !scale || !Y || !Y_fp8) return cudaErrorInvalidValue;

  int rows_per_b = (B > 0) ? (M / B) : M;
  int shift_row_stride = K;
  int scale_row_stride = K;
  if (B == 1) {
    rows_per_b = M;
  }

  int threads = 256;
  size_t smem = 2 * ((threads + 31) / 32) * sizeof(float);
  cosmos_layernorm_modulate_to_fp8_kernel<ElementT><<<M, threads, smem, stream>>>(
      X, shift, scale, Y, Y_fp8, M, K, eps, shift_row_stride, scale_row_stride, rows_per_b);
  return cudaGetLastError();
}

template <typename ElementT>
cudaError_t cosmos_layernorm_modulate_to_fp8_only(
    const ElementT* X,
    const ElementT* shift,
    const ElementT* scale,
    cutlass::float_e4m3_t* Y_fp8,
    int M, int K,
    int B,
    float eps,
    cudaStream_t stream)
{
  if (M <= 0 || K <= 0) return cudaSuccess;
  if (!X || !shift || !scale || !Y_fp8) return cudaErrorInvalidValue;

  int rows_per_b = (B > 0) ? (M / B) : M;
  int shift_row_stride = K;
  int scale_row_stride = K;
  if (B == 1) {
    rows_per_b = M;
  }

  int threads = 256;
  size_t smem = 2 * ((threads + 31) / 32) * sizeof(float);
  cosmos_layernorm_modulate_to_fp8_only_kernel<ElementT><<<M, threads, smem, stream>>>(
      X, shift, scale, Y_fp8, M, K, eps, shift_row_stride, scale_row_stride, rows_per_b);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Fused: col_scale + residual + LayerNorm + modulate -> FP8
//
// Replaces the back-to-back pair
//   col_scale_residual_bf16_vec2_kernel  (gemm_half + alpha * scale_mul -> += residual_inout)
//   cosmos_layernorm_modulate_to_fp8_only_kernel (LN(residual_inout), modulate, FP8)
// with a single launch that keeps the BF16-truncated `x_new` row in shared
// memory between the residual write and the LN read, eliminating one full BF16
// re-read of x from DRAM.
//
// Layout:
//   gemm_half        FP16 [M, K]   row-major cuBLASLt FP8 GEMM scratch
//   alpha            FP16 [K]      precomputed weight_scale * gate * input_scale
//   residual_inout   BF16 [M, K]   x, updated in place to x + scale_mul*alpha*gemm_half
//   ln_shift         BF16 [B, K] or [K]  LN modulate shift (broadcast on M-axis)
//   ln_scale         BF16 [B, K] or [K]  LN modulate scale
//   fp8_out          E4M3 [M, K]   modulated FP8 output for the next FP8 GEMM
//
// One CTA per row. Each thread strides over D in chunks of blockDim.x. The
// new x_new value is BF16-truncated to match the byte-equivalent semantics of
// the unfused two-kernel sequence (DRAM round-trip would have done the same).
template <typename ElementT>
__global__ void cosmos_col_scale_residual_layernorm_modulate_to_fp8_kernel(
    const cutlass::half_t* __restrict__ gemm_half,
    const cutlass::half_t* __restrict__ alpha,
    ElementT* __restrict__ residual_inout,
    const ElementT* __restrict__ ln_shift,
    const ElementT* __restrict__ ln_scale,
    cutlass::float_e4m3_t* __restrict__ fp8_out,
    float scale_mul,
    int M, int K, float eps,
    int shift_row_stride,
    int scale_row_stride,
    int rows_per_b)
{
  int row = blockIdx.x;
  if (row >= M) return;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warp_count = (blockDim.x + 31) >> 5;
  extern __shared__ float smem[];
  float* ssum = smem;
  float* ssum2 = smem + warp_count;
  // sx[K] holds the BF16-truncated new x for the LN second pass, eliminating
  // the BF16 re-read from DRAM. Layout: smem = [ssum(warp_count), ssum2(warp_count), sx(K)]
  float* sx = smem + 2 * warp_count;

  float acc1 = 0.f, acc2 = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    size_t idx = size_t(row) * K + j;
    float gv = __half2float(reinterpret_cast<const half*>(gemm_half)[idx]);
    float av = __half2float(reinterpret_cast<const half*>(alpha)[j]);
    float scaled = gv * (av * scale_mul);
    float resv = omnidreams_singleview::to_float(residual_inout[idx]);
    float xnew = scaled + resv;
    ElementT xnew_e = omnidreams_singleview::from_float<ElementT>(xnew);
    residual_inout[idx] = xnew_e;
    float xnew_f = omnidreams_singleview::to_float(xnew_e);
    sx[j] = xnew_f;
    acc1 += xnew_f;
    acc2 += xnew_f * xnew_f;
  }
  acc1 = cosmos_mod_warp_sum(acc1);
  acc2 = cosmos_mod_warp_sum(acc2);
  if (lane == 0) {
    ssum[warp] = acc1;
    ssum2[warp] = acc2;
  }
  __syncthreads();
  float total1 = (tid < warp_count) ? ssum[tid] : 0.0f;
  float total2 = (tid < warp_count) ? ssum2[tid] : 0.0f;
  total1 = (warp == 0) ? cosmos_mod_warp_sum(total1) : 0.0f;
  total2 = (warp == 0) ? cosmos_mod_warp_sum(total2) : 0.0f;
  if (tid == 0) {
    ssum[0] = total1;
    ssum2[0] = total2;
  }
  __syncthreads();
  float mean = ssum[0]  / float(K);
  float var  = ssum2[0] / float(K) - mean * mean;
  float inv  = rsqrtf(var + eps);
  int b_row = (rows_per_b > 0) ? (row / rows_per_b) : 0;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;

  for (int j = tid; j < K; j += blockDim.x) {
    size_t idx = size_t(row) * K + j;
    float v = (sx[j] - mean) * inv;
    float sh = omnidreams_singleview::to_float(ln_shift[size_t(b_row) * shift_row_stride + j]);
    float sc = omnidreams_singleview::to_float(ln_scale[size_t(b_row) * scale_row_stride + j]);
    v = v * (1.f + sc) + sh;
    fp8_out[idx] = to_fp8(v);
  }
}

template <typename ElementT>
cudaError_t cosmos_col_scale_residual_layernorm_modulate_to_fp8_only(
    const cutlass::half_t* gemm_half,
    const cutlass::half_t* alpha,
    ElementT* residual_inout,
    const ElementT* ln_shift,
    const ElementT* ln_scale,
    cutlass::float_e4m3_t* fp8_out,
    float scale_mul,
    int M, int K,
    int B,
    float eps,
    cudaStream_t stream)
{
  if (M <= 0 || K <= 0) return cudaSuccess;
  if (!gemm_half || !alpha || !residual_inout || !ln_shift || !ln_scale || !fp8_out) {
    return cudaErrorInvalidValue;
  }

  int rows_per_b = (B > 0) ? (M / B) : M;
  int shift_row_stride = K;
  int scale_row_stride = K;
  if (B == 1) {
    rows_per_b = M;
  }

  int threads = 256;
  int warp_count = (threads + 31) / 32;
  size_t smem = (2 * warp_count + K) * sizeof(float);
  cosmos_col_scale_residual_layernorm_modulate_to_fp8_kernel<ElementT>
      <<<M, threads, smem, stream>>>(
          gemm_half, alpha, residual_inout, ln_shift, ln_scale, fp8_out,
          scale_mul, M, K, eps, shift_row_stride, scale_row_stride, rows_per_b);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// 2) gated residual: dest = src + gate * y
//
// dest, src, y: [M, K] row-major
// gate:         [B, K] or [K] broadcast across M-rows in batches of (M / B)
// ---------------------------------------------------------------------------
template <typename ElementT>
__global__ void cosmos_residual_gate_kernel(
    ElementT* __restrict__ dest,
    const ElementT* __restrict__ src,
    const ElementT* __restrict__ y,
    const ElementT* __restrict__ gate,     // [B, K] or [K]
    int M, int K,
    int gate_row_stride,                   // 0 = broadcast, K = per-row
    int rows_per_b)                        // M / B
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= K) return;

  size_t pos = size_t(i) * K + j;
  int b_row = (rows_per_b > 0) ? (i / rows_per_b) : 0;
  float g = omnidreams_singleview::to_float(gate[size_t(b_row) * gate_row_stride + j]);
  float s = omnidreams_singleview::to_float(src[pos]);
  float yv = omnidreams_singleview::to_float(y[pos]);
  dest[pos] = omnidreams_singleview::from_float<ElementT>(s + g * yv);
}

template <typename ElementT>
cudaError_t cosmos_residual_gate(
    ElementT* dest,
    const ElementT* src,
    const ElementT* y,
    const ElementT* gate,
    int M, int K,
    int B,
    cudaStream_t stream)
{
  if (M <= 0 || K <= 0) return cudaSuccess;

  int rows_per_b = (B > 0) ? (M / B) : M;
  int gate_row_stride = (B > 1) ? K : K;  // always K; b_row stays 0 when B==1
  if (B <= 1) rows_per_b = M;

  dim3 block(32, 8);
  dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  cosmos_residual_gate_kernel<ElementT><<<grid, block, 0, stream>>>(
      dest, src, y, gate, M, K, gate_row_stride, rows_per_b);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// 3) RMSNorm on packed [B, M, H, D] Q or K (per-head-dim normalization)
//
// Cosmos applies q_norm / k_norm AFTER QKV projection but BEFORE RoPE. The
// existing WAN `apply_rmsnorm_pack_rope` kernel bakes adjacent-pair RoPE in,
// which is the wrong rotation convention for Cosmos. We factor RMSNorm out
// here so the bridge can chain it with `apply_cosmos_rope` (rotate-half).
//
// Layout: data is [B, M, H, D] row-major (BMHK). gamma is [D] -- same gamma
// applies per-head.
// ---------------------------------------------------------------------------
template <typename ElementT>
__global__ void cosmos_rmsnorm_per_head_kernel(
    ElementT* __restrict__ data,           // in-place [B*M*H, D]
    const ElementT* __restrict__ gamma,    // [D]
    int total_rows, int D, float eps)
{
  int row = blockIdx.x;
  if (row >= total_rows) return;

  extern __shared__ float smem[];
  float* ssum2 = smem;
  int tid = threadIdx.x;

  float acc = 0.f;
  for (int j = tid; j < D; j += blockDim.x) {
    float v = omnidreams_singleview::to_float(data[size_t(row) * D + j]);
    acc += v * v;
  }
  ssum2[tid] = acc;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) ssum2[tid] += ssum2[tid + off];
    __syncthreads();
  }
  float rms = rsqrtf(ssum2[0] / float(D) + eps);

  for (int j = tid; j < D; j += blockDim.x) {
    float v = omnidreams_singleview::to_float(data[size_t(row) * D + j]) * rms;
    float g = omnidreams_singleview::to_float(gamma[j]);
    data[size_t(row) * D + j] = omnidreams_singleview::from_float<ElementT>(v * g);
  }
}

template <typename ElementT>
__global__ void cosmos_rmsnorm_per_head_to_fp8_kernel(
    ElementT* __restrict__ data,                   // in-place [B*M*H, D]
    const ElementT* __restrict__ gamma,            // [D]
    cutlass::float_e4m3_t* __restrict__ fp8_out,   // [B*M*H, D]
    cutlass::float_e4m3_t* __restrict__ fp8_out_bhmd, // optional [B,H,M,D]
    int B, int M, int H, int D, float eps)
{
  int total_rows = B * M * H;
  int row = blockIdx.x;
  if (row >= total_rows) return;

  extern __shared__ float smem[];
  float* ssum2 = smem;
  int tid = threadIdx.x;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;

  float acc = 0.f;
  for (int j = tid; j < D; j += blockDim.x) {
    float v = omnidreams_singleview::to_float(data[size_t(row) * D + j]);
    acc += v * v;
  }
  ssum2[tid] = acc;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) ssum2[tid] += ssum2[tid + off];
    __syncthreads();
  }
  float rms = rsqrtf(ssum2[0] / float(D) + eps);

  for (int j = tid; j < D; j += blockDim.x) {
    size_t idx = size_t(row) * D + j;
    float v = omnidreams_singleview::to_float(data[idx]) * rms;
    float g = omnidreams_singleview::to_float(gamma[j]);
    float out = v * g;
    data[idx] = omnidreams_singleview::from_float<ElementT>(out);
    fp8_out[idx] = to_fp8(out);
    if (fp8_out_bhmd) {
      int h = row % H;
      int m = (row / H) % M;
      int b = row / (M * H);
      size_t bhmd_idx = ((size_t(b) * H + size_t(h)) * M + size_t(m)) * D + j;
      fp8_out_bhmd[bhmd_idx] = to_fp8(out);
    }
  }
}

template <typename ElementT>
__global__ void cosmos_rmsnorm_per_head_to_fp8_d128_kernel(
    ElementT* __restrict__ data,
    const ElementT* __restrict__ gamma,
    cutlass::float_e4m3_t* __restrict__ fp8_out,
    cutlass::float_e4m3_t* __restrict__ fp8_out_bhmd,
    int B, int M, int H, float eps)
{
  constexpr int D = 128;
  int total_rows = B * M * H;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= total_rows || tid >= 64) return;

  int j0 = tid;
  int j1 = tid + 64;
  size_t base = size_t(row) * D;
  float v0 = omnidreams_singleview::to_float(data[base + j0]);
  float v1 = omnidreams_singleview::to_float(data[base + j1]);
  float sum2 = cosmos_mod_block_sum_2warps(v0 * v0 + v1 * v1);
  float rms = rsqrtf(sum2 / float(D) + eps);

  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  float o0 = v0 * rms * omnidreams_singleview::to_float(gamma[j0]);
  float o1 = v1 * rms * omnidreams_singleview::to_float(gamma[j1]);
  data[base + j0] = omnidreams_singleview::from_float<ElementT>(o0);
  data[base + j1] = omnidreams_singleview::from_float<ElementT>(o1);
  cutlass::float_e4m3_t fp0 = to_fp8(o0);
  cutlass::float_e4m3_t fp1 = to_fp8(o1);
  fp8_out[base + j0] = fp0;
  fp8_out[base + j1] = fp1;
  if (fp8_out_bhmd) {
    int h = row % H;
    int m = (row / H) % M;
    int b = row / (M * H);
    size_t bhmd_base = ((size_t(b) * H + size_t(h)) * M + size_t(m)) * D;
    fp8_out_bhmd[bhmd_base + j0] = fp0;
    fp8_out_bhmd[bhmd_base + j1] = fp1;
  }
}

// D=128 RMSNorm-to-FP8 v2: one warp per row, two rows per CTA, no __syncthreads.
//
// Motivation: the original `_d128_kernel` uses block size 64 (2 warps) and an
// inter-warp `__syncthreads()` for the RMS reduction. With only 2 warps in a
// CTA there is nothing for the warp scheduler to do during the barrier wait,
// so achieved occupancy is ~30% (vs theoretical 100%) per NCU. This v2 layout
// has each warp own its own row and reduce purely with `__shfl_*_sync`,
// removing the barrier and lifting achieved occupancy to the same range as
// every other kernel in the block (~80%).
//
// Layout per warp (lane = threadIdx.x & 31):
//   row offsets per thread = {lane, lane+32, lane+64, lane+96}
//   covers all 128 D values with a single 32-thread warp using vec1 loads.
template <typename ElementT>
__global__ void cosmos_rmsnorm_per_head_to_fp8_d128_v2_kernel(
    ElementT* __restrict__ data,
    const ElementT* __restrict__ gamma,
    cutlass::float_e4m3_t* __restrict__ fp8_out,
    cutlass::float_e4m3_t* __restrict__ fp8_out_bhmd,
    int B, int M, int H, float eps)
{
  constexpr int D = 128;
  const int total_rows = B * M * H;
  const int warp_id = threadIdx.x >> 5;          // 0 or 1
  const int lane    = threadIdx.x & 31;
  const int row     = blockIdx.x * 2 + warp_id;
  if (row >= total_rows) return;

  const size_t base = size_t(row) * D;
  const int j0 = lane;
  const int j1 = lane + 32;
  const int j2 = lane + 64;
  const int j3 = lane + 96;

  float v0 = omnidreams_singleview::to_float(data[base + j0]);
  float v1 = omnidreams_singleview::to_float(data[base + j1]);
  float v2 = omnidreams_singleview::to_float(data[base + j2]);
  float v3 = omnidreams_singleview::to_float(data[base + j3]);

  float s = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    s += __shfl_down_sync(0xffffffffu, s, off);
  }
  s = __shfl_sync(0xffffffffu, s, 0);
  float rms = rsqrtf(s / float(D) + eps);

  float g0 = omnidreams_singleview::to_float(gamma[j0]);
  float g1 = omnidreams_singleview::to_float(gamma[j1]);
  float g2 = omnidreams_singleview::to_float(gamma[j2]);
  float g3 = omnidreams_singleview::to_float(gamma[j3]);

  float o0 = v0 * rms * g0;
  float o1 = v1 * rms * g1;
  float o2 = v2 * rms * g2;
  float o3 = v3 * rms * g3;

  data[base + j0] = omnidreams_singleview::from_float<ElementT>(o0);
  data[base + j1] = omnidreams_singleview::from_float<ElementT>(o1);
  data[base + j2] = omnidreams_singleview::from_float<ElementT>(o2);
  data[base + j3] = omnidreams_singleview::from_float<ElementT>(o3);

  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  cutlass::float_e4m3_t f0 = to_fp8(o0);
  cutlass::float_e4m3_t f1 = to_fp8(o1);
  cutlass::float_e4m3_t f2 = to_fp8(o2);
  cutlass::float_e4m3_t f3 = to_fp8(o3);

  fp8_out[base + j0] = f0;
  fp8_out[base + j1] = f1;
  fp8_out[base + j2] = f2;
  fp8_out[base + j3] = f3;

  if (fp8_out_bhmd) {
    int h = row % H;
    int m = (row / H) % M;
    int b = row / (M * H);
    size_t bhmd_base = ((size_t(b) * H + size_t(h)) * M + size_t(m)) * D;
    fp8_out_bhmd[bhmd_base + j0] = f0;
    fp8_out_bhmd[bhmd_base + j1] = f1;
    fp8_out_bhmd[bhmd_base + j2] = f2;
    fp8_out_bhmd[bhmd_base + j3] = f3;
  }
}

template <typename ElementT>
cudaError_t cosmos_rmsnorm_per_head(
    ElementT* data,
    const ElementT* gamma,
    int B, int M, int H, int D,
    float eps,
    cudaStream_t stream)
{
  if (B <= 0 || M <= 0 || H <= 0 || D <= 0) return cudaSuccess;
  int total_rows = B * M * H;
  int threads = (D <= 64) ? 64 : (D <= 128 ? 128 : 256);
  size_t smem = threads * sizeof(float);
  cosmos_rmsnorm_per_head_kernel<ElementT><<<total_rows, threads, smem, stream>>>(
      data, gamma, total_rows, D, eps);
  return cudaGetLastError();
}

template <typename ElementT>
cudaError_t cosmos_rmsnorm_per_head_to_fp8(
    ElementT* data,
    const ElementT* gamma,
    cutlass::float_e4m3_t* fp8_out,
    cutlass::float_e4m3_t* fp8_out_bhmd,
    int B, int M, int H, int D,
    float eps,
    cudaStream_t stream)
{
  if (B <= 0 || M <= 0 || H <= 0 || D <= 0) return cudaSuccess;
  if (!data || !gamma || !fp8_out) return cudaErrorInvalidValue;
  int total_rows = B * M * H;
  if (D == 128) {
    if (cosmos_rmsnorm_d128_v2_enabled()) {
      // Block size 64 = 2 warps, each warp owns one row -> 2 rows per CTA.
      int blocks = (total_rows + 1) / 2;
      cosmos_rmsnorm_per_head_to_fp8_d128_v2_kernel<ElementT>
          <<<blocks, 64, 0, stream>>>(
              data, gamma, fp8_out, fp8_out_bhmd, B, M, H, eps);
      return cudaGetLastError();
    }
    cosmos_rmsnorm_per_head_to_fp8_d128_kernel<ElementT><<<total_rows, 64, 0, stream>>>(
        data, gamma, fp8_out, fp8_out_bhmd, B, M, H, eps);
    return cudaGetLastError();
  }
  int threads = (D <= 64) ? 64 : (D <= 128 ? 128 : 256);
  size_t smem = threads * sizeof(float);
  cosmos_rmsnorm_per_head_to_fp8_kernel<ElementT><<<total_rows, threads, smem, stream>>>(
      data, gamma, fp8_out, fp8_out_bhmd, B, M, H, D, eps);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Explicit instantiations for the dtypes the cosmos forward needs.
// (CUDA __global__ functions can't be explicitly instantiated, but the host
// launchers can.)
// ---------------------------------------------------------------------------
template cudaError_t cosmos_layernorm_modulate<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, const cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    cutlass::bfloat16_t*, int, int, int, float, cudaStream_t);
template cudaError_t cosmos_layernorm_modulate<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*, const cutlass::half_t*,
    cutlass::half_t*, int, int, int, float, cudaStream_t);

template cudaError_t cosmos_layernorm_modulate_to_fp8<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, const cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    cutlass::bfloat16_t*, cutlass::float_e4m3_t*, int, int, int, float, cudaStream_t);
template cudaError_t cosmos_layernorm_modulate_to_fp8<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*, const cutlass::half_t*,
    cutlass::half_t*, cutlass::float_e4m3_t*, int, int, int, float, cudaStream_t);

template cudaError_t cosmos_layernorm_modulate_to_fp8_only<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, const cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    cutlass::float_e4m3_t*, int, int, int, float, cudaStream_t);
template cudaError_t cosmos_layernorm_modulate_to_fp8_only<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*, const cutlass::half_t*,
    cutlass::float_e4m3_t*, int, int, int, float, cudaStream_t);

template cudaError_t cosmos_col_scale_residual_layernorm_modulate_to_fp8_only<cutlass::bfloat16_t>(
    const cutlass::half_t*, const cutlass::half_t*,
    cutlass::bfloat16_t*, const cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    cutlass::float_e4m3_t*, float, int, int, int, float, cudaStream_t);
template cudaError_t cosmos_col_scale_residual_layernorm_modulate_to_fp8_only<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*,
    cutlass::half_t*, const cutlass::half_t*, const cutlass::half_t*,
    cutlass::float_e4m3_t*, float, int, int, int, float, cudaStream_t);

template cudaError_t cosmos_residual_gate<cutlass::bfloat16_t>(
    cutlass::bfloat16_t*, const cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    const cutlass::bfloat16_t*, int, int, int, cudaStream_t);
template cudaError_t cosmos_residual_gate<cutlass::half_t>(
    cutlass::half_t*, const cutlass::half_t*, const cutlass::half_t*,
    const cutlass::half_t*, int, int, int, cudaStream_t);

template cudaError_t cosmos_rmsnorm_per_head<cutlass::bfloat16_t>(
    cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    int, int, int, int, float, cudaStream_t);
template cudaError_t cosmos_rmsnorm_per_head<cutlass::half_t>(
    cutlass::half_t*, const cutlass::half_t*,
    int, int, int, int, float, cudaStream_t);

template cudaError_t cosmos_rmsnorm_per_head_to_fp8<cutlass::bfloat16_t>(
    cutlass::bfloat16_t*, const cutlass::bfloat16_t*, cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*, int, int, int, int, float, cudaStream_t);
template cudaError_t cosmos_rmsnorm_per_head_to_fp8<cutlass::half_t>(
    cutlass::half_t*, const cutlass::half_t*, cutlass::float_e4m3_t*,
    cutlass::float_e4m3_t*, int, int, int, int, float, cudaStream_t);

} // namespace omnidreams_singleview
