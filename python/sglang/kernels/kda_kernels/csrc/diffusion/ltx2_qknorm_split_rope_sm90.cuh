// LTX2 Q/K RMSNorm + split RoPE for SM90.
// Fuse Q/K launches and use bf16x2 loads/stores for aligned 64/128-wide heads,
// preserving the original reduction order and intermediate BF16 rounding.
// Other head dimensions, unaligned pointers and odd RoPE strides use scalar
// loads/stores without narrowing the public input contract.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel and CUDA dtype aliases

#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>

namespace sglang {

namespace ltx2_qknorm_split_rope_kda {

constexpr int kThreads = 128;
// Fuse Q/K below this row-count threshold. Larger grids and empty sides use
// separate launches, skipping any side with zero rows.
constexpr int64_t kSplitThreshold = KDA_SPLIT_THRESHOLD;

inline const char* data_ptr(const tvm::ffi::TensorView& t) {
  return static_cast<const char*>(t.data_ptr()) + t.byte_offset();
}

inline char* mutable_data_ptr(const tvm::ffi::TensorView& t) {
  return static_cast<char*>(t.data_ptr()) + t.byte_offset();
}

// Bit-exact copy of the pinned baseline sum-of-squares reduction (scalar bf16
// loads). Used by the scalar row path for any input that may be 2-byte
// aligned.
SGL_DEVICE float compute_rstd_scalar(
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

// bf16x2-vectorized sum-of-squares. Requires xrow to be 4-byte aligned (the
// host only calls this when x is 4-byte aligned and hidden_size % 4 == 0, so
// every 8-byte group read here is 4-byte aligned). The per-thread fmaf
// accumulation order is identical to the scalar version, so the reduction is
// bit-identical.
SGL_DEVICE float compute_rstd_vec(
    const bf16_t* __restrict__ xrow,
    int64_t hidden_size,
    float eps,
    int tid,
    int lane,
    int warp_id,
    float* warp_sum,
    float* s_rstd) {
  float local = 0.f;
  const int64_t n_vec2 = hidden_size >> 2;  // groups of 4 elems = 2 bf16x2
  for (int64_t i = tid; i < n_vec2; i += kThreads) {
    const int64_t base = i << 2;
    const __nv_bfloat162 p0 = reinterpret_cast<const __nv_bfloat162*>(xrow + base)[0];
    const __nv_bfloat162 p1 = reinterpret_cast<const __nv_bfloat162*>(xrow + base)[1];
    const float v0 = __bfloat162float(p0.x);
    const float v1 = __bfloat162float(p0.y);
    const float v2 = __bfloat162float(p1.x);
    const float v3 = __bfloat162float(p1.y);
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

template <bool kRoundIntermediates>
SGL_DEVICE float norm_value(float x, float weight, float rstd) {
  const float value = weight * (rstd * x);
  if constexpr (kRoundIntermediates) {
    return __bfloat162float(__float2bfloat16_rn(value));
  }
  return value;
}

template <bool kRoundIntermediates>
SGL_DEVICE void rope_pair(float x0, float x1, float cos, float sin, float& y0, float& y1) {
  float p0 = x0 * cos;
  float p1 = x1 * cos;
  if constexpr (kRoundIntermediates) {
    // Hopper eager stores the BF16 cosine product before its addcmul update.
    p0 = __bfloat162float(__float2bfloat16_rn(p0));
    p1 = __bfloat162float(__float2bfloat16_rn(p1));
  }
  y0 = fmaf(-sin, x1, p0);
  y1 = fmaf(sin, x0, p1);
}

// Scalar row processing: bit-exact copy of the pinned baseline inner loop.
template <bool kRoundIntermediates>
SGL_DEVICE void process_row(
    int64_t row,
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
    int64_t stride_sin_t,
    int tid,
    int lane,
    int warp_id,
    float* warp_sum,
    float* s_rstd) {
  const int64_t batch = row / seq_len;
  const int64_t token = row - batch * seq_len;
  const int64_t hidden_size = num_heads * head_dim;
  const int64_t half_dim = head_dim >> 1;
  const auto* __restrict__ xrow = x + row * hidden_size;
  auto* __restrict__ outrow = out + row * hidden_size;

  const float rstd = compute_rstd_scalar(xrow, hidden_size, eps, tid, lane, warp_id, warp_sum, s_rstd);

  const int64_t num_pairs = num_heads * half_dim;
  for (int64_t pair = tid; pair < num_pairs; pair += kThreads) {
    const int64_t head = pair / half_dim;
    const int64_t offset = pair - head * half_dim;
    const int64_t idx0 = head * head_dim + offset;
    const int64_t idx1 = idx0 + half_dim;
    const float n0 =
        norm_value<kRoundIntermediates>(__bfloat162float(xrow[idx0]), __bfloat162float(weight[idx0]), rstd);
    const float n1 =
        norm_value<kRoundIntermediates>(__bfloat162float(xrow[idx1]), __bfloat162float(weight[idx1]), rstd);
    const int64_t cos_offset = batch * stride_cos_b + head * stride_cos_h + token * stride_cos_t + offset;
    const int64_t sin_offset = batch * stride_sin_b + head * stride_sin_h + token * stride_sin_t + offset;

    float y0;
    float y1;
    rope_pair<kRoundIntermediates>(
        n0, n1, __bfloat162float(cos[cos_offset]), __bfloat162float(sin[sin_offset]), y0, y1);
    outrow[idx0] = __float2bfloat16_rn(y0);
    outrow[idx1] = __float2bfloat16_rn(y1);
  }
}

// bf16x2-vectorized row processing. Requires every dereferenced address to be
// 4-byte aligned; the host guarantees this via a runtime alignment/stride
// check before selecting this path (even head/token/batch cos/sin strides and
// 4-byte-aligned x, out, weight, cos, sin). head_dim is a compile-time
// template parameter for unrolling. Arithmetic is bit-identical to the scalar
// path.
template <bool kRoundIntermediates, int HEAD_DIM>
SGL_DEVICE void process_row_vec(
    int64_t row,
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ cos,
    const bf16_t* __restrict__ sin,
    const bf16_t* __restrict__ weight,
    bf16_t* __restrict__ out,
    float eps,
    int64_t seq_len,
    int64_t num_heads,
    int64_t stride_cos_b,
    int64_t stride_cos_h,
    int64_t stride_cos_t,
    int64_t stride_sin_b,
    int64_t stride_sin_h,
    int64_t stride_sin_t,
    int tid,
    int lane,
    int warp_id,
    float* warp_sum,
    float* s_rstd) {
  constexpr int64_t head_dim = HEAD_DIM;
  constexpr int64_t half_dim = HEAD_DIM >> 1;
  const int64_t batch = row / seq_len;
  const int64_t token = row - batch * seq_len;
  const int64_t hidden_size = num_heads * head_dim;
  const auto* __restrict__ xrow = x + row * hidden_size;
  auto* __restrict__ outrow = out + row * hidden_size;

  const float rstd = compute_rstd_vec(xrow, hidden_size, eps, tid, lane, warp_id, warp_sum, s_rstd);

  constexpr int64_t PAIRS_PER_HEAD = HEAD_DIM >> 2;  // bf16x2 pairs per head
  const int64_t num_vec = num_heads * PAIRS_PER_HEAD;
  for (int64_t i = tid; i < num_vec; i += kThreads) {
    const int64_t head = i / PAIRS_PER_HEAD;
    const int64_t j = i - head * PAIRS_PER_HEAD;  // pair index within head
    const int64_t e0 = 2 * j;
    const int64_t h = head * head_dim;
    const int64_t idx0a = h + e0;
    const int64_t idx1a = h + half_dim + e0;

    const __nv_bfloat162 x0 = reinterpret_cast<const __nv_bfloat162*>(xrow + idx0a)[0];
    const __nv_bfloat162 x1 = reinterpret_cast<const __nv_bfloat162*>(xrow + idx1a)[0];
    const __nv_bfloat162 w0 = reinterpret_cast<const __nv_bfloat162*>(weight + idx0a)[0];
    const __nv_bfloat162 w1 = reinterpret_cast<const __nv_bfloat162*>(weight + idx1a)[0];

    const float n00 = norm_value<kRoundIntermediates>(__bfloat162float(x0.x), __bfloat162float(w0.x), rstd);
    const float n01 = norm_value<kRoundIntermediates>(__bfloat162float(x0.y), __bfloat162float(w0.y), rstd);
    const float n10 = norm_value<kRoundIntermediates>(__bfloat162float(x1.x), __bfloat162float(w1.x), rstd);
    const float n11 = norm_value<kRoundIntermediates>(__bfloat162float(x1.y), __bfloat162float(w1.y), rstd);

    const int64_t cs_base = batch * stride_cos_b + head * stride_cos_h + token * stride_cos_t;
    const int64_t sin_base = batch * stride_sin_b + head * stride_sin_h + token * stride_sin_t;
    const __nv_bfloat162 c = reinterpret_cast<const __nv_bfloat162*>(cos + cs_base + e0)[0];
    const __nv_bfloat162 sn = reinterpret_cast<const __nv_bfloat162*>(sin + sin_base + e0)[0];

    float y00, y01, y10, y11;
    rope_pair<kRoundIntermediates>(n00, n10, __bfloat162float(c.x), __bfloat162float(sn.x), y00, y10);
    rope_pair<kRoundIntermediates>(n01, n11, __bfloat162float(c.y), __bfloat162float(sn.y), y01, y11);

    __nv_bfloat162 o0, o1;
    o0.x = __float2bfloat16_rn(y00);
    o0.y = __float2bfloat16_rn(y01);
    o1.x = __float2bfloat16_rn(y10);
    o1.y = __float2bfloat16_rn(y11);
    // Stream the outputs, which are not read again by this kernel.
    __stcs(reinterpret_cast<__nv_bfloat162*>(outrow + idx0a), o0);
    __stcs(reinterpret_cast<__nv_bfloat162*>(outrow + idx1a), o1);
  }
}

// Scalar single-side kernel (baseline-identical structure).
template <bool kRoundIntermediates>
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
  const int tid = threadIdx.x + threadIdx.y * 32;
  const int lane = threadIdx.x;
  const int warp_id = threadIdx.y;

  __shared__ float warp_sum[4];
  __shared__ float s_rstd;

  process_row<kRoundIntermediates>(
      row,
      x,
      cos,
      sin,
      weight,
      out,
      eps,
      seq_len,
      num_heads,
      head_dim,
      stride_cos_b,
      stride_cos_h,
      stride_cos_t,
      stride_sin_b,
      stride_sin_h,
      stride_sin_t,
      tid,
      lane,
      warp_id,
      warp_sum,
      &s_rstd);
}

// Vectorized single-side kernel (used by the split path for one side).
template <bool kRoundIntermediates, int HEAD_DIM>
__global__ void ltx2_qknorm_split_rope_kernel_vec(
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ cos,
    const bf16_t* __restrict__ sin,
    const bf16_t* __restrict__ weight,
    bf16_t* __restrict__ out,
    float eps,
    int64_t seq_len,
    int64_t num_heads,
    int64_t stride_cos_b,
    int64_t stride_cos_h,
    int64_t stride_cos_t,
    int64_t stride_sin_b,
    int64_t stride_sin_h,
    int64_t stride_sin_t) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int tid = threadIdx.x + threadIdx.y * 32;
  const int lane = threadIdx.x;
  const int warp_id = threadIdx.y;

  __shared__ float warp_sum[4];
  __shared__ float s_rstd;

  process_row_vec<kRoundIntermediates, HEAD_DIM>(
      row,
      x,
      cos,
      sin,
      weight,
      out,
      eps,
      seq_len,
      num_heads,
      stride_cos_b,
      stride_cos_h,
      stride_cos_t,
      stride_sin_b,
      stride_sin_h,
      stride_sin_t,
      tid,
      lane,
      warp_id,
      warp_sum,
      &s_rstd);
}

// Scalar fused q+k kernel (one launch, both sides, scalar inner loop).
template <bool kRoundIntermediates>
__global__ void ltx2_qknorm_split_rope_fused_kernel(
    const bf16_t* __restrict__ q,
    const bf16_t* __restrict__ q_cos,
    const bf16_t* __restrict__ q_sin,
    const bf16_t* __restrict__ q_weight,
    bf16_t* __restrict__ q_out,
    const bf16_t* __restrict__ k,
    const bf16_t* __restrict__ k_cos,
    const bf16_t* __restrict__ k_sin,
    const bf16_t* __restrict__ k_weight,
    bf16_t* __restrict__ k_out,
    float eps,
    int64_t q_rows,
    int64_t q_seq_len,
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
    int64_t k_stride_sin_t) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int tid = threadIdx.x + threadIdx.y * 32;
  const int lane = threadIdx.x;
  const int warp_id = threadIdx.y;

  __shared__ float warp_sum[4];
  __shared__ float s_rstd;

  if (row < q_rows) {
    process_row<kRoundIntermediates>(
        row,
        q,
        q_cos,
        q_sin,
        q_weight,
        q_out,
        eps,
        q_seq_len,
        num_heads,
        head_dim,
        q_stride_cos_b,
        q_stride_cos_h,
        q_stride_cos_t,
        q_stride_sin_b,
        q_stride_sin_h,
        q_stride_sin_t,
        tid,
        lane,
        warp_id,
        warp_sum,
        &s_rstd);
  } else {
    process_row<kRoundIntermediates>(
        row - q_rows,
        k,
        k_cos,
        k_sin,
        k_weight,
        k_out,
        eps,
        k_seq_len,
        num_heads,
        head_dim,
        k_stride_cos_b,
        k_stride_cos_h,
        k_stride_cos_t,
        k_stride_sin_b,
        k_stride_sin_h,
        k_stride_sin_t,
        tid,
        lane,
        warp_id,
        warp_sum,
        &s_rstd);
  }
}

// Vectorized fused q+k kernel: q side may independently use the vectorized or
// scalar row path, and likewise for k (mixed support). The per-side choice is
// a template parameter selected by the host from runtime alignment checks.
template <bool kRoundIntermediates, int HEAD_DIM, bool Q_VEC, bool K_VEC>
__global__ void ltx2_qknorm_split_rope_fused_kernel_vec(
    const bf16_t* __restrict__ q,
    const bf16_t* __restrict__ q_cos,
    const bf16_t* __restrict__ q_sin,
    const bf16_t* __restrict__ q_weight,
    bf16_t* __restrict__ q_out,
    const bf16_t* __restrict__ k,
    const bf16_t* __restrict__ k_cos,
    const bf16_t* __restrict__ k_sin,
    const bf16_t* __restrict__ k_weight,
    bf16_t* __restrict__ k_out,
    float eps,
    int64_t q_rows,
    int64_t q_seq_len,
    int64_t k_seq_len,
    int64_t num_heads,
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
    int64_t k_stride_sin_t) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int tid = threadIdx.x + threadIdx.y * 32;
  const int lane = threadIdx.x;
  const int warp_id = threadIdx.y;

  __shared__ float warp_sum[4];
  __shared__ float s_rstd;

  if (row < q_rows) {
    if constexpr (Q_VEC) {
      process_row_vec<kRoundIntermediates, HEAD_DIM>(
          row,
          q,
          q_cos,
          q_sin,
          q_weight,
          q_out,
          eps,
          q_seq_len,
          num_heads,
          q_stride_cos_b,
          q_stride_cos_h,
          q_stride_cos_t,
          q_stride_sin_b,
          q_stride_sin_h,
          q_stride_sin_t,
          tid,
          lane,
          warp_id,
          warp_sum,
          &s_rstd);
    } else {
      process_row<kRoundIntermediates>(
          row,
          q,
          q_cos,
          q_sin,
          q_weight,
          q_out,
          eps,
          q_seq_len,
          num_heads,
          HEAD_DIM,
          q_stride_cos_b,
          q_stride_cos_h,
          q_stride_cos_t,
          q_stride_sin_b,
          q_stride_sin_h,
          q_stride_sin_t,
          tid,
          lane,
          warp_id,
          warp_sum,
          &s_rstd);
    }
  } else {
    if constexpr (K_VEC) {
      process_row_vec<kRoundIntermediates, HEAD_DIM>(
          row - q_rows,
          k,
          k_cos,
          k_sin,
          k_weight,
          k_out,
          eps,
          k_seq_len,
          num_heads,
          k_stride_cos_b,
          k_stride_cos_h,
          k_stride_cos_t,
          k_stride_sin_b,
          k_stride_sin_h,
          k_stride_sin_t,
          tid,
          lane,
          warp_id,
          warp_sum,
          &s_rstd);
    } else {
      process_row<kRoundIntermediates>(
          row - q_rows,
          k,
          k_cos,
          k_sin,
          k_weight,
          k_out,
          eps,
          k_seq_len,
          num_heads,
          HEAD_DIM,
          k_stride_cos_b,
          k_stride_cos_h,
          k_stride_cos_t,
          k_stride_sin_b,
          k_stride_sin_h,
          k_stride_sin_t,
          tid,
          lane,
          warp_id,
          warp_sum,
          &s_rstd);
    }
  }
}

struct LTX2QKNormSplitRopeKernel {
  template <bool kRoundIntermediates>
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

    const int64_t q_rows = batch.unwrap() * q_seq_len.unwrap();
    const int64_t k_rows = batch.unwrap() * k_seq_len.unwrap();
    const int64_t total_rows = q_rows + k_rows;
    // Only global early return: nothing to do when both sides are empty. A
    // single empty side is handled per-path below so the non-empty side is
    // still computed (an empty side never selects the fused kernel).
    if (total_rows == 0) {
      return;
    }
    RuntimeCheck(total_rows <= static_cast<int64_t>(UINT32_MAX), "LTX2 QKNorm split-RoPE grid is too large");
    const DLDevice dl_device = device.unwrap();

    // Runtime eligibility for the bf16x2-vectorized row path, per side. Every
    // bf16x2 address the kernel dereferences must be 4-byte aligned. With the
    // guard's last-dim stride 1 and even head_dim this reduces to: the tensor
    // base pointer is 4-byte aligned, and (for cos/sin) the head/token/batch
    // strides are even so per-head offsets stay even. x/out/weight are
    // contiguous with hidden_size % 4 == 0 and half_dim even, so their vector
    // offsets are even whenever the base is 4-byte aligned.
    const auto side_vec_ok = [&](const tvm::ffi::TensorView& x,
                                 const tvm::ffi::TensorView& cos,
                                 const tvm::ffi::TensorView& sin,
                                 const tvm::ffi::TensorView& weight,
                                 const tvm::ffi::TensorView& out) {
      const bool base_aligned = (reinterpret_cast<uintptr_t>(data_ptr(x)) % 4 == 0) &&
                                (reinterpret_cast<uintptr_t>(data_ptr(cos)) % 4 == 0) &&
                                (reinterpret_cast<uintptr_t>(data_ptr(sin)) % 4 == 0) &&
                                (reinterpret_cast<uintptr_t>(data_ptr(weight)) % 4 == 0) &&
                                (reinterpret_cast<uintptr_t>(mutable_data_ptr(out)) % 4 == 0);
      const bool strides_even = (cos.stride(0) % 2 == 0) && (cos.stride(1) % 2 == 0) && (cos.stride(2) % 2 == 0) &&
                                (sin.stride(0) % 2 == 0) && (sin.stride(1) % 2 == 0) && (sin.stride(2) % 2 == 0);
      return base_aligned && strides_even;
    };
    const bool q_vec = side_vec_ok(q, q_cos, q_sin, q_weight, q_out);
    const bool k_vec = side_vec_ok(k, k_cos, k_sin, k_weight, k_out);

    // Use the single-side launch path when only one side has rows.
    if (total_rows <= kSplitThreshold && q_rows > 0 && k_rows > 0) {
      auto launch_vec = [&](auto head_dim_constant, auto q_vec_c, auto k_vec_c) {
        constexpr int HD = decltype(head_dim_constant)::value;
        constexpr bool QV = decltype(q_vec_c)::value;
        constexpr bool KV = decltype(k_vec_c)::value;
        LaunchKernel(dim3(static_cast<uint32_t>(total_rows)), dim3(32, 4), dl_device)(
            ltx2_qknorm_split_rope_fused_kernel_vec<kRoundIntermediates, HD, QV, KV>,
            reinterpret_cast<const bf16_t*>(data_ptr(q)),
            reinterpret_cast<const bf16_t*>(data_ptr(q_cos)),
            reinterpret_cast<const bf16_t*>(data_ptr(q_sin)),
            reinterpret_cast<const bf16_t*>(data_ptr(q_weight)),
            reinterpret_cast<bf16_t*>(mutable_data_ptr(q_out)),
            reinterpret_cast<const bf16_t*>(data_ptr(k)),
            reinterpret_cast<const bf16_t*>(data_ptr(k_cos)),
            reinterpret_cast<const bf16_t*>(data_ptr(k_sin)),
            reinterpret_cast<const bf16_t*>(data_ptr(k_weight)),
            reinterpret_cast<bf16_t*>(mutable_data_ptr(k_out)),
            static_cast<float>(eps),
            q_rows,
            q_seq_len.unwrap(),
            k_seq_len.unwrap(),
            num_heads,
            q_cos.stride(0),
            q_cos.stride(1),
            q_cos.stride(2),
            q_sin.stride(0),
            q_sin.stride(1),
            q_sin.stride(2),
            k_cos.stride(0),
            k_cos.stride(1),
            k_cos.stride(2),
            k_sin.stride(0),
            k_sin.stride(1),
            k_sin.stride(2));
      };
      auto launch_fused_scalar = [&]() {
        LaunchKernel(dim3(static_cast<uint32_t>(total_rows)), dim3(32, 4), dl_device)(
            ltx2_qknorm_split_rope_fused_kernel<kRoundIntermediates>,
            reinterpret_cast<const bf16_t*>(data_ptr(q)),
            reinterpret_cast<const bf16_t*>(data_ptr(q_cos)),
            reinterpret_cast<const bf16_t*>(data_ptr(q_sin)),
            reinterpret_cast<const bf16_t*>(data_ptr(q_weight)),
            reinterpret_cast<bf16_t*>(mutable_data_ptr(q_out)),
            reinterpret_cast<const bf16_t*>(data_ptr(k)),
            reinterpret_cast<const bf16_t*>(data_ptr(k_cos)),
            reinterpret_cast<const bf16_t*>(data_ptr(k_sin)),
            reinterpret_cast<const bf16_t*>(data_ptr(k_weight)),
            reinterpret_cast<bf16_t*>(mutable_data_ptr(k_out)),
            static_cast<float>(eps),
            q_rows,
            q_seq_len.unwrap(),
            k_seq_len.unwrap(),
            num_heads,
            head_dim,
            q_cos.stride(0),
            q_cos.stride(1),
            q_cos.stride(2),
            q_sin.stride(0),
            q_sin.stride(1),
            q_sin.stride(2),
            k_cos.stride(0),
            k_cos.stride(1),
            k_cos.stride(2),
            k_sin.stride(0),
            k_sin.stride(1),
            k_sin.stride(2));
      };
      if (head_dim == 64) {
        if (q_vec && k_vec) {
          launch_vec(std::integral_constant<int, 64>{}, std::true_type{}, std::true_type{});
        } else if (q_vec) {
          launch_vec(std::integral_constant<int, 64>{}, std::true_type{}, std::false_type{});
        } else if (k_vec) {
          launch_vec(std::integral_constant<int, 64>{}, std::false_type{}, std::true_type{});
        } else {
          launch_fused_scalar();
        }
      } else if (head_dim == 128) {
        if (q_vec && k_vec) {
          launch_vec(std::integral_constant<int, 128>{}, std::true_type{}, std::true_type{});
        } else if (q_vec) {
          launch_vec(std::integral_constant<int, 128>{}, std::true_type{}, std::false_type{});
        } else if (k_vec) {
          launch_vec(std::integral_constant<int, 128>{}, std::false_type{}, std::true_type{});
        } else {
          launch_fused_scalar();
        }
      } else {
        launch_fused_scalar();
      }
      return;
    }

    // Large-grid or single-side path: separate launches. Each side independently uses the vectorized kernel when
    // eligible, else the scalar kernel. Empty sides are skipped below.
    auto launch_side = [&](const tvm::ffi::TensorView& x,
                           const tvm::ffi::TensorView& cos,
                           const tvm::ffi::TensorView& sin,
                           const tvm::ffi::TensorView& weight,
                           const tvm::ffi::TensorView& out,
                           int64_t rows,
                           int64_t seq,
                           bool vec_ok) {
      if (rows == 0) {
        return;  // empty side: its output is already zero-sized, nothing to compute
      }
      if (vec_ok && (head_dim == 64 || head_dim == 128)) {
        auto launch_vec_side = [&](auto head_dim_constant) {
          constexpr int HD = decltype(head_dim_constant)::value;
          LaunchKernel(dim3(static_cast<uint32_t>(rows)), dim3(32, 4), dl_device)(
              ltx2_qknorm_split_rope_kernel_vec<kRoundIntermediates, HD>,
              reinterpret_cast<const bf16_t*>(data_ptr(x)),
              reinterpret_cast<const bf16_t*>(data_ptr(cos)),
              reinterpret_cast<const bf16_t*>(data_ptr(sin)),
              reinterpret_cast<const bf16_t*>(data_ptr(weight)),
              reinterpret_cast<bf16_t*>(mutable_data_ptr(out)),
              static_cast<float>(eps),
              seq,
              num_heads,
              cos.stride(0),
              cos.stride(1),
              cos.stride(2),
              sin.stride(0),
              sin.stride(1),
              sin.stride(2));
        };
        if (head_dim == 64) {
          launch_vec_side(std::integral_constant<int, 64>{});
        } else {
          launch_vec_side(std::integral_constant<int, 128>{});
        }
      } else {
        LaunchKernel(dim3(static_cast<uint32_t>(rows)), dim3(32, 4), dl_device)(
            ltx2_qknorm_split_rope_kernel<kRoundIntermediates>,
            reinterpret_cast<const bf16_t*>(data_ptr(x)),
            reinterpret_cast<const bf16_t*>(data_ptr(cos)),
            reinterpret_cast<const bf16_t*>(data_ptr(sin)),
            reinterpret_cast<const bf16_t*>(data_ptr(weight)),
            reinterpret_cast<bf16_t*>(mutable_data_ptr(out)),
            static_cast<float>(eps),
            seq,
            num_heads,
            head_dim,
            cos.stride(0),
            cos.stride(1),
            cos.stride(2),
            sin.stride(0),
            sin.stride(1),
            sin.stride(2));
      }
    };
    launch_side(q, q_cos, q_sin, q_weight, q_out, q_rows, q_seq_len.unwrap(), q_vec);
    launch_side(k, k_cos, k_sin, k_weight, k_out, k_rows, k_seq_len.unwrap(), k_vec);
  }
};

}  // namespace ltx2_qknorm_split_rope_kda

}  // namespace sglang
