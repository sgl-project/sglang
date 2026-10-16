/// A copy of aiter's indexer_qk_rope_quant_and_cache_kernel with an inline
/// 128-point Hadamard, so the front end keeps rotate_activation and the fusion
/// at once.  aiter is untouched; this includes its headers (JIT dependency
/// "aiter"), so the maths match.

#pragma once

#ifndef USE_ROCM
#error "qk_rope_hadamard_quant.cuh targets gfx950; it folds a head onto one wave64"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include "hip_reduce.h"
#include "opus/opus.hpp"
#include <algorithm>
#include <cstdint>

namespace sglang {

namespace dsa_gfx950::qk_rope {

using cache_t = opus::fp8_t;
using scalar_t = opus::bf16_t;

constexpr int HEAD_DIM = 128;
constexpr int ROPE_DIM = 64;
constexpr int LOG2_HEAD_DIM = 7;

// Folded onto one 64-lane wavefront: lane t owns dim t and t+64, so both halves
// of every reduction are an independent DPP tree.  Zero LDS, zero syncthreads,
// and bit-exact against the 128-thread form operand for operand.

constexpr int HALF_DIM = HEAD_DIM / 2;  // 64 == one wavefront

template <typename F>
__device__ __forceinline__ float wred64(float v, F op) {
  return wave_reduce<float, F, 64, true>(v, op);
}

// Interleaved rope on the low 64 dims, held one-per-lane in `v`.  The pair
// comes from a cross-lane shuffle instead of LDS; the arithmetic is the same.
__device__ __forceinline__ float
rope_lo(float v, int t, const scalar_t* __restrict__ cos_ptr, const scalar_t* __restrict__ sin_ptr) {
  const float pair_val = __shfl_xor(v, 1, 64);
  const int cos_idx = t / 2;
  const float cos_v = static_cast<float>(cos_ptr[cos_idx]);
  const float sin_v = static_cast<float>(sin_ptr[cos_idx]);
  v = (t % 2 == 0) ? (v * cos_v - pair_val * sin_v) : (v * cos_v + pair_val * sin_v);
  return static_cast<float>(static_cast<scalar_t>(v));
}

// 128-point Hadamard on a pair (lo = dim t, hi = dim t+64) held in one lane.
__device__ __forceinline__ void hadamard128_wave(float& lo, float& hi, int t, float scale) {
#pragma unroll
  for (int s = 0; s < LOG2_HEAD_DIM - 1; ++s) {  // strides 1, 2, 4, 8, 16, 32
    const int d = 1 << s;
    const float plo = __shfl_xor(lo, d, 64);
    const float phi = __shfl_xor(hi, d, 64);
    const bool a = ((t & d) == 0);  // (dim & d) is the same for dim and dim+64
    lo = a ? (lo + plo) : (plo - lo);
    hi = a ? (hi + phi) : (phi - hi);
  }
  // stride 64: partner of dim t is dim t+64 -- both already in this lane.
  const float nlo = lo + hi;
  const float nhi = lo - hi;  // (dim & 64) != 0 -> partner - v
  lo = nlo * scale;
  hi = nhi * scale;
}

// The k-side epilogue runs in its own block rather than at the end of the
// head_idx==0 blocks, where it would serialise behind them.
__global__ __launch_bounds__(HALF_DIM) void qk_rope_hadamard_quant_kernel(
    const scalar_t* __restrict__ q,
    cache_t* __restrict__ q_out,
    const scalar_t* __restrict__ weights,
    float* __restrict__ weights_out,
    const scalar_t* __restrict__ k,
    cache_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ norm_weight,
    const float* __restrict__ norm_bias,
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ cos_cache,
    const scalar_t* __restrict__ sin_cache,
    const int n_heads,
    const int cache_block_size,
    const int cache_stride,
    const int q_stride_t,
    const int q_stride_h,
    const int q_out_stride_t,
    const int q_out_stride_h,
    const int weights_stride_t,
    const int weights_out_stride_t,
    const int k_stride_t,
    const int cos_stride0,
    const int sin_stride0,
    const float epsilon,
    const float weights_scale,
    const float hadamard_scale,
    const int max_position) {
  const int64_t token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;  // n_heads + 1 of them: the last is the k side
  const int t = threadIdx.x;        // owns dim t and dim t + 64
  const bool do_q = head_idx < n_heads;

  const int64_t slot_idx = slot_mapping[token_idx];
  // Clamped unconditionally: pos indexes cos_cache/sin_cache on both branches,
  // and clamping only the slot_idx < 0 one left the common path free to read at
  // an arbitrary offset.
  int64_t pos = positions[token_idx];
  pos = pos < 0 ? 0 : (pos >= max_position ? max_position - 1 : pos);
  const scalar_t* cos_ptr = cos_cache + pos * cos_stride0;
  const scalar_t* sin_ptr = sin_cache + pos * sin_stride0;

  auto max_func = [](float a, float b) { return fmaxf(a, b); };
  auto sum_func = [](float a, float b) { return a + b; };
  const float q_fp8_max = static_cast<float>(opus::finfo<cache_t>::max());

  // ------------------------------- q side ---------------------------------
  if (do_q) {
    const scalar_t* q_row = q + token_idx * q_stride_t + head_idx * q_stride_h;
    float lo = static_cast<float>(q_row[t]);
    float hi = static_cast<float>(q_row[t + HALF_DIM]);

    lo = rope_lo(lo, t, cos_ptr, sin_ptr);  // dims >=64 untouched

    hadamard128_wave(lo, hi, t, hadamard_scale);
    lo = static_cast<float>(static_cast<scalar_t>(lo));
    hi = static_cast<float>(static_cast<scalar_t>(hi));

    const float a0 = wred64(fabsf(lo), max_func);
    const float a1 = wred64(fabsf(hi), max_func);
    const float q_amax = max_func(a1, a0);  // block_reduce's order

    const float q_inv_fp8_max = 1.0f / q_fp8_max;
    // ue8m0: the scale is rounded up to a power of two.
    const float q_scale = exp2f(ceilf(log2f(fmaxf(q_amax, 1e-10f) * q_inv_fp8_max)));
    const float q_inv_scale = 1.0f / q_scale;
    cache_t* qo = q_out + token_idx * q_out_stride_t + head_idx * q_out_stride_h;
    qo[t] = opus::cast<cache_t>(lo * q_inv_scale);
    qo[t + HALF_DIM] = opus::cast<cache_t>(hi * q_inv_scale);
    if (t == 0) {
      const float w = static_cast<float>(weights[token_idx * weights_stride_t + head_idx]);
      const float head_scale = rsqrtf(static_cast<float>(n_heads));
      const scalar_t w_head = static_cast<scalar_t>(w * head_scale);
      const float softmax_scale = weights_scale / head_scale;
      weights_out[token_idx * weights_out_stride_t + head_idx] = static_cast<float>(w_head) * q_scale * softmax_scale;
    }
  }

  if (do_q || slot_idx < 0) return;

  // ------------------------------- k side ---------------------------------
  const scalar_t* k_row = k + token_idx * k_stride_t;
  const float xlo = static_cast<float>(k_row[t]);
  const float xhi = static_cast<float>(k_row[t + HALF_DIM]);

  const float s0 = wred64(xlo, sum_func);
  const float s1 = wred64(xhi, sum_func);
  const float mean = sum_func(s1, s0) / static_cast<float>(HEAD_DIM);

  const float clo = xlo - mean;
  const float chi = xhi - mean;
  const float q0 = wred64(clo * clo, sum_func);
  const float q1 = wred64(chi * chi, sum_func);
  const float ss = sum_func(q1, q0);
  const float inv_std = rsqrtf(ss / static_cast<float>(HEAD_DIM) + epsilon);

  float klo = clo * inv_std * norm_weight[t] + norm_bias[t];
  float khi = chi * inv_std * norm_weight[t + HALF_DIM] + norm_bias[t + HALF_DIM];
  klo = static_cast<float>(static_cast<scalar_t>(klo));
  khi = static_cast<float>(static_cast<scalar_t>(khi));

  klo = rope_lo(klo, t, cos_ptr, sin_ptr);

  hadamard128_wave(klo, khi, t, hadamard_scale);
  klo = static_cast<float>(static_cast<scalar_t>(klo));
  khi = static_cast<float>(static_cast<scalar_t>(khi));

  const float m0 = wred64(fabsf(klo), max_func);
  const float m1 = wred64(fabsf(khi), max_func);
  const float k_amax = max_func(m1, m0);
  const float k_scale = exp2f(ceilf(log2f(fmaxf(k_amax, 1e-4f) / q_fp8_max)));  // ue8m0

  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;
  const int64_t page_base = block_idx * cache_block_size * cache_stride;
  if (t == 0) {
    // One fp32 scale per token, after the page's K bytes: quant block == HEAD_DIM.
    const int64_t dst_scale_idx = page_base + cache_block_size * HEAD_DIM + block_offset * 4;
    reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = k_scale;
  }
  const float k_inv_scale = 1.0f / k_scale;

  // Preshuffled page layout: 16x16 (token x dim) tiles.
  constexpr int TILE = 16;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int dim = t + h * HALF_DIM;
    const int token_tile_id = block_offset / TILE;
    const int token_in_tile = block_offset % TILE;
    const int col_tile_id = dim / TILE;
    const int col_in_tile = dim % TILE;
    const int64_t dst_offset = page_base + token_tile_id * (TILE * HEAD_DIM) + col_tile_id * (TILE * TILE) +
                               token_in_tile * TILE + col_in_tile;
    kv_cache[dst_offset] = opus::cast<cache_t>((h == 0 ? klo : khi) * k_inv_scale);
  }
}

}  // namespace dsa_gfx950::qk_rope

struct QkRopeHadamardQuantKernel {
  /// AITER indexer_qk_rope_quant_and_cache plus the inline 128-point Hadamard,
  /// specialised to ue8m0 scales, the preshuffled cache, interleaved rope and a
  /// 128-wide quant block; the Python gate admits nothing else.
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView q_out,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView weights_out,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView kv_cache,
      const tvm::ffi::TensorView slot_mapping,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView norm_bias,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView cos_cache,
      const tvm::ffi::TensorView sin_cache,
      double epsilon,
      double weights_scale) {
    using namespace host;
    namespace impl = dsa_gfx950::qk_rope;

    /// ``q_out`` and ``kv_cache`` are reinterpret_cast to a 1-byte cache_t and
    /// written as raw fp8, so what matters is the width, not the flavour: a
    /// wider dtype fills half the row and leaves the rest untouched.
    const auto byte_width = [](DLDataType dtype) -> int64_t { return (int64_t{dtype.bits} * dtype.lanes + 7) / 8; };

    const int num_tokens = static_cast<int>(std::min(k.size(0), slot_mapping.size(0)));
    // num_tokens becomes gridDim.x and indexes all six per-token tensors, but
    // only two of them took part in the min above.
    RuntimeCheck(
        q.size(0) >= num_tokens && q_out.size(0) >= num_tokens && weights.size(0) >= num_tokens &&
            weights_out.size(0) >= num_tokens && positions.size(0) >= num_tokens,
        "every per-token tensor must have at least num_tokens rows");
    const int head_dim = static_cast<int>(k.size(1));
    const int n_heads = static_cast<int>(q.size(1));
    RuntimeCheck(cos_cache.dim() == 2 && sin_cache.dim() == 2, "cos/sin must be 2-D");
    const int rope_dim = static_cast<int>(cos_cache.size(1)) * 2;
    const int cache_block_size = static_cast<int>(kv_cache.size(1));
    const int cache_stride = static_cast<int>(kv_cache.size(2));
    const int max_position = static_cast<int>(cos_cache.size(0));

    RuntimeCheck(head_dim == impl::HEAD_DIM, "head_dim must be 128");
    RuntimeCheck(rope_dim == impl::ROPE_DIM, "rope_dim must be 64");
    RuntimeCheck(is_type<bf16_t>(q.dtype()) && is_type<bf16_t>(k.dtype()), "q/k must be bf16");
    RuntimeCheck(is_type<bf16_t>(weights.dtype()), "weights must be bf16");
    RuntimeCheck(
        byte_width(q_out.dtype()) == 1 && byte_width(kv_cache.dtype()) == 1,
        "q_out and kv_cache must be 1-byte fp8, got ",
        q_out.dtype(),
        "/",
        kv_cache.dtype());
    RuntimeCheck(is_type<fp32_t>(weights_out.dtype()), "weights_out must be fp32");
    RuntimeCheck(
        is_type<fp32_t>(norm_weight.dtype()) && is_type<fp32_t>(norm_bias.dtype()), "norm params must be fp32");
    RuntimeCheck(cos_cache.stride(1) == 1 && sin_cache.stride(1) == 1, "cos/sin last dim must be contiguous");
    RuntimeCheck(is_type<int64_t>(slot_mapping.dtype()), "slot_mapping must be int64");
    RuntimeCheck(is_type<int64_t>(positions.dtype()), "positions must be int64");

    const auto* qp = reinterpret_cast<const impl::scalar_t*>(q.data_ptr());
    auto* qop = reinterpret_cast<impl::cache_t*>(q_out.data_ptr());
    const auto* wp = reinterpret_cast<const impl::scalar_t*>(weights.data_ptr());
    auto* wop = static_cast<fp32_t*>(weights_out.data_ptr());
    const auto* kp = reinterpret_cast<const impl::scalar_t*>(k.data_ptr());
    auto* cp = reinterpret_cast<impl::cache_t*>(kv_cache.data_ptr());
    const auto* sp = static_cast<const int64_t*>(slot_mapping.data_ptr());
    const auto* nwp = static_cast<const fp32_t*>(norm_weight.data_ptr());
    const auto* nbp = static_cast<const fp32_t*>(norm_bias.data_ptr());
    const auto* pp = static_cast<const int64_t*>(positions.data_ptr());
    const auto* cosp = reinterpret_cast<const impl::scalar_t*>(cos_cache.data_ptr());
    const auto* sinp = reinterpret_cast<const impl::scalar_t*>(sin_cache.data_ptr());

    // rotate_activation(x) == hadamard_transform(x, scale=x.size(-1) ** -0.5)
    const float had_scale = 1.0f / sqrtf(static_cast<float>(impl::HEAD_DIM));

    // The wave path indexes the head dimension directly, so the layout
    // invariants it relies on are asserted rather than carried as arguments.
    RuntimeCheck(q.stride(2) == 1 && q_out.stride(2) == 1 && k.stride(1) == 1, "wave kernel needs contiguous head dim");
    RuntimeCheck(
        weights.stride(1) == 1 && weights_out.stride(1) == 1, "wave kernel needs contiguous head axis on weights");

    LaunchKernel(dim3(num_tokens, n_heads + 1), dim3(impl::HALF_DIM), q.device())(
        impl::qk_rope_hadamard_quant_kernel,
        qp,
        qop,
        wp,
        wop,
        kp,
        cp,
        sp,
        nwp,
        nbp,
        pp,
        cosp,
        sinp,
        n_heads,
        cache_block_size,
        cache_stride,
        static_cast<int>(q.stride(0)),
        static_cast<int>(q.stride(1)),
        static_cast<int>(q_out.stride(0)),
        static_cast<int>(q_out.stride(1)),
        static_cast<int>(weights.stride(0)),
        static_cast<int>(weights_out.stride(0)),
        static_cast<int>(k.stride(0)),
        static_cast<int>(cos_cache.stride(0)),
        static_cast<int>(sin_cache.stride(0)),
        static_cast<float>(epsilon),
        static_cast<float>(weights_scale),
        had_scale,
        max_position);
  }
};

}  // namespace sglang
