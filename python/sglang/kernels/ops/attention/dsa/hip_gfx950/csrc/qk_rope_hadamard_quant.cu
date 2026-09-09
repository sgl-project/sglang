// A copy of aiter's indexer_qk_rope_quant_and_cache_kernel with an inline
// 128-point Hadamard, so the front end keeps rotate_activation and the fusion
// at once.  aiter is untouched; this includes its headers, so the maths match.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include "hip_reduce.h"
#include "opus/opus.hpp"

namespace {

using cache_t = opus::fp8_t;
using scalar_t = opus::bf16_t;

constexpr int HEAD_DIM = 128;
constexpr int ROPE_DIM = 64;
constexpr int LOG2_HEAD_DIM = 7;

// Folded onto one 64-lane wavefront: lane t owns dim t and t+64, so both halves
// of every reduction are an independent DPP tree.  Zero LDS, zero syncthreads,
// and bit-exact against the 128-thread form operand for operand.

constexpr int HALF_DIM = HEAD_DIM / 2; // 64 == one wavefront

template <typename F> __device__ __forceinline__ float wred64(float v, F op) {
  return wave_reduce<float, F, 64, true>(v, op);
}

// rope on the low 64 dims, held one-per-lane in `v`.  `partner` comes from a
// cross-lane shuffle instead of LDS; the arithmetic is byte-for-byte the same.
__device__ __forceinline__ float rope_lo(float v, int t, bool is_neox,
                                         const scalar_t *__restrict__ cos_ptr,
                                         const scalar_t *__restrict__ sin_ptr) {
  float pair_val;
  int cos_idx;
  bool first;
  if (is_neox) {
    constexpr int HALF = ROPE_DIM / 2;
    pair_val = __shfl_xor(v, HALF, 64);
    cos_idx = t < HALF ? t : t - HALF;
    first = t < HALF;
  } else {
    pair_val = __shfl_xor(v, 1, 64);
    cos_idx = t / 2;
    first = (t % 2 == 0);
  }
  float cos_v, sin_v;
  cos_v = static_cast<float>(cos_ptr[cos_idx]);
  sin_v = static_cast<float>(sin_ptr[cos_idx]);
  v = first ? (v * cos_v - pair_val * sin_v) : (v * cos_v + pair_val * sin_v);
  return static_cast<float>(static_cast<scalar_t>(v));
}

// 128-point Hadamard on a pair (lo = dim t, hi = dim t+64) held in one lane.
__device__ __forceinline__ void hadamard128_wave(float &lo, float &hi, int t,
                                                 float scale) {
#pragma unroll
  for (int s = 0; s < LOG2_HEAD_DIM - 1; ++s) { // strides 1, 2, 4, 8, 16, 32
    const int d = 1 << s;
    const float plo = __shfl_xor(lo, d, 64);
    const float phi = __shfl_xor(hi, d, 64);
    const bool a = ((t & d) == 0); // (dim & d) is the same for dim and dim+64
    lo = a ? (lo + plo) : (plo - lo);
    hi = a ? (hi + phi) : (phi - hi);
  }
  // stride 64: partner of dim t is dim t+64 -- both already in this lane.
  const float nlo = lo + hi;
  const float nhi = lo - hi; // (dim & 64) != 0 -> partner - v
  lo = nlo * scale;
  hi = nhi * scale;
}

// The k-side epilogue runs in its own block rather than at the end of the
// head_idx==0 blocks, where it would serialise behind them.
__global__ __launch_bounds__(HALF_DIM) void qk_rope_hadamard_quant_kernel(
    const scalar_t *__restrict__ q, cache_t *__restrict__ q_out,
    const scalar_t *__restrict__ weights, float *__restrict__ weights_out,
    const scalar_t *__restrict__ k, cache_t *__restrict__ kv_cache,
    const int64_t *__restrict__ slot_mapping,
    const float *__restrict__ norm_weight, const float *__restrict__ norm_bias,
    const int64_t *__restrict__ positions,
    const scalar_t *__restrict__ cos_cache,
    const scalar_t *__restrict__ sin_cache, const int n_heads,
    const int cache_block_size, const int cache_stride, const int q_stride_t,
    const int q_stride_h, const int q_out_stride_t, const int q_out_stride_h,
    const int weights_stride_t, const int weights_out_stride_t,
    const int k_stride_t, const int cos_stride0, const int sin_stride0,
    const float epsilon, const float weights_scale, const float hadamard_scale,
    const int max_position) {
  constexpr bool use_ue8m0 = true;
  constexpr bool preshuffle = true;
  constexpr bool is_neox = false;
  constexpr bool compute_all_q_rope = true;
  constexpr int qblk = HEAD_DIM;
  const int64_t token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int t = threadIdx.x; // owns dim t and dim t + 64
  const bool do_q = head_idx < n_heads;
  const bool do_k = head_idx == n_heads;
  // num_tokens == gridDim.x by construction, so the bound check is free
  if (head_idx >= n_heads + 1)
    return;

  const int64_t slot_idx = slot_mapping[token_idx];
  if (!compute_all_q_rope && slot_idx < 0)
    return;
  // Clamped unconditionally: pos indexes cos_cache/sin_cache on both branches,
  // and clamping only the slot_idx < 0 one left the common path free to read at
  // an arbitrary offset.
  int64_t pos = positions[token_idx];
  pos = pos < 0 ? 0 : (pos >= max_position ? max_position - 1 : pos);
  const scalar_t *cos_ptr = cos_cache + pos * cos_stride0;
  const scalar_t *sin_ptr = sin_cache + pos * sin_stride0;

  auto max_func = [](float a, float b) { return fmaxf(a, b); };
  auto sum_func = [](float a, float b) { return a + b; };
  const float q_fp8_max = static_cast<float>(opus::finfo<cache_t>::max());

  // ------------------------------- q side ---------------------------------
  if (do_q) {
    const scalar_t *q_row = q + token_idx * q_stride_t + head_idx * q_stride_h;
    float lo = static_cast<float>(q_row[t]);
    float hi = static_cast<float>(q_row[t + HALF_DIM]);

    float q_amax;
    lo = rope_lo(lo, t, is_neox, cos_ptr, sin_ptr); // dims >=64 untouched

    {
      hadamard128_wave(lo, hi, t, hadamard_scale);
      lo = static_cast<float>(static_cast<scalar_t>(lo));
      hi = static_cast<float>(static_cast<scalar_t>(hi));
    }

    const float a0 = wred64(fabsf(lo), max_func);
    const float a1 = wred64(fabsf(hi), max_func);
    q_amax = max_func(a1, a0); // block_reduce's order

    const float q_inv_fp8_max = 1.0f / q_fp8_max;
    float q_scale = fmaxf(q_amax, 1e-10f) * q_inv_fp8_max;
    if (use_ue8m0)
      q_scale = exp2f(ceilf(log2f(q_scale)));
    const float q_inv_scale = 1.0f / q_scale;
    cache_t *qo =
        q_out + token_idx * q_out_stride_t + head_idx * q_out_stride_h;
    qo[t] = opus::cast<cache_t>(lo * q_inv_scale);
    qo[t + HALF_DIM] = opus::cast<cache_t>(hi * q_inv_scale);
    if (t == 0) {
      const float w =
          static_cast<float>(weights[token_idx * weights_stride_t + head_idx]);
      const float head_scale = rsqrtf(static_cast<float>(n_heads));
      const scalar_t w_head = static_cast<scalar_t>(w * head_scale);
      const float softmax_scale = weights_scale / head_scale;
      weights_out[token_idx * weights_out_stride_t + head_idx] =
          static_cast<float>(w_head) * q_scale * softmax_scale;
    }
  }

  if (!do_k || slot_idx < 0)
    return;

  // ------------------------------- k side ---------------------------------
  const scalar_t *k_row = k + token_idx * k_stride_t;
  float xlo = static_cast<float>(k_row[t]);
  float xhi = static_cast<float>(k_row[t + HALF_DIM]);

  float klo, khi, k_amax;
  const float s0 = wred64(xlo, sum_func);
  const float s1 = wred64(xhi, sum_func);
  const float mean = sum_func(s1, s0) / static_cast<float>(HEAD_DIM);

  float clo = xlo - mean;
  float chi = xhi - mean;
  const float q0 = wred64(clo * clo, sum_func);
  const float q1 = wred64(chi * chi, sum_func);
  const float ss = sum_func(q1, q0);
  const float inv_std = rsqrtf(ss / static_cast<float>(HEAD_DIM) + epsilon);

  klo = clo * inv_std * norm_weight[t] + norm_bias[t];
  khi = chi * inv_std * norm_weight[t + HALF_DIM] + norm_bias[t + HALF_DIM];
  klo = static_cast<float>(static_cast<scalar_t>(klo));
  khi = static_cast<float>(static_cast<scalar_t>(khi));

  klo = rope_lo(klo, t, is_neox, cos_ptr, sin_ptr);

  {
    hadamard128_wave(klo, khi, t, hadamard_scale);
    klo = static_cast<float>(static_cast<scalar_t>(klo));
    khi = static_cast<float>(static_cast<scalar_t>(khi));
  }

  const float m0 = wred64(fabsf(klo), max_func);
  const float m1 = wred64(fabsf(khi), max_func);
  k_amax = max_func(m1, m0);
  float k_scale = fmaxf(k_amax, 1e-4f) / q_fp8_max;
  if (use_ue8m0)
    k_scale = exp2f(ceilf(log2f(k_scale)));

  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;
  const int64_t page_base = block_idx * cache_block_size * cache_stride;
  if (t == 0) {
    const int64_t dst_scale_idx = page_base + cache_block_size * HEAD_DIM +
                                  block_offset * HEAD_DIM * 4 / qblk;
    reinterpret_cast<float *>(kv_cache)[dst_scale_idx / 4] = k_scale;
  }
  const float k_inv_scale = 1.0f / k_scale;

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int dim = t + h * HALF_DIM;
    int64_t dst_offset;
    if (preshuffle) {
      constexpr int TILE = 16;
      const int token_tile_id = block_offset / TILE;
      const int token_in_tile = block_offset % TILE;
      const int col_tile_id = dim / TILE;
      const int col_in_tile = dim % TILE;
      dst_offset = page_base + token_tile_id * (TILE * HEAD_DIM) +
                   col_tile_id * (TILE * TILE) + token_in_tile * TILE +
                   col_in_tile;
    } else {
      dst_offset = page_base + block_offset * HEAD_DIM + dim;
    }
    kv_cache[dst_offset] =
        opus::cast<cache_t>((h == 0 ? klo : khi) * k_inv_scale);
  }
}

} // namespace

void indexer_qk_rope_hadamard_quant_and_cache(
    at::Tensor q, at::Tensor q_out, at::Tensor weights, at::Tensor weights_out,
    at::Tensor k, at::Tensor kv_cache, at::Tensor slot_mapping,
    at::Tensor norm_weight, at::Tensor norm_bias, at::Tensor positions,
    at::Tensor cos_cache, at::Tensor sin_cache, double epsilon,
    int64_t quant_block_size, const std::string &scale_fmt,
    double weights_scale, bool preshuffle, bool is_neox,
    bool compute_all_q_rope, bool hadamard) {
  const int num_tokens = std::min<int>(k.size(0), slot_mapping.size(0));
  // num_tokens becomes gridDim.x and indexes all six per-token tensors, but
  // only two of them took part in the min above.
  TORCH_CHECK(q.size(0) >= num_tokens && q_out.size(0) >= num_tokens &&
                  weights.size(0) >= num_tokens &&
                  weights_out.size(0) >= num_tokens &&
                  positions.size(0) >= num_tokens,
              "every per-token tensor must have at least num_tokens rows");
  const int head_dim = k.size(1);
  const int n_heads = q.size(1);
  const int rope_dim = cos_cache.size(-1) * 2;
  const int cache_block_size = kv_cache.size(1);
  const int cache_stride = kv_cache.size(2);
  const int max_position = cos_cache.size(0);
  const bool use_ue8m0 = scale_fmt == "ue8m0";

  TORCH_CHECK(head_dim == HEAD_DIM, "head_dim must be 128");
  TORCH_CHECK(rope_dim == ROPE_DIM, "rope_dim must be 64");
  TORCH_CHECK(quant_block_size == head_dim,
              "quant_block_size must equal head_dim");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                  k.scalar_type() == at::kBFloat16,
              "q/k must be bf16");
  TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bf16");
  // q_out and kv_cache are reinterpret_cast to a 1-byte cache_t and written as
  // raw fp8; a wider dtype fills half the row and leaves the rest untouched.
  TORCH_CHECK(q_out.element_size() == 1 && kv_cache.element_size() == 1,
              "q_out and kv_cache must be 1-byte fp8, got ",
              q_out.scalar_type(), "/", kv_cache.scalar_type());
  TORCH_CHECK(weights_out.scalar_type() == at::kFloat,
              "weights_out must be fp32");
  TORCH_CHECK(norm_weight.scalar_type() == at::kFloat &&
                  norm_bias.scalar_type() == at::kFloat,
              "norm params must be fp32");
  TORCH_CHECK(cos_cache.dim() == 2 && sin_cache.dim() == 2,
              "cos/sin must be 2-D");
  TORCH_CHECK(cos_cache.stride(1) == 1 && sin_cache.stride(1) == 1,
              "cos/sin last dim must be contiguous");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong,
              "slot_mapping must be int64");
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");

  const auto *qp = reinterpret_cast<const scalar_t *>(q.data_ptr());
  auto *qop = reinterpret_cast<cache_t *>(q_out.data_ptr());
  const auto *wp = reinterpret_cast<const scalar_t *>(weights.data_ptr());
  auto *wop = weights_out.data_ptr<float>();
  const auto *kp = reinterpret_cast<const scalar_t *>(k.data_ptr());
  auto *cp = reinterpret_cast<cache_t *>(kv_cache.data_ptr());
  const auto *sp = slot_mapping.data_ptr<int64_t>();
  const auto *nwp = norm_weight.data_ptr<float>();
  const auto *nbp = norm_bias.data_ptr<float>();
  const auto *pp = positions.data_ptr<int64_t>();
  const auto *cosp = reinterpret_cast<const scalar_t *>(cos_cache.data_ptr());
  const auto *sinp = reinterpret_cast<const scalar_t *>(sin_cache.data_ptr());

  // rotate_activation(x) == hadamard_transform(x, scale=x.size(-1) ** -0.5)
  const float had_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
  auto stream = c10::cuda::getCurrentCUDAStream();

  // The wave path indexes the head dimension directly, so the layout invariants
  // it relies on are asserted rather than carried as arguments.
  TORCH_CHECK(q.stride(2) == 1 && q_out.stride(2) == 1 && k.stride(1) == 1,
              "wave kernel needs contiguous head dim");
  TORCH_CHECK(weights.stride(1) == 1 && weights_out.stride(1) == 1,
              "wave kernel needs contiguous head axis on weights");
  // The kernel is written for exactly this configuration, so refuse any call
  // that is not on it rather than silently computing something else.
  TORCH_CHECK(hadamard, "the fused indexer requires the inline Hadamard");
  TORCH_CHECK(use_ue8m0 && preshuffle && !is_neox && compute_all_q_rope &&
                  quant_block_size == HEAD_DIM,
              "specialised for ue8m0 + preshuffle + interleaved rope + "
              "compute_all_q_rope + quant_block == 128");
  dim3 wgrid(num_tokens, n_heads + 1);
  dim3 wblock(HALF_DIM);
  qk_rope_hadamard_quant_kernel<<<wgrid, wblock, 0, stream>>>(
      qp, qop, wp, wop, kp, cp, sp, nwp, nbp, pp, cosp, sinp, n_heads,
      cache_block_size, cache_stride, (int)q.stride(0), (int)q.stride(1),
      (int)q_out.stride(0), (int)q_out.stride(1), (int)weights.stride(0),
      (int)weights_out.stride(0), (int)k.stride(0), (int)cos_cache.stride(0),
      (int)sin_cache.stride(0), static_cast<float>(epsilon),
      static_cast<float>(weights_scale), had_scale, max_position);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "indexer_qk_rope_hadamard_quant_and_cache",
      &indexer_qk_rope_hadamard_quant_and_cache,
      "AITER indexer_qk_rope_quant_and_cache + optional inline 128-pt Hadamard",
      pybind11::arg("q"), pybind11::arg("q_out"), pybind11::arg("weights"),
      pybind11::arg("weights_out"), pybind11::arg("k"),
      pybind11::arg("kv_cache"), pybind11::arg("slot_mapping"),
      pybind11::arg("norm_weight"), pybind11::arg("norm_bias"),
      pybind11::arg("positions"), pybind11::arg("cos_cache"),
      pybind11::arg("sin_cache"), pybind11::arg("epsilon"),
      pybind11::arg("quant_block_size"), pybind11::arg("scale_fmt"),
      pybind11::arg("weights_scale"), pybind11::arg("preshuffle"),
      pybind11::arg("is_neox"), pybind11::arg("compute_all_q_rope"),
      pybind11::arg("hadamard"));
}
