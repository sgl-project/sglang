// DeepSeek-V3.2 DSA indexer only.
//
// K: LayerNorm(128) + RoPE (interleaved pairs, first 64 dims) + fp8
//    act-quant (absmax over 128 dims) [+ write into the paged index-K
//    cache], in one call, replacing 3-4 separate ops (k_norm -> rotary_emb
//    -> act_quant_cpu -> set_index_k_scale_buffer) and their intermediate
//    tensor materializations.
// Q: RoPE (interleaved pairs, first 64 dims) + fp8 act-quant (absmax over
//    128 dims, per head) + head-gate weight scaling, in one call.
//
// V3.2 drops the Hadamard incoherence rotation entirely (logit-preserving,
// see rotate_activation/_maybe_rotate in dsa_indexer.py), so neither kernel
// here applies it.

#include "common.h"
#include "vec.h"

namespace {

using namespace at::vec;

constexpr int64_t kHeadDim = 128;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kHalfRopeDim = kRopeDim / 2;  // 32
constexpr float kFP8Max = 448.0f;
constexpr float kFP8Min = -448.0f;
constexpr float kMinScaleAmax = 1.0e-4f;

// In-place LayerNorm: reads the bf16 raw input `in`, converts to fp32,
// computes standard (biased) variance, and writes normalized+affine
// output y = (x - mean) / sqrt(var + eps) * weight + bias into `data`.
// The conversion is fused into the reduction pass below (no separate
// bf16->fp32 copy pass first).
inline void layer_norm_head_dim(
    float* __restrict__ data,             // [128], write-only: normalized+affine output
    const at::BFloat16* __restrict__ in,  // [128], bf16 raw input
    const at::BFloat16* __restrict__ weight,
    const at::BFloat16* __restrict__ bias,
    float eps) {
  using fVec = Vectorized<float>;
  constexpr int64_t kStep = 2 * fVec::size();

  fVec sum_fvec(0.f), sum2_fvec(0.f);
  for (int64_t d = 0; d < kHeadDim; d += kStep) {
    auto [x0, x1] = load_float_vec2(in + d);
    sum_fvec += x0;
    sum_fvec += x1;
    sum2_fvec += x0 * x0;
    sum2_fvec += x1 * x1;
    x0.store(data + d);
    x1.store(data + d + fVec::size());
  }
  const float sum = vec_reduce_sum(sum_fvec);
  const float sum2 = vec_reduce_sum(sum2_fvec);

  const float mean = sum / kHeadDim;
  const float var = sum2 / kHeadDim - mean * mean;
  const float inv_std = 1.0f / std::sqrt(var + eps);

  const fVec mean_fvec(mean), inv_std_fvec(inv_std);
  for (int64_t d = 0; d < kHeadDim; d += kStep) {
    fVec x0 = fVec::loadu(data + d);
    fVec x1 = fVec::loadu(data + d + fVec::size());
    auto [w0, w1] = load_float_vec2(weight + d);
    auto [b0, b1] = load_float_vec2(bias + d);
    x0 = (x0 - mean_fvec) * inv_std_fvec * w0 + b0;
    x1 = (x1 - mean_fvec) * inv_std_fvec * w1 + b1;
    x0.store(data + d);
    x1.store(data + d + fVec::size());
  }
}

// Core of one RoPE round, shared by apply_rope_K and apply_rope_Q: given a
// pair of "raw interleaved" fVecs (already loaded/converted by the caller,
// fp32 either way), deinterleaves them into an all-real and all-imag
// vector -- aligned with cos_sin_row's per-pair layout, so cos_vec/sin_vec
// need no permute -- rotates, re-interleaves, and stores the result at
// `out + 2*p` / `out + 2*p + fVec::size()`.
inline __attribute__((always_inline)) void rotate_interleaved_pair_block(
    const Vectorized<float>& raw_a,
    const Vectorized<float>& raw_b,
    const Vectorized<float>& cos_vec,
    const Vectorized<float>& sin_vec,
    float* __restrict__ out,
    int64_t p) {
  using fVec = Vectorized<float>;
  auto [re, im] = deinterleave2(raw_a, raw_b);
  fVec out_re = re * cos_vec - im * sin_vec;
  fVec out_im = re * sin_vec + im * cos_vec;
  auto [oa, ob] = interleave2(out_re, out_im);
  oa.store(out + 2 * p);
  ob.store(out + 2 * p + fVec::size());
}

// RoPE on the leading `kRopeDim` (64) dims of a `kHeadDim`-length fp32
// buffer, using the "interleaved pairs" convention (pairs (2k, 2k+1) rotate
// with frequency index k, k in [0, 32)) -- matches is_neox_style=False /
// indexer_rope_interleave=True, the layout `use_dsa_indexer_fusion` is
// gated on. The trailing 64 dims pass through unchanged. In-place: used by
// the K path, where `data` is already the (fp32) LayerNorm output, not the
// raw bf16 input.
inline void apply_rope_K(
    float* __restrict__ data,  // [128], in/out
    const at::BFloat16* __restrict__ cos_sin_row) {  // [64]: [:32]=cos, [32:64]=sin
  using fVec = Vectorized<float>;
  constexpr int64_t kPairStep = fVec::size();

  for (int64_t p = 0; p < kHalfRopeDim; p += 2 * kPairStep) {
    auto [c0, c1] = load_float_vec2(cos_sin_row + p);
    auto [s0, s1] = load_float_vec2(cos_sin_row + kHalfRopeDim + p);

    fVec raw_a0 = fVec::loadu(data + 2 * p);
    fVec raw_b0 = fVec::loadu(data + 2 * p + kPairStep);
    const int64_t p1 = p + kPairStep;
    fVec raw_a1 = fVec::loadu(data + 2 * p1);
    fVec raw_b1 = fVec::loadu(data + 2 * p1 + kPairStep);
    rotate_interleaved_pair_block(raw_a0, raw_b0, c0, s0, data, p);
    rotate_interleaved_pair_block(raw_a1, raw_b1, c1, s1, data, p1);
  }
}

// Same RoPE as apply_rope_K, but reads the bf16 raw input `in` directly and
// writes the fp32 result (rotated leading 64 dims, converted-passthrough
// trailing 64 dims) into `out` -- fuses the bf16->fp32 conversion into the
// same pass. Used by the Q path, where RoPE is the first (and only)
// transform applied to the raw bf16 input (no LayerNorm precedes it).
//
// `load_float_vec2(in + 2*p)` gives exactly the (raw_a, raw_b) pair
// rotate_interleaved_pair_block expects (bf16->fp32 conversion is
// transparent to the interleaved-pairs layout), so the leading 64 dims
// fuse the conversion into the same deinterleave2/rotate/interleave2 round
// used by apply_rope_K; the trailing 64 dims are a plain vectorized
// convert-passthrough.
inline void apply_rope_Q(
    float* __restrict__ out,              // [128], write-only
    const at::BFloat16* __restrict__ in,  // [128], bf16 raw input
    const at::BFloat16* __restrict__ cos_sin_row) {  // [64]: [:32]=cos, [32:64]=sin
  using fVec = Vectorized<float>;
  constexpr int64_t kPairStep = fVec::size();

  for (int64_t p = 0; p < kHalfRopeDim; p += 2 * kPairStep) {
    auto [c0, c1] = load_float_vec2(cos_sin_row + p);
    auto [s0, s1] = load_float_vec2(cos_sin_row + kHalfRopeDim + p);

    auto [raw_a0, raw_b0] = load_float_vec2(in + 2 * p);
    rotate_interleaved_pair_block(raw_a0, raw_b0, c0, s0, out, p);

    const int64_t p1 = p + kPairStep;
    auto [raw_a1, raw_b1] = load_float_vec2(in + 2 * p1);
    rotate_interleaved_pair_block(raw_a1, raw_b1, c1, s1, out, p1);
  }

  constexpr int64_t kStep = 2 * fVec::size();
  for (int64_t d = kRopeDim; d < kHeadDim; d += kStep) {
    auto [x0, x1] = load_float_vec2(in + d);
    x0.store(out + d);
    x1.store(out + d + fVec::size());
  }
}

// fp8 (e4m3fn) absmax quantization over `kHeadDim` (128) fp32 elements.
// Returns the scale.
inline float quantize_fp8_head_dim(at::Float8_e4m3fn* __restrict__ out, const float* __restrict__ data) {
  float amax = 0.f;
  for (int64_t d = 0; d < kHeadDim; ++d) {
    amax = std::max(amax, std::abs(data[d]));
  }
  const float scale = std::max(amax, kMinScaleAmax) / kFP8Max;
  const float inv_scale = 1.0f / scale;
  for (int64_t d = 0; d < kHeadDim; ++d) {
    float v = data[d] * inv_scale;
    v = std::min(std::max(v, kFP8Min), kFP8Max);
    out[d] = at::Float8_e4m3fn(v);
  }
  return scale;
}

}  // namespace

// K: LayerNorm + RoPE (first 64 dims) + fp8 act-quant + paged index-K cache
// write, in one call. Buffer layout matches DSATokenToKVPool's
// index_k_with_scale_buffer / set_k_cpu+set_s_cpu: page_size*(128+4) bytes
// per page, i.e. [page_size*128 bytes fp8][page_size*4 bytes fp32 scale].
void fused_k_indexer_norm_rope_store_cpu(
    at::Tensor& k_input,        // (N, 128), bf16; last-dim contiguous (row stride may exceed 128)
    at::Tensor& cache,          // (num_pages, page_size*132), uint8
    at::Tensor& out_cache_loc,  // (N,), int32 or int64
    at::Tensor& weight,         // (128,), bf16 -- LayerNorm gamma
    at::Tensor& bias,           // (128,), bf16 -- LayerNorm beta
    double eps,
    at::Tensor& cos_sin_cache,  // (max_pos, 64), bf16
    at::Tensor& positions,      // (N,), int64
    int64_t page_size) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_input);
  CHECK_INPUT(cache);
  CHECK_INPUT(out_cache_loc);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(positions);
  CHECK_DIM(1, out_cache_loc);
  CHECK_DIM(1, positions);
  CHECK_DIM(2, k_input);
  CHECK_DIM(2, cache);
  CHECK_EQ(k_input.size(1), kHeadDim);
  CHECK_EQ(out_cache_loc.size(0), k_input.size(0));
  CHECK_EQ(positions.size(0), k_input.size(0));
  CHECK_EQ(weight.numel(), kHeadDim);
  CHECK_EQ(bias.numel(), kHeadDim);
  CHECK_EQ(cos_sin_cache.size(-1), kRopeDim);
  TORCH_CHECK(cache.scalar_type() == at::kByte, "cache must be uint8");
  TORCH_CHECK(k_input.scalar_type() == at::kBFloat16, "k_input must be bf16, got ", k_input.scalar_type());
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bf16, got ", weight.scalar_type());
  TORCH_CHECK(bias.scalar_type() == at::kBFloat16, "bias must be bf16, got ", bias.scalar_type());
  TORCH_CHECK(
      cos_sin_cache.scalar_type() == at::kBFloat16, "cos_sin_cache must be bf16, got ", cos_sin_cache.scalar_type());
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64, got ", positions.scalar_type());

  const int64_t N = k_input.size(0);
  if (N == 0) return;
  const int64_t k_stride0 = k_input.stride(0);
  const int64_t buf_numel_per_page = cache.size(1);
  const int64_t s_offset_in_page = page_size * kHeadDim;

  const bool loc_is_i64 = (out_cache_loc.scalar_type() == at::kLong);
  const int32_t* loc_i32 = loc_is_i64 ? nullptr : out_cache_loc.data_ptr<int32_t>();
  const int64_t* loc_i64 = loc_is_i64 ? out_cache_loc.data_ptr<int64_t>() : nullptr;

  const int64_t* pos_ptr = positions.data_ptr<int64_t>();

  uint8_t* cache_ptr = cache.data_ptr<uint8_t>();
  const at::BFloat16* k_ptr = k_input.data_ptr<at::BFloat16>();
  const at::BFloat16* weight_ptr = weight.data_ptr<at::BFloat16>();
  const at::BFloat16* bias_ptr = bias.data_ptr<at::BFloat16>();
  const at::BFloat16* cos_sin_ptr = cos_sin_cache.data_ptr<at::BFloat16>();
  const float eps_f = static_cast<float>(eps);

  at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float buf[kHeadDim];
    for (int64_t i = begin; i < end; ++i) {
      layer_norm_head_dim(buf, k_ptr + i * k_stride0, weight_ptr, bias_ptr, eps_f);

      const int64_t position = pos_ptr[i];
      apply_rope_K(buf, cos_sin_ptr + position * kRopeDim);

      const int64_t token_loc = loc_is_i64 ? loc_i64[i] : static_cast<int64_t>(loc_i32[i]);
      const int64_t page_idx = token_loc / page_size;
      const int64_t token_off = token_loc % page_size;
      uint8_t* page_ptr = cache_ptr + page_idx * buf_numel_per_page;
      auto* k_dst = reinterpret_cast<at::Float8_e4m3fn*>(page_ptr + token_off * kHeadDim);
      const float scale = quantize_fp8_head_dim(k_dst, buf);
      auto* s_dst = reinterpret_cast<float*>(page_ptr + s_offset_in_page + token_off * 4);
      *s_dst = scale;
    }
  });
}

// Q: RoPE (first 64 dims) + fp8 act-quant (per head) + head-gate weight
// scaling, in one call. weights_out[b, h] = weight_raw[b, h] * weight_scale
// * fp8_scale[b, h].
std::tuple<at::Tensor, at::Tensor> fused_q_indexer_rope_first_quant_cpu(
    at::Tensor& q_input,       // (N, H, 128), bf16, contiguous
    at::Tensor& weight_raw,    // (N, H), bf16; last-dim contiguous
    double weight_scale,
    at::Tensor& cos_sin_cache,  // (max_pos, 64), bf16
    at::Tensor& positions) {    // (N,), int64
  CHECK_INPUT(q_input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight_raw);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(positions);
  CHECK_DIM(1, positions);
  CHECK_DIM(3, q_input);
  CHECK_DIM(2, weight_raw);
  CHECK_EQ(q_input.size(2), kHeadDim);
  CHECK_EQ(cos_sin_cache.size(-1), kRopeDim);
  TORCH_CHECK(q_input.scalar_type() == at::kBFloat16, "q_input must be bf16, got ", q_input.scalar_type());
  TORCH_CHECK(weight_raw.scalar_type() == at::kBFloat16, "weight_raw must be bf16, got ", weight_raw.scalar_type());
  TORCH_CHECK(
      cos_sin_cache.scalar_type() == at::kBFloat16, "cos_sin_cache must be bf16, got ", cos_sin_cache.scalar_type());
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64, got ", positions.scalar_type());

  const int64_t N = q_input.size(0);
  const int64_t H = q_input.size(1);
  CHECK_EQ(weight_raw.size(0), N);
  CHECK_EQ(weight_raw.size(1), H);
  CHECK_EQ(positions.size(0), N);

  at::Tensor q_fp8 = at::empty({N, H, kHeadDim}, q_input.options().dtype(at::kFloat8_e4m3fn));
  at::Tensor weights_out = at::empty({N, H, 1}, q_input.options().dtype(at::kFloat));
  if (N == 0 || H == 0) return {q_fp8, weights_out};

  const int64_t weight_stride0 = weight_raw.stride(0);
  const int64_t* pos_ptr = positions.data_ptr<int64_t>();

  const at::BFloat16* q_ptr = q_input.data_ptr<at::BFloat16>();
  const at::BFloat16* weight_ptr = weight_raw.data_ptr<at::BFloat16>();
  const at::BFloat16* cos_sin_ptr = cos_sin_cache.data_ptr<at::BFloat16>();
  float* weights_out_ptr = weights_out.data_ptr<float>();
  at::Float8_e4m3fn* q_fp8_ptr = q_fp8.data_ptr<at::Float8_e4m3fn>();
  const float weight_scale_f = static_cast<float>(weight_scale);

  at::parallel_for(0, N * H, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float buf[kHeadDim];
    for (int64_t work = begin; work < end; ++work) {
      const int64_t token = work / H;
      const int64_t head = work % H;

      const int64_t position = pos_ptr[token];
      apply_rope_Q(buf, q_ptr + work * kHeadDim, cos_sin_ptr + position * kRopeDim);

      const float scale = quantize_fp8_head_dim(q_fp8_ptr + work * kHeadDim, buf);

      const float weight_val = static_cast<float>(weight_ptr[token * weight_stride0 + head]);
      weights_out_ptr[work] = weight_val * weight_scale_f * scale;
    }
  });

  return {q_fp8, weights_out};
}
