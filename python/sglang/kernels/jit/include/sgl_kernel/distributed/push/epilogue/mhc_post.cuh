#pragma once

#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/push/epilogue/common.cuh>

#include <type_traits>

namespace sglang::device::distributed::push {

using tvm::ffi::TensorView, tvm::ffi::Optional;

/**
 * \brief DeepSeek V4 (HC = 4): `out = x`; the post mix of `x` into the 4 residual streams, rounded in
 *        `mhc_post_split_h`'s order, to `mhc_out`; with `kNorm`, the pre-collapse of the 4 rounded streams, rounded to
 *        bf16 and RMS-normed into `normalized`; with `kQuant`, `normalized` also as fp8 e4m3 with one ue8m0 scale per
 *        32 values, written into a single 128-row swizzled scale tile, whose padding rows (rows >= num_tokens) the
 *        constructor zeroes, spread over every thread of the grid.
 */
template <uint32_t kHiddenSize, bool kNorm, bool kQuant>
struct MHCPost {
 public:
  using Vec = VecStorage<bf16_t>;
  static constexpr uint32_t kNumStreams = 4;  // residual streams
  using Mix = AlignedVector<float, kNumStreams>;
  template <uint32_t kClusterSize, uint32_t kNumWarps>
  using Smem = std::conditional_t<kNorm, ClusterRowSum<kClusterSize, kNumWarps, 1>, NoSmem>;
  static constexpr uint32_t kRowVecs = kHiddenSize / 8;
  static constexpr uint32_t kScaleWords = kHiddenSize / 32 * 128 / 4;  // u32 words of the scale tile
  static_assert(!kQuant || kNorm, "the quantized row is the normalized one");
  static_assert(!kQuant || kHiddenSize % 128 == 0, "a scale word holds 4 groups of 32 values");

  struct Params {
    bf16_t* out;                // [rows, kHiddenSize]
    bf16_t* mhc_out;            // [rows, 4, kHiddenSize]
    const bf16_t* residual;     // [rows, 4, kHiddenSize]
    const float* post;          // [rows, 4]
    const float* comb;          // [rows, 4, 4], comb[r][c] mixes residual stream r into output stream c
    const float* pre;           // [rows, 4] (kNorm)
    const bf16_t* norm_weight;  // [kHiddenSize] (kNorm)
    float eps;                  // (kNorm)
    bf16_t* normalized;         // [rows, kHiddenSize] (kNorm)
    fp8_e4m3_t* quantized;      // [rows, kHiddenSize] (kQuant)
    uint8_t* scales;            // [kHiddenSize / 32 * 128] (kQuant)
    uint32_t num_tokens;
  };

  SGL_DEVICE MHCPost(const Params& p, uint32_t row, uint32_t col, uint32_t vid)
      : m_params(p), m_row(row), m_col(col), m_vid(vid) {
    load_inputs();
    if constexpr (kQuant) zero_scale_padding();
  }

  template <typename SmemT>
  SGL_DEVICE void store(const Vec& x, SmemT& smem) const {
    if constexpr (kNorm) smem.wait_peers();
    x.store(assume_global(m_params.out), m_vid);
    fp32x2_t collapsed[kVecPairs<bf16_t>] = {};
    Vec stream[kNumStreams];
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      const auto v = cast<fp32x2_t>(x[j]);
      fp32x2_t rv[kNumStreams];
#pragma unroll
      for (uint32_t r = 0; r < kNumStreams; ++r) {
        rv[r] = cast<fp32x2_t>(m_residual[r][j]);
      }
#pragma unroll
      for (uint32_t c = 0; c < kNumStreams; ++c) {
        fp32x2_t acc{m_post[c] * v.x, m_post[c] * v.y};
#pragma unroll
        for (uint32_t r = 0; r < kNumStreams; ++r) {
          acc.x = fmaf(m_comb[r][c], rv[r].x, acc.x);
          acc.y = fmaf(m_comb[r][c], rv[r].y, acc.y);
        }
        stream[c][j] = cast<bf16x2_t>(acc);
        if constexpr (kNorm) {
          const auto rounded = cast<fp32x2_t>(stream[c][j]);
          collapsed[j].x = fmaf(rounded.x, m_pre[c], collapsed[j].x);
          collapsed[j].y = fmaf(rounded.y, m_pre[c], collapsed[j].y);
        }
      }
    }
#pragma unroll
    for (uint32_t c = 0; c < kNumStreams; ++c) {
      stream[c].store(assume_global(m_params.mhc_out), (m_row * kNumStreams + c) * kRowVecs + m_col);
    }
    if constexpr (kNorm) {
      Vec y;
#pragma unroll
      for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
        y[j] = cast<bf16x2_t>(collapsed[j]);
      }
      const auto normed = smem.rms_norm(y, m_norm_weight, m_params.eps, 1.0f / kHiddenSize);
      normed.store(assume_global(m_params.normalized), m_vid);
      if constexpr (kQuant) quantize(normed);
    }
  }

  static Params make_params(
      CallShape& shape,  //
      TensorView out,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb,
      Optional<TensorView> pre,
      Optional<TensorView> norm_weight,
      float eps,
      Optional<TensorView> normalized,
      Optional<TensorView> quantized,
      Optional<TensorView> scales) {
    CHECK_HOST(pre.has_value() == kNorm && norm_weight.has_value() == kNorm && normalized.has_value() == kNorm);
    CHECK_HOST(quantized.has_value() == kQuant && scales.has_value() == kQuant);
    // check MHC params
    using host::TensorMatcher;
    auto& T = shape.m_tokens;
    auto& device = shape.m_device;
    shape.verify_tensor(out, kHiddenSize);
    TensorMatcher({T, 4, kHiddenSize})  // mhc_out, residual
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(mhc_out)
        .verify(residual);
    TensorMatcher({T, 4})  // post, pre
        .with_dtype<fp32_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(post)
        .verify(pre.value_or(post));
    TensorMatcher({T, 4, 4})  // comb
        .with_dtype<fp32_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(comb);

    if constexpr (kNorm) {
      shape.verify_tensor(*normalized, kHiddenSize);
      shape.verify_weight(*norm_weight, kHiddenSize);
    }

    const auto num_tokens = shape.get_num_tokens();
    if constexpr (kQuant) {
      CHECK_HOST(num_tokens <= 128) << "supports up to 128 tokens for quant";
      TensorMatcher({T, kHiddenSize})  // quantized (output)
          .with_dtype<fp8_e4m3_t>()
          .with_device(device)
          .ensure_alignment(8)
          .verify(*quantized);
      TensorMatcher({kHiddenSize / 32 * 128})  // scale (output)
          .with_dtype<uint8_t>()
          .with_device(device)
          .ensure_alignment(16)
          .verify(*scales);
    }

    return {
        .out = get_pointer<bf16_t>(out),
        .mhc_out = get_pointer<bf16_t>(mhc_out),
        .residual = get_pointer<const bf16_t>(residual),
        .post = get_pointer<const float>(post),
        .comb = get_pointer<const float>(comb),
        .pre = get_pointer<const float>(pre),
        .norm_weight = get_pointer<const bf16_t>(norm_weight),
        .eps = eps,
        .normalized = get_pointer<bf16_t>(normalized),
        .quantized = get_pointer<fp8_e4m3_t>(quantized),
        .scales = get_pointer<uint8_t>(scales),
        .num_tokens = static_cast<uint32_t>(num_tokens),
    };
  }

 private:
  /// Zeroes the scale words of the tile's padding rows; a word holds one row's scales of 4 groups, so these words
  /// are disjoint from the bytes `store` writes.
  SGL_DEVICE void zero_scale_padding() const {
    const auto scales = reinterpret_cast<uint32_t*>(assume_global(m_params.scales));
    for (uint32_t w = m_row * kRowVecs + m_col; w < kScaleWords; w += m_params.num_tokens * kRowVecs) {
      const uint32_t swizzled = w % 128;
      if (swizzled / 4 + (swizzled % 4) * 32 >= m_params.num_tokens) scales[w] = 0;
    }
  }

  SGL_DEVICE void load_inputs() {
#pragma unroll
    for (uint32_t s = 0; s < kNumStreams; ++s) {
      m_residual[s].load(assume_global(m_params.residual), (m_row * kNumStreams + s) * kRowVecs + m_col);
      m_comb[s].load(assume_global(m_params.comb), m_row * kNumStreams + s);
    }
    m_post.load(assume_global(m_params.post), m_row);
    if constexpr (kNorm) {
      m_pre.load(assume_global(m_params.pre), m_row);
      m_norm_weight.load(assume_global(m_params.norm_weight), m_col);
    }
  }

  /// fp8 e4m3 codes of one vector and, from the first of the 4 threads of a 32-value group, its ue8m0 scale.
  SGL_DEVICE void quantize(const Vec& value) const {
    fp32x2_t v[kVecPairs<bf16_t>];
    float amax = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      v[j] = cast<fp32x2_t>(value[j]);
      amax = fmaxf(amax, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
    }
    amax = warp::reduce_max</*kStart=*/1, /*kFinish=*/4>(amax);
    const float normalized = amax * (1.0f / 448.0f);
    const uint32_t bits = __float_as_uint(normalized);
    const uint32_t exponent = (bits >> 23) & 255;
    const uint32_t mantissa = bits & 0x7fffff;
    const bool bump = mantissa != 0 && !(exponent == 0 && mantissa <= 0x400000);
    const uint32_t sf = normalized <= 0 ? 0 : min(exponent + uint32_t(bump), 254u);
    const float inv_scale = __uint_as_float(sf == 0 ? 0 : (254 - sf) << 23);
    AlignedVector<fp8x2_e4m3_t, kVecPairs<bf16_t>> q;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      q[j] = to_fp8_e4m3(fp32x2_t{v[j].x * inv_scale, v[j].y * inv_scale});
    }
    q.store(assume_global(m_params.quantized), m_vid);
    if (m_col % 4 == 0) {
      const uint32_t g = m_col / 4;
      m_params.scales[(g / 4) * 512 + ((m_row % 32) * 4 + (m_row / 32) % 4) * 4 + g % 4] = sf;
    }
  }

  const Params& m_params;
  uint32_t m_row;
  uint32_t m_col;
  uint32_t m_vid;
  Vec m_residual[kNumStreams];
  Mix m_comb[kNumStreams];  // m_comb[r][c]
  Mix m_post;
  Mix m_pre;
  Vec m_norm_weight;
};

static_assert(Epilogue<MHCPost<5120, false, false>>);

}  // namespace sglang::device::distributed::push
