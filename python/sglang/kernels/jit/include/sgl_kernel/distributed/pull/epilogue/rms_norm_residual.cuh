#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/distributed/pull/epilogue/common.cuh>

#include <tvm/ffi/optional.h>

#include <cstdint>

namespace sglang::device::distributed::pull {

using tvm::ffi::TensorView, tvm::ffi::Optional;

/**
 * \brief Post-norm and residual add on this rank's rows of a row-sharded residual stream (Gemma 4), every op rounded
 *        to bf16 as Hugging Face's layer: `a = bf16(x * post_norm_weight * rsqrt(mean(x^2) + eps))`,
 *        `v = bf16(a + residual)`, with `kScalar` then `v = bf16(v * scalar)` (Gemma 4's layer scalar), `residual = v`
 *        (local, this rank's rows only). Without `kNextNorm` (the last layer) `v` is multicast as bf16. With it, the
 *        next block's input norm and its FP8 GEMM's quant:
 *        `y = bf16(v * next_norm_weight * rsqrt(mean(v^2) + eps))`, then per 128 values
 *        `scale = max(amax, 1e-10) / 448`, `q = fp8(y / scale)`; `q` and the fp32 scales are multicast.
 */
template <uint32_t kHiddenSize, bool kNextNorm, bool kScalar>
struct RMSNormResidual {
 public:
  using Tile = RowTile<kHiddenSize>;
  using Vec = VecStorage<bf16_t>;
  static constexpr uint32_t kNumSums = kNextNorm ? 2 : 1;
  static constexpr uint32_t kGroupVecs = 128 / 8;  // vectors per quant group: 16 consecutive threads
  static constexpr float kFp8Max = 448.0f;

  struct Params {
    bf16_t* residual;                // [shard rows, hidden], this rank's rows of the stream, updated in place
    const bf16_t* post_norm_weight;  // [hidden]
    const bf16_t* next_norm_weight;  // [hidden] (kNextNorm)
    const void* scalar;              // [1], fp32 or bf16 (kScalar)
    bool scalar_bf16;                // (kScalar)
    float eps;                       //
    uint8_t* out_mc;                 // [num_tokens, hidden]: fp8 (kNextNorm) or bf16, multicast VA
    float* scales_mc;                // [num_tokens, hidden / 128] fp32, multicast VA (kNextNorm)
    int64_t scale_row;               // scales stride between rows, in floats
    int64_t scale_group;             // scales stride between groups, in floats
  };

  template <uint32_t kNumRows>
  struct Inputs {
    Vec residual[kNumRows];
  };

  SGL_DEVICE explicit RMSNormResidual(const Params& p) : m_params(p) {
    m_post_norm_weight.load(assume_global(p.post_norm_weight), threadIdx.x);
    if constexpr (kNextNorm) m_next_norm_weight.load(assume_global(p.next_norm_weight), threadIdx.x);
    if constexpr (kScalar) {
      // Hugging Face's layer scalar is a bf16 buffer; an fp32 one is rounded to bf16
      const float scalar = p.scalar_bf16 ? static_cast<float>(*static_cast<const bf16_t*>(p.scalar))
                                         : *static_cast<const float*>(p.scalar);
      m_scalar = cast<bf16x2_t>(fp32x2_t{scalar, scalar});
    }
  }

  template <uint32_t kNumRows>
  SGL_DEVICE Inputs<kNumRows> prefetch(RowGroup group) const {
    Inputs<kNumRows> in;
#pragma unroll
    for (uint32_t u = 0; u < kNumRows; ++u) {
      in.residual[u].load(assume_global(m_params.residual), group.template local_vec<Tile::kRowVecs>(u));
    }
    return in;
  }

  template <uint32_t kNumRows, typename Smem>
  SGL_DEVICE void
  store(RowGroup group, const Vec (&x)[kNumRows], const Inputs<kNumRows>& in, Smem& smem, uint32_t phase) const {
    float sum[kNumRows];
#pragma unroll
    for (uint32_t u = 0; u < kNumRows; ++u) {
      sum[u] = square_sum(x[u]);
    }
    smem.template publish<kNumRows>(sum, phase, 0);

    Vec stream[kNumRows];
#pragma unroll
    for (uint32_t u = 0; u < kNumRows; ++u) {
      const auto r = rms_factor(smem.total(u, phase, 0), 1.0f / kHiddenSize, m_params.eps);
      const auto a = bf16_norm_scale(x[u], m_post_norm_weight, r);
#pragma unroll
      for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
        // bf16 ops, each rounded once: bf16(a + residual), then bf16(* scalar)
        stream[u][j] = __hadd2(a[j], in.residual[u][j]);
        if constexpr (kScalar) stream[u][j] = __hmul2(stream[u][j], m_scalar);
      }
      stream[u].store(assume_global(m_params.residual), group.template local_vec<Tile::kRowVecs>(u));
      if constexpr (!kNextNorm) {
        ptx::st_multimem_16B(stream[u], m_params.out_mc, group.template vec<Tile::kRowVecs>(u));
      }
    }
    if constexpr (kNextNorm) {
#pragma unroll
      for (uint32_t u = 0; u < kNumRows; ++u) {
        sum[u] = square_sum(stream[u]);
      }
      smem.template publish<kNumRows>(sum, phase, 1);
#pragma unroll
      for (uint32_t u = 0; u < kNumRows; ++u) {
        const auto r = rms_factor(smem.total(u, phase, 1), 1.0f / kHiddenSize, m_params.eps);
        quantize(bf16_norm_scale(stream[u], m_next_norm_weight, r), group, u);
      }
    }
  }

  /// `shape` holds the all-reduced tensor's rows; `residual` this rank's `rows` of the stream.
  static Params make_params(
      CallShape& shape,  //
      TensorView residual,
      int64_t rows,
      TensorView post_norm_weight,
      Optional<TensorView> next_norm_weight,
      Optional<TensorView> scalar,
      float eps,
      TensorView out,
      int64_t out_mc,
      Optional<TensorView> scales,
      int64_t scales_mc) {
    using host::TensorMatcher;
    CHECK_HOST(
        next_norm_weight.has_value() == kNextNorm && scales.has_value() == kNextNorm && scalar.has_value() == kScalar);
    TensorMatcher({-1, kHiddenSize})  // residual
        .with_dtype<bf16_t>()
        .with_device(shape.m_device)
        .ensure_alignment(16)
        .verify(residual);
    CHECK_HOST(residual.size(0) == rows) << "residual holds " << residual.size(0) << " rows, the shard " << rows;
    shape.verify_weight(post_norm_weight, kHiddenSize);
    if constexpr (kScalar) {
      TensorMatcher({1})  // scalar
          .with_dtype<fp32_t, bf16_t>()
          .with_device(shape.m_device)
          .verify(*scalar);
    }
    CHECK_HOST(out_mc != 0) << "the output needs a multicast address";
    Params p{
        .residual = get_pointer<bf16_t>(residual),
        .post_norm_weight = get_pointer<const bf16_t>(post_norm_weight),
        .next_norm_weight = get_pointer<const bf16_t>(next_norm_weight),
        .scalar = kScalar ? (*scalar).data_ptr() : nullptr,
        .scalar_bf16 = kScalar && (*scalar).dtype().code == kDLBfloat,
        .eps = eps,
        .out_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(out_mc)),
        .scales_mc = nullptr,
        .scale_row = 0,
        .scale_group = 0,
    };
    if constexpr (kNextNorm) {
      shape.verify_weight(*next_norm_weight, kHiddenSize);
      TensorMatcher({shape.m_tokens, kHiddenSize})  // out
          .with_dtype<fp8_e4m3_t>()
          .with_device(shape.m_device)
          .verify(out);
      TensorMatcher({shape.m_tokens, kHiddenSize / 128})  // scales, e.g. DeepGEMM's MN-major layout
          .with_strides({-1, -1})
          .with_dtype<fp32_t>()
          .with_device(shape.m_device)
          .verify(*scales);
      CHECK_HOST(scales_mc != 0) << "the scales need a multicast address";
      p.scales_mc = reinterpret_cast<float*>(static_cast<uintptr_t>(scales_mc));
      p.scale_row = (*scales).stride(0);
      p.scale_group = (*scales).stride(1);
    } else {
      shape.verify_tensor(out, kHiddenSize);
    }
    return p;
  }

 private:
  SGL_DEVICE static float square_sum(const Vec& x) {
    float acc0 = 0.0f, acc1 = 0.0f;
    bf16_square_sum(x, acc0, acc1);
    return acc0 + acc1;
  }

  /// fp8 e4m3 codes of this thread's vector of row `group.row + u`, stored by the even lane of each pair as one
  /// 16-byte multicast store, and the fp32 scale of its 128-value group (16 consecutive threads) from the group's first
  /// thread.
  SGL_DEVICE void quantize(const Vec& y, RowGroup group, uint32_t u) const {
    const uint32_t col = threadIdx.x;
    fp32x2_t f[kVecPairs<bf16_t>];
    float amax = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      f[j] = cast<fp32x2_t>(y[j]);
      amax = fmaxf(amax, fmaxf(fabsf(f[j].x), fabsf(f[j].y)));
    }
    amax = warp::reduce_max<kGroupVecs>(amax);
    const float scale = fmaxf(amax, 1e-10f) / kFp8Max;
    AlignedVector<fp8x2_e4m3_t, kVecPairs<bf16_t>> q;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      q[j] = to_fp8_e4m3(fp32x2_t{f[j].x / scale, f[j].y / scale});
    }
    const auto words = *reinterpret_cast<const uint2*>(&q);
    const uint32_t peer_x = __shfl_xor_sync(0xffffffff, words.x, 1);
    const uint32_t peer_y = __shfl_xor_sync(0xffffffff, words.y, 1);
    if (col % 2 == 0) {
      const uint4 pair = make_uint4(words.x, words.y, peer_x, peer_y);
      ptx::st_multimem_16B(
          pair, m_params.out_mc, group.template vec<Tile::kRowVecs>(u) / 2);  // fp8: 2 vectors per 16 B
    }
    if (col % kGroupVecs == 0) {
      const auto row = int64_t{group.row} + u;
      const auto dst = m_params.scales_mc + row * m_params.scale_row + (col / kGroupVecs) * m_params.scale_group;
      ptx::multimem_store_relaxed(reinterpret_cast<uint32_t*>(dst), __float_as_uint(scale));
    }
  }

  const Params& m_params;
  bf16x2_t m_scalar;  // kScalar
  Vec m_post_norm_weight;
  Vec m_next_norm_weight;  // kNextNorm
};

}  // namespace sglang::device::distributed::pull
