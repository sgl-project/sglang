/**
 * \brief Fused RMS norm + per-token NVFP4 quantization for the MuseSpark v1.2 expert input.
 *
 * x:            [M, K] bf16, the pooler output.
 * weight:       [K] bf16 effective norm weight; null when the norm is weightless.
 * normed_out:   [M, K] bf16, the normed rows; null unless requested.
 * fp4_out:      [M, K/2] u8, packed e2m1 pairs (low nibble first).
 * sf_out:       [M, K/16] u8, e4m3 block scales in linear (row-major) layout.
 * per_token_scale: [M] fp32; per-token mode only.
 * global_scale: [1] fp32 static encode scale (w13_input_scale_quant); static mode only.
 *
 * The quantizer reproduces FlashInfer's `nvfp4_quantize(per_token_activation=True,
 * backend="cute-dsl", sfLayout=layout_linear)` byte for byte: the same
 * rcp.approx / cvt.rn.satfinite sequence applied to the bf16-rounded normed row.
 * Static mode reproduces FlashInfer's CUDA `fp4_quantize(global_sf, sf_vec_size=16,
 * is_sf_swizzled_layout=False)` the same way.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace sglang {

struct NormFp4QuantParams {
  const void* __restrict__ x;
  const void* __restrict__ weight;
  void* __restrict__ normed_out;
  void* __restrict__ fp4_out;
  void* __restrict__ sf_out;
  void* __restrict__ per_token_scale;
  const float* __restrict__ global_scale;
  float eps;
  float global_scale_inv;
  int num_rows;
};

namespace device::norm_fp4_quant {

inline constexpr int kBlock = 16;  // elements per e4m3 scale factor
inline constexpr float kFloat32Max = 3.4028234663852886e38f;

SGL_DEVICE float rcp_approx_ftz(float a) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

SGL_DEVICE uint32_t cvt_f32_to_e4m3(float a) {
  uint32_t r;
  asm("{\n"
      "  .reg .b16 pair;\n"
      "  cvt.rn.satfinite.e4m3x2.f32 pair, 0f00000000, %1;\n"
      "  cvt.u32.u16 %0, pair;\n"
      "}"
      : "=r"(r)
      : "f"(a));
  return r;
}

// FlashInfer's nvfp4_compute_output_scale, kept instruction for instruction.
SGL_DEVICE float output_scale(uint32_t fp8, float encode_scale) {
  float r;
  asm("{\n"
      "  .reg .pred p_zero;\n"
      "  .reg .b16 fp8_pair;\n"
      "  .reg .b32 h2_32;\n"
      "  .reg .b16 h_lo, h_hi;\n"
      "  .reg .f32 scale_f32, rcp_gs, product, result;\n"
      "  cvt.u16.u32 fp8_pair, %1;\n"
      "  cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;\n"
      "  mov.b32 {h_lo, h_hi}, h2_32;\n"
      "  cvt.f32.f16 scale_f32, h_lo;\n"
      "  rcp.approx.ftz.f32 rcp_gs, %2;\n"
      "  mul.f32 product, scale_f32, rcp_gs;\n"
      "  rcp.approx.ftz.f32 result, product;\n"
      "  setp.eq.f32 p_zero, scale_f32, 0f00000000;\n"
      "  selp.f32 %0, 0f00000000, result, p_zero;\n"
      "}"
      : "=f"(r)
      : "r"(fp8), "f"(encode_scale));
  return r;
}

SGL_DEVICE uint32_t cvt_e2m1x8(const float* v) {
  uint32_t r;
  asm("{\n"
      "  .reg .b8 b0, b1, b2, b3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "  mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(r)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return r;
}

SGL_DEVICE float mul_f32(float a, float b) {
  float r;
  asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

SGL_DEVICE float output_scale_static(uint32_t fp8, float block_max, float global_scale) {
  if (block_max == 0.0f) return 0.0f;
  fp8_e4m3_t sf;
  sf.__x = static_cast<uint8_t>(fp8);
  return rcp_approx_ftz(mul_f32(static_cast<float>(sf), rcp_approx_ftz(global_scale)));
}

}  // namespace device::norm_fp4_quant

template <
    int64_t kHidden,
    int kNumThreads,
    bool kUsePDL,
    bool kQuant,
    bool kNormedOut,
    bool kHasWeight,
    bool kStaticScale>
__global__ void __launch_bounds__(kNumThreads)
    muse_spark_norm_fp4_quant_kernel(const __grid_constant__ NormFp4QuantParams params) {
  using namespace device;
  using namespace device::norm_fp4_quant;

  constexpr int kPacks = kHidden / (kNumThreads * kBlock);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  static_assert(kHidden % (kNumThreads * kBlock) == 0);
  static_assert(kQuant || kNormedOut);
  static_assert(kQuant || !kStaticScale);

  using BF16Vec = AlignedVector<bf16x2_t, 4>;  // 8 bf16 = 16B; two per block

  const int64_t row = blockIdx.x;
  const auto x = static_cast<const bf16_t*>(params.x) + row * kHidden;
  const auto weight = static_cast<const bf16_t*>(params.weight);

  __shared__ float smem[kNumWarps == 1 ? 1 : 2 * kNumWarps];

  fp32x2_t vals[kPacks][kBlock / 2];

  PDLWaitPrimary<kUsePDL>();

  float sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int blk = threadIdx.x + p * kNumThreads;
    BF16Vec lo, hi;
    lo.load(x, blk * 2);
    hi.load(x, blk * 2 + 1);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      vals[p][q] = cast<fp32x2_t>(lo[q]);
      vals[p][q + 4] = cast<fp32x2_t>(hi[q]);
    }
#pragma unroll
    for (int q = 0; q < kBlock / 2; ++q) {
      sum_sq = fmaf(vals[p][q].x, vals[p][q].x, sum_sq);
      sum_sq = fmaf(vals[p][q].y, vals[p][q].y, sum_sq);
    }
  }

  sum_sq = cta::reduce_sum<kNumThreads>(sum_sq, smem);
  const float rstd = math::rsqrt(sum_sq / kHidden + params.eps);

  // The quantizer must see the bf16-rounded row, as the two-kernel path did.
  bf16x2_t normed[kPacks][kBlock / 2];
  float local_amax = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int blk = threadIdx.x + p * kNumThreads;
    BF16Vec w_lo, w_hi;
    if constexpr (kHasWeight) {
      w_lo.load(weight, blk * 2);
      w_hi.load(weight, blk * 2 + 1);
    }
#pragma unroll
    for (int q = 0; q < kBlock / 2; ++q) {
      fp32x2_t v = vals[p][q];
      v.x *= rstd;
      v.y *= rstd;
      if constexpr (kHasWeight) {
        const auto wf = cast<fp32x2_t>(q < 4 ? w_lo[q] : w_hi[q - 4]);
        v.x *= wf.x;
        v.y *= wf.y;
      }
      normed[p][q] = cast<bf16x2_t>(v);
      if constexpr (kQuant && !kStaticScale) {
        const auto r = cast<fp32x2_t>(normed[p][q]);
        local_amax = fmaxf(local_amax, fmaxf(fabsf(r.x), fabsf(r.y)));
      }
    }
    if constexpr (kNormedOut) {
      const auto out = static_cast<bf16_t*>(params.normed_out) + row * kHidden;
      BF16Vec lo, hi;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        lo[q] = normed[p][q];
        hi[q] = normed[p][q + 4];
      }
      lo.store(out, blk * 2);
      hi.store(out, blk * 2 + 1);
    }
  }

  if constexpr (kQuant) {
    float encode_scale = kFloat32Max;
    if constexpr (kStaticScale) {
      encode_scale = params.global_scale[0];
    } else {
      float* max_smem = kNumWarps == 1 ? smem : smem + kNumWarps;
      float row_amax = warp::reduce_max(local_amax);
      if constexpr (kNumWarps > 1) {
        const int lane = threadIdx.x % kWarpThreads;
        if (lane == 0) max_smem[threadIdx.x / kWarpThreads] = row_amax;
        __syncthreads();
        row_amax = warp::reduce_max(lane < kNumWarps ? max_smem[lane] : 0.0f);
      }

      float per_token_scale = 0.0f;
      if (row_amax != 0.0f) {
        per_token_scale = mul_f32(row_amax, params.global_scale_inv);
        encode_scale = rcp_approx_ftz(per_token_scale);
      }
      if (threadIdx.x == 0) {
        static_cast<float*>(params.per_token_scale)[row] = per_token_scale;
      }
    }

    const float fp4_max_rcp = rcp_approx_ftz(6.0f);
    const auto fp4 = static_cast<uint64_t*>(params.fp4_out) + row * (kHidden / kBlock);
    const auto sf = static_cast<uint8_t*>(params.sf_out) + row * (kHidden / kBlock);
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int blk = threadIdx.x + p * kNumThreads;
      float f[kBlock];
      float block_max = 0.0f;
#pragma unroll
      for (int q = 0; q < kBlock / 2; ++q) {
        const auto r = cast<fp32x2_t>(normed[p][q]);
        f[2 * q] = r.x;
        f[2 * q + 1] = r.y;
        block_max = fmaxf(block_max, fmaxf(fabsf(r.x), fabsf(r.y)));
      }
      const float scale_f = mul_f32(encode_scale, mul_f32(block_max, fp4_max_rcp));
      const uint32_t fp8 = cvt_f32_to_e4m3(scale_f) & 0xFFu;
      const float out_scale =
          kStaticScale ? output_scale_static(fp8, block_max, encode_scale) : output_scale(fp8, encode_scale);
#pragma unroll
      for (int i = 0; i < kBlock; ++i)
        f[i] = mul_f32(f[i], out_scale);
      const uint64_t packed = (static_cast<uint64_t>(cvt_e2m1x8(f + 8)) << 32) | static_cast<uint64_t>(cvt_e2m1x8(f));
      fp4[blk] = packed;
      sf[blk] = static_cast<uint8_t>(fp8);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHidden, int kNumThreads, bool kUsePDL, bool kQuant, bool kNormedOut, bool kStaticScale>
struct MuseSparkNormFp4Quant {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::Optional<tvm::ffi::TensorView> weight,
      const tvm::ffi::Optional<tvm::ffi::TensorView> normed_out,
      const tvm::ffi::Optional<tvm::ffi::TensorView> fp4_out,
      const tvm::ffi::Optional<tvm::ffi::TensorView> sf_out,
      const tvm::ffi::Optional<tvm::ffi::TensorView> per_token_scale,
      const tvm::ffi::Optional<tvm::ffi::TensorView> global_scale,
      double eps,
      double global_scale_inv) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hidden_size"};
    auto K2 = SymbolicSize{"hidden_size_div_2"};
    auto K16 = SymbolicSize{"hidden_size_div_16"};
    auto device_ = SymbolicDevice{};
    K.set_value(kHidden);
    K2.set_value(kHidden / 2);
    K16.set_value(kHidden / 16);
    device_.set_options<kDLCUDA>();

    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device_).verify(x);
    if (weight.has_value()) {
      TensorMatcher({K}).with_dtype<bf16_t>().with_device(device_).verify(weight.value());
    }
    RuntimeCheck(normed_out.has_value() == kNormedOut, "normed_out presence must match the instantiation");
    if constexpr (kNormedOut) {
      TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device_).verify(normed_out.value());
    }
    RuntimeCheck(fp4_out.has_value() == kQuant, "fp4_out presence must match the instantiation");
    if constexpr (kQuant) {
      TensorMatcher({M, K2}).with_dtype<uint8_t>().with_device(device_).verify(fp4_out.value());
      TensorMatcher({M, K16}).with_dtype<uint8_t>().with_device(device_).verify(sf_out.value());
    }
    RuntimeCheck(per_token_scale.has_value() == (kQuant && !kStaticScale), "per_token_scale is per-token mode only");
    if (per_token_scale.has_value()) {
      TensorMatcher({M}).with_dtype<fp32_t>().with_device(device_).verify(per_token_scale.value());
    }
    RuntimeCheck(global_scale.has_value() == kStaticScale, "global_scale is static mode only");
    if constexpr (kStaticScale) {
      TensorMatcher({1}).with_dtype<fp32_t>().with_device(device_).verify(global_scale.value());
    }

    const auto num_rows = static_cast<int>(M.unwrap());
    if (num_rows == 0) return;

    const auto params = NormFp4QuantParams{
        .x = x.data_ptr(),
        .weight = weight.has_value() ? weight.value().data_ptr() : nullptr,
        .normed_out = kNormedOut ? normed_out.value().data_ptr() : nullptr,
        .fp4_out = kQuant ? fp4_out.value().data_ptr() : nullptr,
        .sf_out = kQuant ? sf_out.value().data_ptr() : nullptr,
        .per_token_scale = per_token_scale.has_value() ? per_token_scale.value().data_ptr() : nullptr,
        .global_scale = kStaticScale ? static_cast<const float*>(global_scale.value().data_ptr()) : nullptr,
        .eps = static_cast<float>(eps),
        .global_scale_inv = static_cast<float>(global_scale_inv),
        .num_rows = num_rows,
    };

    const auto grid = static_cast<unsigned>(num_rows);
    const auto device = device_.unwrap();
    if (weight.has_value()) {
      LaunchKernel(grid, kNumThreads, device)
          .enable_pdl(kUsePDL)(
              muse_spark_norm_fp4_quant_kernel<kHidden, kNumThreads, kUsePDL, kQuant, kNormedOut, true, kStaticScale>,
              params);
    } else {
      LaunchKernel(grid, kNumThreads, device)
          .enable_pdl(kUsePDL)(
              muse_spark_norm_fp4_quant_kernel<kHidden, kNumThreads, kUsePDL, kQuant, kNormedOut, false, kStaticScale>,
              params);
    }
  }
};

}  // namespace sglang
