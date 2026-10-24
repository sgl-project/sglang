// Decode/verify mHC post + next combine + RMSNorm (+ optional MXFP8 of the normed row).
// One CTA per row; each thread owns kVPT 8-element vectors of the 5120 hidden columns.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {
struct MhcPostCombineNormDecodeParams {
  const bf16_t *x, *residual, *weight;
  bf16_t *residual_out, *output;
  const float *post, *comb, *pre;
  uint8_t *q, *sf;
  int64_t rows, padded_rows;
  float eps;
};

template <int Threads, bool Quant>
__global__ __launch_bounds__(Threads) void mhc_post_combine_norm_decode_kernel(
    const __grid_constant__ MhcPostCombineNormDecodeParams p) {
  using namespace device;
  using V = AlignedVector<bf16x2_t, 4>;
  constexpr int kVectors = 640;
  constexpr int kVPT = kVectors / Threads;
  constexpr int kWarps = Threads / 32;
  static_assert(kVPT * Threads == kVectors && Threads % 32 == 0);
  constexpr int kGroups = 5120 / 32;
  constexpr int64_t kTileBytes = ((kGroups + 3) / 4) * 512;
  __shared__ float warp_sums[kWarps];
  __shared__ float inv_rms_shared;

  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int64_t row = blockIdx.x;

  float comb[16], post[4], pre[4];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    comb[i] = __ldg(p.comb + row * 16 + i);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    post[i] = __ldg(p.post + row * 4 + i);
    pre[i] = __ldg(p.pre + row * 4 + i);
  }

  float value[kVPT][8];
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    const int vid = tid + j * Threads;
    V x, residual[4];
    x.load(p.x + row * 5120, vid);
#pragma unroll
    for (int r = 0; r < 4; ++r)
      residual[r].load(p.residual + row * 20480 + r * 5120, vid);
    float combined[8] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      V updated;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const auto xx = cast<fp32x2_t>(x[k]);
        const auto r0 = cast<fp32x2_t>(residual[0][k]);
        const auto r1 = cast<fp32x2_t>(residual[1][k]);
        const auto r2 = cast<fp32x2_t>(residual[2][k]);
        const auto r3 = cast<fp32x2_t>(residual[3][k]);
        float a = __fmaf_rn(post[i], xx.x, __fmul_rn(comb[i], r0.x));
        float b = __fmaf_rn(post[i], xx.y, __fmul_rn(comb[i], r0.y));
        a = __fmaf_rn(comb[4 + i], r1.x, a);
        b = __fmaf_rn(comb[4 + i], r1.y, b);
        a = __fmaf_rn(comb[8 + i], r2.x, a);
        b = __fmaf_rn(comb[8 + i], r2.y, b);
        a = __fmaf_rn(comb[12 + i], r3.x, a);
        b = __fmaf_rn(comb[12 + i], r3.y, b);
        updated[k] = cast<bf16x2_t>(fp32x2_t{a, b});
        const auto rounded = cast<fp32x2_t>(updated[k]);
        combined[2 * k] = __fmaf_rn(pre[i], rounded.x, combined[2 * k]);
        combined[2 * k + 1] = __fmaf_rn(pre[i], rounded.y, combined[2 * k + 1]);
      }
      updated.store(p.residual_out + row * 20480 + i * 5120, vid);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const auto rounded = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{combined[2 * k], combined[2 * k + 1]}));
      value[j][2 * k] = rounded.x;
      value[j][2 * k + 1] = rounded.y;
      sum = __fmaf_rn(rounded.x, rounded.x, sum);
      sum = __fmaf_rn(rounded.y, rounded.y, sum);
    }
  }

#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
  if (lane == 0) warp_sums[tid / 32] = sum;
  __syncthreads();
  if (tid < 32) {
    float total = lane < kWarps ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, offset));
    if (lane == 0) {
      float mean, inverse;
      asm("div.full.f32 %0, %1, %2;" : "=f"(mean) : "f"(total), "f"(5120.f));
      asm("add.f32 %0, %1, %2;" : "=f"(mean) : "f"(mean), "f"(p.eps));
      asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(mean));
      inv_rms_shared = inverse;
    }
  }
  __syncthreads();
  const float inv_rms = inv_rms_shared;

#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    const int vid = tid + j * Threads;
    V w, out;
    w.load(p.weight, vid);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const auto weight = cast<fp32x2_t>(w[k]);
      out[k] = cast<bf16x2_t>(fp32x2_t{
          __fmul_rn(__fmul_rn(value[j][2 * k], inv_rms), weight.x),
          __fmul_rn(__fmul_rn(value[j][2 * k + 1], inv_rms), weight.y)});
    }
    out.store(p.output + row * 5120, vid);
    if constexpr (Quant) {
      float y[8];
      float amax = 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const auto v = cast<fp32x2_t>(out[k]);
        y[2 * k] = v.x;
        y[2 * k + 1] = v.y;
        amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
      }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
      const float normalized = __fmul_rn(amax, 1.0f / 448.0f);
      const uint32_t bits = __float_as_uint(normalized);
      const int exponent = (bits >> 23) & 255;
      const uint32_t mantissa = bits & 0x7FFFFF;
      const bool bump = (mantissa != 0) && !((exponent == 0) && (mantissa <= 0x400000));
      const int sf = normalized <= 0.f ? 0 : min(exponent + static_cast<int>(bump), 254);
      const float scale = sf == 0 ? 0.f : __uint_as_float(static_cast<uint32_t>(254 - sf) << 23);
      uint16_t packed[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float a = fminf(fmaxf(__fmul_rn(y[2 * k], scale), -448.f), 448.f);
        const float b = fminf(fmaxf(__fmul_rn(y[2 * k + 1], scale), -448.f), 448.f);
        packed[k] = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
      }
      *reinterpret_cast<uint2*>(p.q + row * 5120 + vid * 8) =
          make_uint2(packed[0] | (uint32_t(packed[1]) << 16), packed[2] | (uint32_t(packed[3]) << 16));
      if (vid % 4 == 0) {
        const int g = vid / 4;
        const int64_t off =
            (row / 128) * kTileBytes + (g / 4) * 512 + ((row % 32) * 4 + ((row / 32) % 4)) * 4 + (g % 4);
        p.sf[off] = static_cast<uint8_t>(sf);
      }
    }
  }

  if constexpr (Quant) {
    for (int64_t pad = p.rows + row; pad < p.padded_rows; pad += p.rows) {
      for (int g = tid; g < kGroups; g += Threads) {
        const int64_t off =
            (pad / 128) * kTileBytes + (g / 4) * 512 + ((pad % 32) * 4 + ((pad / 32) % 4)) * 4 + (g % 4);
        p.sf[off] = 0;
      }
    }
  }
}

template <int Threads>
struct MhcPostCombineNormDecode {
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView post,
      tvm::ffi::TensorView comb,
      tvm::ffi::TensorView pre,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView sf,
      float eps,
      bool quant) {
    using namespace host;
    auto m = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, 5120}).with_dtype<bf16_t>().with_device(dev).verify(x).verify(output);
    TensorMatcher({m, 4, 5120}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(residual_out);
    TensorMatcher({m, 4}).with_dtype<float>().with_device(dev).verify(post).verify(pre);
    TensorMatcher({m, 4, 4}).with_dtype<float>().with_device(dev).verify(comb);
    TensorMatcher({5120}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    const int64_t rows = m.unwrap();
    const int64_t padded_rows = (rows + 127) / 128 * 128;
    if (quant) {
      TensorMatcher({m, 5120}).with_dtype<fp8_e4m3_t>().with_device(dev).verify(q);
      TensorMatcher({padded_rows * 160}).with_dtype<uint8_t>().with_device(dev).verify(sf);
    }
    for (const auto* ptr :
         {x.data_ptr(), residual.data_ptr(), weight.data_ptr(), residual_out.data_ptr(), output.data_ptr()}) {
      CHECK_HOST(reinterpret_cast<uintptr_t>(ptr) % 16 == 0) << "BF16 pointers must be 16-byte aligned";
    }
    if (rows == 0) return;
    const MhcPostCombineNormDecodeParams params{
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()),
        static_cast<bf16_t*>(residual_out.data_ptr()),
        static_cast<bf16_t*>(output.data_ptr()),
        static_cast<const float*>(post.data_ptr()),
        static_cast<const float*>(comb.data_ptr()),
        static_cast<const float*>(pre.data_ptr()),
        quant ? static_cast<uint8_t*>(q.data_ptr()) : nullptr,
        quant ? static_cast<uint8_t*>(sf.data_ptr()) : nullptr,
        rows,
        padded_rows,
        eps};
    if (quant)
      LaunchKernel(dim3(rows), Threads, dev.unwrap())
          .launch(mhc_post_combine_norm_decode_kernel<Threads, true>, params);
    else
      LaunchKernel(dim3(rows), Threads, dev.unwrap())
          .launch(mhc_post_combine_norm_decode_kernel<Threads, false>, params);
  }
};
}  // namespace sglang
