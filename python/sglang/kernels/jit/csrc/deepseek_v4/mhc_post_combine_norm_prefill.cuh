// Prefill mHC post/combine/RMSNorm with the original BF16 intermediates.
// Stage the collapsed row in shared memory to bound register use while preserving
// the original Triton prefill normalization reduction and PTX arithmetic.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {
struct MhcPostCombineNormPrefillParams {
  const bf16_t *x, *residual, *weight;
  bf16_t *residual_out, *output;
  const float *post, *comb, *pre;
  float eps;
};

template <int Threads>
__global__ __launch_bounds__(Threads) void mhc_post_combine_norm_prefill_kernel(
    const __grid_constant__ MhcPostCombineNormPrefillParams p) {
  using namespace device;
  using V = AlignedVector<bf16x2_t, 4>;
  __shared__ __align__(16) bf16_t collapsed[5120];
  __shared__ float warp_sums[4];
  __shared__ float inv_rms;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int64_t row = blockIdx.x;
  const float coeff = lane < 16   ? p.comb[row * 16 + lane]
                      : lane < 20 ? p.post[row * 4 + lane - 16]
                      : lane < 24 ? p.pre[row * 4 + lane - 20]
                                  : 0.f;

  // Vectorize loads but keep only one 8-element tile live at a time.
#pragma unroll 1
  for (int vid = tid; vid < 640; vid += Threads) {
    V x, residual[4];
    x.load(p.x + row * 5120, vid);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      residual[j].load(p.residual + row * 20480 + j * 5120, vid);
    float combined[8] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float post = __shfl_sync(0xffffffff, coeff, 16 + i);
      const float pre = __shfl_sync(0xffffffff, coeff, 20 + i);
      const float c0 = __shfl_sync(0xffffffff, coeff, i);
      const float c1 = __shfl_sync(0xffffffff, coeff, 4 + i);
      const float c2 = __shfl_sync(0xffffffff, coeff, 8 + i);
      const float c3 = __shfl_sync(0xffffffff, coeff, 12 + i);
      V updated;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const auto xx = cast<fp32x2_t>(x[k]);
        const auto r0 = cast<fp32x2_t>(residual[0][k]);
        const auto r1 = cast<fp32x2_t>(residual[1][k]);
        const auto r2 = cast<fp32x2_t>(residual[2][k]);
        const auto r3 = cast<fp32x2_t>(residual[3][k]);
        float a = __fmaf_rn(post, xx.x, __fmul_rn(c0, r0.x));
        float b = __fmaf_rn(post, xx.y, __fmul_rn(c0, r0.y));
        a = __fmaf_rn(c1, r1.x, a);
        b = __fmaf_rn(c1, r1.y, b);
        a = __fmaf_rn(c2, r2.x, a);
        b = __fmaf_rn(c2, r2.y, b);
        a = __fmaf_rn(c3, r3.x, a);
        b = __fmaf_rn(c3, r3.y, b);
        updated[k] = cast<bf16x2_t>(fp32x2_t{a, b});
        const auto rounded = cast<fp32x2_t>(updated[k]);
        combined[2 * k] = __fmaf_rn(pre, rounded.x, combined[2 * k]);
        combined[2 * k + 1] = __fmaf_rn(pre, rounded.y, combined[2 * k + 1]);
      }
      updated.store(p.residual_out + row * 20480 + i * 5120, vid);
    }
    V rounded;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      rounded[k] = cast<bf16x2_t>(fp32x2_t{combined[2 * k], combined[2 * k + 1]});
    rounded.store(collapsed, vid);
  }
  __syncthreads();

  // Match the actual prefill Triton layout: 8 consecutive elements/thread,
  // 128 threads, then stride 1024. The PTX folds each thread's elements in
  // order, followed by XOR 16,8,4,2,1 and a four-warp XOR 2,1 reduction.
  if (tid < 128) {
    float sum = 0.f;
#pragma unroll
    for (int group = 0; group < 5; ++group) {
      V v;
      v.load(collapsed + group * 1024, tid);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const auto value = cast<fp32x2_t>(v[k]);
        if (group == 0 && k == 0)
          sum = __fadd_rn(__fmul_rn(value.y, value.y), __fmul_rn(value.x, value.x));
        else {
          sum = __fmaf_rn(value.x, value.x, sum);
          sum = __fadd_rn(__fmul_rn(value.y, value.y), sum);
        }
      }
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
    if (lane == 0) warp_sums[tid / 32] = sum;
  }
  __syncthreads();
  if (tid < 32) {
    float sum = lane < 4 ? warp_sums[lane] : 0.f;
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 2));
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 1));
    if (lane == 0) {
      float mean, inverse;
      asm("div.full.f32 %0, %1, %2;" : "=f"(mean) : "f"(sum), "f"(5120.f));
      // Match Triton's unqualified PTX add: ptxas may contract the
      // constant division's reciprocal multiply with this epsilon add.
      asm("add.f32 %0, %1, %2;" : "=f"(mean) : "f"(mean), "f"(p.eps));
      asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(mean));
      inv_rms = inverse;
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int vid = tid; vid < 640; vid += Threads) {
    V v, w, out;
    v.load(collapsed, vid);
    w.load(p.weight, vid);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const auto value = cast<fp32x2_t>(v[k]);
      const auto weight = cast<fp32x2_t>(w[k]);
      out[k] = cast<bf16x2_t>(
          fp32x2_t{__fmul_rn(__fmul_rn(value.x, inv_rms), weight.x), __fmul_rn(__fmul_rn(value.y, inv_rms), weight.y)});
    }
    out.store(p.output + row * 5120, vid);
  }
}

template <int Threads>
struct MhcPostCombineNormPrefill {
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView post,
      tvm::ffi::TensorView comb,
      tvm::ffi::TensorView pre,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto m = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, 5120}).with_dtype<bf16_t>().with_device(dev).verify(x).verify(output);
    TensorMatcher({m, 4, 5120}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(residual_out);
    TensorMatcher({m, 4}).with_dtype<float>().with_device(dev).verify(post).verify(pre);
    TensorMatcher({m, 4, 4}).with_dtype<float>().with_device(dev).verify(comb);
    TensorMatcher({5120}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    CHECK_HOST(m.unwrap() >= 4096 && m.unwrap() <= 65536) << "prefill rows must be in [4096, 65536]";
    for (const auto* ptr :
         {x.data_ptr(), residual.data_ptr(), weight.data_ptr(), residual_out.data_ptr(), output.data_ptr()}) {
      CHECK_HOST(reinterpret_cast<uintptr_t>(ptr) % 16 == 0) << "BF16 pointers must be 16-byte aligned";
    }
    CHECK_HOST(
        residual_out.data_ptr() != residual.data_ptr() && residual_out.data_ptr() != x.data_ptr() &&
        output.data_ptr() != residual.data_ptr() && output.data_ptr() != x.data_ptr() &&
        output.data_ptr() != residual_out.data_ptr())
        << "outputs must not alias inputs or each other";
    const MhcPostCombineNormPrefillParams params{
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()),
        static_cast<bf16_t*>(residual_out.data_ptr()),
        static_cast<bf16_t*>(output.data_ptr()),
        static_cast<const float*>(post.data_ptr()),
        static_cast<const float*>(comb.data_ptr()),
        static_cast<const float*>(pre.data_ptr()),
        eps};
    LaunchKernel(dim3(m.unwrap()), Threads, dev.unwrap()).launch(mhc_post_combine_norm_prefill_kernel<Threads>, params);
  }
};
}  // namespace sglang
