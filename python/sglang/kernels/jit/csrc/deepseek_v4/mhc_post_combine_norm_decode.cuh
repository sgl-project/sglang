// Decode/verify mHC post + next combine + RMSNorm (+ optional MXFP8 of the normed row).
// A row is split over a cluster of Split CTAs (blockIdx.x is the rank, blockIdx.y the row; Split 1 uses blockIdx.x);
// each thread owns kVPT 8-element vectors of the 5120 hidden columns.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>
#include <sgl_kernel/ptx/barrier.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

namespace device::ptx {

SGL_DEVICE uint32_t mapa_shared(uint32_t addr, uint32_t rank) {
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(result) : "r"(addr), "r"(rank));
  return result;
}

SGL_DEVICE void st_async_f32(uint32_t remote_addr, float x, uint32_t remote_bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(remote_addr),
               "f"(x),
               "r"(remote_bar)
               : "memory");
}

SGL_DEVICE void mbar_wait_parity_acquire_cta(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}\n" ::"r"(to_shared(bar)),
      "r"(parity)
      : "memory");
}

}  // namespace device::ptx

struct MhcPostCombineNormDecodeParams {
  const bf16_t *x, *residual, *weight;
  bf16_t *residual_out, *output;
  const float *post, *comb, *pre;
  uint8_t *q, *sf;
  int64_t rows;
  float eps;
  // MegaMoE input (fp8 rows + UE8M0 scales), written as mega_moe_pre_dispatch writes them.
  uint8_t *moe_x, *moe_sf, *moe_shared_sf;
  uint64_t moe_sf_stride_bytes, moe_shared_col_stride;
  uint32_t moe_shared_block_m;
};

template <int Threads, int Split, bool Quant, bool MoeQuant = false>
__global__ __launch_bounds__(Threads) void mhc_post_combine_norm_decode_kernel(
    const __grid_constant__ MhcPostCombineNormDecodeParams p) {
  using namespace device;
  using V = AlignedVector<bf16x2_t, 4>;
  constexpr int kVectors = 640 / Split;
  constexpr int kVPT = kVectors / Threads;
  constexpr int kWarps = Threads / 32;
  constexpr int kPool = Split * kWarps;
  constexpr bool kPrefetchWeight = Split > 1 || kVPT == 1;
  static_assert(kVPT * Threads * Split == 640 && Threads % 32 == 0 && kPool <= 32);
  constexpr int kGroups = 5120 / 32;
  constexpr int64_t kTileBytes = ((kGroups + 3) / 4) * 512;
  __shared__ float pool[kPool];
  __shared__ uint64_t bar;

  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int rank = Split > 1 ? blockIdx.x : 0;
  const int64_t row = Split > 1 ? blockIdx.y : blockIdx.x;

  if constexpr (Split > 1) {
    if (tid == 0) {
      ptx::mbar_init(&bar, 1);
      ptx::mbar_arrive_expect_tx(&bar, kPool * sizeof(float));
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  }
  const bf16_t* x_ptr = p.x;
  const bf16_t* residual_ptr = p.residual;
  const bf16_t* weight_ptr = p.weight;
  const float* comb_ptr = p.comb;
  const float* post_ptr = p.post;
  const float* pre_ptr = p.pre;
  __builtin_assume(__isGlobal(x_ptr));
  __builtin_assume(__isGlobal(residual_ptr));
  __builtin_assume(__isGlobal(weight_ptr));
  __builtin_assume(__isGlobal(comb_ptr));
  __builtin_assume(__isGlobal(post_ptr));
  __builtin_assume(__isGlobal(pre_ptr));

  PDLWaitPrimary<true>();
  float comb[16], post[4], pre[4];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    comb[i] = __ldg(comb_ptr + row * 16 + i);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    post[i] = __ldg(post_ptr + row * 4 + i);
    pre[i] = __ldg(pre_ptr + row * 4 + i);
  }

  V w[kVPT];
  if constexpr (kPrefetchWeight) {
#pragma unroll
    for (int j = 0; j < kVPT; ++j)
      w[j].load(weight_ptr, rank * kVectors + tid + j * Threads);
  }

  float value[kVPT][8];
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    const int vid = rank * kVectors + tid + j * Threads;
    V x, residual[4];
    x.load(x_ptr + row * 5120, vid);
#pragma unroll
    for (int r = 0; r < 4; ++r)
      residual[r].load(residual_ptr + row * 20480 + r * 5120, vid);
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
  if constexpr (Split > 1) {
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (lane < Split) {
      const uint32_t slot = ptx::to_shared(&pool[rank * kWarps + warp]);
      ptx::st_async_f32(ptx::mapa_shared(slot, lane), sum, ptx::mapa_shared(ptx::to_shared(&bar), lane));
    }
    ptx::mbar_wait_parity_acquire_cta(&bar, 0);
  } else {
    if (lane == 0) pool[warp] = sum;
    __syncthreads();
  }
  float total = lane < kPool ? pool[lane] : 0.f;
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    total = __fadd_rn(total, __shfl_xor_sync(0xffffffff, total, offset));
  float mean, inv_rms;
  asm("div.full.f32 %0, %1, %2;" : "=f"(mean) : "f"(total), "f"(5120.f));
  asm("add.f32 %0, %1, %2;" : "=f"(mean) : "f"(mean), "f"(p.eps));
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv_rms) : "f"(mean));

#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    const int vid = rank * kVectors + tid + j * Threads;
    if constexpr (!kPrefetchWeight) w[j].load(weight_ptr, vid);
    V out;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const auto weight = cast<fp32x2_t>(w[j][k]);
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
    if constexpr (MoeQuant) {
      using deepseek_v4::fp8::cast_to_ue8m0;
      using deepseek_v4::fp8::pack_fp8;
      float y[8];
      float local_max = 0.0f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const auto v = cast<fp32x2_t>(out[k]);
        y[2 * k] = v.x;
        y[2 * k + 1] = v.y;
        local_max = fmaxf(local_max, fmaxf(fabsf(v.x), fabsf(v.y)));
      }
      local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 1));
      local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 2));
      const float absmax = fmaxf(local_max, 1e-10f);
      const float raw_scale = absmax / math::FP8_E4M3_MAX;
      const uint32_t ue8m0_exp = cast_to_ue8m0(raw_scale);
      const float inv_scale = __uint_as_float((127u + 127u - ue8m0_exp) << 23);
      AlignedVector<fp8x2_e4m3_t, 4> qv;
#pragma unroll
      for (int k = 0; k < 4; ++k)
        qv[k] = pack_fp8(y[2 * k] * inv_scale, y[2 * k + 1] * inv_scale);
      qv.store(reinterpret_cast<fp8_e4m3_t*>(p.moe_x + row * 5120), vid);
      if (vid % 4 == 0) {
        const int g = vid / 4;
        p.moe_sf[static_cast<uint64_t>(row) * p.moe_sf_stride_bytes + g] = static_cast<uint8_t>(ue8m0_exp);
        const uint32_t bm = p.moe_shared_block_m;
        const uint32_t t = static_cast<uint32_t>(row);
        const uint32_t m = t % bm;
        const uint64_t shared_row =
            t / bm * ((bm + 127u) / 128u * 128u) + m / 128u * 128u + m % 32u * 4u + m % 128u / 32u;
        const uint64_t off = shared_row + static_cast<uint64_t>(g / 4) * p.moe_shared_col_stride;
        p.moe_shared_sf[off * sizeof(int32_t) + g % 4] = static_cast<uint8_t>(ue8m0_exp);
      }
    }
  }

  if constexpr (Quant) {
    // Word i of the last 128-row tile holds tile row (i % 4) * 32 + (i % 128) / 4.
    const int first_pad = static_cast<int>(p.rows % 128);
    const int stride = static_cast<int>(p.rows) * Split * Threads;
    auto* tile = reinterpret_cast<uint32_t*>(p.sf + (p.rows / 128) * kTileBytes);
    for (int i = (static_cast<int>(row) * Split + rank) * Threads + tid; first_pad && i < kTileBytes / 4; i += stride) {
      if ((i % 4) * 32 + (i % 128) / 4 >= first_pad) tile[i] = 0;
    }
  }
}

template <int Threads, int Split>
struct MhcPostCombineNormDecode {
  static int64_t max_active_ctas(bool quant) {
    int ctas = 0;
    if (quant)
      CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &ctas, mhc_post_combine_norm_decode_kernel<Threads, Split, true>, Threads, 0));
    else
      CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &ctas, mhc_post_combine_norm_decode_kernel<Threads, Split, false>, Threads, 0));
    return ctas;
  }

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
      bool quant,
      tvm::ffi::Optional<tvm::ffi::TensorView> moe_x,
      tvm::ffi::Optional<tvm::ffi::TensorView> moe_sf,
      tvm::ffi::Optional<tvm::ffi::TensorView> moe_shared_sf,
      int64_t moe_shared_block_m) {
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
    CHECK_HOST(Split == 1 || rows <= 65535) << "rows go in grid.y";
    MhcPostCombineNormDecodeParams params{
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
        eps,
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        0};
    LaunchKernel launch(Split > 1 ? dim3(Split, rows) : dim3(rows), Threads, dev.unwrap());
    if (Split > 1) launch.enable_cluster(dim3(Split, 1, 1));
    launch.enable_pdl(true);
    if (moe_x.has_value()) {
      RuntimeCheck(!quant, "MegaMoE input and the swizzled MXFP8 output are exclusive");
      RuntimeCheck(moe_sf.has_value() && moe_shared_sf.has_value(), "MegaMoE input needs both scale buffers");
      const auto mx = moe_x.value();
      const auto ms = moe_sf.value();
      const auto ss = moe_shared_sf.value();
      RuntimeCheck(
          mx.size(0) >= rows && mx.size(1) == 5120 && mx.stride(1) == 1 && mx.stride(0) == 5120,
          "MegaMoE x must be [>= rows, 5120] row-major");
      RuntimeCheck(ms.size(0) >= rows && ms.size(1) * 4 >= 160 && ms.stride(1) == 1, "MegaMoE x_sf layout");
      RuntimeCheck((ms.stride(0) * 4) % 16 == 0, "MegaMoE x_sf row stride must be 16B-aligned");
      RuntimeCheck(moe_shared_block_m > 0 && ss.stride(0) == 1, "MegaMoE shared scales are column-major");
      params.moe_x = static_cast<uint8_t*>(mx.data_ptr());
      params.moe_sf = static_cast<uint8_t*>(ms.data_ptr());
      params.moe_sf_stride_bytes = static_cast<uint64_t>(ms.stride(0)) * 4;
      params.moe_shared_sf = static_cast<uint8_t*>(ss.data_ptr());
      params.moe_shared_col_stride = static_cast<uint64_t>(ss.stride(1));
      params.moe_shared_block_m = static_cast<uint32_t>(moe_shared_block_m);
      launch.launch(mhc_post_combine_norm_decode_kernel<Threads, Split, false, true>, params);
      return;
    }
    if (quant)
      launch.launch(mhc_post_combine_norm_decode_kernel<Threads, Split, true>, params);
    else
      launch.launch(mhc_post_combine_norm_decode_kernel<Threads, Split, false>, params);
  }
};
}  // namespace sglang
