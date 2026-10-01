#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

using namespace device;

// Decode (M=1) NVFP4 W4A4 grouped-GEMV pair for the non-gated (ReLU^2)
// Nemotron latent MoE, replacing CUTLASS grouped GEMM + expandInputRows /
// doActivation / finalizeMoeRouting glue at single-token batches.
//
// Numerics: e2m1 nibbles decode to integer 2x-magnitudes (0,1,2,3,4,6,8,12
// with sign) via one PRMT per 4 elements; 16-element block dots run in
// int32 dp4a and are scaled by the e4m3 block-scale product in fp32, the
// exact association of the block-scaled MMA path. Activation quantization
// uses the same cvt.rn.satfinite.e2m1x2 / e4m3 hardware ops as flashinfer
// fp4_quantize, with the checkpoint global scales (1/s_in) and alpha =
// s_in * s_w dequant products.

__device__ __forceinline__ int dp4a_s32(uint32_t a, uint32_t b, int c) {
  return __dp4a(static_cast<int>(a), static_cast<int>(b), c);
}

__device__ __forceinline__ float dq_e4m3(uint8_t v) {
  __nv_fp8_e4m3 f;
  f.__x = v;
  return static_cast<float>(f);
}

// Decode 4 low (even) e2m1 nibbles of a packed u32 into signed 2x-magnitude
// int8 lanes: mag2 table {0,1,2,3,4,6,8,12}, sign from nibble bit 3.
__device__ __forceinline__ uint32_t nib4_lo_to_mag2(uint32_t w4) {
  const uint32_t idx = w4 & 0x07070707u;
  const uint32_t u = idx | (idx >> 4);              // b0:i0|i1<<4, b2:i2|i3<<4
  const uint32_t sel = __byte_perm(u, 0u, 0x5420u); // nibbles i0 i1 i2 i3
  const uint32_t mag = __byte_perm(0x03020100u, 0x0C080604u, sel);
  const uint32_t m = ((w4 >> 3) & 0x01010101u) * 0xFFu;
  return __vsub4(mag ^ m, m);
}

__device__ __forceinline__ uint32_t nib4_hi_to_mag2(uint32_t w4) {
  return nib4_lo_to_mag2(w4 >> 4);
}

// (hi, lo) -> one packed e2m1x2 byte, exactly cvt.rn.satfinite.
__device__ __forceinline__ uint32_t cvt_e2m1x2(float hi, float lo) {
  uint32_t d;
  asm volatile(
      "{\n\t.reg .b8 t;\n\tcvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n\tcvt.u32.u8 %0, t;\n\t}"
      : "=r"(d)
      : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ uint8_t cvt_e4m3(float v) {
  return __nv_fp8_e4m3(v).__x;
}

// ---------------- Kernel A: fc1 (up) + ReLU^2 + fp4 requant ----------------
// Block covers 16 consecutive output rows (= one z scale group). Emits z as
// signed mag2 planes + fp32 group scales, ready for the fc2 dp4a.
template <uint32_t N, uint32_t K, uint32_t kTopK, bool kUsePDL>
__global__ void __launch_bounds__(256) nvfp4_moe_gemv_fc1_kernel(
    int32_t* __restrict__ zev,          // [kTopK, N/8] even-elem mag2 planes
    int32_t* __restrict__ zod,          // [kTopK, N/8]
    float* __restrict__ zsf,            // [kTopK, N/16] fp32(e4m3 sf)
    const bf16_t* __restrict__ x,       // [K]
    const uint8_t* __restrict__ w,      // [E, N, K/2] packed fp4
    const uint8_t* __restrict__ wsf,    // [E, N, K/16] e4m3
    const float* __restrict__ alpha1,   // [E] = s_in1 * s_w1
    const float* __restrict__ ga1,      // scalar = 1 / s_in1
    const float* __restrict__ ga2,      // scalar = 1 / s_in2
    const int32_t* __restrict__ topk_ids) {
  constexpr uint32_t kWords = K / 8;    // source packed u32 count
  constexpr uint32_t kGroups = K / 16;
  __shared__ int32_t sxe[kWords];
  __shared__ int32_t sxo[kWords];
  __shared__ float sxf[kGroups];
  __shared__ float szv[16];

  PDLTriggerSecondary<kUsePDL>();
  const uint32_t tid = threadIdx.x;
  PDLWaitPrimary<kUsePDL>();
  // Quantize x once per block (redundant across blocks, zero extra kernels):
  // thread g handles scale-group g (16 bf16 -> 1 sf + 2 plane words).
  if (tid < kGroups) {
    const float g1 = *ga1;
    float vals[16];
    float amax = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      vals[i] = static_cast<float>(x[tid * 16 + i]);
      amax = fmaxf(amax, fabsf(vals[i]));
    }
    const uint8_t sf8 = cvt_e4m3(amax * g1 / 6.0f);
    const float sf = dq_e4m3(sf8);
    const float qs = sf > 0.0f ? g1 / sf : 0.0f;
    uint32_t words[2] = {0u, 0u};
#pragma unroll
    for (uint32_t b = 0; b < 8; ++b) {
      const uint32_t byte = cvt_e2m1x2(vals[2 * b + 1] * qs, vals[2 * b] * qs);
      words[b / 4] |= byte << (8 * (b % 4));
    }
    sxe[2 * tid] = nib4_lo_to_mag2(words[0]);
    sxo[2 * tid] = nib4_hi_to_mag2(words[0]);
    sxe[2 * tid + 1] = nib4_lo_to_mag2(words[1]);
    sxo[2 * tid + 1] = nib4_hi_to_mag2(words[1]);
    sxf[tid] = sf;
  }
  __syncthreads();

  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t slot = blockIdx.y;
  const uint32_t e = static_cast<uint32_t>(topk_ids[slot]);

  // Two rows per warp (n, n+8): two independent weight streams in flight per
  // warp keep DRAM saturated even when the shared-expert stream co-resides
  // on the SMs.
  const uint32_t n0 = blockIdx.x * 16 + warp;
  const uint32_t n1r = n0 + 8;
  const uint8_t* wr0 = w + (static_cast<size_t>(e) * N + n0) * (K / 2);
  const uint8_t* sr0 = wsf + (static_cast<size_t>(e) * N + n0) * kGroups;
  const uint8_t* wr1 = w + (static_cast<size_t>(e) * N + n1r) * (K / 2);
  const uint8_t* sr1 = wsf + (static_cast<size_t>(e) * N + n1r) * kGroups;

  float facc0 = 0.0f;
  float facc1 = 0.0f;
  for (uint32_t c = lane * 16; c < K / 2; c += 32 * 16) {
    const uint4 wa = __ldcs(reinterpret_cast<const uint4*>(wr0 + c));
    const uint4 wb = __ldcs(reinterpret_cast<const uint4*>(wr1 + c));
    const uint32_t wi = c / 4;
    const uint32_t g0 = c / 8;
    const int xe0 = sxe[wi], xo0 = sxo[wi];
    const int xe1 = sxe[wi + 1], xo1 = sxo[wi + 1];
    const int xe2 = sxe[wi + 2], xo2 = sxo[wi + 2];
    const int xe3 = sxe[wi + 3], xo3 = sxo[wi + 3];
    int a0 = dp4a_s32(nib4_lo_to_mag2(wa.x), static_cast<uint32_t>(xe0), 0);
    a0 = dp4a_s32(nib4_hi_to_mag2(wa.x), static_cast<uint32_t>(xo0), a0);
    a0 = dp4a_s32(nib4_lo_to_mag2(wa.y), static_cast<uint32_t>(xe1), a0);
    a0 = dp4a_s32(nib4_hi_to_mag2(wa.y), static_cast<uint32_t>(xo1), a0);
    int a1 = dp4a_s32(nib4_lo_to_mag2(wa.z), static_cast<uint32_t>(xe2), 0);
    a1 = dp4a_s32(nib4_hi_to_mag2(wa.z), static_cast<uint32_t>(xo2), a1);
    a1 = dp4a_s32(nib4_lo_to_mag2(wa.w), static_cast<uint32_t>(xe3), a1);
    a1 = dp4a_s32(nib4_hi_to_mag2(wa.w), static_cast<uint32_t>(xo3), a1);
    int b0 = dp4a_s32(nib4_lo_to_mag2(wb.x), static_cast<uint32_t>(xe0), 0);
    b0 = dp4a_s32(nib4_hi_to_mag2(wb.x), static_cast<uint32_t>(xo0), b0);
    b0 = dp4a_s32(nib4_lo_to_mag2(wb.y), static_cast<uint32_t>(xe1), b0);
    b0 = dp4a_s32(nib4_hi_to_mag2(wb.y), static_cast<uint32_t>(xo1), b0);
    int b1 = dp4a_s32(nib4_lo_to_mag2(wb.z), static_cast<uint32_t>(xe2), 0);
    b1 = dp4a_s32(nib4_hi_to_mag2(wb.z), static_cast<uint32_t>(xo2), b1);
    b1 = dp4a_s32(nib4_lo_to_mag2(wb.w), static_cast<uint32_t>(xe3), b1);
    b1 = dp4a_s32(nib4_hi_to_mag2(wb.w), static_cast<uint32_t>(xo3), b1);
    facc0 = fmaf(static_cast<float>(a0), dq_e4m3(sr0[g0]) * sxf[g0], facc0);
    facc0 = fmaf(static_cast<float>(a1), dq_e4m3(sr0[g0 + 1]) * sxf[g0 + 1], facc0);
    facc1 = fmaf(static_cast<float>(b0), dq_e4m3(sr1[g0]) * sxf[g0], facc1);
    facc1 = fmaf(static_cast<float>(b1), dq_e4m3(sr1[g0 + 1]) * sxf[g0 + 1], facc1);
  }
#pragma unroll
  for (uint32_t off = 16; off > 0; off >>= 1) {
    facc0 += __shfl_down_sync(0xffffffff, facc0, off);
    facc1 += __shfl_down_sync(0xffffffff, facc1, off);
  }
  if (lane == 0) {
    const float a1v = alpha1[e];
    const float y0 = facc0 * 0.25f * a1v;
    const float y1 = facc1 * 0.25f * a1v;
    const float r0v = y0 > 0.0f ? y0 : 0.0f;
    const float r1v = y1 > 0.0f ? y1 : 0.0f;
    szv[warp] = r0v * r0v;
    szv[warp + 8] = r1v * r1v;
  }
  __syncthreads();

  // One z scale group per block: quantize the 16 outputs, emit planes + sf.
  if (tid == 0) {
    const float g2 = *ga2;
    float amax = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      amax = fmaxf(amax, fabsf(szv[i]));
    }
    const uint8_t sf8 = cvt_e4m3(amax * g2 / 6.0f);
    const float sf = dq_e4m3(sf8);
    const float qs = sf > 0.0f ? g2 / sf : 0.0f;
    uint32_t words[2] = {0u, 0u};
#pragma unroll
    for (uint32_t b = 0; b < 8; ++b) {
      const uint32_t byte = cvt_e2m1x2(szv[2 * b + 1] * qs, szv[2 * b] * qs);
      words[b / 4] |= byte << (8 * (b % 4));
    }
    const size_t base = static_cast<size_t>(slot) * (N / 8) + blockIdx.x * 2;
    zev[base] = nib4_lo_to_mag2(words[0]);
    zod[base] = nib4_hi_to_mag2(words[0]);
    zev[base + 1] = nib4_lo_to_mag2(words[1]);
    zod[base + 1] = nib4_hi_to_mag2(words[1]);
    zsf[static_cast<size_t>(slot) * (N / 16) + blockIdx.x] = sf;
  }
}

// ------------- Kernel B: fc2 + routed finalize -> bf16 out -------------
template <uint32_t N, uint32_t K, uint32_t kTopK, uint32_t kNumWarps, bool kUsePDL>
__global__ void __launch_bounds__(kNumWarps * 32, 2) nvfp4_moe_gemv_fc2_kernel(
    bf16_t* __restrict__ out,             // [N]
    const int32_t* __restrict__ zev,      // [kTopK, K/8]
    const int32_t* __restrict__ zod,      // [kTopK, K/8]
    const float* __restrict__ zsf,        // [kTopK, K/16]
    const uint8_t* __restrict__ w,        // [E, N, K/2]
    const uint8_t* __restrict__ wsf,      // [E, N, K/16]
    const float* __restrict__ alpha2,     // [E] = s_in2 * s_w2
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights) {
  constexpr uint32_t kBytes = K / 2;
  constexpr uint32_t kGroups = K / 16;
  __shared__ float spartial[kNumWarps];

  PDLTriggerSecondary<kUsePDL>();
  const uint32_t tid = threadIdx.x;
  PDLWaitPrimary<kUsePDL>();
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t n = blockIdx.x;

  float wsum = 0.0f;
  for (uint32_t slot = warp; slot < kTopK; slot += kNumWarps) {
    // With kNumWarps == kTopK this loop body runs exactly once per warp.
    const uint32_t e = static_cast<uint32_t>(topk_ids[slot]);
    const uint8_t* wr = w + (static_cast<size_t>(e) * N + n) * kBytes;
    const uint8_t* sr = wsf + (static_cast<size_t>(e) * N + n) * kGroups;
    const int32_t* ze = zev + static_cast<size_t>(slot) * (K / 8);
    const int32_t* zo = zod + static_cast<size_t>(slot) * (K / 8);
    const float* zs = zsf + static_cast<size_t>(slot) * kGroups;
    float facc = 0.0f;
    // Software pipeline: keep the next weight chunk in flight while the
    // dp4a/PRMT decode of the current one runs, so the DRAM stream does not
    // stall behind the ALU tail.
    uint32_t c = lane * 16;
    uint4 wv = c < kBytes ? __ldcs(reinterpret_cast<const uint4*>(wr + c))
                          : uint4{0u, 0u, 0u, 0u};
    while (c < kBytes) {
      const uint32_t cn = c + 32 * 16;
      uint4 wnext;
      if (cn < kBytes) {
        wnext = __ldcs(reinterpret_cast<const uint4*>(wr + cn));
      }
      const uint32_t wi = c / 4;
      const uint32_t g0 = c / 8;
      int ig0 = dp4a_s32(nib4_lo_to_mag2(wv.x), static_cast<uint32_t>(ze[wi]), 0);
      ig0 = dp4a_s32(nib4_hi_to_mag2(wv.x), static_cast<uint32_t>(zo[wi]), ig0);
      ig0 = dp4a_s32(nib4_lo_to_mag2(wv.y), static_cast<uint32_t>(ze[wi + 1]), ig0);
      ig0 = dp4a_s32(nib4_hi_to_mag2(wv.y), static_cast<uint32_t>(zo[wi + 1]), ig0);
      int ig1 = dp4a_s32(nib4_lo_to_mag2(wv.z), static_cast<uint32_t>(ze[wi + 2]), 0);
      ig1 = dp4a_s32(nib4_hi_to_mag2(wv.z), static_cast<uint32_t>(zo[wi + 2]), ig1);
      ig1 = dp4a_s32(nib4_lo_to_mag2(wv.w), static_cast<uint32_t>(ze[wi + 3]), ig1);
      ig1 = dp4a_s32(nib4_hi_to_mag2(wv.w), static_cast<uint32_t>(zo[wi + 3]), ig1);
      facc = fmaf(static_cast<float>(ig0), dq_e4m3(sr[g0]) * zs[g0], facc);
      facc = fmaf(static_cast<float>(ig1), dq_e4m3(sr[g0 + 1]) * zs[g0 + 1], facc);
      wv = wnext;
      c = cn;
    }
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
      facc += __shfl_down_sync(0xffffffff, facc, off);
    }
    if (lane == 0) {
      wsum = fmaf(facc, 0.25f * alpha2[e] * topk_weights[slot], wsum);
    }
  }
  if (lane == 0) {
    spartial[warp] = wsum;
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kNumWarps; ++i) {
      total += spartial[i];
    }
    out[n] = cast<bf16_t>(total);
  }
}

template <uint32_t N1, uint32_t K1, uint32_t N2, uint32_t K2, uint32_t kTopK, bool kUsePDL>
struct Sm120Nvfp4MoeGemvKernel {
  static_assert(N1 % 16 == 0 && K1 % 512 == 0, "fc1: N1 % 16, K1 % 512");
  static_assert(K2 == N1 && K2 % 16 == 0, "fc2 K must equal fc1 N");

  static void run_fc1(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView wsf,
      const tvm::ffi::TensorView alpha1,
      const tvm::ffi::TensorView ga1,
      const tvm::ffi::TensorView ga2,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView zev,
      const tvm::ffi::TensorView zod,
      const tvm::ffi::TensorView zsf) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({K1}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({kTopK * (N1 / 8)}).with_dtype<int32_t>().with_device(device).verify(zev);
    TensorMatcher({kTopK * (N1 / 8)}).with_dtype<int32_t>().with_device(device).verify(zod);
    TensorMatcher({kTopK * (N1 / 16)}).with_dtype<fp32_t>().with_device(device).verify(zsf);
    LaunchKernel({N1 / 16, kTopK, 1}, 256, device.unwrap())
        .enable_pdl(kUsePDL)(
        nvfp4_moe_gemv_fc1_kernel<N1, K1, kTopK, kUsePDL>,
        static_cast<int32_t*>(zev.data_ptr()),
        static_cast<int32_t*>(zod.data_ptr()),
        static_cast<float*>(zsf.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const uint8_t*>(w.data_ptr()),
        static_cast<const uint8_t*>(wsf.data_ptr()),
        static_cast<const float*>(alpha1.data_ptr()),
        static_cast<const float*>(ga1.data_ptr()),
        static_cast<const float*>(ga2.data_ptr()),
        static_cast<const int32_t*>(topk_ids.data_ptr()));
  }

  static void run_fc2(
      const tvm::ffi::TensorView zev,
      const tvm::ffi::TensorView zod,
      const tvm::ffi::TensorView zsf,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView wsf,
      const tvm::ffi::TensorView alpha2,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({kTopK * (K2 / 8)}).with_dtype<int32_t>().with_device(device).verify(zev);
    TensorMatcher({kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({kTopK}).with_dtype<fp32_t>().with_device(device).verify(topk_weights);
    TensorMatcher({N2}).with_dtype<bf16_t>().with_device(device).verify(out);
    constexpr uint32_t kNumWarps = kTopK;
    LaunchKernel({N2, 1, 1}, kNumWarps * 32, device.unwrap())
        .enable_pdl(kUsePDL)(
        nvfp4_moe_gemv_fc2_kernel<N2, K2, kTopK, kNumWarps, kUsePDL>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const int32_t*>(zev.data_ptr()),
        static_cast<const int32_t*>(zod.data_ptr()),
        static_cast<const float*>(zsf.data_ptr()),
        static_cast<const uint8_t*>(w.data_ptr()),
        static_cast<const uint8_t*>(wsf.data_ptr()),
        static_cast<const float*>(alpha2.data_ptr()),
        static_cast<const int32_t*>(topk_ids.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()));
  }
};

}  // namespace sglang
