// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include "routed_quant.cuh"
#include <cmath>
#include <cstdint>
#include <limits>

#if !defined(USE_ROCM) && defined(SGL_CUDA_ARCH) && (SGL_CUDA_ARCH < 1200 || SGL_CUDA_ARCH >= 1300)
#error "moe_nvfp4_quant_lowm_sm120.cuh requires SM12x (compute capability 12.x)"
#endif

namespace sglang::inkling2_moe_nvfp4_quant_sm120 {

namespace {

constexpr int32_t kNumExperts = 128;
constexpr int32_t kTopK = 8;

// One definition of the NVFP4 numerics with the general routed path.
using inkling2_routed_quant::capped_swiglu;
using inkling2_routed_quant::e2m1_pack;
using inkling2_routed_quant::e4m3_round;
using inkling2_routed_quant::kFltMax;
using inkling2_routed_quant::load_offsets;
using inkling2_routed_quant::part_max;
using inkling2_routed_quant::row_expert;
using inkling2_routed_quant::scale_base;

SGL_DEVICE uint4 ld_cs_u4(void const* ptr) {
  uint4 v;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
  return v;
}

template <int32_t kThreads>
SGL_DEVICE float block_max_reduce_with_enc(float v, float* red, float* s_enc_dec_out) {
#pragma unroll
  for (int32_t s = 16; s >= 1; s /= 2) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, s));
  }
  constexpr int32_t kWarps = kThreads / 32;
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = red[0];
#pragma unroll
    for (int32_t w = 1; w < kWarps; ++w) {
      m = fmaxf(m, red[w]);
    }
    float const enc = fminf(__fdiv_rn(1536.0f, m), kFltMax);
    float const dec = __fdiv_rn(1.0f, enc);
    float const out = __fmul_rn(m, static_cast<float>(1.0 / 1536.0));
    s_enc_dec_out[0] = m;
    s_enc_dec_out[1] = enc;
    s_enc_dec_out[2] = dec;
    s_enc_dec_out[3] = out;
  }
  __syncthreads();
  return s_enc_dec_out[0];
}

SGL_DEVICE void quantize_part_fast4(
    float const (&x)[4], float encoder, float decoder, float outer, float block_max, uint32_t& packed, uint8_t& scale) {
  int32_t const group_leader = static_cast<int32_t>(device::get_lane_id()) & ~3;
  uint8_t bits6 = 0, bits4 = 0;
  float scale6 = 0.0f, scale4 = 0.0f, inverse6 = 0.0f, inverse4 = 0.0f;
  if ((threadIdx.x & 3) == 0) {
    scale6 = e4m3_round(fminf(__fmul_rn(block_max, __fmul_rn(encoder, static_cast<float>(1.0 / 6.0))), 448.0f), bits6);
    scale4 = e4m3_round(fminf(__fmul_rn(block_max, __fmul_rn(encoder, 0.25f)), 448.0f), bits4);
    inverse6 = fminf(__fdiv_rn(1.0f, __fmul_rn(scale6, decoder)), kFltMax);
    inverse4 = fminf(__fdiv_rn(1.0f, __fmul_rn(scale4, decoder)), kFltMax);
  }
  scale6 = __shfl_sync(0xFFFFFFFFu, scale6, group_leader);
  scale4 = __shfl_sync(0xFFFFFFFFu, scale4, group_leader);
  inverse6 = __shfl_sync(0xFFFFFFFFu, inverse6, group_leader);
  inverse4 = __shfl_sync(0xFFFFFFFFu, inverse4, group_leader);
  uint32_t p6 = 0, p4 = 0;
  float e6[4], e4[4];
#pragma unroll
  for (int32_t i = 0; i < 2; ++i) {
    float r0, r1;
    p6 |= e2m1_pack(__fmul_rn(x[2 * i], inverse6), __fmul_rn(x[2 * i + 1], inverse6), r0, r1) << (8 * i);
    e6[2 * i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r0, scale6), outer), x[2 * i]));
    e6[2 * i + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r1, scale6), outer), x[2 * i + 1]));
    p4 |= e2m1_pack(__fmul_rn(x[2 * i], inverse4), __fmul_rn(x[2 * i + 1], inverse4), r0, r1) << (8 * i);
    e4[2 * i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r0, scale4), outer), x[2 * i]));
    e4[2 * i + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r1, scale4), outer), x[2 * i + 1]));
  }
  e6[0] = __fadd_rn(__fadd_rn(e6[0], e6[1]), __fadd_rn(e6[2], e6[3]));
  e4[0] = __fadd_rn(__fadd_rn(e4[0], e4[1]), __fadd_rn(e4[2], e4[3]));
  e6[0] = __fadd_rn(e6[0], __shfl_xor_sync(0xFFFFFFFFu, e6[0], 1));
  e4[0] = __fadd_rn(e4[0], __shfl_xor_sync(0xFFFFFFFFu, e4[0], 1));
  e6[0] = __fadd_rn(e6[0], __shfl_xor_sync(0xFFFFFFFFu, e6[0], 2));
  e4[0] = __fadd_rn(e4[0], __shfl_xor_sync(0xFFFFFFFFu, e4[0], 2));
  bool const choose4 = e4[0] < e6[0];
  packed = choose4 ? p4 : p6;
  scale = choose4 ? bits4 : bits6;
}

}  // namespace

template <int32_t K>
__global__ void __launch_bounds__((K / 16) * 4) nvfp4_quant_dedup_lowm_kernel(
    bf16_t const* __restrict__ x,
    int32_t const* __restrict__ ids,
    int32_t const* __restrict__ src2dst,
    int32_t const* __restrict__ offsets,
    uint8_t* __restrict__ values,
    uint8_t* __restrict__ x_scales,
    float* __restrict__ x_amax,
    int32_t T) {
  constexpr int32_t kScaleK = K / 16;
  constexpr int32_t kThreads = kScaleK * 4;
  static_assert(K % 128 == 0 && kThreads % 32 == 0 && kThreads <= 1024);

  __shared__ int32_t s_offsets[kNumExperts + 1];
  __shared__ float red[2][kThreads / 32];
  __shared__ float s_enc[2][4];
  __shared__ int32_t s_dst[2][kTopK];
  __shared__ int64_t s_scale_row_off[2][kTopK];

  device::PDLWaitPrimary<true>();
  load_offsets(offsets, kNumExperts, s_offsets);
  int32_t const max_rows = T * kTopK;
  int32_t const total_routed = (s_offsets[kNumExperts] < max_rows) ? s_offsets[kNumExperts] : max_rows;

  int32_t const block = static_cast<int32_t>(threadIdx.x >> 2);
  int32_t const part = static_cast<int32_t>(threadIdx.x & 3);
  int32_t const scale_atom_off = (block >> 2) * 512;
  int32_t const val_block_off = block * 8;
  bool const write_scale4 = ((threadIdx.x & 15) == 0);
  int32_t parity = 0;

  for (int32_t t = static_cast<int32_t>(blockIdx.x); t < T; t += static_cast<int32_t>(gridDim.x), parity ^= 1) {
    if (threadIdx.x < static_cast<uint32_t>(kTopK)) {
      int32_t const dst = src2dst[static_cast<int64_t>(t) * kTopK + threadIdx.x];
      int32_t const e = ids[static_cast<int64_t>(t) * kTopK + threadIdx.x];
      bool const valid = (static_cast<uint32_t>(e) < static_cast<uint32_t>(kNumExperts)) &&
                         (static_cast<uint32_t>(dst) < static_cast<uint32_t>(total_routed));
      if (valid) {
        int32_t const local_r = dst - s_offsets[e];
        if (local_r >= 0 && dst < s_offsets[e + 1]) {
          s_dst[parity][threadIdx.x] = dst;
          s_scale_row_off[parity][threadIdx.x] = scale_base(s_offsets, e, kScaleK) +
                                                 static_cast<int64_t>(local_r >> 7) * (kScaleK / 4) * 512 +
                                                 (local_r & 31) * 16 + ((local_r >> 5) & 3) * 4;
        } else {
          s_dst[parity][threadIdx.x] = -1;
          s_scale_row_off[parity][threadIdx.x] = 0;
        }
      } else {
        s_dst[parity][threadIdx.x] = -1;
        s_scale_row_off[parity][threadIdx.x] = 0;
      }
    }

    float v[4];
    bf16_t const* src = x + static_cast<int64_t>(t) * K + block * 16 + part * 4;
    uint2 const raw = *reinterpret_cast<uint2 const*>(src);
    bf16_t const* h = reinterpret_cast<bf16_t const*>(&raw);
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
      v[i] = __bfloat162float(h[i]);

    float const block_max = part_max<4>(v);
    float const amax = block_max_reduce_with_enc<kThreads>(block_max, red[parity], s_enc[parity]);
    uint32_t packed;
    uint8_t scale;
    quantize_part_fast4(v, s_enc[parity][1], s_enc[parity][2], s_enc[parity][3], block_max, packed, scale);

    uint32_t const p0 = packed & 0xFFFFu;
    uint32_t const p1 = __shfl_down_sync(0xFFFFFFFFu, p0, 1);
    uint32_t const lo32 = p0 | (p1 << 16);
    uint32_t const hi32 = __shfl_down_sync(0xFFFFFFFFu, lo32, 2);
    uint2 const packed64 = make_uint2(lo32, hi32);

    uint32_t const s0 = static_cast<uint32_t>(scale);
    uint32_t const s1 = __shfl_down_sync(0xFFFFFFFFu, s0, 4);
    uint32_t const s2 = __shfl_down_sync(0xFFFFFFFFu, s0, 8);
    uint32_t const s3 = __shfl_down_sync(0xFFFFFFFFu, s0, 12);
    uint32_t const scale4 = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);

    if (threadIdx.x < static_cast<uint32_t>(kTopK)) {
      int32_t const dst = s_dst[parity][threadIdx.x];
      if (dst >= 0) {
        x_amax[dst] = amax;
      }
    }
    if (part == 0) {
#pragma unroll
      for (int32_t k = 0; k < kTopK; ++k) {
        int32_t const dst = s_dst[parity][k];
        if (dst >= 0) {
          *reinterpret_cast<uint2*>(values + static_cast<int64_t>(dst) * (K / 2) + val_block_off) = packed64;
          if (write_scale4) {
            *reinterpret_cast<uint32_t*>(x_scales + s_scale_row_off[parity][k] + scale_atom_off) = scale4;
          }
        }
      }
    }
  }
  device::PDLTriggerSecondary<true>();
}

template <int32_t N>
__global__ void __launch_bounds__((N / 16) * 4) nvfp4_swiglu_quant_bf16_lowm_kernel(
    bf16_t const* __restrict__ w13_bf16,
    float const* __restrict__ gammas,
    int32_t const* __restrict__ offsets,
    uint8_t* __restrict__ values,
    uint8_t* __restrict__ x_scales,
    float* __restrict__ act_amax,
    float gate_cap,
    float up_cap) {
  constexpr int32_t kScaleK = N / 16;
  constexpr int32_t kThreads = kScaleK * 4;
  static_assert(N % 128 == 0 && kThreads % 32 == 0 && kThreads <= 1024);

  __shared__ int32_t s_offsets[kNumExperts + 1];
  __shared__ float red[2][kThreads / 32];
  __shared__ float s_enc[2][4];
  __shared__ int64_t s_scale_row_off[2];

  device::PDLWaitPrimary<true>();
  load_offsets(offsets, kNumExperts, s_offsets);

  int32_t const block = static_cast<int32_t>(threadIdx.x >> 2);
  int32_t const part = static_cast<int32_t>(threadIdx.x & 3);
  int32_t const scale_atom_off = (block >> 2) * 512;
  int32_t const val_block_off = block * 8;
  bool const write_scale4 = ((threadIdx.x & 15) == 0);
  int32_t const total_rows = s_offsets[kNumExperts];
  int32_t parity = 0;

  for (int32_t row = static_cast<int32_t>(blockIdx.x); row < total_rows;
       row += static_cast<int32_t>(gridDim.x), parity ^= 1) {
    if (threadIdx.x == 0) {
      int32_t const expert = row_expert(s_offsets, kNumExperts, row);
      int32_t const local_r = row - s_offsets[expert];
      s_scale_row_off[parity] = scale_base(s_offsets, expert, kScaleK) +
                                static_cast<int64_t>(local_r >> 7) * (kScaleK / 4) * 512 + (local_r & 31) * 16 +
                                ((local_r >> 5) & 3) * 4;
    }

    uint4 const raw = ld_cs_u4(w13_bf16 + static_cast<int64_t>(row) * (2 * N) + block * 32 + part * 8);
    bf16_t const* h = reinterpret_cast<bf16_t const*>(&raw);
    float const gamma = gammas[row];

    float v[4];
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      float const gate = __bfloat162float(h[2 * i]);
      float const up = __bfloat162float(h[2 * i + 1]);
      v[i] = __bfloat162float(__float2bfloat16_rn(capped_swiglu(gate, up, gamma, gate_cap, up_cap)));
    }

    float const block_max = part_max<4>(v);
    float const amax = block_max_reduce_with_enc<kThreads>(block_max, red[parity], s_enc[parity]);
    uint32_t packed;
    uint8_t scale_bits;
    quantize_part_fast4(v, s_enc[parity][1], s_enc[parity][2], s_enc[parity][3], block_max, packed, scale_bits);

    uint32_t const p0 = packed & 0xFFFFu;
    uint32_t const p1 = __shfl_down_sync(0xFFFFFFFFu, p0, 1);
    uint32_t const lo32 = p0 | (p1 << 16);
    uint32_t const hi32 = __shfl_down_sync(0xFFFFFFFFu, lo32, 2);
    uint2 const packed64 = make_uint2(lo32, hi32);

    uint32_t const s0 = static_cast<uint32_t>(scale_bits);
    uint32_t const s1 = __shfl_down_sync(0xFFFFFFFFu, s0, 4);
    uint32_t const s2 = __shfl_down_sync(0xFFFFFFFFu, s0, 8);
    uint32_t const s3 = __shfl_down_sync(0xFFFFFFFFu, s0, 12);
    uint32_t const scale4 = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);

    if (threadIdx.x == 0) act_amax[row] = amax;
    if (part == 0) {
      *reinterpret_cast<uint2*>(values + static_cast<int64_t>(row) * (N / 2) + val_block_off) = packed64;
      if (write_scale4) {
        *reinterpret_cast<uint32_t*>(x_scales + s_scale_row_off[parity] + scale_atom_off) = scale4;
      }
    }
  }
  device::PDLTriggerSecondary<true>();
}

/**
 * \brief Deduplicated per-token NVFP4 4over6 quantization scattered to top-8 expert slots.
 * \note Leaves 128-row per-expert scale tails unzeroed; only safe for predicated consumers
 *       such as nvfp4_gemm_lowm_scaled.
 * \param x Input [T, K] BF16 token activations.
 * \param ids Flat [T * 8] int32 expert IDs.
 * \param src2dst Flat [T * 8] int32 token-slot to expert-order row indices.
 * \param offsets [129] int32 CSR expert row offsets.
 * \param values Output [T * 8, K / 2] uint8 packed E2M1 values.
 * \param x_scales Output [scale_rows, K / 16] uint8 swizzled E4M3 block scales.
 * \param x_amax Output [T * 8] FP32 per-row absmax.
 */
template <int32_t K>
inline void nvfp4_quant_dedup_lowm(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView src2dst,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView x_scales,
    tvm::ffi::TensorView x_amax) {
  auto T = host::SymbolicSize{"tokens"};
  auto rows = host::SymbolicSize{"routed_rows"};
  auto scale_rows = host::SymbolicSize{"scale_rows"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();
  host::TensorMatcher({T, K}).with_dtype<bf16_t>().with_device(device).ensure_alignment(8).verify(x);
  host::TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(ids).verify(src2dst);
  host::TensorMatcher({kNumExperts + 1}).with_dtype<int32_t>().with_device(device).verify(offsets);
  host::TensorMatcher({rows, K / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(8).verify(values);
  host::TensorMatcher({scale_rows, K / 16})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(4)
      .verify(x_scales);
  host::TensorMatcher({rows}).with_dtype<fp32_t>().with_device(device).verify(x_amax);

  int64_t const tokens = T.unwrap();
  int64_t const num_rows = rows.unwrap();
  CHECK_HOST(tokens >= 0 && tokens <= std::numeric_limits<int32_t>::max())
      << "nvfp4_quant_dedup_lowm: invalid token count " << tokens;
  CHECK_HOST(num_rows == tokens * kTopK) << "nvfp4_quant_dedup_lowm: rows must equal tokens * 8";
  CHECK_HOST(scale_rows.unwrap() >= host::div_ceil(num_rows + 127 * int64_t{kNumExperts}, 128) * 128)
      << "nvfp4_quant_dedup_lowm: insufficient scale rows " << scale_rows.unwrap();
  if (tokens == 0) return;

  constexpr int32_t kThreads = (K / 16) * 4;
  host::LaunchKernel(static_cast<uint32_t>(tokens), kThreads, device.unwrap())
      .enable_pdl(true)(
          nvfp4_quant_dedup_lowm_kernel<K>,
          static_cast<bf16_t const*>(x.data_ptr()),
          static_cast<int32_t const*>(ids.data_ptr()),
          static_cast<int32_t const*>(src2dst.data_ptr()),
          static_cast<int32_t const*>(offsets.data_ptr()),
          static_cast<uint8_t*>(values.data_ptr()),
          static_cast<uint8_t*>(x_scales.data_ptr()),
          static_cast<float*>(x_amax.data_ptr()),
          static_cast<int32_t>(tokens));
}

/**
 * \brief Capped SwiGLU on BF16 W13 outputs fused with NVFP4 4over6 quantization for W2.
 * \note Leaves 128-row per-expert scale tails unzeroed; only safe for predicated consumers
 *       such as nvfp4_gemm_lowm_scaled.
 * \param w13_bf16 Input [rows, 2 * N] BF16 interleaved (gate, up) activations.
 * \param gammas Input [rows] FP32 sorted routing weights.
 * \param offsets [129] int32 CSR expert row offsets.
 * \param values Output [rows, N / 2] uint8 packed E2M1 values.
 * \param x_scales Output [scale_rows, N / 16] uint8 swizzled E4M3 block scales.
 * \param act_amax Output [rows] FP32 per-row absmax.
 * \param gate_cap Gate softcap (<= 0 disables).
 * \param up_cap Up softcap (<= 0 disables).
 * \param sm_count Device SM count for grid cap.
 */
template <int32_t N>
inline void nvfp4_swiglu_quant_bf16_lowm(
    tvm::ffi::TensorView w13_bf16,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView x_scales,
    tvm::ffi::TensorView act_amax,
    double gate_cap,
    double up_cap,
    int64_t sm_count) {
  auto rows = host::SymbolicSize{"routed_rows"};
  auto scale_rows = host::SymbolicSize{"scale_rows"};
  auto device = host::SymbolicDevice{};
  device.set_options<kDLCUDA>();
  host::TensorMatcher({rows, 2 * N})  //
      .with_dtype<bf16_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w13_bf16);
  host::TensorMatcher({rows}).with_dtype<fp32_t>().with_device(device).verify(gammas).verify(act_amax);
  host::TensorMatcher({kNumExperts + 1}).with_dtype<int32_t>().with_device(device).verify(offsets);
  host::TensorMatcher({rows, N / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(8).verify(values);
  host::TensorMatcher({scale_rows, N / 16})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(4)
      .verify(x_scales);

  int64_t const num_rows = rows.unwrap();
  CHECK_HOST(num_rows >= 0 && num_rows <= std::numeric_limits<int32_t>::max())
      << "nvfp4_swiglu_quant_bf16_lowm: invalid row count " << num_rows;
  CHECK_HOST(sm_count > 0) << "nvfp4_swiglu_quant_bf16_lowm: sm_count must be positive, got " << sm_count;
  CHECK_HOST(scale_rows.unwrap() >= host::div_ceil(num_rows + 127 * int64_t{kNumExperts}, 128) * 128)
      << "nvfp4_swiglu_quant_bf16_lowm: insufficient scale rows " << scale_rows.unwrap();
  if (num_rows == 0) return;

  constexpr int32_t kThreads = (N / 16) * 4;
  int64_t const max_grid = sm_count * 4;
  int32_t const grid = static_cast<int32_t>(num_rows < max_grid ? num_rows : max_grid);
  host::LaunchKernel(static_cast<uint32_t>(grid), kThreads, device.unwrap())
      .enable_pdl(true)(
          nvfp4_swiglu_quant_bf16_lowm_kernel<N>,
          static_cast<bf16_t const*>(w13_bf16.data_ptr()),
          static_cast<float const*>(gammas.data_ptr()),
          static_cast<int32_t const*>(offsets.data_ptr()),
          static_cast<uint8_t*>(values.data_ptr()),
          static_cast<uint8_t*>(x_scales.data_ptr()),
          static_cast<float*>(act_amax.data_ptr()),
          static_cast<float>(gate_cap),
          static_cast<float>(up_cap));
}

}  // namespace sglang::inkling2_moe_nvfp4_quant_sm120
