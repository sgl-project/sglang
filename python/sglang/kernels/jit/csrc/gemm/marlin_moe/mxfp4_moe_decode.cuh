// Decode-shape MoE GEMM over Marlin-packed MXFP4 weights (E8M0 scales, group 32),
// bf16 activations, fp32 accumulation, moe_block_size 8.
//
// At decode shapes the Marlin MoE kernel is bound by instruction issue, not HBM: for a 3584 x 384 gate_up
// (per TP rank) with 8 tokens it executes about 4.7 instructions per weight (FP4 bit placement, a multiply by
// the FP4 exponent bias 2^126, the scale multiply, the shared-memory round trip and the stripe bookkeeping)
// and keeps HBM under 25 % busy. This kernel reads the same Marlin-packed B, scale and bias layouts and
// produces the same bf16 operands as Marlin's dequant, but
//   * loads B, scales and activations straight from global memory into registers (no smem staging),
//     software-pipelined over kStages k16 blocks, with compile-time strides (kN, kK);
//   * folds 2^126 into the per-group scale once (an integer add to the E8M0 exponent: exact for bytes 1..128,
//     i.e. scales in [2^-126, 2]; a warp-uniform fallback multiplies like Marlin for any other byte), so each
//     weight costs one bf16 multiply;
//   * places the FP4 bits with shifts split between the ALU and the FMA pipe (both are half rate and both
//     saturate here);
//   * gives every warp 64 output columns and a contiguous K range; the kWarpsK partial sums of a threadblock are
//     added in a fixed order in shared memory. No threadblock shares an output, so there are no locks or global
//     reductions and the result does not depend on the batch.
// The dequantized weights and the epilogue (bf16 rounding, bias add, top-k weight multiply in bf16) are
// bit-identical to Marlin; only the fp32 summation order over K differs.
//
// Optional padding skip: k_valid < kK stops the K loop early, and output tiles whose columns lie in the padded
// part of each half (column % (kN / 2) >= n_half_valid) skip the K loop. Both are exact only when the skipped
// weights are zero (Marlin padding); callers must establish that.

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/scalar_type.hpp>

#include "kernel.h"
#include "marlin_template.h"

namespace sglang {

namespace device::mxfp4_moe_decode {

using namespace device::marlin;
using Dtype = ScalarType<bf16_t>;
using FragA = Dtype::FragA;
using FragB = Dtype::FragB;
using FragC = Dtype::FragC;

constexpr int kRows = 8;    // moe_block_size
constexpr int kTileN = 64;  // output columns per warp

__device__ __forceinline__ int4 ldg_stream_v4(const int4* p) {
  int4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ldg_stream_v2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ldg_u32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.b32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}

// Multipliers passed as kernel arguments, so that ptxas keeps these shifts as IMAD on the FMA pipe instead of
// strength-reducing them to shifts on the ALU pipe.
struct DeqMul {
  uint32_t x4, x16, x64, x128, x256, x4096, fold;
};

// The eight FP4 values of one Marlin-packed int32 as the bf16 bit patterns of Marlin's
// dequant<bf16x2, kFE2M1f, true> applied to (q << 8) -> fb0 and q -> fb1: sign to bit 15/31, e1 e0 m to bits
// 8:6/24:22 (the value times 2^-126). kMode 0 runs Marlin's own sequence; kMode 2 the same bits with the left
// shifts as IMAD (checked equal on 1M random words).
template <int kMode>
__device__ __forceinline__ void dequant_fp4x8(uint32_t q, const DeqMul& k, FragB& fb0, FragB& fb1) {
  if constexpr (kMode == 0) {
    dequant<bf16x2_t, host::kFE2M1f.id(), true>(int(q << 8), reinterpret_cast<bf16x2_t*>(&fb0));
    dequant<bf16x2_t, host::kFE2M1f.id(), true>(int(q), reinterpret_cast<bf16x2_t*>(&fb1));
  } else {
    constexpr uint32_t kSign = 0x80008000u;
    const uint32_t d12 = (q & kSign) | ((q & 0x70007000u) >> 6);
    const uint32_t d8 = ((q * k.x16) & kSign) | ((q & 0x07000700u) >> 2);
    const uint32_t d4 = ((q * k.x256) & kSign) | ((q & 0x00700070u) * k.x4);
    const uint32_t d0 = ((q * k.x4096) & kSign) | ((q & 0x00070007u) * k.x64);
    fb0[0] = *reinterpret_cast<const bf16x2_t*>(&d0);
    fb0[1] = *reinterpret_cast<const bf16x2_t*>(&d4);
    fb1[0] = *reinterpret_cast<const bf16x2_t*>(&d8);
    fb1[1] = *reinterpret_cast<const bf16x2_t*>(&d12);
  }
}

// E8M0 byte `sel` (0..3) of w as a broadcast bf16x2. kFolded: 2^(e - 1) = 2^126 * 2^(e - 127) (valid for
// 1 <= e <= 128); otherwise 2^(e - 127), 0 for e = 0, as Marlin's dequant_fp8_scales.
template <int sel, bool kFolded>
__device__ __forceinline__ bf16x2_t scale_pair(uint32_t w, const DeqMul& k) {
  const uint32_t b = __byte_perm(w, 0u, 0x4040u | (sel << 8) | sel);  // byte sel to bits 7:0 and 23:16
  const uint32_t bits = kFolded ? b * k.x128 + k.fold : b * k.x128;
  return *reinterpret_cast<const bf16x2_t*>(&bits);
}

template <int kN, int kK, int kWarpsN, int kWarpsK, int kStages, bool kIsEP, bool kHasBias, int kDeqMode>
__global__ void __launch_bounds__(32 * kWarpsN * kWarpsK) mxfp4_moe_decode_kernel(
    const bf16_t* __restrict__ A,                  // [rows, kK]; row = sorted id / top_k
    const int4* __restrict__ B,                    // Marlin-packed [E, kK / 16, kN / 2] int4
    bf16_t* __restrict__ C,                        // [prob_m * top_k, kN]
    const bf16_t* __restrict__ bias,               // Marlin-permuted [E, kN]
    const uint8_t* __restrict__ scales,            // Marlin-permuted E8M0 [E, kK / 32, kN]
    const int32_t* __restrict__ sorted_token_ids,  // moe_align_block_size(block 8) outputs
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights,
    int top_k,
    bool mul_topk_weights,
    int prob_m,
    int k_valid,
    int n_half_valid,
    const DeqMul mul) {
  static_assert(kStages % 2 == 0, "a scale group spans two k16 blocks");
  constexpr int kBStride = kN / 2;  // int4 per k16 row (16 k x kN x 4 bit)
  const int blk = blockIdx.y;
  if (blk * kRows >= num_tokens_post_padded[0]) return;
  const int64_t expert = expert_ids[blk];
  if constexpr (kIsEP) {
    if (expert < 0) return;
  }

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int warp_n = warp % kWarpsN, warp_k = warp / kWarpsN;
  const int t64 = blockIdx.x * kWarpsN + warp_n;  // 64-column output tile
  const int g = lane / 4, c = lane % 4;

  // Token slot g of this moe block feeds the activation (MMA B operand) column g.
  const int prob_m_top_k = prob_m * top_k;
  const int sid = sorted_token_ids[blk * kRows + g];
  const bool row_valid = sid < prob_m_top_k;

  const int k16_per_warp = k_valid / 16 / kWarpsK;
  const int kb0 = warp_k * k16_per_warp;
  const bool tile_pad = (kTileN * t64) % (kN / 2) >= n_half_valid;

  FragC acc[4];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int i = 0; i < 4; i++)
      acc[j][i] = 0.f;

  if (!tile_pad) {
    const int4* b_ptr = B + (expert * (kK / 16) + kb0) * kBStride + 32 * t64 + lane;
    const uint8_t* s_ptr = scales + (expert * (kK / 32) + kb0 / 2) * kN + kTileN * t64 + 8 * g;
    const bf16_t* a_ptr = A + (int64_t)(row_valid ? sid / top_k : 0) * kK + 16 * kb0 + 2 * c;

    int4 bq[kStages];
    uint32_t a0[kStages], a1[kStages];
    uint2 sq[kStages / 2];

    // Stage s of the k16 blocks starting at b / sc / a (compile-time offsets from the three pointers).
    auto load = [&](int s, const int4* b, const uint8_t* sc, const bf16_t* a) {
      bq[s] = ldg_stream_v4(b + s * kBStride);
      a0[s] = row_valid ? ldg_u32(a + 16 * s) : 0u;
      a1[s] = row_valid ? ldg_u32(a + 16 * s + 8) : 0u;
      if (s % 2 == 0) sq[s / 2] = ldg_stream_v2(sc + (s / 2) * kN);
    };

#pragma unroll
    for (int s = 0; s < kStages; s++)
      load(s, b_ptr, s_ptr, a_ptr);

    // FP4 exponent bias of Marlin's dequant (see dequant<bf16x2, kFE2M1f, false>)
    constexpr uint32_t kBiasBits = ((1 << (8 - 1)) - (1 << (2 - 1)) + 127) << 23;
    const bf16x2_t bias_reg = __float2bfloat162_rn(__uint_as_float(kBiasBits));

    // kStages k16 blocks from the registers; with kLoad, refill each stage from the next kStages blocks.
    auto run_stages = [&](auto load_tag) {
      constexpr bool kLoad = decltype(load_tag)::value;
      const int4* b_next = b_ptr + kStages * kBStride;
      const uint8_t* s_next = s_ptr + (kStages / 2) * kN;
      const bf16_t* a_next = a_ptr + 16 * kStages;
#pragma unroll
      for (int gs = 0; gs < kStages / 2; gs++) {
        // Scales of this group, broadcast per output column as Marlin's scale(): the j-th 16-column block uses
        // bytes (0, 2), (1, 3), (4, 6), (5, 7) for columns 16j + g and 16j + g + 8.
        const uint32_t w0 = sq[gs].x, w1 = sq[gs].y;
        const bool slow = __any_sync(
            0xffffffffu,
            (__vcmpeq4(w0, 0u) | __vcmpeq4(w1, 0u) | __vcmpgtu4(w0, 0x80808080u) | __vcmpgtu4(w1, 0x80808080u)) != 0);
        auto group = [&](auto slow_tag) {
          constexpr bool kSlow = decltype(slow_tag)::value;
          bf16x2_t sx[4], sy[4];
          sx[0] = scale_pair<0, !kSlow>(w0, mul), sy[0] = scale_pair<2, !kSlow>(w0, mul);
          sx[1] = scale_pair<1, !kSlow>(w0, mul), sy[1] = scale_pair<3, !kSlow>(w0, mul);
          sx[2] = scale_pair<0, !kSlow>(w1, mul), sy[2] = scale_pair<2, !kSlow>(w1, mul);
          sx[3] = scale_pair<1, !kSlow>(w1, mul), sy[3] = scale_pair<3, !kSlow>(w1, mul);
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int st = 2 * gs + h;
            FragA fa;
            fa[0] = *reinterpret_cast<bf16x2_t*>(&a0[st]);
            fa[1] = *reinterpret_cast<bf16x2_t*>(&a1[st]);
            const uint32_t q4[4] = {(uint32_t)bq[st].x, (uint32_t)bq[st].y, (uint32_t)bq[st].z, (uint32_t)bq[st].w};
#pragma unroll
            for (int j = 0; j < 4; j++) {
              FragB fb0, fb1;
              dequant_fp4x8<kDeqMode>(q4[j], mul, fb0, fb1);
              if constexpr (kSlow) {
                fb0[0] = __hmul2(fb0[0], bias_reg);
                fb0[1] = __hmul2(fb0[1], bias_reg);
                fb1[0] = __hmul2(fb1[0], bias_reg);
                fb1[1] = __hmul2(fb1[1], bias_reg);
              }
              fb0[0] = __hmul2(fb0[0], sx[j]);
              fb0[1] = __hmul2(fb0[1], sx[j]);
              fb1[0] = __hmul2(fb1[0], sy[j]);
              fb1[1] = __hmul2(fb1[1], sy[j]);
              device::marlin_moe::mma_trans<bf16_t>(fa, fb0, fb1, acc[j]);
            }
            if constexpr (kLoad) load(st, b_next, s_next, a_next);
          }
        };
        if (slow) {
          group(std::true_type{});
        } else {
          group(std::false_type{});
        }
      }
      b_ptr = b_next, s_ptr = s_next, a_ptr = a_next;
    };

    for (int it = kStages; it < k16_per_warp; it += kStages)
      run_stages(std::true_type{});
    run_stages(std::false_type{});
  }

  // Fixed-order reduction of the kWarpsK partial sums (warp_k 0 adds 1, 2, ... in turn).
  if constexpr (kWarpsK > 1) {
    __shared__ float4 red[kWarpsK - 1][kWarpsN][4][32];
    if (warp_k > 0) {
#pragma unroll
      for (int j = 0; j < 4; j++)
        red[warp_k - 1][warp_n][j][lane] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
    }
    __syncthreads();
    if (warp_k > 0) return;
#pragma unroll
    for (int w = 0; w < kWarpsK - 1; w++)
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float4 v = red[w][warp_n][j][lane];
        acc[j][0] += v.x;
        acc[j][1] += v.y;
        acc[j][2] += v.z;
        acc[j][3] += v.w;
      }
  }

  // Epilogue, as Marlin's write_result for moe_block_size 8: res = bf16(acc); res += bias (bf16);
  // res *= bf16(topk weight) (bf16). acc[j] = (col 16j+g, slot 2c), (col 16j+g, slot 2c+1),
  // (col 16j+g+8, slot 2c), (col 16j+g+8, slot 2c+1).
  constexpr int kStageStride = kTileN + 8;
  __shared__ __align__(16) bf16_t stage[kWarpsN][kRows][kStageStride];
  const int64_t col0 = (int64_t)kTileN * t64;
  bf16_t bias_v[4][2];
  if constexpr (kHasBias) {
    // Marlin's permuted bias layout (see bias_sh_rd and write() in marlin_template.h)
    const bf16_t* bias_e = bias + expert * kN + col0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int base = 8 * (g / 2) + 32 * (j / 2) + 4 * (j % 2) + (g % 2);
      bias_v[j][0] = bias_e[base];
      bias_v[j][1] = bias_e[base + 2];
    }
  }
  bf16_t tw[2];
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const int slot_sid = __shfl_sync(0xffffffffu, sid, 4 * (2 * c + r));
    tw[r] = Dtype::float2num(topk_weights[slot_sid < prob_m_top_k ? slot_sid : 0]);
  }
#pragma unroll
  for (int j = 0; j < 4; j++) {
#pragma unroll
    for (int h = 0; h < 2; h++) {  // column 16j+g (+8)
#pragma unroll
      for (int r = 0; r < 2; r++) {  // token slot 2c (+1)
        bf16_t v = Dtype::float2num(acc[j][2 * h + r]);
        if constexpr (kHasBias) v = __hadd(v, bias_v[j][h]);
        if (mul_topk_weights) v = __hmul(v, tw[r]);
        stage[warp_n][2 * c + r][16 * j + g + 8 * h] = v;
      }
    }
  }
  __syncwarp();
#pragma unroll
  for (int p = 0; p < 2; p++) {
    const int row = 4 * p + lane / 8, q = lane % 8;
    const int row_sid = __shfl_sync(0xffffffffu, sid, 4 * row);
    if (row_sid < prob_m_top_k) {
      *reinterpret_cast<int4*>(C + (int64_t)row_sid * kN + col0 + 8 * q) =
          *reinterpret_cast<const int4*>(&stage[warp_n][row][8 * q]);
    }
  }
}

}  // namespace device::mxfp4_moe_decode

template <int kN, int kK, bool kIsEP, bool kHasBias, int kWarpsN, int kWarpsK, int kStages, int kDeqMode>
void mxfp4_moe_decode_gemm(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_bias,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView topk_weights,
    int64_t top_k,
    bool mul_topk_weights,
    int64_t size_m,
    int64_t max_moe_blocks,
    int64_t k_valid,
    int64_t n_half_valid) {
  using namespace host;
  static_assert(kN % (64 * kWarpsN) == 0, "kN must be a multiple of 64 * kWarpsN");
  RuntimeCheck(a.size(0) == size_m && a.size(1) == kK, "a must be [size_m, ", kK, "]");
  RuntimeCheck(c.size(0) == size_m * top_k && c.size(1) == kN, "c must be [size_m * top_k, ", kN, "]");
  RuntimeCheck(b_q_weight.size(1) * 16 == kK && b_q_weight.size(2) * 8 / 16 == kN, "b_q_weight shape");
  RuntimeCheck(b_q_weight.is_contiguous() && b_scales.is_contiguous() && c.is_contiguous(), "non-contiguous input");
  RuntimeCheck(a.stride(1) == 1 && a.stride(0) == kK, "a must be contiguous");
  RuntimeCheck(b_scales.dim() == 3 && b_scales.size(1) * 32 == kK && b_scales.size(2) == kN, "b_scales shape");
  if (kHasBias) RuntimeCheck(b_bias.size(1) == kN && b_bias.is_contiguous(), "b_bias must be [E, ", kN, "]");
  RuntimeCheck(
      k_valid > 0 && k_valid <= kK && k_valid % (16 * kWarpsK * kStages) == 0,
      "k_valid = ",
      k_valid,
      " not divisible by 16 * kWarpsK * kStages");
  RuntimeCheck(n_half_valid > 0 && n_half_valid <= kN / 2, "n_half_valid out of range");
  RuntimeCheck(max_moe_blocks > 0, "max_moe_blocks must be positive");
  if (size_m == 0) return;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({-1, -1}).with_dtype<bf16_t>().with_device(device).verify(a);
  cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());

  constexpr uint32_t kFold = (126u << 7) | (126u << 23);  // + 126 on both E8M0 exponents: 2^(e - 127) * 2^126
  dim3 grid(kN / (64 * kWarpsN), static_cast<unsigned>(max_moe_blocks));
  device::mxfp4_moe_decode::mxfp4_moe_decode_kernel<kN, kK, kWarpsN, kWarpsK, kStages, kIsEP, kHasBias, kDeqMode>
      <<<grid, 32 * kWarpsN * kWarpsK, 0, stream>>>(
          (const bf16_t*)a.data_ptr(),
          (const int4*)b_q_weight.data_ptr(),
          (bf16_t*)c.data_ptr(),
          kHasBias ? (const bf16_t*)b_bias.data_ptr() : nullptr,
          (const uint8_t*)b_scales.data_ptr(),
          (const int32_t*)sorted_token_ids.data_ptr(),
          (const int32_t*)expert_ids.data_ptr(),
          (const int32_t*)num_tokens_post_padded.data_ptr(),
          (const float*)topk_weights.data_ptr(),
          static_cast<int>(top_k),
          mul_topk_weights,
          static_cast<int>(size_m),
          static_cast<int>(k_valid),
          static_cast<int>(n_half_valid),
          device::mxfp4_moe_decode::DeqMul{4u, 16u, 64u, 128u, 256u, 4096u, kFold});
  RuntimeDeviceCheck(cudaGetLastError());
}

}  // namespace sglang
