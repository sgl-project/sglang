#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

namespace sglang {

constexpr int kWarpThreads = 32;
constexpr int kElemPerThread = 8;
constexpr float kFp8E4M3Max = 448.0f;
constexpr int kBlocksYZTarget = 2048;

struct alignas(16) SiluMulFp8MaskedParams {
  const __nv_bfloat16* __restrict__ input;
  __nv_fp8_e4m3* __restrict__ output;
  float* __restrict__ output_scale;
  const int32_t* __restrict__ masked_m;
  int64_t stride_input_e;
  int64_t stride_input_t;
  int64_t stride_output_e;
  int64_t stride_output_t;
  int64_t stride_scale_e;
  int64_t stride_scale_t;
  int32_t N;  // output half width H
  float eps;
  float swiglu_limit;
  float gemm1_alpha;
  float gemm1_clamp_limit;
};

__device__ __forceinline__ uint16_t cvt_fp32x2_to_e4m3x2(float lo, float hi) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  uint16_t packed;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n" : "=h"(packed) : "f"(lo), "f"(hi));
  return packed;
#else
  lo = fmaxf(fminf(lo, kFp8E4M3Max), -kFp8E4M3Max);
  hi = fmaxf(fminf(hi, kFp8E4M3Max), -kFp8E4M3Max);
  const __nv_fp8_e4m3 b_lo = static_cast<__nv_fp8_e4m3>(lo);
  const __nv_fp8_e4m3 b_hi = static_cast<__nv_fp8_e4m3>(hi);
  uint16_t packed;
  reinterpret_cast<uint8_t*>(&packed)[0] = *reinterpret_cast<const uint8_t*>(&b_lo);
  reinterpret_cast<uint8_t*>(&packed)[1] = *reinterpret_cast<const uint8_t*>(&b_hi);
  return packed;
#endif
}

// DeepSeek V4 swiglu_limit path: clamp in bf16 (to match upstream sglang /
// DeepGEMM), then silu(gate) * up computed in bf16 like the non-masked twin.
template <bool kApplySwigluLimit>
__device__ __forceinline__ __nv_bfloat162 silu_and_mul(__nv_bfloat162 gate, __nv_bfloat162 up, float swiglu_limit) {
  if constexpr (kApplySwigluLimit) {
    const __nv_bfloat16 lim = __float2bfloat16_rn(swiglu_limit);
    const __nv_bfloat16 nlim = __float2bfloat16_rn(-swiglu_limit);
    const __nv_bfloat162 lim2 = __halves2bfloat162(lim, lim);
    const __nv_bfloat162 nlim2 = __halves2bfloat162(nlim, nlim);
    gate = __hmin2(gate, lim2);
    up = __hmin2(__hmax2(up, nlim2), lim2);
  }
  const float g0 = __bfloat162float(__low2bfloat16(gate));
  const float g1 = __bfloat162float(__high2bfloat16(gate));
  const float silu0 = g0 * __ppu_sgmdf(g0);
  const float silu1 = g1 * __ppu_sgmdf(g1);
  const __nv_bfloat162 silu = __floats2bfloat162_rn(silu0, silu1);
  return __hmul2(up, silu);
}

// oai-swiglu (MiniMax-M3 / gpt-oss), computed in fp32 to match the masked
// triton kernel (_silu_and_mul_post_quant_kernel GEMM1_ALPHA branch) that
// serves the same activation:
//   gate = min(gate, L); up = clamp(up, -L, L);
//   out  = gate * sigmoid(gate * alpha) * (up + 1)
__device__ __forceinline__ float2 oai_swiglu(__nv_bfloat162 gate, __nv_bfloat162 up, float alpha, float clamp_limit) {
  const float g0 = __bfloat162float(__low2bfloat16(gate));
  const float g1 = __bfloat162float(__high2bfloat16(gate));
  const float u0 = __bfloat162float(__low2bfloat16(up));
  const float u1 = __bfloat162float(__high2bfloat16(up));
  const float gc0 = fminf(g0, clamp_limit);
  const float gc1 = fminf(g1, clamp_limit);
  const float uc0 = fmaxf(fminf(u0, clamp_limit), -clamp_limit);
  const float uc1 = fmaxf(fminf(u1, clamp_limit), -clamp_limit);
  const float p0 = gc0 * __ppu_sgmdf(gc0 * alpha) * (uc0 + 1.0f);
  const float p1 = gc1 * __ppu_sgmdf(gc1 * alpha) * (uc1 + 1.0f);
  return make_float2(p0, p1);
}

template <int kBlockThreads>
__device__ __forceinline__ float block_reduce_max(float val, float* smem_warp_max) {
  static_assert(kBlockThreads % kWarpThreads == 0, "");
  constexpr int kWarpsPerBlock = kBlockThreads / kWarpThreads;

#pragma unroll
  for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
  }
  if constexpr (kWarpsPerBlock == 1) {
    return val;
  }
  const int lane_id = threadIdx.x & (kWarpThreads - 1);
  const int warp_id = threadIdx.x / kWarpThreads;
  if (lane_id == 0) smem_warp_max[warp_id] = val;
  __syncthreads();

  if (warp_id == 0) {
    val = (lane_id < kWarpsPerBlock) ? smem_warp_max[lane_id] : 0.0f;
#pragma unroll
    for (int offset = kWarpsPerBlock / 2; offset > 0; offset >>= 1) {
      val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
    }
    if (lane_id == 0) smem_warp_max[0] = val;
  }
  __syncthreads();
  return smem_warp_max[0];
}

// Masked (expert-grouped) variant of silu_and_mul_post_per_token_quant_fp8:
// input [E, T_padded, 2H] bf16 -> output [E, T_padded, H] e4m3 with a single
// per-token float32 scale. Grid: (E, blocks_per_expert); each block iterates
// over its expert's tokens (masked_m valid range) in a grid-stride loop.
//
// Activation paths (mutually exclusive):
//   - plain silu(gate) * up (bf16, matches the triton fallback)
//   - kApplySwigluLimit: DeepSeek V4 clamped swiglu (bf16 clamp)
//   - kApplyGemm1Alpha: oai-swiglu gate * sigmoid(alpha*gate) * (up+1) in
//     fp32; the smem product cache then stores fp32 (2x bytes of the bf16
//     path) so the quantized values stay bit-identical to the triton kernel.
template <int kBlockThreads, bool kApplySwigluLimit, bool kApplyGemm1Alpha, bool kScaleUe8m0, bool kCacheInSmem>
__global__ __launch_bounds__(kBlockThreads, 2) void silu_and_mul_masked_post_per_token_quant_fp8_kernel(
    const SiluMulFp8MaskedParams __grid_constant__ params) {
  constexpr int kPairsPerThread = kElemPerThread / 2;  // 4
  static_assert(!(kApplySwigluLimit && kApplyGemm1Alpha), "swiglu_limit and gemm1_alpha are mutually exclusive");

  const int expert_id = blockIdx.x;
  const int token_block_id = blockIdx.y;
  const int tid = threadIdx.x;

  const int n_tokens = __ldg(params.masked_m + expert_id);
  if (n_tokens == 0) return;

  const __nv_bfloat16* in_e = params.input + expert_id * params.stride_input_e;
  __nv_fp8_e4m3* out_e = params.output + expert_id * params.stride_output_e;
  float* scl_e = params.output_scale + expert_id * params.stride_scale_e;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  [[maybe_unused]] __nv_bfloat16* smem_prod_bf16 = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  [[maybe_unused]] float* smem_prod_f32 = reinterpret_cast<float*>(smem_raw);
  constexpr size_t kProdElemBytes = kApplyGemm1Alpha ? sizeof(float) : sizeof(__nv_bfloat16);
  float* smem_warp_max =
      reinterpret_cast<float*>(smem_raw + (kCacheInSmem ? (kProdElemBytes * params.N + 15u) / 16u * 16u : 0u));

  const int n_vec = params.N / kElemPerThread;
  const int token_stride = gridDim.y;

  for (int t = token_block_id; t < n_tokens; t += token_stride) {
    const __nv_bfloat16* in_row = in_e + t * params.stride_input_t;
    __nv_fp8_e4m3* out_row = out_e + t * params.stride_output_t;

    // Pass 1: activation + per-token absmax (products cached in smem when possible).
    float local_absmax = 0.0f;
    for (int v = tid; v < n_vec; v += kBlockThreads) {
      const int elem_off = v * kElemPerThread;

      uint4 gate_pack = *reinterpret_cast<const uint4*>(in_row + elem_off);
      uint4 up_pack = *reinterpret_cast<const uint4*>(in_row + params.N + elem_off);
      auto* gate_pairs = reinterpret_cast<__nv_bfloat162*>(&gate_pack);
      auto* up_pairs = reinterpret_cast<__nv_bfloat162*>(&up_pack);

      if constexpr (kApplyGemm1Alpha) {
        alignas(16) float prods[kElemPerThread];
#pragma unroll
        for (int i = 0; i < kPairsPerThread; ++i) {
          const float2 p = oai_swiglu(gate_pairs[i], up_pairs[i], params.gemm1_alpha, params.gemm1_clamp_limit);
          prods[2 * i] = p.x;
          prods[2 * i + 1] = p.y;
          local_absmax = fmaxf(local_absmax, fmaxf(fabsf(p.x), fabsf(p.y)));
        }
        if constexpr (kCacheInSmem) {
          *reinterpret_cast<float4*>(smem_prod_f32 + elem_off) = *reinterpret_cast<const float4*>(prods);
          *reinterpret_cast<float4*>(smem_prod_f32 + elem_off + 4) = *reinterpret_cast<const float4*>(prods + 4);
        }
      } else {
        __nv_bfloat162 prod_pairs[kPairsPerThread];
        __nv_bfloat162 absmax_v2 = __float2bfloat162_rn(0.0f);

#pragma unroll
        for (int i = 0; i < kPairsPerThread; ++i) {
          const __nv_bfloat162 p = silu_and_mul<kApplySwigluLimit>(gate_pairs[i], up_pairs[i], params.swiglu_limit);
          prod_pairs[i] = p;
          absmax_v2 = __hmax2(absmax_v2, __habs2(p));
        }

        if constexpr (kCacheInSmem) {
          *reinterpret_cast<uint4*>(smem_prod_bf16 + elem_off) = *reinterpret_cast<const uint4*>(prod_pairs);
        }

        const __nv_bfloat16 m_bf16 = __hmax(__low2bfloat16(absmax_v2), __high2bfloat16(absmax_v2));
        local_absmax = fmaxf(local_absmax, __bfloat162float(m_bf16));
      }
    }
    // smem_prod writes must complete before the block reduction reads them
    // (block_reduce_max only syncs internally when kWarpsPerBlock > 1).
    __syncthreads();

    const float row_absmax = block_reduce_max<kBlockThreads>(local_absmax, smem_warp_max);

    float scale = fmaxf(row_absmax, params.eps) / kFp8E4M3Max;
    if constexpr (kScaleUe8m0) {
      // Round the scale up to the next power of two (positive input), matching
      // the triton fallback's tl.exp2(tl.ceil(tl.log2(x))) semantics.
      const uint32_t u = __float_as_uint(scale);
      scale = __uint_as_float((u + 0x007FFFFFu) & 0x7F800000u);
    }
    const float inv_scale = 1.0f / scale;
    if (tid == 0) scl_e[t * params.stride_scale_t] = scale;

    // Pass 2: rescale the (cached) products and store packed e4m3.
    for (int v = tid; v < n_vec; v += kBlockThreads) {
      const int elem_off = v * kElemPerThread;
      uint2 fp8_pack;
      uint16_t* fp8_arr = reinterpret_cast<uint16_t*>(&fp8_pack);

      if constexpr (kApplyGemm1Alpha) {
        alignas(16) float prods[kElemPerThread];

        if constexpr (kCacheInSmem) {
          *reinterpret_cast<float4*>(prods) = *reinterpret_cast<const float4*>(smem_prod_f32 + elem_off);
          *reinterpret_cast<float4*>(prods + 4) = *reinterpret_cast<const float4*>(smem_prod_f32 + elem_off + 4);
        } else {
          uint4 gate_pack = *reinterpret_cast<const uint4*>(in_row + elem_off);
          uint4 up_pack = *reinterpret_cast<const uint4*>(in_row + params.N + elem_off);
          auto* gate_pairs = reinterpret_cast<__nv_bfloat162*>(&gate_pack);
          auto* up_pairs = reinterpret_cast<__nv_bfloat162*>(&up_pack);
#pragma unroll
          for (int i = 0; i < kPairsPerThread; ++i) {
            const float2 p = oai_swiglu(gate_pairs[i], up_pairs[i], params.gemm1_alpha, params.gemm1_clamp_limit);
            prods[2 * i] = p.x;
            prods[2 * i + 1] = p.y;
          }
        }

#pragma unroll
        for (int i = 0; i < kPairsPerThread; ++i) {
          fp8_arr[i] = cvt_fp32x2_to_e4m3x2(prods[2 * i] * inv_scale, prods[2 * i + 1] * inv_scale);
        }
      } else {
        __nv_bfloat162 prod_pairs[kPairsPerThread];

        if constexpr (kCacheInSmem) {
          *reinterpret_cast<uint4*>(prod_pairs) = *reinterpret_cast<const uint4*>(smem_prod_bf16 + elem_off);
        } else {
          uint4 gate_pack = *reinterpret_cast<const uint4*>(in_row + elem_off);
          uint4 up_pack = *reinterpret_cast<const uint4*>(in_row + params.N + elem_off);
          auto* gate_pairs = reinterpret_cast<__nv_bfloat162*>(&gate_pack);
          auto* up_pairs = reinterpret_cast<__nv_bfloat162*>(&up_pack);
#pragma unroll
          for (int i = 0; i < kPairsPerThread; ++i) {
            prod_pairs[i] = silu_and_mul<kApplySwigluLimit>(gate_pairs[i], up_pairs[i], params.swiglu_limit);
          }
        }

#pragma unroll
        for (int i = 0; i < kPairsPerThread; ++i) {
          const float f_lo = __bfloat162float(__low2bfloat16(prod_pairs[i])) * inv_scale;
          const float f_hi = __bfloat162float(__high2bfloat16(prod_pairs[i])) * inv_scale;
          fp8_arr[i] = cvt_fp32x2_to_e4m3x2(f_lo, f_hi);
        }
      }
      *reinterpret_cast<uint2*>(out_row + elem_off) = fp8_pack;
    }
    // smem_prod is reused by the next token; make sure every thread finished
    // reading it before pass 1 of the next iteration overwrites the cache.
    __syncthreads();
  }
}

template <int kBlockThreads, bool kApplySwigluLimit, bool kApplyGemm1Alpha, bool kScaleUe8m0>
struct SiluMulFp8MaskedEP {
  static constexpr int kWPB = kBlockThreads / kWarpThreads;
  static_assert(kBlockThreads % kWarpThreads == 0, "");
  static_assert(!kApplySwigluLimit || !kApplyGemm1Alpha, "swiglu_limit and gemm1_alpha are mutually exclusive");

  // Shared-memory budget for the product cache, expressed in bytes so the
  // fp32 (oai-swiglu) path gets half the elements of the bf16 path. These
  // match the non-masked per-token fp8 kernel's 24000 / 49000 bf16 elements.
  static constexpr size_t kSmemMaxBytesDefault = 48000;   // default 48KB
  static constexpr size_t kSmemMaxBytesExtended = 98000;  // ~98KB

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView masked_m,
      double swiglu_limit,
      double gemm1_alpha,
      double gemm1_clamp_limit,
      double eps,
      int64_t max_masked_m) {
    using namespace host;

    const int E = static_cast<int>(input.size(0));
    const int T_padded = static_cast<int>(input.size(1));
    const int two_N = static_cast<int>(input.size(2));
    const int N = two_N / 2;

    RuntimeCheck(input.ndim() == 3, "input must be 3D (E, T, 2H)");
    RuntimeCheck(output.ndim() == 3, "output must be 3D (E, T, H)");
    RuntimeCheck(output_scale.ndim() == 3, "output_scale must be 3D (E, T, 1)");
    RuntimeCheck(
        output.size(0) == E && output.size(1) == T_padded && output.size(2) == N,
        "output must have shape (E, T_padded, N)");
    RuntimeCheck(input.dtype() == DLDataType{DLDataTypeCode::kDLBfloat, 16, 1}, "input dtype must be BFloat16");
    RuntimeCheck(E > 0, "E must be positive");
    RuntimeCheck(T_padded > 0, "T_padded must be positive");
    RuntimeCheck(two_N % 2 == 0, "input last dim must be even");
    RuntimeCheck(N % kElemPerThread == 0, "N must be multiple of ", kElemPerThread, ", got N=", N);
    RuntimeCheck(masked_m.ndim() == 1 && masked_m.size(0) == E, "masked_m must have shape (E,)");
    RuntimeCheck(input.stride(2) == 1, "input must be contiguous in the last dim");
    RuntimeCheck(input.stride(1) % 8 == 0 && input.stride(0) % 8 == 0, "input strides must be 16-byte aligned");
    RuntimeCheck(
        output.stride(2) == 1 && output.stride(1) % 8 == 0 && output.stride(0) % 8 == 0,
        "output strides must be 8-byte aligned");
    RuntimeCheck(output_scale.size(2) == 1, "output_scale last dim must be 1");

    int blocks_per_expert = (kBlocksYZTarget + E - 1) / E;
    const int amort_cap = static_cast<int>(max_masked_m > 0 ? max_masked_m : T_padded);
    if (blocks_per_expert > amort_cap) blocks_per_expert = amort_cap;
    if (blocks_per_expert < 1) blocks_per_expert = 1;
    dim3 grid(E, blocks_per_expert);

    constexpr size_t kProdElemBytes = kApplyGemm1Alpha ? sizeof(float) : sizeof(__nv_bfloat16);
    const size_t prod_bytes = static_cast<size_t>(N) * kProdElemBytes;
    const bool use_smem = (prod_bytes <= kSmemMaxBytesExtended);
    const bool need_ext_smem = (prod_bytes > kSmemMaxBytesDefault) && use_smem;
    const size_t prod_bytes_aligned = use_smem ? (prod_bytes + 15u) / 16u * 16u : 0u;
    const size_t smem_bytes = prod_bytes_aligned + (kWPB > 1 ? kWPB * sizeof(float) : 0u);

    const SiluMulFp8MaskedParams params = {
        .input = static_cast<const __nv_bfloat16*>(input.data_ptr()),
        .output = static_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        .output_scale = static_cast<float*>(output_scale.data_ptr()),
        .masked_m = static_cast<const int32_t*>(masked_m.data_ptr()),
        .stride_input_e = input.stride(0),
        .stride_input_t = input.stride(1),
        .stride_output_e = output.stride(0),
        .stride_output_t = output.stride(1),
        .stride_scale_e = output_scale.stride(0),
        .stride_scale_t = output_scale.stride(1),
        .N = N,
        .eps = static_cast<float>(eps),
        .swiglu_limit = static_cast<float>(swiglu_limit),
        .gemm1_alpha = static_cast<float>(gemm1_alpha),
        .gemm1_clamp_limit = static_cast<float>(gemm1_clamp_limit),
    };

    auto device = input.device();

    if (need_ext_smem) {
#define SET_EXT_SMEM(UE8M0, CACHE)                                                                             \
  do {                                                                                                         \
    auto fptr = std::bit_cast<const void*>(&silu_and_mul_masked_post_per_token_quant_fp8_kernel<               \
                                           kBlockThreads,                                                      \
                                           kApplySwigluLimit,                                                  \
                                           kApplyGemm1Alpha,                                                   \
                                           UE8M0,                                                              \
                                           CACHE>);                                                            \
    ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)); \
  } while (0)
      SET_EXT_SMEM(true, true);
      SET_EXT_SMEM(true, false);
      SET_EXT_SMEM(false, true);
      SET_EXT_SMEM(false, false);
#undef SET_EXT_SMEM
    }

#define DISPATCH(UE8M0, CACHE)                             \
  LaunchKernel(grid, kBlockThreads, device, smem_bytes)(   \
      silu_and_mul_masked_post_per_token_quant_fp8_kernel< \
          kBlockThreads,                                   \
          kApplySwigluLimit,                               \
          kApplyGemm1Alpha,                                \
          UE8M0,                                           \
          CACHE>,                                          \
      params)

    if (kScaleUe8m0) {
      if (use_smem)
        DISPATCH(true, true);
      else
        DISPATCH(true, false);
    } else {
      if (use_smem)
        DISPATCH(false, true);
      else
        DISPATCH(false, false);
    }
#undef DISPATCH
  }
};

}  // namespace
