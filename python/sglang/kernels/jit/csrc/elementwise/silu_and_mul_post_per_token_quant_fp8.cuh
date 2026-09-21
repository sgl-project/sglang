#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sglang {
namespace {

constexpr int kWarpThreads = 32;
constexpr int kElemPerThread = 8;
constexpr float kFp8E4M3Max = 448.0f;
constexpr float kAbsmaxFloor = 1e-10f;

struct alignas(16) SiluMulFp8Params {
  const __nv_bfloat16* __restrict__ input;
  __nv_fp8_e4m3* __restrict__ output;
  float* __restrict__ output_scale;
  int32_t N;
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

// oai-swiglu (MiniMax-M3 / gpt-oss), matching the bf16 eager reference
// swiglu_no_interleaved_with_alpha_and_limit:
//   gate = min(gate, L); up = clamp(up, -L, L);
//   out  = gate * sigmoid(gate * alpha) * (up + 1)
// All arithmetic stays in bf16; only the sigmoid input (gate * alpha) is
// computed in fp32, mirroring torch's fp32 opmath for bf16 ops.
__device__ __forceinline__ __nv_bfloat162
oai_swiglu(__nv_bfloat162 gate, __nv_bfloat162 up, float alpha, float clamp_limit) {
  const __nv_bfloat16 lim = __float2bfloat16_rn(clamp_limit);
  const __nv_bfloat16 nlim = __float2bfloat16_rn(-clamp_limit);
  const __nv_bfloat162 lim2 = __halves2bfloat162(lim, lim);
  const __nv_bfloat162 nlim2 = __halves2bfloat162(nlim, nlim);
  gate = __hmin2(gate, lim2);
  up = __hmin2(__hmax2(up, nlim2), lim2);
  const float a0 = __bfloat162float(__low2bfloat16(gate)) * alpha;
  const float a1 = __bfloat162float(__high2bfloat16(gate)) * alpha;
  const __nv_bfloat162 sig = __floats2bfloat162_rn(__ppu_sgmdf(a0), __ppu_sgmdf(a1));
  const __nv_bfloat162 one2 = __float2bfloat162_rn(1.0f);
  return __hmul2(__hmul2(gate, sig), __hadd2(up, one2));
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

template <int kBlockThreads, bool kApplySwigluLimit, bool kApplyGemm1Alpha, bool kCacheInSmem>
__global__ __launch_bounds__(kBlockThreads, 2) void silu_and_mul_post_per_token_quant_fp8_kernel(
    const SiluMulFp8Params __grid_constant__ params) {
  constexpr int kPairsPerThread = kElemPerThread / 2;  // 4
  static_assert(!(kApplySwigluLimit && kApplyGemm1Alpha), "swiglu_limit and gemm1_alpha are mutually exclusive");

  const int token_id = blockIdx.x;
  const int tid = threadIdx.x;

  const __nv_bfloat16* in_row = params.input + static_cast<size_t>(token_id) * 2 * params.N;
  __nv_fp8_e4m3* out_row = params.output + static_cast<size_t>(token_id) * params.N;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* smem_prod = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  float* smem_warp_max =
      reinterpret_cast<float*>(smem_raw + (kCacheInSmem ? (sizeof(__nv_bfloat16) * params.N + 15u) / 16u * 16u : 0u));

  const int n_vec = params.N / kElemPerThread;
  float local_absmax = 0.0f;

  // Pass 1: activation + per-token absmax (products cached in smem when possible).
  for (int v = tid; v < n_vec; v += kBlockThreads) {
    const int elem_off = v * kElemPerThread;

    uint4 gate_pack = *reinterpret_cast<const uint4*>(in_row + elem_off);
    uint4 up_pack = *reinterpret_cast<const uint4*>(in_row + params.N + elem_off);
    auto* gate_pairs = reinterpret_cast<__nv_bfloat162*>(&gate_pack);
    auto* up_pairs = reinterpret_cast<__nv_bfloat162*>(&up_pack);

    __nv_bfloat162 prod_pairs[kPairsPerThread];
    __nv_bfloat162 absmax_v2 = __float2bfloat162_rn(0.0f);

#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
      __nv_bfloat162 p;
      if constexpr (kApplyGemm1Alpha) {
        p = oai_swiglu(gate_pairs[i], up_pairs[i], params.gemm1_alpha, params.gemm1_clamp_limit);
      } else {
        p = silu_and_mul<kApplySwigluLimit>(gate_pairs[i], up_pairs[i], params.swiglu_limit);
      }
      prod_pairs[i] = p;
      absmax_v2 = __hmax2(absmax_v2, __habs2(p));
    }

    if constexpr (kCacheInSmem) {
      *reinterpret_cast<uint4*>(smem_prod + elem_off) = *reinterpret_cast<const uint4*>(prod_pairs);
    }

    const __nv_bfloat16 m_bf16 = __hmax(__low2bfloat16(absmax_v2), __high2bfloat16(absmax_v2));
    local_absmax = fmaxf(local_absmax, __bfloat162float(m_bf16));
  }

  const float row_absmax = block_reduce_max<kBlockThreads>(local_absmax, smem_warp_max);

  const float scale = fmaxf(row_absmax, params.eps) / kFp8E4M3Max;
  const float inv_scale = 1.0f / scale;
  if (tid == 0) params.output_scale[token_id] = scale;

  // Pass 2: rescale the (cached) products and store packed e4m3. Each thread
  // re-reads only the smem slots it wrote in pass 1, so no cross-thread sync
  // is needed between the passes.
  for (int v = tid; v < n_vec; v += kBlockThreads) {
    const int elem_off = v * kElemPerThread;
    __nv_bfloat162 prod_pairs[kPairsPerThread];

    if constexpr (kCacheInSmem) {
      *reinterpret_cast<uint4*>(prod_pairs) = *reinterpret_cast<const uint4*>(smem_prod + elem_off);
    } else {
      uint4 gate_pack = *reinterpret_cast<const uint4*>(in_row + elem_off);
      uint4 up_pack = *reinterpret_cast<const uint4*>(in_row + params.N + elem_off);
      auto* gate_pairs = reinterpret_cast<__nv_bfloat162*>(&gate_pack);
      auto* up_pairs = reinterpret_cast<__nv_bfloat162*>(&up_pack);
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        if constexpr (kApplyGemm1Alpha) {
          prod_pairs[i] = oai_swiglu(gate_pairs[i], up_pairs[i], params.gemm1_alpha, params.gemm1_clamp_limit);
        } else {
          prod_pairs[i] = silu_and_mul<kApplySwigluLimit>(gate_pairs[i], up_pairs[i], params.swiglu_limit);
        }
      }
    }

    uint2 fp8_pack;
    uint16_t* fp8_arr = reinterpret_cast<uint16_t*>(&fp8_pack);
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
      const float f_lo = __bfloat162float(__low2bfloat16(prod_pairs[i])) * inv_scale;
      const float f_hi = __bfloat162float(__high2bfloat16(prod_pairs[i])) * inv_scale;
      fp8_arr[i] = cvt_fp32x2_to_e4m3x2(f_lo, f_hi);
    }
    *reinterpret_cast<uint2*>(out_row + elem_off) = fp8_pack;
  }
}

template <int kBlockThreads, bool kApplySwigluLimit, bool kApplyGemm1Alpha>
struct SiluMulFp8TP {
  static_assert(!kApplySwigluLimit || !kApplyGemm1Alpha, "swiglu_limit and gemm1_alpha are mutually exclusive");

  static constexpr int kSmemNMaxDefault = 24000;   // fits in default 48KB
  static constexpr int kSmemNMaxExtended = 49000;  // fits with ~98KB

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      double swiglu_limit,
      double gemm1_alpha,
      double gemm1_clamp_limit,
      double eps) {
    using namespace host;

    const int M = static_cast<int>(input.size(0));
    const int two_N = static_cast<int>(input.size(1));
    const int N = two_N / 2;

    RuntimeCheck(input.ndim() == 2, "input must be 2D (M, 2N)");
    RuntimeCheck(output.ndim() == 2, "output must be 2D (M, N)");
    RuntimeCheck(output_scale.ndim() == 2, "output_scale must be 2D (M, 1)");
    RuntimeCheck(output.size(0) == M && output.size(1) == N, "output must have shape (M, N)");
    RuntimeCheck(output_scale.size(0) == M && output_scale.size(1) == 1, "output_scale must have shape (M, 1)");
    RuntimeCheck(input.dtype() == DLDataType{DLDataTypeCode::kDLBfloat, 16, 1}, "input dtype must be BFloat16");
    RuntimeCheck(M > 0, "M must be positive");
    RuntimeCheck(two_N % 2 == 0, "input last dim must be even");
    RuntimeCheck(N % kElemPerThread == 0, "N must be multiple of ", kElemPerThread, ", got N=", N);
    RuntimeCheck(input.stride(1) == 1, "input must be contiguous in the last dim");
    RuntimeCheck(
        input.stride(0) == two_N && output.stride(1) == 1 && output.stride(0) == N,
        "input/output must be row-major contiguous");
    RuntimeCheck(output_scale.stride(0) == 1, "output_scale must be contiguous in dim 0");

    const bool use_smem = (N <= kSmemNMaxExtended);
    const bool need_ext_smem = (N > kSmemNMaxDefault) && use_smem;
    constexpr int kWPB = kBlockThreads / kWarpThreads;

    const size_t prod_bytes_aligned =
        use_smem ? (static_cast<size_t>(N) * sizeof(__nv_bfloat16) + 15u) / 16u * 16u : 0u;
    const size_t smem_bytes = prod_bytes_aligned + (kWPB > 1 ? kWPB * sizeof(float) : 0u);

    const SiluMulFp8Params params = {
        .input = static_cast<const __nv_bfloat16*>(input.data_ptr()),
        .output = static_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        .output_scale = static_cast<float*>(output_scale.data_ptr()),
        .N = N,
        .eps = static_cast<float>(eps),
        .swiglu_limit = static_cast<float>(swiglu_limit),
        .gemm1_alpha = static_cast<float>(gemm1_alpha),
        .gemm1_clamp_limit = static_cast<float>(gemm1_clamp_limit),
    };

    auto device = input.device();
    dim3 grid(M);

    if (need_ext_smem) {
#define SET_EXT_SMEM(CACHE)                                                                                        \
  do {                                                                                                             \
    auto fptr = std::bit_cast<const void*>(                                                                        \
        &silu_and_mul_post_per_token_quant_fp8_kernel<kBlockThreads, kApplySwigluLimit, kApplyGemm1Alpha, CACHE>); \
    ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));     \
  } while (0)
      SET_EXT_SMEM(true);
      SET_EXT_SMEM(false);
#undef SET_EXT_SMEM
    }

#define DISPATCH(CACHE)                                  \
  LaunchKernel(grid, kBlockThreads, device, smem_bytes)( \
      silu_and_mul_post_per_token_quant_fp8_kernel<kBlockThreads, kApplySwigluLimit, kApplyGemm1Alpha, CACHE>, params)

    if (use_smem)
      DISPATCH(true);
    else
      DISPATCH(false);
#undef DISPATCH
  }
};

}  // namespace
}  // namespace sglang
