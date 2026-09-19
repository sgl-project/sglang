#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <limits>

static_assert(sizeof(float) == 4, "bit-trick reciprocal assumes IEEE-754 binary32");

namespace sglang {
namespace {

constexpr int kGroupSize = 32;
constexpr int kWarpThreads = 32;
constexpr int kElemPerThread = 16;
constexpr int kThreadsPerGroup = kGroupSize / kElemPerThread;
constexpr float kQuantMax = 6.0f;
constexpr float kAbsmaxFloor = 1e-10f;
constexpr int kStages = 2;
constexpr int kBlocksTargetDefault = 2048;

struct alignas(16) SiluMulMxfp4Params2D {
  const __nv_bfloat16* __restrict__ input;
  uint8_t* __restrict__ output;
  uint8_t* __restrict__ output_scale;
  int64_t stride_input_n;
  int64_t stride_output_n;
  int64_t stride_scale_p_bytes;
  int64_t stride_scale_n_bytes;
  int32_t N;         // actual H (may not be multiple of kBlockN)
  int32_t N_padded;  // ceil to kBlockN (for grid sizing)
  int32_t num_tokens;
  float swiglu_limit;
};

__device__ __forceinline__ uint32_t
pack_4xe2m1x2(float q0, float q1, float q2, float q3, float q4, float q5, float q6, float q7) {
  uint32_t packed;
  asm volatile(
      "{\n\t"
      ".reg .b8 r0, r1, r2, r3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r0, %2, %1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r1, %4, %3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r2, %6, %5;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r3, %8, %7;\n\t"
      "mov.b32 %0, {r0, r1, r2, r3};\n\t"
      "}\n"
      : "=r"(packed)
      : "f"(q0), "f"(q1), "f"(q2), "f"(q3), "f"(q4), "f"(q5), "f"(q6), "f"(q7));
  return packed;
}

template <bool kApplySwigluLimit>
__device__ __forceinline__ __nv_bfloat162
silu_and_mul(__nv_bfloat162 gate, __nv_bfloat162 up, float swiglu_limit, float* fp32_prod = nullptr) {
  if constexpr (kApplySwigluLimit) {
    // clamp in bf16 to match upstream sglang / DeepGEMM
    const __nv_bfloat16 lim = __float2bfloat16_rn(swiglu_limit);
    const __nv_bfloat16 nlim = __float2bfloat16_rn(-swiglu_limit);
    const __nv_bfloat162 lim2 = __halves2bfloat162(lim, lim);
    const __nv_bfloat162 nlim2 = __halves2bfloat162(nlim, nlim);
    gate = __hmin2(gate, lim2);
    up = __hmin2(__hmax2(up, nlim2), lim2);
    const float g0 = __bfloat162float(__low2bfloat16(gate));
    const float g1 = __bfloat162float(__high2bfloat16(gate));
    const float u0 = __bfloat162float(__low2bfloat16(up));
    const float u1 = __bfloat162float(__high2bfloat16(up));
    const float silu0 = g0 * __ppu_sgmdf(g0);
    const float silu1 = g1 * __ppu_sgmdf(g1);
    const float p0 = u0 * silu0;
    const float p1 = u1 * silu1;
    if (fp32_prod) {
      fp32_prod[0] = p0;
      fp32_prod[1] = p1;
    }
    return __floats2bfloat162_rn(p0, p1);
  }
  const float g0 = __bfloat162float(__low2bfloat16(gate));
  const float g1 = __bfloat162float(__high2bfloat16(gate));
  const float silu0 = g0 * __ppu_sgmdf(g0);
  const float silu1 = g1 * __ppu_sgmdf(g1);
  const __nv_bfloat162 silu = __floats2bfloat162_rn(silu0, silu1);
  return __hmul2(up, silu);
}

__device__ __forceinline__ float group_reduce_max(float v) {
#pragma unroll
  for (int offset = kThreadsPerGroup >> 1; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
  }
  return v;
}

__device__ __forceinline__ void e8m0_scale(float absmax, float& quant_scale, uint8_t& scale_byte) {
  constexpr float kQuantMaxRcp = 1.0f / kQuantMax;
  const float dequant_scale = absmax * kQuantMaxRcp;
  const uint32_t ds_u32 = __float_as_uint(dequant_scale);
  const uint32_t ds_e8m0 = (ds_u32 + 0x007FFFFFu) & 0x7F800000u;
  quant_scale = __uint_as_float(0x7F000000u - ds_e8m0);
  scale_byte = static_cast<uint8_t>(ds_e8m0 >> 23);
}

__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <int kBlockN, bool kApplySwigluLimit, bool kIsFullBlock>
__global__ __launch_bounds__(kBlockN / kElemPerThread) void silu_and_mul_post_quant_mxfp4_kernel(
    const SiluMulMxfp4Params2D __grid_constant__ params) {
  constexpr int kThreadsPerBlock = kBlockN / kElemPerThread;
  constexpr int kPairsPerThread = kElemPerThread / 2;
  static_assert(kBlockN % kGroupSize == 0, "kBlockN must be multiple of 32");
  static_assert(kElemPerThread % 2 == 0, "bf16x2 path needs even elements per thread");

  constexpr bool kSingleWarp = (kThreadsPerBlock <= kWarpThreads);

  __shared__ alignas(16) __nv_bfloat16 smem_gate[kStages][kThreadsPerBlock][kElemPerThread];
  __shared__ alignas(16) __nv_bfloat16 smem_up[kStages][kThreadsPerBlock][kElemPerThread];

  const int token_block_id = blockIdx.y;
  const int hidden_block = blockIdx.x;
  const int tid = threadIdx.x;

  const int n_offset_elem = hidden_block * kBlockN;
  if (n_offset_elem >= params.N_padded) return;

  const int group_in_block = tid / kThreadsPerGroup;
  const int lane_in_group = tid % kThreadsPerGroup;
  const int thread_n_offset = n_offset_elem + group_in_block * kGroupSize + lane_in_group * kElemPerThread;

  const bool thread_valid = (thread_n_offset + kElemPerThread <= params.N);

  const int group_index_in_full = (n_offset_elem / kGroupSize) + group_in_block;
  const int scale_pair_idx = group_index_in_full >> 1;
  const int scale_lo_hi = group_index_in_full & 1;

  const int n_tokens = params.num_tokens;
  const int token_stride = gridDim.y;

  constexpr int kCpAsync16PerThread = kElemPerThread / 8;
  static_assert(kElemPerThread % 8 == 0, "kElemPerThread must be multiple of 8 for cp.async.16");

  auto issue_token_load = [&](int stage, int t) {
    if (t < n_tokens) {
      const __nv_bfloat16* row_base = params.input + t * params.stride_input_n;
      if constexpr (kIsFullBlock) {
        const __nv_bfloat16* gate_ptr = row_base + thread_n_offset;
        const __nv_bfloat16* up_ptr = row_base + params.N + thread_n_offset;
#pragma unroll
        for (int i = 0; i < kCpAsync16PerThread; ++i) {
          cp_async_16B(&smem_gate[stage][tid][i * 8], gate_ptr + i * 8);
          cp_async_16B(&smem_up[stage][tid][i * 8], up_ptr + i * 8);
        }
      } else {
#pragma unroll
        for (int i = 0; i < kElemPerThread; ++i) {
          const int elem_idx = thread_n_offset + i;
          if (elem_idx < params.N) {
            smem_gate[stage][tid][i] = row_base[elem_idx];
            smem_up[stage][tid][i] = row_base[params.N + elem_idx];
          } else {
            smem_gate[stage][tid][i] = __float2bfloat16_rn(0.0f);
            smem_up[stage][tid][i] = __float2bfloat16_rn(0.0f);
          }
        }
      }
    } else {
      if constexpr (!kIsFullBlock) {
#pragma unroll
        for (int i = 0; i < kElemPerThread; ++i) {
          smem_gate[stage][tid][i] = __float2bfloat16_rn(0.0f);
          smem_up[stage][tid][i] = __float2bfloat16_rn(0.0f);
        }
      }
    }
    cp_async_commit_group();
  };

  int issue_stage = 0;
  int t_load = token_block_id;
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    issue_token_load(issue_stage, t_load);
    issue_stage = (issue_stage + 1) % kStages;
    t_load += token_stride;
  }

  int compute_stage = 0;
  for (int t = token_block_id; t < n_tokens; t += token_stride) {
    issue_token_load(issue_stage, t_load);
    issue_stage = (issue_stage + 1) % kStages;
    t_load += token_stride;

    cp_async_wait_group<kStages - 1>();
    if constexpr (!kSingleWarp) __syncthreads();

    auto* gate_pairs = reinterpret_cast<__nv_bfloat162*>(&smem_gate[compute_stage][tid][0]);
    auto* up_pairs = reinterpret_cast<__nv_bfloat162*>(&smem_up[compute_stage][tid][0]);

    alignas(16) __nv_bfloat16 prod_bf16[kElemPerThread];
    auto* prod_pairs = reinterpret_cast<__nv_bfloat162*>(prod_bf16);

    float local_absmax;
    float fp32_prods[kPairsPerThread][2];
    if constexpr (kApplySwigluLimit) {
      local_absmax = 0.0f;
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        prod_pairs[i] = silu_and_mul<kApplySwigluLimit>(gate_pairs[i], up_pairs[i], params.swiglu_limit, fp32_prods[i]);
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(fp32_prods[i][0]), fabsf(fp32_prods[i][1])));
      }
    } else {
      __nv_bfloat162 absmax_v2 = __float2bfloat162_rn(0.0f);
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        prod_pairs[i] = silu_and_mul<kApplySwigluLimit>(gate_pairs[i], up_pairs[i], params.swiglu_limit);
        absmax_v2 = __hmax2(absmax_v2, __habs2(prod_pairs[i]));
      }
      local_absmax = fmaxf(__bfloat162float(__low2bfloat16(absmax_v2)), __bfloat162float(__high2bfloat16(absmax_v2)));
    }
    local_absmax = fmaxf(group_reduce_max(local_absmax), kAbsmaxFloor);

    float quant_scale;
    uint8_t scale_byte;
    e8m0_scale(local_absmax, quant_scale, scale_byte);

    float q[kElemPerThread];
    if constexpr (kApplySwigluLimit) {
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        q[2 * i] = fp32_prods[i][0] * quant_scale;
        q[2 * i + 1] = fp32_prods[i][1] * quant_scale;
      }
    } else {
      const __nv_bfloat162 quant_scale_v2 = __float2bfloat162_rn(quant_scale);
      __nv_bfloat162 q_pairs_scaled[kPairsPerThread];
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        q_pairs_scaled[i] = __hmul2(prod_pairs[i], quant_scale_v2);
      }
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        q[2 * i] = __bfloat162float(__low2bfloat16(q_pairs_scaled[i]));
        q[2 * i + 1] = __bfloat162float(__high2bfloat16(q_pairs_scaled[i]));
      }
    }
    uint32_t packed[kCpAsync16PerThread];
#pragma unroll
    for (int p = 0; p < kCpAsync16PerThread; ++p) {
      packed[p] = pack_4xe2m1x2(
          q[p * 8 + 0],
          q[p * 8 + 1],
          q[p * 8 + 2],
          q[p * 8 + 3],
          q[p * 8 + 4],
          q[p * 8 + 5],
          q[p * 8 + 6],
          q[p * 8 + 7]);
    }

    if constexpr (kIsFullBlock) {
      const int out_byte_offset = thread_n_offset / 2;
      auto* out_u32 = reinterpret_cast<uint32_t*>(params.output + t * params.stride_output_n + out_byte_offset);
      if constexpr (kCpAsync16PerThread == 1) {
        out_u32[0] = packed[0];
      } else if constexpr (kCpAsync16PerThread == 2) {
        *reinterpret_cast<uint2*>(out_u32) = make_uint2(packed[0], packed[1]);
      } else {
#pragma unroll
        for (int p = 0; p < kCpAsync16PerThread; ++p)
          out_u32[p] = packed[p];
      }
    } else {
      const int valid_elems =
          (thread_n_offset >= params.N)
              ? 0
              : ((thread_n_offset + kElemPerThread <= params.N) ? kElemPerThread : (params.N - thread_n_offset));
      if (valid_elems == kElemPerThread) {
        const int out_byte_offset = thread_n_offset / 2;
        auto* out_u32 = reinterpret_cast<uint32_t*>(params.output + t * params.stride_output_n + out_byte_offset);
        if constexpr (kCpAsync16PerThread == 1) {
          out_u32[0] = packed[0];
        } else if constexpr (kCpAsync16PerThread == 2) {
          *reinterpret_cast<uint2*>(out_u32) = make_uint2(packed[0], packed[1]);
        } else {
#pragma unroll
          for (int p = 0; p < kCpAsync16PerThread; ++p)
            out_u32[p] = packed[p];
        }
      } else if (valid_elems > 0) {
        // valid_elems is always even: N is even (Python assert) and
        // thread_n_offset is a multiple of kElemPerThread (16).
        const int valid_bytes = valid_elems / 2;
        uint8_t* byte_out = params.output + t * params.stride_output_n + thread_n_offset / 2;
        const uint8_t* packed_bytes = reinterpret_cast<const uint8_t*>(packed);
        for (int b = 0; b < valid_bytes; ++b) {
          byte_out[b] = packed_bytes[b];
        }
      }
    }

    if (lane_in_group == 0) {
      uint8_t byte;
      if constexpr (kIsFullBlock) {
        byte = scale_byte;
      } else {
        byte = (group_index_in_full * kGroupSize < params.N) ? scale_byte : 0;
      }
      params
          .output_scale[scale_pair_idx * params.stride_scale_p_bytes + t * params.stride_scale_n_bytes + scale_lo_hi] =
          byte;
    }

    compute_stage = (compute_stage + 1) % kStages;
  }
}

template <int kBlockN, bool kApplySwigluLimit>
struct SiluMulMxfp4TP {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      double swiglu_limit) {
    using namespace host;

    const int64_t num_tokens = input.size(0);
    const int64_t two_H = input.size(1);
    const int32_t N = static_cast<int32_t>(two_H / 2);
    const int32_t N_padded = ((N + kBlockN - 1) / kBlockN) * kBlockN;
    const int hidden_blocks = N_padded / kBlockN;

    RuntimeCheck(num_tokens > 0, "num_tokens must be positive");
    RuntimeCheck(N > 0 && N % 2 == 0, "N must be positive and even, got N=", N);

    int blocks_per_hidden = kBlocksTargetDefault / hidden_blocks;
    const int amort_cap = (static_cast<int>(num_tokens) + 1) / 2;
    if (blocks_per_hidden > amort_cap) blocks_per_hidden = amort_cap;
    if (blocks_per_hidden < 1) blocks_per_hidden = 1;

    dim3 grid(hidden_blocks, blocks_per_hidden);
    constexpr int threads = kBlockN / kElemPerThread;

    constexpr int64_t scale_es = 2;

    const SiluMulMxfp4Params2D params = {
        .input = static_cast<const __nv_bfloat16*>(input.data_ptr()),
        .output = static_cast<uint8_t*>(output.data_ptr()),
        .output_scale = static_cast<uint8_t*>(output_scale.data_ptr()),
        .stride_input_n = input.stride(0),
        .stride_output_n = output.stride(0),
        .stride_scale_p_bytes = output_scale.stride(0) * scale_es,
        .stride_scale_n_bytes = output_scale.stride(1) * scale_es,
        .N = N,
        .N_padded = N_padded,
        .num_tokens = static_cast<int32_t>(num_tokens),
        .swiglu_limit = static_cast<float>(swiglu_limit),
    };

    auto device = input.device();
    if (N == N_padded) {
      LaunchKernel(grid, threads, device)(
          silu_and_mul_post_quant_mxfp4_kernel<kBlockN, kApplySwigluLimit, true>, params);
    } else {
      LaunchKernel(grid, threads, device)(
          silu_and_mul_post_quant_mxfp4_kernel<kBlockN, kApplySwigluLimit, false>, params);
    }
  }
};

}  // namespace
}  // namespace sglang
