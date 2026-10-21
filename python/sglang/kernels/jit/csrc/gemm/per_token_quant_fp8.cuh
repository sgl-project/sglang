#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#ifndef USE_ROCM
#include <cuda_pipeline.h>
#endif

// Ported from the v0.5.18_rel AOT gemm/per_token_quant_fp8.cu implementation.
// Restore persistent-prefill and small-batch-decode dispatch while retaining
// the JIT TensorView interface, stream resolution and padded-output contract.
namespace sglang {

constexpr uint32_t kPerTokenQuantWarpSize = 32;
constexpr uint32_t kPerTokenQuantTokensPerCTA = 8;

#ifndef USE_ROCM
// Match AOT's 16/8/4-element chunks using at most 128-bit memory transactions.
// AlignedVector<T, 16> itself exceeds the sm80 vector limit for fp16/fp32.
template <typename T, int N>
struct PerTokenQuantVector {
  alignas(16) T values[N];
  SGL_DEVICE void cast_load(const T* ptr) {
    constexpr int kChunk = N < 16 / sizeof(T) ? N : 16 / sizeof(T);
#pragma unroll
    for (int i = 0; i < N; i += kChunk) {
      device::AlignedVector<T, kChunk> chunk;
      chunk.load(ptr + i);
#pragma unroll
      for (int j = 0; j < kChunk; ++j)
        values[i + j] = chunk[j];
    }
  }
  SGL_DEVICE T& operator[](int i) {
    return values[i];
  }
  SGL_DEVICE const T& operator[](int i) const {
    return values[i];
  }
};

SGL_DEVICE float per_token_quant_block_max(float value) {
  __shared__ float warp_maxima[kPerTokenQuantWarpSize];
  const int lane = threadIdx.x % kPerTokenQuantWarpSize;
  const int warp = threadIdx.x / kPerTokenQuantWarpSize;
  value = device::warp::reduce_max(value);
  if (lane == 0) warp_maxima[warp] = value;
  __syncthreads();
  value = threadIdx.x < blockDim.x / kPerTokenQuantWarpSize ? warp_maxima[lane] : 0.0f;
  if (warp == 0) value = device::warp::reduce_max(value);
  return value;
}

#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
template <typename T, typename DST_DTYPE, int kVecSize>
SGL_DEVICE void
per_token_quant_store_fp32(const PerTokenQuantVector<T, kVecSize>& input_vec, float scale_inv, DST_DTYPE* output_arr) {
  // bf16/fp32 input: use cvt.rn.satfinite.e4m3x2.f32
  uint16_t* output_u16 = reinterpret_cast<uint16_t*>(output_arr);
#pragma unroll
  for (uint32_t j = 0; j < kVecSize / 2; ++j) {
    float val_a = static_cast<float>(input_vec[j * 2]) * scale_inv;
    float val_b = static_cast<float>(input_vec[j * 2 + 1]) * scale_inv;

    // cvt.rn.satfinite.e4m3x2.f32: d[15:8]=cvt(first_arg), d[7:0]=cvt(second_arg)
    // On little-endian: low byte (d[7:0]) stored first -> output_arr[j*2]
    // So pass val_b as first arg (-> high byte = arr[j*2+1]),
    //         val_a as second arg (-> low byte = arr[j*2])
    uint16_t packed_fp8x2;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(packed_fp8x2) : "f"(val_b), "f"(val_a));

    output_u16[j] = packed_fp8x2;
  }

  // Handle odd element (won't trigger for kVecSize = 4, 8, 16)
  if constexpr (kVecSize % 2 != 0) {
    float val = static_cast<float>(input_vec[kVecSize - 1]) * scale_inv;
    val = fmaxf(fminf(val, device::math::FP8_E4M3_MAX), -device::math::FP8_E4M3_MAX);
    output_arr[kVecSize - 1] = static_cast<DST_DTYPE>(val);
  }
}
#endif

template <typename T, typename DST_DTYPE, int kVecSize>
SGL_DEVICE void per_token_quant_store(
    const PerTokenQuantVector<T, kVecSize>& input_vec, float scale_inv, DST_DTYPE* token_output, int i) {
  alignas(16) DST_DTYPE output_arr[kVecSize];
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  // SM89+ path: use cvt.rn.satfinite.e4m3x2 for packed 2-element conversion
  // satfinite automatically clamps, no manual fmaxf/fminf needed
  if constexpr (std::is_same_v<T, __nv_half>) {
    // Tiny-magnitude rows (max < 448 / 65504) produce a scale_inv above the
    // fp16 finite range; converting it to fp16 yields inf, so 0 * inf = NaN.
    // Keep the packed fp16 fast path only when scale_inv is representable.
    if (scale_inv <= 65504.0f) {
      // Full fp16 path: mul.rn.f16x2 + cvt.rn.satfinite.e4m3x2.f16x2
      __half scale_inv_h = __float2half(scale_inv);
      uint32_t scale_inv_f16x2;
      asm("mov.b32 %0, {%1, %2};\n"
          : "=r"(scale_inv_f16x2)
          : "h"(*(reinterpret_cast<const uint16_t*>(&scale_inv_h))),
            "h"(*(reinterpret_cast<const uint16_t*>(&scale_inv_h))));

      uint16_t* output_u16 = reinterpret_cast<uint16_t*>(output_arr);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize / 2; ++j) {
        // Pack two fp16 inputs into a uint32
        uint32_t input_f16x2;
        asm("mov.b32 %0, {%1, %2};\n"
            : "=r"(input_f16x2)
            : "h"(reinterpret_cast<const uint16_t*>(&input_vec[0])[j * 2]),
              "h"(reinterpret_cast<const uint16_t*>(&input_vec[0])[j * 2 + 1]));

        // fp16x2 multiply
        uint32_t scaled_f16x2;
        asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(scaled_f16x2) : "r"(input_f16x2), "r"(scale_inv_f16x2));

        // fp16x2 -> e4m3x2
        // cvt.rn.satfinite.e4m3x2.f16x2: d[15:8]=cvt(a[31:16]), d[7:0]=cvt(a[15:0])
        // On little-endian: a[15:0] is first element, a[31:16] is second
        // So d[7:0]=cvt(first)=output_arr[j*2], d[15:8]=cvt(second)=output_arr[j*2+1] -> correct!
        uint16_t packed_fp8x2;
        asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(packed_fp8x2) : "r"(scaled_f16x2));

        output_u16[j] = packed_fp8x2;
      }

      // Handle odd element (won't trigger for kVecSize = 4, 8, 16)
      if constexpr (kVecSize % 2 != 0) {
        float val = static_cast<float>(input_vec[kVecSize - 1]) * scale_inv;
        val = fmaxf(fminf(val, device::math::FP8_E4M3_MAX), -device::math::FP8_E4M3_MAX);
        output_arr[kVecSize - 1] = static_cast<DST_DTYPE>(val);
      }
    } else {
      per_token_quant_store_fp32(input_vec, scale_inv, output_arr);
    }
  } else {
    per_token_quant_store_fp32(input_vec, scale_inv, output_arr);
  }
#else
  // Original path for SM < 89 and ROCm
#pragma unroll
  for (uint32_t j = 0; j < kVecSize; ++j) {
    float val = static_cast<float>(input_vec[j]) * scale_inv;
    val = fmaxf(fminf(val, device::math::FP8_E4M3_MAX), -device::math::FP8_E4M3_MAX);
    output_arr[j] = static_cast<DST_DTYPE>(val);
  }
#endif
  if constexpr (kVecSize == 16) {
    *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
  } else {
    // Use element-wise copy for vector size 8 to ensure correctness
    for (int k = 0; k < kVecSize; ++k) {
      token_output[i * kVecSize + k] = output_arr[k];
    }
  }
}

#if defined(SGL_PER_TOKEN_QUANT_PPU) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
template <typename Dtype, bool trans>
__device__ __forceinline__ void load_128b_aiu(
    void* smem_ptr,
    const void* gmem_ptr,
    const int32_t tensor_dims_x,
    const int32_t tensor_dims_z,
    const int32_t cube_w,
    const int32_t cube_h,
    const int32_t x_offset,
    const int32_t z_offset) {
  uint64_t tensor_stride_w = tensor_dims_x * 2;
  uint64_t tensor_stride_n = tensor_stride_w * tensor_dims_z;
  int swzl_mode = cube_w * 2 == 128 ? 0 : 1;
  static_assert(!trans && sizeof(Dtype) == 2);
  asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
      "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n" ::"r"(smem_ptr),
      "l"(gmem_ptr),
      "r"(tensor_dims_x),
      "r"(tensor_dims_z),
      "r"(1),
      "r"(cube_w),
      "r"(cube_h),
      "r"(1),
      "l"(tensor_stride_w),
      "l"(tensor_stride_n),
      "r"(x_offset),
      "r"(z_offset),
      "r"(0),
      "r"(swzl_mode));
}
#endif

// ---------------------------------------------------------------------------
// 1. Warp‑local with configurable shared memory
//    • One warp handles one token.
//    • Eight tokens per 256‑thread CTA.
//    • Shared memory usage is configurable via template parameter.
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, int kTokensPerCTA = 8, int kVecSize = 16, bool USE_SMEM = true>
__global__ void per_token_quant_fp8_warp_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int warp_id = threadIdx.x / kPerTokenQuantWarpSize;        // 0‑7  (8 warps)
  const int lane_id = threadIdx.x & (kPerTokenQuantWarpSize - 1);  // 0‑31
  const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
  if (token_id >= num_tokens) return;

  // Global tensors for this token
  const T* token_input = input + token_id * hidden_dim;
  DST_DTYPE* token_output = output_q + token_id * hidden_dim;
  float* token_scale = output_s + token_id;

  extern __shared__ __align__(16) char smem_buffer[];
  const int smem_padding = 32;  // Pad to bank boundary (32 banks * 4 bytes = 128 bytes)
  const int warp_smem_stride = (hidden_dim * sizeof(T) + smem_padding - 1) / smem_padding * smem_padding;
  const int warp_smem_offset = warp_id * warp_smem_stride;
  T* shared_input = reinterpret_cast<T*>(smem_buffer + warp_smem_offset);

  //
  // Pass-1: Load data and compute max_value
  //
  float max_value = 0.f;
  using vec_t = PerTokenQuantVector<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  for (int32_t i = lane_id; i < num_vec_elems; i += kPerTokenQuantWarpSize) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

    // Store to shared memory if USE_SMEM=true
    if constexpr (USE_SMEM) {
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        shared_input[i * kVecSize + j] = input_vec[j];
      }
    }

    // Compute max value in parallel
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
    }
  }

  // Ensure all threads in the warp have finished writing to shared memory
  if constexpr (USE_SMEM) {
    __syncwarp();
  }

  float warp_max = device::warp::reduce_max(max_value);

  // NOTE: one CTA has multiple warps (each warp handles one token), so `scale`
  // must be per-warp/per-thread (register) instead of a single shared variable.
  const float scale = warp_max / device::math::FP8_E4M3_MAX;
  // Broadcast scale
  if (lane_id == 0) {
    token_scale[0] = scale;
  }
  const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  //
  // Pass-2: Quantize and write back
  //
  for (int i = lane_id; i < num_vec_elems; i += kPerTokenQuantWarpSize) {
    vec_t input_vec;

    if constexpr (USE_SMEM) {
      // Load from shared memory
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        input_vec[j] = shared_input[i * kVecSize + j];
      }
    } else {
      // Reload from global memory
      input_vec.cast_load(token_input + i * kVecSize);
    }

    per_token_quant_store(input_vec, scale_inv, token_output, i);
  }
}

// ---------------------------------------------------------------------------
// 1b. Persistent warp‑local kernel with cp.async single-buffered smem
//     • Each warp persistently iterates over multiple tokens.
//     • Single smem buffer per warp: rely on multi-CTA warp scheduling
//       (8 CTA/SM target) to hide memory latency instead of double-buffer.
//     • Grid size is sized to OCC (sm_count * blocks_per_sm) so launch
//       overhead is paid once instead of per token-batch.
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, int kTokensPerCTA = 8, int kVecSize = 16, bool useAIU = false>
__global__ void per_token_quant_fp8_persistent_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int warp_id = threadIdx.x / kPerTokenQuantWarpSize;
  const int lane_id = threadIdx.x & (kPerTokenQuantWarpSize - 1);
  const int64_t first_token = static_cast<int64_t>(blockIdx.x) * kTokensPerCTA + warp_id;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kTokensPerCTA;

  // 128B aligned per-warp stride (cp.async friendly + bank-conflict avoidance)
  extern __shared__ __align__(16) char smem_buffer[];
  const int warp_smem_stride = (hidden_dim * static_cast<int>(sizeof(T)) + 127) / 128 * 128;
  T* smem_buf = reinterpret_cast<T*>(smem_buffer + warp_id * warp_smem_stride);

  // Number of 16B copies per kVecSize-element chunk (compile-time constant)
  constexpr int kCopiesPerVec = kVecSize * static_cast<int>(sizeof(T)) / 16;

  using vec_t = PerTokenQuantVector<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  // Main persistent loop: each iteration handles one token per warp
  for (int64_t token_id = first_token; token_id < num_tokens; token_id += stride) {
    // Step 1: async copy current token to smem
    // Use outer loop over num_vec_elems + inner unrolled loop (kCopiesPerVec is
    // compile-time: 2 for fp16/bf16, 4 for fp32) to enable full unrolling.
    const T* src = input + token_id * hidden_dim;
    // hidden_dim x fp16 per warp
#if defined(SGL_PER_TOKEN_QUANT_PPU) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    if constexpr (sizeof(T) == 2 and useAIU) {
      constexpr int AIU_CHUNK_SLISE_4 = 16 * 64;
      for (int aiu_cp_offset = 0; aiu_cp_offset < hidden_dim; aiu_cp_offset += AIU_CHUNK_SLISE_4) {
        load_128b_aiu<T, false>(smem_buf + aiu_cp_offset, src + aiu_cp_offset, 64, 16, 64, 16, 0, 0);
      }
    } else
#endif
    {
      for (int32_t i = lane_id; i < num_vec_elems; i += kPerTokenQuantWarpSize) {
#pragma unroll
        for (int j = 0; j < kCopiesPerVec; ++j) {
          __pipeline_memcpy_async(
              reinterpret_cast<char*>(smem_buf + i * kVecSize) + j * 16,
              reinterpret_cast<const char*>(src + i * kVecSize) + j * 16,
              16);
        }
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncwarp();

    //
    // Pass-1: read from smem, compute warp-level absolute max
    //         Vectorized path: stay in native fp16x2/bf16x2 precision
    //
    float warp_max_result;

    if constexpr (std::is_same_v<T, __nv_half>) {
      // fp16 path: __habs2 + __hmax2 (SM53+)
      // Use outer loop over num_vec_elems + inner unrolled loop over kVecSize/2
      // to enable compile-time unrolling (kVecSize is a template constant).
      __half2 max_h2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));

      for (int32_t i = lane_id; i < num_vec_elems; i += kPerTokenQuantWarpSize) {
        const __half2* chunk = reinterpret_cast<const __half2*>(smem_buf + i * kVecSize);
#pragma unroll
        for (uint32_t j = 0; j < kVecSize / 2; ++j) {
          __half2 val = chunk[j];
          __half2 abs_val = __habs2(val);
          max_h2 = __hmax2(max_h2, abs_val);
        }
      }

      // Warp reduce in fp16x2
#pragma unroll
      for (int offset = kPerTokenQuantWarpSize / 2; offset > 0; offset >>= 1) {
        __half2 tmp = __shfl_xor_sync(0xffffffff, max_h2, offset);
        max_h2 = __hmax2(max_h2, tmp);
      }

      // Final conversion: reduce the 2 lanes of half2 to a single fp32 max
      float2 max_f2 = __half22float2(max_h2);
      warp_max_result = fmaxf(max_f2.x, max_f2.y);

#if defined(SGL_PER_TOKEN_QUANT_PPU) || SGL_CUDA_ARCH >= 900
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      // PPU / NVIDIA SM90+: packed bf16 absolute maximum via inline PTX.
      // Use outer loop over num_vec_elems + inner unrolled loop over kVecSize/2
      // to enable compile-time unrolling.
      uint32_t max_bits = 0;  // two bf16 zeros packed

      for (int32_t i = lane_id; i < num_vec_elems; i += kPerTokenQuantWarpSize) {
        const uint32_t* chunk = reinterpret_cast<const uint32_t*>(smem_buf + i * kVecSize);
#pragma unroll
        for (uint32_t j = 0; j < kVecSize / 2; ++j) {
          uint32_t val_bits = chunk[j];
          uint32_t abs_bits;
          asm("abs.bf16x2 %0, %1;\n" : "=r"(abs_bits) : "r"(val_bits));
          asm("max.bf16x2 %0, %1, %2;\n" : "=r"(max_bits) : "r"(max_bits), "r"(abs_bits));
        }
      }

      // Warp reduce in bf16x2
#pragma unroll
      for (int offset = kPerTokenQuantWarpSize / 2; offset > 0; offset >>= 1) {
        uint32_t tmp = __shfl_xor_sync(0xffffffff, max_bits, offset);
        asm("max.bf16x2 %0, %1, %2;\n" : "=r"(max_bits) : "r"(max_bits), "r"(tmp));
      }

      // Final conversion to fp32
      __nv_bfloat162 max_bf2 = *reinterpret_cast<__nv_bfloat162*>(&max_bits);
      warp_max_result = fmaxf(__bfloat162float(max_bf2.x), __bfloat162float(max_bf2.y));

#endif
    } else {
      // Also use fp32 reduction for bf16 on NVIDIA targets without bf16x2 PTX.
      float max_value = 0.f;
      for (int32_t i = lane_id; i < num_vec_elems; i += kPerTokenQuantWarpSize) {
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          max_value = fmaxf(max_value, fabsf(static_cast<float>(smem_buf[i * kVecSize + j])));
        }
      }
      warp_max_result = device::warp::reduce_max(max_value);
    }

    const float scale = warp_max_result / device::math::FP8_E4M3_MAX;
    if (lane_id == 0) {
      output_s[token_id] = scale;
    }
    const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

    //
    // Pass-2: re-read smem, quantize and write back (mirrors the original
    //         kernel, including the SM89+ PTX fast path)
    //
    DST_DTYPE* token_output = output_q + token_id * hidden_dim;
    for (int i = lane_id; i < num_vec_elems; i += kPerTokenQuantWarpSize) {
      vec_t input_vec;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        input_vec[j] = smem_buf[i * kVecSize + j];
      }

      per_token_quant_store(input_vec, scale_inv, token_output, i);
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// 2. Baseline kernel (1 token / CTA, two-level warp reduction)
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, int kVecSize = 16>
__global__ void per_token_quant_fp8_small_batch_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  const T* token_input = input + token_idx * hidden_dim;
  DST_DTYPE* token_output = output_q + token_idx * hidden_dim;

  float max_value = 0.0f;

  // Use template parameter for vector size
  using vec_t = PerTokenQuantVector<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  // Find max using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  max_value = per_token_quant_block_max(max_value);

  __shared__ float scale;
  if (tid == 0) {
    scale = max_value / device::math::FP8_E4M3_MAX;
    output_s[token_idx] = scale;
  }
  __syncthreads();

  const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  // Quantize using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

    per_token_quant_store(input_vec, scale_inv, token_output, i);
  }
}

// AOT's selection policy: maximise active warps/SM, prefer larger CTAs on ties.
// Deliberately do not gate on runtime capability >= sm89: PPU reports sm80.
template <typename T, bool kUseAIU>
bool launch_per_token_quant_fp8_persistent(
    DLDevice device,
    const T* input,
    fp8_e4m3_t* output_q,
    float* output_s,
    uint32_t hidden_dim,
    uint32_t num_tokens,
    uint32_t sm_count) {
  const size_t stride = (static_cast<size_t>(hidden_dim) * sizeof(T) + 127) / 128 * 128;
  int max_smem = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device.device_id));
  int best_cta = 0, best_score = 0, best_blocks = 0;
  size_t best_smem = 0;
  auto consider = [&](auto kernel, int cta) {
    const size_t smem = stride * cta;
    if (smem > static_cast<size_t>(max_smem)) return;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)) !=
        cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }
    const int blocks = host::runtime::get_blocks_per_sm(kernel, cta * kPerTokenQuantWarpSize, smem);
    const int score = blocks * cta;
    if (score > best_score || (score == best_score && cta > best_cta)) {
      best_cta = cta;
      best_score = score;
      best_blocks = blocks;
      best_smem = smem;
    }
  };
  consider(per_token_quant_fp8_persistent_kernel<T, fp8_e4m3_t, 2, 16, kUseAIU>, 2);
  consider(per_token_quant_fp8_persistent_kernel<T, fp8_e4m3_t, 4, 16, kUseAIU>, 4);
  consider(per_token_quant_fp8_persistent_kernel<T, fp8_e4m3_t, 8, 16, kUseAIU>, 8);
  if (best_score == 0) return false;
  const uint32_t grid = std::min(sm_count * best_blocks, host::div_ceil(num_tokens, static_cast<uint32_t>(best_cta)));
  // Use the selected typed kernel with the JIT stream and launch error checking.
  auto launch = [&](auto kernel) {
    host::LaunchKernel(grid, best_cta * kPerTokenQuantWarpSize, device, best_smem)(
        kernel, input, output_q, output_s, static_cast<int64_t>(hidden_dim), static_cast<int64_t>(num_tokens));
  };
  switch (best_cta) {
    case 2:
      launch(per_token_quant_fp8_persistent_kernel<T, fp8_e4m3_t, 2, 16, kUseAIU>);
      break;
    case 4:
      launch(per_token_quant_fp8_persistent_kernel<T, fp8_e4m3_t, 4, 16, kUseAIU>);
      break;
    case 8:
      launch(per_token_quant_fp8_persistent_kernel<T, fp8_e4m3_t, 8, 16, kUseAIU>);
      break;
  }
  return true;
}

template <typename T, int kVecSize>
void launch_per_token_quant_fp8(
    DLDevice device, const T* input, fp8_e4m3_t* output_q, float* output_s, uint32_t hidden_dim, uint32_t num_tokens) {
  constexpr uint32_t kBlockSize = 256;
  const uint32_t sm_count = host::runtime::get_sm_count(device.device_id);
  const bool use_large_batch = num_tokens >= sm_count * 2 * kPerTokenQuantTokensPerCTA;
  if (use_large_batch) {
#if SGL_CUDA_ARCH >= 800
    if constexpr (kVecSize == 16) {
      if (hidden_dim % 1024 == 0) {
        if (launch_per_token_quant_fp8_persistent<T, true>(
                device, input, output_q, output_s, hidden_dim, num_tokens, sm_count))
          return;
      } else {
        if (launch_per_token_quant_fp8_persistent<T, false>(
                device, input, output_q, output_s, hidden_dim, num_tokens, sm_count))
          return;
      }
    }
#endif
    const uint32_t grid = host::div_ceil(num_tokens, kPerTokenQuantTokensPerCTA);
    // Preserve AOT's fallback smem policy (including its legacy byte threshold).
    const size_t smem = ((static_cast<size_t>(hidden_dim) * sizeof(T) + 31) / 32 * 32) * kPerTokenQuantTokensPerCTA;
    if (hidden_dim < 2048 && smem < 48) {
      host::LaunchKernel(grid, kBlockSize, device, smem)(
          per_token_quant_fp8_warp_kernel<T, fp8_e4m3_t, 8, kVecSize, true>,
          input,
          output_q,
          output_s,
          static_cast<int64_t>(hidden_dim),
          static_cast<int64_t>(num_tokens));
    } else {
      host::LaunchKernel(grid, kBlockSize, device)(
          per_token_quant_fp8_warp_kernel<T, fp8_e4m3_t, 8, kVecSize, false>,
          input,
          output_q,
          output_s,
          static_cast<int64_t>(hidden_dim),
          static_cast<int64_t>(num_tokens));
    }
  } else {
    host::LaunchKernel(num_tokens, kBlockSize, device)(
        per_token_quant_fp8_small_batch_kernel<T, fp8_e4m3_t, kVecSize>,
        input,
        output_q,
        output_s,
        static_cast<int64_t>(hidden_dim),
        static_cast<int64_t>(num_tokens));
  }
}
#else
/** \brief Quantize one token per warp for large token batches. */
template <typename T, int kVecSize>
__global__ void per_token_quant_fp8_warp_kernel(
    const T* __restrict__ input,
    fp8_e4m3_t* __restrict__ output_q,
    float* __restrict__ output_s,
    uint32_t hidden_dim,
    uint32_t num_tokens) {
  using namespace device;
  using input_vec_t = AlignedVector<T, kVecSize>;
  using output_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  const uint32_t warp_id = threadIdx.x / kPerTokenQuantWarpSize;
  const uint32_t lane_id = threadIdx.x % kPerTokenQuantWarpSize;
  const uint32_t token_id = blockIdx.x * kPerTokenQuantTokensPerCTA + warp_id;
  if (token_id >= num_tokens) {
    return;
  }

  const T* token_input = input + token_id * hidden_dim;
  fp8_e4m3_t* token_output = output_q + token_id * hidden_dim;
  const uint32_t num_vecs = hidden_dim / kVecSize;

  float max_value = 0.0f;
  for (uint32_t i = lane_id; i < num_vecs; i += kPerTokenQuantWarpSize) {
    input_vec_t input_vec;
    input_vec.load(token_input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      max_value = math::max(max_value, math::abs(static_cast<float>(input_vec[j])));
    }
  }

  const float scale = warp::reduce_max(max_value) / math::FP8_E4M3_MAX;
  if (lane_id == 0) {
    output_s[token_id] = scale;
  }
  const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;

  for (uint32_t i = lane_id; i < num_vecs; i += kPerTokenQuantWarpSize) {
    input_vec_t input_vec;
    output_vec_t output_vec;
    input_vec.load(token_input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const float value = static_cast<float>(input_vec[j]) * scale_inv;
      output_vec[j] = static_cast<fp8_e4m3_t>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
    }
    output_vec.store(token_output, i);
  }
}

/** \brief Quantize one token per CTA for small token batches. */
template <typename T, int kVecSize>
__global__ void per_token_quant_fp8_cta_kernel(
    const T* __restrict__ input, fp8_e4m3_t* __restrict__ output_q, float* __restrict__ output_s, uint32_t hidden_dim) {
  using namespace device;
  using input_vec_t = AlignedVector<T, kVecSize>;
  using output_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  const uint32_t token_id = blockIdx.x;
  const T* token_input = input + token_id * hidden_dim;
  fp8_e4m3_t* token_output = output_q + token_id * hidden_dim;
  const uint32_t num_vecs = hidden_dim / kVecSize;

  float max_value = 0.0f;
  for (uint32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    input_vec_t input_vec;
    input_vec.load(token_input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      max_value = math::max(max_value, math::abs(static_cast<float>(input_vec[j])));
    }
  }

  __shared__ float reduction_smem[kPerTokenQuantWarpSize];
  __shared__ float scale_smem;
  cta::reduce_max(max_value, reduction_smem);
  __syncthreads();
  if (threadIdx.x == 0) {
    scale_smem = reduction_smem[0] / math::FP8_E4M3_MAX;
    output_s[token_id] = scale_smem;
  }
  __syncthreads();
  const float scale_inv = 1.0f / scale_smem;

  for (uint32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    input_vec_t input_vec;
    output_vec_t output_vec;
    input_vec.load(token_input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const float value = static_cast<float>(input_vec[j]) * scale_inv;
      output_vec[j] = static_cast<fp8_e4m3_t>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
    }
    output_vec.store(token_output, i);
  }
}

template <typename T, int kVecSize>
void launch_per_token_quant_fp8(
    DLDevice device, const T* input, fp8_e4m3_t* output_q, float* output_s, uint32_t hidden_dim, uint32_t num_tokens) {
  constexpr uint32_t kBlockSize = 256;
  const uint32_t sm_count = host::runtime::get_sm_count(device.device_id);
  const bool use_warp_kernel = num_tokens >= sm_count * 2 * kPerTokenQuantTokensPerCTA;
  if (use_warp_kernel) {
    const uint32_t grid = host::div_ceil(num_tokens, kPerTokenQuantTokensPerCTA);
    host::LaunchKernel(grid, kBlockSize, device)(
        per_token_quant_fp8_warp_kernel<T, kVecSize>, input, output_q, output_s, hidden_dim, num_tokens);
  } else {
    host::LaunchKernel(num_tokens, kBlockSize, device)(
        per_token_quant_fp8_cta_kernel<T, kVecSize>, input, output_q, output_s, hidden_dim);
  }
}

#endif

/** \brief Validate and launch dynamic per-token FP8 E4M3 quantization. */
template <typename T>
void per_token_quant_fp8(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
  using namespace host;
  auto M = SymbolicSize{"num_tokens"};
  auto MOutput = SymbolicSize{"output_num_tokens"};
  auto K = SymbolicSize{"hidden_dim"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, K}).with_dtype<T>().with_device(device).verify(input);
  TensorMatcher({MOutput, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);
  TensorMatcher({MOutput, 1}).with_dtype<float>().with_device(device).verify(output_s);

  CHECK_HOST(M.unwrap() > 0) << "per_token_quant_fp8: num_tokens must be positive";
  CHECK_HOST(MOutput.unwrap() >= M.unwrap())
      << "per_token_quant_fp8: output buffers must have at least " << M.unwrap() << " rows, got " << MOutput.unwrap();
  CHECK_HOST(K.unwrap() > 0 && K.unwrap() % 4 == 0)
      << "per_token_quant_fp8: hidden_dim must be positive and divisible by 4, got " << K.unwrap();
  CHECK_HOST(M.unwrap() <= UINT32_MAX && K.unwrap() <= UINT32_MAX)
      << "per_token_quant_fp8: dimensions exceed uint32 indexing";

  const uint32_t num_tokens = static_cast<uint32_t>(M.unwrap());
  const uint32_t hidden_dim = static_cast<uint32_t>(K.unwrap());
#ifndef USE_ROCM
  constexpr uint32_t kMaxVecSize = 16;
#else
  constexpr uint32_t kMaxVecSize = 16 / sizeof(T);
#endif
  if (hidden_dim % kMaxVecSize == 0) {
    launch_per_token_quant_fp8<T, kMaxVecSize>(
        device.unwrap(),
        static_cast<const T*>(input.data_ptr()),
        static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
#ifndef USE_ROCM
  } else if (hidden_dim % 8 == 0) {
    launch_per_token_quant_fp8<T, 8>(
        device.unwrap(),
        static_cast<const T*>(input.data_ptr()),
        static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
#endif
  } else {
    launch_per_token_quant_fp8<T, 4>(
        device.unwrap(),
        static_cast<const T*>(input.data_ptr()),
        static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
  }
}

}  // namespace sglang
