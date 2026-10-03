#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>
#include <type_traits>

#include "utils.h"

static constexpr int kWarpSize = 32;
static constexpr int DEFAULT_SHARED_MEM_THRESHOLD_KB = 48;  // Default shared memory quota in KB

#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
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
__global__ void per_token_quant_fp8_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int warp_id = threadIdx.x / kWarpSize;        // 0‑7  (8 warps)
  const int lane_id = threadIdx.x & (kWarpSize - 1);  // 0‑31
  const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
  if (token_id >= num_tokens) return;

  // Global tensors for this token
  const T* token_input = input + token_id * hidden_dim;
  DST_DTYPE* token_output = output_q + token_id * hidden_dim;
  float* token_scale = output_s + token_id;

  extern __shared__ char smem_buffer[];
  const int smem_padding = 32;  // Pad to bank boundary (32 banks * 4 bytes = 128 bytes)
  const int warp_smem_stride = (hidden_dim * sizeof(T) + smem_padding - 1) / smem_padding * smem_padding;
  const int warp_smem_offset = warp_id * warp_smem_stride;
  T* shared_input = reinterpret_cast<T*>(smem_buffer + warp_smem_offset);

  //
  // Pass-1: Load data and compute max_value
  //
  float max_value = 0.f;
  using vec_t = flashinfer::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
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

  float warp_max = warpReduceMax(max_value);

  // NOTE: one CTA has multiple warps (each warp handles one token), so `scale`
  // must be per-warp/per-thread (register) instead of a single shared variable.
  const float scale = warp_max / FP8_E4M3_MAX;
  // Broadcast scale
  if (lane_id == 0) {
    token_scale[0] = scale;
  }
  const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  //
  // Pass-2: Quantize and write back
  //
  for (int i = lane_id; i < num_vec_elems; i += kWarpSize) {
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

    DST_DTYPE output_arr[kVecSize];
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    // SM89+ path: use cvt.rn.satfinite.e4m3x2 for packed 2-element conversion
    // satfinite automatically clamps, no manual fmaxf/fminf needed
    if constexpr (std::is_same_v<T, __nv_half>) {
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
        val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
        output_arr[kVecSize - 1] = static_cast<DST_DTYPE>(val);
      }
    } else {
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
        val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
        output_arr[kVecSize - 1] = static_cast<DST_DTYPE>(val);
      }
    }
#else
    // Original path for SM < 89 and ROCm
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(input_vec[j]) * scale_inv;
      val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_DTYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
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
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x & (kWarpSize - 1);
  const int64_t first_token = static_cast<int64_t>(blockIdx.x) * kTokensPerCTA + warp_id;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kTokensPerCTA;

  // 128B aligned per-warp stride (cp.async friendly + bank-conflict avoidance)
  extern __shared__ char smem_buffer[];
  const int warp_smem_stride = (hidden_dim * static_cast<int>(sizeof(T)) + 127) / 128 * 128;
  T* smem_buf = reinterpret_cast<T*>(smem_buffer + warp_id * warp_smem_stride);

  // Number of 16B copies per kVecSize-element chunk (compile-time constant)
  constexpr int kCopiesPerVec = kVecSize * static_cast<int>(sizeof(T)) / 16;

  using vec_t = flashinfer::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  // Main persistent loop: each iteration handles one token per warp
  for (int64_t token_id = first_token; token_id < num_tokens; token_id += stride) {
    // Step 1: async copy current token to smem
    // Use outer loop over num_vec_elems + inner unrolled loop (kCopiesPerVec is
    // compile-time: 2 for fp16/bf16, 4 for fp32) to enable full unrolling.
    const T* src = input + token_id * hidden_dim;
    // hidden_dim x fp16 per warp
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    if constexpr (sizeof(T) == 2 and useAIU) {
      constexpr int AIU_CHUNK_SLISE_4 = 16 * 64;
      for (int aiu_cp_offset = 0; aiu_cp_offset < hidden_dim; aiu_cp_offset += AIU_CHUNK_SLISE_4) {
        load_128b_aiu<T, false>(smem_buf + aiu_cp_offset, src + aiu_cp_offset, 64, 16, 64, 16, 0, 0);
      }
    } else
#endif
    {
      for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
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

#if !defined(USE_ROCM)
    if constexpr (std::is_same_v<T, __nv_half>) {
      // fp16 path: __habs2 + __hmax2 (SM53+)
      // Use outer loop over num_vec_elems + inner unrolled loop over kVecSize/2
      // to enable compile-time unrolling (kVecSize is a template constant).
      __half2 max_h2 = __halves2half2(__float2half(0.0f), __float2half(0.0f));

      for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
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
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        __half2 tmp = __shfl_xor_sync(0xffffffff, max_h2, offset);
        max_h2 = __hmax2(max_h2, tmp);
      }

      // Final conversion: reduce the 2 lanes of half2 to a single fp32 max
      float2 max_f2 = __half22float2(max_h2);
      warp_max_result = fmaxf(max_f2.x, max_f2.y);

    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      // bf16 path: abs.bf16x2 + max.bf16x2 via inline PTX (SM80+)
      // Use outer loop over num_vec_elems + inner unrolled loop over kVecSize/2
      // to enable compile-time unrolling.
      uint32_t max_bits = 0;  // two bf16 zeros packed

      for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
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
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        uint32_t tmp = __shfl_xor_sync(0xffffffff, max_bits, offset);
        asm("max.bf16x2 %0, %1, %2;\n" : "=r"(max_bits) : "r"(max_bits), "r"(tmp));
      }

      // Final conversion to fp32
      __nv_bfloat162 max_bf2 = *reinterpret_cast<__nv_bfloat162*>(&max_bits);
      warp_max_result = fmaxf(__bfloat162float(max_bf2.x), __bfloat162float(max_bf2.y));

    } else {
      // fp32 input: keep original fp32 path
      float max_value = 0.f;
      for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          max_value = fmaxf(max_value, fabsf(smem_buf[i * kVecSize + j]));
        }
      }
      warp_max_result = warpReduceMax(max_value);
    }
#else
    // ROCm: keep original fp32 path
    {
      float max_value = 0.f;
      for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
#pragma unroll
        for (uint32_t j = 0; j < kVecSize; ++j) {
          max_value = fmaxf(max_value, fabsf(static_cast<float>(smem_buf[i * kVecSize + j])));
        }
      }
      warp_max_result = warpReduceMax(max_value);
    }
#endif

    const float scale = warp_max_result / FP8_E4M3_MAX;
    if (lane_id == 0) {
      output_s[token_id] = scale;
    }
    const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

    //
    // Pass-2: re-read smem, quantize and write back (mirrors the original
    //         kernel, including the SM89+ PTX fast path)
    //
    DST_DTYPE* token_output = output_q + token_id * hidden_dim;
    for (int i = lane_id; i < num_vec_elems; i += kWarpSize) {
      vec_t input_vec;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        input_vec[j] = smem_buf[i * kVecSize + j];
      }

      DST_DTYPE output_arr[kVecSize];
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
      // SM89+ path: use cvt.rn.satfinite.e4m3x2 for packed 2-element conversion
      if constexpr (std::is_same_v<T, __nv_half>) {
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
          uint32_t input_f16x2;
          asm("mov.b32 %0, {%1, %2};\n"
              : "=r"(input_f16x2)
              : "h"(reinterpret_cast<const uint16_t*>(&input_vec[0])[j * 2]),
                "h"(reinterpret_cast<const uint16_t*>(&input_vec[0])[j * 2 + 1]));

          uint32_t scaled_f16x2;
          asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(scaled_f16x2) : "r"(input_f16x2), "r"(scale_inv_f16x2));

          uint16_t packed_fp8x2;
          asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(packed_fp8x2) : "r"(scaled_f16x2));

          output_u16[j] = packed_fp8x2;
        }

        if constexpr (kVecSize % 2 != 0) {
          float val = static_cast<float>(input_vec[kVecSize - 1]) * scale_inv;
          val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
          output_arr[kVecSize - 1] = static_cast<DST_DTYPE>(val);
        }
      } else {
        // bf16/fp32 input: use cvt.rn.satfinite.e4m3x2.f32
        uint16_t* output_u16 = reinterpret_cast<uint16_t*>(output_arr);
#pragma unroll
        for (uint32_t j = 0; j < kVecSize / 2; ++j) {
          float val_a = static_cast<float>(input_vec[j * 2]) * scale_inv;
          float val_b = static_cast<float>(input_vec[j * 2 + 1]) * scale_inv;

          uint16_t packed_fp8x2;
          asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(packed_fp8x2) : "f"(val_b), "f"(val_a));

          output_u16[j] = packed_fp8x2;
        }

        if constexpr (kVecSize % 2 != 0) {
          float val = static_cast<float>(input_vec[kVecSize - 1]) * scale_inv;
          val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
          output_arr[kVecSize - 1] = static_cast<DST_DTYPE>(val);
        }
      }
#else
      // Original path for SM < 89 and ROCm
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        float val = static_cast<float>(input_vec[j]) * scale_inv;
        val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
        output_arr[j] = static_cast<DST_DTYPE>(val);
#else
        output_arr[j] = c10::Float8_e4m3fnuz(
            __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
            c10::Float8_e4m3fnuz::from_bits());
#endif
      }
#endif
      if constexpr (kVecSize == 16) {
        *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
      } else {
        for (int k = 0; k < kVecSize; ++k) {
          token_output[i * kVecSize + k] = output_arr[k];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2.  Baseline kernel (1 token / CTA, CUB block reduce)
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
  using vec_t = flashinfer::vec_t<T, kVecSize>;
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

  max_value = blockReduceMax(max_value);

  __shared__ float scale;
  if (tid == 0) {
    scale = max_value / FP8_E4M3_MAX;
    output_s[token_idx] = scale;
  }
  __syncthreads();

  const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  // Quantize using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

    DST_DTYPE output_arr[kVecSize];
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    // SM89+ path: use cvt.rn.satfinite.e4m3x2 for packed 2-element conversion
    if constexpr (std::is_same_v<T, __nv_half>) {
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
        uint32_t input_f16x2;
        asm("mov.b32 %0, {%1, %2};\n"
            : "=r"(input_f16x2)
            : "h"(reinterpret_cast<const uint16_t*>(&input_vec[0])[j * 2]),
              "h"(reinterpret_cast<const uint16_t*>(&input_vec[0])[j * 2 + 1]));

        uint32_t scaled_f16x2;
        asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(scaled_f16x2) : "r"(input_f16x2), "r"(scale_inv_f16x2));

        uint16_t packed_fp8x2;
        asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(packed_fp8x2) : "r"(scaled_f16x2));

        output_u16[j] = packed_fp8x2;
      }

      if constexpr (kVecSize % 2 != 0) {
        float val = static_cast<float>(input_vec[kVecSize - 1]) * scale_inv;
        val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
        output_arr[kVecSize - 1] = static_cast<DST_DTYPE>(val);
      }
    } else {
      // bf16/fp32 input: use cvt.rn.satfinite.e4m3x2.f32
      uint16_t* output_u16 = reinterpret_cast<uint16_t*>(output_arr);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize / 2; ++j) {
        float val_a = static_cast<float>(input_vec[j * 2]) * scale_inv;
        float val_b = static_cast<float>(input_vec[j * 2 + 1]) * scale_inv;

        uint16_t packed_fp8x2;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(packed_fp8x2) : "f"(val_b), "f"(val_a));

        output_u16[j] = packed_fp8x2;
      }

      if constexpr (kVecSize % 2 != 0) {
        float val = static_cast<float>(input_vec[kVecSize - 1]) * scale_inv;
        val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
        output_arr[kVecSize - 1] = static_cast<DST_DTYPE>(val);
      }
    }
#else
    // Original path for SM < 89 and ROCm
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_DTYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
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
}

template <bool USE_SMEM, typename scalar_t, int TOKENS_PER_CTA>
static inline void launch_per_token_quant_fp8_warp_kernel(
    const dim3& grid,
    const dim3& block,
    size_t dynamicSmemSz,
    cudaStream_t stream,
    bool use_vec16,
    bool use_vec8,
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const size_t smem_size = USE_SMEM ? dynamicSmemSz : 0;

  if (use_vec16) {
    per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 16, USE_SMEM>
        <<<grid, block, smem_size, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
  } else if (use_vec8) {
    per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 8, USE_SMEM>
        <<<grid, block, smem_size, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
  } else {
    per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 4, USE_SMEM>
        <<<grid, block, smem_size, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
  }
}

void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  const auto input_sizes = input.sizes();
  const int64_t num_tokens = input_sizes[0];
  const int64_t hidden_dim = input_sizes[1];
  TORCH_CHECK(hidden_dim % 4 == 0, "Hidden dimension must be divisible by 4, but got ", hidden_dim);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int TOKENS_PER_CTA = 8;
  const bool use_warp_kernel = (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);
  const bool use_vec16 = (hidden_dim % 16 == 0);
  const bool use_vec8 = (hidden_dim % 8 == 0);

  const int sizeof_T = input.scalar_type() == torch::kFloat16 ? 2 : (input.scalar_type() == torch::kBFloat16 ? 2 : 4);
  const int smem_padding = 32;  // Pad to bank boundary to avoid conflicts
  const int warp_smem_stride = (hidden_dim * sizeof_T + smem_padding - 1) / smem_padding * smem_padding;
  const size_t dynamicSmemSz = warp_smem_stride * TOKENS_PER_CTA;

  bool use_smem = (hidden_dim < 2048);

  if (dynamicSmemSz >= DEFAULT_SHARED_MEM_THRESHOLD_KB) {
    use_smem = false;  // Disable shared memory if >= 48KB to avoid allocation failures
  }

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    // -------- persistent (cp.async double-buffered) -----------------------
    // Preconditions:
    //   1. enough work for warp_kernel grid sizing assumption
    //   2. hidden_dim is 16-element vec aligned (kVecSize=16 path)
    //   3. hidden_dim * sizeof(T) is 16B aligned (cp.async requirement)
    //   4. smem fits the device opt-in budget
    //   5. occupancy >= 1 block / SM
    // Dynamic kTokensPerCTA selection: evaluate candidates and pick the one
    // that maximises (blocks_per_sm * cta_tokens), i.e. active warps per SM.
    bool use_persistent = false;
    if (use_warp_kernel && use_vec16 && (hidden_dim * static_cast<int64_t>(sizeof(scalar_t)) % 16 == 0)) {
      const int warp_smem_stride_p = (hidden_dim * static_cast<int>(sizeof(scalar_t)) + 127) / 128 * 128;
      const int max_smem = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;

      // Candidate kTokensPerCTA values (from small smem to large)
      constexpr int kCandidates[] = {2, 4, 8};
      constexpr int kNumCandidates = 3;

      int best_cta = 0;
      int best_score = 0;  // score = blocks_per_sm * cta_tokens (proxy for active warps/SM)
      int best_blocks_per_sm = 0;
      size_t best_smem = 0;
      const bool useAIU = (hidden_dim % 1024 == 0);

      for (int ci = 0; ci < kNumCandidates; ++ci) {
        const int cta = kCandidates[ci];
        const int threads = cta * kWarpSize;
        const size_t smem = static_cast<size_t>(warp_smem_stride_p) * cta;  // single buffer
        if (smem > static_cast<size_t>(max_smem)) continue;

        // Get the kernel function pointer for this candidate
        void* kfunc = nullptr;
        if (useAIU) {
          switch (cta) {
            case 2:
              kfunc =
                  reinterpret_cast<void*>(per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 2, 16, true>);
              break;
            case 4:
              kfunc =
                  reinterpret_cast<void*>(per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 4, 16, true>);
              break;
            case 8:
              kfunc =
                  reinterpret_cast<void*>(per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 8, 16, true>);
              break;
            default:
              continue;
          }
        } else {
          switch (cta) {
            case 2:
              kfunc =
                  reinterpret_cast<void*>(per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 2, 16, false>);
              break;
            case 4:
              kfunc =
                  reinterpret_cast<void*>(per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 4, 16, false>);
              break;
            case 8:
              kfunc =
                  reinterpret_cast<void*>(per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 8, 16, false>);
              break;
            default:
              continue;
          }
        }

        cudaError_t attr_err =
            cudaFuncSetAttribute(kfunc, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (attr_err != cudaSuccess) {
          // Clear error state so subsequent CUDA calls are not affected
          (void)cudaGetLastError();
          continue;
        }

        int bpsm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, kfunc, threads, smem);
        if (bpsm <= 0) continue;

        const int score = bpsm * cta;  // active warps per SM
        if (score > best_score || (score == best_score && cta > best_cta)) {
          best_score = score;
          best_cta = cta;
          best_blocks_per_sm = bpsm;
          best_smem = smem;
        }
      }

      if (best_score > 0) {
        int max_useful_grid = static_cast<int>((num_tokens + best_cta - 1) / best_cta);
        int grid_size = sm_count * best_blocks_per_sm;
        if (grid_size > max_useful_grid) grid_size = max_useful_grid;

        if (useAIU) {
          switch (best_cta) {
            case 2:
              per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 2, 16, true>
                  <<<grid_size, 2 * kWarpSize, best_smem, stream>>>(
                      static_cast<const scalar_t*>(input.data_ptr()),
                      static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                      static_cast<float*>(output_s.data_ptr()),
                      hidden_dim,
                      num_tokens);
              break;
            case 4:
              per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 4, 16, true>
                  <<<grid_size, 4 * kWarpSize, best_smem, stream>>>(
                      static_cast<const scalar_t*>(input.data_ptr()),
                      static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                      static_cast<float*>(output_s.data_ptr()),
                      hidden_dim,
                      num_tokens);
              break;
            case 8:
              per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 8, 16, true>
                  <<<grid_size, 8 * kWarpSize, best_smem, stream>>>(
                      static_cast<const scalar_t*>(input.data_ptr()),
                      static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                      static_cast<float*>(output_s.data_ptr()),
                      hidden_dim,
                      num_tokens);
              break;
          }
        } else {
          switch (best_cta) {
            case 2:
              per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 2, 16, false>
                  <<<grid_size, 2 * kWarpSize, best_smem, stream>>>(
                      static_cast<const scalar_t*>(input.data_ptr()),
                      static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                      static_cast<float*>(output_s.data_ptr()),
                      hidden_dim,
                      num_tokens);
              break;
            case 4:
              per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 4, 16, false>
                  <<<grid_size, 4 * kWarpSize, best_smem, stream>>>(
                      static_cast<const scalar_t*>(input.data_ptr()),
                      static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                      static_cast<float*>(output_s.data_ptr()),
                      hidden_dim,
                      num_tokens);
              break;
            case 8:
              per_token_quant_fp8_persistent_kernel<scalar_t, __nv_fp8_e4m3, 8, 16, false>
                  <<<grid_size, 8 * kWarpSize, best_smem, stream>>>(
                      static_cast<const scalar_t*>(input.data_ptr()),
                      static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                      static_cast<float*>(output_s.data_ptr()),
                      hidden_dim,
                      num_tokens);
              break;
          }
        }

        use_persistent = true;
      }
    }

    if (use_persistent) {
      return true;
    }

    if (use_warp_kernel) {
      // -------- warp‑local ---------------------------------------------------
      constexpr int THREADS = TOKENS_PER_CTA * kWarpSize;
      dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
      dim3 block(THREADS);

      if (use_smem) {
        launch_per_token_quant_fp8_warp_kernel</*USE_SMEM=*/true, scalar_t, TOKENS_PER_CTA>(
            grid, block, dynamicSmemSz, stream, use_vec16, use_vec8, input, output_q, output_s, hidden_dim, num_tokens);
      } else {
        launch_per_token_quant_fp8_warp_kernel</*USE_SMEM=*/false, scalar_t, TOKENS_PER_CTA>(
            grid, block, dynamicSmemSz, stream, use_vec16, use_vec8, input, output_q, output_s, hidden_dim, num_tokens);
      }
    } else {
      // -------- baseline -----------------------------------------------------
      constexpr int THREADS = 256;
      dim3 grid(num_tokens);
      dim3 block(THREADS);

      if (use_vec16) {
        per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 16><<<grid, block, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
      } else if (use_vec8) {
        per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 8><<<grid, block, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
      } else {
        per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 4><<<grid, block, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
      }
    }
    return true;
  });
}
