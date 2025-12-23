/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*
 * AWQ GEMV kernel for W4A16 quantization.
 * Modified from NVIDIA TensorRT-LLM and Nunchaku project.
 *
 * @article{lin2023awq,
 *   title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
 *   author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
 *   journal={arXiv},
 *   year={2023}
 * }
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cassert>

#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MEM_ACCESS_SIZE 128

template <typename T, int N>
struct packed_as {
  using type = T;
};

template <>
struct packed_as<half, 2> {
  using type = half2;
};

template <>
struct packed_as<__nv_bfloat16, 2> {
  using type = __nv_bfloat162;
};

template <typename T>
__device__ __forceinline__ typename packed_as<T, 2>::type half2half2(T x);

template <>
__device__ __forceinline__ half2 half2half2<half>(half x) {
  return __half2half2(x);
}

template <>
__device__ __forceinline__ __nv_bfloat162 half2half2<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162bfloat162(x);
}

template <typename T>
__device__ __forceinline__ float2 half22float2(T val);

template <>
__device__ __forceinline__ float2 half22float2<half2>(half2 val) {
  return __half22float2(val);
}

template <>
__device__ __forceinline__ float2 half22float2<__nv_bfloat162>(__nv_bfloat162 val) {
  return __bfloat1622float2(val);
}

// Dequantize int4 to fp16x2
__forceinline__ __device__ void dequantize_s4_to_fp16x2(half2 const& source, uint4* result) {
  uint32_t* h = reinterpret_cast<uint32_t*>(result);
  uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

  const uint32_t top_i4s = i4s >> 8;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  static constexpr uint32_t NEG_64 = 0xd400d400;

  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
}

__forceinline__ __device__ void dequantize_s4_to_fp16x2(__nv_bfloat162 const& source, uint4* result) {
  uint32_t* h = reinterpret_cast<uint32_t*>(result);
  uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(i4s >> 4), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(i4s >> 8), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(i4s >> 12), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));

  static constexpr uint32_t BF16_BIAS = 0xC300C300;
  static constexpr uint32_t BF16_ONE = 0x3F803F80;

  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[0]) : "r"(h[0]), "r"(BF16_ONE), "r"(BF16_BIAS));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(BF16_ONE), "r"(BF16_BIAS));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[2]) : "r"(h[2]), "r"(BF16_ONE), "r"(BF16_BIAS));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(BF16_ONE), "r"(BF16_BIAS));
}

// Warp reduce sum
template <typename float_t, int Num, int WarpSize>
__device__ __forceinline__ static void warp_reduce(float_t* psum, float (*out_smem)[Num * 4]) {
  float fpsum[Num];
#pragma unroll
  for (int i = 0; i < Num; ++i) {
    fpsum[i] = static_cast<float>(psum[i]);
  }

#pragma unroll
  for (int i = 0; i < Num; ++i) {
    fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 16);
    fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 8);
    fpsum[i] += __shfl_xor_sync(~0, fpsum[i], 1);
  }
  __syncthreads();
  int warp = threadIdx.x / WarpSize, lane = threadIdx.x % WarpSize;
  if (lane == 0 || lane == 2 || lane == 4 || lane == 6) {
#pragma unroll
    for (int i = 0; i < Num; ++i) {
      out_smem[warp][i * 4 + lane / 2] = fpsum[i];
    }
  }
  __syncthreads();
}

template <typename half_t, int NPerBlock, int Batch, int BlockSize, int GroupSize>
__global__ void gemv_awq_kernel(
    const half_t* inputs,
    const uint32_t* weight,
    const half_t* scales,
    const half_t* zeros,
    half_t* outputs,
    const int IC,
    const int OC) {
  using half2_t = typename packed_as<half_t, 2>::type;
  using accum_t = float;

  const int kStride = 64;
  const int kElemsPerThread = MEM_ACCESS_SIZE / 4;
  const int kThreadsNumPerTile = kStride / kElemsPerThread;

  static constexpr int kShuffleBasicTile = 2;
  static constexpr int kShuffleContinous = 4;
  static constexpr int kShuffleStrided = 4;

  constexpr int Num = NPerBlock * Batch;
  constexpr int kInterleave = 4;

  alignas(16) half_t local_inputs[kElemsPerThread];
  alignas(16) uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
  alignas(16) half_t half_weight_buffer[kElemsPerThread];
  alignas(16) half_t dequantized_weight[kElemsPerThread * NPerBlock];
  alignas(16) half_t local_scale[NPerBlock];
  alignas(16) half_t local_scaled_zeros[NPerBlock];

  accum_t psum[Num];
  for (int i = 0; i < Num; ++i)
    psum[i] = static_cast<accum_t>(0.f);

  __shared__ float out_smem[BlockSize / WARP_SIZE * 2][Num * kInterleave];

  const int blk_row_offset = blockIdx.x * NPerBlock * kInterleave;
  const int thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;
  const int act_k_offset =
      threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
  const int group_offset = act_k_offset / GroupSize;

  const uint32_t* blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;
  const half_t* scale_ptr = scales + blk_row_offset + thd_row_offset + group_offset * OC;
  const half_t* zeros_ptr = zeros + blk_row_offset + thd_row_offset + group_offset * OC;
  const half_t* inputs_ptr = inputs + act_k_offset;

  const int act_forward_step = BlockSize * kElemsPerThread / kInterleave;
  const int scale_forward_step = act_forward_step / GroupSize * OC;

  for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread) {
#pragma unroll
    for (int idx = 0; idx < NPerBlock; ++idx) {
      *((float4*)(local_qweights)) = *((float4*)(blk_weight_ptr + (idx * kInterleave * IC + kk) / PACK_FACTOR));
      local_scale[idx] = *(scale_ptr + idx * kInterleave);
      local_scaled_zeros[idx] = *(zeros_ptr + idx * kInterleave);

#pragma unroll
      for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i) {
        dequantize_s4_to_fp16x2(
            *reinterpret_cast<half2_t*>(local_qweights + i),
            reinterpret_cast<uint4*>(half_weight_buffer + i * PACK_FACTOR));
      }

#pragma unroll
      for (int i = 0; i < kShuffleContinous; ++i) {
#pragma unroll
        for (int j = 0; j < kShuffleStrided; ++j) {
          half2_t w = *reinterpret_cast<half2_t*>(half_weight_buffer + (i + j * kShuffleContinous) * kShuffleBasicTile);
          w = __hfma2(w, half2half2(local_scale[idx]), half2half2(local_scaled_zeros[idx]));
          dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 0) * NPerBlock + idx] = w.x;
          dequantized_weight[((i * kShuffleStrided + j) * kShuffleBasicTile + 1) * NPerBlock + idx] = w.y;
        }
      }
    }

#pragma unroll
    for (int batch_idx = 0; batch_idx < Batch; ++batch_idx) {
      const half_t* local_inputs_ptr = inputs_ptr + batch_idx * IC;
#pragma unroll
      for (int idx = 0; idx < kElemsPerThread / 8; ++idx) {
        *((float4*)(local_inputs + idx * 8)) = *((float4*)(local_inputs_ptr + idx * 8));
      }

#pragma unroll
      for (int x = 0; x < NPerBlock / 2; ++x) {
#pragma unroll
        for (int y = 0; y < kElemsPerThread; ++y) {
          float2 prod = half22float2(__hmul2(
              *reinterpret_cast<half2_t*>(dequantized_weight + y * NPerBlock + x * 2), half2half2(local_inputs[y])));
          *reinterpret_cast<float2*>(psum + batch_idx * NPerBlock + x * 2) = make_float2(
              prod.x + psum[batch_idx * NPerBlock + x * 2], prod.y + psum[batch_idx * NPerBlock + x * 2 + 1]);
        }
      }
    }
    inputs_ptr += act_forward_step;
    scale_ptr += scale_forward_step;
    zeros_ptr += scale_forward_step;
  }

  warp_reduce<accum_t, Num, WARP_SIZE>(psum, out_smem);

  for (int i = threadIdx.x; i < Num * kInterleave; i += BlockSize) {
    int batch_idx = i / (NPerBlock * kInterleave);
    int oc_idx = i % (NPerBlock * kInterleave);
    float acc = 0.f;
    for (int j = 0; j < BlockSize / WARP_SIZE; ++j) {
      acc += out_smem[j][i];
    }
    outputs[batch_idx * OC + blk_row_offset + oc_idx] = static_cast<half_t>(acc);
  }
}

template <typename half_t>
void launch_gemv_awq(
    const half_t* in_feats,
    const uint32_t* kernel,
    const half_t* scales,
    const half_t* zeros,
    half_t* out_feats,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  static constexpr int N_PER_BLOCK = 2;
  static constexpr int K_INTERLEAVE = 4;
  static constexpr int BLOCK_SIZE = 256;
  static constexpr int GROUP_SIZE = 64;

  dim3 num_blocks(n / N_PER_BLOCK / K_INTERLEAVE);
  dim3 num_threads(BLOCK_SIZE);

  // Dispatch based on batch size
  switch (m) {
    case 1:
      gemv_awq_kernel<half_t, N_PER_BLOCK, 1, BLOCK_SIZE, GROUP_SIZE>
          <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scales, zeros, out_feats, k, n);
      break;
    case 2:
      gemv_awq_kernel<half_t, N_PER_BLOCK, 2, BLOCK_SIZE, GROUP_SIZE>
          <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scales, zeros, out_feats, k, n);
      break;
    case 3:
      gemv_awq_kernel<half_t, N_PER_BLOCK, 3, BLOCK_SIZE, GROUP_SIZE>
          <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scales, zeros, out_feats, k, n);
      break;
    case 4:
      gemv_awq_kernel<half_t, N_PER_BLOCK, 4, BLOCK_SIZE, GROUP_SIZE>
          <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scales, zeros, out_feats, k, n);
      break;
    case 5:
      gemv_awq_kernel<half_t, N_PER_BLOCK, 5, BLOCK_SIZE, GROUP_SIZE>
          <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scales, zeros, out_feats, k, n);
      break;
    case 6:
      gemv_awq_kernel<half_t, N_PER_BLOCK, 6, BLOCK_SIZE, GROUP_SIZE>
          <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scales, zeros, out_feats, k, n);
      break;
    case 7:
      gemv_awq_kernel<half_t, N_PER_BLOCK, 7, BLOCK_SIZE, GROUP_SIZE>
          <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scales, zeros, out_feats, k, n);
      break;
    case 8:
      gemv_awq_kernel<half_t, N_PER_BLOCK, 8, BLOCK_SIZE, GROUP_SIZE>
          <<<num_blocks, num_threads, 0, stream>>>(in_feats, kernel, scales, zeros, out_feats, k, n);
      break;
    default:
      TORCH_CHECK(false, "Unsupported batch size: ", m, ", must be 1-8");
  }
}

torch::Tensor svdq_gemv_awq(
    torch::Tensor in_feats,
    torch::Tensor kernel,
    torch::Tensor scaling_factors,
    torch::Tensor zeros,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t group_size) {
  TORCH_CHECK(in_feats.is_cuda(), "in_feats must be a CUDA tensor");
  TORCH_CHECK(kernel.is_cuda(), "kernel must be a CUDA tensor");
  TORCH_CHECK(scaling_factors.is_cuda(), "scaling_factors must be a CUDA tensor");
  TORCH_CHECK(zeros.is_cuda(), "zeros must be a CUDA tensor");

  TORCH_CHECK(m > 0 && m <= 8, "m must be between 1 and 8, got ", m);
  TORCH_CHECK(group_size == 64, "group_size must be 64, got ", group_size);

  // Create output tensor
  auto output_shape = in_feats.sizes().vec();
  output_shape.back() = n;
  auto out_feats = torch::empty(output_shape, in_feats.options());

  auto stream = at::cuda::getCurrentCUDAStream();

  if (scaling_factors.scalar_type() == at::ScalarType::Half) {
    launch_gemv_awq<half>(
        reinterpret_cast<const half*>(in_feats.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(kernel.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scaling_factors.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zeros.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_feats.data_ptr<at::Half>()),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        stream);
  } else if (scaling_factors.scalar_type() == at::ScalarType::BFloat16) {
    launch_gemv_awq<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(in_feats.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint32_t*>(kernel.data_ptr<int32_t>()),
        reinterpret_cast<const __nv_bfloat16*>(scaling_factors.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(zeros.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out_feats.data_ptr<at::BFloat16>()),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        stream);
  } else {
    TORCH_CHECK(false, "Unsupported dtype: ", scaling_factors.scalar_type());
  }

  return out_feats;
}
