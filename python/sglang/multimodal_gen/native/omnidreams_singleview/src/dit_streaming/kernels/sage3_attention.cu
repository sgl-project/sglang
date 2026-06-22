// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "attention.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_fp8.h>
#include <cutlass/numeric_conversion.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

void scaled_fp4_quant(torch::Tensor const& input,
                      torch::Tensor const& output,
                      torch::Tensor const& output_sf,
                      int tensor_layout);
void scaled_fp4_quant_permute(torch::Tensor const& input,
                              torch::Tensor const& output,
                              torch::Tensor const& output_sf,
                              int tensor_layout);
void scaled_fp4_quant_trans(torch::Tensor const& input,
                            torch::Tensor const& output,
                            torch::Tensor const& output_sf,
                            int tensor_layout);

std::vector<at::Tensor> mha_fwd(at::Tensor& q,
                                const at::Tensor& k,
                                const at::Tensor& v,
                                const at::Tensor& sfq,
                                const at::Tensor& sfk,
                                const at::Tensor& sfv,
                                const at::Tensor& delta_s,
                                int unpadded_k,
                                c10::optional<at::Tensor>& out_,
                                const float softmax_scale,
                                bool is_causal,
                                bool per_block_mean,
                                bool is_bf16);

namespace omnidreams_singleview {
namespace {

struct CurrentStreamScope {
  explicit CurrentStreamScope(cudaStream_t stream, int device)
      : previous_(c10::cuda::getCurrentCUDAStream(device)),
        external_(c10::cuda::getStreamFromExternal(stream, device)) {
    c10::cuda::setCurrentCUDAStream(external_);
  }

  ~CurrentStreamScope() {
    c10::cuda::setCurrentCUDAStream(previous_);
  }

  c10::cuda::CUDAStream previous_;
  c10::cuda::CUDAStream external_;
};

int round_up(int x, int multiple) {
  return ((x + multiple - 1) / multiple) * multiple;
}

bool env_flag_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v && v[0] && v[0] != '0';
}

constexpr int kFp4EltsPerThread = 16;
constexpr int kSage3QuantBlockTokens = 128;

inline __device__ float bf16_to_float(cutlass::bfloat16_t value) {
  return float(value);
}

// Convert 4 float2 values into 8 e2m1 values packed in one uint32_t. This is
// the same conversion sequence used by Sage3's BF16/FP16 quantizer.
inline __device__ uint32_t fp32_vec_to_e2m1_local(float2* array) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

inline __device__ uint8_t fp32_to_e4m3_byte(float value) {
  uint8_t raw;
  reinterpret_cast<__nv_fp8_e4m3&>(raw) = __nv_fp8_e4m3(value);
  return raw;
}

inline __device__ float e4m3_byte_to_fp32(uint8_t raw) {
  return float(reinterpret_cast<__nv_fp8_e4m3&>(raw));
}

inline __device__ uint64_t scaled_floats_to_fp4(float vals[kFp4EltsPerThread],
                                                uint8_t* sf_raw_out) {
  float vec_max = 0.0f;
#pragma unroll
  for (int i = 0; i < kFp4EltsPerThread; ++i) {
    vec_max = fmaxf(vec_max, fabsf(vals[i]));
  }

  float sf_value = vec_max / 6.0f;
  const uint8_t sf_raw = fp32_to_e4m3_byte(sf_value);
  sf_value = e4m3_byte_to_fp32(sf_raw);
  const float sf_inv = (sf_value == 0.0f) ? 0.0f : 1.0f / sf_value;
  *sf_raw_out = sf_raw;

  float2 fp2_vals[kFp4EltsPerThread / 2];
#pragma unroll
  for (int i = 0; i < kFp4EltsPerThread / 2; ++i) {
    fp2_vals[i].x = vals[2 * i] * sf_inv;
    fp2_vals[i].y = vals[2 * i + 1] * sf_inv;
  }

  uint32_t e2m1_vals[kFp4EltsPerThread / 8];
#pragma unroll
  for (int i = 0; i < kFp4EltsPerThread / 8; ++i) {
    e2m1_vals[i] = fp32_vec_to_e2m1_local(fp2_vals + i * 4);
  }
  return uint64_t(e2m1_vals[0]) | (uint64_t(e2m1_vals[1]) << 32);
}

template <int HeadDim, int BlockSize, bool Permute>
__global__ void fp8_scaled_fp4_quant_kernel(
    const cutlass::float_e4m3_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ output_sf,
    int B,
    int H,
    int num_tokens,
    int padded_tokens) {
  static_assert(BlockSize == kSage3QuantBlockTokens,
                "Sage3 quantizer assumes 128-token blocks");
  static_assert(kFp4EltsPerThread == 16,
                "Sage3 scale layout below assumes 16 FP4 elements/thread");
  constexpr int kThreadsPerToken = HeadDim / kFp4EltsPerThread;

  const int b = blockIdx.y;
  const int h = blockIdx.z;
  const int token_block = blockIdx.x;
  const int local_token = threadIdx.x / kThreadsPerToken;
  const int col = threadIdx.x % kThreadsPerToken;
  const int token_id = token_block * BlockSize + local_token;

  int load_token_id = token_id;
  if constexpr (Permute) {
    const int residue = local_token % 32;
    load_token_id = token_block * BlockSize + (local_token / 32) * 32 +
                    (residue / 8) * 2 + ((residue % 8) / 2) * 8 +
                    (residue % 8) % 2;
  }

  cutlass::NumericConverter<float, cutlass::float_e4m3_t> to_float;
  float vals[kFp4EltsPerThread];
#pragma unroll
  for (int i = 0; i < kFp4EltsPerThread; ++i) {
    vals[i] = 0.0f;
  }
  if (b < B && h < H && load_token_id < num_tokens) {
    const int d_base = col * kFp4EltsPerThread;
    const int64_t input_base =
        ((int64_t(b) * num_tokens + load_token_id) * H + h) * HeadDim + d_base;
#pragma unroll
    for (int i = 0; i < kFp4EltsPerThread; ++i) {
      vals[i] = to_float(input[input_base + i]);
    }
  }

  uint8_t sf_raw = 0;
  const uint64_t packed = scaled_floats_to_fp4(vals, &sf_raw);
  const int64_t out_offset =
      ((int64_t(b) * H + h) * padded_tokens + token_id) * (HeadDim / 2) +
      col * (kFp4EltsPerThread / 2);
  reinterpret_cast<uint64_t*>(output + out_offset)[0] = packed;

  uint8_t* sf_base =
      output_sf +
      ((int64_t(b) * H + h) * padded_tokens + (token_id / 64) * 64) *
          (HeadDim / kFp4EltsPerThread);
  const uint32_t token_local = token_id % 64;
  const uint32_t col_local = col;
  const uint32_t sf_offset =
      (col_local / 4) * 256 + (col_local % 4) +
      (token_local / 16) * 4 + (token_local % 16) * 16;
  sf_base[sf_offset] = sf_raw;
}

template <int HeadDim, int BlockSize>
__global__ void fp8_scaled_fp4_quant_trans_kernel(
    const cutlass::float_e4m3_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ output_sf,
    int B,
    int H,
    int num_tokens,
    int padded_tokens) {
  static_assert(BlockSize == kSage3QuantBlockTokens,
                "Sage3 quantizer assumes 128-token blocks");
  static_assert(kFp4EltsPerThread == 16,
                "Sage3 scale layout below assumes 16 FP4 elements/thread");
  constexpr int kThreadsPerToken = HeadDim / kFp4EltsPerThread;
  constexpr int kThreadsPerSeq = BlockSize / kFp4EltsPerThread;

  const int b = blockIdx.y;
  const int h = blockIdx.z;
  const int token_block = blockIdx.x;
  const int local_token = threadIdx.x / kThreadsPerToken;
  const int col = threadIdx.x % kThreadsPerToken;
  const int token_id = token_block * BlockSize + local_token;
  __shared__ uint8_t shared[BlockSize * HeadDim];

  cutlass::NumericConverter<float, cutlass::float_e4m3_t> to_float;
  const int d_base = col * kFp4EltsPerThread;
#pragma unroll
  for (int i = 0; i < kFp4EltsPerThread; ++i) {
    uint8_t value = 0;
    if (b < B && h < H && token_id < num_tokens) {
      const int64_t input_idx =
          ((int64_t(b) * num_tokens + token_id) * H + h) * HeadDim + d_base + i;
      value = input[input_idx].storage;
    }
    shared[local_token * HeadDim + d_base + i] = value;
  }
  __syncthreads();

  const int d = threadIdx.x / kThreadsPerSeq;
  const int token_group = threadIdx.x % kThreadsPerSeq;
  float vals[kFp4EltsPerThread];
#pragma unroll
  for (int i = 0; i < kFp4EltsPerThread; ++i) {
    vals[i] = to_float(
        cutlass::float_e4m3_t::bitcast(
            shared[(token_group * kFp4EltsPerThread + i) * HeadDim + d]));
  }

  uint8_t sf_raw = 0;
  const uint64_t packed = scaled_floats_to_fp4(vals, &sf_raw);
  const int64_t out_offset =
      ((int64_t(b) * H + h) * HeadDim + d) * (padded_tokens / 2) +
      (token_block * BlockSize + token_group * kFp4EltsPerThread) / 2;
  reinterpret_cast<uint64_t*>(output + out_offset)[0] = packed;

  uint8_t* sf_base =
      output_sf +
      ((int64_t(b) * H + h) * HeadDim + (d / 64) * 64) *
          (padded_tokens / kFp4EltsPerThread);
  const uint32_t row_local = d % 64;
  const uint32_t col_local = token_block * BlockSize / kFp4EltsPerThread +
                             token_group;
  const uint32_t sf_offset =
      (col_local / 4) * 256 + (col_local % 4) +
      (row_local / 16) * 4 + (row_local % 16) * 16;
  sf_base[sf_offset] = sf_raw;
}

template <bool Permute>
cudaError_t launch_fp8_scaled_fp4_quant(
    const cutlass::float_e4m3_t* input,
    int B,
    int num_tokens,
    int H,
    int D,
    int padded_tokens,
    const at::Tensor& fp4,
    const at::Tensor& sf,
    cudaStream_t stream) {
  if (!input || !fp4.defined() || !sf.defined()) return cudaErrorInvalidValue;
  if (D != 64 && D != 128) return cudaErrorInvalidValue;
  if (padded_tokens % kSage3QuantBlockTokens != 0) return cudaErrorInvalidValue;

  const dim3 grid(padded_tokens / kSage3QuantBlockTokens, B, H);
  if (D == 64) {
    constexpr int kHeadDim = 64;
    constexpr int kThreads =
        kSage3QuantBlockTokens * kHeadDim / kFp4EltsPerThread;
    fp8_scaled_fp4_quant_kernel<kHeadDim, kSage3QuantBlockTokens, Permute>
        <<<grid, kThreads, 0, stream>>>(
            input,
            fp4.data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(sf.data_ptr()),
            B, H, num_tokens, padded_tokens);
  } else {
    constexpr int kHeadDim = 128;
    constexpr int kThreads =
        kSage3QuantBlockTokens * kHeadDim / kFp4EltsPerThread;
    fp8_scaled_fp4_quant_kernel<kHeadDim, kSage3QuantBlockTokens, Permute>
        <<<grid, kThreads, 0, stream>>>(
            input,
            fp4.data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(sf.data_ptr()),
            B, H, num_tokens, padded_tokens);
  }
  return cudaGetLastError();
}

cudaError_t launch_fp8_scaled_fp4_quant_trans(
    const cutlass::float_e4m3_t* input,
    int B,
    int num_tokens,
    int H,
    int D,
    int padded_tokens,
    const at::Tensor& fp4,
    const at::Tensor& sf,
    cudaStream_t stream) {
  if (!input || !fp4.defined() || !sf.defined()) return cudaErrorInvalidValue;
  if (D != 64 && D != 128) return cudaErrorInvalidValue;
  if (padded_tokens % kSage3QuantBlockTokens != 0) return cudaErrorInvalidValue;

  const dim3 grid(padded_tokens / kSage3QuantBlockTokens, B, H);
  if (D == 64) {
    constexpr int kHeadDim = 64;
    constexpr int kThreads =
        kSage3QuantBlockTokens * kHeadDim / kFp4EltsPerThread;
    fp8_scaled_fp4_quant_trans_kernel<kHeadDim, kSage3QuantBlockTokens>
        <<<grid, kThreads, 0, stream>>>(
            input,
            fp4.data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(sf.data_ptr()),
            B, H, num_tokens, padded_tokens);
  } else {
    constexpr int kHeadDim = 128;
    constexpr int kThreads =
        kSage3QuantBlockTokens * kHeadDim / kFp4EltsPerThread;
    fp8_scaled_fp4_quant_trans_kernel<kHeadDim, kSage3QuantBlockTokens>
        <<<grid, kThreads, 0, stream>>>(
            input,
            fp4.data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(sf.data_ptr()),
            B, H, num_tokens, padded_tokens);
  }
  return cudaGetLastError();
}

template <bool ApplyRope>
__global__ void bf16_q_to_sage3_fp4_kernel(
    const cutlass::bfloat16_t* __restrict__ input,
    const cutlass::bfloat16_t* __restrict__ gamma,
    const cutlass::bfloat16_t* __restrict__ rope_cos,
    const cutlass::bfloat16_t* __restrict__ rope_sin,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ output_sf,
    int B,
    int num_tokens,
    int H,
    int D,
    int input_row_stride,
    int input_head_offset,
    int padded_tokens) {
  const int token_id = blockIdx.x;
  const int b = blockIdx.y;
  const int h = blockIdx.z;
  const int tid = threadIdx.x;

  extern __shared__ float smem[];
  float acc = 0.0f;
  if (token_id < num_tokens) {
    const int64_t input_base =
        int64_t(b * num_tokens + token_id) * input_row_stride +
        input_head_offset + h * D;
    for (int d = tid; d < D; d += blockDim.x) {
      const float v = bf16_to_float(input[input_base + d]);
      acc += v * v;
    }
  }
  smem[tid] = acc;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) smem[tid] += smem[tid + off];
    __syncthreads();
  }
  const float rms = rsqrtf(smem[0] / float(D) + 1e-6f);

  if (tid >= D / kFp4EltsPerThread) return;
  const int col = tid;
  const int d_base = col * kFp4EltsPerThread;
  const int d_half = D / 2;
  float vals[kFp4EltsPerThread];
#pragma unroll
  for (int i = 0; i < kFp4EltsPerThread; ++i) {
    vals[i] = 0.0f;
  }

  if (token_id < num_tokens) {
    const int64_t input_base =
        int64_t(b * num_tokens + token_id) * input_row_stride +
        input_head_offset + h * D;
    const int64_t rope_base = int64_t(token_id) * D;
#pragma unroll
    for (int i = 0; i < kFp4EltsPerThread; ++i) {
      const int d = d_base + i;
      float q = bf16_to_float(input[input_base + d]) *
                rms * bf16_to_float(gamma[d]);
      if constexpr (ApplyRope) {
        const int d_op = (d < d_half) ? (d + d_half) : (d - d_half);
        const float q_op =
            bf16_to_float(input[input_base + d_op]) *
            rms * bf16_to_float(gamma[d_op]);
        const float c = bf16_to_float(rope_cos[rope_base + d]);
        const float s = bf16_to_float(rope_sin[rope_base + d]);
        const float rot = (d < d_half) ? -q_op : q_op;
        q = q * c + rot * s;
      }
      vals[i] = q;
    }
  }

  uint8_t sf_raw = 0;
  const uint64_t packed = scaled_floats_to_fp4(vals, &sf_raw);
  const int64_t out_offset =
      ((int64_t(b) * H + h) * padded_tokens + token_id) * (D / 2) +
      col * (kFp4EltsPerThread / 2);
  reinterpret_cast<uint64_t*>(output + out_offset)[0] = packed;

  uint8_t* sf_base =
      output_sf +
      ((int64_t(b) * H + h) * padded_tokens + (token_id / 64) * 64) *
          (D / kFp4EltsPerThread);
  const uint32_t token_local = token_id % 64;
  const uint32_t col_local = col;
  const uint32_t sf_offset =
      (col_local / 4) * 256 + (col_local % 4) +
      (token_local / 16) * 4 + (token_local % 16) * 16;
  sf_base[sf_offset] = sf_raw;
}

at::Tensor packed_bmhk_view(void* ptr, int B, int M, int H, int D,
                            int device) {
  auto opts = at::TensorOptions()
                  .device(at::kCUDA, device)
                  .dtype(at::kBFloat16);
  return torch::from_blob(
      ptr,
      {B, H, M, D},
      {int64_t(M) * H * D, D, int64_t(H) * D, 1},
      opts);
}

at::Tensor sage3_fp4_q_or_k_view(const uint8_t* ptr, int B, int H,
                                  int M_padded, int D, int device) {
  auto opts = at::TensorOptions().device(at::kCUDA, device).dtype(at::kByte);
  return torch::from_blob(
      const_cast<uint8_t*>(ptr),
      {B, H, M_padded, D / 2},
      opts);
}

at::Tensor sage3_fp4_v_view(const uint8_t* ptr, int B, int H,
                             int M_padded, int D, int device) {
  auto opts = at::TensorOptions().device(at::kCUDA, device).dtype(at::kByte);
  return torch::from_blob(
      const_cast<uint8_t*>(ptr),
      {B, H, D, M_padded / 2},
      opts);
}

at::Tensor sage3_sf_q_or_k_view(const cutlass::float_e4m3_t* ptr, int B, int H,
                                 int M_padded, int D, int device) {
  auto opts = at::TensorOptions()
                  .device(at::kCUDA, device)
                  .dtype(at::ScalarType::Float8_e4m3fn);
  return torch::from_blob(
      const_cast<cutlass::float_e4m3_t*>(ptr),
      {B, H, M_padded, D / kFp4EltsPerThread},
      opts);
}

at::Tensor sage3_sf_v_view(const cutlass::float_e4m3_t* ptr, int B, int H,
                            int M_padded, int D, int device) {
  auto opts = at::TensorOptions()
                  .device(at::kCUDA, device)
                  .dtype(at::ScalarType::Float8_e4m3fn);
  return torch::from_blob(
      const_cast<cutlass::float_e4m3_t*>(ptr),
      {B, H, D, M_padded / kFp4EltsPerThread},
      opts);
}

at::Tensor pad_seq_to_128(const at::Tensor& x) {
  const int64_t L = x.size(2);
  const int64_t L_round = round_up(static_cast<int>(L), 128);
  if (L == L_round) {
    return x.contiguous();
  }

  auto out = at::zeros({x.size(0), x.size(1), L_round, x.size(3)}, x.options());
  out.narrow(2, 0, L).copy_(x);
  return out;
}

at::Tensor subtract_group_mean(const at::Tensor& q, at::Tensor* qm_out,
                               bool per_block_mean) {
  const int64_t B = q.size(0);
  const int64_t H = q.size(1);
  const int64_t L = q.size(2);
  const int64_t D = q.size(3);

  if (per_block_mean) {
    TORCH_CHECK(L % 128 == 0, "Sage3 Q length must be padded to 128");
    const int64_t groups = L / 128;
    auto q_blocks = q.view({B, H, groups, 128, D});
    auto qm = q_blocks.mean(/*dim=*/3);
    auto expanded = qm.unsqueeze(3).expand({B, H, groups, 128, D})
                        .reshape({B, H, L, D});
    *qm_out = qm.contiguous();
    return (q - expanded).contiguous();
  }

  auto qm = q.mean(/*dim=*/2, /*keepdim=*/true);
  *qm_out = qm.contiguous();
  return (q - qm).contiguous();
}

void quantize_fp4(const at::Tensor& q,
                  const at::Tensor& k,
                  const at::Tensor& v,
                  at::Tensor* q_fp4,
                  at::Tensor* k_fp4,
                  at::Tensor* v_fp4,
                  at::Tensor* q_sf,
                  at::Tensor* k_sf,
                  at::Tensor* v_sf) {
  auto u8_opts = q.options().dtype(at::kByte);
  auto fp8_opts = q.options().dtype(at::ScalarType::Float8_e4m3fn);

  const int64_t B = q.size(0);
  const int64_t H = q.size(1);
  const int64_t QL = q.size(2);
  const int64_t KL = k.size(2);
  const int64_t D = q.size(3);

  *q_fp4 = at::empty({B, H, QL, D / 2}, u8_opts);
  *k_fp4 = at::empty({B, H, KL, D / 2}, u8_opts);
  *v_fp4 = at::empty({B, H, D, KL / 2}, u8_opts);
  *q_sf = at::empty({B, H, QL, D / 16}, fp8_opts);
  *k_sf = at::empty({B, H, KL, D / 16}, fp8_opts);
  *v_sf = at::empty({B, H, D, KL / 16}, fp8_opts);

  scaled_fp4_quant(q, *q_fp4, *q_sf, 1);
  scaled_fp4_quant_permute(k, *k_fp4, *k_sf, 1);
  scaled_fp4_quant_trans(v, *v_fp4, *v_sf, 1);
}

cudaError_t launch_bf16_q_to_sage3_fp4(
    const cutlass::bfloat16_t* input,
    const cutlass::bfloat16_t* gamma,
    const cutlass::bfloat16_t* rope_cos,
    const cutlass::bfloat16_t* rope_sin,
    uint8_t* q_fp4,
    cutlass::float_e4m3_t* q_sf,
    int B,
    int Mq,
    int H,
    int D,
    int input_row_stride,
    int input_head_offset,
    bool apply_rope,
    int padded_mq,
    cudaStream_t stream) {
  if (!input || !gamma || !q_fp4 || !q_sf) return cudaErrorInvalidValue;
  if (apply_rope && (!rope_cos || !rope_sin)) return cudaErrorInvalidValue;
  if (B <= 0 || Mq <= 0 || H <= 0) return cudaErrorInvalidValue;
  if (D != 64 && D != 128) return cudaErrorInvalidValue;
  if (padded_mq < Mq || padded_mq % kSage3QuantBlockTokens != 0) {
    return cudaErrorInvalidValue;
  }

  const int threads = (D <= 64) ? 64 : 128;
  const dim3 grid(padded_mq, B, H);
  const size_t smem = threads * sizeof(float);
  if (apply_rope) {
    bf16_q_to_sage3_fp4_kernel</*ApplyRope=*/true>
        <<<grid, threads, smem, stream>>>(
            input, gamma, rope_cos, rope_sin, q_fp4,
            reinterpret_cast<uint8_t*>(q_sf), B, Mq, H, D,
            input_row_stride, input_head_offset, padded_mq);
  } else {
    bf16_q_to_sage3_fp4_kernel</*ApplyRope=*/false>
        <<<grid, threads, smem, stream>>>(
            input, gamma, nullptr, nullptr, q_fp4,
            reinterpret_cast<uint8_t*>(q_sf), B, Mq, H, D,
            input_row_stride, input_head_offset, padded_mq);
  }
  return cudaGetLastError();
}

}  // namespace

bool sage3_is_built() {
  return true;
}

bool sage3_is_runtime_supported(int device) {
  if (device < 0) {
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return false;
  }
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) return false;
  // sage3_ops is compiled with -gencode=arch=compute_120a,code=sm_120a.
  // All known SM120 devices support SM120a; if a future SKU does not, the
  // CUDA driver will reject the kernel at launch.
  return prop.major == 12 && prop.minor == 0;
}

cudaError_t sage3_quantize_q_bf16(
    const cutlass::bfloat16_t* Q,
    const cutlass::bfloat16_t* gamma,
    const cutlass::bfloat16_t* rope_cos,
    const cutlass::bfloat16_t* rope_sin,
    uint8_t* Q_fp4,
    cutlass::float_e4m3_t* Q_sf,
    int B, int Mq,
    int H, int D,
    int input_row_stride,
    int input_head_offset,
    bool apply_rope,
    int padded_mq,
    cudaStream_t stream) {
  return launch_bf16_q_to_sage3_fp4(
      Q, gamma, rope_cos, rope_sin, Q_fp4, Q_sf,
      B, Mq, H, D, input_row_stride, input_head_offset,
      apply_rope, padded_mq, stream);
}

std::vector<at::Tensor> sage3_quantize_cross_kv_bf16(
    at::Tensor k_bmhd,
    at::Tensor v_bmhd) {
  TORCH_CHECK(k_bmhd.is_cuda() && v_bmhd.is_cuda(), "Sage3 K/V inputs must be CUDA tensors");
  TORCH_CHECK(k_bmhd.device() == v_bmhd.device(),
              "Sage3 K/V inputs must be on the same CUDA device");
  TORCH_CHECK(k_bmhd.scalar_type() == at::kBFloat16 &&
              v_bmhd.scalar_type() == at::kBFloat16,
              "Sage3 K/V inputs must be bfloat16");
  TORCH_CHECK(k_bmhd.dim() == 4 && v_bmhd.dim() == 4,
              "Sage3 K/V inputs must have shape [B, M, H, D]");
  TORCH_CHECK(k_bmhd.sizes() == v_bmhd.sizes(),
              "Sage3 K/V inputs must have matching shapes");
  c10::cuda::CUDAGuard device_guard(k_bmhd.device());
  at::NoGradGuard no_grad;

  const int64_t B = k_bmhd.size(0);
  const int64_t M = k_bmhd.size(1);
  const int64_t H = k_bmhd.size(2);
  const int64_t D = k_bmhd.size(3);
  TORCH_CHECK(D == 64 || D == 128, "Sage3 head_dim must be 64 or 128, got ", D);

  auto k = k_bmhd.permute({0, 2, 1, 3}).contiguous();
  auto v = v_bmhd.permute({0, 2, 1, 3}).contiguous();
  auto k_padded = pad_seq_to_128(k);
  auto v_padded = pad_seq_to_128(v);
  const int64_t Mp = k_padded.size(2);

  auto u8_opts = k.options().dtype(at::kByte);
  auto fp8_opts = k.options().dtype(at::ScalarType::Float8_e4m3fn);
  auto k_fp4 = at::empty({B, H, Mp, D / 2}, u8_opts);
  auto v_fp4 = at::empty({B, H, D, Mp / 2}, u8_opts);
  auto k_sf = at::empty({B, H, Mp, D / kFp4EltsPerThread}, fp8_opts);
  auto v_sf = at::empty({B, H, D, Mp / kFp4EltsPerThread}, fp8_opts);

  scaled_fp4_quant_permute(k_padded, k_fp4, k_sf, 1);
  scaled_fp4_quant_trans(v_padded, v_fp4, v_sf, 1);
  return {k_fp4, v_fp4, k_sf, v_sf};
}

cudaError_t run_sage3_fmha_packed_qkv(
    const cutlass::bfloat16_t* Q,
    const cutlass::bfloat16_t* K,
    const cutlass::bfloat16_t* V,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    cudaStream_t stream) {
  if (Mq <= 0 || Mk <= 0) return cudaErrorInvalidValue;
  if (D != 64 && D != 128) return cudaErrorInvalidValue;

  int device = 0;
  cudaError_t device_err = cudaGetDevice(&device);
  if (device_err != cudaSuccess) return device_err;

  c10::cuda::CUDAGuard device_guard(device);
  CurrentStreamScope stream_scope(stream, device);
  at::NoGradGuard no_grad;

  auto q_view = packed_bmhk_view(const_cast<cutlass::bfloat16_t*>(Q), B, Mq, H, D, device);
  auto k_view = packed_bmhk_view(const_cast<cutlass::bfloat16_t*>(K), B, Mk, H, D, device);
  auto v_view = packed_bmhk_view(const_cast<cutlass::bfloat16_t*>(V), B, Mk, H, D, device);
  auto o_view = packed_bmhk_view(O, B, Mq, H, D, device);

  constexpr bool per_block_mean = true;
  auto q_padded = pad_seq_to_128(q_view);
  auto k_centered = k_view - k_view.mean(/*dim=*/2, /*keepdim=*/true);
  auto k_padded = pad_seq_to_128(k_centered);
  auto v_padded = pad_seq_to_128(v_view);

  at::Tensor q_mean;
  auto q_centered = subtract_group_mean(q_padded, &q_mean, per_block_mean);
  auto delta_s = at::matmul(q_mean, k_padded.transpose(-2, -1))
                     .to(at::kFloat)
                     .contiguous();

  at::Tensor q_fp4, k_fp4, v_fp4, q_sf, k_sf, v_sf;
  quantize_fp4(q_centered, k_padded.contiguous(), v_padded.contiguous(),
               &q_fp4, &k_fp4, &v_fp4, &q_sf, &k_sf, &v_sf);

  c10::optional<at::Tensor> out_opt = c10::nullopt;
  const float softmax_scale = (scale > 0.f) ? scale : (1.0f / std::sqrt(float(D)));
  auto out = mha_fwd(q_fp4, k_fp4, v_fp4, q_sf, k_sf, v_sf, delta_s,
                     Mk, out_opt, softmax_scale, causal,
                     per_block_mean, /*is_bf16=*/true)
                 .at(0);
  o_view.copy_(out.narrow(2, 0, Mq));
  return cudaGetLastError();
}

static cudaError_t run_sage3_fmha_packed_fp4_impl(
    const uint8_t* Q_fp4,
    const uint8_t* K_fp4,
    const uint8_t* V_fp4,
    const cutlass::float_e4m3_t* Q_sf,
    const cutlass::float_e4m3_t* K_sf,
    const cutlass::float_e4m3_t* V_sf,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    int padded_mq,
    int padded_mk,
    cudaStream_t stream) {
  if (B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0) return cudaErrorInvalidValue;
  if (D != 64 && D != 128) return cudaErrorInvalidValue;
  if (!Q_fp4 || !K_fp4 || !V_fp4 || !Q_sf || !K_sf || !V_sf || !O) {
    return cudaErrorInvalidValue;
  }
  if (padded_mq < Mq || padded_mk < Mk ||
      padded_mq % kSage3QuantBlockTokens != 0 ||
      padded_mk % kSage3QuantBlockTokens != 0) {
    return cudaErrorInvalidValue;
  }

  int device = 0;
  cudaError_t device_err = cudaGetDevice(&device);
  if (device_err != cudaSuccess) return device_err;

  c10::cuda::CUDAGuard device_guard(device);
  CurrentStreamScope stream_scope(stream, device);
  at::NoGradGuard no_grad;

  auto q_fp4 = sage3_fp4_q_or_k_view(Q_fp4, B, H, padded_mq, D, device);
  auto k_fp4 = sage3_fp4_q_or_k_view(K_fp4, B, H, padded_mk, D, device);
  auto v_fp4 = sage3_fp4_v_view(V_fp4, B, H, padded_mk, D, device);
  auto q_sf = sage3_sf_q_or_k_view(Q_sf, B, H, padded_mq, D, device);
  auto k_sf = sage3_sf_q_or_k_view(K_sf, B, H, padded_mk, D, device);
  auto v_sf = sage3_sf_v_view(V_sf, B, H, padded_mk, D, device);

  const int QGroups = padded_mq / kSage3QuantBlockTokens;
  auto delta_s = at::empty(
      {B, H, QGroups, padded_mk},
      at::TensorOptions().device(at::kCUDA, device).dtype(at::kFloat));
  cudaError_t err = cudaMemsetAsync(delta_s.data_ptr(), 0, delta_s.nbytes(), stream);
  if (err != cudaSuccess) return err;

  c10::optional<at::Tensor> out_opt = c10::nullopt;
  const float softmax_scale = (scale > 0.f) ? scale : (1.0f / std::sqrt(float(D)));
  auto out = mha_fwd(q_fp4, k_fp4, v_fp4, q_sf, k_sf, v_sf, delta_s,
                     Mk, out_opt, softmax_scale, causal,
                     /*per_block_mean=*/true, /*is_bf16=*/true)
                 .at(0);
  auto o_view = packed_bmhk_view(O, B, Mq, H, D, device);
  o_view.copy_(out.narrow(2, 0, Mq));
  return cudaGetLastError();
}

cudaError_t run_sage3_fmha_packed_qfp4_kvfp8(
    const uint8_t* Q_fp4,
    const cutlass::float_e4m3_t* Q_sf,
    const cutlass::float_e4m3_t* K,
    const cutlass::float_e4m3_t* V,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    int padded_mq,
    cudaStream_t stream) {
  if (!Q_fp4 || !Q_sf || !K || !V || !O) return cudaErrorInvalidValue;
  if (B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0) return cudaErrorInvalidValue;
  if (D != 64 && D != 128) return cudaErrorInvalidValue;
  const int padded_mk = round_up(Mk, kSage3QuantBlockTokens);

  int device = 0;
  cudaError_t device_err = cudaGetDevice(&device);
  if (device_err != cudaSuccess) return device_err;

  c10::cuda::CUDAGuard device_guard(device);
  CurrentStreamScope stream_scope(stream, device);
  at::NoGradGuard no_grad;

  auto opts = at::TensorOptions().device(at::kCUDA, device);
  auto u8_opts = opts.dtype(at::kByte);
  auto fp8_opts = opts.dtype(at::ScalarType::Float8_e4m3fn);
  auto k_fp4 = at::empty({B, H, padded_mk, D / 2}, u8_opts);
  auto v_fp4 = at::empty({B, H, D, padded_mk / 2}, u8_opts);
  auto k_sf = at::empty({B, H, padded_mk, D / kFp4EltsPerThread}, fp8_opts);
  auto v_sf = at::empty({B, H, D, padded_mk / kFp4EltsPerThread}, fp8_opts);

  cudaError_t err = launch_fp8_scaled_fp4_quant</*Permute=*/true>(
      K, B, Mk, H, D, padded_mk, k_fp4, k_sf, stream);
  if (err != cudaSuccess) return err;
  err = launch_fp8_scaled_fp4_quant_trans(
      V, B, Mk, H, D, padded_mk, v_fp4, v_sf, stream);
  if (err != cudaSuccess) return err;

  return run_sage3_fmha_packed_fp4_impl(
      Q_fp4,
      k_fp4.data_ptr<uint8_t>(),
      v_fp4.data_ptr<uint8_t>(),
      Q_sf,
      reinterpret_cast<const cutlass::float_e4m3_t*>(k_sf.data_ptr()),
      reinterpret_cast<const cutlass::float_e4m3_t*>(v_sf.data_ptr()),
      O, B, Mq, Mk, H, D, causal, scale, padded_mq, padded_mk, stream);
}

cudaError_t run_sage3_fmha_packed_qkv_fp4(
    const uint8_t* Q_fp4,
    const uint8_t* K_fp4,
    const uint8_t* V_fp4,
    const cutlass::float_e4m3_t* Q_sf,
    const cutlass::float_e4m3_t* K_sf,
    const cutlass::float_e4m3_t* V_sf,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    int padded_mq,
    int padded_mk,
    cudaStream_t stream) {
  return run_sage3_fmha_packed_fp4_impl(
      Q_fp4, K_fp4, V_fp4, Q_sf, K_sf, V_sf, O,
      B, Mq, Mk, H, D, causal, scale, padded_mq, padded_mk, stream);
}

cudaError_t run_sage3_fmha_packed_qkv_fp8(
    const cutlass::float_e4m3_t* Q,
    const cutlass::float_e4m3_t* K,
    const cutlass::float_e4m3_t* V,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    cudaStream_t stream) {
  if (B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0) return cudaErrorInvalidValue;
  if (D != 64 && D != 128) return cudaErrorInvalidValue;
  if (!Q || !K || !V || !O) return cudaErrorInvalidValue;

  int device = 0;
  cudaError_t device_err = cudaGetDevice(&device);
  if (device_err != cudaSuccess) return device_err;

  c10::cuda::CUDAGuard device_guard(device);
  CurrentStreamScope stream_scope(stream, device);
  at::NoGradGuard no_grad;

  const bool profile = env_flag_enabled("OMNIDREAMS_DIT_SAGE3_PROFILE");
  enum {
    EV_START,
    EV_AFTER_Q_QUANT,
    EV_AFTER_K_QUANT,
    EV_AFTER_V_QUANT,
    EV_AFTER_DELTA,
    EV_AFTER_MHA,
    EV_AFTER_COPY,
    EV_COUNT,
  };
  cudaEvent_t ev[EV_COUNT] = {};
  auto cleanup_events = [&]() {
    if (!profile) return;
    for (int i = 0; i < EV_COUNT; ++i) {
      if (ev[i]) cudaEventDestroy(ev[i]);
    }
  };
  auto rec = [&](int idx) {
    if (profile) cudaEventRecord(ev[idx], stream);
  };
  if (profile) {
    for (int i = 0; i < EV_COUNT; ++i) cudaEventCreate(&ev[i]);
    rec(EV_START);
  }

  const int QL = round_up(Mq, kSage3QuantBlockTokens);
  const int KL = round_up(Mk, kSage3QuantBlockTokens);
  const int QGroups = QL / kSage3QuantBlockTokens;
  auto opts = at::TensorOptions().device(at::kCUDA, device);
  auto u8_opts = opts.dtype(at::kByte);
  auto fp8_opts = opts.dtype(at::ScalarType::Float8_e4m3fn);
  auto q_fp4 = at::empty({B, H, QL, D / 2}, u8_opts);
  auto k_fp4 = at::empty({B, H, KL, D / 2}, u8_opts);
  auto v_fp4 = at::empty({B, H, D, KL / 2}, u8_opts);
  auto q_sf = at::empty({B, H, QL, D / 16}, fp8_opts);
  auto k_sf = at::empty({B, H, KL, D / 16}, fp8_opts);
  auto v_sf = at::empty({B, H, D, KL / 16}, fp8_opts);

  cudaError_t err = launch_fp8_scaled_fp4_quant</*Permute=*/false>(
      Q, B, Mq, H, D, QL, q_fp4, q_sf, stream);
  if (err != cudaSuccess) {
    cleanup_events();
    return err;
  }
  rec(EV_AFTER_Q_QUANT);

  err = launch_fp8_scaled_fp4_quant</*Permute=*/true>(
      K, B, Mk, H, D, KL, k_fp4, k_sf, stream);
  if (err != cudaSuccess) {
    cleanup_events();
    return err;
  }
  rec(EV_AFTER_K_QUANT);

  err = launch_fp8_scaled_fp4_quant_trans(
      V, B, Mk, H, D, KL, v_fp4, v_sf, stream);
  if (err != cudaSuccess) {
    cleanup_events();
    return err;
  }
  rec(EV_AFTER_V_QUANT);

  // This FP8 path intentionally does not recenter Q/K. A zero correction keeps
  // Sage3's per-block-mean interface equivalent to uncentered attention.
  auto delta_s = at::empty(
      {B, H, QGroups, KL},
      at::TensorOptions().device(at::kCUDA, device).dtype(at::kFloat));
  err = cudaMemsetAsync(delta_s.data_ptr(), 0, delta_s.nbytes(), stream);
  if (err != cudaSuccess) {
    cleanup_events();
    return err;
  }
  rec(EV_AFTER_DELTA);

  c10::optional<at::Tensor> out_opt = c10::nullopt;
  const float softmax_scale = (scale > 0.f) ? scale : (1.0f / std::sqrt(float(D)));
  auto out = mha_fwd(q_fp4, k_fp4, v_fp4, q_sf, k_sf, v_sf, delta_s,
                     Mk, out_opt, softmax_scale, causal,
                     /*per_block_mean=*/true, /*is_bf16=*/true)
                 .at(0);
  rec(EV_AFTER_MHA);

  auto o_view = packed_bmhk_view(O, B, Mq, H, D, device);
  o_view.copy_(out.narrow(2, 0, Mq));
  rec(EV_AFTER_COPY);

  err = cudaGetLastError();
  if (profile) {
    cudaEventSynchronize(ev[EV_AFTER_COPY]);
    auto ms = [&](int a, int b) {
      float out_ms = 0.0f;
      cudaEventElapsedTime(&out_ms, ev[a], ev[b]);
      return out_ms;
    };
    std::printf(
        "[sage3-fp8] B=%d Mq=%d Mk=%d H=%d D=%d "
        "q=%.3f k=%.3f v=%.3f delta=%.3f mha=%.3f copy=%.3f total=%.3f ms\n",
        B, Mq, Mk, H, D,
        ms(EV_START, EV_AFTER_Q_QUANT),
        ms(EV_AFTER_Q_QUANT, EV_AFTER_K_QUANT),
        ms(EV_AFTER_K_QUANT, EV_AFTER_V_QUANT),
        ms(EV_AFTER_V_QUANT, EV_AFTER_DELTA),
        ms(EV_AFTER_DELTA, EV_AFTER_MHA),
        ms(EV_AFTER_MHA, EV_AFTER_COPY),
        ms(EV_START, EV_AFTER_COPY));
    std::fflush(stdout);
  }
  cleanup_events();
  return err;
}

}  // namespace omnidreams_singleview
