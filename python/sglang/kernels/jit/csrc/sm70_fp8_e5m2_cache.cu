// SPDX-License-Identifier: Apache-2.0
// Fused FP16 -> E5M2 paged-cache writer for Volta.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include <algorithm>
#include <climits>

namespace {

__device__ __forceinline__ uint8_t fp16_bits_to_e5m2_satfinite_rn(
    const uint16_t bits) {
  const uint16_t sign = (bits >> 8) & 0x80u;
  const uint16_t magnitude = bits & 0x7fffu;
  const uint16_t exponent = magnitude >> 10;
  const uint16_t mantissa = magnitude & 0x03ffu;
  if (exponent == 0x1fu) {
    return mantissa == 0 ? static_cast<uint8_t>(sign | 0x7bu) : 0x7fu;
  }
  uint16_t rounded = magnitude >> 8;
  const uint16_t remainder = magnitude & 0x00ffu;
  rounded += remainder > 0x80u ||
             (remainder == 0x80u && (rounded & 1u) != 0u);
  rounded = rounded > 0x7bu ? 0x7bu : rounded;
  return static_cast<uint8_t>(sign | rounded);
}

__device__ __forceinline__ uint16_t scaled_half_bits(
    const __half value, const float scale) {
  if (scale == 1.f) {
    return __half_as_ushort(value);
  }
  return __half_as_ushort(__float2half_rn(__half2float(value) / scale));
}

__global__ void fp8_e5m2_cache_write_kernel(
    const __half* __restrict__ key, const __half* __restrict__ value,
    uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
    const int64_t* __restrict__ locations, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, int row_dim, int64_t key_stride,
    int64_t value_stride, int64_t key_cache_stride,
    int64_t value_cache_stride) {
  const int token = blockIdx.x;
  const int64_t location = locations[token];
  if (location < 0) {
    return;
  }
  const float ks = k_scale == nullptr ? 1.f : *k_scale;
  const float vs = v_scale == nullptr ? 1.f : *v_scale;
  const __half* key_row = key + static_cast<int64_t>(token) * key_stride;
  const __half* value_row = value + static_cast<int64_t>(token) * value_stride;
  uint8_t* key_out = key_cache + location * key_cache_stride;
  uint8_t* value_out = value_cache + location * value_cache_stride;

  // One thread converts eight adjacent values. Hkv=1,D=256 therefore uses
  // exactly one warp instead of the eight mostly-idle warps in the old path.
  for (int base = threadIdx.x * 8; base < row_dim;
       base += blockDim.x * 8) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int element = base + i;
      if (element < row_dim) {
        key_out[element] = fp16_bits_to_e5m2_satfinite_rn(
            scaled_half_bits(key_row[element], ks));
        value_out[element] = fp16_bits_to_e5m2_satfinite_rn(
            scaled_half_bits(value_row[element], vs));
      }
    }
  }
}

}  // namespace

void sm70_fp8_e5m2_cache_write(
    torch::Tensor key, torch::Tensor value, torch::Tensor key_cache,
    torch::Tensor value_cache, torch::Tensor locations,
    std::optional<torch::Tensor> k_scale,
    std::optional<torch::Tensor> v_scale) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda() && key_cache.is_cuda() &&
                  value_cache.is_cuda() && locations.is_cuda(),
              "E5M2 cache writer tensors must be CUDA tensors");
  TORCH_CHECK(key.scalar_type() == torch::kFloat16 &&
                  value.scalar_type() == torch::kFloat16,
              "E5M2 cache writer inputs must be FP16");
  TORCH_CHECK(key_cache.scalar_type() == torch::kUInt8 &&
                  value_cache.scalar_type() == torch::kUInt8,
              "E5M2 cache writer destinations must be uint8 storage");
  TORCH_CHECK(locations.scalar_type() == torch::kInt64 && locations.dim() == 1,
              "E5M2 cache writer locations must be one-dimensional int64");
  TORCH_CHECK(key.sizes() == value.sizes() && key.size(0) == locations.numel(),
              "E5M2 cache writer input shape mismatch");
  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
              "E5M2 cache writer cache shape mismatch");
  TORCH_CHECK(key.dim() >= 2 && key_cache.dim() >= 2 && key.stride(-1) == 1 &&
                  value.stride(-1) == 1 && key_cache.stride(-1) == 1 &&
                  value_cache.stride(-1) == 1,
              "E5M2 cache writer rows must have contiguous inner dimensions");

  const int64_t row_dim64 = key.numel() / key.size(0);
  const int64_t cache_row_dim = key_cache.numel() / key_cache.size(0);
  TORCH_CHECK(row_dim64 == cache_row_dim && row_dim64 <= INT_MAX,
              "E5M2 cache writer row dimensions must match");
  const int row_dim = static_cast<int>(row_dim64);
  const int vectors = (row_dim + 7) / 8;
  const int threads = std::min(512, std::max(32, ((vectors + 31) / 32) * 32));
  const float* k_scale_ptr = k_scale.has_value()
      ? k_scale->data_ptr<float>()
      : nullptr;
  const float* v_scale_ptr = v_scale.has_value()
      ? v_scale->data_ptr<float>()
      : nullptr;

  c10::cuda::CUDAGuard guard(key.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  fp8_e5m2_cache_write_kernel<<<key.size(0), threads, 0, stream>>>(
      reinterpret_cast<const __half*>(key.data_ptr()),
      reinterpret_cast<const __half*>(value.data_ptr()),
      key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
      locations.data_ptr<int64_t>(), k_scale_ptr, v_scale_ptr, row_dim,
      key.stride(0), value.stride(0), key_cache.stride(0),
      value_cache.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
