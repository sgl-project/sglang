// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>

namespace {

constexpr int kQuantBlock = 32;
constexpr int kBlockBytes = 17;
constexpr int kWarpsPerBlock = 4;
constexpr int kRowsPerWarp = 4;
constexpr int kMarlinTileK = 16;
constexpr int kMarlinTileN = 64;
constexpr int kMarlinTileWords = 128;

__device__ __forceinline__ float fp4_value(uint8_t value) {
  constexpr float table[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return table[value & 0x0f];
}

template <typename scalar_t>
__device__ __forceinline__ float load_scalar(const scalar_t* input, int index);

template <>
__device__ __forceinline__ float load_scalar<__nv_bfloat16>(const __nv_bfloat16* input, int index) {
  return __bfloat162float(input[index]);
}

template <>
__device__ __forceinline__ float load_scalar<half>(const half* input, int index) {
  return __half2float(input[index]);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_scalar(float value);

template <>
__device__ __forceinline__ __nv_bfloat16 store_scalar<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ half store_scalar<half>(float value) {
  return __float2half_rn(value);
}

template <typename scalar_t>
__global__ void mxfp4_matvec_kernel(
    const scalar_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    int64_t cache_stride,
    const int32_t* __restrict__ slot_ids,
    int64_t role_offset,
    int input_size,
    int output_size,
    int records,
    int records_per_input,
    scalar_t* __restrict__ output) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int output_row_base = (blockIdx.x * kWarpsPerBlock + warp) * kRowsPerWarp;
  const int record = blockIdx.y;
  if (record >= records || output_row_base >= output_size) {
    return;
  }

  const int input_row = record / records_per_input;
  const scalar_t* input_ptr = input + static_cast<int64_t>(input_row) * input_size;
  const int blocks_per_row = input_size / kQuantBlock;
  const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * kBlockBytes;
  const int32_t slot = slot_ids[record];
  const uint8_t* weight_base = cache + static_cast<int64_t>(slot) * cache_stride + role_offset;

  float sums[kRowsPerWarp] = {};
  for (int block = lane; block < blocks_per_row; block += 32) {
    const int input_base = block * kQuantBlock;
    float block_sums[kRowsPerWarp] = {};
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      const float input_low = load_scalar(input_ptr, input_base + index);
      const float input_high = load_scalar(input_ptr, input_base + index + 16);
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int output_row = output_row_base + row;
        if (output_row < output_size) {
          const uint8_t* quant =
              weight_base + static_cast<int64_t>(output_row) * row_bytes + static_cast<int64_t>(block) * kBlockBytes;
          const uint8_t packed = quant[index + 1];
          block_sums[row] = fmaf(input_low, fp4_value(packed), block_sums[row]);
          block_sums[row] = fmaf(input_high, fp4_value(packed >> 4), block_sums[row]);
        }
      }
    }
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const int output_row = output_row_base + row;
      if (output_row < output_size) {
        const uint8_t* quant =
            weight_base + static_cast<int64_t>(output_row) * row_bytes + static_cast<int64_t>(block) * kBlockBytes;
        const int exponent = static_cast<int>(quant[0]) - 127;
        sums[row] = fmaf(block_sums[row], ldexpf(1.0f, exponent), sums[row]);
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      sums[row] += __shfl_down_sync(0xffffffffu, sums[row], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const int output_row = output_row_base + row;
      if (output_row < output_size) {
        output[static_cast<int64_t>(record) * output_size + output_row] = store_scalar<scalar_t>(sums[row]);
      }
    }
  }
}

// Compute gate and up together so the hidden-state vector is loaded once.
template <typename scalar_t>
__global__ void mxfp4_matvec_dual_kernel(
    const scalar_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    int64_t cache_stride,
    const int32_t* __restrict__ slot_ids,
    int64_t role_offset_a,
    int64_t role_offset_b,
    int input_size,
    int output_size,
    int records,
    int records_per_input,
    scalar_t* __restrict__ output_a,
    scalar_t* __restrict__ output_b) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int output_row_base = (blockIdx.x * kWarpsPerBlock + warp) * kRowsPerWarp;
  const int record = blockIdx.y;
  if (record >= records || output_row_base >= output_size) {
    return;
  }

  const int input_row = record / records_per_input;
  const scalar_t* input_ptr = input + static_cast<int64_t>(input_row) * input_size;
  const int blocks_per_row = input_size / kQuantBlock;
  const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * kBlockBytes;
  const int32_t slot = slot_ids[record];
  const uint8_t* weight_base_a = cache + static_cast<int64_t>(slot) * cache_stride + role_offset_a;
  const uint8_t* weight_base_b = cache + static_cast<int64_t>(slot) * cache_stride + role_offset_b;

  float sums_a[kRowsPerWarp] = {};
  float sums_b[kRowsPerWarp] = {};
  for (int block = lane; block < blocks_per_row; block += 32) {
    const int input_base = block * kQuantBlock;
    float block_sums_a[kRowsPerWarp] = {};
    float block_sums_b[kRowsPerWarp] = {};
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      const float input_low = load_scalar(input_ptr, input_base + index);
      const float input_high = load_scalar(input_ptr, input_base + index + 16);
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int output_row = output_row_base + row;
        if (output_row < output_size) {
          const uint8_t* quant_a =
              weight_base_a + static_cast<int64_t>(output_row) * row_bytes + static_cast<int64_t>(block) * kBlockBytes;
          const uint8_t* quant_b =
              weight_base_b + static_cast<int64_t>(output_row) * row_bytes + static_cast<int64_t>(block) * kBlockBytes;
          const uint8_t packed_a = quant_a[index + 1];
          const uint8_t packed_b = quant_b[index + 1];
          block_sums_a[row] = fmaf(input_low, fp4_value(packed_a), block_sums_a[row]);
          block_sums_a[row] = fmaf(input_high, fp4_value(packed_a >> 4), block_sums_a[row]);
          block_sums_b[row] = fmaf(input_low, fp4_value(packed_b), block_sums_b[row]);
          block_sums_b[row] = fmaf(input_high, fp4_value(packed_b >> 4), block_sums_b[row]);
        }
      }
    }
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const int output_row = output_row_base + row;
      if (output_row < output_size) {
        const uint8_t* quant_a =
            weight_base_a + static_cast<int64_t>(output_row) * row_bytes + static_cast<int64_t>(block) * kBlockBytes;
        const uint8_t* quant_b =
            weight_base_b + static_cast<int64_t>(output_row) * row_bytes + static_cast<int64_t>(block) * kBlockBytes;
        const int exponent_a = static_cast<int>(quant_a[0]) - 127;
        const int exponent_b = static_cast<int>(quant_b[0]) - 127;
        sums_a[row] = fmaf(block_sums_a[row], ldexpf(1.0f, exponent_a), sums_a[row]);
        sums_b[row] = fmaf(block_sums_b[row], ldexpf(1.0f, exponent_b), sums_b[row]);
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      sums_a[row] += __shfl_down_sync(0xffffffffu, sums_a[row], offset);
      sums_b[row] += __shfl_down_sync(0xffffffffu, sums_b[row], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const int output_row = output_row_base + row;
      if (output_row < output_size) {
        output_a[static_cast<int64_t>(record) * output_size + output_row] = store_scalar<scalar_t>(sums_a[row]);
        output_b[static_cast<int64_t>(record) * output_size + output_row] = store_scalar<scalar_t>(sums_b[row]);
      }
    }
  }
}

__device__ __forceinline__ uint32_t load_raw_word(
    const uint8_t* raw, int64_t cache_stride, int slot, int role_offset, int row, int blocks_per_row, int packed_word) {
  const int block = packed_word / 4;
  const int word_in_block = packed_word & 3;
  const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * kBlockBytes;
  const uint8_t* ptr = raw + static_cast<int64_t>(slot) * cache_stride + role_offset +
                       static_cast<int64_t>(row) * row_bytes + static_cast<int64_t>(block) * kBlockBytes + 1 +
                       word_in_block * 4;
  return static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) | (static_cast<uint32_t>(ptr[2]) << 16) |
         (static_cast<uint32_t>(ptr[3]) << 24);
}

__device__ __forceinline__ uint8_t load_raw_scale(
    const uint8_t* raw, int64_t cache_stride, int slot, int role_offset, int row, int blocks_per_row, int block) {
  const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * kBlockBytes;
  const uint8_t* ptr = raw + static_cast<int64_t>(slot) * cache_stride + role_offset +
                       static_cast<int64_t>(row) * row_bytes + static_cast<int64_t>(block) * kBlockBytes;
  return *ptr;
}

__device__ __forceinline__ uint8_t marlin_scale_perm(int index) {
  constexpr int local_perm[4] = {0, 2, 1, 3};
  const int interleaved = (index / 4) * 4 + local_perm[index & 3];
  return static_cast<uint8_t>(((interleaved & 7) * 8) + (interleaved >> 3));
}

__device__ __forceinline__ uint8_t marlin_nibble(uint32_t word, int value_index) {
  return static_cast<uint8_t>((word >> ((value_index & 7) * 4)) & 0x0f);
}

__global__ void mxfp4_marlin_repack_weight_kernel(
    const uint8_t* __restrict__ raw,
    int64_t raw_stride,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ target_slots,
    int64_t role_bytes,
    int input_size,
    int output_size,
    bool gate_up,
    int32_t* __restrict__ output,
    int64_t output_stride) {
  const int batch = blockIdx.y;
  const int64_t total_words = static_cast<int64_t>(input_size / kMarlinTileK) * (output_size * 2);
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (batch >= gridDim.y || index >= total_words) return;

  const int64_t tile_span = static_cast<int64_t>(output_size / kMarlinTileN) * kMarlinTileWords;
  const int tile_k = static_cast<int>(index / tile_span);
  const int64_t tile_rem = index % tile_span;
  const int tile_n = static_cast<int>(tile_rem / kMarlinTileWords);
  const int local = static_cast<int>(tile_rem % kMarlinTileWords);
  const int warp = local & 3;
  const int thread = local >> 2;
  const int cur_n = warp * 16 + thread / 4;
  const int tc_row = (thread & 3) * 2;
  constexpr int offsets[4] = {0, 1, 8, 9};
  constexpr int pack_index[8] = {0, 2, 4, 6, 1, 3, 5, 7};

  const int source_slot = source_slots[batch];
  const int target_slot = target_slots[batch];
  const int rows_per_role = gate_up ? output_size / 2 : output_size;
  const int blocks_per_row = input_size / kQuantBlock;
  const int role0_offset = 0;
  const int role1_offset = static_cast<int>(role_bytes);
  const int role2_offset = static_cast<int>(2 * role_bytes);
  uint8_t values[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int value_index = tc_row + offsets[i];
    const int source_row = tile_n * kMarlinTileN + cur_n;
    const int role = gate_up && source_row >= rows_per_role ? 1 : (gate_up ? 0 : 2);
    const int row = gate_up ? source_row % rows_per_role : source_row;
    const int role_offset = role == 0 ? role0_offset : (role == 1 ? role1_offset : role2_offset);
    const uint32_t word =
        load_raw_word(raw, raw_stride, source_slot, role_offset, row, blocks_per_row, tile_k * 2 + value_index / 8);
    values[i] = marlin_nibble(word, value_index);
    const int high_source_row = tile_n * kMarlinTileN + cur_n + 8;
    const int high_role = gate_up && high_source_row >= rows_per_role ? 1 : (gate_up ? 0 : 2);
    const int high_role_offset = high_role == 0 ? role0_offset : (high_role == 1 ? role1_offset : role2_offset);
    const int high_row = gate_up ? high_source_row % rows_per_role : high_source_row;
    const uint32_t high_word = load_raw_word(
        raw, raw_stride, source_slot, high_role_offset, high_row, blocks_per_row, tile_k * 2 + value_index / 8);
    values[4 + i] = marlin_nibble(high_word, value_index);
  }

  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    packed |= static_cast<uint32_t>(values[pack_index[i]]) << (i * 4);
  }
  output[static_cast<int64_t>(target_slot) * output_stride + index] = static_cast<int32_t>(packed);
}

__global__ void mxfp4_marlin_repack_scale_kernel(
    const uint8_t* __restrict__ raw,
    int64_t raw_stride,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ target_slots,
    int64_t role_bytes,
    int input_size,
    int output_size,
    bool gate_up,
    uint8_t* __restrict__ output,
    int64_t output_stride) {
  const int batch = blockIdx.y;
  const int groups = input_size / kQuantBlock;
  const int64_t total = static_cast<int64_t>(groups) * output_size;
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (batch >= gridDim.y || index >= total) return;

  const int group = static_cast<int>(index / output_size);
  const int column = static_cast<int>(index % output_size);
  const int source_column = (column / 64) * 64 + marlin_scale_perm(column & 63);
  const int rows_per_role = gate_up ? output_size / 2 : output_size;
  const int role = gate_up && source_column >= rows_per_role ? 1 : (gate_up ? 0 : 2);
  const int row = gate_up ? source_column % rows_per_role : source_column;
  const int role_offset = role == 0 ? 0 : (role == 1 ? static_cast<int>(role_bytes) : static_cast<int>(2 * role_bytes));
  const uint8_t value = load_raw_scale(raw, raw_stride, source_slots[batch], role_offset, row, groups, group);
  output[static_cast<int64_t>(target_slots[batch]) * output_stride + index] = value;
}

void mxfp4_marlin_repack(
    torch::Tensor raw,
    torch::Tensor source_slots,
    torch::Tensor target_slots,
    int64_t role_bytes,
    int64_t hidden_size,
    int64_t intermediate_size,
    torch::Tensor w13,
    torch::Tensor w2,
    torch::Tensor w13_scale,
    torch::Tensor w2_scale) {
  TORCH_CHECK(raw.is_cuda() && source_slots.is_cuda() && target_slots.is_cuda(), "repack inputs must be CUDA tensors");
  TORCH_CHECK(raw.scalar_type() == at::kByte && raw.dim() == 2, "raw cache must be a uint8 matrix");
  TORCH_CHECK(
      source_slots.scalar_type() == at::kInt && target_slots.scalar_type() == at::kInt, "slot ids must be int32");
  TORCH_CHECK(source_slots.numel() == target_slots.numel(), "slot id size mismatch");
  TORCH_CHECK(w13.scalar_type() == at::kInt && w2.scalar_type() == at::kInt, "Marlin weights must be int32");
  TORCH_CHECK(
      w13_scale.scalar_type() == at::kByte && w2_scale.scalar_type() == at::kByte,
      "Marlin scales must be uint8 storage");
  TORCH_CHECK(hidden_size % 32 == 0 && intermediate_size % 32 == 0, "MXFP4 dimensions must be divisible by 32");
  const int batch = static_cast<int>(source_slots.numel());
  if (batch == 0) return;
  const int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int w13_n = static_cast<int>(2 * intermediate_size);
  const int w2_n = static_cast<int>(hidden_size);
  const int w13_k = static_cast<int>(hidden_size);
  const int w2_k = static_cast<int>(intermediate_size);
  const int64_t w13_words = static_cast<int64_t>(w13_k / kMarlinTileK) * w13_n * 2;
  const int64_t w2_words = static_cast<int64_t>(w2_k / kMarlinTileK) * w2_n * 2;
  const int64_t w13_scales = static_cast<int64_t>(w13_k / kQuantBlock) * w13_n;
  const int64_t w2_scales = static_cast<int64_t>(w2_k / kQuantBlock) * w2_n;
  mxfp4_marlin_repack_weight_kernel<<<dim3((w13_words + threads - 1) / threads, batch), threads, 0, stream>>>(
      raw.data_ptr<uint8_t>(),
      raw.stride(0),
      source_slots.data_ptr<int32_t>(),
      target_slots.data_ptr<int32_t>(),
      role_bytes,
      w13_k,
      w13_n,
      true,
      w13.data_ptr<int32_t>(),
      w13.stride(0));
  mxfp4_marlin_repack_weight_kernel<<<dim3((w2_words + threads - 1) / threads, batch), threads, 0, stream>>>(
      raw.data_ptr<uint8_t>(),
      raw.stride(0),
      source_slots.data_ptr<int32_t>(),
      target_slots.data_ptr<int32_t>(),
      role_bytes,
      w2_k,
      w2_n,
      false,
      w2.data_ptr<int32_t>(),
      w2.stride(0));
  mxfp4_marlin_repack_scale_kernel<<<dim3((w13_scales + threads - 1) / threads, batch), threads, 0, stream>>>(
      raw.data_ptr<uint8_t>(),
      raw.stride(0),
      source_slots.data_ptr<int32_t>(),
      target_slots.data_ptr<int32_t>(),
      role_bytes,
      w13_k,
      w13_n,
      true,
      w13_scale.data_ptr<uint8_t>(),
      w13_scale.stride(0));
  mxfp4_marlin_repack_scale_kernel<<<dim3((w2_scales + threads - 1) / threads, batch), threads, 0, stream>>>(
      raw.data_ptr<uint8_t>(),
      raw.stride(0),
      source_slots.data_ptr<int32_t>(),
      target_slots.data_ptr<int32_t>(),
      role_bytes,
      w2_k,
      w2_n,
      false,
      w2_scale.data_ptr<uint8_t>(),
      w2_scale.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor mxfp4_matvec(
    torch::Tensor input,
    torch::Tensor cache,
    torch::Tensor slot_ids,
    int64_t role_offset,
    int64_t role_bytes,
    int64_t input_size,
    int64_t output_size,
    int64_t records_per_input) {
  TORCH_CHECK(
      input.is_cuda() && cache.is_cuda() && slot_ids.is_cuda(), "input, cache, and slot_ids must be CUDA tensors");
  TORCH_CHECK(
      input.is_contiguous() && cache.is_contiguous() && slot_ids.is_contiguous(),
      "input, cache, and slot_ids must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf, "input must be BF16 or FP16");
  TORCH_CHECK(cache.scalar_type() == at::kByte && cache.dim() == 2, "cache must be a two-dimensional uint8 tensor");
  TORCH_CHECK(
      slot_ids.scalar_type() == at::kInt && slot_ids.dim() == 1, "slot_ids must be a one-dimensional int32 tensor");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == input_size, "input shape does not match input_size");
  TORCH_CHECK(input_size > 0 && input_size % kQuantBlock == 0, "input_size must be divisible by 32");
  TORCH_CHECK(records_per_input > 0, "records_per_input must be positive");
  TORCH_CHECK(
      slot_ids.numel() == input.size(0) * records_per_input,
      "slot count does not match input rows and records_per_input");
  const int64_t expected_role_bytes = output_size * (input_size / kQuantBlock) * kBlockBytes;
  TORCH_CHECK(role_bytes == expected_role_bytes, "role byte count does not match matrix dimensions");
  TORCH_CHECK(role_offset >= 0 && role_offset + role_bytes <= cache.size(1), "role range is outside each cache slot");

  const auto records = slot_ids.numel();
  auto output = torch::empty({records, output_size}, input.options());
  const dim3 block(kWarpsPerBlock * 32);
  const dim3 grid((output_size + kWarpsPerBlock * kRowsPerWarp - 1) / (kWarpsPerBlock * kRowsPerWarp), records);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (input.scalar_type() == at::kBFloat16) {
    mxfp4_matvec_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        cache.data_ptr<uint8_t>(),
        cache.stride(0),
        slot_ids.data_ptr<int32_t>(),
        role_offset,
        input_size,
        output_size,
        records,
        records_per_input,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()));
  } else {
    mxfp4_matvec_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr()),
        cache.data_ptr<uint8_t>(),
        cache.stride(0),
        slot_ids.data_ptr<int32_t>(),
        role_offset,
        input_size,
        output_size,
        records,
        records_per_input,
        reinterpret_cast<half*>(output.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> mxfp4_matvec_dual(
    torch::Tensor input,
    torch::Tensor cache,
    torch::Tensor slot_ids,
    int64_t role_offset_a,
    int64_t role_offset_b,
    int64_t role_bytes,
    int64_t input_size,
    int64_t output_size,
    int64_t records_per_input) {
  TORCH_CHECK(
      input.is_cuda() && cache.is_cuda() && slot_ids.is_cuda(), "input, cache, and slot_ids must be CUDA tensors");
  TORCH_CHECK(
      input.is_contiguous() && cache.is_contiguous() && slot_ids.is_contiguous(),
      "input, cache, and slot_ids must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf, "input must be BF16 or FP16");
  TORCH_CHECK(cache.scalar_type() == at::kByte && cache.dim() == 2, "cache must be a two-dimensional uint8 tensor");
  TORCH_CHECK(
      slot_ids.scalar_type() == at::kInt && slot_ids.dim() == 1, "slot_ids must be a one-dimensional int32 tensor");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == input_size, "input shape does not match input_size");
  TORCH_CHECK(input_size > 0 && input_size % kQuantBlock == 0, "input_size must be divisible by 32");
  TORCH_CHECK(records_per_input > 0, "records_per_input must be positive");
  TORCH_CHECK(
      slot_ids.numel() == input.size(0) * records_per_input,
      "slot count does not match input rows and records_per_input");
  const int64_t expected_role_bytes = output_size * (input_size / kQuantBlock) * kBlockBytes;
  TORCH_CHECK(role_bytes == expected_role_bytes, "role byte count does not match matrix dimensions");
  TORCH_CHECK(
      role_offset_a >= 0 && role_offset_a + role_bytes <= cache.size(1), "gate role range is outside each cache slot");
  TORCH_CHECK(
      role_offset_b >= 0 && role_offset_b + role_bytes <= cache.size(1), "up role range is outside each cache slot");

  const auto records = slot_ids.numel();
  auto output_a = torch::empty({records, output_size}, input.options());
  auto output_b = torch::empty({records, output_size}, input.options());
  const dim3 block(kWarpsPerBlock * 32);
  const dim3 grid((output_size + kWarpsPerBlock * kRowsPerWarp - 1) / (kWarpsPerBlock * kRowsPerWarp), records);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (input.scalar_type() == at::kBFloat16) {
    mxfp4_matvec_dual_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        cache.data_ptr<uint8_t>(),
        cache.stride(0),
        slot_ids.data_ptr<int32_t>(),
        role_offset_a,
        role_offset_b,
        input_size,
        output_size,
        records,
        records_per_input,
        reinterpret_cast<__nv_bfloat16*>(output_a.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output_b.data_ptr()));
  } else {
    mxfp4_matvec_dual_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr()),
        cache.data_ptr<uint8_t>(),
        cache.stride(0),
        slot_ids.data_ptr<int32_t>(),
        role_offset_a,
        role_offset_b,
        input_size,
        output_size,
        records,
        records_per_input,
        reinterpret_cast<half*>(output_a.data_ptr()),
        reinterpret_cast<half*>(output_b.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(output_a, output_b);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("mxfp4_matvec", &mxfp4_matvec, "GGUF MXFP4 matrix-vector multiply");
  module.def("mxfp4_matvec_dual", &mxfp4_matvec_dual, "GGUF MXFP4 gate/up matrix-vector multiply");
  module.def("mxfp4_marlin_repack", &mxfp4_marlin_repack, "Repack raw GGUF MXFP4 objects to Marlin layout");
}
