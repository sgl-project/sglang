#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kPackedDim = 256;
constexpr int kScaleDim = 16;
constexpr int kTailDim = 64;
constexpr int kHeadDim = kPackedDim * 2;
constexpr int kOutputDim = kHeadDim + kTailDim;

__device__ __forceinline__ float decode_e2m1(uint8_t code) {
  constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float value = values[code & 0x7];
  return (code & 0x8) ? -value : value;
}

__device__ __forceinline__ uint16_t decode_e2m1_scaled_bf16_bits(
    uint8_t code, uint8_t scale_byte) {
  const uint16_t sign = static_cast<uint16_t>(code & 0x8) << 12;
  const int magnitude = code & 0x7;
  if (magnitude == 0 || scale_byte == 0) return sign;

  // E2M1 magnitudes have either 1.0 or 1.5 as their significand. Multiplying
  // by an E8M0 power of two therefore only changes the BF16 exponent.
  const int output_exponent = static_cast<int>(scale_byte) + (magnitude >> 1) - 1;
  if (output_exponent <= 0) {
    // The only reachable nonzero underflow is 0.5 * 2^-126 = 2^-127.
    return sign | 0x0040;
  }
  if (output_exponent >= 255) return sign | 0x7f80;

  const uint16_t mantissa =
      magnitude > 1 && (magnitude & 1) ? 0x0040 : 0;
  return sign | static_cast<uint16_t>(output_exponent << 7) | mantissa;
}

__device__ __forceinline__ uint8_t encode_e2m1(float value) {
  const float magnitude = fminf(fabsf(value), 6.0f);
  uint8_t code = 0;
  code += magnitude > 0.25f;
  code += magnitude > 0.75f;
  code += magnitude > 1.25f;
  code += magnitude > 1.75f;
  code += magnitude > 2.5f;
  code += magnitude > 3.5f;
  code += magnitude > 5.0f;
  return code | (value < 0.0f ? 0x8 : 0x0);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ float reciprocal_e8m0(int scale_byte) {
  // E8M0 byte 254 represents 2^127. Its reciprocal is the FP32
  // subnormal 2^-127; all other valid scales have a normal reciprocal.
  const int bits =
      scale_byte == 254 ? 0x00400000 : (254 - scale_byte) << 23;
  return __int_as_float(bits);
}

__device__ __forceinline__ void quantize_mxfp4_group(
    const __nv_bfloat16* __restrict__ row_input,
    uint8_t* __restrict__ row_data_output,
    uint8_t* __restrict__ row_scale_output,
    int group,
    int lane) {
  const float value = __bfloat162float(row_input[group * 32 + lane]);
  const float amax = warp_max(fabsf(value));
  int scale_byte = 0;
  float inv_scale = 0.0f;
  if (lane == 0 && amax != 0.0f) {
    // ceil(log2(amax / 6)) is either e - 2 or e - 1. The latter
    // applies only when amax is above 1.5 * 2^e.
    const int amax_bits = __float_as_int(amax);
    const int exponent_bits = (amax_bits >> 23) & 0xff;
    if (exponent_bits == 0xff) {
      scale_byte = 254;
    } else if (exponent_bits == 0) {
      scale_byte = 1;
    } else {
      const int exponent = exponent_bits - 127;
      const float power_of_two = __int_as_float(exponent_bits << 23);
      const int scale_exponent =
          amax > power_of_two * 1.5f ? exponent - 1 : exponent - 2;
      scale_byte = scale_exponent + 127;
    }
    scale_byte = scale_byte < 1 ? 1 : scale_byte;
    scale_byte = scale_byte > 254 ? 254 : scale_byte;
    inv_scale = reciprocal_e8m0(scale_byte);
  }
  scale_byte = __shfl_sync(0xffffffff, scale_byte, 0);
  inv_scale = __shfl_sync(0xffffffff, inv_scale, 0);
  const int code = static_cast<int>(encode_e2m1(value * inv_scale));
  const int pair_lane = (lane & 15) << 1;
  const int low = __shfl_sync(0xffffffff, code, pair_lane);
  const int high = __shfl_sync(0xffffffff, code, pair_lane + 1);
  if (lane < 16) {
    row_data_output[group * 16 + lane] =
        static_cast<uint8_t>(low | (high << 4));
  }
  if (lane == 0) {
    row_scale_output[group] = static_cast<uint8_t>(scale_byte);
    }
}
template <int WarpsPerToken, bool LocationsInt64>
__global__ void mxfp4_quantize_and_store_paged_parallel_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ tail_input,
    const void* __restrict__ locations,
    uint8_t* __restrict__ data_output,
    uint8_t* __restrict__ scale_output,
    __nv_bfloat16* __restrict__ tail_output,
    int num_rows,
    int64_t input_stride_token,
    int64_t tail_input_stride_token,
    int64_t data_output_stride_token,
    int64_t scale_output_stride_token,
    int64_t tail_output_stride_token) {
  const int source_row = blockIdx.x;
  if (source_row >= num_rows) return;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  int output_row = 0;
  if (lane == 0) {
    if constexpr (LocationsInt64) {
      output_row = static_cast<int>(
          reinterpret_cast<const int64_t*>(locations)[source_row]);
    } else {
      output_row =
          static_cast<int>(reinterpret_cast<const int32_t*>(locations)[source_row]);
    }
  }
  output_row = __shfl_sync(0xffffffff, output_row, 0);

  const __nv_bfloat16* row_input =
      input + static_cast<int64_t>(source_row) * input_stride_token;
  const __nv_bfloat16* row_tail_input =
      tail_input + static_cast<int64_t>(source_row) * tail_input_stride_token;
  uint8_t* row_data_output =
      data_output + static_cast<int64_t>(output_row) * data_output_stride_token;
  uint8_t* row_scale_output =
      scale_output + static_cast<int64_t>(output_row) * scale_output_stride_token;
  __nv_bfloat16* row_tail_output =
      tail_output + static_cast<int64_t>(output_row) * tail_output_stride_token;

#pragma unroll
  for (int group = warp; group < kScaleDim; group += WarpsPerToken) {
    quantize_mxfp4_group(
        row_input, row_data_output, row_scale_output, group, lane);
  }

  if (warp == 0) {
    reinterpret_cast<uint32_t*>(row_tail_output)[lane] =
        reinterpret_cast<const uint32_t*>(row_tail_input)[lane];
  }
}

template <int WarpsPerBlock>
__global__ void mxfp4_dequantize_paged_kernel(
    const uint8_t* __restrict__ data,
    const uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ tail,
    const int32_t* __restrict__ locations,
    __nv_bfloat16* __restrict__ output,
    int32_t* __restrict__ compact_page_table,
    int num_rows,
    int64_t data_stride_token,
    int64_t scale_stride_token,
    int64_t tail_stride_token,
    int64_t output_stride_token) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int output_row = blockIdx.x * WarpsPerBlock + warp;
  if (output_row >= num_rows) return;

  int source_row = 0;
  if (lane == 0) source_row = locations[output_row];
  source_row = __shfl_sync(0xffffffff, source_row, 0);
  const bool valid = source_row >= 0;
  source_row = valid ? source_row : 0;

  if (lane == 0) compact_page_table[output_row] = valid ? output_row : -1;

  const uint8_t* row_data = data + static_cast<int64_t>(source_row) * data_stride_token;
  const uint8_t* row_scales = scales + static_cast<int64_t>(source_row) * scale_stride_token;
  const __nv_bfloat16* row_tail = tail + static_cast<int64_t>(source_row) * tail_stride_token;
  __nv_bfloat16* row_output = output + static_cast<int64_t>(output_row) * output_stride_token;
  auto* packed_output = reinterpret_cast<__nv_bfloat162*>(row_output);

#pragma unroll
  for (int packed_idx = lane; packed_idx < kPackedDim; packed_idx += 32) {
    const uint8_t packed = row_data[packed_idx];
    const uint8_t scale_byte = row_scales[packed_idx >> 4];
    const uint32_t lo = decode_e2m1_scaled_bf16_bits(packed & 0xf, scale_byte);
    const uint32_t hi = decode_e2m1_scaled_bf16_bits(packed >> 4, scale_byte);
    reinterpret_cast<uint32_t*>(packed_output)[packed_idx] = lo | (hi << 16);
  }

  reinterpret_cast<uint32_t*>(row_output + kHeadDim)[lane] =
      reinterpret_cast<const uint32_t*>(row_tail)[lane];
}

template <int WarpsPerBlock, bool CopyTail = true>
__global__ void mxfp4_dequantize_paged_dedup_kernel(
    const uint8_t* __restrict__ data,
    const uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ tail,
    const int32_t* __restrict__ locations,
    __nv_bfloat16* __restrict__ output,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ claimed,
    int num_occurrences,
    int64_t data_stride_token,
    int64_t scale_stride_token,
    int64_t tail_stride_token,
    int64_t output_stride_token) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int occurrence = blockIdx.x * WarpsPerBlock + warp;
  if (occurrence >= num_occurrences) return;

  int source_row = 0;
  int should_write = 0;
  if (lane == 0) {
    source_row = locations[occurrence];
    page_table[occurrence] = source_row;
    if (source_row >= 0) should_write = atomicCAS(claimed + source_row, 0, 1) == 0;
  }
  source_row = __shfl_sync(0xffffffff, source_row, 0);
  should_write = __shfl_sync(0xffffffff, should_write, 0);
  if (!should_write) return;

  const uint8_t* row_data = data + static_cast<int64_t>(source_row) * data_stride_token;
  const uint8_t* row_scales = scales + static_cast<int64_t>(source_row) * scale_stride_token;
  __nv_bfloat16* row_output = output + static_cast<int64_t>(source_row) * output_stride_token;
  auto* packed_output = reinterpret_cast<__nv_bfloat162*>(row_output);

#pragma unroll
  for (int packed_idx = lane; packed_idx < kPackedDim; packed_idx += 32) {
    const uint8_t packed = row_data[packed_idx];
    const uint8_t scale_byte = row_scales[packed_idx >> 4];
    const uint32_t lo = decode_e2m1_scaled_bf16_bits(packed & 0xf, scale_byte);
    const uint32_t hi = decode_e2m1_scaled_bf16_bits(packed >> 4, scale_byte);
    reinterpret_cast<uint32_t*>(packed_output)[packed_idx] = lo | (hi << 16);
  }

  if constexpr (CopyTail) {
    const __nv_bfloat16* row_tail = tail + static_cast<int64_t>(source_row) * tail_stride_token;
    reinterpret_cast<uint32_t*>(row_output + kHeadDim)[lane] =
        reinterpret_cast<const uint32_t*>(row_tail)[lane];
  }
}

template <int WarpsPerBlock>
void mxfp4_dequantize_paged(
    tvm::ffi::TensorView data,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView tail,
    tvm::ffi::TensorView locations,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView compact_page_table) {
  using namespace host;

  auto CacheRows = SymbolicSize{"cache_rows"};
  auto OutputRows = SymbolicSize{"output_rows"};
  auto DataStride = SymbolicSize{"data_stride"};
  auto ScaleStride = SymbolicSize{"scale_stride"};
  auto TailStride = SymbolicSize{"tail_stride"};
  auto OutputStride = SymbolicSize{"output_stride"};
  auto device = SymbolicDevice{};

  TensorMatcher({CacheRows, 1, kPackedDim})
      .with_strides({DataStride, kPackedDim, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(data);
  TensorMatcher({CacheRows, 1, kScaleDim})
      .with_strides({ScaleStride, kScaleDim, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(scales);
  TensorMatcher({CacheRows, 1, kTailDim})
      .with_strides({TailStride, kTailDim, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(tail);
  TensorMatcher({OutputRows})
      .with_strides({1})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(locations)
      .verify(compact_page_table);
  TensorMatcher({OutputRows, 1, kOutputDim})
      .with_strides({OutputStride, kOutputDim, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(output);

  RuntimeCheck(
      OutputRows.unwrap() > 0,
      "MXFP4 dequantization requires at least one output row");
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(tail.data_ptr()) % 4 == 0,
      "MXFP4 tail must be 4-byte aligned");
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(output.data_ptr()) % 4 == 0,
      "MXFP4 output must be 4-byte aligned");

  const int num_rows = static_cast<int>(OutputRows.unwrap());
  const int block_size = WarpsPerBlock * 32;
  const int grid_size = div_ceil(num_rows, WarpsPerBlock);
  LaunchKernel(grid_size, block_size, device.unwrap())(
      mxfp4_dequantize_paged_kernel<WarpsPerBlock>,
      static_cast<const uint8_t*>(data.data_ptr()),
      static_cast<const uint8_t*>(scales.data_ptr()),
      static_cast<const __nv_bfloat16*>(tail.data_ptr()),
      static_cast<const int32_t*>(locations.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<int32_t*>(compact_page_table.data_ptr()),
      num_rows,
      DataStride.unwrap(),
      ScaleStride.unwrap(),
      TailStride.unwrap(),
      OutputStride.unwrap());
}

template <int WarpsPerToken, bool LocationsInt64>
void mxfp4_quantize_and_store_paged(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView tail_input,
    tvm::ffi::TensorView locations,
    tvm::ffi::TensorView data_output,
    tvm::ffi::TensorView scale_output,
    tvm::ffi::TensorView tail_output) {
  using namespace host;

  auto InputRows = SymbolicSize{"input_rows"};
  auto CacheRows = SymbolicSize{"cache_rows"};
  auto InputStride = SymbolicSize{"input_stride"};
  auto TailInputStride = SymbolicSize{"tail_input_stride"};
  auto DataOutputStride = SymbolicSize{"data_output_stride"};
  auto ScaleOutputStride = SymbolicSize{"scale_output_stride"};
  auto TailOutputStride = SymbolicSize{"tail_output_stride"};
  auto device = SymbolicDevice{};

  TensorMatcher({InputRows, 1, kHeadDim})
      .with_strides({InputStride, kHeadDim, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(input);
  TensorMatcher({InputRows, 1, kTailDim})
      .with_strides({TailInputStride, kTailDim, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(tail_input);
  if constexpr (LocationsInt64) {
    TensorMatcher({InputRows})
        .with_strides({1})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(locations);
  } else {
    TensorMatcher({InputRows})
        .with_strides({1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(locations);
  }
  TensorMatcher({CacheRows, 1, kPackedDim})
      .with_strides({DataOutputStride, kPackedDim, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(data_output);
  TensorMatcher({CacheRows, 1, kScaleDim})
      .with_strides({ScaleOutputStride, kScaleDim, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(scale_output);
  TensorMatcher({CacheRows, 1, kTailDim})
      .with_strides({TailOutputStride, kTailDim, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(tail_output);

  RuntimeCheck(
      InputRows.unwrap() > 0,
      "MXFP4 quantization requires at least one input row");
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(tail_input.data_ptr()) % 4 == 0,
      "MXFP4 tail input must be 4-byte aligned");
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(tail_output.data_ptr()) % 4 == 0,
      "MXFP4 tail output must be 4-byte aligned");

  const int num_rows = static_cast<int>(InputRows.unwrap());
  const int block_size = WarpsPerToken * 32;
  LaunchKernel(num_rows, block_size, device.unwrap())(
      mxfp4_quantize_and_store_paged_parallel_kernel<
          WarpsPerToken, LocationsInt64>,
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<const __nv_bfloat16*>(tail_input.data_ptr()),
      locations.data_ptr(),
      static_cast<uint8_t*>(data_output.data_ptr()),
      static_cast<uint8_t*>(scale_output.data_ptr()),
      static_cast<__nv_bfloat16*>(tail_output.data_ptr()),
      num_rows,
      InputStride.unwrap(),
      TailInputStride.unwrap(),
      DataOutputStride.unwrap(),
      ScaleOutputStride.unwrap(),
      TailOutputStride.unwrap());
}

template <int WarpsPerBlock, bool CopyTail = true>
void mxfp4_dequantize_paged_dedup(
    tvm::ffi::TensorView data,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView tail,
    tvm::ffi::TensorView locations,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView page_table,
    tvm::ffi::TensorView claimed) {
  using namespace host;

  auto CacheRows = SymbolicSize{"cache_rows"};
  auto Occurrences = SymbolicSize{"occurrences"};
  auto DataStride = SymbolicSize{"data_stride"};
  auto ScaleStride = SymbolicSize{"scale_stride"};
  auto TailStride = SymbolicSize{"tail_stride"};
  auto OutputStride = SymbolicSize{"output_stride"};
  auto device = SymbolicDevice{};

  TensorMatcher({CacheRows, 1, kPackedDim})
      .with_strides({DataStride, kPackedDim, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(data);
  TensorMatcher({CacheRows, 1, kScaleDim})
      .with_strides({ScaleStride, kScaleDim, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(scales);
  TensorMatcher({CacheRows, 1, kTailDim})
      .with_strides({TailStride, kTailDim, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(tail);
  TensorMatcher({Occurrences})
      .with_strides({1})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(locations)
      .verify(page_table);
  constexpr int OutputDim = CopyTail ? kOutputDim : kHeadDim;
  TensorMatcher({CacheRows, 1, OutputDim})
      .with_strides({OutputStride, OutputDim, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(output);
  TensorMatcher({CacheRows})
      .with_strides({1})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(claimed);

  const int num_occurrences = static_cast<int>(Occurrences.unwrap());
  const int block_size = WarpsPerBlock * 32;
  const int grid_size = div_ceil(num_occurrences, WarpsPerBlock);
  LaunchKernel(grid_size, block_size, device.unwrap())(
      mxfp4_dequantize_paged_dedup_kernel<WarpsPerBlock, CopyTail>,
      static_cast<const uint8_t*>(data.data_ptr()),
      static_cast<const uint8_t*>(scales.data_ptr()),
      static_cast<const __nv_bfloat16*>(tail.data_ptr()),
      static_cast<const int32_t*>(locations.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<int32_t*>(page_table.data_ptr()),
      static_cast<int32_t*>(claimed.data_ptr()),
      num_occurrences,
      DataStride.unwrap(),
      ScaleStride.unwrap(),
      TailStride.unwrap(),
      OutputStride.unwrap());
}

}  // namespace
