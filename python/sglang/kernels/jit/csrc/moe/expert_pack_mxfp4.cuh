// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <utility>

namespace sglang {

namespace {

constexpr int kQuantBlock = 32;
constexpr int kBlockBytes = 17;
constexpr int kWarpsPerBlock = 4;
constexpr int kRowsPerWarp = 4;

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
  const int output_row_base = (blockIdx.y * kWarpsPerBlock + warp) * kRowsPerWarp;
  const int record = blockIdx.x;
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
  const int output_row_base = (blockIdx.y * kWarpsPerBlock + warp) * kRowsPerWarp;
  const int record = blockIdx.x;
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

}  // namespace

// ---------------------------------------------------------------------------
// Host layer
// ---------------------------------------------------------------------------

namespace {

/// \brief What `verify_matvec_inputs` learned about a validated operand set.
struct MatvecOperands {
  int64_t records;
  int64_t cache_bytes;
  DLDataType dtype;
  DLDevice device;
  bool is_bf16;
};

/**
 * \brief Validate the operands shared by both matvec entry points.
 *
 * The check order is deliberate: `input_size` is the value every later
 * computation is derived from, so its divisibility is reported before the
 * role-byte arithmetic that a bad `input_size` would also make wrong.
 *
 * \return The record count, per-slot cache width, dtype, device, and which of
 *         the two supported element types was passed.
 */
auto verify_matvec_inputs(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView cache,
    tvm::ffi::TensorView slot_ids,
    int64_t role_bytes,
    int64_t input_size,
    int64_t output_size,
    int64_t records_per_input) -> MatvecOperands {
  using namespace host;

  auto rows = SymbolicSize{"input_rows"};
  auto records = SymbolicSize{"records"};
  auto slots = SymbolicSize{"cache_slots"};
  auto cache_bytes = SymbolicSize{"cache_bytes_per_slot"};
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};

  TensorMatcher({rows, input_size})  //
      .with_dtype<fp16_t, bf16_t>(dtype)
      .with_device<kDLCUDA>(device)
      .verify(input);

  CHECK_HOST(input_size > 0 && input_size % kQuantBlock == 0)
      << "input_size must be divisible by 32, got " << input_size;
  CHECK_HOST(records_per_input > 0) << "records_per_input must be positive, got " << records_per_input;

  TensorMatcher({slots, cache_bytes})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(cache);
  TensorMatcher({records})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(slot_ids);

  CHECK_HOST(records.unwrap() == rows.unwrap() * records_per_input)
      << "slot count does not match input rows and records_per_input: " << records.unwrap() << " != " << rows.unwrap()
      << " * " << records_per_input;

  const int64_t expected_role_bytes = output_size * (input_size / kQuantBlock) * kBlockBytes;
  CHECK_HOST(role_bytes == expected_role_bytes)
      << "role byte count does not match matrix dimensions: " << role_bytes << " != " << expected_role_bytes;

  return MatvecOperands{
      records.unwrap(), cache_bytes.unwrap(), dtype.unwrap(), device.unwrap(), dtype.is_type<bf16_t>()};
}

/// \brief Check that one role's byte range lies inside every cache slot.
void verify_role_range(int64_t role_offset, int64_t role_bytes, int64_t cache_bytes, const char* name) {
  CHECK_HOST(role_offset >= 0 && role_offset + role_bytes <= cache_bytes)
      << name << " role range is outside each cache slot: [" << role_offset << ", " << role_offset + role_bytes
      << ") not within [0, " << cache_bytes << ")";
}

/// \brief The launch geometry both matvec kernels use: one block column per record, one warp group per row tile.
///
/// `records` scales with the tokens in a forward pass and is the dimension that
/// grows without a useful bound, so it takes `grid.x` (capped at 2^31-1). The
/// row tiles take `grid.y`, whose 65535 cap only a matrix with more than a
/// million rows could reach -- checked rather than assumed.
auto matvec_launch_shape(int64_t output_size, int64_t records) -> std::pair<dim3, dim3> {
  constexpr uint32_t kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
  constexpr int64_t kMaxGridX = 2147483647;
  constexpr int64_t kMaxGridY = 65535;

  const int64_t row_tiles = host::div_ceil(output_size, static_cast<int64_t>(kRowsPerBlock));
  CHECK_HOST(records <= kMaxGridX) << "records exceeds the CUDA grid limit: " << records << " > " << kMaxGridX;
  CHECK_HOST(row_tiles <= kMaxGridY) << "output_size needs " << row_tiles << " row tiles, over the CUDA grid limit of "
                                     << kMaxGridY;

  const dim3 grid(static_cast<uint32_t>(records), static_cast<uint32_t>(row_tiles));
  const dim3 block(kWarpsPerBlock * device::kWarpThreads);
  return {grid, block};
}

/// \brief Verify an output tensor against the shape, dtype, and device of its inputs.
void verify_matvec_output(tvm::ffi::TensorView out, const MatvecOperands& operands, int64_t output_size) {
  using namespace host;
  TensorMatcher({operands.records, output_size})  //
      .with_dtype(operands.dtype)
      .with_device(operands.device)
      .verify(out);
}

template <typename scalar_t>
void launch_matvec(
    const MatvecOperands& operands,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView cache,
    tvm::ffi::TensorView slot_ids,
    int64_t role_offset,
    int64_t input_size,
    int64_t output_size,
    int64_t records_per_input) {
  const auto [grid, block] = matvec_launch_shape(output_size, operands.records);
  host::LaunchKernel(grid, block, operands.device)(
      mxfp4_matvec_kernel<scalar_t>,
      static_cast<const scalar_t*>(input.data_ptr()),
      static_cast<const uint8_t*>(cache.data_ptr()),
      cache.stride(0),
      static_cast<const int32_t*>(slot_ids.data_ptr()),
      role_offset,
      static_cast<int>(input_size),
      static_cast<int>(output_size),
      static_cast<int>(operands.records),
      static_cast<int>(records_per_input),
      static_cast<scalar_t*>(out.data_ptr()));
}

template <typename scalar_t>
void launch_matvec_dual(
    const MatvecOperands& operands,
    tvm::ffi::TensorView out_a,
    tvm::ffi::TensorView out_b,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView cache,
    tvm::ffi::TensorView slot_ids,
    int64_t role_offset_a,
    int64_t role_offset_b,
    int64_t input_size,
    int64_t output_size,
    int64_t records_per_input) {
  const auto [grid, block] = matvec_launch_shape(output_size, operands.records);
  host::LaunchKernel(grid, block, operands.device)(
      mxfp4_matvec_dual_kernel<scalar_t>,
      static_cast<const scalar_t*>(input.data_ptr()),
      static_cast<const uint8_t*>(cache.data_ptr()),
      cache.stride(0),
      static_cast<const int32_t*>(slot_ids.data_ptr()),
      role_offset_a,
      role_offset_b,
      static_cast<int>(input_size),
      static_cast<int>(output_size),
      static_cast<int>(operands.records),
      static_cast<int>(records_per_input),
      static_cast<scalar_t*>(out_a.data_ptr()),
      static_cast<scalar_t*>(out_b.data_ptr()));
}

}  // namespace

/**
 * \brief Multiply selected raw GGUF MXFP4 matrices by BF16/FP16 rows.
 *
 * \param out               Output, `[records, output_size]`, same dtype as `input`.
 * \param input             Hidden states, `[rows, input_size]`, FP16 or BF16.
 * \param cache             Raw MXFP4 slot bank, `[slots, bytes_per_slot]` uint8.
 * \param slot_ids          One slot per record, `[records]` int32.
 * \param role_offset       Byte offset of the matrix within each cache slot.
 * \param role_bytes        Byte size of one matrix; must match the dimensions.
 * \param input_size        Columns of `input`; must be divisible by 32.
 * \param output_size       Rows of the MXFP4 matrix.
 * \param records_per_input How many records share one row of `input`.
 */
inline void mxfp4_matvec(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView cache,
    tvm::ffi::TensorView slot_ids,
    int64_t role_offset,
    int64_t role_bytes,
    int64_t input_size,
    int64_t output_size,
    int64_t records_per_input) {
  const auto operands =
      verify_matvec_inputs(input, cache, slot_ids, role_bytes, input_size, output_size, records_per_input);
  verify_role_range(role_offset, role_bytes, operands.cache_bytes, "matrix");
  verify_matvec_output(out, operands, output_size);

  // An empty batch has nothing to compute, and `records` is a grid dimension.
  if (operands.records == 0) return;

  if (operands.is_bf16) {
    launch_matvec<bf16_t>(
        operands, out, input, cache, slot_ids, role_offset, input_size, output_size, records_per_input);
  } else {
    launch_matvec<fp16_t>(
        operands, out, input, cache, slot_ids, role_offset, input_size, output_size, records_per_input);
  }
}

/**
 * \brief Compute gate and up projections while loading each input row once.
 *
 * Same contract as `mxfp4_matvec`, with two roles read per record and two
 * outputs written. Both roles must have the same dimensions.
 *
 * \param out_a            Gate output, `[records, output_size]`.
 * \param out_b            Up output, `[records, output_size]`.
 * \param role_offset_a    Byte offset of the gate matrix within each slot.
 * \param role_offset_b    Byte offset of the up matrix within each slot.
 */
inline void mxfp4_matvec_dual(
    tvm::ffi::TensorView out_a,
    tvm::ffi::TensorView out_b,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView cache,
    tvm::ffi::TensorView slot_ids,
    int64_t role_offset_a,
    int64_t role_offset_b,
    int64_t role_bytes,
    int64_t input_size,
    int64_t output_size,
    int64_t records_per_input) {
  const auto operands =
      verify_matvec_inputs(input, cache, slot_ids, role_bytes, input_size, output_size, records_per_input);
  verify_role_range(role_offset_a, role_bytes, operands.cache_bytes, "gate");
  verify_role_range(role_offset_b, role_bytes, operands.cache_bytes, "up");
  verify_matvec_output(out_a, operands, output_size);
  verify_matvec_output(out_b, operands, output_size);

  // An empty batch has nothing to compute, and `records` is a grid dimension.
  if (operands.records == 0) return;

  if (operands.is_bf16) {
    launch_matvec_dual<bf16_t>(
        operands,
        out_a,
        out_b,
        input,
        cache,
        slot_ids,
        role_offset_a,
        role_offset_b,
        input_size,
        output_size,
        records_per_input);
  } else {
    launch_matvec_dual<fp16_t>(
        operands,
        out_a,
        out_b,
        input,
        cache,
        slot_ids,
        role_offset_a,
        role_offset_b,
        input_size,
        output_size,
        records_per_input);
  }
}

}  // namespace sglang
