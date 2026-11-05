#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cub/block/block_scan.cuh>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {
namespace msa_v1::attn::compact_q8k8::dequant {

namespace {

constexpr int32_t kHeadDim = 128;
constexpr int32_t kPackedHeadDim = kHeadDim / 2;
constexpr int32_t kScaleGroups = kHeadDim / 16;
constexpr int32_t kPageSize = 128;
constexpr int32_t kTopK = 16;
constexpr int32_t kThreads = 256;
constexpr int32_t kWarpsPerBlock = kThreads / 32;
constexpr int32_t kRowsPerWarp = 4;
constexpr int32_t kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;
constexpr int64_t kMaxFullPageBatch = 4096;

#if (__CUDACC_VER_MAJOR__ > 13) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
#define SGLANG_MSA_V1_HAS_QMUL4 1
#else
#define SGLANG_MSA_V1_HAS_QMUL4 0
#endif

#if SGLANG_MSA_V1_HAS_QMUL4
SGL_DEVICE uint4 convert_group_qmul4(uint2 packed, uint32_t scale_e4m3x4) {
  uint4 output;
  const uint16_t packed_0 = static_cast<uint16_t>(packed.x);
  const uint16_t packed_1 = static_cast<uint16_t>(packed.x >> 16);
  const uint16_t packed_2 = static_cast<uint16_t>(packed.y);
  const uint16_t packed_3 = static_cast<uint16_t>(packed.y >> 16);
  asm volatile(
      "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, %4, %8;\n"
      "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %1, %5, %8;\n"
      "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %2, %6, %8;\n"
      "mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %3, %7, %8;"
      : "=r"(output.x), "=r"(output.y), "=r"(output.z), "=r"(output.w)
      : "h"(packed_0), "h"(packed_1), "h"(packed_2), "h"(packed_3), "r"(scale_e4m3x4));
  return output;
}
#endif

SGL_DEVICE void convert_e2m1x8_fallback(uint32_t& output_0, uint32_t& output_1, uint32_t packed, uint32_t scale_f16x2) {
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      ".reg .b32 h0, h1, h2, h3;\n"
      ".reg .b16 e0, e1, e2, e3;\n"
      "mov.b32 {b0, b1, b2, b3}, %2;\n"
      "cvt.rn.f16x2.e2m1x2 h0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 h1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 h2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 h3, b3;\n"
      "mul.rn.f16x2 h0, h0, %3;\n"
      "mul.rn.f16x2 h1, h1, %3;\n"
      "mul.rn.f16x2 h2, h2, %3;\n"
      "mul.rn.f16x2 h3, h3, %3;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
      "mov.b32 %0, {e0, e1};\n"
      "mov.b32 %1, {e2, e3};\n"
      "}"
      : "=r"(output_0), "=r"(output_1)
      : "r"(packed), "r"(scale_f16x2));
}

SGL_DEVICE uint4 convert_group_fallback(uint2 packed, uint32_t scale_byte) {
  const uint16_t scale_e4m3x2 = static_cast<uint16_t>(scale_byte | (scale_byte << 8));
  uint32_t scale_f16x2;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(scale_f16x2) : "h"(scale_e4m3x2));

  uint4 output;
  convert_e2m1x8_fallback(output.x, output.y, packed.x, scale_f16x2);
  convert_e2m1x8_fallback(output.z, output.w, packed.y, scale_f16x2);
  return output;
}

SGL_DEVICE float decode_e2m1(uint32_t nibble) {
  constexpr float kMagnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = kMagnitude[nibble & 7u];
  return nibble & 8u ? -value : value;
}

SGL_DEVICE uint32_t pack_e4m3x4(float a, float b, float c, float d) {
  const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}

SGL_DEVICE float decode_e4m3(uint32_t scale_byte) {
  const uint16_t scale_e4m3x2 = static_cast<uint16_t>(scale_byte | (scale_byte << 8));
  uint32_t scale_f16x2;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(scale_f16x2) : "h"(scale_e4m3x2));
  const uint16_t scale_f16 = static_cast<uint16_t>(scale_f16x2);
  float result;
  asm volatile("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(scale_f16));
  return result;
}

SGL_DEVICE uint4 convert_group_scaled(uint2 packed, uint32_t scale_byte, float global_scale) {
  const float scale = decode_e4m3(scale_byte) * global_scale;
  const uint64_t bits = static_cast<uint64_t>(packed.x) | (static_cast<uint64_t>(packed.y) << 32);
  uint4 output;
  uint32_t* words = reinterpret_cast<uint32_t*>(&output);
#pragma unroll
  for (uint32_t group = 0; group < 4; ++group) {
    const uint32_t nibbles = static_cast<uint32_t>(bits >> (group * 16));
    words[group] = pack_e4m3x4(
        decode_e2m1(nibbles) * scale,
        decode_e2m1(nibbles >> 4) * scale,
        decode_e2m1(nibbles >> 8) * scale,
        decode_e2m1(nibbles >> 12) * scale);
  }
  return output;
}

SGL_DEVICE uint4 convert_group(uint2 packed, uint32_t scale_byte, float global_scale) {
  if (global_scale != 1.0f) {
    return convert_group_scaled(packed, scale_byte, global_scale);
  }
#if SGLANG_MSA_V1_HAS_QMUL4
  return convert_group_qmul4(packed, scale_byte * 0x01010101u);
#else
  return convert_group_fallback(packed, scale_byte);
#endif
}

__global__ __launch_bounds__(kThreads) void gather_dequant_pair_kernel(
    const uint8_t* __restrict__ packed_k,
    const uint8_t* __restrict__ packed_v,
    const uint8_t* __restrict__ scale_k,
    const uint8_t* __restrict__ scale_v,
    const float* __restrict__ global_scale_ptr,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ topk,
    const int32_t* __restrict__ query_seq_lens,
    uint8_t* __restrict__ output_k,
    uint8_t* __restrict__ output_v,
    int64_t rows,
    int64_t payload_page_stride,
    int64_t scale_page_stride,
    int32_t num_query_rows,
    int32_t num_kv_heads,
    int32_t max_pages,
    int32_t q_len_per_req) {
  const int32_t lane = threadIdx.x & 31;
  const int32_t warp = threadIdx.x >> 5;
  const int32_t row_in_warp = lane >> 3;
  const int32_t scale_group = lane & 7;
  const int32_t source_lane = row_in_warp << 3;
  const uint32_t subgroup_mask = 0xffu << source_lane;
  int64_t row = (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp) * kRowsPerWarp + row_in_warp;
  const int64_t row_stride = static_cast<int64_t>(gridDim.x) * kRowsPerBlock;
  const float global_scale = *global_scale_ptr;

  for (; row < rows; row += row_stride) {
    // Payload and scale pages may sit inside wider per-page rows, so each keeps its own offset.
    int64_t payload_offset = 0;
    int64_t scale_offset = 0;
    int32_t valid = 0;
    if (scale_group == 0) {
      int64_t cursor = row;
      const int32_t token = static_cast<int32_t>(cursor & (kPageSize - 1));
      cursor >>= 7;
      const int32_t rank = static_cast<int32_t>(cursor & (kTopK - 1));
      cursor >>= 4;
      const int32_t head = static_cast<int32_t>(cursor % num_kv_heads);
      const int32_t query = static_cast<int32_t>(cursor / num_kv_heads);
      const int32_t logical_page = topk[(query * num_kv_heads + head) * kTopK + rank];
      const int32_t logical_token = logical_page * kPageSize + token;
      valid = query < num_query_rows && logical_page >= 0 && logical_page < max_pages &&
              logical_token < query_seq_lens[query];
      if (valid) {
        const int32_t request = query / q_len_per_req;
        const int32_t physical_page = page_table[request * max_pages + logical_page];
        const int64_t page_row = static_cast<int64_t>(head) * kPageSize + token;
        payload_offset = physical_page * payload_page_stride + page_row * kPackedHeadDim;
        scale_offset = physical_page * scale_page_stride + page_row * kScaleGroups;
      }
    }
    valid = __shfl_sync(subgroup_mask, valid, source_lane);
    payload_offset = __shfl_sync(subgroup_mask, payload_offset, source_lane);
    scale_offset = __shfl_sync(subgroup_mask, scale_offset, source_lane);

    uint4 converted_k = make_uint4(0, 0, 0, 0);
    uint4 converted_v = make_uint4(0, 0, 0, 0);
    if (valid) {
      const uint2 k_values = *reinterpret_cast<const uint2*>(packed_k + payload_offset + scale_group * 8);
      const uint2 v_values = *reinterpret_cast<const uint2*>(packed_v + payload_offset + scale_group * 8);
      const uint32_t k_scale = scale_k[scale_offset + scale_group];
      const uint32_t v_scale = scale_v[scale_offset + scale_group];
      converted_k = convert_group(k_values, k_scale, global_scale);
      converted_v = convert_group(v_values, v_scale, global_scale);
    }
    *reinterpret_cast<uint4*>(output_k + row * kHeadDim + scale_group * 16) = converted_k;
    *reinterpret_cast<uint4*>(output_v + row * kHeadDim + scale_group * 16) = converted_v;
  }
}

// Rows are numbered over live tokens only: row_offsets is the exclusive prefix sum of min(seq_len, max_pages * 128).
__global__ __launch_bounds__(kThreads) void full_page_dequant_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scale,
    const float* __restrict__ global_scale_ptr,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seq_lens,
    uint8_t* __restrict__ output,
    int32_t batch,
    int64_t payload_page_stride,
    int64_t scale_page_stride,
    int32_t max_pages) {
  using BlockScan = cub::BlockScan<int64_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  extern __shared__ int64_t row_offsets[];

  const int32_t max_rows = max_pages * kPageSize;
  const int32_t per_thread = (batch + kThreads - 1) / kThreads;
  const int32_t first = min(static_cast<int32_t>(threadIdx.x) * per_thread, batch);
  const int32_t last = min(first + per_thread, batch);
  int64_t live_rows = 0;
  for (int32_t request = first; request < last; ++request) {
    live_rows += max(0, min(seq_lens[request], max_rows));
  }
  int64_t offset;
  BlockScan(scan_storage).ExclusiveSum(live_rows, offset);
  for (int32_t request = first; request < last; ++request) {
    row_offsets[request] = offset;
    offset += max(0, min(seq_lens[request], max_rows));
  }
  if (first < last && last == batch) {
    row_offsets[batch] = offset;
  }
  __syncthreads();

  const int32_t lane = threadIdx.x & 31;
  const int32_t warp = threadIdx.x >> 5;
  const int32_t scale_group = lane & 7;
  const float global_scale = *global_scale_ptr;

  // Each block takes one contiguous slice of the live rows, so a thread searches for its request only when it
  // crosses into a later one.
  const int64_t total_rows = row_offsets[batch];
  const int64_t block_rows = static_cast<int64_t>(gridDim.x) * kRowsPerBlock;
  const int64_t slice = (total_rows + block_rows - 1) / block_rows * kRowsPerBlock;
  const int64_t end = min(static_cast<int64_t>(blockIdx.x + 1) * slice, total_rows);
  int32_t request = 0;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * slice + warp * kRowsPerWarp + (lane >> 3); row < end;
       row += kRowsPerBlock) {
    if (row >= row_offsets[request + 1]) {
      int32_t high = batch - 1;
      while (request < high) {
        const int32_t mid = (request + high + 1) >> 1;
        if (row_offsets[mid] <= row) {
          request = mid;
        } else {
          high = mid - 1;
        }
      }
    }
    const int32_t logical_token = static_cast<int32_t>(row - row_offsets[request]);
    const int32_t physical_page = page_table[static_cast<int64_t>(request) * max_pages + (logical_token >> 7)];
    const int64_t physical_row = static_cast<int64_t>(physical_page) * kPageSize + (logical_token & (kPageSize - 1));
    const int32_t token = logical_token & (kPageSize - 1);
    const int64_t payload_offset = physical_page * payload_page_stride + token * kPackedHeadDim;
    const int64_t scale_offset = physical_page * scale_page_stride + token * kScaleGroups;

    const uint2 values = *reinterpret_cast<const uint2*>(packed + payload_offset + scale_group * 8);
    const uint32_t block_scale = scale[scale_offset + scale_group];
    const uint4 converted = convert_group(values, block_scale, global_scale);
    *reinterpret_cast<uint4*>(output + physical_row * kHeadDim + scale_group * 16) = converted;
  }
}

}  // namespace

inline void gather_dequant_pair(
    const tvm::ffi::TensorView packed_k,
    const tvm::ffi::TensorView packed_v,
    const tvm::ffi::TensorView scale_k,
    const tvm::ffi::TensorView scale_v,
    const tvm::ffi::TensorView global_scale,
    const tvm::ffi::TensorView page_table,
    const tvm::ffi::TensorView topk,
    const tvm::ffi::TensorView query_seq_lens,
    const tvm::ffi::TensorView output_k,
    const tvm::ffi::TensorView output_v,
    const int64_t q_len_per_req_arg) {
  using namespace host;
  auto P = SymbolicSize{"pages"};
  auto B = SymbolicSize{"batch"};
  auto M = SymbolicSize{"max_pages"};
  auto Q = SymbolicSize{"query_rows"};
  auto H = SymbolicSize{"kv_heads"};
  auto PS = SymbolicSize{"payload_page_stride"};
  auto SS = SymbolicSize{"scale_page_stride"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({P, H, kPageSize, kPackedHeadDim})
      .with_strides({PS, kPageSize * kPackedHeadDim, kPackedHeadDim, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(packed_k)
      .verify(packed_v);
  TensorMatcher({P, H, kPageSize, kScaleGroups})
      .with_strides({SS, kPageSize * kScaleGroups, kScaleGroups, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(scale_k)
      .verify(scale_v);
  TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device).verify(global_scale);
  TensorMatcher({B, M}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(page_table);
  TensorMatcher({Q, H, kTopK}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(topk);
  TensorMatcher({Q}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(query_seq_lens);
  TensorMatcher({Q, H, kTopK, kPageSize, kHeadDim}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(output_k);
  TensorMatcher({Q, H, kTopK, kPageSize, kHeadDim}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(output_v);

  const int32_t q_len_per_req = static_cast<int32_t>(q_len_per_req_arg);
  CHECK_HOST(q_len_per_req > 0);
  CHECK_HOST(B.unwrap() * q_len_per_req == Q.unwrap()) << "batch * q_len_per_req must equal query_rows";
  const int64_t rows = Q.unwrap() * H.unwrap() * kTopK * kPageSize;
  const int64_t required_blocks = host::div_ceil(rows, int64_t{kRowsPerBlock});
  int32_t sm_count = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device.unwrap().device_id));
  const int32_t grid = static_cast<int32_t>(std::min(required_blocks, static_cast<int64_t>(sm_count) * 8));
  LaunchKernel(grid, kThreads, device.unwrap())(  //
      gather_dequant_pair_kernel,
      static_cast<const uint8_t*>(packed_k.data_ptr()),
      static_cast<const uint8_t*>(packed_v.data_ptr()),
      static_cast<const uint8_t*>(scale_k.data_ptr()),
      static_cast<const uint8_t*>(scale_v.data_ptr()),
      static_cast<const float*>(global_scale.data_ptr()),
      static_cast<const int32_t*>(page_table.data_ptr()),
      static_cast<const int32_t*>(topk.data_ptr()),
      static_cast<const int32_t*>(query_seq_lens.data_ptr()),
      static_cast<uint8_t*>(output_k.data_ptr()),
      static_cast<uint8_t*>(output_v.data_ptr()),
      rows,
      PS.unwrap(),
      SS.unwrap(),
      static_cast<int32_t>(Q.unwrap()),
      static_cast<int32_t>(H.unwrap()),
      static_cast<int32_t>(M.unwrap()),
      q_len_per_req);
}

inline void full_page_dequant(
    const tvm::ffi::TensorView packed,
    const tvm::ffi::TensorView scale,
    const tvm::ffi::TensorView global_scale,
    const tvm::ffi::TensorView page_table,
    const tvm::ffi::TensorView seq_lens,
    const tvm::ffi::TensorView output) {
  using namespace host;
  auto P = SymbolicSize{"pages"};
  auto B = SymbolicSize{"batch"};
  auto M = SymbolicSize{"max_pages"};
  auto PS = SymbolicSize{"payload_page_stride"};
  auto SS = SymbolicSize{"scale_page_stride"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({P, kPageSize, kPackedHeadDim})
      .with_strides({PS, kPackedHeadDim, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(packed);
  TensorMatcher({P, kPageSize, kScaleGroups})
      .with_strides({SS, kScaleGroups, 1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device)
      .verify(scale);
  TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device).verify(global_scale);
  TensorMatcher({B, M}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(page_table);
  TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(seq_lens);
  TensorMatcher({P, kPageSize, kHeadDim}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(output);

  const int64_t batch = B.unwrap();
  CHECK_HOST(batch > 0 && batch <= kMaxFullPageBatch)
      << "full_page_dequant supports 1 to " << kMaxFullPageBatch << " requests, got " << batch;
  const int64_t rows = batch * M.unwrap() * kPageSize;
  CHECK_HOST(rows > 0);
  const int64_t required_blocks = host::div_ceil(rows, int64_t{kRowsPerBlock});
  int32_t sm_count = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device.unwrap().device_id));
  const std::size_t row_offset_bytes = static_cast<std::size_t>(batch + 1) * sizeof(int64_t);
  // One wave: the row offsets take shared memory, which can cut the blocks that fit on an SM.
  int32_t blocks_per_sm = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, full_page_dequant_kernel, kThreads, row_offset_bytes));
  const int32_t grid =
      static_cast<int32_t>(std::min(required_blocks, static_cast<int64_t>(sm_count) * std::max(blocks_per_sm, 1)));
  LaunchKernel(grid, kThreads, device.unwrap(), row_offset_bytes)(  //
      full_page_dequant_kernel,
      static_cast<const uint8_t*>(packed.data_ptr()),
      static_cast<const uint8_t*>(scale.data_ptr()),
      static_cast<const float*>(global_scale.data_ptr()),
      static_cast<const int32_t*>(page_table.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<uint8_t*>(output.data_ptr()),
      static_cast<int32_t>(batch),
      PS.unwrap(),
      SS.unwrap(),
      static_cast<int32_t>(M.unwrap()));
}

}  // namespace msa_v1::attn::compact_q8k8::dequant
}  // namespace sglang
