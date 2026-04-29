#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
constexpr int kKvDim = kLatentDim + kRopeDim;
constexpr int kPackedBytes4Bit = kLatentDim / 2;
constexpr int kPackedBytes2p5 = kLatentDim / 128 * 36;
constexpr int kNormBytes = 2;
constexpr int kSlotBytes4Bit = kPackedBytes4Bit + kNormBytes + kRopeDim * 2;
constexpr int kSlotBytes2p5 = kPackedBytes2p5 + kNormBytes + kRopeDim * 2;
constexpr int kFP8ScaleBytes = (kLatentDim / 128) * 4;
constexpr int kFlashMLAFP8SlotBytes = kLatentDim + kFP8ScaleBytes + kRopeDim * 2;
constexpr float kInvSqrtLatentDim = 0.044194173824159216f;

__global__ void dequantize_selected_4bit_kernel(
    const uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    bf16_t* __restrict__ out,
    const float* __restrict__ centroids,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int64_t loc = locs[row];
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  const uint8_t packed = slot[tid >> 1];
  const uint8_t index = (tid & 1) ? (packed >> 4) : (packed & 0x0F);
  buf[tid] = centroids[index] * signs2[tid];
  __syncthreads();

#pragma unroll
  for (int len = 1; len < kLatentDim; len <<= 1) {
    const int group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = buf[a];
    const float y = buf[b];
    __syncthreads();
    if (pos < len) {
      buf[a] = x + y;
      buf[b] = x - y;
    }
    __syncthreads();
  }

  const half norm = *reinterpret_cast<const half*>(slot + kPackedBytes4Bit);
  const float value = buf[tid] * kInvSqrtLatentDim * signs1[tid] * __half2float(norm);
  row_out[tid] = __float2bfloat16(value);

  if (tid < kRopeDim) {
    const bf16_t* rope = reinterpret_cast<const bf16_t*>(slot + kPackedBytes4Bit + kNormBytes);
    row_out[kLatentDim + tid] = rope[tid];
  }
}

__device__ __forceinline__ uint8_t unpack_3bit(const uint8_t* __restrict__ ptr, int idx) {
  const int bit = idx * 3;
  const uint16_t word = static_cast<uint16_t>(ptr[bit >> 3]) |
                        (static_cast<uint16_t>(ptr[(bit >> 3) + 1]) << 8);
  return (word >> (bit & 7)) & 0x07;
}

__device__ __forceinline__ uint8_t unpack_2bit(const uint8_t* __restrict__ ptr, int idx) {
  const int bit = idx * 2;
  return (ptr[bit >> 3] >> (bit & 7)) & 0x03;
}

template <int N>
__device__ __forceinline__ uint8_t quantize_with_boundaries(
    const float* __restrict__ boundaries,
    float value) {
  uint8_t index = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    index += value > boundaries[i];
  }
  return index;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum_512(float value, float* __restrict__ scratch) {
  const int tid = threadIdx.x;
  value = warp_reduce_sum(value);
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = value;
  }
  __syncthreads();

  float block_sum = 0.0f;
  if (tid < 32) {
    block_sum = tid < 16 ? scratch[tid] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
  }
  __syncthreads();
  return block_sum;
}

__device__ __forceinline__ float fwht_512(float value, float* __restrict__ scratch) {
  const int tid = threadIdx.x;

#pragma unroll
  for (int len = 1; len < 32; len <<= 1) {
    const float other = __shfl_xor_sync(0xffffffff, value, len);
    value = (tid & len) ? other - value : value + other;
  }

  scratch[tid] = value;
  __syncthreads();

#pragma unroll
  for (int len = 32; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = scratch[a];
    const float y = scratch[b];
    __syncthreads();
    if (pos < len) {
      scratch[a] = x + y;
      scratch[b] = x - y;
    }
    __syncthreads();
  }

  return scratch[tid];
}

__global__ void store_2p5_kernel(
    uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    const bf16_t* __restrict__ latent,
    const bf16_t* __restrict__ rope,
    const float* __restrict__ boundaries_high,
    const float* __restrict__ boundaries_low,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t latent_stride_0,
    int64_t latent_stride_1,
    int64_t rope_stride_0,
    int64_t rope_stride_1) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  __shared__ float buf[kLatentDim];
  __shared__ uint8_t indices[kLatentDim];
  __shared__ float norm;

  const bf16_t* latent_row = latent + row * latent_stride_0;
  const float value = __bfloat162float(latent_row[tid]);
  const float norm_sum = block_reduce_sum_512(value * value, buf);
  if (tid == 0) {
    norm = sqrtf(fmaxf(norm_sum, 1.0e-16f));
  }
  __syncthreads();

  const float transformed = fwht_512((value / norm) * signs1[tid], buf);

  const int group = tid >> 7;
  const int channel = tid & 127;
  const float rotated = transformed * kInvSqrtLatentDim * signs2[tid];
  uint8_t index;
  float centroid;
  if (channel < 32) {
    index = quantize_with_boundaries<7>(boundaries_high, rotated);
    centroid = centroids_high[index];
  } else {
    index = quantize_with_boundaries<3>(boundaries_low, rotated);
    centroid = centroids_low[index];
  }
  indices[tid] = index;
  const float recon_norm_sum = block_reduce_sum_512(centroid * centroid, buf);

  const int64_t loc = locs[row];
  uint8_t* slot = compressed + loc * compressed_stride_0;
  if (tid == 0) {
    const float recon_norm = sqrtf(fmaxf(recon_norm_sum, 1.0e-16f));
    const half corrected_norm = __float2half(norm / recon_norm);
    *reinterpret_cast<half*>(slot + kPackedBytes2p5) = corrected_norm;
  }

  if (tid < 48) {
    const int pack_group = tid / 12;
    const int byte_idx = tid - pack_group * 12;
    uint8_t byte = 0;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {
      const int packed_bit = byte_idx * 8 + bit;
      const int channel_idx = packed_bit / 3;
      const int bit_idx = packed_bit - channel_idx * 3;
      if (channel_idx < 32) {
        byte |= ((indices[pack_group * 128 + channel_idx] >> bit_idx) & 1) << bit;
      }
    }
    slot[pack_group * 36 + byte_idx] = byte;
  }

  if (tid < 96) {
    const int pack_group = tid / 24;
    const int byte_idx = tid - pack_group * 24;
    const int base_channel = pack_group * 128 + 32 + byte_idx * 4;
    slot[pack_group * 36 + 12 + byte_idx] =
        indices[base_channel] |
        (indices[base_channel + 1] << 2) |
        (indices[base_channel + 2] << 4) |
        (indices[base_channel + 3] << 6);
  }

  if (tid < kRopeDim) {
    const bf16_t* rope_row = rope + row * rope_stride_0;
    bf16_t* rope_slot = reinterpret_cast<bf16_t*>(slot + kPackedBytes2p5 + kNormBytes);
    rope_slot[tid] = rope_row[tid];
  }
}

__global__ void dequantize_selected_2p5_kernel(
    const uint8_t* __restrict__ compressed,
    const int64_t* __restrict__ locs,
    bf16_t* __restrict__ out,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int64_t loc = locs[row];
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  const int group = tid >> 7;
  const int channel = tid & 127;
  const uint8_t* group_ptr = slot + group * 36;
  uint8_t index;
  float centroid;
  if (channel < 32) {
    index = unpack_3bit(group_ptr, channel);
    centroid = centroids_high[index];
  } else {
    index = unpack_2bit(group_ptr + 12, channel - 32);
    centroid = centroids_low[index];
  }
  buf[tid] = centroid * signs2[tid];
  __syncthreads();

#pragma unroll
  for (int len = 1; len < kLatentDim; len <<= 1) {
    const int group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = buf[a];
    const float y = buf[b];
    __syncthreads();
    if (pos < len) {
      buf[a] = x + y;
      buf[b] = x - y;
    }
    __syncthreads();
  }

  const half norm = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
  const float value = buf[tid] * kInvSqrtLatentDim * signs1[tid] * __half2float(norm);
  row_out[tid] = __float2bfloat16(value);

  if (tid < kRopeDim) {
    const bf16_t* rope = reinterpret_cast<const bf16_t*>(slot + kPackedBytes2p5 + kNormBytes);
    row_out[kLatentDim + tid] = rope[tid];
  }
}

__global__ void dequantize_page_table_selected_2p5_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    bf16_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t page = page_table[row];
  if (tid == 0) {
    compact_page_table[row] = page >= 0 ? static_cast<int32_t>(row) : -1;
  }

  const int64_t loc = page >= 0 ? static_cast<int64_t>(page) : 0;
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  bf16_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  const int group = tid >> 7;
  const int channel = tid & 127;
  const uint8_t* group_ptr = slot + group * 36;
  uint8_t index;
  float centroid;
  if (channel < 32) {
    index = unpack_3bit(group_ptr, channel);
    centroid = centroids_high[index];
  } else {
    index = unpack_2bit(group_ptr + 12, channel - 32);
    centroid = centroids_low[index];
  }
  buf[tid] = centroid * signs2[tid];
  __syncthreads();

#pragma unroll
  for (int len = 1; len < kLatentDim; len <<= 1) {
    const int group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = buf[a];
    const float y = buf[b];
    __syncthreads();
    if (pos < len) {
      buf[a] = x + y;
      buf[b] = x - y;
    }
    __syncthreads();
  }

  const half norm = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
  const float value = buf[tid] * kInvSqrtLatentDim * signs1[tid] * __half2float(norm);
  row_out[tid] = __float2bfloat16(value);

  if (tid < kRopeDim) {
    const bf16_t* rope = reinterpret_cast<const bf16_t*>(slot + kPackedBytes2p5 + kNormBytes);
    row_out[kLatentDim + tid] = rope[tid];
  }
}

__global__ void dequantize_page_table_selected_2p5_fp8_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    uint8_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int32_t page = page_table[row];
  if (tid == 0) {
    compact_page_table[row] = page >= 0 ? static_cast<int32_t>(row) : -1;
  }

  const int64_t loc = page >= 0 ? static_cast<int64_t>(page) : 0;
  const uint8_t* slot = compressed + loc * compressed_stride_0;
  uint8_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  const int group = tid >> 7;
  const int channel = tid & 127;
  const uint8_t* group_ptr = slot + group * 36;
  uint8_t index;
  float centroid;
  if (channel < 32) {
    index = unpack_3bit(group_ptr, channel);
    centroid = centroids_high[index];
  } else {
    index = unpack_2bit(group_ptr + 12, channel - 32);
    centroid = centroids_low[index];
  }
  buf[tid] = centroid * signs2[tid];
  __syncthreads();

#pragma unroll
  for (int len = 1; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = buf[a];
    const float y = buf[b];
    __syncthreads();
    if (pos < len) {
      buf[a] = x + y;
      buf[b] = x - y;
    }
    __syncthreads();
  }

  const half norm = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
  const float value = buf[tid] * kInvSqrtLatentDim * signs1[tid] * __half2float(norm);
  const int tile = tid >> 7;
  const int local = tid & 127;
  buf[tid] = fabsf(value);
  __syncthreads();

#pragma unroll
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (local < stride) {
      const int base = tile * 128 + local;
      buf[base] = fmaxf(buf[base], buf[base + stride]);
    }
    __syncthreads();
  }

  const float scale = buf[tile * 128] / 448.0f;
  if (local == 0) {
    reinterpret_cast<float*>(row_out + kLatentDim)[tile] = scale;
  }
  const float scaled = scale > 0.0f ? value / scale : 0.0f;
  const float clipped = fminf(448.0f, fmaxf(-448.0f, scaled));
  const __nv_fp8_e4m3 fp8_value(clipped);
  row_out[tid] = fp8_value.__x;

  if (tid < kRopeDim) {
    const uint8_t* rope = slot + kPackedBytes2p5 + kNormBytes;
    uint16_t* rope_out = reinterpret_cast<uint16_t*>(row_out + kLatentDim + kFP8ScaleBytes);
    const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(rope);
    rope_out[tid] = rope_in[tid];
  }
}

__global__ void dequantize_page_table_selected_2p5_fp8_reuse_kernel(
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    uint8_t* __restrict__ out,
    int32_t* __restrict__ compact_page_table,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t topk,
    int64_t compressed_stride_0,
    int64_t out_stride_0) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= num_rows) return;

  const int64_t query_row = row / topk;
  const int64_t col = row - query_row * topk;
  const int32_t page = page_table[row];
  if (page < 0) {
    if (tid == 0) {
      compact_page_table[row] = -1;
    }
    return;
  }

  for (int64_t prev = 0; prev < query_row; ++prev) {
    const int64_t prev_row = prev * topk + col;
    if (page_table[prev_row] == page) {
      if (tid == 0) {
        compact_page_table[row] = static_cast<int32_t>(prev_row);
      }
      return;
    }
  }

  if (tid == 0) {
    compact_page_table[row] = static_cast<int32_t>(row);
  }

  const uint8_t* slot = compressed + static_cast<int64_t>(page) * compressed_stride_0;
  uint8_t* row_out = out + row * out_stride_0;

  __shared__ float buf[kLatentDim];

  const int group = tid >> 7;
  const int channel = tid & 127;
  const uint8_t* group_ptr = slot + group * 36;
  uint8_t index;
  float centroid;
  if (channel < 32) {
    index = unpack_3bit(group_ptr, channel);
    centroid = centroids_high[index];
  } else {
    index = unpack_2bit(group_ptr + 12, channel - 32);
    centroid = centroids_low[index];
  }
  buf[tid] = centroid * signs2[tid];
  __syncthreads();

#pragma unroll
  for (int len = 1; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = buf[a];
    const float y = buf[b];
    __syncthreads();
    if (pos < len) {
      buf[a] = x + y;
      buf[b] = x - y;
    }
    __syncthreads();
  }

  const half norm = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
  const float value = buf[tid] * kInvSqrtLatentDim * signs1[tid] * __half2float(norm);
  const int tile = tid >> 7;
  const int local = tid & 127;
  buf[tid] = fabsf(value);
  __syncthreads();

#pragma unroll
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (local < stride) {
      const int base = tile * 128 + local;
      buf[base] = fmaxf(buf[base], buf[base + stride]);
    }
    __syncthreads();
  }

  const float scale = buf[tile * 128] / 448.0f;
  if (local == 0) {
    reinterpret_cast<float*>(row_out + kLatentDim)[tile] = scale;
  }
  const float scaled = scale > 0.0f ? value / scale : 0.0f;
  const float clipped = fminf(448.0f, fmaxf(-448.0f, scaled));
  const __nv_fp8_e4m3 fp8_value(clipped);
  row_out[tid] = fp8_value.__x;

  if (tid < kRopeDim) {
    const uint8_t* rope = slot + kPackedBytes2p5 + kNormBytes;
    uint16_t* rope_out = reinterpret_cast<uint16_t*>(row_out + kLatentDim + kFP8ScaleBytes);
    const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(rope);
    rope_out[tid] = rope_in[tid];
  }
}

struct TurboQuantDenseKVDequant4BitKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView centroids,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes4Bit})
        .with_strides({compressed_stride_0, kSlotBytes4Bit, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({16}).with_dtype<float>().with_device(device).verify(centroids);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        dequantize_selected_4bit_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(centroids.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct TurboQuantDenseKVDequant2p5BitKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        dequantize_selected_2p5_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct TurboQuantDenseKVDequantPageTable2p5BitKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto B = SymbolicSize{"num_query_rows"};
    auto K = SymbolicSize{"topk"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({B, K})
        .with_strides({K, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({B, K})
        .with_strides({K, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(compact_page_table);
    TensorMatcher({N, 1, kKvDim})
        .with_strides({out_stride_0, kKvDim, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        dequantize_page_table_selected_2p5_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct TurboQuantDenseKVDequantPageTable2p5BitFP8Kernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto B = SymbolicSize{"num_query_rows"};
    auto K = SymbolicSize{"topk"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({B, K})
        .with_strides({K, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({B, K})
        .with_strides({K, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(compact_page_table);
    TensorMatcher({N, 1, kFlashMLAFP8SlotBytes})
        .with_strides({out_stride_0, kFlashMLAFP8SlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        dequantize_page_table_selected_2p5_fp8_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<uint8_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct TurboQuantDenseKVDequantPageTable2p5BitFP8ReuseKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView compact_page_table,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto B = SymbolicSize{"num_query_rows"};
    auto K = SymbolicSize{"topk"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({B, K})
        .with_strides({K, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({B, K})
        .with_strides({K, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(compact_page_table);
    TensorMatcher({N, 1, kFlashMLAFP8SlotBytes})
        .with_strides({out_stride_0, kFlashMLAFP8SlotBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = B.unwrap() * K.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        dequantize_page_table_selected_2p5_fp8_reuse_kernel,
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<uint8_t*>(out.data_ptr()),
        static_cast<int32_t*>(compact_page_table.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        K.unwrap(),
        compressed_stride_0.unwrap(),
        out_stride_0.unwrap());
  }
};

struct TurboQuantDenseKVStore2p5BitKernel {
  static void run(
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView locs,
      tvm::ffi::TensorView latent,
      tvm::ffi::TensorView rope,
      tvm::ffi::TensorView boundaries_high,
      tvm::ffi::TensorView boundaries_low,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto S = SymbolicSize{"num_slots"};
    auto N = SymbolicSize{"num_rows"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto latent_stride_0 = SymbolicSize{"latent_stride_0"};
    auto latent_stride_1 = SymbolicSize{"latent_stride_1"};
    auto rope_stride_0 = SymbolicSize{"rope_stride_0"};
    auto rope_stride_1 = SymbolicSize{"rope_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(locs);
    TensorMatcher({N, 1, kLatentDim})
        .with_strides({latent_stride_0, latent_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(latent);
    TensorMatcher({N, 1, kRopeDim})
        .with_strides({rope_stride_0, rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(rope);
    TensorMatcher({7}).with_dtype<float>().with_device(device).verify(boundaries_high);
    TensorMatcher({3}).with_dtype<float>().with_device(device).verify(boundaries_low);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    const int64_t num_rows = N.unwrap();
    if (num_rows == 0) return;

    LaunchKernel(num_rows, kLatentDim, device.unwrap())(
        store_2p5_kernel,
        static_cast<uint8_t*>(compressed.data_ptr()),
        static_cast<const int64_t*>(locs.data_ptr()),
        static_cast<const bf16_t*>(latent.data_ptr()),
        static_cast<const bf16_t*>(rope.data_ptr()),
        static_cast<const float*>(boundaries_high.data_ptr()),
        static_cast<const float*>(boundaries_low.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        num_rows,
        compressed_stride_0.unwrap(),
        latent_stride_0.unwrap(),
        latent_stride_1.unwrap(),
        rope_stride_0.unwrap(),
        rope_stride_1.unwrap());
  }
};

}  // namespace
