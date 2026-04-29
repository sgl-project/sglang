#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kLatentDim = 512;
constexpr int kRopeDim = 64;
constexpr int kPackedBytes2p5 = kLatentDim / 128 * 36;
constexpr int kNormBytes = 2;
constexpr int kSlotBytes2p5 = kPackedBytes2p5 + kNormBytes + kRopeDim * 2;
constexpr float kInvSqrtLatentDim = 0.044194173824159216f;
constexpr float kNegInf = -3.4028234663852886e38f;

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

__device__ __forceinline__ float bf16_to_float(const bf16_t value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum_128(float value, float* __restrict__ warp_sums) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  float sum = 0.0f;
  if (tid < 4) {
    sum = warp_sums[tid];
  }
  if (warp == 0) {
    sum = warp_reduce_sum(sum);
  }
  if (tid == 0) {
    warp_sums[0] = sum;
  }
  __syncthreads();
  return warp_sums[0];
}

__global__ void turboquant_dense_mla_decode_2p5_kernel(
    const bf16_t* __restrict__ q_nope,
    const bf16_t* __restrict__ q_rope,
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    bf16_t* __restrict__ out,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t num_heads,
    int64_t topk,
    int64_t q_nope_stride_0,
    int64_t q_nope_stride_1,
    int64_t q_rope_stride_0,
    int64_t q_rope_stride_1,
    int64_t compressed_stride_0,
    int64_t page_table_stride_0,
    int64_t out_stride_0,
    int64_t out_stride_1,
    float sm_scale) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads) return;

  __shared__ float q_rot[kLatentDim];
  __shared__ float acc[kLatentDim];
  __shared__ float score_latent[kLatentDim];
  __shared__ float score_rope[kLatentDim];
  __shared__ float softmax_state[4];

  const bf16_t* q_nope_row = q_nope + row * q_nope_stride_0 + head * q_nope_stride_1;
  q_rot[tid] = bf16_to_float(q_nope_row[tid]) * signs1[tid];
  acc[tid] = 0.0f;
  if (tid == 0) {
    softmax_state[0] = kNegInf;
    softmax_state[1] = 0.0f;
  }
  __syncthreads();

#pragma unroll
  for (int len = 1; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = q_rot[a];
    const float y = q_rot[b];
    __syncthreads();
    if (pos < len) {
      q_rot[a] = x + y;
      q_rot[b] = x - y;
    }
    __syncthreads();
  }
  q_rot[tid] = q_rot[tid] * kInvSqrtLatentDim * signs2[tid];
  __syncthreads();

  const bf16_t* q_rope_row = q_rope + row * q_rope_stride_0 + head * q_rope_stride_1;
  const int32_t* pages = page_table + row * page_table_stride_0;

  for (int col = 0; col < topk; ++col) {
    const int32_t page = pages[col];
    const bool valid = page >= 0;
    const uint8_t* slot = compressed + (valid ? static_cast<int64_t>(page) : 0) * compressed_stride_0;

    const int group = tid >> 7;
    const int channel = tid & 127;
    const uint8_t* group_ptr = slot + group * 36;
    float centroid;
    if (channel < 32) {
      centroid = centroids_high[unpack_3bit(group_ptr, channel)];
    } else {
      centroid = centroids_low[unpack_2bit(group_ptr + 12, channel - 32)];
    }

    const half norm_half = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
    const float norm = __half2float(norm_half);
    score_latent[tid] = valid ? q_rot[tid] * centroid : 0.0f;
    if (tid < kRopeDim && valid) {
      const bf16_t* rope = reinterpret_cast<const bf16_t*>(slot + kPackedBytes2p5 + kNormBytes);
      score_rope[tid] = bf16_to_float(q_rope_row[tid]) * bf16_to_float(rope[tid]);
    } else {
      score_rope[tid] = 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int stride = 256; stride > 0; stride >>= 1) {
      if (tid < stride) {
        score_latent[tid] += score_latent[tid + stride];
        score_rope[tid] += score_rope[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float score = valid ? (norm * score_latent[0] + score_rope[0]) * sm_scale : kNegInf;
      const float old_m = softmax_state[0];
      const float old_l = softmax_state[1];
      const float new_m = fmaxf(old_m, score);
      const float alpha = __expf(old_m - new_m);
      const float beta = __expf(score - new_m);
      softmax_state[0] = new_m;
      softmax_state[1] = old_l * alpha + beta;
      softmax_state[2] = alpha;
      softmax_state[3] = beta;
    }
    __syncthreads();

    acc[tid] = acc[tid] * softmax_state[2] + softmax_state[3] * norm * centroid;
    __syncthreads();
  }

  const float denom = softmax_state[1];
  q_rot[tid] = denom > 0.0f ? (acc[tid] / denom) * signs2[tid] : 0.0f;
  __syncthreads();

#pragma unroll
  for (int len = 1; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = q_rot[a];
    const float y = q_rot[b];
    __syncthreads();
    if (pos < len) {
      q_rot[a] = x + y;
      q_rot[b] = x - y;
    }
    __syncthreads();
  }

  bf16_t* out_row = out + row * out_stride_0 + head * out_stride_1;
  out_row[tid] = __float2bfloat16(q_rot[tid] * kInvSqrtLatentDim * signs1[tid]);
}

__global__ void turboquant_dense_mla_decode_2p5_stage1_kernel(
    const bf16_t* __restrict__ q_nope,
    const bf16_t* __restrict__ q_rope,
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    float* __restrict__ mid,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t num_heads,
    int64_t topk,
    int64_t num_splits,
    int64_t q_nope_stride_0,
    int64_t q_nope_stride_1,
    int64_t q_rope_stride_0,
    int64_t q_rope_stride_1,
    int64_t compressed_stride_0,
    int64_t page_table_stride_0,
    int64_t mid_stride_0,
    int64_t mid_stride_1,
    int64_t mid_stride_2,
    float sm_scale) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads || split >= num_splits) return;

  __shared__ float q_rot[kLatentDim];
  __shared__ float acc[kLatentDim];
  __shared__ float score_latent[kLatentDim];
  __shared__ float score_rope[kLatentDim];
  __shared__ float softmax_state[4];

  const bf16_t* q_nope_row = q_nope + row * q_nope_stride_0 + head * q_nope_stride_1;
  q_rot[tid] = bf16_to_float(q_nope_row[tid]) * signs1[tid];
  acc[tid] = 0.0f;
  if (tid == 0) {
    softmax_state[0] = kNegInf;
    softmax_state[1] = 0.0f;
  }
  __syncthreads();

#pragma unroll
  for (int len = 1; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = q_rot[a];
    const float y = q_rot[b];
    __syncthreads();
    if (pos < len) {
      q_rot[a] = x + y;
      q_rot[b] = x - y;
    }
    __syncthreads();
  }
  q_rot[tid] = q_rot[tid] * kInvSqrtLatentDim * signs2[tid];
  __syncthreads();

  const int64_t chunk = (topk + num_splits - 1) / num_splits;
  const int64_t begin = split * chunk;
  const int64_t end = min(begin + chunk, topk);
  const bf16_t* q_rope_row = q_rope + row * q_rope_stride_0 + head * q_rope_stride_1;
  const int32_t* pages = page_table + row * page_table_stride_0;

  for (int64_t col = begin; col < end; ++col) {
    const int32_t page = pages[col];
    const bool valid = page >= 0;
    const uint8_t* slot = compressed + (valid ? static_cast<int64_t>(page) : 0) * compressed_stride_0;

    const int group = tid >> 7;
    const int channel = tid & 127;
    const uint8_t* group_ptr = slot + group * 36;
    float centroid;
    if (channel < 32) {
      centroid = centroids_high[unpack_3bit(group_ptr, channel)];
    } else {
      centroid = centroids_low[unpack_2bit(group_ptr + 12, channel - 32)];
    }

    const half norm_half = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
    const float norm = __half2float(norm_half);
    score_latent[tid] = valid ? q_rot[tid] * centroid : 0.0f;
    if (tid < kRopeDim && valid) {
      const bf16_t* rope = reinterpret_cast<const bf16_t*>(slot + kPackedBytes2p5 + kNormBytes);
      score_rope[tid] = bf16_to_float(q_rope_row[tid]) * bf16_to_float(rope[tid]);
    } else {
      score_rope[tid] = 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int stride = 256; stride > 0; stride >>= 1) {
      if (tid < stride) {
        score_latent[tid] += score_latent[tid + stride];
        score_rope[tid] += score_rope[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float score = valid ? (norm * score_latent[0] + score_rope[0]) * sm_scale : kNegInf;
      const float old_m = softmax_state[0];
      const float old_l = softmax_state[1];
      const float new_m = fmaxf(old_m, score);
      const float alpha = __expf(old_m - new_m);
      const float beta = __expf(score - new_m);
      softmax_state[0] = new_m;
      softmax_state[1] = old_l * alpha + beta;
      softmax_state[2] = alpha;
      softmax_state[3] = beta;
    }
    __syncthreads();

    acc[tid] = acc[tid] * softmax_state[2] + softmax_state[3] * norm * centroid;
    __syncthreads();
  }

  float* mid_row = mid + row * mid_stride_0 + head * mid_stride_1 + split * mid_stride_2;
  if (tid == 0) {
    mid_row[0] = softmax_state[0];
    mid_row[1] = softmax_state[1];
  }
  mid_row[2 + tid] = acc[tid];
}

__global__ void turboquant_dense_mla_rotate_query_kernel(
    const bf16_t* __restrict__ q_nope,
    float* __restrict__ q_rot_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t num_heads,
    int64_t q_nope_stride_0,
    int64_t q_nope_stride_1,
    int64_t q_rot_stride_0,
    int64_t q_rot_stride_1) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads) return;

  __shared__ float q_rot[kLatentDim];
  const bf16_t* q_nope_row = q_nope + row * q_nope_stride_0 + head * q_nope_stride_1;
  q_rot[tid] = bf16_to_float(q_nope_row[tid]) * signs1[tid];
  __syncthreads();

#pragma unroll
  for (int len = 1; len < kLatentDim; len <<= 1) {
    const int wht_group = tid / (len << 1);
    const int pos = tid & ((len << 1) - 1);
    const int a = wht_group * (len << 1) + (pos & (len - 1));
    const int b = a + len;
    const float x = q_rot[a];
    const float y = q_rot[b];
    __syncthreads();
    if (pos < len) {
      q_rot[a] = x + y;
      q_rot[b] = x - y;
    }
    __syncthreads();
  }

  float* out_row = q_rot_out + row * q_rot_stride_0 + head * q_rot_stride_1;
  out_row[tid] = q_rot[tid] * kInvSqrtLatentDim * signs2[tid];
}

__global__ void turboquant_dense_mla_decode_2p5_stage1_rotated_kernel(
    const float* __restrict__ q_rotated,
    const bf16_t* __restrict__ q_rope,
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    float* __restrict__ mid,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    int64_t num_rows,
    int64_t num_heads,
    int64_t topk,
    int64_t num_splits,
    int64_t q_rot_stride_0,
    int64_t q_rot_stride_1,
    int64_t q_rope_stride_0,
    int64_t q_rope_stride_1,
    int64_t compressed_stride_0,
    int64_t page_table_stride_0,
    int64_t mid_stride_0,
    int64_t mid_stride_1,
    int64_t mid_stride_2,
    float sm_scale) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads || split >= num_splits) return;

  __shared__ float acc[kLatentDim];
  __shared__ float score_latent[kLatentDim];
  __shared__ float score_rope[kLatentDim];
  __shared__ float softmax_state[4];

  const float* q_rot_row = q_rotated + row * q_rot_stride_0 + head * q_rot_stride_1;
  acc[tid] = 0.0f;
  if (tid == 0) {
    softmax_state[0] = kNegInf;
    softmax_state[1] = 0.0f;
  }
  __syncthreads();

  const int64_t chunk = (topk + num_splits - 1) / num_splits;
  const int64_t begin = split * chunk;
  const int64_t end = min(begin + chunk, topk);
  const bf16_t* q_rope_row = q_rope + row * q_rope_stride_0 + head * q_rope_stride_1;
  const int32_t* pages = page_table + row * page_table_stride_0;

  for (int64_t col = begin; col < end; ++col) {
    const int32_t page = pages[col];
    const bool valid = page >= 0;
    const uint8_t* slot = compressed + (valid ? static_cast<int64_t>(page) : 0) * compressed_stride_0;

    const int group = tid >> 7;
    const int channel = tid & 127;
    const uint8_t* group_ptr = slot + group * 36;
    float centroid;
    if (channel < 32) {
      centroid = centroids_high[unpack_3bit(group_ptr, channel)];
    } else {
      centroid = centroids_low[unpack_2bit(group_ptr + 12, channel - 32)];
    }

    const half norm_half = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
    const float norm = __half2float(norm_half);
    score_latent[tid] = valid ? q_rot_row[tid] * centroid : 0.0f;
    if (tid < kRopeDim && valid) {
      const bf16_t* rope = reinterpret_cast<const bf16_t*>(slot + kPackedBytes2p5 + kNormBytes);
      score_rope[tid] = bf16_to_float(q_rope_row[tid]) * bf16_to_float(rope[tid]);
    } else {
      score_rope[tid] = 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int stride = 256; stride > 0; stride >>= 1) {
      if (tid < stride) {
        score_latent[tid] += score_latent[tid + stride];
        score_rope[tid] += score_rope[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float score = valid ? (norm * score_latent[0] + score_rope[0]) * sm_scale : kNegInf;
      const float old_m = softmax_state[0];
      const float old_l = softmax_state[1];
      const float new_m = fmaxf(old_m, score);
      const float alpha = __expf(old_m - new_m);
      const float beta = __expf(score - new_m);
      softmax_state[0] = new_m;
      softmax_state[1] = old_l * alpha + beta;
      softmax_state[2] = alpha;
      softmax_state[3] = beta;
    }
    __syncthreads();

    acc[tid] = acc[tid] * softmax_state[2] + softmax_state[3] * norm * centroid;
    __syncthreads();
  }

  float* mid_row = mid + row * mid_stride_0 + head * mid_stride_1 + split * mid_stride_2;
  if (tid == 0) {
    mid_row[0] = softmax_state[0];
    mid_row[1] = softmax_state[1];
  }
  mid_row[2 + tid] = acc[tid];
}

__global__ void turboquant_dense_mla_decode_2p5_stage1_rotated_fast_kernel(
    const float* __restrict__ q_rotated,
    const bf16_t* __restrict__ q_rope,
    const uint8_t* __restrict__ compressed,
    const int32_t* __restrict__ page_table,
    float* __restrict__ mid,
    const float* __restrict__ centroids_high,
    const float* __restrict__ centroids_low,
    int64_t num_rows,
    int64_t num_heads,
    int64_t topk,
    int64_t num_splits,
    int64_t q_rot_stride_0,
    int64_t q_rot_stride_1,
    int64_t q_rope_stride_0,
    int64_t q_rope_stride_1,
    int64_t compressed_stride_0,
    int64_t page_table_stride_0,
    int64_t mid_stride_0,
    int64_t mid_stride_1,
    int64_t mid_stride_2,
    float sm_scale) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads || split >= num_splits) return;

  __shared__ float reduce_scratch[4];
  __shared__ float softmax_state[4];

  const float* q_rot_row = q_rotated + row * q_rot_stride_0 + head * q_rot_stride_1;
  const bf16_t* q_rope_row = q_rope + row * q_rope_stride_0 + head * q_rope_stride_1;
  const int32_t* pages = page_table + row * page_table_stride_0;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  if (tid == 0) {
    softmax_state[0] = kNegInf;
    softmax_state[1] = 0.0f;
  }
  __syncthreads();

  const int64_t chunk = (topk + num_splits - 1) / num_splits;
  const int64_t begin = split * chunk;
  const int64_t end = min(begin + chunk, topk);

  for (int64_t col = begin; col < end; ++col) {
    const int32_t page = pages[col];
    const bool valid = page >= 0;
    const uint8_t* slot = compressed + (valid ? static_cast<int64_t>(page) : 0) * compressed_stride_0;

    const uint8_t* group0 = slot;
    const uint8_t* group1 = slot + 36;
    const uint8_t* group2 = slot + 72;
    const uint8_t* group3 = slot + 108;

    float c0;
    float c1;
    float c2;
    float c3;
    if (tid < 32) {
      c0 = centroids_high[unpack_3bit(group0, tid)];
      c1 = centroids_high[unpack_3bit(group1, tid)];
      c2 = centroids_high[unpack_3bit(group2, tid)];
      c3 = centroids_high[unpack_3bit(group3, tid)];
    } else {
      c0 = centroids_low[unpack_2bit(group0 + 12, tid - 32)];
      c1 = centroids_low[unpack_2bit(group1 + 12, tid - 32)];
      c2 = centroids_low[unpack_2bit(group2 + 12, tid - 32)];
      c3 = centroids_low[unpack_2bit(group3 + 12, tid - 32)];
    }

    float latent_part =
        q_rot_row[tid] * c0 +
        q_rot_row[128 + tid] * c1 +
        q_rot_row[256 + tid] * c2 +
        q_rot_row[384 + tid] * c3;
    float rope_part = 0.0f;
    if (tid < kRopeDim && valid) {
      const bf16_t* rope = reinterpret_cast<const bf16_t*>(slot + kPackedBytes2p5 + kNormBytes);
      rope_part = bf16_to_float(q_rope_row[tid]) * bf16_to_float(rope[tid]);
    }
    if (!valid) {
      latent_part = 0.0f;
    }

    const float latent_sum = block_reduce_sum_128(latent_part, reduce_scratch);
    const float rope_sum = block_reduce_sum_128(rope_part, reduce_scratch);
    const half norm_half = *reinterpret_cast<const half*>(slot + kPackedBytes2p5);
    const float norm = __half2float(norm_half);

    if (tid == 0) {
      const float score = valid ? (norm * latent_sum + rope_sum) * sm_scale : kNegInf;
      const float old_m = softmax_state[0];
      const float old_l = softmax_state[1];
      const float new_m = fmaxf(old_m, score);
      const float alpha = __expf(old_m - new_m);
      const float beta = __expf(score - new_m);
      softmax_state[0] = new_m;
      softmax_state[1] = old_l * alpha + beta;
      softmax_state[2] = alpha;
      softmax_state[3] = beta;
    }
    __syncthreads();

    const float alpha = softmax_state[2];
    const float beta_norm = softmax_state[3] * norm;
    acc0 = acc0 * alpha + beta_norm * c0;
    acc1 = acc1 * alpha + beta_norm * c1;
    acc2 = acc2 * alpha + beta_norm * c2;
    acc3 = acc3 * alpha + beta_norm * c3;
    __syncthreads();
  }

  float* mid_row = mid + row * mid_stride_0 + head * mid_stride_1 + split * mid_stride_2;
  if (tid == 0) {
    mid_row[0] = softmax_state[0];
    mid_row[1] = softmax_state[1];
  }
  mid_row[2 + tid] = acc0;
  mid_row[2 + 128 + tid] = acc1;
  mid_row[2 + 256 + tid] = acc2;
  mid_row[2 + 384 + tid] = acc3;
}

__global__ void turboquant_dense_mla_decode_2p5_stage2_kernel(
    const float* __restrict__ mid,
    bf16_t* __restrict__ out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int64_t num_rows,
    int64_t num_heads,
    int64_t num_splits,
    int64_t mid_stride_0,
    int64_t mid_stride_1,
    int64_t mid_stride_2,
    int64_t out_stride_0,
    int64_t out_stride_1) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= num_rows || head >= num_heads) return;

  __shared__ float buf[kLatentDim];
  __shared__ float state[2];

  const float* mid_base = mid + row * mid_stride_0 + head * mid_stride_1;
  float m = kNegInf;
  for (int split = 0; split < num_splits; ++split) {
    m = fmaxf(m, mid_base[split * mid_stride_2]);
  }

  float l = 0.0f;
  float value = 0.0f;
  for (int split = 0; split < num_splits; ++split) {
    const float* split_base = mid_base + split * mid_stride_2;
    const float split_m = split_base[0];
    const float split_l = split_base[1];
    const float scale = split_l > 0.0f ? __expf(split_m - m) : 0.0f;
    l += split_l * scale;
    value += split_base[2 + tid] * scale;
  }

  if (tid == 0) {
    state[0] = l;
  }
  __syncthreads();

  buf[tid] = state[0] > 0.0f ? (value / state[0]) * signs2[tid] : 0.0f;
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

  bf16_t* out_row = out + row * out_stride_0 + head * out_stride_1;
  out_row[tid] = __float2bfloat16(buf[tid] * kInvSqrtLatentDim * signs1[tid]);
}

struct TurboQuantDenseMLADecode2p5Kernel {
  static void run(
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2,
      double sm_scale) {
    using namespace host;

    auto R = SymbolicSize{"num_rows"};
    auto H = SymbolicSize{"num_heads"};
    auto K = SymbolicSize{"topk"};
    auto S = SymbolicSize{"num_slots"};
    auto q_nope_stride_0 = SymbolicSize{"q_nope_stride_0"};
    auto q_nope_stride_1 = SymbolicSize{"q_nope_stride_1"};
    auto q_rope_stride_0 = SymbolicSize{"q_rope_stride_0"};
    auto q_rope_stride_1 = SymbolicSize{"q_rope_stride_1"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto page_table_stride_0 = SymbolicSize{"page_table_stride_0"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto out_stride_1 = SymbolicSize{"out_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_nope_stride_0, q_nope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({R, H, kRopeDim})
        .with_strides({q_rope_stride_0, q_rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_rope);
    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({R, K})
        .with_strides({page_table_stride_0, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({R, H, kLatentDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    if (R.unwrap() == 0 || H.unwrap() == 0 || K.unwrap() == 0) return;

    dim3 grid(R.unwrap(), H.unwrap());
    LaunchKernel(grid, kLatentDim, device.unwrap())(
        turboquant_dense_mla_decode_2p5_kernel,
        static_cast<const bf16_t*>(q_nope.data_ptr()),
        static_cast<const bf16_t*>(q_rope.data_ptr()),
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        K.unwrap(),
        q_nope_stride_0.unwrap(),
        q_nope_stride_1.unwrap(),
        q_rope_stride_0.unwrap(),
        q_rope_stride_1.unwrap(),
        compressed_stride_0.unwrap(),
        page_table_stride_0.unwrap(),
        out_stride_0.unwrap(),
        out_stride_1.unwrap(),
        static_cast<float>(sm_scale));
  }
};

struct TurboQuantDenseMLADecode2p5SplitKernel {
  static void run(
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView mid,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2,
      double sm_scale) {
    using namespace host;

    auto R = SymbolicSize{"num_rows"};
    auto H = SymbolicSize{"num_heads"};
    auto K = SymbolicSize{"topk"};
    auto S = SymbolicSize{"num_slots"};
    auto P = SymbolicSize{"num_splits"};
    auto q_nope_stride_0 = SymbolicSize{"q_nope_stride_0"};
    auto q_nope_stride_1 = SymbolicSize{"q_nope_stride_1"};
    auto q_rope_stride_0 = SymbolicSize{"q_rope_stride_0"};
    auto q_rope_stride_1 = SymbolicSize{"q_rope_stride_1"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto page_table_stride_0 = SymbolicSize{"page_table_stride_0"};
    auto mid_stride_0 = SymbolicSize{"mid_stride_0"};
    auto mid_stride_1 = SymbolicSize{"mid_stride_1"};
    auto mid_stride_2 = SymbolicSize{"mid_stride_2"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto out_stride_1 = SymbolicSize{"out_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_nope_stride_0, q_nope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({R, H, kRopeDim})
        .with_strides({q_rope_stride_0, q_rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_rope);
    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({R, K})
        .with_strides({page_table_stride_0, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({R, H, P, kLatentDim + 2})
        .with_strides({mid_stride_0, mid_stride_1, mid_stride_2, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(mid);
    TensorMatcher({R, H, kLatentDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    if (R.unwrap() == 0 || H.unwrap() == 0 || K.unwrap() == 0 || P.unwrap() == 0) return;

    LaunchKernel(dim3(R.unwrap(), H.unwrap(), P.unwrap()), kLatentDim, device.unwrap())(
        turboquant_dense_mla_decode_2p5_stage1_kernel,
        static_cast<const bf16_t*>(q_nope.data_ptr()),
        static_cast<const bf16_t*>(q_rope.data_ptr()),
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<float*>(mid.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        K.unwrap(),
        P.unwrap(),
        q_nope_stride_0.unwrap(),
        q_nope_stride_1.unwrap(),
        q_rope_stride_0.unwrap(),
        q_rope_stride_1.unwrap(),
        compressed_stride_0.unwrap(),
        page_table_stride_0.unwrap(),
        mid_stride_0.unwrap(),
        mid_stride_1.unwrap(),
        mid_stride_2.unwrap(),
        static_cast<float>(sm_scale));

    LaunchKernel(dim3(R.unwrap(), H.unwrap()), kLatentDim, device.unwrap())(
        turboquant_dense_mla_decode_2p5_stage2_kernel,
        static_cast<const float*>(mid.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        P.unwrap(),
        mid_stride_0.unwrap(),
        mid_stride_1.unwrap(),
        mid_stride_2.unwrap(),
        out_stride_0.unwrap(),
        out_stride_1.unwrap());
  }
};

struct TurboQuantDenseMLARotateQueryKernel {
  static void run(
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rotated,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2) {
    using namespace host;

    auto R = SymbolicSize{"num_rows"};
    auto H = SymbolicSize{"num_heads"};
    auto q_nope_stride_0 = SymbolicSize{"q_nope_stride_0"};
    auto q_nope_stride_1 = SymbolicSize{"q_nope_stride_1"};
    auto q_rot_stride_0 = SymbolicSize{"q_rot_stride_0"};
    auto q_rot_stride_1 = SymbolicSize{"q_rot_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_nope_stride_0, q_nope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_nope);
    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_rot_stride_0, q_rot_stride_1, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(q_rotated);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    if (R.unwrap() == 0 || H.unwrap() == 0) return;

    LaunchKernel(dim3(R.unwrap(), H.unwrap()), kLatentDim, device.unwrap())(
        turboquant_dense_mla_rotate_query_kernel,
        static_cast<const bf16_t*>(q_nope.data_ptr()),
        static_cast<float*>(q_rotated.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        q_nope_stride_0.unwrap(),
        q_nope_stride_1.unwrap(),
        q_rot_stride_0.unwrap(),
        q_rot_stride_1.unwrap());
  }
};

struct TurboQuantDenseMLADecode2p5SplitRotatedKernel {
  static void run(
      tvm::ffi::TensorView q_rotated,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView compressed,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView mid,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView centroids_high,
      tvm::ffi::TensorView centroids_low,
      tvm::ffi::TensorView signs1,
      tvm::ffi::TensorView signs2,
      double sm_scale) {
    using namespace host;

    auto R = SymbolicSize{"num_rows"};
    auto H = SymbolicSize{"num_heads"};
    auto K = SymbolicSize{"topk"};
    auto S = SymbolicSize{"num_slots"};
    auto P = SymbolicSize{"num_splits"};
    auto q_rot_stride_0 = SymbolicSize{"q_rot_stride_0"};
    auto q_rot_stride_1 = SymbolicSize{"q_rot_stride_1"};
    auto q_rope_stride_0 = SymbolicSize{"q_rope_stride_0"};
    auto q_rope_stride_1 = SymbolicSize{"q_rope_stride_1"};
    auto compressed_stride_0 = SymbolicSize{"compressed_stride_0"};
    auto page_table_stride_0 = SymbolicSize{"page_table_stride_0"};
    auto mid_stride_0 = SymbolicSize{"mid_stride_0"};
    auto mid_stride_1 = SymbolicSize{"mid_stride_1"};
    auto mid_stride_2 = SymbolicSize{"mid_stride_2"};
    auto out_stride_0 = SymbolicSize{"out_stride_0"};
    auto out_stride_1 = SymbolicSize{"out_stride_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, H, kLatentDim})
        .with_strides({q_rot_stride_0, q_rot_stride_1, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(q_rotated);
    TensorMatcher({R, H, kRopeDim})
        .with_strides({q_rope_stride_0, q_rope_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(q_rope);
    TensorMatcher({S, 1, kSlotBytes2p5})
        .with_strides({compressed_stride_0, kSlotBytes2p5, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(compressed);
    TensorMatcher({R, K})
        .with_strides({page_table_stride_0, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({R, H, P, kLatentDim + 2})
        .with_strides({mid_stride_0, mid_stride_1, mid_stride_2, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(mid);
    TensorMatcher({R, H, kLatentDim})
        .with_strides({out_stride_0, out_stride_1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);
    TensorMatcher({8}).with_dtype<float>().with_device(device).verify(centroids_high);
    TensorMatcher({4}).with_dtype<float>().with_device(device).verify(centroids_low);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs1);
    TensorMatcher({kLatentDim}).with_dtype<float>().with_device(device).verify(signs2);

    if (R.unwrap() == 0 || H.unwrap() == 0 || K.unwrap() == 0 || P.unwrap() == 0) return;

    LaunchKernel(dim3(R.unwrap(), H.unwrap(), P.unwrap()), 128, device.unwrap())(
        turboquant_dense_mla_decode_2p5_stage1_rotated_fast_kernel,
        static_cast<const float*>(q_rotated.data_ptr()),
        static_cast<const bf16_t*>(q_rope.data_ptr()),
        static_cast<const uint8_t*>(compressed.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        static_cast<float*>(mid.data_ptr()),
        static_cast<const float*>(centroids_high.data_ptr()),
        static_cast<const float*>(centroids_low.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        K.unwrap(),
        P.unwrap(),
        q_rot_stride_0.unwrap(),
        q_rot_stride_1.unwrap(),
        q_rope_stride_0.unwrap(),
        q_rope_stride_1.unwrap(),
        compressed_stride_0.unwrap(),
        page_table_stride_0.unwrap(),
        mid_stride_0.unwrap(),
        mid_stride_1.unwrap(),
        mid_stride_2.unwrap(),
        static_cast<float>(sm_scale));

    LaunchKernel(dim3(R.unwrap(), H.unwrap()), kLatentDim, device.unwrap())(
        turboquant_dense_mla_decode_2p5_stage2_kernel,
        static_cast<const float*>(mid.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(signs1.data_ptr()),
        static_cast<const float*>(signs2.data_ptr()),
        R.unwrap(),
        H.unwrap(),
        P.unwrap(),
        mid_stride_0.unwrap(),
        mid_stride_1.unwrap(),
        mid_stride_2.unwrap(),
        out_stride_0.unwrap(),
        out_stride_1.unwrap());
  }
};

}  // namespace
