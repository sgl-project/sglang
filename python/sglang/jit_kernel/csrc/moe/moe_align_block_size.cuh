#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

static constexpr int kWarpSize = 32;

__host__ __forceinline__ uint32_t next_pow2(uint32_t x) noexcept {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

__device__ __forceinline__ int warp_exclusive_scan(int v) {
  int original = v;
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    int n = __shfl_up_sync(0xffffffffu, v, offset);
    if ((threadIdx.x & (kWarpSize - 1)) >= offset) v += n;
  }
  return v - original;
}

template <typename IdType>
__global__ void count_and_sort_expert_tokens_kernel(
    const IdType* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    int64_t numel) {
  const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)blockDim.x * gridDim.x;

  for (int64_t i = tid; i < numel; i += stride) {
    int32_t expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
  }
}

// 2-block launch: block 0 = align logic, block 1 = padding fill
template <typename IdType>
__global__ void moe_align_block_size_kernel(
    const IdType* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    int64_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    const int32_t scan_size,
    int32_t max_num_tokens_padded) {
  if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      int32_t numel32 = static_cast<int32_t>(numel);
      for (int32_t i = threadIdx.x; i < max_num_tokens_padded; i += blockDim.x) {
        sorted_token_ids[i] = numel32;
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;
  int32_t* prefix = shared_counts + num_experts;
  int32_t* scan_buf = prefix + num_experts + 1;
  int32_t* warp_sums = scan_buf + scan_size;
  __shared__ int32_t s_total_tokens_post_pad;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }
  __syncthreads();

  for (int64_t i = tid; i < numel; i += stride) {
    int expert_id = static_cast<int>(topk_ids[i]) + 1;
    atomicAdd(&shared_counts[expert_id], 1);
  }
  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = ((count + block_size - 1) / block_size) * block_size;
    scan_buf[tid] = padded_count;
  }
  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }
  __syncthreads();

  const int warp_id = tid / kWarpSize;
  const int lane_id = tid & (kWarpSize - 1);
  const int num_warps_for_scan = (scan_size + kWarpSize - 1) / kWarpSize;

  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = warp_exclusive_scan(v);
  if (lane_id == kWarpSize - 1) warp_sums[warp_id] = pre + v;
  __syncthreads();

  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = warp_exclusive_scan(val);
  }
  __syncthreads();

  int offset = warp_sums[warp_id];
  if (tid < scan_size) scan_buf[tid] = pre + offset;
  __syncthreads();

  if (tid < num_experts) prefix[tid] = scan_buf[tid];
  if (tid == 0) {
    int32_t last_pre = (num_experts > 0) ? scan_buf[num_experts - 1] : 0;
    int32_t last_count = (num_experts > 0) ? shared_counts[num_experts - 1] : 0;
    int32_t last_padded = ((last_count + block_size - 1) / block_size) * block_size;
    prefix[num_experts] = last_pre + last_padded;
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  if (tid <= num_experts) {
    cumsum[tid] = prefix[tid];
  }

  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 2;
  }
}

void moe_align_block_size(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_pad,
    tvm::ffi::TensorView cumsum_buffer,
    int64_t num_experts,
    int64_t block_size,
    bool pad_sorted_token_ids,
    int64_t max_num_tokens_padded) {
  using namespace host;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  auto N = SymbolicSize{"numel"};
  TensorMatcher({N})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(topk_ids);

  const auto numel = static_cast<int64_t>(N.unwrap());
  const auto dev = device.unwrap();

  const int32_t scan_size = static_cast<int32_t>(next_pow2(static_cast<uint32_t>(num_experts)));

  constexpr int kThreads = 1024;

  RuntimeCheck(scan_size <= kThreads, "moe_align_block_size: num_experts too large for single-pass scan, got ", num_experts);

  const size_t shared_mem_size =
      (num_experts + (num_experts + 1) + scan_size + kWarpSize) * sizeof(int32_t);

  LaunchKernel(dim3(2), dim3(kThreads), dev, shared_mem_size)(
      moe_align_block_size_kernel<int32_t>,
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(sorted_token_ids.data_ptr()),
      static_cast<int32_t*>(expert_ids.data_ptr()),
      static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(block_size),
      numel,
      static_cast<int32_t*>(cumsum_buffer.data_ptr()),
      pad_sorted_token_ids,
      scan_size,
      static_cast<int32_t>(max_num_tokens_padded));

  constexpr int kBlockThreads = 256;
  const int64_t num_blocks_sort = std::min((numel + kBlockThreads - 1) / kBlockThreads, (int64_t)65535);

  LaunchKernel(dim3(static_cast<unsigned>(num_blocks_sort)), dim3(kBlockThreads), dev)(
      count_and_sort_expert_tokens_kernel<int32_t>,
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(sorted_token_ids.data_ptr()),
      static_cast<int32_t*>(cumsum_buffer.data_ptr()),
      numel);
}

}  // namespace
