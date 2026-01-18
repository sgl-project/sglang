
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>

constexpr int WARP_SIZE = 64;

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor(val, mask);
  }
  return val;
}

template <typename T, int NUM_WARPS>
__inline__ __device__ T block_reduce_sum(T val) {
  if constexpr (NUM_WARPS == 1) {
    return warp_reduce_sum<T>(val);
  } 
  static __shared__ T shared[NUM_WARPS]; // Shared memory for partial sums
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warp_reduce_sum<T>(val);     // Each warp performs partial reduction

  if (lane == 0) {
    shared[wid] = val;                // Write reduced value to shared memory
  }
  __syncthreads();                    // Wait for all partial reductions

  // Read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

  if (wid == 0) {
    val = warp_reduce_sum<T>(val);    // Final reduce within first warp
    shared[0] = val;
  }
  __syncthreads();
  return shared[0];

}

template <int NUM_WARPS>
__global__ void allocate_decode_kernel(
    int64_t* seq_lens_ptr,      // [bs]
    int64_t* last_loc_ptr,      // [bs]
    int64_t* free_page_ptr,     // [num_free_pages]
    int64_t* out_indices_ptr,   // [bs]
    int page_size) {
      int64_t batch_id = blockIdx.x;
      int64_t self_seqlen = seq_lens_ptr[batch_id];
      int64_t self_pre_seqlen = self_seqlen - 1;
      int64_t new_page_for_alloc = (self_seqlen + page_size - 1) / page_size - (self_pre_seqlen + page_size - 1) / page_size;
      if (new_page_for_alloc <= 0) {
        out_indices_ptr[batch_id] = last_loc_ptr[batch_id] + 1;
        return ;
      }
      int64_t new_pages = 0;
      for (int i = threadIdx.x; i <= batch_id; i += blockDim.x) {
        int64_t seq_len = seq_lens_ptr[i];
        int64_t pre_seq_len = seq_len - 1;
        int64_t num_pages_before = (pre_seq_len + page_size - 1) / page_size;
        int64_t num_pages_after = (seq_len + page_size - 1) / page_size;
        new_pages += (num_pages_after - num_pages_before);
      }

      new_pages = block_reduce_sum<int64_t, NUM_WARPS>(new_pages);
      if (threadIdx.x == 0) {
        out_indices_ptr[batch_id] = free_page_ptr[new_pages - new_page_for_alloc] * page_size;
      }
  }


template <int NUM_WARPS>
__global__ void allocate_extend_kernel(
  int64_t* pre_lens_ptr,
  int64_t* seq_lens_ptr,
  int64_t* last_loc_ptr,
  int64_t* free_page_ptr,
  int64_t* out_indices_ptr,
  int page_size) {
  int64_t batch_id = blockIdx.x;
  
  int64_t self_pre_len = pre_lens_ptr[batch_id];
  int64_t self_seq_len = seq_lens_ptr[batch_id];
  int64_t self_page_required = (self_seq_len + page_size - 1) / page_size - (self_pre_len + page_size - 1) / page_size;
  int64_t new_pages = 0;
  int64_t extend_len = 0;
  for (int i = threadIdx.x; i < batch_id; i += blockDim.x) {
    int64_t pre_len = pre_lens_ptr[i];
    int64_t seq_len = seq_lens_ptr[i];
    extend_len += seq_len - pre_len;

    int64_t before_pages = (pre_len + page_size - 1) / page_size;
    int64_t after_pages = (seq_len + page_size - 1) / page_size;
    new_pages += after_pages - before_pages;
  }
  int64_t output_start_loc = block_reduce_sum<int64_t, NUM_WARPS>(extend_len);
  new_pages = block_reduce_sum<int64_t, NUM_WARPS>(new_pages);

  // Part1: fill the old part
  int64_t last_loc = last_loc_ptr[batch_id];
  int prev_blocks_seq_len = (self_pre_len + page_size - 1) / page_size * page_size;
  int64_t num_part1 = min(self_seq_len, prev_blocks_seq_len) - self_pre_len;

  for (int i = threadIdx.x; i < num_part1; i += blockDim.x) {
    out_indices_ptr[output_start_loc + i] = last_loc + 1 + i;
  }
  if (self_pre_len + num_part1 == self_seq_len) return ;

  // Part2: fill the new part
  int num_part2 = self_seq_len / page_size * page_size - prev_blocks_seq_len;

  for (int i = threadIdx.x; i < num_part2; i += blockDim.x) {
    int64_t page_start = free_page_ptr[new_pages + i / page_size];
    out_indices_ptr[output_start_loc + num_part1 + i] = page_start * page_size + i % page_size;
  }

  if (self_pre_len + num_part1 + num_part2 == self_seq_len) return ;

  // part3: fill the rest part
  int num_part3 = self_seq_len - self_seq_len / page_size * page_size;
  int64_t start_loc = free_page_ptr[new_pages + self_page_required - 1];
  for (int i = threadIdx.x; i < num_part3; i += blockDim.x) {
    out_indices_ptr[output_start_loc + num_part1 + num_part2 + i] = start_loc * page_size + i;
  }
}

#define LAUNCH_EXTEND_ALLOCATE_KERNEL(FUNC, CTA_SIZE) \
  FUNC<CTA_SIZE / WARP_SIZE><<<batch_size, CTA_SIZE>>>( \
      pre_lens_ptr, \
      seq_lens_ptr, \
      last_loc_ptr, \
      free_page_ptr, \
      out_indices_ptr, \
      page_size);

#define LAUNCH_DECODE_ALLOCATE_KERNEL(FUNC, CTA_SIZE) \
  FUNC<CTA_SIZE / WARP_SIZE><<<batch_size, CTA_SIZE>>>( \
      seq_lens_ptr, \
      last_loc_ptr, \
      free_page_ptr, \
      out_indices_ptr, \
      page_size);


void allocate_decode(
    at::Tensor& seq_lens,      // [bs]
    at::Tensor& last_loc,      // [bs]
    at::Tensor& free_page,     // [num_free_pages]
    at::Tensor& out_indices,   // [bs]
    int64_t page_size) {

    auto batch_size = seq_lens.size(0);
    TORCH_CHECK(last_loc.size(0) == batch_size);
    TORCH_CHECK(out_indices.size(0) == batch_size);
    TORCH_CHECK(last_loc.dtype() == at::kLong);
    TORCH_CHECK(seq_lens.dtype() == at::kLong);
    TORCH_CHECK(free_page.dtype() == at::kLong);
    TORCH_CHECK(out_indices.dtype() == at::kLong);

    auto seq_lens_ptr = seq_lens.data_ptr<int64_t>();
    auto last_loc_ptr = last_loc.data_ptr<int64_t>();
    auto free_page_ptr = free_page.data_ptr<int64_t>();
    auto out_indices_ptr = out_indices.data_ptr<int64_t>();

    if (batch_size <= 64) {
      LAUNCH_DECODE_ALLOCATE_KERNEL(allocate_decode_kernel, 64);
    } else if (batch_size <= 128 && batch_size > 64) {
      LAUNCH_DECODE_ALLOCATE_KERNEL(allocate_decode_kernel, 128);
    } else if (batch_size > 128 && batch_size <= 256) {
      LAUNCH_DECODE_ALLOCATE_KERNEL(allocate_decode_kernel, 256);
    } else if (batch_size > 256 && batch_size <= 512) {
      LAUNCH_DECODE_ALLOCATE_KERNEL(allocate_decode_kernel, 512);
    }
}

void allocate_extend(
    at::Tensor& pre_lens,
    at::Tensor& seq_lens,
    at::Tensor& last_loc,
    at::Tensor& free_page,
    at::Tensor& out_indices,
    int64_t page_size) {

    auto batch_size = seq_lens.size(0);
    TORCH_CHECK(pre_lens.size(0) == batch_size);
    TORCH_CHECK(last_loc.size(0) == batch_size);
    TORCH_CHECK(last_loc.dtype() == at::kLong);
    TORCH_CHECK(pre_lens.dtype() == at::kLong);
    TORCH_CHECK(seq_lens.dtype() == at::kLong);
    TORCH_CHECK(free_page.dtype() == at::kLong);
    TORCH_CHECK(out_indices.dtype() == at::kLong);

    auto pre_lens_ptr = pre_lens.data_ptr<int64_t>();
    auto seq_lens_ptr = seq_lens.data_ptr<int64_t>();
    auto last_loc_ptr = last_loc.data_ptr<int64_t>();
    auto free_page_ptr = free_page.data_ptr<int64_t>();
    auto out_indices_ptr = out_indices.data_ptr<int64_t>();

    if (batch_size <= 64) {
      LAUNCH_EXTEND_ALLOCATE_KERNEL(allocate_extend_kernel, 64);
    } else if (batch_size <= 128 && batch_size > 64) {
      LAUNCH_EXTEND_ALLOCATE_KERNEL(allocate_extend_kernel, 128);
    } else if (batch_size > 128 && batch_size <= 256) {
      LAUNCH_EXTEND_ALLOCATE_KERNEL(allocate_extend_kernel, 256);
    } else if (batch_size > 256 && batch_size <= 512) {
      LAUNCH_EXTEND_ALLOCATE_KERNEL(allocate_extend_kernel, 512);
    }
}
