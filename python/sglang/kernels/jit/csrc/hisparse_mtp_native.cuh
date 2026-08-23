#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_runtime.h>
#include <stdint.h>

namespace sglang {

constexpr int kMtpWarpSize = 32;

__device__ __forceinline__ void
copy_mtp_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const auto* src = static_cast<const uint64_t*>(src_addr);
  auto* dst = static_cast<uint64_t*>(dst_addr);
  const int64_t chunks = item_size_bytes / static_cast<int64_t>(sizeof(uint64_t));
  for (int64_t chunk = lane_id; chunk < chunks; chunk += kMtpWarpSize) {
    uint64_t value;
    asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(value) : "l"(src + chunk) : "memory");
    asm volatile("st.global.cg.b64 [%0], %1;" : : "l"(dst + chunk), "l"(value) : "memory");
  }
}

template <
    int BLOCK_SIZE,
    int NUM_TOP_K,
    int HOT_BUFFER_SIZE,
    bool IsMLA,
    bool IsDsv4Layout,
    typename SeqLensT,
    typename ReqPoolIndicesT>
__global__ void load_cache_to_device_buffer_kernel(
    const int32_t* __restrict__ top_k_tokens,
    const int32_t* __restrict__ device_buffer_tokens,
    const int64_t* __restrict__ host_cache_locs,
    const int32_t* __restrict__ device_buffer_locs,
    const void* __restrict__ host_cache_k,
    const void* __restrict__ host_cache_v,
    void* __restrict__ device_buffer_k,
    void* __restrict__ device_buffer_v,
    int32_t* __restrict__ top_k_device_locs,
    const ReqPoolIndicesT* __restrict__ req_pool_indices,
    const SeqLensT* __restrict__ seq_lens,
    const int64_t* __restrict__ mtp_staging_locs,
    const int32_t* __restrict__ num_real_reqs,
    int64_t buffer_stride,
    int64_t host_stride,
    int64_t top_k_stride,
    int64_t output_stride,
    int64_t staging_stride,
    int64_t page_size,
    int64_t item_size_bytes,
    int64_t num_steps) {
  static_assert(IsMLA, "Native HiSparse MTP materialization is MLA-only.");
  static_assert(!IsDsv4Layout, "Native HiSparse MTP materialization does not support DSV4 layout.");
  static_assert(BLOCK_SIZE % kMtpWarpSize == 0);
  constexpr int kWarps = BLOCK_SIZE / kMtpWarpSize;

  const int32_t bid = blockIdx.x;
  if (bid >= num_real_reqs[0]) return;

  const int32_t lane = threadIdx.x % kMtpWarpSize;
  const int32_t warp = threadIdx.x / kMtpWarpSize;
  const int64_t rid = static_cast<int64_t>(req_pool_indices[bid]);
  const int64_t occurrence_count = num_steps * NUM_TOP_K;

  const int32_t* req_tokens = device_buffer_tokens + rid * buffer_stride;
  const int32_t* req_locs = device_buffer_locs + rid * buffer_stride;
  const int64_t* req_host_locs = host_cache_locs + rid * host_stride;
  const int64_t* req_staging_locs = mtp_staging_locs + rid * staging_stride;
  const int32_t* req_top_k = top_k_tokens + bid * top_k_stride;
  int32_t* req_output = top_k_device_locs + bid * output_stride;

  for (int64_t occurrence = warp; occurrence < occurrence_count; occurrence += kWarps) {
    const int64_t step = occurrence / NUM_TOP_K;
    const int32_t token = req_top_k[occurrence];
    const int64_t seq_len = static_cast<int64_t>(seq_lens[bid * num_steps + step]);

    int64_t src_loc = -1;
    int64_t dst_loc = -1;
    int32_t resolved_loc = -1;
    if (lane == 0 && token >= 0 && token < seq_len) {
      // Current verify rows live in the request's fixed extra-page overlay.
      for (int64_t slot = HOT_BUFFER_SIZE; slot < HOT_BUFFER_SIZE + page_size; ++slot) {
        if (req_tokens[slot] == token) {
          resolved_loc = req_locs[slot];
          break;
        }
      }

      // Rows still in their original hot position are already stable because
      // this materialization path never mutates the native hot slice.
      if (resolved_loc < 0 && token < HOT_BUFFER_SIZE && req_tokens[token] == token) {
        resolved_loc = req_locs[token];
      }

      // Every other historical occurrence gets its own stable row. Duplicate
      // TopK entries may copy twice, but no later speculative row can overwrite
      // a physical page table returned to an earlier row.
      if (resolved_loc < 0) {
        src_loc = req_host_locs[token];
        if (src_loc >= 0) {
          dst_loc = req_staging_locs[occurrence];
          resolved_loc = static_cast<int32_t>(dst_loc);
        }
      }
      req_output[occurrence] = resolved_loc;
    }

    src_loc = __shfl_sync(0xffffffff, src_loc, 0);
    dst_loc = __shfl_sync(0xffffffff, dst_loc, 0);
    if (src_loc >= 0 && dst_loc > 0) {
      const auto* src = static_cast<const char*>(host_cache_k) + src_loc * item_size_bytes;
      auto* dst = static_cast<char*>(device_buffer_k) + dst_loc * item_size_bytes;
      copy_mtp_item_warp(lane, src, dst, item_size_bytes);
    }
  }
}

template <int BLOCK_SIZE, int NUM_TOP_K, int HOT_BUFFER_SIZE, bool IsMLA, bool IsDsv4Layout>
void load_cache_to_device_buffer(
    tvm::ffi::TensorView top_k_tokens,
    tvm::ffi::TensorView device_buffer_tokens,
    tvm::ffi::TensorView host_cache_locs,
    tvm::ffi::TensorView device_buffer_locs,
    tvm::ffi::TensorView host_cache_k,
    tvm::ffi::TensorView host_cache_v,
    tvm::ffi::TensorView device_buffer_k,
    tvm::ffi::TensorView device_buffer_v,
    tvm::ffi::TensorView top_k_device_locs,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView mtp_staging_locs,
    tvm::ffi::TensorView num_real_reqs,
    int64_t page_size,
    int64_t item_size_bytes,
    int64_t num_steps) {
  using namespace host;

  RuntimeCheck(item_size_bytes > 0 && item_size_bytes % 8 == 0, "MTP item size must be 8-byte aligned.");
  RuntimeCheck(top_k_tokens.ndim() == 3, "MTP TopK must have shape [request, step, topk].");
  RuntimeCheck(top_k_tokens.shape()[1] == num_steps, "MTP TopK step count mismatch.");
  RuntimeCheck(top_k_tokens.shape()[2] == NUM_TOP_K, "MTP TopK width mismatch.");
  RuntimeCheck(
      top_k_device_locs.ndim() == 3 && top_k_device_locs.shape()[0] == top_k_tokens.shape()[0] &&
          top_k_device_locs.shape()[1] == top_k_tokens.shape()[1] &&
          top_k_device_locs.shape()[2] == top_k_tokens.shape()[2],
      "MTP output shape mismatch.");
  RuntimeCheck(
      mtp_staging_locs.ndim() == 2 && mtp_staging_locs.shape()[1] >= num_steps * NUM_TOP_K,
      "MTP staging capacity is smaller than steps * topk.");

  const int64_t batch_size = top_k_tokens.shape()[0];
  const int64_t buffer_stride = device_buffer_tokens.strides()[0];
  const int64_t host_stride = host_cache_locs.strides()[0];
  const int64_t top_k_stride = top_k_tokens.strides()[0];
  const int64_t output_stride = top_k_device_locs.strides()[0];
  const int64_t staging_stride = mtp_staging_locs.strides()[0];
  const auto device = LaunchKernel::resolve_device(top_k_tokens.device());

  auto launch = [&](auto kernel, const auto* seq_lens_ptr, const auto* req_pool_indices_ptr) {
    LaunchKernel(batch_size, BLOCK_SIZE, device)(
        kernel,
        static_cast<const int32_t*>(top_k_tokens.data_ptr()),
        static_cast<const int32_t*>(device_buffer_tokens.data_ptr()),
        static_cast<const int64_t*>(host_cache_locs.data_ptr()),
        static_cast<const int32_t*>(device_buffer_locs.data_ptr()),
        host_cache_k.data_ptr(),
        host_cache_v.data_ptr(),
        device_buffer_k.data_ptr(),
        device_buffer_v.data_ptr(),
        static_cast<int32_t*>(top_k_device_locs.data_ptr()),
        req_pool_indices_ptr,
        seq_lens_ptr,
        static_cast<const int64_t*>(mtp_staging_locs.data_ptr()),
        static_cast<const int32_t*>(num_real_reqs.data_ptr()),
        buffer_stride,
        host_stride,
        top_k_stride,
        output_stride,
        staging_stride,
        page_size,
        item_size_bytes,
        num_steps);
  };

  const bool seq_i64 = seq_lens.dtype().bits == 64;
  const bool req_i64 = req_pool_indices.dtype().bits == 64;
  if (seq_i64 && req_i64) {
    launch(
        load_cache_to_device_buffer_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            IsMLA,
            IsDsv4Layout,
            int64_t,
            int64_t>,
        static_cast<const int64_t*>(seq_lens.data_ptr()),
        static_cast<const int64_t*>(req_pool_indices.data_ptr()));
  } else if (seq_i64) {
    launch(
        load_cache_to_device_buffer_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            IsMLA,
            IsDsv4Layout,
            int64_t,
            int32_t>,
        static_cast<const int64_t*>(seq_lens.data_ptr()),
        static_cast<const int32_t*>(req_pool_indices.data_ptr()));
  } else if (req_i64) {
    launch(
        load_cache_to_device_buffer_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            IsMLA,
            IsDsv4Layout,
            int32_t,
            int64_t>,
        static_cast<const int32_t*>(seq_lens.data_ptr()),
        static_cast<const int64_t*>(req_pool_indices.data_ptr()));
  } else {
    launch(
        load_cache_to_device_buffer_kernel<
            BLOCK_SIZE,
            NUM_TOP_K,
            HOT_BUFFER_SIZE,
            IsMLA,
            IsDsv4Layout,
            int32_t,
            int32_t>,
        static_cast<const int32_t*>(seq_lens.data_ptr()),
        static_cast<const int32_t*>(req_pool_indices.data_ptr()));
  }
}

}  // namespace sglang
