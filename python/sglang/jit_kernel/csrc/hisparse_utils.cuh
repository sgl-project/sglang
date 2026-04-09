#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// Variable-length gather from a 2D buffer into a flat output.
// One block per request. Each request gathers counts[bid] contiguous elements
// starting at column start_col from the row indexed by req_indices[bid].
template <int BLOCK_SIZE>
__global__ void gather_variable_length_kernel(
    const int64_t* __restrict__ buffer,        // (max_reqs, padded_buf_size)
    const int64_t* __restrict__ req_indices,   // (bs,)
    const int32_t* __restrict__ counts,        // (bs,) elements per request
    const int64_t* __restrict__ offsets,        // (bs+1,) exclusive prefix-sum
    int64_t* __restrict__ output,              // (total_elements,)
    int64_t buffer_stride,                      // row stride of buffer
    int64_t start_col,                          // column offset
    int32_t bs) {

  const int bid = blockIdx.x;
  if (bid >= bs) return;

  const int n = counts[bid];
  if (n <= 0) return;

  const int64_t req_idx = req_indices[bid];
  const int64_t out_start = offsets[bid];
  const int64_t* src = buffer + req_idx * buffer_stride + start_col;

  for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
    output[out_start + i] = src[i];
  }
}

template <int BLOCK_SIZE>
void gather_variable_length(
    tvm::ffi::TensorView buffer,
    tvm::ffi::TensorView req_indices,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView output,
    int64_t start_col) {
  using namespace host;

  const int64_t bs = req_indices.shape()[0];
  if (bs == 0) return;
  const int64_t buffer_stride = buffer.strides()[0];
  const auto device = LaunchKernel::resolve_device(buffer.device());

  LaunchKernel(bs, BLOCK_SIZE, device)(
      gather_variable_length_kernel<BLOCK_SIZE>,
      static_cast<const int64_t*>(buffer.data_ptr()),
      static_cast<const int64_t*>(req_indices.data_ptr()),
      static_cast<const int32_t*>(counts.data_ptr()),
      static_cast<const int64_t*>(offsets.data_ptr()),
      static_cast<int64_t*>(output.data_ptr()),
      buffer_stride,
      start_col,
      static_cast<int32_t>(bs));
}


// Per-request finalization of accepted speculative tokens.
// One block per request. Performs three operations in parallel:
// 1. Scatter host_locs into req_to_host_pool at correct token positions
// 2. Update device_mapping: last accepted -> newest_phys, others -> 0
// 3. Output needs_move flag per request
template <int BLOCK_SIZE>
__global__ void finalize_accepted_kernel(
    int64_t* __restrict__ req_to_host_pool,        // (max_reqs, max_seq_len)
    int64_t* __restrict__ device_mapping,           // (total_pool_size,)
    const int64_t* __restrict__ req_pool_indices,   // (bs,)
    const int32_t* __restrict__ seq_lens,           // (bs,)
    const int64_t* __restrict__ host_locs,          // (total_accepted,)
    const int64_t* __restrict__ accepted_locs,      // (total_accepted,) logical cache locs
    const int64_t* __restrict__ device_locs,        // (total_accepted,) physical device locs
    const int64_t* __restrict__ newest_phys,        // (bs,) newest slot per request
    const int64_t* __restrict__ cumsum,             // (bs+1,) exclusive prefix-sum of counts
    int32_t* __restrict__ needs_move,               // (bs,) output
    int64_t* __restrict__ last_accepted_out,        // (bs,) output: last accepted device loc
    int64_t* __restrict__ newest_slot_out,          // (bs,) output: newest slot device loc
    int64_t host_pool_stride,
    int32_t bs) {

  const int bid = blockIdx.x;
  if (bid >= bs) return;

  const int64_t start = cumsum[bid];
  const int64_t end = cumsum[bid + 1];
  const int n = static_cast<int>(end - start);

  if (n == 0) {
    if (threadIdx.x == 0) {
      needs_move[bid] = 0;
    }
    return;
  }

  const int64_t req_idx = req_pool_indices[bid];
  const int32_t seq_len = seq_lens[bid];
  const int64_t newest = newest_phys[bid];

  // 1. Scatter host_locs into req_to_host_pool[req_idx, seq_len-n .. seq_len-1]
  int64_t* host_row = req_to_host_pool + req_idx * host_pool_stride;
  for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
    host_row[seq_len - n + i] = host_locs[start + i];
  }

  // 2. Update device_mapping: last accepted -> newest slot; rest -> 0 (host-only)
  for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
    int64_t loc = accepted_locs[start + i];
    device_mapping[loc] = (i == n - 1) ? newest : 0;
  }

  // 3. Output needs_move flag and device locs for batched KV move
  if (threadIdx.x == 0) {
    int64_t last_dev = device_locs[end - 1];
    needs_move[bid] = (last_dev != newest) ? 1 : 0;
    last_accepted_out[bid] = last_dev;
    newest_slot_out[bid] = newest;
  }
}

template <int BLOCK_SIZE>
void finalize_accepted(
    tvm::ffi::TensorView req_to_host_pool,
    tvm::ffi::TensorView device_mapping,
    tvm::ffi::TensorView req_pool_indices,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView host_locs,
    tvm::ffi::TensorView accepted_locs,
    tvm::ffi::TensorView device_locs,
    tvm::ffi::TensorView newest_phys,
    tvm::ffi::TensorView cumsum,
    tvm::ffi::TensorView needs_move,
    tvm::ffi::TensorView last_accepted_out,
    tvm::ffi::TensorView newest_slot_out) {
  using namespace host;

  const int64_t bs = req_pool_indices.shape()[0];
  if (bs == 0) return;
  const int64_t host_pool_stride = req_to_host_pool.strides()[0];
  const auto device = LaunchKernel::resolve_device(req_pool_indices.device());

  LaunchKernel(bs, BLOCK_SIZE, device)(
      finalize_accepted_kernel<BLOCK_SIZE>,
      static_cast<int64_t*>(req_to_host_pool.data_ptr()),
      static_cast<int64_t*>(device_mapping.data_ptr()),
      static_cast<const int64_t*>(req_pool_indices.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<const int64_t*>(host_locs.data_ptr()),
      static_cast<const int64_t*>(accepted_locs.data_ptr()),
      static_cast<const int64_t*>(device_locs.data_ptr()),
      static_cast<const int64_t*>(newest_phys.data_ptr()),
      static_cast<const int64_t*>(cumsum.data_ptr()),
      static_cast<int32_t*>(needs_move.data_ptr()),
      static_cast<int64_t*>(last_accepted_out.data_ptr()),
      static_cast<int64_t*>(newest_slot_out.data_ptr()),
      host_pool_stride,
      static_cast<int32_t>(bs));
}

}  // namespace
