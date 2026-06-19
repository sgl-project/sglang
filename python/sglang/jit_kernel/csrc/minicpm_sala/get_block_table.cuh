// MiniCPM-SALA sparse attention: build the per-token sparse block table.
//
// Migrated from `3rdparty/sparse_kernel/get_table_kernel.cu`. The original
// CUDA kernels are kept almost verbatim; only the host-side wrappers are
// rewritten from the torch::Tensor + pybind interface to the jit_kernel
// tvm::ffi::TensorView + TensorMatcher/LaunchKernel convention.
//
// The compile-time `kSparseTopK` template parameter replaces the original
// `VALUE_SPLITS_SWITCH(topk, ...)` runtime dispatch: the Python wrapper
// compiles (and caches) one module per supported topk value (96 / 128).

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace minicpm_sala {

// Layout constants (fixed by the MiniCPM-SALA model configuration).
constexpr int kHeadGroup = 2;
constexpr int kSparseBlockSize = 64;
constexpr int kTopkPerBlock = 16;

// topk_idx:        [head_group, token_num, kSparseTopK]  int32
// block_table:     [batch_size, seqlen_q_max]           int32
// token_to_bs:     [token_num]                           int32
// token_pos_in_bs: [token_num]                           int32
// seqlen_q:        [batch_size]                          int32
// out_block_table: [token_num, head_group, kSparseTopK * kSparseBlockSize] int32

template <int kSparseTopK>
__global__ void get_block_table_cuda_v1(
    const int* topk_idx,
    const int* block_table,
    const int* token_to_bs,
    const int* token_pos_in_bs,
    const int* seqlen_q,
    int* out_block_table,
    const int seqlen_q_max,
    const int token_num) {
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= token_num) return;
  int bs = token_to_bs[token_idx];
  int pos_in_bs = token_pos_in_bs[token_idx];

  for (int h = 0; h < kHeadGroup; h++) {
    for (int i = 0; i < kSparseTopK * kSparseBlockSize; i++) {
      int sparse_block_idx = topk_idx[h * token_num * kSparseTopK + token_idx * kSparseTopK + i / kSparseBlockSize];
      if (sparse_block_idx < 0) continue;
      int token_idx_in_batch = sparse_block_idx * kSparseBlockSize + (i % kSparseBlockSize);

      if (token_idx_in_batch < seqlen_q[bs] && token_idx_in_batch < pos_in_bs) {
        out_block_table
            [token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize + h * kSparseTopK * kSparseBlockSize + i] =
                kHeadGroup * block_table[bs * seqlen_q_max + token_idx_in_batch] + h;
      } else {
        out_block_table
            [token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize + h * kSparseTopK * kSparseBlockSize + i] = 0;
      }
    }
  }
}

// 1 thread calc 64 element of out_block_table.
// This allows topk_idx to be read once and all corresponding
// out_block_table elements calculated, reducing memory access.
template <int kSparseTopK>
__global__ void get_block_table_cuda_v2(
    const int* topk_idx,
    const int* block_table,
    const int* token_to_bs,
    const int* token_pos_in_bs,
    const int* seqlen_q,
    int* out_block_table,
    const int seqlen_q_max,
    const int token_num) {
  int token_idx = (blockIdx.x * blockDim.x + threadIdx.x) / (kSparseTopK * kHeadGroup);
  if (token_idx >= token_num) return;
  int head_group_idx = ((blockIdx.x * blockDim.x + threadIdx.x) / kSparseTopK) % kHeadGroup;
  int topk_idx_in_head = (blockIdx.x * blockDim.x + threadIdx.x) % kSparseTopK;
  int bs = token_to_bs[token_idx];
  int pos_in_bs = token_pos_in_bs[token_idx];
  int seqlen_q_bs = seqlen_q[bs];
  int sparse_block_idx =
      topk_idx[head_group_idx * token_num * kSparseTopK + token_idx * kSparseTopK + topk_idx_in_head];

  if (sparse_block_idx < 0) return;
  for (int i = 0; i < kSparseBlockSize; i++) {
    int token_idx_in_batch = sparse_block_idx * kSparseBlockSize + i;

    if (token_idx_in_batch < seqlen_q_bs && token_idx_in_batch < pos_in_bs) {
      out_block_table
          [token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize + head_group_idx * kSparseTopK * kSparseBlockSize +
           topk_idx_in_head * kSparseBlockSize + i] =
              kHeadGroup * block_table[bs * seqlen_q_max + token_idx_in_batch] + head_group_idx;
    } else {
      out_block_table
          [token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize + head_group_idx * kSparseTopK * kSparseBlockSize +
           topk_idx_in_head * kSparseBlockSize + i] = 0;
    }
  }
}

// opt for decode: 1 thread calc 1 element of out_block_table, block size 1024,
// smem 1024 / 64 = 16.
template <int kSparseTopK>
__global__ void get_block_table_cuda_v3(
    const int* topk_idx,
    const int* block_table,
    const int* token_to_bs,
    const int* token_pos_in_bs,
    const int* seqlen_q,
    int* out_block_table,
    const int seqlen_q_max,
    const int token_num) {
  constexpr int kBlockPerTokenHead = kSparseTopK / kTopkPerBlock;
  // calc 16 topk -> 1024 output
  __shared__ int topk_idx_share[kTopkPerBlock];
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  if (threadIdx.x < kTopkPerBlock) {
    topk_idx_share[tidx] = topk_idx[bidx * kTopkPerBlock + tidx];
  }

  __syncthreads();

  const int head_group_idx = (bidx / kBlockPerTokenHead) / token_num;
  const int token_idx = (bidx / kBlockPerTokenHead) % token_num;
  const int topk_idx_in_head = bidx % kBlockPerTokenHead * kTopkPerBlock + tidx / kSparseBlockSize;

  const int sparse_block_idx = topk_idx_share[tidx / kSparseBlockSize];

  const int token_idx_src = sparse_block_idx * kSparseBlockSize + tidx % kSparseBlockSize;
  const int token_idx_dst = token_idx * kHeadGroup * kSparseTopK * kSparseBlockSize +
                            head_group_idx * kSparseTopK * kSparseBlockSize + topk_idx_in_head * kSparseBlockSize +
                            tidx % kSparseBlockSize;

  const int bs = token_to_bs[token_idx];
  const int pos_in_bs = token_pos_in_bs[token_idx];
  const int seqlen_q_bs = seqlen_q[bs];

  if (token_idx_src < seqlen_q_bs && token_idx_src < pos_in_bs) {
    out_block_table[token_idx_dst] = kHeadGroup * block_table[bs * seqlen_q_max + token_idx_src] + head_group_idx;
  } else {
    out_block_table[token_idx_dst] = 0;
  }
}

namespace {

// Validate all inputs that are shared across the three kernel variants and
// bind the symbolic dims (token_num / batch_size / seqlen_q_max). The output
// tensor is pre-allocated and zero-initialized on the Python side.
template <int kSparseTopK>
void verify_inputs(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView topk_idx,
    tvm::ffi::TensorView block_table,
    tvm::ffi::TensorView token_to_bs,
    tvm::ffi::TensorView token_pos_in_bs,
    tvm::ffi::TensorView seqlen_q,
    host::SymbolicSize& token_num,
    host::SymbolicSize& batch_size,
    host::SymbolicSize& seqlen_q_max,
    host::SymbolicDevice& device) {
  using namespace host;
  constexpr int64_t kOutLastDim = static_cast<int64_t>(kSparseTopK) * kSparseBlockSize;

  // topk_idx: [kHeadGroup, token_num, kSparseTopK]
  TensorMatcher({static_cast<int64_t>(kHeadGroup), token_num, static_cast<int64_t>(kSparseTopK)})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(topk_idx);
  // block_table: [batch_size, seqlen_q_max]
  TensorMatcher({batch_size, seqlen_q_max})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(block_table);
  // token_to_bs / token_pos_in_bs: [token_num]
  TensorMatcher({token_num})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(token_to_bs)
      .verify(token_pos_in_bs);
  // seqlen_q: [batch_size]
  TensorMatcher({batch_size})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(seqlen_q);
  // out: [token_num, kHeadGroup, kSparseTopK * kSparseBlockSize]
  TensorMatcher({token_num, static_cast<int64_t>(kHeadGroup), kOutLastDim})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(out);
}

}  // namespace

template <int kSparseTopK>
void get_block_table_v1(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView topk_idx,
    tvm::ffi::TensorView block_table,
    tvm::ffi::TensorView token_to_bs,
    tvm::ffi::TensorView token_pos_in_bs,
    tvm::ffi::TensorView seqlen_q) {
  using namespace host;
  SymbolicSize token_num{"token_num"}, batch_size{"batch_size"}, seqlen_q_max{"seqlen_q_max"};
  SymbolicDevice device;
  device.set_options<kDLCUDA>();
  verify_inputs<kSparseTopK>(
      out, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, token_num, batch_size, seqlen_q_max, device);

  const int n_token = static_cast<int>(token_num.unwrap());
  const int s_q_max = static_cast<int>(seqlen_q_max.unwrap());
  const DLDevice dev = device.unwrap();

  constexpr int kThreadsPerBlock = 256;
  const int64_t num_blocks = (static_cast<int64_t>(n_token) + kThreadsPerBlock - 1) / kThreadsPerBlock;

  LaunchKernel(num_blocks, kThreadsPerBlock, dev)(
      get_block_table_cuda_v1<kSparseTopK>,
      static_cast<const int*>(topk_idx.data_ptr()),
      static_cast<const int*>(block_table.data_ptr()),
      static_cast<const int*>(token_to_bs.data_ptr()),
      static_cast<const int*>(token_pos_in_bs.data_ptr()),
      static_cast<const int*>(seqlen_q.data_ptr()),
      static_cast<int*>(out.data_ptr()),
      s_q_max,
      n_token);
}

template <int kSparseTopK>
void get_block_table_v2(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView topk_idx,
    tvm::ffi::TensorView block_table,
    tvm::ffi::TensorView token_to_bs,
    tvm::ffi::TensorView token_pos_in_bs,
    tvm::ffi::TensorView seqlen_q) {
  using namespace host;
  SymbolicSize token_num{"token_num"}, batch_size{"batch_size"}, seqlen_q_max{"seqlen_q_max"};
  SymbolicDevice device;
  device.set_options<kDLCUDA>();
  verify_inputs<kSparseTopK>(
      out, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, token_num, batch_size, seqlen_q_max, device);

  const int n_token = static_cast<int>(token_num.unwrap());
  const int s_q_max = static_cast<int>(seqlen_q_max.unwrap());
  const DLDevice dev = device.unwrap();

  constexpr int kThreadsPerBlock = 1024;
  const int64_t total = static_cast<int64_t>(n_token) * kHeadGroup * kSparseTopK;
  const int64_t num_blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;

  LaunchKernel(num_blocks, kThreadsPerBlock, dev)(
      get_block_table_cuda_v2<kSparseTopK>,
      static_cast<const int*>(topk_idx.data_ptr()),
      static_cast<const int*>(block_table.data_ptr()),
      static_cast<const int*>(token_to_bs.data_ptr()),
      static_cast<const int*>(token_pos_in_bs.data_ptr()),
      static_cast<const int*>(seqlen_q.data_ptr()),
      static_cast<int*>(out.data_ptr()),
      s_q_max,
      n_token);
}

template <int kSparseTopK>
void get_block_table_v3(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView topk_idx,
    tvm::ffi::TensorView block_table,
    tvm::ffi::TensorView token_to_bs,
    tvm::ffi::TensorView token_pos_in_bs,
    tvm::ffi::TensorView seqlen_q) {
  using namespace host;
  SymbolicSize token_num{"token_num"}, batch_size{"batch_size"}, seqlen_q_max{"seqlen_q_max"};
  SymbolicDevice device;
  device.set_options<kDLCUDA>();
  verify_inputs<kSparseTopK>(
      out, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, token_num, batch_size, seqlen_q_max, device);

  const int n_token = static_cast<int>(token_num.unwrap());
  const int s_q_max = static_cast<int>(seqlen_q_max.unwrap());
  const DLDevice dev = device.unwrap();

  constexpr int kThreadsPerBlock = 1024;
  const int64_t total = static_cast<int64_t>(n_token) * kHeadGroup * kSparseTopK * kSparseBlockSize;
  const int64_t num_blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;

  LaunchKernel(num_blocks, kThreadsPerBlock, dev)(
      get_block_table_cuda_v3<kSparseTopK>,
      static_cast<const int*>(topk_idx.data_ptr()),
      static_cast<const int*>(block_table.data_ptr()),
      static_cast<const int*>(token_to_bs.data_ptr()),
      static_cast<const int*>(token_pos_in_bs.data_ptr()),
      static_cast<const int*>(seqlen_q.data_ptr()),
      static_cast<int*>(out.data_ptr()),
      s_q_max,
      n_token);
}

}  // namespace minicpm_sala
