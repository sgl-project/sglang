// MiniCPM-SALA sparse attention: build the per-token sparse block table.
//
// Migrated from `3rdparty/sparse_kernel/get_table_kernel.cu`. The original
// CUDA kernels are kept almost verbatim; only the host-side wrappers are
// rewritten from the torch::Tensor + pybind interface to the jit_kernel
// tvm::ffi::TensorView + TensorMatcher/LaunchKernel convention.
//
// The Python wrapper compiles and caches one module per sparse layout.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice

#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::minicpm_sala {

constexpr int kTopkPerBlock = 16;

// Vector width used by the blockwise kernel to write one out_block_table row.
// Rows are int32 and start at a multiple of kSparseBlockSize elements, so a
// vectorized store is naturally aligned whenever the row length is a multiple
// of 4. AlignedVector rejects widths above the architecture limit, so this
// stays portable (16 B on pre-Blackwell, 32 B on Blackwell and later).
template <int kSparseBlockSize>
constexpr int kBlockwiseVecWidth = (kSparseBlockSize % 4 == 0) ? 4 : 1;

// Threads cooperating on one out_block_table row in the blockwise kernel.
template <int kSparseBlockSize>
constexpr int kBlockwiseThreadsPerRow = kSparseBlockSize / kBlockwiseVecWidth<kSparseBlockSize>;

// topk_idx:        [head_group, token_num, kSparseTopK]  int32
// block_table:     [batch_size, seqlen_q_max]           int32
// token_to_bs:     [token_num]                           int32
// token_pos_in_bs: [token_num]                           int32
// seqlen_q:        [batch_size]                          int32
// out_block_table: [token_num, head_group, kSparseTopK * kSparseBlockSize] int32

// kBlockwiseThreadsPerRow threads cooperate on one out_block_table row: each
// thread computes kBlockwiseVecWidth consecutive elements and writes them with a
// single vectorized store. Threads of a row are adjacent, so a warp covers whole
// rows and its stores stay contiguous. The former one-thread-per-row layout
// strided the stores by kSparseBlockSize and needed ~8x the memory transactions.
// topk_idx is still read once per row: every thread of a row reads the same
// address, which the hardware broadcasts.
template <int kSparseTopK, int kHeadGroup, int kSparseBlockSize>
__global__ void get_block_table_cuda_blockwise(
    const int* topk_idx,
    const int* block_table,
    const int* token_to_bs,
    const int* token_pos_in_bs,
    const int* seqlen_q,
    int* out_block_table,
    const int seqlen_q_max,
    const int token_num) {
  constexpr int kVec = kBlockwiseVecWidth<kSparseBlockSize>;
  constexpr int kThreadsPerRow = kBlockwiseThreadsPerRow<kSparseBlockSize>;

  // The grid is kThreadsPerRow times larger than one-thread-per-row, so the flat
  // thread id no longer fits an int for long sequences.
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row = gid / kThreadsPerRow;
  if (row >= static_cast<int64_t>(token_num) * kSparseTopK * kHeadGroup) return;
  const int lane = static_cast<int>(gid % kThreadsPerRow);

  const int token_idx = static_cast<int>(row / (kSparseTopK * kHeadGroup));
  const int head_group_idx = static_cast<int>((row / kSparseTopK) % kHeadGroup);
  const int topk_idx_in_head = static_cast<int>(row % kSparseTopK);
  const int bs = token_to_bs[token_idx];
  const int pos_in_bs = token_pos_in_bs[token_idx];
  const int seqlen_q_bs = seqlen_q[bs];
  const int sparse_block_idx =
      topk_idx[head_group_idx * token_num * kSparseTopK + token_idx * kSparseTopK + topk_idx_in_head];

  auto out_view = reinterpret_cast<int (*)[kHeadGroup][kSparseTopK][kSparseBlockSize]>(out_block_table);
  int* out_row = &out_view[token_idx][head_group_idx][topk_idx_in_head][0];

  device::AlignedVector<int, kVec> out_vec;
#pragma unroll
  for (int j = 0; j < kVec; j++) {
    const int i = lane * kVec + j;
    const int token_idx_in_batch = sparse_block_idx * kSparseBlockSize + i;
    if (sparse_block_idx >= 0 && token_idx_in_batch < seqlen_q_bs && token_idx_in_batch < pos_in_bs) {
      out_vec[j] = kHeadGroup * block_table[bs * seqlen_q_max + token_idx_in_batch] + head_group_idx;
    } else {
      out_vec[j] = 0;
    }
  }
  out_vec.store(out_row + lane * kVec);
}

// 1 thread calculates 1 element of out_block_table. A 1024-thread block
// expands 16 selected blocks in parallel when kSparseBlockSize is 64.
template <int kSparseTopK, int kHeadGroup, int kSparseBlockSize>
__global__ void get_block_table_cuda_elementwise(
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
  if (sparse_block_idx < 0) {
    out_block_table[token_idx_dst] = 0;
    return;
  }

  const int bs = token_to_bs[token_idx];
  const int pos_in_bs = token_pos_in_bs[token_idx];
  const int seqlen_q_bs = seqlen_q[bs];

  if (token_idx_src < seqlen_q_bs && token_idx_src < pos_in_bs) {
    out_block_table[token_idx_dst] = kHeadGroup * block_table[bs * seqlen_q_max + token_idx_src] + head_group_idx;
  } else {
    out_block_table[token_idx_dst] = 0;
  }
}

// Validate all inputs that are shared across the two kernel variants and
// bind the symbolic dims (token_num / batch_size / seqlen_q_max). The output
// tensor is pre-allocated and fully initialized by the selected kernel.
template <int kSparseTopK, int kHeadGroup, int kSparseBlockSize>
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

template <bool kElementwise, int kSparseTopK, int kHeadGroup, int kSparseBlockSize>
void get_block_table(
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
  verify_inputs<kSparseTopK, kHeadGroup, kSparseBlockSize>(
      out, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q, token_num, batch_size, seqlen_q_max, device);

  const int n_token = static_cast<int>(token_num.unwrap());
  const int s_q_max = static_cast<int>(seqlen_q_max.unwrap());
  const DLDevice dev = device.unwrap();

  constexpr int kThreadsPerBlock = 1024;
  // The blockwise kernel spreads one (token, head_group, topk) entry over
  // kBlockwiseThreadsPerRow cooperating threads; the elementwise kernel uses one
  // thread per output element.
  constexpr int kThreadsPerEntry = kElementwise ? kSparseBlockSize : kBlockwiseThreadsPerRow<kSparseBlockSize>;
  const int64_t total = static_cast<int64_t>(n_token) * kHeadGroup * kSparseTopK * kThreadsPerEntry;
  const int64_t num_blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;

  if constexpr (!kElementwise) {
    LaunchKernel(num_blocks, kThreadsPerBlock, dev)(
        get_block_table_cuda_blockwise<kSparseTopK, kHeadGroup, kSparseBlockSize>,
        static_cast<const int*>(topk_idx.data_ptr()),
        static_cast<const int*>(block_table.data_ptr()),
        static_cast<const int*>(token_to_bs.data_ptr()),
        static_cast<const int*>(token_pos_in_bs.data_ptr()),
        static_cast<const int*>(seqlen_q.data_ptr()),
        static_cast<int*>(out.data_ptr()),
        s_q_max,
        n_token);
  } else {
    LaunchKernel(num_blocks, kThreadsPerBlock, dev)(
        get_block_table_cuda_elementwise<kSparseTopK, kHeadGroup, kSparseBlockSize>,
        static_cast<const int*>(topk_idx.data_ptr()),
        static_cast<const int*>(block_table.data_ptr()),
        static_cast<const int*>(token_to_bs.data_ptr()),
        static_cast<const int*>(token_pos_in_bs.data_ptr()),
        static_cast<const int*>(seqlen_q.data_ptr()),
        static_cast<int*>(out.data_ptr()),
        s_q_max,
        n_token);
  }
}

}  // namespace sglang::minicpm_sala
