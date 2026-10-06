#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

// Tile-scheduler metadata for FlashMLA's split-KV decode.
//
// FlashMLA computes this itself when handed no metadata, in a <<<1, 32>>> kernel
// whose whole partition walk runs on thread 0 and stores each 32-byte entry
// straight to global memory.  Inside a cuda graph that sits fully exposed on the
// critical path, and it cannot be hoisted out because the schedule depends on
// the per-request `topk_length` of the step being replayed.
//
// This kernel produces the same output bit for bit; the shared-memory layout
// matches FlashMLA's exactly because the idle-tail fill below reads one entry
// past `first_block_idx_shared`, as FlashMLA does.

namespace sglang {

namespace flashmla {

// sizeof(DecodingSchedMeta)/4: begin/end req, begin/end block, begin split,
// the two per-end split flags, one pad word.
constexpr int kMetaInts = 8;
constexpr int kBlockSize = 256;

struct Params {
  int b;
  int block_size_n;
  int fixed_overhead_num_blocks;
  int topk;        // -1 for a dense model
  int extra_topk;  // 0 when there is no extra cache
  const int* __restrict__ topk_length;
  const int* __restrict__ extra_topk_length;
  const int* __restrict__ seqlens_k;  // dense only
  int* __restrict__ tile_scheduler_metadata;
  int* __restrict__ num_splits;
  int num_sm_parts;
};

__device__ __forceinline__ int ceil_div_i(int a, int b) {
  return (a + b - 1) / b;
}

// ku::ceil: round up to a multiple of b.
__device__ __forceinline__ int ceil_to_i(int a, int b) {
  return (a + b - 1) / b * b;
}

__global__ void __launch_bounds__(kBlockSize) flashmla_sched_meta_kernel(__grid_constant__ const Params p) {
  extern __shared__ int smem[];
  const int b = p.b;
  int* num_blocks_shared = smem;                   // [b]
  int* num_splits_shared = smem + b;               // [b + 1]
  int* seqlens_k_shared = smem + b * 2 + 1;        // [b]
  int* first_block_idx_shared = smem + b * 3 + 1;  // [b]
  int* last_block_idx_shared = smem + b * 4 + 1;   // [b]
  int* out_shared = smem + b * 5 + 1;              // [num_sm_parts * kMetaInts]

  __shared__ int total_num_blocks_shared;

  int partial = 0;
  for (int i = threadIdx.x; i < b; i += kBlockSize) {
    int cur_s_k;
    if (p.topk == -1) {
      cur_s_k = __ldg(p.seqlens_k + i);
    } else {
      cur_s_k = p.topk_length ? __ldg(p.topk_length + i) : p.topk;
      if (cur_s_k == 0) cur_s_k = 1;  // the main loop must never be empty
      if (p.extra_topk) {
        cur_s_k = ceil_to_i(cur_s_k, p.block_size_n);
        cur_s_k += p.extra_topk_length ? __ldg(p.extra_topk_length + i) : p.extra_topk;
      }
    }
    seqlens_k_shared[i] = cur_s_k;
    const int last_token_idx = max(cur_s_k - 1, 0);
    const int cur_first_block_idx = 0;  // first_token_idx is always 0
    const int cur_last_block_idx = last_token_idx / p.block_size_n;
    const int num_blocks = cur_last_block_idx - cur_first_block_idx + 1;
    partial += num_blocks + p.fixed_overhead_num_blocks;
    num_blocks_shared[i] = num_blocks;
    first_block_idx_shared[i] = cur_first_block_idx;
    last_block_idx_shared[i] = cur_last_block_idx;
  }

  // Integer sum, so the tree order does not change the result.
  for (int offset = 16; offset >= 1; offset /= 2) {
    partial += __shfl_xor_sync(uint32_t(-1), partial, offset);
  }
  __shared__ int warp_sums[kBlockSize / 32];
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = partial;
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
#pragma unroll
    for (int w = 0; w < kBlockSize / 32; ++w)
      total += warp_sums[w];
    total_num_blocks_shared = total;
  }
  __syncthreads();

  const int fixed_overhead_num_blocks = p.fixed_overhead_num_blocks;
  __shared__ int first_idle_part_shared;

  if (threadIdx.x == 0) {
    const int payload = ceil_div_i(total_num_blocks_shared, p.num_sm_parts) + fixed_overhead_num_blocks;

    int now_req_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
    // The request being consumed, and the one before it, held across partitions.
    int cur_c = num_blocks_shared[0], cur_lb = last_block_idx_shared[0], cur_sk = seqlens_k_shared[0];
    int prev_lb = 0, prev_sk = 0;
    num_splits_shared[0] = 0;
    int i = 0;
    for (; i < p.num_sm_parts; ++i) {
      int* meta = out_shared + i * kMetaInts;
      const int begin_req_idx = now_req_idx;
      // first_block_idx is 0 for every request: the first token index is 0.
      const int begin_block_idx = now_block;
      const int begin_split_idx = now_n_split_idx;
      int is_first_req_splitted = (now_block != 0);
      int remain_payload = payload;
      while (now_req_idx < b) {
        const int now_remain_blocks = cur_c - now_block;
        if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
          cum_num_splits += now_n_split_idx + 1;
          num_splits_shared[now_req_idx + 1] = cum_num_splits;
          remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
          ++now_req_idx;
          now_block = 0;
          now_n_split_idx = 0;
          prev_lb = cur_lb;
          prev_sk = cur_sk;
          if (now_req_idx < b) {
            cur_c = num_blocks_shared[now_req_idx];
            cur_lb = last_block_idx_shared[now_req_idx];
            cur_sk = seqlens_k_shared[now_req_idx];
          }
        } else {
          if (remain_payload - fixed_overhead_num_blocks > 0) {
            now_block += remain_payload - fixed_overhead_num_blocks;
            ++now_n_split_idx;
            remain_payload = 0;
          }
          break;
        }
      }
      const int split_open = now_block > 0;
      const int end_req_idx = split_open ? now_req_idx : now_req_idx - 1;
      const int end_lb = split_open ? cur_lb : prev_lb;
      const int end_sk = split_open ? cur_sk : prev_sk;
      const int end_block_idx = split_open ? now_block : (end_sk == 0 ? 0 : end_lb + 1);
      int is_last_req_splitted = (end_block_idx != end_lb + 1) && (end_sk != 0);
      if (begin_req_idx == end_req_idx) {
        is_first_req_splitted = is_last_req_splitted = is_first_req_splitted || is_last_req_splitted;
      }
      meta[0] = begin_req_idx;
      meta[1] = end_req_idx;
      meta[2] = begin_block_idx;
      meta[3] = end_block_idx;
      meta[4] = begin_split_idx;
      meta[5] = is_first_req_splitted;
      meta[6] = is_last_req_splitted;
      meta[7] = 0;
      if (now_req_idx == b) {
        ++i;
        break;
      }
    }
    first_idle_part_shared = i;
  }
  __syncthreads();

  // Every partition past the walk describes the same empty range.
  {
    const int lb_last = last_block_idx_shared[b - 1];
    const int sk_last = seqlens_k_shared[b - 1];
    const int end_block_idx = (sk_last == 0) ? 0 : lb_last + 1;
    // FlashMLA reads first_block_idx_shared[batch_size] for these, one past the
    // end of that array, which aliases last_block_idx_shared[0].
    const int begin_block_idx = last_block_idx_shared[0];
    const int is_last_req_splitted = (end_block_idx != lb_last + 1) && (sk_last != 0);
    for (int i = first_idle_part_shared + threadIdx.x; i < p.num_sm_parts; i += kBlockSize) {
      int* meta = out_shared + i * kMetaInts;
      meta[0] = b;
      meta[1] = b - 1;
      meta[2] = begin_block_idx;
      meta[3] = end_block_idx;
      meta[4] = 0;
      meta[5] = 0;
      meta[6] = is_last_req_splitted;
      meta[7] = 0;
    }
  }
  __syncthreads();

  const int meta_words = p.num_sm_parts * kMetaInts;
  for (int i = threadIdx.x; i < meta_words; i += kBlockSize) {
    p.tile_scheduler_metadata[i] = out_shared[i];
  }
  for (int i = threadIdx.x; i <= b; i += kBlockSize) {
    p.num_splits[i] = num_splits_shared[i];
  }
}

}  // namespace flashmla

void flashmla_sched_meta(
    tvm::ffi::TensorView tile_scheduler_metadata,
    tvm::ffi::TensorView num_splits,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_length,
    tvm::ffi::Optional<tvm::ffi::TensorView> extra_topk_length,
    tvm::ffi::Optional<tvm::ffi::TensorView> seqlens_k,
    int64_t block_size_n,
    int64_t fixed_overhead_num_blocks,
    int64_t topk,
    int64_t extra_topk) {
  using namespace host;
  using namespace flashmla;

  auto parts = SymbolicSize{"num_sm_parts"};
  auto meta_ints = SymbolicSize{"meta_ints"};
  auto b_plus_one = SymbolicSize{"batch_size_plus_one"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLGPU>();

  TensorMatcher({parts, meta_ints}).with_dtype<int32_t>().with_device(device_).verify(tile_scheduler_metadata);
  TensorMatcher({b_plus_one}).with_strides({1}).with_dtype<int32_t>().with_device(device_).verify(num_splits);

  const int num_sm_parts = static_cast<int>(parts.unwrap());
  const int b = static_cast<int>(b_plus_one.unwrap()) - 1;
  RuntimeCheck(
      static_cast<int>(meta_ints.unwrap()) == kMetaInts,
      "tile_scheduler_metadata must be [num_sm_parts, ",
      kMetaInts,
      "], got last dim ",
      meta_ints.unwrap());
  RuntimeCheck(b >= 1, "batch size must be positive, got ", b);
  RuntimeCheck(num_sm_parts >= 1, "num_sm_parts must be positive, got ", num_sm_parts);
  RuntimeCheck(block_size_n >= 1, "block_size_n must be positive, got ", block_size_n);

  auto opt_ptr = [&](const tvm::ffi::Optional<tvm::ffi::TensorView>& t, const char* name) -> const int* {
    if (!t.has_value()) return nullptr;
    auto n = SymbolicSize{"batch_size"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLGPU>();
    TensorMatcher({n}).with_strides({1}).with_dtype<int32_t>().with_device(dev).verify(t.value());
    RuntimeCheck(static_cast<int>(n.unwrap()) == b, name, " must have ", b, " entries, got ", n.unwrap());
    return static_cast<const int*>(t.value().data_ptr());
  };

  RuntimeCheck(topk != -1 || seqlens_k.has_value(), "a dense schedule (topk == -1) needs seqlens_k");

  const Params p{
      b,
      static_cast<int>(block_size_n),
      static_cast<int>(fixed_overhead_num_blocks),
      static_cast<int>(topk),
      static_cast<int>(extra_topk),
      opt_ptr(topk_length, "topk_length"),
      opt_ptr(extra_topk_length, "extra_topk_length"),
      opt_ptr(seqlens_k, "seqlens_k"),
      static_cast<int*>(tile_scheduler_metadata.data_ptr()),
      static_cast<int*>(num_splits.data_ptr()),
      num_sm_parts,
  };

  const std::size_t smem = sizeof(int) * static_cast<std::size_t>(b * 5 + 1 + num_sm_parts * kMetaInts);
  RuntimeCheck(smem <= 48 * 1024, "schedule does not fit in shared memory: ", smem, " bytes");
  LaunchKernel(1, kBlockSize, device_.unwrap(), smem)(flashmla_sched_meta_kernel, p);
}

}  // namespace sglang
