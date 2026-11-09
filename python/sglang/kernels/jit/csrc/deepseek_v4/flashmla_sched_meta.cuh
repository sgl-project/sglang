#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

// Tile-scheduler metadata for FlashMLA's split-KV decode.
//
// FlashMLA computes this itself when handed no metadata, in a <<<1, 32>>> kernel
// whose whole partition walk runs on thread 0 and stores each 32-byte entry
// straight to global memory.  Inside a cuda graph that sits fully exposed on the
// critical path, and it cannot be hoisted out because the schedule depends on
// the per-request `topk_length` of the step being replayed.
//
// This kernel produces the same output bit for bit as the legacy scheduler in
// sgl-flash-mla 1.0.0, idle-tail partitions included.

namespace sglang {

namespace flashmla {

// sizeof(DecodingSchedMeta)/4: begin/end req, begin/end block, begin split,
// the two per-end split flags, one pad word.
constexpr int kMetaInts = 8;
constexpr int kBlockSize = 256;
// The dynamic shared memory every schedule kernel opts into.
constexpr int kMaxDynamicSmem = 48 * 1024;
// Smallest batch that takes the parallel walk; below it the serial walk is the shorter one.
constexpr int kMinParallelBatch = 8;
// The parallel walk's cap: leaves 1 KiB for its static scan scratch and allocation alignment.
constexpr int kParallelSmemLimit = kMaxDynamicSmem - 1024;

// Schedules, as the host op takes them.
enum Schedule : int { kSerial = 0, kParallel = 1, kWindowed = 2 };

// Dynamic shared memory of each schedule in ints, in the order its kernel carves it.
__host__ __device__ constexpr int serial_smem_ints(int b, int parts) {
  return b * 5 + 1 + parts * kMetaInts;  // num_blocks, num_splits, seqlens_k, first/last block, out
}
__host__ __device__ constexpr int parallel_walk_smem_ints(int b, int parts, int max_states) {
  return (b + 1) + 2 * (max_states + 1) + (parts + 1) + b;  // row_start, jump x2, part_start, split_count
}
__host__ __device__ constexpr int windowed_smem_ints(int b, int parts) {
  return 2 * (b + 1) + 4 * (parts + 1);  // cost_prefix, split_prefix, cut_req x2, cut_block x2
}

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
  // Capacity of the parallel walk's shared-memory tables. A larger actual
  // state count (or zero capacity) uses the serial walk.
  int max_states;
};

__device__ __forceinline__ int ceil_div_i(int a, int b) {
  return (a + b - 1) / b;
}

// ku::ceil: round up to a multiple of b.
__device__ __forceinline__ int ceil_to_i(int a, int b) {
  return (a + b - 1) / b * b;
}

// Parallel walk. State row_start_shared[r] + blk is block `blk` of request `r`
// (row_start_shared[b]: all consumed). next_partition_state is the serial walk's
// greedy step, so pointer doubling finds every partition start at once.
__device__ __forceinline__ int last_le_index(const int* prefix, int b, int x) {
  int lo = 0, hi = b - 1;  // the last r in [0, b) with prefix[r] <= x
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (prefix[mid] <= x) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

// Exclusive prefix sum of f(0..n-1) into out[0..n], every thread of the block
// taking part; scratch holds kBlockSize / 32 ints.
template <typename F>
__device__ __forceinline__ void block_exclusive_scan(int n, F f, int* out, int* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int carry = 0;
  for (int base = 0; base < n; base += kBlockSize) {
    const int i = base + threadIdx.x;
    const int v = i < n ? f(i) : 0;
    int incl = v;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int up = __shfl_up_sync(uint32_t(-1), incl, offset);
      if (lane >= offset) incl += up;
    }
    if (lane == 31) scratch[warp] = incl;
    __syncthreads();
    int warp_base = 0;
    for (int w = 0; w < warp; ++w)
      warp_base += scratch[w];
    if (i < n) out[i] = carry + warp_base + incl - v;
    int chunk = 0;
    for (int w = 0; w < kBlockSize / 32; ++w)
      chunk += scratch[w];
    carry += chunk;
    __syncthreads();
  }
  if (threadIdx.x == 0) out[n] = carry;
}

__device__ __forceinline__ int next_partition_state(
    const int* row_start_shared,
    const int* num_blocks_shared,
    int b,
    int num_states,
    int payload,
    int overhead,
    int x) {
  if (x == num_states) return x;
  int r = last_le_index(row_start_shared, b, x);
  int blk = x - row_start_shared[r];
  int remain = payload;
  while (r < b) {
    const int rem = num_blocks_shared[r] - blk;
    if (remain >= rem + overhead) {
      remain -= rem + overhead;
      ++r;
      blk = 0;
    } else {
      if (remain - overhead > 0) blk += remain - overhead;
      break;
    }
  }
  return r == b ? num_states : row_start_shared[r] + blk;
}

// The serial walk's output (entries, num_splits_shared, first idle partition),
// computed by the whole block. Its tables follow out_shared in shared memory.
__device__ __forceinline__ void parallel_walk(
    const Params& p,
    const int total_num_blocks,
    const int* num_blocks_shared,
    const int* seqlens_k_shared,
    const int* last_block_idx_shared,
    int* num_splits_shared,
    int* out_shared,
    int* first_idle_part_shared) {
  const int b = p.b;
  const int parts = p.num_sm_parts;
  const int overhead = p.fixed_overhead_num_blocks;
  const int payload = ceil_div_i(total_num_blocks, parts) + overhead;
  int* row_start_shared = out_shared + parts * kMetaInts;        // [b + 1]
  int* jump_shared = row_start_shared + b + 1;                   // [max_states + 1]
  int* jump_next_shared = jump_shared + p.max_states + 1;        // [max_states + 1]
  int* part_start_shared = jump_next_shared + p.max_states + 1;  // [parts + 1]
  int* split_count_shared = part_start_shared + parts + 1;       // [b]
  __shared__ int scan_scratch[kBlockSize / 32];
  block_exclusive_scan(b, [&](int r) { return num_blocks_shared[r]; }, row_start_shared, scan_scratch);
  if (threadIdx.x == 0) {
    part_start_shared[0] = 0;
    *first_idle_part_shared = parts;
  }
  __syncthreads();
  const int num_states = row_start_shared[b];
  for (int x = threadIdx.x; x <= num_states; x += kBlockSize) {
    jump_shared[x] = next_partition_state(row_start_shared, num_blocks_shared, b, num_states, payload, overhead, x);
  }
  for (int r = threadIdx.x; r < b; r += kBlockSize) {
    split_count_shared[r] = 0;
  }
  __syncthreads();
  // jump = next^len: part_start[len, 2 len) from part_start[0, len), then square jump.
  for (int len = 1; len <= parts; len <<= 1) {
    for (int i = len + threadIdx.x; i < min(2 * len, parts + 1); i += kBlockSize) {
      part_start_shared[i] = jump_shared[part_start_shared[i - len]];
    }
    __syncthreads();
    if (2 * len <= parts) {
      for (int x = threadIdx.x; x <= num_states; x += kBlockSize) {
        jump_next_shared[x] = jump_shared[jump_shared[x]];
      }
      __syncthreads();
      int* tmp = jump_shared;
      jump_shared = jump_next_shared;
      jump_next_shared = tmp;
    }
  }
  // The serial walk stops after the partition that consumes the last request.
  for (int i = 1 + threadIdx.x; i <= parts; i += kBlockSize) {
    if (part_start_shared[i] == num_states) atomicMin(first_idle_part_shared, i);
  }
  __syncthreads();
  const int active = *first_idle_part_shared;
  // A request is split once per partition that starts inside it.
  for (int i = 1 + threadIdx.x; i < active; i += kBlockSize) {
    const int r = last_le_index(row_start_shared, b, part_start_shared[i]);
    if (part_start_shared[i] > row_start_shared[r]) atomicAdd(&split_count_shared[r], 1);
  }
  __syncthreads();
  // num_splits[r] = sum over earlier requests of their split count + 1
  block_exclusive_scan(b, [&](int r) { return split_count_shared[r] + 1; }, num_splits_shared, scan_scratch);
  for (int i = threadIdx.x; i < active; i += kBlockSize) {
    const int x0 = part_start_shared[i];
    const int r0 = last_le_index(row_start_shared, b, x0);
    const int blk0 = x0 - row_start_shared[r0];
    const int x1 = part_start_shared[i + 1];
    const int r1 = x1 == num_states ? b : last_le_index(row_start_shared, b, x1);
    const int blk1 = x1 == num_states ? 0 : x1 - row_start_shared[r1];
    int begin_split_idx = 0;
    if (blk0 > 0) {  // partitions that started inside r0 so far, this one included
      for (int j = i;
           j >= 0 && part_start_shared[j] > row_start_shared[r0] && part_start_shared[j] < row_start_shared[r0 + 1];
           --j) {
        ++begin_split_idx;
      }
    }
    const int split_open = blk1 > 0;
    const int end_req_idx = split_open ? r1 : r1 - 1;
    const int end_lb = last_block_idx_shared[end_req_idx];
    const int end_sk = seqlens_k_shared[end_req_idx];
    const int end_block_idx = split_open ? blk1 : (end_sk == 0 ? 0 : end_lb + 1);
    int is_first_req_splitted = blk0 != 0;
    int is_last_req_splitted = (end_block_idx != end_lb + 1) && (end_sk != 0);
    if (r0 == end_req_idx) {
      is_first_req_splitted = is_last_req_splitted = is_first_req_splitted || is_last_req_splitted;
    }
    int* meta = out_shared + i * kMetaInts;
    meta[0] = r0;
    meta[1] = end_req_idx;
    meta[2] = blk0;
    meta[3] = end_block_idx;
    meta[4] = begin_split_idx;
    meta[5] = is_first_req_splitted;
    meta[6] = is_last_req_splitted;
    meta[7] = 0;
  }
}

// Windowed schedule (sparse only): a valid FlashMLA schedule, not FlashMLA's
// own, with the same bound on the costliest partition. Request costs are laid
// end to end and cut into num_sm_parts equal windows; a cut within a request's
// first `overhead` positions moves to its start. A second cutting also snaps a
// cut leaving <= `overhead` blocks to the request's end; the cheaper one is kept.
__device__ __forceinline__ int sparse_seqlen_k(const Params& p, int i) {
  int cur_s_k = p.topk_length ? __ldg(p.topk_length + i) : p.topk;
  if (cur_s_k == 0) cur_s_k = 1;  // the main loop must never be empty
  if (p.extra_topk) {
    cur_s_k = ceil_to_i(cur_s_k, p.block_size_n);
    cur_s_k += p.extra_topk_length ? __ldg(p.extra_topk_length + i) : p.extra_topk;
  }
  return cur_s_k;
}

__device__ __forceinline__ int sparse_num_blocks(const Params& p, int i) {
  return max(sparse_seqlen_k(p, i) - 1, 0) / p.block_size_n + 1;
}

__global__ void __launch_bounds__(kBlockSize) flashmla_windowed_sched_meta_kernel(__grid_constant__ const Params p) {
  extern __shared__ int smem[];
  const int b = p.b;
  const int parts = p.num_sm_parts;
  const int overhead = p.fixed_overhead_num_blocks;
  int* cost_prefix = smem;                     // [b + 1]: the cost of requests [0, r)
  int* split_prefix = cost_prefix + b + 1;     // [b + 1]: the splits of requests [0, r)
  int* cut_req = split_prefix + b + 1;         // [2, parts + 1]: plain, snapping
  int* cut_block = cut_req + 2 * (parts + 1);  // [2, parts + 1]
  __shared__ int scan_scratch[kBlockSize / 32];
  __shared__ int max_cost[2];

  block_exclusive_scan(b, [&](int r) { return sparse_num_blocks(p, r) + overhead; }, cost_prefix, scan_scratch);
  if (threadIdx.x < 2) max_cost[threadIdx.x] = 0;
  __syncthreads();
  const int total_cost = cost_prefix[b];
  const int window = ceil_div_i(total_cost, parts);

  for (int c = threadIdx.x; c < 2 * (parts + 1); c += kBlockSize) {
    const bool snap = c > parts;
    const int pos = (snap ? c - parts - 1 : c) * window;
    int req = b, block = 0;
    if (pos < total_cost) {
      req = last_le_index(cost_prefix, b, pos);
      const int offset = pos - cost_prefix[req];
      if (snap && cost_prefix[req + 1] - pos <= overhead) {
        ++req;
      } else if (offset > overhead) {
        block = offset - overhead;
      }
    }
    cut_req[c] = req;
    cut_block[c] = block;
  }
  __syncthreads();
  // A partition costs its positions, plus an overhead when it starts inside a request.
  auto position = [&](int req, int block) { return cost_prefix[req] + (block > 0 ? overhead + block : 0); };
  for (int c = threadIdx.x; c < 2 * (parts + 1); c += kBlockSize) {
    if (c == parts || c == 2 * parts + 1) continue;
    const int cost = position(cut_req[c + 1], cut_block[c + 1]) - position(cut_req[c], cut_block[c]) +
                     (cut_block[c] > 0 ? overhead : 0);
    atomicMax(&max_cost[c > parts], cost);
  }
  __syncthreads();
  const bool snap = max_cost[1] <= max_cost[0];
  if (snap) {
    cut_req += parts + 1;
    cut_block += parts + 1;
  }
  // A request splits at every multiple of the window in (its start + overhead,
  // its end), or (its start + overhead, its end - overhead) when snapping.
  block_exclusive_scan(
      b,
      [&](int r) {
        const int last = cost_prefix[r + 1] - 1 - (snap ? overhead : 0);
        return 1 + max(last / window - (cost_prefix[r] + overhead) / window, 0);
      },
      split_prefix,
      scan_scratch);
  __syncthreads();

  for (int i = threadIdx.x; i < parts; i += kBlockSize) {
    const int begin_req = cut_req[i], begin_block = cut_block[i];
    const int end_req = cut_req[i + 1], end_block = cut_block[i + 1];
    int meta[kMetaInts] = {};
    if (begin_req == end_req && begin_block == end_block) {
      // An empty partition: the kernel skips a partition beginning past the batch.
      meta[0] = b;
      meta[1] = b - 1;
    } else {
      // A partition ending at a request's start ends with the whole request before it.
      const int last_req = end_block > 0 ? end_req : end_req - 1;
      meta[0] = begin_req;
      meta[1] = last_req;
      meta[2] = begin_block;
      meta[3] = end_block > 0 ? end_block : cost_prefix[end_req] - cost_prefix[end_req - 1] - overhead;
      // The cuts inside begin_req up to this one: multiples of the window in (its start + overhead, i * window].
      meta[4] = begin_block > 0 ? i - (cost_prefix[begin_req] + overhead) / window : 0;
      meta[5] = split_prefix[begin_req + 1] - split_prefix[begin_req] > 1;
      meta[6] = split_prefix[last_req + 1] - split_prefix[last_req] > 1;
    }
    int4* out = reinterpret_cast<int4*>(p.tile_scheduler_metadata + i * kMetaInts);
    out[0] = make_int4(meta[0], meta[1], meta[2], meta[3]);
    out[1] = make_int4(meta[4], meta[5], meta[6], meta[7]);
  }
  for (int i = threadIdx.x; i <= b; i += kBlockSize) {
    p.num_splits[i] = split_prefix[i];
  }
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
    const int cur_s_k = p.topk == -1 ? __ldg(p.seqlens_k + i) : sparse_seqlen_k(p, i);
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

  const int num_states = total_num_blocks_shared - b * fixed_overhead_num_blocks;
  if (p.max_states > 0 && num_states <= p.max_states) {
    parallel_walk(
        p,
        total_num_blocks_shared,
        num_blocks_shared,
        seqlens_k_shared,
        last_block_idx_shared,
        num_splits_shared,
        out_shared,
        &first_idle_part_shared);
  } else if (threadIdx.x == 0) {
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
    // `now_block` is 0 once every request is consumed.
    const int begin_block_idx = 0;
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
    int64_t extra_topk,
    int64_t schedule) {
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
  RuntimeCheck(schedule == kSerial || schedule == kParallel || schedule == kWindowed, "invalid schedule ", schedule);

  Params p{
      .b = b,
      .block_size_n = static_cast<int>(block_size_n),
      .fixed_overhead_num_blocks = static_cast<int>(fixed_overhead_num_blocks),
      .topk = static_cast<int>(topk),
      .extra_topk = static_cast<int>(extra_topk),
      .topk_length = opt_ptr(topk_length, "topk_length"),
      .extra_topk_length = opt_ptr(extra_topk_length, "extra_topk_length"),
      .seqlens_k = opt_ptr(seqlens_k, "seqlens_k"),
      .tile_scheduler_metadata = static_cast<int*>(tile_scheduler_metadata.data_ptr()),
      .num_splits = static_cast<int*>(num_splits.data_ptr()),
      .num_sm_parts = num_sm_parts,
      .max_states = 0,
  };

  // A windowed or parallel schedule that does not fit falls back to the serial one.
  if (schedule == kWindowed && topk > 0) {
    const std::size_t windowed_smem = sizeof(int) * static_cast<std::size_t>(windowed_smem_ints(b, num_sm_parts));
    if (windowed_smem <= kMaxDynamicSmem) {
      runtime::set_max_dynamic_smem_per_device<flashmla_windowed_sched_meta_kernel>(device_.unwrap(), kMaxDynamicSmem);
      LaunchKernel(1, kBlockSize, device_.unwrap(), windowed_smem)(flashmla_windowed_sched_meta_kernel, p);
      return;
    }
  }

  std::size_t smem = sizeof(int) * static_cast<std::size_t>(serial_smem_ints(b, num_sm_parts));
  RuntimeCheck(smem <= kMaxDynamicSmem, "schedule does not fit in shared memory: ", smem, " bytes");
  // DCP's actual per-rank lengths can be much smaller than top-k. Reserve
  // what fits, and let the kernel fall back if this replay needs more states.
  if (schedule == kParallel && topk > 0 && b >= kMinParallelBatch) {
    const int64_t row_blocks = (topk + extra_topk + block_size_n - 1) / block_size_n + 1;
    const int64_t free_ints = kParallelSmemLimit / static_cast<int>(sizeof(int)) - serial_smem_ints(b, num_sm_parts) -
                              parallel_walk_smem_ints(b, num_sm_parts, 0);
    const int64_t max_states = std::min(static_cast<int64_t>(b) * row_blocks, std::max<int64_t>(free_ints / 2, 0));
    const std::size_t extra =
        sizeof(int) * static_cast<std::size_t>(parallel_walk_smem_ints(b, num_sm_parts, static_cast<int>(max_states)));
    if (max_states > 0 && smem + extra <= kParallelSmemLimit) {
      p.max_states = static_cast<int>(max_states);
      smem += extra;
    }
  }
  // The kernel also has static shared memory. Opt in so schedules near
  // 48 KiB do not exceed the default combined static + dynamic limit.
  runtime::set_max_dynamic_smem_per_device<flashmla_sched_meta_kernel>(device_.unwrap(), kMaxDynamicSmem);
  LaunchKernel(1, kBlockSize, device_.unwrap(), smem)(flashmla_sched_meta_kernel, p);
}

}  // namespace sglang
