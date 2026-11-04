// Exact FP32 QSA selection: one resident cluster per row, no candidate clipping.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/runtime.cuh>
#ifdef Q38_PERF_SHORT_SCAN
#include "q38_perf_topk_short512.cuh"
#else
#include "q38_mk_topk_short512.cuh"
#endif
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

namespace sglang {

// Keep this integer operation in PTX. An FP abs/neg rewrite canonicalizes NaNs
// differently and invalidates the histogram/scatter agreement (SHARD audit).
__device__ __forceinline__ uint32_t q38_mk_ordered_key(float value) {
  return deterministic_topk_detail::ordered_key(value);
}

template <int kItems, int kCluster, bool kPDL, bool kLinear, int kNarrow, bool kSingleLaunch = false>
__global__ __launch_bounds__(512) void q38_mk_topk_kernel(
    const float* scores, const int32_t* lengths, const int32_t* starts,
    int32_t* output, int64_t stride) {
  static_assert(kItems <= 32, "adaptive compact bitmask capacity");
  static_assert(kNarrow == 32 || kNarrow == 64);
  namespace cg = cooperative_groups;
  auto cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int row = blockIdx.x / kCluster;
  const int tid = threadIdx.x;
  device::PDLWaitPrimary<kPDL>();
  const int length = lengths[row];
  if (length <= 4096) {
    if constexpr (kSingleLaunch) {
      // The branch is uniform within a cluster. Other CTAs can exit because
      // the owner short path has no DSMEM accesses or cluster barriers.
      if (rank == 0) {
        int32_t* indices = output + row * 512;
        if (length <= 512)
          q38_mk_short512_detail::naive_topk<512>(indices, length);
        else
          q38_mk_short512_detail::radix_select_topk<512>(
              scores + row * stride, indices, starts ? starts[row] : 0, length);
      }
    }
    device::PDLTriggerSecondary<kPDL>();
    return;
  }
  const int start = starts ? starts[row] : 0;
  // Contiguous shards fix cutoff tie membership independently of emission.
  constexpr int kShard = kItems * 512;
  uint32_t keys[kItems];
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    const int index = rank * kShard + i * 512 + tid;
    keys[i] = index < length ? q38_mk_ordered_key(scores[row * stride + start + index]) : 0u;
  }
  __shared__ uint32_t hist[2048];
  __shared__ uint32_t merged_hist[2048];
  __shared__ uint32_t pivot, need, cursor, ties, done, narrow_count;
  __shared__ uint32_t small_count, small_keys[kNarrow], small_ids[kNarrow];
  __shared__ unsigned long long compact_counts;
  __shared__ unsigned long long warp_counts[kItems][16], prior_counts;
  __shared__ uint32_t total_above, output_warps[16];
  __shared__ int tie_item, tie_warp, tie_need, tie_id;
  using Scan = cub::BlockScan<uint32_t, 512>;
  union Scratch {
    typename Scan::TempStorage cutoff;
  };
  __shared__ Scratch scratch;
  if (tid == 0) { pivot = 0; need = 512; cursor = 0; ties = 0; done = 0; }
  uint32_t prefix = 0, mask = 0;
  // 11+11+10 covers the same 32-bit ordered FP32 key exactly. Each
  // thread owns four descending bins for the root's 2048-item block scan.
  for (int pass = 0; pass < 3; ++pass) {
    const int shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
    const uint32_t digit_mask = pass == 2 ? 1023u : 2047u;
    hist[tid] = 0;
    hist[tid + 512] = 0;
    hist[tid + 1024] = 0;
    hist[tid + 1536] = 0;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int index = rank * kShard + i * 512 + tid;
      const uint32_t key = keys[i];
      const bool active = index < length && (key & mask) == prefix;
      if (active) {
        const uint32_t digit = (key >> shift) & digit_mask;
        atomicAdd(hist + digit, 1u);
      }
    }
    cluster.sync();
    if (rank == 0) {
      uint32_t counts[4], inclusive[4];
      const uint32_t remaining = need;
      // Coalesced DSMEM reads first. The blocked CUB scan layout otherwise
      // turns each remote warp load into four bank-conflicted transactions.
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int bin = tid + j * 512;
        uint32_t count = 0;
#pragma unroll
        for (int r = 0; r < kCluster; ++r)
          count += cluster.map_shared_rank(hist, r)[bin];
        merged_hist[bin] = count;
      }
      // Publish the local transpose and snapshot need before cutoff writes.
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int digit = int(digit_mask) - (4 * tid + j);
        counts[j] = digit >= 0 ? merged_hist[digit] : 0u;
      }
      Scan(scratch.cutoff).InclusiveSum(counts, inclusive);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (inclusive[j] >= remaining && inclusive[j] - counts[j] < remaining) {
          const int digit = int(digit_mask) - (4 * tid + j);
          pivot |= uint32_t(digit) << shift;
          need = remaining - (inclusive[j] - counts[j]);
          done = inclusive[j] == remaining;
          narrow_count = counts[j];
        }
      }
    }
    cluster.sync();
    prefix = *cluster.map_shared_rank(&pivot, 0);
    mask |= digit_mask << shift;
    if (*cluster.map_shared_rank(&done, 0)) break;
    // Every rank in this row takes the same branch. Histograms prove that the
    // entire boundary bucket fits, so the local arrays cannot clip candidates.
    const uint32_t boundary_count = *cluster.map_shared_rank(&narrow_count, 0);
    if (pass == 1 && boundary_count <= kNarrow) {
      if (tid == 0) small_count = 0;
      __syncthreads();
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const int index = rank * kShard + i * 512 + tid;
        if (index < length && (keys[i] & mask) == prefix) {
          const uint32_t slot = atomicAdd(&small_count, 1u);
          small_keys[slot] = keys[i];
          small_ids[slot] = index;
        }
      }
      cluster.sync();
      if (rank == 0 && tid < 32) {
        constexpr int kRegs = kNarrow / 32;
        unsigned long long packed[kRegs];
#pragma unroll
        for (int i = 0; i < kRegs; ++i) {
          const int position = tid + i * 32;
          packed[i] = 0ull;
          if (position < boundary_count) {
            int offset = position, owner = 0;
#pragma unroll
            for (int r = 0; r < kCluster; ++r) {
              const uint32_t count = *cluster.map_shared_rank(&small_count, r);
              if (offset < count) { owner = r; break; }
              offset -= count;
            }
            const uint32_t key = cluster.map_shared_rank(small_keys, owner)[offset];
            const uint32_t id = cluster.map_shared_rank(small_ids, owner)[offset];
            packed[i] = (static_cast<unsigned long long>(key) << 32) | (0xffffffffu - id);
          }
        }
        // Sort at most 32/64 keys in one warp, descending score then ascending
        // relative ID. Padding0 is strictly below every valid packed entry.
#pragma unroll
        for (int size = 2; size <= kNarrow; size *= 2) {
#pragma unroll
          for (int step = size / 2; step > 0; step /= 2) {
            unsigned long long other[kRegs];
#pragma unroll
            for (int i = 0; i < kRegs; ++i) {
              if constexpr (kNarrow == 64) {
                other[i] = step == 32 ? packed[i ^ 1] : __shfl_xor_sync(0xffffffffu, packed[i], step);
              } else {
                other[i] = __shfl_xor_sync(0xffffffffu, packed[i], step);
              }
            }
#pragma unroll
            for (int i = 0; i < kRegs; ++i) {
              const int index = tid + i * 32;
              const bool take_max = ((index & size) == 0) == ((index & step) == 0);
              packed[i] = take_max ? (packed[i] > other[i] ? packed[i] : other[i])
                                   : (packed[i] < other[i] ? packed[i] : other[i]);
            }
          }
        }
        const uint32_t take = need;
#pragma unroll
        for (int i = 0; i < kRegs; ++i) {
          const int position = tid + i * 32;
          if (position + 1 == take) pivot = uint32_t(packed[i] >> 32);
        }
      }
      cluster.sync();
      prefix = *cluster.map_shared_rank(&pivot, 0);
      mask = 0xffffffffu;
      break;
    }
  }
  uint32_t above_bits = 0, equal_bits = 0;
#pragma unroll
  for (int i = 0; i < kItems; ++i) {
    const int id = rank * kShard + i * 512 + tid;
    const uint32_t key = keys[i] & mask;
    const bool above = id < length && key > prefix, equal = id < length && key == prefix;
    above_bits |= uint32_t(above) << i;
    equal_bits |= uint32_t(equal) << i;
    const unsigned a_mask = __ballot_sync(0xffffffffu, above);
    const unsigned e_mask = __ballot_sync(0xffffffffu, equal);
    if ((tid & 31) == 0)
      warp_counts[i][tid / 32] = uint32_t(__popc(a_mask)) | (static_cast<unsigned long long>(__popc(e_mask)) << 32);
  }
  __syncthreads();
  if (tid < 32) {
    uint32_t above = 0, equal = 0;
#pragma unroll
    for (int i = 0; i < kItems; ++i) if (tid < 16) {
      above += uint32_t(warp_counts[i][tid]);
      equal += uint32_t(warp_counts[i][tid] >> 32);
    }
    above = __reduce_add_sync(0xffffffffu, above);
    equal = __reduce_add_sync(0xffffffffu, equal);
    if (tid == 0) compact_counts = above | (static_cast<unsigned long long>(equal) << 32);
  }
  cluster.sync();
  if (tid == 0) {
    unsigned long long total = 0, prior = 0;
#pragma unroll
    for (int r = 0; r < kCluster; ++r) {
      const auto counts = *cluster.map_shared_rank(&compact_counts, r);
      total += counts;
      if (r < rank) prior += counts;
    }
    total_above = uint32_t(total);
    prior_counts = prior;
  }
  __syncthreads();
  const uint32_t take_equal = 512u - total_above;
  const int local_need = int(take_equal) - int(prior_counts >> 32);
  if (local_need <= 0) {
    if (tid == 0) tie_id = -1;
  } else if (local_need >= int(compact_counts >> 32)) {
    if (tid == 0) tie_id = length;
  } else {
    // Only the CTA containing the cutoff tie resolves its original ID.
    if (tid < 32) {
      uint32_t carry = 0;
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const uint32_t count = tid < 16 ? uint32_t(warp_counts[i][tid] >> 32) : 0;
        const uint32_t prior = deterministic_topk_detail::warp_exclusive(count);
        const uint32_t total = __shfl_sync(0xffffffffu, prior + count, 15);
        if (carry + total >= uint32_t(local_need)) {
          const unsigned hit = __ballot_sync(0xffffffffu, tid < 16 && carry + prior < uint32_t(local_need) && carry + prior + count >= uint32_t(local_need));
          const int winner = __ffs(hit) - 1;
          const int need = local_need - int(carry + __shfl_sync(0xffffffffu, prior, winner));
          if (tid == 0) { tie_item = i; tie_warp = winner; tie_need = need; }
          break;
        }
        carry += total;
      }
    }
    __syncthreads();
    if (tid / 32 == tie_warp) {
      const unsigned equal = __ballot_sync(0xffffffffu, (equal_bits >> tie_item) & 1u);
      const unsigned chosen = deterministic_topk_detail::lowest_bits(equal, tie_need);
      if ((tid & 31) == 0) tie_id = rank * kShard + tie_item * 512 + tie_warp * 32 + 31 - __clz(chosen);
    }
  }
  __syncthreads();
  const int delta = tie_id - (rank * kShard + tid);
  const uint32_t allowed_count = delta < 0 ? 0u : min(uint32_t(kItems), uint32_t(delta / 512 + 1));
  const uint32_t allowed = allowed_count == 32 ? 0xffffffffu : (1u << allowed_count) - 1u;
  const uint32_t chosen = above_bits | (equal_bits & allowed);
  const uint32_t count = __popc(chosen);
  const uint32_t lane_base = deterministic_topk_detail::warp_exclusive(count);
  if ((tid & 31) == 31) output_warps[tid / 32] = lane_base + count;
  __syncthreads();
  if (tid < 32) {
    const uint32_t mine = tid < 16 ? output_warps[tid] : 0;
    const uint32_t prior = deterministic_topk_detail::warp_exclusive(mine);
    if (tid < 16) output_warps[tid] = prior;
  }
  __syncthreads();
  uint32_t slot = uint32_t(prior_counts) + min(uint32_t(prior_counts >> 32), take_equal)
      + output_warps[tid / 32] + lane_base;
#pragma unroll
  for (int i = 0; i < kItems; ++i)
    if (chosen & (1u << i)) output[row * 512 + slot++] = rank * kShard + i * 512 + tid;
  // DSMEM owners must not exit while another rank can access their memory.
  cluster.sync();
  device::PDLTriggerSecondary<kPDL>();
}

template <int kItems, int kCluster, bool kPDL, bool kLinear, int kNarrow, bool kSingleLaunch = false>
void q38_mk_topk(tvm::ffi::TensorView scores, tvm::ffi::TensorView lengths,
                 tvm::ffi::Optional<tvm::ffi::TensorView> starts, tvm::ffi::TensorView output) {
  using namespace host;
  auto rows = SymbolicSize{"rows"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({rows, -1}).with_strides({-1, 1}).with_dtype<float>().with_device(device).verify(scores);
  TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(lengths);
  TensorMatcher({rows, 512}).with_dtype<int32_t>().with_device(device).verify(output);
  const int32_t* ptr = nullptr;
  if (starts.has_value()) {
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(starts.value());
    ptr = static_cast<const int32_t*>(starts.value().data_ptr());
  }
  CHECK_HOST(scores.size(1) <= int64_t(kItems) * 512 * kCluster);
  constexpr std::size_t dynamic_smem = kSingleLaunch ? q38_mk_short512_detail::kSmemBytes : 0;
  if constexpr (kSingleLaunch) {
    // Combined static + native short stash exceeds the48KiB default. Set
    // the per-device opt-in once before capture, without a device sync.
    runtime::set_max_dynamic_smem_per_device<
        q38_mk_topk_kernel<kItems, kCluster, kPDL, kLinear, kNarrow, kSingleLaunch>>(
            device.unwrap(), dynamic_smem);
  }
  LaunchKernel(uint32_t(rows.unwrap()) * kCluster, 512, device.unwrap(), dynamic_smem)
      .enable_cluster(dim3(kCluster, 1, 1)).enable_pdl(kPDL)(
          q38_mk_topk_kernel<kItems, kCluster, kPDL, kLinear, kNarrow, kSingleLaunch>, static_cast<const float*>(scores.data_ptr()),
          static_cast<const int32_t*>(lengths.data_ptr()), ptr,
          static_cast<int32_t*>(output.data_ptr()), scores.stride(0));
}
}  // namespace sglang
