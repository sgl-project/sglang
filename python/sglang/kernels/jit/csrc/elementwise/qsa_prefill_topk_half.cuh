// qsa_prefill_topk.cuh's sampled select on 512-thread blocks, four per SM instead
// of two 1024-thread blocks, with the same output: the same winners in the same
// slots for every row.
//
// The winner set does not depend on the block size (histogram counts, an atomic
// candidate compaction whose order the selection never reads, smallest-index
// ties). The slot order does: fast_topk emits the winner bitmap by its 1024-thread
// walk (warp, then lane, then word round, then bit). Each warp here walks two of
// those 32 virtual warps, lane for lane, so the slots come out the same.
//
// Rows the sampled select does not take (4096 keys or fewer, past one 131072-key
// tile, too few or too many candidates, heavy ties) are flagged; a second launch
// of qsa_prefill_topk.cuh's own 1024-thread code serves exactly those rows.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include "deterministic_topk.cuh"
#include "qsa_prefill_topk.cuh"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

namespace sglang {

namespace qsa_prefill_topk_half_detail {

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kStockThreads = 1024;  // fast_topk's block: its emission order's unit
constexpr int kStockWarps = kStockThreads / 32;
constexpr int kVirtual = kStockWarps / kWarps;  // virtual warps walked per warp
constexpr int kTopK = 512;
constexpr int kSample = 2048;
constexpr int kCap = 2048;
constexpr int kMaxTies = 256;
constexpr int kTile = 8192 * 16;
constexpr int kWords = kTile / 32;
constexpr int kWordRounds = kWords / kStockThreads;
constexpr size_t kSmemBytes = 2 * kCap * sizeof(uint32_t) + kWords * sizeof(uint32_t);
static_assert(kSmemBytes == qsa_prefill_topk_detail::kSmemBytes);
static_assert(kStockWarps % kWarps == 0);

struct Params {
  const float* __restrict__ input;
  const int32_t* __restrict__ row_starts;
  int32_t* __restrict__ indices;
  const int32_t* __restrict__ lengths;
  int32_t* __restrict__ fallback;  // [B]: 1 where the 1024-thread pass must serve the row
  int64_t input_stride;
};

struct Scratch {
  int hist[256];
  uint32_t ties[kMaxTies];
  uint32_t warp_sel[kStockWarps];
  uint32_t bin;
  int remain, count, above, tie_count;
};

// qsa_prefill_topk_detail::kth_largest with this block's stride.
SGL_DEVICE uint32_t kth_largest(const uint32_t* keys, int n, int k, Scratch& s) {
  const int tid = threadIdx.x, lane = tid & 31;
  uint32_t prefix = 0, mask = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = tid; b < 256; b += kThreads) s.hist[b] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += kThreads) {
      const uint32_t key = keys[i];
      if ((key & mask) == prefix) atomicAdd(&s.hist[(key >> shift) & 255u], 1);
    }
    __syncthreads();
    if (tid < 32) {
      int c[8], sum = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) sum += (c[j] = s.hist[255 - 8 * lane - j]);
      const int before = deterministic_topk_detail::warp_exclusive(sum);
      if (before < k && k <= before + sum) {
        int acc = before;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (acc < k && k <= acc + c[j]) {
            s.bin = 255u - 8u * lane - j;
            s.remain = k - acc;
          }
          acc += c[j];
        }
      }
    }
    __syncthreads();
    prefix |= s.bin << shift;
    mask |= 255u << shift;
    k = s.remain;
    __syncthreads();
  }
  return prefix;
}

// qsa_prefill_topk_detail::sampled_select with this block's stride and the
// 1024-thread emission walk; false (CTA-uniform, output untouched) for a row
// the stock select must serve.
SGL_DEVICE bool sampled_select(const float* __restrict__ input, int32_t* __restrict__ output, int length,
                               uint8_t* smem, Scratch& s) {
  using deterministic_topk_detail::ordered_key;
  using deterministic_topk_detail::warp_exclusive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  auto* cand_keys = reinterpret_cast<uint32_t*>(smem);
  auto* cand_ids = cand_keys + kCap;
  auto* bitmap = cand_ids + kCap;
  auto* sample = bitmap;

  for (int i = tid; i < kSample; i += kThreads) {
    sample[i] = ordered_key(input[static_cast<int64_t>(i) * length / kSample]);
  }
  if (tid == 0) s.count = 0;
  __syncthreads();
  const int rank = min(kSample, static_cast<int>((2LL * kTopK * kSample) / length) + 9);
  const uint32_t floor_key = kth_largest(sample, kSample, rank, s);

  for (int base = 0; base < length; base += 4 * kThreads) {
    uint32_t keys[4];
    unsigned masks[4];
    int total = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int id = base + i * kThreads + tid;
      keys[i] = id < length ? ordered_key(input[id]) : 0u;
      masks[i] = __ballot_sync(0xffffffffu, id < length && keys[i] >= floor_key);
      total += __popc(masks[i]);
    }
    int slot = 0;
    if (lane == 0 && total) slot = atomicAdd(&s.count, total);
    slot = __shfl_sync(0xffffffffu, slot, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if ((masks[i] >> lane) & 1u) {
        const int at = slot + __popc(masks[i] & ((1u << lane) - 1u));
        if (at < kCap) {
          cand_keys[at] = keys[i];
          cand_ids[at] = static_cast<uint32_t>(base + i * kThreads + tid);
        }
      }
      slot += __popc(masks[i]);
    }
  }
  __syncthreads();
  const int found = s.count;
  if (found < kTopK || found > kCap) return false;

  const uint32_t cutoff = kth_largest(cand_keys, found, kTopK, s);
  const int words = (length + 31) / 32;
  for (int w = tid; w < words; w += kThreads) bitmap[w] = 0u;
  if (tid == 0) s.above = s.tie_count = 0;
  __syncthreads();
  for (int i = tid; i < found; i += kThreads) {
    const uint32_t key = cand_keys[i], id = cand_ids[i];
    if (key > cutoff) {
      atomicOr(&bitmap[id >> 5], 1u << (id & 31u));
      atomicAdd(&s.above, 1);
    } else if (key == cutoff) {
      const int at = atomicAdd(&s.tie_count, 1);
      if (at < kMaxTies) s.ties[at] = id;
    }
  }
  __syncthreads();
  const int need = kTopK - s.above, ties = s.tie_count;
  if (ties > kMaxTies) return false;
  for (int i = tid; i < ties; i += kThreads) {
    const uint32_t id = s.ties[i];
    int smaller = 0;
    for (int j = 0; j < ties; ++j) smaller += s.ties[j] < id;
    if (smaller < need) atomicOr(&bitmap[id >> 5], 1u << (id & 31u));
  }
  __syncthreads();

  // The 1024-thread emission: virtual warp v reads words v * rounds * 32 + r * 32 + lane
  // and emits after all lower virtual warps, lane by lane, round by round.
  const int rounds = (words + kStockThreads - 1) / kStockThreads;
  uint32_t lane_base[kVirtual];
#pragma unroll
  for (int j = 0; j < kVirtual; ++j) {
    const int v = warp * kVirtual + j;
    uint32_t count = 0;
#pragma unroll
    for (int r = 0; r < kWordRounds; ++r) {
      const int word = v * rounds * 32 + r * 32 + lane;
      if (r < rounds && word < words) count += __popc(bitmap[word]);
    }
    lane_base[j] = warp_exclusive(count);
    if (lane == 31) s.warp_sel[v] = lane_base[j] + count;
  }
  __syncthreads();
  if (tid < 32) {
    const uint32_t count = s.warp_sel[lane];
    const uint32_t prior = warp_exclusive(count);
    s.warp_sel[lane] = prior;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kVirtual; ++j) {
    const int v = warp * kVirtual + j;
    uint32_t slot = s.warp_sel[v] + lane_base[j];
#pragma unroll
    for (int r = 0; r < kWordRounds; ++r) {
      const int word = v * rounds * 32 + r * 32 + lane;
      uint32_t bits = (r < rounds && word < words) ? bitmap[word] : 0u;
      while (bits) {
        const int bit = __ffs(bits) - 1;
        output[slot++] = word * 32 + bit;
        bits &= bits - 1;
      }
    }
  }
  return true;
}

template <bool kUsePDL>
__global__ __launch_bounds__(kThreads, 4) void qsa_prefill_topk_half_kernel(const Params __grid_constant__ params) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ Scratch scratch;
  device::PDLWaitPrimary<kUsePDL>();
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const int row_start = params.row_starts[bid];
  const int length = params.lengths[bid];
  int32_t* indice = params.indices + bid * kTopK;
  const float* score = params.input + bid * params.input_stride;
  bool stock = false;
  if (length <= kTopK) {
    for (int i = threadIdx.x; i < kTopK; i += kThreads) indice[i] = (i < length) ? i : -1;
  } else if (length <= 4096 || length > kTile) {
    stock = true;
  } else {
    stock = !sampled_select(score + row_start, indice, length, smem, scratch);
  }
  if (threadIdx.x == 0) params.fallback[bid] = stock ? 1 : 0;
  device::PDLTriggerSecondary<kUsePDL>();
}

// qsa_prefill_topk_kernel's body, for the flagged rows only.
template <bool kUsePDL>
__global__ __launch_bounds__(kStockThreads, 2) void qsa_prefill_topk_rest_kernel(const Params __grid_constant__ params) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ qsa_prefill_topk_detail::Scratch scratch;
  device::PDLWaitPrimary<kUsePDL>();
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  if (params.fallback[bid] != 0) {
    const int row_start = params.row_starts[bid];
    const int length = params.lengths[bid];
    int32_t* indice = params.indices + bid * kTopK;
    const float* score = params.input + bid * params.input_stride;
    if (length <= 4096 || length > kTile ||
        !qsa_prefill_topk_detail::sampled_select(score + row_start, indice, length, smem, scratch)) {
      __syncthreads();
      deterministic_topk_detail::radix_select<kTopK, kStockThreads, false, true>(score, indice, row_start, length);
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_prefill_topk_half_detail

/**
 * \brief QsaPrefillTopKKernel's output from 512-thread blocks plus a 1024-thread
 * pass over the rows they leave (flagged in fallback[b]).
 */
template <bool kUsePDL>
struct QsaPrefillTopKHalfKernel {
  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView row_starts,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView fallback) {
    using namespace host;
    namespace d = qsa_prefill_topk_half_detail;
    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"length"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, L}).with_strides({S, 1}).with_dtype<fp32_t>().with_device(device).verify(score);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(row_starts);
    TensorMatcher({B, d::kTopK}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(fallback);

    const auto params = d::Params{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = static_cast<const int32_t*>(row_starts.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .fallback = static_cast<int32_t*>(fallback.data_ptr()),
        .input_stride = S.unwrap(),
    };
    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    LaunchKernel(num_rows, d::kThreads, device.unwrap(), d::kSmemBytes)
        .enable_pdl(kUsePDL)(d::qsa_prefill_topk_half_kernel<kUsePDL>, params);
    LaunchKernel(num_rows, d::kStockThreads, device.unwrap(), d::kSmemBytes)
        .enable_pdl(kUsePDL)(d::qsa_prefill_topk_rest_kernel<kUsePDL>, params);
  }
};

}  // namespace sglang
