// Prefill top-512 for the QSA indexer with fast_topk's exact output: the same
// indices in the same slots for every row.
//
// fast_topk (fast_topk.cuh, deterministic_topk.cuh) returns, for a row longer than
// 4096, the 512 largest ordered_key()s with ties at the cutoff taken by smallest
// index, emitted by its fixed warp / lane / word-round / bit walk of a bitmap of
// the winners. This kernel finds the same winners from one pass over the row: a
// sampled floor that keeps between 512 and kCap candidates, an 8-bit radix select
// of the 512th candidate key in shared memory, smallest-index ties, and then the
// same bitmap walk. Rows of 4096 or fewer, rows past one 131072-key tile, and rows
// whose candidates or ties do not fit (heavy ties) run fast_topk's own device
// function, so they are fast_topk by construction.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include "deterministic_topk.cuh"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

namespace sglang {

namespace qsa_prefill_topk_detail {

// fast_topk's block size: its emission order is defined in terms of it.
constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kTopK = 512;
constexpr int kSample = 2048;
constexpr int kCap = 2048;
constexpr int kMaxTies = 256;
constexpr int kTile = 8192 * 16;  // deterministic_topk's single-tile limit
constexpr int kWords = kTile / 32;
constexpr int kWordRounds = kWords / kThreads;
// Candidate keys and ids, then the winner bitmap (which first holds the sample):
// fast_topk's 32 KB of bins, which its select reuses on the fallback.
constexpr size_t kSmemBytes = 2 * kCap * sizeof(uint32_t) + kWords * sizeof(uint32_t);
static_assert(kSmemBytes == 8 * 1024 * sizeof(uint32_t));
static_assert(kSample <= kWords);

struct Params {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, kTopK]
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

struct Scratch {
  int hist[256];
  uint32_t ties[kMaxTies];
  uint32_t warp_sel[kWarps];
  uint32_t bin;
  int remain, count, above, tie_count;
};

// The k-th largest (1-based, k <= n) of n keys in shared memory: four 8-bit passes.
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
      // Lane l owns bins 255 - 8l down to 248 - 8l; scan from the top bin.
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

// Returns false (CTA-uniform, output untouched) when the row needs the stock select.
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
  // Twice the expected sample rank of the 512th key, plus slack: about 2x512 candidates.
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
  // Ties at the cutoff: the `need` smallest indices.
  for (int i = tid; i < ties; i += kThreads) {
    const uint32_t id = s.ties[i];
    int smaller = 0;
    for (int j = 0; j < ties; ++j) smaller += s.ties[j] < id;
    if (smaller < need) atomicOr(&bitmap[id >> 5], 1u << (id & 31u));
  }
  __syncthreads();

  // fast_topk's emission over the winner bitmap (radix_select_impl's last loop, one tile).
  const int rounds = (words + kThreads - 1) / kThreads;
  uint32_t chosen[kWordRounds];
  uint32_t chosen_count = 0;
#pragma unroll
  for (int r = 0; r < kWordRounds; ++r) {
    const int word = warp * rounds * 32 + r * 32 + lane;
    chosen[r] = (r < rounds && word < words) ? bitmap[word] : 0u;
    chosen_count += __popc(chosen[r]);
  }
  const uint32_t lane_base = warp_exclusive(chosen_count);
  if (lane == 31) s.warp_sel[warp] = lane_base + chosen_count;
  __syncthreads();
  if (tid < 32) {
    const uint32_t count = lane < kWarps ? s.warp_sel[lane] : 0;
    const uint32_t prior = warp_exclusive(count);
    if (lane < kWarps) s.warp_sel[lane] = prior;
  }
  __syncthreads();
  uint32_t slot = s.warp_sel[warp] + lane_base;
#pragma unroll
  for (int r = 0; r < kWordRounds; ++r) {
    const int word = warp * rounds * 32 + r * 32 + lane;
    uint32_t bits = chosen[r];
    while (bits) {
      const int bit = __ffs(bits) - 1;
      output[slot++] = word * 32 + bit;
      bits &= bits - 1;
    }
  }
  return true;
}

template <bool kUsePDL>
__global__ __launch_bounds__(kThreads, 2) void qsa_prefill_topk_kernel(const Params __grid_constant__ params) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ Scratch scratch;
  device::PDLWaitPrimary<kUsePDL>();
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const int row_start = params.row_starts[bid];
  const int length = params.lengths[bid];
  int32_t* indice = params.indices + bid * kTopK;
  const float* score = params.input + bid * params.input_stride;
  if (length <= kTopK) {
    for (int i = threadIdx.x; i < kTopK; i += kThreads) indice[i] = (i < length) ? i : -1;
  } else if (length <= 4096 || length > kTile || !sampled_select(score + row_start, indice, length, smem, scratch)) {
    __syncthreads();
    deterministic_topk_detail::radix_select<kTopK, kThreads, false, true>(score, indice, row_start, length);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_prefill_topk_detail

/**
 * \brief fast_topk<512>'s output (same indices, same slots) for prefill rows.
 *
 * Row b selects from score[b, row_starts[b] : row_starts[b] + lengths[b]) and
 * writes indices relative to row_starts[b] into indices[b]; -1 fills the slots
 * past lengths[b] when lengths[b] < 512.
 */
template <bool kUsePDL>
struct QsaPrefillTopKKernel {
  static constexpr auto kernel = qsa_prefill_topk_detail::qsa_prefill_topk_kernel<kUsePDL>;

  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView row_starts,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"length"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(score);
    TensorMatcher({B})  // row_starts
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(row_starts);
    TensorMatcher({B, qsa_prefill_topk_detail::kTopK})  // indices
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(indices);
    TensorMatcher({B})  // lengths
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(lengths);

    const auto params = qsa_prefill_topk_detail::Params{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = static_cast<const int32_t*>(row_starts.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = S.unwrap(),
    };
    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    LaunchKernel(num_rows, qsa_prefill_topk_detail::kThreads, device.unwrap(), qsa_prefill_topk_detail::kSmemBytes)
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
