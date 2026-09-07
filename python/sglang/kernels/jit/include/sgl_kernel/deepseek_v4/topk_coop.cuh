/// \file topk_coop.cuh
/// \brief Grid-wide cooperative top-k for a single very long row.
///
/// The kernels in `csrc/deepseek_v4/topk_v2.cuh` spread one row over at most one
/// 8-block cluster, which leaves most of a B200 idle at batch size 1. This kernel
/// spreads that row over every SM instead, so it needs a cooperative launch.
///
/// It does not replace them: the two run back to back on the same stream and
/// partition the rows by length. `topk_small_batch_kernel` returns early for rows
/// above `TopKPagedParams::coop_floor`, this kernel returns early for rows at or
/// below `CoopParams::floor`, and `TopKKernel::transform_paged` sets both from one
/// host value, so exactly one of the two writes any row.

#pragma once

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

#include <cstdint>

#ifndef USE_ROCM
#include <cooperative_groups.h>

namespace sglang::coop_topk {

namespace impl = device::topk;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kMaxNumTie = impl::TopKConfig::kMaxNumTie;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;

// Level 0 bins the top 12 bits of the key and each refine the next 10, so the
// three levels (shifts 20/10/0) cover the whole 32-bit key and ties are exact.
constexpr uint32_t kHist0 = 4096;
constexpr uint32_t kHistRefine = 1024;
constexpr uint32_t kNumLevels = 3;
static_assert(kHist0 % kBlockSize == 0 && kHistRefine % kBlockSize == 0);

// Keys come from impl::extract_exact_bin, so the band search and the
// handle_tie it feeds agree on ordering. NaN is removed before the map is
// applied; see drop_nan.

struct Counters {
  uint32_t win;
  uint32_t tie;
};
constexpr uint32_t kCounterWords = sizeof(Counters) / sizeof(uint32_t);

/**
 * \brief Cross-block state, owned by the caller and reused across launches.
 *
 * Must be zero at first use. Each launch leaves `hist` zero again and zeroes the
 * counter slot the next launch will read, so the caller never clears it;
 * `parity` is the one field that carries over, and it alternates.
 */
struct CoopWorkspace {
  uint32_t hist[kNumLevels][kHist0];
  Counters cnt[2];
  uint32_t parity;
  impl::TieValue ties[kMaxNumTie];
};

struct CoopParams {
  const float* __restrict__ scores;      // one row, [seq_len], 16B-aligned base
  const int32_t* __restrict__ seq_lens;  // row length, read on the device
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ out;  // [topk]
  CoopWorkspace* __restrict__ ws;
  uint32_t topk;
  uint32_t page_bits;
  uint32_t floor;  // rows with seq_len <= floor belong to the official kernel
};

struct CoopSmem {
  uint32_t hist[kHist0];
  uint32_t warp_sum[kNumWarps];
  uint32_t bin;      // threshold bin of the current level
  uint32_t above;    // elements above the threshold bin
  uint32_t incount;  // elements inside the threshold bin
  uint32_t seq_len;
  impl::TopKConfig::TieHandleSmem tie_handle;
  impl::TieValue ties[kMaxNumTie];
};

// Past 48 KB a block's shared memory has to be dynamic and opted into with
// cudaFuncAttributeMaxDynamicSharedMemorySize before any occupancy query.
static_assert(sizeof(CoopSmem) <= 48 * 1024, "static shared memory limit");

/// Map NaN to -inf. Its integer key outranks +inf, so a NaN score would be
/// selected, and it makes the tie comparator answer false both ways, which
/// collides ranks and leaves output slots unwritten.
SGL_DEVICE float drop_nan(float v) {
  return v != v ? -INFINITY : v;
}

/// Visit block `g`'s share of the row, interleaved in runs of 16 float4, so one
/// iteration of a block covers a 256-byte-contiguous stretch. Every value is
/// passed through `drop_nan`, so all three passes rank the same sanitized row.
template <typename F>
SGL_DEVICE void for_each_score(const float* scores, uint32_t seq_len, uint32_t g, uint32_t G, F&& fn) {
  constexpr uint32_t kRun = 16;
  const uint32_t tx = threadIdx.x;
  const uint32_t num_vecs = seq_len / 4;
  const auto* vecs = reinterpret_cast<const float4*>(scores);
  for (uint32_t j = tx;; j += kBlockSize) {
    const uint32_t vi = (j / kRun) * (G * kRun) + g * kRun + (j % kRun);
    if (vi >= num_vecs) break;
    const float4 v = vecs[vi];
    const uint32_t base = vi * 4u;
    fn(drop_nan(v.x), base);
    fn(drop_nan(v.y), base + 1);
    fn(drop_nan(v.z), base + 2);
    fn(drop_nan(v.w), base + 3);
  }
  const uint32_t tail_start = num_vecs * 4;
  if (g == 0 && tx < seq_len - tail_start) {
    fn(drop_nan(scores[tail_start + tx]), tail_start + tx);
  }
}

/// Suffix-scan the histogram for the bin holding the `need`-th largest key, with
/// `kBinsPerThread` consecutive bins per thread. Writes `bin` (the threshold bin),
/// `above` (count strictly above it) and `incount` (its own population).
template <uint32_t kBinsPerThread>
SGL_DEVICE void scan_hist(CoopSmem* smem, uint32_t need) {
  const uint32_t tx = threadIdx.x;
  uint32_t bins[kBinsPerThread];
  uint32_t owned = 0;
#pragma unroll
  for (uint32_t i = 0; i < kBinsPerThread; ++i) {
    bins[i] = smem->hist[tx * kBinsPerThread + i];
    owned += bins[i];
  }
  const uint32_t lane = tx % device::kWarpThreads;
  const uint32_t warp = tx / device::kWarpThreads;
  uint32_t inclusive = owned;
#pragma unroll
  for (uint32_t off = 1; off < device::kWarpThreads; off <<= 1) {
    const uint32_t peer = __shfl_down_sync(device::kFullMask, inclusive, off);
    if (lane + off < device::kWarpThreads) inclusive += peer;
  }
  if (lane == 0) smem->warp_sum[warp] = inclusive;
  __syncthreads();
  uint32_t higher_warps = 0;
  for (uint32_t w = warp + 1; w < kNumWarps; ++w)
    higher_warps += smem->warp_sum[w];
  uint32_t run = inclusive + higher_warps - owned;  // bins owned by threads above tx
#pragma unroll
  for (int i = static_cast<int>(kBinsPerThread) - 1; i >= 0; --i) {
    const uint32_t count = bins[i];
    if (run < need && run + count >= need) {
      smem->bin = tx * kBinsPerThread + static_cast<uint32_t>(i);
      smem->above = run;
      smem->incount = count;
    }
    run += count;
  }
  __syncthreads();
}

/**
 * \brief Select the top-k of one row across the whole grid.
 *
 * \tparam kTransform apply the page-table transform to the selected indices.
 *
 * Every block histograms its share of the row into shared memory and merges into
 * the workspace; after a grid barrier all blocks derive the same threshold band,
 * and refines narrow it to the full key width. The final pass writes the elements
 * above the band straight into `out` and collects the band's members as tie
 * candidates, which block 0 then resolves with the shared `handle_tie`. A band
 * still holding more than kMaxNumTie candidates at full key width keeps an
 * arbitrary kMaxNumTie of them; they are bit-identical, so the result is a valid
 * top-k, but which indices it names is not fixed across launches.
 */
template <bool kTransform>
__global__ __launch_bounds__(kBlockSize) void coop_topk_kernel(const __grid_constant__ CoopParams p) {
  device::enable_smem_spilling();
  __shared__ CoopSmem smem;
  const auto grid = cooperative_groups::this_grid();
  const uint32_t g = blockIdx.x;
  const uint32_t G = gridDim.x;
  const uint32_t tx = threadIdx.x;

  if (tx == 0) smem.seq_len = static_cast<uint32_t>(p.seq_lens[0]);
  __syncthreads();
  const uint32_t seq_len = smem.seq_len;
  // Inverse of the guard in topk_small_batch_kernel. Every block reads the same
  // device-side length and returns together before touching the workspace or the
  // grid barrier, so a captured graph routes per replay with no host branch. A
  // length of 0 (a padded or capture-time row) exits here as well.
  if (seq_len <= p.floor) return;
  // The host holds floor >= kClusterFloorSmall > kMaxTopK, so seq_len <= topk
  // belongs to the official kernel and needs no trivial path here.
  const auto problem = impl::TopKProblem{
      .in = p.scores,
      .out = p.out,
      .page_table = p.page_table,
      .topk = p.topk,
      .seq_len = seq_len,
      .page_bits = p.page_bits,
  };

  // Read before the first barrier; block 0 flips it after the last one.
  const uint32_t parity = p.ws->parity;
  Counters* cnt = &p.ws->cnt[parity];

  for (uint32_t b = tx; b < kHist0; b += kBlockSize)
    smem.hist[b] = 0;
  __syncthreads();
  for_each_score(p.scores, seq_len, g, G, [&](float val, uint32_t) {
    atomicAdd(&smem.hist[impl::extract_exact_bin(val) >> 20], 1);
  });
  __syncthreads();
  for (uint32_t b = tx; b < kHist0; b += kBlockSize) {
    if (smem.hist[b]) atomicAdd(&p.ws->hist[0][b], smem.hist[b]);
  }
  grid.sync();

  uint32_t level = 0, band_shift = 20, band_prefix = 0, need = p.topk;
  for (;;) {
    const uint32_t num_bins = level == 0 ? kHist0 : kHistRefine;
    for (uint32_t b = tx; b < num_bins; b += kBlockSize)
      smem.hist[b] = p.ws->hist[level][b];
    __syncthreads();
    if (level == 0) {
      scan_hist<kHist0 / kBlockSize>(&smem, need);
      band_prefix = smem.bin;
    } else {
      scan_hist<kHistRefine / kBlockSize>(&smem, need);
      band_prefix = (band_prefix << 10) | smem.bin;
    }
    if (smem.incount <= kMaxNumTie || band_shift == 0) break;

    need -= smem.above;
    band_shift -= 10;
    ++level;
    for (uint32_t b = tx; b < kHistRefine; b += kBlockSize)
      smem.hist[b] = 0;
    __syncthreads();
    for_each_score(p.scores, seq_len, g, G, [&](float val, uint32_t) {
      const uint32_t key = impl::extract_exact_bin(val);
      if ((key >> (band_shift + 10)) == band_prefix) {
        atomicAdd(&smem.hist[(key >> band_shift) & (kHistRefine - 1)], 1);
      }
    });
    __syncthreads();
    for (uint32_t b = tx; b < kHistRefine; b += kBlockSize) {
      if (smem.hist[b]) atomicAdd(&p.ws->hist[level][b], smem.hist[b]);
    }
    grid.sync();
  }

  for_each_score(p.scores, seq_len, g, G, [&](float val, uint32_t idx) {
    const uint32_t key = impl::extract_exact_bin(val);
    const uint32_t banded = band_shift ? (key >> band_shift) : key;
    if (banded > band_prefix) {
      const uint32_t pos = atomicAdd(&cnt->win, 1);
      if (pos < p.topk) [[likely]]
        p.out[pos] = static_cast<int32_t>(idx);
    } else if (banded == band_prefix) {
      const uint32_t slot = atomicAdd(&cnt->tie, 1);
      if (slot < kMaxNumTie) [[likely]]
        p.ws->ties[slot] = {val, idx};
    }
  });
  grid.sync();

  // Slot `parity` is read-only from here on, so block 0 can resolve the ties
  // while the others transform the winner slots and clear the workspace for the
  // next launch, including `cnt[1 - parity]` -- the slot the next launch reads,
  // since block 0 flips `parity` below.
  const uint32_t above_count = min(cnt->win, p.topk);
  auto* hist = reinterpret_cast<uint32_t*>(p.ws->hist);
  if (g != 0) {
    if constexpr (kTransform) {
      for (uint32_t t = (g - 1) * kBlockSize + tx; t < above_count; t += (G - 1) * kBlockSize) {
        problem.transform_output(t, p.out[t]);
      }
    }
    for (uint32_t i = (g - 1) * kBlockSize + tx; i < kNumLevels * kHist0; i += (G - 1) * kBlockSize) {
      hist[i] = 0;
    }
    if (g == 1 && tx < kCounterWords) reinterpret_cast<uint32_t*>(&p.ws->cnt[1u - parity])[tx] = 0;
    return;
  }

  const uint32_t tie_count = min(cnt->tie, kMaxNumTie);
  for (uint32_t t = tx; t < tie_count; t += kBlockSize)
    smem.ties[t] = p.ws->ties[t];
  __syncthreads();
  if (tx == 0) p.ws->parity = 1u - parity;
  impl::TopKConfig::handle_tie(smem.ties, problem, above_count, tie_count, p.topk - above_count, &smem.tie_handle);
  __syncthreads();
  if constexpr (kTransform) {
    for (uint32_t t = above_count + tx; t < p.topk; t += kBlockSize) {
      problem.transform_output(t, p.out[t]);
    }
  }
  if (G == 1) {
    if constexpr (kTransform) {
      for (uint32_t t = tx; t < above_count; t += kBlockSize)
        problem.transform_output(t, p.out[t]);
    }
    for (uint32_t i = tx; i < kNumLevels * kHist0; i += kBlockSize)
      hist[i] = 0;
    if (tx < kCounterWords) {
      reinterpret_cast<uint32_t*>(&p.ws->cnt[0])[tx] = 0;
      reinterpret_cast<uint32_t*>(&p.ws->cnt[1])[tx] = 0;
    }
  }
}

/// Bytes of workspace the caller must provide, zero-initialized on allocation.
constexpr int64_t workspace_bytes() {
  return static_cast<int64_t>(sizeof(CoopWorkspace));
}

/**
 * \brief Enqueue the cooperative top-k on the same stream as the official kernel.
 *
 * One block per SM, which on a 148-SM B200 is close to the 144 blocks the speedup
 * was measured with. The occupancy query is what makes the grid legal: a
 * cooperative launch needs every block co-resident, so it fails outright if even
 * one block does not fit.
 *
 * No PDL here. The kernel it takes rows from runs first on the same stream, so
 * stream order already sequences them.
 */
template <bool kTransform>
inline void launch(const CoopParams& params, DLDevice device) {
  const auto kernel = coop_topk_kernel<kTransform>;
  // Not cached: get_blocks_per_sm queries whichever device is current, so a
  // process driving two devices would launch the second with the first's grid.
  const uint32_t blocks_per_sm = host::runtime::get_blocks_per_sm(kernel, kBlockSize);
  const uint32_t sm_count = host::runtime::get_sm_count(device.device_id);
  host::RuntimeCheck(blocks_per_sm >= 1, "cooperative top-k does not fit one block per SM");
  host::LaunchKernel(sm_count, kBlockSize, device).config({.cooperative = true}).launch(kernel, params);
}

}  // namespace sglang::coop_topk

#endif  // !USE_ROCM
