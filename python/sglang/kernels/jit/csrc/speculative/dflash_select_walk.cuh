#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/activation.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>
#include <sgl_kernel/ptx/mma.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace device::dflash_select_walk {

namespace ptx {

using device::activation::ptx::ex2_approx_ftz;
using namespace device::ptx;

}  // namespace ptx

inline constexpr uint32_t kMaxGamma = 8;  // gamma = the cluster size, at most the portable limit

template <uint32_t K>
struct SelectWalkTrait {
  static constexpr uint32_t R = 256;  // codebook row length
  static constexpr uint32_t kNumWarps = 4;
  static constexpr uint32_t kNumThreads = kNumWarps * 32;
  static constexpr uint32_t kRowPitch = R + 8;   // bf16 per codebook row in smem: 528 B, conflict-free ldmatrix
  static constexpr uint32_t kListPitch = K + 4;  // words per rank list in smem: conflict-free 16 B stores
  // Lattice: m16n8k16 tiles, the R reduction split over the warps.
  static constexpr uint32_t kNumMTiles = K / 16;
  static constexpr uint32_t kNumNTiles = K / 8;
  static constexpr uint32_t kKStepsPerWarp = R / 16 / kNumWarps;
  // Row epilogue: kLanesPerRow threads per lattice row, kColsPerLane columns each.
  static constexpr uint32_t kLanesPerRow = kNumThreads / K;
  static constexpr uint32_t kColsPerLane = K / kLanesPerRow;
  static_assert(K == 16 || K == 32);
};

template <uint32_t K, uint32_t kGamma>
struct alignas(16) SelectWalkSmem {
  using Trait = SelectWalkTrait<K>;
  struct CodebookRows {
    bf16_t pred[K][Trait::kRowPitch];  // A rows of the predecessors (lattice rows)
    bf16_t succ[K][Trait::kRowPitch];  // B rows of the candidates (lattice columns)
  };
  // The partials reuse the rows' bytes (the rows are dead by then): K 32 needs 43 KB of static smem instead of 61 KB.
  union {
    CodebookRows rows;
    float partial[Trait::kNumWarps][K][K + 4];  // each warp's k-split partial lattice, once the rows are read
  };
  static_assert(sizeof(float) * Trait::kNumWarps * K * (K + 4) <= sizeof(CodebookRows));
  alignas(16) uint32_t lists[2][distributed::kMaxWorldSize][Trait::kListPitch];  // per merging warp, per rank
  alignas(16) bf16x2_t hp[Trait::R / 2];
  alignas(16) float q[K][K];
  float unary[K];
  int32_t cand_ids[K];
  alignas(16) uint8_t maps[kGamma][K];  // every slot's map, sent by its CTA
  alignas(16) uint8_t map[K];           // this slot's map
  uint64_t bar;
  float temperature, uniform;
  bool greedy;
};

enum Role { WARP_MERGE_SUCC = 0, WARP_MERGE_PRED = 1, WARP_INPUTS = 2, WARP_BAR_INIT = 3 };

SGL_DEVICE uint32_t unorder16(uint32_t key) {
  return key ^ ((key & 0x8000u) ? 0x8000u : 0xFFFFu);
}

struct SelectWalkParams {
  uint8_t* __restrict__ workspace;           // this rank's push workspace: [2][TP] slots of [Nmax][K] words
  distributed::EpochCounter* counter;        // the plane's phase
  const bf16_t* __restrict__ pred_codebook;  // A [V, R]
  const bf16_t* __restrict__ succ_codebook;  // B [V, R]
  const bf16_t* __restrict__ hp;             // [bs * gamma, R], row stride hp_stride
  const int64_t* __restrict__ anchor_ids;    // [bs], element stride anchor_stride
  const float* __restrict__ temperatures;    // [bs]
  const bool* __restrict__ greedy_mask;      // [bs]
  const float* __restrict__ uniforms;        // [bs, gamma]
  int64_t* __restrict__ tokens;              // [bs, gamma]
  int64_t* __restrict__ candidate_ids;       // [bs, gamma, K]
  float* __restrict__ q;                     // [bs, gamma, K]
  int64_t hp_stride;
  int64_t anchor_stride;
  float output_multiplier;
  uint32_t tp_size;
  uint32_t shard_width;
  uint32_t slot_bytes;
};

// Rank `src`'s words of `row` in this rank's workspace, for `phase`.
template <uint32_t K>
SGL_DEVICE uint32_t* slot_row(const SelectWalkParams& p, uint32_t phase, uint32_t src, uint32_t row) {
  return reinterpret_cast<uint32_t*>(assume_global(p.workspace) + (phase * p.tp_size + src) * p.slot_bytes) + row * K;
}

struct Candidate {
  int32_t id;
  uint32_t word;
};

// The smallest of words [kBegin, kEnd), as a tree of 3-input mins (VIMNMX3): depth log3(N).
template <uint32_t kBegin, uint32_t kEnd, std::size_t M>
SGL_DEVICE uint32_t min_words(const AlignedVector<uint32_t, 8> (&words)[M]) {
  constexpr uint32_t N = kEnd - kBegin;
  if constexpr (N == 1) {
    return words[kBegin / 8][kBegin % 8];
  } else if constexpr (N == 2) {
    return min(min_words<kBegin, kBegin + 1>(words), min_words<kBegin + 1, kEnd>(words));
  } else {
    constexpr uint32_t a = kBegin + N / 3, b = a + (N - N / 3) / 2;
    return min(min(min_words<kBegin, a>(words), min_words<a, b>(words)), min_words<b, kEnd>(words));
  }
}

// One warp polls the TP sorted lists of `row` (lane r < TP: rank r's list), merges them into the global top K and
// gathers the codebook row of each candidate as soon as it is picked: round t pops candidate t (warp max of the list
// heads; the winner is uniform), round t + 1 reads its word, round t + 2 issues the warp's cp.async of its row into
// `rows[t]`, so the gathers overlap the rounds still to come. Returns candidate c in lane c < K.
template <uint32_t K>
SGL_DEVICE Candidate merge_gather_row(
    const SelectWalkParams& p,
    uint32_t row,
    uint32_t phase,
    uint32_t (*lists)[SelectWalkTrait<K>::kListPitch],
    const bf16_t* table,
    bf16_t (*rows)[SelectWalkTrait<K>::kRowPitch]) {
  using Trait = SelectWalkTrait<K>;
  static_assert(distributed::kMaxWorldSize <= 32 && K <= 32);  // lane r polls rank r; lane c returns candidate c
  const auto lane_id = threadIdx.x % 32;
  const bool active = lane_id < p.tp_size;
  const auto src = slot_row<K>(p, phase, active ? lane_id : 0, row);

  AlignedVector<uint32_t, 8> list[K / 8];
  do {
#pragma unroll
    for (uint32_t v = 0; v < K / 8; ++v) {
      ptx::ld_relaxed_32B(list[v], src, v);
    }
  } while (!__all_sync(0xFFFFFFFFu, min_words<0, K>(list) != 0));

  // Merge keys: the value, then the lower rank (ranks hold increasing id ranges). Distinct across lanes, and never 0
  // in an active lane.
  const auto rank_tag = 0xFFFFu - lane_id;
  const auto merge_key = [rank_tag](uint32_t word) { return (word & 0xFFFF0000u) | rank_tag; };
  // Each active lane holds the keys of its next three list entries and loads the fourth (refill) a round ahead.
  uint32_t head = 0, next1 = 0, next2 = 0, refill = 0;
  if (active) {
#pragma unroll
    for (uint32_t v = 0; v < K / 4; ++v) {  // 16 B stores: kListPitch keeps the rows 16 B aligned, not 32 B
      AlignedVector<uint32_t, 4> quarter;
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        quarter[j] = list[v / 2][(v % 2) * 4 + j];
      }
      quarter.store(lists[lane_id], v);
    }
    head = merge_key(list[0][0]);
    next1 = merge_key(list[0][1]);
    next2 = merge_key(list[0][2]);
    refill = merge_key(list[0][3]);
  }
  __syncwarp();  // round t + 1 reads the winner's word from another lane's list

  uint32_t pos = 2;                // the list index of next2
  uint32_t rank1 = 0, index1 = 0;  // candidate t - 1: its rank, its index in that rank's list
  uint32_t rank2 = 0, word2 = 0;   // candidate t - 2: its rank and word
  Candidate mine = {0, 0};         // lane c: candidate c
#pragma unroll
  for (uint32_t t = 0; t < K + 2; ++t) {
    uint32_t word1 = 0;
    if (t >= 1 && t <= K) word1 = lists[rank1][index1];
    if (t >= 2) {
      const auto c = t - 2;
      const auto id = static_cast<int32_t>(rank2 * p.shard_width + (word2 & 0xFFFFu) - 1);  // position + 1
      // One 16 B copy per lane.
      static_assert(Trait::R == 32 * 8);
      const auto offset = lane_id * 8;
      ptx::cp_async_16B(&rows[c][offset], table + static_cast<int64_t>(id) * Trait::R + offset);
      mine.id = lane_id == c ? id : mine.id;
      mine.word = lane_id == c ? word2 : mine.word;
    }
    uint32_t rank0 = 0, index0 = 0;
    if (t < K) {
      const auto m = warp::reduce_max(head);
      rank0 = 0xFFFFu - (m & 0xFFFFu);
      index0 = warp::broadcast(pos - 2, rank0);
      const bool hit = head == m;
      head = hit ? next1 : head;
      next1 = hit ? next2 : next1;
      next2 = hit ? (pos + 1 < K ? refill : 0u) : next2;
      pos += hit;
      refill = active ? merge_key(lists[lane_id][min(pos + 1, K - 1)]) : 0u;
    }
    rank2 = rank1;
    word2 = word1;
    rank1 = rank0;
    index1 = index0;
  }
  return mine;
}

template <uint32_t K, uint32_t kGamma, bool kUsePDL>
__global__ __launch_bounds__(SelectWalkTrait<K>::kNumThreads)  //
    void select_walk_kernel(const __grid_constant__ SelectWalkParams p) {
  using Trait = SelectWalkTrait<K>;
  constexpr uint32_t R = Trait::R;
  static_assert(kGamma >= 1 && kGamma <= kMaxGamma);
  static_assert(R == 32 * 8);  // a warp moves an hp or codebook row as one 16 B vector per lane
  __shared__ SelectWalkSmem<K, kGamma> smem;

  const auto lane_id = threadIdx.x % 32;
  const auto warp_id = warp::reduce_max(threadIdx.x / 32);
  const auto e = ptx::cluster_ctarank();  // the slot
  const auto b = blockIdx.y;              // the request
  const auto n = b * kGamma + e;          // the row

  if (threadIdx.x == WARP_BAR_INIT * 32) {
    ptx::mbar_init(&smem.bar, 1);
    ptx::mbar_arrive_expect_tx(&smem.bar, kGamma * K);
    ptx::fence_barrier_init();
  }
  ptx::bar_cluster_arrive_relaxed();

  const auto phase = p.counter->load();
  if (warp_id <= WARP_MERGE_PRED) {  // either pred/succ
    if (warp_id == WARP_MERGE_PRED && e == 0) {
      // Every predecessor row of slot 0 is the anchor's: one load, K shared-memory copies.
      const auto anchor = p.anchor_ids[b * p.anchor_stride];
      AlignedVector<bf16_t, 8> v;
      v.load(p.pred_codebook + anchor * R, lane_id);
#pragma unroll
      for (uint32_t i = 0; i < K; ++i) {
        v.store(smem.rows.pred[i], lane_id);
      }
    } else {
      const bool succ = warp_id == WARP_MERGE_SUCC;
      const auto mine = merge_gather_row<K>(
          p,
          succ ? n : n - 1,
          phase,
          succ ? smem.lists[0] : smem.lists[1],
          succ ? p.succ_codebook : p.pred_codebook,
          succ ? smem.rows.succ : smem.rows.pred);
      if (succ && lane_id < K) {
        smem.cand_ids[lane_id] = mine.id;
        smem.unary[lane_id] = __uint_as_float(unorder16(mine.word >> 16) << 16) * p.output_multiplier;
        p.candidate_ids[n * K + lane_id] = mine.id;
      }
      ptx::cp_async_wait_all();
    }
  } else if (warp_id == WARP_INPUTS) {
    // The row's hp and the request's scalars, for every warp behind the __syncthreads below.
    AlignedVector<bf16x2_t, 4> hp;
    hp.load(p.hp + p.hp_stride * n, lane_id);
    hp.store(smem.hp, lane_id);
    smem.temperature = p.temperatures[b];
    smem.greedy = p.greedy_mask[b];
    smem.uniform = p.uniforms[n];
  }
  __syncthreads();

  // score[i][c] = unary[c] + bf16(sum_r bf16(A[prev_i][r] * hp[r]) * B[c][r]), i on M, c on N; k split over warps.
  {
    float acc[Trait::kNumMTiles][Trait::kNumNTiles][4] = {};
    const uint32_t t = lane_id % 4;
#pragma unroll
    for (uint32_t ks = 0; ks < Trait::kKStepsPerWarp; ++ks) {
      const uint32_t kk = warp_id * Trait::kKStepsPerWarp + ks;
      const auto h0 = smem.hp[kk * 8 + 0 + t];
      const auto h1 = smem.hp[kk * 8 + 4 + t];
      // B fragments, two n-tiles per ldmatrix: n-tile 2 nn in words 0-1, n-tile 2 nn + 1 in words 2-3.
      AlignedVector<bf16x2_t, 4> bf[Trait::kNumNTiles / 2];
#pragma unroll
      for (uint32_t nn = 0; nn < Trait::kNumNTiles / 2; ++nn) {
        const uint32_t mat = lane_id / 8;
        const uint32_t row = nn * 16 + (mat / 2) * 8 + lane_id % 8;
        ptx::ldmatrix_x4(&smem.rows.succ[row][kk * 16 + (mat % 2) * 8], bf[nn]);
      }
#pragma unroll
      for (uint32_t mm = 0; mm < Trait::kNumMTiles; ++mm) {
        AlignedVector<bf16x2_t, 4> af;
        ptx::ldmatrix_x4(&smem.rows.pred[mm * 16 + lane_id % 16][kk * 16 + (lane_id / 16) * 8], af);
        af[0] = __hmul2(af[0], h0);
        af[1] = __hmul2(af[1], h0);
        af[2] = __hmul2(af[2], h1);
        af[3] = __hmul2(af[3], h1);
#pragma unroll
        for (uint32_t nt = 0; nt < Trait::kNumNTiles; ++nt) {
          AlignedVector<bf16x2_t, 2> b;
          b[0] = bf[nt / 2][(nt % 2) * 2];
          b[1] = bf[nt / 2][(nt % 2) * 2 + 1];
          ptx::mma_m16n8k16_bf16(acc[mm][nt], af, b);
        }
      }
    }
    const uint32_t g = lane_id / 4;
    __syncthreads();  // every warp's ldmatrix reads of the rows are done before the partials overwrite them
#pragma unroll
    for (uint32_t mm = 0; mm < Trait::kNumMTiles; ++mm) {
#pragma unroll
      for (uint32_t nt = 0; nt < Trait::kNumNTiles; ++nt) {
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
          AlignedVector<float, 2> pair;
          pair[0] = acc[mm][nt][h * 2];
          pair[1] = acc[mm][nt][h * 2 + 1];
          pair.store(&smem.partial[warp_id][mm * 16 + h * 8 + g][nt * 8 + 2 * t]);
        }
      }
    }
  }
  __syncthreads();

  // Row epilogue: sum the k-split partials, round the dot to bf16, add the unary, then the walk's choice from this
  // row (map) and the row's q. All four warps, kLanesPerRow lanes per row; lane h of a row takes the kColsPerLane
  // columns starting at h * kColsPerLane.
  {
    constexpr uint32_t kLanesPerRow = Trait::kLanesPerRow;
    constexpr uint32_t kColsPerLane = Trait::kColsPerLane;
    const uint32_t row = threadIdx.x / kLanesPerRow;
    const uint32_t h = threadIdx.x % kLanesPerRow;
    const uint32_t c0 = h * kColsPerLane;
    float v[kColsPerLane];
    static_assert(Trait::kNumWarps == 4);
#pragma unroll
    for (uint32_t c = 0; c < kColsPerLane; ++c) {
      const float x0 = smem.partial[0][row][c0 + c], x1 = smem.partial[1][row][c0 + c];
      const float x2 = smem.partial[2][row][c0 + c], x3 = smem.partial[3][row][c0 + c];
      v[c] = __bfloat162float(__float2bfloat16_rn((x0 + x1) + (x2 + x3))) + smem.unary[c0 + c];
    }
    uint32_t index;
    if (smem.greedy) {
      // First argmax, NaN first (torch.argmax): per lane the first NaN and the first max, then across the row's
      // lanes; ties go to the lower column.
      uint32_t nan_at = K, best_at = c0;
      float best = v[0];
#pragma unroll
      for (uint32_t c = 0; c < kColsPerLane; ++c) {
        nan_at = (v[c] != v[c] && nan_at == K) ? c0 + c : nan_at;
        const bool take = v[c] > best;
        best_at = take ? c0 + c : best_at;
        best = take ? v[c] : best;
      }
#pragma unroll
      for (uint32_t o = 1; o < kLanesPerRow; o *= 2) {
        nan_at = min(nan_at, __shfl_xor_sync(0xFFFFFFFFu, nan_at, o));
        const float ob = __shfl_xor_sync(0xFFFFFFFFu, best, o);
        const uint32_t oa = __shfl_xor_sync(0xFFFFFFFFu, best_at, o);
        const bool other_wins = ob > best || (ob == best && oa < best_at);
        best = other_wins ? ob : best;
        best_at = other_wins ? oa : best_at;
      }
      index = nan_at < K ? nan_at : best_at;
#pragma unroll
      for (uint32_t c = 0; c < kColsPerLane; ++c) {
        v[c] = c0 + c == index ? 1.0f : 0.0f;
      }
    } else {
      float mx = v[0];
#pragma unroll
      for (uint32_t c = 1; c < kColsPerLane; ++c) {
        mx = fmaxf(mx, v[c]);
      }
      mx = warp::reduce_max<1, kLanesPerRow>(mx);
      // exp(s / T - max(s / T)) as ex2((s - max s) * log2(e) / T): T > 0 keeps the argmax; each exponent may move
      // by an ulp against the division form.
      const float scale = 1.4426950408889634f / smem.temperature;
      const float shift = mx * scale;
      float sum[4] = {};
#pragma unroll
      for (uint32_t c = 0; c < kColsPerLane; ++c) {
        v[c] = ptx::ex2_approx_ftz(fmaf(v[c], scale, -shift));
        sum[c % 4] += v[c];
      }
      const auto mine = (sum[0] + sum[1]) + (sum[2] + sum[3]);
      const auto incl = warp::inclusive_sum<kLanesPerRow>(mine, h);  // over the row's lanes, in column order
      const auto total = warp::broadcast(incl, (lane_id / kLanesPerRow) * kLanesPerRow + kLanesPerRow - 1);
      // Inverse CDF on the unnormalized running sum: u * total >= cumsum(e) instead of u >= cumsum(e / total).
      const auto target = smem.uniform * total;
      uint32_t count = 0;
      float cdf = incl - mine;
#pragma unroll
      for (uint32_t c = 0; c < kColsPerLane; ++c) {
        cdf += v[c];
        count += target >= cdf;
      }
      count = warp::reduce_sum<1, kLanesPerRow>(count);
      index = min(count, K - 1);
      const auto inv_total = 1.0f / total;
#pragma unroll
      for (uint32_t c = 0; c < kColsPerLane; ++c) {
        v[c] *= inv_total;
      }
    }
#pragma unroll
    for (uint32_t c = 0; c < kColsPerLane; ++c) {
      smem.q[row][c0 + c] = v[c];
    }
    if (h == 0) smem.map[row] = static_cast<uint8_t>(index);
  }
  __syncthreads();
  ptx::bar_cluster_wait_acquire();  // every peer's mbarrier is initialized past this point
  // Warp 0 finishes the round: map exchange, walk, outputs, refill, commit.
  if (warp_id != 0) return;

  static_assert(kGamma * (K / 16) <= 32);
  if (lane_id < kGamma * (K / 16)) {
    const uint32_t peer = lane_id / (K / 16);
    const uint32_t part = lane_id % (K / 16);
    AlignedVector<uint8_t, 16> map_part;
    map_part.load(smem.map, part);
    ptx::st_async_16B(ptx::mapa_shared(&smem.maps[e][part * 16], peer), map_part, ptx::mapa_shared(&smem.bar, peer));
  }
  ptx::mbar_wait_parity_cluster(&smem.bar, 0);
  // i_s = map_s[i_{s - 1}] from i_{-1} = 0 (the anchor); this slot's token is candidate i_e, its q row lattice row
  // i_{e - 1}.
  uint32_t prev = 0;
  uint32_t idx = 0;
#pragma unroll
  for (uint32_t s = 0; s < kGamma; ++s) {
    if (s <= e) {
      prev = idx;
      idx = smem.maps[s][prev];
    }
  }
  if (lane_id < K) p.q[n * K + lane_id] = smem.q[prev][lane_id];
  if (lane_id == 0) p.tokens[n] = smem.cand_ids[idx];
  // Every peer read row n (CTA e + 1 merges it as its predecessors) before sending its map.
  if (lane_id < p.tp_size) {
    const auto row = slot_row<K>(p, phase, lane_id, n);
    AlignedVector<uint32_t, 4> zero;
    zero.fill(0);
#pragma unroll
    for (uint32_t v = 0; v < K / 4; ++v) {
      zero.store(row, v);
    }
  }
  // Every warp's read of the phase precedes the __syncthreads above, so this CTA's one commit may go now.
  if (lane_id == 0) p.counter->commit(n, gridDim.y * kGamma);
  PDLTriggerSecondary<kUsePDL>();
}

}  // namespace device::dflash_select_walk

template <uint32_t K, uint32_t kGamma, bool kUsePDL>
struct DFlashSelectWalkKernel {
  static constexpr auto kernel = device::dflash_select_walk::select_walk_kernel<K, kGamma, kUsePDL>;

  static void
  run(host::distributed::PushPlaneRef plane,
      tvm::ffi::TensorView pred_codebook,
      tvm::ffi::TensorView succ_codebook,
      tvm::ffi::TensorView hp,
      tvm::ffi::TensorView anchor_ids,
      tvm::ffi::TensorView temperatures,
      tvm::ffi::TensorView greedy_mask,
      tvm::ffi::TensorView uniforms,
      tvm::ffi::TensorView tokens,
      tvm::ffi::TensorView candidate_ids,
      tvm::ffi::TensorView q,
      float output_multiplier,
      uint32_t shard_width) {
    using namespace host;
    using Trait = device::dflash_select_walk::SelectWalkTrait<K>;
    auto BS = SymbolicSize{"batch_size"};
    auto V = SymbolicSize{"vocab_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({V, Trait::R})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(pred_codebook)
        .verify(succ_codebook);
    // hp may be a column view of the projection GEMM's output (row stride W + 256).
    TensorMatcher({-1, Trait::R})  //
        .with_strides({-1, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(hp);
    // anchor_ids may be a column view of the draft block's token ids.
    TensorMatcher({BS})  //
        .with_strides({-1})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(anchor_ids);
    TensorMatcher({BS})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(temperatures);
    TensorMatcher({BS})  //
        .with_device(device_)
        .verify(greedy_mask);
    CHECK_HOST(greedy_mask.dtype().code == kDLBool && greedy_mask.dtype().bits == 8) << "greedy_mask must be bool";
    TensorMatcher({BS, kGamma})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(uniforms);
    TensorMatcher({BS, kGamma})  //
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(tokens);
    TensorMatcher({BS, kGamma, K})  //
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(candidate_ids);
    TensorMatcher({BS, kGamma, K})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(q);
    const auto bs = static_cast<uint32_t>(BS.unwrap());
    const auto& push = *plane.get();
    const auto tp = push.world_size;
    CHECK_HOST(tp <= device::distributed::kMaxWorldSize) << "select_walk: tp " << tp;
    const auto required_bytes = static_cast<int64_t>(bs) * kGamma * K * 4;
    CHECK_HOST(required_bytes <= push.slot_bytes) << required_bytes << " > " << push.slot_bytes;
    CHECK_HOST(hp.size(0) == bs * kGamma) << "select_walk: hp rows " << hp.size(0);
    CHECK_HOST(shard_width <= 0xFFFFu && static_cast<int64_t>(shard_width) * tp == V.unwrap())
        << "select_walk: shard width " << shard_width << " x tp " << tp << " != vocab " << V.unwrap();
    if (bs == 0) return;
    const auto params = device::dflash_select_walk::SelectWalkParams{
        .workspace = push.workspaces[push.rank],
        .counter = push.counter,
        .pred_codebook = static_cast<const bf16_t*>(pred_codebook.data_ptr()),
        .succ_codebook = static_cast<const bf16_t*>(succ_codebook.data_ptr()),
        .hp = static_cast<const bf16_t*>(hp.data_ptr()),
        .anchor_ids = static_cast<const int64_t*>(anchor_ids.data_ptr()),
        .temperatures = static_cast<const float*>(temperatures.data_ptr()),
        .greedy_mask = static_cast<const bool*>(greedy_mask.data_ptr()),
        .uniforms = static_cast<const float*>(uniforms.data_ptr()),
        .tokens = static_cast<int64_t*>(tokens.data_ptr()),
        .candidate_ids = static_cast<int64_t*>(candidate_ids.data_ptr()),
        .q = static_cast<float*>(q.data_ptr()),
        .hp_stride = hp.stride(0),
        .anchor_stride = anchor_ids.stride(0),
        .output_multiplier = output_multiplier,
        .tp_size = tp,
        .shard_width = shard_width,
        .slot_bytes = static_cast<uint32_t>(push.slot_bytes),
    };
    LaunchKernel({kGamma, bs}, Trait::kNumThreads, device_.unwrap())
        .config({.use_pdl = kUsePDL, .cluster_dim = {kGamma}})(kernel, params);
  }
};

}  // namespace sglang
