/// \file dsv41/mhc.cuh
/// \brief Shared pieces of the DeepSeek-V4.1 multi-head hyper-connection kernels.
///
/// A sublayer boundary consumes `pre` and `post` (one scalar per stream) and `comb` (a
/// kNumStreams x kNumStreams mixing matrix), all derived from the old residual
/// R [T, kNumStreams, kHidden]. The kernels live in csrc/deepseek_v4/; what they agree
/// on lives here.
#pragma once
#include <sgl_kernel/atomic.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang {

// The mhc kernels' own ptx, nested like `device::atomic::ptx` so the helpers cannot
// collide with the shared `device::ptx`. The using-directive keeps the shared helpers
// reachable through this one prefix, so a kernel aliases `ptx` here and sees both.
namespace device::mhc::ptx {

using namespace device::ptx;

/// \brief One `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, accumulating in `d`.
SGL_DEVICE void mma_m16n8k16_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/// \brief global -> shared::cta. Arm `bar` with `mbar_arrive_expect_tx(bytes)` first and
///        wait on it with `mbar_wait_parity`. One elected lane issues it for the warp.
SGL_DEVICE void cp_async_bulk_g2s(void* dst_smem, const void* src_gmem, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   to_shared(dst_smem)),
               "l"(src_gmem),
               "r"(bytes),
               "r"(to_shared(bar))
               : "memory");
}

SGL_DEVICE void bar_sync(uint32_t bar, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(bar), "r"(num_threads) : "memory");
}

}  // namespace device::mhc::ptx

namespace device::mhc {

inline constexpr uint32_t kHidden = 5120;
inline constexpr uint32_t kNumStreams = 4;  // hyper-connection streams
// The GEMM that produces the mixes: [M, K] x [N, K]^T, one M_TILE of rows per work item.
inline constexpr uint32_t K = kNumStreams * kHidden;                        // 20480, the residual flattened
inline constexpr uint32_t N = 2 * kNumStreams + kNumStreams * kNumStreams;  // 4 pre, 4 post, 4x4 comb
static_assert(N <= kWarpThreads, "a row's coefficients must fit one register per lane");
inline constexpr uint32_t M_TILE = 64;  // rows per work item

/// Shared comes out of the unified L1, so a large carveout starves in-flight global
/// loads. Measured on B200: crossing this costs 15% of memory throughput, as a step. The
/// budget is per kernel, so in a fused kernel every CTA pays the largest role's request.
inline constexpr uint32_t kSmemCliffBytes = 195u * 1024u;

/// The rows of the batch, split ragged rather than padded: rank r owns
/// `average + (r < remainder)` of them starting at `r * average + min(r, remainder)`. Same
/// convention as nvlink_comm's reduce-scatter, and the caller has to agree -- both
/// transports' `run` check the offset and count it passed against this.
struct RowSplit {
  uint32_t offset;
  uint32_t count;
};

inline RowSplit row_split(uint32_t total_rows, uint32_t rank, uint32_t world_size) {
  const auto avg = total_rows / world_size;
  const auto rem = total_rows % world_size;
  return {rank * avg + std::min(rank, rem), avg + (rank < rem ? 1u : 0u)};
}

/// 16B: one mma operand fragment (8 bf16), and the unit the partial mixes move in.
using Frag = device::AlignedVector<uint32_t, 4>;
using Vec4 = device::AlignedVector<float, 4>;

/// \brief The two bf16 packed into one mma operand register, widened in place.
SGL_DEVICE fp32x2_t bf16_unpack(uint32_t packed) {
  return {__uint_as_float(packed << 16), __uint_as_float(packed & 0xffff0000u)};
}

/// \brief The inverse: two floats rounded into one packed bf16 pair.
SGL_DEVICE uint32_t bf16_pack(float low, float high) {
  uint32_t packed;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(high), "f"(low));
  return packed;
}

/// \brief Call `fn(index)` for every index in [0, kCount), spread over a block's threads.
///        The trip count is compile time, so this unrolls and only the tail is predicated.
template <uint32_t kBlockSize, uint32_t kCount, typename Fn>
SGL_DEVICE void for_each(Fn&& fn) {
  constexpr uint32_t kFullPasses = kCount / kBlockSize;
  const auto tx = threadIdx.x;
  if constexpr (kFullPasses > 0) {
#pragma unroll
    for (uint32_t pass = 0; pass < kFullPasses; ++pass)
      fn(pass * kBlockSize + tx);
  }
  if constexpr (kCount % kBlockSize != 0) {
    if (const uint32_t index = kFullPasses * kBlockSize + tx; index < kCount) fn(index);
  }
}

// One row's coefficients, one per lane: comb[from][to] at lane 4 * from + to, post[i] at
// 16 + i, pre[i] at 20 + i. One coalesced read per row; every use is a __shfl.
SGL_DEVICE float load_row_coeff(const float* pre, const float* post, const float* comb, uint32_t lane) {
  return lane < 16 ? comb[lane] : lane < 20 ? post[lane - 16] : lane < 24 ? pre[lane - 20] : 0.f;
}

/// \brief comb[from][to]: how much of residual stream `from` goes into stream `to`.
SGL_DEVICE float coeff_comb(float packed, uint32_t from, uint32_t to) {
  return __shfl_sync(0xffffffff, packed, from * kNumStreams + to);
}
SGL_DEVICE float coeff_post(float packed, uint32_t stream) {
  return __shfl_sync(0xffffffff, packed, 16 + stream);
}
SGL_DEVICE float coeff_pre(float packed, uint32_t stream) {
  return __shfl_sync(0xffffffff, packed, 20 + stream);
}

/// \brief One row's mixes -> pre / post / comb: sigmoid for pre and post, then a row
///        softmax over comb and sinkhorn to a doubly stochastic matrix.
SGL_DEVICE void sinkhorn_row(
    const float (&mixes)[N],
    float inv_rms,
    const float (&scale)[3],
    const float (&base)[N],
    float eps,
    float* pre,
    float* post,
    float* comb) {
  const float scale_pre = scale[0], scale_post = scale[1], scale_comb = scale[2];
  Vec4 pre_out, post_out;
#pragma unroll
  for (uint32_t j = 0; j < kNumStreams; ++j) {
    pre_out[j] = device::math::sigmoid_fast<true>(mixes[j] * inv_rms * scale_pre + base[j]) + eps;
    post_out[j] = 2.f * device::math::sigmoid_fast<true>(mixes[4 + j] * inv_rms * scale_post + base[4 + j]);
  }
  // Every coefficient row is 16B aligned: one vector store each, not four scalars.
  pre_out.store(pre);
  post_out.store(post);

  Vec4 c[kNumStreams];
#pragma unroll
  for (uint32_t j = 0; j < kNumStreams; ++j) {
    float row_max = -INFINITY;
#pragma unroll
    for (uint32_t k = 0; k < kNumStreams; ++k) {
      c[j][k] = mixes[8 + j * 4 + k] * inv_rms * scale_comb + base[8 + j * 4 + k];
      row_max = fmaxf(row_max, c[j][k]);
    }
    float row_sum = 0.f;
#pragma unroll
    for (uint32_t k = 0; k < kNumStreams; ++k) {
      c[j][k] = __expf(c[j][k] - row_max);
      row_sum += c[j][k];
    }
    const auto inv = device::math::rcp_fast(row_sum);
#pragma unroll
    for (uint32_t k = 0; k < kNumStreams; ++k)
      c[j][k] = c[j][k] * inv + eps;
  }

  const auto normalize_columns = [&] {
#pragma unroll
    for (uint32_t k = 0; k < kNumStreams; ++k) {
      const auto inv = device::math::rcp_fast(c[0][k] + c[1][k] + c[2][k] + c[3][k] + eps);
#pragma unroll
      for (uint32_t j = 0; j < kNumStreams; ++j)
        c[j][k] *= inv;
    }
  };
  normalize_columns();
#pragma unroll 1
  for (uint32_t iter = 0; iter < 19; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < kNumStreams; ++j) {
      const auto inv = device::math::rcp_fast(c[j][0] + c[j][1] + c[j][2] + c[j][3] + eps);
#pragma unroll
      for (uint32_t k = 0; k < kNumStreams; ++k)
        c[j][k] *= inv;
    }
    normalize_columns();
  }
#pragma unroll
  for (uint32_t j = 0; j < kNumStreams; ++j)
    c[j].store(comb, j);
}

// ===========================================================================
// MHC statistics: mixes = R @ W^T on tensor cores split over k, then the
// split-K sum and the sinkhorn. Two roles sharing one grid, dispatched by
// whichever kernel fuses them.
// ===========================================================================

template <uint32_t SPLIT_K_, uint32_t kNumWeightParts_, uint32_t kNumMMABlocks_, uint32_t kNumReduceBlocks_>
struct MHCStatTrait {
  // ---- configuration ----
  static constexpr uint32_t SPLIT_K = SPLIT_K_;
  static constexpr uint32_t kNumWeightParts = kNumWeightParts_;

  // ---- one mma.sync.aligned.m16n8k16 ----
  static constexpr uint32_t MMA_M = 16;
  static constexpr uint32_t MMA_N = 8;
  static constexpr uint32_t MMA_K = 16;
  static constexpr uint32_t kNumNTiles = N / MMA_N;  // 3
  static_assert(N % MMA_N == 0, "the mixes must tile the mma's n");

  // ---- block shape ----
  static constexpr uint32_t kBlockSize = 512;
  static constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;  // 16
  // m tiles per warp: the weight fragments of a k step are reused this many times.
  static constexpr uint32_t kNumMTilesPerWarp = 2;
  static constexpr uint32_t kNumWarpsM = M_TILE / MMA_M / kNumMTilesPerWarp;  // 2
  static constexpr uint32_t kNumWarpsK = kNumWarps / kNumWarpsM;              // 8
  static_assert(kNumWarpsM * kNumWarpsK == kNumWarps);

  // ---- k decomposition: split -> warp -> 16B steps ----
  static constexpr uint32_t kVecElems = 8;  // bf16 per 16B load, two mma k steps
  static constexpr uint32_t K_PER_SPLIT = K / SPLIT_K;
  static constexpr uint32_t K_PER_WARP = K_PER_SPLIT / kNumWarpsK;
  // One 16B load per lane covers kWarpThreads / 4 * kVecElems of k.
  static constexpr uint32_t K_PER_STEP = kVecElems * 4;
  static constexpr uint32_t kNumKSteps = K_PER_WARP / K_PER_STEP;
  static_assert(K % (SPLIT_K * kNumWarpsK * K_PER_STEP) == 0, "k must split evenly");

  // ---- grid ----
  // Persistent: the reduce blocks spin on the mma blocks, so the whole grid has to be
  // co-resident, and it is worth exactly as much of the device as the caller can give
  // it. The caller picks the pair.
  static constexpr uint32_t kNumMMABlocks = kNumMMABlocks_;
  static constexpr uint32_t kNumReduceBlocks = kNumReduceBlocks_;
  static constexpr uint32_t kNumBlocks = kNumMMABlocks + kNumReduceBlocks;
  static constexpr uint32_t kNumBlocksPerSplit = kNumMMABlocks / SPLIT_K;
  static_assert(kNumMMABlocks % SPLIT_K == 0, "the mma blocks split evenly over the k-splits");
  static_assert(kNumReduceBlocks > 0, "somebody has to sum the k-splits");

  // ---- shared memory ----
  static constexpr uint32_t kWeightRowVecs = K_PER_SPLIT / kVecElems;
  // An LDS.128 phase is 8 lanes, and those 8 lanes read 2 weight rows of 4 consecutive
  // vectors. The unpadded row stride is a multiple of 128B, so both rows would land on
  // the same banks; +4 vectors rotates the second row by 64B, which is exactly the half
  // of a bank row the first one leaves free.
  static constexpr uint32_t kWeightRowStride = kWeightRowVecs + 4;
  static constexpr uint32_t kNumWeightRows = kNumWeightParts * N;
  static constexpr uint32_t kWeightVecsPerWarp = K_PER_WARP / kVecElems;
  // A row is a whole number of warp-wide vectors, so the prologue needs no division.
  static constexpr uint32_t kWeightRowChunks = kWeightRowVecs / kWarpThreads;
  static constexpr uint32_t kNumWeightSweeps = (kNumWeightRows + kNumWarps - 1) / kNumWarps;
  static_assert(kWeightRowVecs % kWarpThreads == 0);

  // The mixes, the sum of squares, then padding to a multiple of 4: that keeps every
  // row 16B aligned, so the partials move as float4 instead of 25 scalar accesses.
  static constexpr uint32_t kPartialStride = N + 4;
  static constexpr uint32_t kTileVecs = M_TILE * kPartialStride / 4;
  static_assert(kPartialStride % 4 == 0 && N % 4 == 0);
  static constexpr uint32_t kNumReduceSlots = 4;  // shared slots the k-warps meet in
  static_assert(kNumWarpsK % kNumReduceSlots == 0);
};

// A block's shared memory. The two roles never coexist, so they overlay: the mma role
// holds this k-split's weight plus the slots its k-warps reduce through, the reduce
// role only the summed mixes of one tile.
template <typename Trait>
union MHCStatSmem {
  struct {
    Frag weight[Trait::kNumWeightRows][Trait::kWeightRowStride];
    float slots[Trait::kNumReduceSlots][M_TILE][Trait::kPartialStride];
  } mma;
  struct {
    float mixes[M_TILE][Trait::kPartialStride];
    // Uniform for the whole block: read once instead of per row.
    float base[N];
    float scale[3];
  } reduce;
};

// Back off between polls: each one is an atomic on the L2 slice holding the counter,
// and the mma CTAs arriving on it are the bottleneck. Arbitrary; not swept.
inline constexpr uint32_t kPollSleepNanoSecond = 64;

struct MHCStatParams {
  const bf16_t* residual;    // [M_total, K] bf16, the A operand
  const bf16_t* weight;      // [kNumWeightParts, N, K] bf16 parts of the fp32 hc_fn
  const float* scale;        // [3]
  const float* base;         // [N]
  float *pre, *post, *comb;  // [M_total, 4], [M_total, 4], [M_total, 4, 4]
  float* partial;            // [tiles][SPLIT_K][M_TILE][kPartialStride]
  atomic::Event* done;       // [tiles] one per tile, counts the k-splits written
  atomic::Event* ready;      // [tiles] coefficients published, if kPublish
  uint32_t num_rows;
  float rms_eps, hc_eps;
};

// ---------------------------------------------------------------------------
// MMA role. One k-split per CTA; warp w takes m tiles w % kNumWarpsM and the k
// sub-range w / kNumWarpsM. The fragment k order is permuted identically for A and
// B, so one 16B load of each feeds two mma k steps.
// ---------------------------------------------------------------------------
template <typename Trait>
SGL_DEVICE void run_mma(const MHCStatParams& params, uint32_t bx, MHCStatSmem<Trait>& smem) {
  const auto split_index = bx / Trait::kNumBlocksPerSplit;
  const auto first_tile = bx % Trait::kNumBlocksPerSplit;
  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto warp_m = warp_id % Trait::kNumWarpsM;
  const auto warp_k = warp_id / Trait::kNumWarpsM;
  const auto lane_n = lane_id / 4;  // n within an mma tile, also the weight row
  const auto lane_k = lane_id % 4;  // 16B slot within a k step

  // This split's weight, read once for every tile this CTA will visit. A warp owns whole
  // rows: the global read and the shared store are then both 32 consecutive vectors, and
  // the whole copy unrolls with at most the last sweep predicated.
  {
    const uint32_t split_base = split_index * Trait::kWeightRowVecs;
#pragma unroll
    for (uint32_t sweep = 0; sweep < Trait::kNumWeightSweeps; ++sweep) {
      const uint32_t row = warp_id + sweep * Trait::kNumWarps;
      if (sweep + 1 != Trait::kNumWeightSweeps || row < Trait::kNumWeightRows) {
#pragma unroll
        for (uint32_t chunk = 0; chunk < Trait::kWeightRowChunks; ++chunk) {
          const uint32_t vec = chunk * kWarpThreads + lane_id;
          smem.mma.weight[row][vec].load(
              params.weight, static_cast<int64_t>(row) * (K / Trait::kVecElems) + split_base + vec);
        }
      }
    }
  }
  // Only [0, N] of a row is ever written below; zero the padding once so a whole row
  // reduces as float4 without reading uninitialized shared.
  {
    Vec4 zero;
    zero.fill(0.f);
    for_each<Trait::kBlockSize, Trait::kNumReduceSlots * Trait::kTileVecs>([&](uint32_t i) {
      zero.store(smem.mma.slots, i);  //
    });
    __syncthreads();
  }

  const auto num_tiles = div_ceil(params.num_rows, M_TILE);
  for (uint32_t tile = first_tile; tile < num_tiles; tile += Trait::kNumBlocksPerSplit) {
    const uint32_t k_base = split_index * Trait::K_PER_SPLIT + warp_k * Trait::K_PER_WARP + lane_k * Trait::kVecElems;
    const bf16_t* residual_rows[Trait::kNumMTilesPerWarp][2];
    bool row_valid[Trait::kNumMTilesPerWarp][2];
#pragma unroll
    for (uint32_t i = 0; i < Trait::kNumMTilesPerWarp; ++i)
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t row = tile * M_TILE + (warp_m * Trait::kNumMTilesPerWarp + i) * Trait::MMA_M + lane_n + 8 * half;
        row_valid[i][half] = row < params.num_rows;
        residual_rows[i][half] = params.residual + static_cast<uint64_t>(row_valid[i][half] ? row : 0) * K + k_base;
      }

    // Only the residual is double buffered; the weight is a shared-memory read away.
    float accum[Trait::kNumMTilesPerWarp][Trait::kNumNTiles][4] = {};
    float sum_squares[Trait::kNumMTilesPerWarp][2] = {};
    Frag residual_vec[2][Trait::kNumMTilesPerWarp][2];
    Frag zero_frag;
    zero_frag.fill(0u);
    const auto load_residual = [&](uint32_t step, uint32_t buffer) {
#pragma unroll
      for (uint32_t i = 0; i < Trait::kNumMTilesPerWarp; ++i)
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half)
          if (row_valid[i][half])
            residual_vec[buffer][i][half].load(residual_rows[i][half], step * 4);
          else
            residual_vec[buffer][i][half] = zero_frag;
    };
    load_residual(0, 0);
#pragma unroll
    for (uint32_t step = 0; step < Trait::kNumKSteps; ++step) {
      const uint32_t buffer = step & 1;
      if (step + 1 < Trait::kNumKSteps) load_residual(step + 1, buffer ^ 1);
#pragma unroll
      for (uint32_t i = 0; i < Trait::kNumMTilesPerWarp; ++i) {
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          Frag vec = residual_vec[buffer][i][half];
#pragma unroll
          for (uint32_t e = 0; e < 4; ++e) {
            const auto [low, high] = bf16_unpack(vec[e]);
            sum_squares[i][half] += low * low;
            sum_squares[i][half] += high * high;
          }
        }
      }
      // One weight fragment at a time, read where it is used: it already lives in
      // shared, and this order lets the same fragment serve both m tiles.
#pragma unroll
      for (uint32_t part = 0; part < Trait::kNumWeightParts; ++part)
#pragma unroll
        for (uint32_t n_tile = 0; n_tile < Trait::kNumNTiles; ++n_tile) {
          Frag w = smem.mma.weight[part * N + n_tile * Trait::MMA_N + lane_n]
                                  [warp_k * Trait::kWeightVecsPerWarp + lane_k + step * 4];
#pragma unroll
          for (uint32_t i = 0; i < Trait::kNumMTilesPerWarp; ++i) {
            Frag lo = residual_vec[buffer][i][0], hi = residual_vec[buffer][i][1];
            const uint32_t a0[4] = {lo[0], hi[0], lo[1], hi[1]};
            const uint32_t a1[4] = {lo[2], hi[2], lo[3], hi[3]};
            ptx::mma_m16n8k16_bf16(accum[i][n_tile], a0, w[0], w[1]);
            ptx::mma_m16n8k16_bf16(accum[i][n_tile], a1, w[2], w[3]);
          }
        }
    }

    // The k-warps meet in kNumReduceSlots shared slots over as many rounds.
#pragma unroll
    for (uint32_t i = 0; i < Trait::kNumMTilesPerWarp; ++i)
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
        sum_squares[i][half] = warp::reduce_sum<4, 1>(sum_squares[i][half]);
      }
#pragma unroll
    for (uint32_t round = 0; round < Trait::kNumWarpsK / Trait::kNumReduceSlots; ++round) {
      if (warp_k / Trait::kNumReduceSlots == round) {
        auto* slot = smem.mma.slots[warp_k % Trait::kNumReduceSlots];
#pragma unroll
        for (uint32_t i = 0; i < Trait::kNumMTilesPerWarp; ++i) {
          const uint32_t row = (warp_m * Trait::kNumMTilesPerWarp + i) * Trait::MMA_M + lane_n;
#pragma unroll
          for (uint32_t n_tile = 0; n_tile < Trait::kNumNTiles; ++n_tile)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
              const uint32_t r = row + 8 * (e / 2);
              const uint32_t n = n_tile * Trait::MMA_N + 2 * lane_k + (e % 2);
              slot[r][n] = round == 0 ? accum[i][n_tile][e] : slot[r][n] + accum[i][n_tile][e];
            }
          if (lane_k == 0) {
            slot[row][N] = round == 0 ? sum_squares[i][0] : slot[row][N] + sum_squares[i][0];
            slot[row + 8][N] = round == 0 ? sum_squares[i][1] : slot[row + 8][N] + sum_squares[i][1];
          }
        }
      }
      __syncthreads();
    }

    float* out =
        params.partial + static_cast<uint64_t>(tile * Trait::SPLIT_K + split_index) * M_TILE * Trait::kPartialStride;
    for_each<Trait::kBlockSize, Trait::kTileVecs>([&](uint32_t i) {
      Vec4 sum;
      sum.fill(0.f);
#pragma unroll
      for (uint32_t slot = 0; slot < Trait::kNumReduceSlots; ++slot) {
        Vec4 v;
        v.load(smem.mma.slots[slot], i);
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e)
          sum[e] += v[e];
      }
      sum.store(out, i);
    });
    __syncthreads();
    if (warp_id == 0 && warp::elect_one_lane()) params.done[tile].arrive_async();
  }
}

// ---------------------------------------------------------------------------
// Reduce role: sum a tile's k-splits and turn the mixes into the coefficients.
// ---------------------------------------------------------------------------
template <typename Trait, bool kPublish>
SGL_DEVICE void run_reduction(const MHCStatParams& params, uint32_t bx, MHCStatSmem<Trait>& smem) {
  const auto tx = threadIdx.x;
  for_each<Trait::kBlockSize, N / 4>([&](uint32_t i) {
    Vec4 v;
    v.load(params.base, i);
    v.store(smem.reduce.base, i);
  });
  if (tx < 3) smem.reduce.scale[tx] = params.scale[tx];
  __syncthreads();

  const auto num_tiles = div_ceil(params.num_rows, M_TILE);
  const auto warp_id = tx / kWarpThreads;
  for (uint32_t tile = bx; tile < num_tiles; tile += Trait::kNumReduceBlocks) {
    // One elected lane polls
    if (warp_id == 0 && warp::elect_one_lane()) {
      params.done[tile].wait_unique(Trait::SPLIT_K, kPollSleepNanoSecond);
    }
    __syncthreads();

    const auto rows_in_tile = min(M_TILE, params.num_rows - tile * M_TILE);
    const auto base = params.partial + static_cast<uint64_t>(tile) * Trait::SPLIT_K * M_TILE * Trait::kPartialStride;
    for_each<Trait::kBlockSize, Trait::kTileVecs>([&](uint32_t i) {
      Vec4 sum;
      sum.fill(0.f);
#pragma unroll
      for (uint32_t split = 0; split < Trait::SPLIT_K; ++split) {
        Vec4 v;
        v.load(base, split * Trait::kTileVecs + i);
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
          sum[e] += v[e];
        }
      }
      sum.store(smem.reduce.mixes, i);
    });
    __syncthreads();
    if (tx < rows_in_tile) {
      const auto r = tx;
      __align__(16) float row_mixes[N];  // Vec4 stores through a 16B-aligned type
#pragma unroll
      for (uint32_t i = 0; i < N / 4; ++i) {
        Vec4 v;
        v.load(smem.reduce.mixes[r], i);
        v.store(row_mixes, i);
      }
      const auto inv_rms = math::rsqrt(smem.reduce.mixes[r][N] / float(K) + params.rms_eps);
      const auto row = tile * M_TILE + r;
      sinkhorn_row(
          row_mixes,
          inv_rms,
          smem.reduce.scale,
          smem.reduce.base,
          params.hc_eps,
          params.pre + row * 4,
          params.post + row * 4,
          params.comb + row * 16);
    }
    __syncthreads();
    if constexpr (kPublish) {
      // The publishing warp should not be the polling one
      if (warp_id == 1 && warp::elect_one_lane()) {
        // A red.add, not a store: the boundary kernel's comm CTAs consume this with the
        // multi-consumer wait(), which registers itself in the same word with an atom.add.
        params.ready[tile].arrive_async();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// MXFP8 quantization of the all-gathered next input, shared by both transports.
// One lane holds a 16B bf16 vector (8 values); 4 consecutive lanes form one
// 32-value OCP MX block with a shared e8m0 scale.
// ---------------------------------------------------------------------------

struct MXFP8Lane {
  uint32_t values[2];         // this lane's 8 e4m3 bytes, value k in byte k % 4
  uint32_t half_warp_scales;  // the 4 block scales of this lane's half-warp, block b in byte b
};

// OCP MX: shared exponent floor(log2(amax)) - 8 (e4m3 emax), stored as e8m0.
SGL_DEVICE int mxfp8_exponent(float amax) {
  return amax > 0.f ? max(-127, min(127, ilogbf(amax) - 8)) : -127;
}

SGL_DEVICE float mxfp8_unpack_amax(const AlignedVector<bf16x2_t, 4>& v, float (&f)[8]) {
  float amax = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const auto p = cast<fp32x2_t>(v[k]);
    f[2 * k + 0] = p.x;
    f[2 * k + 1] = p.y;
    amax = fmaxf(amax, fmaxf(fabsf(p.x), fabsf(p.y)));
  }
  return amax;
}

SGL_DEVICE void mxfp8_scaled_words(const float (&f)[8], float inv, uint32_t (&out)[2]) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const auto packed =
        cast<fp8x4_e4m3_t>(fp32x4_t{f[4 * h + 0] * inv, f[4 * h + 1] * inv, f[4 * h + 2] * inv, f[4 * h + 3] * inv});
    out[h] = packed.__x;
  }
}

SGL_DEVICE MXFP8Lane quantize_mxfp8(const AlignedVector<bf16x2_t, 4>& out) {
  float f[8];
  const float amax = warp::reduce_max<4, 1>(mxfp8_unpack_amax(out, f));
  const int exponent = mxfp8_exponent(amax);
  MXFP8Lane q;
  mxfp8_scaled_words(f, exp2f(-float(exponent)), q.values);
  const uint32_t scale_byte = uint32_t(exponent + 127) & 0xff;
  q.half_warp_scales = 0;
  // Block b's scale into byte b, for the four blocks of this lane's half-warp: the
  // width-16 shuffle reads lane 4b of the half this lane is in.
#pragma unroll
  for (int b = 0; b < 4; ++b)
    q.half_warp_scales |= __shfl_sync(0xffffffff, scale_byte, 4 * b, 16) << (8 * b);
  return q;
}

// Pair-map variant (kStore256b): a lane holds two adjacent vectors, 16 consecutive
// values, so an adjacent LANE PAIR forms the 32-value block and a warp covers 16 blocks
// -- whose scales leave as one 16B store instead of the stock map's 8B.
struct MXFP8LanePair {
  uint32_t values[4];   // this lane's 16 e4m3 bytes, value k in byte k % 4
  uint32_t seg_scales;  // the 4 block scales of this lane's 8-lane segment, block b in byte b
};

SGL_DEVICE MXFP8LanePair quantize_mxfp8_pair(const AlignedVector<bf16x2_t, 4>& a, const AlignedVector<bf16x2_t, 4>& b) {
  float fa[8], fb[8];
  const float amax = warp::reduce_max<2, 1>(fmaxf(mxfp8_unpack_amax(a, fa), mxfp8_unpack_amax(b, fb)));
  const int exponent = mxfp8_exponent(amax);
  const float inv = exp2f(-float(exponent));
  MXFP8LanePair q;
  mxfp8_scaled_words(fa, inv, reinterpret_cast<uint32_t (&)[2]>(q.values[0]));
  mxfp8_scaled_words(fb, inv, reinterpret_cast<uint32_t (&)[2]>(q.values[2]));
  const uint32_t scale_byte = uint32_t(exponent + 127) & 0xff;
  q.seg_scales = 0;
  // Block b of this lane's 8-lane segment lives on lane pair 2b: width-8 shuffle.
#pragma unroll
  for (int b = 0; b < 4; ++b)
    q.seg_scales |= __shfl_sync(0xffffffff, scale_byte, 2 * b, 8) << (8 * b);
  return q;
}

/// Byte offsets into the MXFP8 output: the e8m0 scale plane sits past the value plane,
/// which is why y and x cannot alias under the quantized all-gather.
SGL_DEVICE uint64_t mxfp8_value_offset(uint64_t global_row, uint32_t vec_index) {
  return global_row * kHidden + uint64_t(vec_index) * 8;
}

SGL_DEVICE uint64_t mxfp8_scale_offset(uint32_t total_rows, uint64_t global_row, uint32_t vec_index) {
  return uint64_t(total_rows) * kHidden + global_row * (kHidden / 32) + (vec_index / 16) * 4;
}

}  // namespace device::mhc

}  // namespace sglang
