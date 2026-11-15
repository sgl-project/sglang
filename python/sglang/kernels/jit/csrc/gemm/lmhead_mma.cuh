// lm_head shard GEMM for few rows of x, mma.sync: out[n, w] = bf16(sum_k x[n, k] * W[w, k]), weight far larger than
// L2. FlashInfer's warp-split-K structure (flashinfer/gemm/kernels/dense_bf16_gemm_warp_splitk.py):
//
// - One CTA per BLOCK_M = 16 weight rows, for all N <= BLOCK_N (8 / 16) rows of x over the whole K, several CTAs per
//   SM: the block scheduler balances the SMs, and no tile needs a cross-CTA reduction.
// - Warps: 2 weight loaders (cp.async 16 B per lane, L2 evict_first; a warp instruction covers 512 B of one row),
//   4 MMA warps, BLOCK_N / 8 x loaders (plain cp.async; the only warps that wait for the PDL primary).
// - Ring of kNumStages stages of BLOCK_K k: the loaders' cp.async complete on the stage's full barrier
//   (`.noinc` arrive per loader thread); the MMA warps split each stage's k range in four, load all their fragments
//   of the stage (ldmatrix), issue the MMAs, then release the stage (empty barrier, one arrive per warp).
//   cp.async and ldmatrix are both generic-proxy accesses, so the release orders the reads before the refill.
// - Epilogue: the four warps' fp32 partials are summed through shared memory in a fixed order (deterministic).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>
#include <sgl_kernel/ptx/mma.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <optional>

namespace sglang {

namespace device::lmhead_mma {

// mma.sync m16n8k16: M = weight rows, N = rows of x.
inline constexpr uint32_t MMA_M = 16;
inline constexpr uint32_t MMA_N = 8;
inline constexpr uint32_t MMA_K = 16;
inline constexpr uint32_t BLOCK_M = 16;   // weight rows per CTA
inline constexpr uint32_t BLOCK_K = 256;  // k per stage: 512 B per row, one row per loader warp instruction
inline constexpr uint32_t kNumWLoadWarps = 2;
inline constexpr uint32_t kNumMMAWarps = 4;

enum Role { WARP_W_LOAD_START = 0, WARP_MMA_START = 2, WARP_X_LOAD_START = 6 };
static_assert(WARP_MMA_START - WARP_W_LOAD_START == kNumWLoadWarps);
static_assert(WARP_X_LOAD_START - WARP_MMA_START == kNumMMAWarps);

enum NamedBarrier { BAR_EPI = 1 };

template <uint32_t BLOCK_N, uint32_t kNumStages, uint32_t SHAPE_K>
struct MMATrait {
  static_assert((BLOCK_N == 8 || BLOCK_N == 16) && SHAPE_K % BLOCK_K == 0);
  static constexpr uint32_t kNumXLoadWarps = BLOCK_N / 8;
  static constexpr uint32_t kNumThreads = (WARP_X_LOAD_START + kNumXLoadWarps) * 32;
  static constexpr uint32_t kChunksPerRow = BLOCK_K / 8;  // 16 B chunks
  static constexpr uint32_t kWStageBytes = BLOCK_M * BLOCK_K * 2;
  static constexpr uint32_t kXStageBytes = BLOCK_N * BLOCK_K * 2;
  static constexpr uint32_t kWCopies = BLOCK_M * kChunksPerRow / (kNumWLoadWarps * 32);
  static constexpr uint32_t kXCopies = BLOCK_N * kChunksPerRow / (kNumXLoadWarps * 32);
  static constexpr uint32_t kNumKBlocks = SHAPE_K / BLOCK_K;
  static constexpr uint32_t kSteps = BLOCK_K / (MMA_K * kNumMMAWarps);  // MMA_K steps per warp and stage
  static constexpr uint32_t kMIters = BLOCK_M / MMA_M;
  static constexpr uint32_t kNIters = BLOCK_N / MMA_N;
  static constexpr uint32_t kRedPitch = BLOCK_M + 4;  // padded against bank conflicts
  static_assert(kWCopies * kNumWLoadWarps * 32 == BLOCK_M * kChunksPerRow);
  static_assert(kXCopies * kNumXLoadWarps * 32 == BLOCK_N * kChunksPerRow);
  static_assert(kSteps % 2 == 0 && kNumStages <= kNumKBlocks);
};

// Tiles are rows of BLOCK_K bf16; 16 B chunk c of row r sits at chunk c ^ (r % 8), so the 8 rows of an ldmatrix matrix
// hit distinct banks. The launcher's byte count and the Python occupancy model both use sizeof(MMASmem).
template <uint32_t BLOCK_N, uint32_t kNumStages, uint32_t SHAPE_K>
struct MMASmem {
  using Trait = MMATrait<BLOCK_N, kNumStages, SHAPE_K>;
  uint8_t w[kNumStages][Trait::kWStageBytes];
  uint8_t x[kNumStages][Trait::kXStageBytes];
  float red[kNumMMAWarps][BLOCK_N][Trait::kRedPitch];  // each MMA warp's fp32 partials, [token][weight row]
  uint64_t full_bar[kNumStages];
  uint64_t empty_bar[kNumStages];
};

// Byte offset of 16 B chunk `chunk` of row `row` in a swizzled tile.
SGL_DEVICE uint32_t tile_off(uint32_t row, uint32_t chunk) {
  return row * (BLOCK_K * 2) + ((chunk ^ (row % 8)) << 4);
}

template <uint32_t BLOCK_N, uint32_t kNumStages, uint32_t SHAPE_K, bool kUsePDL>
__global__ __launch_bounds__(MMATrait<BLOCK_N, kNumStages, SHAPE_K>::kNumThreads) void lmhead_mma_kernel(
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ w,
    bf16_t* __restrict__ out,
    const uint32_t n_rows,
    const uint32_t ld_out) {
  using Trait = MMATrait<BLOCK_N, kNumStages, SHAPE_K>;
  using Smem = MMASmem<BLOCK_N, kNumStages, SHAPE_K>;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  auto& smem = *reinterpret_cast<Smem*>(smem_raw);

  const auto warp_id = warp::reduce_max(threadIdx.x / 32);
  const auto lane_id = threadIdx.x % 32;
  const auto row0 = blockIdx.x * BLOCK_M;
  const auto w_tile = w + static_cast<int64_t>(row0) * SHAPE_K;

  // Weight loader thread t: chunk t + 64 i of the stage's tile.
  const auto load_w = [&](uint32_t kb, std::optional<uint64_t> policy) {
#pragma unroll
    for (uint32_t i = 0; i < Trait::kWCopies; ++i) {
      const auto q = threadIdx.x - WARP_W_LOAD_START * 32 + i * kNumWLoadWarps * 32;
      const auto r = q / Trait::kChunksPerRow, c = q % Trait::kChunksPerRow;
      const auto src = w_tile + r * SHAPE_K + kb * BLOCK_K + c * 8;
      const auto dst = smem.w[kb % kNumStages] + tile_off(r, c);
      if (policy.has_value()) {
        ptx::cp_async_16B_hint(dst, src, *policy);
      } else {
        ptx::cp_async_16B(dst, src);
      }
    }
  };

  if (warp_id < WARP_MMA_START) {
    // Stage 0 waits on no barrier: issue it before the init so its latency overlaps the setup. Without the
    // evict_first hint: measured 0.6-0.9 us faster than with it (GB300, TP8 / TP16 N <= 16; cause not known).
    load_w(0, std::nullopt);
  } else if (warp_id == WARP_MMA_START) {
    if (lane_id < kNumStages) {
      ptx::mbar_init(&smem.full_bar[lane_id], (kNumWLoadWarps + Trait::kNumXLoadWarps) * 32);
      ptx::mbar_init(&smem.empty_bar[lane_id], kNumMMAWarps);
    }
    ptx::fence_barrier_init();
  }
  __syncthreads();

  if (warp_id < WARP_MMA_START) {
    // The weight is read once: evict_first keeps it from displacing x, which every CTA re-reads from L2.
    const auto policy = ptx::createpolicy_evict_first();
    ptx::cp_async_mbarrier_arrive_noinc(&smem.full_bar[0]);
#pragma unroll 1
    for (uint32_t kb = 1; kb < Trait::kNumKBlocks; ++kb) {
      const auto stage = kb % kNumStages;
      if (kb >= kNumStages) ptx::mbar_wait_parity(&smem.empty_bar[stage], (kb / kNumStages - 1) & 1);
      load_w(kb, policy);
      ptx::cp_async_mbarrier_arrive_noinc(&smem.full_bar[stage]);
    }
    // Every weight load of this CTA is issued: let the dependent kernel launch (its griddepcontrol.wait still waits
    // for this grid to complete). Independent of kUsePDL, which only decides whether the x loads wait on the primary.
    PDLTriggerSecondary<true>();
  } else if (warp_id < WARP_X_LOAD_START) {
    const auto mma_warp = warp_id - WARP_MMA_START;
    const auto chunk0 = mma_warp * Trait::kSteps * 2;  // this warp's first 16 B chunk of each tile row
    // Per-lane ldmatrix offsets within a stage's tile.
    uint32_t a_off[Trait::kMIters][Trait::kSteps], b_off[Trait::kNIters][Trait::kSteps / 2];
#pragma unroll
    for (uint32_t m = 0; m < Trait::kMIters; ++m) {
#pragma unroll
      for (uint32_t s = 0; s < Trait::kSteps; ++s) {
        a_off[m][s] = tile_off(m * MMA_M + lane_id % 16, chunk0 + 2 * s + lane_id / 16);
      }
    }
#pragma unroll
    for (uint32_t n = 0; n < Trait::kNIters; ++n) {
#pragma unroll
      for (uint32_t p = 0; p < Trait::kSteps / 2; ++p) {
        b_off[n][p] = tile_off(n * MMA_N + lane_id % 8, chunk0 + 4 * p + lane_id / 8);
      }
    }
    float acc[Trait::kMIters][Trait::kNIters][4] = {};
    const auto smem_w = ptx::to_shared(smem.w[0]);
    const auto smem_x = ptx::to_shared(smem.x[0]);
#pragma unroll
    for (uint32_t kb = 0; kb < Trait::kNumKBlocks; ++kb) {
      const auto stage = kb % kNumStages;
      ptx::mbar_wait_parity(&smem.full_bar[stage], (kb / kNumStages) & 1);
      // Every fragment of the stage is loaded before the first MMA.
      AlignedVector<bf16x2_t, 4> a[Trait::kMIters][Trait::kSteps], b[Trait::kNIters][Trait::kSteps / 2];
#pragma unroll
      for (uint32_t m = 0; m < Trait::kMIters; ++m) {
#pragma unroll
        for (uint32_t s = 0; s < Trait::kSteps; ++s) {
          ptx::ldmatrix_x4(smem_w + stage * Trait::kWStageBytes + a_off[m][s], a[m][s]);
        }
      }
#pragma unroll
      for (uint32_t n = 0; n < Trait::kNIters; ++n) {
#pragma unroll
        for (uint32_t p = 0; p < Trait::kSteps / 2; ++p) {
          ptx::ldmatrix_x4(smem_x + stage * Trait::kXStageBytes + b_off[n][p], b[n][p]);
        }
      }
#pragma unroll
      for (uint32_t s = 0; s < Trait::kSteps; ++s) {
#pragma unroll
        for (uint32_t n = 0; n < Trait::kNIters; ++n) {
          // One ldmatrix_x4 holds the B fragments of two MMA_K steps.
          AlignedVector<bf16x2_t, 2> bs;
          bs[0] = b[n][s / 2][2 * (s % 2)];
          bs[1] = b[n][s / 2][2 * (s % 2) + 1];
#pragma unroll
          for (uint32_t m = 0; m < Trait::kMIters; ++m) {
            ptx::mma_m16n8k16_bf16(acc[m][n], a[m][s], bs);
          }
        }
      }
      if (kb + kNumStages < Trait::kNumKBlocks && warp::elect_one_lane()) ptx::mbar_arrive(&smem.empty_bar[stage]);
    }

    // acc[m][n][j]: weight row m * MMA_M + lane_id / 4 (+ 8 for j >= 2), x row n * MMA_N + 2 (lane_id % 4) (+ 1 for
    // odd j).
#pragma unroll
    for (uint32_t m = 0; m < Trait::kMIters; ++m) {
#pragma unroll
      for (uint32_t n = 0; n < Trait::kNIters; ++n) {
        const auto r = m * MMA_M + lane_id / 4, t = n * MMA_N + 2 * (lane_id % 4);
        smem.red[mma_warp][t][r] = acc[m][n][0];
        smem.red[mma_warp][t + 1][r] = acc[m][n][1];
        smem.red[mma_warp][t][r + 8] = acc[m][n][2];
        smem.red[mma_warp][t + 1][r + 8] = acc[m][n][3];
      }
    }
    ptx::bar_sync(BAR_EPI, kNumMMAWarps * 32);
    // Thread i sums weight rows 2 (i % 8), +1 of x row i / 8 over the four warps.
    const auto i = threadIdx.x - WARP_MMA_START * 32;
    static_assert(BLOCK_M * BLOCK_N / 2 <= kNumMMAWarps * 32);
    if (i < BLOCK_M * BLOCK_N / 2) {
      const auto t = i / (BLOCK_M / 2), r = 2 * (i % (BLOCK_M / 2));
      fp32x2_t sum = {0.0f, 0.0f};
#pragma unroll
      for (uint32_t j = 0; j < kNumMMAWarps; ++j) {
        AlignedVector<float, 2> v;
        v.load(&smem.red[j][t][r]);
        sum.x += v[0];
        sum.y += v[1];
      }
      if (t < n_rows) {
        AlignedVector<bf16x2_t, 1> o;
        o[0] = cast<bf16x2_t>(sum);
        o.store(out + static_cast<int64_t>(t) * ld_out + row0 + r);
      }
    }
  } else {
    PDLWaitPrimary<kUsePDL>();
    const auto t = threadIdx.x - WARP_X_LOAD_START * 32;
#pragma unroll 1
    for (uint32_t kb = 0; kb < Trait::kNumKBlocks; ++kb) {
      const auto stage = kb % kNumStages;
      if (kb >= kNumStages) ptx::mbar_wait_parity(&smem.empty_bar[stage], (kb / kNumStages - 1) & 1);
#pragma unroll
      for (uint32_t i = 0; i < Trait::kXCopies; ++i) {
        const auto q = t + i * Trait::kNumXLoadWarps * 32;
        const auto r = q / Trait::kChunksPerRow, c = q % Trait::kChunksPerRow;
        // Rows past n_rows stay unloaded: they only feed output columns that are never stored.
        if (r < n_rows) {
          ptx::cp_async_16B(
              smem.x[stage] + tile_off(r, c), x + static_cast<int64_t>(r) * SHAPE_K + kb * BLOCK_K + c * 8);
        }
      }
      ptx::cp_async_mbarrier_arrive_noinc(&smem.full_bar[stage]);
    }
  }
}

}  // namespace device::lmhead_mma

template <uint32_t BLOCK_N, uint32_t kNumStages, uint32_t SHAPE_K, bool kUsePDL>
struct LMHeadMMAKernel {
  static constexpr auto kernel = device::lmhead_mma::lmhead_mma_kernel<BLOCK_N, kNumStages, SHAPE_K, kUsePDL>;

  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView out) {
    using namespace host;
    using namespace device::lmhead_mma;
    using Trait = MMATrait<BLOCK_N, kNumStages, SHAPE_K>;
    constexpr uint32_t kSmemBytes = sizeof(MMASmem<BLOCK_N, kNumStages, SHAPE_K>);
    auto N = SymbolicSize{"num_rows"};
    auto W = SymbolicSize{"out_features"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, SHAPE_K})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(x);
    TensorMatcher({W, SHAPE_K})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(w);
    TensorMatcher({N, W})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(4)
        .verify(out);
    const auto n = static_cast<uint32_t>(N.unwrap());
    const auto w_rows = static_cast<uint32_t>(W.unwrap());
    CHECK_HOST(w_rows % BLOCK_M == 0) << "out_features " << w_rows << " must be a multiple of " << BLOCK_M;
    // One CTA covers every row of x, so each weight byte is read from HBM once.
    CHECK_HOST(n <= BLOCK_N) << "num_rows " << n << " > " << BLOCK_N;
    if (n == 0) return;
    runtime::set_max_dynamic_smem_per_device<kernel>(device_.unwrap(), kSmemBytes);
    LaunchKernel(w_rows / BLOCK_M, Trait::kNumThreads, device_.unwrap(), kSmemBytes)
        .config({.use_pdl = kUsePDL})(
            kernel,
            static_cast<const bf16_t*>(x.data_ptr()),
            static_cast<const bf16_t*>(w.data_ptr()),
            static_cast<bf16_t*>(out.data_ptr()),
            n,
            w_rows);
  }
};

}  // namespace sglang
