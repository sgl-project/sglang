// lm_head shard GEMM, UMMA (tcgen05.mma), one wave: out[n, w] = bf16(sum_k x[n, k] * W[w, k]), N <= 128 rows of x,
// weight far larger than L2. cuBLAS's structure for this shape on GB300, without its cluster multicast of x:
//
// - The weight is the UMMA N operand (B): a CTA's tile is BLOCK_N weight rows (a multiple of 8 / 16), chosen on the
//   host so that ceil(W / BLOCK_N) tiles fill one wave (TP8 144 rows: 144 CTAs; TP16 72 rows: 146 CTAs). One whole
//   tile (BLOCK_N rows x the full K) per CTA: no stream-K, no split-K, no tile loop.
// - x is the UMMA M operand (A): BLOCK_M = 64 or 128 rows (rows >= N are zero-filled by TMA); the fp32 accumulator is
//   BLOCK_N TMEM columns (lane = x row, column = weight row).
// - Ring of kNumStages stages of BLOCK_K = 64 (one 128 B swizzle row): full barrier (TMA bytes), empty barrier (MMA
//   commit); one commit to the accumulator barrier after the last stage.
// - Epilogue: each thread holds one x row (M = 64, layout F: x row 16 q + i at TMEM lane 32 q + i): kChunk = 16 / 8
//   consecutive weight rows per tcgen05.ld, packed to bf16 and stored contiguously in out[n, :] (32 / 16 B).
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
#include <sgl_kernel/ptx/tcgen05.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cuda.h>

namespace sglang {

namespace device::lmhead_umma {

using tcgen05 = ptx::TCGen05<1>;

inline constexpr uint32_t BLOCK_K = 64;  // k per stage: one 128 B swizzle row
inline constexpr uint32_t UMMA_K = 16;

// Epilogue warp w reads TMEM lane quadrant w % 4.
enum Role { WARP_TMA = 0, WARP_MMA = 1, WARP_EPI_START = 2 };
inline constexpr uint32_t kNumThreads = (WARP_EPI_START + 4) * 32;

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumStages>
struct MMATrait {
  static_assert((BLOCK_M == 64 || BLOCK_M == 128) && BLOCK_N % (BLOCK_M == 64 ? 8 : 16) == 0 && BLOCK_N <= 256);
  static constexpr uint32_t kChunk = BLOCK_N % 16 == 0 ? 16 : 8;  // accumulator columns per tcgen05.ld and store
  static constexpr uint32_t kWTileBytes = BLOCK_N * BLOCK_K * 2;
  static constexpr uint32_t kXTileBytes = BLOCK_M * BLOCK_K * 2;
  static constexpr uint32_t kTmemCols = std::max(32u, std::bit_ceil(BLOCK_N));
  // SW128 K-major tiles start on 1024 B boundaries.
  static_assert(kWTileBytes % 1024 == 0 && kXTileBytes % 1024 == 0);
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumStages>
struct MMASmem {
  using Trait = MMATrait<BLOCK_M, BLOCK_N, kNumStages>;
  uint8_t w[kNumStages][Trait::kWTileBytes];
  uint8_t x[kNumStages][Trait::kXTileBytes];
  uint64_t full_bar[kNumStages];
  uint64_t empty_bar[kNumStages];
  uint64_t acc_full;
  uint32_t tmem_base;
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumStages, bool kUsePDL>
__global__ void __launch_bounds__(kNumThreads, 1) lmhead_umma_kernel(
    const __grid_constant__ CUtensorMap map_w,
    const __grid_constant__ CUtensorMap map_x,
    bf16_t* __restrict__ out,
    const uint32_t n_rows,
    const uint32_t w_rows,
    const uint32_t num_k_blocks) {
  using Trait = MMATrait<BLOCK_M, BLOCK_N, kNumStages>;
  using Smem = MMASmem<BLOCK_M, BLOCK_N, kNumStages>;
  constexpr uint32_t kChunk = Trait::kChunk;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  auto& smem = *reinterpret_cast<Smem*>(smem_raw);

  const auto warp_id = warp::reduce_max(threadIdx.x / 32);
  const auto lane_id = threadIdx.x % 32;
  const auto row0 = blockIdx.x * BLOCK_N;

  if (warp_id == WARP_TMA) {
    if (warp::elect_one_lane()) {
      ptx::prefetch_tensormap(&map_w);
      ptx::prefetch_tensormap(&map_x);
      for (uint32_t s = 0; s < kNumStages; ++s) {
        ptx::mbar_init(&smem.full_bar[s], 1);
        ptx::mbar_init(&smem.empty_bar[s], 1);
      }
      ptx::mbar_init(&smem.acc_full, 1);
      ptx::fence_barrier_init();
    }
  } else if (warp_id == WARP_MMA) {
    tcgen05::alloc(&smem.tmem_base, Trait::kTmemCols);
    tcgen05::fence_before_thread_sync();
  }
  __syncthreads();

  if (warp_id == WARP_TMA) {
    if (warp::elect_one_lane()) {
      PDLWaitPrimary<kUsePDL>();
      // The weight is read once: evict_first keeps it from displacing x, which every CTA re-reads from L2 (measured
      // at TP8 N = 128: 47.4 vs 49.2 us without; no effect at N <= 32).
      const auto policy = ptx::createpolicy_evict_first();
      uint32_t stage = 0, phase = 0;
      for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        ptx::mbar_wait_parity(&smem.empty_bar[stage], phase ^ 1);
        ptx::mbar_arrive_expect_tx(&smem.full_bar[stage], Trait::kWTileBytes + Trait::kXTileBytes);
        const auto k = static_cast<int32_t>(kb * BLOCK_K);
        ptx::cp_async_bulk_tensor_2d_hint(
            smem.w[stage], &map_w, k, static_cast<int32_t>(row0), &smem.full_bar[stage], policy);
        ptx::cp_async_bulk_tensor_2d(smem.x[stage], &map_x, k, 0, &smem.full_bar[stage]);
        if (++stage == kNumStages) {
          stage = 0;
          phase ^= 1;
        }
      }
      // Every load of this CTA is issued: let the dependent kernel launch (its griddepcontrol.wait still waits for
      // this grid to complete). Independent of kUsePDL, which only decides whether this kernel waits on its primary.
      PDLTriggerSecondary<true>();
    }
  } else if (warp_id == WARP_MMA) {
    tcgen05::fence_after_thread_sync();
    const auto tmem_base = smem.tmem_base;
    if (warp::elect_one_lane()) {
      const auto idesc = tcgen05::make_instr_desc_bf16<BLOCK_M, BLOCK_N>().bits;
      const auto desc_w = tcgen05::make_desc_kmajor_sw128(smem.w[0]);
      const auto desc_x = tcgen05::make_desc_kmajor_sw128(smem.x[0]);
      uint32_t stage = 0, phase = 0;
      for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        ptx::mbar_wait_parity(&smem.full_bar[stage], phase);
        tcgen05::fence_after_thread_sync();
#pragma unroll
        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
          const auto dw = desc_w.advance_bytes(stage * Trait::kWTileBytes + k * UMMA_K * 2);
          const auto dx = desc_x.advance_bytes(stage * Trait::kXTileBytes + k * UMMA_K * 2);
          tcgen05::mma_f16_ss(tmem_base, dx.bits, dw.bits, idesc, (kb | k) != 0 ? 1u : 0u);
        }
        tcgen05::commit(&smem.empty_bar[stage]);
        if (++stage == kNumStages) {
          stage = 0;
          phase ^= 1;
        }
      }
      tcgen05::commit(&smem.acc_full);
    }
  } else {
    // Thread = one x row (TMEM lane), kChunk weight rows per load.
    constexpr uint32_t kQuadRows = BLOCK_M / 4;
    const auto quad = warp_id % 4;
    if (quad * kQuadRows < n_rows) {  // warp-uniform: some row of this warp exists
      ptx::mbar_wait_parity(&smem.acc_full, 0);
      tcgen05::fence_after_thread_sync();
      float acc[BLOCK_N];
#pragma unroll
      for (uint32_t c = 0; c < BLOCK_N / kChunk; ++c) {
        tcgen05::ld_32x32b<kChunk>(smem.tmem_base + ((quad * 32) << 16) + c * kChunk, acc + c * kChunk);
      }
      // The stores' register dependency waits for the loads; no tcgen05.wait::ld needed for that.
      const auto n = quad * kQuadRows + lane_id;
      if (lane_id < kQuadRows && n < n_rows) {
        const auto dst = out + static_cast<int64_t>(n) * w_rows + row0;
#pragma unroll
        for (uint32_t c = 0; c < BLOCK_N / kChunk; ++c) {
          AlignedVector<bf16x2_t, kChunk / 2> v;
#pragma unroll
          for (uint32_t i = 0; i < kChunk / 2; ++i) {
            v[i] = cast<bf16x2_t>(fp32x2_t{acc[c * kChunk + 2 * i], acc[c * kChunk + 2 * i + 1]});
          }
          if (row0 + c * kChunk < w_rows) v.store(dst + c * kChunk);
        }
      }
      // Every TMEM read completes before the barrier after which the MMA warp deallocates.
      tcgen05::wait_ld();
      tcgen05::fence_before_thread_sync();
    }
  }

  __syncthreads();
  if (warp_id == WARP_MMA) {
    tcgen05::fence_after_thread_sync();
    tcgen05::dealloc(smem.tmem_base, Trait::kTmemCols);
  }
}

}  // namespace device::lmhead_umma

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumStages, bool kUsePDL>
struct LMHeadUMMAKernel {
  static constexpr auto kernel = device::lmhead_umma::lmhead_umma_kernel<BLOCK_M, BLOCK_N, kNumStages, kUsePDL>;

  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView out) {
    using namespace host;
    using namespace device::lmhead_umma;
    using Trait = MMATrait<BLOCK_M, BLOCK_N, kNumStages>;
    constexpr uint32_t kSmemBytes = sizeof(MMASmem<BLOCK_M, BLOCK_N, kNumStages>);
    auto N = SymbolicSize{"num_rows"};
    auto K = SymbolicSize{"hidden"};
    auto W = SymbolicSize{"out_features"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, K})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(x);
    TensorMatcher({W, K})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(w);
    TensorMatcher({N, W})  //
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(out);
    const auto n = static_cast<uint32_t>(N.unwrap());
    const auto k = static_cast<uint64_t>(K.unwrap());
    const auto w_rows = static_cast<uint32_t>(W.unwrap());
    CHECK_HOST(n <= BLOCK_M) << "num_rows " << n << " > " << BLOCK_M;
    CHECK_HOST(k % BLOCK_K == 0 && w_rows % Trait::kChunk == 0)
        << "hidden " << k << " % " << BLOCK_K << ", out_features " << w_rows << " % " << Trait::kChunk;
    if (n == 0) return;
    // K-major bf16 rows, a box of BLOCK_K k x `box_rows` rows, 128 B swizzle.
    const auto make_map = [k](const void* base, uint64_t rows, uint32_t box_rows) {
      return tma::make_map(
          base,
          CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          {k, rows},
          {k * sizeof(bf16_t)},
          {BLOCK_K, box_rows},
          {.swizzle = CU_TENSOR_MAP_SWIZZLE_128B});
    };
    const auto map_w = make_map(w.data_ptr(), w_rows, BLOCK_N);
    const auto map_x = make_map(x.data_ptr(), n, BLOCK_M);
    runtime::set_max_dynamic_smem_per_device<kernel>(device_.unwrap(), kSmemBytes);
    LaunchKernel(div_ceil(w_rows, BLOCK_N), kNumThreads, device_.unwrap(), kSmemBytes)
        .config({.use_pdl = kUsePDL})(
            kernel, map_w, map_x, static_cast<bf16_t*>(out.data_ptr()), n, w_rows, static_cast<uint32_t>(k / BLOCK_K));
  }
};

}  // namespace sglang
