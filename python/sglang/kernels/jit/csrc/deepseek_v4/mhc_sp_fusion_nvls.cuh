// MHC sequence-parallel sublayer boundary over NVLink multicast (NVLS), fused with its
// communication. The peer-to-peer variant lives in mhc_sp_fusion_p2p.cuh.
//
// Every TP rank holds a partial sublayer output y_r [T_pad, kHidden] (bf16) in symmetric
// memory and owns M = T_pad / world rows of the hc residual. One kernel does
//
//   RS:   y[row] = sum_r y_r[row]          (row in this rank's slice, multimem.ld_reduce)
//   post: R_new = post * y + comb^T R_old  (local rows)
//   norm: x = RMSNorm(pre . R_new)         (local rows)
//   AG:   x -> every rank's x buffer       (multimem.st)
//
// A row is reduced by exactly one rank, and that rank reads every copy of it before
// writing its own result back over them, so no rank ever reads a row another has
// overwritten: with a bf16 all-gather the caller may pass one buffer as both y and x
// and halve the symmetric memory. The MXFP8 gather writes a different layout, so that
// aliasing does not hold there.
//
// With kStats the remaining CTAs also compute this boundary's own coefficients from
// R_old on tensor cores and publish them per tile; the comm CTAs wait on that before
// they consume a row. R_old is in memory before the launch, so the statistics start
// immediately and run under the reduce-scatter.
//
// A group of kGroupThreads owns one row at a time. Rows are assigned statically: on
// NVSwitch every group issues the same instructions for every row, so work stealing
// would only cost an atomic and a barrier per row.
#pragma once
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/atomic.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/dsv41/mhc.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace mhc_nvls {

using namespace device::mhc;

// `using namespace device` in the functions would make a bare `ptx` ambiguous with
// device::ptx; resolve it to the mhc toolbox, which re-exports the shared helpers.
namespace ptx = device::mhc::ptx;

// One row spread over a group of threads: kRowVecs 16B vectors, kVecsPerThread each.
inline constexpr uint32_t kGroupThreads = 128;
inline constexpr uint32_t kRowVecs = kHidden / 8;
inline constexpr uint32_t kVecsPerThread = kRowVecs / kGroupThreads;  // 5
inline constexpr uint32_t kWarpsPerGroup = kGroupThreads / device::kWarpThreads;
static_assert(kRowVecs % kGroupThreads == 0);

using vec_t = device::AlignedVector<bf16x2_t, 4>;

inline constexpr uint32_t kNormWeightBytes = kHidden * sizeof(bf16_t);

// The comm role's shared. `warp_sums` is double buffered so a row's read cannot race the
// next row's write, which saves the second group barrier a single buffer would need.
template <uint32_t kRowsPerCTA>
struct CommSmem {
  __align__(128) bf16_t norm_weight[kHidden];
  uint64_t weight_arrived;
  float warp_sums[2][kRowsPerCTA][kWarpsPerGroup];
};

// The two roles never share a CTA, so they share the allocation: the comm role's 10 KB
// costs nothing next to the statistics working set.
template <uint32_t kRowsPerCTA, typename StatTrait>
union BoundarySmem {
  CommSmem<kRowsPerCTA> comm;
  MHCStatSmem<StatTrait> stats;
};

struct Params {
  // ---- communication ----
  const uint8_t* y_mc;                           // multicast VA of the partial output [T_pad, kHidden] bf16
  uint8_t* x_mc;                                 // multicast VA of the next input [T_pad, kHidden] bf16
  device::distributed::Semaphore* semaphore;     // this rank's semaphore
  device::distributed::Semaphore* semaphore_mc;  // its multicast VA
  const bf16_t* residual;                        // [M, kNumStreams, kHidden] local rows
  bf16_t* residual_out;                          // [M, kNumStreams, kHidden]
  const float *post, *comb, *pre;                // [M, 4], [M, 4, 4], [M, 4]
  const bf16_t* norm_weight;                     // [kHidden]
  uint32_t num_rows;                             // M
  uint32_t row_offset;                           // rank * M
  uint32_t total_rows;                           // rows of the whole batch, this rank's included
  float eps;
  uint32_t num_comm_blocks;  // blocks [0, this) run the comm role
  // ---- this boundary's own statistics, when kStats ----
  // Its pre / post / comb alias the comm role's, which is the point: the statistics CTAs
  // produce exactly what the comm CTAs of this same launch consume.
  MHCStatParams stats;
};

SGL_DEVICE void load_partial_sum(vec_t (&y)[kVecsPerThread], const Params& params, uint64_t global_row, uint32_t slot) {
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j)
    device::ptx::ld_multimem_16B(y[j], params.y_mc, global_row * kRowVecs + slot + j * kGroupThreads);
}

// MXFP8 all-gather: quantize this lane's vector (sgl_kernel/dsv41/mhc.cuh) and push
// values + scales through the multicast address.
SGL_DEVICE void store_mxfp8(const Params& params, const vec_t& out, uint64_t global_row, uint32_t vec_index) {
  using namespace device;
  const auto q = mhc::quantize_mxfp8(out);
  const uint64_t value_offset = mhc::mxfp8_value_offset(global_row, vec_index);
  const uint64_t scale_offset = mhc::mxfp8_scale_offset(params.total_rows, global_row, vec_index);
  asm volatile("multimem.st.weak.global.v2.f32 [%0], {%1, %2};" ::"l"(params.x_mc + value_offset),
               "f"(__uint_as_float(q.values[0])),
               "f"(__uint_as_float(q.values[1]))
               : "memory");
  if (threadIdx.x % kWarpThreads % 16 == 0)
    asm volatile("multimem.st.weak.global.b32 [%0], %1;" ::"l"(params.x_mc + scale_offset), "r"(q.half_warp_scales)
                 : "memory");
}

// ---------------------------------------------------------------------------
// Comm role: RS + post / combine / norm + AG for this rank's rows.
// ---------------------------------------------------------------------------
template <uint32_t kWorldSize, uint32_t kRowsPerCTA, bool kStats, bool kFP8AG, bool kEpilogue>
SGL_DEVICE void run_comm(const Params& params, uint32_t comm_bx, uint32_t num_blocks, CommSmem<kRowsPerCTA>& smem) {
  using namespace device;
  // Barrier 0 is __syncthreads(), so group g takes g + 1 and the hardware has 16.
  static_assert(kRowsPerCTA + 1 <= 16, "one named barrier per group");
  const auto tx = threadIdx.x;
  const auto group = tx / kGroupThreads;
  const auto slot = tx % kGroupThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto warp_id_in_group = slot / kWarpThreads;

  // The RMSNorm weight is uniform across rows, so fetch it once rather than leaning on it
  // staying in L1 for every row's epilogue. Warp 1, so the fetch and its wait run
  // alongside thread 0 polling the entry barrier.
  if (kEpilogue && tx / kWarpThreads == 1 && warp::elect_one_lane()) {
    ptx::mbar_init(&smem.weight_arrived, 1);
    ptx::mbar_arrive_expect_tx(&smem.weight_arrived, kNormWeightBytes);
    ptx::cp_async_bulk_g2s(smem.norm_weight, params.norm_weight, kNormWeightBytes, &smem.weight_arrived);
    ptx::mbar_wait_parity(&smem.weight_arrived, 0);
  }
  // Entry barrier: every rank has finished writing its partial y. Relaxed is enough --
  // the launch that wrote y already ordered those stores, so this only counts arrivals.
  // The semaphore plane holds one slot per block of the all-reduce that owns it, and only
  // the comm role barriers here, so index it by that role rather than by blockIdx.x --
  // the statistics blocks sit below the comm blocks and would push the slot past capacity.
  const distributed::McBarrier barrier{params.semaphore, params.semaphore_mc, kWorldSize, 2, comm_bx};
  barrier.arrive_relaxed(0);
  // Every rank is in, and the weight is in shared.
  __syncthreads();

  const auto normalize = [&](const vec_t(&collapsed)[kVecsPerThread], float inv_rms, uint64_t global_row) {
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const uint32_t vec_index = slot + j * kGroupThreads;
      vec_t weight, out;
      weight.load(smem.norm_weight, vec_index);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const auto v = cast<fp32x2_t>(collapsed[j][k]);
        const auto w = cast<fp32x2_t>(weight[k]);
        out[k] = cast<bf16x2_t>(fp32x2_t{v.x * inv_rms * w.x, v.y * inv_rms * w.y});
      }
      if constexpr (kFP8AG) {
        store_mxfp8(params, out, global_row, vec_index);
      } else {
        ptx::st_multimem_16B(out, params.x_mc, global_row * kRowVecs + vec_index);
      }
    }
  };

  uint32_t parity = 0;
  for (uint32_t row = comm_bx * kRowsPerCTA + group; row < params.num_rows; row += num_blocks * kRowsPerCTA) {
    const uint64_t global_row = params.row_offset + row;
    vec_t y[kVecsPerThread];
    load_partial_sum(y, params, global_row, slot);

    if constexpr (kStats) {
      if (slot == 0) {
        const auto tile = row / M_TILE;
        const auto rows_in_tile = min(M_TILE, params.num_rows - tile * M_TILE);
        params.stats.ready[tile].wait(1, rows_in_tile, 1, kPollSleepNanoSecond);
      }
      ptx::bar_sync(group + 1, kGroupThreads);
    }

    const auto coeff = load_row_coeff(params.pre + row * 4, params.post + row * 4, params.comb + row * 16, lane_id);
    float post[kNumStreams], pre[kNumStreams], comb[kNumStreams][kNumStreams];
#pragma unroll
    for (uint32_t to = 0; to < kNumStreams; ++to) {
      post[to] = coeff_post(coeff, to);
      pre[to] = coeff_pre(coeff, to);
#pragma unroll
      for (uint32_t from = 0; from < kNumStreams; ++from) {
        comb[from][to] = coeff_comb(coeff, from, to);
      }
    }

    const bf16_t* old_row = params.residual + static_cast<uint64_t>(row) * K;
    bf16_t* new_row = params.residual_out + static_cast<uint64_t>(row) * K;
    vec_t collapsed[kVecsPerThread];
    float sum_squares = 0.f;
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const auto vec_index = slot + j * kGroupThreads;
      vec_t old_streams[kNumStreams];
#pragma unroll
      for (uint32_t from = 0; from < kNumStreams; ++from)
        old_streams[from].load(old_row + from * kHidden, vec_index);
      fp32x2_t collapse_acc[4] = {};
#pragma unroll
      for (uint32_t to = 0; to < kNumStreams; ++to) {
        vec_t updated;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const auto contribution = cast<fp32x2_t>(y[j][k]);
          auto acc = fp32x2_t{post[to] * contribution.x, post[to] * contribution.y};
#pragma unroll
          for (uint32_t from = 0; from < kNumStreams; ++from) {
            const auto old = cast<fp32x2_t>(old_streams[from][k]);
            acc = math::fma_f32x2(fp32x2_t{comb[from][to], comb[from][to]}, old, acc);
          }
          updated[k] = cast<bf16x2_t>(acc);
          if constexpr (kEpilogue) {
            // The next combine consumes the rounded residual, as the reference does.
            const auto rounded = cast<fp32x2_t>(updated[k]);
            collapse_acc[k] = math::fma_f32x2(fp32x2_t{pre[to], pre[to]}, rounded, collapse_acc[k]);
          }
        }
        updated.store(new_row + to * kHidden, vec_index);
      }
      if constexpr (kEpilogue) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          collapsed[j][k] = cast<bf16x2_t>(collapse_acc[k]);
          const auto v = cast<fp32x2_t>(collapsed[j][k]);
          sum_squares = fmaf(v.x, v.x, sum_squares);
          sum_squares = fmaf(v.y, v.y, sum_squares);
        }
      }
    }

    if constexpr (kEpilogue) {
      // broadcast write same value to same smem dst, ok
      smem.warp_sums[parity][group][warp_id_in_group] = warp::reduce_sum(sum_squares);
      ptx::bar_sync(group + 1, kGroupThreads);
      float row_total = 0.f;
#pragma unroll
      for (uint32_t w = 0; w < kWarpsPerGroup; ++w)
        row_total += smem.warp_sums[parity][group][w];
      parity ^= 1;
      normalize(collapsed, math::rsqrt(row_total / float(kHidden) + params.eps), global_row);
    }
  }

  __syncthreads();
  barrier.arrive_rel_acq(1);
}

}  // namespace mhc_nvls

extern __shared__ __align__(16) char smem_base[];

template <uint32_t kWorldSize, uint32_t kRowsPerCTA, bool kStats, typename StatTrait, bool kFP8AG, bool kEpilogue>
__global__ __launch_bounds__(mhc_nvls::kGroupThreads* kRowsPerCTA, 1) void mhc_sp_fusion_nvls_kernel(
    const __grid_constant__ mhc_nvls::Params params) {
  auto& smem = *reinterpret_cast<mhc_nvls::BoundarySmem<kRowsPerCTA, StatTrait>*>(smem_base);
  if constexpr (!kStats) {
    mhc_nvls::run_comm<kWorldSize, kRowsPerCTA, false, kFP8AG, kEpilogue>(params, blockIdx.x, gridDim.x, smem.comm);
  } else {
    static_assert(mhc_nvls::kGroupThreads * kRowsPerCTA == StatTrait::kBlockSize, "the roles share a block shape");
    // Roles are laid out in dependency order: mma, then the reduction that consumes it,
    // then the communication that consumes the published tiles. Blocks are dispatched by
    // increasing blockIdx, so a grid that cannot be seated all at once still makes
    // progress -- a producer is never left behind a consumer that is waiting for it.
    const auto bx = blockIdx.x;
    if (bx < StatTrait::kNumMMABlocks) {
      device::mhc::run_mma<StatTrait>(params.stats, bx, smem.stats);
    } else if (bx < StatTrait::kNumBlocks) {
      device::mhc::run_reduction<StatTrait, true>(params.stats, bx - StatTrait::kNumMMABlocks, smem.stats);
    } else {
      mhc_nvls::run_comm<kWorldSize, kRowsPerCTA, true, kFP8AG, kEpilogue>(
          params, bx - StatTrait::kNumBlocks, params.num_comm_blocks, smem.comm);
    }
  }
}

template <
    uint32_t kWorldSize,
    uint32_t kRowsPerCTA,
    bool kStats,
    uint32_t kNumWeightParts,
    uint32_t SPLIT_K,
    uint32_t kNumStatMMABlocks,
    uint32_t kNumStatReduceBlocks,
    bool kFP8AG,
    bool kEpilogue>
struct MHCSPFusionNVLS {
  using StatTrait = device::mhc::MHCStatTrait<SPLIT_K, kNumWeightParts, kNumStatMMABlocks, kNumStatReduceBlocks>;

  static constexpr auto kernel =
      mhc_sp_fusion_nvls_kernel<kWorldSize, kRowsPerCTA, kStats, StatTrait, kFP8AG, kEpilogue>;
  static constexpr uint32_t kBlockSize = mhc_nvls::kGroupThreads * kRowsPerCTA;
  static constexpr size_t kSmemBytes =
      kStats ? sizeof(mhc_nvls::BoundarySmem<kRowsPerCTA, StatTrait>) : sizeof(mhc_nvls::CommSmem<kRowsPerCTA>);
  static_assert(kSmemBytes <= device::mhc::kSmemCliffBytes, "past the shared cliff; raise SPLIT_K");
  static_assert(sizeof(device::atomic::Event) == sizeof(uint32_t), "counters is one int32 per Event");
  static_assert(kEpilogue || !kFP8AG, "the MXFP8 all-gather is epilogue output");

  static void
  run(const host::distributed::CommunicatorRef communicator,  // barriered on; its buffers are untouched
      const int64_t y_mc,                                     // multicast VA of the caller's symmetric y
      const int64_t x_mc,                                     // and of its symmetric x
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView residual_out,
      const tvm::ffi::TensorView post,
      const tvm::ffi::TensorView comb,
      const tvm::ffi::TensorView pre,
      const tvm::ffi::Optional<tvm::ffi::TensorView> weight,  // RMSNorm weight; folded only under kEpilogue
      const int64_t row_offset,
      const int64_t total_rows,
      const float eps,
      const int64_t num_blocks,
      // statistics (ignored unless kStats)
      const tvm::ffi::Optional<tvm::ffi::TensorView> hc_w,
      const tvm::ffi::Optional<tvm::ffi::TensorView> hc_scale,
      const tvm::ffi::Optional<tvm::ffi::TensorView> hc_base,
      const tvm::ffi::Optional<tvm::ffi::TensorView> workspace,  // fp32 partial mixes
      const tvm::ffi::Optional<tvm::ffi::TensorView> counters,   // int32 [2 * tiles], zeroed once
      const float rms_eps,
      const float hc_eps) {
    using namespace host;
    using namespace device::mhc;

    constexpr int64_t S = kNumStreams;
    constexpr int64_t D = kHidden;
    auto M = SymbolicSize{"num_rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, S, D}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(residual_out);
    TensorMatcher({M, S}).with_dtype<float>().with_device(dev).verify(post).verify(pre);
    TensorMatcher({M, S, S}).with_dtype<float>().with_device(dev).verify(comb);
    if constexpr (kEpilogue) {
      TensorMatcher({D}).with_dtype<bf16_t>().with_device(dev).verify(weight.value());
    }
    if constexpr (kStats) {
      TensorMatcher({kNumWeightParts, N, K}).with_dtype<bf16_t>().with_device(dev).verify(hc_w.value());
      TensorMatcher({3}).with_dtype<float>().with_device(dev).verify(hc_scale.value());
      TensorMatcher({N}).with_dtype<float>().with_device(dev).verify(hc_base.value());
    }

    // The semaphores are the all-reduce pull plane's: this kernel only barriers on them
    // and reduces in place on the caller's own symmetric y / x.
    const auto& pull = communicator.get()->get_pull_obj();
    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    const auto num_comm_blocks = static_cast<uint32_t>(num_blocks);
    const auto num_tiles = div_ceil(num_rows, M_TILE);
    const auto split = row_split(static_cast<uint32_t>(total_rows), pull.rank, kWorldSize);

    CHECK_HOST(pull.world_size == kWorldSize);
    CHECK_HOST(num_comm_blocks <= pull.num_blocks);
    CHECK_HOST(pull.mc_semaphore && y_mc && (x_mc || !kEpilogue))
        << "NVLS needs a multicast VA for the semaphores, y and x";
    CHECK_HOST(static_cast<uint32_t>(row_offset) == split.offset && num_rows == split.count)
        << "the caller's row split disagrees: rank " << pull.rank << " passed [" << row_offset << ", +" << num_rows
        << ") of " << total_rows << ", this kernel owns [" << split.offset << ", +" << split.count << ")";
    if constexpr (kStats) {
      CHECK_HOST(workspace.value().numel() >= int64_t(num_tiles) * SPLIT_K * M_TILE * StatTrait::kPartialStride);
      CHECK_HOST(counters.value().numel() >= int64_t(2 * num_tiles));
    }

    const auto residual_ptr = static_cast<const bf16_t*>(residual.data_ptr());
    const auto post_ptr = static_cast<float*>(post.data_ptr());
    const auto comb_ptr = static_cast<float*>(comb.data_ptr());
    const auto pre_ptr = static_cast<float*>(pre.data_ptr());
    const auto stat_params = [&]() -> MHCStatParams {
      if constexpr (!kStats) {
        return {};
      } else {
        auto* const counter_base = static_cast<device::atomic::Event*>(counters.value().data_ptr());
        return {
            .residual = residual_ptr,
            .weight = static_cast<const bf16_t*>(hc_w.value().data_ptr()),
            .scale = static_cast<const float*>(hc_scale.value().data_ptr()),
            .base = static_cast<const float*>(hc_base.value().data_ptr()),
            .pre = pre_ptr,
            .post = post_ptr,
            .comb = comb_ptr,
            .partial = static_cast<float*>(workspace.value().data_ptr()),
            .done = counter_base,
            .ready = counter_base + num_tiles,
            .num_rows = num_rows,
            .rms_eps = rms_eps,
            .hc_eps = hc_eps,
        };
      }
    }();
    const auto params = mhc_nvls::Params{
        .y_mc = reinterpret_cast<const uint8_t*>(y_mc),
        .x_mc = reinterpret_cast<uint8_t*>(x_mc),
        .semaphore = pull.semaphores[pull.rank],
        .semaphore_mc = pull.mc_semaphore,
        .residual = residual_ptr,
        .residual_out = static_cast<bf16_t*>(residual_out.data_ptr()),
        .post = post_ptr,
        .comb = comb_ptr,
        .pre = pre_ptr,
        .norm_weight = kEpilogue ? static_cast<const bf16_t*>(weight.value().data_ptr()) : nullptr,
        .num_rows = num_rows,
        .row_offset = static_cast<uint32_t>(row_offset),
        .total_rows = static_cast<uint32_t>(total_rows),
        .eps = eps,
        .num_comm_blocks = num_comm_blocks,
        .stats = stat_params,
    };

    CHECK_CUDA(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    const auto device = dev.unwrap();
    const auto grid = num_comm_blocks + (kStats ? StatTrait::kNumBlocks : 0u);
    const auto num_sm = runtime::get_sm_count(device.device_id);
    CHECK_HOST(grid <= num_sm) << "the grid must be co-resident: " << grid << " blocks, " << num_sm << " SM";
    LaunchKernel(grid, kBlockSize, dev.unwrap(), kSmemBytes).launch(kernel, params);
  }
};

}  // namespace sglang
