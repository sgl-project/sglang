// Fused bf16 2shot push all-reduce + Muse Spark attention boundary (ROCm).
//
//   branch     = all_reduce(input)                                  (bf16)
//   normed     = bf16(rmsnorm(branch))                              (post-norm, no weight)
//   new_stream = alpha * stream + beta * normed                     (fp32, per-channel gate)
//   output     = bf16(rmsnorm(new_stream) * pre_weight)             (pre-norm)
//
// The rounding points are those of `_fused_norm_residual_triton_kernel` in
// rocm_muse_spark/fused_norm_residual.py for this form. The branch is the bf16
// sum `reduce_vec` gives the plain push all-reduce, so it matches that path
// bit for bit; the two row reductions run in a different order than Triton's.
//
// The exchange is the 2shot_push of `custom_all_reduce.cuh` with ownership by
// whole rows (rank c owns rows [c * R, (c + 1) * R)), on the same push plane,
// epoch counter and slot conventions, so it interleaves freely with the plain
// push all-reduce. The owner fans the bf16 sum out to every rank, itself
// included, and one block per row (block b takes rows b, b + grid, ...) drains
// it from local memory and runs the boundary on it.
//
// From `kTwoShotPushUnrolledMinBytes` the plain all-reduce runs the unrolled
// 2shot_push (two_shot_push_unrolled.cuh), so the unrolled kernel below takes its
// reduce-scatter and reduce phases (per-peer warp groups, whole hunks, batched
// system-scope stores, late-peer re-poll with a sleep) and keeps the per-row drain.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include "../distributed/two_shot_push_unrolled.cuh"

#include <tvm/ffi/object.h>

#include <cstdint>

namespace sglang {

using device::distributed::PushWorkSpace;
using host::distributed::CommunicatorRef;

template <uint32_t kWorldSize>
struct MuseArBoundaryParams {
  const void* __restrict__ input;
  const void* __restrict__ stream;
  const void* __restrict__ pre_weight;
  const void* __restrict__ alpha;
  const void* __restrict__ beta;
  void* __restrict__ output;
  void* __restrict__ new_stream;
  uint32_t num_rows;
  uint32_t row_vecs;
  uint32_t rows_per_rank;
  uint32_t rank;
  float post_eps;
  float pre_eps;
  float inv_hidden;
  /// Factored kernel only: 128-vector segments per row.
  uint32_t segs;
  PushWorkSpace<kWorldSize> ws;
};

/// Round-robin warps to blocks, as `custom_all_reduce.cuh` does: a decode-sized
/// push then spreads over many CUs instead of saturating the first few.
SGL_DEVICE uint32_t muse_ar_round_robin_tid() {
  using device::kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * (threadIdx.x / kWarpThreads);
  return global_warp_id * kWarpThreads + threadIdx.x % kWarpThreads;
}

template <uint32_t kBlockSize, uint32_t kMaxVecs>
struct MuseBoundaryRow {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;
  /// Half of a bf16 vector's 8 channels in fp32.
  using f32_vec_t = device::AlignedVector<fp32x2_t, 2>;

  /// Element `k` of every per-thread array is row vector `threadIdx.x + k * kBlockSize`.
  struct Operands {
    f32_vec_t stream[kMaxVecs][2];
  };

  SGL_DEVICE static bool valid(uint32_t k, uint32_t row_vecs) {
    return threadIdx.x + k * kBlockSize < row_vecs;
  }

  /// Issued before polling, so these loads overlap the wait for peers.
  template <uint32_t kWorldSize>
  SGL_DEVICE static Operands load(const MuseArBoundaryParams<kWorldSize>& params, uint32_t row) {
    const auto base = static_cast<int64_t>(row) * params.row_vecs;
    Operands ops;
#pragma unroll
    for (uint32_t k = 0; k < kMaxVecs; ++k) {
      const auto v = threadIdx.x + k * kBlockSize;
      if (!valid(k, params.row_vecs)) continue;
      ops.stream[k][0].load(params.stream, (base + v) * 2);
      ops.stream[k][1].load(params.stream, (base + v) * 2 + 1);
    }
    return ops;
  }

  template <uint32_t kWorldSize>
  SGL_DEVICE static void apply(
      const MuseArBoundaryParams<kWorldSize>& params,
      uint32_t row,
      const vec_t (&x)[kMaxVecs],
      Operands& ops,
      float* smem_post,
      float* smem_pre) {
    using namespace device;
    const auto row_vecs = params.row_vecs;
    const auto base = static_cast<int64_t>(row) * row_vecs;

    float post_sum_sq = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kMaxVecs; ++k) {
      if (!valid(k, row_vecs)) continue;
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const auto b = cast<fp32x2_t>(x[k][j]);
        post_sum_sq += b.x * b.x + b.y * b.y;
      }
    }
    post_sum_sq = cta::reduce_sum<kBlockSize>(post_sum_sq, smem_post);
    const auto post_rstd = math::rsqrt(post_sum_sq * params.inv_hidden + params.post_eps);

    float pre_sum_sq = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kMaxVecs; ++k) {
      const auto v = threadIdx.x + k * kBlockSize;
      if (!valid(k, row_vecs)) continue;
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
        f32_vec_t alpha, beta;
        alpha.load(params.alpha, v * 2 + h);
        beta.load(params.beta, v * 2 + h);
        auto& s = ops.stream[k][h];
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          const auto b = cast<fp32x2_t>(x[k][h * 2 + j]);
          // the post-norm returns the branch dtype
          const auto normed = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{b.x * post_rstd, b.y * post_rstd}));
          s[j] = fp32x2_t{
              alpha[j].x * s[j].x + beta[j].x * normed.x,
              alpha[j].y * s[j].y + beta[j].y * normed.y,
          };
          pre_sum_sq += s[j].x * s[j].x + s[j].y * s[j].y;
        }
        s.store(params.new_stream, (base + v) * 2 + h);
      }
    }
    pre_sum_sq = cta::reduce_sum<kBlockSize>(pre_sum_sq, smem_pre);
    const auto pre_rstd = math::rsqrt(pre_sum_sq * params.inv_hidden + params.pre_eps);

#pragma unroll
    for (uint32_t k = 0; k < kMaxVecs; ++k) {
      const auto v = threadIdx.x + k * kBlockSize;
      if (!valid(k, row_vecs)) continue;
      vec_t w, y;
      w.load(params.pre_weight, v);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const auto& s = ops.stream[k][j / 2][j % 2];
        const auto wf = cast<fp32x2_t>(w[j]);
        y[j] = cast<bf16x2_t>(fp32x2_t{s.x * pre_rstd * wf.x, s.y * pre_rstd * wf.y});
      }
      y.store(params.output, base + v);
    }
    // the smem partials are reused by the next row's reductions
    __syncthreads();
  }

  /// Poll reduced row `row` from `poll_ptr` (vectors from `offset` on), reset
  /// its slots and run the boundary on it. `kSleep` backs off between polls.
  template <typename Lamport, bool kSleep = false, uint32_t kWorldSize>
  SGL_DEVICE static void drain_apply(
      const MuseArBoundaryParams<kWorldSize>& params,
      uint32_t row,
      void* poll_ptr,
      uint32_t offset,
      float* smem_post,
      float* smem_pre) {
    using namespace device;
    const auto row_vecs = params.row_vecs;
    const auto base = offset + threadIdx.x;
    auto ops = load(params, row);
    vec_t x[kMaxVecs];
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t k = 0; k < kMaxVecs; ++k) {
        if (valid(k, row_vecs)) ptx::ld_relaxed_16B(x[k], poll_ptr, base + k * kBlockSize);
      }
#pragma unroll
      for (uint32_t k = 0; k < kMaxVecs; ++k) {
        has_zero |= valid(k, row_vecs) && Lamport::has_pos_zero(x[k].data());
      }
      if (!has_zero) break;
      if constexpr (kSleep) __builtin_amdgcn_s_sleep(1);
    } while (true);
    vec_t pos_zero_vec;
    Lamport::fill_pos_zero(pos_zero_vec.data());
#pragma unroll
    for (uint32_t k = 0; k < kMaxVecs; ++k) {
      if (valid(k, row_vecs)) ptx::st_global_16B(pos_zero_vec, poll_ptr, base + k * kBlockSize);
    }
    apply(params, row, x, ops, smem_post, smem_pre);
  }
};

/// A slot (dst, src) holds two `chunk_vecs = rows_per_rank * row_vecs` halves:
/// [0, chunk) takes src's partial of dst's rows, and [chunk, 2 * chunk) takes
/// src's reduced rows. Only the second half of the (r, r) slot is used.
template <uint32_t kWorldSize, uint32_t kBlockSize, uint32_t kMaxVecs, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize, 1) void muse_all_reduce_norm_residual_2shot_push_kernel(
    const __grid_constant__ MuseArBoundaryParams<kWorldSize> params) {
  using namespace device;
  using Row = MuseBoundaryRow<kBlockSize, kMaxVecs>;
  using vec_t = typename Row::vec_t;
  using Lamport = distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;
  __shared__ float smem_post[kBlockSize / kWarpThreads];
  __shared__ float smem_pre[kBlockSize / kWarpThreads];
  const auto r = params.rank;
  const auto row_vecs = params.row_vecs;
  const auto rows_per_rank = params.rows_per_rank;
  const auto chunk_vecs = rows_per_rank * row_vecs;
  const auto num_vecs = params.num_rows * row_vecs;
  const auto num_threads = kBlockSize * gridDim.x;
  const auto global_tid = muse_ar_round_robin_tid();
  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  // reduce-scatter: my partial of every peer-owned row to its owner
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    const auto owner = vid / chunk_vecs;
    if (owner == r) continue;
    vec_t vec;
    vec.load(params.input, vid);
    Lamport::clear_pos_zero(vec.data());
    ptx::st_relaxed_16B(vec, epoch.slot_ptr(/*dst=*/owner, /*src=*/r), vid - owner * chunk_vecs);
  }

  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());
  void* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = epoch.slot_ptr(/*dst=*/r, /*src=*/i);
  }

  // reduce my rows across the whole grid and fan the sum out to every rank,
  // this one included, so the boundary below reads every row the same way
  const auto my_len = min(chunk_vecs, num_vecs - min(num_vecs, r * chunk_vecs));
  for (auto j = global_tid; j < my_len; j += num_threads) {
    vec_t vec[kWorldSize];
    vec_t local_vec;
    local_vec.load(params.input, r * chunk_vecs + j);
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], j);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        has_zero |= ((i != r) & Lamport::has_pos_zero(vec[i].data()));
      }
      if (!has_zero) break;
    } while (true);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      if (i == r) vec[i] = local_vec;
    }
    auto out_vec = reduce_vec(vec);
    Lamport::clear_pos_zero(out_vec.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_relaxed_16B(out_vec, epoch.slot_ptr(/*dst=*/i, /*src=*/r), chunk_vecs + j);
    }
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      if (i != r) ptx::st_global_16B(pos_zero_vec, poll_ptrs[i], j);
    }
  }

  // all-gather + boundary: one block per row
  for (auto row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const auto owner = row / rows_per_rank;
    const auto row_offset = chunk_vecs + (row - owner * rows_per_rank) * row_vecs;
    Row::template drain_apply<Lamport>(params, row, poll_ptrs[owner], row_offset, smem_post, smem_pre);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.commit();
}

/// The kernel above with the reduce-scatter and reduce phases of
/// `all_reduce_2shot_push_unrolled_kernel` on its row-owned chunks; the fan-out
/// also goes to this rank's own slot, so the per-row drain is unchanged. Needs
/// `kWorldSize - 1` warps and every block resident (one per CU).
template <uint32_t kWorldSize, uint32_t kBlockSize, uint32_t kMaxVecs, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize, 1) __attribute__((amdgpu_waves_per_eu(1, 2))) void
muse_all_reduce_norm_residual_2shot_push_unrolled_kernel(
    const __grid_constant__ MuseArBoundaryParams<kWorldSize> params) {
  using namespace device;
  using Row = MuseBoundaryRow<kBlockSize, kMaxVecs>;
  using vec_t = typename Row::vec_t;
  using Lamport = distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;
  constexpr uint32_t kFan = kWorldSize - 1;
  constexpr uint32_t kUnroll = kTwoShotPushUnroll;
  constexpr uint32_t kReduceUnroll = kTwoShotPushReduceUnroll<kWorldSize>;
  constexpr uint32_t kHunk = kWarpThreads * kUnroll;
  constexpr uint32_t kReduceHunk = kWarpThreads * kReduceUnroll;
  __shared__ float smem_post[kBlockSize / kWarpThreads];
  __shared__ float smem_pre[kBlockSize / kWarpThreads];
  const auto r = params.rank;
  const auto row_vecs = params.row_vecs;
  const auto chunk_vecs = params.rows_per_rank * row_vecs;
  const auto num_vecs = params.num_rows * row_vecs;
  const auto get_chunk_len = [&](uint32_t c) {  //
    return min(chunk_vecs, num_vecs - min(num_vecs, c * chunk_vecs));
  };
  const auto input = static_cast<const vec_t*>(params.input);
  const auto num_threads = blockDim.x * gridDim.x;
  const auto num_warps = num_threads / kWarpThreads;
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;

  const auto peer = (r + 1 + global_warp_id % kFan) % kWorldSize;
  const auto group_warp = global_warp_id / kFan;
  const auto group_warps = num_warps / kFan + (global_warp_id % kFan < num_warps % kFan ? 1 : 0);
  const auto group_tid = group_warp * kWarpThreads + lane_id;
  const auto group_stride = group_warps * kWarpThreads;
  const auto peer_len = get_chunk_len(peer);
  const auto peer_full = peer_len / kHunk * kHunk;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  // reduce-scatter: my partial of the peer's rows to its owner
  {
    const auto src = input + peer * chunk_vecs;
    const auto push_ptr = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/peer, /*src=*/r));
    for (uint32_t base = group_warp * kHunk; base < peer_full; base += group_warps * kHunk) {
      const auto j = base + lane_id;
      const auto in = src + j;
      vec_t vecs[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        vecs[u] = in[u * kWarpThreads];
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        Lamport::clear_pos_zero(vecs[u].data());
      }
      ptx::st_relaxed_run_16B<kWarpThreads, kUnroll>(vecs, push_ptr + j);
    }
    for (auto j = peer_full + group_tid; j < peer_len; j += group_stride) {
      vec_t vec = src[j];
      Lamport::clear_pos_zero(vec.data());
      ptx::st_relaxed_16B(vec, push_ptr, j);
    }
  }

  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());
  vec_t* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/r, /*src=*/i));
  }

  // reduce my rows and fan the sum out to every rank, this one included
  {
    const auto my_len = get_chunk_len(r);
    const auto my_full = my_len / kReduceHunk * kReduceHunk;
    const auto my_in = input + r * chunk_vecs;
    for (uint32_t base = global_warp_id * kReduceHunk; base < my_full; base += num_warps * kReduceHunk) {
      const auto j = base + lane_id;
      vec_t vecs[kReduceUnroll][kWorldSize];
      vec_t mine[kReduceUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
        mine[u] = my_in[j + u * kWarpThreads];
      }
      // every rank's slot, the unused (r, r) one too, so no branch on r splits
      // the batch; this rank's own partial is the input, swapped in below
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          ptx::ld_relaxed_16B(vecs[u][i], poll_ptrs[i], j + u * kWarpThreads);
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        while (true) {
          bool has_zero = false;
#pragma unroll
          for (uint32_t u = 0; u < kReduceUnroll; ++u) {
            has_zero |= ((i != r) & Lamport::has_pos_zero(vecs[u][i].data()));
          }
          if (!has_zero) break;
          __builtin_amdgcn_s_sleep(1);
#pragma unroll
          for (uint32_t u = 0; u < kReduceUnroll; ++u) {
            ptx::ld_relaxed_16B(vecs[u][i], poll_ptrs[i], j + u * kWarpThreads);
          }
        }
      }
      vec_t outs[kReduceUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          if (i == r) vecs[u][i] = mine[u];
        }
        outs[u] = reduce_vec(vecs[u]);
        Lamport::clear_pos_zero(outs[u].data());
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        const auto send_ptr = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/i, /*src=*/r));
        ptx::st_relaxed_run_16B<kWarpThreads, kReduceUnroll>(outs, send_ptr + chunk_vecs + j);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) {
#pragma unroll
          for (uint32_t u = 0; u < kReduceUnroll; ++u) {
            ptx::st_global_16B(pos_zero_vec, poll_ptrs[i] + j, u * kWarpThreads);
          }
        }
      }
    }
    for (auto j = my_full + global_tid; j < my_len; j += num_threads) {
      vec_t vec[kWorldSize];
      const vec_t mine = my_in[j];
      do {
        bool has_zero = false;
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          if (i != r) ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], j);
        }
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          has_zero |= ((i != r) & Lamport::has_pos_zero(vec[i].data()));
        }
        if (!has_zero) break;
      } while (true);
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i == r) vec[i] = mine;
      }
      auto out_vec = reduce_vec(vec);
      Lamport::clear_pos_zero(out_vec.data());
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::st_relaxed_16B(out_vec, epoch.slot_ptr(/*dst=*/i, /*src=*/r), chunk_vecs + j);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) ptx::st_global_16B(pos_zero_vec, poll_ptrs[i], j);
      }
    }
  }

  // all-gather + boundary: one block per row
  for (auto row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const auto owner = row / params.rows_per_rank;
    const auto row_offset = chunk_vecs + (row - owner * params.rows_per_rank) * row_vecs;
    Row::template drain_apply<Lamport, /*kSleep=*/true>(
        params, row, poll_ptrs[owner], row_offset, smem_post, smem_pre);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.commit();
}

/// The boundary without a per-row block phase. The owner of a row reduces it in
/// 128-vector segments, one warp each, and fans out the bf16 sum together with the
/// segment's four sums x.x, a.a, a.c, c.c (a = alpha * stream, c = beta * x). The
/// warp applying a segment on any rank adds the row's segment sums in a fixed order,
///   post_rstd = rsqrt(x.x / H + eps),  s.s = a.a + 2 post_rstd a.c + post_rstd^2 c.c,
/// and applies the boundary elementwise. s.s skips the bf16 rounding of the post-norm
/// output, so the pre-norm factor is not bitwise the Triton kernel's.
///
/// A slot (dst, src) holds [0, chunk) src's partial of dst's rows, [chunk, 2 chunk)
/// src's reduced rows and then the sums: a vector per (row, segment, applying segment).
template <uint32_t kWorldSize, uint32_t kBlockSize, uint32_t kPieceUnroll, bool kUsePDL>
__global__ __launch_bounds__(kBlockSize, 1) __attribute__((amdgpu_waves_per_eu(1, 2))) void
muse_all_reduce_norm_residual_2shot_push_factored_kernel(
    const __grid_constant__ MuseArBoundaryParams<kWorldSize> params) {
  using namespace device;
  using vec_t = AlignedVector<bf16x2_t, 4>;
  using f32_vec_t = AlignedVector<fp32x2_t, 2>;
  using Lamport = distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;
  using SumsLamport = distributed::LamportTrait<float, 4, /*kAtom=*/4>;
  constexpr uint32_t kFan = kWorldSize - 1;
  constexpr uint32_t kUnroll = kTwoShotPushUnroll;
  constexpr uint32_t kReduceUnroll = kTwoShotPushReduceUnroll<kWorldSize>;
  constexpr uint32_t kHunk = kWarpThreads * kUnroll;
  constexpr uint32_t kSeg = kWarpThreads * kReduceUnroll;
  constexpr uint32_t kPiece = kWarpThreads * kPieceUnroll;
  static_assert(kReduceUnroll % kPieceUnroll == 0);
  const auto r = params.rank;
  const auto row_vecs = params.row_vecs;
  const auto rows_per_rank = params.rows_per_rank;
  const auto segs = params.segs;
  const auto pieces = segs * (kReduceUnroll / kPieceUnroll);
  const auto chunk_vecs = rows_per_rank * row_vecs;
  const auto num_vecs = params.num_rows * row_vecs;
  const auto get_chunk_len = [&](uint32_t c) {  //
    return min(chunk_vecs, num_vecs - min(num_vecs, c * chunk_vecs));
  };
  const auto input = static_cast<const vec_t*>(params.input);
  const auto num_warps = blockDim.x * gridDim.x / kWarpThreads;
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;

  const auto peer = (r + 1 + global_warp_id % kFan) % kWorldSize;
  const auto group_warp = global_warp_id / kFan;
  const auto group_warps = num_warps / kFan + (global_warp_id % kFan < num_warps % kFan ? 1 : 0);
  const auto group_tid = group_warp * kWarpThreads + lane_id;
  const auto group_stride = group_warps * kWarpThreads;
  const auto peer_len = get_chunk_len(peer);
  const auto peer_full = peer_len / kHunk * kHunk;

  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  // reduce-scatter: my partial of the peer's rows to its owner
  {
    const auto src = input + peer * chunk_vecs;
    const auto push_ptr = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/peer, /*src=*/r));
    for (uint32_t base = group_warp * kHunk; base < peer_full; base += group_warps * kHunk) {
      const auto j = base + lane_id;
      const auto in = src + j;
      vec_t vecs[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        vecs[u] = in[u * kWarpThreads];
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        Lamport::clear_pos_zero(vecs[u].data());
      }
      ptx::st_relaxed_run_16B<kWarpThreads, kUnroll>(vecs, push_ptr + j);
    }
    for (auto j = peer_full + group_tid; j < peer_len; j += group_stride) {
      vec_t vec = src[j];
      Lamport::clear_pos_zero(vec.data());
      ptx::st_relaxed_16B(vec, push_ptr, j);
    }
  }

  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());
  vec_t* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/r, /*src=*/i));
  }
  const auto sums_base = 2 * chunk_vecs;

  // reduce my rows a piece per warp, fan the sum out with the piece's sums
  {
    const auto my_rows = min(rows_per_rank, params.num_rows - min(params.num_rows, r * rows_per_rank));
    const auto my_in = input + r * chunk_vecs;
    for (uint32_t piece_id = global_warp_id; piece_id < my_rows * pieces; piece_id += num_warps) {
      const auto row_local = piece_id / pieces;
      const auto row = r * rows_per_rank + row_local;
      const auto j = piece_id * kPiece + lane_id;
      const auto v = (piece_id % pieces) * kPiece + lane_id;
      const auto row_base = static_cast<int64_t>(row) * row_vecs;
      vec_t mine[kPieceUnroll];
      f32_vec_t st[kPieceUnroll][2], al[kPieceUnroll][2], be[kPieceUnroll][2];
#pragma unroll
      for (uint32_t u = 0; u < kPieceUnroll; ++u) {
        const auto vu = v + u * kWarpThreads;
        mine[u] = my_in[j + u * kWarpThreads];
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
          st[u][h].load(params.stream, (row_base + vu) * 2 + h);
          al[u][h].load(params.alpha, vu * 2 + h);
          be[u][h].load(params.beta, vu * 2 + h);
        }
      }
      vec_t vecs[kPieceUnroll][kWorldSize];
#pragma unroll
      for (uint32_t u = 0; u < kPieceUnroll; ++u) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          ptx::ld_relaxed_16B(vecs[u][i], poll_ptrs[i], j + u * kWarpThreads);
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        while (true) {
          bool has_zero = false;
#pragma unroll
          for (uint32_t u = 0; u < kPieceUnroll; ++u) {
            has_zero |= ((i != r) & Lamport::has_pos_zero(vecs[u][i].data()));
          }
          if (!has_zero) break;
          __builtin_amdgcn_s_sleep(1);
#pragma unroll
          for (uint32_t u = 0; u < kPieceUnroll; ++u) {
            ptx::ld_relaxed_16B(vecs[u][i], poll_ptrs[i], j + u * kWarpThreads);
          }
        }
      }
      vec_t outs[kPieceUnroll];
      float xx = 0.0f, aa = 0.0f, ac = 0.0f, cc = 0.0f;
#pragma unroll
      for (uint32_t u = 0; u < kPieceUnroll; ++u) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          if (i == r) vecs[u][i] = mine[u];
        }
        outs[u] = reduce_vec(vecs[u]);
#pragma unroll
        for (uint32_t k = 0; k < 4; ++k) {
          const auto x = cast<fp32x2_t>(outs[u][k]);
          const auto& s = st[u][k / 2][k % 2];
          const auto& a = al[u][k / 2][k % 2];
          const auto& b = be[u][k / 2][k % 2];
          const fp32x2_t av{a.x * s.x, a.y * s.y}, cv{b.x * x.x, b.y * x.y};
          xx += x.x * x.x + x.y * x.y;
          aa += av.x * av.x + av.y * av.y;
          ac += av.x * cv.x + av.y * cv.y;
          cc += cv.x * cv.x + cv.y * cv.y;
        }
        Lamport::clear_pos_zero(outs[u].data());
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        const auto send_ptr = static_cast<vec_t*>(epoch.slot_ptr(/*dst=*/i, /*src=*/r));
        if constexpr (kPieceUnroll % 4 == 0) {
          ptx::st_relaxed_run_16B<kWarpThreads, kPieceUnroll>(outs, send_ptr + chunk_vecs + j);
        } else {
#pragma unroll
          for (uint32_t u = 0; u < kPieceUnroll; ++u) {
            ptx::st_relaxed_16B(outs[u], send_ptr + chunk_vecs + j, u * kWarpThreads);
          }
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (i != r) {
#pragma unroll
          for (uint32_t u = 0; u < kPieceUnroll; ++u) {
            ptx::st_global_16B(pos_zero_vec, poll_ptrs[i] + j, u * kWarpThreads);
          }
        }
      }
      AlignedVector<fp32x2_t, 2> sums;
      sums[0] = fp32x2_t{warp::reduce_sum(xx), warp::reduce_sum(aa)};
      sums[1] = fp32x2_t{warp::reduce_sum(ac), warp::reduce_sum(cc)};
      SumsLamport::clear_pos_zero(sums.data());
      // one copy for each segment's applier on every rank, so each resets its own
      const auto piece = piece_id % pieces;
      for (auto idx = lane_id; idx < kWorldSize * segs; idx += kWarpThreads) {
        ptx::st_relaxed_16B(
            sums,
            epoch.slot_ptr(/*dst=*/idx / segs, /*src=*/r),
            sums_base + (row_local * pieces + piece) * segs + idx % segs);
      }
    }
  }

  // every row's segments, elementwise. A warp takes up to two segments at a time
  // and issues all their loads (reduced rows, sums, stream, gates, weight) before
  // waiting on any, so a segment costs one memory round trip, not three.
  struct Seg {
    vec_t x[kReduceUnroll];
    vec_t w[kReduceUnroll];
    f32_vec_t st[kReduceUnroll][2], al[kReduceUnroll][2], be[kReduceUnroll][2];
    AlignedVector<fp32x2_t, 2> sums;
  };
  const auto num_segs = params.num_rows * segs;
  const auto x_ptr_of = [&](uint32_t gs) {
    const auto row = gs / segs;
    const auto owner = row / rows_per_rank;
    return poll_ptrs[owner] + chunk_vecs + (row - owner * rows_per_rank) * row_vecs + gs % segs * kSeg + lane_id;
  };
  const auto sums_ptr_of = [&](uint32_t gs) {
    const auto row = gs / segs;
    const auto owner = row / rows_per_rank;
    return poll_ptrs[owner] + sums_base + ((row - owner * rows_per_rank) * pieces + lane_id) * segs + gs % segs;
  };
  const auto issue = [&](uint32_t gs, Seg& q) {
    const auto v = gs % segs * kSeg + lane_id;
    const auto row_base = static_cast<int64_t>(gs / segs) * row_vecs;
    const auto x_ptr = x_ptr_of(gs);
#pragma unroll
    for (uint32_t u = 0; u < kReduceUnroll; ++u) {
      const auto vu = v + u * kWarpThreads;
      ptx::ld_relaxed_16B(q.x[u], x_ptr, u * kWarpThreads);
      q.w[u].load(params.pre_weight, vu);
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
        q.st[u][h].load(params.stream, (row_base + vu) * 2 + h);
        q.al[u][h].load(params.alpha, vu * 2 + h);
        q.be[u][h].load(params.beta, vu * 2 + h);
      }
    }
    if (lane_id < pieces) ptx::ld_relaxed_16B(q.sums, sums_ptr_of(gs), 0);
  };
  // whether this lane's reduced vectors and sums have landed; re-issues the ones that have not
  const auto landed = [&](uint32_t gs, Seg& q) {
    bool x_pending = false;
#pragma unroll
    for (uint32_t u = 0; u < kReduceUnroll; ++u) {
      x_pending |= Lamport::has_pos_zero(q.x[u].data());
    }
    const bool sums_pending = lane_id < pieces && SumsLamport::has_pos_zero(q.sums.data());
    if (x_pending) {
      const auto x_ptr = x_ptr_of(gs);
#pragma unroll
      for (uint32_t u = 0; u < kReduceUnroll; ++u) {
        ptx::ld_relaxed_16B(q.x[u], x_ptr, u * kWarpThreads);
      }
    }
    if (sums_pending) ptx::ld_relaxed_16B(q.sums, sums_ptr_of(gs), 0);
    return !x_pending && !sums_pending;
  };
  const auto finish = [&](uint32_t gs, Seg& q) {
    const auto x_ptr = x_ptr_of(gs);
    // lane l < pieces takes piece l's sums; the same tree on every rank and segment
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane_id < pieces) {
      ptx::st_global_16B(pos_zero_vec, sums_ptr_of(gs), 0);
      sums[0] = q.sums[0].x, sums[1] = q.sums[0].y, sums[2] = q.sums[1].x, sums[3] = q.sums[1].y;
    }
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
      sums[k] = warp::reduce_sum(sums[k]);
    }
    const auto post_rstd = math::rsqrt(sums[0] * params.inv_hidden + params.post_eps);
    const auto ss = sums[1] + 2.0f * post_rstd * sums[2] + post_rstd * post_rstd * sums[3];
    const auto pre_rstd = math::rsqrt(ss * params.inv_hidden + params.pre_eps);
#pragma unroll
    for (uint32_t u = 0; u < kReduceUnroll; ++u) {
      ptx::st_global_16B(pos_zero_vec, x_ptr, u * kWarpThreads);
    }
    const auto v = gs % segs * kSeg + lane_id;
    const auto row_base = static_cast<int64_t>(gs / segs) * row_vecs;
#pragma unroll
    for (uint32_t u = 0; u < kReduceUnroll; ++u) {
      const auto vu = v + u * kWarpThreads;
      vec_t y;
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
        auto& s = q.st[u][h];
#pragma unroll
        for (uint32_t k = 0; k < 2; ++k) {
          const auto& alpha = q.al[u][h][k];
          const auto& beta = q.be[u][h][k];
          const auto b = cast<fp32x2_t>(q.x[u][h * 2 + k]);
          const auto normed = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{b.x * post_rstd, b.y * post_rstd}));
          s[k] = fp32x2_t{alpha.x * s[k].x + beta.x * normed.x, alpha.y * s[k].y + beta.y * normed.y};
          const auto wf = cast<fp32x2_t>(q.w[u][h * 2 + k]);
          y[h * 2 + k] = cast<bf16x2_t>(fp32x2_t{s[k].x * pre_rstd * wf.x, s[k].y * pre_rstd * wf.y});
        }
        s.store(params.new_stream, (row_base + vu) * 2 + h);
      }
      y.store(params.output, row_base + vu);
    }
  };
  for (uint32_t gs = global_warp_id; gs < num_segs; gs += 2 * num_warps) {
    const auto gs2 = gs + num_warps;
    const bool two = gs2 < num_segs;
    Seg q0, q1;
    issue(gs, q0);
    if (two) issue(gs2, q1);
    // one wait for both segments: they land at about the same time
    for (bool done0 = false, done1 = !two;;) {
      done0 = done0 || landed(gs, q0);
      done1 = done1 || landed(gs2, q1);
      if (done0 && done1) break;
      __builtin_amdgcn_s_sleep(1);
    }
    finish(gs, q0);
    if (two) finish(gs2, q1);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.commit();
}

/// kMode: 0 per-vector exchange, 1 unrolled exchange from kTwoShotPushUnrolledMinBytes, 2 factored.
template <uint32_t kWorldSize, bool kUsePDL, uint32_t kMode>
struct MuseAllReduceNormResidualKernel {
  static constexpr uint32_t kBlockSize = 256;
  static constexpr uint32_t kMaxVecs = 4;
  /// factored: 16-byte vectors per lane in a reduce piece. Half a segment: at 97+ rows a
  /// rank's 104+ whole segments leave some of the 96 CUs reducing two, which costs ~3 us
  static constexpr uint32_t kPieceUnroll = 2;
  static constexpr uint32_t kVecElems = 8;

  static void
  run(const CommunicatorRef comm_ref,
      const tvm::ffi::TensorView in,
      const tvm::ffi::TensorView stream,
      const tvm::ffi::TensorView pre_weight,
      const tvm::ffi::TensorView alpha,
      const tvm::ffi::TensorView beta,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView new_stream,
      const double post_eps,
      const double pre_eps) {
    using namespace host;
    const auto& comm = *comm_ref.get();
    CHECK_HOST(comm.get_world_size() == kWorldSize) << comm.get_world_size();
    CHECK_HOST(in.ndim() == 2 && in.IsContiguous() && is_type<bf16_t>(in.dtype()) && in.device().device_type == kDLGPU);
    const auto num_rows = in.size(0);
    const auto hidden = in.size(1);
    for (const auto& t : {stream, out, new_stream}) {
      CHECK_HOST(t.ndim() == 2 && t.size(0) == num_rows && t.size(1) == hidden && t.IsContiguous());
    }
    CHECK_HOST(is_type<fp32_t>(stream.dtype()) && is_type<fp32_t>(new_stream.dtype()) && is_type<bf16_t>(out.dtype()));
    CHECK_HOST(pre_weight.ndim() == 1 && pre_weight.size(0) == hidden && is_type<bf16_t>(pre_weight.dtype()));
    for (const auto& t : {alpha, beta}) {
      CHECK_HOST(t.ndim() == 1 && t.size(0) == hidden && is_type<fp32_t>(t.dtype()));
    }
    for (const auto& t : {in, stream, pre_weight, alpha, beta, out, new_stream}) {
      CHECK_HOST(t.IsContiguous() && reinterpret_cast<intptr_t>(t.data_ptr()) % 16 == 0);
    }
    CHECK_HOST(hidden % kVecElems == 0 && hidden / kVecElems <= kBlockSize * kMaxVecs) << hidden;
    CHECK_HOST(num_rows > 0 && num_rows * hidden <= INT32_MAX) << num_rows;
    const auto row_vecs = static_cast<uint32_t>(hidden / kVecElems);
    const auto rows = static_cast<uint32_t>(num_rows);

    const auto& push = comm.get_push_obj();
    const auto rows_per_rank = div_ceil(rows, kWorldSize);
    constexpr uint32_t kSeg = device::kWarpThreads * kTwoShotPushReduceUnroll<kWorldSize>;
    const auto segs = row_vecs / kSeg;
    const auto pieces = segs * (kTwoShotPushReduceUnroll<kWorldSize> / kPieceUnroll);
    // a lane per piece gathers the row's sums
    const bool factored = kMode == 2 && kWorldSize <= 16 && row_vecs % kSeg == 0 && pieces <= device::kWarpThreads;
    const auto slot_bytes =
        int64_t{2} * rows_per_rank * row_vecs * 16 + (factored ? int64_t{rows_per_rank} * pieces * segs * 16 : 0);
    const auto params = MuseArBoundaryParams<kWorldSize>{
        .input = in.data_ptr(),
        .stream = stream.data_ptr(),
        .pre_weight = pre_weight.data_ptr(),
        .alpha = alpha.data_ptr(),
        .beta = beta.data_ptr(),
        .output = out.data_ptr(),
        .new_stream = new_stream.data_ptr(),
        .num_rows = rows,
        .row_vecs = row_vecs,
        .rows_per_rank = rows_per_rank,
        .rank = push.rank,
        .post_eps = static_cast<float>(post_eps),
        .pre_eps = static_cast<float>(pre_eps),
        .inv_hidden = 1.0f / static_cast<float>(hidden),
        .segs = segs,
        .ws = push.get_workspace<kWorldSize>(slot_bytes),
    };
    const auto device_stream = LaunchKernel::resolve_device(in.device());
    // one block per CU: more than one co-resident block is not guaranteed (the Lamport polls would hang)
    const auto one_per_cu = kWorldSize <= 8
                                ? std::max(push.default_blocks, runtime::get_sm_count(in.device().device_id))
                                : push.default_blocks;
    if constexpr (kMode == 2 && kWorldSize <= 16) {
      static_assert(kBlockSize == kTwoShotPushThreads);
      if (factored) {
        CHECK_HOST(one_per_cu * (kBlockSize / device::kWarpThreads) >= kWorldSize - 1);
        LaunchKernel(one_per_cu, kBlockSize, device_stream)  //
            .enable_pdl(kUsePDL)(
                muse_all_reduce_norm_residual_2shot_push_factored_kernel<kWorldSize, kBlockSize, kPieceUnroll, kUsePDL>,
                params);
        return;
      }
    }
    if constexpr (kMode == 1 && kWorldSize <= 16) {
      static_assert(kBlockSize == kTwoShotPushThreads);
      // the plain all-reduce's pick (custom_all_reduce.cuh), so both arms run the same exchange
      const auto nbytes = int64_t{num_rows} * hidden * static_cast<int64_t>(sizeof(bf16_t));
      const bool enough_warps = push.default_blocks * (kBlockSize / device::kWarpThreads) >= kWorldSize - 1;
      if (enough_warps && nbytes >= kTwoShotPushUnrolledMinBytes<kWorldSize>) {
        LaunchKernel(one_per_cu, kBlockSize, device_stream)  //
            .enable_pdl(kUsePDL)(
                muse_all_reduce_norm_residual_2shot_push_unrolled_kernel<kWorldSize, kBlockSize, kMaxVecs, kUsePDL>,
                params);
        return;
      }
    }
    LaunchKernel(push.default_blocks, kBlockSize, device_stream)  //
        .enable_pdl(kUsePDL)(muse_all_reduce_norm_residual_2shot_push_kernel<kWorldSize, kBlockSize, kMaxVecs, kUsePDL>,
                             params);
  }
};

}  // namespace sglang
