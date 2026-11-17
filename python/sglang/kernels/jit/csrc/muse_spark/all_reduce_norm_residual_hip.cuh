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
  /// its slots and run the boundary on it.
  template <typename Lamport, uint32_t kWorldSize>
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

template <uint32_t kWorldSize, bool kUsePDL>
struct MuseAllReduceNormResidualKernel {
  static constexpr uint32_t kBlockSize = 256;
  static constexpr uint32_t kMaxVecs = 4;
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
    const auto slot_bytes = int64_t{2} * rows_per_rank * row_vecs * 16;
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
        .ws = push.get_workspace<kWorldSize>(slot_bytes),
    };
    const auto device_stream = LaunchKernel::resolve_device(in.device());
    LaunchKernel(push.default_blocks, kBlockSize, device_stream)  //
        .enable_pdl(kUsePDL)(
            muse_all_reduce_norm_residual_2shot_push_kernel<kWorldSize, kBlockSize, kMaxVecs, kUsePDL>, params);
  }
};

}  // namespace sglang
