// Fused all-reduce + residual add + RMSNorm over the Lamport push plane.
//
//   residual_out = all_reduce(input) + residual
//   output       = rmsnorm(residual_out) * weight
//
// Both algorithms share the push plane, the per-block epoch counters and the
// slot layout conventions of `custom_all_reduce.cuh`, so they interleave freely
// with the plain push all-reduce. The exchange and reduction are spread over
// the whole grid exactly as in the plain kernels; the reduced bf16 sum then
// lands in this rank's own slot with a Lamport marker, and one block per row
// (block b takes rows b, b + grid, ...) drains it and normalises it:
//   - 1shot_push: every rank reduces every row and stages the sum in the
//     spare half of its (r, r) slot.
//   - 2shot_push: ownership is by whole rows (rank c owns rows
//     [c * R, (c + 1) * R)), so the owner reduces its rows and fans the sum
//     out to every rank, itself included.
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
#include <string>

namespace sglang {

using device::distributed::PushWorkSpace;
using host::distributed::CommunicatorRef;

template <uint32_t kWorldSize>
struct AllReduceRMSNormParams {
  const void* __restrict__ input;
  const void* __restrict__ residual;
  const void* __restrict__ weight;
  void* __restrict__ output;
  void* __restrict__ residual_out;
  uint32_t num_rows;
  uint32_t row_vecs;
  uint32_t rows_per_rank;  // 2shot only
  uint32_t rank;
  float eps;
  float inv_hidden;
  PushWorkSpace<kWorldSize> ws;
};

/// Store to this GPU's own workspace for another block of the same kernel: the
/// agent scope is enough, and its polling load (agent scope on ROCm) hits L2.
template <typename V>
SGL_DEVICE void st_local_16B(const V& x, void* addr, int64_t vec_offset) {
#if defined(USE_ROCM) && defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_global_store_b128)
  using u32x4_t = uint32_t __attribute__((ext_vector_type(4)));
  using global_u32x4_t = __attribute__((address_space(1))) u32x4_t;
  const auto ptr = static_cast<u32x4_t*>(addr) + vec_offset;
  __builtin_amdgcn_global_store_b128((global_u32x4_t*)ptr, *reinterpret_cast<const u32x4_t*>(&x), "agent");
#else
  device::ptx::st_relaxed_16B(x, addr, vec_offset);
#endif
}

/// Round-robin warps to blocks, as `custom_all_reduce.cuh` does: a decode-sized
/// push then spreads over many CUs instead of saturating the first few.
SGL_DEVICE uint32_t round_robin_tid() {
  using device::kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * (threadIdx.x / kWarpThreads);
  return global_warp_id * kWarpThreads + threadIdx.x % kWarpThreads;
}

template <typename T, uint32_t kBlockSize, uint32_t kMaxVecs>
struct RowEpilogue {
  static constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using T2 = packed_t<T>;
  using vec_t = device::AlignedVector<T2, kVecSize>;

  /// Element `k` of every per-thread array is row vector `threadIdx.x + k * kBlockSize`.
  struct Operands {
    vec_t residual[kMaxVecs];
    vec_t weight[kMaxVecs];
  };

  SGL_DEVICE static bool valid(uint32_t k, uint32_t row_vecs) {
    return threadIdx.x + k * kBlockSize < row_vecs;
  }

  /// Issued before polling, so these loads overlap the wait for peers.
  template <uint32_t kWorldSize>
  SGL_DEVICE static Operands load(const AllReduceRMSNormParams<kWorldSize>& params, uint32_t row) {
    const auto row_vecs = params.row_vecs;
    const auto base = static_cast<int64_t>(row) * row_vecs;
    Operands ops;
#pragma unroll
    for (uint32_t k = 0; k < kMaxVecs; ++k) {
      const auto v = threadIdx.x + k * kBlockSize;
      if (!valid(k, row_vecs)) continue;
      ops.residual[k].load(params.residual, base + v);
      ops.weight[k].load(params.weight, v);
    }
    return ops;
  }

  template <uint32_t kWorldSize>
  SGL_DEVICE static void apply(
      const AllReduceRMSNormParams<kWorldSize>& params,
      uint32_t row,
      const vec_t (&x)[kMaxVecs],
      Operands& ops,
      float* smem) {
    using namespace device;
    const auto row_vecs = params.row_vecs;
    const auto base = static_cast<int64_t>(row) * row_vecs;
    fp32x2_t acc[kMaxVecs][kVecSize];
    float sum_sq = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kMaxVecs; ++k) {
      const auto v = threadIdx.x + k * kBlockSize;
      if (!valid(k, row_vecs)) continue;
      auto& res = ops.residual[k];
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const auto a = cast<fp32x2_t>(x[k][j]);
        const auto b = cast<fp32x2_t>(res[j]);
        acc[k][j] = fp32x2_t{a.x + b.x, a.y + b.y};
        sum_sq += acc[k][j].x * acc[k][j].x + acc[k][j].y * acc[k][j].y;
        res[j] = cast<T2>(acc[k][j]);
      }
      res.store(params.residual_out, base + v);
    }
    sum_sq = cta::reduce_sum<kBlockSize>(sum_sq, smem);
    const auto scale = math::rsqrt(sum_sq * params.inv_hidden + params.eps);
#pragma unroll
    for (uint32_t k = 0; k < kMaxVecs; ++k) {
      const auto v = threadIdx.x + k * kBlockSize;
      if (!valid(k, row_vecs)) continue;
      const auto& w = ops.weight[k];
      vec_t y;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        const auto wf = cast<fp32x2_t>(w[j]);
        y[j] = cast<T2>(fp32x2_t{acc[k][j].x * scale * wf.x, acc[k][j].y * scale * wf.y});
      }
      y.store(params.output, base + v);
    }
    // `smem` is reused by the next row's reduction
    __syncthreads();
  }

  /// Poll reduced row `row` from `poll_ptr` (vectors from `offset` on), reset
  /// its slots and normalise it.
  template <typename Lamport, uint32_t kWorldSize>
  SGL_DEVICE static void drain_apply(
      const AllReduceRMSNormParams<kWorldSize>& params, uint32_t row, void* poll_ptr, uint32_t offset, float* smem) {
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
    apply(params, row, x, ops, smem);
  }
};

#define ALL_REDUCE_RMSNORM_KERNEL __global__ __launch_bounds__(1024, 1)

template <typename T, uint32_t kWorldSize, uint32_t kBlockSize, uint32_t kMaxVecs, bool kUsePDL>
ALL_REDUCE_RMSNORM_KERNEL void
all_reduce_rmsnorm_1shot_push_kernel(const __grid_constant__ AllReduceRMSNormParams<kWorldSize> params) {
  using namespace device;
  using Epilogue = RowEpilogue<T, kBlockSize, kMaxVecs>;
  using vec_t = typename Epilogue::vec_t;
  using Lamport = distributed::LamportTrait<T, Epilogue::kVecSize * 2, /*kAtom=*/4>;
  __shared__ float smem[kBlockSize / kWarpThreads];
  const auto r = params.rank;
  const auto row_vecs = params.row_vecs;
  const auto num_vecs = params.num_rows * row_vecs;
  const auto num_threads = kBlockSize * gridDim.x;
  const auto global_tid = round_robin_tid();
  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  void* push_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    push_ptrs[i] = epoch.slot_ptr(/*dst=*/i, /*src=*/r);
  }
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    vec.load(params.input, vid);
    Lamport::clear_pos_zero(vec.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_relaxed_16B(vec, push_ptrs[i], vid);
    }
  }

  void* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = epoch.slot_ptr(/*dst=*/r, /*src=*/i);
  }
  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());

  // reduce across the whole grid, staging the sum in the spare half of the
  // (r, r) slot for the per-row norm below
  const auto stage_ptr = poll_ptrs[r];
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec[kWorldSize];
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], vid);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        has_zero |= Lamport::has_pos_zero(vec[i].data());
      }
      if (!has_zero) break;
    } while (true);
    auto out_vec = reduce_vec(vec);
    Lamport::clear_pos_zero(out_vec.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_global_16B(pos_zero_vec, poll_ptrs[i], vid);
    }
    st_local_16B(out_vec, stage_ptr, num_vecs + vid);
  }

  for (auto row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    Epilogue::template drain_apply<Lamport>(params, row, stage_ptr, num_vecs + row * row_vecs, smem);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.flip();
}

/// A slot (dst, src) holds two `chunk_vecs = rows_per_rank * row_vecs` halves:
/// [0, chunk) takes src's partial of dst's rows, and [chunk, 2 * chunk) takes
/// src's reduced rows. Only the second half of the (r, r) slot is used.
template <typename T, uint32_t kWorldSize, uint32_t kBlockSize, uint32_t kMaxVecs, bool kUsePDL>
ALL_REDUCE_RMSNORM_KERNEL void
all_reduce_rmsnorm_2shot_push_kernel(const __grid_constant__ AllReduceRMSNormParams<kWorldSize> params) {
  using namespace device;
  using Epilogue = RowEpilogue<T, kBlockSize, kMaxVecs>;
  using vec_t = typename Epilogue::vec_t;
  using Lamport = distributed::LamportTrait<T, Epilogue::kVecSize * 2, /*kAtom=*/4>;
  __shared__ float smem[kBlockSize / kWarpThreads];
  const auto r = params.rank;
  const auto row_vecs = params.row_vecs;
  const auto rows_per_rank = params.rows_per_rank;
  const auto chunk_vecs = rows_per_rank * row_vecs;
  const auto num_vecs = params.num_rows * row_vecs;
  const auto num_threads = kBlockSize * gridDim.x;
  const auto global_tid = round_robin_tid();
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
  // this one included, so the norm below reads every row the same way
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

  // all-gather + norm: one block per row
  for (auto row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const auto owner = row / rows_per_rank;
    const auto poll_ptr = poll_ptrs[owner];
    const auto row_offset = chunk_vecs + (row - owner * rows_per_rank) * row_vecs;
    Epilogue::template drain_apply<Lamport>(params, row, poll_ptr, row_offset, smem);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.flip();
}

template <typename T, uint32_t kWorldSize, bool kUsePDL>
struct AllReduceRMSNormKernel {
  static constexpr uint32_t kSmallBlockSize = 256;
  static constexpr uint32_t kSmallMaxVecs = 4;
  static constexpr uint32_t kBlockSize = 1024;
  static constexpr uint32_t kMaxVecs = 4;
  static constexpr uint32_t kVecElems = 16 / sizeof(T);

  static void
  run(const CommunicatorRef comm_ref,
      const tvm::ffi::TensorView in,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView residual_out,
      const double eps,
      const std::string algo) {
    using namespace host;
    const auto& comm = *comm_ref.get();
    CHECK_HOST(algo == "1shot_push" || algo == "2shot_push") << algo;
    CHECK_HOST(comm.get_world_size() == kWorldSize) << comm.get_world_size();
    CHECK_HOST(in.ndim() == 2 && in.IsContiguous() && is_type<T>(in.dtype()) && in.device().device_type == kDLGPU);
    const auto num_rows = in.size(0);
    const auto hidden = in.size(1);
    for (const auto& t : {residual, out, residual_out}) {
      CHECK_HOST(t.ndim() == 2 && t.size(0) == num_rows && t.size(1) == hidden);
      CHECK_HOST(t.IsContiguous() && is_type<T>(t.dtype()));
    }
    CHECK_HOST(weight.ndim() == 1 && weight.size(0) == hidden && weight.IsContiguous() && is_type<T>(weight.dtype()));
    for (const auto& t : {in, residual, weight, out, residual_out}) {
      CHECK_HOST(reinterpret_cast<intptr_t>(t.data_ptr()) % 16 == 0);
    }
    CHECK_HOST(hidden % kVecElems == 0 && hidden / kVecElems <= kBlockSize * kMaxVecs) << hidden;
    CHECK_HOST(num_rows > 0 && num_rows * hidden <= INT32_MAX) << num_rows;
    const auto row_vecs = static_cast<uint32_t>(hidden / kVecElems);
    const auto rows = static_cast<uint32_t>(num_rows);

    const auto& push = comm.get_push_obj();
    const auto two_shot = algo == "2shot_push";
    const auto rows_per_rank = two_shot ? div_ceil(rows, kWorldSize) : rows;
    const auto slot_bytes = int64_t{2} * rows_per_rank * row_vecs * 16;
    const auto params = AllReduceRMSNormParams<kWorldSize>{
        .input = in.data_ptr(),
        .residual = residual.data_ptr(),
        .weight = weight.data_ptr(),
        .output = out.data_ptr(),
        .residual_out = residual_out.data_ptr(),
        .num_rows = rows,
        .row_vecs = row_vecs,
        .rows_per_rank = rows_per_rank,
        .rank = push.rank,
        .eps = static_cast<float>(eps),
        .inv_hidden = 1.0f / static_cast<float>(hidden),
        .ws = push.get_workspace<kWorldSize>(slot_bytes),
    };
    const auto stream = LaunchKernel::resolve_device(in.device());
    const auto launch = [&]<uint32_t kThreads, uint32_t kVecs>() {
      const auto kernel = two_shot ? all_reduce_rmsnorm_2shot_push_kernel<T, kWorldSize, kThreads, kVecs, kUsePDL>
                                   : all_reduce_rmsnorm_1shot_push_kernel<T, kWorldSize, kThreads, kVecs, kUsePDL>;
      // the grid is bound to the counter array and must stay constant
      LaunchKernel(push.num_blocks, kThreads, stream)  //
          .enable_pdl(kUsePDL)(kernel, params);
    };
    // gfx1250 TP8: wider blocks are slower (512 x 2 loses ~3 us to 256 x 4 at
    // 7168), and 128 x 8 spills; 1024 threads only for rows 256 x 4 cannot hold
    if (row_vecs <= kSmallBlockSize * kSmallMaxVecs) {
      launch.template operator()<kSmallBlockSize, kSmallMaxVecs>();
    } else {
      launch.template operator()<kBlockSize, kMaxVecs>();
    }
  }
};

}  // namespace sglang
