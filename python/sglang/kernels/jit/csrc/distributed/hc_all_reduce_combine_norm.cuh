// TP all-reduce fused with the gated hyper-connection write-back and the next
// read's grouped Gemma RMSNorm (HC = 4 branches of 2560 BF16), for decode-sized
// batches, over the CustomAllReduceV2 push plane:
//
//   y          = all_reduce(input)                      (1shot lamport push)
//   gate[b]    = 2 * sigmoid(sum_s partials[row, s, b] / 4)
//   combined   = bf16(residual[b] + gate[b] * y)
//   normalized = bf16(combined * rsqrt(mean_b(combined^2) + eps) * (1 + w[b]))
//
// It replaces all_reduce_1shot_push_kernel (custom_all_reduce.cuh) followed by
// hc_combine_norm_decode_kernel (elementwise/hc_combine_decode.cuh), one kernel
// boundary per HC write-back, and is bit-identical to that pair:
//   * the all-reduce stage is the 1shot push kernel's: +0.0 words remapped to
//     -0.0 before the push, every peer's vector reduced in rank order with fp32
//     accumulation (reduce_vec), one BF16 rounding; the reduced row is handed
//     through shared memory instead of global memory;
//   * the epilogue is the combine-norm kernel's: one CTA of 160 threads per
//     (row, branch), thread t on elements [16t, 16t + 16), the same gate, the
//     same BF16 store of `combined` before the norm, and the same square-sum
//     tree (per-thread sum, warp XOR reduction, five ordered warp partials plus
//     27 zeros reduced by warp 0).
//
// Geometry: one 4-CTA cluster per row along y (cluster rank = branch). Each CTA
// reduces a quarter of the row (80 16B vectors) and writes it into the shared
// memory of all four CTAs of its cluster; after a cluster barrier every CTA
// holds the reduced row and runs its branch. One 640-thread CTA per row measured
// slower (fewer SMs pushing). The dependent kernel is let launch as early as the
// caller asks (params.trigger): it reads nothing of ours before its own PDL wait.
//
// Push-plane protocol (include/sgl_kernel/distributed/communicator.cuh), as in
// moe_finalize_all_reduce_kernel (all_reduce_fusion.cuh): a round uses phase
// `counter & 1`; one phase counter per row cluster, flipped by its leader after
// the cluster barrier (every CTA has read it by then); a trailing bumper cluster
// flips the remaining counters, so the array keeps one parity and the generic
// push kernel shares the plane. Every rank must call with the same row count.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/extra/stl.h>

#include <cooperative_groups.h>
#include <cstdint>

namespace sglang {

namespace hc_ar {

inline constexpr uint32_t kHidden = 2560;
inline constexpr uint32_t kBranches = 4;
inline constexpr uint32_t kThreads = 160;                       // the combine-norm CTA
inline constexpr uint32_t kWarps = kThreads / 32;               // 5
inline constexpr uint32_t kRowVecs = kHidden / 8;               // 320 16B vectors per row
inline constexpr uint32_t kVecsPerCta = kRowVecs / kBranches;   // 80
inline constexpr uint32_t kMaxRows = 16;
// When this kernel lets its dependent (the next read's mix) launch and prefetch: on entry,
// once the producer is complete, or once the row is reduced (the stock all-reduce's point).
inline constexpr uint32_t kTriggerEntry = 0;
inline constexpr uint32_t kTriggerWait = 1;
inline constexpr uint32_t kTriggerReduced = 2;
static_assert(kThreads * 16 == kHidden, "thread t owns elements [16t, 16t + 16) of a branch");
static_assert(kVecsPerCta <= kThreads, "every reduced vector has a thread");

// Unaligned forms: callers may sit past a branch that splits a warp (tid < 80). The
// arrive orders no memory; a remote shared-memory store only needs its target CTA running.
SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
}

SGL_DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait;" ::: "memory");
}

}  // namespace hc_ar

template <uint32_t kWorldSize>
struct HcAllReduceCombineNormParams {
  const bf16_t* input;     // [rows, 2560], this rank's partial sum
  const bf16_t* residual;  // [rows, 4 * 2560]
  const float* partials;   // [rows, 8, 4], gate dot partials (only their sum over the 8 splits counts)
  const bf16_t* weight;    // [4 * 2560], the next read's per-branch norm weight
  bf16_t* combined;        // [rows, 4 * 2560]
  bf16_t* normalized;      // [rows, 4 * 2560]
  float eps;
  uint32_t rank;
  uint32_t num_rows;
  uint32_t num_push_counters;  // the plane's counter array size (the bumper's range end)
  uint32_t trigger;             // hc_ar::kTrigger*: when the dependent kernel may launch
  device::distributed::PushWorkSpace<kWorldSize> ws;
};

template <uint32_t kWorldSize, bool kUsePDL>
__global__ __launch_bounds__(hc_ar::kThreads) __cluster_dims__(1, hc_ar::kBranches, 1)
    void hc_all_reduce_combine_norm_kernel(const __grid_constant__ HcAllReduceCombineNormParams<kWorldSize> params) {
  namespace cg = cooperative_groups;
  using namespace device;
  using namespace hc_ar;
  using BFloat16x2 = packed_t<bf16_t>;
  using Storage = AlignedVector<BFloat16x2, 8>;   // 16 elements: the combine-norm thread's slice
  using StageVec = AlignedVector<BFloat16x2, 4>;  // 8 elements: the push kernel's 16B vector
  using Lamport = distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;

  const uint32_t tid = threadIdx.x;
  const uint32_t row = blockIdx.x;
  const uint32_t branch = blockIdx.y;  // the rank inside the row's cluster
  if (params.trigger == kTriggerEntry) PDLTriggerSecondary<kUsePDL>();

  if (row == params.num_rows) {
    // bumper: keep the counters no row uses at the plane's parity
    PDLWaitPrimary<kUsePDL>();
    if (branch == 0) {
      const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};
      __syncthreads();
      epoch.unsafe_flip_range(row, params.num_push_counters);
    }
    PDLTriggerSecondary<kUsePDL>();
    return;
  }
  __shared__ uint4 reduced_raw[kRowVecs];  // the all-reduced row, filled by all four CTAs
  const auto reduced_row = reinterpret_cast<StageVec*>(reduced_raw);
  __shared__ float warp_sums[kWarpThreads];
  const auto gmem = tile::Memory<Storage>::cta(kThreads);

  // The norm weight is a parameter: no producer writes it.
  const auto weight_values = gmem.load(params.weight + branch * kHidden, 0);

  // The epoch is read after the wait: a preceding all-reduce on this plane may
  // still be flipping it.
  PDLWaitPrimary<kUsePDL>();
  if (params.trigger == kTriggerWait) PDLTriggerSecondary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};
  const uint32_t r = params.rank;
  const bool reduces = tid < kVecsPerCta;
  const uint32_t hvec = branch * kVecsPerCta + tid;  // vector within the row
  const uint32_t vid = row * kRowVecs + hvec;        // vector within the [rows, 2560] view: the slot address

  // push this rank's quarter-row to every peer (custom_all_reduce.cuh, 1shot push)
  if (reduces) {
    StageVec staged;
    staged.load(params.input + row * kHidden, hvec);
    Lamport::clear_pos_zero(staged.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_relaxed_16B(staged, epoch.slot_ptr(/*dst=*/i, /*src=*/r), vid);
    }
  }
  // Peers may write a CTA's shared memory only once it runs (CUDA guide, distributed
  // shared memory); arrive past the push, wait before the poll that precedes those writes.
  cluster_arrive_relaxed();

  // the epilogue's own inputs, loaded while the peers' vectors are in flight
  const auto residual_values = gmem.load(params.residual + (row * kBranches + branch) * kHidden, 0);
  float total = 0.0f;
#pragma unroll
  for (uint32_t split = 0; split < 8; ++split)
    total += params.partials[(row * 8 + split) * kBranches + branch];
  const float gate = 2.0f / (1.0f + math::exp(-total / 4));

  // every CTA of the cluster runs, so its shared memory may be written; off the
  // critical path, as the peers' vectors are still in flight
  cluster_wait();

  // poll, reduce in rank order, and hand the vector to every CTA of the cluster
  const auto cluster = cg::this_cluster();
  if (reduces) {
    StageVec vec[kWorldSize];
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::ld_relaxed_16B(vec[i], epoch.slot_ptr(/*dst=*/r, /*src=*/i), vid);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        has_zero |= Lamport::has_pos_zero(vec[i].data());
      }
      if (!has_zero) break;
    } while (true);
    const auto out_vec = reduce_vec(vec);
#pragma unroll
    for (uint32_t c = 0; c < kBranches; ++c) {
      *cluster.map_shared_rank(&reduced_row[hvec], c) = out_vec;
    }
  }
  // The row is complete in every CTA, and every CTA has read the epoch. No
  // remote shared memory access follows, so any CTA may exit after this.
  cluster.sync();
  PDLTriggerSecondary<kUsePDL>();  // kTriggerReduced; a repeat is a no-op

  // hc_combine_norm_decode_kernel's epilogue, on the reduced row
  Storage block_values;
  reinterpret_cast<StageVec*>(&block_values)[0] = reduced_row[2 * tid];
  reinterpret_cast<StageVec*>(&block_values)[1] = reduced_row[2 * tid + 1];
  Storage combined;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [rx, ry] = cast<fp32x2_t>(residual_values[i]);
    const auto [yx, yy] = cast<fp32x2_t>(block_values[i]);
    combined[i] = cast<BFloat16x2>(fp32x2_t{rx + gate * yx, ry + gate * yy});
  }
  float sum = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [x, y] = cast<fp32x2_t>(combined[i]);
    sum += x * x + y * y;
  }
  sum = warp::reduce_sum(sum);
  const auto warp_id = tid / kWarpThreads;
  warp_sums[warp_id] = sum;
  __syncthreads();
  if (warp_id == 0) {
    const auto local_sum = tid < kWarps ? warp_sums[tid] : 0.0f;
    sum = warp::reduce_sum(local_sum);
    warp_sums[tid] = math::rsqrt(sum / 2560 + params.eps);
  }
  __syncthreads();
  const float inv_rms = warp_sums[warp_id];
  Storage normalized;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto [x, y] = cast<fp32x2_t>(combined[i]);
    const auto [wx, wy] = cast<fp32x2_t>(weight_values[i]);
    normalized[i] = cast<BFloat16x2>(fp32x2_t{x * inv_rms * (1.0f + wx), y * inv_rms * (1.0f + wy)});
  }
  gmem.store(params.combined + (row * kBranches + branch) * kHidden, combined, 0);
  gmem.store(params.normalized + (row * kBranches + branch) * kHidden, normalized, 0);

  // re-establish the empty markers for the next same-phase round
  if (reduces) {
    StageVec zero_vec;
    Lamport::fill_pos_zero(zero_vec.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_global_16B(zero_vec, epoch.slot_ptr(/*dst=*/r, /*src=*/i), vid);
    }
  }
  if (branch == 0) epoch.flip();
}

template <uint32_t kWorldSize, bool kUsePDL>
struct HcAllReduceCombineNorm {
  using TensorView = tvm::ffi::TensorView;

  static void
  run(host::distributed::CommunicatorRef ref,
      TensorView input,
      TensorView residual,
      TensorView partials,
      TensorView weight,
      TensorView combined,
      TensorView normalized,
      double eps,
      int64_t trigger) {
    using namespace host;
    using namespace hc_ar;
    const auto& comm = *ref.get();
    const auto& push = comm.get_push_obj();
    CHECK_HOST(push.world_size == kWorldSize)
        << "communicator holds " << push.world_size << " ranks, kernel built for " << kWorldSize;

    auto M = SymbolicSize{"rows"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({M, kHidden})
        .with_strides({kHidden, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(input);
    TensorMatcher({M, kBranches * kHidden})
        .with_strides({kBranches * kHidden, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(residual)
        .verify(combined)
        .verify(normalized);
    TensorMatcher({M, 8, kBranches})
        .with_strides({8 * kBranches, kBranches, 1})
        .with_dtype<fp32_t>()
        .with_device<kDLCUDA>(device)
        .verify(partials);
    TensorMatcher({kBranches * kHidden}).with_strides({1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(
        weight);

    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(num_rows > 0 && num_rows <= kMaxRows) << "rows = " << num_rows << " outside 1.." << kMaxRows;
    for (const auto* ptr : {input.data_ptr(), residual.data_ptr(), combined.data_ptr(), normalized.data_ptr()}) {
      CHECK_HOST(reinterpret_cast<uintptr_t>(ptr) % 32 == 0) << "tensors must be 32B aligned";
    }
    CHECK_HOST(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 32 == 0) << "weight must be 32B aligned";
    // the [rows, 2560] view is staged by vector index, so it must fit one push slot
    const int64_t nbytes = int64_t(num_rows) * kHidden * sizeof(bf16_t);
    CHECK_HOST(nbytes <= push.slot_bytes)
        << nbytes << " bytes exceed the " << push.slot_bytes << "-byte push slot of the plane";
    CHECK_HOST(num_rows <= push.num_blocks)
        << "rows = " << num_rows << " exceeds the " << push.num_blocks << " push phase counters of the plane";
    CHECK_HOST(trigger == kTriggerEntry || trigger == kTriggerWait || trigger == kTriggerReduced) << trigger;
    // one cluster and phase counter per row, plus the bumper when counters are left over
    const uint32_t num_clusters = num_rows + (num_rows < push.num_blocks ? 1 : 0);

    const auto params = HcAllReduceCombineNormParams<kWorldSize>{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .residual = static_cast<const bf16_t*>(residual.data_ptr()),
        .partials = static_cast<const float*>(partials.data_ptr()),
        .weight = static_cast<const bf16_t*>(weight.data_ptr()),
        .combined = static_cast<bf16_t*>(combined.data_ptr()),
        .normalized = static_cast<bf16_t*>(normalized.data_ptr()),
        .eps = static_cast<float>(eps),
        .rank = push.rank,
        .num_rows = num_rows,
        .num_push_counters = push.num_blocks,
        .trigger = static_cast<uint32_t>(trigger),
        .ws = push.get_workspace<kWorldSize>(nbytes),
    };
    // __cluster_dims__(1, 4, 1) is compiled in: a plain launch forms the clusters along y
    LaunchKernel(dim3(num_clusters, kBranches), kThreads, input.device())
        .enable_pdl(kUsePDL)(hc_all_reduce_combine_norm_kernel<kWorldSize, kUsePDL>, params);
  }
};

}  // namespace sglang
