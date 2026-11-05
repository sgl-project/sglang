// SPDX-License-Identifier: Apache-2.0
// Beam's BF16 sandwich norm over the existing CustomAllReduceV2 push protocol.
#pragma once
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/extra/stl.h>

#include <cooperative_groups.h>

namespace sglang {
namespace beam_ar {
using Vec = device::AlignedVector<bf16x2_t, 4>;
using Lamport = device::distributed::LamportTrait<bf16_t, 8, 4>;

template <uint32_t World>
struct Params {
  const bf16_t *input, *residual, *post_weight, *next_weight;
  bf16_t *combined, *normalized;
  float eps, x_scale, residual_scale, routed_scale;
  uint32_t rows, rank, counters;
  device::distributed::PushWorkSpace<World> workspace;
};

SGL_DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
SGL_DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

/// Every thread holds eight adjacent BF16 values of one row.
template <uint32_t Hidden, uint32_t Cluster>
SGL_DEVICE float inv_rms(const Vec& v, float eps) {
  using namespace device;
  constexpr uint32_t warps = Hidden / 8 / Cluster / 32;
  __shared__ float partial[Cluster][warps];
  float sum = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    const auto f = cast<fp32x2_t>(v[j]);
    sum += f.x * f.x;
    sum += f.y * f.y;
  }
  sum = warp::reduce_sum(sum);
  const auto lane = threadIdx.x % 32;
  if constexpr (Cluster > 1) {
    const auto cluster = cooperative_groups::this_cluster();
    if (lane < Cluster) *cluster.map_shared_rank(&partial[blockIdx.y][threadIdx.x / 32], lane) = sum;
    cluster.sync();
  } else {
    if (lane == 0) partial[0][threadIdx.x / 32] = sum;
    __syncthreads();
  }
  sum = lane < Cluster * warps ? partial[lane / warps][lane % warps] : 0.0f;
  sum = warp::reduce_sum(sum);
  if constexpr (Cluster > 1)
    cooperative_groups::this_cluster().sync();
  else
    __syncthreads();
  return rsqrtf(sum / float(Hidden) + eps);
}

template <uint32_t World, uint32_t Hidden, uint32_t Cluster, uint32_t TopK, bool NextNorm>
__global__ void __launch_bounds__(Hidden / 8 / Cluster) __cluster_dims__(1, Cluster, 1)
    kernel(const __grid_constant__ Params<World> p) {
  using namespace device;
  namespace ptx = device::ptx;
  device::PDLWaitPrimary<true>();
  const auto row = blockIdx.x;
  const auto tx = blockIdx.y * blockDim.x + threadIdx.x;
  const auto epoch = distributed::PushEpoch<World>{p.workspace};
  if (row == p.rows) {
    // Keep every counter in phase when interleaving with generic all-reduces.
    __syncthreads();
    if (blockIdx.y == 0) epoch.unsafe_flip_range(row, p.counters);
    device::PDLTriggerSecondary<true>();
    return;
  }
  // inv_rms writes peers' shared memory, which is valid only once every CTA in the cluster has started.
  if constexpr (Cluster > 1) cluster_arrive_relaxed();
  const auto idx = row * (Hidden / 8) + tx;
  Vec value;
  if constexpr (TopK == 1) {
    value.load(p.input, idx);
  } else {
    fp32x2_t sum[4]{};
#pragma unroll
    for (uint32_t k = 0; k < TopK; ++k) {
      Vec expert;
      expert.load(p.input, (row * TopK + k) * (Hidden / 8) + tx);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        auto v = cast<fp32x2_t>(expert[j]);
        sum[j].x += v.x;
        sum[j].y += v.y;
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j)
      value[j] = cast<bf16x2_t>(fp32x2_t{sum[j].x * p.routed_scale, sum[j].y * p.routed_scale});
  }
  Lamport::clear_pos_zero(value.data());
  if (p.workspace.mc_workspace != nullptr) {
    ptx::st_multimem_16B(value, p.workspace.mc_workspace + epoch.slot_offset(p.rank), idx);
  } else {
#pragma unroll
    for (uint32_t rank = 0; rank < World; ++rank)
      ptx::st_relaxed_16B(value, epoch.slot_ptr(rank, p.rank), idx);
  }
  // Fetch epilogue inputs while the peers publish; asm loads stay ahead of the poll.
  Vec post, residual;
  ptx::ld_relaxed_16B(post, p.post_weight, tx);
  ptx::ld_relaxed_16B(residual, p.residual, idx);
  Vec peers[World];
  bool pending;
  do {
    pending = false;
#pragma unroll
    for (uint32_t rank = 0; rank < World; ++rank) {
      ptx::ld_relaxed_16B(peers[rank], epoch.slot_ptr(p.rank, rank), idx);
      pending |= Lamport::has_pos_zero(peers[rank].data());
    }
  } while (pending);
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    fp32x2_t sum{0.0f, 0.0f};
#pragma unroll
    for (uint32_t rank = 0; rank < World; ++rank) {
      const auto x = cast<fp32x2_t>(peers[rank][j]);
      sum.x += x.x;
      sum.y += x.y;
    }
    value[j] = cast<bf16x2_t>(sum);
  }
  if constexpr (Cluster > 1) cluster_wait();
  const float post_rstd = inv_rms<Hidden, Cluster>(value, p.eps);
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    auto v = cast<fp32x2_t>(value[j]);
    const auto w = cast<fp32x2_t>(post[j]);
    v = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{v.x * post_rstd * w.x, v.y * post_rstd * w.y}));
    v = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{v.x * p.x_scale, v.y * p.x_scale}));
    auto r = cast<fp32x2_t>(residual[j]);
    r = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{r.x * p.residual_scale, r.y * p.residual_scale}));
    value[j] = cast<bf16x2_t>(fp32x2_t{v.x + r.x, v.y + r.y});
  }
  ptx::st_global_16B(value, p.combined, idx);
  if constexpr (NextNorm) {
    const float rstd = inv_rms<Hidden, Cluster>(value, p.eps);
    Vec weight;
    weight.load(p.next_weight, tx);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const auto v = cast<fp32x2_t>(value[j]);
      const auto w = cast<fp32x2_t>(weight[j]);
      value[j] = cast<bf16x2_t>(fp32x2_t{v.x * rstd * w.x, v.y * rstd * w.y});
    }
    ptx::st_global_16B(value, p.normalized, idx);
  }
  Vec zero;
  Lamport::fill_pos_zero(zero.data());
#pragma unroll
  for (uint32_t rank = 0; rank < World; ++rank)
    ptx::st_global_16B(zero, epoch.slot_ptr(p.rank, rank), idx);
  // A successor must wait for completion before consuming this plane again.
  device::PDLTriggerSecondary<true>();
  if constexpr (Cluster > 1)
    cooperative_groups::this_cluster().sync();
  else
    __syncthreads();
  if (blockIdx.y == 0) epoch.flip();
}
}  // namespace beam_ar

/// Validate and launch a BF16 all-reduce + sandwich/residual/next norm.
template <uint32_t World, uint32_t Hidden, uint32_t Cluster, uint32_t TopK>
struct BeamAllReduceNorm {
  static_assert(World == 2 || World == 4 || World == 8);
  // Whole warps per CTA, and at most 32 in a cluster so inv_rms reduces them in one warp.
  static_assert(Hidden % (Cluster * 256) == 0 && Hidden <= 8192);
  static void
  run(host::distributed::CommunicatorRef ref,
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView post,
      std::optional<tvm::ffi::TensorView> next,
      tvm::ffi::TensorView combined,
      std::optional<tvm::ffi::TensorView> normalized,
      double eps,
      double x_scale,
      double residual_scale,
      double routed_scale) {
    using namespace host;
    const auto& push = ref.get()->get_push_obj();
    CHECK_HOST(push.world_size == World);
    SymbolicSize rows{"rows"};
    SymbolicDevice device;
    const auto matrix = [&](tvm::ffi::TensorView tensor) {
      TensorMatcher({rows, Hidden})
          .with_strides({Hidden, 1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .ensure_alignment(16)
          .verify(tensor);
    };
    const auto weight = [&](tvm::ffi::TensorView tensor) {
      TensorMatcher({Hidden})
          .with_strides({1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .ensure_alignment(16)
          .verify(tensor);
    };
    if constexpr (TopK == 1)
      matrix(input);
    else
      TensorMatcher({rows, TopK, Hidden})
          .with_strides({TopK * Hidden, Hidden, 1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .ensure_alignment(16)
          .verify(input);
    matrix(residual);
    matrix(combined);
    weight(post);
    CHECK_HOST(next.has_value() == normalized.has_value());
    if (next) {
      weight(*next);
      matrix(*normalized);
    }
    const auto n = rows.unwrap();
    const int64_t bytes = n * Hidden * sizeof(bf16_t);
    CHECK_HOST(n > 0 && n <= push.num_blocks && bytes <= push.slot_bytes);
    const auto p = beam_ar::Params<World>{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .residual = static_cast<const bf16_t*>(residual.data_ptr()),
        .post_weight = static_cast<const bf16_t*>(post.data_ptr()),
        .next_weight = next ? static_cast<const bf16_t*>(next->data_ptr()) : nullptr,
        .combined = static_cast<bf16_t*>(combined.data_ptr()),
        .normalized = normalized ? static_cast<bf16_t*>(normalized->data_ptr()) : nullptr,
        .eps = static_cast<float>(eps),
        .x_scale = static_cast<float>(x_scale),
        .residual_scale = static_cast<float>(residual_scale),
        .routed_scale = static_cast<float>(routed_scale),
        .rows = static_cast<uint32_t>(n),
        .rank = push.rank,
        .counters = push.num_blocks,
        .workspace = push.get_workspace<World>(bytes)};
    auto kernel = next ? beam_ar::kernel<World, Hidden, Cluster, TopK, true>
                       : beam_ar::kernel<World, Hidden, Cluster, TopK, false>;
    LaunchKernel(dim3(n + (n < push.num_blocks), Cluster), Hidden / 8 / Cluster, input.device())
        .enable_pdl(true)(kernel, p);
  }
};
}  // namespace sglang
