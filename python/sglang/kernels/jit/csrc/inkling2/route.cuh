// SPDX-License-Identifier: Apache-2.0
// Stable sort of routed expert ids by expert, and the grouped GEMM metadata of
// one rank's experts, in one CTA. Equal to the Triton _route_metadata in route.py.

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::inkling2_route {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxIds = 4096;
constexpr int kRounds = kMaxIds / kThreads;

// Exclusive scan of values[0, count) in place (count <= kThreads); returns the total.
__device__ __forceinline__ int block_exclusive_scan(int* values, int count, int* warp_sums) {
  int const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int v = threadIdx.x < count ? values[threadIdx.x] : 0;
  int x = v;
#pragma unroll
  for (int s = 1; s < 32; s *= 2) {
    int y = __shfl_up_sync(0xFFFFFFFF, x, s);
    if (lane >= s) x += y;
  }
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = warp_sums[lane];
#pragma unroll
    for (int s = 1; s < 32; s *= 2) {
      int y = __shfl_up_sync(0xFFFFFFFF, w, s);
      if (lane >= s) w += y;
    }
    warp_sums[lane] = w;
  }
  __syncthreads();
  int total = warp_sums[kWarps - 1];
  if (threadIdx.x < count) values[threadIdx.x] = x - v + (warp ? warp_sums[warp - 1] : 0);
  __syncthreads();
  return total;
}

// Shared memory of route_metadata_kernel for n ids and E experts.
struct Layout {
  int chunks, stride;  // 32-id chunks; experts per chunk row, rounded to 8
  size_t counts, expert_start, sorted, dst2src, gammas, bounds, bytes;
  __host__ __device__ Layout(int n, int experts, int num_local) {
    chunks = (n + 31) / 32;
    stride = (experts + 7) / 8 * 8;
    auto round = [](size_t v) { return (v + 15) / 16 * 16; };
    counts = 0;
    expert_start = round(counts + size_t(chunks) * stride * 2);
    sorted = round(expert_start + (experts + 1) * 4);
    dst2src = round(sorted + n * 4);
    gammas = round(dst2src + n * 4);
    bounds = round(gammas + n * 4);
    bytes = round(bounds + (num_local + 1) * 4);
  }
};

// Lanes of the warp holding the same expert id (ids below 1024).
__device__ __forceinline__ unsigned same_lanes(int expert) {
  unsigned same = 0xFFFFFFFF;
#pragma unroll
  for (int bit = 0; bit < 10; ++bit) {
    bool const set = (expert >> bit) & 1;
    unsigned const lanes = __ballot_sync(0xFFFFFFFF, set);
    same &= set ? lanes : ~lanes;
  }
  return same;
}

// Stable counting sort: each 32-id chunk ranks its ids per expert and records
// per-expert chunk counts in counts[chunk][expert]; a prefix over chunks per
// expert and a scan over experts give every id its sorted position.
__global__ void __launch_bounds__(kThreads) route_metadata_kernel(
    int const* __restrict__ ids,
    float const* __restrict__ weights,
    int* sorted_ids,
    int* src2dst,
    int* dst2src,
    float* gammas,
    int* counts,
    int* offsets,
    int* block_offsets,
    int* schedule,
    int n,
    int start,
    int num_local,
    int num_experts,
    int max_blocks,
    int block_m) {
  extern __shared__ __align__(16) unsigned char smem[];
  Layout const layout(n, num_experts, num_local);
  int const chunks = layout.chunks, stride = layout.stride;
  uint16_t* chunk_counts = reinterpret_cast<uint16_t*>(smem + layout.counts);
  int* expert_start = reinterpret_cast<int*>(smem + layout.expert_start);
  int* sorted_s = reinterpret_cast<int*>(smem + layout.sorted);
  int* dst2src_s = reinterpret_cast<int*>(smem + layout.dst2src);
  float* gammas_s = reinterpret_cast<float*>(smem + layout.gammas);
  int* bounds = reinterpret_cast<int*>(smem + layout.bounds);
  __shared__ int warp_sums[kWarps];
  int const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int i = threadIdx.x; i < chunks * stride / 8; i += kThreads)
    reinterpret_cast<uint4*>(chunk_counts)[i] = make_uint4(0, 0, 0, 0);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  int expert[kRounds], rank[kRounds];
  float weight[kRounds];
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    int const p = r * kThreads + threadIdx.x;
    expert[r] = p < n ? ids[p] : 1023;
    weight[r] = p < n ? weights[p] : 0.0f;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    if (r * kThreads + warp * 32 >= n) break;
    unsigned const same = same_lanes(expert[r]);
    rank[r] = __popc(same & ((1u << lane) - 1));
    if (expert[r] < num_experts && lane == __ffs(same) - 1)
      chunk_counts[(r * kWarps + warp) * stride + expert[r]] = __popc(same);
  }
  __syncthreads();
  for (int e = threadIdx.x; e < num_experts; e += kThreads) {
    int sum = 0;
#pragma unroll 8
    for (int c = 0; c < chunks; ++c) {
      int const count = chunk_counts[c * stride + e];
      chunk_counts[c * stride + e] = uint16_t(sum);
      sum += count;
    }
    expert_start[e] = sum;
  }
  __syncthreads();
  for (int base = 0, chunk = 0; chunk < num_experts; chunk += kThreads) {
    int const count = min(kThreads, num_experts - chunk);
    int const total = block_exclusive_scan(expert_start + chunk, count, warp_sums);
    if (threadIdx.x < count) expert_start[chunk + threadIdx.x] += base;
    __syncthreads();
    base += total;
    if (chunk + kThreads >= num_experts && threadIdx.x == 0) expert_start[num_experts] = base;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    int const p = r * kThreads + threadIdx.x;
    if (p < n) {
      int const e = expert[r];
      int const dst = expert_start[e] + chunk_counts[p / 32 * stride + e] + rank[r];
      sorted_s[dst] = e;
      dst2src_s[dst] = p;
      gammas_s[dst] = weight[r];
      src2dst[p] = dst;
    }
  }
  for (int e = threadIdx.x; e <= num_local; e += kThreads) {
    int const offset = expert_start[min(start + e, num_experts)];
    offsets[e] = offset;
    bounds[e] = offset;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += kThreads) {
    sorted_ids[i] = sorted_s[i];
    dst2src[i] = dst2src_s[i];
    gammas[i] = gammas_s[i];
  }
  int c = 0;
  if (threadIdx.x < num_local) {
    c = bounds[threadIdx.x + 1] - bounds[threadIdx.x];
    counts[threadIdx.x] = c;
  }
  __syncthreads();
  if (threadIdx.x < num_local) bounds[threadIdx.x] = (c + block_m - 1) / block_m;
  __syncthreads();
  int const total = block_exclusive_scan(bounds, num_local, warp_sums);
  if (threadIdx.x < num_local) block_offsets[threadIdx.x] = bounds[threadIdx.x];
  if (threadIdx.x == 0) {
    block_offsets[num_local] = total;
    bounds[num_local] = total;
  }
  __syncthreads();
  for (int slot = threadIdx.x; slot < max_blocks; slot += kThreads) {
    int lo = 0, hi = num_local;
    while (lo < hi) {
      int mid = (lo + hi + 1) / 2;
      if (bounds[mid] > slot)
        hi = mid - 1;
      else
        lo = mid;
    }
    schedule[slot] = slot < total ? ((slot - bounds[lo]) << 16) + lo : -1;
  }
}

void route_metadata(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView sorted_ids,
    tvm::ffi::TensorView src2dst,
    tvm::ffi::TensorView dst2src,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView block_offsets,
    tvm::ffi::TensorView schedule,
    int64_t start,
    int64_t num_experts,
    int64_t block_m) {
  int n = ids.numel(), num_local = counts.numel();
  host::RuntimeCheck(n <= kMaxIds && start + num_local <= num_experts, "Unsupported route metadata shape");
  host::RuntimeCheck(num_local < kThreads && num_experts < 1023, "Unsupported route metadata shape");
  size_t const smem = Layout(n, int(num_experts), num_local).bytes;
  auto kernel = route_metadata_kernel;
  if (smem > 48 * 1024)
    host::RuntimeCheck(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)) == cudaSuccess,
        "Cannot set route metadata shared memory");
  host::LaunchKernel(1, kThreads, ids.device(), smem)
      .enable_pdl(true)(
          kernel,
          static_cast<int const*>(ids.data_ptr()),
          static_cast<float const*>(weights.data_ptr()),
          static_cast<int*>(sorted_ids.data_ptr()),
          static_cast<int*>(src2dst.data_ptr()),
          static_cast<int*>(dst2src.data_ptr()),
          static_cast<float*>(gammas.data_ptr()),
          static_cast<int*>(counts.data_ptr()),
          static_cast<int*>(offsets.data_ptr()),
          static_cast<int*>(block_offsets.data_ptr()),
          static_cast<int*>(schedule.data_ptr()),
          n,
          int(start),
          num_local,
          int(num_experts),
          int(schedule.numel()),
          int(block_m));
}

}  // namespace sglang::inkling2_route
