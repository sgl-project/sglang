#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// Up to this many norm "groups" are fused into one launch. A group is a
// contiguous run of heads (within the per-token row) that share one norm
// weight and all receive RoPE. Heads not covered by any group (e.g. the V /
// index-V regions) are simply never assigned a job and left untouched.
//
// For MiniMax-M3 the groups are, in order: Q (main), K (main), index-Q,
// index-K -- which is why 4 slots are enough. This mirrors the multi-branch
// single-launch design of fused_store_kv_index.cuh (one kernel handles main
// K/V + index K/V), here applied to GemmaRMSNorm + partial RoPE.
constexpr int kMaxGroups = 4;

struct FusedGemmaQKNormParams {
  bf16_t* __restrict__ qkv;
  const bf16_t* __restrict__ weight[kMaxGroups];  // per-group norm weight [head_dim]
  uint32_t group_offset[kMaxGroups];              // head offset of the group in the row
  uint32_t group_count[kMaxGroups];               // number of heads in the group
  uint32_t num_groups;
  uint32_t total_heads;  // sum of group_count[0..num_groups)
  const float* __restrict__ cos_sin_cache;
  const void* __restrict__ positions;  // dtype depends on PosT template
  uint32_t num_tokens;
  int64_t token_stride;
  float eps;
};

template <typename PosT, int64_t kHeadDim, int64_t kRopeDim, bool kUsePDL>
struct FusedTrait {
  static_assert(kHeadDim == 128 && kRopeDim == 64, "kernel specialized for HEAD_DIM=128, ROTARY_DIM=64");
  static constexpr uint32_t kWorkerSize = device::kWarpThreads;  // full warp per head
  SGL_DEVICE static void forward(const FusedGemmaQKNormParams& params) {
    using namespace device;
    const auto tx = threadIdx.x;
    const auto bx = blockIdx.x;
    const auto lane_id = tx % kWorkerSize;
    const auto work_id = (bx * blockDim.x + tx) / kWorkerSize;
    const auto total_heads = params.total_heads;
    if (work_id >= params.num_tokens * total_heads) return;
    const auto token_id = work_id / total_heads;
    const auto grouped_head = work_id % total_heads;

    // Resolve which group this job belongs to (num_groups <= kMaxGroups, so a
    // short scan is cheaper than any precomputed table and warp-uniform).
    uint32_t g = 0, base = 0;
    for (uint32_t i = 0; i < params.num_groups; ++i) {
      const auto cnt = params.group_count[i];
      if (grouped_head < base + cnt) {
        g = i;
        break;
      }
      base += cnt;
    }
    const auto local_head = grouped_head - base;
    const auto head_id = params.group_offset[g] + local_head;
    const auto weight = params.weight[g];
    const auto input = params.qkv + token_id * params.token_stride + head_id * kHeadDim;

    // prefetch weight and rope index
    const auto idx_0 = lane_id + 0;   // rope first half  [0,32)
    const auto idx_1 = lane_id + 32;  // rope second half [32, 64)
    const auto idx_2 = lane_id + 64;  // pass
    const auto idx_3 = lane_id + 96;  // pass
    const auto w0_bf16 = weight[idx_0];
    const auto w1_bf16 = weight[idx_1];
    const auto w2_bf16 = weight[idx_2];
    const auto w3_bf16 = weight[idx_3];
    const auto rope_idx = static_cast<const PosT*>(params.positions)[token_id];
    PDLWaitPrimary<kUsePDL>();

    // load input and compute RMS, fp32 accumulation for stability
    const auto i0_bf16 = input[idx_0];
    const auto i1_bf16 = input[idx_1];
    const auto i2_bf16 = input[idx_2];
    const auto i3_bf16 = input[idx_3];
    const auto [i0, i1] = cast<fp32x2_t>(bf16x2_t{i0_bf16, i1_bf16});
    const auto [i2, i3] = cast<fp32x2_t>(bf16x2_t{i2_bf16, i3_bf16});
    const auto ss = warp::reduce_sum(i0 * i0 + i1 * i1 + i2 * i2 + i3 * i3);
    const auto inv_rms = rsqrtf(ss / static_cast<float>(kHeadDim) + params.eps);

    // apply norm
    const auto [w0, w1] = cast<fp32x2_t>(bf16x2_t{w0_bf16, w1_bf16});
    const auto [w2, w3] = cast<fp32x2_t>(bf16x2_t{w2_bf16, w3_bf16});
    const auto n0 = i0 * inv_rms * (1.0f + w0);
    const auto n1 = i1 * inv_rms * (1.0f + w1);
    const auto n2 = i2 * inv_rms * (1.0f + w2);
    const auto n3 = i3 * inv_rms * (1.0f + w3);

    const auto cs = params.cos_sin_cache + rope_idx * kRopeDim;
    const auto cos = cs[lane_id];
    const auto sin = cs[lane_id + kRopeDim / 2];

    // apply rope to the first kRopeDim dims, and write back
    device::PDLTriggerSecondary<kUsePDL>();
    const auto o0 = n0 * cos - n1 * sin;
    const auto o1 = n1 * cos + n0 * sin;
    const auto [o0_bf16, o1_bf16] = cast<bf16x2_t>(fp32x2_t{o0, o1});
    const auto [o2_bf16, o3_bf16] = cast<bf16x2_t>(fp32x2_t{n2, n3});
    input[idx_0] = o0_bf16;
    input[idx_1] = o1_bf16;
    input[idx_2] = o2_bf16;
    input[idx_3] = o3_bf16;
  }
};

template <typename Trait>
__global__ void fused_gemma_qknorm_rope_kernel(const __grid_constant__ FusedGemmaQKNormParams params) {
  return Trait::forward(params);
}

// Multi-group fused GemmaRMSNorm + partial NeoX RoPE, in place over `qkv`.
//
// Up to kMaxGroups norm groups are passed as (weight, head offset, head count)
// triples. `w0..w3` are the per-group norm weights ([head_dim] bf16 each); the
// `offN` / `cntN` scalars give each group's head offset (within the per-token
// row) and head count. These offsets/counts are host-known constants (passed as
// scalars, never device tensors) so the launch stays CUDA-graph capturable.
// Weight slots beyond `num_groups` may be dummies (e.g. == w0); the kernel
// never reads them because `num_groups` bounds the group scan. The main Q/K and
// index-Q/index-K heads are all normed and rotated in one launch; the V /
// index-V heads, lying outside every group, are left untouched.
template <typename PosT, int64_t HEAD_DIM, int64_t ROTARY_DIM, bool kUsePDL>
void fused_gemma_qknorm_rope(
    tvm::ffi::TensorView qkv,
    tvm::ffi::TensorView w0,
    tvm::ffi::TensorView w1,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView w3,
    tvm::ffi::TensorView cos_sin_cache,
    tvm::ffi::TensorView positions,
    int64_t off0,
    int64_t cnt0,
    int64_t off1,
    int64_t cnt1,
    int64_t off2,
    int64_t cnt2,
    int64_t off3,
    int64_t cnt3,
    int64_t num_groups,
    double eps) {
  using namespace host;
  auto N = SymbolicSize{"num_tokens"};
  auto device = SymbolicDevice{};
  constexpr auto D = HEAD_DIM;
  constexpr auto R = ROTARY_DIM;
  device.set_options<kDLCUDA>();
  TensorMatcher({N, -1}).with_dtype<bf16_t>().with_device(device).verify(qkv);
  TensorMatcher({D}).with_dtype<bf16_t>().with_device(device).verify(w0).verify(w1).verify(w2).verify(w3);
  TensorMatcher({-1, R}).with_dtype<fp32_t>().with_device(device).verify(cos_sin_cache);
  TensorMatcher({N}).with_dtype<PosT>().with_device(device).verify(positions);

  RuntimeCheck(num_groups >= 1 && num_groups <= kMaxGroups);

  const tvm::ffi::TensorView weights[kMaxGroups] = {w0, w1, w2, w3};
  const int64_t offsets[kMaxGroups] = {off0, off1, off2, off3};
  const int64_t counts[kMaxGroups] = {cnt0, cnt1, cnt2, cnt3};

  auto params = FusedGemmaQKNormParams{};
  params.qkv = static_cast<bf16_t*>(qkv.data_ptr());
  params.cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr());
  params.positions = positions.data_ptr();
  params.num_tokens = static_cast<uint32_t>(N.unwrap());
  params.num_groups = static_cast<uint32_t>(num_groups);
  params.token_stride = static_cast<int64_t>(qkv.stride(0));
  params.eps = static_cast<float>(eps);

  uint32_t total_heads = 0;
  for (int64_t i = 0; i < kMaxGroups; ++i) {
    const auto cnt = (i < num_groups) ? counts[i] : 0;
    params.weight[i] = static_cast<const bf16_t*>(weights[i].data_ptr());
    params.group_offset[i] = static_cast<uint32_t>(i < num_groups ? offsets[i] : 0);
    params.group_count[i] = static_cast<uint32_t>(cnt);
    total_heads += static_cast<uint32_t>(cnt);
  }
  params.total_heads = total_heads;

  using Trait = FusedTrait<PosT, HEAD_DIM, ROTARY_DIM, kUsePDL>;
  const auto needed_threads = static_cast<int64_t>(params.num_tokens) * total_heads * device::kWarpThreads;
  RuntimeCheck(needed_threads < std::numeric_limits<uint32_t>::max());
  if (needed_threads == 0) return;
  const uint32_t block_size = 256u;
  const uint32_t num_blocks = div_ceil(static_cast<uint32_t>(needed_threads), block_size);
  LaunchKernel(num_blocks, block_size, device.unwrap())  //
      .enable_pdl(kUsePDL)(fused_gemma_qknorm_rope_kernel<Trait>, params);
}

}  // namespace
