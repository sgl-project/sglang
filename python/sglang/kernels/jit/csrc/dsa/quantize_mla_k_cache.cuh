#pragma once

// Quantizer producing separate NoPE and RoPE staging tensors.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace sglang {

constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kBlockThreads = kWarpsPerBlock * device::kWarpThreads;
constexpr float kInvFp8Max = 1.0f / kFP8E4M3Max;
// Persistent remainders are below INT32_MAX; reserve this value for the split grid.
constexpr uint32_t kSplitGeometrySentinel = UINT32_MAX;

struct QuantizeKCacheFastSeparateCudaParams {
  const void* __restrict__ k_nope;
  const void* __restrict__ k_rope;
  void* __restrict__ nope_part;
  void* __restrict__ rope_part;
  int64_t stride_nope_bytes;
  int64_t stride_rope_bytes;
  int64_t stride_nope_part_bytes;
  int64_t stride_rope_part_bytes;
  uint64_t groups_per_cta;
  uint32_t extra_group_ctas;
};

SGL_DEVICE float quant_fp8_clip(float value) {
  namespace math = device::math;
  return math::max(math::min(value, kFP8E4M3Max), -kFP8E4M3Max);
}

SGL_DEVICE fp8x2_e4m3_t quant_pack_fp8(float x, float y) {
  return fp8x2_e4m3_t{fp32x2_t{quant_fp8_clip(x), quant_fp8_clip(y)}};
}

// One warp owns one group. Derive its vector width and unrolled passes from
// the group size; keep stores at most four bytes wide so every scale-padded
// output row stays aligned, including rows with only one scale value.
template <typename KeyT, int64_t kNopeDim, int64_t kGroupSize>
SGL_DEVICE void quantize_single_group_warp(
    const void* __restrict__ nope_row, void* __restrict__ nope_part_row, uint32_t group_id, uint32_t lane_id) {
  using namespace device;

  static_assert(std::is_same_v<KeyT, bf16_t>);
  constexpr uint32_t kPairsPerLane = std::min(int64_t{2}, div_ceil(kGroupSize, int64_t{2 * kWarpThreads}));
  constexpr uint32_t kLanesPerGroup = std::min(int64_t{kWarpThreads}, kGroupSize / (2 * kPairsPerLane));
  constexpr uint32_t kVectorsPerGroup = kGroupSize / (2 * kPairsPerLane);
  constexpr uint32_t kPasses = kVectorsPerGroup / kLanesPerGroup;
  using InputStorage = AlignedVector<packed_t<KeyT>, kPairsPerLane>;
  using OutputStorage = AlignedVector<fp8x2_e4m3_t, kPairsPerLane>;
  // Full-warp groups compile out the lane predicate, including the split CTA.
  const bool active = kLanesPerGroup == kWarpThreads || lane_id < kLanesPerGroup;
  fp32x2_t values[kPasses][kPairsPerLane];
  float local_abs_max = 0.0f;
  if (active) {
#pragma unroll
    for (uint32_t pass = 0; pass < kPasses; ++pass) {
      const uint32_t vector_index = group_id * kVectorsPerGroup + pass * kLanesPerGroup + lane_id;
      InputStorage input;
      input.load(nope_row, vector_index);
#pragma unroll
      for (uint32_t pair = 0; pair < kPairsPerLane; ++pair) {
        values[pass][pair] = cast<fp32x2_t>(input[pair]);
        local_abs_max = fmaxf(local_abs_max, fabsf(values[pass][pair].x));
        local_abs_max = fmaxf(local_abs_max, fabsf(values[pass][pair].y));
      }
    }
  }
  // Reduce within the live lane group; all warp lanes still execute the shuffles.
  const float abs_max = warp::reduce_max<kLanesPerGroup>(local_abs_max);
  const float scale = abs_max * kInvFp8Max;
  const float inv_scale = __fdividef(1.0f, scale);
  if (active) {
#pragma unroll
    for (uint32_t pass = 0; pass < kPasses; ++pass) {
      OutputStorage output;
#pragma unroll
      for (uint32_t pair = 0; pair < kPairsPerLane; ++pair) {
        output[pair] = quant_pack_fp8(values[pass][pair].x * inv_scale, values[pass][pair].y * inv_scale);
      }
      const uint32_t vector_index = group_id * kVectorsPerGroup + pass * kLanesPerGroup + lane_id;
      output.store(nope_part_row, vector_index);
    }
  }
  if (lane_id == 0) {
    auto* scale_ptr = static_cast<float*>(pointer::offset(nope_part_row, kNopeDim));
    scale_ptr[group_id] = scale;
  }
}

// Pass fields separately: a by-value aggregate adds argument-memory loads on PPU.
template <typename KeyT, int64_t kNopeDim, int64_t kRopeDim, int64_t kGroupSize, bool kUsePDL>
__global__ void quantize_k_cache_fast_separate_cuda_kernel(
    const void* k_nope,
    const void* k_rope,
    void* nope_part,
    void* rope_part,
    int64_t stride_nope_bytes,
    int64_t stride_rope_bytes,
    int64_t stride_nope_part_bytes,
    int64_t stride_rope_part_bytes,
    uint64_t groups_per_cta,
    uint32_t extra_group_ctas) {
  using namespace device;

  constexpr uint32_t kNumGroups = kNopeDim / kGroupSize;
  constexpr uint32_t kRopePairs = kRopeDim / 2;
  constexpr uint32_t kRopePairsPerGroup = kRopePairs / kNumGroups;
  constexpr uint32_t kExtraRopeGroups = kRopePairs % kNumGroups;
  // Keep row-local offsets in 32-bit bytes, allowing reuse of the NoPE group
  // offset and loop-invariant lane scaling in the default layout.
  constexpr uint32_t kRopePairBytes = sizeof(packed_t<KeyT>);
  constexpr uint32_t kRopeGroupBytes = kRopePairsPerGroup * kRopePairBytes;
  // Small and large token counts share this entry with different launch geometry.
  if (extra_group_ctas == kSplitGeometrySentinel) {
    const int32_t token_id = static_cast<int32_t>(blockIdx.x);
    const uint32_t group_id = blockIdx.y;
    const uint32_t lane_id = threadIdx.x;
    PDLWaitPrimary<kUsePDL>();
    if (group_id < kNumGroups) {
      const auto nope_row = pointer::offset(k_nope, token_id * static_cast<int32_t>(stride_nope_bytes));
      const auto nope_part_row = pointer::offset(nope_part, token_id * static_cast<int32_t>(stride_nope_part_bytes));
      quantize_single_group_warp<KeyT, kNopeDim, kGroupSize>(nope_row, nope_part_row, group_id, lane_id);
    } else {
      const auto src = pointer::offset(k_rope, token_id * static_cast<int32_t>(stride_rope_bytes));
      const auto dst = pointer::offset(rope_part, token_id * static_cast<int32_t>(stride_rope_part_bytes));
#pragma unroll
      for (uint32_t pair_base = 0; pair_base < kRopePairs; pair_base += kWarpThreads) {
        const uint32_t pair_id = pair_base + lane_id;
        // Full-warp chunks need no lane predicate.
        if (kRopePairs % kWarpThreads == 0 || pair_id < kRopePairs) {
          AlignedVector<packed_t<KeyT>, 1> rope;
          rope.load(pointer::offset(src, pair_id * kRopePairBytes));
          rope.store(pointer::offset(dst, pair_id * kRopePairBytes));
        }
      }
    }
    PDLTriggerSecondary<kUsePDL>();
    return;
  }
  // Use the same group helper as the small geometry.
  // One warp owns one group; the CTA advances by four groups per iteration.
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint64_t begin = static_cast<uint64_t>(blockIdx.x) * groups_per_cta + min(blockIdx.x, extra_group_ctas);
  const uint64_t end = begin + groups_per_cta + (blockIdx.x < extra_group_ctas);
  PDLWaitPrimary<kUsePDL>();
  for (uint64_t base = begin; base < end; base += kWarpsPerBlock) {
    const uint64_t group = base + warp_id;
    // Validity is uniform within each full warp, so no subgroup ballot is needed.
    if (group < end) {
      const uint64_t token_id = group / kNumGroups;
      const uint32_t group_id = static_cast<uint32_t>(group % kNumGroups);
      const auto nope_row = pointer::offset(k_nope, token_id * stride_nope_bytes);
      const auto nope_part_row = pointer::offset(nope_part, token_id * stride_nope_part_bytes);
      quantize_single_group_warp<KeyT, kNopeDim, kGroupSize>(nope_row, nope_part_row, group_id, lane_id);
      // Give each group a fixed number of pairs, plus one for the first
      // remainder groups. Exact divisions compile to a constant lane bound.
      const uint32_t num_pairs = kRopePairsPerGroup + (group_id < kExtraRopeGroups);
      if (lane_id < num_pairs) {
        const uint32_t pair_byte_offset =
            group_id * kRopeGroupBytes + min(group_id, kExtraRopeGroups) * kRopePairBytes + lane_id * kRopePairBytes;
        const auto src = pointer::offset(k_rope, token_id * stride_rope_bytes + pair_byte_offset);
        const auto dst = pointer::offset(rope_part, token_id * stride_rope_part_bytes + pair_byte_offset);
        AlignedVector<packed_t<KeyT>, 1> rope;
        rope.load(src);
        rope.store(dst);
      }
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <typename KeyT, int64_t kNopeDim, int64_t kRopeDim, int64_t kGroupSize, bool kUsePDL>
struct QuantizeKCacheFastSeparateCudaKernel {
  static constexpr int64_t kNumGroups = kNopeDim / kGroupSize;
  static constexpr int64_t kNopePartBytes = kNopeDim + kNumGroups * sizeof(float);
  static constexpr int64_t kRopePartBytes = kRopeDim * sizeof(KeyT);
  static constexpr auto kernel =
      quantize_k_cache_fast_separate_cuda_kernel<KeyT, kNopeDim, kRopeDim, kGroupSize, kUsePDL>;

  static void
  run(tvm::ffi::TensorView nope_part,
      tvm::ffi::TensorView rope_part,
      tvm::ffi::TensorView k_nope,
      tvm::ffi::TensorView k_rope) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto SNope = SymbolicSize{"nope_stride"};
    auto SRope = SymbolicSize{"rope_stride"};
    auto SNopePart = SymbolicSize{"nope_part_stride"};
    auto SRopePart = SymbolicSize{"rope_part_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, kNopeDim})  // k_nope
        .with_strides({SNope, 1})
        .with_dtype<KeyT>()
        .with_device(device)
        .verify(k_nope);
    TensorMatcher({N, kRopeDim})  // k_rope
        .with_strides({SRope, 1})
        .with_dtype<KeyT>()
        .with_device(device)
        .verify(k_rope);
    TensorMatcher({N, kNopePartBytes})  // nope_part
        .with_strides({SNopePart, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(nope_part);
    TensorMatcher({N, kRopePartBytes})  // rope_part
        .with_strides({SRopePart, 1})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(rope_part);

    const uint64_t total_groups = static_cast<uint64_t>(N.unwrap()) * kNumGroups;
    if (total_groups == 0) return;

    // The Python entry point selects the input device before querying occupancy.
    // Do not cache a device-dependent budget in a process-wide static variable.
    const auto sm_count = static_cast<uint64_t>(runtime::get_sm_count(device.unwrap().device_id));
    const auto blocks_per_sm = runtime::get_blocks_per_sm(kernel, kBlockThreads);
    const auto small_blocks_per_sm = runtime::get_blocks_per_sm(kernel, device::kWarpThreads);
    RuntimeCheck(blocks_per_sm > 0 && small_blocks_per_sm > 0, "MLA K-cache kernel has no resident CTA capacity");
    // Bound the grid's i32 indices and reserve UINT32_MAX as the split sentinel.
    constexpr uint64_t kCtaLimit = std::numeric_limits<int32_t>::max();
    const auto cta_budget = std::min(sm_count * blocks_per_sm, kCtaLimit);
    const auto small_cta_budget = std::min(sm_count * small_blocks_per_sm, kCtaLimit);
    // Cap the grid by the number of four-group tiles as well as the occupancy budget.
    const uint64_t ctas_for_one_iteration = (total_groups - 1) / kWarpsPerBlock + 1;
    const auto num_blocks = static_cast<uint32_t>(std::min(ctas_for_one_iteration, cta_budget));

    const auto params = QuantizeKCacheFastSeparateCudaParams{
        .k_nope = k_nope.data_ptr(),
        .k_rope = k_rope.data_ptr(),
        .nope_part = nope_part.data_ptr(),
        .rope_part = rope_part.data_ptr(),
        .stride_nope_bytes = SNope.unwrap() * static_cast<int64_t>(sizeof(KeyT)),
        .stride_rope_bytes = SRope.unwrap() * static_cast<int64_t>(sizeof(KeyT)),
        .stride_nope_part_bytes = SNopePart.unwrap(),
        .stride_rope_part_bytes = SRopePart.unwrap(),
        .groups_per_cta = total_groups / num_blocks,
        .extra_group_ctas = static_cast<uint32_t>(total_groups % num_blocks),
    };

    // One CTA per quantization group plus one RoPE CTA per token must all fit.
    // Division avoids overflow and bounds blockIdx.x for its i32 cast.
    constexpr uint32_t kCtasPerToken = kNumGroups + 1;
    if (static_cast<uint64_t>(N.unwrap()) <= small_cta_budget / kCtasPerToken && params.stride_nope_bytes >= 0) {
      const dim3 grid(static_cast<uint32_t>(N.unwrap()), kCtasPerToken);
      LaunchKernel(grid, device::kWarpThreads, device.unwrap())
          .enable_pdl(kUsePDL)(
              kernel,
              params.k_nope,
              params.k_rope,
              params.nope_part,
              params.rope_part,
              params.stride_nope_bytes,
              params.stride_rope_bytes,
              params.stride_nope_part_bytes,
              params.stride_rope_part_bytes,
              params.groups_per_cta,
              kSplitGeometrySentinel);
      return;
    }
    LaunchKernel(num_blocks, kBlockThreads, device.unwrap())
        .enable_pdl(kUsePDL)(
            kernel,
            params.k_nope,
            params.k_rope,
            params.nope_part,
            params.rope_part,
            params.stride_nope_bytes,
            params.stride_rope_bytes,
            params.stride_nope_part_bytes,
            params.stride_rope_part_bytes,
            params.groups_per_cta,
            params.extra_group_ctas);
  }
};

}  // namespace sglang
