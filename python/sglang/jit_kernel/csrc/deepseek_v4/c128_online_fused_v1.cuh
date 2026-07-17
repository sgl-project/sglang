#include <sgl_kernel/tensor.h>  // For TensorMatcher and symbolic tensor validation.
#include <sgl_kernel/utils.h>   // For RuntimeCheck and integer helpers.

#include <sgl_kernel/math.cuh>   // For device RMSNorm math.
#include <sgl_kernel/type.cuh>   // For bf16/fp32 packed types and casts.
#include <sgl_kernel/utils.cuh>  // For LaunchKernel and PDL primitives.
#include <sgl_kernel/vec.cuh>    // For aligned float2 loads and stores.
#include <sgl_kernel/warp.cuh>   // For warp reductions.

#include <sgl_kernel/deepseek_v4/compress_v2.cuh>  // For DecodePlan validation and layout.
#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>    // For FlashMLA UE8M0/FP8 packing.

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cmath>
#include <cstdint>

namespace {

using PlanD = device::compress::DecodePlan;
using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

constexpr uint32_t kC128FusedBlockSize = 256;
constexpr uint32_t kC128FusedNumWarps = kC128FusedBlockSize / device::kWarpThreads;

struct Compress128OnlineFusedParams {
  float* __restrict__ kv_score_buffer;       // [num_slots, 1, 3 * 512]
  const float* __restrict__ kv_score_input;  // [batch_size, 2 * 512]
  const float* __restrict__ score_bias;      // [128, 512]
  const PlanD* __restrict__ plan_d;
  const float* __restrict__ norm_weight;  // [512]
  const float* __restrict__ freqs_cis;    // [max_position, 64]
  const int64_t* __restrict__ out_loc;    // global compressed-cache loc
  uint8_t* __restrict__ kvcache;
  float eps;
  uint32_t batch_size;
  int32_t dcp_world_size;
  int32_t dcp_rank;
};

SGL_DEVICE int64_t map_c128_dcp_token_loc(const int64_t loc, const int32_t dcp_world_size, const int32_t dcp_rank) {
  if (loc < 0) return -1;
  if (dcp_world_size <= 1) return loc;
  if (loc % dcp_world_size != dcp_rank) return -1;
  return loc / dcp_world_size;
}

template <int64_t kHeadDim, int32_t kPageBits, bool kUsePDL, bool kBf16Store>
__global__ __launch_bounds__(kC128FusedBlockSize, 8) void flash_c128_online_decode_fused_v1(
    const __grid_constant__ Compress128OnlineFusedParams params) {
  using namespace device;

  constexpr uint32_t kVecSize = 2;
  constexpr uint32_t kCompressRatio = 128;
  constexpr uint32_t kRopeDim = 64;
  constexpr uint32_t kRopeWarp = kC128FusedNumWarps - 1;
  constexpr int64_t kPageBytes =
      kBf16Store ? ((kHeadDim * sizeof(bf16_t)) << kPageBits) : host::div_ceil(584ll << kPageBits, 576) * 576;
  static_assert(kHeadDim == kC128FusedBlockSize * kVecSize);
  static_assert(kRopeDim == device::kWarpThreads * kVecSize);
  static_assert(kHeadDim - kRopeDim == kRopeWarp * device::kWarpThreads * kVecSize);

  using Float2 = AlignedVector<float, kVecSize>;

  const uint32_t batch_id = blockIdx.x;
  if (batch_id >= params.batch_size) return;

  // The online planner runs immediately before this kernel and does not issue
  // a PDL trigger. Wait before consuming its device-written DecodePlan.
  PDLWaitPrimary<kUsePDL>();

  const auto plan = params.plan_d[batch_id];
  if (plan.seq_len == 0 || plan.write_loc < 0 || plan.read_page_0 < 0) return;

  const uint32_t pos_in_chunk = (plan.seq_len - 1) % kCompressRatio;
  const uint32_t tx = threadIdx.x;
  const uint32_t warp_id = tx / device::kWarpThreads;
  const uint32_t lane_id = tx % device::kWarpThreads;

  const auto kv_src = params.kv_score_input + batch_id * (kHeadDim * 2);
  const auto kv_load_buf = params.kv_score_buffer + plan.read_page_0 * (kHeadDim * 3);
  const auto kv_store_buf = params.kv_score_buffer + plan.write_loc * (kHeadDim * 3);

  Float2 new_kv;
  Float2 new_score_raw;
  Float2 bias;
  new_kv.load(kv_src, tx);
  new_score_raw.load(kv_src + kHeadDim, tx);
  bias.load(params.score_bias + pos_in_chunk * kHeadDim, tx);

  Float2 out_kv;
  Float2 out_max;
  Float2 out_sum;
  if (pos_in_chunk == 0) {
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      out_kv[i] = new_kv[i];
      out_max[i] = new_score_raw[i] + bias[i];
      out_sum[i] = 1.0f;
    }
  } else {
    Float2 old_max;
    Float2 old_sum;
    Float2 old_kv;
    old_max.load(kv_load_buf, tx);
    old_sum.load(kv_load_buf + kHeadDim, tx);
    old_kv.load(kv_load_buf + 2 * kHeadDim, tx);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const float score = new_score_raw[i] + bias[i];
      const float next_max = fmaxf(old_max[i], score);
      const float prev_weight = old_sum[i] * expf(old_max[i] - next_max);
      const float next_weight = expf(score - next_max);
      const float next_sum = prev_weight + next_weight;
      out_kv[i] = (old_kv[i] * prev_weight + new_kv[i] * next_weight) / next_sum;
      out_max[i] = next_max;
      out_sum[i] = next_sum;
    }
  }

  if (pos_in_chunk != kCompressRatio - 1) {
    PDLTriggerSecondary<kUsePDL>();
    out_max.store(kv_store_buf, tx);
    out_sum.store(kv_store_buf + kHeadDim, tx);
    out_kv.store(kv_store_buf + 2 * kHeadDim, tx);
    return;
  }

  // Boundary tokens write directly to the compressed cache. Ownership is
  // based on the global compressed loc, matching the existing DCP Triton
  // scatter-store path; local slot 0 remains a valid rank-0 destination.
  const int64_t out_loc = map_c128_dcp_token_loc(params.out_loc[batch_id], params.dcp_world_size, params.dcp_rank);
  if (out_loc < 0) {
    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  __shared__ float partial_sums[kC128FusedNumWarps];
  float sum_of_squares = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    sum_of_squares += out_kv[i] * out_kv[i];
  }
  const float warp_sum = warp::reduce_sum(sum_of_squares);
  if (lane_id == 0) partial_sums[warp_id] = warp_sum;
  __syncthreads();
  sum_of_squares = warp::reduce_sum<kC128FusedNumWarps>(partial_sums[lane_id % kC128FusedNumWarps]);
  const float norm_factor = math::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + params.eps);

  Float2 weight;
  Float2 data;
  weight.load(params.norm_weight, tx);
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    data[i] = out_kv[i] * norm_factor * weight[i];
  }

  Float2 freq;
  if (warp_id == kRopeWarp) {
    const uint32_t position = plan.seq_len - kCompressRatio;
    freq.load(params.freqs_cis + position * kRopeDim, lane_id);
  }

  const int64_t page = out_loc >> kPageBits;
  const int64_t offset = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * (kBf16Store ? (kHeadDim * sizeof(bf16_t)) : 576);

  PDLTriggerSecondary<kUsePDL>();

  if constexpr (kBf16Store) {
    if (warp_id == kRopeWarp) {
      const float real = data[0];
      const float imag = data[1];
      data[0] = real * freq[0] - imag * freq[1];
      data[1] = real * freq[1] + imag * freq[0];
    }
    reinterpret_cast<bf16x2_t*>(value_ptr)[tx] = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
  } else if (warp_id == kRopeWarp) {
    const float real = data[0];
    const float imag = data[1];
    data[0] = real * freq[0] - imag * freq[1];
    data[1] = real * freq[1] + imag * freq[0];
    reinterpret_cast<bf16x2_t*>(value_ptr + 448)[lane_id] = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
  } else {
    // Match the existing store's BF16 round-trip before FP8 quantization.
    const float x = cast<float>(cast<bf16_t>(data[0]));
    const float y = cast<float>(cast<bf16_t>(data[1]));
    const float abs_max = warp::reduce_max(fmaxf(fabsf(x), fabsf(y)));
    const float scale_raw = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const float inv_scale = inv_scale_ue8m0(scale_ue8m0);
    reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[tx] = pack_fp8(x * inv_scale, y * inv_scale);
    if (lane_id == 0) {
      const auto scale_ptr = page_ptr + (576 << kPageBits) + offset * 8;
      scale_ptr[warp_id] = scale_ue8m0;
    }
  }
}

template <int64_t kHeadDim, uint32_t kPageSize, bool kUsePDL, bool kBf16Store>
struct FlashCompress128OnlineFusedKernel {
  static_assert(kHeadDim == 512, "online C128 fused decode requires head_dim=512");
  static_assert(std::has_single_bit(kPageSize), "C128 cache page size must be a power of two");
  static constexpr int32_t kPageBits = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes =
      kBf16Store ? kHeadDim * sizeof(bf16_t) * kPageSize : host::div_ceil(584ll * kPageSize, 576) * 576;
  static constexpr auto kernel = flash_c128_online_decode_fused_v1<kHeadDim, kPageBits, kUsePDL, kBf16Store>;

  static void run_decode(
      const tvm::ffi::TensorView kv_score_buffer,
      const tvm::ffi::TensorView kv_score_input,
      const tvm::ffi::TensorView ape,
      const tvm::ffi::TensorView plan_d_,
      const tvm::ffi::TensorView norm_weight,
      const float norm_eps,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      const int32_t dcp_world_size,
      const int32_t dcp_rank) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({-1, 1, kHeadDim * 3}).with_dtype<float>().with_device(device_).verify(kv_score_buffer);
    TensorMatcher({B, kHeadDim * 2}).with_dtype<float>().with_device(device_).verify(kv_score_input);
    TensorMatcher({128, kHeadDim}).with_dtype<float>().with_device(device_).verify(ape);
    TensorMatcher({kHeadDim}).with_dtype<float>().with_device(device_).verify(norm_weight);
    TensorMatcher({-1, 64}).with_dtype<float>().with_device(device_).verify(freqs_cis);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device_).verify(out_loc);
    TensorMatcher({-1, -1}).with_strides({kPageBytes, 1}).with_dtype<uint8_t>().with_device(device_).verify(kvcache);
    const auto plan_d = host::compress::verify_plan_d(plan_d_, B, device_);

    RuntimeCheck(dcp_world_size >= 1);
    RuntimeCheck(dcp_rank >= 0 && dcp_rank < dcp_world_size);
    if constexpr (kBf16Store) {
      RuntimeCheck(dcp_world_size == 1, "unified BF16 C128 store does not support DCP");
    }

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;
    const auto params = Compress128OnlineFusedParams{
        .kv_score_buffer = static_cast<float*>(kv_score_buffer.data_ptr()),
        .kv_score_input = static_cast<const float*>(kv_score_input.data_ptr()),
        .score_bias = static_cast<const float*>(ape.data_ptr()),
        .plan_d = plan_d,
        .norm_weight = static_cast<const float*>(norm_weight.data_ptr()),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .out_loc = static_cast<const int64_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .eps = norm_eps,
        .batch_size = batch_size,
        .dcp_world_size = dcp_world_size,
        .dcp_rank = dcp_rank,
    };
    LaunchKernel(batch_size, kC128FusedBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
