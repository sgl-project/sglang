/// Index-K write of `fp4_rope.cuh` into the split FlyDSL layout: payload `[npages, 1, 4, kPageSize, 16]`
/// (chunk `c` holds elements `[32c, 32c + 32)`) and ue8m0 exponents `[npages, 1, 4, kPageSize]` with
/// the slot axis transposed as a 16 x 4 tile -- the bytes `store_fp4_index_k_cache_split` writes.

#pragma once

#ifndef USE_ROCM
#error "fp4_rope_hip.cuh writes the FlyDSL index-K layout, which exists on ROCm only"
#endif

#include "fp4_rope.cuh"

namespace sglang {

struct IndexKSplitParams {
  const bf16_t* __restrict__ input;        // [num_tokens, kHeadDim] bf16, pre-norm
  const bf16_t* __restrict__ norm_weight;  // [kHeadDim] bf16
  const float* __restrict__ freqs_cis;     // [max_pos, kRopeDim] fp32, real/imag interleaved
  const void* __restrict__ positions;      // [num_tokens] PosT
  const int64_t* __restrict__ loc;         // [num_tokens] index-K slot; 0 publishes nothing
  uint8_t* __restrict__ payload;           // [npages, 1, 4, kPageSize, 16] uint8
  uint8_t* __restrict__ scale;             // [npages, 1, 4, kPageSize] uint8
  uint32_t num_tokens;
  float eps;
};

template <bool kUsePDL, int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, uint32_t kRatio, typename PosT>
__global__ __launch_bounds__(kFp4RopeWarpsPerCTA* device::kWarpThreads) void flash_index_k_split_kernel(
    const IndexKSplitParams params) {
  using namespace device;
  namespace fp4 = deepseek_v4::fp4;

  static_assert(
      kHeadDim == 128 && kRopeDim == 64,
      "the one-warp tiling is specific to a 128-wide row whose second half is the RoPE tail");
  static_assert(kHeadDim == 2 * kWarpThreads * 2, "a lane owns one bf16x2 of each half");
  static_assert(std::has_single_bit(kRatio), "group_pos is derived by masking, so the ratio must be a power of two");

  using bf16_vec_t = AlignedVector<bf16x2_t, 1>;
  using fp32_vec_t = AlignedVector<float, 2>;

  const auto lane = threadIdx.x % kWarpThreads;
  const auto row = blockIdx.x * kFp4RopeWarpsPerCTA + threadIdx.x / kWarpThreads;
  // Warp-uniform, so the reductions below still see a full warp.
  if (row >= params.num_tokens) return;

  // step metadata, not the predecessor's output: loaded ahead of the PDL gate; slot 0 rows drop at the store
  const auto slot_id = params.loc[row];
  const auto position = static_cast<int64_t>(static_cast<const PosT*>(params.positions)[row]);
  PDLWaitPrimary<kUsePDL>();

  bf16_vec_t head_in, tail_in, head_w, tail_w;
  head_in.load(params.input + row * kHeadDim, lane);
  tail_in.load(params.input + row * kHeadDim, lane + kWarpThreads);
  head_w.load(params.norm_weight, lane);
  tail_w.load(params.norm_weight, lane + kWarpThreads);
  fp32_vec_t freq;
  freq.load(params.freqs_cis + (position & ~static_cast<int64_t>(kRatio - 1)) * kRopeDim, lane);

  fp32x2_t head, tail;
  {
    const auto [h0, h1] = cast<fp32x2_t>(head_in[0]);
    const auto [t0, t1] = cast<fp32x2_t>(tail_in[0]);
    const auto sqrsum = warp::reduce_sum(h0 * h0 + h1 * h1 + t0 * t0 + t1 * t1);
    const auto inv_rms = math::rsqrt(sqrsum * (1.0f / static_cast<float>(kHeadDim)) + params.eps);
    const auto [wh0, wh1] = cast<fp32x2_t>(head_w[0]);
    const auto [wt0, wt1] = cast<fp32x2_t>(tail_w[0]);
    // `k_norm` materializes a bf16 tensor, so the norm result is rounded before
    // anything downstream sees it -- the RoPE below included.
    head = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{wh0 * (h0 * inv_rms), wh1 * (h1 * inv_rms)}));
    tail = cast<fp32x2_t>(cast<bf16x2_t>(fp32x2_t{wt0 * (t0 * inv_rms), wt1 * (t1 * inv_rms)}));
  }

  const auto packed = index_rope_quant_pack(head, tail, fp32x2_t{freq[0], freq[1]});

  // slot 0 is reserved: padded graph rows and, at ratio > 1, rows completing no group publish nothing
  if (slot_id <= 0) return;
  const auto page = slot_id / kPageSize;
  const auto slot = slot_id % kPageSize;

  // byte j of the slot is chunk j / 16, byte j % 16: the head pair at byte lane, the tail pair at lane + 32
  const auto payload_ptr = params.payload + page * (4 * kPageSize * 16) + slot * 16;
  payload_ptr[(lane / 16) * (kPageSize * 16) + lane % 16] = static_cast<uint8_t>(packed.payload[0]);
  payload_ptr[(2 + lane / 16) * (kPageSize * 16) + lane % 16] = static_cast<uint8_t>(packed.payload[1]);

  // Lane 0 holds the exponents of blocks 0 and 2, lane 16 those of 1 and 3.
  if (lane % (kWarpThreads / 2) == 0) {
    const auto shuffled = (slot % 16) * 4 + slot / 16;
    const auto scale_ptr = params.scale + page * (4 * kPageSize) + shuffled;
    const auto first = lane / (kWarpThreads / 2);
    scale_ptr[first * kPageSize] = static_cast<uint8_t>(packed.exponent[0]);
    scale_ptr[(2 + first) * kPageSize] = static_cast<uint8_t>(packed.exponent[1]);
  }
}

/// \brief Host side of `flash_index_k_split_kernel`.
template <int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, uint32_t kRatio, bool kUsePDL>
struct FlashIndexKSplitKernel {
  static constexpr uint32_t kBlockSize = kFp4RopeWarpsPerCTA * device::kWarpThreads;

  template <typename PosT>
  static constexpr auto kernel = flash_index_k_split_kernel<kUsePDL, kHeadDim, kRopeDim, kPageSize, kRatio, PosT>;

  /// `FlashIndexKKernel::run_index_k`'s arguments, the cache replaced by
  /// \param payload `[npages, 1, 4, kPageSize, 16]` uint8 (the pool's
  ///        `float4_e2m1fn_x2` buffer viewed as bytes).
  /// \param scale `[npages, 1, 4, kPageSize]` uint8.
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView loc,
      const tvm::ffi::TensorView payload,
      const tvm::ffi::TensorView scale,
      const float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"npages"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({N, kHeadDim}).with_dtype<bf16_t>().with_device(device_).verify(input);
    TensorMatcher({kHeadDim}).with_dtype<bf16_t>().with_device(device_).verify(norm_weight);
    // Real/imag interleaved, so the trailing dim is kRopeDim, not kRopeDim / 2.
    TensorMatcher({-1, kRopeDim}).with_dtype<fp32_t>().with_device(device_).verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(pos_dtype).with_device(device_).verify(positions);
    TensorMatcher({N}).with_dtype<int64_t>().with_device(device_).verify(loc);
    TensorMatcher({P, 1, 4, kPageSize, 16}).with_dtype<uint8_t>().with_device(device_).verify(payload);
    TensorMatcher({P, 1, 4, kPageSize}).with_dtype<uint8_t>().with_device(device_).verify(scale);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;

    const auto params = IndexKSplitParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .norm_weight = static_cast<const bf16_t*>(norm_weight.data_ptr()),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .loc = static_cast<const int64_t*>(loc.data_ptr()),
        .payload = static_cast<uint8_t*>(payload.data_ptr()),
        .scale = static_cast<uint8_t*>(scale.data_ptr()),
        .num_tokens = num_tokens,
        .eps = eps,
    };
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(div_ceil(num_tokens, kFp4RopeWarpsPerCTA), kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

}  // namespace sglang
