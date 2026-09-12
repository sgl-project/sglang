/// `main_norm_rope.cuh`'s K rmsnorm + RoPE + FlashMLA store with the same tokens' query heads
/// roped in place by the same launch (ROCm).

#pragma once

#ifndef USE_ROCM
#error "main_norm_rope_hip.cuh fuses the query rope into the K launch on ROCm only"
#endif

#include "main_norm_rope.cuh"
#include <type_traits>

namespace sglang {

struct FusedKNormRopeQFlashMLAParams {
  const void* __restrict__ kv;          // (B, kHeadDim) DType
  const void* __restrict__ kv_weight;   // (kHeadDim,) DType
  const float* __restrict__ freqs_cis;  // (max_pos, kRopeDim) fp32
  const void* __restrict__ positions;   // (B,) PosT
  const int32_t* __restrict__ out_loc;  // (B,) int32 -> cache slot id
  uint8_t* __restrict__ kvcache;        // (npages, kPageBytes) uint8
  // Row stride for `kv` in elements. Required because the upstream caller often
  // passes `qkv_a[..., q_lora_rank:]`, a non-contiguous slice whose stride[0]
  // equals `q_lora_rank + kHeadDim` rather than `kHeadDim`.
  int64_t kv_stride_batch;
  uint32_t batch_size;
  float eps;
  // (B, num_q_heads, kHeadDim) DType; the trailing kRopeDim of every head are
  // rotated in place with this token's frequencies.
  void* __restrict__ q;
  int64_t q_stride_batch;
  int64_t q_stride_head;
  uint32_t num_q_heads;
};

// copied from main_norm_rope.cuh fused_k_norm_rope_flashmla; the query-rope block is the addition
template <typename DType, int64_t kHeadDim, int64_t kRopeDim, typename PosT, int32_t kPageBits, bool kUsePDL>
K_KERNEL void fused_k_norm_rope_q_flashmla(const __grid_constant__ FusedKNormRopeQFlashMLAParams params) {
  using namespace device;

  constexpr int64_t kVecSize = 2;
  constexpr uint32_t kRopeWarp = kFusedKNumWarps - 1;
  constexpr int64_t kPageBytes = host::div_ceil(584ll << kPageBits, 576) * 576;
  static_assert(kHeadDim == kFusedKBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpThreads * kVecSize);
  static_assert(kHeadDim - kRopeDim == kRopeWarp * kWarpThreads * kVecSize);
  using Storage = AlignedVector<DType, kVecSize>;
  using Float2 = AlignedVector<float, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto work_id = blockIdx.x;
  if (work_id >= params.batch_size) return;

  const auto input_ptr = static_cast<const DType*>(params.kv) + work_id * params.kv_stride_batch;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[work_id]);
  const auto out_loc = params.out_loc[work_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float2 data, freq;

  // part 1: norm. Each thread owns one 2-elem pack (the `tx`-th).
  // Sum-of-squares is reduced block-wide via per-warp partials.
  {
    __shared__ float partial_sums[kFusedKNumWarps];

    Storage input_vec, weight_vec;
    input_vec.load(input_ptr, tx);
    weight_vec.load(params.kv_weight, tx);
    if (warp_id == kRopeWarp) freq.load(freqs_cis, lane_id);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      sum_of_squares += x * x;
    }
    const auto warp_sum = warp::reduce_sum(sum_of_squares);
    if (lane_id == 0) partial_sums[warp_id] = warp_sum;
    __syncthreads();
    // Replicate the per-warp partial sums onto all lanes of one warp and
    // reduce. Every group of `kBlockItemNumWarps` lanes ends up with the
    // global sum.
    sum_of_squares = warp::reduce_sum<kFusedKNumWarps>(partial_sums[lane_id % kFusedKNumWarps]);
    const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const auto x = cast<float>(input_vec[i]);
      const auto w = cast<float>(weight_vec[i]);
      data[i] = x * norm_factor * w;
    }
  }

  // Query rope, pair j of head h at q[h * stride + 448 + 2j]: the cross product is rounded, then
  // one fma with the cosine, so the bf16 result is bitwise the Triton flat rope kernel's on gfx950.
  {
    static_assert(std::is_same_v<DType, bf16_t>, "the in-place query rope reads q as bf16 pairs");
    constexpr uint32_t kPairsPerHead = kRopeDim / 2;
    const auto q_row = static_cast<DType*>(params.q) + work_id * params.q_stride_batch + (kHeadDim - kRopeDim);
    const auto n_pairs = params.num_q_heads * kPairsPerHead;
    for (uint32_t p = tx; p < n_pairs; p += kFusedKBlockSize) {
      const auto head = p / kPairsPerHead;
      const auto pair = p % kPairsPerHead;
      auto* ptr = reinterpret_cast<bf16x2_t*>(q_row + head * params.q_stride_head) + pair;
      const auto qv = cast<fp32x2_t>(*ptr);
      const auto cos_v = freqs_cis[2 * pair];
      const auto sin_v = freqs_cis[2 * pair + 1];
      const float rot_real = -__fmul_rn(qv.y, sin_v);
      const float rot_imag = __fmul_rn(qv.x, sin_v);
      const float out_real = fmaf(qv.x, cos_v, rot_real);
      const float out_imag = fmaf(qv.y, cos_v, rot_imag);
      *ptr = cast<bf16x2_t>(fp32x2_t{out_real, out_imag});
    }
  }

  // A negative out_loc marks a slot with no KV write target (e.g. the -1
  // sentinel from the full->SWA translation for out-of-window tokens or
  // padded rows); skip the row instead of writing out of bounds. Checked
  // here, not at the load, so the out_loc prefetch overlaps the norm above.
  if (out_loc < 0) return;

  const int32_t page = out_loc >> kPageBits;
  const int32_t offset = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * 576;

  PDLTriggerSecondary<kUsePDL>();

  // part 2: rope on warp 7 (BF16 store), per-warp UE8M0 quant + store on warps 0..6.
  if (warp_id == kRopeWarp) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto freq_real = freq[0];
    const auto freq_imag = freq[1];
    data[0] = x_real * freq_real - x_imag * freq_imag;
    data[1] = x_real * freq_imag + x_imag * freq_real;
    const auto result = cast<bf16x2_t>(fp32x2_t{data[0], data[1]});
    const auto rope_ptr = value_ptr + 448;
    reinterpret_cast<bf16x2_t*>(rope_ptr)[lane_id] = result;
  } else {
    const auto x = data[0];
    const auto y = data[1];
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_raw = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const auto result = pack_fp8(x * inv_scale, y * inv_scale);
    const auto scale_ptr = page_ptr + (576 << kPageBits) + offset * 8;
    reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[tx] = result;
    if (lane_id == 0) static_cast<uint8_t*>(scale_ptr)[warp_id] = scale_ue8m0;
  }
}

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, uint32_t kPageSize, bool kUsePDL>
struct FusedKNormRopeQFlashMLAKernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = host::div_ceil(584 * kPageSize, 576) * 576;
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogPageSize == kPageSize);
  static_assert(kHeadDim == 512 && kRopeDim == 64, "FlashMLA layout requires (512, 64)");

  template <typename PosT>
  static constexpr auto kernel = fused_k_norm_rope_q_flashmla<DType, kHeadDim, kRopeDim, PosT, kLogPageSize, kUsePDL>;

  /// `FusedKNormRopeFlashMLAKernel::forward`'s arguments plus `q` (B, H, kHeadDim).
  static void forward(
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView kv_weight,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      float eps,
      const tvm::ffi::TensorView q) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({B, kHeadDim})  //
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(kv);
    TensorMatcher({kHeadDim})  //
        .with_dtype<DType>()
        .with_device(device_)
        .verify(kv_weight);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);
    TensorMatcher({B})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(out_loc);
    TensorMatcher({-1, -1})  //
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(kvcache);
    auto H = SymbolicSize{"num_q_heads"};
    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, -1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q);
    // The rope pairs are read and written as 4-byte packs.
    RuntimeCheck(q.stride(1) % 2 == 0 && q.stride(0) % 2 == 0, "q strides must be even");

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedKNormRopeQFlashMLAParams{
        .kv = kv.data_ptr(),
        .kv_weight = kv_weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .out_loc = static_cast<const int32_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .kv_stride_batch = kv.stride(0),
        .batch_size = batch_size,
        .eps = eps,
        .q = q.data_ptr(),
        .q_stride_batch = q.stride(0),
        .q_stride_head = q.stride(1),
        .num_q_heads = static_cast<uint32_t>(H.unwrap()),
    };
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(batch_size, kFusedKBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

}  // namespace sglang
