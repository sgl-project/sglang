#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

// 4 warps per block: warp-per-(token, head) work-item dispatch (Q kernel).
constexpr uint32_t kFusedQBlockSize = 128;
constexpr uint32_t kFusedQNumWarps = kFusedQBlockSize / device::kWarpThreads;

// 8 warps per block: block-per-token work-item dispatch (K kernel).
constexpr uint32_t kFusedKBlockSize = 256;
constexpr uint32_t kFusedKNumWarps = kFusedKBlockSize / device::kWarpThreads;

#define Q_KERNEL __global__ __launch_bounds__(kFusedQBlockSize, 16)
#define K_KERNEL __global__ __launch_bounds__(kFusedKBlockSize, 8)

// ============================================================================
// Q kernel: warp-per-(token, head) rmsnorm-self + RoPE + write to q_out.
// ============================================================================

struct FusedQNormRopeParams {
  const void* __restrict__ q_input;     // (B, num_q_heads, kHeadDim) DType
  void* __restrict__ q_output;          // (B, num_q_heads, kHeadDim) DType
  const float* __restrict__ freqs_cis;  // (max_pos, kRopeDim) fp32 (re/im interleaved)
  const void* __restrict__ positions;   // (B,) PosT
  int64_t q_input_stride_batch;
  int64_t q_output_stride_batch;
  uint32_t batch_size;
  uint32_t num_q_heads;
  float eps;
};

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, typename PosT, bool kUsePDL>
Q_KERNEL void fused_q_norm_rope(const __grid_constant__ FusedQNormRopeParams params) {
  using namespace device;

  constexpr int64_t kMaxVecSize = 16 / sizeof(DType);
  constexpr int64_t kVecSize = std::min(kMaxVecSize, kHeadDim / kWarpThreads);
  constexpr int64_t kLocalSize = kHeadDim / (kWarpThreads * kVecSize);
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim % (kWarpThreads * kVecSize) == 0);
  static_assert(kLocalSize * kVecSize * kWarpThreads == kHeadDim);
  static_assert(kRopeDim % kVecSize == 0);
  static_assert(kRopeSize <= kWarpThreads);
  static_assert(kRopeDim == kWarpThreads * 2, "1 (real, imag) pair per lane");

  using Storage = AlignedVector<DType, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;

  const uint32_t total_works = params.batch_size * params.num_q_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_q_heads;
  const uint32_t head_id = work_id % params.num_q_heads;
  const auto input_ptr =
      static_cast<const DType*>(params.q_input) + batch_id * params.q_input_stride_batch + head_id * kHeadDim;
  const auto output_ptr =
      static_cast<DType*>(params.q_output) + batch_id * params.q_output_stride_batch + head_id * kHeadDim;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);

  __shared__ Storage s_rope[kFusedQNumWarps][kRopeSize];

  // Prefetch this lane's freq pair before the PDL gate so the wait happens
  // outside the dependency chain on `position`.
  const auto mem_freq = tile::Memory<fp32x2_t>{lane_id, kWarpThreads};

  PDLWaitPrimary<kUsePDL>();

  // part 1: rmsnorm-self (no weight).
  const auto gmem = tile::Memory<Storage>{lane_id, kWarpThreads};
  Storage input_vec[kLocalSize];
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
    input_vec[i] = gmem.load(input_ptr, i);
  }

  const auto freq = mem_freq.load(params.freqs_cis + position * kRopeDim);

  float sum_of_squares = 0.0f;
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const auto x = cast<float>(input_vec[i][j]);
      sum_of_squares += x * x;
    }
  }
  sum_of_squares = warp::reduce_sum(sum_of_squares);
  const auto norm_factor = math::rsqrt(sum_of_squares / kHeadDim + params.eps);

#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const auto x = cast<float>(input_vec[i][j]);
      input_vec[i][j] = cast<DType>(x * norm_factor);
    }
  }

  // Stash the rope tail (last kRopeSize lanes' last tile) into shared memory;
  // write nope tiles to gmem directly.
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
    if (i == kLocalSize - 1 && is_rope_lane) {
      const auto rope_id = lane_id - (kWarpThreads - kRopeSize);
      s_rope[warp_id][rope_id] = input_vec[i];
    } else {
      gmem.store(output_ptr, input_vec[i], i);
    }
  }
  __syncwarp();

  // part 2: RoPE on all 32 lanes -- one (real, imag) bf16x2 pair per lane.
  using DType2 = packed_t<DType>;
  const auto mem_elem = tile::Memory<DType2>{lane_id, kWarpThreads};
  const auto elem = mem_elem.load(s_rope[warp_id]);
  const auto [x_real, x_imag] = cast<fp32x2_t>(elem);
  const auto [freq_real, freq_imag] = freq;
  const fp32x2_t rotated = {
      x_real * freq_real - x_imag * freq_imag,
      x_real * freq_imag + x_imag * freq_real,
  };
  mem_elem.store(output_ptr + (kHeadDim - kRopeDim), cast<DType2>(rotated));

  PDLTriggerSecondary<kUsePDL>();
}

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, bool kUsePDL>
struct FusedQNormRopeKernel {
  template <typename PosT>
  static constexpr auto kernel = fused_q_norm_rope<DType, kHeadDim, kRopeDim, PosT, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView q_input,
      const tvm::ffi::TensorView q_output,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      float eps) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_q_heads"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_input);
    TensorMatcher({B, H, kHeadDim})  //
        .with_strides({-1, kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q_output);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto num_q_heads = static_cast<uint32_t>(H.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedQNormRopeParams{
        .q_input = q_input.data_ptr(),
        .q_output = q_output.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .q_input_stride_batch = q_input.stride(0),
        .q_output_stride_batch = q_output.stride(0),
        .batch_size = batch_size,
        .num_q_heads = num_q_heads,
        .eps = eps,
    };
    const auto total_works = batch_size * num_q_heads;
    const auto num_blocks = div_ceil(total_works, kFusedQNumWarps);
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kFusedQBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

// ============================================================================
// K kernel: block-per-token rmsnorm (with kv_weight) + RoPE + FlashMLA store.
// ============================================================================

struct FusedKNormRopeFlashMLAParams {
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
};

template <typename DType, int64_t kHeadDim, int64_t kRopeDim, typename PosT, int32_t kPageBits, bool kUsePDL>
K_KERNEL void fused_k_norm_rope_flashmla(const __grid_constant__ FusedKNormRopeFlashMLAParams params) {
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
struct FusedKNormRopeFlashMLAKernel {
  static constexpr int32_t kLogPageSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = host::div_ceil(584 * kPageSize, 576) * 576;
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogPageSize == kPageSize);
  static_assert(kHeadDim == 512 && kRopeDim == 64, "FlashMLA layout requires (512, 64)");

  template <typename PosT>
  static constexpr auto kernel = fused_k_norm_rope_flashmla<DType, kHeadDim, kRopeDim, PosT, kLogPageSize, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView kv_weight,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView out_loc,
      const tvm::ffi::TensorView kvcache,
      float eps) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

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

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedKNormRopeFlashMLAParams{
        .kv = kv.data_ptr(),
        .kv_weight = kv_weight.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .out_loc = static_cast<const int32_t*>(out_loc.data_ptr()),
        .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
        .kv_stride_batch = kv.stride(0),
        .batch_size = batch_size,
        .eps = eps,
    };
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(batch_size, kFusedKBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

}  // namespace
