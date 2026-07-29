// DeepSeek-V3.2 only.
//
// DSA indexer K kernels: single-head LayerNorm (not RMS), ropes the leading
// kRopeDim dims, and fp8-quantizes the rotated activations. V3.2 drops the
// Hadamard incoherence rotation; it is logit-preserving (see main_norm_rope.cuh).
//
// Independent of the wk + weights_proj GEMM fusion (dsa_indexer.py): `k_input`
// here is the non-contiguous wk slice kw[:, :head_dim] read via
// k_input_stride_batch (no copy).
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

using deepseek_v4::fp8::pack_fp8;

constexpr uint32_t kFusedKIndexerBlockSize = 128;
constexpr uint32_t kFusedKIndexerNumWarps = kFusedKIndexerBlockSize / device::kWarpThreads;

#define K_INDEXER_KERNEL __global__ __launch_bounds__(kFusedKIndexerBlockSize, 16)

template <int64_t kRopeDim>
SGL_DEVICE device::AlignedVector<float, 4>
load_rope_first_cos_sin(const float* __restrict__ cos_sin_cache, int32_t lane_id) {
  constexpr int64_t kHalfRopeDim = kRopeDim / 2;
  const int32_t pair0 = lane_id * 2;
  const int32_t pair1 = pair0 + 1;
  device::AlignedVector<float, 4> freq;
  freq[0] = cos_sin_cache[pair0];
  freq[1] = cos_sin_cache[kHalfRopeDim + pair0];
  freq[2] = cos_sin_cache[pair1];
  freq[3] = cos_sin_cache[kHalfRopeDim + pair1];
  return freq;
}

// Indexer K: LayerNorm + RoPE -> bf16.
struct FusedKIndexerNormRopeParams {
  const void* __restrict__ k_input;         // (B, 128) DType
  void* __restrict__ k_out;                 // (B, 128) DType
  const float* __restrict__ weight;         // (128,) fp32  -- LayerNorm gamma
  const float* __restrict__ bias;           // (128,) fp32  -- LayerNorm beta
  const float* __restrict__ cos_sin_cache;  // (max_pos, 64) fp32 [cos..., sin...]
  const void* __restrict__ positions;       // (B,) PosT
  // Row stride for `k_input` in elements (caller passes the wk slice directly).
  int64_t k_input_stride_batch;
  uint32_t batch_size;
  float eps;
};

template <typename DType, typename PosT, bool kUsePDL>
K_INDEXER_KERNEL void fused_k_indexer_norm_rope(const __grid_constant__ FusedKIndexerNormRopeParams params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;  // = 16
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kFusedKIndexerNumWarps + warp_id;
  const bool is_rope_lane = lane_id < kRopeSize;

  if (work_id >= params.batch_size) return;

  const auto input_ptr = static_cast<const DType*>(params.k_input) + work_id * params.k_input_stride_batch;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[work_id]);
  const auto cos_sin_cache = params.cos_sin_cache + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq, gamma, beta;

  // part 1: LayerNorm
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    gamma.load(params.weight, lane_id);
    beta.load(params.bias, lane_id);
    if (is_rope_lane) freq = load_rope_first_cos_sin<kRopeDim>(cos_sin_cache, lane_id);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
      sum += data[i];
    }
    const float mean = warp::reduce_sum(sum) / kHeadDim;

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float centered = data[i] - mean;
      var += centered * centered;
    }
    const float inv_std = math::rsqrt(warp::reduce_sum(var) / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = (data[i] - mean) * inv_std * gamma[i] + beta[i];
    }
  }

  // part 2: rope on rope lanes
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto fxr = freq[0];
    const auto fxi = freq[1];
    const auto fyr = freq[2];
    const auto fyi = freq[3];
    data[0] = x_real * fxr - x_imag * fxi;
    data[1] = x_real * fxi + x_imag * fxr;
    data[2] = y_real * fyr - y_imag * fyi;
    data[3] = y_real * fyi + y_imag * fyr;
  }

  PDLTriggerSecondary<kUsePDL>();

  {
    Storage out_vec;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      out_vec[i] = cast<DType>(data[i]);
    auto out_row = static_cast<DType*>(params.k_out) + work_id * kHeadDim;
    out_vec.store(out_row, lane_id);
  }
}

template <typename DType, bool kUsePDL>
struct FusedKIndexerNormRopeKernel {
  template <typename PosT>
  static constexpr auto kernel = fused_k_indexer_norm_rope<DType, PosT, kUsePDL>;

  static void forward(
      const tvm::ffi::TensorView k_input,
      const tvm::ffi::TensorView k_out,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps) {
    using namespace host;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRopeDim = 64;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, kHeadDim})  //
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(k_input);
    TensorMatcher({B, kHeadDim})  //
        .with_strides({kHeadDim, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(k_out);
    TensorMatcher({kHeadDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({kHeadDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(bias);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(cos_sin_cache);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedKIndexerNormRopeParams{
        .k_input = k_input.data_ptr(),
        .k_out = k_out.data_ptr(),
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
        .positions = positions.data_ptr(),
        .k_input_stride_batch = k_input.stride(0),
        .batch_size = batch_size,
        .eps = static_cast<float>(eps),
    };
    const auto num_blocks = div_ceil(batch_size, kFusedKIndexerNumWarps);
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kFusedKIndexerBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

// Indexer K + fused store: LayerNorm + RoPE + fp8 quant + paged store in one
// launch. Page layout matches fused_store_index_cache.cuh: each page is
// 132*page_size bytes (128*page_size fp8 keys, then 4*page_size fp32 scales).
struct FusedKIndexerNormRopeStoreParams {
  const void* __restrict__ k_input;         // (B, 128) DType
  void* __restrict__ cache;                 // (num_pages, 132*page_size) uint8
  const void* __restrict__ indices;         // (B,) int64  -- out_cache_loc
  const float* __restrict__ weight;         // (128,) fp32  -- LayerNorm gamma
  const float* __restrict__ bias;           // (128,) fp32  -- LayerNorm beta
  const float* __restrict__ cos_sin_cache;  // (max_pos, 64) fp32 [cos..., sin...]
  const void* __restrict__ positions;       // (B,) PosT
  // Row stride for `k_input` (caller passes the non-contiguous wk slice directly).
  int64_t k_input_stride_batch;
  uint32_t batch_size;
  uint32_t owner_rank;
  uint32_t owner_size;
  float eps;
};

template <typename DType, typename PosT, bool kUsePDL, int32_t kPageBits>
K_INDEXER_KERNEL void fused_k_indexer_norm_rope_store(const __grid_constant__ FusedKIndexerNormRopeStoreParams params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;  // = 16
  constexpr int64_t kPageBytes = 132ll << kPageBits;
  static_assert(kHeadDim == kWarpThreads * kVecSize);
  static_assert(kRopeDim == kWarpThreads * 2);
  static_assert(kRopeSize <= kWarpThreads);

  using Storage = AlignedVector<DType, kVecSize>;
  using Float4 = AlignedVector<float, kVecSize>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;  // 4 fp8 / lane

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto work_id = blockIdx.x * kFusedKIndexerNumWarps + warp_id;
  const bool is_rope_lane = lane_id < kRopeSize;

  if (work_id >= params.batch_size) return;

  const auto index = static_cast<const int64_t*>(params.indices)[work_id];
  if (index < 0) {
    PDLWaitPrimary<kUsePDL>();
    PDLTriggerSecondary<kUsePDL>();
    return;
  }
  auto page = static_cast<int32_t>(index >> kPageBits);
  if (page % params.owner_size != params.owner_rank) {
    PDLWaitPrimary<kUsePDL>();
    PDLTriggerSecondary<kUsePDL>();
    return;
  }
  page /= params.owner_size;

  const auto input_ptr = static_cast<const DType*>(params.k_input) + work_id * params.k_input_stride_batch;
  const auto position = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[work_id]);
  const auto cos_sin_cache = params.cos_sin_cache + position * kRopeDim;

  PDLWaitPrimary<kUsePDL>();
  Float4 data, freq, gamma, beta;

  // part 1: LayerNorm
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    gamma.load(params.weight, lane_id);
    beta.load(params.bias, lane_id);
    if (is_rope_lane) freq = load_rope_first_cos_sin<kRopeDim>(cos_sin_cache, lane_id);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = cast<float>(input_vec[i]);
      sum += data[i];
    }
    const float mean = warp::reduce_sum(sum) / kHeadDim;

    float var = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float centered = data[i] - mean;
      var += centered * centered;
    }
    const float inv_std = math::rsqrt(warp::reduce_sum(var) / kHeadDim + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      data[i] = (data[i] - mean) * inv_std * gamma[i] + beta[i];
    }
  }

  // part 2: rope on rope lanes
  if (is_rope_lane) {
    const auto x_real = data[0];
    const auto x_imag = data[1];
    const auto y_real = data[2];
    const auto y_imag = data[3];
    const auto fxr = freq[0];
    const auto fxi = freq[1];
    const auto fyr = freq[2];
    const auto fyi = freq[3];
    data[0] = x_real * fxr - x_imag * fxi;
    data[1] = x_real * fxi + x_imag * fxr;
    data[2] = y_real * fyr - y_imag * fyi;
    data[3] = y_real * fyi + y_imag * fyr;
  }

  PDLTriggerSecondary<kUsePDL>();

  // part 3: fp8 act-quant + paged store. Round through bf16 first so the fp8
  // scale matches the un-fused path.
#pragma unroll
  for (int i = 0; i < kVecSize; ++i)
    data[i] = cast<float>(cast<DType>(data[i]));

  float local_max = math::abs(data[0]);
#pragma unroll
  for (int i = 1; i < kVecSize; ++i)
    local_max = math::max(local_max, math::abs(data[i]));
  const auto abs_max = warp::reduce_max(local_max);
  const auto scale = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
  const auto inv_scale = 1.0f / scale;

  const int32_t offset = static_cast<int32_t>(index & ((1 << kPageBits) - 1));
  const auto page_ptr = static_cast<uint8_t*>(params.cache) + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * kHeadDim;
  const auto scale_ptr = page_ptr + (kHeadDim << kPageBits) + offset * 4;

  OutStorage result;
  result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
  result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);
  reinterpret_cast<OutStorage*>(value_ptr)[lane_id] = result;
  if (lane_id == 0) *reinterpret_cast<float*>(scale_ptr) = scale;
}

template <typename DType, bool kUsePDL, uint32_t kPageSize>
struct FusedKIndexerNormRopeStoreKernel {
  static constexpr int32_t kPageBits = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = 132ll * kPageSize;
  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");

  template <typename PosT>
  static constexpr auto kernel = fused_k_indexer_norm_rope_store<DType, PosT, kUsePDL, kPageBits>;

  static void forward(
      const tvm::ffi::TensorView k_input,
      const tvm::ffi::TensorView cache,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      double eps,
      int64_t owner_rank,
      int64_t owner_size) {
    using namespace host;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRopeDim = 64;

    auto B = SymbolicSize{"batch_size"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, kHeadDim})  //
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(k_input);
    TensorMatcher({-1, -1})  //
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({B})  //
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(indices);
    TensorMatcher({kHeadDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(weight);
    TensorMatcher({kHeadDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(bias);
    TensorMatcher({-1, kRopeDim})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(cos_sin_cache);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = FusedKIndexerNormRopeStoreParams{
        .k_input = k_input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .weight = static_cast<const float*>(weight.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
        .positions = positions.data_ptr(),
        .k_input_stride_batch = k_input.stride(0),
        .batch_size = batch_size,
        .owner_rank = static_cast<uint32_t>(owner_rank),
        .owner_size = static_cast<uint32_t>(owner_size),
        .eps = static_cast<float>(eps),
    };
    const auto num_blocks = div_ceil(batch_size, kFusedKIndexerNumWarps);
    const auto k_int32 = kernel<int32_t>;
    const auto k_int64 = kernel<int64_t>;
    const auto k = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
    LaunchKernel(num_blocks, kFusedKIndexerBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(k, params);
  }
};

}  // namespace
