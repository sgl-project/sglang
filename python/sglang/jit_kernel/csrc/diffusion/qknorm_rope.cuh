#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>

#include <cstdint>
#include <type_traits>

namespace {

struct QKNormRopeParams {
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;  // pre-offset by -num_qo_heads * head_stride_bytes
  const void* __restrict__ q_weight_ptr;
  const void* __restrict__ k_weight_ptr;
  const void* __restrict__ cos_sin_cache_ptr;
  const void* __restrict__ positions;
  int64_t q_stride_bytes;
  int64_t k_stride_bytes;
  int64_t head_stride_bytes;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t num_tokens;
  float eps;
};

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kWarpsPerBlock = kThreadsPerBlock / device::kWarpThreads;

template <uint32_t kLaneCount>
constexpr uint32_t active_mask() {
  static_assert(kLaneCount <= device::kWarpThreads, "active_mask lane count must not exceed warp size");
  if constexpr (kLaneCount == device::kWarpThreads) {
    return 0xffffffffu;
  } else {
    return (1u << kLaneCount) - 1u;
  }
}

SGL_DEVICE float load_cache_value(const float* ptr, int64_t idx) {
#ifdef USE_ROCM
  return ptr[idx];
#else
  return __ldg(ptr + idx);
#endif
}

template <int64_t kHeadDim, int64_t kRopeDim, bool kIsNeox, bool kUsePDL, typename DType, typename IdType>
__global__ void fused_qknorm_rope_warp(const QKNormRopeParams __grid_constant__ params) {
  using namespace device;

  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(kHeadDim <= 256, "Only warp-level fused qknorm+rope is supported");
  static_assert(kHeadDim % kWarpThreads == 0, "head_dim must be divisible by warp size");

  constexpr uint32_t kElemsPerThread = kHeadDim / kWarpThreads;
  constexpr uint32_t kVecSize = kElemsPerThread / 2;
  constexpr uint32_t kRotaryLanes = kRopeDim / kElemsPerThread;
  constexpr uint32_t kHalfRotaryLanes = kRotaryLanes / 2;
  constexpr uint32_t kActiveMask = active_mask<kRotaryLanes>();
  constexpr int64_t kCosSinStrideBytes = kRopeDim * sizeof(float);

  static_assert(kElemsPerThread % 2 == 0, "Each lane must own an even number of elements");
  static_assert(kRopeDim > 0 && kRopeDim <= kHeadDim, "Invalid rope dimension");
  static_assert(kRopeDim % kElemsPerThread == 0, "rope_dim must align with per-lane vector width");
  static_assert(
      !kIsNeox || (kRotaryLanes >= 2 && ((kRotaryLanes & (kRotaryLanes - 1)) == 0)),
      "NeoX fused qknorm+rope requires rotary lane count to be a power of 2");

  using Packed = packed_t<DType>;
  using Storage = AlignedVector<Packed, kVecSize>;

  const auto& [q_ptr, k_ptr, q_weight_ptr, k_weight_ptr, cos_sin_cache_ptr, positions, q_stride_bytes, k_stride_bytes, head_stride_bytes, num_qo_heads, num_kv_heads, num_tokens, eps] =
      params;

  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t start_worker_id = blockIdx.x * kWarpsPerBlock + warp_id;
  const uint32_t num_workers = gridDim.x * kWarpsPerBlock;
  const uint32_t num_qk_heads = num_qo_heads + num_kv_heads;
  const uint32_t num_works = num_qk_heads * num_tokens;

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t idx = start_worker_id; idx < num_works; idx += num_workers) {
    const uint32_t token_id = idx / num_qk_heads;
    const uint32_t head_id = idx % num_qk_heads;
    const bool load_q = head_id < num_qo_heads;
    const void* input = load_q ? pointer::offset(q_ptr, token_id * q_stride_bytes, head_id * head_stride_bytes)
                               : pointer::offset(k_ptr, token_id * k_stride_bytes, head_id * head_stride_bytes);
    const void* weight_ptr = load_q ? q_weight_ptr : k_weight_ptr;

    auto input_vec = load_as<Storage>(input, lane_id);
    const auto weight_vec = load_as<Storage>(weight_ptr, lane_id);

    float elems[kElemsPerThread];
    float sum_of_squares = 0.0f;

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const auto [x0, x1] = cast<fp32x2_t>(input_vec[j]);
      elems[2 * j] = x0;
      elems[2 * j + 1] = x1;
      sum_of_squares += x0 * x0 + x1 * x1;
    }

    sum_of_squares = warp::reduce_sum(sum_of_squares);
    const float norm_factor = math::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + eps);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const auto [w0, w1] = cast<fp32x2_t>(weight_vec[j]);
      elems[2 * j] *= norm_factor * w0;
      elems[2 * j + 1] *= norm_factor * w1;
    }

    if constexpr (kIsNeox) {
      if (lane_id < kRotaryLanes) {
        const auto pos = static_cast<int64_t>(static_cast<const IdType*>(positions)[token_id]);
        const auto cos_ptr = static_cast<const float*>(pointer::offset(cos_sin_cache_ptr, pos * kCosSinStrideBytes));
        const auto sin_ptr = cos_ptr + kRopeDim / 2;

#pragma unroll
        for (uint32_t i = 0; i < kElemsPerThread; ++i) {
          float swapped = __shfl_xor_sync(kActiveMask, elems[i], kHalfRotaryLanes);
          if (lane_id < kHalfRotaryLanes) {
            swapped = -swapped;
          }
          int dim_idx = static_cast<int>(lane_id * kElemsPerThread + i);
          dim_idx = (dim_idx * 2) % kRopeDim;
          const int half_idx = dim_idx / 2;
          const float cos = load_cache_value(cos_ptr, half_idx);
          const float sin = load_cache_value(sin_ptr, half_idx);
          elems[i] = elems[i] * cos + swapped * sin;
        }
      }
    } else {
      if (lane_id < kRotaryLanes) {
        const auto pos = static_cast<int64_t>(static_cast<const IdType*>(positions)[token_id]);
        const auto cos_ptr = static_cast<const float*>(pointer::offset(cos_sin_cache_ptr, pos * kCosSinStrideBytes));
        const auto sin_ptr = cos_ptr + kRopeDim / 2;

#pragma unroll
        for (uint32_t i = 0; i < kElemsPerThread; i += 2) {
          const float x = elems[i];
          const float y = elems[i + 1];
          const int half_idx = static_cast<int>(lane_id * kElemsPerThread + i) / 2;
          const float cos = load_cache_value(cos_ptr, half_idx);
          const float sin = load_cache_value(sin_ptr, half_idx);
          elems[i] = x * cos - y * sin;
          elems[i + 1] = y * cos + x * sin;
        }
      }
    }

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      input_vec[j] = cast<Packed, fp32x2_t>({elems[2 * j], elems[2 * j + 1]});
    }
    store_as<Storage>(const_cast<void*>(input), input_vec, lane_id);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHeadDim, int64_t kRopeDim, bool kIsNeox, bool kUsePDL, typename DType>
struct QKNormRopeKernel {
  static_assert(kHeadDim <= 256, "Only head_dim <= 256 is supported");
  template <typename IdType>
  static constexpr auto kernel = fused_qknorm_rope_warp<kHeadDim, kRopeDim, kIsNeox, kUsePDL, DType, IdType>;

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto R = SymbolicSize{"rope_dim"};
    auto Dq = SymbolicSize{"q_stride"};
    auto Dk = SymbolicSize{"k_stride"};
    auto Dd = SymbolicSize{"head_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    D.set_value(kHeadDim);
    R.set_value(kRopeDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D}).with_strides({Dq, Dd, 1}).with_dtype<DType>().with_device(device).verify(q);
    TensorMatcher({N, K, D}).with_strides({Dk, Dd, 1}).with_dtype<DType>().with_device(device).verify(k);
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(q_weight).verify(k_weight);
    TensorMatcher({-1, R}).with_dtype<float>().with_device(device).verify(cos_sin_cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(positions);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    const auto q_stride_bytes = static_cast<int64_t>(Dq.unwrap() * sizeof(DType));
    const auto k_stride_bytes = static_cast<int64_t>(Dk.unwrap() * sizeof(DType));
    const auto head_stride_bytes = static_cast<int64_t>(Dd.unwrap() * sizeof(DType));

    const int64_t k_offset = static_cast<int64_t>(num_qo_heads) * head_stride_bytes;
    const auto params = QKNormRopeParams{
        .q_ptr = q.data_ptr(),
        .k_ptr = pointer::offset(k.data_ptr(), -k_offset),
        .q_weight_ptr = q_weight.data_ptr(),
        .k_weight_ptr = k_weight.data_ptr(),
        .cos_sin_cache_ptr = cos_sin_cache.data_ptr(),
        .positions = positions.data_ptr(),
        .q_stride_bytes = q_stride_bytes,
        .k_stride_bytes = k_stride_bytes,
        .head_stride_bytes = head_stride_bytes,
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .num_tokens = num_tokens,
        .eps = eps,
    };

    const auto is_int32 = id_type.is_type<int32_t>();
    const auto selected_kernel = is_int32 ? kernel<int32_t> : kernel<int64_t>;
    const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    static const uint32_t kOccupancyTable[2] = {
        runtime::get_blocks_per_sm(kernel<int32_t>, kThreadsPerBlock),
        runtime::get_blocks_per_sm(kernel<int64_t>, kThreadsPerBlock),
    };
    const auto max_blocks = kOccupancyTable[is_int32 ? 0 : 1] * kNumSM;
    const auto num_works = (num_qo_heads + num_kv_heads) * num_tokens;
    const auto needed_blocks = div_ceil(num_works, kWarpsPerBlock);
    const auto num_blocks = std::min(max_blocks, needed_blocks);
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(selected_kernel, params);
  }
};

}  // namespace
