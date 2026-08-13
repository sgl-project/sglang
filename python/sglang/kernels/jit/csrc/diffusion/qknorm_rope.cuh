#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <dlpack/dlpack.h>

#include <cstdint>
#include <type_traits>

namespace sglang {

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

struct QKNormRopePackKVParams : QKNormRopeParams {
  const void* __restrict__ v_ptr;
  const void* __restrict__ k_prefix_ptr;
  const void* __restrict__ v_prefix_ptr;
  void* __restrict__ packed_k_ptr;
  void* __restrict__ packed_v_ptr;
  int64_t v_stride_bytes;
  int64_t k_prefix_stride_bytes;
  int64_t v_prefix_stride_bytes;
  int64_t packed_token_stride_bytes;
  int64_t packed_head_stride_bytes;
  uint32_t batch_size;
  uint32_t prefix_tokens;
  uint32_t suffix_tokens;
};

template <bool kPackKV>
using QKNormRopeParamsT = std::conditional_t<kPackKV, QKNormRopePackKVParams, QKNormRopeParams>;

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

template <typename CacheDType>
SGL_DEVICE CacheDType load_cache_value(const CacheDType* ptr, int64_t idx) {
#ifdef USE_ROCM
  return ptr[idx];
#else
  return __ldg(ptr + idx);
#endif
}

template <typename T>
SGL_DEVICE T rotary_mul_rn(T lhs, T rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030 || __CUDA_ARCH__ >= 1200)
  uint16_t lhs_bits;
  uint16_t rhs_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    lhs_bits = __bfloat16_as_ushort(lhs);
    rhs_bits = __bfloat16_as_ushort(rhs);
  } else {
    lhs_bits = __half_as_ushort(lhs);
    rhs_bits = __half_as_ushort(rhs);
  }
  uint16_t out_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    asm volatile("mul.rn.bf16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  } else {
    asm volatile("mul.rn.f16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  }
  if constexpr (std::is_same_v<T, bf16_t>) {
    return __ushort_as_bfloat16(out_bits);
  } else {
    return __ushort_as_half(out_bits);
  }
#else
  return lhs * rhs;
#endif
}

template <typename T>
SGL_DEVICE T rotary_add(T x, T cos, T y, T sin) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030 || __CUDA_ARCH__ >= 1200)
  // nvcc may contract the packed local expression on Blackwell SM103/SM120
  // even though the reference RoPE kernel rounds both products to the
  // activation dtype first.
  const T lhs = rotary_mul_rn(x, cos);
  const T rhs = rotary_mul_rn(y, sin);
  uint16_t lhs_bits;
  uint16_t rhs_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    lhs_bits = __bfloat16_as_ushort(lhs);
    rhs_bits = __bfloat16_as_ushort(rhs);
  } else {
    lhs_bits = __half_as_ushort(lhs);
    rhs_bits = __half_as_ushort(rhs);
  }
  uint16_t out_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    asm volatile("add.rn.bf16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  } else {
    asm volatile("add.rn.f16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  }
  if constexpr (std::is_same_v<T, bf16_t>) {
    return __ushort_as_bfloat16(out_bits);
  } else {
    return __ushort_as_half(out_bits);
  }
#else
  return x * cos + y * sin;
#endif
}

template <typename T>
SGL_DEVICE T rotary_sub(T x, T cos, T y, T sin) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030 || __CUDA_ARCH__ >= 1200)
  const T lhs = rotary_mul_rn(x, cos);
  const T rhs = rotary_mul_rn(y, sin);
  uint16_t lhs_bits;
  uint16_t rhs_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    lhs_bits = __bfloat16_as_ushort(lhs);
    rhs_bits = __bfloat16_as_ushort(rhs);
  } else {
    lhs_bits = __half_as_ushort(lhs);
    rhs_bits = __half_as_ushort(rhs);
  }
  uint16_t out_bits;
  if constexpr (std::is_same_v<T, bf16_t>) {
    asm volatile("sub.rn.bf16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  } else {
    asm volatile("sub.rn.f16 %0, %1, %2;" : "=h"(out_bits) : "h"(lhs_bits), "h"(rhs_bits));
  }
  if constexpr (std::is_same_v<T, bf16_t>) {
    return __ushort_as_bfloat16(out_bits);
  } else {
    return __ushort_as_half(out_bits);
  }
#else
  return x * cos - y * sin;
#endif
}

template <
    int64_t kHeadDim,
    int64_t kRopeDim,
    bool kIsNeox,
    bool kUsePDL,
    typename DType,
    typename CacheDType,
    bool kRoundNormBeforeRope,
    bool kPackKV,
    typename IdType>
__global__ void fused_qknorm_rope_warp(const QKNormRopeParamsT<kPackKV> __grid_constant__ params) {
  using namespace device;

  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(kHeadDim <= 256, "Only warp-level fused qknorm+rope is supported");
  static_assert(kHeadDim % kWarpThreads == 0, "head_dim must be divisible by warp size");

  constexpr uint32_t kElemsPerThread = kHeadDim / kWarpThreads;
  constexpr uint32_t kVecSize = kElemsPerThread / 2;
  constexpr uint32_t kRotaryLanes = kRopeDim / kElemsPerThread;
  constexpr uint32_t kHalfRotaryLanes = kRotaryLanes / 2;
  constexpr uint32_t kActiveMask = active_mask<kRotaryLanes>();
  constexpr int64_t kCosSinStrideBytes = kRopeDim * sizeof(CacheDType);

  static_assert(kElemsPerThread % 2 == 0, "Each lane must own an even number of elements");
  static_assert(kRopeDim > 0 && kRopeDim <= kHeadDim, "Invalid rope dimension");
  static_assert(kRopeDim % kElemsPerThread == 0, "rope_dim must align with per-lane vector width");
  static_assert(
      !kIsNeox || (kRotaryLanes >= 2 && kRotaryLanes % 2 == 0),
      "NeoX fused qknorm+rope requires an even rotary lane count");
  static_assert(
      !kRoundNormBeforeRope || std::is_same_v<DType, CacheDType>,
      "Rounded QKNorm+RoPE requires cache and activation dtypes to match");

  using Packed = packed_t<DType>;
  using Storage = AlignedVector<Packed, kVecSize>;

  const auto& [q_ptr, k_ptr, q_weight_ptr, k_weight_ptr, cos_sin_cache_ptr, positions, q_stride_bytes, k_stride_bytes, head_stride_bytes, num_qo_heads, num_kv_heads, num_tokens, eps] =
      static_cast<const QKNormRopeParams&>(params);

  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t start_worker_id = blockIdx.x * kWarpsPerBlock + warp_id;
  const uint32_t num_workers = gridDim.x * kWarpsPerBlock;
  const uint32_t num_qk_heads = num_qo_heads + num_kv_heads;
  const uint32_t num_qk_works = num_qk_heads * num_tokens;
  uint32_t num_prefix_works = 0;
  uint32_t num_works = num_qk_works;
  if constexpr (kPackKV) {
    num_prefix_works = params.batch_size * params.prefix_tokens * num_kv_heads;
    num_works += 2 * num_prefix_works + num_tokens * num_kv_heads;
  }

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t idx = start_worker_id; idx < num_works; idx += num_workers) {
    if constexpr (kPackKV) {
      if (idx >= num_qk_works) {
        const uint32_t copy_idx = idx - num_qk_works;
        const bool copy_k_prefix = copy_idx < num_prefix_works;
        const bool copy_v_prefix = copy_idx >= num_prefix_works && copy_idx < 2 * num_prefix_works;
        const uint32_t local_idx =
            copy_k_prefix ? copy_idx : (copy_v_prefix ? copy_idx - num_prefix_works : copy_idx - 2 * num_prefix_works);
        const uint32_t token_id = local_idx / num_kv_heads;
        const uint32_t head_id = local_idx % num_kv_heads;
        const bool copy_prefix = copy_k_prefix || copy_v_prefix;
        const uint32_t batch_id = token_id / (copy_prefix ? params.prefix_tokens : params.suffix_tokens);
        const uint32_t sequence_id = token_id % (copy_prefix ? params.prefix_tokens : params.suffix_tokens);
        const uint32_t packed_token_id = batch_id * (params.prefix_tokens + params.suffix_tokens) +
                                         (copy_prefix ? sequence_id : params.prefix_tokens + sequence_id);
        const void* input = nullptr;
        void* output = nullptr;
        if (copy_k_prefix) {
          input = pointer::offset(
              params.k_prefix_ptr, token_id * params.k_prefix_stride_bytes, head_id * head_stride_bytes);
          output = pointer::offset(
              params.packed_k_ptr,
              packed_token_id * params.packed_token_stride_bytes,
              head_id * params.packed_head_stride_bytes);
        } else {
          const void* v_ptr = copy_v_prefix ? params.v_prefix_ptr : params.v_ptr;
          const int64_t v_stride = copy_v_prefix ? params.v_prefix_stride_bytes : params.v_stride_bytes;
          input = pointer::offset(v_ptr, token_id * v_stride, head_id * head_stride_bytes);
          output = pointer::offset(
              params.packed_v_ptr,
              packed_token_id * params.packed_token_stride_bytes,
              head_id * params.packed_head_stride_bytes);
        }
        const auto copy_vec = load_as<Storage>(input, lane_id);
        store_as<Storage>(output, copy_vec, lane_id);
        continue;
      }
    }

    const uint32_t token_id = idx / num_qk_heads;
    const uint32_t head_id = idx % num_qk_heads;
    const bool load_q = head_id < num_qo_heads;
    const void* input = load_q ? pointer::offset(q_ptr, token_id * q_stride_bytes, head_id * head_stride_bytes)
                               : pointer::offset(k_ptr, token_id * k_stride_bytes, head_id * head_stride_bytes);
    void* output = const_cast<void*>(input);
    if constexpr (kPackKV) {
      if (!load_q) {
        const uint32_t batch_id = token_id / params.suffix_tokens;
        const uint32_t sequence_id = token_id % params.suffix_tokens;
        const uint32_t kv_head_id = head_id - num_qo_heads;
        const uint32_t packed_token_id =
            batch_id * (params.prefix_tokens + params.suffix_tokens) + params.prefix_tokens + sequence_id;
        output = pointer::offset(
            params.packed_k_ptr,
            packed_token_id * params.packed_token_stride_bytes,
            kv_head_id * params.packed_head_stride_bytes);
      }
    }
    const void* weight_ptr = load_q ? q_weight_ptr : k_weight_ptr;

    auto input_vec = load_as<Storage>(input, lane_id);
    const auto weight_vec = load_as<Storage>(weight_ptr, lane_id);

    if constexpr (kRoundNormBeforeRope) {
      auto output_vec = norm::apply_norm_warp<kHeadDim>(input_vec, weight_vec, eps);
      const auto pos = static_cast<int64_t>(static_cast<const IdType*>(positions)[token_id]);
      const auto cos_ptr = static_cast<const CacheDType*>(pointer::offset(cos_sin_cache_ptr, pos * kCosSinStrideBytes));
      const auto sin_ptr = cos_ptr + kRopeDim / 2;

      if constexpr (kIsNeox) {
        if (lane_id < kRotaryLanes) {
          const auto partner_lane =
              lane_id < kHalfRotaryLanes ? lane_id + kHalfRotaryLanes : lane_id - kHalfRotaryLanes;
#pragma unroll
          for (uint32_t j = 0; j < kVecSize; ++j) {
            auto partner_vec = output_vec[j];
            auto partner_bits = reinterpret_cast<const uint32_t&>(partner_vec);
            partner_bits = __shfl_sync(kActiveMask, partner_bits, partner_lane);
            reinterpret_cast<uint32_t&>(partner_vec) = partner_bits;
            auto& values = unpack(output_vec[j]);
            const auto& partner_values = unpack(partner_vec);
#pragma unroll
            for (uint32_t i = 0; i < 2; ++i) {
              const auto half_idx = (lane_id % kHalfRotaryLanes) * kElemsPerThread + 2 * j + i;
              const auto cos = load_cache_value(cos_ptr, half_idx);
              const auto sin = load_cache_value(sin_ptr, half_idx);
              values[i] = lane_id < kHalfRotaryLanes ? rotary_sub(values[i], cos, partner_values[i], sin)
                                                     : rotary_add(values[i], cos, partner_values[i], sin);
            }
          }
        }
      } else {
        if (lane_id < kRotaryLanes) {
#pragma unroll
          for (uint32_t j = 0; j < kVecSize; ++j) {
            auto& values = unpack(output_vec[j]);
            const auto half_idx = lane_id * kElemsPerThread / 2 + j;
            const auto cos = load_cache_value(cos_ptr, half_idx);
            const auto sin = load_cache_value(sin_ptr, half_idx);
            const auto x = values[0];
            const auto y = values[1];
            values[0] = rotary_sub(x, cos, y, sin);
            values[1] = rotary_add(y, cos, x, sin);
          }
        }
      }
      store_as<Storage>(output, output_vec, lane_id);
      continue;
    }

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
        const auto cos_ptr =
            static_cast<const CacheDType*>(pointer::offset(cos_sin_cache_ptr, pos * kCosSinStrideBytes));
        const auto sin_ptr = cos_ptr + kRopeDim / 2;
        const auto partner_lane = lane_id < kHalfRotaryLanes ? lane_id + kHalfRotaryLanes : lane_id - kHalfRotaryLanes;

#pragma unroll
        for (uint32_t i = 0; i < kElemsPerThread; ++i) {
          float swapped = __shfl_sync(kActiveMask, elems[i], partner_lane);
          if (lane_id < kHalfRotaryLanes) {
            swapped = -swapped;
          }
          const auto half_idx = (lane_id % kHalfRotaryLanes) * kElemsPerThread + i;
          const float cos = cast<fp32_t>(load_cache_value(cos_ptr, half_idx));
          const float sin = cast<fp32_t>(load_cache_value(sin_ptr, half_idx));
          elems[i] = elems[i] * cos + swapped * sin;
        }
      }
    } else {
      if (lane_id < kRotaryLanes) {
        const auto pos = static_cast<int64_t>(static_cast<const IdType*>(positions)[token_id]);
        const auto cos_ptr =
            static_cast<const CacheDType*>(pointer::offset(cos_sin_cache_ptr, pos * kCosSinStrideBytes));
        const auto sin_ptr = cos_ptr + kRopeDim / 2;

#pragma unroll
        for (uint32_t i = 0; i < kElemsPerThread; i += 2) {
          const float x = elems[i];
          const float y = elems[i + 1];
          const int half_idx = static_cast<int>(lane_id * kElemsPerThread + i) / 2;
          const float cos = cast<fp32_t>(load_cache_value(cos_ptr, half_idx));
          const float sin = cast<fp32_t>(load_cache_value(sin_ptr, half_idx));
          elems[i] = x * cos - y * sin;
          elems[i + 1] = y * cos + x * sin;
        }
      }
    }

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      input_vec[j] = cast<Packed, fp32x2_t>({elems[2 * j], elems[2 * j + 1]});
    }
    store_as<Storage>(output, input_vec, lane_id);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <
    int64_t kHeadDim,
    int64_t kRopeDim,
    bool kIsNeox,
    bool kUsePDL,
    typename DType,
    typename CacheDType,
    bool kRoundNormBeforeRope>
struct QKNormRopeKernel {
  static_assert(kHeadDim <= 256, "Only head_dim <= 256 is supported");
  template <typename IdType>
  static constexpr auto kernel = fused_qknorm_rope_warp<
      kHeadDim,
      kRopeDim,
      kIsNeox,
      kUsePDL,
      DType,
      CacheDType,
      kRoundNormBeforeRope,
      false,
      IdType>;

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
    TensorMatcher({-1, R}).with_dtype<CacheDType>().with_device(device).verify(cos_sin_cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(positions);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    if (num_tokens == 0 || (num_qo_heads == 0 && num_kv_heads == 0)) return;
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

template <
    int64_t kHeadDim,
    int64_t kRopeDim,
    bool kIsNeox,
    bool kUsePDL,
    typename DType,
    typename CacheDType,
    bool kRoundNormBeforeRope>
struct QKNormRopePackKVKernel {
  template <typename IdType>
  static constexpr auto kernel = fused_qknorm_rope_warp<
      kHeadDim,
      kRopeDim,
      kIsNeox,
      kUsePDL,
      DType,
      CacheDType,
      kRoundNormBeforeRope,
      true,
      IdType>;

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_prefix,
      const tvm::ffi::TensorView v_prefix,
      const tvm::ffi::TensorView packed_k,
      const tvm::ffi::TensorView packed_v,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      int64_t batch_size,
      int64_t prefix_tokens,
      int64_t suffix_tokens,
      float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto NP = SymbolicSize{"num_prefix_tokens"};
    auto B = SymbolicSize{"batch_size"};
    auto T = SymbolicSize{"packed_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto R = SymbolicSize{"rope_dim"};
    auto Dq = SymbolicSize{"q_stride"};
    auto Dk = SymbolicSize{"k_stride"};
    auto Dv = SymbolicSize{"v_stride"};
    auto Dkp = SymbolicSize{"k_prefix_stride"};
    auto Dvp = SymbolicSize{"v_prefix_stride"};
    auto Dd = SymbolicSize{"head_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    N.set_value(batch_size * suffix_tokens);
    NP.set_value(batch_size * prefix_tokens);
    B.set_value(batch_size);
    T.set_value(prefix_tokens + suffix_tokens);
    D.set_value(kHeadDim);
    R.set_value(kRopeDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D}).with_strides({Dq, Dd, 1}).with_dtype<DType>().with_device(device).verify(q);
    TensorMatcher({N, K, D}).with_strides({Dk, Dd, 1}).with_dtype<DType>().with_device(device).verify(k);
    TensorMatcher({N, K, D}).with_strides({Dv, Dd, 1}).with_dtype<DType>().with_device(device).verify(v);
    TensorMatcher({NP, K, D}).with_strides({Dkp, Dd, 1}).with_dtype<DType>().with_device(device).verify(k_prefix);
    TensorMatcher({NP, K, D}).with_strides({Dvp, Dd, 1}).with_dtype<DType>().with_device(device).verify(v_prefix);
    TensorMatcher({B, T, K, D}).with_dtype<DType>().with_device(device).verify(packed_k).verify(packed_v);
    RuntimeCheck(packed_k.is_contiguous(), "packed_k must be contiguous");
    RuntimeCheck(packed_v.is_contiguous(), "packed_v must be contiguous");
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(q_weight).verify(k_weight);
    TensorMatcher({-1, R}).with_dtype<CacheDType>().with_device(device).verify(cos_sin_cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(positions);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    if (num_tokens == 0 || (num_qo_heads == 0 && num_kv_heads == 0)) return;
    const auto head_stride_bytes = static_cast<int64_t>(Dd.unwrap() * sizeof(DType));
    const int64_t k_offset = static_cast<int64_t>(num_qo_heads) * head_stride_bytes;
    QKNormRopePackKVParams params{};
    params.q_ptr = q.data_ptr();
    params.k_ptr = pointer::offset(k.data_ptr(), -k_offset);
    params.q_weight_ptr = q_weight.data_ptr();
    params.k_weight_ptr = k_weight.data_ptr();
    params.cos_sin_cache_ptr = cos_sin_cache.data_ptr();
    params.positions = positions.data_ptr();
    params.q_stride_bytes = static_cast<int64_t>(Dq.unwrap() * sizeof(DType));
    params.k_stride_bytes = static_cast<int64_t>(Dk.unwrap() * sizeof(DType));
    params.head_stride_bytes = head_stride_bytes;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.num_tokens = num_tokens;
    params.eps = eps;
    params.v_ptr = v.data_ptr();
    params.k_prefix_ptr = k_prefix.data_ptr();
    params.v_prefix_ptr = v_prefix.data_ptr();
    params.packed_k_ptr = packed_k.data_ptr();
    params.packed_v_ptr = packed_v.data_ptr();
    params.v_stride_bytes = static_cast<int64_t>(Dv.unwrap() * sizeof(DType));
    params.k_prefix_stride_bytes = static_cast<int64_t>(Dkp.unwrap() * sizeof(DType));
    params.v_prefix_stride_bytes = static_cast<int64_t>(Dvp.unwrap() * sizeof(DType));
    params.packed_token_stride_bytes = static_cast<int64_t>(num_kv_heads * kHeadDim * sizeof(DType));
    params.packed_head_stride_bytes = static_cast<int64_t>(kHeadDim * sizeof(DType));
    params.batch_size = static_cast<uint32_t>(batch_size);
    params.prefix_tokens = static_cast<uint32_t>(prefix_tokens);
    params.suffix_tokens = static_cast<uint32_t>(suffix_tokens);

    const auto is_int32 = id_type.is_type<int32_t>();
    const auto selected_kernel = is_int32 ? kernel<int32_t> : kernel<int64_t>;
    const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    static const uint32_t kOccupancyTable[2] = {
        runtime::get_blocks_per_sm(kernel<int32_t>, kThreadsPerBlock),
        runtime::get_blocks_per_sm(kernel<int64_t>, kThreadsPerBlock),
    };
    const auto max_blocks = kOccupancyTable[is_int32 ? 0 : 1] * kNumSM;
    const uint32_t num_prefix_works = static_cast<uint32_t>(batch_size * prefix_tokens) * num_kv_heads;
    const uint32_t num_works =
        (num_qo_heads + num_kv_heads) * num_tokens + 2 * num_prefix_works + num_tokens * num_kv_heads;
    const auto needed_blocks = div_ceil(num_works, kWarpsPerBlock);
    const auto num_blocks = std::min(max_blocks, needed_blocks);
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(selected_kernel, params);
  }
};

}  // namespace sglang
