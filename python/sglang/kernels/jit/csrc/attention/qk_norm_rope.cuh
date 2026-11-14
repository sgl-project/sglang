#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <cstdint>
#include <limits>

namespace sglang {

template <typename T, int64_t kHeadDim, typename IdType, bool kUsePDL>
__global__ void qk_norm_rope_kernel(
    const T* __restrict__ q,
    T* __restrict__ k,
    T* __restrict__ out,
    const T* __restrict__ qw,
    const T* __restrict__ kw,
    const float* __restrict__ cache,
    const IdType* __restrict__ positions,
    int64_t q_stride,
    int64_t k_stride,
    uint32_t q_heads,
    uint32_t k_heads,
    uint32_t num_rows,
    float eps) {
  using namespace device;
  constexpr uint32_t kLanes = kHeadDim == 128 ? 16 : 32;
  constexpr uint32_t kPairs = kHeadDim / (2 * kLanes);
  using Storage = AlignedVector<packed_t<T>, kPairs>;
  const auto lane = get_lane_id<kLanes>();
  const auto worker = (blockIdx.x * blockDim.x + threadIdx.x) / kLanes;
  const auto stride = gridDim.x * blockDim.x / kLanes;
  PDLWaitPrimary<kUsePDL>();
  // Both half-warps participate in shuffles, including an unmatched final row.
  const auto padded_rows = num_rows + (kHeadDim == 128 && num_rows % 2);
  for (uint64_t row = worker; row < padded_rows; row += stride) {
    const bool valid = row < num_rows;
    const int64_t token = valid ? row / (q_heads + k_heads) : 0;
    const auto head = valid ? static_cast<uint32_t>(row % (q_heads + k_heads)) : 0;
    const bool is_q = head < q_heads;
    const auto head_offset = int64_t{is_q ? head : head - q_heads} * kHeadDim;
    const auto src = is_q ? q + token * q_stride + head_offset : k + token * k_stride + head_offset;
    const auto dst = is_q ? out + (token * q_heads + head) * kHeadDim : k + token * k_stride + head_offset;
    Storage input;
    if (valid)
      input = load_as<Storage>(src, lane);
    else
      input.fill(cast<packed_t<T>, fp32x2_t>({0.0f, 0.0f}));
    const auto weight = load_as<Storage>(is_q ? qw : kw, lane);
    const int64_t pos = positions[token];
    fp32x2_t cos[kPairs], sin[kPairs];
#pragma unroll
    for (auto i = 0u; i < kPairs; ++i) {
      const auto dim = (lane % (kLanes / 2)) * (2 * kPairs) + 2 * i;
      cos[i] = load_as<fp32x2_t>(cache + pos * kHeadDim + dim);
      sin[i] = load_as<fp32x2_t>(cache + pos * kHeadDim + kHeadDim / 2 + dim);
    }
    auto normalized = [&] {
      if constexpr (kHeadDim == 128) {
        return norm::apply_norm_half_warp(input, weight, eps);
      } else {
        return norm::apply_norm_warp<kHeadDim>(input, weight, eps);
      }
    }();
#pragma unroll
    for (auto i = 0u; i < kPairs; ++i) {
      const auto x = cast<fp32x2_t>(normalized[i]);
      const auto partner_lane = lane ^ (kLanes / 2);
      const auto bits = __shfl_sync(0xffffffffu, load_as<uint32_t>(&normalized[i]), partner_lane, kLanes);
      packed_t<T> partner;
      store_as<uint32_t>(&partner, bits);
      const auto y = cast<fp32x2_t>(partner);
      const auto sign = lane < kLanes / 2 ? -1.0f : 1.0f;
      normalized[i] = cast<packed_t<T>, fp32x2_t>(
          {__fmaf_rn(x.x, cos[i].x, sign * __fmul_rn(y.x, sin[i].x)),
           __fmaf_rn(x.y, cos[i].y, sign * __fmul_rn(y.y, sin[i].y))});
    }
    if (valid) store_as<Storage>(dst, normalized, lane);
  }
  PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Normalize and apply cached full-head NeoX RoPE; write separate Q and in-place K.
 * \tparam T FP16 or BF16 activation and weight type
 * \tparam kHeadDim Head dimension: 64, 128, or 256
 * \tparam kUsePDL Whether to use programmatic dependent launch
 * \param q Read-only query heads, with contiguous columns and aligned row strides
 * \param k Key heads, overwritten in their original storage
 * \param out Contiguous query output, disjoint from Q and K
 * \param qw Query norm weights
 * \param kw Key norm weights
 * \param cache Contiguous FP32 cosine/sine table; positions must index valid rows
 * \param positions Contiguous int32/int64 position indices
 * \param eps Shared Q/K normalization epsilon
 */
template <typename T, int64_t kHeadDim, bool kUsePDL>
void qk_norm_rope(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView qw,
    tvm::ffi::TensorView kw,
    tvm::ffi::TensorView cache,
    tvm::ffi::TensorView positions,
    float eps) {
  using namespace host;
  static_assert(kHeadDim == 64 || kHeadDim == 128 || kHeadDim == 256);
  static_assert(std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>);
  constexpr int64_t kLanes = kHeadDim == 128 ? 16 : 32;
  constexpr auto kAlignment = kHeadDim * sizeof(T) / kLanes;
  auto N = SymbolicSize{"tokens"};
  auto Q = SymbolicSize{"query_heads"};
  auto K = SymbolicSize{"key_heads"};
  auto Sq = SymbolicSize{"query_stride"};
  auto Sk = SymbolicSize{"key_stride"};
  auto P = SymbolicSize{"positions"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({N, Q, kHeadDim})
      .with_strides({Sq, kHeadDim, 1})
      .with_dtype<T>()
      .with_device(device)
      .ensure_alignment(kAlignment)
      .verify(q);
  TensorMatcher({N, K, kHeadDim})
      .with_strides({Sk, kHeadDim, 1})
      .with_dtype<T>()
      .with_device(device)
      .ensure_alignment(kAlignment)
      .verify(k);
  TensorMatcher({N, Q, kHeadDim}).with_dtype<T>().with_device(device).ensure_alignment(kAlignment).verify(out);
  TensorMatcher({kHeadDim}).with_dtype<T>().with_device(device).ensure_alignment(kAlignment).verify(qw).verify(kw);
  TensorMatcher({P, kHeadDim}).with_dtype<float>().with_device(device).ensure_alignment(8).verify(cache);
  TensorMatcher({N}).with_dtype<int32_t, int64_t>().with_device(device).verify(positions);
  CHECK_HOST(Q.unwrap() > 0 && K.unwrap() > 0 && P.unwrap() > 0);
  CHECK_HOST(Sq.unwrap() >= Q.unwrap() * kHeadDim && Sk.unwrap() >= K.unwrap() * kHeadDim);
  const auto rows = N.unwrap() * (Q.unwrap() + K.unwrap());
  CHECK_HOST(rows < std::numeric_limits<uint32_t>::max());
  if (rows == 0) return;
  const auto launch = [&]<typename IdType>() {
    constexpr auto kernel = qk_norm_rope_kernel<T, kHeadDim, IdType, kUsePDL>;
    constexpr uint32_t kThreads = 256;
    static const auto max_occupancy = runtime::get_blocks_per_sm(kernel, kThreads);
    const auto blocks = std::min(
        runtime::get_sm_count(device.unwrap().device_id) * max_occupancy,
        div_ceil(static_cast<uint32_t>(rows), static_cast<uint32_t>(kThreads / kLanes)));
    LaunchKernel(blocks, kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(
            kernel,
            static_cast<const T*>(q.data_ptr()),
            static_cast<T*>(k.data_ptr()),
            static_cast<T*>(out.data_ptr()),
            static_cast<const T*>(qw.data_ptr()),
            static_cast<const T*>(kw.data_ptr()),
            static_cast<const float*>(cache.data_ptr()),
            static_cast<const IdType*>(positions.data_ptr()),
            Sq.unwrap(),
            Sk.unwrap(),
            static_cast<uint32_t>(Q.unwrap()),
            static_cast<uint32_t>(K.unwrap()),
            static_cast<uint32_t>(rows),
            eps);
  };
  if (is_type<int32_t>(positions.dtype()))
    launch.template operator()<int32_t>();
  else
    launch.template operator()<int64_t>();
}

}  // namespace sglang
