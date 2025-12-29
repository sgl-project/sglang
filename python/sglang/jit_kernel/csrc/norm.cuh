#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/warp.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

namespace {

[[maybe_unused]]
__device__ auto to_float2(nv_bfloat162 x) -> float2 {
  return __bfloat1622float2(x);
}

[[maybe_unused]]
__device__ auto to_float2(half2 x) -> float2 {
  return __half22float2(x);
}

template <typename T>
__device__ auto from_float2(float2 x) -> T {
  if constexpr (std::is_same_v<T, nv_bfloat162>) {
    return __float22bfloat162_rn(x);
  } else if constexpr (std::is_same_v<T, half2>) {
    return __float22half2_rn(x);
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type");
  }
}

struct QKNormParams {
  void* __restrict__ q;
  void* __restrict__ k;  // k is offset by (-num_qo_heads * head_dim) elements
  int64_t q_stride;
  int64_t k_stride;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  float eps;
  const void* __restrict__ q_weight;
  const void* __restrict__ k_weight;
  uint32_t num_tokens;
};

template <int64_t kHeadDim, typename PackedFloat>
__always_inline __device__ void apply_norm(void* __restrict__ input, const void* __restrict__ weight, float eps) {
  using namespace device;

  constexpr auto kLoopCount = kHeadDim / (kWarpThreads * 2);
  static_assert(kHeadDim % (kWarpThreads * 2) == 0);

  const auto lane_id = threadIdx.x % kWarpThreads;
  float sum_of_squares = 0.0f;

  using vec_t = device_vec<PackedFloat, kLoopCount>;
  auto input_vec = static_cast<const vec_t*>(input)[lane_id];

#pragma unroll
  for (auto i = 0u; i < kLoopCount; ++i) {
    const auto fp16_input = input_vec.data[i];
    const auto fp32_input = to_float2(fp16_input);
    sum_of_squares += fp32_input.x * fp32_input.x;
    sum_of_squares += fp32_input.y * fp32_input.y;
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  const auto norm_factor = rsqrtf(sum_of_squares / kHeadDim + eps);
  const auto weight_vec = static_cast<const vec_t*>(weight)[lane_id];

  vec_t output_vec;
#pragma unroll
  for (auto i = 0u; i < kLoopCount; ++i) {
    const auto fp32_weight = to_float2(weight_vec.data[i]);
    const auto fp32_input = to_float2(input_vec.data[i]);
    output_vec.data[i] = from_float2<PackedFloat>({
        fp32_input.x * norm_factor * fp32_weight.x,
        fp32_input.y * norm_factor * fp32_weight.y,
    });
  }

  static_cast<vec_t*>(input)[lane_id] = output_vec;
}

constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kThreadsPerBlock = kWarpsPerBlock * device::kWarpThreads;

template <int64_t kHeadDim, bool kUsePDL, typename PackedFloat, typename Float>
__global__ void fused_qknorm(const QKNormParams __grid_constant__ params) {
  using namespace device;

  static_assert(sizeof(Float) == 2 && sizeof(PackedFloat) == 4, "Only support FP16/BF16");
  const auto& [q, k, q_stride, k_stride, num_qo_heads, num_kv_heads, eps, q_weight, k_weight, num_tokens] = params;

  const auto num_blks = gridDim.x;
  const auto num_workers = num_blks * kWarpsPerBlock;
  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_works = num_q_and_k_heads * num_tokens;
  const auto start_worker_id = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpThreads;

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (auto idx = start_worker_id; idx < num_works; idx += num_workers) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto load_q = head_id < num_qo_heads;
    const auto input = load_q ? pointer::offset(q, 2 * (token_id * q_stride + head_id * kHeadDim))
                              : pointer::offset(k, 2 * (token_id * k_stride + head_id * kHeadDim));
    const auto weight = load_q ? q_weight : k_weight;
    apply_norm<kHeadDim, PackedFloat>(input, weight, eps);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

template <int64_t kHeadDim, bool kUsePDL>
struct QKNormKernel {
  template <typename PackedFloat, typename Float>
  static constexpr auto qknorm_kernel = fused_qknorm<kHeadDim, kUsePDL, PackedFloat, Float>;

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto Sq = SymbolicSize{"q_stride"};
    auto Sk = SymbolicSize{"k_stride"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};

    TensorMatcher({N, Q, D})  // q input
        .with_strides({Sq, D, 1})
        .with_dtype<nv_bfloat16, half>(dtype)
        .with_device<kDLCUDA>(device)
        .verify(q);
    TensorMatcher({N, K, D})  // k input
        .with_strides({Sk, D, 1})
        .with_dtype<nv_bfloat16, half>(dtype)
        .with_device<kDLCUDA>(device)
        .verify(k);
    TensorMatcher({D})  // weight
        .with_dtype<nv_bfloat16, half>(dtype)
        .with_device<kDLCUDA>(device)
        .verify(q_weight)
        .verify(k_weight);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    const auto head_dim = D.unwrap();
    RuntimeCheck(head_dim == kHeadDim, "Wrong head_dim: ", head_dim, ". Expected:", kHeadDim);

    // NOTE: we offset the k here to reduce computation cost in the kernel
    const auto params = QKNormParams{
        .q = q.data_ptr(),
        .k = pointer::offset(k.data_ptr(), -2 * static_cast<int64_t>(num_qo_heads) * kHeadDim),
        .q_stride = static_cast<int64_t>(Sq.unwrap()),
        .k_stride = static_cast<int64_t>(Sk.unwrap()),
        .num_qo_heads = num_qo_heads,
        .num_kv_heads = num_kv_heads,
        .eps = eps,
        .q_weight = q_weight.data_ptr(),
        .k_weight = k_weight.data_ptr(),
        .num_tokens = num_tokens,
    };

    // only initialize once (static variable) to avoid overhead
    static constexpr auto bf16_kernel = qknorm_kernel<nv_bfloat162, nv_bfloat16>;
    static constexpr auto fp16_kernel = qknorm_kernel<half2, half>;
    static const uint32_t kMaxOccupancyTable[2] = {
        runtime::get_blocks_per_sm(fp16_kernel, kThreadsPerBlock),
        runtime::get_blocks_per_sm(bf16_kernel, kThreadsPerBlock),
    };
    static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);

    // choose kernel based on dtype
    const bool use_bf16 = dtype.is_type<nv_bfloat16>();
    const auto kernel = use_bf16 ? bf16_kernel : fp16_kernel;
    const auto max_occupancy = kMaxOccupancyTable[use_bf16 ? 1 : 0];
    const auto num_works = (num_qo_heads + num_kv_heads) * num_tokens;
    const auto needed_blocks = div_ceil(num_works, kWarpsPerBlock);

    // we use persistent kernel, which limit the number of blocks to reduce overhead
    const auto num_blocks = std::min(kNumSM * max_occupancy, needed_blocks);
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
