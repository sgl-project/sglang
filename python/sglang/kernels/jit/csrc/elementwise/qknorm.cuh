#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace sglang {

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

constexpr uint32_t kWarpsPerBlock = 4;
constexpr uint32_t kThreadsPerBlock = kWarpsPerBlock * device::kWarpThreads;

// Warp-level kernel for head_dim <= 256
template <int64_t kHeadDim, bool kUsePDL, typename Float>
__global__ void fused_qknorm_warp(const QKNormParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kHeadDim>;

  static_assert(sizeof(Float) == 2, "Only support FP16/BF16");
  const auto& [q, k, q_stride, k_stride, num_qo_heads, num_kv_heads, eps, q_weight, k_weight, num_tokens] = params;

  const auto num_blks = gridDim.x;
  const auto num_workers = num_blks * kWarpsPerBlock;
  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_works = num_q_and_k_heads * num_tokens;
  const auto start_worker_id = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpThreads;
  const auto gmem = tile::Memory<Storage>::warp();

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (auto idx = start_worker_id; idx < num_works; idx += num_workers) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto load_q = head_id < num_qo_heads;
    const auto input = load_q ? pointer::offset(q, 2 * (token_id * q_stride + head_id * kHeadDim))
                              : pointer::offset(k, 2 * (token_id * k_stride + head_id * kHeadDim));
    const auto weight = load_q ? q_weight : k_weight;
    const auto input_vec = gmem.load(input);
    const auto weight_vec = gmem.load(weight);
    const auto output_vec = norm::apply_norm_warp<kHeadDim>(input_vec, weight_vec, eps);
    gmem.store(input, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

// Head_dim-128 variant of fused_qknorm_warp: two rows per warp, 16 lanes x 16 B per row.
// Each lane replays the warp kernel's butterfly on two partial sums, so the output is bit-identical.
template <bool kUsePDL, typename Float>
__global__ void fused_qknorm_half_warp(const QKNormParams __grid_constant__ params) {
  using namespace device;

  constexpr int64_t kHeadDim = 128;
  constexpr uint32_t kRowLanes = 16;
  using Packed = packed_t<Float>;
  constexpr uint32_t kPacks = kHeadDim / (2 * kRowLanes);
  using Storage = AlignedVector<Packed, kPacks>;

  static_assert(sizeof(Float) == 2, "Only support FP16/BF16");
  const auto& [q, k, q_stride, k_stride, num_qo_heads, num_kv_heads, eps, q_weight, k_weight, num_tokens] = params;

  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto row_lane = lane_id % kRowLanes;
  const auto num_workers = gridDim.x * kWarpsPerBlock;
  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_rows = num_q_and_k_heads * num_tokens;
  const auto num_pairs = (num_rows + 1) / 2;
  const auto start_worker_id = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpThreads;

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (auto pair = start_worker_id; pair < num_pairs; pair += num_workers) {
    // a missing second row recomputes the last one and skips the store
    const auto wanted_row = 2 * pair + lane_id / kRowLanes;
    const auto row = wanted_row < num_rows ? wanted_row : num_rows - 1;
    const int64_t token_id = row / num_q_and_k_heads;
    const int64_t head_id = row % num_q_and_k_heads;
    const auto load_q = head_id < num_qo_heads;
    const auto input = load_q ? pointer::offset(q, 2 * (token_id * q_stride + head_id * kHeadDim))
                              : pointer::offset(k, 2 * (token_id * k_stride + head_id * kHeadDim));
    const auto weight = load_q ? q_weight : k_weight;
    auto vec = load_as<Storage>(input, row_lane);
    const auto weight_vec = load_as<Storage>(weight, row_lane);

    // part[h] is the sum of warp-kernel lane 2 * row_lane + h, and reduce_sum below
    // is that kernel's 16..2 xor steps; its final xor-1 step adds the pair.
    float part[2] = {0.0f, 0.0f};
#pragma unroll
    for (auto i = 0u; i < kPacks; ++i) {
      const auto x = cast<fp32x2_t>(vec[i]);
      part[i / 2] += x.x * x.x;
      part[i / 2] += x.y * x.y;
    }
    part[0] = warp::reduce_sum<kRowLanes>(part[0]);
    part[1] = warp::reduce_sum<kRowLanes>(part[1]);
    const float norm_factor = math::rsqrt((part[0] + part[1]) / kHeadDim + eps);

#pragma unroll
    for (auto i = 0u; i < kPacks; ++i) {
      const auto x = cast<fp32x2_t>(vec[i]);
      const auto w = cast<fp32x2_t>(weight_vec[i]);
      vec[i] = cast<Packed, fp32x2_t>({x.x * norm_factor * w.x, x.y * norm_factor * w.y});
    }
    if (wanted_row < num_rows) store_as<Storage>(input, vec, row_lane);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

// For CTA level, used for head_dim > 256 (512,1024)
template <int64_t kHeadDim, bool kUsePDL, typename Float>
__global__ void fused_qknorm_cta(const QKNormParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kHeadDim>;

  constexpr auto kNumThreads = host::norm::get_cta_threads<Float, kHeadDim>();
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  static_assert(sizeof(Float) == 2, "Only support FP16/BF16");
  const auto& [q, k, q_stride, k_stride, num_qo_heads, num_kv_heads, eps, q_weight, k_weight, num_tokens] = params;

  const auto num_q_and_k_heads = num_qo_heads + num_kv_heads;
  const auto num_works = num_q_and_k_heads * num_tokens;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[norm::kSmemBufferSize];

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (auto idx = blockIdx.x; idx < num_works; idx += gridDim.x) {
    const int64_t token_id = idx / num_q_and_k_heads;
    const int64_t head_id = idx % num_q_and_k_heads;
    const auto load_q = head_id < num_qo_heads;
    const auto input = load_q ? pointer::offset(q, 2 * (token_id * q_stride + head_id * kHeadDim))
                              : pointer::offset(k, 2 * (token_id * k_stride + head_id * kHeadDim));
    const auto weight = load_q ? q_weight : k_weight;
    const auto input_vec = gmem.load(input);
    const auto weight_vec = gmem.load(weight);
    const auto output_vec = norm::apply_norm_cta<kHeadDim>(input_vec, weight_vec, eps, smem, kNumWarps);
    gmem.store(input, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

// Head_dims that fused_qknorm_half_warp covers
template <int64_t kHeadDim>
inline constexpr bool kUseHalfWarp =
#ifdef USE_ROCM
    false;
#else
    kHeadDim == 128;
#endif

// Warp-level kernel struct for head_dim <= 256
template <int64_t kHeadDim, bool kUsePDL, typename DType, bool kAllowHalfWarp = true>
struct QKNormKernelWarp {
  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(!host::norm::should_use_cta<DType, kHeadDim>(), "Use QKNormKernelCTA for head_dim > 256");
  static constexpr auto kernel = fused_qknorm_warp<kHeadDim, kUsePDL, DType>;
  // kAllowHalfWarp = false keeps the one-row-per-warp kernel, which tests compare against
  static constexpr bool kHalfWarp = kAllowHalfWarp && kUseHalfWarp<kHeadDim>;

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
    auto device = SymbolicDevice{};
    D.set_value(kHeadDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D})  // q input
        .with_strides({Sq, D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, K, D})  // k input
        .with_strides({Sk, D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(q_weight)
        .verify(k_weight);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());

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

    const auto num_rows = (num_qo_heads + num_kv_heads) * num_tokens;
    if constexpr (kHalfWarp) {
      // the half-warp kernel moves 16 B per lane, the warp kernel only 8 B
      const auto address_bits = reinterpret_cast<uintptr_t>(q.data_ptr()) | reinterpret_cast<uintptr_t>(k.data_ptr()) |
                                reinterpret_cast<uintptr_t>(q_weight.data_ptr()) |
                                reinterpret_cast<uintptr_t>(k_weight.data_ptr()) |
                                static_cast<uintptr_t>(2 * (Sq.unwrap() | Sk.unwrap()));
      if (address_bits % 16 == 0) {
        return launch<fused_qknorm_half_warp<kUsePDL, DType>>(params, div_ceil(num_rows, 2u), device.unwrap());
      }
    }
    launch<kernel>(params, num_rows, device.unwrap());
  }

  template <auto kKernel>
  static void launch(const QKNormParams& params, uint32_t num_works, DLDevice device) {
    using namespace host;

    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kKernel, kThreadsPerBlock);
    static const uint32_t kNumSM = runtime::get_sm_count(device.device_id);

    const auto needed_blocks = div_ceil(num_works, kWarpsPerBlock);

    // we use persistent kernel, which limit the number of blocks to reduce overhead
    const auto num_blocks = std::min(kNumSM * max_occupancy, needed_blocks);
    LaunchKernel(num_blocks, kThreadsPerBlock, device)  //
        .enable_pdl(kUsePDL)(kKernel, params);
  }
};

// This goes with fused_qknorm_cta
template <int64_t kHeadDim, bool kUsePDL, typename DType>
struct QKNormKernelCTA {
  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(host::norm::should_use_cta<DType, kHeadDim>(), "Use QKNormKernelWarp for head_dim <= 256");
  static constexpr auto kernel = fused_qknorm_cta<kHeadDim, kUsePDL, DType>;
  static constexpr auto kNumThreads = host::norm::get_cta_threads<DType, kHeadDim>();

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
    auto device = SymbolicDevice{};
    D.set_value(kHeadDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, D})  // q input
        .with_strides({Sq, D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, K, D})  // k input
        .with_strides({Sk, D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(q_weight)
        .verify(k_weight);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());

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

    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kNumThreads);
    static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);

    const auto num_works = (num_qo_heads + num_kv_heads) * num_tokens;

    // we use persistent kernel, which limit the number of blocks to reduce overhead
    const auto num_blocks = std::min<uint32_t>(num_works, max_occupancy * kNumSM);
    LaunchKernel(num_blocks, kNumThreads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

// Unified dispatch: select warp or CTA kernel based on head_dim
template <int64_t kHeadDim, bool kUsePDL, typename DType, bool kAllowHalfWarp = true>
using QKNormKernel = std::conditional_t<
    host::norm::should_use_cta<DType, kHeadDim>(),
    QKNormKernelCTA<kHeadDim, kUsePDL, DType>,
    QKNormKernelWarp<kHeadDim, kUsePDL, DType, kAllowHalfWarp>>;

}  // namespace sglang
