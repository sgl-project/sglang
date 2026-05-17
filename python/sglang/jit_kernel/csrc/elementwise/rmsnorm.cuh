#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <tvm/ffi/container/tensor.h>

namespace {

struct RMSNormParams {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  int64_t input_stride;
  int64_t output_stride;
  uint32_t num_tokens;
  float eps;
};

template <int64_t kDim, bool kUsePDL, typename Float>
__global__ void rmsnorm_cta(const RMSNormParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kDim>;

  constexpr auto kNumThreads = host::norm::get_cta_threads<Float, kDim>();
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[norm::kSmemBufferSize];

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (uint32_t i = blockIdx.x; i < num_tokens; i += gridDim.x) {
    const auto input_ptr = pointer::offset<Float>(input, i * input_stride);
    const auto output_ptr = pointer::offset<Float>(output, i * output_stride);
    const auto input_vec = gmem.load(input_ptr);
    const auto weight_vec = gmem.load(weight_ptr);
    const auto output_vec = norm::apply_norm_cta<kDim>(input_vec, weight_vec, eps, smem, kNumWarps);
    gmem.store(output_ptr, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

// Pre-Blackwell: 16B vector, each thread loads/stores twice
template <int64_t kDim, bool kUsePDL, typename Float>
__global__ __launch_bounds__(kDim / 16) void rmsnorm_cta_double(const RMSNormParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;

  constexpr auto kNumThreads = kDim / 16;
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[32];

  PDLWaitPrimary<kUsePDL>();

  const auto input_ptr = pointer::offset<Float>(input, blockIdx.x * input_stride);
  const auto output_ptr = pointer::offset<Float>(output, blockIdx.x * output_stride);

  const auto input_first = gmem.load(input_ptr, 0);
  const auto input_second = gmem.load(input_ptr, 1);
  const auto weight_first = gmem.load(weight_ptr, 0);
  const auto weight_second = gmem.load(weight_ptr, 1);

  float sum_of_squares = 0.0f;
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [x, y] = cast<fp32x2_t>(input_first[j]);
    sum_of_squares += x * x + y * y;
  }
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [x, y] = cast<fp32x2_t>(input_second[j]);
    sum_of_squares += x * x + y * y;
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  const auto warp_id = threadIdx.x / kWarpThreads;
  smem[warp_id] = sum_of_squares;
  __syncthreads();
  if (warp_id == 0) {
    const auto tx = threadIdx.x;
    const auto local_sum = tx < kNumWarps ? smem[tx] : 0.0f;
    sum_of_squares = warp::reduce_sum(local_sum);
    smem[tx] = math::rsqrt(sum_of_squares / kDim + eps);
  }
  __syncthreads();
  const float norm_factor = smem[warp_id];

  Storage output_first, output_second;
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [ix, iy] = cast<fp32x2_t>(input_first[j]);
    const auto [wx, wy] = cast<fp32x2_t>(weight_first[j]);
    output_first[j] = cast<Float2>(fp32x2_t{ix * norm_factor * wx, iy * norm_factor * wy});
  }
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [ix, iy] = cast<fp32x2_t>(input_second[j]);
    const auto [wx, wy] = cast<fp32x2_t>(weight_second[j]);
    output_second[j] = cast<Float2>(fp32x2_t{ix * norm_factor * wx, iy * norm_factor * wy});
  }

  gmem.store(output_ptr, output_first, 0);
  gmem.store(output_ptr, output_second, 1);

  PDLTriggerSecondary<kUsePDL>();
}

// Blackwell: 32B vector, each thread loads/stores once
template <int64_t kDim, bool kUsePDL, typename Float>
__global__ __launch_bounds__(kDim / 16) void rmsnorm_cta_wide(const RMSNormParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 8>;

  constexpr auto kNumThreads = kDim / 16;
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[32];

  PDLWaitPrimary<kUsePDL>();

  const auto input_ptr = pointer::offset<Float>(input, blockIdx.x * input_stride);
  const auto output_ptr = pointer::offset<Float>(output, blockIdx.x * output_stride);

  const auto input_vec = gmem.load(input_ptr);
  const auto weight_vec = gmem.load(weight_ptr);

  float sum_of_squares = 0.0f;
#pragma unroll
  for (auto j = 0u; j < 8u; ++j) {
    const auto [x, y] = cast<fp32x2_t>(input_vec[j]);
    sum_of_squares += x * x + y * y;
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  const auto warp_id = threadIdx.x / kWarpThreads;
  smem[warp_id] = sum_of_squares;
  __syncthreads();
  if (warp_id == 0) {
    const auto tx = threadIdx.x;
    const auto local_sum = tx < kNumWarps ? smem[tx] : 0.0f;
    sum_of_squares = warp::reduce_sum(local_sum);
    smem[tx] = math::rsqrt(sum_of_squares / kDim + eps);
  }
  __syncthreads();
  const float norm_factor = smem[warp_id];

  Storage output_vec;
#pragma unroll
  for (auto j = 0u; j < 8u; ++j) {
    const auto [ix, iy] = cast<fp32x2_t>(input_vec[j]);
    const auto [wx, wy] = cast<fp32x2_t>(weight_vec[j]);
    output_vec[j] = cast<Float2>(fp32x2_t{ix * norm_factor * wx, iy * norm_factor * wy});
  }

  gmem.store(output_ptr, output_vec);

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kDim, bool kUsePDL, typename Float>
__global__ void rmsnorm_warp(const RMSNormParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kDim>;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::warp();

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (uint32_t i = blockIdx.x; i < num_tokens; i += gridDim.x) {
    const auto input_ptr = pointer::offset<Float>(input, i * input_stride);
    const auto output_ptr = pointer::offset<Float>(output, i * output_stride);
    const auto input_vec = gmem.load(input_ptr);
    const auto weight_vec = gmem.load(weight_ptr);
    const auto output_vec = norm::apply_norm_warp<kDim>(input_vec, weight_vec, eps);
    gmem.store(output_ptr, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

template <int64_t kDim, bool kUsePDL, typename DType>
struct RMSNormWarpKernel {
  static_assert(host::norm::is_config_supported<DType, kDim>(), "Unsupported norm configuration");
  static_assert(kDim <= 256, "Use RMSNormKernel for hidden sizes > 256");
  static constexpr auto kernel = rmsnorm_warp<kDim, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // input
        .with_strides({SI, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({N, D})  // output
        .with_strides({SO, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = RMSNormParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    static constexpr uint32_t kNumThreads = device::kWarpThreads;
    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kNumThreads);
    static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    const auto num_blocks = std::min<uint32_t>(num_tokens, max_occupancy * kNumSM);
    LaunchKernel(num_blocks, kNumThreads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

template <int64_t kDim, bool kUsePDL, typename DType>
struct RMSNormKernel {
  static_assert(host::norm::should_use_cta<DType, kDim>(), "Hidden size invalid for RMSNorm");
  static constexpr auto kernel = rmsnorm_cta<kDim, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // input
        .with_strides({SI, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({N, D})  // output
        .with_strides({SO, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = RMSNormParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    static constexpr auto kNumThreads = norm::get_cta_threads<DType, kDim>();
    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kNumThreads);
    static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    const auto num_blocks = std::min<uint32_t>(num_tokens, max_occupancy * kNumSM);
    LaunchKernel(num_blocks, kNumThreads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

template <int64_t kDim, bool kUsePDL, typename DType>
struct RMSNormHalfKernel {
  static_assert(kDim % 512 == 0 && sizeof(DType) == 2);
#if SGL_ARCH_BLACKWELL_OR_GREATER
  static constexpr auto kernel = rmsnorm_cta_wide<kDim, kUsePDL, DType>;
#else
  static constexpr auto kernel = rmsnorm_cta_double<kDim, kUsePDL, DType>;
#endif
  static constexpr auto kBlockSize = static_cast<uint32_t>(kDim / 16);

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // input
        .with_strides({SI, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({N, D})  // output
        .with_strides({SO, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = RMSNormParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    LaunchKernel(num_tokens, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
