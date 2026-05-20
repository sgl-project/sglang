#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <tvm/ffi/container/tensor.h>

namespace {

struct RMSNormPerTensorQuantParams {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  const float* __restrict__ scale;
  int64_t input_stride;
  int64_t output_stride;
  uint32_t num_tokens;
  float eps;
};

template <int64_t kDim, bool kUsePDL, typename Float>
__global__ void rmsnorm_per_tensor_quant_cta(const RMSNormPerTensorQuantParams __grid_constant__ params) {
  using namespace device;
  using InputStorage = norm::StorageType<Float, kDim>;

  constexpr auto kNumThreads = host::norm::get_cta_threads<Float, kDim>();
  constexpr auto kFP8VecSize = 8u;

  const auto& [input, weight_ptr, output, scale_ptr, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem_in = tile::Memory<InputStorage>::cta(kNumThreads);
  const auto gmem_out = tile::Memory<AlignedVector<fp8_e4m3_t, kFP8VecSize>>::cta(kNumThreads);
  __shared__ float smem[norm::kSmemBufferSize];

  const float inv_scale = 1.0f / (*scale_ptr);

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t i = blockIdx.x; i < num_tokens; i += gridDim.x) {
    const auto input_ptr = pointer::offset<Float>(input, i * input_stride);
    const auto output_ptr = pointer::offset<fp8_e4m3_t>(output, i * output_stride);
    const auto input_vec = gmem_in.load(input_ptr);
    const auto weight_vec = gmem_in.load(weight_ptr);

    AlignedVector<fp8_e4m3_t, kFP8VecSize> output_vec;
    norm::apply_norm_cta_with_epilogue<kDim>(
        input_vec, weight_vec, eps, smem, norm::details::PerTensorQuantFp8Epilogue<4>{output_vec, inv_scale});

    gmem_out.store(output_ptr, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kDim, bool kUsePDL, typename DType>
struct RMSNormPerTensorQuantKernel {
  static_assert(host::norm::should_use_cta<DType, kDim>(), "Hidden size must be > 256 for RMSNormPerTensorQuant");
  static constexpr auto kernel = rmsnorm_per_tensor_quant_cta<kDim, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView scale,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto SI = SymbolicSize{"input_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_strides({SI, 1}).with_dtype<DType>().with_device(device).verify(input);
    TensorMatcher({D}).with_dtype<DType>().with_device(device).verify(weight);
    TensorMatcher({N, D}).with_strides({SO, 1}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output);
    TensorMatcher({1}).with_dtype<float>().with_device(device).verify(scale);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = RMSNormPerTensorQuantParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .scale = static_cast<const float*>(scale.data_ptr()),
        .input_stride = SI.unwrap(),
        .output_stride = SO.unwrap(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    static constexpr auto kNumThreads = norm::get_cta_threads<DType, kDim>();
    static const uint32_t max_occupancy = runtime::get_blocks_per_sm(kernel, kNumThreads);
    static const uint32_t kNumSM = runtime::get_sm_count(device.unwrap().device_id);
    const auto num_blocks = std::min<uint32_t>(num_tokens, max_occupancy * kNumSM);
    LaunchKernel(num_blocks, kNumThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
