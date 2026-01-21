#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>

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

  void* output_ptr = nullptr;
  Storage output_vec;
  for (uint32_t i = blockIdx.x; i < num_tokens; i += gridDim.x) {
    const auto input_ptr = pointer::offset<Float>(input, i * input_stride);
    const auto input_vec = gmem.load(input_ptr);
    const auto weight_vec = gmem.load(weight_ptr);
    if (output_ptr != nullptr) {
      gmem.store(output_ptr, output_vec);
    }
    output_ptr = pointer::offset<Float>(output, i * output_stride);
    output_vec = norm::apply_norm_cta<kDim>(input_vec, weight_vec, eps, smem, kNumWarps);
  }
  gmem.store(output_ptr, output_vec);

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

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

}  // namespace
