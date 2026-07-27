#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/impl/norm.cuh>

#include <tvm/ffi/container/tensor.h>

namespace {

struct FusedEHNormParams {
  const void* __restrict__ embeds;
  const void* __restrict__ previous_hidden;
  const void* __restrict__ enorm_weight;
  const void* __restrict__ hnorm_weight;
  void* __restrict__ output;
  int64_t embeds_stride;
  int64_t previous_hidden_stride;
  int64_t output_stride;
  float eps;
};

template <int64_t kHidden, bool kUsePDL, typename T>
__global__ void fused_eh_norm_kernel(const __grid_constant__ FusedEHNormParams params) {
  using namespace device;
  using Storage = norm::StorageType<T, kHidden>;

  constexpr auto kNumThreads = host::norm::get_cta_threads<T, kHidden>();
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const auto embeds = static_cast<const T*>(pointer::offset<T>(params.embeds, blockIdx.x * params.embeds_stride));
  const auto previous_hidden =
      static_cast<const T*>(pointer::offset<T>(params.previous_hidden, blockIdx.x * params.previous_hidden_stride));
  const auto enorm_weight = static_cast<const T*>(params.enorm_weight);
  const auto hnorm_weight = static_cast<const T*>(params.hnorm_weight);
  const auto output = static_cast<T*>(pointer::offset<T>(params.output, blockIdx.x * params.output_stride));

  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[norm::kSmemBufferSize];

  PDLWaitPrimary<kUsePDL>();

  const auto embeds_vec = gmem.load(embeds);
  const auto enorm_weight_vec = gmem.load(enorm_weight);
  const auto embeds_output_vec =
      norm::apply_norm_cta<kHidden>(embeds_vec, enorm_weight_vec, params.eps, smem, kNumWarps);
  gmem.store(output, embeds_output_vec);

  const auto prev_vec = gmem.load(previous_hidden);
  const auto hnorm_weight_vec = gmem.load(hnorm_weight);
  const auto prev_output_vec = norm::apply_norm_cta<kHidden>(prev_vec, hnorm_weight_vec, params.eps, smem, kNumWarps);
  gmem.store(output + kHidden, prev_output_vec);

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHidden, bool kUsePDL, typename T>
struct FusedEHNormKernel {
  static_assert(host::norm::is_config_supported<T, kHidden>(), "Unsupported norm configuration");
  static_assert(host::norm::should_use_cta<T, kHidden>(), "fused_eh_norm requires CTA norm");
  static constexpr auto kernel = fused_eh_norm_kernel<kHidden, kUsePDL, T>;
  static constexpr uint32_t kBlockSize = host::norm::get_cta_threads<T, kHidden>();

  static void
  run(const tvm::ffi::TensorView embeds,
      const tvm::ffi::TensorView previous_hidden,
      const tvm::ffi::TensorView enorm_weight,
      const tvm::ffi::TensorView hnorm_weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto H2 = SymbolicSize{"hidden_size_times_2"};
    auto SE = SymbolicSize{"embeds_stride"};
    auto SP = SymbolicSize{"previous_hidden_stride"};
    auto SO = SymbolicSize{"output_stride"};
    auto device_ = SymbolicDevice{};
    H.set_value(kHidden);
    H2.set_value(kHidden * 2);
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, H}).with_strides({SE, 1}).with_dtype<T>().with_device(device_).verify(embeds);
    TensorMatcher({N, H}).with_strides({SP, 1}).with_dtype<T>().with_device(device_).verify(previous_hidden);
    TensorMatcher({H}).with_dtype<T>().with_device(device_).verify(enorm_weight);
    TensorMatcher({H}).with_dtype<T>().with_device(device_).verify(hnorm_weight);
    TensorMatcher({N, H2}).with_strides({SO, 1}).with_dtype<T>().with_device(device_).verify(output);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedEHNormParams{
        .embeds = embeds.data_ptr(),
        .previous_hidden = previous_hidden.data_ptr(),
        .enorm_weight = enorm_weight.data_ptr(),
        .hnorm_weight = hnorm_weight.data_ptr(),
        .output = output.data_ptr(),
        .embeds_stride = SE.unwrap(),
        .previous_hidden_stride = SP.unwrap(),
        .output_stride = SO.unwrap(),
        .eps = eps,
    };

    const auto num_blocks = num_tokens;
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
