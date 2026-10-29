// Native geometry and arithmetic for <=4096 compressed keys. The long kernel
// writes only the complementary rows; PDL waits preserve their output dependency.
#include "../elementwise/fast_topk.cuh"

namespace sglang {
template <bool kPDL, bool kZeroStart>
__global__ __launch_bounds__(fast_topk_detail::kThreadsPerBlock)
void q38_mk_short_kernel(const fast_topk_detail::FastTopKParams __grid_constant__ params) {
  device::PDLWaitPrimary<kPDL>();
  const int row = blockIdx.x;
  const int length = params.lengths[row];
  if (length <= 4096) {
    const int start = kZeroStart ? 0 : params.row_starts[row];
    int32_t* indices = params.indices + row * 512;
    if (length <= 512) fast_topk_detail::naive_topk<512>(indices, length);
    else fast_topk_detail::radix_select_topk<512>(
        params.input + row * params.input_stride, indices, start, length);
  }
  device::PDLTriggerSecondary<kPDL>();
}

template <bool kPDL, bool kZeroStart>
struct Q38MkShortTopK {
  static void run(tvm::ffi::TensorView score,
                  tvm::ffi::Optional<tvm::ffi::TensorView> starts,
                  tvm::ffi::TensorView output, tvm::ffi::TensorView lengths) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({rows, -1}).with_strides({-1, 1}).with_dtype<float>().with_device(device).verify(score);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({rows, 512}).with_dtype<int32_t>().with_device(device).verify(output);
    CHECK_HOST(starts.has_value() != kZeroStart);
    if (starts.has_value())
      TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(starts.value());
    const auto params = fast_topk_detail::FastTopKParams{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = kZeroStart ? nullptr : static_cast<const int32_t*>(starts.value().data_ptr()),
        .indices = static_cast<int32_t*>(output.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = score.stride(0),
    };
    LaunchKernel(uint32_t(rows.unwrap()), fast_topk_detail::kThreadsPerBlock,
                 device.unwrap(), fast_topk_detail::kSmemBytes)
        .enable_pdl(kPDL)(q38_mk_short_kernel<kPDL, kZeroStart>, params);
  }
};
}  // namespace sglang
