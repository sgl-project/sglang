#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "cuda_kernels/v3_fp32/topk_select.cuh"

namespace sglang {

template <int kTopK>
struct DeepSelectTopKFp32Kernel {
  using Config = TopkSelectConfig<float, int32_t, false, false, false, kTopK, 512, 1, 8192, 4096, 3>;

  static void
  run(const tvm::ffi::TensorView scores, const tvm::ffi::TensorView ends, const tvm::ffi::TensorView indices) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto N = SymbolicSize{"width"};
    auto S = SymbolicSize{"input_stride"};
    auto O = SymbolicSize{"output_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, N}).with_strides({S, 1}).with_dtype<fp32_t>().with_device(device).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(ends);
    TensorMatcher({B, kTopK}).with_strides({O, 1}).with_dtype<int32_t>().with_device(device).verify(indices);

    const auto stride = static_cast<uint64_t>(S.unwrap());
    const auto out_stride = static_cast<uint64_t>(O.unwrap());
    const auto width = static_cast<uint64_t>(N.unwrap());
    RuntimeCheck(
        stride * sizeof(float) % INPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
        "DeepSelect requires the score row stride to be a multiple of ",
        INPUT_STRIDE_ALIGNMENT_REQUIREMENT,
        " bytes, got ",
        stride);
    RuntimeCheck(
        out_stride * sizeof(int32_t) % OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
        "DeepSelect requires the index row stride to be a multiple of ",
        OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT,
        " bytes, got ",
        out_stride);
    RuntimeCheck(width < MAX_VOCAB_SIZE, "DeepSelect width must be below ", MAX_VOCAB_SIZE, ", got ", width);

    const auto batch = static_cast<uint32_t>(B.unwrap());
    if (batch == 0) {
      return;
    }
    const auto dev = device.unwrap();
    int smem_optin = 0;
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev.device_id));

    TopkSelectArgs args{};
    args.batch_size = batch;
    args.vocab_size = static_cast<uint32_t>(width);
    args.topk = kTopK;
    args.input = scores.data_ptr();
    args.output_index = indices.data_ptr();
    args.end_ptr = static_cast<int*>(ends.data_ptr());
    args.stride_input_batch = stride;
    args.stride_output_index_batch = out_stride;
    args.idx_oob_fill_value = -1;
    args.value_oob_fill_value = -INFINITY;
    args.abort_when_nan_found = false;
    args.shared_memory_size_per_sm = static_cast<uint64_t>(smem_optin);
    args.stream = LaunchKernel::resolve_device(dev);
    topk_select_fp32::run_topk_select_kernel<Config>(args);
  }
};

}  // namespace sglang
