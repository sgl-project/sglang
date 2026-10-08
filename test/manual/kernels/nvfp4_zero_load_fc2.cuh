// Test-only entry point for exact zero, signed-zero, and dense FC2 operands.
#include "../../../python/sglang/kernels/jit/csrc/gemm/sm120_nvfp4_moe_mtp_mma.cuh"
namespace sglang {
template <bool ZeroLoad> struct TestNvfp4ZeroFc2 {
  static void run(tvm::ffi::TensorView z, tvm::ffi::TensorView zs,
                  tvm::ffi::TensorView w, tvm::ffi::TensorView sf,
                  tvm::ffi::TensorView alpha, tvm::ffi::TensorView ids,
                  tvm::ffi::TensorView routing, tvm::ffi::TensorView partial) {
    using namespace host;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    auto rows = SymbolicSize{"rows"};
    TensorMatcher({rows, 22})
        .with_dtype<int32_t>()
        .with_device(dev)
        .verify(ids);
    unsigned M = rows.unwrap();
    LaunchKernel({16, 22, M}, 128, dev.unwrap())(
        nvfp4_mtp_mma::fc2<1024, 2688, 22, false, ZeroLoad>,
        static_cast<const uint32_t *>(z.data_ptr()),
        static_cast<const uint8_t *>(zs.data_ptr()),
        static_cast<const uint8_t *>(w.data_ptr()),
        static_cast<const uint8_t *>(sf.data_ptr()),
        static_cast<const float *>(alpha.data_ptr()),
        static_cast<const int *>(ids.data_ptr()),
        static_cast<const float *>(routing.data_ptr()),
        static_cast<bf16_t *>(partial.data_ptr()));
  }
};
} // namespace sglang
