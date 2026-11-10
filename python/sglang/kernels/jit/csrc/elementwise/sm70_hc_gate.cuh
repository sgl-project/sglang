// SPDX-License-Identifier: Apache-2.0
// Compute HC injection gates alongside the mix projection, then reuse them
// when the attention/MLP output is ready.
#include "hc_combine.cuh"
#include "sm70_hc_batch.cuh"

namespace sglang::sm70_hc_gate {
template <int M>
void down(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView inject,
    tvm::ffi::TensorView o,
    tvm::ffi::TensorView partials) {
  static_assert(M == 2 || M == 4);
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  TensorMatcher({M, 10240}).with_dtype<half>().with_device(dev).verify(x);
  TensorMatcher({320, 10240}).with_dtype<half>().with_device(dev).verify(w);
  TensorMatcher({4, 10240}).with_dtype<half>().with_device(dev).verify(inject);
  TensorMatcher({M, 320}).with_dtype<half>().with_device(dev).verify(o);
  TensorMatcher({M, 8, 4}).with_dtype<float>().with_device(dev).verify(partials);
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(inject.data_ptr()) % 16 == 0,
      "Projection operands must be 16-byte aligned");
  LaunchKernel(324, 256, dev.unwrap())(
      sm70_hc_batch::down<M, 256, true>,
      static_cast<const half*>(x.data_ptr()),
      static_cast<const half*>(w.data_ptr()),
      static_cast<half*>(o.data_ptr()),
      static_cast<const half*>(inject.data_ptr()),
      static_cast<float*>(partials.data_ptr()));
}

void apply(
    tvm::ffi::TensorView y, tvm::ffi::TensorView residual, tvm::ffi::TensorView partials, tvm::ffi::TensorView output) {
  using namespace host;
  auto dev = SymbolicDevice{};
  dev.set_options<kDLCUDA>();
  auto rows = SymbolicSize{};
  TensorMatcher({rows, 2560}).with_dtype<half>().with_device(dev).verify(y);
  TensorMatcher({rows, 10240}).with_dtype<half>().with_device(dev).verify(residual).verify(output);
  TensorMatcher({rows, 8, 4}).with_dtype<float>().with_device(dev).verify(partials);
  RuntimeCheck(rows.unwrap() == 2 || rows.unwrap() == 4, "Expected two or four rows");
  RuntimeCheck(
      reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
      "Combine operands must be 16-byte aligned");
  const HcCombineSplitParams params{
      y.data_ptr(), residual.data_ptr(), nullptr, nullptr, output.data_ptr(), static_cast<float*>(partials.data_ptr())};
  LaunchKernel(dim3(rows.unwrap(), 8), 160, dev.unwrap())(hc_combine_apply_kernel<4, 2560, false, half>, params);
}
}  // namespace sglang::sm70_hc_gate
