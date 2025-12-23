#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

[[maybe_unused]]
void assert_same_shape(tvm::ffi::TensorView a, tvm::ffi::TensorView b) {
  using namespace host;
  auto N = SymbolicSize{"N"};
  auto D = SymbolicSize{"D"};
  TensorMatcher({N, D})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>()
      .verify(a)
      .verify(b);
  RuntimeCheck(N.unwrap() > 0 && D.unwrap() > 0);
}

}  // namespace
