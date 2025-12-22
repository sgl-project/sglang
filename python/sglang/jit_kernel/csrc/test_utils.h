#include <sgl_kernel/tensor.h>

#include <tvm/ffi/container/tensor.h>

namespace {

[[maybe_unused]]
void assert_same_shape(tvm::ffi::TensorView a, tvm::ffi::TensorView b) {
  using namespace host;
  auto N = SymbolicSize{"N"};
  auto D = SymbolicSize{"D"};
  TensorMatcher({N, D}).verify(a).verify(b);
}

}  // namespace
