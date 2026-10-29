"""``TensorMatcher`` keeps the device / dtype options already set on a symbol.

Passing a restricted symbol to ``.with_device(sym)`` / ``.with_dtype(sym)``
without a type list used to wipe its options, so a CPU tensor passed a launcher
that declared ``set_options<kDLGPU>()``.
"""

import sys

import pytest
import torch

from sglang.kernels.jit.utils.compile import load_jit
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_SOURCE = r"""
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace tensor_matcher_test {

using sglang::fp32_t;
using sglang::kDLGPU;

// options set on the symbol, then passed without a type list
void device_from_symbol(tvm::ffi::TensorView x) {
  using namespace sglang::host;
  auto device = SymbolicDevice{};
  device.set_options<kDLGPU>();
  TensorMatcher({-1}).with_device(device).verify(x);
}

void dtype_from_symbol(tvm::ffi::TensorView x) {
  using namespace sglang::host;
  auto dtype = SymbolicDType{};
  dtype.set_options<fp32_t>();
  TensorMatcher({-1}).with_dtype(dtype).with_device<kDLGPU, kDLCPU>().verify(x);
}

// a runtime value bound before the options are set
void device_from_value(tvm::ffi::TensorView x) {
  using namespace sglang::host;
  TensorMatcher({-1}).with_device<kDLGPU>(x.device()).verify(x);
}

void dtype_from_value(tvm::ffi::TensorView x) {
  using namespace sglang::host;
  TensorMatcher({-1}).with_dtype<fp32_t>(x.dtype()).with_device<kDLGPU, kDLCPU>().verify(x);
}

}  // namespace tensor_matcher_test
"""

_FUNCS = ("device_from_symbol", "dtype_from_symbol", "device_from_value", "dtype_from_value")


@pytest.fixture(scope="module")
def module(tmp_path_factory):
    source = tmp_path_factory.mktemp("tensor_matcher") / "tensor_matcher_test.cuh"
    source.write_text(_SOURCE)
    return load_jit(
        "test_tensor_matcher",
        cuda_files=[str(source)],
        cuda_wrappers=[(name, f"tensor_matcher_test::{name}") for name in _FUNCS],
    )


@pytest.mark.parametrize("name", ["device_from_symbol", "device_from_value"])
def test_device_options_reject_cpu(module, name):
    getattr(module, name)(torch.empty(4, device="cuda"))
    with pytest.raises(RuntimeError, match="not in the allowed options"):
        getattr(module, name)(torch.empty(4))


@pytest.mark.parametrize("name", ["dtype_from_symbol", "dtype_from_value"])
def test_dtype_options_reject_other_dtype(module, name):
    getattr(module, name)(torch.empty(4, dtype=torch.float32))
    with pytest.raises(RuntimeError, match="not in the allowed options"):
        getattr(module, name)(torch.empty(4, dtype=torch.bfloat16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
