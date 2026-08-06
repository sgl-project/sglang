"""Regression test for https://github.com/sgl-project/sglang/issues/28999.

``pynccl_allocator`` imports private ``torch.cuda.memory`` memory-pool APIs
(e.g. ``_cuda_beginAllocateCurrentThreadToPool``) that only exist on
torch>=2.8 CUDA builds. On other builds/backends -- notably Ascend NPU, which
pins torch 2.7 via ``torch_npu`` -- those names are absent. The module is
imported eagerly across the codebase (model_runner, linear, dp_attention,
fp8, ... and even the NPU graph runners), so an unconditional top-level import
crashed SGLang at startup before any device dispatch.

The import is now guarded; this test simulates the missing symbol and asserts
the module still imports (with the guarded names falling back to ``None``).
It mocks the symbol away, so it needs no GPU/NPU and runs on CPU CI.
"""

import importlib
import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # pynccl_allocator transitively pulls GPU-only packages

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODULE = "sglang.srt.distributed.device_communicators.pynccl_allocator"
GUARDED_SYMBOLS = (
    "CUDAPluggableAllocator",
    "_cuda_beginAllocateCurrentThreadToPool",
    "_cuda_endAllocateToPool",
    "_cuda_releasePool",
)
# The torch>=2.8-only name whose absence triggers the ImportError on torch 2.7.
MISSING_ON_OLD_TORCH = "_cuda_beginAllocateCurrentThreadToPool"


class TestPyncclAllocatorImportGuard(CustomTestCase):
    def test_import_succeeds_when_cuda_pool_symbol_missing(self):
        mem = torch.cuda.memory
        had_symbol = hasattr(mem, MISSING_ON_OLD_TORCH)
        saved_symbol = getattr(mem, MISSING_ON_OLD_TORCH, None)
        saved_module = sys.modules.pop(MODULE, None)
        try:
            # Simulate a torch build / NPU backend that does not expose the
            # torch>=2.8 CUDA memory-pool API.
            if had_symbol:
                delattr(mem, MISSING_ON_OLD_TORCH)

            module = importlib.import_module(MODULE)

            # Import must not raise, and the guarded name falls back to None.
            self.assertIsNone(getattr(module, MISSING_ON_OLD_TORCH))
            for name in GUARDED_SYMBOLS:
                self.assertTrue(hasattr(module, name))
        finally:
            if had_symbol:
                setattr(mem, MISSING_ON_OLD_TORCH, saved_symbol)
            # Restore a clean, fully-imported module for any later tests.
            sys.modules.pop(MODULE, None)
            if saved_module is not None:
                sys.modules[MODULE] = saved_module
            else:
                importlib.import_module(MODULE)


if __name__ == "__main__":
    unittest.main()
