"""Conftest for function_call unit tests.

Handles two CPU/macOS import issues so serving_chat can be imported:

1. **triton_heuristics TypeError**: torch._inductor.runtime.triton_heuristics
   uses `CompiledKernel | StaticallyLaunchedCudaKernel` (module | type),
   which raises TypeError on macOS.  Triggered by @torch.compile at module
   level in deep_gemm.py.  Patched by making torch.compile a passthrough.
   MUST be done before ANY sglang import (test_utils.py itself triggers it).

2. **sgl_kernel missing**: Stubbed via maybe_stub_sgl_kernel() (no GPU).

Both patches are no-ops on Linux/CUDA where everything works correctly.
"""


def _patch_torch_compile_if_needed():
    """Patch torch.compile to passthrough if triton_heuristics is broken."""
    try:
        import torch._inductor.runtime.triton_heuristics  # noqa: F401

        return  # triton_heuristics imported fine, no patch needed
    except TypeError:
        pass  # known macOS issue: module | type union

    import torch

    if getattr(torch, "_sglang_compile_patched", False):
        return

    def _passthrough_compile(fn=None, *args, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f

    torch.compile = _passthrough_compile
    torch._sglang_compile_patched = True


# 1. Patch torch.compile BEFORE any sglang import
_patch_torch_compile_if_needed()

# 2. Now safe to import sglang — stub sgl_kernel too
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()
