"""Shared helper to import norm.py (fused RMSNorm+RoPE kernels) in isolation.

Avoids the heavy multimodal_gen.__init__ dependency chain by creating
minimal module stubs and a mock current_platform.
"""

import importlib
import importlib.util
import os
import sys
import types

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

_NORM_PY_PATH = os.path.join(
    _REPO_ROOT, "python", "sglang", "jit_kernel", "diffusion", "triton", "norm.py"
)


def import_fused_norm_module():
    """Return the ``sglang.jit_kernel.diffusion.triton.norm`` module."""
    cached = sys.modules.get("sglang.jit_kernel.diffusion.triton.norm")
    if cached is not None:
        return cached

    for pkg in [
        "sglang.multimodal_gen",
        "sglang.multimodal_gen.runtime",
        "sglang.multimodal_gen.runtime.platforms",
        "sglang.jit_kernel",
        "sglang.jit_kernel.diffusion",
        "sglang.jit_kernel.diffusion.triton",
    ]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    class _MockPlatform:
        def is_npu(self):
            return False

        def is_cuda(self):
            return True

    platforms_mod = sys.modules["sglang.multimodal_gen.runtime.platforms"]
    platforms_mod.current_platform = _MockPlatform()

    spec = importlib.util.spec_from_file_location(
        "sglang.jit_kernel.diffusion.triton.norm",
        _NORM_PY_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sglang.jit_kernel.diffusion.triton.norm"] = mod
    spec.loader.exec_module(mod)
    return mod
