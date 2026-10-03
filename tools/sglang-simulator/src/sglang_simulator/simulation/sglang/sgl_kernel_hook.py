import os
import sys
import types

# CPU ops that sglang model modules resolve at module import time when
# SGLANG_USE_CPU_ENGINE=1 (the "if _is_cpu:" blocks in
# python/sglang/srt/models/qwen3_5.py:238-253 and muse_glimmer.py:69-70). The
# CUDA sgl_kernel wheel does not register them, so the import fails and
# ModelRegistry silently drops the module. The simulator never executes a real
# model forward, so import-time name resolution is all it needs. Every name
# here must exist in the m.def block of
# python/sglang/kernels/aot/csrc/cpu/torch_extension_cpu.cpp, but this list is
# deliberately the import-time-only subset: never add names that sglang probes
# via hasattr (convert_weight_packed, fused_qk_norm_rope_cpu,
# fused_qk_norm_cpu), because a stand-in would flip those feature probes and
# change which code paths sglang selects. gdn_backend.py's module-level
# fused_gdn_gating_cpu binding is also out of scope on purpose: that module is
# imported lazily at attention-backend selection, never by the registry scan,
# and its sibling "from sgl_kernel.mamba import" fails on the CUDA wheel
# regardless, so an op stand-in alone would not make it importable.
_CPU_OP_STANDIN_NAMES = (
    "fused_sigmoid_mul_cpu",
    "fused_qk_gemma_rmsnorm_cpu",
    "fused_qk_gemma_rmsnorm_with_gate_cpu",
    "fused_qkvzba_split_reshape_cat_contiguous_cpu",
)


def install_load_utils_stub() -> None:
    """Install the kernel loader stub before importing the sgl_kernel package."""
    module_name = "sgl_kernel.load_utils"
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__package__ = "sgl_kernel"
        sys.modules[module_name] = module

    module._load_architecture_specific_ops = lambda *args, **kwargs: None
    module._preload_cuda_library = lambda *args, **kwargs: None


def _make_cpu_op_standin(op_name):
    def _standin(*args, **kwargs):
        raise RuntimeError(
            f"sgl_kernel.{op_name} is an SGLang Simulator import-time stand-in; "
            "a real CPU kernel call was attempted, which the simulator must "
            "never execute. This process runs under the SGLang Simulator "
            "bootstrap (SGLANG_SIMULATOR_BOOTSTRAP=1, SGLANG_USE_CPU_ENGINE=1); "
            "launch plain sglang serving for real CPU inference."
        )

    # Tag the stand-in so shadowing reports are diagnosable from the callable.
    _standin._sglang_simulator_standin = True
    return _standin


def install_cpu_op_standins() -> None:
    """Register import-time stand-ins for the sgl_kernel CPU ops the CUDA wheel lacks.

    Model modules bind ``torch.ops.sgl_kernel.*_cpu`` at import time when
    SGLANG_USE_CPU_ENGINE=1. The simulator replaces forward passes with a
    latency predictor, so the names only need to resolve; the stand-ins raise
    if ever called so an accidental real forward is loud rather than silently
    wrong. The stand-ins must be installed before any sgl_kernel import:
    ``install_load_utils_stub()``, called at hook_bootstrap module level, is
    the backstop that keeps the real CPU shared library (and any later op
    registration) out of simulator processes.
    """
    if os.environ.get("SGLANG_USE_CPU_ENGINE") != "1":
        return

    import torch

    namespace = torch.ops.sgl_kernel
    for op_name in _CPU_OP_STANDIN_NAMES:
        if hasattr(namespace, op_name):
            # Never shadow an op that already resolves (real CPU wheel or
            # cached packet).
            continue
        setattr(namespace, op_name, _make_cpu_op_standin(op_name))
