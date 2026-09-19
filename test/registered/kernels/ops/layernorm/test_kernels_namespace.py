"""GPU-free import / registry / selector tests for ``sglang.kernels`` (RFC #29630)."""

import importlib
import subprocess
import sys

import pytest

import sglang.kernels as K
import sglang.kernels.fused_op as fo
import sglang.kernels.selector as sel
from sglang.kernels import KernelBackend, PlatformInfo
from sglang.kernels.spec import CapabilityRequirement as Cap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=24, suite="base-a-test-cpu")

_CPU = PlatformInfo(device_type="cpu")
_SM90 = PlatformInfo(device_type="cuda", cuda_arch_major=9, cuda_arch_minor=0)
_SM100 = PlatformInfo(device_type="cuda", cuda_arch_major=10, cuda_arch_minor=0)
_HIP = PlatformInfo(device_type="hip")


def test_single_backend_resolves_without_backend():
    assert (
        K.select_kernel("kvcache.reshape_and_cache_flash").backend
        is KernelBackend.TRITON
    )


def test_fp8_scaled_mm_requires_explicit_registry_backend(monkeypatch):
    monkeypatch.setattr(sel, "_platform", lambda: _SM90)
    with pytest.raises(ValueError, match="multiple backends"):
        K.select_kernel("gemm.fp8_scaled_mm")
    assert (
        K.select_kernel("gemm.fp8_scaled_mm", backend=KernelBackend.AOT).backend
        is KernelBackend.AOT
    )


def test_unknown_op_or_backend_raises():
    with pytest.raises(KeyError):
        K.select_kernel("does_not.exist")
    with pytest.raises(KeyError):
        K.select_kernel("gemm.fp8_scaled_mm", backend=KernelBackend.TRITON)


@pytest.mark.parametrize(
    "platform, eligible",
    [
        (_SM90, True),
        (_SM100, True),
        (PlatformInfo(device_type="cuda", cuda_arch_major=8, cuda_arch_minor=0), False),
        (_CPU, False),
        (_HIP, False),
    ],
)
def test_qwen_qkv_registry_accepts_hopper(platform, eligible):
    spec = K.select_kernel("diffusion.qwen_qkv_epilogue")
    assert spec.backend is KernelBackend.JIT
    assert K.capabilities_satisfied(spec.capabilities, platform) is eligible


def test_multi_backend_requires_explicit_backend(monkeypatch):
    # Device is a hard eligibility filter, not a ranking: >1 usable backend on
    # the current device means selection must name one.
    monkeypatch.setattr(sel, "_platform", lambda: _SM90)
    with pytest.raises(ValueError):
        K.select_kernel("layernorm.rmsnorm")
    spec = K.select_kernel("layernorm.rmsnorm", backend=KernelBackend.JIT)
    assert spec.backend is KernelBackend.JIT
    assert spec.target == "sglang.kernels.ops.layernorm:_RMSNORM.forward_jit"


@pytest.mark.parametrize("device, expect", [("cuda", "jit"), ("hip", "aot")])
def test_activation_default_backend(monkeypatch, device, expect):
    # silu_and_mul default matches production: jit on CUDA, aot (sgl_kernel) on HIP.
    from sglang.kernels.ops.activation import _SILU_AND_MUL

    monkeypatch.setattr(fo, "_platform", lambda: PlatformInfo(device_type=device))
    assert _SILU_AND_MUL.auto_selected_backend().value == expect


@pytest.mark.parametrize(
    "op_attr, device, expect",
    [
        ("_RMSNORM", "cuda", "aot"),
        ("_RMSNORM", "hip", "aiter"),
        ("_RMSNORM", "npu", "torch_npu"),
        ("_GEMMA_RMSNORM", "cuda", "aot"),
        ("_GEMMA_RMSNORM", "hip", "jit"),  # rocm-triton JIT pinned to HIP
        ("_GEMMA_RMSNORM", "npu", "torch_npu"),
    ],
)
def test_layernorm_default_backend(monkeypatch, op_attr, device, expect):
    # Same AOT provenance, different device coverage per op: rmsnorm's AOT is
    # CUDA-only, so HIP falls to aiter and NPU to torch_npu.
    ln = importlib.import_module("sglang.kernels.ops.layernorm")
    monkeypatch.setattr(fo, "_platform", lambda: PlatformInfo(device_type=device))
    assert getattr(ln, op_attr).auto_selected_backend().value == expect


@pytest.mark.parametrize(
    "req, plat, ok",
    [
        (Cap.CUDA, _CPU, False),
        (Cap.CUDA, _SM90, True),
        (Cap.CUDA, _HIP, False),
        (Cap.HIP, _HIP, True),
        (Cap.cuda(min_sm=(10, 0)), _SM90, False),
        (Cap.cuda(min_sm=(10, 0)), _SM100, True),
        (Cap.cuda(max_sm=(9, 0)), _SM100, False),
    ],
)
def test_capability_is_satisfied_by(req, plat, ok):
    assert req.is_satisfied_by(plat) is ok


def test_capabilities_or_semantics():
    both = {Cap.CUDA, Cap.HIP}
    assert K.capabilities_satisfied(both, _SM90)
    assert K.capabilities_satisfied(both, _HIP)
    assert not K.capabilities_satisfied(both, _CPU)
    assert K.capabilities_satisfied((), _CPU)  # empty = unrestricted
    assert K.capabilities_satisfied(Cap.CUDA, _SM90)  # single tolerated


@pytest.mark.parametrize(
    "relative_path",
    (
        "srt/utils/common.py",
        "multimodal_gen/runtime/utils/common.py",
    ),
)
def test_amx_backend_probe_is_lazy(relative_path):
    loader = (
        "package = importlib.util.find_spec('sglang'); "
        "path = pathlib.Path(next(iter(package.submodule_search_locations))) / "
        f"{relative_path!r}; "
        "spec = importlib.util.spec_from_file_location('_common_under_test', path); "
        "module = importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name] = module; "
        "spec.loader.exec_module(module)"
    )
    code = "; ".join(
        (
            "import builtins, importlib.util, pathlib, sys",
            "from unittest import mock",
            "real_import = builtins.__import__",
            "import_mock = mock.Mock(wraps=real_import)",
            "builtins.__import__ = import_mock",
            loader,
            "builtins.__import__ = real_import",
            "attempted = any(call.args and call.args[0] == 'sgl_kernel' "
            "for call in import_mock.call_args_list)",
            "print('DIRTY' if attempted else 'CLEAN')",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_import_stays_metadata_only():
    # Importing the namespace must not pull in the AOT backend (sgl_kernel) or
    # the JIT compilation infra (sglang.kernels.jit), which import torch / nvcc.
    code = (
        "import sys, sglang.kernels.ops; "
        "print('DIRTY' if 'sgl_kernel' in sys.modules or any("
        "m.startswith('sglang.kernels.jit') for m in sys.modules) else 'CLEAN')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "CLEAN" in r.stdout


@pytest.mark.parametrize("warm_cache", [False, True])
def test_kernel_resolution_torch_compile_fullgraph(monkeypatch, tmp_path, warm_cache):
    import torch

    from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8

    # Exercise the production wrapper and lazy import with a CPU implementation
    # of its in-place ABI. No PPU or sgl_kernel wheel is needed for this test.
    module_name = "_sglang_test_compile_quant_kernel"
    (tmp_path / f"{module_name}.py").write_text(
        "def quantize(x, output_q, output_s):\n"
        "    output_q.copy_(x * 2)\n"
        "    output_s.fill_(3)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    registry = K.KernelRegistry()
    registry.register(
        K.KernelSpec(
            op="quantization.sgl_per_token_quant_fp8",
            backend=KernelBackend.JIT,
            target=f"{module_name}:quantize",
        )
    )
    monkeypatch.setattr(sel, "registry", registry)
    sel.clear_cache()
    torch._dynamo.reset()
    try:
        if warm_cache:
            sel.get_kernel("quantization.sgl_per_token_quant_fp8", KernelBackend.JIT)
        else:
            assert module_name not in sys.modules

        def quantize(x):
            output_q = torch.empty_like(x)
            output_s = x.new_empty((x.shape[0], 1))
            sgl_per_token_quant_fp8(x, output_q, output_s)
            return output_q, output_s

        compiled = torch.compile(quantize, backend="aot_eager", fullgraph=True)
        for rows in (3, 3, 7):
            x = torch.randn(rows, 16)
            output_q, output_s = compiled(x)
            torch.testing.assert_close(output_q, x * 2)
            torch.testing.assert_close(output_s, x.new_full((rows, 1), 3))
        assert module_name in sys.modules
    finally:
        torch._dynamo.reset()
        sel.clear_cache()
        sys.modules.pop(module_name, None)


def test_kernel_cache_preserves_backend_and_clears(monkeypatch):
    registry = K.KernelRegistry()
    for backend, target in (
        (KernelBackend.AOT, "operator:neg"),
        (KernelBackend.TORCH, "operator:pos"),
    ):
        registry.register(K.KernelSpec(op="test.sign", backend=backend, target=target))
    monkeypatch.setattr(sel, "registry", registry)
    original_select = sel.select_kernel
    selections = []

    def select(op, backend=None):
        selections.append((op, backend))
        return original_select(op, backend)

    monkeypatch.setattr(sel, "select_kernel", select)
    sel.clear_cache()
    try:
        for _ in range(2):
            assert sel.get_kernel("test.sign", KernelBackend.AOT)(2) == -2
            assert sel.get_kernel("test.sign", KernelBackend.TORCH)(2) == 2
        assert len(selections) == 2
        sel.clear_cache()
        assert sel.get_kernel("test.sign", KernelBackend.AOT)(2) == -2
        assert len(selections) == 3
    finally:
        sel.clear_cache()


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("pause_before_resolve", [False, True])
def test_kernel_cache_clear_during_lookup(
    monkeypatch, warm_cache, pause_before_resolve
):
    import operator
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    registry = K.KernelRegistry()
    registry.register(
        K.KernelSpec(
            op="test.concurrent_neg",
            backend=KernelBackend.AOT,
            target="operator:neg",
        )
    )
    monkeypatch.setattr(sel, "registry", registry)
    original_select = sel.select_kernel
    selections = []

    def select(op, backend=None):
        selections.append((op, backend))
        return original_select(op, backend)

    monkeypatch.setattr(sel, "select_kernel", select)
    sel.clear_cache()
    try:
        if warm_cache:
            sel.get_kernel("test.concurrent_neg", KernelBackend.AOT)

        original_resolve = sel._resolve
        paused = Event()
        resume = Event()

        def pause():
            paused.set()
            assert resume.wait(timeout=10), "cache-clear thread did not resume lookup"

        def resolve(*args):
            if pause_before_resolve:
                pause()
            original_resolve(*args)
            if not pause_before_resolve:
                pause()

        monkeypatch.setattr(sel, "_resolve", resolve)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                sel.get_kernel, "test.concurrent_neg", KernelBackend.AOT
            )
            try:
                assert paused.wait(timeout=10), "lookup did not reach the pause point"
                sel.clear_cache()
            finally:
                resume.set()
            assert pending.result(timeout=10) is operator.neg

        monkeypatch.setattr(sel, "_resolve", original_resolve)
        # The in-flight lookup must not populate the new cache after a clear.
        # A fresh lookup resolves once, and subsequent lookups reuse it.
        for _ in range(2):
            assert (
                sel.get_kernel("test.concurrent_neg", KernelBackend.AOT) is operator.neg
            )
        assert len(selections) == 2
    finally:
        sel.clear_cache()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
