"""On CC 10.7 only MODULE_USES_CUTLASS builds the sm_100f family target (until FlashInfer's CUTLASS has Sm107)."""

from types import SimpleNamespace

import pytest

from sglang.kernels.jit.utils import arch
from sglang.kernels.jit.utils.arch import CompileArch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("cutlass_sm107", [False, True])
def test_cc107_family_target_only_for_cutlass_modules(monkeypatch, cutlass_sm107):
    cc = pytest.importorskip("flashinfer.compilation_context")
    monkeypatch.setattr(cc, "cutlass_supports_sm107", lambda: cutlass_sm107)
    monkeypatch.setattr(
        arch,
        "current_arch",
        lambda: SimpleNamespace(cuda=SimpleNamespace(capability=(10, 7))),
    )
    for flag in ("is_hip_runtime", "is_musa_runtime", "is_sail_runtime"):
        monkeypatch.setattr(arch, flag, lambda: False)
    compile_arch = arch.get_compile_arch.__wrapped__  # bypass cache_once
    sm107a, sm100f = CompileArch(10, 7, "a"), CompileArch(10, 0, "f")
    assert compile_arch("dpsk_v4_topk_v2") == sm107a
    assert compile_arch() == sm107a
    for name in arch.MODULE_USES_CUTLASS:
        assert compile_arch(name) == (sm107a if cutlass_sm107 else sm100f)
