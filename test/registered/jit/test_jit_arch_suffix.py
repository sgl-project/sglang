"""
Unit tests for JIT CUDA arch-suffix defaulting (`_init_jit_cuda_arch_once`).

Covers sgl-project/sglang#19963: JIT compiles only for the physically
present device, so Hopper+ (CC >= 9.0) should default to an arch-specific
target, mirroring FlashInfer's own `_normalize_cuda_arch`
(`flashinfer/compilation_context.py`) -- "a" for 9.x/10.x+, except SM 12.x
carves out "f" for .0 (SM120) vs "a" for .1+ (CUDA >= 12.9 only), since
SM120 and SM121 need separate cubins to avoid `cudaErrorIllegalInstruction`.
HIP/MUSA capability numbers are not CUDA SM versions and must stay
unsuffixed.

Pure CPU logic (monkeypatched `torch.cuda`), no GPU required.
"""

import sys

import pytest

from sglang.jit_kernel import utils
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=3, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_UNSET = object()


@pytest.fixture(autouse=True)
def _assert_arch_not_leaked():
    # Teardown guard for the save/restore in `_init_arch` below: every test in
    # this file must leave `utils._CUDA_ARCH` exactly as it found it, since
    # it's process-wide state other @cache_once functions latch onto.
    before = getattr(utils, "_CUDA_ARCH", _UNSET)
    yield
    after = getattr(utils, "_CUDA_ARCH", _UNSET)
    assert after is before, "utils._CUDA_ARCH leaked past test teardown"


def _init_arch(monkeypatch, capability, *, hip=False, musa=False, cuda_version=(13, 0)):
    monkeypatch.setattr(
        utils.torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(utils, "is_hip_runtime", lambda: hip)
    monkeypatch.setattr(utils, "is_musa_runtime", lambda: musa)
    # Only the major==12 branch reads this; default (13, 0) keeps every
    # other capability's expectation independent of the toolkit version.
    monkeypatch.setattr(utils, "get_cuda_version", lambda: cuda_version)
    # `_init_jit_cuda_arch_once` writes the module-level `utils._CUDA_ARCH`
    # global directly, so bypassing @cache_once via __wrapped__() leaves it
    # mutated for the rest of the process (see `_assert_arch_not_leaked`
    # above for why that matters). Save/restore explicitly, including the
    # "never initialized" case.
    previous = getattr(utils, "_CUDA_ARCH", _UNSET)
    try:
        utils._init_jit_cuda_arch_once.__wrapped__()
        return utils._CUDA_ARCH.target_name
    finally:
        if previous is _UNSET:
            if hasattr(utils, "_CUDA_ARCH"):
                del utils._CUDA_ARCH
        else:
            utils._CUDA_ARCH = previous


@pytest.mark.parametrize(
    "capability,expected",
    [
        ((9, 0), "9.0a"),  # Hopper
        ((10, 0), "10.0a"),  # Blackwell datacenter
        ((10, 3), "10.3a"),  # Blackwell datacenter family variant (SM103)
        ((8, 0), "8.0"),  # Ampere: below threshold, unsuffixed
        ((7, 5), "7.5"),  # Turing: below threshold, unsuffixed
    ],
)
def test_default_cuda_arch_suffix(monkeypatch, capability, expected):
    # Default cuda_version=(13, 0): irrelevant here except as a control for
    # the SM 12.x cases below, which vary it explicitly.
    assert _init_arch(monkeypatch, capability) == expected


@pytest.mark.parametrize(
    "capability,cuda_version,expected",
    [
        ((12, 0), (13, 0), "12.0f"),  # SM120: carved out of "a" to avoid
        # cudaErrorIllegalInstruction on SM121 (DGX Spark) running SM120 cubin
        ((12, 1), (13, 0), "12.1a"),  # SM121: not carved out, "a" like 10.x+
        ((12, 0), (12, 4), "12.0"),  # pre-12.9 toolkit: no family/"f" cubin
        # support -- fall back to plain instead of FlashInfer's RuntimeError
    ],
)
def test_sm12x_cuda_arch_suffix(monkeypatch, capability, cuda_version, expected):
    assert _init_arch(monkeypatch, capability, cuda_version=cuda_version) == expected


def test_hip_runtime_stays_unsuffixed(monkeypatch):
    # AMD MI300-class capability numbers alias into major >= 9 too, but HIP
    # has no CUDA "a"/"f" target concept.
    assert _init_arch(monkeypatch, (9, 4), hip=True) == "9.4"


def test_musa_runtime_stays_unsuffixed(monkeypatch):
    assert _init_arch(monkeypatch, (9, 0), musa=True) == "9.0"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
