"""``load_jit``'s fast-math switch, checked on CPU with no compiler or device.

``SGLANG_JIT_DISABLE_FAST_MATH`` drops nvcc's ``--use_fast_math`` from a CUDA
build the way the HIP path always has. The build is intercepted at ``BuildSpec``,
before ninja or the cache see it, so the flags asserted on are exactly the ones
that would have reached the compiler.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import pytest

from sglang.kernels.jit.utils.compile import loader
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SWITCH = "SGLANG_JIT_DISABLE_FAST_MATH"


def test_strip_drops_both_spellings_and_keeps_the_rest_in_order():
    flags = ["-O3", "--use_fast_math", "--ftz=false", "-use_fast_math", "-lineinfo"]
    assert loader._strip_fast_math_flags(flags) == ["-O3", "--ftz=false", "-lineinfo"]


def test_strip_leaves_a_list_without_fast_math_alone():
    flags = ["-O3", "--ftz=false"]
    assert loader._strip_fast_math_flags(flags) == flags
    assert loader._strip_fast_math_flags([]) == []


class _Intercepted(Exception):
    """Raised by the BuildSpec stand-in: load_jit stops before any build step."""


def _cuda_cflags(monkeypatch, *, hip: bool, extra: List[str]) -> Tuple[str, ...]:
    """The ``cuda_cflags`` load_jit hands to the build, with no device queried."""
    monkeypatch.setattr(loader, "is_hip_runtime", lambda: hip)
    monkeypatch.setattr(loader, "is_sail_runtime", lambda: False)
    monkeypatch.setattr(loader, "get_default_target_flags", lambda: ["-std=c++20"])
    monkeypatch.delenv("SGLANG_JIT_LOG_RESOURCE_USAGE", raising=False)

    def record(**kwargs):
        raise _Intercepted(kwargs["cuda_cflags"])

    monkeypatch.setattr(loader, "BuildSpec", record)
    with pytest.raises(_Intercepted) as caught:
        loader.load_jit("fast_math_probe", extra_cuda_cflags=extra)
    return caught.value.args[0]


def test_cuda_keeps_fast_math_by_default(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    flags = _cuda_cflags(monkeypatch, hip=False, extra=["--use_fast_math", "-O3"])
    assert flags == ("-std=c++20", "--use_fast_math", "-O3")


def test_cuda_drops_fast_math_when_the_switch_is_on(monkeypatch):
    with envs.SGLANG_JIT_DISABLE_FAST_MATH.override(True):
        flags = _cuda_cflags(monkeypatch, hip=False, extra=["--use_fast_math", "-O3"])
    assert flags == ("-std=c++20", "-O3")


@pytest.mark.parametrize("switch", [False, True])
def test_hip_drops_fast_math_regardless_of_the_switch(monkeypatch, switch):
    with envs.SGLANG_JIT_DISABLE_FAST_MATH.override(switch):
        flags = _cuda_cflags(monkeypatch, hip=True, extra=["--use_fast_math", "-O3"])
    assert flags == ("-std=c++20", "-O3")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
