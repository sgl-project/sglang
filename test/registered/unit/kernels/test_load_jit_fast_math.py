"""``load_jit``'s fast-math switch, checked on CPU with no compiler or device.

``SGLANG_JIT_DISABLE_FAST_MATH`` drops nvcc's ``--use_fast_math`` and
``-use_fast_math`` from a CUDA build the way the HIP path always has. The build
is intercepted at ``BuildSpec``, before ninja or the cache see it, so the flags
asserted on are exactly the ones that would have reached the compiler.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import pytest

from sglang.kernels.jit.utils.arch import CompileArch
from sglang.kernels.jit.utils.compile import loader
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SWITCH = "SGLANG_JIT_DISABLE_FAST_MATH"
_FAKE_ARCH = CompileArch(9, 0, "a")


def test_strip_drops_both_spellings_and_keeps_the_rest_in_order():
    flags = [
        "-O3",
        "--use_fast_math",
        "--ftz=false",
        "-use_fast_math",
        "-Duse_fast_math",
        "--use_fast_math=1",
        "-lineinfo",
    ]
    assert loader._strip_fast_math_flags(flags) == [
        "-O3",
        "--ftz=false",
        "-Duse_fast_math",
        "--use_fast_math=1",
        "-lineinfo",
    ]


def test_strip_leaves_a_list_without_fast_math_alone():
    flags = ["-O3", "--ftz=false"]
    assert loader._strip_fast_math_flags(flags) == flags
    assert loader._strip_fast_math_flags([]) == []


class _Intercepted(Exception):
    """Raised by the BuildSpec stand-in: load_jit stops before any build step."""


def _cuda_cflags(monkeypatch, *, hip: bool, extra: List[str] | None) -> Tuple[str, ...]:
    """The ``cuda_cflags`` load_jit hands to the build, with no device queried."""
    monkeypatch.setattr(loader, "is_hip_runtime", lambda: hip)
    monkeypatch.setattr(loader, "is_sail_runtime", lambda: False)
    monkeypatch.setattr(loader, "resolve_compile_arch", lambda policy, name: _FAKE_ARCH)
    monkeypatch.setattr(
        loader,
        "get_default_target_flags",
        lambda arch: [arch.jit_flag, "-std=c++20"],
    )
    monkeypatch.delenv("SGLANG_JIT_LOG_RESOURCE_USAGE", raising=False)
    monkeypatch.delenv("SGLANG_JIT_ENABLE_LINEINFO", raising=False)
    monkeypatch.delenv("SGLANG_JIT_DEBUG_TRACE", raising=False)

    def record(**kwargs):
        raise _Intercepted(kwargs["cuda_cflags"])

    monkeypatch.setattr(loader, "BuildSpec", record)
    with pytest.raises(_Intercepted) as caught:
        loader.load_jit("fast_math_probe", extra_cuda_cflags=extra)
    return caught.value.args[0]


def test_cuda_keeps_fast_math_by_default(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    extra = ["--use_fast_math", "-use_fast_math", "-O3"]
    flags = _cuda_cflags(monkeypatch, hip=False, extra=extra)
    assert flags == (
        "-DSGL_CUDA_ARCH=900",
        "-std=c++20",
        "--use_fast_math",
        "-use_fast_math",
        "-O3",
    )
    assert extra == ["--use_fast_math", "-use_fast_math", "-O3"]


def test_cuda_drops_fast_math_when_the_switch_is_on(monkeypatch):
    extra = ["--use_fast_math", "-use_fast_math", "--ftz=false", "-O3"]
    with envs.SGLANG_JIT_DISABLE_FAST_MATH.override(True):
        flags = _cuda_cflags(monkeypatch, hip=False, extra=extra)
    assert flags == ("-DSGL_CUDA_ARCH=900", "-std=c++20", "--ftz=false", "-O3")
    assert extra == ["--use_fast_math", "-use_fast_math", "--ftz=false", "-O3"]


def test_cuda_handles_none_extra_cflags_when_the_switch_is_on(monkeypatch):
    with envs.SGLANG_JIT_DISABLE_FAST_MATH.override(True):
        flags = _cuda_cflags(monkeypatch, hip=False, extra=None)
    assert flags == ("-DSGL_CUDA_ARCH=900", "-std=c++20")


@pytest.mark.parametrize("switch", [False, True])
def test_hip_drops_fast_math_regardless_of_the_switch(monkeypatch, switch):
    with envs.SGLANG_JIT_DISABLE_FAST_MATH.override(switch):
        flags = _cuda_cflags(
            monkeypatch,
            hip=True,
            extra=["--use_fast_math", "-use_fast_math", "-O3"],
        )
    assert flags == ("-DSGL_CUDA_ARCH=900", "-std=c++20", "-O3")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
