from __future__ import annotations

import pytest

from sglang.srt.kv_canary.pool_patch.wrap_method import wrap_method
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-large")


class _FakeObj:
    def greet(self, name: str) -> str:
        return f"hello {name}"


def test_wrap_method_delegates_to_wrapper() -> None:
    """Verify wrapped methods delegate through the wrapper."""
    obj = _FakeObj()

    def _with_validation(original, name: str) -> str:
        return original(name) + "!"

    wrap_method(obj, "greet", wrapper=_with_validation)
    assert obj.greet("world") == "hello world!"


def test_wrap_method_missing_method_raises_attribute_error() -> None:
    """Verify wrapping a missing method raises AttributeError."""
    obj = _FakeObj()
    with pytest.raises(AttributeError, match="missing required method"):
        wrap_method(obj, "nonexistent", wrapper=lambda orig, *a, **kw: orig(*a, **kw))


def test_wrap_method_double_wrap_raises_runtime_error() -> None:
    """Verify wrapping the same method twice raises RuntimeError."""
    obj = _FakeObj()
    wrap_method(obj, "greet", wrapper=lambda orig, *a, **kw: orig(*a, **kw))
    with pytest.raises(RuntimeError, match="already wrapped by kv-canary"):
        wrap_method(obj, "greet", wrapper=lambda orig, *a, **kw: orig(*a, **kw))


def test_wrap_method_preserves_functools_wraps_metadata() -> None:
    """Verify wrapping preserves method metadata."""
    obj = _FakeObj()
    original_name = obj.greet.__name__
    wrap_method(obj, "greet", wrapper=lambda orig, *a, **kw: orig(*a, **kw))
    assert obj.greet.__name__ == original_name
