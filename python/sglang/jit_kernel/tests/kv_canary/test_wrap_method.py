from __future__ import annotations

import pytest

from sglang.srt.kv_canary.pool_patch.helpers import _wrap_method


class _FakeObj:
    def greet(self, name: str) -> str:
        return f"hello {name}"


def test_wrap_method_delegates_to_wrapper() -> None:
    obj = _FakeObj()

    def _with_validation(original, name: str) -> str:
        return original(name) + "!"

    _wrap_method(obj, "greet", wrapper=_with_validation)
    assert obj.greet("world") == "hello world!"


def test_wrap_method_missing_method_raises_attribute_error() -> None:
    obj = _FakeObj()
    with pytest.raises(AttributeError, match="missing required method"):
        _wrap_method(obj, "nonexistent", wrapper=lambda orig, *a, **kw: orig(*a, **kw))


def test_wrap_method_double_wrap_raises_runtime_error() -> None:
    obj = _FakeObj()
    _wrap_method(obj, "greet", wrapper=lambda orig, *a, **kw: orig(*a, **kw))
    with pytest.raises(RuntimeError, match="already wrapped by kv-canary"):
        _wrap_method(obj, "greet", wrapper=lambda orig, *a, **kw: orig(*a, **kw))


def test_wrap_method_preserves_functools_wraps_metadata() -> None:
    obj = _FakeObj()
    original_name = obj.greet.__name__
    _wrap_method(obj, "greet", wrapper=lambda orig, *a, **kw: orig(*a, **kw))
    assert obj.greet.__name__ == original_name
