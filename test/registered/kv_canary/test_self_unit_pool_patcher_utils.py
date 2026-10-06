from __future__ import annotations

import unittest

from sglang.srt.kv_canary.pool_patcher.utils import wrap_method
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="extra-a-test-1-gpu-small-amd")


class _FakeObj:
    def greet(self, name: str) -> str:
        return f"hello {name}"


class TestPoolPatcherUtils(CustomTestCase):
    def test_wrap_method_delegates_to_wrapper(self) -> None:
        """Verify wrapped methods delegate through the wrapper."""
        obj = _FakeObj()

        def _with_validation(original, name: str) -> str:
            return original(name) + "!"

        wrap_method(obj, "greet", wrapper=_with_validation)
        self.assertEqual(obj.greet("world"), "hello world!")

    def test_wrap_method_missing_method_raises_attribute_error(self) -> None:
        """Verify wrapping a missing method raises AttributeError."""
        obj = _FakeObj()
        with self.assertRaisesRegex(AttributeError, "missing required method"):
            wrap_method(
                obj, "nonexistent", wrapper=lambda orig, *a, **kw: orig(*a, **kw)
            )

    def test_wrap_method_double_wrap_raises_runtime_error(self) -> None:
        """Verify wrapping the same method twice raises RuntimeError."""
        obj = _FakeObj()
        wrap_method(obj, "greet", wrapper=lambda orig, *a, **kw: orig(*a, **kw))
        with self.assertRaisesRegex(RuntimeError, "already wrapped by kv-canary"):
            wrap_method(obj, "greet", wrapper=lambda orig, *a, **kw: orig(*a, **kw))

    def test_wrap_method_preserves_functools_wraps_metadata(self) -> None:
        """Verify wrapping preserves method metadata."""
        obj = _FakeObj()
        original_name = obj.greet.__name__
        wrap_method(obj, "greet", wrapper=lambda orig, *a, **kw: orig(*a, **kw))
        self.assertEqual(obj.greet.__name__, original_name)


if __name__ == "__main__":
    unittest.main()
