"""CPU-only tests for multi-stream scheduling helpers."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.utils import multi_stream_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Event:
    def __init__(self, name: str, calls: list[object]):
        self.name = name
        self.calls = calls

    def record(self):
        self.calls.append(f"{self.name}.record")

    def wait(self):
        self.calls.append(f"{self.name}.wait")


class TestMultiStreamContext(CustomTestCase):
    def test_forward_context_helpers_delegate_to_runtime_context(self):
        scoped = object()
        forward = Mock(multi_stream=True)
        forward.scoped.return_value = scoped

        with patch.object(multi_stream_utils, "get_forward", return_value=forward):
            multi_stream_utils.set_do_multi_stream(False)

            self.assertTrue(multi_stream_utils.do_multi_stream())
            self.assertIs(multi_stream_utils.with_multi_stream(True), scoped)

        forward.set.assert_called_once_with("multi_stream", False)
        forward.scoped.assert_called_once_with(multi_stream=True)


class TestMaybeExecuteInParallel(CustomTestCase):
    def test_falls_back_to_ordered_execution_without_aux_stream(self):
        calls: list[object] = []
        events = [_Event("start", calls), _Event("done", calls)]
        forward = SimpleNamespace(multi_stream=True)

        def fn0():
            calls.append("fn0")
            return "left"

        def fn1():
            calls.append("fn1")
            return "right"

        with patch.object(multi_stream_utils, "get_forward", return_value=forward):
            result = multi_stream_utils.maybe_execute_in_parallel(
                fn0, fn1, events, aux_stream=None
            )

        self.assertEqual(result, ("left", "right"))
        self.assertEqual(calls, ["fn0", "fn1"])

    def test_uses_event_ordering_when_multi_stream_is_enabled(self):
        calls: list[object] = []
        events = [_Event("start", calls), _Event("done", calls)]
        aux_stream = object()
        forward = SimpleNamespace(multi_stream=True)

        @contextmanager
        def fake_stream(stream):
            calls.append(("stream.enter", stream))
            yield
            calls.append(("stream.exit", stream))

        def fn0():
            calls.append("fn0")
            return "left"

        def fn1():
            calls.append("fn1")
            return "right"

        with (
            patch.object(multi_stream_utils, "get_forward", return_value=forward),
            patch.object(
                multi_stream_utils.torch.cuda, "stream", side_effect=fake_stream
            ),
        ):
            result = multi_stream_utils.maybe_execute_in_parallel(
                fn0, fn1, events, aux_stream=aux_stream
            )

        self.assertEqual(result, ("left", "right"))
        self.assertEqual(
            calls,
            [
                "start.record",
                "fn0",
                ("stream.enter", aux_stream),
                "start.wait",
                "fn1",
                "done.record",
                ("stream.exit", aux_stream),
                "done.wait",
            ],
        )


if __name__ == "__main__":
    unittest.main()
