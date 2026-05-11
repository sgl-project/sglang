"""Unit tests for srt/utils/multi_stream_utils.py -- no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import threading
import unittest
from unittest.mock import MagicMock, patch

import sglang.srt.utils.multi_stream_utils as multi_stream_utils
from sglang.srt.utils.multi_stream_utils import (
    do_multi_stream,
    maybe_execute_in_parallel,
    set_do_multi_stream,
    with_multi_stream,
)
from sglang.test.test_utils import CustomTestCase


class TestMultiStreamFlag(CustomTestCase):
    def setUp(self):
        set_do_multi_stream(False)

    def tearDown(self):
        set_do_multi_stream(False)

    def test_default_state_is_disabled(self):
        self.assertFalse(do_multi_stream())

    def test_set_do_multi_stream_updates_current_thread(self):
        set_do_multi_stream(True)
        self.assertTrue(do_multi_stream())

        set_do_multi_stream(False)
        self.assertFalse(do_multi_stream())

    def test_with_multi_stream_restores_previous_state(self):
        set_do_multi_stream(True)

        with with_multi_stream(False):
            self.assertFalse(do_multi_stream())

        self.assertTrue(do_multi_stream())

    def test_with_multi_stream_restores_after_exception(self):
        set_do_multi_stream(True)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with with_multi_stream(False):
                raise RuntimeError("boom")

        self.assertTrue(do_multi_stream())

    def test_state_is_thread_local(self):
        set_do_multi_stream(True)
        worker_states = []

        def worker():
            worker_states.append(do_multi_stream())
            set_do_multi_stream(True)
            worker_states.append(do_multi_stream())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        self.assertEqual(worker_states, [False, True])
        self.assertTrue(do_multi_stream())


class TestMaybeExecuteInParallel(CustomTestCase):
    def setUp(self):
        set_do_multi_stream(False)

    def tearDown(self):
        set_do_multi_stream(False)

    def test_fallback_runs_functions_in_order_without_aux_stream(self):
        calls = []

        def fn0():
            calls.append("fn0")
            return "left"

        def fn1():
            calls.append("fn1")
            return "right"

        with with_multi_stream(True):
            result = maybe_execute_in_parallel(fn0, fn1, events=[])

        self.assertEqual(result, ("left", "right"))
        self.assertEqual(calls, ["fn0", "fn1"])

    def test_fallback_ignores_aux_stream_when_multi_stream_disabled(self):
        calls = []
        event0 = MagicMock()
        event1 = MagicMock()

        def fn0():
            calls.append("fn0")
            return "left"

        def fn1():
            calls.append("fn1")
            return "right"

        with patch.object(multi_stream_utils.torch.cuda, "stream") as stream:
            result = maybe_execute_in_parallel(
                fn0, fn1, events=[event0, event1], aux_stream=object()
            )

        self.assertEqual(result, ("left", "right"))
        self.assertEqual(calls, ["fn0", "fn1"])
        stream.assert_not_called()
        event0.record.assert_not_called()
        event0.wait.assert_not_called()
        event1.record.assert_not_called()
        event1.wait.assert_not_called()

    def test_fallback_propagates_fn0_exception_without_calling_fn1(self):
        calls = []

        def fn0():
            calls.append("fn0")
            raise RuntimeError("fn0 failed")

        def fn1():
            calls.append("fn1")
            return "right"

        with self.assertRaisesRegex(RuntimeError, "fn0 failed"):
            maybe_execute_in_parallel(fn0, fn1, events=[])

        self.assertEqual(calls, ["fn0"])

    def test_fallback_propagates_fn1_exception_after_calling_fn0(self):
        calls = []

        def fn0():
            calls.append("fn0")
            return "left"

        def fn1():
            calls.append("fn1")
            raise RuntimeError("fn1 failed")

        with self.assertRaisesRegex(RuntimeError, "fn1 failed"):
            maybe_execute_in_parallel(fn0, fn1, events=[])

        self.assertEqual(calls, ["fn0", "fn1"])

    def test_parallel_path_records_and_waits_in_expected_order(self):
        calls = []
        event0 = MagicMock()
        event1 = MagicMock()
        event0.record.side_effect = lambda: calls.append("event0.record")
        event0.wait.side_effect = lambda: calls.append("event0.wait")
        event1.record.side_effect = lambda: calls.append("event1.record")
        event1.wait.side_effect = lambda: calls.append("event1.wait")

        def fn0():
            calls.append("fn0")
            return "left"

        def fn1():
            calls.append("fn1")
            return "right"

        stream_context = MagicMock()
        stream_context.__enter__.side_effect = lambda: calls.append("stream.enter")
        stream_context.__exit__.side_effect = lambda *args: calls.append("stream.exit")
        aux_stream = object()

        with patch.object(
            multi_stream_utils.torch.cuda, "stream", return_value=stream_context
        ) as stream:
            with with_multi_stream(True):
                result = maybe_execute_in_parallel(
                    fn0, fn1, events=[event0, event1], aux_stream=aux_stream
                )

        self.assertEqual(result, ("left", "right"))
        stream.assert_called_once_with(aux_stream)
        self.assertEqual(
            calls,
            [
                "event0.record",
                "fn0",
                "stream.enter",
                "event0.wait",
                "fn1",
                "event1.record",
                "stream.exit",
                "event1.wait",
            ],
        )


if __name__ == "__main__":
    unittest.main()
