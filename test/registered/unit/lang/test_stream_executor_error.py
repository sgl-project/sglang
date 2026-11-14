"""Unit tests for StreamExecutor error handling and queue synchronization."""

import threading
import time
import unittest

import sglang as sgl
from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglConstantText, SglGen, SglSamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FailingBackend(BaseBackend):
    def __init__(self, fail_msg: str = "simulated backend error"):
        super().__init__()
        self.fail_msg = fail_msg

    def generate(self, s: StreamExecutor, sampling_params: SglSamplingParams):
        raise RuntimeError(self.fail_msg)

    def generate_stream(self, s: StreamExecutor, sampling_params: SglSamplingParams):
        raise RuntimeError(self.fail_msg)


class TestStreamExecutorError(CustomTestCase):
    def _run_with_timeout(self, fn, timeout: float = 2.0):
        result = {}
        exc = {}

        def _target():
            try:
                result["value"] = fn()
            except BaseException as e:
                exc["error"] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=timeout)
        self.assertFalse(t.is_alive(), "Operation deadlocked after worker error")
        if "error" in exc:
            raise exc["error"]
        return result.get("value")

    def test_submit_after_worker_error_does_not_deadlock(self):
        backend = _FailingBackend("boom")
        exe = StreamExecutor(
            backend=backend,
            arguments={},
            default_sampling_para=SglSamplingParams(),
            chat_template=get_chat_template("default"),
            stream=False,
            use_thread=True,
        )

        exe.submit(SglConstantText("Prompt: "))
        exe.submit(SglGen("first"))

        # Wait briefly for worker thread to fail and exit
        deadline = time.monotonic() + 2.0
        while exe.worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(exe.worker.is_alive())

        # Submitting additional expressions after worker exit must not leave
        # unconsumed items in the queue or unset variable events.
        exe.submit(SglConstantText(" trailing"))
        exe.submit(SglGen("second"))
        exe.end()

        err = self._run_with_timeout(exe.error)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("boom", str(err))
        self.assertEqual(self._run_with_timeout(exe.text), "Prompt: ")
        self.assertEqual(self._run_with_timeout(exe.messages), [])

        with self.assertRaises(KeyError):
            self._run_with_timeout(lambda: exe.get_var("second"))

    def test_multi_step_sgl_function_error_does_not_deadlock(self):
        @sgl.function
        def multi_step(s):
            s += "Step 1: " + sgl.gen("step1")
            time.sleep(0.05)
            s += "Step 2: " + sgl.gen("step2")

        backend = _FailingBackend("step1 failed")
        state = multi_step.run(backend=backend)

        err = self._run_with_timeout(state.error)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("step1 failed", str(err))
        self.assertEqual(self._run_with_timeout(state.text), "Step 1: ")

    def test_streaming_events_signaled_when_submitted_after_error(self):
        backend = _FailingBackend("stream boom")
        exe = StreamExecutor(
            backend=backend,
            arguments={},
            default_sampling_para=SglSamplingParams(),
            chat_template=get_chat_template("default"),
            stream=True,
            use_thread=True,
        )

        exe.submit(SglGen("first"))
        deadline = time.monotonic() + 2.0
        while exe.worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(exe.worker.is_alive())

        exe.submit(SglGen("second"))
        self.assertTrue(exe.variable_event["second"].is_set())
        self.assertTrue(exe.stream_var_event["second"].is_set())
        self.assertTrue(exe.stream_text_event.is_set())
        self._run_with_timeout(exe.sync)
        exe.end()

    def test_repeated_end_does_not_deadlock_sync(self):
        backend = _FailingBackend()
        exe = StreamExecutor(
            backend=backend,
            arguments={},
            default_sampling_para=SglSamplingParams(),
            chat_template=get_chat_template("default"),
            stream=False,
            use_thread=True,
        )
        exe.submit(SglConstantText("hello"))
        exe.end()
        exe.end()
        self.assertEqual(self._run_with_timeout(exe.text), "hello")


if __name__ == "__main__":
    unittest.main()
