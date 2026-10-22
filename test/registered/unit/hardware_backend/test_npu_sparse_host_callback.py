"""Callback startup errors must reach the caller, rather than strand ready.wait()."""

import importlib.util
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a")

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/host_callback.py"
)


def load_callback(rt):
    spec = importlib.util.spec_from_file_location("isolated_host_callback", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"acl": SimpleNamespace(rt=rt)}):
        spec.loader.exec_module(module)
    return module


class TestSparseHostCallback(CustomTestCase):
    def test_failed_context_initialization_unblocks_caller(self):
        module = load_callback(
            SimpleNamespace(get_context=lambda: (123, 0), set_context=lambda _ctx: 71)
        )

        def failed_device_init(_device):
            raise RuntimeError("device startup failed")

        module.torch = SimpleNamespace(
            npu=SimpleNamespace(set_device=failed_device_init)
        )
        finished = threading.Event()
        errors = []

        def register():
            try:
                module.register_npu_host_callback_stream(
                    SimpleNamespace(npu_stream=5), SimpleNamespace(index=0)
                )
            except RuntimeError as error:
                errors.append(error)
            finally:
                finished.set()

        caller = threading.Thread(target=register, daemon=True)
        caller.start()
        self.assertTrue(finished.wait(2), "Reporter failure stranded its caller")
        caller.join(timeout=1)
        self.assertEqual(len(errors), 1)
        self.assertIn("initialization failed", str(errors[0]))
        self.assertIn("ret=71", str(errors[0].__cause__))
        self.assertFalse(module._REPORTERS)

    def test_callback_uses_stream_context_on_reporter_thread(self):
        local = threading.local()
        callbacks = {}
        bindings = {}
        current_context = [123]
        observed = []
        done = threading.Event()

        def set_context(context):
            local.context = context
            callbacks[threading.get_ident()] = queue.Queue()
            return 0

        def process_report(_timeout):
            try:
                callback = callbacks[threading.get_ident()].get(timeout=0.01)
            except queue.Empty:
                return 0
            callback(local.context, threading.get_ident())
            return 0

        def subscribe(thread_id, stream):
            bindings[stream] = thread_id
            return 0

        rt = SimpleNamespace(
            get_context=lambda: (current_context[0], 0),
            set_context=set_context,
            process_report=process_report,
            subscribe_report=subscribe,
            unsubscribe_report=lambda *_args: 0,
        )
        module = load_callback(rt)
        try:
            module.register_npu_host_callback_stream(
                SimpleNamespace(npu_stream=5), SimpleNamespace(index=0)
            )
            current_context[0] = 456
            module.register_npu_host_callback_stream(
                SimpleNamespace(npu_stream=6), SimpleNamespace(index=0)
            )

            def callback(context, thread_id):
                observed.append((context, thread_id))
                if len(observed) == 2:
                    done.set()

            for stream in (5, 6):
                callbacks[bindings[stream]].put(callback)
            self.assertTrue(done.wait(2))
            self.assertEqual(sorted(context for context, _ in observed), [123, 456])
            self.assertEqual(len({thread_id for _, thread_id in observed}), 2)
            self.assertTrue(
                all(thread_id != threading.get_ident() for _, thread_id in observed)
            )
        finally:
            for reporter in module._REPORTERS.values():
                reporter.close()


if __name__ == "__main__":
    unittest.main()
