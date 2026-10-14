"""The schedule stream is settled at construction: the stream the event loop
runs on is the one every component built in __init__ waits on."""

import ast
import inspect
import textwrap
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_mod  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Stream:
    def __init__(self, handle):
        self.cuda_stream = handle


class _DeviceModule:
    """Hands out pool streams in order; the first aliases the forward stream."""

    def __init__(self, handles):
        self.handles = list(handles)
        self.entered = []

    def Stream(self, priority=0):
        return _Stream(self.handles.pop(0))

    @contextmanager
    def StreamContext(self, stream):
        self.entered.append(stream)
        yield


def _scheduler(device_module, schedule_stream, forward_stream):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.device = "cuda"
    get_context().override("test", disable_overlap_schedule=False)
    scheduler.device_module = device_module
    scheduler.schedule_stream = schedule_stream
    scheduler.forward_stream = forward_stream
    scheduler.metrics_sink = SimpleNamespace(
        start_scheduler_time_accounting=lambda: None
    )
    return scheduler


class TestScheduleStreamSettledAtInit(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_loop_runs_on_the_stream_settled_at_init(self):
        forward_stream = _Stream(1)
        device_module = _DeviceModule(handles=[1, 2])
        scheduler = _scheduler(device_module, _Stream(1), forward_stream)

        with patch.object(scheduler_mod, "is_cuda", return_value=True):
            scheduler.finalize_schedule_stream()
        settled = scheduler.schedule_stream
        self.assertNotEqual(settled.cuda_stream, forward_stream.cuda_stream)

        # A component built after the settle holds this stream; the loop must
        # run on the same object.
        with (
            patch.object(scheduler_mod, "is_cuda", return_value=True),
            patch.object(scheduler_mod, "dispatch_event_loop"),
            patch.object(scheduler_mod, "triton_load_watch"),
        ):
            scheduler.run_event_loop()
        self.assertIs(scheduler.schedule_stream, settled)
        self.assertEqual(device_module.entered, [settled])

    def test_settled_before_the_components_that_wait_on_it(self):
        source = textwrap.dedent(inspect.getsource(Scheduler.__init__))
        calls = [
            node.func.attr
            for node in sorted(
                ast.walk(ast.parse(source)), key=lambda n: getattr(n, "lineno", 0)
            )
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ]
        settle = calls.index("finalize_schedule_stream")
        for owner in ("maybe_init_disagg_prefill", "maybe_init_disagg_decode"):
            self.assertLess(settle, calls.index(owner), owner)


if __name__ == "__main__":
    unittest.main()
