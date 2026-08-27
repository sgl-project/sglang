from types import SimpleNamespace

from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Future:
    def __init__(self, events):
        self.events = events
        self.result_calls = 0

    def result(self):
        self.result_calls += 1
        self.events.append("wait")


class _Executor:
    def __init__(self, events):
        self.events = events
        self.futures = []

    def submit(self, fn, **kwargs):
        self.events.append(("submit", fn, kwargs))
        future = _Future(self.events)
        self.futures.append(future)
        return future


class _Graph:
    def __init__(self, events):
        self.events = events

    def update(self, **kwargs):
        raise AssertionError("the fake executor must not run graph.update")

    def replay(self):
        self.events.append("replay")


def _backend(defer):
    events = []
    graph = _Graph(events)
    backend = NPUCudaGraphBackend.__new__(NPUCudaGraphBackend)
    backend._graphs = {"shape": graph}
    backend._outputs = {"shape": SimpleNamespace(value=1)}
    backend._update_executor = _Executor(events)
    backend._defer_update_wait = defer
    backend._pending_update = None
    return backend, events


def test_deferred_update_is_reaped_before_next_updated_replay():
    backend, events = _backend(defer=True)

    backend.replay_with_input_update(
        "shape", seq_lens=None, cpu_update_input=[{"seq": [1]}]
    )
    first = backend._update_executor.futures[0]
    assert first.result_calls == 0

    backend.replay_with_input_update(
        "shape", seq_lens=None, cpu_update_input=[{"seq": [2]}]
    )
    assert first.result_calls == 1
    labels = [event[0] if isinstance(event, tuple) else event for event in events]
    assert labels[:4] == ["submit", "replay", "wait", "submit"]


def test_plain_replay_reaps_pending_update():
    backend, events = _backend(defer=True)
    backend.replay_with_input_update(
        "shape", seq_lens=None, cpu_update_input=[{"seq": [1]}]
    )

    backend.replay("shape", static_forward_batch=None)

    assert backend._update_executor.futures[0].result_calls == 1
    assert events[-2:] == ["wait", "replay"]


def test_default_path_waits_before_returning():
    backend, _ = _backend(defer=False)

    backend.replay_with_input_update(
        "shape", seq_lens=None, cpu_update_input=[{"seq": [1]}]
    )

    assert backend._update_executor.futures[0].result_calls == 1
    assert backend._pending_update is None
