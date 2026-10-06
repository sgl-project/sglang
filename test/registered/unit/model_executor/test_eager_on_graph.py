"""CPU tests of scoped eager replay without requiring CUDA graph capture."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_context import (
    get_forward_batch,
    set_forward_batch,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@contextmanager
def recording_capture():
    events = []
    graph = bcg.BreakableCUDAGraph()
    capture = SimpleNamespace(
        cuda_graph=graph,
        _barrier_fn=None,
        _end_current_segment=lambda: events.append("end"),
        _begin_new_segment=lambda: events.append("begin"),
    )
    token = bcg._current_capture_var.set(capture)
    try:
        yield graph, events
    finally:
        bcg._current_capture_var.reset(token)


def test_method_replay_reads_current_batch_and_preserves_output_address(monkeypatch):
    class Layer:
        @bcg.eager_on_graph
        def _eager_scale(self, x):
            return x * get_forward_batch().scale

    layer = Layer()
    x = torch.tensor([2.0])
    with set_forward_batch(SimpleNamespace(scale=3)):
        with recording_capture() as (graph, events):
            output = layer._eager_scale(x)
    assert events == ["end", "begin"]
    pointer = output.data_ptr()
    # Exercise the public, argument-free replay API between two segments.
    graph._segments = [
        SimpleNamespace(replay=lambda: events.append("first")),
        SimpleNamespace(replay=lambda: events.append("last")),
    ]
    monkeypatch.setattr(
        bcg, "get_device_module", lambda: SimpleNamespace(current_stream=lambda: None)
    )
    x.fill_(4)
    with set_forward_batch(SimpleNamespace(scale=5)):
        graph.replay()
    assert events == ["end", "begin", "first", "last"]
    assert output.item() == 20 and output.data_ptr() == pointer
    with pytest.raises(RuntimeError, match="No forward batch"):
        graph.replay()


def test_method_capture_stub_and_real_replay():
    calls = []

    class Layer:
        def _capture_stub(self, x):
            calls.append("stub")
            return torch.zeros_like(x)

        @bcg.eager_on_graph(capture_stub=_capture_stub)
        def _eager_scale(self, x):
            calls.append("real")
            return x * get_forward_batch().scale

    layer = Layer()
    with recording_capture() as (graph, _):
        output = layer._eager_scale(torch.ones(2))
    assert calls == ["stub"]
    with set_forward_batch(SimpleNamespace(scale=7)):
        graph._break_fns[0]()
    assert calls == ["stub", "real"]
    torch.testing.assert_close(output, torch.full((2,), 7.0))


def test_plain_callable_and_eager_passthrough():
    @bcg.eager_on_graph
    def increment(x):
        return x + 1

    assert increment(torch.tensor(2)).item() == 3
    x = torch.tensor(3)
    with recording_capture() as (graph, _):
        output = increment(x)
    x.fill_(9)
    graph._break_fns[0]()
    assert output.item() == 10


def test_replay_does_not_retain_capture_or_serving_batches():
    import gc
    import weakref

    class Batch:
        scale = 3

    @bcg.eager_on_graph
    def scale(x):
        return x * get_forward_batch().scale

    batch = Batch()
    capture_ref = weakref.ref(batch)
    with set_forward_batch(batch):
        with recording_capture() as (graph, _):
            scale(torch.ones(2))
    del batch
    gc.collect()
    assert capture_ref() is None

    batch = Batch()
    live_ref = weakref.ref(batch)
    with set_forward_batch(batch):
        graph._break_fns[0]()
    del batch
    gc.collect()
    assert live_ref() is None


@pytest.mark.parametrize(
    "decorator", [bcg.eager_on_graph(True), bcg.eager_on_graph(enable=True)]
)
def test_legacy_decorator_replays(decorator):
    increment = decorator(lambda x: x + 1)
    with recording_capture() as (graph, _):
        output = increment(torch.tensor(2))
    graph._break_fns[0]()
    assert output.item() == 3


def test_legacy_disabled_decorator_is_identity():
    fn = lambda x: x + 1
    assert bcg.eager_on_graph(False)(fn) is fn
    assert bcg.eager_on_graph(enable=False)(fn) is fn


def test_batch_scope_restores_outer_batch_after_exception():
    outer = SimpleNamespace(scale=2)
    with set_forward_batch(outer):
        with pytest.raises(ValueError, match="eager failure"):
            with set_forward_batch(SimpleNamespace(scale=9)):
                assert get_forward_batch().scale == 9
                raise ValueError("eager failure")
        assert get_forward_batch() is outer
    with pytest.raises(RuntimeError, match="No forward batch"):
        get_forward_batch()


def test_batch_scope_is_isolated_between_execution_contexts():
    from contextvars import Context

    with set_forward_batch(SimpleNamespace(scale=2)):
        with pytest.raises(RuntimeError, match="No forward batch"):
            Context().run(get_forward_batch)
        assert get_forward_batch().scale == 2
