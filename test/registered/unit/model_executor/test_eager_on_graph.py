"""CPU tests of scoped eager replay without requiring CUDA graph capture."""

import gc
import sys
import weakref
from contextlib import contextmanager
from contextvars import Context
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.srt.model_executor.runner_utils.forward_batch import (
    get_forward_batch,
    set_forward_batch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


@contextmanager
def recording_capture():
    events = []
    graph = bcg.BreakableCUDAGraph()
    capture = SimpleNamespace(
        cuda_graph=graph,
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
        def _eager_scale(self, x, out):
            out.copy_(x * get_forward_batch().scale)

    layer = Layer()
    x = torch.tensor([2.0])
    output = torch.zeros(1)
    with set_forward_batch(SimpleNamespace(scale=3)):
        with recording_capture() as (graph, events):
            assert layer._eager_scale(x, output) is None
    assert events == ["end", "begin"]
    assert output.item() == 0
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


def test_method_on_capture_and_real_replay():
    calls = []

    class Layer:
        def _on_capture(self, x, out):
            calls.append("stub")

        @bcg.eager_on_graph(on_capture=_on_capture)
        def _eager_scale(self, x, out):
            calls.append("real")
            out.copy_(x * get_forward_batch().scale)

    layer = Layer()
    output = torch.zeros(2)
    with recording_capture() as (graph, _):
        layer._eager_scale(torch.ones(2), output)
    assert calls == ["stub"]
    with set_forward_batch(SimpleNamespace(scale=7)):
        graph._break_fns[0]()
    assert calls == ["stub", "real"]
    torch.testing.assert_close(output, torch.full((2,), 7.0))


def test_plain_callable_and_eager_passthrough():
    @bcg.eager_on_graph
    def increment(x, out):
        out.copy_(x + 1)

    output = torch.tensor(0)
    increment(torch.tensor(2), output)
    assert output.item() == 3
    x = torch.tensor(3)
    with recording_capture() as (graph, _):
        increment(x, output)
    x.fill_(9)
    graph._break_fns[0]()
    assert output.item() == 10


def test_a_break_returning_a_value_fails():
    @bcg.eager_on_graph
    def increment(x):
        return x + 1

    with pytest.raises(AssertionError, match="must return None"):
        increment(torch.tensor(2))
    with recording_capture() as (graph, _):
        increment(torch.tensor(2))
    with pytest.raises(AssertionError, match="must return None"):
        graph._break_fns[0]()


def test_replay_does_not_retain_capture_or_serving_batches():
    class Batch:
        scale = 3

    @bcg.eager_on_graph
    def scale(x, out):
        out.copy_(x * get_forward_batch().scale)

    batch = Batch()
    capture_ref = weakref.ref(batch)
    with set_forward_batch(batch):
        with recording_capture() as (graph, _):
            scale(torch.ones(2), torch.zeros(2))
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
    @decorator
    def increment(x, out):
        out.copy_(x + 1)

    output = torch.tensor(0)
    with recording_capture() as (graph, _):
        increment(torch.tensor(2), output)
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
    with set_forward_batch(SimpleNamespace(scale=2)):
        with pytest.raises(RuntimeError, match="No forward batch"):
            Context().run(get_forward_batch)
        assert get_forward_batch().scale == 2


def test_run_eager_binds_instance_and_keeps_one_outer_break():
    calls = []

    class Layer:
        def __init__(self, scale):
            self.scale = scale

        def stub(self, x, *, offset=0):
            raise AssertionError("run_eager must bypass on_capture")

        @bcg.eager_on_graph(on_capture=stub)
        def inner(self, x, *, offset=0):
            calls.append(self.scale)
            return x * self.scale + offset

        @bcg.eager_on_graph
        def outer(self, x, out):
            out.copy_(self.inner.run_eager(x, offset=1))

    first, second = Layer(2), Layer(5)
    output = torch.zeros(1)
    with recording_capture() as (graph, events):
        first.outer(torch.ones(1), output)
    assert events == ["end", "begin"]
    assert len(graph._break_fns) == 1
    assert calls == [] and output.item() == 0
    assert second.inner.run_eager(torch.ones(1), offset=1).item() == 6
    graph._break_fns[0]()
    assert calls == [5, 2]
    assert output.item() == 3


def test_run_eager_plain_function_does_not_record_a_break():
    @bcg.eager_on_graph
    def increment(x):
        return x + 1

    with recording_capture() as (graph, events):
        assert increment.run_eager(torch.tensor(2)).item() == 3
    assert events == [] and graph._break_fns == []


def test_run_eager_method_torch_compile():
    class Layer(torch.nn.Module):
        @bcg.eager_on_graph
        def inner(self, x):
            return x + 1

        def forward(self, x):
            return self.inner.run_eager(x)

    layer = Layer()
    x = torch.ones(2)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), layer(x))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
