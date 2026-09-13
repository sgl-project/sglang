"""CPU tests for NPU graph update handling, with real background threads."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_backend(graph, records=2):
    graph.auto_dispatch_capture = True
    graph.graph_dispatch_mode = SimpleNamespace(
        graph_dispatch_records=[object() for _ in range(records)]
    )
    backend = NPUCudaGraphBackend.__new__(NPUCudaGraphBackend)
    backend._graphs = {1: graph}
    backend._outputs = {1: object()}
    backend._device_module = SimpleNamespace(set_device=Mock())
    backend._device_id = 0
    return backend


@pytest.mark.parametrize("legacy", [False, True])
def test_npu_graph_update_success(legacy):
    graph = SimpleNamespace(update=Mock(), replay=Mock())
    backend = _make_backend(graph)
    if legacy:
        output = backend.replay_with_input_update(
            1, [3, 5], attr_name="actual_seq_lengths_kv", attr_type=torch.empty(0)
        )
        values = graph.update.call_args.kwargs["cpu_update_input"][0]
        torch.testing.assert_close(
            values["actual_seq_lengths_kv"], torch.tensor([3, 5], dtype=torch.int32)
        )
    else:
        inputs = [{"actual_seq_lengths_kv": [3]}, {"actual_seq_lengths_kv": [4]}]
        output = backend.replay_with_input_update(1, None, cpu_update_input=inputs)
        graph.update.assert_called_once_with(cpu_update_input=inputs)
    assert output is backend._outputs[1]
    graph.replay.assert_called_once_with()
    backend._device_module.set_device.assert_called_once_with(0)


@pytest.mark.parametrize(
    "message",
    [
        "Currently, there are 3 operators that need to be updated by capture, "
        "and there are only 2 elements in the incoming cpu_update_input list",
        "NPU stream update failed",
        # Every update failure must reach the caller, regardless of its message.
        "there are 0 operators that need to be updated",
    ],
)
def test_npu_graph_update_errors_reach_caller(message):
    error = RuntimeError(message)
    graph = SimpleNamespace(update=Mock(side_effect=error), replay=Mock())
    backend = _make_backend(graph)
    with pytest.raises(RuntimeError) as raised:
        backend.replay_with_input_update(1, None, cpu_update_input=[{}, {}])
    assert raised.value is error


def test_npu_graph_waits_for_update_before_returning():
    replay_started = threading.Event()
    update_finished = threading.Event()

    def update(**kwargs):
        if not replay_started.wait(timeout=5):
            raise RuntimeError("Replay did not run concurrently with update")
        update_finished.set()

    graph = SimpleNamespace(update=update, replay=replay_started.set)
    backend = _make_backend(graph)
    result = backend.replay_with_input_update(1, None, cpu_update_input=[{}, {}])
    assert update_finished.is_set()
    assert result is backend._outputs[1]


@pytest.mark.parametrize("failure_site", ["set_device", "update"])
def test_npu_graph_worker_exception_reaches_caller(failure_site):
    error = ValueError("Worker failed")
    graph = SimpleNamespace(update=Mock(), replay=Mock())
    backend = _make_backend(graph)
    if failure_site == "set_device":
        backend._device_module.set_device.side_effect = error
    else:
        graph.update.side_effect = error
    with pytest.raises(ValueError) as raised:
        backend.replay_with_input_update(1, None, cpu_update_input=[{}, {}])
    assert raised.value is error


@pytest.mark.parametrize("legacy", [False, True])
def test_npu_graph_zero_records_skip_update_and_replay(legacy):
    source = torch.tensor([1.0, 2.0])
    output = torch.zeros_like(source)
    graph = SimpleNamespace(
        update=Mock(side_effect=AssertionError("Empty graph must not update")),
        replay=Mock(side_effect=lambda: output.copy_(source * 2)),
    )
    backend = _make_backend(graph, records=0)
    backend._outputs[1] = output
    for value in (3.0, 7.0):
        source.fill_(value)
        if legacy:
            result = backend.replay_with_input_update(
                1, [3], attr_name="actual_seq_lengths_kv", attr_type=torch.empty(0)
            )
        else:
            result = backend.replay_with_input_update(1, None, cpu_update_input=[{}, {}])
        assert result is output
        torch.testing.assert_close(result, torch.full_like(output, value * 2))
    assert graph.replay.call_count == 2
    graph.update.assert_not_called()
    backend._device_module.set_device.assert_not_called()


@pytest.mark.parametrize("missing", ["auto_dispatch_capture", "mode", "records", "none"])
def test_npu_graph_missing_capture_state_fails_before_replay(missing):
    graph = SimpleNamespace(update=Mock(), replay=Mock())
    backend = _make_backend(graph)
    if missing == "auto_dispatch_capture":
        del graph.auto_dispatch_capture
    elif missing == "mode":
        del graph.graph_dispatch_mode
    elif missing == "records":
        del graph.graph_dispatch_mode.graph_dispatch_records
    else:
        graph.graph_dispatch_mode.graph_dispatch_records = None
    with pytest.raises(RuntimeError, match="Cannot inspect NPU graph update records"):
        backend.replay_with_input_update(1, None, cpu_update_input=[{}, {}])
    graph.update.assert_not_called()
    graph.replay.assert_not_called()


def test_npu_graph_without_auto_dispatch_fails_before_replay():
    graph = SimpleNamespace(update=Mock(), replay=Mock())
    backend = _make_backend(graph, records=0)
    graph.auto_dispatch_capture = False
    with pytest.raises(RuntimeError, match="auto_dispatch_capture=True"):
        backend.replay_with_input_update(1, None, cpu_update_input=[{}, {}])
    graph.update.assert_not_called()
    graph.replay.assert_not_called()


def test_npu_graph_update_decision_is_per_graph():
    empty = SimpleNamespace(update=Mock(), replay=Mock())
    backend = _make_backend(empty, records=0)
    nonempty = SimpleNamespace(
        auto_dispatch_capture=True,
        graph_dispatch_mode=SimpleNamespace(graph_dispatch_records=[object()]),
        update=Mock(),
        replay=Mock(),
    )
    backend._graphs[2] = nonempty
    backend._outputs[2] = object()
    for key in (1, 2, 1, 2):
        result = backend.replay_with_input_update(key, None, cpu_update_input=[{}])
        assert result is backend._outputs[key]
    empty.update.assert_not_called()
    assert empty.replay.call_count == 2
    assert nonempty.update.call_count == 2
    assert nonempty.replay.call_count == 2
