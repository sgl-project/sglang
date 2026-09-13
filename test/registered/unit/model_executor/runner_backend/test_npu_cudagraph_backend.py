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


def _make_backend(graph):
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
