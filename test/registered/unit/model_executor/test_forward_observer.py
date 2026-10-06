"""Unit tests for ModelRunner.forward_observer and its auxiliary-output lifecycle."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.auxiliary_output import CompositeHostAuxiliaryOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@dataclass
class HostOutput:
    values: torch.Tensor

    def consume(self, batch, commits):
        pass


class DeviceOutput:
    def __init__(self, values: torch.Tensor):
        self.values = values

    def copy_to_host(self, copy_tensor):
        return HostOutput(copy_tensor(self.values))


class CopyDone:
    def record(self):
        pass


class RecordingObserver:
    def __init__(self, output=None):
        self.output = output
        self.calls = []

    def after_forward(self, batch, forward_batch, logits_output, *, can_run_graph):
        self.calls.append((batch, forward_batch, logits_output, can_run_graph))
        return self.output


def _model_runner(*, spec_algorithm=SpeculativeAlgorithm.NONE, is_draft_worker=False):
    runner = object.__new__(ModelRunner)
    runner.spec_algorithm = spec_algorithm
    runner.is_draft_worker = is_draft_worker
    runner._forward_observer = None
    return runner


def test_forward_observer_installs_on_a_target_runner():
    observer = RecordingObserver()
    with published_topology():
        runner = _model_runner()
        runner.forward_observer = observer

    assert runner.forward_observer is observer


@pytest.mark.parametrize(
    ("runner_kwargs", "topology"),
    [
        (dict(spec_algorithm=SpeculativeAlgorithm.EAGLE), {}),
        (dict(is_draft_worker=True), {}),
        ({}, dict(dllm_algorithm="dream")),
        ({}, dict(pp_size=2)),
    ],
)
def test_forward_observer_rejects_paths_that_bypass_it(runner_kwargs, topology):
    with published_topology(**topology):
        runner = _model_runner(**runner_kwargs)
        with pytest.raises(ValueError, match="configured forward path"):
            runner.forward_observer = RecordingObserver()


def _tp_worker(runner):
    worker = object.__new__(TpModelWorker)
    worker._model_runner = runner
    worker.hicache_layer_transfer_counter = None
    worker.dllm_algorithm = None
    worker.pp_group = SimpleNamespace(is_last_rank=True)
    worker.enable_overlap = False
    worker.enable_spec = False
    return worker


@pytest.mark.parametrize("can_run_graph", [False, True])
def test_tp_worker_attaches_observer_output_to_the_result(can_run_graph):
    logits_output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
    device_output = DeviceOutput(torch.tensor([1.0]))
    observer = RecordingObserver(device_output)
    runner = SimpleNamespace(
        forward=Mock(
            return_value=ModelRunnerOutput(
                logits_output=logits_output, can_run_graph=can_run_graph
            )
        ),
        forward_observer=observer,
        sample=Mock(return_value=torch.tensor([3])),
        _elastic_cuda_graph_enabled=Mock(return_value=False),
    )
    batch = SimpleNamespace(hicache_consumer_index=0)
    forward_batch = SimpleNamespace(
        is_prefill_only=False,
        apply_deprecated_skip_attn_backend_init=Mock(),
    )

    with (
        published_topology(),
        patch.object(ForwardBatch, "init_new", return_value=forward_batch),
        patch("sglang.srt.managers.tp_worker.capture_pre_sample_logits"),
    ):
        result = _tp_worker(runner).forward_batch_generation(batch)

    assert observer.calls == [(batch, forward_batch, logits_output, can_run_graph)]
    assert result.forward_auxiliary_output is device_output
    assert result.next_token_ids.tolist() == [3]


def test_tp_worker_without_observer_leaves_the_result_unchanged():
    logits_output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
    runner = SimpleNamespace(
        forward=Mock(
            return_value=ModelRunnerOutput(
                logits_output=logits_output, can_run_graph=False
            )
        ),
        forward_observer=None,
        sample=Mock(return_value=torch.tensor([3])),
        _elastic_cuda_graph_enabled=Mock(return_value=False),
    )
    forward_batch = SimpleNamespace(
        is_prefill_only=False,
        apply_deprecated_skip_attn_backend_init=Mock(),
    )

    with (
        published_topology(),
        patch.object(ForwardBatch, "init_new", return_value=forward_batch),
        patch("sglang.srt.managers.tp_worker.capture_pre_sample_logits"),
    ):
        result = _tp_worker(runner).forward_batch_generation(
            SimpleNamespace(hicache_consumer_index=0)
        )

    assert result.forward_auxiliary_output is None


def test_forward_and_sampling_outputs_share_one_host_copy():
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            auxiliary_device_output=DeviceOutput(torch.tensor([2.0])),
        ),
        next_token_ids=torch.tensor([7]),
        copy_done=CopyDone(),
        forward_auxiliary_output=DeviceOutput(torch.tensor([1.0])),
    )

    result.copy_to_cpu(return_logprob=False)

    host_output = result.auxiliary_host_output
    assert isinstance(host_output, CompositeHostAuxiliaryOutput)
    assert [output.values.tolist() for output in host_output.outputs] == [
        [1.0],
        [2.0],
    ]
    assert result.forward_auxiliary_output is None
    assert result.logits_output.auxiliary_device_output is None


def test_forward_output_alone_is_copied_to_the_host():
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=None),
        next_token_ids=torch.tensor([7]),
        copy_done=CopyDone(),
        forward_auxiliary_output=DeviceOutput(torch.tensor([1.0])),
    )

    result.copy_to_cpu(return_logprob=False)

    assert result.auxiliary_host_output.values.tolist() == [1.0]
    assert result.forward_auxiliary_output is None


@pytest.mark.parametrize("has_forward_output", [False, True])
def test_scheduler_copies_forward_output_for_non_overlap_results(has_forward_output):
    event = object()
    scheduler = object.__new__(Scheduler)
    scheduler.device_module = SimpleNamespace(Event=Mock(return_value=event))
    result = SimpleNamespace(
        logits_output=SimpleNamespace(auxiliary_device_output=None),
        forward_auxiliary_output=object() if has_forward_output else None,
        auxiliary_host_output=None,
        copy_done=None,
        copy_to_cpu=Mock(),
    )
    batch = SimpleNamespace(return_logprob=False, return_hidden_states=False)

    with published_topology():
        Scheduler._copy_auxiliary_output_to_cpu(scheduler, batch, result)

    if has_forward_output:
        assert result.copy_done is event
        result.copy_to_cpu.assert_called_once_with(
            return_logprob=False,
            return_hidden_states=False,
        )
    else:
        assert result.copy_done is None
        result.copy_to_cpu.assert_not_called()
