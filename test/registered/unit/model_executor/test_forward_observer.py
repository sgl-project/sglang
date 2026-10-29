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
class DeviceOutput:
    values: torch.Tensor

    def copy_to_host(self, copy_tensor):
        return SimpleNamespace(values=copy_tensor(self.values), consume=Mock())


@pytest.mark.parametrize(
    ("runner_kwargs", "topology", "mlx"),
    [
        ({}, {}, False),
        (dict(spec_algorithm=SpeculativeAlgorithm.EAGLE), {}, False),
        (dict(is_draft_worker=True), {}, False),
        (dict(is_generation=False), {}, False),
        ({}, dict(dllm_algorithm="dream"), False),
        ({}, dict(pp_size=2), False),
        ({}, dict(enable_pdmux=True), False),
        ({}, {}, True),
    ],
)
def test_forward_observer_registration(runner_kwargs, topology, mlx):
    observer = Mock()
    with (
        published_topology(**topology),
        patch("sglang.srt.hardware_backend.mlx.runtime.use_mlx", return_value=mlx),
    ):
        runner = object.__new__(ModelRunner)
        runner.spec_algorithm = SpeculativeAlgorithm.NONE
        runner.is_draft_worker = False
        runner.is_generation = True
        runner._forward_observer = None
        runner.__dict__.update(runner_kwargs)
        if runner_kwargs or topology or mlx:
            with pytest.raises(ValueError, match="configured forward path"):
                runner.forward_observer = observer
        else:
            runner.forward_observer = observer
            assert runner.forward_observer is observer


def _tp_worker(runner):
    worker = object.__new__(TpModelWorker)
    worker._model_runner = runner
    worker.hicache_layer_transfer_counter = None
    worker.dllm_algorithm = None
    worker.pp_group = SimpleNamespace(is_last_rank=True)
    worker.enable_overlap = False
    worker.enable_spec = False
    return worker


@pytest.mark.parametrize(
    ("has_observer", "can_run_graph"), [(False, False), (True, False), (True, True)]
)
def test_tp_worker_attaches_observer_output_to_the_result(has_observer, can_run_graph):
    logits_output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
    device_output = DeviceOutput(torch.tensor([1.0]))
    observer = Mock(after_forward=Mock(return_value=device_output))
    runner = SimpleNamespace(
        forward=Mock(
            return_value=ModelRunnerOutput(
                logits_output=logits_output, can_run_graph=can_run_graph
            )
        ),
        forward_observer=observer if has_observer else None,
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

    if has_observer:
        observer.after_forward.assert_called_once_with(
            batch, forward_batch, logits_output, can_run_graph=can_run_graph
        )
        assert result.forward_auxiliary_output is device_output
    else:
        assert result.forward_auxiliary_output is None
    assert result.next_token_ids.tolist() == [3]


@pytest.mark.parametrize("has_sampling_output", [False, True])
def test_forward_outputs_are_copied_to_the_host(has_sampling_output):
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            auxiliary_device_output=(
                DeviceOutput(torch.tensor([2.0])) if has_sampling_output else None
            ),
        ),
        next_token_ids=torch.tensor([7]),
        copy_done=Mock(),
        forward_auxiliary_output=DeviceOutput(torch.tensor([1.0])),
    )

    result.copy_to_cpu(return_logprob=False)

    host_output = result.auxiliary_host_output
    if has_sampling_output:
        assert isinstance(host_output, CompositeHostAuxiliaryOutput)
        assert [output.values.tolist() for output in host_output.outputs] == [
            [1.0],
            [2.0],
        ]
    else:
        assert host_output.values.tolist() == [1.0]
    assert result.forward_auxiliary_output is None
    assert result.logits_output.auxiliary_device_output is None


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
