from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@dataclass
class HostOutput:
    values: torch.Tensor


class DeviceOutput:
    def __init__(self, values: torch.Tensor):
        self.values = values
        self.copy_count = 0

    def copy_to_host(self, copy_tensor):
        self.copy_count += 1
        return HostOutput(copy_tensor(self.values))

    def to_pp_tensors(self):
        return {"values": self.values}


class HostOnlyDeviceOutput:
    def __init__(self, values: torch.Tensor):
        self.values = values

    def copy_to_host(self, copy_tensor):
        return HostOutput(copy_tensor(self.values))


class Observer:
    def __init__(self):
        self.received_tensors = None

    def from_pp_tensors(self, tensors):
        self.received_tensors = tensors
        return DeviceOutput(tensors["values"])


class CopyDone:
    def __init__(self):
        self.record_count = 0

    def record(self):
        self.record_count += 1


def _model_runner_for_sampling_path(
    *,
    spec_algorithm=SpeculativeAlgorithm.NONE,
    dllm_algorithm=None,
):
    runner = object.__new__(ModelRunner)
    runner.server_args = SimpleNamespace(dllm_algorithm=dllm_algorithm)
    runner.spec_algorithm = spec_algorithm
    runner._sampling_observer = None
    return runner


def test_auxiliary_output_releases_device_holder_after_copy():
    device_output = DeviceOutput(torch.tensor([1.0, 2.0]))
    logits_output = LogitsProcessorOutput(
        next_token_logits=None,
        auxiliary_device_output=device_output,
    )
    result = GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=torch.tensor([7]),
        copy_done=CopyDone(),
    )

    result.copy_to_cpu(return_logprob=False)

    assert logits_output.auxiliary_device_output is None
    assert result.auxiliary_host_output is not device_output
    assert device_output.copy_count == 1
    assert result.copy_done.record_count == 1


def test_non_pp_auxiliary_output_only_requires_host_copy_support():
    device_output = HostOnlyDeviceOutput(torch.tensor([1.0, 2.0]))
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            auxiliary_device_output=device_output,
        ),
        next_token_ids=torch.tensor([7]),
        copy_done=CopyDone(),
    )

    result.copy_to_cpu(return_logprob=False)

    assert torch.equal(result.auxiliary_host_output.values, device_output.values)


def test_auxiliary_host_outputs_are_owned_by_each_generation_result():
    logits_output = LogitsProcessorOutput(next_token_logits=None)
    first_device = DeviceOutput(torch.tensor([1.0]))
    logits_output.auxiliary_device_output = first_device
    first = GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=torch.tensor([1]),
        copy_done=CopyDone(),
    )
    first.copy_to_cpu(return_logprob=False)

    second_device = DeviceOutput(torch.tensor([2.0]))
    logits_output.auxiliary_device_output = second_device
    second = GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=torch.tensor([2]),
        copy_done=CopyDone(),
    )
    second.copy_to_cpu(return_logprob=False)

    assert first.auxiliary_host_output.values.tolist() == [1.0]
    assert second.auxiliary_host_output.values.tolist() == [2.0]
    assert first_device.copy_count == second_device.copy_count == 1


def test_sampling_clears_stale_device_output_when_observer_produces_no_state():
    runner = _model_runner_for_sampling_path()
    runner.sampling_observer = SimpleNamespace(
        is_active=lambda sampling_info: True,
        after_sample=lambda state, token_ids: None,
    )
    runner._preprocess_logits = Mock(return_value=None)
    runner.sampler = lambda *args, **kwargs: torch.tensor([3])
    runner.ngram_embedding_manager = SimpleNamespace(
        update_after_decode=lambda **kwargs: None
    )
    logits_output = LogitsProcessorOutput(
        next_token_logits=torch.zeros(1, 4),
        auxiliary_device_output=DeviceOutput(torch.tensor([99.0])),
    )
    forward_batch = SimpleNamespace(
        sampling_info=object(),
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        positions=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )

    ModelRunner.sample(runner, logits_output, forward_batch)

    assert logits_output.auxiliary_device_output is None
    runner._preprocess_logits.assert_called_once_with(
        logits_output,
        forward_batch.sampling_info,
        observer=runner.sampling_observer,
    )


def test_sampling_publishes_observer_output_for_the_sampled_tokens():
    state = object()
    device_output = DeviceOutput(torch.tensor([4.0]))
    observer = SimpleNamespace(
        is_active=Mock(return_value=True),
        after_sample=Mock(return_value=device_output),
    )
    runner = _model_runner_for_sampling_path()
    runner.sampling_observer = observer
    runner._preprocess_logits = Mock(return_value=state)
    sampled_tokens = torch.tensor([3])
    runner.sampler = Mock(return_value=sampled_tokens)
    runner.ngram_embedding_manager = SimpleNamespace(
        update_after_decode=lambda **kwargs: None
    )
    logits_output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
    forward_batch = SimpleNamespace(
        sampling_info=object(),
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        positions=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )

    ModelRunner.sample(runner, logits_output, forward_batch)

    assert logits_output.auxiliary_device_output is device_output
    runner._preprocess_logits.assert_called_once_with(
        logits_output,
        forward_batch.sampling_info,
        observer=observer,
    )
    observer.is_active.assert_called_once_with(forward_batch.sampling_info)
    observer.after_sample.assert_called_once_with(state, sampled_tokens)


@pytest.mark.parametrize(
    ("spec_algorithm", "dllm_algorithm"),
    [
        (SpeculativeAlgorithm.EAGLE, None),
        (SpeculativeAlgorithm.NONE, "dream"),
    ],
)
def test_sampling_observer_rejects_sampling_paths_that_bypass_hooks(
    spec_algorithm,
    dllm_algorithm,
):
    runner = _model_runner_for_sampling_path(
        spec_algorithm=spec_algorithm,
        dllm_algorithm=dllm_algorithm,
    )

    with pytest.raises(ValueError, match="configured sampling path"):
        runner.sampling_observer = Observer()


def test_custom_sampling_path_can_enable_sampling_observer():
    class SupportedModelRunner(ModelRunner):
        def supports_sampling_observer(self):
            return True

    runner = object.__new__(SupportedModelRunner)
    observer = Observer()

    runner.sampling_observer = observer

    assert runner.sampling_observer is observer


@pytest.mark.parametrize("has_inactive_observer", [False, True])
def test_sampling_without_active_observer_preserves_preprocess_override(
    has_inactive_observer,
):
    observer = (
        SimpleNamespace(is_active=Mock(return_value=False))
        if has_inactive_observer
        else None
    )
    runner = _model_runner_for_sampling_path()
    runner.sampling_observer = observer
    runner._preprocess_logits = Mock(
        side_effect=lambda logits_output, sampling_info: None
    )
    runner.sampler = Mock(return_value=torch.tensor([3]))
    runner.ngram_embedding_manager = SimpleNamespace(
        update_after_decode=lambda **kwargs: None
    )
    logits_output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
    forward_batch = SimpleNamespace(
        sampling_info=object(),
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        positions=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )

    ModelRunner.sample(runner, logits_output, forward_batch)

    runner._preprocess_logits.assert_called_once_with(
        logits_output, forward_batch.sampling_info
    )
    if observer is not None:
        observer.is_active.assert_called_once_with(forward_batch.sampling_info)


def test_preprocess_logits_without_observer_uses_standard_path():
    runner = object.__new__(ModelRunner)
    logits_output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
    grammar_mask = object()
    sampling_info = SimpleNamespace(
        grammar_mask=grammar_mask,
        update_regex_vocab_mask=Mock(),
        apply_logits_bias=Mock(),
        apply_logits_bias_with_observer=Mock(),
    )

    state = ModelRunner._preprocess_logits(
        runner,
        logits_output,
        sampling_info,
    )

    assert state is None
    sampling_info.update_regex_vocab_mask.assert_called_once_with()
    sampling_info.apply_logits_bias.assert_called_once_with(
        logits_output.next_token_logits
    )
    sampling_info.apply_logits_bias_with_observer.assert_not_called()
    assert sampling_info.grammar_mask is None


def test_active_observer_uses_observer_logits_preprocessing():
    runner = object.__new__(ModelRunner)
    observer = SimpleNamespace()
    observer_state = object()
    logits_output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
    sampling_info = SimpleNamespace(
        grammar_mask=object(),
        update_regex_vocab_mask=Mock(),
        apply_logits_bias=Mock(),
        apply_logits_bias_with_observer=Mock(return_value=observer_state),
    )

    state = ModelRunner._preprocess_logits(
        runner,
        logits_output,
        sampling_info,
        observer=observer,
    )

    assert state is observer_state
    sampling_info.update_regex_vocab_mask.assert_called_once_with()
    sampling_info.apply_logits_bias.assert_not_called()
    sampling_info.apply_logits_bias_with_observer.assert_called_once_with(
        logits_output.next_token_logits,
        observer=observer,
    )
    assert sampling_info.grammar_mask is None


def test_scheduler_copies_auxiliary_output_for_non_overlap_results():
    event = object()
    scheduler = object.__new__(Scheduler)
    scheduler.ps = SimpleNamespace(pp_size=1)
    scheduler.device_module = SimpleNamespace(Event=Mock(return_value=event))
    result = SimpleNamespace(
        logits_output=SimpleNamespace(auxiliary_device_output=object()),
        auxiliary_host_output=None,
        copy_done=None,
        copy_to_cpu=Mock(),
    )
    batch = SimpleNamespace(return_logprob=False, return_hidden_states=False)

    Scheduler._copy_auxiliary_output_to_cpu(scheduler, batch, result)

    assert result.copy_done is event
    result.copy_to_cpu.assert_called_once_with(
        return_logprob=False,
        return_hidden_states=False,
    )


def test_scheduler_preserves_pipeline_parallel_output_for_transport():
    scheduler = object.__new__(Scheduler)
    scheduler.ps = SimpleNamespace(pp_size=2)
    scheduler.device_module = SimpleNamespace(Event=Mock())
    result = SimpleNamespace(
        logits_output=SimpleNamespace(auxiliary_device_output=object()),
        auxiliary_host_output=None,
        copy_done=None,
        copy_to_cpu=Mock(),
    )
    batch = SimpleNamespace(return_logprob=False, return_hidden_states=False)

    Scheduler._copy_auxiliary_output_to_cpu(scheduler, batch, result)

    assert result.copy_done is None
    result.copy_to_cpu.assert_not_called()
    scheduler.device_module.Event.assert_not_called()


def test_pdmux_split_prefill_schedules_auxiliary_output_copy():
    device_output = DeviceOutput(torch.tensor([1.0]))
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            auxiliary_device_output=device_output,
        ),
        next_token_ids=torch.tensor([7]),
    )
    copy_done = CopyDone()
    scheduler = object.__new__(Scheduler)
    scheduler.scheduler_stage_metrics = None
    scheduler.metrics_reporter = Mock()
    scheduler.forward_ct = 0
    scheduler._sched_idled = False
    scheduler.scripted_scheduler_hook = None
    scheduler.profiler_manager = SimpleNamespace(_profile_batch_predicate=Mock())
    scheduler.forward_sleep_time = None
    scheduler.disaggregation_mode = None
    scheduler.is_generation = True
    scheduler.enable_overlap = False
    scheduler.enable_pdmux = True
    scheduler.ps = SimpleNamespace(pp_size=1)
    scheduler.tp_worker = SimpleNamespace(
        forward_batch_split_prefill=Mock(return_value=result)
    )
    scheduler.future_map = object()
    scheduler._relay_forward_payload = Mock()
    scheduler.device_module = SimpleNamespace(Event=Mock(return_value=copy_done))
    scheduler.enable_dp_attention = False
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_prebuilt=lambda: False,
            is_split_prefill=lambda: True,
        ),
        reqs=[],
        req_pool_indices=torch.tensor([3]),
        input_ids=torch.tensor([5]),
        return_logprob=False,
        return_hidden_states=False,
    )

    with patch(
        "sglang.srt.managers.scheduler.resolve_forward_inputs"
    ) as resolve_forward_inputs:
        output_result = Scheduler.run_batch(scheduler, batch)

    resolve_forward_inputs.assert_called_once_with(batch, scheduler.future_map)
    assert output_result is result
    assert result.auxiliary_host_output.values.tolist() == [1.0]
    assert copy_done.record_count == 1


def test_disaggregated_prefill_consumes_auxiliary_output_after_commit():
    host_output = HostOutput(torch.tensor([1.0]))
    copy_done = SimpleNamespace(synchronize=Mock())
    result = GenerationBatchResult(
        logits_output=None,
        next_token_ids=torch.tensor([7]),
        next_draft_input=None,
        copy_done=copy_done,
        auxiliary_host_output=host_output,
    )
    req = SimpleNamespace(
        output_ids=[],
        finished_len=None,
        to_finish=None,
        finished_reason=None,
        inflight_middle_chunks=0,
        pending_bootstrap=False,
        return_logprob=False,
        return_sampling_mask=False,
        grammar=None,
        time_stats=SimpleNamespace(
            set_prefill_finished_time=Mock(),
            set_prefill_transfer_queue_entry_time=Mock(),
        ),
    )
    batch = SimpleNamespace(
        reqs=[req],
        spec_info=None,
        prefill_stats=None,
        dp_cooperation_info=None,
    )
    snapshot_auxiliary_output_starts = Mock(
        side_effect=SchedulerBatchResultProcessor.snapshot_auxiliary_output_starts
    )
    processor = SimpleNamespace(
        move_logprobs_to_cpu=Mock(),
        consume_auxiliary_output=Mock(),
        snapshot_auxiliary_output_starts=snapshot_auxiliary_output_starts,
    )
    scheduler = SimpleNamespace(
        batch_result_processor=processor,
        spec_algorithm=SimpleNamespace(is_eagle=lambda: False),
        tree_cache=object(),
        disagg_prefill_inflight_queue=[],
        send_kv_chunk=Mock(),
        metrics_reporter=SimpleNamespace(report_prefill_stats=Mock()),
    )

    with patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req"):
        SchedulerDisaggregationPrefillMixin.process_batch_result_disagg_prefill(
            scheduler,
            batch,
            result,
        )

    assert req.output_ids == [7]
    snapshot_auxiliary_output_starts.assert_called_once_with(batch, result)
    processor.consume_auxiliary_output.assert_called_once_with(
        batch,
        host_output,
        [0],
    )


def test_logprob_only_reuses_preprocessing_without_observer_lifecycle():
    runner = object.__new__(ModelRunner)
    runner._preprocess_logits = Mock()
    runner.sampler = SimpleNamespace(compute_logprobs_only=Mock())
    logits_output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
    sampling_info = object()
    forward_batch = SimpleNamespace(
        sampling_info=sampling_info,
        top_logprobs_nums=None,
        token_ids_logprobs=[1],
    )

    ModelRunner.compute_logprobs_only(runner, logits_output, forward_batch)

    runner._preprocess_logits.assert_called_once_with(logits_output, sampling_info)
    runner.sampler.compute_logprobs_only.assert_called_once()


def test_logprob_only_clears_stale_output_before_early_return():
    runner = object.__new__(ModelRunner)
    runner.sampler = SimpleNamespace(compute_logprobs_only=Mock())
    logits_output = LogitsProcessorOutput(
        next_token_logits=None,
        auxiliary_device_output=DeviceOutput(torch.tensor([99.0])),
    )
    forward_batch = SimpleNamespace(token_ids_logprobs=None)

    ModelRunner.compute_logprobs_only(runner, logits_output, forward_batch)

    assert logits_output.auxiliary_device_output is None
    runner.sampler.compute_logprobs_only.assert_not_called()


def test_pipeline_parallel_auxiliary_output_round_trip():
    device_output = DeviceOutput(torch.tensor([1.0, 2.0]))
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            auxiliary_device_output=device_output,
        ),
        next_token_ids=torch.tensor([7]),
    )
    batch = SimpleNamespace(
        return_logprob=False,
        req_pool_indices=torch.tensor([3]),
        input_ids=torch.tensor([5]),
    )

    tensors = Scheduler._pp_prepare_tensor_dict(
        object.__new__(Scheduler), result, batch
    )
    observer = Observer()
    receiver = object.__new__(Scheduler)
    receiver.pp_group = SimpleNamespace(is_first_rank=True)
    receiver.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(sampling_observer=observer)
    )
    receiver.future_map = SimpleNamespace(stash=Mock())

    output_result = Scheduler._pp_prep_batch_result(
        receiver,
        batch,
        PPBatchMetadata(can_run_cuda_graph=True),
        PPProxyTensors(tensors),
    )

    assert set(observer.received_tensors) == {"values"}
    assert torch.equal(observer.received_tensors["values"], device_output.values)
    assert output_result.logits_output.auxiliary_device_output is not device_output
    assert torch.equal(output_result.auxiliary_host_output.values, device_output.values)
    assert all("sampling_observer_output" not in key for key in tensors)
    receiver.future_map.stash.assert_called_once()


def test_pipeline_parallel_auxiliary_output_stays_packed_before_first_rank():
    device_output = DeviceOutput(torch.tensor([1.0]))
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            auxiliary_device_output=device_output,
        ),
        next_token_ids=torch.tensor([7]),
    )
    batch = SimpleNamespace(
        return_logprob=False,
        req_pool_indices=torch.tensor([3]),
        input_ids=torch.tensor([5]),
    )
    tensors = Scheduler._pp_prepare_tensor_dict(
        object.__new__(Scheduler), result, batch
    )
    receiver = object.__new__(Scheduler)
    receiver.pp_group = SimpleNamespace(is_first_rank=False)
    receiver.future_map = SimpleNamespace(stash=Mock())

    output_result = Scheduler._pp_prep_batch_result(
        receiver,
        batch,
        PPBatchMetadata(can_run_cuda_graph=False),
        PPProxyTensors(tensors),
    )

    assert output_result.logits_output is None
    assert any("sampling_observer_output" in key for key in tensors)


def test_pipeline_parallel_auxiliary_output_requires_receiver_observer():
    device_output = DeviceOutput(torch.tensor([1.0]))
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            auxiliary_device_output=device_output,
        ),
        next_token_ids=torch.tensor([7]),
    )
    batch = SimpleNamespace(return_logprob=False)
    tensors = Scheduler._pp_prepare_tensor_dict(
        object.__new__(Scheduler), result, batch
    )
    receiver = object.__new__(Scheduler)
    receiver.pp_group = SimpleNamespace(is_first_rank=True)
    receiver.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(sampling_observer=None)
    )

    with pytest.raises(RuntimeError, match="without a sampling observer"):
        Scheduler._pp_prep_batch_result(
            receiver,
            batch,
            PPBatchMetadata(can_run_cuda_graph=False),
            PPProxyTensors(tensors),
        )


def test_pipeline_parallel_auxiliary_output_requires_transport_support():
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            auxiliary_device_output=HostOnlyDeviceOutput(torch.tensor([1.0])),
        ),
        next_token_ids=torch.tensor([7]),
    )
    batch = SimpleNamespace(return_logprob=False)

    with pytest.raises(RuntimeError, match="does not support pipeline-parallel"):
        Scheduler._pp_prepare_tensor_dict(object.__new__(Scheduler), result, batch)


def test_pipeline_parallel_auxiliary_output_requires_transport_observer():
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            auxiliary_device_output=DeviceOutput(torch.tensor([1.0])),
        ),
        next_token_ids=torch.tensor([7]),
    )
    batch = SimpleNamespace(return_logprob=False)
    tensors = Scheduler._pp_prepare_tensor_dict(
        object.__new__(Scheduler), result, batch
    )
    receiver = object.__new__(Scheduler)
    receiver.pp_group = SimpleNamespace(is_first_rank=True)
    receiver.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(sampling_observer=SimpleNamespace())
    )

    with pytest.raises(RuntimeError, match="does not support pipeline-parallel"):
        Scheduler._pp_prep_batch_result(
            receiver,
            batch,
            PPBatchMetadata(can_run_cuda_graph=False),
            PPProxyTensors(tensors),
        )


def test_auxiliary_output_snapshot_uses_visible_request_lengths():
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(output_ids=[10, 11, 12], finished_len=2),
            SimpleNamespace(output_ids=[20], finished_len=None),
        ]
    )
    result = SimpleNamespace(auxiliary_host_output=None)

    assert (
        SchedulerBatchResultProcessor.snapshot_auxiliary_output_starts(batch, result)
        is None
    )

    result.auxiliary_host_output = object()

    assert SchedulerBatchResultProcessor.snapshot_auxiliary_output_starts(
        batch, result
    ) == [2, 1]


def test_auxiliary_commit_uses_the_scheduler_visible_prefix():
    req = SimpleNamespace(output_ids=[10, 11, 12], finished_len=2)

    commits = SchedulerBatchResultProcessor._build_auxiliary_commits(
        SimpleNamespace(reqs=[req]),
        output_starts=[1],
    )

    assert commits[0].output_index == 1
    assert commits[0].token_ids == (11,)


def test_auxiliary_commit_discards_samples_outside_the_visible_output():
    req = SimpleNamespace(output_ids=[10], finished_len=None)

    commits = SchedulerBatchResultProcessor._build_auxiliary_commits(
        SimpleNamespace(reqs=[req]),
        output_starts=[1],
    )

    assert commits == [None]


def test_auxiliary_output_consumes_only_newly_visible_tokens():
    req = SimpleNamespace(output_ids=[10, 11, 12], finished_len=2)
    output = Mock()
    batch = SimpleNamespace(reqs=[req])

    SchedulerBatchResultProcessor.consume_auxiliary_output(
        batch,
        output,
        output_starts=[1],
    )

    commits = output.consume.call_args.args[1]
    assert commits[0].output_index == 1
    assert commits[0].token_ids == (11,)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
