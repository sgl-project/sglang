from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    set_schedule_time_batch,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_custom_decode_first_step_completes_prefill_timing_once() -> None:
    time_stats = SimpleNamespace(
        set_prefill_finished_time=Mock(),
        set_last_decode_finish_time=Mock(),
    )
    request = SimpleNamespace(
        custom_decode_needs_prefill_schedule=True,
        custom_decode_needs_prefill_completion=True,
        time_stats=time_stats,
    )

    SchedulerBatchResultProcessor._record_decode_step_finish_time(request)

    assert not request.custom_decode_needs_prefill_schedule
    assert not request.custom_decode_needs_prefill_completion
    time_stats.set_prefill_finished_time.assert_called_once_with()
    time_stats.set_last_decode_finish_time.assert_not_called()

    SchedulerBatchResultProcessor._record_decode_step_finish_time(request)

    time_stats.set_prefill_finished_time.assert_called_once_with()
    time_stats.set_last_decode_finish_time.assert_called_once_with()


def test_custom_decode_first_schedule_uses_prefill_timing_once() -> None:
    request = SimpleNamespace(
        custom_decode_needs_prefill_schedule=True,
        time_stats=SimpleNamespace(set_last_scheduled_time=Mock()),
    )
    batch = SimpleNamespace(reqs=[request], forward_mode=ForwardMode.DECODE)

    with patch(
        "sglang.srt.observability.req_time_stats.get_global_tracing_enabled",
        return_value=True,
    ):
        set_schedule_time_batch(batch)
        set_schedule_time_batch(batch)

    first_call, second_call = request.time_stats.set_last_scheduled_time.call_args_list
    assert first_call.args[0] is ForwardMode.EXTEND
    assert first_call.args[2]["forward_mode"] == "prefill"
    assert second_call.args[0] is ForwardMode.DECODE
    assert second_call.args[2]["forward_mode"] == "decode"
    assert not request.custom_decode_needs_prefill_schedule


def test_pending_custom_decode_blocks_later_prefill_across_iterations() -> None:
    time_stats = SimpleNamespace(
        set_prefill_finished_time=Mock(),
        set_last_decode_finish_time=Mock(),
    )
    custom_request = SimpleNamespace(
        custom_decode_needs_prefill_schedule=False,
        custom_decode_needs_prefill_completion=True,
        time_stats=time_stats,
    )
    running_batch = SimpleNamespace(
        reqs=[custom_request],
        is_prefill_only=False,
        is_empty=lambda: False,
    )
    penalized_request = SimpleNamespace(
        sampling_params=SimpleNamespace(frequency_penalty=0.5)
    )

    scheduler = object.__new__(Scheduler)
    scheduler.enable_fpm = False
    scheduler.dllm_config = None
    scheduler.chunked_req = None
    scheduler.enable_hisparse = False
    scheduler.require_mlp_sync = False
    scheduler.waiting_queue = [penalized_request]
    scheduler.process_pending_chunked_abort = Mock()
    scheduler._abort_on_waiting_timeout = Mock()
    scheduler._abort_on_running_timeout = Mock()
    scheduler.build_custom_decode_admission = Mock(return_value=None)
    scheduler.update_running_batch = Mock(return_value=running_batch)
    scheduler.get_new_batch_prefill = Mock(
        return_value=SimpleNamespace(
            batch_to_run=None,
            running_batch=running_batch,
        )
    )
    scheduler.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda batch, need_sync: batch
    )
    scheduler.ngram_embedding_manager = SimpleNamespace(
        prepare_for_forward=lambda batch, chunked_req: batch
    )

    with patch("sglang.srt.managers.scheduler.set_schedule_time_batch"):
        for _ in range(2):
            plan = scheduler.get_next_batch_to_run(running_batch, last_batch=None)
            assert plan.batch_to_run is running_batch

        scheduler.get_new_batch_prefill.assert_not_called()

        SchedulerBatchResultProcessor._record_decode_step_finish_time(custom_request)
        scheduler.get_next_batch_to_run(running_batch, last_batch=None)

    scheduler.get_new_batch_prefill.assert_called_once_with(running_batch)


def test_batch_merge_preserves_direct_decode_timing_contract() -> None:
    running_request = SimpleNamespace(
        custom_decode_needs_prefill_completion=False,
        return_logprob=False,
        return_hidden_states=False,
        grammar=None,
    )
    custom_request = SimpleNamespace(
        custom_decode_needs_prefill_schedule=True,
        custom_decode_needs_prefill_completion=True,
        return_logprob=False,
        return_hidden_states=False,
        grammar=None,
        time_stats=SimpleNamespace(
            set_prefill_finished_time=Mock(),
            set_last_decode_finish_time=Mock(),
        ),
    )

    def make_batch(request, pool_index: int) -> ScheduleBatch:
        return ScheduleBatch(
            reqs=[request],
            sampling_info=SimpleNamespace(merge_batch=Mock()),
            model_config=SimpleNamespace(is_encoder_decoder=False),
            req_pool_indices=torch.tensor([pool_index]),
            req_pool_indices_cpu=torch.tensor([pool_index]),
            seq_lens=torch.tensor([1]),
            orig_seq_lens=torch.tensor([1], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1]),
            input_ids=None,
            multimodal_inputs=[None],
            return_logprob=False,
            has_grammar=False,
            return_hidden_states=False,
            is_prefill_only=False,
            spec_info=None,
        )

    running_batch = make_batch(running_request, 0)
    custom_batch = make_batch(custom_request, 1)

    running_batch.merge_batch(custom_batch)

    merged_request = running_batch.reqs[1]
    assert merged_request.custom_decode_needs_prefill_schedule
    SchedulerBatchResultProcessor._record_decode_step_finish_time(merged_request)
    assert not merged_request.custom_decode_needs_prefill_schedule
    assert not merged_request.custom_decode_needs_prefill_completion
