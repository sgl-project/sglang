from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.layers.logits_processor import LogitsProcessorOutput, SamplingMaskStatus
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Req:
    def __init__(self, *, inflight_middle_chunks: int, allocated: bool = True):
        self.rid = "aborted-prefill"
        self.inflight_middle_chunks = inflight_middle_chunks
        self.kv = ReqKvInfo(
            req_pool_idx=1 if allocated else None,
            kv_allocated_len=1 if allocated else 0,
            mamba_pool_idx=object() if allocated else None,
        )
        self.metadata_buffer_index = 7 if allocated else -1
        self.pending_bootstrap = allocated
        self.disagg_kv_sender = Mock()
        self.to_finish = FINISH_ABORT() if allocated else None
        self.finished_reason = None if allocated else FINISH_ABORT()
        self.return_logprob = False
        self.return_sampling_mask = False
        self.grammar = None
        self.output_ids = []
        self.origin_input_ids = list(range(100))
        self.extend_range = None
        self.time_stats = SimpleNamespace(
            set_prefill_finished_time=Mock(),
            set_last_chunked_prefill_finish_time=Mock(),
            set_completion_time=Mock(),
        )

    def finished(self):
        return self.finished_reason is not None

    def update_finish_state(self):
        self.finished_reason = self.to_finish
        self.to_finish = None


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self):
        self.batch_result_processor = SimpleNamespace(
            snapshot_auxiliary_output_starts=Mock(return_value=[]),
            move_logprobs_to_cpu=Mock(),
            consume_auxiliary_output=Mock(),
        )
        self.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        self.tree_cache = Mock()
        self.disagg_prefill_inflight_queue = []
        self.disagg_prefill_pending_chunk_rids = {"aborted-prefill"}
        self.send_kv_chunk = Mock()
        self.output_streamer = Mock()
        self.metrics_reporter = SimpleNamespace(report_prefill_stats=Mock())
        self.maybe_send_health_check_signal = Mock()
        self.req_to_metadata_buffer_idx_allocator = Mock()
        self.enable_hicache_storage = True
        self.chunked_req = None


def _batch(req):
    return SimpleNamespace(
        reqs=[req],
        spec_info=None,
        prefill_stats=None,
        dp_cooperation_info=None,
    )


def _result():
    return GenerationBatchResult(next_token_ids=torch.tensor([11]))


def _free_req(req, _tree_cache, *, is_insert):
    assert is_insert is False
    req.kv.req_pool_idx = None
    req.kv.mark_kv_released()
    req.kv.mamba_pool_idx = None


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
@patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req")
def test_aborted_final_result_releases_hybrid_cache(
    maybe_cache_unfinished_req, release_kv_cache
):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=0)

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    release_kv_cache.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
    maybe_cache_unfinished_req.assert_not_called()
    req.disagg_kv_sender.abort.assert_called_once_with()
    scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
    scheduler.tree_cache.release_aborted_request.assert_called_once_with(req.rid)
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
    scheduler.send_kv_chunk.assert_not_called()
    assert req.output_ids == []
    assert req.finished()
    assert req.metadata_buffer_index == -1
    assert req.rid not in scheduler.disagg_prefill_pending_chunk_rids


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_aborted_middle_result_releases_after_last_chunk(release_kv_cache):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=1)
    req.extend_range = SimpleNamespace(end=50)

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    assert req.inflight_middle_chunks == 0
    release_kv_cache.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_aborted_middle_result_waits_for_inflight_chunk(release_kv_cache):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=1)
    req.extend_range = SimpleNamespace(end=len(req.origin_input_ids))

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    release_kv_cache.assert_not_called()
    scheduler.output_streamer.stream_output.assert_not_called()

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    release_kv_cache.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


@patch("sglang.srt.disaggregation.prefill.release_kv_cache")
def test_delayed_result_ignores_already_retired_request(release_kv_cache):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=0, allocated=False)

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    release_kv_cache.assert_not_called()
    req.disagg_kv_sender.abort.assert_not_called()
    scheduler.output_streamer.stream_output.assert_not_called()


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_sender_abort_failure_does_not_skip_local_cleanup(release_kv_cache):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=0)
    req.disagg_kv_sender.abort.side_effect = RuntimeError("transport is down")

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    release_kv_cache.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
    scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
    assert req.finished()


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
@patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req")
def test_grammar_rejection_retires_prefill_before_transfer(
    maybe_cache_unfinished_req, release_kv_cache
):
    scheduler = _Scheduler()
    req = _Req(inflight_middle_chunks=0)
    req.to_finish = None
    req.grammar = Mock()
    req.grammar.accept_token.side_effect = ValueError("invalid token")

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    req.grammar.accept_token.assert_called_once_with(11)
    assert req.grammar.finished
    assert req.finished()
    release_kv_cache.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
    maybe_cache_unfinished_req.assert_not_called()
    scheduler.send_kv_chunk.assert_not_called()
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
    assert scheduler.disagg_prefill_inflight_queue == []


def test_aborted_result_releases_mamba_allocated_before_kv():
    scheduler = _Scheduler()
    scheduler.enable_hicache_storage = False
    scheduler.tree_cache.supports_mamba.return_value = True
    scheduler.tree_cache.req_to_token_pool.mamba_allocator.free = Mock()
    req = _Req(inflight_middle_chunks=0, allocated=False)
    req.kv.mamba_pool_idx = torch.tensor([3])
    req.to_finish = FINISH_ABORT()
    req.finished_reason = None

    scheduler.process_batch_result_disagg_prefill(_batch(req), _result())

    scheduler.tree_cache.req_to_token_pool.mamba_allocator.free.assert_called_once()
    assert req.kv.mamba_pool_idx is None
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


@pytest.mark.parametrize("transport_error", [False, True])
@pytest.mark.parametrize(
    "status,http_status,err_type",
    [
        (SamplingMaskStatus.OVERFLOW, 400, "BadRequestError"),
        (SamplingMaskStatus.INVALID, 500, "InternalServerError"),
    ],
)
@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_sampling_mask_abort_preserves_error_and_releases_once(
    release_kv_cache, status, http_status, err_type, transport_error
):
    """A failed sender notification must not leak ownership or lose the API error."""
    scheduler = _Scheduler()
    scheduler.batch_result_processor.get_sampling_mask_finish_reason = (
        lambda **kwargs: SchedulerBatchResultProcessor.get_sampling_mask_finish_reason(
            None, **kwargs
        )
    )
    req = _Req(inflight_middle_chunks=0)
    req.to_finish = None
    req.return_sampling_mask = True
    req.time_stats.trace_ctx = Mock()
    if transport_error:
        req.disagg_kv_sender.abort.side_effect = RuntimeError("transport is down")
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([11]),
        logits_output=LogitsProcessorOutput(
            next_token_logits=None, next_token_sampling_mask_status=[status]
        ),
    )

    with get_context().override_server_args(sampling_mask_max_tokens=64):
        scheduler.process_batch_result_disagg_prefill(_batch(req), result)
        scheduler.process_batch_result_disagg_prefill(_batch(req), result)

    assert req.finished_reason.status_code == http_status
    assert req.finished_reason.err_type == err_type
    assert req.output_ids == []
    assert not req.kv.holds_kv and not req.kv.holds_mamba
    assert req.metadata_buffer_index == -1
    assert not req.pending_bootstrap
    assert req.rid not in scheduler.disagg_prefill_pending_chunk_rids
    release_kv_cache.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
    req.disagg_kv_sender.abort.assert_called_once_with()
    scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
    scheduler.tree_cache.release_aborted_request.assert_called_once_with(req.rid)
    scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
    scheduler.send_kv_chunk.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
