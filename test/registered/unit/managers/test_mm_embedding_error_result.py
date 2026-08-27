from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_pp_mixin import (
    _pp_can_skip_output_comm,
    _pp_filter_failed_rows,
)
from sglang.srt.managers.utils import (
    complete_mm_embedding_validations,
    decode_mm_embedding_errors,
    merge_mm_embedding_error_tensors,
    synchronize_mm_embedding_errors,
)
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_beam_failure():
    req = Mock()
    req.finished_reason = None
    req.finished.side_effect = lambda: req.finished_reason is not None
    req.to_finish = FINISH_ABORT("mm failure", HTTPStatus.INTERNAL_SERVER_ERROR)
    req.mm_embedding_abort_pending = True
    req.multimodal_inputs = Mock()
    req.session = None
    req.req_pool_idx = 0
    req.mamba_pool_idx = None
    req.kv = SimpleNamespace(kv_allocated_len=4)
    req.kv_committed_len = 4
    req.time_stats = Mock()

    req_to_token_pool = Mock()
    req_to_token_pool.req_to_token = torch.arange(12).reshape(3, 4)
    allocator = Mock()
    coordinator = BeamCoordinator(
        model_config=None,
        spec_algorithm=None,
        dllm_enabled=False,
        max_req_len=0,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=None,
        future_map=None,
    )
    group = SimpleNamespace(
        leader=req,
        state=None,
        final_results=None,
        retired=False,
        prompt_len=2,
        member_rows=torch.tensor([1]),
        member_rows_cpu=torch.tensor([1]),
        all_rows=torch.tensor([0, 1]),
        pending_orphans=[],
        slots_freed=0,
        _pending_steps={},
    )
    req.beam_group = group
    coordinator._num_live_groups = 1
    return req, group, coordinator, req_to_token_pool, allocator


def _make_result_processor(tree_cache, beam_coordinator):
    processor = object.__new__(SchedulerBatchResultProcessor)
    object.__setattr__(processor, "tree_cache", tree_cache)
    object.__setattr__(processor, "draft_worker", None)
    object.__setattr__(processor, "beam_coordinator", beam_coordinator)
    return processor


def test_short_mm_embedding_aborts_with_http_500_and_cleans_up():
    processor = object.__new__(SchedulerBatchResultProcessor)
    object.__setattr__(processor, "tree_cache", Mock())
    object.__setattr__(processor, "draft_worker", None)
    object.__setattr__(processor, "beam_coordinator", Mock())
    failed = Mock()
    failed.finished_reason = None
    failed.finished.side_effect = lambda: failed.finished_reason is not None
    failed.session = None
    failed.mm_embedding_abort_pending = True
    failed.to_finish = FINISH_ABORT(
        "Insufficient multimodal embedding length: expected 6 tokens, got 2.",
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )

    def update_finish_state():
        failed.finished_reason = failed.to_finish
        failed.to_finish = None

    failed.update_finish_state.side_effect = update_finish_state

    with patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache"
    ) as release_kv_cache:
        processor._finish_deferred_mm_embedding_abort(failed)
        processor._finish_deferred_mm_embedding_abort(failed)

    assert isinstance(failed.finished_reason, FINISH_ABORT)
    assert failed.finished_reason.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "expected 6 tokens, got 2" in failed.finished_reason.message
    failed.update_finish_state.assert_called_once_with()
    failed.multimodal_inputs.release_features.assert_called_once_with()
    release_kv_cache.assert_called_once_with(
        failed, processor.tree_cache, is_insert=False
    )
    assert not failed.mm_embedding_abort_pending
    processor.tree_cache.release_aborted_request.assert_called_once_with(failed.rid)


def test_normal_mm_beam_failure_retires_group_and_releases_resources_once():
    req, group, coordinator, req_to_token_pool, allocator = _make_beam_failure()
    existing_reason = req.to_finish
    tree_cache = Mock()
    processor = _make_result_processor(tree_cache, coordinator)

    with patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache"
    ) as release_kv_cache:
        processor._finish_deferred_mm_embedding_abort(req)
        processor._finish_deferred_mm_embedding_abort(req)

    assert req.finished_reason is existing_reason
    assert group.retired
    assert coordinator._num_live_groups == 0
    allocator.free.assert_called_once()
    req_to_token_pool.free_rows.assert_called_once_with([1])
    tree_cache.release_aborted_request.assert_called_once_with(req.rid)
    release_kv_cache.assert_called_once_with(req, tree_cache, is_insert=False)


def test_scheduler_marks_failed_request_before_deferred_drain():
    req = Mock()
    req.finished.return_value = False
    req.mm_embedding_abort_pending = False
    req.to_finish = None
    req.inflight_middle_chunks = 1
    scheduler = SimpleNamespace(
        chunked_req=req,
        _pending_chunked_abort_req=None,
        disaggregation_mode=DisaggregationMode.NULL,
    )

    Scheduler.defer_mm_embedding_abort(scheduler, req, 6, 2)

    assert req.skip_radix_cache_insert
    assert req.mm_embedding_abort_pending
    assert req.to_finish.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert scheduler._pending_chunked_abort_req is req


def test_mm_failure_preserves_existing_finish_reason_while_fencing_cache():
    existing_reason = FINISH_ABORT("cancelled", 499)
    req = Mock()
    req.finished.return_value = True
    req.finished_reason = existing_reason
    req.to_finish = None
    req.mm_embedding_abort_pending = False
    req.inflight_middle_chunks = 0
    scheduler = SimpleNamespace(
        chunked_req=None,
        _pending_chunked_abort_req=None,
        disaggregation_mode=DisaggregationMode.NULL,
    )

    Scheduler.defer_mm_embedding_abort(scheduler, req, 6, 2)

    assert req.finished_reason is existing_reason
    assert req.to_finish is None
    assert req.skip_radix_cache_insert
    assert req.mm_embedding_abort_pending


def test_marked_failure_cannot_enter_unfinished_radix_or_hicache():
    req = SimpleNamespace(
        skip_radix_cache_insert=False, mm_embedding_validation_count=1
    )
    tree_cache = Mock()

    maybe_cache_unfinished_req(req, tree_cache, chunked=True)

    tree_cache.cache_unfinished_req.assert_not_called()


def test_two_overlapped_mm_chunks_keep_cache_fenced_until_both_complete():
    req = SimpleNamespace(
        skip_radix_cache_insert=False, mm_embedding_validation_count=2
    )
    tree_cache = Mock()
    success = torch.tensor([[1, 0, 0, 0]])

    complete_mm_embedding_validations([req], success)
    maybe_cache_unfinished_req(req, tree_cache, chunked=True)

    assert req.mm_embedding_validation_count == 1
    tree_cache.cache_unfinished_req.assert_not_called()

    complete_mm_embedding_validations([req], success)
    maybe_cache_unfinished_req(req, tree_cache, chunked=True)

    assert req.mm_embedding_validation_count == 0
    tree_cache.cache_unfinished_req.assert_called_once_with(req, chunked=True)


def test_pp_embedding_error_metadata_remains_forwardable():
    error_tensor = torch.tensor([[1, 1, 6, 2]], dtype=torch.int64)

    assert decode_mm_embedding_errors(error_tensor) == [(0, 6, 2)]


def test_pp_future_map_filter_excludes_failed_rows():
    indices = torch.tensor([10, 11])
    token_ids = torch.tensor([100, 101])
    errors = torch.tensor([[1, 0, 0, 0], [1, 1, 6, 2]])

    filtered_indices, filtered_tokens = _pp_filter_failed_rows(
        indices, token_ids, errors
    )

    assert filtered_indices.tolist() == [10]
    assert filtered_tokens.tolist() == [100]


def test_pp_multistage_merge_preserves_incoming_failure_details():
    incoming = torch.tensor([[1, 1, 6, 2], [1, 0, 0, 0]])
    downstream = torch.tensor([[1, 0, 0, 0], [1, 1, 8, 3]])

    merged = merge_mm_embedding_error_tensors(incoming, downstream)
    merged_again = merge_mm_embedding_error_tensors(merged, None)

    assert merged_again.tolist() == [[1, 1, 6, 2], [1, 1, 8, 3]]


def test_speculative_relay_selects_only_unaffected_rows():
    payload = RelayPayload(
        bonus_tokens=torch.tensor([10, 20]),
        hidden_states=torch.tensor([[1.0], [2.0]]),
    )
    keep = torch.tensor([True, False])

    selected = payload.select_rows(keep)

    assert selected.bonus_tokens.tolist() == [10]
    assert selected.hidden_states.tolist() == [[1.0]]


def test_all_zero_mm_validation_still_runs_beam_coordination():
    scheduler = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_ngram=lambda: False),
        future_map=Mock(),
        beam_coordinator=Mock(),
        chunked_req=None,
    )
    req = SimpleNamespace(beam_group=object())
    batch = SimpleNamespace(reqs=[req], beam_tail=None)
    result = SimpleNamespace(
        mm_embedding_errors=torch.tensor([[1, 0, 0, 0]]),
        next_draft_input=None,
        has_sampled_token_ids=True,
        next_token_ids=torch.tensor([10]),
    )

    Scheduler._relay_forward_payload(scheduler, batch, torch.tensor([4]), result)

    scheduler.future_map.stash.assert_called_once()
    scheduler.beam_coordinator.maybe_select_and_relay.assert_called_once_with(
        batch,
        result,
        chunked_req=None,
        skip_rows=set(),
    )


def test_pp_does_not_skip_output_for_chunk_overlapping_mm_placeholder():
    mm_input = SimpleNamespace(mm_items=[SimpleNamespace(offsets=[(6, 9)])])
    batch = SimpleNamespace(
        reqs=[SimpleNamespace()],
        multimodal_inputs=[mm_input],
        prefix_lens=[4],
        extend_lens=[4],
        forward_mode=ForwardMode.EXTEND,
        contains_last_prefill_chunk=False,
        return_logprob=False,
        mm_embedding_validation_indices=lambda: [0],
    )

    with envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.override(True):
        assert not _pp_can_skip_output_comm(batch)
        batch.mm_embedding_validation_indices = lambda: []
        assert _pp_can_skip_output_comm(batch)


def test_mm_embedding_errors_are_synchronized_as_batch_aligned_details(monkeypatch):
    group = SimpleNamespace(world_size=2, device_group=object())
    parallel = SimpleNamespace(attn_tp_group=group, attn_cp_group=group)
    reduce_ops = []

    def all_reduce(values, op, **_kwargs):
        reduce_ops.append(op)
        if values.ndim == 2:
            values[1] = torch.tensor([6, 2])
        elif op == torch.distributed.ReduceOp.MAX:
            values[1] = 1
        else:
            values[1] = 1

    monkeypatch.setattr("sglang.srt.managers.utils.get_parallel", lambda: parallel)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    synchronized = synchronize_mm_embedding_errors(None, 2, torch.device("cpu"), [1])

    assert synchronized.tolist() == [[0, 0, 0, 0], [1, 1, 6, 2]]
    assert decode_mm_embedding_errors(synchronized) == [(1, 6, 2)]
    assert reduce_ops == [
        torch.distributed.ReduceOp.MAX,
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.SUM,
    ]


def test_disagg_beam_failure_retires_group_and_releases_resources_once():
    req, group, coordinator, req_to_token_pool, allocator = _make_beam_failure()
    req.mm_embedding_abort_pending = False
    req.finished.return_value = False
    req.inflight_middle_chunks = 0
    req.pending_bootstrap = True
    other_req = Mock()
    tree_cache = Mock()
    processor = _make_result_processor(tree_cache, coordinator)
    scheduler = SimpleNamespace(
        clear_pending_chunk_send=Mock(),
        req_to_metadata_buffer_idx_allocator=Mock(),
        disagg_prefill_inflight_queue=[req, other_req],
        batch_result_processor=processor,
        disaggregation_mode=DisaggregationMode.PREFILL,
        chunked_req=None,
        _pending_chunked_abort_req=None,
    )

    with (
        patch(
            "sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer"
        ) as release_metadata,
        patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache"
        ) as release_kv_cache,
    ):
        Scheduler.defer_mm_embedding_abort(scheduler, req, 6, 2)
        SchedulerDisaggregationPrefillMixin.finish_disagg_mm_embedding_abort(
            scheduler, req
        )

    scheduler.clear_pending_chunk_send.assert_called_once_with(req)
    req.disagg_kv_sender.abort.assert_called_once_with()
    req.disagg_kv_sender.clear.assert_called_once_with()
    release_metadata.assert_called_once_with(
        req, scheduler.req_to_metadata_buffer_idx_allocator
    )
    assert not req.pending_bootstrap
    assert scheduler.disagg_prefill_inflight_queue == [other_req]
    assert group.retired
    assert coordinator._num_live_groups == 0
    allocator.free.assert_called_once()
    req_to_token_pool.free_rows.assert_called_once_with([1])
    tree_cache.release_aborted_request.assert_called_once_with(req.rid)
    release_kv_cache.assert_called_once_with(req, tree_cache, is_insert=False)
