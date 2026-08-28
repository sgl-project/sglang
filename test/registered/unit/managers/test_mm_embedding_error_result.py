import inspect
from http import HTTPStatus
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
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
    SchedulerPPMixin,
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


def test_short_mm_embedding_failure_cannot_enter_unfinished_radix_or_hicache():
    req = Mock(
        skip_radix_cache_insert=False,
        mm_embedding_validation_count=1,
        mm_embedding_abort_pending=False,
        inflight_middle_chunks=1,
        to_finish=None,
    )
    req.finished.return_value = False
    scheduler = SimpleNamespace(
        chunked_req=req,
        _pending_chunked_abort_req=None,
        disaggregation_mode=DisaggregationMode.NULL,
    )
    tree_cache = Mock()

    Scheduler.defer_mm_embedding_abort(scheduler, req, 6, 2)
    maybe_cache_unfinished_req(req, tree_cache, chunked=True)

    assert req.skip_radix_cache_insert
    assert "expected 6 tokens, got 2" in req.to_finish.message
    tree_cache.cache_unfinished_req.assert_not_called()


def test_failed_validation_while_parked_restores_pool_without_kv_send():
    req = Mock()
    req.mm_embedding_validation_count = 1
    req.mm_embedding_abort_pending = False
    req.inflight_middle_chunks = 0
    req.finished.return_value = False
    req.to_finish = None
    req.time_stats = Mock()
    pool = SimpleNamespace(available=9, baseline=10)
    scheduler = SimpleNamespace(
        chunked_req=req,
        _pending_chunked_abort_req=req,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_hicache_storage=False,
        tree_cache=Mock(),
        ipc_channels=SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        ),
        check_bootstrap=Mock(),
        send_kv_chunk=Mock(),
    )
    Scheduler.defer_mm_embedding_abort(scheduler, req, 6, 2)

    def free_once(*_args, **_kwargs):
        pool.available += 1

    with (
        patch("sglang.srt.managers.scheduler.prepare_abort"),
        patch(
            "sglang.srt.managers.scheduler.release_kv_cache", side_effect=free_once
        ) as release_kv_cache,
        patch("sglang.srt.managers.scheduler._make_abort_req", return_value=Mock()),
    ):
        Scheduler.process_pending_chunked_abort(scheduler)
        assert scheduler.chunked_req is None
        assert pool.available == 9

        complete_mm_embedding_validations([req], torch.tensor([[1, 0, 0, 0]]))
        Scheduler.process_pending_chunked_abort(scheduler)
        Scheduler.process_pending_chunked_abort(scheduler)
        SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
            scheduler, None, SimpleNamespace(batch_is_full=True)
        )

    assert pool.available == pool.baseline
    release_kv_cache.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
    scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()
    scheduler.check_bootstrap.assert_not_called()
    scheduler.send_kv_chunk.assert_not_called()


def test_disagg_prefill_gates_kv_send_but_not_bootstrap_on_validation():
    req = Mock()
    req.mm_embedding_validation_count = 1
    req.extend_range = SimpleNamespace(end=8)
    req.origin_input_ids = list(range(16))
    req.tmp_end_idx = 2
    scheduler = SimpleNamespace(
        chunked_req=req,
        tree_cache=Mock(),
        check_bootstrap=Mock(return_value=True),
        enable_overlap=False,
        send_kv_chunk=Mock(),
    )
    running_batch = SimpleNamespace(batch_is_full=True)
    last_batch = Mock()
    last_batch.forward_mode.is_extend.return_value = True
    last_batch.chunked_req = req
    last_batch.batch_size.side_effect = [2, 1]

    with patch(
        "sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req"
    ) as cache_unfinished:
        SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
            scheduler, last_batch, running_batch
        )
        for _ in range(2):
            SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
                scheduler, None, running_batch
            )

        assert scheduler.chunked_req is req
        assert not running_batch.batch_is_full
        assert req.tmp_end_idx == 2
        assert scheduler.check_bootstrap.call_count == 3
        scheduler.send_kv_chunk.assert_not_called()
        assert cache_unfinished.call_count == 3
        cache_unfinished.assert_called_with(req, scheduler.tree_cache, chunked=True)
        last_batch.filter_batch.assert_called_once_with(chunked_req_to_exclude=[req])

        complete_mm_embedding_validations([req], torch.tensor([[1, 0, 0, 0]]))
        SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
            scheduler, None, running_batch
        )

    assert scheduler.check_bootstrap.call_count == 4
    scheduler.check_bootstrap.assert_called_with(req)
    scheduler.send_kv_chunk.assert_called_once_with(req)


def test_overlap_pending_validation_keeps_boundary_and_yield_bookkeeping():
    req = Mock()
    req.mm_embedding_validation_count = 1
    req.extend_range = SimpleNamespace(end=8)
    req.origin_input_ids = list(range(16))
    req.tmp_end_idx = 2
    req.to_finish = None
    req.finished_reason = None
    scheduler = SimpleNamespace(
        chunked_req=req,
        tree_cache=Mock(),
        check_bootstrap=Mock(return_value=True),
        enable_overlap=True,
        has_bootstrapped_waiting_req=Mock(return_value=False),
        optimistic_release_and_requeue=Mock(),
        send_kv_chunk=Mock(),
    )
    running_batch = SimpleNamespace(batch_is_full=True)

    with patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req"):
        SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
            scheduler, None, running_batch
        )

        assert req.tmp_end_idx == 8
        assert scheduler.chunked_req is req
        assert not running_batch.batch_is_full
        scheduler.check_bootstrap.assert_called_once_with(req)
        scheduler.send_kv_chunk.assert_not_called()

        scheduler.check_bootstrap.return_value = False
        scheduler.has_bootstrapped_waiting_req.return_value = True
        SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
            scheduler, None, running_batch
        )

    assert scheduler.chunked_req is None
    scheduler.has_bootstrapped_waiting_req.assert_called_once_with()
    scheduler.optimistic_release_and_requeue.assert_not_called()
    scheduler.send_kv_chunk.assert_not_called()


def test_run_batch_registers_validation_before_optional_early_send():
    validation_req = Mock()
    validation_req.mm_embedding_validation_count = 0
    validation_req.start_send_idx = 0
    validation_req.extend_range = SimpleNamespace(end=8)
    validation_req.origin_input_ids = list(range(16))
    plain_req = SimpleNamespace(
        mm_embedding_validation_count=0,
        pending_bootstrap=False,
        prefix_indices=[0, 1, 2, 3],
        host_hit_length=0,
        start_send_idx=0,
    )
    send_kv_chunk = Mock()
    scheduler = SimpleNamespace(
        forward_ct=0,
        _sched_idled=False,
        scripted_scheduler_hook=None,
        profiler_manager=Mock(),
        forward_sleep_time=None,
        disaggregation_mode=DisaggregationMode.PREFILL,
        enable_staging=False,
        enable_overlap=False,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        send_kv_chunk=send_kv_chunk,
        is_generation=True,
        enable_pdmux=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        future_map=Mock(),
    )
    scheduler.maybe_send_cached_prefix_chunk = MethodType(
        SchedulerDisaggregationPrefillMixin.maybe_send_cached_prefix_chunk,
        scheduler,
    )

    def stop_at_forward(_batch, **_kwargs):
        assert validation_req.mm_embedding_validation_count == 1
        assert plain_req.mm_embedding_validation_count == 0
        send_kv_chunk.assert_called_once_with(plain_req, last_chunk=False, end_idx=4)
        raise RuntimeError("stop at model forward")

    scheduler.model_worker = SimpleNamespace(
        forward_batch_generation=Mock(side_effect=stop_at_forward)
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_prebuilt=lambda: False,
            is_extend=lambda: True,
            is_split_prefill=lambda: False,
        ),
        reqs=[validation_req, plain_req],
        mm_embedding_validation_indices=lambda: [0],
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )

    with (
        envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(True),
        patch("sglang.srt.managers.scheduler.resolve_forward_inputs"),
        pytest.raises(RuntimeError, match="stop at model forward"),
    ):
        inspect.unwrap(Scheduler.run_batch)(scheduler, batch)

    assert validation_req.mm_embedding_validation_count == 1
    assert validation_req.start_send_idx == 0
    scheduler.model_worker.forward_batch_generation.assert_called_once()

    complete_mm_embedding_validations([validation_req], torch.tensor([[1, 0, 0, 0]]))
    send_kv_chunk.reset_mock()
    scheduler.chunked_req = validation_req
    scheduler.tree_cache = Mock()
    scheduler.check_bootstrap = Mock(return_value=True)
    running_batch = SimpleNamespace(batch_is_full=True)
    with patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req"):
        SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
            scheduler, None, running_batch
        )

    send_kv_chunk.assert_called_once_with(validation_req)


def test_pp_disagg_prefill_drains_pending_abort_before_chunk_processing():
    order = []

    def stop_after_chunk(**_kwargs):
        order.append("chunk")
        raise RuntimeError("stop loop")

    scheduler = SimpleNamespace(
        init_pp_loop_state=Mock(),
        pp_loop_size=1,
        running_mbs=[SimpleNamespace()],
        last_mbs=[None],
        ps=SimpleNamespace(pp_size=1),
        request_receiver=SimpleNamespace(recv_requests=Mock(return_value=[])),
        process_input_requests=Mock(),
        pp_group=SimpleNamespace(is_last_rank=True),
        _pp_pd_get_bootstrapped_ids=Mock(return_value=[]),
        _pp_commit_comm_work=Mock(),
        _pp_pd_get_prefill_transferred_ids=Mock(return_value=[]),
        process_pending_chunked_abort=Mock(side_effect=lambda: order.append("abort")),
        process_prefill_chunk=Mock(side_effect=stop_after_chunk),
    )

    with pytest.raises(RuntimeError, match="stop loop"):
        inspect.unwrap(SchedulerPPMixin.event_loop_pp_disagg_prefill)(scheduler)

    assert order == ["abort", "chunk"]


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
