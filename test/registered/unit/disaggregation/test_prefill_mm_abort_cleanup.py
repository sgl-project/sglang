from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.disaggregation import prefill
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu-intel")


@pytest.mark.parametrize("poll", [KVPoll.Success, KVPoll.Failed])
def test_pending_mm_abort_cleanup_waits_for_global_transport_quiescence(poll):
    error = FINISH_ABORT(message="Multimodal embedding count mismatch")
    sender = SimpleNamespace(
        is_transfer_quiesced=Mock(return_value=True),
        clear=Mock(),
        get_transfer_metric=Mock(return_value=None),
    )
    req = SimpleNamespace(
        rid="request",
        disagg_abort_cleanup_pending=True,
        disagg_kv_sender=sender,
        kv=SimpleNamespace(holds_kv=True, holds_mamba=False),
        metadata_buffer_index=-1,
        pending_bootstrap=False,
        mm_embedding_abort_pending=True,
        finished_reason=error,
        to_finish=None,
        return_logprob=False,
        bootstrap_host=None,
        time_stats=SimpleNamespace(
            set_completion_time=Mock(),
            set_prefill_kv_transfer_finish_time=Mock(),
        ),
    )
    releases = []

    def finish_deferred(aborted_req):
        releases.append(aborted_req)
        aborted_req.kv.holds_kv = False
        aborted_req.mm_embedding_abort_pending = False

    scheduler = SimpleNamespace(
        disagg_prefill_inflight_queue=[req],
        attn_cp_cpu_group=None,
        attn_tp_cpu_group=None,
        output_streamer=SimpleNamespace(stream_output=Mock()),
        req_to_metadata_buffer_idx_allocator=Mock(),
        batch_result_processor=SimpleNamespace(
            _finish_deferred_mm_embedding_abort=Mock(side_effect=finish_deferred)
        ),
        enable_hicache_storage=False,
        tree_cache=Mock(),
        metrics_reporter=SimpleNamespace(enable_metrics=False),
        handle_pending_bootstrap=Mock(),
        handle_inflight_transfer_failure=Mock(
            side_effect=lambda failed_req: releases.append(failed_req)
        ),
        _retire_aborted_prefill_result=None,
    )
    scheduler._retire_aborted_prefill_result = MethodType(
        SchedulerDisaggregationPrefillMixin._retire_aborted_prefill_result,
        scheduler,
    )

    with (
        patch.object(
            prefill,
            "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[poll],
        ),
        patch.object(
            prefill,
            "all_reduce_transfer_quiesced_attn_cp_tp_group",
            side_effect=[[False], [True]],
            create=True,
        ),
        patch.object(
            prefill,
            "release_kv_cache",
            side_effect=lambda *_a, **_k: releases.append(req),
        ),
    ):
        first_done = (
            SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler
            )
        )
        assert first_done == []
        assert scheduler.disagg_prefill_inflight_queue == [req]
        assert req.disagg_abort_cleanup_pending
        assert req.finished_reason is error
        assert releases == []
        sender.clear.assert_not_called()

        second_done = (
            SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler
            )
        )

    assert second_done == [req]
    assert scheduler.disagg_prefill_inflight_queue == []
    assert not req.disagg_abort_cleanup_pending
    assert req.finished_reason is error
    assert releases == [req]
    sender.clear.assert_called_once_with()
