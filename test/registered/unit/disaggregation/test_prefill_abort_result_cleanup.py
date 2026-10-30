from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _Req:
    def __init__(self, *, inflight_middle_chunks: int, allocated: bool = True):
        self.rid = "aborted-prefill"
        self.cache_request_handle = CacheRequestHandle(self.rid, 0)
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
        self.discard_output_reason = None
        self.return_logprob = False
        self.return_sampling_mask = False
        self.grammar = None
        self.output_ids = []
        self.origin_input_ids = list(range(100))
        self.extend_end = None
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


def _free_req(req, _tree_cache, *, checkpoint):
    assert checkpoint is False
    req.kv.req_pool_idx = None
    req.kv.mark_kv_released()
    req.kv.mamba_pool_idx = None


class TestPrefillCompleteResult(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="prefill_complete"
        )
        override.install()
        self.addCleanup(override.restore)

    @patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
    def test_ready_observes_accepted_token_and_speculative_sidecars(self, cache):
        scheduler = _Scheduler()
        scheduler.spec_algorithm = SimpleNamespace(is_eagle=lambda: True)
        req = _Req(inflight_middle_chunks=0)
        req.to_finish = None
        req.time_stats.set_prefill_transfer_queue_entry_time = Mock()
        draft = SimpleNamespace(
            hidden_states=torch.tensor([[1.0, 2.0]]),
            topk_p=torch.tensor([[0.75, 0.25]]),
            topk_index=torch.tensor([[3, 4]]),
            dsa_topk_indices=torch.tensor([[5, 6]]),
        )
        batch = _batch(req)
        batch.spec_info = draft
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([11]), next_draft_input=draft
        )
        observed = []

        def ready():
            observed.append(
                (
                    list(req.output_ids),
                    req.hidden_states_tensor.tolist(),
                    req.output_dsa_topk_indices.tolist(),
                    req in scheduler.disagg_prefill_inflight_queue,
                )
            )

        req.disagg_kv_sender.mark_prefill_complete.side_effect = ready
        scheduler.process_batch_result_disagg_prefill(batch, result)
        self.assertEqual(observed, [([11], [1.0, 2.0], [5, 6], True)])
        self.assertTrue(req.pending_bootstrap)
        scheduler.send_kv_chunk.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_aborted_final_result_never_publishes_ready(self, release):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        scheduler.process_batch_result_disagg_prefill(_batch(req), _result())
        self.assertTrue(req.finished())
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        req.disagg_kv_sender.mark_prefill_complete.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.should_force_retry", return_value=True)
    def test_discarded_attempt_never_publishes_ready(self, force_retry):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        req.to_finish = None
        retried = []
        scheduler.optimistic_release_and_requeue = lambda r: retried.append(r)
        scheduler.process_batch_result_disagg_prefill(_batch(req), _result())
        self.assertEqual(retried, [req])
        self.assertEqual(req.output_ids, [])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        req.disagg_kv_sender.mark_prefill_complete.assert_not_called()


class _EarlySendScheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self, device_module):
        self.device_module = device_module
        self.enable_staging = False
        self.enable_overlap = True
        self.forward_stream = object()
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=4)
        self._batched_page_indices = Mock(return_value=None)
        self.send_kv_chunk = Mock()


def _early_send_req(prefix_len, start_send_idx=0):
    return SimpleNamespace(
        pending_bootstrap=False,
        prefix_len=prefix_len,
        host_hit_length=0,
        early_send_prefix_end=None,
        start_send_idx=start_send_idx,
        disagg_kv_sender=SimpleNamespace(_early_send_wait_event=None),
    )


def test_early_send_wait_event_comes_from_the_scheduler_device_module():
    """One early-send event, created on the scheduler's device, covers the batch.

    forward_stream is built from that same device, and an event of one accelerator
    cannot record against another's stream. Resolving the module from the process
    default instead reinstates that mismatch whenever --device names something
    other than the default accelerator. Requests with nothing to early-send
    (already sent, or a prefix ending inside a page) get neither event nor send.
    """
    recorded = []

    class _Event:
        def record(self, stream):
            recorded.append(stream)

    scheduler = _EarlySendScheduler(SimpleNamespace(Event=_Event))
    reqs = [
        _early_send_req(8),
        _early_send_req(8, start_send_idx=8),
        _early_send_req(6),
        _early_send_req(16, start_send_idx=4),
    ]

    with envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(True):
        scheduler.send_cached_prefix_chunks(reqs)

    assert recorded == [scheduler.forward_stream]
    event = reqs[0].disagg_kv_sender._early_send_wait_event
    assert isinstance(event, _Event)
    assert reqs[3].disagg_kv_sender._early_send_wait_event is event
    assert reqs[1].disagg_kv_sender._early_send_wait_event is None
    assert reqs[2].disagg_kv_sender._early_send_wait_event is None
    scheduler._batched_page_indices.assert_called_once_with([reqs[0], reqs[3]], [8, 16])
    assert [c.args[0] for c in scheduler.send_kv_chunk.call_args_list] == [
        reqs[0],
        reqs[3],
    ]
    assert [c.kwargs["end_idx"] for c in scheduler.send_kv_chunk.call_args_list] == [
        8,
        16,
    ]


def test_early_send_skips_a_batch_with_a_pending_load_back():
    scheduler = _EarlySendScheduler(SimpleNamespace(Event=Mock()))
    with envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(True):
        scheduler.send_cached_prefix_chunks([_early_send_req(8)], allow_send=False)
    scheduler.device_module.Event.assert_not_called()
    scheduler.send_kv_chunk.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
