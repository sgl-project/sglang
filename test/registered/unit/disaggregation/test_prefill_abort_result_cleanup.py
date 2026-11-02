from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ReqKvInfo
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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
            _maybe_collect_customized_info=Mock(),
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


@pytest.mark.parametrize(
    "capacity,customized_info,error",
    [
        (4096, {1: ["x" * 8192]}, "exceeds"),
        (4096, {"scores": [lambda: None]}, "Invalid customized_info"),
        (4096, {1: [0.5]}, "Expected `str`"),
        (4096, {"scores": 1}, "Expected `array`"),
        (0, {"scores": [0.5]}, "SGLANG_DISAGG_CUSTOMIZED_INFO_MAX_BYTES"),
    ],
    ids=["overflow", "encoding", "non-string-key", "non-list-value", "disabled"],
)
def test_customized_info_failure_is_request_local(capacity, customized_info, error):
    scheduler = SchedulerDisaggregationPrefillMixin()
    scheduler.disagg_prefill_pending_chunk_rids = set()
    scheduler.enable_staging = False
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=1, translate_kv_indices_for_transfer=lambda indices: indices
    )
    scheduler.req_to_token_pool = SimpleNamespace(req_to_token=torch.tensor([[0]]))
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
        kv_manager=SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
    )
    with envs.SGLANG_DISAGG_CUSTOMIZED_INFO_MAX_BYTES.override(capacity):
        scheduler.disagg_metadata_buffers = MetadataBuffers(
            size=1,
            hidden_size=1,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=1,
        )
    valid_info = {"scores": [0.5]} if capacity else None
    for index, payload in enumerate((customized_info, valid_info)):
        req = Req(f"customized-info-{index}", "", array("q", [1]), SamplingParams())
        req.kv = ReqKvInfo(req_pool_idx=0, kv_allocated_len=1)
        req.metadata_buffer_index = 0
        req.extend_end = 1
        req.output_ids.append(2)
        req.customized_info = payload
        req.disagg_kv_sender = Mock()
        req.disagg_kv_sender.supports_host_tensor_indices = False
        req.disagg_kv_sender.get_max_transfer_tokens.return_value = None
        scheduler.disagg_prefill_pending_chunk_rids.add(req.rid)

        scheduler._send_kv_chunk(req, last_chunk=True)

        if index == 0:
            assert isinstance(req.finished_reason, FINISH_ABORT)
            assert req.finished_reason.status_code == 500
            assert error in req.finished_reason.message
            req.disagg_kv_sender.abort.assert_called_once_with()
            req.disagg_kv_sender.send.assert_not_called()
            assert req.customized_info is None
        else:
            assert not req.finished()
            req.disagg_kv_sender.abort.assert_not_called()
            req.disagg_kv_sender.send.assert_called_once()
            assert (
                scheduler.disagg_metadata_buffers.get_customized_info(0)
                == req.customized_info
            )
        assert req.rid not in scheduler.disagg_prefill_pending_chunk_rids

    with (
        patch.object(
            scheduler.disagg_metadata_buffers,
            "set_buf",
            side_effect=RuntimeError("device failure"),
        ),
        pytest.raises(RuntimeError, match="device failure"),
    ):
        scheduler._send_kv_chunk(req, last_chunk=True)


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
        release.assert_called_once_with(req, scheduler.tree_cache, checkpoint=False)
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
        self.send_kv_chunk = Mock()


def test_early_send_wait_event_comes_from_the_scheduler_device_module():
    """The early-send event is created on the scheduler's device.

    forward_stream is built from that same device, and an event of one accelerator
    cannot record against another's stream. Resolving the module from the process
    default instead reinstates that mismatch whenever --device names something
    other than the default accelerator.
    """
    recorded = []

    class _Event:
        def record(self, stream):
            recorded.append(stream)

    scheduler = _EarlySendScheduler(SimpleNamespace(Event=_Event))
    req = SimpleNamespace(
        pending_bootstrap=False,
        prefix_len=8,
        host_hit_length=0,
        early_send_prefix_end=None,
        start_send_idx=0,
        disagg_kv_sender=SimpleNamespace(_early_send_wait_event=None),
    )

    with envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(True):
        scheduler.maybe_send_cached_prefix_chunk(req)

    assert isinstance(req.disagg_kv_sender._early_send_wait_event, _Event)
    assert recorded == [scheduler.forward_stream]
    scheduler.send_kv_chunk.assert_called_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
