from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import FreeDeviceKV
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq, release_only_retirer
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _Req(FakeReq):
    def __init__(self, *, allocated: bool = True):
        self.rid = "aborted-prefill"
        self.cache_request_handle = CacheRequestHandle(self.rid, 0)
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
        self.session = None
        self.get_mm_token_counts = Mock()
        self.release_mm_inputs_on_abort = Mock()
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
        self.prefill_launched_reqs = KeyedReqList(ListKind.PREFILL_LAUNCHED)
        self.batch_result_processor = SimpleNamespace(
            snapshot_auxiliary_output_starts=Mock(return_value=[]),
            move_logprobs_to_cpu=Mock(),
            consume_auxiliary_output=Mock(),
            _maybe_collect_customized_info=Mock(),
        )
        self.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        self.tree_cache = Mock()
        self.request_retirer = release_only_retirer(self.tree_cache)
        self.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
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
        chunked_req=None,
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
    scheduler.enable_staging = False
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=1,
        translate_kv_indices_for_transfer=lambda indices: indices,
        storage=None,
    )
    scheduler.req_to_token_pool = SimpleNamespace(req_to_token=torch.tensor([[0]]))
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
        kv_manager=SimpleNamespace(
            kv_args=SimpleNamespace(state_types=[], state_sub_pools=[])
        )
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
        req = _Req()
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
                    req in scheduler.disagg_prefill_transfer_queue,
                )
            )

        req.disagg_kv_sender.mark_prefill_complete.side_effect = ready
        move([req], scheduler.prefill_launched_reqs)
        scheduler.process_batch_result_disagg_prefill(batch, result)
        self.assertEqual(observed, [([11], [1.0, 2.0], [5, 6], True)])
        self.assertTrue(req.pending_bootstrap)
        scheduler.send_kv_chunk.assert_not_called()

    @patch(
        "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache",
        side_effect=_free_req,
    )
    def test_aborted_final_result_never_publishes_ready(self, release):
        scheduler = _Scheduler()
        req = _Req()
        scheduler.process_batch_result_disagg_prefill(_batch(req), _result())
        self.assertTrue(req.finished())
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(list(scheduler.disagg_prefill_transfer_queue), [])
        release.assert_called_once_with(req, scheduler.tree_cache, checkpoint=False)
        req.disagg_kv_sender.mark_prefill_complete.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.should_force_retry", return_value=True)
    def test_discarded_attempt_never_publishes_ready(self, force_retry):
        scheduler = _Scheduler()
        req = _Req()
        req.to_finish = None
        retried = []
        scheduler.optimistic_release_and_requeue = lambda r: retried.append(r)
        scheduler.process_batch_result_disagg_prefill(_batch(req), _result())
        self.assertEqual(retried, [req])
        self.assertEqual(req.output_ids, [])
        self.assertEqual(list(scheduler.disagg_prefill_transfer_queue), [])
        req.disagg_kv_sender.mark_prefill_complete.assert_not_called()


class TestEarlySentLastChunkResult(CustomTestCase):
    """SGLANG_ENABLE_PP_PD_PREFILL_EARLY_SEND: a PP stage may have sent the last chunk
    when its forward ended; the result never waits for it, and every stage holds the
    checkpoint's frees alike whether or not it sent early itself."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="early"
        )
        override.install()
        self.addCleanup(override.restore)

    def _process_result(self, early_sent, tree_cache=None, early_send_on=True):
        scheduler = _Scheduler()
        del scheduler.send_kv_chunk  # the real one, over a stub transfer
        scheduler._send_kv_chunk = Mock()
        scheduler.kv_checksum_computer = None
        scheduler.disagg_prefill_early_send_on = early_send_on
        scheduler.disagg_prefill_early_send_stats = {"deferred": 0}
        scheduler.disagg_prefill_deferred_frees = {}
        if tree_cache is not None:
            scheduler.tree_cache = tree_cache
        req = _Req()
        req.to_finish = None
        req.pending_bootstrap = False
        req.return_sampling_mask = False
        req.bootstrap_host = "10.0.0.1"
        req.disagg_kv_sender = SimpleNamespace(
            supports_host_tensor_indices=True,
            kv_mgr=SimpleNamespace(
                kv_shard_size=1, enable_all_cp_ranks_for_transfer=False
            ),
        )
        req.time_stats.set_prefill_transfer_queue_entry_time = Mock()
        if early_sent:
            scheduler.disagg_prefill_early_sent_rids = {req.rid}
        move([req], scheduler.prefill_launched_reqs)
        scheduler.process_batch_result_disagg_prefill(_batch(req), _result())
        self.assertIn(req, scheduler.disagg_prefill_transfer_queue)
        # A stage that did not send early sends the last chunk from the result.
        self.assertEqual(scheduler._send_kv_chunk.called, not early_sent)
        self.assertFalse(scheduler.disagg_prefill_early_sent_rids)
        return scheduler, req

    @patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
    def test_every_stage_skips_the_checkpoint_on_a_cache_that_cannot_hold(
        self, checkpoint
    ):
        # The release after the transfer publishes the KV. Whether this stage sent
        # early does not matter, so a stage that kept the result path skips too.
        for early_sent, early_send_on, checkpointed in (
            (True, True, False),
            (False, True, False),
            (False, False, True),  # flag off: the regular path
        ):
            with self.subTest(early_sent=early_sent, early_send_on=early_send_on):
                checkpoint.reset_mock()
                self._process_result(early_sent, early_send_on=early_send_on)
                self.assertEqual(checkpoint.called, checkpointed)

    def test_stages_that_disagree_on_early_send_hold_and_free_alike(self):
        # Stage A sent the last chunk early, stage B kept the result path (e.g. a
        # PP-matched decode). Both checkpoint now, hold the dedup's frees, and free
        # them at the same release.
        dup = torch.arange(4)

        def checkpoint(req, tree_cache):  # the insert found the prefix: free the copy
            tree_cache._apply_cache_actions([FreeDeviceKV(indices=[dup])])

        stages = []
        with patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache", checkpoint):
            for early_sent in (True, False):
                storage = SimpleNamespace(free_token_rows=Mock())
                cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
                cache.cache_controller = None
                cache.kv_storage = lambda storage=storage: storage
                scheduler, req = self._process_result(early_sent, cache)
                stages.append((scheduler, req, storage))
        for scheduler, req, storage in stages:
            storage.free_token_rows.assert_not_called()
            self.assertIn(req, scheduler.disagg_prefill_deferred_frees)
        for scheduler, req, storage in stages:  # the consensus release
            scheduler._apply_deferred_frees(req)
            storage.free_token_rows.assert_called_once_with(dup)
            self.assertFalse(scheduler.disagg_prefill_deferred_frees)

    def test_release_after_a_skipped_checkpoint_matches_the_regular_path(self):
        # A real RadixCache cannot hold frees, so a held result skips the checkpoint
        # and the release after the transfer publishes the KV. A sibling inserted
        # this request's last token meanwhile, so the release dedups it. Tree and
        # freed pages end as on the regular path (checkpoint, then release).
        ends = {}
        for held in (False, True):
            freed = []
            allocator = Mock(device=torch.device("cpu"), page_size=1)
            allocator.free_segment.side_effect = lambda idx, start_pos=0: freed.extend(
                idx.tolist()
            )
            cache = RadixCache.create_simulated(mock_allocator=allocator)
            req_to_token = torch.zeros(2, 8, dtype=torch.int64)
            cache.req_to_token_pool = SimpleNamespace(
                req_to_token=req_to_token,
                write=lambda idx, values: req_to_token.__setitem__(idx, values),
            )
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", [1, 2, 3])),
                    value=torch.tensor([10, 20, 30]),
                )
            )
            match = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3])))
            )
            req_to_token[0, :4] = torch.tensor([10, 20, 30, 40])
            req = _RadixReq([1, 2, 3, 4], match.last_device_node, cache)
            cache.insert(  # the sibling's copy of token 4
                InsertParams(
                    key=RadixKey(array("q", [1, 2, 3, 4])),
                    value=torch.tensor([10, 20, 30, 41]),
                )
            )
            scheduler = SimpleNamespace(
                tree_cache=cache,
                disagg_prefill_early_send_stats={"deferred": 0},
                disagg_prefill_deferred_frees={},
            )
            scheduler.checkpoint_disagg_prefill = lambda r: (
                SchedulerDisaggregationPrefillMixin.checkpoint_disagg_prefill(
                    scheduler, r
                )
            )
            if held:
                SchedulerDisaggregationPrefillMixin._checkpoint_holding_frees(
                    scheduler, req
                )
            else:
                scheduler.checkpoint_disagg_prefill(req)
            req.finished_reason = FINISH_LENGTH(length=0)
            SchedulerDisaggregationPrefillMixin._apply_deferred_frees(scheduler, req)
            release_kv_cache(req, cache, checkpoint=True)
            ends[held] = (
                cache.evictable_size(),
                cache.protected_size(),
                sorted(freed),
            )
        self.assertEqual(ends[True], ends[False])
        self.assertEqual(ends[False], (4, 0, [40]))

    def test_inflight_release_applies_held_frees_first(self):
        for poll in (KVPoll.Success, KVPoll.Failed):
            with (
                self.subTest(poll=poll),
                patch(
                    "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                    return_value=[poll],
                ),
            ):
                calls = Mock()
                scheduler = _Scheduler()
                scheduler.tree_cache = calls.tree_cache
                scheduler.request_retirer = calls.retirer
                scheduler.handle_inflight_transfer_failure = calls.failure
                scheduler._release_mm_inputs = Mock()
                scheduler.attn_cp_cpu_group = scheduler.attn_tp_cpu_group = object()
                scheduler.scheduler_stage_metrics = None
                req = _Req()
                req.pending_bootstrap = False
                req.return_routed_experts = False
                req.finished_reason = FINISH_ABORT()
                req.time_stats.set_prefill_kv_transfer_finish_time = Mock()
                frees = [FreeDeviceKV(indices=[torch.arange(4)])]
                scheduler.disagg_prefill_deferred_frees = {req: frees}
                move([req], scheduler.disagg_prefill_transfer_queue)
                self.assertEqual(
                    scheduler.process_disagg_prefill_transfer_queue(), [req]
                )
                self.assertFalse(scheduler.disagg_prefill_deferred_frees)
                names = [c[0] for c in calls.mock_calls]
                self.assertEqual(names[0], "tree_cache.apply_deferred_frees")
                self.assertEqual(calls.mock_calls[0].args, (frees,))
                self.assertIn(
                    "retirer.release" if poll == KVPoll.Success else "failure",
                    names[1:],
                )

    def test_eviction_runs_outside_a_deferral(self):
        # Eviction frees only unlocked nodes, so it must not join the held frees.
        class _Cache(UnifiedRadixCache):
            def __init__(self):
                self.sinks_seen = []

            @property
            def disable(self):  # read first by the eviction body
                self.sinks_seen.append(self._deferred_free_sink)
                return True

        cache, held = _Cache(), []
        with cache.defer_device_frees(held):
            cache._evict(EvictParams(num_tokens=1))
            self.assertIs(cache._deferred_free_sink, held)
        self.assertEqual(cache.sinks_seen, [None])


class _RadixReq(FakeReq):
    """A request over a real RadixCache: matched prefix locked, all of
    ``fill_ids`` in its req_to_token row 0."""

    def __init__(self, fill_ids, last_node, cache):
        super().__init__(
            rid="radix",
            origin_input_ids=array("q", fill_ids),
            output_ids=array("q"),
            full_untruncated_fill_ids=array("q", fill_ids),
            extend_end=len(fill_ids),
            last_node=last_node,
            lock=cache.lock(last_node),
            extra_key=None,
            cache_salt=None,
            prefix_len=0,
            priority=0,
            pending_bootstrap=False,
            kv=ReqKvInfo(
                req_pool_idx=0,
                kv_committed_len=len(fill_ids),
                kv_allocated_len=len(fill_ids),
                cache_protected_len=3,
            ),
            kv_rotation_base=None,
            skip_radix_cache_insert=False,
            finished_reason=None,
        )

    def get_fill_ids(self):
        return self.full_untruncated_fill_ids[: self.extend_end]

    def finished(self):
        return self.finished_reason is not None

    def refresh_fill_ids(self):
        pass

    def owned_kv_len(self):
        return self.kv.kv_committed_len


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
