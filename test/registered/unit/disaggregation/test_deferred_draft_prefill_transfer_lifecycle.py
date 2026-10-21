"""Exercise transport publication, worker completion and destination retirement."""

import pickle
import threading
import unittest
from array import array
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.deferred_draft_prefill import DeferredDraftPrefillPool
from sglang.srt.disaggregation.deferred_draft_prefill_lifecycle import (
    DeferredDraftPrefillRetirementState as State,
)
from sglang.srt.disaggregation.deferred_draft_prefill_lifecycle import (
    DeferredDraftPrefillSourceFence,
    DeferredDraftPrefillSourceRetirements,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    maybe_release_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    ReqToMetadataIdxAllocator,
    TransferBackend,
)
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeferredDraftPrefillTransferLifecycle(unittest.TestCase):
    def manager(self):
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.bootstrap_port = 8998
        manager.enable_trace = manager.enable_staging = False
        manager.enable_deferred_decode_kv_release = True
        manager._staging_outstanding = defaultdict(int)
        manager.request_status = {7: KVPoll.Transferring}
        manager.check_status = lambda room: manager.request_status[room]
        manager.session_lock = threading.Lock()
        manager.state_layout_rejections = {}
        manager.failed_sessions = set()
        manager.session_failures = defaultdict(int)
        manager.failed_session_probes = {}
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.attn_tp_size = 1
        manager.kv_args = SimpleNamespace(kv_data_ptrs=[1])
        manager.req_to_decode_prefix_len = {}
        manager._prefill_unique_rank = lambda: 0
        manager._get_dsa_cache_transfer_skip_flags = lambda registration: (False, False)
        manager._maybe_ack_drained_abort = Mock()
        manager.poison_deferred_ack_room = Mock()
        manager.send_kvcache = Mock(return_value=0)
        manager.maybe_send_extra = Mock(return_value=0)
        manager.send_aux = Mock(return_value=0)
        manager.conclude_transfer = lambda **kwargs: manager.request_status.update(
            {kwargs["bootstrap_room"]: kwargs["status"]}
        )
        manager.conclude_failure = lambda **kwargs: manager.request_status.update(
            {kwargs["bootstrap_room"]: KVPoll.Failed}
        )
        manager.transfer_queues = [Queue()]
        registration = SimpleNamespace(
            requires_dcp_relayout=False,
            dst_kv_ptrs=[2],
            dst_aux_ptrs=[3],
            dst_attn_tp_size=4,
            dst_kv_layer_ids=None,
            dst_kv_item_len=8,
        )
        manager.decode_kv_args_table = {"peer:1": registration}
        manager.transfer_infos = {
            7: {
                "peer:1": SimpleNamespace(
                    is_dummy=False,
                    mooncake_session_id="peer:1",
                    dst_device_kv_indices=None,
                    dst_kv_indices=np.array([4], dtype=np.int32),
                    endpoint="127.0.0.1",
                    dst_port=8998,
                    room=7,
                    required_dst_info_num=1,
                )
            }
        }
        return manager

    def publish(self, manager, fence):
        manager.add_transfer_request(
            7,
            np.array([1], dtype=np.int32),
            slice(0, 1),
            True,
            aux_index=0,
            state_indices=[[1]],
            deferred_draft_source_fence=fence,
        )

    def run_worker(self, manager):
        manager.transfer_queues[0].put(None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            manager.transfer_worker(manager.transfer_queues[0], executor)

    def test_queued_final_chunk_holds_until_worker_explicitly_skips(self):
        manager, fence = self.manager(), DeferredDraftPrefillSourceFence()
        self.publish(manager, fence)
        self.assertEqual(manager._staging_outstanding.get(7, 0), 0)
        self.assertEqual(fence.seal(), State.HOLD)
        manager.request_status[7] = KVPoll.Failed
        self.run_worker(manager)
        self.assertEqual(fence.retirement_state, State.FREE)
        manager.send_kvcache.assert_not_called()

    def test_abort_during_sync_transfer_waits_for_return_and_worker_cleanup(self):
        manager, fence = self.manager(), DeferredDraftPrefillSourceFence()
        entered, finish = threading.Event(), threading.Event()
        errors = []

        def transfer(*args, **kwargs):
            entered.set()
            if not finish.wait(2):
                raise RuntimeError("Test transfer was not released")
            return 0

        manager.send_kvcache.side_effect = transfer
        self.publish(manager, fence)

        def worker():
            try:
                self.run_worker(manager)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            manager.request_status[7] = KVPoll.Failed
            self.assertEqual(fence.seal(), State.HOLD)
        finally:
            finish.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(fence.retirement_state, State.FREE)
        self.assertNotIn(7, manager._staging_outstanding)

    def test_timeout_with_unknown_background_read_stays_poisoned_after_cleanup(self):
        manager, fence = self.manager(), DeferredDraftPrefillSourceFence()
        manager.maybe_send_extra.return_value = -1
        self.publish(manager, fence)
        fence.seal()
        self.run_worker(manager)
        self.assertEqual(fence.retirement_state, State.QUARANTINE)
        self.assertIn("State transfer", fence.poison_reason)
        manager.send_aux.assert_not_called()
        fence.settled()
        self.assertEqual(fence.retirement_state, State.QUARANTINE)

    def test_unknown_worker_exception_cannot_settle_generation(self):
        manager, fence = self.manager(), DeferredDraftPrefillSourceFence()
        manager.maybe_send_extra.side_effect = RuntimeError("unknown submitted send")
        self.publish(manager, fence)
        fence.seal()
        with self.assertRaisesRegex(RuntimeError, "unknown submitted send"):
            self.run_worker(manager)
        self.assertEqual(fence.retirement_state, State.QUARANTINE)

    def test_feature_off_worker_retains_ordinary_transfer_behavior(self):
        manager = self.manager()
        self.publish(manager, None)
        self.run_worker(manager)
        self.assertEqual(manager.request_status[7], KVPoll.Success)
        manager.send_aux.assert_called_once()

    def test_decode_resolver_forwards_drain_proof_to_new_hidden_pool(self):
        for drained in (True, False):
            with self.subTest(drained=drained):
                pool = DeferredDraftPrefillPool(
                    2, 8, 4, "torch.bfloat16", "cpu", allocate_storage=False
                )
                req = SimpleNamespace(
                    rid="same-rid", origin_input_ids=[1, 2], bootstrap_room=7
                )
                pool.alloc(req)
                manager = SimpleNamespace(
                    is_abort_release_safe=Mock(return_value=drained),
                    clear_deferred_abort_state=Mock(),
                )
                receiver = SimpleNamespace(kv_mgr=manager, clear=Mock())
                decode_req = SimpleNamespace(
                    req=req, kv_receiver=receiver, host_staged=False
                )
                queue = object.__new__(DecodeTransferQueue)
                queue.scheduler = SimpleNamespace(
                    deferred_draft_prefill_pool=pool,
                    metrics_reporter=SimpleNamespace(enable_metrics=False),
                )
                queue.enable_host_receive = queue.enable_staging = False
                queue.tree_cache = Mock()
                queue.metadata_buffers = SimpleNamespace(bootstrap_room={1: 7})
                queue.req_to_metadata_buffer_idx_allocator = Mock()
                queue.deferred_kv_release_timeout = 1
                queue._deferred_releases = [(decode_req, 0, float("-inf"), 1, 1)]
                with patch(
                    "sglang.srt.disaggregation.decode.release_kv_cache"
                ) as release:
                    queue.resolve_deferred_releases()
                release.assert_called_once_with(req, queue.tree_cache, checkpoint=False)
                self.assertEqual(pool.available_slots, 2 if drained else 1)
                self.assertEqual(pool.quarantined_slots, 0 if drained else 1)
                self.assertIsNone(decode_req.kv_receiver)
                self.assertEqual(queue._deferred_releases, [])
                newer = SimpleNamespace(
                    rid=req.rid, origin_input_ids=[1, 2], bootstrap_room=8
                )
                pool.alloc(newer)
                self.assertFalse(pool.free(req))
                self.assertTrue(pool.owns(newer))

    def test_pp_pyobj_roundtrip_preserves_internal_request_generation(self):
        base = dict(
            rid="reused",
            input_text="",
            input_ids=array("q", [1, 2]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=8),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            bootstrap_room=7,
        )
        ordinary = TokenizedGenerateReqInput(**base)
        self.assertIsNone(ordinary.pd_deferred_draft_source_generation)
        tokenized = TokenizedGenerateReqInput(
            **base, pd_deferred_draft_source_generation="request-generation"
        )
        restored = pickle.loads(pickle.dumps(tokenized))
        self.assertEqual(
            restored.pd_deferred_draft_source_generation, "request-generation"
        )
        self.assertEqual(restored.rid, tokenized.rid)
        self.assertEqual(
            TokenizedGenerateReqInput.__struct_fields__[-1],
            "pd_deferred_draft_source_generation",
        )

    def test_real_pp_transfer_and_release_callers_preserve_unsafe_vote(self):
        key = ("reused", 7, "request-generation")
        previous = None
        for rank, state in enumerate(
            (State.FREE, State.HOLD, State.FREE, State.QUARANTINE)
        ):
            scheduler = SimpleNamespace(
                pp_group=SimpleNamespace(is_first_rank=rank == 0),
                deferred_draft_source_retirements=SimpleNamespace(
                    snapshot=lambda state=state: {key: state}
                ),
                disagg_prefill_inflight_queue=[],
                get_rids=lambda *args: ["normal-transfer"],
                _pp_recv_pyobj_from_prev_stage=lambda previous=previous: previous,
            )
            previous = SchedulerPPMixin._pp_pd_get_prefill_transferred_ids(scheduler)
            previous = pickle.loads(pickle.dumps(previous))
        self.assertEqual(list(previous), ["normal-transfer"])
        self.assertEqual(previous.retirement_states, {key: State.QUARANTINE})
        sent = []
        last = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            _pp_send_pyobj_to_next_stage=lambda payload, **kwargs: sent.append(payload),
        )
        _, release = SchedulerPPMixin._pp_pd_send_consensus_release_ids(
            last, [previous], 0, None, previous
        )
        self.assertIs(release, previous)
        self.assertIs(sent[0], previous)
        self.assertEqual(
            pickle.loads(pickle.dumps(sent[0])).retirement_states,
            {key: State.QUARANTINE},
        )

    def test_reused_rid_waits_for_previous_source_owner_without_metadata_allocation(
        self,
    ):
        pool = DeferredDraftPrefillPool(
            2, 8, 4, "torch.bfloat16", "cpu", allocate_storage=False
        )
        old = SimpleNamespace(rid="reused", origin_input_ids=[1, 2])
        newer = SimpleNamespace(
            rid="reused", origin_input_ids=[1, 2], metadata_buffer_index=-1
        )
        pool.alloc(old)
        queue = object.__new__(PrefillBootstrapQueue)
        queue.scheduler = SimpleNamespace(deferred_draft_prefill_pool=pool)
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 2, alloc=Mock(return_value=1)
        )
        self.assertFalse(queue.ensure_metadata_buffer(newer))
        queue.req_to_metadata_buffer_idx_allocator.alloc.assert_not_called()
        self.assertEqual(newer.metadata_buffer_index, -1)
        self.assertTrue(pool.owns(old))
        pool.free_source(old)
        self.assertTrue(queue.ensure_metadata_buffer(newer))
        self.assertTrue(pool.owns(newer))
        self.assertEqual(newer.metadata_buffer_index, 1)

    def test_decode_admission_waits_if_any_tp_peer_still_owns_reused_rid(self):
        for local_old_owner in (True, False):
            with self.subTest(local_old_owner=local_old_owner):
                pool = DeferredDraftPrefillPool(
                    2, 8, 4, "torch.bfloat16", "cpu", allocate_storage=False
                )
                old = SimpleNamespace(rid="reused", origin_input_ids=[1, 2])
                if local_old_owner:
                    pool.alloc(old)
                req = SimpleNamespace(rid="reused", finished_reason=None)
                pending = SimpleNamespace(req=req, waiting_for_input=True)
                queue = object.__new__(DecodePreallocQueue)
                queue.pp_size = 1
                queue.queue = [pending]
                queue.pending_reqs = []
                queue._resolve_pending_reqs = Mock()
                queue._update_handshake_waiters = Mock()
                queue._uses_swa_tail_prealloc = Mock(return_value=False)
                queue._uses_swa_reservation = Mock(return_value=False)
                queue._allocatable_token_budgets = Mock(return_value=1024)
                queue._hicache_pending_restore_tokens = Mock(return_value=0)
                queue._pre_alloc = Mock()
                queue.gloo_group = None
                queue.scheduler = SimpleNamespace(
                    deferred_draft_prefill_pool=pool,
                    running_batch=SimpleNamespace(reqs=[]),
                    enable_priority_scheduling=False,
                    enable_hisparse=False,
                    enable_lora=False,
                )
                queue.req_to_metadata_buffer_idx_allocator = Mock()
                before = pool.available_slots

                def peer_owner_vote(vector, **kwargs):
                    self.assertEqual(vector.numel(), 2)
                    # A peer may still be draining while this rank has released.
                    vector[1] = 0

                with (
                    patch("torch.distributed.get_world_size", return_value=2),
                    patch(
                        "torch.distributed.all_reduce", side_effect=peer_owner_vote
                    ) as reduced,
                ):
                    preallocated, failed = queue.pop_preallocated()
                reduced.assert_called_once()
                self.assertEqual((preallocated, failed), ([], []))
                self.assertEqual(queue.queue, [pending])
                self.assertEqual(pool.available_slots, before)
                queue.req_to_metadata_buffer_idx_allocator.alloc.assert_not_called()
                queue._pre_alloc.assert_not_called()
                if local_old_owner:
                    self.assertTrue(pool.owns(old))

    def test_real_fake_health_sender_retires_repeatedly_without_capacity_loss(self):
        pool = DeferredDraftPrefillPool(
            2, 8, 4, "torch.bfloat16", "cpu", allocate_storage=False
        )
        allocator = ReqToMetadataIdxAllocator(2)
        registry = DeferredDraftPrefillSourceRetirements(allocator)
        allocator.deferred_draft_source_retirements = registry
        queue = object.__new__(PrefillBootstrapQueue)
        queue.scheduler = SimpleNamespace(deferred_draft_prefill_pool=pool)
        queue.req_to_metadata_buffer_idx_allocator = allocator
        queue._check_if_req_exceed_kv_capacity = Mock(return_value=False)
        queue._process_req = Mock()
        queue.transfer_backend = TransferBackend.MOONCAKE
        queue.bootstrap_port = 8998
        queue.kv_manager = SimpleNamespace()
        for index in range(5):
            req = SimpleNamespace(
                rid="health",
                origin_input_ids=[1, 2],
                metadata_buffer_index=-1,
                bootstrap_host=FAKE_BOOTSTRAP_HOST,
                bootstrap_room=index,
                disagg_prefill_dp_rank=None,
                pd_deferred_draft_source_generation="health-" + str(index),
            )
            self.assertTrue(queue.create_sender(req, 1))
            self.assertTrue(queue.ensure_metadata_buffer(req))
            req.disagg_kv_sender.send(np.array([1], dtype=np.int32))
            self.assertEqual(req.disagg_kv_sender.poll(), KVPoll.Success)
            maybe_release_metadata_buffer(req, allocator)
            registry.resolve()
            self.assertEqual(pool.available_slots, 2)
            self.assertEqual(allocator.available_size(), 2)
            self.assertEqual(pool.quarantined_slots, 0)
            self.assertEqual(registry.quarantined_metadata_slots, 0)


if __name__ == "__main__":
    unittest.main()
