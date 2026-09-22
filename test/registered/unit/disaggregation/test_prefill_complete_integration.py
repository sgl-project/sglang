"""Exercise the allocation policy through production transport and queue methods."""

import threading
import time
import unittest
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver, PrefillServerInfo
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.prefill_complete import PrefillCompleteManager
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestAllocationConfiguration(CustomTestCase):
    def test_policy_owns_optimistic_prefill_dependency(self):
        args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
            disaggregation_decode_allocation_policy="prefill_complete",
        )
        handle_pd_disaggregation(args)
        self.assertEqual(resolving_view(args).optimistic_prefill_attempts, 1)
        # Resolution must not rewrite the raw user input.
        self.assertEqual(args.optimistic_prefill_attempts, 0)

    def test_early_does_not_enable_optimistic_prefill(self):
        args = ServerArgs(model_path="dummy", disaggregation_mode="prefill")
        handle_pd_disaggregation(args)
        self.assertEqual(resolving_view(args).optimistic_prefill_attempts, 0)

    def test_dp_attention_topology_is_supported(self):
        for size in (1, 4, 8):
            with self.subTest(size=size):
                args = ServerArgs(
                    model_path="dummy",
                    disaggregation_mode="prefill",
                    disaggregation_decode_allocation_policy="prefill_complete",
                    tp_size=size,
                    dp_size=size,
                    enable_dp_attention=size > 1,
                )
                handle_pd_disaggregation(args)
                self.assertEqual(resolving_view(args).optimistic_prefill_attempts, 1)

    def test_unsupported_modes_fail_explicitly(self):
        combinations = [
            {"disaggregation_mode": "null"},
            {"disaggregation_transfer_backend": "nixl"},
            {"tp_size": 2},
            {"pp_size": 2},
            {"attn_cp_size": 2},
            {"dcp_size": 2},
            {"enable_hisparse": True},
            {"enable_pdmux": True},
            {"enable_pd_role_switch": True},
            {"language_only": True},
            {
                "enable_hierarchical_cache": True,
                "hicache_write_policy": "write_through",
            },
            {
                "disaggregation_mode": "decode",
                "disaggregation_decode_enable_radix_cache": True,
            },
        ]
        for changed in combinations:
            with self.subTest(changed=changed):
                values = dict(
                    model_path="dummy",
                    disaggregation_mode="prefill",
                    disaggregation_decode_allocation_policy="prefill_complete",
                )
                values.update(changed)
                with self.assertRaisesRegex(ValueError, "prefill_complete"):
                    handle_pd_disaggregation(ServerArgs(**values))


class TestAllocationIntegration(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="prefill_complete"
        )
        override.install()
        self.addCleanup(override.restore)
        self.wire = deque()
        self.prefill = self.manager("prefill", 1)
        self.decode = self.manager("decode", 2)
        self.endpoints = {("prefill", 1): self.prefill, ("decode", 2): self.decode}

    def manager(self, ip, port):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.local_ip, mgr.rank_port = ip, port
        mgr.request_status = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.waiting_timeout = 5
        mgr.req_to_decode_prefix_len = {}
        mgr.transfer_infos = {}
        mgr._deferred_ack_targets = {}
        mgr.required_prefill_response_num_table = {}
        mgr.prefill_response_tracker = defaultdict(set)
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr._send_prefill_complete_message = lambda endpoint, msg: self.wire.append(
            (endpoint, msg)
        )
        mgr.prefill_complete = PrefillCompleteManager(
            send=mgr._send_prefill_complete_message,
            on_cancel=mgr._cancel_before_decode_allocation,
        )
        return mgr

    def flush(self):
        while self.wire:
            endpoint, msg = self.wire.popleft()
            self.endpoints[endpoint].prefill_complete.handle_message(msg)

    def sender(self, room=1):
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr, sender.bootstrap_room = self.prefill, room
        sender.conclude_state = None
        sender.trace_ctx = MagicMock()
        self.prefill.update_status(room, KVPoll.Bootstrapping)
        sender._source_readiness = self.prefill.prefill_complete.add_source(room=room)
        return sender

    def receiver(self, room=1):
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.kv_mgr, receiver.bootstrap_room = self.decode, room
        receiver.conclude_state = None
        receiver._metadata_sent = False
        receiver.abort_notified = False
        receiver.bootstrap_addr = "prefill:8998"
        receiver.bootstrap_infos = [{"rank_ip": "prefill", "rank_port": 1}]
        receiver.init_time = time.time()
        receiver._connection_pool_entries = {}
        receiver._destination_readiness = self.decode.prefill_complete.add_destination(
            room=room, endpoint=("prefill", 1), timeout=5
        )
        self.decode.update_status(room, KVPoll.WaitingForInput)
        self.decode.addr_to_rooms_tracker[receiver.bootstrap_addr].add(room)
        return receiver

    def test_receiver_hides_allocatable_status_until_source_finishes(self):
        sender, receiver = self.sender(), self.receiver()
        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        sender.mark_prefill_complete()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)

    def test_sender_abort_clears_readiness_and_fails_decoder_before_metadata(self):
        sender, receiver = self.sender(), self.receiver()
        receiver.poll()
        self.flush()
        sender.abort()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertEqual(self.prefill.prefill_complete._sources, {})
        self.assertEqual(self.decode.prefill_complete._destinations, {})
        self.assertFalse(receiver._metadata_sent)

    def test_receiver_abort_notifies_live_source_and_cleans_local_state(self):
        sender, receiver = self.sender(), self.receiver()
        receiver.poll()
        self.flush()
        receiver.abort()
        self.flush()
        self.assertEqual(self.prefill.check_status(1), KVPoll.Failed)
        self.assertEqual(self.decode.prefill_complete._destinations, {})
        sender.clear()
        receiver.clear()
        self.assertEqual(self.prefill.prefill_complete._sources, {})
        self.assertNotIn(1, self.decode.request_status)

    def test_readiness_timeout_aborts_before_allocation(self):
        self.sender()
        receiver = self.receiver()
        receiver._destination_readiness.deadline = 0
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertFalse(receiver._metadata_sent)
        self.assertEqual(self.decode.prefill_complete._destinations, {})

    def test_abort_after_metadata_uses_existing_drain_protocol(self):
        self.sender()
        receiver = self.receiver()
        receiver._metadata_sent = True
        with patch.object(CommonKVReceiver, "_send_abort_notification") as notify:
            receiver.abort()
        notify.assert_called_once()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertEqual(self.decode.prefill_complete._destinations, {})
        self.assertEqual(list(self.wire), [])

    def test_source_error_racing_metadata_publication_uses_abort_drain(self):
        self.sender()
        receiver = self.receiver()
        receiver.poll()
        self.flush()
        receiver._metadata_sent = True
        self.prefill.prefill_complete.fail_source(room=1, reason="source failed")
        self.flush()
        with patch.object(CommonKVReceiver, "_send_abort_notification") as notify:
            self.assertEqual(receiver.poll(), KVPoll.Failed)
        notify.assert_called_once()
        self.assertEqual(self.decode.failure_records[1], "source failed")
        self.assertEqual(self.decode.prefill_complete._destinations, {})

    def test_mismatched_peer_fails_request_before_destination_registration(self):
        receiver = self.receiver()
        self.decode.prefill_info_table = {
            receiver.bootstrap_addr: PrefillServerInfo(
                attn_tp_size=1,
                attn_cp_size=1,
                dp_size=1,
                pp_size=1,
                page_size=1,
                kv_cache_dtype=None,
                follow_bootstrap_room=True,
                decode_allocation_policy="early",
            )
        }
        with patch.object(receiver, "_setup_bootstrap_infos") as setup:
            CommonKVReceiver.init(receiver, 0)
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertIn("policy mismatch", self.decode.failure_records[1])
        setup.assert_not_called()

    def test_scheduler_does_not_reserve_request_rows_for_unready_prefill(self):
        sender, receiver = self.sender(), self.receiver()
        req = SimpleNamespace(
            rid="req",
            bootstrap_room=1,
            finished_reason=None,
            return_logprob=False,
            time_stats=MagicMock(),
        )
        entry = DecodeRequest(req=req, kv_receiver=receiver)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue, queue.pending_reqs = [entry], []
        queue.pp_size, queue.tp_rank = 1, 0
        queue.gloo_group = object()
        queue._resolve_pending_reqs = Mock()
        queue._uses_swa_tail_prealloc = Mock(return_value=False)
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            prealloc_fits_assumes_reclaim=Mock(return_value=False)
        )
        queue._allocatable_token_budgets = Mock(return_value=100000)
        queue._hicache_pending_restore_tokens = Mock(return_value=0)
        queue.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            enable_priority_scheduling=False,
            enable_hisparse=False,
            enable_lora=False,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
        )
        # This is the first admission resource checked after waiting_for_input.
        queue.req_to_token_pool = SimpleNamespace(available_size=Mock(return_value=0))
        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            side_effect=lambda receivers, group: [r.poll() for r in receivers],
        ):
            self.assertEqual(queue.pop_preallocated(), ([], []))
            self.flush()
            self.assertFalse(entry.waiting_for_input)
            queue.req_to_token_pool.available_size.assert_not_called()
            sender.mark_prefill_complete()
            self.flush()
            self.assertEqual(queue.pop_preallocated(), ([], []))
        self.assertTrue(entry.waiting_for_input)
        queue.req_to_token_pool.available_size.assert_called_once()
        self.assertEqual(queue.queue, [entry])

    def test_exhausted_optimistic_attempt_remains_compute_eligible(self):
        override = get_context().override_server_args(
            optimistic_prefill_attempts=1,
            disaggregation_decode_allocation_policy="prefill_complete",
        )
        override.install()
        self.addCleanup(override.restore)
        req = SimpleNamespace(
            rid="retry",
            prefill_attempt_count=1,
            output_ids=[42],
            skip_radix_cache_insert=False,
            kv=SimpleNamespace(cache_protected_len=16),
            full_untruncated_fill_ids=list(range(32)),
            _compute_max_prefix_len=Mock(return_value=31),
            storage_prefetch_retry_attempts=3,
            storage_prefetch_last_match_len=8,
            reset_for_retract=Mock(),
            advance_cache_request_handle=Mock(),
            time_stats=MagicMock(),
        )
        scheduler = SimpleNamespace(
            tree_cache=Mock(supports_mamba=Mock(return_value=False)),
            _release_aborted_request=Mock(),
            clear_pending_chunk_send=Mock(),
            waiting_queue=[],
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            processed_tokens_counter=0,
        )
        with (
            patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req"),
            patch("sglang.srt.disaggregation.prefill.release_kv_cache"),
        ):
            SchedulerDisaggregationPrefillMixin.optimistic_release_and_requeue(
                scheduler, req
            )
        self.assertEqual(scheduler.waiting_queue, [req])
        self.assertEqual(scheduler.disagg_prefill_bootstrap_queue.queue, [])
        self.assertEqual(req.prefill_attempt_count, 2)
        self.assertEqual(list(req.output_ids), [])
        self.assertTrue(req.pending_bootstrap)
        self.assertEqual(req.storage_prefetch_retry_attempts, 0)
        self.assertEqual(req.storage_prefetch_last_match_len, 16)

    def test_rebootstrap_dispatch_precedes_ready_without_allocating(self):
        receiver = self.receiver()
        receiver.init = Mock()
        req = SimpleNamespace(build_rebootstrap_payload=lambda: {"rid": "resume"})
        entry = DecodeRequest(req=req, kv_receiver=receiver, is_rebootstrap=True)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        dispatched = []
        queue.kv_manager = SimpleNamespace(
            submit_prefill_recompute=lambda r, payload: dispatched.append((r, payload))
        )
        queue._init_receiver(entry, 0)
        self.assertEqual(dispatched, [(receiver, {"rid": "resume"})])
        self.assertFalse(receiver._metadata_sent)
        self.assertFalse(entry.waiting_for_input)


if __name__ == "__main__":
    unittest.main()
