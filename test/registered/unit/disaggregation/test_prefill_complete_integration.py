"""Exercise the allocation policy through production transport and queue methods."""

import threading
import time
import unittest
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.bootstrap import DeferredBootstrap
from sglang.srt.disaggregation.common.conn import CommonKVReceiver, PrefillServerInfo
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
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
            {
                "disaggregation_mode": "decode",
                "disaggregation_decode_host_receive_threshold": 0.5,
            },
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
        mgr.kv_args = SimpleNamespace(aux_data_ptrs=[], aux_item_lens=[])
        mgr.request_status = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.waiting_timeout = 5
        mgr.bootstrap_timeout = 3
        mgr.req_to_decode_prefix_len = {}
        mgr.transfer_infos = {}
        mgr._deferred_ack_targets = {}
        mgr._deferred_ack_poisoned_rooms = set()
        mgr.required_prefill_response_num_table = {}
        mgr.prefill_response_tracker = defaultdict(set)
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.defer_decode_allocation = True
        mgr.deferred_bootstrap = DeferredBootstrap(8) if ip == "prefill" else None
        mgr.enable_trace = False
        mgr.enable_staging = False
        mgr.is_dummy_cp_rank = False
        mgr._staging_outstanding = {}
        mgr._prefill_unique_rank = Mock(return_value=0)
        mgr.get_session_id = Mock(return_value="decode-session")
        mgr._send_multipart_locked = lambda endpoint, msg, **kw: self.wire.append(
            (endpoint, msg)
        )
        return mgr

    def flush(self):
        while self.wire:
            endpoint, msg = self.wire.popleft()
            if endpoint == "tcp://prefill:1":
                self.prefill._handle_bootstrap_message(msg)
            else:
                self.assertEqual(endpoint, "tcp://decode:2")
                room, status, rank, reason = self.decode.parse_kv_status_message(msg)
                self.decode.apply_prefill_status(
                    bootstrap_room=room,
                    status=status,
                    prefill_rank=rank,
                    failure_reason=reason,
                )

    def sender(self, room=1):
        return MooncakeKVSender(self.prefill, "prefill:8998", room)

    def receiver(self, room=1, policy="prefill_complete"):
        receiver = MooncakeKVReceiver(self.decode, "prefill:8998", room)
        receiver.bootstrap_infos = [{"rank_ip": "prefill", "rank_port": 1}]
        self.decode.prefill_info_table = {
            receiver.bootstrap_addr: PrefillServerInfo(
                attn_tp_size=1,
                attn_cp_size=1,
                dp_size=1,
                pp_size=1,
                page_size=1,
                kv_cache_dtype=None,
                follow_bootstrap_room=True,
                decode_allocation_policy=policy,
                target_tp_rank=0,
                target_tp_ranks=[0],
                target_cp_ranks=[0],
                target_pp_ranks=[0],
                required_dst_info_num=1,
                required_prefill_response_num=1,
            )
        }
        sock = SimpleNamespace(
            send_multipart=lambda msg: self.wire.append(("tcp://prefill:1", msg))
        )
        receiver._connect_to_bootstrap_server = Mock(
            return_value=(sock, threading.Lock())
        )
        with patch.object(receiver, "_setup_bootstrap_infos"):
            receiver.init(0)
        return receiver

    def test_receiver_hides_allocatable_status_until_source_finishes(self):
        sender, receiver = self.sender(), self.receiver()
        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        sender.mark_prefill_complete()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)

    def test_early_receiver_stays_allocatable_without_control_registration(self):
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="early"
        )
        override.install()
        self.addCleanup(override.restore)
        self.decode.defer_decode_allocation = False
        receiver = self.receiver(policy="early")
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        self.assertIsNone(receiver.init_time)
        self.assertEqual(list(self.wire), [])

    def test_shared_receive_wrapper_preserves_native_abort_and_metadata(self):
        sender = self.sender()
        bootstrap = [b"BOOTSTRAP", b"1", b"decode", b"2"]
        abort = [b"ABORT", b"1", b"decode", b"2"]
        metadata = [b"1", b"decode", b"2", b"session", b"indices"]
        sock = Mock(recv_multipart=Mock(side_effect=[bootstrap, abort, metadata]))
        recv = self.prefill._make_worker_recv(sock)
        self.assertIsNone(recv())
        self.assertEqual(recv(), abort)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(recv(), metadata)

    def test_repeated_transfers_reuse_room_after_success_cleanup(self):
        retired = []
        for receiver_first in (True, False, True):
            with self.subTest(receiver_first=receiver_first, iteration=len(retired)):
                if receiver_first:
                    receiver = self.receiver()
                    self.flush()
                    sender = self.sender()
                else:
                    sender, receiver = self.sender(), self.receiver()
                    self.flush()
                self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
                for old_sender, old_receiver in retired:
                    old_sender.clear()
                    old_sender.abort()
                    old_receiver.clear()
                    self.decode._fail_prefill_recompute(old_receiver, "late HTTP error")
                self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
                self.assertNotIn(1, self.decode.failure_records)
                sender.mark_prefill_complete()
                self.flush()
                self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
                # Duplicate control registration must not enqueue another ready
                # message that could outlive the transfer's completion.
                self.prefill._handle_bootstrap_message(
                    [b"BOOTSTRAP", b"1", b"decode", b"2"]
                )
                self.assertEqual(list(self.wire), [])
                self.prefill.conclude_transfer(bootstrap_room=1, status=KVPoll.Success)
                self.flush()
                self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
                self.assertEqual(sender.poll(), KVPoll.Success)
                sender.clear()
                self.flush()
                self.assertEqual(receiver.poll(), KVPoll.Success)
                receiver.clear()
                self.assertNotIn(1, self.prefill.deferred_bootstrap.rooms)
                self.assertNotIn(1, self.prefill.request_status)
                self.assertEqual(list(self.wire), [])
                retired.append((sender, receiver))

    def test_completion_before_receiver_uses_same_bootstrap(self):
        sender = self.sender()
        sender.mark_prefill_complete()
        receiver = self.receiver()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)

    def test_receiver_before_sender_uses_no_subscription_retries(self):
        receiver = self.receiver()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        sender = self.sender()
        sender.mark_prefill_complete()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        self.assertEqual(list(self.wire), [])

    def test_sender_abort_fails_decoder_before_metadata(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.abort()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        sender.clear()
        self.assertIsNone(self.prefill.deferred_bootstrap.rooms[1].owner)

    def test_sender_failure_before_receiver_is_not_lost(self):
        sender = self.sender()
        sender.abort()
        sender.clear()
        receiver = self.receiver()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Failed)

    def test_receiver_abort_notifies_source_using_normal_abort(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        receiver.abort()
        self.assertEqual(self.wire[0][1][0], b"ABORT")
        self.flush()
        self.assertEqual(sender.poll(), KVPoll.Failed)
        sender.clear()
        receiver.clear()
        self.assertNotIn(1, self.decode.request_status)

    def test_cancel_before_sender_creation_is_not_lost(self):
        receiver = self.receiver()
        receiver.abort()
        self.flush()
        sender = self.sender()
        self.assertEqual(sender.poll(), KVPoll.Failed)
        sender.clear()
        self.assertNotIn(1, self.prefill.request_status)

    def test_readiness_timeout_aborts_before_allocation(self):
        sender, receiver = self.sender(), self.receiver()
        receiver._prefill_wait_start = 0
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.flush()
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_ready_observed_after_compute_deadline_wins(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        receiver._prefill_wait_start = 0
        sender.mark_prefill_complete()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        self.assertIsNone(receiver._prefill_wait_start)

    def test_admission_timeout_excludes_prefill_compute(self):
        sender, receiver = self.sender(), self.receiver()
        sender.init_time = 0
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        self.flush()
        sender.mark_prefill_complete()
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        sender._prefill_complete_time = 0
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Failed)

    def test_failure_wins_over_late_ready(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.mark_prefill_complete()
        sender.abort()
        # Deliver failure before the already-emitted ready status.
        self.wire.reverse()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Failed)

    def test_duplicate_sender_does_not_crash_or_clear_owner(self):
        sender, receiver = self.sender(), self.receiver()
        duplicate = self.sender()
        self.assertEqual(duplicate.poll(), KVPoll.Failed)
        duplicate.clear()
        self.assertIs(self.prefill.deferred_bootstrap.rooms[1].owner, sender)
        sender.clear()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Failed)

    def test_duplicate_receiver_does_not_crash_or_clear_owner(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        duplicate = self.receiver()
        self.assertEqual(duplicate.poll(), KVPoll.Failed)
        duplicate.clear()
        self.assertIn(1, self.decode.request_status)
        self.assertIn(1, self.decode.addr_to_rooms_tracker[receiver.bootstrap_addr])
        receiver.clear()
        self.flush()
        self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_rejected_sender_stays_failed_after_original_clears(self):
        original = self.sender()
        duplicate = self.sender()
        original.clear()
        self.assertEqual(duplicate.poll(), KVPoll.Failed)
        duplicate.clear()

    def test_fake_warmup_sender_never_registers_deferred_bootstrap(self):
        sender = FakeKVSender(self.prefill, "fake:0", 0)
        sender.init(0)
        sender.mark_prefill_complete()
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        sender.send([])
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(self.prefill.deferred_bootstrap.rooms, {})
        self.assertEqual(list(self.wire), [])

    def test_abort_after_metadata_uses_existing_drain_protocol(self):
        receiver = self.receiver()
        receiver._prefill_wait_start = None
        receiver.init_time = time.time()
        self.wire.clear()
        with patch.object(CommonKVReceiver, "_send_abort_notification") as notify:
            receiver.abort()
        notify.assert_called_once()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)

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
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=Mock(return_value=1)
        )
        queue.req_to_token_pool = SimpleNamespace(available_size=Mock(return_value=0))
        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            side_effect=lambda receivers, group: [r.poll() for r in receivers],
        ):
            self.assertEqual(queue.pop_preallocated(), ([], []))
            self.flush()
            self.assertFalse(entry.waiting_for_input)
            queue.req_to_metadata_buffer_idx_allocator.available_size.assert_not_called()
            queue.req_to_token_pool.available_size.assert_not_called()
            sender.mark_prefill_complete()
            self.flush()
            self.assertEqual(queue.pop_preallocated(), ([], []))
        self.assertTrue(entry.waiting_for_input)
        queue.req_to_metadata_buffer_idx_allocator.available_size.assert_called_once()
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
            patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache"),
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

    def test_chunked_prefill_does_not_yield_for_decode_admission(self):
        for policy in ("early", "prefill_complete"):
            for overlap in (False, True):
                with self.subTest(policy=policy, overlap=overlap):
                    override = get_context().override_server_args(
                        disaggregation_decode_allocation_policy=policy
                    )
                    override.install()
                    try:
                        req = Mock(finished_reason=None, to_finish=None)
                        scheduler = SimpleNamespace(
                            chunked_req=req,
                            tree_cache=Mock(),
                            enable_overlap=overlap,
                            check_bootstrap=Mock(return_value=False),
                            has_bootstrapped_waiting_req=Mock(return_value=True),
                            optimistic_release_and_requeue=Mock(),
                        )
                        scheduler.checkpoint_disagg_prefill = MethodType(
                            SchedulerDisaggregationPrefillMixin.checkpoint_disagg_prefill,
                            scheduler,
                        )
                        running = SimpleNamespace(batch_is_full=True)
                        with patch(
                            "sglang.srt.disaggregation.prefill.checkpoint_kv_cache"
                        ):
                            SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
                                scheduler, None, running
                            )
                        if policy == "prefill_complete":
                            self.assertIs(scheduler.chunked_req, req)
                            self.assertFalse(running.batch_is_full)
                            scheduler.optimistic_release_and_requeue.assert_not_called()
                        else:
                            self.assertIsNone(scheduler.chunked_req)
                            self.assertEqual(
                                scheduler.optimistic_release_and_requeue.call_count,
                                0 if overlap else 1,
                            )
                    finally:
                        override.restore()

    def test_completed_prefill_parks_until_normal_pointer_handshake(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.mark_prefill_complete()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        # Decode may reserve KV now, but prefill still has no destination indices.
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        req = SimpleNamespace(
            rid="parked",
            disagg_kv_sender=sender,
            pending_bootstrap=True,
            prefill_attempt_count=1,
            skip_radix_cache_insert=False,
        )
        scheduler = SchedulerDisaggregationPrefillMixin()
        scheduler.tree_cache = Mock()
        scheduler.scheduler_stage_metrics = None
        scheduler.disagg_prefill_inflight_queue = [req]
        scheduler.attn_cp_cpu_group = scheduler.attn_tp_cpu_group = object()
        scheduler.output_streamer = Mock()
        scheduler.send_kv_chunk = Mock()

        def finalize(r):
            r.pending_bootstrap = False
            return True

        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            finalize_bootstrap=Mock(side_effect=finalize)
        )
        with (
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                side_effect=lambda senders, *groups: [s.poll() for s in senders],
            ),
            patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache"),
        ):
            self.assertEqual(scheduler.process_disagg_prefill_inflight_queue(), [])
            self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
            scheduler.send_kv_chunk.assert_not_called()
            # This is the existing metadata-received transition, not readiness.
            self.prefill.update_status(1, KVPoll.WaitingForInput)
            self.assertEqual(scheduler.process_disagg_prefill_inflight_queue(), [])
            scheduler.send_kv_chunk.assert_called_once_with(req, last_chunk=True)
            self.assertFalse(req.pending_bootstrap)

    def test_clear_before_allocation_notifies_prefill(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        receiver.clear()
        self.flush()
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertNotIn(1, self.decode.request_status)

    def test_success_is_sent_once_after_all_sender_state_is_cleared(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.mark_prefill_complete()
        self.flush()
        self.prefill.conclude_transfer(bootstrap_room=1, status=KVPoll.Success)
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.prefill.req_to_decode_prefix_len[1] = 0
        self.prefill.transfer_infos[1] = {}
        send = self.prefill.send_kv_status_message
        next_transfer = []

        def success(**kwargs):
            self.assertEqual(kwargs["status"], KVPoll.Success)
            for table in (
                self.prefill.deferred_bootstrap.rooms,
                self.prefill.request_status,
                self.prefill.req_to_decode_prefix_len,
                self.prefill.transfer_infos,
            ):
                self.assertNotIn(1, table)
            send(**kwargs)
            self.flush()
            self.assertEqual(receiver.poll(), KVPoll.Success)
            receiver.clear()
            # Resume immediately, before the old clear() call even returns.
            new_sender, new_receiver = self.sender(), self.receiver()
            self.flush()
            next_transfer.extend((new_sender, new_receiver))

        self.prefill.send_kv_status_message = Mock(side_effect=success)
        sender.clear()
        sender.clear()
        self.prefill.send_kv_status_message.assert_called_once()
        new_sender, new_receiver = next_transfer
        self.assertIs(self.prefill.deferred_bootstrap.rooms[1].owner, new_sender)
        self.assertEqual(new_receiver.poll(), KVPoll.Bootstrapping)
        self.assertEqual(list(self.wire), [])

    def test_worker_cleanup_finishes_before_sender_can_report_success(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.mark_prefill_complete()
        self.flush()
        mgr = self.prefill
        mgr._staging_outstanding = defaultdict(int)
        mgr.session_lock = threading.Lock()
        mgr.failed_sessions = set()
        mgr.state_layout_rejections = {}
        mgr.state_strides_validated = set()
        mgr.enable_deferred_decode_kv_release = True
        mgr.decode_kv_args_table = {
            "session": SimpleNamespace(
                requires_dcp_relayout=False, dst_aux_ptrs=[], dst_aux_item_lens=[]
            )
        }
        mgr._get_dsa_cache_transfer_skip_flags = Mock(return_value=(False, False))
        mgr.send_aux = Mock(return_value=0)
        mgr.transfer_infos[1] = {
            "session": SimpleNamespace(
                is_dummy=False,
                mooncake_session_id="session",
                dst_kv_indices=[],
                dst_device_kv_indices=None,
                endpoint="decode",
                dst_port=2,
                room=1,
                required_dst_info_num=1,
            )
        }
        cleanup_observed = []

        def during_cleanup(room, default=None):
            # The worker has finished writes and decremented its counter, but
            # is still deleting room-indexed state. Force a scheduler poll here.
            self.assertEqual(mgr._staging_outstanding[room], 0)
            self.assertEqual(sender.poll(), KVPoll.Transferring)
            self.flush()
            self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
            cleanup_observed.append(room)

        mgr.req_to_decode_prefix_len = SimpleNamespace(pop=during_cleanup)
        chunk = TransferKVChunk(
            room=1,
            index_slice=slice(0, 0),
            prefill_kv_indices=[],
            is_last_chunk=True,
            state_indices=None,
            prefill_aux_index=0,
        )
        mgr.transfer_worker(Mock(get=Mock(side_effect=[chunk, None])), None)
        self.assertEqual(cleanup_observed, [1])
        self.assertNotIn(1, mgr._staging_outstanding)
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        mgr.req_to_decode_prefix_len = {}
        sender.clear()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Success)

    def test_early_transfer_success_does_not_wait_for_sender_cleanup(self):
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="early"
        )
        override.install()
        self.addCleanup(override.restore)
        self.prefill.defer_decode_allocation = self.decode.defer_decode_allocation = (
            False
        )
        self.prefill.deferred_bootstrap = None
        sender, receiver = self.sender(), self.receiver(policy="early")
        self.prefill.conclude_transfer(
            bootstrap_room=1, status=KVPoll.Success, targets=[("decode", 2)]
        )
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertIn(1, self.prefill.request_status)
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_recompute_http_error_after_success_before_retirement_is_ignored(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.mark_prefill_complete()
        self.flush()
        self.prefill.conclude_transfer(bootstrap_room=1, status=KVPoll.Success)
        self.assertEqual(sender.poll(), KVPoll.Success)
        sender.clear()
        self.flush()
        # Success is in the status table, but the scheduler has not polled it.
        self.assertIsNone(receiver.conclude_state)
        self.decode._fail_prefill_recompute(receiver, "late HTTP timeout")
        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertNotIn(1, self.decode.failure_records)

    def test_receiver_retirement_waits_for_recompute_error_callback(self):
        receiver = self.receiver()
        error_started = threading.Event()
        finish_error = threading.Event()
        retirement_started = threading.Event()
        record_failure = self.decode.record_failure

        def delayed_failure(room, reason):
            error_started.set()
            if not finish_error.wait(timeout=5):
                raise TimeoutError("test did not release the error callback")
            record_failure(room, reason)

        def retire():
            retirement_started.set()
            receiver.clear()

        self.decode.record_failure = delayed_failure
        with ThreadPoolExecutor(max_workers=2) as executor:
            error = executor.submit(
                self.decode._fail_prefill_recompute, receiver, "HTTP error"
            )
            try:
                self.assertTrue(error_started.wait(timeout=5))
                retirement = executor.submit(retire)
                self.assertTrue(retirement_started.wait(timeout=5))
                with self.assertRaises(TimeoutError):
                    retirement.result(timeout=0.05)
            finally:
                finish_error.set()
            error.result(timeout=5)
            retirement.result(timeout=5)
        self.assertFalse(receiver._owns_bootstrap_room)
        self.assertEqual(self.decode.failure_records[1], "HTTP error")

    def test_slow_status_delivery_does_not_hold_bootstrap_lock(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()

        def send(*args, **kwargs):
            acquired = self.prefill.deferred_bootstrap.lock.acquire(blocking=False)
            self.assertTrue(acquired)
            if acquired:
                self.prefill.deferred_bootstrap.lock.release()

        self.prefill.send_kv_status_message = send
        sender.mark_prefill_complete()
        sender.abort()
        sender.clear()

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
        self.assertFalse(entry.waiting_for_input)
        self.assertEqual(req.disagg_prefill_dp_rank, 0)

    def test_rebootstrap_dispatches_once_across_init_and_metadata(self):
        for policy in ("early", "prefill_complete"):
            with self.subTest(policy=policy):
                override = get_context().override_server_args(
                    disaggregation_decode_allocation_policy=policy
                )
                override.install()
                try:
                    receiver = Mock(conclude_state=None)
                    req = SimpleNamespace(
                        build_rebootstrap_payload=lambda: {"rid": "resume"},
                        time_stats=Mock(),
                    )
                    entry = DecodeRequest(
                        req=req, kv_receiver=receiver, is_rebootstrap=True
                    )
                    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
                    queue.kv_manager = Mock()
                    queue.req_to_metadata_buffer_idx_allocator = Mock()
                    queue._transfer_page_indices = Mock(return_value=[0])
                    queue.transfer_queue = SimpleNamespace(enable_staging=False)
                    queue._num_published_destinations = 0
                    events = Mock()
                    events.attach_mock(receiver.send_metadata, "metadata")
                    events.attach_mock(
                        queue.kv_manager.submit_prefill_recompute, "recompute"
                    )
                    queue._init_receiver(entry, 0)
                    queue._send_kv_metadata(entry, [0], 1)
                    queue.kv_manager.submit_prefill_recompute.assert_called_once_with(
                        receiver, {"rid": "resume"}
                    )
                    self.assertEqual(
                        [event[0] for event in events.mock_calls],
                        ["metadata", "recompute"]
                        if policy == "early"
                        else ["recompute", "metadata"],
                    )
                    self.assertEqual(queue._num_published_destinations, 1)
                finally:
                    override.restore()

    def test_normal_bootstrap_keeps_room_without_dispatching_recompute(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._check_if_req_exceed_kv_capacity = Mock(return_value=False)
        queue.pending_reqs = []
        req = SimpleNamespace(
            bootstrap_room=7,
            bootstrap_host="prefill",
            bootstrap_port=8998,
            disagg_prefill_dp_rank=None,
            to_finish=None,
            finished_reason=None,
        )
        receiver = Mock(conclude_state=None)
        entry = DecodeRequest(req=req, kv_receiver=receiver)
        queue._create_receiver_and_enqueue = Mock(return_value=entry)
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={
                "prefill:8998": SimpleNamespace(dp_size=1),
            },
            submit_prefill_recompute=Mock(),
        )
        queue.add(req)
        self.assertEqual(req.bootstrap_room, 7)
        receiver.init.assert_called_once_with(0)
        queue.kv_manager.submit_prefill_recompute.assert_not_called()
        self.assertFalse(entry.waiting_for_input)

    def test_rebootstrap_keeps_room_and_prefill_rank(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._check_if_req_exceed_kv_capacity = Mock(return_value=False)
        queue.queue = []
        queue.pending_reqs = []
        queue._create_receiver_and_enqueue = Mock()
        queue._init_receiver = Mock()
        req = SimpleNamespace(
            bootstrap_room=1,
            bootstrap_host="prefill",
            bootstrap_port=8998,
            disagg_prefill_dp_rank=3,
        )
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={"prefill:8998": SimpleNamespace(dp_size=4)}
        )
        queue.add(req, is_rebootstrap=True)
        self.assertEqual(req.bootstrap_room, 1)
        queue._init_receiver.assert_called_once_with(
            queue._create_receiver_and_enqueue.return_value, 3
        )
        queue.add(req, is_rebootstrap=True)
        self.assertEqual(req.bootstrap_room, 1)
        self.assertEqual(req.disagg_prefill_dp_rank, 3)

    def test_receiver_clear_preserves_failure_reason(self):
        receiver = self.receiver()
        self.decode.record_failure(1, "original failure")
        self.decode.update_status(1, KVPoll.Failed)
        receiver.clear()
        self.assertEqual(self.decode.failure_records[1], "original failure")


if __name__ == "__main__":
    unittest.main()
