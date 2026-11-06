"""Exercise the allocation policy through production transport and queue methods."""

import threading
import time
import unittest
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import torch

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.arg_groups.serving_hook import handle_other_validations
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.bootstrap import (
    DecodeHandshake,
    DeferredBootstrap,
)
from sglang.srt.disaggregation.common.conn import (
    CommonKVReceiver,
    KVTransferError,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.decode_readiness import DecodeReadinessPacer
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import poll_and_all_reduce_prefill_pp_sections
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import (
    get_context,
    get_parallel,
    publish,
    reset_context,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq, decode_queue
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# The decode manager's control endpoint in TestAllocationIntegration.
DECODE = ("decode", 2)


class _BroadcastGroup:
    """Stands in for the PP group's object broadcast in one process."""

    def __init__(self):
        self.obj = None

    def broadcast_object(self, obj, src=0):
        if obj is not None:
            self.obj = obj
        return self.obj


class TestAllocationConfiguration(CustomTestCase):
    def test_pp_prefill_keeps_optimistic_prefill_only_for_deferred_policies(self):
        for policy, attempts in (
            ("prefill_complete", 1),
            ("just_in_time", 1),
            ("early", 0),
        ):
            with self.subTest(policy=policy):
                args = ServerArgs(
                    model_path="dummy",
                    disaggregation_mode="prefill",
                    disaggregation_decode_allocation_policy=policy,
                    optimistic_prefill_attempts=1,
                    pp_size=4,
                )
                handle_pd_disaggregation(args)
                handle_other_validations(args)
                self.assertEqual(
                    resolving_view(args).optimistic_prefill_attempts, attempts
                )

    def test_unsupported_modes_fail_explicitly(self):
        combinations = [
            {"disaggregation_mode": "null"},
            {"disaggregation_transfer_backend": "nixl"},
            {"tp_size": 2},
            # Rejected here, before the topology validators that would raise a
            # bare assertion or divide by zero.
            {"tp_size": 3, "attn_dp_size": 2},
            {"attn_dp_size": 0},
            {"disaggregation_mode": "decode", "pp_size": 2},
            {"attn_cp_size": 2},
            {"dcp_size": 2},
            {"enable_hisparse": True},
            {
                "disaggregation_mode": "decode",
                "disaggregation_decode_host_receive_threshold": 0.5,
            },
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
        for policy in ("prefill_complete", "just_in_time"):
            for changed in combinations:
                with self.subTest(policy=policy, changed=changed):
                    values = dict(
                        model_path="dummy",
                        disaggregation_mode="prefill",
                        disaggregation_decode_allocation_policy=policy,
                    )
                    values.update(changed)
                    with self.assertRaisesRegex(ValueError, policy):
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
        mgr.customized_info_enabled = False
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
            ip, port = endpoint.removeprefix("tcp://").split(":")
            mgr = self.endpoints[(ip, int(port))]
            if mgr is not self.decode:
                mgr._handle_bootstrap_message(msg)
                continue
            room, status, rank, reason = self.decode.parse_kv_status_message(msg)
            self.decode.apply_prefill_status(
                bootstrap_room=room,
                status=status,
                prefill_rank=rank,
                failure_reason=reason,
            )

    def pp_stages(self, pp_size):
        """Prefill managers of one PP server; stage statuses replicate to stage 0."""
        group = _BroadcastGroup()
        stages = [self.prefill]
        for pp_rank in range(1, pp_size):
            stages.append(self.manager("prefill", 1 + 2 * pp_rank))
        for pp_rank, mgr in enumerate(stages):
            mgr.request_status = PPConsensusStore(pp_size, pp_rank, group)
            self.addCleanup(mgr.request_status.close)
            mgr._prefill_unique_rank = Mock(return_value=pp_rank)
            self.endpoints[(mgr.local_ip, mgr.rank_port)] = mgr
        return stages

    def wait_until(self, condition):
        deadline = time.monotonic() + 5
        while not condition():
            self.assertLess(time.monotonic(), deadline, "condition not met")
            time.sleep(0.01)

    def sender(self, room=1):
        return MooncakeKVSender(self.prefill, "prefill:8998", room)

    def receiver(self, room=1, policy="prefill_complete", stages=None):
        stages = stages or [self.prefill]
        receiver = MooncakeKVReceiver(self.decode, "prefill:8998", room)
        receiver.bootstrap_infos = [
            {"rank_ip": mgr.local_ip, "rank_port": mgr.rank_port} for mgr in stages
        ]
        self.decode.prefill_info_table = {
            receiver.bootstrap_addr: PrefillServerInfo(
                attn_tp_size=1,
                attn_cp_size=1,
                dp_size=1,
                pp_size=len(stages),
                page_size=1,
                kv_cache_dtype=None,
                follow_bootstrap_room=True,
                decode_allocation_policy=policy,
                target_tp_rank=0,
                target_tp_ranks=[0],
                target_cp_ranks=[0],
                target_pp_ranks=list(range(len(stages))),
                required_dst_info_num=1,
                required_prefill_response_num=len(stages),
            )
        }

        def connect(info):
            endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
            sock = SimpleNamespace(
                send_multipart=lambda msg: self.wire.append((endpoint, msg))
            )
            return sock, threading.Lock()

        receiver._connect_to_bootstrap_server = Mock(side_effect=connect)
        with patch.object(receiver, "_setup_bootstrap_infos"):
            receiver.init(0)
        return receiver

    def test_receiver_hides_allocatable_status_until_source_finishes(self):
        sender, receiver = self.sender(), self.receiver()
        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
        sender.signal_decode_readiness()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)

    def test_handshake_runs_from_readiness_message_to_first_destination(self):
        """just_in_time estimates the handshake from these transport timestamps,
        not from when the scheduler next polls the request. Readiness that waits
        for decode to register is sent, and timed, on registration: signaling
        earlier cannot shorten that wait."""
        now = [0.0]
        with patch(
            "sglang.srt.disaggregation.common.bootstrap.time.monotonic",
            side_effect=lambda: now[0],
        ):
            for room, registered_first, handshake in (
                (1, True, 0.75),
                (2, False, 0.25),
            ):
                with self.subTest(registered_first=registered_first):
                    now[0] = 0.0
                    sender = self.sender(room)
                    if registered_first:
                        receiver = self.receiver(room)
                        self.flush()
                    now[0] = 10.0
                    sender.signal_decode_readiness()
                    self.flush()
                    if not registered_first:
                        now[0] = 10.5
                        receiver = self.receiver(room)
                        self.flush()
                    self.assertIsNone(sender.decode_handshake())
                    now[0] = 10.75
                    self.prefill.update_status(room, KVPoll.WaitingForInput)
                    now[0] = 11.5
                    self.prefill.update_status(room, KVPoll.WaitingForInput)
                    # A duplicate BOOTSTRAP does not move the registration time.
                    self.prefill.deferred_bootstrap.register(room, ("decode", 2))
                    self.assertEqual(
                        sender.decode_handshake(),
                        DecodeHandshake(
                            endpoint=("decode", 2),
                            seconds=handshake,
                            registered_at=0.0 if registered_first else 10.5,
                        ),
                    )
                    self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)

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
                sender.signal_decode_readiness()
                self.flush()
                self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
                # Duplicate control registration must not enqueue another ready
                # message that could outlive the transfer's completion.
                self.prefill._handle_bootstrap_message(
                    [b"BOOTSTRAP", b"1", b"decode", b"2"]
                )
                self.assertEqual(list(self.wire), [])
                self.prefill.conclude_transfer(
                    bootstrap_room=1, status=KVPoll.Success, targets=[DECODE]
                )
                self.flush()
                # The transfer worker's Success reaches decode before release.
                self.assertEqual(receiver.poll(), KVPoll.Success)
                self.assertEqual(sender.poll(), KVPoll.Success)
                sender.clear()
                receiver.clear()
                self.assertNotIn(1, self.prefill.deferred_bootstrap.rooms)
                self.assertNotIn(1, self.prefill.request_status)
                self.assertEqual(list(self.wire), [])
                retired.append((sender, receiver))

    def test_sender_abort_fails_decoder_before_metadata(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.abort()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        sender.clear()
        self.assertIsNone(self.prefill.deferred_bootstrap.rooms[1].owner)

    def test_failure_while_decode_holds_mid_prefill_destination(self):
        """Under just_in_time decode can hold a request's destination while it
        still prefills, at cold start for its whole prefill. A failure on either
        side then must reach the other, or decode keeps the reservation until
        its waiting timeout and prefill computes a request nobody receives."""
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="just_in_time"
        )
        override.install()
        self.addCleanup(override.restore)

        def destination_held(room):
            sender = self.sender(room)
            receiver = self.receiver(room, policy="just_in_time")
            self.flush()
            sender.signal_decode_readiness()
            self.flush()
            self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
            # Decode's destination arrives before any chunk was sent.
            self.prefill.update_status(room, KVPoll.WaitingForInput)
            return sender, receiver

        with self.subTest("prefill aborts"):
            sender, receiver = destination_held(1)
            sender.abort()
            self.flush()
            self.assertEqual(receiver.poll(), KVPoll.Failed)
            sender.clear()
            self.assertIsNone(self.prefill.deferred_bootstrap.rooms[1].owner)
        with self.subTest("decode aborts or times out"):
            sender, _ = destination_held(2)
            self.prefill._handle_bootstrap_message([b"ABORT", b"2", b"decode", b"2"])
            self.assertEqual(sender.poll(), KVPoll.Failed)

    def test_fake_warmup_sender_never_registers_deferred_bootstrap(self):
        sender = FakeKVSender(self.prefill, "fake:0", 0)
        sender.init(0)
        sender.signal_decode_readiness()
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        sender.send([])
        self.assertEqual(sender.poll(), KVPoll.Success)
        self.assertEqual(self.prefill.deferred_bootstrap.rooms, {})
        self.assertEqual(list(self.wire), [])

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
        req = FakeReq(
            rid="req",
            bootstrap_room=1,
            finished_reason=None,
            return_logprob=False,
            time_stats=MagicMock(),
        )
        entry = DecodeRequest(req=req, kv_receiver=receiver)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = decode_queue(ListKind.PD_PREALLOC, [entry])
        queue.pending_reqs = []
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
            sender.signal_decode_readiness()
            self.flush()
            self.assertEqual(queue.pop_preallocated(), ([], []))
        self.assertTrue(entry.waiting_for_input)
        queue.req_to_metadata_buffer_idx_allocator.available_size.assert_called_once()
        queue.req_to_token_pool.available_size.assert_called_once()
        self.assertEqual(list(queue.queue.items()), [entry])

    def test_exhausted_optimistic_attempt_follows_allocation_policy(self):
        override = get_context().override_server_args(
            optimistic_prefill_attempts=1,
            disaggregation_decode_allocation_policy="prefill_complete",
        )
        override.install()
        self.addCleanup(override.restore)
        req = FakeReq(
            rid="retry",
            cache_request_handle=CacheRequestHandle(rid="retry", attempt_id=0),
            prefill_attempt_count=1,
            output_ids=[42],
            skip_radix_cache_insert=False,
            kv=SimpleNamespace(cache_protected_len=16),
            full_untruncated_fill_ids=list(range(32)),
            _compute_max_prefix_len=Mock(return_value=31),
            storage_prefetch_retry_attempts=3,
            storage_prefetch_last_match_len=8,
            metadata_buffer_index=0,
            reset_for_retract=Mock(),
            advance_cache_request_handle=Mock(),
            time_stats=MagicMock(),
        )
        scheduler = SimpleNamespace(
            tree_cache=Mock(supports_mamba=Mock(return_value=False)),
            clear_pending_chunk_send=Mock(),
            prefill_launched_reqs=KeyedReqList(ListKind.PREFILL_LAUNCHED),
            waiting_queue=IndexedReqList(ListKind.WAITING),
            disagg_prefill_bootstrap_queue=SimpleNamespace(
                queue=IndexedReqList(ListKind.PD_BOOTSTRAP)
            ),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            processed_tokens_counter=0,
            req_to_metadata_buffer_idx_allocator=Mock(),
        )
        launched = scheduler.prefill_launched_reqs
        with (
            patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache"),
            patch("sglang.srt.disaggregation.prefill.release_kv_cache"),
        ):
            launched.join([req])
            SchedulerDisaggregationPrefillMixin.optimistic_release_and_requeue(
                scheduler, req, launched
            )
            self.assertEqual(list(scheduler.waiting_queue), [req])
            self.assertEqual(list(launched), [])
            self.assertEqual(list(scheduler.disagg_prefill_bootstrap_queue.queue), [])
            self.assertEqual(req.prefill_attempt_count, 2)
            self.assertEqual(list(req.output_ids), [])
            self.assertTrue(req.pending_bootstrap)
            self.assertEqual(req.storage_prefetch_retry_attempts, 0)
            self.assertEqual(req.storage_prefetch_last_match_len, 16)
            scheduler.req_to_metadata_buffer_idx_allocator.free.assert_not_called()

            with get_context().override_server_args(
                disaggregation_decode_allocation_policy="early"
            ):
                scheduler.waiting_queue.clear()
                launched.join([req])
                SchedulerDisaggregationPrefillMixin.optimistic_release_and_requeue(
                    scheduler, req, launched
                )
            self.assertEqual(list(scheduler.waiting_queue), [])
            self.assertEqual(list(launched), [])
            self.assertEqual(
                list(scheduler.disagg_prefill_bootstrap_queue.queue), [req]
            )
            self.assertEqual(req.metadata_buffer_index, -1)
            scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(
                0
            )

    def test_requeued_optimistic_request_keeps_multimodal_inputs(self):
        """A yielded optimistic prefill is retried, so it must keep its media.

        The requeue used to go through the scheduler's abort release, which
        also drops the request's multimodal inputs. The retry then prefilled
        image placeholder ids with no image features to embed and crashed in
        the embedding lookup. The requeue must still finish the cache attempt.
        """
        override = get_context().override_server_args(optimistic_prefill_attempts=3)
        override.install()
        self.addCleanup(override.restore)

        class _Req:
            release_mm_inputs_on_abort = Req.release_mm_inputs_on_abort

        req = _Req()
        req.rid = "retry-mm"
        req.cache_request_handle = CacheRequestHandle(rid="retry-mm", attempt_id=0)
        req.session = None
        req.multimodal_inputs = Mock()
        req.prefill_attempt_count = 0
        req.output_ids = [42]
        req.skip_radix_cache_insert = False
        req.kv = SimpleNamespace(cache_protected_len=16)
        req.full_untruncated_fill_ids = list(range(32))
        req._compute_max_prefix_len = Mock(return_value=31)
        req.storage_prefetch_retry_attempts = 0
        req.storage_prefetch_last_match_len = None
        req.reset_for_retract = Mock()
        req.advance_cache_request_handle = Mock()
        req.time_stats = MagicMock()
        scheduler = SimpleNamespace(
            tree_cache=Mock(supports_mamba=Mock(return_value=False)),
            clear_pending_chunk_send=Mock(),
            prefill_launched_reqs=Mock(),
            waiting_queue=IndexedReqList(ListKind.WAITING),
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            processed_tokens_counter=0,
        )
        scheduler.release_aborted_request = MethodType(
            Scheduler.release_aborted_request, scheduler
        )
        with (
            patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache"),
            patch("sglang.srt.disaggregation.prefill.release_kv_cache"),
        ):
            SchedulerDisaggregationPrefillMixin.optimistic_release_and_requeue(
                scheduler, req, scheduler.prefill_launched_reqs
            )

        self.assertEqual(list(scheduler.waiting_queue), [req])
        self.assertIsNotNone(req.multimodal_inputs)
        req.multimodal_inputs.release_features.assert_not_called()
        scheduler.tree_cache.finish.assert_called_once_with(
            req.cache_request_handle, CacheRequestOutcome.ABORT
        )

    def test_chunked_prefill_does_not_yield_for_decode_admission(self):
        for policy in ("early", "prefill_complete", "just_in_time"):
            for inflight_steps in (0, 1):
                with self.subTest(policy=policy, inflight_steps=inflight_steps):
                    override = get_context().override_server_args(
                        disaggregation_decode_allocation_policy=policy
                    )
                    override.install()
                    try:
                        req = Mock(
                            finished_reason=None,
                            to_finish=None,
                            inflight_steps=inflight_steps,
                        )
                        scheduler = SimpleNamespace(
                            chunked_req=req,
                            tree_cache=Mock(),
                            check_bootstrap=Mock(return_value=False),
                            has_bootstrapped_waiting_req=Mock(return_value=True),
                            optimistic_release_and_requeue=Mock(),
                            prefill_launched_reqs=KeyedReqList(
                                ListKind.PREFILL_LAUNCHED
                            ),
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
                        if policy != "early":
                            self.assertIs(scheduler.chunked_req, req)
                            self.assertFalse(running.batch_is_full)
                            scheduler.optimistic_release_and_requeue.assert_not_called()
                        else:
                            self.assertIsNone(scheduler.chunked_req)
                            # With a step in flight, it is requeued once processed.
                            launched = scheduler.prefill_launched_reqs
                            self.assertEqual(
                                scheduler.optimistic_release_and_requeue.call_args_list,
                                [] if inflight_steps else [call(req, launched)],
                            )
                            self.assertEqual(list(launched), [req])
                    finally:
                        override.restore()

    def test_yielded_request_requeues_after_its_chunk_once_bootstrap_completes(self):
        # The yield is decided before its chunk in flight is processed; a bootstrap
        # completing in between must not leave the request without a next holder.
        req = Req("yielded", "", list(range(12)), SamplingParams())
        req.extend_end = 4
        req.pending_bootstrap = False
        batch = SimpleNamespace(
            reqs=[req],
            chunked_req=req,
            spec_info=None,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([7]),
            next_draft_input=None,
            copy_done=None,
        )
        scheduler = SimpleNamespace(
            chunked_req=None,
            batch_result_processor=SimpleNamespace(
                snapshot_auxiliary_output_starts=Mock(return_value=[]),
                move_logprobs_to_cpu=Mock(),
                consume_auxiliary_output=Mock(),
            ),
            spec_algorithm=SimpleNamespace(is_eagle=lambda: False),
            optimistic_release_and_requeue=Mock(),
            prefill_launched_reqs=KeyedReqList(ListKind.PREFILL_LAUNCHED),
            metrics_reporter=SimpleNamespace(report_prefill_stats=Mock()),
            maybe_send_health_check_signal=Mock(),
        )

        SchedulerDisaggregationPrefillMixin.process_batch_result_disagg_prefill(
            scheduler, batch, result
        )

        scheduler.optimistic_release_and_requeue.assert_called_once_with(
            req, scheduler.prefill_launched_reqs
        )

    def inflight_scheduler(self, reqs):
        scheduler = SchedulerDisaggregationPrefillMixin()
        scheduler.tree_cache = Mock()
        scheduler.scheduler_stage_metrics = None
        scheduler.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
        scheduler.disagg_prefill_transfer_queue.join(reqs)
        scheduler.attn_cp_cpu_group = scheduler.attn_tp_cpu_group = object()
        scheduler.output_streamer = Mock()
        scheduler.send_kv_chunk = Mock()
        enter_scope(
            self,
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                side_effect=lambda senders, *groups: [s.poll() for s in senders],
            ),
        )
        enter_scope(
            self, patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
        )
        return scheduler

    def test_completed_prefill_parks_until_normal_pointer_handshake(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.signal_decode_readiness()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        # Decode may reserve KV now, but prefill still has no destination indices.
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        req = FakeReq(
            rid="parked",
            disagg_kv_sender=sender,
            pending_bootstrap=True,
            prefill_attempt_count=1,
            skip_radix_cache_insert=False,
        )
        scheduler = self.inflight_scheduler([req])
        scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 0)
        scheduler.waiting_queue = [Mock()]

        def finalize(r):
            r.pending_bootstrap = False
            return True

        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            finalize_bootstrap=Mock(side_effect=finalize)
        )
        self.assertEqual(scheduler.process_disagg_prefill_transfer_queue(), [])
        self.assertEqual(list(scheduler.disagg_prefill_transfer_queue), [req])
        scheduler.send_kv_chunk.assert_not_called()
        # This is the existing metadata-received transition, not readiness.
        self.prefill.update_status(1, KVPoll.WaitingForInput)
        self.assertEqual(scheduler.process_disagg_prefill_transfer_queue(), [])
        scheduler.send_kv_chunk.assert_called_once_with(req, last_chunk=True)
        self.assertFalse(req.pending_bootstrap)

    def test_parked_prefill_reclaims_only_one_slot(self):
        reqs = [Req(str(i), "", [1], SamplingParams()) for i in range(2)]
        for i, req in enumerate(reqs):
            req.pending_bootstrap = True
            req.disagg_kv_sender = self.sender(room=i + 1)
        scheduler = self.inflight_scheduler(reqs)
        pool = scheduler.req_to_token_pool = ReqToTokenPool(2, 2, "cpu", False)
        pool.alloc(reqs)
        scheduler.waiting_queue = []
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.optimistic_release_and_requeue = Mock(
            side_effect=lambda req, src: pool.free(req)
        )
        with get_context().override_server_args(
            disaggregation_decode_allocation_policy="early"
        ):
            scheduler.process_disagg_prefill_transfer_queue()
            self.assertEqual(list(scheduler.disagg_prefill_transfer_queue), reqs)

            scheduler.waiting_queue.append(object())
            scheduler.process_disagg_prefill_transfer_queue()
            self.assertEqual(list(scheduler.disagg_prefill_transfer_queue), reqs[1:])
            self.assertEqual(pool.available_size(), 1)

            scheduler.process_disagg_prefill_transfer_queue()
            scheduler.optimistic_release_and_requeue.assert_called_once_with(
                reqs[0], scheduler.disagg_prefill_transfer_queue
            )

    def test_pp_prefill_room_succeeds_only_after_every_stage(self):
        stages = self.pp_stages(2)
        senders = [MooncakeKVSender(mgr, "prefill:8998", 1) for mgr in stages]
        receiver = self.receiver(stages=stages)
        self.flush()
        # Stage 0 finishes the last chunk first; its readiness alone admits the room.
        senders[0].signal_decode_readiness()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        stages[0].conclude_transfer(
            bootstrap_room=1, status=KVPoll.Success, targets=[DECODE]
        )
        senders[0].clear()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        # The last stage's readiness can trail stage 0's Success, never its own.
        senders[1].signal_decode_readiness()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        stages[1].conclude_transfer(
            bootstrap_room=1, status=KVPoll.Success, targets=[DECODE]
        )
        senders[1].clear()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Success)

    def mock_pp_sync(self):
        """Every stage's PP poll sync returns PP0's last payload; self.pp_syncs
        records the syncing ranks."""
        polls_from_stage0 = []
        self.pp_syncs = []

        def sync_polls(polls, pp_group, pp_rank, pp_size, work_list):
            self.pp_syncs.append(pp_rank)
            if pp_rank == 0:
                polls_from_stage0.append(polls)
            return polls_from_stage0[-1]

        enter_scope(
            self,
            patch(
                "sglang.srt.disaggregation.utils.pp_sync_polls", side_effect=sync_polls
            ),
        )
        enter_scope(
            self,
            patch(
                "sglang.srt.disaggregation.utils._all_reduce_polls",
                side_effect=lambda polls, group: polls,
            ),
        )

    def pp_parked_schedulers(self, senders, queued_senders=None, waiting=False):
        """One scheduler per PP stage, each holding the same pending request,
        parked or, if ``waiting``, in the waiting queue, and, if given, the same
        request in the bootstrap queue."""
        self.mock_pp_sync()

        def finalize(req):
            req.pending_bootstrap = False
            return True

        schedulers = []
        for pp_rank, sender in enumerate(senders):
            req = FakeReq(
                rid="pending",
                disagg_kv_sender=sender,
                pending_bootstrap=True,
                to_finish=None,
                finished_reason=None,
            )
            scheduler = self.inflight_scheduler([] if waiting else [req])
            scheduler.pp_group = None
            scheduler.waiting_queue = IndexedReqList(ListKind.WAITING)
            if waiting:
                scheduler.waiting_queue.join([req])
            scheduler.handle_bootstrap_failure = Mock()
            queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
            queue.queue = IndexedReqList(ListKind.PD_BOOTSTRAP)
            if queued_senders is not None:
                queue.queue.join(
                    [
                        FakeReq(
                            rid="queued",
                            disagg_kv_sender=queued_senders[pp_rank],
                            pending_bootstrap=True,
                            to_finish=None,
                            finished_reason=None,
                            prefill_attempt_count=0,
                            is_retracted=False,
                            metadata_buffer_index=0,
                            time_stats=MagicMock(),
                        )
                    ]
                )
            queue.pp_rank, queue.pp_size = pp_rank, len(senders)
            queue.pp_poll_sync_work_list = []
            queue.room_waiters = IndexedReqList(ListKind.PD_ROOM_WAITERS)
            scheduler.processed_tokens_counter = 0
            queue.scheduler = scheduler
            queue.scheduler_stage_metrics = None
            queue.finalize_bootstrap = Mock(side_effect=finalize)
            scheduler.disagg_prefill_bootstrap_queue = queue
            schedulers.append(scheduler)
        return schedulers

    def pp_step(self, schedulers):
        """One PP loop iteration's bootstrap step, in PP rank order."""
        syncs = []
        for scheduler in schedulers:
            self.pp_syncs.clear()
            SchedulerPPMixin.process_bootstrapped_queue(scheduler)
            syncs.append(len(self.pp_syncs))
        # A stage's send of a poll sync completes only once the next stage
        # receives it in the same iteration, so every stage must sync as often
        # as PP0, and at most once.
        self.assertIn(syncs, ([0] * len(schedulers), [1] * len(schedulers)))

    def test_pp_stages_finalize_parked_prefill_on_consensus_only(self):
        """Finalizing re-checkpoints the radix cache, so a stage acting on its own
        poll would schedule against a different cache than the other stages."""
        stages = self.pp_stages(2)
        senders = [MooncakeKVSender(mgr, "prefill:8998", 1) for mgr in stages]
        queued = [MooncakeKVSender(mgr, "prefill:8998", 2) for mgr in stages]
        schedulers = self.pp_parked_schedulers(senders, queued_senders=queued)
        reqs = [scheduler.disagg_prefill_transfer_queue[0] for scheduler in schedulers]
        # Decode destinations reach the last stage first.
        stages[1].update_status(1, KVPoll.WaitingForInput)
        self.wait_until(
            lambda: (
                stages[0].request_status.collect(1)
                == [KVPoll.Bootstrapping, KVPoll.WaitingForInput]
            )
        )
        self.pp_step(schedulers)
        for scheduler, req in zip(schedulers, reqs):
            scheduler.send_kv_chunk.assert_not_called()
            self.assertTrue(req.pending_bootstrap)
            # The queued request is admitted on the same shared poll.
            self.assertEqual([r.rid for r in scheduler.waiting_queue], ["queued"])
            self.assertEqual(list(scheduler.disagg_prefill_bootstrap_queue.queue), [])
        stages[0].update_status(1, KVPoll.WaitingForInput)
        self.wait_until(
            lambda: (
                stages[0].request_status.collect(1)
                == [KVPoll.WaitingForInput, KVPoll.WaitingForInput]
            )
        )
        self.pp_step(schedulers)
        for scheduler, req in zip(schedulers, reqs):
            scheduler.send_kv_chunk.assert_called_once_with(req, last_chunk=True)
            self.assertFalse(req.pending_bootstrap)
            self.assertEqual(list(scheduler.disagg_prefill_transfer_queue), [req])

    def test_pp_stage_local_bootstrap_failure_drops_parked_prefill_everywhere(self):
        stages = self.pp_stages(2)
        senders = [MooncakeKVSender(mgr, "prefill:8998", 1) for mgr in stages]
        schedulers = self.pp_parked_schedulers(senders)
        reqs = [scheduler.disagg_prefill_transfer_queue[0] for scheduler in schedulers]
        # Decode's cancellation reached the last stage only.
        senders[1].abort()
        self.wait_until(lambda: stages[0].request_status.collect(1)[1] == KVPoll.Failed)
        self.pp_step(schedulers)
        for scheduler, req in zip(schedulers, reqs):
            scheduler.handle_bootstrap_failure.assert_called_once_with(req)
            self.assertEqual(list(scheduler.disagg_prefill_transfer_queue), [])
            scheduler.send_kv_chunk.assert_not_called()

    def test_pp_stage_local_bootstrap_failure_drops_waiting_prefill_everywhere(self):
        """Under PP no stage-local poll may drop a waiting request, so without the
        consensus one whose decode peer gave up is prefilled in full first."""
        stages = self.pp_stages(2)
        senders = [MooncakeKVSender(mgr, "prefill:8998", 1) for mgr in stages]
        schedulers = self.pp_parked_schedulers(senders, waiting=True)
        reqs = [scheduler.waiting_queue[0] for scheduler in schedulers]
        senders[1].abort()
        self.wait_until(lambda: stages[0].request_status.collect(1)[1] == KVPoll.Failed)
        self.pp_step(schedulers)
        for scheduler, req in zip(schedulers, reqs):
            scheduler.handle_bootstrap_failure.assert_called_once_with(req)
            self.assertEqual(list(scheduler.waiting_queue), [])

    def test_pp_release_leaves_parked_prefill_to_consensus(self):
        """The release poll is stage-local: failing a parked request there lets
        the stages' parked sets diverge and the next consensus poll misalign."""
        sender = self.sender()
        sender.abort()
        req = FakeReq(rid="parked", disagg_kv_sender=sender, pending_bootstrap=True)
        scheduler = self.inflight_scheduler([req])
        scheduler.handle_bootstrap_failure = Mock()
        scheduler.process_disagg_prefill_transfer_queue(rids_to_check={"parked": True})
        scheduler.handle_bootstrap_failure.assert_not_called()
        self.assertEqual(list(scheduler.disagg_prefill_transfer_queue), [req])

    def test_pp_consensus_fails_when_stages_order_requests_differently(self):
        """Stages holding as many requests as PP0 per queue, or in total, but in
        another order or split would otherwise apply PP0's polls to other
        requests; each stage re-sorts its waiting queue on its own."""
        a, b = (
            SimpleNamespace(
                bootstrap_room=room, poll_pp_consensus=lambda: KVPoll.Bootstrapping
            )
            for room in (1, 2)
        )
        self.mock_pp_sync()
        groups = dict(
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            pp_group=None,
            pp_size=2,
            pp_poll_sync_work_list=[],
        )
        self.assertEqual(
            poll_and_all_reduce_prefill_pp_sections([[a, b], []], pp_rank=0, **groups),
            [[KVPoll.Bootstrapping] * 2, []],
        )
        for label, sections in (("reordered", [[b, a], []]), ("split", [[a], [b]])):
            with self.subTest(label):
                with self.assertRaisesRegex(AssertionError, "PP stages disagree"):
                    poll_and_all_reduce_prefill_pp_sections(
                        sections, pp_rank=1, **groups
                    )

    def test_pp_chunked_prefill_ignores_stage_local_bootstrap_failure(self):
        """Releasing a chunked request on one stage's poll desynchronizes the PP
        stages, which keep scheduling its next chunk."""
        for pp_size in (1, 2):
            with self.subTest(pp_size=pp_size):
                sender = self.sender(room=pp_size)
                sender.abort()
                req = SimpleNamespace(
                    rid="chunked", disagg_kv_sender=sender, pending_bootstrap=True
                )
                scheduler = self.inflight_scheduler([])
                scheduler.handle_bootstrap_failure = Mock()
                with get_parallel().override(pp_size=pp_size):
                    self.assertFalse(scheduler.check_bootstrap(req))
                self.assertEqual(
                    scheduler.handle_bootstrap_failure.call_count, int(pp_size == 1)
                )

    def test_destination_mid_prefill_starts_transfer_only_without_pp(self):
        """just_in_time lets decode's destination arrive while the request still
        chunks. Without PP the next chunk round finalizes it and sends the
        finished chunks, or under overlap marks them for the next result's
        send; under PP only the parked consensus may finalize."""
        room = 10
        for pp_size in (1, 2):
            for overlap in (False, True):
                with self.subTest(pp_size=pp_size, overlap=overlap):
                    room += 1
                    sender = self.sender(room=room)
                    self.prefill.update_status(room, KVPoll.WaitingForInput)
                    req = Mock(
                        disagg_kv_sender=sender,
                        pending_bootstrap=True,
                        to_finish=None,
                        finished_reason=None,
                        extend_end=1024,
                        origin_input_ids=[0] * 4096,
                        tmp_end_idx=None,
                    )
                    scheduler = self.inflight_scheduler([])
                    scheduler.chunked_req = req
                    scheduler.enable_overlap = overlap

                    def finalize(r):
                        r.pending_bootstrap = False
                        return True

                    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
                        finalize_bootstrap=Mock(side_effect=finalize)
                    )
                    running = SimpleNamespace(batch_is_full=True)
                    with get_parallel().override(pp_size=pp_size):
                        SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
                            scheduler, None, running
                        )
                    finalized = pp_size == 1
                    self.assertIs(scheduler.chunked_req, req)
                    self.assertEqual(req.pending_bootstrap, not finalized)
                    self.assertEqual(
                        req.tmp_end_idx, 1024 if finalized and overlap else None
                    )
                    if finalized and not overlap:
                        scheduler.send_kv_chunk.assert_called_once_with(req)
                    else:
                        scheduler.send_kv_chunk.assert_not_called()

    def test_recompute_waits_in_bootstrap_queue_until_room_release(self):
        """A retraction recompute that reaches prefill before it released the
        room's previous request would fail to open the room as a duplicate."""
        table = self.prefill.deferred_bootstrap
        owner = object()
        table.open(1, owner)
        table.register(1, DECODE)
        table.complete(1, owner)
        table.conclude(1)
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.kv_manager = SimpleNamespace(deferred_bootstrap=table)
        queue.queue = IndexedReqList(ListKind.PD_BOOTSTRAP)
        queue.room_waiters = IndexedReqList(
            ListKind.PD_ROOM_WAITERS, req_of=lambda waiter: waiter[0]
        )
        queue.create_sender = Mock(return_value=True)
        req = SimpleNamespace(
            to_finish=None,
            finished_reason=None,
            bootstrap_host="prefill",
            bootstrap_room=1,
        )
        queue.add(req, 8)
        queue.admit_room_waiters()
        queue.create_sender.assert_not_called()
        self.assertEqual(list(queue.queue), [])
        table.close(1, owner, success=True)
        queue.admit_room_waiters()
        queue.create_sender.assert_called_once_with(req, 8)
        self.assertEqual((list(queue.queue), list(queue.room_waiters)), ([req], []))

    def test_success_reaches_decode_before_release_and_resume_waits(self):
        """The transfer worker tells decode at once. A retraction resume that
        registers before prefill released the room is carried over to the room's
        next request rather than attached to the finished one, and no second
        Success can admit it early."""
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.signal_decode_readiness()
        self.flush()
        self.prefill.conclude_transfer(
            bootstrap_room=1, status=KVPoll.Success, targets=[DECODE]
        )
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Success)
        receiver.clear()
        new_receiver = self.receiver()
        self.flush()
        table = self.prefill.deferred_bootstrap
        self.assertTrue(table.awaits_release(1))
        self.assertEqual(sender.poll(), KVPoll.Success)
        sender.clear()
        sender.clear()
        self.assertFalse(table.awaits_release(1))
        self.assertEqual(list(self.wire), [])
        self.assertEqual(new_receiver.poll(), KVPoll.Bootstrapping)
        new_sender = self.sender()
        new_sender.signal_decode_readiness()
        self.flush()
        self.assertEqual(new_receiver.poll(), KVPoll.WaitingForInput)

    def test_prefill_abort_after_worker_success_leaves_decode_alone(self):
        """Decode already heard Success, so an abort of the request on prefill
        before release must neither send decode Failed for the room nor cancel
        decode's retraction resume that registered meanwhile."""
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.signal_decode_readiness()
        self.flush()
        self.prefill.conclude_transfer(
            bootstrap_room=1, status=KVPoll.Success, targets=[DECODE]
        )
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Success)
        receiver.clear()
        new_receiver = self.receiver()
        self.flush()
        sender.abort()
        self.assertEqual(sender.poll(), KVPoll.Failed)
        with self.assertRaises(KVTransferError):
            sender.failure_exception()
        self.assertEqual(list(self.wire), [])
        new_sender = self.sender()
        new_sender.signal_decode_readiness()
        self.flush()
        self.assertEqual(new_receiver.poll(), KVPoll.WaitingForInput)

    def _run_last_chunk_through_worker(self, during_cleanup):
        """Run room 1's last chunk through the Mooncake transfer worker, calling
        during_cleanup() after its writes and Success, inside its room cleanup."""
        mgr = self.prefill
        mgr._staging_outstanding = defaultdict(int)
        mgr.session_lock = threading.Lock()
        mgr.failed_sessions = set()
        mgr.state_layout_rejections = {}
        mgr.state_strides_validated = set()
        mgr.enable_deferred_decode_kv_release = True
        mgr.decode_kv_args_table = {
            "session": SimpleNamespace(
                requires_dcp_relayout=False, dst_aux_ptrs=[], dst_aux_item_lens=None
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

        def pop(room, default=None):
            # The worker has finished writes and decremented its counter, but
            # is still deleting room-indexed state.
            self.assertEqual(mgr._staging_outstanding[room], 0)
            during_cleanup()
            cleanup_observed.append(room)

        mgr.req_to_decode_prefix_len = SimpleNamespace(pop=pop)
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
        mgr.req_to_decode_prefix_len = {}

    def test_release_waits_for_worker_cleanup(self):
        """Decode hears Success from the worker before its room cleanup, but
        prefill's sender, whose release lets a recompute open the room, reports
        Success only after it."""
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.signal_decode_readiness()
        self.flush()

        def during_cleanup():
            self.assertEqual(sender.poll(), KVPoll.Transferring)
            self.flush()
            self.assertEqual(receiver.poll(), KVPoll.Success)

        self._run_last_chunk_through_worker(during_cleanup)
        self.assertEqual(sender.poll(), KVPoll.Success)
        sender.clear()
        self.assertEqual(list(self.wire), [])

    def test_abort_after_success_still_waits_for_worker_cleanup(self):
        """An abort after the worker's Success caches Failed, but the room stays
        held until the worker's room-keyed cleanup ends; a recompute reusing it
        would otherwise lose the transfer state the old worker still erases."""
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.signal_decode_readiness()
        self.flush()

        def during_cleanup():
            sender.abort()
            self.assertEqual(sender.poll(), KVPoll.Transferring)

        self._run_last_chunk_through_worker(during_cleanup)
        self.assertEqual(sender.poll(), KVPoll.Failed)
        with self.assertRaises(KVTransferError):
            sender.failure_exception()
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Success)

    def test_failure_racing_worker_success_is_decodes_only_notice(self):
        """An abort landing between the worker's Success decision and its commit
        fails the room; decode must then hear Failed, never Success."""
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.signal_decode_readiness()
        self.flush()
        table = self.prefill.deferred_bootstrap
        commit = table.conclude

        def abort_first(room):
            sender.abort()
            return commit(room)

        table.conclude = abort_first
        send = Mock(wraps=self.prefill.send_kv_status_message)
        self.prefill.send_kv_status_message = send
        self.prefill.conclude_transfer(
            bootstrap_room=1, status=KVPoll.Success, targets=[DECODE]
        )
        self.assertEqual(
            [call.kwargs["status"] for call in send.call_args_list], [KVPoll.Failed]
        )
        self.flush()
        self.assertEqual(receiver.poll(), KVPoll.Failed)

    def test_recompute_http_error_after_success_before_retirement_is_ignored(self):
        sender, receiver = self.sender(), self.receiver()
        self.flush()
        sender.signal_decode_readiness()
        self.flush()
        self.prefill.conclude_transfer(
            bootstrap_room=1, status=KVPoll.Success, targets=[DECODE]
        )
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

    def test_rebootstrap_dispatch_precedes_ready_without_allocating(self):
        receiver = self.receiver()
        receiver.init = Mock()
        req = SimpleNamespace(
            bootstrap_host="prefill",
            build_rebootstrap_payload=lambda: {"rid": "resume"},
        )
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
                        bootstrap_host="prefill",
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
                    queue.scheduler = SimpleNamespace(enable_overlap=False)
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


PACED_CHUNK = 100


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _paced_req(
    rid,
    *,
    input_len,
    handshake=None,
    endpoint=("decode", 1),
    pending=True,
    registered_at=0.0,
):
    sender = Mock()
    sender.decode_handshake.return_value = (
        None
        if handshake is None
        else DecodeHandshake(
            endpoint=endpoint, seconds=handshake, registered_at=registered_at
        )
    )
    return SimpleNamespace(
        rid=rid,
        bootstrap_room=hash(rid),
        origin_input_ids=[0] * input_len,
        extend_end=0,
        pending_bootstrap=pending,
        to_finish=None,
        finished_reason=None,
        disagg_kv_sender=sender,
    )


class TestDecodeReadinessPacer(CustomTestCase):
    """When just_in_time signals decode readiness during prefill."""

    def setUp(self):
        super().setUp()
        self.reset()

    def reset(self, **pacer_kwargs):
        self.clock = _Clock()
        self.pacer = DecodeReadinessPacer(
            chunk_tokens=PACED_CHUNK, clock=self.clock, **pacer_kwargs
        )

    def prime(self, *, handshakes, interval, lead):
        for seconds in handshakes:
            self.pacer.on_bootstrap_done(
                _paced_req("h", input_len=1, handshake=seconds)
            )
        warm = _paced_req("warm", input_len=PACED_CHUNK)
        self.schedule(warm)
        self.clock.now += interval
        self.pacer.on_round([_paced_req("busy", input_len=1, pending=False)])
        self.clock.now += lead - interval
        self.pacer.on_prefill_finished(warm)
        # An idle round, so the next scheduled round starts a new interval.
        self.pacer.on_round([])

    def schedule(self, req, tokens=PACED_CHUNK):
        req.extend_end = min(req.extend_end + tokens, len(req.origin_input_ids))
        self.pacer.on_round([req])

    def rounds_until_signal(self, req, *, interval, tokens=PACED_CHUNK, admit_at=None):
        """Schedule req's chunks one round apart; return the 1-based chunk whose
        round signaled readiness, or None. From chunk admit_at on, decode has
        admitted the request."""
        chunk, signaled = 0, None
        while req.extend_end < len(req.origin_input_ids):
            chunk += 1
            if chunk == admit_at:
                req.pending_bootstrap = False
            self.schedule(req, tokens)
            if signaled is None and req.disagg_kv_sender.signal_decode_readiness.called:
                signaled = chunk
            self.clock.now += interval
        return signaled

    def test_signals_at_last_round_whose_remaining_prefill_covers_handshake(self):
        # After scheduling chunk c of 10 at 1 s rounds with a 2 s lead, 12 - c
        # seconds of prefill remain. Chunk 9 is the last round covering a 2.5 s
        # handshake; a 1 s handshake stays covered, so the final result signals.
        def steady():
            self.prime(handshakes=[2.5, 2.5], interval=1.0, lead=2.0)

        def first_handshake_per_endpoint():
            # Kept as an estimate, the 100 s connection setup would signal at chunk 1.
            self.prime(handshakes=[100.0, 1.0], interval=1.0, lead=2.0)

        def slow_startup():
            # The first requests also pay compilation; with a mean, their round
            # and lead would keep later requests signaling early.
            self.prime(handshakes=[2.5, 2.5], interval=30.0, lead=60.0)
            for _ in range(2):
                self.prime(handshakes=[], interval=1.0, lead=2.0)

        def admitted_mid_prefill():
            # Decode admits these at chunk 3. Their later chunks must still move
            # the lead's start, or the lead would make later requests signal late.
            steady()
            for i in range(2):
                req = _paced_req(f"admitted-{i}", input_len=10 * PACED_CHUNK)
                self.rounds_until_signal(req, interval=1.0, admit_at=3)
                self.clock.now += 2.0 - 1.0
                self.pacer.on_prefill_finished(req)
                self.pacer.on_round([])

        for setup, chunk in (
            (steady, 9),
            (first_handshake_per_endpoint, None),
            (slow_startup, 9),
            (admitted_mid_prefill, 9),
        ):
            with self.subTest(setup.__name__):
                self.reset()
                setup()
                req = _paced_req("long", input_len=10 * PACED_CHUNK)
                self.assertEqual(self.rounds_until_signal(req, interval=1.0), chunk)
                self.assertEqual(
                    req.disagg_kv_sender.signal_decode_readiness.call_count,
                    0 if chunk is None else 1,
                )

    def test_signals_before_first_forward_without_estimates_or_short_prefill(self):
        with self.subTest("no estimates yet"):
            req = _paced_req("cold", input_len=10 * PACED_CHUNK)
            self.assertEqual(self.rounds_until_signal(req, interval=1.0), 1)
        with self.subTest("prefill shorter than the handshake"):
            self.prime(handshakes=[10.0, 10.0], interval=1.0, lead=2.0)
            req = _paced_req("short", input_len=3 * PACED_CHUNK)
            self.assertEqual(self.rounds_until_signal(req, interval=1.0), 1)

    def test_pp_drops_handshakes_decode_registered_after_finish(self):
        """Under PP any stage's readiness admits a room, so when decode registers
        only after PP0's final result another stage's message may admit it and
        PP0 sees a residual. Such samples would pull H toward zero; without PP,
        and with registration before the finish, the sample is the handshake."""
        for pipelined, registered_at, kept in (
            (True, 6.0, False),
            (True, 4.0, True),
            (False, 6.0, True),
        ):
            with self.subTest(pipelined=pipelined, registered_at=registered_at):
                self.reset(pipelined=pipelined)
                self.prime(handshakes=[2.5, 2.5], interval=1.0, lead=2.0)
                self.clock.now = 5.0
                for i in range(10):
                    req = _paced_req(
                        f"late-{i}",
                        input_len=PACED_CHUNK,
                        handshake=0.001,
                        registered_at=registered_at,
                    )
                    self.pacer.on_prefill_finished(req)
                    self.pacer.on_bootstrap_done(req)
                self.pacer.on_round([])
                # 2.5 s handshake: signal at chunk 9 of 10; 1 ms: never by round.
                req = _paced_req("long", input_len=10 * PACED_CHUNK)
                self.assertEqual(
                    self.rounds_until_signal(req, interval=1.0), None if kept else 9
                )

    def test_remaining_chunks_follow_dynamic_chunk_sizes(self):
        # PP dynamic chunking shrinks continuation chunks to 50 tokens; with 1 s
        # rounds, a 2 s lead and a 2.5 s handshake the request must signal once
        # at most one 50-token chunk is left, not one 100-token chunk.
        self.reset(next_chunk_tokens=lambda prefix_len: PACED_CHUNK // 2)
        self.prime(handshakes=[2.5, 2.5], interval=1.0, lead=2.0)
        req = _paced_req("dynamic", input_len=10 * PACED_CHUNK)
        chunk = self.rounds_until_signal(req, interval=1.0, tokens=PACED_CHUNK // 2)
        self.assertEqual(chunk * (PACED_CHUNK // 2), 9.5 * PACED_CHUNK)


class TestDecodeReadinessWiring(CustomTestCase):
    """Both prefill loops hand each scheduled batch to the pacer before it
    launches, and finalized bootstraps feed the handshake estimate; without
    either, just_in_time silently degrades to another policy's timing."""

    def scheduler_with_cold_pacer(self, req):
        s = Scheduler.__new__(Scheduler)
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(
            readiness_pacer=DecodeReadinessPacer(chunk_tokens=PACED_CHUNK)
        )
        s.dp_attn_adapter = Mock()
        s.get_new_batch_prefill = Mock(
            return_value=NextBatchPlan(
                batch_to_run=SimpleNamespace(reqs=[req]),
                running_batch=SimpleNamespace(batch_is_full=False),
            )
        )
        return s

    def test_scheduled_batch_reaches_pacer_before_launch(self):
        req = _paced_req("r", input_len=PACED_CHUNK)
        s = self.scheduler_with_cold_pacer(req)
        s.scheduler_stage_metrics = None
        s.process_pending_chunked_abort = Mock()
        s._process_hicache_events = Mock()
        s.resolve_waiting_queue_bootstrap = Mock()
        s.process_prefill_chunk = Mock()
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None
        s.get_next_disagg_prefill_batch_to_run(
            running_batch=SimpleNamespace(batch_is_full=False), last_batch=None
        )
        req.disagg_kv_sender.signal_decode_readiness.assert_called_once()

    def test_pp_scheduled_batch_reaches_pacer_before_launch(self):
        req = _paced_req("r", input_len=PACED_CHUNK)
        s = self.scheduler_with_cold_pacer(req)
        enter_scope(self, published_topology(pp_size=2))
        s.init_pp_loop_state = Mock()
        s.pp_loop_size = 1
        s.pp_group = SimpleNamespace(is_last_rank=True)
        s.running_mbs = [SimpleNamespace(batch_is_full=False)]
        s.last_mbs = [None]
        s.mbs = [None]
        s.ingest_requests = Mock(return_value=[])
        s.process_bootstrapped_queue = Mock()
        s._pp_pd_get_prefill_transferred_ids = Mock(return_value=[])
        s._pp_commit_comm_work = Mock()
        s.process_prefill_chunk = Mock()
        s._process_hicache_events = Mock()
        # Stop the infinite loop right after admission, before any launch.
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = StopIteration
        with self.assertRaises(StopIteration):
            s.event_loop_pp_disagg_prefill()
        req.disagg_kv_sender.signal_decode_readiness.assert_called_once()

    def test_finalized_bootstrap_feeds_handshake_estimate(self):
        clock = _Clock()
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.readiness_pacer = DecodeReadinessPacer(
            chunk_tokens=PACED_CHUNK, clock=clock
        )
        queue.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1)
        )
        for _ in range(2):
            req = _paced_req("r", input_len=PACED_CHUNK, handshake=0.5)
            req.metadata_buffer_index = 0
            req.time_stats = Mock()
            req.disagg_kv_sender.pop_decode_prefix_len.return_value = 0
            self.assertTrue(queue.finalize_bootstrap(req))
        # The only other estimates: a 1 s round and a 1 s lead.
        pacer = queue.readiness_pacer
        warm = _paced_req("warm", input_len=PACED_CHUNK)
        pacer.on_round([warm])
        clock.now += 1.0
        pacer.on_round([_paced_req("busy", input_len=1, pending=False)])
        pacer.on_prefill_finished(warm)
        pacer.on_round([])
        # 10 s of prefill remain against a 0.5 s handshake; with no handshake
        # estimate the request would signal at once.
        req = _paced_req("long", input_len=10 * PACED_CHUNK)
        req.extend_end = PACED_CHUNK
        pacer.on_round([req])
        req.disagg_kv_sender.signal_decode_readiness.assert_not_called()


if __name__ == "__main__":
    unittest.main()
