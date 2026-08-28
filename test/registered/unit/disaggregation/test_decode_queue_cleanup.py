import threading
import time
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePollCoordinator,
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    _build_poll_tensor,
)
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


class FakeDecodeScheduler(SchedulerDisaggregationDecodeMixin, SimpleNamespace):
    """Minimal scheduler whose mixin helpers remain real in queue tests."""


class TestDecodeQueueCleanup(CustomTestCase):
    def test_fixed_poll_tensor_pads_missing_tp_entries_as_bootstrapping(self):
        tensor = _build_poll_tensor([KVPoll.Success], collective_size=3)

        self.assertEqual(
            tensor.tolist(),
            [KVPoll.Success, KVPoll.Bootstrapping, KVPoll.Bootstrapping],
        )

    def test_fixed_poll_tensor_rejects_queue_overflow(self):
        with self.assertRaisesRegex(RuntimeError, "exceeded its collective capacity"):
            _build_poll_tensor(
                [KVPoll.Bootstrapping, KVPoll.WaitingForInput], collective_size=1
            )

    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_prealloc_poll_uses_metadata_sized_fifo_window(self, mock_poll):
        receivers = [MagicMock() for _ in range(3)]
        decode_reqs = [
            SimpleNamespace(
                kv_receiver=receiver,
                waiting_for_input=False,
                req=SimpleNamespace(time_stats=MagicMock()),
            )
            for receiver in receivers
        ]
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = decode_reqs
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(size=2)
        mock_poll.return_value = [KVPoll.WaitingForInput] * 2

        queue._update_handshake_waiters()

        mock_poll.assert_called_once_with(
            receivers[:2], queue.gloo_group, collective_size=2
        )
        self.assertTrue(decode_reqs[0].waiting_for_input)
        self.assertTrue(decode_reqs[1].waiting_for_input)
        self.assertFalse(decode_reqs[2].waiting_for_input)

    def test_empty_transfer_queue_builds_fixed_combined_poll_segment(self):
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = []
        queue.enable_staging = False
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(size=8)
        queue.metadata_buffers = MagicMock()
        queue.scheduler = SimpleNamespace(
            enable_decode_hicache=False,
            server_args=MagicMock(),
        )

        count, tensor = queue.prepare_poll_tensor()

        self.assertEqual(count, 0)
        self.assertEqual(tensor.tolist(), [KVPoll.Bootstrapping] * 8)

    @patch("sglang.srt.disaggregation.decode.torch.distributed.all_reduce")
    def test_poll_coordinator_keeps_collective_off_caller_thread(self, mock_all_reduce):
        entered = threading.Event()
        release = threading.Event()

        def blocking_all_reduce(*_args, **_kwargs):
            entered.set()
            self.assertTrue(release.wait(timeout=1))

        mock_all_reduce.side_effect = blocking_all_reduce
        coordinator = DecodePollCoordinator(MagicMock())
        state = {"tensor": torch.tensor([KVPoll.Success], dtype=torch.uint8)}

        coordinator.submit(state)

        self.assertTrue(entered.wait(timeout=1))
        self.assertIsNone(coordinator.poll())
        release.set()
        deadline = time.monotonic() + 1
        result = None
        while result is None and time.monotonic() < deadline:
            result = coordinator.poll()
            time.sleep(0.001)
        self.assertIs(result, state)

    @patch("sglang.srt.disaggregation.decode.torch.distributed.all_reduce")
    def test_poll_and_request_collectives_use_one_worker_thread(self, mock_all_reduce):
        collective_threads = []

        def record_poll(*_args, **_kwargs):
            collective_threads.append(threading.get_ident())

        mock_all_reduce.side_effect = record_poll
        coordinator = DecodePollCoordinator(MagicMock())
        state = {"tensor": torch.tensor([KVPoll.Success], dtype=torch.uint8)}

        coordinator.submit(state)
        deadline = time.monotonic() + 1
        while coordinator.poll() is None and time.monotonic() < deadline:
            time.sleep(0.001)

        result = coordinator.execute(
            lambda: collective_threads.append(threading.get_ident()) or ["request"]
        )

        self.assertEqual(result, ["request"])
        self.assertEqual(len(collective_threads), 2)
        self.assertEqual(collective_threads[0], collective_threads[1])
        self.assertNotEqual(collective_threads[0], threading.get_ident())

    @patch("sglang.srt.disaggregation.decode.torch.distributed.all_reduce")
    def test_poll_result_is_bound_to_its_task(self, mock_all_reduce):
        entered = threading.Event()
        release = threading.Event()

        def blocking_all_reduce(*_args, **_kwargs):
            entered.set()
            self.assertTrue(release.wait(timeout=1))

        mock_all_reduce.side_effect = blocking_all_reduce
        coordinator = DecodePollCoordinator(MagicMock())
        state = {"tensor": torch.tensor([KVPoll.Success], dtype=torch.uint8)}

        coordinator.submit(state)
        self.assertTrue(entered.wait(timeout=1))
        with self.assertRaisesRegex(RuntimeError, "pending poll result"):
            coordinator.execute(lambda: [])
        with self.assertRaisesRegex(RuntimeError, "already pending"):
            coordinator.submit(state)

        release.set()
        deadline = time.monotonic() + 1
        result = None
        while result is None and time.monotonic() < deadline:
            result = coordinator.poll()
            time.sleep(0.001)
        self.assertIs(result, state)
        self.assertEqual(coordinator.execute(lambda: ["request"]), ["request"])

    @patch("sglang.srt.disaggregation.decode.get_disagg")
    def test_decode_queue_submits_one_combined_poll(self, mock_get_disagg):
        mock_get_disagg.return_value = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=1,
        )
        prealloc_req = MagicMock()
        prealloc_queue = MagicMock(pp_size=1, queue=[prealloc_req], retracted_queue=[])
        prealloc_queue.resume_retracted_reqs.return_value = []
        prealloc_queue.prepare_poll_tensor.return_value = (
            [prealloc_req],
            torch.tensor([KVPoll.WaitingForInput, KVPoll.Bootstrapping]),
        )
        transfer_queue = MagicMock(queue=[])
        transfer_queue.prepare_poll_tensor.return_value = (
            1,
            torch.tensor([KVPoll.Success, KVPoll.Bootstrapping]),
        )
        coordinator = MagicMock()
        scheduler = FakeDecodeScheduler(
            enable_decode_hicache=False,
            disagg_decode_prealloc_queue=prealloc_queue,
            disagg_decode_transfer_queue=transfer_queue,
            disagg_decode_poll_pending=False,
            disagg_decode_poll_coordinator=coordinator,
            waiting_queue=[],
        )

        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        poll_state = coordinator.submit.call_args.args[0]
        self.assertEqual(
            poll_state["tensor"].tolist(),
            [
                KVPoll.WaitingForInput,
                KVPoll.Bootstrapping,
                KVPoll.Success,
                KVPoll.Bootstrapping,
                1,
            ],
        )
        self.assertIs(poll_state["prealloc_window"][0], prealloc_req)
        self.assertEqual(poll_state["transfer_count"], 1)
        self.assertTrue(scheduler.disagg_decode_poll_pending)
        transfer_queue.pop_transferred.assert_not_called()
        prealloc_queue.pop_preallocated.assert_not_called()

    @patch("sglang.srt.disaggregation.decode.get_disagg")
    def test_decode_queue_does_not_block_on_incomplete_background_poll(
        self, mock_get_disagg
    ):
        mock_get_disagg.return_value = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=1,
        )
        prealloc_queue = MagicMock(pp_size=1, queue=[], retracted_queue=[])
        prealloc_queue.resume_retracted_reqs.return_value = []
        transfer_queue = MagicMock(queue=[])
        coordinator = MagicMock()
        coordinator.poll.return_value = None
        scheduler = FakeDecodeScheduler(
            enable_decode_hicache=False,
            disagg_decode_prealloc_queue=prealloc_queue,
            disagg_decode_transfer_queue=transfer_queue,
            disagg_decode_poll_pending=True,
            disagg_decode_poll_coordinator=coordinator,
            waiting_queue=[],
        )

        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        coordinator.poll.assert_called_once_with()
        coordinator.submit.assert_not_called()
        prealloc_queue.prepare_poll_tensor.assert_not_called()
        transfer_queue.prepare_poll_tensor.assert_not_called()

    def test_pending_poll_blocks_next_request_epoch_until_result_is_staged(self):
        coordinator = MagicMock()
        coordinator.poll.side_effect = [None, {"tensor": torch.tensor([1])}]
        scheduler = FakeDecodeScheduler(
            disagg_decode_prealloc_queue=SimpleNamespace(pp_size=1),
            disagg_decode_poll_pending=True,
            disagg_decode_poll_result=None,
            disagg_decode_poll_coordinator=coordinator,
        )

        self.assertFalse(
            SchedulerDisaggregationDecodeMixin._stage_completed_decode_poll(scheduler)
        )
        self.assertTrue(scheduler.disagg_decode_poll_pending)
        self.assertIsNone(scheduler.disagg_decode_poll_result)

        self.assertTrue(
            SchedulerDisaggregationDecodeMixin._stage_completed_decode_poll(scheduler)
        )
        self.assertFalse(scheduler.disagg_decode_poll_pending)
        self.assertEqual(scheduler.disagg_decode_poll_result["tensor"].tolist(), [1])
        self.assertEqual(coordinator.poll.call_count, 2)

    @patch("sglang.srt.disaggregation.decode.get_disagg")
    def test_decode_queue_consumes_completed_background_poll(self, mock_get_disagg):
        mock_get_disagg.return_value = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=1,
        )
        prealloc_req = MagicMock()
        prealloc_queue = MagicMock(pp_size=1, queue=[prealloc_req], retracted_queue=[])
        prealloc_queue.resume_retracted_reqs.return_value = []
        prealloc_queue.pop_preallocated.return_value = (["new-transfer"], [])
        prealloc_queue.prepare_poll_tensor.return_value = (
            [],
            torch.tensor([KVPoll.Bootstrapping, KVPoll.Bootstrapping]),
        )
        transfer_queue = MagicMock(queue=[])
        transfer_queue.pop_transferred.return_value = ["ready"]
        transfer_queue.prepare_poll_tensor.return_value = (
            0,
            torch.tensor([KVPoll.Bootstrapping, KVPoll.Bootstrapping]),
        )
        coordinator = MagicMock()
        coordinator.poll.return_value = {
            "tensor": torch.tensor(
                [
                    KVPoll.WaitingForInput,
                    KVPoll.Bootstrapping,
                    KVPoll.Success,
                    KVPoll.Bootstrapping,
                    1,
                ],
                dtype=torch.uint8,
            ),
            "prealloc_window": [prealloc_req],
            "transfer_count": 1,
        }
        scheduler = FakeDecodeScheduler(
            enable_decode_hicache=False,
            enable_hisparse=False,
            disagg_decode_prealloc_queue=prealloc_queue,
            disagg_decode_transfer_queue=transfer_queue,
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(size=2),
            disagg_decode_poll_pending=True,
            disagg_decode_poll_coordinator=coordinator,
            waiting_queue=[],
        )

        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        self.assertTrue(scheduler.disagg_decode_poll_pending)
        coordinator.submit.assert_called_once()
        next_poll_state = coordinator.submit.call_args.args[0]
        self.assertEqual(
            next_poll_state["tensor"].tolist(),
            [
                KVPoll.Bootstrapping,
                KVPoll.Bootstrapping,
                KVPoll.Bootstrapping,
                KVPoll.Bootstrapping,
                1,
            ],
        )
        self.assertEqual(next_poll_state["prealloc_window"], [])
        self.assertEqual(next_poll_state["transfer_count"], 0)
        transfer_queue.pop_transferred.assert_called_once_with(
            precomputed_polls=[KVPoll.Success]
        )
        prealloc_queue.pop_preallocated.assert_called_once_with(
            precomputed_polls=([prealloc_req], [KVPoll.WaitingForInput])
        )
        transfer_queue.extend.assert_called_once_with(["new-transfer"])
        self.assertEqual(scheduler.waiting_queue, ["ready"])

    @patch("sglang.srt.disaggregation.decode.get_disagg")
    def test_retracted_queue_keeps_fixed_poll_order(self, mock_get_disagg):
        mock_get_disagg.return_value = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=1,
        )
        prealloc_queue = MagicMock(pp_size=1, queue=[], retracted_queue=[MagicMock()])
        prealloc_queue.resume_retracted_reqs.return_value = []
        prealloc_queue.prepare_poll_tensor.return_value = (
            [],
            torch.tensor([KVPoll.Bootstrapping]),
        )
        transfer_queue = MagicMock(queue=[])
        transfer_queue.prepare_poll_tensor.return_value = (
            0,
            torch.tensor([KVPoll.Bootstrapping]),
        )
        coordinator = MagicMock()
        scheduler = FakeDecodeScheduler(
            enable_decode_hicache=False,
            disagg_decode_prealloc_queue=prealloc_queue,
            disagg_decode_transfer_queue=transfer_queue,
            disagg_decode_poll_pending=False,
            disagg_decode_poll_coordinator=coordinator,
            waiting_queue=[],
        )

        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        self.assertEqual(
            coordinator.submit.call_args.args[0]["tensor"].tolist(),
            [KVPoll.Bootstrapping, KVPoll.Bootstrapping, 0],
        )
        prealloc_queue.pop_preallocated.assert_not_called()
        transfer_queue.pop_transferred.assert_not_called()

    @patch("sglang.srt.disaggregation.decode.get_disagg")
    def test_idle_decode_queue_still_submits_poll_epoch(self, mock_get_disagg):
        mock_get_disagg.return_value = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=1,
        )
        prealloc_queue = MagicMock(pp_size=1, queue=[], retracted_queue=[])
        prealloc_queue.resume_retracted_reqs.return_value = []
        prealloc_queue.prepare_poll_tensor.return_value = (
            [],
            torch.tensor([KVPoll.Bootstrapping]),
        )
        transfer_queue = MagicMock(queue=[])
        transfer_queue.prepare_poll_tensor.return_value = (
            0,
            torch.tensor([KVPoll.Bootstrapping]),
        )
        coordinator = MagicMock()
        scheduler = FakeDecodeScheduler(
            enable_decode_hicache=False,
            disagg_decode_prealloc_queue=prealloc_queue,
            disagg_decode_transfer_queue=transfer_queue,
            disagg_decode_poll_pending=False,
            disagg_decode_poll_coordinator=coordinator,
            waiting_queue=[],
        )

        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        self.assertEqual(
            coordinator.submit.call_args.args[0]["tensor"].tolist(),
            [KVPoll.Bootstrapping, KVPoll.Bootstrapping, 1],
        )
        coordinator.poll.assert_not_called()
        self.assertTrue(scheduler.disagg_decode_poll_pending)

    @patch("sglang.srt.disaggregation.decode.get_disagg")
    def test_last_empty_poll_immediately_submits_next_epoch(self, mock_get_disagg):
        mock_get_disagg.return_value = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=1,
        )
        prealloc_queue = MagicMock(pp_size=1, queue=[], retracted_queue=[])
        prealloc_queue.resume_retracted_reqs.return_value = []
        prealloc_queue.prepare_poll_tensor.return_value = (
            [],
            torch.tensor([KVPoll.Bootstrapping]),
        )
        prealloc_queue.pop_preallocated.return_value = ([], [])
        transfer_queue = MagicMock(queue=[])
        transfer_queue.prepare_poll_tensor.return_value = (
            0,
            torch.tensor([KVPoll.Bootstrapping]),
        )
        transfer_queue.pop_transferred.return_value = []
        coordinator = MagicMock()
        scheduler = FakeDecodeScheduler(
            enable_decode_hicache=False,
            enable_hisparse=False,
            disagg_decode_prealloc_queue=prealloc_queue,
            disagg_decode_transfer_queue=transfer_queue,
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(size=1),
            disagg_decode_poll_pending=False,
            disagg_decode_poll_coordinator=coordinator,
            waiting_queue=[],
        )

        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)
        coordinator.poll.return_value = coordinator.submit.call_args.args[0]
        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        self.assertTrue(scheduler.disagg_decode_poll_pending)
        self.assertEqual(coordinator.submit.call_count, 2)

    def test_paged_swa_retraction_resume_uses_physical_page_budget(self):
        # resume_retracted_reqs reads the retraction backend off the disagg
        # bag, so the case publishes a config instead of injecting one.
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        page_size = 128
        fill_len = 574
        physical_tokens_per_req = 5 * page_size
        physical_available = 18 * page_size

        reqs = [
            SimpleNamespace(
                rid=f"req-{i}",
                origin_input_ids=[0] * fill_len,
                output_ids=[],
                is_retracted=True,
                retraction_backup=None,
                load_kv_cache=MagicMock(),
            )
            for i in range(4)
        ]

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = reqs.copy()
        queue.num_reserved_decode_tokens = 0
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=page_size)
        queue.tree_cache = MagicMock()
        queue.scheduler = SimpleNamespace(
            sliding_window_size=2047,
            server_args=SimpleNamespace(disable_radix_cache=True),
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(physical_available, physical_available)
        )
        queue._swa_tail_allocatable_token_budget = MagicMock(
            side_effect=lambda **_: physical_available
        )

        def pre_alloc(_req):
            nonlocal physical_available
            self.assertGreaterEqual(physical_available, physical_tokens_per_req)
            physical_available -= physical_tokens_per_req

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)

        resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, reqs[:3])
        self.assertEqual(queue.retracted_queue, reqs[3:])
        self.assertEqual(physical_available, 3 * page_size)
        self.assertEqual(queue._pre_alloc.call_count, 3)

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        # Same DecodeRequest lives in both queue and pending_reqs (add() slow
        # path). Aborting must drop it from both, and compare by identity since
        # DecodeRequest's dataclass __eq__ would compare the tensor receiver.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = SimpleNamespace(
            rid="abort-shared",
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(all(r is not decode_req for r in queue.pending_reqs))
        self.assertIsNone(decode_req.kv_receiver)

    def test_swa_reclaim_failure_rejects_only_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="swa-reclaim-failed",
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            finished_reason=None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=1),
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            waiting_for_input=True,
            is_rebootstrap=False,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(1024, 1024)
        )
        queue._prealloc_required_tokens = MagicMock(return_value=(3, 3))
        queue._prealloc_kv_lens = MagicMock(return_value=(3, 3))
        queue._reclaim_swa_tail_capacity = MagicMock(
            return_value=(
                "SWA eviction insufficient: needed=64, available=0, "
                "req=swa-reclaim-failed"
            )
        )
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._pre_alloc = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        queue._pre_alloc.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    def test_prefetches_prefill_dp_rank_query(self):
        addr = "127.0.0.1:11500"
        executor = MagicMock()
        future = Future()
        future.set_result({"7": 1})
        executor.submit.return_value = future

        def decode_req(room):
            return SimpleNamespace(
                req=SimpleNamespace(
                    bootstrap_host="127.0.0.1",
                    bootstrap_port=11500,
                    bootstrap_room=room,
                ),
                kv_receiver=MagicMock(),
            )

        first = decode_req(7)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [first]
        queue._prefill_dp_rank_queries = {}
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={addr: object()},
            _ensure_prefill_recompute_executor=lambda: executor,
        )
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])

        queue.prefetch_prefill_dp_rank_queries()
        tail = decode_req(8)
        queue.pending_reqs.append(tail)
        with patch(
            "sglang.srt.disaggregation.decode."
            "CommonKVReceiver.query_prefill_dp_ranks",
            return_value={"8": 2},
        ) as query:
            queue._resolve_pending_reqs()

        _, called_addr, called_rooms = executor.submit.call_args.args
        self.assertEqual((called_addr, called_rooms), (addr, [7]))
        query.assert_called_once_with(addr, [8])
        first.kv_receiver.init.assert_called_once_with(1)
        tail.kv_receiver.init.assert_called_once_with(2)
        self.assertEqual(queue.pending_reqs, [])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_clears_receiver_before_removing_request(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.ps = ParallelState.trivial()
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False

        self.assertFalse(scheduler.is_fully_idle())


if __name__ == "__main__":
    unittest.main()
