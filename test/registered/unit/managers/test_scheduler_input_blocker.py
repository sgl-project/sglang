import asyncio
import datetime
import multiprocessing
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch.distributed as dist
import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    BlockReqInput,
    BlockReqType,
    FlushCacheReqInput,
    GenerateReqInput,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.flush_wrapper import (  # noqa: E402
    SchedulerFlushWrapper,
)
from sglang.srt.managers.scheduler_components.idle_sleeper import (  # noqa: E402
    IdleSleeper,
)
from sglang.srt.managers.scheduler_components.pause_state import (  # noqa: E402
    PauseState,
)
from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_input_blocker import (  # noqa: E402
    SchedulerInputBlocker,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.models.unlimited_ocr_v2_support.request_receiver import (  # noqa: E402
    UocrRequestReceiver,
)

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


def _block():
    return BlockReqInput(req_type=BlockReqType.BLOCK)


def _unblock():
    return BlockReqInput(req_type=BlockReqType.UNBLOCK)


def _blocker(case, *, noop=False, global_ready=True):
    blocker = SchedulerInputBlocker(noop=noop)
    barrier = blocker._global_unblock_barrier
    control = SimpleNamespace(
        global_ready=global_ready, global_pending=False, global_polling=False, votes=[]
    )

    def collective():
        local_ready = barrier._noop or barrier._local_arrived
        control.votes.append(local_ready)
        barrier.global_pending = barrier._local_pending or control.global_pending
        barrier.global_polling = barrier._local_polling or control.global_polling
        return local_ready and control.global_ready

    mocked = patch.object(barrier, "_compute_global_arrived", side_effect=collective)
    mocked.start()
    case.addCleanup(mocked.stop)
    return blocker, control


def _asymmetric_tail_worker(rank, rendezvous, connection):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    try:
        dist.init_process_group(
            "gloo",
            init_method=Path(rendezvous).as_uri(),
            rank=rank,
            world_size=4,
            timeout=datetime.timedelta(seconds=120),
        )
        port = receiver.bind_to_random_port("tcp://127.0.0.1")
        connection.send(f"tcp://127.0.0.1:{port}")
        packets = receiver.recv_pyobj()
        noop = rank % 2 != 0
        blocker = SchedulerInputBlocker(noop=noop)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.input_blocker = blocker
        with envs.SGLANG_EMPTY_CACHE_INTERVAL.override(-1):
            scheduler.idle_sleeper = None if noop else IdleSleeper([receiver])
        parallel = SimpleNamespace(
            world_group=SimpleNamespace(cpu_group=dist.group.WORLD)
        )
        sleep_probe = (
            nullcontext(Mock())
            if noop
            else patch.object(
                scheduler.idle_sleeper,
                "maybe_sleep",
                wraps=scheduler.idle_sleeper.maybe_sleep,
            )
        )
        with (
            patch(
                "sglang.srt.utils.poll_based_barrier.get_parallel",
                return_value=parallel,
            ),
            patch.object(dist, "all_reduce", wraps=dist.all_reduce) as collective,
            sleep_probe as socket_sleep,
        ):
            first = blocker.handle(packets)
            started = time.monotonic()
            scheduler.maybe_sleep_on_idle()
            second = blocker.handle(None if noop else [])
            elapsed = time.monotonic() - started
        connection.send(
            dict(
                first=first,
                second=second,
                socket_sleeps=socket_sleep.call_count,
                collectives=collective.call_count,
                elapsed=elapsed,
                pending=blocker.has_pending(),
            )
        )
    finally:
        connection.close()
        receiver.close(linger=0)
        context.term()
        if dist.is_initialized():
            dist.destroy_process_group()


class TestSchedulerInputBlocker(CustomTestCase):
    def test_asymmetric_tail_keeps_both_dp_leaders_polling(self):
        process_context = multiprocessing.get_context("spawn")
        processes = []
        connections = []
        context = zmq.Context()
        senders = []
        with tempfile.TemporaryDirectory() as directory:
            try:
                for rank in range(4):
                    parent, child = process_context.Pipe()
                    process = process_context.Process(
                        target=_asymmetric_tail_worker,
                        args=(rank, str(Path(directory) / "rendezvous"), child),
                    )
                    process.start()
                    child.close()
                    processes.append(process)
                    connections.append(parent)
                for rank, connection in enumerate(connections):
                    self.assertTrue(connection.poll(120), "DP rank did not start")
                    sender = context.socket(zmq.PUSH)
                    sender.connect(connection.recv())
                    senders.append(sender)
                    packets = None if rank % 2 else [_block(), _unblock()]
                    if rank == 0:
                        packets.append("tail")
                    sender.send_pyobj(packets)
                results = []
                for connection in connections:
                    self.assertTrue(connection.poll(60), "DP cohort did not drain")
                    results.append(connection.recv())
                self.assertEqual([r["first"] for r in results], [[], None, [], None])
                self.assertEqual(
                    [r["second"] for r in results], [["tail"], None, [], None]
                )
                self.assertEqual(
                    [r["socket_sleeps"] for r in results], [0, 0, 0, 0], results
                )
                self.assertEqual([r["collectives"] for r in results], [2, 2, 2, 2])
                self.assertEqual([r["pending"] for r in results], [False] * 4)
                for process in processes:
                    process.join(30)
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(10)
                for connection in connections:
                    connection.close()
                for sender in senders:
                    sender.close(linger=0)
                context.term()

    def test_unblocked_requests_pass_through(self):
        blocker, _ = _blocker(self)
        self.assertEqual(blocker.handle(["request"]), ["request"])

    def test_overlapping_guards_wait_for_both_unblocks(self):
        blocker, control = _blocker(self)
        self.assertEqual(blocker.handle([_block(), "a", _block(), "b"]), [])
        self.assertEqual(blocker.handle([_unblock(), "c"]), [])
        self.assertEqual(control.votes, [False, False])
        self.assertEqual(blocker.handle([_unblock()]), ["a", "b", "c"])
        self.assertEqual(blocker.handle([]), [])
        self.assertEqual(blocker.handle(["next"]), ["next"])

    def test_coalesced_interleaved_guards_release_requests_once(self):
        blocker, _ = _blocker(self)
        self.assertEqual(
            blocker.handle([_block(), "a", _block(), "b", _unblock(), "c", _unblock()]),
            ["a", "b", "c"],
        )
        self.assertEqual(blocker.handle([]), [])

    def test_coalesced_complete_batches_keep_separate_barriers(self):
        blocker, _ = _blocker(self)
        self.assertEqual(
            blocker.handle([_block(), "a", _unblock(), _block(), "b", _unblock()]),
            ["a"],
        )
        self.assertEqual(blocker.handle([]), ["b"])
        self.assertEqual(blocker.handle([_block(), "c", _unblock()]), ["c"])

    def test_new_guard_waits_for_pending_global_arrival(self):
        blocker, control = _blocker(self, global_ready=False)
        self.assertEqual(blocker.handle([_block(), "a", _unblock()]), [])
        self.assertEqual(control.votes, [True])

        control.global_ready = True
        self.assertEqual(blocker.handle([_block(), "b", _unblock()]), ["a"])
        self.assertEqual(control.votes, [True, True])
        self.assertEqual(blocker.handle([]), ["b"])
        self.assertEqual(blocker.handle([]), [])

    def test_asymmetric_receive_boundaries_preserve_barrier_order(self):
        first, first_control = _blocker(self)
        second, second_control = _blocker(self)
        batch_a = [_block(), "a", _unblock()]
        batch_b = [_block(), "b", _unblock()]

        self.assertEqual(first.handle(batch_a + batch_b), ["a"])
        self.assertEqual(second.handle(batch_a), ["a"])
        self.assertEqual(first.handle([]), ["b"])
        self.assertEqual(second.handle(batch_b), ["b"])
        self.assertEqual(first_control.votes, [True, True])
        self.assertEqual(second_control.votes, [True, True])
        self.assertEqual(first.handle([]), [])
        self.assertEqual(second.handle([]), [])

    def test_global_barrier_delays_completed_local_batch(self):
        blocker, control = _blocker(self, global_ready=False)
        self.assertEqual(blocker.handle([_block(), "a", _unblock()]), [])
        self.assertEqual(blocker.handle(["b"]), [])
        control.global_ready = True
        self.assertEqual(blocker.handle([]), ["a"])
        self.assertEqual(blocker.handle([]), ["b"])

    def test_noop_ranks_still_participate_in_every_collective(self):
        blocker, control = _blocker(self, noop=True)
        self.assertIsNone(blocker.handle(None))
        self.assertIsNone(blocker.handle(None))
        self.assertEqual(control.votes, [True, True])

    def test_unbalanced_unblock_is_rejected(self):
        blocker, _ = _blocker(self)
        with self.assertRaises(AssertionError):
            blocker.handle([_unblock()])
        blocker.handle([_block(), _unblock()])
        with self.assertRaises(AssertionError):
            blocker.handle([_unblock()])


class TestSchedulerInputBlockerPending(CustomTestCase):
    def _receiver(self, receiver_type, blocker, materializer=None):
        kwargs = dict(
            recv_from_tokenizer=None,
            recv_skipper=None,
            input_blocker=blocker,
            mm_receiver=None,
            model_config=None,
            max_recv_per_poll=-1,
            request_retirer=Mock(),
            get_last_batch=lambda: None,
            rust_server_mode=False,
        )
        if receiver_type is UocrRequestReceiver:
            kwargs.update(
                async_shm_materializer=materializer, submit_crop_prefetch=None
            )
        return receiver_type(**kwargs)

    def _scheduler(self, receiver):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = SimpleNamespace(is_empty=lambda: True)
        scheduler.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler._pp_microbatches_drained = lambda: True
        scheduler.request_lists = lambda: {}
        scheduler.request_receiver = receiver
        scheduler.uocr_runtime = None
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.input_blocker = receiver.input_blocker
        scheduler.idle_sleeper = Mock()
        return scheduler

    @patch("sglang.srt.managers.scheduler.time.sleep")
    def test_pending_input_keeps_polling_until_guards_and_deferred_tail_drain(
        self, sleep
    ):
        quiet_materializer = SimpleNamespace(has_pending=Mock(return_value=False))
        cases = (
            (SchedulerRequestReceiver, None),
            (UocrRequestReceiver, None),
            (UocrRequestReceiver, quiet_materializer),
        )
        for receiver_type, materializer in cases:
            with self.subTest(receiver=receiver_type, materializer=materializer):
                blocker, control = _blocker(self, global_ready=False)
                receiver = self._receiver(receiver_type, blocker, materializer)
                scheduler = self._scheduler(receiver)
                self.assertFalse(receiver.has_pending())
                self.assertTrue(scheduler.is_fully_idle())
                scheduler.maybe_sleep_on_idle()
                scheduler.idle_sleeper.maybe_sleep.assert_called_once_with()
                scheduler.idle_sleeper.maybe_sleep.reset_mock()

                for packets in ([_block()], ["request"], [_unblock()]):
                    self.assertEqual(blocker.handle(packets), [])
                    self.assertTrue(blocker.has_pending())
                    self.assertFalse(receiver.has_pending())
                    self.assertTrue(scheduler.is_fully_idle())
                    scheduler.maybe_sleep_on_idle()
                    scheduler.idle_sleeper.maybe_sleep.assert_not_called()
                    sleep.assert_called_with(
                        0
                        if isinstance(packets[0], BlockReqInput)
                        and packets[0].req_type == BlockReqType.UNBLOCK
                        else 0.001
                    )

                control.global_ready = True
                self.assertEqual(blocker.handle(["tail"]), ["request"])
                self.assertTrue(blocker.has_pending())
                self.assertFalse(receiver.has_pending())
                self.assertTrue(scheduler.is_fully_idle())
                scheduler.maybe_sleep_on_idle()
                scheduler.idle_sleeper.maybe_sleep.assert_not_called()

                self.assertEqual(blocker.handle([]), ["tail"])
                self.assertFalse(blocker.has_pending())
                self.assertFalse(receiver.has_pending())
                self.assertTrue(scheduler.is_fully_idle())
                scheduler.maybe_sleep_on_idle()
                scheduler.idle_sleeper.maybe_sleep.assert_called_once_with()

    def test_uocr_materializer_pending_survives_blocker_completion(self):
        blocker, _ = _blocker(self)
        materializer = SimpleNamespace(has_pending=Mock(return_value=True))
        receiver = self._receiver(UocrRequestReceiver, blocker, materializer)
        scheduler = self._scheduler(receiver)

        self.assertEqual(blocker.handle([_block(), "request", _unblock()]), ["request"])
        blocker.handle([])
        self.assertFalse(blocker.has_pending())
        self.assertTrue(receiver.has_pending())
        self.assertFalse(scheduler.is_fully_idle())

        materializer.has_pending.return_value = False
        self.assertFalse(receiver.has_pending())
        self.assertTrue(scheduler.is_fully_idle())

    def test_receivers_without_input_blocker_remain_idle(self):
        quiet_materializer = SimpleNamespace(has_pending=Mock(return_value=False))
        for receiver_type, materializer in (
            (SchedulerRequestReceiver, None),
            (UocrRequestReceiver, None),
            (UocrRequestReceiver, quiet_materializer),
        ):
            with self.subTest(receiver=receiver_type, materializer=materializer):
                receiver = self._receiver(receiver_type, None, materializer)
                self.assertFalse(receiver.has_pending())
                self.assertTrue(self._scheduler(receiver).is_fully_idle())

    def test_deferred_flush_clears_every_tp_rank_with_pending_guard(self):
        for receiver_type in (SchedulerRequestReceiver, UocrRequestReceiver):
            for stage in ("blocked", "barrier", "deferred"):
                with self.subTest(receiver=receiver_type, stage=stage):
                    schedulers = []
                    replies = []
                    wrappers = []
                    for noop in (False, True):
                        blocker, _ = _blocker(
                            self, noop=noop, global_ready=stage == "deferred"
                        )
                        packets = [_block()]
                        if stage != "blocked":
                            packets.append(_unblock())
                        if stage == "deferred":
                            packets.append("tail")
                        blocker.handle(None if noop else packets)
                        self.assertEqual(blocker.has_pending(), not noop)
                        scheduler = self._scheduler(
                            self._receiver(receiver_type, blocker)
                        )
                        scheduler.running_batch = Mock()
                        scheduler.running_batch.is_empty.return_value = False
                        scheduler.pause_state = PauseState()
                        scheduler.waiting_queue = []
                        scheduler.tree_cache = Mock()
                        scheduler.req_to_token_pool = Mock()
                        scheduler.token_to_kv_pool_allocator = Mock()
                        scheduler.grammar_manager = Mock()
                        scheduler.metrics_reporter = Mock()
                        scheduler.invariant_checker = Mock()
                        scheduler.gc_freezer = Mock()
                        scheduler.draft_worker = None
                        reply = Mock()
                        wrapper = SchedulerFlushWrapper(
                            flush_cache=lambda s=scheduler: s.flush_cache(
                                empty_cache=False
                            ),
                            is_fully_idle=scheduler.is_fully_idle,
                            send_output=reply,
                        )
                        self.assertIsNone(
                            wrapper.handle(FlushCacheReqInput(timeout_s=30))
                        )
                        schedulers.append(scheduler)
                        wrappers.append(wrapper)
                        replies.append(reply)

                    for scheduler, wrapper in zip(schedulers, wrappers):
                        scheduler.running_batch.is_empty.return_value = True
                        wrapper.check_pending()
                    self.assertEqual(
                        [s.tree_cache.reset.call_count for s in schedulers], [1, 1]
                    )
                    for scheduler, reply in zip(schedulers, replies):
                        scheduler.req_to_token_pool.clear.assert_called_once_with()
                        scheduler.token_to_kv_pool_allocator.clear.assert_called_once_with()
                        self.assertTrue(reply.call_args.args[0].success)


class TestColocatedHttpBatchGuards(CustomTestCase):
    async def _wait_for_sends(self, entered, tasks):
        waiter = asyncio.create_task(entered.wait())
        try:
            done, _ = await asyncio.wait(
                [waiter, *tasks], timeout=5, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
            self.assertIn(waiter, done, "Both batch senders did not start")
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    def _manager(self):
        blocker, _ = _blocker(self)
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.rid_to_state = {}
        manager._should_use_batch_tokenization = Mock(return_value=False)
        sent = []
        released = []

        def dispatch(packet):
            sent.append(packet)
            released.extend(blocker.handle([packet]))
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return future

        manager._dispatch_to_scheduler = dispatch
        manager._tokenize_one_request = AsyncMock(
            side_effect=lambda request: SimpleNamespace(rid=request.rid)
        )

        async def output(request, *args, **kwargs):
            yield {"meta_info": {"id": request.rid}}

        manager._wait_one_response = output
        return manager, sent, released

    def _batch(self, manager, prefix):
        request = GenerateReqInput(
            text=["first", "second"], rid=[prefix + "0", prefix + "1"]
        )
        request.normalize_batch_and_arguments()
        states = {rid: SimpleNamespace() for rid in request.rid}
        manager.rid_to_state.update(states)
        http = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        return manager._handle_batch_request(request, http, states)

    def test_concurrent_http_batches_balance_immediate_and_pending_sends(self):
        for pending in (False, True):
            with self.subTest(pending=pending):
                asyncio.run(self._concurrent_batches(pending))

    async def _concurrent_batches(self, pending):
        manager, sent, released = self._manager()
        entered = asyncio.Event()
        resume = asyncio.Event()
        started = set()

        async def send(request):
            manager._dispatch_to_scheduler(request)
            started.add(request.rid[0])
            if len(started) == 2:
                entered.set()
            if pending:
                await resume.wait()

        manager._send_one_request = send
        generators = [self._batch(manager, prefix) for prefix in ("a", "b")]
        tasks = []
        try:
            with envs.SGLANG_ENABLE_COLOCATED_BATCH_GEN.override(True):
                tasks = [asyncio.create_task(gen.__anext__()) for gen in generators]
                if pending:
                    await self._wait_for_sends(entered, tasks)
                    self.assertEqual(released, [])
                    resume.set()
                results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            self.assertEqual(
                [[row["meta_info"]["id"] for row in result] for result in results],
                [["a0", "a1"], ["b0", "b1"]],
            )
            self.assertCountEqual(
                [request.rid for request in released], ["a0", "a1", "b0", "b1"]
            )
            guards = [
                packet.req_type for packet in sent if isinstance(packet, BlockReqInput)
            ]
            nested = [
                BlockReqType.BLOCK,
                BlockReqType.BLOCK,
                BlockReqType.UNBLOCK,
                BlockReqType.UNBLOCK,
            ]
            sequential = [
                BlockReqType.BLOCK,
                BlockReqType.UNBLOCK,
                BlockReqType.BLOCK,
                BlockReqType.UNBLOCK,
            ]
            if pending:
                self.assertEqual(guards, nested)
            else:
                self.assertIn(guards, (sequential, nested))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(*(gen.aclose() for gen in generators))

    def test_failed_or_cancelled_batch_does_not_unblock_a_live_peer(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                asyncio.run(self._failed_batch(cancelled))

    async def _failed_batch(self, cancelled):
        manager, sent, released = self._manager()
        entered = asyncio.Event()
        fail = asyncio.Event()
        resume_peer = asyncio.Event()
        started = set()

        async def send(request):
            manager._dispatch_to_scheduler(request)
            started.add(request.rid[0])
            if len(started) == 2:
                entered.set()
            if request.rid.startswith("a"):
                await fail.wait()
                raise RuntimeError("dispatch failed")
            await resume_peer.wait()

        manager._send_one_request = send
        generators = [self._batch(manager, prefix) for prefix in ("a", "b")]
        tasks = []
        try:
            with envs.SGLANG_ENABLE_COLOCATED_BATCH_GEN.override(True):
                tasks = [asyncio.create_task(gen.__anext__()) for gen in generators]
                await self._wait_for_sends(entered, tasks)
                if cancelled:
                    tasks[0].cancel()
                else:
                    fail.set()
                with self.assertRaises(
                    asyncio.CancelledError if cancelled else RuntimeError
                ):
                    await asyncio.wait_for(tasks[0], timeout=5)
                self.assertEqual(released, [])
                resume_peer.set()
                result = await asyncio.wait_for(tasks[1], timeout=5)
            self.assertEqual([row["meta_info"]["id"] for row in result], ["b0", "b1"])
            self.assertCountEqual(
                [request.rid for request in released], ["a0", "b0", "b1"]
            )
            self.assertEqual(
                [
                    packet.req_type
                    for packet in sent
                    if isinstance(packet, BlockReqInput)
                ],
                [
                    BlockReqType.BLOCK,
                    BlockReqType.BLOCK,
                    BlockReqType.UNBLOCK,
                    BlockReqType.UNBLOCK,
                ],
            )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(*(gen.aclose() for gen in generators))


if __name__ == "__main__":
    unittest.main()
