"""Instance control protocol tests with real managers and a simulated GPU backend."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import io_struct as io
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiTokenizerRouter,
    TokenizerWorker,
)
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.managers.tokenizer_control import (
    CONTROL_RETURN_PREFIX,
    ControlCoordinator,
    ControlRpc,
    WorkerBroadcast,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.utils import TypeBasedDispatcher

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


async def _wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), 2)


class _ReceiveSocket:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def recv(self, **kwargs):
        return msgpack_encode(await self.queue.get())

    async def recv_pyobj(self, **kwargs):
        return await self.queue.get()


WEIGHT_CASES = [
    (
        "disk",
        io.UpdateWeightFromDiskReqInput(model_path="new-model", weight_version="disk"),
    ),
    (
        "tensor",
        io.UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"tensor"], weight_version="tensor"
        ),
    ),
    (
        "distributed",
        io.UpdateWeightsFromDistributedReqInput(
            names=[], dtypes=[], shapes=[], weight_version="distributed"
        ),
    ),
    ("ipc", io.UpdateWeightsFromIPCReqInput(zmq_handles={}, weight_version="ipc")),
]


class _ControlHarness:
    """Real managers, coordinator, RPC codec and response sender; fake GPU work."""

    def __init__(self, worker_count=2, ranks=2):
        self.context_override = get_context().override_server_args(
            tokenizer_worker_num=worker_count,
            dp_size=ranks,
            enable_dp_attention=True,
            weight_version="old",
            load_format="dummy",
            skip_tokenizer_init=True,
        )
        self.context_override.install()
        self.worker_count = worker_count
        self.ranks = ranks
        self.sent = []
        self.outputs = []
        self.held = []
        self.drop = set()
        self.auto_reply = True
        self.fail_ranks = set()
        self.idle = [True] * ranks
        self.backend_error = None
        self.tasks = set()
        self.workers = {}
        self.router = MultiTokenizerRouter.__new__(MultiTokenizerRouter)
        router = self.router
        router.server_args = SimpleNamespace(tokenizer_worker_num=worker_count)
        router.all_worker_ipcs = {f"worker-{i}" for i in range(worker_count)}
        router._control_tasks = set()
        router.receive_from_worker = _ReceiveSocket()
        router.send_to_scheduler = None
        router.socket_mapping = SimpleNamespace(send_output=self.send_output)
        router._control_rpc = ControlRpc(self.backend_send)
        router._worker_broadcast = WorkerBroadcast(
            lambda: router.all_worker_ipcs, self.send_output
        )
        router._control = ControlCoordinator(
            router._control_rpc.call,
            router._worker_broadcast.broadcast_and_wait,
            router._fail_control,
        )

        async def send_to_backend(_, obj):
            await self.backend_send(obj)

        self.send_patch = patch(
            "sglang.srt.managers.multi_tokenizer_mixin.async_sock_send",
            side_effect=send_to_backend,
        )
        self.send_patch.start()

        for ipc in sorted(router.all_worker_ipcs):
            cls = TokenizerManager if worker_count == 1 else TokenizerWorker
            worker = cls.__new__(cls)
            worker.tokenizer_ipc_name = ipc
            worker.server_args = SimpleNamespace(
                tokenizer_worker_num=worker_count,
                dp_size=ranks,
                enable_dp_attention=True,
                weight_version="old",
                load_format="dummy",
                skip_tokenizer_init=True,
            )
            worker.elastic_worker_count = ranks
            worker.is_pause = False
            worker.is_pause_cond = asyncio.Condition()
            worker.model_update_lock = RWLock()
            worker.rid_to_state = {}
            # Each actor represents a process with its own config bag.
            values = dict(weight_version="old", load_format="dummy")
            worker.record_config_updates = lambda source, values=values, **fields: (
                values.update(fields)
            )
            worker.config_value = lambda name, values=values: values[name]
            worker.mm_processor = Mock()
            worker.model_path = worker.served_model_name = "old-model"
            worker.initial_weights_loaded = False
            worker.enable_metrics = False
            worker.auto_create_handle_loop = lambda: None
            worker._result_dispatcher = TypeBasedDispatcher([])

            def dispatch(obj, ipc=ipc):
                obj.http_worker_ipc = ipc
                if self.worker_count == 1:
                    task = asyncio.create_task(self.backend_send(obj))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
                else:
                    # Both codecs are used in the protocol tests.
                    router.receive_from_worker.queue.put_nowait(
                        msgpack_decode(msgpack_encode(obj))
                    )

            async def async_dispatch(obj, dispatch=dispatch):
                dispatch(obj)

            worker._dispatch_to_scheduler = dispatch
            worker._async_dispatch_to_scheduler = async_dispatch
            worker._send_control_to_scheduler = self.backend_send
            worker.init_communicators()
            self.workers[ipc] = worker
        self.loop_task = asyncio.create_task(router.router_worker_obj())

    @property
    def control(self):
        return self.origin._control if self.worker_count == 1 else self.router._control

    @property
    def origin(self):
        return self.workers["worker-0"]

    async def backend_send(self, obj):
        self.sent.append(obj)
        if self.backend_error is not None:
            raise self.backend_error
        if not (obj.http_worker_ipc or "").startswith(CONTROL_RETURN_PREFIX):
            return
        if self.auto_reply:
            for rank in range(self.ranks):
                await self.reply(obj, rank)

    async def reply(self, request, rank):
        name = type(request).__name__
        if isinstance(request, io.UpdateWeightVersionReqInput):
            output = io.UpdateWeightVersionReqOutput()
        elif name.startswith("UpdateWeight"):
            output_cls = getattr(io, name.removesuffix("Input") + "Output")
            output = output_cls(
                success=rank not in self.fail_ranks, message="backend result"
            )
        elif isinstance(request, io.GetInternalStateReq):
            output = io.GetInternalStateReqOutput(
                internal_state={"is_fully_idle": self.idle[rank]}
            )
        elif isinstance(request, io.ReleaseMemoryOccupationReqInput):
            output = io.ReleaseMemoryOccupationReqOutput()
        elif isinstance(request, io.ResumeMemoryOccupationReqInput):
            output = io.ResumeMemoryOccupationReqOutput()
        else:
            output = io.TokenizerControlBackendAckReq()
        messages = []
        sender = SenderWrapper(object())
        sender.worker_id = f"rank-{rank}"
        with patch(
            "sglang.srt.managers.scheduler_components.output_sender.sock_send",
            side_effect=lambda _, msg: messages.append(msg),
        ):
            sender.send_output(output, request)
        wire = msgpack_decode(msgpack_encode(messages[0]))
        if self.worker_count == 1:
            self.origin._control_rpc.handle_recv(wire)
        else:
            await self.router._distribute_result_to_workers(wire)

    def send_output(self, ipc, obj):
        obj = msgpack_decode(msgpack_encode(obj))
        self.outputs.append((ipc, obj))
        if (ipc, getattr(obj, "action", None)) in self.drop:
            self.held.append((ipc, obj))
            return
        worker = self.workers[ipc]
        worker._result_dispatcher(obj)

    async def close(self):
        self.loop_task.cancel()
        tasks = self.tasks | self.router._control_tasks
        for worker in self.workers.values():
            tasks |= worker._control_tasks
        for task in tasks:
            task.cancel()
        await asyncio.gather(self.loop_task, *tasks, return_exceptions=True)
        # Cancellation can emit a final fence to the worker receive loops.
        late = set().union(*(w._control_tasks for w in self.workers.values()))
        if late:
            await asyncio.gather(*late, return_exceptions=True)
        self.send_patch.stop()
        self.context_override.restore()


class TestTokenizerControlProtocol(CustomTestCase, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.timeout_patch = patch(
            "sglang.srt.managers.tokenizer_control.CONTROL_TIMEOUT_SECONDS", 1.0
        )
        self.timeout_patch.start()
        self.h = _ControlHarness()
        self.a = self.h.origin
        self.b = self.h.workers["worker-1"]

    async def asyncTearDown(self):
        await self.h.close()
        self.timeout_patch.stop()

    def assert_worker_actions(self, expected):
        for ipc in self.h.workers:
            self.assertEqual(
                [
                    obj.action
                    for worker, obj in self.h.outputs
                    if worker == ipc
                    and isinstance(obj, io.TokenizerControlBroadcastReq)
                ],
                expected,
            )

    async def test_pause_and_continue_broadcast_only_the_required_state(self):
        for mode in ("in_place", "retract"):
            with self.subTest(mode=mode):
                self.h.outputs.clear()
                await self.a.pause_generation(PauseGenerationReqInput(mode=mode))
                self.assert_worker_actions(["pause"])
                self.h.outputs.clear()
                await self.b.continue_generation(ContinueGenerationReqInput())
                self.assert_worker_actions(["update_state"])
                self.assertFalse(self.a.is_pause or self.b.is_pause)

    async def test_continue_waits_for_backend_then_all_worker_acks(self):
        await self.a.pause_generation(PauseGenerationReqInput(mode="retract"))
        self.h.outputs.clear()
        self.h.sent.clear()
        self.h.auto_reply = False
        self.h.drop.add(("worker-1", "update_state"))
        task = asyncio.create_task(
            self.a.continue_generation(ContinueGenerationReqInput())
        )
        await _wait_until(lambda: bool(self.h.sent))
        self.assert_worker_actions([])
        request = self.h.sent[0]
        await self.h.reply(request, 0)
        self.assertTrue(self.a.is_pause and self.b.is_pause)
        await self.h.reply(request, 1)
        await _wait_until(lambda: bool(self.h.held))
        self.assertFalse(task.done())
        self.assertTrue(self.b.is_pause)
        self.h.drop.clear()
        self.h.send_output(*self.h.held.pop())
        await task
        self.assertFalse(self.a.is_pause or self.b.is_pause)

    async def test_paused_weight_updates_only_broadcast_the_new_state(self):
        await self.a.pause_generation(PauseGenerationReqInput(mode="retract"))
        for suffix, request in WEIGHT_CASES:
            with self.subTest(endpoint=suffix):
                self.h.outputs.clear()
                self.h.sent.clear()
                result = await getattr(self.b, "update_weights_from_" + suffix)(request)
                self.assertTrue(result[0], result)
                self.assert_worker_actions(["update_state"])
                self.assertEqual([type(req) for req in self.h.sent], [type(request)])
                self.assertTrue(self.a.is_pause and self.b.is_pause)
        self.h.outputs.clear()
        self.h.sent.clear()
        await self.a.update_weight_version(
            io.UpdateWeightVersionReqInput(
                new_version="manual", abort_all_requests=False
            )
        )
        self.assert_worker_actions(["update_state"])
        self.assertEqual(
            [type(req) for req in self.h.sent], [io.UpdateWeightVersionReqInput]
        )
        self.assertEqual(self.b.config_value("weight_version"), "manual")

    async def test_release_and_resume_only_broadcast_admission_changes(self):
        await self.a.release_memory_occupation(
            io.ReleaseMemoryOccupationReqInput(tags=["weights", "kv_cache"])
        )
        self.assert_worker_actions(["pause"])
        self.h.outputs.clear()
        await self.b.resume_memory_occupation(
            io.ResumeMemoryOccupationReqInput(tags=["weights"])
        )
        self.assert_worker_actions([])
        self.assertTrue(self.a.is_pause and self.b.is_pause)
        await self.a.resume_memory_occupation(
            io.ResumeMemoryOccupationReqInput(tags=["kv_cache"])
        )
        self.assert_worker_actions(["update_state"])
        self.assertFalse(self.a.is_pause or self.b.is_pause)
        await self.a.pause_generation(PauseGenerationReqInput(mode="retract"))
        self.h.outputs.clear()
        await self.b.release_memory_occupation(io.ReleaseMemoryOccupationReqInput())
        await self.a.resume_memory_occupation(io.ResumeMemoryOccupationReqInput())
        self.assert_worker_actions([])
        self.assertTrue(self.a.is_pause and self.b.is_pause)

    async def test_missing_pause_ack_never_reaches_the_backend(self):
        self.h.drop.add(("worker-1", "pause"))
        with patch(
            "sglang.srt.managers.tokenizer_control.CONTROL_TIMEOUT_SECONDS", 0.03
        ):
            result = await self.a.update_weights_from_tensor(
                io.UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[], weight_version="blocked"
                )
            )
        self.assertFalse(result[0])
        self.assertEqual(self.h.sent, [])
        await _wait_until(lambda: self.b._control_error is not None)
        self.assertEqual(self.h.router._worker_broadcast.pending, {})
        self.assertTrue(self.a.is_pause and self.b.is_pause)

    async def test_concurrent_broadcasts_collect_acks_independently(self):
        self.h.drop.add(("worker-1", "abort"))
        broadcast = self.h.router._worker_broadcast
        action = io.TokenizerControlBroadcastReq(action="abort", rid="shared-action")
        first = asyncio.create_task(broadcast.broadcast_and_wait(action))
        second = asyncio.create_task(broadcast.broadcast_and_wait(action))
        await _wait_until(lambda: len(self.h.held) == 2)
        first_msg, second_msg = [obj for _, obj in self.h.held]
        for worker in ("worker-0", "unknown-worker"):
            broadcast.handle_recv(
                io.TokenizerControlAckReq(
                    broadcast_id=first_msg.broadcast_id, worker_ipc_name=worker
                )
            )
        self.assertFalse(first.done() or second.done())
        self.h.drop.clear()
        self.h.send_output("worker-1", second_msg)
        await second
        self.assertFalse(first.done())
        self.h.send_output("worker-1", first_msg)
        await first
        self.assertEqual(broadcast.pending, {})

    async def test_abort_update_still_drains_when_already_paused(self):
        await self.a.pause_generation(PauseGenerationReqInput(mode="retract"))
        self.h.outputs.clear()
        self.h.sent.clear()
        await self.b.model_update_lock.acquire_reader()
        state = SimpleNamespace(abort_requested=False, finished=False)
        self.b.rid_to_state["retained"] = state
        task = asyncio.create_task(
            self.a.update_weights_from_disk(
                io.UpdateWeightFromDiskReqInput(
                    model_path="new-model",
                    weight_version="new",
                    abort_all_requests=True,
                )
            )
        )
        await _wait_until(
            lambda: any(
                isinstance(req, io.ContinueGenerationReqInput) for req in self.h.sent
            )
        )
        self.assertTrue(state.abort_requested)
        self.assertFalse(
            any(isinstance(req, io.UpdateWeightFromDiskReqInput) for req in self.h.sent)
        )
        self.b.rid_to_state.clear()
        await self.b.model_update_lock.release_reader()
        self.assertTrue((await task)[0])
        self.assert_worker_actions(["pause", "drain", "update_state"])
        self.assertTrue(self.a.is_pause and self.b.is_pause)
        self.assertEqual(self.b.config_value("weight_version"), "new")

    async def test_every_weight_endpoint_publishes_after_backend_success(self):
        for suffix, request in WEIGHT_CASES:
            with self.subTest(endpoint=suffix):
                caller = self.a if suffix in ("disk", "distributed") else self.b
                result = await getattr(caller, "update_weights_from_" + suffix)(request)
                self.assertTrue(result[0], result)
                for worker in self.h.workers.values():
                    self.assertEqual(worker.config_value("weight_version"), suffix)
                    self.assertEqual(worker.model_path, "new-model")
                    self.assertTrue(worker.initial_weights_loaded)
                    self.assertFalse(worker.is_pause)
                self.assertEqual(sum(type(r) is type(request) for r in self.h.sent), 1)
        self.assertEqual(
            self.a.mm_processor.clear_preprocess_cache.call_count, len(WEIGHT_CASES)
        )
        self.assertEqual(
            self.b.mm_processor.clear_preprocess_cache.call_count, len(WEIGHT_CASES)
        )
        await self.b.update_weight_version(
            io.UpdateWeightVersionReqInput(
                new_version="manual", abort_all_requests=False
            )
        )
        self.assertEqual(self.a.config_value("weight_version"), "manual")
        self.assertEqual(
            sum(type(r).__name__.startswith("UpdateWeight") for r in self.h.sent),
            len(WEIGHT_CASES) + 1,
        )

    async def test_version_only_update_preserves_active_generation(self):
        await self.b.model_update_lock.acquire_reader()
        try:
            await self.a.update_weight_version(
                io.UpdateWeightVersionReqInput(
                    new_version="v42", abort_all_requests=False
                )
            )
            self.assertTrue(await self.b.model_update_lock.is_locked())
            self.assertFalse(self.a.is_pause or self.b.is_pause)
            self.assertEqual(
                [type(req) for req in self.h.sent], [io.UpdateWeightVersionReqInput]
            )
            self.assert_worker_actions(["update_state"])
            for worker in self.h.workers.values():
                self.assertEqual(worker.config_value("weight_version"), "v42")
                worker.mm_processor.clear_preprocess_cache.assert_not_called()
        finally:
            await self.b.model_update_lock.release_reader()

    async def test_version_update_with_abort_waits_for_drain(self):
        state = SimpleNamespace(abort_requested=False, finished=False)
        self.b.rid_to_state["active"] = state
        task = asyncio.create_task(
            self.a.update_weight_version(
                io.UpdateWeightVersionReqInput(new_version="v42")
            )
        )
        await _wait_until(
            lambda: any(isinstance(req, io.AbortReq) for req in self.h.sent)
        )
        self.assertTrue(state.abort_requested)
        self.assertFalse(
            any(isinstance(req, io.UpdateWeightVersionReqInput) for req in self.h.sent)
        )
        state.finished = True
        await task
        self.assert_worker_actions(["pause", "drain", "update_state"])
        self.assertEqual(self.b.config_value("weight_version"), "v42")
        self.assertFalse(self.a.is_pause or self.b.is_pause)

    async def test_update_waits_for_sibling_readers_before_touching_backend(self):
        await self.b.model_update_lock.acquire_reader()
        task = asyncio.create_task(
            self.a.update_weights_from_tensor(
                io.UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[], weight_version="new"
                )
            )
        )
        await _wait_until(lambda: self.b.is_pause)
        self.assertEqual(self.h.sent, [])
        await self.b.model_update_lock.release_reader()
        self.assertTrue((await task)[0])
        self.assert_worker_actions(["pause", "update_state"])

    async def test_cancellation_does_not_release_operation_before_publication(self):
        self.h.drop.add(("worker-1", "update_state"))
        task = asyncio.create_task(
            self.a.update_weights_from_tensor(
                io.UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[], weight_version="158"
                )
            )
        )
        await _wait_until(lambda: bool(self.h.held))
        self.assertFalse(task.done())
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        failed = await self.b.update_weights_from_tensor(
            io.UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[], weight_version="159"
            )
        )
        self.assertFalse(failed[0])
        self.assertIn("in progress", failed[1])
        self.assertEqual(
            sum(type(r).__name__.startswith("UpdateWeight") for r in self.h.sent), 1
        )
        self.h.drop.clear()
        ipc, state_update = self.h.held.pop()
        self.h.send_output(ipc, state_update)
        await _wait_until(
            lambda: self.h.control.active is None and not self.a._control_futures
        )
        self.assertTrue(
            (
                await self.b.update_weights_from_tensor(
                    io.UpdateWeightsFromTensorReqInput(
                        serialized_named_tensors=[], weight_version="159"
                    )
                )
            )[0]
        )
        self.h.router._worker_broadcast.handle_recv(
            io.TokenizerControlAckReq(
                broadcast_id=state_update.broadcast_id, worker_ipc_name=ipc
            )
        )
        self.h.send_output(ipc, state_update)
        await asyncio.sleep(0.01)
        for worker in (self.a, self.b):
            self.assertEqual(worker.config_value("weight_version"), "159")

    async def test_missing_ack_fences_workers_and_late_state_cannot_reopen_them(self):
        self.h.drop.add(("worker-1", "update_state"))
        with patch(
            "sglang.srt.managers.tokenizer_control.CONTROL_TIMEOUT_SECONDS", 0.04
        ):
            result = await self.a.update_weights_from_tensor(
                io.UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[], weight_version="new"
                )
            )
        self.assertFalse(result[0])
        await _wait_until(lambda: all(w._control_error for w in (self.a, self.b)))
        self.assertIsNone(self.h.control.active)
        self.assertEqual(self.a._control_futures, {})
        self.h.drop.clear()
        ipc, state_update = self.h.held.pop()
        self.h.send_output(ipc, state_update)
        await asyncio.sleep(0.01)
        with self.assertRaisesRegex(RuntimeError, "restart the server"):
            await self.b.continue_generation(ContinueGenerationReqInput())
        self.assertTrue(self.a.is_pause and self.b.is_pause)
        self.assertEqual(self.b.config_value("weight_version"), "old")

    async def test_duplicate_and_late_backend_replies_do_not_complete_operations(self):
        await self.a.pause_generation(PauseGenerationReqInput(mode="retract"))
        self.h.sent.clear()
        self.h.auto_reply = False
        first = asyncio.create_task(
            self.a.update_weights_from_tensor(
                io.UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[], weight_version="first"
                )
            )
        )
        await _wait_until(lambda: len(self.h.sent) == 1)
        old = self.h.sent[0]
        await self.h.reply(old, 0)
        await self.h.reply(old, 0)
        self.assertFalse(first.done())
        await self.h.reply(old, 1)
        self.assertTrue((await first)[0])
        second = asyncio.create_task(
            self.b.update_weights_from_tensor(
                io.UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[], weight_version="second"
                )
            )
        )
        await _wait_until(lambda: len(self.h.sent) == 2)
        await self.h.reply(old, 1)
        self.assertFalse(second.done())
        await self.h.reply(self.h.sent[1], 0)
        self.assertFalse(second.done())
        await self.h.reply(self.h.sent[1], 1)
        self.assertTrue((await second)[0])
        self.assertEqual(self.a.config_value("weight_version"), "second")

    async def test_partial_backend_failure_does_not_publish_success(self):
        self.h.fail_ranks.add(1)
        result = await self.a.update_weights_from_tensor(
            io.UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[], weight_version="bad"
            )
        )
        self.assertFalse(result[0])
        await _wait_until(lambda: self.b._control_error is not None)
        for worker in (self.a, self.b):
            self.assertTrue(worker.is_pause)
            self.assertEqual(worker.config_value("weight_version"), "old")

    async def test_in_place_and_retract_keep_requests_and_wait_for_backend_ack(self):
        for mode in ("in_place", "retract"):
            with self.subTest(mode=mode):
                await self.b.model_update_lock.acquire_reader()
                self.h.auto_reply = False
                task = asyncio.create_task(
                    self.a.pause_generation(PauseGenerationReqInput(mode=mode))
                )
                count = len(self.h.sent)
                await _wait_until(lambda: len(self.h.sent) > count)
                self.assertTrue(self.b.is_pause)
                self.assertFalse(task.done())
                req = self.h.sent[-1]
                await self.h.reply(req, 0)
                self.assertFalse(task.done())
                await self.h.reply(req, 1)
                await task
                self.h.auto_reply = True
                # The retained reader must not deadlock a paused weight update.
                self.assertTrue(
                    (
                        await self.b.update_weights_from_tensor(
                            io.UpdateWeightsFromTensorReqInput(
                                serialized_named_tensors=[], weight_version=mode
                            )
                        )
                    )[0]
                )
                await self.a.continue_generation(ContinueGenerationReqInput())
                self.assertFalse(self.a.is_pause or self.b.is_pause)
                await self.b.model_update_lock.release_reader()

    async def test_abort_pause_waits_for_local_drain_and_all_scheduler_ranks(self):
        await self.b.model_update_lock.acquire_reader()
        self.h.idle[1] = False
        task = asyncio.create_task(
            self.a.pause_generation(PauseGenerationReqInput(mode="abort"))
        )
        await _wait_until(lambda: any(isinstance(r, io.AbortReq) for r in self.h.sent))
        with self.assertRaisesRegex(RuntimeError, "in progress"):
            await self.b.continue_generation(ContinueGenerationReqInput())
        self.assertFalse(task.done())
        await self.b.model_update_lock.release_reader()
        await _wait_until(
            lambda: any(isinstance(r, io.GetInternalStateReq) for r in self.h.sent)
        )
        self.assertFalse(task.done())
        self.h.idle[1] = True
        await task
        self.assertEqual(sum(isinstance(r, io.AbortReq) for r in self.h.sent), 1)
        self.assert_worker_actions(["pause", "drain"])
        await self.b.continue_generation(ContinueGenerationReqInput())

    async def test_abort_is_not_blocked_by_a_waiting_mutation(self):
        await self.b.model_update_lock.acquire_reader()
        update = asyncio.create_task(
            self.a.update_weights_from_tensor(
                io.UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[], weight_version="new"
                )
            )
        )
        await _wait_until(lambda: self.b.is_pause)
        state = SimpleNamespace(abort_requested=False)
        self.b.rid_to_state["owned-by-b"] = state
        self.a.abort_request("owned-by-b")
        await _wait_until(lambda: state.abort_requested)
        await _wait_until(
            lambda: any(
                isinstance(r, io.AbortReq) and r.rid == "owned-by-b"
                for r in self.h.sent
            )
        )
        self.b.rid_to_state.clear()
        await self.b.model_update_lock.release_reader()
        self.assertTrue((await update)[0])

    async def test_memory_must_be_resumed_before_generation(self):
        await self.a.release_memory_occupation(
            io.ReleaseMemoryOccupationReqInput(tags=["weights", "kv_cache"])
        )
        with self.assertRaisesRegex(RuntimeError, "Resume memory"):
            await self.b.continue_generation(ContinueGenerationReqInput())
        await self.b.resume_memory_occupation(
            io.ResumeMemoryOccupationReqInput(tags=["weights"])
        )
        with self.assertRaisesRegex(RuntimeError, "kv_cache"):
            await self.a.continue_generation(ContinueGenerationReqInput())
        await self.a.resume_memory_occupation(
            io.ResumeMemoryOccupationReqInput(tags=["kv_cache"])
        )
        await self.b.continue_generation(ContinueGenerationReqInput())
        self.assertFalse(self.a.is_pause or self.b.is_pause)

    async def test_duplicate_pause_cannot_ack_an_unfinished_drain(self):
        await self.b.model_update_lock.acquire_reader()
        task = asyncio.create_task(
            self.a.update_weights_from_tensor(
                io.UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[], weight_version="new"
                )
            )
        )
        await _wait_until(lambda: self.b.is_pause)
        pause = next(
            obj
            for ipc, obj in self.h.outputs
            if ipc == "worker-1" and getattr(obj, "action", None) == "pause"
        )
        self.h.send_output("worker-1", pause)
        await asyncio.sleep(0.02)
        self.assertEqual(self.h.sent, [])
        await self.b.model_update_lock.release_reader()
        self.assertTrue((await task)[0])

    async def test_abort_waits_for_frontend_markers_before_backend_dispatch(self):
        self.h.drop.add(("worker-1", "abort"))
        self.a.abort_request("request")
        await _wait_until(lambda: bool(self.h.held))
        self.assertFalse(any(isinstance(req, io.AbortReq) for req in self.h.sent))
        self.h.drop.clear()
        self.h.send_output(*self.h.held.pop())
        await _wait_until(
            lambda: any(isinstance(req, io.AbortReq) for req in self.h.sent)
        )
        self.assertEqual(sum(isinstance(req, io.AbortReq) for req in self.h.sent), 1)

    async def test_backend_timeout_fences_and_ignores_late_success(self):
        await self.a.pause_generation(PauseGenerationReqInput(mode="retract"))
        self.h.auto_reply = False
        with patch(
            "sglang.srt.managers.tokenizer_control.CONTROL_TIMEOUT_SECONDS", 0.03
        ):
            result = await self.a.update_weights_from_tensor(
                io.UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[], weight_version="unknown"
                )
            )
        self.assertFalse(result[0])
        self.assertIn("backend completion", result[1])
        request = self.h.sent[-1]
        for rank in range(self.h.ranks):
            await self.h.reply(request, rank)
        self.assertEqual(self.h.router._control_rpc.pending, {})
        self.assertEqual(self.a.config_value("weight_version"), "old")
        self.assertEqual(self.b.config_value("weight_version"), "old")

    async def test_resume_memory_restores_the_prior_generation_state(self):
        for paused in (False, True):
            if paused:
                await self.a.pause_generation(PauseGenerationReqInput(mode="retract"))
            await self.b.release_memory_occupation(
                io.ReleaseMemoryOccupationReqInput(tags=[])
            )
            self.assertTrue(self.a.is_pause and self.b.is_pause)
            await self.a.resume_memory_occupation(
                io.ResumeMemoryOccupationReqInput(tags=[])
            )
            self.assertEqual(self.a.is_pause, paused)
            self.assertEqual(self.b.is_pause, paused)

    async def test_registration_failure_has_no_side_effects(self):
        self.h.router.all_worker_ipcs.remove("worker-1")
        with self.assertRaisesRegex(RuntimeError, "registration incomplete"):
            await self.a.pause_generation(PauseGenerationReqInput())
        self.assertFalse(self.a.is_pause or self.b.is_pause)
        self.assertEqual(self.h.sent, [])

    async def test_transport_failure_cleans_pending_rpc_and_fences_admission(self):
        self.h.backend_error = RuntimeError("scheduler send failed")
        with self.assertRaisesRegex(RuntimeError, "scheduler send failed"):
            await self.a.pause_generation(PauseGenerationReqInput(mode="retract"))
        await _wait_until(lambda: self.b._control_error is not None)
        self.assertEqual(self.h.router._control_rpc.pending, {})
        self.assertIsNone(self.h.control.active)


class TestWorkerBroadcast(CustomTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_synchronous_acks_use_the_original_participant_snapshot(self):
        workers = {"worker-0", "worker-1"}
        received = set()

        def send(worker, obj):
            workers.discard(worker)
            received.add(worker)
            broadcast.handle_recv(
                io.TokenizerControlAckReq(
                    broadcast_id=obj.broadcast_id, worker_ipc_name=worker
                )
            )

        broadcast = WorkerBroadcast(lambda: workers, send)
        await broadcast.broadcast_and_wait(
            io.TokenizerControlBroadcastReq(action="abort", rid="request")
        )
        self.assertEqual(received, {"worker-0", "worker-1"})
        self.assertEqual(broadcast.pending, {})


class TestSingleTokenizerControl(CustomTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_single_worker_uses_the_same_completion_contract(self):
        h = _ControlHarness(worker_count=1)
        try:
            await h.origin.pause_generation(PauseGenerationReqInput(mode="retract"))
            self.assertTrue(h.origin.is_pause)
            for suffix, request in WEIGHT_CASES:
                result = await getattr(h.origin, "update_weights_from_" + suffix)(
                    request
                )
                self.assertTrue(result[0])
                self.assertTrue(h.origin.is_pause)
                self.assertEqual(h.origin.config_value("weight_version"), suffix)
            await h.origin.continue_generation(ContinueGenerationReqInput())
            self.assertFalse(h.origin.is_pause)
            await h.origin.pause_generation(PauseGenerationReqInput(mode="abort"))
            self.assertTrue(h.origin.is_pause)
        finally:
            await h.close()


if __name__ == "__main__":
    unittest.main()
