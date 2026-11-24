"""Unit tests for TokenizerControlMixin request normalization."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import msgspec

from sglang.srt.managers import tokenizer_control_mixin
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    BeginWeightUpdateReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightVersionReqOutput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
    _WeightUpdateSession,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _QuietProfilePlatform(SRTPlatform):
    def apply_profile_request_defaults(self, req: ProfileReq) -> None:
        if req.with_stack is None:
            req.with_stack = False
        if req.record_shapes is None:
            req.record_shapes = False
        if req.num_steps is None:
            req.num_steps = 2


class TestStartProfilePlatformDefaults(CustomTestCase):
    def _start_profile(self, req):
        manager = MagicMock()
        manager._execute_profile = AsyncMock(
            return_value=ProfileReqOutput(success=True, message="")
        )
        with (
            patch.object(
                tokenizer_control_mixin, "current_platform", _QuietProfilePlatform()
            ),
            patch.dict(
                "os.environ",
                {
                    "SGLANG_PROFILE_WITH_STACK": "true",
                    "SGLANG_PROFILE_RECORD_SHAPES": "true",
                },
            ),
        ):
            asyncio.run(TokenizerControlMixin.start_profile(manager, req))
        return manager._execute_profile.call_args.args[0]

    def test_platform_sees_unset_fields_before_env_defaults(self):
        """Platform defaults apply only to fields the caller left unset;
        the env defaults must not fill them first."""
        for req in (None, ProfileReq()):
            with self.subTest(req=req):
                sent = self._start_profile(req)
                self.assertEqual(sent.req_type, ProfileReqType.START_PROFILE)
                self.assertFalse(sent.with_stack)
                self.assertFalse(sent.record_shapes)
                self.assertEqual(sent.num_steps, 2)


class TestGlobalNativeWeightVersion(CustomTestCase):
    def test_legacy_disk_reply_keeps_its_paused_request_count(self):
        """The IPC schema is array encoded: inserting load metadata before an
        existing field would reinterpret the old paused-request count as a bool."""
        legacy = msgspec.msgpack.encode(
            ["UpdateWeightFromDiskReqOutput", None, None, True, "Success", 7]
        )
        reply = msgspec.msgpack.decode(legacy, type=UpdateWeightFromDiskReqOutput)
        self.assertEqual(reply.num_paused_requests, 7)
        self.assertFalse(reply.weights_loaded)

    _requests = {
        "disk": lambda version: UpdateWeightFromDiskReqInput(
            model_path="checkpoint", weight_version=version
        ),
        "distributed": lambda version: UpdateWeightsFromDistributedReqInput(
            names=[], dtypes=[], shapes=[], weight_version=version
        ),
        "tensor": lambda version: UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b""], weight_version=version
        ),
        "ipc": lambda version: UpdateWeightsFromIPCReqInput(
            zmq_handles={}, weight_version=version
        ),
    }

    def _manager(
        self, route, failed_rank=None, *, paused=False, commit_gate=None, worker_count=2
    ):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.auto_create_handle_loop = MagicMock()
        manager.is_pause_cond = asyncio.Condition()
        manager.is_pause = paused
        manager.model_update_lock = RWLock()
        manager.mm_processor = None
        manager.elastic_worker_count = worker_count
        manager._update_model_path_info = MagicMock()
        states = [
            SimpleNamespace(version="v1", weights="v1") for _ in range(worker_count)
        ]
        state = SimpleNamespace(
            failed_rank=failed_rank,
            checkpoint="v2",
            commits=0,
            flush_ok=True,
            received=0,
        )
        tasks = []
        updates = []
        commit_started = asyncio.Event()
        for rank, resident in enumerate(states):

            def load(*args, rank=rank, resident=resident, **kwargs):
                if rank == state.failed_rank:
                    return False, "checkpoint missing on this shard"
                resident.weights = state.checkpoint
                return True, "Success"

            def record(new_version, resident=resident):
                if new_version is not None:
                    resident.version = new_version

            def receive(**kwargs):
                state.received += 1
                return []

            updater = SimpleNamespace(
                **{
                    name: load
                    for name in (
                        "update_weights_from_disk",
                        "update_weights_from_tensor",
                        "update_weights_from_ipc",
                        "load_weights_from_distributed",
                    )
                },
                receive_weights_from_distributed=receive,
            )
            runner = SimpleNamespace(weight_updater=updater, device="cpu")
            native = SchedulerWeightUpdaterManager(
                tp_worker=SimpleNamespace(
                    model_runner=runner,
                    weight_update_runners=lambda runner=runner: [("target", runner)],
                    deserialize_own_rank=lambda payload: [],
                ),
                draft_worker=None,
                memory_saver_adapter=None,
                flush_cache=lambda **kwargs: state.flush_ok,
                is_fully_idle=lambda **kwargs: True,
                scheduler=SimpleNamespace(
                    record_weight_version_change=record,
                    wait_for_launched_forward=lambda: None,
                ),
            )
            native._session = _WeightUpdateSession(selector="all")
            updates.append(native)
        state.native_updates = updates

        async def deliver(req):
            for native in updates:
                output = getattr(native, f"update_weights_from_{route}")(req)
                if route == "disk":
                    manager._handle_update_weights_from_disk_req_output(output)
                else:
                    communicator.handle_recv(output)
                await asyncio.sleep(0)

        def send(req):
            tasks.append(asyncio.create_task(deliver(req)))

        communicator = FanOutCommunicator(send, worker_count)
        if route == "disk":
            manager._dispatch_to_scheduler = send
        else:
            setattr(manager, f"update_weights_from_{route}_communicator", communicator)

        async def commit(req):
            state.commits += 1
            self.assertTrue(all(s.weights == state.checkpoint for s in states))
            commit_started.set()
            if commit_gate is not None:
                await commit_gate.wait()
            for resident in states:
                resident.version = req.new_version
            return [UpdateWeightVersionReqOutput() for _ in states]

        manager.update_weight_version_communicator = commit
        return manager, states, state, tasks, commit_started

    def test_failed_shard_then_unversioned_recovery_keeps_the_old_token_version(self):
        """One shard's successful load must not label restored v1 weights as v2
        after another shard made the global update fail."""

        async def scenario(route, failed_rank):
            manager, residents, state, tasks, _ = self._manager(route, failed_rank)
            update = getattr(manager, f"update_weights_from_{route}")
            request = self._requests[route]("v2")
            result = await asyncio.wait_for(update(request), timeout=5)
            self.assertFalse(result[0])
            self.assertEqual([s.version for s in residents], ["v1", "v1"])
            self.assertEqual(manager.config_value("weight_version"), "v1")
            self.assertEqual(state.commits, 0)
            self.assertEqual(request.weight_version, "v2")
            state.failed_rank, state.checkpoint = None, "v1"
            result = await asyncio.wait_for(
                update(self._requests[route](None)), timeout=5
            )
            self.assertTrue(result[0])
            self.assertTrue(all(s.weights == s.version == "v1" for s in residents))
            self.assertEqual(state.commits, 0)
            await asyncio.gather(*tasks)

        with (
            get_context().override_server_args(
                weight_version="v1", tp_size=2, attn_dp_size=2
            ),
            get_parallel().override(
                tp_rank=0,
                tp_size=2,
                dp_size=1,
                num_dp_ranks=2,
                attn_dp_enabled=True,
                enable_dp_attention_local_control_broadcast=True,
                tp_group=SimpleNamespace(cpu_group=None),
                attn_tp_group=SimpleNamespace(world_size=1),
                attn_cp_group=SimpleNamespace(world_size=1),
            ),
        ):
            for route in self._requests:
                for failed_rank in (0, 1):
                    with self.subTest(route=route, failed_rank=failed_rank):
                        asyncio.run(scenario(route, failed_rank))

    def test_success_waits_for_version_commit_before_unlocking_or_resuming(self):
        """The frontend must keep generation excluded through the second phase,
        including a caller trying to resume an already paused engine."""

        async def scenario(route, paused):
            gate = asyncio.Event()
            manager, residents, state, tasks, started = self._manager(
                route, paused=paused, commit_gate=gate
            )
            if paused:
                await manager.model_update_lock.acquire_writer()
            update = asyncio.create_task(
                getattr(manager, f"update_weights_from_{route}")(
                    self._requests[route]("v2")
                )
            )
            waiter = None
            try:
                # Yield until the update reaches commit, or finishes on the old code.
                async def reach_commit():
                    while not started.is_set() and not update.done():
                        await asyncio.sleep(0)

                await asyncio.wait_for(reach_commit(), timeout=5)
                self.assertTrue(
                    started.is_set(), "successful global load skipped version commit"
                )
                admitted = asyncio.Event()

                async def resume_or_generate():
                    lock = (
                        manager.is_pause_cond
                        if paused
                        else manager.model_update_lock.reader_lock
                    )
                    async with lock:
                        admitted.set()

                waiter = asyncio.create_task(resume_or_generate())
                await asyncio.sleep(0)
                self.assertFalse(admitted.is_set())
                self.assertFalse(update.done())
                gate.set()
                self.assertTrue((await asyncio.wait_for(update, timeout=5))[0])
                await asyncio.wait_for(waiter, timeout=5)
                self.assertTrue(all(s.version == "v2" for s in residents))
                self.assertEqual(manager.config_value("weight_version"), "v2")
                self.assertEqual(state.commits, 1)
            finally:
                gate.set()
                if not started.is_set() and not update.done():
                    update.cancel()
                await asyncio.gather(
                    update,
                    *tasks,
                    *([waiter] if waiter else []),
                    return_exceptions=True,
                )
                if paused:
                    await manager.model_update_lock.release_writer()

        for route in self._requests:
            for paused in (False, True):
                with (
                    self.subTest(route=route, paused=paused),
                    get_context().override_server_args(
                        weight_version="v1", tp_size=2, attn_dp_size=2
                    ),
                    get_parallel().override(
                        tp_rank=0,
                        tp_size=2,
                        dp_size=1,
                        num_dp_ranks=2,
                        attn_dp_enabled=True,
                        enable_dp_attention_local_control_broadcast=True,
                        tp_group=SimpleNamespace(cpu_group=None),
                        attn_tp_group=SimpleNamespace(world_size=1),
                        attn_cp_group=SimpleNamespace(world_size=1),
                    ),
                ):
                    asyncio.run(scenario(route, paused))

    def test_single_replica_keeps_its_local_commit_without_an_extra_fanout(self):
        """Deferral must not lose the native version or add a round trip on
        every tensor bucket when only one replica participates."""

        async def scenario(route):
            manager, residents, state, tasks, _ = self._manager(route, worker_count=1)
            result = await asyncio.wait_for(
                getattr(manager, f"update_weights_from_{route}")(
                    self._requests[route]("v2")
                ),
                timeout=5,
            )
            self.assertTrue(result[0])
            self.assertEqual(residents[0].version, "v2")
            self.assertEqual(manager.config_value("weight_version"), "v2")
            self.assertEqual(state.commits, 0)
            await asyncio.gather(*tasks)

        for route in self._requests:
            with (
                self.subTest(route=route),
                get_context().override_server_args(weight_version="v1"),
                get_parallel().override(
                    tp_rank=0,
                    tp_size=1,
                    dp_size=1,
                    num_dp_ranks=1,
                    attn_dp_enabled=False,
                    tp_group=SimpleNamespace(cpu_group=None, world_size=1),
                ),
            ):
                asyncio.run(scenario(route))

    def test_flush_refusal_still_commits_the_version_of_successfully_loaded_weights(
        self,
    ):
        """A paused v2 load can fail only at cache flush; later flushing must
        not leave fully installed v2 weights tagged as v1."""

        async def scenario(route, worker_count):
            manager, residents, state, tasks, _ = self._manager(
                route, paused=True, worker_count=worker_count
            )
            state.flush_ok = False
            result = await asyncio.wait_for(
                getattr(manager, f"update_weights_from_{route}")(
                    self._requests[route]("v2")
                ),
                timeout=5,
            )
            self.assertFalse(result[0])
            self.assertIn("cache was not flushed", result[1])
            self.assertTrue(all(s.weights == s.version == "v2" for s in residents))
            self.assertEqual(manager.config_value("weight_version"), "v2")
            self.assertTrue(manager.is_pause)
            self.assertEqual(state.commits, int(worker_count > 1))
            await asyncio.gather(*tasks)

        for route in self._requests:
            for workers in (1, 2):
                with (
                    self.subTest(route=route, workers=workers),
                    get_context().override_server_args(
                        weight_version="v1", tp_size=workers, attn_dp_size=workers
                    ),
                    get_parallel().override(
                        tp_rank=0,
                        tp_size=workers,
                        dp_size=1,
                        num_dp_ranks=workers,
                        pp_size=1,
                        attn_dp_enabled=True,
                        enable_dp_attention_local_control_broadcast=True,
                        tp_group=SimpleNamespace(cpu_group=None),
                        attn_tp_group=SimpleNamespace(world_size=1),
                        attn_cp_group=SimpleNamespace(world_size=1),
                    ),
                ):
                    asyncio.run(scenario(route, workers))

    def test_unsupported_global_version_configs_are_rejected_before_loading(self):
        """A local lock and stage-zero replies cannot protect other tokenizer
        workers or prove a later pipeline stage loaded the checkpoint."""

        async def scenario(route, reason):
            manager, residents, state, tasks, _ = self._manager(route)
            manager.abort_request = MagicMock()
            request = self._requests[route]("v2")
            if hasattr(request, "abort_all_requests"):
                request.abort_all_requests = True
            result = await asyncio.wait_for(
                getattr(manager, f"update_weights_from_{route}")(request), timeout=5
            )
            self.assertFalse(result[0])
            self.assertIn(reason, result[1])
            self.assertTrue(all(s.weights == s.version == "v1" for s in residents))
            self.assertEqual(manager.config_value("weight_version"), "v1")
            self.assertEqual(state.commits, 0)
            if route == "distributed":
                self.assertEqual(state.received, len(residents))
                await asyncio.gather(*tasks)
            else:
                self.assertEqual(tasks, [])
            manager.abort_request.assert_not_called()

        for route in self._requests:
            for pp_size, tokenizer_workers, reason in (
                (2, 1, "pipeline parallelism"),
                (1, 2, "tokenizer workers"),
            ):
                with (
                    self.subTest(
                        route=route,
                        pp_size=pp_size,
                        tokenizer_workers=tokenizer_workers,
                    ),
                    get_context().override_server_args(
                        weight_version="v1",
                        tp_size=2,
                        attn_dp_size=2,
                        tokenizer_worker_num=tokenizer_workers,
                    ),
                    get_parallel().override(
                        tp_rank=0,
                        tp_size=2,
                        dp_size=1,
                        num_dp_ranks=2,
                        pp_size=pp_size,
                        attn_dp_enabled=True,
                        enable_dp_attention_local_control_broadcast=True,
                        tp_group=SimpleNamespace(cpu_group=None),
                        attn_tp_group=SimpleNamespace(world_size=1),
                        attn_cp_group=SimpleNamespace(world_size=1),
                    ),
                ):
                    asyncio.run(scenario(route, reason))

    def test_single_dp_pipeline_reply_cannot_publish_checkpoint_metadata(self):
        """Stage-zero load/flush results cannot establish which checkpoint a
        later pipeline stage loaded, even with only one DP leader."""

        async def scenario(route, version):
            manager, residents, state, tasks, _ = self._manager(
                route, worker_count=1, paused=True
            )
            manager.model_path = manager.served_model_name = "old-checkpoint"
            manager._update_model_path_info = (
                TokenizerManager._update_model_path_info.__get__(manager)
            )
            state.flush_ok = False
            result = await asyncio.wait_for(
                getattr(manager, f"update_weights_from_{route}")(
                    self._requests[route](version)
                ),
                timeout=5,
            )
            self.assertFalse(result[0])
            self.assertIn("pipeline parallelism", result[1])
            self.assertEqual(manager.model_path, "old-checkpoint")
            self.assertEqual(manager.config_value("weight_version"), "v1")
            self.assertTrue(all(s.weights == s.version == "v1" for s in residents))
            await asyncio.gather(*tasks)

        for route in self._requests:
            for version in (None, "v2"):
                with (
                    self.subTest(route=route, version=version),
                    get_context().override_server_args(weight_version="v1"),
                    get_parallel().override(
                        tp_rank=0,
                        tp_size=1,
                        dp_size=1,
                        num_dp_ranks=1,
                        pp_size=2,
                        attn_dp_enabled=False,
                        tp_group=SimpleNamespace(cpu_group=None, world_size=1),
                    ),
                ):
                    asyncio.run(scenario(route, version))

    def test_rejected_distributed_round_still_completes_an_already_started_send(self):
        """Rejecting before posting receive strands a trainer that is already
        broadcasting; drain into temporary buffers without loading or locking out live requests."""

        async def scenario():
            manager, residents, state, tasks, _ = self._manager("distributed")
            for native in state.native_updates:
                native._session = None
            async with manager.model_update_lock.reader_lock:
                result = await asyncio.wait_for(
                    manager.update_weights_from_distributed(
                        self._requests["distributed"]("v2")
                    ),
                    timeout=5,
                )
            self.assertFalse(result[0])
            self.assertIn("tokenizer workers", result[1])
            self.assertEqual(
                state.received,
                2,
                "both scheduler receives must match the sender's round",
            )
            self.assertTrue(all(s.weights == s.version == "v1" for s in residents))
            self.assertTrue(all(n._session is None for n in state.native_updates))
            self.assertEqual(state.commits, 0)
            await asyncio.gather(*tasks)

        with (
            get_context().override_server_args(
                weight_version="v1", tp_size=2, attn_dp_size=2, tokenizer_worker_num=2
            ),
            get_parallel().override(
                tp_rank=0,
                tp_size=2,
                dp_size=1,
                num_dp_ranks=2,
                pp_size=1,
                attn_dp_enabled=True,
                enable_dp_attention_local_control_broadcast=True,
                tp_group=SimpleNamespace(cpu_group=None),
                attn_tp_group=SimpleNamespace(world_size=1),
                attn_cp_group=SimpleNamespace(world_size=1),
            ),
        ):
            asyncio.run(scenario())

    def test_version_preflight_rejects_before_the_trainer_enters_broadcast(self):
        """A trainer declaring its intended version can stop at preparation,
        before restoring weights, starting its collective or opening a session."""

        async def scenario():
            manager, residents, state, tasks, _ = self._manager("distributed")
            manager.begin_weight_update_communicator = AsyncMock(
                return_value=[BeginWeightUpdateReqOutput(success=True, message="ok")]
            )
            sender_started = False
            result = await manager.begin_weight_update(
                BeginWeightUpdateReqInput(weight_version="v2")
            )
            if result[0]:
                sender_started = True
            self.assertFalse(result[0])
            self.assertFalse(sender_started)
            self.assertIn("tokenizer workers", result[1])
            self.assertEqual(state.received, 0)
            self.assertEqual(tasks, [])
            self.assertTrue(all(s.weights == s.version == "v1" for s in residents))
            manager.begin_weight_update_communicator.assert_not_awaited()

        with (
            get_context().override_server_args(
                weight_version="v1", tp_size=2, attn_dp_size=2, tokenizer_worker_num=2
            ),
            get_parallel().override(
                tp_rank=0,
                tp_size=2,
                num_dp_ranks=2,
                pp_size=1,
                dp_size=1,
                tp_group=SimpleNamespace(cpu_group=None),
            ),
        ):
            asyncio.run(scenario())

    def test_accepted_preflight_does_not_publish_a_version_before_a_load(self):
        """The version in preparation is capability metadata, not evidence
        that any model weights have changed."""

        async def scenario():
            manager, residents, state, tasks, _ = self._manager("distributed")
            manager.begin_weight_update_communicator = AsyncMock(
                return_value=[
                    BeginWeightUpdateReqOutput(success=True, message="ok")
                    for _ in residents
                ]
            )
            result = await manager.begin_weight_update(
                BeginWeightUpdateReqInput(weight_version="v2")
            )
            self.assertTrue(result[0])
            self.assertEqual(manager.config_value("weight_version"), "v1")
            self.assertTrue(all(s.weights == s.version == "v1" for s in residents))
            self.assertEqual(state.commits, 0)
            self.assertEqual(tasks, [])

        with (
            get_context().override_server_args(
                weight_version="v1", tp_size=2, attn_dp_size=2
            ),
            get_parallel().override(
                tp_rank=0,
                tp_size=2,
                num_dp_ranks=2,
                pp_size=1,
                dp_size=1,
                tp_group=SimpleNamespace(cpu_group=None),
            ),
        ):
            asyncio.run(scenario())

    def test_engine_passes_the_preflight_and_load_versions_without_early_commit(self):
        """The SDK must carry the intended version to both phases; dropping
        either argument bypasses preflight or leaves the loaded version stale."""
        from sglang.srt.entrypoints.engine import Engine

        with (
            get_context().override_server_args(weight_version="v1"),
            get_parallel().override(
                tp_rank=0,
                tp_size=1,
                num_dp_ranks=1,
                pp_size=1,
                dp_size=1,
                attn_dp_enabled=False,
                tp_group=SimpleNamespace(cpu_group=None, world_size=1),
            ),
        ):
            manager, residents, state, tasks, _ = self._manager(
                "distributed", worker_count=1
            )
            manager.begin_weight_update_communicator = AsyncMock(
                return_value=[BeginWeightUpdateReqOutput(success=True, message="ok")]
            )
            engine = Engine.__new__(Engine)
            engine.loop = asyncio.new_event_loop()
            engine.tokenizer_manager = manager
            try:
                self.assertTrue(engine.begin_weight_update(weight_version="v2")[0])
                self.assertEqual(manager.config_value("weight_version"), "v1")
                self.assertTrue(
                    engine.update_weights_from_distributed(
                        names=[], dtypes=[], shapes=[], weight_version="v2"
                    )[0]
                )
                self.assertEqual(manager.config_value("weight_version"), "v2")
                self.assertTrue(all(s.weights == s.version == "v2" for s in residents))
                self.assertEqual(state.commits, 0)
            finally:

                async def finish():
                    await asyncio.gather(*tasks)

                engine.loop.run_until_complete(finish())
                engine.loop.close()


if __name__ == "__main__":
    unittest.main()
