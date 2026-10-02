"""Receiver request fields reach the model-runner weight updater unchanged.

A client names an allowlisted receiver factory through
``InitWeightsUpdateGroupReqInput.receiver`` with an opaque
``receiver_init_payload``, and passes an opaque per-round ``receiver_payload``
to ``update_weights_from_distributed``. Each hop between the public Engine
API and the model-runner ``WeightUpdater`` is pinned separately so a dropped
or copied field fails at the hop responsible, not downstream of it.
"""

import asyncio
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt.entrypoints.engine import Engine  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager,
    _WeightUpdateSession,
)
from sglang.srt.managers.tokenizer_control_mixin import (  # noqa: E402
    TokenizerControlMixin,
)
from sglang.srt.managers.tp_worker import BaseTpWorker  # noqa: E402
from sglang.srt.model_executor import model_runner as model_runner_module  # noqa: E402
from sglang.srt.model_executor.model_runner import ModelRunner  # noqa: E402

RECEIVER = "my_pkg.receivers.create_receiver"
INIT_PAYLOAD = {"manifest": {"epoch": 3, "names": ["w1", "w2"]}}
ROUND_PAYLOAD = {"operation_id": "op-17", "version": "42"}


def _run(coroutine):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


class TestWireRoundTrip(unittest.TestCase):
    """io_struct: the fields survive the actual ZMQ/msgpack IPC codec."""

    def _round_trip(self, request):
        decoded = msgpack_decode(msgpack_encode(request))
        self.assertEqual(pickle.loads(pickle.dumps(request)), decoded)
        return decoded

    def test_init_group_fields_round_trip(self):
        request = InitWeightsUpdateGroupReqInput(
            master_address="10.1.2.3",
            master_port=23456,
            rank_offset=0,
            world_size=4,
            group_name="ext",
            receiver=RECEIVER,
            receiver_init_payload=INIT_PAYLOAD,
        )
        decoded = self._round_trip(request)
        self.assertEqual(decoded.receiver, RECEIVER)
        self.assertEqual(decoded.receiver_init_payload, INIT_PAYLOAD)

    def test_update_fields_round_trip(self):
        request = UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            group_name="ext",
            receiver_payload=ROUND_PAYLOAD,
        )
        decoded = self._round_trip(request)
        self.assertEqual(decoded.receiver_payload, ROUND_PAYLOAD)

    def test_destroy_group_name_round_trip(self):
        decoded = self._round_trip(DestroyWeightsUpdateGroupReqInput(group_name="ext"))
        self.assertEqual(decoded.group_name, "ext")


class _RecordingAsync:
    """Async stand-in recording the exact request object it was handed."""

    def __init__(self, result):
        self.result = result
        self.received = None

    async def __call__(self, obj, *args):
        self.received = obj
        return self.result


class TestEngineEntrypoint(unittest.TestCase):
    """entrypoints/engine.py: keyword args land on the request object."""

    def _engine(self):
        engine = object.__new__(Engine)
        engine.loop = SimpleNamespace(run_until_complete=_run)
        engine.tokenizer_manager = SimpleNamespace(
            init_weights_update_group=_RecordingAsync((True, "ok")),
            update_weights_from_distributed=_RecordingAsync((True, "ok")),
            destroy_weights_update_group=_RecordingAsync((True, "ok")),
        )
        return engine

    def test_init_group_forwards_receiver_fields(self):
        engine = self._engine()
        self.assertEqual(
            engine.init_weights_update_group(
                "10.1.2.3",
                23456,
                0,
                4,
                "ext",
                receiver=RECEIVER,
                receiver_init_payload=INIT_PAYLOAD,
            ),
            (True, "ok"),
        )
        obj = engine.tokenizer_manager.init_weights_update_group.received
        self.assertEqual(obj.receiver, RECEIVER)
        self.assertIs(obj.receiver_init_payload, INIT_PAYLOAD)
        self.assertEqual(obj.group_name, "ext")

    def test_update_forwards_receiver_payload(self):
        engine = self._engine()
        engine.update_weights_from_distributed(
            names=[],
            dtypes=[],
            shapes=[],
            group_name="ext",
            receiver_payload=ROUND_PAYLOAD,
        )
        obj = engine.tokenizer_manager.update_weights_from_distributed.received
        self.assertIs(obj.receiver_payload, ROUND_PAYLOAD)
        self.assertEqual((obj.names, obj.dtypes, obj.shapes), ([], [], []))

    def test_destroy_forwards_group_name(self):
        engine = self._engine()
        engine.destroy_weights_update_group("ext")
        obj = engine.tokenizer_manager.destroy_weights_update_group.received
        self.assertEqual(obj.group_name, "ext")


class _AsyncContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class _ControlManager(TokenizerControlMixin):
    """Bare control-plane object exposing the mixin's weight-update methods."""

    def __init__(self):
        self.is_pause = False
        self.is_pause_cond = _AsyncContext()
        self.model_update_lock = SimpleNamespace(
            reader_lock=_AsyncContext(), writer_lock=_AsyncContext()
        )
        self._weight_update_staged_session = False
        self.mm_processor = None
        self.init_weights_update_group_communicator = _RecordingAsync(
            [InitWeightsUpdateGroupReqOutput(success=True, message="ok")]
        )
        self.update_weights_from_distributed_communicator = _RecordingAsync(
            [UpdateWeightsFromDistributedReqOutput(success=True, message="ok")]
        )
        self.destroy_weights_update_group_communicator = _RecordingAsync(
            [DestroyWeightsUpdateGroupReqOutput(success=True, message="ok")]
        )

    def auto_create_handle_loop(self):
        pass


@patch(
    "sglang.srt.managers.tokenizer_control_mixin.get_parallel",
    return_value=SimpleNamespace(dp_size=1, enable_dp_attention=False),
)
class TestTokenizerControlPlane(unittest.TestCase):
    """tokenizer manager: the request object reaches the fan-out unchanged."""

    def setUp(self):
        self.manager = _ControlManager()

    def test_init_group_passthrough(self, _get_parallel):
        request = InitWeightsUpdateGroupReqInput(
            master_address="10.1.2.3",
            master_port=23456,
            rank_offset=0,
            world_size=4,
            group_name="ext",
            receiver=RECEIVER,
            receiver_init_payload=INIT_PAYLOAD,
        )
        self.assertEqual(
            _run(self.manager.init_weights_update_group(request)), (True, "ok")
        )
        self.assertIs(
            self.manager.init_weights_update_group_communicator.received, request
        )

    def test_update_passthrough(self, _get_parallel):
        request = UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            group_name="ext",
            receiver_payload=ROUND_PAYLOAD,
        )
        self.assertEqual(
            _run(self.manager.update_weights_from_distributed(request)), (True, "ok")
        )
        self.assertIs(
            self.manager.update_weights_from_distributed_communicator.received, request
        )

    def test_destroy_passthrough(self, _get_parallel):
        request = DestroyWeightsUpdateGroupReqInput(group_name="ext")
        self.assertEqual(
            _run(self.manager.destroy_weights_update_group(request)), (True, "ok")
        )
        self.assertIs(
            self.manager.destroy_weights_update_group_communicator.received, request
        )


class TestSchedulerWeightUpdater(unittest.TestCase):
    """scheduler_components/weight_updater: fan-in to the worker forwards fields."""

    def _manager(self):
        self.model_weight_updater = Mock(name="model_runner.weight_updater")
        self.model_weight_updater.receive_weights_from_distributed.return_value = []
        tp_worker = SimpleNamespace(
            init_weights_update_group=Mock(return_value=(True, "ok")),
            destroy_weights_update_group=Mock(return_value=(True, "ok")),
            model_runner=SimpleNamespace(weight_updater=self.model_weight_updater),
        )
        return SchedulerWeightUpdaterManager(
            tp_worker=tp_worker,
            draft_worker=None,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=Mock(return_value=True),
            is_fully_idle=lambda: True,
        ), tp_worker

    def _init_receiver_group(self, manager, group_name="ext"):
        out = manager.init_weights_update_group(
            InitWeightsUpdateGroupReqInput(
                master_address="10.1.2.3",
                master_port=23456,
                rank_offset=0,
                world_size=4,
                group_name=group_name,
                receiver=RECEIVER,
                receiver_init_payload=INIT_PAYLOAD,
            )
        )
        self.assertTrue(out.success)

    def test_init_group_passthrough(self):
        manager, tp_worker = self._manager()
        request = InitWeightsUpdateGroupReqInput(
            master_address="10.1.2.3",
            master_port=23456,
            rank_offset=0,
            world_size=4,
            group_name="ext",
            receiver=RECEIVER,
            receiver_init_payload=INIT_PAYLOAD,
        )
        out = manager.init_weights_update_group(request)
        self.assertTrue(out.success)
        (forwarded,), _ = tp_worker.init_weights_update_group.call_args
        self.assertIs(forwarded, request)

    def test_destroy_group_passthrough(self):
        manager, tp_worker = self._manager()
        request = DestroyWeightsUpdateGroupReqInput(group_name="ext")
        out = manager.destroy_weights_update_group(request)
        self.assertTrue(out.success)
        (forwarded,), _ = tp_worker.destroy_weights_update_group.call_args
        self.assertIs(forwarded, request)

    def test_successful_receiver_init_records_the_group_name(self):
        manager, _ = self._manager()
        self._init_receiver_group(manager)
        self.assertEqual(manager._receiver_group_names, {"ext"})

    def test_torch_group_init_does_not_record_a_receiver_name(self):
        manager, _ = self._manager()
        out = manager.init_weights_update_group(
            InitWeightsUpdateGroupReqInput(
                master_address="10.1.2.3",
                master_port=23456,
                rank_offset=0,
                world_size=4,
                group_name="ext",
            )
        )
        self.assertTrue(out.success)
        self.assertEqual(manager._receiver_group_names, set())

    def test_failed_init_does_not_record_a_receiver_name(self):
        manager, tp_worker = self._manager()
        tp_worker.init_weights_update_group.return_value = (False, "no")
        out = manager.init_weights_update_group(
            InitWeightsUpdateGroupReqInput(
                master_address="10.1.2.3",
                master_port=23456,
                rank_offset=0,
                world_size=4,
                group_name="ext",
                receiver=RECEIVER,
            )
        )
        self.assertFalse(out.success)
        self.assertEqual(manager._receiver_group_names, set())

    def test_update_forwards_payload_to_model_runner_updater(self):
        manager, _ = self._manager()
        manager._session = _WeightUpdateSession(selector="all")
        request = UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            group_name="ext",
            receiver_payload=ROUND_PAYLOAD,
        )
        out = manager.update_weights_from_distributed(request)
        self.assertTrue(out.success)
        kwargs = (
            self.model_weight_updater.receive_weights_from_distributed.call_args.kwargs
        )
        self.assertEqual(kwargs["names"], [])
        self.assertEqual(kwargs["dtypes"], [])
        self.assertEqual(kwargs["shapes"], [])
        self.assertEqual(kwargs["group_name"], "ext")
        self.assertIs(kwargs["receiver_payload"], ROUND_PAYLOAD)
        # The receiver wrote into the live model; no load fan-out happened.
        manager.flush_cache.assert_called_once()

    def test_update_failure_surfaces_the_updaters_message(self):
        manager, _ = self._manager()
        manager._session = _WeightUpdateSession(selector="all")
        self.model_weight_updater.receive_weights_from_distributed.side_effect = (
            ValueError(
                "Group 'plain' has no external receiver to take receiver_payload"
            )
        )
        request = UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            group_name="plain",
            receiver_payload=ROUND_PAYLOAD,
        )
        out = manager.update_weights_from_distributed(request)
        self.assertFalse(out.success)
        self.assertIn("no external receiver", out.message)
        manager.flush_cache.assert_not_called()

    def test_receiver_round_is_rejected_while_a_draft_model_exists(self):
        manager, _ = self._manager()
        manager.draft_worker = Mock(name="draft_worker")
        self._init_receiver_group(manager)
        manager._session = _WeightUpdateSession(selector="all")
        request = UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            group_name="ext",
            receiver_payload=ROUND_PAYLOAD,
        )
        out = manager.update_weights_from_distributed(request)
        self.assertFalse(out.success)
        self.assertIn("draft model", out.message)
        self.model_weight_updater.receive_weights_from_distributed.assert_not_called()

    def test_none_payload_receiver_round_is_rejected_while_a_draft_model_exists(self):
        """receive(None) is a legal round, so the guard keys on the group, not the payload."""
        manager, _ = self._manager()
        manager.draft_worker = Mock(name="draft_worker")
        self._init_receiver_group(manager)
        manager._session = _WeightUpdateSession(selector="all")
        request = UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            group_name="ext",
            receiver_payload=None,
            weight_version="v2",
        )
        out = manager.update_weights_from_distributed(request)
        self.assertFalse(out.success)
        self.assertIn("draft model", out.message)
        self.model_weight_updater.receive_weights_from_distributed.assert_not_called()
        manager.flush_cache.assert_not_called()
        self.assertIsNone(manager._session.pending_version)

    def test_destroy_forgets_the_receiver_group_name(self):
        manager, tp_worker = self._manager()
        self._init_receiver_group(manager)
        request = DestroyWeightsUpdateGroupReqInput(group_name="ext")
        self.assertTrue(manager.destroy_weights_update_group(request).success)
        self.assertEqual(manager._receiver_group_names, set())

        # The worker forgets a receiver even when its destroy() raises; the
        # tracked name must go too, or the draft guard outlives the receiver.
        self._init_receiver_group(manager)
        tp_worker.destroy_weights_update_group.return_value = (False, "boom")
        out = manager.destroy_weights_update_group(request)
        self.assertFalse(out.success)
        self.assertEqual(manager._receiver_group_names, set())

    def test_update_outside_a_session_is_rejected(self):
        manager, _ = self._manager()
        request = UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            group_name="ext",
            receiver_payload=ROUND_PAYLOAD,
        )
        out = manager.update_weights_from_distributed(request)
        self.assertFalse(out.success)
        self.assertIn("begin_weight_update", out.message)
        self.model_weight_updater.receive_weights_from_distributed.assert_not_called()


class TestTpWorkerHop(unittest.TestCase):
    """tp_worker: the request fields reach the model-runner weight updater."""

    def _worker(self):
        updater = Mock(name="weight_updater")
        updater.init_weights_update_group.return_value = (True, "ok")
        updater.destroy_weights_update_group.return_value = (True, "ok")
        return SimpleNamespace(
            model_runner=SimpleNamespace(weight_updater=updater)
        ), updater

    def test_init_group_forwards_receiver_fields(self):
        worker, updater = self._worker()
        request = InitWeightsUpdateGroupReqInput(
            master_address="10.1.2.3",
            master_port=23456,
            rank_offset=0,
            world_size=4,
            group_name="ext",
            receiver=RECEIVER,
            receiver_init_payload=INIT_PAYLOAD,
        )
        self.assertEqual(
            BaseTpWorker.init_weights_update_group(worker, request), (True, "ok")
        )
        kwargs = updater.init_weights_update_group.call_args.kwargs
        self.assertEqual(
            kwargs,
            {
                "master_address": "10.1.2.3",
                "master_port": 23456,
                "rank_offset": 0,
                "world_size": 4,
                "group_name": "ext",
                "backend": "nccl",
                "receiver": RECEIVER,
                "receiver_init_payload": INIT_PAYLOAD,
            },
        )

    def test_destroy_forwards_group_name(self):
        worker, updater = self._worker()
        BaseTpWorker.destroy_weights_update_group(
            worker, DestroyWeightsUpdateGroupReqInput(group_name="ext")
        )
        updater.destroy_weights_update_group.assert_called_once_with("ext")


class TestModelRunnerWiring(unittest.TestCase):
    """model_runner: the server-args allowlist reaches the WeightUpdater."""

    def test_init_weight_updater_passes_the_allowlist(self):
        runner = SimpleNamespace(
            tp_rank=2,
            device="cpu",
            gpu_id=0,
            model_config=None,
            model=SimpleNamespace(),
            update_model_fields=lambda *args, **kwargs: None,
            init_decode_cuda_graph=lambda: None,
        )
        config_model = SimpleNamespace(
            custom_weight_loader={}, weight_update_receivers=[RECEIVER]
        )
        with (
            patch.object(model_runner_module, "WeightUpdater") as weight_updater,
            patch.object(model_runner_module, "get_model", return_value=config_model),
        ):
            ModelRunner.init_weight_updater(runner)
        kwargs = weight_updater.call_args.kwargs
        self.assertEqual(kwargs["weight_update_receivers"], [RECEIVER])
        self.assertEqual(kwargs["custom_weight_loaders"], {})
        self.assertIs(runner.weight_updater, weight_updater.return_value)


if __name__ == "__main__":
    unittest.main()
