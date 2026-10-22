import asyncio
import json
import sys
import time
import unittest
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPState,
    ElasticEPStateManager,
    RecoveryLifecycle,
    RecoveryOperation,
    get_recovery_operation,
    get_recovery_operation_for_slot,
    get_scale_cohort,
    register_scale_cohort,
    register_scale_operation,
)
from sglang.srt.entrypoints.elastic_ep import (
    observe_elastic_ep_membership as observe_elastic_ep_membership_endpoint,
)
from sglang.srt.entrypoints.elastic_ep import (
    recover_elastic_ep as recover_elastic_ep_endpoint,
)
from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    ElasticScaleUpdateReq,
    ObserveElasticEPMembershipProgressOutput,
    ObserveElasticEPMembershipReqInput,
    ObserveElasticEPMembershipReqOutput,
    RecoverElasticEPProgressOutput,
    RecoverElasticEPReqInput,
    RecoverElasticEPReqOutput,
    ScaleElasticEPReqInput,
    ScaleElasticEPReqOutput,
    sock_recv,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.managers.tokenizer_manager import (
    MembershipObservationTimeout,
    RecoverySubmissionTimeout,
    TokenizerManager,
)
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Store:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def check(self, keys):
        return all(key in self.values for key in keys)

    def get(self, key):
        return self.values[key]

    def compare_set(self, key, expected, value):
        existing = self.values.get(key, expected)
        if existing == expected:
            self.values[key] = value
            return value
        return existing


def _manager() -> TokenizerManager:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.elastic_worker_count = 4
    manager.attn_replica_size = 1
    manager.elastic_instance_id = "runtime-1"
    manager.elastic_initial_ep_size = 4
    manager.elastic_allocation_width = 4
    manager.elastic_max_committed_ep_size = 4
    manager.elastic_operation_id = None
    manager.elastic_operation_target = None
    manager.elastic_operation_succeeded = None
    manager.elastic_expected_joining_allocation_ids = []
    manager.elastic_pending_ep_size = None
    manager.elastic_scale_phase = "idle"
    manager.elastic_last_error = None
    manager.elastic_runtime_health = "healthy"
    manager.elastic_runtime_error = None
    manager.elastic_scheduler_response_timeout = 30
    manager.elastic_membership_observation_timeout = 60
    manager.elastic_membership_observation_reply_grace = 5
    manager.elastic_joining_rank_offset = None
    manager.elastic_joining_rank_count = 0
    manager.elastic_ready_rank_count = 0
    manager.elastic_joining_allocation_ids = []
    manager._elastic_scale_lock = asyncio.Lock()
    manager.elastic_recovery_operation_id = None
    manager.elastic_recovery_topology_generation = None
    manager.elastic_recovery_allocation_id = None
    manager.elastic_recovery_failed_allocation_incarnation = None
    manager.elastic_recovery_rank_offset = None
    manager.elastic_recovery_phase = "idle"
    manager.elastic_recovery_succeeded = None
    manager.elastic_recovery_error = None
    manager.elastic_active_rank_status = [True] * manager.elastic_worker_count
    manager._elastic_reported_rank_status = list(manager.elastic_active_rank_status)
    manager._elastic_membership_fences = {}
    manager._elastic_recovery_lock = asyncio.Lock()
    manager.auto_create_handle_loop = MagicMock()
    manager.scale_elastic_ep_communicator = AsyncMock(
        return_value=[
            ScaleElasticEPReqOutput(
                success=True,
                message="accepted",
                old_ep_size=4,
                new_ep_size=8,
                pending_ep_size=8,
                scale_phase="waiting_for_cohort",
            )
        ]
    )
    manager.update_control_communicator_fan_out = MagicMock()
    manager._dispatch_to_scheduler = MagicMock()
    manager.recover_elastic_ep_communicator = AsyncMock(
        return_value=[
            RecoverElasticEPReqOutput(
                success=True,
                message="accepted",
                operation_id="recover-1",
                recovery_phase="restoring",
            )
        ]
    )
    manager.observe_elastic_ep_membership_communicator = FanOutCommunicator(
        send=manager._dispatch_to_scheduler,
        fan_out=manager.elastic_worker_count,
    )
    return manager


class TestElasticEPLifecycle(CustomTestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            get_parallel().override(
                attn_tp_size=1,
                attn_cp_size=1,
            )
        )

    def test_scheduler_observation_update_mirrors_awaiting_recovery(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=MagicMock(),
        )
        state = ElasticEPState(
            active_ranks=None,
            last_active_ranks=None,
            active_ranks_cpu=None,
            effective_ep_size=2,
            recovery_phase="awaiting_recovery",
            runtime_instance_id="runtime-1",
            recovery_topology_generation=0,
            recovery_rank_offset=1,
            recovery_failed_allocation_incarnation="allocation-a",
        )
        with (
            patch(
                "sglang.srt.elastic_ep.elastic_ep.poll_recovery_control_plane",
                return_value=True,
            ),
            patch.object(ElasticEPStateManager, "instance", return_value=state),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ),
        ):
            scheduler._poll_elastic_ep_recovery()

        update = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertTrue(update.recovery_update)
        self.assertFalse(update.terminal)
        self.assertEqual(update.recovery_phase, "awaiting_recovery")
        self.assertEqual(update.recovery_topology_generation, 0)
        self.assertEqual(update.recovery_rank_offset, 1)

        manager = _manager()
        manager.forward_elastic_scale_update(update)
        self.assertEqual(manager.elastic_recovery_phase, "awaiting_recovery")
        self.assertEqual(manager.elastic_recovery_topology_generation, 0)
        self.assertEqual(manager.elastic_recovery_rank_offset, 1)
        self.assertEqual(
            manager.elastic_recovery_failed_allocation_incarnation,
            "allocation-a",
        )

    async def test_retry_returns_existing_operation_without_resubmitting(self):
        manager = _manager()
        request = ScaleElasticEPReqInput(
            new_ep_size=8,
            operation_id="grow-1",
            expected_instance_id="runtime-1",
            expected_joining_allocation_ids=["pod-uid-5"],
        )

        first = await manager.scale_elastic_ep(request)
        second = await manager.scale_elastic_ep(request)

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(second.operation_id, "grow-1")
        self.assertEqual(second.instance_id, "runtime-1")
        manager.scale_elastic_ep_communicator.assert_awaited_once()

    async def test_retry_reconciles_unknown_scheduler_submission(self):
        manager = _manager()
        manager.scale_elastic_ep_communicator.side_effect = [
            TimeoutError("response lost"),
            [
                ScaleElasticEPReqOutput(
                    success=True,
                    message="existing operation",
                    old_ep_size=4,
                    new_ep_size=8,
                    pending_ep_size=8,
                    scale_phase="waiting_for_cohort",
                )
            ],
        ]
        request = ScaleElasticEPReqInput(
            new_ep_size=8,
            operation_id="grow-1",
            expected_instance_id="runtime-1",
            expected_joining_allocation_ids=["pod-uid-5"],
        )

        with self.assertRaisesRegex(TimeoutError, "response lost"):
            await manager.scale_elastic_ep(request)
        self.assertEqual(manager.elastic_scale_phase, "submission_unknown")

        result = await manager.scale_elastic_ep(request)

        self.assertTrue(result.success)
        self.assertEqual(result.scale_phase, "waiting_for_cohort")
        self.assertEqual(manager.scale_elastic_ep_communicator.await_count, 2)

    async def test_submission_timeout_releases_lock_for_retry(self):
        manager = _manager()
        submission_ids = []

        async def submit(request):
            submission_ids.append(request.submission_id)
            if len(submission_ids) == 1:
                await asyncio.Event().wait()
            return [
                ScaleElasticEPReqOutput(
                    success=True,
                    message="existing operation",
                    old_ep_size=4,
                    new_ep_size=8,
                    pending_ep_size=8,
                    scale_phase="waiting_for_cohort",
                    submission_id=request.submission_id,
                )
            ]

        manager.scale_elastic_ep_communicator.side_effect = submit
        manager.elastic_scheduler_response_timeout = 0.01
        request = ScaleElasticEPReqInput(new_ep_size=8, operation_id="grow-1")

        with self.assertRaises(asyncio.TimeoutError):
            await manager.scale_elastic_ep(request)

        self.assertFalse(manager._elastic_scale_lock.locked())
        self.assertEqual(manager.elastic_scale_phase, "submission_unknown")

        result = await manager.scale_elastic_ep(request)

        self.assertTrue(result.success)
        self.assertFalse(result.terminal)
        self.assertEqual(result.scale_phase, "waiting_for_cohort")
        self.assertEqual(len(submission_ids), 2)
        self.assertNotEqual(submission_ids[0], submission_ids[1])

    async def test_terminal_update_wins_over_late_acceptance_response(self):
        manager = _manager()

        async def complete_then_accept(request):
            manager.forward_elastic_scale_update(
                ElasticScaleUpdateReq(
                    success=True,
                    effective_ep_size=8,
                    operation_id="grow-1",
                    scale_phase="serving_expanded",
                )
            )
            return [
                ScaleElasticEPReqOutput(
                    success=True,
                    message="accepted",
                    old_ep_size=4,
                    new_ep_size=8,
                    pending_ep_size=8,
                    scale_phase="waiting_for_cohort",
                    submission_id=request.submission_id,
                )
            ]

        manager.scale_elastic_ep_communicator.side_effect = complete_then_accept

        result = await manager.scale_elastic_ep(
            ScaleElasticEPReqInput(new_ep_size=8, operation_id="grow-1")
        )

        self.assertTrue(result.success)
        self.assertTrue(result.terminal)
        self.assertEqual(result.scale_phase, "serving_expanded")
        self.assertEqual(result.effective_ep_size, 8)
        self.assertTrue(manager.elastic_operation_succeeded)
        self.assertIsNone(manager.elastic_pending_ep_size)

    async def test_retry_observes_operation_completed_after_unknown_submission(self):
        manager = _manager()
        manager.scale_elastic_ep_communicator.side_effect = [
            TimeoutError("response lost"),
            [
                ScaleElasticEPReqOutput(
                    success=True,
                    message="existing operation completed",
                    old_ep_size=8,
                    new_ep_size=8,
                    pending_ep_size=None,
                    scale_phase="serving_expanded",
                    terminal=True,
                    effective_ep_size=8,
                )
            ],
        ]
        request = ScaleElasticEPReqInput(new_ep_size=8, operation_id="grow-1")

        with self.assertRaisesRegex(TimeoutError, "response lost"):
            await manager.scale_elastic_ep(request)
        result = await manager.scale_elastic_ep(request)

        self.assertTrue(result.success)
        self.assertTrue(result.terminal)
        self.assertEqual(result.effective_ep_size, 8)
        self.assertEqual(manager.elastic_worker_count, 8)
        self.assertTrue(manager.elastic_operation_succeeded)
        manager.update_control_communicator_fan_out.assert_called_once_with(8)

    async def test_same_operation_with_different_target_conflicts(self):
        manager = _manager()
        await manager.scale_elastic_ep(
            ScaleElasticEPReqInput(new_ep_size=8, operation_id="grow-1")
        )

        result = await manager.scale_elastic_ep(
            ScaleElasticEPReqInput(new_ep_size=9, operation_id="grow-1")
        )

        self.assertFalse(result.success)
        self.assertTrue(result.conflict)
        self.assertIn("already targets EP size 8", result.message)

    async def test_stale_runtime_instance_conflicts_before_submission(self):
        manager = _manager()

        result = await manager.scale_elastic_ep(
            ScaleElasticEPReqInput(
                new_ep_size=8,
                operation_id="grow-1",
                expected_instance_id="runtime-old",
            )
        )

        self.assertFalse(result.success)
        self.assertTrue(result.conflict)
        self.assertEqual(result.instance_id, "runtime-1")
        manager.scale_elastic_ep_communicator.assert_not_awaited()

    async def test_progress_exposes_joining_identity_and_readiness(self):
        manager = _manager()
        await manager.scale_elastic_ep(
            ScaleElasticEPReqInput(
                new_ep_size=8,
                operation_id="grow-1",
                expected_joining_allocation_ids=["pod-uid-5"],
            )
        )

        manager.forward_elastic_scale_update(
            ElasticScaleUpdateReq(
                success=True,
                terminal=False,
                effective_ep_size=4,
                operation_id="grow-1",
                scale_phase="cohort_ready",
                joining_rank_offset=4,
                joining_rank_count=4,
                ready_rank_count=4,
                joining_allocation_ids=["pod-uid-5"],
            )
        )

        status = manager.get_elastic_ep_state()
        self.assertEqual(status["instance_id"], "runtime-1")
        self.assertEqual(status["initial_ep_size"], 4)
        self.assertEqual(status["allocation_width"], 4)
        self.assertEqual(status["max_committed_ep_size"], 4)
        self.assertEqual(status["operation_id"], "grow-1")
        self.assertEqual(status["scale_phase"], "cohort_ready")
        self.assertEqual(status["joining_rank_offset"], 4)
        self.assertEqual(status["joining_rank_count"], 4)
        self.assertEqual(status["ready_rank_count"], 4)
        self.assertEqual(status["joining_allocation_ids"], ["pod-uid-5"])

    async def test_completed_operation_remains_retryable(self):
        manager = _manager()
        request = ScaleElasticEPReqInput(new_ep_size=8, operation_id="grow-1")
        await manager.scale_elastic_ep(request)
        manager.forward_elastic_scale_update(
            ElasticScaleUpdateReq(
                success=True,
                effective_ep_size=8,
                operation_id="grow-1",
                scale_phase="serving_expanded",
                joining_rank_offset=4,
                joining_rank_count=4,
                ready_rank_count=4,
            )
        )

        retry = await manager.scale_elastic_ep(request)

        self.assertTrue(retry.success)
        self.assertEqual(retry.pending_ep_size, None)
        status = manager.get_elastic_ep_state()
        self.assertTrue(status["operation_succeeded"])
        self.assertEqual(status["target_ep_size"], 8)
        self.assertEqual(status["max_committed_ep_size"], 8)
        manager.scale_elastic_ep_communicator.assert_awaited_once()

    async def test_recovery_failure_does_not_overwrite_completed_operation(self):
        manager = _manager()
        request = ScaleElasticEPReqInput(new_ep_size=8, operation_id="grow-1")
        await manager.scale_elastic_ep(request)
        manager.forward_elastic_scale_update(
            ElasticScaleUpdateReq(
                success=True,
                effective_ep_size=8,
                operation_id="grow-1",
                scale_phase="serving_expanded",
            )
        )

        manager.forward_elastic_scale_update(
            ElasticScaleUpdateReq(
                success=False,
                terminal=False,
                effective_ep_size=8,
                operation_update=False,
                runtime_health="recovery_unsupported",
                runtime_error="rank recovery requires a restart",
            )
        )

        retry = await manager.scale_elastic_ep(request)
        status = manager.get_elastic_ep_state()
        self.assertTrue(retry.success)
        self.assertTrue(retry.terminal)
        self.assertEqual(status["operation_id"], "grow-1")
        self.assertTrue(status["operation_succeeded"])
        self.assertEqual(status["scale_phase"], "serving_expanded")
        self.assertIsNone(status["last_error"])
        self.assertEqual(status["runtime_health"], "recovery_unsupported")
        self.assertEqual(status["runtime_error"], "rank recovery requires a restart")

    async def test_recovery_fanout_waits_only_for_active_workers(self):
        """A dead rank must not remain in the recovery response quorum."""
        manager = _manager()
        sent = []
        manager.recover_elastic_ep_communicator = FanOutCommunicator(
            send=sent.append,
            fan_out=4,
            mode="queueing",
            correlation_attr="submission_id",
        )
        ranks = ActiveRanksOutput(status=[True, True, False, True])
        manager.update_active_ranks(ranks)

        task = asyncio.create_task(
            manager.recover_elastic_ep(
                RecoverElasticEPReqInput(
                    operation_id="recover-1",
                    runtime_instance_id="ignored",
                    topology_generation=3,
                    allocation_id="pod-uid-5",
                    rank_offset=2,
                )
            )
        )
        await asyncio.sleep(0)
        submission_id = sent[0].submission_id
        for _ in range(3):
            manager.recover_elastic_ep_communicator.handle_recv(
                RecoverElasticEPReqOutput(
                    success=True,
                    message="accepted",
                    operation_id="recover-1",
                    submission_id=submission_id,
                    recovery_phase="restoring",
                )
            )

        result = await task
        self.assertTrue(result.success)
        manager._dispatch_to_scheduler.assert_called_once_with(ranks)

    async def test_recovery_traverses_real_zmq_control_route(self):
        manager = _manager()
        manager.elastic_worker_count = 2
        manager.elastic_active_rank_status = [True, True]

        context = zmq.Context()

        def make_channel():
            endpoint = f"inproc://recovery-{uuid.uuid4().hex}"
            receiver = context.socket(zmq.PULL)
            receiver.setsockopt(zmq.RCVTIMEO, 1000)
            receiver.bind(endpoint)
            sender = context.socket(zmq.PUSH)
            sender.connect(endpoint)
            return sender, receiver

        tokenizer_to_dpc, dpc_receiver = make_channel()
        dpc_to_scheduler, scheduler_receiver = make_channel()
        dpc_to_failed_scheduler, failed_scheduler_receiver = make_channel()
        scheduler_to_tokenizer, tokenizer_receiver = make_channel()

        manager.send_to_scheduler = tokenizer_to_dpc
        manager.tokenizer_ipc_name = None
        del manager._dispatch_to_scheduler

        controller = DataParallelController.__new__(DataParallelController)
        controller.workers = [dpc_to_scheduler, dpc_to_failed_scheduler]
        controller.status = [True, True]
        controller.dp_active = [True, True]
        controller.max_dp_size = 2
        controller._active_workers = [0, 1]
        controller.control_message_step = 1
        controller.init_dispatcher()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.flush_wrapper = MagicMock()
        scheduler.weight_updater = MagicMock()
        scheduler.profiler_manager = MagicMock()
        scheduler.rust_server = None
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SenderWrapper(scheduler_to_tokenizer)
        )
        scheduler.init_request_dispatcher()

        with get_parallel().override(num_dp_ranks=2):
            manager.init_request_dispatcher()
        manager.observe_elastic_ep_membership_communicator.set_fan_out(2)
        manager.recover_elastic_ep_communicator.set_fan_out(2)
        self.assertEqual(manager.elastic_active_rank_status, [True, True])
        self.assertEqual(manager.observe_elastic_ep_membership_communicator._fan_out, 2)

        # Report rank 1's failure while both ranks are still present. The
        # tokenizer quorum and DPC route must both exclude the failed rank.
        try:
            with (
                patch(
                    "sglang.srt.elastic_ep.elastic_ep.observe_recovery_membership",
                    return_value="observation-key",
                ),
                patch(
                    "sglang.srt.managers.scheduler.get_parallel",
                    return_value=SimpleNamespace(dp_rank=0),
                ),
            ):
                observation_task = asyncio.create_task(
                    manager.observe_elastic_ep_membership(
                        ObserveElasticEPMembershipReqInput(
                            runtime_instance_id="runtime-1",
                            topology_generation=3,
                            rank=1,
                            failed_allocation_incarnation="failed-pod-uid",
                            membership_epoch=4,
                        )
                    )
                )
                await asyncio.sleep(0)
                observation_request = sock_recv(dpc_receiver)
                self.assertIsInstance(
                    observation_request, ObserveElasticEPMembershipReqInput
                )
                self.assertEqual(
                    manager.observe_elastic_ep_membership_communicator._result_fan_out,
                    1,
                )
                controller._request_dispatcher(observation_request)
                rank0_observation = sock_recv(scheduler_receiver)
                self.assertEqual(rank0_observation.rank, 1)
                self.assertEqual(
                    rank0_observation.failed_allocation_incarnation,
                    "failed-pod-uid",
                )
                with self.assertRaises(zmq.Again):
                    sock_recv(failed_scheduler_receiver, flags=zmq.NOBLOCK)
                observation_response = scheduler._request_dispatcher(rank0_observation)
                scheduler.send_control_output(observation_response, rank0_observation)
                while True:
                    output = sock_recv(tokenizer_receiver)
                    manager._result_dispatcher(output)
                    if isinstance(output, ObserveElasticEPMembershipReqOutput):
                        break
                observation_result = await observation_task
                self.assertTrue(observation_result.accepted)
                self.assertEqual(observation_result.observation_key, "observation-key")
                status_before_recovery = manager.get_elastic_ep_state()
                self.assertEqual(
                    status_before_recovery["recovery_phase"], "awaiting_recovery"
                )
                self.assertIsNone(status_before_recovery["recovery_operation_id"])
                self.assertEqual(
                    status_before_recovery["recovery_topology_generation"], 3
                )
                self.assertEqual(status_before_recovery["recovery_rank_offset"], 1)
                self.assertEqual(
                    status_before_recovery["recovery_failed_allocation_incarnation"],
                    "failed-pod-uid",
                )
                self.assertIsNone(status_before_recovery["recovery_succeeded"])
                self.assertIsNone(status_before_recovery["recovery_error"])

            ranks = sock_recv(dpc_receiver)
            self.assertIsInstance(ranks, ActiveRanksOutput)
            self.assertEqual(ranks.status, [True, False])
            with patch(
                "sglang.srt.managers.data_parallel_controller.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(elastic_ep_backend="nixl")
                ),
            ):
                controller._request_dispatcher(ranks)
            self.assertEqual(manager.elastic_active_rank_status, [True, False])
            self.assertEqual(manager.recover_elastic_ep_communicator._fan_out, 1)
            self.assertEqual(controller._active_workers, [0])

            state = SimpleNamespace(recovery_phase="restoring", recovery_succeeded=None)
            with (
                patch.object(ElasticEPStateManager, "instance", return_value=state),
                patch.object(
                    ElasticEPStateManager,
                    "request_recovery",
                    return_value=True,
                ) as request_recovery,
                patch(
                    "sglang.srt.managers.scheduler.get_parallel",
                    return_value=SimpleNamespace(dp_rank=0),
                ),
            ):
                recovery_request = RecoverElasticEPReqInput(
                    operation_id="recover-1",
                    runtime_instance_id="ignored",
                    topology_generation=3,
                    allocation_id="pod-uid-5",
                    rank_offset=1,
                    failed_allocation_incarnation="failed-pod-uid",
                )
                task = asyncio.create_task(manager.recover_elastic_ep(recovery_request))
                await asyncio.sleep(0)

                dpc_request = sock_recv(dpc_receiver)
                self.assertIsInstance(dpc_request, RecoverElasticEPReqInput)
                controller._request_dispatcher(dpc_request)
                scheduler_request = sock_recv(scheduler_receiver)
                self.assertIsInstance(scheduler_request, RecoverElasticEPReqInput)
                with self.assertRaises(zmq.Again):
                    sock_recv(failed_scheduler_receiver, flags=zmq.NOBLOCK)
                scheduler_output = scheduler._request_dispatcher(scheduler_request)
                scheduler.send_control_output(scheduler_output, scheduler_request)

                received_stages = []
                while True:
                    tokenizer_output = sock_recv(tokenizer_receiver)
                    manager._result_dispatcher(tokenizer_output)
                    if isinstance(tokenizer_output, RecoverElasticEPProgressOutput):
                        received_stages.append(tokenizer_output.completed_stage)
                        continue
                    self.assertIsInstance(tokenizer_output, RecoverElasticEPReqOutput)
                    break

                result = await task
                retry = await manager.recover_elastic_ep(recovery_request)
                request_recovery.assert_called_once()
                with self.assertRaises(zmq.Again):
                    sock_recv(dpc_receiver, flags=zmq.NOBLOCK)

        finally:
            for socket in (
                tokenizer_to_dpc,
                dpc_receiver,
                dpc_to_scheduler,
                scheduler_receiver,
                dpc_to_failed_scheduler,
                failed_scheduler_receiver,
                scheduler_to_tokenizer,
                tokenizer_receiver,
            ):
                socket.close(linger=0)
            context.term()

        self.assertTrue(result.success)
        self.assertEqual(result.recovery_phase, "restoring")
        self.assertTrue(retry.success)
        self.assertEqual(retry.operation_id, result.operation_id)
        self.assertEqual(result.submission_id, scheduler_request.submission_id)
        self.assertEqual(
            received_stages,
            [
                "scheduler_handler_entered",
                "scheduler_store_started",
                "scheduler_store_completed",
            ],
        )
        self.assertIsNone(manager.recover_elastic_ep_communicator._result_event)

    async def test_membership_observation_uses_active_survivor_quorum(self):
        """The observation itself must exclude the newly failed rank."""
        manager = _manager()
        manager.elastic_worker_count = 2
        manager.elastic_active_rank_status = [True, True]
        sent = []
        manager.observe_elastic_ep_membership_communicator = FanOutCommunicator(
            send=sent.append,
            fan_out=2,
            correlation_attr="submission_id",
        )
        recovery_sent = []
        manager.recover_elastic_ep_communicator = FanOutCommunicator(
            send=recovery_sent.append,
            fan_out=4,
        )
        task = asyncio.create_task(
            manager.observe_elastic_ep_membership(
                ObserveElasticEPMembershipReqInput(
                    runtime_instance_id="runtime-1",
                    topology_generation=3,
                    rank=1,
                    failed_allocation_incarnation="pod-uid-5",
                    membership_epoch=4,
                )
            )
        )
        await asyncio.sleep(0)
        self.assertEqual(len(sent), 1)
        manager.observe_elastic_ep_membership_communicator.handle_recv(
            ObserveElasticEPMembershipReqOutput(
                accepted=True,
                message="accepted",
                observation_key="observation-key",
                submission_id=sent[0].submission_id,
                responder_rank=0,
            )
        )

        result = await task
        self.assertTrue(result.accepted)
        self.assertEqual(result.observation_key, "observation-key")
        self.assertEqual(manager.elastic_active_rank_status, [True, False])
        self.assertEqual(manager.recover_elastic_ep_communicator._fan_out, 1)
        manager._dispatch_to_scheduler.assert_called_once_with(
            ActiveRanksOutput(status=[True, False])
        )
        self.assertEqual(recovery_sent, [])

    def test_membership_fence_masks_stale_status_until_replacement_succeeds(self):
        manager = _manager()
        manager.elastic_recovery_operation_id = "recover-1"
        manager.elastic_recovery_topology_generation = 3
        manager.elastic_recovery_rank_offset = 1
        manager.elastic_recovery_failed_allocation_incarnation = "failed-pod-uid"
        manager._elastic_membership_fences[1] = (
            "runtime-1",
            3,
            "failed-pod-uid",
            4,
        )

        # A scheduler report may include the replacement before its terminal
        # recovery update. Keep the fixed slot fenced until that update proves
        # the matching recovery operation succeeded.
        manager.update_active_ranks(ActiveRanksOutput(status=[True, True, True, True]))
        self.assertEqual(manager.elastic_active_rank_status, [True, False, True, True])
        manager._dispatch_to_scheduler.assert_called_with(
            ActiveRanksOutput(status=[True, False, True, True])
        )

        manager.forward_elastic_scale_update(
            ElasticScaleUpdateReq(
                success=True,
                effective_ep_size=4,
                operation_id="recover-1",
                recovery_update=True,
                recovery_phase="ready",
                recovery_topology_generation=3,
                recovery_rank_offset=1,
                recovery_failed_allocation_incarnation="failed-pod-uid",
            )
        )

        self.assertNotIn(1, manager._elastic_membership_fences)
        self.assertEqual(manager.elastic_active_rank_status, [True, True, True, True])
        manager._dispatch_to_scheduler.assert_called_with(
            ActiveRanksOutput(status=[True, True, True, True])
        )

    async def test_membership_observation_rejects_without_active_ranks(self):
        manager = _manager()
        manager.recover_elastic_ep_communicator = FanOutCommunicator(
            send=MagicMock(),
            fan_out=4,
        )
        manager.update_active_ranks(
            ActiveRanksOutput(status=[False, False, False, False])
        )

        result = await manager.observe_elastic_ep_membership(
            ObserveElasticEPMembershipReqInput(
                runtime_instance_id="runtime-1",
                topology_generation=3,
                rank=2,
                failed_allocation_incarnation="pod-uid-5",
                membership_epoch=4,
            )
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            result.message,
            "Membership observation requires at least one survivor.",
        )
        manager._dispatch_to_scheduler.assert_called_once()

    async def test_membership_observation_rejects_out_of_range_rank(self):
        for rank in (-1, 4):
            with self.subTest(rank=rank):
                manager = _manager()

                result = await manager.observe_elastic_ep_membership(
                    ObserveElasticEPMembershipReqInput(
                        runtime_instance_id="runtime-1",
                        topology_generation=3,
                        rank=rank,
                        failed_allocation_incarnation="pod-uid-5",
                        membership_epoch=4,
                    )
                )

                self.assertFalse(result.accepted)
                self.assertEqual(
                    result.message,
                    "Membership observation rank is outside the active topology.",
                )
                manager._dispatch_to_scheduler.assert_not_called()

    async def test_membership_observation_requires_identical_survivor_keys(self):
        manager = _manager()
        manager.elastic_active_rank_status = [True, True, True, True]
        sent = []
        manager.observe_elastic_ep_membership_communicator = FanOutCommunicator(
            send=sent.append,
            fan_out=4,
            correlation_attr="submission_id",
        )
        task = asyncio.create_task(
            manager.observe_elastic_ep_membership(
                ObserveElasticEPMembershipReqInput(
                    runtime_instance_id="runtime-1",
                    topology_generation=3,
                    rank=2,
                    failed_allocation_incarnation="pod-uid-5",
                    membership_epoch=4,
                )
            )
        )
        await asyncio.sleep(0)
        for responder_rank, observation_key in ((0, "key"), (1, "key"), (3, "other")):
            manager.observe_elastic_ep_membership_communicator.handle_recv(
                ObserveElasticEPMembershipReqOutput(
                    accepted=True,
                    message="accepted",
                    observation_key=observation_key,
                    submission_id=sent[0].submission_id,
                    responder_rank=responder_rank,
                )
            )

        result = await task
        self.assertFalse(result.accepted)
        self.assertIn("same membership observation key", result.message)

    async def test_membership_observation_accepts_all_survivors(self):
        manager = _manager()
        sent = []
        manager.observe_elastic_ep_membership_communicator = FanOutCommunicator(
            send=sent.append,
            fan_out=4,
            correlation_attr="submission_id",
        )
        task = asyncio.create_task(
            manager.observe_elastic_ep_membership(
                ObserveElasticEPMembershipReqInput(
                    runtime_instance_id="runtime-1",
                    topology_generation=3,
                    rank=2,
                    failed_allocation_incarnation="pod-uid-5",
                    membership_epoch=4,
                )
            )
        )
        await asyncio.sleep(0)
        for responder_rank in (0, 1, 3):
            manager.observe_elastic_ep_membership_communicator.handle_recv(
                ObserveElasticEPMembershipReqOutput(
                    accepted=True,
                    message="accepted",
                    observation_key="observation-key",
                    submission_id=sent[0].submission_id,
                    responder_rank=responder_rank,
                )
            )

        result = await task
        self.assertTrue(result.accepted)
        self.assertEqual(result.observation_key, "observation-key")

    async def test_membership_observation_traverses_real_zmq_control_route(self):
        manager = _manager()
        manager.elastic_worker_count = 2
        manager.elastic_active_rank_status = [True, True]
        manager.elastic_scheduler_response_timeout = 0.01
        manager.elastic_membership_observation_timeout = 0.2
        manager.elastic_membership_observation_reply_grace = 0.05

        context = zmq.Context()

        def make_channel():
            endpoint = f"inproc://membership-{uuid.uuid4().hex}"
            receiver = context.socket(zmq.PULL)
            receiver.setsockopt(zmq.RCVTIMEO, 1000)
            receiver.bind(endpoint)
            sender = context.socket(zmq.PUSH)
            sender.connect(endpoint)
            return sender, receiver

        tokenizer_to_dpc, dpc_receiver = make_channel()
        dpc_to_scheduler, scheduler_receiver = make_channel()
        scheduler_to_tokenizer, tokenizer_receiver = make_channel()

        manager.send_to_scheduler = tokenizer_to_dpc
        manager.tokenizer_ipc_name = None
        del manager._dispatch_to_scheduler

        controller = DataParallelController.__new__(DataParallelController)
        controller.workers = [dpc_to_scheduler, None]
        controller.status = [True, True]
        controller._active_workers = [0, 1]
        controller.control_message_step = 1
        controller.init_dispatcher()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.flush_wrapper = MagicMock()
        scheduler.weight_updater = MagicMock()
        scheduler.profiler_manager = MagicMock()
        scheduler.rust_server = None
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SenderWrapper(scheduler_to_tokenizer)
        )
        scheduler.init_request_dispatcher()

        with get_parallel().override(num_dp_ranks=2):
            manager.init_request_dispatcher()

        try:
            with (
                patch(
                    "sglang.srt.elastic_ep.elastic_ep.observe_recovery_membership",
                    return_value="observation-key",
                ),
                patch(
                    "sglang.srt.managers.scheduler.get_parallel",
                    return_value=SimpleNamespace(dp_rank=0),
                ),
            ):
                task = asyncio.create_task(
                    manager.observe_elastic_ep_membership(
                        ObserveElasticEPMembershipReqInput(
                            runtime_instance_id="runtime-1",
                            topology_generation=3,
                            rank=1,
                            failed_allocation_incarnation="pod-uid-5",
                            membership_epoch=4,
                        )
                    )
                )
                await asyncio.sleep(0)

                dpc_request = sock_recv(dpc_receiver)
                self.assertIsInstance(dpc_request, ObserveElasticEPMembershipReqInput)
                controller._request_dispatcher(dpc_request)

                scheduler_request = sock_recv(scheduler_receiver)
                self.assertIsInstance(
                    scheduler_request, ObserveElasticEPMembershipReqInput
                )
                self.assertIsNotNone(scheduler_request.deadline_unix_seconds)
                await asyncio.sleep(0.02)
                scheduler_output = scheduler._request_dispatcher(scheduler_request)
                scheduler.send_control_output(scheduler_output, scheduler_request)

                received_stages = []
                while True:
                    tokenizer_output = sock_recv(tokenizer_receiver)
                    manager._result_dispatcher(tokenizer_output)
                    if isinstance(
                        tokenizer_output, ObserveElasticEPMembershipProgressOutput
                    ):
                        received_stages.append(tokenizer_output.completed_stage)
                        continue
                    self.assertIsInstance(
                        tokenizer_output, ObserveElasticEPMembershipReqOutput
                    )
                    break

                result = await task
        finally:
            for socket in (
                tokenizer_to_dpc,
                dpc_receiver,
                dpc_to_scheduler,
                scheduler_receiver,
                scheduler_to_tokenizer,
                tokenizer_receiver,
            ):
                socket.close(linger=0)
            context.term()

        self.assertTrue(result.accepted)
        self.assertEqual(result.observation_key, "observation-key")
        self.assertEqual(
            received_stages,
            [
                "scheduler_handler_entered",
                "scheduler_store_started",
                "scheduler_store_completed",
            ],
        )

    def test_expired_membership_observation_rejects_without_mutation(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.rust_server = None
        scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
        request = ObserveElasticEPMembershipReqInput(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            rank=1,
            failed_allocation_incarnation="pod-uid-5",
            membership_epoch=4,
            submission_id="submission-1",
            deadline_unix_seconds=time.time() - 1,
        )

        with (
            patch(
                "sglang.srt.elastic_ep.elastic_ep.observe_recovery_membership"
            ) as observe,
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(dp_rank=0),
            ),
        ):
            result = scheduler.handle_observe_elastic_ep_membership(request)

        self.assertFalse(result.accepted)
        self.assertIn("no state was changed", result.message)
        observe.assert_not_called()
        progress = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[
            0
        ]
        self.assertEqual(progress.completed_stage, "scheduler_deadline_rejected")

    def test_membership_observation_expiring_under_store_lock_is_rejected(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.rust_server = None
        scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
        request = ObserveElasticEPMembershipReqInput(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            rank=1,
            failed_allocation_incarnation="pod-uid-5",
            membership_epoch=4,
            submission_id="submission-1",
            deadline_unix_seconds=time.time() + 30,
        )

        with (
            patch(
                "sglang.srt.elastic_ep.elastic_ep.observe_recovery_membership",
                side_effect=TimeoutError(
                    "Membership observation expired before publication; "
                    "no state was changed."
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(dp_rank=0),
            ),
        ):
            result = scheduler.handle_observe_elastic_ep_membership(request)

        self.assertFalse(result.accepted)
        self.assertIn("no state was changed", result.message)
        progress_calls = (
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_args_list
        )
        self.assertEqual(
            progress_calls[-1].args[0].completed_stage,
            "scheduler_deadline_rejected",
        )

    async def test_membership_timeout_reports_cross_path_completed_stage(self):
        manager = _manager()
        manager.elastic_active_rank_status = [True, True]
        manager.elastic_membership_observation_timeout = 0.01
        manager.elastic_membership_observation_reply_grace = 0.001
        sent = []
        manager.observe_elastic_ep_membership_communicator = FanOutCommunicator(
            send=sent.append,
            fan_out=2,
            correlation_attr="submission_id",
        )

        task = asyncio.create_task(
            manager.observe_elastic_ep_membership(
                ObserveElasticEPMembershipReqInput(
                    runtime_instance_id="runtime-1",
                    topology_generation=3,
                    rank=1,
                    failed_allocation_incarnation="pod-uid-5",
                    membership_epoch=4,
                )
            )
        )
        await asyncio.sleep(0)
        manager._record_membership_observation_progress(
            ObserveElasticEPMembershipProgressOutput(
                submission_id=sent[0].submission_id,
                responder_rank=0,
                completed_stage="scheduler_store_started",
            )
        )
        with self.assertRaises(MembershipObservationTimeout) as raised:
            await task
        self.assertEqual(raised.exception.expected_ranks, [0])
        self.assertEqual(raised.exception.received_ranks, [])
        self.assertEqual(raised.exception.missing_ranks, [0])
        self.assertEqual(
            raised.exception.last_completed_stage, "scheduler_store_started"
        )
        self.assertGreater(raised.exception.elapsed_seconds, 0)
        self.assertIsNone(
            manager.observe_elastic_ep_membership_communicator._result_event
        )

    async def test_membership_timeout_http_response_preserves_diagnostics(self):
        request = SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "runtime_instance_id": "runtime-1",
                    "topology_generation": 3,
                    "rank": 1,
                    "failed_allocation_incarnation": "pod-uid-5",
                    "membership_epoch": 4,
                }
            )
        )
        tokenizer_manager = SimpleNamespace(
            observe_elastic_ep_membership=AsyncMock(
                side_effect=MembershipObservationTimeout(
                    submission_id="submission-1",
                    expected_ranks=[0],
                    received_ranks=[],
                    last_completed_stage="scheduler_store_started",
                    elapsed_seconds=30.001,
                )
            )
        )

        with (
            patch(
                "sglang.srt.entrypoints.elastic_ep.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(elastic_ep_backend="nixl")
                ),
            ),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.entrypoints.http_server": SimpleNamespace(
                        _global_state=SimpleNamespace(
                            tokenizer_manager=tokenizer_manager
                        )
                    )
                },
            ),
        ):
            response = await observe_elastic_ep_membership_endpoint(request)

        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            json.loads(response.body),
            {
                "error": "Timed out waiting for the membership survivor quorum.",
                "submission_id": "submission-1",
                "expected_ranks": [0],
                "received_ranks": [],
                "missing_ranks": [0],
                "last_completed_stage": "scheduler_store_started",
                "elapsed_seconds": 30.001,
            },
        )

    async def test_unknown_recovery_submission_is_reconciled(self):
        """A timed-out admission must be resubmitted with the same identity."""
        manager = _manager()
        manager.elastic_active_rank_status = [True, False, False, False]
        manager.elastic_scheduler_response_timeout = 0.01
        sent = []
        manager.recover_elastic_ep_communicator = FanOutCommunicator(
            send=sent.append,
            fan_out=1,
            correlation_attr="submission_id",
        )
        request = RecoverElasticEPReqInput(
            operation_id="recover-1",
            runtime_instance_id="ignored",
            topology_generation=3,
            allocation_id="pod-uid-5",
            rank_offset=2,
        )

        with self.assertRaises(RecoverySubmissionTimeout):
            await manager.recover_elastic_ep(request)
        first_submission_id = sent[0].submission_id

        manager.elastic_scheduler_response_timeout = 1
        retry = asyncio.create_task(manager.recover_elastic_ep(request))
        await asyncio.sleep(0)
        second_submission_id = sent[1].submission_id
        manager.recover_elastic_ep_communicator.handle_recv(
            RecoverElasticEPReqOutput(
                success=True,
                message="late attempt 1",
                submission_id=first_submission_id,
                responder_rank=0,
                recovery_phase="restoring",
            )
        )
        await asyncio.sleep(0)
        self.assertFalse(retry.done())
        manager.recover_elastic_ep_communicator.handle_recv(
            RecoverElasticEPReqOutput(
                success=True,
                message="attempt 2 accepted",
                submission_id=second_submission_id,
                responder_rank=0,
                recovery_phase="restoring",
            )
        )
        result = await retry

        self.assertTrue(result.success)
        self.assertEqual(result.recovery_phase, "restoring")
        self.assertNotEqual(first_submission_id, second_submission_id)

    async def test_recovery_timeout_reports_cross_path_completed_stage(self):
        manager = _manager()
        manager.elastic_active_rank_status = [True, False, False, False]
        manager.elastic_scheduler_response_timeout = 0.01
        sent = []
        manager.recover_elastic_ep_communicator = FanOutCommunicator(
            send=sent.append,
            fan_out=1,
            correlation_attr="submission_id",
        )

        task = asyncio.create_task(
            manager.recover_elastic_ep(
                RecoverElasticEPReqInput(
                    operation_id="recover-1",
                    runtime_instance_id="ignored",
                    topology_generation=3,
                    allocation_id="pod-uid-5",
                    rank_offset=1,
                    failed_allocation_incarnation="failed-pod-uid",
                )
            )
        )
        await asyncio.sleep(0)
        manager._record_recovery_submission_progress(
            RecoverElasticEPProgressOutput(
                operation_id="recover-1",
                submission_id=sent[0].submission_id,
                responder_rank=0,
                completed_stage="scheduler_store_started",
            )
        )

        with self.assertRaises(RecoverySubmissionTimeout) as raised:
            await task

        self.assertEqual(raised.exception.operation_id, "recover-1")
        self.assertEqual(raised.exception.submission_id, sent[0].submission_id)
        self.assertEqual(
            raised.exception.last_completed_stage, "scheduler_store_started"
        )
        self.assertEqual(raised.exception.recovery_phase, "submission_unknown")
        self.assertGreater(raised.exception.elapsed_seconds, 0)

    async def test_recovery_timeout_http_response_preserves_diagnostics(self):
        request = SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "operation_id": "recover-1",
                    "topology_generation": 3,
                    "allocation_id": "pod-uid-5",
                    "rank_offset": 1,
                    "failed_allocation_incarnation": "failed-pod-uid",
                }
            )
        )
        tokenizer_manager = SimpleNamespace(
            recover_elastic_ep=AsyncMock(
                side_effect=RecoverySubmissionTimeout(
                    operation_id="recover-1",
                    submission_id="submission-1",
                    last_completed_stage="scheduler_store_started",
                    elapsed_seconds=30.001,
                    recovery_phase="submission_unknown",
                )
            )
        )

        with (
            patch(
                "sglang.srt.entrypoints.elastic_ep.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(elastic_ep_backend="nixl")
                ),
            ),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.entrypoints.http_server": SimpleNamespace(
                        _global_state=SimpleNamespace(
                            tokenizer_manager=tokenizer_manager
                        )
                    )
                },
            ),
        ):
            response = await recover_elastic_ep_endpoint(request)

        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            json.loads(response.body),
            {
                "error": "Timed out waiting for the recovery scheduler quorum.",
                "operation_id": "recover-1",
                "submission_id": "submission-1",
                "elapsed_seconds": 30.001,
                "last_completed_stage": "scheduler_store_started",
                "recovery_phase": "submission_unknown",
            },
        )

    async def test_terminal_recovery_update_wins_over_late_acceptance(self):
        """A terminal update must not regress to the earlier admission phase."""
        manager = _manager()

        async def complete_then_accept(request, **kwargs):
            manager.forward_elastic_scale_update(
                ElasticScaleUpdateReq(
                    success=True,
                    terminal=True,
                    effective_ep_size=4,
                    operation_id=request.operation_id,
                    operation_update=False,
                    recovery_update=True,
                    recovery_phase="ready",
                )
            )
            return [
                RecoverElasticEPReqOutput(
                    success=True,
                    message="accepted",
                    submission_id=request.submission_id,
                    recovery_phase="restoring",
                )
            ]

        manager.recover_elastic_ep_communicator.side_effect = complete_then_accept
        result = await manager.recover_elastic_ep(
            RecoverElasticEPReqInput(
                operation_id="recover-1",
                runtime_instance_id="ignored",
                topology_generation=3,
                allocation_id="pod-uid-5",
                rank_offset=2,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.terminal)
        self.assertEqual(result.recovery_phase, "ready")


class TestElasticEPCohortBinding(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            get_parallel().override(
                attn_tp_size=1,
                attn_cp_size=1,
            )
        )

    def test_cohort_inherits_operation_and_runtime_identity(self):
        store = _Store()
        with patch(
            "sglang.srt.elastic_ep.elastic_ep.get_global_tcp_store",
            return_value=store,
        ):
            register_scale_operation(
                4,
                8,
                "runtime-1",
                "grow-1",
                ["pod-uid-5"],
            )
            cohort = register_scale_cohort(4, 8, 1, "pod-uid-5")
            stored_cohort = get_scale_cohort(4, "runtime-1", "grow-1")

        self.assertEqual(cohort.runtime_instance_id, "runtime-1")
        self.assertEqual(cohort.operation_id, "grow-1")
        self.assertEqual(cohort.rank_offset, 4)
        self.assertEqual(cohort.ready_rank_count, 4)
        self.assertEqual(cohort.allocation_id, "pod-uid-5")
        self.assertEqual(stored_cohort, cohort)

    def test_stale_cohort_does_not_poison_next_operation(self):
        store = _Store()
        with patch(
            "sglang.srt.elastic_ep.elastic_ep.get_global_tcp_store",
            return_value=store,
        ):
            register_scale_operation(4, 8, "runtime-1", "grow-1")
            stale_cohort = register_scale_cohort(4, 8, 1, "old-pod")

            register_scale_operation(4, 8, "runtime-1", "grow-2")
            self.assertIsNone(get_scale_cohort(4, "runtime-1", "grow-2"))
            current_cohort = register_scale_cohort(4, 8, 1, "new-pod")

            self.assertEqual(get_scale_cohort(4, "runtime-1", "grow-1"), stale_cohort)
            self.assertEqual(get_scale_cohort(4, "runtime-1", "grow-2"), current_cohort)

    def test_cohort_is_scoped_to_runtime_instance(self):
        store = _Store()
        with patch(
            "sglang.srt.elastic_ep.elastic_ep.get_global_tcp_store",
            return_value=store,
        ):
            register_scale_operation(4, 8, "runtime-1", "grow-1")
            old_runtime_cohort = register_scale_cohort(4, 8, 1, "old-pod")

            register_scale_operation(4, 8, "runtime-2", "grow-1")
            self.assertIsNone(get_scale_cohort(4, "runtime-2", "grow-1"))
            new_runtime_cohort = register_scale_cohort(4, 8, 1, "new-pod")

            self.assertEqual(
                get_scale_cohort(4, "runtime-1", "grow-1"), old_runtime_cohort
            )
            self.assertEqual(
                get_scale_cohort(4, "runtime-2", "grow-1"), new_runtime_cohort
            )

    def test_unexpected_allocation_is_rejected(self):
        store = _Store()
        with patch(
            "sglang.srt.elastic_ep.elastic_ep.get_global_tcp_store",
            return_value=store,
        ):
            register_scale_operation(
                4,
                8,
                "runtime-1",
                "grow-1",
                ["pod-uid-5"],
            )
            with self.assertRaisesRegex(RuntimeError, "is not authorized"):
                register_scale_cohort(4, 8, 1, "stale-pod-uid")


class TestElasticEPRecoveryLifecycle(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            get_parallel().override(
                attn_tp_size=1,
                attn_cp_size=1,
                dp_rank=0,
            )
        )
        self.store = _Store()
        self.store_patch = patch(
            "sglang.srt.elastic_ep.elastic_ep.get_global_tcp_store",
            return_value=self.store,
        )
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)
        self.topology_patch = patch(
            "sglang.srt.elastic_ep.elastic_ep.get_runtime_topology",
            return_value=MagicMock(
                runtime_instance_id="runtime-1",
                topology_generation=3,
                allocation_width=1,
                effective_ep_size=4,
            ),
        )
        self.topology_patch.start()
        self.addCleanup(self.topology_patch.stop)
        self.state = ElasticEPState(
            active_ranks=MagicMock(),
            last_active_ranks=None,
            active_ranks_cpu=MagicMock(),
            effective_ep_size=4,
            runtime_instance_id="runtime-1",
        )
        self.state.active_ranks_cpu.__len__.return_value = 4
        self.state.active_ranks_cpu.__getitem__.return_value.item.return_value = 0
        self.instance_patch = patch.object(
            ElasticEPStateManager, "_instance", self.state
        )
        self.instance_patch.start()
        self.addCleanup(self.instance_patch.stop)
        self.exec_patch = patch(
            "sglang.srt.managers.scheduler.get_exec",
            return_value=SimpleNamespace(moe=SimpleNamespace(moe_a2a_backend="deepep")),
        )
        self.exec_patch.start()
        self.addCleanup(self.exec_patch.stop)

    def test_recovery_record_is_idempotent_and_fenced(self):
        operation = RecoveryOperation(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            operation_id="recover-1",
            allocation_id="pod-uid-5",
            rank_offset=2,
        )

        self.assertTrue(ElasticEPStateManager.request_recovery(operation))
        self.assertTrue(ElasticEPStateManager.request_recovery(operation))
        self.assertEqual(self.state.recovery_phase, "restoring")
        self.assertEqual(get_recovery_operation("runtime-1", "recover-1"), operation)
        self.assertEqual(
            get_recovery_operation_for_slot("runtime-1", 3, 2, "pod-uid-5"),
            operation,
        )

        conflicting = RecoveryOperation(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            operation_id="recover-1",
            allocation_id="replacement-pod",
            rank_offset=2,
        )
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            from sglang.srt.elastic_ep.elastic_ep import register_recovery_operation

            register_recovery_operation(conflicting)

    def test_recovery_rejects_stale_generation_and_active_slot(self):
        stale = RecoveryOperation(
            runtime_instance_id="runtime-1",
            topology_generation=2,
            operation_id="recover-1",
            allocation_id="pod-uid-5",
            rank_offset=2,
        )
        with self.assertRaisesRegex(RuntimeError, "generation"):
            ElasticEPStateManager.request_recovery(stale)

        self.state.active_ranks_cpu.__getitem__.return_value.item.return_value = 1
        active = RecoveryOperation(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            operation_id="recover-1",
            allocation_id="pod-uid-5",
            rank_offset=2,
        )
        with self.assertRaisesRegex(RuntimeError, "inactive"):
            ElasticEPStateManager.request_recovery(active)

    def test_recovery_rejects_out_of_bounds_rank_before_indexing(self):
        operation = RecoveryOperation(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            operation_id="recover-oob",
            allocation_id="pod-uid-5",
            rank_offset=99,
        )

        with self.assertRaisesRegex(RuntimeError, "rank offset"):
            ElasticEPStateManager.request_recovery(operation)
        self.state.active_ranks_cpu.__getitem__.assert_not_called()

    def test_recovery_rejects_wrong_runtime_primary_slot_and_non_tp1_width(self):
        invalid_operations = [
            RecoveryOperation(
                runtime_instance_id="runtime-old",
                topology_generation=3,
                operation_id="recover-runtime",
                allocation_id="pod-uid-5",
                rank_offset=2,
            ),
            RecoveryOperation(
                runtime_instance_id="runtime-1",
                topology_generation=3,
                operation_id="recover-primary",
                allocation_id="pod-uid-5",
                rank_offset=0,
            ),
            RecoveryOperation(
                runtime_instance_id="runtime-1",
                topology_generation=3,
                operation_id="recover-allocation",
                allocation_id="",
                rank_offset=2,
            ),
        ]
        for operation in invalid_operations:
            with self.subTest(operation_id=operation.operation_id):
                with self.assertRaisesRegex(
                    (RuntimeError, ValueError), "runtime|non-primary|allocation"
                ):
                    ElasticEPStateManager.request_recovery(operation)

        self.topology_patch.stop()
        self.topology_patch = patch(
            "sglang.srt.elastic_ep.elastic_ep.get_runtime_topology",
            return_value=MagicMock(
                runtime_instance_id="runtime-1",
                topology_generation=3,
                allocation_width=2,
                effective_ep_size=4,
            ),
        )
        self.topology_patch.start()
        self.addCleanup(self.topology_patch.stop)
        with self.assertRaisesRegex(RuntimeError, "TP1"):
            ElasticEPStateManager.request_recovery(
                RecoveryOperation(
                    runtime_instance_id="runtime-1",
                    topology_generation=3,
                    operation_id="recover-width",
                    allocation_id="pod-uid-5",
                    rank_offset=2,
                )
            )

    def test_ready_requires_explicit_successful_internal_warmup(self):
        lifecycle = RecoveryLifecycle()

        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            lifecycle.advance("ready", warmup_succeeded=True)
        lifecycle.advance("slot_restored")
        lifecycle.advance("warming_up")
        with self.assertRaisesRegex(RuntimeError, "warmup"):
            lifecycle.advance("ready", warmup_succeeded=False)
        lifecycle.advance("ready", warmup_succeeded=True)

        self.assertEqual(lifecycle.phase, "ready")

    def test_recovery_state_requires_the_matching_warmup_operation(self):
        operation = RecoveryOperation(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            operation_id="recover-1",
            allocation_id="pod-uid-5",
            rank_offset=2,
        )
        ElasticEPStateManager.request_recovery(operation)
        ElasticEPStateManager.mark_recovery_slot_restored()
        self.assertTrue(ElasticEPStateManager.begin_recovery_warmup("recover-1"))
        self.assertFalse(
            ElasticEPStateManager.complete_recovery_warmup(
                "stale-recovery", success=True
            )
        )
        self.assertEqual(self.state.recovery_phase, "warming_up")
        self.assertTrue(
            ElasticEPStateManager.complete_recovery_warmup("recover-1", success=False)
        )
        self.assertEqual(self.state.recovery_phase, "failed")
        self.assertFalse(self.state.recovery_succeeded)
        self.assertEqual(self.state.runtime_health, "recovery_unsupported")
        self.state.recovery_phase = "slot_restored"
        self.state.recovery_succeeded = None
        self.assertTrue(ElasticEPStateManager.begin_recovery_warmup("recover-1"))
        self.assertTrue(
            ElasticEPStateManager.complete_recovery_warmup("recover-1", success=True)
        )
        self.assertEqual(self.state.recovery_phase, "ready")

    def test_scheduler_warmup_completion_requires_a_successful_normal_request(self):
        scheduler = Scheduler.__new__(Scheduler)
        self.state.recovery_operation_id = "recover-1"
        self.state.recovery_topology_generation = 3
        self.state.recovery_rank_offset = 2
        self.state.recovery_phase = "warming_up"
        self.state.active_ranks_cpu.__getitem__.return_value.item.return_value = 1
        finished = MagicMock(
            rid="health-check-elastic-recovery-recover-1",
            finished=MagicMock(return_value=True),
            finished_reason=MagicMock(),
        )
        with (
            patch(
                "sglang.srt.managers.scheduler.HEALTH_CHECK_RID_PREFIX",
                "health-check-",
            ),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=MagicMock(tp_rank=1),
            ),
        ):
            scheduler._maybe_complete_recovery_warmup(MagicMock(reqs=[finished]))
        self.assertEqual(self.state.recovery_phase, "ready")

    def test_scheduler_rejects_warmup_when_membership_is_lost(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ipc_channels = MagicMock()
        self.state.recovery_operation_id = "recover-1"
        self.state.recovery_topology_generation = 3
        self.state.recovery_rank_offset = 2
        self.state.recovery_phase = "warming_up"
        finished = MagicMock(
            rid="health-check-elastic-recovery-recover-1",
            finished=MagicMock(return_value=True),
            finished_reason=MagicMock(),
        )
        with (
            patch(
                "sglang.srt.managers.scheduler.HEALTH_CHECK_RID_PREFIX",
                "health-check-",
            ),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=MagicMock(tp_rank=0),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(moe_a2a_backend="nixl")
                ),
            ),
            patch.object(NixlEPBuffer, "abort_recovery") as abort_recovery,
        ):
            scheduler._maybe_complete_recovery_warmup(MagicMock(reqs=[finished]))

        self.assertEqual(self.state.recovery_phase, "failed")
        self.assertEqual(self.state.runtime_health, "recovery_unsupported")
        abort_recovery.assert_called_once_with("recover-1", 2)
        update = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertFalse(update.success)
        self.assertEqual(update.recovery_phase, "failed")

    def test_tokenizerless_scheduler_uses_model_eos_for_warmup(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tokenizer = None
        scheduler.model_config = MagicMock(hf_eos_token_id={7}, vocab_size=32)
        scheduler.handle_generate_request = MagicMock()
        self.state.recovery_operation_id = "recover-1"
        self.state.recovery_phase = "slot_restored"

        scheduler._maybe_schedule_recovery_warmup()

        request = scheduler.handle_generate_request.call_args.args[0]
        self.assertEqual(list(request.input_ids), [7])
        self.assertTrue(request.sampling_params.is_normalized)
        self.assertEqual(request.sampling_params.stop_strs, [])
        self.assertEqual(request.sampling_params.stop_regex_strs, [])
        finish_probe = Req.__new__(Req)
        finish_probe.sampling_params = request.sampling_params
        self.assertFalse(finish_probe._check_str_based_finish())

    def test_scheduler_reports_failure_when_no_warmup_token_exists(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tokenizer = None
        scheduler.model_config = MagicMock(hf_eos_token_id=set())
        scheduler.ipc_channels = MagicMock()
        self.state.recovery_operation_id = "recover-1"
        self.state.recovery_phase = "slot_restored"

        with patch(
            "sglang.srt.managers.scheduler.get_parallel",
            return_value=MagicMock(tp_rank=0),
        ):
            scheduler._maybe_schedule_recovery_warmup()

        update = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertFalse(update.success)
        self.assertTrue(update.terminal)
        self.assertEqual(update.recovery_phase, "failed")

    def test_terminal_warmup_releases_local_nixl_recovery_fence(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ipc_channels = MagicMock()
        self.state.recovery_operation_id = "recover-1"
        self.state.recovery_phase = "warming_up"

        with (
            patch(
                "sglang.srt.managers.scheduler.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(moe_a2a_backend="nixl")
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(tp_rank=1),
            ),
            patch.object(NixlEPBuffer, "finish_recovery") as finish_recovery,
        ):
            scheduler._report_recovery_warmup_result("recover-1", success=True)

        finish_recovery.assert_called_once_with("recover-1")
        self.assertEqual(self.state.recovery_phase, "ready")
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_recovery_preserves_scale_result_and_releases_terminal_exclusion(self):
        self.state.operation_id = "grow-1"
        self.state.operation_succeeded = False
        self.state.last_error = "scale failed"
        first = RecoveryOperation(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            operation_id="recover-1",
            allocation_id="pod-uid-5",
            rank_offset=2,
        )
        self.assertTrue(ElasticEPStateManager.request_recovery(first))
        self.assertFalse(self.state.operation_succeeded)
        self.assertEqual(self.state.last_error, "scale failed")
        self.state.recovery_phase = "warming_up"
        self.assertTrue(
            ElasticEPStateManager.complete_recovery_warmup("recover-1", success=True)
        )
        self.assertFalse(self.state.operation_succeeded)
        self.assertEqual(self.state.last_error, "scale failed")

        second = RecoveryOperation(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            operation_id="recover-2",
            allocation_id="replacement-pod",
            rank_offset=2,
        )
        self.assertTrue(ElasticEPStateManager.request_recovery(second))
        self.assertEqual(self.state.recovery_operation_id, "recover-2")

    def test_terminal_recovery_allows_a_later_scale(self):
        self.state.recovery_phase = "ready"
        self.state.recovery_succeeded = True

        accepted = ElasticEPStateManager.request_scale(
            8,
            "runtime-1",
            "grow-after-recovery",
            ["pod-uid-6"],
        )

        self.assertTrue(accepted)
        self.assertEqual(self.state.operation_id, "grow-after-recovery")

    def test_conflicting_pending_recovery_is_terminal(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.rust_server = None
        scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
        first = RecoverElasticEPReqInput(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            operation_id="recover-1",
            allocation_id="pod-uid-5",
            rank_offset=2,
        )
        conflicting = RecoverElasticEPReqInput(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            operation_id="recover-2",
            allocation_id="replacement-pod",
            rank_offset=2,
        )

        self.assertTrue(scheduler.handle_recover_elastic_ep(first).success)
        result = scheduler.handle_recover_elastic_ep(conflicting)

        self.assertFalse(result.success)
        self.assertTrue(result.conflict)
        self.assertTrue(result.terminal)

    def test_scheduler_retries_same_recovery_without_conflict(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.rust_server = None
        scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
        request = RecoverElasticEPReqInput(
            runtime_instance_id="runtime-1",
            topology_generation=3,
            operation_id="recover-1",
            allocation_id="pod-uid-5",
            rank_offset=2,
        )

        first = scheduler.handle_recover_elastic_ep(request)
        second = scheduler.handle_recover_elastic_ep(request)

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(second.recovery_phase, "restoring")

        self.state.recovery_phase = "ready"
        self.state.recovery_succeeded = True
        terminal = scheduler.handle_recover_elastic_ep(request)
        self.assertTrue(terminal.success)
        self.assertTrue(terminal.terminal)
        self.assertEqual(terminal.recovery_phase, "ready")


class TestElasticEPSchedulerIdempotency(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            get_parallel().override(
                attn_tp_size=1,
                attn_cp_size=1,
            )
        )

    def test_same_operation_is_reconciled_while_pending(self):
        state = ElasticEPState(
            active_ranks=None,
            last_active_ranks=None,
            active_ranks_cpu=None,
            effective_ep_size=4,
            pending_ep_size=8,
            scale_phase="waiting_for_cohort",
            runtime_instance_id="runtime-1",
            operation_id="grow-1",
            operation_target_ep_size=8,
            operation_expected_joining_allocation_ids=["pod-uid-5"],
        )
        scheduler = Scheduler.__new__(Scheduler)
        request = ScaleElasticEPReqInput(
            new_ep_size=8,
            operation_id="grow-1",
            runtime_instance_id="runtime-1",
            expected_joining_allocation_ids=["pod-uid-5"],
            submission_id="submission-2",
        )

        with (
            patch.object(ElasticEPStateManager, "_instance", state),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=MagicMock(max_ep_size=16),
            ),
        ):
            result = scheduler.handle_scale_elastic_ep(request)

        self.assertTrue(result.success)
        self.assertFalse(result.terminal)
        self.assertEqual(result.pending_ep_size, 8)
        self.assertEqual(result.scale_phase, "waiting_for_cohort")
        self.assertEqual(result.submission_id, "submission-2")

    def test_same_operation_with_different_target_conflicts_in_scheduler(self):
        state = ElasticEPState(
            active_ranks=None,
            last_active_ranks=None,
            active_ranks_cpu=None,
            effective_ep_size=4,
            pending_ep_size=8,
            scale_phase="waiting_for_cohort",
            runtime_instance_id="runtime-1",
            operation_id="grow-1",
            operation_target_ep_size=8,
            operation_expected_joining_allocation_ids=[],
        )
        scheduler = Scheduler.__new__(Scheduler)
        request = ScaleElasticEPReqInput(
            new_ep_size=9,
            operation_id="grow-1",
            runtime_instance_id="runtime-1",
        )

        with (
            patch.object(ElasticEPStateManager, "_instance", state),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=MagicMock(max_ep_size=16),
            ),
        ):
            result = scheduler.handle_scale_elastic_ep(request)

        self.assertFalse(result.success)
        self.assertTrue(result.conflict)
        self.assertIn("already targets EP size 8", result.message)

    def test_multi_allocation_world_accepts_one_allocation_growth(self):
        state = ElasticEPState(
            active_ranks=None,
            last_active_ranks=None,
            active_ranks_cpu=None,
            effective_ep_size=8,
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_worker = MagicMock(model_runner=MagicMock(eplb_manager=None))
        request = ScaleElasticEPReqInput(
            new_ep_size=12,
            operation_id="grow-1",
            runtime_instance_id="runtime-1",
            expected_joining_allocation_ids=["pod-uid-2"],
        )

        with (
            patch.object(ElasticEPStateManager, "_instance", state),
            patch.object(
                ElasticEPStateManager,
                "request_scale",
                return_value=True,
            ) as request_scale,
            patch.object(
                ElasticEPStateManager,
                "get_pending_ep_size",
                return_value=12,
            ),
            patch.object(
                ElasticEPStateManager,
                "get_scale_phase",
                return_value="waiting_for_cohort",
            ),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=MagicMock(
                    max_world_size=16,
                    tp_size=8,
                    nnodes=2,
                    elastic_ep_allocation_width=4,
                ),
            ),
        ):
            result = scheduler.handle_scale_elastic_ep(request)

        self.assertTrue(result.success)
        self.assertEqual(result.pending_ep_size, 12)
        request_scale.assert_called_once_with(
            12,
            "runtime-1",
            "grow-1",
            ["pod-uid-2"],
        )

    def test_multi_allocation_world_rejects_two_allocation_growth(self):
        state = ElasticEPState(
            active_ranks=None,
            last_active_ranks=None,
            active_ranks_cpu=None,
            effective_ep_size=8,
        )
        scheduler = Scheduler.__new__(Scheduler)
        request = ScaleElasticEPReqInput(
            new_ep_size=16,
            operation_id="grow-1",
            runtime_instance_id="runtime-1",
            expected_joining_allocation_ids=["pod-uid-2"],
        )

        with (
            patch.object(ElasticEPStateManager, "_instance", state),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=MagicMock(
                    max_world_size=16,
                    tp_size=8,
                    nnodes=2,
                    elastic_ep_allocation_width=4,
                ),
            ),
        ):
            result = scheduler.handle_scale_elastic_ep(request)

        self.assertFalse(result.success)
        self.assertTrue(result.terminal)
        self.assertIn("exactly one joining allocation", result.message)
        self.assertIn("local rank width (4), got 8", result.message)
        self.assertIsNone(state.operation_id)

    def test_initial_participant_rejects_inconsistent_allocation_width(self):
        state = ElasticEPState(
            active_ranks=None,
            last_active_ranks=None,
            active_ranks_cpu=None,
            effective_ep_size=8,
            original_ep_size=8,
        )

        with (
            patch(
                "sglang.srt.elastic_ep.elastic_ep.get_parallel",
                return_value=MagicMock(
                    elastic_ep_runtime_instance_id=None,
                    elastic_ep_allocation_width=4,
                    tp_size=8,
                    nnodes=4,
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "inconsistent local allocation width"),
        ):
            ElasticEPStateManager._publish_initial_runtime_topology(state)

    def test_recovery_failure_preserves_completed_operation_result(self):
        state = ElasticEPState(
            active_ranks=None,
            last_active_ranks=None,
            active_ranks_cpu=None,
            effective_ep_size=4,
            pending_ep_size=8,
            scale_phase="syncing_new_world",
            runtime_instance_id="runtime-1",
            operation_id="grow-1",
            operation_target_ep_size=8,
            operation_expected_joining_allocation_ids=[],
        )
        scheduler = Scheduler.__new__(Scheduler)
        request = ScaleElasticEPReqInput(
            new_ep_size=8,
            operation_id="grow-1",
            runtime_instance_id="runtime-1",
        )

        with (
            patch.object(ElasticEPStateManager, "_instance", state),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=MagicMock(max_ep_size=16),
            ),
        ):
            ElasticEPStateManager.commit_scale()
            ElasticEPStateManager.fail_recovery("rank recovery requires a restart")
            result = scheduler.handle_scale_elastic_ep(request)

        self.assertTrue(result.success)
        self.assertTrue(result.terminal)
        self.assertEqual(result.scale_phase, "serving_expanded")
        self.assertEqual(state.runtime_health, "recovery_unsupported")
        self.assertEqual(state.runtime_error, "rank recovery requires a restart")
        self.assertTrue(state.operation_succeeded)
        self.assertIsNone(state.last_error)


class _RecoveryPrefixStore:
    def __init__(self, prefix, store):
        self.prefix = prefix
        self.store = store
        self.waited = None
        self.on_wait = None

    def set(self, key, value):
        self.store.set(f"{self.prefix}{key}", value)

    def wait(self, keys, timeout):
        self.waited = (keys, timeout)
        if self.on_wait is not None:
            self.on_wait()


class TestNixlEPRecoveryConnections(CustomTestCase):
    def setUp(self):
        self.buffer = MagicMock(group_size=4)
        self.state = SimpleNamespace(
            buffer=self.buffer,
            connected_ep_size=0,
            scale_to=0,
            dispatch_ep_size=0,
            recovery_operation_id=None,
            recovery_disconnected_rank=None,
            recovery_connections_ready=False,
        )
        self.operation = RecoveryOperation(
            runtime_instance_id="runtime/1",
            topology_generation=3,
            operation_id="recover/1",
            allocation_id="pod/5",
            rank_offset=2,
        )
        self.store = _Store()
        self.prefix_stores = []
        self.prefix_store_on_wait = None

        def make_prefix_store(prefix, store):
            scoped = _RecoveryPrefixStore(prefix, store)
            scoped.on_wait = self.prefix_store_on_wait
            self.prefix_stores.append(scoped)
            return scoped

        self.patches = [
            patch.object(NixlEPBuffer, "_state", return_value=self.state),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.nixl.get_global_tcp_store",
                return_value=self.store,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.nixl.dist.PrefixStore",
                side_effect=make_prefix_store,
            ),
            patch.object(
                ElasticEPStateManager, "get_effective_ep_size", return_value=4
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.nixl.get_parallel",
                return_value=SimpleNamespace(tp_rank=0, ep_join_rank_offset=0),
            ),
        ]
        for active_patch in self.patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)

    def test_survivor_disconnects_then_reconnects_under_operation_scope(self):
        NixlEPBuffer.prepare_recovery(self.operation)
        NixlEPBuffer.connect_recovery(self.operation, replacement=False)

        self.buffer.disconnect_ranks.assert_called_once_with([2])
        self.buffer.connect_ranks.assert_called_once_with([2])
        scoped = self.prefix_stores[0]
        self.assertEqual(
            scoped.prefix,
            "elastic_ep/nixl_recovery/runtime%2F1/3/recover%2F1/pod%2F5/",
        )
        self.assertEqual(
            scoped.waited[0],
            [
                "connected/0",
                "connected/1",
                "connected/2",
                "connected/3",
            ],
        )
        self.assertTrue(self.state.recovery_connections_ready)

    def test_replacement_initializes_only_its_fenced_self_connection(self):
        NixlEPBuffer.initialize_recovery_joiner(self.operation, global_rank=2)

        self.buffer.connect_ranks.assert_called_once_with([2])
        self.assertEqual(self.state.connected_ep_size, 0)
        self.assertEqual(self.state.dispatch_ep_size, 4)
        self.assertEqual(self.state.recovery_operation_id, "recover/1")
        self.assertFalse(self.state.recovery_connections_ready)

    def test_replacement_connects_all_ranks_without_reusing_stale_scope(self):
        stale = RecoveryOperation(
            runtime_instance_id="runtime/1",
            topology_generation=2,
            operation_id="recover/old",
            allocation_id="pod/old",
            rank_offset=2,
        )
        stale_store = NixlEPBuffer._recovery_store(stale)
        observed_dispatch_ep_sizes = []

        def observe_dispatch_gate():
            observed_dispatch_ep_sizes.append(self.state.dispatch_ep_size)

        self.prefix_store_on_wait = observe_dispatch_gate
        self.state.dispatch_ep_size = 4
        self.state.recovery_operation_id = self.operation.operation_id

        NixlEPBuffer.connect_recovery(self.operation, replacement=True)

        self.buffer.disconnect_ranks.assert_not_called()
        self.buffer.connect_ranks.assert_called_once_with([0, 1, 2, 3])
        self.assertNotEqual(stale_store.prefix, self.prefix_stores[-1].prefix)
        self.assertEqual(observed_dispatch_ep_sizes, [0])
        self.assertEqual(self.state.connected_ep_size, 4)
        self.assertEqual(self.state.dispatch_ep_size, 4)

    def test_completed_recovery_releases_nixl_operation_fence(self):
        self.state.recovery_operation_id = self.operation.operation_id
        self.state.recovery_disconnected_rank = self.operation.rank_offset
        self.state.recovery_connections_ready = True

        NixlEPBuffer.finish_recovery(self.operation.operation_id)

        self.assertIsNone(self.state.recovery_operation_id)
        self.assertIsNone(self.state.recovery_disconnected_rank)
        self.assertFalse(self.state.recovery_connections_ready)

        next_operation = replace(
            self.operation,
            operation_id="recover/2",
            allocation_id="pod/6",
        )
        NixlEPBuffer.prepare_recovery(next_operation)
        self.buffer.disconnect_ranks.assert_called_once_with([2])

    def test_aborted_recovery_releases_nixl_operation_fence(self):
        self.state.recovery_operation_id = self.operation.operation_id
        self.state.recovery_disconnected_rank = self.operation.rank_offset
        self.state.recovery_connections_ready = True

        NixlEPBuffer.abort_recovery(
            self.operation.operation_id, self.operation.rank_offset
        )

        self.assertIsNone(self.state.recovery_operation_id)
        self.assertIsNone(self.state.recovery_disconnected_rank)
        self.assertFalse(self.state.recovery_connections_ready)

        next_operation = replace(
            self.operation,
            operation_id="recover/2",
            allocation_id="pod/6",
        )
        NixlEPBuffer.prepare_recovery(next_operation)
        self.assertEqual(
            self.buffer.disconnect_ranks.call_args_list,
            [call([2]), call([2])],
        )


if __name__ == "__main__":
    unittest.main()
