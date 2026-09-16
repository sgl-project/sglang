import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPState,
    ElasticEPStateManager,
    get_scale_cohort,
    register_scale_cohort,
    register_scale_operation,
)
from sglang.srt.managers.io_struct import (
    ElasticScaleUpdateReq,
    ScaleElasticEPReqInput,
    ScaleElasticEPReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager

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


def _manager() -> TokenizerManager:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.elastic_worker_count = 4
    manager.elastic_instance_id = "runtime-1"
    manager.elastic_operation_id = None
    manager.elastic_operation_target = None
    manager.elastic_operation_succeeded = None
    manager.elastic_expected_joining_member_ids = []
    manager.elastic_pending_ep_size = None
    manager.elastic_scale_phase = "idle"
    manager.elastic_last_error = None
    manager.elastic_runtime_health = "healthy"
    manager.elastic_runtime_error = None
    manager.elastic_joining_rank_offset = None
    manager.elastic_joining_rank_count = 0
    manager.elastic_ready_rank_count = 0
    manager.elastic_joining_member_ids = []
    manager._elastic_scale_lock = asyncio.Lock()
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
    return manager


class TestElasticEPLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_retry_returns_existing_operation_without_resubmitting(self):
        manager = _manager()
        request = ScaleElasticEPReqInput(
            new_ep_size=8,
            operation_id="grow-1",
            expected_instance_id="runtime-1",
            expected_joining_member_ids=["pod-uid-5"],
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
            expected_joining_member_ids=["pod-uid-5"],
        )

        with self.assertRaisesRegex(TimeoutError, "response lost"):
            await manager.scale_elastic_ep(request)
        self.assertEqual(manager.elastic_scale_phase, "submission_unknown")

        result = await manager.scale_elastic_ep(request)

        self.assertTrue(result.success)
        self.assertEqual(result.scale_phase, "waiting_for_cohort")
        self.assertEqual(manager.scale_elastic_ep_communicator.await_count, 2)

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
                expected_joining_member_ids=["pod-uid-5"],
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
                joining_member_ids=["pod-uid-5"],
            )
        )

        status = manager.get_elastic_ep_state()
        self.assertEqual(status["instance_id"], "runtime-1")
        self.assertEqual(status["operation_id"], "grow-1")
        self.assertEqual(status["scale_phase"], "cohort_ready")
        self.assertEqual(status["joining_rank_offset"], 4)
        self.assertEqual(status["joining_rank_count"], 4)
        self.assertEqual(status["ready_rank_count"], 4)
        self.assertEqual(status["joining_member_ids"], ["pod-uid-5"])

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


class TestElasticEPCohortBinding(unittest.TestCase):
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
        self.assertEqual(cohort.member_id, "pod-uid-5")
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

    def test_unexpected_member_is_rejected(self):
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


class TestElasticEPSchedulerIdempotency(unittest.TestCase):
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
            operation_expected_joining_member_ids=["pod-uid-5"],
        )
        scheduler = Scheduler.__new__(Scheduler)
        request = ScaleElasticEPReqInput(
            new_ep_size=8,
            operation_id="grow-1",
            runtime_instance_id="runtime-1",
            expected_joining_member_ids=["pod-uid-5"],
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
            operation_expected_joining_member_ids=[],
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
            operation_expected_joining_member_ids=[],
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


if __name__ == "__main__":
    unittest.main()
