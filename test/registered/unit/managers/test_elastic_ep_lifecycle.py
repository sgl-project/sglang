import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.elastic_ep.elastic_ep import (
    register_scale_cohort,
    register_scale_operation,
)
from sglang.srt.managers.io_struct import (
    ElasticScaleUpdateReq,
    ScaleElasticEPReqInput,
    ScaleElasticEPReqOutput,
)
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
        self.assertTrue(manager.get_elastic_ep_state()["operation_succeeded"])
        manager.scale_elastic_ep_communicator.assert_awaited_once()


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

        self.assertEqual(cohort.runtime_instance_id, "runtime-1")
        self.assertEqual(cohort.operation_id, "grow-1")
        self.assertEqual(cohort.rank_offset, 4)
        self.assertEqual(cohort.ready_rank_count, 4)
        self.assertEqual(cohort.member_id, "pod-uid-5")

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


if __name__ == "__main__":
    unittest.main()
