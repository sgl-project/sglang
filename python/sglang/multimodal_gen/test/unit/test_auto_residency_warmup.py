import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.multimodal_gen.runtime import server_warmup
from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    AutoResidencyReq,
)
from sglang.multimodal_gen.runtime.managers import gpu_worker as gpu_worker_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
    GIB_BYTES,
    PLACEMENT_STATUS_ADJUSTED,
    PLACEMENT_STATUS_ROLLBACK_FAILED,
    PLACEMENT_STATUS_ROLLED_BACK,
    PLACEMENT_STATUS_SKIPPED,
    AppliedResidencyChange,
    AutoResidencyPlan,
    DefaultWorkload,
    RankResidencyReport,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


class TestAutoResidencyWarmup(unittest.TestCase):
    def test_worker_rolls_back_a_materially_slower_calibrated_round(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = 0
        worker.is_output_rank = False
        worker.server_args = SimpleNamespace()
        worker._auto_residency_warmup_records = [object()]
        worker._auto_residency_round_sizes = [1]
        report = RankResidencyReport(
            rank=0,
            budget_bytes=1,
            estimated_peak_bytes=1,
            estimated_request_duration_ns=10_000_000_000,
            measured_request_duration_ns=20_000_000_000,
        )
        worker._build_auto_residency_report = mock.Mock(return_value=report)
        worker._auto_residency_all_gather = mock.Mock(return_value=[report])
        rolled_back = OutputBatch(
            error="request duration regressed",
            output={"status": PLACEMENT_STATUS_ROLLED_BACK},
        )
        worker._rollback_everywhere = mock.Mock(return_value=rolled_back)

        with mock.patch.object(
            gpu_worker_module,
            "resolve_default_workload",
            return_value=SimpleNamespace(),
        ), mock.patch.object(
            gpu_worker_module,
            "resolve_measured_default_workload",
            return_value=SimpleNamespace(),
        ):
            response = worker.apply_auto_residency()

        self.assertIs(response, rolled_back)
        worker._rollback_everywhere.assert_called_once()
        kwargs = worker._rollback_everywhere.call_args.kwargs
        self.assertIn("10.00s to 20.00s", kwargs["cause"])
        self.assertFalse(kwargs["already_failed"])
        self.assertTrue(kwargs["latest_round_only"])

    def test_new_placement_that_spends_the_reserve_rolls_back(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = 0
        worker.is_output_rank = False
        worker.pipeline = SimpleNamespace(modules={})
        worker.server_args = SimpleNamespace()
        worker._auto_residency_warmup_records = []
        worker._auto_residency_round_sizes = [1]
        worker._build_auto_residency_report = mock.Mock(
            return_value=RankResidencyReport(
                rank=0,
                budget_bytes=30 * GIB_BYTES,
                estimated_peak_bytes=28 * GIB_BYTES,
            )
        )
        worker._auto_residency_all_gather = lambda value: [value]
        expected = OutputBatch(output={"status": PLACEMENT_STATUS_ROLLED_BACK})
        worker._rollback_everywhere = mock.Mock(return_value=expected)
        workload = DefaultWorkload(832, 480, 24, 50)
        plan = AutoResidencyPlan(
            current_placement_reserve_shortfall_bytes=2 * GIB_BYTES
        )

        with (
            mock.patch.object(
                gpu_worker_module, "resolve_default_workload", return_value=workload
            ),
            mock.patch.object(
                gpu_worker_module,
                "resolve_measured_default_workload",
                return_value=workload,
            ),
            mock.patch.object(
                gpu_worker_module, "plan_auto_residency", return_value=plan
            ),
        ):
            response = worker.apply_auto_residency()

        self.assertIs(response, expected)
        worker._rollback_everywhere.assert_called_once_with(
            cause="VRAM reserve exceeded by 2.0 GiB",
            already_failed=False,
            latest_round_only=True,
        )

    def test_worker_rollback_response_rewarms_the_restored_layout(self):
        actions = []

        async def forward(req):
            self.assertIsInstance(req, AutoResidencyReq)
            actions.append(req.action)
            return OutputBatch(
                error="post-adjustment warmup output changed",
                output={"status": PLACEMENT_STATUS_ROLLED_BACK},
            )

        rewarm = mock.AsyncMock()
        server_args = SimpleNamespace(
            performance_mode="auto",
            warmup_resolutions=None,
        )
        with (
            mock.patch.object(
                server_warmup, "auto_residency_skip_reason", return_value=None
            ),
            mock.patch.object(server_warmup, "run_async_client_warmup", rewarm),
        ):
            asyncio.run(server_warmup.maybe_apply_auto_residency(server_args, forward))

        self.assertEqual(actions, ["apply"])
        rewarm.assert_awaited_once_with(
            server_args,
            forward,
            fail_open=True,
            rewarm=True,
        )

    def test_replans_until_the_calibrated_layout_reaches_a_fixed_point(self):
        responses = iter(
            [
                OutputBatch(output={"status": PLACEMENT_STATUS_ADJUSTED}),
                OutputBatch(output={"status": PLACEMENT_STATUS_ADJUSTED}),
                OutputBatch(output={"status": PLACEMENT_STATUS_SKIPPED}),
            ]
        )
        actions = []

        async def forward(req):
            self.assertIsInstance(req, AutoResidencyReq)
            actions.append(req.action)
            return next(responses)

        rewarm = mock.AsyncMock()
        server_args = SimpleNamespace(
            performance_mode="auto",
            warmup_resolutions=None,
        )
        with (
            mock.patch.object(
                server_warmup, "auto_residency_skip_reason", return_value=None
            ),
            mock.patch.object(server_warmup, "run_async_client_warmup", rewarm),
        ):
            asyncio.run(server_warmup.maybe_apply_auto_residency(server_args, forward))

        self.assertEqual(actions, ["apply", "apply", "apply"])
        self.assertEqual(rewarm.await_count, 2)

    def test_failed_first_calibration_rolls_back_and_rewarms(self):
        actions = []

        async def forward(req):
            self.assertIsInstance(req, AutoResidencyReq)
            actions.append(req.action)
            status = (
                PLACEMENT_STATUS_ADJUSTED
                if req.action == "apply"
                else PLACEMENT_STATUS_ROLLED_BACK
            )
            return OutputBatch(output={"status": status})

        rewarm = mock.AsyncMock(side_effect=[RuntimeError("oom"), None])
        server_args = SimpleNamespace(
            performance_mode="auto",
            warmup_resolutions=None,
        )
        with (
            mock.patch.object(
                server_warmup, "auto_residency_skip_reason", return_value=None
            ),
            mock.patch.object(server_warmup, "run_async_client_warmup", rewarm),
        ):
            asyncio.run(server_warmup.maybe_apply_auto_residency(server_args, forward))

        self.assertEqual(actions, ["apply", "rollback"])
        self.assertEqual(rewarm.await_count, 2)

    def test_failed_calibration_reports_rollback_failure(self):
        async def forward(req):
            self.assertIsInstance(req, AutoResidencyReq)
            if req.action == "apply":
                return OutputBatch(output={"status": PLACEMENT_STATUS_ADJUSTED})
            return OutputBatch(
                error="rollback failed",
                output={"status": PLACEMENT_STATUS_ROLLBACK_FAILED},
            )

        rewarm = mock.AsyncMock(side_effect=RuntimeError("oom"))
        server_args = SimpleNamespace(
            performance_mode="auto",
            warmup_resolutions=None,
        )
        with (
            mock.patch.object(
                server_warmup, "auto_residency_skip_reason", return_value=None
            ),
            mock.patch.object(server_warmup, "run_async_client_warmup", rewarm),
            self.assertRaisesRegex(RuntimeError, "auto residency rollback failed"),
        ):
            asyncio.run(server_warmup.maybe_apply_auto_residency(server_args, forward))

    def test_apply_rpc_failure_after_calibration_aborts_without_rollback(self):
        actions = []
        responses = iter(
            [
                OutputBatch(output={"status": PLACEMENT_STATUS_ADJUSTED}),
                RuntimeError("rpc failed"),
            ]
        )

        async def forward(req):
            self.assertIsInstance(req, AutoResidencyReq)
            actions.append(req.action)
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        rewarm = mock.AsyncMock()
        server_args = SimpleNamespace(
            performance_mode="auto",
            warmup_resolutions=None,
        )
        with (
            mock.patch.object(
                server_warmup, "auto_residency_skip_reason", return_value=None
            ),
            mock.patch.object(server_warmup, "run_async_client_warmup", rewarm),
            self.assertRaisesRegex(
                RuntimeError,
                "auto residency apply failed after a calibrated adjustment",
            ),
        ):
            asyncio.run(server_warmup.maybe_apply_auto_residency(server_args, forward))

        self.assertEqual(actions, ["apply", "apply"])
        self.assertEqual(rewarm.await_count, 1)

    def test_failed_later_round_keeps_earlier_calibrated_promotions(self):
        first_round = AppliedResidencyChange("transformer", COMPONENT_OFFLOAD)
        latest_round = [
            AppliedResidencyChange("text_encoder", COMPONENT_OFFLOAD),
            AppliedResidencyChange("vae", COMPONENT_OFFLOAD),
        ]
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = 0
        worker.is_output_rank = False
        worker.pipeline = SimpleNamespace(modules={})
        worker.server_args = SimpleNamespace()
        worker._auto_residency_applied = [first_round, *latest_round]
        worker._auto_residency_round_sizes = [1, 2]
        worker._auto_residency_all_gather = lambda value: [value]
        worker._invalidate_component_strategies = mock.Mock()

        with mock.patch.object(
            gpu_worker_module, "rollback_residency_changes"
        ) as rollback:
            response = worker.rollback_auto_residency()

        rollback.assert_called_once_with(
            applied=latest_round,
            modules={},
            server_args=worker.server_args,
        )
        self.assertEqual(response.output["status"], PLACEMENT_STATUS_ROLLED_BACK)
        self.assertEqual(worker._auto_residency_applied, [first_round])
        self.assertEqual(worker._auto_residency_round_sizes, [1])

    def test_failed_apply_round_does_not_rollback_prior_round(self):
        first_round = AppliedResidencyChange("transformer", COMPONENT_OFFLOAD)
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = 0
        worker.pipeline = SimpleNamespace(modules={})
        worker.server_args = SimpleNamespace()
        worker._auto_residency_applied = [first_round]
        worker._auto_residency_round_sizes = [1, 0]
        worker._invalidate_component_strategies = mock.Mock()

        with mock.patch.object(
            gpu_worker_module, "rollback_residency_changes"
        ) as rollback:
            error = worker._rollback_applied_residency_changes(latest_round_only=True)

        self.assertIsNone(error)
        rollback.assert_not_called()
        self.assertEqual(worker._auto_residency_applied, [first_round])
        self.assertEqual(worker._auto_residency_round_sizes, [1])
