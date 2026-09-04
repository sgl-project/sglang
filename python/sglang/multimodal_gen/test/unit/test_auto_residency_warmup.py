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
    PLACEMENT_STATUS_ADJUSTED,
    PLACEMENT_STATUS_ROLLBACK_FAILED,
    PLACEMENT_STATUS_ROLLED_BACK,
    PLACEMENT_STATUS_VALIDATED,
    AppliedResidencyChange,
    AutoResidencyPlan,
    RankResidencyReport,
    ResidencyTarget,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    RESIDENT,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentResidencyStrategy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


class TestAutoResidencyWarmup(unittest.TestCase):
    def test_worker_skips_only_fixed_custom_residency_strategies(self):
        class AutoCompatibleStrategy(ComponentResidencyStrategy):
            def supports_auto_residency(self) -> bool:
                return True

        worker = GPUWorker.__new__(GPUWorker)
        worker.pipeline = SimpleNamespace(
            component_residency_strategies={
                "fixed": ComponentResidencyStrategy(),
                "dynamic": AutoCompatibleStrategy(),
            },
            modules={},
        )
        worker.server_args = SimpleNamespace(
            component_quantizations=(),
            quantization=None,
            direct_gpu_weight_loading=False,
            ltx2_two_stage_device_mode=None,
            nunchaku_config=None,
        )

        self.assertEqual(worker._fixed_custom_residency_strategy_names(), {"fixed"})

    def test_rewarm_step_limit_keeps_full_shape_but_shortens_denoising(self):
        req = SimpleNamespace(is_warmup=True, extra={}, num_inference_steps=4)
        with mock.patch.object(server_warmup, "build_warmup_reqs", return_value=[req]):
            result = server_warmup.build_client_warmup_reqs(
                SimpleNamespace(warmup_resolutions=None),
                rewarm=True,
                step_limit=1,
            )

        self.assertEqual(result, [req])
        self.assertEqual(req.num_inference_steps, 1)
        self.assertTrue(req.extra["server_warmup_rewarm"])

    def test_worker_rolls_back_a_materially_slower_calibrated_round(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = 0
        worker.is_output_rank = False
        worker.pipeline = SimpleNamespace(modules={})
        worker.server_args = SimpleNamespace(
            residency_mode=lambda _name: COMPONENT_OFFLOAD
        )
        worker._auto_residency_warmup_records = [object()]
        worker._auto_residency_applied = []
        worker._auto_residency_round_sizes = [1]
        report = RankResidencyReport(
            rank=0,
            budget_bytes=1,
            estimated_peak_bytes=1,
            estimated_request_duration_ns=10_000_000_000,
            measured_request_duration_ns=20_000_000_000,
        )
        worker._build_auto_residency_report = mock.Mock(return_value=report)
        worker._auto_residency_all_gather = mock.Mock(side_effect=lambda value: [value])
        rolled_back = OutputBatch(
            error="request duration regressed",
            output={"status": PLACEMENT_STATUS_ROLLED_BACK},
        )
        worker._rollback_everywhere = mock.Mock(return_value=rolled_back)

        with (
            mock.patch.object(
                gpu_worker_module,
                "resolve_default_workload",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(
                gpu_worker_module,
                "resolve_measured_default_workload",
                return_value=SimpleNamespace(),
            ),
        ):
            response = worker.apply_auto_residency(validate_only=True)

        self.assertIs(response, rolled_back)
        worker._rollback_everywhere.assert_called_once()
        kwargs = worker._rollback_everywhere.call_args.kwargs
        self.assertIn("10.00s to 20.00s", kwargs["cause"])
        self.assertFalse(kwargs["already_failed"])
        self.assertTrue(kwargs["latest_round_only"])

    def test_worker_validation_does_not_apply_a_second_candidate_plan(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = 0
        worker.is_output_rank = False
        worker.pipeline = SimpleNamespace(modules={})
        worker.server_args = SimpleNamespace(
            residency_mode=lambda _name: COMPONENT_OFFLOAD
        )
        worker._auto_residency_warmup_records = [object()]
        worker._auto_residency_applied = []
        worker._auto_residency_round_sizes = [1]
        report = RankResidencyReport(
            rank=0,
            budget_bytes=1,
            estimated_peak_bytes=1,
            estimated_request_duration_ns=10_000_000_000,
            measured_request_duration_ns=10_000_000_000,
        )
        worker._build_auto_residency_report = mock.Mock(return_value=report)
        worker._auto_residency_all_gather = mock.Mock(side_effect=lambda value: [value])
        candidate = ResidencyTarget(
            component_name="vae",
            residency_mode=COMPONENT_OFFLOAD,
            target_resident_weight_bytes=1,
            h2d_bytes_per_request=1,
        )
        worker._auto_residency_last_applied_plan = AutoResidencyPlan(
            changes=[candidate]
        )

        with (
            mock.patch.object(
                gpu_worker_module,
                "resolve_default_workload",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(
                gpu_worker_module,
                "resolve_measured_default_workload",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(
                gpu_worker_module,
                "plan_auto_residency",
            ) as plan,
            mock.patch.object(gpu_worker_module, "apply_residency_changes") as apply,
        ):
            response = worker.apply_auto_residency(validate_only=True)

        self.assertEqual(response.output["status"], PLACEMENT_STATUS_VALIDATED)
        self.assertIsNone(worker._auto_residency_last_applied_plan)
        worker._build_auto_residency_report.assert_called_once_with(
            workload=mock.ANY,
            records=[mock.ANY],
            include_candidates=False,
        )
        plan.assert_not_called()
        apply.assert_not_called()

    def test_resident_promotions_use_short_validation(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker.server_args = SimpleNamespace(residency_mode=lambda _name: RESIDENT)
        worker._auto_residency_round_sizes = [1]

        worker._auto_residency_applied = [
            AppliedResidencyChange("text_encoder", COMPONENT_OFFLOAD)
        ]
        self.assertTrue(worker._latest_auto_residency_round_supports_short_validation())

        worker._auto_residency_applied = [
            AppliedResidencyChange("transformer", COMPONENT_OFFLOAD)
        ]
        self.assertTrue(worker._latest_auto_residency_round_supports_short_validation())

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
            fail_open=False,
            rewarm=True,
        )

    def test_successful_adjustment_runs_one_validation_without_replanning(self):
        responses = iter(
            [
                OutputBatch(
                    output={
                        "status": PLACEMENT_STATUS_ADJUSTED,
                        "short_validation": True,
                    }
                ),
                OutputBatch(output={"status": PLACEMENT_STATUS_VALIDATED}),
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

        self.assertEqual(actions, ["apply", "validate"])
        rewarm.assert_awaited_once_with(
            server_args,
            forward,
            fail_open=False,
            rewarm=True,
            step_limit=1,
        )

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

    def test_validation_rpc_failure_rolls_back_and_rewarms(self):
        actions = []
        responses = iter(
            [
                OutputBatch(output={"status": PLACEMENT_STATUS_ADJUSTED}),
                RuntimeError("rpc failed"),
                OutputBatch(output={"status": PLACEMENT_STATUS_ROLLED_BACK}),
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
        ):
            asyncio.run(server_warmup.maybe_apply_auto_residency(server_args, forward))

        self.assertEqual(actions, ["apply", "validate", "rollback"])
        self.assertEqual(rewarm.await_count, 2)

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
