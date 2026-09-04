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
    PLACEMENT_STATUS_VALIDATED,
    AppliedResidencyChange,
    AutoResidencyPlan,
    DefaultWorkload,
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
        req = SimpleNamespace(
            is_warmup=True,
            extra={},
            num_inference_steps=4,
        )
        with mock.patch.object(
            server_warmup,
            "build_warmup_reqs",
            return_value=[req],
        ):
            result = server_warmup.build_client_warmup_reqs(
                SimpleNamespace(warmup_resolutions=None),
                rewarm=True,
                step_limit=1,
            )

        self.assertEqual(result, [req])
        self.assertEqual(req.num_inference_steps, 1)
        self.assertEqual(req.extra["warmup_total"], 1)
        self.assertTrue(req.extra["server_warmup_rewarm"])

    def test_residency_hint_uses_effective_device_budget(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker._auto_residency_budget_bytes = mock.Mock(return_value=12 * GIB_BYTES)
        worker.get_can_stay_resident_components = mock.Mock(return_value=[])
        output = SimpleNamespace(metrics=None)
        snapshot = SimpleNamespace(
            peak_reserved_mb=11 * 1024,
            peak_allocated_mb=10 * 1024,
        )

        with mock.patch.object(
            gpu_worker_module, "capture_memory_snapshot", return_value=snapshot
        ):
            worker.do_mem_analysis(output)

        worker.get_can_stay_resident_components.assert_called_once_with(1.0)
        self.assertEqual(output.peak_memory_mb, 11 * 1024)

    def test_sync_startup_uses_shared_auto_residency_sequence(self):
        server_args = SimpleNamespace()
        forward = mock.Mock()

        with (
            mock.patch.object(
                server_warmup, "maybe_apply_pre_warmup_auto_residency"
            ) as apply_static,
            mock.patch.object(server_warmup, "run_async_client_warmup") as warmup,
            mock.patch.object(server_warmup, "maybe_apply_auto_residency") as refine,
            mock.patch.object(
                server_warmup,
                "should_run_synthetic_server_warmup",
                return_value=True,
            ),
        ):
            server_warmup.run_sync_startup_warmup(server_args, forward)

        apply_static.assert_awaited_once()
        warmup.assert_awaited_once()
        refine.assert_awaited_once()
        self.assertIs(apply_static.await_args.args[0], server_args)
        self.assertIs(warmup.await_args.args[0], server_args)
        self.assertIs(refine.await_args.args[0], server_args)
        self.assertTrue(warmup.await_args.kwargs["fail_open"])
        self.assertIs(apply_static.await_args.args[1], warmup.await_args.args[1])
        self.assertIs(apply_static.await_args.args[1], refine.await_args.args[1])

    def test_sync_startup_static_planning_does_not_require_warmup(self):
        with (
            mock.patch.object(
                server_warmup, "maybe_apply_pre_warmup_auto_residency"
            ) as apply_static,
            mock.patch.object(
                server_warmup,
                "should_run_synthetic_server_warmup",
                return_value=False,
            ),
            mock.patch.object(server_warmup, "run_async_client_warmup") as warmup,
            mock.patch.object(server_warmup, "maybe_apply_auto_residency") as refine,
        ):
            server_warmup.run_sync_startup_warmup(SimpleNamespace(), mock.Mock())

        apply_static.assert_awaited_once()
        warmup.assert_not_awaited()
        refine.assert_not_awaited()

    def test_pre_warmup_planner_uses_static_action(self):
        actions = []

        async def forward(req):
            actions.append(req.action)
            return OutputBatch(output={"status": PLACEMENT_STATUS_SKIPPED})

        with mock.patch.object(
            server_warmup,
            "should_apply_pre_warmup_auto_residency",
            return_value=True,
        ):
            asyncio.run(
                server_warmup.maybe_apply_pre_warmup_auto_residency(
                    SimpleNamespace(), forward
                )
            )

        self.assertEqual(actions, ["apply_static"])

    def test_pre_warmup_worker_commits_seed_without_pending_round(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = 0
        worker.is_output_rank = False
        worker.server_args = SimpleNamespace(
            residency_mode=lambda _name: COMPONENT_OFFLOAD
        )
        worker._auto_residency_warmup_records = []
        worker._auto_residency_applied = []
        worker._auto_residency_round_sizes = []
        worker._auto_residency_last_applied_plan = None
        report = RankResidencyReport(
            rank=0,
            budget_bytes=10 * GIB_BYTES,
            estimated_peak_bytes=5 * GIB_BYTES,
        )
        worker._build_pre_warmup_auto_residency_report = mock.Mock(return_value=report)
        worker._auto_residency_all_gather = mock.Mock(side_effect=lambda value: [value])
        worker._auto_residency_modules = mock.Mock(return_value={})
        worker._invalidate_component_strategies = mock.Mock()
        candidate = ResidencyTarget(
            component_name="transformer",
            residency_mode=RESIDENT,
            target_resident_weight_bytes=0,
            h2d_bytes_per_request=1,
            target_residency_mode=COMPONENT_OFFLOAD,
        )
        plan = AutoResidencyPlan(changes=[candidate])
        applied = AppliedResidencyChange("transformer", RESIDENT)
        workload = DefaultWorkload(512, 512, 1, 4)

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
            mock.patch.object(
                gpu_worker_module, "apply_residency_changes", return_value=[applied]
            ),
            mock.patch.object(gpu_worker_module, "commit_residency_changes") as commit,
        ):
            response = worker.apply_auto_residency(pre_warmup=True)

        self.assertEqual(response.output["status"], PLACEMENT_STATUS_ADJUSTED)
        commit.assert_called_once_with(
            applied=[applied], modules={}, server_args=worker.server_args
        )
        self.assertEqual(worker._auto_residency_applied, [])
        self.assertEqual(worker._auto_residency_round_sizes, [])
        self.assertIsNone(worker._auto_residency_last_applied_plan)

    def test_auto_residency_uses_first_oom_without_degrading_probe(self):
        req = SimpleNamespace()
        forward = mock.AsyncMock(return_value=OutputBatch(error="CUDA out of memory"))

        with (
            mock.patch.object(
                server_warmup, "auto_residency_skip_reason", return_value=None
            ),
            mock.patch.object(
                server_warmup,
                "should_include_warmup_image",
                return_value=False,
            ),
            mock.patch.object(
                server_warmup, "build_client_warmup_reqs", return_value=[req]
            ),
            mock.patch.object(server_warmup, "_degrade_after_oom") as degrade,
        ):
            asyncio.run(
                server_warmup.run_async_client_warmup(
                    SimpleNamespace(), forward, fail_open=True
                )
            )

        forward.assert_awaited_once_with(req)
        degrade.assert_not_called()

    def test_non_auto_warmup_keeps_oom_probe_degradation(self):
        req = SimpleNamespace()
        lighter = SimpleNamespace()
        forward = mock.AsyncMock(
            side_effect=[
                OutputBatch(error="CUDA out of memory"),
                OutputBatch(),
            ]
        )

        with (
            mock.patch.object(
                server_warmup,
                "auto_residency_skip_reason",
                return_value="not auto",
            ),
            mock.patch.object(
                server_warmup,
                "should_include_warmup_image",
                return_value=False,
            ),
            mock.patch.object(
                server_warmup, "build_client_warmup_reqs", return_value=[req]
            ),
            mock.patch.object(
                server_warmup, "_degrade_after_oom", return_value=lighter
            ) as degrade,
        ):
            asyncio.run(
                server_warmup.run_async_client_warmup(SimpleNamespace(), forward)
            )

        self.assertEqual(
            [call.args[0] for call in forward.await_args_list],
            [req, lighter],
        )
        degrade.assert_called_once_with(mock.ANY, req)

    def test_worker_rolls_back_a_materially_slower_calibrated_round(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = 0
        worker.is_output_rank = False
        worker.server_args = SimpleNamespace(
            residency_mode=lambda _name: COMPONENT_OFFLOAD
        )
        worker._auto_residency_warmup_records = [object()]
        worker._auto_residency_applied = [
            AppliedResidencyChange("transformer", COMPONENT_OFFLOAD)
        ]
        worker._auto_residency_round_sizes = [1]
        report = RankResidencyReport(
            rank=0,
            budget_bytes=1,
            estimated_peak_bytes=1,
            estimated_request_duration_ns=10_000_000_000,
            measured_request_duration_ns=20_000_000_000,
            reference_probe_duration_ns=10_000_000_000,
            probe_duration_ns=20_000_000_000,
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

    def test_worker_keeps_resident_only_round_despite_noisy_duration(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = 0
        worker.is_output_rank = False
        worker.server_args = SimpleNamespace(residency_mode=lambda _name: RESIDENT)
        worker._auto_residency_warmup_records = [object()]
        worker._auto_residency_applied = [
            AppliedResidencyChange("text_encoder", COMPONENT_OFFLOAD),
            AppliedResidencyChange("vae", COMPONENT_OFFLOAD),
        ]
        worker._auto_residency_repeated_components = set()
        worker._auto_residency_round_sizes = [2]
        worker._auto_residency_last_applied_plan = AutoResidencyPlan()
        report = RankResidencyReport(
            rank=0,
            budget_bytes=10 * GIB_BYTES,
            estimated_peak_bytes=1,
            estimated_request_duration_ns=10_000_000_000,
            measured_request_duration_ns=20_000_000_000,
            reference_probe_duration_ns=10_000_000_000,
            probe_duration_ns=20_000_000_000,
        )
        worker._build_auto_residency_report = mock.Mock(return_value=report)
        worker._auto_residency_all_gather = mock.Mock(side_effect=lambda value: [value])
        worker._rollback_everywhere = mock.Mock()
        self.assertTrue(worker._latest_auto_residency_round_supports_short_validation())

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

        self.assertEqual(response.output["status"], PLACEMENT_STATUS_VALIDATED)
        worker._rollback_everywhere.assert_not_called()
        self.assertEqual(worker._auto_residency_round_sizes, [])

    def test_validation_rolls_back_when_selected_placement_spends_the_reserve(self):
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = 0
        worker.is_output_rank = False
        worker.pipeline = SimpleNamespace(modules={})
        worker.server_args = SimpleNamespace(
            residency_mode=lambda _name: COMPONENT_OFFLOAD
        )
        worker._auto_residency_warmup_records = []
        worker._auto_residency_applied = []
        worker._auto_residency_round_sizes = [1]
        worker._auto_residency_last_applied_plan = AutoResidencyPlan(
            resource_budget_bytes={"gpu:rank0:denoise": 8 * GIB_BYTES},
            resource_delta_bytes={"gpu:rank0:denoise": 7 * GIB_BYTES},
        )
        worker._build_auto_residency_report = mock.Mock(
            return_value=RankResidencyReport(
                rank=0,
                budget_bytes=30 * GIB_BYTES,
                estimated_peak_bytes=28 * GIB_BYTES,
                target_workload_measured=True,
            )
        )
        worker._auto_residency_all_gather = lambda value: [value]
        expected = OutputBatch(output={"status": PLACEMENT_STATUS_ROLLED_BACK})
        worker._rollback_everywhere = mock.Mock(return_value=expected)
        workload = DefaultWorkload(832, 480, 24, 50)

        with (
            mock.patch.object(
                gpu_worker_module, "resolve_default_workload", return_value=workload
            ),
            mock.patch.object(
                gpu_worker_module,
                "resolve_measured_default_workload",
                return_value=workload,
            ),
            mock.patch.object(gpu_worker_module, "plan_auto_residency") as plan,
        ):
            response = worker.apply_auto_residency(validate_only=True)

        self.assertIs(response, expected)
        worker._rollback_everywhere.assert_called_once_with(
            cause="VRAM reserve exceeded by 1.0 GiB",
            already_failed=False,
            latest_round_only=True,
        )
        plan.assert_not_called()

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
            budget_bytes=10 * GIB_BYTES,
            estimated_peak_bytes=1,
            estimated_request_duration_ns=10_000_000_000,
            measured_request_duration_ns=10_000_000_000,
            reference_probe_duration_ns=10_000_000_000,
            probe_duration_ns=10_000_000_000,
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
        worker._auto_residency_repeated_components = set()

        worker._auto_residency_applied = [
            AppliedResidencyChange("text_encoder", COMPONENT_OFFLOAD)
        ]
        self.assertTrue(worker._latest_auto_residency_round_supports_short_validation())

        worker._auto_residency_applied = [
            AppliedResidencyChange("transformer", COMPONENT_OFFLOAD)
        ]
        self.assertTrue(worker._latest_auto_residency_round_supports_short_validation())

        worker._auto_residency_applied = [
            AppliedResidencyChange("custom_refiner", COMPONENT_OFFLOAD)
        ]
        worker._auto_residency_repeated_components = {"custom_refiner"}
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

    def test_unrecoverable_warmup_oom_aborts_startup(self):
        async def forward(_req):
            return OutputBatch(
                output={
                    "status": PLACEMENT_STATUS_SKIPPED,
                    "recovering_from_oom": True,
                }
            )

        server_args = SimpleNamespace(
            performance_mode="auto",
            warmup_resolutions=None,
        )
        with (
            mock.patch.object(
                server_warmup, "auto_residency_skip_reason", return_value=None
            ),
            self.assertRaisesRegex(RuntimeError, "found no feasible placement"),
        ):
            asyncio.run(server_warmup.maybe_apply_auto_residency(server_args, forward))

    def test_non_resident_adjustment_keeps_timing_validation_steps(self):
        async def forward(req):
            status = (
                PLACEMENT_STATUS_ADJUSTED
                if req.action == "apply"
                else PLACEMENT_STATUS_VALIDATED
            )
            return OutputBatch(output={"status": status})

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

        rewarm.assert_awaited_once_with(
            server_args,
            forward,
            fail_open=False,
            rewarm=True,
        )

    def test_regressed_validation_rolls_back_and_rewarms_original_placement(self):
        responses = iter(
            [
                OutputBatch(output={"status": PLACEMENT_STATUS_ADJUSTED}),
                OutputBatch(
                    error="request duration regressed",
                    output={"status": PLACEMENT_STATUS_ROLLED_BACK},
                ),
            ]
        )
        actions = []

        async def forward(req):
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

    def test_successful_calibration_does_not_start_another_apply_round(self):
        actions = []

        async def forward(req):
            self.assertIsInstance(req, AutoResidencyReq)
            actions.append(req.action)
            status = (
                PLACEMENT_STATUS_ADJUSTED
                if req.action == "apply"
                else PLACEMENT_STATUS_VALIDATED
            )
            return OutputBatch(output={"status": status})

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


def test_short_validation_respects_pipeline_minimum_steps():
    from sglang.multimodal_gen.runtime.server_warmup import _short_validation_step_limit

    assert _short_validation_step_limit(SimpleNamespace(pipeline_config=None)) == 1
    assert (
        _short_validation_step_limit(
            SimpleNamespace(pipeline_config=SimpleNamespace(minimum_inference_steps=1))
        )
        == 1
    )
    assert (
        _short_validation_step_limit(
            SimpleNamespace(pipeline_config=SimpleNamespace(minimum_inference_steps=2))
        )
        == 2
    )
