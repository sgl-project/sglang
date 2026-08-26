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
    PLACEMENT_STATUS_SKIPPED,
    AppliedResidencyChange,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


class TestAutoResidencyWarmup(unittest.TestCase):
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
