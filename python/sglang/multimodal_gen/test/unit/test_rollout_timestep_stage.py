import types
import unittest

import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)


def _build_batch(*, rollout: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        rollout=rollout,
        scheduler=None,
        timesteps=None,
        sigmas=None,
        num_inference_steps=8,
        n_tokens=None,
        is_warmup=False,
        extra={},
    )


def _build_server_args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        pipeline_config=types.SimpleNamespace(
            prepare_sigmas=lambda sigmas, num_inference_steps: sigmas
        )
    )


class TestRolloutTimestepStage(unittest.TestCase):
    """Per-request scheduler binding on batch.rollout (Wan-style binding:
    UniPC serves, flow-match Euler handles the rollout SDE/log-prob path)."""

    def _build_stage(self) -> TimestepPreparationStage:
        return TimestepPreparationStage(
            scheduler=FlowUniPCMultistepScheduler(shift=5.0),
            rollout_scheduler=FlowMatchEulerDiscreteScheduler(shift=5.0),
        )

    def test_serving_request_binds_serving_scheduler(self):
        stage = self._build_stage()
        batch = stage.forward(_build_batch(rollout=False), _build_server_args())
        self.assertIsInstance(batch.scheduler, FlowUniPCMultistepScheduler)
        self.assertIs(batch.scheduler, stage.scheduler)
        self.assertEqual(batch.timesteps.shape[0], 8)

    def test_rollout_request_binds_rollout_scheduler(self):
        stage = self._build_stage()
        batch = stage.forward(_build_batch(rollout=True), _build_server_args())
        self.assertIsInstance(batch.scheduler, FlowMatchEulerDiscreteScheduler)
        self.assertIs(batch.scheduler, stage.rollout_scheduler)
        self.assertEqual(batch.timesteps.shape[0], 8)

    def test_rollout_request_without_rollout_scheduler_uses_serving(self):
        # Pipelines that already serve flow-match Euler pass no
        # rollout_scheduler; rollout requests keep the serving scheduler.
        stage = TimestepPreparationStage(
            scheduler=FlowMatchEulerDiscreteScheduler(shift=5.0)
        )
        batch = stage.forward(_build_batch(rollout=True), _build_server_args())
        self.assertIs(batch.scheduler, stage.scheduler)

    def test_rollout_timesteps_follow_flow_match_convention(self):
        # The rollout SDE/log-prob math assumes
        # timesteps == sigmas[:-1] * num_train_timesteps.
        stage = self._build_stage()
        batch = stage.forward(_build_batch(rollout=True), _build_server_args())
        scheduler = batch.scheduler
        reconstructed = scheduler.sigmas[:-1].to(
            device=scheduler.timesteps.device
        ) * float(scheduler.config.num_train_timesteps)
        max_abs_diff = (
            (scheduler.timesteps.float() - reconstructed.float()).abs().max().item()
        )
        self.assertLessEqual(max_abs_diff, 1e-3)

    def test_convention_violation_raises(self):
        stage = self._build_stage()
        bad = types.SimpleNamespace(
            sigmas=torch.tensor([1.0, 0.5, 0.0]),
            timesteps=torch.tensor([123.0, 45.0]),
            config=types.SimpleNamespace(num_train_timesteps=1000),
        )
        with self.assertRaisesRegex(ValueError, "rollout timestep/sigma mismatch"):
            stage._check_rollout_timesteps(bad)

    def test_shift_reaches_rollout_scheduler(self):
        # A launch-time flow_shift override must shape the rollout sigmas the
        # same way it shapes serving; the binding passes shift at construction.
        shifted = TimestepPreparationStage(
            scheduler=FlowUniPCMultistepScheduler(shift=5.0),
            rollout_scheduler=FlowMatchEulerDiscreteScheduler(shift=5.0),
        )
        unshifted = TimestepPreparationStage(
            scheduler=FlowUniPCMultistepScheduler(shift=1.0),
            rollout_scheduler=FlowMatchEulerDiscreteScheduler(shift=1.0),
        )
        batch_shifted = shifted.forward(
            _build_batch(rollout=True), _build_server_args()
        )
        batch_unshifted = unshifted.forward(
            _build_batch(rollout=True), _build_server_args()
        )
        self.assertFalse(
            torch.allclose(
                batch_shifted.scheduler.sigmas, batch_unshifted.scheduler.sigmas
            )
        )


if __name__ == "__main__":
    unittest.main()
