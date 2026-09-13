# SPDX-License-Identifier: Apache-2.0
"""Observer is start/end only; per-step collection is ``step_latents``."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.post_training.denoise_loop_observer import (
    NullDenoiseLoopObserver,
    RolloutDenoiseLoopObserver,
    get_denoise_loop_observer,
)
from sglang.multimodal_gen.runtime.post_training.rollout_denoising_mixin import (
    RolloutDenoisingMixin,
)


class _RecordingStage(RolloutDenoisingMixin):
    def __init__(self):
        self.server_args = SimpleNamespace(pipeline_config=SimpleNamespace())


def _batch(*, rollout: bool, return_traj: bool = True):
    return SimpleNamespace(
        rollout=rollout,
        rollout_return_dit_trajectory=return_traj,
        rollout_return_denoising_env=False,
        rollout_return_step_indices=None,
        rollout_trajectory_data=None,
        scheduler=SimpleNamespace(sigmas=torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])),
        _denoise_loop_observer=None,
        _rollout_denoising_env_state=None,
    )


def test_observer_factory_and_serving_is_noop():
    serving = _batch(rollout=False)
    assert isinstance(get_denoise_loop_observer(serving), NullDenoiseLoopObserver)
    assert get_denoise_loop_observer(serving) is serving._denoise_loop_observer

    rollout = _batch(rollout=True)
    assert isinstance(get_denoise_loop_observer(rollout), RolloutDenoiseLoopObserver)

    stage = _RecordingStage()
    observer = get_denoise_loop_observer(serving)
    stage.step_latents(
        serving, torch.zeros(1, 4), torch.tensor(1.0), 0, apply=lambda: None
    )
    observer.finalize(
        stage,
        serving,
        latents=torch.zeros(1, 4),
        num_inference_steps=1,
        final_timestep=torch.zeros(()),
        server_args=SimpleNamespace(pipeline_config=SimpleNamespace()),
    )
    assert serving.rollout_trajectory_data is None


def test_step_latents_collects_trajectory():
    batch = _batch(rollout=True)
    stage = _RecordingStage()
    observer = get_denoise_loop_observer(batch)
    server_args = SimpleNamespace(pipeline_config=SimpleNamespace())

    observer.init_env(
        stage,
        batch,
        pipeline_config=server_args.pipeline_config,
        image_kwargs={},
        pos_cond_kwargs={"encoder_hidden_states": torch.zeros(1, 2, 4)},
        neg_cond_kwargs=None,
        guidance=None,
    )
    latents = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    for step in range(3):
        latents = stage.step_latents(
            batch,
            latents,
            torch.tensor(1.0 - 0.25 * step),
            step,
            apply=lambda current=latents: current + 1,
        )

    with patch(
        "sglang.multimodal_gen.runtime.post_training.sp_utils.get_sp_world_size",
        return_value=1,
    ):
        observer.finalize(
            stage,
            batch,
            latents=latents,
            num_inference_steps=3,
            final_timestep=torch.zeros(()),
            server_args=server_args,
        )

    traj = batch.rollout_trajectory_data.dit_trajectory
    assert traj.latents.shape[1] == 4
