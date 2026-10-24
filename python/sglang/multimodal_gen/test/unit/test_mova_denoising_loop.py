import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    mova,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.mova import (
    MOVADenoisingStage,
)

_POS = 1.0
_NEG = 0.5
_GUIDANCE = 5.0
_STEPS = 2


class _AccumulatingScheduler:
    def step_from_to(self, model_output, timestep, next_timestep, sample):
        return sample + model_output


def _predict(visual_dit, latents, audio_latents, y, context, *args):
    value = context.flatten()[0].item()
    return torch.full_like(latents, value), torch.full_like(audio_latents, value)


def _run(*, do_cfg: bool, enable_cfg_parallel: bool = False, cfg_rank: int = 0):
    stage = object.__new__(MOVADenoisingStage)
    stage.video_dit = SimpleNamespace(require_vae_embedding=False)
    stage.video_dit_2 = None
    stage._torch_compiled = False
    stage._predict = _predict
    batch = SimpleNamespace(
        paired_timesteps=torch.tensor([[1000.0, 1000.0], [500.0, 500.0]]),
        scheduler=_AccumulatingScheduler(),
        y=torch.zeros(1),
        image_latent=None,
        latents=torch.zeros(1, 2, 1, 2, 2),
        audio_latents=torch.zeros(1, 2, 4),
        prompt_embeds=[torch.full((1, 3, 4), _POS)],
        negative_prompt_embeds=[torch.full((1, 3, 4), _NEG)],
        fps=24,
        do_classifier_free_guidance=do_cfg,
        guidance_scale=_GUIDANCE,
        guidance_rescale=0.0,
        is_warmup=True,
        metrics=None,
        perf_dump_path=None,
    )
    server_args = SimpleNamespace(
        enable_torch_compile=False,
        enable_cfg_parallel=enable_cfg_parallel,
        pipeline_config=SimpleNamespace(boundary_ratio=None),
    )
    # stands in for the cfg all-reduce by adding the other rank's partial
    other_partial = (1 - _GUIDANCE) * _NEG if cfg_rank == 0 else _GUIDANCE * _POS
    with (
        patch.object(mova, "get_local_torch_device", return_value="cpu"),
        patch.object(mova, "get_classifier_free_guidance_rank", return_value=cfg_rank),
        patch.object(
            mova,
            "cfg_model_parallel_all_reduce",
            side_effect=lambda partial: partial + other_partial,
        ),
    ):
        stage.forward(batch, server_args)
    return batch


class TestMOVADenoisingLoop(unittest.TestCase):
    def _assert_stepped_by(self, batch, value):
        torch.testing.assert_close(
            batch.latents, torch.full_like(batch.latents, _STEPS * value)
        )
        torch.testing.assert_close(
            batch.audio_latents, torch.full_like(batch.audio_latents, _STEPS * value)
        )

    def test_without_cfg_steps_latents(self):
        self._assert_stepped_by(_run(do_cfg=False), _POS)

    def test_serial_cfg_steps_with_combined_prediction(self):
        self._assert_stepped_by(_run(do_cfg=True), _NEG + _GUIDANCE * (_POS - _NEG))

    def test_cfg_parallel_steps_with_combined_prediction_on_every_rank(self):
        for cfg_rank in (0, 1):
            with self.subTest(cfg_rank=cfg_rank):
                batch = _run(do_cfg=True, enable_cfg_parallel=True, cfg_rank=cfg_rank)
                self._assert_stepped_by(batch, _NEG + _GUIDANCE * (_POS - _NEG))


if __name__ == "__main__":
    unittest.main()
