import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.helios_denoising import (
    HeliosChunkedDenoisingStage,
)


class _Transformer:
    def __call__(self, **kwargs):
        return torch.zeros_like(kwargs["hidden_states"])


class _Scheduler:
    def step(self, noise_pred, timestep, latents, return_dict=False):
        return (latents,)


class _Profiler:
    def __init__(self):
        self.steps = 0

    def step_denoising_step(self):
        self.steps += 1


class TestHeliosDenoisingProfiler(unittest.TestCase):
    def test_stage1_advances_profiler_once_per_timestep(self):
        stage = HeliosChunkedDenoisingStage.__new__(HeliosChunkedDenoisingStage)
        stage.transformer = _Transformer()
        stage.scheduler = _Scheduler()
        profiler = _Profiler()
        timesteps = torch.tensor([2.0, 1.0])

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.helios_denoising."
            "SGLDiffusionProfiler.get_instance",
            return_value=profiler,
        ):
            output = stage._denoise_one_chunk(
                latents=torch.ones(1, 2),
                prompt_embeds=torch.ones(1, 2),
                negative_prompt_embeds=torch.ones(1, 2),
                timesteps=timesteps,
                guidance_scale=1.0,
                indices_hidden_states=None,
                indices_latents_history_short=None,
                indices_latents_history_mid=None,
                indices_latents_history_long=None,
                latents_history_short=None,
                latents_history_mid=None,
                latents_history_long=None,
                target_dtype=torch.float32,
                device=torch.device("cpu"),
                batch=None,
                scheduler=stage.scheduler,
            )

        torch.testing.assert_close(output, torch.ones(1, 2))
        self.assertEqual(profiler.steps, len(timesteps))


if __name__ == "__main__":
    unittest.main()
