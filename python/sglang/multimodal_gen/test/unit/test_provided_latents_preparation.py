import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)


class _PackingPipelineConfig:
    """Minimal stand-in for a packed model (FLUX-style).

    ``maybe_prepare_latent_ids`` / ``maybe_pack_latents`` both expect unpacked
    ``(B, C, H, W)`` latents, matching the real configs. ``maybe_pack_latents``
    reshapes off the spec dims rather than validating the input layout, which is
    what makes packing an already-packed tensor silently wrong instead of loud --
    the real ``_pack_latents`` does a ``view`` for the same reason.
    """

    channels = 4
    height = 8
    width = 8

    def __init__(self):
        self.pack_calls = 0

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        return (batch_size, self.channels, self.height, self.width)

    def get_latent_dtype(self, dtype):
        return torch.float32

    def maybe_prepare_latent_ids(self, latents):
        batch = latents.shape[0]
        return torch.zeros(batch, self.height * self.width, 4)

    def maybe_pack_latents(self, latents, batch_size, batch):
        self.pack_calls += 1
        return latents.reshape(
            batch_size, self.channels, self.height * self.width
        ).permute(0, 2, 1)


def _make_stage(latents):
    stage = object.__new__(LatentPreparationStage)
    # No init_noise_sigma -> the scale step is skipped, keeping the assertions on
    # shape/latent_ids rather than values.
    stage.scheduler = SimpleNamespace()
    batch = SimpleNamespace(
        batch_size=1,
        generator=None,
        latents=latents,
        height=16,
        width=16,
        num_frames=1,
        latent_ids=None,
        raw_latent_shape=None,
        num_outputs_per_prompt=1,
        prompt_embeds=[torch.zeros(1, 1, dtype=torch.float32)],
    )
    pipeline_config = _PackingPipelineConfig()
    server_args = SimpleNamespace(pipeline_config=pipeline_config)
    return stage, batch, server_args, pipeline_config


class TestProvidedLatentsPreparation(unittest.TestCase):
    """Guard packing and latent IDs for caller-supplied initial latents."""

    def test_provided_latents_get_latent_ids_and_packing(self):
        provided = torch.zeros(1, 4, 8, 8)
        stage, batch, server_args, pipeline_config = _make_stage(provided)

        stage.get_forward_latent_num_frames = lambda batch, server_args: 1

        result = stage.forward(batch, server_args)

        self.assertIsNotNone(result.latent_ids)
        self.assertEqual(tuple(result.latent_ids.shape), (1, 64, 4))
        self.assertEqual(pipeline_config.pack_calls, 1)
        self.assertEqual(tuple(result.latents.shape), (1, 64, 4))

    def test_already_packed_latents_are_left_untouched(self):
        """Do not pack ComfyUI-style flat latents twice."""
        packed = torch.arange(1 * 16 * 4, dtype=torch.float32).reshape(1, 16, 4)
        stage, batch, server_args, pipeline_config = _make_stage(packed.clone())

        stage.get_forward_latent_num_frames = lambda batch, server_args: 1

        result = stage.forward(batch, server_args)

        self.assertEqual(pipeline_config.pack_calls, 0)
        self.assertIsNone(result.latent_ids)
        self.assertTrue(torch.equal(result.latents, packed))

    def test_randn_and_provided_latents_agree_on_shapes(self):
        """Leave generated and unpacked provided latents in the same shape."""
        stage_r, batch_r, args_r, _ = _make_stage(None)
        stage_r.get_forward_latent_num_frames = lambda batch, server_args: 1
        drawn = stage_r.forward(batch_r, args_r)

        stage_p, batch_p, args_p, _ = _make_stage(torch.zeros(1, 4, 8, 8))
        stage_p.get_forward_latent_num_frames = lambda batch, server_args: 1
        injected = stage_p.forward(batch_p, args_p)

        self.assertEqual(drawn.latents.shape, injected.latents.shape)
        self.assertEqual(drawn.latent_ids.shape, injected.latent_ids.shape)
        self.assertEqual(drawn.raw_latent_shape, injected.raw_latent_shape)


if __name__ == "__main__":
    unittest.main()
