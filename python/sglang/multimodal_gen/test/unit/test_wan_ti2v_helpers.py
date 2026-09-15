import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import maybe_unpad_latents
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_TI2V_5B_Config
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_ti2v import (
    expand_wan_ti2v_timestep,
    prepare_wan_ti2v_latents,
    prepare_wan_ti2v_sp_inputs,
)


class TestWanTI2VHelpers(unittest.TestCase):
    def test_conditioning_mask_preserves_first_frame_only(self):
        latents = torch.arange(1 * 2 * 3 * 4 * 4).reshape(1, 2, 3, 4, 4).float()
        image_latent = torch.full((1, 2, 1, 4, 4), 0.25)
        vae = SimpleNamespace(
            encode=lambda image: SimpleNamespace(mean=image_latent),
            scaling_factor=1.0,
            shift_factor=None,
        )
        config = Wan2_2_TI2V_5B_Config()
        batch = SimpleNamespace(
            image_latent=None,
            condition_image=torch.zeros(1, 3, 1, 64, 64),
            num_frames=9,
            height=64,
            width=64,
        )
        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_ti2v.get_local_torch_device",
                return_value="cpu",
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_ti2v.get_sp_world_size",
                return_value=1,
            ),
        ):
            _, _, masks = prepare_wan_ti2v_latents(
                vae,
                latents,
                torch.bfloat16,
                torch.float32,
                batch,
                SimpleNamespace(pipeline_config=config),
            )
        self.assertEqual(masks[0].dtype, torch.bfloat16)
        self.assertEqual(masks[0].shape, (2, 3, 4, 4))
        self.assertTrue(torch.all(masks[0][:, 0] == 0))
        self.assertTrue(torch.all(masks[0][:, 1:] == 1))
        torch.testing.assert_close(batch.latents[:, :, :1], image_latent)
        torch.testing.assert_close(batch.latents[:, :, 1:], latents[:, :, 1:])

    def test_sp_mask_is_padded_before_sharding(self):
        mask = torch.ones(1, 21, 4, 4)
        mask[:, 0] = 0
        batch = SimpleNamespace(did_sp_shard_latents=True)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_ti2v.get_sp_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_ti2v.get_sp_parallel_rank",
                return_value=0,
            ),
        ):
            mask_rank0, _ = prepare_wan_ti2v_sp_inputs(None, [mask], batch)

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_ti2v.get_sp_world_size",
                return_value=2,
            ),
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_ti2v.get_sp_parallel_rank",
                return_value=1,
            ),
        ):
            mask_rank1, _ = prepare_wan_ti2v_sp_inputs(None, [mask], batch)

        self.assertEqual(mask_rank0.shape, (1, 11, 4, 4))
        self.assertEqual(mask_rank1.shape, (1, 11, 4, 4))
        self.assertTrue(torch.all(mask_rank0[:, 0] == 0))
        self.assertTrue(torch.all(mask_rank1 == 1))

    def test_expanded_timestep_uses_local_latent_shape(self):
        mask = torch.ones(1, 11, 4, 4)
        mask[:, 0] = 0
        batch = SimpleNamespace(
            raw_latent_shape=(1, 16, 21, 4, 4),
            latents=torch.empty(1, 16, 11, 4, 4),
        )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_ti2v.get_sp_parallel_rank",
            return_value=0,
        ):
            timestep = expand_wan_ti2v_timestep(
                batch,
                torch.tensor(1000.0),
                torch.float32,
                seq_len=84,
                reserved_frames_mask=mask,
                patch_size=(1, 2, 2),
            )

        self.assertEqual(timestep.shape, (1, 44))
        self.assertTrue(torch.all(timestep[:, :4] == 0))
        self.assertTrue(torch.all(timestep[:, 4:] == 1000))

    def test_video_latent_sp_padding_is_trimmed_on_time_dim(self):
        batch = SimpleNamespace(raw_latent_shape=(1, 16, 21, 4, 4))
        latents = torch.empty(1, 16, 22, 4, 4)

        trimmed = maybe_unpad_latents(latents, batch)

        self.assertEqual(trimmed.shape, (1, 16, 21, 4, 4))


if __name__ == "__main__":
    unittest.main()
