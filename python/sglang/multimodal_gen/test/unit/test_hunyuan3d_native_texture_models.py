# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
from diffusers import UNet2DConditionModel

from sglang.multimodal_gen.runtime.models.dits.hunyuan3d_paint import (
    Hunyuan3DPaintUNet,
)
from sglang.multimodal_gen.runtime.models.dits.stable_diffusion import (
    StableDiffusionUNet2DConditionModel,
    StableDiffusionUNetConfig,
)


def _unet_config() -> dict:
    return {
        "sample_size": 8,
        "in_channels": 4,
        "out_channels": 4,
        "center_input_sample": False,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "down_block_types": (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        "block_out_channels": (32, 32, 32, 32),
        "layers_per_block": 2,
        "downsample_padding": 1,
        "dropout": 0.0,
        "norm_num_groups": 8,
        "norm_eps": 1e-5,
        "cross_attention_dim": 16,
        "attention_head_dim": (4, 4, 4, 4),
        "transformer_layers_per_block": 1,
        "use_linear_projection": True,
    }


class TestNativeStableDiffusionUNet(unittest.TestCase):
    def test_matches_diffusers_sd21_layout_and_forward(self):
        torch.manual_seed(0)
        raw_config = _unet_config()
        reference = UNet2DConditionModel(**raw_config).eval()
        native = StableDiffusionUNet2DConditionModel(
            StableDiffusionUNetConfig.from_dict(raw_config)
        ).eval()
        native.load_state_dict(reference.state_dict(), strict=True)

        sample = torch.randn(1, 4, 8, 8)
        timestep = torch.tensor([10])
        encoder_hidden_states = torch.randn(1, 5, 16)
        class_labels = torch.tensor([0])
        with torch.inference_mode():
            expected = reference(
                sample,
                timestep,
                encoder_hidden_states,
                class_labels=class_labels,
            ).sample
            actual = native(
                sample,
                timestep,
                encoder_hidden_states,
                class_labels=class_labels,
            ).sample
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_paint_reference_branch_and_layer_groups(self):
        config = StableDiffusionUNetConfig.from_dict(_unet_config())
        model = Hunyuan3DPaintUNet(config).eval()
        modules = dict(model.named_modules())
        self.assertTrue(all(name in modules for name in model.layer_names))

        sample = torch.randn(1, 2, 4, 8, 8)
        prompt = model.learned_text_clip_gen
        condition_cache: dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            output = model(
                sample,
                torch.tensor(10),
                prompt,
                ref_latents=torch.randn(1, 1, 4, 8, 8),
                num_in_batch=2,
                condition_embed_dict=condition_cache,
                normal_imgs=torch.randn(1, 2, 4, 8, 8),
                position_imgs=torch.randn(1, 2, 4, 8, 8),
                camera_info_gen=torch.tensor([[12, 15]]),
                camera_info_ref=torch.tensor([[0]]),
            ).sample

        self.assertEqual(output.shape, (2, 4, 8, 8))
        self.assertTrue(condition_cache)


if __name__ == "__main__":
    unittest.main()
