# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKL as DiffusersAutoencoderKL
from diffusers import LCMScheduler, UNet2DConditionModel

from sglang.multimodal_gen.configs.models.vaes.stable_diffusion import (
    StableDiffusionVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuan3d_paint import (
    Hunyuan3DPaintUNet,
)
from sglang.multimodal_gen.runtime.models.dits.stable_diffusion import (
    StableDiffusionUNet2DConditionModel,
    StableDiffusionUNetConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder import AutoencoderKL
from sglang.multimodal_gen.runtime.pipelines.hunyuan3d_pipeline import (
    Hunyuan3D2Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan3d.paint import (
    Hunyuan3DPaintPostprocessStage,
    Hunyuan3DPaintTexGenStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan3d.shape import (
    Hunyuan3DShapeSaveStage,
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


class TestNativeStableDiffusionVAE(unittest.TestCase):
    def test_old_diffusers_config_defaults_and_forward(self):
        raw_config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 8,
            "block_out_channels": (32, 32),
            "layers_per_block": 1,
            "act_fn": "silu",
            "norm_num_groups": 8,
            "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D"),
            "up_block_types": ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        }
        reference = DiffusersAutoencoderKL(**raw_config).eval()
        config = StableDiffusionVAEConfig()
        config.update_model_arch(raw_config)
        native = AutoencoderKL(config).eval()
        native.load_state_dict(reference.state_dict(), strict=True)

        image = torch.randn(1, 3, 8, 8)
        latent = torch.randn(1, 4, 4, 4)
        with torch.inference_mode():
            expected_posterior = reference.encode(image).latent_dist
            actual_posterior = native.encode(image).latent_dist
            expected_decoded = reference.decode(latent).sample
            actual_decoded = native.decode(latent)

        torch.testing.assert_close(
            actual_posterior.parameters,
            expected_posterior.parameters,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            actual_decoded, expected_decoded, rtol=1e-5, atol=1e-5
        )


class TestHunyuan3DWarmupOutput(unittest.TestCase):
    @staticmethod
    def _batch():
        return SimpleNamespace(
            extra={"shape_meshes": [object()]},
            is_warmup=True,
            metrics=None,
        )

    def test_shape_save_does_not_require_output_path_during_paint_warmup(self):
        batch = self._batch()
        stage = Hunyuan3DShapeSaveStage(Hunyuan3D2PipelineConfig(paint_enable=True))

        self.assertIs(stage.forward(batch, SimpleNamespace()), batch)

    def test_shape_only_warmup_returns_no_files(self):
        stage = Hunyuan3DShapeSaveStage(Hunyuan3D2PipelineConfig(paint_enable=False))

        output = stage.forward(self._batch(), SimpleNamespace())

        self.assertEqual(output.output_file_paths, [])

    def test_paint_postprocess_skips_export_during_warmup(self):
        stage = Hunyuan3DPaintPostprocessStage(Hunyuan3D2PipelineConfig())

        output = stage.forward(self._batch(), SimpleNamespace())

        self.assertEqual(output.output_file_paths, [])


class TestHunyuan3DComponentResidency(unittest.TestCase):
    def test_layerwise_texture_component_starts_on_cpu(self):
        server_args = SimpleNamespace(
            should_start_component_on_cpu=lambda component_name: (
                component_name == "paint_transformer"
            )
        )

        self.assertEqual(
            Hunyuan3D2Pipeline._component_device(server_args, "paint_transformer"),
            torch.device("cpu"),
        )


class TestHunyuan3DPaintTurboSchedule(unittest.TestCase):
    def test_uses_standard_lcm_schedule_without_custom_timesteps(self):
        stage = Hunyuan3DPaintTexGenStage.__new__(Hunyuan3DPaintTexGenStage)
        stage.config = Hunyuan3D2PipelineConfig(paint_turbo_mode=True)
        stage.scheduler = LCMScheduler(
            num_train_timesteps=1000,
            original_inference_steps=50,
        )

        timesteps = stage._timesteps(torch.device("cpu"))

        self.assertEqual(
            timesteps.tolist(),
            [989, 890, 791, 692, 593, 494, 395, 296, 197, 98],
        )
        self.assertFalse(stage.scheduler.custom_timesteps)


if __name__ == "__main__":
    unittest.main()
