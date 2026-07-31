# SPDX-License-Identifier: Apache-2.0

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image
from torch import nn

from sglang.multimodal_gen.configs.pipeline_configs.bagel import (
    BagelEditPipelineConfig,
    BagelPipelineConfig,
    BagelThinkingPipelineConfig,
    BagelUnderstandingPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.sample.bagel import (
    BagelEditSamplingParams,
    BagelSamplingParams,
    BagelThinkingSamplingParams,
    BagelUnderstandingSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage


class TestBagelSamplingParams(unittest.TestCase):
    def test_official_t2i_defaults(self) -> None:
        params = BagelSamplingParams(prompt="a cat")

        self.assertEqual((params.width, params.height), (1024, 1024))
        self.assertEqual(params.num_inference_steps, 50)
        self.assertEqual(params.guidance_scale, 4.0)
        self.assertEqual(params.flow_shift, 3.0)
        self.assertIsNone(params.negative_prompt)
        self.assertEqual(params.num_outputs_per_prompt, 1)
        self.assertFalse(params.enable_taylorseer)

    def test_editing_reuses_standard_true_cfg_field(self) -> None:
        params = BagelEditSamplingParams(prompt="make the sky blue")
        self.assertIsNone(params.true_cfg_scale)
        self.assertEqual(params.guidance_scale, 4.0)

    def test_thinking_defaults_are_greedy_and_load_language_head(self) -> None:
        params = BagelThinkingSamplingParams(prompt="a cat")
        config = BagelThinkingPipelineConfig()

        self.assertEqual(params.max_think_tokens, 1000)
        self.assertFalse(params.think_do_sample)
        self.assertEqual(params.think_temperature, 0.3)
        self.assertEqual(params.guidance_scale, 4.0)
        self.assertTrue(config.dit_config.load_lm_head)
        self.assertEqual(config.thinking_image_guidance_scale, 1.5)

    def test_thinking_sampling_controls_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_think_tokens"):
            BagelThinkingSamplingParams(prompt="a cat", max_think_tokens=0)
        with self.assertRaisesRegex(ValueError, "think_temperature"):
            BagelThinkingSamplingParams(
                prompt="a cat", think_do_sample=True, think_temperature=0.0
            )

    def test_thinking_controls_are_available_to_offline_cli(self) -> None:
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--max-think-tokens",
                "12",
                "--think-do-sample",
                "--think-temperature",
                "0.8",
            ]
        )

        cli_args = BagelThinkingSamplingParams.get_cli_args(args)

        self.assertEqual(cli_args["max_think_tokens"], 12)
        self.assertTrue(cli_args["think_do_sample"])
        self.assertEqual(cli_args["think_temperature"], 0.8)

    def test_taylorseer_is_available_to_image_generators_and_cli(self) -> None:
        self.assertTrue(
            BagelSamplingParams(
                prompt="a cat", enable_taylorseer=True
            ).enable_taylorseer
        )
        self.assertTrue(
            BagelThinkingSamplingParams(
                prompt="a cat", enable_taylorseer=True
            ).enable_taylorseer
        )
        self.assertTrue(
            BagelEditSamplingParams(
                prompt="a cat", enable_taylorseer=True
            ).enable_taylorseer
        )
        with self.assertRaisesRegex(ValueError, "does not run image denoising"):
            BagelUnderstandingSamplingParams(prompt="describe", enable_taylorseer=True)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            BagelSamplingParams(prompt="a cat", enable_taylorseer=1)

        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        args = parser.parse_args(["--enable-taylorseer"])
        cli_args = BagelSamplingParams.get_cli_args(args)
        self.assertTrue(cli_args["enable_taylorseer"])


class TestBagelPipelineConfig(unittest.TestCase):
    def test_internal_cfg_and_capability_defaults(self) -> None:
        config = BagelPipelineConfig()

        self.assertFalse(config.should_use_guidance)
        self.assertTrue(config.supports_dynamic_batching())
        self.assertFalse(config.vae_tiling)
        self.assertFalse(config.vae_sp)
        self.assertEqual(config.generator_device, "cpu")
        deployment = config.get_model_deployment_config()
        self.assertFalse(deployment.auto_enable_cfg_parallel)
        self.assertEqual(deployment.keep_resident_components, ("dit", "vae"))
        self.assertEqual(deployment.implicit_auxiliary_layerwise_offload_components, ())

    def test_only_baseline_t2i_supports_dynamic_batching(self) -> None:
        self.assertTrue(BagelPipelineConfig().supports_dynamic_batching())
        for config in (
            BagelThinkingPipelineConfig(),
            BagelEditPipelineConfig(),
            BagelUnderstandingPipelineConfig(),
        ):
            with self.subTest(config=type(config).__name__):
                self.assertFalse(config.supports_dynamic_batching())

    def test_editing_config_is_explicit_and_keeps_t2i_decoder_only(self) -> None:
        t2i = BagelPipelineConfig()
        editing = BagelEditPipelineConfig()

        self.assertEqual(t2i.task_type, ModelTaskType.T2I)
        self.assertFalse(t2i.vae_config.load_encoder)
        self.assertEqual(editing.task_type, ModelTaskType.I2I)
        self.assertTrue(editing.vae_config.load_encoder)
        self.assertEqual(editing.image_encoder_precision, "bf16")
        self.assertEqual(
            editing.get_model_deployment_config().keep_resident_components,
            ("dit", "vae", "image_encoder"),
        )
        self.assertEqual(
            editing.get_model_deployment_config().implicit_auxiliary_layerwise_offload_components,
            (),
        )

    def test_editing_resize_matches_official_transform(self) -> None:
        config = BagelEditPipelineConfig()
        image = Image.new("RGB", (400, 800))

        width, height = config.calculate_condition_image_size(image, 400, 800)
        resized, size = config.preprocess_condition_image(image, width, height, None)

        self.assertEqual((width, height), (512, 1024))
        self.assertEqual(resized.size, (512, 1024))
        self.assertEqual(size, (512, 1024))

    def test_editing_transparency_is_composited_on_white(self) -> None:
        config = BagelEditPipelineConfig()
        image = Image.new("RGBA", (1, 1), (10, 20, 30, 0))

        converted = config.condition_image_convert_method(image)

        self.assertEqual(converted.mode, "RGB")
        self.assertEqual(converted.getpixel((0, 0)), (255, 255, 255))

    def test_decode_scale_and_shift_owned_by_standard_decoding(self) -> None:
        config = BagelPipelineConfig()

        scale, shift = config.get_decode_scale_and_shift(
            torch.device("cpu"), torch.float32, SimpleNamespace()
        )

        self.assertAlmostEqual(scale, 0.3611)
        self.assertAlmostEqual(shift, 0.1159)

    def test_standard_decoding_applies_scale_shift_and_output_mapping_once(
        self,
    ) -> None:
        class IdentityVAE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.seen_latents: torch.Tensor | None = None

            def decode(self, latents: torch.Tensor) -> torch.Tensor:
                self.seen_latents = latents.clone()
                return latents

        config = BagelPipelineConfig(vae_precision="fp32")
        server_args = SimpleNamespace(
            pipeline_config=config,
            disable_autocast=True,
            enable_torch_compile=False,
        )
        vae = IdentityVAE()
        stage = DecodingStage(vae)
        latents = torch.tensor([[[[-0.25, 0.25]]]], dtype=torch.float32)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding."
            "get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            decoded = stage.decode(latents, server_args, vae_dtype=torch.float32)

        expected_vae_input = latents / 0.3611 + 0.1159
        assert vae.seen_latents is not None
        torch.testing.assert_close(vae.seen_latents, expected_vae_input)
        torch.testing.assert_close(decoded, (expected_vae_input / 2 + 0.5).clamp(0, 1))

    def test_prepare_kwargs_passes_request_context_and_guidance(self) -> None:
        config = BagelPipelineConfig()
        context = object()
        taylorseer_context = object()
        batch = SimpleNamespace(
            guidance_scale=6.5,
            extra={
                "bagel_context": context,
                "bagel_taylorseer_context": taylorseer_context,
            },
        )

        kwargs = config.prepare_pos_cond_kwargs(
            batch,
            torch.device("cpu"),
            rotary_emb=None,
            dtype=torch.float32,
        )

        self.assertIs(kwargs["bagel_context"], context)
        self.assertIs(kwargs["taylorseer_context"], taylorseer_context)
        self.assertEqual(kwargs["guidance_scale"], 6.5)
        self.assertEqual(kwargs["cfg_interval"], (0.4, 1.0))
        self.assertEqual(kwargs["cfg_renorm_type"], "global")

    def test_editing_prepare_kwargs_selects_three_way_defaults(self) -> None:
        config = BagelEditPipelineConfig()
        context = SimpleNamespace(is_editing=True)
        batch = SimpleNamespace(
            guidance_scale=4.0,
            true_cfg_scale=None,
            extra={"bagel_context": context},
        )

        kwargs = config.prepare_pos_cond_kwargs(
            batch,
            torch.device("cpu"),
            rotary_emb=None,
            dtype=torch.float32,
        )

        self.assertEqual(kwargs["guidance_scale"], 4.0)
        self.assertEqual(kwargs["image_guidance_scale"], 2.0)
        self.assertEqual(kwargs["cfg_interval"], (0.0, 1.0))
        self.assertEqual(kwargs["cfg_renorm_type"], "text_channel")

    def test_thinking_prepare_kwargs_selects_official_three_way_defaults(self) -> None:
        config = BagelThinkingPipelineConfig()
        context = SimpleNamespace(is_thinking=True)
        batch = SimpleNamespace(
            guidance_scale=4.0,
            extra={"bagel_context": context},
        )

        kwargs = config.prepare_pos_cond_kwargs(
            batch,
            torch.device("cpu"),
            rotary_emb=None,
            dtype=torch.float32,
        )

        self.assertEqual(kwargs["guidance_scale"], 4.0)
        self.assertEqual(kwargs["image_guidance_scale"], 1.5)
        self.assertEqual(kwargs["cfg_interval"], (0.4, 1.0))
        self.assertEqual(kwargs["cfg_renorm_type"], "global")

    def test_unpatchify_uses_request_shape_and_releases_context(self) -> None:
        config = BagelPipelineConfig()
        taylorseer_context = SimpleNamespace(release=Mock())
        batch = SimpleNamespace(
            height=32,
            width=48,
            extra={
                "bagel_context": SimpleNamespace(batch_size=1),
                "bagel_taylorseer_context": taylorseer_context,
            },
        )
        # 32/16 * 48/16 = 6 tokens; every token holds 2*2*16 values.
        tokens = torch.arange(6 * 64, dtype=torch.float32).reshape(6, 64)

        latents = config.post_denoising_loop(tokens, batch)

        self.assertEqual(tuple(latents.shape), (1, 16, 4, 6))
        self.assertNotIn("bagel_context", batch.extra)
        self.assertNotIn("bagel_taylorseer_context", batch.extra)
        taylorseer_context.release.assert_called_once_with()
        # Compare against the explicit official patch permutation.
        expected = torch.einsum(
            "nhwpqc->nchpwq", tokens.reshape(1, 2, 3, 2, 2, 16)
        ).reshape(1, 16, 4, 6)
        torch.testing.assert_close(latents, expected)

    def test_unpatchify_supports_batched_tokens_and_preserves_batch_one(self) -> None:
        config = BagelPipelineConfig()

        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                batch = SimpleNamespace(
                    height=32,
                    width=48,
                    extra={"bagel_context": SimpleNamespace(batch_size=batch_size)},
                )
                tokens = torch.arange(batch_size * 6 * 64, dtype=torch.float32).reshape(
                    batch_size, 6, 64
                )

                latents = config.post_denoising_loop(tokens, batch)

                self.assertEqual(tuple(latents.shape), (batch_size, 16, 4, 6))
                self.assertNotIn("bagel_context", batch.extra)
                expected = torch.einsum(
                    "nhwpqc->nchpwq",
                    tokens.reshape(batch_size, 2, 3, 2, 2, 16),
                ).reshape(batch_size, 16, 4, 6)
                torch.testing.assert_close(latents, expected)

    def test_unpatchify_validates_dynamic_batch_metadata(self) -> None:
        config = BagelPipelineConfig()
        batch = SimpleNamespace(
            height=32,
            width=48,
            extra={
                "bagel_context": SimpleNamespace(batch_size=2),
                "dynamic_batch_seeds": [1],
            },
        )

        with self.assertRaisesRegex(ValueError, "dynamic request seeds"):
            config.post_denoising_loop(torch.zeros(2, 6, 64), batch)

        self.assertNotIn("bagel_context", batch.extra)

    def test_unpatchify_requires_request_context(self) -> None:
        config = BagelPipelineConfig()
        batch = SimpleNamespace(height=32, width=48, extra={})

        with self.assertRaisesRegex(RuntimeError, "request context is missing"):
            config.post_denoising_loop(torch.zeros(6, 64), batch)

    def test_unpatchify_releases_context_on_shape_error(self) -> None:
        config = BagelPipelineConfig()
        taylorseer_context = SimpleNamespace(release=Mock())
        batch = SimpleNamespace(
            height=32,
            width=32,
            extra={
                "bagel_context": SimpleNamespace(batch_size=1),
                "bagel_taylorseer_context": taylorseer_context,
            },
        )

        with self.assertRaisesRegex(ValueError, "latent shape"):
            config.post_denoising_loop(torch.zeros(3, 64), batch)

        self.assertNotIn("bagel_context", batch.extra)
        self.assertNotIn("bagel_taylorseer_context", batch.extra)
        taylorseer_context.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
