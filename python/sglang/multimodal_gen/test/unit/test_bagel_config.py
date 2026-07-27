# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.multimodal_gen.configs.pipeline_configs.bagel import BagelPipelineConfig
from sglang.multimodal_gen.configs.sample.bagel import BagelSamplingParams
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


class TestBagelPipelineConfig(unittest.TestCase):
    def test_internal_cfg_and_capability_defaults(self) -> None:
        config = BagelPipelineConfig()

        self.assertFalse(config.should_use_guidance)
        self.assertFalse(config.supports_dynamic_batching())
        self.assertFalse(config.vae_tiling)
        self.assertFalse(config.vae_sp)
        self.assertEqual(config.generator_device, "cpu")
        deployment = config.get_model_deployment_config()
        self.assertFalse(deployment.auto_enable_cfg_parallel)
        self.assertEqual(deployment.keep_resident_components, ("dit", "vae"))

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
        batch = SimpleNamespace(
            guidance_scale=6.5,
            extra={"bagel_context": context},
        )

        kwargs = config.prepare_pos_cond_kwargs(
            batch,
            torch.device("cpu"),
            rotary_emb=None,
            dtype=torch.float32,
        )

        self.assertIs(kwargs["bagel_context"], context)
        self.assertEqual(kwargs["guidance_scale"], 6.5)
        self.assertEqual(kwargs["cfg_interval"], (0.4, 1.0))
        self.assertEqual(kwargs["cfg_renorm_type"], "global")

    def test_unpatchify_uses_request_shape_and_releases_context(self) -> None:
        config = BagelPipelineConfig()
        batch = SimpleNamespace(
            height=32,
            width=48,
            extra={"bagel_context": object()},
        )
        # 32/16 * 48/16 = 6 tokens; every token holds 2*2*16 values.
        tokens = torch.arange(6 * 64, dtype=torch.float32).reshape(6, 64)

        latents = config.post_denoising_loop(tokens, batch)

        self.assertEqual(tuple(latents.shape), (1, 16, 4, 6))
        self.assertNotIn("bagel_context", batch.extra)
        # Compare against the explicit official patch permutation.
        expected = torch.einsum(
            "nhwpqc->nchpwq", tokens.reshape(1, 2, 3, 2, 2, 16)
        ).reshape(1, 16, 4, 6)
        torch.testing.assert_close(latents, expected)

    def test_unpatchify_releases_context_on_shape_error(self) -> None:
        config = BagelPipelineConfig()
        batch = SimpleNamespace(
            height=32,
            width=32,
            extra={"bagel_context": object()},
        )

        with self.assertRaisesRegex(ValueError, "latent shape"):
            config.post_denoising_loop(torch.zeros(3, 64), batch)

        self.assertNotIn("bagel_context", batch.extra)


if __name__ == "__main__":
    unittest.main()
