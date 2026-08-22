# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)


class TestLLaDAImagePipelineConfig(unittest.TestCase):
    def setUp(self):
        self.config = LLaDAImagePipelineConfig()

    def test_edit_keeps_requested_default_output_size(self):
        image = Image.new("RGB", (768, 512))

        self.assertIsNone(
            self.config.calculate_condition_image_size(image, image.width, image.height)
        )
        self.assertIsNone(self.config.prepare_calculated_size(image))

    def test_decode_preprocessing_matches_official_vae_dtype_order(self):
        class FakeVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(
                    torch.zeros((), dtype=torch.bfloat16), requires_grad=False
                )
                self.bn = SimpleNamespace(
                    running_mean=torch.tensor(
                        [0.01, -0.02, 0.03, -0.04], dtype=torch.float32
                    ),
                    running_var=torch.full((4,), 0.0129973, dtype=torch.float32),
                )
                self.config = SimpleNamespace(
                    arch_config=SimpleNamespace(batch_norm_eps=0.003)
                )

        vae = FakeVAE()
        latents = torch.tensor(
            [[[[0.501]], [[-0.249]], [[0.126]], [[-0.751]]]],
            dtype=torch.float32,
        )
        official_latents = latents.to(torch.bfloat16)
        latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(official_latents)
        latent_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.arch_config.batch_norm_eps
        ).to(official_latents)
        expected = official_latents * latent_std + latent_mean
        expected = expected.reshape(1, 1, 2, 2, 1, 1)
        expected = expected.permute(0, 1, 4, 2, 5, 3).reshape(1, 1, 2, 2)

        actual = self.config.preprocess_decoding(latents, vae=vae)

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_edit_sp_shards_target_and_source_latents(self):
        latents = torch.randn(1, 128, 4, 4)
        source_latents = torch.randn(128, 1, 4, 4)
        batch = SimpleNamespace(
            batch_size=1,
            condition_image=Image.new("RGB", (64, 64)),
            image_embeds=None,
            source_latents=[source_latents],
            enable_sequence_shard=False,
        )
        base_module = "sglang.multimodal_gen.configs.pipeline_configs.base"
        llada_module = "sglang.multimodal_gen.configs.pipeline_configs.llada_image"
        with (
            patch(f"{base_module}.get_sp_world_size", return_value=2),
            patch(f"{base_module}.get_sp_parallel_rank", return_value=0),
            patch(f"{llada_module}.get_sp_world_size", return_value=2),
        ):
            actual, did_shard = self.config.shard_latents_for_sp(batch, latents)
            batch.did_sp_shard_latents = did_shard
            source = self.config.prepare_pos_cond_kwargs(
                batch, torch.device("cpu"), rotary_emb=None, dtype=torch.float32
            )["source_latents"][0]

        self.assertEqual(actual.shape, (1, 128, 2, 4))
        torch.testing.assert_close(actual, latents[:, :, :2, :])
        self.assertTrue(did_shard)
        self.assertEqual(source.shape, (128, 1, 2, 4))
        torch.testing.assert_close(source, source_latents[:, :, :2, :])

    def test_generation_sp_rejects_latent_height_padding(self):
        latents = torch.randn(1, 128, 63, 64)
        batch = SimpleNamespace(
            condition_image=None,
            source_latents=None,
            enable_sequence_shard=False,
        )
        module = "sglang.multimodal_gen.configs.pipeline_configs.llada_image"
        with (
            patch(f"{module}.get_sp_world_size", return_value=2),
            self.assertRaisesRegex(
                ValueError, "latent height 63 must be divisible by SP degree 2"
            ),
        ):
            self.config.shard_latents_for_sp(batch, latents)

    def test_validates_supported_parallel_topology(self):
        defaults = dict(
            sp_degree=2,
            ulysses_degree=2,
            ring_degree=1,
            tp_size=1,
            dp_size=1,
            cfg_parallel_degree=1,
            num_gpus=2,
            text_encoder_cpu_offload=False,
            llada_image_max_pixel_area=None,
            llada_image_max_text_tokens=None,
            llada_image_max_total_pixel_area=None,
            is_arg_explicitly_set=lambda name: False,
            residency_mode=lambda name: "resident",
            explicit_residency_mode=lambda name: None,
            layerwise_offload_components=None,
            cpu_offload_components=None,
        )
        self.config.validate_server_args(SimpleNamespace(**defaults))

        with (
            patch.dict(os.environ, {"SGLANG_CACHE_DIT_ENABLED": "true"}),
            self.assertRaisesRegex(ValueError, "does not support"),
        ):
            self.config.validate_server_args(SimpleNamespace(**defaults))

        auto_offload = SimpleNamespace(
            **(defaults | {"text_encoder_cpu_offload": True})
        )
        self.config.validate_server_args(auto_offload)
        self.assertFalse(auto_offload.text_encoder_cpu_offload)

        class _AutoTunedArgs(SimpleNamespace):
            def residency_mode(self, name):
                components = self.layerwise_offload_components or []
                return "layerwise_offload" if name in components else "resident"

        auto_layerwise = _AutoTunedArgs(
            **{key: value for key, value in defaults.items() if key != "residency_mode"}
        )
        auto_layerwise.layerwise_offload_components = [
            "text_encoder",
            "image_encoder",
        ]
        self.config.validate_server_args(auto_layerwise)
        self.assertEqual(auto_layerwise.layerwise_offload_components, ["image_encoder"])

        invalid_cases = [
            ({"ulysses_degree": 1, "ring_degree": 2}, "ring_degree=1"),
            (
                {"sp_degree": 4, "ulysses_degree": 4, "num_gpus": 4},
                "supports only SP degrees 1 and 2",
            ),
            ({"cfg_parallel_degree": 2}, "TP=DP=CFG=1"),
            ({"sp_degree": 1, "ulysses_degree": 1}, "num_gpus == sp_degree"),
            (
                {
                    "text_encoder_cpu_offload": True,
                    "is_arg_explicitly_set": lambda name: True,
                },
                "residency manager cannot offload",
            ),
            (
                {"residency_mode": lambda name: "component_offload"},
                "embedded text encoder to remain resident",
            ),
            (
                {
                    "residency_mode": lambda name: "layerwise_offload",
                    "is_arg_explicitly_set": (
                        lambda name: name == "layerwise_offload_components"
                    ),
                },
                "embedded text encoder to remain resident",
            ),
            ({"llada_image_max_pixel_area": 0}, "must be positive"),
            ({"llada_image_max_total_pixel_area": 0}, "must be positive"),
            (
                {"llada_image_max_text_tokens": 4000},
                "at most 3584",
            ),
        ]
        for overrides, message in invalid_cases:
            args = defaults | overrides
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, message
            ):
                self.config.validate_server_args(SimpleNamespace(**args))

    def test_validates_request_sampling_params_bounds(self):
        def params(**overrides):
            values = dict(
                width=1024,
                height=1024,
                max_sequence_length=2048,
                num_outputs_per_prompt=1,
                diffusers_kwargs=None,
                enable_cache_dit=None,
                cache_dit_params=None,
                cfg_gate_step=None,
                attention_backend_override=None,
            )
            values.update(overrides)
            return SimpleNamespace(**values)

        def server(**overrides):
            values = dict(
                sp_degree=1,
                llada_image_max_pixel_area=None,
                llada_image_max_text_tokens=None,
                llada_image_max_total_pixel_area=None,
            )
            values.update(overrides)
            return SimpleNamespace(**values)

        sp1 = server()
        sp2 = server(sp_degree=2)

        self.config.validate_request_sampling_params(params(), sp1)
        self.config.validate_request_sampling_params(
            params(max_sequence_length=None), sp1
        )
        self.config.validate_request_sampling_params(
            params(width=2048, height=2048, max_sequence_length=3584), sp2
        )
        self.config.validate_request_sampling_params(
            params(num_outputs_per_prompt=10), sp1
        )
        self.config.validate_request_sampling_params(
            params(enable_cache_dit=False), sp1
        )
        self.config.validate_request_sampling_params(params(cfg_gate_step=0.5), sp1)
        self.config.validate_request_sampling_params(
            params(attention_backend_override="fa"), sp1
        )
        self.config.validate_request_sampling_params(
            params(attention_backend_override="TORCH_SDPA"), sp1
        )

        invalid_cases = [
            (params(width=0), sp1, "positive width and height"),
            (params(width=1000), sp1, "divisible by 16"),
            (params(height=1040), sp2, "divisible by 32 at SP degree 2"),
            (params(width=2064, height=2064), sp1, "exceeds the supported maximum"),
            (params(max_sequence_length=0), sp1, "positive integer"),
            (params(max_sequence_length=True), sp1, "positive integer"),
            (params(max_sequence_length="2048"), sp1, "positive integer"),
            (
                params(max_sequence_length=3585),
                sp1,
                "exceeds the embedded text encoder budget",
            ),
            (
                params(),
                server(llada_image_max_pixel_area=512 * 512),
                "exceeds the supported maximum of 262144",
            ),
            (
                params(),
                server(llada_image_max_text_tokens=1024),
                "budget of 1024 tokens",
            ),
            (
                params(width=2048, height=2048, num_outputs_per_prompt=10),
                sp1,
                "total output area",
            ),
            (
                params(num_outputs_per_prompt=2),
                server(llada_image_max_total_pixel_area=1024 * 1024),
                "total output area",
            ),
            (
                params(diffusers_kwargs={"max_sequence_length": 4096}),
                sp1,
                "exceeds the embedded text encoder budget",
            ),
            (
                params(diffusers_kwargs={"max_sequence_length": "x"}),
                sp1,
                "positive integer",
            ),
            (params(enable_cache_dit=True), sp1, "does not support enable_cache_dit"),
            (
                params(cache_dit_params={}),
                sp1,
                "does not support cache_dit_params",
            ),
            (
                params(attention_backend_override="sage_attn"),
                sp1,
                "must be fa or torch_sdpa",
            ),
            (
                params(attention_backend_override=1),
                sp1,
                "must be fa or torch_sdpa",
            ),
            (params(cfg_gate_step=-0.1), sp1, "must be between 0.0 and 1.0"),
            (params(cfg_gate_step=1.1), sp1, "must be between 0.0 and 1.0"),
            (params(cfg_gate_step=True), sp1, "must be between 0.0 and 1.0"),
            (params(cfg_gate_step=float("nan")), sp1, "must be between 0.0 and 1.0"),
            (params(cfg_gate_step=float("inf")), sp1, "must be between 0.0 and 1.0"),
            (params(cfg_gate_step=10**1000), sp1, "must be between 0.0 and 1.0"),
            (params(cfg_gate_step="0.5"), sp1, "must be between 0.0 and 1.0"),
        ]
        for sampling_params, server_args, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                self.config.validate_request_sampling_params(
                    sampling_params, server_args
                )


if __name__ == "__main__":
    unittest.main()
