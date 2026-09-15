# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import fields
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.llada_image import LLaDAImageSamplingParams
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.llada_image import (
    LLaDAImageQueryFormerModel,
    LLaDAImageSigVQModel,
    LLaDAImageTextProjectionModel,
    _LLaDAImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.conditioning import (
    format_llada_image_prompt,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import (
    get_global_server_args,
    set_global_server_args,
)


class _CaptureSPBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.skip_values = []
        self.replicated_suffixes = []

    def forward(self, hidden_states, *args, **kwargs):
        self.skip_values.append(kwargs.get("skip_sequence_parallel_override", False))
        self.replicated_suffixes.append(kwargs.get("num_replicated_suffix", 0))
        return hidden_states


class TestLLaDAImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.previous_server_args = get_global_server_args()
        except ValueError:
            cls.previous_server_args = None
        set_global_server_args(SimpleNamespace(kv_gather_degree=1, sp_split_auto=False))

    @classmethod
    def tearDownClass(cls):
        set_global_server_args(cls.previous_server_args)

    def test_serving_models_reject_gradient_checkpointing(self):
        small_config = dict(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        models = [
            _LLaDAImageTransformer2DModel(
                in_channels=4,
                dim=128,
                n_layers=0,
                n_refiner_layers=0,
                n_heads=1,
                cap_feat_dim=8,
                semantic_feat_dim=8,
            ),
            LLaDAImageQueryFormerModel(num_queries=2, **small_config),
            LLaDAImageTextProjectionModel(projection_dim=4, **small_config),
            LLaDAImageSigVQModel(
                image_size=4,
                patch_size=2,
                codebook_size=8,
                codebook_embed_dim=4,
                semantic_embed_dim=4,
                **small_config,
            ),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                with self.assertRaisesRegex(
                    ValueError, "does not support gradient checkpointing"
                ):
                    model.enable_gradient_checkpointing()

    def test_dit_supports_sglang_flash_attention_and_sdpa(self):
        backends = LLaDAImagePipelineConfig().dit_config.arch_config._supported_attention_backends
        self.assertEqual(
            backends,
            {AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
        )

    def test_dit_weight_mapping_fuses_qkv_and_swiglu_inputs(self):
        mapping = get_param_names_mapping(
            LLaDAImagePipelineConfig().dit_config.arch_config.param_names_mapping
        )

        self.assertEqual(
            mapping("layers.0.attention.to_k.weight"),
            ("layers.0.attention.to_qkv.weight", 1, 3),
        )
        self.assertEqual(
            mapping("layers.0.feed_forward.w1.weight"),
            ("layers.0.feed_forward.w13.weight", 0, 2),
        )
        self.assertEqual(
            mapping("layers.0.feed_forward.w3.weight"),
            ("layers.0.feed_forward.w13.weight", 1, 2),
        )
        self.assertEqual(
            mapping("layers.0.attention.to_q.weight_scale"),
            ("layers.0.attention.to_qkv.weight_scale", None, None),
        )
        self.assertEqual(
            mapping("layers.0.feed_forward.w1.weight_scale"),
            ("layers.0.feed_forward.w13.weight_scale", None, None),
        )

    def test_prompt_format_matches_official_pipeline(self):
        self.assertEqual(
            format_llada_image_prompt("a red car"),
            "<role>HUMAN</role> Generate an image: a red car\n"
            "<role>ASSISTANT</role>\n<IMAGE1>",
        )
        self.assertEqual(
            format_llada_image_prompt(None),
            "<role>HUMAN</role> Generate an image.\n<role>ASSISTANT</role>\n<IMAGE1>",
        )

    def test_pipeline_config_defaults_schedule_and_shape(self):
        config = LLaDAImagePipelineConfig()
        sigmas = config.prepare_sigmas(None, num_inference_steps=8)

        self.assertIsNone(config.text_encoder_mem_fraction_static)
        self.assertEqual(len(sigmas), 8)
        self.assertTrue(all(left > right for left, right in pairwise(sigmas)))
        self.assertEqual(
            config.prepare_latent_shape(
                SimpleNamespace(height=1024, width=768),
                batch_size=1,
                num_frames=1,
            ),
            (1, 128, 64, 48),
        )

    def test_timestep_stage_uses_uniform_scheduler_schedule(self):
        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=3.0,
            use_uniform_sigmas=True,
        )
        stage = TimestepPreparationStage(scheduler)
        batch = SimpleNamespace(
            scheduler=None,
            timesteps=None,
            sigmas=None,
            num_inference_steps=4,
            n_tokens=None,
            extra={},
            is_warmup=True,
            rollout=False,
        )
        server_args = SimpleNamespace(pipeline_config=LLaDAImagePipelineConfig())
        module = (
            "sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation"
        )

        with (
            patch(f"{module}.get_local_torch_device", return_value=torch.device("cpu")),
            patch(f"{module}.get_or_create_request_scheduler", return_value=scheduler),
        ):
            stage.forward(batch, server_args)

        self.assertIsNone(batch.sigmas)
        torch.testing.assert_close(
            scheduler.sigmas,
            torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0]),
        )

    def test_stochastic_scheduler_accepts_per_sample_generators(self):
        for seeds in ((11,), (11, 29)):
            with self.subTest(batch_size=len(seeds)):
                scheduler = FlowMatchEulerDiscreteScheduler(
                    shift=3.0,
                    use_uniform_sigmas=True,
                    stochastic_sampling=True,
                )
                scheduler.set_timesteps(4, device="cpu")

                sample = torch.arange(len(seeds) * 8, dtype=torch.float32).reshape(
                    len(seeds), 2, 2, 2
                )
                model_output = sample / 16
                generators = [
                    torch.Generator(device="cpu").manual_seed(seed) for seed in seeds
                ]
                reference_noise = torch.cat(
                    [
                        torch.randn(
                            (1, *sample.shape[1:]),
                            generator=torch.Generator(device="cpu").manual_seed(seed),
                            dtype=sample.dtype,
                        )
                        for seed in seeds
                    ]
                )
                current_sigma, next_sigma = scheduler.sigmas[:2]
                expected = (1.0 - next_sigma) * (
                    sample - current_sigma * model_output
                ) + next_sigma * reference_noise

                actual = scheduler.step(
                    model_output=model_output,
                    timestep=scheduler.timesteps[0],
                    sample=sample,
                    generator=generators,
                    return_dict=False,
                )[0]

                self.assertEqual(actual.shape, sample.shape)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pipeline_config_rejects_unaligned_resolution(self):
        config = LLaDAImagePipelineConfig()
        with self.assertRaisesRegex(ValueError, "height must be divisible by 16"):
            config.prepare_latent_shape(
                SimpleNamespace(height=1023, width=1024),
                batch_size=1,
                num_frames=1,
            )

    def test_service_sampling_params_use_few_step_defaults_without_vq_mode(self):
        params = LLaDAImageSamplingParams()
        field_names = {field.name for field in fields(params)}

        self.assertEqual(params.num_inference_steps, 4)
        self.assertEqual(params.guidance_scale, 1.0)
        self.assertNotIn("generation_mode", field_names)
        self.assertNotIn("vq_token_ids", field_names)

    def test_edit_skips_sigvq_refiner_for_empty_cfg_condition(self):
        model_module = "sglang.multimodal_gen.runtime.models.dits.llada_image"
        linear_module = "sglang.multimodal_gen.runtime.layers.linear"
        with (
            patch(f"{model_module}.get_tp_world_size", return_value=1),
            patch(f"{linear_module}.get_tp_group", return_value=None),
            patch(f"{linear_module}.get_group_size", return_value=1),
            patch(f"{linear_module}.get_group_rank", return_value=0),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                return_value=1,
            ),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
                return_value=SimpleNamespace(attention_backend="torch_sdpa"),
            ),
        ):
            model = _LLaDAImageTransformer2DModel(
                in_channels=4,
                dim=64,
                n_layers=1,
                n_refiner_layers=1,
                n_heads=2,
                cap_feat_dim=8,
                semantic_feat_dim=10,
                axes_dims=(8, 12, 12),
                axes_lens=(256, 32, 32),
            )

        noise_refiner = _CaptureSPBlock()
        context_refiner = _CaptureSPBlock()
        sigvq_refiner = _CaptureSPBlock()
        main_block = _CaptureSPBlock()
        model.noise_refiner = torch.nn.ModuleList([noise_refiner])
        model.context_refiner = torch.nn.ModuleList([context_refiner])
        model.sigvq_refiner = torch.nn.ModuleList([sigvq_refiner])
        model.layers = torch.nn.ModuleList([main_block])

        with (
            torch.no_grad(),
        ):
            model(
                x=[torch.randn(4, 1, 4, 4)],
                t=torch.tensor([0.5]),
                cap_feats=[torch.randn(3, 8)],
                glm_cap_feats=[torch.empty(0, 10)],
                source_latents=[torch.randn(4, 1, 4, 4)],
            )

        self.assertEqual(noise_refiner.skip_values, [False])
        self.assertEqual(context_refiner.skip_values, [False])
        self.assertEqual(sigvq_refiner.skip_values, [])
        self.assertEqual(main_block.skip_values, [False])


if __name__ == "__main__":
    unittest.main()
