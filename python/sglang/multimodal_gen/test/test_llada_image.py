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
    LLaDAImageRMSNorm,
    LLaDAImageTransformerBlock,
    _LLaDAImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    SP_STOCHASTIC_NOISE_KEY,
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.llada_image import (
    LLaDAImageLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning import (
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

    def test_llada_rmsnorm_uses_diffusers_cast_order(self):
        generator = torch.Generator().manual_seed(0)
        norm = LLaDAImageRMSNorm(17, eps=1e-5).to(torch.bfloat16)
        # Non-unit weights make the cast order observable.
        with torch.no_grad():
            norm.weight.copy_(
                torch.randn(17, generator=generator, dtype=torch.bfloat16)
            )
        hidden_states = torch.randn(
            2,
            17,
            generator=generator,
            dtype=torch.bfloat16,
        )

        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        expected = norm.weight * (
            hidden_states.float() * torch.rsqrt(variance + norm.variance_epsilon)
        ).to(hidden_states.dtype)

        self.assertEqual(norm.weight.dtype, torch.bfloat16)
        torch.testing.assert_close(
            norm.forward_native(hidden_states), expected, rtol=0, atol=0
        )
        self.assertTrue(norm.cast_x_before_out_mul)

    def test_dit_supports_sglang_flash_attention_and_sdpa(self):
        backends = (
            LLaDAImagePipelineConfig().dit_config.arch_config._supported_attention_backends
        )
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

    def test_latent_stage_records_sp_stochastic_noise_slice(self):
        scheduler = FlowMatchEulerDiscreteScheduler(stochastic_sampling=True)
        stage = LLaDAImageLatentPreparationStage(
            scheduler=scheduler,
            transformer=None,
        )
        batch = SimpleNamespace(
            latents=torch.zeros(1, 2, 4, 3),
            raw_latent_shape=torch.Size((1, 2, 4, 3)),
            extra={},
        )
        parent_forward = (
            "sglang.multimodal_gen.runtime.pipelines.llada_image."
            "LatentPreparationStage.forward"
        )
        module = "sglang.multimodal_gen.runtime.pipelines.llada_image"

        with (
            patch(parent_forward, return_value=batch),
            patch(f"{module}.get_sp_world_size", return_value=2),
            patch(f"{module}.get_sp_parallel_rank", return_value=1),
        ):
            actual = stage.forward(batch, server_args=None)

        self.assertIs(actual, batch)
        self.assertEqual(
            batch.extra[SP_STOCHASTIC_NOISE_KEY],
            {
                "full_shape": (1, 2, 4, 3),
                "dim": 2,
                "start": 2,
                "length": 2,
            },
        )

        with (
            patch(parent_forward, return_value=batch),
            patch(f"{module}.get_sp_world_size", return_value=1),
        ):
            stage.forward(batch, server_args=None)

        self.assertNotIn(SP_STOCHASTIC_NOISE_KEY, batch.extra)

        batch.extra[SP_STOCHASTIC_NOISE_KEY] = {"stale": True}
        stage.scheduler = FlowMatchEulerDiscreteScheduler(stochastic_sampling=False)
        with (
            patch(parent_forward, return_value=batch),
            patch(f"{module}.get_sp_world_size", return_value=2),
        ):
            stage.forward(batch, server_args=None)

        self.assertNotIn(SP_STOCHASTIC_NOISE_KEY, batch.extra)

    def test_stochastic_scheduler_matches_spatial_sp_slices(self):
        full_shape = (1, 2, 4, 3)

        def make_stream():
            generators = [torch.Generator(device="cpu").manual_seed(71)]
            initial = torch.randn(
                full_shape,
                generator=generators[0],
                dtype=torch.float32,
            )
            return generators, initial

        full_generators, full_sample = make_stream()
        rank0_generators, rank0_full = make_stream()
        rank1_generators, rank1_full = make_stream()
        rank0_sample = rank0_full[:, :, :2, :].contiguous()
        rank1_sample = rank1_full[:, :, 2:, :].contiguous()

        schedulers = [
            FlowMatchEulerDiscreteScheduler(
                shift=3.0,
                use_uniform_sigmas=True,
                stochastic_sampling=True,
            )
            for _ in range(3)
        ]
        for scheduler in schedulers:
            scheduler.set_timesteps(4, device="cpu")
        full_scheduler, rank0_scheduler, rank1_scheduler = schedulers

        def rank_batch(start):
            return SimpleNamespace(
                rollout=False,
                did_sp_shard_latents=True,
                raw_latent_shape=torch.Size(full_shape),
                extra={
                    SP_STOCHASTIC_NOISE_KEY: {
                        "full_shape": full_shape,
                        "dim": 2,
                        "start": start,
                        "length": 2,
                    }
                },
            )

        rank0_batch = rank_batch(0)
        rank1_batch = rank_batch(2)

        for step_index in range(4):
            full_model_output = full_sample / (step_index + 2)
            rank0_model_output = full_model_output[:, :, :2, :].contiguous()
            rank1_model_output = full_model_output[:, :, 2:, :].contiguous()

            full_sample = full_scheduler.step(
                model_output=full_model_output,
                timestep=full_scheduler.timesteps[step_index],
                sample=full_sample,
                generator=full_generators,
                return_dict=False,
            )[0]
            rank0_sample = rank0_scheduler.step(
                model_output=rank0_model_output,
                timestep=rank0_scheduler.timesteps[step_index],
                sample=rank0_sample,
                generator=rank0_generators,
                batch=rank0_batch,
                return_dict=False,
            )[0]
            rank1_sample = rank1_scheduler.step(
                model_output=rank1_model_output,
                timestep=rank1_scheduler.timesteps[step_index],
                sample=rank1_sample,
                generator=rank1_generators,
                batch=rank1_batch,
                return_dict=False,
            )[0]

            torch.testing.assert_close(
                torch.cat([rank0_sample, rank1_sample], dim=2),
                full_sample,
                rtol=0,
                atol=0,
            )
            self.assertTrue(
                torch.equal(
                    full_generators[0].get_state(),
                    rank0_generators[0].get_state(),
                )
            )
            self.assertTrue(
                torch.equal(
                    full_generators[0].get_state(),
                    rank1_generators[0].get_state(),
                )
            )

    def test_stochastic_scheduler_without_opt_in_keeps_local_draw(self):
        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=3.0,
            use_uniform_sigmas=True,
            stochastic_sampling=True,
        )
        scheduler.set_timesteps(4, device="cpu")
        sample = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
        model_output = sample / 16
        generator = [torch.Generator(device="cpu").manual_seed(23)]
        reference_noise = torch.randn(
            sample.shape,
            generator=torch.Generator(device="cpu").manual_seed(23),
        )
        current_sigma, next_sigma = scheduler.sigmas[:2]
        expected = (1.0 - next_sigma) * (
            sample - current_sigma * model_output
        ) + next_sigma * reference_noise
        batch = SimpleNamespace(
            rollout=False,
            did_sp_shard_latents=True,
            raw_latent_shape=torch.Size((1, 2, 4, 3)),
            extra={},
        )

        actual = scheduler.step(
            model_output=model_output,
            timestep=scheduler.timesteps[0],
            sample=sample,
            generator=generator,
            batch=batch,
            return_dict=False,
        )[0]

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_stochastic_scheduler_rejects_invalid_sp_metadata(self):
        full_shape = (1, 2, 4, 3)
        valid = {
            "full_shape": full_shape,
            "dim": 2,
            "start": 0,
            "length": 2,
        }
        cases = (
            ("metadata type", "bad", True, full_shape, 1),
            ("missing field", {"full_shape": full_shape}, True, full_shape, 1),
            (
                "full shape type",
                {**valid, "full_shape": list(full_shape)},
                True,
                full_shape,
                1,
            ),
            (
                "full shape ndim",
                {**valid, "full_shape": (1, 2, 4)},
                True,
                full_shape,
                1,
            ),
            (
                "full shape value",
                {**valid, "full_shape": (1, 2, 0, 3)},
                True,
                full_shape,
                1,
            ),
            ("without sharding", valid, False, full_shape, 1),
            ("batch dimension", {**valid, "dim": 0}, True, full_shape, 1),
            ("dimension type", {**valid, "dim": 2.0}, True, full_shape, 1),
            ("dimension range", {**valid, "dim": 4}, True, full_shape, 1),
            ("slice type", {**valid, "start": 0.0}, True, full_shape, 1),
            ("negative start", {**valid, "start": -1}, True, full_shape, 1),
            ("slice bounds", {**valid, "start": 3}, True, full_shape, 1),
            ("local length", {**valid, "length": 1}, True, full_shape, 1),
            ("raw shape", valid, True, (1, 2, 6, 3), 1),
            (
                "sample shape",
                {**valid, "full_shape": (1, 3, 4, 3)},
                True,
                (1, 3, 4, 3),
                1,
            ),
            ("generator count", valid, True, full_shape, 2),
        )

        for name, metadata, did_shard, raw_shape, generator_count in cases:
            with self.subTest(name=name):
                scheduler = FlowMatchEulerDiscreteScheduler(
                    shift=3.0,
                    use_uniform_sigmas=True,
                    stochastic_sampling=True,
                )
                scheduler.set_timesteps(4, device="cpu")
                sample = torch.zeros(1, 2, 2, 3)
                generators = [
                    torch.Generator(device="cpu").manual_seed(seed)
                    for seed in range(generator_count)
                ]
                batch = SimpleNamespace(
                    rollout=False,
                    did_sp_shard_latents=did_shard,
                    raw_latent_shape=torch.Size(raw_shape),
                    extra={SP_STOCHASTIC_NOISE_KEY: metadata},
                )

                with self.assertRaisesRegex(
                    ValueError, "Invalid SP stochastic noise metadata"
                ):
                    scheduler.step(
                        model_output=torch.zeros_like(sample),
                        timestep=scheduler.timesteps[0],
                        sample=sample,
                        generator=generators,
                        batch=batch,
                        return_dict=False,
                    )

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

    def test_dit_block_forwards_sequence_parallel_controls(self):
        class CaptureAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.received = None

            def forward(
                self,
                hidden_states,
                attention_mask,
                freqs_cis,
                num_replicated_suffix=0,
                skip_sequence_parallel_override=False,
            ):
                self.received = (
                    num_replicated_suffix,
                    skip_sequence_parallel_override,
                )
                return torch.zeros_like(hidden_states)

        block = object.__new__(LLaDAImageTransformerBlock)
        torch.nn.Module.__init__(block)
        block.modulation = False
        block.attention = CaptureAttention()
        block.attention_norm1 = torch.nn.Identity()
        block.attention_norm2 = torch.nn.Identity()
        block.ffn_norm1 = torch.nn.Identity()
        block.ffn_norm2 = torch.nn.Identity()
        block.feed_forward = torch.nn.Identity()

        block(
            torch.ones(1, 4, 8),
            attention_mask=None,
            freqs_cis=torch.empty(0),
            num_replicated_suffix=32,
            skip_sequence_parallel_override=True,
        )

        self.assertEqual(block.attention.received, (32, True))

    def test_edit_sp_shards_images_and_replicates_conditions(self):
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
            patch(f"{model_module}.get_sp_world_size", return_value=2),
            patch(f"{model_module}.get_sp_parallel_rank", return_value=0),
            torch.no_grad(),
        ):
            model(
                x=[torch.randn(4, 1, 4, 4)],
                t=torch.tensor([0.5]),
                cap_feats=[torch.randn(3, 8)],
                glm_cap_feats=[torch.randn(5, 10)],
                source_latents=[torch.randn(4, 1, 4, 4)],
            )

        self.assertEqual(noise_refiner.skip_values, [False])
        self.assertEqual(context_refiner.skip_values, [True])
        self.assertEqual(sigvq_refiner.skip_values, [True])
        self.assertEqual(main_block.skip_values, [False])
        self.assertEqual(main_block.replicated_suffixes, [96])

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
            patch(f"{model_module}.get_sp_world_size", return_value=1),
            patch(f"{model_module}.get_sp_parallel_rank", return_value=0),
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
        self.assertEqual(context_refiner.skip_values, [True])
        self.assertEqual(sigvq_refiner.skip_values, [])
        self.assertEqual(main_block.skip_values, [False])

    def test_sp_uses_global_height_coordinates_for_generation_and_edit(self):
        calls = []

        def pad_with_ids(
            features, grid_size, start, noise_value=None, sequence_multiple=32
        ):
            del sequence_multiple
            length = len(features)
            calls.append((grid_size, start))
            return (
                features,
                torch.zeros((length, 3), dtype=torch.int32),
                torch.zeros(length, dtype=torch.bool),
                length,
                [noise_value] * length if noise_value is not None else None,
            )

        subject = SimpleNamespace(
            _patchify_image=lambda _image, _patch, _f_patch: (
                torch.zeros(8, 1),
                (1, 2, 4),
                (1, 2, 4),
            ),
            _pad_with_ids=pad_with_ids,
        )
        module = "sglang.multimodal_gen.runtime.models.dits.llada_image"
        with (
            patch(f"{module}.get_sp_world_size", return_value=2),
            patch(f"{module}.get_sp_parallel_rank", return_value=1),
        ):
            _LLaDAImageTransformer2DModel._prepare_t2i_sequences(
                subject,
                [torch.zeros(1, 1, 2, 4)],
                cap_feats=None,
                glm_features=None,
                patch_size=1,
                f_patch_size=1,
            )

        self.assertEqual(calls[-1], ((1, 2, 4), (1, 2, 0)))

        calls.clear()
        with (
            patch(f"{module}.get_sp_world_size", return_value=2),
            patch(f"{module}.get_sp_parallel_rank", return_value=1),
        ):
            _LLaDAImageTransformer2DModel._prepare_editing_sequences(
                subject,
                [torch.zeros(1, 1, 2, 4)],
                cap_feats=[torch.zeros(1, 1)],
                glm_cap_feats=[torch.zeros(1, 1)],
                source_latents=[torch.zeros(1, 1, 2, 4)],
                patch_size=1,
                f_patch_size=1,
            )

        image_calls = [call for call in calls if call[0] == (1, 2, 4)]
        self.assertEqual(len(image_calls), 2)
        self.assertTrue(all(start[1:] == (2, 0) for _, start in image_calls))
        self.assertEqual(calls[-1], ((1, 1, 1), (35, 0, 0)))


if __name__ == "__main__":
    unittest.main()
