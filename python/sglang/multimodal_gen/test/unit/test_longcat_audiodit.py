# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.longcat_audiodit import (
    LongCatAudioDiTPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.longcat_image import (
    LongCatImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.longcat_audiodit import (
    LongCatAudioDiTSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.longcat_audiodit import (
    LongCatAudioDiTCrossAttention,
    LongCatAudioDiTSelfAttention,
    LongCatAudioDiTTransformer,
    _apply_rotary_emb,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_audiodit_flow_match import (
    AudioDiTFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.models.vaes.longcat_audiodit_vae import (
    randn_like_with_generator,
)
from sglang.multimodal_gen.runtime.pipelines.longcat_audiodit import (
    LongCatAudioDiTPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_audiodit import (
    AudioDiTCFGPolicy,
    LongCatAudioDiTDecodingStage,
    LongCatAudioDiTDenoisingStage,
    _coerce_seed,
    _coerce_single_prompt,
    _padding_mask_if_needed,
    _project,
    _resolve_duration_frames,
    prepare_branch_latent,
    resolve_cfg_policy,
    rewrite_prompt_region,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


class TestLongCatAudioDiTRegistry(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def tearDown(self):
        _get_config_info.cache_clear()

    def test_hf_id_resolves_to_audio_dit_not_image(self):
        info = _get_config_info("meituan-longcat/LongCat-AudioDiT-1B")
        self.assertIsNotNone(info)
        self.assertIs(info.pipeline_config_cls, LongCatAudioDiTPipelineConfig)

    def test_image_id_still_resolves_to_image(self):
        info = _get_config_info("meituan-longcat/LongCat-Image")
        self.assertIsNotNone(info)
        self.assertIs(info.pipeline_config_cls, LongCatImagePipelineConfig)

    def test_non_diffusers_pattern(self):
        self.assertEqual(
            get_non_diffusers_pipeline_name(
                "/root/models/meituan-longcat/LongCat-AudioDiT-1B"
            ),
            "LongCatAudioDiTPipeline",
        )


class TestAudioDiTFlowMatchScheduler(unittest.TestCase):
    def test_ascending_timesteps_and_positive_dt(self):
        scheduler = AudioDiTFlowMatchScheduler()
        scheduler.set_timesteps(4, device="cpu")
        self.assertEqual(tuple(scheduler.timesteps.tolist()), (0.0, 0.25, 0.5, 0.75))
        self.assertEqual(tuple(scheduler.sigmas.tolist()), (0.0, 0.25, 0.5, 0.75, 1.0))

        sample = torch.zeros(1, 2, 4)
        model_output = torch.ones_like(sample)
        prev = scheduler.step(model_output, scheduler.timesteps[0], sample)[0]
        self.assertTrue(torch.allclose(prev, torch.full_like(sample, 0.25)))

    def test_index_for_timestep_falls_back_to_nearest(self):
        scheduler = AudioDiTFlowMatchScheduler()
        scheduler.set_timesteps(4, device="cpu")
        self.assertEqual(scheduler.index_for_timestep(torch.tensor(0.24)), 1)
        self.assertEqual(scheduler.index_for_timestep(scheduler.timesteps[2]), 2)


class TestAudioDiTCFGPolicy(unittest.TestCase):
    def test_build_populates_branches(self):
        policy = AudioDiTCFGPolicy(guidance_method="cfg")
        batch = MagicMock()
        batch.do_classifier_free_guidance = True
        built = policy.build(batch, {}, {"encoder_hidden_states": 1}, {"x": 2})
        self.assertEqual(len(built.branches), 2)
        self.assertTrue(built.branches[0].is_conditional)
        self.assertFalse(built.branches[1].is_conditional)

    def test_cfg_combine_matches_parent(self):
        policy = AudioDiTCFGPolicy(guidance_method="cfg")
        req = MagicMock()
        req.cfg_normalization = 0
        req.guidance_rescale = 0
        pipeline_config = MagicMock()
        pipeline_config.postprocess_cfg_noise.side_effect = lambda _, noise, __: noise
        pos = torch.tensor([2.0])
        neg = torch.tensor([0.0])
        out = policy.combine([pos, neg], req, 4.0, pipeline_config)
        self.assertTrue(torch.equal(out, torch.tensor([8.0])))

    def test_apg_combine_preserves_prompt_region_zero(self):
        policy = AudioDiTCFGPolicy(guidance_method="apg")
        batch = SimpleNamespace(
            _current_latent=torch.zeros(1, 4, 2),
            _current_t=torch.tensor(0.5),
            _audio_prompt_latent_len=1,
        )
        pred = torch.ones(1, 4, 2)
        null = torch.zeros(1, 4, 2)
        out = policy.combine([pred, null], batch, 1.0, MagicMock())
        self.assertTrue(torch.all(out[:, :1] == 0))
        self.assertEqual(out.shape, pred.shape)


class TestLongCatAudioDiTPipelineConfig(unittest.TestCase):
    def test_task_type_is_audio_gen(self):
        cfg = LongCatAudioDiTPipelineConfig()
        self.assertTrue(cfg.task_type.is_audio_gen())
        self.assertEqual(cfg.task_type.data_type(), DataType.AUDIO)
        self.assertFalse(cfg.supports_dynamic_batching())
        self.assertFalse(cfg.supports_disaggregation())

    def test_dit_config_heads_nonzero(self):
        cfg = LongCatAudioDiTPipelineConfig()
        self.assertGreater(cfg.dit_config.hidden_size, 0)
        self.assertGreater(cfg.dit_config.num_attention_heads, 0)
        self.assertEqual(
            cfg.dit_config.hidden_size // cfg.dit_config.num_attention_heads,
            64,
        )

    def test_prompt_embeds_are_tensors(self):
        cfg = LongCatAudioDiTPipelineConfig()
        batch = SimpleNamespace(
            prompt_embeds=[torch.zeros(1, 3, 8)],
            negative_prompt_embeds=[torch.ones(1, 3, 8)],
        )
        self.assertIsInstance(cfg.get_pos_prompt_embeds(batch), torch.Tensor)
        self.assertIsInstance(cfg.get_neg_prompt_embeds(batch), torch.Tensor)

    def test_cond_kwargs_accept_none_masks(self):
        cfg = LongCatAudioDiTPipelineConfig()
        batch = SimpleNamespace(
            _audio_text_condition_len=torch.tensor([3]),
            _audio_mask=None,
            _audio_cond_mask=None,
            _audio_latent_cond=torch.zeros(1, 2, 4),
            _audio_empty_latent_cond=torch.ones(1, 2, 4),
            _audio_repa_dit_layer=8,
        )
        pos = cfg.prepare_pos_cond_kwargs(batch, "cpu", None, torch.float32)
        neg = cfg.prepare_neg_cond_kwargs(batch, "cpu", None, torch.float32)
        self.assertIsNone(pos["mask"])
        self.assertIsNone(pos["cond_mask"])
        self.assertIsNone(neg["mask"])
        self.assertIsNone(neg["cond_mask"])
        self.assertEqual(pos["latent_cond"].dtype, torch.float32)
        self.assertEqual(pos["return_ith_layer"], 8)
        self.assertEqual(neg["return_ith_layer"], 8)

    def test_cond_kwargs_pass_repa_dit_layer(self):
        cfg = LongCatAudioDiTPipelineConfig()
        batch = SimpleNamespace(
            _audio_text_condition_len=torch.tensor([3]),
            _audio_mask=None,
            _audio_cond_mask=None,
            _audio_latent_cond=torch.zeros(1, 2, 4),
            _audio_empty_latent_cond=torch.ones(1, 2, 4),
            _audio_repa_dit_layer=8,
        )
        pos = cfg.prepare_pos_cond_kwargs(batch, "cpu", None, torch.float32)
        self.assertEqual(pos["return_ith_layer"], 8)

    def test_resolve_cfg_policy_prefers_batch(self):
        pipeline_cfg = LongCatAudioDiTPipelineConfig()
        per_request = AudioDiTCFGPolicy(guidance_method="apg")
        batch = SimpleNamespace(extra={"cfg_policy": per_request})
        self.assertIs(resolve_cfg_policy(batch, pipeline_cfg), per_request)
        self.assertIs(
            resolve_cfg_policy(SimpleNamespace(extra={}), pipeline_cfg),
            pipeline_cfg.cfg_policy,
        )

    def test_rewrite_prompt_region_interpolates(self):
        t = torch.tensor(0.25)
        prompt_noise = torch.ones(1, 3, 2)
        latent_cond = torch.zeros(1, 8, 2)
        latent_cond[:, :3] = 2.0
        latents = torch.randn(1, 8, 2)
        gen_region = latents[:, 3:].clone()
        original = latents.clone()
        batch = SimpleNamespace(
            _audio_prompt_latent_len=3,
            _audio_prompt_noise=prompt_noise,
            _audio_latent_cond=latent_cond,
        )
        out = rewrite_prompt_region(latents, t, batch)
        expected = prompt_noise * 0.75 + latent_cond[:, :3] * 0.25
        torch.testing.assert_close(out[:, :3], expected)
        torch.testing.assert_close(out[:, 3:], gen_region)
        torch.testing.assert_close(latents, original)
        torch.testing.assert_close(batch._current_latent, out)
        torch.testing.assert_close(batch._current_t, t)

    def test_prepare_branch_latent_zeros_uncond_prompt_region(self):
        latents = torch.ones(1, 8, 2)
        original = latents.clone()
        batch = SimpleNamespace(_audio_prompt_latent_len=3, is_cfg_negative=True)
        out = prepare_branch_latent(latents, batch)
        torch.testing.assert_close(out[:, :3], torch.zeros(1, 3, 2))
        torch.testing.assert_close(out[:, 3:], original[:, 3:])
        torch.testing.assert_close(latents, original)

        cond_batch = SimpleNamespace(_audio_prompt_latent_len=3, is_cfg_negative=False)
        cond_out = prepare_branch_latent(latents, cond_batch)
        self.assertIs(cond_out, latents)

    def test_uncond_zero_does_not_clobber_apg_stash(self):
        t = torch.tensor(0.25)
        prompt_noise = torch.ones(1, 3, 2)
        latent_cond = torch.zeros(1, 8, 2)
        latent_cond[:, :3] = 2.0
        latents = torch.randn(1, 8, 2)
        batch = SimpleNamespace(
            _audio_prompt_latent_len=3,
            _audio_prompt_noise=prompt_noise,
            _audio_latent_cond=latent_cond,
            is_cfg_negative=True,
        )
        rewritten = rewrite_prompt_region(latents, t, batch)
        zeroed = prepare_branch_latent(rewritten, batch)
        expected = prompt_noise * 0.75 + latent_cond[:, :3] * 0.25
        torch.testing.assert_close(zeroed[:, :3], torch.zeros(1, 3, 2))
        torch.testing.assert_close(batch._current_latent[:, :3], expected)
        torch.testing.assert_close(batch._current_latent, rewritten)

    def test_pipeline_wires_model_specific_denoising_stage(self):
        import inspect

        src = inspect.getsource(LongCatAudioDiTPipeline.create_pipeline_stages)
        self.assertNotIn("add_standard_denoising_stage", src)
        self.assertIn("LongCatAudioDiTDenoisingStage", src)
        self.assertTrue(issubclass(LongCatAudioDiTDenoisingStage, DenoisingStage))
        self.assertIs(
            LongCatAudioDiTDenoisingStage._run_denoising_step,
            DenoisingStage._run_denoising_step,
        )

    def test_decoding_stage_permutes_3d_latent(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        class _Vae:
            def decode(self, latent):
                return latent.new_zeros(latent.shape[0], 1, latent.shape[-1] * 2)

        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(SimpleNamespace())
            stage = LongCatAudioDiTDecodingStage(
                vae=_Vae(),
                model=SimpleNamespace(config=SimpleNamespace(sampling_rate=24000)),
            )
            batch = SimpleNamespace(
                latents=torch.zeros(1, 4, 8),
                metrics=None,
            )
            out = stage.forward(batch, SimpleNamespace())
        finally:
            set_global_server_args(prev_args)
        self.assertEqual(len(out.output), 1)
        self.assertEqual(tuple(out.output[0].shape), (1, 8))
        self.assertIsNone(out.audio)
        self.assertEqual(out.audio_sample_rate, 24000)


class TestLongCatAudioDiTSamplingAdjust(unittest.TestCase):
    def test_adjust_assigns_wav_output_path(self):
        params = LongCatAudioDiTSamplingParams(prompt="hello audio")
        server_args = SimpleNamespace(
            pipeline_config=LongCatAudioDiTPipelineConfig(),
            output_path="outputs/",
            comfyui_mode=False,
        )
        params._adjust(server_args)
        self.assertEqual(params.data_type, DataType.AUDIO)
        self.assertEqual(params.output_path, "outputs/")
        self.assertIsNotNone(params.output_file_name)
        self.assertTrue(params.output_file_name.endswith(".wav"))
        self.assertEqual(
            params.output_file_path(),
            os.path.join("outputs/", params.output_file_name),
        )


class TestLongCatAudioDiTCli(unittest.TestCase):
    def test_audio_fields_are_not_on_base_sampling_params(self):
        import dataclasses

        base_fields = {f.name for f in dataclasses.fields(SamplingParams)}
        audio_fields = {
            f.name for f in dataclasses.fields(LongCatAudioDiTSamplingParams)
        }
        for name in (
            "prompt_audio_path",
            "prompt_text",
            "guidance_method",
            "duration_seconds",
        ):
            self.assertNotIn(name, base_fields)
            self.assertIn(name, audio_fields)

    def test_audio_cli_flags_are_on_base_parser(self):
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--prompt-audio-path",
                "/ref.wav",
                "--prompt-text",
                "ref",
                "--guidance-method",
                "apg",
                "--duration-seconds",
                "5.5",
            ]
        )
        self.assertEqual(args.prompt_audio_path, "/ref.wav")
        self.assertEqual(args.prompt_text, "ref")
        self.assertEqual(args.guidance_method, "apg")
        self.assertEqual(args.duration_seconds, 5.5)

    def test_audio_cli_flags_are_not_unknown_args(self):
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        _, unknown = parser.parse_known_args(
            ["--prompt-audio-path", "/ref.wav", "--ulysses-degree", "2"]
        )
        self.assertEqual(unknown, ["--ulysses-degree", "2"])

    def test_rejects_non_positive_duration_seconds(self):
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            LongCatAudioDiTSamplingParams(prompt="hi", duration_seconds=0)
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            LongCatAudioDiTSamplingParams(prompt="hi", duration_seconds=-1)


class TestLongCatAudioDiTHelpers(unittest.TestCase):
    def test_coerce_single_prompt_rejects_batch(self):
        self.assertEqual(_coerce_single_prompt("hello"), "hello")
        self.assertEqual(_coerce_single_prompt(["hello"]), "hello")
        with self.assertRaises(ValueError):
            _coerce_single_prompt(["a", "b"])

    def test_coerce_seed_accepts_int_or_list(self):
        self.assertEqual(_coerce_seed(42), 42)
        self.assertEqual(_coerce_seed([7]), 7)
        self.assertEqual(_coerce_seed([1, 2]), 1)
        with self.assertRaises(ValueError):
            _coerce_seed([])
        gen = torch.Generator(device="cpu").manual_seed(_coerce_seed([9]))
        self.assertEqual(gen.initial_seed(), 9)

    def test_padding_mask_if_needed_skips_all_valid(self):
        lengths = torch.tensor([4, 4])
        self.assertIsNone(_padding_mask_if_needed(lengths, 4, has_padding=False))
        mask = _padding_mask_if_needed(lengths, 4, has_padding=True)
        self.assertTrue(torch.equal(mask, torch.ones(2, 4, dtype=torch.bool)))
        lengths = torch.tensor([2, 4])
        mask = _padding_mask_if_needed(lengths, 4, has_padding=True)
        self.assertTrue(
            torch.equal(
                mask,
                torch.tensor([[True, True, False, False], [True, True, True, True]]),
            )
        )

    def test_duration_seconds_applies_to_clone_generated_region(self):
        sr, hop, max_dur = 24000, 2048, 30.0
        prompt_dur = 10
        frames = _resolve_duration_frames(
            gen_text="ignored because duration is set",
            prompt_dur=prompt_dur,
            prompt_time=1.0,
            prompt_text="ref",
            duration_seconds=2.0,
            sr=sr,
            full_hop=hop,
            max_duration=max_dur,
        )
        gen_frames = int(2.0 * sr // hop)
        self.assertEqual(frames, gen_frames + prompt_dur)

    def test_duration_seconds_zero_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            _resolve_duration_frames(
                gen_text="",
                prompt_dur=0,
                prompt_time=0.0,
                prompt_text=None,
                duration_seconds=0.0,
                sr=24000,
                full_hop=2048,
                max_duration=30.0,
            )

    def test_randn_like_with_cpu_generator_is_reproducible(self):
        tensor = torch.zeros(2, 3)
        a = randn_like_with_generator(
            tensor, generator=torch.Generator(device="cpu").manual_seed(7)
        )
        b = randn_like_with_generator(
            tensor, generator=torch.Generator(device="cpu").manual_seed(7)
        )
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.device, tensor.device)

    def test_apg_project_keeps_input_device(self):
        v0 = torch.ones(1, 4, 2)
        v1 = torch.ones(1, 4, 2)
        parallel, orthogonal = _project(v0, v1)
        self.assertEqual(parallel.device, v0.device)
        self.assertEqual(orthogonal.device, v0.device)
        self.assertEqual(parallel.dtype, v0.dtype)

    def test_checkpoint_without_tokenizer_files_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(LongCatAudioDiTPipeline._has_local_tokenizer_files(tmp))
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            self.assertFalse(LongCatAudioDiTPipeline._has_local_tokenizer_files(tmp))
            with open(os.path.join(tmp, "tokenizer.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            self.assertTrue(LongCatAudioDiTPipeline._has_local_tokenizer_files(tmp))


class TestLongCatAudioDiTAttention(unittest.TestCase):
    def test_transformer_supported_attention_backends(self):
        self.assertEqual(
            LongCatAudioDiTTransformer._supported_attention_backends,
            {
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            },
        )

    def test_self_attn_skips_sequence_parallel(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(
                    attention_backend="torch_sdpa",
                    comfyui_mode=False,
                    kv_gather_degree=1,
                    sp_split_auto=False,
                )
            )
            with (
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                    return_value=1,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_sequence_parallel_world_size",
                    return_value=1,
                ),
            ):
                attn = LongCatAudioDiTSelfAttention(dim=16, heads=2, dim_head=8)
                self.assertTrue(attn.attn.skip_sequence_parallel)
        finally:
            set_global_server_args(prev_args)

    def test_rejects_sequence_parallel_server_args(self):
        cfg = LongCatAudioDiTPipelineConfig()
        cfg.validate_server_args(SimpleNamespace(ulysses_degree=1, ring_degree=1))
        with self.assertRaisesRegex(ValueError, "single-GPU"):
            cfg.validate_server_args(SimpleNamespace(ulysses_degree=2, ring_degree=1))

    def test_self_attn_matches_sdpa_with_key_padding(self):
        import torch.nn.functional as F

        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(
                    attention_backend="torch_sdpa",
                    comfyui_mode=False,
                    kv_gather_degree=1,
                    sp_split_auto=False,
                )
            )
            torch.manual_seed(0)
            with (
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                    return_value=1,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_sequence_parallel_world_size",
                    return_value=1,
                ),
            ):
                attn = LongCatAudioDiTSelfAttention(
                    dim=16, heads=2, dim_head=8, qk_norm=True
                )
                attn.eval()
                x = torch.randn(2, 5, 16)
                mask = torch.tensor(
                    [
                        [True, True, True, False, False],
                        [True, True, False, False, False],
                    ]
                )
                cos = torch.randn(5, 8)
                sin = torch.randn(5, 8)
                rope = (cos, sin)

                with set_forward_context(current_timestep=0, attn_metadata=None):
                    actual = attn(x, mask=mask, rope=rope)

            query = attn.q_norm(attn.to_q(x))
            key = attn.k_norm(attn.to_k(x))
            value = attn.to_v(x)
            query = query.view(2, 5, 2, 8).transpose(1, 2)
            key = key.view(2, 5, 2, 8).transpose(1, 2)
            value = value.view(2, 5, 2, 8).transpose(1, 2)
            query = _apply_rotary_emb(query, rope)
            key = _apply_rotary_emb(key, rope)
            sdpa_mask = mask[:, None, None, :].expand(2, 2, 5, 5)
            expected = F.scaled_dot_product_attention(
                query, key, value, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
            )
            expected = expected.transpose(1, 2).reshape(2, 5, 16)
            expected = attn.to_out[0](expected)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        finally:
            set_global_server_args(prev_args)

    def test_cross_attn_matches_sdpa_with_text_padding(self):
        import torch.nn.functional as F

        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(
                    attention_backend="torch_sdpa",
                    comfyui_mode=False,
                    kv_gather_degree=1,
                    sp_split_auto=False,
                )
            )
            torch.manual_seed(1)
            with (
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                    return_value=1,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_sequence_parallel_world_size",
                    return_value=1,
                ),
            ):
                attn = LongCatAudioDiTCrossAttention(
                    q_dim=16, kv_dim=16, heads=2, dim_head=8, qk_norm=True
                )
                attn.eval()
                x = torch.randn(1, 4, 16)
                cond = torch.randn(1, 6, 16)
                mask = torch.tensor([[True, True, True, False]])
                cond_mask = torch.tensor([[True, True, True, True, False, False]])

                with set_forward_context(current_timestep=0, attn_metadata=None):
                    actual = attn(x, cond, mask=mask, cond_mask=cond_mask)

            query = attn.q_norm(attn.to_q(x)).view(1, 4, 2, 8).transpose(1, 2)
            key = attn.k_norm(attn.to_k(cond)).view(1, 6, 2, 8).transpose(1, 2)
            value = attn.to_v(cond).view(1, 6, 2, 8).transpose(1, 2)
            sdpa_mask = cond_mask[:, None, None, :].expand(1, 2, 4, 6)
            expected = F.scaled_dot_product_attention(
                query, key, value, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
            )
            expected = expected.transpose(1, 2).reshape(1, 4, 16)
            expected = attn.to_out[0](expected)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        finally:
            set_global_server_args(prev_args)

    def test_cross_attn_matches_sdpa_with_rope(self):
        import torch.nn.functional as F

        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(
                    attention_backend="torch_sdpa",
                    comfyui_mode=False,
                    kv_gather_degree=1,
                    sp_split_auto=False,
                )
            )
            torch.manual_seed(2)
            with (
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                    return_value=1,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_sequence_parallel_world_size",
                    return_value=1,
                ),
            ):
                attn = LongCatAudioDiTCrossAttention(
                    q_dim=16, kv_dim=16, heads=2, dim_head=8, qk_norm=True
                )
                attn.eval()
                x = torch.randn(1, 4, 16)
                cond = torch.randn(1, 6, 16)
                mask = torch.tensor([[True, True, True, False]])
                cond_mask = torch.tensor([[True, True, True, True, False, False]])
                rope = (torch.randn(4, 8), torch.randn(4, 8))
                cond_rope = (torch.randn(6, 8), torch.randn(6, 8))

                with set_forward_context(current_timestep=0, attn_metadata=None):
                    actual = attn(
                        x,
                        cond,
                        mask=mask,
                        cond_mask=cond_mask,
                        rope=rope,
                        cond_rope=cond_rope,
                    )

            query = attn.q_norm(attn.to_q(x)).view(1, 4, 2, 8).transpose(1, 2)
            key = attn.k_norm(attn.to_k(cond)).view(1, 6, 2, 8).transpose(1, 2)
            value = attn.to_v(cond).view(1, 6, 2, 8).transpose(1, 2)
            query = _apply_rotary_emb(query, rope)
            key = _apply_rotary_emb(key, cond_rope)
            sdpa_mask = cond_mask[:, None, None, :].expand(1, 2, 4, 6)
            expected = F.scaled_dot_product_attention(
                query, key, value, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
            )
            expected = expected.transpose(1, 2).reshape(1, 4, 16)
            expected = attn.to_out[0](expected)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        finally:
            set_global_server_args(prev_args)


if __name__ == "__main__":
    unittest.main()
