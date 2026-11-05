import unittest
from collections import OrderedDict
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.configs.sample.ltx_2 import (
    LTX2SamplingParams,
    LTX23HQSamplingParams,
    LTX23SamplingParams,
)
from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    BaseBreakableCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2AudioVideoRotaryPosEmbed,
    _ltx2_build_batched_perturbation_states,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
    _prepare_ltx2_rope_coords_for_bcg,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.unit.host_sync_guard import (
    RecordOps,
    forbid_host_syncs,
)
from sglang.test.test_utils import CustomTestCase


class TestLTX2BCGCoords(CustomTestCase):
    def setUp(self):
        self.video_result = torch.ones(1, 4)
        self.audio_result = torch.ones(1, 2)
        self.model = SimpleNamespace(
            rope=SimpleNamespace(
                prepare_video_coords=Mock(return_value=self.video_result)
            ),
            audio_rope=SimpleNamespace(
                prepare_audio_coords=Mock(return_value=self.audio_result)
            ),
        )
        self.video_latents = torch.zeros(2, 8, 16)
        self.audio_latents = torch.zeros(2, 4, 16)

    def _call(self, *, enabled, video_coords=None, audio_coords=None):
        return _prepare_ltx2_rope_coords_for_bcg(
            enabled=enabled,
            current_model=self.model,
            latent_model_input=self.video_latents,
            audio_latent_model_input=self.audio_latents,
            video_coords=video_coords,
            audio_coords=audio_coords,
            num_frames=16,
            video_start_frame=0,
            height=32,
            width=48,
            audio_num_frames=126,
            fps=24,
        )

    def test_prepares_missing_coords_before_bcg_capture(self):
        video, audio = self._call(enabled=True)

        self.assertIs(video, self.video_result)
        self.assertIs(audio, self.audio_result)
        self.model.rope.prepare_video_coords.assert_called_once_with(
            batch_size=2,
            num_frames=16,
            height=32,
            width=48,
            device=self.video_latents.device,
            fps=24,
            start_frame=0,
        )
        self.model.audio_rope.prepare_audio_coords.assert_called_once_with(
            batch_size=2,
            num_frames=126,
            device=self.audio_latents.device,
        )

    def test_disabled_bcg_keeps_legacy_none_coords(self):
        video, audio = self._call(enabled=False)

        self.assertIsNone(video)
        self.assertIsNone(audio)
        self.model.rope.prepare_video_coords.assert_not_called()
        self.model.audio_rope.prepare_audio_coords.assert_not_called()

    def test_existing_parallel_coords_are_preserved(self):
        existing_video = torch.zeros(3)
        existing_audio = torch.zeros(5)

        video, audio = self._call(
            enabled=True,
            video_coords=existing_video,
            audio_coords=existing_audio,
        )

        self.assertIs(video, existing_video)
        self.assertIs(audio, existing_audio)
        self.model.rope.prepare_video_coords.assert_not_called()
        self.model.audio_rope.prepare_audio_coords.assert_not_called()

    def test_stage_reuses_coords_across_denoising_steps(self):
        stage = object.__new__(LTX2DenoisingStage)
        stage._ltx2_coords_cache = OrderedDict()
        stage._ltx2_coords_cache_max_entries = 4
        pipeline_config = SimpleNamespace(
            prepare_video_rope_coords_for_sp=Mock(return_value=self.video_result),
            prepare_audio_rope_coords_for_sp=Mock(return_value=self.audio_result),
        )
        ctx = SimpleNamespace(
            latent_num_frames_for_model=16,
            latent_height=32,
            latent_width=48,
            use_ltx23_legacy_one_stage=False,
        )
        step = SimpleNamespace(current_model=self.model)
        batch = SimpleNamespace(
            fps=24,
            did_sp_shard_latents=False,
            did_sp_shard_audio_latents=False,
        )
        server_args = SimpleNamespace(
            pipeline_config=pipeline_config,
            enable_breakable_cuda_graph=False,
        )

        first = stage._get_ltx2_rope_coords(
            ctx,
            step,
            batch,
            server_args,
            self.video_latents,
            self.audio_latents,
            audio_num_frames_latent=126,
        )
        second = stage._get_ltx2_rope_coords(
            ctx,
            step,
            batch,
            server_args,
            self.video_latents,
            self.audio_latents,
            audio_num_frames_latent=126,
        )

        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        pipeline_config.prepare_video_rope_coords_for_sp.assert_called_once()
        pipeline_config.prepare_audio_rope_coords_for_sp.assert_called_once()

    def test_legacy_bcg_coords_keep_the_sp_shard_start_frame(self):
        rope = LTX2AudioVideoRotaryPosEmbed(dim=64)
        model = SimpleNamespace(
            rope=rope,
            audio_rope=LTX2AudioVideoRotaryPosEmbed(
                dim=64, scale_factors=(4,), modality="audio"
            ),
        )
        stage = object.__new__(LTX2DenoisingStage)
        stage._ltx2_coords_cache = OrderedDict()
        stage._ltx2_coords_cache_max_entries = 4
        local_frames = 4
        ctx = SimpleNamespace(
            latent_num_frames_for_model=local_frames,
            latent_height=2,
            latent_width=3,
            use_ltx23_legacy_one_stage=True,
        )
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(), enable_breakable_cuda_graph=True
        )
        video_latents = torch.zeros(1, local_frames * 2 * 3, 16)
        for sp_rank in (0, 1):
            with self.subTest(sp_rank=sp_rank):
                batch = SimpleNamespace(
                    fps=24,
                    did_sp_shard_latents=True,
                    sp_video_start_frame=sp_rank * local_frames,
                    did_sp_shard_audio_latents=False,
                )
                video_coords, _ = stage._get_ltx2_rope_coords(
                    ctx,
                    SimpleNamespace(current_model=model),
                    batch,
                    server_args,
                    video_latents,
                    self.audio_latents,
                    audio_num_frames_latent=126,
                )
                # the eager forward shifts each shard by sp_rank * local frames
                expected = rope.prepare_video_coords(
                    batch_size=1,
                    num_frames=local_frames,
                    height=2,
                    width=3,
                    device=video_latents.device,
                    fps=24,
                    start_frame=sp_rank * local_frames,
                )
                torch.testing.assert_close(video_coords, expected, rtol=0, atol=0)


class TestLTX2GuidanceBCG(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graphs")
    def test_mixed_perturbation_masks_are_cuda_graph_capturable(self):
        configs = (
            {"skip_video_self_attn_blocks": (2, 3)},
            {"skip_video_self_attn_blocks": (3,)},
            {"skip_video_self_attn_blocks": (2, 3)},
            {"skip_video_self_attn_blocks": (3,)},
        )
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                values = torch.empty(4, 2, 8, device="cuda", dtype=dtype)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    states = _ltx2_build_batched_perturbation_states(
                        configs, "skip_video_self_attn_blocks", (1, 2, 3), values
                    )
                self.assertEqual(states[1], (None, False))
                self.assertEqual(states[3], (None, True))
                mask, skipped = states[2]
                self.assertFalse(skipped)
                # Replay must regenerate the constant, even if its buffer changed.
                mask.fill_(7)
                graph.replay()
                expected = torch.tensor([0, 1, 0, 1], device="cuda", dtype=dtype)
                torch.testing.assert_close(mask.flatten(), expected, rtol=0, atol=0)

    def test_warmup_preserves_guider_and_hq_lora_extras(self):
        for params in (
            LTX2SamplingParams(),
            LTX23SamplingParams(video_stg_blocks=[7], audio_modality_scale=2.0),
            LTX23HQSamplingParams(distilled_lora_strength_stage_1=0.125),
            LTX23SamplingParams(rollout=True),
        ):
            with self.subTest(params=type(params).__name__, rollout=params.rollout):
                req = SimpleNamespace(extra={"server_internal_prewarm": True})
                params.prepare_synthetic_warmup_request_for_queue(req, None)
                self.assertEqual(
                    req.extra,
                    {"server_internal_prewarm": True, **params.build_request_extra()},
                )

    @staticmethod
    def _model(**kwargs):
        # Different prompt branches and requests produce different predictions.
        prompt = kwargs["encoder_hidden_states"].mean(dim=(1, 2))[:, None, None]
        return (
            kwargs["hidden_states"] * 0.25 + prompt,
            kwargs["audio_hidden_states"] * 0.5 + prompt,
        )

    def _run_step(self, *, mode, warmup, runner, value, sigmas=None):
        stage = object.__new__(LTX2DenoisingStage)
        stage.sampler_name = "euler"
        stage._attention_backend_active_override = None
        stage._skip_softmax_forced_fa = False
        stage._maybe_get_bcg_runner = Mock(return_value=runner)
        stage._ltx2_model_forward_context = lambda *_: nullcontext()
        stage.post_forward_for_ti2v_task = lambda batch, args, mask, latents, z: latents
        # the stub DiT reads the prompt itself instead of projecting cached K/V
        stage._use_cached_ltx2_prompt_kv = lambda ctx, step, kwargs: kwargs
        args = SimpleNamespace(
            enable_cfg_parallel=False,
            pipeline_class_name=(
                "LTX2TwoStageHQPipeline" if mode == "hq" else "LTX2TwoStagePipeline"
            ),
        )
        stage.server_args = args
        video = torch.arange(8, dtype=torch.float32).view(1, 2, 4) + value
        audio = torch.arange(4, dtype=torch.float32).view(1, 1, 4) + value
        if sigmas is None:
            sigmas = torch.tensor([1.0, 0.5])
        scheduler = SimpleNamespace(
            sigmas=sigmas,
            step=lambda output, t, sample, return_dict: (sample - 0.5 * output,),
        )
        ctx = SimpleNamespace(
            latents=video,
            audio_latents=audio,
            scheduler=scheduler,
            sigmas_host=torch.tensor([1.0, 0.5]),
            sigmas_device=sigmas,
            audio_scheduler=scheduler,
            stage="stage1",
            is_warmup=warmup,
            is_ltx23_variant=False,
            use_ltx23_legacy_one_stage=mode in ("legacy", "legacy_batched"),
            use_ltx23_hq_timestep_semantics=False,
            denoise_mask=None,
            clean_latent=None,
            last_denoised_video=None,
            last_denoised_audio=None,
            reserved_frames_mask=None,
            z=None,
        )
        step = SimpleNamespace(
            step_index=0, current_model=self._model, t_device=torch.tensor(1000.0)
        )
        params = (
            LTX2SamplingParams()
            if mode == "official"
            else LTX23SamplingParams(video_rescale_scale=0.0, audio_rescale_scale=0.0)
        )
        embeds = torch.full((1, 2, 4), value)
        batch = SimpleNamespace(
            quality="exact" if mode == "legacy" else "lossless",
            extra=params.build_request_extra(),
            prompt_embeds=[embeds],
            audio_prompt_embeds=[embeds],
            negative_prompt_embeds=[-embeds],
            negative_audio_prompt_embeds=[-embeds],
            prompt_attention_mask=torch.ones(1, 2),
            negative_attention_mask=torch.ones(1, 2),
            ltx2_num_image_tokens=int(mode == "ti2v"),
            do_classifier_free_guidance=True,
            guidance_scale=4.0,
            rollout_controllers={},
        )
        stage._prepare_ltx2_model_inputs = lambda *_: SimpleNamespace(
            latent_model_input=ctx.latents,
            audio_latent_model_input=ctx.audio_latents,
        )
        stage._build_ltx2_base_model_kwargs = lambda *_: {
            "hidden_states": ctx.latents,
            "audio_hidden_states": ctx.audio_latents,
        }
        stage._run_denoising_step(ctx, step, batch, args)
        return ctx.latents, ctx.audio_latents

    def test_guidance_paths_capture_only_at_warmup_and_preserve_predictions(self):
        for mode in ("batched", "ti2v", "hq", "legacy", "legacy_batched"):
            with self.subTest(mode=mode):
                runner = Mock(side_effect=self._model)
                batched = mode in ("batched", "legacy_batched")
                calls_per_step = 1 if batched else 4
                for warmup, value in ((True, 1.0), (False, 2.0)):
                    expected = self._run_step(
                        mode=mode, warmup=warmup, runner=None, value=value
                    )
                    actual = self._run_step(
                        mode=mode, warmup=warmup, runner=runner, value=value
                    )
                    for result, reference in zip(actual, expected):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)
                self.assertEqual(runner.capture.call_count, calls_per_step)
                self.assertEqual(runner.call_count, calls_per_step * 2)
                if batched:
                    kwargs = runner.call_args.kwargs
                    self.assertEqual(kwargs["hidden_states"].shape[0], 4)
                    self.assertEqual(len(kwargs["perturbation_configs"]), 4)
                    self.assertEqual(
                        kwargs["perturbation_configs"][2][
                            "skip_video_self_attn_blocks"
                        ],
                        (28,),
                    )
                    self.assertTrue(
                        kwargs["perturbation_configs"][3]["skip_a2v_cross_attn"]
                    )

    def test_steps_never_read_device_sigmas_on_host(self):
        modes = ("official", "batched", "ti2v", "hq", "legacy", "legacy_batched")
        for mode in modes:
            with self.subTest(mode=mode):
                expected = self._run_step(
                    mode=mode, warmup=False, runner=None, value=1.0
                )
                actual = self._run_step(
                    mode=mode,
                    warmup=False,
                    runner=None,
                    value=1.0,
                    sigmas=torch.tensor([1.0, 0.5]).as_subclass(_DeviceSigmas),
                )
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(
                        result.as_subclass(torch.Tensor), reference, rtol=0, atol=0
                    )

    def test_tensor_sigma_x0_matches_python_float(self):
        torch.manual_seed(0)
        sample = torch.randn(2, 6, 8, dtype=torch.bfloat16)
        velocity = torch.randn(2, 6, 8, dtype=torch.bfloat16)
        for sigma in (
            torch.tensor(0.7182, dtype=torch.float32),
            torch.tensor(0.3141592653589793, dtype=torch.float64),
        ):
            with self.subTest(dtype=sigma.dtype):
                self.assertTrue(
                    torch.equal(
                        LTX2DenoisingStage._ltx2_velocity_to_x0(
                            sample, velocity, sigma
                        ),
                        LTX2DenoisingStage._ltx2_velocity_to_x0(
                            sample, velocity, float(sigma)
                        ),
                    )
                )


class _DeviceSigmas(torch.Tensor):
    # a CUDA schedule in a CPU test: any host read of a value would sync
    def item(self):
        raise AssertionError("denoising step read a device sigma on the host")

    __float__ = __bool__ = item


class _AttentionBlock(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, q, k, v, attn_mask):
        return self.attention(q + 0.25, k, v, attn_mask=attn_mask).sin()


class TestLTX2DenseAttentionBCG(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graphs")
    @torch.no_grad()
    def test_capture_and_eager_break_replay_changed_masks(self):
        with ExitStack() as stack:
            for name, value in (
                ("get_attn_backend", SDPABackend),
                ("get_compute_dtype", torch.bfloat16),
                ("get_sequence_parallel_world_size", 1),
                ("get_ring_parallel_world_size", 1),
                ("_resolve_sp_attention_mode", ("ulysses", False)),
                ("_maybe_install_backend_autotune", None),
            ):
                stack.enter_context(patch.object(layer, name, return_value=value))
            stack.enter_context(set_forward_context(0, None))
            for cls in (layer.LocalAttention, layer.USPAttention):
                for capture_dense in (False, True):
                    with self.subTest(layer=cls.__name__, capture_dense=capture_dense):
                        attention = cls(
                            num_heads=4,
                            head_size=64,
                            is_cross_attention=True,
                            allow_cudnn_sdp=True,
                            capture_dense_attention=capture_dense,
                        )
                        model = _AttentionBlock(attention)
                        runner = BaseBreakableCudaGraphRunner(
                            model, torch.device("cuda:0")
                        )
                        kwargs = dict(
                            q=torch.randn(
                                2, 32, 4, 64, device="cuda", dtype=torch.bfloat16
                            ),
                            k=torch.randn(
                                2, 48, 4, 64, device="cuda", dtype=torch.bfloat16
                            ),
                            v=torch.randn(
                                2, 48, 4, 64, device="cuda", dtype=torch.bfloat16
                            ),
                            attn_mask=torch.ones(
                                2, 48, device="cuda", dtype=torch.int64
                            ),
                        )
                        self.assertTrue(runner.capture(**kwargs))
                        entry = next(iter(runner.entries.values()))
                        self.assertEqual(entry.num_segments, 1 if capture_dense else 2)
                        for start in (0, 16):
                            for key in ("q", "k", "v"):
                                kwargs[key].normal_()
                            kwargs["attn_mask"].fill_(1)
                            kwargs["attn_mask"][:, start : start + 16] = 0
                            expected = model(**kwargs)
                            actual = runner(**kwargs)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        runner.reset()

    def test_request_backend_overrides_bypass_captured_default(self):
        for backend, skip_softmax in (
            (AttentionBackendEnum.TORCH_SDPA, False),
            (None, True),
        ):
            with self.subTest(backend=backend, skip_softmax=skip_softmax):
                stage = object.__new__(LTX2DenoisingStage)
                stage._attention_backend_active_override = backend
                stage._skip_softmax_forced_fa = skip_softmax
                stage._maybe_get_bcg_runner = Mock(
                    side_effect=AssertionError("stale graph")
                )
                model = Mock(return_value="eager")
                result = stage._ltx2_call_current_model(
                    SimpleNamespace(is_warmup=False),
                    SimpleNamespace(current_model=model),
                    {"value": 7},
                )
                self.assertEqual(result, "eager")
                model.assert_called_once_with(value=7)
                stage._maybe_get_bcg_runner.assert_not_called()


class TestLTX2StepHostOverhead(CustomTestCase):
    def test_video_coord_constants_uploaded_once(self):
        def rope():
            return LTX2AudioVideoRotaryPosEmbed(
                dim=48, patch_size=1, patch_size_t=1, scale_factors=(8, 32, 32)
            )

        kwargs = dict(
            batch_size=2, num_frames=3, height=4, width=5, device=torch.device("cpu")
        )
        expected = rope().prepare_video_coords(**kwargs, fps=24.0)
        cached = rope()
        cached.prepare_video_coords(**kwargs, fps=24.0)
        # every later call reuses the device constants instead of uploading them
        with patch.object(torch, "tensor", wraps=torch.tensor) as upload:
            actual = [
                cached.prepare_video_coords(**kwargs, fps=24.0, start_frame=start)
                for start in (0, 3)
            ]
        self.assertEqual(upload.call_count, 0)
        torch.testing.assert_close(
            actual,
            [expected, rope().prepare_video_coords(**kwargs, fps=24.0, start_frame=3)],
            rtol=0,
            atol=0,
        )

    def test_sde_coefficients_never_read_on_host(self):
        sigma_next = torch.tensor(0.4, dtype=torch.float64)
        # a 0-dim boolean index goes through nonzero, a device sync
        with forbid_host_syncs():
            actual = LTX2DenoisingStage._ltx2_get_sde_coeff(
                sigma_next, sigma_up=sigma_next * 0.5
            )
        sigma_up = (sigma_next * 0.5).clamp(max=sigma_next * 0.9999)
        residual = (sigma_next**2 - sigma_up**2).clamp(min=0) ** 0.5
        alpha_ratio = (1 - sigma_next) + residual
        torch.testing.assert_close(
            actual, (alpha_ratio, residual / alpha_ratio, sigma_up), rtol=0, atol=0
        )

    @torch.no_grad()
    def test_bongmath_refinement_casts_once(self):
        stage = object.__new__(LTX2DenoisingStage)
        stage._ltx2_apply_clean_latent_mask = lambda latents, ctx: latents
        generator = torch.Generator().manual_seed(0)
        ctx = SimpleNamespace(
            latents=torch.randn(1, 6, 4, generator=generator),
            audio_latents=torch.randn(1, 3, 4, generator=generator),
            use_native_hq_res2s_sde_noise=False,
        )
        stage._randn_like_with_batch_generators = lambda latents, batch: torch.randn(
            latents.shape, generator=generator
        )

        def midpoint_model_call(video, audio, sigma):
            return video * 0.5, audio * 0.5, None, None

        with RecordOps() as ops:
            video, audio = stage._ltx2_stage2_res2s_step(
                ctx=ctx,
                batch=None,
                sigma=torch.tensor(0.6),
                sigma_next=torch.tensor(0.45),
                model_video_velocity=torch.ones(1, 6, 4),
                model_audio_velocity=torch.ones(1, 3, 4),
                model_video_timestep=None,
                model_audio_timestep=None,
                midpoint_model_call=midpoint_model_call,
            )
        self.assertTrue(torch.isfinite(video).all() and torch.isfinite(audio).all())
        # the 100-iteration refinement casts the denoised estimates once, not per
        # iteration (two modalities x 100 casts before)
        self.assertLess(ops.casts(), 100)


if __name__ == "__main__":
    unittest.main()
