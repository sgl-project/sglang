# SPDX-License-Identifier: Apache-2.0
"""LTX guidance must follow the same warmup/capture route as plain CFG."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.multimodal_gen.configs.sample.ltx_2 import (
    LTX2SamplingParams,
    LTX23HQSamplingParams,
    LTX23SamplingParams,
)
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    _ltx2_build_batched_perturbation_states,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Importing the native denoising stage needs the diffusion CUDA dependencies.
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class TestLTX2GuidanceBCG(CustomTestCase):
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

    def _run_step(self, *, mode, warmup, runner, value):
        stage = object.__new__(LTX2DenoisingStage)
        stage.sampler_name = "euler"
        stage._attention_backend_active_override = None
        stage._skip_softmax_forced_fa = False
        stage._maybe_get_bcg_runner = Mock(return_value=runner)
        stage._ltx2_model_forward_context = lambda *_: nullcontext()
        stage.post_forward_for_ti2v_task = lambda batch, args, mask, latents, z: latents
        args = SimpleNamespace(
            enable_cfg_parallel=False,
            pipeline_class_name=(
                "LTX2TwoStageHQPipeline" if mode == "hq" else "LTX2TwoStagePipeline"
            ),
        )
        stage.server_args = args
        video = torch.arange(8, dtype=torch.float32).view(1, 2, 4) + value
        audio = torch.arange(4, dtype=torch.float32).view(1, 1, 4) + value
        ctx = SimpleNamespace(
            latents=video,
            audio_latents=audio,
            scheduler=SimpleNamespace(sigmas=torch.tensor([1.0, 0.5])),
            audio_scheduler=object(),
            stage="stage1",
            is_warmup=warmup,
            is_ltx23_variant=False,
            use_ltx23_legacy_one_stage=mode == "legacy",
            use_ltx23_hq_timestep_semantics=False,
            denoise_mask=None,
            clean_latent=None,
            last_denoised_video=None,
            last_denoised_audio=None,
            reserved_frames_mask=None,
            z=None,
        )
        step = SimpleNamespace(step_index=0, current_model=self._model)
        params = LTX23SamplingParams(video_rescale_scale=0.0, audio_rescale_scale=0.0)
        embeds = torch.full((1, 2, 4), value)
        batch = SimpleNamespace(
            extra=params.build_request_extra(),
            prompt_embeds=[embeds],
            audio_prompt_embeds=[embeds],
            negative_prompt_embeds=[-embeds],
            negative_audio_prompt_embeds=[-embeds],
            prompt_attention_mask=torch.ones(1, 2),
            negative_attention_mask=torch.ones(1, 2),
            ltx2_num_image_tokens=int(mode == "ti2v"),
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
        for mode in ("batched", "ti2v", "hq", "legacy"):
            with self.subTest(mode=mode):
                runner = Mock(side_effect=self._model)
                calls_per_step = 1 if mode == "batched" else 4
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
                if mode == "batched":
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


if __name__ == "__main__":
    unittest.main()
