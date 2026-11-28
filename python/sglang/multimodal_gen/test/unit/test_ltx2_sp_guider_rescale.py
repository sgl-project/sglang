import unittest
from contextlib import nullcontext
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX23PipelineConfig
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX23SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2 import (
    denoising as ltx2_denoising,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.test.test_utils import CustomTestCase

_TOKENS_PER_FRAME = 2
_VIDEO_FRAMES = 3
_AUDIO_FRAMES = 5


def _model(**kwargs):
    # nonlinear in the prompt, so the rescale factor depends on which tokens it sees
    prompt = kwargs["encoder_hidden_states"].mean(dim=(1, 2))[:, None, None]
    video, audio = kwargs["hidden_states"], kwargs["audio_hidden_states"]
    return video * 0.25 + prompt * video.square(), audio * 0.5 + prompt * audio.square()


_model.transformer_blocks = []


def _shard(tensor, frames, tokens_per_frame, rank, world_size):
    local = -(-frames // world_size)
    padded = local * world_size * tokens_per_frame
    pad = tensor.new_zeros(tensor.shape[0], padded - tensor.shape[1], tensor.shape[2])
    full = torch.cat([tensor, pad], dim=1)
    size = local * tokens_per_frame
    return full[:, rank * size : (rank + 1) * size]


class TestLTX2SequenceParallelGuiderRescale(CustomTestCase):
    """Under SP every rank must rescale its slice like one GPU rescales the whole latent."""

    def setUp(self):
        gen = torch.Generator().manual_seed(0)
        self.video = torch.randn(1, _VIDEO_FRAMES * _TOKENS_PER_FRAME, 4, generator=gen)
        self.audio = torch.randn(1, _AUDIO_FRAMES, 4, generator=gen)

    def _run_step(self, video, audio, *, legacy, world_size=1, gather=None):
        stage = object.__new__(LTX2DenoisingStage)
        stage.sampler_name = "euler"
        stage._attention_backend_active_override = None
        stage._skip_softmax_forced_fa = False
        stage._cache_dit_enabled = False
        stage._maybe_get_bcg_runner = Mock(return_value=None)
        stage._ltx2_model_forward_context = lambda *_: nullcontext()
        stage.post_forward_for_ti2v_task = lambda batch, args, mask, latents, z: latents
        stage._use_cached_ltx2_prompt_kv = lambda ctx, step, kwargs: kwargs
        pipeline_config = LTX23PipelineConfig()
        if gather is not None:
            pipeline_config._gather_sp_tensor = gather
        args = SimpleNamespace(
            enable_cfg_parallel=False,
            pipeline_class_name="LTX2TwoStagePipeline",
            pipeline_config=pipeline_config,
        )
        stage.server_args = args
        scheduler = SimpleNamespace(
            sigmas=torch.tensor([1.0, 0.5]),
            step=lambda output, t, sample, return_dict: (sample - 0.5 * output,),
        )
        ctx = SimpleNamespace(
            latents=video,
            audio_latents=audio,
            scheduler=scheduler,
            sigmas_host=torch.tensor([1.0, 0.5]),
            sigmas_device=torch.tensor([1.0, 0.5]),
            audio_scheduler=scheduler,
            stage="stage1",
            is_warmup=False,
            is_ltx23_variant=False,
            use_ltx23_legacy_one_stage=legacy,
            use_ltx23_hq_timestep_semantics=False,
            denoise_mask=None,
            clean_latent=None,
            last_denoised_video=None,
            last_denoised_audio=None,
            reserved_frames_mask=None,
            z=None,
        )
        step = SimpleNamespace(
            step_index=0, current_model=_model, t_device=torch.tensor(1000.0)
        )
        embeds = torch.full((1, 2, 4), 0.3)
        sharded = world_size > 1
        batch = SimpleNamespace(
            quality="exact",
            extra=LTX23SamplingParams().build_request_extra(),
            prompt_embeds=[embeds],
            audio_prompt_embeds=[embeds],
            negative_prompt_embeds=[-embeds],
            negative_audio_prompt_embeds=[-embeds],
            prompt_attention_mask=torch.ones(1, 2),
            negative_attention_mask=torch.ones(1, 2),
            ltx2_num_image_tokens=0,
            do_classifier_free_guidance=True,
            guidance_scale=3.0,
            rollout_controllers={},
            did_sp_shard_latents=sharded,
            # legacy one-stage keeps the audio whole on every rank
            did_sp_shard_audio_latents=sharded and not legacy,
            raw_latent_shape=self.video.shape,
            sp_audio_orig_num_frames=_AUDIO_FRAMES,
        )
        stage._prepare_ltx2_model_inputs = lambda *_: SimpleNamespace(
            latent_model_input=ctx.latents,
            audio_latent_model_input=ctx.audio_latents,
        )
        stage._build_ltx2_base_model_kwargs = lambda *_: {
            "hidden_states": ctx.latents,
            "audio_hidden_states": ctx.audio_latents,
        }
        with (
            patch.object(ltx2_denoising, "get_sp_world_size", return_value=world_size),
            patch(
                "sglang.multimodal_gen.configs.pipeline_configs.base.get_sp_world_size",
                return_value=world_size,
            ),
            patch(
                "sglang.multimodal_gen.configs.pipeline_configs.ltx_2.get_sp_world_size",
                return_value=world_size,
            ),
        ):
            stage._run_denoising_step(ctx, step, batch, args)
        return ctx.latents, ctx.audio_latents

    def _ranks(self, legacy, world_size, gather):
        results = []
        for rank in range(world_size):
            video = _shard(
                self.video, _VIDEO_FRAMES, _TOKENS_PER_FRAME, rank, world_size
            )
            audio = (
                self.audio
                if legacy
                else _shard(self.audio, _AUDIO_FRAMES, 1, rank, world_size)
            )
            results.append(
                self._run_step(
                    video,
                    audio,
                    legacy=legacy,
                    world_size=world_size,
                    gather=gather(rank),
                )
            )
        return results

    def test_sp_ranks_match_single_gpu(self):
        for legacy, world_size in product((True, False), (2, 4)):
            expected_video, expected_audio = self._run_step(
                self.video, self.audio, legacy=legacy
            )
            with self.subTest(legacy=legacy, world_size=world_size):
                # The first pass records what each rank contributes to every
                # gather; the second returns the concatenation, as all-gather does.
                recorded = [[] for _ in range(world_size)]

                def record(rank):
                    def gather(tensor, dim):
                        recorded[rank].append(tensor.clone())
                        return torch.cat([tensor] * world_size, dim=dim)

                    return gather

                self._ranks(legacy, world_size, record)

                def replay(rank):
                    calls = iter(range(len(recorded[rank])))

                    def gather(tensor, dim):
                        i = next(calls)
                        return torch.cat([r[i] for r in recorded], dim=dim)

                    return gather

                results = self._ranks(legacy, world_size, replay)
                video = torch.cat([r[0] for r in results], dim=1)
                audios = (
                    [r[1] for r in results]
                    if legacy
                    else [torch.cat([r[1] for r in results], dim=1)]
                )
                with self.subTest(modality="video"):
                    torch.testing.assert_close(
                        video[:, : expected_video.shape[1]],
                        expected_video,
                        rtol=0,
                        atol=0,
                    )
                for audio in audios:
                    with self.subTest(modality="audio"):
                        torch.testing.assert_close(
                            audio[:, : expected_audio.shape[1]],
                            expected_audio,
                            rtol=0,
                            atol=0,
                        )


if __name__ == "__main__":
    unittest.main()
