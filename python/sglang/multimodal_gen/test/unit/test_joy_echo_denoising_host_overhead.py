import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2AudioVideoRotaryPosEmbed,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joy_echo import (
    denoising as joy_echo_denoising,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2ModelInputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av import (
    LTX2AVDenoisingStage,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import SyncOnRead
from sglang.test.test_utils import CustomTestCase

JoyEchoDMDDenoisingStage = joy_echo_denoising.JoyEchoDMDDenoisingStage


def _model_inputs(video, audio, timestep):
    return LTX2ModelInputs(
        latent_model_input=video,
        audio_latent_model_input=audio,
        audio_num_frames_latent=audio.shape[1],
        video_coords=None,
        audio_coords=None,
        timestep_video=timestep,
        timestep_audio=timestep,
        prompt_timestep_video=None,
        prompt_timestep_audio=None,
        video_self_attention_mask=None,
        audio_self_attention_mask=None,
        a2v_cross_attention_mask=None,
        v2a_cross_attention_mask=None,
    )


class TestJoyEchoDenoisingHostOverhead(CustomTestCase):
    def test_loop_takes_host_and_device_sigma_copies(self):
        stage = object.__new__(JoyEchoDMDDenoisingStage)
        sigmas = torch.tensor([1.0, 0.75, 0.0], dtype=torch.float64)
        ctx = SimpleNamespace(
            scheduler=SimpleNamespace(sigmas=sigmas),
            latents=torch.zeros(1, 2, 3),
            extra={},
        )
        with (
            patch.object(
                LTX2AVDenoisingStage, "_prepare_denoising_loop", return_value=ctx
            ),
            patch.object(joy_echo_denoising, "get_sp_world_size", return_value=1),
        ):
            stage._prepare_denoising_loop(None, None)
        host, device = ctx.extra["joy_echo_sigmas"]
        self.assertEqual(host.device.type, "cpu")
        self.assertEqual(device.device, ctx.latents.device)
        for copy in (host, device):
            self.assertEqual(copy.dtype, torch.float32)
            self.assertTrue(torch.equal(copy, sigmas.float()))

    def _run_step(self, step_index, sigmas_device):
        stage = object.__new__(JoyEchoDMDDenoisingStage)
        model_sigmas = []
        generator = torch.Generator().manual_seed(step_index)
        video = torch.randn(1, 4, 3, generator=generator)
        audio = torch.randn(1, 2, 3, generator=generator)

        def prepare_inputs(ctx, step, batch, server_args, sigma):
            model_sigmas.append(sigma.as_subclass(torch.Tensor))
            return _model_inputs(ctx.latents, ctx.audio_latents, sigma.expand(1))

        for name, value in (
            ("_prepare_ltx2_model_inputs", prepare_inputs),
            ("_get_ltx_prompt_attention_mask", lambda *_, **__: None),
            ("_build_ltx2_base_model_kwargs", lambda *_: {}),
            ("_build_ltx2_model_kwargs", lambda *_, **__: {}),
            ("_ltx2_model_forward_context", lambda *_: nullcontext()),
            ("_ltx2_call_current_model", lambda *_: (video * 0.5, audio * 0.25)),
            ("_ltx2_apply_clean_latent_mask", lambda latents, ctx: latents),
            (
                "_sample_sp_consistent_noise",
                lambda latents, *_, **__: torch.ones_like(latents),
            ),
            ("post_forward_for_ti2v_task", lambda *args: args[3]),
        ):
            setattr(stage, name, value)
        sigmas = torch.tensor([1.0, 0.75, 0.0])
        ctx = SimpleNamespace(
            latents=video.clone(),
            audio_latents=audio.clone(),
            audio_scheduler=object(),
            scheduler=SimpleNamespace(sigmas=sigmas_device),
            extra={"joy_echo_sigmas": (sigmas, sigmas_device)},
            is_ltx23_variant=True,
            reserved_frames_mask=None,
            z=None,
        )
        batch = SimpleNamespace(
            prompt_embeds=[None],
            audio_prompt_embeds=[None],
            did_sp_shard_latents=False,
            did_sp_shard_audio_latents=False,
            extra={},
        )
        step = SimpleNamespace(step_index=step_index, current_model=None)
        stage._run_denoising_step(ctx, step, batch, None)
        return ctx.latents, ctx.audio_latents, model_sigmas

    def test_steps_never_read_device_sigmas_on_host(self):
        for step_index in (0, 1):
            with self.subTest(step_index=step_index):
                expected = self._run_step(step_index, torch.tensor([1.0, 0.75, 0.0]))
                actual = self._run_step(
                    step_index,
                    torch.tensor([1.0, 0.75, 0.0]).as_subclass(SyncOnRead),
                )
                for result, reference in zip(actual[:2], expected[:2], strict=True):
                    self.assertTrue(
                        torch.equal(result.as_subclass(torch.Tensor), reference)
                    )
                self.assertTrue(torch.equal(actual[2][0], expected[2][0]))

    @patch.object(joy_echo_denoising, "get_sp_world_size", return_value=1)
    def test_memory_prefix_built_once_per_loop(self, _):
        stage = object.__new__(JoyEchoDMDDenoisingStage)
        step_model = SimpleNamespace(
            rope=LTX2AudioVideoRotaryPosEmbed(dim=48, modality="video"),
            audio_rope=LTX2AudioVideoRotaryPosEmbed(dim=24, modality="audio"),
        )
        generator = torch.Generator().manual_seed(0)
        memory_info = {
            "memory_video_packed": torch.randn(1, 12, 3, generator=generator),
            "memory_audio": torch.randn(1, 4, 5, generator=generator),
            "memory_audio_segment_lengths": ((2, 2),),
            "num_memory_slots": 2,
            "memory_position_mode": "reference",
            "memory_downscale_factor": 1,
        }
        batch = SimpleNamespace(
            extra={"joy_echo_memory": memory_info},
            did_sp_shard_latents=False,
            fps=24.0,
        )
        ctx = SimpleNamespace(
            latent_height=2, latent_width=3, latent_num_frames_for_model=2, extra={}
        )
        server_args = SimpleNamespace(pipeline_config=None)

        def step_inputs(step):
            return _model_inputs(
                torch.randn(1, 12, 3, generator=generator),
                torch.randn(1, 6, 5, generator=generator),
                torch.tensor([1.0 - 0.25 * step]),
            )

        steps = [step_inputs(step) for step in range(3)]
        # rebuilding the memory inputs every step, as the loop used to
        per_step = []
        for inputs in steps:
            ctx.extra.clear()
            per_step.append(
                stage._build_memory_model_inputs(
                    inputs, batch, ctx, server_args, step_model
                )
            )
        ctx.extra.clear()
        builders = (
            "build_paired_memory_cross_mask",
            "build_memory_self_attention_block_mask",
            "build_memory_video_rope_coords",
            "build_memory_audio_rope_coords",
        )
        with ExitStack() as stack:
            mocks = [
                stack.enter_context(
                    patch.object(
                        joy_echo_denoising,
                        name,
                        wraps=getattr(joy_echo_denoising, name),
                    )
                )
                for name in builders
            ]
            per_loop = [
                stage._build_memory_model_inputs(
                    inputs, batch, ctx, server_args, step_model
                )
                for inputs in steps
            ]
        # the a2v and v2a masks, the audio block mask and the two coords, once
        self.assertEqual([mock.call_count for mock in mocks], [2, 1, 1, 1])
        for (actual, actual_meta), (expected, expected_meta) in zip(
            per_loop, per_step, strict=True
        ):
            self.assertEqual(actual_meta, expected_meta)
            for field in LTX2ModelInputs.__dataclass_fields__:
                actual_value = getattr(actual, field)
                expected_value = getattr(expected, field)
                if isinstance(expected_value, torch.Tensor):
                    self.assertTrue(torch.equal(actual_value, expected_value), field)
                else:
                    self.assertEqual(actual_value, expected_value, field)


if __name__ == "__main__":
    unittest.main()
