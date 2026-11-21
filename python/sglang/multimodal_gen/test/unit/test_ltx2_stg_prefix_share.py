import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

import sglang.multimodal_gen.runtime.models.dits.ltx_2 as ltx_2
from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2ArchConfig, LTX2Config
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.test_utils import CustomTestCase

STG_BLOCK = 2


class TestLTX2STGPrefixShare(CustomTestCase):
    """An STG pass resumed from the cond pass's block input equals a full STG pass."""

    @torch.no_grad()
    def test_resumed_stg_pass_matches_full_pass(self):
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        with ExitStack() as stack:
            for name, value in (
                ("get_attn_backend", SDPABackend),
                ("get_compute_dtype", torch.bfloat16),
                ("_resolve_sp_attention_mode", ("ulysses", False)),
                ("_maybe_install_backend_autotune", None),
            ):
                stack.enter_context(patch.object(layer, name, return_value=value))
            stack.enter_context(set_forward_context(0, None))
            torch.manual_seed(0)
            arch = LTX2ArchConfig(
                num_layers=4,
                num_attention_heads=2,
                attention_head_dim=32,
                in_channels=16,
                out_channels=16,
                cross_attention_dim=64,
                caption_channels=64,
                audio_num_attention_heads=2,
                audio_attention_head_dim=16,
                audio_in_channels=16,
                audio_out_channels=16,
                audio_cross_attention_dim=32,
                cross_attention_adaln=True,
                apply_gated_attention=True,
            )
            arch.ltx_variant = "ltx_2_3"
            model = ltx_2.LTX2VideoTransformer3DModel(
                LTX2Config(arch_config=arch), hf_config={}
            ).to(torch.bfloat16)
            for parameter in model.parameters():
                parameter.normal_(std=0.05)

            def inputs(batch_size):
                generator = torch.Generator().manual_seed(1)
                sigma = torch.full((batch_size,), 600.0)

                def randn(*shape):
                    return torch.randn(*shape, generator=generator).to(torch.bfloat16)

                row = dict(
                    hidden_states=randn(1, 32, 16),
                    audio_hidden_states=randn(1, 5, 16),
                    encoder_hidden_states=randn(1, 6, 64),
                    audio_encoder_hidden_states=randn(1, 6, 64),
                )
                return dict(
                    **{k: v.expand(batch_size, *v.shape[1:]) for k, v in row.items()},
                    timestep=sigma.view(batch_size, 1),
                    audio_timestep=sigma.view(batch_size, 1),
                    num_frames=2,
                    height=4,
                    width=4,
                    audio_num_frames=5,
                    return_latents=False,
                )

            stg = dict(
                skip_video_self_attn_blocks=(STG_BLOCK,),
                skip_audio_self_attn_blocks=(STG_BLOCK,),
            )
            full = model(**inputs(1), **stg)

            cond_video, cond_audio, (block_video, block_audio) = model(
                **inputs(1), ltx2_stg_capture_block=STG_BLOCK
            )
            resumed = model(
                **inputs(1),
                **stg,
                ltx2_stg_resume_block=STG_BLOCK,
                ltx2_stg_resume_hidden_states=block_video,
                ltx2_stg_resume_audio_hidden_states=block_audio,
            )
            for got, want in zip(resumed, full, strict=True):
                self.assertTrue(torch.equal(got, want))

            # Capturing leaves the cond pass's own output unchanged.
            plain = model(**inputs(1))
            for got, want in zip((cond_video, cond_audio), plain, strict=True):
                self.assertTrue(torch.equal(got, want))

            # Batched: rows 1..2 of a 3-row call feed a 2-row STG pass.
            _, _, (block_video, block_audio) = model(
                **inputs(3), ltx2_stg_capture_block=STG_BLOCK
            )
            resumed = model(
                **inputs(2),
                **stg,
                ltx2_stg_resume_block=STG_BLOCK,
                ltx2_stg_resume_hidden_states=block_video[1:3],
                ltx2_stg_resume_audio_hidden_states=block_audio[1:3],
            )
            full = model(**inputs(2), **stg)
            for got, want in zip(resumed, full, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=2e-2)


class TestLTX2STGQuantizedGate(CustomTestCase):
    def test_quantized_dit_keeps_the_batched_guidance_forward(self):
        from types import SimpleNamespace

        from sglang.multimodal_gen.runtime.layers.linear import (
            UnquantizedLinearMethod,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
            LTX2DenoisingStage,
        )

        def dit(method):
            model = torch.nn.Sequential(torch.nn.Linear(2, 2))
            model[0].quant_method = method
            return model

        check = LTX2DenoisingStage._ltx2_dit_has_quantized_linears
        self.assertFalse(check(dit(UnquantizedLinearMethod())))
        self.assertTrue(check(dit(SimpleNamespace(name="fp8"))))


if __name__ == "__main__":
    unittest.main()
