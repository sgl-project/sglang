import unittest
from contextlib import ExitStack
from types import SimpleNamespace
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.test_utils import CustomTestCase


class TestLTX23PerSampleAdaLN(CustomTestCase):
    def test_lossless_builds_one_row_per_sample(self):
        timestep = torch.tensor([700.0, 700.0])
        rows = LTX2DenoisingStage._ltx23_token_timesteps(
            timestep, 6, SimpleNamespace(quality="lossless")
        )
        self.assertEqual(tuple(rows.shape), (2, 1))
        tokens = LTX2DenoisingStage._ltx23_token_timesteps(
            timestep, 6, SimpleNamespace(quality="exact")
        )
        self.assertEqual(tuple(tokens.shape), (2, 6))
        self.assertTrue(torch.equal(tokens, rows.expand(2, 6)))

    @torch.no_grad()
    def test_one_row_per_sample_matches_per_token_rows(self):
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
                num_layers=2,
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
            sigma = torch.tensor([0.6, 0.6])
            kwargs = dict(
                hidden_states=torch.randn(2, 32, 16, dtype=torch.bfloat16),
                audio_hidden_states=torch.randn(2, 5, 16, dtype=torch.bfloat16),
                encoder_hidden_states=torch.randn(2, 6, 64, dtype=torch.bfloat16),
                audio_encoder_hidden_states=torch.randn(2, 6, 64, dtype=torch.bfloat16),
                prompt_timestep=sigma * 1000,
                audio_prompt_timestep=sigma * 1000,
                num_frames=2,
                height=4,
                width=4,
                audio_num_frames=5,
                return_latents=False,
            )
            per_token = model(
                **kwargs,
                timestep=(sigma * 1000).view(2, 1).expand(2, 32),
                audio_timestep=(sigma * 1000).view(2, 1).expand(2, 5),
            )
            per_sample = model(
                **kwargs,
                timestep=(sigma * 1000).view(2, 1),
                audio_timestep=(sigma * 1000).view(2, 1),
            )
            for a, e in zip(per_sample, per_token, strict=True):
                torch.testing.assert_close(a, e, rtol=0, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
