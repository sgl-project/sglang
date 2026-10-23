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


def _patch_attention_layers(stack: ExitStack) -> None:
    for name, value in (
        ("get_attn_backend", SDPABackend),
        ("get_compute_dtype", torch.bfloat16),
        ("_resolve_sp_attention_mode", ("ulysses", False)),
        ("_maybe_install_backend_autotune", None),
    ):
        stack.enter_context(patch.object(layer, name, return_value=value))
    stack.enter_context(set_forward_context(0, None))


class TestLTX2PromptMask(CustomTestCase):
    @torch.no_grad()
    def test_local_attention_takes_prebuilt_mask_unchanged(self):
        with ExitStack() as stack:
            _patch_attention_layers(stack)
            attention = layer.LocalAttention(
                num_heads=4,
                head_size=16,
                is_cross_attention=True,
                allow_cudnn_sdp=True,
                capture_dense_attention=True,
            )
            q = torch.randn(2, 8, 4, 16, dtype=torch.bfloat16)
            k = torch.randn(2, 12, 4, 16, dtype=torch.bfloat16)
            v = torch.randn(2, 12, 4, 16, dtype=torch.bfloat16)
            mask = torch.ones(2, 12, dtype=torch.int64)
            mask[1, 7:] = 0
            prebuilt = layer.prepare_sdpa_mask(mask, dtype=q.dtype, device=q.device)
            self.assertTrue(
                torch.equal(
                    attention(q, k, v, attn_mask=prebuilt),
                    attention(q, k, v, attn_mask=mask),
                )
            )

    @torch.no_grad()
    def test_model_mask_built_once_matches_per_block_conversion(self):
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        with ExitStack() as stack:
            _patch_attention_layers(stack)
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
            )
            model = ltx_2.LTX2VideoTransformer3DModel(
                LTX2Config(arch_config=arch), hf_config={}
            ).to(torch.bfloat16)
            for parameter in model.parameters():
                parameter.normal_(std=0.05)
            mask = torch.ones(1, 6, dtype=torch.int64)
            mask[:, 4:] = 0
            kwargs = dict(
                hidden_states=torch.randn(1, 32, 16, dtype=torch.bfloat16),
                audio_hidden_states=torch.randn(1, 5, 16, dtype=torch.bfloat16),
                encoder_hidden_states=torch.randn(1, 6, 64, dtype=torch.bfloat16),
                audio_encoder_hidden_states=torch.randn(1, 6, 64, dtype=torch.bfloat16),
                timestep=torch.tensor([500.0]),
                encoder_attention_mask=mask,
                audio_encoder_attention_mask=mask,
                num_frames=2,
                height=4,
                width=4,
                audio_num_frames=5,
                return_latents=False,
            )
            built_once = model(**kwargs)
            # the blocks' LocalAttention converts the raw mask on every call instead
            with patch.object(
                ltx_2, "prepare_sdpa_mask", side_effect=lambda mask, **_: mask
            ) as convert:
                per_block = model(**kwargs)
            self.assertEqual(convert.call_count, 2)
            for actual, expected in zip(built_once, per_block, strict=True):
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
