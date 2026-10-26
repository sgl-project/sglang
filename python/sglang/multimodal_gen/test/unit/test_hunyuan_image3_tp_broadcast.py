"""HunyuanImage-3 denoising steps: tensor-parallel input broadcasts and the
host work each step skips."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.models.dits import hunyuan_image3_utils
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    ar_stage,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import forbid_host_syncs
from sglang.test.test_utils import CustomTestCase


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def contiguous(self):
        return self

    def reshape(self, *_shape):
        return self


class _FakeTPGroup:
    world_size = 2

    def __init__(self):
        self.broadcast_calls = []

    def broadcast(self, tensor, src):
        self.broadcast_calls.append((tensor, src))
        return tensor


class TestHunyuanImage3TPBroadcast(CustomTestCase):
    def test_static_inputs_are_broadcast_once(self):
        stage = ar_stage.HunyuanImage3AR.__new__(ar_stage.HunyuanImage3AR)
        tp_group = _FakeTPGroup()
        attention_mask = _FakeTensor((2, 1, 8, 8))
        cos = _FakeTensor((2, 8, 4))
        sin = _FakeTensor((2, 8, 4))

        with (
            patch.object(ar_stage, "model_parallel_is_initialized", return_value=True),
            patch.object(ar_stage, "get_tp_group", return_value=tp_group),
        ):
            result = stage._broadcast_static_inputs(attention_mask, (cos, sin))

        self.assertEqual(result, (attention_mask, (cos, sin)))
        self.assertEqual(
            tp_group.broadcast_calls,
            [(attention_mask, 0), (cos, 0), (sin, 0)],
        )

    def test_backbone_forward_only_broadcasts_dynamic_hidden_states(self):
        stage = ar_stage.HunyuanImage3AR.__new__(ar_stage.HunyuanImage3AR)
        tp_group = _FakeTPGroup()
        hidden_states = _FakeTensor((2, 3, 4))
        attention_mask = _FakeTensor((2, 1, 3, 3))
        cos = _FakeTensor((2, 3, 2))
        sin = _FakeTensor((2, 3, 2))
        output = MagicMock()
        output.shape = (6, 4)
        output.view.return_value = "reshaped-output"
        stage._cache_dit_adapter = None
        stage.ar_model = SimpleNamespace(forward_block=MagicMock(return_value=output))

        with (
            patch.object(ar_stage, "model_parallel_is_initialized", return_value=True),
            patch.object(ar_stage, "get_tp_group", return_value=tp_group),
        ):
            result = stage._backbone_forward(
                num_image_tokens=2,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                custom_pos_emb=(cos, sin),
                first_step=True,
            )

        self.assertEqual(result, "reshaped-output")
        self.assertEqual(tp_group.broadcast_calls, [(hidden_states, 0)])


class TestHunyuanImage3DenoiseStep(CustomTestCase):
    def test_timestep_frequencies_are_built_once(self):
        # the host-built frequencies are uploaded once, not on each of the five
        # timestep embeddings per step (each pageable upload blocks the host)
        hunyuan_image3_utils._timestep_freqs.cache_clear()
        t = torch.tensor([999.0, 500.0])
        half = 128
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        expected = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        with patch.object(hunyuan_image3_utils.torch, "exp", wraps=torch.exp) as exp:
            for _ in range(3):
                actual = hunyuan_image3_utils.timestep_embedding(t, 2 * half)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(exp.call_count, 1)

    def test_image_tokens_move_through_precomputed_positions(self):
        # scatter/gather through positions taken once per request: the per-step
        # masked_select sized its output on the host every step
        generator = torch.Generator().manual_seed(0)
        bsz, seqlen, n_embd, token_h, token_w = 2, 10, 6, 2, 2
        image_mask = torch.zeros(bsz, seqlen, dtype=torch.bool)
        image_mask[0, 3:7] = True
        image_mask[1, 5:9] = True
        hidden = torch.randn(bsz, seqlen, n_embd, generator=generator)
        image_seq = torch.randn(bsz, token_h * token_w, n_embd, generator=generator)
        stage = object.__new__(ar_stage.HunyuanImage3AR)
        stage.ar_model = SimpleNamespace(
            time_embed=lambda t: t,
            time_embed_2=lambda t: t,
            patch_embed=lambda images, t_emb: (image_seq, token_h, token_w),
            final_layer=lambda x, t_emb, h, w: x,
        )
        reference_index = (
            torch.arange(seqlen).unsqueeze(0).expand(bsz, -1).masked_select(image_mask)
        ).reshape(bsz, -1)
        scattered = hidden.clone().scatter_(
            1, reference_index.unsqueeze(-1).expand(-1, -1, n_embd), image_seq
        )
        extracted = hidden.masked_select(image_mask.unsqueeze(-1)).reshape(
            -1, token_h * token_w, n_embd
        )

        index = stage._mask_positions(image_mask, bsz, seqlen, hidden.device)
        t = torch.tensor([999.0, 999.0])
        with forbid_host_syncs():
            actual = (
                stage._instantiate_vae_tokens_first_step(
                    hidden.clone(), None, t, index
                ),
                stage._extract_diffusion_pred(hidden, t, index, token_h, token_w),
            )
        torch.testing.assert_close(actual, (scattered, extracted), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
