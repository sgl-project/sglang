"""HunyuanImage-3 denoising steps read nothing back to the host."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.models.dits import hunyuan_image3_utils
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    ar_stage,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import forbid_host_syncs


def test_timestep_frequencies_are_built_once():
    """The host-built frequencies are uploaded once, not on each of the five
    timestep embeddings per step (each pageable upload blocks the host)."""
    hunyuan_image3_utils._timestep_freqs.cache_clear()
    t = torch.tensor([999.0, 500.0])
    half = 128
    freqs = torch.exp(
        -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    expected = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    with patch.object(hunyuan_image3_utils.torch, "exp", wraps=torch.exp) as exp:
        for _ in range(3):
            actual = hunyuan_image3_utils.timestep_embedding(t, 2 * half)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert exp.call_count == 1


def test_image_tokens_move_through_precomputed_positions():
    """Scatter/gather through positions taken once per request: the per-step
    masked_select sized its output on the host every step."""
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
        actual_scattered = stage._instantiate_vae_tokens_first_step(
            hidden.clone(), None, t, index
        )
        actual_extracted = stage._extract_diffusion_pred(
            hidden, t, index, token_h, token_w
        )
    torch.testing.assert_close(actual_scattered, scattered, atol=0, rtol=0)
    torch.testing.assert_close(actual_extracted, extracted, atol=0, rtol=0)
