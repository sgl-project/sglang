# SPDX-License-Identifier: Apache-2.0
"""GLM-Image attention rotates its image tokens on every CUDA layout."""

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention import layer as attention_layer
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.glm_image import (
    GlmImageAttention,
    GlmImageRotaryPosEmbed,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs the CUDA RoPE")
@torch.no_grad()
def test_in_place_rope_on_contiguous_views_matches_the_copy_back_path(
    single_process_model_parallel, monkeypatch
):
    """The CUDA RoPE rotates a contiguous single-request Q/K view in place and
    returns that view; writing it back onto itself must not fail the request."""
    torch.manual_seed(0)
    attn = (
        GlmImageAttention(
            query_dim=64,
            heads=2,
            dim_head=32,
            out_dim=64,
            bias=True,
            qk_norm="layer_norm",
            elementwise_affine=False,
            eps=1e-5,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    for param in attn.parameters():
        param.normal_(0, 0.05)
    hidden = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16)
    encoder = torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16)
    rope = GlmImageRotaryPosEmbed(32, 1)(torch.empty(1, 4, 4, 4, device="cuda"))

    with set_forward_context(current_timestep=0, attn_metadata=None):
        in_place = attn(hidden, encoder, image_rotary_emb=rope)
        rotate = attn.rotary_emb.forward
        monkeypatch.setattr(
            attn.rotary_emb,
            "forward",
            lambda **kwargs: tuple(t.clone() for t in rotate(**kwargs)),
        )
        copied = attn(hidden, encoder, image_rotary_emb=rope)

    for actual, expected in zip(in_place, copied, strict=True):
        assert torch.equal(actual, expected)


def test_declared_attention_backends_constrain_backend_selection(
    single_process_model_parallel, monkeypatch
):
    """The backend set the model declares for its attention layers reaches the
    backend selector; without it selection is unconstrained."""
    declared = {AttentionBackendEnum.TORCH_SDPA}
    selections = []
    select = attention_layer.get_attn_backend

    def record(*args, **kwargs):
        selections.append(kwargs["supported_attention_backends"])
        return select(*args, **kwargs)

    monkeypatch.setattr(attention_layer, "get_attn_backend", record)
    attn = GlmImageAttention(
        query_dim=64,
        heads=2,
        dim_head=32,
        out_dim=64,
        bias=True,
        qk_norm="layer_norm",
        elementwise_affine=False,
        eps=1e-5,
        supported_attention_backends=declared,
    )
    assert selections == [declared]
    assert attn.attn.backend is AttentionBackendEnum.TORCH_SDPA
