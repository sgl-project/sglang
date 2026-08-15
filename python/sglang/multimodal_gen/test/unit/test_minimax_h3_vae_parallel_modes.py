# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 released VAE decode contract."""

from unittest import mock

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
    _apply_qk_norm,
)


def _init_kwargs(config: MiniMaxH3VideoVAEConfig):
    with mock.patch.object(
        AutoencoderKLLegacy, "__init__", autospec=True, return_value=None
    ) as init:
        model = MiniMaxH3VideoVAE(config)
    return model, init.call_args.kwargs


@pytest.mark.parametrize(
    "mode",
    [
        None,
        "auto",
        "tiled",
    ],
)
def test_decode_mode_uses_released_tiled_recipe(mode):
    config = (
        MiniMaxH3VideoVAEConfig()
        if mode is None
        else MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    )
    model, kwargs = _init_kwargs(config)

    assert model.parallel_decode_mode == "tiled"
    assert kwargs["decoder_tiling"] is True
    assert kwargs["parallel_tiling"] is True
    assert kwargs["decoder_parallel"] is False


@pytest.mark.parametrize("mode", ["spatial", "spatial_shard", "patch"])
def test_unvalidated_decode_modes_are_rejected(mode):
    config = MiniMaxH3VideoVAEConfig(parallel_decode_mode=mode)
    with pytest.raises(ValueError, match="use tiled"):
        config.resolved_parallel_decode_mode()


def test_vit_attention_uses_local_usp_backend_dispatch():
    module = (
        "sglang.multimodal_gen.runtime.models.vaes." "minimax_h3_video_vae.attention"
    )
    with (
        mock.patch(f"{module}.current_platform.is_cuda", return_value=True),
        mock.patch(f"{module}.USPAttention", autospec=True) as usp_attention,
    ):
        Attention(heads=2, dim_head=64)

    assert usp_attention.call_args.kwargs["skip_sequence_parallel"] is True


def test_vit_qk_norm_supports_affine_free_rmsnorm():
    norm = nn.RMSNorm(64, elementwise_affine=False)
    hidden_states = torch.randn(1, 2, 2, 64)

    output = _apply_qk_norm(norm, hidden_states)

    assert output.shape == hidden_states.shape
