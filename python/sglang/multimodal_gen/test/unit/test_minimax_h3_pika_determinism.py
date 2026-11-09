# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 / Pika H3 Flex run-to-run determinism contracts (CPU)."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import (
    cudnn_composite_sdpa as ccs,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae import bigvgan


def test_bigvgan_snakebeta_is_eager() -> None:
    # TorchScript's profiling executor changes rounding after the first calls.
    assert not isinstance(bigvgan.snakebeta, torch.jit.ScriptFunction)
    x = torch.randn(1, 4, 16)
    alpha = torch.rand(1, 4, 1)
    beta = torch.rand(1, 4, 1)
    first = bigvgan.snakebeta(x, alpha, beta)
    for _ in range(3):
        assert torch.equal(bigvgan.snakebeta(x, alpha, beta), first)


def test_composite_sdpa_declines_unsupported_inputs() -> None:
    q = torch.randn(1, 2, 8, 64, dtype=torch.bfloat16)
    # CPU tensors (or a missing cuDNN frontend) fall back to the caller's path.
    assert ccs.cudnn_composite_sdpa(q, q, q, 0.125) is None


def _impl(**extra) -> SDPAImpl:
    return SDPAImpl(
        num_heads=2, head_size=64, causal=False, softmax_scale=0.125, **extra
    )


def test_composite_sdpa_needs_cudnn_opt_in() -> None:
    assert not _impl().cudnn_composite_sdp
    assert not _impl(cudnn_composite_sdp=True).cudnn_composite_sdp
    assert _impl(allow_cudnn_sdp=True, cudnn_composite_sdp=True).cudnn_composite_sdp
