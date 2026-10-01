# SPDX-License-Identifier: Apache-2.0
"""Chunked YuE2 NAR/VAE streaming helpers.

The reference pipeline solves the full codec sequence in large context windows.
Serving wants partial audio while the AR stream is still active, so this module
splits the codec stream into small independent acoustic chunks and decodes each
chunk immediately. Chunk boundaries are therefore a serving policy, not part of
the model protocol.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from .modeling_vae import YuE2VAE
from .nar import Chunk, CachedNAR
from .protocol import CODEC_OFFSET, CONTEXT, MUSIC_END


@dataclass
class StreamingConfig:
    """Serving-side streaming policy.

    `codec_window` and `codec_overlap` are in codec frames (25 frames/s).
    The VAE halo must satisfy the decoder's dependency interval.
    """

    codec_window: int = 50
    codec_overlap: int = 0
    ode_steps: int = 32
    context: int = CONTEXT
    attention: str = "sdpa"
    query_chunk_size: int | None = None
    offload_ar: bool = False
    vae_core_frames: int = 256
    vae_halo_frames: int = 16


def _windows(total: int, window: int, overlap: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    if window <= overlap:
        raise ValueError("codec_window must be greater than codec_overlap")
    step = window - overlap
    ranges = []
    start = 0
    while start < total:
        end = min(total, start + window)
        ranges.append((start, end))
        if end == total:
            break
        start += step
    return ranges


@torch.inference_mode()
def stream_audio(
    model,
    vae: YuE2VAE,
    *,
    prefix: list[int],
    codec: list[int],
    seed: int,
    config: StreamingConfig | None = None,
) -> Iterator[tuple[torch.Tensor, int, int]]:
    """Yield ``(audio [channels,samples], start_code, end_code)`` for each chunk.

    Each NAR chunk conditions on the original AR prefix plus the local codec
    span, exactly like the reference solver. The generator keeps no rolling
    latent state because YuE2's NAR path is condition-only across chunks.
    """
    config = config or StreamingConfig()
    if not codec:
        return
    for start, end in _windows(len(codec), config.codec_window, config.codec_overlap):
        local_codec = codec[start:end]
        chunk = Chunk(
            ar_tokens=prefix
            + [value + CODEC_OFFSET for value in local_codec]
            + [MUSIC_END],
            noise=torch.randn(
                (len(local_codec), 64),
                dtype=torch.float32,
                device="cpu",
                generator=torch.Generator(device="cpu").manual_seed(int(seed) + start),
            ),
        )
        engine = CachedNAR(
            model,
            chunk,
            attention=config.attention,
            query_chunk_size=config.query_chunk_size,
        )
        try:
            latents = engine.solve(steps=config.ode_steps)
            audio = vae.decode_tiled(
                latents.unsqueeze(0).transpose(1, 2),
                core_frames=config.vae_core_frames,
                halo_frames=config.vae_halo_frames,
                output_device="cpu",
            )
            yield audio[0], start, end
        finally:
            engine.close()
