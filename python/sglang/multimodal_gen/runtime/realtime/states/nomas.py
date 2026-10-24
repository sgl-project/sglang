# SPDX-License-Identifier: Apache-2.0
"""Worker-side state for a Nomas realtime session (one event segment per chunk)."""

from typing import Any

import torch

from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState


class NomasRealtimeState(BaseRealtimeState):
    """DiT segment cache, noise bank and streaming-decode timeline for one session."""

    def __init__(self):
        super().__init__()
        # NomasLatentSession; owns the KV cache and RoPE frame offset.
        self.session: Any = None
        # Pre-sampled latent noise [1, C, T, h, w] on CPU, sliced by absolute
        # latent frame so a segment's noise does not depend on earlier event
        # lengths (upstream interactive_noise bank). Its generator also draws
        # every cache noise, in upstream's order.
        self.noise_bank: torch.Tensor | None = None
        self.noise_generator: torch.Generator | None = None
        # NomasStreamingDecoder: the PixelUMM timeline of every decoded event.
        self.decoder: Any = None
        self.generated_latents: int = 0

    def dispose(self) -> None:
        self.session = None
        self.noise_bank = None
        self.noise_generator = None
        self.decoder = None
        self.generated_latents = 0
