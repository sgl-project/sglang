# SPDX-License-Identifier: Apache-2.0
"""Valen's FP32 four-role Mixer, with checkpoint-compatible parameter names."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from sglang.srt.layers.valen_layout import ValenLayout, validate_valen_layout


class ValenMixerBlock(nn.Module):
    def __init__(self, width: int, token_hidden: int, channel_hidden: int):
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.token_mlp = nn.Sequential(
            nn.Linear(4, token_hidden), nn.GELU(), nn.Linear(token_hidden, 4)
        )
        self.channel_norm = nn.LayerNorm(width)
        self.channel_mlp = nn.Sequential(
            nn.Linear(width, channel_hidden),
            nn.GELU(),
            nn.Linear(channel_hidden, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


class ValenMixerHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_width: int = 256,
        head_layers: int = 2,
        head_token_hidden_dim: int = 16,
        head_channel_hidden_dim: int = 1024,
    ):
        super().__init__()
        if any(
            type(v) is not int or v < 1
            for v in (
                hidden_size,
                head_width,
                head_layers,
                head_token_hidden_dim,
                head_channel_hidden_dim,
            )
        ):
            raise ValueError("Valen Mixer dimensions must be positive integers")
        self.input_norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, head_width)
        self.roles = nn.Parameter(torch.zeros(1, 4, head_width))
        self.blocks = nn.Sequential(
            *(
                ValenMixerBlock(
                    head_width, head_token_hidden_dim, head_channel_hidden_dim
                )
                for _ in range(head_layers)
            )
        )
        self.output = nn.Sequential(nn.LayerNorm(head_width), nn.Linear(head_width, 1))
        # SGLang may construct models under a BF16 default dtype.
        self.float()

    def score_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (
            4,
            self.input_norm.normalized_shape[0],
        ):
            raise ValueError(
                "Expected Valen candidate features shaped [N, 4, hidden_size]"
            )
        # A surrounding BF16 autocast must not cast Mixer projections.
        with torch.autocast(device_type=features.device.type, enabled=False):
            x = self.projection(self.input_norm(features.float())) + self.roles
            return self.output(self.blocks(x)[:, 2]).squeeze(-1)

    @torch.no_grad()
    def forward_batch(
        self,
        hidden: torch.Tensor,
        lengths: Sequence[int],
        layouts: Sequence[ValenLayout],
    ) -> list[torch.Tensor]:
        """Read packed, uncached prefill rows; return candidate logits per request."""
        if hidden.ndim != 2 or hidden.shape[1] != self.input_norm.normalized_shape[0]:
            raise ValueError(
                "Expected packed hidden states shaped [tokens, hidden_size]"
            )
        if len(lengths) != len(layouts) or any(
            type(n) is not int or n < 1 for n in lengths
        ):
            raise ValueError("Valen request lengths and layouts do not match")
        if sum(lengths) != hidden.shape[0]:
            raise ValueError(
                "Valen packed hidden states include missing or extra tokens"
            )
        if not layouts:
            return []
        features, counts, offset = [], [], 0
        for length, layout in zip(lengths, layouts):
            validate_valen_layout(layout, length)
            row = hidden[offset : offset + length]
            a, b = layout.context_span
            context = row[a:b].float().mean(0)
            count = 0
            for question in layout.questions:
                a, b = question.instruction_span
                instruction = row[a:b].float().mean(0)
                decision = row[question.decision_position].float()
                for a, b in question.candidate_spans:
                    candidate = row[a:b].float().mean(0)
                    features.append(
                        torch.stack((context, instruction, candidate, decision))
                    )
                    count += 1
            counts.append(count)
            offset += length
        return list(self.score_features(torch.stack(features)).split(counts))
