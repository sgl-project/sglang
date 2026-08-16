# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
from torch import nn


def sinkhorn_knopp(h: torch.Tensor, *, num_iters: int, eps: float) -> torch.Tensor:
    m = torch.exp(h - h.amax(dim=(-2, -1), keepdim=True))
    for _ in range(num_iters):
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
    return m


class Magi2MHC(nn.Module):
    """``phi_fused`` projects to ``2 * n + n * n``: pre-mix, post-mix, and a flattened stream-to-stream matrix."""

    def __init__(
        self,
        *,
        num_stream: int,
        hidden_size: int,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.num_stream = num_stream
        self.hidden_size = hidden_size
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        # Scaled against the full concatenated stream, not hidden_size.
        self.matmul_scale = 1.0 / math.sqrt(num_stream * hidden_size)

        n = num_stream
        self.phi_fused = nn.Parameter(
            torch.zeros(n * hidden_size, 2 * n + n * n, dtype=torch.float32)
        )
        self.alpha_pre = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_post = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_res = nn.Parameter(torch.full((1,), alpha_init))
        self.bias_pre = nn.Parameter(torch.zeros(n))
        self.bias_post = nn.Parameter(torch.zeros(n))
        self.bias_res = nn.Parameter(torch.zeros(n, n))

    def project(self, streams_flat: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n = self.num_stream
        fused = torch.matmul(streams_flat.float(), self.phi_fused)
        h_pre, h_post, h_res = torch.split(fused, [n, n, n * n], dim=-1)
        return h_pre, h_post, h_res.view(-1, n, n)

    def mix_input(self, streams: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha_pre * self.matmul_scale * h_pre + self.bias_pre)
        return torch.einsum("tn,tnc->tc", gate.to(streams.dtype), streams)

    def mix_output(
        self,
        streams: torch.Tensor,
        block_out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        # Scaled by 2 so the gate spans (0, 2) and can amplify, not just damp.
        post = 2.0 * torch.sigmoid(
            self.alpha_post * self.matmul_scale * h_post + self.bias_post
        )
        res = sinkhorn_knopp(
            self.alpha_res * self.matmul_scale * h_res.float() + self.bias_res,
            num_iters=self.sinkhorn_iters,
            eps=self.eps,
        )
        mixed = torch.einsum("tij,tjc->tic", res.to(streams.dtype), streams)
        written = torch.einsum("tn,tc->tnc", post.to(streams.dtype), block_out)
        return mixed + written
