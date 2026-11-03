# SPDX-License-Identifier: Apache-2.0
"""Dynamic linear LoKr, without expanding a Kronecker product into a dense weight."""

import torch
from torch import nn


class LoKrTerm(nn.Module):
    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, start: int = 0):
        super().__init__()
        if any(w.ndim != 2 or min(w.shape) == 0 for w in (w1, w2)):
            raise ValueError("LoKr requires nonempty 2D lokr_w1 and lokr_w2 factors")
        if w1.dtype != w2.dtype or not w1.is_floating_point():
            raise ValueError("LoKr factors must have the same floating-point dtype")
        self.register_buffer("w1", w1)
        self.register_buffer("w2", w2)
        self.start = start
        self.input_size = w1.shape[1] * w2.shape[1]
        self.output_size = w1.shape[0] * w2.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ kron(w1, w2).T = w1 @ reshape(x) @ w2.T for each token.
        # Factors stay compact, including full-rank LoKr checkpoints.
        x = x.to(self.w1.dtype).reshape(
            *x.shape[:-1], self.w1.shape[1], self.w2.shape[1]
        )
        return (self.w1 @ (x @ self.w2.T)).flatten(start_dim=-2)


class LoKrAdapter(nn.Module):
    """One adapter's disjoint output slices, including a fused QKV or gate/up layer."""

    def __init__(self, terms: list[LoKrTerm], output_size: int):
        super().__init__()
        if not terms:
            raise ValueError("LoKr requires at least one output slice")
        self.terms = nn.ModuleList(sorted(terms, key=lambda term: term.start))
        self.output_size = output_size
        end = 0
        for term in self.terms:
            if term.start < end or term.start + term.output_size > output_size:
                raise ValueError("Overlapping or out-of-bounds LoKr output slices")
            end = term.start + term.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        end = 0
        for term in self.terms:
            delta = term(x)
            if term.start > end:
                parts.append(delta.new_zeros(*delta.shape[:-1], term.start - end))
            parts.append(delta)
            end = term.start + term.output_size
        if end < self.output_size:
            parts.append(parts[-1].new_zeros(*x.shape[:-1], self.output_size - end))
        return torch.cat(parts, dim=-1)
