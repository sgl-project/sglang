# SPDX-License-Identifier: Apache-2.0
"""Per-embodiment-domain linear projection for Cosmos3 action heads."""

import torch
from torch import nn


class DomainAwareLinear(nn.Module):
    """Linear projection with one weight/bias pair per action embodiment domain.

    ``fc`` and ``bias`` are ``nn.Embedding`` tables holding the flattened
    ``(out*in)`` weight and ``(out)`` bias for each of ``num_domains``
    embodiments, selected per token by ``domain_id``. Stored as embeddings so the
    checkpoint keys ``<name>.fc.weight`` / ``<name>.bias.weight`` load directly,
    with the matmul reshape done at call time.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_domains = int(num_domains)
        self.fc = nn.Embedding(self.num_domains, self.output_size * self.input_size)
        self.bias = nn.Embedding(self.num_domains, self.output_size)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        if domain_id.ndim == 0:
            domain_id = domain_id.unsqueeze(0)
        domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
        if x.shape[0] != domain_id.shape[0]:
            raise ValueError(
                "Cosmos3 action domain_id batch size must match action tokens: "
                f"tokens={x.shape[0]}, domain_id={domain_id.shape[0]}."
            )
        if torch.any((domain_id < 0) | (domain_id >= self.num_domains)):
            raise ValueError(
                f"Cosmos3 action domain_id must be in [0, {self.num_domains}), "
                f"got {domain_id.tolist()}."
            )
        weight = self.fc(domain_id).view(
            domain_id.shape[0], self.input_size, self.output_size
        )
        bias = self.bias(domain_id).view(domain_id.shape[0], self.output_size)
        if x.ndim == 2:
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        if x.ndim == 3:
            return torch.bmm(x, weight) + bias.unsqueeze(1)
        raise ValueError(
            "Cosmos3 DomainAwareLinear expected rank-2 or rank-3 input, "
            f"got {tuple(x.shape)}."
        )
