from __future__ import annotations

from typing import Tuple

import torch

from sglang.srt.runtime_context import get_spec


class PPSpecRelay:
    """Per request slot (req_pool_idx): the draft tree its next verify round
    runs, drafted by the last PP stage (the only one holding the draft model)
    and brought back by the output relay. The ring commits a slot's step before
    it schedules the slot again, so a row's tree is written before it is read.
    """

    def __init__(self, req_pool_size: int, device: str):
        spec = get_spec()
        self.num_draft_tokens = spec.speculative_num_draft_tokens
        num_steps = spec.speculative_num_steps
        parent_width = (
            spec.speculative_eagle_topk * (num_steps - 1) + 1 if num_steps > 1 else 0
        )
        # Column 0 of a tree is its root, the bonus token.
        self.tokens = torch.zeros(
            (req_pool_size, self.num_draft_tokens), dtype=torch.int64, device=device
        )
        self.parents = torch.zeros(
            (req_pool_size, parent_width), dtype=torch.long, device=device
        )
        self.top_scores = torch.zeros(
            (req_pool_size, self.num_draft_tokens - 1), dtype=torch.long, device=device
        )
        # The topology of a tree that proposes nothing, in the tree kernel's
        # row strides: consecutive selected indices and earlier parents give an
        # acyclic tree; topk = 1 reduces to a chain.
        self.chain_parents = torch.arange(
            -1, parent_width - 1, dtype=torch.long, device=device
        )
        self.chain_top_scores = torch.arange(
            self.num_draft_tokens - 1, dtype=torch.long, device=device
        )

    def write_roots(self, indices: torch.Tensor, bonus_tokens: torch.Tensor) -> None:
        """Trees that propose nothing: the sampled token, then zero drafts that
        the verify rejects. What a request carries out of prefill, before the
        last stage has drafted for it."""
        rows = torch.zeros(
            (indices.shape[0], self.num_draft_tokens),
            dtype=torch.int64,
            device=self.tokens.device,
        )
        rows[:, 0] = bonus_tokens
        self.tokens[indices] = rows
        self.parents[indices] = self.chain_parents
        self.top_scores[indices] = self.chain_top_scores

    def write_trees(
        self,
        indices: torch.Tensor,
        tokens: torch.Tensor,
        parents: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> None:
        """The trees the last stage drafted, with the topology their tokens
        were arranged by (data-dependent once topk > 1)."""
        self.tokens[indices] = tokens.to(torch.int64).reshape(-1, self.num_draft_tokens)
        self.parents[indices] = parents.to(torch.long)
        self.top_scores[indices] = top_scores.to(torch.long)

    def gather(
        self, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The trees of these slots: tokens, parents, top scores."""
        return self.tokens[indices], self.parents[indices], self.top_scores[indices]
