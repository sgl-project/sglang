from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


class PPSpecRelayInput(SpecInput):
    """The draft tree the last PP stage produced, carried by the requests it
    belongs to so every stage can rebuild the same verify input.

    Under PP the draft model lives on the last stage only, so the other stages
    receive the tree over the output relay instead of drafting it. It has to
    survive between rounds and across microbatch recomposition, which is why
    it rides on ``ScheduleBatch.spec_info`` and implements the filter / merge
    hooks rather than living in a side table: a request that finishes, gets
    retracted, or is merged in from a just-finished prefill carries its own
    row along with it.

    Rows are per request and aligned with ``batch.reqs``. ``rids`` is kept
    alongside them because the relayed tensors are sized by the composition
    that ran the forward, which can differ from the live batch by the time
    the result comes back around the ring.

    This is the algorithm-agnostic half of the PP relay: the tokens plus the
    topology they were arranged by. A speculative algorithm whose proposal is
    not an EAGLE-style tree can subclass it (or mirror it) and only has to
    supply its own rebuild.
    """

    def __init__(
        self,
        rids: List[str],
        tokens: torch.Tensor,
        parents: Optional[torch.Tensor] = None,
        top_scores: Optional[torch.Tensor] = None,
    ):
        super().__init__(SpecInputType.PP_SPEC_RELAY)
        # [bs, num_draft_tokens], column 0 is the bonus token
        self.rids = rids
        self.tokens = tokens
        # parent_list / top_scores_index, [bs, *]. None until the request has
        # been drafted for: its first decode after prefill carries zero drafts,
        # which are rejected whatever tree shape they hang on.
        self.parents = parents
        self.top_scores = top_scores

    def __repr__(self) -> str:
        return (
            f"PPSpecRelayInput(bs={len(self.rids)}, "
            f"drafted={self.parents is not None})"
        )

    @classmethod
    def degenerate(
        cls, rids: List[str], bonus_tokens: torch.Tensor, num_draft_tokens: int
    ) -> PPSpecRelayInput:
        """A tree that proposes nothing: just the sampled token, padded with
        zeros. What a request carries out of prefill, before the last stage
        has drafted for it."""
        tokens = torch.zeros(
            (len(rids), num_draft_tokens),
            dtype=torch.int64,
            device=bonus_tokens.device,
        )
        tokens[:, 0] = bonus_tokens.to(torch.int64)
        return cls(rids=list(rids), tokens=tokens)

    def filter_batch(
        self, new_indices: torch.Tensor, new_indices_cpu: Optional[List[int]] = None
    ) -> None:
        keep = new_indices_cpu if new_indices_cpu is not None else new_indices.tolist()
        self.rids = [self.rids[i] for i in keep]
        self.tokens = self.tokens[new_indices]
        if self.parents is not None:
            self.parents = self.parents[new_indices]
            self.top_scores = self.top_scores[new_indices]

    def merge_batch(self, other: PPSpecRelayInput) -> None:
        if not other.rids:
            return
        if not self.rids:
            self.rids, self.tokens = list(other.rids), other.tokens
            self.parents, self.top_scores = other.parents, other.top_scores
            return
        self.rids = self.rids + list(other.rids)
        self.tokens = torch.cat([self.tokens, other.tokens])
        # A batch merging in from prefill has no topology yet; give it the
        # other side's widths so the rows stay stackable.
        left, right = self._widths(), other._widths()
        widths = left if left is not None else right
        if widths is None:
            self.parents = self.top_scores = None
            return
        self.parents = torch.cat(
            [self._parents_or_chain(widths), other._parents_or_chain(widths)]
        )
        self.top_scores = torch.cat(
            [self._top_scores_or_chain(widths), other._top_scores_or_chain(widths)]
        )

    def adopt(self, relayed: PPSpecRelayInput) -> None:
        """Take the relayed rows for the requests they cover, in this input's
        order, keeping the current row for any request the relay does not
        mention (one merged in after the forward was launched)."""
        by_rid = {rid: i for i, rid in enumerate(relayed.rids)}
        rows = [by_rid.get(rid) for rid in self.rids]
        if all(r is None for r in rows):
            return
        take = torch.tensor(
            [r if r is not None else 0 for r in rows],
            dtype=torch.long,
            device=relayed.tokens.device,
        )
        keep = torch.tensor(
            [r is None for r in rows], dtype=torch.bool, device=self.tokens.device
        )
        relayed_tokens = relayed.tokens.to(self.tokens.device)[take]
        self.tokens = torch.where(keep.unsqueeze(1), self.tokens, relayed_tokens)
        if relayed.parents is None:
            return
        widths = relayed._widths()
        self.parents = torch.where(
            keep.unsqueeze(1),
            self._parents_or_chain(widths),
            relayed.parents.to(self.tokens.device)[take],
        )
        self.top_scores = torch.where(
            keep.unsqueeze(1),
            self._top_scores_or_chain(widths),
            relayed.top_scores.to(self.tokens.device)[take],
        )

    def reindex(self, rids: List[str]) -> PPSpecRelayInput:
        """This input's rows in another composition's order. Every rid must be
        covered -- the caller is relabelling the same requests, not adding."""
        by_rid = {rid: i for i, rid in enumerate(self.rids)}
        take = torch.tensor(
            [by_rid[rid] for rid in rids], dtype=torch.long, device=self.tokens.device
        )
        return PPSpecRelayInput(
            rids=list(rids),
            tokens=self.tokens[take],
            parents=None if self.parents is None else self.parents[take],
            top_scores=None if self.top_scores is None else self.top_scores[take],
        )

    def topology(self, *, fallback):
        """The rows' tree shape, as a rectangular pair. ``fallback`` supplies
        chain constants for the case where no request has been drafted for
        yet, since their width comes from the spec config, not from a row."""
        widths = self._widths()
        if widths is None:
            return fallback()
        return self._parents_or_chain(widths), self._top_scores_or_chain(widths)

    def _widths(self):
        if self.parents is None:
            return None
        return self.parents.shape[1], self.top_scores.shape[1]

    def _parents_or_chain(self, widths) -> torch.Tensor:
        if self.parents is not None:
            return self.parents
        width = widths[0]
        return torch.arange(
            -1, width - 1, dtype=torch.long, device=self.tokens.device
        ).repeat(len(self.rids), 1)

    def _top_scores_or_chain(self, widths) -> torch.Tensor:
        if self.top_scores is not None:
            return self.top_scores
        width = widths[1]
        return torch.arange(width, dtype=torch.long, device=self.tokens.device).repeat(
            len(self.rids), 1
        )
