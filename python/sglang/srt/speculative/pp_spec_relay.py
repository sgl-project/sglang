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
        *,
        speculative_num_steps: Optional[int] = None,
    ):
        super().__init__(SpecInputType.PP_SPEC_RELAY)
        if tokens.ndim != 2:
            raise ValueError(
                f"PP speculative relay tokens must be 2-D, got shape={tokens.shape}"
            )
        if tokens.shape[0] != len(rids):
            raise ValueError(
                "PP speculative relay row count must match rids: "
                f"rows={tokens.shape[0]}, rids={len(rids)}"
            )
        if tokens.shape[1] < 1:
            raise ValueError("PP speculative relay must contain a root token")
        if (parents is None) != (top_scores is None):
            raise ValueError(
                "PP speculative topology requires both parents and top_scores"
            )
        for name, value in (("parents", parents), ("top_scores", top_scores)):
            if value is not None and (value.ndim != 2 or value.shape[0] != len(rids)):
                raise ValueError(
                    f"PP speculative {name} rows must match rids: "
                    f"shape={tuple(value.shape)}, rids={len(rids)}"
                )

        if speculative_num_steps is None:
            # PP adaptive currently supports topk=1, where width = steps + 1.
            speculative_num_steps = tokens.shape[1] - 1
        if speculative_num_steps < 0:
            raise ValueError("speculative_num_steps must be non-negative")

        # [bs, logical_width], column 0 is the bonus token.
        self.rids = list(rids)
        self.tokens = tokens
        self.speculative_num_steps = speculative_num_steps
        # parent_list / top_scores_index, [bs, *]. None until the request has
        # been drafted for: its first decode after prefill carries zero drafts,
        # which are rejected whatever tree shape they hang on.
        self.parents = parents
        self.top_scores = top_scores

    def __repr__(self) -> str:
        return (
            "PPSpecRelayInput("
            f"bs={len(self.rids)}, steps={self.speculative_num_steps}, "
            f"width={self.speculative_num_draft_tokens}, "
            f"drafted={self.parents is not None})"
        )

    @classmethod
    def degenerate(
        cls,
        rids: List[str],
        bonus_tokens: torch.Tensor,
        num_draft_tokens: int,
        *,
        speculative_num_steps: Optional[int] = None,
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
        return cls(
            rids=list(rids),
            tokens=tokens,
            speculative_num_steps=(
                num_draft_tokens - 1
                if speculative_num_steps is None
                else speculative_num_steps
            ),
        )

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
            self.speculative_num_steps = other.speculative_num_steps
            return
        if self.configuration != other.configuration:
            if other.parents is not None:
                raise ValueError(
                    "Cannot merge drafted PP speculative relays with different "
                    f"configurations: current={self.configuration}, "
                    f"other={other.configuration}"
                )
            # A just-prefilled batch only owns its sampled root. Re-encode
            # those rows as degenerate proposals under the running
            # microbatch's configuration before continuous-batch merge.
            other = PPSpecRelayInput.degenerate(
                rids=other.rids,
                bonus_tokens=other.tokens[:, 0],
                num_draft_tokens=self.speculative_num_draft_tokens,
                speculative_num_steps=self.speculative_num_steps,
            )
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
        is_transition = self.configuration != relayed.configuration
        if is_transition:
            # Requests merged after the completed forward are absent from the
            # relay. During a configuration transition retain their root token
            # but discard old-width draft claims.
            kept_tokens = torch.zeros(
                (len(self.rids), relayed.speculative_num_draft_tokens),
                dtype=self.tokens.dtype,
                device=self.tokens.device,
            )
            kept_tokens[:, 0] = self.tokens[:, 0]
        else:
            kept_tokens = self.tokens
        self.tokens = torch.where(keep.unsqueeze(1), kept_tokens, relayed_tokens)
        self.speculative_num_steps = relayed.speculative_num_steps
        if relayed.parents is None:
            if is_transition:
                self.parents = self.top_scores = None
            return
        widths = relayed._widths()
        if is_transition:
            kept_parents = self._chain_parents(widths[0])
            kept_top_scores = self._chain_top_scores(widths[1])
        else:
            kept_parents = self._parents_or_chain(widths)
            kept_top_scores = self._top_scores_or_chain(widths)
        self.parents = torch.where(
            keep.unsqueeze(1),
            kept_parents,
            relayed.parents.to(self.tokens.device)[take],
        )
        self.top_scores = torch.where(
            keep.unsqueeze(1),
            kept_top_scores,
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
            speculative_num_steps=self.speculative_num_steps,
        )

    @property
    def speculative_num_draft_tokens(self) -> int:
        return self.tokens.shape[1]

    @property
    def configuration(self) -> tuple[int, int]:
        return (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
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
        return self._chain_parents(widths[0])

    def _chain_parents(self, width: int) -> torch.Tensor:
        return torch.arange(
            -1, width - 1, dtype=torch.long, device=self.tokens.device
        ).repeat(len(self.rids), 1)

    def _top_scores_or_chain(self, widths) -> torch.Tensor:
        if self.top_scores is not None:
            return self.top_scores
        return self._chain_top_scores(widths[1])

    def _chain_top_scores(self, width: int) -> torch.Tensor:
        return torch.arange(width, dtype=torch.long, device=self.tokens.device).repeat(
            len(self.rids), 1
        )
