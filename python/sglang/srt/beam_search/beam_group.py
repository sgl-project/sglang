"""Per-request beam search state: frontier, completed pool, lifecycle.

BeamGroup is the CPU-side coordination object of one beam request. It owns
no batch rows and no KV: members are plain requests; the group only holds
search state and consumes SelectResult objects produced by joint_select.

Sync discipline: advance()/advance_final() are the designated sync points --
they read the (small, fixed-shape) result tensors to CPU to append history
nodes and maintain the completed pool. Under overlap this consumption moves
off the launch path and replays from an event stream; the frontier tensor
itself always stays on device (it is simply the new_cum_logprobs of the last
step, already gathered).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from sglang.srt.beam_search.history import BeamNode, materialize_tokens
from sglang.srt.beam_search.joint_select import FinalSelect, SelectResult


class BeamGroupState(enum.Enum):
    DECODING = enum.auto()
    FINISHED = enum.auto()


@dataclass
class CompletedBeam:
    """A finished candidate: leaf node + score inputs + finish reason."""

    leaf: Optional[BeamNode]
    cum_logprob: float
    num_tokens: int
    matched_token: Optional[int]  # stop token that ended it; None = length / cutoff


@dataclass
class BeamResult:
    """One output sequence of a finished group."""

    tokens: List[int]
    cum_logprob: float
    beam_score: float
    matched_token: Optional[int]


class BeamGroup:
    """State machine: one prefill selection, then decode selections, then finalize.

    The frontier starts as a single pseudo-row (the prompt, cum_logprob 0), so
    the prefill selection is just joint_select with num_rows=1.
    """

    def __init__(
        self,
        *,
        beam_width: int,
        length_penalty: float = 1.0,
        stop_token_ids: Sequence[int] = (),
        max_new_tokens: int,
        device: torch.device | str = "cpu",
    ):
        self.beam_width = beam_width
        self.num_candidates = 2 * beam_width
        self.length_penalty = length_penalty
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids = torch.tensor(
            sorted(stop_token_ids), dtype=torch.int64, device=device
        )

        self.frontier_cum_logprobs = torch.zeros(1, dtype=torch.float32, device=device)
        self.leaves: List[Optional[BeamNode]] = [None]  # parents of the next tokens
        self.num_generated = 0
        self.completed: List[CompletedBeam] = []
        self.state = BeamGroupState.DECODING

    # ==================== step consumption (sync points) ====================

    def next_step_is_final(self) -> bool:
        """The upcoming selection hits max_new_tokens (decided host-side)."""
        return self.num_generated + 1 >= self.max_new_tokens

    def advance(self, sel: SelectResult) -> bool:
        """Consume one joint_select result; returns True if the group finished."""
        assert self.state == BeamGroupState.DECODING
        num_survivors = int(sel.num_survivors)
        num_finished = int(sel.num_finished)
        new_len = self.num_generated + 1

        fin_tokens = sel.fin_tokens[:num_finished].tolist()
        fin_parents = sel.fin_parent_idx[:num_finished].tolist()
        fin_cums = sel.fin_cum_logprobs[:num_finished].tolist()
        for token, parent, cum in zip(fin_tokens, fin_parents, fin_cums):
            leaf = BeamNode(token, self.leaves[parent])
            self.completed.append(
                CompletedBeam(leaf, cum, new_len, matched_token=token)
            )

        surv_tokens = sel.next_tokens[:num_survivors].tolist()
        surv_parents = sel.parent_idx[:num_survivors].tolist()
        self.leaves = [
            BeamNode(token, self.leaves[parent])
            for token, parent in zip(surv_tokens, surv_parents)
        ]
        self.frontier_cum_logprobs = sel.new_cum_logprobs
        self.num_generated = new_len

        if num_survivors < self.beam_width:
            # Not enough live beams to continue: fold the partial frontier into
            # the pool (unfinished, scored at current length) and finish.
            surv_cums = sel.new_cum_logprobs[:num_survivors].tolist()
            for leaf, cum in zip(self.leaves, surv_cums):
                self.completed.append(
                    CompletedBeam(leaf, cum, new_len, matched_token=None)
                )
            self.leaves = []
            self.state = BeamGroupState.FINISHED
            return True
        return False

    def advance_final(self, sel: FinalSelect) -> bool:
        """Consume a length-terminated select_final_topk result; always finishes."""
        assert self.state == BeamGroupState.DECODING
        new_len = self.num_generated + 1
        tokens = sel.tokens.tolist()
        parents = sel.parent_idx.tolist()
        cums = sel.cum_logprobs.tolist()
        for token, parent, cum in zip(tokens, parents, cums):
            leaf = BeamNode(token, self.leaves[parent])
            self.completed.append(CompletedBeam(leaf, cum, new_len, matched_token=None))
        self.leaves = []
        self.num_generated = new_len
        self.state = BeamGroupState.FINISHED
        return True

    # ==================== finalize ====================

    def beam_score(self, cum_logprob: float, num_tokens: int) -> float:
        """Length-normalized score: cum_logprob / num_tokens ** length_penalty."""
        return cum_logprob / (num_tokens**self.length_penalty)

    def finalize(self) -> List[BeamResult]:
        """Materialize the top beam_width sequences, best score first."""
        assert self.state == BeamGroupState.FINISHED
        results = [
            BeamResult(
                tokens=materialize_tokens(beam.leaf),
                cum_logprob=beam.cum_logprob,
                beam_score=self.beam_score(beam.cum_logprob, beam.num_tokens),
                matched_token=beam.matched_token,
            )
            for beam in self.completed
        ]
        results.sort(key=lambda r: r.beam_score, reverse=True)
        return results[: self.beam_width]
