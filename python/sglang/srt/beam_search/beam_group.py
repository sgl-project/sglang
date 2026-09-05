# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Per-request beam search state: frontier, completed pool, lifecycle.

BeamGroup holds search state and consumes joint_select results. Members are
not requests: the group tracks them columnarly as req_to_token rows
(member_rows) that decode in lockstep with the leader row.

advance_*_frontier / commit_pending are this class's halves of the overlap
split documented in coordinator.py.
"""

from __future__ import annotations

import enum
from typing import List, Optional, Sequence

import msgspec
import torch

from sglang.srt.beam_search.fork import StagedOrphans
from sglang.srt.beam_search.history import BeamNode, materialize_tokens
from sglang.srt.beam_search.joint_select import FinalSelect, SelectResult


class BeamGroupState(enum.Enum):
    DECODING = enum.auto()
    FINISHED = enum.auto()


class CompletedBeam(msgspec.Struct):
    """A finished candidate: leaf node + score inputs + finish reason."""

    leaf: Optional[BeamNode]
    cum_logprob: float
    num_tokens: int
    matched_token: Optional[int]  # stop token that ended it; None = length / cutoff


class BeamResult(msgspec.Struct):
    """One output sequence of a finished group."""

    tokens: List[int]
    cum_logprob: float
    beam_score: float
    matched_token: Optional[int]


class BeamGroup:
    """State machine: one prefill selection, then decode selections, then
    finalize. The frontier starts as one pseudo-row (the prompt, cum_logprob 0)."""

    def __init__(
        self,
        *,
        beam_width: int,
        length_penalty: float = 1.0,
        stop_token_ids: Sequence[int] = (),
        max_new_tokens: int,
        num_return: Optional[int] = None,
        device: torch.device | str = "cpu",
    ):
        self.beam_width = beam_width
        self.num_candidates = 2 * beam_width
        self.length_penalty = length_penalty
        self.max_new_tokens = max_new_tokens
        self.num_return = num_return if num_return is not None else beam_width
        self.stop_token_ids = torch.tensor(
            sorted(stop_token_ids), dtype=torch.int64, device=device
        )

        self.frontier_cum_logprobs = torch.zeros(1, dtype=torch.float32, device=device)
        self.leaves: List[Optional[BeamNode]] = [None]  # parents of the next tokens
        # num_generated is the launch half's count, num_committed the deferred
        # half's (the true length); generated may lead by one under overlap.
        self.num_generated = 0
        self.num_committed = 0
        self.completed: List[CompletedBeam] = []
        self.state = BeamGroupState.DECODING
        # Selection results staged by the launch half as (forward tick, sel),
        # consumed in tick order by commit.
        self._pending_steps: List[tuple] = []
        # Set once by the coordinator when the group leaves the live set
        # (finish / abort / dead-leader); guards double bookkeeping.
        self.retired = False

        # Scheduler wiring, filled in from outside the search core. Members have
        # no Req, so their seq len and KV lengths are implied by the leader's.
        self.leader = None
        # Device [k-1] member row indices, and the same rows on host (for
        # row-slot free). None until the post-prefill spawn / after free.
        self.member_rows: Optional[torch.Tensor] = None
        self.member_rows_cpu: Optional[torch.Tensor] = None
        # Device [k]: leader row first, then member_rows (frontier-row order).
        self.all_rows: Optional[torch.Tensor] = None
        # Staged by the launch half; the deferred half frees them, gated on the
        # tick whose copy_done sync already happened.
        self.pending_orphans: List[StagedOrphans] = []
        # Running total the GC has returned, so held KV is a host-side
        # arithmetic (allocated - freed) rather than a tensor read.
        self.slots_freed = 0

    @property
    def num_member_rows(self) -> int:
        return 0 if self.member_rows is None else self.member_rows.shape[0]

    def extra_uncached_tokens(self) -> int:
        """Uncached KV the group holds beyond the leader's own window, which the
        generic per-req sum already counts."""
        if self.all_rows is None:
            return 0
        end = self.leader.kv.kv_allocated_len
        start = self.prompt_len
        # Host-side arithmetic, not distinct slots off req_to_token: that would
        # read the launch half's staged tensors, unsafe on the checker's stream.
        held = self.beam_width * (end - start) - self.slots_freed
        return held - (end - start)

    def next_step_is_final(self) -> bool:
        """The upcoming selection hits max_new_tokens (decided host-side)."""
        return self.num_generated + 1 >= self.max_new_tokens

    def advance_frontier(self, sel: SelectResult, tick: int = 0) -> None:
        """Launch half of one selection step: evolve the frontier tensor and
        stage the result for commit, stamped with its forward tick."""
        assert self.state == BeamGroupState.DECODING
        self.frontier_cum_logprobs = sel.new_cum_logprobs
        self.num_generated += 1
        self._pending_steps.append((tick, sel))

    def advance_final_frontier(self, sel: FinalSelect, tick: int = 0) -> None:
        """Launch half of a length-terminated step: stage only (the final step
        needs no next frontier)."""
        assert self.state == BeamGroupState.DECODING
        self.num_generated += 1
        self._pending_steps.append((tick, sel))

    def commit_pending(self, up_to_tick: Optional[int] = None) -> bool:
        """Deferred half: consume staged selections into the DAG (the D2H sync
        point). Returns True when this commit finishes the group."""
        if self.state == BeamGroupState.FINISHED:
            self._pending_steps.clear()
            return False
        while self._pending_steps:
            tick, sel = self._pending_steps[0]
            # Tick gate: copy_done covers only kernels enqueued up to that
            # forward, so a later-staged step may not be readable yet.
            if up_to_tick is not None and tick > up_to_tick:
                break
            self._pending_steps.pop(0)
            finished = (
                self._commit_final(sel)
                if isinstance(sel, FinalSelect)
                else self._commit_step(sel)
            )
            if finished:
                self._pending_steps.clear()
                return True
        return False

    def _commit_step(self, sel: SelectResult) -> bool:
        num_survivors = int(sel.num_survivors)
        num_finished = int(sel.num_finished)
        new_len = self.num_committed + 1

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
        self.num_committed = new_len

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

    def _commit_final(self, sel: FinalSelect) -> bool:
        new_len = self.num_committed + 1
        tokens = sel.tokens.tolist()
        parents = sel.parent_idx.tolist()
        cums = sel.cum_logprobs.tolist()
        # A parent outside the committed frontier means a tick-gating bug let an
        # unsynchronized step through; fail rather than build a corrupt DAG.
        assert not parents or max(parents) < len(self.leaves), (
            "beam commit consumed an unsynced or misordered step"
        )
        for token, parent, cum in zip(tokens, parents, cums):
            leaf = BeamNode(token, self.leaves[parent])
            self.completed.append(CompletedBeam(leaf, cum, new_len, matched_token=None))
        self.leaves = []
        self.num_committed = new_len
        self.state = BeamGroupState.FINISHED
        return True

    # Sync wrappers: both halves back-to-back. UT-only -- the scheduler path
    # drives the two halves separately so they can straddle a forward.

    def advance(self, sel: SelectResult) -> bool:
        """Consume one joint_select result; returns True if the group finished."""
        self.advance_frontier(sel)
        return self.commit_pending()

    def advance_final(self, sel: FinalSelect) -> bool:
        """Consume a length-terminated select_final_topk result; always finishes."""
        self.advance_final_frontier(sel)
        return self.commit_pending()

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
