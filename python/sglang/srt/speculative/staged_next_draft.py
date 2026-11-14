"""The next round's draft inputs staged one round early (SGLANG_OPT_SPEC_STAGED_NEXT_DRAFT).

The post-verify graph ends in prepare_draft_window, which writes the next round's draft
lengths, positions (and M-RoPE rows) and the draft steps' write slots from this round's
new lengths into the draft graph runner's static inputs, and the round's write window into
its slot's own buffer; the post-verify slots alternate per round, so a window stays valid
through the round that takes it. A decode round takes the staging when its batch is the
previous decode round's, at a size the draft graph runs unpadded, with every request's next
window already allocated: prepare_for_draft then skips its slot / position launch and the
draft graph runner its copies of those inputs.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.kernels.ops.speculative.cache_locs import prepare_draft_window

logger = logging.getLogger(__name__)

# decode rounds between two log lines of the staging counts
LOG_EVERY = 2000
# the draft graph runner's inputs the post-verify graph writes; with the request slots a
# staged round keeps, they stay out of the runners' shared input pool
STAGED_INPUTS = frozenset(
    {"seq_lens", "out_cache_loc", "positions", "mrope_positions", "req_pool_indices"}
)


@dataclass(frozen=True, eq=False)
class NextDraft:
    """The next round's draft inputs for ``num_requests`` rows: views of the draft graph
    runner's int64 lengths, positions, M-RoPE rows ([3, num_requests] or None) and draft
    write slots [num_requests * num_steps], and the slot's own write window
    [num_requests * window]."""

    num_requests: int
    seq_lens: torch.Tensor
    positions: torch.Tensor
    mrope: Optional[torch.Tensor]
    draft_out_cache_loc: torch.Tensor
    window: torch.Tensor


def make_next_draft(
    *, num_requests: int, num_steps: int, window: int, mrope: bool, runner_buffers: Any
) -> NextDraft:
    rb = runner_buffers
    return NextDraft(
        num_requests=num_requests,
        seq_lens=rb.seq_lens[:num_requests],
        positions=rb.positions[:num_requests],
        mrope=rb.mrope_positions[:, :num_requests] if mrope else None,
        draft_out_cache_loc=rb.out_cache_loc[: num_requests * num_steps],
        window=torch.zeros(
            num_requests * window, dtype=torch.int64, device=rb.seq_lens.device
        ),
    )


def stage_next_draft(
    next_draft: NextDraft,
    *,
    req_pool_indices: torch.Tensor,
    new_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    num_steps: int,
    window: int,
    seq_len_fill_value: int,
) -> None:
    prepare_draft_window(
        req_pool_indices,
        new_seq_lens,
        req_to_token,
        num_steps=num_steps,
        window=window,
        seq_len_fill_value=seq_len_fill_value,
        seq_lens_out=next_draft.seq_lens,
        positions=next_draft.positions,
        mrope=next_draft.mrope,
        draft_out_cache_loc=next_draft.draft_out_cache_loc,
        write_virtual=next_draft.window,
    )


def takes_staging(
    staged: NextDraft, *, num_requests: int, plans_window: bool, with_mrope: bool
) -> bool:
    """Whether a draft takes the staging instead of launching its own slots and
    positions: the staging is this batch's size, the round plans its window, and the
    M-RoPE choice matches (a batch with multimodal inputs computes its rows itself)."""
    return (
        plans_window
        and staged.num_requests == num_requests
        and (staged.mrope is not None) == with_mrope
    )


class StagedDraftTracker:
    """Whether a decode round may take what the previous decode round's post-verify graph
    staged for it, and how many rounds did (logged every LOG_EVERY decode rounds)."""

    def __init__(self, *, width: int, graph_sizes):
        self.width = width
        self.graph_sizes = frozenset(graph_sizes or ())
        self.noted: Optional[tuple] = None
        self.counts = collections.Counter()
        self.decode_round_ct = 0
        # SGLANG_OPT_SPEC_STAGED_SEQ_LENS: staged rounds that took their lengths
        self.seq_lens_taken_ct = 0

    def note_extend(
        self,
        batch,
        next_draft: Optional[NextDraft],
        new_seq_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """At a decode round's end: the request slots in order, whether the draft graph
        runs that size unpadded, whether every request holds the next round's whole window
        (kv_allocated_len covers kv_committed_len plus three windows: the two rounds whose
        acceptance the host has not read yet, then the window), the staging, and the
        lengths the round published (SGLANG_OPT_SPEC_STAGED_SEQ_LENS)."""
        reqs = batch.reqs
        need = 3 * self.width
        self.noted = (
            tuple(r.kv.req_pool_idx for r in reqs),
            len(reqs) in self.graph_sizes,
            all(r.kv.kv_allocated_len >= r.kv.kv_committed_len + need for r in reqs),
            next_draft,
            new_seq_lens,
        )

    def forget(self) -> None:
        """A round that is not a decode round (prefill, mixed) stages nothing."""
        self.noted = None

    def _reason(self, noted, batch) -> str:
        if noted is None:
            return "no_previous_decode"
        if noted[0] != tuple(r.kv.req_pool_idx for r in batch.reqs):
            return "batch_changed"
        if not noted[1]:
            return "padded"
        if not noted[2]:
            return "window_not_allocated"
        if noted[3] is None:
            return "not_staged_by_post_verify_graph"
        return "staged"

    def take_seq_lens(self, batch) -> Optional[torch.Tensor]:
        """Before a decode round's resolve, without consuming the note: the lengths the
        previous decode round published when this round takes its staging, else None."""
        noted = self.noted
        if noted is None or noted[4] is None or self._reason(noted, batch) != "staged":
            return None
        self.seq_lens_taken_ct += 1
        return noted[4]

    def count_draft(self, batch) -> Optional[NextDraft]:
        """At a decode round's start: the staging when it holds for this batch, else
        None; counts the first reason it does not."""
        noted, self.noted = self.noted, None
        reason = self._reason(noted, batch)
        self.counts[reason] += 1
        self.decode_round_ct += 1
        if self.decode_round_ct % LOG_EVERY == 0:
            logger.info(
                "Staged next draft: %d decode rounds: %s; lengths taken %d",
                self.decode_round_ct,
                ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items())),
                self.seq_lens_taken_ct,
            )
        return noted[3] if reason == "staged" else None
