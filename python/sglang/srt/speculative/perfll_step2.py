"""PERF-LL step 2 (SGLANG_PERFLL_STEP2): the next round's draft inputs staged one round early.

Mode 1 counts only. At the end of a decode pass it notes what the staging would need -- the
batch's request slots in order, whether the draft graph runs that size unpadded, and whether
every request already holds the next round's whole window (kv_allocated_len covers the host's
kv_committed_len plus three verify windows: the two rounds whose acceptance the host has not
read yet, then the window itself). The next decode pass says whether its draft inputs could
have come from there (``staged``) or why not, in one log line every LOG_EVERY passes.

Mode 2 also stages them: the post-verify graph ends in PERF-ST's E2 (prepare_draft_window),
which writes the next round's draft inputs and write window from this round's new lengths
into its slot's NextDraft buffers; a staged pass that mode 3 would take (takes_staging)
compares them with what its eager path computes (device-side mismatch counts, read only when
the line is logged); the others count as declined.

Mode 3 uses them: E2 writes the draft fields straight into the draft graph runner's static
inputs (the window stays in the slot, whose buffers alternate per round), and a staged pass
skips prepare_for_draft's slot / position launch and the runner's copies of those fields.

With SGLANG_PERFLL_STEP5A_SEQ_LENS a staged pass also takes batch.seq_lens from the previous
pass's post-verify output (take_seq_lens): the same request slots in the same order read back
exactly what that pass published, so the schedule stream's gather is skipped. The output lives
in its post-verify slot until the slot's next replay, two rounds later.
"""

from __future__ import annotations

import collections
import logging
import os
import time
from typing import Optional

import msgspec
import torch

logger = logging.getLogger(__name__)

LOG_EVERY = 2000


class NextDraft(msgspec.Struct):
    """The next round's draft inputs for ``num_requests`` rows: int64 lengths, positions,
    M-RoPE rows ([3, num_requests] or None), the draft steps' write slots
    [num_requests * num_steps] and the round's write window [num_requests * window]."""

    num_requests: int
    seq_lens: torch.Tensor
    positions: torch.Tensor
    mrope: Optional[torch.Tensor]
    draft_out_cache_loc: torch.Tensor
    window: torch.Tensor


def make_next_draft(
    *,
    num_requests: int,
    num_steps: int,
    window: int,
    mrope: bool,
    device,
    runner_buffers=None,
) -> NextDraft:
    """Own buffers (mode 2), or with ``runner_buffers`` (mode 3) views of the draft graph
    runner's static inputs for the draft fields; the window is always the slot's own."""

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.int64, device=device)

    if runner_buffers is None:
        return NextDraft(
            num_requests=num_requests,
            seq_lens=zeros(num_requests),
            positions=zeros(num_requests),
            mrope=zeros(3, num_requests) if mrope else None,
            draft_out_cache_loc=zeros(num_requests * num_steps),
            window=zeros(num_requests * window),
        )
    rb = runner_buffers
    return NextDraft(
        num_requests=num_requests,
        seq_lens=rb.seq_lens[:num_requests],
        positions=rb.positions[:num_requests],
        mrope=rb.mrope_positions[:, :num_requests] if mrope else None,
        draft_out_cache_loc=rb.out_cache_loc[: num_requests * num_steps],
        window=zeros(num_requests * window),
    )


class DraftExtendInputs(msgspec.Struct, frozen=True):
    """SGLANG_PERFLL_RS_D1: views of the draft extend graph runner's static inputs for one batch
    size (``num_tokens`` = rows x width), which the post-verify graph writes in place, and the
    FutureMap's new-lengths buffer (int64, by request slot) it publishes to."""

    input_ids: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    mrope_positions: Optional[torch.Tensor]
    out_cache_loc: torch.Tensor
    req_pool_indices: torch.Tensor
    extend_seq_lens: torch.Tensor
    num_correct_drafts: torch.Tensor
    num_accept_tokens: torch.Tensor
    select_index: torch.Tensor
    mamba_track_indices: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    published_seq_lens: torch.Tensor


# the draft extend runner's inputs the post-verify graph writes (SGLANG_PERFLL_RS_D1), kept out of
# the shared input pool
DRAFT_EXTEND_STAGED_INPUTS = frozenset(
    {
        "input_ids",
        "seq_lens",
        "positions",
        "mrope_positions",
        "out_cache_loc",
        "req_pool_indices",
        "extend_seq_lens",
        "num_correct_drafts",
        "num_accept_tokens",
        "mamba_track_indices",
        "hidden_states",
    }
)


def make_draft_extend_inputs(
    *,
    runner_buffers,
    num_requests: int,
    width: int,
    mrope: bool,
    published_seq_lens: torch.Tensor,
) -> DraftExtendInputs:
    rb, num_tokens = runner_buffers, num_requests * width
    return DraftExtendInputs(
        input_ids=rb.input_ids[:num_tokens],
        seq_lens=rb.seq_lens[:num_requests],
        positions=rb.positions[:num_tokens],
        mrope_positions=rb.mrope_positions[:, :num_tokens] if mrope else None,
        out_cache_loc=rb.out_cache_loc[:num_tokens],
        req_pool_indices=rb.req_pool_indices[:num_requests],
        extend_seq_lens=rb.extend_seq_lens[:num_requests],
        num_correct_drafts=rb.num_correct_drafts[:num_requests],
        num_accept_tokens=rb.num_accept_tokens[:num_requests],
        select_index=rb.select_index[:num_requests],
        mamba_track_indices=(
            None
            if rb.mamba_track_indices is None
            else rb.mamba_track_indices[:num_requests]
        ),
        hidden_states=rb.hidden_states[:num_tokens],
        published_seq_lens=published_seq_lens,
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
    from sglang.kernels.ops.speculative.cache_locs import prepare_draft_window

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
    """Whether a draft takes this staging (mode 3) instead of launching its own slots and
    positions: the staging is this batch's size, the iteration plans its window, and the
    M-RoPE choice matches (a batch with multimodal inputs computes its M-RoPE rows itself)."""
    return (
        plans_window
        and staged.num_requests == num_requests
        and (staged.mrope is not None) == with_mrope
    )


class Step2Check:
    """Device-side counts of staged passes whose staged inputs differ from the eager ones, per
    field and per batch size, and the first difference seen (field, row-major position, batch
    size, staged value, eager value); read only when summarized."""

    FIELDS = ("seq_lens", "positions", "mrope", "draft_out_cache_loc", "window")

    def __init__(self, device):
        self.device = device
        self.mismatch_by_bs: dict[int, torch.Tensor] = {}
        self.check_ct_by_bs: collections.Counter = collections.Counter()
        # staged passes a draft would not take (takes_staging), not compared
        self.declined_ct = 0
        self.shape_mismatch_ct = 0
        # field, position, batch size, staged value, eager value; field -1: none yet
        self.first = torch.tensor([-1, -1, 0, 0, 0], dtype=torch.int64, device=device)

    def compare(
        self,
        staged: NextDraft,
        *,
        seq_lens: torch.Tensor,
        positions: torch.Tensor,
        mrope: Optional[torch.Tensor],
        window: Optional[torch.Tensor],
        num_steps: int,
    ) -> None:
        bs = staged.num_requests
        counts = self.mismatch_by_bs.get(bs)
        if counts is None:
            counts = self.mismatch_by_bs[bs] = torch.zeros(
                len(self.FIELDS), dtype=torch.int64, device=self.device
            )
        draft_locs = None
        if window is not None and window.numel() % bs == 0:
            draft_locs = window.view(bs, -1)[:, :num_steps].reshape(-1)
        eager = (seq_lens, positions, mrope, draft_locs, window)
        mine = (
            staged.seq_lens,
            staged.positions,
            staged.mrope,
            staged.draft_out_cache_loc,
            staged.window,
        )
        for i, (a, b) in enumerate(zip(mine, eager)):
            if a is None or b is None:
                continue
            if a.shape != b.shape:
                self.shape_mismatch_ct += 1
                continue
            a, b = a.reshape(-1), b.reshape(-1).to(a.dtype)
            diff = a != b
            any_diff = diff.any()
            counts[i] += any_diff
            pos = torch.argmax(diff.to(torch.int8)).view(1)
            record = torch.cat(
                [
                    torch.full_like(pos, i),
                    pos,
                    torch.full_like(pos, bs),
                    a.index_select(0, pos),
                    b.index_select(0, pos),
                ]
            )
            self.first.copy_(
                torch.where(any_diff & (self.first[0] < 0), record, self.first)
            )
        self.check_ct_by_bs[bs] += 1

    def summary(self) -> str:
        parts = []
        for bs in sorted(self.mismatch_by_bs):
            counts = self.mismatch_by_bs[bs].tolist()
            parts.append(
                f"B{bs}: compared {self.check_ct_by_bs[bs]}, mismatching "
                + " ".join(f"{name} {n}" for name, n in zip(self.FIELDS, counts))
            )
        field, pos, bs, mine, eager = self.first.tolist()
        first = (
            "none"
            if field < 0
            else f"{self.FIELDS[field]}[{pos}] at B{bs}: staged {mine}, eager {eager}"
        )
        return (
            "; ".join(parts)
            + f"; declined {self.declined_ct}; shape {self.shape_mismatch_ct}; first difference {first}"
        )


class SeqLensCheck:
    """SGLANG_DEBUG_PERFLL_STEP5A_SEQ_LENS_CHECK, run on the schedule stream behind the publish:
    per batch size, the staged passes whose post-verify lengths differ from the gather they
    replace; and at the next resolve (that pass's forward over, the slot's next replay not
    begun), whether those lengths still hold what the gather read. Counts stay on the device
    and are read every LOG_EVERY comparisons."""

    def __init__(self, device, gate: Optional[str] = None):
        self.device = device
        # with a gate file, the check runs only while the file exists
        self.gate = gate
        self.active = gate is None
        self._gate_poll = 0.0
        self.mismatch_by_bs: dict[int, torch.Tensor] = {}
        self.check_ct_by_bs: collections.Counter = collections.Counter()
        self.compare_ct = 0
        self.shape_mismatch_ct = 0
        # passes whose lengths changed before the next resolve, and how many were checked
        self.overwritten = torch.zeros((), dtype=torch.int64, device=device)
        self.retire_ct = 0
        # row, batch size, staged value, gathered value; row -1: none yet
        self.first = torch.tensor([-1, 0, 0, 0], dtype=torch.int64, device=device)
        # the last staged lengths and the gather's copy of them, until the next resolve
        self._live: Optional[tuple] = None

    def poll_gate(self) -> None:
        """Call only while the scheduler is idle: follow the gate file, at most twice a second."""
        now = time.monotonic()
        if self.gate is None or now - self._gate_poll < 0.5:
            return
        self._gate_poll = now
        on = os.path.exists(self.gate)
        if self.active and not on:
            self._live = None
            logger.info(
                "PERF-LL step5a seq_lens check (gate closed): %s", self.summary()
            )
        self.active = on

    def retire(self) -> None:
        live, self._live = self._live, None
        if live is not None:
            self.overwritten += (live[0] != live[1]).any()
            self.retire_ct += 1

    def compare(self, *, staged: torch.Tensor, gathered: torch.Tensor) -> None:
        if staged.shape != gathered.shape or staged.dtype != gathered.dtype:
            self.shape_mismatch_ct += 1
            return
        bs = staged.numel()
        counts = self.mismatch_by_bs.get(bs)
        if counts is None:
            counts = self.mismatch_by_bs[bs] = torch.zeros(
                (), dtype=torch.int64, device=self.device
            )
        diff = staged != gathered
        any_diff = diff.any()
        counts += any_diff
        pos = torch.argmax(diff.to(torch.int8)).view(1)
        record = torch.cat(
            [
                pos,
                torch.full_like(pos, bs),
                staged.index_select(0, pos),
                gathered.index_select(0, pos),
            ]
        )
        self.first.copy_(
            torch.where(any_diff & (self.first[0] < 0), record, self.first)
        )
        self.check_ct_by_bs[bs] += 1
        self._live = (staged, gathered)
        self.compare_ct += 1
        if self.compare_ct % LOG_EVERY == 0:
            logger.info("PERF-LL step5a seq_lens check: %s", self.summary())

    def summary(self) -> str:
        parts = [
            f"B{bs}: compared {self.check_ct_by_bs[bs]}, mismatching {int(n)}"
            for bs, n in sorted(self.mismatch_by_bs.items())
        ]
        row, bs, mine, gathered = self.first.tolist()
        first = (
            "none"
            if row < 0
            else f"row {row} at B{bs}: staged {mine}, gathered {gathered}"
        )
        return (
            "; ".join(parts)
            + f"; shape {self.shape_mismatch_ct}; first difference {first}"
            + f"; overwritten before the next resolve {int(self.overwritten)} of {self.retire_ct}"
        )


class DraftExtendInputsCheck:
    """SGLANG_DEBUG_PERFLL_RS_D1_CHECK_GATE: while the gate file exists, a pass whose draft extend
    takes the post-verify graph's inputs compares every one of them, and the lengths it
    published, with what the eager path builds; device-side counts per field, read when logged."""

    FIELDS = (
        "input_ids",
        "seq_lens",
        "positions",
        "mrope_positions",
        "out_cache_loc",
        "req_pool_indices",
        "extend_seq_lens",
        "num_correct_drafts",
        "num_accept_tokens",
        "select_index",
        "mamba_track_indices",
        "hidden_states",
        "published",
    )

    def __init__(self, device, gate: str):
        self.device = device
        self.gate = gate
        self.active = False
        self.pass_ct = 0
        self.compare_ct = 0
        self.mismatch = torch.zeros(len(self.FIELDS), dtype=torch.int64, device=device)
        # field index, batch size, row; -1: none yet
        self.first = torch.tensor([-1, 0, 0], dtype=torch.int64, device=device)

    def poll_gate(self) -> None:
        """Every 256th pass: follow the gate file (one stat per poll)."""
        self.pass_ct += 1
        if self.pass_ct % 256:
            return
        on = os.path.exists(self.gate)
        if self.active and not on:
            logger.info("PERF-LL rs-d1 check (gate closed): %s", self.summary())
        self.active = on

    def compare(self, field: str, staged: torch.Tensor, built: torch.Tensor) -> None:
        index = self.FIELDS.index(field)
        if staged.shape != built.shape:
            self.mismatch[index] += 1
            return
        diff = (
            (staged != built.to(staged.dtype)).reshape(staged.shape[0], -1).any(dim=1)
        )
        any_diff = diff.any()
        self.mismatch[index] += any_diff
        row = torch.argmax(diff.to(torch.int8)).view(1)
        record = torch.cat(
            [torch.full_like(row, index), torch.full_like(row, staged.shape[0]), row]
        )
        self.first.copy_(
            torch.where(any_diff & (self.first[0] < 0), record, self.first)
        )

    def counted(self) -> None:
        self.compare_ct += 1
        if self.compare_ct % LOG_EVERY == 0:
            logger.info("PERF-LL rs-d1 check: %s", self.summary())

    def summary(self) -> str:
        counts = ", ".join(
            f"{name} {int(n)}" for name, n in zip(self.FIELDS, self.mismatch.tolist())
        )
        index, bs, row = self.first.tolist()
        first = "none" if index < 0 else f"{self.FIELDS[index]} row {row} at {bs} rows"
        return f"passes compared {self.compare_ct}; mismatching {counts}; first difference {first}"


def check_draft_extend_inputs(
    check: DraftExtendInputsCheck,
    *,
    staged: DraftExtendInputs,
    width: int,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    predict: torch.Tensor,
    out_cache_loc: torch.Tensor,
    mamba_track_indices: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
) -> None:
    """The eager path's draft extend inputs for this batch, built again from the round's
    tensors and compared with what the post-verify graph wrote (``staged``). The layout is
    recomputed with torch ops into fresh tensors: prepare_draft_extend_layout's outputs come
    from SGLANG_PERFLL_HOST_OUT_POOL's rings, whose depth this extra use would outrun."""
    from sglang.kernels.ops.speculative.eagle import prepare_draft_extend_inputs

    correct, select, ids = prepare_draft_extend_inputs(accept_lens, predict, width)
    extend = torch.full_like(seq_lens, width, dtype=torch.int32)
    post = seq_lens + width
    cols = torch.arange(width, device=seq_lens.device, dtype=torch.int64)
    positions = (seq_lens.clamp(min=0).to(torch.int64)[:, None] + cols).reshape(-1)
    mrope = positions.expand(3, -1)
    pairs = [
        ("input_ids", staged.input_ids, ids),
        ("seq_lens", staged.seq_lens, post),
        ("positions", staged.positions, positions),
        ("out_cache_loc", staged.out_cache_loc, out_cache_loc),
        ("req_pool_indices", staged.req_pool_indices, req_pool_indices),
        ("extend_seq_lens", staged.extend_seq_lens, extend),
        ("num_correct_drafts", staged.num_correct_drafts, correct),
        ("num_accept_tokens", staged.num_accept_tokens, accept_lens),
        ("select_index", staged.select_index, select),
        ("hidden_states", staged.hidden_states, hidden_states),
    ]
    if staged.mrope_positions is not None:
        pairs.append(("mrope_positions", staged.mrope_positions, mrope))
    if staged.mamba_track_indices is not None:
        pairs.append(
            ("mamba_track_indices", staged.mamba_track_indices, mamba_track_indices)
        )
    for field, mine, built in pairs:
        check.compare(field, mine, built)
    check.counted()


class Step2Counter:
    def __init__(self, width: int, graph_sizes, needs_next_draft: bool = False):
        self.width = width
        self.graph_sizes = frozenset(graph_sizes or ())
        self.needs_next_draft = needs_next_draft
        self.check: Optional[Step2Check] = None
        self.noted = None
        self.counts = collections.Counter()
        self.decode_pass_ct = 0
        # SGLANG_PERFLL_STEP5A_SEQ_LENS: staged passes that took their lengths, and those whose
        # decode preparation launched work on the schedule stream (the forward waits for it)
        self.seq_lens_taken_ct = 0
        self.seq_lens_sched_work_ct = 0

    def note_extend(
        self,
        batch,
        next_draft: Optional[NextDraft] = None,
        new_seq_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """At a decode pass's end: what staging the next pass's draft inputs would rest on, and
        the lengths this pass published (its post-verify output)."""
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
        """A pass that is not a decode round (prefill, mixed): nothing is staged from it."""
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
        if self.needs_next_draft and noted[3] is None:
            return "not_staged_by_post_verify_graph"
        return "staged"

    def take_seq_lens(self, batch, *, sched_idle: bool) -> Optional[torch.Tensor]:
        """Before a decode pass's resolve, without consuming the note: the previous pass's
        published lengths when this pass is staged (count_draft will say so), else None."""
        noted = self.noted
        if noted is None or noted[4] is None or self._reason(noted, batch) != "staged":
            return None
        self.seq_lens_taken_ct += 1
        self.seq_lens_sched_work_ct += not sched_idle
        return noted[4]

    def count_draft(self, batch) -> Optional[NextDraft]:
        """At a decode pass's start: the inputs the previous pass staged for it when they hold
        (counted ``staged``), else None, counting the first reason they do not."""
        noted, self.noted = self.noted, None
        reason = self._reason(noted, batch)
        staged = noted[3] if reason == "staged" else None
        self.counts[reason] += 1
        self.decode_pass_ct += 1
        if self.decode_pass_ct % LOG_EVERY == 0:
            logger.info(
                "PERF-LL step2: %d decode passes: %s%s%s",
                self.decode_pass_ct,
                ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items())),
                "" if self.check is None else "; " + self.check.summary(),
                (
                    f"; step5a seq_lens taken {self.seq_lens_taken_ct}, forward waited for "
                    f"schedule-stream work in {self.seq_lens_sched_work_ct}"
                    if self.seq_lens_taken_ct
                    else ""
                ),
            )
        return staged
