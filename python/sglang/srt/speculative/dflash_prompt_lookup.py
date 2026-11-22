"""Prompt lookup for DFLASH: a request's draft block taken from its own tokens.

Agentic and coding requests often copy long spans of their own context (file
contents, tool output, code written earlier). A looked-up step finds the longest
suffix of each request's tokens, up to ``max_match`` of them, that also ends at
an earlier position of the same request. When the match is long enough, the
tokens that followed it replace the drafter's tokens in that request's verify
block. Only the request's own prompt and output tokens are used.

Verification is the stock one, so every committed token is still the target's:
greedy verification commits a drafted token only where it equals the target's
argmax, and the speculative sampling accept takes a looked-up token, a point
mass (q = 1), with its target probability and on rejection samples the target
distribution without it. A selector draft's sparse q rows are turned into those
point masses (``_point_mass_proposals``).

The token history lives on the device, one table row per request of the batch,
and each decode step keeps it current with device ops. Whether a lookup replaced
any draft reaches the host two decode steps later through pinned memory, when
the device has long finished that step; after ``quiet`` lookups in a row that
replace nothing, only every ``probe``-th step looks up (``LookupSchedule``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import msgspec
import numpy as np
import torch

from sglang.srt.speculative.dflash_utils import verify_logits_unadjusted
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask
from sglang.srt.utils.device import async_h2d, is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

logger = logging.getLogger(__name__)


class PromptLookupConfig(msgspec.Struct, frozen=True):
    # Measured on one agentic coding workload with greedy decoding: the longest
    # suffix compared, and the shortest match that replaces a greedy draft.
    max_match: int = 16
    min_match: int = 8
    # Not measured. A sampled request takes a looked-up token only with its
    # target probability, so it waits for a longer match.
    min_match_sampled: int = 16
    # Arbitrary: larger decode batches keep the drafter's blocks, and the token
    # table holds this many rows.
    max_bs: int = 32


def _host_tokens(seqs: Sequence[Sequence[int]]) -> np.ndarray:
    """The concatenated tokens as int32; ids outside int32 (multimodal pad values)
    become -1, which never matches."""
    parts = [np.asarray(s, dtype=np.int64).reshape(-1) for s in seqs]
    tokens = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.int64)
    bad = (tokens < 0) | (tokens > np.iinfo(np.int32).max)
    return np.where(bad, -1, tokens).astype(np.int32)


class LookupSchedule:
    """Which decode steps look up: every step while one of the last ``quiet``
    lookups replaced a draft (a hit), else every ``probe``-th step. Fed only with
    results every rank reads alike, so every rank decides alike."""

    # Picked by hand, not tuned.
    def __init__(self, *, quiet: int = 8, probe: int = 8) -> None:
        self._quiet = quiet
        self._probe = probe
        self._misses = 0  # lookups in a row without a hit
        self._skipped = 0  # steps since the last lookup

    def next_step(self, hit: Optional[bool]) -> bool:
        """Whether this step looks up, given whether the lookup two steps back hit
        (None when that step did not look up)."""
        if hit is not None:
            self._misses = 0 if hit else self._misses + 1
        due = self._misses < self._quiet or self._skipped + 1 >= self._probe
        self._skipped = 0 if due else self._skipped + 1
        return due


class HitRing:
    """Whether each lookup hit, read on the host ``lag`` decode steps later from
    pinned memory. By then the scheduler has processed that step's results, which
    the device produced after the copy, so the event wait returns at once and
    every rank reads the same value."""

    def __init__(self, *, device: torch.device, lag: int = 2) -> None:
        num_slots = lag + 1
        self._lag = lag
        self._host = torch.zeros(
            (num_slots,), dtype=torch.bool, pin_memory=is_pin_memory_available(device)
        )
        self._events = [
            torch.cuda.Event() if device.type == "cuda" else None
            for _ in range(num_slots)
        ]
        self._pushed = [False] * num_slots
        self._step = 0  # takes so far: one per decode step, before its push

    def push(self, hit: torch.Tensor) -> None:
        slot = self._step % len(self._pushed)
        self._host[slot : slot + 1].copy_(hit.view(1), non_blocking=True)
        if self._events[slot] is not None:
            self._events[slot].record()
        self._pushed[slot] = True

    def take(self) -> Optional[bool]:
        """Called once per decode step, before its push: whether the lookup ``lag``
        steps back hit, or None when that step did not look up."""
        self._step += 1
        slot = (self._step - self._lag) % len(self._pushed)
        if not self._pushed[slot]:
            return None
        self._pushed[slot] = False
        if self._events[slot] is not None:
            self._events[slot].synchronize()
        return bool(self._host[slot])


class PromptLookupTable:
    """The token history of up to ``rows`` requests, and the lookup over it.

    Column ``pad + p`` of a row holds the token at position p of the request
    that owns the row. The ``pad`` leading columns hold -1, so a suffix compare
    never reads before position 0, and the last column absorbs writes past the
    end of a row. Row ``spare_row`` takes the writes of requests without a row
    of their own and is never looked up. The lookup's scratch is allocated here,
    once, so a step allocates nothing large.
    """

    def __init__(
        self, *, rows: int, positions: int, max_match: int, device: torch.device
    ) -> None:
        self.rows = rows
        self.spare_row = rows
        self.positions = positions
        self.device = device
        self._pad = max_match
        self._width = self._pad + positions + 1
        self.tokens = torch.full(
            (rows + 1, self._width), -1, dtype=torch.int32, device=device
        )
        cells = rows * self._width
        self._gathered = torch.empty((cells,), dtype=torch.int32, device=device)
        self._match = torch.empty((cells,), dtype=torch.int32, device=device)
        self._alive = torch.empty((cells,), dtype=torch.bool, device=device)
        self._equal = torch.empty((cells,), dtype=torch.bool, device=device)
        self._arange = torch.arange(self._width, dtype=torch.int32, device=device)
        self._back = torch.arange(max_match, dtype=torch.int64, device=device)

    @property
    def nbytes(self) -> int:
        tensors = (self.tokens, self._gathered, self._match, self._alive, self._equal)
        return sum(t.numel() * t.element_size() for t in tensors)

    def fill(self, *, row: int, tokens: np.ndarray) -> None:
        """Gives the row to a new owner: its host-known tokens, then -1 to the end
        of the row, so no earlier owner's token stays behind."""
        n = min(int(tokens.shape[0]), self.positions)
        self.tokens[row, self._pad + n :].fill_(-1)
        if n == 0:
            return
        src = torch.from_numpy(np.ascontiguousarray(tokens[:n]))
        if is_pin_memory_available(self.device):
            src = src.pin_memory()
        self.tokens[row, self._pad : self._pad + n].copy_(src, non_blocking=True)

    def write_last(
        self, *, rows: torch.Tensor, last: torch.Tensor, tokens: torch.Tensor
    ) -> None:
        """tokens[rows, position last] = tokens, the step's bonus tokens."""
        cols = (last.to(torch.int64) + self._pad).clamp_(max=self._width - 1)
        self.tokens.index_put_((rows, cols), tokens.to(torch.int32))

    def write_after(
        self, *, rows: torch.Tensor, last: torch.Tensor, tokens: torch.Tensor
    ) -> None:
        """tokens[rows, last + 1 + j] = tokens[:, j], a step's committed tokens; the
        columns past the commit are rewritten before they are ever read."""
        steps = torch.arange(tokens.shape[1], dtype=torch.int64, device=tokens.device)
        cols = last.to(torch.int64)[:, None] + (self._pad + 1) + steps[None, :]
        cols = cols.clamp_(max=self._width - 1)
        self.tokens.index_put_(
            (rows[:, None].expand_as(cols), cols), tokens.to(torch.int32)
        )

    def lookup(
        self, *, rows: torch.Tensor, last: torch.Tensor, span: int, out_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(match, cont) of request i, at table row rows[i] and position last[i].

        match[i] is the longest match, at most max_match, of the tokens ending at
        last[i] against the tokens ending at an earlier position e < last[i];
        cont[i] are the out_len tokens after the latest such e, repeating tokens
        e+1..last[i] where the copy runs into last[i]. ``span`` is a host bound on
        last + 1 over the batch. Device ops only, all in bounds.
        """
        n = int(rows.shape[0])
        if n > self.rows:
            raise ValueError(f"prompt lookup scratch holds {self.rows} rows, got {n}")
        pad = self._pad
        span = max(1, min(int(span), self.positions))
        last = last.to(torch.int64)

        cols = pad + span
        hist = self._gathered[: n * cols].view(n, cols)
        torch.index_select(self.tokens[:, :cols], 0, rows, out=hist)

        # The suffix, newest first; a position before 0 or an unknown token (-1)
        # becomes -2, which matches nothing.
        idx = ((last + pad)[:, None] - self._back[None, :]).clamp_(0, cols - 1)
        suffix = torch.gather(hist, 1, idx)
        suffix = torch.where(
            (idx >= pad) & (suffix >= 0), suffix, torch.full_like(suffix, -2)
        )

        match = self._match[: n * span].view(n, span)
        alive = self._alive[: n * span].view(n, span)
        equal = self._equal[: n * span].view(n, span)
        torch.eq(hist[:, pad : pad + span], suffix[:, 0:1], out=alive)
        match.copy_(alive)
        for k in range(1, self._back.shape[0]):
            torch.eq(hist[:, pad - k : pad - k + span], suffix[:, k : k + 1], out=equal)
            alive.logical_and_(equal)
            match.add_(alive)

        positions = self._arange[:span][None, :]
        # Only earlier end positions count; the others read -1.
        torch.ge(positions, last[:, None], out=alive)
        match.masked_fill_(alive, -1)
        longest = match.amax(dim=1)
        torch.eq(match, longest[:, None], out=equal)
        torch.where(equal, positions, torch.full_like(positions, -1), out=match)
        end = match.amax(dim=1).to(torch.int64)

        # Keeps the gather in bounds for rows without a match too.
        end = torch.minimum(end, last - 1).clamp_(min=0)
        period = (last - end).clamp_(min=1)
        steps = torch.arange(out_len, dtype=torch.int64, device=rows.device)
        cont_idx = (end + 1 + pad)[:, None] + steps[None, :] % period[:, None]
        cont = torch.gather(hist, 1, cont_idx.clamp_(0, cols - 1))
        return longest.clamp(min=0), cont


def _point_mass_proposals(
    *,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    use_lookup: torch.Tensor,
    tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A selector draft's sparse proposal rows, with the requests ``use_lookup``
    marks turned into point masses on their looked-up ``tokens``.

    candidate_ids, q_rows: [bs, gamma, top_k]; tokens: [bs, gamma]. Every top-k
    slot of a looked-up position carries its token with q = 1, so scattering them
    into a zeroed dense q writes exactly one 1.0 per position, and the cleanup
    scatter over the same ids zeroes it again.
    """
    mask = use_lookup.view(-1, 1, 1)
    ids = torch.where(
        mask,
        tokens.to(candidate_ids.dtype).unsqueeze(-1).expand_as(candidate_ids),
        candidate_ids,
    )
    q = torch.where(mask, torch.ones_like(q_rows), q_rows)
    return ids, q


def _lookup_block(
    *,
    table: PromptLookupTable,
    rows: torch.Tensor,
    last: torch.Tensor,
    span: int,
    drafts: torch.Tensor,
    greedy: torch.Tensor,
    config: PromptLookupConfig,
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(block, use_lookup): ``drafts`` [bs, gamma] with the rows of the requests
    whose match reaches their minimum replaced by the looked-up continuation."""
    match, cont = table.lookup(
        rows=rows, last=last, span=span, out_len=int(drafts.shape[1])
    )
    need = torch.where(
        greedy,
        torch.full_like(match, config.min_match),
        torch.full_like(match, config.min_match_sampled),
    )
    use = (match >= need) & ((cont >= 0) & (cont < vocab_size)).all(dim=1)
    block = torch.where(use[:, None], cont.to(drafts.dtype), drafts)
    return block, use


def _batch_allows_lookup(batch: ScheduleBatch) -> bool:
    """Whether looked-up blocks leave this batch's outputs those of stock DFLASH:
    verify logit adjustments such as penalties see a request's history as of its
    block's start, which the lookup moves; grammar and logprob batches stay stock."""
    if batch.has_grammar or batch.return_logprob:
        return False
    info = batch.sampling_info
    return info is None or verify_logits_unadjusted(info)


def _span_bound(*, batch: ScheduleBatch, draft_input: DFlashDraftInputV2) -> int:
    """A host bound on prefix_lens + 1 over the batch, without a device sync."""
    bounds = []
    if batch.seq_lens_cpu is not None:
        bounds.append(int(batch.seq_lens_cpu.max()) + 1)
    if draft_input.nxt_kv_lens_cpu is not None:
        # The committed length plus the reservation of the steps in flight.
        bounds.append(int(draft_input.nxt_kv_lens_cpu.max()))
    if not bounds:
        bounds.append(max(req.kv.kv_allocated_len for req in batch.reqs))
    return max(bounds)


class DFlashPromptLookup:
    """Prompt lookup for one DFLASH worker: the token table, which request holds
    which row of it, and which steps look up.

    A row belongs to one allocation of a request slot, keyed by (req_pool_idx,
    req_generation). The pool bumps a slot's generation each time it hands the
    slot out, so a new request, a retracted request prefilled again and a reused
    slot fill a row anew, while a running request keeps its row through mixed
    and chunked prefill steps. Every decision is made from batch state, the
    replicated slot generations, or device results every TP rank computes alike,
    so all ranks propose the same blocks.
    """

    def __init__(
        self,
        *,
        config: PromptLookupConfig,
        vocab_size: int,
        max_rows: int,
        positions: int,
        device: torch.device,
        log_rank: bool,
    ) -> None:
        self._config = config
        self._vocab_size = vocab_size
        rows = min(config.max_bs, max_rows)
        self._table = PromptLookupTable(
            rows=rows, positions=positions, max_match=config.max_match, device=device
        )
        self._ring = HitRing(device=device)
        self._schedule = LookupSchedule()
        self._row_of: Dict[Tuple[int, int], int] = {}  # (slot, generation) -> row
        self._owners: List[Optional[Tuple[int, int]]] = [None] * rows
        # This step's table rows, from propose() until record().
        self._step_rows: Optional[torch.Tensor] = None
        if log_rank:
            logger.info(
                "DFLASH prompt lookup: on, with a token table of %d rows x %d "
                "positions (%.1f MiB).",
                rows,
                positions,
                self._table.nbytes / (1 << 20),
            )

    def propose(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        draft_tokens: torch.Tensor,
        selector_sample: Optional[Tuple[torch.Tensor, torch.Tensor]],
        req_generation: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Brings the batch's rows up to this step's bonus tokens, then replaces
        ``draft_tokens[:, 1:]`` of the requests whose last tokens repeat earlier
        ones with the tokens that followed them. Returns the selector sample the
        accept must use: looked-up rows become point masses."""
        table = self._table
        rows = self._assign_rows(reqs=batch.reqs, req_generation=req_generation)
        rows_device = async_h2d(rows, dtype=torch.int64, device=table.device)
        table.write_last(
            rows=rows_device, last=batch.seq_lens, tokens=draft_input.bonus_tokens
        )
        self._step_rows = rows_device
        due = self._schedule.next_step(self._ring.take())
        if not due or table.spare_row in rows or not _batch_allows_lookup(batch):
            return selector_sample
        block, use = _lookup_block(
            table=table,
            rows=rows_device,
            last=batch.seq_lens,
            span=_span_bound(batch=batch, draft_input=draft_input),
            drafts=draft_tokens[:, 1:],
            greedy=resolve_greedy_mask(
                bs=len(rows), sampling_info=batch.sampling_info, device=table.device
            ),
            config=self._config,
            vocab_size=self._vocab_size,
        )
        self._ring.push(use.any())
        draft_tokens[:, 1:].copy_(block)
        if selector_sample is None:
            return None
        candidate_ids, q_rows = selector_sample
        return _point_mass_proposals(
            candidate_ids=candidate_ids, q_rows=q_rows, use_lookup=use, tokens=block
        )

    def record(self, *, prefix_lens: torch.Tensor, out_tokens: torch.Tensor) -> None:
        """Writes the step's committed tokens after position ``prefix_lens``."""
        rows, self._step_rows = self._step_rows, None
        self._table.write_after(rows=rows, last=prefix_lens, tokens=out_tokens)

    def _assign_rows(
        self, *, reqs: Sequence[Req], req_generation: torch.Tensor
    ) -> List[int]:
        """Each request's table row: its own; else a row no request of the batch
        holds, filled from the request's host tokens; else the spare row."""
        table = self._table
        slots = [req.kv.req_pool_idx for req in reqs]
        keys = list(zip(slots, req_generation[slots].tolist()))
        rows = [self._row_of.get(key, table.spare_row) for key in keys]
        held = set(rows)
        free = (row for row in range(table.rows) if row not in held)
        for i, req in enumerate(reqs):
            if rows[i] != table.spare_row:
                continue
            row = next(free, None)
            if row is None:
                break
            previous = self._owners[row]
            if previous is not None:
                del self._row_of[previous]
            self._owners[row] = keys[i]
            self._row_of[keys[i]] = row
            # Positions follow the KV: the (padded) prompt, then the output.
            table.fill(
                row=row, tokens=_host_tokens((req.origin_input_ids, req.output_ids))
            )
            rows[i] = row
        return rows
