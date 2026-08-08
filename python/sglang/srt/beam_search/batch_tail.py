"""Beam member rows riding a decode batch: layout, append/strip, retract.

Everything here mutates or reads ScheduleBatch, but lives outside it so the
batch class carries only the beam_tail field and one-line hook calls.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, List, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


@dataclasses.dataclass
class BeamTail:
    """Layout of the beam member rows appended after the reqs-aligned rows.

    Member rows are physical req_to_token rows without a Req; they exist in
    the batch row tensors only between prepare_for_decode (append) and the
    next filter/merge (strip). entries hold, per group in batch order:
    (group, leader index in reqs, tail-relative start, tail-relative end).
    """

    num_base_rows: int
    num_tail_rows: int
    entries: List[Tuple[Any, int, int, int]]


def append_beam_tail(batch: ScheduleBatch) -> None:
    """Append every live beam group's member rows after the reqs-aligned
    rows, so the decode forward (allocation, relay resolve, attention)
    covers them. Member rows decode in lockstep with their leader: their
    seq lens are the leader's. Reqs-sized host metadata (sampling_info,
    top_logprobs_nums, rids, ...) is intentionally NOT extended -- the
    worker slices the member rows off the logits before sampling."""
    assert batch.beam_tail is None
    entries = []
    tails = []
    tails_cpu = []
    leader_idx = []
    t = 0
    for i, req in enumerate(batch.reqs):
        group = req.beam_group
        if group is None or group.member_rows is None or group.retired:
            continue
        m = group.num_member_rows
        entries.append((group, i, t, t + m))
        tails.append(group.member_rows)
        tails_cpu.append(group.member_rows_cpu)
        leader_idx.append(i)
        t += m
    if not entries:
        return

    leader_idx_cpu = torch.tensor(leader_idx, dtype=torch.int64)
    widths_cpu = torch.tensor([e[3] - e[2] for e in entries], dtype=torch.int64)
    leader_idx_dev = leader_idx_cpu.to(batch.device, non_blocking=True)
    widths_dev = widths_cpu.to(batch.device, non_blocking=True)

    batch.req_pool_indices = torch.cat([batch.req_pool_indices, *tails])
    batch.req_pool_indices_cpu = torch.cat([batch.req_pool_indices_cpu, *tails_cpu])
    batch.seq_lens = torch.cat(
        [
            batch.seq_lens,
            torch.repeat_interleave(batch.seq_lens[leader_idx_dev], widths_dev),
        ]
    )
    if batch.seq_lens_cpu is not None:
        batch.seq_lens_cpu = torch.cat(
            [
                batch.seq_lens_cpu,
                torch.repeat_interleave(batch.seq_lens_cpu[leader_idx_cpu], widths_cpu),
            ]
        )
    batch.orig_seq_lens = torch.cat(
        [
            batch.orig_seq_lens,
            torch.repeat_interleave(batch.orig_seq_lens[leader_idx_dev], widths_dev),
        ]
    )
    batch.seq_lens_sum = None
    batch.beam_tail = BeamTail(
        num_base_rows=len(batch.reqs), num_tail_rows=t, entries=entries
    )


def strip_beam_tail(batch: ScheduleBatch) -> None:
    """Restore the 1:1 reqs<->rows layout by slicing the member rows off.
    Lazily invoked at the entry of every batch mutation (filter / merge /
    prepare), so the tail only ever spans one forward."""
    tail = batch.beam_tail
    if tail is None:
        return
    n = tail.num_base_rows
    assert n == len(batch.reqs), "reqs changed while a beam tail was attached"
    batch.beam_tail = None
    batch.req_pool_indices = batch.req_pool_indices[:n]
    batch.req_pool_indices_cpu = batch.req_pool_indices_cpu[:n]
    batch.seq_lens = batch.seq_lens[:n]
    if batch.seq_lens_cpu is not None:
        batch.seq_lens_cpu = batch.seq_lens_cpu[:n]
    batch.orig_seq_lens = batch.orig_seq_lens[:n]
    if batch.input_ids is not None:
        batch.input_ids = batch.input_ids[:n]
    batch.out_cache_loc = None
    batch.seq_lens_sum = None


def num_beam_member_rows(reqs) -> int:
    """Extra decode-slot demand: member rows decode in lockstep with their
    leader, one slot per row per step (beam requires page_size == 1)."""
    return sum(r.beam_group.num_member_rows for r in reqs if r.beam_group is not None)


def beam_retraction_order(sorted_indices: List[int], reqs: List[Req]) -> List[int]:
    """Beam groups are not retractable yet: members alias the leader's prompt
    KV, so a partial retract corrupts the group. Prefer keeping them (retract
    normal reqs first); under sustained pressure the caller aborts the whole
    group atomically."""
    if not any(reqs[i].beam_group is not None for i in sorted_indices):
        return sorted_indices
    return [i for i in sorted_indices if reqs[i].beam_group is not None] + [
        i for i in sorted_indices if reqs[i].beam_group is None
    ]
