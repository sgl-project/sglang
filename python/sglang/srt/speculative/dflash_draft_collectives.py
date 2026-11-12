# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""One-shot collectives for the DFLASH draft over epoch-tagged mailboxes.

Each draft layer all-reduces a [block_size, hidden] BF16 partial twice, and the
selector gathers every rank's top-k (value, token id). At 16 ranks the shipped
paths are a Lamport 1shot_push with re-arm and two NCCL ring all-gathers; these
push 64-bit records that carry their own epoch tag (kernel:
``jit/csrc/distributed/draft_collectives.cuh``).

Both results are bitwise equal to the shipped ones: the all-reduce sums in rank
order in FP32 and rounds once, as 1shot_push does, and the gather moves bits.
Only the one-request draft shape takes the new path; every
other shape keeps the group's own collective, so a mailbox region is rewritten
whole on every call and a stale record is always two calls old. DraftTopkGathers
takes the gather to more batch sizes with one mailbox per row count, which keeps
that rule.

Every mailbox poll is bounded: a rank still missing a peer's record after the
deadline traps (the step fails) instead of publishing a partial result. An
instance serves one stream at a time (its calls share one epoch counter).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.draft_collectives import (
    MAX_GATHER_TOKEN_ID,
    all_reduce_num_ctas,
    all_reduce_slot_bytes,
    draft_all_reduce_pk3,
    draft_topk_gather,
    load_draft_collectives,
    topk_gather_num_ctas,
    topk_gather_slot_bytes,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    _allocate_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.runtime_context import get_device

# The kernels address at most 16 processes (draft_collectives.cuh kMaxGpus).
_MAX_SLOTS = 16


def serving_deadline_ns() -> int:
    """The megakernel's cross-GPU wait deadline (half the watchdog, at most 60 s)."""
    # Imported here: the plugin binding module pulls in the runner stack.
    from sglang.srt.kernel_plugins.nvidia.binding import MIN_DEADLINE_S, deadline_s

    seconds = deadline_s(get_device().watchdog_timeout)
    if seconds < MIN_DEADLINE_S:
        raise ValueError(
            f"DFLASH draft collectives need --watchdog-timeout >= {2 * MIN_DEADLINE_S:g} s "
            f"(their wait deadline is half of it, at least {MIN_DEADLINE_S:g} s)."
        )
    return int(seconds * 1_000_000_000)


class _Mailboxes:
    """One symmetric allocation of [2 parities][num_slots][slot_bytes] per process,
    plus this process's per-CTA epoch counters. Collective over the CPU group (as
    CustomAllReduceV2 rendezvous): every rank builds it in the same call, outside
    CUDA-graph capture."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        device: torch.device,
        num_slots: int,
        slot_bytes: int,
        num_ctas: int,
    ) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DFLASH draft collectives are built by their first eager call; "
                "the first call happened inside CUDA-graph capture."
            )
        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        nbytes = 2 * num_slots * slot_bytes
        # (owner, per-rank uint8 views of nbytes, multicast VA); the owner keeps the allocation alive.
        self._symm_owner, self._buffers, _ = _allocate_symmetric_memory(
            nbytes, device=device, group=group
        )
        # Tag 0 is never current: the first two calls use tags 1 and 2.
        self._buffers[rank].zero_()
        self.counters = torch.zeros((num_ctas,), dtype=torch.int32, device=device)
        # Before the barrier: a rank still compiling the module must not hold up peers that already poll its records.
        load_draft_collectives(
            num_slots=num_slots, ranks_per_gpu=num_slots // world_size
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        self.peers: List[int] = [int(b.data_ptr()) for b in self._buffers]
        self.gpu = rank
        self.num_slots = num_slots
        self.slot_bytes = slot_bytes


def _check_slots(group: GroupCoordinator, ranks_per_gpu: int) -> int:
    num_slots = group.world_size * ranks_per_gpu
    if num_slots > _MAX_SLOTS:
        raise ValueError(
            f"DFLASH draft collectives support at most {_MAX_SLOTS} ranks, got {num_slots}."
        )
    return num_slots


def _check_deadline(deadline_ns: int) -> int:
    if deadline_ns <= 0:
        raise ValueError(
            f"DFLASH draft collectives need a positive deadline, got {deadline_ns} ns."
        )
    return int(deadline_ns)


class DraftAllReduce:
    """Rank-order one-shot all-reduce of the draft's [rows, hidden] BF16 partials
    (SGLANG_OPT_DFLASH_DRAFT_PK3_ALL_REDUCE)."""

    def __init__(
        self,
        *,
        group: GroupCoordinator,
        rows: int,
        hidden_size: int,
        deadline_ns: int,
        ranks_per_gpu: int = 1,
    ) -> None:
        self._group = group
        self._hidden_size = int(hidden_size)
        self._ranks_per_gpu = int(ranks_per_gpu)
        self._num_slots = _check_slots(group, self._ranks_per_gpu)
        self._deadline_ns = _check_deadline(deadline_ns)
        self._mailboxes: Optional[_Mailboxes] = None
        self.set_rows(rows)

    def set_rows(self, rows: int) -> None:
        if self._mailboxes is not None:
            raise RuntimeError("DraftAllReduce rows are fixed once it has run.")
        if rows * self._hidden_size % 8:
            raise ValueError("DraftAllReduce needs rows * hidden to be a multiple of 8.")
        self.numel = int(rows) * self._hidden_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() != self.numel or x.dtype != torch.bfloat16:
            return self._group.all_reduce(x)
        return self.reduce_partials(x.contiguous().view(1, *x.shape))

    def reduce_partials(self, partials: torch.Tensor) -> torch.Tensor:
        """Sum of the [ranks_per_gpu, *shape] partials this process hosts and those of
        every other process; [*shape]."""
        mailboxes = self._mailboxes
        if mailboxes is None:
            mailboxes = self._mailboxes = _Mailboxes(
                group=self._group.cpu_group,
                device=partials.device,
                num_slots=self._num_slots,
                slot_bytes=all_reduce_slot_bytes(self.numel),
                num_ctas=all_reduce_num_ctas(self.numel),
            )
        out = torch.empty(
            partials.shape[1:], dtype=partials.dtype, device=partials.device
        )
        draft_all_reduce_pk3(
            inputs=partials,
            output=out,
            counters=mailboxes.counters,
            peers=mailboxes.peers,
            gpu=mailboxes.gpu,
            slot_bytes=mailboxes.slot_bytes,
            num_slots=self._num_slots,
            deadline_ns=self._deadline_ns,
        )
        return out


class DraftTopkGather:
    """All-gather along the last dim of every rank's top-k fp32 values and int64
    token ids in one launch (SGLANG_OPT_DFLASH_PACKED_TOPK_GATHER)."""

    def __init__(
        self,
        *,
        rows: int,
        k: int,
        vocab_size: int,
        deadline_ns: int,
        ranks_per_gpu: int = 1,
    ) -> None:
        if vocab_size - 1 > MAX_GATHER_TOKEN_ID:
            raise ValueError(
                f"DFLASH packed top-k gather holds token ids below "
                f"{MAX_GATHER_TOKEN_ID + 1}; vocab_size is {vocab_size}."
            )
        self._k = int(k)
        self._ranks_per_gpu = int(ranks_per_gpu)
        self._deadline_ns = _check_deadline(deadline_ns)
        self._group: Optional[GroupCoordinator] = None
        self._num_slots = 0
        self._mailboxes: Optional[_Mailboxes] = None
        self.set_rows(rows)

    def set_rows(self, rows: int) -> None:
        if self._mailboxes is not None:
            raise RuntimeError("DraftTopkGather rows are fixed once it has run.")
        self.rows = int(rows)

    def __call__(
        self, group: GroupCoordinator, vals: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(group.all_gather(vals, dim=-1), group.all_gather(ids, dim=-1)) for fp32
        vals and int64 ids of shape [rows, k]."""
        if (
            vals.shape != (self.rows, self._k)
            or vals.dtype != torch.float32
            or ids.dtype != torch.int64
        ):
            return group.all_gather(vals, dim=-1), group.all_gather(ids, dim=-1)
        return self.gather_partials(
            group, vals.reshape(1, *vals.shape), ids.reshape(1, *ids.shape)
        )

    def build(self, group: GroupCoordinator, device: torch.device) -> _Mailboxes:
        """This gather's mailboxes, built on the first call (collective over the
        group's CPU group, eager only)."""
        if self._mailboxes is None:
            self._group = group
            self._num_slots = _check_slots(group, self._ranks_per_gpu)
            self._mailboxes = _Mailboxes(
                group=group.cpu_group,
                device=device,
                num_slots=self._num_slots,
                slot_bytes=topk_gather_slot_bytes(self.rows, self._k),
                num_ctas=topk_gather_num_ctas(self._num_slots, self.rows, self._k),
            )
        elif group is not self._group:
            raise RuntimeError("DraftTopkGather serves one group.")
        return self._mailboxes

    def gather_partials(
        self, group: GroupCoordinator, vals: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather of the [ranks_per_gpu, rows, k] inputs this process hosts and those
        of every other process; ([rows, num_slots * k] fp32, same int64)."""
        mailboxes = self.build(group, vals.device)
        width = self._num_slots * self._k
        out_vals = torch.empty((self.rows, width), dtype=torch.float32, device=vals.device)
        out_ids = torch.empty((self.rows, width), dtype=torch.int64, device=vals.device)
        draft_topk_gather(
            vals=vals.contiguous(),
            ids=ids.contiguous(),
            out_vals=out_vals,
            out_ids=out_ids,
            counters=mailboxes.counters,
            peers=mailboxes.peers,
            gpu=mailboxes.gpu,
            slot_bytes=mailboxes.slot_bytes,
            num_slots=self._num_slots,
            deadline_ns=self._deadline_ns,
        )
        return out_vals, out_ids


class DraftTopkGathers:
    """The packed top-k gather at every draft batch size up to ``max_requests``
    (SGLANG_OPT_DFLASH_PACKED_TOPK_GATHER_MAX_BS): one DraftTopkGather, so one
    mailbox, per row count, because every call has to rewrite its mailbox whole.
    The first call builds them all (eager, the same order on every rank); a row
    count outside the set keeps the group's all-gather."""

    def __init__(
        self,
        *,
        rows: int,
        max_requests: int,
        k: int,
        vocab_size: int,
        deadline_ns: int,
        ranks_per_gpu: int = 1,
    ) -> None:
        if max_requests < 1:
            raise ValueError(f"DraftTopkGathers needs max_requests >= 1, got {max_requests}.")
        self._max_requests = int(max_requests)
        self._kwargs = dict(
            k=k, vocab_size=vocab_size, deadline_ns=deadline_ns, ranks_per_gpu=ranks_per_gpu
        )
        self._built = False
        self.set_rows(rows)

    def set_rows(self, rows: int) -> None:
        """``rows``: one request's draft rows (block_size - 1)."""
        if self._built:
            raise RuntimeError("DraftTopkGathers rows are fixed once it has run.")
        self.rows = int(rows)
        self.by_rows = {
            self.rows * n: DraftTopkGather(rows=self.rows * n, **self._kwargs)
            for n in range(1, self._max_requests + 1)
        }

    def _for_rows(
        self, group: GroupCoordinator, rows: int, device: torch.device
    ) -> Optional[DraftTopkGather]:
        if not self._built:
            for gather in self.by_rows.values():
                gather.build(group, device)
            self._built = True
        return self.by_rows.get(rows)

    def __call__(
        self, group: GroupCoordinator, vals: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DraftTopkGather.__call__ for [n * rows, k] inputs, n <= max_requests."""
        gather = self._for_rows(group, vals.shape[0], vals.device) if vals.dim() == 2 else None
        if gather is None:
            return group.all_gather(vals, dim=-1), group.all_gather(ids, dim=-1)
        return gather(group, vals, ids)

    def gather_partials(
        self, group: GroupCoordinator, vals: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DraftTopkGather.gather_partials for [ranks_per_gpu, n * rows, k] inputs."""
        gather = self._for_rows(group, vals.shape[1], vals.device)
        if gather is None:
            raise ValueError(f"DraftTopkGathers has no gather for {vals.shape[1]} rows.")
        return gather.gather_partials(group, vals, ids)
