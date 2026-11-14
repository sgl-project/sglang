# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Two-shot Lamport all-reduce of BF16 rows [B, width] over TP ranks.

Rank ``o`` owns columns [o W, (o + 1) W), W = width / owners. Every buffer is
Lamport-armed and only its local reader re-arms it:

- ``push``: a rank stores its 8-column chunk of row ``token`` into the owner's
  receive slot [rank] (a peer store; the owner's ``recv`` [TP, cap, W]; callers with a
  runtime parity pass the whole [parities x TP, cap, W] and slot parity x TP + rank).
- ``reduce``: the owner polls the TP slots of an owned chunk in its own memory, sums
  them in FP32 in rank order, rounds once and multicasts the BF16 sum into every
  rank's gather row [cap, width]; then re-arms the slots.
- every rank then polls its gather row (one slot) and re-arms it.

Each GPU moves ~2 x B x width x 2 bytes instead of TP x B x width x 2 (one-shot), and
every rank gets the same bits. ``owners`` = 1 is a one-GPU stand-in: rank 0 owns every
column and its TP local slots play the ranks. The o-projection reduces with ``reduce``
and re-arms at once (two parities); the MoE tail reduces in WaveMoeTail and re-arms the
previous MoE layer's rows (three parities).
"""

import torch
import torch.distributed._symmetric_memory as symm_mem

from sglang.kernels.ops.communication.kimi_k3_specialized.twoshot import SENTINEL
from sglang.srt.kernel_plugins.nvidia.kimi_k3.cutedsl.common.host import (
    validate_init_barrier_group,
)


class TwoShot:
    """Symmetric buffers per layer parity: ``recv`` [TP, cap, W] (this rank's owned
    columns, one slot per source rank), ``gather`` [1, cap, width] (multicast) and the
    ranks' ``recv`` base addresses [parity, TP]."""

    def __init__(
        self,
        group,
        device,
        tp,
        rank,
        width=7168,
        cap=64,
        parities=2,
        *,
        init_barrier_group=None,
    ):
        init_group = validate_init_barrier_group(group, init_barrier_group)
        self.tp, self.rank, self.width = tp, rank, width
        shape = (parities, tp, cap, width // tp)
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.recv = symm_mem.empty(shape, **bf16)
        self.gather = symm_mem.empty((parities, 1, cap, width), **bf16)
        handles = [symm_mem.rendezvous(t, group) for t in (self.recv, self.gather)]
        size = self.recv[0].numel() * 2
        ptrs = handles[0].buffer_ptrs
        bases = [[int(b) + p * size for b in ptrs] for p in range(parities)]
        self.peers = torch.tensor(bases, dtype=torch.int64, device=device)
        self.gather_mc = int(handles[1].multicast_ptr or 0)
        if not self.gather_mc:  # views() offsets this address
            raise RuntimeError("Kimi K3 two-shot all-reduce needs multicast memory")
        for t in (self.recv, self.gather):
            t.view(torch.int32).fill_(SENTINEL)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=init_group)

    def views(self, parity):
        """(recv, peers, gather, gather multicast address) of layer parity."""
        gather_mc = self.gather_mc + parity * self.gather[0].numel() * 2
        return self.recv[parity], self.peers[parity], self.gather[parity], gather_mc
