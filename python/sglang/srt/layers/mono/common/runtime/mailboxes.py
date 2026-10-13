# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/runtime/mailboxes.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A runner's mailboxes and where a step begins with them.

A mailbox tag is a layer's, so a pair left from one step would pass for the
next step's: every buffer holding pairs is cleared before a step's first
kernel, then the peer fence (``PeerBuffer.fence``) holds every rank until all
of them cleared, so no peer's first push of the step lands on an uncleared
buffer or is wiped by a late clear. At the step's start, not its end: a step
that stopped half-way leaves nothing for the next one to trip on.
"""

import torch

from sglang.srt.layers.mono.common.runtime.peer_memory import PeerBuffer


class StepMailboxes:
    """``peers``' data region and this rank's ``scratch`` buffers, cleared
    together by ``begin_step``."""

    def __init__(self, peers: PeerBuffer, *scratch: torch.Tensor) -> None:
        self.peers = peers
        self._buffers = [*scratch, peers.bytes]

    def begin_step(self) -> None:
        """Before the step's first kernel, on its stream (a graph records it)."""
        torch._foreach_zero_(self._buffers)
        self.peers.fence()
