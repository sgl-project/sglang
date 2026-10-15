# SPDX-License-Identifier: Apache-2.0
"""Host side of the guarded waits (RFC #43949 section 8).

Every cross-workgroup or cross-GPU wait has a deadline. A wait that passes it records the reason and the site in
this rank's stop block, flags every peer, and ends its launch with a store to address 0; a wait that sees a flag
ends its launch the same way without writing a reason, so the first reason stays.

Stop block, one per rank in peer memory: word STOP_REASON is this rank's deadline stop (reason | site << 8 |
rank << 16); word STOP_FLAG is (peer rank + 1) | site << 8 of the peer whose deadline passed.

Guard block, int32 words in device memory, found by the kernels through the epoch buffer (words EPOCH_GUARD and
EPOCH_GUARD + 1 hold its address): G_DEADLINE (i64 clock ticks), G_OWN (i64 address of this rank's stop block),
G_RANK, G_NPES, G_MIRROR (i64 address of words a test can read after the processes end, see MIRROR_*; 0 for none),
and from G_PEERS the i64 address of every rank's stop block."""

from __future__ import annotations

import torch

from sglang.experimental.monokernels.hip.common.peer_memory import PeerBuffer

TICKS_PER_S = 100_000_000  # s_memrealtime runs at 100 MHz
EPOCH_GUARD = 2
STOP_REASON, STOP_FLAG = 0, 1
STOP_BYTES = 64
G_DEADLINE, G_OWN, G_RANK, G_NPES, G_MIRROR, G_PEERS = 0, 2, 4, 5, 6, 8
MAX_PEERS = 8
GUARD_WORDS = G_PEERS + 2 * MAX_PEERS
REASON_DEADLINE, REASON_PEER = 1, 2
# mirror words: the deadline reason, the reason of a stop after a peer's flag, and (fault-injection builds
# only) the site of the wait the injected stall holds
MIRROR_DEADLINE, MIRROR_PEER, MIRROR_ARMED = 0, 1, 2

# wait-site names; a site's id is its index + 1. Kernel modules register their sites when imported, in a fixed
# order, so an id names the same site in every process.
_SITES: list = []


def register_sites(names) -> None:
    for name in names:
        if name not in _SITES:
            if len(_SITES) == 255:
                raise ValueError("more than 255 named wait sites")
            _SITES.append(name)


def site_id(name: str) -> int:
    if name not in _SITES:
        raise ValueError(f"wait site {name!r} is not registered")
    return _SITES.index(name) + 1


def site_name(sid: int) -> str:
    return _SITES[sid - 1] if 1 <= sid <= len(_SITES) else f"site {sid}"


def sites() -> tuple:
    """The registered wait sites."""
    return tuple(_SITES)


def decode_reason(word: int) -> dict:
    """A STOP_REASON word as its reason, site and rank (None for 0)."""
    if word == 0:
        return {}
    reason = {REASON_DEADLINE: "deadline", REASON_PEER: "peer stopped"}.get(
        word & 0xFF, f"reason {word & 0xFF}"
    )
    return {"reason": reason, "site": site_name((word >> 8) & 0xFF), "rank": word >> 16}


def decode_flag(word: int) -> dict:
    """A STOP_FLAG word as the flagging peer and its site (None for 0)."""
    if word == 0:
        return {}
    return {"peer": (word & 0xFF) - 1, "site": site_name((word >> 8) & 0xFF)}


class Guard:
    """One TP rank's guard: its stop block in peer memory and the guard block. The constructor is collective."""

    def __init__(
        self, tp: int, rank: int, group, device, deadline_s: float, mirror_ptr: int = 0
    ):
        if tp > MAX_PEERS:
            raise ValueError(f"guarded waits support up to {MAX_PEERS} ranks, got {tp}")
        self.deadline_s = deadline_s
        self.stop = PeerBuffer(STOP_BYTES, group, rank, tp, device)
        self.stop.bytes.zero_()
        words = torch.zeros(GUARD_WORDS, dtype=torch.int32)
        w64 = words.view(torch.int64)
        w64[G_DEADLINE // 2] = int(deadline_s * TICKS_PER_S)
        w64[G_OWN // 2] = self.stop.local
        words[G_RANK], words[G_NPES] = rank, tp
        w64[G_MIRROR // 2] = mirror_ptr
        for p, addr in enumerate(self.stop.addresses.tolist()):
            w64[G_PEERS // 2 + p] = addr
        self.block = words.to(device)

    def bind(self, epoch: torch.Tensor) -> None:
        """Point an epoch buffer at this guard block (before any launch, outside graph capture)."""
        addr = torch.tensor([self.block.data_ptr()], dtype=torch.int64).view(
            torch.int32
        )
        epoch[EPOCH_GUARD : EPOCH_GUARD + 2].copy_(addr.to(epoch.device))

    def reason(self) -> dict:
        """This rank's stop block as the host reads it."""
        words = self.stop.bytes[:8].view(torch.int32).tolist()
        return {
            "own": decode_reason(words[STOP_REASON]),
            "flag": decode_flag(words[STOP_FLAG]),
        }

    def close(self) -> None:
        self.stop.close()
