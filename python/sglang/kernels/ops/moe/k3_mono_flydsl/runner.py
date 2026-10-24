# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono decode's process-wide state: one scratch buffer every launch
shares (their mailbox tags differ), the device step epoch and the TP group's
peer memory.

Mailbox tags carry the epoch the model bumps once a forward (``step_begin``,
which a CUDA graph captures), so no buffer is zeroed between steps.
"""

from typing import TYPE_CHECKING

import torch

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.kernels.ops.moe.k3_mono_flydsl.common.peer_memory import PeerBuffer

MAX_TOKENS = 8  # the x rows fit one CTA's LDS


class _State:
    scratch: torch.Tensor | None = None
    zero_bytes = 0
    epoch: torch.Tensor | None = None
    queue: torch.Tensor | None = None
    peers: "PeerBuffer | None" = None


def alloc_scratch(device) -> None:
    """At model init, outside any graph capture: the largest build's scratch
    and the epoch."""
    if _State.scratch is not None:
        return
    from sglang.kernels.ops.moe.k3_mono_flydsl import layer
    from sglang.kernels.ops.moe.k3_mono_flydsl.attention import kda
    from sglang.kernels.ops.moe.k3_mono_flydsl.stages import moe

    n = max(
        kda.scratch_bytes(
            kda.KdaPreBuild(
                tokens=MAX_TOKENS,
                qlen=MAX_TOKENS,
                nblocks=8,
                delta=True,
                state_len=3 + MAX_TOKENS,
            )
        ),
        moe.scratch_bytes(moe.MoeBuild(tokens=MAX_TOKENS)),
        layer.scratch_bytes(layer.K2Build(tokens=MAX_TOKENS, nblocks=8)),
    )
    _State.zero_bytes = (n + 255) // 256 * 256
    _State.scratch = torch.zeros(_State.zero_bytes, dtype=torch.uint8, device=device)
    _State.epoch = torch.zeros(1, dtype=torch.int32, device=device)
    _State.queue = torch.zeros(moe.QUEUE_BYTES // 4, dtype=torch.int32, device=device)


def step_begin() -> None:
    """Once a forward, before its first mono launch (a graph captures it)."""
    if _State.epoch is not None:
        _State.epoch.add_(1)


def peers(device) -> "PeerBuffer":
    """The TP group's peer buffer. Collective: every rank builds the same layers
    in the same order, so every rank reaches it at the first MoE layer."""
    if _State.peers is None:
        from sglang.kernels.ops.moe.k3_mono_flydsl import layer
        from sglang.kernels.ops.moe.k3_mono_flydsl.common.peer_memory import PeerBuffer
        from sglang.kernels.ops.moe.k3_mono_flydsl.stages import moe

        tp = get_parallel().tp_group
        nbytes = max(moe.peer_bytes(MAX_TOKENS), layer.peer_bytes(MAX_TOKENS))
        buf = PeerBuffer(nbytes, tp.cpu_group, tp.rank_in_group, tp.world_size, device)
        buf.bytes.zero_()
        _State.peers = buf
    return _State.peers


def launch_args() -> dict:
    """The state every launch takes."""
    assert _State.peers is not None and _State.epoch is not None
    return {
        "scratch": _State.scratch,
        "peers": _State.peers.addresses,
        "rank": get_parallel().tp_group.rank_in_group,
        "epoch": _State.epoch,
    }


def scratch_bytes() -> int:
    return _State.zero_bytes


def scratch() -> torch.Tensor:
    assert _State.scratch is not None
    return _State.scratch


def epoch() -> torch.Tensor:
    assert _State.epoch is not None
    return _State.epoch


def queue() -> torch.Tensor:
    assert _State.queue is not None
    return _State.queue
