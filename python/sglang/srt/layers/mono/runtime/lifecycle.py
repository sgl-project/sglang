# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/runtime/lifecycle.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A mono runner's peer buffer, owned for its lifetime. The buffer's handshake is
collective: a runner reaches it only after every rank bound and agreed
(``consensus.bind_agreed``), so a rank that refuses turns mono off on every rank
instead of leaving the others waiting in a collective it never reaches."""

import weakref

from sglang.srt.layers.mono.runtime.peer_memory import PeerBuffer


def owned_peer_buffer(owner, nbytes, group, rank, npes, device, *, debug=False):
    """``(PeerBuffer, finalizer)``: the buffer is freed when ``owner`` is dropped
    (a model reload), not at interpreter exit, where the HIP runtime may already
    be gone; calling the finalizer frees it now. Collective (``PeerBuffer``)."""
    peers = PeerBuffer(nbytes, group, rank, npes, device, debug=debug)
    finalizer = weakref.finalize(owner, peers.close)
    finalizer.atexit = False
    return peers, finalizer
