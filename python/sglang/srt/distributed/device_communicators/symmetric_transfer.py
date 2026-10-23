# SPDX-License-Identifier: Apache-2.0
"""One-sided, zero-SM allgather and broadcast over an NCCL symmetric window.

A rank pushes its data into the output on its peers with copy-engine copies,
through one of two kinds of target addresses into the window (identical
offsets on every rank):

- multicast: the window's NVLS multicast VA; one copy reaches every rank's
  backing memory, so a rank's NVLink egress is 1x what it sends, not (N-1)x;
- unicast: each peer's window, mapped through NCCL's LSA team; one copy per
  peer (rotating peers, one copy per gated chunk), no NVLS needed.

- ``SymmetricAllGather``: every rank pushes its own block into its slot.
- ``SymmetricBroadcast``: the root pushes its buffer; the others only sync.

No SM runs and no NCCL collective is involved: the data copies are chunk ops
(``gated_transfer.CopyChunk``), so a gated ``TransferChain`` can pause them,
and the only two-sided step is the barrier pair around them, never gated:

- entry, on the chain: every rank's output is free to be overwritten (each
  rank's previous readers are ordered before its signal by the chain's
  ``after`` stream);
- exit: the chain signals it once this rank's copies have landed, and the
  consumer waits for every rank's signal on its own stream (``wait``), so a
  reader ordered after ``wait`` on any rank sees the full output.

A barrier (``WindowBarrier``) is a 4-byte copy-engine copy of the constant 1
into this rank's lane through every target, then one batch of stream memops on
the LOCAL lanes: wait until every lane is 1, then clear every lane. Stream
memops are rejected on multicast VAs, so only a copy can signal. The entry and
exit barriers alternate between two lane arrays: reusing one array back to
back would let a fast rank's next signal land before a slow rank's clear and
be lost.

A paused rank holds every peer's readers until it resumes (its exit signal
follows its last chunk). The exit wait sits on the consumer's stream, at its
read point, not on the chain: no chain stream ever parks on a peer, so a
peer's lazy module load, device sync or shared hardware queue cannot hold
back the collective that the paused rank's resume waits on (the peer issued
that collective before its read). The entry wait only waits on work every
rank submitted earlier, so a shared hardware queue costs overlap but cannot
deadlock it.

Even unpaused, a long unbroken multicast copy starves concurrent NVLink
collectives (measured on 8x B200: a 32 MiB fused all-reduce 110 -> 960 us),
which chunking alone reduces (-> ~250-350 us) and pausing removes (-> 114 us).

Layout contract: the output is ``window[off : off + size]`` at an offset
identical on every rank (an allgather's rank r slot at ``off + r * block``).
The control region (``control_bytes(world_size)``) is carved out of the same
window at an offset identical on every rank and must be zeroed on every rank
before any rank's first call (``init_control``, a device sync, then a host
barrier). Transfers sharing one barrier must be issued in the same order on
every rank and run on one chain, and each must be consumed (``wait``) on a
stream that the next transfer's ``after`` stream is ordered behind; a
transfer issued while the previous one is unconsumed consumes it on its own
``after`` stream first.
"""

from __future__ import annotations

import ctypes
import functools
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch

from sglang.srt.distributed.device_communicators.gated_transfer import (
    CU_STREAM_WAIT_VALUE_EQ,
    CU_STREAM_WRITE_VALUE_DEFAULT,
    ChunkOp,
    CopyChunk,
    CudaDriver,
    TransferChain,
    chunk_spans,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        SymmetricWindowBuffer,
    )

_U32 = 4
_LINE = 64
# cuStreamBatchMemOp requires count < 256.
_MAX_BATCH_MEM_OPS = 255
# cuda.h: CUstreamBatchMemOpType
_CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1
_CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2


class CUstreamMemOpValueParams(ctypes.Structure):
    """The waitValue / writeValue member of the 48-byte
    ``CUstreamBatchMemOpParams`` union; the u32 value is the low half of a
    u32/u64 union."""

    _fields_ = [
        ("operation", ctypes.c_uint32),
        ("address", ctypes.c_uint64),
        ("value", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("alias", ctypes.c_uint64),
        ("_pad", ctypes.c_uint64),
    ]


assert ctypes.sizeof(CUstreamMemOpValueParams) == 48


@functools.lru_cache(maxsize=None)
def _batch_memop():
    # The _v2 symbol: the bare cuStreamBatchMemOp is the legacy one, which
    # cannot be graph-captured.
    return CudaDriver.get().bind(
        "cuStreamBatchMemOp_v2",
        [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(CUstreamMemOpValueParams),
            ctypes.c_uint,
        ],
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def control_bytes(world_size: int) -> int:
    """Two lane arrays and the constant 1 that signals copy from, each on its
    own 64-byte line."""
    return 2 * _align_up(world_size * _U32, _LINE) + _LINE


class WindowBarrier:
    """Entry / exit barrier of the symmetric transfers over one window.

    ``targets``: the base addresses through which a copy reaches the window
    at the same offset on every rank: ``[multicast_base]``, or every rank's
    window base (this rank's local base included).
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        local_base: int,
        targets: Sequence[int],
        control_offset: int,
    ):
        assert 0 <= rank < world_size
        assert targets and all(targets), "a window target is NULL (NVLS off?)"
        assert len(targets) in (1, world_size)
        assert 2 * world_size <= _MAX_BATCH_MEM_OPS
        self.rank = rank
        self.world_size = world_size
        self.local_base = local_base
        self.targets = tuple(targets)
        # Unicast: one target per rank, indexed by rank. Multicast: one.
        self.unicast = len(self.targets) > 1
        self._drv = CudaDriver.get()

        lane_stride = _align_up(world_size * _U32, _LINE)
        sync_offsets = (control_offset, control_offset + lane_stride)
        self._control_offset = control_offset
        self._one_offset = control_offset + 2 * lane_stride
        one_addr = local_base + self._one_offset
        # Per channel: this rank's lane through every target.
        self._signals = tuple(
            CopyChunk(
                tuple((t + off + rank * _U32, one_addr, _U32) for t in self.targets)
            )
            for off in sync_offsets
        )
        # Per channel: wait for every lane (this rank's included) to read 1,
        # then clear them all; batch ops execute in array order.
        self._ops = tuple(
            self._wait_then_clear_ops(local_base + off) for off in sync_offsets
        )

    def _wait_then_clear_ops(self, lanes_addr: int):
        n = self.world_size
        ops = (CUstreamMemOpValueParams * (2 * n))()
        for peer in range(n):
            wait, clear = ops[peer], ops[n + peer]
            wait.operation = _CU_STREAM_MEM_OP_WAIT_VALUE_32
            wait.address = lanes_addr + peer * _U32
            wait.value = 1
            wait.flags = CU_STREAM_WAIT_VALUE_EQ
            clear.operation = _CU_STREAM_MEM_OP_WRITE_VALUE_32
            clear.address = lanes_addr + peer * _U32
            clear.value = 0
            clear.flags = CU_STREAM_WRITE_VALUE_DEFAULT
        return ops

    def init_control(self, window: torch.Tensor) -> None:
        """Zero this rank's control region and set its constant 1, given a
        uint8 view of the whole window."""
        start = self._control_offset
        window[start : start + control_bytes(self.world_size)].zero_()
        window[self._one_offset : self._one_offset + _U32].view(torch.int32).fill_(1)

    def signal(self, channel: int, stream: torch.cuda.Stream) -> None:
        """Set this rank's lane of ``channel`` on every rank."""
        self._signals[channel](stream)

    def wait_lanes(self, channel: int, stream: torch.cuda.Stream) -> None:
        """Wait until every rank's lane of ``channel`` is set, then clear
        them."""
        ops = self._ops[channel]
        self._drv.check(
            _batch_memop()(stream.cuda_stream, len(ops), ops, 0),
            "barrier wait+clear (cuStreamBatchMemOp)",
        )

    def entry(self, stream: torch.cuda.Stream) -> None:
        self.signal(0, stream)
        self.wait_lanes(0, stream)

    def exit_signal(self, stream: torch.cuda.Stream) -> None:
        self.signal(1, stream)

    def exit_wait(self, stream: torch.cuda.Stream) -> None:
        self.wait_lanes(1, stream)


class _SymmetricTransfer:
    def __init__(
        self,
        barrier: WindowBarrier,
        *,
        chunk_bytes: Optional[int],
        chain: TransferChain,
    ):
        assert chunk_bytes is None or chunk_bytes > 0
        assert chain.gate is None or chunk_bytes is not None, "gates need chunks"
        self.barrier = barrier
        self.chunk_bytes = chunk_bytes
        self.chain = chain
        # The last transfer, issued and its exit not yet waited for on this
        # rank (its done event identifies it to wait).
        self._unconsumed = False
        self._unconsumed_done: Optional[torch.cuda.Event] = None

    @property
    def local_base(self) -> int:
        return self.barrier.local_base

    def _push_chunks(self, src: int, dst_offset: int, nbytes: int) -> List[CopyChunk]:
        """Chunks of one push of ``nbytes`` at ``src`` to ``dst_offset`` on
        every rank."""
        barrier = self.barrier
        spans = chunk_spans(nbytes, self.chunk_bytes)
        if not barrier.unicast:
            (target,) = barrier.targets
            return [
                CopyChunk(((target + dst_offset + off, src + off, n),))
                for off, n in spans
            ]
        # One chunk per (span, peer), so a pause lands within one chunk_bytes
        # copy. Peers rotate from rank + 1: at any step every rank pushes to a
        # different peer, not all into one (NCCL's CE allgather does the
        # same). This rank comes last, and not at all when ``src`` already is
        # the destination.
        rank, world_size = barrier.rank, barrier.world_size
        peers = [(rank + i) % world_size for i in range(1, world_size + 1)]
        if src == barrier.local_base + dst_offset:
            peers.pop()
        return [
            CopyChunk(((barrier.targets[peer] + dst_offset + off, src + off, n),))
            for off, n in spans
            for peer in peers
        ]

    def _run(self, ops, after, setup, done) -> None:
        barrier = self.barrier
        if self._unconsumed:
            # The lanes must be clear before any rank can signal this
            # transfer's exit.
            barrier.exit_wait(after)
        self.chain.run(
            ops,
            after,
            setup=(*setup, barrier.entry),
            finish=(barrier.exit_signal,),
            done=done,
        )
        self._unconsumed = True
        self._unconsumed_done = done

    def wait(
        self, stream: torch.cuda.Stream, done: Optional[torch.cuda.Event] = None
    ) -> None:
        """Make ``stream`` wait until a transfer has landed on every rank:
        the one that recorded ``done``, or the last one. The exit barrier is
        waited for once per transfer; a repeated wait, or one for a transfer
        older than the last, only waits on its event, so it must come on a
        stream ordered after the first wait (the same stream, typically)."""
        self.chain.wait(stream, done)
        if self._unconsumed and (done is None or done is self._unconsumed_done):
            self.barrier.exit_wait(stream)
            self._unconsumed = False
            self._unconsumed_done = None


class SymmetricAllGather(_SymmetricTransfer):
    """Chunked copy-engine allgather into a symmetric window."""

    def all_gather(
        self,
        blocks: Sequence[Tuple[int, int, int]],
        after: torch.cuda.Stream,
        *,
        setup: Sequence[ChunkOp] = (),
        done: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Enqueue one allgather on the chain, after ``after``'s queued work
        and ``setup``.

        ``blocks``: ``(src_ptr, dst_offset, nbytes)`` per contiguous piece of
        this rank's contribution, ``dst_offset`` being where the piece sits in
        this rank's slot of the window. ``src_ptr`` may be that slot's own
        local address (in place).
        """
        ops: List[ChunkOp] = []
        for src, dst_offset, nbytes in blocks:
            ops += self._push_chunks(src, dst_offset, nbytes)
        self._run(ops, after, setup, done)


class SymmetricBroadcast(_SymmetricTransfer):
    """Chunked copy-engine broadcast into a symmetric window: the root pushes
    its buffer, the others only sync."""

    def broadcast(
        self,
        root: int,
        src_ptr: int,
        dst_offset: int,
        nbytes: int,
        after: torch.cuda.Stream,
        *,
        setup: Sequence[ChunkOp] = (),
        done: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Enqueue one broadcast of ``nbytes`` at the root's ``src_ptr`` into
        ``window[dst_offset : dst_offset + nbytes]`` on every rank.
        ``src_ptr`` is ignored on the other ranks; on the root it may be the
        destination's own local address."""
        ops: List[ChunkOp] = []
        if self.barrier.rank == root:
            ops += self._push_chunks(src_ptr, dst_offset, nbytes)
        self._run(ops, after, setup, done)


# ---- targets in an NCCL symmetric window --------------------------------------


def window_targets(
    comm, buffer: SymmetricWindowBuffer, mode: str
) -> Optional[Tuple[int, ...]]:
    """The barrier / push targets of ``buffer``'s window for ``mode``
    (``"multicast"`` or ``"unicast"``), or None if this rank cannot reach
    every rank of ``comm`` that way. Collective over ``comm``; call it before
    anything is written to the window. Callers must still agree across ranks
    (e.g. a MIN all-reduce)."""
    nccl = comm.nccl
    usable = buffer.window is not None and nccl.has_function("ncclGetPeerDevicePointer")
    # NULL for a rank outside this rank's LSA team, which the multicast team
    # spans too (NCCL_LSA_TEAM_SIZE, multi-node, a split NVLink domain).
    bases = [
        (nccl.ncclGetPeerDevicePointer(buffer.window, 0, peer) or 0) if usable else 0
        for peer in range(comm.world_size)
    ]
    if mode == "multicast":
        if all(bases) and buffer.multicast_ptr:
            return (buffer.multicast_ptr,)
        return None
    assert mode == "unicast", mode
    # Every rank takes part in the probe, reachable or not.
    if not _lsa_bases_map_ranks(comm, buffer.tensor, bases):
        return None
    # NCCL maps this rank's own window at another VA of the LSA range; the
    # local VA lets an in-place push skip this rank.
    bases[comm.rank] = buffer.tensor.data_ptr()
    return tuple(bases)


def _lsa_bases_map_ranks(comm, window: torch.Tensor, bases: List[int]) -> bool:
    """Whether ``bases[r]`` reaches rank r's window for every rank (the LSA
    team is the whole communicator, in rank order). Every rank stamps the
    first word of its window, then reads every rank's stamp through the
    bases; the window is zeroed again. Collective over ``comm``."""
    drv = CudaDriver.get()
    stamp = window[:_U32].view(torch.int32)
    stamp.fill_(comm.rank + 1)
    torch.cuda.synchronize(window.device)
    torch.distributed.barrier(group=comm.group)
    seen = torch.zeros(comm.world_size, dtype=torch.int32).pin_memory()
    if all(bases):
        stream = torch.cuda.current_stream(window.device)
        for peer, base in enumerate(bases):
            drv.check(
                drv.memcpy(
                    seen.data_ptr() + peer * _U32, base, _U32, stream.cuda_stream
                ),
                "LSA probe (cuMemcpyAsync)",
            )
        stream.synchronize()
    torch.distributed.barrier(group=comm.group)
    stamp.zero_()
    torch.cuda.synchronize(window.device)
    torch.distributed.barrier(group=comm.group)
    return seen.tolist() == list(range(1, comm.world_size + 1))
