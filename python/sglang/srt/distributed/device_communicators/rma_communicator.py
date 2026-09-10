# SPDX-License-Identifier: Apache-2.0
"""NCCL one-sided RMA communicator (symmetric memory, NCCL 2.29-2.30+).

Wraps ncclPutSignal/ncclWaitSignal on top of a PyNcclCommunicator. __init__
runs an all-rank warm-up that registers a symmetric window, exchanges
handles, and verifies a put/wait round-trip; on hardware where NCCL returns
a NULL window handle (no NVLink, vGPU) it skips and leaves rma_available
False so callers fall back to collective NCCL.
"""

import ctypes
import logging
from typing import List

import torch

from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
    NCCL_WIN_COLL_SYMMETRIC,
)

logger = logging.getLogger(__name__)

# Warm-up window size, in int32 elements. Only needs to carry the ring
# handshake marker.
_WARMUP_ELEMS = 4
_NCCL_INT32 = 2  # ncclDataTypeEnum.ncclInt32


class NcclRmaCommunicator:
    """One-sided RMA communicator over a PyNcclCommunicator."""

    def __init__(self, pynccl_comm: PyNcclCommunicator, cpu_group=None):
        self.pynccl = pynccl_comm
        self.cpu_group = cpu_group
        self.rank = pynccl_comm.rank
        self.world_size = pynccl_comm.world_size

        # Capability gate: library has RMA symbols at a usable version and the
        # group has more than one rank. Necessary but not sufficient -- the HW
        # may still return NULL handles, see rma_available.
        self.enabled = bool(pynccl_comm.supports_rma()) and self.world_size > 1
        # True only after warm-up succeeds. Callers must check this before
        # issuing RMA ops and fall back to collective NCCL otherwise.
        self.rma_available = False

        if not self.enabled:
            return

        # Warm-up must never raise out of __init__: any failure is a fallback.
        try:
            with pynccl_comm.change_state(enable=True):
                self._warmup()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "NCCL RMA warm-up failed for rank %d, falling back to "
                "collective NCCL: %s",
                self.rank,
                e,
            )
            self.rma_available = False

        if self.rma_available and self.rank == 0:
            logger.info(
                "NCCL one-sided RMA warm-up succeeded (world_size=%d).",
                self.world_size,
            )

    def _warmup(self) -> None:
        """Register a temp symmetric window, exchange handles, run a ring
        put/wait handshake, verify, deregister. Skips (rma_available stays
        False) if any rank observes a NULL window handle."""
        nbytes = _WARMUP_ELEMS * 4  # int32
        buf = self.pynccl.nccl_mem_alloc(nbytes)
        win = None
        try:
            win = self._register_warmup_window(buf, nbytes)
            my_handle = win.value if win and win.value else 0

            # Each rank writes its own handle into its slot; all_reduce(SUM)
            # leaves each slot holding the corresponding rank's handle (0 for
            # NULL handles, since every other slot is zero).
            handles = self._allgather_handles(my_handle)

            if any(h == 0 for h in handles):
                if self.rank == 0:
                    logger.info(
                        "NCCL RMA unavailable: a rank got a NULL window handle "
                        "(no NVLink / vGPU?); falling back to collective NCCL."
                    )
                return

            self._ring_handshake(buf, handles)
            self.rma_available = True
        finally:
            if win:
                self.pynccl.deregister_comm_window(win)
            self.pynccl.nccl_mem_free(buf)

    def _register_warmup_window(self, buf: int, nbytes: int):
        return self.pynccl.register_comm_window_raw(
            buf, nbytes, win_flags=NCCL_WIN_COLL_SYMMETRIC
        )

    def _allgather_handles(self, my_handle: int) -> List[int]:
        t = torch.zeros(self.world_size, dtype=torch.int64, device=self.pynccl.device)
        t[self.rank] = my_handle
        self.pynccl.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return [int(v) for v in t.tolist()]

    def _ring_handshake(self, buf: int, handles: List[int]) -> None:
        """Rank i puts marker (i+1) to rank (i+1)%n's window, waits for a
        marker from rank (i-1)%n, and verifies the received value."""
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1) % self.world_size
        next_win = handles[next_rank]

        # ncclPutSignal copies from a local buffer; write the marker into our
        # own warm-up buffer and use it as the source.
        cudart = self._cudart()
        host = (ctypes.c_int32 * 1)(self.rank + 1)
        stream = self.pynccl._resolve_stream()
        sptr = ctypes.c_void_p(stream.cuda_stream)
        cudart.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        cudart.cudaMemcpyAsync.restype = ctypes.c_int
        cudart.cudaMemcpyAsync(buf, ctypes.cast(host, ctypes.c_void_p), 4, 1, sptr)
        stream.synchronize()

        self.pynccl.put_signal(
            buf,
            1,
            _NCCL_INT32,
            peer=next_rank,
            peer_win=next_win,
            peer_win_offset=0,
            stream=stream,
        )

        descs_ptr, n = PyNcclCommunicator.make_wait_descs([(prev_rank, 1)])
        self.pynccl.wait_signal(descs_ptr, n, stream=stream)
        stream.synchronize()

        out = (ctypes.c_int32 * 1)()
        cudart.cudaMemcpyAsync(ctypes.cast(out, ctypes.c_void_p), buf, 4, 2, sptr)
        stream.synchronize()
        got = out[0]
        expected = prev_rank + 1
        if got != expected:
            raise RuntimeError(
                f"RMA warm-up verify failed on rank {self.rank}: got {got}, "
                f"expected {expected} (prev_rank={prev_rank})"
            )

    _cudart_lib = None

    @classmethod
    def _cudart(cls):
        if cls._cudart_lib is None:
            cls._cudart_lib = ctypes.CDLL("libcudart.so")
        return cls._cudart_lib
